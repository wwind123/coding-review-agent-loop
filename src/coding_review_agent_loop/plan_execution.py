"""Approved-plan execution and child dispatch, independent of the planning driver."""
from __future__ import annotations
from .pr_head_sync import get_observed_pr_review_context
from types import SimpleNamespace
import json
from collections.abc import Callable
from dataclasses import replace as dataclasses_replace
from .config import AgentLoopConfig, reviewers
from .plan_issue_scope import scope_plan_comments
from .plan_verification import plan_ledger_unreconstructible_items
from .decomposition import (
    CreatedPhaseIssue,
    approved_plan_hash,
    find_existing_one_shot_impl_handoff,
    find_latest_one_shot_impl_handoff,
    find_existing_phase_implementation_handoff,
    post_phase_implementation_handoff_comment,
    normalize_execution_recommendation,
    InheritedMatrixBinding,
    risk_matrix_row_ids_for_owner,
    EXECUTION_TOPOLOGY_SOURCE,
    ChildDispositionOverride,
    PhaseImplementationHandoffMetadata,
    collect_child_disposition_overrides,
)
from .protocol import EXECUTION_DISPOSITION_DIRECT
from .child_topology import (
    NeedsHumanDecision,
    NestedTopologyDecision,
)
from .errors import AgentLoopError
from .github import (
    IssueContext,
    get_issue_context,
    get_pr_state,
    validate_pr_body_does_not_close_issue,
)
from .issue_pr_handoff import (
    authenticate_canonical_issue_pr,
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .phase_progress import (
    STATUS_HUMAN_PENDING,
    StagedTopologyOutcome,
    record_staged_completion,
    resolve_staged_phase_progress,
    select_current_phase,
)
from .split_materialization import (
    find_existing_split_materialization,
    find_existing_split_stage_handoff,
    post_split_stage_handoff_comment,
    resolve_selected_stage_child,
)
from .logging import log
from .prompt_support import format_agent_list
from .protocol import ExecutionStrategyRecommendation
from .protocol_risk import validate_risk_test_matrix_revision
from .runner import Runner
from .usage import RunUsageContext
from .followups import (
    _plan_followup_source_from_unresolved_item,
    _publish_plan_approved_followups,
    FollowupSourceContext,
)
from .round_metadata import ApprovedPlanContext, _plan_subject, scope_approved_plan_matrix
from .round_state import _extract_round_metadata_records, make_approved_plan_context
from .plan_growth import plan_growth_approval_verdict
from .response_validation import _describe_requirement_set_change
from .execution_policy import (
    _extract_current_deferred_stages,
    _extract_current_expected_closing_issue_ids,
    _extract_current_child_stages,
    _plan_first_line,
    _current_execution_recommendation,
    _resolve_execution_policy,
    ChildExecutionRoute,
    resolve_child_execution_route,
    _fresh_phase_marker_payload,
    _child_resume_hint,
    child_relaunch_line,
    _post_child_planning_handoff,
    _print_staged_phase_progress,
    _print_staged_terminal_report,
    _print_dry_run_execution_preview,
    _print_execution_resolution_summary,
    _persist_execution_decision_if_needed,
    _retirable_execution_decision_hashes,
    _preflight_fresh_staged_topology,
    _preflight_fresh_one_shot_recovery,
    _handle_plan_first_split_scope,
)
from .child_plan_binding import _inherited_matrix_binding, _rebind_superseded_child_plan
from .pr_loop import run_pr_loop
from .issue_implementation import (
    _implement_approved_issue,
    _decompose_approved_plan,
)


def _run_child_planning_cycle(
    runner: Runner,
    *,
    run_plan_first_loop: Callable[..., int],
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    parent_issue: int,
    child_issue_number: int,
    inherited_matrix_binding: InheritedMatrixBinding | None = None,
) -> int:
    """Run the child's own plan/review cycle (policy ``auto``) after its handoff."""
    # The child inherits the operator's run-wide plan-review policy and primary
    # plan reviewer (#929), but no parent scheduling state crosses the
    # boundary: the child plan is a fresh artifact, so the force-full latch is
    # reset and the execution mode is ``auto`` (#905, from #841).
    child_config = dataclasses_replace(
        config,
        plan_execution_mode="auto",
        plan_review_force_full=False,
        plan_reset_stall_streak=False,
        plan_resume_keep_on_issue_edit=False,
    )
    if config.plan_resume_keep_on_issue_edit:
        log(
            config,
            f"Issue #{parent_issue}: child #{child_issue_number} plan review does not "
            "inherit --plan-resume-keep-on-issue-edit",
        )
    if config.plan_reset_stall_streak:
        log(
            config,
            f"Issue #{parent_issue}: child #{child_issue_number} plan review does not "
            "inherit --plan-reset-stall-streak",
        )
    if config.plan_review_force_full:
        log(
            config,
            f"Issue #{parent_issue}: child #{child_issue_number} plan review does not "
            "inherit --plan-review-force-full; the child plan starts under "
            f"--plan-review-policy {config.plan_review_policy} without the "
            "parent's full-board override",
        )
    child_issue_context = get_issue_context(
        runner, config=child_config, issue_number=child_issue_number
    )
    log(
        config,
        f"Issue #{parent_issue}: child #{child_issue_number} requires its own reviewed plan; "
        "starting the child plan-first cycle with policy auto before any implementation coder",
    )
    return run_plan_first_loop(
        runner,
        issue_number=child_issue_number,
        config=child_config,
        memory=memory,
        issue_context=child_issue_context,
        requested_policy="auto",
        usage_context=usage_context,
        inherited_matrix_binding=inherited_matrix_binding,
    )


def _dispatch_decomposition_child(
    runner: Runner,
    *,
    run_plan_first_loop: Callable[..., int],
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    parent_issue: int,
    approved_plan: str,
    plan_hash: str,
    plan_subject: str,
    recommendation: ExecutionStrategyRecommendation | None,
    approved_plan_context: ApprovedPlanContext,
    created: CreatedPhaseIssue,
    phase_index: int,
    route: ChildExecutionRoute,
    child_issue_context: IssueContext,
    parent_issue_context: IssueContext,
    coder_session_id: str | None,
    existing_handoff: PhaseImplementationHandoffMetadata | None,
) -> int:
    """Dispatch one materialized child along its resolved route.

    Shared by the parent implement-by-phase path and the direct child-issue
    entry.  The handoff is persisted before any coder or planning work; when
    a matching handoff already exists it is reused, never re-posted.
    """
    if created.issue_number is None:
        raise AgentLoopError(
            "Cannot implement decomposed phase because its child issue number "
            "was not available from GitHub CLI output."
        )
    if route.is_human:
        raise AgentLoopError(
            f"Issue #{created.issue_number} is a {created.phase.automation} stage; no agent "
            "implementation or planning is dispatched for human-owned stages."
        )
    stage_id = (
        getattr(created.phase, "stage_id", None)
        or str(getattr(created.phase, "position", None) or phase_index)
    )
    inherited_matrix_row_ids = risk_matrix_row_ids_for_owner(
        approved_plan_context.risk_test_matrix_payload
        if approved_plan_context.matrix_available else None,
        stage_id,
    )
    if route.is_planning:
        if recommendation is None:
            raise AgentLoopError(
                "Child planning requires a fresh approved-plan-v1 topology; legacy topologies "
                "cannot route a child to planning."
            )
        if existing_handoff is None:
            # Persistence before any planning agent runs.
            _post_child_planning_handoff(
                runner,
                config=config,
                parent_issue=parent_issue,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                phase_index=phase_index,
                created=created,
                recommendation=recommendation,
                inherited_matrix_row_ids=inherited_matrix_row_ids,
                override_digest=route.override_digest,
            )
        return _run_child_planning_cycle(
            runner,
            run_plan_first_loop=run_plan_first_loop,
            config=config,
            memory=memory,
            usage_context=usage_context,
            parent_issue=parent_issue,
            child_issue_number=created.issue_number,
            inherited_matrix_binding=_inherited_matrix_binding(
                parent_issue=parent_issue,
                stage_id=stage_id,
                parent_plan_context=approved_plan_context,
            ),
        )
    if existing_handoff is None:
        # Persist the parent-owned assignment before child execution so
        # PR validation and crash recovery see the same phase identity.
        post_phase_implementation_handoff_comment(
            runner,
            config=config,
            parent_issue=parent_issue,
            mode="implement-by-phase",
            plan_hash=plan_hash,
            phase_index=phase_index,
            created=created,
            strategy=recommendation.strategy if recommendation is not None else None,
            topology_source=EXECUTION_TOPOLOGY_SOURCE if recommendation is not None else None,
            execution_strategy_contract_version=1 if recommendation is not None else None,
            recommendation_digest=(
                str(recommendation.identity()["recommendation_sha256"])
                if recommendation is not None else None
            ),
            plan_subject=plan_subject,
            inherited_matrix_row_ids=inherited_matrix_row_ids,
            execution_disposition=(
                EXECUTION_DISPOSITION_DIRECT if recommendation is not None else None
            ),
            override_digest=route.override_digest if recommendation is not None else None,
        )
    child_plan_context = make_approved_plan_context(
        approved_plan,
        source_locator=f"issue #{parent_issue} topology checkpoint phase {phase_index}",
        expected_hash=plan_hash,
        expected_subject=plan_subject,
    )
    child_plan_context = scope_approved_plan_matrix(
        child_plan_context,
        execution_owner=stage_id,
        valid_stage_ids=tuple(
            phase.stage_id or str(phase.position)
            for phase in recommendation.child_stages
        ) if recommendation is not None else (),
    )
    return _implement_approved_issue(
        runner,
        issue_number=created.issue_number,
        approved_plan=created.phase.parent_context or approved_plan,
        config=config,
        memory=memory,
        issue_context=child_issue_context,
        approved_plan_context=child_plan_context,
        parent_issue_context=parent_issue_context,
        coder_session_id=coder_session_id,
        usage_context=usage_context,
        execution_recommendation=recommendation,
    )


def _dispatch_current_decomposition_phase(
    runner: Runner,
    *,
    run_plan_first_loop: Callable[..., int],
    config: AgentLoopConfig,
    memory,
    usage_context: RunUsageContext,
    issue_number: int,
    current_plan: str,
    plan_subject: str,
    outcome: StagedTopologyOutcome,
    recommendation: ExecutionStrategyRecommendation | None,
    approved_plan_context: ApprovedPlanContext,
    issue_context: IssueContext,
    mode: str,
    coder_session_id: str | None,
) -> int:
    """Route and dispatch the *current* phase of an implement-by-phase topology.

    The current phase is the first phase that is not complete, authenticated
    from live child state (#918).  A completed stage-1 therefore advances to
    stage-2 with its real ``phase_index``, and a fully delivered topology
    reports a terminal state instead of pointing the operator at a finished
    child.
    """
    created = outcome.created
    if not created:
        raise AgentLoopError("Plan decomposition produced no phases.")
    plan_hash = outcome.plan_hash
    fresh = recommendation is not None
    topology_stage_ids = outcome.stage_ids

    if config.dry_run:
        # Dry run resolves no child state and issues no GitHub reads: it
        # reports the first phase's declared disposition, exactly as today.
        first_phase = created[0]
        first_stage_id = topology_stage_ids[0]
        if first_phase.phase.automation != "agent-pr":
            print(
                f"Issue #{issue_number} approved plan decomposed; first phase requires human work "
                f"({first_phase.phase.automation}), so implementation is stopping."
            )
            return 0
        declared = getattr(first_phase.phase, "execution_disposition", None) or "legacy-ambiguous"
        print(
            f"Issue #{issue_number} dry-run decomposed the approved plan; "
            "phase implementation is not started. "
            f"First phase `{first_stage_id}` declared disposition: {declared}."
        )
        return 0

    parent_issue_context = get_issue_context(
        runner, config=config, issue_number=issue_number
    )
    progress = resolve_staged_phase_progress(
        runner,
        config=config,
        parent_issue=issue_number,
        parent_comments=parent_issue_context.comments,
        outcome=outcome,
    )
    _print_staged_phase_progress(issue_number, progress)
    selected = select_current_phase(progress)
    if selected is None:
        _print_staged_terminal_report(
            issue_number=issue_number, progress=progress, outcome=outcome
        )
        try:
            recorded = record_staged_completion(
                runner,
                config=config,
                parent_issue=issue_number,
                progress=progress,
                outcome=outcome,
            )
        except AgentLoopError as exc:
            # The delivery report above stands; only the write-back is
            # withheld, and nothing incomplete was published.
            log(config, f"Issue #{issue_number} completion record not written: {exc}")
            print(
                f"No completion record was written to issue #{issue_number} ({exc}). "
                "Rerun the parent once that is resolved to record it."
            )
            return 0
        if recorded:
            print(f"Recorded the staged completion on issue #{issue_number}.")
        return 0

    phase_index = selected.phase_index
    stage_id = selected.stage_id
    selected_created = selected.created
    child_issue_number = selected.child_issue_number
    if child_issue_number is None:
        raise AgentLoopError(
            "Cannot implement decomposed phase because its child issue number "
            "was not available from GitHub CLI output."
        )

    if selected.status == STATUS_HUMAN_PENDING:
        if fresh:
            # Stage-scoped validation: an override naming the human stage is a
            # human-decision error; a valid override for another stage is
            # neither an error nor an input here.
            child_comments = get_issue_context(
                runner, config=config, issue_number=child_issue_number
            ).comments
            overrides = collect_child_disposition_overrides(
                parent_comments=issue_context.comments,
                child_comments=child_comments,
                parent_issue=issue_number,
                plan_hash=plan_hash,
                topology_stage_ids=topology_stage_ids,
                routed_stage_id=stage_id,
                child_stage_id=stage_id,
                child_issue_number=child_issue_number,
            )
            resolve_child_execution_route(
                selected_created.phase,
                topology_source=EXECUTION_TOPOLOGY_SOURCE,
                recorded_handoff=None,
                overrides=overrides,
            )
        print(
            f"Issue #{issue_number} approved plan decomposed; phase {phase_index} "
            f"(`{stage_id}`) requires human work ({selected.automation}) on child issue "
            f"#{child_issue_number}, so implementation is stopping."
        )
        return 0

    handoff = find_existing_phase_implementation_handoff(
        parent_issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=phase_index,
        child_issue_number=child_issue_number,
    )
    if not fresh:
        if handoff is not None:
            # Only an in-progress phase reaches here: a completed child is
            # never advertised as a resume target (#918).
            print(
                f"Issue #{issue_number} approved plan already handed off to child issue "
                f"#{handoff.child_issue_number}; resume directly with "
                f"`agent-loop issue {handoff.child_issue_number}`."
            )
            return 0
        route = ChildExecutionRoute(EXECUTION_DISPOSITION_DIRECT, None, "legacy-topology")
        overrides: tuple[ChildDispositionOverride, ...] = ()
    else:
        child_issue_context = get_issue_context(
            runner, config=config, issue_number=child_issue_number
        )
        overrides = collect_child_disposition_overrides(
            parent_comments=parent_issue_context.comments,
            child_comments=child_issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            topology_stage_ids=topology_stage_ids,
            routed_stage_id=stage_id,
            child_stage_id=stage_id,
            child_issue_number=child_issue_number,
        )
        route = resolve_child_execution_route(
            selected_created.phase,
            topology_source=EXECUTION_TOPOLOGY_SOURCE,
            recorded_handoff=handoff,
            overrides=overrides,
        )
        if handoff is not None:
            # Reconciliation ran inside the seam; the hint follows the
            # recorded route so it never guides into the conflicting form.
            print(
                f"Issue #{issue_number} approved plan already handed off to child issue "
                f"#{handoff.child_issue_number} with disposition `{route.disposition}`; "
                f"resume directly with `{_child_resume_hint(handoff.child_issue_number, route.disposition)}`."
            )
            return 0
    if not fresh:
        child_issue_context = get_issue_context(
            runner, config=config, issue_number=child_issue_number
        )
    return _dispatch_decomposition_child(
        runner,
        run_plan_first_loop=run_plan_first_loop,
        config=config,
        memory=memory,
        usage_context=usage_context,
        parent_issue=issue_number,
        approved_plan=current_plan,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        recommendation=recommendation,
        approved_plan_context=approved_plan_context,
        created=selected_created,
        phase_index=phase_index,
        route=route,
        child_issue_context=child_issue_context,
        parent_issue_context=parent_issue_context,
        coder_session_id=coder_session_id,
        existing_handoff=handoff,
    )


def _typed_deferred_work_titles(sidecar: object | None) -> tuple[str, ...]:
    """Titles of the typed ``deferred_work`` in the canonical plan, best effort (#1268)."""
    canonical = getattr(sidecar, "canonical_json", None)
    if not isinstance(canonical, dict):
        return ()
    typed = canonical.get("typed_stages")
    for container in (canonical, typed if isinstance(typed, dict) else {}):
        items = container.get("deferred_work")
        if isinstance(items, list):
            titles = tuple(
                str(item["title"]) for item in items
                if isinstance(item, dict) and isinstance(item.get("title"), str)
            )
            if titles:
                return titles
    return ()


# Invocation board fields re-pointed at an approved plan's approving board for
# the planning phase only, and restored before implementation (#1377).
_PLAN_BOARD_CONFIG_FIELDS = (
    "reviewer",
    "reviewer_seats",
    "active_reviewer_seat_id",
    "pr_seat_binding_override",
    "plan_review_policy",
    "primary_plan_reviewer",
    "plan_review_force_full",
    "plan_reset_stall_streak",
    "plan_resume_keep_on_issue_edit",
    "plan_primary_stall_rounds",
    "plan_step_back_rounds",
    "plan_step_back_escalation_rounds",
    "plan_sweep_rounds",
    "plan_sweep_reasoning_effort",
    "plan_provenance_baseline_max_chars",
)


def _with_planning_requirements(
    issue_context: IssueContext, parent_issue_context: IssueContext | None
) -> IssueContext:
    """``issue_context`` carrying its own and its authoritative parent's signed requirements."""
    if parent_issue_context is None or not parent_issue_context.human_requirements:
        return issue_context
    from .github import deduplicate_human_requirements

    return dataclasses_replace(
        issue_context,
        human_requirements=deduplicate_human_requirements(
            (*parent_issue_context.human_requirements, *issue_context.human_requirements)
        ),
    )


def _restore_invocation_board(
    config: AgentLoopConfig, invocation: AgentLoopConfig
) -> AgentLoopConfig:
    """``config`` with the invocation's own board for implementation (#1377).

    After an approved plan the stage 1 handoff rule alone governs the PR
    board, against the approving plan context.
    """
    return dataclasses_replace(
        config, **{name: getattr(invocation, name) for name in _PLAN_BOARD_CONFIG_FIELDS}
    )


def _staged_child_relaunch_lines(decomposition, outcome) -> tuple[str, ...]:
    """Per-child relaunch lines for the parent's stdout report (#1331).

    The route comes from the reviewed topology phase, which carries the
    execution disposition an adopted ``RecordedPhase`` does not; the child
    number comes from the materialized outcome at the same index.
    """
    phases = decomposition.phases
    return tuple(
        child_relaunch_line(
            index,
            phases[index - 1] if index <= len(phases) else created.phase,
            created.issue_number,
        )
        for index, created in enumerate(outcome.created, start=1)
    )


def _execute_approved_plan(
    *,
    coder_session_id,
    config,
    configured_reviewers,
    current_plan,
    current_plan_sidecar,
    implement_after_approval,
    invocation_config,
    issue_context,
    issue_number,
    memory,
    parent_issue_context,
    plan_amendment_note,
    plan_approved_shortcut,
    plan_growth_assessment,
    plan_own_requirement_ids,
    plan_scope_start_round,
    plan_supersession,
    prior_unresolved_items,
    requested_policy,
    round_new_unresolved_items,
    round_number,
    round_outcome_issue_context,
    runner,
    staged_parent_number,
    superseded_prepanel_item_ids,
    unresolved_items,
    usage_context,
    run_plan_first_loop: Callable[..., int],
) -> int:
    """Cross the approved-plan boundary with the reviewed snapshot and live inputs."""
    plan_approving_reviewers = tuple(configured_reviewers)
    if plan_approved_shortcut:
        # The approving board judged the plan; implementation uses the
        # invocation's board under the stage 1 handoff rule.
        config = _restore_invocation_board(config, invocation_config)
        configured_reviewers = reviewers(config)
    if plan_amendment_note:
        log(
            config,
            f"Planning issue #{issue_number}: plan approved on a reduced board. "
            f"{plan_amendment_note}",
        )
    # Re-read both sides at the approval-to-implementation boundary so
    # a human instruction posted during planning cannot be hidden by
    # the original snapshot. New signed IDs require a fresh planning
    # acknowledgement rather than being silently folded into the plan.
    # The latest round-outcome read is that boundary read: it was taken
    # after this round's reviews and after any acknowledgement repair,
    # so no agent turn has run since, and its board history was
    # re-validated then (#1377).
    refreshed_issue_context = round_outcome_issue_context
    # The shared completion predicate's ledger rule (#1377): an
    # obligation history published but neither carried nor provably
    # cleared fails closed instead of being approved over.
    # The complete history: the epoch and binding filters of
    # plan_history_records select approval evidence, never obligations.
    # A planning-scope boundary retires the archived plan's
    # obligations (#1390): only the current scope is reconstructed.
    lost_obligations = plan_ledger_unreconstructible_items(
        _extract_round_metadata_records(
            scope_plan_comments(refreshed_issue_context.comments, plan_scope_start_round),
            flow="plan",
        ),
        SimpleNamespace(
            prior_items=(*prior_unresolved_items, *unresolved_items),
            current_round_new_items=tuple(round_new_unresolved_items),
        ),
        ignored_ids=tuple(superseded_prepanel_item_ids),
    )
    if lost_obligations:
        raise AgentLoopError(
            f"Planning round {round_number}, plan {approved_plan_hash(current_plan)}: "
            "every reviewer approved, but plan "
            f"obligation(s) {', '.join(lost_obligations)} published earlier are neither "
            "carried in the current ledger nor cleared by recorded history, so the "
            "plan ledger cannot be reconstructed. Refusing to approve; inspect the "
            "plan round metadata and rerun."
        )
    previous_requirement_ids = set(plan_own_requirement_ids)
    refreshed_requirement_ids = {
        requirement.requirement_id
        for requirement in refreshed_issue_context.human_requirements
    }
    if refreshed_requirement_ids != previous_requirement_ids:
        # Any set inequality is a changed requirement set, not only an
        # addition.  A withdrawal is equally disqualifying: every plan
        # review and every carried exact-key approval was bound to the
        # earlier surfaced requirement digest, so a withdrawn ID leaves
        # the approvals acknowledging a requirement that no longer
        # exists (#905, from #841).
        raise AgentLoopError(
            f"Plan {approved_plan_hash(current_plan)}, round {round_number}: "
            f"Issue #{issue_number} signed human requirement(s) changed after plan "
            "approval: "
            f"{_describe_requirement_set_change(previous_requirement_ids, refreshed_requirement_ids)}. "
            "Re-run planning so the current signed requirements receive explicit "
            "acknowledgement."
        )
    issue_context = refreshed_issue_context
    if parent_issue_context is not None:
        previous_parent_requirement_ids = {
            requirement.requirement_id
            for requirement in parent_issue_context.human_requirements
        }
        refreshed_parent_context = get_issue_context(
            runner, config=config, issue_number=parent_issue_context.number
        )
        refreshed_parent_requirement_ids = {
            requirement.requirement_id
            for requirement in refreshed_parent_context.human_requirements
        }
        if refreshed_parent_requirement_ids != previous_parent_requirement_ids:
            raise AgentLoopError(
                f"Plan {approved_plan_hash(current_plan)}, round {round_number}: "
                f"Authoritative parent issue #{parent_issue_context.number} signed human "
                "requirement(s) changed after plan approval: "
                f"{_describe_requirement_set_change(previous_parent_requirement_ids, refreshed_parent_requirement_ids)}"
                ". Re-run planning so the current signed parent requirements receive "
                "explicit acknowledgement."
            )
        parent_issue_context = refreshed_parent_context
    issue_context = _with_planning_requirements(issue_context, parent_issue_context)
    approved_future_followup_sources = [
        _plan_followup_source_from_unresolved_item(item)
        for item in unresolved_items
        if item.status == "future"
    ]
    plan_hash = approved_plan_hash(current_plan)
    plan_subject = _plan_subject(current_plan)
    # The gate's own judgement of the candidate it just let through,
    # recorded in the handoff so resume re-validates against it (#1074).
    approval_growth_verdict = plan_growth_approval_verdict(
        config,
        current_plan_sidecar.canonical_json if current_plan_sidecar is not None else None,
        plan_growth_assessment,
    )
    approved_plan_context = make_approved_plan_context(
        current_plan,
        source_locator=f"issue #{issue_number} approved-plan round",
        expected_hash=plan_hash,
        expected_subject=plan_subject,
    )
    if approved_plan_context.matrix_available:
        # Cross the approval boundary through the same immutable
        # validator used by recovery. There is no prior approved
        # baseline to diff on a first approval, so comparing the
        # accepted payload with itself specifically asserts that the
        # boundary cannot mutate it while it is being bound.
        validate_risk_test_matrix_revision(
            approved_plan_context.risk_test_matrix_payload or {},
            approved_plan_context.risk_test_matrix_payload or {},
            approved_plan_context.risk_test_matrix_changes_payload,
            approved=True,
        )
    recommendation = _current_execution_recommendation(
        current_plan, issue_context.comments
    )
    resolved_execution = _resolve_execution_policy(
        config,
        requested_policy=requested_policy,
        implement_after_approval=implement_after_approval,
        recommendation=recommendation,
    )
    mode = resolved_execution.action
    canonical_strategy = resolved_execution.strategy
    if plan_supersession is not None:
        # Signed re-plan (#936): the existing PR is rebound to the
        # approved revision.  No fresh implementation turn runs and no
        # second PR is opened.
        if mode == "plan-only":
            print(
                f"Issue #{issue_number} re-planned child plan {plan_hash} approved by "
                f"{format_agent_list(configured_reviewers)}; rerun without plan-only to "
                f"rebind PR #{plan_supersession.pr_number}."
            )
            return 0
        _rebind_superseded_child_plan(
            runner,
            config=config,
            issue_number=issue_number,
            issue_context=issue_context,
            plan_supersession=plan_supersession,
            plan_hash=plan_hash,
        )
        issue_context = get_issue_context(
            runner, config=config, issue_number=issue_number
        )
        if parent_issue_context is not None:
            validate_pr_body_does_not_close_issue(
                runner,
                config=config,
                pr_number=plan_supersession.pr_number,
                issue_number=parent_issue_context.number,
            )
        return run_pr_loop(
            runner,
            pr_number=plan_supersession.pr_number,
            config=config,
            issue_context=issue_context,
            approved_plan_context=approved_plan_context,
            parent_issue_context=parent_issue_context,
            usage_context=usage_context,
            managed_ci_issue_number=issue_number,
        )
    if (
        canonical_strategy == "staged"
        and mode != "plan-only"
        and staged_parent_number is not None
        and _fresh_phase_marker_payload(issue_context) is not None
    ):
        # Nested-topology guard (#808): a planning child whose own
        # reviewed recommendation is staged stops for a human decision
        # before any decision record, checkpoint, summary, child
        # issue, handoff, or coder work.  Hierarchical execution is
        # tracked in #720.
        nested = NestedTopologyDecision(
            parent_issue=staged_parent_number,
            child_issue=issue_number,
            plan_hash=plan_hash,
            requested_policy=resolved_execution.requested_policy,
        )
        log(config, str(nested))
        print(json.dumps(nested.as_dict(), sort_keys=True))
        return 2
    # Entry deferral grants no execution authority to legacy plans.
    # Reapply the canonical barrier before any execution writes.
    if recommendation is None and mode != "plan-only":
        resolve_canonical_pr_for_issue(
            runner, config=config, issue_number=issue_number,
            issue_context=issue_context,
            expected_fallback_scope=IssuePrProvenanceScope(
                repository=config.repo, issue_number=issue_number,
                flow="approved", approved_plan_hash=plan_hash,
            ),
        )
    # Retirement verifies terminal history before any execution writes.
    retired_decision_hashes: frozenset[str] = frozenset()
    retiring_decision_hashes: tuple[str, ...] = ()
    if recommendation is not None and mode != "plan-only":
        (
            retired_decision_hashes,
            retiring_decision_hashes,
            execution_history,
        ) = _retirable_execution_decision_hashes(
            runner,
            config=config,
            issue_number=issue_number,
            comments=issue_context.comments,
            plan_hash=plan_hash,
        )
        if execution_history is not None:
            config = dataclasses_replace(config, execution_history=execution_history)
            issue_context = execution_history.view(issue_context)
    normalized_topology = None
    if recommendation is not None and canonical_strategy == "staged":
        normalized_topology = normalize_execution_recommendation(
            recommendation,
            approved_plan=current_plan,
            plan_subject=plan_subject,
        )
    if (
        normalized_topology is not None
        and canonical_strategy == "staged"
        and mode != "plan-only"
    ):
        staged_preflight = _preflight_fresh_staged_topology(
            runner,
            issue_number=issue_number,
            approved_plan=current_plan,
            config=config,
            issue_context=issue_context,
            mode=mode,
            normalized_topology=normalized_topology,
            retired_plan_hashes=retired_decision_hashes,
        )
        if isinstance(staged_preflight, NeedsHumanDecision):
            print(json.dumps(staged_preflight.as_dict(), sort_keys=True))
            return 2
    elif recommendation is not None and canonical_strategy == "one-shot" and mode != "plan-only":
        _preflight_fresh_one_shot_recovery(
            runner,
            issue_number=issue_number,
            approved_plan=current_plan,
            config=config,
            issue_context=issue_context,
            recommendation=recommendation,
            retired_plan_hashes=retired_decision_hashes,
        )
    plan_additions = _extract_current_expected_closing_issue_ids(current_plan)
    split_topology = bool(
        mode in {"decompose-only", "implement-by-phase"}
        or config.materialize_split_issues
        or _extract_current_child_stages(current_plan)
        or _extract_current_deferred_stages(current_plan)
    )
    if split_topology and (config.expected_closing_issue_ids or plan_additions):
        raise AgentLoopError(
            "Additional expected closing issue IDs are single-PR-only and cannot be "
            "carried through split/decomposition materialization. Invoke the actual "
            "child issue with a child-scoped --expected-closing-issue declaration."
        )
    if recommendation is not None:
        _print_execution_resolution_summary(
            issue_number=issue_number,
            resolved=resolved_execution,
            normalized_topology=normalized_topology,
            # The staged implement-by-phase dispatch path is the only
            # path that prints a progress-derived replacement for this
            # line; every other path keeps it.
            defer_child_work=(
                normalized_topology is not None
                and mode == "implement-by-phase"
                and not config.dry_run
            ),
        )
    if config.dry_run and resolved_execution.is_automatic:
        _print_dry_run_execution_preview(
            issue_number=issue_number,
            resolved=resolved_execution,
            normalized_topology=normalized_topology,
        )
        return 0
    _persist_execution_decision_if_needed(
        runner,
        config=config,
        issue_number=issue_number,
        current_plan=current_plan,
        issue_comments=issue_context.comments,
        recommendation=recommendation,
        requested_policy=resolved_execution.requested_policy,
        resolved_execution=resolved_execution,
        retired_plan_hashes=retired_decision_hashes,
        retires_plan_hashes=retiring_decision_hashes,
    )
    _publish_plan_approved_followups(
        runner,
        config=config,
        issue_number=issue_number,
        approved_plan=current_plan,
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        issue_comments=issue_context.comments,
        sources=approved_future_followup_sources,
        source_context=FollowupSourceContext(
            repo=config.repo,
            source_kind="plan",
            source_number=issue_number,
            source_identity=plan_hash,
            parent_issue_numbers=(issue_number,),
        ),
        allow_issue_filing=mode in {"implement-one-shot", "implement-by-phase"},
        usage_context=usage_context,
    )
    # Terminal entry deferral permits planning, not split writes. Keep
    # ordinary plan-only split behavior only without that handoff.
    terminal_plan_only = False
    if mode == "plan-only" and not config.dry_run:
        canonical = authenticate_canonical_issue_pr(
            runner, config=config, issue_number=issue_number,
            issue_context=issue_context,
        )
        terminal_plan_only = canonical is not None and canonical.state in {"CLOSED", "MERGED"}
    split_scope_materialized = False if terminal_plan_only else _handle_plan_first_split_scope(
        runner,
        issue_number=issue_number,
        config=config,
        current_plan=current_plan,
        plan_subject=plan_subject,
        issue_context=issue_context,
        execution_mode=mode,
        resolved_execution=resolved_execution,
    )
    if isinstance(split_scope_materialized, NeedsHumanDecision):
        print(json.dumps(split_scope_materialized.as_dict(), sort_keys=True))
        return 2
    if split_scope_materialized:
        # Refetch so downstream logic (selected-stage resolution below,
        # decomposition, etc.) sees the `AGENT_DISCUSS_SPLIT` comment
        # this call may have just posted, instead of the stale
        # pre-materialization snapshot (#492 review).
        issue_context = get_issue_context(runner, config=config, issue_number=issue_number)
    if mode == "plan-only":
        print(
            f"Issue #{issue_number} plan approved by "
            f"{format_agent_list(plan_approving_reviewers)}."
            + (f" {plan_amendment_note}" if plan_amendment_note else "")
        )
        deferred_titles = _typed_deferred_work_titles(current_plan_sidecar)
        if deferred_titles:
            print(
                "Deferred work recorded only (not tracked or materialized; file "
                "follow-up issues manually): " + "; ".join(deferred_titles)
            )
        return 0

    if mode in {"decompose-only", "implement-by-phase"}:
        staged_outcome = _decompose_approved_plan(
            runner,
            issue_number=issue_number,
            approved_plan=current_plan,
            config=config,
            memory=memory,
            issue_context=issue_context,
            mode=mode,
            coder_session_id=coder_session_id,
            usage_context=usage_context,
            execution_recommendation=recommendation,
            normalized_topology=normalized_topology,
        )
        if isinstance(staged_outcome, NeedsHumanDecision):
            print(json.dumps(staged_outcome.as_dict(), sort_keys=True))
            return 2
        if normalized_topology is not None:
            final_integration = normalized_topology[0].final_integration_work
            print(
                "Final integration work: "
                f"{final_integration.status}; deliverables: "
                f"{', '.join(final_integration.deliverables) or 'none'}; "
                "acceptance criteria: "
                f"{', '.join(final_integration.acceptance_criteria) or 'none'}; "
                "covered scope items: "
                f"{', '.join(final_integration.covered_scope_item_ids) or 'none'}"
            )
            if mode == "decompose-only":
                # `decompose-only` never enters the phase dispatcher,
                # so it keeps today's topology-only stage listing.
                remaining_stage_ids = tuple(
                    phase.stage_id or str(phase.position)
                    for phase in normalized_topology[0].phases[1:]
                )
                print(
                    "Remaining child work after the first phase: "
                    f"{', '.join(remaining_stage_ids) or 'none'}"
                )
            # One relaunch line per child, by its reviewed route (#1331).
            print("Relaunch commands:")
            for line in _staged_child_relaunch_lines(
                normalized_topology[0], staged_outcome
            ):
                print(f"  {line}")
        if mode == "decompose-only":
            print(f"Issue #{issue_number} approved plan decomposed into child issues.")
            return 0
        return _dispatch_current_decomposition_phase(
            runner,
            run_plan_first_loop=run_plan_first_loop,
            config=config,
            memory=memory,
            usage_context=usage_context,
            issue_number=issue_number,
            current_plan=current_plan,
            plan_subject=plan_subject,
            outcome=staged_outcome,
            recommendation=recommendation,
            approved_plan_context=approved_plan_context,
            issue_context=issue_context,
            mode=mode,
            coder_session_id=coder_session_id,
        )

    if mode == "implement-one-shot":
        plan_hash = approved_plan_hash(current_plan)
        plan_subject = _plan_subject(current_plan)

        # Selected-stage handoff (#476): when the parent's split proposals were
        # already fully materialized into child issues (discuss `split` or a prior
        # plan-first run), implementation must target the child the approved plan
        # actually covers instead of treating the whole parent as solved.
        #
        # This does NOT apply when the approved plan itself structurally declares
        # `deferred_stages`: that plan keeps its own primary scope on the parent and
        # only files the *remainder* as children (`_handle_plan_first_split_scope`),
        # so the parent is never one of the materialized children. Skipping stage
        # resolution here also fixes a rerun/resume regression (#492 review): once
        # such a plan's own one-shot handoff is posted on the parent, a later rerun
        # would otherwise see the freshly materialized children, find no stage-handoff
        # marker, and fail to match the plan's own title against a sibling stage's
        # title, raising instead of resuming the already-handed-off parent PR.
        target_issue_number = issue_number
        target_issue_context = issue_context
        staged_parent_issue: int | None = None
        if (
            staged_parent_number is not None
            and parent_issue_context is not None
            and _fresh_phase_marker_payload(issue_context) is not None
        ):
            # A separately planned decomposition child (#808)
            # implements its own one-shot recommendation with the
            # parent recorded as the staged parent.
            staged_parent_issue = staged_parent_number
        # Explicit child stages are the parent plan's bounded remainder;
        # legacy deferred entries preserve the same historical parent-owned
        # behavior. Dependencies/actions alone must not suppress handoff.
        current_plan_declares_own_deferred_stages = bool(
            _extract_current_child_stages(current_plan)
            or _extract_current_deferred_stages(current_plan)
        )
        split_metadata = find_existing_split_materialization(
            issue_context.comments, parent_issue=issue_number
        )
        if (
            split_metadata is not None
            and split_metadata.children
            and not current_plan_declares_own_deferred_stages
        ):
            stage_handoff = find_existing_split_stage_handoff(
                issue_context.comments, parent_issue=issue_number, plan_hash=plan_hash
            )
            if stage_handoff is not None:
                selected_child = next(
                    (
                        child
                        for child in split_metadata.children
                        if child.number == stage_handoff.child_issue_number
                    ),
                    None,
                )
                if selected_child is None or selected_child.number is None:
                    raise AgentLoopError(
                        f"Recorded split-stage handoff for issue #{issue_number} references "
                        "a child issue that is no longer in the materialized split metadata; "
                        "manual recovery required."
                    )
            else:
                selected_child = resolve_selected_stage_child(
                    split_metadata.children,
                    parent_issue=issue_number,
                    plan_title_or_subject=_plan_first_line(current_plan),
                    split_stage_flag=config.split_stage,
                )
                post_split_stage_handoff_comment(
                    runner,
                    config=config,
                    parent_issue=issue_number,
                    plan_hash=plan_hash,
                    child=selected_child,
                )
            target_issue_number = selected_child.number
            target_issue_context = get_issue_context(
                runner, config=config, issue_number=target_issue_number
            )
            staged_parent_issue = issue_number
            parent_issue_context = get_issue_context(
                runner, config=config, issue_number=issue_number
            )

        # Canonical handoff resolution runs before the (older,
        # plan-hash-scoped) one-shot handoff lookup below, so a stale
        # canonical record fails safely instead of being bypassed by
        # it (#589). It is scoped to the selected target issue, since
        # a split-stage plan implements a child, not the parent.
        resolved_pr = resolve_canonical_pr_for_issue(
            runner,
            config=config,
            issue_number=target_issue_number,
            issue_context=target_issue_context,
            expected_fallback_scope=IssuePrProvenanceScope(
                repository=config.repo,
                issue_number=target_issue_number,
                flow="approved",
                approved_plan_hash=plan_hash,
            ),
        )
        if resolved_pr is not None:
            if (
                resolved_pr.source == "canonical"
                and resolved_pr.metadata is not None
                and resolved_pr.metadata.flow == "approved-plan-implementation"
                and resolved_pr.metadata.plan_hash != plan_hash
            ):
                raise AgentLoopError(
                    f"Canonical approved-plan handoff for issue #{target_issue_number} "
                    f"points to PR #{resolved_pr.pr_number} with plan hash "
                    f"{resolved_pr.metadata.plan_hash}, but the current approved plan "
                    f"has hash {plan_hash}. Review it with `agent-loop pr "
                    f"{resolved_pr.pr_number}` or remove the stale handoff marker."
                )
            log(
                config,
                f"Issue #{target_issue_number}: resuming PR #{resolved_pr.pr_number} review for "
                f"already-handed-off plan (source={resolved_pr.source}, "
                f"evidence={resolved_pr.evidence_summary})",
            )
            if staged_parent_issue is not None:
                validate_pr_body_does_not_close_issue(
                    runner,
                    config=config,
                    pr_number=resolved_pr.pr_number,
                    issue_number=staged_parent_issue,
                )
            if resolved_pr.source == "legacy-closing-reference":
                resumed_pr_context = get_observed_pr_review_context(
                    runner, config=config, pr_number=resolved_pr.pr_number
                )
                pr_url, pr_head_sha = require_pr_metadata_for_handoff(resumed_pr_context.metadata)
                post_issue_pr_handoff_comment(
                    runner,
                    config=config,
                    issue_number=target_issue_number,
                    pr_number=resolved_pr.pr_number,
                    pr_url=pr_url,
                    pr_head_sha=pr_head_sha,
                    flow="approved-plan-implementation",
                    plan_hash=plan_hash,
                    plan_growth_verdict=approval_growth_verdict,
                )
            return run_pr_loop(
                runner,
                pr_number=resolved_pr.pr_number,
                config=config,
                issue_context=target_issue_context,
                approved_plan_context=approved_plan_context,
                parent_issue_context=(
                    parent_issue_context if staged_parent_issue is not None else None
                ),
                usage_context=usage_context,
                managed_ci_issue_number=target_issue_number,
            )

        existing_handoff = find_existing_one_shot_impl_handoff(
            target_issue_context.comments,
            parent_issue=target_issue_number,
            plan_hash=plan_hash,
            mode="implement-one-shot",
        )
        any_one_shot_handoff = find_latest_one_shot_impl_handoff(
            target_issue_context.comments,
            parent_issue=target_issue_number,
            mode="implement-one-shot",
        )
        if (
            existing_handoff is None
            and any_one_shot_handoff is not None
            and any_one_shot_handoff.plan_hash != plan_hash
        ):
            try:
                older_state = get_pr_state(
                    runner, config=config, pr_number=any_one_shot_handoff.pr_number
                )
            except AgentLoopError as exc:
                raise AgentLoopError(
                    f"Older one-shot handoff for PR #{any_one_shot_handoff.pr_number} "
                    f"cannot be validated ({exc}). Review it directly with `agent-loop pr "
                    f"{any_one_shot_handoff.pr_number}` or remove the stale handoff."
                ) from exc
            if older_state == "OPEN":
                raise AgentLoopError(
                    f"Open one-shot handoff for PR #{any_one_shot_handoff.pr_number} has "
                    f"older plan hash {any_one_shot_handoff.plan_hash}, but the current "
                    f"approved plan has hash {plan_hash}. Review the recorded PR with "
                    f"`agent-loop pr {any_one_shot_handoff.pr_number}` or remove the "
                    "stale handoff before creating another implementation PR."
                )
        if existing_handoff is not None:
            try:
                pr_state = get_pr_state(
                    runner, config=config, pr_number=existing_handoff.pr_number
                )
            except AgentLoopError:
                raise AgentLoopError(
                    f"PR #{existing_handoff.pr_number} recorded in the one-shot handoff "
                    f"for issue #{target_issue_number} cannot be found in {config.repo}. "
                    f"Verify the PR exists and rerun "
                    f"`agent-loop pr {existing_handoff.pr_number}` directly to continue, "
                    "or remove the handoff comment from the issue and rerun to re-implement."
                )
            if pr_state == "OPEN":
                log(
                    config,
                    f"Issue #{target_issue_number}: resuming PR #{existing_handoff.pr_number} "
                    "review for already-handed-off plan",
                )
                if staged_parent_issue is not None:
                    validate_pr_body_does_not_close_issue(
                        runner,
                        config=config,
                        pr_number=existing_handoff.pr_number,
                        issue_number=staged_parent_issue,
                    )
                return run_pr_loop(
                    runner,
                    pr_number=existing_handoff.pr_number,
                    config=config,
                    issue_context=target_issue_context,
                    approved_plan_context=approved_plan_context,
                    parent_issue_context=(
                        parent_issue_context if staged_parent_issue is not None else None
                    ),
                    usage_context=usage_context,
                    managed_ci_issue_number=target_issue_number,
                )
            else:
                print(
                    f"Issue #{target_issue_number} approved plan was handed off to "
                    f"PR #{existing_handoff.pr_number}, which is "
                    f"{pr_state.lower()}. Nothing to resume."
                )
                return 0
        return _implement_approved_issue(
            runner,
            issue_number=target_issue_number,
            approved_plan=current_plan,
            config=config,
            memory=memory,
            issue_context=target_issue_context,
            approved_plan_context=approved_plan_context,
            parent_issue_context=(
                parent_issue_context if staged_parent_issue is not None else None
            ),
            coder_session_id=coder_session_id,
            usage_context=usage_context,
            one_shot_parent_issue=target_issue_number,
            plan_subject=plan_subject,
            staged_parent_issue=staged_parent_issue,
            execution_recommendation=recommendation,
            plan_growth_verdict=approval_growth_verdict,
        )
    raise AgentLoopError(f"Unknown plan execution mode: {mode}")
