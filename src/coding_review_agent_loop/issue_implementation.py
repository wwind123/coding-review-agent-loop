"""Approved-plan implementation and plan decomposition.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1201); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

from dataclasses import replace as dataclasses_replace
from .agents.registry import agent_display_name
from .config import (
    AgentLoopConfig,
    sync_coder_base_before_implementation,
)
from .decomposition import (
    CreatedPhaseIssue,
    PlanDecomposition,
    RecordedPhase,
    approved_plan_hash,
    create_decomposition_child_issues,
    find_existing_decomposition,
    parse_plan_decomposition,
    post_decomposition_parent_summary,
    post_one_shot_impl_handoff_comment,
    adapt_typed_child_stages,
    normalize_execution_recommendation,
    validate_risk_matrix_ownership,
    EXECUTION_TOPOLOGY_SOURCE,
    reject_legacy_topology_collision,
    find_existing_topology_checkpoint,
)
from .child_topology import NeedsHumanDecision
from .errors import (
    AgentInvocationError,
    CheckoutVerificationError,
    AgentLoopError,
)
from .expected_closure import (
    reject_parent_from_contract,
    resolve_issue_contract,
)
from .github import (
    IssueContext,
    PullRequestMetadata,
    deduplicate_human_requirements,
    get_pr_review_context,
    post_trusted_pr_contract_record,
    post_trusted_pr_comment,
    reject_forged_protocol_markers,
    validate_open_pr,
    validate_pr_body_does_not_close_issue,
    validate_pr_expected_closing_issues,
    validate_pr_references_issue,
    validate_pull_request_provenance,
)
from .issue_pr_handoff import (
    post_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
    resolve_canonical_pr_for_issue,
)
from .issue_pr_provenance import IssuePrProvenanceScope
from .phase_progress import StagedTopologyOutcome
from .pr_contract import (
    PrExpectedClosingContract,
    format_pr_contract_comment,
    make_pr_contract,
    render_pr_contract_marker,
)
from .logging import log
from .managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    authenticate_issue_created_handoff,
    preflight_managed_ci_creation,
    publish_issue_created_authorization,
    render_managed_ci_resume_command,
)
from .prompts import (
    build_issue_implementation_prompt,
    build_plan_decomposition_prompt,
    render_coder_human_requirements_prompt_context,
)
from .protocol import (
    StructuredIssueImplementation,
    parse_architecture_impact,
    sanitize_architecture_impact,
)
from .runner import Runner
from .salvage import (
    SalvageContext,
    latest_salvage_context,
)
from .usage import RunUsageContext
from .workdirs import active_workdir
from .workdir_guard import validate_assigned_head_advanced
from .comment_rendering import render_public_agent_comment
from .round_state import (
    ApprovedPlanContext,
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
    make_approved_plan_context,
)
from .plan_growth import PlanGrowthApprovalVerdict
from .protocol_markers import (
    TrustedBody,
    scan_reserved_markers,
)
from .agent_failure import (
    APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
    _metadata_identity_fields,
)
from .architecture_contract import (
    _freeze_prompt_architecture,
    _architecture_metadata_fields,
    _test_observation_degradation_fields,
    _TerminalNoPrImplementation,
    _TerminalIssueImplementationConflict,
    _architecture_mode_validators,
    _risk_coverage_reask_prompt,
    _surface_decomposition_degradations,
    _surface_refused_decomposition,
)
from .risk_coverage_map import coverage_map_applies
from .validated_agent import (
    CompletionRecoveryPolicy,
    _run_validated_agent,
)
from .response_validation import (
    _validate_issue_implementation_response,
    _current_test_turn_observations,
    _derive_authenticated_risk_evidence_for_coder,
    assess_risk_coverage_map,
    final_risk_coverage_assessment,
    _post_no_pr_implementation_terminal_comment,
    _post_structured_issue_implementation_terminal_comment,
    _degrade_out_of_checkout_tests,
    _validate_structured_response_tests_with_post_pr_context,
    _validate_structured_response_observations_with_post_pr_context,
)
from .panel_evidence import _plan_growth_verdict_for_hash
from .execution_policy import (
    _extract_current_expected_closing_issue_ids,
    _extract_current_child_stages,
    _resolved_stage_ids,
)
from .pr_loop_support import (
    _print_unprotected_managed_ci_warning,
    _read_assigned_workdir_head,
)
from .pr_loop import run_pr_loop


def _embed_pr_contract_marker(body: str | TrustedBody, contract: PrExpectedClosingContract) -> TrustedBody:
    marker = render_pr_contract_marker(contract)
    body_text = str(body)
    if "\n-- " in body_text:
        prefix, signature = body_text.rsplit("\n-- ", 1)
        rendered = f"{prefix}\n{marker}\n-- {signature}"
    else:
        rendered = f"{body_text.rstrip()}\n{marker}"
    expected = tuple(item.definition.token for item in scan_reserved_markers(rendered))
    return TrustedBody.canonical(rendered, expected_tokens=expected)


def _advisory_issue_pr_provenance(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_scope: IssuePrProvenanceScope,
) -> None:
    """Warn on missing provenance without blocking a newly created PR handoff."""
    if config.dry_run:
        return
    try:
        validate_pull_request_provenance(
            runner,
            config=config,
            pr_number=pr_number,
            expected_scope=expected_scope,
        )
    except AgentLoopError as exc:
        log(
            config,
            f"WARNING: PR #{pr_number} did not prove expected issue commit provenance "
            f"(repository={expected_scope.repository}, issue=#{expected_scope.issue_number}, "
            f"flow={expected_scope.flow}, plan={expected_scope.approved_plan_hash or 'none'}): {exc}. "
            "Do not rewrite or force-push solely to satisfy this warning. If execution is "
            "interrupted before handoff, resume the PR directly with "
            f"`agent-loop pr {pr_number}`.",
        )


def _publish_issue_authorization_with_recovery(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    handoff: AuthenticatedIssueCreatedHandoff,
    metadata: PullRequestMetadata,
    issue_number: int,
    approved_plan_hash_value: str | None = None,
) -> AuthenticatedIssueCreatedHandoff:
    """Publish the durable checkpoint or give an authority-changing remedy."""
    try:
        return publish_issue_created_authorization(
            runner,
            config=config,
            handoff=handoff,
            metadata=metadata,
            approved_plan_hash=approved_plan_hash_value,
        )
    except AgentLoopError as exc:
        command = render_managed_ci_resume_command(
            config,
            pr_number=handoff.pr_number,
            issue_number=issue_number,
            managed_ci=True,
            fresh_authorization=True,
            fresh_issue_number=issue_number,
        )
        raise AgentLoopError(
            f"{exc}\n\nManaged-CI authorization publication was interrupted and no "
            "handoff or qualification was claimed. After verifying the PR, create an "
            f"explicit new operator authorization with `{command}`."
        ) from exc


def _approved_implementation_config(config: AgentLoopConfig) -> tuple[AgentLoopConfig, bool]:
    """Return the config and session-reuse policy for approved plan implementation."""
    implementation_coder = config.implementation_coder or config.coder
    updates: dict[str, object] = {"coder": implementation_coder}
    reuse_session = implementation_coder == config.coder

    model = config.implementation_coder_model.strip()
    if model:
        reuse_session = False
        if implementation_coder == "claude":
            updates["claude_model"] = model
        elif implementation_coder == "codex":
            updates["codex_model"] = model
        elif implementation_coder == "gemini":
            updates["gemini_model"] = model
        elif implementation_coder == "antigravity":
            updates["antigravity_model"] = None
            updates["antigravity_models"] = (model,)

    effort = config.implementation_codex_reasoning_effort.strip()
    if effort:
        reuse_session = False
        updates["implementation_effort_active"] = True

    claude_effort = config.implementation_claude_effort.strip()
    if claude_effort:
        reuse_session = False
        updates["implementation_effort_active"] = True

    if updates == {"coder": config.coder}:
        return config, True
    return dataclasses_replace(config, **updates), reuse_session


def _implement_approved_issue(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    coder_session_id: str | None,
    usage_context: RunUsageContext,
    one_shot_parent_issue: int | None = None,
    plan_subject: str | None = None,
    staged_parent_issue: int | None = None,
    approved_plan_context: ApprovedPlanContext | None = None,
    parent_issue_context: IssueContext | None = None,
    execution_recommendation=None,
    plan_growth_verdict: PlanGrowthApprovalVerdict | None = None,
) -> int:
    implementation_config, reuse_planning_session = _approved_implementation_config(config)
    coder_name = agent_display_name(implementation_config.coder)
    implementation_session_id = coder_session_id if reuse_planning_session else None
    plan_hash = (
        approved_plan_context.plan_hash
        if approved_plan_context is not None and approved_plan_context.plan_hash
        else approved_plan_hash(approved_plan)
    )
    if plan_growth_verdict is None:
        # Callers without the planning loop's own approval-time assessment
        # measure the approved candidate from its authenticated plan round.
        plan_growth_verdict = _plan_growth_verdict_for_hash(
            config,
            plan_hash=plan_hash,
            comment_sources=(
                issue_context.comments,
                parent_issue_context.comments if parent_issue_context is not None else None,
            ),
        )
    execution_identity = (
        execution_recommendation.identity()
        if execution_recommendation is not None else None
    )
    if approved_plan_context is None:
        approved_plan_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan implementation",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
    if execution_recommendation is not None and approved_plan_context.matrix_available:
        validate_risk_matrix_ownership(
            approved_plan_context.risk_test_matrix_payload,
            execution_recommendation,
        )
    plan_additions = _extract_current_expected_closing_issue_ids(approved_plan)
    implementation_requirements = deduplicate_human_requirements(
        [
            *(parent_issue_context.human_requirements if parent_issue_context is not None else ()),
            *issue_context.human_requirements,
        ]
    )
    implementation_human_requirements_context = render_coder_human_requirements_prompt_context(
        implementation_requirements,
    )

    # A prior implementation attempt may have created a PR and then aborted
    # before recording any handoff marker/comment (e.g. the #493 test-report
    # false positive, which produced the duplicate PR #494 for #492). Resolve
    # the canonical AGENT_ISSUE_PR_HANDOFF record first, falling back to the
    # legacy exactly-one-open-PR GitHub search when no record exists yet
    # (#495, #589).
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
    recovered_contract_ids = (
        resolved_pr.metadata.expected_closing_issue_ids
        if resolved_pr is not None and resolved_pr.metadata is not None
        else (issue_number,) if resolved_pr is not None else None
    )
    closing_contract = resolve_issue_contract(
        primary_issue=issue_number,
        cli_additions=config.expected_closing_issue_ids,
        plan_additions=plan_additions,
        recovered=recovered_contract_ids,
        supersede=config.supersede_expected_closing_contract,
    )
    reject_parent_from_contract(closing_contract, parent_issue=staged_parent_issue)
    implementation_config = dataclasses_replace(
        implementation_config,
        expected_closing_issue_ids=closing_contract.issue_ids,
        expected_closing_contract_resolved=True,
    )
    if resolved_pr is not None:
        existing_pr_number = resolved_pr.pr_number
        if (
            resolved_pr.source == "canonical"
            and resolved_pr.metadata is not None
            and resolved_pr.metadata.flow == "approved-plan-implementation"
            and resolved_pr.metadata.plan_hash != plan_hash
        ):
            raise AgentLoopError(
                f"Canonical approved-plan handoff for issue #{issue_number} points to PR "
                f"#{existing_pr_number} with plan hash {resolved_pr.metadata.plan_hash}, "
                f"but the current approved plan has hash {plan_hash}. Review the recorded PR "
                f"with `agent-loop pr {existing_pr_number}` or remove the stale handoff marker."
            )
        log(
            config,
            f"Existing implementation PR #{existing_pr_number} found for issue #{issue_number} "
            f"/ approved plan {plan_hash}; resuming PR review instead of invoking {coder_name} "
            f"(source={resolved_pr.source}, evidence={resolved_pr.evidence_summary}).",
        )
        if staged_parent_issue is not None:
            validate_pr_body_does_not_close_issue(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                issue_number=staged_parent_issue,
            )
        resumed_pr_context = None
        if resolved_pr.source == "legacy-closing-reference" or one_shot_parent_issue is not None:
            resumed_pr_context = get_pr_review_context(
                runner, config=implementation_config, pr_number=existing_pr_number
            )
        # Explicit managed recovery may be resuming a PR whose implementation
        # report was rejected before the canonical handoff checkpoint. Its
        # durable authorization is sufficient to enter the PR loop, but must
        # not launder that rejected report into a canonical handoff.
        if resolved_pr.source == "legacy-closing-reference":
            pr_url, pr_head_sha = require_pr_metadata_for_handoff(resumed_pr_context.metadata)
            validate_pr_expected_closing_issues(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                expected_issue_ids=closing_contract.issue_ids,
                body=resumed_pr_context.metadata.body,
                reject_unexpected=implementation_config.managed_ci,
            )
            pr_contract = make_pr_contract(
                repository=implementation_config.repo,
                pr_number=existing_pr_number,
                origin_flow="approved-plan-implementation",
                primary_issue_number=issue_number,
                expected_closing_issue_ids=closing_contract.issue_ids,
                supersedes_hash=closing_contract.supersedes_hash,
            )
            post_trusted_pr_comment(
                runner,
                config=implementation_config,
                pr_number=existing_pr_number,
                body=TrustedBody.canonical(
                    format_pr_contract_comment(pr_contract),
                    expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
                ),
            )
            if not implementation_config.managed_ci:
                post_issue_pr_handoff_comment(
                    runner,
                    config=implementation_config,
                    issue_number=issue_number,
                    pr_number=existing_pr_number,
                    pr_url=pr_url,
                    pr_head_sha=pr_head_sha,
                    flow="approved-plan-implementation",
                    plan_hash=plan_hash,
                    expected_closing_issue_ids=closing_contract.issue_ids,
                    supersedes_hash=closing_contract.supersedes_hash,
                    plan_growth_verdict=plan_growth_verdict,
                )
        if one_shot_parent_issue is not None:
            post_one_shot_impl_handoff_comment(
                runner,
                config=implementation_config,
                parent_issue=one_shot_parent_issue,
                mode="implement-one-shot",
                plan_hash=plan_hash,
                plan_subject=plan_subject or "",
                pr_number=existing_pr_number,
                pr_head_sha=resumed_pr_context.metadata.head_sha,
                strategy=(execution_recommendation.strategy if execution_recommendation is not None else None),
                topology_source=(
                    str(execution_identity["topology_source"])
                    if execution_identity is not None else None
                ),
                execution_strategy_contract_version=(
                    1 if execution_recommendation is not None else None
                ),
                recommendation_digest=(
                    str(execution_identity["recommendation_sha256"])
                    if execution_identity is not None else None
                ),
            )
        return run_pr_loop(
            runner,
            pr_number=existing_pr_number,
            config=implementation_config,
            issue_context=issue_context,
            approved_plan_context=approved_plan_context,
            parent_issue_context=parent_issue_context,
            usage_context=usage_context,
            managed_ci_issue_number=issue_number,
        )

    salvage_summary = latest_salvage_context(
        implementation_config.log_dir,
        issue_context.comments,
        repo=implementation_config.repo,
        issue_number=issue_number,
        scope=APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
        approved_plan_hash=plan_hash,
    )
    sync_coder_base_before_implementation(implementation_config, runner)
    implementation_config = _freeze_prompt_architecture(runner, implementation_config)
    managed_ci_creation_intent = preflight_managed_ci_creation(
        runner, config=implementation_config, issue_number=issue_number
    )
    if managed_ci_creation_intent is not None and managed_ci_creation_intent.audit_nonce:
        _print_unprotected_managed_ci_warning(managed_ci_creation_intent.protection_mode)
    log(config, f"Planning approved; invoking {coder_name} to implement issue #{issue_number}")
    assigned_head_before = _read_assigned_workdir_head(runner, implementation_config)
    implementation_prompt = build_issue_implementation_prompt(
        issue_number,
        approved_plan,
        implementation_config,
        memory,
        issue_context=issue_context,
        salvage_summary=salvage_summary,
        staged_parent_issue=staged_parent_issue,
        managed_ci_creation_intent=managed_ci_creation_intent,
        approved_plan_context=approved_plan_context,
        parent_issue_context=parent_issue_context,
    )

    def _invoke_implementation_coder(prompt, *, session_id, attempt_label, invoke_config=implementation_config):
        log(config, f"Invoking {coder_name} for approved-plan implementation ({attempt_label})")
        return _run_validated_agent(
            runner,
            agent=invoke_config.coder,
            config=invoke_config,
            prompt=prompt,
            session_id=session_id,
            marker_description="structured issue_implementation result, blocking, or clarification",
            require_architecture_impact_contract=True,
            **_architecture_mode_validators(lambda mode: lambda text: _validate_issue_implementation_response(
                text,
                human_requirements=implementation_requirements,
                require_architecture_impact=True,
                delivered_risk_test_matrix=(
                    approved_plan_context.risk_test_matrix_payload
                    if approved_plan_context is not None and approved_plan_context.matrix_available
                    else None
                ),
                delivered_risk_test_matrix_identity=(
                    approved_plan_context.risk_test_matrix_identity
                    if approved_plan_context is not None and approved_plan_context.matrix_available
                    else None
                ),
                require_risk_test_matrix_contract=(
                    approved_plan_context is not None and approved_plan_context.matrix_available
                ),
                authoritative_test_observations=_current_test_turn_observations(runner),
                execution_catalog=_current_test_turn_observations(runner),
                delivered_risk_test_matrix_row_ids=(
                    approved_plan_context.risk_test_matrix_expected_row_ids
                    if approved_plan_context is not None and approved_plan_context.matrix_available
                    else None
                ), architecture_status_mode=mode,
            )),
            usage_context=usage_context,
            role="coder",
            reask_on_evidence_rejection=True,
            use_repair=True,
            repair_expected_kind="issue_implementation",
            repair_surfaced_requirement_ids=implementation_human_requirements_context.surfaced_requirement_ids,
            repair_requires_direct_discussion_ack=implementation_human_requirements_context.requires_direct_discussion_ack,
            salvage_context=SalvageContext(
                repo=implementation_config.repo,
                issue_number=issue_number,
                scope=APPROVED_PLAN_IMPLEMENTATION_SALVAGE_SCOPE,
                agent=implementation_config.coder,
                run_id=usage_context.run_id,
                approved_plan_hash=plan_hash,
            ),
            operation_description="approved-plan implementation",
            completion_recovery=CompletionRecoveryPolicy(
                issue_number=issue_number,
                issue_context=issue_context,
                approved_plan_context=approved_plan_context,
                parent_issue_context=parent_issue_context,
                human_requirements=implementation_requirements,
            ),
            managed_ci_recovery_protection=(
                managed_ci_creation_intent.protection_mode
                if managed_ci_creation_intent is not None else None
            ),
        )
    coder_response = _invoke_implementation_coder(
        implementation_prompt,
        session_id=implementation_session_id,
        attempt_label="initial",
    )
    coder_output = coder_response.text
    implementation_result = coder_response.marker_value
    if isinstance(implementation_result, _TerminalIssueImplementationConflict):
        implementation_result = _TerminalIssueImplementationConflict(
            _degrade_out_of_checkout_tests(
                implementation_result.parsed, config=implementation_config
            )
        )
        _post_structured_issue_implementation_terminal_comment(
            runner,
            config=implementation_config,
            issue_number=issue_number,
            parsed=implementation_result.parsed,
            model_used=coder_response.model_used,
        )
        raise AgentLoopError(
            "Coder implementation result was not accepted for handoff because a signed "
            "human requirement is blocked."
        )
    if isinstance(implementation_result, StructuredIssueImplementation):
        if implementation_result.pr_number is None:
            implementation_result = _degrade_out_of_checkout_tests(
                implementation_result, config=implementation_config
            )
            _post_structured_issue_implementation_terminal_comment(
                runner,
                config=implementation_config,
                issue_number=issue_number,
                parsed=implementation_result,
                model_used=coder_response.model_used,
            )
            raise AgentLoopError(
                "Coder did not create a valid PR; implementation is blocking."
            )
        pr_number = implementation_result.pr_number
    elif isinstance(implementation_result, _TerminalNoPrImplementation):
        _post_no_pr_implementation_terminal_comment(
            runner,
            config=implementation_config,
            issue_number=issue_number,
            coder_response=coder_response,
        )
        raise AgentLoopError(
            "Coder did not create a valid PR; implementation is " + implementation_result.state + "."
        )
    else:
        raise AgentLoopError("Issue implementation validator returned an unknown result type.")
    def _authenticate_reported_pr(cfg):
        """Read-only identity and managed-CI authorization of the reported PR."""
        validate_assigned_head_advanced(
            before_head=assigned_head_before,
            after_head=_read_assigned_workdir_head(runner, cfg),
            assigned_workdir=active_workdir(cfg),
        )
        validate_open_pr(runner, config=cfg, pr_number=pr_number)
        context = get_pr_review_context(runner, config=cfg, pr_number=pr_number)
        handoff: AuthenticatedIssueCreatedHandoff | None = None
        if managed_ci_creation_intent is not None:
            handoff = authenticate_issue_created_handoff(
                runner,
                config=cfg,
                intent=managed_ci_creation_intent,
                issue_number=issue_number,
                pr_number=pr_number,
                metadata=context.metadata,
            )
            if handoff.override_nonce is not None:
                # Install the expected nonce before any PR/issue publication.  It
                # remains runtime-only and is revalidated at run_pr_loop entry.
                cfg = dataclasses_replace(
                    cfg, managed_ci_expected_override_nonce=handoff.override_nonce,
                )
        else:
            reject_forged_protocol_markers(
                context.metadata.body or "",
                surface=f"pull-request #{pr_number} body",
            )
        return context, handoff, cfg

    def _read_only_reference_guards(context, cfg):
        validate_pr_references_issue(
            runner,
            config=cfg,
            pr_number=pr_number,
            issue_number=issue_number,
            staged_parent_issue=staged_parent_issue,
            body=context.metadata.body,
        )
        validate_pr_expected_closing_issues(
            runner,
            config=cfg,
            pr_number=pr_number,
            expected_issue_ids=closing_contract.issue_ids,
            body=context.metadata.body,
        )

    log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
    initial_pr_context, managed_ci_handoff, implementation_config = _authenticate_reported_pr(
        implementation_config
    )
    coverage_reask_used = False
    coverage_discard_note: str | None = None
    if coverage_map_applies(approved_plan_context):
        # Read-only guards run before any coverage invocation so a wrongly
        # reported PR never receives a mutating continuation (#1290).
        _read_only_reference_guards(initial_pr_context, implementation_config)
        gate_assessment = assess_risk_coverage_map(
            implementation_result,
            approved_plan_context=approved_plan_context,
            workdir=active_workdir(implementation_config),
            head_sha=initial_pr_context.metadata.head_sha,
        )
        if gate_assessment is not None and gate_assessment.deficiencies and gate_assessment.tree_available:
            coverage_reask_used = True
            log(
                config,
                f"Risk-matrix coverage map incomplete for PR #{pr_number} "
                f"({', '.join(gate_assessment.deficient_row_ids)}); sending one coverage re-ask",
            )
            try:
                reask_response = _invoke_implementation_coder(
                    _risk_coverage_reask_prompt(implementation_prompt, gate_assessment),
                    session_id=coder_response.session_id,
                    attempt_label="coverage-reask",
                    invoke_config=implementation_config,
                )
            except CheckoutVerificationError:
                raise
            except AgentLoopError as exc:
                coverage_discard_note = (
                    f"coverage re-ask response discarded: invocation failed ({type(exc).__name__})"
                )
                log(config, coverage_discard_note)
            else:
                reask_result = reask_response.marker_value
                if isinstance(reask_result, _TerminalIssueImplementationConflict):
                    reask_result = _TerminalIssueImplementationConflict(
                        _degrade_out_of_checkout_tests(
                            reask_result.parsed, config=implementation_config
                        )
                    )
                    _post_structured_issue_implementation_terminal_comment(
                        runner,
                        config=implementation_config,
                        issue_number=issue_number,
                        parsed=reask_result.parsed,
                        model_used=reask_response.model_used,
                    )
                    raise AgentLoopError(
                        "Coder implementation result was not accepted for handoff because a signed "
                        "human requirement is blocked."
                    )
                if (
                    isinstance(reask_result, StructuredIssueImplementation)
                    and reask_result.pr_number == pr_number
                ):
                    coder_response = reask_response
                    coder_output = reask_response.text
                    implementation_result = reask_result
                else:
                    coverage_discard_note = (
                        "coverage re-ask response discarded: it did not report the same PR"
                    )
                    log(config, coverage_discard_note)
            # The re-ask may have pushed or edited the PR whether or not its
            # response was adopted, so re-authenticate the refreshed PR before
            # any publication binds its head or body.
            initial_pr_context, managed_ci_handoff, implementation_config = _authenticate_reported_pr(
                implementation_config
            )
            _read_only_reference_guards(initial_pr_context, implementation_config)
    if managed_ci_handoff is not None:
        managed_ci_handoff = _publish_issue_authorization_with_recovery(
            runner,
            config=implementation_config,
            handoff=managed_ci_handoff,
            metadata=initial_pr_context.metadata,
            issue_number=issue_number,
            approved_plan_hash_value=plan_hash,
        )
    if isinstance(implementation_result, StructuredIssueImplementation):
        implementation_result = _validate_structured_response_tests_with_post_pr_context(
            implementation_result,
            runner=runner,
            config=implementation_config,
            pr_number=pr_number,
        )
        _validate_structured_response_observations_with_post_pr_context(
            implementation_result.test_observations,
            runner=runner,
            config=implementation_config,
            pr_number=pr_number,
        )
    validate_pr_references_issue(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        issue_number=issue_number,
        staged_parent_issue=staged_parent_issue,
        body=initial_pr_context.metadata.body,
    )
    validate_pr_expected_closing_issues(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        expected_issue_ids=closing_contract.issue_ids,
        body=initial_pr_context.metadata.body,
    )
    _advisory_issue_pr_provenance(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        expected_scope=IssuePrProvenanceScope(
            repository=implementation_config.repo,
            issue_number=issue_number,
            flow="approved",
            approved_plan_hash=plan_hash,
        ),
    )
    initial_pr_url, initial_pr_head_sha = require_pr_metadata_for_handoff(initial_pr_context.metadata)
    pr_contract = make_pr_contract(
        repository=implementation_config.repo,
        pr_number=pr_number,
        origin_flow="approved-plan-implementation",
        primary_issue_number=issue_number,
        expected_closing_issue_ids=closing_contract.issue_ids,
        supersedes_hash=closing_contract.supersedes_hash,
    )
    post_trusted_pr_contract_record(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        body=TrustedBody.canonical(
            format_pr_contract_comment(pr_contract),
            expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
        ),
    )
    post_issue_pr_handoff_comment(
        runner,
        config=implementation_config,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=initial_pr_url,
        pr_head_sha=initial_pr_head_sha,
        flow="approved-plan-implementation",
        plan_hash=plan_hash,
        expected_closing_issue_ids=closing_contract.issue_ids,
        supersedes_hash=closing_contract.supersedes_hash,
        plan_growth_verdict=plan_growth_verdict,
    )
    if one_shot_parent_issue is not None:
        post_one_shot_impl_handoff_comment(
            runner,
            config=implementation_config,
            parent_issue=one_shot_parent_issue,
            mode="implement-one-shot",
            plan_hash=plan_hash,
            plan_subject=plan_subject or "",
            pr_number=pr_number,
            pr_head_sha=initial_pr_context.metadata.head_sha,
            strategy=(execution_recommendation.strategy if execution_recommendation is not None else None),
            topology_source=(
                str(execution_identity["topology_source"])
                if execution_identity is not None else None
            ),
            execution_strategy_contract_version=(
                1 if execution_recommendation is not None else None
            ),
            recommendation_digest=(
                str(execution_identity["recommendation_sha256"])
                if execution_identity is not None else None
            ),
        )
    implementation_result, _initial_derived_risk_evidence = _derive_authenticated_risk_evidence_for_coder(
        implementation_result,
        approved_plan_context=approved_plan_context,
        runner=runner,
        assigned_workdir=active_workdir(implementation_config),
        head_sha=initial_pr_context.metadata.head_sha,
        config=implementation_config,
        session_id=coder_response.session_id,
        invocation_id=coder_response.acquisition_test_turn_id,
        _closed_execution_catalog=coder_response.acquisition_test_observations,
        _journal_observations=coder_response.acquisition_test_observations,
        reauthenticate_head=lambda: get_pr_review_context(
            runner, config=implementation_config, pr_number=pr_number
        ).metadata.head_sha,
    )
    final_coverage_assessment = final_risk_coverage_assessment(
        implementation_result,
        approved_plan_context=approved_plan_context,
        workdir=active_workdir(implementation_config),
        initial_head_sha=initial_pr_context.metadata.head_sha,
        derived=_initial_derived_risk_evidence,
    )
    # Evidence was derived for this exact head (the raced head when semantic
    # correction observed a push), so the round record binds to it too.
    bound_round_head = (
        _initial_derived_risk_evidence.bound_head_sha
        if _initial_derived_risk_evidence is not None
        and _initial_derived_risk_evidence.bound_head_sha is not None
        else initial_pr_context.metadata.head_sha
    )
    initial_local_test_evidence = runner.render_local_test_evidence(
        current_head=initial_pr_context.metadata.head_sha,
        legacy_tests_run=implementation_result.tests_run,
        cwd=active_workdir(implementation_config),
    )
    initial_coder_body = _attach_round_metadata(
        render_public_agent_comment(
            kind="issue_implementation",
            parsed=implementation_result,
            agent=implementation_config.coder,
            config=implementation_config,
            model_used=coder_response.model_used,
            local_test_evidence=initial_local_test_evidence,
            current_test_turn_id=coder_response.acquisition_test_turn_id,
            coverage_assessment=final_coverage_assessment,
            coverage_reask_used=coverage_reask_used,
            coverage_discard_note=coverage_discard_note,
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent=coder_name,
            round_number=1,
            subject=str(bound_round_head or "unknown"),
            prior_items=(),
            raw_structured_coder_response=coder_output,
            local_test_evidence=initial_local_test_evidence,
            risk_test_matrix_evidence=(
                implementation_result.risk_test_matrix_evidence.to_payload()
                if implementation_result.risk_test_matrix_evidence is not None
                else None
            ),
            # The establishing comment renders the full row list (#959).
            risk_test_matrix_evidence_full_round=(
                1 if implementation_result.risk_test_matrix_evidence is not None else None
            ),
            risk_test_matrix_diagnostics=tuple(
                diagnostic.to_payload()
                for diagnostic in implementation_result.risk_test_matrix_diagnostics
            ),
            model_used=coder_response.model_used,
            **_metadata_identity_fields(coder_response),
            **_test_observation_degradation_fields(implementation_result),
            **_architecture_metadata_fields(
                implementation_config,
                result=implementation_result,
            ),
            acquisition_outcome=coder_response.acquisition_outcome,
            acquisition_returncode=coder_response.acquisition_returncode,
        ),
    )
    post_trusted_pr_comment(
        runner,
        config=implementation_config,
        pr_number=pr_number,
        body=_embed_pr_contract_marker(initial_coder_body, pr_contract),
    )
    return run_pr_loop(
        runner,
        pr_number=pr_number,
        config=implementation_config,
        coder_session_id=coder_response.session_id,
        issue_context=issue_context,
        approved_plan_context=approved_plan_context,
        parent_issue_context=parent_issue_context,
        workdirs_ready=True,
        usage_context=usage_context,
        pre_review_test_pending=True,
        managed_ci_handoff=managed_ci_handoff,
    )


def _decompose_approved_plan(
    runner: Runner,
    *,
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory,
    issue_context: IssueContext,
    mode: str,
    coder_session_id: str | None,
    usage_context: RunUsageContext,
    execution_recommendation=None,
    normalized_topology=None,
) -> StagedTopologyOutcome | NeedsHumanDecision:
    plan_hash = approved_plan_hash(approved_plan)
    plan_subject = _plan_subject(approved_plan)
    if execution_recommendation is not None and normalized_topology is None:
        normalized_topology = normalize_execution_recommendation(
            execution_recommendation,
            approved_plan=approved_plan,
            plan_subject=plan_subject,
        )
    if execution_recommendation is not None:
        parent_matrix_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan topology",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
        if parent_matrix_context.matrix_available:
            risk_matrix_payload = parent_matrix_context.risk_test_matrix_payload
            validate_risk_matrix_ownership(
                parent_matrix_context.risk_test_matrix_payload,
                execution_recommendation,
            )
        else:
            risk_matrix_payload = None
    else:
        risk_matrix_payload = None
    if normalized_topology is not None:
        decomposition, retained_parent_scope = normalized_topology
        reject_legacy_topology_collision(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
        )
        topology_source = EXECUTION_TOPOLOGY_SOURCE
        canonical_strategy = decomposition.strategy
        recommendation_digest = decomposition.recommendation_digest
        execution_contract_version = decomposition.execution_strategy_contract_version
        existing = find_existing_decomposition(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            strategy=canonical_strategy,
            topology_source=topology_source,
            recommendation_digest=recommendation_digest,
            plan_subject=plan_subject,
        )
    else:
        canonical_strategy = None
        recommendation_digest = None
        execution_contract_version = None
    existing = find_existing_decomposition(
        issue_context.comments,
        parent_issue=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    ) if normalized_topology is None else existing
    if existing is not None:
        log(config, f"Plan decomposition already exists for issue #{issue_number} ({mode}); not recreating children")
        adopted = tuple(
            CreatedPhaseIssue(
                phase=(
                    decomposition.phases[index]
                    if normalized_topology is not None
                    and index < len(decomposition.phases)
                    else RecordedPhase(title=title, automation=automation)
                ),
                issue_url=url,
                issue_number=number,
            )
            for index, ((title, url, number), automation) in enumerate(
                zip(existing.children, existing.automation, strict=False)
            )
        )
        # The recovered summary is the only source of stage identity and of
        # the parent's own obligations on the adopted and legacy paths, so it
        # is carried forward here instead of being discarded.
        return StagedTopologyOutcome(
            created=adopted,
            stage_ids=_resolved_stage_ids(
                adopted,
                normalized_topology=normalized_topology,
                recorded_stage_ids=existing.stage_ids,
            ),
            automations=tuple(item.phase.automation for item in adopted),
            plan_hash=plan_hash,
            mode=mode,
            topology_source=existing.topology_source,
            retained_parent_scope=(
                retained_parent_scope
                if normalized_topology is not None
                else existing.retained_parent_scope
            ),
            final_integration_work=(
                decomposition.final_integration_work
                if normalized_topology is not None
                else existing.final_integration_work
            ),
        )

    checkpoint = None
    if normalized_topology is None:
        checkpoint = find_existing_topology_checkpoint(
            issue_context.comments,
            parent_issue=issue_number,
            plan_hash=plan_hash,
            mode=mode,
        )
        retained_parent_scope = None
        topology_source = "model"
    if checkpoint is not None:
        # A checkpoint is the normalized model/typed output.  Reuse it before
        # invoking a coder so a create-before-summary failure is resumable.
        decomposition = PlanDecomposition(
            phases=checkpoint.phases,
            architecture_impact=(
                parse_architecture_impact(
                    # The checkpoint decoder restores lists as tuples; give the
                    # parser its JSON-array wire shape back.  Stored text keeps
                    # the explicit legacy decode, exactly as before #925.
                    sanitize_architecture_impact(checkpoint.architecture_impact),
                    context="checkpoint.architecture_impact",
                    architecture_status_mode="legacy",
                )
                if checkpoint.architecture_impact is not None else None
            ),
        )
        topology_source = checkpoint.topology_source
        retained_parent_scope = checkpoint.retained_parent_scope
    elif normalized_topology is None and mode == "decompose-only":
        typed_stages = _extract_current_child_stages(approved_plan)
        if typed_stages:
            decomposition, retained_parent_scope = adapt_typed_child_stages(
                typed_stages,
                approved_plan=approved_plan,
                plan_subject=_plan_subject(approved_plan),
            )
            topology_source = "typed"

    if normalized_topology is None and checkpoint is None and topology_source == "model":
        coder_name = agent_display_name(config.coder)
        log(config, f"Planning approved; invoking {coder_name} to decompose issue #{issue_number}")
        try:
            decomposition_response = _run_validated_agent(
                runner,
                agent=config.coder,
                config=config,
                prompt=build_plan_decomposition_prompt(
                    issue_number,
                    approved_plan,
                    config,
                    memory,
                    issue_context=issue_context,
                ),
                session_id=coder_session_id,
                marker_description="plan decomposition JSON",
                **_architecture_mode_validators(lambda mode: lambda text: parse_plan_decomposition(
                    text, required_architecture_impact_contract=1, architecture_status_mode=mode
                )),
                usage_context=usage_context,
                operation_description="plan decomposition",
                require_architecture_impact_contract=True,
            )
        except AgentInvocationError as exc:
            _surface_refused_decomposition(
                runner, config=config, issue_number=issue_number, error=exc
            )
            raise
        decomposition = decomposition_response.marker_value
        _surface_decomposition_degradations(
            runner, config=config, issue_number=issue_number, decomposition=decomposition
        )
    if topology_source == EXECUTION_TOPOLOGY_SOURCE and risk_matrix_payload is None:
        recovered_matrix_context = make_approved_plan_context(
            approved_plan,
            source_locator=f"issue #{issue_number} approved-plan topology",
            expected_hash=plan_hash,
            expected_subject=plan_subject,
        )
        if recovered_matrix_context.matrix_available:
            risk_matrix_payload = recovered_matrix_context.risk_test_matrix_payload
    created = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=issue_number,
        approved_plan=approved_plan,
        decomposition=decomposition,
        topology_source=topology_source,
        issue_comments=issue_context.comments,
        mode=mode,
        retained_parent_scope=retained_parent_scope,
        strategy=canonical_strategy,
        execution_strategy_contract_version=execution_contract_version,
        recommendation_digest=recommendation_digest,
        plan_subject=plan_subject,
        risk_test_matrix=risk_matrix_payload,
    )
    if isinstance(created, NeedsHumanDecision):
        return created
    summary_allocation_kwargs = (
        {"final_integration_work": decomposition.final_integration_work}
        if topology_source == EXECUTION_TOPOLOGY_SOURCE
        else {}
    )
    post_decomposition_parent_summary(
        runner,
        config=config,
        parent_issue=issue_number,
        mode=mode,
        plan_hash=plan_hash,
        created=created,
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
        strategy=canonical_strategy,
        execution_strategy_contract_version=execution_contract_version,
        recommendation_digest=recommendation_digest,
        plan_subject=plan_subject,
        **summary_allocation_kwargs,
    )
    return StagedTopologyOutcome(
        created=tuple(created),
        stage_ids=_resolved_stage_ids(created, normalized_topology=normalized_topology),
        automations=tuple(item.phase.automation for item in created),
        plan_hash=plan_hash,
        mode=mode,
        topology_source=topology_source,
        retained_parent_scope=retained_parent_scope,
        final_integration_work=(
            decomposition.final_integration_work
            if topology_source == EXECUTION_TOPOLOGY_SOURCE
            else None
        ),
    )
