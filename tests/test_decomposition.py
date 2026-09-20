import base64
import dataclasses
import json
import re

import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
import coding_review_agent_loop.orchestrator as orchestrator_module
from coding_review_agent_loop.config import DEFAULT_FLAT_CHILD_LIMIT
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    DecompositionMetadata,
    ExecutionDecision,
    ExecutionAllocation,
    ExecutionCouplingConstraint,
    ExecutionScopeItem,
    ExecutionStrategyRecommendation,
    PlanDecomposition,
    PlanPhase,
    RecordedPhase,
    RetainedParentScope,
    TopologyCheckpoint,
    PhaseImplementationHandoffMetadata,
    _decode_phase_implementation_handoff_metadata,
    _decode_metadata,
    _encode_phase_implementation_handoff_metadata,
    _encode_metadata,
    _fresh_phase_payload,
    _phase_from_payload,
    approved_plan_hash,
    child_disposition_override_digest,
    format_phase_issue_body,
    handoff_effective_disposition,
    find_existing_phase_implementation_handoff,
    find_existing_decomposition,
    find_existing_topology_checkpoint,
    format_decomposition_parent_summary,
    retained_parent_excerpt_section,
    retained_parent_scope_matches,
    format_execution_decision,
    format_phase_implementation_handoff_comment,
    format_topology_checkpoint,
    phase_identity,
    parse_plan_decomposition,
    create_decomposition_child_issues,
    adapt_typed_child_stages,
    normalize_execution_recommendation,
    find_existing_execution_decision,
    PHASE_IMPLEMENTATION_MARKER_RE,
)
from coding_review_agent_loop.issue_body_limits import shortened_section, shortening_notice
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
    ExecutionChildStage,
    validate_structured_plan_state,
)
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
import coding_review_agent_loop.phase_progress as phase_progress_module
from coding_review_agent_loop.phase_progress import (
    StagedTopologyOutcome,
    resolve_staged_phase_progress,
)
from coding_review_agent_loop.child_topology import NeedsHumanDecision
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
    _preflight_fresh_one_shot_recovery,
    _preflight_fresh_staged_topology,
    _dispatch_current_decomposition_phase,
)
from coding_review_agent_loop.split_materialization import (
    MaterializedSplitChild,
    SplitMaterializationMetadata,
    format_split_materialization_summary,
)
from agent_loop_helpers import (
    FakeRunner,
    approved_plan_comments,
    child_pr_handoff_comment,
    make_config,
    phase_handoff_comment,
    pr_payload_for_state,
    staged_legacy_plan_records,
    staged_v1_recorded_plan_records,
    plan_decomposition_json,
    structured_plan_review,
    structured_plan_state,
    structured_v1_plan_state,
)


def test_parse_plan_decomposition_accepts_agent_and_human_phases():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["Internal schema utilities"],
            },
        )
    )

    assert [phase.title for phase in parsed.phases] == [
        "Internal schema utilities",
        "Manual rollout checkpoint",
    ]
    assert parsed.phases[1].automation == "human-action"
    assert parsed.phases[1].depends_on == ("Internal schema utilities",)


def test_legacy_typed_adapter_rejects_reviewed_generation_one_children():
    stage = ExecutionChildStage(
        stage_id="stage-1",
        position=1,
        title="Reviewed stage",
        summary="Audit-only recommendation stage.",
        deliverables=("A deliverable.",),
        non_goals=(),
        acceptance_criteria=("The stage is complete.",),
        depends_on_stage_ids=(),
        dependency_notes="No dependencies.",
        automation="agent-pr",
        rollout_risk="low",
        compatibility_constraints=(),
        covered_scope_item_ids=("scope-1",),
    )

    with pytest.raises(AgentLoopError, match="cannot reach the legacy typed-stage adapter"):
        adapt_typed_child_stages(
            (stage,), approved_plan="Approved plan.", plan_subject="subject"
        )


def test_fresh_recommendation_normalizer_preserves_reviewed_allocation_and_identity():
    recommendation = ExecutionStrategyRecommendation(
        strategy="staged",
        rationale="The API and its compatibility tests have a safe boundary.",
        staging_feasibility="safe",
        scope_items=(
            ExecutionScopeItem("scope-api", "Preserve the API.", ("Callers still work.",)),
            ExecutionScopeItem("scope-tests", "Cover the behavior.", ("The focused test passes.",)),
        ),
        coupling_constraints=(),
        one_shot_delivery=None,
        child_stages=(
            ExecutionChildStage(
                stage_id="stage-api",
                position=1,
                title="API change",
                summary="Implement the compatibility-preserving API change.",
                deliverables=("API implementation.",),
                non_goals=("No rollout.",),
                acceptance_criteria=("The API remains compatible.",),
                depends_on_stage_ids=(),
                dependency_notes="This is the first stage.",
                automation="agent-pr",
                rollout_risk="low",
                compatibility_constraints=("Preserve existing callers.",),
                covered_scope_item_ids=("scope-api",),
            ),
        ),
        retained_parent_work=ExecutionAllocation("none", (), (), ()),
        final_integration_work=ExecutionAllocation(
            "required", ("Compatibility tests." ,), ("The focused test passes.",), ("scope-tests",)
        ),
        caveats=("Run the focused suite.",),
    )
    plan = "Approved fresh plan."
    decomposition, retained = normalize_execution_recommendation(
        recommendation,
        approved_plan=plan,
        plan_subject="fresh-subject",
    )

    phase = decomposition.phases[0]
    assert decomposition.strategy == "staged"
    assert decomposition.topology_source == "approved-plan-v1"
    assert phase.stage_id == "stage-api"
    assert phase.position == 1
    assert phase.deliverables == ("API implementation.",)
    assert phase.non_goals_items == ("No rollout.",)
    assert phase.acceptance_criteria == ("The API remains compatible.",)
    assert phase.compatibility_constraints == ("Preserve existing callers.",)
    assert phase.covered_scope_item_ids == ("scope-api",)
    assert retained.status == "none"
    assert retained.deliverables == ()
    assert retained.covered_scope_item_ids == ()


def test_legacy_phase_identity_digest_remains_byte_stable():
    phase = PlanPhase(
        title="Legacy phase",
        scope="Keep old behavior.",
        non_goals="No new path.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="Run tests.",
        parent_context="Approved context.",
        automation="agent-pr",
        depends_on=(),
    )
    assert phase_identity(
        parent_issue=56,
        plan_hash="0123456789abcdef",
        topology_source="model",
        phase_index=1,
        phase=phase,
    ) == "445715b616bf8e71572a804c6172dc6645d2ed97d040c0be5110a60c115403fd"


def _fresh_recovery_topology(plan: str = "Approved fresh staged plan"):
    first = PlanPhase(
        title="API contract",
        scope="Implement the reviewed API contract.",
        non_goals="No rollout.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="The API contract tests pass.",
        parent_context=plan,
        automation="agent-pr",
        stage_id="stage-api",
        position=1,
        deliverables=("API contract implementation.",),
        non_goals_items=("No rollout.",),
        acceptance_criteria=("The API contract tests pass.",),
        depends_on_stage_ids=(),
        compatibility_constraints=("Keep existing callers working.",),
        covered_scope_item_ids=("scope-api",),
    )
    second = PlanPhase(
        title="Integration verification",
        scope="Verify the integrated behavior.",
        non_goals="No new API surface.",
        dependency_notes="After the API contract phase.",
        rollout_risk="medium.",
        validation="The integration tests pass.",
        parent_context=plan,
        automation="human-action",
        depends_on=("API contract",),
        stage_id="stage-integration",
        position=2,
        deliverables=("Integration verification record.",),
        non_goals_items=("No new API surface.",),
        acceptance_criteria=("The integration tests pass.",),
        depends_on_stage_ids=("stage-api",),
        compatibility_constraints=(),
        covered_scope_item_ids=("scope-integration",),
    )
    return PlanDecomposition(
        phases=(first, second),
        strategy="staged",
        topology_source="approved-plan-v1",
        execution_strategy_contract_version=1,
        recommendation_digest="recommendation-digest",
        scope_items=(
            ExecutionScopeItem("scope-api", "Implement the API.", ("The API contract tests pass.",)),
            ExecutionScopeItem(
                "scope-integration", "Verify integration.", ("The integration tests pass.",)
            ),
        ),
        retained_parent_work=ExecutionAllocation("none", (), (), ()),
        final_integration_work=ExecutionAllocation("none", (), (), ()),
    )


def _fresh_child_issue(phase: PlanPhase, plan: str, issue_number: int) -> dict[str, object]:
    plan_hash = approved_plan_hash(plan)
    identity = phase_identity(
        parent_issue=56,
        plan_hash=plan_hash,
        topology_source="approved-plan-v1",
        phase_index=phase.position or 0,
        phase=phase,
        stage_id=phase.stage_id,
        execution_strategy_contract_version=1,
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phase,
        created_so_far=(),
        phase_identity_value=identity,
        topology_source="approved-plan-v1",
        phase_index=phase.position or 0,
        phase_plan_hash=plan_hash,
        strategy="staged",
        recommendation_digest="recommendation-digest",
        execution_strategy_contract_version=1,
    )
    return {
        "number": issue_number,
        "title": _phase_issue_title_for_test(56, phase),
        "url": f"https://github.com/OWNER/REPO/issues/{issue_number}",
        "body": body,
    }


def _phase_issue_title_for_test(parent_issue: int, phase: PlanPhase) -> str:
    prefix = "[Human] " if phase.automation in {"human-action", "manual-close"} else ""
    return f"{prefix}Phase {phase.position}: {phase.title} (from #{parent_issue})"


def test_fresh_child_recovery_adopts_each_matching_stage_without_unbound_phase(tmp_path):
    plan = "Approved fresh staged plan"
    topology = _fresh_recovery_topology(plan)
    candidates = [
        _fresh_child_issue(phase, plan, 100 + index)
        for index, phase in enumerate(topology.phases, start=1)
    ]
    runner = FakeRunner(search_issues_payload=candidates)

    recovered = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan=plan,
        decomposition=topology,
        topology_source="approved-plan-v1",
        issue_comments=(),
        mode="implement-by-phase",
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="recommendation-digest",
        preflight_only=True,
    )

    assert [item.origin for item in recovered] == ["adopted", "adopted"]
    assert [item.issue_number for item in recovered] == [101, 102]


def test_fresh_child_recovery_adopts_partial_topology_and_creates_remainder(tmp_path):
    plan = "Approved fresh staged plan"
    topology = _fresh_recovery_topology(plan)
    runner = FakeRunner(
        search_issues_payload=[_fresh_child_issue(topology.phases[0], plan, 101)],
        issue_urls=["https://github.com/OWNER/REPO/issues/102"],
    )

    recovered = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan=plan,
        decomposition=topology,
        topology_source="approved-plan-v1",
        issue_comments=(),
        mode="implement-by-phase",
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="recommendation-digest",
    )

    assert [item.origin for item in recovered] == ["adopted", "created"]
    assert [item.issue_number for item in recovered] == [101, 102]
    assert len(runner.issues) == 1


def test_execution_decision_format_is_reused_for_same_identity():
    from types import SimpleNamespace

    decision = ExecutionDecision(
        parent_issue=56,
        plan_hash="0123456789abcdef",
        plan_subject="subject",
        execution_strategy_contract_version=1,
        strategy="one-shot",
        topology_source="approved-plan-v1",
        recommendation_digest="digest",
        requested_policy="implement-one-shot",
        current_action="implement-one-shot",
        scope_item_ids=("scope-1",),
    )
    body = format_execution_decision(decision)
    comments = (SimpleNamespace(body=body),)

    recovered = find_existing_execution_decision(
        comments,
        parent_issue=56,
        plan_hash="0123456789abcdef",
        plan_subject="subject",
        strategy="one-shot",
        recommendation_digest="digest",
    )

    assert recovered == decision


def test_execution_decision_recovery_rejects_a_changed_parent_plan():
    from types import SimpleNamespace

    decision = ExecutionDecision(
        parent_issue=56,
        plan_hash="old-plan-hash",
        plan_subject="old-subject",
        execution_strategy_contract_version=1,
        strategy="one-shot",
        topology_source="approved-plan-v1",
        recommendation_digest="old-digest",
        requested_policy="implement-one-shot",
        current_action="implement-one-shot",
    )

    with pytest.raises(AgentLoopError, match="different approved plan hash"):
        find_existing_execution_decision(
            (SimpleNamespace(body=format_execution_decision(decision)),),
            parent_issue=56,
            plan_hash="new-plan-hash",
            plan_subject="new-subject",
            strategy="one-shot",
            recommendation_digest="new-digest",
        )


def _parent_recovery_record(kind: str) -> str:
    old_plan_hash = "old-plan-hash"
    phase = _phase("Recorded stage")
    created = CreatedPhaseIssue(
        phase=phase,
        issue_url="https://github.com/OWNER/REPO/issues/101",
        issue_number=101,
    )
    if kind == "summary":
        return format_decomposition_parent_summary(
            parent_issue=56,
            mode="decompose-only",
            plan_hash=old_plan_hash,
            created=(created,),
            topology_source="model",
        )
    if kind == "checkpoint":
        return format_topology_checkpoint(
            TopologyCheckpoint(
                parent_issue=56,
                plan_hash=old_plan_hash,
                mode="decompose-only",
                topology_source="model",
                phases=(phase,),
            )
        )
    if kind == "phase-handoff":
        return format_phase_implementation_handoff_comment(
            parent_issue=56,
            mode="implement-by-phase",
            plan_hash=old_plan_hash,
            phase_index=1,
            created=created,
        )
    raise AssertionError(f"unknown recovery record kind: {kind}")


@pytest.mark.parametrize("record_kind", ["summary", "checkpoint", "phase-handoff"])
def test_fresh_staged_preflight_inventories_record_only_parent_state_before_writes(
    tmp_path, record_kind
):
    plan = "Approved fresh staged plan"
    topology = _fresh_recovery_topology(plan)
    context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(IssueComment(author="bot", created_at=None, body=_parent_recovery_record(record_kind)),),
    )
    runner = FakeRunner()

    with pytest.raises(AgentLoopError, match="recorded topology|handoff|record-only"):
        _preflight_fresh_staged_topology(
            runner,
            issue_number=56,
            approved_plan=plan,
            config=make_config(tmp_path),
            issue_context=context,
            mode="decompose-only",
            normalized_topology=(topology, RetainedParentScope(
                plan_subject=_plan_subject(plan), plan_hash=approved_plan_hash(plan), excerpt=plan,
                status="none",
            )),
        )

    assert runner.comments == []
    assert runner.issues == []


@pytest.mark.parametrize("record_kind", ["summary", "checkpoint", "phase-handoff"])
def test_fresh_one_shot_preflight_inventories_record_only_parent_state_before_writes(
    tmp_path, record_kind
):
    plan = structured_v1_plan_state()
    recommendation = validate_structured_plan_state(plan).execution_recommendation
    assert recommendation is not None
    context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(IssueComment(author="bot", created_at=None, body=_parent_recovery_record(record_kind)),),
    )
    runner = FakeRunner()

    with pytest.raises(AgentLoopError, match="existing|recorded topology"):
        _preflight_fresh_one_shot_recovery(
            runner,
            issue_number=56,
            approved_plan=plan,
            config=make_config(tmp_path),
            issue_context=context,
            recommendation=recommendation,
        )

    assert runner.comments == []
    assert runner.issues == []


@pytest.mark.parametrize("split_state", ["materialized", "orphan-child"])
def test_fresh_one_shot_preflight_rejects_existing_split_state_before_writes(
    tmp_path, split_state
):
    plan = structured_v1_plan_state()
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    recommendation = validate_structured_plan_state(plan).execution_recommendation
    assert recommendation is not None
    if split_state == "materialized":
        split_body = format_split_materialization_summary(
            parent_issue=56,
            metadata=SplitMaterializationMetadata(
                parent_issue=56,
                subject="split subject",
                children=(
                    MaterializedSplitChild(
                        title="Existing split stage",
                        key="a" * 64,
                        url="https://github.com/OWNER/REPO/issues/99",
                        number=99,
                        origin="created",
                    ),
                ),
            ),
        )
        comments = (IssueComment(author="bot", created_at=None, body=split_body),)
        runner = FakeRunner()
    else:
        comments = ()
        runner = FakeRunner(
            search_issues_payload=[
                [
                    {
                        "number": 99,
                        "title": "[#56 stage] Existing split stage",
                        "url": "https://github.com/OWNER/REPO/issues/99",
                        "body": "",
                    }
                ]
            ]
        )
    context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=comments,
    )

    with pytest.raises(AgentLoopError, match="existing split"):
        _preflight_fresh_one_shot_recovery(
            runner,
            issue_number=56,
            approved_plan=plan,
            config=make_config(tmp_path),
            issue_context=context,
            recommendation=recommendation,
        )

    assert runner.comments == []
    assert runner.issues == []


def test_fresh_parent_summary_round_trips_final_integration_obligation():
    phase = PlanPhase(
        title="First stage",
        scope="Deliver the first stage.",
        non_goals="No final integration.",
        dependency_notes="No dependencies.",
        rollout_risk="low",
        validation="The first stage passes.",
        parent_context="Approved parent plan.",
        automation="agent-pr",
        stage_id="stage-1",
        position=1,
        deliverables=("First stage.",),
        non_goals_items=("No final integration.",),
        acceptance_criteria=("The first stage passes.",),
        covered_scope_item_ids=("scope-1",),
    )
    final = ExecutionAllocation(
        "required",
        ("Run the final integration verification.",),
        ("The integrated behavior passes.",),
        ("scope-final",),
    )
    body = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash="a" * 16,
        created=(CreatedPhaseIssue(phase=phase, issue_url="https://example/issues/99", issue_number=99),),
        topology_source="approved-plan-v1",
        retained_parent_scope=RetainedParentScope(
            plan_subject="b" * 64,
            plan_hash="a" * 16,
            excerpt="Approved parent plan.",
            status="none",
        ),
        final_integration_work=final,
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="c" * 64,
        plan_subject="b" * 64,
    )
    restored = find_existing_decomposition(
        [IssueComment(author="bot", created_at=None, body=body)],
        parent_issue=56,
        plan_hash="a" * 16,
    )
    assert restored is not None
    assert restored.final_integration_work == final
    assert "## Final integration work" in body
    assert "scope-final" in body


def test_fresh_plan_decomposition_requires_architecture_impact():
    payload = plan_decomposition_json(
        {
            "title": "Internal schema utilities",
            "scope": "Add helpers.",
            "non_goals": "No live switch.",
            "dependency_notes": "First phase.",
            "rollout_risk": "low - internal only.",
            "validation": "Run python -m pytest.",
            "parent_context": "Approved plan slice and invariant details.",
            "automation": "agent-pr",
            "depends_on": [],
        }
    )
    payload_dict = json.loads(payload)
    payload_dict.pop("architecture_impact")
    payload_without_impact = json.dumps(payload_dict)
    with pytest.raises(AgentLoopError, match="architecture_impact"):
        parse_plan_decomposition(
            payload_without_impact, required_architecture_impact_contract=1
        )

def test_parse_plan_decomposition_accepts_normalized_earlier_phase_dependency():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["  internal   SCHEMA utilities  "],
            },
        )
    )

    assert parsed.phases[1].depends_on == ("internal   SCHEMA utilities",)

def test_parse_plan_decomposition_rejects_self_dependency():
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Internal schema utilities"],
    }

    with pytest.raises(AgentLoopError, match="cannot depend on itself"):
        parse_plan_decomposition(plan_decomposition_json(phase))

def test_parse_plan_decomposition_rejects_forward_dependency():
    first_phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Manual rollout checkpoint"],
    }
    second_phase = {
        "title": "Manual rollout checkpoint",
        "scope": "Human validates the deployed behavior.",
        "non_goals": "No code changes.",
        "dependency_notes": "After Internal schema utilities.",
        "rollout_risk": "medium - live checkpoint.",
        "validation": "Human remark and closure required.",
        "parent_context": "Approved plan slice for the manual checkpoint.",
        "automation": "human-action",
        "depends_on": [],
    }

    with pytest.raises(AgentLoopError, match="dependencies must reference an earlier phase"):
        parse_plan_decomposition(plan_decomposition_json(first_phase, second_phase))

@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda phase: phase.pop("parent_context"), "parent_context"),
        (lambda phase: phase.pop("rollout_risk"), "rollout_risk"),
        (lambda phase: phase.pop("validation"), "validation"),
        (lambda phase: phase.__setitem__("automation", "robot"), "invalid automation"),
        (lambda phase: phase.__setitem__("depends_on", ["Missing phase"]), "unknown phase"),
    ],
)
def test_parse_plan_decomposition_rejects_invalid_phase_fields(mutate, message):
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    mutate(phase)

    with pytest.raises(AgentLoopError, match=message):
        parse_plan_decomposition(plan_decomposition_json(phase))

def test_parse_plan_decomposition_rejects_duplicates_but_leaves_cap_to_preflight():
    phase = {
        "title": "Repeated phase",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    with pytest.raises(AgentLoopError, match="duplicate phase title"):
        parse_plan_decomposition(plan_decomposition_json(phase, dict(phase)))

    phases = [dict(phase, title=f"Phase {index}") for index in range(DEFAULT_FLAT_CHILD_LIMIT + 1)]
    parsed = parse_plan_decomposition(plan_decomposition_json(*phases))
    assert len(parsed.phases) == DEFAULT_FLAT_CHILD_LIMIT + 1


def _phase(title: str, *, depends_on: tuple[str, ...] = ()) -> PlanPhase:
    return PlanPhase(
        title=title,
        scope=f"Implement {title}.",
        non_goals="No unrelated changes.",
        dependency_notes="Follow the parent plan.",
        rollout_risk="low.",
        validation="Run focused tests.",
        parent_context="Approved parent constraints.",
        automation="agent-pr",
        depends_on=depends_on,
    )


def test_format_phase_body_points_to_parent_for_complete_constraints():
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan="The complete approved plan.",
        phase=_phase("Model-selected phase"),
        created_so_far=(),
    )

    assert "The linked parent issue is the source of truth" in body
    assert "complete historical constraint context" not in body


def test_child_disposition_persistence_is_optional_and_legacy_stable():
    legacy = dataclasses.replace(_phase("Disposition phase"), stage_id="stage-one", position=1)
    legacy_payload = _fresh_phase_payload(legacy)
    assert legacy_payload == {
        "stage_id": "stage-one",
        "position": 1,
        "title": "Disposition phase",
        "summary": "Implement Disposition phase.",
        "deliverables": [],
        "non_goals": [],
        "acceptance_criteria": [],
        "depends_on_stage_ids": [],
        "dependency_notes": "Follow the parent plan.",
        "automation": "agent-pr",
        "rollout_risk": "low.",
        "compatibility_constraints": [],
        "covered_scope_item_ids": [],
        "parent_context": "Approved parent constraints.",
    }
    restored_legacy = _phase_from_payload(legacy_payload, fresh=True)
    assert restored_legacy.execution_disposition is None
    assert _fresh_phase_payload(restored_legacy) == legacy_payload
    # Pin the pre-disposition phase identity: the additive field must not
    # perturb already-recorded checkpoints that do not carry it.
    legacy_identity = phase_identity(
        parent_issue=55,
        plan_hash="parent-plan",
        topology_source="approved-plan-v1",
        phase_index=1,
        phase=legacy,
        stage_id="stage-one",
        execution_strategy_contract_version=1,
    )
    assert legacy_identity == "b4ccf9463522f008bfe7a70758f66710fd1491af8141f7bfc18196237aa3dd03"
    assert legacy_identity == phase_identity(
        parent_issue=55,
        plan_hash="parent-plan",
        topology_source="approved-plan-v1",
        phase_index=1,
        phase=dataclasses.replace(legacy, execution_disposition=None),
        stage_id="stage-one",
        execution_strategy_contract_version=1,
    )

    planning = dataclasses.replace(
        legacy,
        execution_disposition=EXECUTION_DISPOSITION_PLANNING,
        disposition_rationale="Resolve the remaining design choice first.",
        unresolved_design_decisions=("Choose the storage representation.",),
    )
    planning_payload = _fresh_phase_payload(planning)
    assert planning_payload["execution_disposition"] == {
        "disposition": EXECUTION_DISPOSITION_PLANNING,
        "rationale": planning.disposition_rationale,
        "unresolved_design_decisions": list(planning.unresolved_design_decisions),
    }
    restored_planning = _phase_from_payload(planning_payload, fresh=True)
    assert restored_planning.execution_disposition == EXECUTION_DISPOSITION_PLANNING
    assert restored_planning.disposition_rationale == planning.disposition_rationale
    assert restored_planning.unresolved_design_decisions == planning.unresolved_design_decisions
    assert _fresh_phase_payload(restored_planning) == planning_payload

    decomposition_metadata = DecompositionMetadata(
        parent_issue=55,
        plan_hash="parent-plan",
        mode="implement-by-phase",
        phase_count=2,
        phase_titles=("Direct child", "Planning child"),
        automation=("agent-pr", "agent-pr"),
        children=(("Direct child", "direct-url", 56), ("Planning child", "plan-url", 57)),
        topology_source="approved-plan-v1",
        final_integration_work=ExecutionAllocation("none", (), (), ()),
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="r" * 64,
        plan_subject="s" * 64,
        stage_ids=("stage-one", "stage-two"),
        phase_identities=("identity-one", "identity-two"),
        dispositions=(EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING),
    )
    assert _decode_metadata(_encode_metadata(decomposition_metadata)) == decomposition_metadata
    legacy_metadata = dataclasses.replace(decomposition_metadata, dispositions=())
    encoded_legacy_metadata = _encode_metadata(legacy_metadata)
    assert "dispositions" not in json.loads(
        base64.urlsafe_b64decode(encoded_legacy_metadata).decode("utf-8")
    )
    assert _decode_metadata(encoded_legacy_metadata) == legacy_metadata

    metadata = PhaseImplementationHandoffMetadata(
        parent_issue=55,
        plan_hash="parent-plan",
        mode="implement-by-phase",
        phase_index=1,
        phase_title=legacy.title,
        automation="agent-pr",
        child_issue_number=56,
        child_issue_url="https://github.com/OWNER/REPO/issues/56",
        strategy="staged",
        topology_source="approved-plan-v1",
        execution_strategy_contract_version=1,
        recommendation_digest="r" * 64,
        stage_id="stage-one",
        plan_subject="s" * 64,
        execution_disposition=EXECUTION_DISPOSITION_PLANNING,
        override_digest="d" * 64,
    )
    assert _decode_phase_implementation_handoff_metadata(
        _encode_phase_implementation_handoff_metadata(metadata)
    ) == metadata
    legacy_handoff = dataclasses.replace(
        metadata, execution_disposition=None, override_digest=None
    )
    decoded_legacy = _decode_phase_implementation_handoff_metadata(
        _encode_phase_implementation_handoff_metadata(legacy_handoff)
    )
    assert decoded_legacy == legacy_handoff
    assert handoff_effective_disposition(decoded_legacy) == EXECUTION_DISPOSITION_DIRECT


def test_child_disposition_digest_and_issue_instructions_are_canonical():
    payload = {
        "kind": "child-execution-disposition-override",
        "schema_version": 1,
        "parent_issue": 55,
        "plan_hash": "parent-plan",
        "stage_id": "stage-one",
        "disposition": EXECUTION_DISPOSITION_PLANNING,
        "rationale": "Require reviewed design decisions.",
    }
    assert child_disposition_override_digest(payload) == child_disposition_override_digest(
        dict(reversed(tuple(payload.items())))
    )
    direct = dataclasses.replace(
        _phase("Direct child"), execution_disposition=EXECUTION_DISPOSITION_DIRECT
    )
    planning = dataclasses.replace(
        _phase("Planning child"), execution_disposition=EXECUTION_DISPOSITION_PLANNING
    )
    direct_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan="Parent plan",
        phase=direct, created_so_far=(),
    )
    planning_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan="Parent plan",
        phase=planning, created_so_far=(),
    )
    assert "implementation-ready" in direct_body
    assert "do not run it with `--plan-first`" in direct_body
    assert "requires its own reviewed plan" in planning_body
    assert "--plan-first --plan-execution-mode auto" in planning_body


def test_topology_checkpoint_stores_shared_context_once_and_round_trips(tmp_path):
    excerpt = "Approved parent constraints.\n" + ("constraint detail\n" * 500)
    phases = tuple(
        PlanPhase(
            title=f"Stage {index}",
            scope=f"Implement stage {index}.",
            non_goals="No unrelated changes.",
            dependency_notes="Follow the parent plan.",
            rollout_risk="low.",
            validation="Run focused tests.",
            parent_context=excerpt,
            automation="agent-pr",
            depends_on=(),
        )
        for index in range(13)
    )
    checkpoint = TopologyCheckpoint(
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
        topology_source="typed",
        phases=phases,
        retained_parent_scope=RetainedParentScope(
            plan_subject="Primary scope",
            plan_hash="plan-hash",
            excerpt=excerpt,
        ),
        architecture_identity={"repository": "OWNER/REPO", "revision": "abc"},
        architecture_impact={
            "status": "unchanged", "rationale": "No architectural contract changed.",
        },
        architecture_contract_version=1,
    )

    body = format_topology_checkpoint(checkpoint)
    restored = find_existing_topology_checkpoint(
        (IssueComment(author="bot", created_at=None, body=body),),
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
    )

    assert len(body) < 60000
    assert "constraint detail" not in body
    assert restored is not None
    assert restored.phases == checkpoint.phases
    assert restored.architecture_identity == checkpoint.architecture_identity
    assert restored.architecture_impact is not None
    assert restored.architecture_impact["status"] == "unchanged"
    assert restored.architecture_contract_version == 1


def test_topology_checkpoint_sanitizes_agent_impact_before_serialization():
    checkpoint = TopologyCheckpoint(
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
        topology_source="model",
        phases=(_phase("Stage"),),
        architecture_impact={
            "status": "unchanged",
            "rationale": "The contract changed. <!-- AGENT_LOOP_SIDECAR: eyJ4IjoxfQ== -->",
            "affected_components": ["worker <!-- AGENT_PLAN_PHASE_IDENTITY: eyJ4IjoxfQ== -->"],
            "canonical_document_path": "docs/ARCHITECTURE.md",
        },
        architecture_contract_version=1,
    )

    body = format_topology_checkpoint(checkpoint)
    assert "<!-- AGENT_LOOP_SIDECAR:" not in body
    assert "<!-- AGENT_PLAN_PHASE_IDENTITY:" not in body
    restored = find_existing_topology_checkpoint(
        (IssueComment(author="bot", created_at=None, body=body),),
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
    )
    assert restored is not None
    assert "AGENT_LOOP_SIDECAR" not in str(restored.architecture_impact)
    assert "AGENT_PLAN_PHASE_IDENTITY" not in str(restored.architecture_impact)


def test_dry_run_decomposition_previews_dependency_phases_without_issue_numbers(tmp_path):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "First phase",
                "scope": "First.",
                "non_goals": "None.",
                "dependency_notes": "First.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Second phase",
                "scope": "Second.",
                "non_goals": "None.",
                "dependency_notes": "After first.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": ["First phase"],
            },
        )
    )
    runner = FakeRunner()

    created = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path, dry_run=True),
        parent_issue=56,
        approved_plan="approved plan",
        decomposition=phases,
    )

    assert len(created) == 2
    assert all(item.issue_url is None and item.issue_number is None for item in created)
    assert len(runner.issues) == 2
    assert runner.comments == []


def test_decomposition_preflight_counts_split_children_toward_shared_limit(tmp_path):
    existing_children = [
        {
            "number": 100 + index,
            "title": f"[#56 stage] Existing {index}",
            "url": f"https://github.com/OWNER/REPO/issues/{100 + index}",
            "body": f"Part of #56\n<!-- AGENT_SPLIT_CHILD: parent=56 key={index + 1:064x} -->",
        }
        for index in range(DEFAULT_FLAT_CHILD_LIMIT)
    ]
    runner = FakeRunner(search_issues_payload=existing_children)

    decision = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan="approved plan",
        decomposition=PlanDecomposition(phases=(_phase("New phase"),)),
    )

    assert isinstance(decision, NeedsHumanDecision)
    assert decision.recognized_existing_count == DEFAULT_FLAT_CHILD_LIMIT
    assert decision.projected_total == DEFAULT_FLAT_CHILD_LIMIT + 1
    assert runner.issues == []
    assert runner.comments == []


def test_decomposition_adopts_closed_exact_identity(tmp_path):
    phase = _phase("Schema helpers")
    plan = "approved plan"
    identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phase,
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phase,
        created_so_far=(),
        phase_identity_value=identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner = FakeRunner(
        search_issues_payload=[
            {
                "number": 101,
                "title": "Phase 1: Schema helpers (from #56)",
                "url": "https://github.com/OWNER/REPO/issues/101",
                "body": body,
                "state": "closed",
            }
        ]
    )

    created = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan=plan,
        decomposition=PlanDecomposition(phases=(phase,)),
    )

    assert created[0].origin == "adopted"
    assert created[0].issue_number == 101
    assert runner.issues == []


def test_decomposition_rejects_ambiguous_exact_identity_before_checkpoint(tmp_path):
    phase = _phase("Schema helpers")
    plan = "approved plan"
    identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phase,
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phase,
        created_so_far=(),
        phase_identity_value=identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner = FakeRunner(search_issues_payload=[
        {"number": 101, "title": "Phase 1: Schema helpers (from #56)", "url": "u101", "body": body},
        {"number": 102, "title": "Phase 1: Schema helpers (from #56)", "url": "u102", "body": body},
    ])

    with pytest.raises(AgentLoopError, match="Ambiguous decomposition recovery"):
        create_decomposition_child_issues(
            runner,
            config=make_config(tmp_path),
            parent_issue=56,
            approved_plan=plan,
            decomposition=PlanDecomposition(phases=(phase,)),
        )
    assert runner.issues == []
    assert runner.comments == []


def test_decomposition_partial_create_recovers_from_checkpoint_and_identity(tmp_path, monkeypatch):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "First phase",
                "scope": "First.",
                "non_goals": "None.",
                "dependency_notes": "First.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Second phase",
                "scope": "Second.",
                "non_goals": "None.",
                "dependency_notes": "Second.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
        )
    )
    plan = "approved plan"
    runner = FakeRunner(
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
        search_issues_payload=[[]],
    )
    config = make_config(tmp_path)
    original_create = __import__(
        "coding_review_agent_loop.decomposition", fromlist=["create_issue"]
    ).create_issue
    calls = {"count": 0}

    def fail_after_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise AgentLoopError("simulated create failure")
        return original_create(*args, **kwargs)

    import coding_review_agent_loop.decomposition as decomp

    monkeypatch.setattr(decomp, "create_issue", fail_after_first)
    with pytest.raises(AgentLoopError, match="simulated create failure"):
        create_decomposition_child_issues(
            runner,
            config=config,
            parent_issue=56,
            approved_plan=plan,
            decomposition=phases,
        )
    assert len(runner.issues) == 1
    checkpoint_comments = tuple(
        IssueComment(author="bot", created_at=None, body=comment["body"])
        for comment in runner.issue_comments
    )
    first_identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phases.phases[0],
    )
    first_body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phases.phases[0],
        created_so_far=(),
        phase_identity_value=first_identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner.search_issues_payload = [{
        "number": 101,
        "title": "Phase 1: First phase (from #56)",
        "url": "https://github.com/OWNER/REPO/issues/101",
        "body": first_body,
    }]
    monkeypatch.setattr(decomp, "create_issue", original_create)

    resumed = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=56,
        approved_plan=plan,
        decomposition=phases,
        issue_comments=checkpoint_comments,
    )

    assert [item.origin for item in resumed] == ["adopted", "created"]
    assert [item.issue_number for item in resumed] == [101, 102]
    assert len(runner.issues) == 2
    assert sum("Topology checkpoint recorded" in comment for comment in runner.comments) == 1


def test_typed_decompose_only_materializes_one_thirteen_stage_topology(tmp_path):
    stages = [
        {"title": f"Backend stage {index}", "summary": f"Backend work {index}."}
        for index in range(8)
    ] + [
        {"title": f"Frontend stage {index}", "summary": f"Frontend work {index}."}
        for index in range(5)
    ]
    runner = FakeRunner(
        claude_outputs=[structured_plan_state(summary="Primary approved scope", child_stages=stages)],
        codex_outputs=[structured_plan_review(state="approved")],
        issue_urls=[f"https://github.com/OWNER/REPO/issues/{100 + index}" for index in range(13)],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert len(runner.issues) == 13
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1
    assert not any("AGENT_TYPED_PLAN_STAGES" in issue["body"] for issue in runner.issues)
    assert "Retained parent scope" in runner.comments[-1]


def test_decomposition_preflights_last_unsafe_phase_before_any_write(tmp_path):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Safe phase",
                "scope": "Safe.",
                "non_goals": "None.",
                "dependency_notes": "None.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Unsafe phase",
                "scope": "Contains <!-- AGENT_TYPED_PLAN_STAGES: historical -->",
                "non_goals": "None.",
                "dependency_notes": "None.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
        )
    )
    runner = FakeRunner(issue_urls=["https://github.com/OWNER/REPO/issues/101"])
    with pytest.raises(AgentLoopError, match="marker set mismatch"):
        create_decomposition_child_issues(
            runner,
            config=make_config(tmp_path),
            parent_issue=56,
            approved_plan="approved plan",
            decomposition=phases,
        )
    assert runner.issues == []
    assert runner.comments == []

def test_issue_loop_plan_first_decompose_only_summarizes_instead_of_filing_plan_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Split the implementation into phases."),
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(
        tmp_path,
        approved_followups="issue",
        plan_execution_mode="decompose-only",
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert not any(
        issue["title"].startswith("Follow up future plan-review note:")
        for issue in runner.issues
    )
    planning_summary = runner.comments[2]
    assert planning_summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in planning_summary
    assert "Filed future follow-up issues:" not in planning_summary
    assert "mode=summarize" in planning_summary
    assert "mode=issue" not in planning_summary

@pytest.mark.parametrize("additional_closing_ids", [None, []])
def test_issue_loop_plan_first_decompose_only_creates_child_issues(tmp_path, additional_closing_ids):
    plan = structured_plan_state(summary="Add schema helpers.")
    if additional_closing_ids is not None:
        payload, end = json.JSONDecoder().raw_decode(plan)
        payload["additional_closing_issue_ids"] = additional_closing_ids
        plan = json.dumps(payload) + plan[end:]
    runner = FakeRunner(
        claude_outputs=[
            plan,
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                },
                {
                    "title": "Human rollout checkpoint",
                    "scope": "Human validates rollout readiness.",
                    "non_goals": "No code changes.",
                    "dependency_notes": "Depends on Schema helpers.",
                    "rollout_risk": "medium - manual checkpoint.",
                    "validation": "Human must add a remark and close the issue.",
                    "parent_context": "Approved plan slice: stop for human validation.",
                    "automation": "manual-close",
                    "depends_on": ["Schema helpers"],
                },
            ),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 2
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert "Run `agent-loop issue <this issue number>`" in runner.issues[0]["body"]
    assert "Approved plan slice: add schema helpers" in runner.issues[0]["body"]
    assert runner.issues[1]["title"] == "[Human] Phase 2: Human rollout checkpoint (from #56)"
    assert "depends on #101: Schema helpers" in runner.issues[1]["body"]
    assert "human should add the required remark/update and close this issue" in runner.issues[1]["body"]
    summary = runner.comments[-1]
    assert summary.startswith("Approved plan decomposed for issue #56.")
    assert "Every phase above has a GitHub child issue" in summary
    assert "<!-- AGENT_PLAN_DECOMPOSITION:" in summary
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("mode", ["decompose-only", "implement-by-phase"])
@pytest.mark.parametrize("source", ["plan", "cli"])
def test_issue_loop_split_rejects_nonempty_closing_ids_before_materialization(tmp_path, mode, source):
    plan = structured_plan_state(summary="Add schema helpers.")
    payload, end = json.JSONDecoder().raw_decode(plan)
    payload["additional_closing_issue_ids"] = [99] if source == "plan" else []
    plan = json.dumps(payload) + plan[end:]
    runner = FakeRunner(
        claude_outputs=[plan],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(
        tmp_path, plan_execution_mode=mode,
        expected_closing_issue_ids=(99,) if source == "cli" else None,
    )
    with pytest.raises(AgentLoopError, match="Additional expected closing issue IDs are single-PR-only"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert runner.issues == []
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_decompose_only_is_idempotent(tmp_path):
    plan = structured_plan_state(summary="Add schema helpers.")
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="decompose-only",
        plan_hash=approved_plan_hash(plan),
        created=(),
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_rerun_without_handoff_implements_once(tmp_path):
    plan = structured_plan_state(summary="Add schema helpers.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "GitHub issue #99" in claude_calls[0][-1]

def test_issue_loop_plan_first_implement_by_phase_rerun_with_handoff_stops(tmp_path, capsys):
    plan = structured_plan_state(summary="Add schema helpers.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        phase_index=1,
        created=child,
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:03Z", "body": handoff},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "already handed off to child issue #99" in output
    assert "agent-loop issue 99" in output
    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_human_first_rerun_does_not_handoff(tmp_path):
    plan = structured_plan_state(summary="Validate migration manually first.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Manual readiness check", automation="human-action"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_stops_on_human_first_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Validate migration manually first."),
            plan_decomposition_json(
                {
                    "title": "Manual readiness check",
                    "scope": "Human validates external readiness.",
                    "non_goals": "No agent PR.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "medium - manual readiness gate.",
                    "validation": "Human remark and closure required.",
                    "parent_context": "Approved plan slice: manual readiness gate.",
                    "automation": "human-action",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"].startswith("[Human] Phase 1")
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


def test_issue_loop_dry_run_implement_by_phase_allows_dependency_preview(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(
                {
                    "title": "First phase",
                    "scope": "First.",
                    "non_goals": "None.",
                    "dependency_notes": "First.",
                    "rollout_risk": "low.",
                    "validation": "Tests.",
                    "parent_context": "Context.",
                    "automation": "agent-pr",
                    "depends_on": [],
                },
                {
                    "title": "Second phase",
                    "scope": "Second.",
                    "non_goals": "None.",
                    "dependency_notes": "After first.",
                    "rollout_risk": "low.",
                    "validation": "Tests.",
                    "parent_context": "Context.",
                    "automation": "agent-pr",
                    "depends_on": ["First phase"],
                },
            ),
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    result = run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(
            tmp_path,
            dry_run=True,
            plan_execution_mode="implement-by-phase",
        ),
        plan_first=True,
    )

    assert result == 0
    assert len(runner.issues) == 2
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2

def test_issue_loop_plan_first_implement_by_phase_implements_first_agent_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(),
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    decomposition_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_DECOMPOSITION:" in comment
    )
    implementation_index = next(
        index for index, comment in enumerate(runner.comments) if comment.startswith("## Issue implementation")
    )
    handoff_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment
    )
    assert decomposition_index < handoff_index < implementation_index
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 3
    assert "GitHub issue #99" in claude_calls[2][-1]
    assert "Approved implementation plan" in claude_calls[2][-1]

def test_issue_loop_plan_first_implement_by_phase_missing_child_number_does_not_handoff(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[None],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match="child issue number"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)

def test_phase_implementation_handoff_rejects_malformed_marker():
    comment = IssueComment(
        author="bot",
        created_at="2026-05-23T00:00:00Z",
        body="<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: not-valid-base64 -->",
    )

    with pytest.raises(AgentLoopError, match="Invalid AGENT_PLAN_PHASE_IMPLEMENTATION payload"):
        find_existing_phase_implementation_handoff(
            (comment,),
            parent_issue=56,
            plan_hash="abc123",
            mode="implement-by-phase",
            phase_index=1,
            child_issue_number=99,
        )


def test_oversized_parent_excerpt_is_shortened_so_the_child_publishes():
    """#902: a plan large enough to deserve staging must not block its children.

    Issue #841 planned three stages and then died with
    'GitHub issue body exceeds 60000 characters' while creating child issues.
    """
    from coding_review_agent_loop.decomposition import format_phase_issue_body
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    huge_context = "\n".join(
        f"Parent plan line {index}: " + "context " * 30 for index in range(3_000)
    )
    assert len(huge_context) > MAX_GITHUB_BODY_CHARS
    phase = PlanPhase(
        title="Stage one",
        scope="Implement the reviewed stage contract.",
        non_goals="No rollout.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="Run the focused tests.",
        parent_context=huge_context,
        automation="agent-pr",
        depends_on=(),
    )

    body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=841, approved_plan=huge_context, phase=phase,
        created_so_far=(),
    )

    assert len(body) <= MAX_GITHUB_BODY_CHARS
    assert "Child phase issue for parent #841" in body
    # The stage contract survives; only the inherited excerpt is cut.
    assert "Implement the reviewed stage contract." in body
    assert "Run the focused tests." in body
    assert "canonical plan comment" in body
    assert "Parent plan line 0:" in body


def test_fresh_child_recovery_adopts_children_whose_excerpt_was_shortened(tmp_path):
    """#902: an interrupted staged run over a huge plan stays resumable.

    The published children carry a shortened parent-plan excerpt, so the fresh
    recovery content check must compare against the bounded excerpt the body
    actually carries rather than the unbounded parent context.
    """
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    plan = "\n".join(f"Approved plan line {index}: " + "context " * 30 for index in range(3_000))
    assert len(plan) > MAX_GITHUB_BODY_CHARS
    topology = _fresh_recovery_topology(plan)
    candidates = [
        _fresh_child_issue(phase, plan, 100 + index)
        for index, phase in enumerate(topology.phases, start=1)
    ]
    for candidate in candidates:
        body = candidate["body"]
        assert isinstance(body, str)
        assert len(body) <= MAX_GITHUB_BODY_CHARS
        assert "canonical plan comment" in body
    runner = FakeRunner(search_issues_payload=candidates)

    recovered = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan=plan,
        decomposition=topology,
        topology_source="approved-plan-v1",
        issue_comments=(),
        mode="implement-by-phase",
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="recommendation-digest",
        preflight_only=True,
    )

    assert [item.origin for item in recovered] == ["adopted", "adopted"]


def test_fresh_child_recovery_still_rejects_a_foreign_parent_excerpt(tmp_path):
    """#902: bounded matching must not accept a different plan's excerpt."""
    from coding_review_agent_loop.decomposition import _fresh_phase_content_matches
    from coding_review_agent_loop.github import FoundIssue

    plan = "\n".join(f"Approved plan line {index}: " + "context " * 30 for index in range(3_000))
    topology = _fresh_recovery_topology(plan)
    phase = topology.phases[0]
    candidate = _fresh_child_issue(phase, plan, 101)
    foreign = candidate["body"].replace("Approved plan line 0:", "Unrelated plan line 0:", 1)

    assert _fresh_phase_content_matches(
        FoundIssue(
            number=101,
            title=candidate["title"],
            url=candidate["url"],
            body=candidate["body"],
        ),
        parent_issue=56,
        phase=phase,
    )
    assert not _fresh_phase_content_matches(
        FoundIssue(number=101, title=candidate["title"], url=candidate["url"], body=foreign),
        parent_issue=56,
        phase=phase,
    )


def test_oversized_retained_excerpt_is_shortened_in_the_parent_summary():
    """#907: the parent summary embeds the retained-parent excerpt.

    Issue #841 created its three child issues and then failed to publish the
    summary with 'GitHub comment body exceeds 60000 characters'.
    """
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    huge_excerpt = "\n".join(
        f"Retained scope line {index}: " + "detail " * 30 for index in range(3_000)
    )
    assert len(huge_excerpt) > MAX_GITHUB_BODY_CHARS
    phase = PlanPhase(
        title="Stage one",
        scope="Implement the reviewed stage contract.",
        non_goals="No rollout.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="Run the focused tests.",
        parent_context="Approved parent plan.",
        automation="agent-pr",
        depends_on=(),
    )

    body = format_decomposition_parent_summary(
        parent_issue=841,
        mode="implement-by-phase",
        plan_hash="a" * 16,
        created=(CreatedPhaseIssue(phase=phase, issue_url="https://example/issues/904", issue_number=904),),
        retained_parent_scope=RetainedParentScope(
            plan_subject="b" * 64, plan_hash="a" * 16, excerpt=huge_excerpt,
        ),
    )

    assert len(body) <= MAX_GITHUB_BODY_CHARS
    assert "Approved plan decomposed for issue #841." in body
    # The child table and the plan identity survive; only the excerpt is cut.
    assert "https://example/issues/904" in body
    assert "Retained scope line 0:" in body
    assert "canonical plan comment" in body


@pytest.mark.parametrize(
    "unit",
    [
        pytest.param("保留された親スコープの詳細な説明文です。", id="cjk"),
        pytest.param('He said "\\\\path\\to\\file" — \U0001f9ea test\t', id="escape-heavy"),
    ],
)
def test_non_ascii_retained_excerpt_is_shortened_in_the_parent_summary(unit):
    """#907: the excerpt budget must measure the rendered body, not characters.

    `_encode_json_payload` serializes with `ensure_ascii=True`, so a CJK
    character costs a six-character escape (an emoji, a surrogate pair) before
    base64 expands it again.  A character-ratio budget retains far more text
    than the body can hold and the summary still overflows.
    """
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    huge_excerpt = "\n".join(f"{index}: {unit * 20}" for index in range(3_000))
    assert len(huge_excerpt) > MAX_GITHUB_BODY_CHARS
    phase = PlanPhase(
        title="Stage one",
        scope="Implement the reviewed stage contract.",
        non_goals="No rollout.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="Run the focused tests.",
        parent_context="Approved parent plan.",
        automation="agent-pr",
        depends_on=(),
    )

    body = format_decomposition_parent_summary(
        parent_issue=841,
        mode="implement-by-phase",
        plan_hash="a" * 16,
        created=(
            CreatedPhaseIssue(
                phase=phase, issue_url="https://example/issues/904", issue_number=904
            ),
        ),
        retained_parent_scope=RetainedParentScope(
            plan_subject="b" * 64, plan_hash="a" * 16, excerpt=huge_excerpt,
        ),
    )

    assert len(body) <= MAX_GITHUB_BODY_CHARS
    assert "Approved plan decomposed for issue #841." in body
    assert "https://example/issues/904" in body
    assert "canonical plan comment" in body
    # The embedded record still decodes, and carries the same shortened excerpt.
    encoded = re.search(r"<!-- AGENT_PLAN_DECOMPOSITION: (\S+) -->", body).group(1)
    metadata = _decode_metadata(encoded)
    assert metadata.retained_parent_scope is not None
    assert metadata.retained_parent_scope.excerpt in body
    assert len(metadata.retained_parent_scope.excerpt) < len(huge_excerpt)


def test_parent_summary_overflow_names_the_surface_when_nothing_can_be_cut():
    """A summary whose fixed sections overflow raises a precise diagnostic."""
    from coding_review_agent_loop.round_transport import MAX_GITHUB_BODY_CHARS

    phase = PlanPhase(
        title="T" * 400,
        scope="Implement the reviewed stage contract.",
        non_goals="No rollout.",
        dependency_notes="No dependencies.",
        rollout_risk="low.",
        validation="Run the focused tests.",
        parent_context="Approved parent plan.",
        automation="agent-pr",
        depends_on=(),
    )
    created = tuple(
        CreatedPhaseIssue(phase=phase, issue_url=f"https://example/issues/{n}", issue_number=n)
        for n in range(1, 120)
    )

    with pytest.raises(AgentLoopError) as excinfo:
        format_decomposition_parent_summary(
            parent_issue=841,
            mode="implement-by-phase",
            plan_hash="a" * 16,
            created=created,
        )

    message = str(excinfo.value)
    assert "Decomposition parent summary for issue #841" in message
    assert str(MAX_GITHUB_BODY_CHARS) in message


def test_retained_parent_scope_matches_only_reconciles_the_excerpt():
    """#907: every field but the excerpt must still match exactly."""
    expected = RetainedParentScope(
        plan_subject="s" * 32,
        plan_hash="a" * 16,
        excerpt="The approved plan's retained scope.\nSecond line of detail.",
        status="required",
        deliverables=("Deliver the retained scope.",),
    )
    section = retained_parent_excerpt_section(expected.excerpt, parent_issue=841)

    assert retained_parent_scope_matches(expected, expected, parent_issue=841)
    assert retained_parent_scope_matches(None, None, parent_issue=841)
    assert not retained_parent_scope_matches(None, expected, parent_issue=841)
    shortened = dataclasses.replace(
        expected, excerpt=shortened_section(section, budget=len(expected.excerpt) // 2)
    )
    assert shortened.excerpt != expected.excerpt
    assert retained_parent_scope_matches(shortened, expected, parent_issue=841)
    # A shortened excerpt recorded under a different parent points elsewhere.
    assert not retained_parent_scope_matches(
        dataclasses.replace(
            expected,
            excerpt=shortened_section(
                retained_parent_excerpt_section(expected.excerpt, parent_issue=999),
                budget=len(expected.excerpt) // 2,
            ),
        ),
        expected,
        parent_issue=841,
    )
    assert not retained_parent_scope_matches(
        dataclasses.replace(shortened, status="none"), expected, parent_issue=841
    )
    assert not retained_parent_scope_matches(
        dataclasses.replace(expected, excerpt="A different plan's scope."),
        expected,
        parent_issue=841,
    )


def test_retained_parent_scope_rejects_content_around_a_bounded_excerpt():
    """#907: the stored excerpt must be an exact full or shortened form.

    The recovery check reads an isolated record field, so anything appended to
    or prefixed onto the full text or a shortened form is divergent and must
    fail closed rather than be tolerated as surrounding body text.
    """
    expected = RetainedParentScope(
        plan_subject="s" * 32,
        plan_hash="a" * 16,
        excerpt="The approved plan's retained scope.\nSecond line of detail.",
    )
    section = retained_parent_excerpt_section(expected.excerpt, parent_issue=841)
    notice = shortening_notice(section)
    shortened = shortened_section(section, budget=len(expected.excerpt) // 2)
    assert shortened != expected.excerpt

    def matches(excerpt):
        return retained_parent_scope_matches(
            dataclasses.replace(expected, excerpt=excerpt), expected, parent_issue=841
        )

    assert matches(expected.excerpt)
    assert matches(shortened)
    assert matches(notice)
    # Appended, prefixed or interleaved content is divergent, in both forms.
    assert not matches(expected.excerpt + "\n\nForeign appended scope.")
    assert not matches("Foreign leading scope.\n\n" + expected.excerpt)
    assert not matches(shortened + "\n\nForeign appended scope.")
    assert not matches("Foreign leading scope.\n\n" + shortened)
    assert not matches(notice + "\n\nForeign appended scope.")
    assert not matches(expected.excerpt + "\n\n" + notice + "\n\nForeign scope.")
    # A retained opening that is not an opening of the recomputed plan fails.
    assert not matches("Foreign opening.\n\n" + notice)


# --- Staged parent advancement (#918) -------------------------------------


def test_staged_parent_advances_to_the_next_phase_after_the_first_completes(
    tmp_path, monkeypatch
):
    """Matrix row `advance-next-phase`: a merged stage-1 dispatches stage-2."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        claude_outputs=[
            "Implemented stage 2.\n<!-- AGENT_PR: 78 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [child_pr_handoff_comment(99, 912)],
            100: [],
        },
        issue_payloads_by_number={99: {"state": "closed"}, 100: {"state": "open"}},
        pr_payloads_by_number={912: pr_payload_for_state(912, "MERGED")},
        pr_payload={"body": "Fixes #100"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")
    # Order the durable handoff against the coder invocation without relying on
    # comment/command index arithmetic.
    events = []
    real_post = orchestrator_module.post_phase_implementation_handoff_comment
    real_implement = orchestrator_module._implement_approved_issue

    def _record_post(*args, **kwargs):
        events.append(f"handoff-{kwargs['phase_index']}")
        return real_post(*args, **kwargs)

    def _record_implement(*args, **kwargs):
        events.append("coder")
        return real_implement(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module, "post_phase_implementation_handoff_comment", _record_post
    )
    monkeypatch.setattr(orchestrator_module, "_implement_approved_issue", _record_implement)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    handoffs = [
        _decode_phase_implementation_handoff_metadata(
            PHASE_IMPLEMENTATION_MARKER_RE.search(comment).group("payload")
        )
        for comment in runner.comments
        if PHASE_IMPLEMENTATION_MARKER_RE.search(comment)
    ]
    assert [item.phase_index for item in handoffs] == [2]
    assert handoffs[0].child_issue_number == 100
    # Stage-1 is never re-dispatched and never gets a duplicate handoff, and the
    # stage-2 handoff is persisted before any coder runs.
    assert events == ["handoff-2", "coder"]
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "GitHub issue #100" in claude_calls[0][-1]
    assert "GitHub issue #99" not in claude_calls[0][-1]


def test_staged_parent_reports_a_terminal_state_when_every_phase_is_complete(tmp_path, capsys):
    """Matrix row `all-phases-complete-terminal`."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
        phase_handoff_comment(plan, created, 2),
        phase_handoff_comment(plan, created, 3),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [child_pr_handoff_comment(99, 912)],
            100: [child_pr_handoff_comment(100, 913)],
            101: [child_pr_handoff_comment(101, 914)],
        },
        issue_payloads_by_number={
            99: {"state": "closed"}, 100: {"state": "closed"}, 101: {"state": "closed"},
        },
        pr_payloads_by_number={
            912: pr_payload_for_state(912, "MERGED"),
            913: pr_payload_for_state(913, "MERGED"),
            914: pr_payload_for_state(914, "MERGED"),
        },
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "all 3 staged phases are delivered" in output
    assert "delivered by child issue #99 with merged PR #912" in output
    assert "delivered by child issue #101 with merged PR #914" in output
    assert "resume directly with" not in output
    assert "No parent-side work remains" in output
    assert not any("AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "issue", "close"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("pr_state", [None, "OPEN"])
def test_staged_parent_hints_only_for_an_open_child(tmp_path, capsys, pr_state):
    """Matrix row `open-child-runnable-hint`."""
    plan, created, summary = staged_legacy_plan_records(stage_count=1)
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [] if pr_state is None else [child_pr_handoff_comment(99, 912)]
        },
        issue_payloads_by_number={99: {"state": "open"}},
        pr_payloads_by_number=(
            {} if pr_state is None else {912: pr_payload_for_state(912, pr_state)}
        ),
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "already handed off to child issue #99" in output
    assert "`agent-loop issue 99`" in output
    assert "--implementation-coder" not in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("pr_state", [None, "OPEN", "CLOSED"])
def test_staged_parent_fails_closed_when_a_closed_child_lacks_a_merged_pr(tmp_path, pr_state):
    """Matrix row `unauthenticated-child-fails-closed`."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [] if pr_state is None else [child_pr_handoff_comment(99, 912)]
        },
        issue_payloads_by_number={99: {"state": "closed"}},
        pr_payloads_by_number=(
            {} if pr_state is None else {912: pr_payload_for_state(912, pr_state)}
        ),
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match="cannot be authenticated as delivered"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert not any("AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("pr_state", ["MERGED", "CLOSED"])
def test_staged_parent_fails_closed_when_an_open_child_has_a_nonopen_pr(tmp_path, capsys, pr_state):
    """Matrix row `open-child-nonopen-pr-fails-closed`."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={99: [child_pr_handoff_comment(99, 912)]},
        issue_payloads_by_number={99: {"state": "open"}},
        pr_payloads_by_number={912: pr_payload_for_state(912, pr_state)},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match=f"is {pr_state}, not OPEN"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert "resume directly with" not in capsys.readouterr().out
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_staged_parent_fails_closed_on_an_out_of_order_handoff(tmp_path, capsys):
    """Matrix row `out-of-order-handoff-fails-closed`."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 2),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_payloads_by_number={99: {"state": "open"}, 100: {"state": "open"}},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match="while phase 1"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    output = capsys.readouterr().out
    assert "resume directly with" not in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_first_staged_run_without_a_handoff_dispatches_phase_one(tmp_path):
    """Matrix row `first-run-no-handoff`: no child-state reads, phase 1 dispatched."""
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
    ]
    runner = FakeRunner(
        claude_outputs=[
            "Implemented stage 1.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=parent_comments,
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    handoffs = [
        _decode_phase_implementation_handoff_metadata(
            PHASE_IMPLEMENTATION_MARKER_RE.search(comment).group("payload")
        )
        for comment in runner.comments
        if PHASE_IMPLEMENTATION_MARKER_RE.search(comment)
    ]
    assert [item.phase_index for item in handoffs] == [1]
    # No child issue-state lookup is made for any phase, including the selected
    # phase 1, because no phase carries a recorded handoff.
    assert not any(
        cmd[:2] == ["gh", "api"] and cmd[2].endswith(f"/issues/{child}")
        for cmd, _cwd in runner.commands
        for child in (99, 100, 101)
    )
    # Child PR evidence is only read for a phase whose child state was read, so
    # the absence of any child issue-state read above also excludes it.
    assert not any(
        cmd[:3] == ["gh", "pr", "view"] and "912" in cmd for cmd, _cwd in runner.commands
    )


def _outcome(created, *, plan_hash="parent-plan-hash", mode="implement-by-phase",
             stage_ids=None, automations=None, retained=None, final=None):
    return StagedTopologyOutcome(
        created=tuple(created),
        stage_ids=tuple(
            stage_ids or tuple(str(index) for index in range(1, len(created) + 1))
        ),
        automations=tuple(automations or tuple(item.phase.automation for item in created)),
        plan_hash=plan_hash,
        mode=mode,
        topology_source="model",
        retained_parent_scope=retained,
        final_integration_work=final,
    )


def _handoff(phase_index, child_issue_number, *, plan_hash="parent-plan-hash",
             mode="implement-by-phase", stage_id=None):
    return PhaseImplementationHandoffMetadata(
        parent_issue=56,
        plan_hash=plan_hash,
        mode=mode,
        phase_index=phase_index,
        phase_title=f"Stage {phase_index}",
        automation="agent-pr",
        child_issue_number=child_issue_number,
        child_issue_url=f"https://github.com/OWNER/REPO/issues/{child_issue_number}",
        stage_id=stage_id,
    )


class _ExplodingRunner:
    """Any GitHub read is a failure: reconciliation must precede child reads."""

    def run(self, *_args, **_kwargs):  # pragma: no cover - only reached on a bug
        raise AssertionError("child state must not be read before reconciliation")


def _recorded(title, automation="agent-pr"):
    return RecordedPhase(title=title, automation=automation)


@pytest.mark.parametrize(
    ("created", "handoffs", "message"),
    [
        (
            (CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=None),),
            (),
            "has no child issue number recorded",
        ),
        (
            (
                CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),
                CreatedPhaseIssue(phase=_recorded("Two"), issue_url=None, issue_number=99),
            ),
            (),
            "maps child issue #99 to both phase 1",
        ),
        (
            (CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),),
            (_handoff(1, 100),),
            "its handoff record names child issue #100",
        ),
        (
            (CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),),
            (_handoff(2, 99),),
            "outside its 1-phase topology",
        ),
        (
            (
                CreatedPhaseIssue(
                    phase=_recorded("One", "human-action"), issue_url=None, issue_number=99
                ),
            ),
            (_handoff(1, 99),),
            "human-owned stages are never dispatched",
        ),
        (
            (CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),),
            (_handoff(1, 99, stage_id="stage-one"),),
            "handoff record records `stage-one`",
        ),
    ],
)
def test_staged_progress_reconciliation_fails_closed(monkeypatch, tmp_path, created, handoffs, message):
    """Matrix row `child-mapping-reconciliation-fails-closed`."""
    monkeypatch.setattr(
        phase_progress_module, "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: handoffs,
    )
    with pytest.raises(AgentLoopError, match=re.escape(message)):
        resolve_staged_phase_progress(
            _ExplodingRunner(),
            config=make_config(tmp_path),
            parent_issue=56,
            parent_comments=(),
            outcome=_outcome(created),
        )


@pytest.mark.parametrize(
    ("second", "shape"),
    [
        (lambda: dataclasses.replace(_handoff(1, 99), phase_title="Renamed"), "divergent"),
        (lambda: _handoff(1, 99), "duplicate"),
    ],
)
def test_staged_progress_rejects_more_than_one_handoff_for_one_index(
    monkeypatch, tmp_path, second, shape
):
    """At most one record may exist per index, identical duplicates included."""
    created = (
        CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),
    )
    monkeypatch.setattr(
        phase_progress_module, "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: (_handoff(1, 99), second()),
    )
    with pytest.raises(AgentLoopError, match=f"{shape} phase handoff records"):
        resolve_staged_phase_progress(
            _ExplodingRunner(),
            config=make_config(tmp_path),
            parent_issue=56,
            parent_comments=(),
            outcome=_outcome(created),
        )


def test_staged_progress_excludes_handoffs_outside_the_outcome_plan_identity(
    monkeypatch, tmp_path
):
    """Matrix row `legacy-topology-rerun`, handoff-filter aspect only.

    The orchestrated identity, obligation and hint behavior of that row is
    covered by the legacy rerun tests above; this asserts the filter directly.

    A later-index handoff recorded under a different plan hash (or a different
    mode) is invisible to progress, so it cannot trip the ordered-prefix
    invariant on a legacy rerun.
    """
    created = (
        CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),
        CreatedPhaseIssue(phase=_recorded("Two"), issue_url=None, issue_number=100),
    )
    monkeypatch.setattr(
        phase_progress_module, "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: (
            _handoff(2, 100, plan_hash="a-stale-plan-hash"),
            _handoff(2, 100, mode="decompose-only"),
        ),
    )
    progress = resolve_staged_phase_progress(
        _ExplodingRunner(),
        config=make_config(tmp_path),
        parent_issue=56,
        parent_comments=(),
        outcome=_outcome(created),
    )
    assert [item.status for item in progress] == ["not-dispatched", "not-dispatched"]
    assert [item.handoff for item in progress] == [None, None]


def test_staged_dry_run_dispatch_reads_no_child_state(tmp_path, capsys):
    """Matrix row `dry-run-unchanged`: topology-only, network-free."""
    plan, created, _summary = staged_legacy_plan_records()
    runner = FakeRunner(issue_payloads_by_number={99: {"state": "closed"}})
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase", dry_run=True)
    outcome = StagedTopologyOutcome(
        created=created,
        stage_ids=("1", "2", "3"),
        automations=("agent-pr",) * 3,
        plan_hash=approved_plan_hash(plan),
        mode="implement-by-phase",
        topology_source="model",
    )
    parent = IssueContext(
        number=56, repo="OWNER/REPO", title="Parent", body="Parent",
        url="https://github.com/OWNER/REPO/issues/56", comments=(),
    )

    result = _dispatch_current_decomposition_phase(
        runner, config=config, memory=None, usage_context=None, issue_number=56,
        current_plan=plan, plan_subject=_plan_subject(plan), outcome=outcome,
        recommendation=None,
        approved_plan_context=None, issue_context=parent,
        mode="implement-by-phase", coder_session_id=None,
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "dry-run decomposed the approved plan" in output
    assert "staged child work:" not in output
    assert runner.commands == []
    assert runner.comments == []


def test_staged_parent_fails_closed_when_child_pr_evidence_is_unreadable(tmp_path):
    """Matrix row `unauthenticated-child-fails-closed`: unreadable PR evidence.

    The canonical record names PR #912, but GitHub answers with a PR whose URL
    does not match the recorded one, so the evidence cannot be authenticated.
    """
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    unreadable = pr_payload_for_state(912, "MERGED")
    unreadable["url"] = "https://github.com/OWNER/REPO/pull/913"
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={99: [child_pr_handoff_comment(99, 912)]},
        issue_payloads_by_number={99: {"state": "closed"}},
        pr_payloads_by_number={912: unreadable},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError) as failure:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    message = str(failure.value)
    # The progress boundary names the staged parent, phase, stage and child ...
    assert "Issue #56 could not authenticate phase 1 (`1`) from child issue #99" in message
    assert "then rerun the parent" in message
    # ... while preserving the underlying cause.
    assert "state could not be determined" in message
    assert "#912" in message
    assert not any("AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("automation", ["agent-pr", "human-action"])
def test_staged_parent_names_the_phase_when_a_child_state_is_malformed(tmp_path, automation):
    """An unreadable child issue state names the parent, phase, stage and child."""
    plan, created, summary = staged_legacy_plan_records(
        stage_count=1, automations=(automation,)
    )
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
    ]
    if automation == "agent-pr":
        parent_comments.append(phase_handoff_comment(plan, created, 1))
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_payloads_by_number={99: {"state": "merged"}},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError) as failure:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    message = str(failure.value)
    assert "Issue #56 could not authenticate phase 1 (`1`) from child issue #99" in message
    assert "reported unexpected state 'merged'" in message
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_legacy_rerun_with_recorded_stage_ids_keeps_todays_hint(tmp_path, capsys):
    """Matrix row `legacy-topology-rerun`: recorded stage identity and hint."""
    plan, created, summary = staged_v1_recorded_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={99: []},
        issue_payloads_by_number={99: {"state": "open"}},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    # Stage identity comes from the recorded summary, not the ordinal fallback.
    assert "recorded-stage-1: in progress (#99)" in output
    assert "recorded-stage-2: pending" in output
    # A legacy handoff resumes as direct implementation, never child planning.
    assert "resume directly with `agent-loop issue 99`" in output
    assert "--plan-first" not in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any("AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in runner.comments)


def test_legacy_rerun_without_recorded_stage_ids_uses_the_ordinal_fallback(tmp_path, capsys):
    """Matrix row `legacy-topology-rerun`: ordinal identity and `none` obligations."""
    plan, created, summary = staged_legacy_plan_records(stage_count=2)
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
        phase_handoff_comment(plan, created, 2),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [child_pr_handoff_comment(99, 912)],
            100: [child_pr_handoff_comment(100, 913)],
        },
        issue_payloads_by_number={99: {"state": "closed"}, 100: {"state": "closed"}},
        pr_payloads_by_number={
            912: pr_payload_for_state(912, "MERGED"),
            913: pr_payload_for_state(913, "MERGED"),
        },
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "1: delivered by child issue #99 with merged PR #912" in output
    assert "2: delivered by child issue #100 with merged PR #913" in output
    # A legacy summary records no obligations, so both report `none`.
    assert "Retained-parent obligations: none." in output
    assert "Final-integration obligations: none." in output
    assert "No parent-side work remains" in output


def test_legacy_rerun_reports_recorded_summary_obligations(tmp_path, capsys):
    """Matrix row `legacy-topology-rerun`: obligations come from the summary."""
    plan, created, summary = staged_v1_recorded_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
        phase_handoff_comment(plan, created, 2),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={
            99: [child_pr_handoff_comment(99, 912)],
            100: [child_pr_handoff_comment(100, 913)],
        },
        issue_payloads_by_number={99: {"state": "closed"}, 100: {"state": "closed"}},
        pr_payloads_by_number={
            912: pr_payload_for_state(912, "MERGED"),
            913: pr_payload_for_state(913, "MERGED"),
        },
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "recorded-stage-1: delivered by child issue #99 with merged PR #912" in output
    assert "Retained-parent obligations: required" in output
    assert "The recorded rollout note." in output
    assert "Final-integration obligations: required" in output
    assert "Wire the recorded stages together." in output
    assert "remains open pending that operator-owned parent work" in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("surface", ["child-context", "pr-evidence"])
def test_staged_parent_names_the_phase_when_a_child_payload_is_unreadable(tmp_path, surface):
    """Unreadable child-context or PR-evidence output still names the phase.

    `gh ... view --json` output that is not JSON raises inside the GitHub
    readers, which is where it is translated to an `AgentLoopError`; the
    progress boundary then attaches the parent, phase, stage and child.
    """
    plan, created, summary = staged_legacy_plan_records()
    parent_comments = approved_plan_comments(plan) + [
        {"author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:02Z", "body": summary},
        phase_handoff_comment(plan, created, 1),
    ]
    runner = FakeRunner(
        issue_comments=parent_comments,
        issue_comments_by_number={99: [child_pr_handoff_comment(99, 912)]},
        issue_payloads_by_number={99: {"state": "closed"}},
        pr_payloads_by_number={912: pr_payload_for_state(912, "MERGED")},
        malformed_issue_view_numbers=(99,) if surface == "child-context" else (),
        malformed_pr_view_numbers=(912,) if surface == "pr-evidence" else (),
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError) as failure:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    message = str(failure.value)
    assert "Issue #56 could not authenticate phase 1 (`1`) from child issue #99" in message
    assert "then rerun the parent" in message
    # The underlying unreadable-payload cause survives.
    assert "GitHub CLI output is not JSON" in message
    assert failure.value.__cause__ is not None
    assert not any("AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_staged_progress_wraps_a_raw_parsing_failure_with_phase_context(monkeypatch, tmp_path):
    """The boundary's defence-in-depth layer labels a non-AgentLoopError too."""
    created = (
        CreatedPhaseIssue(phase=_recorded("One"), issue_url=None, issue_number=99),
    )

    def _raise(*_args, **_kwargs):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    monkeypatch.setattr(
        phase_progress_module, "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: (_handoff(1, 99),),
    )
    monkeypatch.setattr(phase_progress_module, "get_issue_state", _raise)

    with pytest.raises(AgentLoopError) as failure:
        resolve_staged_phase_progress(
            _ExplodingRunner(),
            config=make_config(tmp_path),
            parent_issue=56,
            parent_comments=(),
            outcome=_outcome(created),
        )
    message = str(failure.value)
    assert "Issue #56 could not authenticate phase 1 (`1`) from child issue #99" in message
    assert "Expecting value" in message
    assert isinstance(failure.value.__cause__, ValueError)
