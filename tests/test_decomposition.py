import base64
import dataclasses
import json

import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
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
    format_execution_decision,
    format_phase_implementation_handoff_comment,
    format_topology_checkpoint,
    phase_identity,
    parse_plan_decomposition,
    create_decomposition_child_issues,
    adapt_typed_child_stages,
    normalize_execution_recommendation,
    find_existing_execution_decision,
)
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
    ExecutionChildStage,
    validate_structured_plan_state,
)
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.child_topology import NeedsHumanDecision
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
    _preflight_fresh_one_shot_recovery,
    _preflight_fresh_staged_topology,
)
from coding_review_agent_loop.split_materialization import (
    MaterializedSplitChild,
    SplitMaterializationMetadata,
    format_split_materialization_summary,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
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
