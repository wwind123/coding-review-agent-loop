import dataclasses
import json

import pytest

from agent_loop_helpers import structured_v1_plan_state
from coding_review_agent_loop.decomposition import (
    ChildDispositionOverride,
    PhaseImplementationHandoffMetadata,
    PlanPhase,
    child_disposition_override_digest,
    collect_child_disposition_overrides,
    format_child_disposition_override_comment,
    parse_child_disposition_override_records,
    reconcile_handoff_disposition,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.orchestrator import (
    CHILD_ROUTE_HUMAN,
    resolve_child_execution_route,
)
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
    validate_structured_plan_state,
)


def _phase(disposition=EXECUTION_DISPOSITION_DIRECT, *, automation="agent-pr"):
    return PlanPhase(
        title="Child contract",
        scope="Implement the child contract.",
        non_goals="Do not change unrelated behavior.",
        dependency_notes="No external dependencies.",
        rollout_risk="low",
        validation="Run focused tests.",
        parent_context="Approved parent slice.",
        automation=automation,
        stage_id="stage-one",
        position=1,
        deliverables=("Implementation",),
        non_goals_items=("Unrelated behavior",),
        acceptance_criteria=("Focused tests pass",),
        compatibility_constraints=("Preserve callers",),
        covered_scope_item_ids=("scope-1",),
        execution_disposition=disposition,
        disposition_rationale="The reviewed slice is complete.",
    )


def _handoff(disposition=EXECUTION_DISPOSITION_DIRECT, *, digest=None):
    return PhaseImplementationHandoffMetadata(
        parent_issue=55,
        plan_hash="plan-hash",
        mode="implement-by-phase",
        phase_index=1,
        phase_title="Child contract",
        automation="agent-pr",
        child_issue_number=56,
        child_issue_url="https://github.com/OWNER/REPO/issues/56",
        strategy="staged",
        topology_source="approved-plan-v1",
        execution_strategy_contract_version=1,
        recommendation_digest="recommendation-digest",
        stage_id="stage-one",
        plan_subject="plan-subject",
        execution_disposition=disposition,
        override_digest=digest,
    )


def _override(disposition, *, rationale="Human route correction"):
    payload = {
        "kind": "child-execution-disposition-override",
        "schema_version": 1,
        "parent_issue": 55,
        "plan_hash": "plan-hash",
        "stage_id": "stage-one",
        "disposition": disposition,
        "rationale": rationale,
    }
    return ChildDispositionOverride(
        parent_issue=55,
        plan_hash="plan-hash",
        stage_id="stage-one",
        disposition=disposition,
        rationale=rationale,
        digest=child_disposition_override_digest(payload),
        comment_locator="parent issue #55 comment 1",
    )


def _fresh_staged_payload():
    payload, _ = json.JSONDecoder().raw_decode(structured_v1_plan_state())
    recommendation = payload["execution_recommendation"]
    recommendation.pop("one_shot_delivery")
    recommendation.update(
        strategy="staged",
        staging_feasibility="safe",
        scope_items=[
            {
                "scope_item_id": "scope-1",
                "requirement": "Implement the child contract.",
                "acceptance_criteria": ["The contract is verified."],
            },
            {
                "scope_item_id": "scope-2",
                "requirement": "Complete integration.",
                "acceptance_criteria": ["Integration is verified."],
            },
        ],
        child_stages=[
            {
                "stage_id": "stage-one",
                "position": 1,
                "title": "Child contract",
                "summary": "Implement the child contract.",
                "deliverables": ["Implementation"],
                "non_goals": ["Unrelated behavior"],
                "acceptance_criteria": ["Focused tests pass"],
                "depends_on_stage_ids": [],
                "dependency_notes": "No external dependencies.",
                "automation": "agent-pr",
                "rollout_risk": "low",
                "compatibility_constraints": ["Preserve callers"],
                "covered_scope_item_ids": ["scope-1"],
                "execution_disposition": {
                    "disposition": "direct-implementation",
                    "rationale": "The reviewed slice is complete.",
                    "unresolved_design_decisions": [],
                },
            }
        ],
        retained_parent_work={
            "status": "required",
            "deliverables": ["Integrate the child."],
            "acceptance_criteria": ["Integration is verified."],
            "covered_scope_item_ids": ["scope-2"],
        },
        final_integration_work={
            "status": "none",
            "deliverables": [],
            "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
    )
    return payload


def _plan_text(payload):
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Coder"


def test_rtm_direct_ready_and_rtm_planning_child_routes_are_explicit():
    direct = resolve_child_execution_route(
        _phase(), topology_source="approved-plan-v1"
    )
    planning = resolve_child_execution_route(
        _phase(EXECUTION_DISPOSITION_PLANNING),
        topology_source="approved-plan-v1",
    )
    assert direct.is_direct and direct.origin == "phase"
    assert planning.is_planning and planning.origin == "phase"


def test_rtm_incomplete_direct_rejected_and_recovery_absence_is_legacy_ambiguous():
    payload = _fresh_staged_payload()
    stage = payload["execution_recommendation"]["child_stages"][0]
    stage.pop("execution_disposition")
    with pytest.raises(AgentLoopError, match="execution_disposition is required"):
        validate_structured_plan_state(
            _plan_text(payload),
            require_execution_strategy_contract=1,
            require_child_dispositions=True,
        )
    recovered = validate_structured_plan_state(_plan_text(payload))
    assert recovered.execution_recommendation.child_stages[0].execution_disposition is None

    stage["execution_disposition"] = {
        "disposition": "direct-implementation",
        "rationale": "Incomplete declaration.",
        "unresolved_design_decisions": ["Choose a storage format"],
    }
    with pytest.raises(AgentLoopError, match="unresolved_design_decisions must be empty"):
        validate_structured_plan_state(
            _plan_text(payload),
            require_execution_strategy_contract=1,
            require_child_dispositions=True,
        )


def test_rtm_legacy_ambiguous_routes_to_planning_but_legacy_handoff_is_direct():
    legacy = dataclasses.replace(
        _phase(), execution_disposition=None, disposition_rationale=None
    )
    route = resolve_child_execution_route(
        legacy, topology_source="approved-plan-v1"
    )
    assert route.is_planning and route.origin == "legacy-ambiguous"
    resumed = resolve_child_execution_route(
        legacy,
        topology_source="approved-plan-v1",
        recorded_handoff=dataclasses.replace(
            _handoff(), execution_disposition=None, override_digest=None
        ),
    )
    assert resumed.is_direct and resumed.origin == "handoff"


def test_rtm_signed_override_uses_parent_readiness_and_records_digest():
    override = _override(EXECUTION_DISPOSITION_PLANNING)
    route = resolve_child_execution_route(
        _phase(), topology_source="approved-plan-v1", overrides=(override,)
    )
    assert route.is_planning
    assert route.override_digest == override.digest

    incomplete = dataclasses.replace(_phase(EXECUTION_DISPOSITION_PLANNING), non_goals_items=())
    to_direct = _override(EXECUTION_DISPOSITION_DIRECT)
    with pytest.raises(AgentLoopError, match="reviewed parent stage alone is not direct-ready"):
        resolve_child_execution_route(
            incomplete, topology_source="approved-plan-v1", overrides=(to_direct,)
        )


def test_rtm_override_bound_handoff_survives_only_with_exact_record():
    override = _override(EXECUTION_DISPOSITION_PLANNING)
    handoff = _handoff(EXECUTION_DISPOSITION_PLANNING, digest=override.digest)
    assert (
        reconcile_handoff_disposition(_phase(), handoff, (override,))
        == EXECUTION_DISPOSITION_PLANNING
    )
    with pytest.raises(AgentLoopError, match="undiscoverable"):
        reconcile_handoff_disposition(_phase(), handoff, ())
    superseding = _override(
        EXECUTION_DISPOSITION_PLANNING, rationale="A distinct later record"
    )
    with pytest.raises(AgentLoopError, match="superseded"):
        reconcile_handoff_disposition(_phase(), handoff, (override, superseding))


def test_rtm_rerun_idempotent_and_rtm_flag_cannot_switch_after_handoff():
    handoff = _handoff()
    first = resolve_child_execution_route(
        _phase(), topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    second = resolve_child_execution_route(
        _phase(), topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    assert first == second
    late = _override(EXECUTION_DISPOSITION_PLANNING)
    with pytest.raises(AgentLoopError, match="cannot be added after dispatch"):
        resolve_child_execution_route(
            _phase(),
            topology_source="approved-plan-v1",
            recorded_handoff=handoff,
            overrides=(late,),
        )


def test_rtm_human_stage_keeps_stop_and_rejects_override():
    human = dataclasses.replace(
        _phase(automation="human-action"),
        execution_disposition="human-owned",
    )
    assert (
        resolve_child_execution_route(human, topology_source="approved-plan-v1").disposition
        == CHILD_ROUTE_HUMAN
    )
    with pytest.raises(AgentLoopError, match="Human-owned stages"):
        resolve_child_execution_route(
            human,
            topology_source="approved-plan-v1",
            overrides=(_override(EXECUTION_DISPOSITION_DIRECT),),
        )


def test_rtm_multi_stage_overrides_scope_independently_and_dedupe():
    first_body = format_child_disposition_override_comment(
        parent_issue=55,
        plan_hash="plan-hash",
        stage_id="stage-one",
        disposition=EXECUTION_DISPOSITION_PLANNING,
        rationale="Plan this child.",
    )
    second_body = format_child_disposition_override_comment(
        parent_issue=55,
        plan_hash="plan-hash",
        stage_id="stage-two",
        disposition=EXECUTION_DISPOSITION_DIRECT,
        rationale="The second child is ready.",
    )
    comments = (
        IssueComment(author="human", created_at="2026-01-01T00:00:00Z", body=first_body),
        IssueComment(author="human", created_at="2026-01-02T00:00:00Z", body=first_body),
        IssueComment(author="human", created_at="2026-01-03T00:00:00Z", body=second_body),
    )
    scoped = collect_child_disposition_overrides(
        parent_comments=comments,
        parent_issue=55,
        plan_hash="plan-hash",
        topology_stage_ids=("stage-one", "stage-two"),
        routed_stage_id="stage-one",
    )
    assert len(scoped) == 1
    assert scoped[0].stage_id == "stage-one"


def test_rtm_override_topology_rejection_precedes_stage_scoping():
    body = format_child_disposition_override_comment(
        parent_issue=55,
        plan_hash="plan-hash",
        stage_id="unknown-stage",
        disposition=EXECUTION_DISPOSITION_PLANNING,
        rationale="Invalid target.",
    )
    comment = IssueComment(
        author="human", created_at="2026-01-01T00:00:00Z", body=body
    )
    with pytest.raises(AgentLoopError, match="not a stage of the bound topology"):
        collect_child_disposition_overrides(
            parent_comments=(comment,),
            parent_issue=55,
            plan_hash="plan-hash",
            topology_stage_ids=("stage-one", "stage-two"),
            routed_stage_id="stage-two",
        )


def test_override_parser_ignores_unsigned_and_malformed_records():
    signed = format_child_disposition_override_comment(
        parent_issue=55,
        plan_hash="plan-hash",
        stage_id="stage-one",
        disposition=EXECUTION_DISPOSITION_PLANNING,
        rationale="Plan this child.",
    )
    records, ignored = parse_child_disposition_override_records(
        signed.removesuffix("\n-- Human Reviewer"), comment_locator="unsigned"
    )
    assert records == () and ignored == ()
    malformed = signed.replace('"schema_version": 1', '"schema_version": 2')
    records, ignored = parse_child_disposition_override_records(
        malformed, comment_locator="malformed"
    )
    assert records == ()
    assert ignored and "schema_version must be 1" in ignored[0]
