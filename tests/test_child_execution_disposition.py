import dataclasses
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.decomposition import (
    ChildDispositionOverride,
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    PlanPhase,
    child_disposition_override_digest,
    collect_child_disposition_overrides,
    format_child_disposition_override_comment,
    parse_child_disposition_override_records,
    reconcile_handoff_disposition,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
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


@pytest.mark.parametrize(
    ("disposition", "expected_events", "return_code"),
    [
        (EXECUTION_DISPOSITION_DIRECT, ["direct-handoff", "implement"], 6),
        (EXECUTION_DISPOSITION_PLANNING, ["planning-handoff", "plan"], 7),
    ],
)
def test_rtm_direct_and_planning_dispatch_persists_handoff_before_agent(
    tmp_path, monkeypatch, disposition, expected_events, return_code
):
    events = []
    phase = _phase(disposition)
    created = CreatedPhaseIssue(
        phase=phase,
        issue_url="https://github.com/OWNER/REPO/issues/56",
        issue_number=56,
    )
    route = resolve_child_execution_route(
        phase, topology_source="approved-plan-v1"
    )
    recommendation = SimpleNamespace(
        strategy="staged",
        identity=lambda: {"recommendation_sha256": "recommendation-digest"},
        child_stages=(SimpleNamespace(stage_id="stage-one"),),
    )
    context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child contract",
        body="Child body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
    )
    parent_context = dataclasses.replace(
        context,
        number=55,
        title="Parent",
        url="https://github.com/OWNER/REPO/issues/55",
    )
    monkeypatch.setattr(
        orchestrator,
        "post_phase_implementation_handoff_comment",
        lambda *_args, **_kwargs: events.append("direct-handoff"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_post_child_planning_handoff",
        lambda *_args, **_kwargs: events.append("planning-handoff"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_implement_approved_issue",
        lambda *_args, **_kwargs: events.append("implement") or 6,
    )
    monkeypatch.setattr(
        orchestrator,
        "_run_child_planning_cycle",
        lambda *_args, **_kwargs: events.append("plan") or 7,
    )
    result = orchestrator._dispatch_decomposition_child(
        FakeRunner(),
        config=make_config(tmp_path),
        memory=None,
        usage_context=SimpleNamespace(),
        parent_issue=55,
        approved_plan="Approved parent plan.",
        plan_hash="plan-hash",
        plan_subject="plan-subject",
        recommendation=recommendation,
        approved_plan_context=SimpleNamespace(
            matrix_available=False, risk_test_matrix_payload=None
        ),
        created=created,
        phase_index=1,
        route=route,
        child_issue_context=context,
        parent_issue_context=parent_context,
        coder_session_id=None,
        existing_handoff=None,
    )
    assert result == return_code
    assert events == expected_events


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


def test_rtm_contradictory_handoff_reconciliation_rejects_each_binding_fault():
    planning = _override(EXECUTION_DISPOSITION_PLANNING)
    with pytest.raises(AgentLoopError, match="carries no override digest"):
        reconcile_handoff_disposition(
            _phase(), _handoff(EXECUTION_DISPOSITION_PLANNING), ()
        )

    handoff = _handoff(EXECUTION_DISPOSITION_PLANNING, digest=planning.digest)
    wrong_identity = dataclasses.replace(planning, parent_issue=999)
    with pytest.raises(AgentLoopError, match="different parent, plan hash, or stage"):
        reconcile_handoff_disposition(_phase(), handoff, (wrong_identity,))

    wrong_disposition = dataclasses.replace(
        planning, disposition=EXECUTION_DISPOSITION_DIRECT
    )
    with pytest.raises(AgentLoopError, match="differs from the recorded handoff disposition"):
        reconcile_handoff_disposition(_phase(), handoff, (wrong_disposition,))


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


def test_rtm_human_first_stage_workflow_stops_without_handoff_and_other_override_is_ignored(
    tmp_path, monkeypatch, capsys
):
    approved_plan = "Approved mixed topology."
    from coding_review_agent_loop.decomposition import approved_plan_hash

    human_phase = dataclasses.replace(
        _phase(automation="human-action"),
        stage_id="human-stage",
        execution_disposition="human-owned",
    )
    agent_phase = dataclasses.replace(
        _phase(EXECUTION_DISPOSITION_PLANNING), stage_id="stage-two", position=2
    )
    created = (
        CreatedPhaseIssue(human_phase, "human-url", 56),
        CreatedPhaseIssue(agent_phase, "agent-url", 57),
    )
    other_override = format_child_disposition_override_comment(
        parent_issue=55,
        plan_hash=approved_plan_hash(approved_plan),
        stage_id="stage-two",
        disposition=EXECUTION_DISPOSITION_DIRECT,
        rationale="Only the later agent stage changes route.",
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent", url="parent-url",
        comments=(IssueComment(author="human", created_at=None, body=other_override),),
    )
    child = dataclasses.replace(parent, number=56, title="Human", comments=())
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        orchestrator, "post_phase_implementation_handoff_comment",
        lambda *_args, **_kwargs: pytest.fail("human stage must not post a handoff"),
    )
    recommendation = SimpleNamespace(
        strategy="staged",
        identity=lambda: {"recommendation_sha256": "digest"},
        child_stages=(human_phase, agent_phase),
    )
    result = orchestrator._dispatch_first_decomposition_phase(
        FakeRunner(), config=make_config(tmp_path), memory=None,
        usage_context=SimpleNamespace(), issue_number=55,
        current_plan=approved_plan, plan_subject="subject", created=created,
        recommendation=recommendation,
        approved_plan_context=SimpleNamespace(matrix_available=False),
        issue_context=parent, mode="implement-by-phase", coder_session_id=None,
    )
    assert result == 0
    assert "first phase requires human work" in capsys.readouterr().out


def test_rtm_nested_staged_child_stops_before_topology_mutation(
    tmp_path, monkeypatch, capsys
):
    payload = _fresh_staged_payload()
    plan = _plan_text(payload)
    child = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Planning child",
        body="Fresh staged child body.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
    )
    parent = dataclasses.replace(
        child,
        number=55,
        title="Parent",
        url="https://github.com/OWNER/REPO/issues/55",
    )
    monkeypatch.setattr(orchestrator, "_infer_staged_parent_issue", lambda _context: 55)
    monkeypatch.setattr(
        orchestrator,
        "_fresh_phase_marker_payload",
        lambda _context: {"source": "approved-plan-v1"},
    )
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        claude_outputs=[plan],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        plan_execution_mode="auto",
        execution_strategy_contract_required=True,
    )
    result = orchestrator._run_plan_first_loop(
        runner,
        issue_number=56,
        config=config,
        memory=None,
        issue_context=child,
        requested_policy="auto",
        usage_context=orchestrator._new_usage_context(config),
    )
    assert result == 2
    output = capsys.readouterr().out
    assert '"reason": "nested-staged-child"' in output
    bodies = "\n".join(runner.comments)
    assert "AGENT_PLAN_EXECUTION_DECISION" not in bodies
    assert "AGENT_PLAN_TOPOLOGY_CHECKPOINT" not in bodies
    assert "AGENT_PLAN_DECOMPOSITION" not in bodies
    assert "AGENT_PLAN_PHASE_IMPLEMENTATION" not in bodies
    assert not any(command[:3] == ["gh", "issue", "create"] for command, _cwd in runner.commands)


def test_rtm_multi_stage_overrides_scope_independently_and_dedupe(
    tmp_path, monkeypatch
):
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
    second_scoped = collect_child_disposition_overrides(
        parent_comments=comments,
        parent_issue=55,
        plan_hash="plan-hash",
        topology_stage_ids=("stage-one", "stage-two"),
        routed_stage_id="stage-two",
    )
    assert len(second_scoped) == 1
    first_route = resolve_child_execution_route(
        _phase(), topology_source="approved-plan-v1", overrides=scoped
    )
    second_phase = dataclasses.replace(
        _phase(EXECUTION_DISPOSITION_PLANNING), stage_id="stage-two"
    )
    second_route = resolve_child_execution_route(
        second_phase, topology_source="approved-plan-v1", overrides=second_scoped
    )
    assert first_route.disposition == EXECUTION_DISPOSITION_PLANNING
    assert first_route.override_digest == scoped[0].digest
    assert second_route.disposition == EXECUTION_DISPOSITION_DIRECT
    assert second_route.override_digest == second_scoped[0].digest

    recorded_handoffs = []
    monkeypatch.setattr(
        orchestrator,
        "_post_child_planning_handoff",
        lambda *_args, **kwargs: recorded_handoffs.append(("stage-one", kwargs)),
    )
    monkeypatch.setattr(
        orchestrator,
        "post_phase_implementation_handoff_comment",
        lambda *_args, **kwargs: recorded_handoffs.append(("stage-two", kwargs)),
    )
    monkeypatch.setattr(
        orchestrator, "_run_child_planning_cycle", lambda *_args, **_kwargs: 0
    )
    monkeypatch.setattr(
        orchestrator, "_implement_approved_issue", lambda *_args, **_kwargs: 0
    )
    recommendation = SimpleNamespace(
        strategy="staged",
        identity=lambda: {"recommendation_sha256": "recommendation-digest"},
        child_stages=(
            SimpleNamespace(stage_id="stage-one"),
            SimpleNamespace(stage_id="stage-two"),
        ),
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent",
        url="parent-url", comments=(),
    )
    for index, (phase, route, issue_number) in enumerate(
        ((_phase(), first_route, 56), (second_phase, second_route, 57)), start=1
    ):
        child = dataclasses.replace(
            parent, number=issue_number, title=phase.title, url=f"child-{issue_number}"
        )
        assert orchestrator._dispatch_decomposition_child(
            FakeRunner(), config=make_config(tmp_path), memory=None,
            usage_context=SimpleNamespace(), parent_issue=55,
            approved_plan="Approved parent plan.", plan_hash="plan-hash",
            plan_subject="plan-subject", recommendation=recommendation,
            approved_plan_context=SimpleNamespace(
                matrix_available=False, risk_test_matrix_payload=None
            ),
            created=CreatedPhaseIssue(phase, child.url, issue_number),
            phase_index=index, route=route, child_issue_context=child,
            parent_issue_context=parent, coder_session_id=None,
            existing_handoff=None,
        ) == 0

    assert [stage for stage, _kwargs in recorded_handoffs] == ["stage-one", "stage-two"]
    assert recorded_handoffs[0][1]["override_digest"] == scoped[0].digest
    assert recorded_handoffs[1][1]["override_digest"] == second_scoped[0].digest


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
