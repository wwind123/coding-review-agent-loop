import base64
import dataclasses
import json

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_v1_plan_state
from coding_review_agent_loop import orchestrator
from coding_review_agent_loop.comment_rendering import render_execution_recommendation_section
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue, PlanPhase, TopologyCheckpoint, approved_plan_hash,
    format_decomposition_parent_summary, format_phase_issue_body,
    format_phase_implementation_handoff_comment,
    format_child_disposition_override_comment,
    parse_child_disposition_override_records,
    PHASE_IMPLEMENTATION_MARKER_RE,
    normalize_execution_recommendation,
    format_topology_checkpoint, phase_identity,
    risk_matrix_row_ids_for_owner, validate_separately_planned_child_matrix,
    _decode_phase_implementation_handoff_metadata,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata, _plan_subject
from coding_review_agent_loop.protocol import parse_risk_test_matrix


def plan_record(plan):
    return _attach_round_metadata(plan, PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1,
        subject=_plan_subject(plan), canonical_plan=plan, raw_structured_coder_response=plan,
    ))


def comment(body):
    return IssueComment(author="bot", created_at="2026-09-13T00:00:00Z", body=body)


def replace_phase_plan_hash(body, plan_hash):
    marker = orchestrator.PHASE_IDENTITY_MARKER_RE.search(body)
    assert marker is not None
    payload = json.loads(
        base64.urlsafe_b64decode(marker.group("payload").encode("ascii")).decode("utf-8")
    )
    payload["plan_hash"] = plan_hash
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return body[:marker.start("payload")] + encoded + body[marker.end("payload"):]


def replace_phase_handoff_payload(body, **updates):
    marker = PHASE_IMPLEMENTATION_MARKER_RE.search(body)
    assert marker is not None
    payload = json.loads(
        base64.urlsafe_b64decode(marker.group("payload").encode("ascii")).decode("utf-8")
    )
    payload.update(updates)
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    return body[:marker.start("payload")] + encoded + body[marker.end("payload"):]


def fresh_staged_plan(*, first_disposition=None):
    raw = structured_v1_plan_state()
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["summary"] = "Fresh staged parent plan."
    payload["execution_recommendation"] = {
        "strategy": "staged",
        "rationale": "The two contracts are independently verifiable in order.",
        "staging_feasibility": "safe",
        "scope_items": [
            {
                "scope_item_id": "scope-1",
                "requirement": "Deliver the first contract.",
                "acceptance_criteria": ["The first contract is verified."],
            },
            {
                "scope_item_id": "scope-2",
                "requirement": "Deliver the second contract.",
                "acceptance_criteria": ["The second contract is verified."],
            },
        ],
        "coupling_constraints": [],
        "child_stages": [
            {
                "stage_id": "stage-one",
                "position": 1,
                "title": "First contract",
                "summary": "Deliver the first contract.",
                "deliverables": ["First contract."],
                "non_goals": [],
                "acceptance_criteria": ["The first contract is verified."],
                "depends_on_stage_ids": [],
                "dependency_notes": "No dependencies.",
                "automation": "agent-pr",
                "rollout_risk": "low",
                "compatibility_constraints": [],
                "covered_scope_item_ids": ["scope-1"],
            },
            {
                "stage_id": "stage-two",
                "position": 2,
                "title": "Second contract",
                "summary": "Deliver the second contract.",
                "deliverables": ["Second contract."],
                "non_goals": [],
                "acceptance_criteria": ["The second contract is verified."],
                "depends_on_stage_ids": ["stage-one"],
                "dependency_notes": "After the first contract.",
                "automation": "agent-pr",
                "rollout_risk": "medium",
                "compatibility_constraints": [],
                "covered_scope_item_ids": ["scope-2"],
            },
        ],
        "retained_parent_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "final_integration_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "caveats": [],
    }
    if first_disposition is not None:
        payload["execution_recommendation"]["child_stages"][0]["execution_disposition"] = {
            "disposition": first_disposition,
            "rationale": "The reviewed parent selects this route.",
            "unresolved_design_decisions": [],
        }
        payload["execution_recommendation"]["child_stages"][0]["non_goals"] = [
            "No unrelated changes."
        ]
        payload["execution_recommendation"]["child_stages"][0][
            "compatibility_constraints"
        ] = ["Preserve callers."]
    plan = json.dumps(payload) + raw[end:]
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    recommendation = validate_structured_plan_state(plan).execution_recommendation
    assert recommendation is not None
    json_part, footer = plan.split("\n<!-- AGENT_PLAN_STATE:", 1)
    return (
        json_part
        + "\n\n"
        + render_execution_recommendation_section(recommendation)
        + "\n<!-- AGENT_PLAN_STATE:"
        + footer
    )


def fresh_staged_matrix_plan():
    """Fresh staged parent fixture with explicit owned transition rows."""
    plan = fresh_staged_plan()
    payload, end = json.JSONDecoder().raw_decode(plan)
    rows = []
    for row_id, owner in (("row-stage-one", "stage-one"), ("row-stage-two", "stage-two")):
        rows.append({
            "row_id": row_id,
            "label": f"{owner} transition",
            "entry_path_or_mode": "review-only recovery",
            "initial_state": "review complete",
            "event": "repaired head passes",
            "expected_outcome": "The owned transition completes.",
            "forbidden_side_effects": ["No stale merge."],
            "proposed_test_level": "orchestrator",
            "proposed_test_location": "tests/test_child_plan_provenance.py",
            "applicability": "applicable",
            "related_scope_item_ids": ["scope-1" if owner == "stage-one" else "scope-2"],
            "execution_owner": owner,
        })
    payload["risk_test_matrix"] = {
        "applicability": "applicable",
        "rows": rows,
        "important_exclusions": ["Unrelated flag combinations are not required."],
    }
    payload["risk_test_matrix_changes"] = []
    json_plan = json.dumps(payload) + plan[end:]
    from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section
    matrix = payload["risk_test_matrix"]
    footer = "\n<!-- AGENT_PLAN_STATE:"
    return json_plan.replace(
        footer,
        "\n\n" + render_risk_test_matrix_section(
            parse_risk_test_matrix(matrix),
            (),
        ) + footer,
        1,
    )


def fresh_child_contexts(
    plan,
    *,
    stable_stage_id="stage-one",
    summary_mode="implement-by-phase",
    handoff_mode="implement-by-phase",
    inherited_matrix_row_ids=(),
    handoff_execution_disposition=None,
    handoff_override_digest=None,
    child_plan=None,
):
    raw_payload, _ = json.JSONDecoder().raw_decode(plan)
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    recommendation = parse_execution_recommendation_payload(
        raw_payload["execution_recommendation"], context="test recommendation"
    )
    assert recommendation is not None
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=orchestrator._plan_subject(plan)
    )
    phase = normalized.phases[0]
    parent_hash = approved_plan_hash(plan)
    identity = phase_identity(
        parent_issue=55,
        plan_hash=parent_hash,
        topology_source="approved-plan-v1",
        phase_index=1,
        phase=phase,
        stage_id=phase.stage_id,
        execution_strategy_contract_version=1,
    )
    child_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan=plan, phase=phase,
        created_so_far=(), phase_identity_value=identity,
        topology_source="approved-plan-v1", phase_index=1,
        phase_plan_hash=parent_hash, strategy="staged",
        recommendation_digest=normalized.recommendation_digest,
        execution_strategy_contract_version=1,
        inherited_matrix_row_ids=inherited_matrix_row_ids,
    )
    if stable_stage_id != phase.stage_id:
        marker = orchestrator.PHASE_IDENTITY_MARKER_RE.search(child_body)
        assert marker is not None
        marker_payload = json.loads(
            base64.urlsafe_b64decode(marker.group("payload").encode("ascii")).decode("utf-8")
        )
        marker_payload["stage_id"] = stable_stage_id
        encoded = base64.urlsafe_b64encode(
            json.dumps(marker_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).decode("ascii")
        child_body = child_body[:marker.start("payload")] + encoded + child_body[marker.end("payload"):]
    child = IssueContext(
        number=56, repo="OWNER/REPO", title="First contract", body=child_body,
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(comment(format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123", flow="approved-plan-implementation", plan_hash=parent_hash,
        )),),
    )
    parent_children = (
        CreatedPhaseIssue(phase=normalized.phases[0], issue_url=child.url, issue_number=56),
        CreatedPhaseIssue(
            phase=normalized.phases[1],
            issue_url="https://github.com/OWNER/REPO/issues/57",
            issue_number=57,
        ),
    )
    summary = format_decomposition_parent_summary(
        parent_issue=55, mode=summary_mode, plan_hash=parent_hash,
        created=parent_children, topology_source="approved-plan-v1",
        retained_parent_scope=retained, final_integration_work=normalized.final_integration_work,
        strategy="staged", execution_strategy_contract_version=1,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=orchestrator._plan_subject(plan),
        inherited_matrix_row_ids=inherited_matrix_row_ids,
    )
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=55, mode=handoff_mode, plan_hash=parent_hash,
        phase_index=1, created=parent_children[0], strategy="staged",
        topology_source="approved-plan-v1", execution_strategy_contract_version=1,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=orchestrator._plan_subject(plan),
        inherited_matrix_row_ids=inherited_matrix_row_ids,
        execution_disposition=handoff_execution_disposition,
        override_digest=handoff_override_digest,
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent scope.",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(comment(plan_record(plan)), comment(summary), comment(handoff)),
    )
    if child_plan is not None:
        child_hash = approved_plan_hash(child_plan)
        child = dataclasses.replace(
            child,
            comments=(
                comment(plan_record(child_plan)),
                comment(format_issue_pr_handoff_comment(
                    issue_number=56, pr_number=77,
                    pr_url="https://github.com/OWNER/REPO/pull/77",
                    pr_head_sha="abc123", flow="approved-plan-implementation",
                    plan_hash=child_hash,
                )),
            ),
        )
    return child, parent


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize(
    "fault", [None, "parent_checkpoint", "phase_identity", "phase_plan_hash", "child_plan"]
)
def test_separately_planned_child_preserves_both_plan_bindings(tmp_path, monkeypatch, entry, fault):
    parent_plan = "Approved parent plan.\n\n## Scope\n- Deliver three sequential stages."
    child_plan = "Approved child plan.\n\n## Scope\n- Implement only the reviewed strategy schema."
    parent_hash = approved_plan_hash(parent_plan)
    child_hash = approved_plan_hash(child_plan)
    assert parent_hash != child_hash
    phase = PlanPhase(
        title="Strategy schema", scope="Implement the contract.", non_goals="No routing.",
        dependency_notes="First stage.", rollout_risk="low", validation="Run focused tests.",
        parent_context="Approved first-stage scope.", automation="agent-pr", depends_on=(),
    )
    identity = phase_identity(parent_issue=55, plan_hash=parent_hash, topology_source="typed",
                              phase_index=1, phase=phase)
    child_body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan=parent_plan, phase=phase,
        created_so_far=(), phase_identity_value="wrong" if fault == "phase_identity" else identity,
        topology_source="typed", phase_index=1, phase_plan_hash=parent_hash,
    )
    if fault == "phase_plan_hash":
        child_body = replace_phase_plan_hash(child_body, "")
    handoff = format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123", flow="approved-plan-implementation", plan_hash=child_hash,
    )
    child = IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body=child_body,
        url="https://github.com/OWNER/REPO/issues/56",
        comments=tuple([comment(handoff)] if fault == "child_plan" else
                       [comment(plan_record(child_plan)), comment(handoff)]),
    )
    checkpoint = format_topology_checkpoint(TopologyCheckpoint(
        parent_issue=55, plan_hash=parent_hash, mode="decompose-only",
        topology_source="typed", phases=(phase,),
    ))
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent scope.",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=tuple([comment(plan_record(parent_plan))] +
                       ([] if fault == "parent_checkpoint" else [comment(checkpoint)])),
    )
    monkeypatch.setattr(orchestrator, "get_issue_context",
                        lambda runner, *, config, issue_number: child if issue_number == 56 else parent)
    runner = FakeRunner(
        pr_payload={"number": 77, "body": "Fixes #56", "url": "https://github.com/OWNER/REPO/pull/77"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    def run():
        if entry == "issue":
            return orchestrator.run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return orchestrator.run_pr_loop(runner, pr_number=77, config=config)

    if fault:
        with pytest.raises(AgentLoopError):
            run()
        assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
        assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    else:
        assert run() == 0
        prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
        plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
            "Target child/primary issue context", 1
        )[0]
        assert "Implement only the reviewed strategy schema." in plan_block
        assert "Deliver three sequential stages." not in plan_block


@pytest.mark.parametrize("stable_stage_id, should_fail", [("stage-one", False), ("wrong-stage", True)])
def test_cli_run_pr_loop_validates_fresh_child_phase_identity(
    tmp_path, monkeypatch, stable_stage_id, should_fail
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan, stable_stage_id=stable_stage_id)
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77,
            "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    if should_fail:
        with pytest.raises(AgentLoopError, match="stable ID disagrees with its ordinal"):
            orchestrator.run_pr_loop(runner, pr_number=77, config=config)
        assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    else:
        assert orchestrator.run_pr_loop(runner, pr_number=77, config=config) == 0
        assert any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


@pytest.mark.parametrize(
    "handoff_updates",
    [
        {},
        {"plan_hash": "different-plan"},
        {"phase_index": 2},
        {"stage_id": "stage-two"},
    ],
)
def test_cli_run_pr_loop_validates_parent_handoff_after_decompose_only_transition(
    tmp_path, monkeypatch, handoff_updates
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(
        plan,
        summary_mode="decompose-only",
        handoff_mode="implement-by-phase",
    )
    if handoff_updates:
        parent = dataclasses.replace(
            parent,
            comments=parent.comments[:-1]
            + (comment(replace_phase_handoff_payload(parent.comments[-1].body, **handoff_updates)),),
        )
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77,
            "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    if handoff_updates:
        with pytest.raises(AgentLoopError, match="implementation handoff"):
            orchestrator.run_pr_loop(runner, pr_number=77, config=config)
        assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    else:
        assert orchestrator.run_pr_loop(runner, pr_number=77, config=config) == 0
        assert any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def test_cli_run_pr_loop_rejects_mismatched_parent_handoff_after_decompose_only_transition(
    tmp_path, monkeypatch
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(
        plan,
        summary_mode="decompose-only",
        handoff_mode="decompose-only",
    )
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77,
            "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="implementation handoff"):
        orchestrator.run_pr_loop(runner, pr_number=77, config=config)
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def test_cli_run_pr_loop_requires_child_plan_without_parent_phase_handoff(
    tmp_path, monkeypatch
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(
        plan,
        summary_mode="decompose-only",
        handoff_mode="implement-by-phase",
    )
    # The child handoff deliberately carries the parent topology hash, but the
    # parent phase handoff is absent and the child has no approved plan.
    parent = dataclasses.replace(parent, comments=parent.comments[:-1])
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77,
            "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="approved child plan"):
        orchestrator.run_pr_loop(runner, pr_number=77, config=config)
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


@pytest.mark.parametrize("fault", [None, "parent_hash_copy", "missing_child_plan"])
def test_planning_handoff_pr_binds_to_distinct_reviewed_child_plan(
    tmp_path, monkeypatch, fault
):
    parent_plan = fresh_staged_plan(first_disposition="requires-child-planning")
    child_plan = "Reviewed child plan.\n\n## Scope\n- Implement the selected child design."
    child, parent = fresh_child_contexts(
        parent_plan,
        handoff_execution_disposition="requires-child-planning",
        child_plan=child_plan,
    )
    if fault == "parent_hash_copy":
        child = dataclasses.replace(
            child,
            comments=child.comments[:-1] + (
                comment(format_issue_pr_handoff_comment(
                    issue_number=56, pr_number=77,
                    pr_url="https://github.com/OWNER/REPO/pull/77",
                    pr_head_sha="abc123", flow="approved-plan-implementation",
                    plan_hash=approved_plan_hash(parent_plan),
                )),
            ),
        )
    elif fault == "missing_child_plan":
        child = dataclasses.replace(child, comments=child.comments[1:])
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77, "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    if fault == "parent_hash_copy":
        with pytest.raises(AgentLoopError, match="parent phase hash"):
            orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    elif fault == "missing_child_plan":
        with pytest.raises(AgentLoopError, match="no reviewed approved child plan"):
            orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    else:
        assert orchestrator.run_pr_loop(
            runner, pr_number=77, config=make_config(tmp_path)
        ) == 0
        prompt = next(
            command[-1] for command, _cwd in runner.commands
            if command[:2] == ["codex", "exec"]
        )
        assert "Implement the selected child design" in prompt
        assert "Fresh staged parent plan" not in prompt.split(
            "Approved implementation plan context", 1
        )[1].split("Target child/primary issue context", 1)[0]


def test_direct_and_legacy_handoff_pr_still_require_parent_phase_hash(
    tmp_path, monkeypatch
):
    plan = fresh_staged_plan(first_disposition="direct-implementation")
    child, parent = fresh_child_contexts(
        plan, handoff_execution_disposition="direct-implementation"
    )
    child = dataclasses.replace(
        child,
        comments=(comment(format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77", pr_head_sha="abc123",
            flow="approved-plan-implementation", plan_hash="wrong-child-hash",
        )),),
    )
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(pr_payload={"number": 77, "body": "Fixes #56", "url": "pr-url"})
    with pytest.raises(AgentLoopError, match="implementation handoff disagrees"):
        orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


@pytest.mark.parametrize("entry", ["issue", "pr"])
def test_contradictory_unbound_handoff_fails_closed_in_workflow_paths(
    tmp_path, monkeypatch, entry
):
    plan = fresh_staged_plan(first_disposition="direct-implementation")
    child, parent = fresh_child_contexts(
        plan, handoff_execution_disposition="requires-child-planning"
    )
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={"number": 77, "body": "Fixes #56", "url": "pr-url"},
        claude_outputs=["A coder must never run."],
        codex_outputs=["A reviewer must never run."],
    )
    with pytest.raises(AgentLoopError, match="carries no override digest"):
        if entry == "issue":
            orchestrator.run_issue_loop(
                runner, issue_number=56, config=make_config(tmp_path)
            )
        else:
            orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    assert runner.issue_comments == []


@pytest.mark.parametrize(
    ("declared", "effective"),
    [
        ("requires-child-planning", "direct-implementation"),
        ("direct-implementation", "requires-child-planning"),
    ],
)
def test_override_bound_handoff_survives_child_entry_and_pr_validation(
    tmp_path, monkeypatch, declared, effective
):
    plan = fresh_staged_plan(first_disposition=declared)
    plan_hash = approved_plan_hash(plan)
    override_body = format_child_disposition_override_comment(
        parent_issue=55, plan_hash=plan_hash, stage_id="stage-one",
        disposition=effective, rationale="Durably select the reviewed alternate route.",
    )
    records, ignored = parse_child_disposition_override_records(
        override_body, comment_locator="parent issue #55 comment"
    )
    assert not ignored and len(records) == 1
    child_plan = (
        "Reviewed child plan.\n\n## Scope\n- Implement the selected child design."
        if effective == "requires-child-planning" else None
    )
    child, parent = fresh_child_contexts(
        plan,
        handoff_execution_disposition=effective,
        handoff_override_digest=records[0].digest,
        child_plan=child_plan,
    )
    parent = dataclasses.replace(
        parent, comments=parent.comments + (comment(override_body),)
    )
    resolved = orchestrator._resolve_fresh_child_provenance(
        issue_context=child, parent_issue_context=parent
    )
    assert resolved is not None
    assert resolved.route.disposition == effective
    assert resolved.route.override_digest == records[0].digest

    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={"number": 77, "body": "Fixes #56", "url": "pr-url"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    assert orchestrator.run_pr_loop(
        runner, pr_number=77, config=make_config(tmp_path)
    ) == 0

    without_record = dataclasses.replace(parent, comments=parent.comments[:-1])
    with pytest.raises(AgentLoopError, match="undiscoverable"):
        orchestrator._resolve_fresh_child_provenance(
            issue_context=child, parent_issue_context=without_record
        )


@pytest.mark.parametrize("with_legacy_handoff", [False, True])
def test_legacy_ambiguous_child_workflow_plans_unless_legacy_handoff_exists(
    tmp_path, monkeypatch, with_legacy_handoff
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    if not with_legacy_handoff:
        parent = dataclasses.replace(parent, comments=parent.comments[:-1])
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    dispatched = []
    monkeypatch.setattr(orchestrator, "prepare_agent_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        orchestrator, "_dispatch_decomposition_child",
        lambda *_args, **kwargs: dispatched.append(kwargs) or 0,
    )
    runner = FakeRunner()
    if with_legacy_handoff:
        assert orchestrator.run_issue_loop(
            runner, issue_number=56, config=make_config(tmp_path)
        ) == 0
        assert dispatched and dispatched[0]["route"].is_direct
        assert dispatched[0]["existing_handoff"].execution_disposition is None
    else:
        with pytest.raises(AgentLoopError, match="--plan-first"):
            orchestrator.run_issue_loop(
                runner, issue_number=56, config=make_config(tmp_path)
            )
        assert dispatched == []


def _matrix_row(row_id, owner, *, expected="The transition completes."):
    return {
        "row_id": row_id,
        "label": row_id,
        "entry_path_or_mode": "review-only recovery",
        "initial_state": "review complete",
        "event": "repaired head passes",
        "expected_outcome": expected,
        "forbidden_side_effects": ["No stale merge."],
        "proposed_test_level": "orchestrator",
        "proposed_test_location": "tests/test_child_plan_provenance.py",
        "applicability": "applicable",
        "related_scope_item_ids": ["scope-1"],
        "execution_owner": owner,
    }


def test_m780_12_generated_handoff_records_owned_rows_and_separate_child_must_link_them():
    parent_matrix = {
        "applicability": "applicable",
        "rows": [
            _matrix_row("row-owned", "api"),
            _matrix_row("row-later", "later"),
        ],
        "important_exclusions": [],
    }
    assert risk_matrix_row_ids_for_owner(parent_matrix, "api") == ("row-owned",)
    child_matrix = {
        **parent_matrix,
        "rows": [
            {**_matrix_row("row-owned", "api")},
            _matrix_row("child-local", "api"),
        ],
    }
    assert validate_separately_planned_child_matrix(
        parent_matrix, child_matrix, execution_owner="api"
    ) == ("row-owned",)
    with pytest.raises(AgentLoopError, match="missing inherited"):
        validate_separately_planned_child_matrix(
            parent_matrix,
            {**child_matrix, "rows": [_matrix_row("child-local", "api")]},
            execution_owner="api",
        )

    phase = PlanPhase(
        title="API", scope="API.", non_goals="None.", dependency_notes="None.",
        rollout_risk="low", validation="Run the workflow test.", parent_context="Parent.",
        automation="agent-pr", depends_on=(), stage_id="api", position=1,
        deliverables=("API.",), acceptance_criteria=("Done.",),
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO", parent_issue=55, approved_plan="Parent.", phase=phase,
        created_so_far=(), phase_identity_value="identity", topology_source="approved-plan-v1",
        phase_index=1, phase_plan_hash="a" * 16, strategy="staged",
        recommendation_digest="b" * 64, execution_strategy_contract_version=1,
        inherited_matrix_row_ids=("row-owned",),
    )
    assert "Inherited parent risk-matrix obligations" in body
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=55, mode="implement-by-phase", plan_hash="a" * 16,
        phase_index=1,
        created=CreatedPhaseIssue(phase=phase, issue_url="https://github.com/OWNER/REPO/issues/56", issue_number=56),
        strategy="staged", topology_source="approved-plan-v1",
        execution_strategy_contract_version=1, recommendation_digest="b" * 64,
        plan_subject="c" * 64, inherited_matrix_row_ids=("row-owned",),
    )
    decoded = _decode_phase_implementation_handoff_metadata(
        PHASE_IMPLEMENTATION_MARKER_RE.search(handoff).group("payload")
    )
    assert decoded.inherited_matrix_row_ids == ("row-owned",)


def test_m780_12_generated_child_dispatch_enforces_owned_rows_and_keeps_later_rows_pending(
    tmp_path, monkeypatch
):
    plan = fresh_staged_matrix_plan()
    child, parent = fresh_child_contexts(
        plan, inherited_matrix_row_ids=("row-stage-one",)
    )
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77, "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    assert orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Row row-stage-one" in prompt
    assert "Row row-stage-two" in prompt
    assert "[enforceable]" in prompt
    assert "[read-only pending obligation]" in prompt


def test_m780_12_separately_planned_child_requires_parent_row_and_uses_child_provenance(
    tmp_path, monkeypatch
):
    parent_plan = fresh_staged_matrix_plan()
    child, parent = fresh_child_contexts(
        parent_plan,
        summary_mode="decompose-only",
        handoff_mode="implement-by-phase",
        inherited_matrix_row_ids=("row-stage-one",),
    )
    inherited_row = next(
        row for row in json.JSONDecoder().raw_decode(parent_plan)[0]["risk_test_matrix"]["rows"]
        if row["row_id"] == "row-stage-one"
    )
    child_matrix = {
        "applicability": "applicable",
        # A separately planned child owns its own matrix. Its one-shot owner
        # is valid for the child plan even though the inherited parent row was
        # allocated to the parent topology's stage-one owner.
        "rows": [
            {**inherited_row, "execution_owner": "one-shot"},
            _matrix_row("child-local", "one-shot"),
        ],
        "important_exclusions": ["Unrelated flag combinations are not required."],
    }
    child_plan = "Approved child plan.\n\n" + __import__(
        "coding_review_agent_loop.comment_rendering", fromlist=["render_risk_test_matrix_section"]
    ).render_risk_test_matrix_section(parse_risk_test_matrix(child_matrix))
    child_hash = approved_plan_hash(child_plan)
    child_handoff = format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77", pr_head_sha="abc123",
        flow="approved-plan-implementation", plan_hash=child_hash,
    )
    child = dataclasses.replace(
        child,
        comments=(comment(plan_record(child_plan)), comment(child_handoff)),
    )
    parent = dataclasses.replace(parent, comments=parent.comments[:-1])
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={
            "number": 77, "body": "Fixes #56",
            "url": "https://github.com/OWNER/REPO/pull/77",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    assert orchestrator.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Row row-stage-one" in prompt
    assert "[enforceable]" in prompt
    assert "Row child-local" in prompt
