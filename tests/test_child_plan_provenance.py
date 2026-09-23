import base64
import dataclasses
import json
import re

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


def fresh_staged_matrix_plan(*, first_disposition=None):
    """Fresh staged parent fixture with explicit owned transition rows."""
    plan = fresh_staged_plan(first_disposition=first_disposition)
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
    recommendation_payload=None,
):
    # A rendered canonical plan carries no leading JSON; callers supply the
    # recommendation its sidecar encodes.
    raw_payload = (
        {"execution_recommendation": recommendation_payload}
        if recommendation_payload is not None
        else json.JSONDecoder().raw_decode(plan)[0]
    )
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


@pytest.mark.parametrize("legacy", [False, True])
def test_direct_and_legacy_handoff_pr_still_require_parent_phase_hash(
    tmp_path, monkeypatch, legacy
):
    plan = fresh_staged_plan(
        first_disposition=None if legacy else "direct-implementation"
    )
    child, parent = fresh_child_contexts(
        plan,
        handoff_execution_disposition=(
            None if legacy else "direct-implementation"
        ),
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
    raw_payload, _ = json.JSONDecoder().raw_decode(plan)
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    recommendation = parse_execution_recommendation_payload(
        raw_payload["execution_recommendation"], context="test recommendation"
    )
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=_plan_subject(plan)
    )
    created = (
        CreatedPhaseIssue(normalized.phases[0], child.url, child.number),
        CreatedPhaseIssue(
            normalized.phases[1], "https://github.com/OWNER/REPO/issues/57", 57
        ),
    )
    monkeypatch.setattr(
        orchestrator, "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "create_decomposition_child_issues",
        lambda *_args, **_kwargs: created,
    )
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    assert orchestrator._preflight_fresh_staged_topology(
        FakeRunner(), issue_number=55, approved_plan=plan,
        config=make_config(tmp_path), issue_context=parent,
        mode="implement-by-phase", normalized_topology=(normalized, retained),
    ) == created

    resolved = orchestrator._resolve_fresh_child_provenance(
        issue_context=child, parent_issue_context=parent
    )
    assert resolved is not None
    assert resolved.route.disposition == effective
    assert resolved.route.override_digest == records[0].digest

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


def oversized_fresh_staged_plan():
    """A fresh staged plan whose retained-parent excerpt cannot be published whole."""
    plan = fresh_staged_plan()
    payload, end = json.JSONDecoder().raw_decode(plan)
    payload["summary"] = "Fresh staged parent plan. " + "retained detail " * 6000
    return json.dumps(payload) + plan[end:]


def test_rerun_preflight_reconciles_a_published_shortened_retained_excerpt(
    tmp_path, monkeypatch
):
    """#907: a summary published for a large plan carries a shortened excerpt.

    The scope recomputed from the approved plan on a rerun always carries the
    full text, so a plain equality check would wedge exactly the parents the
    bounding makes publishable.
    """
    from coding_review_agent_loop.decomposition import (
        EXECUTION_TOPOLOGY_SOURCE, _decode_metadata,
    )
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    plan = oversized_fresh_staged_plan()
    plan_hash = approved_plan_hash(plan)
    payload, _end = json.JSONDecoder().raw_decode(plan)
    recommendation = parse_execution_recommendation_payload(
        payload["execution_recommendation"], context="test recommendation"
    )
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=_plan_subject(plan)
    )
    created = (
        CreatedPhaseIssue(normalized.phases[0], "https://github.com/OWNER/REPO/issues/56", 56),
        CreatedPhaseIssue(normalized.phases[1], "https://github.com/OWNER/REPO/issues/57", 57),
    )
    summary = format_decomposition_parent_summary(
        parent_issue=55,
        mode="implement-by-phase",
        plan_hash=plan_hash,
        created=created,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        retained_parent_scope=retained,
        final_integration_work=normalized.final_integration_work,
        strategy=normalized.strategy,
        execution_strategy_contract_version=normalized.execution_strategy_contract_version,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=_plan_subject(plan),
    )
    from coding_review_agent_loop.decomposition import DECOMPOSITION_MARKER_RE

    recorded = _decode_metadata(
        DECOMPOSITION_MARKER_RE.search(summary).group("payload")
    ).retained_parent_scope
    # The published record really does carry a shortened excerpt.
    assert recorded is not None
    assert len(recorded.excerpt) < len(retained.excerpt)

    parent = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(comment(plan_record(plan)), comment(summary)),
    )
    monkeypatch.setattr(
        orchestrator, "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "create_decomposition_child_issues",
        lambda *_args, **_kwargs: created,
    )

    assert orchestrator._preflight_fresh_staged_topology(
        FakeRunner(), issue_number=55, approved_plan=plan,
        config=make_config(tmp_path), issue_context=parent,
        mode="implement-by-phase", normalized_topology=(normalized, retained),
    ) == created


def test_rerun_preflight_still_rejects_a_foreign_retained_excerpt(tmp_path, monkeypatch):
    """A recorded excerpt that is not an opening of the approved plan still fails."""
    from coding_review_agent_loop.decomposition import EXECUTION_TOPOLOGY_SOURCE

    plan = oversized_fresh_staged_plan()
    plan_hash = approved_plan_hash(plan)
    payload, _end = json.JSONDecoder().raw_decode(plan)
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    recommendation = parse_execution_recommendation_payload(
        payload["execution_recommendation"], context="test recommendation"
    )
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=_plan_subject(plan)
    )
    created = (
        CreatedPhaseIssue(normalized.phases[0], "https://github.com/OWNER/REPO/issues/56", 56),
        CreatedPhaseIssue(normalized.phases[1], "https://github.com/OWNER/REPO/issues/57", 57),
    )
    foreign = dataclasses.replace(retained, excerpt="A different approved plan's scope.")
    summary = format_decomposition_parent_summary(
        parent_issue=55,
        mode="implement-by-phase",
        plan_hash=plan_hash,
        created=created,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        retained_parent_scope=foreign,
        final_integration_work=normalized.final_integration_work,
        strategy=normalized.strategy,
        execution_strategy_contract_version=normalized.execution_strategy_contract_version,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=_plan_subject(plan),
    )
    parent = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(comment(plan_record(plan)), comment(summary)),
    )
    monkeypatch.setattr(
        orchestrator, "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "create_decomposition_child_issues",
        lambda *_args, **_kwargs: created,
    )

    with pytest.raises(AgentLoopError, match="disagrees with the approved normalized topology"):
        orchestrator._preflight_fresh_staged_topology(
            FakeRunner(), issue_number=55, approved_plan=plan,
            config=make_config(tmp_path), issue_context=parent,
            mode="implement-by-phase", normalized_topology=(normalized, retained),
        )


def test_rerun_preflight_rejects_an_appended_retained_excerpt(tmp_path, monkeypatch):
    """#907: a record carrying the plan text plus extra content is divergent."""
    from coding_review_agent_loop.decomposition import EXECUTION_TOPOLOGY_SOURCE
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    plan = fresh_staged_plan()
    payload, _end = json.JSONDecoder().raw_decode(plan)
    recommendation = parse_execution_recommendation_payload(
        payload["execution_recommendation"], context="test recommendation"
    )
    normalized, retained = normalize_execution_recommendation(
        recommendation, approved_plan=plan, plan_subject=_plan_subject(plan)
    )
    created = (
        CreatedPhaseIssue(normalized.phases[0], "https://github.com/OWNER/REPO/issues/56", 56),
        CreatedPhaseIssue(normalized.phases[1], "https://github.com/OWNER/REPO/issues/57", 57),
    )
    appended = dataclasses.replace(
        retained, excerpt=retained.excerpt + "\n\nForeign appended scope."
    )
    summary = format_decomposition_parent_summary(
        parent_issue=55,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=created,
        topology_source=EXECUTION_TOPOLOGY_SOURCE,
        retained_parent_scope=appended,
        final_integration_work=normalized.final_integration_work,
        strategy=normalized.strategy,
        execution_strategy_contract_version=normalized.execution_strategy_contract_version,
        recommendation_digest=normalized.recommendation_digest,
        plan_subject=_plan_subject(plan),
    )
    # The record is short enough to carry the divergent text verbatim.
    assert "Foreign appended scope." in summary
    parent = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(comment(plan_record(plan)), comment(summary)),
    )
    monkeypatch.setattr(
        orchestrator, "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator, "create_decomposition_child_issues",
        lambda *_args, **_kwargs: created,
    )

    with pytest.raises(AgentLoopError, match="disagrees with the approved normalized topology"):
        orchestrator._preflight_fresh_staged_topology(
            FakeRunner(), issue_number=55, approved_plan=plan,
            config=make_config(tmp_path), issue_context=parent,
            mode="implement-by-phase", normalized_topology=(normalized, retained),
        )


class _ManagedResumeCaptured(Exception):
    """Sentinel raised at handoff revalidation, after plan recovery."""


def legacy_plan_record(plan, *, subject):
    """A pre-canonical free-form plan record carrying a divergent subject."""
    return _attach_round_metadata(plan, PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1, subject=subject,
    ))


def managed_ci_resume(
    tmp_path,
    monkeypatch,
    *,
    child,
    parent,
    parent_issue_context,
    approved_plan_context=None,
):
    """Drive the managed-CI ordinary resume up to handoff revalidation."""
    fetched = []

    def fake_get_issue_context(_runner, *, config, issue_number):
        fetched.append(issue_number)
        return child if issue_number == child.number else parent

    monkeypatch.setattr(orchestrator, "get_issue_context", fake_get_issue_context)
    monkeypatch.setattr(orchestrator, "validate_open_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(
        orchestrator,
        "recover_issue_created_handoff",
        lambda *_a, **_k: orchestrator.AuthenticatedIssueCreatedHandoff(
            pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
            head_sha="abc123", branch="agent-loop/managed-56",
            trusted_actor_login="agent-loop", trusted_actor_id=1,
            protection_mode="voluntary", override_nonce="opening-nonce",
        ),
    )
    captured = {}

    def revalidate(*_args, **kwargs):
        captured.update(kwargs)
        raise _ManagedResumeCaptured

    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", revalidate)
    runner = FakeRunner(
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        reviewer=("codex",),
    )

    def run():
        return orchestrator.run_pr_loop(
            runner, pr_number=77, config=config,
            parent_issue_context=parent_issue_context,
            approved_plan_context=approved_plan_context,
        )

    return run, runner, captured, fetched


def assert_no_agent_process(runner):
    assert not any(
        command[:1] in (["claude"], ["codex"], ["gemini"])
        for command, _cwd in runner.commands
    )


def test_managed_ci_resume_recovers_staged_parent_held_plan(tmp_path, monkeypatch):
    """A staged child with only its handoff record resumes from the parent plan."""
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    assert len(child.comments) == 1
    run, runner, captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent, parent_issue_context=parent
    )

    with pytest.raises(_ManagedResumeCaptured):
        run()

    assert captured["handoff"].approved_plan_hash == approved_plan_hash(plan)
    assert_no_agent_process(runner)


def test_managed_ci_resume_refreshes_stale_parent_snapshot_once(tmp_path, monkeypatch):
    """A caller snapshot predating plan approval is refreshed before the lookup."""
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    stale_parent = dataclasses.replace(parent, comments=parent.comments[1:])
    assert orchestrator.recover_approved_plan_context(
        stale_parent.comments, expected_hash=approved_plan_hash(plan)
    ).is_available is False
    run, runner, captured, fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent,
        parent_issue_context=stale_parent,
    )

    with pytest.raises(_ManagedResumeCaptured):
        run()

    assert captured["handoff"].approved_plan_hash == approved_plan_hash(plan)
    assert fetched.count(parent.number) == 1
    assert_no_agent_process(runner)


def test_managed_ci_resume_without_parent_context_still_fails_closed(tmp_path, monkeypatch):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    run, runner, _captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent, parent_issue_context=None
    )

    with pytest.raises(
        AgentLoopError,
        match="Managed-CI ordinary resume could not recover the canonical approved plan.",
    ):
        run()

    assert_no_agent_process(runner)


def test_managed_ci_resume_parent_without_matching_hash_fails_closed(tmp_path, monkeypatch):
    """The recovered parent plan must still hash-match the canonical handoff."""
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    other_plan = "Unrelated approved plan.\n\n### Plan steps\n1. Do something else."
    mismatched_parent = dataclasses.replace(
        parent, comments=(comment(plan_record(other_plan)),) + parent.comments[1:]
    )
    run, runner, _captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=mismatched_parent,
        parent_issue_context=mismatched_parent,
    )

    with pytest.raises(
        AgentLoopError,
        match="Managed-CI ordinary resume could not recover the canonical approved plan.",
    ):
        run()

    assert_no_agent_process(runner)


def test_managed_ci_resume_subject_rejected_child_record_permits_parent_fallback(
    tmp_path, monkeypatch
):
    """A hash-matching child record rejected on its derived subject is not adopted."""
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    child = dataclasses.replace(
        child,
        comments=(comment(legacy_plan_record(plan, subject="divergent-subject")),)
        + child.comments,
    )
    child_only = orchestrator.recover_approved_plan_context(
        child.comments, expected_hash=approved_plan_hash(plan)
    )
    assert not child_only.is_available
    assert not child_only.has_matching_candidate
    run, runner, captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent, parent_issue_context=parent
    )

    with pytest.raises(_ManagedResumeCaptured):
        run()

    assert captured["handoff"].approved_plan_hash == approved_plan_hash(plan)
    assert_no_agent_process(runner)


def test_managed_ci_resume_divergent_child_records_never_consult_parent(
    tmp_path, monkeypatch
):
    """Divergent hash-matching child records keep failing closed."""
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    seen = []

    def divergent(comments, **kwargs):
        seen.append(comments)
        return orchestrator.ApprovedPlanContext(
            plan_hash=kwargs["expected_hash"],
            availability="mismatched",
            has_matching_candidate=True,
            diagnostic="Multiple divergent canonical plan records match handoff hash.",
        )

    monkeypatch.setattr(orchestrator, "recover_approved_plan_context", divergent)
    run, runner, _captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent, parent_issue_context=parent
    )

    with pytest.raises(
        AgentLoopError,
        match="Managed-CI ordinary resume could not recover the canonical approved plan.",
    ):
        run()

    assert len(seen) == 1
    assert seen[0] is child.comments
    assert_no_agent_process(runner)


def test_managed_ci_resume_rejects_supplied_scope_conflicting_with_parent_plan(
    tmp_path, monkeypatch
):
    plan = fresh_staged_plan()
    child, parent = fresh_child_contexts(plan)
    supplied = orchestrator.make_approved_plan_context(
        "Different plan.\n\n### Plan steps\n1. Change the boundary.",
        source_locator="test",
    )
    run, runner, _captured, _fetched = managed_ci_resume(
        tmp_path, monkeypatch, child=child, parent=parent, parent_issue_context=parent,
        approved_plan_context=supplied,
    )

    with pytest.raises(
        AgentLoopError, match="does not match the canonical issue plan"
    ):
        run()

    assert_no_agent_process(runner)


def test_child_planning_cycle_resets_the_staged_planning_policy(tmp_path, monkeypatch):
    """`derived-configs-neutralize-planning-policy` (#905, from #841)."""
    parent_config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
        plan_review_force_full=True,
    )
    captured = {}

    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: IssueContext(
            number=issue_number, repo="OWNER/REPO", title="Child",
            body="Child", url="child-url", comments=(),
        ),
    )

    def fake_plan_first_loop(runner, **kwargs):
        captured["config"] = kwargs["config"]
        return 0

    monkeypatch.setattr(orchestrator, "_run_plan_first_loop", fake_plan_first_loop)

    assert orchestrator._run_child_planning_cycle(
        FakeRunner(),
        config=parent_config,
        memory=None,
        usage_context=orchestrator._new_usage_context(parent_config),
        parent_issue=55,
        child_issue_number=56,
    ) == 0

    child_config = captured["config"]
    assert child_config.plan_execution_mode == "auto"
    # The child plan review stays full-board by configuration reset, not by
    # convention: the parent's staged policy is never inherited.
    assert child_config.plan_review_policy == "all-reviewers"
    assert child_config.primary_plan_reviewer is None
    assert child_config.plan_review_force_full is False
    assert child_config.reviewer == parent_config.reviewer


# --- #931: field-classified inherited-row comparison -----------------------

from coding_review_agent_loop.decomposition import (  # noqa: E402
    INHERITED_MATRIX_ROUTE_FORWARD,
    InheritedMatrixBinding,
    InheritedRowDifference,
    inherited_matrix_reviewed_deltas,
)
from coding_review_agent_loop.round_state import sanitize_plan_validation_diagnostic  # noqa: E402

CASE_SENSITIVE_EFFECT = "Do not unset $HOME_Dir or write /Var/Agent/State."
WHITESPACE_SENSITIVE_EFFECT = "Do not run `git  push   --force origin main`."


def _matrix(*rows):
    return {"applicability": "applicable", "rows": list(rows), "important_exclusions": []}


def _parent_row(**overrides):
    return {
        **_matrix_row("row-owned", "api"),
        "forbidden_side_effects": [CASE_SENSITIVE_EFFECT, WHITESPACE_SENSITIVE_EFFECT],
        **overrides,
    }


def _check(child_row, *, parent_row=None):
    # The child-local row keeps the child matrix parseable when the inherited
    # row is lowered to not-applicable.
    return validate_separately_planned_child_matrix(
        _matrix(parent_row or _parent_row()),
        _matrix(child_row, _matrix_row("child-local", "api")),
        execution_owner="api",
    )


def _deltas(child_row, *, parent_row=None):
    return inherited_matrix_reviewed_deltas(
        _matrix(parent_row or _parent_row()),
        _matrix(child_row, _matrix_row("child-local", "api")),
        execution_owner="api",
    )


def test_m931_identical_and_free_field_differences_pass_without_reviewed_deltas():
    for child in (
        _parent_row(),
        _parent_row(label="A reworded, clearer label"),
        _parent_row(execution_owner="one-shot", related_scope_item_ids=["child-scope-9"]),
        _parent_row(
            forbidden_side_effects=[WHITESPACE_SENSITIVE_EFFECT, CASE_SENSITIVE_EFFECT]
        ),
    ):
        assert _check(child) == ("row-owned",)
        assert _deltas(child) == ()


def test_m931_refined_row_passes_and_reports_exactly_the_reviewed_deltas():
    parent = _parent_row()
    child = _parent_row(
        label="Reworded",
        applicability="required",
        forbidden_side_effects=[
            "No duplicate comment.", WHITESPACE_SENSITIVE_EFFECT, CASE_SENSITIVE_EFFECT,
        ],
        entry_path_or_mode=parent["entry_path_or_mode"] + " via parent dispatch",
        initial_state="Given a draft PR: " + parent["initial_state"],
        event="first " + parent["event"] + " twice",
        expected_outcome=parent["expected_outcome"] + " Exactly one record is written.",
        proposed_test_level="unit",
        proposed_test_location="tests/test_other.py",
    )
    assert _check(child) == ("row-owned",)
    assert _deltas(child) == (
        InheritedRowDifference("row-owned", "applicability", "applicable", "required", "applicability raised"),
        InheritedRowDifference(
            "row-owned", "forbidden_side_effects", "", "No duplicate comment.",
            "forbidden side effect added",
        ),
        *(
            InheritedRowDifference("row-owned", field, parent[field], child[field], "coverage text extended")
            for field in ("entry_path_or_mode", "initial_state", "event", "expected_outcome")
        ),
        InheritedRowDifference("row-owned", "proposed_test_level", "orchestrator", "unit", "proposed test changed"),
        InheritedRowDifference(
            "row-owned", "proposed_test_location", "tests/test_child_plan_provenance.py",
            "tests/test_other.py", "proposed test changed",
        ),
    )


@pytest.mark.parametrize(
    "parent_applicability, child_applicability",
    [("required", "applicable"), ("required", "not-applicable"), ("applicable", "not-applicable")],
)
def test_m931_lowered_applicability_names_row_field_values_and_route(
    parent_applicability, child_applicability
):
    with pytest.raises(AgentLoopError) as error:
        _check(
            _parent_row(applicability=child_applicability),
            parent_row=_parent_row(applicability=parent_applicability),
        )
    text = str(error.value)
    assert (
        f"row-owned: applicability weakened (parent={parent_applicability}, "
        f"child={child_applicability})"
    ) in text
    assert text.endswith(INHERITED_MATRIX_ROUTE_FORWARD)


@pytest.mark.parametrize(
    "child_effects, dropped",
    [
        ([CASE_SENSITIVE_EFFECT, "An unrelated added entry."], [WHITESPACE_SENSITIVE_EFFECT]),
        (
            [CASE_SENSITIVE_EFFECT.replace("$HOME_Dir", "$home_dir"), WHITESPACE_SENSITIVE_EFFECT,
             "An unrelated added entry."],
            [CASE_SENSITIVE_EFFECT],
        ),
        (
            [CASE_SENSITIVE_EFFECT, WHITESPACE_SENSITIVE_EFFECT.replace("git  push   --force", "git push --force")],
            [WHITESPACE_SENSITIVE_EFFECT],
        ),
        (["Something else entirely."], [CASE_SENSITIVE_EFFECT, WHITESPACE_SENSITIVE_EFFECT]),
    ],
)
def test_m931_dropped_or_normalized_forbidden_side_effect_is_rejected(child_effects, dropped):
    with pytest.raises(AgentLoopError) as error:
        _check(_parent_row(forbidden_side_effects=child_effects))
    text = str(error.value)
    for effect in dropped:
        assert f'row-owned: forbidden_side_effects dropped "{effect}"' in text
    assert text.count("forbidden_side_effects dropped") == len(dropped)
    assert sanitize_plan_validation_diagnostic(text) == text


def test_m931_diagnostic_neutralizes_reserved_marker_text():
    marker = "Never post <!-- AGENT_LOOP_META: abc --> early."
    with pytest.raises(AgentLoopError) as error:
        _check(
            _parent_row(forbidden_side_effects=["kept"]),
            parent_row=_parent_row(forbidden_side_effects=[marker, "kept"]),
        )
    assert "AGENT_LOOP_META" not in str(error.value)
    assert "row-owned: forbidden_side_effects dropped" in str(error.value)


@pytest.mark.parametrize(
    "field", ["entry_path_or_mode", "initial_state", "event", "expected_outcome"]
)
@pytest.mark.parametrize("rewrite", ["substitute", "case", "whitespace"])
def test_m931_substituted_scenario_coverage_is_rejected_per_field(field, rewrite):
    parent = _parent_row(
        entry_path_or_mode="issue mode", initial_state="Review  complete",
        event="Repaired head passes", expected_outcome="The Transition completes.",
    )
    replacement = {
        "substitute": {"entry_path_or_mode": "pr mode"}.get(field, "something different"),
        "case": parent[field].swapcase(),
        "whitespace": parent[field].replace(" ", "  ", 1) if "  " not in parent[field]
        else parent[field].replace("  ", " "),
    }[rewrite]
    assert replacement != parent[field]
    with pytest.raises(AgentLoopError) as error:
        _check({**parent, field: replacement}, parent_row=parent)
    text = str(error.value)
    assert f"row-owned: {field} replaced" in text
    assert "must be kept verbatim and refinements added after it" in text
    # An unchanged applicability and side-effect list masks nothing.
    assert text.count("row-owned:") == 1


def test_m931_parent_scenario_field_at_the_size_bound_admits_only_the_identical_value():
    full = "x" * 1024
    parent = _parent_row(event=full)
    assert _check(dict(parent), parent_row=parent) == ("row-owned",)
    with pytest.raises(AgentLoopError, match="row-owned: event replaced"):
        _check({**parent, "event": "y" + full[1:]}, parent_row=parent)
    # Extending would exceed the wire bound, so no extension is expressible.
    with pytest.raises(AgentLoopError, match="1024-byte bound"):
        _check({**parent, "event": full + "!"}, parent_row=parent)


def test_m931_many_weakened_rows_fit_the_diagnostic_bound_with_whole_entries():
    parents, children = [], []
    for index in range(24):
        row = {
            **_matrix_row(f"row-{index:02d}", "api"),
            "forbidden_side_effects": [f"effect {index}-{n} " + "z" * 400 for n in range(6)],
            "event": f"event {index} " + "e" * 400,
        }
        parents.append(row)
        children.append({**row, "forbidden_side_effects": ["other"], "event": "replaced"})
    with pytest.raises(AgentLoopError) as error:
        validate_separately_planned_child_matrix(
            _matrix(*parents), _matrix(*children), execution_owner="api"
        )
    text = str(error.value)
    assert len(text) <= 4096
    assert sanitize_plan_validation_diagnostic(text) == text
    lines = text.split("\n")
    assert lines[-1] == INHERITED_MATRIX_ROUTE_FORWARD
    assert re.fullmatch(r"- and \d+ more weakened fields in \d+ rows", lines[-2])
    for entry in lines[1:-2]:
        assert entry.startswith("- row-") and entry.endswith(("(no exactly equal child entry)", "after it"))
    emitted = len(lines) - 3
    remainder = int(lines[-2].split()[2])
    assert emitted + remainder == 24 * 7


def test_m931_omission_checks_are_unchanged():
    parent = _matrix(_parent_row())
    with pytest.raises(AgentLoopError, match="omitted the approved parent risk matrix"):
        validate_separately_planned_child_matrix(parent, None, execution_owner="api")
    with pytest.raises(
        AgentLoopError, match="missing inherited parent matrix row IDs: row-owned"
    ):
        validate_separately_planned_child_matrix(
            parent, _matrix(_matrix_row("child-local", "api")), execution_owner="api"
        )
    assert validate_separately_planned_child_matrix(parent, None, execution_owner="other") == ()


# --- #931: planning-time enforcement through a bounded replan ---------------

from agent_loop_helpers import structured_plan_review  # noqa: E402
from coding_review_agent_loop.errors import AgentInvocationError  # noqa: E402
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState  # noqa: E402
from coding_review_agent_loop.protocol import validate_structured_plan_state  # noqa: E402
from coding_review_agent_loop.runner import CommandResult  # noqa: E402

PLAN_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
WEAK_SUMMARY = "WEAKENED-CANDIDATE-SUMMARY"


def _child_row(**overrides):
    return _parent_row(execution_owner="one-shot", **overrides)


def _weak_child_row():
    return _child_row(forbidden_side_effects=[CASE_SENSITIVE_EFFECT])


def _child_plan_payload(row, *, summary="Child plan."):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["summary"] = summary
    payload["risk_test_matrix"] = _matrix(row)
    return payload


def _child_plan_state(row, **kwargs):
    return json.dumps(_child_plan_payload(row, **kwargs)) + PLAN_FOOTER


def _binding():
    return InheritedMatrixBinding(
        parent_issue=55, stage_id="api", parent_matrix=_matrix(_parent_row())
    )


class _ChildPlanningRunner(FakeRunner):
    """FakeRunner that also serves the authenticated diagnostic-record seam."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diagnostic_posts = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, json.dumps({"login": "agent", "id": 7}), "", 0)
        if command[:4] == ["gh", "api", "--method", "POST"] and command[4:5] == [
            "repos/OWNER/REPO/issues/56/comments"
        ]:
            recorded, cwd_path = self._record_command(args, cwd)
            body = json.loads(input_text or "{}")["body"]
            self.diagnostic_posts.append(body)
            return CommandResult(
                recorded, cwd_path,
                json.dumps({
                    "id": 901, "created_at": "2026-09-17T05:30:00Z", "body": body,
                    "user": {"login": "agent", "id": 7},
                }),
                "", 0,
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _bind_child_planning(monkeypatch, binding=None):
    """Run the plan-first loop as the child cycle of a bound stage."""
    real = orchestrator._run_plan_first_loop
    monkeypatch.setattr(
        orchestrator,
        "_run_plan_first_loop",
        lambda runner, **kwargs: real(
            runner, **{**kwargs, "inherited_matrix_binding": binding or _binding()}
        ),
    )

    def forbid_repair(*args, **kwargs):
        raise AssertionError("a rejected inherited candidate must never reach the repair model")

    monkeypatch.setattr(orchestrator, "_run_structured_repair", forbid_repair)


def _agent_prompts(runner, agent):
    head = [agent] if agent == "claude" else [agent, "exec"]
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[: len(head)] == head]


def _plan_config(tmp_path, **kwargs):
    return make_config(
        tmp_path, max_rounds=3, plan_execution_mode="plan-only",
        execution_strategy_contract_required=True, **kwargs,
    )


def _published(runner):
    return "\n".join(str(item["body"]) for item in runner.issue_comments)


def test_m931_weakened_fresh_candidate_is_replanned_before_publication(tmp_path, monkeypatch):
    _bind_child_planning(monkeypatch)
    good_row = _child_row(expected_outcome=_parent_row()["expected_outcome"] + " Once only.")
    runner = _ChildPlanningRunner(
        claude_outputs=[
            _child_plan_state(_weak_child_row(), summary=WEAK_SUMMARY),
            _child_plan_state(good_row),
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0

    planner_prompts = _agent_prompts(runner, "claude")
    assert len(planner_prompts) == 2
    for prompt in planner_prompts:
        assert "Inherited parent risk-matrix obligations" in prompt
        assert json.dumps(WHITESPACE_SENSITIVE_EFFECT) in prompt
    assert "forbidden_side_effects dropped" not in planner_prompts[0]
    assert f'row-owned: forbidden_side_effects dropped "{WHITESPACE_SENSITIVE_EFFECT}"' in planner_prompts[1]
    assert "Trusted orchestration correction record" in planner_prompts[1]
    # The rejected candidate is never rendered, posted, recorded, or reviewed.
    assert WEAK_SUMMARY not in _published(runner)
    assert runner.diagnostic_posts == []
    review_prompts = _agent_prompts(runner, "codex")
    assert len(review_prompts) == 1
    assert WEAK_SUMMARY not in review_prompts[0]
    assert "Inherited coverage delta" in review_prompts[0]
    assert "field `expected_outcome` (coverage text extended)" in review_prompts[0]
    assert json.dumps(good_row["expected_outcome"]) in review_prompts[0]


def _revision_payload(row, *, summary, changes=()):
    payload = _child_plan_payload(row, summary=summary)
    payload["kind"] = "plan_revision"
    payload["prior_plan_item_dispositions"] = [
        {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
    ]
    payload["risk_test_matrix_changes"] = list(changes)
    return json.dumps(payload) + PLAN_FOOTER


def _semantic_patch(base_plan_text, row, *, summary):
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(base_plan_text), round_number=1
    )
    operations = [{"op": "replace", "field": "summary", "value": summary}]
    if row is not None:
        operations.append({
            "op": "matrix_edit", "row_id": "row-owned", "row": row,
            "rationale": "Revise the inherited row.",
        })
    return json.dumps({
        "schema_version": 1, "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1, "state": "blocking",
        "summary": "Patch.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ],
        "base_round_number": 1, "base_state_identity": base.state_identity,
        "operations": operations,
    }) + PLAN_FOOTER


@pytest.mark.parametrize("form", ["semantic-patch", "full-state"])
def test_m931_weakened_revision_candidate_is_replanned_over_the_unchanged_base(
    tmp_path, monkeypatch, form
):
    _bind_child_planning(monkeypatch)
    fresh = _child_plan_state(_child_row())
    if form == "semantic-patch":
        weak = _semantic_patch(fresh, _weak_child_row(), summary=WEAK_SUMMARY)
        good = _semantic_patch(fresh, None, summary="Corrected revision.")
    else:
        # Without an assembled sidecar the revision uses the full-state form.
        monkeypatch.setattr(orchestrator, "make_assembled_plan_sidecar", lambda *a, **k: None)
        weak = _revision_payload(
            _weak_child_row(), summary=WEAK_SUMMARY,
            changes=[{"operation": "change", "row_ids": ["row-owned"], "rationale": "Trim."}],
        )
        good = _revision_payload(_child_row(), summary="Corrected revision.")
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh, weak, good],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0

    planner_prompts = _agent_prompts(runner, "claude")
    assert len(planner_prompts) == 3
    assert "forbidden_side_effects dropped" not in planner_prompts[1]
    assert "row-owned: forbidden_side_effects dropped" in planner_prompts[2]
    assert "Inherited parent risk-matrix obligations" in planner_prompts[2]
    if form == "semantic-patch":
        # Same authenticated base binding on the replan turn.
        binding_line = re.search(r"- base_state_identity: (\S+)", planner_prompts[1]).group(0)
        assert binding_line in planner_prompts[2]
    assert WEAK_SUMMARY not in _published(runner)
    assert "Corrected revision." in _published(runner)
    assert runner.diagnostic_posts == []
    review_prompts = _agent_prompts(runner, "codex")
    assert len(review_prompts) == 2
    assert all(WEAK_SUMMARY not in prompt for prompt in review_prompts)
    assert "Inherited coverage delta: none" in review_prompts[1]


def _m976_full_plan_state():
    from coding_review_agent_loop.protocol import RISK_MATRIX_MAX_ROWS

    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["risk_test_matrix"] = _matrix(*[
        _matrix_row(f"row-{index:02d}", "one-shot") for index in range(RISK_MATRIX_MAX_ROWS)
    ])
    return json.dumps(payload) + PLAN_FOOTER


def _m976_patch(base_plan_text, *, summary, add_row=False):
    patch = json.loads(_semantic_patch(base_plan_text, None, summary=summary).split("\n<!--", 1)[0])
    if add_row:
        patch["operations"].append({
            "op": "matrix_add", "row": _matrix_row("row-new", "one-shot"),
            "final_position": 0, "rationale": "Cover the reviewer finding.",
        })
    return json.dumps(patch) + PLAN_FOOTER


def test_m976_top_level_row_bound_overflow_is_replanned_instead_of_fatal(tmp_path, monkeypatch):
    """#976: a top-level (non-child) revision that overflows the row bound replans."""
    def forbid_repair(*args, **kwargs):
        raise AssertionError("a deterministic assembly rejection must never reach the repair model")

    monkeypatch.setattr(orchestrator, "_run_structured_repair", forbid_repair)
    fresh = _m976_full_plan_state()
    overflow = _m976_patch(fresh, summary=WEAK_SUMMARY, add_row=True)
    consolidated = _m976_patch(fresh, summary="Consolidated revision.")
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh, overflow, consolidated],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Add coverage."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0

    planner_prompts = _agent_prompts(runner, "claude")
    assert len(planner_prompts) == 3
    assert "Trusted orchestration correction record" not in planner_prompts[1]
    assert "Trusted orchestration correction record" in planner_prompts[2]
    assert "24-row bound" in planner_prompts[2]
    assert "consolidate scenarios explicitly" in planner_prompts[2]
    # The overflowing candidate is never published or reviewed.
    assert WEAK_SUMMARY not in _published(runner)
    assert "Consolidated revision." in _published(runner)
    assert runner.diagnostic_posts == []
    review_prompts = _agent_prompts(runner, "codex")
    assert len(review_prompts) == 2
    assert all(WEAK_SUMMARY not in prompt for prompt in review_prompts)


def test_m976_top_level_row_bound_overflow_exhausts_as_deterministic_failure(tmp_path):
    fresh = _m976_full_plan_state()
    overflow = _m976_patch(fresh, summary=WEAK_SUMMARY, add_row=True)
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh] + [overflow] * 3,
        codex_outputs=[structured_plan_review(state="blocking", blocking_plan_issues=["Add coverage."])],
    )
    with pytest.raises(AgentInvocationError, match="fail deterministic plan assembly") as error:
        orchestrator.run_issue_loop(
            runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )
    assert error.value.failure_category == "deterministic"
    assert "consolidate scenarios explicitly" in str(error.value)
    assert len(_agent_prompts(runner, "claude")) == 4
    assert WEAK_SUMMARY not in _published(runner)


def _m979_patch_with_unknown_disposition(base_plan_text, *, summary):
    patch = json.loads(_m976_patch(base_plan_text, summary=summary).split("\n<!--", 1)[0])
    patch["prior_plan_item_dispositions"].append(
        {"item_id": "item-43", "disposition": "resolved", "note": "Same-round finding."}
    )
    return json.dumps(patch) + PLAN_FOOTER


def _m979_forbid_repair(*args, **kwargs):
    raise AssertionError("a semantic patch payload rejection must never reach the repair model")


def test_m979_patch_payload_rejection_is_replanned_without_repair(tmp_path, monkeypatch):
    """#979: repair pins the patch payload, so a payload rejection replans instead."""
    monkeypatch.setattr(orchestrator, "_run_structured_repair", _m979_forbid_repair)
    fresh = _m976_full_plan_state()
    rejected = _m979_patch_with_unknown_disposition(fresh, summary=WEAK_SUMMARY)
    corrected = _m976_patch(fresh, summary="Corrected revision.")
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh, rejected, corrected],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Add coverage."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0

    planner_prompts = _agent_prompts(runner, "claude")
    assert len(planner_prompts) == 3
    assert "Trusted orchestration correction record" not in planner_prompts[1]
    assert "Trusted orchestration correction record" in planner_prompts[2]
    assert "Unknown prior-item disposition ID(s) ['item-43']" in planner_prompts[2]
    # Same authenticated base binding on the replan turn.
    binding_line = re.search(r"- base_state_identity: (\S+)", planner_prompts[1]).group(0)
    assert binding_line in planner_prompts[2]
    assert WEAK_SUMMARY not in _published(runner)
    assert "Corrected revision." in _published(runner)
    assert runner.diagnostic_posts == []
    review_prompts = _agent_prompts(runner, "codex")
    assert len(review_prompts) == 2
    assert all(WEAK_SUMMARY not in prompt for prompt in review_prompts)


def test_m979_patch_payload_rejection_exhausts_as_deterministic_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "_run_structured_repair", _m979_forbid_repair)
    fresh = _m976_full_plan_state()
    rejected = _m979_patch_with_unknown_disposition(fresh, summary=WEAK_SUMMARY)
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh] + [rejected] * 3,
        codex_outputs=[structured_plan_review(state="blocking", blocking_plan_issues=["Add coverage."])],
    )
    with pytest.raises(
        AgentInvocationError, match="fail semantic patch payload validation"
    ) as error:
        orchestrator.run_issue_loop(
            runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )
    assert error.value.failure_category == "deterministic"
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None and exhaustion.candidate_kind == "plan_revision"
    assert "item-43" in exhaustion.diagnostic
    assert len(_agent_prompts(runner, "claude")) == orchestrator.MAX_INHERITED_MATRIX_REPLANS + 2
    assert WEAK_SUMMARY not in _published(runner)


def test_m979_patch_envelope_defect_still_routes_to_repair(tmp_path, monkeypatch):
    """#979: an envelope-only defect is exactly what patch repair may fix."""
    fresh = _m976_full_plan_state()
    corrected = _m976_patch(fresh, summary="Repaired revision.")
    missing_footer = corrected.split("\n<!--", 1)[0]
    repair_calls = []

    def envelope_repair(raw, *, validate, repair_kwargs, **kwargs):
        repair_calls.append((raw, repair_kwargs["expected_kind"]))
        return corrected, validate(corrected), []

    monkeypatch.setattr(orchestrator, "_run_structured_repair", envelope_repair)
    runner = _ChildPlanningRunner(
        claude_outputs=[fresh, missing_footer],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Add coverage."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    assert repair_calls == [(missing_footer, "plan_revision_patch")]
    assert len(_agent_prompts(runner, "claude")) == 2
    assert "Repaired revision." in _published(runner)


def test_m979_payload_rejection_classifier_matches_what_repair_may_change():
    """No rejection class both invokes patch repair and is guaranteed to fail preservation."""
    fresh = _m976_full_plan_state()
    good = _m976_patch(fresh, summary="Good.")
    unknown = _m979_patch_with_unknown_disposition(fresh, summary="Bad.")

    def validate(text):
        return orchestrator._validate_plan_revision_patch_response(
            text,
            unresolved_items=(
                orchestrator.UnresolvedReviewItem(
                    item_id="item-1", reviewer="OpenAI Codex", source_round=1,
                    text="Add coverage.", status="blocking", source_status="blocking",
                ),
            ),
        )

    def classify(text, normalized=None):
        with pytest.raises(AgentLoopError) as error:
            validate(text)
        return orchestrator._semantic_patch_payload_rejection(
            error.value, text=text, normalized=normalized, validate=validate
        )

    # Payload-level: the unknown disposition sits inside the pinned payload.
    candidate, diagnostic = classify(unknown)
    assert candidate == unknown and "item-43" in diagnostic
    # Envelope-only: a missing footer is exactly what repair may change.
    assert classify(good.split("\n<!--", 1)[0]) is None
    # An envelope defect hiding a payload defect is still unsatisfiable once
    # the envelope is normalized.
    prose_wrapped = "Here it is:\n" + unknown
    normalized = orchestrator.attempt_envelope_normalization(
        prose_wrapped, expected_kind="plan_revision_patch"
    )
    assert normalized is not None
    candidate, diagnostic = classify(prose_wrapped, normalized)
    assert candidate == normalized and "item-43" in diagnostic
    # The classifier agrees with the preservation check: any repair that
    # clears the payload rejection must change the payload, which is vetoed.
    from coding_review_agent_loop.repair_preservation import validate_repair_preservation

    stripped = _m976_patch(fresh, summary="Bad.")
    with pytest.raises(AgentLoopError, match="must be preserved exactly"):
        validate_repair_preservation(unknown, stripped)


def test_m931_replan_exhaustion_persists_one_record_and_resume_feeds_the_next_turn(
    tmp_path, monkeypatch
):
    _bind_child_planning(monkeypatch)
    weak = _child_plan_state(_weak_child_row(), summary=WEAK_SUMMARY)
    runner = _ChildPlanningRunner(claude_outputs=[weak] * 5)
    with pytest.raises(AgentInvocationError) as error:
        orchestrator.run_issue_loop(
            runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )
    assert error.value.failure_category == "deterministic"
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None and exhaustion.candidate_kind == "plan_state"
    assert "row-owned: forbidden_side_effects dropped" in exhaustion.diagnostic
    assert len(_agent_prompts(runner, "claude")) == orchestrator.MAX_INHERITED_MATRIX_REPLANS + 1 == 3
    assert _agent_prompts(runner, "codex") == []
    assert len(runner.diagnostic_posts) == 1
    assert WEAK_SUMMARY not in runner.diagnostic_posts[0]
    assert WEAK_SUMMARY not in _published(runner)

    # A new invocation recovers the record, feeds it to the first planner turn,
    # and proceeds from the last canonical state with a fresh replan budget.
    resumed = _ChildPlanningRunner(
        issue_comments=[{
            "author": {"login": "agent"}, "authorId": 7, "id": 901, "databaseId": 901,
            "createdAt": "2026-09-17T05:30:00Z", "body": runner.diagnostic_posts[0],
        }],
        claude_outputs=[weak, _child_plan_state(_child_row())],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda r, *, config, issue_number: IssueContext(
            number=56, repo="OWNER/REPO", title="Child", body="Child", url="child-url",
            comments=(IssueComment(
                author="agent", author_id=7, comment_id=901,
                created_at="2026-09-17T05:30:00Z", body=runner.diagnostic_posts[0],
            ),),
        ),
    )
    assert orchestrator.run_issue_loop(
        resumed, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    resumed_prompts = _agent_prompts(resumed, "claude")
    assert len(resumed_prompts) == 2
    assert "- Failed validation attempt: 1" in resumed_prompts[0]
    assert "row-owned: forbidden_side_effects dropped" in resumed_prompts[0]
    assert len(_agent_prompts(resumed, "codex")) == 1


def test_m931_over_cap_delta_set_rejects_the_candidate_into_the_replan_loop(
    tmp_path, monkeypatch
):
    _bind_child_planning(monkeypatch)
    monkeypatch.setattr(orchestrator, "INHERITED_COVERAGE_DELTA_MAX_BYTES", 64)
    extended = _child_row(event=_parent_row()["event"] + " " + "with detail " * 20)
    runner = _ChildPlanningRunner(
        claude_outputs=[_child_plan_state(extended, summary=WEAK_SUMMARY), _child_plan_state(_child_row())],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    planner_prompts = _agent_prompts(runner, "claude")
    assert len(planner_prompts) == 2
    assert "Make fewer or smaller departures from the inherited text" in planner_prompts[1]
    assert WEAK_SUMMARY not in _published(runner)


def test_m931_oversized_inherited_obligations_stop_before_any_planner_turn(tmp_path, monkeypatch):
    _bind_child_planning(monkeypatch)
    monkeypatch.setattr(orchestrator, "INHERITED_OBLIGATIONS_ENFORCEABLE_MAX_BYTES", 32)
    runner = _ChildPlanningRunner(claude_outputs=[_child_plan_state(_child_row())])
    with pytest.raises(orchestrator.PlanPrePanelSafetyError) as error:
        orchestrator.run_issue_loop(
            runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )
    text = str(error.value)
    assert "stage `api`" in text and "permitted 32 bytes" in text and "parent issue #55" in text
    assert re.search(r"measure \d+ bytes", text)
    assert _agent_prompts(runner, "claude") == [] and _agent_prompts(runner, "codex") == []


def test_m931_ordinary_plan_first_issue_is_unchanged(tmp_path):
    runner = _ChildPlanningRunner(
        claude_outputs=[_child_plan_state(_weak_child_row())],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        runner, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    assert len(_agent_prompts(runner, "claude")) == 1
    assert "Inherited" not in _agent_prompts(runner, "claude")[0]
    assert "Inherited" not in _agent_prompts(runner, "codex")[0]


def test_m931_resumed_review_round_recomputes_the_identical_delta_block(tmp_path, monkeypatch):
    _bind_child_planning(monkeypatch)
    refined = _child_row(
        applicability="required", proposed_test_location="tests/test_elsewhere.py"
    )

    def delta_block(prompt):
        return prompt.split("Inherited coverage delta", 1)[1].split("test reachability", 1)[0]

    first = _ChildPlanningRunner(
        claude_outputs=[_child_plan_state(refined)],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        first, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    resumed = _ChildPlanningRunner(
        issue_comments=[first.issue_comments[0]],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        resumed, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    assert _agent_prompts(resumed, "claude") == []
    fresh_block = delta_block(_agent_prompts(first, "codex")[0])
    assert "field `applicability` (applicability raised)" in fresh_block
    assert "field `proposed_test_location` (proposed test changed)" in fresh_block
    assert delta_block(_agent_prompts(resumed, "codex")[0]) == fresh_block


def test_m931_historical_weakened_plan_is_blocked_in_review_and_never_approved(
    tmp_path, monkeypatch
):
    # Publish the weakened plan without a binding, as a run before this check would.
    first = _ChildPlanningRunner(
        claude_outputs=[_child_plan_state(_weak_child_row())],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert orchestrator.run_issue_loop(
        first, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    _bind_child_planning(monkeypatch)
    resumed = _ChildPlanningRunner(
        issue_comments=[first.issue_comments[0]],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    # The weakened plan is never approved for implementation: the guard sends
    # it to an enforced revision turn (#936), which this script does not supply.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        orchestrator.run_issue_loop(
            resumed, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )
    review_prompt = _agent_prompts(resumed, "codex")[0]
    assert "Inherited coverage check FAILED" in review_prompt
    assert "row-owned: forbidden_side_effects dropped" in review_prompt
    assert "plan approved" not in _published(resumed)
    revision_prompt = _agent_prompts(resumed, "claude")[0]
    assert "row-owned: forbidden_side_effects dropped" in revision_prompt


def test_m931_parent_dispatch_and_direct_child_entry_bind_the_inherited_rows(
    tmp_path, monkeypatch
):
    captured = []
    monkeypatch.setattr(
        orchestrator,
        "_run_plan_first_loop",
        lambda runner, **kwargs: captured.append(kwargs["inherited_matrix_binding"]) or 0,
    )
    plan = fresh_staged_matrix_plan().replace(
        '"execution_disposition": "direct-implementation"',
        '"execution_disposition": "requires-child-planning"',
        1,
    )
    child, parent = fresh_child_contexts(
        plan,
        inherited_matrix_row_ids=("row-stage-one",),
        handoff_execution_disposition="requires-child-planning",
    )
    child = dataclasses.replace(child, comments=())
    monkeypatch.setattr(
        orchestrator,
        "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    # Direct child `--plan-first` entry.
    assert orchestrator.run_issue_loop(
        FakeRunner(), issue_number=56, config=make_config(tmp_path), plan_first=True
    ) == 0
    # Parent dispatch entry.
    fresh_child = orchestrator._resolve_fresh_child_provenance(
        issue_context=child, parent_issue_context=parent
    )
    assert fresh_child is not None and fresh_child.route.is_planning
    config = make_config(tmp_path)
    assert orchestrator._dispatch_decomposition_child(
        FakeRunner(), config=config, memory=None,
        usage_context=orchestrator._new_usage_context(config),
        parent_issue=55, approved_plan=fresh_child.approved_plan,
        plan_hash=fresh_child.plan_hash, plan_subject=fresh_child.plan_subject,
        recommendation=fresh_child.recommendation,
        approved_plan_context=fresh_child.parent_plan_context,
        created=fresh_child.created, phase_index=fresh_child.phase_index,
        route=fresh_child.route, child_issue_context=child, parent_issue_context=parent,
        coder_session_id=None, existing_handoff=fresh_child.handoff,
    ) == 0
    assert len(captured) == 2 and captured[0] == captured[1]
    binding = captured[0]
    assert (binding.parent_issue, binding.stage_id) == (55, "stage-one")
    assert [row.row_id for row in binding.inherited_rows()] == ["row-stage-one"]


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize("variant", ["refined", "weakened"])
def test_m931_open_child_pr_is_readmitted_or_rejected_identically_on_both_paths(
    tmp_path, monkeypatch, entry, variant
):
    from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section

    parent_plan = fresh_staged_matrix_plan()
    inherited_row = next(
        row for row in json.JSONDecoder().raw_decode(parent_plan)[0]["risk_test_matrix"]["rows"]
        if row["row_id"] == "row-stage-one"
    )
    child_row = {
        **inherited_row,
        "execution_owner": "one-shot",
        "label": "Reworded label",
        "applicability": "required",
        "forbidden_side_effects": ["No duplicate record.", *inherited_row["forbidden_side_effects"]],
        "event": inherited_row["event"] + " on the second attempt",
        "proposed_test_location": "tests/test_elsewhere.py",
    }
    if variant == "weakened":
        child_row["entry_path_or_mode"] = "pr mode"
    child_plan = "Approved child plan.\n\n" + render_risk_test_matrix_section(
        parse_risk_test_matrix(_matrix(child_row, _matrix_row("child-local", "one-shot")))
    )
    child, parent = fresh_child_contexts(
        parent_plan, summary_mode="decompose-only", handoff_mode="implement-by-phase",
        inherited_matrix_row_ids=("row-stage-one",), child_plan=child_plan,
    )
    parent = dataclasses.replace(parent, comments=parent.comments[:-1])
    monkeypatch.setattr(
        orchestrator, "get_issue_context",
        lambda runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    runner = FakeRunner(
        pr_payload={"number": 77, "body": "Fixes #56", "url": "https://github.com/OWNER/REPO/pull/77"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    def run():
        if entry == "issue":
            return orchestrator.run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return orchestrator.run_pr_loop(runner, pr_number=77, config=config)

    if variant == "weakened":
        with pytest.raises(AgentLoopError, match="row-stage-one: entry_path_or_mode replaced"):
            run()
        assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    else:
        assert run() == 0
        prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
        assert "Row row-stage-one" in prompt and "Row child-local" in prompt
    # Neither path replans the child or re-invokes the implementation coder.
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


# ---------------------------------------------------------------------------
# Re-planning an approved child plan that is already bound to a PR (#936)
# ---------------------------------------------------------------------------

from coding_review_agent_loop.decomposition import (  # noqa: E402
    CHILD_PLAN_REBIND_MARKER_RE,
    collect_child_plan_supersessions,
    format_child_plan_supersession_comment,
)
from coding_review_agent_loop.issue_pr_handoff import (  # noqa: E402
    AGENT_ISSUE_PR_HANDOFF_RE,
    find_latest_issue_pr_handoff,
)
from coding_review_agent_loop.round_state import _extract_round_metadata_records  # noqa: E402

PR_APPROVAL = "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"


def _m936_parent_plan():
    return fresh_staged_matrix_plan(first_disposition="requires-child-planning")


def _m936_inherited_row(**overrides):
    row = next(
        row for row in json.JSONDecoder().raw_decode(_m936_parent_plan())[0]["risk_test_matrix"]["rows"]
        if row["row_id"] == "row-stage-one"
    )
    return {**row, "execution_owner": "one-shot", **overrides}


def _m936_child_state(row, *, summary="Child plan."):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["summary"] = summary
    payload["risk_test_matrix"] = _matrix(row)
    return json.dumps(payload) + PLAN_FOOTER


def _m936_patch(base_plan_text, row, *, summary, base_round=1):
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(base_plan_text), round_number=base_round
    )
    operations = [{"op": "replace", "field": "summary", "value": summary}]
    if row is not None:
        operations.append({
            "op": "matrix_edit", "row_id": "row-stage-one", "row": row,
            "rationale": "Restore the inherited row.",
        })
    return json.dumps({
        "schema_version": 1, "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1, "state": "blocking", "summary": "Patch.",
        "prior_plan_item_dispositions": [],
        "base_round_number": base_round, "base_state_identity": base.state_identity,
        "operations": operations,
    }) + PLAN_FOOTER


_M936_STAGED = dict(
    reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
    primary_plan_reviewer="codex",
)


class _M936World:
    """A planning child whose approved plan is bound to open PR #77.

    Comments the run posts are appended to the child issue, so every rerun
    sees the durable state the previous one left behind.
    """

    def __init__(self, tmp_path, monkeypatch, *, weak=True, signed=True, staged=False):
        self.tmp_path = tmp_path
        row = _m936_inherited_row(entry_path_or_mode="pr mode") if weak else _m936_inherited_row()
        self.old_state = _m936_child_state(row)
        history = _ChildPlanningRunner(
            claude_outputs=[self.old_state],
            codex_outputs=[structured_plan_review(state="approved")],
            gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
        )
        # ``staged`` plans the history under primary-then-panel, so its panel
        # opening and scheduler contract are already durable.
        history_config = _plan_config(tmp_path, **(_M936_STAGED if staged else {}))
        assert orchestrator.run_issue_loop(
            history, issue_number=56, config=history_config, plan_first=True
        ) == 0
        plan_comments = [comment(str(item["body"])) for item in history.issue_comments]
        records = _extract_round_metadata_records(plan_comments, flow="plan")
        self.old_hash = approved_plan_hash(records[0].metadata.canonical_plan)
        child, self.parent = fresh_child_contexts(
            _m936_parent_plan(),
            handoff_execution_disposition="requires-child-planning",
            inherited_matrix_row_ids=("row-stage-one",),
        )
        self.child = child
        self.comments = [
            *plan_comments,
            comment(format_issue_pr_handoff_comment(
                issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
                pr_head_sha="abc123", flow="approved-plan-implementation", plan_hash=self.old_hash,
            )),
        ]
        if signed:
            self.comments.append(comment(self.signed_record()))
        self.runner = None
        monkeypatch.setattr(orchestrator, "get_issue_context", self._issue_context)

        def forbid_repair(*args, **kwargs):
            raise AssertionError("the repair model must never see an inherited-row rejection")

        monkeypatch.setattr(orchestrator, "_run_structured_repair", forbid_repair)

    def signed_record(self, **overrides):
        fields = dict(
            child_issue=56, parent_issue=55, stage_id="stage-one",
            superseded_plan_hash=self.old_hash, rationale="Contract tightened in #934.",
        )
        fields.update(overrides)
        return format_child_plan_supersession_comment(**fields)

    def _issue_context(self, _runner, *, config, issue_number):
        if issue_number != 56:
            return self.parent
        posted = [comment(str(item["body"])) for item in (self.runner.issue_comments if self.runner else [])]
        return dataclasses.replace(self.child, comments=(*self.comments, *posted))

    def settle(self):
        """Fold the finished run's comments into durable history."""
        if self.runner is not None:
            self.comments.extend(comment(str(item["body"])) for item in self.runner.issue_comments)
            self.runner = None

    def config(self, **kwargs):
        kwargs.setdefault("max_rounds", 4)
        kwargs.setdefault("plan_execution_mode", "auto")
        return make_config(
            self.tmp_path, execution_strategy_contract_required=True, **kwargs,
        )

    def run_issue(self, *, pr_state="OPEN", config=None, **outputs):
        self.settle()
        self.runner = _ChildPlanningRunner(
            pr_payload={
                "number": 77, "body": "Fixes #56", "state": pr_state,
                "url": "https://github.com/OWNER/REPO/pull/77",
            },
            **outputs,
        )
        return orchestrator.run_issue_loop(
            self.runner, issue_number=56, config=config or self.config(), plan_first=True
        )

    def run_pr(self, **outputs):
        self.settle()
        self.runner = _ChildPlanningRunner(
            pr_payload={
                "number": 77, "body": "Fixes #56", "url": "https://github.com/OWNER/REPO/pull/77",
            },
            **outputs,
        )
        return orchestrator.run_pr_loop(self.runner, pr_number=77, config=self.config())

    def good_patch(self, *, summary="Inherited rows restored."):
        return _m936_patch(self.old_state, _m936_inherited_row(), summary=summary)

    def agent_calls(self, agent):
        return _agent_prompts(self.runner, agent)

    def posted(self):
        return [str(item["body"]) for item in self.runner.issue_comments]

    def all_comments(self):
        return [*self.comments, *(comment(body) for body in self.posted())]


def test_m936_admissible_handoff_resumes_its_pr_unchanged(tmp_path, monkeypatch):
    world = _M936World(tmp_path, monkeypatch, weak=False, signed=False)
    assert world.run_issue(codex_outputs=[PR_APPROVAL]) == 0
    assert world.agent_calls("claude") == []
    assert len(world.agent_calls("codex")) == 1
    assert not any(
        AGENT_ISSUE_PR_HANDOFF_RE.search(body) or CHILD_PLAN_REBIND_MARKER_RE.search(body)
        for body in world.posted()
    )


def test_m936_inadmissible_handoff_without_record_gives_the_template_and_runs_nothing(
    tmp_path, monkeypatch
):
    world = _M936World(tmp_path, monkeypatch, signed=False)
    with pytest.raises(AgentLoopError) as excinfo:
        world.run_issue()
    message = str(excinfo.value)
    assert "row-stage-one: entry_path_or_mode replaced" in message
    assert '"kind": "child-plan-supersession"' in message
    assert f'"superseded_plan_hash": "{world.old_hash}"' in message
    assert '"child_issue": 56' in message and '"stage_id": "stage-one"' in message
    assert "agent-loop issue 56 --plan-first --plan-execution-mode auto" in message
    assert world.agent_calls("claude") == [] and world.agent_calls("codex") == []
    assert world.posted() == [] and world.runner.comments == []
    # A record for another plan hash, or an unsigned one, authorizes nothing.
    world.comments.append(comment(world.signed_record(superseded_plan_hash="0" * 16)))
    world.comments.append(comment(world.signed_record().replace("\n-- Human Reviewer", "")))
    with pytest.raises(AgentLoopError, match="child-plan-supersession"):
        world.run_issue()
    assert world.agent_calls("claude") == [] and world.posted() == []


def test_m936_signed_replan_rebinds_the_same_pr_and_reviews_under_the_new_plan(
    tmp_path, monkeypatch
):
    world = _M936World(tmp_path, monkeypatch)
    assert world.run_issue(
        claude_outputs=[world.good_patch()],
        codex_outputs=[structured_plan_review(state="approved"), PR_APPROVAL],
    ) == 0

    comments = world.all_comments()
    digest = collect_child_plan_supersessions(
        comments, child_issue=56, parent_issue=55, stage_id="stage-one"
    )[0].digest
    coder_rounds = [
        record.metadata for record in _extract_round_metadata_records(comments, flow="plan")
        if record.metadata.role == "coder"
    ]
    assert [item.plan_supersession_digest for item in coder_rounds] == [None, digest]
    assert coder_rounds[1].plan_supersession_superseded_hash == world.old_hash
    new_hash = approved_plan_hash(coder_rounds[1].canonical_plan)
    # Exactly one comment carries both records, for the same PR and closing IDs.
    rebinds = [body for body in world.posted() if CHILD_PLAN_REBIND_MARKER_RE.search(body)]
    assert len(rebinds) == 1 and AGENT_ISSUE_PR_HANDOFF_RE.search(rebinds[0])
    assert sum(1 for body in world.posted() if AGENT_ISSUE_PR_HANDOFF_RE.search(body)) == 1
    latest = find_latest_issue_pr_handoff(comments, issue_number=56, repo="OWNER/REPO")
    assert (latest.pr_number, latest.plan_hash) == (77, new_hash)
    assert latest.expected_closing_issue_ids == (56,)
    # No second PR, no implementation turn, no PR-side contract write.
    planner_prompts = world.agent_calls("claude")
    assert len(planner_prompts) == 1 and "row-stage-one: entry_path_or_mode replaced" in planner_prompts[0]
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in world.runner.commands)
    assert not any("AGENT_PR_EXPECTED_CLOSING_ISSUES" in body for body in world.runner.comments)
    # The revised plan got a plan review, then the PR was reviewed under it.
    reviewer_prompts = world.agent_calls("codex")
    assert len(reviewer_prompts) == 2
    assert "Inherited rows restored." in reviewer_prompts[1]


def test_m936_each_restart_point_continues_from_durable_state(tmp_path, monkeypatch):
    world = _M936World(tmp_path, monkeypatch)
    # Interrupted after the digest-bound revised round: the reviewer never ran.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        world.run_issue(claude_outputs=[world.good_patch()])
    assert len(world.agent_calls("claude")) == 1
    assert not any(CHILD_PLAN_REBIND_MARKER_RE.search(body) for body in world.posted())
    # Rerun resumes inside the bound lineage; the planner turn is not repeated.
    plan_only = world.config(plan_execution_mode="plan-only")
    assert world.run_issue(
        config=plan_only, codex_outputs=[structured_plan_review(state="approved")]
    ) == 0
    assert world.agent_calls("claude") == []
    # Approved but not yet rebound: the old binding is still the binding.
    assert not any(CHILD_PLAN_REBIND_MARKER_RE.search(body) for body in world.posted())
    world.settle()
    assert find_latest_issue_pr_handoff(
        world.comments, issue_number=56, repo="OWNER/REPO"
    ).plan_hash == world.old_hash
    # Rebind-only rerun: no planner, no plan reviewer, one rebind comment.
    assert world.run_issue(codex_outputs=[PR_APPROVAL]) == 0
    assert world.agent_calls("claude") == []
    assert sum(1 for body in world.posted() if CHILD_PLAN_REBIND_MARKER_RE.search(body)) == 1
    # After the rebind: plain PR resume with no second rebind write.
    assert world.run_issue(codex_outputs=[PR_APPROVAL]) == 0
    assert world.agent_calls("claude") == []
    assert not any(
        AGENT_ISSUE_PR_HANDOFF_RE.search(body) or CHILD_PLAN_REBIND_MARKER_RE.search(body)
        for body in world.posted()
    )


@pytest.mark.parametrize("pr_state", ["CLOSED", "MERGED"])
def test_m936_closed_or_merged_canonical_pr_fails_closed_before_any_agent(
    tmp_path, monkeypatch, pr_state
):
    world = _M936World(tmp_path, monkeypatch)
    with pytest.raises(AgentLoopError, match="not OPEN"):
        world.run_issue(pr_state=pr_state, claude_outputs=[world.good_patch()])
    assert world.agent_calls("claude") == [] and world.posted() == []


def test_m936_pr_closed_between_approval_and_rebind_posts_no_handoff(tmp_path, monkeypatch):
    world = _M936World(tmp_path, monkeypatch)
    assert world.run_issue(
        config=world.config(plan_execution_mode="plan-only"),
        claude_outputs=[world.good_patch()],
        codex_outputs=[structured_plan_review(state="approved")],
    ) == 0
    with pytest.raises(AgentLoopError, match="not OPEN"):
        world.run_issue(pr_state="CLOSED")
    assert world.posted() == [] and world.agent_calls("claude") == []


def test_m936_pr_mode_names_the_issue_route_and_proceeds_after_a_rebind(tmp_path, monkeypatch):
    world = _M936World(tmp_path, monkeypatch)
    with pytest.raises(AgentLoopError) as excinfo:
        world.run_pr(codex_outputs=[PR_APPROVAL])
    assert "row-stage-one: entry_path_or_mode replaced" in str(excinfo.value)
    assert "agent-loop issue 56 --plan-first" in str(excinfo.value)
    assert '"kind": "child-plan-supersession"' in str(excinfo.value)
    assert world.agent_calls("codex") == [] and world.agent_calls("claude") == []
    assert world.posted() == []
    # A PR approval recorded under the old plan, before the rebind.
    old_context = orchestrator.recover_approved_plan_context(
        world.comments, expected_hash=world.old_hash
    )
    stale_approval = _attach_round_metadata(PR_APPROVAL, PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=1, subject="abc123",
        state="approved", approved_plan_hash=world.old_hash,
        approved_plan_subject=old_context.plan_subject,
    ))
    assert world.run_issue(
        claude_outputs=[world.good_patch()],
        codex_outputs=[structured_plan_review(state="approved"), PR_APPROVAL],
    ) == 0
    # The same PR command now proceeds, and the pre-rebind approval is not carried.
    world.settle()
    world.runner = None
    runner_kwargs = dict(codex_outputs=[PR_APPROVAL])
    assert world.run_pr(**runner_kwargs) == 0
    assert len(world.agent_calls("codex")) == 1
    from coding_review_agent_loop.round_state import _latest_pr_approved_reviews_for_head

    new_context = orchestrator.recover_approved_plan_context(
        world.all_comments(),
        expected_hash=find_latest_issue_pr_handoff(
            world.all_comments(), issue_number=56, repo="OWNER/REPO"
        ).plan_hash,
    )
    carried = _latest_pr_approved_reviews_for_head(
        [comment(stale_approval)], head_sha="abc123", configured_reviewers=("codex",),
        approved_plan_context=new_context,
    )
    assert carried == {}
    # The same record would have carried under the plan it was recorded for.
    assert list(_latest_pr_approved_reviews_for_head(
        [comment(stale_approval)], head_sha="abc123", configured_reviewers=("codex",),
        approved_plan_context=old_context,
    )) == ["Codex"]


def _m936_unbound_round(world, plan_state, *, number, digest=None):
    """A genuine generation-1 plan round that did not come from the authorized re-plan."""
    base = _extract_round_metadata_records(world.comments, flow="plan")[0].metadata
    structured = validate_structured_plan_state(plan_state)
    plan = orchestrator.render_canonical_plan_state(structured, world.config())
    sidecar = orchestrator.make_assembled_plan_sidecar(
        structured, round_number=number, response_form="fresh-plan-state", rendered_plan=plan
    )
    identity = orchestrator.risk_test_matrix_identity(
        structured.risk_test_matrix, structured.risk_test_matrix_changes
    )
    return comment(_attach_round_metadata(plan, dataclasses.replace(
        base, round_number=number, subject=_plan_subject(plan), prior_plan_subject=base.subject,
        canonical_plan=plan, raw_structured_coder_response=plan_state,
        response_form="fresh-plan-state", aggregate_plan_identity=sidecar.aggregate_identity,
        assembled_plan_sidecar=sidecar.to_payload(),
        risk_test_matrix_payload=structured.risk_test_matrix.to_payload(),
        risk_test_matrix_identity=identity, risk_test_matrix_boundary_digest=identity,
        plan_supersession_digest=digest,
        plan_supersession_superseded_hash=world.old_hash if digest else None,
    )))


@pytest.mark.parametrize("variant", ["predates", "other-digest", "conflicting-records"])
def test_m936_only_the_authorized_lineage_is_resumed(tmp_path, monkeypatch, variant):
    world = _M936World(tmp_path, monkeypatch)
    later_state = _m936_child_state(_m936_inherited_row(), summary="A later admissible plan.")
    if variant == "predates":
        # An admissible later plan that carries no digest bypasses the authorization.
        record = _m936_unbound_round(world, later_state, number=2)
        world.comments.insert(len(world.comments) - 1, record)
        expected = "not part of the authorized re-plan"
    elif variant == "other-digest":
        record = _m936_unbound_round(world, later_state, number=2, digest="e" * 64)
        world.comments.append(record)
        expected = "not part of the authorized re-plan"
    else:
        world.comments.append(comment(world.signed_record(rationale="A second, different reason.")))
        expected = "two distinct signed child-plan supersession records"
    with pytest.raises(AgentLoopError, match=expected):
        world.run_issue(
            claude_outputs=[world.good_patch()],
            codex_outputs=[structured_plan_review(state="approved")],
        )
    assert world.agent_calls("claude") == [] and world.agent_calls("codex") == []
    assert world.posted() == []


def _m936_rebound_world(tmp_path, monkeypatch):
    world = _M936World(tmp_path, monkeypatch)
    assert world.run_issue(
        claude_outputs=[world.good_patch()],
        codex_outputs=[structured_plan_review(state="approved"), PR_APPROVAL],
    ) == 0
    world.settle()
    return world


def _m936_break_rebind(world, fault):
    index = next(
        position for position, item in enumerate(world.comments)
        if CHILD_PLAN_REBIND_MARKER_RE.search(item.body)
    )
    body = world.comments[index].body
    if fault == "missing-audit":
        body = CHILD_PLAN_REBIND_MARKER_RE.sub("", body)
    elif fault == "signed-record-deleted":
        world.comments = [
            item for item in world.comments if "child-plan-supersession" not in item.body
        ]
        return
    else:
        marker = CHILD_PLAN_REBIND_MARKER_RE.search(body)
        payload = json.loads(base64.urlsafe_b64decode(marker.group("payload")).decode("utf-8"))
        payload.update({
            "wrong-pr": {"pr_number": 78},
            "wrong-round": {"approved_round": payload["approved_round"] + 1},
            "wrong-digest": {"plan_supersession_digest": "d" * 64},
            "wrong-superseded": {"superseded_plan_hash": "0" * 16},
        }[fault])
        encoded = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).decode("ascii")
        body = body[:marker.start("payload")] + encoded + body[marker.end("payload"):]
    world.comments[index] = comment(body)


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize(
    "fault",
    ["missing-audit", "wrong-pr", "wrong-round", "wrong-digest", "wrong-superseded",
     "signed-record-deleted"],
)
def test_m936_unverifiable_rebind_fails_closed_on_both_entry_paths(
    tmp_path, monkeypatch, entry, fault
):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    _m936_break_rebind(world, fault)
    with pytest.raises(AgentLoopError, match="Human repair required"):
        if entry == "issue":
            world.run_issue(codex_outputs=[PR_APPROVAL])
        else:
            world.run_pr(codex_outputs=[PR_APPROVAL])
    assert world.agent_calls("codex") == [] and world.agent_calls("claude") == []
    assert world.posted() == [] and world.runner.comments == []


def _m936_snapshot(world, *, binding, approved_plan_context):
    world.settle()
    world.runner = _ChildPlanningRunner(pr_payload={
        "number": 77, "body": "Fixes #56", "url": "https://github.com/OWNER/REPO/pull/77",
    })
    child = world._issue_context(None, config=None, issue_number=56)
    return orchestrator._fresh_pr_qualification_snapshot(
        world.runner, config=world.config(), pr_number=77, issue_context=child,
        parent_issue_context=world.parent, approved_plan_context=approved_plan_context,
        allow_plan_handoff_change=True, planning_child_binding=binding,
    )


def _m936_binding(world):
    parent_plan = orchestrator.recover_approved_plan_context(
        world.parent.comments, expected_hash=approved_plan_hash(_m936_parent_plan())
    )
    return orchestrator._PlanningChildBinding(
        child_issue=56, parent_issue=55, stage_id="stage-one", parent_plan_context=parent_plan
    )


@pytest.mark.parametrize("fault", [None, "missing-audit", "wrong-digest", "non-child"])
def test_m936_mid_run_plan_adoption_is_verified_for_planning_children(
    tmp_path, monkeypatch, fault
):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    # The PR run started under the old plan; the rebind arrived while it was active.
    old_context = orchestrator.recover_approved_plan_context(
        world.comments, expected_hash=world.old_hash
    )
    new_hash = find_latest_issue_pr_handoff(
        world.comments, issue_number=56, repo="OWNER/REPO"
    ).plan_hash
    if fault in {"missing-audit", "wrong-digest"}:
        _m936_break_rebind(world, fault)
        with pytest.raises(AgentLoopError, match="Human repair required"):
            _m936_snapshot(world, binding=_m936_binding(world), approved_plan_context=old_context)
        return
    if fault == "non-child":
        # Without the binding the branch behaves exactly as before, audit record or not.
        _m936_break_rebind(world, "missing-audit")
        binding = None
    else:
        binding = _m936_binding(world)
    _context, _ids, adopted, _config = _m936_snapshot(
        world, binding=binding, approved_plan_context=old_context
    )
    # A changed plan identity is what triggers the existing fresh-sweep invalidation.
    assert adopted.plan_hash == new_hash != old_context.plan_hash


def test_m936_mid_run_adoption_rejects_an_inadmissible_replacement(tmp_path, monkeypatch):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    old_context = orchestrator.recover_approved_plan_context(
        world.comments, expected_hash=world.old_hash
    )
    monkeypatch.setattr(
        orchestrator, "_child_plan_admissibility_failure",
        lambda *args, **kwargs: "row-stage-one: entry_path_or_mode replaced",
    )
    with pytest.raises(AgentLoopError, match="replacement plan is itself inadmissible"):
        _m936_snapshot(world, binding=_m936_binding(world), approved_plan_context=old_context)


def _m936_append_closing_superset(world):
    """A closing-ID superset posted after the rebind, as the PR loop writes it."""
    latest = find_latest_issue_pr_handoff(world.comments, issue_number=56, repo="OWNER/REPO")
    world.comments.append(comment(format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url=latest.pr_url, pr_head_sha=latest.pr_head_sha,
        flow="approved-plan-implementation", plan_hash=latest.plan_hash,
        expected_closing_issue_ids=(56, 60), supersedes_hash=latest.contract_hash,
    )))


@pytest.mark.parametrize("entry", ["issue", "pr"])
@pytest.mark.parametrize("fault", ["missing-audit", "wrong-digest", "signed-record-deleted"])
def test_m936_closing_superset_after_a_rebind_never_hides_an_unverifiable_rebind(
    tmp_path, monkeypatch, entry, fault
):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    _m936_append_closing_superset(world)
    lineage = orchestrator.resolve_issue_pr_handoff_lineage(
        world.comments, issue_number=56, repo="OWNER/REPO"
    )
    assert lineage.closing_base is lineage.latest and lineage.replaced is not None
    _m936_break_rebind(world, fault)
    with pytest.raises(AgentLoopError, match="Human repair required"):
        if entry == "issue":
            world.run_issue(codex_outputs=[PR_APPROVAL])
        else:
            world.run_pr(codex_outputs=[PR_APPROVAL])
    assert world.agent_calls("codex") == [] and world.agent_calls("claude") == []
    assert world.posted() == [] and world.runner.comments == []


def test_m936_intact_rebind_still_verifies_after_a_closing_superset(tmp_path, monkeypatch):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    _m936_append_closing_superset(world)
    parent_plan = _m936_binding(world).parent_plan_context
    verified = orchestrator.verify_child_plan_rebind(
        world.comments, repo="OWNER/REPO", parent_plan_context=parent_plan,
        child_issue=56, parent_issue=55, stage_id="stage-one", pr_number=77,
    )
    assert verified is not None
    assert verified.plan_hash == find_latest_issue_pr_handoff(
        world.comments, issue_number=56, repo="OWNER/REPO"
    ).plan_hash


def test_m936_signed_record_cannot_launder_an_unverified_replacement(tmp_path, monkeypatch):
    """An unaudited replacement to an inadmissible plan, plus a signed record for it."""
    world = _M936World(tmp_path, monkeypatch, weak=False, signed=False)
    weak_state = _m936_child_state(
        _m936_inherited_row(entry_path_or_mode="pr mode"), summary="Swapped-in plan."
    )
    record = _m936_unbound_round(world, weak_state, number=2)
    weak_hash = approved_plan_hash(
        _extract_round_metadata_records([record], flow="plan")[0].metadata.canonical_plan
    )
    latest = find_latest_issue_pr_handoff(world.comments, issue_number=56, repo="OWNER/REPO")
    world.comments.extend([
        record,
        # Annotated equal-ID replacement handoff, but with no rebind audit record.
        comment(format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77, pr_url=latest.pr_url, pr_head_sha=latest.pr_head_sha,
            flow="approved-plan-implementation", plan_hash=weak_hash,
            supersedes_hash=latest.contract_hash,
        )),
        comment(world.signed_record(superseded_plan_hash=weak_hash)),
    ])
    with pytest.raises(AgentLoopError, match="Human repair required"):
        world.run_issue(
            claude_outputs=[world.good_patch()],
            codex_outputs=[structured_plan_review(state="approved")],
        )
    assert world.agent_calls("claude") == [] and world.agent_calls("codex") == []
    assert world.posted() == [] and world.runner.comments == []


def test_m936_verified_replacement_that_became_inadmissible_can_still_be_superseded(
    tmp_path, monkeypatch
):
    world = _m936_rebound_world(tmp_path, monkeypatch)
    rebound_hash = find_latest_issue_pr_handoff(
        world.comments, issue_number=56, repo="OWNER/REPO"
    ).plan_hash
    # The contract tightens again: the legitimately rebound plan is now inadmissible.
    real = orchestrator._child_plan_admissibility_failure

    def tightened(parent_plan_context, child_plan_context, *, stage_id):
        if child_plan_context.plan_hash == rebound_hash:
            return "row-stage-one: entry_path_or_mode replaced"
        return real(parent_plan_context, child_plan_context, stage_id=stage_id)

    monkeypatch.setattr(orchestrator, "_child_plan_admissibility_failure", tightened)
    # Provenance verifies, so the route is the signed-record template, not human repair.
    with pytest.raises(AgentLoopError) as excinfo:
        world.run_issue()
    assert "Human repair required" not in str(excinfo.value)
    assert f'"superseded_plan_hash": "{rebound_hash}"' in str(excinfo.value)
    assert world.agent_calls("claude") == [] and world.posted() == []
    with pytest.raises(AgentLoopError) as pr_excinfo:
        world.run_pr(codex_outputs=[PR_APPROVAL])
    assert f'"superseded_plan_hash": "{rebound_hash}"' in str(pr_excinfo.value)
    assert world.agent_calls("codex") == []


def _m936_staged_config(world, **kwargs):
    return world.config(max_rounds=8, **_M936_STAGED, **kwargs)


def _m936_truncate_after_latest_coder_round(world):
    """Durable state of a run that stopped right after the revised plan round."""
    world.settle()
    records = _extract_round_metadata_records(world.comments, flow="plan")
    last_coder = max(r.index for r in records if r.metadata.role == "coder")
    world.comments = world.comments[: last_coder + 1]


def test_m936_full_board_latch_survives_a_restart_after_the_digest_bound_round(
    tmp_path, monkeypatch
):
    world = _M936World(tmp_path, monkeypatch, staged=True)
    # Narrow classification would otherwise select only the primary on restart.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        world.run_issue(config=_m936_staged_config(world), claude_outputs=[world.good_patch()])
    _m936_truncate_after_latest_coder_round(world)
    assert orchestrator._resumed_inherited_replan_force_full(world.comments)
    assert world.run_issue(
        config=_m936_staged_config(world, plan_execution_mode="plan-only"),
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    ) == 0
    assert world.agent_calls("claude") == []
    audits = [body for body in world.posted() if "Plan review scheduling audit" in body]
    assert "Force-full: True (source: automatic)" in audits[0]
    assert "Selected reviewers: Codex, Gemini" in audits[0]


# --- #948: an oversized child re-plan posts as a bounded digest and resumes ---

import hashlib  # noqa: E402

import coding_review_agent_loop.round_transport as _m948_transport  # noqa: E402
from coding_review_agent_loop.comment_rendering import (  # noqa: E402
    COMPACT_PLAN_DIGEST_NOTICE,
)

M948_ROW_IDS = tuple(f"row-inherited-{index:02d}" for index in range(11))


def _m948_noise(seed, chars=256):
    text = ""
    while len(text) < chars:
        text += hashlib.sha256(f"{seed}-{len(text)}".encode()).hexdigest()
    return text[:chars]


def _m948_parent_rows():
    return [_parent_row(row_id=row_id, label=row_id) for row_id in M948_ROW_IDS]


def _m948_binding():
    return InheritedMatrixBinding(
        parent_issue=55, stage_id="api", parent_matrix=_matrix(*_m948_parent_rows())
    )


def _m948_child_payload(*, summary):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["summary"] = summary
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    payload["risk_test_matrix"] = _matrix(
        *[{**row, "execution_owner": "one-shot"} for row in _m948_parent_rows()]
    )
    return payload


def _m948_anchors(comments):
    return [
        str(item["body"]) for item in comments
        if not _m948_transport.is_round_transport_sidecar(str(item["body"]))
    ]


def _m948_coder_anchors(comments):
    return [body for body in _m948_anchors(comments) if COMPACT_PLAN_DIGEST_NOTICE in body]


@pytest.mark.parametrize("form", ["semantic-patch", "full-state"])
def test_m948_oversized_child_revision_posts_digest_and_resumes_losslessly(
    tmp_path, monkeypatch, form
):
    _bind_child_planning(monkeypatch, _m948_binding())
    fresh_payload = _m948_child_payload(summary="Child plan.")
    fresh = json.dumps(fresh_payload) + PLAN_FOOTER
    if form == "semantic-patch":
        base = AuthenticatedPlanState.from_plan(validate_structured_plan_state(fresh), round_number=1)
        revision = json.dumps({
            "schema_version": 1, "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1, "state": "blocking", "summary": "Patch.",
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
            ],
            "base_round_number": 1, "base_state_identity": base.state_identity,
            "operations": [{"op": "replace", "field": "summary", "value": "Revised child plan."}],
        }) + PLAN_FOOTER
    else:
        monkeypatch.setattr(orchestrator, "make_assembled_plan_sidecar", lambda *a, **k: None)
        revised = {**fresh_payload, "kind": "plan_revision", "summary": "Revised child plan."}
        revised["prior_plan_item_dispositions"] = [
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ]
        revision = json.dumps(revised) + PLAN_FOOTER

    # The second reviewer turn has no output, so the run stops with the
    # revision published and unreviewed; a new invocation must resume it.
    first = _ChildPlanningRunner(
        claude_outputs=[fresh, revision],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
        ],
    )
    with pytest.raises(Exception):
        orchestrator.run_issue_loop(
            first, issue_number=56, config=_plan_config(tmp_path), plan_first=True
        )

    bodies = [str(item["body"]) for item in first.issue_comments]
    assert all(len(body) <= _m948_transport.MAX_GITHUB_BODY_CHARS for body in bodies)
    coder_anchors = _m948_coder_anchors(first.issue_comments)
    assert len(coder_anchors) == 2
    revision_anchor = coder_anchors[-1]
    assert revision_anchor.startswith("## Revised plan")
    # Sidecars precede the anchor they belong to.
    revision_index = bodies.index(revision_anchor)
    assert _m948_transport.is_round_transport_sidecar(bodies[revision_index - 1])
    for record in (
        "AGENT_RISK_TEST_MATRIX:", "AGENT_EXECUTION_RECOMMENDATION:", "AGENT_LOOP_META:",
        "<!-- AGENT_PLAN_STATE: blocking -->", "-- Anthropic Claude",
    ):
        assert record in revision_anchor
    assert "Step 299:" not in revision_anchor
    assert "more of 300 omitted; complete list in the authenticated attachments" in revision_anchor

    # Metadata carries the full canonical plan, unchanged by the digest.
    match = list(_m948_transport.ROUND_RESUME_MARKER_RE.finditer(revision_anchor))[-1]
    hydrated, missing = _m948_transport.hydrate_mapping(
        _m948_transport.decode_mapping(match.group("payload")), bodies
    )
    assert missing == set()
    canonical_plan = hydrated["canonical_plan"]
    assert f"Step 299: {_m948_noise(299)}" in canonical_plan
    assert all(row_id in canonical_plan for row_id in M948_ROW_IDS)
    assert COMPACT_PLAN_DIGEST_NOTICE not in canonical_plan
    assert hydrated["subject"] == orchestrator._plan_subject(canonical_plan)

    # Resume: no planner turn, the reviewer sees the full plan, same subject.
    second = _ChildPlanningRunner(
        issue_comments=first.issue_comments,
        claude_outputs=[],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert orchestrator.run_issue_loop(
        second, issue_number=56, config=_plan_config(tmp_path), plan_first=True
    ) == 0
    assert _agent_prompts(second, "claude") == []
    assert len(_agent_prompts(second, "codex")) == 1
    # A prompt this large is delivered on stdin; the reviewer turn is the only
    # agent call of the resumed run.
    review_prompt = second.last_input_text
    assert f"Step 299: {_m948_noise(299)}" in review_prompt
    assert "Revised child plan." in review_prompt
    assert COMPACT_PLAN_DIGEST_NOTICE not in review_prompt
    # No forked plan round: nothing new was published by the coder.
    assert len(_m948_coder_anchors(second.issue_comments)) == 2
