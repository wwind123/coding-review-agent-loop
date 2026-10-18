"""Non-collected issue-loop test helpers shared with other test files (#1541).

These helpers used to live in tests/test_orchestrator_issue.py, and other test
files imported them from there.  tests/test_orchestrator_issue.py imports them
back under the same names.
"""

import json
from dataclasses import replace
from types import SimpleNamespace

import coding_review_agent_loop.execution_policy as execution_policy
from coding_review_agent_loop.decomposition import CreatedPhaseIssue, PlanPhase
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState, assemble_plan_revision
from coding_review_agent_loop.protocol import validate_structured_plan_state

from agent_loop_helpers import structured_plan_review, structured_v1_plan_state


def _fresh_child_route_fixture(disposition, *, handoff=None):
    phase = PlanPhase(
        title="Child", scope="Implement child.", non_goals="No unrelated work.",
        dependency_notes="No dependencies.", rollout_risk="low",
        validation="Run focused tests.", parent_context="Approved slice.",
        automation="agent-pr", stage_id="stage-one", position=1,
        deliverables=("Child",), non_goals_items=("Unrelated work",),
        acceptance_criteria=("Tests pass",), compatibility_constraints=("Preserve API",),
        covered_scope_item_ids=("scope-1",), execution_disposition=disposition,
        disposition_rationale="Reviewed route.",
    )
    route = execution_policy.resolve_child_execution_route(
        phase, topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body="Child", url="child-url", comments=()
    )
    parent = replace(issue, number=55, title="Parent", url="parent-url")
    return SimpleNamespace(
        parent_issue=55, plan_hash="plan-hash", plan_subject="plan-subject",
        approved_plan="Approved parent plan", parent_plan_context=SimpleNamespace(
            matrix_available=False, risk_test_matrix_payload=None
        ), recommendation=SimpleNamespace(
            strategy="staged", identity=lambda: {"recommendation_sha256": "digest"},
            child_stages=(phase,),
        ), decomposition=SimpleNamespace(phases=(phase,)),
        created=CreatedPhaseIssue(phase=phase, issue_url=issue.url, issue_number=56),
        phase_index=1, stage_id="stage-one", handoff=handoff, overrides=(), route=route,
    ), issue, parent


def _m1103_blocking_chain(
    first_round, last_round, *, base, resolve_first=True, item_offset=0
):
    """Primary blocking reviews and chained planner patches for a round range.

    Round ``n``'s primary review blocks with a fresh finding ``item-k`` (and
    resolves ``item-(k-1)``), where ``k`` is ``n - item_offset``; the
    planner's revision after it is an authenticated patch bound to the
    round-``n`` state that resolves ``item-k``.  Returns the scripted outputs
    and the next base.
    """
    codex, claude = [], []
    for n in range(first_round, last_round + 1):
        k = n - item_offset
        codex.append(
            structured_plan_review(
                state="blocking",
                summary=f"Gap {n} remains.",
                blocking_plan_issues=[f"Close gap {n}."],
                prior_plan_item_dispositions=(
                    [{"item_id": f"item-{k - 1}", "disposition": "resolved"}]
                    if k > 1 and (resolve_first or n > first_round)
                    else None
                ),
            )
        )
        patch_payload = {
            "schema_version": 1,
            "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1,
            "state": "blocking",
            "summary": f"Close gap {n}.",
            "prior_plan_item_dispositions": [
                {"item_id": f"item-{k}", "disposition": "resolved"}
            ],
            "base_round_number": n,
            "base_state_identity": base.state_identity,
            "operations": [
                {
                    "op": "replace",
                    "field": "plan_steps",
                    "value": [f"Implement the reviewed scope, revision {n}."],
                }
            ],
        }
        base = AuthenticatedPlanState.from_plan(
            assemble_plan_revision(base, patch_payload), round_number=n + 1
        )
        claude.append(
            json.dumps(patch_payload)
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
        )
    return codex, claude, base


def _m1103_fresh_base():
    fresh = structured_v1_plan_state()
    return fresh, AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )


_STEP_BACK_MARKER = "STEP-BACK REVISION"


def _m1251_justify_patch(base, *, resolved_item, summary="Justify one-shot."):
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": [
            {"item_id": resolved_item, "disposition": "resolved"}
        ],
        "base_round_number": base.round_number,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "one_shot_growth_justification",
                "value": {
                    "crossed_signals": ["revision-count"],
                    "rationale": "The revisions share one seam and cannot ship separately.",
                },
            }
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=base.round_number + 1
    )
    text = (
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    return text, next_base
