import json

import pytest

from agent_loop_helpers import structured_coder_followup
import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.protocol import (
    validate_human_requirement_dispositions,
    validate_structured_plan_state,
)


def _plan(dispositions):
    payload = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": "blocking",
        "summary": "Plan the requested integration.",
        "plan_steps": ["Add the Grafana dashboard provisioning artifact."],
        "human_requirement_dispositions": dispositions,
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- coder"


def test_signed_requirement_requires_exact_structured_disposition():
    parsed = validate_structured_plan_state(
        _plan(
            [{
                "requirement_id": "Requirement 1",
                "disposition": "addressed",
                "evidence": "The plan names Grafana dashboard provisioning.",
            }]
        )
    )
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=("Requirement 1",),
    )


@pytest.mark.parametrize(
    "dispositions, message",
    [
        ([], "missing"),
        ([{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": "x"},
          {"requirement_id": "Requirement 1", "disposition": "blocked", "evidence": "y"}], "duplicate"),
        ([{"requirement_id": "Requirement 2", "disposition": "addressed", "evidence": "x"}], "unknown"),
    ],
)
def test_invalid_requirement_coverage_is_rejected(dispositions, message):
    parsed = validate_structured_plan_state(_plan(dispositions))
    with pytest.raises(AgentLoopError, match=message):
        validate_human_requirement_dispositions(
            parsed.human_requirement_dispositions,
            surfaced_requirement_ids=("Requirement 1",),
        )


def test_invalid_status_and_empty_evidence_are_rejected_at_schema_boundary():
    with pytest.raises(AgentLoopError, match="disposition"):
        validate_structured_plan_state(
            _plan([{"requirement_id": "Requirement 1", "disposition": "maybe", "evidence": "x"}])
        )
    with pytest.raises(AgentLoopError, match="evidence"):
        validate_structured_plan_state(
            _plan([{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": ""}])
        )


def test_no_signed_requirements_requires_empty_collection():
    parsed = validate_structured_plan_state(_plan([]))
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=(),
    )


def test_initial_plan_validator_checks_coder_dispositions():
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at=None,
        url=None,
        body="Provide Grafana.",
    )
    with pytest.raises(AgentLoopError, match="missing requirement ID"):
        orchestrator._validate_response_with_human_requirements(
            _plan([]).replace(
                "\n<!-- AGENT_PLAN_STATE: blocking -->",
                "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
                f"- Requirement {requirement.requirement_id}: Grafana is planned.\n"
                "<!-- AGENT_PLAN_STATE: blocking -->",
            ),
            marker_validator=lambda text: orchestrator._require_plan_state_or_clarification(
                text, architecture_status_mode="legacy"
            ),
            human_requirements=(requirement,),
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the discussion.",
        )


def test_current_plan_gate_rejects_missing_coder_dispositions():
    assert not orchestrator._current_plan_has_complete_human_requirement_dispositions(
        _plan([]),
        surfaced_requirement_ids=("Requirement 1",),
    )


def test_ack_gate_requires_dedicated_coder_disposition_evidence():
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="reviewer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Please use the absolute URL.",
    )
    output = structured_coder_followup(
        addressed_items=[],
        human_requirement_ids=[],
        human_requirement_dispositions=[],
    )

    with pytest.raises(AgentLoopError, match="missing"):
        validate_human_requirement_dispositions(
            [],
            surfaced_requirement_ids=(requirement.requirement_id,),
            context="coder_followup.human_requirement_dispositions",
        )
    assert '"human_requirement_dispositions": []' in output


# ---------------------------------------------------------------------------
# #905 (from #841): a carried plan approval cannot satisfy the gate vacuously


def _carried_plan_record(*, agent, state, key, requirement_ids, index):
    from coding_review_agent_loop.round_state import PostedRoundMetadata, PostedRoundRecord

    return PostedRoundRecord(
        index=index,
        body=f"{agent} plan review.",
        metadata=PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent=agent,
            round_number=1,
            subject=key.subject,
            state=state,
            plan_candidate_key=key.as_dict(),
            surfaced_reviewer_requirement_ids=tuple(requirement_ids),
        ),
    )


def _plan_key(requirement_ids):
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanCandidateKey,
        surfaced_requirement_id_digest,
    )

    return PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity="c" * 32,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest=surfaced_requirement_id_digest(requirement_ids),
    )


def test_carried_plan_approval_requires_the_current_requirement_acknowledgement():
    """`carried-approval-requires-current-ack`."""
    evidence = orchestrator.PlanPanelEvidence(opening_index=0, opening_source="operator")
    key = _plan_key(("req-1",))

    acknowledged = orchestrator._carried_plan_approvals(
        (
            _carried_plan_record(
                agent="Gemini", state="approved", key=key,
                requirement_ids=("req-1",), index=1,
            ),
        ),
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=("req-1",),
        panel_evidence=evidence,
        primary_reviewer="Codex",
    )
    assert acknowledged == ("Gemini",)

    # The stored approval carried no acknowledgement: it cannot be repaired in
    # place, so the reviewer must be re-invoked.
    unacknowledged = orchestrator._carried_plan_approvals(
        (
            _carried_plan_record(
                agent="Gemini", state="approved", key=key,
                requirement_ids=(), index=1,
            ),
        ),
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=("req-1",),
        panel_evidence=evidence,
        primary_reviewer="Codex",
    )
    assert unacknowledged == ()


def test_a_changed_surfaced_requirement_set_invalidates_every_carried_plan_approval():
    """`plan-identity-change-invalidates-approvals` and
    `human-requirements-and-decomposition`: an added, edited, replaced, or
    withdrawn requirement changes the key's requirement digest."""
    evidence = orchestrator.PlanPanelEvidence(opening_index=0, opening_source="operator")
    stored_key = _plan_key(("req-1",))
    records = (
        _carried_plan_record(
            agent="Gemini", state="approved", key=stored_key,
            requirement_ids=("req-1",), index=1,
        ),
    )

    for surfaced in (("req-1", "req-2"), ("req-2",), ()):
        current_key = _plan_key(surfaced)
        assert current_key.components != stored_key.components
        assert (
            orchestrator._carried_plan_approvals(
                records,
                current_key=current_key,
                required_reviewers=("Codex", "Gemini"),
                surfaced_requirement_ids=surfaced,
                panel_evidence=evidence,
                primary_reviewer="Codex",
            )
            == ()
        )


def test_a_premature_secondary_plan_approval_is_never_carried():
    """A secondary approval before any qualified opening is unqualified."""
    key = _plan_key(())
    records = (
        _carried_plan_record(
            agent="Gemini", state="approved", key=key, requirement_ids=(), index=1
        ),
        _carried_plan_record(
            agent="Codex", state="approved", key=key, requirement_ids=(), index=2
        ),
    )

    carried = orchestrator._carried_plan_approvals(
        records,
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=(),
        panel_evidence=orchestrator.PlanPanelEvidence(),
        primary_reviewer="Codex",
    )

    # Only the primary can hold a pre-opening approval.
    assert carried == ("Codex",)
