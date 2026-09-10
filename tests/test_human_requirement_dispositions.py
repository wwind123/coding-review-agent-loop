import json

import pytest

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
            marker_validator=orchestrator._require_plan_state_or_clarification,
            human_requirements=(requirement,),
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the discussion.",
        )


def test_current_plan_gate_rejects_missing_coder_dispositions():
    assert not orchestrator._current_plan_has_complete_human_requirement_dispositions(
        _plan([]),
        surfaced_requirement_ids=("Requirement 1",),
    )
