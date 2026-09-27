"""Semantic evidence rejections never route to structured repair (#990).

Selecting an in-catalog broker handle without authoritative launch integrity
is an authority decision, not a formatting defect. Repair cannot satisfy it,
so it must be skipped (together with its model fallback chain), and the
terminal failure must name the validation rejection rather than a repair
timeout. Envelope-shaped defects on the same kind still route to repair.
"""

import json
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator_module
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.errors import AgentInvocationError
from coding_review_agent_loop.orchestrator import _run_validated_agent
from coding_review_agent_loop.protocol import validate_structured_issue_implementation

from agent_loop_helpers import FakeRunner, make_config, structured_issue_implementation

_SELECTOR = "turn:observation-3"
_CLAIM = {
    "row_id": "row-1",
    "execution_refs": [_SELECTOR],
    "test_identifiers": ["tests/test_example.py::test_behavior"],
    "test_locations": ["tests/test_example.py"],
    "workflow_path_claim": "issue mode / fresh implementation parse",
    "outcome_assertions": ["The response parses."],
    "forbidden_effect_assertions": ["No envelope rejection is raised."],
}


def _observation(**launch_state):
    return {
        "execution_ref": _SELECTOR,
        "outcome": "passed",
        "provenance": "parent-observed",
        "wrapper_bootstrap": "verified",
        "inner_exec": "started",
        "suite_start": "verified",
        **launch_state,
    }


def _implementation_with_claim(*, summary="Implemented the change.") -> str:
    text = structured_issue_implementation(summary=summary)
    body, footer = text.split("\n<!--", 1)
    payload = json.loads(body)
    payload["risk_test_matrix_claims"] = [dict(_CLAIM)]
    return json.dumps(payload) + "\n<!--" + footer


def _validator(catalog):
    def validate(text):
        return validate_structured_issue_implementation(
            text,
            delivered_risk_test_matrix_row_ids=["row-1"],
            execution_catalog=catalog,
        )

    return validate


def _run(tmp_path, text, validate):
    return _run_validated_agent(
        FakeRunner(),
        agent="claude",
        config=make_config(tmp_path, agent_max_retries=1),
        prompt="Implement the issue.",
        marker_description="structured issue_implementation result",
        validate=validate,
        role="coder",
        use_repair=True,
        repair_expected_kind="issue_implementation",
    )


def _forbid_repair(*args, **kwargs):
    pytest.fail("a semantic evidence rejection must not invoke structured repair")


@pytest.mark.parametrize(
    "response_text",
    [
        pytest.param(_implementation_with_claim(), id="raw-response"),
        # An envelope defect masks the authority defect until normalization.
        pytest.param(_implementation_with_claim() + "\ntrailing prose", id="envelope-masked"),
    ],
)
def test_non_authoritative_selector_skips_repair_and_reports_the_rejection(
    tmp_path, monkeypatch, response_text
):
    catalog = [_observation(suite_start="unknown")]
    monkeypatch.setattr(orchestrator_module, "_run_structured_repair", _forbid_repair)
    result = AgentResult(text=response_text, returncode=0)

    with patch.object(orchestrator_module, "run_agent_result", return_value=result) as invoke:
        with pytest.raises(AgentInvocationError) as excinfo:
            _run(tmp_path, response_text, _validator(catalog))

    # The rejection is final for this output: no coder retry either.
    assert invoke.call_count == 1
    error = excinfo.value
    assert error.failure_category == "deterministic"
    message = str(error)
    assert "launch-integrity" in message
    assert _SELECTOR in message
    assert "Failure category: semantic-evidence-rejection" in message
    assert "Failure category: timeout" not in message


def test_non_admissible_selector_also_skips_repair(tmp_path, monkeypatch):
    catalog = [_observation(outcome="failed")]
    monkeypatch.setattr(orchestrator_module, "_run_structured_repair", _forbid_repair)
    text = _implementation_with_claim()
    result = AgentResult(text=text, returncode=0)

    with patch.object(orchestrator_module, "run_agent_result", return_value=result):
        with pytest.raises(AgentInvocationError, match="not an admissible passing observation"):
            _run(tmp_path, text, _validator(catalog))


def test_envelope_defect_with_authoritative_selector_still_routes_to_repair(
    tmp_path, monkeypatch
):
    catalog = [_observation()]
    validate = _validator(catalog)
    # A blank summary is a schema defect normalization cannot fix.
    malformed = _implementation_with_claim(summary="")
    repaired = _implementation_with_claim(summary="Implemented the change.")
    with pytest.raises(Exception):
        validate(malformed)
    repair_calls = []

    def repair(raw, *, validate, **kwargs):
        repair_calls.append(raw)
        return repaired, validate(repaired), []

    monkeypatch.setattr(orchestrator_module, "_run_structured_repair", repair)
    result = AgentResult(text=malformed, returncode=0)

    with patch.object(orchestrator_module, "run_agent_result", return_value=result):
        response = _run(tmp_path, malformed, validate)

    assert len(repair_calls) == 1
    assert response.text == repaired
    assert response.marker_value.risk_test_matrix_claims.claims[0].execution_refs == (_SELECTOR,)
