from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.comment_rendering import (
    decode_risk_test_matrix_marker,
    render_canonical_plan_state,
    render_public_agent_comment,
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import (
    RiskTestMatrixChange,
    parse_risk_test_matrix,
    parse_risk_test_matrix_evidence,
    risk_test_matrix_identity,
    validate_structured_plan_state,
    validate_risk_test_matrix_revision,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
    make_approved_plan_context,
)
from coding_review_agent_loop.round_transport import (
    prepare_round_comment,
    risk_test_matrix_section_boundary,
)


def _row(row_id: str = "row-ordinary") -> dict[str, object]:
    return {
        "row_id": row_id,
        "label": "Ordinary recovery reaches completion",
        "entry_path_or_mode": "ordinary / review-only",
        "initial_state": "review complete, CI obligation pending",
        "event": "repaired head passes",
        "expected_outcome": "Complete without requiring a watcher",
        "forbidden_side_effects": ["Do not merge a stale head"],
        "proposed_test_level": "orchestrator",
        "proposed_test_location": "tests/test_orchestrator_pr.py::test_review_only_recovery",
        "applicability": "applicable",
        "related_scope_item_ids": ["scope-matrix-evidence-review"],
        "execution_owner": "one-shot",
    }


def _matrix() -> dict[str, object]:
    return {
        "applicability": "applicable",
        "rows": [_row()],
        "important_exclusions": ["No Cartesian product of unrelated feature flags."],
    }


def _not_applicable() -> dict[str, object]:
    return {
        "applicability": "not-applicable",
        "rows": [],
        "important_exclusions": [],
        "not_applicable_rationale": "This is a local formatting-only change with no stateful entry path.",
    }


def test_m780_01_matrix_is_bounded_and_rendered_from_structured_payload() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    changes = (RiskTestMatrixChange("change", ("row-ordinary",), "Clarified the recovery outcome."),)
    section = render_risk_test_matrix_section(matrix, changes)
    assert "### Risk-based mode and transition test matrix" in section
    assert section.index("| ID |") < section.index("row-ordinary")
    assert "Clarified the recovery outcome." in section
    marker = section.rsplit("<!-- AGENT_RISK_TEST_MATRIX: ", 1)[1].split(" -->", 1)[0]
    decoded = decode_risk_test_matrix_marker(marker)
    assert decoded["identity"] == risk_test_matrix_identity(matrix, changes)
    assert risk_test_matrix_section_boundary(decoded["identity"]) in section


def test_m780_02_trivial_work_accepts_only_a_non_empty_not_applicable_rationale() -> None:
    matrix = parse_risk_test_matrix(_not_applicable())
    assert not matrix.is_applicable
    with pytest.raises(AgentLoopError, match="not_applicable_rationale|not-applicable matrices require"):
        parse_risk_test_matrix({**_not_applicable(), "not_applicable_rationale": "   "})


def test_m780_03_draft_changes_are_explicit_and_approved_rows_are_immutable() -> None:
    changed = {**_matrix(), "rows": [{**_row(), "expected_outcome": "A different outcome"}]}
    changes = [{"operation": "change", "row_ids": ["row-ordinary"], "rationale": "The state transition was clarified."}]
    validate_risk_test_matrix_revision(_matrix(), changed, changes)
    with pytest.raises(AgentLoopError, match="Approved risk matrix"):
        validate_risk_test_matrix_revision(_matrix(), changed, changes, approved=True)
    with pytest.raises(AgentLoopError, match="omit audit operations"):
        validate_risk_test_matrix_revision(_matrix(), changed, [])


def test_m780_04_not_applicable_and_matrix_less_plan_payloads_remain_compatible() -> None:
    base = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": "blocking",
        "summary": "A narrow plan.",
        "plan_steps": ["Make the local change."],
    }
    text = json.dumps(base) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    assert validate_structured_plan_state(text).risk_test_matrix is None
    matrix_payload = {
        **base,
        "risk_test_matrix_contract_version": 1,
        "risk_test_matrix": _not_applicable(),
        "risk_test_matrix_changes": [],
    }
    parsed = validate_structured_plan_state(
        json.dumps(matrix_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        require_risk_test_matrix_contract=1,
    )
    assert parsed is not None and parsed.risk_test_matrix is not None


def test_m780_05_evidence_requires_exact_row_identity_and_preserves_incomplete_status() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = parse_risk_test_matrix_evidence(
        {
            "matrix_identity": identity,
            "rows": [{
                "row_id": "row-ordinary",
                "status": "timed-out",
                "test_identifiers": ["test_review_only_recovery"],
                "test_locations": ["tests/test_orchestrator_pr.py:100"],
                "workflow_path_claim": "Reached the post-review recovery branch.",
                "outcome_assertions": ["Completion was not observed before timeout."],
                "forbidden_effect_assertions": ["No merge call occurred."],
                "evidence_citations": [{"command": "python -m pytest tests/test_orchestrator_pr.py -q", "receipt_id": "receipt-1", "claim": "current-result"}],
                "caveats": ["The managed test gate timed out."],
            }],
        },
        matrix=matrix,
    )
    assert evidence.rows[0].status == "timed-out"
    with pytest.raises(AgentLoopError):
        parse_risk_test_matrix_evidence(
            {"matrix_identity": identity, "rows": [{"row_id": "row-ordinary"}]},
            matrix=matrix,
        )


def test_m780_06_metadata_round_trip_authenticates_payload_not_rendered_words() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Codex",
        round_number=1,
        subject="subject",
        canonical_plan="A canonical plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.risk_test_matrix_payload == matrix.to_payload()
    assert decoded.risk_test_matrix_identity == identity
    corrupted = _encode_round_metadata(
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Codex", round_number=1, subject="subject",
            canonical_plan="A canonical plan", risk_test_matrix_contract_version=1,
            risk_test_matrix_payload={**matrix.to_payload(), "rows": []},
            risk_test_matrix_identity=identity,
        )
    )
    closed = _decode_round_metadata(corrupted)
    assert closed.risk_test_matrix_payload is None
    assert closed.risk_test_matrix_diagnostic


def test_m780_11_payload_bound_boundary_mismatch_closes_only_matrix_channel() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "Canonical plan prose",
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest="0" * 64,
    )
    assert context.availability == "available"
    assert not context.matrix_available
    assert context.risk_test_matrix_diagnostic


def test_m780_07_approved_context_has_matrix_channel_even_if_renderer_wording_changes() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "Summary\n\n### Risk-based mode and transition test matrix\nA renderer changed this table.",
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    assert context.availability == "available"
    assert context.matrix_available
    assert context.risk_test_matrix_payload == matrix.to_payload()


def test_m780_08a_matrix_semantics_remain_available_without_canonical_prose() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    assert context.is_available
    assert context.canonical_text is None
    assert context.matrix_available


def test_m780_09_public_evidence_renders_outstanding_rows_first() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = parse_risk_test_matrix_evidence(
        {
            "matrix_identity": identity,
            "rows": [{
                "row_id": "row-ordinary", "status": "missing", "test_identifiers": [],
                "test_locations": [], "workflow_path_claim": "Not exercised.",
                "outcome_assertions": [], "forbidden_effect_assertions": [], "evidence_citations": [],
            }],
        },
        matrix=matrix,
    )
    from coding_review_agent_loop.protocol import StructuredIssueImplementation, StructuredHumanRequirementsPayload

    text = render_public_agent_comment(
        kind="issue_implementation",
        parsed=StructuredIssueImplementation(
            schema_version=1, kind="issue_implementation", state="blocking",
            summary="Implementation is complete but the workflow row is still open.", pr_number=1,
            human_requirements=StructuredHumanRequirementsPayload((), False),
            human_requirement_dispositions=(), risk_test_matrix_evidence=evidence,
        ),
        agent="Codex",
    )
    assert text.index("row-ordinary") < text.index("AGENT_STATE")
