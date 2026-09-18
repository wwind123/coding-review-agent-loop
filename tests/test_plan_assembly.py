import copy
import json

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.plan_assembly import (
    AuthenticatedPlanState,
    aggregate_plan_identity,
    assemble_authenticated_plan_revision,
    decode_assembled_plan_sidecar,
    hydrate_authenticated_plan_state,
)
from coding_review_agent_loop.protocol import (
    parse_plan_revision_patch,
    validate_structured_plan_revision_patch,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
)


def _row(row_id: str, *, outcome: str | None = None) -> dict[str, object]:
    return {
        "row_id": row_id,
        "label": f"Scenario {row_id}",
        "entry_path_or_mode": "semantic draft revision",
        "initial_state": "An authenticated unapproved base exists.",
        "event": "A bounded semantic operation is submitted.",
        "expected_outcome": outcome or "The deterministic assembler emits one canonical result.",
        "forbidden_side_effects": ["Do not publish a partial candidate."],
        "proposed_test_level": "assembler unit",
        "proposed_test_location": "tests/test_plan_assembly.py",
        "applicability": "required",
        "related_scope_item_ids": ["deterministic-assembly"],
        "execution_owner": "assembler-foundation",
    }


def _base(rows: list[dict[str, object]], *, exclusions: list[str] | None = None) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "plan_revision",
        "state": "blocking",
        "summary": "Authenticated base summary.",
        "prior_plan_item_dispositions": [],
        "plan_steps": ["Keep this step byte-for-byte unchanged."],
        "external_dependencies": [{"title": "#827", "summary": "Publication remains external."}],
        "risk_test_matrix_contract_version": 1,
        "risk_test_matrix": {
            "applicability": "applicable",
            "rows": rows,
            "important_exclusions": exclusions or ["The base exclusion is preserved."],
        },
        "risk_test_matrix_changes": [],
    }


def _state(payload: dict[str, object], round_number: int = 4) -> AuthenticatedPlanState:
    return AuthenticatedPlanState.from_plan(payload, round_number=round_number)


def _patch(state: AuthenticatedPlanState, operations: list[dict[str, object]], **extra: object) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "The model explains the bounded semantic decision.",
        "prior_plan_item_dispositions": [],
        "base_round_number": state.round_number,
        "base_state_identity": state.state_identity,
        "operations": operations,
        **extra,
    }


def test_patch_parser_is_strict_and_distinguishes_legacy_plan_revision() -> None:
    state = _state(_base([_row("row-a")]))
    payload = _patch(state, [{"op": "replace", "field": "summary", "value": "Revised summary."}])
    parsed = parse_plan_revision_patch(payload)
    assert parsed.operations[0].field == "summary"
    text = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    assert validate_structured_plan_revision_patch(text) == parsed
    with pytest.raises(AgentLoopError, match="plan_revision_patch"):
        parse_plan_revision_patch({**payload, "kind": "plan_revision"})
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_plan_revision_patch({**payload, "unexpected": True})


def test_single_field_replacement_preserves_undeclared_authenticated_payload() -> None:
    base = _base([_row("row-a")])
    base["risk_test_matrix"] = {
        **base["risk_test_matrix"],
        "rows": [{**_row("row-a"), "label": "  Authenticated formatting  "}],
    }
    state = _state(base)
    assembled, sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New summary."}]),
    )
    assert assembled.summary == "New summary."
    assert sidecar.canonical_json["plan_steps"] == base["plan_steps"]
    assert sidecar.canonical_json["risk_test_matrix"]["rows"][0]["label"] == "  Authenticated formatting  "
    assert sidecar.canonical_json["external_dependencies"] == base["external_dependencies"]


def test_add_positions_split_and_noncontiguous_merge_are_patch_order_independent() -> None:
    base = _base([_row("row-a"), _row("row-b"), _row("row-c"), _row("row-d")])
    state = _state(base)
    add_a = {"op": "matrix_add", "row": _row("row-new-a"), "final_position": 1, "rationale": "Place A."}
    add_b = {"op": "matrix_add", "row": _row("row-new-b"), "final_position": 4, "rationale": "Place B."}
    first, first_sidecar = assemble_authenticated_plan_revision(state, _patch(state, [add_b, add_a]))
    second, second_sidecar = assemble_authenticated_plan_revision(state, _patch(state, [add_a, add_b]))
    assert [row.row_id for row in first.risk_test_matrix.rows] == [
        "row-a", "row-new-a", "row-b", "row-c", "row-new-b", "row-d"
    ]
    assert first_sidecar.canonical_json == second_sidecar.canonical_json
    assert first_sidecar.aggregate_identity == second_sidecar.aggregate_identity

    split_patch = _patch(
        state,
        [{
            "op": "matrix_split",
            "source_row_id": "row-b",
            "target_rows": [_row("row-b1"), _row("row-b2")],
            "rationale": "Split B in declared order.",
        }],
    )
    split, _ = assemble_authenticated_plan_revision(state, split_patch)
    assert [row.row_id for row in split.risk_test_matrix.rows] == [
        "row-a", "row-b1", "row-b2", "row-c", "row-d"
    ]

    merge_patch = _patch(
        state,
        [{
            "op": "matrix_merge",
            "source_row_ids": ["row-d", "row-b"],
            "target_row": _row("row-bd"),
            "rationale": "Merge the non-contiguous sources.",
        }],
    )
    merged, _ = assemble_authenticated_plan_revision(state, merge_patch)
    assert [row.row_id for row in merged.risk_test_matrix.rows] == ["row-a", "row-bd", "row-c"]


def test_audits_are_normalized_and_metadata_keeps_precise_structural_entries() -> None:
    base = _base([_row("row-a"), _row("row-b")])
    state = _state(base)
    operations = [
        {
            "op": "matrix_edit",
            "row_id": "row-b",
            "row": _row("row-b", outcome="B changed."),
            "rationale": "Change B.",
        },
        {
            "op": "matrix_metadata_replace",
            "value": {
                "applicability": "applicable",
                "important_exclusions": ["A revised exclusion."],
            },
            "audit_operation": "change",
            "rationale": "Change the complete matrix metadata.",
        },
    ]
    assembled, _ = assemble_authenticated_plan_revision(state, _patch(state, operations))
    assert [change.operation for change in assembled.risk_test_matrix_changes] == ["change", "change"]
    assert assembled.risk_test_matrix_changes[0].row_ids == ("row-a", "row-b")
    assert assembled.risk_test_matrix_changes[1].row_ids == ("row-b",)


def test_zero_row_not_applicable_metadata_uses_matrix_sentinel() -> None:
    base = {
        **_base([]),
        "risk_test_matrix": {
            "applicability": "not-applicable",
            "rows": [],
            "important_exclusions": [],
            "not_applicable_rationale": "There is no transition surface.",
        },
    }
    state = _state(base)
    patch = _patch(
        state,
        [{
            "op": "matrix_metadata_replace",
            "value": {
                "applicability": "not-applicable",
                "important_exclusions": [],
                "not_applicable_rationale": "The transition remains out of scope.",
            },
            "audit_operation": "change",
            "rationale": "Clarify the not-applicable boundary.",
        }],
    )
    assembled, _ = assemble_authenticated_plan_revision(state, patch)
    assert assembled.risk_test_matrix_changes[0].row_ids == ("matrix",)


@pytest.mark.parametrize(
    "operation, message",
    [
        ({"op": "matrix_edit", "row_id": "unknown", "row": _row("unknown"), "rationale": "No."}, "unknown source"),
        ({"op": "matrix_retire", "row_id": "row-a", "rationale": "One."}, "consumed by both"),
        ({"op": "replace", "field": "risk_test_matrix_changes", "value": []}, "derived"),
    ],
)
def test_conflicting_or_derived_operations_fail_before_mutation(operation: dict[str, object], message: str) -> None:
    state = _state(_base([_row("row-a")]))
    if operation["op"] == "matrix_retire":
        operation = [operation, {"op": "matrix_edit", "row_id": "row-a", "row": _row("row-a", outcome="Changed"), "rationale": "Two consumers."}]
    else:
        operation = [operation]
    with pytest.raises(AgentLoopError, match=message):
        assemble_authenticated_plan_revision(state, _patch(state, operation))


def test_no_ops_stale_bases_approved_bases_and_sidecar_hydration_fail_closed() -> None:
    state = _state(_base([_row("row-a")]))
    with pytest.raises(AgentLoopError, match="payload-identical"):
        assemble_authenticated_plan_revision(
            state,
            _patch(state, [{"op": "replace", "field": "summary", "value": state.plan.summary}]),
        )
    stale = _patch(state, [{"op": "replace", "field": "summary", "value": "New."}], base_round_number=3)
    with pytest.raises(AgentLoopError, match="base_round_number"):
        assemble_authenticated_plan_revision(state, stale)
    approved = AuthenticatedPlanState.from_plan(_base([_row("row-a")]), round_number=4, approved=True)
    with pytest.raises(AgentLoopError, match="unapproved"):
        assemble_authenticated_plan_revision(approved, _patch(approved, [{"op": "replace", "field": "summary", "value": "New."}]))

    _, sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
    )
    decoded = decode_assembled_plan_sidecar(sidecar.encode())
    hydrated = hydrate_authenticated_plan_state(decoded)
    assert hydrated.state_identity == decoded.aggregate_identity
    assert aggregate_plan_identity(decoded.canonical_json) == decoded.aggregate_identity
    corrupted = copy.deepcopy(decoded.to_payload())
    corrupted["canonical_json"]["summary"] = "tampered"
    with pytest.raises(AgentLoopError, match="identity mismatch"):
        decode_assembled_plan_sidecar(corrupted)


def test_semantic_round_metadata_round_trips_provenance_and_sidecar_without_legacy_fields() -> None:
    state = _state(_base([_row("row-a")]))
    _, sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
    )
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Codex",
        round_number=5,
        subject="subject",
        response_form="semantic-patch-v1",
        base_round_number=state.round_number,
        base_state_identity=state.state_identity,
        aggregate_plan_identity=sidecar.aggregate_identity,
        raw_patch_provenance=sidecar.raw_patch,
        assembled_plan_sidecar=sidecar.to_payload(),
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.response_form == "semantic-patch-v1"
    assert decoded.base_state_identity == state.state_identity
    assert decoded.aggregate_plan_identity == sidecar.aggregate_identity
    assert decoded.raw_patch_provenance == sidecar.raw_patch
    assert decoded.assembled_plan_sidecar == sidecar.to_payload()
