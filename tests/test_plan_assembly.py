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
    make_assembled_plan_sidecar,
    structured_plan_revision_to_payload,
)
from coding_review_agent_loop.protocol import (
    parse_plan_revision_patch,
    validate_structured_plan_revision_patch,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
)
from coding_review_agent_loop.round_transport import decode_mapping


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


def test_same_base_replay_is_byte_identical() -> None:
    state = _state(_base([_row("row-a")]))
    patch = _patch(state, [{"op": "replace", "field": "summary", "value": "Replay me."}])
    first, first_sidecar = assemble_authenticated_plan_revision(state, patch)
    second, second_sidecar = assemble_authenticated_plan_revision(state, patch)
    assert structured_plan_revision_to_payload(first) == structured_plan_revision_to_payload(second)
    assert first_sidecar.to_payload() == second_sidecar.to_payload()
    assert first_sidecar.aggregate_identity == second_sidecar.aggregate_identity


def test_published_round_identity_survives_chained_restart_and_revision() -> None:
    state = _state(_base([_row("row-a")]), round_number=4)
    first, first_sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(
            state,
            [{"op": "replace", "field": "summary", "value": "First published result."}],
        ),
        result_round_number=5,
    )
    assert first.summary == "First published result."
    assert first_sidecar.round_number == 5

    restarted = hydrate_authenticated_plan_state(first_sidecar, round_number=5)
    assert restarted.round_number == 5
    assert restarted.state_identity == first_sidecar.aggregate_identity
    second, second_sidecar = assemble_authenticated_plan_revision(
        restarted,
        _patch(
            restarted,
            [{"op": "replace", "field": "summary", "value": "Second published result."}],
        ),
        result_round_number=6,
    )
    assert second.summary == "Second published result."
    assert second_sidecar.round_number == 6


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

    for result_round_number in (4, 3):
        with pytest.raises(AgentLoopError, match="greater than the authenticated base round"):
            assemble_authenticated_plan_revision(
                state,
                _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
                result_round_number=result_round_number,
            )

    _, sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
        result_round_number=5,
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
        result_round_number=5,
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


def test_semantic_round_metadata_rejects_partial_wrong_type_and_conflicting_authority() -> None:
    state = _state(_base([_row("row-a")]))
    _, sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
        result_round_number=5,
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
    payload = decode_mapping(_encode_round_metadata(metadata))

    malformed = [
        {key: value for key, value in payload.items() if key != "aggregate_plan_identity"},
        {**payload, "raw_patch_provenance": [payload["raw_patch_provenance"]]},
        {**payload, "assembled_plan_sidecar": "not-an-object"},
        {**payload, "round_number": 4},
    ]
    conflicting_sidecar = copy.deepcopy(sidecar.to_payload())
    conflicting_sidecar["raw_patch"] = {"conflicting": True}
    malformed.append({**payload, "assembled_plan_sidecar": conflicting_sidecar})
    for candidate in malformed:
        with pytest.raises(AgentLoopError):
            _decode_round_metadata_mapping(candidate)

    with pytest.raises(AgentLoopError, match="response_form"):
        _decode_round_metadata_mapping(
            {
                "flow": "plan",
                "role": "coder",
                "agent": "Codex",
                "round_number": 5,
                "subject": "subject",
                "base_round_number": None,
            }
        )


@pytest.mark.parametrize(
    ("response_form", "wrong_kind"),
    (
        ("semantic-patch-v1", "plan_state"),
        ("legacy-full-state", "plan_state"),
        ("fresh-plan-state", "plan_revision"),
    ),
)
def test_semantic_round_metadata_rejects_response_form_kind_mismatch(
    response_form: str, wrong_kind: str
) -> None:
    state = _state(_base([_row("row-a")]))
    _, revision_sidecar = assemble_authenticated_plan_revision(
        state,
        _patch(state, [{"op": "replace", "field": "summary", "value": "New."}]),
        result_round_number=5,
    )
    plan_state_payload = copy.deepcopy(revision_sidecar.canonical_json)
    plan_state_payload["kind"] = "plan_state"
    plan_state_payload.pop("prior_plan_item_dispositions", None)
    fresh_sidecar = make_assembled_plan_sidecar(
        plan_state_payload,
        round_number=5,
        response_form="fresh-plan-state",
    )
    valid_sidecar = {
        "semantic-patch-v1": revision_sidecar,
        "legacy-full-state": make_assembled_plan_sidecar(
            revision_sidecar.canonical_json,
            round_number=5,
            response_form="legacy-full-state",
        ),
        "fresh-plan-state": fresh_sidecar,
    }[response_form]
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Codex",
        round_number=5,
        subject="subject",
        response_form=response_form,
        base_round_number=state.round_number if response_form == "semantic-patch-v1" else None,
        base_state_identity=state.state_identity if response_form == "semantic-patch-v1" else None,
        aggregate_plan_identity=valid_sidecar.aggregate_identity,
        raw_patch_provenance=(
            valid_sidecar.raw_patch if response_form == "semantic-patch-v1" else None
        ),
        assembled_plan_sidecar=valid_sidecar.to_payload(),
    )
    payload = decode_mapping(_encode_round_metadata(metadata))
    wrong_payload = copy.deepcopy(payload)
    wrong_canonical = copy.deepcopy(valid_sidecar.canonical_json)
    wrong_canonical["kind"] = wrong_kind
    if wrong_kind == "plan_state":
        wrong_canonical.pop("prior_plan_item_dispositions", None)
    else:
        wrong_canonical["prior_plan_item_dispositions"] = []
    wrong_sidecar = make_assembled_plan_sidecar(
        wrong_canonical,
        round_number=5,
        response_form=response_form,
        raw_patch=valid_sidecar.raw_patch,
    )
    wrong_payload["assembled_plan_sidecar"] = wrong_sidecar.to_payload()
    wrong_payload["aggregate_plan_identity"] = wrong_sidecar.aggregate_identity

    with pytest.raises(AgentLoopError, match="does not match canonical plan kind"):
        _decode_round_metadata_mapping(wrong_payload)
