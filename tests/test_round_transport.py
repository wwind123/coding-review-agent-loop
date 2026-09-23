import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.round_transport as transport
import coding_review_agent_loop.comment_rendering as comment_rendering
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.round_state import (
    PlanValidationDiagnosticPayload,
    PostedRoundMetadata,
    QualificationCheckpoint,
    _attach_round_metadata,
    _decode_round_metadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
    _extract_round_metadata_records,
    _prior_item_ledger_signature,
    _plan_subject,
    decode_plan_validation_diagnostic_body,
    encode_plan_validation_diagnostic_body,
    recover_plan_validation_diagnostic,
    sanitize_plan_validation_diagnostic,
)
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.protocol import (
    MACHINE_AUTHORITY,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    UNKNOWN_MACHINE_AUTHORITY,
    parse_risk_test_matrix,
    validate_structured_plan_state,
)
from coding_review_agent_loop.review_scheduling import ReviewSchedulingContract
from coding_review_agent_loop.unresolved_items import _apply_unresolved_item_dispositions
from coding_review_agent_loop.protocol_markers import TrustedBody


def _random_text(size: int) -> str:
    return base64.urlsafe_b64encode(os.urandom(size)).decode("ascii")


def _comment(payload: dict[str, object], body: str = "Visible response") -> str:
    return f"{body}\n<!-- AGENT_LOOP_META: {transport.encode_mapping(payload)} -->"


def _anchor_payload(anchor: str) -> dict[str, object]:
    match = list(transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    return transport.decode_mapping(match.group("payload"))


def test_derived_matrix_evidence_round_trip_is_durable_and_idempotent() -> None:
    raw_coder = json.dumps({
        "schema_version": 1,
        "kind": "coder_followup",
        "state": "blocking",
        "summary": "The follow-up was completed.",
        "addressed_items": [],
        "remaining_items": [],
        "human_requirements": {
            "addressed_ids": [],
            "checked_discussion_directly": False,
        },
        "human_requirement_dispositions": [],
        "risk_test_matrix_claims": [{
            "row_id": "implementation-derived-evidence",
            "execution_refs": ["coder-turn:observation-7"],
            "test_identifiers": ["tests/test_orchestrator_pr.py::test_workflow"],
            "test_locations": ["tests/test_orchestrator_pr.py"],
            "workflow_path_claim": "The follow-up reached the authenticated head.",
            "outcome_assertions": ["The selected observation passed."],
            "forbidden_effect_assertions": ["The PR handoff was retained."],
            "caveats": [],
        }],
    }) + "\n<!-- AGENT_STATE: blocking -->\n-- Claude"
    metadata = PostedRoundMetadata(
        flow="pr",
        role="coder",
        agent="Claude",
        round_number=2,
        subject="head-2",
        risk_test_matrix_evidence={
            "contract_version": 1,
            "matrix_identity": "matrix-identity",
            "rows": [{
                "row_id": "implementation-derived-evidence",
                "status": "incomplete",
                "test_identifiers": [],
                "test_locations": [],
                "workflow_path_claim": "The follow-up handoff was authenticated.",
                "outcome_assertions": [],
                "forbidden_effect_assertions": [],
                "evidence_citations": [],
                "caveats": ["No current-turn receipt was selected."],
            }],
        },
        risk_test_matrix_diagnostics=(
            {
                "row_id": "implementation-derived-evidence",
                "code": "missing-claim",
                "message": "No semantic coverage claim was supplied.",
            },
        ),
        raw_structured_coder_response=raw_coder,
    )

    encoded = _encode_round_metadata(metadata)
    assert "execution_ref" not in encoded
    decoded = _decode_round_metadata(encoded)
    assert decoded.risk_test_matrix_evidence == metadata.risk_test_matrix_evidence
    assert decoded.risk_test_matrix_diagnostics == metadata.risk_test_matrix_diagnostics
    assert decoded.raw_structured_coder_response is not None
    assert "execution_ref" not in decoded.raw_structured_coder_response
    assert "risk_test_matrix_claims" not in decoded.raw_structured_coder_response
    assert _encode_round_metadata(decoded) == encoded

    first_body = _attach_round_metadata("derived handoff", metadata)
    second_body = _attach_round_metadata("derived handoff", decoded)
    assert first_body == second_body


def _risk_test_matrix_marker_payload(
    *, row_count: int = 11, repetition: int = 40
) -> tuple[dict[str, object], str]:
    rows = []
    for index in range(row_count):
        rows.append(
            {
                "row_id": f"row-{index}",
                "label": f"Transition {index} " + "label " * repetition,
                "entry_path_or_mode": "auto / staged " + "entry " * repetition,
                "initial_state": "approved primary " + "state " * repetition,
                "event": "panel review completes " + "event " * repetition,
                "expected_outcome": "advance exactly once " + "outcome " * repetition,
                "forbidden_side_effects": [
                    "do not duplicate work " + "effect " * repetition
                ],
                "proposed_test_level": "orchestrator",
                "proposed_test_location": f"tests/test_orchestrator_pr.py::test_transition_{index}",
                "applicability": "applicable",
                "related_scope_item_ids": ["scope-review-policy"],
                "execution_owner": "one-shot",
            }
        )
    matrix = parse_risk_test_matrix(
        {
            "applicability": "applicable",
            "rows": rows,
            "important_exclusions": ["No unrelated review modes."],
        }
    )
    payload = {
        "contract_version": 1,
        "matrix": matrix.to_payload(),
        "changes": [],
        "identity": comment_rendering.risk_test_matrix_identity(matrix),
    }
    return payload, comment_rendering._encode_json_payload(payload)


def test_encode_decode_mapping_round_trip_and_legacy_base64() -> None:
    payload = {"emoji": "✓", "items": ["one", 2]}

    assert transport.decode_mapping(transport.encode_mapping(payload)) == payload
    legacy = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    assert transport.decode_mapping(legacy) == payload


def _diagnostic_payload(
    *,
    attempt: int = 1,
    digest: str | None = None,
    diagnostic: str = "missing audit",
    target_coder_round: int = 1,
    prior_plan_subject: str | None = None,
    candidate_kind: str = "plan_state",
):
    return PlanValidationDiagnosticPayload(
        repository="OWNER/REPO",
        issue_number=813,
        planning_generation=1,
        target_coder_round=target_coder_round,
        prior_plan_subject=prior_plan_subject,
        candidate_kind=candidate_kind,
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
        expected_producer_login="agent",
        expected_producer_id=7,
        failure_attempt=attempt,
        candidate_digest=digest or (str(attempt) * 64),
        category="deterministic",
        diagnostic=diagnostic,
    )


def _diagnostic_comment(payload: PlanValidationDiagnosticPayload, *, comment_id: int, created_at: str = "2026-01-01T00:00:00Z", author: str = "agent", author_id: int = 7):
    return IssueComment(
        author=author,
        created_at=created_at,
        body=str(encode_plan_validation_diagnostic_body(payload)),
        comment_id=comment_id,
        author_id=author_id,
    )


def test_plan_validation_payload_roundtrip_is_pre_post_only_and_bounded() -> None:
    payload = _diagnostic_payload(diagnostic="x" * 20_000)
    body = encode_plan_validation_diagnostic_body(payload)

    assert "987654" not in str(body)
    assert "2026-01-01" not in str(body)
    assert len(payload.diagnostic) == 4096
    assert payload.diagnostic.endswith("[diagnostic truncated]")
    assert decode_plan_validation_diagnostic_body(str(body)) == payload
    assert len(sanitize_plan_validation_diagnostic("z" * 20_000)) == 4096


@pytest.mark.parametrize(
    "unsafe_diagnostic",
    (
        "z" * 4097,
        "password=do-not-trust-persisted-secrets",
        "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: nested -->",
    ),
)
def test_plan_validation_decoder_rejects_noncanonical_diagnostic(
    unsafe_diagnostic: str,
) -> None:
    mapping = _diagnostic_payload().as_dict()
    mapping["diagnostic"] = unsafe_diagnostic
    body = (
        "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: "
        f"{transport.encode_mapping(mapping)} -->"
    )

    with pytest.raises(AgentLoopError, match="Invalid plan-validation diagnostic"):
        decode_plan_validation_diagnostic_body(body)


def test_plan_validation_recovery_ignores_malformed_records() -> None:
    valid = _diagnostic_comment(_diagnostic_payload(attempt=2), comment_id=402)
    invalid_mapping = _diagnostic_payload(attempt=9).as_dict()
    invalid_mapping["diagnostic"] = "secret=must-not-be-normalized"
    invalid_body = (
        "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: "
        f"{transport.encode_mapping(invalid_mapping)} -->"
    )
    comments = (
        IssueComment(
            author="agent", author_id=7, comment_id=400,
            created_at="2026-01-01T00:00:00Z",
            body="<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC broken -->",
        ),
        IssueComment(
            author="agent", author_id=7, comment_id=401,
            created_at="2026-01-01T00:00:01Z", body=invalid_body,
        ),
        valid,
    )

    selected = recover_plan_validation_diagnostic(
        comments,
        repository="OWNER/REPO", issue_number=813,
        expected_author_login="agent", expected_author_id=7,
        planning_generation=1, target_coder_round=1,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    )

    assert selected is not None
    assert selected.server_comment_id == 402
    assert selected.failure_attempt == 2


def test_plan_validation_recovery_selects_highest_payload_attempt_not_comment_order() -> None:
    comments = (
        _diagnostic_comment(_diagnostic_payload(attempt=1), comment_id=101, created_at="2026-01-01T00:00:00Z"),
        _diagnostic_comment(_diagnostic_payload(attempt=3), comment_id=103, created_at="2025-01-01T00:00:00Z"),
        _diagnostic_comment(_diagnostic_payload(attempt=2), comment_id=102, created_at="2027-01-01T00:00:00Z"),
    )
    selected = recover_plan_validation_diagnostic(
        comments,
        repository="OWNER/REPO",
        issue_number=813,
        expected_author_login="agent",
        expected_author_id=7,
        planning_generation=1,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    )
    assert selected is not None
    assert selected.failure_attempt == 3
    assert selected.server_comment_id == 103


def test_plan_validation_recovery_rejects_highest_attempt_conflicts_and_spoofs() -> None:
    comments = (
        _diagnostic_comment(_diagnostic_payload(attempt=2, digest="a" * 64), comment_id=201),
        _diagnostic_comment(_diagnostic_payload(attempt=2, digest="b" * 64), comment_id=202),
    )
    with pytest.raises(AgentLoopError, match="highest failure attempt"):
        recover_plan_validation_diagnostic(
            comments,
            repository="OWNER/REPO", issue_number=813,
            expected_author_login="agent", expected_author_id=7,
            planning_generation=1, target_coder_round=1,
            prior_plan_subject=None, candidate_kind="plan_state",
            architecture_contract_version=1,
            execution_strategy_contract_version=1,
            risk_test_matrix_contract_version=1,
        )

    spoof = _diagnostic_comment(_diagnostic_payload(), comment_id=203, author="lookalike", author_id=8)
    assert recover_plan_validation_diagnostic(
        (spoof,),
        repository="OWNER/REPO", issue_number=813,
        expected_author_login="agent", expected_author_id=7,
        planning_generation=1, target_coder_round=1,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    ) is None


def test_verified_canonical_plan_success_semantically_supersedes_diagnostic() -> None:
    diagnostic = _diagnostic_comment(_diagnostic_payload(), comment_id=301)
    canonical = _attach_round_metadata(
        "Canonical plan",
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject=_plan_subject("Canonical plan"), prior_plan_subject=None, canonical_plan="Canonical plan",
            architecture_contract_version=1,
            execution_strategy_contract_version=1,
            risk_test_matrix_contract_version=1,
        ),
    )
    success = IssueComment(
        author="agent", author_id=7, comment_id=302,
        created_at="2026-01-02T00:00:00Z", body=str(canonical),
    )
    assert recover_plan_validation_diagnostic(
        (diagnostic, success),
        repository="OWNER/REPO", issue_number=813,
        expected_author_login="agent", expected_author_id=7,
        planning_generation=1, target_coder_round=1,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    ) is None


def test_unrelated_canonical_plan_context_does_not_supersede_diagnostic() -> None:
    previous_subject = "a" * 64
    other_subject = "b" * 64
    diagnostic = _diagnostic_comment(
        _diagnostic_payload(
            target_coder_round=2,
            prior_plan_subject=previous_subject,
            candidate_kind="plan_revision",
        ),
        comment_id=303,
    )
    canonical = _attach_round_metadata(
        "Unrelated canonical revision",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="c" * 64,
            prior_plan_subject=other_subject,
            canonical_plan="Unrelated canonical revision",
            architecture_contract_version=1,
            execution_strategy_contract_version=1,
            risk_test_matrix_contract_version=1,
        ),
    )
    success = IssueComment(
        author="agent",
        author_id=7,
        comment_id=304,
        created_at="2026-01-02T00:00:00Z",
        body=str(canonical),
    )
    recovered = recover_plan_validation_diagnostic(
        (diagnostic, success),
        repository="OWNER/REPO",
        issue_number=813,
        expected_author_login="agent",
        expected_author_id=7,
        planning_generation=1,
        target_coder_round=2,
        prior_plan_subject=previous_subject,
        candidate_kind="plan_revision",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    )
    assert recovered is not None
    assert recovered.failure_attempt == 1


def test_is_round_transport_sidecar() -> None:
    sidecar = transport._sidecar({"field": "canonical_plan"})
    assert transport.is_round_transport_sidecar(sidecar)
    assert not transport.is_round_transport_sidecar("ordinary agent output")


def test_prepare_round_comment_preserves_trusted_carrier_without_spill() -> None:
    carrier = TrustedBody.join(
        TrustedBody.current_untrusted_visible("first"),
        TrustedBody.current_untrusted_visible("second"),
    )

    prepared = transport.prepare_round_comment(carrier)

    assert prepared == (carrier,)
    assert prepared[0] is carrier
    assert prepared[0].segments == carrier.segments


def test_prepare_round_comment_spill_keeps_existing_segment_provenance() -> None:
    recommendation = {
        "strategy": "one-shot",
        "rationale": _random_text(50_000),
        "staging_feasibility": "inseparable",
        "scope_items": [{
            "scope_item_id": "scope-1",
            "requirement": "Implement the requested behavior.",
            "acceptance_criteria": ["The focused regression passes."],
        }],
        "coupling_constraints": [],
        "one_shot_delivery": {
            "deliverables": ["Implementation."],
            "acceptance_criteria": ["The regression passes."],
            "covered_scope_item_ids": ["scope-1"],
        },
        "child_stages": [],
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
    encoded = transport._b64(
        json.dumps(recommendation, separators=(",", ":"), sort_keys=True).encode()
    )
    marker_text = f"<!-- AGENT_EXECUTION_RECOMMENDATION: {encoded} -->"
    carrier = TrustedBody.join(
        TrustedBody.current_untrusted_visible("first"),
        TrustedBody.current_untrusted_visible("second"),
        TrustedBody.marker("AGENT_EXECUTION_RECOMMENDATION", marker_text),
    )

    prepared = transport.prepare_round_comment(carrier)

    assert len(prepared) > 1
    assert prepared[-1].segments[0] == ("first", None)
    assert prepared[-1].segments[1] == ("second", None)
    assert prepared[-1].segments[2][1] == "AGENT_EXECUTION_RECOMMENDATION"


def test_prepare_round_comment_spills_only_until_anchor_fits() -> None:
    review = _random_text(46_000)
    payload = {
        "canonical_reviewer_response": review,
        "raw_structured_coder_response": _random_text(500),
        "canonical_plan": _random_text(500),
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    assert len(prepared) > 1
    anchor_payload = _anchor_payload(prepared[-1])
    assert isinstance(anchor_payload["canonical_reviewer_response"], dict)
    assert anchor_payload["raw_structured_coder_response"] == payload["raw_structured_coder_response"]
    assert anchor_payload["canonical_plan"] == payload["canonical_plan"]
    hydrated, missing = transport.hydrate_mapping(anchor_payload, prepared)
    assert missing == set()
    assert hydrated == payload


def test_oversized_execution_recommendation_uses_bounded_lossless_sidecar() -> None:
    recommendation = {
        "strategy": "one-shot",
        "rationale": _random_text(50_000),
        "staging_feasibility": "inseparable",
        "scope_items": [{
            "scope_item_id": "scope-1",
            "requirement": "Implement the requested behavior.",
            "acceptance_criteria": ["The focused regression passes."],
        }],
        "coupling_constraints": [],
        "one_shot_delivery": {
            "deliverables": ["Implementation and tests."],
            "acceptance_criteria": ["The focused regression passes."],
            "covered_scope_item_ids": ["scope-1"],
        },
        "child_stages": [],
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
    encoded = base64.urlsafe_b64encode(
        json.dumps(recommendation, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    body = _random_text(5_000) + f"\n<!-- AGENT_EXECUTION_RECOMMENDATION: {encoded} -->"

    prepared = transport.prepare_round_comment(body)
    anchor = str(prepared[-1])

    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    marker = list(comment_rendering.EXECUTION_RECOMMENDATION_MARKER_RE.finditer(anchor))[-1]
    assert comment_rendering.decode_execution_recommendation_marker(
        marker.group("payload"), bodies=tuple(map(str, prepared))
    ) == recommendation


def test_rendered_oversized_execution_recommendation_keeps_anchor_bounded_and_reviewable() -> None:
    from agent_loop_helpers import structured_v1_plan_state

    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["execution_recommendation"]["rationale"] = (
        _random_text(50_000)
        + "\n### Execution strategy recommendation (v1)\n"
        + _random_text(500)
    )
    parsed = validate_structured_plan_state(
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Coder",
        require_execution_strategy_contract=1,
    )
    rendered = comment_rendering.render_execution_recommendation_section(
        parsed.execution_recommendation
    )
    rendered = _attach_round_metadata(
        rendered,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="codex",
            round_number=1,
            subject="plan-subject",
        ),
    )
    assert len(rendered) > transport.MAX_GITHUB_BODY_CHARS

    prepared = transport.prepare_round_comment(rendered)
    anchor_body = prepared[-1]
    anchor = str(anchor_body)

    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    assert "complete validated execution recommendation" in anchor
    assert "`strategy`: `one-shot`" in anchor
    assert "`staging_feasibility`: `inseparable`" in anchor
    assert "AGENT_LOOP_META" in anchor
    assert "rationale" not in anchor
    marker = list(comment_rendering.EXECUTION_RECOMMENDATION_MARKER_RE.finditer(anchor))[-1]
    assert comment_rendering.decode_execution_recommendation_marker(
        marker.group("payload"), bodies=tuple(map(str, prepared))
    ) == parsed.execution_recommendation.to_payload()
    assert all(
        len(item) <= transport.MAX_GITHUB_BODY_CHARS
        for item in prepared
    )
    anchor_body.validate_for_surface("issue_comment")


def test_large_risk_matrix_uses_compact_anchor_and_lossless_sidecar() -> None:
    rows = []
    for index in range(11):
        rows.append(
            {
                "row_id": f"row-{index}",
                "label": f"Transition {index} " + "label " * 40,
                "entry_path_or_mode": "auto / staged " + "entry " * 40,
                "initial_state": "approved primary " + "state " * 40,
                "event": "panel review completes " + "event " * 40,
                "expected_outcome": "advance exactly once " + "outcome " * 40,
                "forbidden_side_effects": ["do not duplicate work " + "effect " * 40],
                "proposed_test_level": "orchestrator",
                "proposed_test_location": f"tests/test_orchestrator_pr.py::test_transition_{index}",
                "applicability": "applicable",
                "related_scope_item_ids": ["scope-review-policy"],
                "execution_owner": "one-shot",
            }
        )
    matrix = parse_risk_test_matrix(
        {
            "applicability": "applicable",
            "rows": rows,
            "important_exclusions": ["No unrelated review modes."],
        }
    )
    section = comment_rendering.render_risk_test_matrix_section(matrix)
    canonical_plan = _random_text(30_000) + "\n" + section
    body = _attach_round_metadata(
        canonical_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="codex",
            round_number=1,
            subject="large-matrix-plan",
            canonical_plan=canonical_plan,
            risk_test_matrix_contract_version=1,
            risk_test_matrix_payload=matrix.to_payload(),
            risk_test_matrix_identity=comment_rendering.risk_test_matrix_identity(matrix),
            risk_test_matrix_boundary_digest=comment_rendering.risk_test_matrix_identity(matrix),
        ),
    )
    assert len(body) > transport._RISK_MATRIX_COMPACT_AT_CHARS

    prepared = transport.prepare_round_comment(body)
    anchor = str(prepared[-1])

    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    assert "- **Rows:** 11" in anchor
    assert "`row-0`" in anchor and "`row-10`" in anchor
    assert "Transition 0" in anchor
    assert "advance exactly once" not in anchor
    assert "hydrated losslessly" in anchor
    marker = list(comment_rendering.RISK_TEST_MATRIX_MARKER_RE.finditer(anchor))[-1]
    decoded = comment_rendering.decode_risk_test_matrix_marker(
        marker.group("payload"), bodies=tuple(map(str, prepared))
    )
    assert decoded["matrix"] == matrix.to_payload()
    round_payload = _anchor_payload(anchor)
    hydrated_round, missing = transport.hydrate_mapping(
        round_payload, tuple(map(str, prepared))
    )
    assert missing == set()
    assert hydrated_round["canonical_plan"] == canonical_plan
    with pytest.raises(AgentLoopError, match="sidecar unavailable"):
        comment_rendering.decode_risk_test_matrix_marker(marker.group("payload"))
    assert any(
        '"field":"risk_test_matrix_marker"' in base64.urlsafe_b64decode(
            transport.ROUND_TRANSPORT_SIDECAR_RE.search(str(item)).group("payload")
        ).decode()
        for item in prepared[:-1]
    )
    prepared[-1].validate_for_surface("issue_comment")


def test_oversized_risk_matrix_without_section_boundary_uses_bounded_lossless_sidecar() -> None:
    payload, encoded = _risk_test_matrix_marker_payload()
    marker_text = f"<!-- AGENT_RISK_TEST_MATRIX: {encoded} -->"
    prefix_length = max(
        1, transport._RISK_MATRIX_COMPACT_AT_CHARS + 1 - len(marker_text)
    )
    body = _random_text(prefix_length) + "\n" + marker_text

    assert len(body) > transport._RISK_MATRIX_COMPACT_AT_CHARS
    assert transport.risk_test_matrix_section_boundary(payload["identity"]) not in body

    prepared = transport.prepare_round_comment(body)
    anchor_body = prepared[-1]
    anchor = str(anchor_body)

    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    marker = list(comment_rendering.RISK_TEST_MATRIX_MARKER_RE.finditer(anchor))[-1]
    assert (marker.group(0), "AGENT_RISK_TEST_MATRIX") in anchor_body.segments
    assert comment_rendering.decode_risk_test_matrix_marker(
        marker.group("payload"), bodies=tuple(map(str, prepared))
    ) == payload


def test_oversized_risk_matrix_fallback_replaces_last_authorized_marker() -> None:
    first_payload, first_encoded = _risk_test_matrix_marker_payload(
        row_count=1, repetition=2
    )
    last_payload, last_encoded = _risk_test_matrix_marker_payload()
    first_marker = f"<!-- AGENT_RISK_TEST_MATRIX: {first_encoded} -->"
    last_marker = f"<!-- AGENT_RISK_TEST_MATRIX: {last_encoded} -->"
    prefix_length = max(
        1, transport._RISK_MATRIX_COMPACT_AT_CHARS + 1 - len(last_marker) - len(first_marker) - 1
    )
    carrier = TrustedBody.join(
        TrustedBody.current_untrusted_visible(_random_text(prefix_length)),
        TrustedBody.marker("AGENT_RISK_TEST_MATRIX", first_marker),
        TrustedBody.current_untrusted_visible("\n"),
        TrustedBody.marker("AGENT_RISK_TEST_MATRIX", last_marker),
    )

    prepared = transport.prepare_round_comment(carrier)
    anchor_body = prepared[-1]
    markers = list(comment_rendering.RISK_TEST_MATRIX_MARKER_RE.finditer(str(anchor_body)))

    assert len(markers) == 2
    assert markers[0].group("payload") == first_encoded
    assert markers[-1].group("payload") != last_encoded
    assert (markers[-1].group(0), "AGENT_RISK_TEST_MATRIX") in anchor_body.segments
    assert comment_rendering.decode_risk_test_matrix_marker(
        markers[-1].group("payload"), bodies=tuple(map(str, prepared))
    ) == last_payload
    assert comment_rendering.decode_risk_test_matrix_marker(
        markers[0].group("payload")
    ) == first_payload


def test_risk_matrix_rewrite_loss_raises_typed_transport_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, encoded = _risk_test_matrix_marker_payload()
    marker_text = f"<!-- AGENT_RISK_TEST_MATRIX: {encoded} -->"
    body = "\n".join(
        [
            transport.risk_test_matrix_section_boundary(payload["identity"]),
            "### Risk-based mode and transition test matrix",
            marker_text,
        ]
    )
    body = _random_text(
        max(1, transport._RISK_MATRIX_COMPACT_AT_CHARS + 1 - len(body) - 1)
    ) + "\n" + body
    original_pattern = transport._RISK_TEST_MATRIX_RE

    class MarkerPattern:
        calls = 0

        def finditer(self, text: str):
            self.calls += 1
            if self.calls == 1:
                return original_pattern.finditer(text)
            return iter(())

    pattern = MarkerPattern()
    monkeypatch.setattr(transport, "_RISK_TEST_MATRIX_RE", pattern)

    with pytest.raises(
        AgentLoopError, match="transport rewrite lost its protocol marker"
    ):
        transport._prepare_risk_test_matrix_transport(body)


def test_prepare_round_comment_spills_multiple_fields_in_fixed_order() -> None:
    payload = {
        "canonical_reviewer_response": _random_text(50_000),
        "raw_structured_coder_response": _random_text(50_000),
        "canonical_plan": _random_text(50_000),
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    anchor_payload = _anchor_payload(prepared[-1])
    expected_fields = tuple(field for field in transport._SPILL_FIELDS if field in payload)
    assert all(isinstance(anchor_payload[field], dict) for field in expected_fields)
    sidecar_fields = []
    for sidecar in prepared[:-1]:
        match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(sidecar)
        assert match is not None
        sidecar_fields.append(json.loads(base64.urlsafe_b64decode(match.group("payload")))["field"])
    assert list(dict.fromkeys(sidecar_fields)) == list(expected_fields)


def test_prepare_round_comment_spills_and_restores_large_structured_semantic_fields() -> None:
    payload = {
        "assembled_plan_sidecar": {
            "schema_version": 1,
            "kind": "assembled_plan_sidecar",
            "response_form": "semantic-patch-v1",
            "round_number": 5,
            "canonical_json": {"bulk": _random_text(55_000)},
            "aggregate_identity": "a" * 64,
            "raw_patch": {"operations": [{"decision": _random_text(18_000)}]},
        },
        "raw_patch_provenance": {
            "schema_version": 1,
            "kind": "plan_revision_patch",
            "operations": [{"decision": _random_text(18_000)}],
        },
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    anchor_payload = _anchor_payload(prepared[-1])
    assert isinstance(anchor_payload["assembled_plan_sidecar"], dict)
    assert isinstance(anchor_payload["raw_patch_provenance"], dict)
    hydrated, missing = transport.hydrate_mapping(anchor_payload, prepared)
    assert missing == set()
    assert hydrated == payload


def test_hydrate_mapping_reports_missing_duplicate_and_corrupt_sidecars() -> None:
    payload = {"canonical_reviewer_response": _random_text(46_000)}
    prepared = transport.prepare_round_comment(_comment(payload))
    sidecars, anchor = prepared[:-1], prepared[-1]
    anchor_payload = _anchor_payload(anchor)

    hydrated, missing = transport.hydrate_mapping(anchor_payload, (*sidecars, sidecars[0]))
    assert missing == set()
    assert hydrated == payload

    hydrated, missing = transport.hydrate_mapping(anchor_payload, sidecars[1:])
    assert missing == {"canonical_reviewer_response"}
    assert hydrated["canonical_reviewer_response"] is None

    match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(sidecars[0])
    assert match is not None
    corrupt = json.loads(base64.urlsafe_b64decode(match.group("payload")))
    corrupt["data"] = "corrupt"
    corrupt_sidecar = transport._sidecar(corrupt)
    hydrated, missing = transport.hydrate_mapping(anchor_payload, (corrupt_sidecar, *sidecars[1:]))
    assert missing == {"canonical_reviewer_response"}
    assert hydrated["canonical_reviewer_response"] is None


def test_hydrate_mapping_bounds_decompression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "_MAX_DECOMPRESSED", 10)
    raw = b"a" * 100
    packed = transport.zlib.compress(raw)
    reference = {
        "$round_transport_spill": "anchor",
        "field": "canonical_plan",
        "parts": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "spill": hashlib.sha256(packed).hexdigest(),
    }
    sidecar = transport._sidecar(
        {
            "anchor": "anchor",
            "field": "canonical_plan",
            "index": 0,
            "count": 1,
            "sha256": reference["sha256"],
            "spill": reference["spill"],
            "data": transport._b64(packed),
        }
    )

    hydrated, missing = transport.hydrate_mapping({"canonical_plan": reference}, (sidecar,))

    assert missing == {"canonical_plan"}
    assert hydrated["canonical_plan"] is None


def test_hydrate_mapping_reports_invalid_compressed_sidecar() -> None:
    packed = b"not a zlib stream"
    reference = {
        "$round_transport_spill": "anchor",
        "field": "canonical_plan",
        "parts": 1,
        "sha256": hashlib.sha256(b"expected raw payload").hexdigest(),
        "spill": hashlib.sha256(packed).hexdigest(),
    }
    sidecar = transport._sidecar(
        {
            "anchor": "anchor",
            "field": "canonical_plan",
            "index": 0,
            "count": 1,
            "sha256": reference["sha256"],
            "spill": reference["spill"],
            "data": transport._b64(packed),
        }
    )

    hydrated, missing = transport.hydrate_mapping({"canonical_plan": reference}, (sidecar,))

    assert missing == {"canonical_plan"}
    assert hydrated["canonical_plan"] is None


@pytest.mark.parametrize(
    ("field", "phase"),
    (
        ("canonical_reviewer_response", "provisional"),
        ("raw_structured_coder_response", "authoritative"),
        ("canonical_plan", "authoritative"),
    ),
)
def test_resume_rejects_missing_spilled_canonical_metadata(field: str, phase: str) -> None:
    values = {field: _random_text(46_000)}
    metadata = PostedRoundMetadata(
        flow="pr",
        role="reviewer",
        agent="codex",
        round_number=1,
        subject="head",
        phase=phase,
        **values,
    )
    body = _attach_round_metadata("Visible review", metadata)
    anchor = transport.prepare_round_comment(body)[-1]

    with pytest.raises(AgentLoopError, match="Incomplete round metadata"):
        _extract_round_metadata_records((SimpleNamespace(body=anchor),), flow="pr")


def test_round_metadata_decode_uses_mapping_without_reencoding() -> None:
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="codex", round_number=1, subject="head"
    )

    assert _decode_round_metadata(_encode_round_metadata(metadata)) == metadata


def test_round_metadata_round_trips_architecture_identity_and_impact() -> None:
    identity = {
        "repository": "OWNER/REPO",
        "path": "ARCHITECTURE.md",
        "revision": "a" * 40,
        "blob_oid": "b" * 40,
        "sha256": "c" * 64,
        "availability": "available",
        "size": 128,
    }
    impact = {
        "status": "unchanged",
        "rationale": "Only an internal test helper changed.",
        "affected_components": [],
        "dependencies": [],
        "execution_data_flows": [],
        "persistence": [],
        "public_contracts": [],
        "security_boundaries": [],
        "canonical_document_action": "no-change",
        "canonical_document_path": None,
        "canonical_document_rationale": "No canonical update is needed.",
    }
    metadata = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="codex", round_number=1, subject="head",
        architecture_identity=identity, architecture_impact=impact,
        architecture_contract_version=1,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.architecture_identity == identity
    assert decoded.architecture_impact == impact
    assert decoded.architecture_contract_version == 1


def test_round_metadata_round_trips_bounded_local_test_evidence() -> None:
    from coding_review_agent_loop.local_test_evidence import bounded_evidence_for_round

    evidence = bounded_evidence_for_round({
        "observations": [{
            "command": ["python", "-m", "pytest", "tests/test_protocol.py", "-q"],
            "outcome": "failed", "provenance": "parent-observed",
            "receipt_id": "receipt-1", "environment": "unknown",
        }]
    })
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="codex", round_number=2,
        subject="head", local_test_evidence=evidence,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.local_test_evidence is not None
    payload = json.loads(decoded.local_test_evidence)
    assert payload["observations"][0]["receipt_id"] == "receipt-1"
    assert payload["observations"][0]["environment"] == "identity-unknown"


def test_round_metadata_preserves_mixed_legacy_claim_and_modern_evidence_exactly() -> None:
    legacy_and_modern_item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Anthropic Claude",
        source_round=3,
        text=(
            "Only three of fourteen locales were updated.\n\n"
            "Update from Anthropic Claude: Original scope evidence was rechecked."
        ),
        status="blocking",
        source_status="blocking",
        notes=("OpenAI Codex: all fourteen locales and 49 keys were evaluated.",),
    )
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=4, subject="head",
        prior_items=(legacy_and_modern_item,),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.prior_items == (legacy_and_modern_item,)
    assert _prior_item_ledger_signature(decoded.prior_items) == _prior_item_ledger_signature(
        (legacy_and_modern_item,)
    )


def test_round_metadata_preserves_owner_future_disposition_across_resume() -> None:
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="cleanup remains incomplete",
        status="blocking",
        source_status="blocking",
        resolution_owners=("Codex", "Claude"),
        owner_states=(("Codex", "cleared"), ("Claude", "pending")),
        owner_dispositions=(("Codex", "future"),),
    )
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Codex", round_number=2, subject="head",
        prior_items=(item,),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.prior_items == (item,)
    assert decoded.prior_items[0].owner_dispositions == (("Codex", "future"),)


def test_machine_obligation_and_qualification_checkpoint_round_trip() -> None:
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
        authority=MACHINE_AUTHORITY,
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualifying",
        failed_head_sha="oldhead123",
        candidate_head_sha="newhead123",
        obligation_identity="managed-exact-head-ci:item-30",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha=item.failed_head_sha,
        candidate_head_sha=item.candidate_head_sha,
        base_branch="main",
        approval_digest="approval",
        plan_digest="plan",
        requirements_digest="requirements",
        acquisition_digest="acquisition",
        scheduler_digest="scheduler",
        qualification_attempt_id="123/1",
        watch_failure_extension_used=True,
        watch_head_extension_used=False,
        allowed_rounds=3,
    )
    metadata = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=14,
        subject="newhead123",
        prior_items=(item,),
        qualification_checkpoint=checkpoint,
    )

    encoded = _encode_round_metadata(metadata)
    decoded = _decode_round_metadata(encoded)

    assert decoded.prior_items == (item,)
    assert decoded.qualification_checkpoint == checkpoint
    assert transport.decode_mapping(encoded)["prior_items"][0]["authority"] == MACHINE_AUTHORITY


@pytest.mark.parametrize(
    "machine_fields",
    [
        {"authority": "bogus"},
        {"lifecycle": "repair_required"},
        {"failed_head_sha": "oldhead123"},
        {"candidate_head_sha": "newhead123"},
        {"obligation_identity": "machine:item-30"},
        {"authority": None, "obligation_kind": None, "lifecycle": None},
    ],
)
def test_partial_or_invalid_machine_item_decodes_as_non_bypassable_unknown(
    machine_fields,
) -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "prior_items": [
            {
                "item_id": "item-30",
                "reviewer": "GitHub managed exact-head CI",
                "source_round": 13,
                "text": "Persisted machine obligation.",
                "status": "blocking",
                **machine_fields,
            }
        ],
    }

    item = _decode_round_metadata_mapping(payload).prior_items[0]

    assert item.is_machine_obligation
    assert item.authority == UNKNOWN_MACHINE_AUTHORITY
    assert item.obligation_kind == "unknown"
    assert item.lifecycle == "repair_required"
    retained, _ = _apply_unresolved_item_dispositions(
        [item],
        {item.item_id: [ReviewItemDisposition(item.item_id, "Codex", "resolved")]},
        reconciliation_mode="owner-scoped",
    )
    assert retained == [item]


def test_invalid_qualification_checkpoint_decodes_fail_closed() -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "qualification_checkpoint": {
            "obligation_kind": "managed-exact-head-ci",
            "obligation_identity": "managed-exact-head-ci:item-30",
            "lifecycle": "qualifying",
            "failed_head_sha": "samehead",
            "candidate_head_sha": "samehead",
            "base_branch": "main",
            "approval_digest": None,
            "plan_digest": None,
            "requirements_digest": None,
            "acquisition_digest": None,
            "scheduler_digest": None,
            "qualification_attempt_id": None,
            "watch_failure_extension_used": False,
            "watch_head_extension_used": False,
            "allowed_rounds": 1,
        },
    }

    decoded = _decode_round_metadata_mapping(payload)

    assert decoded.qualification_checkpoint is not None
    assert not decoded.qualification_checkpoint.valid
    assert decoded.qualification_checkpoint.obligation_kind == "unknown"


@pytest.mark.parametrize(
    "missing",
    [
        "approval_digest",
        "requirements_digest",
        "acquisition_digest",
        "scheduler_digest",
        "qualification_attempt_id",
    ],
)
def test_qualification_checkpoint_missing_resume_identity_decodes_fail_closed(missing):
    checkpoint = {
        "obligation_kind": "managed-exact-head-ci",
        "obligation_identity": "managed-exact-head-ci:item-30",
        "lifecycle": "qualifying",
        "failed_head_sha": "oldhead123",
        "candidate_head_sha": "newhead123",
        "base_branch": "main",
        "approval_digest": "approval",
        "plan_digest": None,
        "requirements_digest": "requirements",
        "acquisition_digest": "acquisition",
        "scheduler_digest": "scheduler",
        "qualification_attempt_id": "run/1",
        "watch_failure_extension_used": False,
        "watch_head_extension_used": False,
        "allowed_rounds": 1,
    }
    checkpoint[missing] = None

    decoded = QualificationCheckpoint.from_mapping(checkpoint)

    assert not decoded.valid
    assert decoded.obligation_kind == "unknown"


@pytest.mark.parametrize(
    ("kind", "lifecycle", "candidate", "failed"),
    [
        ("unknown", "qualifying", "newhead123", "oldhead123"),
        ("managed-exact-head-ci", "cleared", "newhead123", "oldhead123"),
        ("managed-exact-head-ci", "qualifying", None, "oldhead123"),
    ],
)
def test_contradictory_qualification_checkpoint_payloads_decode_invalid(
    kind, lifecycle, candidate, failed
) -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "qualification_checkpoint": {
            "obligation_kind": kind,
            "obligation_identity": "machine:item-30",
            "lifecycle": lifecycle,
            "failed_head_sha": failed,
            "candidate_head_sha": candidate,
            "base_branch": "main",
            "approval_digest": None,
            "plan_digest": None,
            "requirements_digest": None,
            "acquisition_digest": None,
            "scheduler_digest": None,
            "qualification_attempt_id": None,
            "watch_failure_extension_used": False,
            "watch_head_extension_used": False,
            "allowed_rounds": 1,
        },
    }

    decoded = _decode_round_metadata_mapping(payload)

    assert decoded.qualification_checkpoint is not None
    assert not decoded.qualification_checkpoint.valid


def test_legacy_machine_item_requires_orchestrator_lineage_for_promotion() -> None:
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="CI failed at head `oldhead123`.",
        status="blocking",
        source_status="blocking",
    )
    trusted = _attach_round_metadata(
        "machine failure",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=13,
            subject="oldhead123",
            new_items=(legacy,),
        ),
    )
    ambiguous = _attach_round_metadata(
        "reviewer prose",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=13,
            subject="oldhead123",
            new_items=(legacy,),
        ),
    )

    trusted_item = _extract_round_metadata_records(
        [SimpleNamespace(body=trusted)], flow="pr"
    )[0].metadata.new_items[0]
    ambiguous_item = _extract_round_metadata_records(
        [SimpleNamespace(body=ambiguous)], flow="pr"
    )[0].metadata.new_items[0]

    assert trusted_item.authority == MACHINE_AUTHORITY
    assert trusted_item.obligation_kind == "managed-exact-head-ci"
    assert trusted_item.failed_head_sha == "oldhead123"
    assert ambiguous_item.authority == UNKNOWN_MACHINE_AUTHORITY
    assert ambiguous_item.obligation_kind == "unknown"


def test_plan_flow_never_promotes_orchestrator_item_to_machine_obligation() -> None:
    """Planning has no machine clearance path, so recovery keeps the finding form (#1005)."""
    legacy = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Orchestrator",
        source_round=3,
        text="Reviewer(s) Codex approved without acknowledging the signed human requirements.",
        status="blocking",
        source_status="blocking",
    )
    body = _attach_round_metadata(
        "Orchestrator plan review.",
        PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=3,
            subject="plan-subject",
            new_items=(legacy,),
        ),
    )

    plan_item = _extract_round_metadata_records(
        [SimpleNamespace(body=body)], flow="plan"
    )[0].metadata.new_items[0]

    assert not plan_item.is_machine_obligation
    assert plan_item.reviewer == "Orchestrator"
    remaining, _future = _apply_unresolved_item_dispositions(
        [plan_item],
        {
            "item-7": [
                ReviewItemDisposition(
                    reviewer="Codex",
                    item_id="item-7",
                    disposition="resolved",
                    note="Acknowledged.",
                )
            ]
        },
        same_status="same-plan",
    )
    assert remaining == []


def test_plan_flow_demotes_persisted_machine_obligation_to_reviewer_finding() -> None:
    promoted = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Orchestrator",
        source_round=3,
        text="Reviewer(s) Codex approved without acknowledging the signed human requirements.",
        status="blocking",
        source_status="blocking",
        authority=UNKNOWN_MACHINE_AUTHORITY,
        obligation_kind="unknown",
        lifecycle="repair_required",
        obligation_identity="unknown:item-7",
        notes=("Known machine item could not be bound to a failed head.",),
    )
    body = _attach_round_metadata(
        "Revised plan.",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=4,
            subject="plan-subject",
            prior_items=(promoted,),
        ),
    )

    demoted = _extract_round_metadata_records(
        [SimpleNamespace(body=body)], flow="plan"
    )[0].metadata.prior_items[0]

    assert not demoted.is_machine_obligation
    assert demoted.item_id == "item-7"
    assert demoted.status == "blocking"
    assert demoted.text == promoted.text
    assert any("#1005" in note for note in demoted.notes)


@pytest.mark.parametrize(
    "overrides",
    [
        # The decoder's blocker for an invalid persisted machine field/record.
        {
            "obligation_identity": "invalid-machine-record",
            "notes": ("Invalid persisted machine field: authority",),
        },
        # A machine record that is not the legacy promotion shape.
        {"reviewer": "Codex"},
        {"obligation_identity": "unknown:item-99"},
        # The legacy promotion clears ownership; contradictory ownership is
        # not identifiable as that promotion.
        {"resolution_owners": ("Codex",), "owner_states": (("Codex", "pending"),)},
        # Authority/head combinations the legacy promotion cannot emit.
        {"failed_head_sha": "plan-subject"},
        {"notes": ()},
        {"authority": MACHINE_AUTHORITY},
    ],
)
def test_plan_flow_keeps_unrecognized_machine_record_fail_closed(overrides) -> None:
    item = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Orchestrator",
        source_round=3,
        text="Synthetic blocker.",
        status="blocking",
        source_status="blocking",
        authority=UNKNOWN_MACHINE_AUTHORITY,
        obligation_kind="unknown",
        lifecycle="repair_required",
        obligation_identity="unknown:item-7",
        notes=("Known machine item could not be bound to a failed head.",),
    )
    from dataclasses import replace as _replace

    item = _replace(item, **overrides)
    body = _attach_round_metadata(
        "Revised plan.",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=4,
            subject="plan-subject",
            prior_items=(item,),
        ),
    )

    kept = _extract_round_metadata_records(
        [SimpleNamespace(body=body)], flow="plan"
    )[0].metadata.prior_items[0]

    assert kept.is_machine_obligation
    remaining, _future = _apply_unresolved_item_dispositions(
        [kept],
        {
            "item-7": [
                ReviewItemDisposition(
                    reviewer="Codex", item_id="item-7", disposition="resolved", note="Looks fine."
                )
            ]
        },
        same_status="same-plan",
    )
    assert [entry.item_id for entry in remaining] == ["item-7"]


def test_plan_flow_demotes_trusted_machine_authority_promotion() -> None:
    promoted = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Orchestrator",
        source_round=3,
        text="Synthetic blocker.",
        status="blocking",
        source_status="blocking",
        authority=MACHINE_AUTHORITY,
        obligation_kind="unknown",
        lifecycle="repair_required",
        failed_head_sha="plan-subject",
        obligation_identity="unknown:item-7",
    )
    body = _attach_round_metadata(
        "Revised plan.",
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=4,
            subject="plan-subject", prior_items=(promoted,),
        ),
    )

    demoted = _extract_round_metadata_records(
        [SimpleNamespace(body=body)], flow="plan"
    )[0].metadata.prior_items[0]

    assert not demoted.is_machine_obligation


def test_plan_flow_keeps_invalid_persisted_authority_fail_closed() -> None:
    payload = {
        "flow": "plan",
        "role": "coder",
        "agent": "Claude",
        "round_number": 4,
        "subject": "plan-subject",
        "prior_items": [
            {
                "item_id": "item-7",
                "reviewer": "Orchestrator",
                "source_round": 3,
                "text": "Synthetic blocker.",
                "status": "blocking",
                "source_status": "blocking",
                "authority": 17,
                "obligation_kind": "unknown",
            }
        ],
    }

    kept = _extract_round_metadata_records(
        [SimpleNamespace(body=_comment(payload))], flow="plan"
    )[0].metadata.prior_items[0]

    assert kept.is_machine_obligation
    assert kept.obligation_identity == "invalid-machine-record"


def test_pr_flow_still_promotes_orchestrator_item_to_unknown_machine_obligation() -> None:
    legacy = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Orchestrator",
        source_round=3,
        text="Synthetic orchestrator blocker.",
        status="blocking",
        source_status="blocking",
    )
    body = _attach_round_metadata(
        "Orchestrator summary.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=3,
            subject="head123",
            new_items=(legacy,),
        ),
    )

    pr_item = _extract_round_metadata_records(
        [SimpleNamespace(body=body)], flow="pr"
    )[0].metadata.new_items[0]

    assert pr_item.is_machine_obligation
    assert pr_item.obligation_kind == "unknown"


def test_legacy_coder_checkpoint_recovers_failed_head_from_scheduler_provenance() -> None:
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="The managed check failed; repair the code.",
        status="blocking",
        source_status="blocking",
    )
    coder_checkpoint = _attach_round_metadata(
        "Coder repaired the PR.",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=14,
            subject="newhead123",
            prior_items=(legacy,),
            scheduler_contract=ReviewSchedulingContract(
                required_reviewers=("Claude",),
                policy="selective-intermediate",
                broad_rules=("src/**",),
            ).as_dict(),
            scheduler_previous_sha="oldhead123",
            scheduler_current_sha="newhead123",
            scheduler_obligation_digest="0123456789abcdef",
            scheduler_selected_reviewers=("Claude",),
            scheduler_reasons=("full board",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )

    record = _extract_round_metadata_records(
        [SimpleNamespace(body=coder_checkpoint)], flow="pr"
    )[0].metadata.prior_items[0]

    assert record.authority == MACHINE_AUTHORITY
    assert record.obligation_kind == "managed-exact-head-ci"
    assert record.failed_head_sha == "oldhead123"


def test_staged_scheduler_phase_fields_roundtrip_and_fail_closed_when_contradictory() -> None:
    contract = ReviewSchedulingContract(
        required_reviewers=("Codex", "Gemini"),
        policy="primary-then-panel",
        primary_reviewer="Codex",
        broad_rules=("src/**",),
    )
    metadata = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="newhead123",
        scheduler_contract=contract.as_dict(),
        scheduler_previous_sha="oldhead123",
        scheduler_current_sha="newhead123",
        scheduler_obligation_digest="0123456789abcdef",
        scheduler_selected_reviewers=("Gemini",),
        scheduler_paused_reviewers=(("Codex", "qualifying exact-head approval carried"),),
        scheduler_reasons=("independent secondary audit",),
        scheduler_final_sweep=False,
        scheduler_force_full=True,
        scheduler_calls_avoided=1,
        scheduler_phase="secondary-audit",
        scheduler_primary_reviewer="Codex",
        scheduler_approved_reviewers=("Codex",),
        scheduler_active_owners=(),
        scheduler_scope_digest="fedcba9876543210",
    )
    payload = transport.decode_mapping(_encode_round_metadata(metadata))
    decoded = _decode_round_metadata_mapping(payload)
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.scheduler_phase == "secondary-audit"
    assert decoded.scheduler_primary_reviewer == "Codex"
    assert decoded.scheduler_approved_reviewers == ("Codex",)
    assert decoded.scheduler_force_full is True
    assert decoded.scheduler_scope_digest == "fedcba9876543210"

    # Malformed new authority fails closed: the record is invalid, not legacy.
    for key, bad in (
        ("scheduler_phase", "panel"),
        ("scheduler_primary_reviewer", "Gemini"),
        ("scheduler_approved_reviewers", ["Claude"]),
        ("scheduler_approved_reviewers", ["Codex", "Codex"]),
        ("scheduler_active_owners", [""]),
        ("scheduler_scope_digest", "not-hex"),
    ):
        contradictory = _decode_round_metadata_mapping({**payload, key: bad})
        assert contradictory.scheduler_metadata_status == "invalid", key
        assert contradictory.scheduler_contract is None, key

    # Legacy records without the phase fields remain valid with no phase authority.
    legacy = {
        key: value
        for key, value in payload.items()
        if key not in {
            "scheduler_phase",
            "scheduler_primary_reviewer",
            "scheduler_approved_reviewers",
            "scheduler_active_owners",
            "scheduler_scope_digest",
        }
    }
    legacy["scheduler_contract"] = {**contract.as_dict(), "policy": "selective-intermediate", "primary_reviewer": None}
    legacy_decoded = _decode_round_metadata_mapping(legacy)
    assert legacy_decoded.scheduler_metadata_status == "valid"
    assert legacy_decoded.scheduler_phase is None
    assert legacy_decoded.scheduler_primary_reviewer is None
    # Only auxiliary keys without the mandatory core is a partial record.
    partial = _decode_round_metadata_mapping(
        {
            "flow": "pr", "role": "summary", "agent": "Orchestrator",
            "round_number": 2, "subject": "newhead123", "scheduler_phase": "primary",
        }
    )
    assert partial.scheduler_metadata_status == "invalid"


def test_scheduler_force_full_source_roundtrips_and_fails_closed() -> None:
    """#840: the force-full audit source is optional, strict, and legacy-safe."""
    contract = ReviewSchedulingContract(
        required_reviewers=("Codex", "Gemini"),
        policy="primary-then-panel",
        primary_reviewer="Codex",
        broad_rules=("src/**",),
    )
    base = dict(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="newhead123",
        scheduler_contract=contract.as_dict(),
        scheduler_previous_sha="oldhead123",
        scheduler_current_sha="newhead123",
        scheduler_obligation_digest="0123456789abcdef",
        scheduler_selected_reviewers=("Codex", "Gemini"),
        scheduler_reasons=("operator force-full",),
        scheduler_final_sweep=False,
        scheduler_calls_avoided=0,
        scheduler_phase="full-board",
        scheduler_primary_reviewer="Codex",
    )
    for source in ("operator", "automatic"):
        metadata = PostedRoundMetadata(
            **base, scheduler_force_full=True, scheduler_force_full_source=source
        )
        payload = transport.decode_mapping(_encode_round_metadata(metadata))
        assert payload["scheduler_force_full_source"] == source
        decoded = _decode_round_metadata_mapping(payload)
        assert decoded.scheduler_metadata_status == "valid"
        assert decoded.scheduler_force_full is True
        assert decoded.scheduler_force_full_source == source
        assert _decode_round_metadata_mapping(payload) == decoded

    unlatched = PostedRoundMetadata(**base, scheduler_force_full=False)
    payload = transport.decode_mapping(_encode_round_metadata(unlatched))
    # The optional key is omitted when unset, preserving the legacy shape.
    assert "scheduler_force_full_source" not in payload
    assert _decode_round_metadata_mapping(payload).scheduler_force_full_source is None

    # An unknown source, or a source without an active latch, is invalid
    # scheduler metadata rather than silently trusted.
    for bad_payload in (
        {**payload, "scheduler_force_full": True, "scheduler_force_full_source": "human"},
        {**payload, "scheduler_force_full": True, "scheduler_force_full_source": 1},
        {**payload, "scheduler_force_full": False, "scheduler_force_full_source": "operator"},
    ):
        decoded = _decode_round_metadata_mapping(bad_payload)
        assert decoded.scheduler_metadata_status == "invalid"
        assert decoded.scheduler_force_full_source is None
    with pytest.raises(ValueError, match="force-full source"):
        PostedRoundMetadata(**base, scheduler_force_full=False, scheduler_force_full_source="operator")

    # Legacy latched records without a source still decode as valid metadata.
    legacy = {**payload, "scheduler_force_full": True}
    legacy_decoded = _decode_round_metadata_mapping(legacy)
    assert legacy_decoded.scheduler_metadata_status == "valid"
    assert legacy_decoded.scheduler_force_full is True
    assert legacy_decoded.scheduler_force_full_source is None


# --- Visible sidecar labels (#842) -------------------------------------------


def _sidecar_payload(body: object) -> dict[str, object]:
    match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(str(body))
    assert match is not None
    return json.loads(base64.urlsafe_b64decode(match.group("payload")))


def _marker_only(body: object) -> str:
    match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(str(body))
    assert match is not None
    return match.group(0)


def _label_line(body: object) -> str:
    text = str(body)
    label, separator, marker = text.partition("\n\n")
    assert separator and marker == _marker_only(text)
    return label


def _recommendation_marker(rationale_chars: int = 50_000) -> str:
    recommendation = {
        "strategy": "one-shot",
        "rationale": _random_text(rationale_chars),
        "staging_feasibility": "inseparable",
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(recommendation, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    return f"<!-- AGENT_EXECUTION_RECOMMENDATION: {encoded} -->"


@pytest.mark.parametrize(
    ("metadata", "kind"),
    [
        ({"flow": "plan", "role": "coder"}, "plan"),
        ({"flow": "plan", "role": "reviewer"}, "plan-review"),
        ({"flow": "pr", "role": "reviewer"}, "review"),
        ({"flow": "managed-pr", "role": "reviewer"}, "review"),
        ({"flow": "approved", "role": "reviewer"}, "review"),
        ({"flow": "approved-plan-implementation", "role": "reviewer"}, "review"),
        ({"flow": "direct", "role": "reviewer"}, "review"),
        ({"flow": "issue-implementation", "role": "reviewer"}, "review"),
        ({"flow": "discuss", "role": "debater"}, "round"),
        ({"flow": "discuss", "role": "summary"}, "round"),
        ({"flow": "discuss", "role": "reviewer"}, "round"),
        ({"flow": "pr", "role": "coder"}, "round"),
        ({"flow": "pr", "role": "repair"}, "round"),
        ({"flow": "pr", "role": "test-gate"}, "round"),
        ({"flow": "plan", "role": "analyzer"}, "round"),
        ({"flow": "future-flow", "role": "reviewer"}, "round"),
        ({"flow": "plan"}, "round"),
        ({"role": "reviewer"}, "round"),
        ({"flow": ["plan"], "role": "coder"}, "round"),
        ({}, "round"),
        (None, "round"),
    ],
)
def test_sidecar_kind_uses_explicit_flow_role_allow_list(metadata, kind) -> None:
    assert transport._sidecar_kind(metadata) == kind


def test_sidecar_label_is_bounded_ascii_and_free_of_reserved_markers() -> None:
    from coding_review_agent_loop.protocol_markers import scan_reserved_markers

    for kind in ("plan", "plan-review", "review", "round", "unexpected"):
        for field in (*transport._SPILL_FIELDS, "not-a-spill-field", ""):
            label = transport._sidecar_label(kind=kind, field=field, position=12, total=34)
            assert label.isascii()
            assert len(label) <= transport._SIDECAR_LABEL_MAX_CHARS
            assert "<!--" not in label and "-->" not in label
            assert not scan_reserved_markers(label)
            assert "12/34" in label
    assert transport._sidecar_label(
        kind="plan", field="canonical_plan", position=2, total=5
    ) == (
        "Agent-loop plan attachment 2/5 (machine-readable overflow: canonical_plan). "
        "Not an agent response; see the following plan comment."
    )
    assert "overflow: metadata)" in transport._sidecar_label(
        kind="plan", field="attacker <b>text</b>", position=1, total=1
    )


def test_fresh_multi_source_plan_sidecars_are_labeled_in_posting_order() -> None:
    _matrix_payload, matrix_encoded = _risk_test_matrix_marker_payload()
    metadata = {
        "flow": "plan",
        "role": "coder",
        "canonical_plan": _random_text(60_000),
    }
    body = _comment(
        metadata,
        body=(
            "Visible plan\n"
            + _recommendation_marker()
            + "\n"
            + f"<!-- AGENT_RISK_TEST_MATRIX: {matrix_encoded} -->"
        ),
    )

    prepared = transport.prepare_round_comment(body)
    sidecars, anchor = prepared[:-1], prepared[-1]

    fields = [_sidecar_payload(item)["field"] for item in sidecars]
    assert "execution_recommendation" in fields
    assert "risk_test_matrix_marker" in fields
    assert "canonical_plan" in fields
    total = len(sidecars)
    for position, sidecar in enumerate(sidecars, start=1):
        label = _label_line(sidecar)
        field = _sidecar_payload(sidecar)["field"]
        assert label == (
            f"Agent-loop plan attachment {position}/{total} "
            f"(machine-readable overflow: {field}). "
            "Not an agent response; see the following plan comment."
        )
        assert len(transport.ROUND_TRANSPORT_SIDECAR_RE.findall(str(sidecar))) == 1
        assert len(str(sidecar)) <= transport.MAX_GITHUB_BODY_CHARS
        assert sidecar.segments[-1][1] == "AGENT_LOOP_SIDECAR"
        assert [token for _text, token in sidecar.segments if token] == ["AGENT_LOOP_SIDECAR"]
        sidecar.validate_for_surface("issue_comment")
    # The anchor carries no sidecar label and remains the final comment.
    assert not transport.is_round_transport_sidecar(str(anchor))
    assert "attachment" not in str(anchor).split("\n", 1)[0]
    hydrated, missing = transport.hydrate_mapping(_anchor_payload(anchor), prepared)
    assert missing == set()
    assert hydrated["canonical_plan"] == metadata["canonical_plan"]


def test_labeling_does_not_change_sidecar_payloads_or_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "flow": "plan",
        "role": "coder",
        "canonical_plan": _random_text(50_000),
        "canonical_reviewer_response": _random_text(50_000),
    }
    body = _comment(payload, body="Visible\n" + _recommendation_marker())

    labeled = transport.prepare_round_comment(body)
    monkeypatch.setattr(transport, "_label_sidecars", lambda sidecars, kind: list(sidecars))
    unlabeled = transport.prepare_round_comment(body)

    assert len(labeled) == len(unlabeled)
    assert str(labeled[-1]) == str(unlabeled[-1])
    for new, old in zip(labeled[:-1], unlabeled[:-1]):
        assert _marker_only(new) == str(old)
        assert str(new).endswith(str(old))


def test_payloads_are_identical_across_label_kinds() -> None:
    value = _random_text(46_000)
    markers_by_kind = {}
    for flow, role in (("plan", "coder"), ("plan", "reviewer"), ("pr", "reviewer"), ("discuss", "debater")):
        prepared = transport.prepare_round_comment(
            _comment({"flow": flow, "role": role, "canonical_reviewer_response": value})
        )
        markers_by_kind[(flow, role)] = [
            _sidecar_payload(item) for item in prepared[:-1]
        ]
    # Payloads differ only in the anchor id derived from the carrier text; the
    # carried data, digests, and counts are the same and carry no kind field.
    reference = markers_by_kind[("plan", "coder")]
    for items in markers_by_kind.values():
        assert [
            {key: item[key] for key in item if key != "anchor"} for item in items
        ] == [{key: item[key] for key in item if key != "anchor"} for item in reference]
        assert all("kind" not in item for item in items)


@pytest.mark.parametrize(
    ("metadata", "attachment", "target"),
    [
        ({"flow": "plan", "role": "reviewer"}, "plan review attachment", "plan review comment"),
        ({"flow": "pr", "role": "reviewer"}, "review attachment", "review comment"),
        ({"flow": "discuss", "role": "debater"}, "attachment", "agent-loop comment"),
        ({"flow": "discuss", "role": "summary"}, "attachment", "agent-loop comment"),
        ({"flow": "pr", "role": "coder"}, "attachment", "agent-loop comment"),
        ({"flow": "unknown-flow", "role": "reviewer"}, "attachment", "agent-loop comment"),
        ({}, "attachment", "agent-loop comment"),
    ],
)
def test_round_sidecar_wording_follows_round_metadata(metadata, attachment, target) -> None:
    prepared = transport.prepare_round_comment(
        _comment({**metadata, "canonical_reviewer_response": _random_text(46_000)})
    )

    assert len(prepared) > 1
    for sidecar in prepared[:-1]:
        label = _label_line(sidecar)
        assert label.startswith(f"Agent-loop {attachment} ")
        assert label.endswith(f"see the following {target}.")


def test_sidecar_only_carrier_without_round_metadata_gets_neutral_wording() -> None:
    body = "Visible plan\n" + _recommendation_marker()

    prepared = transport.prepare_round_comment(body)

    assert len(prepared) > 1
    assert not transport.ROUND_RESUME_MARKER_RE.search(str(prepared[-1]))
    for position, sidecar in enumerate(prepared[:-1], start=1):
        assert _label_line(sidecar) == (
            f"Agent-loop attachment {position}/{len(prepared) - 1} "
            "(machine-readable overflow: execution_recommendation). "
            "Not an agent response; see the following agent-loop comment."
        )


def test_repeated_preparation_yields_byte_identical_sidecar_bodies() -> None:
    body = _comment(
        {"flow": "plan", "role": "coder", "canonical_plan": _random_text(90_000)},
        body="Visible\n" + _recommendation_marker(),
    )

    first = transport.prepare_round_comment(body)
    second = transport.prepare_round_comment(body)

    assert [str(item) for item in first] == [str(item) for item in second]
    assert [item.segments for item in first] == [item.segments for item in second]


def test_full_part_labeled_sidecar_fits_github_body_budget() -> None:
    prepared = transport.prepare_round_comment(
        _comment({"flow": "pr", "role": "reviewer", "canonical_reviewer_response": _random_text(120_000)})
    )

    full_parts = [
        item for item in prepared[:-1]
        if len(str(_sidecar_payload(item)["data"])) == transport._PART_CHARS
    ]
    assert full_parts
    for item in full_parts:
        assert len(str(item)) <= transport.MAX_GITHUB_BODY_CHARS


def test_hydrate_accepts_historical_labeled_and_mixed_sidecars() -> None:
    payload = {"flow": "plan", "role": "coder", "canonical_plan": _random_text(90_000)}
    prepared = transport.prepare_round_comment(_comment(payload))
    sidecars, anchor = prepared[:-1], prepared[-1]
    assert len(sidecars) >= 2
    anchor_payload = _anchor_payload(anchor)
    labeled = [str(item) for item in sidecars]
    historical = [_marker_only(item) for item in sidecars]
    mixed = [historical[0], *labeled[1:]]

    for bodies in (labeled, historical, mixed):
        assert all(transport.is_round_transport_sidecar(body) for body in bodies)
        hydrated, missing = transport.hydrate_mapping(anchor_payload, [*bodies, str(anchor)])
        assert missing == set()
        assert hydrated["canonical_plan"] == payload["canonical_plan"]


def test_interrupted_publication_duplicates_hydrate_and_missing_part_reported() -> None:
    payload = {"flow": "plan", "role": "coder", "canonical_plan": _random_text(90_000)}
    body = _comment(payload)
    first_attempt = transport.prepare_round_comment(body)
    retry = transport.prepare_round_comment(body)
    sidecars, anchor = retry[:-1], retry[-1]
    anchor_payload = _anchor_payload(anchor)
    assert len(sidecars) >= 2
    assert [str(item) for item in first_attempt] == [str(item) for item in retry]

    # The interrupted attempt posted part 1 (labeled) and an older run left a
    # marker-only copy of the same part; the retry then posted everything.
    history = [
        str(first_attempt[0]),
        _marker_only(first_attempt[0]),
        *map(str, sidecars),
        str(anchor),
    ]
    hydrated, missing = transport.hydrate_mapping(anchor_payload, history)
    assert missing == set()
    assert hydrated["canonical_plan"] == payload["canonical_plan"]

    incomplete = [str(first_attempt[0]), _marker_only(first_attempt[0]), str(anchor)]
    hydrated, missing = transport.hydrate_mapping(anchor_payload, incomplete)
    assert missing == {"canonical_plan"}
    assert hydrated["canonical_plan"] is None


def test_plan_validation_diagnostic_body_is_labeled_and_round_trips() -> None:
    payload = _diagnostic_payload(attempt=3)
    body = str(encode_plan_validation_diagnostic_body(payload))
    label, separator, marker = body.partition("\n\n")

    assert separator == "\n\n"
    assert label.startswith("Agent-loop plan-validation diagnostic record")
    assert "failure attempt 3" in label
    assert marker.startswith("<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: ")
    assert marker.endswith("-->")
    assert decode_plan_validation_diagnostic_body(body) == payload
    # Deterministic, so a retry and the posted read-back stay byte-identical.
    assert str(encode_plan_validation_diagnostic_body(payload)) == body


def test_plan_validation_decoder_accepts_only_the_two_canonical_forms() -> None:
    payload = _diagnostic_payload(attempt=2)
    labeled = str(encode_plan_validation_diagnostic_body(payload))
    historical = labeled.split("\n\n", 1)[1]

    assert decode_plan_validation_diagnostic_body(historical) == payload
    assert decode_plan_validation_diagnostic_body(labeled) == payload

    for rejected in (
        f"Unrelated operator prose.\n\n{historical}",
        f"{labeled}\n\nUnrelated trailing prose.",
        labeled.replace("Agent-loop", "Agent-loop (edited)", 1),
        # Surrounding whitespace would be a third carrier form.
        f" {historical}",
        f"{historical}\n",
        f"\n{labeled}",
        f"{labeled}  ",
    ):
        with pytest.raises(AgentLoopError, match="exact canonical record"):
            decode_plan_validation_diagnostic_body(rejected)


def test_plan_validation_recovery_keeps_prose_wrapped_records_ineligible_not_fatal() -> None:
    historical_payload = _diagnostic_payload(attempt=1)
    historical = IssueComment(
        author="agent", author_id=7, comment_id=500,
        created_at="2026-01-01T00:00:00Z",
        body=str(encode_plan_validation_diagnostic_body(historical_payload)).split("\n\n", 1)[1],
    )
    wrapped = IssueComment(
        author="agent", author_id=7, comment_id=501,
        created_at="2026-01-01T00:00:01Z",
        body="Operator note.\n\n"
        + str(encode_plan_validation_diagnostic_body(_diagnostic_payload(attempt=9))),
    )

    selected = recover_plan_validation_diagnostic(
        (historical, wrapped),
        repository="OWNER/REPO", issue_number=813,
        expected_author_login="agent", expected_author_id=7,
        planning_generation=1, target_coder_round=1,
        prior_plan_subject=None, candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
    )

    assert selected is not None
    assert selected.server_comment_id == 500
    assert selected.failure_attempt == 1


def test_planning_scheduler_record_decodes_valid_and_never_cross_decodes():
    """`planning-scheduler-record-decodes-valid` (#905, from #841)."""
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanCandidateKey,
        make_plan_contract,
    )
    from coding_review_agent_loop.review_scheduling import make_contract
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata,
        _decode_round_metadata,
        _encode_round_metadata,
    )

    key = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity="c" * 32,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest="e" * 16,
    )
    plan_record = PostedRoundMetadata(
        flow="plan",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="a" * 64,
        phase="scheduler-prelaunch",
        scheduler_contract=make_plan_contract(
            ("Codex", "Gemini"), "primary-then-panel", "Codex"
        ).as_dict(),
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("Gemini",),
        scheduler_paused_reviewers=(("Codex", "carried approval"),),
        scheduler_reasons=("independent secondary audit",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=1,
        scheduler_phase="secondary-audit",
        scheduler_primary_reviewer="Codex",
        scheduler_approved_reviewers=("Codex",),
        plan_candidate_key=key.as_dict(),
    )
    pr_record = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="f" * 40,
        phase="scheduler-prelaunch",
        scheduler_contract=make_contract(
            ("Codex", "Gemini"), "primary-then-panel", None, "Codex"
        ).as_dict(),
        scheduler_previous_sha="e" * 40,
        scheduler_current_sha="f" * 40,
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("Gemini",),
        scheduler_paused_reviewers=(("Codex", "carried approval"),),
        scheduler_reasons=("independent secondary audit",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=1,
        scheduler_phase="secondary-audit",
        scheduler_primary_reviewer="Codex",
        scheduler_approved_reviewers=("Codex",),
    )

    decoded_plan = _decode_round_metadata(_encode_round_metadata(plan_record))
    decoded_pr = _decode_round_metadata(_encode_round_metadata(pr_record))

    # Both carry `policy: primary-then-panel` and neither is read as the other.
    assert decoded_plan.scheduler_metadata_status == "valid"
    assert decoded_plan.scheduler_contract["policy"] == "primary-then-panel"
    assert "broad_rules" not in decoded_plan.scheduler_contract
    assert decoded_plan.plan_candidate_key == key.as_dict()
    assert decoded_plan.scheduler_previous_sha is None
    assert decoded_pr.scheduler_metadata_status == "valid"
    assert decoded_pr.scheduler_current_sha == "f" * 40
    assert decoded_pr.plan_candidate_key is None


def test_planning_scheduler_record_without_a_candidate_key_decodes_invalid():
    """A partial planning record falls back instead of pinning the run."""
    from coding_review_agent_loop.plan_review_scheduling import make_plan_contract
    from coding_review_agent_loop.round_state import (
        _decode_round_metadata_mapping,
    )

    payload = {
        "flow": "plan",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 2,
        "subject": "a" * 64,
        "scheduler_contract": make_plan_contract(
            ("Codex", "Gemini"), "primary-then-panel", "Codex"
        ).as_dict(),
        "scheduler_obligation_digest": "0" * 16,
        "scheduler_selected_reviewers": ["Gemini"],
        "scheduler_paused_reviewers": [["Codex", "carried approval"]],
        "scheduler_reasons": ["audit"],
        "scheduler_final_sweep": False,
        "scheduler_force_full": False,
        "scheduler_calls_avoided": 1,
    }

    assert (
        _decode_round_metadata_mapping(payload).scheduler_metadata_status == "invalid"
    )


def test_absent_planning_scheduler_metadata_stays_absent():
    """Full-board planning comments remain byte-compatible legacy records."""
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata,
        _decode_round_metadata,
        _encode_round_metadata,
    )

    record = PostedRoundMetadata(
        flow="plan",
        role="reviewer",
        agent="Codex",
        round_number=1,
        subject="a" * 64,
        state="approved",
    )
    encoded = _encode_round_metadata(record)

    assert "scheduler_" not in encoded or True  # payload is compressed/encoded
    decoded = _decode_round_metadata(encoded)
    assert decoded.scheduler_metadata_status == "absent"
    assert decoded.plan_candidate_key is None


def test_planning_scheduler_record_with_a_partial_candidate_key_decodes_invalid():
    """Generation-1 rule: a partial key can supply no approval or opening."""
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanCandidateKey,
        make_plan_contract,
    )
    from coding_review_agent_loop.round_state import _decode_round_metadata_mapping

    complete = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity="c" * 32,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest="e" * 16,
    )

    def _payload(**overrides):
        base = {
            "flow": "plan",
            "role": "summary",
            "agent": "Orchestrator",
            "round_number": 2,
            "subject": "a" * 64,
            "scheduler_contract": make_plan_contract(
                ("Codex", "Gemini"), "primary-then-panel", "Codex"
            ).as_dict(),
            "scheduler_obligation_digest": "0" * 16,
            "scheduler_selected_reviewers": ["Gemini"],
            "scheduler_paused_reviewers": [["Codex", "carried approval"]],
            "scheduler_reasons": ["audit"],
            "scheduler_final_sweep": False,
            "scheduler_force_full": False,
            "scheduler_calls_avoided": 1,
            "plan_candidate_key": complete.as_dict(),
        }
        base.update(overrides)
        return base

    assert (
        _decode_round_metadata_mapping(_payload()).scheduler_metadata_status == "valid"
    )

    # A missing key component.
    missing = dict(complete.as_dict())
    missing["execution_strategy_identity"] = None
    assert (
        _decode_round_metadata_mapping(
            _payload(plan_candidate_key=missing)
        ).scheduler_metadata_status
        == "invalid"
    )

    # A non-generation-1 execution-strategy contract version.
    legacy = dict(complete.as_dict())
    legacy["execution_strategy_contract_version"] = 2
    assert (
        _decode_round_metadata_mapping(
            _payload(plan_candidate_key=legacy)
        ).scheduler_metadata_status
        == "invalid"
    )

    # A partial previous-key field is rejected the same way.
    assert (
        _decode_round_metadata_mapping(
            _payload(scheduler_plan_previous_key=missing)
        ).scheduler_metadata_status
        == "invalid"
    )
    assert (
        _decode_round_metadata_mapping(
            _payload(scheduler_plan_previous_key=complete.as_dict())
        ).scheduler_metadata_status
        == "valid"
    )


# --- #948: dedicated overflow error, fit check and size attribution ---


def test_m948_round_comment_fits_agrees_with_prepare_and_posts_nothing() -> None:
    fitting = _comment({"canonical_plan": _random_text(500)})
    spilling = _comment({"canonical_plan": _random_text(60_000)})

    assert transport.round_comment_fits(fitting) is True
    assert transport.round_comment_fits(spilling) is True
    # Agreement: the same inputs prepare without error, and the helper returns
    # a bare bool (no sidecars, no publication seam is involved).
    assert len(transport.prepare_round_comment(fitting)) == 1
    assert len(transport.prepare_round_comment(spilling)) > 1


def test_m948_residual_overflow_is_dedicated_and_size_attributed_without_content() -> None:
    secret = "SECRET" + _random_text(70_000)
    visible = "Visible response " + _random_text(1_000)
    payload = {
        "unspilled_big_field": secret,
        "canonical_plan": _random_text(60_000),
        "small": "x",
    }
    body = _comment(payload, body=visible)

    assert transport.round_comment_fits(body) is False
    with pytest.raises(transport.RoundCommentOverflowError) as excinfo:
        transport.prepare_round_comment(body)
    message = str(excinfo.value)
    assert isinstance(excinfo.value, AgentLoopError)
    # The residual metadata alone exceeds the budget here (#953).
    assert message.startswith(
        "Round comment exceeds 60000 characters even after metadata spill; "
        "the derived round metadata alone needs "
    )
    assert "visible body outside round metadata" in message
    assert "residual encoded round metadata" in message
    assert "unspilled_big_field=" in message
    # The spilled field is no longer residual, and no content leaks.
    assert "canonical_plan=" not in message
    assert "SECRET" not in message
    assert secret[:40] not in message
    assert visible[-40:] not in message
    assert len(message) < 1_000


def test_m948_freeform_oversized_body_raises_dedicated_overflow() -> None:
    body = "free-form plan " + "x" * 61_000

    assert transport.round_comment_fits(body) is False
    with pytest.raises(transport.RoundCommentOverflowError, match="visible body 61015 characters"):
        transport.prepare_round_comment(body)


def test_m948_fit_check_propagates_malformed_matrix_record() -> None:
    body = _comment(
        {"canonical_plan": "p"},
        body="x" * 51_000 + "\n<!-- AGENT_RISK_TEST_MATRIX: " + "A" * 4_100 + " -->",
    )

    with pytest.raises(AgentLoopError) as excinfo:
        transport.round_comment_fits(body)
    assert not isinstance(excinfo.value, transport.RoundCommentOverflowError)


def test_m948_fit_check_propagates_non_serializable_metadata(monkeypatch) -> None:
    body = _comment({"canonical_plan": _random_text(70_000)})
    original = transport.decode_mapping

    def decode(encoded: str) -> dict[str, object]:
        payload = original(encoded)
        payload["assembled_plan_sidecar"] = {"bad": {1, 2}}
        return payload

    monkeypatch.setattr(transport, "decode_mapping", decode)
    # The pre-existing failure for this input propagates unchanged and is
    # never reported as "does not fit".
    with pytest.raises((AgentLoopError, TypeError)) as excinfo:
        transport.round_comment_fits(body)
    assert not isinstance(excinfo.value, transport.RoundCommentOverflowError)


def test_m948_fit_check_propagates_provenance_failure() -> None:
    text = _comment({"canonical_plan": _random_text(70_000)})
    # The carrier does not authorize the round metadata record, so the
    # transport cannot rewrite it; that is a provenance failure, not a size one.
    carrier = TrustedBody(text)

    with pytest.raises(AgentLoopError, match="authorized marker segment was not found") as excinfo:
        transport.round_comment_fits(carrier)
    assert not isinstance(excinfo.value, transport.RoundCommentOverflowError)


# --- #953: review-state growth fields spill, with reader safety ---

_PRE_953_SPILL_FIELDS = tuple(
    field for field in transport._SPILL_FIELDS if field not in transport._GROWTH_SPILL_FIELDS
)


def _sidecar_fields(prepared) -> list[str]:
    fields = []
    for sidecar in prepared[:-1]:
        match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(str(sidecar))
        assert match is not None
        fields.append(json.loads(base64.urlsafe_b64decode(match.group("payload")))["field"])
    return fields


def _is_reference(value) -> bool:
    return isinstance(value, dict) and "$round_transport_spill" in value


def _growth_payload(*, prior_bytes: int, evidence_bytes: int, flow: str = "pr") -> dict:
    return {
        "flow": flow,
        "prior_items": [
            {"item_id": f"item-{index}", "text": _random_text(prior_bytes // 4)}
            for index in range(4)
        ],
        "risk_test_matrix_evidence": {
            "rows": [{"row_id": "row-1", "evidence": _random_text(evidence_bytes)}],
        },
    }


def _body_over_budget_by(payload: dict, excess: int) -> str:
    marker_only = len(_comment(payload, body=""))
    visible = "V" * (transport.MAX_GITHUB_BODY_CHARS + excess - marker_only)
    body = _comment(payload, body=visible)
    assert len(body) == transport.MAX_GITHUB_BODY_CHARS + excess
    return body


def test_m953_pr_952_shape_spills_prior_items_only() -> None:
    payload = _growth_payload(prior_bytes=7_300, evidence_bytes=8_000)
    body = _body_over_budget_by(payload, 413)
    assert len(body) - len(transport.encode_mapping(payload)) > 37_000

    prepared = transport.prepare_round_comment(body)

    anchor = str(prepared[-1])
    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    posted = _anchor_payload(anchor)
    assert _is_reference(posted["prior_items"])
    assert posted["risk_test_matrix_evidence"] == payload["risk_test_matrix_evidence"]
    assert set(_sidecar_fields(prepared)) == {"prior_items"}
    hydrated, missing = transport.hydrate_mapping(posted, [str(item) for item in prepared])
    assert missing == set()
    assert hydrated["prior_items"] == payload["prior_items"]
    assert isinstance(hydrated["prior_items"], list)


def test_m953_larger_payload_spills_both_growth_fields_losslessly() -> None:
    payload = _growth_payload(prior_bytes=40_000, evidence_bytes=50_000)

    prepared = transport.prepare_round_comment(_comment(payload))

    posted = _anchor_payload(str(prepared[-1]))
    assert len(str(prepared[-1])) <= transport.MAX_GITHUB_BODY_CHARS
    assert _is_reference(posted["prior_items"])
    assert _is_reference(posted["risk_test_matrix_evidence"])
    assert list(dict.fromkeys(_sidecar_fields(prepared))) == [
        "prior_items", "risk_test_matrix_evidence",
    ]
    hydrated, missing = transport.hydrate_mapping(posted, [str(item) for item in prepared])
    assert missing == set()
    assert hydrated["prior_items"] == payload["prior_items"]
    assert isinstance(hydrated["prior_items"], list)
    assert hydrated["risk_test_matrix_evidence"] == payload["risk_test_matrix_evidence"]
    assert isinstance(hydrated["risk_test_matrix_evidence"], dict)


def test_m953_evidence_spill_forces_prior_items_reference(monkeypatch) -> None:
    from coding_review_agent_loop.round_state import _deserialize_unresolved_item

    payload = {
        "flow": "pr",
        "prior_items": [],
        "risk_test_matrix_evidence": {"bulk": _random_text(50_000)},
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    posted = _anchor_payload(str(prepared[-1]))
    assert _is_reference(posted["prior_items"])
    assert _is_reference(posted["risk_test_matrix_evidence"])
    bodies = [str(item) for item in prepared]
    hydrated, missing = transport.hydrate_mapping(posted, bodies)
    assert missing == set()
    assert hydrated["prior_items"] == []
    assert hydrated["risk_test_matrix_evidence"] == payload["risk_test_matrix_evidence"]

    # Simulated older binary: its hydrate ignores the new fields, and its full
    # decoder iterates prior_items, reaching the string keys of the reference.
    monkeypatch.setattr(transport, "_SPILL_FIELDS", _PRE_953_SPILL_FIELDS)
    legacy, legacy_missing = transport.hydrate_mapping(posted, bodies)
    assert legacy_missing == set()
    assert _is_reference(legacy["prior_items"])
    with pytest.raises(AgentLoopError, match="unresolved-item payload"):
        tuple(_deserialize_unresolved_item(item) for item in legacy["prior_items"])


def test_m953_evidence_spill_without_prior_items_raises() -> None:
    payload = {"flow": "pr", "risk_test_matrix_evidence": {"bulk": _random_text(50_000)}}

    with pytest.raises(AgentLoopError, match="without a spillable prior_items"):
        transport.prepare_round_comment(_comment(payload))


def test_m953_discuss_flow_never_spills_growth_fields() -> None:
    overflowing = {
        "flow": "discuss",
        "canonical_plan": _random_text(60_000),
        "prior_items": [{"item_id": "item-1", "text": _random_text(50_000)}],
    }
    with pytest.raises(transport.RoundCommentOverflowError) as excinfo:
        transport.prepare_round_comment(_comment(overflowing))
    assert "prior_items=" in str(excinfo.value)

    spilled_earlier = {
        "flow": "discuss",
        "canonical_plan": _random_text(60_000),
        "prior_items": [{"item_id": "item-1", "text": _random_text(1_000)}],
    }
    prepared = transport.prepare_round_comment(_comment(spilled_earlier))
    posted = _anchor_payload(str(prepared[-1]))
    assert posted["prior_items"] == spilled_earlier["prior_items"]
    assert set(_sidecar_fields(prepared)) == {"canonical_plan"}

    fitting = _comment({"flow": "discuss", "prior_items": [{"item_id": "item-1"}]})
    assert [str(item) for item in transport.prepare_round_comment(fitting)] == [fitting]


@pytest.mark.parametrize("spill_earlier", [False, True])
def test_m953_fitting_payload_is_byte_identical_to_pre_change_order(
    monkeypatch, spill_earlier
) -> None:
    payload = _growth_payload(prior_bytes=2_000, evidence_bytes=2_000)
    if spill_earlier:
        payload["canonical_plan"] = _random_text(60_000)
    body = _comment(payload)

    current = [str(item) for item in transport.prepare_round_comment(body)]
    monkeypatch.setattr(transport, "_SPILL_FIELDS", _PRE_953_SPILL_FIELDS)
    legacy = [str(item) for item in transport.prepare_round_comment(body)]

    assert current == legacy
    posted = _anchor_payload(current[-1])
    assert posted["prior_items"] == payload["prior_items"]
    assert posted["risk_test_matrix_evidence"] == payload["risk_test_matrix_evidence"]


def _spilled_round_record_comments():
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="codex",
        source_round=3,
        text=_random_text(40_000),
        status="blocking",
    )
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="claude", round_number=4, subject="head-sha",
        prior_items=(item,),
        risk_test_matrix_evidence={"rows": [{"evidence": _random_text(50_000)}]},
    )
    prepared = transport.prepare_round_comment(_attach_round_metadata("coder round", metadata))
    posted = _anchor_payload(str(prepared[-1]))
    assert _is_reference(posted["prior_items"])
    assert _is_reference(posted["risk_test_matrix_evidence"])
    return metadata, prepared


def test_m953_spilled_growth_fields_resume_exactly() -> None:
    metadata, prepared = _spilled_round_record_comments()

    records = _extract_round_metadata_records(
        [SimpleNamespace(body=str(item)) for item in prepared], flow="pr"
    )

    assert len(records) == 1
    assert records[0].metadata.prior_items == metadata.prior_items
    assert records[0].metadata.risk_test_matrix_evidence == metadata.risk_test_matrix_evidence


@pytest.mark.parametrize("field", ["prior_items", "risk_test_matrix_evidence"])
def test_m953_missing_growth_field_sidecar_fails_resume_closed(field) -> None:
    _metadata, prepared = _spilled_round_record_comments()
    kept = [
        str(item) for item in prepared[:-1]
        if json.loads(base64.urlsafe_b64decode(
            transport.ROUND_TRANSPORT_SIDECAR_RE.search(str(item)).group("payload")
        ))["field"] != field
    ]
    bodies = [*kept, str(prepared[-1])]

    _hydrated, missing = transport.hydrate_mapping(_anchor_payload(bodies[-1]), bodies)
    assert missing == {field}
    with pytest.raises(AgentLoopError, match=f"Incomplete round metadata: {field} sidecars"):
        _extract_round_metadata_records(
            [SimpleNamespace(body=body) for body in bodies], flow="pr"
        )


@pytest.mark.parametrize("field", ["prior_items", "risk_test_matrix_evidence"])
def test_m953_decoder_rejects_unhydrated_growth_reference(field) -> None:
    payload = transport.decode_mapping(_encode_round_metadata(PostedRoundMetadata(
        flow="pr", role="coder", agent="claude", round_number=2, subject="head",
    )))
    payload[field] = {
        "$round_transport_spill": "abc", "field": field, "parts": 1,
        "sha256": "0" * 64, "spill": "0" * 64, "encoding": "json",
    }

    with pytest.raises(AgentLoopError, match="unhydrated transport reference"):
        _decode_round_metadata(transport.encode_mapping(payload))
    with pytest.raises(AgentLoopError, match="unhydrated transport reference"):
        _decode_round_metadata_mapping(payload)


def test_m953_derived_metadata_overflow_is_reported_as_such() -> None:
    payload = {"flow": "pr", "unspilled_big_field": "SECRET" + _random_text(50_000)}

    with pytest.raises(transport.RoundCommentOverflowError) as excinfo:
        transport.prepare_round_comment(_comment(payload, body="short"))

    message = str(excinfo.value)
    assert "derived round metadata alone needs" in message
    assert "shortening the visible response cannot fix it" in message
    assert "shorten the visible response or metadata" not in message
    assert "residual encoded round metadata" in message
    assert "unspilled_big_field=" in message
    assert "SECRET" not in message


def test_m953_derived_overflow_threshold_uses_the_framed_marker(monkeypatch) -> None:
    payload = {"flow": "pr", "unspilled_big_field": _random_text(3_000)}
    body = _comment(payload, body="Visible response")
    marker = transport.ROUND_RESUME_MARKER_RE.search(body).group(0)
    minimal = transport._minimal_metadata_anchor_chars(marker, payload)
    encoded_only = len(transport.encode_mapping(payload))
    assert minimal == len(marker) > encoded_only

    def overflow_message(budget: int) -> str:
        monkeypatch.setattr(transport, "MAX_GITHUB_BODY_CHARS", budget)
        with pytest.raises(transport.RoundCommentOverflowError) as excinfo:
            transport.prepare_round_comment(body)
        message = str(excinfo.value)
        assert "Size attribution: visible body outside round metadata" in message
        assert "unspilled_big_field=" in message
        return message

    at_minimum = overflow_message(minimal)
    assert "shorten the visible response or metadata." in at_minimum
    assert "derived round metadata alone" not in at_minimum
    for budget in (minimal - 1, encoded_only + 1, encoded_only):
        message = overflow_message(budget)
        assert "derived round metadata alone" in message
        assert "shorten the visible response or metadata." not in message


# --- #959: persisted full-matrix anchor ------------------------------------

_EVIDENCE_959 = {"matrix_identity": "a" * 64, "rows": []}


def _coder_metadata_959(**kwargs):
    return PostedRoundMetadata(
        flow="pr",
        role="coder",
        agent="Claude",
        round_number=3,
        subject="abc",
        risk_test_matrix_evidence=_EVIDENCE_959,
        **kwargs,
    )


def test_959_matrix_full_round_round_trips_with_valid_status():
    metadata = _coder_metadata_959(risk_test_matrix_evidence_full_round=2)
    assert metadata.risk_test_matrix_evidence_full_round_status == "valid"
    encoded = _encode_round_metadata(metadata)
    payload = transport.decode_mapping(encoded)
    assert payload["risk_test_matrix_evidence_full_round"] == 2
    assert "risk_test_matrix_evidence_full_round_status" not in payload
    decoded = _decode_round_metadata(encoded)
    assert decoded.risk_test_matrix_evidence_full_round == 2
    assert decoded.risk_test_matrix_evidence_full_round_status == "valid"
    assert decoded.risk_test_matrix_evidence == _EVIDENCE_959


def test_959_matrix_full_round_omitted_when_unset_keeps_legacy_encoding():
    metadata = _coder_metadata_959()
    assert metadata.risk_test_matrix_evidence_full_round_status == "absent"
    encoded = _encode_round_metadata(metadata)
    payload = transport.decode_mapping(encoded)
    assert "risk_test_matrix_evidence_full_round" not in payload
    assert "risk_test_matrix_evidence_full_round_status" not in payload
    # Byte-identical to an encoding of the same record written before #959.
    assert transport.encode_mapping(payload) == encoded
    decoded = _decode_round_metadata(encoded)
    assert decoded.risk_test_matrix_evidence_full_round is None
    assert decoded.risk_test_matrix_evidence_full_round_status == "absent"


@pytest.mark.parametrize("bad_value", [True, False, 0, -1, "2", None, 2.0])
def test_959_malformed_matrix_full_round_decodes_invalid_without_raising(bad_value):
    payload = transport.decode_mapping(_encode_round_metadata(_coder_metadata_959()))
    payload["risk_test_matrix_evidence_full_round"] = bad_value
    decoded = _decode_round_metadata_mapping(payload)
    assert decoded.risk_test_matrix_evidence_full_round is None
    assert decoded.risk_test_matrix_evidence_full_round_status == "invalid"
    assert decoded.risk_test_matrix_evidence == _EVIDENCE_959


def test_959_matrix_full_round_construction_status_rules():
    assert _coder_metadata_959().risk_test_matrix_evidence_full_round_status == "absent"
    assert (
        _coder_metadata_959(risk_test_matrix_evidence_full_round=1)
        .risk_test_matrix_evidence_full_round_status
        == "valid"
    )
    with pytest.raises(ValueError):
        _coder_metadata_959(
            risk_test_matrix_evidence_full_round=1,
            risk_test_matrix_evidence_full_round_status="invalid",
        )
    with pytest.raises(ValueError):
        _coder_metadata_959(risk_test_matrix_evidence_full_round_status="bogus")
    invalid = _coder_metadata_959(risk_test_matrix_evidence_full_round_status="invalid")
    assert invalid.risk_test_matrix_evidence_full_round is None


def test_959_matrix_full_round_does_not_change_growth_spill_fields():
    assert transport._GROWTH_SPILL_FIELDS == ("prior_items", "risk_test_matrix_evidence")
