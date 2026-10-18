"""Architecture-contract tests: architecture_impact refusal and degradation seam (#1543)."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata
from coding_review_agent_loop.protocol import (
    validate_structured_plan_state,
    validate_structured_issue_implementation,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_object_in
from issue_loop_test_helpers import _DEG_CORROBORATED, _deg_task_text, _deg_with
from agent_loop_helpers import (
    make_config,
    structured_plan_revision,
    structured_plan_state,
    structured_issue_implementation,
)
from coding_review_agent_loop.errors import AgentInvocationError as _DegInvocationError  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    validate_structured_coder_followup as _deg_validate_coder_followup,
    validate_structured_plan_revision as _deg_validate_plan_revision,
    validate_structured_task_result as _deg_validate_task_result,
)
from agent_loop_helpers import structured_coder_followup as _deg_coder_followup  # noqa: E402
from agent_loop_helpers import (
    FakeRunner as _r2_FakeRunner,
    structured_pr_review as _r2_pr_review,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_unsatisfied_conflict_payload_is_not_hidden_by_the_conflict_wrapper():
    blocked = [{"requirement_id": "Requirement 1", "disposition": "blocked", "evidence": "Blocked."}]
    text = _deg_with(
        structured_issue_implementation(
            human_requirement_ids=["Requirement 1"], human_requirement_dispositions=blocked
        ),
        None,
    )
    # The typed conflict is raised only for a satisfied contract; an
    # unsatisfied one returns the parsed payload for the seam to refuse.
    result = validate_structured_issue_implementation(text, required_architecture_impact_contract=1)
    assert result.pr_number == 77
    assert architecture_contract.architecture_impact_contract_unsatisfied(result)
    # A conflict wrapper around an unsatisfied payload is still unsatisfied.
    wrapped = architecture_contract._TerminalIssueImplementationConflict(result)
    assert architecture_contract.architecture_impact_contract_unsatisfied(wrapped)


def test_metadata_writer_requires_the_whole_result():
    config_like = SimpleNamespace(architecture_context=None)
    with pytest.raises(TypeError):
        architecture_contract._architecture_metadata_fields(config_like, impact=None)


def _deg_round_trip(result):
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=1, subject="head",
        **architecture_contract._architecture_metadata_fields(
            make_config_for_metadata(), result=result
        ),
    )
    body = _attach_round_metadata("Coder round summary.\n-- Anthropic Claude", metadata)
    return metadata, body, _decode_round_metadata(
        re.search(r"AGENT_LOOP_META: ([A-Za-z0-9+/=_-]+)", body).group(1)
    )


def make_config_for_metadata():
    return SimpleNamespace(architecture_context=None)


@pytest.mark.parametrize(
    "name,parse",
    [
        ("issue_implementation", lambda: validate_structured_issue_implementation(
            _deg_with(structured_issue_implementation(), _DEG_CORROBORATED),
            required_architecture_impact_contract=1, architecture_status_mode="degradable")),
        ("task_result", lambda: _deg_validate_task_result(
            _deg_task_text(_DEG_CORROBORATED), required_architecture_impact_contract=1,
            architecture_status_mode="degradable")),
        ("plan_state", lambda: validate_structured_plan_state(
            _deg_with(structured_plan_state(), _DEG_CORROBORATED),
            required_architecture_impact_contract=1, architecture_status_mode="degradable")),
        ("plan_revision", lambda: _deg_validate_plan_revision(
            _deg_with(structured_plan_revision(), _DEG_CORROBORATED),
            required_architecture_impact_contract=1, architecture_status_mode="degradable")),
        ("coder_followup", lambda: _deg_validate_coder_followup(
            _deg_with(_deg_coder_followup(), _DEG_CORROBORATED),
            required_architecture_impact_contract=1, architecture_status_mode="degradable")),
    ],
)
def test_accepted_non_review_carrier_persists_and_renders_its_record(name, parse):
    parsed = parse()
    metadata, body, decoded = _deg_round_trip(parsed)
    assert metadata.architecture_impact["status"] == "changed"
    (record,) = decoded.architecture_impact_degradations
    assert record == parsed.architecture_impact_degradations[0]
    assert record.element_path == f"{name}.architecture_impact.status"
    assert record.outcome == "normalized-to-changed"
    assert "### Parse degradations" in body
    assert "normalized-to-changed" in body


def test_accepted_conflict_wrapper_persists_its_record():
    parsed = validate_structured_issue_implementation(
        _deg_with(structured_issue_implementation(), _DEG_CORROBORATED),
        architecture_status_mode="degradable",
    )
    wrapped = architecture_contract._TerminalIssueImplementationConflict(parsed)
    fields = architecture_contract._architecture_metadata_fields(make_config_for_metadata(), result=wrapped)
    assert [r.outcome for r in fields["architecture_impact_degradations"]] == ["normalized-to-changed"]


def test_metadata_helper_rejects_an_unlisted_assessment_bearing_object():
    # An object carrying an assessment but outside the enumerated carriers is
    # never persisted as fully assessed with its records silently dropped.
    impostor = SimpleNamespace(
        architecture_impact=validate_structured_issue_implementation(
            _deg_with(structured_issue_implementation(), _DEG_CORROBORATED),
            architecture_status_mode="degradable",
        ).architecture_impact,
        architecture_impact_degradations=(),
    )
    with pytest.raises(AgentLoopError, match="not an enumerated"):
        architecture_contract._architecture_result_fields(impostor)
    with pytest.raises(AgentLoopError, match="not an enumerated"):
        architecture_contract._architecture_metadata_fields(
            make_config_for_metadata(), result=impostor
        )


def test_record_free_candidates_pass_the_boundary_byte_identical(tmp_path):
    parsed = validate_structured_issue_implementation(structured_issue_implementation())
    accepted = architecture_contract._accept_candidate(
        structured_issue_implementation(), parsed,
        strict_revalidate=None, runner=None, acquisition=None,
    )
    assert accepted.text == structured_issue_implementation()
    assert accepted.marker_value is parsed


def test_projection_mismatch_is_refused_rather_than_accepted():
    text = _deg_with(structured_issue_implementation(), _DEG_CORROBORATED)
    degraded = validate_structured_issue_implementation(text, architecture_status_mode="degradable")
    other = validate_structured_issue_implementation(
        _deg_with(structured_issue_implementation(pr_number=99), dict(_DEG_CORROBORATED, status="changed"))
    )
    with pytest.raises(architecture_contract._AcceptedTextCanonicalizationError):
        architecture_contract._accept_candidate(
            text, degraded, strict_revalidate=lambda _text: other, runner=None, acquisition=None,
        )


def test_satisfied_blocking_task_keeps_its_payload_and_records():
    text = _deg_task_text(_DEG_CORROBORATED, outcome="blocking")
    result = response_validation._require_task_implementation_result(
        text, required_architecture_impact_contract=1, architecture_status_mode="degradable"
    )
    assert isinstance(result, architecture_contract._TerminalNoPrImplementation)
    assert result.state == "blocking"
    assert result.parsed.architecture_impact.status == "changed"
    fields = architecture_contract._architecture_metadata_fields(
        make_config_for_metadata(), result=result
    )
    assert fields["architecture_impact"]["status"] == "changed"
    assert [r.outcome for r in fields["architecture_impact_degradations"]] == ["normalized-to-changed"]
    # The accepted text is canonical and the wrapper survives the boundary.
    accepted = architecture_contract._accept_candidate(
        text, result,
        strict_revalidate=lambda candidate: response_validation._require_task_implementation_result(
            candidate, required_architecture_impact_contract=1, architecture_status_mode="strict"
        ),
        runner=None, acquisition=None,
    )
    assert isinstance(accepted.marker_value, architecture_contract._TerminalNoPrImplementation)
    assert '"modified"' not in accepted.text
    assert accepted.marker_value.parsed.architecture_impact_degradations == (
        result.parsed.architecture_impact_degradations
    )


def test_refused_decomposition_posts_one_comment_and_never_masks_the_error(tmp_path):
    runner = _r2_FakeRunner()
    error = _DegInvocationError(
        "exhausted",
        preserved_unsatisfied_response=validated_agent_module.PreservedUnsatisfiedResponse(
            text="{}", diagnostic="plan_decomposition must include architecture_impact",
            architecture_impact_degradations=(),
        ),
    )
    architecture_contract._surface_refused_decomposition(
        runner, config=make_config(tmp_path), issue_number=56, error=error
    )
    bodies = [c["body"] for c in runner.issue_comments]
    assert len(bodies) == 1
    assert "Decomposition parse degradations" in bodies[0]
    assert "omitted" in bodies[0]
    assert "AGENT_LOOP_META" not in bodies[0]

    def failing_post(*args, **kwargs):
        raise RuntimeError("GitHub unavailable")

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", failing_post):
        architecture_contract._surface_refused_decomposition(
            runner, config=make_config(tmp_path), issue_number=56, error=error
        )


def test_resumed_review_architecture_comes_from_round_metadata():
    impact = validate_structured_issue_implementation(
        _deg_with(structured_issue_implementation(), dict(_DEG_CORROBORATED, status="changed"))
    ).architecture_impact
    record = architecture_contract.ParseDegradation.build(
        element_path="plan_review.architecture_impact.status", rule="r",
        observed="modified", outcome="normalized-to-changed",
    )
    metadata = SimpleNamespace(
        architecture_contract_version=1,
        architecture_impact=architecture_contract.sanitize_architecture_impact(impact),
        architecture_impact_degradations=(record,),
    )
    rebuilt, records = architecture_contract._resumed_review_architecture(metadata, None)
    assert rebuilt == impact
    assert records == (record,)
    # A degraded review stored no assessment: it stays absent, with its record.
    degraded = SimpleNamespace(
        architecture_contract_version=1, architecture_impact=None,
        architecture_impact_degradations=(record,),
    )
    assert architecture_contract._resumed_review_architecture(degraded, impact) == (None, (record,))
    # Wrong keys or a non-declared status never fabricate an assessment.
    bad = dict(architecture_contract.sanitize_architecture_impact(impact), status="undetermined")
    assert architecture_contract._architecture_impact_from_metadata(bad) is None
    assert architecture_contract._architecture_impact_from_metadata({"status": "changed"}) is None
    # A record older than architecture metadata falls back to its text.
    legacy = SimpleNamespace(architecture_contract_version=None)
    assert architecture_contract._resumed_review_architecture(legacy, impact) == (impact, ())


def test_acknowledgement_repair_pins_the_accepted_assessment(tmp_path):
    record = architecture_contract.ParseDegradation.build(
        element_path="pr_review.architecture_impact.status", rule="r",
        observed="modified", outcome="degraded-to-undetermined",
    )
    accepted = SimpleNamespace(architecture_impact=None, architecture_impact_degradations=(record,))
    fabricated = pr_loop._validate_review_response(
        _deg_with(_r2_pr_review(), {"status": "unchanged", "rationale": "No change."}),
        reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="strict",
    )
    config = make_config(tmp_path)
    assert architecture_contract._pin_acknowledgement_repair(
        accepted, fabricated, config=config, reviewer_name="Codex"
    ) is None
    ack_only = pr_loop._validate_review_response(
        _deg_with(_r2_pr_review(), None),
        reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="strict",
    )
    pinned = architecture_contract._pin_acknowledgement_repair(
        accepted, ack_only, config=config, reviewer_name="Codex"
    )
    assert pinned.architecture_impact is None
    assert pinned.architecture_impact_degradations == (record,)
    # Absence is pinned for the repair when the accepted source has none.
    assert architecture_contract._acknowledgement_repair_forbids_assessment(_deg_with(_r2_pr_review(), None))
    # New repair output is validated strictly: a legacy synonym is rejected.
    with pytest.raises(AgentLoopError, match="must be `changed` or `unchanged`"):
        pr_loop._validate_review_response(
            _deg_with(_r2_pr_review(), {"status": "none", "rationale": "No change."}),
            reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="strict",
        )


def test_acknowledgement_equality_check_refuses_a_changed_assessment_directly(tmp_path):
    # Defense in depth behind preservation: any durable difference fails.
    accepted = validate_structured_plan_state(
        _deg_with(structured_plan_state(), dict(_DEG_CORROBORATED, status="changed"))
    )
    altered = replace(
        accepted,
        architecture_impact=replace(accepted.architecture_impact, rationale="Different."),
    )
    assert architecture_contract._pin_acknowledgement_repair(
        accepted, altered, config=make_config(tmp_path), reviewer_name="Codex"
    ) is None
