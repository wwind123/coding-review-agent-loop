"""Deterministic response normalization outside the retired repair pass (#2066)."""

import json

import pytest

import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.response_normalization as response_normalization
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_images_validation as review_images_validation
import coding_review_agent_loop.validated_agent as validated_agent
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.pr_loop import _validate_review_response
from coding_review_agent_loop.protocol import validate_structured_issue_implementation
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    patch_object_in,
    structured_issue_implementation,
    structured_pr_review,
    structured_v1_plan_state,
)
from issue_loop_test_helpers import _DEG_CORROBORATED, _DEG_UNCORROBORATED, _deg_with


NORMALIZERS = (
    "strip_unknown_prior_item_dispositions",
    "unknown_dispositions_are_resolved_history",
    "attempt_envelope_normalization",
    "attempt_risk_test_matrix_string_list_normalization",
    "attempt_semantic_patch_disposition_normalization",
)


@pytest.mark.parametrize("name", NORMALIZERS)
def test_normalizers_live_in_response_normalization(name):
    owner = getattr(response_normalization, name)
    assert owner.__module__ == "coding_review_agent_loop.response_normalization"
    # The validated turn binds the relocated object.
    assert getattr(validated_agent, name, owner) is owner


def test_image_turn_and_architecture_boundary_import_the_relocated_owners():
    assert (
        review_images_validation.attempt_envelope_normalization
        is response_normalization.attempt_envelope_normalization
    )
    assert (
        architecture_contract.canonicalize_architecture_near_miss_text
        is response_normalization.canonicalize_architecture_near_miss_text
    )


@pytest.mark.parametrize(
    "module", ["repair", "repair_preservation", "shape_check_audit", "agents.format_repair"]
)
def test_retired_repair_modules_are_gone(module):
    """`architecture-canonicalization-retained`: no import of a deleted module."""
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"coding_review_agent_loop.{module}")


def test_corroborated_near_miss_canonicalizes_and_strictly_reparses():
    """`architecture-canonicalization-retained`."""
    text = _deg_with(structured_issue_implementation(), _DEG_CORROBORATED)
    canonical = response_normalization.canonicalize_architecture_near_miss_text(text)
    assert '"modified"' not in canonical
    payload = json.loads(canonical[: canonical.index("}\n") + 1])
    assert payload["architecture_impact"]["status"] == "changed"
    # Footer and signature bytes after the JSON object are kept.
    assert canonical.endswith(text[text.index("}\n") + 1:].lstrip("\n"))
    parsed = validate_structured_issue_implementation(
        canonical, required_architecture_impact_contract=1
    )
    assert parsed.architecture_impact.status == "changed"


def test_uncorroborated_near_miss_is_removed_with_its_record():
    text = _deg_with(structured_issue_implementation(), _DEG_UNCORROBORATED)
    result = response_normalization.normalize_architecture_impact_near_miss(
        text, required_contract=False, expected_kind="issue_implementation"
    )
    assert "architecture_impact" not in json.loads(result.raw[: result.raw.index("}\n") + 1])
    assert result.record is not None and result.record.outcome == "degraded-to-undetermined"
    assert result.forbid_architecture_impact is True


def test_text_without_a_near_miss_is_unchanged():
    text = structured_issue_implementation()
    assert response_normalization.canonicalize_architecture_near_miss_text(text) == text


def _review_validator(text):
    return _validate_review_response(
        text, reviewer="OpenAI Codex", unresolved_items=(), architecture_status_mode="legacy"
    )


def test_envelope_defect_is_normalized_without_a_reask(tmp_path):
    """`mechanical-normalization-no-reask`: no model call for an envelope defect."""
    valid = structured_pr_review(summary="Looks good.")
    fenced = "```json\n" + valid.replace("\n<!--", "\n```\n<!--", 1)
    with patch_object_in(
        (validated_agent, response_validation), "run_agent_result",
        side_effect=[AgentResult(text=fenced, returncode=0, session_id="s")],
    ) as invoke:
        response = validated_agent._run_validated_agent(
            FakeRunner(), agent="codex", config=make_config(tmp_path, agent_max_retries=0),
            prompt="Review the PR.", marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=_review_validator, role="reviewer", repair_expected_kind="pr_review",
        )
    assert invoke.call_count == 1
    assert response.marker_value.summary == "Looks good."


def test_unknown_prior_disposition_is_stripped_without_a_model_call(tmp_path):
    payload, end = json.JSONDecoder().raw_decode(structured_pr_review(summary="Looks good."))
    source = structured_pr_review(summary="Looks good.")
    payload["prior_item_dispositions"] = [
        {"item_id": "item-404", "disposition": "resolved", "note": "Stale."}
    ]
    text = json.dumps(payload) + source[end:]
    with patch_object_in(
        (validated_agent, response_validation), "run_agent_result",
        side_effect=[AgentResult(text=text, returncode=0, session_id="s")],
    ) as invoke:
        response = validated_agent._run_validated_agent(
            FakeRunner(), agent="codex", config=make_config(tmp_path, agent_max_retries=0),
            prompt="Review the PR.", marker_description="<!-- AGENT_STATE: approved|blocking -->",
            validate=_review_validator, role="reviewer", repair_expected_kind="pr_review",
        )
    assert invoke.call_count == 1
    assert "item-404" not in response.text


def test_risk_matrix_string_field_is_wrapped_without_a_model_call(tmp_path):
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["risk_test_matrix"] = {
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-a", "label": "L", "entry_path_or_mode": "m", "initial_state": "i",
            "event": "e", "expected_outcome": "o", "forbidden_side_effects": "Must not refuse",
            "proposed_test_level": "unit", "proposed_test_location": "tests/x.py",
            "applicability": "required", "related_scope_item_ids": ["scope-1"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": [],
    }
    source = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    with patch_object_in(
        (validated_agent, response_validation), "run_agent_result",
        side_effect=[AgentResult(text=source, returncode=0, session_id="s")],
    ) as invoke:
        response = validated_agent._run_validated_agent(
            FakeRunner(), agent="claude", config=make_config(tmp_path, agent_max_retries=0),
            prompt="Plan.", marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
            validate=validate_structured_plan_state, role="planner", repair_expected_kind="plan_state",
        )
    assert invoke.call_count == 1
    stored = json.loads(response.text.split("\n<!--", 1)[0])
    assert stored["risk_test_matrix"]["rows"][0]["forbidden_side_effects"] == ["Must not refuse"]


# Near-miss normalization cases moved from the deleted
# tests/test_repair_preservation.py (#2092): the normalizer outlives the
# repair modules, so its behavior stays pinned here.


def _near_miss_payload(text: str) -> dict:
    return json.loads(text[: text.index("}\n") + 1])


def _near_miss_text(impact) -> str:
    rendered = structured_issue_implementation()
    split = rendered.index("}\n") + 1
    payload = json.loads(rendered[:split])
    if impact is None:
        payload.pop("architecture_impact", None)
    else:
        payload["architecture_impact"] = impact
    return json.dumps(payload) + rendered[split:]


def test_near_miss_normalization_rewrites_corroborated_status_to_changed():
    result = response_normalization.normalize_architecture_impact_near_miss(
        _near_miss_text(_DEG_CORROBORATED), required_contract=True, expected_kind="issue_implementation"
    )
    assert _near_miss_payload(result.raw)["architecture_impact"]["status"] == "changed"
    assert result.record.outcome == "normalized-to-changed"
    assert result.record.element_path == "issue_implementation.architecture_impact.status"
    assert result.forbid_architecture_impact is False
    assert "undetermined" not in result.raw


def test_near_miss_normalization_removes_uncorroborated_assessment():
    required = response_normalization.normalize_architecture_impact_near_miss(
        _near_miss_text(_DEG_UNCORROBORATED), required_contract=True, expected_kind="issue_implementation"
    )
    assert "architecture_impact" not in _near_miss_payload(required.raw)
    assert required.record.outcome == "degraded-to-undetermined"
    assert required.forbid_architecture_impact is True
    assert "undetermined" not in required.raw
    # A normalization removal is pinned for every kind, including the
    # optional-assessment reviews.
    optional = response_normalization.normalize_architecture_impact_near_miss(
        _near_miss_text(_DEG_UNCORROBORATED), required_contract=False
    )
    assert optional.forbid_architecture_impact is True
    # A genuine omission is pinned only under a required contract.
    omitted = response_normalization.normalize_architecture_impact_near_miss(
        _near_miss_text(None), required_contract=False
    )
    assert omitted.forbid_architecture_impact is False


@pytest.mark.parametrize(
    "raw",
    [
        structured_issue_implementation(),
        '{"kind":"issue_implementation","architecture_impact":{"status":"modified"',
        "plain prose with no JSON",
    ],
    ids=["valid", "truncated", "prose"],
)
def test_near_miss_normalization_leaves_other_payloads_byte_identical(raw):
    result = response_normalization.normalize_architecture_impact_near_miss(raw, required_contract=True)
    assert result.raw == raw
    assert result.record is None


def test_near_miss_normalization_pins_absence_of_a_genuine_omission():
    result = response_normalization.normalize_architecture_impact_near_miss(
        _near_miss_text(None), required_contract=True, expected_kind="issue_implementation"
    )
    assert result.record is None
    assert result.forbid_architecture_impact is True
