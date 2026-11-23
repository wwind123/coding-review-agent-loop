"""Lossless shape normalization in the shared parser helpers (#2109).

Each mutation is applied to a canonical valid payload and must parse equal to
the unmutated parse while logging a normalization line.  Ambiguous inputs
(multi-element lists for a string, nested lists, non-numeric integers, enum
synonyms, list-wrapped identifiers) keep rejecting.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest

import coding_review_agent_loop.validated_agent as validated_agent_module
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import (
    ARCHITECTURE_IMPACT_UNDETERMINED,
    _parse_plan_patch_field_value,
    classify_architecture_status_near_miss,
    parse_architecture_impact,
    parse_architecture_impact_degradable,
    parse_structured_plan_review,
    parse_structured_pr_review,
    validate_structured_issue_implementation,
    validate_structured_plan_state,
    validate_structured_task_result,
)
from coding_review_agent_loop.protocol_common import (
    _expect_int_like,
    _expect_non_empty_string,
    _expect_strict_string,
    _expect_string_list,
    fold_closed_enum,
    normalize_staging_feasibility,
)
from coding_review_agent_loop.protocol_risk import (
    _parse_risk_test_matrix,
    _parse_risk_test_matrix_changes,
)
from coding_review_agent_loop.response_normalization import (
    canonicalize_architecture_near_miss_text,
)
from agent_loop_helpers import FakeRunner, make_config

LOGGER = "coding_review_agent_loop.protocol"
PLAN_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
STATE_FOOTER = "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"

ARCH = {
    "status": "unchanged",
    "rationale": "No architectural contract changed.",
    "affected_components": [],
    "dependencies": [],
    "execution_data_flows": [],
    "persistence": [],
    "public_contracts": [],
    "security_boundaries": [],
    "canonical_document_action": "no-change",
    "canonical_document_path": None,
    "canonical_document_rationale": "",
}

CORROBORATED_CHANGED = {
    "status": "changed",
    "rationale": "The parser contract changed.",
    "affected_components": ["Parser helpers normalize shapes."],
    "dependencies": ["None new."],
    "execution_data_flows": ["Parsed values are normalized before validation."],
    "persistence": ["No persistence change."],
    "public_contracts": ["Accepted response shapes widen losslessly."],
    "security_boundaries": ["Identifiers stay strict."],
    "canonical_document_action": "update",
    "canonical_document_path": "docs/architecture/testing-and-doc-maintenance.md",
    "canonical_document_rationale": "The response-contract owner documents it.",
}


def _stage(index: int, **extra) -> dict:
    stage = {
        "stage_id": f"stage-{index}", "position": index, "title": f"Stage {index}",
        "summary": f"Stage {index} work.", "deliverables": [f"Deliverable {index}."],
        "non_goals": [], "acceptance_criteria": ["Done."],
        "depends_on_stage_ids": [] if index == 1 else [f"stage-{index - 1}"],
        "dependency_notes": "The implementation must be merged first." if index > 1 else "None.",
        "automation": "agent-pr", "rollout_risk": "low",
        "compatibility_constraints": [], "covered_scope_item_ids": [f"scope-{index}"],
        "execution_disposition": {
            "disposition": "requires-child-planning",
            "rationale": "Child planning follows.",
            "unresolved_design_decisions": [],
        },
    }
    stage.update(extra)
    return stage


def _staged_recommendation() -> dict:
    return {
        "strategy": "staged",
        "rationale": "Two independent stages.",
        "staging_feasibility": "safe",
        "scope_items": [
            {"scope_item_id": "scope-1", "requirement": "First.", "acceptance_criteria": ["Done."]},
            {"scope_item_id": "scope-2", "requirement": "Second.", "acceptance_criteria": ["Done."]},
        ],
        "coupling_constraints": [],
        "child_stages": [_stage(1), _stage(2)],
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


def _one_shot_recommendation() -> dict:
    return {
        "strategy": "one-shot",
        "rationale": "One coherent delivery.",
        "staging_feasibility": "inseparable",
        "scope_items": [
            {"scope_item_id": "scope-1", "requirement": "Implement.", "acceptance_criteria": ["Done."]},
        ],
        "coupling_constraints": [],
        "one_shot_delivery": {
            "deliverables": ["The implementation."],
            "acceptance_criteria": ["Done."],
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


def _plan_state(recommendation: dict | None = None, **extra) -> dict:
    payload = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": "blocking",
        "summary": "Implementation plan.",
        "plan_steps": ["Implement the reviewed scope."],
        "execution_strategy_contract_version": 1,
        "execution_recommendation": recommendation or _staged_recommendation(),
        "human_requirement_dispositions": [],
        "architecture_impact": dict(ARCH),
    }
    payload.update(extra)
    return payload


def _plan_text(payload: dict) -> str:
    return json.dumps(payload) + PLAN_FOOTER


def _parse_plan(payload: dict, **kwargs):
    return validate_structured_plan_state(
        _plan_text(payload), require_execution_strategy_contract=1, **kwargs
    )


def _normalization_lines(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


def _assert_logged(caplog, *fragments: str) -> None:
    lines = _normalization_lines(caplog)
    for fragment in fragments:
        assert any(fragment in line for line in lines), (fragment, lines)


# ---------------------------------------------------------------------------
# (a) singleton-list text fields
# ---------------------------------------------------------------------------


def test_helper_unwraps_singleton_list_and_keeps_strict_variant(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _expect_non_empty_string([" text "], context="ctx") == "text"
    _assert_logged(caplog, "ctx: unwrapped single-element list to string")
    with pytest.raises(AgentLoopError, match="ctx must be a string"):
        _expect_strict_string(["text"], context="ctx")


@pytest.mark.parametrize(
    "value",
    [["a", "b"], [], [["a"]], [7], {"text": "a"}, 7, True, None, [""]],
    ids=["multi", "empty", "nested", "number-item", "object", "number", "bool", "null", "blank-item"],
)
def test_ambiguous_string_shapes_still_reject(value):
    with pytest.raises(AgentLoopError, match="ctx must be"):
        _expect_non_empty_string(value, context="ctx")


def _recommendation_text_paths():
    return [
        ("rationale", lambda r: r),
        ("rationale", lambda r: r["child_stages"][0]["execution_disposition"]),
        ("rollout_risk", lambda r: r["child_stages"][0]),
        ("title", lambda r: r["child_stages"][0]),
        ("summary", lambda r: r["child_stages"][0]),
        ("dependency_notes", lambda r: r["child_stages"][1]),
        ("requirement", lambda r: r["scope_items"][0]),
    ]


@pytest.mark.parametrize(
    "field,locate", _recommendation_text_paths(),
    ids=["rationale", "disposition-rationale", "rollout_risk", "title", "summary", "dependency_notes", "requirement"],
)
def test_recommendation_singleton_text_parses_equal(field, locate, caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    target = locate(mutated["execution_recommendation"])
    target[field] = [target[field]]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert parsed == canonical
    assert parsed.execution_recommendation.identity() == canonical.execution_recommendation.identity()
    _assert_logged(caplog, f".{field}: unwrapped single-element list to string")


def test_recommendation_multi_element_text_still_rejects():
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][1]["dependency_notes"] = ["a", "b"]
    with pytest.raises(AgentLoopError, match=r"dependency_notes must be a string"):
        _parse_plan(mutated)


def test_issue_2368_dependency_notes_shape_parses_first_attempt_and_stores_text(caplog):
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][1]["dependency_notes"] = [
        "The implementation must be merged first."
    ]
    text = _plan_text(mutated)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = validate_structured_plan_state(text, require_execution_strategy_contract=1)
    stage = parsed.execution_recommendation.child_stages[1]
    assert stage.dependency_notes == "The implementation must be merged first."
    assert parsed.execution_recommendation.to_payload()["child_stages"][1]["dependency_notes"] == (
        "The implementation must be merged first."
    )
    _assert_logged(caplog, "child_stages[1].dependency_notes: unwrapped single-element list")


def test_issue_2368_shape_is_accepted_by_validated_turn_with_zero_reasks(tmp_path):
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][1]["dependency_notes"] = ["Merged first."]
    text = _plan_text(mutated)
    runner = FakeRunner(claude_outputs=[text])
    response = validated_agent_module._run_validated_agent(
        runner, agent="claude", config=make_config(tmp_path, agent_max_retries=0),
        prompt="Plan.", marker_description="structured plan_state",
        validate=lambda raw: validate_structured_plan_state(raw, require_execution_strategy_contract=1),
    )
    assert response.marker_value.execution_recommendation.child_stages[1].dependency_notes == "Merged first."
    # The raw response text is never rewritten by the normalization.
    assert response.text.strip() == text.strip()
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert not any("re-ask" in cmd[-1] for cmd in claude_calls)


@pytest.mark.parametrize("field", ["summary"])
def test_plan_state_top_level_singleton_text(field, caplog):
    canonical = _parse_plan(_plan_state())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(_plan_state(**{field: [canonical.summary]}))
    assert parsed == canonical
    _assert_logged(caplog, f"plan_state.{field}: unwrapped")


# ---------------------------------------------------------------------------
# (b) null / lone-string list fields
# ---------------------------------------------------------------------------


def test_list_helper_wraps_lone_string_and_nulls_without_minimum(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _expect_string_list("a", context="ctx", item_context="item") == ("a",)
        assert _expect_string_list(None, context="ctx", item_context="item") == ()
        assert _expect_string_list("a", context="ctx", item_context="item", min_length=1) == ("a",)
    _assert_logged(caplog, "ctx: wrapped a lone string", "ctx: normalized null to an empty list")


@pytest.mark.parametrize(
    "value,kwargs",
    [
        (None, {"min_length": 1}),
        (["a", ["b"]], {}),
        ([["a"]], {}),
        ({"a": "b"}, {}),
        (7, {}),
        ("", {}),
        ("a", {"strict_shape": True}),
        (None, {"strict_shape": True}),
    ],
    ids=["null-min1", "nested-item", "singleton-nested", "object", "number", "blank", "strict-str", "strict-null"],
)
def test_list_helper_still_rejects_ambiguous(value, kwargs):
    with pytest.raises(AgentLoopError):
        _expect_string_list(value, context="ctx", item_context="item", **kwargs)


def _recommendation_list_paths():
    return [
        ("caveats", lambda r: r, None),
        ("non_goals", lambda r: r["child_stages"][0], None),
        ("compatibility_constraints", lambda r: r["child_stages"][0], None),
        ("deliverables", lambda r: r["child_stages"][0], "lone"),
        ("acceptance_criteria", lambda r: r["child_stages"][0], "lone"),
        ("acceptance_criteria", lambda r: r["scope_items"][0], "lone"),
        ("unresolved_design_decisions", lambda r: r["child_stages"][0]["execution_disposition"], None),
    ]


@pytest.mark.parametrize(
    "field,locate,mode", _recommendation_list_paths(),
    ids=["caveats", "non_goals", "compatibility", "deliverables", "stage-criteria", "scope-criteria", "unresolved"],
)
def test_recommendation_list_fields_normalize(field, locate, mode, caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    target = locate(mutated["execution_recommendation"])
    if mode == "lone":
        (only,) = target[field]
        target[field] = only
        expected = "wrapped a lone string"
    else:
        assert target[field] == []
        target[field] = None
        expected = "normalized null to an empty list"
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert parsed == canonical
    _assert_logged(caplog, f".{field}: {expected}")


@pytest.mark.parametrize("value,expected", [(None, ()), ("stage-1", ("stage-1",))], ids=["null", "lone"])
def test_depends_on_stage_ids_outer_shape(value, expected, caplog):
    mutated = _plan_state()
    stage = mutated["execution_recommendation"]["child_stages"][1]
    stage["depends_on_stage_ids"] = value
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert parsed.execution_recommendation.child_stages[1].depends_on_stage_ids == expected
    _assert_logged(caplog, "depends_on_stage_ids:")


@pytest.mark.parametrize(
    "value,match",
    [
        ([["stage-1"]], "must be a string"),
        ([""], "must be a non-empty string"),
        ("stage-2", "must reference earlier stages"),
        ("", "must be a non-empty string"),
    ],
    ids=["nested", "empty-item", "later-stage", "blank"],
)
def test_depends_on_stage_ids_still_strict_per_element(value, match):
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][1]["depends_on_stage_ids"] = value
    with pytest.raises(AgentLoopError, match=match):
        _parse_plan(mutated)


@pytest.mark.parametrize("field", ["covered_scope_item_ids"])
@pytest.mark.parametrize("value", [None, "scope-1"], ids=["null", "lone"])
def test_other_id_lists_keep_strict_outer_shape(field, value):
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][0][field] = value
    with pytest.raises(AgentLoopError, match=f"{field} must be a JSON array"):
        _parse_plan(mutated)


def test_plan_steps_lone_string_and_null_rejects(caplog):
    canonical = _parse_plan(_plan_state())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(_plan_state(plan_steps="Implement the reviewed scope."))
    assert parsed == canonical
    _assert_logged(caplog, "plan_steps: wrapped a lone string")
    with pytest.raises(AgentLoopError, match="plan_steps must be a JSON array"):
        _parse_plan(_plan_state(plan_steps=None))


def _issue_impl(**extra) -> dict:
    payload = {
        "schema_version": 1,
        "kind": "issue_implementation",
        "state": "blocking",
        "summary": "Implemented.",
        "pr_number": 77,
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [],
        "architecture_impact": dict(ARCH),
    }
    payload.update(extra)
    return payload


def _parse_impl(payload: dict):
    return validate_structured_issue_implementation(json.dumps(payload) + STATE_FOOTER)


def test_tests_run_lone_string(caplog):
    canonical = _parse_impl(_issue_impl(tests_run=["pytest -q"]))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_impl(_issue_impl(tests_run="pytest -q"))
    assert parsed == canonical
    _assert_logged(caplog, "issue_implementation.tests_run: wrapped a lone string")


# ---------------------------------------------------------------------------
# (c) closed-enum case / separator fold
# ---------------------------------------------------------------------------


def test_fold_closed_enum_matches_only_allowed_tokens():
    assert fold_closed_enum("Opened PR", {"opened_pr", "blocking"}) == ("opened_pr", True)
    assert fold_closed_enum(" same_pr ", {"same-pr"}) == ("same-pr", True)
    assert fold_closed_enum("fixed", {"resolved"}) == ("fixed", False)
    assert fold_closed_enum(7, {"resolved"}) == (7, False)
    # The audited staging aliases survive the rebuild on the shared fold.
    assert normalize_staging_feasibility("Safe To Stage") == ("safe", True)
    assert normalize_staging_feasibility("Inseparable") == ("inseparable", True)
    assert normalize_staging_feasibility("unsafe") == ("unsafe", False)


@pytest.mark.parametrize(
    "mutate,read,expected",
    [
        (lambda r: r.update(strategy="Staged"), lambda p: p.execution_recommendation.strategy, "staged"),
        (lambda r: r["child_stages"][0].update(automation="agent_pr"),
         lambda p: p.execution_recommendation.child_stages[0].automation, "agent-pr"),
        (lambda r: r["child_stages"][0]["execution_disposition"].update(disposition="Requires Child Planning"),
         lambda p: p.execution_recommendation.child_stages[0].execution_disposition.disposition,
         "requires-child-planning"),
        (lambda r: r["retained_parent_work"].update(status="None"),
         lambda p: p.execution_recommendation.retained_parent_work.status, "none"),
    ],
    ids=["strategy", "automation", "execution-disposition", "allocation-status"],
)
def test_recommendation_enum_variants_fold(mutate, read, expected, caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    mutate(mutated["execution_recommendation"])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert read(parsed) == expected
    assert parsed == canonical
    _assert_logged(caplog, "folded enum value")


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r.update(strategy="single"), "strategy must be exactly"),
        (lambda r: r["child_stages"][0].update(automation="agent"), "automation must be one of"),
        (lambda r: r["retained_parent_work"].update(status="optional"), "status must be `none` or `required`"),
    ],
    ids=["strategy", "automation", "allocation"],
)
def test_recommendation_enum_synonyms_still_reject(mutate, match):
    mutated = _plan_state()
    mutate(mutated["execution_recommendation"])
    with pytest.raises(AgentLoopError, match=match):
        _parse_plan(mutated)


def test_payload_state_case_matches_footer(caplog):
    canonical = _parse_plan(_plan_state())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(_plan_state(state="Blocking"))
    assert parsed == canonical
    assert parsed.state == "blocking"
    _assert_logged(caplog, "plan_state.state: folded enum value")


def test_genuinely_different_payload_state_still_raises():
    with pytest.raises(AgentLoopError, match="must match the payload state"):
        _parse_plan(_plan_state(state="Approved"))


def test_footer_grammar_itself_is_not_relaxed():
    text = json.dumps(_plan_state()) + "\n<!-- AGENT_PLAN_STATE: Pending -->\n-- Anthropic Claude"
    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        validate_structured_plan_state(text, require_execution_strategy_contract=1)


def _pr_review(**extra) -> str:
    payload = {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "blocking",
        "summary": "Review complete.",
        "blocking_items": ["Gap remains."],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [],
    }
    payload.update(extra)
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"


def _parse_pr(**extra):
    return parse_structured_pr_review(_pr_review(**extra), reviewer="OpenAI Codex")


@pytest.mark.parametrize("spelling,canonical", [("Resolved", "resolved"), ("same_pr", "same-pr"), ("FUTURE", "future")])
def test_review_disposition_fold(spelling, canonical, caplog):
    def dispositions(value):
        return [{"item_id": "item-1", "disposition": value, "note": "Checked the change."}]

    expected = _parse_pr(prior_item_dispositions=dispositions(canonical))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_pr(prior_item_dispositions=dispositions(spelling))
    assert parsed == expected
    _assert_logged(caplog, "disposition: folded enum value")


def test_review_disposition_synonym_still_rejects():
    with pytest.raises(AgentLoopError, match="disposition must be one of"):
        _parse_pr(prior_item_dispositions=[{"item_id": "item-1", "disposition": "fixed", "note": "x"}])


def test_review_state_fold(caplog):
    expected = _parse_pr()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _parse_pr(state="Blocking") == expected


@pytest.mark.parametrize("spelling", ["Addressed", "not_applicable", "Not Applicable"])
def test_human_requirement_disposition_fold(spelling, caplog):
    label = "hr-" + "a" * 64
    def payload(value):
        return _issue_impl(
            human_requirements={"addressed_ids": [label], "checked_discussion_directly": False},
            human_requirement_dispositions=[{"requirement_id": label, "disposition": value, "evidence": "Done."}],
        )
    canonical_value = "addressed" if spelling == "Addressed" else "not-applicable"
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_impl(payload(spelling))
    assert parsed.human_requirement_dispositions[0].disposition == canonical_value


def _task(**extra) -> str:
    payload = {
        "schema_version": 1, "kind": "task_result", "state": "blocking",
        "outcome": "opened_pr", "summary": "Implemented.", "pr_number": 12,
    }
    payload.update(extra)
    return json.dumps(payload) + STATE_FOOTER


@pytest.mark.parametrize("spelling", ["Opened PR", "opened-pr", "OPENED_PR"])
def test_task_result_outcome_fold(spelling):
    assert validate_structured_task_result(_task(outcome=spelling)) == validate_structured_task_result(_task())


def test_task_result_outcome_synonym_rejects():
    with pytest.raises(AgentLoopError, match="task_result.outcome must be"):
        validate_structured_task_result(_task(outcome="pr_opened"))


def _risk_row(**extra) -> dict:
    row = {
        "row_id": "row-1", "label": "Label", "entry_path_or_mode": "mode",
        "initial_state": "start", "event": "event", "expected_outcome": "outcome",
        "forbidden_side_effects": ["none"], "proposed_test_level": "unit",
        "proposed_test_location": "tests/test_x.py", "related_scope_item_ids": [],
        "applicability": "required", "execution_owner": "one-shot",
    }
    row.update(extra)
    return row


def _risk_matrix(**extra) -> dict:
    matrix = {"applicability": "applicable", "rows": [_risk_row()], "important_exclusions": []}
    matrix.update(extra)
    return matrix


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(applicability="Applicable"),
        lambda m: m["rows"][0].update(applicability="Required"),
        lambda m: m["rows"][0].update(applicability=" REQUIRED "),
    ],
    ids=["matrix-applicability", "row-applicability", "row-applicability-spaced"],
)
def test_risk_matrix_applicability_fold(mutate, caplog):
    matrix = _risk_matrix()
    mutate(matrix)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _parse_risk_test_matrix(matrix) == _parse_risk_test_matrix(_risk_matrix())
    _assert_logged(caplog, "applicability: folded enum value")


def test_risk_matrix_change_operation_fold_and_synonym_rejects(caplog):
    canonical = _parse_risk_test_matrix_changes(
        [{"operation": "add", "row_ids": ["row-1"], "rationale": "New row."}]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _parse_risk_test_matrix_changes(
            [{"operation": "Add", "row_ids": ["row-1"], "rationale": "New row."}]
        ) == canonical
    with pytest.raises(AgentLoopError, match="operation is invalid"):
        _parse_risk_test_matrix_changes(
            [{"operation": "insert", "row_ids": ["row-1"], "rationale": "New row."}]
        )
    with pytest.raises(AgentLoopError, match="applicability is invalid"):
        _parse_risk_test_matrix(_risk_matrix(rows=[_risk_row(applicability="mandatory")]))


# ---------------------------------------------------------------------------
# (c') architecture status case fold ahead of near-miss handling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["Unchanged", "UNCHANGED", " unchanged "])
def test_architecture_status_unchanged_fold_all_modes(spelling, caplog):
    impact = dict(ARCH, status=spelling)
    assert classify_architecture_status_near_miss(impact, context="ctx") is None
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert parse_architecture_impact(impact) == parse_architecture_impact(dict(ARCH))
        degraded, record = parse_architecture_impact_degradable(impact)
    assert degraded.status == "unchanged" and record is None
    if spelling.strip() != "unchanged":
        _assert_logged(caplog, "architecture_impact.status: folded enum value")


@pytest.mark.parametrize("spelling", ["Changed", "CHANGED", " changed "])
def test_architecture_status_changed_fold(spelling):
    impact = dict(CORROBORATED_CHANGED, status=spelling)
    assert parse_architecture_impact(impact) == parse_architecture_impact(dict(CORROBORATED_CHANGED))


def test_required_planner_response_folds_architecture_status():
    canonical = _parse_plan(_plan_state(), required_architecture_impact_contract=1)
    parsed = _parse_plan(
        _plan_state(architecture_impact=dict(ARCH, status="Unchanged")),
        required_architecture_impact_contract=1, architecture_status_mode="degradable",
    )
    assert parsed.architecture_impact == canonical.architecture_impact
    assert parsed.architecture_impact_contract.satisfied is True
    assert parsed.architecture_impact_degradations == ()


def test_optional_review_assessment_text_is_kept_byte_identical():
    text = _pr_review(architecture_impact=dict(ARCH, status="Unchanged"))
    # Response normalization no longer removes the optional assessment.
    assert canonicalize_architecture_near_miss_text(text) == text
    parsed = parse_structured_pr_review(text, reviewer="OpenAI Codex", architecture_status_mode="degradable")
    assert parsed.architecture_impact.status == "unchanged"
    # The stored text re-parses strictly to the same value.
    strict = parse_structured_pr_review(text, reviewer="OpenAI Codex")
    assert strict.architecture_impact == parsed.architecture_impact


def test_semantic_patch_architecture_status_fold_preserves_patch_bytes():
    value = dict(ARCH, status="Unchanged")
    before = json.dumps(value, sort_keys=True)
    parsed = _parse_plan_patch_field_value("architecture_impact", value, context="patch.value")
    assert parsed.status == "unchanged"
    assert json.dumps(value, sort_keys=True) == before


@pytest.mark.parametrize("spelling", ["modified", "no_change", "Same"])
def test_architecture_synonyms_keep_degrade_handling(spelling):
    impact, record = parse_architecture_impact_degradable({"status": spelling, "rationale": "Moved."})
    assert impact.status == ARCHITECTURE_IMPACT_UNDETERMINED
    assert record is not None and record.outcome == "degraded-to-undetermined"
    with pytest.raises(AgentLoopError, match="not accepted in a patch"):
        _parse_plan_patch_field_value(
            "architecture_impact", {"status": spelling, "rationale": "Moved."}, context="patch.value"
        )


# ---------------------------------------------------------------------------
# (d) architecture_impact defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation,fragment",
    [
        ({"dependencies": None}, "dependencies: normalized null"),
        ({"affected_components": None}, "affected_components: normalized null"),
        ({"security_boundaries": None}, "security_boundaries: normalized null"),
        ({"canonical_document_path": ""}, "canonical_document_path: normalized an empty string"),
        ({"canonical_document_path": "  "}, "canonical_document_path: normalized an empty string"),
        ({"canonical_document_action": None}, "canonical_document_action: normalized null"),
        ({"rationale": ["No architectural contract changed."]}, "rationale: unwrapped"),
    ],
    ids=["deps-null", "components-null", "boundaries-null", "path-empty", "path-blank", "action-null", "rationale-list"],
)
def test_architecture_impact_defaults(mutation, fragment, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_architecture_impact(dict(ARCH, **mutation))
    assert parsed == parse_architecture_impact(dict(ARCH))
    _assert_logged(caplog, fragment)


def test_architecture_impact_lone_string_list():
    lone = dict(CORROBORATED_CHANGED, dependencies="None new.")
    assert parse_architecture_impact(lone) == parse_architecture_impact(dict(CORROBORATED_CHANGED))


@pytest.mark.parametrize(
    "mutation",
    [{"canonical_document_path": 7}, {"dependencies": 123}, {"rationale": ["a", "b"]}],
    ids=["path-number", "deps-number", "rationale-multi"],
)
def test_architecture_impact_ambiguous_still_rejects(mutation):
    with pytest.raises(AgentLoopError):
        parse_architecture_impact(dict(ARCH, **mutation))


# ---------------------------------------------------------------------------
# (e) allocation defaults and staged null one_shot_delivery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [None, {"status": "none"}, {"status": "None", "deliverables": []}],
    ids=["null", "status-only", "partial"],
)
def test_none_allocation_defaults(value, caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    mutated["execution_recommendation"]["retained_parent_work"] = value
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert parsed == canonical
    _assert_logged(caplog, "retained_parent_work:")


def test_staged_null_one_shot_delivery_is_dropped(caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    mutated["execution_recommendation"]["one_shot_delivery"] = None
    mutated["execution_recommendation"]["retained_parent_work"] = None
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_plan(mutated)
    assert parsed == canonical
    _assert_logged(caplog, "dropped null one_shot_delivery")


def test_required_allocation_missing_lists_still_raises():
    mutated = _plan_state()
    mutated["execution_recommendation"]["final_integration_work"] = {"status": "required"}
    with pytest.raises(AgentLoopError, match="missing required field"):
        _parse_plan(mutated)


def test_one_shot_null_delivery_still_raises():
    recommendation = _one_shot_recommendation()
    recommendation["one_shot_delivery"] = None
    with pytest.raises(AgentLoopError, match="one_shot_delivery must be a JSON object"):
        _parse_plan(_plan_state(recommendation))


# ---------------------------------------------------------------------------
# (f) digit-string integers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["77", "#77", " 77 ", " #77\n"])
def test_issue_implementation_pr_number_digit_string(value, caplog):
    canonical = _parse_impl(_issue_impl())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_impl(_issue_impl(pr_number=value))
    assert parsed.pr_number == 77 and isinstance(parsed.pr_number, int)
    assert parsed == canonical
    _assert_logged(caplog, "issue_implementation.pr_number: converted digit string to integer 77")


@pytest.mark.parametrize("value", ["PR 77", "77a", "-3", "0", True, False, "", "7.0", 7.0])
def test_issue_implementation_pr_number_ambiguous_still_raises(value):
    with pytest.raises(AgentLoopError, match="pr_number must be"):
        _parse_impl(_issue_impl(pr_number=value))


def test_int_like_helper():
    assert _expect_int_like(5, context="ctx") == 5
    assert _expect_int_like("#5", context="ctx") == 5
    with pytest.raises(AgentLoopError, match="ctx must be an integer"):
        _expect_int_like("five", context="ctx")


def test_task_result_pr_number_digit_string():
    assert validate_structured_task_result(_task(pr_number="#12")) == validate_structured_task_result(_task())


def test_stage_position_digit_string(caplog):
    canonical = _parse_plan(_plan_state())
    mutated = _plan_state()
    mutated["execution_recommendation"]["child_stages"][0]["position"] = "1"
    mutated["execution_recommendation"]["child_stages"][1]["position"] = "#2"
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert _parse_plan(mutated) == canonical
    _assert_logged(caplog, "position: converted digit string")
    mutated["execution_recommendation"]["child_stages"][0]["position"] = "first"
    with pytest.raises(AgentLoopError, match="position must be an integer"):
        _parse_plan(mutated)


# ---------------------------------------------------------------------------
# (g) PR finding aliases and fix_scope wrapping
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alias", ["description", "issue"])
def test_pr_finding_text_alias(alias, caplog):
    canonical = _parse_pr(blocking_items=[{"text": "Gap remains.", "fix_scope": ["src/a.py"]}])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_pr(blocking_items=[{alias: "Gap remains.", "fix_scope": ["src/a.py"]}])
    assert parsed == canonical
    _assert_logged(caplog, f"aliased finding key `{alias}` to `text`")


def test_pr_finding_both_aliases_without_text_still_raises():
    with pytest.raises(AgentLoopError, match="missing required field"):
        _parse_pr(blocking_items=[{"description": "A.", "issue": "B."}])


def test_pr_finding_alias_does_not_override_text():
    parsed = _parse_pr(blocking_items=[{"text": "Gap.", "description": "More detail."}])
    # With `text` present, `description` keeps the existing unknown-key fold.
    assert parsed.blocking_items[0].text.startswith("Gap.")
    assert "More detail." in parsed.blocking_items[0].text


def test_pr_finding_lone_string_fix_scope(caplog):
    canonical = _parse_pr(blocking_items=[{"text": "Gap.", "fix_scope": ["a.py"]}])
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_pr(blocking_items=[{"text": "Gap.", "fix_scope": "a.py"}])
    assert parsed == canonical
    assert parsed.blocking_items[0].fix_scope == ("a.py",)
    _assert_logged(caplog, "fix_scope: wrapped a lone string")


@pytest.mark.parametrize("key", ["item_id", "id", "disposition"])
def test_pr_finding_collision_keys_still_reject(key):
    with pytest.raises(AgentLoopError):
        _parse_pr(blocking_items=[{"description": "Gap.", key: "item-1"}])


# ---------------------------------------------------------------------------
# (h) finding lists as null or a lone string
# ---------------------------------------------------------------------------


def _plan_review(**extra) -> str:
    payload = {
        "schema_version": 1, "kind": "plan_review", "state": "blocking",
        "summary": "Plan review.", "blocking_plan_issues": ["Fix the plan."],
        "same_plan_followups": [], "future_followups": [],
        "prior_plan_item_dispositions": [], "human_requirement_dispositions": [],
    }
    payload.update(extra)
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ({"future_followups": None}, {}),
        ({"same_plan_followups": None}, {}),
        ({"blocking_plan_issues": "Fix the plan."}, {}),
    ],
    ids=["future-null", "same-null", "blocking-lone"],
)
def test_plan_review_finding_lists_normalize(mutation, expected, caplog):
    canonical = parse_structured_plan_review(_plan_review(), reviewer="OpenAI Codex")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_structured_plan_review(_plan_review(**mutation), reviewer="OpenAI Codex")
    assert parsed == canonical
    _assert_logged(caplog, f"plan_review.{next(iter(mutation))}:")


@pytest.mark.parametrize(
    "mutation",
    [{"future_followups": None}, {"same_pr_followups": None}, {"blocking_items": "Gap remains."}],
    ids=["future-null", "same-null", "blocking-lone"],
)
def test_pr_review_finding_lists_normalize(mutation, caplog):
    canonical = _parse_pr()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _parse_pr(**mutation)
    assert parsed == canonical
    _assert_logged(caplog, f"pr_review.{next(iter(mutation))}:")


@pytest.mark.parametrize("value", [{"a": "b"}, 7], ids=["object", "number"])
def test_finding_lists_still_reject_objects_and_numbers(value):
    with pytest.raises(AgentLoopError, match="must be a JSON array"):
        _parse_pr(future_followups=value)
    with pytest.raises(AgentLoopError, match="must be a JSON array"):
        parse_structured_plan_review(_plan_review(future_followups=value), reviewer="OpenAI Codex")


def test_plan_review_finding_collision_still_rejects():
    with pytest.raises(AgentLoopError):
        parse_structured_plan_review(
            _plan_review(blocking_plan_issues=[{"text": "Fix.", "disposition": "resolved"}]),
            reviewer="OpenAI Codex",
        )


# ---------------------------------------------------------------------------
# strict identifiers and signed labels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda r: r["child_stages"][0].update(stage_id=["stage-1"]), "stage_id must be a string"),
        (lambda r: r["scope_items"][0].update(scope_item_id=["scope-1"]), "scope_item_id must be a string"),
        (lambda r: r["child_stages"][0].update(covered_scope_item_ids=[["scope-1"]]), "must be a string"),
    ],
    ids=["stage_id", "scope_item_id", "covered-entry"],
)
def test_list_wrapped_identifiers_still_reject(mutate, match):
    mutated = _plan_state()
    mutate(mutated["execution_recommendation"])
    with pytest.raises(AgentLoopError, match=match):
        _parse_plan(mutated)


def test_list_wrapped_signed_label_and_item_id_still_reject():
    label = "hr-" + "b" * 64
    with pytest.raises(AgentLoopError, match="requirement_id must be a string"):
        _parse_impl(_issue_impl(
            human_requirement_dispositions=[{"requirement_id": [label], "disposition": "addressed", "evidence": "x"}],
        ))
    with pytest.raises(AgentLoopError, match="item_id must be a string"):
        _parse_pr(prior_item_dispositions=[{"item_id": ["item-1"], "disposition": "resolved", "note": "x"}])


# ---------------------------------------------------------------------------
# deterministic re-parse of stored text
# ---------------------------------------------------------------------------


def test_stored_text_with_non_canonical_shapes_reparses_identically():
    mutated = _plan_state(state="Blocking", plan_steps="Implement the reviewed scope.")
    recommendation = mutated["execution_recommendation"]
    recommendation["strategy"] = "Staged"
    recommendation["caveats"] = None
    recommendation["retained_parent_work"] = None
    recommendation["child_stages"][0]["position"] = "1"
    recommendation["child_stages"][1]["dependency_notes"] = ["The implementation must be merged first."]
    mutated["architecture_impact"] = dict(ARCH, status="Unchanged", dependencies=None)
    text = _plan_text(mutated)
    snapshot = copy.copy(text)
    first = validate_structured_plan_state(text, require_execution_strategy_contract=1)
    second = validate_structured_plan_state(text, require_execution_strategy_contract=1)
    assert first == second
    assert first.execution_recommendation.identity() == second.execution_recommendation.identity()
    assert first == _parse_plan(_plan_state())
    assert text == snapshot


# ---------------------------------------------------------------------------
# Round 2: envelope state stays string-shaped; action fold; discussion enums
# and strict discussion identifiers.
# ---------------------------------------------------------------------------

from coding_review_agent_loop.protocol import (  # noqa: E402
    architecture_impact_near_miss_corroborated,
    validate_structured_coder_followup,
)
from coding_review_agent_loop.protocol_discuss import (  # noqa: E402
    _parse_discuss_research,
    _parse_discuss_synthesis_change,
    _parse_discuss_synthesis_position,
    _parse_discuss_synthesis_reference,
    _parse_discuss_unresolved_items,
    parse_structured_discuss_answer,
    parse_structured_discuss_review,
    validate_structured_discuss_answer_confirmation,
    validate_structured_discuss_final_synthesis,
    validate_structured_discuss_semantic_comparison,
)


def _coder_followup(state) -> str:
    payload = {
        "schema_version": 1, "kind": "coder_followup", "state": state, "summary": "Updated.",
        "addressed_items": [], "remaining_items": [], "addressed_item_notes": {},
        "remaining_item_notes": {},
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [], "architecture_impact": dict(ARCH),
    }
    return json.dumps(payload) + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"


def _contradictory_footer_texts():
    plan = json.dumps(_plan_state(state=["blocking"])) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    pr = json.dumps({
        "schema_version": 1, "kind": "pr_review", "state": ["approved"], "summary": "Done.",
        "blocking_items": [], "same_pr_followups": [], "future_followups": [],
        "prior_item_dispositions": [],
    }) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    plan_review = json.dumps({
        "schema_version": 1, "kind": "plan_review", "state": ["approved"], "summary": "Done.",
        "blocking_plan_issues": [], "same_plan_followups": [], "future_followups": [],
        "prior_plan_item_dispositions": [], "human_requirement_dispositions": [],
    }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    impl = json.dumps(_issue_impl(state=["blocking"])) + STATE_FOOTER
    return [
        ("plan_state", lambda: validate_structured_plan_state(plan, require_execution_strategy_contract=1)),
        ("pr_review", lambda: parse_structured_pr_review(pr, reviewer="OpenAI Codex")),
        ("plan_review", lambda: parse_structured_plan_review(plan_review, reviewer="OpenAI Codex")),
        ("coder_followup", lambda: validate_structured_coder_followup(_coder_followup(["blocking"]))),
        ("issue_implementation", lambda: validate_structured_issue_implementation(impl)),
        ("task_result", lambda: validate_structured_task_result(_task(state=["blocking"]))),
    ]


@pytest.mark.parametrize(
    "name,parse", _contradictory_footer_texts(), ids=[n for n, _ in _contradictory_footer_texts()]
)
def test_singleton_list_state_never_bypasses_the_footer(name, parse):
    # The payload `state` is an envelope field compared with the footer: it
    # stays strictly string-shaped, so a list never slips past the comparison.
    with pytest.raises(AgentLoopError, match=r"state must be a string"):
        parse()


@pytest.mark.parametrize("spelling", ["NO_CHANGE", "No Change", "no_change", " no-change "])
def test_canonical_document_action_folds_documented_token(spelling, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_architecture_impact(dict(ARCH, canonical_document_action=spelling))
    assert parsed == parse_architecture_impact(dict(ARCH))
    assert parsed.canonical_document_action == "no-change"


def test_canonical_document_action_open_text_is_kept_and_not_corroborating():
    assert parse_architecture_impact(dict(CORROBORATED_CHANGED)).canonical_document_action == "update"
    # A folded no-change spelling is not a real document action for `modified`.
    assert not architecture_impact_near_miss_corroborated(
        dict(CORROBORATED_CHANGED, status="modified", canonical_document_action="NO_CHANGE")
    )


def _discuss_review(**extra) -> str:
    payload = {"schema_version": 1, "kind": "discuss_review", "outcome": "implement", "rationale": "Do it."}
    payload.update(extra)
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"


def _discuss_answer(**extra) -> str:
    payload = {
        "schema_version": 1, "kind": "discuss_answer", "position": "answer",
        "rationale": "Because.", "confidence": "high", "answer": "Yes.", "unresolved_items": [],
    }
    payload.update(extra)
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"


def _confirmation(decision) -> str:
    payload = {"schema_version": 1, "kind": "discuss_answer_confirmation", "decision": decision,
               "rationale": "Still right.", "reviewer": "OpenAI Codex"}
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"


def _semantic(classification) -> str:
    payload = {"schema_version": 1, "kind": "discuss_semantic_comparison",
               "classification": classification, "shared_recommendation": "Do it.",
               "remaining_decisions": [], "evidence": [{"reviewer": "A", "supports": "Yes."}]}
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Orchestrator"


def _final_synthesis(classification) -> str:
    payload = {"schema_version": 1, "kind": "discuss_final_synthesis", "classification": classification,
               "agreed_conclusions": [{"text": "Ship.", "references": [{"reviewer": "A", "round": 1}]}],
               "remaining_disagreements": [], "next_action": "Ship it."}
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Orchestrator"


DISCUSS_FOLDS = [
    ("review-outcome", lambda v: parse_structured_discuss_review(_discuss_review(outcome=v), reviewer="X").outcome,
     "Do Not Implement", "do-not-implement", "skip"),
    ("review-framing", lambda v: parse_structured_discuss_review(
        _discuss_review(analyzer_framing=v), reviewer="X").analyzer_framing, "Accurate", "accurate", "right"),
    ("research-status", lambda v: _parse_discuss_research({"status": v})[0], "Not Needed", "not-needed", "skipped"),
    ("research-target", lambda v: _parse_discuss_research(
        {"status": "inconclusive", "target": v, "questions": ["Q?"]})[2],
     "Solution Design", "solution-design", "design"),
    ("answer-position", lambda v: parse_structured_discuss_answer(
        _discuss_answer(position=v, answer=None) if v.lower().startswith("needs") else _discuss_answer(position=v),
        reviewer="X").position, "Answer", "answer", "reply"),
    ("answer-confidence", lambda v: parse_structured_discuss_answer(_discuss_answer(confidence=v), reviewer="X").confidence,
     "High", "high", "certain"),
    ("unresolved-status", lambda v: _parse_discuss_unresolved_items([{"status": v, "text": "Q?"}], context="ctx")[0].status,
     "Human Decision", "human-decision", "question"),
    ("confirmation-decision", lambda v: validate_structured_discuss_answer_confirmation(
        _confirmation(v), reviewer="OpenAI Codex").decision, "Confirm", "confirm", "accept"),
    ("semantic-classification", lambda v: validate_structured_discuss_semantic_comparison(
        _semantic(v), reviewers=["A"]).classification, "Material Conflict", "material_conflict", "conflict"),
    ("final-synthesis-classification", lambda v: validate_structured_discuss_final_synthesis(
        _final_synthesis(v)).classification, " CONSENSUS ", "consensus", "agreement"),
]


@pytest.mark.parametrize("name,read,variant,canonical,synonym", DISCUSS_FOLDS, ids=[d[0] for d in DISCUSS_FOLDS])
def test_discussion_closed_enums_fold_and_synonyms_reject(name, read, variant, canonical, synonym):
    assert read(variant) == canonical
    assert read(canonical) == canonical
    with pytest.raises(AgentLoopError):
        read(synonym)


@pytest.mark.parametrize(
    "parse",
    [
        lambda: _parse_discuss_synthesis_reference({"reviewer": ["Claude"], "round": 1}, context="ref"),
        lambda: _parse_discuss_synthesis_change(
            {"kind": ["introduced"], "topic": "T", "text": "X", "references": []}, context="change"),
        lambda: _parse_discuss_synthesis_position({"reviewers": "Claude", "position": "P"}, context="pos"),
        lambda: _parse_discuss_synthesis_position({"reviewers": [["Claude"]], "position": "P"}, context="pos"),
    ],
    ids=["reference-reviewer", "change-kind", "position-reviewers-lone", "position-reviewers-nested"],
)
def test_discussion_synthesis_identifiers_and_kinds_stay_strict(parse):
    with pytest.raises(AgentLoopError):
        parse()


def test_discussion_synthesis_strict_fields_still_accept_canonical():
    assert _parse_discuss_synthesis_reference({"reviewer": "Claude", "round": 1}, context="ref").reviewer == "Claude"
    assert _parse_discuss_synthesis_position(
        {"reviewers": ["Claude"], "position": ["P"]}, context="pos"
    ).position == "P"
