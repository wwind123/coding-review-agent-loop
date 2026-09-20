import json
from copy import deepcopy

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.protocol import validate_structured_task_result
from coding_review_agent_loop.repair import _build_repair_prompt
from coding_review_agent_loop.repair_preservation import validate_repair_preservation
from agent_loop_helpers import structured_v1_plan_state


def check(source, target):
    validate_repair_preservation(json.dumps(source), json.dumps(target))


def valid_task_result(impact):
    payload = {
        "schema_version": 1,
        "kind": "task_result",
        "state": "blocking",
        "outcome": "blocking",
        "summary": "The task is blocked.",
        "architecture_impact": impact,
    }
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Coder"


def _execution_source(*, strategy: str) -> dict:
    raw_source = structured_v1_plan_state()
    source, _ = json.JSONDecoder().raw_decode(raw_source.lstrip())
    recommendation = source["execution_recommendation"]
    if strategy == "one-shot":
        recommendation["scope_items"] = [
            {
                "scope_item_id": "scope-1",
                "requirement": "Deliver the API behavior.",
                "acceptance_criteria": ["The API behavior is complete."],
            },
            {
                "scope_item_id": "scope-2",
                "requirement": "Deliver its compatibility tests.",
                "acceptance_criteria": ["The compatibility tests pass."],
            },
        ]
        recommendation["coupling_constraints"] = [{
            "constraint_id": "coupling-1",
            "scope_item_ids": ["scope-1", "scope-2"],
            "rationale": "The API and compatibility tests ship together.",
        }]
        recommendation["one_shot_delivery"]["covered_scope_item_ids"] = [
            "scope-1", "scope-2"
        ]
    else:
        recommendation.update({
            "strategy": "staged",
            "staging_feasibility": "safe",
            "scope_items": [
                {
                    "scope_item_id": "scope-1",
                    "requirement": "Deliver the intermediate API behavior.",
                    "acceptance_criteria": ["The intermediate API works."],
                },
                {
                    "scope_item_id": "scope-2",
                    "requirement": "Deliver compatibility tests.",
                    "acceptance_criteria": ["Compatibility tests pass."],
                },
                {
                    "scope_item_id": "scope-3",
                    "requirement": "Complete final integration.",
                    "acceptance_criteria": ["The integrated behavior works."],
                },
            ],
            "coupling_constraints": [{
                "constraint_id": "coupling-1",
                "scope_item_ids": ["scope-1", "scope-2"],
                "rationale": "The API and its tests must stay together.",
            }],
            "child_stages": [
                {
                    "stage_id": "stage-1", "position": 1, "title": "API and tests",
                    "summary": "Deliver the intermediate API and its tests.",
                    "deliverables": ["API behavior and tests."],
                    "non_goals": ["No final integration."],
                    "acceptance_criteria": ["The intermediate API and tests pass."],
                    "depends_on_stage_ids": [], "dependency_notes": "No dependencies.",
                    "automation": "agent-pr", "rollout_risk": "low",
                    "compatibility_constraints": ["Keep the old entry point."],
                    "covered_scope_item_ids": ["scope-1", "scope-2"],
                },
                {
                    "stage_id": "stage-2", "position": 2, "title": "Final integration",
                    "summary": "Complete the final integration.",
                    "deliverables": ["Integrated behavior."],
                    "non_goals": [],
                    "acceptance_criteria": ["The integrated behavior passes."],
                    "depends_on_stage_ids": ["stage-1"],
                    "dependency_notes": "Run after the API and tests.",
                    "automation": "human-action", "rollout_risk": "medium",
                    "compatibility_constraints": ["Preserve the compatibility boundary."],
                    "covered_scope_item_ids": ["scope-3"],
                },
            ],
        })
        recommendation.pop("one_shot_delivery", None)
    return source


@pytest.mark.parametrize("strategy", ["one-shot", "staged"])
def test_repair_preserves_every_nested_execution_recommendation_field(strategy):
    source = _execution_source(strategy=strategy)
    check(source, deepcopy(source))

    recommendation = source["execution_recommendation"]
    paths = [
        ("strategy", "one-shot" if strategy == "staged" else "staged"),
        ("rationale", "A changed rationale."),
        ("staging_feasibility", "safe" if strategy == "one-shot" else "inseparable"),
        ("caveats", ["A changed caveat."]),
        ("scope_items.0.scope_item_id", "scope-other"),
        ("scope_items.0.requirement", "A changed requirement."),
        ("scope_items.0.acceptance_criteria.0", "A changed criterion."),
        ("coupling_constraints.0.constraint_id", "coupling-other"),
        ("coupling_constraints.0.scope_item_ids.0", "scope-other"),
        ("coupling_constraints.0.rationale", "A changed coupling rationale."),
        ("retained_parent_work.status", "required"),
        ("final_integration_work.status", "required"),
    ]
    if strategy == "one-shot":
        paths.extend([
            ("one_shot_delivery.deliverables.0", "A changed deliverable."),
            ("one_shot_delivery.acceptance_criteria.0", "A changed delivery criterion."),
            ("one_shot_delivery.covered_scope_item_ids.0", "scope-other"),
        ])
    else:
        paths.extend([
            ("child_stages.0.stage_id", "stage-other"),
            ("child_stages.0.position", 2),
            ("child_stages.0.title", "Changed stage"),
            ("child_stages.0.summary", "Changed stage summary."),
            ("child_stages.0.deliverables.0", "Changed stage deliverable."),
            ("child_stages.0.non_goals.0", "Changed stage non-goal."),
            ("child_stages.0.acceptance_criteria.0", "Changed stage criterion."),
            ("child_stages.0.dependency_notes", "Changed dependency notes."),
            ("child_stages.0.automation", "manual-close"),
            ("child_stages.0.rollout_risk", "high"),
            ("child_stages.0.compatibility_constraints.0", "Changed compatibility."),
            ("child_stages.0.covered_scope_item_ids.0", "scope-other"),
            ("child_stages.1.depends_on_stage_ids.0", "stage-other"),
        ])
    for path, replacement in paths:
        target = deepcopy(source)
        cursor = target["execution_recommendation"]
        parts = path.split(".")
        for part in parts[:-1]:
            cursor = cursor[int(part)] if part.isdigit() else cursor[part]
        last = parts[-1]
        cursor[int(last) if last.isdigit() else last] = replacement
        with pytest.raises(AgentLoopError, match="execution_recommendation"):
            check(source, target)


def test_prompt_demands_lossless_repair_and_separate_ledgers():
    prompt = _build_repair_prompt("malformed", expected_kind="coder_followup",
                                  unresolved_item_ids=("item-1",),
                                  surfaced_requirement_ids=())
    assert "This is format repair, not summarization" in prompt
    assert "two-round claim test and assert no mutation on 503" in prompt
    assert "including no not-applicable rows" in prompt
    assert "Retain failure, timeout, skipped-test" in prompt
    assert "`addressed_items`, `remaining_items`, or `disputed_items`" in prompt
    assert "Preserve a source `disputed_items` classification" in prompt


def test_repair_preserves_matrix_evidence_status_and_caveats():
    source = {
        "schema_version": 1,
        "kind": "coder_followup",
        "state": "blocking",
        "summary": "The owned workflow row remains incomplete.",
        "addressed_items": [],
        "remaining_items": ["item-1"],
        "disputed_items": [],
        "risk_test_matrix_evidence": {
            "matrix_identity": "a" * 64,
            "rows": [{
                "row_id": "row-recovery",
                "status": "timed-out",
                "test_identifiers": ["test_recovery"],
                "test_locations": ["tests/test_orchestrator_pr.py:1"],
                "workflow_path_claim": "The post-review recovery path was not reached.",
                "outcome_assertions": ["The gate timed out."],
                "forbidden_effect_assertions": ["No merge occurred."],
                "evidence_citations": [{
                    "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
                    "receipt_id": "receipt-timeout", "claim": "current-result",
                }],
                "caveats": ["managed test gate timed out before branch entry"],
            }],
        },
    }
    check(source, deepcopy(source))
    changed = deepcopy(source)
    changed["risk_test_matrix_evidence"]["rows"][0]["status"] = "verified"
    with pytest.raises(AgentLoopError, match="risk_test_matrix_evidence"):
        check(source, changed)


def test_fresh_coder_repair_may_remove_legacy_canonical_evidence():
    source = {
        "schema_version": 1,
        "kind": "issue_implementation",
        "state": "blocking",
        "summary": "The implementation is complete.",
        "pr_number": 77,
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [],
        "risk_test_matrix_evidence": {"matrix_identity": "a" * 64, "rows": []},
    }
    repaired = deepcopy(source)
    repaired.pop("risk_test_matrix_evidence")

    # Preservation keeps valid coder-owned facts, but does not force a fresh
    # semantic repair to carry an orchestrator-owned legacy field forward.
    check(source, repaired)


def test_semantic_repair_may_remove_invalid_selector_and_keep_valid_facts():
    source = {
        "kind": "coder_followup",
        "risk_test_matrix_claims": [{
            "row_id": "row-1",
            "execution_refs": ["turn:valid", "other-turn:invalid"],
            "test_identifiers": ["tests/test_protocol.py::test_valid"],
            "test_locations": ["tests/test_protocol.py"],
            "workflow_path_claim": "The current workflow path ran.",
            "outcome_assertions": ["The selected test passed."],
            "forbidden_effect_assertions": ["No unauthorized effect occurred."],
        }],
    }
    repaired = deepcopy(source)
    repaired["risk_test_matrix_claims"][0]["execution_refs"] = ["turn:valid"]

    # The repaired response keeps the valid selector and all semantic facts;
    # the current-turn catalog validator will reject the removed cross-turn
    # selector separately.
    check(source, repaired)


def test_fresh_coder_repair_cannot_invent_canonical_evidence():
    source = {
        "schema_version": 1,
        "kind": "coder_followup",
        "state": "blocking",
        "summary": "The implementation is complete.",
        "addressed_items": [],
        "remaining_items": [],
    }
    target = deepcopy(source)
    target["risk_test_matrix_evidence"] = {
        "matrix_identity": "a" * 64,
        "rows": [],
    }

    with pytest.raises(AgentLoopError, match="cannot invent risk_test_matrix_evidence"):
        check(source, target)


def test_repair_preserves_architecture_impact_fields():
    source = {
        "kind": "task_result",
        "summary": "Implemented the task.",
        "architecture_impact": {
            "status": "changed",
            "rationale": "The execution flow crosses a new persistence boundary.",
            "affected_components": ["task loop"],
            "dependencies": ["round metadata"],
            "execution_data_flows": ["task -> round metadata"],
            "persistence": ["architecture impact sidecar"],
            "public_contracts": ["task_result"],
            "security_boundaries": ["untrusted prompt context"],
            "canonical_document_action": "update",
            "canonical_document_path": "ARCHITECTURE.md",
            "canonical_document_rationale": "Document the new persistence boundary.",
            "uncertainty": ["provider behavior remains external"],
        },
    }
    repaired = json.loads(json.dumps(source))
    check(source, repaired)
    repaired["architecture_impact"]["uncertainty"] = []
    with pytest.raises(AgentLoopError, match="architecture_impact.uncertainty"):
        check(source, repaired)


def test_repair_preserves_marker_only_architecture_entry():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "uncertainty": ["`AGENT_SPLIT_UNFILED_WARNING`"],
        },
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "uncertainty": ["[protocol split-warning record]"],
        },
    }
    check(source, repaired)
    with pytest.raises(AgentLoopError, match="architecture_impact.uncertainty"):
        check(source, {"kind": "task_result", "architecture_impact": {"uncertainty": []}})


def test_architecture_entries_require_distinct_one_to_one_matches():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "affected_components": ["shared component", "shared component"],
        },
    }
    check(source, source)
    with pytest.raises(AgentLoopError, match="architecture_impact.affected_components"):
        check(source, {
            "kind": "task_result",
            "architecture_impact": {"affected_components": ["shared component", "unrelated"]},
        })
    with pytest.raises(AgentLoopError, match="architecture_impact.affected_components"):
        check(source, {
            "kind": "task_result",
            "architecture_impact": {"affected_components": ["shared component"]},
        })


def test_malformed_architecture_list_cannot_drop_valid_entries():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "affected_components": ["orchestrator.py", 123],
        },
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "unchanged",
            "rationale": "No architectural contract changed.",
            "affected_components": ["orchestrator.py"],
        },
    }
    validate_structured_task_result(valid_task_result(repaired["architecture_impact"]))
    check(source, repaired)

    dropped = deepcopy(repaired)
    dropped["architecture_impact"]["affected_components"] = []
    with pytest.raises(AgentLoopError, match="architecture_impact.affected_components"):
        check(source, dropped)


def test_architecture_scalar_fields_cannot_be_omitted_or_type_changed():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "rationale": "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
    }

    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "rationale": "Managed CI override record.",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
    }
    check(source, repaired)

    for key in source["architecture_impact"]:
        omitted = deepcopy(repaired)
        del omitted["architecture_impact"][key]
        with pytest.raises(AgentLoopError, match=f"architecture_impact\\.{key}"):
            check(source, omitted)

    wrong_type = deepcopy(repaired)
    wrong_type["architecture_impact"]["rationale"] = None
    with pytest.raises(AgentLoopError, match="architecture_impact.rationale"):
        check(source, wrong_type)

    changed_null = deepcopy(repaired)
    changed_null["architecture_impact"]["canonical_document_path"] = "ARCHITECTURE.md"
    with pytest.raises(AgentLoopError, match="architecture_impact.canonical_document_path"):
        check(source, changed_null)

    changed_empty = deepcopy(repaired)
    changed_empty["architecture_impact"]["canonical_document_rationale"] = "Added a default."
    with pytest.raises(AgentLoopError, match="architecture_impact.canonical_document_rationale"):
        check(source, changed_empty)


def test_invalid_architecture_source_key_does_not_deadlock_repair():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "changed",
            "rationale": "The source rationale remains relevant.",
            "componets": ["Malformed source key."],
        },
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "changed",
            "rationale": "The source rationale remains relevant.",
            "affected_components": [],
            "dependencies": [],
            "execution_data_flows": [],
            "persistence": [],
            "public_contracts": [],
            "security_boundaries": [],
            "canonical_document_action": "no-change",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
    }
    validate_structured_task_result(valid_task_result(repaired["architecture_impact"]))
    check(source, repaired)


def test_invalid_architecture_empty_rationale_does_not_deadlock_repair():
    source = {
        "kind": "task_result",
        "architecture_impact": {"status": "changed", "rationale": ""},
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "changed",
            "rationale": "The corrected assessment is complete.",
            "affected_components": [],
            "dependencies": [],
            "execution_data_flows": [],
            "persistence": [],
            "public_contracts": [],
            "security_boundaries": [],
            "canonical_document_action": "no-change",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
    }
    validate_structured_task_result(valid_task_result(repaired["architecture_impact"]))
    check(source, repaired)


def test_invalid_architecture_status_does_not_deadlock_repair():
    source = {
        "kind": "task_result",
        "architecture_impact": {"status": 123, "rationale": "The rationale remains."},
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "changed",
            "rationale": "The rationale remains.",
            "affected_components": [],
            "dependencies": [],
            "execution_data_flows": [],
            "persistence": [],
            "public_contracts": [],
            "security_boundaries": [],
            "canonical_document_action": "no-change",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
    }
    validate_structured_task_result(valid_task_result(repaired["architecture_impact"]))
    check(source, repaired)


def test_architecture_flow_aliases_can_be_normalized_to_combined_field():
    source = {
        "kind": "task_result",
        "architecture_impact": {
            "execution_flows": ["agent -> round metadata"],
            "data_flows": ["round metadata -> review"],
        },
    }
    repaired = {
        "kind": "task_result",
        "architecture_impact": {
            "status": "unchanged",
            "rationale": "No architectural contract changed.",
            "execution_data_flows": [
                "agent -> round metadata",
                "round metadata -> review",
            ],
        },
    }
    validate_structured_task_result(valid_task_result(repaired["architecture_impact"]))
    check(source, repaired)


def test_case06_object_to_string_keeps_every_detail():
    detail = "Wire the capability getter in app.js:42; add a two-round test; no mutation on 503."
    source = {"kind": "plan_review", "blocking_plan_issues": [
        {"id": "item-1", "title": "Add coverage", "detail": detail},
    ]}
    check(source, {"kind": "plan_review", "blocking_plan_issues": ["Add coverage: " + detail]})
    with pytest.raises(AgentLoopError, match="blocking_plan_issues"):
        check(source, {"kind": "plan_review", "blocking_plan_issues": ["Add coverage for claims."]})


def test_finding_metadata_is_not_required_in_object_to_string_repair():
    source = {"kind": "pr_review", "blocking_items": [{
        "title": "Keep the diagnostic",
        "detail": "The fallback drops the original failure.",
        "severity": "high",
        "category": "correctness",
        "state": "blocking",
        "disposition": "unresolved",
        "verdict": "must-fix",
        "file": "src/repair.py",
        "line": "1435",
        "evidence": "The lossy candidate is accepted.",
    }]}
    check(source, {"kind": "pr_review", "blocking_items": [
        "Keep the diagnostic: The fallback drops the original failure. "
        "src/repair.py:1435 — The lossy candidate is accepted.",
    ]})

    with pytest.raises(AgentLoopError, match="blocking_items"):
        check(source, {"kind": "pr_review", "blocking_items": [
            "Keep the diagnostic: The fallback drops the original failure.",
        ]})


@pytest.mark.parametrize("field", ["summary", "tests_run", "plan_steps",
                                 "addressed_item_notes", "remaining_item_notes"])
def test_dropped_or_summarized_content_rejected(field):
    text = "pytest tests/test_claims.py: 5 passed; one unrelated test FAILED."
    value = {"summary": text, "tests_run": [text], "plan_steps": [text],
             "addressed_item_notes": {"item-1": text},
             "remaining_item_notes": {"item-1": text}}[field]
    kind = "plan_revision" if field == "plan_steps" else "coder_followup"
    source = {"kind": kind, field: value}
    check(source, source)
    with pytest.raises(AgentLoopError, match=field):
        check(source, {"kind": kind})


def test_case10_does_not_lose_failing_test_from_summary():
    source = {"kind": "coder_followup", "summary": "Coverage incomplete. Localization test FAILED."}
    with pytest.raises(AgentLoopError, match="summary"):
        check(source, {"kind": "coder_followup", "summary": "Coverage incomplete."})


def test_case10_gemini_test_list_omission_rejected():
    source = {"kind": "coder_followup", "tests_run": ["pytest tests/test_jobs.py -q"]}
    with pytest.raises(AgentLoopError, match="tests_run"):
        check(source, {"kind": "coder_followup"})


def test_findings_cannot_be_combined_or_dropped():
    source = {"kind": "pr_review", "blocking_items": ["Fix A", "Fix B"]}
    with pytest.raises(AgentLoopError):
        check(source, {"kind": "pr_review", "blocking_items": ["Fix A and Fix B"]})


def test_reordering_whitespace_and_scope_normalization_allowed():
    source = {"kind": "pr_review", "same_pr_followups": ["Fix A", "Fix\nB"]}
    check(source, {"kind": "pr_review", "blocking_items": ["Fix B", "Fix A"]})


def test_note_can_move_between_disposition_buckets():
    source = {"kind": "coder_followup", "remaining_item_notes": {"item-1": "Already covered by test_x."}}
    check(source, {"kind": "coder_followup", "addressed_item_notes": {"item-1": "Already covered by test_x."}})


def test_unknown_item_note_can_be_removed_with_authoritative_context():
    source = {
        "kind": "coder_followup",
        "remaining_item_notes": {"item-unknown": "This ID is not in the round context."},
    }
    target = {"kind": "coder_followup", "remaining_item_notes": {}}
    validate_repair_preservation(
        json.dumps(source),
        json.dumps(target),
        unresolved_item_ids=("item-1",),
    )


def test_disputed_item_and_evidence_cannot_be_reclassified_or_dropped():
    source = {
        "kind": "coder_followup",
        "disputed_items": ["item-1"],
        "dispute_evidence": {"item-1": "Official docs confirm the current price."},
    }
    check(source, source)
    with pytest.raises(AgentLoopError, match="disputed_items"):
        check(source, {
            "kind": "coder_followup",
            "addressed_items": ["item-1"],
        })
    with pytest.raises(AgentLoopError, match="dispute_evidence"):
        check(source, {
            "kind": "coder_followup",
            "disputed_items": ["item-1"],
            "dispute_evidence": {"item-1": "The price is correct."},
        })


def test_unknown_disputed_item_and_evidence_can_be_removed():
    source = {
        "kind": "coder_followup",
        "disputed_items": ["item-unknown"],
        "dispute_evidence": {"item-unknown": "This ID is not in the round context."},
    }
    target = {"kind": "coder_followup", "disputed_items": [], "dispute_evidence": {}}
    validate_repair_preservation(
        json.dumps(source),
        json.dumps(target),
        unresolved_item_ids=("item-1",),
    )


def test_human_requirement_disposition_and_evidence_cannot_change():
    blocked = {
        "requirement_id": "Requirement 1",
        "disposition": "blocked",
        "evidence": "The requested compatibility test is still missing.",
    }
    source = {"kind": "coder_followup", "human_requirement_dispositions": [blocked]}
    check(source, source)
    with pytest.raises(AgentLoopError, match="human_requirement_dispositions"):
        check(source, {
            "kind": "coder_followup",
            "human_requirement_dispositions": [{
                "requirement_id": "Requirement 1",
                "disposition": "addressed",
                "evidence": blocked["evidence"],
            }],
        })
    with pytest.raises(AgentLoopError, match="human_requirement_dispositions"):
        check(source, {
            "kind": "coder_followup",
            "human_requirement_dispositions": [{
                "requirement_id": "Requirement 1",
                "disposition": "blocked",
                "evidence": "More coverage is needed.",
            }],
        })


def test_requirement_preservation_uses_surfaced_context_and_normalized_labels():
    evidence = "The requested compatibility test is still missing."
    source = {
        "kind": "coder_followup",
        "human_requirement_dispositions": [
            {"requirement_id": "requirement 1", "disposition": "blocked", "evidence": evidence},
            {"requirement_id": "Requirement 2", "disposition": "not-applicable",
             "evidence": "This requirement was fabricated."},
            {"requirement_id": "item-1", "disposition": "not-applicable",
             "evidence": "Reviewer IDs are not signed requirements."},
        ],
    }
    target = {
        "kind": "coder_followup",
        "human_requirement_dispositions": [
            {"requirement_id": "Requirement 1", "disposition": "blocked", "evidence": evidence},
        ],
    }
    validate_repair_preservation(
        json.dumps(source),
        json.dumps(target),
        surfaced_requirement_ids=("Requirement 1",),
    )

    with pytest.raises(AgentLoopError, match="human_requirement_dispositions"):
        validate_repair_preservation(
            json.dumps(source),
            json.dumps({"kind": "coder_followup", "human_requirement_dispositions": []}),
            surfaced_requirement_ids=("Requirement 1",),
        )


def test_fabricated_requirement_can_be_removed_with_authoritative_empty_context():
    source = {
        "kind": "coder_followup",
        "human_requirement_dispositions": [{
            "requirement_id": "item-1",
            "disposition": "not-applicable",
            "evidence": "No signed requirements were surfaced.",
        }],
    }
    validate_repair_preservation(
        json.dumps(source),
        json.dumps({"kind": "coder_followup", "human_requirement_dispositions": []}),
        surfaced_requirement_ids=(),
    )


def test_review_requirement_preservation_uses_kind_specific_context():
    disposition = {
        "requirement_id": "Requirement 1",
        "disposition": "addressed",
        "evidence": "The public API remains unchanged.",
    }
    empty = {"human_requirement_dispositions": []}

    validate_repair_preservation(
        json.dumps({"kind": "pr_review", "human_requirement_dispositions": [disposition]}),
        json.dumps({"kind": "pr_review", **empty}),
        reviewer_requirement_ids=("Requirement 1",),
    )
    validate_repair_preservation(
        json.dumps({"kind": "plan_review", "human_requirement_dispositions": [disposition]}),
        json.dumps({"kind": "plan_review", **empty}),
        reviewer_requirement_ids=(),
    )
    with pytest.raises(AgentLoopError, match="human_requirement_dispositions"):
        validate_repair_preservation(
            json.dumps({"kind": "plan_review", "human_requirement_dispositions": [disposition]}),
            json.dumps({"kind": "plan_review", **empty}),
            reviewer_requirement_ids=("Requirement 1",),
        )


def test_reordered_findings_with_shared_prefix_are_not_combined():
    source = {"kind": "pr_review", "blocking_items": ["Fix A", "Fix A and add test B"]}
    check(source, {"kind": "pr_review", "blocking_items": ["Fix A and add test B", "Fix A"]})


def test_forbidden_fields_and_future_items_can_be_removed():
    source = {"kind": "issue_implementation", "addressed_item_notes": {"item-1": "Invalid field"}}
    check(source, {"kind": "issue_implementation"})
    check({"kind": "pr_review", "future_followups": ["Later work"]}, {"kind": "pr_review"})


def test_unparseable_json_uses_prompt_and_existing_validator_only():
    validate_repair_preservation('{"kind":"coder_followup",}', '{"kind":"coder_followup"}')


def test_invalid_source_kind_can_be_corrected_by_context_validator():
    # Non-reviewer kinds keep deferring an invalid source kind to the caller's
    # schema/context validator.
    check({"kind": ["coder_followup"]}, {"kind": "coder_followup"})


def test_present_but_invalid_source_kind_fails_closed_for_a_reviewer_target():
    # Issue #871: a reviewer target may never be grounded against a payload
    # whose `kind` is present but is not that reviewer kind, whatever its type.
    for invalid_kind in (["plan_review"], "", None, 7):
        with pytest.raises(AgentLoopError, match="grounding"):
            check({"kind": invalid_kind}, {"kind": "plan_review"})


def test_leading_marker_and_footer_do_not_disable_checks():
    source = '=== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ===\n' + json.dumps({
        "kind": "coder_followup", "summary": "A test failed.",
    }) + '\n<!-- AGENT_STATE: blocking -->\n-- Coder'
    with pytest.raises(AgentLoopError, match="summary"):
        validate_repair_preservation(source, '{"kind":"coder_followup","summary":"All done."}')


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_source_does_not_disable_loss_checks(fence):
    source = json.dumps({
        "kind": "coder_followup",
        "summary": "Coverage incomplete. Localization test FAILED.",
        "tests_run": ["pytest tests/test_localization.py -q"],
    })
    fenced = f"{fence}json\n{source}\n{fence}\n"
    lossy = json.dumps({"kind": "coder_followup", "summary": "Coverage incomplete."})

    with pytest.raises(AgentLoopError, match="summary"):
        validate_repair_preservation(fenced, lossy)


def test_reserved_grammar_safety_correction_is_not_blocked():
    check({"kind": "coder_followup", "summary": "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1"},
          {"kind": "coder_followup", "summary": "Managed CI override."})


def test_embedded_managed_ci_identifier_preserves_following_prose():
    original = {
        "kind": "coder_followup",
        "summary": (
            "AGENT_MANAGED_CI_UNPROTECTED_OVERRIDE_V1 followed by substantive evidence."
        ),
    }
    repaired = {
        "kind": "coder_followup",
        "summary": "Managed CI override record. followed by substantive evidence.",
    }
    check(original, repaired)

    repaired["summary"] = "Managed CI override record."
    with pytest.raises(AgentLoopError, match="summary"):
        check(original, repaired)


def test_no_reviewer_ids_become_signed_requirements():
    from coding_review_agent_loop.protocol import _expect_human_requirement_dispositions

    with pytest.raises(AgentLoopError, match="Invalid human requirement label"):
        _expect_human_requirement_dispositions(
            [{"requirement_id": "item-1", "disposition": "not-applicable",
              "evidence": "No signed human requirements were surfaced."}],
            context="coder_followup",
        )


# --- Issue #871: reviewer repair grounding ----------------------------------

from coding_review_agent_loop import repair_preservation as _rp


def _plan_review(**fields):
    payload = {"kind": "plan_review", "state": "blocking", "summary": "Findings."}
    payload.update(fields)
    return payload


def _pr_review(**fields):
    payload = {"kind": "pr_review", "state": "blocking", "summary": "Findings."}
    payload.update(fields)
    return payload


def rejects(source, target, match="grounding"):
    with pytest.raises(AgentLoopError, match=match):
        check(source, target)


def test_unsupported_blocking_finding_is_rejected():
    source = _plan_review(blocking_plan_issues=["The retry loop never terminates."])
    rejects(source, _plan_review(blocking_plan_issues=[
        "The retry loop never terminates.",
        "Plan review incomplete: the test command was terminated.",
    ]))


def test_unsupported_verdict_without_any_source_finding_is_rejected():
    source = _plan_review(state="approved", summary="Plan is sound.", blocking_plan_issues=[])
    rejects(source, _plan_review(
        state="blocking",
        summary="Plan is sound.",
        blocking_plan_issues=["Plan review incomplete: the test command was terminated."],
    ))


def test_synthesized_approval_is_rejected_in_both_directions():
    rejects(
        _pr_review(state="blocking", blocking_items=["The pool is never closed."]),
        _pr_review(state="approved", summary="Findings.", blocking_items=[]),
    )
    rejects(
        _pr_review(state="unknown", summary="Findings."),
        _pr_review(state="approved", summary="Findings."),
    )


def test_approval_by_demotion_into_the_future_bucket_is_rejected():
    source = _pr_review(state="approved", blocking_items=["The pool is never closed."])
    rejects(source, _pr_review(
        state="approved",
        blocking_items=[],
        future_followups=["The pool is never closed."],
    ))


def test_approval_with_an_active_source_disposition_is_rejected():
    source = _pr_review(
        state="approved",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "same-pr", "note": "Open."}],
    )
    rejects(source, _pr_review(state="approved", prior_item_dispositions=[]))


def test_synthesized_future_followup_is_rejected():
    source = _plan_review(blocking_plan_issues=["The retry loop never terminates."])
    rejects(source, _plan_review(
        blocking_plan_issues=["The retry loop never terminates."],
        future_followups=["Consider adding a metrics dashboard."],
    ))


def test_repaired_finding_count_may_not_exceed_the_source_total():
    source = _pr_review(blocking_items=["Fix the leak in the pool handler."])
    rejects(source, _pr_review(blocking_items=[
        "Fix the leak in the pool handler.",
        "Fix the leak in the pool handler.",
    ]))


def test_two_repaired_findings_may_not_share_one_source_finding():
    source = _pr_review(blocking_items=[
        "Fix the leak in the pool handler.",
        "Close the socket on error.",
    ])
    rejects(source, _pr_review(blocking_items=[
        "Fix the leak in the pool handler.",
        "Fix the leak.",
    ]))


@pytest.mark.parametrize("builder,bucket", [(_plan_review, "blocking_plan_issues"),
                                            (_pr_review, "blocking_items")])
def test_deleted_negation_inverts_a_matched_finding(builder, bucket):
    source = builder(**{bucket: ["This path is not exploitable."]})
    rejects(source, builder(**{bucket: ["This path is exploitable."]}))


@pytest.mark.parametrize("apostrophe", ["'", "’"])
def test_deleted_contracted_negation_inverts_a_matched_finding(apostrophe):
    source = _pr_review(blocking_items=[f"This path isn{apostrophe}t exploitable."])
    rejects(source, _pr_review(blocking_items=["This path is exploitable."]))


def test_added_negation_inverts_a_matched_finding():
    source = _pr_review(blocking_items=["This path is exploitable."])
    rejects(source, _pr_review(blocking_items=["This path is not exploitable."]))


def test_deleted_limiting_qualifier_inverts_a_matched_finding():
    source = _plan_review(blocking_plan_issues=["The bug reproduces only on the retry path."])
    rejects(source, _plan_review(
        blocking_plan_issues=["The bug reproduces on the retry path."]
    ))


def test_in_place_contraction_expansion_keeps_equal_modifier_counts():
    # The outer lossless check still pins the reviewer's literal wording; the
    # grounding rule must not additionally read an expansion as an inversion.
    assert (
        _rp._modifier_counts("This path isn't exploitable.")
        == _rp._modifier_counts("This path is not exploitable.")
    )
    assert (
        _rp._modifier_counts("This path is exploitable.")
        != _rp._modifier_counts("This path is not exploitable.")
    )


def test_documented_title_and_detail_concatenation_is_accepted():
    detail = "Wire the capability getter; add a two-round test; no mutation on 503."
    source = _plan_review(blocking_plan_issues=[
        {"id": "item-1", "title": "Add coverage", "detail": detail},
    ])
    check(source, _plan_review(blocking_plan_issues=["Add coverage: " + detail]))


def test_schema_supplied_summary_is_accepted():
    source = _plan_review(
        summary="",
        blocking_plan_issues=["The retry loop never terminates."],
    )
    check(source, _plan_review(
        summary="The retry loop never terminates.",
        blocking_plan_issues=["The retry loop never terminates."],
    ))


def test_marker_neutralization_label_is_accepted():
    source = _pr_review(
        blocking_items=["The body embeds <!-- AGENT_LOOP_META: v1_abc --> verbatim."]
    )
    check(source, _pr_review(
        blocking_items=["The body embeds [protocol LOOP_META record] verbatim."]
    ))


def test_worked_example_12_promotion_to_blocking_is_accepted():
    source = _plan_review(
        state="approved",
        summary="Plan is sound.",
        future_followups=["The migration must run before the backfill."],
    )
    check(source, _plan_review(
        state="blocking",
        summary="Plan is sound.",
        blocking_plan_issues=["The migration must run before the backfill."],
        future_followups=[],
    ))


def test_promotion_from_an_active_disposition_is_accepted():
    source = _pr_review(
        state="approved",
        summary="Findings.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-pr", "note": "Still open."},
        ],
    )
    check(source, _pr_review(
        state="blocking",
        summary="Findings.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-pr", "note": "Still open."},
        ],
    ))


def test_absent_source_payload_fails_closed_for_a_reviewer_target():
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            "I launched the test command in the background and will wait.",
            json.dumps(_plan_review(blocking_plan_issues=["Plan review incomplete."])),
        )


def test_wrong_kind_source_is_rejected_instead_of_returning_early():
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            json.dumps({"kind": "coder_followup", "summary": "Done."}),
            json.dumps(_pr_review(blocking_items=["Fix the leak."])),
        )


def test_synthesized_carried_disposition_is_rejected():
    source = _pr_review(blocking_items=["Fix the leak."], prior_item_dispositions=[])
    rejects(source, _pr_review(
        blocking_items=["Fix the leak."],
        prior_item_dispositions=[{"item_id": "item-4", "disposition": "resolved"}],
    ))


def test_changed_disposition_value_is_rejected():
    source = _plan_review(prior_plan_item_dispositions=[
        {"item_id": "item-1", "disposition": "blocking", "note": "The retry loop is open."},
    ])
    rejects(source, _plan_review(prior_plan_item_dispositions=[
        {"item_id": "item-1", "disposition": "resolved", "note": "The retry loop is open."},
    ]))


@pytest.mark.parametrize("note", [
    "This is still open in the revised plan.",
    "This is not already covered by the revised plan.",
    "not handled by the current plan",
])
def test_unjustified_resolution_is_rejected(note):
    source = _plan_review(prior_plan_item_dispositions=[
        {"item_id": "item-1", "disposition": "blocking", "note": note},
    ])
    rejects(source, _plan_review(prior_plan_item_dispositions=[
        {"item_id": "item-1", "disposition": "resolved", "note": note},
    ]))


def test_context_completed_id_written_as_resolved_is_rejected():
    source = _pr_review(blocking_items=["Fix the leak."], prior_item_dispositions=[])
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            json.dumps(source),
            json.dumps(_pr_review(
                blocking_items=["Fix the leak."],
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            )),
            allowed_prior_item_ids=("item-2",),
        )


def test_context_completed_id_as_active_disposition_is_accepted():
    source = _pr_review(blocking_items=["Fix the leak."], prior_item_dispositions=[])
    validate_repair_preservation(
        json.dumps(source),
        json.dumps(_pr_review(
            blocking_items=["Fix the leak."],
            prior_item_dispositions=[{"item_id": "item-2", "disposition": "blocking"}],
        )),
        allowed_prior_item_ids=("item-2",),
    )


def test_disposition_note_dropping_a_modifier_is_rejected():
    source = _pr_review(prior_item_dispositions=[
        {"item_id": "item-1", "disposition": "same-pr", "note": "Only the retry path is affected."},
    ])
    rejects(source, _pr_review(prior_item_dispositions=[
        {"item_id": "item-1", "disposition": "same-pr", "note": "The retry path is affected."},
    ]))


def test_authorized_disposition_normalizations_are_accepted():
    # Enum alias normalization plus a justified active-to-resolved change.
    source = _plan_review(
        state="blocking",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "still blocking", "note": "The retry loop is open."},
            {"item_id": "item-2", "disposition": "same-plan",
             "note": "The revised plan already covers this."},
            {"item_id": "item-3", "disposition": "future", "note": "Deferred work."},
        ],
    )
    check(source, _plan_review(
        state="blocking",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "blocking", "note": "The retry loop is open."},
            {"item_id": "item-2", "disposition": "resolved",
             "note": "The revised plan already covers this."},
            {"item_id": "item-3", "disposition": "blocking", "note": "Deferred work."},
        ],
    ))


def test_deterministic_unknown_id_removal_stays_allowed():
    source = _pr_review(prior_item_dispositions=[
        {"item_id": "item-9", "disposition": "resolved", "note": "Unknown carried ID."},
    ])
    check(source, _pr_review(prior_item_dispositions=[]))


def test_pinned_grounding_vocabulary_cannot_drift():
    assert _rp.GROUNDING_STOP_WORDS == frozenset({
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
        "has", "have", "in", "into", "is", "it", "its", "no", "not", "of", "on",
        "or", "that", "the", "their", "this", "to", "was", "were", "will", "with",
    })
    assert _rp.CONTRACTION_NORMALIZATIONS == (
        ("can't", "cannot"),
        ("cannot", "cannot"),
        ("won't", "will not"),
        ("shan't", "shall not"),
    )
    assert _rp.SEMANTIC_MODIFIERS == frozenset({
        "no", "not", "never", "none", "neither", "nor", "cannot", "without",
        "unless", "except", "only", "always", "must", "should", "may", "optional",
        "required", "all", "any", "every", "some", "most", "least", "more", "less",
        "fewer", "before", "after", "until",
    })
    assert _rp.COVERAGE_PHRASES == (
        "already covered",
        "is covered by",
        "covers this",
        "addressed by the current plan",
        "addressed by the current pr",
        "handled by the current plan",
        "handled by the current pr",
    )
    assert _rp.NEGATION_MARKERS == (
        "not", "never", "nor", "cannot", "without", "fails to", "yet to be",
        "rather than",
    )
    # The still-open list is generated from the coverage list, so every coverage
    # phrase automatically carries its single-word negated counterparts.
    assert _rp.STILL_OPEN_PHRASES == frozenset({
        "still open", "still missing", "still blocking", "remains open",
        "remains unresolved",
    }) | {
        f"{marker} {phrase}"
        for marker in _rp.NEGATION_MARKERS if " " not in marker
        for phrase in _rp.COVERAGE_PHRASES
    }
    assert _rp.REVIEW_KIND_UNIQUE_FIELDS == {
        "plan_review": frozenset({
            "blocking_plan_issues", "same_plan_followups", "prior_plan_item_dispositions",
        }),
        "pr_review": frozenset({
            "blocking_items", "same_pr_followups", "prior_item_dispositions",
        }),
    }
    # Derived exempt sets stay tied to their single source of truth.
    assert "blocking_plan_issues" in _rp.REVIEW_SCHEMA_VOCABULARY
    assert {"protocol", "record"} <= _rp.NEUTRALIZATION_LABEL_TOKENS
    from coding_review_agent_loop.protocol_markers import RESERVED_MARKER_REGISTRY
    expected_labels = set()
    for definition in RESERVED_MARKER_REGISTRY:
        expected_labels.update(
            token for token in definition.safe_label.casefold().replace("[", " ")
            .replace("]", " ").replace("_", " ").replace("-", " ").split() if token
        )
    assert expected_labels <= _rp.NEUTRALIZATION_LABEL_TOKENS


def test_coverage_predicate_is_negation_safe():
    assert _rp.coverage_predicate("The revised plan already covers this.")
    assert not _rp.coverage_predicate("This is not already covered by the revised plan.")
    assert not _rp.coverage_predicate("not handled by the current plan")
    assert not _rp.coverage_predicate("This isn't already covered.")
    assert not _rp.coverage_predicate("The plan already covers this, but it is still open.")


def test_context_completed_active_id_cannot_ground_a_blocking_verdict():
    # Issue #871 round 1, item-1: an ID completed from the repair context is
    # supplied by the orchestrator, not authored by the reviewer, so it is not
    # evidence of open work and may never manufacture a blocking verdict from a
    # source that carries no blocking state, finding, or active disposition.
    source = _pr_review(
        state="approved",
        summary="The diff is correct.",
        blocking_items=[],
        prior_item_dispositions=[],
    )
    with pytest.raises(AgentLoopError, match="state: blocking"):
        validate_repair_preservation(
            json.dumps(source),
            json.dumps(_pr_review(
                state="blocking",
                summary="The diff is correct.",
                blocking_items=[],
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "blocking"}],
            )),
            allowed_prior_item_ids=("item-2",),
        )


def test_future_to_active_promotion_cannot_ground_a_blocking_verdict():
    # The source `future` disposition is not open current-scope work, and the
    # schema-mandated re-statement is authorized only because the target is
    # blocking, so it can never be that blocking state's own support.
    source = _plan_review(
        state="approved",
        summary="Plan is sound.",
        blocking_plan_issues=[],
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "future", "note": "Deferred work."},
        ],
    )
    with pytest.raises(AgentLoopError, match="state: blocking"):
        check(source, _plan_review(
            state="blocking",
            summary="Plan is sound.",
            blocking_plan_issues=[],
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "blocking", "note": "Deferred work."},
            ],
        ))


def test_preserved_active_source_disposition_still_grounds_a_blocking_verdict():
    source = _pr_review(
        state="approved",
        summary="The diff is correct.",
        blocking_items=[],
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "blocking", "note": "The leak is open."},
        ],
    )
    validate_repair_preservation(
        json.dumps(source),
        json.dumps(_pr_review(
            state="blocking",
            summary="The diff is correct.",
            blocking_items=[],
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "blocking", "note": "The leak is open."},
            ],
        )),
        allowed_prior_item_ids=("item-1",),
    )


@pytest.mark.parametrize(
    ("builder", "bucket", "findings_key"),
    [
        (_pr_review, "blocking_items", "blocking_items"),
        (_plan_review, "blocking_plan_issues", "blocking_plan_issues"),
    ],
)
def test_empty_source_finding_cannot_absorb_global_source_prose(
    builder, bucket, findings_key
):
    # Issue #871 round 2, item-3: an empty source finding carries no prose, so
    # it may not act as a wildcard. Whole-source token coverage alone cannot
    # tell a finding apart from the summary or any other global prose, so a
    # repair that promotes the source summary into a finding must be rejected
    # even though every one of its tokens appears somewhere in the source.
    source = builder(
        state="approved",
        summary="Close the socket leak",
        **{findings_key: [{}]},
    )
    rejects(source, builder(
        state="blocking",
        summary="Close the socket leak",
        **{bucket: ["Close the socket leak"]},
    ))


@pytest.mark.parametrize("builder", [_pr_review, _plan_review])
def test_empty_source_finding_cannot_absorb_another_source_finding_text(builder):
    # The same wildcard route must not let one source finding's text be
    # duplicated into a second target finding through an empty source entry.
    bucket = "blocking_items" if builder is _pr_review else "blocking_plan_issues"
    source = builder(
        state="blocking",
        summary="Findings.",
        **{bucket: ["The socket leak is unbounded.", {}]},
    )
    rejects(source, builder(
        state="blocking",
        summary="Findings.",
        **{bucket: [
            "The socket leak is unbounded.",
            "The socket leak is unbounded.",
        ]},
    ))


def test_marker_only_source_finding_still_accepts_its_neutralization():
    # The empty-candidate branch stays open for the documented case it exists
    # for: a source finding that is nothing but a reserved marker, replaced by a
    # neutralization label whose tokens are exempt.
    source = _pr_review(blocking_items=["<!-- AGENT_LOOP_META: v1_abc -->"])
    check(source, _pr_review(blocking_items=["[protocol LOOP_META record]"]))


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_summary_cannot_become_a_blocking_finding_without_source_findings(builder, bucket):
    # Issue #871 round 3, item-4: when the payload declares no finding at all,
    # the freeform fallback may not split the serialized payload into candidate
    # prose. Otherwise an approved source whose summary reads like a defect can
    # be repaired into a blocking finding, and that finding then grounds the
    # inverted verdict.
    source = builder(
        state="approved",
        summary="Close the socket leak",
        **{bucket: []},
    )
    rejects(source, builder(
        state="blocking",
        summary="Close the socket leak",
        **{bucket: ["Close the socket leak"]},
    ))


@pytest.mark.parametrize("builder", [_pr_review, _plan_review])
def test_no_payload_field_can_become_a_finding_without_source_findings(builder):
    # The same hole would also let a disposition note, or any other payload
    # field, be promoted into a finding.
    bucket = "blocking_items" if builder is _pr_review else "blocking_plan_issues"
    field = "prior_item_dispositions" if builder is _pr_review else "prior_plan_item_dispositions"
    source = builder(
        state="blocking",
        summary="Findings.",
        **{
            bucket: [],
            field: [
                {"item_id": "item-1", "disposition": "blocking",
                 "note": "The retry loop never terminates."},
            ],
        },
    )
    rejects(source, builder(
        state="blocking",
        summary="Findings.",
        **{
            bucket: ["The retry loop never terminates."],
            field: [
                {"item_id": "item-1", "disposition": "blocking",
                 "note": "The retry loop never terminates."},
            ],
        },
    ))


def test_freeform_prose_outside_the_payload_still_supports_a_finding():
    # The fallback the approved plan describes stays available for its real
    # case: reviewer prose that sits outside the recovered JSON object.
    source = (
        json.dumps(_pr_review(state="blocking", summary="Findings.", blocking_items=[]))
        + "\n- The retry loop never terminates on a truncated response.\n"
        + "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    validate_repair_preservation(
        source,
        json.dumps(_pr_review(
            state="blocking",
            summary="Findings.",
            blocking_items=["The retry loop never terminates on a truncated response."],
        )),
    )


def _trailing_prose_source(builder, bucket, prose):
    """A payload declaring no finding, followed by reviewer prose."""
    return (
        json.dumps(builder(state="blocking", summary="Findings.", **{bucket: []}))
        + f"\n- {prose}\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
@pytest.mark.parametrize(
    ("source_prose", "inverted"),
    [
        ("This path is not exploitable.", "This path is exploitable."),
        ("This path isn't exploitable.", "This path is exploitable."),
        ("This path isn’t exploitable.", "This path is exploitable."),
        ("The leak happens only on the retry path.",
         "The leak happens on the retry path."),
        ("The leak happens on the retry path.",
         "The leak happens only on the retry path."),
    ],
)
def test_freeform_fallback_candidate_rejects_a_modifier_change(
    builder, bucket, source_prose, inverted
):
    # Issue #871 round 4, item-5: a trailing-prose candidate is still a matched
    # source/target pair, so modifier-count equality applies to it exactly as it
    # does to a declared source finding. Subset coverage alone cannot see the
    # deletion, because `not` is exempt and `only` simply disappears.
    source = _trailing_prose_source(builder, bucket, source_prose)
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            source,
            json.dumps(builder(
                state="blocking", summary="Findings.", **{bucket: [inverted]},
            )),
        )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_freeform_fallback_candidate_accepts_equal_modifier_counts(builder, bucket):
    # The fallback still works for a faithful recovery, including an in-place
    # contraction expansion, which normalization makes identical on both sides.
    source = _trailing_prose_source(
        builder, bucket, "This path isn't exploitable without the retry loop."
    )
    validate_repair_preservation(
        source,
        json.dumps(builder(
            state="blocking",
            summary="Findings.",
            **{bucket: ["This path is not exploitable without the retry loop."]},
        )),
    )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_one_trailing_prose_finding_cannot_support_two_repaired_findings(builder, bucket):
    # Issue #871 round 5, item-6: the freeform candidate list must represent one
    # concern exactly once. Emitting both a line and the paragraph containing it
    # gave a single trailing statement two equivalent candidates, so repair could
    # duplicate it into two findings, match each copy injectively, and clear the
    # inflated cardinality ceiling.
    source = _trailing_prose_source(
        builder, bucket, "The retry loop never terminates on a truncated response."
    )
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            source,
            json.dumps(builder(
                state="blocking",
                summary="Findings.",
                **{bucket: [
                    "The retry loop never terminates on a truncated response.",
                    "The retry loop never terminates on a truncated response.",
                ]},
            )),
        )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_a_wrapped_trailing_paragraph_is_one_candidate(builder, bucket):
    # A multi-line paragraph is wrapped prose, so it joins into a single
    # candidate and cannot support two findings either.
    source = (
        json.dumps(builder(state="blocking", summary="Findings.", **{bucket: []}))
        + "\nThe retry loop never terminates\non a truncated response.\n"
        + "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    validate_repair_preservation(
        source,
        json.dumps(builder(
            state="blocking",
            summary="Findings.",
            **{bucket: ["The retry loop never terminates on a truncated response."]},
        )),
    )
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            source,
            json.dumps(builder(
                state="blocking",
                summary="Findings.",
                **{bucket: [
                    "The retry loop never terminates on a truncated response.",
                    "The retry loop never terminates on a truncated response.",
                ]},
            )),
        )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_protocol_footer_and_signature_are_not_finding_candidates(builder, bucket):
    # The footer and signature are tool-owned structural records, so they may
    # not stand in as reviewer prose for a repaired finding.
    source = (
        json.dumps(builder(state="blocking", summary="Findings.", **{bucket: []}))
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    with pytest.raises(AgentLoopError, match="grounding"):
        validate_repair_preservation(
            source,
            json.dumps(builder(
                state="blocking", summary="Findings.", **{bucket: ["OpenAI Codex"]},
            )),
        )


@pytest.mark.parametrize(
    ("builder", "bucket"),
    [(_pr_review, "blocking_items"), (_plan_review, "blocking_plan_issues")],
)
def test_two_distinct_trailing_bullets_support_two_findings(builder, bucket):
    # Two genuinely distinct bulleted concerns still yield two candidates.
    source = (
        json.dumps(builder(state="blocking", summary="Findings.", **{bucket: []}))
        + "\n- The retry loop never terminates on a truncated response.\n"
        + "- The socket leak is unbounded under backpressure.\n"
        + "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    validate_repair_preservation(
        source,
        json.dumps(builder(
            state="blocking",
            summary="Findings.",
            **{bucket: [
                "The retry loop never terminates on a truncated response.",
                "The socket leak is unbounded under backpressure.",
            ]},
        )),
    )
