import json
from copy import deepcopy

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.repair import _build_repair_prompt
from coding_review_agent_loop.repair_preservation import validate_repair_preservation
from agent_loop_helpers import structured_v1_plan_state


def check(source, target):
    validate_repair_preservation(json.dumps(source), json.dumps(target))


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
    check({"kind": ["plan_review"]}, {"kind": "plan_review"})


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
