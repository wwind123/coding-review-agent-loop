import json

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.repair import _build_repair_prompt
from coding_review_agent_loop.repair_preservation import validate_repair_preservation


def check(source, target):
    validate_repair_preservation(json.dumps(source), json.dumps(target))


def test_prompt_demands_lossless_repair_and_separate_ledgers():
    prompt = _build_repair_prompt("malformed", expected_kind="coder_followup",
                                  surfaced_requirement_ids=())
    assert "This is format repair, not summarization" in prompt
    assert "two-round claim test and assert no mutation on 503" in prompt
    assert "including no not-applicable rows" in prompt
    assert "Retain failure, timeout, skipped-test" in prompt


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


def test_no_reviewer_ids_become_signed_requirements():
    from coding_review_agent_loop.protocol import _expect_human_requirement_dispositions

    with pytest.raises(AgentLoopError, match="Invalid human requirement label"):
        _expect_human_requirement_dispositions(
            [{"requirement_id": "item-1", "disposition": "not-applicable",
              "evidence": "No signed human requirements were surfaced."}],
            context="coder_followup",
        )
