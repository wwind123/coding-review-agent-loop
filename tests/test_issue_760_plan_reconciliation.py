"""Scenario coverage for approved-plan versus PR-review reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import PullRequestMetadata
from coding_review_agent_loop.prompts import (
    approved_plan_reconciliation_guidance,
    build_followup_prompt,
    build_review_prompt,
    format_approved_plan_context,
)
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import ApprovedPlanContext, make_approved_plan_context
from coding_review_agent_loop.unresolved_items import (
    _apply_dispute_evidence,
    _is_disputed_item,
    _validate_coder_followup_response,
    apply_item_dispositions,
)
from coding_review_agent_loop.repair import _build_repair_prompt


FIXTURE = Path(__file__).parent / "fixtures" / "issue_760_plan_reconciliation.json"


def _scenarios() -> list[dict]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["scenarios"]


def _plan_context(raw: dict | None) -> ApprovedPlanContext | None:
    if raw is None:
        return None
    availability = raw["availability"]
    if availability == "available":
        return make_approved_plan_context(
            raw["canonical_text"],
            source_locator=raw.get("source_locator"),
        )
    return ApprovedPlanContext(
        canonical_text=None,
        plan_hash=raw.get("plan_hash"),
        plan_subject=raw.get("plan_subject"),
        source_locator=raw.get("source_locator"),
        availability=availability,
        diagnostic=raw.get("diagnostic"),
    )


def _review_prompt(tmp_path: Path, context: ApprovedPlanContext | None, *, compact: bool) -> str:
    config = make_config(tmp_path, reviewer=("codex",))
    metadata = PullRequestMetadata(
        number=760,
        repo=config.repo,
        title="Plan reconciliation",
        head_branch="feature/760",
        base_branch="main",
        head_sha="a" * 40,
        url="https://github.com/OWNER/REPO/pull/760",
        body="Implementation PR",
    )
    return build_review_prompt(
        760,
        1,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        compact_context=compact,
        approved_plan_context=context,
    )


def _coder_response(payload: dict) -> str:
    body = {
        "schema_version": 1,
        "kind": "coder_followup",
        "state": payload.get("state", "blocking"),
        "summary": "Applied the scenario outcome.",
        "addressed_items": payload.get("addressed_items", []),
        "remaining_items": payload.get("remaining_items", []),
        "disputed_items": payload.get("disputed_items", []),
        "addressed_item_notes": payload.get("addressed_item_notes", {}),
        "remaining_item_notes": payload.get("remaining_item_notes", {}),
        "dispute_evidence": payload.get("dispute_evidence", {}),
        "human_requirements": {
            "addressed_ids": [],
            "checked_discussion_directly": False,
        },
        "human_requirement_dispositions": [],
    }
    return json.dumps(body) + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"


def test_fixture_contains_the_six_required_examples_and_context_guardrails():
    names = {scenario["name"] for scenario in _scenarios()}
    assert {
        "stable_id_discretionary_positional_labels",
        "stable_id_evidence_backed_compatibility_defect",
        "stable_id_implementation_noncompliance",
        "legacy_notice_discretionary_removal",
        "legacy_recovery_failure_is_correctable",
        "legacy_unavailable_notice_validation_bug",
    } <= names
    assert {
        "budget_omitted_plan_text",
        "mismatched_plan_identity",
        "direct_pr_without_plan",
    } <= names


def test_full_and_compact_review_prompts_share_three_way_reconciliation_guidance(tmp_path):
    context = _plan_context(_scenarios()[0]["plan_context"])
    full = _review_prompt(tmp_path, context, compact=False)
    compact = _review_prompt(tmp_path, context, compact=True)

    expected = (
        "Implementation noncompliance or an ordinary defect",
        "evidence-backed correctness, security, compatibility, or test defect",
        "discretionary scope or policy request incompatible",
        "name the approved decision, concrete evidence, and proposed change",
        "Plan conformance never defeats category 1 or 2",
    )
    for prompt in (full, compact):
        for phrase in expected:
            assert phrase in prompt
        assert prompt.count("Approved-plan reconciliation guidance") == 1
        assert "No approved plan is bound to this PR" not in prompt


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda scenario: scenario["name"])
def test_context_states_never_invent_a_plan_decision(tmp_path, scenario):
    context = _plan_context(scenario["plan_context"])
    guidance = approved_plan_reconciliation_guidance(context)
    prompt = _review_prompt(tmp_path, context, compact=False)

    if scenario["plan_context"] is not None and scenario["plan_context"]["availability"] == "available":
        assert "For each concern against a verified canonical plan" in guidance
    else:
        assert "Do not invent" in guidance
        assert (
            "No verified canonical approved-plan decision is available" in guidance
            or "No approved plan is bound" in guidance
            or "canonical approved-plan text is omitted for provider budget" in guidance
        )
        assert "For each concern against a verified canonical plan" not in guidance
        assert "Do not invent" in prompt


def test_budget_omitted_context_preserves_identity_and_has_no_enforceable_decision():
    scenario = next(item for item in _scenarios() if item["name"] == "budget_omitted_plan_text")
    context = _plan_context(scenario["plan_context"])
    rendered = format_approved_plan_context(context)
    guidance = approved_plan_reconciliation_guidance(context)

    assert "Availability: omitted" in rendered
    assert "fd8029a78e94aba0" in rendered
    assert "issue #760 approved-plan round" in rendered
    assert "Canonical approved plan text omitted" in rendered
    assert "no enforceable decision" in guidance
    assert "ordinary correctness, security, compatibility, and test defects remain reviewable" in guidance
    with pytest.raises(AgentLoopError, match="cannot fit the final provider prompt limit"):
        format_approved_plan_context(context, max_chars=100)


def test_coder_followup_and_repair_guidance_separate_plan_conflicts_from_fixes(tmp_path):
    context = _plan_context(_scenarios()[0]["plan_context"])
    config = make_config(tmp_path, reviewer=("codex",))
    followup = build_followup_prompt(
        760,
        2,
        "[item-1] Restore positional labels.",
        config,
        approved_plan_context=context,
    )
    repair = _build_repair_prompt(
        "malformed",
        expected_kind="coder_followup",
        unresolved_item_ids=("item-1",),
    )

    for text in (followup, repair):
        normalized = " ".join(text.split())
        assert "mutually incompatible with a verified approved-plan decision" in normalized
        assert "never" in normalized and "remaining_items" in normalized
        assert "evidence-backed correctness, security, compatibility, or test defects" in normalized
        assert (
            "never disputed merely because the implementation followed the plan" in normalized
            or "must never be disputed merely because the implementation followed the plan" in normalized
        )


def test_named_scenarios_use_existing_followup_partition_and_dispute_annotation():
    scenarios = _scenarios()
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Scenario reviewer finding.",
        status="blocking",
    )
    for scenario in scenarios[:6]:
        parsed = _validate_coder_followup_response(
            _coder_response(scenario["coder_followup"]),
            unresolved_items=(item,),
            human_requirements=(),
        )
        expected = scenario["expected"]["classification"]
        if expected == "discretionary_plan_conflict":
            assert parsed.disputed_items == ("item-1",)
            evidence = parsed.dispute_evidence["item-1"].lower()
            assert "approved decision" in evidence
            assert "incompatible" in evidence
            annotated = _apply_dispute_evidence(
                (item,),
                disputed_items=parsed.disputed_items,
                dispute_evidence=parsed.dispute_evidence,
            )
            assert _is_disputed_item(annotated[0])
            assert parsed.remaining_items == ()
        else:
            assert parsed.addressed_items == ("item-1",)
            remaining, future = apply_item_dispositions(
                (item,),
                {
                    "item-1": [
                        ReviewItemDisposition(
                            item_id="item-1",
                            reviewer="OpenAI Codex",
                            disposition="resolved",
                            note=parsed.addressed_item_notes.get("item-1"),
                        )
                    ]
                },
                same_status="same-pr",
                retain_future=False,
            )
            assert remaining == []
            assert future == []


def test_dispute_evidence_is_preserved_as_the_existing_plan_conflict_channel():
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Restore positional labels.",
        status="blocking",
    )
    response = _coder_response(
        {
            "disputed_items": ["item-1"],
            "dispute_evidence": {"item-1": "The reviewer is wrong."},
        }
    )
    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=(item,),
        human_requirements=(),
    )
    assert parsed.disputed_items == ("item-1",)
    assert parsed.dispute_evidence["item-1"] == "The reviewer is wrong."
