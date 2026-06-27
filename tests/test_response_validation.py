from agent_loop_helpers import *  # noqa: F403
from coding_review_agent_loop.protocol import StructuredCoderFollowup
from coding_review_agent_loop.unresolved_items import (
    CODER_DISPUTE_NOTE_PREFIX,
    _apply_dispute_evidence,
    _is_disputed_item,
)


def test_structured_plan_review_preserves_human_requirements_resolution_marker():
    review = structured_plan_review(human_requirements_resolved=True)

    assert _extract_structured_plan_review_payload(review) is not None
    parsed = parse_plan_review(review, reviewer="OpenAI Codex")
    public = _render_public_plan_review_comment(
        parsed,
        reviewer="OpenAI Codex",
        prior_items=(),
        dispositions=(),
        human_requirements_resolved_flag=True,
    )

    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in public
    assert public.index("<!-- HUMAN_REQUIREMENTS_RESOLVED -->") < public.index("<!-- AGENT_PLAN_STATE: approved -->")

def test_validate_review_response_accepts_structured_pr_review():
    review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Need one more regression test before merge.",
                "blocking_items": ["Add the mixed-history regression case to the suite."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = _validate_review_response(review, reviewer="OpenAI Codex", unresolved_items=())

    assert parsed.summary == "Need one more regression test before merge."
    assert [item.text for item in parsed.blocking_items] == [
        "Add the mixed-history regression case to the suite."
    ]

def test_validate_coder_followup_response_accepts_structured_item_partition():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Rename the helper.",
            status="same-pr",
        ),
        UnresolvedReviewItem(
            item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
            reviewer="Orchestrator",
            source_round=1,
            text="Ack missing.",
            status="blocking",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the test, helper rename still pending.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=unresolved_items,
        human_requirements=(),
    )

    assert parsed.addressed_items == ("item-1",)
    assert parsed.remaining_items == ("item-2",)

def test_validate_coder_followup_response_rejects_issue_acceptance_criteria_as_human_requirement():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Issue #221 acceptance criteria"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="issue acceptance criteria.*not signed human requirements"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(),
        )

def test_validate_coder_followup_response_rejects_requirement_label_when_none_surfaced():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="no signed human requirements were surfaced"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(),
        )

def test_validate_coder_followup_response_accepts_surfaced_requirement_label():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1"],
        reviewer="OpenAI Codex",
    )

    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Add a regression test.",
                status="blocking",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="maintainer",
                created_at="2026-06-02T12:00:00Z",
                url="https://github.com/OWNER/REPO/pull/1#issuecomment-1",
                body="Add coverage for the rejected label case.",
            ),
        ),
    )

    assert parsed.human_requirements.addressed_ids == ("Requirement 1",)

def test_validate_coder_followup_response_rejects_mixed_valid_and_invalid_requirement_labels():
    response = structured_coder_followup(
        state="blocking",
        addressed_items=["item-1"],
        remaining_items=[],
        human_requirement_ids=["Requirement 1", "Issue #221 acceptance criteria"],
        reviewer="OpenAI Codex",
    )

    with pytest.raises(AgentLoopError, match="issue acceptance criteria.*not signed human requirements"):
        _validate_coder_followup_response(
            response,
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Add a regression test.",
                    status="blocking",
                ),
            ),
            human_requirements=(
                HumanReviewRequirement(
                    source_type="PR comment",
                    author="maintainer",
                    created_at="2026-06-02T12:00:00Z",
                    url="https://github.com/OWNER/REPO/pull/1#issuecomment-1",
                    body="Add coverage for the rejected label case.",
                ),
            ),
        )

@pytest.mark.parametrize(
    ("addressed_items", "remaining_items", "message"),
    [
        (["item-1"], ["item-1"], "listed unresolved reviewer item IDs more than once"),
        (["item-9"], [], "referenced unknown unresolved reviewer item IDs"),
        (["item-1"], [], "did not classify all unresolved reviewer items"),
    ],
)
def test_validate_coder_followup_response_rejects_invalid_structured_item_partition(
    addressed_items,
    remaining_items,
    message,
):
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Rename the helper.",
            status="same-pr",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Status update.",
                "addressed_items": addressed_items,
                "remaining_items": remaining_items,
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match=message):
        _validate_coder_followup_response(
            response,
            unresolved_items=unresolved_items,
            human_requirements=(),
        )

def test_validate_coder_followup_response_rejects_marker_only_markdown():
    with pytest.raises(AgentLoopError, match="Coder response did not use the required structured format"):
        _validate_coder_followup_response(
            "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            unresolved_items=(),
            human_requirements=(),
        )

def test_validate_coder_followup_response_requires_regular_synthetic_human_requirement_item():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-8",
            reviewer="Orchestrator",
            source_round=4,
            text="Reviewers approved without acknowledging signed human requirements.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
            reviewer="Orchestrator",
            source_round=4,
            text="Internal human requirements acknowledgement pseudo-item.",
            status="blocking",
        ),
    )
    response = structured_coder_followup(
        state="approved",
        addressed_items=[],
        remaining_items=[],
        reviewer="Anthropic Claude",
    )

    with pytest.raises(AgentLoopError, match="item-8"):
        _validate_coder_followup_response(
            response,
            unresolved_items=unresolved_items,
            human_requirements=(),
        )

def test_apply_unresolved_item_dispositions_appends_disposition_notes_to_text():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Needs regression coverage before merge.",
            status="blocking",
        ),
    )
    dispositions_by_item = {
        "item-1": [
            parse_unresolved_item_dispositions(
                prior_item_dispositions("[item-1] still blocking: include API error path too"),
                reviewer="Anthropic Claude",
            )[0]
        ]
    }

    updated_items, future_items = _apply_unresolved_item_dispositions(
        unresolved_items, dispositions_by_item
    )

    assert len(updated_items) == 1
    assert future_items == []
    assert updated_items[0].text == (
        "Needs regression coverage before merge.\n\n"
        "Update from Anthropic Claude: include API error path too"
    )
    assert updated_items[0].notes == ("Anthropic Claude: include API error path too",)

@pytest.mark.parametrize(
    ("disposition", "expected_label"),
    [("future", "future follow-up"), ("resolved", "resolved")],
)
def test_collect_prior_compact_summaries_infers_departed_item_label(
    disposition, expected_label
):
    prior_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Preserve the original item detail.",
        status="blocking",
    )
    dispositions_by_item = {
        "item-1": [
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Anthropic Claude",
                disposition=disposition,
                note="Reconciled in the current round.",
            )
        ]
    }

    summaries = _collect_prior_compact_summaries(
        (prior_item,),
        (),
        dispositions_by_item,
    )

    assert summaries[0].startswith(f"[item-1] {expected_label}:")
    assert "Anthropic Claude: " + disposition in summaries[0]

def test_collect_prior_compact_summaries_future_blocked_by_blocking_outcome_infers_resolved():
    """An item with both 'future' and 'blocking' outcomes is labelled 'resolved' (blocking wins)."""
    prior_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Original blocking item.",
        status="blocking",
    )
    dispositions_by_item = {
        "item-1": [
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Reviewer A",
                disposition="future",
            ),
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Reviewer B",
                disposition="blocking",
            ),
        ]
    }

    summaries = _collect_prior_compact_summaries(
        (prior_item,),
        (),
        dispositions_by_item,
    )

    assert summaries[0].startswith("[item-1] resolved:")

@pytest.mark.parametrize("terminator", ["<!-- AGENT_STATE: approved -->", "-- OpenAI Codex"])
def test_parse_non_blocking_followups_stops_at_final_markers(terminator):
    review = f"""
    Looks good.

    ### Non-blocking follow-ups
    - Add cleanup docs.
    {terminator}
    - This is outside the follow-up section.
    """

    followups = parse_non_blocking_followups(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in followups] == [
        ("OpenAI Codex", "Add cleanup docs."),
    ]

def test_parse_non_blocking_followups_returns_empty_without_section():
    review = "LGTM.\n- A normal bullet outside the section.\n<!-- AGENT_STATE: approved -->"

    assert parse_non_blocking_followups(review, reviewer="OpenAI Codex") == []

def test_parse_pr_number_accepts_marker_and_url():
    assert parse_pr_number("opened\n<!-- AGENT_PR: 61 -->") == 61
    assert parse_pr_number("https://github.com/OWNER/REPO/pull/62") == 62
    assert parse_pr_number("no pr here") is None

def test_parse_pr_number_uses_final_marker():
    # When multiple AGENT_PR markers are present, the last one is authoritative.
    assert parse_pr_number("<!-- AGENT_PR: 10 -->\n<!-- AGENT_PR: 20 -->") == 20
    # Same for PR URLs.
    assert (
        parse_pr_number(
            "https://github.com/OWNER/REPO/pull/1 and https://github.com/OWNER/REPO/pull/2"
        )
        == 2
    )
    # Marker takes precedence over URL when both present (marker checked first).
    assert parse_pr_number("https://github.com/OWNER/REPO/pull/5\n<!-- AGENT_PR: 7 -->") == 7

def test_validate_human_requirements_acknowledgement_accepts_multiple_bullet_styles():
    response = f"""Implemented the fix.
{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}
### Human requirements
1. Requirement 1: updated the URL handling.
* Requirement 2: could not satisfy safely without widening scope, so I documented the limit.
<!-- AGENT_STATE: blocking -->
"""

    validate_human_requirements_acknowledgement(
        response,
        surfaced_requirement_ids=("Requirement 1", "Requirement 2"),
        requires_direct_discussion_ack=False,
    )

    parsed = parse_human_requirements_acknowledgement(response)
    assert parsed.addressed_ids == ("Requirement 1", "Requirement 2")

@pytest.mark.parametrize(
    ("response", "surfaced_ids", "requires_direct_discussion_ack", "message"),
    [
        (
            "Implemented.\n### Human requirements\n- Requirement 1: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "missing required signed human requirements marker",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "missing required `### Human requirements` section",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 1: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1", "Requirement 2"),
            False,
            "did not address all surfaced signed human requirement IDs",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 1: handled.\n- Requirement 1: repeated.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "listed signed human requirement IDs more than once",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Requirement 99: handled.\n<!-- AGENT_STATE: blocking -->",
            ("Requirement 1",),
            False,
            "referenced unknown signed human requirement IDs",
        ),
        (
            f"Implemented.\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n- Prompt omitted details.\n<!-- AGENT_STATE: blocking -->",
            (),
            True,
            "must acknowledge that the prompt omitted the detailed signed human requirements",
        ),
    ],
)
def test_validate_human_requirements_acknowledgement_rejects_structural_failures(
    response,
    surfaced_ids,
    requires_direct_discussion_ack,
    message,
):
    with pytest.raises(AgentLoopError, match=message):
        validate_human_requirements_acknowledgement(
            response,
            surfaced_requirement_ids=surfaced_ids,
            requires_direct_discussion_ack=requires_direct_discussion_ack,
        )

def test_validate_human_requirements_acknowledgement_accepts_full_truncation_fallback():
    response = f"""Implemented the fix.
{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}
### Human requirements
- The prompt omitted the detailed signed human requirements, so I {HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK}.
<!-- AGENT_STATE: blocking -->
"""

    validate_human_requirements_acknowledgement(
        response,
        surfaced_requirement_ids=(),
        requires_direct_discussion_ack=True,
    )


def test_validate_coder_followup_response_accepts_disputed_items():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="The pricing constant is wrong.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test.",
            status="blocking",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Disputed item-1 with evidence; fixed item-2.",
                "addressed_items": ["item-2"],
                "remaining_items": [],
                "disputed_items": ["item-1"],
                "dispute_evidence": {
                    "item-1": "Checked Google pricing page: $1.50/1M tokens is correct per official docs."
                },
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = _validate_coder_followup_response(
        response,
        unresolved_items=unresolved_items,
        human_requirements=(),
    )

    assert isinstance(parsed, StructuredCoderFollowup)
    assert parsed.addressed_items == ("item-2",)
    assert parsed.disputed_items == ("item-1",)
    assert "item-1" in parsed.dispute_evidence


def test_validate_coder_followup_response_rejects_missing_item_when_disputed_not_counted():
    unresolved_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Fix the bug.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Rename the variable.",
            status="blocking",
        ),
    )
    response = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Only addressed one item.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "disputed_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(
        AgentLoopError,
        match="did not classify all unresolved reviewer items",
    ):
        _validate_coder_followup_response(
            response,
            unresolved_items=unresolved_items,
            human_requirements=(),
        )


def test_apply_dispute_evidence_adds_note_to_disputed_item():
    items = [
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Antigravity",
            source_round=1,
            text="Pricing constant is wrong.",
            status="blocking",
            notes=(),
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Antigravity",
            source_round=1,
            text="Add a test.",
            status="blocking",
            notes=(),
        ),
    ]

    result = _apply_dispute_evidence(
        items,
        disputed_items=["item-1"],
        dispute_evidence={"item-1": "Official docs confirm $1.50/1M is correct."},
    )

    assert len(result) == 2
    item1 = next(i for i in result if i.item_id == "item-1")
    item2 = next(i for i in result if i.item_id == "item-2")
    assert any(CODER_DISPUTE_NOTE_PREFIX in note for note in item1.notes)
    assert "Official docs confirm $1.50/1M is correct." in item1.notes[0]
    assert not item2.notes


def test_apply_dispute_evidence_is_idempotent():
    evidence = "Official docs confirm $1.50/1M is correct."
    items = [
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Antigravity",
            source_round=1,
            text="Pricing constant is wrong.",
            status="blocking",
            notes=(f"{CODER_DISPUTE_NOTE_PREFIX}: {evidence}",),
        ),
    ]

    result = _apply_dispute_evidence(
        items,
        disputed_items=["item-1"],
        dispute_evidence={"item-1": evidence},
    )

    assert len(result) == 1
    assert len(result[0].notes) == 1


def test_is_disputed_item_detects_dispute_note():
    disputed_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Antigravity",
        source_round=1,
        text="Pricing constant is wrong.",
        status="blocking",
        notes=(f"{CODER_DISPUTE_NOTE_PREFIX}: some evidence",),
    )
    non_disputed_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Antigravity",
        source_round=1,
        text="Add a test.",
        status="blocking",
        notes=("Antigravity: still blocking",),
    )

    assert _is_disputed_item(disputed_item)
    assert not _is_disputed_item(non_disputed_item)
