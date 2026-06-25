"""Protocol tests extracted from test_agent_loop.py (lines 2016–4430).

Tests for parse_agent_state, parse_plan_state, parse_review, parse_plan_review,
structured response parsing and validation, and related protocol functions.
"""
import json

import pytest

from coding_review_agent_loop.cli import (
    AgentLoopError,
    parse_agent_state,
)
from coding_review_agent_loop.orchestrator import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    _review_freeform_summary_text,
    _validate_coder_followup_response,
    _validate_plan_revision_response,
    _validate_review_response,
    _validate_plan_review_response,
)
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    _expect_string_list,
    _extract_structured_coder_followup_payload,
    _extract_structured_plan_review_payload,
    _extract_structured_plan_revision_payload,
    _extract_structured_pr_review_payload,
    normalize_response_file_structured_text,
    parse_approved_followups,
    parse_human_requirements_acknowledgement,
    parse_pr_review,
    parse_plan_item_dispositions,
    parse_plan_review,
    parse_plan_review_items,
    parse_plan_state,
    parse_structured_plan_review,
    parse_structured_pr_review,
    parse_review,
    parse_non_blocking_followups,
    parse_signed_human_requirement_body,
    parse_unresolved_item_dispositions,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    validate_human_requirements_acknowledgement,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_plan_state,
    validate_structured_plan_revision,
)
from coding_review_agent_loop.agents.gemini import PUBLIC_RESPONSE_MARKER
from coding_review_agent_loop.errors import UnknownPriorItemDispositionError
from coding_review_agent_loop.orchestrator import (
    _decode_public_response_json_prefix,
    _failure_category,
    _is_transient_public_response,
    _plan_subject,
    render_canonical_plan_revision,
)
from coding_review_agent_loop.prompts import (
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK,
)

from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    make_config,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_coder_followup,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _no_real_repair():
    """Prevent attempt_repair from calling the real Gemini CLI in all tests."""
    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=None):
        yield


@pytest.fixture(autouse=True)
def _agent_commands_available(monkeypatch):
    """Keep config tests independent of agent CLIs installed on the test host."""
    import coding_review_agent_loop.config as config_module

    real_which = config_module.shutil.which

    def which(command):
        resolved = real_which(command)
        if resolved is not None:
            return resolved
        if command in {"claude", "codex", "gemini", "agy"}:
            return f"/mock/bin/{command}"
        return None

    monkeypatch.setattr(config_module.shutil, "which", which)

def test_parse_agent_state_accepts_html_marker():
    assert parse_agent_state("looks fine\n<!-- AGENT_STATE: approved -->") == "approved"
    assert parse_agent_state("needs work\n<!-- agent_state: BLOCKING -->") == "blocking"


def test_parse_agent_state_uses_last_marker_as_authoritative():
    text = """
    Quoting earlier review: <!-- AGENT_STATE: blocking -->

    Final decision:
    <!-- AGENT_STATE: approved -->
    """
    assert parse_agent_state(text) == "approved"


def test_parse_agent_state_requires_marker():
    with pytest.raises(AgentLoopError):
        parse_agent_state("LGTM")


def test_parse_plan_state_uses_last_marker_as_authoritative():
    text = """
    Quoting earlier plan review: <!-- AGENT_PLAN_STATE: blocking -->

    Final decision:
    <!-- AGENT_PLAN_STATE: approved -->
    """
    assert parse_plan_state(text) == "approved"


def test_parse_plan_state_requires_plan_marker():
    with pytest.raises(AgentLoopError):
        parse_plan_state("<!-- AGENT_STATE: approved -->")


def test_parse_signed_human_requirement_body_extracts_text_before_signature():
    body = parse_signed_human_requirement_body(
        "Please use the absolute URL.\n\n-- Human Reviewer\n\nExtra text ignored."
    )

    assert body == "Please use the absolute URL."


@pytest.mark.parametrize(
    "signature",
    [
        "-- Human Reviewer",
        "  -- Human Reviewer  ",
        "-- human reviewer",
        "-- HUMAN REVIEWER",
    ],
)
def test_parse_signed_human_requirement_body_accepts_standalone_signature_variants(signature):
    assert parse_signed_human_requirement_body(f"Required change.\n{signature}\n") == "Required change."


@pytest.mark.parametrize(
    "signature",
    [
        "-- OpenAI Codex",
        "-- Anthropic Claude",
        "-- Google Gemini",
        "-- coding-review-agent-loop",
        "Inline text -- Human Reviewer",
    ],
)
def test_parse_signed_human_requirement_body_rejects_agent_and_non_standalone_signatures(
    signature,
):
    assert parse_signed_human_requirement_body(f"Comment body.\n{signature}\n") is None


def test_parse_non_blocking_followups_extracts_bullets_only_from_section():
    review = """
    Looks good.

    ### Non-blocking follow-ups
    - Add `.agent-loop/` to `.gitignore`.
    1. Add regression coverage for stale memory refresh.
       Include multiple reviewers.

    ### Notes
    - This is not a follow-up.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    followups = parse_non_blocking_followups(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in followups] == [
        ("OpenAI Codex", "Add `.agent-loop/` to `.gitignore`."),
        (
            "OpenAI Codex",
            "Add regression coverage for stale memory refresh. Include multiple reviewers.",
        ),
    ]


def test_parse_approved_followups_extracts_same_pr_and_future_independently():
    review = """
    LGTM with cleanup.

    ### Same-PR follow-ups
    - Rename the helper for clarity.
      Keep the public behavior unchanged.

    ### Future follow-ups
    1. Add an integration fixture later.

    ### Non-blocking follow-ups
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    followups = parse_approved_followups(review, reviewer="OpenAI Codex")

    assert isinstance(followups.same_pr, tuple)
    assert isinstance(followups.future, tuple)
    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("OpenAI Codex", "Rename the helper for clarity. Keep the public behavior unchanged.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("OpenAI Codex", "Add an integration fixture later."),
        ("OpenAI Codex", "Legacy future item."),
    ]


def test_parse_approved_followups_accepts_trailing_colons_on_headings():
    review = """
    LGTM with follow-ups.

    ### Same-PR follow-ups:
    - Rename the helper for clarity.

    ### Future follow-ups:
    - Add an integration fixture later.

    ### Non-blocking follow-ups:
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("Gemini", "Rename the helper for clarity.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
        ("Gemini", "Legacy future item."),
    ]


@pytest.mark.parametrize(
    ("same_pr_heading", "future_heading", "legacy_heading"),
    [
        (
            "### **Same-PR follow-ups**",
            "### **Future follow-ups**",
            "### **Non-blocking follow-ups**",
        ),
        (
            "### **Same-PR follow-ups**:",
            "### **Future follow-ups.**",
            "### **Non-blocking follow-ups:**",
        ),
        (
            "### Same-PR follow-ups.",
            "### Future follow-ups.",
            "### Non-blocking follow-ups.",
        ),
    ],
)
def test_parse_approved_followups_accepts_common_markdown_heading_variants(
    same_pr_heading, future_heading, legacy_heading
):
    review = f"""
    LGTM with follow-ups.

    {same_pr_heading}
    - Rename the helper for clarity.

    {future_heading}
    - Add an integration fixture later.

    {legacy_heading}
    - Legacy future item.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        ("Gemini", "Rename the helper for clarity.")
    ]
    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
        ("Gemini", "Legacy future item."),
    ]


def test_parse_approved_followups_stops_at_unrelated_bold_heading():
    review = """
    LGTM with follow-ups.

    ### Future follow-ups
    - Add an integration fixture later.

    ### **Notes**
    - This is not a follow-up.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert [(item.reviewer, item.text) for item in followups.future] == [
        ("Gemini", "Add an integration fixture later."),
    ]


def test_parse_approved_followups_extracts_bullets_and_prose_paragraphs():
    bullet_review = """
    Codex approves final pass.

    ### Future follow-ups
    - Refine token estimation for large review prompts.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """
    prose_review = """
    Claude approves final pass.

    ### Future follow-ups
    The `_parse_gemini_output` helper is dead production code and could be removed
    in a future cleanup.

    ### Same-PR follow-ups
    Rename the helper in this PR before merge.
    Keep the behavior unchanged.

    ### Notes
    This note is outside the follow-up sections.

    <!-- AGENT_STATE: approved -->
    -- Anthropic Claude
    """

    bullet_followups = parse_approved_followups(bullet_review, reviewer="Codex")
    prose_followups = parse_approved_followups(prose_review, reviewer="Claude")

    assert [(item.reviewer, item.text) for item in bullet_followups.future] == [
        ("Codex", "Refine token estimation for large review prompts."),
    ]
    assert [(item.reviewer, item.text) for item in prose_followups.future] == [
        (
            "Claude",
            "The `_parse_gemini_output` helper is dead production code and could be removed in a future cleanup.",
        ),
    ]
    assert [(item.reviewer, item.text) for item in prose_followups.same_pr] == [
        ("Claude", "Rename the helper in this PR before merge. Keep the behavior unchanged."),
    ]


def test_parse_approved_followups_keeps_multiline_markdown_finding_as_one_item():
    review = """
    Still blocked.

    ### Same-PR follow-ups
    #### Normalize `_plan_subject` whitespace handling

    Keep the helper from creating distinct round subjects for leading/trailing
    whitespace-only differences.

    ```python
    assert _plan_subject("x") == _plan_subject(" x ")
    ```

    The implementation should preserve the current hash format.

    ---

    #### Harden `_decode_round_metadata` exception handling

    Invalid base64 and invalid JSON should still become `AgentLoopError`
    consistently.

    ### Notes
    CI note outside the section.

    <!-- AGENT_STATE: blocking -->
    -- Anthropic Claude
    """

    followups = parse_approved_followups(review, reviewer="Anthropic Claude")

    assert [(item.reviewer, item.text) for item in followups.same_pr] == [
        (
            "Anthropic Claude",
            "\n".join(
                [
                    "#### Normalize `_plan_subject` whitespace handling",
                    "",
                    "Keep the helper from creating distinct round subjects for leading/trailing whitespace-only differences.",
                    "",
                    "```python",
                    'assert _plan_subject("x") == _plan_subject(" x ")',
                    "```",
                    "",
                    "The implementation should preserve the current hash format.",
                ]
            ),
        ),
        (
            "Anthropic Claude",
            "\n".join(
                [
                    "#### Harden `_decode_round_metadata` exception handling",
                    "",
                    "Invalid base64 and invalid JSON should still become `AgentLoopError` consistently.",
                ]
            ),
        ),
    ]


@pytest.mark.parametrize(
    "placeholder",
    [
        "None",
        "none.",
        "(none)",
        "(n/a)",
        "N/A",
        "No follow-ups",
        "No same-PR follow-ups.",
        "No future follow-ups",
    ],
)
def test_parse_approved_followups_ignores_empty_placeholders(placeholder):
    review = f"""
    LGTM.

    ### Same-PR follow-ups
    - {placeholder}

    ### Future follow-ups
    - {placeholder}

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert followups.same_pr == ()
    assert followups.future == ()


def test_parse_approved_followups_ignores_prose_empty_placeholders():
    review = """
    LGTM.

    ### Same-PR follow-ups
    No same-PR follow-ups.

    ### Future follow-ups
    None

    ### Notes
    This sentence should not be captured.

    <!-- AGENT_STATE: approved -->
    -- Google Gemini
    """

    followups = parse_approved_followups(review, reviewer="Gemini")

    assert followups.same_pr == ()
    assert followups.future == ()


def test_parse_plan_review_items_extracts_structured_sections():
    review = """
    Plan looks sound with one required revision.

    ### Blocking plan issues
    - Cover how the plan avoids mixing `AGENT_STATE` and `AGENT_PLAN_STATE`.

    ### Same-plan follow-ups
    - Mention the exact docs pages to update.

    ### Future follow-ups
    - Consider a later helper to unify plan and PR disposition rendering.

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in items.blocking] == [
        (
            "OpenAI Codex",
            "Cover how the plan avoids mixing `AGENT_STATE` and `AGENT_PLAN_STATE`.",
        )
    ]
    assert [(item.reviewer, item.text) for item in items.same_plan] == [
        ("OpenAI Codex", "Mention the exact docs pages to update.")
    ]
    assert [(item.reviewer, item.text) for item in items.future] == [
        (
            "OpenAI Codex",
            "Consider a later helper to unify plan and PR disposition rendering.",
        )
    ]


def test_parse_plan_review_items_keeps_multiline_markdown_blocking_item_as_one_entry():
    review = """
    Plan needs one revision.

    ### Blocking plan issues
    #### Preserve multiline review items during tracking

    Do not split one reviewer-authored finding into separate ledger entries for
    paragraphs or code blocks.

    ```text
    item-2: heading
    item-3: paragraph
    ```

    ### Same-plan follow-ups
    - Mention the regression shape in the implementation plan.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [(item.reviewer, item.text) for item in items.blocking] == [
        (
            "OpenAI Codex",
            "\n".join(
                [
                    "#### Preserve multiline review items during tracking",
                    "",
                    "Do not split one reviewer-authored finding into separate ledger entries for paragraphs or code blocks.",
                    "",
                    "```text",
                    "item-2: heading",
                    "item-3: paragraph",
                    "```",
                ]
            ),
        )
    ]


@pytest.mark.parametrize(
    "placeholder",
    [
        "None",
        "(none)",
        "(n/a)",
        "No blocking plan issues.",
        "No same-plan follow-ups",
        "No future follow-ups.",
    ],
)
def test_parse_plan_review_items_ignores_empty_placeholders(placeholder):
    review = f"""
    Looks good.

    ### Blocking plan issues
    - {placeholder}

    ### Same-plan follow-ups
    {placeholder}

    ### Future follow-ups
    - {placeholder}

    <!-- AGENT_PLAN_STATE: approved -->
    -- Google Gemini
    """

    items = parse_plan_review_items(review, reviewer="Gemini")

    assert items.blocking == ()
    assert items.same_plan == ()
    assert items.future == ()


def test_parse_plan_item_dispositions_extracts_same_plan_status():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - [item-1] resolved
    - [item-2] still blocking
    - [item-3] same-plan
    - [item-4] future follow-up: okay to track separately now

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_plan_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "blocking", None),
        ("item-3", "same-plan", None),
        ("item-4", "future", "okay to track separately now"),
    ]


def test_parse_plan_item_dispositions_accepts_enriched_labels_with_trailing_arrow():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - [item-1] Same-plan follow-up from Google Gemini, round 1: keep the exact wording distinct -> same-plan: still need the mixed-reviewer case
    - [item-2] Blocking issue from OpenAI Codex, round 1: preserve public labels -> resolved

    <!-- AGENT_PLAN_STATE: blocking -->
    -- Anthropic Claude
    """

    dispositions = parse_plan_item_dispositions(review, reviewer="Anthropic Claude")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "same-plan", "still need the mixed-reviewer case"),
        ("item-2", "resolved", None),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-plan: none",
        "[item-1] same-plan: N/A",
        "[item-1] same-plan: no same-plan follow-ups",
        "[item-1] still blocking: none",
        "[item-1] still blocking: no blocking plan issues",
        "[item-1] future follow-up: none",
        "[item-1] future follow-up: no future follow-ups",
    ],
)
def test_parse_plan_item_dispositions_rejects_contradictory_active_notes(line):
    review = (
        "Approved after the latest revision."
        + prior_plan_item_dispositions(line)
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_plan_item_dispositions(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] same-plan:", "[item-1] still blocking:"])
def test_parse_plan_item_dispositions_rejects_trailing_colon_syntax(line):
    review = (
        "Approved after the latest revision."
        + prior_plan_item_dispositions(line)
        + "\n\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Invalid prior unresolved plan item disposition"):
        parse_plan_item_dispositions(review, reviewer="OpenAI Codex")


def test_parse_plan_item_dispositions_allows_resolved_none_and_substantive_same_plan():
    review = """
    Approved after the latest revision.
    """
    review += prior_plan_item_dispositions(
        "[item-1] resolved: none",
        "[item-2] same-plan: still need the mixed-reviewer case",
    )
    review += "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"

    dispositions = parse_plan_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", "none"),
        ("item-2", "same-plan", "still need the mixed-reviewer case"),
    ]


def test_parse_plan_item_dispositions_ignores_parenthesized_empty_placeholders():
    review = """
    Approved after the latest revision.

    ### Prior unresolved plan item dispositions
    - (none)
    - (n/a)

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_plan_item_dispositions(review, reviewer="OpenAI Codex") == ()


def test_parse_plan_review_drops_future_followups_in_blocking_reviews():
    review = structured_plan_review(
        state="blocking",
        summary="Still blocked.",
        blocking_plan_issues=["Need clearer rollback coverage."],
        future_followups=["Do this later."],
    )

    result = parse_plan_review(review, reviewer="OpenAI Codex")
    assert result.items.future == ()
    assert result.items.blocking  # blocking item survives


def test_parse_plan_review_rejects_future_disposition_in_blocking_reviews():
    review = structured_plan_review(
        state="blocking",
        summary="Still blocked.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "future", "note": "maybe later"}
        ],
    )

    with pytest.raises(AgentLoopError, match="Blocking plan reviews may not downgrade"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_contradictory_prior_plan_item_disposition():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-plan", "note": "none"}
        ],
    )

    with pytest.raises(AgentLoopError, match="empty placeholder"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_approved_state_with_active_items():
    plan_review = structured_plan_review(
        state="approved",
        summary="Needs work.",
        same_plan_followups=["Add one more orchestration test."],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(plan_review, reviewer="OpenAI Codex")

    with pytest.raises(AgentLoopError, match="AGENT_STATE"):
        parse_review(plan_review, reviewer="OpenAI Codex")


def test_parse_plan_review_rejects_approved_state_with_blocking_items():
    review = structured_plan_review(
        state="approved",
        summary="Needs work.",
        blocking_plan_issues=["Add one more orchestration test."],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] still blocking", "[item-1] same-plan"])
def test_parse_plan_review_rejects_approved_state_with_active_prior_disposition(line):
    item_id, disposition = ("item-1", "blocking") if "blocking" in line else ("item-1", "same-plan")
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": item_id, "disposition": disposition}],
    )

    with pytest.raises(AgentLoopError, match="Approved plan reviews must be fully complete"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_validate_plan_review_response_rejects_duplicate_item_ids():
    review = structured_plan_review(
        state="blocking",
        summary="Still refining the plan.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "same-plan", "note": "keep the extra regression coverage"},
            {"item_id": "item-1", "disposition": "resolved"},
        ],
    )

    with pytest.raises(AgentLoopError, match="more than once: item-1"):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
            ),
        )


def test_validate_plan_review_response_rejects_unknown_item_ids():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-9", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError, match="item-9"):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
            ),
        )


def test_validate_plan_review_response_rejects_unknown_item_with_empty_prior_ledger():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
        )

    assert exc_info.value.unknown_ids == ("item-1",)
    assert exc_info.value.allowed_ids == ()
    assert "Same-round findings are informational only" in exc_info.value.same_round_description


def test_unknown_prior_item_disposition_error_message_includes_ids():
    exc = UnknownPriorItemDispositionError(
        unknown_ids=("item-15",),
        allowed_ids=("item-12", "item-17"),
        same_round_description="Same-round findings are informational only.",
    )

    message = str(exc)
    assert "item-15" in message
    assert "item-12" in message
    assert "item-17" in message


def test_validate_plan_review_response_describes_same_round_unknown_item():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
    )
    current_round_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=2,
        text="Same-round finding.",
        status="same-plan",
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
            current_round_items=(current_round_item,),
        )

    assert "item-2" in exc_info.value.same_round_description


def test_validate_plan_review_response_accepts_structured_resolved_dispositions():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    )

    parsed = _validate_plan_review_response(
        review,
        reviewer="OpenAI Codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Keep the extra regression coverage.",
                status="same-plan",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Clarify the fallback trigger.",
                status="blocking",
            ),
        ),
    )

    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved"),
        ("item-2", "resolved"),
    ]


def test_validate_plan_review_response_rejects_missing_structured_dispositions():
    review = structured_plan_review(
        state="approved",
        summary="Looks good now.",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(
        AgentLoopError, match="did not evaluate all prior unresolved plan items: item-2"
    ):
        _validate_plan_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Keep the extra regression coverage.",
                    status="same-plan",
                ),
                UnresolvedReviewItem(
                    item_id="item-2",
                    reviewer="Google Gemini",
                    source_round=1,
                    text="Clarify the fallback trigger.",
                    status="blocking",
                ),
            ),
        )


def test_parse_unresolved_item_dispositions_extracts_structured_updates():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - [item-1] resolved
    - [item-2] still blocking
    - [item-3] same-pr
    - [item-4] future follow-up: split this into a separate PR

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "blocking", None),
        ("item-3", "same-pr", None),
        ("item-4", "future", "split this into a separate PR"),
    ]


def test_parse_unresolved_item_dispositions_accepts_enriched_labels_with_trailing_arrow():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - [item-1] Same-PR follow-up from Google Gemini, round 1: require source issue reference in PR body -> same-pr: keep the body reference
    - [item-2] Blocking issue from OpenAI Codex, round 1: rename the helper -> resolved

    <!-- AGENT_STATE: blocking -->
    -- Anthropic Claude
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="Anthropic Claude")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "same-pr", "keep the body reference"),
        ("item-2", "resolved", None),
    ]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-pr: none",
        "[item-1] same-pr: N/A",
        "[item-1] same-pr: no same-pr follow-ups",
        "[item-1] still blocking: none",
        "[item-1] still blocking: no blocking issues",
        "[item-1] future follow-up: none",
        "[item-1] future follow up: none",
        "[item-1] future follow-up: no future follow-ups",
        "[item-1] future follow-up: no follow-ups",
    ],
)
def test_parse_unresolved_item_dispositions_rejects_contradictory_active_notes(line):
    review = "LGTM." + prior_item_dispositions(line) + "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] same-pr:", "[item-1] still blocking:"])
def test_parse_unresolved_item_dispositions_rejects_trailing_colon_syntax(line):
    review = "LGTM." + prior_item_dispositions(line) + "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(
        AgentLoopError,
        match=r"Invalid prior unresolved item disposition.*section `### Prior unresolved item dispositions`, line 4",
    ):
        parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")


def test_parse_unresolved_item_dispositions_allows_resolved_none_and_substantive_same_pr():
    review = """
    LGTM.
    """
    review += prior_item_dispositions(
        "[item-1] resolved: none",
        "[item-2] same-pr: rename the helper before merge",
    )
    review += "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", "none"),
        ("item-2", "same-pr", "rename the helper before merge"),
    ]


def test_parse_unresolved_item_dispositions_ignores_parenthesized_empty_placeholders():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    - (none)
    - (n/a)

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex") == ()


def test_parse_unresolved_item_dispositions_ignores_non_bullet_prose():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    These are the remaining status calls.
    - [item-1] resolved
    Closing thought after the bullets.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    dispositions = parse_unresolved_item_dispositions(review, reviewer="OpenAI Codex")

    assert [(item.item_id, item.disposition, item.note) for item in dispositions] == [
        ("item-1", "resolved", None),
    ]


def test_validate_review_response_accepts_structured_resolved_dispositions():
    review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    )

    parsed = _validate_review_response(
        review,
        reviewer="OpenAI Codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Rename the helper.",
                status="same-pr",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Keep the PR body issue reference.",
                status="blocking",
            ),
        ),
    )

    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved"),
        ("item-2", "resolved"),
    ]


def test_validate_review_response_rejects_unknown_item_with_empty_prior_ledger():
    review = structured_pr_review(
        state="approved",
        summary="LGTM.",
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(),
        )

    assert exc_info.value.unknown_ids == ("item-1",)
    assert exc_info.value.allowed_ids == ()
    assert "Same-round findings are informational only" in exc_info.value.same_round_description


def test_validate_review_response_rejects_ambiguous_blanket_prose():
    review = """
    LGTM.

    ### Prior unresolved item dispositions
    All prior items look resolved.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    with pytest.raises(AgentLoopError, match="required structured format"):
        _validate_review_response(
            review,
            reviewer="OpenAI Codex",
            unresolved_items=(
                UnresolvedReviewItem(
                    item_id="item-1",
                    reviewer="Anthropic Claude",
                    source_round=1,
                    text="Rename the helper.",
                    status="same-pr",
                ),
            ),
        )


def test_parse_review_drops_future_followups_in_blocking_reviews():
    review = """
    Still blocked.

    ### Same-PR follow-ups
    - Tighten the helper in this file.

    ### Future follow-ups
    - Do this later.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.state == "blocking"
    assert [item.text for item in parsed.followups.same_pr] == ["Tighten the helper in this file."]
    assert parsed.followups.future == ()


def test_parse_review_rejects_contradictory_prior_item_disposition():
    review = "LGTM." + prior_item_dispositions("[item-1] same-pr: none")
    review += "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        parse_review(review, reviewer="OpenAI Codex")


def test_parse_review_rejects_approved_state_with_same_pr_followups():
    review = """
    LGTM.

    ### Same-PR follow-ups
    - Tighten the helper in this file.

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_review(review, reviewer="OpenAI Codex")


@pytest.mark.parametrize("line", ["[item-1] still blocking", "[item-1] same-pr"])
def test_parse_review_rejects_approved_state_with_active_prior_disposition(line):
    review = "LGTM." + prior_item_dispositions(line)
    review += "\n\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_review(review, reviewer="OpenAI Codex")


def test_parse_review_populates_summary_from_legacy_markdown():
    review = """
    Blocking issue summary.

    ### Same-PR follow-ups
    - Rename the helper.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.summary == "Blocking issue summary."


def test_parse_review_round_trips_blocking_issues_section_without_polluting_summary():
    review = (
        "Blocking issue summary."
        + blocking_issues(
            "Cover the regression case in the PR test suite.",
            "Tighten the error assertion wording.",
        )
        + "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.summary == "Blocking issue summary."
    assert [item.text for item in parsed.blocking_items] == [
        "Cover the regression case in the PR test suite.",
        "Tighten the error assertion wording.",
    ]


def test_parse_review_dedupes_same_pr_items_that_duplicate_blocking_items():
    review = (
        "Blocking issue summary."
        + blocking_issues("`Add the missing share.html CSS update.`")
        + "\n\n### Same-PR follow-ups\n"
        + "- Add the missing share.html CSS update.\n"
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "`Add the missing share.html CSS update.`"
    ]
    assert parsed.followups.same_pr == ()


def test_parse_review_prefers_same_pr_over_duplicate_future_followups():
    review = """
    Blocking on a local cleanup.

    ### Same-PR follow-ups
    - Fix the duplicated prompt wording introduced by this PR.

    ### Future follow-ups
    - `Fix the duplicated prompt wording introduced by this PR.`

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.followups.same_pr] == [
        "Fix the duplicated prompt wording introduced by this PR."
    ]
    assert parsed.followups.future == ()


def test_parse_review_prefers_blocking_over_duplicate_future_followups():
    review = (
        "Blocking issue summary."
        + blocking_issues("Fix the indentation in the touched `orchestrator.py` call.")
        + "\n\n### Future follow-ups\n"
        + "- fix the indentation in the touched orchestrator.py call\n"
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Fix the indentation in the touched `orchestrator.py` call."
    ]
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_dedupes_exact_normalized_same_pr_duplicates():
    review = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Blocked.",
            "blocking_items": ["- Add the missing `share.html` CSS update."],
            "same_pr_followups": ["Add the missing share.html CSS update"],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )
    review += "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "- Add the missing `share.html` CSS update."
    ]
    assert parsed.followups.same_pr == ()


def test_parse_structured_pr_review_prefers_same_pr_over_duplicate_future_followups():
    review = structured_pr_review(
        state="blocking",
        summary="Blocked on local cleanup.",
        blocking_items=[],
        same_pr_followups=["Fix the duplicated prompt wording introduced by this PR."],
        future_followups=["fix the duplicated prompt wording introduced by this PR"],
    )

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert parsed.blocking_items == ()
    assert [item.text for item in parsed.followups.same_pr] == [
        "Fix the duplicated prompt wording introduced by this PR."
    ]
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_prefers_blocking_over_duplicate_future_followups():
    review = structured_pr_review(
        state="blocking",
        summary="Blocked.",
        blocking_items=["Fix the indentation in the touched `orchestrator.py` call."],
        same_pr_followups=[],
        future_followups=["fix the indentation in the touched orchestrator.py call"],
    )

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Fix the indentation in the touched `orchestrator.py` call."
    ]
    assert parsed.followups.same_pr == ()
    assert parsed.followups.future == ()


def test_parse_structured_pr_review_keeps_near_but_distinct_same_pr_items():
    review = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "blocking",
            "summary": "Blocked.",
            "blocking_items": ["Add the missing share.html CSS update."],
            "same_pr_followups": ["Add the missing share.html print CSS update."],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )
    review += "\n\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"

    parsed = parse_pr_review(review, reviewer="OpenAI Codex")

    assert [item.text for item in parsed.blocking_items] == [
        "Add the missing share.html CSS update."
    ]
    assert [item.text for item in parsed.followups.same_pr] == [
        "Add the missing share.html print CSS update."
    ]


def test_pr_239_style_followup_classification_fixture():
    review = """
    PR #239-style cleanup classification.

    ### Same-PR follow-ups
    - orchestrator.py line 2100: subject=current_pr_subject is indented 4 extra spaces relative to sibling keyword arguments.
    - _repair_prior_item_ids_instruction duplicates the same-round warning/context in the repair prompt.

    ### Future follow-ups
    - _round_ledger_may_be_incomplete cross-subject branch could be bounded if comment history grows very large.

    <!-- AGENT_STATE: blocking -->
    -- OpenAI Codex
    """

    followups = parse_approved_followups(review, reviewer="OpenAI Codex")

    assert [item.text for item in followups.same_pr] == [
        "orchestrator.py line 2100: subject=current_pr_subject is indented 4 extra spaces relative to sibling keyword arguments.",
        "_repair_prior_item_ids_instruction duplicates the same-round warning/context in the repair prompt.",
    ]
    assert [item.text for item in followups.future] == [
        "_round_ledger_may_be_incomplete cross-subject branch could be bounded if comment history grows very large."
    ]


def test_legacy_plan_review_helpers_populate_summary_from_markdown():
    review = """
    Plan needs one more regression test.

    ### Same-plan follow-ups
    - Add a regression test matrix.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert _review_freeform_summary_text(review) == "Plan needs one more regression test."
    assert [item.text for item in items.same_plan] == ["Add a regression test matrix."]


def test_parse_plan_review_items_dedupes_plan_buckets_by_normalized_text():
    review = """
    Plan still needs cleanup.

    ### Blocking plan issues
    - Add `retry` coverage.

    ### Same-plan follow-ups
    - *add retry coverage!*
    - Add parser comment.

    ### Future follow-ups
    - ADD RETRY COVERAGE.
    - add `parser` comment
    - Add parser documentation later.

    <!-- AGENT_PLAN_STATE: blocking -->
    -- OpenAI Codex
    """

    items = parse_plan_review_items(review, reviewer="OpenAI Codex")

    assert [item.text for item in items.blocking] == ["Add `retry` coverage."]
    assert [item.text for item in items.same_plan] == ["Add parser comment."]
    assert [item.text for item in items.future] == ["Add parser documentation later."]


def test_parse_structured_pr_review_normalizes_v1_payload_with_footer_contract():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Looks good after the latest fix.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": ["Document cleanup for a later PR."],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved"},
                    {
                        "item_id": "item-2",
                        "disposition": "future",
                        "note": "okay to split into follow-up work",
                    },
                ],
            }
        )
        + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.state == "approved"
    assert parsed.summary == "Looks good after the latest fix."
    assert parsed.blocking_items == ()
    assert [item.text for item in parsed.followups.future] == ["Document cleanup for a later PR."]
    assert [(item.item_id, item.disposition, item.note) for item in parsed.dispositions] == [
        ("item-1", "resolved", None),
        ("item-2", "future", "okay to split into follow-up work"),
    ]


def test_parse_structured_pr_review_tolerates_omitted_empty_collections():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Looks good after the latest fix.",
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.state == "approved"
    assert parsed.blocking_items == ()
    assert parsed.followups.same_pr == ()
    assert parsed.followups.future == ()
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_pr_review_strips_verdict_and_sections_from_json_summary():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": (
                    "**Review verdict:** blocking\n\n"
                    "Need one more regression test.\n\n"
                    "### Blocking issues\n"
                    "- Duplicate line that should not remain in the summary."
                ),
                "blocking_items": ["Need one more regression test."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex\n"
    )

    parsed = parse_structured_pr_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Need one more regression test."


def test_parse_structured_pr_review_rejects_kind_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Wrong kind.",
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="kind mismatch"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_hard_fails_on_unsupported_schema_version():
    payload = (
        json.dumps(
            {
                "schema_version": 2,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Wrong version.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Unsupported structured response schema_version: 2"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_rejects_markdown_when_no_structured_candidate_exists():
    review = "Looks good in markdown.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="required structured format"):
        parse_pr_review(review, reviewer="OpenAI Codex")


def test_legacy_parse_review_still_parses_markdown_for_historical_display():
    review = "Looks good in markdown.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"

    parsed = parse_review(review, reviewer="OpenAI Codex")

    assert parsed.state == "approved"
    assert parsed.summary == "Looks good in markdown."


def test_parse_pr_review_rejects_invalid_structured_candidate_instead_of_falling_back_to_markdown():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Missing required arrays.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="missing required field"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_strips_future_followups_in_blocking_reviews():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_items": ["Needs one more test."],
                "same_pr_followups": [],
                "future_followups": ["Clean this up later."],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    result = parse_structured_pr_review(payload, reviewer="OpenAI Codex")
    assert result is not None
    assert result.followups.future == ()
    assert result.blocking_items  # blocking item survives


def test_parse_pr_review_rejects_structured_candidate_with_unknown_nested_keys():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "extra": "nope"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_rejects_structured_candidate_with_invalid_item_id():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item 1", "disposition": "resolved"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must match"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_pr_review_requires_strict_structured_disposition_enums():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [
                    {"item_id": "item-1", "disposition": "still blocking"},
                ],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must be one of"):
        parse_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_rejects_approved_blocking_items():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "Almost there.",
                "blocking_items": ["Still needs a regression test."],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="Approved reviews must be fully complete"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


@pytest.mark.parametrize(
    "suffix",
    [
        "\nExtra explanation after the payload.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        "\n```text\nextra block\n```\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        "\n- stray bullet\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
    ],
)
def test_parse_structured_pr_review_rejects_trailing_content_before_footer(suffix):
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "pr_review",
            "state": "approved",
            "summary": "LGTM.",
            "blocking_items": [],
            "same_pr_followups": [],
            "future_followups": [],
            "prior_item_dispositions": [],
        }
    )

    with pytest.raises(
        AgentLoopError,
        match="place <!-- AGENT_STATE|may not include prose between|may not include trailing prose",
    ):
        parse_structured_pr_review(payload + suffix, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_rejects_footer_state_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "approved",
                "summary": "LGTM.",
                "blocking_items": [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="must match the payload state"):
        parse_structured_pr_review(payload, reviewer="OpenAI Codex")


def test_parse_structured_pr_review_falls_back_when_json_is_embedded_in_markdown():
    review = """
    Here is an example:

    ```json
    {"schema_version": 1, "kind": "pr_review"}
    ```

    <!-- AGENT_STATE: approved -->
    -- OpenAI Codex
    """

    assert parse_structured_pr_review(review, reviewer="OpenAI Codex") is None
    assert parse_review(review, reviewer="OpenAI Codex").state == "approved"


def test_parse_structured_plan_review_normalizes_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Plan looks good.",
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": ["Consider a later cleanup pass."],
                "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Plan looks good."
    assert [item.text for item in parsed.items.future] == ["Consider a later cleanup pass."]
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_plan_review_tolerates_omitted_empty_collections():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": "Plan looks good.",
                "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.items.blocking == ()
    assert parsed.items.same_plan == ()
    assert parsed.items.future == ()
    assert [(item.item_id, item.disposition) for item in parsed.dispositions] == [
        ("item-1", "resolved")
    ]


def test_parse_structured_plan_review_strips_verdict_and_sections_from_json_summary():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": (
                    "**Review verdict:** blocking\n\n"
                    "Need clearer rollback coverage.\n\n"
                    "### Same-plan follow-ups\n"
                    "- Extra duplicate text."
                ),
                "blocking_plan_issues": ["Need clearer rollback coverage."],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert parsed.summary == "Need clearer rollback coverage."


def test_parse_structured_plan_review_dedupes_same_plan_against_blocking_items():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_plan_issues": ["Add `retry` coverage."],
                "same_plan_followups": [
                    "*add retry coverage!*",
                    "Add retry coverage for timeout handling.",
                ],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = parse_structured_plan_review(payload, reviewer="OpenAI Codex")

    assert parsed is not None
    assert [item.text for item in parsed.items.blocking] == ["Add `retry` coverage."]
    assert [item.text for item in parsed.items.same_plan] == [
        "Add retry coverage for timeout handling."
    ]


def test_parse_structured_plan_review_strips_blocking_future_followups():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked.",
                "blocking_plan_issues": ["Need clearer rollback coverage."],
                "same_plan_followups": [],
                "future_followups": ["Refactor the prompt later."],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    result = parse_structured_plan_review(payload, reviewer="OpenAI Codex")
    assert result is not None
    assert result.items.future == ()
    assert result.items.blocking  # blocking item survives


def test_validate_structured_coder_followup_accepts_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the first item; one remains.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
                "tests_run": ["python -m pytest tests/test_agent_loop.py -k structured"],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_coder_followup(payload)

    assert parsed is not None
    assert parsed.addressed_items == ("item-1",)
    assert parsed.remaining_items == ("item-2",)
    assert parsed.human_requirements.addressed_ids == ("Requirement 1",)
    assert parsed.addressed_item_notes == {}
    assert parsed.remaining_item_notes == {}


def test_validate_structured_coder_followup_accepts_optional_item_notes():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed the parser; deferred the docs.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "addressed_item_notes": {"item-1": "Added parsing coverage."},
                "remaining_item_notes": {"item-2": "Deferred until the docs owner weighs in."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_coder_followup(payload)

    assert parsed is not None
    assert parsed.addressed_item_notes == {"item-1": "Added parsing coverage."}
    assert parsed.remaining_item_notes == {
        "item-2": "Deferred until the docs owner weighs in."
    }


def test_validate_structured_coder_followup_rejects_note_for_unlisted_item():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed one item.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "addressed_item_notes": {"item-2": "This note is stale."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="item-2.*not listed in coder_followup.addressed_items"):
        validate_structured_coder_followup(payload)


@pytest.mark.parametrize("bad_note", ["", "   ", 5, None])
def test_validate_structured_coder_followup_rejects_invalid_note_values(bad_note):
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Addressed one item.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "addressed_item_notes": {"item-1": bad_note},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="coder_followup.addressed_item_notes.item-1"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_returns_none_when_no_structured_candidate_exists():
    assert (
        validate_structured_coder_followup(
            "Implemented the fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        )
        is None
    )


def test_validate_structured_coder_followup_rejects_unknown_keys_in_structured_candidate():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "approved",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                    "extra": "nope",
                },
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="unknown field\\(s\\): extra"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_rejects_footer_state_mismatch():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                },
            }
        )
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="footer AGENT_STATE must match the payload state"):
        validate_structured_coder_followup(payload)


def test_validate_structured_coder_followup_rejects_trailing_prose_after_footer():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Done.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": True,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex\nextra"
    )

    with pytest.raises(AgentLoopError, match="may not include trailing prose"):
        validate_structured_coder_followup(payload)


@pytest.mark.parametrize(
    ("addressed_ids", "checked_discussion_directly", "surfaced_ids", "requires_direct_discussion_ack", "message"),
    [
        (
            ("Requirement 1",),
            False,
            ("Requirement 1", "Requirement 2"),
            False,
            "did not address all surfaced signed human requirement IDs",
        ),
        (
            ("Requirement 1", "Requirement 1"),
            False,
            ("Requirement 1",),
            False,
            "listed signed human requirement IDs more than once",
        ),
        (
            ("Requirement 99",),
            False,
            ("Requirement 1",),
            False,
            "referenced unknown signed human requirement IDs",
        ),
        (
            (),
            False,
            (),
            True,
            "must acknowledge that the prompt omitted the detailed signed human requirements",
        ),
    ],
)
def test_validate_structured_human_requirements_acknowledgement_rejects_invalid_payloads(
    addressed_ids,
    checked_discussion_directly,
    surfaced_ids,
    requires_direct_discussion_ack,
    message,
):
    with pytest.raises(AgentLoopError, match=message):
        validate_structured_human_requirements_acknowledgement(
            addressed_ids,
            checked_discussion_directly=checked_discussion_directly,
            surfaced_requirement_ids=surfaced_ids,
            requires_direct_discussion_ack=requires_direct_discussion_ack,
        )


def test_validate_structured_plan_revision_accepts_v1_payload():
    payload = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised the plan to cover rollback testing.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered in the new tests."}
                ],
                "plan_steps": ["Update protocol.py.", "Add regression tests."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    parsed = validate_structured_plan_revision(payload)

    assert parsed is not None
    assert parsed.state == "blocking"
    assert [(item.item_id, item.disposition) for item in parsed.prior_plan_item_dispositions] == [
        ("item-1", "resolved")
    ]
    assert parsed.plan_steps == ("Update protocol.py.", "Add regression tests.")


def test_validate_plan_revision_response_rejects_marker_only_markdown():
    with pytest.raises(AgentLoopError, match="Plan revision did not use the required structured format"):
        _validate_plan_revision_response(
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
        )


def test_validate_plan_revision_response_rejects_unknown_prior_disposition():
    revision = structured_plan_revision(
        prior_plan_item_dispositions=[
            {"item_id": "item-15", "disposition": "resolved"},
        ],
    )
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )

    with pytest.raises(UnknownPriorItemDispositionError) as exc_info:
        _validate_plan_revision_response(revision, unresolved_items=(active_item,))

    assert exc_info.value.unknown_ids == ("item-15",)
    assert exc_info.value.allowed_ids == ("item-12",)


def test_render_canonical_plan_revision_rejects_unknown_prior_disposition_without_keyerror():
    revision = validate_structured_plan_revision(
        structured_plan_revision(
            prior_plan_item_dispositions=[
                {"item_id": "item-15", "disposition": "resolved"},
            ],
        )
    )
    assert revision is not None
    active_item = UnresolvedReviewItem(
        item_id="item-12",
        reviewer="Google Gemini",
        source_round=5,
        text="Active must-fix item.",
        status="blocking",
    )

    with pytest.raises(AgentLoopError, match="Renderer encountered unknown prior item ID") as exc_info:
        render_canonical_plan_revision(revision, prior_items=(active_item,))

    assert not isinstance(exc_info.value, KeyError)


@pytest.mark.parametrize(
    ("payload", "pattern"),
    [
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "approved",
                "summary": "Wrong state.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Update protocol.py."],
            },
            "plan_revision.state must be `blocking`",
        ),
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Update protocol.py."],
            },
            "plan_revision.summary must be a non-empty string",
        ),
        (
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Missing steps.",
                "prior_plan_item_dispositions": [],
                "plan_steps": [],
            },
            "plan_revision.plan_steps must contain at least 1 item",
        ),
    ],
)
def test_validate_structured_plan_revision_rejects_invalid_payload(payload, pattern):
    footer_state = payload["state"]
    text = json.dumps(payload) + f"\n<!-- AGENT_PLAN_STATE: {footer_state} -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match=pattern):
        validate_structured_plan_revision(text)


def test_extract_structured_plan_review_payload_rejects_embedded_json_markdown():
    review = """
    Here is an example:

    ```json
    {"schema_version": 1, "kind": "plan_review"}
    ```

    <!-- AGENT_PLAN_STATE: approved -->
    -- OpenAI Codex
    """

    assert _extract_structured_plan_review_payload(review) is None


@pytest.mark.parametrize(
    ("builder", "extractor"),
    [
        (lambda: structured_plan_review(reviewer="Google Gemini"), _extract_structured_plan_review_payload),
        (lambda: structured_pr_review(reviewer="Google Gemini"), _extract_structured_pr_review_payload),
        (lambda: structured_coder_followup(reviewer="Anthropic Claude"), _extract_structured_coder_followup_payload),
        (lambda: structured_plan_revision(reviewer="Anthropic Claude"), _extract_structured_plan_revision_payload),
    ],
)
def test_structured_extractors_recover_leading_public_response_marker(builder, extractor):
    text = f"\n\n{PUBLIC_RESPONSE_MARKER}\n{builder()}"

    payload = extractor(text)

    assert payload is not None


def test_response_file_marker_normalization_reports_unrecoverable_marker():
    text = f"{PUBLIC_RESPONSE_MARKER}\n### Review\nLooks good."

    normalized, status = normalize_response_file_structured_text(text)

    assert normalized == text
    assert status == "leading-public-response-marker-not-recoverable"


def test_public_response_json_prefix_strips_marker_variants():
    text = "==== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ====\n" + json.dumps(
        {"error": {"status": 429, "message": "quota exceeded"}}
    )

    payload = _decode_public_response_json_prefix(text)

    assert payload == {"error": {"status": 429, "message": "quota exceeded"}}
    assert _is_transient_public_response(text)


def test_failure_category_threads_public_response_expected_kind():
    text = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "message": "quota exceeded",
        }
    )

    assert _failure_category(text, public_response=True) == "deterministic"
    assert (
        _failure_category(
            text,
            public_response=True,
            repair_expected_kind="future_structured_kind",
        )
        == "transient"
    )


@pytest.mark.parametrize(
    "text",
    [
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "approved",
                "summary": PUBLIC_RESPONSE_MARKER,
                "blocking_plan_issues": [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": [],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        "Some prose first.\n"
        + PUBLIC_RESPONSE_MARKER
        + "\n"
        + structured_plan_review(reviewer="Google Gemini"),
    ],
)
def test_response_file_marker_not_stripped_inside_json_or_mid_prose(text):
    normalized, status = normalize_response_file_structured_text(text)

    assert normalized == text
    assert status is None


def test_extract_structured_plan_review_payload_rejects_footer_state_mismatch():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "state": "approved",
            "summary": "Plan looks good.",
            "blocking_plan_issues": [],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [],
        }
    )
    text = payload + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"

    with pytest.raises(AgentLoopError, match="footer AGENT_PLAN_STATE must match"):
        _extract_structured_plan_review_payload(text)


def test_extract_structured_plan_review_payload_rejects_trailing_prose_after_signature():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_review",
            "state": "approved",
            "summary": "Plan looks good.",
            "blocking_plan_issues": [],
            "same_plan_followups": [],
            "future_followups": [],
            "prior_plan_item_dispositions": [],
        }
    )
    text = payload + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex\nextra"

    with pytest.raises(AgentLoopError, match="trailing prose"):
        _extract_structured_plan_review_payload(text)


def test_parse_plan_review_hard_fails_after_top_level_json_prefix():
    review = (
        '{"schema_version":1,"kind":"plan_review","state":"approved","summary":"Plan looks good.",'
        '"blocking_plan_issues":[],"same_plan_followups":[],"future_followups":[]}\n'
        "<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
    )

    with pytest.raises(AgentLoopError, match="plan_review is missing required field"):
        parse_plan_review(review, reviewer="OpenAI Codex")


def test_extract_structured_plan_revision_payload_accepts_human_requirements_prefix():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised the plan.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Update protocol.py."],
        }
    )
    text = (
        payload
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n- Requirement 1: covered in step 1.\n"
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )

    assert _extract_structured_plan_revision_payload(text) is not None


def test_extract_structured_plan_revision_payload_rejects_bad_footer_ordering():
    payload = json.dumps(
        {
            "schema_version": 1,
            "kind": "plan_revision",
            "state": "blocking",
            "summary": "Revised the plan.",
            "prior_plan_item_dispositions": [],
            "plan_steps": ["Update protocol.py."],
        }
    )
    text = payload + "\n-- OpenAI Codex\n<!-- AGENT_PLAN_STATE: blocking -->"

    with pytest.raises(AgentLoopError, match="AGENT_PLAN_STATE"):
        _extract_structured_plan_revision_payload(text)


def test_expect_string_list_enforces_min_length():
    with pytest.raises(AgentLoopError, match="must contain at least 1 item"):
        _expect_string_list([], context="plan_revision.plan_steps", item_context="plan_revision.plan_steps", min_length=1)

