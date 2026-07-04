"""Comment rendering tests extracted from test_agent_loop.py (lines 5180–5565 and 5856–6115).

Tests for render_canonical_plan_steps, render_canonical_plan_revision,
render_public_agent_comment, _render_public_*_comment, and related functions.
"""
import json
import re

import pytest

from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_discuss_review_comment,
    _render_public_plan_review_comment,
    _render_public_plan_revision_comment,
    _render_public_pr_review_comment,
    decode_deferred_stages_marker,
    normalize_freeform_signature,
    render_deferred_stages_section,
    render_discuss_round_summary_comment,
)
from coding_review_agent_loop.protocol import ParsedDiscussReview
from coding_review_agent_loop.orchestrator import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    ITEM_SUMMARY_LIMIT,
    _extract_current_deferred_stages,
    _format_unresolved_item_label,
    _render_public_review_comment,
    _review_freeform_summary_text,
    render_canonical_plan_revision,
    render_canonical_plan_steps,
    render_public_agent_comment,
)
from coding_review_agent_loop.protocol import (
    DeferredStage,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_pr_review,
    parse_review,
    parse_structured_plan_review,
    parse_unresolved_item_dispositions,
    validate_structured_coder_followup,
    validate_structured_plan_revision,
    validate_structured_plan_state,
)

from agent_loop_helpers import (
    blocking_issues,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_coder_followup,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)

def test_render_canonical_plan_steps_numbers_items():
    assert render_canonical_plan_steps(("Update protocol.py.", "Add tests.")) == (
        "1. Update protocol.py.\n2. Add tests."
    )


def test_render_deferred_stages_section_marker_round_trips_colon_in_title():
    """A title containing its own colon must not be corrupted: the human
    readable `- {title}: {summary}` bullet is not what gets parsed back, an
    AGENT_DEFERRED_STAGES marker carrying the exact structured pairs is
    (#492 review)."""
    stages = (
        DeferredStage(title="Stage 2: API follow-up", summary="Split out the API work."),
        DeferredStage(title="Billing", summary="Reconcile invoices."),
    )

    section = render_deferred_stages_section(stages)

    assert "- Stage 2: API follow-up: Split out the API work." in section
    marker_match = re.search(r"<!--\s*AGENT_DEFERRED_STAGES:\s*(?P<payload>\S+)\s*-->", section)
    assert marker_match is not None
    assert decode_deferred_stages_marker(marker_match.group("payload")) == stages


def test_extract_current_deferred_stages_recovers_colon_title_from_canonical_markdown():
    revision = validate_structured_plan_revision(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Implement the core parser change.",
                "prior_plan_item_dispositions": [],
                "plan_steps": ["Update the parser."],
                "deferred_stages": [
                    {"title": "Stage 2: API follow-up", "summary": "Split out the API work."}
                ],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert revision is not None
    canonical = render_canonical_plan_revision(revision, ())

    recovered = _extract_current_deferred_stages(canonical)

    assert recovered == (
        DeferredStage(title="Stage 2: API follow-up", summary="Split out the API work."),
    )


def test_render_canonical_plan_revision_and_public_comment():
    parsed = validate_structured_plan_revision(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised the plan to cover rollback behavior.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-4", "disposition": "resolved", "note": "Added a resume-path step."}
                ],
                "plan_steps": ["Update protocol.py.", "Add orchestrator resume tests."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-4",
            reviewer="OpenAI Codex",
            source_round=2,
            text="Add a resume-path step.",
            status="blocking",
        ),
    )

    canonical = render_canonical_plan_revision(parsed, prior_items)
    public = _render_public_plan_revision_comment(
        parsed,
        prior_items=prior_items,
        raw_text='{"schema_version":1}\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex',
        agent="Codex",
    )

    assert canonical == (
        "Revised the plan to cover rollback behavior.\n\n"
        "### Prior plan item dispositions\n"
        "- [item-4] Blocking issue from OpenAI Codex, round 2: Add a resume-path step. -> "
        "resolved: Added a resume-path step.\n\n"
        "### Plan steps\n"
        "1. Update protocol.py.\n"
        "2. Add orchestrator resume tests."
    )
    assert public == (
        "## Revised plan\n\n"
        + canonical
        + "\n\n<!-- AGENT_PLAN_STATE: blocking -->\n\n-- OpenAI Codex"
    )
    assert '"kind": "plan_revision"' not in public


def test_render_structured_plan_state_to_public_markdown():
    raw = (
        json.dumps(
            {
                "kind": "plan_state",
                "summary": "Plan the renderer fix.",
                "plan_steps": ["Detect structured plan_state.", "Render public markdown."],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Antigravity"
    )
    parsed = validate_structured_plan_state(raw)

    assert parsed is not None
    public = render_public_agent_comment(
        kind="plan_state",
        parsed=parsed,
        agent="antigravity",
        model_used="Gemini 3.1 Pro (High)",
    )

    assert public == (
        "## Plan\n\n"
        "Plan the renderer fix.\n\n"
        "### Plan steps\n"
        "1. Detect structured plan_state.\n"
        "2. Render public markdown.\n\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n\n"
        "-- Google Antigravity: Gemini 3.1 Pro (High)"
    )
    assert '"kind": "plan_state"' not in public


def test_render_public_coder_followup_comment():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Added the requested regression test.",
                "addressed_items": ["item-1", "item-2"],
                "remaining_items": [],
                "addressed_item_notes": {
                    "item-1": "Added coverage for the parser.",
                    "item-2": "Updated the helper.",
                },
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
                "tests_run": [
                    "python -m pytest tests/test_agent_loop.py -k coder_followup"
                ],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=2,
            text="Rename the shared helper.",
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert rendered == (
        "## Coder follow-up\n\n"
        "Added the requested regression test.\n\n"
        "### Addressed items\n"
        "- item-1: Blocking issue from OpenAI Codex, round 1: Add a regression test before merge.\n"
        "  - Resolution: Added coverage for the parser.\n"
        "- item-2: Same-PR follow-up from Google Gemini, round 2: Rename the shared helper.\n"
        "  - Resolution: Updated the helper.\n\n"
        "### Remaining items\n"
        "- None.\n\n"
        "### Tests run\n"
        "- python -m pytest tests/test_agent_loop.py -k coder_followup\n\n"
        "<!-- AGENT_STATE: blocking -->\n\n"
        "-- Anthropic Claude"
    )
    assert "```json" not in rendered
    assert '"kind": "coder_followup"' not in rendered

    without_tests = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Still working through the review.",
                "addressed_items": [],
                "remaining_items": ["item-3"],
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
                "tests_run": [],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert without_tests is not None
    rendered_without_tests = _render_public_coder_followup_comment(
        without_tests,
        agent="Claude",
    )
    assert "### Tests run" not in rendered_without_tests
    assert "### Addressed items\n- None." in rendered_without_tests
    assert (
        "### Remaining items\n"
        "- item-3: Item context unavailable in current round metadata.\n"
        "  - Reason: No reason provided by coder."
    ) in rendered_without_tests


def test_render_public_coder_followup_comment_expands_carried_items_with_notes_and_placeholders():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Fixed the blocker and deferred the follow-up.",
                "addressed_items": ["item-1"],
                "remaining_items": ["item-2"],
                "addressed_item_notes": {"item-1": "Restored the missing validation branch."},
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=2,
            text="  - Preserve structured coder follow-up metadata.\n\nExtra context should be summarized.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=3,
            text="Move the rendering helper into a shared module.",
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert (
        "- item-1: Blocking issue from OpenAI Codex, round 2: "
        "Preserve structured coder follow-up metadata."
    ) in rendered
    assert "  - Resolution: Restored the missing validation branch." in rendered
    assert (
        "- item-2: Same-PR follow-up from Google Gemini, round 3: "
        "Move the rendering helper into a shared module."
    ) in rendered
    assert "  - Reason: No reason provided by coder." in rendered


def test_render_public_coder_followup_comment_expands_pr_220_remaining_items():
    parsed = validate_structured_coder_followup(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Hardened markdown stripping; two follow-ups remain.",
                "addressed_items": ["item-3", "item-4"],
                "remaining_items": ["item-5", "item-6"],
                "remaining_item_notes": {
                    "item-5": "Deferred because URL canonicalization needs product confirmation.",
                    "item-6": "Deferred because the helper move should be isolated from this fix.",
                },
                "human_requirements": {
                    "addressed_ids": [],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-5",
            reviewer="Google Gemini",
            source_round=3,
            text=(
                "Update `server/static/index.html` and `server/static/landing.html` to use "
                "relative paths for `og:image` and `og:url` if possible."
            ),
            status="same-pr",
        ),
        UnresolvedReviewItem(
            item_id="item-6",
            reviewer="Google Gemini",
            source_round=3,
            text=(
                "Deduplicate `_strip_markdown` helper logic between `server/app.py` and "
                "`core/orchestrator.py` by moving it to `core/utils.py`."
            ),
            status="same-pr",
        ),
    )

    rendered = _render_public_coder_followup_comment(
        parsed,
        agent="Claude",
        prior_items=prior_items,
    )

    assert "- item-5: Same-PR follow-up from Google Gemini, round 3:" in rendered
    assert "relative paths" in rendered
    assert "  - Reason: Deferred because URL canonicalization needs product confirmation." in rendered
    assert "- item-6: Same-PR follow-up from Google Gemini, round 3:" in rendered
    assert "Deduplicate `_strip_markdown` helper logic" in rendered
    assert "  - Reason: Deferred because the helper move should be isolated from this fix." in rendered
    assert "\n- item-5\n" not in rendered
    assert "\n- item-6\n" not in rendered


def test_render_public_plan_review_comment_normalizes_sections():
    parsed = parse_structured_plan_review(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": "blocking",
                "summary": "Still blocked on coverage.",
                "blocking_plan_issues": ["Add a resume coverage test."],
                "same_plan_followups": ["Mention canonical hashing explicitly."],
                "future_followups": [],
                "prior_plan_item_dispositions": [
                    {"item_id": "item-2", "disposition": "same-plan", "note": "Still needs one more prompt assertion."}
                ],
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        reviewer="OpenAI Codex",
    )
    assert parsed is not None
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="Google Gemini",
            source_round=1,
            text="Mention canonical hashing explicitly.",
            status="same-plan",
        ),
    )

    rendered = _render_public_plan_review_comment(
        parsed,
        reviewer="OpenAI Codex",
        prior_items=prior_items,
        dispositions=parsed.dispositions,
    )

    assert rendered == (
        "**Review verdict:** Blocking\n\n"
        "Still blocked on coverage.\n\n"
        "### Blocking plan issues\n"
        "- Add a resume coverage test.\n\n"
        "### Same-plan follow-ups\n"
        "- Mention canonical hashing explicitly.\n\n"
        "### Prior unresolved plan item dispositions\n"
        "- [item-2] Same-plan follow-up from Google Gemini, round 1: Mention canonical hashing explicitly. -> "
        "same-plan: Still needs one more prompt assertion.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )


def test_review_freeform_summary_text_strips_structured_followup_sections():
    review = """**Review verdict:** blocking

Blocking issue summary.

### Blocking issues
- needs one more assertion

### Prior unresolved item dispositions
- [item-1] still blocking: needs one more assertion

### Human requirements
- Requirement 1: addressed in the latest patch

### Same-PR follow-ups
- Rename helper

### Future follow-ups
- Document cleanup later

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""

    assert _review_freeform_summary_text(review) == "Blocking issue summary."


def test_render_public_pr_review_comment_uses_normalized_sections_and_footer():
    parsed = parse_review(
        (
            "Need one more regression test."
            + blocking_issues("Exercise the structured-resume path.")
            + "\n\n### Same-PR follow-ups\n- Rename the helper for clarity."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ),
        reviewer="OpenAI Codex",
    )
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
    )

    rendered = _render_public_pr_review_comment(
        parsed,
        reviewer="Codex",
        human_requirements_resolved_flag=True,
        prior_items=prior_items,
        dispositions=parsed.dispositions,
    )

    assert rendered == (
        "**Review verdict:** Blocking\n\n"
        "Need one more regression test.\n\n"
        "### Blocking issues\n"
        "- Exercise the structured-resume path.\n\n"
        "### Same-PR follow-ups\n"
        "- Rename the helper for clarity.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] Blocking issue from Anthropic Claude, round 1: Add a regression test before merge. -> resolved\n\n"
        "<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )


def test_render_public_pr_review_comment_normalizes_markdown_and_structured_reviews_the_same():
    markdown_review = (
        "Need one more regression test."
        + blocking_issues("Exercise the structured-resume path.")
        + "\n\n### Same-PR follow-ups\n- Rename the helper for clarity."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    structured_review = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": "blocking",
                "summary": "Need one more regression test.",
                "blocking_items": ["Exercise the structured-resume path."],
                "same_pr_followups": ["Rename the helper for clarity."],
                "future_followups": [],
                "prior_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
    )
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Anthropic Claude",
            source_round=1,
            text="Add a regression test before merge.",
            status="blocking",
        ),
    )

    markdown_rendered = _render_public_pr_review_comment(
        parse_review(markdown_review, reviewer="OpenAI Codex"),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=prior_items,
        dispositions=parse_review(markdown_review, reviewer="OpenAI Codex").dispositions,
    )
    structured_parsed = parse_pr_review(structured_review, reviewer="OpenAI Codex")
    structured_rendered = _render_public_pr_review_comment(
        structured_parsed,
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=prior_items,
        dispositions=structured_parsed.dispositions,
    )

    assert markdown_rendered == structured_rendered


def test_render_public_pr_review_comment_includes_visible_approved_verdict():
    rendered = _render_public_pr_review_comment(
        parse_review(
            "Looks good to me.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            reviewer="OpenAI Codex",
        ),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=(),
        dispositions=(),
    )

    assert rendered == (
        "**Review verdict:** Approved\n\n"
        "Looks good to me.\n\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex"
    )


def test_format_unresolved_item_label_normalizes_multiline_text_and_preserves_origin_status():
    item = UnresolvedReviewItem(
        item_id="item-7",
        reviewer="Google Gemini",
        source_round=1,
        text="  - require source issue reference in PR body  \n\nUpdate from Anthropic Claude: keep the wording compact",
        status="resolved",
        source_status="same-pr",
    )

    assert _format_unresolved_item_label(item) == (
        "Same-PR follow-up from Google Gemini, round 1: require source issue reference in PR body"
    )


def test_format_unresolved_item_label_truncates_at_fixed_limit():
    summary = "a" * (ITEM_SUMMARY_LIMIT + 20)
    item = UnresolvedReviewItem(
        item_id="item-8",
        reviewer="OpenAI Codex",
        source_round=2,
        text=summary,
        status="blocking",
    )

    label = _format_unresolved_item_label(item)

    assert label.startswith("Blocking issue from OpenAI Codex, round 2: ")
    assert label.endswith("...")
    rendered_summary = label.split(": ", 1)[1]
    assert len(rendered_summary) == ITEM_SUMMARY_LIMIT


def test_format_unresolved_item_label_special_cases_human_requirements_ack_item():
    item = UnresolvedReviewItem(
        item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
        reviewer="Orchestrator",
        source_round=3,
        text="Coder response missing required `### Human requirements` section.",
        status="blocking",
    )

    assert _format_unresolved_item_label(item) == (
        "Human-requirements acknowledgement item, round 3: "
        "Coder response missing required `### Human requirements` section."
    )


def test_render_public_review_comment_replaces_dispositions_without_exposing_same_round_new_items():
    body = """Still blocked.

### Same-PR follow-ups
- Keep the source issue reference in the PR body.

### Prior unresolved item dispositions
- [item-1] same-pr

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Google Gemini",
            source_round=1,
            text="Require source issue reference in PR body.\n\nUpdate from Anthropic Claude: keep the note compact",
            status="same-pr",
        ),
    )
    dispositions = parse_unresolved_item_dispositions(
        prior_item_dispositions("[item-1] Same-PR follow-up from Google Gemini, round 1: ignored by parser -> same-pr: keep the body reference"),
        reviewer="OpenAI Codex",
    )
    new_items = (
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=2,
            text="Keep the source issue reference in the PR body.",
            status="same-pr",
        ),
    )

    rendered = _render_public_review_comment(
        body,
        review_kind="pr",
        prior_items=prior_items,
        dispositions=dispositions,
        new_items=new_items,
    )

    assert "### Same-PR follow-ups\n- Keep the source issue reference in the PR body." in rendered
    assert (
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from Google Gemini, round 1: Require source issue reference in PR body. -> same-pr: keep the body reference"
    ) in rendered
    assert "### New tracked unresolved items" not in rendered
    assert "[item-2]" not in rendered
    assert rendered.rstrip().endswith("-- OpenAI Codex")


def test_render_public_review_comment_preserves_unknown_disposition_values():
    body = """Still blocked.

### Prior unresolved item dispositions
- [item-1] same-pr

<!-- AGENT_STATE: blocking -->
-- OpenAI Codex
"""
    prior_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="Google Gemini",
            source_round=1,
            text="Keep the parser and renderer aligned when new dispositions are added.",
            status="same-pr",
        ),
    )
    dispositions = (
        ReviewItemDisposition(
            item_id="item-1",
            reviewer="OpenAI Codex",
            disposition="deferred",
            note="tracked for a later parser update",
        ),
    )

    rendered = _render_public_review_comment(
        body,
        review_kind="pr",
        prior_items=prior_items,
        dispositions=dispositions,
        new_items=(),
    )

    assert (
        "### Prior unresolved item dispositions\n"
        "- [item-1] Same-PR follow-up from Google Gemini, round 1: "
        "Keep the parser and renderer aligned when new dispositions are added. "
        "-> deferred: tracked for a later parser update"
    ) in rendered


# --- render_discuss_round_summary_comment tests ---


def _discuss_vote(
    outcome: str = "implement",
    rationale: str = "Good scope.",
    proposals: tuple[str, ...] = (),
    reviewer: str = "Gemini",
    rebuttal: str | None = None,
) -> ParsedDiscussReview:
    return ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=proposals,
        reviewer=reviewer,
        rebuttal=rebuttal,
    )


def test_render_discuss_round_summary_comment_final_implement_heading():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        reviewer_votes=[_discuss_vote("implement")],
        split_proposals=[],
        subject="abc123",
    )
    assert "## Consensus: Implement" in rendered


def test_render_discuss_round_summary_comment_final_do_not_implement_heading():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="do-not-implement",
        reviewer_votes=[_discuss_vote("do-not-implement", rationale="Out of scope.")],
        split_proposals=[],
        subject="deadbeef",
    )
    assert "## Consensus: Do Not Implement" in rendered


def test_render_discuss_round_summary_comment_final_needs_human_heading():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="needs-human",
        reviewer_votes=[_discuss_vote("needs-human", rationale="Unclear requirements.")],
        split_proposals=[],
        subject="cafe1234",
    )
    assert "## Consensus: Needs Human Review" in rendered


def test_render_discuss_round_summary_comment_final_split_heading_and_proposals():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="split",
        reviewer_votes=[_discuss_vote("split", proposals=("Sub A", "Sub B"))],
        split_proposals=["Sub A", "Sub B"],
        subject="f00dbeef",
    )
    assert "## Consensus: Split" in rendered
    assert "### Proposed sub-issues" in rendered
    assert "- Sub A" in rendered
    assert "- Sub B" in rendered


def test_render_discuss_round_summary_comment_final_reviewer_table_row():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        reviewer_votes=[
            _discuss_vote("implement", rationale="Well-scoped.", reviewer="Gemini"),
            _discuss_vote("implement", rationale="Clear value.", reviewer="OpenAI Codex"),
        ],
        split_proposals=[],
        subject="abc123",
    )
    assert "| Gemini |" in rendered
    assert "| OpenAI Codex |" in rendered
    assert "Well-scoped." in rendered
    assert "Clear value." in rendered


def test_render_discuss_round_summary_comment_final_orchestrator_footer():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        reviewer_votes=[_discuss_vote()],
        split_proposals=[],
        subject="abc123",
    )
    assert "-- Orchestrator" in rendered


def test_render_discuss_round_summary_comment_final_marker_last_line():
    subject = "deadbeef1234"
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        reviewer_votes=[_discuss_vote()],
        split_proposals=[],
        subject=subject,
    )
    assert rendered.endswith(f"<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->")


def test_render_discuss_round_summary_comment_final_converged_kind_and_rebuttals():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="converged",
        round_number=2,
        reviewer_votes=[
            _discuss_vote(
                "implement",
                reviewer="Gemini",
                rebuttal="The scope concern is resolved by the issue body.",
            )
        ],
        round_history=[
            [_discuss_vote("needs-human", reviewer="Gemini")],
            [
                _discuss_vote(
                    "implement",
                    reviewer="Gemini",
                    rebuttal="The scope concern is resolved by the issue body.",
                )
            ],
        ],
        split_proposals=[],
        subject="abc123",
    )
    assert "Consensus kind: `converged` after round 2." in rendered
    assert "### Final rebuttals" in rendered
    assert "Round 1: Gemini: `needs-human`" in rendered


def test_render_discuss_round_summary_comment_final_deadlock_summarizes_disagreement():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="needs-human",
        consensus_kind="deadlock",
        round_number=2,
        reviewer_votes=[
            _discuss_vote("implement", rationale="Scoped.", reviewer="Gemini"),
            _discuss_vote("do-not-implement", rationale="Out of scope.", reviewer="OpenAI Codex"),
        ],
        split_proposals=[],
        subject="abc123",
    )
    assert "## Consensus: Needs Human Review (Deadlock)" in rendered
    assert "Consensus kind: `deadlock` after round 2." in rendered
    assert "### Core disagreement" in rendered
    assert "Gemini held `implement`: Scoped." in rendered
    assert "OpenAI Codex held `do-not-implement`: Out of scope." in rendered


def test_render_discuss_round_summary_comment_interim_round_has_agenda_and_no_marker():
    rendered = render_discuss_round_summary_comment(
        is_final=False,
        round_number=1,
        reviewer_votes=[
            _discuss_vote("implement", rationale="Scoped.", reviewer="Gemini"),
            _discuss_vote("do-not-implement", rationale="Out of scope.", reviewer="OpenAI Codex"),
        ],
        split_proposals=[],
        subject="abc123",
    )
    assert "## Round 1 summary: Consensus Pending" in rendered
    assert "### Agenda for round 2" in rendered
    assert "Gemini held `implement`: Scoped." in rendered
    assert "OpenAI Codex held `do-not-implement`: Out of scope." in rendered
    assert "-- Orchestrator" in rendered
    assert "AGENT_DISCUSS_CONSENSUS" not in rendered
    assert "Consensus:" not in rendered


def test_render_discuss_round_summary_comment_interim_round_lists_split_proposals():
    rendered = render_discuss_round_summary_comment(
        is_final=False,
        round_number=1,
        reviewer_votes=[_discuss_vote("split", proposals=("Sub A",), reviewer="Gemini")],
        split_proposals=["Sub A"],
        subject="abc123",
    )
    assert "### Proposed sub-issues raised this round" in rendered
    assert "- Sub A" in rendered


def test_render_discuss_round_summary_comment_final_includes_full_resumed_history():
    """A resumed final summary must list every prior round, not just the last one."""
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="converged",
        round_number=3,
        reviewer_votes=[
            _discuss_vote("implement", reviewer="Gemini"),
            _discuss_vote("implement", reviewer="OpenAI Codex"),
        ],
        round_history=[
            [
                _discuss_vote("do-not-implement", reviewer="Gemini"),
                _discuss_vote("implement", reviewer="OpenAI Codex"),
            ],
            [
                _discuss_vote("do-not-implement", reviewer="Gemini"),
                _discuss_vote("implement", reviewer="OpenAI Codex"),
            ],
            [
                _discuss_vote("implement", reviewer="Gemini"),
                _discuss_vote("implement", reviewer="OpenAI Codex"),
            ],
        ],
        split_proposals=[],
        subject="abc123",
    )
    assert "Round 1: Gemini: `do-not-implement`, OpenAI Codex: `implement`" in rendered
    assert "Round 2: Gemini: `do-not-implement`, OpenAI Codex: `implement`" in rendered
    assert "Round 3: Gemini: `implement`, OpenAI Codex: `implement`" in rendered


def test_render_public_discuss_review_comment_includes_vote_and_signature():
    vote = _discuss_vote("implement", rationale="Well-scoped.", reviewer="Codex")
    rendered = _render_public_discuss_review_comment(vote, reviewer="Codex", round_number=1)
    assert "## Round 1: Codex position" in rendered
    assert "**Vote:** Implement (`implement`)" in rendered
    assert "Well-scoped." in rendered
    assert rendered.strip().endswith("Codex")


def test_render_public_discuss_review_comment_includes_rebuttal_and_split_proposals():
    vote = _discuss_vote(
        "split",
        rationale="Too broad.",
        proposals=("Auth flow",),
        reviewer="Codex",
        rebuttal="I still think it should be split.",
    )
    rendered = _render_public_discuss_review_comment(vote, reviewer="Codex", round_number=2)
    assert "## Round 2: Codex position" in rendered
    assert "### Rebuttal" in rendered
    assert "I still think it should be split." in rendered
    assert "### Proposed sub-issues" in rendered
    assert "- Auth flow" in rendered



# --- analyzer agenda rendering tests (#467) ---

from coding_review_agent_loop.protocol import (
    DiscussAgendaDisagreement,
    ParsedDiscussAgenda,
)


def _rendering_agenda(
    *,
    consensus: tuple[str, ...] = ("The issue is well-motivated.",),
    missing_facts: tuple[str, ...] = ("Whether the API boundary is specified.",),
) -> ParsedDiscussAgenda:
    return ParsedDiscussAgenda(
        consensus=consensus,
        disagreements=(
            DiscussAgendaDisagreement(
                topic="Scope of the change",
                positions=(("Codex", "Narrow enough."), ("Gemini", "Too broad; split it.")),
                question_for_next_round="Would splitting resolve the scope objection?",
            ),
        ),
        missing_facts=missing_facts,
    )


def test_render_discuss_summary_non_final_uses_analyzer_agenda_with_attribution():
    rendered = render_discuss_round_summary_comment(
        is_final=False,
        subject="abc123",
        round_number=1,
        reviewer_votes=[
            _discuss_vote("implement", rationale="Mechanical rationale.", reviewer="Codex"),
            _discuss_vote("do-not-implement", rationale="Other rationale.", reviewer="Gemini"),
        ],
        analyzer_agenda=_rendering_agenda(),
        analyzer_name="Anthropic Claude",
    )
    assert "### Agenda for round 2 (analyzer: Anthropic Claude)" in rendered
    assert "Analyzer-extracted consensus so far (not debater-confirmed):" in rendered
    assert "- The issue is well-motivated." in rendered
    assert "**Scope of the change**" in rendered
    assert "Codex: Narrow enough." in rendered
    assert "Gemini: Too broad; split it." in rendered
    assert "Question for next round: Would splitting resolve the scope objection?" in rendered
    assert "Missing facts:" in rendered
    assert "- Whether the API boundary is specified." in rendered
    # The mechanical per-vote agenda lines are replaced by the analyzer agenda.
    assert "- Codex held `implement`: Mechanical rationale." not in rendered


def test_render_discuss_summary_non_final_without_agenda_keeps_mechanical_lines():
    rendered = render_discuss_round_summary_comment(
        is_final=False,
        subject="abc123",
        round_number=1,
        reviewer_votes=[_discuss_vote("implement", rationale="Mechanical rationale.", reviewer="Codex")],
    )
    assert "### Agenda for round 2" in rendered
    assert "analyzer" not in rendered
    assert "- Codex held `implement`: Mechanical rationale." in rendered


def test_render_discuss_summary_final_distinguishes_analyzer_consensus_from_votes():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="needs-human",
        consensus_kind="deadlock",
        subject="abc123",
        round_number=3,
        reviewer_votes=[
            _discuss_vote("implement", reviewer="Codex"),
            _discuss_vote("do-not-implement", reviewer="Gemini"),
        ],
        split_proposals=[],
        analyzer_agenda=_rendering_agenda(),
        analyzer_name="Anthropic Claude",
    )
    assert (
        "### Analyzer-extracted consensus (analyzer: Anthropic Claude; not debater-confirmed)"
        in rendered
    )
    assert "The debater vote table above is the authoritative consensus." in rendered
    assert "| Codex |" in rendered
    assert "| Gemini |" in rendered
    # The analyzer section comes after the authoritative vote table.
    assert rendered.index("| Codex |") < rendered.index("Analyzer-extracted consensus")


def test_render_discuss_summary_final_without_agenda_has_no_analyzer_section():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[_discuss_vote("implement", reviewer="Codex")],
        split_proposals=[],
    )
    assert "Analyzer-extracted consensus" not in rendered


def test_render_discuss_summary_final_empty_agenda_renders_placeholder():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="needs-human",
        consensus_kind="deadlock",
        subject="abc123",
        reviewer_votes=[_discuss_vote("implement", reviewer="Codex")],
        split_proposals=[],
        analyzer_agenda=ParsedDiscussAgenda(consensus=(), disagreements=(), missing_facts=()),
    )
    assert "### Analyzer-extracted consensus (not debater-confirmed)" in rendered
    assert "(the analyzer extracted no points)" in rendered


def test_render_public_discuss_comment_renders_misframed_correction():
    parsed = ParsedDiscussReview(
        outcome="implement",
        rationale="Still well-scoped.",
        split_proposals=(),
        reviewer="Codex",
        rebuttal="Engages the agenda.",
        analyzer_framing="misframed",
        framing_note="The agenda claims I opposed the feature; I only questioned scope.",
    )
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=2)
    assert "### Analyzer framing correction" in rendered
    assert "The agenda claims I opposed the feature; I only questioned scope." in rendered


def test_render_public_discuss_comment_renders_accurate_framing_line():
    parsed = ParsedDiscussReview(
        outcome="implement",
        rationale="Still well-scoped.",
        split_proposals=(),
        reviewer="Codex",
        rebuttal="Engages the agenda.",
        analyzer_framing="accurate",
    )
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=2)
    assert "**Analyzer framing:** accurate" in rendered
    assert "### Analyzer framing correction" not in rendered


def test_render_public_discuss_comment_without_framing_is_unchanged():
    parsed = ParsedDiscussReview(
        outcome="implement",
        rationale="Well-scoped.",
        split_proposals=(),
        reviewer="Codex",
    )
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=1)
    assert "Analyzer" not in rendered


# --- discuss research policy rendering tests (#477) ---

from coding_review_agent_loop.protocol import DiscussSourcedFact


def _discuss_research_vote(
    *,
    reviewer: str,
    outcome: str = "implement",
    research_status: str | None = None,
    sourced_facts: tuple[DiscussSourcedFact, ...] = (),
) -> ParsedDiscussReview:
    return ParsedDiscussReview(
        outcome=outcome,
        rationale="Good scope.",
        split_proposals=(),
        reviewer=reviewer,
        research_status=research_status,
        sourced_facts=sourced_facts,
    )


def test_render_public_discuss_comment_renders_sourced_facts():
    parsed = _discuss_research_vote(
        reviewer="Codex",
        research_status="sourced",
        sourced_facts=(
            DiscussSourcedFact(
                fact="Gemini CLI remains available for enterprise users.",
                source="https://example.com/gemini-cli-notice",
            ),
        ),
    )
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=1)
    assert "**Research:** done, sourced facts cited below (`sourced`)" in rendered
    assert "### Sourced facts" in rendered
    assert (
        "- Gemini CLI remains available for enterprise users. — source: "
        "https://example.com/gemini-cli-notice" in rendered
    )


def test_render_public_discuss_comment_renders_unavailable_status_without_facts():
    parsed = _discuss_research_vote(reviewer="Codex", research_status="unavailable")
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=1)
    assert "**Research:**" in rendered
    assert "`unavailable`" in rendered
    assert "### Sourced facts" not in rendered


def test_render_public_discuss_comment_without_research_is_unchanged():
    parsed = _discuss_research_vote(reviewer="Codex")
    rendered = _render_public_discuss_review_comment(parsed, reviewer="Codex", round_number=1)
    assert "Research" not in rendered
    assert "Sourced facts" not in rendered


def test_render_discuss_summary_final_default_has_no_research_section():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[_discuss_vote("implement", reviewer="Codex")],
        split_proposals=[],
    )
    assert "### Research" not in rendered


def test_render_discuss_summary_final_research_none_notes_disabled():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[_discuss_vote("implement", reviewer="Codex")],
        split_proposals=[],
        research_mode="none",
    )
    assert "### Research" in rendered
    assert "Research policy: `none`." in rendered
    assert "Online research was disabled; all positions are agent judgment." in rendered


def test_render_discuss_summary_final_research_distinguishes_facts_from_judgment():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[
            _discuss_research_vote(
                reviewer="Codex",
                research_status="sourced",
                sourced_facts=(
                    DiscussSourcedFact(
                        fact="Gemini CLI remains available.",
                        source="https://example.com/notice",
                    ),
                ),
            ),
            _discuss_research_vote(reviewer="Gemini", research_status="unavailable"),
        ],
        split_proposals=[],
        research_mode="required",
    )
    assert "### Research" in rendered
    assert "Research policy: `required`." in rendered
    assert "- Codex: done, sourced facts cited below (`sourced`)" in rendered
    assert "- Gemini: unavailable — related claims are judgment, not sourced fact (`unavailable`)" in rendered
    assert (
        "Sourced facts cited by debaters (everything else above is agent judgment):"
        in rendered
    )
    assert "- Codex: Gemini CLI remains available. — source: https://example.com/notice" in rendered
    assert (
        "Research was unavailable or inconclusive for Gemini; treat their related "
        "claims as judgment, not sourced fact." in rendered
    )


def test_render_discuss_summary_final_research_auto_all_not_needed_is_explicit():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[
            _discuss_research_vote(reviewer="Codex", research_status="not-needed"),
            _discuss_research_vote(reviewer="Gemini", research_status="not-needed"),
        ],
        split_proposals=[],
        research_mode="auto",
    )
    assert "Research policy: `auto`." in rendered
    assert (
        "All debaters determined external research was unnecessary for this question."
        in rendered
    )


def test_render_discuss_summary_final_research_unreported_status_is_explicit():
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="unanimous",
        subject="abc123",
        reviewer_votes=[
            _discuss_research_vote(reviewer="Codex", research_status="not-needed"),
            _discuss_research_vote(reviewer="Gemini"),
        ],
        split_proposals=[],
        research_mode="auto",
    )
    assert "- Gemini: no research status reported" in rendered
    assert (
        "No research status was reported by Gemini; treat their claims as judgment, "
        "not sourced fact." in rendered
    )
    assert "All debaters determined external research was unnecessary" not in rendered


def test_render_discuss_summary_final_research_aggregates_facts_across_rounds():
    round1_codex = _discuss_research_vote(
        reviewer="Codex",
        research_status="sourced",
        sourced_facts=(
            DiscussSourcedFact(fact="Round-one fact.", source="https://example.com/r1"),
        ),
    )
    round2_codex = _discuss_research_vote(reviewer="Codex", research_status="not-needed")
    rendered = render_discuss_round_summary_comment(
        is_final=True,
        outcome="implement",
        consensus_kind="converged",
        subject="abc123",
        round_number=2,
        reviewer_votes=[round2_codex],
        round_history=[[round1_codex], [round2_codex]],
        split_proposals=[],
        research_mode="required",
    )
    # Facts cited in earlier rounds stay visible in the final summary.
    assert "- Codex: Round-one fact. — source: https://example.com/r1" in rendered


def test_render_discuss_summary_non_final_agenda_includes_research_brief():
    agenda = ParsedDiscussAgenda(
        consensus=(),
        disagreements=(),
        missing_facts=(),
        research_required=True,
        research_questions=("Is Gemini CLI still available for enterprise users?",),
    )
    rendered = render_discuss_round_summary_comment(
        is_final=False,
        subject="abc123",
        round_number=1,
        reviewer_votes=[
            _discuss_vote("implement", reviewer="Codex"),
            _discuss_vote("do-not-implement", reviewer="Gemini"),
        ],
        analyzer_agenda=agenda,
        analyzer_name="Anthropic Claude",
        research_mode="auto",
    )
    assert "Research brief for the next round (answer with cited sources):" in rendered
    assert "- Is Gemini CLI still available for enterprise users?" in rendered
