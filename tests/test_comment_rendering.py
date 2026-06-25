"""Comment rendering tests extracted from test_agent_loop.py (lines 5180–5565 and 5856–6115).

Tests for render_canonical_plan_steps, render_canonical_plan_revision,
render_public_agent_comment, _render_public_*_comment, and related functions.
"""
import json

import pytest

from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_plan_review_comment,
    _render_public_plan_revision_comment,
    _render_public_pr_review_comment,
    normalize_freeform_signature,
)
from coding_review_agent_loop.orchestrator import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    ITEM_SUMMARY_LIMIT,
    _format_unresolved_item_label,
    _render_public_review_comment,
    _review_freeform_summary_text,
    render_canonical_plan_revision,
    render_canonical_plan_steps,
    render_public_agent_comment,
)
from coding_review_agent_loop.protocol import (
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

def test_render_canonical_plan_steps_numbers_items():
    assert render_canonical_plan_steps(("Update protocol.py.", "Add tests.")) == (
        "1. Update protocol.py.\n2. Add tests."
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


