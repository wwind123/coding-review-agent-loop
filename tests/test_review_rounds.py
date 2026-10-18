"""Review-round tests: incomplete plan-review detection (#1543)."""

from types import SimpleNamespace

import pytest
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.protocol import (
    ApprovedFollowup,
    ParsedPlanReview,
    PlanReviewItems,
    ReviewItemDisposition,
    UnresolvedReviewItem,
)
import coding_review_agent_loop.review_rounds as review_rounds
from agent_loop_helpers import structured_plan_review


@pytest.mark.parametrize(
    ("state", "blocking", "same_plan", "summary", "expected"),
    [
        ("blocking", (), (), "Unable to verify the plan without more context.", True),
        (
            "blocking",
            (ApprovedFollowup(reviewer="Codex", text="Add a migration step."),),
            (),
            "Unable to verify the plan without more context.",
            False,
        ),
        ("approved", (), (), "Unable to verify the plan without more context.", False),
    ],
)
def test_is_incomplete_plan_review(state, blocking, same_plan, summary, expected):
    parsed_review = ParsedPlanReview(
        state=state,
        summary=summary,
        items=PlanReviewItems(blocking=blocking, same_plan=same_plan, future=()),
        dispositions=(),
    )
    assert review_rounds._is_incomplete_plan_review(parsed_review) is expected


def test_is_incomplete_plan_review_matches_disposition_note():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="still blocking",
                note="Resolution could not be confirmed; plan and referenced files need review.",
            ),
        ),
    )
    assert review_rounds._is_incomplete_plan_review(parsed_review) is True


def test_is_incomplete_plan_review_allows_active_carried_disposition():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="blocking",
                note="Could not confirm the revised plan covers migration ordering.",
            ),
        ),
    )

    assert review_rounds._is_incomplete_plan_review(parsed_review) is False


def test_round_resolved_history_ids_available_when_plan_ledger_is_complete():
    # #872 follow-up: a lossless semantic patch needs resolved-history proof in
    # every ledger state, so the history is computed for complete rounds too.
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    mid_plan = "Mid plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item.",
        status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(state="blocking", blocking_plan_issues=["Old subject item."]),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(old_plan), new_items=(old_item,), state="blocking",
        ),
    )
    resolved = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(mid_plan), prior_items=(old_item,),
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "resolved"),),
            state="blocking",
        ),
    )
    comments = [SimpleNamespace(body=raised), SimpleNamespace(body=resolved)]

    assert review_rounds._round_resolved_history_item_ids(
        prior_unresolved_items=(),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-1",)


def test_post_round_resolved_history_ids_cover_the_round_still_in_progress():
    # #874: the snapshot a round holds was fetched before its review turns, so
    # it cannot record the dispositions that round just applied. The proof for
    # the revision therefore also reads the round's in-process dispositions.
    plan = "Current plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    blocking = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Migration ordering is unspecified.", status="blocking",
        source_status="blocking",
    )
    cleared = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="Rollback path is uncovered.", status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=[blocking.text, cleared.text],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(plan), new_items=(blocking, cleared), state="blocking",
        ),
    )
    # Only round 1 is in the snapshot; round 2's reviewer comment is posted
    # after it was fetched, exactly as the live loop sees it.
    comments = [SimpleNamespace(body=raised)]
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Still unspecified.")],
        "item-2": [ReviewItemDisposition("item-2", "OpenAI Codex", "resolved")],
    }

    assert review_rounds._round_resolved_history_item_ids(
        prior_unresolved_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()
    assert review_rounds._post_round_resolved_history_item_ids(
        prior_unresolved_items=(blocking, cleared),
        dispositions_by_item=dispositions,
        carried_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-2",)


def test_post_round_resolved_history_ids_exclude_actively_disputed_and_carried_items():
    # The whitelist stays fail-closed: a mixed verdict, a still-carried item and
    # an undispositioned item never earn resolved-history proof.
    carried = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Still blocking.", status="blocking", source_status="blocking",
    )
    mixed = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="One reviewer still blocks.", status="blocking", source_status="blocking",
    )
    untouched = UnresolvedReviewItem(
        item_id="item-3", reviewer="OpenAI Codex", source_round=1,
        text="Nobody dispositioned this.", status="blocking", source_status="blocking",
    )
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "resolved")],
        "item-2": [
            ReviewItemDisposition("item-2", "OpenAI Codex", "resolved"),
            ReviewItemDisposition("item-2", "Anthropic Claude", "blocking", "Not covered."),
        ],
    }

    assert review_rounds._post_round_resolved_history_item_ids(
        prior_unresolved_items=(carried, mixed, untouched),
        dispositions_by_item=dispositions,
        # item-1 survived reconciliation, so it is still carried.
        carried_items=(carried,),
        comments=[],
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()
