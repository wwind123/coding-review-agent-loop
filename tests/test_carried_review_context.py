from agent_loop_helpers import *  # noqa: F403
from coding_review_agent_loop.unresolved_items import (
    _apply_unresolved_item_dispositions,
    _format_unresolved_items_for_coder,
    _validate_review_response,
)


CLAIM = "Skill-mode reviewers must receive the PR-bound approved plan."
SUMMARY = "Recovery still picks the first loose issue reference and hides comment-read failures."
NOTE = "helpers/skill_runner.py selects the first issue; use the primary-issue contract and test multiple references."


def carried_item():
    return UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1,
        text=CLAIM, status="blocking", source_status="blocking",
    )


def carried_review(*, disposition="blocking", note=None):
    item = {"item_id": "item-1", "disposition": disposition}
    if note is not None:
        item["note"] = note
    return structured_pr_review(
        state="approved" if disposition == "resolved" else "blocking",
        summary=SUMMARY,
        prior_item_dispositions=[item],
    )


@pytest.mark.parametrize("disposition", ["blocking", "same-pr"])
@pytest.mark.parametrize("note", [None, "still blocking", "unresolved"])
def test_live_carried_item_requires_explanation(disposition, note):
    with pytest.raises(AgentLoopError, match="requires an actionable note"):
        _validate_review_response(
            carried_review(disposition=disposition, note=note),
            reviewer="Codex", unresolved_items=(carried_item(),),
        )


@pytest.mark.parametrize("note", ["", "   "])
def test_live_carried_item_rejects_blank_note(note):
    with pytest.raises(AgentLoopError):
        _validate_review_response(
            carried_review(note=note), reviewer="Codex", unresolved_items=(carried_item(),),
        )


def test_resolved_note_optional_and_historical_blocker_still_readable():
    _validate_review_response(
        carried_review(disposition="resolved"), reviewer="Codex", unresolved_items=(carried_item(),),
    )
    old = parse_pr_review(carried_review(), reviewer="Codex")
    items, _ = _apply_unresolved_item_dispositions(
        (carried_item(),), {"item-1": list(old.dispositions)},
    )
    assert items[0].text == CLAIM
    assert "without an item-specific explanation" in _format_unresolved_items_for_coder(items)


def test_note_survives_publication_metadata_and_reconciliation():
    parsed = _validate_review_response(
        carried_review(note=NOTE), reviewer="Codex", unresolved_items=(carried_item(),),
    )
    public = _render_public_pr_review_comment(
        parsed, reviewer="Codex", prior_items=(carried_item(),), dispositions=parsed.dispositions,
        human_requirements_resolved_flag=False,
    )
    saved = _attach_round_metadata(public, PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=2, subject="abc123",
        prior_items=(carried_item(),), dispositions=parsed.dispositions, state="blocking",
    ))
    reparsed = parse_review(saved, reviewer="Codex")
    assert reparsed.dispositions[0].note == NOTE
    items, _ = _apply_unresolved_item_dispositions(
        (carried_item(),), {"item-1": list(reparsed.dispositions)},
    )
    assert items[0].text == CLAIM
    assert NOTE in _format_unresolved_items_for_coder(items)
    assert SUMMARY not in items[0].text


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("context", ["full", "compact"])
@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("disposition", ["blocking", "same-pr"])
def test_coder_receives_latest_summary_separately_from_carried_claim(tmp_path, parallel, context, resume, disposition):
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=2, subject="abc123",
            prior_items=(carried_item(),),
        ),
    )
    comments = [{"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment}]
    if resume:
        # Historical reviews with no note must not be subjected to live validation.
        review = carried_review(disposition=disposition)
        saved = _attach_round_metadata(review, PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=2, subject="abc123",
            prior_items=(carried_item(),), dispositions=parse_pr_review(review, reviewer="Codex").dispositions,
            state="blocking",
        ))
        comments.append({"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": saved})
    approved = carried_review(disposition="resolved")
    runner = FakeRunner(
        pr_payload={"comments": comments},
        codex_outputs=([carried_review(disposition=disposition, note=NOTE)] if not resume else []) + [approved],
        gemini_outputs=[approved, approved],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=parallel,
        pr_review_context_mode=context, approved_followups="fix-and-summarize", max_rounds=3,
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    prompt = next(cmd[-1] for cmd, _ in runner.commands if cmd[:1] == ["claude"])
    assert "Latest reviewer summaries (review-level context):" in prompt
    assert SUMMARY in prompt
    assert "Codex (round 2, head abc123):" in prompt
    assert CLAIM in prompt
    assert "[item-2] from round" not in prompt
    if resume:
        assert "without an item-specific explanation" in prompt
    else:
        assert NOTE in prompt
        reviewer_prompt = next(cmd[-1] for cmd, _ in runner.commands if cmd[:1] == ["codex"])
        assert "requires an actionable `note`" in reviewer_prompt
    heading = "Codex unresolved blocking item" if disposition == "blocking" else "Codex same-PR follow-up"
    ledger_text = prompt.split(heading, 1)[1]
    assert SUMMARY not in ledger_text
