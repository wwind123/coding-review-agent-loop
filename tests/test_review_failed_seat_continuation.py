"""A blocked review round continues past one failed reviewer seat (#2129).

PR and plan rounds: when one seat fails (a deterministic rejection after its
re-asks, an unavailable seat, or an incomplete review) and another seat
blocked in the same round, the round continues to the coder or planner.  The
failed seat's carried items stay frozen (live, in the reconciliation record,
and through resume and history replay), the seat is never counted as
approving, and it is invoked again next round.  Every existing stop is kept
when no healthy seat blocked, for single-reviewer boards, quota resets,
evidence passes, the final allowed round and an empty must-fix set.
"""

from __future__ import annotations

import json

import pytest

import coding_review_agent_loop.discuss_loop as discuss_loop
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop, run_pr_loop
from coding_review_agent_loop.errors import QuotaResetExceededError
from coding_review_agent_loop.failed_seats import (
    failed_seat_continuation_allowed,
    force_select_failed_seats,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    ReviewSubItem,
    UnresolvedReviewItem,
)
from coding_review_agent_loop.review_scheduling import SchedulingDecision, TransitionClassification
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    PostedRoundRecord,
    _attach_round_metadata,
    _decode_round_metadata,
    bound_failed_reviewer_seats,
)
from coding_review_agent_loop.round_state import (
    _aggregate_record_dispositions,
    _canonically_resolved_history_item_ids,
    _recover_unrecorded_pr_head_advance,
    _replay_history_items,
)
from coding_review_agent_loop.unresolved_items import (
    _apply_unresolved_item_dispositions,
    preserve_failed_seat_items,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)

CODEX = "OpenAI Codex"
CLAUDE = "Anthropic Claude"
GEMINI = "Google Gemini"
QUOTA = "<quota-reset>"


class QuotaRunner(FakeRunner):
    """A fake runner whose scripted ``QUOTA`` output raises a quota reset."""

    def _next_agent_output(self, outputs, *, prompt=None):
        if outputs and outputs[0] == QUOTA:
            outputs.pop(0)
            raise QuotaResetExceededError("Claude quota resets in 9 hours")
        return super()._next_agent_output(outputs, prompt=prompt)


def _records(comments):
    return [
        _decode_round_metadata(match["payload"])
        for body in comments
        if (match := discuss_loop.ROUND_RESUME_MARKER_RE.search(body))
    ]


def _pr_records(runner):
    return _records(comment["body"] for comment in runner.pr_payload["comments"])


def _plan_records(runner):
    return _records(comment["body"] for comment in runner.issue_comments)


def _calls(runner, tool):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == [tool]]


def _reconciliation(records, round_number):
    return next(
        record for record in records
        if record.role == "summary" and record.phase == "reconciliation"
        and record.round_number == round_number
    )


def _ledger_entering(records, round_number):
    """The carried ledger the coder (or next reviewers) received for ``round_number``."""
    record = next(
        record for record in records
        if record.round_number == round_number and record.role in {"coder", "reviewer"}
    )
    return {item.item_id: item for item in record.prior_items}


def _merged(runner):
    return any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def _incomplete_comments(runner):
    return [body for body in runner.comments if "**Review status: Incomplete**" in body]


def _unavailable(signature):
    return json.dumps({
        "schema_version": 1, "kind": "agent_unavailable", "retryable": False,
        "category": "environment", "summary": "The review checkout cannot access the diff.",
        "suggested_action": "Repair the reviewer sandbox before retrying it.",
    }) + f"\n<!-- AGENT_UNAVAILABLE -->\n-- {signature}"


def _round_one_both_block():
    """Round 1: Codex raises item-1 and Claude raises item-2; the coder addresses both."""
    return (
        [structured_pr_review(state="blocking", blocking_items=["Codex finding one."], reviewer=CODEX)],
        [structured_pr_review(state="blocking", blocking_items=["Claude finding two."], reviewer=CLAUDE)],
        [structured_coder_followup(addressed_items=["item-1", "item-2"], reviewer=GEMINI)],
    )


def _codex_blocks_and_clears_claude(disposition="resolved", *, codex_item="item-1", claude_item="item-2"):
    """Codex resolves its own item and dispositions Claude's item; Claude then fails."""
    return structured_pr_review(
        state="blocking", blocking_items=["Codex finding three."], reviewer=CODEX,
        prior_item_dispositions=[
            {"item_id": codex_item, "disposition": "resolved"},
            {"item_id": claude_item, "disposition": disposition},
        ],
    )


def _assert_claude_item_frozen(records, *, continued_round=2):
    reconciliation = _reconciliation(records, continued_round)
    assert reconciliation.failed_reviewer_seats == ("Claude",)
    # The record carries the filtered dispositions: it matches the settled ledger.
    assert "item-2" not in {d.item_id for d in reconciliation.dispositions}
    assert "item-1" in {d.item_id for d in reconciliation.dispositions}
    ledger = _ledger_entering(records, continued_round + 1)
    assert "item-1" not in ledger  # Codex's own item keeps Codex's disposition
    assert ledger["item-2"].status == "blocking"
    assert ledger["item-2"].reviewer == "Claude"
    assert ledger["item-2"].text == "Claude finding two."
    assert ledger["item-3"].reviewer == "Codex"


# ---------------------------------------------------------------------------
# PR loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
def test_pr_blocker_plus_rejected_seat_continues_with_frozen_items(tmp_path, parallel):
    """Rows pr-parallel-block-plus-rejected, failed-seat-items-frozen, pr-failed-seat-never-approves."""
    codex, claude, gemini = _round_one_both_block()
    codex += [
        _codex_blocks_and_clears_claude(),
        structured_pr_review(state="approved", reviewer=CODEX, prior_item_dispositions=[
            {"item_id": "item-2", "disposition": "resolved"},
            {"item_id": "item-3", "disposition": "resolved"},
        ]),
    ]
    # Claude has no scripted round-2 or round-3 answer: it is rejected deterministically.
    gemini.append(structured_coder_followup(addressed_items=["item-2", "item-3"], reviewer=GEMINI))
    runner = FakeRunner(codex_outputs=codex, claude_outputs=claude, gemini_outputs=gemini)
    config = make_config(
        tmp_path, coder="gemini", reviewer=("codex", "claude"), max_rounds=4, review_parallel=parallel,
    )

    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)

    records = _pr_records(runner)
    # Round 2 continued to the coder; round 3 stopped because Codex approved
    # and Claude failed again: a failed seat never satisfies approval.
    assert len(_calls(runner, "gemini")) == 2
    assert len(_calls(runner, "claude")) == 3  # invoked again in rounds 2 and 3
    _assert_claude_item_frozen(records)
    round_two = next(body for body in runner.comments if "PR review round 2 reconciliation" in body)
    assert "continues to the coder if must-fix items remain, with failed seat(s) Claude" in round_two
    assert "Claude (deterministic; frozen carried items: item-2)" in round_two
    assert not any(r.agent == "Claude" and r.round_number > 1 for r in records if r.role == "reviewer")
    assert _ledger_entering(records, 3)  # coder round 3 dispatched with the carried ledger
    assert not _merged(runner)
    assert not _incomplete_comments(runner)


@pytest.mark.parametrize("order", [("codex", "claude"), ("claude", "codex")], ids=["fails-last", "fails-first"])
def test_pr_sequential_failure_in_either_order_runs_remaining_seat(tmp_path, order):
    """Row pr-sequential-block-plus-rejected: the remaining seat runs whatever the order."""
    # Items are numbered in reviewer order: the first seat's finding is item-1.
    codex_item, claude_item = ("item-1", "item-2") if order[0] == "codex" else ("item-2", "item-1")
    codex, claude, gemini = _round_one_both_block()
    codex.append(_codex_blocks_and_clears_claude("future", codex_item=codex_item, claude_item=claude_item))
    runner = FakeRunner(codex_outputs=codex, claude_outputs=claude, gemini_outputs=gemini)
    config = make_config(tmp_path, coder="gemini", reviewer=order, max_rounds=3, review_parallel=False)

    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)

    records = _pr_records(runner)
    assert len(_calls(runner, "codex")) == 2  # Codex still reviewed round 2
    assert any(r.agent == "Codex" and r.round_number == 2 for r in records if r.role == "reviewer")
    # The coder was dispatched for round 2 (its scripted output then ran out).
    assert any(r.role == "summary" and r.phase == "coder-dispatch" and r.round_number == 3 for r in records)
    reconciliation = _reconciliation(records, 2)
    assert reconciliation.failed_reviewer_seats == ("Claude",)
    dispatch = next(r for r in records if r.phase == "coder-dispatch" and r.round_number == 3)
    ledger = {item.item_id: item for item in dispatch.prior_items}
    assert ledger[claude_item].status == "blocking"  # not deferred by Codex's `future`
    assert ledger[claude_item].reviewer == "Claude"
    assert codex_item not in ledger or ledger[codex_item].status != "blocking"


def test_pr_blocker_plus_unavailable_seat_continues_and_reinvokes_it(tmp_path):
    """Row pr-block-plus-unavailable-retry."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=["Codex finding one."], reviewer=CODEX),
            structured_pr_review(state="approved", reviewer=CODEX, prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
            ]),
        ],
        claude_outputs=[_unavailable(CLAUDE), _unavailable(CLAUDE)],
        gemini_outputs=[structured_coder_followup(addressed_items=["item-1"], reviewer=GEMINI)],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=("codex", "claude"), max_rounds=3)

    with pytest.raises(AgentLoopError, match="missing required input from Claude") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    # Round 1 continued (no incomplete comment, coder dispatched); round 2
    # invoked Claude again instead of skipping it, and stopped there because
    # Codex approved while Claude stayed unavailable.
    assert len(_calls(runner, "gemini")) == 1
    assert len(_calls(runner, "claude")) == 2
    assert "Detected in round 2" in str(excinfo.value)
    # Only round 2's stop posts the incomplete-review comment; round 1 did not.
    assert len(_incomplete_comments(runner)) == 1
    assert _reconciliation(_pr_records(runner), 1).failed_reviewer_seats == ("Claude",)
    assert not _merged(runner)


@pytest.mark.parametrize("failure", ["rejected", "unavailable"])
def test_pr_others_approve_failed_seat_still_stops(tmp_path, failure):
    """Row pr-approve-plus-failed-stops: existing error type and message, no coder."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", reviewer=CODEX)],
        claude_outputs=[] if failure == "rejected" else [_unavailable(CLAUDE)],
        gemini_outputs=[structured_coder_followup(reviewer=GEMINI)],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=("codex", "claude"), max_rounds=3)
    expected = "scripted agent output exhausted" if failure == "rejected" else "missing required input from Claude"

    with pytest.raises(AgentLoopError, match=expected):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _calls(runner, "gemini") == []
    assert not _merged(runner)
    assert not any("continues to the coder" in body for body in runner.comments)


@pytest.mark.parametrize("case", ["single-reviewer", "final-round", "quota-reset", "evidence-free-blocker"])
def test_pr_guards_keep_the_stop(tmp_path, case):
    """Row pr-guards-keep-stop."""
    blocker = structured_pr_review(state="blocking", blocking_items=["Codex finding one."], reviewer=CODEX)
    reviewers, max_rounds, claude, error = ("codex", "claude"), 3, [], AgentLoopError
    match = "scripted agent output exhausted"
    codex = [blocker]
    if case == "single-reviewer":
        reviewers, codex = ("claude",), []
    elif case == "final-round":
        max_rounds = 1
    elif case == "quota-reset":
        claude, error, match = [QUOTA], QuotaResetExceededError, "quota resets"
    else:
        # A blocking verdict that yields no must-fix item (only a deferred
        # follow-up) leaves nothing for the coder: the stop is kept.
        codex = [structured_pr_review(state="blocking", future_followups=["Later polish."], reviewer=CODEX)]
    runner = QuotaRunner(
        codex_outputs=codex, claude_outputs=claude,
        gemini_outputs=[structured_coder_followup(addressed_items=["item-1"], reviewer=GEMINI)],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=reviewers, max_rounds=max_rounds)

    with pytest.raises(error, match=match):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _calls(runner, "gemini") == []
    assert not _merged(runner)


def test_pr_resume_after_continued_round_keeps_frozen_item(tmp_path):
    """Row pr-resume-after-continued-round (PR): interrupted after the reconciliation record."""
    codex, claude, gemini = _round_one_both_block()
    codex.append(_codex_blocks_and_clears_claude())
    runner = FakeRunner(codex_outputs=codex, claude_outputs=claude, gemini_outputs=gemini)
    config = make_config(tmp_path, coder="gemini", reviewer=("codex", "claude"), max_rounds=4)
    # The coder's round-2 turn has no scripted answer: the run stops after the
    # round's reconciliation record and before any coder record.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _reconciliation(_pr_records(runner), 2).failed_reviewer_seats == ("Claude",)
    assert not any(r.role == "coder" and r.round_number == 3 for r in _pr_records(runner))

    before = len(_pr_records(runner))
    runner.gemini_outputs.append(structured_coder_followup(addressed_items=["item-2", "item-3"], reviewer=GEMINI))
    runner.codex_outputs.append(structured_pr_review(state="blocking", reviewer=CODEX, prior_item_dispositions=[
        {"item_id": "item-2", "disposition": "blocking", "note": "Claude's carried finding still needs a test."},
        {"item_id": "item-3", "disposition": "resolved"},
    ]))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)

    # The rerun rebuilt the ledger from GitHub records: the first review it
    # published received Claude's item-2 unchanged and Codex's item-1 cleared.
    records = _pr_records(runner)
    resumed_review = next(r for r in records[before:] if r.role == "reviewer" and r.agent == "Codex")
    ledger = {item.item_id: item for item in resumed_review.prior_items}
    assert ledger["item-2"].status == "blocking" and ledger["item-2"].reviewer == "Claude"
    assert ledger["item-2"].text == "Claude finding two."
    assert "item-1" not in ledger and "item-3" in ledger
    assert not any(r.agent == "Claude" and r.state == "approved" for r in records if r.role == "reviewer")
    assert not _merged(runner)


# ---------------------------------------------------------------------------
# Plan loop
# ---------------------------------------------------------------------------


def _plan_round_one():
    claude = [
        structured_plan_state(summary="Initial plan."),
        structured_plan_revision(summary="Revised plan.", prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ]),
    ]
    codex = [structured_plan_review(state="blocking", blocking_plan_issues=["Codex plan gap one."], reviewer=CODEX)]
    gemini = [structured_plan_review(state="blocking", blocking_plan_issues=["Gemini plan gap two."], reviewer=GEMINI)]
    return claude, codex, gemini


def _codex_plan_blocks_and_clears_gemini():
    """Codex resolves both carried items (its own and Gemini's) and raises item-3."""
    return structured_plan_review(
        state="blocking", blocking_plan_issues=["Codex plan gap three."], reviewer=CODEX,
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    )


def _assert_gemini_plan_item_frozen(runner):
    records = _plan_records(runner)
    reconciliation = _reconciliation(records, 2)
    assert reconciliation.failed_reviewer_seats == ("Gemini",)
    assert "item-2" not in {d.item_id for d in reconciliation.dispositions}
    planner = next(r for r in records if r.role == "coder" and r.round_number == 3)
    ledger = {item.item_id: item for item in planner.prior_items}
    assert ledger["item-2"].status == "blocking" and ledger["item-2"].reviewer == "Gemini"
    assert ledger["item-2"].text == "Gemini plan gap two."
    assert "item-1" not in ledger and "item-3" in ledger
    assert not any(r.agent == "Gemini" and r.round_number > 1 for r in records if r.role == "reviewer")
    body = next(body for body in runner.comments if "Plan review round 2 reconciliation" in body)
    assert "continues to the planner revision if must-fix items remain, with failed seat(s) Gemini" in body
    assert "Gemini (deterministic; frozen carried items: item-2)" in body


@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
def test_plan_blocker_plus_failed_seat_continues_with_frozen_items(tmp_path, parallel):
    """Rows plan-block-plus-failed, failed-seat-items-frozen, plan-sequential-reconciliation-record."""
    claude, codex, gemini = _plan_round_one()
    claude.append(structured_plan_revision(summary="Second revision.", prior_plan_item_dispositions=[
        {"item_id": "item-2", "disposition": "resolved"},
        {"item_id": "item-3", "disposition": "resolved"},
    ]))
    codex += [
        _codex_plan_blocks_and_clears_gemini(),
        structured_plan_review(state="approved", reviewer=CODEX, prior_plan_item_dispositions=[
            {"item_id": "item-2", "disposition": "resolved"},
            {"item_id": "item-3", "disposition": "resolved"},
        ]),
    ]
    runner = FakeRunner(claude_outputs=claude, codex_outputs=codex, gemini_outputs=gemini)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=4, review_parallel=parallel)

    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # Round 2 continued to the planner; round 3 stopped (Codex approved,
    # Gemini failed again): the failed seat never approves the plan.
    assert len(_calls(runner, "claude")) == 3
    assert len(_calls(runner, "gemini")) == 3
    _assert_gemini_plan_item_frozen(runner)


@pytest.mark.parametrize("order", [("gemini", "codex"), ("codex", "gemini")], ids=["incomplete-first", "incomplete-last"])
def test_plan_sequential_incomplete_review_in_either_order(tmp_path, order):
    """Row plan-sequential-incomplete-both-orders."""
    # Items are numbered in reviewer order: the first seat's finding is item-1.
    gemini_item = "item-1" if order[0] == "gemini" else "item-2"
    claude, codex, gemini = _plan_round_one()
    codex.append(_codex_plan_blocks_and_clears_gemini())
    # Incomplete: no actionable verdict, only an inability to finish the review.
    gemini.append(structured_plan_review(
        state="blocking", reviewer=GEMINI,
        summary="Review incomplete; unable to verify the retry path without further inspection.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
            {"item_id": "item-2", "disposition": "resolved"},
        ],
    ))
    runner = FakeRunner(claude_outputs=claude, codex_outputs=codex, gemini_outputs=gemini)
    config = make_config(tmp_path, reviewer=order, max_rounds=4, review_parallel=False)

    # The planner's second revision has no scripted answer.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    records = _plan_records(runner)
    assert any(r.agent == "Codex" and r.round_number == 2 for r in records if r.role == "reviewer")
    assert len(_calls(runner, "claude")) == 3  # the planner revision was dispatched
    reconciliation = _reconciliation(records, 2)
    assert reconciliation.failed_reviewer_seats == ("Gemini",)
    assert gemini_item not in {d.item_id for d in reconciliation.dispositions}
    assert not any(r.agent == "Gemini" and r.round_number == 2 for r in records if r.role == "reviewer")


@pytest.mark.parametrize("case", ["other-approves", "single-reviewer"])
def test_plan_incomplete_review_without_a_healthy_blocker_still_stops(tmp_path, case):
    """Row plan-sequential-incomplete-both-orders (stop half)."""
    incomplete = structured_plan_review(
        state="blocking", reviewer=GEMINI,
        summary="Review incomplete; unable to verify the retry path without further inspection.",
    )
    reviewers = ("codex", "gemini") if case == "other-approves" else ("gemini",)
    runner = FakeRunner(
        claude_outputs=[structured_plan_state(summary="Initial plan.")],
        codex_outputs=[structured_plan_review(state="approved", reviewer=CODEX)],
        gemini_outputs=[incomplete],
    )
    config = make_config(tmp_path, reviewer=reviewers, max_rounds=3, review_parallel=False)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert len(_calls(runner, "claude")) == 1
    # No durable failed-seat record is invented on the stop path.
    assert not any(r.failed_reviewer_seats for r in _plan_records(runner))


@pytest.mark.parametrize("case", ["other-approves", "single-reviewer", "quota-reset"])
def test_plan_failed_seat_without_continuation_stops(tmp_path, case):
    """Row plan-approve-plus-failed-stops: the existing raise, quota reset first."""
    gemini_outputs = [structured_plan_review(state="blocking", blocking_plan_issues=["Gap."], reviewer=GEMINI)]
    reviewers, claude_reviews, error, match = ("codex", "gemini"), [], AgentLoopError, "scripted agent output exhausted"
    if case == "other-approves":
        gemini_outputs = [structured_plan_review(state="approved", reviewer=GEMINI)]
    elif case == "single-reviewer":
        reviewers = ("codex",)
    else:
        reviewers = ("codex", "gemini", "claude")
        claude_reviews, error, match = [QUOTA], QuotaResetExceededError, "quota resets"
    runner = QuotaRunner(
        claude_outputs=[structured_plan_state(summary="Initial plan."), *claude_reviews],
        codex_outputs=[], gemini_outputs=gemini_outputs,
    )
    config = make_config(tmp_path, coder="claude", reviewer=reviewers, max_rounds=3, review_parallel=True)

    with pytest.raises(error, match=match):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any("continues to the planner revision" in body for body in runner.comments)
    assert not any(r.role == "coder" and r.round_number == 2 for r in _plan_records(runner))


def test_plan_sequential_round_without_failures_posts_no_reconciliation_record(tmp_path):
    """Row plan-sequential-reconciliation-record: posting is unchanged without failures."""
    runner = FakeRunner(
        claude_outputs=[structured_plan_state(summary="Initial plan.")],
        codex_outputs=[structured_plan_review(state="approved", reviewer=CODEX)],
        gemini_outputs=[structured_plan_review(state="approved", reviewer=GEMINI)],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3, review_parallel=False)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert not [r for r in _plan_records(runner) if r.phase == "reconciliation"]


def test_plan_resume_after_continued_round_keeps_frozen_item(tmp_path):
    """Row pr-resume-after-continued-round (plan): interrupted before the planner record."""
    claude, codex, gemini = _plan_round_one()
    codex.append(_codex_plan_blocks_and_clears_gemini())
    runner = FakeRunner(claude_outputs=claude, codex_outputs=codex, gemini_outputs=gemini)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=5, review_parallel=True)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _reconciliation(_plan_records(runner), 2).failed_reviewer_seats == ("Gemini",)
    assert not any(r.role == "coder" and r.round_number == 3 for r in _plan_records(runner))

    before = len(_plan_records(runner))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # The reconciled round is missing Gemini's review, so the rerun carries
    # the ledger over from the round's GitHub records (a reviewer-only advance
    # on the unchanged plan): Gemini's item-2 survives, Codex's item-1 does not.
    records = _plan_records(runner)
    carried = next(r for r in records[before:] if r.phase == "plan-phase-advance")
    ledger = {item.item_id: item for item in carried.prior_items}
    assert ledger["item-2"].status == "blocking" and ledger["item-2"].reviewer == "Gemini"
    assert ledger["item-2"].text == "Gemini plan gap two."
    assert "item-1" not in ledger and "item-3" in ledger
    assert not any(r.agent == "Gemini" and r.state == "approved" for r in records if r.role == "reviewer")


# ---------------------------------------------------------------------------
# Helper, round-metadata and round-state units
# ---------------------------------------------------------------------------


def _item(item_id, reviewer, *, owners=None, aliases=(), sub_items=()):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer=reviewer, source_round=1, text=f"{item_id} text",
        status="blocking", resolution_owners=tuple(owners or (reviewer,)),
        alias_item_ids=tuple(aliases), sub_items=tuple(sub_items),
    )


def _disposition(item_id, reviewer="Codex", disposition="resolved", **extra):
    return ReviewItemDisposition(item_id=item_id, disposition=disposition, reviewer=reviewer, **extra)


def test_preserve_failed_seat_items_freezes_owned_co_owned_alias_and_sub_items():
    """Row failed-seat-items-frozen (helper unit)."""
    sub = ReviewSubItem(sub_item_id="a", text="part a", status="open")
    prior = [
        _item("item-1", "Codex"),
        _item("item-2", "Claude"),
        _item("item-3", "Codex", owners=("Codex", "Claude")),
        _item("item-4", "Claude", aliases=("item-9",)),
        _item("item-5", "Claude", sub_items=(sub,)),
    ]
    dispositions = {
        "item-1": [_disposition("item-1")],
        "item-2": [_disposition("item-2", disposition="future")],
        "item-3": [_disposition("item-3")],
        "item-9": [_disposition("item-9")],
        "item-5": [_disposition("item-5")],
    }
    filtered, frozen = preserve_failed_seat_items(prior, dispositions, ("Claude",))
    assert frozen == ("item-2", "item-3", "item-4", "item-5")
    assert filtered["item-1"] == dispositions["item-1"]
    assert all(filtered[key] == [] for key in ("item-2", "item-3", "item-9", "item-5"))
    kept, future = _apply_unresolved_item_dispositions(prior, filtered)
    assert [item.item_id for item in kept] == ["item-2", "item-3", "item-4", "item-5"]
    assert kept == prior[1:] and future == []
    # No failed seat: an equal copy, never the same mutable lists.
    same, none = preserve_failed_seat_items(prior, dispositions, ())
    assert same == dispositions and none == ()


def test_failed_seat_continuation_predicate_guards():
    """Row pr-guards-keep-stop (predicate unit): quota, evidence pass, final round, no blocker."""
    base = dict(
        configured_reviewer_count=2, blocking_seats=["Codex"], failed_seats=["Claude"],
        failures=[AgentLoopError("rejected")], round_number=1, allowed_rounds=3,
    )
    assert failed_seat_continuation_allowed(**base)
    assert not failed_seat_continuation_allowed(**{**base, "configured_reviewer_count": 1})
    assert not failed_seat_continuation_allowed(**{**base, "blocking_seats": ["Claude"]})
    assert not failed_seat_continuation_allowed(**{**base, "blocking_seats": []})
    assert not failed_seat_continuation_allowed(**{**base, "round_number": 3})
    assert not failed_seat_continuation_allowed(**{**base, "evidence_pass": "response"})
    assert not failed_seat_continuation_allowed(**{**base, "failures": [QuotaResetExceededError("q")]})
    assert not failed_seat_continuation_allowed(**{**base, "failed_seats": []})


def test_selective_policy_force_selects_a_failed_seat_next_round():
    decision = SchedulingDecision(
        selected_reviewers=("Codex",), paused_reviewers=(("Claude", "approved earlier"),),
        reason="owner-scoped", classification=TransitionClassification(kind="code", reason="x"),
    )
    forced = force_select_failed_seats(decision, {"Claude"}, ("Codex", "Claude"))
    assert forced.selected_reviewers == ("Codex", "Claude")
    assert forced.paused_reviewers == ()
    assert "failed seat re-invoked" in forced.reason
    assert force_select_failed_seats(decision, set(), ("Codex", "Claude")) is decision


def test_failed_reviewer_seats_round_trips_bounded_and_defaults_empty():
    metadata = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="abc",
        phase="reconciliation", failed_reviewer_seats=("Claude",),
    )
    body = _attach_round_metadata("Reconciliation.", metadata)
    decoded = _records([body])[0]
    assert decoded.failed_reviewer_seats == ("Claude",)
    legacy = _attach_round_metadata("Reconciliation.", PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="abc", phase="reconciliation",
    ))
    assert "failed_reviewer_seats" not in legacy
    assert _records([legacy])[0].failed_reviewer_seats == ()
    bounded = bound_failed_reviewer_seats([f"Seat {i}" for i in range(50)] + ["Seat 1", " ", "x" * 500])
    assert len(bounded) == 32 and len(set(bounded)) == 32
    assert all(len(name) <= 200 for name in bound_failed_reviewer_seats(["x" * 500]))


def _round_records(*, failed_seats, head="old"):
    prior = (_item("item-1", "Codex"), _item("item-2", "Claude"))
    new = _item("item-3", "Codex")
    reviewer = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=2, subject=head, state="blocking",
        prior_items=prior, new_items=(new,), phase="publication",
        dispositions=(_disposition("item-1"), _disposition("item-2")),
    )
    summary = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=head,
        prior_items=prior, new_items=(new,), phase="reconciliation",
        dispositions=(_disposition("item-1"),) if failed_seats else (_disposition("item-1"), _disposition("item-2")),
        failed_reviewer_seats=failed_seats,
    )
    return [PostedRoundRecord(index=i, metadata=m, body="") for i, m in enumerate((reviewer, summary))]


@pytest.mark.parametrize("failed_seats", [("Claude",), ()], ids=["failed-seat", "legacy"])
def test_history_replay_and_head_advance_recovery_apply_the_freeze(failed_seats):
    """Row pr-resume-after-continued-round (round-state unit), including legacy compatibility."""
    records = _round_records(failed_seats=failed_seats)
    resolved = _canonically_resolved_history_item_ids(records, reconciliation_mode="aggregate", same_status="same-pr")
    outcomes, _deferred = _replay_history_items(records, reconciliation_mode="aggregate", same_status="same-pr")
    recovered = _recover_unrecorded_pr_head_advance(records, head_sha="new")
    recovered_ids = {item.item_id for item in recovered.prior_items} if recovered is not None else set()
    if failed_seats:
        assert resolved == frozenset({"item-1"})
        assert outcomes["item-2"] == "active"
        assert recovered is not None and {"item-2", "item-3"} <= recovered_ids and "item-1" not in recovered_ids
    else:
        # Legacy records recover exactly as before: Codex's `resolved` clears item-2.
        assert resolved == frozenset({"item-1", "item-2"})
        assert outcomes["item-2"] != "active"
        assert "item-2" not in recovered_ids
    aggregate = _aggregate_record_dispositions(
        records[:1], prior_items=records[0].metadata.prior_items, failed_seats=failed_seats,
    )
    assert bool(aggregate["item-2"]) is (not failed_seats)
