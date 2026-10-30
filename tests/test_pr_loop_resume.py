"""PR-loop tests for round_state and round_transport: PR round resume (#1542)."""

import base64
import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_coder_followup_comment,
    _render_public_pr_review_comment,
)
from coding_review_agent_loop.github import (
    IssueComment,
    PullRequestCheck,
)
from coding_review_agent_loop.round_state import (
    QualificationCheckpoint,
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _resume_pr_round,
    _strip_round_metadata,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_review,
    parse_unresolved_item_dispositions,
    validate_structured_coder_followup,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    command_index,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.round_state as round_state
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in, patch_object_in
from pr_loop_test_helpers import (
    _followup_matrix_context,
    _watch_check_board,
)


def test_resume_pr_round_reparses_orchestrator_rendered_blocking_issues_comment():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    rendered_review = _render_public_pr_review_comment(
        parse_review(
            "Need one more regression test before merge."
            + blocking_issues("Exercise the structured-resume path.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            reviewer="OpenAI Codex",
        ),
        reviewer="Codex",
        human_requirements_resolved_flag=False,
        prior_items=(),
        dispositions=(),
    )
    review_comment = _attach_round_metadata(
        rendered_review,
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
            dispositions=(),
            new_items=(),
            state="blocking",
        ),
    )
    coder_comment = _attach_round_metadata(
        "Addressed the review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    resumed_review = parse_review(resumed.completed_reviews[0].body, reviewer="Codex")
    assert [item.text for item in resumed_review.blocking_items] == [
        "Exercise the structured-resume path."
    ]
    assert resumed_review.summary == "Need one more regression test before merge."

def test_resume_pr_round_prefers_structured_coder_followup_metadata():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Need one more regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    raw_structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Added the requested regression test.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirement_dispositions": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_coder_followup(raw_structured_followup)
    assert parsed is not None
    public_comment = _render_public_coder_followup_comment(parsed, agent="Claude")
    coder_comment = _attach_round_metadata(
        public_comment,
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_structured_followup,
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.coder_output == raw_structured_followup
    resumed_followup = validate_structured_coder_followup(resumed.coder_output)
    assert resumed_followup is not None
    assert resumed_followup.human_requirements.addressed_ids == ("Requirement 1",)
    assert '"kind": "coder_followup"' not in _strip_round_metadata(coder_comment)


def test_resume_pr_round_recovers_summary_only_qualification_checkpoint():
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
        candidate_head_sha=None,
        obligation_identity="managed-exact-head-ci:item-30",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="repair_required",
        failed_head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
        candidate_head_sha=None,
        base_branch="main",
        watch_failure_extension_used=True,
        allowed_rounds=2,
    )
    summary_comment = _attach_round_metadata(
        "Persisted repair handoff.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="0109780857a5f763d3dacdb5f43c546d6776b3ba",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=summary_comment)],
        head_sha="0109780857a5f763d3dacdb5f43c546d6776b3ba",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.round_number == 2
    assert resumed.coder_output is None
    assert resumed.completed_reviews == ()
    assert resumed.prior_items == (item,)
    assert resumed.qualification_checkpoint == checkpoint


def test_pr_loop_resume_after_interrupted_coder_keeps_failed_head_and_budget(
    tmp_path, monkeypatch
):
    failed = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://example.test/555"
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Initial review."),
            structured_pr_review(
                state="approved",
                summary="Repaired head approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Repaired the failing check.", addressed_items=["item-1"]
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=False, watch_pending_ci=False, max_rounds=3)

    def checks(*args, **kwargs):
        if kwargs["metadata"].head_sha == "6367c48dd193d56ea7b0baad25b19455e529f5ee":
            return _watch_check_board("failing", failing=(failed,))
        return _watch_check_board("passing")

    monkeypatch.setattr(pr_loop, "get_pr_checks", checks)
    original = validated_agent._run_validated_agent
    interrupt = True

    class CoderInterrupted(Exception):
        pass

    def interrupt_during_coder(*args, **kwargs):
        nonlocal interrupt
        if kwargs.get("role") == "coder" and interrupt:
            interrupt = False
            raise CoderInterrupted
        return original(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", interrupt_during_coder)
    with pytest.raises(CoderInterrupted):
        run_pr_loop(runner, pr_number=77, config=config)

    checkpoint_records = [
        comment for comment in runner.pr_payload["comments"]
        if "qualification checkpoint: machine obligation state persisted before coder handoff"
        in comment["body"]
    ]
    assert len(checkpoint_records) == 1
    assert "AGENT_LOOP_META" in checkpoint_records[0]["body"]

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_commands = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(review_commands) == 2
    assert sum(
        comment.startswith(
            "PR #77 qualification checkpoint: machine obligation state persisted before coder handoff."
        )
        for comment in runner.comments
    ) == 2

def test_resume_pr_round_marks_empty_ledger_incomplete_after_same_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior same-head item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is True

def test_resume_pr_round_does_not_mark_ledger_incomplete_for_cross_subject_prior_new_items():
    prior_new_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Prior 5e6741348108b11e3f3ef936c335b1967d17ac6c item.",
        status="blocking",
    )
    prior_review_comment = _attach_round_metadata(
        "Prior review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(prior_new_item,),
            state="blocking",
        ),
    )
    current_coder_comment = _attach_round_metadata(
        "Current coder output.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="fd2d39cf710f84ca537091424fc212fa5211885b",
            prior_items=(),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=prior_review_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=current_coder_comment),
        ],
        head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.ledger_may_be_incomplete is False

def test_resume_pr_round_recovers_unrecorded_head_advance_reviewer_new_item():
    active_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Fix the regression before merge.",
        status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Initial PR handoff.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(active_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=review_comment),
        ],
        head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
        configured_reviewers=("gemini",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.ledger_may_be_incomplete is True
    assert resumed.round_number == 1
    assert resumed.completed_reviews == ()
    assert [item.item_id for item in resumed.prior_items] == ["item-2"]
    assert resumed.next_unresolved_item_number == 3


def test_resume_pr_round_recovers_parallel_reconciliation_new_items_after_head_advance():
    """Parallel publication checkpoints omit items; reconciliation owns the ledger."""
    active_items = (
        UnresolvedReviewItem(
            item_id="item-1",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Preserve the first reconciled blocker.",
            status="blocking",
        ),
        UnresolvedReviewItem(
            item_id="item-2",
            reviewer="OpenAI Codex",
            source_round=1,
            text="Preserve the second reconciled blocker.",
            status="same-pr",
        ),
    )
    coder_comment = _attach_round_metadata(
        "Initial PR handoff.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="old-sha", prior_items=(),
        ),
    )
    publication_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="old-sha", prior_items=(), state="blocking", phase="publication",
        ),
    )
    reconciliation_comment = _attach_round_metadata(
        "Reconciliation.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Review reconciliation", round_number=1,
            subject="old-sha", prior_items=(), new_items=active_items, state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=publication_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=reconciliation_comment),
        ],
        head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert [item.item_id for item in resumed.prior_items] == ["item-1", "item-2"]
    assert resumed.next_unresolved_item_number == 3

def test_resume_pr_round_recovers_coder_only_unrecorded_head_advance():
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Still needs a targeted test.",
        status="same-pr",
    )
    future_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Document this later.",
        status="future",
    )
    coder_comment = _attach_round_metadata(
        "Addressed prior feedback.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="old-sha",
            prior_items=(carried_item, future_item),
            compact_prior_summaries=("Older summary.",),
        ),
    )

    resumed = _resume_pr_round(
        [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=coder_comment)],
        head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
        configured_reviewers=("codex",),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert resumed.round_number == 2
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    assert resumed.compact_prior_summaries == ("Older summary.",)

def test_resume_pr_round_recovers_reviewer_only_with_aggregated_dispositions():
    prior_blocking = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Fix the flaky test.",
        status="blocking",
    )
    prior_same_pr = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Tighten the docs.",
        status="same-pr",
    )
    future_new_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=2,
        text="Follow up in another PR.",
        status="future",
    )
    active_new_item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Google Gemini",
        source_round=2,
        text="Add one same-PR assertion.",
        status="same-pr",
    )
    codex_resolution = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
        note=None,
    )
    gemini_same_pr = ReviewItemDisposition(
        item_id="item-2",
        reviewer="Google Gemini",
        disposition="same-pr",
        note="Still needed before merge.",
    )
    codex_comment = _attach_round_metadata(
        "Codex review.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(codex_resolution,),
            new_items=(future_new_item,),
            state="approved",
        ),
    )
    gemini_comment = _attach_round_metadata(
        "Gemini review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="old-sha",
            prior_items=(prior_blocking, prior_same_pr),
            dispositions=(gemini_same_pr,),
            new_items=(active_new_item,),
            state="blocking",
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=codex_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=gemini_comment),
        ],
        head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert resumed.unrecorded_head_advance is True
    assert [item.item_id for item in resumed.prior_items] == ["item-2", "item-4"]
    assert resumed.prior_items[0].status == "same-pr"
    assert "Google Gemini: Still needed before merge." in resumed.prior_items[0].notes

def test_resume_pr_round_ignores_unrecorded_head_advance_with_no_active_items():
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Future cleanup.",
        status="future",
    )
    review_comment = _attach_round_metadata(
        "Approved with future follow-up.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(future_item,),
            state="approved",
        ),
    )

    assert (
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=review_comment)],
            head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
            configured_reviewers=("codex",),
        )
        is None
    )

def test_resume_pr_round_fails_early_for_incoherent_unrecorded_head_advance():
    bad_comment = _attach_round_metadata(
        "Bad metadata.\n<!-- AGENT_STATE: blocking -->\n-- Bot",
        PostedRoundMetadata(
            flow="pr",
            role="observer",
            agent="Bot",
            round_number=1,
            subject="old-sha",
        ),
    )

    with pytest.raises(
        AgentLoopError,
        match=(
            "PR head advanced without a recorded coder follow-up.*"
            "Current head: fd2d39cf710f84ca537091424fc212fa5211885b.*Latest recorded metadata subject: old-sha"
        ),
    ):
        _resume_pr_round(
            [IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=bad_comment)],
            head_sha="fd2d39cf710f84ca537091424fc212fa5211885b",
            configured_reviewers=("codex",),
        )

def test_resume_pr_round_prefers_latest_metadata_ledger_for_same_head_replay():
    stale_item = UnresolvedReviewItem(
        item_id="item-3",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Stale replay item.",
        status="blocking",
        source_status="blocking",
    )
    active_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active replay item.",
        status="blocking",
        source_status="blocking",
    )
    stale_coder_comment = _attach_round_metadata(
        "Stale replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(stale_item,),
        ),
    )
    stale_reviewer_comment = _attach_round_metadata(
        "Still blocked."
        + prior_item_dispositions("[item-3] still blocking: stale replay")
        + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(stale_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-3] still blocking: stale replay"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="blocking",
        ),
    )
    active_coder_comment = _attach_round_metadata(
        "Current replay.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(active_item,),
        ),
    )
    active_reviewer_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(active_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="Google Gemini",
                )[0],
            ),
            state="approved",
        ),
    )
    previous_head_comment = _attach_round_metadata(
        "Older head.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=4,
            subject="old-head",
            prior_items=(
                UnresolvedReviewItem(
                    item_id="item-9",
                    reviewer="OpenAI Codex",
                    source_round=3,
                    text="Older head item.",
                    status="blocking",
                ),
            ),
        ),
    )

    resumed = _resume_pr_round(
        [
            IssueComment(author="bot", created_at="2026-05-25T00:00:00Z", body=previous_head_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:01:00Z", body=stale_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:02:00Z", body=stale_reviewer_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:03:00Z", body=active_coder_comment),
            IssueComment(author="bot", created_at="2026-05-25T00:04:00Z", body=active_reviewer_comment),
        ],
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        configured_reviewers=("codex", "gemini"),
    )

    assert resumed is not None
    assert [item.item_id for item in resumed.prior_items] == ["item-1"]
    # The numbering high-water mark is global (the older head held item-9): IDs are never reused.
    assert resumed.next_unresolved_item_number == 10
    assert [record.metadata.agent for record in resumed.completed_reviews] == ["Gemini"]

def test_pr_loop_resume_hybrid_history_prefers_metadata_ledger_over_legacy_markdown(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    legacy_comment = (
        "Legacy raw markdown review.\n\n"
        "### Blocking issues\n"
        "- Keep the legacy fallback path.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex"
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Looks good."
        + prior_item_dispositions("[item-1] resolved")
        + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
            dispositions=(
                parse_unresolved_item_dispositions(
                    prior_item_dispositions("[item-1] resolved"),
                    reviewer="OpenAI Codex",
                )[0],
            ),
            state="approved",
        ),
    )
    runner = FakeRunner(
        gemini_outputs=[
            "Ship it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": legacy_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:06:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt
    assert "Keep the legacy fallback path." not in gemini_prompt


def test_pr_resume_preserves_persisted_failure_in_next_coder_metadata(tmp_path):
    from coding_review_agent_loop.local_test_evidence import (
        bounded_evidence_for_round,
        decode_bounded_evidence,
    )

    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Preserve the prior local failure.",
        status="blocking",
    )
    prior_evidence = bounded_evidence_for_round({"observations": [{
        "command": ["python", "-m", "pytest"],
        "outcome": "failed",
        "provenance": "parent-observed",
        "receipt_id": "persisted-failure",
        "turn_id": "old-turn",
        "environment": "equivalent",
        "attribution": {"state": "current-head", "head": "6367c48dd193d56ea7b0baad25b19455e529f5ee", "stable": True, "tracked_digest": "tree-a"},
    }]})
    coder_comment = _attach_round_metadata(
        structured_coder_followup(remaining_items=["item-1"]),
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", prior_items=(item,), local_test_evidence=prior_evidence,
            raw_structured_coder_response=structured_coder_followup(remaining_items=["item-1"]),
        ),
    )
    review_comment = _attach_round_metadata(
        structured_pr_review(
            state="blocking",
            summary="Failure remains.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "blocking", "note": "Fix it."}],
        ),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", prior_items=(item,), state="blocking",
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Fix it."),),
        ),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"], tests_run=[])],
        codex_outputs=[structured_pr_review(
            state="approved",
            summary="Resolved.",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        )],
        pr_payload={"comments": [
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": review_comment},
        ]},
    )

    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer="codex")) == 0
    posted = next(
        comment["body"] for comment in reversed(runner.pr_payload["comments"])
        if "## Coder follow-up" in comment["body"]
    )
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", posted)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    decoded = decode_bounded_evidence(metadata.local_test_evidence)
    assert decoded is not None
    assert "persisted-failure" in [row.receipt_id for row in decoded.observations]
    preserved = next(row for row in decoded.observations if row.receipt_id == "persisted-failure")
    assert preserved.environment_state == "identity-unknown"
    assert preserved.attribution.state == "stale"

def test_pr_loop_routes_unrecorded_head_advance_through_coder_before_reviewers(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Preserve the metadata-backed unresolved item on rerun.",
        status="blocking",
    )
    old_coder_comment = _attach_round_metadata(
        "Opened the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-sha",
            prior_items=(),
        ),
    )
    # The historical reviewer is the configured one: a reviewer missing from
    # the configured board would be a recorded board change (#1376).
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Addressed the recovered prior item.",
                addressed_items=["item-2"],
                tests_run=["python -m pytest tests/test_agent_loop.py -k unrecorded_head"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Recovered item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "fd2d39cf710f84ca537091424fc212fa5211885b",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    first_coder = command_index(runner.commands, ["claude"])
    first_reviewer = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_reviewer
    reviewer_prompt = runner.commands[first_reviewer][0][-1]
    assert "[item-2]" in reviewer_prompt
    assert "Preserve the metadata-backed unresolved item on rerun." in reviewer_prompt
    posted_coder_comment = next(
        comment["body"]
        for comment in runner.pr_payload["comments"]
        if "## Coder follow-up" in comment["body"]
    )
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", posted_coder_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.subject == "fd2d39cf710f84ca537091424fc212fa5211885b"
    assert metadata.round_number == 2
    assert [item.item_id for item in metadata.prior_items] == ["item-2"]

def test_pr_loop_unrecorded_head_advance_prevents_empty_ledger_unknown_item_abort(tmp_path):
    old_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="Google Gemini",
        source_round=1,
        text="Carry this item instead of starting an empty ledger.",
        status="blocking",
    )
    old_review_comment = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Gemini",
            round_number=1,
            subject="old-sha",
            prior_items=(),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                summary="Classified the recovered item.",
                addressed_items=["item-2"],
            )
        ],
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Old item is resolved.",
                prior_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            )
        ],
        pr_payload={
            "headRefOid": "fd2d39cf710f84ca537091424fc212fa5211885b",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": old_review_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.claude_outputs == []
    assert runner.codex_outputs == []
    assert not any("unknown item" in comment.lower() for comment in runner.comments)


# --- Visible sidecar labels on the PR posting seam (#842) --------------------


def _oversized_pr_round_body(role: str):
    import os

    from coding_review_agent_loop.protocol_markers import TrustedBody
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    text = _attach_round_metadata(
        "Visible PR round response",
        PostedRoundMetadata(
            flow="pr",
            role=role,
            agent="claude",
            round_number=2,
            subject="sidecar-label-pr-seam",
            canonical_reviewer_response=base64.urlsafe_b64encode(os.urandom(70_000)).decode("ascii"),
        ),
    )
    return TrustedBody.canonical(text, expected_tokens=("AGENT_LOOP_META",))


def _post_oversized_pr_round(monkeypatch, role: str) -> list[str]:
    import coding_review_agent_loop.github as github_module

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    posted: list[str] = []

    class _Runner:
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            if "--body-file" not in args:
                # Reconciliation baseline reads (#510) are unsupported here.
                return SimpleNamespace(returncode=1, stdout="", stderr="unsupported read")
            posted.append(Path(args[args.index("--body-file") + 1]).read_text())
            return SimpleNamespace(returncode=0, stdout="", stderr="")

    github_module.post_pr_comment(
        _Runner(),
        config=SimpleNamespace(quiet=True, dry_run=False, gh_cmd="gh", repo="owner/repo"),
        pr_number=857,
        body=_oversized_pr_round_body(role),
    )
    return posted


def test_pr_reviewer_round_sidecars_use_review_wording_before_anchor(monkeypatch):
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        ROUND_TRANSPORT_SIDECAR_RE,
        is_round_transport_sidecar,
    )

    posted = _post_oversized_pr_round(monkeypatch, "reviewer")

    sidecars, anchor = posted[:-1], posted[-1]
    assert sidecars
    for position, body in enumerate(sidecars, start=1):
        assert body.startswith(f"Agent-loop review attachment {position}/{len(sidecars)} ")
        assert body.endswith(ROUND_TRANSPORT_SIDECAR_RE.search(body).group(0))
        assert "see the following review comment." in body
        payload = json.loads(
            base64.urlsafe_b64decode(ROUND_TRANSPORT_SIDECAR_RE.search(body).group("payload"))
        )
        assert "kind" not in payload
    assert not is_round_transport_sidecar(anchor)
    assert ROUND_RESUME_MARKER_RE.search(anchor)


def test_pr_coder_response_sidecars_use_neutral_wording(monkeypatch):
    posted = _post_oversized_pr_round(monkeypatch, "coder")

    sidecars = posted[:-1]
    assert sidecars
    for body in sidecars:
        assert body.startswith("Agent-loop attachment ")
        assert "review attachment" not in body and "plan attachment" not in body


# --- #959: delta-rendered coder matrix evidence ----------------------------

from coding_review_agent_loop.round_transport import (  # noqa: E402
    decode_mapping,
    encode_mapping,
)

_EVIDENCE_HEADING_959 = "### Risk-based mode and transition test matrix evidence"


def _coder_records_959(runner):
    records = []
    for item in runner.pr_payload.get("comments", []):
        body = item.get("body") if isinstance(item, dict) else None
        if not isinstance(body, str) or "AGENT_LOOP_META: " not in body:
            continue
        metadata = round_state._decode_round_metadata(
            body.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        )
        if metadata.role == "coder":
            records.append((body, metadata))
    return records


def _blocking_reviews_959(count):
    reviews = [
        structured_pr_review(state="blocking", blocking_items=["Exercise the follow-up row."])
    ]
    for _ in range(count - 1):
        reviews.append(structured_pr_review(
            state="blocking",
            prior_item_dispositions=[{
                "item_id": "item-1", "disposition": "blocking", "note": "Still open.",
            }],
        ))
    reviews.append(structured_pr_review(
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    return reviews


def test_959_in_process_coder_rounds_render_full_then_deltas_with_stable_anchor(tmp_path):
    plan_context = _followup_matrix_context()
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary=f"Round {n}.")
            for n in (1, 2, 3)
        ],
        codex_outputs=_blocking_reviews_959(3),
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=4)
    spy_calls = []
    real_resolve = pr_loop.resolve_matrix_evidence_render

    def spy(current, previous, current_round):
        decision = real_resolve(current, previous, current_round)
        spy_calls.append((previous, current_round, decision))
        return decision

    with patch.object(pr_loop, "resolve_matrix_evidence_render", spy):
        assert run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=plan_context
        ) == 0

    records = _coder_records_959(runner)
    assert len(records) == 3
    (first_body, first), (second_body, second), (third_body, third) = records
    # The establishing round renders every row and anchors to its own record.
    assert first.risk_test_matrix_evidence_full_round == first.round_number
    assert first.risk_test_matrix_evidence_full_round_status == "valid"
    assert "<summary>Full matrix evidence (1 row)</summary>" in first_body
    assert "**followup-derived-evidence**" in first_body
    assert [call[1] for call in spy_calls] == [
        first.round_number, second.round_number, third.round_number
    ]
    # The in-process record reused as previous metadata is valid by construction.
    assert spy_calls[1][0].risk_test_matrix_evidence_full_round_status == "valid"
    for body, record, previous in ((second_body, second, first), (third_body, third, second)):
        assert record.risk_test_matrix_evidence_full_round == first.round_number
        assert record.risk_test_matrix_evidence is not None
        assert len(record.risk_test_matrix_evidence["rows"]) == 1
        assert "**followup-derived-evidence**" not in body
        assert (
            f"1 row unchanged since round {previous.round_number}; "
            f"full matrix in round {first.round_number}."
        ) in body
        assert "<details>" in body and body.index(_EVIDENCE_HEADING_959) < body.index("<details>")


def _seeded_resume_runner_959(plan_context, *, anchor_mutation=None, full_round=None):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Exercise follow-up evidence derivation.",
        status="blocking",
        source_status="blocking",
    )
    raw_coder = structured_coder_followup(addressed_items=["item-1"], summary="Seeded.")
    parsed_coder = validate_structured_coder_followup(raw_coder)
    evidence = {
        "matrix_identity": plan_context.risk_test_matrix_identity,
        "rows": [{
            "row_id": "followup-derived-evidence",
            "status": "missing",
            "test_identifiers": [],
            "test_locations": [],
            "workflow_path_claim": "Seeded legacy evidence.",
            "outcome_assertions": [],
            "forbidden_effect_assertions": [],
            "evidence_citations": [],
            "caveats": [],
        }],
    }
    extra = {}
    if full_round is not None:
        extra["risk_test_matrix_evidence_full_round"] = full_round
    coder_comment = _attach_round_metadata(
        _render_public_coder_followup_comment(
            parsed_coder, agent="Claude", prior_items=(carried_item,)
        ),
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
            raw_structured_coder_response=raw_coder,
            risk_test_matrix_evidence=evidence,
            **extra,
        ),
    )
    if anchor_mutation is not None:
        encoded = coder_comment.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
        payload = decode_mapping(encoded)
        payload["risk_test_matrix_evidence_full_round"] = anchor_mutation
        coder_comment = coder_comment.replace(encoded, encode_mapping(payload))
    review_comment = _attach_round_metadata(
        structured_pr_review(
            state="blocking",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "blocking", "note": "Still open."}
            ],
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(carried_item,),
            dispositions=(
                ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Still open."),
            ),
            state="blocking",
        ),
    )
    return FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary="Resumed follow-up.")
        ],
        # Resume re-reviews first, so every review carries the item-1 disposition.
        codex_outputs=_blocking_reviews_959(2)[1:],
        pr_payload={
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-06-01T00:01:00Z", "body": review_comment},
            ],
        },
    )


def test_959_resume_from_legacy_coder_record_anchors_to_its_round(tmp_path):
    plan_context = _followup_matrix_context()
    runner = _seeded_resume_runner_959(plan_context)
    seeded = _coder_records_959(runner)[0][1]
    assert seeded.risk_test_matrix_evidence_full_round_status == "absent"

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3),
        approved_plan_context=plan_context,
    ) == 0

    records = _coder_records_959(runner)
    assert len(records) >= 2
    body, posted = records[-1]
    assert posted.round_number > seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round == seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round_status == "valid"
    assert f"full matrix in round {seeded.round_number}." in body
    assert f"unchanged since round {seeded.round_number}" in body
    assert posted.risk_test_matrix_evidence is not None
    assert len(posted.risk_test_matrix_evidence["rows"]) == 1


@pytest.mark.parametrize(
    "seed", [{"anchor_mutation": "bogus"}, {"anchor_mutation": 0}, {"full_round": 5}],
    ids=["invalid-string", "invalid-zero", "future"],
)
def test_959_unusable_prior_anchor_posts_full_comment_anchored_to_own_record(tmp_path, seed):
    plan_context = _followup_matrix_context()
    runner = _seeded_resume_runner_959(plan_context, **seed)
    seeded = _coder_records_959(runner)[0][1]
    if "anchor_mutation" in seed:
        assert seeded.risk_test_matrix_evidence_full_round_status == "invalid"
    else:
        assert seeded.risk_test_matrix_evidence_full_round > seeded.round_number

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3),
        approved_plan_context=plan_context,
    ) == 0

    body, posted = _coder_records_959(runner)[-1]
    assert posted.round_number > seeded.round_number
    assert posted.risk_test_matrix_evidence_full_round == posted.round_number
    assert "<summary>Full matrix evidence (1 row)</summary>" in body
    assert "**followup-derived-evidence**" in body
    assert "unchanged since round" not in body


def test_959_followup_without_evidence_posts_no_section_then_full_on_evidence(tmp_path):
    plan_context = _followup_matrix_context()
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"], summary=f"Round {n}.")
            for n in (1, 2)
        ],
        codex_outputs=_blocking_reviews_959(2),
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)
    real_derive = response_validation._derive_authenticated_risk_evidence_for_coder
    derive_calls = []

    def derive(followup, **kwargs):
        derive_calls.append(followup)
        derived, result = real_derive(followup, **kwargs)
        if len(derive_calls) == 1:
            # First follow-up: no canonical evidence after derivation.
            return dataclasses.replace(derived, risk_test_matrix_evidence=None), result
        return derived, result

    resolve_calls = []
    real_resolve = pr_loop.resolve_matrix_evidence_render

    def resolve(current, previous, current_round):
        assert current is not None
        resolve_calls.append(current_round)
        return real_resolve(current, previous, current_round)

    with patch_object_in((
        response_validation, pr_loop, issue_implementation, issue_loop,
    ), "_derive_authenticated_risk_evidence_for_coder", derive), \
            patch.object(pr_loop, "resolve_matrix_evidence_render", resolve):
        assert run_pr_loop(
            runner, pr_number=77, config=config, approved_plan_context=plan_context
        ) == 0

    records = _coder_records_959(runner)
    assert len(records) == 2
    (first_body, first), (second_body, second) = records
    assert first.risk_test_matrix_evidence is None
    assert first.risk_test_matrix_evidence_full_round is None
    assert first.risk_test_matrix_evidence_full_round_status == "absent"
    assert _EVIDENCE_HEADING_959 not in first_body
    assert "<details>" not in first_body.split("AGENT_LOOP_META", 1)[0]
    assert resolve_calls == [second.round_number]
    assert second.risk_test_matrix_evidence is not None
    assert second.risk_test_matrix_evidence_full_round == second.round_number
    assert "<summary>Full matrix evidence (1 row)</summary>" in second_body
    assert "unchanged since round" not in second_body


from acknowledgement_helpers import (
    ACK_ID, UNTRUSTED_NOTE, dedicated_ack, durable_record, legacy_ack,
    promoted_ack, pr_payload as ack_pr_payload, unrelated_item, ack_resume_history,
)


def _capture_ack_blockers(monkeypatch):
    captured = []
    real = pr_loop._raise_if_maintained_disputed_items
    def capture(items, **kwargs):
        captured.append(tuple(items))
        return real(items, **kwargs)
    monkeypatch.setattr(pr_loop, "_raise_if_maintained_disputed_items", capture)
    return captured


@pytest.mark.parametrize("promoted", [False, True])
@pytest.mark.parametrize("complete_board", [False, True])
@pytest.mark.parametrize("carrier", ["plain", "matrix", "coder-followup-rejected", "head-review-recovery"])
def test_ack_legacy_resume_board_clearance_without_coder_ack(tmp_path, monkeypatch, promoted, complete_board, carrier):
    from acknowledgement_helpers import ack_origin_history
    history = ack_origin_history(ack_resume_history(promoted=promoted), carrier)
    partial_resume = _resume_pr_round(
        [IssueComment(author="bot", created_at=None, body=b) for b in history],
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", configured_reviewers=("codex", "gemini"),
    )
    assert partial_resume.completed_reviews, partial_resume
    snapshots = _capture_ack_blockers(monkeypatch)
    runner = FakeRunner(
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", human_requirements_resolved=complete_board,
            prior_item_dispositions=[
                {"item_id": ACK_ID, "disposition": "resolved", "note": "Ordinary disposition is insufficient."},
                {"item_id": "item-2", "disposition": "resolved", "note": "Keep unrelated machine gate."},
            ],
        )],
        pr_payload=ack_pr_payload(history), advance_pr_head_on_coder_followup=False,
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2)
    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert snapshots
    assert {i.item_id for i in snapshots[-1]} == ({"item-2"} if complete_board else {ACK_ID, "item-2"})
    assert not any(cmd[0] in {"claude", "pytest"} for cmd, _ in runner.commands)
    comments = [IssueComment(author="bot", created_at=None, body=b) for b in history]
    resumed = _resume_pr_round(comments, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", configured_reviewers=("codex", "gemini"))
    assert resumed.next_unresolved_item_number == 14
    assert [d.item_id for d in resumed.completed_reviews[0].metadata.dispositions] == [ACK_ID, "item-2"]
    # Persist the settled ledger just as a later coder checkpoint does, then
    # restart twice. No original comment needs editing or deletion.
    checkpoint = durable_record(snapshots[-1], round_number=3)
    for _ in range(2):
        restart = _resume_pr_round(
            [*comments, IssueComment(author="bot", created_at=None, body=checkpoint)],
            head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", configured_reviewers=("codex", "gemini"),
        )
        assert {i.item_id for i in restart.prior_items} == {i.item_id for i in snapshots[-1]}
        assert restart.next_unresolved_item_number == 14


@pytest.mark.parametrize("promoted", [False, True])
def test_ack_resume_coder_clearance_consumes_alias_replay(tmp_path, monkeypatch, promoted):
    snapshots = _capture_ack_blockers(monkeypatch)
    runner = FakeRunner(
        pr_payload=ack_pr_payload(ack_resume_history(promoted=promoted, coder_ack=True, conflict=True)),
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", human_requirements_resolved=True,
            prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved", "note": "Keep unrelated machine gate."}],
        )],
    )
    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2))
    assert {i.item_id for i in snapshots[-1]} == {"item-2"}


def test_ack_resume_preserves_unrelated_unknown_disposition_error(tmp_path):
    runner = FakeRunner(pr_payload=ack_pr_payload(ack_resume_history(coder_ack=True, bad_id=True)))
    with pytest.raises(AgentLoopError, match="unknown item `item-999`"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2))


def test_ack_conflicting_alias_resume_retains_machine_gate_and_notes(tmp_path, monkeypatch):
    snapshots = _capture_ack_blockers(monkeypatch)
    runner = FakeRunner(
        pr_payload=ack_pr_payload(ack_resume_history(promoted=True, conflict=True)),
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", human_requirements_resolved=True,
            prior_item_dispositions=[
                {"item_id": ACK_ID, "disposition": "resolved", "note": "Acknowledged."},
                {"item_id": "item-2", "disposition": "resolved", "note": "Ordinary disposition."},
            ],
        )],
    )
    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2))
    ack = next(item for item in snapshots[-1] if item.item_id == ACK_ID)
    assert ack.lifecycle == "repair_required"
    assert any("Alias reviewed." in note and "Keep ack gate." in note for note in ack.notes)
    assert {i.item_id for i in snapshots[-1]} == {ACK_ID, "item-2"}


@pytest.mark.parametrize("round_number", [1, 5])
@pytest.mark.parametrize("head", ["6367c48dd193d56ea7b0baad25b19455e529f5ee", "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"])
@pytest.mark.parametrize("prelaunch", [False, True])
def test_ack_failure_boundary_excludes_same_ledger_earlier_reviews(round_number, head, prelaunch):
    prior = (dedicated_ack(), unrelated_item())
    history = [
        durable_record(prior, round_number=round_number),
        durable_record(prior, round_number=round_number, role="reviewer", agent="Codex",
                       state="approved", body=structured_pr_review()),
        durable_record(prior, round_number=round_number, role="summary", agent="Orchestrator",
                       phase="acknowledgement-repair-failed"),
    ]
    from coding_review_agent_loop.round_state import (
        drop_inert_post_reconciliation_checkpoints, _extract_round_metadata_records,
        _latest_prior_pr_subject_is_coherent,
    )
    # Reconciliation precedes the gate; the failure reopens this same round.
    history.insert(-1, durable_record(prior, round_number=round_number,
                   role="summary", agent="Orchestrator", phase="reconciliation"))
    # An older numbered finding must still reserve its number on either head.
    history.insert(0, durable_record((dataclasses.replace(unrelated_item(), item_id="item-99"),),
                                    round_number=round_number))
    if prelaunch:
        history.append(durable_record(prior, round_number=round_number, role="summary",
                                      agent="Orchestrator", phase="scheduler-prelaunch"))
    comments = [IssueComment(author="bot", created_at=None, body=b) for b in history]
    records = _extract_round_metadata_records(comments, flow="pr")
    assert drop_inert_post_reconciliation_checkpoints(records) == records
    assert _latest_prior_pr_subject_is_coherent(records, head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6")
    resumed = _resume_pr_round(comments, head_sha=head, configured_reviewers=("codex",))
    assert resumed.next_unresolved_item_number == 100
    assert resumed.unrecorded_head_advance == (head != "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert resumed.round_number == round_number
    assert resumed.prior_items == prior
    assert resumed.coder_output is None
    assert resumed.completed_reviews == ()
    assert not resumed.reconciled


@pytest.mark.parametrize('fresh', [False, True])
def test_resume_desync_stops_before_entry_writes(tmp_path, fresh):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    runner = FakeRunner()
    runner.branch_tip = 'b' * 40
    config = make_config(tmp_path, managed_ci=fresh, managed_ci_fresh_authorization=fresh, managed_ci_issue_number=56 if fresh else None)
    with pytest.raises(PrHeadObservationError, match='desync'):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.comments
    assert not any(cmd[:2] in (['claude', '--print'], ['codex', 'exec']) for cmd, _ in runner.commands)
    assert not any('--method' in cmd and 'POST' in cmd for cmd, _ in runner.commands)


def test_entry_transient_lag_uses_latest_complete_context(tmp_path):
    runner = FakeRunner(codex_outputs=[structured_pr_review(state='approved')])
    runner.branch_tip = 'b' * 40
    original = runner.run
    def run(args, **kwargs):
        if args == ['sleep', '2']:
            runner.pr_payload.update(headRefOid='b' * 40, title='Coherent title')
        return original(args, **kwargs)
    runner.run = run
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path)) == 0
    prompt = next(cmd[-1] for cmd, _ in runner.commands if cmd[:2] == ['codex', 'exec'])
    assert 'Coherent title' in prompt
    assert 'b' * 40 in prompt
    assert len([cmd for cmd, _ in runner.commands if cmd == ['sleep', '2']]) == 1
