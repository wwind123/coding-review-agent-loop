"""PR-loop tests for round_state and round_transport: PR round resume (#1542)."""

from coding_review_agent_loop import round_metadata

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
from coding_review_agent_loop.qualification_checkpoint import (
    QualificationCheckpoint,
)
from coding_review_agent_loop.round_state import (
    _resume_pr_round,
)
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _strip_round_metadata,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_review,
    parse_unresolved_item_dispositions,
    validate_structured_coder_followup,
)
from coding_review_agent_loop.github import _parse_pr_human_requirements
from pr_loop_test_helpers import _with_code_classification
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


_HEAD_1981 = "6367c48dd193d56ea7b0baad25b19455e529f5ee"


def _carried_only_round_comments_1981(*, checkpoint=None):
    """Round 2 reconciled with a finding; round 3 settled on carried approvals only."""
    item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Codex",
        source_round=2,
        text="Document the retry bound.",
        status="same-pr",
        source_status="same-pr",
    )

    def summary(round_number, phase, prior_items, **extra):
        return _attach_round_metadata(
            f"Round {round_number} {phase}.",
            PostedRoundMetadata(
                flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
                subject=_HEAD_1981, prior_items=prior_items, phase=phase, **extra,
            ),
        )

    bodies = [
        summary(2, "scheduler-prelaunch", ()),
        summary(2, "reconciliation", (), new_items=(item,)),
        summary(3, "scheduler-prelaunch", (item,)),
        summary(3, "reconciliation", (item,)),
    ]
    if checkpoint is not None:
        bodies.append(summary(
            3, "qualification-checkpoint", (item,), state="blocking",
            qualification_checkpoint=checkpoint,
        ))
    return item, [
        IssueComment(author="bot", created_at=f"2026-11-08T00:00:{i:02d}Z", body=body)
        for i, body in enumerate(bodies)
    ]


def test_1981_carried_only_reconciled_round_anchors_resume():
    """Row reader-carried-only-reconciled."""
    item, comments = _carried_only_round_comments_1981()
    resumed = _resume_pr_round(
        comments, head_sha=_HEAD_1981, configured_reviewers=("codex", "gemini"),
    )
    assert resumed is not None
    assert resumed.round_number == 3
    assert resumed.reconciled is True
    assert resumed.completed_reviews == ()
    assert resumed.coder_output is None
    assert resumed.prior_items == (item,)
    # The global finding-ID high-water mark is kept.
    assert resumed.next_unresolved_item_number == 5
    assert resumed.qualification_checkpoint is None


def test_1981_carried_only_reconciled_round_keeps_a_later_qualification_checkpoint():
    """Row reader-carried-only-with-qualification-checkpoint."""
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity="managed-exact-head-ci:item-5",
        lifecycle="repair_required",
        failed_head_sha=_HEAD_1981,
        candidate_head_sha=None,
        base_branch="main",
        watch_failure_extension_used=False,
        allowed_rounds=2,
    )
    item, comments = _carried_only_round_comments_1981(checkpoint=checkpoint)
    resumed = _resume_pr_round(
        comments, head_sha=_HEAD_1981, configured_reviewers=("codex", "gemini"),
    )
    assert resumed is not None
    assert resumed.round_number == 3
    assert resumed.reconciled is True
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

    patch_attr_in(monkeypatch, (validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage, issue_loop), "_run_validated_agent", interrupt_during_coder)
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
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata,
        _attach_round_metadata,
    )

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
        metadata = round_metadata._decode_round_metadata(
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
    import coding_review_agent_loop.dispute_release as dispute_release
    real = dispute_release._raise_if_maintained_disputed_items
    def capture(items, **kwargs):
        captured.append(tuple(items))
        return real(items, **kwargs)
    monkeypatch.setattr(dispute_release, "_raise_if_maintained_disputed_items", capture)
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
    runner.gemini_outputs = [
        _with_code_classification(output, _parse_pr_human_requirements(runner.pr_payload))
        for output in runner.gemini_outputs
    ]
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


@pytest.mark.parametrize("state,deleted", [("CLOSED", False), ("MERGED", True)])
def test_non_open_entry_stops_before_branch_observation(tmp_path, state, deleted):
    runner = FakeRunner(pr_payload={"state": state})
    runner.branch_tip = "b" * 40
    original = runner.run

    def run(args, **kwargs):
        result = original(args, **kwargs)
        if deleted and len(args) > 2 and "/git/ref/heads/" in args[2]:
            return dataclasses.replace(result, returncode=1, stderr="HTTP 404")
        return result

    runner.run = run
    with pytest.raises(AgentLoopError, match=f"PR #77 is {state}; provide an open PR number") as error:
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    assert "force-push" not in str(error.value)
    assert not runner.comments
    assert not any("/git/ref/heads/" in " ".join(cmd) for cmd, _ in runner.commands)
    assert not any(cmd[:1] in (["claude"], ["codex"]) for cmd, _ in runner.commands)


# --- #2035: classification-only resume after a published round -------------

_CLASSIFY = "Continue the PR review with semantic signed-source classification"


def _classification_failed_round(tmp_path, *, codex_state="approved", codex_blocking=(), signed=("Fix the retry bound.",)):
    """Run A: every review is published, then Codex's classification exhausts its re-ask."""
    from pr_loop_test_helpers import _association, _classified_review, _evidence_review, _signed
    runner = FakeRunner()
    for index, body in enumerate(signed, start=1):
        runner.pr_payload["comments"].append(_signed(body, index))
    rids = [r.requirement_id for r in _parse_pr_human_requirements(runner.pr_payload)]
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    published = _evidence_review(state=codex_state, blocking=list(codex_blocking))
    bad = _classified_review([_association(rid, "mixed", leftover="Fix retries.") for rid in rids],
                             state=codex_state, blocking=list(codex_blocking))
    runner.codex_outputs = [published, bad, bad]
    runner.gemini_outputs = [_classified_review([_association(rid, "code") for rid in rids],
                                                reviewer="Google Gemini", hr=True)]
    with pytest.raises(AgentLoopError, match="repair skipped because the reviewer did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.codex_outputs and not runner.gemini_outputs
    return runner, config, rids


def _agent_prompts(runner, agent):
    return [cmd[-1] for cmd, _ in runner.commands if cmd[:1] == [agent]]


def _no_deletes(runner):
    return not any(cmd[:2] == ["gh", "api"] and "DELETE" in cmd for cmd, _ in runner.commands)


def test_2035_rerun_redoes_only_the_missing_classification(tmp_path):
    from pr_loop_test_helpers import _association, _classified_review, _pr_records
    runner, config, rids = _classification_failed_round(tmp_path)
    before = len(runner.commands)
    publications = [r for r in _pr_records(runner) if r.metadata.phase == "publication"]
    assert {r.metadata.agent for r in publications} == {"Codex", "Gemini"}
    runner.codex_outputs = [_classified_review([_association(rid, "code") for rid in rids], hr=True)]
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    rerun = runner.commands[before:]
    codex = [cmd[-1] for cmd, _ in rerun if cmd[:1] == ["codex"]]
    assert len(codex) == 1 and _CLASSIFY in codex[0]
    assert "already published" in codex[0]
    assert not any(cmd[:1] == ["gemini"] for cmd, _ in rerun)
    assert not runner.codex_outputs
    assert _no_deletes(runner)
    records = _pr_records(runner)
    # Publications are kept, never duplicated; Codex's classified settlement follows.
    assert [r.metadata.agent for r in records if r.metadata.phase == "publication"].count("Codex") == 1
    settlement = [r for r in records if r.metadata.agent == "Codex" and r.metadata.role == "reviewer"][-1]
    assert settlement.metadata.phase != "publication"
    assert settlement.metadata.state == "approved"
    assert [e["role"] for e in settlement.metadata.requirement_evidence] == ["code"]


@pytest.mark.parametrize("change", ["approved", "items"])
def test_2035_classification_redo_cannot_change_published_verdict(tmp_path, change):
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from pr_loop_test_helpers import _association, _classified_review
    runner, config, rids = _classification_failed_round(
        tmp_path, codex_state="blocking", codex_blocking=("Fix the retry bound.",))
    entries = [_association(rid, "code") for rid in rids]
    runner.codex_outputs = [
        _classified_review(entries, hr=True) if change == "approved"
        else _classified_review(entries, hr=True, state="blocking", blocking=["A brand new finding."])
    ]
    with pytest.raises(HumanDecisionRequiredError, match="Codex's review was restored"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not any(cmd[:1] == ["claude"] for cmd, _ in runner.commands)
    assert _no_deletes(runner)


def test_2035_diagnosis_unacknowledged_publication_is_the_rejecting_predicate(tmp_path, monkeypatch):
    """Pin Run B: without the classification admission the rerun hits the deletion refusal."""
    from coding_review_agent_loop.protocol import human_requirements_resolved
    from coding_review_agent_loop.review_rounds import PartialReviewRoundError
    from pr_loop_test_helpers import _association, _classified_review, _pr_records
    runner, config, rids = _classification_failed_round(tmp_path)
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    codex = [r for r in _pr_records(runner) if r.metadata.agent == "Codex"][-1]
    assert codex.metadata.phase == "publication"
    assert not human_requirements_resolved(codex.body)
    assert response_validation._resumed_pr_reviewer_identity_matches(codex)
    assert not response_validation._resumed_pr_reviewer_matches_requirements(codex, requirements)
    assert response_validation.classification_resume_kind(codex, requirements) == "pending"
    monkeypatch.setattr(response_validation, "classification_resume_kind", lambda *a, **k: None)
    runner.codex_outputs = [_classified_review([_association(rid, "code") for rid in rids], hr=True)]
    with pytest.raises(PartialReviewRoundError, match="Delete these"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert runner.codex_outputs


@pytest.mark.parametrize("policy", [None, "selective-intermediate"])
def test_2035_repeated_classification_failure_keeps_classified_peer_resumable(tmp_path, policy):
    from pr_loop_test_helpers import _association, _classified_review, _evidence_review, _signed, _pr_records
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("Fix the retry bound.", 1))
    (rid,) = [r.requirement_id for r in _parse_pr_human_requirements(runner.pr_payload)]
    extra = {"pr_review_policy": policy} if policy else {}
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, **extra)
    bad = lambda **kw: _classified_review([_association(rid, "mixed", leftover="Fix retries.")], **kw)
    code = lambda **kw: _classified_review([_association(rid, "code")], **kw)
    # Run A: both publish without classification; Codex's classification fails.
    runner.codex_outputs = [_evidence_review(), bad(), bad()]
    runner.gemini_outputs = [_evidence_review(reviewer="Google Gemini")]
    with pytest.raises(AgentLoopError, match="did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    # Rerun 1: Codex classifies (no acknowledgement); Gemini's classification fails.
    runner.codex_outputs = [code()]
    runner.gemini_outputs = [bad(reviewer="Google Gemini"), bad(reviewer="Google Gemini")]
    with pytest.raises(AgentLoopError, match="did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.codex_outputs and not runner.gemini_outputs
    codex_settlement = [r for r in _pr_records(runner) if r.metadata.agent == "Codex"][-1]
    assert codex_settlement.metadata.phase != "publication"
    assert response_validation.classification_resume_kind(
        codex_settlement, _parse_pr_human_requirements(runner.pr_payload)) == "complete"
    publication = [r for r in _pr_records(runner) if r.metadata.agent == "Codex" and r.metadata.phase == "publication"][-1]
    # The settlement keeps the independent review's acquisition identity, so
    # scheduler/seat acquisition checks still admit it on a later rerun.
    for field in ("provider", "configured_model", "configured_effort", "effort_source",
                  "observed_model", "observed_effort", "acquisition_outcome"):
        assert getattr(codex_settlement.metadata, field) == getattr(publication.metadata, field), field
    assert codex_settlement.metadata.provider == "codex"
    contract = {"Codex": (publication.metadata.configured_model, publication.metadata.configured_effort,
                          publication.metadata.provider)}
    assert response_validation.classification_resume_kind(
        codex_settlement, _parse_pr_human_requirements(runner.pr_payload), None, contract) == "complete"
    # Rerun 2: Codex resumes as complete with no turn; only Gemini is redone.
    before = len(runner.commands)
    before_records = len(_pr_records(runner))
    runner.gemini_outputs = [code(reviewer="Google Gemini", hr=True)]
    # Codex never acknowledged the source, so after the round settles the loop
    # records the failed-acknowledgement boundary and dispatches the coder,
    # whose empty script ends the run; any earlier stop fails the asserts below.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=config)
    rerun = runner.commands[before:]
    agents = [(cmd[0], cmd[-1]) for cmd, _ in rerun if cmd[:1] in (["codex"], ["gemini"], ["claude"])]
    assert [name for name, _ in agents] == ["gemini", "claude"]
    assert _CLASSIFY in agents[0][1]
    assert not runner.gemini_outputs
    assert _no_deletes(runner)
    new_records = _pr_records(runner)[before_records:]
    gemini_settlement = [r for r in new_records if r.metadata.agent == "Gemini" and r.metadata.role == "reviewer"]
    assert len(gemini_settlement) == 1 and gemini_settlement[0].metadata.phase != "publication"
    assert [e["role"] for e in gemini_settlement[0].metadata.requirement_evidence] == ["code"]
    assert not [r for r in new_records if r.metadata.agent == "Codex"]
    phases = [(r.metadata.phase, r.metadata.round_number) for r in new_records if r.metadata.role == "summary"]
    assert ("reconciliation", 1) in phases
    assert ("acknowledgement-repair-failed", 1) in phases
    assert ("coder-dispatch", 2) in phases
    # The resumed Codex record never gains an acknowledgement it did not give.
    from coding_review_agent_loop.protocol import human_requirements_resolved
    assert not any(human_requirements_resolved(r.body) for r in _pr_records(runner) if r.metadata.agent == "Codex")


@pytest.mark.parametrize("keep", [True, False])
def test_2035_complete_reviewer_peer_conflict_continuation_is_pinned(tmp_path, keep):
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from pr_loop_test_helpers import (_association, _classified_review, _evidence_review, _signed,
                                      _EVIDENCE_TEXT, _pr_records)
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy.", 1))
    (rid,) = [r.requirement_id for r in _parse_pr_human_requirements(runner.pr_payload)]
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    bad = _classified_review([_association(rid, "mixed", leftover="Fix retries.")])
    gemini_args = dict(reviewer="Google Gemini", state="blocking", blocking=["Gemini finding."],
                       evidence=[_EVIDENCE_TEXT])
    # Gemini's unacknowledged publication already carries complete associations.
    runner.codex_outputs = [_evidence_review(), bad, bad]
    runner.gemini_outputs = [_classified_review(
        [_association(rid, "human-only", request=_EVIDENCE_TEXT)], **gemini_args)]
    with pytest.raises(AgentLoopError, match="did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    gemini_record = [r for r in _pr_records(runner) if r.metadata.agent == "Gemini"][-1]
    assert response_validation.classification_resume_kind(
        gemini_record, _parse_pr_human_requirements(runner.pr_payload)) == "complete"
    before = len(runner.commands)
    runner.codex_outputs = [_classified_review([_association(rid, "code")], hr=True)]
    reconciled = (_classified_review([_association(rid, "code")], hr=True, **gemini_args) if keep
                  else _classified_review([_association(rid, "code")], hr=True, reviewer="Google Gemini"))
    runner.gemini_outputs = [reconciled]
    if keep:
        try:
            run_pr_loop(runner, pr_number=77, config=config)
        except AgentLoopError as exc:
            assert "restored" not in str(exc) and "partially published" not in str(exc)
        settlement = [r for r in _pr_records(runner) if r.metadata.agent == "Gemini"][-1]
        assert settlement.metadata.state == "blocking"
        assert [e["role"] for e in settlement.metadata.requirement_evidence] == ["code"]
    else:
        with pytest.raises(HumanDecisionRequiredError, match="Gemini's review was restored"):
            run_pr_loop(runner, pr_number=77, config=config)
        claude = [cmd for cmd, _ in runner.commands[before:] if cmd[:1] == ["claude"]]
        assert not claude
    gemini = [cmd[-1] for cmd, _ in runner.commands[before:] if cmd[:1] == ["gemini"]]
    assert len(gemini) == 1 and _CLASSIFY in gemini[0] and "reconciling your role" in gemini[0]
    assert _no_deletes(runner)


def test_2035_closed_round_admits_no_classification_resume(tmp_path, monkeypatch):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    from types import SimpleNamespace as NS
    from pr_loop_test_helpers import _association, _classified_review, _signed, _pr_records
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("Fix the retry bound.", 1))
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    (rid,) = [r.requirement_id for r in requirements]
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    # Codex's publication carries complete associations but no acknowledgement.
    runner.codex_outputs = [_classified_review([_association(rid, "code")])]
    runner.gemini_outputs = [_classified_review([_association(rid, "code")], reviewer="Google Gemini", hr=True)]
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    # Interrupt right after round 1's reconciliation: the round is closed.
    comments = runner.pr_payload["comments"]
    cut = next(i for i, c in enumerate(comments)
               if (recs := _extract_round_metadata_records([NS(body=c["body"])], flow="pr"))
               and recs[0].metadata.phase == "reconciliation")
    del comments[cut + 1:]
    codex = [r for r in _pr_records(runner) if r.metadata.agent == "Codex"][-1]
    assert response_validation.classification_resume_kind(codex, requirements) == "complete"
    kinds, reasons = response_validation.admit_classification_resumes(
        {"Codex": codex}, {}, requirements, None, None, seat_binding=None, items=(),
        round_number=1, head_sha=codex.metadata.subject, reconciled=True)
    assert kinds == {} and reasons == {"Codex": "the round is already reconciled"}
    before_records = len(_pr_records(runner))
    before = len(runner.commands)
    runner.codex_outputs = [_classified_review([_association(rid, "code")], hr=True)] * 3
    runner.gemini_outputs = [_classified_review([_association(rid, "code")], reviewer="Google Gemini", hr=True)] * 3
    calls = []
    real_admit = pr_loop.admit_classification_resumes
    def spy(*args, **kwargs):
        result = real_admit(*args, **kwargs)
        calls.append((kwargs["round_number"], kwargs["reconciled"], dict(result[0])))
        return result
    monkeypatch.setattr(pr_loop, "admit_classification_resumes", spy)
    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError:
        pass
    round_one = [call for call in calls if call[0] == 1]
    assert round_one and all(reconciled and kinds == {} for _, reconciled, kinds in round_one)
    rerun_prompts = [cmd[-1] for cmd, _ in runner.commands[before:] if cmd[:1] in (["codex"], ["gemini"])]
    assert not any(_CLASSIFY in prompt for prompt in rerun_prompts)
    new_round_one_reviews = [r for r in _pr_records(runner)[before_records:]
                             if r.metadata.role == "reviewer" and r.metadata.round_number == 1]
    assert new_round_one_reviews == []
    assert _no_deletes(runner)


def _plan_context(plan_hash):
    return SimpleNamespace(plan_hash=plan_hash, plan_subject="subject-" + plan_hash)


@pytest.mark.parametrize("mismatch", ["plan", "acquisition", "seat", "legacy", "round", "head"])
def test_2035_other_mismatch_is_never_admitted_and_names_its_cause(tmp_path, mismatch):
    from pr_loop_test_helpers import _pr_records
    runner, config, rids = _classification_failed_round(tmp_path)
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    codex = [r for r in _pr_records(runner) if r.metadata.agent == "Codex"][-1]
    assert response_validation.classification_resume_kind(codex, requirements) == "pending"
    plan = acquisition = seat = None
    round_number, head = 1, codex.metadata.subject
    expected = None
    if mismatch == "plan":
        plan, expected = _plan_context("other"), "the approved plan changed since publication"
    elif mismatch == "acquisition":
        acquisition = {"Codex": ("some-other-model", "high")}
        expected = "the reviewer acquisition contract no longer matches"
    elif mismatch == "seat":
        seat, expected = {"seat": "different"}, "the reviewer seat binding changed"
    elif mismatch == "legacy":
        codex = dataclasses.replace(codex, metadata=dataclasses.replace(
            codex.metadata, surfaced_reviewer_requirement_ids=("1",)))
        expected = "its signed requirement coverage uses legacy identities"
    elif mismatch == "round":
        round_number = 2
    else:
        head = "f" * 40
    if mismatch != "seat":  # the seat binding is checked by the admitting caller
        assert response_validation.classification_resume_kind(
            codex, requirements, plan, acquisition, round_number=round_number, head_sha=head) is None
    resumed = {}
    kinds, reasons = response_validation.admit_classification_resumes(
        {"Codex": codex}, resumed, requirements, plan, acquisition, seat_binding=seat, items=(),
        round_number=round_number, head_sha=head, reconciled=False)
    assert kinds == {} and resumed == {}
    if expected is not None:
        assert reasons["Codex"] == expected


def test_2035_refusal_cause_names_the_rejecting_reason(tmp_path):
    from coding_review_agent_loop import review_rounds
    spool = SimpleNamespace()
    cause = review_rounds._replay_unavailable_cause(
        spool, "Codex", own_posted=True, own_posted_reason="the approved plan changed since publication")
    assert "already posted but no longer accepted (resume rejected it because the approved plan changed" in cause


def _round_two_classification_failure(tmp_path, *, policy=None):
    """#2310 shape: round 1 blocks, the coder runs, then round 2's reviews are all
    published (one blocking without acknowledgement) and a classification fails."""
    from pr_loop_test_helpers import _association, _classified_review, _evidence_review, _signed
    runner = FakeRunner()
    runner.advance_pr_head_on_coder_followup = True
    runner.pr_payload["comments"].append(_signed("Fix the retry bound.", 1))
    (rid,) = [r.requirement_id for r in _parse_pr_human_requirements(runner.pr_payload)]
    code = [_association(rid, "code")]
    extra = {"pr_review_policy": policy} if policy else {}
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, max_rounds=3, **extra)
    keep = [{"item_id": "item-1", "disposition": "blocking", "note": "Still unbounded."}]
    bad = _classified_review([_association(rid, "mixed", leftover="Fix retries.")],
                             state="blocking", blocking=["Bound the retries."], dispositions=keep)
    runner.codex_outputs = [
        _classified_review(code, hr=True, state="blocking", blocking=["Bound the retries."]),
        _evidence_review(state="blocking", blocking=["Bound the retries."], dispositions=keep),
        bad, bad,
    ]
    runner.gemini_outputs = [
        _classified_review(code, hr=True, reviewer="Google Gemini"),
        _classified_review(code, hr=True, reviewer="Google Gemini",
                           dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
    ]
    runner.claude_outputs = [structured_coder_followup(
        addressed_items=["item-1"], human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed",
                                         "evidence": "Retry bound added."}])]
    with pytest.raises(AgentLoopError, match="did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.codex_outputs and not runner.gemini_outputs and not runner.claude_outputs
    return runner, config, rid


@pytest.mark.parametrize("policy", [None, "selective-intermediate"])
@pytest.mark.parametrize("parallel", [True, False], ids=["parallel", "sequential"])
def test_2035_round_two_rerun_preserves_records_and_redoes_only_classification(tmp_path, monkeypatch, parallel, policy):
    from pr_loop_test_helpers import _association, _classified_review, _pr_records
    from coding_review_agent_loop.protocol import human_requirements_resolved
    runner, config, rid = _round_two_classification_failure(tmp_path, policy=policy)
    before_comments = [c["body"] for c in runner.pr_payload["comments"]]
    before_records = _pr_records(runner)
    codex_published = [r for r in before_records if r.metadata.agent == "Codex" and r.metadata.round_number == 2]
    assert [r.metadata.phase for r in codex_published] == ["publication"]
    assert codex_published[0].metadata.state == "blocking"
    assert not human_requirements_resolved(codex_published[0].body)
    round_two_summaries = sorted(r.metadata.phase for r in before_records
                                 if r.metadata.role == "summary" and r.metadata.round_number == 2)
    sequential_guard = []
    real_guard = pr_loop._refuse_partial_round_before_sequential_turns
    def guard_spy(**kwargs):
        sequential_guard.append(kwargs)
        return real_guard(**kwargs)
    monkeypatch.setattr(pr_loop, "_refuse_partial_round_before_sequential_turns", guard_spy)
    keep = [{"item_id": "item-1", "disposition": "blocking", "note": "Still unbounded."}]
    runner.codex_outputs = [_classified_review([_association(rid, "code")], hr=True, state="blocking",
                                               blocking=["Bound the retries."], dispositions=keep)]
    before = len(runner.commands)
    rerun_config = dataclasses.replace(config, review_parallel=parallel)
    # The settled blocking round dispatches the coder, whose script is empty:
    # reaching it proves the round finished without any deletion request.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=rerun_config)
    agents = [(cmd[0], cmd[-1]) for cmd, _ in runner.commands[before:] if cmd[:1] in (["codex"], ["gemini"], ["claude"])]
    assert [name for name, _ in agents] == ["codex", "claude"]
    assert _CLASSIFY in agents[0][1] and "already published" in agents[0][1]
    assert bool(sequential_guard) is (not parallel)
    after_comments = [c["body"] for c in runner.pr_payload["comments"]]
    # Every existing record, including the coder-dispatch record and scheduler
    # checkpoints, is kept verbatim: nothing deleted, edited or reordered.
    assert after_comments[:len(before_comments)] == before_comments
    assert _no_deletes(runner)
    new_records = _pr_records(runner)[len(before_records):]
    # Only the per-invocation scheduler audit may be re-posted for round 2; the
    # coder-dispatch record and any other existing checkpoint never are.
    reposted = [r.metadata.phase for r in new_records if r.metadata.round_number == 2
                and r.metadata.phase in round_two_summaries and r.metadata.phase != "reconciliation"]
    assert reposted in ([], ["scheduler-prelaunch"])
    assert "coder-dispatch" not in reposted
    assert not [r for r in new_records if r.metadata.round_number == 2 and r.metadata.phase == "publication"]
    settlement = [r for r in new_records if r.metadata.agent == "Codex" and r.metadata.round_number == 2]
    assert len(settlement) == 1 and settlement[0].metadata.state == "blocking"
    assert [e["role"] for e in settlement[0].metadata.requirement_evidence] == ["code"]
    assert [(i.text) for i in settlement[0].metadata.new_items] == ["Bound the retries."]
    # Round 2 settled: the next coder dispatch is recorded for round 3 (parallel
    # rounds also post their round 2 reconciliation checkpoint first).
    assert [r for r in new_records if r.metadata.round_number == 3 and r.metadata.phase == "coder-dispatch"]
    if parallel:
        assert [r for r in new_records if r.metadata.round_number == 2 and r.metadata.phase == "reconciliation"]
    # Gemini's published approval survives as its round 2 verdict.
    gemini = [r for r in _pr_records(runner) if r.metadata.agent == "Gemini" and r.metadata.round_number == 2]
    assert [r.metadata.state for r in gemini] == ["approved"]


def _keep_through_last_reviewer_record(runner, agent, *, drop_agent):
    """Simulate an interruption right after ``agent``'s latest reviewer record and the
    loss of ``drop_agent``'s settlements, leaving that peer classification-pending."""
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    from types import SimpleNamespace as NS

    def meta(comment):
        records = _extract_round_metadata_records([NS(body=comment["body"])], flow="pr")
        return records[0].metadata if records else None

    comments = runner.pr_payload["comments"]
    last = max(i for i, c in enumerate(comments)
               if (m := meta(c)) is not None and m.role == "reviewer" and m.agent == agent)
    del comments[last + 1:]
    comments[:] = [c for c in comments
                   if not ((m := meta(c)) is not None and m.role == "reviewer" and m.agent == drop_agent
                           and m.phase != "publication")]


def test_2035_complete_authoritative_settlement_persists_reconciled_role(tmp_path):
    """A classification-complete authoritative settlement whose pinned conflict
    continuation reconciles its role posts the new classification, so a later
    rerun restores the reconciled role, not the stale one (#2035)."""
    from pr_loop_test_helpers import (_association, _classified_review, _evidence_review, _signed,
                                      _EVIDENCE_TEXT, _pr_records)
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy.", 1))
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    (rid,) = [r.requirement_id for r in requirements]
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    bad = lambda **kw: _classified_review([_association(rid, "mixed", leftover="Fix retries.")], **kw)
    codex_args = dict(state="blocking", blocking=["Codex finding."])
    gemini_args = dict(reviewer="Google Gemini", state="blocking", blocking=["Gemini finding."])
    code = lambda **kw: _classified_review([_association(rid, "code")], **kw)
    human = lambda **kw: _classified_review([_association(rid, "human-only", request=_EVIDENCE_TEXT)],
                                            evidence=[_EVIDENCE_TEXT], **kw)
    # Run A: both publish without classification; Codex's classification fails.
    runner.codex_outputs = [_evidence_review(**codex_args), bad(**codex_args), bad(**codex_args)]
    runner.gemini_outputs = [_evidence_review(**gemini_args)]
    with pytest.raises(AgentLoopError, match="did not supply requirement_evidence"):
        run_pr_loop(runner, pr_number=77, config=config)
    # Rerun 1: both classify code without acknowledgement and settle.
    runner.codex_outputs = [code(**codex_args)]
    runner.gemini_outputs = [code(**gemini_args)]
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    _keep_through_last_reviewer_record(runner, "Gemini", drop_agent="Codex")
    gemini_settlement = [r for r in _pr_records(runner) if r.metadata.agent == "Gemini"][-1]
    assert gemini_settlement.metadata.phase != "publication"
    assert response_validation.classification_resume_kind(gemini_settlement, requirements) == "complete"
    settled_items = [(i.item_id, i.text) for i in gemini_settlement.metadata.new_items]
    assert settled_items
    # Rerun 2: Codex now classifies human-only; Gemini's complete authoritative
    # settlement meets the conflict and reconciles to human-only, keeping its verdict.
    before_records = len(_pr_records(runner))
    before = len(runner.commands)
    runner.codex_outputs = [human(**codex_args)]
    runner.gemini_outputs = [human(**gemini_args)]
    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError:
        pass  # the human-only evidence or the empty coder script stops the run
    agents = [(cmd[0], cmd[-1]) for cmd, _ in runner.commands[before:] if cmd[:1] in (["codex"], ["gemini"])]
    assert [name for name, _ in agents] == ["codex", "gemini"]
    assert "reconciling your role" in agents[1][1]
    assert not runner.codex_outputs and not runner.gemini_outputs
    reconciled = [r for r in _pr_records(runner)[before_records:]
                  if r.metadata.agent == "Gemini" and r.metadata.role == "reviewer"]
    assert len(reconciled) == 1
    reconciled = reconciled[0].metadata
    assert reconciled.phase != "publication" and reconciled.state == "blocking"
    assert [e["role"] for e in reconciled.requirement_evidence] == ["human-only"]
    # Existing finding IDs are kept, not re-minted, and acquisition identity survives.
    assert [(i.item_id, i.text) for i in reconciled.new_items] == settled_items
    for field in ("provider", "configured_model", "configured_effort", "acquisition_outcome"):
        assert getattr(reconciled, field) == getattr(gemini_settlement.metadata, field), field
    # A peer failure leaves Codex pending again; the next rerun restores Gemini's
    # reconciled role, so no further Gemini continuation is needed.
    _keep_through_last_reviewer_record(runner, "Gemini", drop_agent="Codex")
    before = len(runner.commands)
    runner.codex_outputs = [human(**codex_args)]
    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError as exc:
        assert "classification remained conflicting" not in str(exc)
        assert "scripted agent output exhausted" not in str(exc) or not any(
            cmd[:1] == ["gemini"] for cmd, _ in runner.commands[before:])
    assert not any(cmd[:1] == ["gemini"] for cmd, _ in runner.commands[before:])
    assert not runner.codex_outputs
    assert _no_deletes(runner)
