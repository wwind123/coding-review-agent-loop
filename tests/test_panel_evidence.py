"""Panel-evidence tests: canonical plan approval gates and plan-round evidence (#1543)."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.plan_review_scheduling import PlanCandidateKey
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from issue_loop_test_helpers import (
    _invalid_plan_scheduler_comment,
    _staged_plan_config,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
from coding_review_agent_loop.round_state import PostedRoundRecord  # noqa: E402


def _completed_staged_plan_comments(tmp_path):
    """Posted comments of a primary-then-panel plan approved across two rounds."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    return config, [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments]


def test_managed_ci_plan_approval_accepts_staged_approvals_across_rounds(tmp_path):
    """#962: the primary approves in round 1 and the panel in round 2."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )
    # The resumed round alone holds only the panel's approval.
    assert {
        record.metadata.agent
        for record in plan_round.completed_reviews
        if record.metadata.state == "approved"
    } == {"Gemini"}

    panel_evidence._require_complete_canonical_plan_approval(
        comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=(),
        error_message="incomplete",
    )


def test_managed_ci_plan_approval_rejects_superseded_staged_approval(tmp_path):
    """A later non-approving record for the exact plan revokes the carry."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    revoked = []
    for comment in comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment.body)
        if match is not None:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.flow == "plan" and metadata.role == "reviewer" and metadata.agent == "Codex":
                revoked.append(
                    SimpleNamespace(
                        body=_attach_round_metadata(
                            "Codex plan review.\n-- Codex",
                            replace(metadata, state="blocking"),
                        )
                    )
                )
    assert len(revoked) == 1
    comments = [*comments, *revoked]
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )

    with pytest.raises(AgentLoopError, match="incomplete"):
        panel_evidence._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


def test_managed_ci_plan_approval_staged_full_board_round_requires_exact_key(tmp_path):
    """A staged full-board round still needs approvals bound to the current key.

    Surfacing a signed planning requirement after the approval changes the
    candidate key, so the same-round full set must not satisfy recovery.
    """
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path, plan_review_force_full=True)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    comments = [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments]
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )
    # The operator force-full run put the whole board in one round.
    assert {
        record.metadata.agent
        for record in plan_round.completed_reviews
        if record.metadata.state == "approved"
    } == {"Codex", "Gemini"}

    panel_evidence._require_complete_canonical_plan_approval(
        comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=(),
        error_message="incomplete",
    )
    requirement = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
        body="Keep the public API unchanged.",
    )
    with pytest.raises(AgentLoopError, match="incomplete"):
        panel_evidence._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(requirement,),
            error_message="incomplete",
        )


def _assert_managed_ci_plan_recovery_fails_closed(config, comments):
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )
    with pytest.raises(AgentLoopError, match="incomplete"):
        panel_evidence._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


@pytest.mark.parametrize("history", ["open", "closed", "closed-matching-idle"])
def test_managed_ci_plan_approval_rejects_post_approval_contradictory_key(tmp_path, history):
    """A later checkpoint contradicting the current key voids carried approvals."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    latest_checkpoint = None
    for comment in comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment.body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow == "plan" and metadata.scheduler_metadata_status == "valid":
            latest_checkpoint = metadata
    assert latest_checkpoint is not None
    stored_key = panel_evidence._plan_key_from_payload(
        latest_checkpoint.plan_candidate_key
    )
    contradictory = replace(stored_key, aggregate_plan_identity="f" * 64)
    assert contradictory.subject == stored_key.subject
    if history != "open":
        comments.append(SimpleNamespace(body=_attach_round_metadata(
            "Settled planning round", PostedRoundMetadata(
                flow="plan", role="summary", agent="Orchestrator",
                round_number=latest_checkpoint.round_number, subject=stored_key.subject,
                phase="reconciliation",
            )
        )))
    comments = [
        *comments,
        SimpleNamespace(
            body=_attach_round_metadata(
                "Plan review scheduling audit.\n\n-- Orchestrator",
                replace(latest_checkpoint, phase="scheduler-prelaunch",
                        plan_candidate_key=contradictory.as_dict()),
            )
        ),
    ]

    if history == "closed-matching-idle":
        comments.append(SimpleNamespace(body=_attach_round_metadata(
            "Idle matching checkpoint", replace(latest_checkpoint, phase="scheduler-prelaunch")
        )))
    _assert_managed_ci_plan_recovery_fails_closed(config, comments)


def test_managed_ci_plan_approval_rejects_post_boundary_invalid_checkpoint(tmp_path):
    """An invalid checkpoint after the recovery boundary degrades the history."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    plan_text, _plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )
    invalid = _invalid_plan_scheduler_comment(_plan_subject(plan_text))
    comments = [*comments, SimpleNamespace(body=invalid["body"])]

    _assert_managed_ci_plan_recovery_fails_closed(config, comments)


def test_managed_ci_plan_approval_all_reviewers_still_requires_one_round(tmp_path):
    """The compatibility policy keeps requiring the full set in one round."""
    _config, comments = _completed_staged_plan_comments(tmp_path)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=6)
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config)
    )

    with pytest.raises(AgentLoopError, match="incomplete"):
        panel_evidence._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


def test_operator_opening_requires_a_complete_candidate_key(tmp_path):
    """A partial-key record can never establish a qualified panel opening."""
    from coding_review_agent_loop.round_state import PostedRoundRecord

    partial = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity=None,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest="e" * 16,
    )
    complete = replace(partial, execution_strategy_identity="c" * 32)

    def _force_full_record(index, key):
        return PostedRoundRecord(
            index=index,
            body="Plan review scheduling audit.",
            metadata=PostedRoundMetadata(
                flow="plan",
                role="summary",
                agent="Orchestrator",
                round_number=1,
                subject="a" * 64,
                phase="scheduler-prelaunch",
                scheduler_contract=panel_evidence.make_plan_contract(
                    ("Codex", "Gemini"), "primary-then-panel", "Codex"
                ).as_dict(),
                scheduler_obligation_digest="0" * 16,
                scheduler_selected_reviewers=("Codex", "Gemini"),
                scheduler_paused_reviewers=(),
                scheduler_reasons=("operator force-full",),
                scheduler_final_sweep=False,
                scheduler_force_full=True,
                scheduler_force_full_source="operator",
                scheduler_calls_avoided=0,
                scheduler_phase="full-board",
                scheduler_primary_reviewer="Codex",
                plan_candidate_key=key.as_dict(),
            ),
        )

    partial_evidence = panel_evidence._derive_plan_panel_evidence(
        (_force_full_record(1, partial),),
        primary_reviewer="Codex",
        required_reviewers=("Codex", "Gemini"),
    )
    complete_evidence = panel_evidence._derive_plan_panel_evidence(
        (_force_full_record(1, complete),),
        primary_reviewer="Codex",
        required_reviewers=("Codex", "Gemini"),
    )

    assert partial_evidence.opened is False
    assert complete_evidence.opened is True
    assert complete_evidence.opening_source == "operator"


# Streak derivation over synthetic records.


def _m1103_checkpoint(index, round_number, *, phase="primary", status="valid"):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject="s",
            phase="scheduler-prelaunch",
            scheduler_phase=phase if status == "valid" else None,
            scheduler_metadata_status=status,
        ),
        body="",
    )


def _m1103_review(index, round_number, state="blocking", *, agent="Codex"):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent=agent,
            round_number=round_number,
            subject="s",
            state=state,
        ),
        body="",
    )


def _m1103_rounds(*states, start=1):
    records = []
    for offset, state in enumerate(states):
        number = start + offset
        records.append(_m1103_checkpoint(len(records), number))
        if state is not None:
            records.append(_m1103_review(len(records), number, state))
    return records


def _m1103_streak(records, **kwargs):
    return panel_evidence.plan_primary_blocking_streak(
        records, primary="Codex", **kwargs
    )


def test_stall_streak_counts_trailing_blocking_primary_reviews():
    assert _m1103_streak(_m1103_rounds("blocking", "blocking", "blocking")) == 3
    assert _m1103_streak([]) == 0


@pytest.mark.parametrize("state", ["approved", "unavailable", None])
def test_stall_streak_resets_on_any_non_blocking_primary_review(state):
    """`stall-growth-gated-approval-resets`: any approval, any key, resets."""
    if state is None:
        records = _m1103_rounds("blocking", "blocking")
        records.append(_m1103_checkpoint(len(records), 3))
        records.append(_m1103_review(len(records), 3, "blocking"))
        # An absent state on the primary's record is not evidence of a block.
        records[-1] = PostedRoundRecord(
            index=records[-1].index,
            metadata=replace(records[-1].metadata, state=None),
            body="",
        )
        assert _m1103_streak(records) == 0
        return
    records = _m1103_rounds("blocking", state, "blocking", "blocking")
    assert _m1103_streak(records) == 2


def test_stall_streak_skips_a_round_with_no_primary_review():
    records = _m1103_rounds("blocking", "blocking", None)
    assert _m1103_streak(records) == 2
    records = _m1103_rounds("blocking", None, "blocking")
    assert _m1103_streak(records) == 2


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"phase": "secondary-audit"},
        {"phase": "remediation"},
        {"phase": "full-board"},
        {"phase": "final-secondary-sweep"},
        {"status": "invalid"},
    ],
)
def test_stall_streak_ends_at_a_non_primary_or_invalid_checkpoint(checkpoint):
    records = _m1103_rounds("blocking", "blocking")
    records.append(_m1103_checkpoint(len(records), 3, **checkpoint))
    records.append(_m1103_review(len(records), 3, "blocking"))
    records.extend(_m1103_rounds("blocking", start=4))
    records = [
        PostedRoundRecord(index=i, metadata=r.metadata, body="")
        for i, r in enumerate(records)
    ]
    assert _m1103_streak(records) == 1


def test_stall_streak_ends_at_a_missing_checkpoint_and_ignores_secondaries():
    records = _m1103_rounds("blocking")
    records.append(_m1103_review(len(records), 2, "blocking"))
    records.extend(_m1103_rounds("blocking", start=3))
    records = [
        PostedRoundRecord(index=i, metadata=r.metadata, body="")
        for i, r in enumerate(records)
    ]
    assert _m1103_streak(records) == 1
    with_secondary = _m1103_rounds("blocking", "blocking")
    with_secondary.append(_m1103_review(len(with_secondary), 2, "approved", agent="Gemini"))
    assert _m1103_streak(with_secondary) == 2


def test_stall_streak_ends_at_invalid_records_phase_advances_and_openings():
    records = _m1103_rounds("blocking", "blocking")
    records.append(
        PostedRoundRecord(
            index=len(records),
            metadata=PostedRoundMetadata(
                flow="plan",
                role="summary",
                agent="Orchestrator",
                round_number=3,
                subject="s",
                phase="plan-phase-advance",
            ),
            body="",
        )
    )
    tail = _m1103_rounds("blocking", start=3)
    records.extend(
        PostedRoundRecord(index=len(records) + i, metadata=r.metadata, body="")
        for i, r in enumerate(tail)
    )
    assert _m1103_streak(records) == 1
    # A trailing invalid scheduler record ends the streak even when it is not
    # the checkpoint any primary review is bound to.
    degraded = _m1103_rounds("blocking", "blocking")
    degraded.append(_m1103_checkpoint(len(degraded), 1, status="invalid"))
    assert _m1103_streak(degraded) == 0
    # Anything at or after a qualified panel opening is excluded.
    opened = _m1103_rounds("blocking", "blocking")
    assert _m1103_streak(opened, panel_opening_index=2) == 0


def test_stall_streak_keeps_the_last_record_of_a_repeated_round():
    records = _m1103_rounds("blocking")
    records.append(_m1103_checkpoint(len(records), 2))
    records.append(_m1103_review(len(records), 2, "approved"))
    records.append(_m1103_checkpoint(len(records), 2))
    records.append(_m1103_review(len(records), 2, "blocking"))
    assert _m1103_streak(records) == 2


def _m1112_digest_record(index, round_number, digest, *, reset=False):
    record = _m1103_checkpoint(index, round_number)
    return PostedRoundRecord(
        index=index,
        metadata=replace(
            record.metadata,
            scheduler_issue_digest=digest,
            scheduler_stall_reset=reset,
        ),
        body="",
    )


def test_streak_ends_at_a_digest_mismatch_and_reaccumulates():
    """`streak-reaccumulates`."""
    records = []
    for number, digest in ((1, "a" * 16), (2, "a" * 16), (3, "b" * 16), (4, "b" * 16)):
        records.append(_m1112_digest_record(len(records), number, digest))
        records.append(_m1103_review(len(records), number, "blocking"))
    detail = panel_evidence.plan_primary_blocking_streak_detail
    assert detail(records, primary="Codex", current_issue_digest="b" * 16).count == 2
    assert detail(records, primary="Codex", current_issue_digest="a" * 16).count == 0
    # No current digest: nothing can mismatch.
    assert detail(records, primary="Codex").count == 4
    assert not detail(records, primary="Codex", current_issue_digest="b" * 16).legacy_undigested


def test_absent_digest_counts_and_is_flagged_legacy():
    records = _m1103_rounds("blocking", "blocking")
    detail = panel_evidence.plan_primary_blocking_streak_detail(
        records, primary="Codex", current_issue_digest="c" * 16
    )
    assert detail.count == 2 and detail.legacy_undigested


def test_mixed_history_flags_legacy_only_when_the_newest_round_is_undigested():
    """An edit ends the streak at a newer digested round; no reset is needed."""
    detail = panel_evidence.plan_primary_blocking_streak_detail
    old = _m1103_rounds("blocking", "blocking")
    records = list(old)
    records.append(_m1112_digest_record(len(records), 3, "a" * 16))
    records.append(_m1103_review(len(records), 3, "blocking"))
    unchanged = detail(records, primary="Codex", current_issue_digest="a" * 16)
    assert unchanged.count == 3 and not unchanged.legacy_undigested
    edited = detail(records, primary="Codex", current_issue_digest="b" * 16)
    assert edited.count == 0
    # Newest counted round undigested: an edit cannot retire it.
    newest_legacy = [
        _m1112_digest_record(0, 1, "a" * 16),
        _m1103_review(1, 1, "blocking"),
        _m1103_checkpoint(2, 2),
        _m1103_review(3, 2, "blocking"),
    ]
    flagged = detail(newest_legacy, primary="Codex", current_issue_digest="b" * 16)
    assert flagged.count == 1 and flagged.legacy_undigested
    assert flagged.undigested_prefix == 1


def test_mixed_history_warns_only_when_undigested_rounds_alone_reach_threshold():
    """Older digested round then a newer undigested one: an edit still clears."""
    detail = panel_evidence.plan_primary_blocking_streak_detail
    message = panel_evidence.plan_primary_stall_message
    threshold = 2
    records = [
        _m1112_digest_record(0, 1, "a" * 16),
        _m1103_review(1, 1, "blocking"),
        _m1103_checkpoint(2, 2),
        _m1103_review(3, 2, "blocking"),
    ]
    tripped = detail(records, primary="Codex", current_issue_digest="a" * 16)
    assert tripped.count == threshold
    assert tripped.undigested_prefix == 1
    assert not tripped.edit_cannot_clear(threshold)
    text = message(
        streak=tripped.count,
        threshold=threshold,
        plan_chars=1,
        legacy_undigested=tripped.edit_cannot_clear(threshold),
    )
    assert "predate issue-text tracking" not in text
    assert "edit the issue title or body" in text
    # The advertised edit really clears the stop: only the undigested round
    # is left counting, below the threshold.
    edited = detail(records, primary="Codex", current_issue_digest="b" * 16)
    assert edited.count == 1 < threshold

    # Two undigested newest rounds alone reach the threshold: an edit leaves
    # the stop tripped, so the diagnostic must require the reset flag.
    legacy_heavy = [
        _m1112_digest_record(0, 1, "a" * 16),
        _m1103_review(1, 1, "blocking"),
        _m1103_checkpoint(2, 2),
        _m1103_review(3, 2, "blocking"),
        _m1103_checkpoint(4, 3),
        _m1103_review(5, 3, "blocking"),
    ]
    heavy = detail(legacy_heavy, primary="Codex", current_issue_digest="a" * 16)
    assert heavy.count == 3 and heavy.undigested_prefix == 2
    assert heavy.edit_cannot_clear(threshold)
    assert not heavy.edit_cannot_clear(3)
    edited_heavy = detail(legacy_heavy, primary="Codex", current_issue_digest="b" * 16)
    assert edited_heavy.count == 2 >= threshold
    heavy_text = message(
        streak=heavy.count,
        threshold=threshold,
        plan_chars=1,
        legacy_undigested=heavy.edit_cannot_clear(threshold),
    )
    assert "predate issue-text tracking" in heavy_text
    assert "--plan-reset-stall-streak is required" in heavy_text


def test_reset_checkpoint_is_a_boundary_with_or_without_a_review():
    old = _m1103_rounds("blocking", "blocking")
    reviewed = [*old, _m1112_digest_record(len(old), 3, None, reset=True)]
    reviewed.append(_m1103_review(len(reviewed), 3, "blocking"))
    assert _m1103_streak(reviewed) == 1
    # Interrupted after the checkpoint: the reset still shields older rounds.
    interrupted = [*old, _m1112_digest_record(len(old), 3, None, reset=True)]
    assert _m1103_streak(interrupted) == 0
    # A flagless restart re-posting an unmarked checkpoint keeps the boundary.
    restarted = [*interrupted, _m1103_checkpoint(len(interrupted), 3)]
    assert _m1103_streak(restarted) == 0
    later = [*reviewed, *_m1103_rounds("blocking", "blocking", start=4)]
    later = [
        record if i < len(reviewed) else PostedRoundRecord(
            index=i, metadata=record.metadata, body=""
        )
        for i, record in enumerate(later)
    ]
    assert _m1103_streak(later) == 3


def test_stall_message_names_every_working_remedy():
    message = panel_evidence.plan_primary_stall_message(
        streak=8, threshold=8, plan_chars=90212
    )
    for needle in (
        "edit the issue title or body",
        "a comment alone does not",
        "--plan-reset-stall-streak",
        "--plan-review-force-full",
        "--plan-primary-stall-rounds",
        "no reviewer and no planner turn were invoked",
    ):
        assert needle in message
    assert "predate issue-text tracking" not in message
    legacy = panel_evidence.plan_primary_stall_message(
        streak=8, threshold=8, plan_chars=1, legacy_undigested=True
    )
    assert "predate issue-text tracking" in legacy
    assert "Plan step-back did not apply" not in message
    suffixed = panel_evidence.plan_primary_stall_message(
        streak=8, threshold=8, plan_chars=1, step_back_status="plan-growth gate is off"
    )
    assert suffixed.endswith(" Plan step-back did not apply: plan-growth gate is off.")
    for needle in ("--plan-reset-stall-streak", "--plan-review-force-full"):
        assert needle in suffixed


def test_streak_ends_at_an_execution_mode_mismatch_and_missing_modes_count():
    """`streak-mode-boundary`."""
    records = []
    for number, mode in ((1, "plan-only"), (2, "plan-only"), (3, "decompose-only"), (4, None)):
        checkpoint = _m1103_checkpoint(len(records), number)
        records.append(
            PostedRoundRecord(
                index=checkpoint.index,
                metadata=replace(checkpoint.metadata, scheduler_execution_mode=mode),
                body="",
            )
        )
        records.append(_m1103_review(len(records), number, "blocking"))
    detail = panel_evidence.plan_primary_blocking_streak_detail
    # Newest first: round 4 (no mode) counts, round 3 matches, round 2 ends it.
    assert detail(records, primary="Codex", current_execution_mode="decompose-only").count == 2
    assert detail(records, primary="Codex", current_execution_mode="plan-only").count == 1
    # No current mode: nothing can mismatch.
    assert detail(records, primary="Codex").count == 4
