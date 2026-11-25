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
from coding_review_agent_loop.round_metadata import (
    _decode_round_metadata,
)
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
    from coding_review_agent_loop.round_metadata import (
        PostedRoundRecord,
    )

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
