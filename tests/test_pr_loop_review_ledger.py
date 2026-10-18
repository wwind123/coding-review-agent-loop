"""PR-loop tests for unresolved_items and review_rounds: the PR review ledger (#1542)."""

import dataclasses
import json
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _strip_round_metadata,
)
from coding_review_agent_loop.prompts import (
    COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER,
)
from coding_review_agent_loop.review_scheduling import TransitionClassification
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_unresolved_item_dispositions,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.integration_close as integration_close
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _agent_sequence,
    _audit_phases,
    _staged_config,
    _staged_review,
)


def test_pr_loop_keeps_blocking_review_with_off_scope_future_followups(tmp_path):
    # Issue #1389: a blocking review may carry an off-scope future follow-up.
    # It is rendered, never sent to the coder, and filed at approval.
    runner = FakeRunner(
        gemini_outputs=[
            "Still blocked.\n\n"
            "### Same-PR follow-ups\n"
            "- Tighten the reset helper.\n\n"
            "### Future follow-ups\n"
            "- Consider a broader cleanup later.\n\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- Google Gemini",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved", "[item-2] future follow-up: still later work")
            + "\n<!-- AGENT_STATE: approved -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved", "[item-2] future follow-up: still later work")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Fixed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("gemini", "codex"),
        approved_followups="fix-and-issue",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "### Future follow-ups\n- Consider a broader cleanup later." in runner.comments[0]
    assert [issue["title"] for issue in runner.issues] == [
        "Follow up future review note: Consider a broader cleanup later."
    ]
    # Only the same-PR item ("Tighten the reset helper.") is tracked -- the
    # blocking summary prose ("Still blocked.") is never itemized when a
    # same_pr_followups entry already represents the actionable concern, so
    # this routes through the lean same-PR-only coder prompt.
    followup_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Tighten the reset helper." in cmd[-1]
    )
    assert "Address the follow-up items below" in followup_prompt
    assert "Still blocked." in followup_prompt
    assert "Latest reviewer summaries (review-level context):" in followup_prompt
    assert "Consider a broader cleanup later." not in followup_prompt

def test_pr_loop_requires_all_reviewers_to_approve(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Codex approves.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        claude_outputs=["Claude approves.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [["codex", "exec"], ["claude", "--print"]]
    assert len(runner.comments) == 3
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 1
    assert ["pytest", "tests/test_agent_loop.py"] in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge", "--match-head-commit", "abc123"] in commands


def test_pr_loop_skips_prior_approval_when_pr_head_is_unchanged(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves the initial head.",
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Claude needs one fix.",
                blocking_items=["Fix the admission cleanup race."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="Claude accepts the follow-up.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                addressed_items=["item-1"],
                remaining_items=[],
                reviewer="Google Gemini",
            )
        ],
        advance_pr_head_on_coder_followup=False,
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [
        cmd
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"]
    ]
    assert len(codex_reviews) == 1
    assert "round 1" in codex_reviews[0][-1]

def test_pr_loop_reruns_all_reviewers_when_any_reviewer_blocks(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("claude", "codex"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(runner.comments) == 6
    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Needs a regression test." in followup_prompt
    assert "Codex approves first pass." in followup_prompt
    assert "Latest reviewer summaries (review-level context):" in followup_prompt
    commands = [cmd for cmd, _cwd in runner.commands]
    metadata_fetches = [
        cmd
        for cmd in commands
        if cmd[:3] == ["gh", "pr", "view"]
        and "--json" in cmd
        and cmd[cmd.index("--json") + 1]
        == "number,title,headRefName,baseRefName,headRefOid,url,body,comments,reviews"
    ]
    assert len(metadata_fetches) == 2

def test_pr_loop_rejects_cross_reviewer_approval_without_prior_item_disposition(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude resolves it."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Codex approves first pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves second pass.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            # The omission is re-asked once (#1167); it still omits item-1.
            "Codex approves second pass again.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=["Implemented fix.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, coder="gemini", reviewer=("claude", "codex"), max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved items: item-1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_codex_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 2" in cmd[-1]
    ][0]
    assert "Prior unresolved review items from earlier rounds" in second_codex_prompt
    assert "[item-1] blocking from Claude in round 1" in second_codex_prompt

def test_pr_loop_can_downgrade_prior_blocker_to_future_followup_only_in_approved_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM now."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, approved_followups="summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary

def test_pr_loop_persists_downgraded_future_followup_across_later_blocking_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented fix for Claude.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] future follow-up: cleanup can wait", "[item-2] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Missing docs cleanup." in summary

def test_pr_loop_finalized_future_followup_summary_preserves_disposition_notes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Missing docs cleanup.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Implemented blocker.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: cleanup can wait until after rollout",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "claude"),
        coder="codex",
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    summary = runner.comments[-1]
    assert "Missing docs cleanup." in summary
    assert "Update from Codex: cleanup can wait until after rollout" in summary

def test_pr_loop_compact_review_mode_uses_fresh_sessions_and_compact_prior_ledger(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves with future work.\n\n"
            "### Future follow-ups\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude still blocks.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Implemented blocker and ran focused tests.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
                tests_run=["python -m pytest tests/test_agent_loop.py -k compact_pr"],
                reviewer="Google Gemini",
            )
        ],
        pr_payload={"body": "PR body used by compact review mode."},
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="summarize",
        max_rounds=2,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_prompts) == 2
    second_codex_prompt = codex_prompts[1]
    assert COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER in second_codex_prompt
    assert "PR body used by compact review mode." in second_codex_prompt
    assert "Implemented blocker and ran focused tests." in second_codex_prompt
    assert "python -m pytest tests/test_agent_loop.py -k compact_pr" in second_codex_prompt
    assert "Document cache cleanup behavior." not in second_codex_prompt
    assert "[item-1] future" not in second_codex_prompt
    assert "Claude still blocks." in second_codex_prompt
    assert not any("--resume" in cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])

    assert runner.comments[-1].startswith("Approved-review future follow-ups for PR #77:")
    assert "Document cache cleanup behavior." in runner.comments[-1]

def test_pr_loop_carries_prior_item_notes_without_creating_duplicate_blocker_items(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Still blocked."
            + prior_item_dispositions("[item-1] still blocking: include API error path too")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Expanded coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][1]
    assert "Latest reviewer updates:" in second_coder_prompt
    assert "Codex: include API error path too" in second_coder_prompt
    assert "[item-2]" not in second_coder_prompt

def test_pr_loop_stops_on_incomplete_review_without_coder_followup(tmp_path):
    """An explicit inability to review is an agent failure, not a new blocker."""
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Review incomplete; item-1 requires further inspection of the PR diff "
            "and referenced files before a disposition can be confirmed."
            + prior_item_dispositions(
                "[item-1] resolved: Resolution could not be confirmed; "
                "PR diff and referenced files need review."
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Added regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_pr_loop(runner, pr_number=77, config=config)

    coder_commands = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(coder_commands) == 1
    assert len(runner.comments) == 3
    assert "Review incomplete" not in "\n".join(runner.comments)


def test_pr_loop_keeps_active_carried_disposition_actionable_without_duplicate_summary_item(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Initial scope concern.",
                blocking_items=["All 14 locale catalogs must be evaluated."],
            ),
            structured_pr_review(
                state="blocking", summary="Unable to verify the translation detail today.",
                prior_item_dispositions=[{
                    "item_id": "item-1", "disposition": "blocking",
                    "note": "The original scope-completeness claim remains unresolved.",
                }],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Evaluated every locale.",
                addressed_items=["item-1"], remaining_items=[],
            ),
            structured_coder_followup(
                state="blocking", summary="Need more evidence for the carried claim.",
                addressed_items=[], remaining_items=["item-1"],
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=2)

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    reviewer_records = [
        _decode_round_metadata(discuss_loop.ROUND_RESUME_MARKER_RE.search(comment["body"])["payload"])
        for comment in runner.pr_payload["comments"]
        if discuss_loop.ROUND_RESUME_MARKER_RE.search(comment["body"])
    ]
    round_two = next(record for record in reviewer_records if record.agent == "Codex" and record.round_number == 2)
    assert round_two.new_items == ()
    assert round_two.dispositions[0].disposition == "blocking"
    assert not any("agent-unavailable" in comment["body"] for comment in runner.pr_payload["comments"])

def test_pr_loop_quarantines_unavailable_reviewer_while_healthy_reviewer_finishes(tmp_path):
    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[unavailable],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="A regression test is required.",
                blocking_items=["Add coverage for the unavailable-reviewer path."],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(
                state="approved",
                summary="The healthy-reviewer finding is resolved.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Anthropic Claude",
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Added the requested regression coverage.",
                addressed_items=["item-1"],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=3,
    )

    with pytest.raises(AgentLoopError, match="missing required input from Codex") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    claude_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    gemini_coder_turns = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    # Decided at detection (#1129): no coder follow-up and no later round.
    assert len(codex_reviews) == 1
    assert len(claude_reviews) == 1
    assert gemini_coder_turns == []
    message = str(excinfo.value)
    assert "Detected in round 1" in message
    assert "Blocking reviewers this round: Claude" in message
    assert "Open must-fix items (including carried): 1" in message
    assert "--reviewer flags" in message
    assert "Signed amendment template" not in message
    incomplete = [body for body in runner.comments if "**Review status: Incomplete**" in body]
    assert len(incomplete) == 1
    assert "- Codex: agent-unavailable" in incomplete[0]
    assert "Claude" in incomplete[0]
    assert "--reviewer flags" in incomplete[0]
    assert "AGENT_" not in incomplete[0]
    assert "{" not in incomplete[0]
    # The advertised route must not contradict the policy-specific remedy.
    assert "replacement reviewer" not in incomplete[0]
    reconciliations = [body for body in runner.comments if "reconciliation" in body]
    assert reconciliations, "sequential stop must post a reconciliation record"
    assert "Finalization stops" in reconciliations[0]
    assert "Finalization continues" not in reconciliations[0]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_incomplete_review_comment_reports_zero_open_must_fix_count():
    comment = review_rounds._format_incomplete_pr_review_comment(
        pr_number=5,
        unavailable_reviewers={
            "codex": __import__("coding_review_agent_loop.errors", fromlist=["x"]).AgentInvocationError("down", failure_category="agent-unavailable")
        },
        approved_reviewer_names=["Claude"],
        detection_round=2,
        open_must_fix_count=0,
        remedy_lines=["- Restore the reviewer backend and rerun unchanged."],
    )
    assert "### Open findings" in comment
    assert "Open must-fix items (including carried): 0" in comment
    assert "replacement reviewer" not in comment


def test_pr_loop_retries_explicit_retryable_agent_unavailable_response(tmp_path):
    retryable_unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": True,
            "category": "provider",
            "summary": "The provider returned a temporary unavailable response.",
            "suggested_action": "Retry the review shortly.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[
            retryable_unavailable,
            structured_pr_review(reviewer="OpenAI Codex"),
        ]
    )
    config = make_config(tmp_path, reviewer="codex", agent_max_retries=1)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_reviews) == 2
    assert any(cmd[:1] == ["sleep"] for cmd, _cwd in runner.commands)


def test_pr_loop_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented the requested PR body change.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", approved_followups="fix-and-summarize", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "### Same-PR follow-ups\n"
        "- Require source issue reference in PR body.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    assert runner.comments[3] == (
        "**Review verdict:** Approved\n\n"
        "Looks good.\n\n"
        "### Prior unresolved item dispositions\n"
        "- [item-1] RESOLVED\n"
        "  - Original finding: Same-PR follow-up from OpenAI Codex, round 1: Require source issue reference in PR body.\n"
        "<!-- AGENT_STATE: approved -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )

def test_pr_loop_tracks_blocking_items_text_not_summary_when_they_differ(tmp_path):
    """The tracked unresolved item must use the blocking_items bullet text, not the
    summary prose, so summary is never itemized when blocking_items is populated."""
    runner = FakeRunner(
        claude_outputs=["Implemented fixes.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[
            "Needs one more regression test before merge."
            + blocking_issues("Add the mixed-history resume case to `tests/test_agent_loop.py`.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_coder_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]][0]
    assert "Add the mixed-history resume case to `tests/test_agent_loop.py`." in second_coder_prompt
    assert "Needs one more regression test before merge." in second_coder_prompt
    assert "Needs one more regression test before merge." not in second_coder_prompt.split("Codex unresolved blocking item", 1)[1]
    assert runner.comments[0] == (
        "**Review verdict:** Blocking\n\n"
        "Needs one more regression test before merge.\n\n"
        "### Blocking issues\n"
        "- Add the mixed-history resume case to `tests/test_agent_loop.py`.\n"
        "<!-- AGENT_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    second_review_prompt = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]][1]
    assert "[item-1]" in second_review_prompt
    assert "Add the mixed-history resume case to `tests/test_agent_loop.py`." in second_review_prompt

def test_pr_loop_same_pr_only_review_does_not_duplicate_summary_as_blocking_item(tmp_path):
    """Regression test for issue #501 / llm-dialectic PR #257: a blocking review whose
    summary prose restates the same concern as its lone same_pr_followups entry must
    produce exactly one tracked item (the same-PR follow-up), not a second,
    summary-derived blocking item."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="The PR is docs-only; the new audit doc contains stale path references.",
                same_pr_followups=[
                    "Update docs/SECURITY_FINDINGS.md references to docs/audit/SECURITY_FINDINGS.md."
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Stale references fixed.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Updated the stale path references.",
                addressed_items=["item-1"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        approved_followups="fix-and-summarize",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert (
        "Update docs/SECURITY_FINDINGS.md references to docs/audit/SECURITY_FINDINGS.md."
        in followup_prompt
    )
    assert "The PR is docs-only" in followup_prompt
    assert "The PR is docs-only" not in followup_prompt.split("Codex same-PR follow-up", 1)[1]
    # Routed through the lean same-PR-only prompt, confirming no separate blocking
    # item was created alongside the same-PR item.
    assert "Address the follow-up items below" in followup_prompt

def test_pr_loop_creates_one_tracked_item_per_blocking_items_entry(tmp_path):
    """Each blocking_items entry gets its own tracked item; the summary is never
    itemized when blocking_items is populated."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Two separate problems found.",
                blocking_items=[
                    "Fix the null pointer dereference in parser.py.",
                    "Add missing docstring to the public API.",
                ],
            ),
            structured_pr_review(
                state="approved",
                summary="Both issues fixed.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed both issues.",
                addressed_items=["item-1", "item-2"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "[item-1]" in followup_prompt
    assert "[item-2]" in followup_prompt
    assert "Fix the null pointer dereference in parser.py." in followup_prompt
    assert "Add missing docstring to the public API." in followup_prompt
    assert "Two separate problems found." in followup_prompt
    assert "Two separate problems found." not in followup_prompt.split("Codex unresolved blocking item", 1)[1]

def test_pr_loop_falls_back_to_summary_item_when_no_structured_fields_present(tmp_path):
    """When a blocking review has neither blocking_items nor same_pr_followups, the
    summary must still become the tracked item so a genuine blocking verdict is not
    silently dropped from the ledger."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="I have concerns about this approach but cannot pinpoint a specific line.",
            ),
            structured_pr_review(
                state="approved",
                summary="Satisfied after discussion.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Clarified the approach.",
                addressed_items=["item-1"],
                remaining_items=[],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert (
        "I have concerns about this approach but cannot pinpoint a specific line."
        in followup_prompt
    )

def test_pr_loop_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "### Same-PR follow-ups\n"
            "- Require source issue reference in PR body.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("claude", "codex"),
        approved_followups="fix-and-summarize",
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[0]

def test_pr_loop_same_pr_items_remain_blocking_until_explicitly_resolved(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "### Same-PR follow-ups\n"
            "- Rename the helper before merge.\n"
            "<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Codex still wants the rename."
            + prior_item_dispositions("[item-1] same-pr: The caller still uses the old helper name; rename that call site too.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Tried a partial fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=2)

    with pytest.raises(AgentLoopError, match="still reported blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

def test_pr_loop_resumes_with_only_missing_reviewer_for_current_head(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a regression test before merge.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR with the requested fix.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
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
            subject="abc123",
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
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    gemini_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert "[item-1]" in gemini_prompt
    assert "Add a regression test before merge." in gemini_prompt

def test_pr_loop_resume_raises_agent_loop_error_for_missing_reconstructed_prior_item(tmp_path):
    carried_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Actual active carried item.",
        status="blocking",
        source_status="blocking",
    )
    coder_comment = _attach_round_metadata(
        "Updated the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="abc123",
            prior_items=(carried_item,),
        ),
    )
    invalid_disposition = ReviewItemDisposition(
        item_id="item-1",
        reviewer="OpenAI Codex",
        disposition="resolved",
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
            subject="abc123",
            prior_items=(carried_item,),
            dispositions=(invalid_disposition,),
            state="approved",
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:00:00Z", "body": coder_comment},
                {"author": {"login": "bot"}, "createdAt": "2026-05-20T10:05:00Z", "body": codex_comment},
            ],
        },
    )
    config = make_config(tmp_path, reviewer=("codex",))

    with pytest.raises(
        AgentLoopError,
        match=r"Resumed pr round 2 reconstructed prior items item-2, but Codex dispositioned unknown item `item-1`",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-pr: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_pr_loop_rejects_contradictory_disposition_before_extra_coder_round(tmp_path, line):
    runner = FakeRunner(
        codex_outputs=[
            "Needs regression coverage.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good overall."
            + prior_item_dispositions(line)
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=["Added coverage.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
    )
    config = make_config(tmp_path, approved_followups="fix-and-summarize", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_pr_loop(runner, pr_number=77, config=config)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1


def test_pr_loop_dispute_resolved_when_reviewer_reconsiders(tmp_path):
    """Coder disputes a blocking item; reviewer sees evidence and approves."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Coder provided valid pricing evidence; approved.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Coder provided valid pricing evidence."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_pr_loop(runner, pr_number=55, config=config) == 0

    followup_comments = [c for c in runner.comments if "## Coder follow-up" in c]
    assert len(followup_comments) == 1
    followup_body = _strip_round_metadata(followup_comments[0])
    assert "### Disputed items" in followup_body
    assert "item-1" in followup_body
    assert "Official docs confirm $1.50/1M tokens is correct." in followup_body


def test_pr_loop_releases_adopted_suppression_when_reviewer_rejects_dispute(
    tmp_path, monkeypatch
):
    """Human-decision escalation releases an invocation-owned adoption label."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="blocking",
                summary="Pricing still incorrect despite coder evidence.",
                blocking_items=["Pricing is still incorrect."],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "I checked and the pricing is still wrong."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    monkeypatch.setattr(
        pr_loop,
        "activate_managed_ci",
        lambda *_args, **_kwargs: ManagedCiContract(
            adopted_existing_pr=True,
            invocation_applied_label=True,
        ),
    )
    posted_comment_calls = []
    original_run = runner.run

    def capture_posted_comment(args, *, cwd, **kwargs):
        command = [str(arg) for arg in args]
        if command[:4] == ["gh", "pr", "comment", "55"]:
            body_path = Path(command[command.index("--body-file") + 1])
            posted_comment_calls.append((command, body_path.read_text(encoding="utf-8")))
        return original_run(args, cwd=cwd, **kwargs)

    monkeypatch.setattr(runner, "run", capture_posted_comment)
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    with pytest.raises(
        HumanDecisionRequiredError,
        match="Reviewer did not resolve 1 disputed item",
    ) as excinfo:
        run_pr_loop(runner, pr_number=55, config=config)

    assert "Update/evidence: Codex: I checked and the pricing is still wrong." in str(excinfo.value)
    release_command = [
        "gh",
        "api",
        "--method",
        "DELETE",
        "repos/OWNER/REPO/issues/55/labels/agent-loop-managed",
    ]
    commands = [command for command, _cwd in runner.commands]
    release_index = commands.index(release_command)
    human_decision_comment_calls = [
        (command, body)
        for command, body in posted_comment_calls
        if "## Human decision required" in body
    ]
    assert len(human_decision_comment_calls) == 1
    comment_command, human_decision_comment = human_decision_comment_calls[0]
    assert comment_command[:4] == ["gh", "pr", "comment", "55"]
    assert "--body-file" in comment_command
    assert "Reviewer did not resolve 1 disputed item(s)" in human_decision_comment
    assert "Update/evidence: Codex: I checked and the pricing is still wrong." in human_decision_comment
    assert "-- Human Reviewer" in human_decision_comment
    comment_index = commands.index(comment_command)
    assert comment_index < release_index


def test_pr_loop_escalates_when_reviewer_downgrades_disputed_item_to_same_pr(tmp_path):
    """Coder disputes a blocking item; reviewer downgrades to same-pr instead of resolving → escalate."""
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Official docs confirm $1.50/1M tokens is correct."},
                summary="Disputing item-1: reviewer pricing claim is factually incorrect.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Pricing constant is wrong.",
                blocking_items=["The gemini-3.5-flash pricing constant is wrong ($0.30 not $1.50)."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="blocking",
                summary="Ok I'll accept the coder's pricing evidence but still want a same-pr fix.",
                same_pr_followups=["Please add a comment citing the pricing source."],
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "same-pr", "note": "Downgraded from blocking but still needs attention."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    with pytest.raises(
        AgentLoopError,
        match="Reviewer did not resolve 1 disputed item",
    ):
        run_pr_loop(runner, pr_number=55, config=config)


def test_pr_loop_dispute_note_is_visible_to_reviewer_in_next_round(tmp_path):
    """After coder disputes, the dispute evidence note appears in prior items for reviewer."""
    from coding_review_agent_loop.unresolved_items import CODER_DISPUTE_NOTE_PREFIX

    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(
                addressed_items=[],
                remaining_items=[],
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "Evidence: price is $1.50 not $0.30."},
                summary="Disputing item-1.",
            ),
        ],
        codex_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Price is wrong.",
                blocking_items=["Price is wrong."],
                prior_item_dispositions=[],
                reviewer="OpenAI Codex",
            ),
            structured_pr_review(
                state="approved",
                summary="Approved after reviewing coder evidence.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Accepted coder evidence."},
                ],
                reviewer="OpenAI Codex",
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)
    run_pr_loop(runner, pr_number=55, config=config)

    # The prior_items stored in the coder's PR comment should include dispute notes
    # runner.pr_payload["comments"] retains the raw body with AGENT_LOOP_META intact
    coder_followup_raw = next(
        (c["body"] for c in runner.pr_payload["comments"] if "## Coder follow-up" in c["body"]),
        None,
    )
    assert coder_followup_raw is not None, "Expected a coder follow-up comment in PR"
    meta_match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        coder_followup_raw,
    )
    assert meta_match is not None, "Expected AGENT_LOOP_META in coder followup comment"
    metadata = _decode_round_metadata(meta_match.group("payload"))
    assert metadata is not None
    disputed_item = next(
        (item for item in metadata.prior_items if item.item_id == "item-1"), None
    )
    assert disputed_item is not None
    assert any(CODER_DISPUTE_NOTE_PREFIX in note for note in disputed_item.notes)


# ---------------------------------------------------------------------------
# #862: resolved-history proof for incomplete-ledger disposition strips
# ---------------------------------------------------------------------------

from coding_review_agent_loop.protocol import ReviewItemDisposition as _Disp862
from coding_review_agent_loop.protocol import UnresolvedReviewItem as _Item862
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata as _Meta862,
    PostedRoundRecord as _Record862,
    _canonically_resolved_history_item_ids,
)


def _item862(item_id="item-1", reviewer="Codex", owners=(), status="blocking", **extra):
    return _Item862(
        item_id=item_id,
        reviewer=reviewer,
        source_round=1,
        text="worker cleanup gap",
        status=status,
        resolution_owners=owners,
        **extra,
    )


def _record862(
    index,
    *,
    subject,
    round_number,
    role="reviewer",
    agent="Codex",
    prior_items=(),
    dispositions=(),
    new_items=(),
    flow="pr",
):
    return _Record862(
        index=index,
        metadata=_Meta862(
            flow=flow,
            role=role,
            agent=agent,
            round_number=round_number,
            subject=subject,
            prior_items=tuple(prior_items),
            dispositions=tuple(dispositions),
            new_items=tuple(new_items),
        ),
        body="",
    )


def _raise_then_resolve_records(*, dispositions, prior=None):
    item = prior or _item862()
    return [
        _record862(0, subject="h1", round_number=1, new_items=[item]),
        _record862(
            1, subject="h2", round_number=2, prior_items=[item], dispositions=dispositions
        ),
    ]


@pytest.mark.parametrize("mode", ["aggregate", "owner-scoped"])
def test_resolved_history_includes_single_owner_raise_then_resolve(mode):
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode=mode, same_status="same-pr"
    ) == frozenset({"item-1"})


@pytest.mark.parametrize("mode", ["aggregate", "owner-scoped"])
@pytest.mark.parametrize(
    "order",
    [("blocking", "resolved"), ("resolved", "blocking")],
)
def test_resolved_history_excludes_same_round_conflict_in_both_orders(mode, order):
    item = _item862()
    records = [
        _record862(0, subject="h1", round_number=1, new_items=[item]),
        _record862(
            1, subject="h2", round_number=2, agent="Codex", prior_items=[item],
            dispositions=[_Disp862("item-1", "Codex", order[0], note="n")],
        ),
        _record862(
            2, subject="h2", round_number=2, agent="Gemini", prior_items=[item],
            dispositions=[_Disp862("item-1", "Gemini", order[1], note="n")],
        ),
    ]
    assert "item-1" not in _canonically_resolved_history_item_ids(
        records, reconciliation_mode=mode, same_status="same-pr"
    )


def test_resolved_history_excludes_owner_scoped_non_owner_only_resolution():
    records = _raise_then_resolve_records(
        prior=_item862(owners=("Codex", "Gemini")),
        dispositions=[_Disp862("item-1", "Antigravity", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_owner_scoped_partial_multi_owner_clearance():
    records = _raise_then_resolve_records(
        prior=_item862(owners=("Codex", "Gemini")),
        dispositions=[_Disp862("item-1", "Codex", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_item_reintroduced_after_resolution():
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    records.append(
        _record862(2, subject="h3", round_number=3, agent="Gemini", new_items=[_item862()])
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_item_recarried_after_resolution():
    records = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    records.append(
        _record862(2, subject="h3", round_number=3, prior_items=[_item862()])
    )
    assert _canonically_resolved_history_item_ids(
        records, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()


def test_resolved_history_excludes_future_machine_other_flow_and_carried_items():
    future = _raise_then_resolve_records(
        prior=_item862("item-2"),
        dispositions=[_Disp862("item-2", "Codex", "future")],
    )
    assert _canonically_resolved_history_item_ids(
        future, reconciliation_mode="aggregate", same_status="same-pr"
    ) == frozenset()

    machine = _item862(
        "item-3",
        reviewer="GitHub managed exact-head CI",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="repair_required",
        failed_head_sha="h1",
    )
    machine_records = _raise_then_resolve_records(
        prior=machine,
        dispositions=[_Disp862("item-3", "Codex", "resolved")],
    )
    assert _canonically_resolved_history_item_ids(
        machine_records, reconciliation_mode="owner-scoped", same_status="same-pr"
    ) == frozenset()

    resolved = _raise_then_resolve_records(
        dispositions=[_Disp862("item-1", "Codex", "resolved")]
    )
    # Other-flow comments never decode into this flow's records.
    plan_records = architecture_contract._extract_round_metadata_records(
        [
            SimpleNamespace(
                body=discuss_loop._attach_round_metadata(
                    "plan review",
                    dataclasses.replace(record.metadata, flow="plan"),
                )
            )
            for record in resolved
        ],
        flow="pr",
    )
    assert plan_records == ()
    assert _canonically_resolved_history_item_ids(
        resolved,
        reconciliation_mode="aggregate",
        same_status="same-pr",
        current_carried_ids=("item-1",),
    ) == frozenset()


def test_staged_secondary_audit_strips_resolved_primary_disposition_under_incomplete_ledger(
    tmp_path, monkeypatch
):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    ledger_flags = []
    real_ledger_check = review_rounds._round_ledger_may_be_incomplete

    def ledger_spy(**kwargs):
        result = real_ledger_check(**kwargs)
        ledger_flags.append(result)
        return result

    patch_attr_in(monkeypatch, (
        review_rounds, pr_loop, plan_first_loop,
    ), "_round_ledger_may_be_incomplete", ledger_spy)
    logged = []
    real_log = agent_failure.log
    patch_attr_in(monkeypatch, (
        agent_failure, integration_close, architecture_contract, validated_agent,
        response_validation, panel_evidence, review_rounds, discuss_loop, execution_policy,
        child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "log",
        lambda config, message, *a, **k: (logged.append(message), real_log(config, message, *a, **k))[1],
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        # The secondary audit repeats the primary's already-cleared item.
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    repair_mock.assert_not_called()
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "primary", "secondary-audit",
    ]
    # The secondary-audit round is exactly the incomplete-ledger state.
    assert ledger_flags[-1] is True
    assert any(
        "removed canonically resolved historical prior-item disposition ID(s) item-1 "
        "despite incomplete ledger" in message
        for message in logged
    )
    assert not any(
        "deterministically removed unknown prior-item" in message for message in logged
    )
    # The stripped response keeps its approval and posts no disposition.
    gemini_record = next(
        record
        for record in architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload.get("comments", [])],
            flow="pr",
        )
        if record.metadata.agent == "Gemini" and record.metadata.role == "reviewer"
    )
    assert gemini_record.metadata.state == "approved"
    assert gemini_record.metadata.dispositions == ()
    assert gemini_record.metadata.new_items == ()
