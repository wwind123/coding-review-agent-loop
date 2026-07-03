"""Integration tests for plan-first split/deferred-stage materialization and
the selected-stage implementation handoff (#476)."""
import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import validate_pr_body_does_not_close_issue
from coding_review_agent_loop.split_materialization import (
    MaterializedSplitChild,
    SplitMaterializationMetadata,
    format_split_materialization_summary,
    split_stage_proposal_from_text,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    structured_plan_review,
    structured_plan_state,
    structured_pr_review,
)


def _existing_split_children_comment() -> dict:
    metadata = SplitMaterializationMetadata(
        parent_issue=56,
        subject="prior-discuss-subject",
        children=(
            MaterializedSplitChild(
                title="Auth flow",
                key=split_stage_proposal_from_text("Auth flow").key,
                url="https://github.com/OWNER/REPO/issues/101",
                number=101,
                origin="created",
            ),
            MaterializedSplitChild(
                title="Billing flow",
                key=split_stage_proposal_from_text("Billing flow").key,
                url="https://github.com/OWNER/REPO/issues/102",
                number=102,
                origin="created",
            ),
        ),
    )
    return {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_split_materialization_summary(parent_issue=56, metadata=metadata),
    }


def test_plan_first_no_prior_split_materializes_deferred_stages_before_implementation(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                state="blocking",
                summary="Implement the core parser change.",
                deferred_stages=[{"title": "Auth flow", "summary": "Split follow-up out."}],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "[#56 stage] Auth flow"
    materialize_index = command_index(runner.commands, ["gh", "issue", "create"])
    pr_create_index = command_index(runner.commands, ["claude", "--print"], start=materialize_index)
    assert materialize_index < pr_create_index
    assert any("<!-- AGENT_DISCUSS_SPLIT:" in comment for comment in runner.comments)
    # The parent's own plan covers everything except the deferred stage, so
    # its own PR still closes the parent issue normally (default PR body
    # scripted by FakeRunner is "Fixes #56").
    assert "Fixes #56" in runner.pr_payload["body"]


def test_plan_first_no_prior_split_warns_when_materialization_disabled(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                state="blocking",
                summary="Implement the core parser change.",
                deferred_stages=[{"title": "Auth flow", "summary": "Split follow-up out."}],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
    )
    config = make_config(tmp_path)  # materialize_split_issues defaults False

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert runner.issues == []
    assert any("NOT filed as issues" in comment and "Auth flow" in comment for comment in runner.comments)


def test_plan_first_selected_stage_handoff_by_unique_title_match(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(state="blocking", summary="Auth flow"),
            "Implemented the auth-flow stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[_existing_split_children_comment()],
        pr_payload={"body": "Closes #101\n\nRefs #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    # No new child issues: both stages were already materialized before this run.
    assert runner.issues == []
    assert any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)
    stage_handoff = next(c for c in runner.comments if "<!-- AGENT_SPLIT_STAGE_HANDOFF:" in c)
    assert "handed off to split stage" in stage_handoff
    # The implementation prompt sent to the coder must target the resolved
    # child (#101) and instruct staged Closes/Refs wording, not Fixes on the parent.
    implement_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    implementation_prompt = implement_calls[1][-1]
    assert "issue #101" in implementation_prompt
    assert "Closes #101" in implementation_prompt
    assert "Refs #56" in implementation_prompt
    assert "Do NOT use a closing keyword" in implementation_prompt


def test_plan_first_selected_stage_handoff_via_split_stage_flag(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            # Plan text/summary does not uniquely match either child title, so
            # resolution must come from --split-stage.
            structured_plan_state(state="blocking", summary="Some unrelated plan summary"),
            "Implemented the billing stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[_existing_split_children_comment()],
        pr_payload={"body": "Closes #102\n\nRefs #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True, split_stage=102)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    stage_handoff = next(c for c in runner.comments if "<!-- AGENT_SPLIT_STAGE_HANDOFF:" in c)
    assert "issues/102" in stage_handoff


def test_plan_first_selected_stage_ambiguous_without_flag_raises(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(state="blocking", summary="Some unrelated plan summary"),
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
        ],
        issue_comments=[_existing_split_children_comment()],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    with pytest.raises(AgentLoopError, match="--split-stage"):
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )


def test_validate_pr_body_does_not_close_issue_rejects_parent_fixes_for_staged_pr(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Fixes #56"})
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match="closing keyword"):
        validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)
