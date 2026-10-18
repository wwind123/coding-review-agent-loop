"""Execution-policy tests: execution-policy resolution and collection extraction (#1543)."""

import json
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.execution_policy import _infer_staged_parent_issue
from coding_review_agent_loop.protocol import validate_structured_plan_state
import coding_review_agent_loop.execution_policy as execution_policy
from issue_loop_test_helpers import _M948_FOOTER, _m948_stages
from agent_loop_helpers import (
    make_config,
    structured_plan_state,
    structured_v1_plan_state,
)
from coding_review_agent_loop.comment_rendering import render_public_agent_comment  # noqa: E402,F811
from coding_review_agent_loop.protocol import (
    validate_structured_plan_state as _m948_validate_state,
)


def test_auto_execution_resolves_one_shot_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation

    resolved = execution_policy._resolve_execution_policy(
        config,
        recommendation=recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-one-shot"
    assert resolved.strategy == "one-shot"


def test_auto_execution_resolves_staged_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation
    staged_recommendation = replace(recommendation, strategy="staged")

    resolved = execution_policy._resolve_execution_policy(
        config,
        recommendation=staged_recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-by-phase"
    assert resolved.strategy == "staged"


def test_auto_execution_rejects_legacy_undecided_plan(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")

    with pytest.raises(AgentLoopError, match="legacy-undecided"):
        execution_policy._resolve_execution_policy(
            config,
            recommendation=None,
        )


def test_infer_staged_parent_requires_generated_marker_or_body_header():
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Ordinary issue text.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Quoted AGENT_SPLIT_CHILD: parent=12 in review prose.",
            ),
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #13: quoted comment text.",
            ),
        ),
    )

    assert _infer_staged_parent_issue(issue_context) is None

    marked_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="<!-- AGENT_SPLIT_CHILD: parent=12 key=" + "a" * 64 + " -->",
        url=issue_context.url,
        comments=(),
    )
    assert _infer_staged_parent_issue(marked_context) == 12

    decomposed_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child phase issue for parent #13: https://github.com/OWNER/REPO/issues/13\n\nDetails.",
        url=issue_context.url,
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #14: quoted comment text.",
            ),
        ),
    )
    assert _infer_staged_parent_issue(decomposed_context) == 13


def test_m948_resumed_collections_equal_the_full_form_publication():
    payload = json.loads(structured_plan_state(summary="Collections plan.").split("\n", 1)[0])
    payload["deferred_stages"] = _m948_stages("Deferred", 5)
    payload["plan_actions"] = _m948_stages("Action", 5)
    payload["additional_closing_issue_ids"] = [101, 102]
    parsed = _m948_validate_state(json.dumps(payload) + _M948_FOOTER)
    canonical_plan = plan_first_loop_module.render_canonical_plan_state(parsed)

    # Stage and closing collections are read from the canonical plan only; the
    # digest carries none of their records.
    assert execution_policy._extract_current_deferred_stages(canonical_plan) == parsed.deferred_stages
    assert execution_policy._extract_current_expected_closing_issue_ids(canonical_plan) == (101, 102)
    digest = render_public_agent_comment(
        kind="plan_state", parsed=parsed, agent="claude", compact=True
    )
    assert execution_policy._extract_current_deferred_stages(digest) == ()
    assert execution_policy._extract_current_expected_closing_issue_ids(digest) is None
