"""Issue-PR handoff tests: canonical handoff recovery through the issue loop (#1543)."""

from coding_review_agent_loop.github import get_issue_context
from coding_review_agent_loop.issue_pr_handoff import (
    format_issue_pr_handoff_comment,
    resolve_canonical_pr_for_issue,
)
from issue_loop_test_helpers import FakeRunner
from agent_loop_helpers import make_config


def test_removed_canonical_handoff_does_not_recreate_from_incidental_pr(tmp_path):
    canonical_comment = {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_issue_pr_handoff_comment(
            issue_number=56,
            pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            flow="issue-implementation",
            plan_hash=None,
        ),
    }
    runner = FakeRunner(
        issue_comments=[canonical_comment],
        open_prs_payload=[
            {"number": 492, "body": "#56's loop is running; this is only a sequencing note."}
        ],
    )
    config = make_config(tmp_path)

    first_context = get_issue_context(runner, config=config, issue_number=56)
    first = resolve_canonical_pr_for_issue(
        runner, config=config, issue_number=56, issue_context=first_context
    )
    assert first is not None
    assert first.source == "canonical"
    assert first.pr_number == 77

    runner.issue_comments.clear()
    second_context = get_issue_context(runner, config=config, issue_number=56)
    assert (
        resolve_canonical_pr_for_issue(
            runner, config=config, issue_number=56, issue_context=second_context
        )
        is None
    )
