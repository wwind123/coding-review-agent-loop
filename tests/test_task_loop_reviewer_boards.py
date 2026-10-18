"""Task-loop tests: named reviewer seats on task-created PRs (#1543)."""

from issue_loop_test_helpers import (
    FakeRunner,
    _assert_named_issue_created_pr,
    _named_1352_seats,
    _queue_named_reviews,
    _serve_named_pr_rest,
)
from agent_loop_helpers import make_config, structured_pr_review


def test_named_task_reaches_pr_review_with_anchored_board(tmp_path):
    """Row task-named: the freeform initial coder record of task mode."""
    from coding_review_agent_loop.board_change import identity_from_config
    from coding_review_agent_loop.cli import run_task_loop

    seats = _named_1352_seats(tmp_path)
    outputs = {
        "claude_outputs": ["Implemented.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
    }
    _queue_named_reviews(outputs, seats, structured_pr_review)
    # A task-origin PR links no issue, so only the PR side is authenticated.
    runner = _serve_named_pr_rest(FakeRunner(
        **outputs, authenticated_actor=("agent-loop", 4242), pr_payload={"body": "Task PR."},
    ))
    config = make_config(tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False)

    assert run_task_loop(runner, task_text="Add a /healthz endpoint.", config=config) == 0

    _assert_named_issue_created_pr(runner, config, identity_from_config(config))
    assert not any(
        "issues/56/comments?per_page" in cmd[2]
        for cmd, _cwd in runner.commands if cmd[:2] == ["gh", "api"] and len(cmd) > 2
    )
