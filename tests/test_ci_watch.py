"""Focused tests for the opt-in full-board post-approval CI watcher (#587)."""

from unittest.mock import patch

import pytest

from coding_review_agent_loop.ci_health import PullRequestCheck, PullRequestChecks
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import PullRequestMetadata, PullRequestMergeability, watch_pr_checks
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config


class _Runner(Runner):
    def __init__(self):
        super().__init__(dry_run=False)
        self.commands = []

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        self.commands.append(list(args))
        return CommandResult(list(args), cwd, "", "", 0)


def _metadata():
    return PullRequestMetadata(7, "OWNER/REPO", "title", "head", "main", "sha", "url")


def _checks(
    state,
    *,
    failing=(),
    pending=(),
    missing_required=(),
    query="ok",
    protection="configured",
):
    return PullRequestChecks(
        state=state, required_checks=(), passing=(), pending=pending, failing=failing,
        missing_required=tuple(missing_required), branch_protection_status=protection, branch_protection_note=None,
        check_query_status=query, check_query_errors=(), infrastructure_stalls=(),
    )


def _mergeability():
    return PullRequestMergeability("mergeable", "MERGEABLE", "CLEAN", "sha", "main")


def test_watch_pending_to_failure_reports_failed_records_and_sleeps(tmp_path):
    runner = _Runner()
    failed = PullRequestCheck(
        name="test",
        kind="check_run",
        status="failure",
        url="https://checks/1",
    )
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", side_effect=[_checks("pending"), _checks("failing", failing=(failed,))]):
        outcome = watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True, ci_timeout_seconds=60, ci_poll_interval_seconds=30), 7, metadata=_metadata())
    assert outcome.status == "failed"
    assert outcome.failed_checks == (failed,)
    assert [command[0] for command in runner.commands] == ["sleep"]


@pytest.mark.parametrize(
    ("missing_required", "query"),
    [(("build",), "ok"), ((), "partial")],
)
def test_watch_reports_observed_failure_despite_incomplete_board_metadata(
    tmp_path, missing_required, query
):
    runner = _Runner()
    failed = PullRequestCheck(
        name="test", kind="check_run", status="failure", url="https://checks/1"
    )
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             return_value=_checks(
                 "failing",
                 failing=(failed,),
                 missing_required=missing_required,
                 query=query,
             ),
         ):
        outcome = watch_pr_checks(
            runner,
            make_config(tmp_path, watch_pending_ci=True),
            7,
            metadata=_metadata(),
        )
    assert outcome.status == "failed"
    assert outcome.failed_checks == (failed,)
    assert runner.commands == []


def test_watch_success_and_head_change_are_terminal(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", return_value=_checks("passing")):
        assert watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True), 7, metadata=_metadata()).status == "passed"
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="new-sha"):
        assert watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True), 7, metadata=_metadata()).status == "head_changed"


def test_watch_empty_rollup_stops_as_not_started_never_as_success(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", return_value=_checks("no_checks")):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path,
                watch_pending_ci=True,
                ci_timeout_seconds=60,
                ci_poll_interval_seconds=30,
                ci_startup_timeout_seconds=30,
            ),
            7,
            metadata=_metadata(),
        )
    assert outcome.status == "not_started"
    assert runner.commands == []


def test_watch_timeout_retries_transient_snapshots_and_dry_run_does_not_poll(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", return_value=_checks("unavailable", query="unavailable", protection="unavailable")):
        outcome = watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True, ci_timeout_seconds=60, ci_poll_interval_seconds=30), 7, metadata=_metadata())
    assert outcome.status == "timeout"
    assert len(runner.commands) == 1
    dry = make_config(tmp_path, watch_pending_ci=True, dry_run=True)
    assert watch_pr_checks(_Runner(), dry, 7, metadata=_metadata()).status == "dry_run"


def test_watch_accepts_forbidden_branch_protection_and_retries_flaky_head_probe(tmp_path):
    runner = _Runner()
    with patch(
        "coding_review_agent_loop.github.get_pr_head_sha",
        side_effect=[AgentLoopError("temporary gh failure"), "sha"],
    ), patch(
        "coding_review_agent_loop.github.get_pr_mergeability",
        return_value=_mergeability(),
    ), patch(
        "coding_review_agent_loop.github.get_pr_checks",
        side_effect=[_checks("pending"), _checks("passing", protection="forbidden")],
    ):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path,
                watch_pending_ci=True,
                ci_timeout_seconds=60,
                ci_poll_interval_seconds=30,
            ),
            7,
            metadata=_metadata(),
        )
    assert outcome.status == "passed"
    assert [command[0] for command in runner.commands] == ["sleep"]
