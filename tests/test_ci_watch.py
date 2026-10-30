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


def _mergeability(merge_state="CLEAN", *, state="mergeable", head_sha="sha"):
    mergeable_raw = "MERGEABLE" if state == "mergeable" else "UNKNOWN"
    return PullRequestMergeability(state, mergeable_raw, merge_state, head_sha, "main")


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
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="fd2d39cf710f84ca537091424fc212fa5211885b"):
        assert watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True), 7, metadata=_metadata()).status == "head_changed"


def test_watch_forbidden_protection_with_clean_merge_state_passes(tmp_path):
    # #1055: a classic-protection 403 is resolved by GitHub's own merge-state
    # computation for the same head, which accounts for unreadable contexts.
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             return_value=_checks("passing", protection="forbidden"),
         ):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path,
                watch_pending_ci=True,
                ci_timeout_seconds=1,
                ci_poll_interval_seconds=1,
            ),
            7,
            metadata=_metadata(),
            attempts=1,
        )

    assert outcome.status == "passed"
    assert outcome.mergeability == _mergeability()
    assert runner.commands == []


@pytest.mark.parametrize(
    "mergeability",
    [
        _mergeability("BLOCKED"),
        _mergeability("UNSTABLE"),
        _mergeability("UNKNOWN", state="unknown"),
        _mergeability(None, state="unknown"),
        _mergeability("CLEAN", head_sha="other-sha"),
    ],
)
def test_watch_forbidden_protection_without_clean_merge_state_stops_fast(
    tmp_path, mergeability
):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=mergeability), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             return_value=_checks("passing", protection="forbidden"),
         ):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path,
                watch_pending_ci=True,
                ci_timeout_seconds=1200,
                ci_poll_interval_seconds=30,
                ci_startup_timeout_seconds=60,
            ),
            7,
            metadata=_metadata(),
        )

    # The bounded startup window (2 polls), not the 1200s watch timeout.
    assert outcome.status == "protection_unreadable"
    assert outcome.attempts_used == 2
    assert outcome.mergeability == mergeability
    assert [command[0] for command in runner.commands] == ["sleep"]


def test_watch_forbidden_protection_pending_board_keeps_waiting(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability("BLOCKED")), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             return_value=_checks("pending", protection="forbidden"),
         ):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path,
                watch_pending_ci=True,
                ci_timeout_seconds=90,
                ci_poll_interval_seconds=30,
                ci_startup_timeout_seconds=30,
            ),
            7,
            metadata=_metadata(),
        )

    assert outcome.status == "timeout"


@pytest.mark.parametrize("protection", ["configured", "not_found"])
def test_watch_readable_protection_ignores_merge_state(tmp_path, protection):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability("BLOCKED")), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             return_value=_checks("passing", protection=protection),
         ):
        outcome = watch_pr_checks(
            runner, make_config(tmp_path, watch_pending_ci=True), 7, metadata=_metadata(),
        )

    assert outcome.status == "passed"


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


def test_watch_rejects_forbidden_branch_protection_after_flaky_head_probe(tmp_path):
    runner = _Runner()
    with patch(
        "coding_review_agent_loop.github.get_pr_head_sha",
        side_effect=[AgentLoopError("temporary gh failure"), "sha"],
    ), patch(
        "coding_review_agent_loop.github.get_pr_mergeability",
        return_value=_mergeability("BLOCKED"),
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
    # The green BLOCKED board arrives on the final poll: the budget expires
    # on the unreadable protection, never a pass (#1055 late-green deadline).
    assert outcome.status == "protection_unreadable"
    assert outcome.mergeability == _mergeability("BLOCKED")
    assert [command[0] for command in runner.commands] == ["sleep"]


@pytest.mark.parametrize("deadline_kind", ["attempts", "clock"])
def test_watch_late_green_unreadable_board_at_deadline_names_protection(tmp_path, deadline_kind):
    runner = _Runner()
    config = make_config(
        tmp_path,
        watch_pending_ci=True,
        ci_timeout_seconds=90,
        ci_poll_interval_seconds=30,
        ci_startup_timeout_seconds=300,
    )
    kwargs = {"attempts": 3} if deadline_kind == "attempts" else {"deadline": 0.0, "attempts": 10}
    boards = (
        [_checks("pending", protection="forbidden")] * 2
        + [_checks("passing", protection="forbidden")]
        if deadline_kind == "attempts"
        else [_checks("passing", protection="forbidden")]
    )
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability",
               return_value=_mergeability("UNKNOWN", state="unknown")), \
         patch("coding_review_agent_loop.github.get_pr_checks", side_effect=boards):
        outcome = watch_pr_checks(runner, config, 7, metadata=_metadata(), **kwargs)

    assert outcome.status == "protection_unreadable"
    assert outcome.mergeability == _mergeability("UNKNOWN", state="unknown")
    assert outcome.attempts_used == len(boards)


def test_watch_deadline_after_green_turned_pending_is_a_plain_timeout(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability("BLOCKED")), \
         patch(
             "coding_review_agent_loop.github.get_pr_checks",
             side_effect=[
                 _checks("passing", protection="forbidden"),
                 _checks("pending", protection="forbidden"),
             ],
         ):
        outcome = watch_pr_checks(
            runner,
            make_config(
                tmp_path, watch_pending_ci=True, ci_timeout_seconds=60,
                ci_poll_interval_seconds=30, ci_startup_timeout_seconds=300,
            ),
            7,
            metadata=_metadata(),
        )
    assert outcome.status == "timeout"
    assert outcome.mergeability is None


# --- Post-approval plumbing classification, rerun-once and stuck pending (#1477) ---

import json
import re

import coding_review_agent_loop.github as github_module
import coding_review_agent_loop.pr_loop as pr_loop_module
from coding_review_agent_loop.github import (
    PlumbingRerunState,
    rerun_ordinary_failed_jobs,
)
from coding_review_agent_loop.managed_ci import plumbing_check_names
from coding_review_agent_loop.pr_loop import run_pr_loop

from agent_loop_helpers import FakeRunner, structured_coder_followup, structured_pr_review

PLUMBING = plumbing_check_names()
RUN_URL = "https://github.com/OWNER/REPO/actions/runs/4242"
HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"


def _publisher(status="failure", *, check_id=11):
    return PullRequestCheck(
        name="Publish final exact-head status", kind="check_run", status=status,
        url=f"{RUN_URL}/job/{check_id}", check_id=check_id, check_suite_id=900,
    )


def _tests(status="success", *, check_id=10):
    return PullRequestCheck(
        name="tests", kind="check_run", status=status,
        url=f"{RUN_URL}/job/{check_id}", check_id=check_id, check_suite_id=900,
    )


def _final_status(state, *, created_at="2026-10-14T19:00:00Z", url="https://github.com/OWNER/REPO/actions/runs/4343"):
    return PullRequestCheck(
        name="final-ci/exact-head", kind="status_context", status=state, url=url,
        created_at=created_at,
    )


def _board(*checks, required=("tests",), complete=True):
    passing = tuple(c for c in checks if c.status in {"success", "neutral", "skipped"})
    failing = tuple(c for c in checks if c.status in {"failure", "error", "cancelled", "timed_out"})
    pending = tuple(c for c in checks if c not in passing and c not in failing)
    state = "failing" if failing else "pending" if pending else "passing" if passing else "no_checks"
    return PullRequestChecks(
        state=state, required_checks=tuple(required), passing=passing, pending=pending,
        failing=failing, missing_required=(), branch_protection_status="configured",
        check_query_status="ok", listing_complete=complete,
    )


def _ordinary_run(run_id=4242, *, attempt=1, head=HEAD, event="pull_request", status="completed",
                  conclusion="failure"):
    return {
        "id": run_id, "head_sha": head, "event": event, "status": status,
        "conclusion": conclusion, "run_attempt": attempt,
        "html_url": f"https://github.com/OWNER/REPO/actions/runs/{run_id}",
    }


class _LoopRunner(FakeRunner):
    """FakeRunner that serves Actions run reads and records ``gh run rerun``."""

    def __init__(self, *, runs=None, suites=None, rerun_returncode=0, jobs=None, **kwargs):
        super().__init__(**kwargs)
        self.runs = {str(key): value for key, value in (runs or {}).items()}
        self.suites = dict(suites or {})
        # Per-run job listings; by default a failed run failed only its publisher.
        self.jobs = {str(key): value for key, value in (jobs or {}).items()}
        self.rerun_returncode = rerun_returncode
        self.reruns = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(arg) for arg in args]
        if cmd[:3] == ["gh", "run", "rerun"]:
            cmd, cwd_path = self._record_command(args, cwd)
            self.reruns.append(cmd)
            run = self.runs.get(cmd[3])
            if run is not None and self.rerun_returncode == 0:
                run["run_attempt"] = run["run_attempt"] + 1
                run["status"] = "in_progress"
            return CommandResult(cmd, cwd_path, "", "denied" if self.rerun_returncode else "", self.rerun_returncode)
        endpoint = next((part for part in cmd if part.startswith("repos/")), "")
        jobs = re.search(r"/actions/runs/(\d+)/jobs\?", endpoint)
        if cmd[:2] == ["gh", "api"] and jobs:
            cmd, cwd_path = self._record_command(args, cwd)
            run = self.runs.get(jobs.group(1))
            listing = self.jobs.get(jobs.group(1))
            if listing is None and run is not None:
                listing = [("tests", "success")] + (
                    [("Publish final exact-head status", "failure")]
                    if run.get("conclusion") == "failure" else []
                )
            if listing is None:
                return CommandResult(cmd, cwd_path, "", "HTTP 404", 1)
            payload = {"total_count": len(listing), "jobs": [
                {"name": name, "conclusion": conclusion} for name, conclusion in listing
            ]}
            return CommandResult(cmd, cwd_path, json.dumps(payload), "", 0)
        if cmd[:2] == ["gh", "api"] and len(cmd) > 2:
            suite = re.search(r"/actions/runs\?check_suite_id=(\d+)$", cmd[2])
            single = re.search(r"/actions/runs/(\d+)$", cmd[2])
            if suite or single:
                cmd, cwd_path = self._record_command(args, cwd)
                if suite:
                    run_id = self.suites.get(int(suite.group(1)))
                    payload = {"workflow_runs": [self.runs[str(run_id)]] if run_id else []}
                    return CommandResult(cmd, cwd_path, json.dumps(payload), "", 0)
                run = self.runs.get(single.group(1))
                if run is None:
                    return CommandResult(cmd, cwd_path, "", "HTTP 404", 1)
                return CommandResult(cmd, cwd_path, json.dumps(run), "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def claude_calls(self):
        return [cmd for cmd, _cwd in self.commands if cmd[:1] == ["claude"]]

    def merges(self):
        return [cmd for cmd, _cwd in self.commands if cmd[:3] == ["gh", "pr", "merge"]]


def _install_boards(monkeypatch, *, snapshot, watcher):
    boards = list(watcher)

    def next_board(*_args, **_kwargs):
        return boards.pop(0) if len(boards) > 1 else boards[0]

    monkeypatch.setattr(pr_loop_module, "get_pr_checks", lambda *_a, **_k: snapshot)
    monkeypatch.setattr(github_module, "get_pr_checks", next_board)
    return boards


def _loop_config(tmp_path, **overrides):
    values = {
        "watch_pending_ci": True, "max_rounds": 1,
        "ci_timeout_seconds": 600, "ci_poll_interval_seconds": 30,
    }
    values.update(overrides)
    return make_config(tmp_path, **values)


def _approved_runner(**kwargs):
    kwargs.setdefault("codex_outputs", [structured_pr_review(state="approved", summary="Approved.")])
    kwargs.setdefault("runs", {4242: _ordinary_run()})
    kwargs.setdefault("suites", {900: 4242})
    return _LoopRunner(**kwargs)


def _comments_with(runner, text):
    return [comment for comment in runner.comments if text in comment]


# -- Watcher-level unit tests --------------------------------------------------


def _watch(tmp_path, boards, *, runner=None, rerun_state=None, **config):
    runner = runner or _Runner()
    values = {"watch_pending_ci": True, "ci_timeout_seconds": 300, "ci_poll_interval_seconds": 30}
    values.update(config)
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", side_effect=list(boards)):
        return watch_pr_checks(
            runner, make_config(tmp_path, **values), 7, metadata=_metadata(),
            plumbing_names=PLUMBING, rerun_state=rerun_state,
        ), runner


def test_watch_plumbing_only_failure_returns_plumbing_failure(tmp_path):
    outcome, runner = _watch(tmp_path, [_board(_tests(), _publisher())])
    assert outcome.status == "plumbing_failure"
    assert outcome.classification == "plumbing-only"
    assert outcome.failed_checks == (_publisher(),)
    assert outcome.run_urls == (RUN_URL,)
    assert runner.commands == []


def test_watch_without_plumbing_names_keeps_legacy_failed_outcome(tmp_path):
    runner = _Runner()
    with patch("coding_review_agent_loop.github.get_pr_head_sha", return_value="sha"), \
         patch("coding_review_agent_loop.github.get_pr_mergeability", return_value=_mergeability()), \
         patch("coding_review_agent_loop.github.get_pr_checks", return_value=_board(_tests(), _publisher())):
        outcome = watch_pr_checks(runner, make_config(tmp_path, watch_pending_ci=True), 7, metadata=_metadata())
    assert outcome.status == "failed"


@pytest.mark.parametrize(
    ("settled", "expected"),
    [("success", "plumbing_failure"), ("failure", "failed")],
)
def test_watch_holds_plumbing_failure_until_pending_tests_settle(tmp_path, settled, expected):
    held = _board(_tests("in_progress"), _publisher())
    outcome, runner = _watch(tmp_path, [held, held, _board(_tests(settled), _publisher())])
    assert outcome.status == expected
    assert outcome.attempts_used == 3
    assert [command[0] for command in runner.commands] == ["sleep", "sleep"]
    if expected == "failed":
        assert outcome.classification == "test-failure"


def test_watch_hold_expiry_takes_the_timeout_path(tmp_path):
    held = _board(_tests("in_progress"), _publisher())
    outcome, _runner = _watch(tmp_path, [held] * 3, ci_timeout_seconds=90)
    assert outcome.status == "timeout"
    assert outcome.hold_reason == "plumbing-awaiting-tests"
    assert outcome.run_urls == (RUN_URL,)


def test_watch_polls_through_stale_prior_attempt_evidence_after_a_rerun(tmp_path):
    state = PlumbingRerunState(
        head_sha="sha", run_id=4242, run_url=RUN_URL, prior_attempt=1,
        stale_keys=frozenset({("check", 11)}),
    )
    boards = [
        _board(_tests(), _publisher()),            # old failure, same check id
        _board(_tests(), _publisher()),
        _board(_tests(), _publisher("in_progress", check_id=12)),
        _board(_tests(), _publisher("success", check_id=12)),
    ]
    outcome, runner = _watch(tmp_path, boards, rerun_state=state)
    assert outcome.status == "passed"
    assert outcome.attempts_used == 4
    assert [command[0] for command in runner.commands] == ["sleep"] * 3


def test_watch_fresh_test_failure_after_rerun_routes_to_coder_immediately(tmp_path):
    state = PlumbingRerunState(
        head_sha="sha", run_id=4242, run_url=RUN_URL, prior_attempt=1,
        stale_keys=frozenset({("check", 11)}),
    )
    outcome, runner = _watch(
        tmp_path, [_board(_tests("failure", check_id=13), _publisher())], rerun_state=state,
    )
    assert outcome.status == "failed"
    assert outcome.classification == "test-failure"
    assert runner.commands == []


def test_watch_stale_evidence_until_deadline_is_a_timeout_not_a_failure(tmp_path):
    state = PlumbingRerunState(
        head_sha="sha", run_id=4242, run_url=RUN_URL, prior_attempt=1,
        stale_keys=frozenset({("check", 11)}),
    )
    outcome, _runner = _watch(
        tmp_path, [_board(_tests(), _publisher())] * 3, rerun_state=state, ci_timeout_seconds=90,
    )
    assert outcome.status == "timeout"
    assert outcome.hold_reason == "stale-after-rerun"


def test_watch_dry_run_issues_no_rerun_subprocess(tmp_path):
    runner = _Runner()
    config = make_config(tmp_path, watch_pending_ci=True, dry_run=True)
    assert watch_pr_checks(runner, config, 7, metadata=_metadata(), plumbing_names=PLUMBING).status == "dry_run"
    result = rerun_ordinary_failed_jobs(runner, config, (_publisher(),), "sha")
    assert result.status == "dry_run"
    assert runner.commands == []


# -- Ordinary rerun resolver ---------------------------------------------------


@pytest.mark.parametrize(
    ("run", "suites", "observations", "status"),
    [
        (_ordinary_run(), {900: 4242}, (_publisher(),), "rerun_started"),
        (_ordinary_run(4343), {}, (_final_status("failure"),), "rerun_started"),
        (_ordinary_run(), {}, (_final_status("failure", url="https://ci.example/x"),), "unresolvable"),
        (_ordinary_run(), {900: 4242}, (_publisher(), _final_status("failure")), "unresolvable"),
        (_ordinary_run(head="other"), {900: 4242}, (_publisher(),), "head_mismatch"),
        (_ordinary_run(event="workflow_dispatch"), {900: 4242}, (_publisher(),), "head_mismatch"),
        (_ordinary_run(status="in_progress"), {900: 4242}, (_publisher(),), "unresolvable"),
    ],
)
def test_ordinary_rerun_resolves_one_verified_run_before_the_write(tmp_path, run, suites, observations, status):
    runner = _LoopRunner(runs={run["id"]: run}, suites=suites)
    result = rerun_ordinary_failed_jobs(runner, make_config(tmp_path), observations, HEAD, PLUMBING)
    assert result.status == status
    expected = [["gh", "run", "rerun", str(run["id"]), "--failed", "--repo", "OWNER/REPO"]]
    assert runner.reruns == (expected if status == "rerun_started" else [])
    if status == "rerun_started":
        assert result.prior_attempt == 1
        assert result.state(HEAD).stale_keys


def test_ordinary_rerun_command_failure_is_reported(tmp_path):
    runner = _LoopRunner(runs={4242: _ordinary_run()}, suites={900: 4242}, rerun_returncode=1)
    result = rerun_ordinary_failed_jobs(runner, make_config(tmp_path), (_publisher(),), HEAD, PLUMBING)
    assert result.status == "command_failed"
    assert result.state(HEAD) is None


# -- Loop integration ----------------------------------------------------------


def test_loop_plumbing_only_failure_reruns_once_then_passes(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _board(_tests(), _publisher()),
            _board(_tests(), _publisher()),                      # stale prior attempt
            _board(_tests(), _publisher("in_progress", check_id=12)),
            _board(_tests(), _publisher("success", check_id=12)),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == [["gh", "run", "rerun", "4242", "--failed", "--repo", "OWNER/REPO"]]
    assert runner.claude_calls() == []
    assert runner.merges() == []
    assert _comments_with(runner, f"re-ran the failed jobs of {RUN_URL} once; no coder round")
    assert not _comments_with(runner, "GitHub PR checks are failing")
    assert "is merge-ready after CI watch completed" in capsys.readouterr().out


@pytest.mark.parametrize("auto_merge", [False, True])
def test_loop_plumbing_failing_twice_stops_on_infrastructure_without_coder(
    tmp_path, monkeypatch, capsys, auto_merge
):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _board(_tests(), _publisher()),
            _board(_tests(), _publisher("failure", check_id=12)),   # fresh attempt fails again
        ],
    )

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, auto_merge=auto_merge)
    ) == 0

    assert len(runner.reruns) == 1
    assert runner.claude_calls() == []
    assert runner.merges() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert len(stop) == 1
    assert "Plumbing failed again after the single automatic rerun" in stop[0]
    assert RUN_URL in stop[0]
    assert not _comments_with(runner, "GitHub PR checks are failing")
    assert "stopped on infrastructure" in capsys.readouterr().out


def test_loop_test_failure_resumes_coder_without_rerun(tmp_path, monkeypatch):
    runner = _approved_runner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Fixed the test.", addressed_items=["item-1"]),
        ],
    )
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _board(_tests("failure"), _publisher()),
            _board(_tests(check_id=20), _publisher("success", check_id=21)),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert len(runner.claude_calls()) == 1
    failing = _comments_with(runner, "GitHub PR checks are failing for PR #77.")
    assert failing and "tests (failure)" in failing[0]


def test_loop_failed_plumbing_status_context_reruns_through_its_target_url(tmp_path, monkeypatch):
    runner = _approved_runner(runs={4343: _ordinary_run(4343)}, suites={})
    old = _final_status("failure")
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _board(_tests(), old),
            _board(_tests(), old),                                        # unchanged prior status
            _board(_tests(), _final_status("success", created_at="2026-10-14T19:05:00Z")),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == [["gh", "run", "rerun", "4343", "--failed", "--repo", "OWNER/REPO"]]
    assert runner.claude_calls() == []
    assert not _comments_with(runner, "CI failed only in plumbing jobs")


def test_loop_new_failed_plumbing_status_in_fresh_attempt_stops_on_infrastructure(tmp_path, monkeypatch):
    runner = _approved_runner(runs={4343: _ordinary_run(4343)}, suites={})
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _board(_tests(), _final_status("failure")),
            _board(_tests(), _final_status("failure", created_at="2026-10-14T19:05:00Z")),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert len(runner.reruns) == 1
    assert runner.claude_calls() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and "https://github.com/OWNER/REPO/actions/runs/4343" in stop[0]
    assert "final-ci/exact-head: failure" in stop[0]


def test_loop_plumbing_status_without_actions_url_is_unresolvable_infrastructure(tmp_path, monkeypatch):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[_board(_tests(), _final_status("failure", url="https://ci.example/status/1"))],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and "could not be performed" in stop[0]
    assert "no resolvable GitHub Actions run" in stop[0]


@pytest.mark.parametrize(
    ("run", "returncode", "reason"),
    [
        (_ordinary_run(head="5e6741348108b11e3f3ef936c335b1967d17ac6c"), 0, "not an ordinary run of 6367c48dd193d56ea7b0baad25b19455e529f5ee"),
        (_ordinary_run(event="workflow_dispatch"), 0, "not an ordinary run of 6367c48dd193d56ea7b0baad25b19455e529f5ee"),
        (_ordinary_run(), 1, "gh run rerun failed"),
    ],
)
def test_loop_unsafe_rerun_stops_on_infrastructure_with_run_url(tmp_path, monkeypatch, run, returncode, reason):
    runner = _approved_runner(runs={4242: run}, rerun_returncode=returncode)
    _install_boards(
        monkeypatch, snapshot=_board(_tests("in_progress")), watcher=[_board(_tests(), _publisher())],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert len(runner.reruns) == (1 if returncode else 0)
    assert runner.claude_calls() == []
    assert runner.merges() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and reason in stop[0] and RUN_URL in stop[0]


def test_loop_snapshot_plumbing_failure_with_watch_reruns_then_watches(tmp_path, monkeypatch):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests(), _publisher()),
        watcher=[
            _board(_tests(), _publisher()),
            _board(_tests(), _publisher("success", check_id=12)),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert len(runner.reruns) == 1
    assert runner.claude_calls() == []
    assert not _comments_with(runner, "GitHub PR checks are failing")


def test_loop_snapshot_plumbing_failure_without_watch_stops_with_hint(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch, snapshot=_board(_tests(), _publisher()), watcher=[_board(_tests(), _publisher())],
    )

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, watch_pending_ci=False)
    ) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and RUN_URL in stop[0]
    assert "--watch-pending-ci" in stop[0] and "--ci-timeout-seconds" in stop[0]
    assert not _comments_with(runner, "GitHub PR checks are failing")


def test_loop_snapshot_plumbing_awaiting_tests_without_watch_takes_hinted_pending_stop(tmp_path, monkeypatch):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress"), _publisher()),
        watcher=[_board(_tests(), _publisher())],
    )

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, watch_pending_ci=False)
    ) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    stop = runner.comments[-1]
    assert stop.startswith("Reviewers approved PR #77, but GitHub checks are still pending.")
    assert "--watch-pending-ci" in stop


@pytest.mark.parametrize("auto_merge", [False, True])
def test_loop_plumbing_failure_while_tests_pending_reruns_once_after_tests_pass(
    tmp_path, monkeypatch, auto_merge
):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress"), _publisher()),
        watcher=[
            _board(_tests("in_progress"), _publisher()),
            _board(_tests(), _publisher()),
            _board(_tests(), _publisher("success", check_id=12)),
        ],
    )
    monkeypatch.setattr(pr_loop_module, "_merge_with_exact_head_proof", lambda *a, **k: None)

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, auto_merge=auto_merge)
    ) == 0

    assert len(runner.reruns) == 1
    assert runner.claude_calls() == []


def test_loop_plumbing_hold_expiry_takes_timeout_path_without_coder(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    held = _board(_tests("in_progress"), _publisher())
    _install_boards(monkeypatch, snapshot=held, watcher=[held])

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, ci_timeout_seconds=90)
    ) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    out = capsys.readouterr().out
    assert "CI watch timed out" in out
    assert "Plumbing failure or stuck required status awaiting tests" in out


def test_loop_incomplete_board_with_plumbing_failure_keeps_coder_routing(tmp_path, monkeypatch):
    runner = _approved_runner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved again.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Investigated.", addressed_items=["item-1"]),
        ],
    )
    partial = _board(_tests(), _publisher(), complete=False)
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[partial, _board(_tests(check_id=20), _publisher("success", check_id=21))],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert len(runner.claude_calls()) == 1


def _stuck_board():
    return _board(_tests(), _final_status("pending"), required=("tests", "final-ci/exact-head"))


def test_loop_stuck_pending_status_is_plumbing_and_reruns_once(tmp_path, monkeypatch):
    runner = _approved_runner(runs={4343: _ordinary_run(4343)}, suites={})
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _stuck_board(), _stuck_board(),          # confirmation poll
            _board(_tests(), _final_status("success", created_at="2026-10-14T19:05:00Z"),
                   required=("tests", "final-ci/exact-head")),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == [["gh", "run", "rerun", "4343", "--failed", "--repo", "OWNER/REPO"]]
    assert runner.claude_calls() == []


def test_loop_stuck_pending_status_without_failed_jobs_stops_without_rerun(tmp_path, monkeypatch):
    runner = _approved_runner(runs={4343: _ordinary_run(4343, conclusion="success")}, suites={})
    _install_boards(
        monkeypatch, snapshot=_board(_tests("in_progress")), watcher=[_stuck_board()],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and "stayed pending after its producing run completed" in stop[0]
    assert "final-ci/exact-head: stuck-pending" in stop[0]


# A stuck status whose run has no failed job waits for other checks (#1498).

STUCK_RUN_URL = "https://github.com/OWNER/REPO/actions/runs/4343"


def _slow_tests(status="in_progress"):
    return PullRequestCheck(
        name="slow tests", kind="check_run", status=status,
        url="https://github.com/OWNER/REPO/actions/runs/5555/job/30", check_id=30,
        check_suite_id=901,
    )


def _stuck_board_with(*others):
    return _board(
        _tests(), _final_status("pending"), *others,
        required=("tests", "final-ci/exact-head", "slow tests"),
    )


def _no_failed_job_runner(**kwargs):
    kwargs.setdefault("runs", {4343: _ordinary_run(4343, conclusion="success")})
    kwargs.setdefault("suites", {})
    return _approved_runner(**kwargs)


def _sleeps(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["sleep"]]


@pytest.mark.parametrize(
    ("settled", "status", "classification"),
    [("success", "plumbing_failure", "stuck-infrastructure"), ("failure", "failed", "test-failure")],
)
def test_watch_holds_stuck_status_without_failed_jobs_until_tests_settle(
    tmp_path, settled, status, classification
):
    held = _stuck_board_with(_slow_tests())
    runner = _LoopRunner(runs={4343: _ordinary_run(4343, conclusion="success")})
    outcome, runner = _watch(
        tmp_path, [held, held, held, _stuck_board_with(_slow_tests(settled))], runner=runner,
    )
    assert outcome.status == status
    assert outcome.classification == classification
    assert outcome.attempts_used == 4
    assert len(_sleeps(runner)) == 3
    assert runner.reruns == []


def test_watch_stuck_status_hold_expiry_takes_the_timeout_path(tmp_path):
    held = _stuck_board_with(_slow_tests())
    runner = _LoopRunner(runs={4343: _ordinary_run(4343, conclusion="success")})
    outcome, runner = _watch(tmp_path, [held] * 3, runner=runner, ci_timeout_seconds=90)
    assert outcome.status == "timeout"
    assert outcome.hold_reason == "plumbing-awaiting-tests"
    assert STUCK_RUN_URL in outcome.run_urls
    assert [check.name for check in outcome.stuck_checks] == ["final-ci/exact-head"]
    assert runner.reruns == []


def test_loop_stuck_status_hold_then_green_stops_on_infrastructure(tmp_path, monkeypatch):
    runner = _no_failed_job_runner()
    held = _stuck_board_with(_slow_tests())
    _install_boards(
        monkeypatch, snapshot=_board(_tests("in_progress")),
        watcher=[held, held, held, _stuck_board_with(_slow_tests("success"))],
    )
    monkeypatch.setattr(
        pr_loop_module, "_merge_with_exact_head_proof",
        lambda *a, **k: pytest.fail("no merge on an infrastructure stop"),
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    assert len(_sleeps(runner)) == 3
    stop = _comments_with(runner, "CI failed only in plumbing jobs")
    assert stop and "no failed job to re-run" in stop[0]


def test_loop_stuck_status_hold_then_test_failure_routes_to_coder(tmp_path, monkeypatch):
    runner = _no_failed_job_runner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved again.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Fixed.", addressed_items=["item-1"]),
        ],
    )
    held = _stuck_board_with(_slow_tests())
    _install_boards(
        monkeypatch, snapshot=_board(_tests("in_progress")),
        watcher=[
            held, held, _stuck_board_with(_slow_tests("failure")),
            _board(
                _tests(check_id=20), _slow_tests("success"),
                _final_status("success", created_at="2026-10-14T19:05:00Z"),
                required=("tests", "final-ci/exact-head", "slow tests"),
            ),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert len(runner.claude_calls()) == 1
    assert not _comments_with(runner, "CI failed only in plumbing jobs")


def test_loop_stuck_status_hold_expiry_takes_timeout_path_without_coder(
    tmp_path, monkeypatch, capsys
):
    runner = _no_failed_job_runner()
    held = _stuck_board_with(_slow_tests())
    _install_boards(monkeypatch, snapshot=_board(_tests("in_progress")), watcher=[held])

    assert run_pr_loop(
        runner, pr_number=77, config=_loop_config(tmp_path, ci_timeout_seconds=90)
    ) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    assert not _comments_with(runner, "CI failed only in plumbing jobs")
    out = capsys.readouterr().out
    assert "CI watch timed out" in out
    assert "stuck required status awaiting tests" in out
    assert f"Run: {STUCK_RUN_URL}" in out


def test_loop_stuck_status_with_genuine_test_failure_routes_to_coder(tmp_path, monkeypatch):
    runner = _approved_runner(
        runs={4343: _ordinary_run(4343)}, suites={},
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved again.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Fixed.", addressed_items=["item-1"]),
        ],
    )
    stuck_with_failure = _board(
        _tests("failure"), _final_status("pending"), required=("tests", "final-ci/exact-head"),
    )
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _stuck_board(), stuck_with_failure,
            _board(_tests(check_id=20), _final_status("success", created_at="2026-10-14T19:05:00Z"),
                   required=("tests", "final-ci/exact-head")),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert len(runner.claude_calls()) == 1


def test_loop_dry_run_preview_never_reruns(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch, snapshot=_board(_tests(), _publisher()), watcher=[_board(_tests(), _publisher())],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path, dry_run=True)) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []


# -- Round-1 review fixes: job-listing verification, shared rerun budget, reporting --

from coding_review_agent_loop.ci_health import CiInfrastructureStall, StalledCheck  # noqa: E402
from coding_review_agent_loop.github import CiWatchOutcome  # noqa: E402


@pytest.mark.parametrize(
    ("jobs", "reason"),
    [
        # A cancelled run with no jobs: --failed would retry nothing.
        ([], "has no failed job for a rerun to retry"),
        ([("tests", "success")], "has no failed job for a rerun to retry"),
        ([("tests", "failure"), ("Publish final exact-head status", "failure")], "failed non-plumbing job"),
        (None, "no readable complete job listing"),
    ],
)
def test_ordinary_rerun_requires_failed_plumbing_jobs_in_the_run_listing(tmp_path, jobs, reason):
    run = _ordinary_run()
    runner = _LoopRunner(runs={4242: run}, suites={900: 4242}, jobs={} if jobs is None else {4242: jobs})
    if jobs is None:
        runner.runs["4242"]["conclusion"] = "cancelled"  # default listing would otherwise exist
        runner.jobs = {}

        def unreadable(self, args, *, cwd, check, input_text=None, _orig=runner._run_locked):
            if any("/jobs?" in str(part) for part in args):
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(cmd, cwd_path, "", "HTTP 502", 1)
            return _orig(args, cwd=cwd, check=check, input_text=input_text)

        runner._run_locked = unreadable.__get__(runner)
    result = rerun_ordinary_failed_jobs(runner, make_config(tmp_path), (_publisher(),), HEAD, PLUMBING)
    assert result.status == "unresolvable"
    assert reason in result.reason
    assert runner.reruns == []


@pytest.mark.parametrize(
    ("jobs", "expect"),
    [
        ([], "infra"),                                                   # cancelled, zero jobs
        ([("tests", "failure"), ("Publish final exact-head status", "failure")], "coder"),
    ],
)
def test_loop_stuck_status_decided_by_run_jobs_not_conclusion(tmp_path, monkeypatch, jobs, expect):
    runner = _approved_runner(
        runs={4343: _ordinary_run(4343, conclusion="cancelled" if not jobs else "failure")},
        suites={}, jobs={4343: jobs},
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved again.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Fixed.", addressed_items=["item-1"]),
        ],
    )
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[
            _stuck_board(), _stuck_board(),
            _board(_tests(check_id=20), _final_status("success", created_at="2026-10-14T19:05:00Z"),
                   required=("tests", "final-ci/exact-head")),
        ],
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    if expect == "infra":
        assert runner.claude_calls() == []
        stop = _comments_with(runner, "CI failed only in plumbing jobs")
        assert stop and "no failed job to re-run" in stop[0]
    else:
        assert len(runner.claude_calls()) == 1


_STATE = PlumbingRerunState(
    head_sha=HEAD, run_id=4242, run_url=RUN_URL, prior_attempt=1,
    stale_keys=frozenset({("check", 11)}),
)


@pytest.mark.parametrize(
    ("observations", "state", "filter_stale", "budget_left", "decision"),
    [
        # Unspent budget: rerun only while the watch can still observe it.
        ((_publisher(),), None, True, True, "rerun"),
        ((_publisher(),), None, False, True, "rerun"),
        ((_publisher(),), None, True, False, "budget_exhausted"),
        ((_publisher(),), None, False, False, "budget_exhausted"),
        # Spent budget, approval snapshot: prior-attempt evidence keeps watching,
        # fresh evidence stops on infrastructure; never a second rerun.
        ((_publisher(),), _STATE, True, True, "watch"),
        ((_publisher(check_id=12),), _STATE, True, True, "infra_stop"),
        ((_publisher(check_id=12),), _STATE, True, False, "infra_stop"),
        # Spent budget, watcher outcome (already stale-filtered): infrastructure.
        ((_publisher(),), _STATE, False, True, "infra_stop"),
    ],
)
def test_every_plumbing_rerun_request_passes_one_budget_decision(
    observations, state, filter_stale, budget_left, decision
):
    assert pr_loop_module._plumbing_rerun_decision(
        observations, state, filter_stale=filter_stale, budget_left=budget_left,
    ) == decision


def test_loop_plumbing_failure_on_the_final_poll_issues_no_rerun(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch,
        snapshot=_board(_tests("in_progress")),
        watcher=[_board(_tests("in_progress")), _board(_tests(), _publisher())],
    )

    assert run_pr_loop(
        runner, pr_number=77,
        config=_loop_config(tmp_path, ci_timeout_seconds=60, ci_poll_interval_seconds=30),
    ) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    out = capsys.readouterr().out
    assert "CI watch timed out" in out
    assert "no rerun was issued" in out


def test_loop_snapshot_plumbing_failure_past_the_deadline_issues_no_rerun(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch, snapshot=_board(_tests(), _publisher()), watcher=[_board(_tests(), _publisher())],
    )
    calls = []
    real_monotonic = pr_loop_module.time.monotonic

    def monotonic():
        # Each reading jumps far past any deadline set from an earlier one,
        # so the shared watch budget is spent before the rerun decision.
        calls.append(None)
        return real_monotonic() + 10_000.0 * len(calls)

    monkeypatch.setattr(pr_loop_module.time, "monotonic", monotonic)

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert runner.reruns == []
    assert runner.claude_calls() == []
    assert "no rerun was issued" in capsys.readouterr().out


_FINAL_PENDING = _final_status("pending")


@pytest.mark.parametrize("variant", ["infrastructure_stall", "auto_merge_passed"])
def test_ordinary_terminal_reports_name_the_final_context_state(tmp_path, monkeypatch, capsys, variant):
    runner = _approved_runner()
    if variant == "infrastructure_stall":
        stalled = PullRequestCheck(name="tests", kind="check_run", status="queued")
        board = _board(stalled, _FINAL_PENDING, required=("tests", "final-ci/exact-head"))
        outcome = CiWatchOutcome(
            status="infrastructure_stall", pr_checks=board, head_sha=HEAD, attempts_used=1,
            stall=CiInfrastructureStall(checks=(StalledCheck(
                name="tests", kind="check_run", reason="queued_too_long",
                check_id=1, run_id="1", url=None, age_seconds=1800.0,
            ),)),
        )
        expected = "final-ci/exact-head: pending"
        config = _loop_config(tmp_path)
    else:
        board = _board(_tests(), _final_status("success"), required=("tests", "final-ci/exact-head"))
        outcome = CiWatchOutcome(
            status="passed", pr_checks=board, head_sha=HEAD, attempts_used=1,
            mergeability=_mergeability(head_sha=HEAD),
        )
        expected = "final-ci/exact-head: success"
        config = _loop_config(tmp_path, auto_merge=True)
        monkeypatch.setattr(pr_loop_module, "_merge_with_exact_head_proof", lambda *a, **k: None)
    monkeypatch.setattr(pr_loop_module, "get_pr_checks", lambda *_a, **_k: _board(_tests("in_progress")))
    monkeypatch.setattr(pr_loop_module, "watch_pr_checks", lambda *a, **k: outcome)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert expected in capsys.readouterr().out
    if variant == "infrastructure_stall":
        assert any(expected in comment for comment in runner.comments)



# -- Round-2: every watch-ending stop names the final-ci/exact-head state ----

from coding_review_agent_loop.pr_loop_support import _stop_after_ci_watch_timeout  # noqa: E402
from coding_review_agent_loop.pr_loop_support import _pr_followup_source_context  # noqa: E402


@pytest.mark.parametrize("auto_merge", [False, True])
@pytest.mark.parametrize(
    "reason", ["budget_exhausted", "timeout", "non_authoritative", "protection_unreadable"],
)
def test_every_watch_timeout_stop_names_the_final_context_state(tmp_path, capsys, reason, auto_merge):
    runner = _LoopRunner()
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=auto_merge)
    board = _board(_tests("in_progress"), _final_status("pending"), required=("tests", "final-ci/exact-head"))
    stuck = _final_status("pending")

    def stop():
        return _stop_after_ci_watch_timeout(
            runner, config=config, pr_number=77, round_number=1, head_sha=HEAD, pr_comments=[],
            followups=[], details=["detail"], reason=reason,
            source_context=_pr_followup_source_context(
                config=config, pr_number=77, pr_metadata=_metadata(), issue_context=None,
            ),
            checks_board=board, stuck_checks=(stuck,) if reason == "timeout" else (),
        )

    expected = "final-ci/exact-head: stuck-pending" if reason == "timeout" else "final-ci/exact-head: pending"
    if auto_merge:
        # The raised report carries the same state as the comment.
        with pytest.raises(AgentLoopError, match=re.escape(expected)):
            stop()
    else:
        stop()
    # Comment and printed report both name the state.
    assert any(expected in comment for comment in runner.comments)
    assert sum(comment.count("final-ci/exact-head:") for comment in runner.comments) == 1
    assert expected in capsys.readouterr().out


def test_loop_budget_exhausted_by_earlier_rounds_names_the_final_context_state(tmp_path, monkeypatch, capsys):
    runner = _approved_runner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved", summary="Approved again.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(state="blocking", summary="Fixed.", addressed_items=["item-1"]),
        ],
    )
    snapshot = _board(_tests("in_progress"), _final_status("pending"), required=("tests", "final-ci/exact-head"))
    monkeypatch.setattr(pr_loop_module, "get_pr_checks", lambda *_a, **_k: snapshot)
    failing = _board(_tests("failure"), required=("tests",))
    monkeypatch.setattr(
        pr_loop_module, "watch_pr_checks",
        lambda *a, **k: CiWatchOutcome(
            status="failed", pr_checks=failing, failed_checks=failing.failing,
            head_sha=HEAD, attempts_used=1,
        ),
    )

    assert run_pr_loop(
        runner, pr_number=77,
        config=_loop_config(tmp_path, ci_timeout_seconds=30, ci_poll_interval_seconds=30),
    ) == 0

    assert len(runner.claude_calls()) == 1
    stop = runner.comments[-1]
    assert "final-ci/exact-head: pending" in stop
    out = capsys.readouterr().out
    assert "no fresh poll was performed" in out
    assert "final-ci/exact-head: pending" in out



# -- Round-3: the deadline is rechecked immediately before the rerun write ----


def test_loop_ordinary_rerun_refused_when_resolver_reads_consume_the_deadline(tmp_path, monkeypatch, capsys):
    runner = _approved_runner()
    _install_boards(
        monkeypatch, snapshot=_board(_tests("in_progress")), watcher=[_board(_tests(), _publisher())],
    )
    elapsed = {"expired": False}
    real_monotonic = github_module.time.monotonic
    real_read_jobs = github_module.read_run_jobs

    def read_jobs(*args, **kwargs):
        # The verification read itself uses up the shared watch budget.
        elapsed["expired"] = True
        return real_read_jobs(*args, **kwargs)

    monkeypatch.setattr(github_module, "read_run_jobs", read_jobs)
    monkeypatch.setattr(
        github_module.time, "monotonic",
        lambda: real_monotonic() + (10_000_000.0 if elapsed["expired"] else 0.0),
    )

    assert run_pr_loop(runner, pr_number=77, config=_loop_config(tmp_path)) == 0

    assert elapsed["expired"]
    assert runner.reruns == []
    assert runner.claude_calls() == []
    assert runner.merges() == []
    out = capsys.readouterr().out
    assert "CI watch timed out" in out and "no rerun was issued" in out
