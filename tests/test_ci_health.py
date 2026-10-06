import datetime
import json
import pytest

from coding_review_agent_loop.ci_health import (
    CiInfrastructureStall,
    PullRequestCheck,
    PullRequestChecks,
    StalledCheck,
    _extract_run_id,
    classify_ci_infrastructure_stall,
    is_canonical_stall_only_text,
    is_canonical_pending_only_text,
    is_wholly_infrastructure_blocked,
)
import coding_review_agent_loop.github as github_module
from coding_review_agent_loop.github import (
    PullRequestCheck as GithubPullRequestCheck,
    PullRequestChecks as GithubPullRequestChecks,
    PullRequestMetadata,
    get_pr_checks,
)
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config

NOW = datetime.datetime(2026, 5, 23, 12, 0, 0, tzinfo=datetime.timezone.utc)


@pytest.mark.parametrize("text", [
    "GitHub check `test` is still pending/in_progress.",
    "CI is pending.",
    "The GitHub PR checks are currently running.",
    "GitHub check status is unavailable.",
    "test is queued",
    "GitHub check `build (linux)` is not yet reporting.",
])
def test_canonical_pending_status_statements(text):
    assert is_canonical_pending_only_text(text, check_names=("test", "build (linux)"))


@pytest.mark.parametrize("text", [
    "GitHub check status is unavailable.",
    "GitHub PR check status is unavailable.",
])
def test_canonical_unavailable_status_without_check_names(text):
    assert is_canonical_pending_only_text(text, check_names=())


@pytest.mark.parametrize("text", [
    "Add a regression test for incorrect plan recovery.",
    "The pending queue loses jobs. Add a regression test.",
    "GitHub check `test` is pending, but authorization is broken.",
    "CI is pending. Fix the missing validation.",
    "The test is still running because the implementation deadlocks.",
    "test", "", "   ",
    "GitHub check `test` is unavailable. The fallback accepts forged approvals.",
    "Review could not be completed because CI is unavailable.",
    "GitHub check `other` is pending.",
    "GitHub check build linux is pending.",
])
def test_pending_filter_preserves_ambiguous_or_substantive_text(text):
    assert not is_canonical_pending_only_text(text, check_names=("test", "build (linux)"))


def _check(**overrides):
    defaults = dict(
        name="test",
        kind="check_run",
        status="queued",
        url=None,
        check_id=None,
        run_id=None,
        created_at=None,
        started_at=None,
        completed_at=None,
    )
    defaults.update(overrides)
    return PullRequestCheck(**defaults)


def test_check_run_parser_uses_github_app_slug_without_misattributing_app_id():
    checks, errors = github_module._parse_check_runs_payload(
        {
            "check_runs": [
                {
                    "id": 7,
                    "name": "CI",
                    "status": "completed",
                    "conclusion": "success",
                    "app": {"slug": "github-actions", "id": 15368},
                }
            ]
        }
    )

    assert errors == []
    assert checks[0].creator_login == "github-actions"
    assert checks[0].creator_id is None


# --- classify_ci_infrastructure_stall ---------------------------------------


def test_queued_past_grace_is_stalled():
    check = _check(status="queued", created_at="2026-05-23T11:00:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert stall.is_stalled
    assert stall.checks[0].reason == "queued_too_long"
    assert stall.checks[0].age_seconds == 3600.0


def test_queued_within_grace_is_not_stalled():
    check = _check(status="queued", created_at="2026-05-23T11:55:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert not stall.is_stalled


def test_long_running_in_progress_is_not_stalled():
    check = _check(status="in_progress", created_at="2026-05-23T09:00:00Z", started_at="2026-05-23T09:00:05Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert not stall.is_stalled


def test_success_and_failure_are_not_stalled():
    for status in ("success", "failure"):
        check = _check(status=status, created_at="2026-05-23T09:00:00Z")
        stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
        assert not stall.is_stalled


def test_cancelled_with_no_start_is_runner_unavailable():
    check = _check(status="cancelled", created_at="2026-05-23T11:58:00Z", started_at=None, completed_at="2026-05-23T11:59:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert stall.is_stalled
    assert stall.checks[0].reason == "runner_unavailable"


def test_cancelled_with_same_start_and_complete_is_runner_unavailable():
    check = _check(status="cancelled", started_at="2026-05-23T11:59:00Z", completed_at="2026-05-23T11:59:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert stall.is_stalled
    assert stall.checks[0].reason == "runner_unavailable"


def test_cancelled_with_real_start_is_not_stalled():
    check = _check(status="cancelled", started_at="2026-05-23T11:00:00Z", completed_at="2026-05-23T11:30:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert not stall.is_stalled


def test_status_context_never_classified():
    check = _check(kind="status_context", status="queued", created_at="2026-05-23T09:00:00Z")
    stall = classify_ci_infrastructure_stall([check], now=NOW, grace_seconds=1200)
    assert not stall.is_stalled


def test_missing_malformed_and_future_timestamps_do_not_raise():
    checks = [
        _check(status="queued", created_at=None),
        _check(status="queued", created_at="not-a-timestamp"),
        _check(status="queued", created_at="2026-05-23T11:00:00Z"),
        _check(status="queued", created_at="2099-01-01T00:00:00Z"),  # future
    ]
    stall = classify_ci_infrastructure_stall(checks, now=NOW, grace_seconds=1200)
    # Only the genuinely-past-grace one should stall; none should raise.
    assert len(stall.checks) == 1
    assert stall.checks[0].age_seconds == 3600.0


def test_z_suffixed_and_offset_timestamps_both_parse():
    z_check = _check(status="queued", created_at="2026-05-23T11:00:00Z")
    offset_check = _check(status="queued", created_at="2026-05-23T11:00:00+00:00")
    stall = classify_ci_infrastructure_stall([z_check, offset_check], now=NOW, grace_seconds=1200)
    assert len(stall.checks) == 2


def test_extract_run_id_from_actions_url():
    url = "https://github.com/OWNER/REPO/actions/runs/31123230205/job/12345"
    assert _extract_run_id(url) == "31123230205"


def test_extract_run_id_non_actions_url():
    assert _extract_run_id("https://example.com/foo") is None


def test_extract_run_id_none():
    assert _extract_run_id(None) is None


# --- is_wholly_infrastructure_blocked ---------------------------------------


def _stalled_check(name="test", run_id="31123230205", check_id=998877, url=None):
    return StalledCheck(
        name=name,
        kind="check_run",
        reason="queued_too_long",
        check_id=check_id,
        run_id=run_id,
        url=url or f"https://github.com/OWNER/REPO/actions/runs/{run_id}",
        age_seconds=3600.0,
    )


def _pr_checks(**overrides):
    defaults = dict(
        state="pending",
        required_checks=(),
        passing=(),
        pending=(_check(status="queued"),),
        failing=(),
        missing_required=(),
        branch_protection_status="configured",
        branch_protection_note=None,
        check_query_status="ok",
        check_query_errors=(),
        infrastructure_stalls=(_stalled_check(),),
    )
    defaults.update(overrides)
    return PullRequestChecks(**defaults)


def test_wholly_blocked_when_all_stalled():
    checks = _pr_checks()
    assert is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_missing_required():
    checks = _pr_checks(missing_required=("lint",))
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_genuine_failing_check():
    checks = _pr_checks(failing=(_check(name="other", status="failure"),))
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_normal_running_check():
    checks = _pr_checks(pending=(_check(status="queued"), _check(name="lint", status="in_progress")))
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_when_branch_protection_unavailable():
    checks = _pr_checks(branch_protection_status="unavailable")
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_when_state_unavailable():
    checks = _pr_checks(state="unavailable")
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_partial_check_query():
    checks = _pr_checks(check_query_status="partial")
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_unavailable_check_query():
    checks = _pr_checks(check_query_status="unavailable")
    assert not is_wholly_infrastructure_blocked(checks)


def test_not_wholly_blocked_with_no_stalls():
    checks = _pr_checks(infrastructure_stalls=(), pending=())
    assert not is_wholly_infrastructure_blocked(checks)


# --- is_canonical_stall_only_text -------------------------------------------


def test_canonical_stall_only_sentence_matches():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = (
        "GitHub check `test` (workflow run 31123230205) has been queued for over "
        "100 minutes because a hosted runner was unavailable; runner unavailable."
    )
    assert is_canonical_stall_only_text(text, stalls=stalls)


def test_canonical_stall_text_plus_code_defect_fails():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = (
        "GitHub check `test` (workflow run 31123230205) runner unavailable, and "
        "models.py has a null pointer bug that crashes on empty input."
    )
    assert not is_canonical_stall_only_text(text, stalls=stalls)


def test_stall_wording_without_identifier_fails():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = "The check is queued and the runner is unavailable."
    assert not is_canonical_stall_only_text(text, stalls=stalls)


def test_identifier_without_stall_semantics_fails():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = "GitHub check `test` (workflow run 31123230205) looks fine to me."
    assert not is_canonical_stall_only_text(text, stalls=stalls)


def test_generic_ci_keywords_only_fails():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = "CI infrastructure failed for this PR."
    assert not is_canonical_stall_only_text(text, stalls=stalls)


def test_overlength_text_fails():
    stalls = [_stalled_check(name="test", run_id="31123230205")]
    text = "GitHub check `test` (workflow run 31123230205) runner unavailable. " + ("queued " * 100)
    assert not is_canonical_stall_only_text(text, stalls=stalls)


def test_empty_text_fails():
    assert not is_canonical_stall_only_text("", stalls=[_stalled_check()])


# --- get_pr_checks -----------------------------------------------------------


class _StubGhRunner(Runner):
    def __init__(
        self,
        *,
        check_runs_payload=None,
        check_runs_returncode=0,
        check_runs_stdout=None,
        status_payload=None,
        status_returncode=0,
        branch_protection_payload=None,
        branch_protection_returncode=0,
        dispatch_payload=None,
        dispatch_returncode=0,
        history_stdout=None,
        history_returncode=0,
    ):
        super().__init__(dry_run=False)
        self.calls = []
        self._dispatch_stdout = json.dumps(dispatch_payload or {"total_count": 0, "workflow_runs": []})
        self._dispatch_returncode = dispatch_returncode
        self._history_stdout = history_stdout
        self._history_returncode = history_returncode
        self._check_runs_stdout = (
            check_runs_stdout if check_runs_stdout is not None else json.dumps(check_runs_payload or {"check_runs": []})
        )
        self._check_runs_returncode = check_runs_returncode
        self._status_stdout = json.dumps(status_payload or {"state": "success", "statuses": []})
        self._status_returncode = status_returncode
        self._branch_protection_stdout = json.dumps(branch_protection_payload or {"contexts": []})
        self._branch_protection_returncode = branch_protection_returncode

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(a) for a in args]
        self.calls.append(cmd)
        if cmd[:3] == ["gh", "api", "--paginate"]:
            return CommandResult(cmd, cwd, self._history_stdout or "", "", self._history_returncode)
        if cmd[:2] == ["gh", "api"] and "/actions/runs?" in cmd[2]:
            return CommandResult(cmd, cwd, self._dispatch_stdout, "", self._dispatch_returncode)
        if cmd[:2] == ["gh", "api"] and "protection/required_status_checks" in cmd[2]:
            return CommandResult(cmd, cwd, self._branch_protection_stdout, "", self._branch_protection_returncode)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/check-runs"):
            return CommandResult(cmd, cwd, self._check_runs_stdout, "", self._check_runs_returncode)
        if cmd[:2] == ["gh", "api"] and cmd[2].endswith("/status"):
            return CommandResult(cmd, cwd, self._status_stdout, "", self._status_returncode)
        raise AssertionError(f"unexpected command: {cmd}")


def _metadata():
    return PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )


def test_get_pr_checks_populates_new_fields(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(
        check_runs_payload={
            "check_runs": [
                {
                    "id": 555,
                    "name": "test",
                    "status": "queued",
                    "conclusion": None,
                    "html_url": "https://github.com/OWNER/REPO/actions/runs/31123230205/job/1",
                    "created_at": "2026-05-23T11:00:00Z",
                    "started_at": None,
                    "completed_at": None,
                }
            ]
        },
        branch_protection_payload={"contexts": ["test"]},
    )

    pr_checks = get_pr_checks(runner, config=config, metadata=_metadata(), now=NOW)

    assert pr_checks.check_query_status == "ok"
    assert pr_checks.check_query_errors == ()
    check = pr_checks.pending[0]
    assert check.check_id == 555
    assert check.run_id == "31123230205"
    assert check.created_at == "2026-05-23T11:00:00Z"
    assert len(pr_checks.infrastructure_stalls) == 1
    assert pr_checks.infrastructure_stalls[0].reason == "queued_too_long"


def test_get_pr_checks_partial_query_failure(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(
        check_runs_payload={"check_runs": [{"id": 1, "name": "test", "status": "completed", "conclusion": "success"}]},
        status_returncode=1,
        branch_protection_payload={"contexts": ["test"]},
    )

    pr_checks = get_pr_checks(runner, config=config, metadata=_metadata(), now=NOW)

    assert pr_checks.check_query_status == "partial"
    assert "commit-status query failed" in pr_checks.check_query_errors


def test_get_pr_checks_malformed_json_is_partial(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(
        check_runs_stdout="not json",
        status_payload={"state": "success", "statuses": []},
        branch_protection_payload={"contexts": []},
    )

    pr_checks = get_pr_checks(runner, config=config, metadata=_metadata(), now=NOW)

    assert pr_checks.check_query_status == "partial"
    assert "check-runs response was not valid JSON" in pr_checks.check_query_errors


def test_get_pr_checks_both_queries_fail_is_unavailable(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(check_runs_returncode=1, status_returncode=1)

    pr_checks = get_pr_checks(runner, config=config, metadata=_metadata(), now=NOW)

    assert pr_checks.check_query_status == "unavailable"


def test_reexported_names_import_from_github_module():
    assert github_module.PullRequestCheck is PullRequestCheck
    assert github_module.PullRequestChecks is PullRequestChecks
    assert GithubPullRequestCheck is PullRequestCheck
    assert GithubPullRequestChecks is PullRequestChecks


def test_full_board_types_remain_reexported_for_current_callers():
    assert github_module.PullRequestCheck.__name__ == "PullRequestCheck"
    assert github_module.PullRequestChecks.__name__ == "PullRequestChecks"


# --- shadowed observations and listing completeness (#1117) -------------------


def _run(name, conclusion="success", run_id=1):
    return {"id": run_id, "name": name, "status": "completed", "conclusion": conclusion}


def _board(tmp_path, check_runs, *, statuses=None, total=None, status_total=None, protection=None):
    payload = {"check_runs": check_runs}
    if total is not False:
        payload["total_count"] = len(check_runs) if total is None else total
    status_payload = {"state": "success", "statuses": statuses or []}
    status_payload["total_count"] = len(statuses or []) if status_total is None else status_total
    runner = _StubGhRunner(
        check_runs_payload=payload,
        status_payload=status_payload,
        branch_protection_payload=protection or {"contexts": []},
    )
    return get_pr_checks(runner, config=make_config(tmp_path), metadata=_metadata(), now=NOW)


@pytest.mark.parametrize("order", ["success-first", "failure-first"])
def test_shadowed_keeps_dropped_same_name_observation_without_changing_board(tmp_path, order):
    runs = [_run("X", "success", 1), _run("X", "failure", 2)]
    if order == "failure-first":
        runs.reverse()
    board = _board(tmp_path, runs)

    assert [c.name for c in board.passing + board.failing] == ["X"]
    assert len(board.shadowed) == 1
    assert board.shadowed[0].name == "X"
    assert board.shadowed[0].status != (board.passing + board.failing)[0].status
    assert board.listing_complete is True


def test_no_duplicates_means_empty_shadowed(tmp_path):
    board = _board(tmp_path, [_run("A"), _run("B")])
    assert board.shadowed == ()
    assert board.listing_complete is True


def test_listing_incomplete_when_total_count_exceeds_entries(tmp_path):
    assert _board(tmp_path, [_run("A")], total=2).listing_complete is False
    assert _board(tmp_path, [_run("A")], status_total=3).listing_complete is False


def test_listing_incomplete_when_total_count_missing(tmp_path):
    assert _board(tmp_path, [_run("A")], total=False).listing_complete is False


@pytest.mark.parametrize("bad", ["not-an-object", {"id": 9, "status": "completed", "conclusion": "failure"}])
def test_listing_incomplete_when_counted_check_run_entry_is_skipped(tmp_path, bad):
    board = _board(tmp_path, [_run("A"), bad])
    assert board.listing_complete is False
    assert board.check_query_errors == ()
    assert [c.name for c in board.passing] == ["A"]


def test_listing_incomplete_when_status_entry_lacks_context(tmp_path):
    board = _board(tmp_path, [_run("A")], statuses=[{"state": "failure"}])
    assert board.listing_complete is False
    assert board.check_query_errors == ()


def test_unset_listing_defaults_to_incomplete():
    board = PullRequestChecks(
        state="passing", required_checks=(), passing=(), pending=(), failing=(),
        missing_required=(), branch_protection_status="not_found",
    )
    assert board.listing_complete is False
    assert board.shadowed == ()


# ---- ad hoc workflow_dispatch exclusion (#1293) ----

SUITE_DISPATCH = 900
SUITE_PR = 901


def _cr(cid, name, conclusion, *, suite=SUITE_PR, app="github-actions", run=None, url=None):
    raw = {
        "id": cid,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "html_url": url or (f"https://github.com/OWNER/REPO/actions/runs/{run}/job/{cid}" if run else None),
    }
    if suite is not None:
        raw["check_suite"] = {"id": suite}
    if app is not None:
        raw["app"] = {"slug": app, "id": 15368, "name": app}
    return raw


def _listing(*runs):
    return {"total_count": len(runs), "check_runs": list(runs)}


def _dispatch(run_id=7000, suite=SUITE_DISPATCH, name="Negative evidence", **extra):
    run = {
        "id": run_id,
        "name": name,
        "display_title": name,
        "event": "workflow_dispatch",
        "head_sha": "abc123",
        "check_suite_id": suite,
        "repository": {"full_name": "owner/repo"},
    }
    run.update(extra)
    return {"total_count": 1, "workflow_runs": [run]}


def _history(*runs, pages=1):
    """Concatenated pages like gh --paginate without --slurp."""
    runs = list(runs)
    total = len(runs)
    size = max(1, -(-total // pages))
    chunks = [runs[i : i + size] for i in range(0, total, size)] or [[]]
    return "".join(json.dumps({"total_count": total, "check_runs": c}) for c in chunks)


def _fetch(tmp_path, **kwargs):
    runner = _StubGhRunner(**kwargs)
    return get_pr_checks(runner, config=make_config(tmp_path), metadata=_metadata()), runner


def test_dispatch_failing_check_is_excluded(tmp_path):
    ci = _cr(1, "ci", "success", run=5000)
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(neg, ci),
        dispatch_payload=_dispatch(),
        history_stdout=_history(neg, ci),
    )
    assert checks.state == "passing"
    assert not checks.failing
    assert [c.name for c in checks.excluded] == ["validate"]
    assert any("7000" in n and "abc123" in n for n in checks.exclusion_notes)


def test_pull_request_failing_check_still_counts(tmp_path):
    ci = _cr(1, "ci", "failure", run=5000)
    checks, runner = _fetch(tmp_path, check_runs_payload=_listing(ci))
    assert checks.state == "failing"
    assert not checks.excluded
    assert not any("--paginate" in c for c in runner.calls)


def test_managed_run_stays_counted(tmp_path):
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(neg),
        dispatch_payload=_dispatch(name="managed-ci-v2 nonce=abc"),
    )
    assert checks.state == "failing"
    assert not checks.excluded


def test_status_context_never_excluded(tmp_path):
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    ci = _cr(1, "ci", "success", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(neg, ci),
        status_payload={
            "state": "failure",
            "total_count": 1,
            "statuses": [{"context": "final-ci/exact-head", "state": "failure"}],
        },
        dispatch_payload=_dispatch(),
        history_stdout=_history(neg, ci),
    )
    assert checks.state == "failing"
    assert [c.name for c in checks.failing] == ["final-ci/exact-head"]
    assert [c.name for c in checks.excluded] == ["validate"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dispatch_returncode": 1},
        {"dispatch_payload": {"total_count": 5, "workflow_runs": []}},
    ],
)
def test_dispatch_lookup_failure_excludes_nothing(tmp_path, kwargs):
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    checks, _ = _fetch(tmp_path, check_runs_payload=_listing(neg), **kwargs)
    assert checks.state == "failing"
    assert not checks.excluded
    assert checks.exclusion_notes


def test_dispatch_lookup_invalid_json_excludes_nothing(tmp_path):
    neg = _cr(2, "validate", "failure", suite=SUITE_DISPATCH, run=7000)
    runner = _StubGhRunner(check_runs_payload=_listing(neg))
    runner._dispatch_stdout = "not json"
    checks = get_pr_checks(runner, config=make_config(tmp_path), metadata=_metadata())
    assert checks.state == "failing"
    assert checks.exclusion_notes


def test_unjoined_suite_stays_counted(tmp_path):
    other = _cr(2, "validate", "failure", suite=12345, run=7000)
    checks, _ = _fetch(tmp_path, check_runs_payload=_listing(other), dispatch_payload=_dispatch())
    assert checks.state == "failing"
    assert not checks.excluded


def test_required_check_only_from_excluded_run_is_missing(tmp_path):
    neg = _cr(2, "validate", "success", suite=SUITE_DISPATCH, run=7000)
    ci = _cr(1, "ci", "success", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(neg, ci),
        branch_protection_payload={"contexts": ["validate"]},
        dispatch_payload=_dispatch(),
        history_stdout=_history(neg, ci),
    )
    assert checks.missing_required == ("validate",)
    assert checks.state == "pending"


def test_excluded_newer_dispatch_does_not_shadow_counted_failure(tmp_path):
    new_dispatch = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    old_pr = _cr(1, "x", "failure", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(new_dispatch, y),
        dispatch_payload=_dispatch(),
        history_stdout=_history(new_dispatch, y, old_pr),
    )
    assert checks.state == "failing"
    assert [c.name for c in checks.failing] == ["x"]
    assert [c.name for c in checks.excluded] == ["x"]


def _shadow_setup():
    new_dispatch = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    old_pr = _cr(1, "x", "failure", run=5000)
    return new_dispatch, y, old_pr


@pytest.mark.parametrize(
    "history",
    [
        None,  # failed
        "truncated",
        "{not json",
    ],
)
def test_history_unavailable_is_non_authoritative(tmp_path, history):
    new_dispatch, y, old_pr = _shadow_setup()
    if history is None:
        kwargs = {"history_returncode": 1, "history_stdout": ""}
    elif history == "truncated":
        full = json.loads(_history(new_dispatch, y, old_pr))
        full["total_count"] = 5
        kwargs = {"history_stdout": json.dumps(full)}
    else:
        kwargs = {"history_stdout": history}
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(new_dispatch, y),
        dispatch_payload=_dispatch(),
        **kwargs,
    )
    assert checks.state == "unavailable"
    assert checks.check_query_status == "partial"
    assert checks.check_query_errors
    assert checks.listing_complete is False
    assert not checks.excluded


def test_history_unavailable_with_failing_check_stays_failing(tmp_path):
    new_dispatch = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    bad = _cr(4, "y", "failure", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(new_dispatch, bad),
        dispatch_payload=_dispatch(),
        history_returncode=1,
    )
    assert checks.state == "failing"
    assert checks.check_query_status == "partial"


def test_non_actions_check_pointing_at_dispatch_run_stays_counted(tmp_path):
    spoof = _cr(
        2, "validate", "failure", suite=12345, app="other-app", url="https://github.com/OWNER/REPO/actions/runs/7000/job/2"
    )
    checks, _ = _fetch(tmp_path, check_runs_payload=_listing(spoof), dispatch_payload=_dispatch())
    assert checks.state == "failing"
    assert not checks.excluded


def test_suite_run_conflict_stays_counted_from_history(tmp_path):
    legit = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    conflicting = _cr(1, "x", "failure", suite=SUITE_DISPATCH, run=7001)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(legit),
        dispatch_payload=_dispatch(),
        history_stdout=_history(legit, conflicting),
    )
    assert checks.state == "failing"
    assert [c.check_id for c in checks.excluded] == [5]
    assert any("different dispatch run" in n for n in checks.exclusion_notes)


def test_suite_run_conflict_both_in_default_listing(tmp_path):
    legit = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    conflicting = _cr(1, "x", "failure", suite=SUITE_DISPATCH, run=7001)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(legit, conflicting),
        dispatch_payload=_dispatch(),
        history_stdout=_history(legit, conflicting),
    )
    assert checks.state == "failing"
    assert [c.check_id for c in checks.failing] == [1]


def test_independent_suite_observation_reaches_shadowed(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    newer = _cr(8, "x", "success", run=5000)
    independent = _cr(3, "x", "failure", suite=555, run=5001)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, newer),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, newer, independent),
    )
    assert [c.check_id for c in checks.shadowed] == [3]
    assert checks.state == "passing"


def test_identity_less_observations_are_not_collapsed(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    newer = _cr(8, "x", "success", suite=None, app=None)
    older = _cr(3, "x", "failure", suite=None, app=None)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, newer),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, newer, older),
    )
    assert [c.check_id for c in checks.shadowed] == [3]


def test_rerun_in_same_suite_is_collapsed(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    new = _cr(8, "x", "success", run=5000)
    old = _cr(3, "x", "failure", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, new),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, new, old),
    )
    assert not checks.shadowed
    assert checks.state == "passing"


def test_history_omitting_counted_default_observation_falls_back(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    z = _cr(6, "z", "success", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, y, z),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, y),
    )
    assert checks.state == "unavailable"
    assert checks.check_query_status == "partial"
    assert not checks.excluded


def test_paginated_history_decodes_all_pages(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    old_x = _cr(1, "x", "failure", run=5000)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, y),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, y, old_x, pages=2),
    )
    assert checks.state == "failing"
    assert [c.name for c in checks.failing] == ["x"]


def _two_pages(mutate):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    pages = [
        {"total_count": 2, "check_runs": [dispatch_x]},
        {"total_count": 2, "check_runs": [y]},
    ]
    return pages, mutate


@pytest.mark.parametrize(
    "variant",
    ["short", "malformed", "non_object", "no_check_runs", "diff_total", "missing_id", "string_id", "bool_id", "dup_id"],
)
def test_bad_history_variants_fall_back_without_raising(tmp_path, variant):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    p1 = {"total_count": 2, "check_runs": [dispatch_x]}
    p2 = {"total_count": 2, "check_runs": [y]}
    if variant == "short":
        p2 = {"total_count": 3, "check_runs": [y]}
        p1["total_count"] = 3
    elif variant == "diff_total":
        p2["total_count"] = 3
    elif variant == "no_check_runs":
        p2 = {"total_count": 2}
    elif variant == "missing_id":
        del p2["check_runs"][0]["id"]
    elif variant == "string_id":
        p2["check_runs"][0]["id"] = "4"
    elif variant == "bool_id":
        p2["check_runs"][0]["id"] = True
    elif variant == "dup_id":
        p2["check_runs"][0]["id"] = 9
    text = json.dumps(p1) + json.dumps(p2)
    if variant == "malformed":
        text = json.dumps(p1) + "{oops"
    elif variant == "non_object":
        text = json.dumps(p1) + "[1]"
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, y),
        dispatch_payload=_dispatch(),
        history_stdout=text,
    )
    assert checks.state == "unavailable"
    assert checks.check_query_status == "partial"
    assert not checks.excluded


def test_default_listing_counted_check_without_int_id_falls_back(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    del y["id"]
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(dispatch_x, y),
        dispatch_payload=_dispatch(),
        history_stdout=_history(dispatch_x, _cr(4, "y", "success", run=5000)),
    )
    assert checks.state == "unavailable"


def test_read_cost_calls(tmp_path):
    # status-context only / non-Actions: no dispatch call
    runner_checks, runner = _fetch(
        tmp_path, check_runs_payload=_listing(_cr(1, "a", "success", app="other-app"))
    )
    assert not any("/actions/runs?" in c[2] for c in runner.calls)
    # Actions without dispatch suites: one dispatch call, no history
    _, runner = _fetch(tmp_path, check_runs_payload=_listing(_cr(1, "a", "success", run=5000)))
    assert sum("/actions/runs?" in c[2] for c in runner.calls) == 1
    assert not any(c[:3] == ["gh", "api", "--paginate"] for c in runner.calls)
    # candidate: both
    neg = _cr(2, "v", "failure", suite=SUITE_DISPATCH, run=7000)
    _, runner = _fetch(
        tmp_path,
        check_runs_payload=_listing(neg),
        dispatch_payload=_dispatch(),
        history_stdout=_history(neg),
    )
    assert sum("/actions/runs?" in c[2] for c in runner.calls) == 1
    assert any(c[:3] == ["gh", "api", "--paginate"] for c in runner.calls)


def test_default_listing_conflict_is_noted_without_a_rebuild(tmp_path):
    conflicting = _cr(1, "x", "failure", suite=SUITE_DISPATCH, run=7001)
    checks, runner = _fetch(
        tmp_path, check_runs_payload=_listing(conflicting), dispatch_payload=_dispatch()
    )
    assert checks.state == "failing"
    assert not checks.excluded
    assert sum("different dispatch run" in n for n in checks.exclusion_notes) == 1
    assert not any(c[:3] == ["gh", "api", "--paginate"] for c in runner.calls)


def test_default_listing_conflict_is_noted_once_when_history_repeats_it(tmp_path):
    legit = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    conflicting = _cr(1, "x", "failure", suite=SUITE_DISPATCH, run=7001)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(legit, conflicting),
        dispatch_payload=_dispatch(),
        history_stdout=_history(legit, conflicting),
    )
    assert sum("different dispatch run" in n for n in checks.exclusion_notes) == 1


def test_default_listing_conflict_is_noted_when_history_falls_back(tmp_path):
    legit = _cr(5, "x", "success", suite=SUITE_DISPATCH, run=7000)
    conflicting = _cr(1, "x", "failure", suite=SUITE_DISPATCH, run=7001)
    checks, _ = _fetch(
        tmp_path,
        check_runs_payload=_listing(legit, conflicting),
        dispatch_payload=_dispatch(),
        history_returncode=1,
    )
    assert any("different dispatch run" in n for n in checks.exclusion_notes)
    assert any("non-authoritative" in n for n in checks.exclusion_notes)
