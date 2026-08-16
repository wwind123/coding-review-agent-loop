import datetime
import json

from coding_review_agent_loop.ci_health import (
    CiInfrastructureStall,
    PullRequestCheck,
    PullRequestChecks,
    StalledCheck,
    _extract_run_id,
    classify_ci_infrastructure_stall,
    is_canonical_stall_only_text,
    is_wholly_infrastructure_blocked,
)
import coding_review_agent_loop.github as github_module
from coding_review_agent_loop.github import (
    PullRequestCheck as GithubPullRequestCheck,
    PullRequestChecks as GithubPullRequestChecks,
    PullRequestMetadata,
    get_check_record,
    get_check_status,
    get_pr_checks,
)
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config

NOW = datetime.datetime(2026, 5, 23, 12, 0, 0, tzinfo=datetime.timezone.utc)


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
    ):
        super().__init__(dry_run=False)
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


def test_get_check_record_returns_full_record(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(
        check_runs_payload={
            "check_runs": [
                {
                    "id": 42,
                    "name": "test",
                    "status": "completed",
                    "conclusion": "success",
                    "started_at": "2026-05-23T11:00:00Z",
                    "completed_at": "2026-05-23T11:05:00Z",
                }
            ]
        }
    )

    record = get_check_record(runner, config, "abc123")

    assert record is not None
    assert record.check_id == 42
    assert record.status == "success"
    assert get_check_status(runner, config, "abc123") == "success"


def test_get_check_record_missing_check_returns_none(tmp_path):
    config = make_config(tmp_path)
    runner = _StubGhRunner(check_runs_payload={"check_runs": []})

    assert get_check_record(runner, config, "abc123") is None
    assert get_check_status(runner, config, "abc123") == "pending"
