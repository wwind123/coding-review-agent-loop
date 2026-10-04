"""Stage 1 of #510: shared transient GitHub policy and read-only retry (#1257)."""

import inspect
import json
import re
from pathlib import Path

import pytest

from coding_review_agent_loop import github, github_retry
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    _gh_api_json,
    get_issue_context,
    read_authenticated_protocol_comments,
)
from coding_review_agent_loop.github_retry import (
    GitHubTransientExhaustedError,
    RetriedCommandResult,
    classify_gh_failure,
    describe_gh_failure,
    run_gh_read,
)
from coding_review_agent_loop.runner import CommandResult, Runner

from agent_loop_helpers import make_config


class ScriptedRunner(Runner):
    """Replays scripted (returncode, stdout, stderr) results; the last one repeats."""

    def __init__(self, script, *, dry_run=False):
        super().__init__(dry_run=False)
        self.dry_run = dry_run
        self.script = list(script)
        self.calls = []

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        self.calls.append(list(args))
        entry = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        rc, out, err = entry
        result = CommandResult(list(args), cwd, out, err, rc)
        if check and rc != 0:
            raise AgentLoopError(f"Command failed with exit {rc}")
        return result


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(github_retry, "_sleep", recorded.append)
    return recorded


def _result(stderr, rc=1, stdout=""):
    return CommandResult(["gh"], Path("."), stdout, stderr, rc)


@pytest.mark.parametrize(
    "stderr",
    [
        "non-200 OK status code: 502 Bad Gateway",
        "HTTP 500: Internal Server Error",
        "HTTP 503: Service Unavailable",
        "gh: HTTP 504 Gateway Timeout",
        "Post https://api.github.com/graphql: couldn't respond to your request in time",
        "read tcp: connection reset by peer",
        "dial tcp: connection refused",
        "net/http: TLS handshake timeout",
        "dial tcp 1.2.3.4:443: i/o timeout",
        "Get https://api.github.com: unexpected EOF",
    ],
)
def test_classifier_transient_forms(stderr):
    assert classify_gh_failure(_result(stderr)) == "transient"


@pytest.mark.parametrize(
    "stderr",
    [
        "HTTP 401: Bad credentials",
        "HTTP 403: Resource not accessible by integration",
        "HTTP 404: Not Found",
        "HTTP 422: Validation Failed",
        "HTTP 429: rate limit exceeded",
        "You have exceeded a secondary rate limit",
        "To authenticate, run: gh auth login",
        "billing issue on account",
        "something unrecognised happened",
        # Mixed text: any permanent marker wins over a transient phrase.
        "HTTP 502 Bad Gateway ... HTTP 404 Not Found",
        "Bad Gateway while validation failed",
    ],
)
def test_classifier_non_transient_forms(stderr):
    assert classify_gh_failure(_result(stderr)) == "permanent"


def test_classifier_success_is_not_retryable():
    assert classify_gh_failure(_result("HTTP 502", rc=0)) == "permanent"


def test_read_transient_then_ok(sleeps):
    runner = ScriptedRunner([(1, "", "HTTP 504 Gateway Timeout"), (0, "{}", "")])
    result = run_gh_read(runner, ["gh", "issue", "view", "1"], cwd=Path("."))
    assert result.returncode == 0
    assert len(runner.calls) == 2
    assert len(sleeps) == 1  # injected sleep: no real waiting
    assert isinstance(result, RetriedCommandResult)
    assert [a.classification for a in result.attempts] == ["transient"]


def test_clean_success_is_plain_result(sleeps):
    runner = ScriptedRunner([(0, "ok", "")])
    result = run_gh_read(runner, ["gh", "pr", "view", "1"], cwd=Path("."))
    assert type(result) is CommandResult
    assert sleeps == []


@pytest.mark.parametrize("stderr", ["HTTP 404: Not Found", "HTTP 401 Bad credentials", "HTTP 422 Validation Failed"])
def test_non_transient_no_retry_keeps_error_shape(stderr, sleeps):
    runner = ScriptedRunner([(1, "", stderr)])
    with pytest.raises(AgentLoopError, match="Command failed with exit 1") as info:
        run_gh_read(runner, ["gh", "pr", "view", "1"], cwd=Path("."))
    assert not isinstance(info.value, GitHubTransientExhaustedError)
    assert len(runner.calls) == 1
    assert sleeps == []


def test_non_transient_check_false_returns_original_result(sleeps):
    runner = ScriptedRunner([(1, "", "HTTP 404: Not Found")])
    result = run_gh_read(runner, ["gh", "api", "x"], cwd=Path("."), check=False)
    assert result.returncode == 1 and len(runner.calls) == 1
    assert type(result) is CommandResult


def test_exhausted_history_check_true(sleeps):
    runner = ScriptedRunner([(1, "", "HTTP 502 first"), (1, "", "HTTP 503 second"), (1, "", "HTTP 504 final")])
    with pytest.raises(GitHubTransientExhaustedError) as info:
        run_gh_read(runner, ["gh", "pr", "view", "1"], cwd=Path("."))
    message = str(info.value)
    assert "HTTP 504 final" in message
    for number in (1, 2, 3):
        assert f"attempt {number}:" in message
    assert len(runner.calls) == 3  # no fourth attempt
    assert len(sleeps) == 2  # backoff only between attempts
    assert 1.5 <= sleeps[0] <= 2.5 and 3.75 <= sleeps[1] <= 6.25
    assert isinstance(info.value, AgentLoopError)


def test_exhausted_history_check_false(sleeps):
    runner = ScriptedRunner([(1, "", "HTTP 502 Bad Gateway")])
    result = run_gh_read(runner, ["gh", "api", "x"], cwd=Path("."), check=False)
    assert result.exhausted and len(result.attempts) == 3
    text = describe_gh_failure(result)
    assert "HTTP 502 Bad Gateway" in text and "attempt 3:" in text


def test_describe_plain_result_is_just_stderr():
    assert describe_gh_failure(_result("boom\n")) == "boom"


def test_dry_run_bypasses_policy(sleeps):
    runner = ScriptedRunner([(1, "", "HTTP 502 Bad Gateway")], dry_run=True)
    result = run_gh_read(runner, ["gh", "api", "x"], cwd=Path("."), check=False)
    assert len(runner.calls) == 1 and sleeps == []
    assert type(result) is CommandResult


# --- public read paths -------------------------------------------------------


def test_get_issue_context_exhaustion_message(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 504 final-diagnostic")])
    with pytest.raises(AgentLoopError) as info:
        get_issue_context(runner, config=config, issue_number=5)
    message = str(info.value)
    assert "final-diagnostic" in message and "attempt 3:" in message


def test_get_issue_context_retries_transient_then_succeeds(tmp_path, sleeps):
    config = make_config(tmp_path)
    payload = json.dumps(
        {"number": 5, "title": "t", "body": "b", "url": "u", "state": "OPEN", "comments": []}
    )
    runner = ScriptedRunner([(1, "", "HTTP 504 Gateway Timeout"), (0, payload, "")])
    context = get_issue_context(runner, config=config, issue_number=5)
    assert context.number == 5
    assert len(runner.calls) == 2


def test_gh_api_json_exhaustion_and_404(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 502 api-final")])
    with pytest.raises(AgentLoopError) as info:
        _gh_api_json(runner, config=config, path="repos/o/r/x", description="thing")
    assert "api-final" in str(info.value) and "attempt 3:" in str(info.value)
    assert len(runner.calls) == 3

    runner = ScriptedRunner([(1, "", "HTTP 404: Not Found")])
    with pytest.raises(AgentLoopError):
        _gh_api_json(runner, config=config, path="repos/o/r/x", description="thing")
    assert len(runner.calls) == 1  # a 404 is permanent and never retried


def test_read_authenticated_protocol_comments_exhaustion_message(tmp_path, sleeps):
    config = make_config(tmp_path)
    user = json.dumps({"login": "bot", "id": 7})

    class Runner2(ScriptedRunner):
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            if list(args)[-1] == "user":
                return CommandResult(list(args), cwd, user, "", 0)
            return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)

    runner = Runner2([(1, "", "HTTP 503 page-final")])
    with pytest.raises(AgentLoopError) as info:
        read_authenticated_protocol_comments(runner, config=config, surface_kind="pr", number=3)
    assert "page-final" in str(info.value) and "attempt 3:" in str(info.value)


# --- guard: no write verb reaches run_gh_read ---------------------------------

_WRITE_TOKENS = re.compile(
    r'"--method"|"-X"|"POST"|"PATCH"|"PUT"|"DELETE"|"comment"|"create"|"ready"|"merge"|"edit"'
)


def test_run_gh_read_call_sites_are_read_only():
    import coding_review_agent_loop.managed_ci as managed_ci

    checked = 0
    for module in (github, managed_ci):
        source = inspect.getsource(module)
        for match in re.finditer(r"run_gh_read\(", source):
            # Examine the call's argument list up to its closing cwd= keyword.
            end = source.find("cwd=", match.end())
            block = source[match.end():end]
            assert not _WRITE_TOKENS.search(block), block
            checked += 1
    assert checked >= 20


# --- remaining diagnostic paths (review round 1) ---------------------------------


def test_stdout_only_failure_keeps_original_error(sleeps):
    graphql = '{"errors":[{"message":"couldn\'t respond to your request in time"}]}'
    runner = ScriptedRunner([(1, graphql, "")])
    with pytest.raises(GitHubTransientExhaustedError) as info:
        run_gh_read(runner, ["gh", "api", "graphql"], cwd=Path("."))
    message = str(info.value)
    assert "couldn't respond to your request in time" in message
    assert "<no stderr>" not in message
    assert len(runner.calls) == 3


def test_stdout_kept_alongside_stderr(sleeps):
    runner = ScriptedRunner([(1, "structured-detail", "HTTP 502 Bad Gateway")])
    result = run_gh_read(runner, ["gh", "api", "x"], cwd=Path("."), check=False)
    text = describe_gh_failure(result)
    assert "HTTP 502 Bad Gateway" in text and "structured-detail" in text


def test_actor_lookup_exhaustion_message(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 503 actor-final")])
    with pytest.raises(AgentLoopError) as info:
        read_authenticated_protocol_comments(runner, config=config, surface_kind="pr", number=3)
    assert "actor-final" in str(info.value) and "attempt 3:" in str(info.value)


def test_envelope_readback_exhaustion_message(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 502 envelope-final")])
    with pytest.raises(AgentLoopError) as info:
        github._fetch_protocol_comment_envelope(
            runner, config=config, comment_id=9, context="Test record"
        )
    assert "envelope-final" in str(info.value) and "attempt 3:" in str(info.value)


def test_merge_commit_exhaustion_message(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 504 merge-final")])
    with pytest.raises(AgentLoopError) as info:
        github.get_pr_merge_commit_sha(runner, config, 7)
    assert "merge-final" in str(info.value) and "attempt 3:" in str(info.value)


def test_pr_commit_query_exhaustion_keeps_history_and_refusal_type(tmp_path, sleeps):
    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 502 commits-final")])
    with pytest.raises(AgentLoopError) as info:
        github._query_pr_commit_connection(runner, config=config, pr_number=7, after=None)
    assert "commits-final" in str(info.value) and "attempt 3:" in str(info.value)


def test_managed_ci_list_failure_reason_reaches_callers(tmp_path, sleeps):
    from coding_review_agent_loop import managed_ci

    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 503 list-final")])
    with pytest.raises(AgentLoopError) as info:
        managed_ci._authorization_comment_records(
            runner, config=config, pr_number=4, actor_login="bot", actor_id=7
        )
    message = str(info.value)
    assert "could not be inspected" in message
    assert "list-final" in message and "attempt 3:" in message
    assert managed_ci._api_list(runner, config, "repos/o/r/issues/4/comments") is None


# --- round 2: branch protection and managed-CI list consumers ---------------------


def test_get_pr_checks_branch_protection_exhaustion_keeps_diagnostics(tmp_path, sleeps):
    config = make_config(tmp_path)
    metadata = github.PullRequestMetadata(7, "OWNER/REPO", "title", "head", "main", "sha", "url")

    class ProtectionRunner(ScriptedRunner):
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            self.calls.append(list(args))
            if "required_status_checks" in " ".join(args):
                return CommandResult(list(args), cwd, "", "HTTP 503 protection-final", 1)
            return CommandResult(list(args), cwd, '{"check_runs": [], "statuses": []}', "", 0)

    runner = ProtectionRunner([(0, "", "")])
    checks = github.get_pr_checks(runner, config=config, metadata=metadata)
    protection_calls = [c for c in runner.calls if "required_status_checks" in " ".join(c)]
    assert len(protection_calls) == 3
    assert checks.branch_protection_status == "unavailable"
    note = checks.branch_protection_note or ""
    assert "protection-final" in note
    assert all(f"attempt {n}:" in note for n in (1, 2, 3))


def test_managed_ci_event_and_timeline_failure_reasons(tmp_path, sleeps):
    from coding_review_agent_loop import managed_ci

    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 503 events-final")])
    history = managed_ci._managed_label_event_history(
        runner, config=config, pr_number=4, actor_login="bot", actor_id=7
    )
    assert history is None
    assert "events-final" in managed_ci._last_list_failure()
    assert "attempt 3:" in managed_ci._last_list_failure()

    ids = managed_ci._actor_owned_label_event_ids(
        runner, config=config, pr_number=4, actor_login="bot", actor_id=7
    )
    assert ids is None
    assert "events-final" in managed_ci._last_list_failure()

    ok = ScriptedRunner([(0, "[]", "")])
    assert managed_ci._api_list(ok, config, "repos/o/r/issues/4/events") == []
    assert managed_ci._last_list_failure() == ""


def test_fresh_authorization_event_history_failure_reaches_error(tmp_path, sleeps):
    from coding_review_agent_loop import managed_ci

    config = make_config(tmp_path)
    runner = ScriptedRunner([(1, "", "HTTP 503 fresh-final")])
    events, reason = managed_ci._api_list_detailed(runner, config, "repos/o/r/issues/4/timeline")
    assert events is None and "fresh-final" in reason and "attempt 3:" in reason
    message = managed_ci._with_reason("requires an association", reason)
    assert "requires an association" in message and "fresh-final" in message
