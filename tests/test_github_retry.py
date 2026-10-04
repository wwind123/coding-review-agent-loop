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
