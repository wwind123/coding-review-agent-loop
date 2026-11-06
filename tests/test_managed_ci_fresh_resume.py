"""Opening-tuple recovery rejects unavailable state without inventing tampering."""

from coding_review_agent_loop import managed_ci_authorization as _managed_ci_authorization
import copy
import json

import pytest

from agent_loop_helpers import make_config
from test_managed_ci import V2ManagedRunner, _ready_issue_metadata, label_event
from coding_review_agent_loop import managed_ci
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import CommandResult

ENDPOINT = "repos/OWNER/REPO/pulls/7"


class TupleRunner(V2ManagedRunner):
    def __init__(self, responses=()):
        super().__init__(rest_pr={"body": "Fixes #643"}, issue_events=[label_event()])
        self.responses = list(responses)
        self.reads = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        if list(args) == ["gh", "api", ENDPOINT]:
            self.reads += 1
            if self.responses:
                response = self.responses.pop(0)
                if response is not None:
                    cmd, path = self._record_command(args, cwd)
                    code, stdout, stderr = response
                    return CommandResult(cmd, path, stdout, stderr, code)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def recover(runner, tmp_path):
    return _managed_ci_authorization.recover_issue_created_handoff(
        runner, config=make_config(tmp_path, managed_ci=True, managed_ci_pr_mode=True,
                                   managed_ci_trusted_actor="agent-loop"),
        pr_number=7, metadata=_ready_issue_metadata(), issue_number=643,
    )


def assert_no_writes(runner):
    assert not any("--method" in cmd or cmd[:3] in (["gh", "pr", "ready"],
                      ["gh", "pr", "merge"], ["gh", "workflow", "run"])
                   for cmd, _ in runner.commands)
    assert not any(cmd[:3] in (["gh", "pr", "comment"], ["gh", "pr", "edit"])
                   for cmd, _ in runner.commands)
    assert runner.dispatch_count == 0


@pytest.mark.parametrize("phase,index", [("lifecycle-classification", 0),
    ("first-validation", 1), ("second-validation", 2)])
@pytest.mark.parametrize("response,kind", [
    ((1, "", "HTTP 403 forbidden SECRET"), "unavailable"),
    ((1, "", "HTTP 503 Service Unavailable SECRET"), "unavailable"),
    ((0, "not json SECRET", ""), "malformed"),
    ((0, "[]", ""), "malformed"),
    ((0, "{}", ""), "malformed"),
])
def test_recovery_read_failure_is_not_tuple_mismatch(tmp_path, monkeypatch, phase, index, response, kind):
    monkeypatch.setattr("coding_review_agent_loop.github_retry._sleep", lambda _: None)
    runner = TupleRunner([None] * index + [response] * 3)
    with pytest.raises(AgentLoopError) as error:
        recover(runner, tmp_path)
    message = str(error.value)
    assert phase in message and ENDPOINT in message and kind in message
    assert "SECRET" not in message
    assert "mixed lifecycle" not in message and "missing or changed" not in message
    assert runner.reads <= index + 3
    assert_no_writes(runner)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_recovery_transient_read_retries_then_succeeds(tmp_path, monkeypatch, index):
    monkeypatch.setattr("coding_review_agent_loop.github_retry._sleep", lambda _: None)
    runner = TupleRunner([None] * index + [(1, "", "HTTP 503 Service Unavailable"), None])
    assert recover(runner, tmp_path).head_sha == _ready_issue_metadata().head_sha
    assert runner.reads == 4
    assert_no_writes(runner)


@pytest.mark.parametrize("field,value", [("draft", None), ("draft", 1),
    ("labels", None), ("labels", {}), ("labels", ["agent-loop-managed"]),
    ("labels", [{"name": 1}])])
def test_malformed_lifecycle_stops_before_tuple(tmp_path, field, value):
    runner = TupleRunner()
    runner.rest_pr[field] = value
    with pytest.raises(AgentLoopError, match="lifecycle-classification.*malformed"):
        recover(runner, tmp_path)
    assert runner.reads == 1
    assert_no_writes(runner)


MUTATIONS = [
    ("state", "closed"), ("draft", False), ("labels", []),
    ("head.repo.full_name", "OTHER/REPO"), ("head.ref", "other"),
    ("base.ref", "other"), ("user.login", "other"), ("user.id", 2),
    ("head.sha", "other"), ("body", "Fixes #643\nSECRET body nonce"),
]


def mutate(payload, field, value):
    parts = field.split(".")
    for part in parts[:-1]:
        payload = payload[part]
    payload[parts[-1]] = value


@pytest.mark.parametrize("field,value", MUTATIONS)
@pytest.mark.parametrize("index,phase", [(1, "first-validation"), (2, "second-validation")])
def test_tuple_mismatch_identifies_field_and_read(tmp_path, field, value, index, phase):
    runner = TupleRunner()
    changed = copy.deepcopy(runner.rest_pr)
    mutate(changed, field, value)
    runner.responses = [None] * index + [(0, json.dumps(changed), "")]
    with pytest.raises(AgentLoopError) as error:
        recover(runner, tmp_path)
    message = str(error.value)
    assert field in message and phase in message
    assert "expected" in message and "REST" in message
    assert "SECRET" not in message
    assert_no_writes(runner)


@pytest.mark.parametrize("field,value", [("head.repo.full_name", 1), ("user.login", []),
    ("user.id", None), ("user.id", True), ("head", []), ("body", None), ("state", []),
    ("head.ref", 1), ("head.sha", None), ("base.ref", {}),
    ("draft", None), ("labels", None)])
@pytest.mark.parametrize("index", [1, 2])
def test_malformed_tuple_is_controlled_refusal(tmp_path, field, value, index):
    runner = TupleRunner()
    changed = copy.deepcopy(runner.rest_pr)
    mutate(changed, field, value)
    runner.responses = [None] * index + [(0, json.dumps(changed), "")]
    with pytest.raises(AgentLoopError, match="malformed"):
        recover(runner, tmp_path)
    assert_no_writes(runner)


@pytest.mark.parametrize("entry", ["issue", "direct"])
@pytest.mark.parametrize("protection", ["strict", "waiver"])
def test_interrupted_creation_resumes_in_new_process(tmp_path, entry, protection):
    import os
    from pathlib import Path
    import subprocess
    import sys

    fixture = Path(__file__).parent / "fixtures/managed_ci/fresh_resume_process.py"
    state = tmp_path / "github.json"
    results = []
    env = dict(os.environ, TMPDIR=str(tmp_path))
    for mode, route in [("create", "issue"), ("resume", entry)]:
        result = subprocess.run([sys.executable, str(fixture), mode, str(state),
                                 str(tmp_path), route, protection],
                                capture_output=True, text=True, timeout=90, env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(json.loads(result.stdout.split("PROCESS_RESULT=")[-1]))
    assert all(result["boundary"] for result in results)
    assert results[0]["pid"] != results[1]["pid"]
    assert results[0]["agent_calls"] == 3
    assert results[1]["agent_calls"] == 0
    assert results[0]["plan_hash"] and results[0]["plan_hash"] == results[1]["plan_hash"]
    assert not any(cmd[:3] in (["gh", "pr", "create"], ["gh", "pr", "ready"],
                               ["gh", "workflow", "run"], ["gh", "pr", "merge"])
                   for cmd in results[1]["commands"])
    durable = json.loads(state.read_text())
    assert durable["rest_pr"]["draft"] is True
    assert durable["rest_pr"]["labels"] == [{"name": "agent-loop-managed"}]
    assert durable["authorization_comments"]


@pytest.mark.parametrize("draft,lifecycle", [(True, "draft-unlabeled-reentry"),
                                            (False, "ready-unlabeled-reentry")])
def test_empty_labels_are_a_valid_lifecycle_observation(tmp_path, draft, lifecycle):
    runner = TupleRunner()
    runner.rest_pr.update(draft=draft, labels=[])
    assert recover(runner, tmp_path).lifecycle == lifecycle
    assert runner.reads == 3
    assert_no_writes(runner)


def test_tuple_reports_all_fields_without_echoing_api_strings(tmp_path):
    runner = TupleRunner()
    changed = copy.deepcopy(runner.rest_pr)
    changed["head"]["ref"] = "SECRET credential"
    changed["body"] += "\r\nSECRET nonce"
    runner.responses = [None, (0, json.dumps(changed), "")]
    with pytest.raises(AgentLoopError) as error:
        recover(runner, tmp_path)
    assert "head.ref" in str(error.value) and "body" in str(error.value)
    assert "SECRET" not in str(error.value)
    assert "sha256=" in str(error.value)
    assert_no_writes(runner)


def test_large_actor_id_diagnostic_is_bounded(tmp_path):
    runner = TupleRunner()
    changed = copy.deepcopy(runner.rest_pr)
    changed["user"]["id"] = 10 ** 1000
    runner.responses = [None, (0, json.dumps(changed), "")]
    with pytest.raises(AgentLoopError) as error:
        recover(runner, tmp_path)
    assert "user.id" in str(error.value)
    assert len(str(error.value)) < 1000
    assert_no_writes(runner)
