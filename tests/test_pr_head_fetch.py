"""Fallback policy: one observation and one exact-SHA branch import."""

import json
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import git_transport, pr_head_fetch
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import Runner

SHA = "a" * 40


def metadata(**updates):
    return SimpleNamespace(**(dict(number=7, repo="OWNER/REPO", head_branch="topic",
                                   base_branch="main", head_sha=SHA) | updates))


def observation():
    return dict(number=7, state="open", base=dict(repo=dict(full_name="owner/repo"), ref="main"),
                head=dict(repo=dict(full_name="Owner/Repo"), ref="topic", sha=SHA))


@pytest.fixture
def policy(tmp_path, monkeypatch):
    calls, reads = [], []
    config = SimpleNamespace(repo="OWNER/REPO", gh_cmd="gh", trusted_local_origin=None)

    def fetch(destination, source, tracking, **kwargs):
        calls.append((source, kwargs))
        if len(calls) == 1:
            raise git_transport.TransportFailure("missing", label="remote ref missing")
        return kwargs["expected_sha"]

    state = SimpleNamespace(config=config, calls=calls, reads=reads, raw=observation(), code=0)

    def read(runner, args, **kwargs):
        reads.append(args)
        return SimpleNamespace(returncode=state.code, stdout=json.dumps(state.raw))

    monkeypatch.setattr(git_transport, "import_ref", fetch)
    monkeypatch.setattr(pr_head_fetch, "run_gh_read", read)
    return state


def run(policy, tmp_path, meta=None):
    return pr_head_fetch.fetch_pr_head(tmp_path, 7, config=policy.config, runner=Runner(),
                                       pr_metadata=meta or metadata())


def test_one_observation_authorizes_one_expected_sha_import(policy, tmp_path):
    assert run(policy, tmp_path) == SHA
    assert [c[0] for c in policy.calls] == ["refs/pull/7/head", "refs/heads/topic"]
    assert len(policy.reads) == 1
    assert policy.calls[1][1]["expected_sha"] == SHA
    assert "clone_url" not in policy.calls[1][1]


@pytest.mark.parametrize("key,value", [
    ("number", 8), ("state", "closed"), ("base", None), ("head", None),
    ("head.repo", None), ("base.repo", {}),
    ("head.repo.full_name", "FORK/REPO"), ("base.repo.full_name", "OTHER/REPO"),
    ("head.ref", "other"), ("head.sha", "b" * 40), ("base.ref", "other"),
])
def test_incoherent_or_fork_identity_refuses(policy, tmp_path, key, value):
    target = policy.raw
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    with pytest.raises(AgentLoopError, match="could not be verified"):
        run(policy, tmp_path)
    assert len(policy.calls) == 1


@pytest.mark.parametrize("raw,code", [(None, 0), ([], 0), ({}, 1)])
def test_bad_api_result_refuses(policy, tmp_path, raw, code):
    policy.raw, policy.code = raw, code
    with pytest.raises(AgentLoopError):
        run(policy, tmp_path)
    assert len(policy.calls) == 1


def test_api_exceptions_are_content_free(policy, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AgentLoopError("SECRET")
    monkeypatch.setattr(pr_head_fetch, "run_gh_read", fail)
    with pytest.raises(AgentLoopError) as caught:
        run(policy, tmp_path)
    assert "SECRET" not in str(caught.value)
    assert len(policy.calls) == 1


@pytest.mark.parametrize("updates", [dict(number=8), dict(repo="FORK/REPO"),
    dict(head_sha=None), dict(head_sha="abc"), dict(head_branch="bad..branch"),
    dict(head_branch="topic.lock"), dict(head_branch="topic\nsecret")])
def test_invalid_review_metadata_refuses_without_api(policy, tmp_path, updates):
    with pytest.raises(AgentLoopError):
        run(policy, tmp_path, metadata(**updates))
    assert not policy.reads and len(policy.calls) == 1


def test_optional_metadata_callers_fail_closed(policy, tmp_path):
    with pytest.raises(AgentLoopError):
        pr_head_fetch.fetch_pr_head(tmp_path, 7, config=policy.config, runner=Runner())
    assert not policy.reads


@pytest.mark.parametrize("label", ["authentication failure", "host-key failure",
    "network or timeout", "remote not found", "unclassified"])
def test_other_transport_failures_do_not_recover(policy, tmp_path, monkeypatch, label):
    def fail(*args, **kwargs):
        raise git_transport.TransportFailure("fixed", label=label)
    monkeypatch.setattr(git_transport, "import_ref", fail)
    with pytest.raises(git_transport.TransportFailure) as caught:
        run(policy, tmp_path)
    assert caught.value.label == label and not policy.reads


def test_primary_success_does_not_read_api(policy, tmp_path, monkeypatch):
    monkeypatch.setattr(git_transport, "import_ref", lambda *a, **kw: SHA)
    assert run(policy, tmp_path) == SHA
    assert not policy.reads


def test_malformed_api_json_refuses(policy, tmp_path, monkeypatch):
    monkeypatch.setattr(pr_head_fetch, "run_gh_read", lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout="not JSON SECRET"))
    with pytest.raises(AgentLoopError) as caught:
        run(policy, tmp_path)
    assert "SECRET" not in str(caught.value)
    assert len(policy.calls) == 1
