"""Repository-explicit API reads must work before agent checkouts exist."""

import datetime
import json
from pathlib import Path

import pytest

from coding_review_agent_loop import github, github_retry
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.runner import CommandResult, Runner
from coding_review_agent_loop.workdirs import active_workdir, github_api_cwd
from agent_loop_helpers import make_config
from test_github import _body
from test_gh_rest_shim import REFUSAL


class RecordingRunner(Runner):
    def __init__(self, responses, *, write_cwd=None):
        super().__init__(dry_run=False)
        self.responses = list(responses)
        self.calls = []
        self.write_cwd = write_cwd

    def run(self, args, *, cwd, check=True, **kwargs):
        assert Path(cwd).is_dir(), f"nonexistent cwd: {cwd}"
        assert args[:2] == ['gh', 'api']  # No checkout initialization or other commands.
        mutation = '--method' in args
        assert cwd == (self.write_cwd if mutation else github_api_cwd())
        if args[2] == 'graphql':
            assert 'owner=OWNER' in args and 'name=REPO' in args
            assert '-F' in args and 'number=7' in args
        elif args[2] != 'user':
            assert any(arg.startswith('repos/OWNER/REPO/') for arg in args)
        assert check is False
        self.calls.append((list(args), cwd))
        assert self.responses, f"unexpected read: {args}"
        value = self.responses.pop(0)
        rc, out, err = value if isinstance(value, tuple) else (0, json.dumps(value), '')
        return CommandResult(list(args), cwd, out, err, rc)

    def exhausted(self):
        assert not self.responses


@pytest.fixture
def config(tmp_path):
    cfg = make_config(tmp_path, create_dirs=False)
    yield cfg
    for path in (cfg.claude_dir, cfg.codex_dir, cfg.gemini_dir, cfg.antigravity_dir):
        assert not path.exists()


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(github_retry, '_sleep', lambda _: None)


def graph_page(oids, *, total=2, head='head', next_page=False):
    return {'data': {'repository': {'pullRequest': {
        'headRefOid': head, 'commits': {'totalCount': total,
        'nodes': [{'commit': {'oid': oid, 'message': oid}} for oid in oids],
        'pageInfo': {'hasNextPage': next_page, 'endCursor': 'cursor' if next_page else None}}
    }}}}


@pytest.mark.parametrize('mode', ['graphql', 'rest', 'auto'])
@pytest.mark.parametrize('fault', [None, 'head', 'count', 'truncated'])
def test_provenance_transports(config, monkeypatch, mode, fault):
    monkeypatch.setenv('AGENT_LOOP_GH_TRANSPORT', mode)
    if mode == 'graphql':
        first = graph_page(['a'], next_page=True)
        responses = [first, graph_page([] if fault == 'truncated' else ['b'])]
        if fault != 'truncated':
            responses.append(graph_page(['a'], head='changed' if fault == 'head' else 'head',
                                        total=3 if fault == 'count' else 2))
        expected = ['a', 'b']
    else:
        first = {'head': {'sha': 'head'}, 'commits': 101}
        commits = [{'sha': str(i), 'commit': {'message': str(i)}} for i in range(101)]
        responses = [first, commits[:100], [] if fault == 'truncated' else commits[100:]]
        if fault != 'truncated':
            responses.append({'head': {'sha': 'changed' if fault == 'head' else 'head'},
                              'commits': 102 if fault == 'count' else 101})
        if mode == 'auto':
            responses.insert(0, (1, '', REFUSAL))
        expected = [str(i) for i in range(101)]
    runner = RecordingRunner(responses)
    if fault:
        with pytest.raises(AgentLoopError, match='changed|truncated'):
            github.read_pull_request_commit_metadata(runner, config=config, pr_number=7)
    else:
        result = github.read_pull_request_commit_metadata(runner, config=config, pr_number=7)
        assert [c.oid for c in result] == expected
        assert [c.message for c in result] == expected
    runner.exhausted()
    if mode == 'graphql':
        assert 'after=cursor' in runner.calls[1][0]
    else:
        assert any('page=2' in arg for args, _ in runner.calls for arg in args)


@pytest.mark.parametrize('error', ['HTTP 401: Bad credentials', 'HTTP 502: Bad Gateway'])
def test_unrelated_graphql_errors_do_not_fallback(config, monkeypatch, error):
    monkeypatch.setenv('AGENT_LOOP_GH_TRANSPORT', 'auto')
    runner = RecordingRunner([(1, '', error)] * (3 if '502' in error else 1))
    with pytest.raises(AgentLoopError, match='query failed'):
        github.read_pull_request_commit_metadata(runner, config=config, pr_number=7)
    runner.exhausted()
    assert all(args[2] == 'graphql' for args, _ in runner.calls)


# Each tuple drives one migrated CI/Actions read and its conservative failures.
CI_READS = [
    (lambda r, c: github._fetch_branch_protection_required_checks(r, config=c, base_branch='main'),
     {'contexts': ['ci'], 'checks': []}, lambda v: v[0] == 'configured', lambda v: v[0] == 'unavailable'),
    (lambda r, c: github._fetch_head_dispatch_runs(r, config=c, head_sha='head'),
     {'total_count': 0, 'workflow_runs': []}, lambda v: v == ({}, []), lambda v: v[0] == {} and bool(v[1])),
    (lambda r, c: github._list_dispatch_runs(r, config=c, query='head_sha=head'),
     {'total_count': 0, 'workflow_runs': []}, lambda v: v == ([], None), lambda v: v[0] is None),
    (lambda r, c: github._fetch_check_run_history(r, config=c, head_sha='head'),
     {'total_count': 0, 'check_runs': []}, lambda v: v == [], lambda v: v is None),
    (lambda r, c: github.read_actions_run(r, c, 42),
     {'id': 42}, lambda v: v == {'id': 42}, lambda v: v is None),
    (lambda r, c: github._run_id_for_check_suite(r, c, 5),
     {'workflow_runs': [{'id': 42}]}, lambda v: v == '42', lambda v: v is None),
    (lambda r, c: github.read_run_jobs(r, c, 42),
     {'total_count': 0, 'jobs': []}, lambda v: v == (), lambda v: v is None),
]


@pytest.mark.parametrize('call,payload,valid,unavailable', CI_READS)
@pytest.mark.parametrize('fault', [None, 'api', 'malformed'])
def test_ci_reads(config, call, payload, valid, unavailable, fault):
    response = payload if fault is None else ((1, '', 'HTTP 401: Bad credentials') if fault == 'api'
                                            else (0, '{broken', ''))
    runner = RecordingRunner([response])
    result = call(runner, config)
    assert (valid if fault is None else unavailable)(result)
    runner.exhausted()


@pytest.mark.parametrize('index', [1, 2, 3, 6])
def test_incomplete_ci_listings(config, index):
    call, payload, _, unavailable = CI_READS[index]
    runner = RecordingRunner([{**payload, 'total_count': 5}])
    assert unavailable(call(runner, config))
    runner.exhausted()


def metadata():
    return github.PullRequestMetadata(7, 'OWNER/REPO', 'title', 'topic', 'main', 'head', None)


@pytest.mark.parametrize('fault', [False, True])
def test_both_check_board_endpoints(config, fault):
    responses = [{'contexts': ['ci'], 'checks': []},
                 {'total_count': 0, 'check_runs': []}, {'total_count': 0, 'statuses': []}]
    if fault:
        responses[1:3] = [(1, '', 'HTTP 401: Bad credentials')] * 2
    runner = RecordingRunner(responses)
    result = github.get_pr_checks(runner, config=config, metadata=metadata())
    assert result.state != 'passing'
    if fault:
        assert result.check_query_status == 'unavailable'
    assert any(args[2].endswith('/check-runs') for args, _ in runner.calls)
    assert any(args[2].endswith('/status') for args, _ in runner.calls)
    runner.exhausted()


STAMP = '2026-10-01T00:00:00Z'
SINCE = datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc)


def comment(i, author=7):
    return {'id': i, 'body': 'audit text', 'user': {'id': author, 'login': 'actor'},
            'created_at': STAMP}


def issue(i):
    return {'number': i, 'body': 'body', 'title': 'title', 'created_at': STAMP,
            'user': {'id': 7, 'login': 'actor'}}


@pytest.mark.parametrize('kind', ['comments', 'issues'])
@pytest.mark.parametrize('fault', [None, 'api', 'duplicate', 'malformed'])
def test_exhaustive_listings(config, kind, fault):
    factory = comment if kind == 'comments' else issue
    later = [factory(101)]
    if fault == 'api':
        later = (1, '', 'HTTP 401: Bad credentials')
    elif fault == 'duplicate':
        later = [factory(1)]
    elif fault == 'malformed':
        later = [{}]
    responses = [[factory(i) for i in range(1, 101)], later]
    if kind == 'comments':
        responses.insert(0, {'login': 'actor', 'id': 7})
    runner = RecordingRunner(responses)
    if kind == 'comments':
        result = github.read_complete_comment_listing(runner, config=config, surface=github.comment_thread_surface('issue', 7))
        assert isinstance(result, github.UnknownCommentListing if fault else github.CompleteCommentListing)
        if not fault:
            assert len(result.comments) == 101
    else:
        if fault:
            with pytest.raises(AgentLoopError):
                github._read_creator_issue_pages(runner, config=config, creator='actor', since=SINCE)
        else:
            assert len(github._read_creator_issue_pages(runner, config=config, creator='actor', since=SINCE)) == 101
    runner.exhausted()
    assert 'page=2' in runner.calls[-1][0][2]


def test_foreign_comments_are_not_adopted(config, monkeypatch):
    monkeypatch.setattr(github, 'resolve_authenticated_github_actor', lambda *a, **k: ('actor', 7))
    runner = RecordingRunner([[comment(1), comment(2, author=9)]])
    view = github.read_authenticated_protocol_comments(runner, config=config, surface_kind='issue', number=7)
    assert [c.comment_id for c in view.authored] == [1]
    runner.exhausted()


@pytest.mark.parametrize('state', ['applied', 'unapplied', 'unreadable', 'invalid', 'foreign'])
def test_patch_reconciliation(tmp_path, state):
    config = make_config(tmp_path)
    body = _body()
    envelope = {'id': 41, 'body': str(body), 'user': {'login': 'actor', 'id': 7},
                'created_at': STAMP}
    responses = [(1, '', 'HTTP 502: Bad Gateway')]
    if state == 'applied':
        responses.append(envelope)
    elif state == 'unapplied':
        responses.extend([{**envelope, 'body': 'old'}, envelope])
    elif state == 'foreign':
        responses.extend([{**envelope, 'user': {'login': 'other', 'id': 9}}] * 2)
    else:
        responses.append((1, '', 'HTTP 401: Bad credentials') if state == 'unreadable' else {})
    runner = RecordingRunner(responses, write_cwd=active_workdir(config))
    if state in ('unreadable', 'invalid', 'foreign'):
        with pytest.raises(AgentLoopError):
            github.patch_verified_trusted_protocol_comment_observed(
                runner, config=config, comment_id=41, body=body,
                expected_author_login='actor', expected_author_id=7)
        assert sum('--method' in args for args, _ in runner.calls) == 1
    else:
        result = github.patch_verified_trusted_protocol_comment_observed(
            runner, config=config, comment_id=41, body=body,
            expected_author_login='actor', expected_author_id=7)
        assert result.comment_id == 41
        assert sum('--method' in args for args, _ in runner.calls) == (2 if state == 'unapplied' else 1)
    assert runner.calls[1][1] == github_api_cwd()
    for args, _ in runner.calls:
        if '--method' in args:
            assert args == ['gh', 'api', '--method', 'PATCH',
                            'repos/OWNER/REPO/issues/comments/41', '-f', f'body={body}']


def test_transient_retry_keeps_neutral_cwd(config):
    runner = RecordingRunner([(1, '', 'HTTP 502: Bad Gateway'), {'id': 42}])
    assert github.read_actions_run(runner, config, 42) == {'id': 42}
    assert len(runner.calls) == 2
    assert runner.calls[0] == runner.calls[1]
    runner.exhausted()


def test_dry_run_short_circuits(tmp_path):
    config = make_config(tmp_path, create_dirs=False, dry_run=True)
    runner = RecordingRunner([])
    assert github.read_pull_request_commit_metadata(runner, config=config, pr_number=7) == ()
    assert github.get_pr_checks(runner, config=config, metadata=metadata()).state == 'no_checks'
    assert runner.calls == []
    assert not config.claude_dir.exists()
