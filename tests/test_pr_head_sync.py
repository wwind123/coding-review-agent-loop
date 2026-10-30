"""Independent GitHub observations and bounded convergence; no remote mutations."""
import dataclasses
import json

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import pr_head_sync as sync
from coding_review_agent_loop.github import get_pr_review_context
from coding_review_agent_loop.runner import CommandResult

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40


def runner_at(head=A, tip=None):
    runner = FakeRunner(pr_payload={'headRefOid': head})
    runner.branch_tip = tip
    return runner


def observe(runner, tmp_path, **kwargs):
    config = make_config(tmp_path)
    return sync.get_observed_pr_review_context(runner, config=config, pr_number=77, **kwargs)


def sleeps(runner):
    return [cmd for cmd, _ in runner.commands if cmd == ['sleep', '2']]


def test_matching_heads_skip_compare_and_sleep(tmp_path):
    runner = runner_at()
    assert observe(runner, tmp_path).metadata.head_sha == A
    assert not sleeps(runner)
    assert not any('/compare/' in ' '.join(cmd) for cmd, _ in runner.commands)


def test_persistent_descendant_stops_after_three_samples(tmp_path):
    runner = runner_at(tip=B)
    with pytest.raises(sync.PrHeadObservationError, match='GitHub PR-head desync') as error:
        observe(runner, tmp_path)
    assert A in str(error.value) and B in str(error.value)
    assert len(sleeps(runner)) == 2
    assert 'preserve commit' in str(error.value)
    assert not any(cmd[:3] in (['git', 'push', '--force'], ['gh', 'pr', 'merge']) for cmd, _ in runner.commands)


@pytest.mark.parametrize('status', ['behind', 'diverged'])
def test_non_descendant_retains_pr_authority(tmp_path, status):
    runner = runner_at(tip=C)
    runner.compare_status = status
    assert observe(runner, tmp_path).metadata.head_sha == A
    assert not sleeps(runner)
    runner.pr_payload['headRefOid'] = C
    assert observe(runner, tmp_path).metadata.head_sha == C


def test_transient_lag_returns_complete_latest_context(tmp_path):
    runner = runner_at(tip=B)
    original = runner.run

    def run(args, **kwargs):
        if args == ['sleep', '2']:
            runner.pr_payload.update(headRefOid=B, title='Latest title', body='Latest body', url='https://example.test/latest')
        return original(args, **kwargs)

    runner.run = run
    context = observe(runner, tmp_path)
    assert context.metadata.head_sha == B
    assert context.metadata.title == 'Latest title'
    assert context.metadata.body == 'Latest body'
    assert context.metadata.url == 'https://example.test/latest'
    assert len(sleeps(runner)) == 1


@pytest.mark.parametrize('head', ['abc', 'z' * 40, '', None])
def test_invalid_equal_heads_never_accepted(tmp_path, head):
    with pytest.raises(sync.PrHeadObservationError, match='full hexadecimal'):
        observe(runner_at(head=head), tmp_path)


@pytest.mark.parametrize('problem', ['null-repo', 'null-ref', 'null-object', 'malformed-json', '403', '404', 'bad-compare', 'inconsistent-merge', 'projection-disagreement'])
def test_observation_failures(tmp_path, problem):
    runner = runner_at(tip=B)
    original = runner.run

    def run(args, **kwargs):
        result = original(args, **kwargs)
        if len(args) > 2 and args[:2] == ['gh', 'api']:
            endpoint = args[2]
            if '/pulls/' in endpoint:
                data = json.loads(result.stdout)
                if problem == 'null-repo': data['head']['repo'] = None
                if problem == 'null-ref': data['head']['ref'] = None
                if problem == 'projection-disagreement': data['head']['sha'] = C
                result = dataclasses.replace(result, stdout=json.dumps(data))
            if '/git/ref/' in endpoint:
                if problem == 'null-object': result = dataclasses.replace(result, stdout='{}')
                if problem == 'malformed-json': result = dataclasses.replace(result, stdout='{')
                if problem in {'403', '404'}: result = dataclasses.replace(result, returncode=1, stderr='HTTP '+problem)
            if '/compare/' in endpoint:
                data = json.loads(result.stdout)
                if problem == 'bad-compare': data['base_commit']['sha'] = C
                if problem == 'inconsistent-merge': data['merge_base_commit']['sha'] = C
                result = dataclasses.replace(result, stdout=json.dumps(data))
        return result

    runner.run = run
    with pytest.raises(sync.PrHeadObservationError):
        observe(runner, tmp_path)
    assert len(sleeps(runner)) <= 2


def test_fork_slash_ref_uses_head_repository_and_encoded_ref(tmp_path):
    runner = runner_at(tip=B)
    runner.head_repository = 'FORK/project'
    runner.pr_payload['headRefName'] = 'fix/topic'
    with pytest.raises(sync.PrHeadObservationError, match='desync'):
        observe(runner, tmp_path)
    commands = [' '.join(cmd) for cmd, _ in runner.commands]
    assert any('repos/FORK/project/git/ref/heads/fix%2Ftopic' in cmd for cmd in commands)
    assert any(f'repos/FORK/project/compare/{A}...{B}' in cmd for cmd in commands)
    assert not any('repos/OWNER/REPO/git/ref/' in cmd for cmd in commands)


@pytest.mark.parametrize('identity', ['ref', 'repository', 'base'])
def test_identity_race_is_verification_error(tmp_path, identity):
    runner = runner_at(tip=B)
    original = runner.run

    def run(args, **kwargs):
        if args == ['sleep', '2']:
            if identity == 'ref': runner.pr_payload['headRefName'] = 'another'
            if identity == 'repository': runner.head_repository = 'OTHER/project'
            if identity == 'base': runner.pr_payload['baseRefName'] = 'other-base'
        return original(args, **kwargs)

    runner.run = run
    with pytest.raises(sync.PrHeadObservationError, match='identity'):
        observe(runner, tmp_path)


@pytest.mark.parametrize('origin,issue,expected', [('ordinary', None, False), ('source-managed', 56, False), ('adopted', 56, False), (None, 56, False), ('issue-created', None, False), ('issue-created', 56, True)])
def test_origin_specific_operator_recovery(tmp_path, origin, issue, expected):
    config = dataclasses.replace(make_config(tmp_path), managed_ci=True)
    message = sync.desync_message(config, number=77, repository='FORK/project', ref='fix/topic', head=A, tip=B, origin=origin, issue_number=issue)
    assert ('--managed-ci-fresh' in message) is expected
    if expected: assert '--managed-ci-issue 56' in message
    assert 'Recheck current remote' in message
    assert 'Rerun only after' in message


def test_transient_projection_disagreement_reacquires_complete_context(tmp_path):
    runner = runner_at()
    original = runner.run
    first = True
    def run(args, **kwargs):
        nonlocal first
        result = original(args, **kwargs)
        if args == ['gh', 'api', 'repos/OWNER/REPO/pulls/77'] and first:
            data = json.loads(result.stdout)
            data['head']['sha'] = B
            first = False
            return dataclasses.replace(result, stdout=json.dumps(data))
        return result
    runner.run = run
    assert observe(runner, tmp_path).metadata.head_sha == A
    assert len(sleeps(runner)) == 1


def test_exhausted_transport_retries_are_observation_stops(tmp_path, monkeypatch):
    from coding_review_agent_loop import github_retry
    monkeypatch.setattr(github_retry, '_sleep', lambda seconds: None)
    runner = runner_at()
    original = runner.run
    reads = []
    def run(args, **kwargs):
        result = original(args, **kwargs)
        if '/git/ref/' in ' '.join(args):
            reads.append(args)
            return dataclasses.replace(result, returncode=1, stderr='HTTP 503 Service Unavailable')
        return result
    runner.run = run
    with pytest.raises(sync.PrHeadObservationError, match='verification failed'):
        observe(runner, tmp_path)
    assert len(reads) == 3
    assert not sleeps(runner)


def test_repository_identity_is_case_insensitive(tmp_path):
    runner = runner_at()
    config = dataclasses.replace(make_config(tmp_path), repo='owner/repo')
    assert sync.get_observed_pr_review_context(runner, config=config, pr_number=77).metadata.head_sha == A


@pytest.mark.parametrize('repository', ['../..', 'fork/repo?query', 'fork/repo%2Fother'])
def test_invalid_repository_paths_fail_before_ref_read(tmp_path, repository):
    runner = runner_at()
    runner.head_repository = repository
    with pytest.raises(sync.PrHeadObservationError, match='repository'):
        observe(runner, tmp_path)
    assert not any('/git/ref/' in ' '.join(cmd) for cmd, _ in runner.commands)


def test_unauthed_labeled_origin_gets_conditional_recovery_without_command(tmp_path):
    runner = runner_at(tip=B)
    original = runner.run
    def run(args, **kwargs):
        result = original(args, **kwargs)
        if args == ['gh', 'api', 'repos/OWNER/REPO/pulls/77']:
            data = json.loads(result.stdout)
            data['labels'] = [{'name': 'agent-loop-managed'}]
            result = dataclasses.replace(result, stdout=json.dumps(data))
        return result
    runner.run = run
    with pytest.raises(sync.PrHeadObservationError) as error:
        observe(runner, tmp_path)
    assert 'If this is an issue-created' in str(error.value)
    assert '--managed-ci-fresh' not in str(error.value)


def test_late_authenticated_scope_enriches_an_observation_stop(tmp_path):
    runner = runner_at(tip=B)
    with pytest.raises(sync.PrHeadObservationError) as error:
        observe(runner, tmp_path)
    config = make_config(tmp_path)
    sync.explain_pr_head_recovery(error.value, config, origin='issue-created', issue_number=56)
    assert '--managed-ci-fresh' in str(error.value)
    assert '--managed-ci-issue 56' in str(error.value)
    assert A in str(error.value) and B in str(error.value)
    sync.explain_pr_head_recovery(error.value, config, origin='source-managed', issue_number=56)
    assert '--managed-ci-fresh' not in str(error.value)


def test_recovery_enrichment_preserves_verification_errors_and_primary_stop(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    error = sync.PrHeadObservationError('verification failed')
    sync.explain_pr_head_recovery(error, config, origin='issue-created', issue_number=56)
    assert str(error) == 'verification failed'
    desync = sync.PrHeadObservationError('original stop', desync={})
    monkeypatch.setattr(sync, 'desync_message', lambda *a, **k: (_ for _ in ()).throw(ValueError('renderer unavailable')))
    sync.explain_pr_head_recovery(desync, config, origin='issue-created', issue_number=56)
    assert str(desync) == 'original stop'


@pytest.mark.parametrize('failure', ['os-error', 'command-error', 'nonzero'])
def test_failed_convergence_wait_is_an_observation_stop(tmp_path, failure):
    runner = runner_at(tip=B)
    original = runner.run
    def run(args, **kwargs):
        if args == ['sleep', '2']:
            if failure == 'os-error': raise OSError('sleep unavailable')
            if failure == 'command-error': raise sync.AgentLoopError('wait failed')
            return dataclasses.replace(original(args, **kwargs), returncode=1)
        return original(args, **kwargs)
    runner.run = run
    with pytest.raises(sync.PrHeadObservationError, match='convergence wait verification failed'):
        observe(runner, tmp_path)


@pytest.mark.parametrize('boundary', ['context', 'branch'])
def test_read_launch_failure_is_an_observation_stop(tmp_path, boundary):
    runner = runner_at()
    original = runner.run
    def run(args, **kwargs):
        if (boundary == 'context' and args[:3] == ['gh', 'pr', 'view']) or (boundary == 'branch' and '/git/ref/' in ' '.join(args)):
            raise OSError('read unavailable')
        return original(args, **kwargs)
    runner.run = run
    with pytest.raises(sync.PrHeadObservationError, match='verification failed'):
        observe(runner, tmp_path)
