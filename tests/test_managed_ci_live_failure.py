"""Current-attempt watch and real PR-loop early-failure handoff regressions."""

from coding_review_agent_loop import managed_ci_status as _managed_ci_status
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from test_managed_ci import valid_v2_contract, v2_intent_comment, V2_HEAD, V2_NONCE
from test_managed_ci_reconciliation import (
    RecoveryRunner, recovery, _handoff, observe, _status, _stop_at_agent,
    RepairReached, _capture_activation,
)
from coding_review_agent_loop import managed_ci as ci, pr_loop
from coding_review_agent_loop.managed_ci_failure import read_attempt_jobs
from coding_review_agent_loop.runner import CommandResult

pytestmark = pytest.mark.usefixtures('private_response_tempdir')


def live_fixture(tmp_path, monkeypatch, *, auto_merge=True, allowed_rounds=3):
    fixture = recovery(tmp_path, auto_merge=auto_merge, allowed_rounds=allowed_rounds)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.workflow_runs[0].update(run_attempt=1, status='in_progress', conclusion=None)
    runner.intent_comments[1] = v2_intent_comment(state='attached', run_id=100, run_attempt=1)
    runner.pr_status_payload = {'state': 'pending', 'statuses': []}
    runner.attempt_jobs = [
        dict(id=1, run_id=100, run_attempt=1, name='tests / shard 1', status='completed',
             conclusion='failure', html_url='https://github.com/OWNER/REPO/actions/runs/100/job/1'),
        dict(id=2, run_id=100, run_attempt=1, name='tests / shard 2', status='in_progress', conclusion=None),
    ]
    runner.job_pages = None
    runner.job_error = False
    runner.after_jobs = None
    real = runner._run_locked
    def run(args, **kwargs):
        endpoint = next((str(x) for x in args if str(x).startswith('repos/')), '')
        if '/attempts/' in endpoint and '/jobs?' in endpoint:
            cmd, cwd = runner._record_command(args, kwargs['cwd'])
            body = runner.job_pages or json.dumps({'total_count': len(runner.attempt_jobs), 'jobs': runner.attempt_jobs})
            if runner.after_jobs:
                runner.after_jobs()
            return CommandResult(cmd, cwd, body, '', int(runner.job_error))
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_locked', run)
    return fixture


def watch(fixture, **kwargs):
    config, runner, metadata, _, _, _, ref = fixture
    contract = valid_v2_contract(repository=config.repo, pr_number=7, expected_head_sha=V2_HEAD,
                                 intent_comment_id=17, attached_run_id=100, run_attempt=1,
                                 created_at=1, intent_state='attached')
    outcome = ci.wait_for_final_qualification(runner, config=config, pr_number=7, metadata=metadata,
                                             contract=contract, **kwargs)
    return outcome, contract


@pytest.mark.parametrize('conclusion', ['failure', 'timed_out'])
def test_waiting_for_next_attempt_ignores_active_baseline_failure(tmp_path, monkeypatch, conclusion):
    fixture = live_fixture(tmp_path, monkeypatch)
    runner = fixture[1]
    runner.attempt_jobs[0]['conclusion'] = conclusion
    before = copy.deepcopy(runner.intent_comments)
    outcome, contract = watch(fixture, attempts=2, min_run_attempt=2)
    assert outcome.status == 'timeout'
    assert contract.run_attempt == 1 and not contract.terminal_attempts
    assert runner.intent_comments == before and runner.dispatch_count == 0
    assert not any('/attempts/1/jobs?' in str(arg) for cmd, _ in runner.commands for arg in cmd)
    assert not any('PATCH' in cmd or 'rerun' in cmd or 'cancel' in cmd for cmd, _ in runner.commands)


@pytest.mark.parametrize('auto_merge', [True, False])
def test_real_restart_advanced_head_uses_fresh_intent(tmp_path, monkeypatch, auto_merge):
    from coding_review_agent_loop.managed_ci_failure import require_live_failure
    from coding_review_agent_loop.managed_ci_reconciliation import reconcile_repair_checkpoint
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    new_head = 'f' * 40
    runner.pr_payload['headRefOid'] = new_head
    runner.rest_pr['head']['sha'] = new_head
    metadata = dataclasses.replace(metadata, head_sha=new_head)
    before = copy.deepcopy(runner.intent_comments[1])
    assert not require_live_failure(runner, config=config, pr_number=7, metadata=metadata,
                                    contract=contract, checkpoint=cp)
    result, checkpoint, items = reconcile_repair_checkpoint(
        runner, config=config, pr_number=7, metadata=metadata, contract=contract,
        checkpoint=cp, items=[item])
    assert result is None and checkpoint.verified_managed_result is None
    assert items == [item] and items[0].lifecycle == 'repair_required'

    runner.codex_outputs = [structured_pr_review(
        state='approved', summary='New head reviewed.',
        prior_item_dispositions=[{'item_id': item.item_id, 'disposition': 'resolved'}],
    )] * 2
    real = runner._run_locked
    dispatches = []
    def stop_at_dispatch(args, **kwargs):
        if any(str(arg).endswith('/dispatches') for arg in args):
            dispatches.append(list(args))
            raise RepairReached('fresh dispatch')
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_locked', stop_at_dispatch)
    with pytest.raises(RepairReached, match='fresh dispatch'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(dispatches) == 1
    assert f'inputs[expected_head_sha]={new_head}' in dispatches[0]
    nonce_arg = next(arg for arg in dispatches[0] if arg.startswith('inputs[managed_nonce]='))
    nonce = nonce_arg.split('=', 1)[1]
    assert nonce != ref.nonce
    fresh = [c for c in runner.intent_comments if nonce in c['body']]
    assert len(fresh) == 1
    assert new_head in fresh[0]['body'] and '"state":"dispatch-requested"' in fresh[0]['body']
    assert runner.intent_comments[1] == before
    records = _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
    assert not any(r.metadata.qualification_checkpoint
                   and r.metadata.qualification_checkpoint.verified_managed_result for r in records)
    assert not any(cmd[:1] == ['claude'] or 'rerun' in cmd or 'cancel' in cmd
                   or cmd[:3] == ['gh', 'pr', 'merge'] for cmd, _ in runner.commands)


@pytest.mark.parametrize('conclusion', ['failure', 'timed_out'])
@pytest.mark.parametrize('final_state', ['absent', 'pending'])
def test_watch_reports_live_failure_without_completing_intent(tmp_path, monkeypatch, conclusion, final_state):
    fixture = live_fixture(tmp_path, monkeypatch)
    runner = fixture[1]
    runner.attempt_jobs[0]['conclusion'] = conclusion
    if final_state == 'pending': runner.pr_status_payload['statuses'] = [_status(1, 'pending')]
    before = copy.deepcopy(runner.intent_comments)
    outcome, contract = watch(fixture, attempts=2)
    assert outcome.status == 'failed' and outcome.attempts_used == 1
    assert (outcome.run_id, outcome.run_attempt, outcome.workflow_status) == (100, 1, 'in_progress')
    assert outcome.final_status == final_state and outcome.workflow_conclusion is None
    assert outcome.jobs[1][1] == 'in_progress'
    assert 'attempt 1' in outcome.failure_details[0]
    assert conclusion in outcome.failure_details[1] and '/job/1' in outcome.failure_details[1]
    assert runner.intent_comments == before and contract.intent_state == 'attached'
    assert not contract.terminal_attempts and runner.dispatch_count == 0
    assert not any('PATCH' in cmd or 'rerun' in cmd or 'cancel' in cmd for cmd, _ in runner.commands)


@pytest.mark.parametrize('mutation', [
    'old-attempt', 'unknown-attempt', 'bool-attempt', 'raced-attempt', 'partial',
    'inconsistent-total', 'missing-total', 'api', 'duplicate', 'wrong-run', 'nonterminal',
    'cancelled', 'startup_failure', 'action_required', 'plumbing', 'actor', 'superseded',
    'raced-head', 'raced-base', 'raced-label', 'raced-ledger',
])
def test_ambiguous_or_excluded_live_evidence_times_out(tmp_path, monkeypatch, mutation):
    fixture = live_fixture(tmp_path, monkeypatch)
    runner = fixture[1]
    job = runner.attempt_jobs[0]
    if mutation == 'old-attempt': job['run_attempt'] = 0
    elif mutation == 'unknown-attempt': runner.workflow_runs[0].pop('run_attempt')
    elif mutation == 'bool-attempt': runner.workflow_runs[0]['run_attempt'] = True
    elif mutation == 'raced-attempt': runner.after_jobs = lambda: runner.workflow_runs[0].update(run_attempt=2)
    elif mutation == 'raced-head': runner.after_jobs = lambda: runner.rest_pr['head'].update(sha='f'*40)
    elif mutation == 'raced-base': runner.after_jobs = lambda: runner.rest_pr['base'].update(ref='other')
    elif mutation == 'raced-label': runner.after_jobs = lambda: runner.rest_pr.update(labels=[])
    elif mutation == 'raced-ledger': runner.after_jobs = lambda: runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
    elif mutation == 'partial': runner.job_pages = json.dumps({'total_count': 3, 'jobs': runner.attempt_jobs})
    elif mutation == 'inconsistent-total':
        runner.job_pages = json.dumps({'total_count': 2, 'jobs': [job]}) + json.dumps({'total_count': 1, 'jobs': [runner.attempt_jobs[1]]})
    elif mutation == 'missing-total': runner.job_pages = json.dumps({'jobs': runner.attempt_jobs})
    elif mutation == 'api': runner.job_error = True
    elif mutation == 'duplicate': runner.attempt_jobs[1]['id'] = job['id']
    elif mutation == 'wrong-run': job['run_id'] = 101
    elif mutation == 'nonterminal': job['status'] = 'in_progress'
    elif mutation in {'cancelled', 'startup_failure', 'action_required'}: job['conclusion'] = mutation
    elif mutation == 'plumbing': job['name'] = 'Publish final exact-head status'
    elif mutation == 'actor': runner.workflow_runs[0]['actor']['id'] = 5
    elif mutation == 'superseded': runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
    outcome, contract = watch(fixture, attempts=1)
    assert outcome.status == 'timeout' and not contract.terminal_attempts
    assert runner.dispatch_count == 0


def test_attempt_jobs_complete_pagination_and_legacy_reader_unchanged(tmp_path, monkeypatch):
    fixture = live_fixture(tmp_path, monkeypatch)
    config, runner, *_ = fixture
    runner.job_pages = ''.join(json.dumps({'total_count': 2, 'jobs': [j]}) for j in runner.attempt_jobs)
    assert len(read_attempt_jobs(runner, config, 100, 1)) == 2
    runner.jobs = [{'name': 'legacy', 'conclusion': 'success'}]
    assert ci.read_run_jobs(runner, config, 100) == (('legacy', 'success', ''),)


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('conclusion', ['failure', 'timed_out'])
def test_real_restart_live_failure_reaches_coder_and_revalidates(tmp_path, monkeypatch, auto_merge, conclusion):
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.attempt_jobs[0]['conclusion'] = conclusion
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    before = copy.deepcopy(runner.intent_comments[1])
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and calls[0][0] == 'claude'
    assert runner.intent_comments[1] == before and runner.dispatch_count == 0
    # Same durable checkpoint loses handoff authority when the live attempt changes.
    runner.workflow_runs[0]['run_attempt'] = 2
    with pytest.raises(ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and observe(fixture) is None


@pytest.mark.parametrize('mutation', ['jobs', 'nonce', 'superseded', 'actor', 'authorization', 'label', 'excluded'])
def test_real_restart_rejects_lost_live_authority(tmp_path, monkeypatch, mutation):
    fixture = live_fixture(tmp_path, monkeypatch)
    config, runner, metadata, contract, cp, item, ref = fixture
    if mutation == 'jobs': runner.job_error = True
    elif mutation == 'nonce': runner.intent_comments[1]['body'] = runner.intent_comments[1]['body'].replace(V2_NONCE, 'z'*32)
    elif mutation == 'superseded': runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
    elif mutation == 'actor': runner.workflow_runs[0]['actor']['id'] = 5
    elif mutation == 'authorization': runner.advertised_actor = 'other'
    elif mutation == 'label': runner.after_run_read = lambda r: r.rest_pr.update(labels=[])
    elif mutation == 'excluded': runner.intent_comments[1] = v2_intent_comment(state='completed', run_id=100, run_attempt=1, terminal_outcome='no-status', terminal_run_id=100, terminal_run_attempt=1, terminal_attempts=[(100, 1)])
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    with pytest.raises(ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == [] and runner.dispatch_count == 0 and observe(fixture) is None


@pytest.mark.parametrize('status', ['absent', 'pending', 'failure'])
def test_original_terminal_baseline_settles_without_mutating_fresh_contract(tmp_path, monkeypatch, status):
    fixture = live_fixture(tmp_path, monkeypatch)
    config, runner, metadata, contract, cp, item, ref = fixture
    before = dataclasses.asdict(contract)
    runner.workflow_runs[0].update(status='completed', conclusion='failure')
    runner.pr_status_payload['statuses'] = [] if status == 'absent' else [_status(1, status)]
    _handoff(fixture)
    assert dataclasses.asdict(contract) == before
    assert '"state":"completed"' in runner.intent_comments[1]['body']
    assert observe(fixture) is None  # same attempt cannot clear repair
    assert runner.dispatch_count == 0


@pytest.mark.parametrize('mode', ['issue', 'task', 'pr'])
@pytest.mark.parametrize('budget', [None, 1200, 4200])
def test_cli_ci_watch_budget_and_workflow_defaults(tmp_path, mode, budget):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import config_from_args
    argv = [mode, '56' if mode != 'task' else 'fix CI', '--repo', 'OWNER/REPO',
            '--codex-dir', str(tmp_path / 'codex'), '--claude-dir', str(tmp_path / 'claude'),
            '--dangerous-agent-permissions']
    if budget is not None: argv += ['--ci-timeout-seconds', str(budget)]
    config = config_from_args(build_parser().parse_args(argv), FakeRunner())
    assert config.ci_timeout_seconds == (2400 if budget is None else budget)
    assert config.ci_startup_timeout_seconds == 120 and not config.watch_pending_ci
    for file in ('managed-ci.yml', 'managed-ci-ordinary.yml'):
        text = (Path(__file__).parents[1] / '.github/workflows' / file).read_text()
        import re
        shard_default = re.search(r'shard_timeout_minutes:.*?default: (\d+)', text, re.S)
        assert shard_default is not None
        if budget is None:
            assert config.ci_timeout_seconds > int(shard_default.group(1)) * 60


@pytest.mark.parametrize('auto_merge', [True, False])
def test_real_inprocess_failure_persists_and_reaches_coder(tmp_path, monkeypatch, auto_merge):
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    # Start with approved review history, before any CI repair checkpoint.
    runner.pr_payload['comments'] = runner.pr_payload['comments'][:1]
    runner.intent_comments = runner.intent_comments[:2]
    real_activate = pr_loop.activate_managed_ci
    def activate(*args, **kwargs):
        active = real_activate(*args, **kwargs)
        assert active is not None
        return dataclasses.replace(active, expected_head_sha=ref.head_sha, repository=ref.repository,
                                   pr_number=7, nonce=ref.nonce, intent_comment_id=17,
                                   intent_generation=ref.generation, attached_run_id=100, run_attempt=1,
                                   created_at=1, intent_state='attached')
    monkeypatch.setattr(pr_loop, 'activate_managed_ci', activate)
    monkeypatch.setattr(pr_loop, 'dispatch_final_qualification', lambda *a, **k: None)
    monkeypatch.setattr(pr_loop, 'wait_for_final_qualification', ci.wait_for_final_qualification)
    real = runner._run_with_log_locked
    calls = []
    def stop(args, **kwargs):
        if args[0] == 'claude':
            calls.append(list(args))
            raise RepairReached('coder')
        return real(args, **kwargs)
    runner.codex_outputs = [structured_pr_review(state='approved', summary='Approved.')]
    monkeypatch.setattr(runner, '_run_with_log_locked', stop)
    with pytest.raises(RepairReached, match='coder'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and runner.dispatch_count == 0
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    records = _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
    checkpoints = [r.metadata.qualification_checkpoint for r in records if r.metadata.qualification_checkpoint]
    assert checkpoints[-1].failed_managed_run == ref
    assert checkpoints[-1].verified_managed_result is None
    assert '"state": "attached"' in runner.intent_comments[1]['body']


@pytest.mark.parametrize('budget', [1200, 2400, 4200])
@pytest.mark.parametrize('managed', [False, True])
def test_watch_budget_reaches_both_watchers_and_slow_success(tmp_path, monkeypatch, budget, managed):
    from coding_review_agent_loop import github
    from pr_loop_test_helpers import _watch_check_board
    from coding_review_agent_loop.ci_health import PullRequestCheck
    fixture = live_fixture(tmp_path, monkeypatch)
    config, runner, metadata, *_ = fixture
    config = dataclasses.replace(config, ci_timeout_seconds=budget, ci_poll_interval_seconds=600)
    fixture = (config, *fixture[1:])
    clock = [0]
    monkeypatch.setattr(ci.time, 'monotonic', lambda: clock[0])
    real = runner._run_locked
    def run(args, **kwargs):
        if args[0] == 'sleep': clock[0] += int(args[1])
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_locked', run)
    def board(*a, **k):
        final = PullRequestCheck(_managed_ci_status.FINAL_CONTEXT, 'status_context', 'success' if clock[0] >= 1800 else 'pending',
                                 url='https://github.com/OWNER/REPO/actions/runs/100')
        return _watch_check_board('passing', passing=(final,)) if clock[0] >= 1800 else _watch_check_board('pending', pending=(final,))
    monkeypatch.setattr(ci, 'get_pr_checks', board)
    monkeypatch.setattr(github, 'get_pr_checks', board)
    runner.attempt_jobs[0]['conclusion'] = 'success'
    runner.pr_status_payload['statuses'] = []
    def status(*a, **k):
        if clock[0] >= 1800:
            return PullRequestCheck(_managed_ci_status.FINAL_CONTEXT, 'status_context', 'success', url='https://github.com/OWNER/REPO/actions/runs/100')
        return None
    monkeypatch.setattr(_managed_ci_status, '_v2_correlated_status', status)
    if managed:
        outcome, _ = watch(fixture, deadline=budget)
    else:
        outcome = github.watch_pr_checks(runner, config, 7, metadata=metadata)
    assert outcome.status == ('timeout' if budget == 1200 else 'passed')
    assert clock[0] == (600 if budget == 1200 else 1800)
    assert runner.dispatch_count == 0


def test_early_test_failure_precedes_sibling_infrastructure_stall(tmp_path, monkeypatch):
    fixture = live_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(ci, 'is_wholly_infrastructure_blocked', lambda board: pytest.fail('test failure must win'))
    outcome, _ = watch(fixture, attempts=1)
    assert outcome.status == 'failed'


@pytest.mark.parametrize('race', ['attempt', 'head', 'publisher', 'superseded'])
def test_terminal_settlement_rechecks_baseline_after_confirmation(tmp_path, monkeypatch, race):
    fixture = live_fixture(tmp_path, monkeypatch)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.workflow_runs[0].update(status='completed', conclusion='failure')
    real = runner._run_locked
    def run(args, **kwargs):
        if args[0] == 'sleep':
            if race == 'attempt': runner.workflow_runs[0]['run_attempt'] = 2
            elif race == 'head': runner.rest_pr['head']['sha'] = 'f'*40
            elif race == 'publisher': runner.pr_status_payload['statuses'] = [_status(1, 'failure')]
            elif race == 'superseded': runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_locked', run)
    before = copy.deepcopy(runner.intent_comments[1])
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    with pytest.raises(RepairReached if race == 'publisher' else ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    if race == 'publisher':
        assert len(calls) == 1 and calls[0][0] == 'claude'
        assert '"state":"completed"' in runner.intent_comments[1]['body']
        assert '"terminal_outcome":null' in runner.intent_comments[1]['body']
    else:
        assert calls == [] and runner.intent_comments[1] == before
    assert runner.dispatch_count == 0 and observe(fixture) is None


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('baseline', ['active', 'settling', 'completed', 'completed-superseded'])
def test_real_restart_conflicted_failure_reaches_conflict_coder(tmp_path, monkeypatch, auto_merge, baseline):
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.pr_payload.update(mergeable='CONFLICTING', mergeStateStatus='DIRTY')
    if baseline != 'active':
        runner.workflow_runs[0].update(status='completed', conclusion='failure')
        runner.pr_status_payload['statuses'] = [_status(1, 'failure')]
    if baseline.startswith('completed'):
        runner.intent_comments[1] = v2_intent_comment(state='completed', run_id=100, run_attempt=1)
    if baseline == 'completed-superseded':
        runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
    before = copy.deepcopy(runner.intent_comments[1])
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    prompts = []
    stop = runner._run_with_log_locked
    def capture_prompt(args, **kwargs):
        if args[0] == 'claude':
            prompts.append(kwargs['input_text'])
        return stop(args, **kwargs)
    monkeypatch.setattr(runner, '_run_with_log_locked', capture_prompt)
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and calls[0][0] == 'claude'
    assert len(prompts) == 1
    assert 'merge conflict with its base branch `main`' in prompts[0]
    assert runner.dispatch_count == 0
    assert not any('rerun' in cmd or 'cancel' in cmd or cmd[:3] == ['gh', 'pr', 'merge']
                   for cmd, _ in runner.commands)
    if baseline == 'settling':
        assert '"state":"completed"' in runner.intent_comments[1]['body']
    else:
        assert runner.intent_comments[1] == before
    assert observe(fixture) is None  # conflict repair never clears CI qualification


def superseded_fixture(tmp_path, monkeypatch, *, auto_merge=True, completed=False):
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    runner = fixture[1]
    runner.intent_comments.append(v2_intent_comment(
        comment_id=19, nonce='z'*32, generation='replacement-generation',
        state='completed' if completed else 'attached', run_id=101, run_attempt=1))
    run = copy.deepcopy(runner.workflow_runs[0])
    run.update(id=101, name=run['name'].replace(V2_NONCE, 'z'*32))
    if isinstance(run.get('display_title'), str):
        run['display_title'] = run['display_title'].replace(V2_NONCE, 'z'*32)
    if completed:
        run.update(status='completed', conclusion='failure')
    runner.workflow_runs.append(run)
    for job in runner.attempt_jobs:
        job['run_id'] = 101
    return fixture


def replacement_checkpoints(runner):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    records = _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
    return [r.metadata.qualification_checkpoint for r in records if r.metadata.qualification_checkpoint]


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('completed', [True, False])
def test_superseded_restart_persists_replacement_before_handoff(tmp_path, monkeypatch, auto_merge, completed):
    fixture = superseded_fixture(tmp_path, monkeypatch, auto_merge=auto_merge, completed=completed)
    config, runner, metadata, contract, cp, item, ref = fixture
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    old = copy.deepcopy(runner.intent_comments[1])
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    latest = replacement_checkpoints(runner)[-1]
    expected = dataclasses.replace(ref, audit_comment_id=19, nonce='z'*32, run_id=101,
                                   generation='replacement-generation')
    assert latest.failed_managed_run == expected
    assert latest.obligation_identity == cp.obligation_identity
    assert latest.allowed_rounds == cp.allowed_rounds and latest.watch_failure_extension_used
    assert latest.verified_managed_result is None and latest.lifecycle == 'repair_required'
    assert len(calls) == 1 and calls[0][0] == 'claude'
    assert runner.intent_comments[1] == old and runner.dispatch_count == 0
    # Interrupted after publication: reconstruct the replacement, without CI discovery.
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert replacement_checkpoints(runner)[-1].failed_managed_run == expected
    assert len(calls) == 2 and runner.dispatch_count == 0


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('interruption', ['before-write', 'after-write'])
def test_superseded_replacement_publication_interrupt(tmp_path, monkeypatch, auto_merge, interruption):
    fixture = superseded_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, *_ = fixture
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    real = pr_loop._persist_qualification_checkpoint
    interrupted = False
    def persist(*args, **kwargs):
        nonlocal interrupted
        cp = kwargs['checkpoint']
        if cp and cp.failed_managed_run and cp.failed_managed_run.run_id == 101 and not interrupted:
            interrupted = True
            if interruption == 'after-write':
                real(*args, **kwargs)
            raise ci.AgentLoopError('publication interrupted')
        return real(*args, **kwargs)
    monkeypatch.setattr(pr_loop, '_persist_qualification_checkpoint', persist)
    with pytest.raises(ci.AgentLoopError, match='publication interrupted'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == [] and runner.dispatch_count == 0
    assert replacement_checkpoints(runner)[-1].failed_managed_run.run_id == (101 if interruption == 'after-write' else 100)
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and replacement_checkpoints(runner)[-1].failed_managed_run.run_id == 101
    assert runner.dispatch_count == 0


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('mutation', [
    'login', 'id', 'repository', 'pr', 'head', 'base', 'nonce', 'generation',
    'workflow', 'run', 'attempt', 'excluded', 'malformed', 'duplicate', 'authorization',
    'original-id', 'original-excluded', 'original-malformed', 'queued', 'pending',
    'unreadable', 'success', 'plumbing', 'uncorrelated', 'race-attempt', 'race-head',
    'race-base', 'race-authority', 'race-intent',
])
def test_superseded_replacement_refuses_invalid_evidence(tmp_path, monkeypatch, auto_merge, mutation):
    fixture = superseded_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    comment = runner.intent_comments[-1]
    raw = json.loads(comment['body'].split(' ', 2)[2].rsplit(' -->', 1)[0])
    if mutation in {'repository', 'pr', 'head', 'base', 'nonce', 'generation', 'workflow', 'run', 'attempt'}:
        field, value = {
            'repository': ('repository', 'OTHER/REPO'), 'pr': ('pr', 8),
            'head': ('expected_head_sha', 'f'*40), 'base': ('base_ref', 'other'),
            'nonce': ('nonce', 'x'*32), 'generation': ('generation', None),
            'workflow': ('workflow_revision', 'f'*40), 'run': ('run_id', 102),
            'attempt': ('run_attempt', 2),
        }[mutation]
        raw[field] = value
    elif mutation == 'excluded': raw['terminal_attempts'] = [{'run_id': 101, 'run_attempt': 1}]
    elif mutation == 'malformed': del raw['created_at']
    comment['body'] = '<!-- AGENT_MANAGED_CI_INTENT_V2 ' + json.dumps(raw) + ' -->'
    if mutation in {'login', 'id'}: comment['user'][mutation] = 'attacker' if mutation == 'login' else 5
    elif mutation == 'duplicate': runner.intent_comments.append(copy.deepcopy(comment))
    elif mutation == 'authorization': runner.advertised_actor = 'other'
    elif mutation == 'original-id': runner.intent_comments[1]['user']['id'] = 5
    elif mutation == 'original-excluded':
        runner.intent_comments[1] = v2_intent_comment(state='attached', run_id=100, run_attempt=1, terminal_attempts=[(100, 1)])
    elif mutation == 'original-malformed': runner.intent_comments[1]['body'] = 'untrusted prose'
    elif mutation in {'queued', 'pending'}: runner.workflow_runs[-1]['status'] = mutation
    elif mutation == 'unreadable': runner.workflow_runs.pop()
    elif mutation == 'success':
        runner.workflow_runs[-1].update(status='completed', conclusion='success')
    elif mutation == 'plumbing': runner.attempt_jobs[0]['name'] = 'Publish final exact-head status'
    elif mutation == 'uncorrelated': runner.attempt_jobs[0]['run_id'] = 100
    elif mutation == 'race-attempt': runner.after_jobs = lambda: runner.workflow_runs[-1].update(run_attempt=2)
    elif mutation == 'race-head': runner.after_jobs = lambda: runner.rest_pr['head'].update(sha='f'*40)
    elif mutation == 'race-base': runner.after_jobs = lambda: runner.rest_pr['base'].update(ref='other')
    elif mutation == 'race-authority': runner.after_jobs = lambda: runner.rest_pr.update(labels=[])
    elif mutation == 'race-intent':
        runner.after_jobs = lambda: runner.intent_comments.append(v2_intent_comment(
            comment_id=20, nonce='x'*32, generation='newer', run_id=102, run_attempt=1))
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    before = copy.deepcopy(runner.intent_comments[1])
    with pytest.raises(ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == [] and runner.dispatch_count == 0
    latest = replacement_checkpoints(runner)[-1]
    assert latest.failed_managed_run == ref and latest.watch_failure_extension_used
    assert latest.verified_managed_result is None and runner.intent_comments[1] == before


@pytest.mark.parametrize('auto_merge', [True, False])
def test_no_push_coder_discovers_existing_superseding_failure(tmp_path, monkeypatch, auto_merge):
    from agent_loop_helpers import structured_coder_followup
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge, allowed_rounds=4)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs = [structured_coder_followup(remaining_items=[item.item_id])]
    runner.codex_outputs = [structured_pr_review(
        state='approved', prior_item_dispositions=[{'item_id': item.item_id, 'disposition': 'resolved'}])]
    _capture_activation(monkeypatch)
    real_advance = runner._maybe_advance_pr_head_for_coder_followup
    def advance(cmd):
        real_advance(cmd)
        if cmd[0] != 'claude':
            return
        # An external publisher supplied R2 before the loop resumes.
        # This covers discovery, not the loop's normal qualification dispatch.
        runner.intent_comments.append(v2_intent_comment(
            comment_id=runner._next_comment_id(), nonce='z'*32, generation='replacement-generation',
            run_id=101, run_attempt=1))
        run = copy.deepcopy(runner.workflow_runs[0])
        run.update(id=101, name=run['name'].replace(V2_NONCE, 'z'*32))
        runner.workflow_runs.append(run)
        for job in runner.attempt_jobs:
            job['run_id'] = 101
    monkeypatch.setattr(runner, '_maybe_advance_pr_head_for_coder_followup', advance)
    real = runner._run_with_log_locked
    calls = []
    def stop(args, **kwargs):
        if args[0] == 'claude':
            calls.append(list(args))
            if len(calls) == 2:
                raise RepairReached('second repair')
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_with_log_locked', stop)
    old = copy.deepcopy(runner.intent_comments[1])
    with pytest.raises(RepairReached, match='second repair'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    latest = replacement_checkpoints(runner)[-1]
    assert len(calls) == 2 and runner.pr_payload['headRefOid'] == ref.head_sha
    assert latest.failed_managed_run.run_id == 101 and latest.failed_managed_run.source_round == item.source_round
    assert latest.obligation_identity == cp.obligation_identity and latest.watch_failure_extension_used
    assert latest.allowed_rounds == cp.allowed_rounds and runner.dispatch_count == 0
    assert runner.intent_comments[1] == old and latest.verified_managed_result is None


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('disposition', ['addressed', 'remaining'])
def test_no_push_coder_keeps_failed_head_out_of_qualification(tmp_path, monkeypatch, auto_merge, disposition):
    """Same-head review approval cannot turn a repair obligation into qualification."""
    from agent_loop_helpers import structured_coder_followup
    fixture = live_fixture(tmp_path, monkeypatch, auto_merge=auto_merge, allowed_rounds=4)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs = [structured_coder_followup(**{
        f'{disposition}_items': [item.item_id],
    })]
    runner.codex_outputs = [structured_pr_review(
        state='approved', prior_item_dispositions=[{'item_id': item.item_id, 'disposition': 'resolved'}])]
    # Real activation and qualification are left enabled. Only agent execution
    # is stopped, immediately before the second repair turn would consume budget.
    old = copy.deepcopy(runner.intent_comments[1])
    real = runner._run_with_log_locked
    calls = []
    def stop(args, **kwargs):
        if args[0] == 'claude':
            calls.append(list(args))
            if len(calls) == 2:
                latest = replacement_checkpoints(runner)[-1]
                assert latest.failed_managed_run == ref
                assert latest.lifecycle == 'repair_required'
                assert latest.candidate_head_sha is None
                assert latest.obligation_identity == cp.obligation_identity
                assert latest.allowed_rounds == cp.allowed_rounds
                assert latest.watch_failure_extension_used == cp.watch_failure_extension_used
                assert latest.watch_head_extension_used == cp.watch_head_extension_used
                assert latest.verified_managed_result is None
                assert runner.dispatch_count == 0
                raise RepairReached('same-head repair still required')
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_with_log_locked', stop)
    with pytest.raises(RepairReached, match='same-head repair still required'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 2 and runner.pr_payload['headRefOid'] == ref.head_sha
    assert len(runner.workflow_runs) == 1 and runner.intent_comments[1] == old
    assert not any('rerun' in cmd or 'cancel' in cmd or cmd[:3] == ['gh', 'pr', 'merge']
                   for cmd, _ in runner.commands)


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('race', ['attempt', 'head', 'base', 'authority', 'intent'])
def test_replacement_handoff_revalidates_after_publication(tmp_path, monkeypatch, auto_merge, race):
    fixture = superseded_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, *_ = fixture
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    real = pr_loop._persist_qualification_checkpoint
    def persist(*args, **kwargs):
        real(*args, **kwargs)
        cp = kwargs['checkpoint']
        if not cp or not cp.failed_managed_run or cp.failed_managed_run.run_id != 101:
            return
        if race == 'attempt': runner.workflow_runs[-1]['run_attempt'] = 2
        elif race == 'head': runner.rest_pr['head']['sha'] = 'f'*40
        elif race == 'base': runner.rest_pr['base']['ref'] = 'other'
        elif race == 'authority': runner.rest_pr['labels'] = []
        else: runner.intent_comments.append(v2_intent_comment(
            comment_id=runner._next_comment_id(), nonce='x'*32, generation='newest', run_id=102, run_attempt=1))
    monkeypatch.setattr(pr_loop, '_persist_qualification_checkpoint', persist)
    with pytest.raises(ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == [] and runner.dispatch_count == 0
    latest = replacement_checkpoints(runner)[-1]
    assert latest.failed_managed_run.run_id == 101 and latest.watch_failure_extension_used
    assert latest.verified_managed_result is None


@pytest.mark.parametrize('auto_merge', [True, False])
def test_completed_history_is_not_replaced_by_newer_failure(tmp_path, monkeypatch, auto_merge):
    fixture = superseded_fixture(tmp_path, monkeypatch, auto_merge=auto_merge)
    config, runner, metadata, contract, cp, item, ref = fixture
    runner.intent_comments[1] = v2_intent_comment(state='completed', run_id=100, run_attempt=1)
    runner.workflow_runs[0].update(status='completed', conclusion='failure')
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert len(calls) == 1 and replacement_checkpoints(runner)[-1].failed_managed_run == ref
    assert runner.dispatch_count == 0
