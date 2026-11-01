"""Later-attempt repair recovery through real activation and PR reconstruction."""
import copy
import dataclasses
import json

import pytest

from agent_loop_helpers import make_config, structured_coder_followup, structured_pr_review
from test_managed_ci import (
    ManualQualificationRunner, SUPPRESSING_V2_WORKFLOW, label_event,
    V2_HEAD, V2_REVISION, V2_NONCE, V2_GENERATION, valid_v2_run, v2_intent_comment,
)
from coding_review_agent_loop import managed_ci as ci, pr_loop, pr_loop_support as support
from coding_review_agent_loop.managed_ci_reconciliation import (
    observe_repair_result, reconcile_repair_checkpoint, capture_failed_run,
)
from coding_review_agent_loop.managed_ci_result import FailedManagedRun, VerifiedManagedResult
from coding_review_agent_loop.round_state import (
    QualificationCheckpoint, PostedRoundMetadata, _attach_round_metadata, _extract_round_metadata_records,
)
from coding_review_agent_loop.github import PullRequestMetadata
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.managed_pr import _source_marker

pytestmark = pytest.mark.usefixtures('private_response_tempdir')


class RecoveryRunner(ManualQualificationRunner):
    """Persist round comments in both GitHub projections, without mock CI authority."""
    def __init__(self, **kwargs):
        body = str(_source_marker(source_branch='feature/source', source_sha='source-sha'))
        super().__init__(
            workflow=SUPPRESSING_V2_WORKFLOW, base_sha=V2_REVISION,
            pr_payload={'number': 7, 'state': 'OPEN', 'title': 'Managed CI', 'body': body,
                        'url': 'https://github.com/OWNER/REPO/pull/7',
                        'headRefName': 'agent-loop/managed-643', 'headRefOid': V2_HEAD,
                        'baseRefName': 'main', 'comments': [], 'reviews': []},
            rest_pr={'draft': True, 'body': body,
                     'head': {'sha': V2_HEAD, 'ref': 'agent-loop/managed-643',
                              'repo': {'full_name': 'OWNER/REPO'}}},
            issue_events=[label_event()], pr_branch_protection_payload={'contexts': [ci.FINAL_CONTEXT]},
            **kwargs,
        )
        self.run_error = False
        self.status_error = False
        self.status_head = None
        self.run_reads = 0
        self.after_run_read = None

    def add_round(self, body, comment_id):
        self.pr_payload['comments'].append({'id': comment_id, 'databaseId': comment_id,
                                           'author': {'login': 'agent-loop'}, 'body': body})
        self.intent_comments.append({'id': comment_id, 'user': {'login': 'agent-loop', 'id': 1}, 'body': body})

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = list(args)
        endpoint = next((x for x in cmd if str(x).startswith('repos/')), '')
        if endpoint == 'repos/OWNER/REPO/actions/runs/100':
            self.run_reads += 1
            if self.after_run_read:
                self.after_run_read(self)
            if self.run_error:
                cmd, cwd_path = self._record_command(args, cwd)
                return CommandResult(cmd, cwd_path, '', 'unavailable', 1)
        if '/statuses?' in endpoint and self.status_error:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, '', 'unavailable', 1)
        if ('/statuses?' in endpoint and self.status_head is not None
            and f'/commits/{self.status_head}/' not in endpoint):
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, '[]', '', 0)
        before = len(self.pr_payload['comments'])
        result = super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)
        if cmd[:3] == ['gh', 'pr', 'comment'] and len(self.pr_payload['comments']) > before:
            comment = self.pr_payload['comments'][-1]
            comment_id = self._next_comment_id()
            comment.update(id=comment_id, databaseId=comment_id, author={'login': 'agent-loop'},
                           _rest_author_id=1)
            self.intent_comments.append({'id': comment_id,
                                         'user': {'login': 'agent-loop', 'id': 1},
                                         'body': comment['body'], 'created_at': comment.get('createdAt')})
        return result


def _status(attempt=2, state='success'):
    return {'context': ci.FINAL_CONTEXT, 'state': state,
            'description': f'nonce={V2_NONCE};run_id=100;attempt={attempt};result={state}',
            'creator': {'login': 'github-actions[bot]', 'id': 41898282},
            'target_url': 'https://github.com/OWNER/REPO/actions/runs/100'}


def recovery(tmp_path, *, legacy=False, auto_merge=True):
    config = make_config(tmp_path, managed_ci=True, managed_ci_pr_mode=True,
                         managed_ci_trusted_actor='agent-loop', auto_merge=auto_merge, max_rounds=3,
                         invocation_argv=('agent-loop', 'pr', '7', '--managed-ci',
                                          '--managed-ci-trusted-actor', 'agent-loop'))
    run = valid_v2_run(run_id=100, attempt=2, status='completed', conclusion='success')
    run.update(actor={'login': 'agent-loop', 'id': 1}, triggering_actor={'login': 'agent-loop', 'id': 1})
    runner = RecoveryRunner(workflow_runs=[run], pr_status_payload={'state': 'success', 'statuses': [_status()]})
    item = UnresolvedReviewItem(
        item_id='item-1', reviewer='GitHub managed exact-head CI', source_round=1,
        text='Managed CI failed.', status='blocking', source_status='blocking', authority='machine',
        obligation_kind='managed-exact-head-ci', lifecycle='repair_required', failed_head_sha=V2_HEAD,
        obligation_identity='managed-exact-head-ci:item-1',
    )
    ref = FailedManagedRun('OWNER/REPO', 7, V2_HEAD, 'main', item.obligation_identity, 1,
                           17, V2_NONCE, 100, 1, V2_GENERATION, V2_REVISION)
    invocation = pr_loop.resolve_invocation(config, provider='codex', role='reviewer')
    checkpoint = QualificationCheckpoint(
        'managed-exact-head-ci', item.obligation_identity, 'repair_required', V2_HEAD, None,
        base_branch='main', failed_managed_run=None if legacy else ref,
        approval_digest=support._qualification_digest(('Codex',)),
        requirements_digest=support._qualification_digest(tuple()),
        acquisition_digest=support._qualification_digest(
            {'Codex': (invocation.configured_model, invocation.resolved_effort, 'codex')}),
        scheduler_digest=support._qualification_digest(support._prior_item_ledger_signature((item,))),
        allowed_rounds=3, watch_failure_extension_used=True,
    )
    runner.add_round(_attach_round_metadata(
        'Approved.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex',
        PostedRoundMetadata(flow='pr', role='reviewer', agent='Codex', round_number=1, subject=V2_HEAD,
                            prior_items=(), state='approved', architecture_contract_version=1)), 16)
    runner.intent_comments.append(v2_intent_comment(state='completed', run_id=100, run_attempt=1))
    runner.add_round(_attach_round_metadata(
        'Failed CI persisted before repair.',
        PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=2,
                            subject=V2_HEAD, prior_items=(item,), state='blocking', phase='qualification-checkpoint',
                            qualification_checkpoint=checkpoint, architecture_contract_version=1)), 18)
    metadata = PullRequestMetadata(7, 'OWNER/REPO', 'Managed CI', 'agent-loop/managed-643', 'main', V2_HEAD, None)
    contract = ci.ManagedCiContract(protocol_version=2, base_ref='main', dispatch_ref='main', repository='OWNER/REPO',
                                    pr_number=7, trusted_actor_login='agent-loop', trusted_actor_id=1,
                                    workflow_revision=V2_REVISION, intent_generation='fresh-generation',
                                    active_label_event_id=101)
    return config, runner, metadata, contract, checkpoint, item, ref


def observe(fixture):
    config, runner, metadata, contract, checkpoint, *_ = fixture
    return observe_repair_result(runner, config=config, pr_number=7, metadata=metadata,
                                 contract=contract, checkpoint=checkpoint)


def _no_dispatch_or_agents(runner):
    assert runner.dispatch_count == 0
    assert not any(cmd[:2] == ['codex', 'exec'] or cmd[:1] == ['claude'] for cmd, _ in runner.commands)


def _capture_activation(monkeypatch):
    captured = []
    real = pr_loop.activate_managed_ci
    def activate(*args, **kwargs):
        contract = real(*args, **kwargs)
        assert contract is not None and contract.intent_generation
        captured.append((contract, dataclasses.asdict(contract)))
        return contract
    monkeypatch.setattr(pr_loop, 'activate_managed_ci', activate)
    monkeypatch.setattr(pr_loop, 'dispatch_final_qualification', lambda *a, **k: pytest.fail('new dispatch'))
    return captured


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('legacy', [False, True])
def test_resume_green_real_activation_preserves_observed_identity(tmp_path, monkeypatch, auto_merge, legacy):
    config, runner, metadata, contract, checkpoint, item, ref = recovery(tmp_path, legacy=legacy, auto_merge=auto_merge)
    captured = _capture_activation(monkeypatch)
    proofs = []
    real_merge = pr_loop._merge_with_exact_head_proof
    def merge(*args, **kwargs):
        proofs.append(kwargs['proof'])
        return real_merge(*args, **kwargs)
    monkeypatch.setattr(pr_loop, '_merge_with_exact_head_proof', merge)
    merges = []
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: merges.append(k))
    assert pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    _no_dispatch_or_agents(runner)
    live, before = captured[0]
    for key in ('nonce', 'attached_run_id', 'run_attempt', 'intent_generation', 'intent_comment_id', 'audit_comment_id'):
        assert getattr(live, key) == before[key]
    result = VerifiedManagedResult(ref, 2)
    if auto_merge:
        assert merges == [{'expected_head_sha': V2_HEAD}]
        assert proofs[0].managed_result == result
    else:
        assert merges == [] and proofs == []
        published = '\n'.join(c['body'] for c in runner.audit_comments.values())
        for value in (f'nonce={ref.nonce}', 'run_id=100', 'attempt=2', f'generation={ref.generation}', 'source_audit_record=17'):
            assert value in published
        assert 'run_id=unknown' not in published
    settlements = [r for r in _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
        if r.metadata.qualification_checkpoint and r.metadata.qualification_checkpoint.verified_managed_result]
    assert settlements[-1].metadata.qualification_checkpoint.verified_managed_result == result
    assert settlements[-1].metadata.qualification_checkpoint.watch_failure_extension_used


@pytest.mark.parametrize('mutation', [
    'nonce', 'run-id', 'pr', 'head', 'base', 'publisher', 'url', 'actor', 'missing-actor',
    'workflow', 'ref', 'attempt-zero', 'attempt-bool', 'baseline', 'excluded', 'superseded',
])
def test_reject_correlation_without_mutation(tmp_path, mutation):
    fixture = recovery(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    raw = runner.intent_comments[1]
    payload = json.loads(raw['body'].split(' ', 2)[2].rsplit(' -->', 1)[0])
    status = runner.pr_status_payload['statuses'][0]
    run = runner.workflow_runs[0]
    if mutation in {'nonce', 'run-id'}:
        status['description'] = status['description'].replace(V2_NONCE, 'wrong') if mutation == 'nonce' else status['description'].replace('run_id=100', 'run_id=101')
    elif mutation in {'pr', 'head', 'base'}:
        payload[{'pr': 'pr', 'head': 'expected_head_sha', 'base': 'base_ref'}[mutation]] = {'pr': 9, 'head': 'f'*40, 'base': 'other'}[mutation]
    elif mutation == 'publisher': status['creator'] = {'login': 'attacker', 'id': 5}
    elif mutation == 'url': status['target_url'] = 'https://evil.test/OWNER/REPO/actions/runs/100'
    elif mutation == 'actor': run['actor']['id'] = 5
    elif mutation == 'missing-actor': del run['triggering_actor']
    elif mutation == 'workflow': run['head_sha'] = 'f'*40
    elif mutation == 'ref': run['head_branch'] = 'feature'
    elif mutation.startswith('attempt-'): run['run_attempt'] = 0 if mutation == 'attempt-zero' else True
    elif mutation == 'baseline': run['run_attempt'] = 1; status.update(_status(1))
    elif mutation == 'excluded': payload['terminal_attempts'] = [{'run_id': 100, 'run_attempt': 2}]
    elif mutation == 'superseded': runner.intent_comments.append(v2_intent_comment(comment_id=19, nonce='z'*32, run_id=101, run_attempt=1))
    raw['body'] = '<!-- AGENT_MANAGED_CI_INTENT_V2 ' + json.dumps(payload) + ' -->'
    before = dataclasses.asdict(contract), copy.deepcopy(runner.intent_comments)
    assert observe(fixture) is None
    assert before == (dataclasses.asdict(contract), runner.intent_comments)
    assert runner.dispatch_count == 0
    assert not any('PATCH' in cmd for cmd, _ in runner.commands)


@pytest.mark.parametrize('mutation', ['pending', 'missing-status', 'run-api', 'status-api', 'malformed',
                                     'partial', 'missing-required', 'other-failure', 'older-green'])
def test_incomplete_evidence_remains_unresolved(tmp_path, mutation):
    fixture = recovery(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    if mutation == 'pending': runner.workflow_runs[0]['status'] = 'in_progress'
    elif mutation == 'missing-status': runner.pr_status_payload['statuses'] = []
    elif mutation == 'run-api': runner.run_error = True
    elif mutation == 'status-api': runner.status_error = True
    elif mutation == 'malformed': runner.workflow_runs[0] = {'id': 100}
    elif mutation == 'partial': runner.pr_check_runs_returncode = 1
    elif mutation == 'missing-required': runner.pr_branch_protection_payload['contexts'].append('missing')
    elif mutation == 'other-failure': runner.pr_check_runs_payload = {'check_runs': [{'name': 'tests', 'status': 'completed', 'conclusion': 'failure'}]}
    elif mutation == 'older-green': runner.workflow_runs[0].update(run_attempt=3, status='in_progress', conclusion=None)
    result, checkpoint, items = reconcile_repair_checkpoint(runner, config=config, pr_number=7, metadata=metadata,
                                                            contract=contract, checkpoint=cp, items=[item])
    assert result is None and checkpoint == cp and items == [item]
    assert runner.run_reads <= 1 and runner.dispatch_count == 0


@pytest.mark.parametrize('mutation', ['missing', 'ambiguous', 'forged'])
def test_legacy_records_fail_closed(tmp_path, mutation):
    fixture = recovery(tmp_path, legacy=True)
    runner = fixture[1]
    if mutation == 'missing': runner.intent_comments = [c for c in runner.intent_comments if c['id'] != 17]
    elif mutation == 'ambiguous':
        runner.intent_comments.append(v2_intent_comment(comment_id=175, nonce='z'*32, run_id=101, run_attempt=1, state='completed'))
        for c in runner.intent_comments:
            if c['id'] == 18: c['id'] = 180
    else: runner.intent_comments[1]['user'] = {'login': 'attacker', 'id': 5}
    assert observe(fixture) is None


@pytest.mark.parametrize('field,value', [('run_id', 0), ('failed_attempt', True), ('audit_comment_id', -1),
                                        ('nonce', ''), ('head_sha', 'bad')])
def test_invalid_failed_references_are_rejected(tmp_path, field, value):
    cp = recovery(tmp_path)[4]
    payload = cp.as_dict()
    payload['failed_managed_run'][field] = value
    assert not QualificationCheckpoint.from_mapping(payload).valid


def test_checkpoint_roundtrip_preserves_reference_and_budgets(tmp_path):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    assert QualificationCheckpoint.from_mapping(cp.as_dict()) == cp
    old = cp.as_dict(); del old['failed_managed_run']
    assert QualificationCheckpoint.from_mapping(old).valid
    result = observe((config, runner, metadata, contract, cp))
    settled = dataclasses.replace(cp, verified_managed_result=result)
    assert QualificationCheckpoint.from_mapping(settled.as_dict()) == settled
    contradictory = settled.as_dict(); contradictory['verified_managed_result']['successful_attempt'] = 1
    assert not QualificationCheckpoint.from_mapping(contradictory).valid
    updated = support._machine_obligation_checkpoint([item], current_head_sha=V2_HEAD, base_branch='main',
        allowed_rounds=3, watch_failure_extension_used=True, watch_head_extension_used=False,
        previous_checkpoint=cp, managed_contract=contract)
    assert updated.failed_managed_run == ref and updated.watch_failure_extension_used
    assert capture_failed_run(contract, cp, 1) is None  # fresh authority owns no historical run


class RepairReached(RuntimeError):
    pass


def _stop_at_agent(monkeypatch, runner):
    calls = []
    real = runner._run_with_log_locked
    def stop(args, **kwargs):
        if args[0] in {'claude', 'codex'}:
            calls.append(list(args))
            raise RepairReached('agent handoff reached')
        return real(args, **kwargs)
    monkeypatch.setattr(runner, '_run_with_log_locked', stop)
    return calls


@pytest.mark.parametrize('state', ['failure', 'in_progress', 'missing', 'unreadable'])
def test_unresolved_live_result_routes_existing_repair_with_spent_budget(tmp_path, monkeypatch, state):
    config, runner, metadata, contract, checkpoint, item, ref = recovery(tmp_path)
    if state == 'failure':
        runner.workflow_runs[0]['conclusion'] = 'failure'
        runner.pr_status_payload = {'state': 'failure', 'statuses': [_status(state='failure')]}
    elif state == 'in_progress': runner.workflow_runs[0].update(status='in_progress', conclusion=None)
    elif state == 'missing': runner.pr_status_payload['statuses'] = []
    else: runner.run_error = True
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('unqualified merge'))
    with pytest.raises(RepairReached if state == 'failure' else ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert (len(calls) == 1 and calls[0][0] == 'claude') if state == 'failure' else calls == []
    assert runner.dispatch_count == 0
    records = _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
    latest = [r.metadata.qualification_checkpoint for r in records if r.metadata.qualification_checkpoint][-1]
    assert latest.failed_managed_run == ref and latest.watch_failure_extension_used
    assert latest.verified_managed_result is None


@pytest.mark.parametrize('auto_merge', [True, False])
@pytest.mark.parametrize('interruption', ['before-write', 'after-write'])
def test_restart_around_settlement_reobserves_without_dispatch(tmp_path, monkeypatch, auto_merge, interruption):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path, auto_merge=auto_merge)
    captured = _capture_activation(monkeypatch)
    real_persist = pr_loop._persist_qualification_checkpoint
    interrupted = False
    def persist(*args, **kwargs):
        nonlocal interrupted
        result = kwargs['checkpoint'].verified_managed_result if kwargs['checkpoint'] else None
        if result and not interrupted:
            interrupted = True
            if interruption == 'after-write': real_persist(*args, **kwargs)
            raise RepairReached('settlement interrupted')
        return real_persist(*args, **kwargs)
    monkeypatch.setattr(pr_loop, '_persist_qualification_checkpoint', persist)
    merges = []
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: merges.append(k))
    with pytest.raises(RepairReached, match='settlement interrupted'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert merges == []
    assert pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    _no_dispatch_or_agents(runner)
    assert len(merges) == int(auto_merge)
    assert len(captured) == 2
    assert captured[0][0].intent_generation != captured[1][0].intent_generation
    for live, before in captured:
        for key in ('nonce', 'attached_run_id', 'run_attempt', 'intent_generation', 'intent_comment_id', 'audit_comment_id'):
            assert getattr(live, key) == before[key]
    if not auto_merge:
        record = '\n'.join(c['body'] for c in runner.audit_comments.values())
        assert f'nonce={ref.nonce} run_id=100 attempt=2 generation={ref.generation} source_audit_record=17' in record


@pytest.mark.parametrize('change', ['head', 'base', 'authorization', 'reviewer', 'plan', 'requirements'])
def test_mutable_input_drift_cannot_inherit_qualification(tmp_path, monkeypatch, change):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    if change == 'head':
        runner.pr_payload['headRefOid'] = 'f'*40
        runner.rest_pr['head']['sha'] = 'f'*40
    elif change == 'base':
        runner.pr_payload['baseRefName'] = 'other'
        runner.rest_pr['base']['ref'] = 'other'
    elif change == 'authorization': runner.advertised_actor = 'someone-else'
    else:
        field = {'reviewer': 'acquisition_digest', 'plan': 'plan_digest', 'requirements': 'requirements_digest'}[change]
        cp = dataclasses.replace(cp, **{field: 'stale'})
        for comments in (runner.intent_comments, runner.pr_payload['comments']):
            for comment in comments:
                if comment.get('id') == 18 or comment.get('databaseId') == 18:
                    comment['body'] = str(_attach_round_metadata('Stale input binding.', PostedRoundMetadata(
                        flow='pr', role='summary', agent='Orchestrator', round_number=2, subject=V2_HEAD,
                        prior_items=(item,), state='blocking', phase='qualification-checkpoint',
                        qualification_checkpoint=cp, architecture_contract_version=1)))
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(pr_loop, 'dispatch_final_qualification', lambda *a, **k: pytest.fail('new dispatch'))
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('merge with stale inputs'))
    with pytest.raises((RepairReached, ci.AgentLoopError)):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    if change in {'reviewer', 'plan', 'requirements'}:
        assert calls and calls[0][0] == 'codex'
    assert runner.dispatch_count == 0


def test_recorded_settlement_loses_live_authority_and_restores_obligation(tmp_path):
    fixture = recovery(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    result = observe(fixture)
    cp = dataclasses.replace(cp, verified_managed_result=result)
    item = dataclasses.replace(item, status='resolved', lifecycle='cleared')
    runner.workflow_runs[0].update(run_attempt=3, status='in_progress', conclusion=None)
    verified, checkpoint, items = reconcile_repair_checkpoint(runner, config=config, pr_number=7, metadata=metadata,
        contract=contract, checkpoint=cp, items=[item])
    assert verified is None and checkpoint.verified_managed_result is None
    assert items[0].status == 'blocking' and items[0].lifecycle == 'repair_required'
    assert checkpoint.watch_failure_extension_used


@pytest.mark.parametrize('arrival', ['before-handoff', 'after-unchanged-coder'])
def test_green_arriving_during_repair_prevents_another_coder_turn(tmp_path, monkeypatch, arrival):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    _capture_activation(monkeypatch)
    runner.workflow_runs[0].update(run_attempt=1, conclusion='failure')
    runner.pr_status_payload = {'state': 'failure', 'statuses': [_status(1, 'failure')]}
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs = [structured_coder_followup(remaining_items=['item-1'], addressed_items=[])]
    def green():
        runner.workflow_runs[0].update(run_attempt=2, status='completed', conclusion='success')
        runner.pr_status_payload = {'state': 'success', 'statuses': [_status()]}
    if arrival == 'before-handoff':
        runner.after_run_read = lambda r: green() if r.run_reads >= 3 else None
    else:
        real_advance = runner._maybe_advance_pr_head_for_coder_followup
        def advance(cmd):
            real_advance(cmd)
            if cmd[0] == 'claude': green()
        monkeypatch.setattr(runner, '_maybe_advance_pr_head_for_coder_followup', advance)
    merges = []
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: merges.append(k))
    assert pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    assert merges == [{'expected_head_sha': V2_HEAD}]
    coder_calls = [cmd for cmd, _ in runner.commands if cmd[:1] == ['claude']]
    assert len(coder_calls) == int(arrival == 'after-unchanged-coder')
    assert not any(cmd[:2] == ['codex', 'exec'] for cmd, _ in runner.commands)
    assert runner.dispatch_count == 0


@pytest.mark.parametrize('legacy_change', ['missing', 'forged', 'ambiguous'])
def test_legacy_unprovable_history_keeps_repair_route(tmp_path, monkeypatch, legacy_change):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path, legacy=True)
    if legacy_change == 'missing': runner.intent_comments.pop(1)
    elif legacy_change == 'forged': runner.intent_comments[1]['user'] = {'login': 'attacker', 'id': 5}
    else:
        runner.intent_comments.append(v2_intent_comment(comment_id=175, nonce='z'*32, run_id=101, run_attempt=1, state='completed'))
        for c in runner.intent_comments:
            if c['id'] == 18: c['id'] = 180
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('ambiguous merge'))
    monkeypatch.setattr(pr_loop, 'dispatch_final_qualification', lambda *a, **k: pytest.fail('ambiguous dispatch'))
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls[0][0] == 'claude' and runner.dispatch_count == 0


def test_other_blocker_is_not_cleared_by_managed_success(tmp_path, monkeypatch):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    other = dataclasses.replace(item, item_id='item-2', reviewer='GitHub PR checks',
                                obligation_kind='github-pr-checks', obligation_identity='github-pr-checks:item-2')
    cp = dataclasses.replace(cp, scheduler_digest=support._qualification_digest(
        support._prior_item_ledger_signature((item, other))))
    body = str(_attach_round_metadata('Two obligations.', PostedRoundMetadata(
        flow='pr', role='summary', agent='Orchestrator', round_number=2, subject=V2_HEAD,
        prior_items=(item, other), state='blocking', phase='qualification-checkpoint',
        qualification_checkpoint=cp, architecture_contract_version=1)))
    runner.pr_payload['comments'][-1]['body'] = body
    runner.intent_comments[-1]['body'] = body
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('remaining blocker merged'))
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls[0][0] == 'claude' and runner.dispatch_count == 0
    records = _extract_round_metadata_records(
        [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
    settled = next(r for r in records if r.metadata.qualification_checkpoint
                   and r.metadata.qualification_checkpoint.verified_managed_result)
    assert [(i.item_id, i.status) for i in settled.metadata.prior_items] == [('item-1', 'resolved'), ('item-2', 'blocking')]


@pytest.mark.parametrize('publisher', ['automatic', 'manual'])
def test_inconsistent_result_binding_rejected_before_finalization(tmp_path, publisher):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    result = VerifiedManagedResult(dataclasses.replace(ref, pr_number=9), 2)
    before = len(runner.commands)
    function = ci.prepare_v2_merge if publisher == 'automatic' else ci.publish_manual_v2_qualification
    kwargs = {} if publisher == 'automatic' else {'reviewers': ('Codex',)}
    with pytest.raises(ci.AgentLoopError, match='binding drifted'):
        function(runner, config=config, pr_number=7, expected_head_sha=V2_HEAD,
                 contract=contract, verified_result=result, **kwargs)
    assert not any(cmd[:3] == ['gh', 'pr', 'ready'] for cmd, _ in runner.commands[before:])


def test_settlement_restart_revalidates_live_attempt_before_finalization(tmp_path, monkeypatch):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    _capture_activation(monkeypatch)
    real = pr_loop._persist_qualification_checkpoint
    interrupted = False
    def persist(*args, **kwargs):
        nonlocal interrupted
        real(*args, **kwargs)
        if kwargs['checkpoint'] and kwargs['checkpoint'].verified_managed_result and not interrupted:
            interrupted = True
            raise RepairReached('after settlement')
    monkeypatch.setattr(pr_loop, '_persist_qualification_checkpoint', persist)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('stale settlement merge'))
    with pytest.raises(RepairReached, match='after settlement'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    runner.workflow_runs[0].update(run_attempt=3, status='in_progress', conclusion=None)
    calls = _stop_at_agent(monkeypatch, runner)
    with pytest.raises(ci.AgentLoopError, match='repair remains unresolved'):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == [] and runner.dispatch_count == 0


def test_architecture_revalidation_still_precedes_historical_finalization(tmp_path, monkeypatch):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    _capture_activation(monkeypatch)
    real = pr_loop._fresh_pr_qualification_snapshot
    calls = []
    def snapshot(*args, **kwargs):
        calls.append(True)
        context, requirements, plan, effective = real(*args, **kwargs)
        # A changed architecture identity is an existing live snapshot gate.
        if len(calls) == 1:
            return dataclasses.replace(context, architecture_identity_changed=True), requirements, plan, effective
        raise RepairReached('fresh snapshot required again after architecture drift')
    monkeypatch.setattr(pr_loop, '_fresh_pr_qualification_snapshot', snapshot)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('architecture drift merge'))
    _stop_at_agent(monkeypatch, runner)
    with pytest.raises(RepairReached):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls and runner.dispatch_count == 0


def test_ambiguous_merge_after_settlement_uses_confirmed_merge_recovery(tmp_path, monkeypatch):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    _capture_activation(monkeypatch)
    real_run = runner.run
    merges = []
    def run(args, **kwargs):
        cmd = list(args)
        if cmd[:3] == ['gh', 'pr', 'merge']:
            real_run(args, **kwargs)
            merges.append(cmd)
            return CommandResult(cmd, config.codex_dir, '', 'HTTP 502: Bad Gateway', 1)
        if cmd[:3] == ['gh', 'pr', 'view'] and 'state,mergedAt,headRefOid,mergeCommit' in cmd:
            return CommandResult(cmd, config.codex_dir, json.dumps({
                'state': 'MERGED', 'mergedAt': '2026-10-27T01:00:00Z',
                'headRefOid': V2_HEAD, 'mergeCommit': {'oid': 'f'*40}}), '', 0)
        return real_run(args, **kwargs)
    monkeypatch.setattr(runner, 'run', run)
    assert pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    assert len(merges) == 1 and '--match-head-commit' in merges[0]
    _no_dispatch_or_agents(runner)


def test_legacy_failure_binding_is_preserved_before_later_success(tmp_path):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path, legacy=True)
    runner.workflow_runs[0].update(conclusion='failure')
    result, bound, items = reconcile_repair_checkpoint(runner, config=config, pr_number=7, metadata=metadata,
        contract=contract, checkpoint=cp, items=[item])
    assert result is None and bound.failed_managed_run == ref and items == [item]
    assert bound.watch_failure_extension_used


def test_failed_reference_source_round_must_match_obligation(tmp_path):
    config, runner, metadata, contract, cp, item, ref = recovery(tmp_path)
    cp = dataclasses.replace(cp, failed_managed_run=dataclasses.replace(ref, source_round=2))
    result, checkpoint, items = reconcile_repair_checkpoint(runner, config=config, pr_number=7, metadata=metadata,
        contract=contract, checkpoint=cp, items=[item])
    assert result is None and items == [item] and checkpoint.verified_managed_result is None


def _baseline_race(tmp_path):
    fixture = recovery(tmp_path)
    runner = fixture[1]
    runner.workflow_runs[0].update(run_attempt=1, status='in_progress', conclusion=None)
    runner.pr_status_payload = {'state': 'failure', 'statuses': [_status(1, 'failure')]}
    return fixture


def _handoff(fixture):
    from coding_review_agent_loop.managed_ci_reconciliation import require_repair_handoff_failure
    config, runner, metadata, contract, checkpoint, *_ = fixture
    return require_repair_handoff_failure(runner, config=config, pr_number=7, metadata=metadata,
                                          contract=contract, checkpoint=checkpoint)


ATTEMPT_CASES = ['omitted', None, '1', True, 0, -1, 1, 2]


def _set_attempt(runner, attempt):
    if attempt == 'omitted':
        runner.workflow_runs[0].pop('run_attempt')
    else:
        runner.workflow_runs[0]['run_attempt'] = attempt


@pytest.mark.parametrize('attempt', ATTEMPT_CASES)
def test_baseline_race_requires_explicit_live_attempt(tmp_path, attempt):
    fixture = _baseline_race(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    _set_attempt(runner, attempt)
    view = dataclasses.replace(contract, nonce=ref.nonce, attached_run_id=100, run_attempt=1)
    snapshot = ci._refresh_v2_attached_run(runner, config=config, contract=view, strict_provenance=True)
    expected_live = attempt if type(attempt) is int and attempt > 0 else None
    assert snapshot.live_run_attempt == expected_live
    assert snapshot.run_attempt == (attempt if isinstance(attempt, int) else 1)
    assert ci.ManagedCiRunSnapshot(100, 1, 'completed', 'failure').live_run_attempt is None
    runner.run_reads = 0
    before = dataclasses.asdict(contract)
    if type(attempt) is int and attempt == 1:
        _handoff(fixture)
    else:
        with pytest.raises(ci.AgentLoopError, match='repair remains unresolved'):
            _handoff(fixture)
    # Live job evidence is probed before the existing published-status route.
    assert runner.run_reads == 2
    assert dataclasses.asdict(contract) == before
    _no_dispatch_or_agents(runner)


@pytest.mark.parametrize('attempt', ATTEMPT_CASES)
def test_baseline_race_real_resume_revalidates_each_handoff(tmp_path, monkeypatch, attempt):
    fixture = _baseline_race(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    _set_attempt(runner, attempt)
    captured = _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('unqualified merge'))
    accepted = type(attempt) is int and attempt == 1
    # Restart the real loop from the persisted checkpoint, with no cached authority.
    for evidence in ('original', 'missing-status', 'newer-pending'):
        if evidence == 'missing-status':
            runner.pr_status_payload['statuses'] = []
        elif evidence == 'newer-pending':
            runner.workflow_runs[0]['run_attempt'] = 2
            runner.pr_status_payload['statuses'] = [_status(1, 'failure')]
        permitted = accepted and evidence == 'original'
        before_calls = len(calls)
        with pytest.raises(RepairReached if permitted else ci.AgentLoopError):
            pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
        assert len(calls) == before_calls + int(permitted)
        if permitted:
            assert calls[-1][0] == 'claude'
        records = _extract_round_metadata_records(
            [type('C', (), {'body': c['body']})() for c in runner.pr_payload['comments']], flow='pr')
        latest = [r.metadata for r in records if r.metadata.qualification_checkpoint][-1]
        checkpoint = latest.qualification_checkpoint
        assert checkpoint.failed_managed_run == ref
        assert checkpoint.allowed_rounds == cp.allowed_rounds
        assert checkpoint.watch_failure_extension_used
        assert checkpoint.verified_managed_result is None
        assert any(i.obligation_identity == item.obligation_identity and i.status == 'blocking'
                   for i in latest.prior_items)
        assert runner.dispatch_count == 0
    for live, before in captured:
        assert dataclasses.asdict(live) == before


@pytest.mark.parametrize('mutation', [
    'absent', 'pending', 'success', 'nonce', 'run', 'attempt', 'head', 'publisher',
    'url', 'http', 'excluded', 'queued', 'conclusion', 'newer-failure', 'unreadable',
])
def test_baseline_race_rejects_untrusted_correlation(tmp_path, mutation):
    fixture = _baseline_race(tmp_path)
    config, runner, metadata, contract, cp, item, ref = fixture
    status = runner.pr_status_payload['statuses'][0]
    run = runner.workflow_runs[0]
    if mutation == 'absent': runner.pr_status_payload['statuses'] = []
    elif mutation in {'pending', 'success'}: status.update(_status(1, mutation))
    elif mutation == 'nonce': status['description'] = status['description'].replace(V2_NONCE, 'wrong')
    elif mutation == 'run': status['description'] = status['description'].replace('run_id=100', 'run_id=101')
    elif mutation == 'attempt': status.update(_status(2, 'failure'))
    elif mutation == 'head':
        runner.status_head = 'f'*40
        contract.expected_head_sha = runner.status_head
        # The fresh contract head cannot replace the historical target head.
    elif mutation == 'publisher': status['creator'] = {'login': 'attacker', 'id': 5}
    elif mutation == 'url': status['target_url'] += '/attempts/1'
    elif mutation == 'http': status['target_url'] = status['target_url'].replace('https:', 'http:')
    elif mutation == 'excluded':
        runner.intent_comments[1] = v2_intent_comment(state='completed', run_id=100, run_attempt=1,
                                                    terminal_attempts=[(100, 1)])
    elif mutation == 'queued': run['status'] = 'queued'
    elif mutation == 'conclusion': run['conclusion'] = 'failure'
    elif mutation == 'newer-failure': run['run_attempt'] = 2; status.update(_status(2, 'failure'))
    elif mutation == 'unreadable': runner.run_error = True
    with pytest.raises(ci.AgentLoopError): _handoff(fixture)
    _no_dispatch_or_agents(runner)
    assert cp.failed_managed_run == ref and cp.watch_failure_extension_used


@pytest.mark.parametrize('mutation', [
    'actor-missing', 'actor-login', 'actor-id', 'triggering_actor-missing',
    'triggering_actor-login', 'triggering_actor-id', 'actor-no-login', 'actor-no-id',
    'triggering_actor-no-login', 'triggering_actor-no-id', 'advertised', 'path', 'revision', 'ref',
])
def test_baseline_race_preserves_strict_provenance(tmp_path, mutation):
    fixture = _baseline_race(tmp_path)
    runner = fixture[1]
    run = runner.workflow_runs[0]
    if '-no-' in mutation:
        field, missing = mutation.split('-no-')
        del run[field][missing]
    elif '-' in mutation:
        field, change = mutation.rsplit('-', 1)
        if change == 'missing': del run[field]
        else: run[field][change] = 'other' if change == 'login' else 5
    elif mutation == 'advertised': runner.advertised_actor = 'other'
    elif mutation == 'path': run['path'] = '.github/workflows/other.yml'
    elif mutation == 'revision': run['head_sha'] = 'f'*40
    elif mutation == 'ref': run['head_branch'] = 'other'
    with pytest.raises(ci.AgentLoopError): _handoff(fixture)
    _no_dispatch_or_agents(runner)


@pytest.mark.parametrize('conclusion', ['failure', 'timed_out', 'cancelled'])
@pytest.mark.parametrize('attempt', [1, 2, 'omitted'])
@pytest.mark.parametrize('correlated', [False, True])
def test_completed_failure_handoff_compatibility(tmp_path, conclusion, attempt, correlated):
    fixture = _baseline_race(tmp_path)
    runner = fixture[1]
    runner.workflow_runs[0].update(status='completed', conclusion=conclusion)
    _set_attempt(runner, attempt)
    runner.pr_status_payload['statuses'] = [_status(attempt, 'failure')] if correlated else []
    if attempt == 2 and not correlated:
        with pytest.raises(ci.AgentLoopError): _handoff(fixture)
    else:
        _handoff(fixture)
    _no_dispatch_or_agents(runner)


@pytest.mark.parametrize('state', ['incomplete-board', 'in-progress-success'])
def test_green_without_full_qualification_never_hands_off_or_merges(tmp_path, monkeypatch, state):
    config, runner, *_ = recovery(tmp_path)
    if state == 'incomplete-board': runner.pr_branch_protection_payload['contexts'].append('missing')
    else: runner.workflow_runs[0].update(status='in_progress', conclusion=None)
    _capture_activation(monkeypatch)
    calls = _stop_at_agent(monkeypatch, runner)
    monkeypatch.setattr(support, 'merge_pr', lambda *a, **k: pytest.fail('unqualified merge'))
    with pytest.raises(ci.AgentLoopError):
        pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert calls == []
    _no_dispatch_or_agents(runner)


@pytest.mark.parametrize('state', ['failure', 'error', 'timed_out', 'cancelled'])
def test_baseline_failure_classification_and_normalized_url(tmp_path, state):
    fixture = _baseline_race(tmp_path)
    runner = fixture[1]
    status = _status(1, state)
    status['target_url'] += '/?view=logs#summary'
    runner.pr_status_payload['statuses'] = [status]
    _handoff(fixture)
    # Live job evidence is probed before the existing published-status route.
    assert runner.run_reads == 2
    assert any(f'/commits/{V2_HEAD}/statuses?' in part
               for cmd, _ in runner.commands for part in cmd)
    _no_dispatch_or_agents(runner)
