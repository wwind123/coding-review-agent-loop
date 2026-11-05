"""Integration tests for admissions, evidence gates, replay and completion I/O."""
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, PullRequestMetadata, PullRequestReviewContext
from coding_review_agent_loop import post_merge_runtime as runtime
from coding_review_agent_loop.post_merge_obligations import parse_request, source_for_item, scheduled_rows
from coding_review_agent_loop.protocol import parse_structured_pr_review
from coding_review_agent_loop.pr_loop_support import _partition_unresolved_items, _render_evidence_freeze_notice, _evidence_freeze_diagnostic
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from test_post_merge_obligations import REQUEST, admission, evidence, signed, transition


def setup_run(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    runner = FakeRunner()
    run = runtime.ObligationRun(runner, config, 1)
    runner._post_merge_run = run
    writes = []
    monkeypatch.setattr(runtime, 'persist', lambda *args, **kwargs: writes.append((args, kwargs)))
    monkeypatch.setattr(runtime, 'read_api', lambda runner, config, path: {'number': int(path.rsplit('/', 1)[1]), 'state': 'open'})
    return run, writes


def metadata(head='a' * 40):
    return PullRequestMetadata(1, 'OWNER/REPO', 'Title', 'feature', 'main', head,
                              'https://github.com/OWNER/REPO/pull/1', 'Fixes #2')


def test_verified_operator_admission_persists_before_mixed_gate_release(tmp_path, monkeypatch):
    run, writes = setup_run(tmp_path, monkeypatch)
    first = evidence()
    other = replace(first, item_id='item-2', text='Other exact-head evidence.', obligation_identity='other')
    assert len(_partition_unresolved_items([first, other], current_head_sha='a' * 40)['finalization_blockers']) == 2
    assert run.operator_inputs([signed()], items=[first, other], closing=(2,), head='a' * 40, round_number=1) == ()
    assert writes
    remaining = run.apply([first, other])
    parts = _partition_unresolved_items(remaining, current_head_sha='a' * 40)
    assert parts['evidence_obligations'] == (other,)
    assert parts['coder_blockers'] == ()
    assert run.apply([first]) == []
    assert not run.visible_requirements([signed()])


def test_failed_durable_publication_cannot_release(tmp_path, monkeypatch):
    run, _ = setup_run(tmp_path, monkeypatch)
    def failed(*args, **kwargs):
        raise AgentLoopError('write failed')
    monkeypatch.setattr(runtime, 'persist', failed)
    requirements = run.operator_inputs([signed()], items=[evidence()], closing=(), head='a' * 40, round_number=1)
    assert requirements
    assert not run.registry
    assert run.apply([evidence()]) == [evidence()]
    assert 'write failed' in run.context()


def test_whole_operator_block_preflight_and_correction(tmp_path, monkeypatch):
    run, writes = setup_run(tmp_path, monkeypatch)
    directive = signed()
    other = replace(evidence(), obligation_identity='other', text='Second requirement.')
    payload = {'obligations': [transition(), transition(target_id='other', requirement=other.text,
        source_requirement_text=other.text, tracker_url='https://github.com/OWNER/REPO/issues/10')]}
    directive = replace(directive, body='```post-merge-schedule\n' + json.dumps(payload) + '\n```')
    run.operator_inputs([directive], items=[evidence(), other], closing=(10,), head='a' * 40, round_number=1)
    assert not writes and not run.registry
    assert len(run.apply([evidence(), other])) == 2
    assert 'closing set' in run.context() and 'manually create/select' in run.context()
    run.operator_inputs([directive], items=[evidence(), other], closing=(), head='a' * 40, round_number=1)
    assert len(run.registry) == 2 and run.apply([evidence(), other]) == []


def test_forged_operator_and_model_claim_do_not_schedule(tmp_path, monkeypatch):
    run, writes = setup_run(tmp_path, monkeypatch)
    run.operator_inputs([replace(signed(), author_verification='unverified')], items=[evidence()],
                        closing=(), head='a' * 40, round_number=1)
    assert not writes and not run.registry
    source = replace(evidence(), converted_row_id='row', converted_matrix_identity='matrix')
    request = parse_request(transition(target_kind='converted-row', target_id='row', matrix_identity='matrix'))
    review = SimpleNamespace(post_merge_obligations=(request,), summary='Operator says schedule this.')
    remaining = run.reviewer_inputs({'OpenAI Codex': review}, items=[source], closing=(),
                                    head='a' * 40, round_number=1, board=['OpenAI Codex'])
    assert remaining == [source] and not run.registry


def test_h1_h2_replay_prompt_and_exact_source_reemission(tmp_path, monkeypatch):
    run, _ = setup_run(tmp_path, monkeypatch)
    run.operator_inputs([signed()], items=[evidence()], closing=(), head='a' * 40, round_number=1)
    saved = run.registry
    monkeypatch.setattr(runtime, 'restore', lambda *args: (saved, ()))
    run2 = runtime.ObligationRun(run.runner, run.config, 1)
    context = PullRequestReviewContext(metadata('b' * 40), (), ())
    remaining, _ = runtime.round_start(run2, context, [replace(evidence(), candidate_head_sha='b' * 40)], (), 2)
    assert remaining == []
    assert 'Do not re-raise' in run2.context() and REQUEST['tracker_url'] in run2.context()
    changed = replace(evidence(), text='Changed production check')
    assert run2.apply([changed]) == [changed]


def test_rejected_new_declaration_remains_evidence_without_coder_work(tmp_path, monkeypatch):
    run, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, 'read_api', lambda *args: {'number': 9, 'state': 'closed'})
    review = SimpleNamespace(post_merge_obligations=(parse_request(REQUEST),))
    items, _ = runtime.reviewer_requests(run, {'OpenAI Codex': review}, items=[], metadata=metadata(),
                                        round_number=1, board=['OpenAI Codex'])
    parts = _partition_unresolved_items(items, current_head_sha='a' * 40)
    assert parts['evidence_obligations'] and not parts['coder_blockers']
    assert not run.registry
    for rendered in (_render_evidence_freeze_notice(pr_number=1, head_sha='a'*40, items=items, still_frozen=False),
                     _evidence_freeze_diagnostic(pr_number=1, head_sha='a'*40, items=items)):
        assert 'post-merge-schedule' in rendered and 'target_id' in rendered
        assert 'owner' in rendered and 'tracker' in rendered


def merged():
    return {'merged': True, 'merged_at': '2026-01-02T00:00:00Z', 'head': {'sha': 'a' * 40},
            'merge_commit_sha': 'c' * 40}


def publication_setup(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    run, writes = setup_run(tmp_path, monkeypatch)
    registry = [admission()]
    monkeypatch.setattr(runtime, 'read_api', lambda runner, config, path:
                        merged() if path.startswith('pulls/') else {'number': 9, 'state': 'open'})
    monkeypatch.setattr(github, 'get_pr_review_context', lambda *args, **kwargs:
                        PullRequestReviewContext(metadata(), (), ()))
    monkeypatch.setattr(github, 'resolve_authenticated_github_actor', lambda *args, **kwargs: ('bot', 1))
    published = {}
    def find(runner, config, number, expected):
        return published.get(expected['identity'])
    def post(*args, **kwargs):
        published[registry[0]['identity']] = 88
        return SimpleNamespace(comment_id=88)
    monkeypatch.setattr(runtime, 'find_publication', find)
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', post)
    return run, registry, published, writes


def test_publication_repeated_and_timeout_after_write_converge(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    run, registry, published, writes = publication_setup(tmp_path, monkeypatch)
    def ambiguous(*args, **kwargs):
        published[registry[0]['identity']] = 88
        raise AgentLoopError('timeout after write')
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', ambiguous)
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 0
    assert registry[0]['state'] == 'tracked' and registry[0]['receipt'] == 88
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', lambda *args, **kwargs: pytest.fail('duplicate write'))
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 0
    assert len(published) == 1


def test_publication_failure_retains_pending_and_recovery(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop import github
    run, registry, _, writes = publication_setup(tmp_path, monkeypatch)
    def failed(*args, **kwargs):
        raise AgentLoopError('unreadable tracker')
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', failed)
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 1
    assert registry[0]['state'] == 'tracking-pending' and registry[0]['receipt'] is None
    assert runtime.recovery_command(1) in capsys.readouterr().out
    assert writes


@pytest.mark.parametrize('state', [{'merged': False, 'state': 'open'}, {'merged': False, 'state': 'closed'}])
def test_completion_only_refuses_unmerged(tmp_path, monkeypatch, state):
    run, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, 'read_api', lambda *args: state)
    with pytest.raises(AgentLoopError, match='confirmed merged'):
        runtime.reconcile(run.runner, run.config, 1)


def test_completion_only_dry_run_and_closed_tracker(tmp_path, monkeypatch, capsys):
    run, registry, published, writes = publication_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, 'read_api', lambda runner, config, path:
                        merged() if path.startswith('pulls/') else {'number': 9, 'state': 'closed'})
    assert runtime.reconcile(run.runner, replace(run.config, dry_run=True), 1, registry=registry) == 0
    assert not published and not writes
    assert 'closed' in capsys.readouterr().out
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 0
    assert published


def test_recovery_option_boundary():
    runtime.validate_recovery_options(None, ['agent-loop', 'pr', '1', '--reconcile-post-merge-obligations', '--dry-run'])
    for option in ('--auto-merge', '--managed-ci', '--review-unrecorded-head', '--test-command'):
        with pytest.raises(AgentLoopError):
            runtime.validate_recovery_options(None, ['agent-loop', 'pr', '1', option])


def test_authenticated_restore_rejects_foreign_and_mutated_history(tmp_path, monkeypatch):
    import coding_review_agent_loop.pr_loop_support as support
    run, _ = setup_run(tmp_path, monkeypatch)
    record = admission()
    m = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1,
                            subject='a'*40, post_merge_registry=(record,))
    comment = IssueComment('bot', '2026-01-01T00:00:00Z', str(_attach_round_metadata('Admission', m)), 1, 42)
    monkeypatch.setattr(support, '_authenticated_pr_comments_status',
                        lambda *args, **kwargs: ((comment,), ('bot', 42), True))
    registry, _ = runtime.restore(run.runner, run.config, 1, [comment])
    assert registry == (record,)
    foreign = replace(comment, author='someone', author_id=99)
    monkeypatch.setattr(support, '_authenticated_pr_comments_status',
                        lambda *args, **kwargs: ((foreign,), ('bot', 42), True))
    assert runtime.restore(run.runner, run.config, 1, [foreign]) == ((), ())
    assert runtime.restore(run.runner, run.config, 1, []) == ((), ())


@pytest.mark.parametrize('source', ['ordinary', 'full-board', 'managed exact-head', 'deliberate recovery'])
def test_all_exact_head_merge_proofs_run_shared_completion_after_guard(tmp_path, monkeypatch, source):
    import coding_review_agent_loop.pr_loop_support as support
    run, _ = setup_run(tmp_path, monkeypatch)
    run.registry = (admission(),)
    events = []
    monkeypatch.setattr(support, 'get_pr_head_sha', lambda *args: 'a'*40)
    monkeypatch.setattr(support, 'require_recorded_base', lambda *args, **kwargs: events.append('base'))
    monkeypatch.setattr(support, 'merge_pr', lambda *args, **kwargs: events.append(('merge', kwargs['expected_head_sha'])))
    monkeypatch.setattr(runtime, 'reconcile', lambda *args, **kwargs: events.append('publish') or 0)
    support._merge_with_exact_head_proof(run.runner, config=run.config, pr_number=1,
                                        proof=support.ExactHeadCiProof(head_sha='a'*40, source=source))
    assert events == ['base', ('merge', 'a'*40), 'publish']
    events.clear()
    monkeypatch.setattr(support, 'get_pr_head_sha', lambda *args: 'b'*40)
    monkeypatch.setattr(support, 'get_observed_pr_review_context', lambda *args, **kwargs:
                        PullRequestReviewContext(metadata('b'*40), (), ()))
    with pytest.raises(AgentLoopError, match='no merge attempted'):
        support._merge_with_exact_head_proof(run.runner, config=run.config, pr_number=1,
                                            proof=support.ExactHeadCiProof(head_sha='a'*40, source=source))
    assert events == []


def test_manual_mode_reports_pending_without_publication(tmp_path, monkeypatch, capsys):
    run, _ = setup_run(tmp_path, monkeypatch)
    run.registry = (admission(),)
    runtime.manual_pending(run)
    output = capsys.readouterr().out
    assert runtime.recovery_command(1) in output and 'has not occurred' in output


def test_cli_completion_dispatch_precedes_execution_setup(tmp_path, monkeypatch):
    import coding_review_agent_loop.cli as cli
    run, _ = setup_run(tmp_path, monkeypatch)
    monkeypatch.setattr(cli, 'config_from_args', lambda *args, **kwargs: run.config)
    monkeypatch.setattr(cli, 'claim_agent_workdirs', lambda *args, **kwargs: pytest.fail('workdir claim'))
    monkeypatch.setattr(cli, 'run_pr_loop', lambda *args, **kwargs: pytest.fail('agent loop'))
    monkeypatch.setattr(runtime, 'reconcile', lambda runner, config, pr: 0 if pr == 1 else 1)
    assert cli.main(['pr', '1', '--repo', 'OWNER/REPO', '--reconcile-post-merge-obligations']) == 0


def test_real_pr_loop_declaration_stays_required_without_freeze_or_coder(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.pr_loop import run_pr_loop
    _, writes = setup_run(tmp_path, monkeypatch)
    raw = structured_pr_review()
    payload = json.loads(raw.split('\n')[0])
    payload['post_merge_obligations'] = [REQUEST]
    runner = FakeRunner(codex_outputs=[json.dumps(payload) + raw[raw.index('\n'):]])
    config = make_config(tmp_path)
    assert run_pr_loop(runner, config=config, pr_number=77) == 0
    assert len(runner._post_merge_run.registry) == 1
    assert not any('claude' in cmd[0] for cmd, _ in runner.commands)
    assert not any('Exact-head evidence freeze' in body for body in runner.comments)
    assert runtime.recovery_command(77) in capsys.readouterr().out


def test_converted_row_overlay_preserves_truth_and_residual_barrier(tmp_path, monkeypatch):
    from coding_review_agent_loop import pr_loop_support as support
    from coding_review_agent_loop.round_state import PostedRoundRecord, admit_evidence_authority_records
    from test_evidence_stall import _evidence
    run, _ = setup_run(tmp_path, monkeypatch)
    row = replace(evidence(), converted_row_id='row-a', converted_matrix_identity='a'*64)
    directive = signed(transition(target_kind='converted-row', target_id='row-a', matrix_identity='a'*64))
    run.operator_inputs([directive], items=[row], closing=(), head='a'*40, round_number=1)
    assert run.registry and run.apply([row]) == []
    canonical = _evidence({'row-a': 'stale/unverified', 'row-b': 'stale/unverified'})
    coder = PostedRoundMetadata(flow='pr', role='coder', agent='Coder', round_number=1,
        subject='b'*40, risk_test_matrix_evidence=canonical)
    admission_record = replace(coder, role='summary', post_merge_registry=run.registry)
    records = (PostedRoundRecord(0, coder, ''), PostedRoundRecord(1, admission_record, ''))
    comments = [IssueComment('bot', None, '', 1, 42), IssueComment('bot', None, '', 2, 42)]
    records = admit_evidence_authority_records(records, comments, ('bot', 42))
    plan = SimpleNamespace(risk_test_matrix_expected_row_ids=('row-a', 'row-b'), risk_test_matrix_identity='a'*64)
    result = support._head_bound_unsatisfied_rows(records, 'b'*40, plan)
    assert result.row_ids == ('row-b',)
    assert canonical['rows'][0]['status'] == 'stale/unverified'
    foreign = admit_evidence_authority_records(records, comments, ('another', 7))
    assert support._head_bound_unsatisfied_rows(foreign, 'b'*40, plan).row_ids == ('row-a', 'row-b')


def test_publication_lookup_uses_only_authenticated_records(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    run, _ = setup_run(tmp_path, monkeypatch)
    expected = runtime.tracker_record(admission(), merged())
    body = runtime.record_body(expected)
    comment = IssueComment('bot', '2026-01-02T00:00:00Z', str(body), 88, 42)
    monkeypatch.setattr(github, 'read_authenticated_protocol_comments',
                        lambda *args, **kwargs: SimpleNamespace(authored=(comment,)))
    assert runtime.find_publication(run.runner, run.config, 9, expected) == 88
    changed = dict(expected, merge_commit='d'*40)
    assert runtime.find_publication(run.runner, run.config, 9, changed) is None
    monkeypatch.setattr(github, 'read_authenticated_protocol_comments',
                        lambda *args, **kwargs: SimpleNamespace(authored=()))
    assert runtime.find_publication(run.runner, run.config, 9, expected) is None


def test_completion_refuses_first_admission_after_merge(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    run, _ = setup_run(tmp_path, monkeypatch)
    m = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1,
        subject='a'*40, post_merge_registry=(admission(),))
    comment = IssueComment('bot', '2026-01-03T00:00:00Z', str(_attach_round_metadata('Late', m)), 1, 42)
    monkeypatch.setattr(github, 'read_authenticated_protocol_comments',
                        lambda *args, **kwargs: SimpleNamespace(authored=(comment,), actor_login='bot', actor_id=42))
    with pytest.raises(AgentLoopError, match='after merge'):
        runtime.restore(run.runner, run.config, 1, [comment], admitted_before='2026-01-02T00:00:00Z')


def test_real_frozen_resume_with_admitted_source_has_no_coder_or_new_freeze(tmp_path, monkeypatch):
    from coding_review_agent_loop.pr_loop import run_pr_loop
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from test_pr_loop_evidence_freeze import _frozen_runner
    from pr_loop_test_helpers import _freeze_records
    from coding_review_agent_loop.post_merge_obligations import make_admission
    runner = _frozen_runner(tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, config=config, pr_number=77)
    freezes = _freeze_records(runner)
    item = next(i for i in freezes[-1].metadata.prior_items if i.obligation_kind == 'human-exact-head-evidence')
    request = parse_request(dict(REQUEST, requirement=item.text, source_requirement_text=item.text,
                                target_kind='evidence-item', target_id=item.obligation_identity))
    record = make_admission(request, repo=config.repo, pr=77, source=source_for_item(item),
                            provenance={'kind': 'reviewer', 'reviewers': ['OpenAI Codex']})
    monkeypatch.setattr(runtime, 'restore', lambda *args, **kwargs: ((record,), ()))
    assert run_pr_loop(runner, config=config, pr_number=77) == 0
    assert len(_freeze_records(runner)) == len(freezes)
    assert not any(cmd[0] == 'claude' for cmd, _ in runner.commands)


def test_partial_publication_retries_only_missing_work(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    from coding_review_agent_loop.post_merge_obligations import make_admission
    run, registry, published, writes = publication_setup(tmp_path, monkeypatch)
    other = replace(evidence(), text='A second production check.', obligation_identity='other')
    request = parse_request(dict(REQUEST, requirement=other.text, source_requirement_text=other.text,
                                target_kind='evidence-item', target_id='other'))
    registry.append(make_admission(request, repo=run.config.repo, pr=1, source=source_for_item(other),
                                   provenance={'kind': 'reviewer', 'reviewers': ['OpenAI Codex']}))
    calls = []
    def post(*args, **kwargs):
        text = str(kwargs['body'])
        calls.append(text)
        if other.text in text:
            raise AgentLoopError('second write unavailable')
        published[registry[0]['identity']] = 88
        return SimpleNamespace(comment_id=88)
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', post)
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 1
    assert registry[0]['state'] == 'tracked' and registry[1]['state'] == 'tracking-pending'
    calls.clear()
    def recover(*args, **kwargs):
        calls.append(str(kwargs['body']))
        published[registry[1]['identity']] = 89
        return SimpleNamespace(comment_id=89)
    monkeypatch.setattr(github, 'post_verified_trusted_issue_protocol_comment', recover)
    assert runtime.reconcile(run.runner, run.config, 1, registry=registry) == 0
    assert len(calls) == 1 and other.text in calls[0]


def test_recovery_rejects_admission_edited_after_merge(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    run, _ = setup_run(tmp_path, monkeypatch)
    m = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1,
        subject='a'*40, post_merge_registry=(admission(),))
    comment = github.AuthenticatedComment('pr#1', 1, 'bot', 42,
        '2026-01-01T00:00:00Z', '2026-01-03T00:00:00Z', str(_attach_round_metadata('Edited', m)))
    monkeypatch.setattr(github, 'read_authenticated_protocol_comments',
                        lambda *args, **kwargs: SimpleNamespace(authored=(comment,), actor_login='bot', actor_id=42))
    with pytest.raises(AgentLoopError, match='after merge'):
        runtime.restore(run.runner, run.config, 1, [], admitted_before='2026-01-02T00:00:00Z')


def test_publication_works_without_an_agent_checkout(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    from test_github import _EchoRunner, ACTOR
    config = make_config(tmp_path, create_dirs=False)
    monkeypatch.setattr(github, 'active_workdir', lambda config: pytest.fail('agent checkout dependency'))
    expected = runtime.tracker_record(admission(), merged())
    result = github.post_verified_trusted_issue_protocol_comment(_EchoRunner(stored=lambda body: body),
        config=config, issue_number=9, body=runtime.record_body(expected),
        expected_author_login=ACTOR[0], expected_author_id=ACTOR[1])
    assert result.comment_id == 41
    # The plain recovery context read also bypasses active_workdir.
    assert github.get_pr_review_context(FakeRunner(), config=config, pr_number=77,
                                      cwd=github.github_api_cwd()).metadata.number == 77


def test_tracking_prose_cannot_launder_round_authority(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    from coding_review_agent_loop.protocol_markers import scan_reserved_markers
    from coding_review_agent_loop.post_merge_obligations import make_admission
    m = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1, subject='a'*40)
    injection = str(_attach_round_metadata('Untrusted quoted audit', m))
    request = parse_request(transition(owner=injection))
    record = make_admission(request, repo='OWNER/REPO', pr=1, source=source_for_item(evidence()),
                            provenance={'kind': 'reviewer', 'reviewers': ['OpenAI Codex']})
    body = runtime.record_body(runtime.tracker_record(record, merged()))
    assert len(scan_reserved_markers(str(body))) == 1
    config = make_config(tmp_path)
    captured = []
    monkeypatch.setattr(github, 'resolve_authenticated_github_actor', lambda *args, **kwargs: ('bot', 42))
    monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment',
                        lambda *args, **kwargs: captured.append(kwargs['body']))
    runtime.persist(FakeRunner(), config, 1, 'a'*40, [record])
    assert len(scan_reserved_markers(str(captured[0]))) == 1


def test_post_merge_declaration_does_not_erase_a_blocking_code_verdict(tmp_path, monkeypatch):
    import coding_review_agent_loop.pr_loop as loop
    setup_run(tmp_path, monkeypatch)
    raw = structured_pr_review(state='blocking', summary='Fix the introduced code regression.')
    payload = json.loads(raw.split('\n')[0])
    payload['post_merge_obligations'] = [REQUEST]
    runner = FakeRunner(codex_outputs=[json.dumps(payload) + raw[raw.index('\n'):]])
    original = loop._run_validated_agent
    class CoderNeeded(Exception):
        pass
    def invoke(*args, **kwargs):
        if kwargs.get('role') == 'coder':
            raise CoderNeeded()
        return original(*args, **kwargs)
    monkeypatch.setattr(loop, '_run_validated_agent', invoke)
    with pytest.raises(CoderNeeded):
        loop.run_pr_loop(runner, config=make_config(tmp_path), pr_number=77)


def test_large_registry_uses_lossless_verified_round_transport(tmp_path, monkeypatch):
    from coding_review_agent_loop import github
    from coding_review_agent_loop.post_merge_obligations import make_admission, digest
    from coding_review_agent_loop.round_transport import prepare_round_comment
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    config = make_config(tmp_path)
    registry = []
    for number in range(12):
        request = parse_request(dict(REQUEST, requirement=f'Production check {number}', rationale=''.join(digest([number, chunk]) for chunk in range(110)), owner='o'*1000))
        source = {'target_kind': 'declaration', 'target_id': digest(['OpenAI Codex', request.requirement]),
                  'requirement': request.requirement, 'requester': 'OpenAI Codex'}
        registry.append(make_admission(request, repo=config.repo, pr=1, source=source,
            provenance={'kind': 'reviewer', 'reviewers': ['OpenAI Codex']}))
    prepared = []
    monkeypatch.setattr(github, 'resolve_authenticated_github_actor', lambda *args, **kwargs: ('bot', 42))
    monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment',
                        lambda *args, **kwargs: prepared.extend(prepare_round_comment(kwargs['body'])))
    runtime.persist(FakeRunner(), config, 1, 'a'*40, registry)
    assert len(prepared) > 1
    comments = [IssueComment('bot', None, str(body), n+1, 42) for n, body in enumerate(prepared)]
    records = _extract_round_metadata_records(comments, flow='pr')
    assert records[-1].metadata.post_merge_registry == tuple(registry)


def test_confirmed_merge_does_not_lose_known_admissions_on_empty_history(tmp_path, monkeypatch):
    run, registry, _, _ = publication_setup(tmp_path, monkeypatch)
    run.registry = tuple(registry)
    monkeypatch.setattr(runtime, 'restore', lambda *args, **kwargs: ((), ()))
    with pytest.raises(AgentLoopError, match='merged, tracking pending.*durable admission history'):
        runtime.complete_merge(run.runner, run.config, 1)
