"""Transport admission through live acquisition, recovery and qualification."""
import base64
from dataclasses import replace
import os
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from coding_review_agent_loop import github, pr_loop, pr_loop_support as support
from coding_review_agent_loop.code_approval import _latest_pr_approved_reviews_for_head
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.native_evidence import native_history
from coding_review_agent_loop.pr_round_ingestion import (
    admit_pr_comments, extract_admitted_round_records,
)
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from coding_review_agent_loop.round_transport import encode_mapping, prepare_round_comment, ROUND_TRANSPORT_SIDECAR_RE

HEAD = '6367c48dd193d56ea7b0baad25b19455e529f5ee'
STAMP = '2026-01-01T00:00:00Z'
BAD = '<!-- AGENT_LOOP_META: invalid -->'


def raw(body, *, foreign=False):
    return {'author': {'login': 'outsider' if foreign else 'bot'},
            '_rest_author_id': 987654 if foreign else 424242,
            'createdAt': STAMP, 'body': str(body)}


def anchor(**kwargs):
    values = dict(flow='pr', role='reviewer', agent='Codex', round_number=3,
                  subject=HEAD, state='approved')
    values.update(kwargs)
    return _attach_round_metadata('Owned history', PostedRoundMetadata(**values))


def observation(tmp_path, bodies, **config_values):
    runner = FakeRunner(authenticated_actor=('bot', 424242), pr_payload={'comments': bodies})
    config = make_config(tmp_path, **config_values)
    context = github.get_pr_review_context(runner, config=config, pr_number=77)
    return runner, config, context


def corruptions():
    return [BAD, '<!-- AGENT_LOOP_META: @@@ -->',
            '<!-- AGENT_LOOP_META: ' + encode_mapping({'flow': 'pr'}) + ' -->',
            '<!-- AGENT_LOOP_META: ' + encode_mapping({'flow': 'pr', 'prior_items': {
                '$round_transport_spill': 'absent', 'parts': 1, 'sha256': '0' * 64, 'spill': '0' * 64}}) + ' -->']


@pytest.mark.parametrize('bad', corruptions())
def test_foreign_anchor_is_excluded_before_decode_with_stable_raw_indices(tmp_path, bad):
    owned = str(anchor())
    runner, config, context = observation(tmp_path, [raw('Human instruction'), raw(owned), raw(bad, foreign=True)])
    records = extract_admitted_round_records(context.comments)
    assert len(records) == 1 and records[0].index == 1
    assert records[0].metadata.round_number == 3
    assert context.comments[2].body == bad and context.comments[0].body == 'Human instruction'
    resumed = support._resume_pr_round_admitted(runner, config=config, pr_number=77,
        comments=context.comments, head_sha=HEAD, configured_reviewers=config.reviewer)
    assert resumed is not None and resumed.round_number == 3
    assert len(resumed.completed_reviews) == 1


@pytest.mark.parametrize('bad', corruptions()[:3])
def test_owned_corruption_stops_real_pr_startup_before_agent_launch(tmp_path, bad):
    runner = FakeRunner(authenticated_actor=('bot', 424242), pr_payload={'comments': [raw(bad)]}, codex_outputs=[structured_pr_review()])
    with pytest.raises(AgentLoopError, match='[Rr]ound metadata|round metadata'):
        pr_loop.run_pr_loop(runner, pr_number=77, config=make_config(tmp_path))
    assert runner.codex_outputs == [structured_pr_review()]
    assert not any(cmd[0] in {'codex', 'claude', 'gemini'} for cmd, _ in runner.commands)
    assert not any(cmd[:3] == ['gh', 'pr', 'merge'] for cmd, _ in runner.commands)


def spilled():
    text = base64.b64encode(os.urandom(70000)).decode()
    parts = prepare_round_comment(anchor(raw_structured_coder_response=text))
    assert len(parts) > 1
    return list(map(str, parts))


@pytest.mark.parametrize('foreign_first', [False, True])
def test_owned_sidecars_ignore_foreign_conflicts_and_foreign_only_completion(tmp_path, foreign_first):
    parts = spilled()
    # A copied chunk with one payload byte changed must not poison owned chunks.
    match = ROUND_TRANSPORT_SIDECAR_RE.search(parts[0])
    payload = json.loads(base64.urlsafe_b64decode(match.group('payload')))
    payload['data'] = 'tampered'
    poison = parts[0].replace(match.group('payload'), base64.urlsafe_b64encode(json.dumps(payload).encode()).decode())
    mixed = [raw(p) for p in parts]
    mixed.insert(0 if foreign_first else len(mixed), raw(poison, foreign=True))
    _, _, context = observation(tmp_path, mixed)
    expected = _extract_round_metadata_records([SimpleNamespace(body=p) for p in parts], flow='pr')
    actual = extract_admitted_round_records(context.comments)
    assert len(actual) == 1 and actual[0].metadata == expected[0].metadata
    runner = FakeRunner(authenticated_actor=('bot', 424242), pr_payload={'comments': [raw(parts[0], foreign=True), *[raw(p) for p in parts[1:]]]})
    config = make_config(tmp_path)
    context = github.get_pr_review_context(runner, config=config, pr_number=77)
    with pytest.raises(AgentLoopError, match='sidecars are unavailable'):
        extract_admitted_round_records(context.comments)


@pytest.mark.parametrize('failure', ['actor', 'page', 'body', 'duplicate'])
def test_provenance_failure_stops_startup_and_refresh_without_stale_reuse(tmp_path, monkeypatch, failure):
    runner, config, context = observation(tmp_path, [raw(anchor())])
    prior = extract_admitted_round_records(context.comments)
    source = github._parse_issue_comments(runner.pr_payload['comments'])
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    if failure == 'actor':
        def broken(*args, **kwargs):
            raise AgentLoopError('unknown actor')
        monkeypatch.setattr(ingestion, 'resolve_authenticated_github_actor', broken)
    elif failure == 'page':
        monkeypatch.setattr(ingestion, 'read_complete_comment_listing',
            lambda *a, **kw: github.UnknownCommentListing('pr:77', 'incomplete page'))
    elif failure == 'body':
        source = (replace(source[0], body=str(anchor()) + ' drift'),)
    else:
        source = (*source, source[0])
    with pytest.raises(AgentLoopError):
        support._resume_pr_round_admitted(runner, config=config, pr_number=77,
            comments=source, head_sha=HEAD, configured_reviewers=config.reviewer)
    assert extract_admitted_round_records(context.comments) == prior
    assert not runner.comments


def test_capped_projection_recovers_older_history_and_preserves_human_prose(tmp_path):
    older = dict(raw(anchor(round_number=1)), databaseId=9000)
    projected = [raw(f'Human instruction {i}') for i in range(100)]
    runner = FakeRunner(authenticated_actor=('bot', 424242), pr_payload={'comments': projected}, issue_comments_by_number={77: [older, *projected]},
                        serve_rest_issue_comments=True)
    config = make_config(tmp_path)
    context = github.get_pr_review_context(runner, config=config, pr_number=77)
    records = extract_admitted_round_records(context.comments)
    assert len(records) == 1 and records[0].metadata.round_number == 1
    assert len(context.comments) == 101
    assert {c.body for c in context.comments} >= {c['body'] for c in projected}
    assert any('page=2' in ' '.join(cmd) for cmd, _ in runner.commands)


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('policy', ['all-reviewers', 'selective-intermediate', 'primary-then-panel'])
def test_live_loop_foreign_injection_after_publication_and_restart(tmp_path, parallel, policy):
    values = dict(review_parallel=parallel, pr_review_policy=policy)
    if policy == 'primary-then-panel':
        values['primary_reviewer'] = 'codex'
        values['reviewer'] = ('codex', 'gemini')
    config = make_config(tmp_path, **values)
    runner = FakeRunner(authenticated_actor=('bot', 424242), codex_outputs=[structured_pr_review(), structured_pr_review()],
                        gemini_outputs=[structured_pr_review(reviewer='Google Gemini')],
                        pr_payload={'comments': [raw(BAD, foreign=True)]})
    original = runner._run_locked
    injected = False
    def inject(args, **kwargs):
        nonlocal injected
        result = original(args, **kwargs)
        if not injected and list(args)[:3] == ['gh', 'pr', 'comment']:
            runner.pr_payload['comments'].append(dict(raw(BAD, foreign=True), createdAt='2026-01-02T00:00:00Z'))
            injected = True
        return result
    runner._run_locked = inject
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=config) == 0
    launches = sum(cmd[0] == 'codex' for cmd, _ in runner.commands)
    assert injected
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=config) == 0
    assert sum(cmd[0] == 'codex' for cmd, _ in runner.commands) == launches
    records = support.admitted_pr_records(runner, config, pr_number=77)
    assert records and all(r.metadata.subject == HEAD for r in records)


def test_native_approval_visual_and_capture_readers_share_admission(tmp_path, monkeypatch):
    from coding_review_agent_loop.capture_dispatch import qualification_capture_pending
    from coding_review_agent_loop import capture_acceptance
    from coding_review_agent_loop.review_images import admitted_visual_history
    owned = anchor(evidence_requests=('Accept screenshot at ' + HEAD,))
    runner, config, context = observation(tmp_path, [raw(owned), raw(BAD, foreign=True), raw(anchor(round_number=90), foreign=True)], architecture_context_enabled=False)
    native = native_history(runner, config, 77, context.comments)
    assert len(native) == 1 and native[0].metadata.evidence_requests
    approvals = _latest_pr_approved_reviews_for_head(context.comments, head_sha=HEAD,
        configured_reviewers=config.reviewer)
    assert set(approvals) == {'Codex'} or set(approvals) == {'OpenAI Codex'}
    seen = []
    monkeypatch.setattr(capture_acceptance, 'capture_pending', lambda records, **kw: seen.append(kw['approvals']) or False)
    assert not qualification_capture_pending(native, config=config, comments=context.comments,
        head=HEAD, plan=None, requirements=())
    assert seen[0] == approvals
    history = admitted_visual_history(runner, config, 77, context.comments)
    assert history.complete


@pytest.mark.parametrize('dry_run', [False, True])
def test_legacy_and_dry_run_use_read_only_admission(tmp_path, dry_run):
    legacy = '<!-- AGENT_LOOP_META: ' + encode_mapping(dict(flow='pr', role='reviewer', agent='Codex', round_number=3, subject=HEAD, state='approved')) + ' -->'
    runner = FakeRunner(authenticated_actor=('bot', 424242), pr_payload={'comments': [raw(legacy), raw(BAD, foreign=True)]})
    config = make_config(tmp_path, dry_run=dry_run)
    comments = github._parse_issue_comments(runner.pr_payload['comments'])
    admitted, actor = admit_pr_comments(runner, config, pr_number=77, comments=comments)
    assert actor == ('bot', 424242)
    assert extract_admitted_round_records(admitted)[0].metadata.round_number == 3
    assert not runner.comments
    assert all(cmd[:2] == ['gh', 'api'] for cmd, _ in runner.commands)


def test_no_history_fast_path_and_binding_drift(tmp_path):
    runner = FakeRunner(authenticated_actor=('bot', 424242))
    config = make_config(tmp_path)
    assert admit_pr_comments(runner, config, pr_number=77, comments=[]) == ((), None)
    assert not runner.commands
    runner.pr_payload['comments'] = [raw(anchor())]
    comments = github._parse_issue_comments(runner.pr_payload['comments'])
    admitted, _ = admit_pr_comments(runner, config, pr_number=77, comments=comments)
    with pytest.raises(AgentLoopError, match='changed after admission'):
        extract_admitted_round_records([replace(admitted[0], body=BAD)])
    with pytest.raises(AgentLoopError, match='lack an admission binding'):
        extract_admitted_round_records(comments)


@pytest.mark.parametrize('admitted_before', [None, '2026-01-02T00:00:00Z'])
@pytest.mark.parametrize('bad', [BAD, '<!-- AGENT_LOOP_META: @@@ -->'])
def test_post_merge_restore_both_branches_exclude_foreign_transport(tmp_path, monkeypatch, admitted_before, bad):
    from test_post_merge_obligations import admission
    from coding_review_agent_loop.post_merge_runtime import restore
    registry = admission()
    text = base64.b64encode(os.urandom(70000)).decode()
    parts = list(map(str, prepare_round_comment(anchor(role='summary', agent='Orchestrator',
        post_merge_registry=(registry,), raw_structured_coder_response=text))))
    runner = FakeRunner(authenticated_actor=('bot', 424242), serve_rest_issue_comments=True,
        pr_payload={'number': 1, 'comments': [raw(p) for p in parts]})
    config = make_config(tmp_path)
    comments = github._parse_issue_comments(runner.pr_payload['comments'])
    baseline = restore(runner, config, 1, comments, admitted_before=admitted_before)
    assert baseline[0] == (registry,)
    match = ROUND_TRANSPORT_SIDECAR_RE.search(parts[0])
    chunk = json.loads(base64.urlsafe_b64decode(match.group('payload')))
    chunk['data'] = 'conflict'
    poison = parts[0].replace(match.group('payload'), base64.urlsafe_b64encode(json.dumps(chunk).encode()).decode())
    runner.pr_payload['comments'].extend([raw(BAD, foreign=True), raw(poison, foreign=True)])
    comments = github._parse_issue_comments(runner.pr_payload['comments'])
    assert restore(runner, config, 1, comments, admitted_before=admitted_before) == baseline
    assert not runner.comments
    runner.pr_payload['comments'].append(raw(bad))
    comments = github._parse_issue_comments(runner.pr_payload['comments'])
    with pytest.raises(AgentLoopError, match='[Rr]ound metadata|Invalid AGENT_LOOP_META payload'):
        restore(runner, config, 1, comments, admitted_before=admitted_before)
    from coding_review_agent_loop import post_merge_runtime as runtime
    monkeypatch.setattr(runtime, 'read_api', lambda *args: {
        'merged': True, 'merged_at': '2026-01-02T00:00:00Z'})
    with pytest.raises(AgentLoopError, match='[Rr]ound metadata|Invalid AGENT_LOOP_META payload'):
        runtime.reconcile(runner, config, 1)
    assert not runner.comments



def test_managed_reconciliation_and_board_entry_ignore_foreign_round_bytes(tmp_path, monkeypatch):
    from test_managed_ci_reconciliation import recovery, _capture_activation, _no_dispatch_or_agents
    config, runner, metadata, contract, checkpoint, item, ref = recovery(tmp_path, legacy=True, auto_merge=False)
    captured = _capture_activation(monkeypatch)
    for index, body in enumerate([BAD, spilled()[0]], start=9000):
        runner.pr_payload['comments'].append(dict(raw(body, foreign=True), id=index, databaseId=index))
        runner.intent_comments.append({'id': index, 'user': {'login': 'outsider', 'id': 987654},
                                       'body': body, 'created_at': STAMP})
    assert pr_loop.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True) == 0
    assert captured
    _no_dispatch_or_agents(runner)
    context = github.get_pr_review_context(runner, config=config, pr_number=7)
    from coding_review_agent_loop.pr_board_state import resolve_pr_board_state
    state = resolve_pr_board_state(runner, config=config, pr_number=7, comments=context.comments,
        drift_error=lambda persisted, detail: AgentLoopError(detail))
    assert state.comments == context.comments


def test_converted_capture_preserves_owned_barrier_with_foreign_transport(tmp_path, monkeypatch):
    from test_pr_loop_evidence_conversion import _prepare_verified_recovery, _review
    from coding_review_agent_loop.converted_capture import recovery_capture
    runner, config, plan = _prepare_verified_recovery(tmp_path, monkeypatch)
    # The fixture drives the real loop through native evidence freeze, conversion
    # and head advance before reconstruction; this is not a mocked admission.
    comments, _, _ = support._authenticated_pr_comments_status(runner, config, pr_number=77)
    records = _extract_round_metadata_records(comments, flow='pr')
    items = next(r.metadata.prior_items for r in reversed(records) if r.metadata.prior_items)
    head = runner.pr_payload['headRefOid']
    baseline = recovery_capture(runner, config, pr_number=77, items=items, head=head, plan=plan)
    runner.pr_payload['comments'].extend([raw(BAD, foreign=True), raw(spilled()[0], foreign=True)])
    assert recovery_capture(runner, config, pr_number=77, items=items, head=head, plan=plan) == baseline
    runner.codex_outputs.append(_review())
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=plan) == 0
    admitted = support.admitted_pr_records(runner, config, pr_number=77)
    assert all(r.metadata.round_number < 90 for r in admitted)


@pytest.mark.parametrize('bad', [BAD, '<!-- AGENT_LOOP_META: @@@ -->'])
def test_managed_continuity_decodes_only_owned_anchors_and_chunks(tmp_path, bad):
    from test_managed_ci import AuthorizationCommentRunner, _ci_repair_round_comment
    from coding_review_agent_loop.managed_ci_authorization import find_actor_round_metadata_comment_ids
    from coding_review_agent_loop.round_metadata import _decode_round_metadata, _strip_round_metadata
    original = _ci_repair_round_comment(61)
    metadata = _extract_round_metadata_records([SimpleNamespace(body=original['body'])], flow='pr')[0].metadata
    body = _attach_round_metadata(_strip_round_metadata(original['body']), replace(metadata,
        raw_structured_coder_response=base64.b64encode(os.urandom(70000)).decode()))
    parts = list(map(str, prepare_round_comment(body)))
    runner = AuthorizationCommentRunner()
    runner.intent_comments.extend({'id': i, 'user': {'login': 'agent-loop', 'id': 1},
        'created_at': STAMP, 'body': part} for i, part in enumerate(parts, start=61))
    config = make_config(tmp_path)
    def select():
        return find_actor_round_metadata_comment_ids(runner, config=config, pr_number=7,
            actor_login='agent-loop', actor_id=1, predecessor_head=HEAD,
            new_head='ci-fix-head', round_number=12, after_comment_id=50)
    baseline = select()
    assert baseline
    match = ROUND_TRANSPORT_SIDECAR_RE.search(parts[0])
    payload = json.loads(base64.urlsafe_b64decode(match.group('payload')))
    payload['data'] = 'conflict'
    poison = parts[0].replace(match.group('payload'), base64.urlsafe_b64encode(json.dumps(payload).encode()).decode())
    runner.intent_comments.extend({'id': i, 'user': {'login': 'outsider', 'id': 99},
        'created_at': STAMP, 'body': part} for i, part in enumerate([BAD, poison], start=900))
    assert select() == baseline
    runner.intent_comments.append({'id': 999, 'user': {'login': 'agent-loop', 'id': 1},
        'created_at': STAMP, 'body': bad})
    with pytest.raises(AgentLoopError, match='[Rr]ound metadata|Invalid AGENT_LOOP_META payload'):
        select()
    assert not runner.comments


@pytest.mark.parametrize('parallel', [False, True])
def test_issue_created_pr_admits_foreign_history_before_review(tmp_path, parallel):
    from test_pr_loop_review_images import run_entry
    runner = FakeRunner(codex_outputs=[structured_pr_review()],
        pr_payload={'comments': [raw(BAD, foreign=True)]})
    config = make_config(tmp_path, review_parallel=parallel)
    assert run_entry(runner, config, 'issue') == 0
    records = support.admitted_pr_records(runner, config, pr_number=77)
    assert records and not any(r.metadata.round_number == 90 for r in records)


def test_parallel_interrupted_publication_restarts_with_foreign_transport(tmp_path):
    from test_review_parallel import _approving_pr_runner, _interrupt_pr_round_between_publications
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=('codex', 'gemini'), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    launches = list(runner.reviewer_launches)
    runner.pr_payload['comments'].extend([raw(BAD, foreign=True), raw(spilled()[0], foreign=True)])
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.reviewer_launches == launches
    assert not runner.peer_body_visible_at_launch
    records = support.admitted_pr_records(runner, config, pr_number=77)
    reviews = [r for r in records if r.metadata.role == 'reviewer']
    assert {r.metadata.agent for r in reviews} == {'Codex', 'Gemini'}


def test_host_footer_binding_preserves_raw_prose_and_canonical_protocol(tmp_path):
    from coding_review_agent_loop.protocol_markers import KNOWN_HOST_COMMENT_FOOTER
    body = str(anchor()) + KNOWN_HOST_COMMENT_FOOTER
    _, _, context = observation(tmp_path, [raw(body), raw(BAD, foreign=True)])
    assert context.comments[0].body == body
    assert extract_admitted_round_records(context.comments)[0].metadata.round_number == 3


@pytest.mark.parametrize('policy', ['selective-intermediate', 'primary-then-panel'])
def test_refresh_identity_failure_stops_without_later_agent_launches(tmp_path, policy):
    from coding_review_agent_loop.runner import CommandResult
    runner = FakeRunner(codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer='Google Gemini')])
    config = make_config(tmp_path, reviewer=('codex', 'gemini'), pr_review_policy=policy, auto_merge=True,
        primary_reviewer='codex' if policy == 'primary-then-panel' else None)
    original = runner._run_locked
    published = False
    launches_at_failure = None
    def fail_refresh(args, **kwargs):
        nonlocal published, launches_at_failure
        cmd = list(args)
        if published and cmd[:2] == ['gh', 'api'] and '/comments?' in cmd[2]:
            launches_at_failure = sum(c[0] in {'codex', 'gemini', 'claude'} for c, _ in runner.commands)
            recorded, cwd = runner._record_command(args, kwargs['cwd'])
            return CommandResult(recorded, cwd, '', 'identity read unavailable', 1)
        result = original(args, **kwargs)
        if cmd[:3] == ['gh', 'pr', 'comment']:
            published = True
        return result
    runner._run_locked = fail_refresh
    with pytest.raises(AgentLoopError, match='cannot be authenticated'):
        pr_loop.run_pr_loop(runner, pr_number=77, config=config)
    assert launches_at_failure is not None
    assert sum(c[0] in {'codex', 'gemini', 'claude'} for c, _ in runner.commands) == launches_at_failure
    assert not any(c[:3] == ['gh', 'pr', 'merge'] for c, _ in runner.commands)


def test_advisory_step_back_reader_cannot_suppress_failed_admission(tmp_path, monkeypatch):
    runner, config, context = observation(tmp_path, [raw(anchor())], pr_step_back_rounds=1)
    monkeypatch.setattr(support, '_pr_step_back_records_may_matter', lambda *a, **kw: True)
    runner.serve_rest_issue_comments = False
    with pytest.raises(AgentLoopError, match='cannot be authenticated'):
        support._pr_step_back_decision(runner, config, pr_number=77, round_number=4,
            snapshot_comments=context.comments, tracked=('Codex',), head_sha=HEAD)
    assert not runner.comments


def test_new_head_readmission_keeps_owned_item_high_water_and_drops_foreign_authority(tmp_path):
    from coding_review_agent_loop.protocol import UnresolvedReviewItem
    item = UnresolvedReviewItem(item_id='item-12', reviewer='Codex', source_round=3,
        text='Optional later improvement.', status='future', source_status='future')
    runner, config, context = observation(tmp_path, [raw(anchor(prior_items=(item,))), raw(BAD, foreign=True)])
    resumed = support._resume_pr_round_admitted(runner, config=config, pr_number=77,
        comments=context.comments, head_sha=HEAD, configured_reviewers=config.reviewer)
    assert resumed.next_unresolved_item_number == 13
    new_head = 'a' * 40
    runner.pr_payload['headRefOid'] = new_head
    runner.git_head = new_head
    runner.simulate_new_process()
    runner.codex_outputs.append(structured_pr_review())
    config = replace(config, review_unrecorded_head=True)
    assert pr_loop.run_pr_loop(runner, pr_number=77, config=config) == 0
    admitted = support.admitted_pr_records(runner, config, pr_number=77)
    assert any(r.metadata.subject == new_head for r in admitted)
    assert any(i.item_id == 'item-12' for r in admitted for i in r.metadata.prior_items)
    assert all(r.metadata.round_number < 90 for r in admitted)


@pytest.mark.parametrize('missing', ['user', 'id'])
def test_rest_only_readers_refuse_unproven_candidates_before_decoding(tmp_path, missing):
    from coding_review_agent_loop.managed_ci_reconciliation import resolve_legacy_reference
    from coding_review_agent_loop.pr_round_ingestion import owned_rest_protocol_bodies
    comment = {'id': 42, 'user': {'login': 'outsider', 'id': 99}, 'body': BAD}
    comment.pop(missing)
    with pytest.raises(AgentLoopError, match='cannot authenticate REST'):
        owned_rest_protocol_bodies([comment], actor_id=1)
    with pytest.raises(AgentLoopError, match='cannot authenticate REST'):
        resolve_legacy_reference([comment], SimpleNamespace(trusted_actor_id=1), None)


@pytest.mark.parametrize('bad', [BAD, '<!-- AGENT_LOOP_META: @@@ -->'])
def test_legacy_reconciliation_rejects_owned_corruption_with_valid_baseline(tmp_path, bad):
    from test_managed_ci_reconciliation import recovery
    from coding_review_agent_loop.managed_ci_reconciliation import resolve_legacy_reference
    config, runner, metadata, contract, checkpoint, item, ref = recovery(tmp_path, legacy=True)
    from coding_review_agent_loop import managed_ci as ci
    page = ci._read_intent_comment_page(runner, config, 7)
    baseline = resolve_legacy_reference(page, contract, checkpoint)
    assert baseline is not None
    corrupt = {'id': 9999, 'user': {'login': 'outsider', 'id': 99}, 'body': bad}
    page.append(corrupt)
    assert resolve_legacy_reference(page, contract, checkpoint) == baseline
    corrupt['user'] = {'login': contract.trusted_actor_login, 'id': contract.trusted_actor_id}
    with pytest.raises(AgentLoopError, match='[Rr]ound metadata|Invalid AGENT_LOOP_META payload'):
        resolve_legacy_reference(page, contract, checkpoint)
    assert not runner.comments


@pytest.mark.parametrize('projection', ['admitted', 'rest'])
@pytest.mark.parametrize('foreign', [False, True])
@pytest.mark.parametrize('bad', ['<!-- AGENT_LOOP_META: -->', '<!-- AGENT_LOOP_META: @@@ -->'])
def test_all_protocol_projections_share_owned_envelope_validation(tmp_path, projection, foreign, bad):
    from coding_review_agent_loop.pr_round_ingestion import owned_rest_protocol_bodies, protocol_comments
    if projection == 'rest':
        comments = [{'id': 1, 'user': {'id': 424242}, 'body': str(anchor())},
                    {'id': 2, 'user': {'id': 99 if foreign else 424242}, 'body': bad}]
        read = lambda: owned_rest_protocol_bodies(comments, actor_id=424242)
    else:
        _, _, context = observation(tmp_path, [raw(anchor()), raw(bad, foreign=foreign)])
        read = lambda: tuple(c.body for c in protocol_comments(context.comments))
    if foreign:
        assert read() == (str(anchor()), '')
    else:
        with pytest.raises(AgentLoopError, match='round metadata envelope.*comment'):
            read()
