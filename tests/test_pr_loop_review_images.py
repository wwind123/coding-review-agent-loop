"""Provider-free visual lifecycle through the production PR orchestrator."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review, structured_coder_followup
from coding_review_agent_loop import review_images as visual
from coding_review_agent_loop import validated_agent
from coding_review_agent_loop.agents.base import AgentResult
from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github_changed_paths import ChangedPath, PullRequestChangedPaths
from coding_review_agent_loop.review_images_capture import CaptureResult
from coding_review_agent_loop.review_images_manifest import publish
from coding_review_agent_loop.review_images_transport import ImageDeliveryProof
from coding_review_agent_loop.review_scope import is_visual_review
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from coding_review_agent_loop.pr_loop import run_pr_loop
from review_images_helpers import png, manifest


@pytest.fixture
def visual_setup(tmp_path, monkeypatch):
    config = make_config(tmp_path, reviewer=('codex',), review_images_seat='codex', reviewer_codex_model='gpt-6.1-sol',
                         review_images_cmd='/usr/bin/python3 capture.py',
                         review_images_paths=('ui/*',), review_images_runtime_grant=('/usr/bin/python3',),
                         max_rounds=3)
    calls = []
    from coding_review_agent_loop import review_images_isolation, github_changed_paths
    monkeypatch.setattr(review_images_isolation, 'validate_grants', lambda grants: None)
    def preflight(cfg, seat=None):
        calls.append(('preflight', cfg))
        from coding_review_agent_loop.reviewer_seats import seat_backend
        selected, _, invocation, _ = visual.reviewer_view(cfg, seat)
        return SimpleNamespace(seat=str(selected), provider=seat_backend(selected),
                               model=invocation.configured_model, effort=invocation.resolved_effort,
                               contract_id='stub', executable_digest='stub')
    monkeypatch.setattr(visual, 'preflight_images', preflight)
    def paths(runner, *, config, pr_number, head_sha):
        calls.append(('paths', head_sha))
        return PullRequestChangedPaths(head_sha, True, (ChangedPath('modified', 'ui/view.tsx'),))
    monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', paths)
    monkeypatch.setattr(visual, 'pinned_diff', lambda *a: calls.append(('diff', a[2].head_sha)) or 'pinned diff')
    monkeypatch.setattr(visual, 'observe_ci', lambda *a, **kw: calls.append(('ci', a[2].head_sha)))
    def capture(checkout, head, settings, changes):
        calls.append(('capture', head))
        value = manifest(head)
        snap = publish(json.dumps(value).encode(), head, settings.limits,
                       {'one.png': png(), 'base.png': png()},
                       {'one.png': (2, 2), 'base.png': (2, 2)}, 'stub-config')
        return CaptureResult('success', snapshots=snap)
    monkeypatch.setattr(visual, 'capture', capture)
    original = validated_agent.run_agent_result
    def reviewer(runner, **kwargs):
        images = kwargs.get('image_inputs')
        if images is None:
            return original(runner, **kwargs)
        calls.append(('images', kwargs))
        text = structured_pr_review()
        proof = ImageDeliveryProof(images.binding, tuple(d for _, d in images.binding.inventory),
                                    'fresh', 'turn', (images.binding.configured_model,), 'stub')
        return AgentResult(text=text, raw_output=text, message_text=text, returncode=0,
                           image_delivery_proof=proof, provider='codex', role='reviewer')
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    return config, calls, reviewer


def records(runner):
    return _extract_round_metadata_records([SimpleNamespace(body=comment['body']) for comment in runner.pr_payload.get('comments', [])], flow='pr')


@pytest.mark.parametrize('entry', ['pr','issue'])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_inert_config_ordinary_workflow(visual_setup, monkeypatch, parallel, enabled, entry):
    config, calls, _ = visual_setup
    config = replace(config, review_parallel=parallel)
    if not enabled:
        config = replace(config, review_images_paths=())
    else:
        from coding_review_agent_loop import github_changed_paths
        monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', lambda *a, **kw:
            PullRequestChangedPaths(kw['head_sha'], True, (ChangedPath('modified', 'server.py'),)))
    runner = FakeRunner(codex_outputs=[structured_pr_review()])
    assert run_entry(runner, config, entry) == 0
    assert not any(call[0] in {'capture', 'images', 'ci', 'diff'} for call in calls)
    if not enabled:
        assert calls == []
    assert not any(is_visual_review(r.metadata) for r in records(runner))


@pytest.mark.parametrize('entry', ['pr', 'issue'])
@pytest.mark.parametrize('parallel', [False, True])
def test_verified_visual_pass_never_supplies_code_approval(visual_setup, parallel, entry):
    config, calls, _ = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review()],
        claude_outputs=["Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
        pr_payload={"body": "Fixes #56"})
    config = replace(config, review_parallel=parallel, review_images_timeout=7)
    if entry == 'issue':
        from coding_review_agent_loop.issue_loop import run_issue_loop
        assert run_issue_loop(runner, issue_number=56, config=config) == 0
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    published = records(runner)
    images = [r for r in published if is_visual_review(r.metadata)]
    assert len(images) == 1, [(r.metadata.review_scope, r.metadata.agent, r.metadata.subject) for r in published]
    ordinary = [r for r in published if r.metadata.role == 'reviewer' and not is_visual_review(r.metadata)]
    assert ordinary and ordinary[-1].metadata.state == 'approved'
    kinds = [c[0] for c in calls]
    assert kinds.index('ci') < kinds.index('capture') < kinds.index('images')
    kwargs = next(c[1] for c in calls if c[0] == 'images')
    assert kwargs['timeout_seconds'] == 300 and kwargs['session_id'] is None
    assert 'pinned diff' in kwargs['prompt'] and 'requester coverage' in kwargs['prompt']
    assert images[0].metadata.scheduler_contract is None
    preflights = [c[1] for c in calls if c[0] == 'preflight']
    assert len(preflights) >= 2
    assert kwargs['config'] is preflights[-1]
    assert kwargs['config'] is not config


@pytest.mark.parametrize('failure', ['capture', 'proof', 'paths', 'diff', 'value-error'])
def test_failed_evidence_continues_ordinary_workflow(visual_setup, monkeypatch, failure):
    config, calls, _ = visual_setup
    if failure == 'capture':
        monkeypatch.setattr(visual, 'capture', lambda *a: CaptureResult('failed', 'stub failure'))
    elif failure in {'proof', 'value-error'}:
        def fail(*a, **kw):
            if failure == 'value-error':
                raise ValueError('capability rebound')
            return AgentResult(text=structured_pr_review(), raw_output='', returncode=0)
        monkeypatch.setattr(validated_agent, 'run_agent_result',
                            lambda runner, **kw: fail() if kw.get('image_inputs') else
                            __import__('coding_review_agent_loop.agents.registry', fromlist=['run_agent_result']).run_agent_result(runner, **kw))
    elif failure == 'diff':
        monkeypatch.setattr(visual, 'pinned_diff', lambda *a: (_ for _ in ()).throw(ValueError('diff failed')))
    else:
        from coding_review_agent_loop import github_changed_paths
        monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', lambda *a, **kw:
            PullRequestChangedPaths(kw['head_sha'], False, diagnostic='incomplete'))
    runner = FakeRunner(codex_outputs=[structured_pr_review()])
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not any(is_visual_review(r.metadata) for r in records(runner))


@pytest.mark.parametrize("entry", ["pr","issue"])
def test_startup_refuses_seat_before_capture(visual_setup, monkeypatch, entry):
    config, calls, _ = visual_setup
    monkeypatch.setattr(visual, 'preflight_images', lambda *a: (_ for _ in ()).throw(ValueError('unsupported seat')))
    with pytest.raises(AgentLoopError, match='startup refused'):
        run_entry(FakeRunner(), config, entry)
    assert not calls


@pytest.mark.parametrize('phase', ['capture', 'images'])
def test_cancellation_propagates_after_snapshot_cleanup(visual_setup, monkeypatch, phase):
    config, calls, _ = visual_setup
    owned = []
    original = visual.capture
    def capture(*args):
        if phase == 'capture':
            raise KeyboardInterrupt()
        result = original(*args)
        owned.extend(result.snapshots.files)
        return result
    monkeypatch.setattr(visual, 'capture', capture)
    monkeypatch.setattr(validated_agent, 'run_agent_result', lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(FakeRunner(), pr_number=77, config=config)
    assert all(file._closed for file in owned)


def test_visual_dispositions_are_explicit_and_scope_isolated():
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item, _validate_review_response
    item = replace(_next_unresolved_item(item_number=1, reviewer='OpenAI Codex', source_round=1,
                         text='Clipped title', status='blocking'), origin_scope='visible-defects')
    clean = structured_pr_review()
    with pytest.raises(AgentLoopError, match='exactly'):
        visual.validate_visual(clean, reviewer='OpenAI Codex', items=(item,))
    resolved = structured_pr_review(prior_item_dispositions=[{'item_id':'item-1','disposition':'resolved'}])
    assert visual.validate_visual(resolved, reviewer='OpenAI Codex', items=(item,)).state == 'approved'
    with pytest.raises(AgentLoopError):
        _validate_review_response(resolved, reviewer='OpenAI Codex', unresolved_items=(item,), architecture_status_mode='strict')
    assert _validate_review_response(clean, reviewer='OpenAI Codex', unresolved_items=(item,), architecture_status_mode='strict').state == 'approved'


@pytest.mark.parametrize('push', [False, True])
@pytest.mark.parametrize('parallel', [False, True])
def test_carried_visual_finding_requires_fresh_verified_disposition(visual_setup, monkeypatch, push, parallel):
    config, calls, original = visual_setup
    texts = [structured_pr_review(state='blocking', blocking_items=['Title is clipped in the sealed mobile image.']),
             structured_pr_review(prior_item_dispositions=[{'item_id':'item-1', 'disposition':'resolved'}])]
    def reviewer(runner, **kw):
        result = original(runner, **kw)
        if kw.get('image_inputs'):
            text = texts.pop(0)
            return replace(result, text=text, message_text=text, raw_output=text)
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    runner = FakeRunner(codex_outputs=[structured_pr_review(), structured_pr_review()],
        claude_outputs=[structured_coder_followup(addressed_items=['item-1'],
                    addressed_item_notes={'item-1':'Adjusted title width.'})],
        advance_pr_head_on_coder_followup=push)
    config = replace(config, review_parallel=parallel)
    if push:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert len([c for c in calls if c[0] == 'capture']) == 2
        image_calls = [c[1] for c in calls if c[0] == 'images']
        assert 'Title is clipped' in image_calls[-1]['prompt'] and 'item-1' in image_calls[-1]['prompt']
    else:
        with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
            run_pr_loop(runner, pr_number=77, config=config)
        assert len([c for c in calls if c[0] == 'capture']) == 1
        assert len([cmd for cmd, _ in runner.commands if cmd[:1] == ['claude']]) == 1
    published = [r for r in records(runner) if is_visual_review(r.metadata)]
    assert published[0].metadata.new_items[0].origin_scope == 'visible-defects'
    assert not any(cmd[:3] == ['gh','pr','merge'] for cmd, _ in runner.commands)


@pytest.mark.parametrize('status,filename,previous', [
    ('added','ui/view.tsx',None), ('modified','ui/view.tsx',None),
    ('renamed','server.py','ui/deleted.tsx'), ('renamed','ui/new.tsx','old.py'),
    ('removed','ui/deleted.tsx',None)])
def test_literal_path_modes_through_orchestration(visual_setup, monkeypatch, status, filename, previous):
    config, calls, _ = visual_setup
    from coding_review_agent_loop import github_changed_paths
    monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', lambda *a, **kw:
        PullRequestChangedPaths(kw['head_sha'], True, (ChangedPath(status, filename, previous),)))
    assert run_pr_loop(FakeRunner(codex_outputs=[structured_pr_review()]), pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0] == 'capture']) == 1


@pytest.mark.parametrize('ci', ['pending','completed','absent','unavailable','api-exception','excluded','managed'])
def test_real_ci_start_observation_precedes_capture(visual_setup, monkeypatch, ci):
    config, calls, _ = visual_setup
    from coding_review_agent_loop import github
    clock = [0]
    monkeypatch.setattr(visual, 'time', SimpleNamespace(monotonic=lambda: clock[0],
        sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)))
    def checks(*a, **kw):
        calls.append(('observed-ci', kw['metadata'].head_sha))
        if ci == 'api-exception':
            raise AgentLoopError('check API unavailable')
        return SimpleNamespace(check_query_status='unavailable' if ci=='unavailable' else 'ok',
            state='pending' if ci=='pending' else 'no_checks',
            passing=(1,) if ci=='completed' else (), pending=(1,) if ci=='pending' else (), failing=(), excluded=(1,) if ci=='excluded' else ())
    monkeypatch.setattr(github, 'get_pr_checks', checks)
    monkeypatch.setattr(visual, 'observe_ci', _PRODUCTION_OBSERVE_CI)
    # Managed absence itself is tested at the production observer; orchestration
    # still never dispatches workflow CI from this visual seam.
    if ci == 'managed':
        _PRODUCTION_OBSERVE_CI(FakeRunner(), config, SimpleNamespace(head_sha='a'*40),
                              managed=True, diagnostic=lambda msg: calls.append(('diagnostic',msg)))
    assert run_pr_loop(FakeRunner(codex_outputs=[structured_pr_review()]), pr_number=77, config=config) == 0
    assert next(i for i,c in enumerate(calls) if c[0]=='observed-ci') < next(i for i,c in enumerate(calls) if c[0]=='capture')
    assert clock[0] == (config.ci_startup_timeout_seconds if ci in {'absent','excluded','managed'} else 0)


# Capture the real observer before the per-test fixture replaces the seam.
_PRODUCTION_OBSERVE_CI = visual.observe_ci


@pytest.mark.parametrize('success', [True, False])
def test_restart_never_reuses_visual_success(visual_setup, monkeypatch, success):
    config, calls, _ = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review(), structured_pr_review()])
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='capture']) == 1
    before = len([r for r in records(runner) if is_visual_review(r.metadata)])
    if not success:
        monkeypatch.setattr(visual, 'capture', lambda *a: calls.append(('capture','failed')) or CaptureResult('failed','restart'))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='capture']) == 2
    assert len([r for r in records(runner) if is_visual_review(r.metadata)]) == before + int(success)


@pytest.mark.parametrize('at', ['listing', 'capture', 'images'])
def test_stale_head_result_is_discarded_and_next_head_is_attempted(visual_setup, monkeypatch, at):
    config, calls, original = visual_setup
    from test_pr_loop_evidence_conversion import _push_external_head
    runner = FakeRunner(codex_outputs=[structured_pr_review()])
    moved = [False]
    def move():
        if not moved[0]:
            moved[0] = True
            _push_external_head(runner, '6946a6f1ac1c65132a6a53600572ceb4af0ee23b')
    if at == 'listing':
        from coding_review_agent_loop import github_changed_paths
        original_paths = github_changed_paths.get_pr_changed_paths
        def paths(*args, **kwargs):
            if not moved[0]:
                move()
                return PullRequestChangedPaths(kwargs['head_sha'], False, diagnostic='head moved')
            return original_paths(*args, **kwargs)
        monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', paths)
    elif at == 'capture':
        original_capture = visual.capture
        def capture(*args):
            result = original_capture(*args)
            move()
            return result
        monkeypatch.setattr(visual, 'capture', capture)
    else:
        def reviewer(runner, **kw):
            result = original(runner, **kw)
            if kw.get('image_inputs'):
                move()
            return result
        monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    pubs = [r for r in records(runner) if is_visual_review(r.metadata)]
    assert len(pubs) == 1 and pubs[0].metadata.subject == '6946a6f1ac1c65132a6a53600572ceb4af0ee23b'
    assert len([c for c in calls if c[0]=='capture']) == (1 if at == 'listing' else 2)


def test_cleanup_uncertainty_stops_ordinary_review(visual_setup, monkeypatch):
    config, calls, _ = visual_setup
    monkeypatch.setattr(visual, 'capture', lambda *a: CaptureResult('failed','hierarchical cleanup could not be verified'))
    with pytest.raises(ValueError, match='cleanup'):
        run_pr_loop(FakeRunner(), pr_number=77, config=config)


def test_visual_tail_cannot_hide_frozen_no_execution_boundary(visual_setup):
    config, calls, _ = visual_setup
    from pr_loop_test_helpers import _evidence_review, _EVIDENCE_TEXT
    from review_images_helpers import append_visual_record
    runner = FakeRunner(codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])])
    disabled = replace(config, review_images_paths=())
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=disabled)
    append_visual_record(runner)
    for _ in range(2):
        runner.simulate_new_process()
        with pytest.raises(HumanDecisionRequiredError, match='frozen at head'):
            run_pr_loop(runner, pr_number=77, config=config)
    assert not any(c[0]=='capture' for c in calls)


@pytest.mark.parametrize('import_mode', ['local','missing','moved','diff-failed'])
def test_pinned_local_dispatch_inputs_use_immutable_objects(visual_setup, tmp_path, monkeypatch, import_mode):
    config, calls, _ = visual_setup
    from review_images_helpers import repository
    from coding_review_agent_loop import github_retry, review_images_worktree, git_transport
    root, base, git = repository(tmp_path)
    git('branch','-M','main')
    (root/'ui').mkdir()
    (root/'ui'/'view.tsx').write_text('visible line\n'*10000)
    git('add','ui')
    git('commit','-qm','visual change')
    head = git('rev-parse','HEAD')
    (root/'ui'/'view.tsx').write_text('dirty content never in diff')
    runner = FakeRunner(git_head=head, pr_payload={'headRefOid':head,'baseRefName':'main'},
                        codex_outputs=[structured_pr_review()])
    config = replace(config, claude_dir=root)
    read = github_retry.run_gh_read
    def tuple_read(runner, args, **kw):
        result = read(runner, args, **kw)
        if args[-1] == 'repos/OWNER/REPO/pulls/77':
            value = json.loads(result.stdout)
            value['base']['sha'] = base
            return replace(result, stdout=json.dumps(value))
        return result
    monkeypatch.setattr(github_retry, 'run_gh_read', tuple_read)
    original_git = review_images_worktree.git
    imports = []
    moved_ref = [False]
    def local_git(checkout, args, **kw):
        calls.append(('git', args))
        if args[0] == 'rev-list' and import_mode in {'missing','moved'}:
            raise ValueError('missing closure')
        if args[0] == 'diff' and import_mode == 'diff-failed':
            raise ValueError('pinned diff read failed')
        return original_git(checkout, args, **kw)
    def import_ref(checkout, source, target, **kw):
        imports.append((source,target))
        if import_mode == 'moved' and target.startswith('refs/agent-loop/images/') and not moved_ref[0]:
            moved_ref[0] = True
            return 'f'*40
        return head if source.endswith('/head') else base
    monkeypatch.setattr(review_images_worktree, 'git', local_git)
    monkeypatch.setattr(git_transport, 'import_ref', import_ref)
    monkeypatch.setattr(visual, 'pinned_diff', _PRODUCTION_PINNED_DIFF)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    if import_mode == 'diff-failed':
        assert not any(c[0]=='capture' for c in calls)
        assert not any(is_visual_review(r.metadata) for r in records(runner))
    else:
        prompt = next(c[1]['prompt'] for c in calls if c[0]=='images')
        assert 'visible line' in prompt and 'dirty content never' not in prompt
        assert 'truncated' in prompt
        diff_args = next(c[1] for c in calls if c[0]=='git' and c[1][0]=='diff')
        assert base in diff_args and head in diff_args
        assert '--no-ext-diff' in diff_args and '--no-textconv' in diff_args
        assert any(target.startswith('refs/agent-loop/images/') for _,target in imports) == (import_mode in {'missing','moved'})
        if import_mode == 'moved':
            assert len([c for c in calls if c[0]=='paths']) == 2
            assert len([c for c in calls if c[0]=='capture']) == 1


_PRODUCTION_PINNED_DIFF = visual.pinned_diff


@pytest.mark.parametrize('disputed', [True, False])
def test_no_push_coder_dispute_or_nothing_to_change_uses_visual_human_stop(visual_setup, monkeypatch, disputed):
    config, calls, original = visual_setup
    def reviewer(runner, **kw):
        result = original(runner, **kw)
        if kw.get('image_inputs'):
            text = structured_pr_review(state='blocking', blocking_items=['Mobile heading clips at the right edge.'])
            return replace(result, text=text, message_text=text, raw_output=text)
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    kwargs = (dict(disputed_items=['item-1'], dispute_evidence={'item-1':'The title fits the supplied viewport; no code change is needed.'})
              if disputed else dict(remaining_items=['item-1'], remaining_item_notes={
                  'item-1':'Nothing to change: the title already fits the supplied viewport.'}))
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*2,
        claude_outputs=[structured_coder_followup(summary='No changes needed.', **kwargs)],
        advance_pr_head_on_coder_followup=False)
    with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len([c for c in calls if c[0]=='capture']) == 1
    assert len([cmd for cmd,_ in runner.commands if cmd[:1]==['claude']]) == 1
    assert not any(cmd[:3]==['gh','pr','merge'] for cmd,_ in runner.commands)


def run_entry(runner, config, entry):
    if entry == 'pr':
        return run_pr_loop(runner, pr_number=77, config=config)
    from coding_review_agent_loop.issue_loop import run_issue_loop
    runner.pr_payload['body'] = 'Fixes #56'
    runner.claude_outputs.insert(0, 'Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->')
    return run_issue_loop(runner, issue_number=56, config=config)


def test_named_visual_seat_publication_preserves_binding_on_restart(visual_setup):
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent, reviewer_seat_binding
    config, calls, _ = visual_setup
    seat = SeatAgent(ReviewerSeat('visual','codex',('gpt-6.1-sol',),'medium'), config.codex_dir)
    config = replace(config, reviewer=(seat,), reviewer_seats=(seat,), review_images_seat='visual')
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*2)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    published = [r.metadata for r in records(runner) if is_visual_review(r.metadata)]
    assert published[0].seat_binding == reviewer_seat_binding(config)
    assert published[0].agent == 'visual'
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='capture']) == 2


@pytest.mark.parametrize('selection', ['explicit', 'primary'])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('status', ['blocking', 'same-pr'])
@pytest.mark.parametrize('opportunity', ['other-seat', 'failed', 'nonmatching', 'assigned'])
def test_carried_visual_owner_disposition_opportunity_controls_human_stop(
        visual_setup, monkeypatch, selection, parallel, status, opportunity):
    from coding_review_agent_loop import pr_loop, github_changed_paths
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
    config, calls, original = visual_setup
    seats = []
    for name in ('alpha', 'beta'):
        directory = config.codex_dir.parent / name
        directory.mkdir()
        seats.append(SeatAgent(ReviewerSeat(name, 'codex', ('gpt-6.1-sol',), 'medium'), directory))
    config = replace(config, reviewer=tuple(seats), reviewer_seats=tuple(seats),
        review_parallel=parallel, review_images_seat='alpha' if selection == 'explicit' else None,
        pr_review_policy='all-reviewers' if selection == 'explicit' else 'primary-then-panel',
        primary_reviewer=None if selection == 'explicit' else seats[0])
    texts = [structured_pr_review(state='blocking', **{
        'blocking_items' if status == 'blocking' else 'same_pr_followups': ['Clipped title in mobile view.']}),
        structured_pr_review(prior_item_dispositions=[{'item_id': 'item-1', 'disposition': 'resolved'}])
        if opportunity == 'assigned' else structured_pr_review()]
    def reviewer(runner, **kwargs):
        result = original(runner, **kwargs)
        if kwargs.get('image_inputs'):
            text = texts.pop(0)
            return replace(result, text=text, message_text=text, raw_output=text)
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    run = pr_loop._run_validated_agent
    coder_calls = []
    def stop_at_coder(*args, **kwargs):
        if kwargs.get('role') == 'coder':
            coder_calls.append(kwargs)
            raise KeyboardInterrupt()
        return run(*args, **kwargs)
    monkeypatch.setattr(pr_loop, '_run_validated_agent', stop_at_coder)
    runner = FakeRunner(codex_outputs=[structured_pr_review()] * 8)
    # A newly reported blocker still gets its first ordinary coder opportunity.
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(coder_calls) == 1
    coder_calls.clear()
    if selection == 'explicit':
        runner.simulate_new_process()
    else:
        # Seed the real visual publication without ordinary board history:
        # an in-flight primary change otherwise stops at the separate board gate.
        visual_comments = [comment for comment in runner.pr_payload['comments'] if any(
            is_visual_review(record.metadata) for record in _extract_round_metadata_records(
                [SimpleNamespace(body=comment['body'])], flow='pr'))]
        runner = FakeRunner(codex_outputs=[structured_pr_review()] * 8,
                            pr_payload={'comments': visual_comments})
    selected = seats[0] if opportunity == 'assigned' else seats[1]
    config = replace(config, review_images_seat=str(selected) if selection == 'explicit' else None,
        primary_reviewer=None if selection == 'explicit' else selected)
    if opportunity == 'failed':
        monkeypatch.setattr(visual, 'capture', lambda *args: CaptureResult('failed', 'stub capture failed'))
    elif opportunity == 'nonmatching':
        monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', lambda *args, **kwargs:
            PullRequestChangedPaths(kwargs['head_sha'], True, (ChangedPath('modified', 'server.py'),)))
    if opportunity == 'assigned':
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    else:
        with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1') as error:
            run_pr_loop(runner, pr_number=77, config=config)
        assert runner.pr_payload['headRefOid'] in str(error.value)
        if opportunity == 'other-seat':
            assert 'visual seat beta' in str(error.value) and 'item-1 owners alpha' in str(error.value)
        else:
            assert ('stub capture failed' if opportunity == 'failed' else 'no matching changed paths') in str(error.value)
        # Ordinary work completes (or exact-head approvals are reused) before the stop.
        assert any(r.metadata.role == 'reviewer' and not is_visual_review(r.metadata)
                   and r.metadata.state == 'approved' for r in records(runner))
        recovered = visual.recover_visual_findings(records(runner), reconciliation_mode='owner-scoped')
        assert len(recovered) == 1
        assert (recovered[0].item_id, recovered[0].reviewer, recovered[0].status) == ('item-1', 'alpha', status)
        assert (recovered[0].resolution_owners or (recovered[0].reviewer,)) == ('alpha',)
    assert not coder_calls
    assert not any(cmd[:3] == ['gh', 'pr', 'merge'] for cmd, _ in runner.commands)
    publications = [r.metadata for r in records(runner) if is_visual_review(r.metadata)]
    assert len(publications) == (2 if opportunity in {'assigned', 'other-seat'} else 1)
    if opportunity == 'other-seat':
        assert publications[-1].agent == 'beta' and publications[-1].prior_items == ()
        assert publications[-1].state == 'approved'
        prompt = [c[1]['prompt'] for c in calls if c[0] == 'images'][-1]
        assert 'Clipped title in mobile view.' not in prompt


def test_amended_visual_owner_can_resolve_and_stays_resolved_after_restart(visual_setup, monkeypatch):
    from coding_review_agent_loop import pr_loop
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from pr_loop_test_helpers import _m943_append
    config, calls, original = visual_setup
    seats = []
    for name in ('alpha','beta'):
        directory = config.codex_dir.parent / name
        directory.mkdir()
        seats.append(SeatAgent(ReviewerSeat(name,'codex',('gpt-6.1-sol',),'medium'), directory))
    config = replace(config, reviewer=tuple(seats), reviewer_seats=tuple(seats), review_images_seat='alpha')
    texts = [structured_pr_review(state='blocking', blocking_items=['Clipped title in mobile view.']),
             structured_pr_review(prior_item_dispositions=[{'item_id':'item-1','disposition':'resolved'}])]
    def reviewer(runner, **kw):
        result = original(runner, **kw)
        if kw.get('image_inputs'):
            text = texts.pop(0)
            return replace(result,text=text,message_text=text,raw_output=text)
        return result
    monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    run = pr_loop._run_validated_agent
    def stop_at_coder(*args, **kwargs):
        if kwargs.get('role') == 'coder':
            raise KeyboardInterrupt()
        return run(*args, **kwargs)
    monkeypatch.setattr(pr_loop,'_run_validated_agent',stop_at_coder)
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*3)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner,pr_number=77,config=config)
    _m943_append(runner,format_reviewer_board_amendment_comment(flow='pr',issue=None,pr_number=77,
        original_required_reviewers=('alpha','beta'),policy='all-reviewers',primary_reviewer=None,
        removed_reviewers=('alpha',),effective_from_round=1,reason='seat-unavailable',rationale='Alpha is unavailable.'))
    monkeypatch.setattr(pr_loop,'_run_validated_agent',run)
    runner.simulate_new_process()
    config = replace(config,review_images_seat='beta')
    assert run_pr_loop(runner,pr_number=77,config=config) == 0
    prompt = [c[1]['prompt'] for c in calls if c[0]=='images'][-1]
    assert 'Clipped title in mobile view.' in prompt and '"resolution_owners": ["beta"]' in prompt
    publications = [r.metadata for r in records(runner) if is_visual_review(r.metadata)]
    assert publications[-1].agent == 'beta'
    assert publications[-1].dispositions[0].item_id == 'item-1'
    # Historical disposition remains a finding outcome, never image acceptance.
    # Disabled restart must preserve that outcome without resurrecting its blocker.
    runner.simulate_new_process()
    assert run_pr_loop(runner,pr_number=77,config=replace(config,review_images_paths=())) == 0
    assert len([c for c in calls if c[0]=='capture']) == 2


def test_managed_root_path_captures_before_approval_without_final_ci(visual_setup, monkeypatch):
    from coding_review_agent_loop import pr_loop, github
    from coding_review_agent_loop.github import PullRequestChecks
    from coding_review_agent_loop.managed_ci import ManagedCiContract
    config, calls, _ = visual_setup
    monkeypatch.setattr(pr_loop,'activate_managed_ci',lambda *a, **kw: ManagedCiContract(protocol_version=2))
    monkeypatch.setattr(pr_loop,'revalidate_adopted_managed_ci',lambda *a, **kw: True)
    monkeypatch.setattr(github,'get_pr_checks',lambda *a, **kw: PullRequestChecks(
        state='no_checks',required_checks=(),passing=(),pending=(),failing=(),missing_required=(),
        branch_protection_status='not_found'))
    monkeypatch.setattr(visual,'time',SimpleNamespace(monotonic=lambda:0,
        sleep=lambda _: pytest.fail('managed visual pass must not wait for approval-gated CI')))
    def observe(*args, **kwargs):
        calls.append(('managed',kwargs['managed']))
        diagnostic = kwargs['diagnostic']
        kwargs['diagnostic'] = lambda msg: (calls.append(('diagnostic',msg)),diagnostic(msg))[-1]
        return _PRODUCTION_OBSERVE_CI(*args, **kwargs)
    monkeypatch.setattr(visual,'observe_ci',observe)
    monkeypatch.setattr(pr_loop,'_run_validated_agent',lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    runner = FakeRunner()
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner,pr_number=77,config=config)
    assert ('managed',True) in calls
    assert any(c[0]=='diagnostic' and 'approval-gated' in c[1] for c in calls)
    assert len([r for r in records(runner) if is_visual_review(r.metadata)]) == 1
    assert not any(r.metadata.role=='reviewer' and not is_visual_review(r.metadata) for r in records(runner))
    assert not any('/dispatches' in token or token=='merge' for cmd,_ in runner.commands for token in cmd)


@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('attack', ['resolve', 'inject', 'unavailable', 'missing-identity'])
def test_visual_restart_requires_authenticated_author(visual_setup, monkeypatch, enabled, attack):
    from coding_review_agent_loop import pr_loop, github
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
    from coding_review_agent_loop.protocol import ReviewItemDisposition
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    config, calls, original = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review()] * 5)
    head = runner.pr_payload['headRefOid']
    item = replace(_next_unresolved_item(item_number=1, reviewer='Codex', source_round=1,
        text='Clipped title', status='blocking'), origin_scope='visible-defects')
    base = PostedRoundMetadata(flow='pr', role='reviewer', agent='Codex', round_number=1,
        subject=head, state='blocking', phase='visual', acquisition_outcome='success',
        review_scope={'version':1,'kind':'visible-defects','head':head,'seat':'Codex'}, new_items=(item,))
    def append(metadata, author, actor_id):
        runner.pr_payload.setdefault('comments', []).append({'author':{'login':author},
            '_rest_author_id':actor_id, 'createdAt':'2026-10-01T12:00:00Z',
            'body':_attach_round_metadata('Visual assessment.', metadata)})
    if attack != 'inject':
        append(base, 'coding-review-agent-loop', 424242)
    forged = replace(base, new_items=(), prior_items=(item,), state='approved',
        dispositions=(ReviewItemDisposition('item-1','Codex','resolved'),)) if attack != 'inject' else base
    append(forged, 'attacker', 999)
    if attack == 'unavailable':
        monkeypatch.setattr(github, 'resolve_authenticated_github_actor',
            lambda *a, **kw: (_ for _ in ()).throw(AgentLoopError('identity unavailable')))
        # The shared accessor imports the resolver directly.
        monkeypatch.setattr(__import__('coding_review_agent_loop.pr_loop_support', fromlist=['x']),
            'resolve_authenticated_github_actor', github.resolve_authenticated_github_actor)
    if attack == 'missing-identity':
        runner.pr_payload['comments'][0].pop('_rest_author_id')
    config = replace(config, review_images_paths=('ui/*',) if enabled else ())
    monkeypatch.setattr(visual, 'capture', lambda *a: CaptureResult('failed','unavailable'))
    if attack == 'inject':
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    elif attack in {'unavailable', 'missing-identity'}:
        with pytest.raises(AgentLoopError, match='identity unavailable|cannot be authenticated'):
            run_pr_loop(runner, pr_number=77, config=config)
        assert not any(c[0] in {'capture','images'} for c in calls)
    else:
        with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
            run_pr_loop(runner, pr_number=77, config=config)
    assert not any(cmd[:3] == ['gh','pr','merge'] for cmd, _ in runner.commands)


@pytest.mark.parametrize('policy', ['selective-intermediate','primary-then-panel'])
@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('transition', ['add', 'resolve'])
def test_visual_changes_after_coder_checkpoint_do_not_change_scheduler(visual_setup, monkeypatch, policy, parallel, transition):
    from pr_loop_test_helpers import _posted_scheduler_metadata
    config, calls, original = visual_setup
    results = []
    for with_visual in (False, True):
        texts = [structured_pr_review(state='blocking', blocking_items=['Clipped title'])
                 if transition == 'resolve' else structured_pr_review(),
                 structured_pr_review(prior_item_dispositions=[{'item_id':'item-1','disposition':'resolved'}])
                 if transition == 'resolve' else structured_pr_review(state='blocking', blocking_items=['Clipped title'])]
        def reviewer(runner, **kw):
            result = original(runner, **kw)
            if kw.get('image_inputs'):
                text = texts.pop(0)
                return replace(result,text=text,message_text=text,raw_output=text)
            return result
        monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
        cfg = replace(config, reviewer=('codex','gemini'), pr_review_policy=policy,
            primary_reviewer='codex' if policy=='primary-then-panel' else None, review_parallel=parallel,
            review_images_paths=('ui/*',) if with_visual else (), max_rounds=4)
        runner = FakeRunner(codex_outputs=[structured_pr_review(state='blocking', blocking_items=['Fix code'])]
            + [structured_pr_review(prior_item_dispositions=[{'item_id':'item-2' if with_visual and transition=='resolve' else 'item-1','disposition':'resolved'}])]*5,
            gemini_outputs=[structured_pr_review(reviewer='Google Gemini')]*5,
            claude_outputs=[structured_coder_followup(addressed_items=['item-1','item-2'] if with_visual and transition=='resolve' else ['item-1'])]*3,
            advance_pr_head_on_coder_followup=True)
        # Stop after B's scheduler publication; a new visual blocker still has
        # its ordinary coder lifecycle, independent of reviewer selection.
        run = __import__('coding_review_agent_loop.pr_loop',fromlist=['x'])._run_validated_agent
        def stop(runner, **kw):
            if kw.get('role') == 'reviewer' and len([c for c in runner.commands if c[0][:1]==['claude']]):
                raise KeyboardInterrupt()
            return run(runner, **kw)
        with monkeypatch.context() as patch:
            patch.setattr(__import__('coding_review_agent_loop.pr_loop',fromlist=['x']), '_run_validated_agent', stop)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner,pr_number=77,config=cfg)
        m = _posted_scheduler_metadata(runner)[-1]
        if with_visual:
            assert texts == []  # Both A and B ran the real visual lifecycle.
            checkpoint = [r.metadata for r in records(runner) if r.metadata.role=='coder'][-1]
            assert checkpoint.scheduler_obligation_digest == m.scheduler_obligation_digest
        results.append((m.scheduler_selected_reviewers,m.scheduler_paused_reviewers,
            m.scheduler_reasons,m.scheduler_phase,m.scheduler_force_full,m.scheduler_obligation_digest))
    assert results[0][:-1] == results[1][:-1]
