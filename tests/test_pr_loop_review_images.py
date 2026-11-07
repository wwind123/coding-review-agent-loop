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
from coding_review_agent_loop.review_scope import is_visual_review, visual_candidate
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
                               contract_id='stub', executable_digest='stub', model_chain=(invocation.configured_model,),
                               permission_mode=cfg.agent_permissions, version='stub-v1')
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
        value['images'][0]['state'] = head  # Default fixture represents changed captures across heads.
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
def test_restart_recovers_terminal_without_recapture(visual_setup, monkeypatch, success):
    config, calls, _ = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review(), structured_pr_review()])
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='capture']) == 1
    before = len([r for r in records(runner) if is_visual_review(r.metadata)])
    if not success:
        monkeypatch.setattr(visual, 'capture', lambda *a: calls.append(('capture','failed')) or CaptureResult('failed','restart'))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='capture']) == 1
    assert len([r for r in records(runner) if is_visual_review(r.metadata)]) == before


@pytest.mark.parametrize('policy,parallel', [('all-reviewers',False),('all-reviewers',True),('primary-then-panel',False)])
@pytest.mark.parametrize('at', ['listing', 'capture', 'acquisition', 'images', 'publication'])
@pytest.mark.parametrize('revisit', [False, True])
def test_stale_head_result_is_discarded_and_next_head_is_attempted(visual_setup, monkeypatch, at, policy, parallel, revisit):
    config, calls, original = visual_setup
    config=replace(config,reviewer=('codex','gemini') if policy=='primary-then-panel' else ('codex',),
        pr_review_policy=policy,primary_reviewer='codex' if policy=='primary-then-panel' else None,review_parallel=parallel, max_rounds=5)
    from test_pr_loop_evidence_conversion import _push_external_head
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*5,
                        gemini_outputs=[structured_pr_review(reviewer='Google Gemini')]*3)
    first_head = runner.pr_payload['headRefOid']
    if revisit:
        attempt = visual.VisualRun.attempt
        def return_to_first_head(self, runner, **kwargs):
            if kwargs['metadata'].head_sha != first_head:
                _push_external_head(runner, first_head)
            return attempt(self, runner, **kwargs)
        monkeypatch.setattr(visual.VisualRun, 'attempt', return_to_first_head)
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
    elif at=='acquisition':
        from contextlib import contextmanager
        from coding_review_agent_loop import review_images_transport
        acquire=review_images_transport.acquire_images
        @contextmanager
        def acquiring(*a,**kw):
            with acquire(*a,**kw) as inputs:
                move()
                yield inputs
        monkeypatch.setattr(review_images_transport,'acquire_images',acquiring)
    elif at=='publication':
        publish_assessment=visual.publish_assessment
        def publish(*a,**kw):
            move()
            return publish_assessment(*a,**kw)
        monkeypatch.setattr(visual,'publish_assessment',publish)
    else:
        def reviewer(runner, **kw):
            result = original(runner, **kw)
            if kw.get('image_inputs'):
                move()
            return result
        monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    pubs = [r for r in records(runner) if is_visual_review(r.metadata)]
    if revisit:
        assert not pubs
        ordinary = [r for r in records(runner) if r.metadata.role == 'reviewer'
                    and not visual_candidate(r.metadata)]
        assert ordinary and ordinary[-1].metadata.subject == first_head
        assert ordinary[-1].metadata.state == 'approved'
        assert len([c for c in calls if c[0] == 'capture' and c[1] == first_head]) <= 1
        assert len([c for c in calls if c[0] == 'images']) <= 1
    else:
        assert len(pubs) == 1 and pubs[0].metadata.subject == '6946a6f1ac1c65132a6a53600572ceb4af0ee23b'
    assert len([c for c in calls if c[0]=='capture']) == (int(at != 'listing') if revisit else (1 if at == 'listing' else 2))


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

    for _ in range(2):
        runner.simulate_new_process()
        with pytest.raises(HumanDecisionRequiredError,match='visual findings item-1'):
            run_pr_loop(runner,pr_number=77,config=config)
    assert len([c for c in calls if c[0]=='images'])==1
    assert len([cmd for cmd,_ in runner.commands if cmd[:1]==['claude']])==1


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
    assert len([c for c in calls if c[0]=='capture']) == 1


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
    selected = seats[0] if opportunity == 'assigned' else seats[1]
    # Keep the scheduling board fixed; vary only the selected visual seat.
    config = replace(config, review_images_seat=str(selected))
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not coder_calls
    assert len([c for c in calls if c[0] == 'images']) == 1
    assert len([c for c in calls if c[0] == 'capture']) == 1
    recovered = visual.recover_visual_findings(visual.load_history(runner, config, 77), reconciliation_mode='owner-scoped')
    assert len(recovered) == 1 and recovered[0].item_id == 'item-1'
    assert recovered[0].status == status
    assert not any(cmd[:3] == ['gh','pr','merge'] for cmd, _ in runner.commands)


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('status', ['blocking', 'same-pr'])
@pytest.mark.parametrize('disposition', ['resolved', 'future', 'keep-open'])
@pytest.mark.parametrize('policy', ['all-reviewers', 'selective-intermediate'])
def test_amended_multi_owner_visual_stop_requires_pending_selected_owner(
        visual_setup, monkeypatch, parallel, status, disposition, policy):
    from coding_review_agent_loop import pr_loop
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from pr_loop_test_helpers import _m943_append
    config, calls, original = visual_setup
    seats = []
    for name in ('alpha', 'beta', 'gamma'):
        directory = config.codex_dir.parent / name
        directory.mkdir()
        seats.append(SeatAgent(ReviewerSeat(name, 'codex', ('gpt-6.1-sol',), 'medium'), directory))
    config = replace(config, reviewer=tuple(seats), reviewer_seats=tuple(seats),
        review_images_seat='alpha', pr_review_policy=policy,
        primary_reviewer=None, review_parallel=parallel)
    kept = disposition == 'keep-open'
    texts = [structured_pr_review(state='blocking', **{
        'blocking_items' if status == 'blocking' else 'same_pr_followups': ['Clipped mobile title.']}),
        structured_pr_review(state='blocking' if kept else 'approved', prior_item_dispositions=[{
            'item_id': 'item-1', 'disposition': status if kept else disposition,
            'note': 'The mobile title is still clipped in the sealed image; widen its container and test the mobile viewport.'}])]
    def reviewer(runner, **kwargs):
        result = original(runner, **kwargs)
        if kwargs.get('image_inputs'):
            text = texts.pop(0)
            return replace(result, text=text, message_text=text, raw_output=text)
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    reconciled = []
    settle = pr_loop._settle_pr_round_ledger
    def record_settlement(*args, **kwargs):
        result = settle(*args, **kwargs)
        reconciled.append(result[0])
        return result
    monkeypatch.setattr(pr_loop, '_settle_pr_round_ledger', record_settlement)
    run = pr_loop._run_validated_agent
    coder_calls = []
    def stop_at_coder(*args, **kwargs):
        if kwargs.get('role') == 'coder':
            coder_calls.append(kwargs)
            raise KeyboardInterrupt()
        return run(*args, **kwargs)
    monkeypatch.setattr(pr_loop, '_run_validated_agent', stop_at_coder)
    runner = FakeRunner(codex_outputs=[structured_pr_review()] * 10)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(coder_calls) == 1
    coder_calls.clear()
    _m943_append(runner, format_reviewer_board_amendment_comment(flow='pr', issue=None, pr_number=77,
        original_required_reviewers=('alpha', 'beta', 'gamma'), policy=policy,
        primary_reviewer=None, removed_reviewers=('alpha',), effective_from_round=1,
        reason='seat-unavailable', rationale='Alpha is unavailable.'))
    config = replace(config, review_images_seat='beta')
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not coder_calls
    assert len([c for c in calls if c[0] == 'images']) == 1
    assert len([c for c in calls if c[0] == 'capture']) == 1
    retained, = [item for item in reconciled[-1] if item.item_id == 'item-1']
    assert retained.status == status
    assert retained.resolution_owners == ('beta', 'gamma')
    assert dict(retained.owner_states) == {'beta': 'pending', 'gamma': 'pending'}
    assert not any(cmd[:3] == ['gh','pr','merge'] for cmd, _ in runner.commands)


def test_amended_visual_owner_cannot_reopen_spent_head(visual_setup, monkeypatch):
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
    config = replace(config,review_images_seat='beta')
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
        run_pr_loop(runner,pr_number=77,config=config)
    assert len([c for c in calls if c[0]=='images']) == 1
    assert len([c for c in calls if c[0]=='capture']) == 1


def test_managed_root_path_captures_before_approval_without_final_ci(visual_setup, monkeypatch):
    from coding_review_agent_loop import pr_loop, github
    from coding_review_agent_loop.github import PullRequestChecks
    from coding_review_agent_loop.managed_ci_types import (
        ManagedCiContract,
    )
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
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata,
        _attach_round_metadata,
    )
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


@pytest.mark.parametrize('policy,parallel', [('all-reviewers',False),('all-reviewers',True),('primary-then-panel',False)])
@pytest.mark.parametrize('crash', ['none','before-intent','after-intent','after-launch','before-result'])
def test_durable_dispatch_order_and_crash_resume(visual_setup, monkeypatch, policy, parallel, crash):
    from coding_review_agent_loop import github
    config, calls, original = visual_setup
    config = replace(config, reviewer=('codex','gemini') if policy=='primary-then-panel' else ('codex',),
                     pr_review_policy=policy, primary_reviewer='codex' if policy=='primary-then-panel' else None,
                     review_parallel=parallel)
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*8,
                        gemini_outputs=[structured_pr_review(reviewer='Google Gemini')]*8)
    persist = visual.persist_lifecycle
    def intent(*args, **kwargs):
        if crash == 'before-intent': raise KeyboardInterrupt()
        written = persist(*args, **kwargs)
        calls.append(('intent', written.comment_id))
        if crash == 'after-intent': raise KeyboardInterrupt()
        return written
    monkeypatch.setattr(visual, 'persist_lifecycle', intent)
    def reviewer(runner, **kwargs):
        if kwargs.get('image_inputs'):
            assert any(c[0]=='intent' for c in calls)
        result = original(runner, **kwargs)
        if kwargs.get('image_inputs') and crash == 'after-launch': raise KeyboardInterrupt()
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    publish = github.post_verified_trusted_pr_round_comment
    if crash == 'before-result':
        monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment',
                            lambda *a, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    if crash != 'none':
        with pytest.raises(KeyboardInterrupt): run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    before = len([c for c in calls if c[0]=='images'])
    monkeypatch.setattr(visual, 'persist_lifecycle', persist)
    monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment', publish)
    monkeypatch.setattr(validated_agent, 'run_agent_result', original)
    runner.simulate_new_process()
    for _ in range(2): assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert len([c for c in calls if c[0]=='images']) == before + int(crash == 'before-intent')
    assert len([c for c in calls if c[0]=='images']) <= 1
    assert len([c for c in calls if c[0]=='capture']) == (2 if crash=='before-intent' else 1)


def _append_lifecycle(runner, *, value=None, head=None, foreign=False):
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata,
        _attach_round_metadata,
    )
    head = runner.pr_payload['headRefOid'] if head is None else head
    m = PostedRoundMetadata(flow='pr', role='reviewer', agent='OpenAI Codex', round_number=1,
                            subject=head, visual_lifecycle=value, visual_lifecycle_present=True)
    runner.pr_payload.setdefault('comments', []).append({'author': {'login':'attacker' if foreign else 'coding-review-agent-loop'},
        '_rest_author_id':999 if foreign else 424242, 'createdAt':'2026-10-01T00:00:00Z',
        'body': _attach_round_metadata('Visual lifecycle audit.',m)})


@pytest.mark.parametrize('mode', ['enabled','disabled','dry-run','nonmatching'])
@pytest.mark.parametrize('foreign', [False,True])
@pytest.mark.parametrize('value', [None, {'version':99}, []])
def test_lifecycle_authentication_precedes_validation_and_obeys_applicability(visual_setup, monkeypatch, mode, foreign, value):
    config, calls, _ = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*5)
    _append_lifecycle(runner, value=value, foreign=foreign)
    if mode == 'disabled': config = replace(config,review_images_paths=())
    elif mode == 'dry-run':
        config = replace(config,dry_run=True)
        _supply_dry_run_history(monkeypatch)
    elif mode == 'nonmatching':
        from coding_review_agent_loop import github_changed_paths
        monkeypatch.setattr(github_changed_paths,'get_pr_changed_paths',lambda *a, **kw:
            PullRequestChangedPaths(kw['head_sha'],True,(ChangedPath('modified','server.py'),)))
    if mode=='enabled' and not foreign:
        with pytest.raises(HumanDecisionRequiredError, match='comment') as exc: run_pr_loop(runner,pr_number=77,config=config)
        assert any(str(exc.value) in body for body in runner.comments)
    else:
        assert run_pr_loop(runner,pr_number=77,config=config)==0
    assert len([c for c in calls if c[0]=='images']) == int(mode=='enabled' and foreign)
    assert len([c for c in calls if c[0]=='capture']) == int(mode=='enabled' and foreign)
    if mode in {'disabled','dry-run'}: assert calls == []


@pytest.mark.parametrize('kind', ['old','unknown','incomplete','legacy'])
def test_authenticated_history_admission_at_production_seam(visual_setup, kind):
    config, calls, _ = visual_setup
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*5)
    if kind=='legacy':
        from review_images_helpers import append_visual_record
        append_visual_record(runner)
    else:
        _append_lifecycle(runner,head='a'*40 if kind=='old' else 'unknown' if kind=='unknown' else None)
        if kind=='incomplete': runner.pr_payload['comments'][0].pop('_rest_author_id')
    if kind in {'unknown','incomplete'}:
        with pytest.raises(HumanDecisionRequiredError): run_pr_loop(runner,pr_number=77,config=config)
    else:
        assert run_pr_loop(runner,pr_number=77,config=config)==0
    assert len([c for c in calls if c[0]=='images']) == int(kind=='old')


@pytest.mark.parametrize('blocking', [False,True])
def test_fresh_capture_skip_chain_preserves_source_blockers_without_coder(visual_setup, monkeypatch, blocking):
    from test_pr_loop_evidence_conversion import _push_external_head
    from coding_review_agent_loop import pr_loop
    config,calls,original = visual_setup
    capture = visual.capture
    def unchanged(checkout, head, settings, changes):
        # Independent head bindings, identical validated manifest/image content.
        calls.append(('capture',head))
        value=manifest(head)
        snap=publish(json.dumps(value).encode(),head,settings.limits,{'one.png':png(),'base.png':png()},
                     {'one.png':(2,2),'base.png':(2,2)},head)
        return CaptureResult('success',snapshots=snap)
    monkeypatch.setattr(visual,'capture',unchanged)
    def reviewer(runner,**kwargs):
        result=original(runner,**kwargs)
        if blocking and kwargs.get('image_inputs'):
            text=structured_pr_review(state='blocking',blocking_items=['Source-raised clipped title'])
            return replace(result,text=text,message_text=text,raw_output=text)
        return result
    monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    run=pr_loop._run_validated_agent
    coder=[]
    def stop(*a,**kw):
        if kw.get('role')=='coder': coder.append(1); raise KeyboardInterrupt()
        return run(*a,**kw)
    monkeypatch.setattr(pr_loop,'_run_validated_agent',stop)
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*12)
    if blocking:
        with pytest.raises(KeyboardInterrupt): run_pr_loop(runner,pr_number=77,config=config)
    else: assert run_pr_loop(runner,pr_number=77,config=config)==0
    coder.clear()
    for head in ('b'*40,'c'*40):
        # Exercise visual recovery independently of interrupted code-recovery checkpoints.
        runner.pr_payload['comments'] = [c for c in runner.pr_payload['comments'] if any(
            visual_candidate(r.metadata) for r in _extract_round_metadata_records([SimpleNamespace(body=c['body'])],flow='pr'))]
        _push_external_head(runner,head)
        for _ in range(2):
            if blocking:
                with pytest.raises(HumanDecisionRequiredError,match='visual findings item-1'):
                    run_pr_loop(runner,pr_number=77,config=config)
            else: assert run_pr_loop(runner,pr_number=77,config=config)==0
    lifecycle=[r.metadata.visual_lifecycle for r in records(runner) if r.metadata.visual_lifecycle_present]
    assert [v['kind'] for v in lifecycle]==['intent','result','skip','skip']
    assert lifecycle[-1]['original_id']==lifecycle[-2]['original_id']
    assert lifecycle[-1]['capture']!=lifecycle[-2]['capture']
    assert lifecycle[-1]['comparison']==lifecycle[-2]['comparison']
    assert len([c for c in calls if c[0]=='capture'])==3
    assert len([c for c in calls if c[0]=='images'])==1
    assert coder==[]


@pytest.mark.parametrize('failure', ['capture','paths','diff'])
def test_preintent_work_once_per_invocation_but_restart_can_retry(visual_setup, monkeypatch, failure):
    config,calls,_=visual_setup
    runner=FakeRunner(codex_outputs=[structured_pr_review(state='blocking',blocking_items=['Fix code']),
                                    structured_pr_review(prior_item_dispositions=[{'item_id':'item-1','disposition':'resolved'}])]*2,
        claude_outputs=[structured_coder_followup(addressed_items=['item-1'])]*2,
        advance_pr_head_on_coder_followup=False)
    if failure=='capture':
        monkeypatch.setattr(visual,'capture',lambda *a: calls.append(('failed-capture',a[1])) or CaptureResult('failed','stub'))
    elif failure=='diff':
        monkeypatch.setattr(visual,'pinned_diff',lambda *a: calls.append(('failed-diff',a[2].head_sha)) or (_ for _ in ()).throw(ValueError('stub')))
    else:
        from coding_review_agent_loop import github_changed_paths
        monkeypatch.setattr(github_changed_paths,'get_pr_changed_paths',lambda *a,**kw:
            calls.append(('failed-paths',kw['head_sha'])) or PullRequestChangedPaths(kw['head_sha'],False,diagnostic='stub'))
    assert run_pr_loop(runner,pr_number=77,config=config)==0
    key={'capture':'failed-capture','paths':'failed-paths','diff':'failed-diff'}[failure]
    assert len([c for c in calls if c[0]==key])==1
    runner.simulate_new_process()
    assert run_pr_loop(runner,pr_number=77,config=config)==0
    assert len([c for c in calls if c[0]==key])==2
    assert not any(c[0]=='images' for c in calls)


@pytest.mark.parametrize('mode', ['enabled','disabled','dry-run','nonmatching'])
@pytest.mark.parametrize('foreign', [False,True])
@pytest.mark.parametrize('broken', ['envelope','shared-payload','missing-sidecar'])
def test_shared_transport_failure_stops_before_visual_admission(visual_setup, monkeypatch, mode, foreign, broken):
    from test_round_transport import _comment, _random_text
    from coding_review_agent_loop import round_transport, github_changed_paths
    config,calls,_=visual_setup
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*3)
    value={'flow':'pr','role':'reviewer','agent':'Codex','round_number':1,'subject':runner.pr_payload['headRefOid']}
    if broken=='envelope':
        body=_comment(value).replace(round_transport.encode_mapping(value),'undecodable')
    elif broken=='shared-payload': body=_comment(dict(value,round_number='invalid'))
    else:
        value['canonical_reviewer_response']=_random_text(70000)
        prepared=round_transport.prepare_round_comment(_comment(value))
        assert len(prepared)>1
        body=prepared[-1]
    runner.pr_payload['comments']=[{'body':body,'author':{'login':'attacker' if foreign else 'coding-review-agent-loop'},
        '_rest_author_id':999 if foreign else 424242,'createdAt':'2026-10-01T00:00:00Z'}]
    if mode=='disabled': config=replace(config,review_images_paths=())
    elif mode=='dry-run':
        config=replace(config,dry_run=True)
        _supply_dry_run_history(monkeypatch)
    elif mode=='nonmatching':
        monkeypatch.setattr(github_changed_paths,'get_pr_changed_paths',lambda *a,**kw:
            PullRequestChangedPaths(kw['head_sha'],True,(ChangedPath('modified','server.py'),)))
    with pytest.raises(AgentLoopError): run_pr_loop(runner,pr_number=77,config=config)
    assert not any(c[0] in {'capture','images','ci','diff'} for c in calls)
    assert len(runner.pr_payload['comments'])==1
    from coding_review_agent_loop.code_approval import (
        _latest_pr_approved_reviews_for_head,
    )
    with pytest.raises(AgentLoopError): _latest_pr_approved_reviews_for_head([SimpleNamespace(body=body)],head_sha=value['subject'],configured_reviewers=('codex',))


@pytest.mark.parametrize('failure', ['before-anchor','readback','lost-response'])
def test_visual_publication_recovery_never_relaunches_or_mutates_ledger_early(visual_setup, monkeypatch, failure):
    from coding_review_agent_loop import github
    config,calls,original=visual_setup
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*6)
    def reviewer(runner,**kw):
        result=original(runner,**kw)
        if kw.get('image_inputs'):
            text=structured_pr_review(state='blocking',blocking_items=['Publication blocker'])
            return replace(result,text=text,message_text=text,raw_output=text)
        return result
    monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    writer=github.post_verified_trusted_pr_round_comment
    def broken(*a,**kw):
        if failure!='before-anchor': writer(*a,**kw)
        raise AgentLoopError('read-back unavailable' if failure=='readback' else 'write response lost')
    monkeypatch.setattr(github,'post_verified_trusted_pr_round_comment',broken)
    # An unpublished assessment has no in-memory finding authority or coder opportunity.
    assert run_pr_loop(runner,pr_number=77,config=config)==0
    assert not any(cmd[:1]==['claude'] for cmd,_ in runner.commands)
    monkeypatch.setattr(github,'post_verified_trusted_pr_round_comment',writer)
    runner.simulate_new_process()
    if failure=='before-anchor': assert run_pr_loop(runner,pr_number=77,config=config)==0
    else:
        with pytest.raises(HumanDecisionRequiredError,match='visual findings item-1'):
            run_pr_loop(runner,pr_number=77,config=config)
    assert len([c for c in calls if c[0]=='images'])==1
    assert len([c for c in calls if c[0]=='capture'])==1
    assert not any(cmd[:1]==['claude'] for cmd,_ in runner.commands)


@pytest.mark.parametrize('applicable', ['enabled','disabled','dry-run','nonmatching'])
@pytest.mark.parametrize('history', ['absent','current-invalid','old-invalid','unknown'])
def test_final_qualification_uses_same_applicable_lifecycle_validation(visual_setup, monkeypatch, applicable, history):
    from coding_review_agent_loop.pr_loop_support import _fresh_pr_qualification_snapshot
    from coding_review_agent_loop import github_changed_paths
    config,calls,_=visual_setup
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*3)
    if history!='absent': _append_lifecycle(runner,head='a'*40 if history=='old-invalid' else 'unknown' if history=='unknown' else None)
    if applicable=='disabled': config=replace(config,review_images_paths=())
    elif applicable=='dry-run':
        config=replace(config,dry_run=True)
        _supply_dry_run_history(monkeypatch)
    elif applicable=='nonmatching':
        monkeypatch.setattr(github_changed_paths,'get_pr_changed_paths',lambda *a,**kw:
            PullRequestChangedPaths(kw['head_sha'],True,(ChangedPath('modified','server.py'),)))
    if applicable=='enabled' and history in {'current-invalid','unknown'}:
        with pytest.raises(HumanDecisionRequiredError): _fresh_pr_qualification_snapshot(runner,config=config,pr_number=77,issue_context=None,parent_issue_context=None)
    else:
        context,_,_,_=_fresh_pr_qualification_snapshot(runner,config=config,pr_number=77,issue_context=None,parent_issue_context=None)
        assert context.metadata.head_sha==runner.pr_payload['headRefOid']
    assert not any(c[0] in {'images','capture','ci','diff'} for c in calls)


def _supply_dry_run_history(monkeypatch):
    """Supply a decoded snapshot: normal dry-run's synthetic GitHub view is empty."""
    from coding_review_agent_loop import pr_loop, pr_loop_support
    from coding_review_agent_loop.pr_head_sync import get_observed_pr_review_context
    def observed(runner, *, config, pr_number, **kwargs):
        return get_observed_pr_review_context(runner, config=replace(config,dry_run=False),pr_number=pr_number,**kwargs)
    monkeypatch.setattr(pr_loop,'get_observed_pr_review_context',observed)
    monkeypatch.setattr(pr_loop_support,'get_observed_pr_review_context',observed)


@pytest.mark.parametrize('change', ['contract','configuration','latest-source','uncovered-assignment'])
def test_new_head_reuse_mismatch_assesses_once_and_never_searches_older_sources(visual_setup, monkeypatch, change):
    from test_pr_loop_evidence_conversion import _push_external_head
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    config,calls,original=visual_setup
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*12,
        claude_outputs=[structured_coder_followup(addressed_items=['item-1'])],advance_pr_head_on_coder_followup=False)
    initial=runner.pr_payload['headRefOid']
    def capture(checkout,head,settings,changes):
        calls.append(('capture',head))
        value=manifest(head)
        if change=='latest-source' and head=='b'*40: value['images'][0]['locale']='fr'
        snap=publish(json.dumps(value).encode(),head,settings.limits,{'one.png':png(),'base.png':png()},
                     {'one.png':(2,2),'base.png':(2,2)},head)
        return CaptureResult('success',snapshots=snap)
    monkeypatch.setattr(visual,'capture',capture)
    assert run_pr_loop(runner,pr_number=77,config=config)==0
    if change=='contract':
        preflight=visual.preflight_images
        monkeypatch.setattr(visual,'preflight_images',lambda *a,**kw: SimpleNamespace(**dict(vars(preflight(*a,**kw)),version='stub-v2')))
    elif change=='configuration': config=replace(config,review_images_timeout=7)
    elif change=='uncovered-assignment':
        replay=visual.recover_visual_findings
        item=replace(_next_unresolved_item(item_number=9,reviewer='Codex',source_round=1,
            text='Transferred pending visual share',status='blocking'),origin_scope='visible-defects')
        monkeypatch.setattr(visual,'recover_visual_findings',lambda records,**kw:(*replay(records,**kw),item))
        def reviewer(runner,**kw):
            result=original(runner,**kw)
            if kw.get('image_inputs'):
                text=structured_pr_review(prior_item_dispositions=[{'item_id':'item-9','disposition':'resolved'}])
                return replace(result,text=text,message_text=text,raw_output=text)
            return result
        monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    if change=='latest-source':
        def reviewer(runner,**kw):
            result=original(runner,**kw)
            if kw.get('image_inputs'):
                head=kw['image_inputs'].binding.head_sha
                text=(structured_pr_review(state='blocking',blocking_items=['Changed B capture clips the title'])
                      if head=='b'*40 else structured_pr_review(prior_item_dispositions=[{'item_id':'item-1','disposition':'resolved'}]))
                return replace(result,text=text,message_text=text,raw_output=text)
            return result
        monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    _push_external_head(runner,'b'*40)
    if change=='latest-source':
        with pytest.raises(HumanDecisionRequiredError,match='visual findings item-1'):
            run_pr_loop(runner,pr_number=77,config=config)
        # C starts ordinary code review independently of B's completed checkpoint.
        runner.pr_payload['comments']=[c for c in runner.pr_payload['comments'] if any(
            visual_candidate(r.metadata) for r in _extract_round_metadata_records([SimpleNamespace(body=c['body'])],flow='pr'))]
        _push_external_head(runner,'c'*40)
        assert run_pr_loop(runner,pr_number=77,config=config)==0
    else: assert run_pr_loop(runner,pr_number=77,config=config)==0
    assert len([c for c in calls if c[0]=='images'])==(3 if change=='latest-source' else 2)
    assert not any(r.metadata.visual_lifecycle.get('kind')=='skip' for r in records(runner)
                   if isinstance(r.metadata.visual_lifecycle,dict))


@pytest.mark.parametrize('lifecycle', ['repair_required','qualification_ready','qualifying'])
@pytest.mark.parametrize('history', ['absent','invalid','old-invalid','unknown'])
def test_recovery_checkpoint_validates_lifecycle_without_opening_execution(visual_setup, monkeypatch, tmp_path, lifecycle, history):
    from test_pr_loop_managed_ci import _managed_qualifying_resume
    from coding_review_agent_loop import pr_loop
    from coding_review_agent_loop.managed_ci_types import (
        ManagedCiContract,
    )
    from coding_review_agent_loop.round_metadata import (
        _attach_round_metadata,
    )
    visual_config,calls,_=visual_setup
    config,runner=_managed_qualifying_resume(tmp_path,reviewer_codex_model='gpt-6.1-sol')
    config=replace(config,review_images_paths=visual_config.review_images_paths,
        review_images_cmd=visual_config.review_images_cmd,review_images_runtime_grant=visual_config.review_images_runtime_grant,
        review_images_seat='codex')
    for comment in runner.pr_payload['comments']:
        r=_extract_round_metadata_records([SimpleNamespace(body=comment['body'])],flow='pr')[0]
        m=r.metadata
        if m.qualification_checkpoint is not None:
            m=replace(m,qualification_checkpoint=replace(m.qualification_checkpoint,lifecycle=lifecycle,
                failed_head_sha=m.subject if lifecycle=='repair_required' else 'old-head',
                candidate_head_sha=None if lifecycle=='repair_required' else m.subject),
                prior_items=tuple(replace(item,lifecycle=lifecycle,
                    failed_head_sha=m.subject if lifecycle=='repair_required' else 'old-head',
                    candidate_head_sha=None if lifecycle=='repair_required' else m.subject) for item in m.prior_items))
            comment['body']=_attach_round_metadata('Recovery checkpoint.',m)
    if history!='absent': _append_lifecycle(runner,head='a'*40 if history=='old-invalid' else 'unknown' if history=='unknown' else None)
    monkeypatch.setattr(pr_loop,'activate_managed_ci',lambda *a,**kw:ManagedCiContract(protocol_version=2))
    monkeypatch.setattr(pr_loop,'revalidate_adopted_managed_ci',lambda *a,**kw:True)
    validate=visual.validate_boundary
    validated=[]
    def boundary(*a,**kw):
        validate(*a,**kw)
        validated.append(1)
        raise KeyboardInterrupt()  # Stop before unrelated qualification actions.
    monkeypatch.setattr(visual,'validate_boundary',boundary)
    if history in {'invalid','unknown'}:
        with pytest.raises(HumanDecisionRequiredError) as exc: run_pr_loop(runner,pr_number=77,config=config)
        assert any(str(exc.value) in body for body in runner.comments)
    else:
        with pytest.raises(KeyboardInterrupt): run_pr_loop(runner,pr_number=77,config=config)
        assert validated==[1]
    assert not any(c[0] in {'capture','images','ci','diff'} for c in calls)


@pytest.mark.parametrize('position', ['before', 'after'])
@pytest.mark.parametrize('mode', ['disabled', 'dry-run', 'nonmatching'])
@pytest.mark.parametrize('corruption', ['unknown-subject', 'contradictory-lifecycle', 'contradictory-scope'])
def test_lifecycle_corruption_cannot_erase_verified_findings(visual_setup, monkeypatch, position, mode, corruption):
    from coding_review_agent_loop import pr_loop
    config,calls,original=visual_setup
    def reviewer(runner,**kwargs):
        result=original(runner,**kwargs)
        if kwargs.get('image_inputs'):
            text=structured_pr_review(state='blocking',blocking_items=['Persist this visual blocker'])
            return replace(result,text=text,message_text=text,raw_output=text)
        return result
    monkeypatch.setattr(validated_agent,'run_agent_result',reviewer)
    run=pr_loop._run_validated_agent
    def stop(*args,**kwargs):
        if kwargs.get('role')=='coder': raise KeyboardInterrupt()
        return run(*args,**kwargs)
    monkeypatch.setattr(pr_loop,'_run_validated_agent',stop)
    runner=FakeRunner(codex_outputs=[structured_pr_review()]*5)
    def corrupt_comment():
        from coding_review_agent_loop.round_metadata import (
            _attach_round_metadata,
        )
        _append_lifecycle(runner, head='unknown' if corruption == 'unknown-subject' else None, foreign=True)
        comment = runner.pr_payload['comments'][-1]
        record = _extract_round_metadata_records([SimpleNamespace(body=comment['body'])], flow='pr')[0]
        metadata = record.metadata
        if corruption == 'contradictory-lifecycle':
            metadata = replace(metadata, visual_lifecycle={'head': 'a'*40})
        elif corruption == 'contradictory-scope':
            metadata = replace(metadata, review_scope={'head': 'a'*40})
        comment['body'] = _attach_round_metadata('Malformed visual audit.', metadata)
        return comment
    if position == 'before': corrupt = corrupt_comment()
    with pytest.raises(KeyboardInterrupt): run_pr_loop(runner,pr_number=77,config=config)
    if position == 'after': corrupt = corrupt_comment()
    corrupt['author']['login'] = 'coding-review-agent-loop'
    corrupt['_rest_author_id'] = 424242
    recovered=visual.recover_visual_findings(visual.load_history(runner,config,77),reconciliation_mode='owner-scoped')
    assert len(recovered)==1 and recovered[0].item_id=='item-1'
    assert recovered[0].text=='Persist this visual blocker'
    assert recovered[0].resolution_owners == ('Codex',)
    if mode == 'dry-run':
        # Supply authenticated history as well as the decoded dry-run snapshot;
        # the normal dry-run GitHub adapter deliberately omits REST identities.
        from coding_review_agent_loop import pr_loop_support
        authenticate = pr_loop_support._authenticated_pr_comments_status
        monkeypatch.setattr(pr_loop_support, '_authenticated_pr_comments_status',
            lambda runner, config, **kw: authenticate(runner, replace(config, dry_run=False), **kw))
    runner.simulate_new_process()
    if mode == 'disabled': config = replace(config, review_images_paths=())
    elif mode == 'dry-run':
        config = replace(config, dry_run=True)
        _supply_dry_run_history(monkeypatch)
    else:
        from coding_review_agent_loop import github_changed_paths
        monkeypatch.setattr(github_changed_paths, 'get_pr_changed_paths', lambda *a, **kw:
            PullRequestChangedPaths(kw['head_sha'], True, (ChangedPath('modified', 'server.py'),)))
    with pytest.raises(HumanDecisionRequiredError,match='visual findings item-1'):
        run_pr_loop(runner,pr_number=77,config=config)
    assert len([c for c in calls if c[0]=='images'])==1
    assert len([c for c in calls if c[0]=='capture'])==1


@pytest.mark.parametrize('policy,parallel', [('all-reviewers', False), ('all-reviewers', True),
                                           ('primary-then-panel', False)])
def test_head_moves_after_verified_publication_retains_blockers_and_ids(visual_setup, monkeypatch, policy, parallel):
    from coding_review_agent_loop import github
    from test_pr_loop_evidence_conversion import _push_external_head
    config, calls, original = visual_setup
    config = replace(config, approved_followups='summarize', review_parallel=parallel, pr_review_policy=policy,
        reviewer=('codex', 'gemini') if policy == 'primary-then-panel' else ('codex',),
        primary_reviewer='codex' if policy == 'primary-then-panel' else None)
    runner = FakeRunner(codex_outputs=[structured_pr_review()]*5,
                        gemini_outputs=[structured_pr_review(reviewer='Google Gemini')]*5)
    head_a, head_b = runner.pr_payload['headRefOid'], 'b'*40
    def unchanged(checkout, head, settings, changes):
        calls.append(('capture', head))
        snap = publish(json.dumps(manifest(head)).encode(), head, settings.limits,
                       {'one.png': png(), 'base.png': png()},
                       {'one.png': (2, 2), 'base.png': (2, 2)}, head)
        return CaptureResult('success', snapshots=snap)
    monkeypatch.setattr(visual, 'capture', unchanged)
    def reviewer(runner, **kw):
        result = original(runner, **kw)
        if kw.get('image_inputs'):
            text = structured_pr_review(state='blocking', blocking_items=['Published blocker'],
                same_pr_followups=['Published same-PR obligation'], future_followups=['Published future work'])
            return replace(result, text=text, message_text=text, raw_output=text)
        return result
    monkeypatch.setattr(validated_agent, 'run_agent_result', reviewer)
    writer = github.post_verified_trusted_pr_round_comment
    def move_after_write(*a, **kw):
        anchor = writer(*a, **kw)
        # Move only after the real verified writer has returned its final anchor.
        assert records(runner)[-1].metadata.new_items[0].text == 'Published blocker'
        _push_external_head(runner, head_b)
        return anchor
    monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment', move_after_write)
    publication = visual.publish_assessment
    observed = []
    def inspect_publication(*a, **kw):
        result = publication(*a, **kw)
        observed.append((kw['metadata'].head_sha, result))
        return result
    monkeypatch.setattr(visual, 'publish_assessment', inspect_publication)
    with pytest.raises(HumanDecisionRequiredError, match='visual findings item-1'):
        run_pr_loop(runner, pr_number=77, config=config)
    assert [head for head, _ in observed] == [head_a, head_b]
    stale, carried, next_number = observed[0][1]
    skip, skipped_items, skipped_number = observed[1][1]
    assert stale.stale and not stale.fresh and stale.response is None
    assert [i.item_id for i in carried] == ['item-1', 'item-2', 'item-3']
    assert next_number == skipped_number == 4
    assert skipped_items == carried and not skip.fresh and skip.response is None
    lifecycle = [r.metadata.visual_lifecycle for r in records(runner) if r.metadata.visual_lifecycle_present]
    assert [v['kind'] for v in lifecycle] == ['intent', 'result', 'skip']
    assert lifecycle[-1]['head'] == head_b and lifecycle[-1]['comparison'] == lifecycle[-2]['comparison']
    assert [c[1] for c in calls if c[0] == 'capture'] == [head_a, head_b]
    assert len([c for c in calls if c[0] == 'images']) == 1
    replay = visual.recover_visual_findings(visual.load_history(runner, config, 77), reconciliation_mode='owner-scoped')
    assert tuple(carried) == replay
    assert not any(cmd[:1] == ['claude'] or cmd[:3] == ['gh', 'pr', 'merge'] for cmd, _ in runner.commands)


@pytest.mark.parametrize('movement', ['before-write', 'during-write', 'stable'])
def test_publication_head_boundary_retains_findings_without_stale_clearance(visual_setup, monkeypatch, movement):
    from coding_review_agent_loop import github
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    config, _, _ = visual_setup
    config = replace(config, approved_followups='summarize')
    head = 'a'*40
    prior = replace(_next_unresolved_item(item_number=1, reviewer='Codex', source_round=1,
                    text='Existing blocker', status='blocking'), origin_scope='visible-defects')
    text = structured_pr_review(state='blocking', blocking_items=['New blocker'],
        same_pr_followups=['New same-PR work'], future_followups=['New future work'],
        prior_item_dispositions=[{'item_id': 'item-1', 'disposition': 'resolved'}])
    outcome = visual.VisualOutcome(response=AgentResult(text=text, raw_output=text, returncode=0),
        parsed=visual.validate_visual(text, reviewer='Codex', items=(prior,)),
        scope={'version': 1, 'kind': 'visible-defects', 'head': head, 'seat': 'Codex'})
    live = [head if movement != 'before-write' else 'b'*40]
    writes = []
    def writer(*a, **kw):
        writes.append(kw['body'])
        if movement == 'during-write': live[0] = 'b'*40
    monkeypatch.setattr(github, 'post_verified_trusted_pr_round_comment', writer)
    result, items, number = visual.publish_assessment(FakeRunner(), config=config,
        metadata=SimpleNamespace(number=77, head_sha=head), outcome=outcome, items=[prior], next_number=2,
        round_number=2, seat_binding=None, reconciliation_mode='owner-scoped',
        current_head=lambda: live[0], diagnostic=lambda _: None)
    assert len(writes) == int(movement != 'before-write')
    assert result.stale == (movement != 'stable')
    assert result.fresh == (movement == 'stable')
    assert number == (2 if movement == 'before-write' else 5)
    if movement != 'stable': assert items[0] == prior
    else: assert prior.item_id not in {i.item_id for i in items}
    new_items = items if movement == 'stable' else items[1:]
    assert [i.item_id for i in new_items] == ([] if movement == 'before-write' else ['item-2', 'item-3', 'item-4'])
