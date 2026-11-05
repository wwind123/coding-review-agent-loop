"""Recovery reads and authenticated admission replay across lifecycle changes."""
from dataclasses import replace

import pytest

from coding_review_agent_loop import post_merge_runtime as runtime
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueComment, PullRequestReviewContext
from coding_review_agent_loop.post_merge_obligations import active_registry, validate_registry
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from test_post_merge_obligations import evidence, signed, transition
from test_pr_loop_post_merge_obligations import metadata, merged, publication_setup, setup_run


@pytest.mark.parametrize('receipt', [None, 88])
def test_cli_dry_run_uses_production_runner_for_live_reads(tmp_path, monkeypatch, capsys, receipt):
    from coding_review_agent_loop import cli
    real_read_api = runtime.read_api
    run, registry, published, writes = publication_setup(tmp_path, monkeypatch)
    # Execute a harmless gh stand-in through the unmodified production Runner.
    gh = tmp_path / 'gh-read-fixture'
    gh.write_text('#!/usr/bin/env python3\nimport json, sys\n'
                  'assert sys.argv[1] == "api"\n'
                  f'print(json.dumps({merged()!r} if "/pulls/" in sys.argv[2] '
                  'else {"number": 9, "state": "open"}))\n')
    gh.chmod(0o755)
    config = replace(run.config, gh_cmd=str(gh), dry_run=True)
    monkeypatch.setattr(cli, 'config_from_args', lambda *args, **kwargs: config)
    monkeypatch.setattr(runtime, 'read_api', real_read_api)
    monkeypatch.setattr(runtime, 'restore', lambda *args, **kwargs: (registry, ()))
    monkeypatch.setattr(runtime, 'find_publication', lambda *args: receipt)
    monkeypatch.setattr(cli, 'claim_agent_workdirs', lambda *args, **kwargs: pytest.fail('worktree'))
    monkeypatch.setattr(cli, 'run_pr_loop', lambda *args, **kwargs: pytest.fail('review loop'))
    assert cli.main(['pr', '1', '--repo', config.repo,
                     '--reconcile-post-merge-obligations', '--dry-run']) == 0
    assert ('tracking-pending' if receipt is None else 'tracked') in capsys.readouterr().out
    assert not published and not writes
    assert registry[0]['state'] == 'scheduled'


@pytest.mark.parametrize('tracker', ['closed', 'unreadable'])
@pytest.mark.parametrize('prose', ['', 'Keep the compatibility requirement.'])
@pytest.mark.parametrize('target_kind', ['evidence-item', 'converted-row', 'signed-requirement'])
def test_round_replay_keeps_consumed_directive_separate_from_prose(
        tmp_path, monkeypatch, tracker, prose, target_kind):
    run, writes = setup_run(tmp_path, monkeypatch)
    item = evidence()
    target = None
    payload = transition()
    if target_kind == 'converted-row':
        item = replace(item, converted_row_id='row-a', converted_matrix_identity='a'*64)
        payload = transition(target_kind=target_kind, target_id='row-a', matrix_identity='a'*64)
    elif target_kind == 'signed-requirement':
        target = signed(body=item.text, url='https://github.com/OWNER/REPO/pull/1#issuecomment-9')
        payload = transition(target_kind=target_kind, target_id=target.requirement_id)
    directive = signed(payload)
    directive = replace(directive, body=directive.body + '\n' + prose)
    requirements = [directive] + ([target] if target else [])
    run.operator_inputs(requirements, items=[item], closing=(), head='a'*40, round_number=1)
    assert len(run.registry) == 1
    saved = run.registry
    monkeypatch.setattr(runtime, 'restore', lambda *args: (saved, ()))
    def tracker_read(*args):
        if tracker == 'unreadable':
            raise AgentLoopError('tracker read failed')
        return {'number': 9, 'state': 'closed'}
    monkeypatch.setattr(runtime, 'read_api', tracker_read)
    resumed = runtime.ObligationRun(run.runner, run.config, 1)
    items, visible = runtime.round_start(resumed,
        PullRequestReviewContext(metadata('b'*40), (), ()), [item], requirements, 2)
    assert [r.body for r in visible] == ([prose] if prose else [])
    assert resumed.registry == saved
    assert items == ([item] if target else [])
    assert ('closed' if tracker == 'closed' else 'tracker read failed') in resumed.context()
    assert writes[-1][0][4] == list(saved)
    assert all('```post-merge-schedule' not in r.body for r in visible)


@pytest.mark.parametrize('tracker', ['closed', 'unreadable'])
def test_frozen_resume_replays_operator_control_without_coder_or_new_freeze(tmp_path, monkeypatch, tracker):
    from coding_review_agent_loop import pr_loop
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from pr_loop_test_helpers import _freeze_records
    from test_pr_loop_evidence_freeze import _frozen_runner
    run, _ = setup_run(tmp_path, monkeypatch)
    runner = _frozen_runner(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        pr_loop.run_pr_loop(runner, config=run.config, pr_number=77)
    freezes = _freeze_records(runner)
    item = next(i for i in freezes[-1].metadata.prior_items
                if i.obligation_kind == 'human-exact-head-evidence')
    directive = signed(transition(target_id=item.obligation_identity,
        requirement=item.text, source_requirement_text=item.text))
    admitted = runtime.ObligationRun(runner, run.config, 77)
    assert admitted.operator_inputs([directive], items=[item], closing=(),
        head=item.candidate_head_sha, round_number=1) == ()
    monkeypatch.setattr(runtime, 'restore', lambda *args, **kwargs: (admitted.registry, ()))
    original_context = pr_loop._build_requirements_context
    def requirements_context(**kwargs):
        return replace(original_context(**kwargs), effective_requirements=(directive,))
    monkeypatch.setattr(pr_loop, '_build_requirements_context', requirements_context)
    def tracker_read(*args):
        if tracker == 'unreadable':
            raise AgentLoopError('tracker read failed')
        return {'number': 9, 'state': 'closed'}
    monkeypatch.setattr(runtime, 'read_api', tracker_read)
    before = len(runner.commands)
    assert pr_loop.run_pr_loop(runner, config=run.config, pr_number=77) == 0
    assert len(_freeze_records(runner)) == len(freezes)
    assert not any(cmd[0] in {'claude', 'codex'} for cmd, _ in runner.commands[before:])
    assert active_registry(runner._post_merge_run.registry) == active_registry(admitted.registry)


@pytest.mark.parametrize('change', ['owner', 'tracker_url', 'same-payload'])
def test_authorized_payload_restoration_survives_restart_and_publishes_final(
        tmp_path, monkeypatch, change):
    from coding_review_agent_loop import github
    from coding_review_agent_loop import pr_loop_support as support
    run, writes = setup_run(tmp_path, monkeypatch)
    first = signed()
    changed = {'owner': 'New owner'} if change == 'owner' else (
        {'tracker_url': 'https://github.com/OWNER/REPO/issues/10'} if change == 'tracker_url' else {})
    second = signed(transition(**changed), url=first.url.replace('10', '11'))
    third = signed(url=first.url.replace('10', '12'))
    # Replay the complete historical input at each round, including an unchanged
    # payload in a fresh comment, which must also be remembered as consumed.
    for number, requirements in enumerate(([first], [first, second], [first, second, third]), 1):
        run.operator_inputs(requirements, items=[evidence()], closing=(), head='a'*40, round_number=number)
        saved = validate_registry(list(run.registry), repo=run.config.repo, pr=1)
        record = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator',
            round_number=number, subject='a'*40, post_merge_registry=saved)
        comment = IssueComment('bot', '2026-01-01T00:00:00Z',
            str(_attach_round_metadata('Admission', record)), number, 42)
        monkeypatch.setattr(support, '_authenticated_pr_comments_status',
            lambda *args, **kwargs: ((comment,), ('bot', 42), True))
        run = runtime.ObligationRun(run.runner, run.config, 1)
        run.restore([comment])
        assert len(run.registry) == number
    active, = active_registry(run.registry)
    assert active['request'] == run.registry[0]['request']
    assert active['identity'] != run.registry[0]['identity']
    assert active['provenance']['locator'] == third.url
    run.operator_inputs([first, second, third], items=[], closing=(), head='b'*40, round_number=4)
    assert len(run.registry) == 3 and active_registry(run.registry) == (active,)
    monkeypatch.setattr(runtime, 'read_api', lambda runner, config, path:
        merged() if path.startswith('pulls/') else {'number': 9, 'state': 'open'})
    monkeypatch.setattr(github, 'get_pr_review_context', lambda *args, **kwargs:
        PullRequestReviewContext(metadata(), (), ()))
    monkeypatch.setattr(github, 'resolve_authenticated_github_actor', lambda *args, **kwargs: ('bot', 42))
    monkeypatch.setattr(runtime, 'find_publication', lambda *args: None)
    published = []
    def publish(*args, **kwargs):
        published.append(kwargs['body'])
        return type('Receipt', (), {'comment_id': 88})()
    monkeypatch.setattr(github, 'post_verified_trusted_issue_round_comment', publish)
    assert runtime.reconcile(run.runner, run.config, 1, registry=run.registry) == 0
    assert len(published) == 1
    assert active['identity'] in published[0]
    assert active['request']['owner'] in published[0]
    assert writes
