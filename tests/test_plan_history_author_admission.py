"""Live, separately authenticated planning and child-routing observations."""
from dataclasses import replace
from unittest.mock import patch

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import github, decomposition as child
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.pr_round_ingestion import admit_pr_comments
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from test_board_author_admission import raw, HUMAN
from test_decomposition import _m936_round, _m936_plan_hash
from test_plan_first_loop_review_scheduling import _approved_closed_history
from coding_review_agent_loop.cli import run_issue_loop
import coding_review_agent_loop.plan_first_loop as planning
import coding_review_agent_loop.plan_execution as execution


def acquire(runner, config, number):
    return github.get_issue_context(runner, config=config, issue_number=number).comments


@pytest.mark.parametrize('login,user_id,trusted,eligible', [
    ('operator', 81, (), False), ('OPERATOR', 81, (HUMAN,), True),
    ('operator', 82, (HUMAN,), False), ('other', 81, (HUMAN,), False),
    ('loop', 42, (), True), ('outsider', 99, (HUMAN,), False),
])
def test_child_controls_and_lineage_use_separate_live_family_views(
    tmp_path, capsys, login, user_id, trusted, eligible,
):
    old = _m936_plan_hash('Old plan.')
    override = child.format_child_disposition_override_comment(
        parent_issue=55, plan_hash=old, stage_id='stage-one',
        disposition='direct-implementation', rationale='Approved routing.')
    signed = child.format_child_plan_supersession_comment(
        child_issue=56, parent_issue=55, stage_id='stage-one',
        superseded_plan_hash=old, rationale='Contract tightened.')
    # Get the unchanged digest through the offline codec; only acquisition
    # below decides whether that signed record has authority on this surface.
    digest = child.parse_child_plan_supersession_records(
        signed, comment_locator='offline', comment_index=0)[0][0].digest
    rebind = child.format_child_plan_rebind_section(child.ChildPlanRebindRecord(
        child_issue=56, pr_number=77, superseded_plan_hash=old,
        new_plan_hash=_m936_plan_hash('New plan.'), plan_supersession_digest=digest,
        first_replan_round=2, approved_round=2))
    runner = FakeRunner(authenticated_actor=('loop', 42), issue_comments=[
        raw(signed, login, user_id, 1),
        raw(_m936_round('Old plan.', 1), 'loop', 42, 0),
        raw(_m936_round('New plan.', 2, prior='Old plan.', digest=digest, superseded=old)
            + '\n' + rebind, login, user_id, 2),
        raw(override, login, user_id, 3),
    ], issue_comments_by_number={55: [raw(override, login, user_id)]})
    config = make_config(tmp_path, human_reviewer_trusted_actors=trusted, quiet=False)
    for _ in range(2):
        parent, comments = acquire(runner, config, 55), acquire(runner, config, 56)
        overrides = child.collect_child_disposition_overrides(
            parent_comments=parent, child_comments=comments, parent_issue=55,
            plan_hash=old, child_issue_number=56, child_stage_id='stage-one',
            topology_stage_ids=('stage-one',), routed_stage_id='stage-one')
        supersessions = child.collect_child_plan_supersessions(
            comments, child_issue=56, parent_issue=55, stage_id='stage-one')
        assert bool(overrides) is eligible
        assert bool(supersessions) is eligible
        owned = user_id == 42
        assert bool(child.find_child_plan_rebind_records(comments)) is owned
        assert len(_extract_round_metadata_records(comments, flow='plan')) == (2 if owned else 1)
        lineage = child.authorized_replan_lineage(
            comments, superseded_hash=old, digest=digest, supersessions=supersessions)
        assert isinstance(lineage, child.AuthorizedReplanLineage) is owned
        assert comments[1].body == signed and comments[1].comment_id == 101
        # A parent observation cannot be reused to authenticate a child thread.
        with pytest.raises(AgentLoopError, match='different surface'):
            admit_pr_comments(runner, config, pr_number=56, comments=parent,
                              surface_kind=github.ISSUE_THREAD_SURFACE)
    if not eligible:
        diagnostic = capsys.readouterr().err
        assert 'ignored child control candidate' in diagnostic and '100' in diagnostic


@pytest.mark.parametrize('mode', ['plan-only', 'auto'])
def test_plan_start_refresh_and_two_restarts_keep_foreign_payloads_opaque(tmp_path, capsys, mode):
    runner, config, _ = _approved_closed_history(tmp_path)
    config = replace(config, plan_execution_mode=mode, quiet=False)
    # Both a malformed anchor and an unhydratable sidecar must be excluded
    # before drift, scope, approval and recovery readers see the history.
    foreign = raw('<!-- AGENT_LOOP_META: invalid -->\n<!-- AGENT_LOOP_SIDECAR: invalid -->',
                  'outsider', 99, 59)
    foreign.pop('id')
    trusted = acquire(runner, config, 56)
    for index, comment in enumerate(trusted):
        # Valid copied approvals and scheduler contracts are just as ineligible
        # as malformed bytes, even when they would otherwise win chronology.
        copied = raw(comment.body, 'outsider', 99, 58)
        copied['createdAt'] = '2026-01-02T00:00:00Z'
        copied['id'] = 80000 + index
        runner.issue_comments.append(copied)
    runner.issue_comments.append(foreign)
    expected = _extract_round_metadata_records(acquire(runner, config, 56), flow='plan')
    assert len(expected) == len(_extract_round_metadata_records(trusted, flow='plan'))
    for _ in range(3):
        with patch.object(planning, 'select_plan_reviewers', side_effect=AssertionError('scheduled')), \
             patch.object(execution, '_implement_approved_issue', return_value=0) as implement:
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert implement.call_count == (mode == 'auto')
        records = _extract_round_metadata_records(acquire(runner, config, 56), flow='plan')
        assert [(r.metadata.subject, r.metadata.phase) for r in records] == [
            (r.metadata.subject, r.metadata.phase) for r in expected]
        # New foreign records on refresh remain opaque too.
        runner.issue_comments.append({**foreign, 'id': 90000 + _, 'createdAt': '2026-01-02T00:00:00Z'})
    diagnostics = [line for line in capsys.readouterr().err.splitlines() if 'ignored loop record' in line]
    assert diagnostics and all(len(line) < 500 for line in diagnostics)


@pytest.mark.parametrize('trusted', [False, True])
def test_live_child_surface_guards_only_parse_admitted_controls(tmp_path, trusted):
    signed = child.format_child_plan_supersession_comment(
        child_issue=999, parent_issue=55, stage_id='stage-one',
        superseded_plan_hash='a' * 16, rationale='Wrong child surface.')
    runner = FakeRunner(authenticated_actor=('loop', 42), issue_comments=[raw(signed)])
    config = make_config(tmp_path, human_reviewer_trusted_actors=(HUMAN,) if trusted else ())
    comments = acquire(runner, config, 56)
    def collect():
        return child.collect_child_plan_supersessions(
            comments, child_issue=56, parent_issue=55, stage_id='stage-one')
    if trusted:
        with pytest.raises(AgentLoopError, match='Human decision required'):
            collect()
    else:
        assert collect() == ()


def test_live_plan_recovery_excludes_foreign_reviews_and_attachments(tmp_path):
    from coding_review_agent_loop.partial_round_recovery import compute_partial_round_recovery
    from test_partial_round_recovery import _History, _meta, _big_text, ROUND, HEAD, REVIEWERS
    from coding_review_agent_loop.round_state import _resume_plan_round

    history = _History()
    history.add_record(replace(_meta(role='coder', agent='Claude'), flow='plan'))
    history.add_record(replace(_meta(), flow='plan'))
    history.add_record(replace(_meta(agent='Gemini'), flow='plan'), author='outsider')
    history.orphan(replace(_meta(agent='Gemini'), flow='plan',
                           canonical_reviewer_response=_big_text(20)), author='outsider')
    runner = FakeRunner(authenticated_actor=('bot', 4242), issue_comments=[
        {'body': c.body, 'author': {'login': c.author}, '_rest_author_id': c.author_id,
         'id': c.comment_id, 'createdAt': c.created_at} for c in history.comments])
    config = make_config(tmp_path)
    comments = acquire(runner, config, 56)
    recovery = compute_partial_round_recovery(
        snapshot=comments, read_rest=lambda: acquire(runner, config, 56),
        flow='plan', round_number=ROUND, subject=HEAD, reviewer_names=REVIEWERS,
        resume=lambda remaining: _resume_plan_round(remaining, configured_reviewers=('codex', 'gemini')))
    assert recovery.verified
    assert {target.comment_id for target in recovery.targets} == {comments[1].comment_id}
    assert not recovery.uncertain


@pytest.mark.parametrize('reader', ['hashes', 'announcement-exists'])
@pytest.mark.parametrize('login,user_id,trusted,eligible', [
    ('loop', 42, (), True),
    ('operator', 81, (HUMAN,), False),
    ('operator', 81, (), False),
    ('outsider', 99, (HUMAN,), False),
])
def test_plan_announcement_readers_use_live_invocation_projection(
    tmp_path, reader, login, user_id, trusted, eligible,
):
    from coding_review_agent_loop.followups import (
        _plan_approved_followups_marker, _has_plan_approved_followups_marker,
        approved_plan_hashes_for_issue,
    )
    plan_hash = _m936_plan_hash('Approved plan.')
    body = _plan_approved_followups_marker(56, plan_hash, 'comment')
    runner = FakeRunner(authenticated_actor=('loop', 42),
                        issue_comments=[raw(body, login, user_id)])
    config = make_config(tmp_path, human_reviewer_trusted_actors=trusted)
    # This announcement is the only candidate in a short thread: acquisition
    # must authenticate it even without round metadata or control records.
    for _ in range(2):
        comments = acquire(runner, config, 56)
        assert comments[0].body == body
        if reader == 'hashes':
            assert approved_plan_hashes_for_issue(comments, issue_number=56) == (
                (plan_hash,) if eligible else ())
        else:
            assert _has_plan_approved_followups_marker(
                comments, issue_number=56, plan_hash=plan_hash, mode='comment') is eligible


@pytest.mark.parametrize('mode', ['plan-only', 'auto'])
@pytest.mark.parametrize('failure', ['malformed', 'sidecar', 'page', 'identity'])
def test_plan_invalid_history_two_restarts_then_fresh_recovery(tmp_path, monkeypatch, mode, failure):
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    from test_pr_round_ingestion import BAD, spilled
    runner, config, _ = _approved_closed_history(tmp_path)
    config = replace(config, plan_execution_mode=mode, quiet=False)
    with patch.object(execution, '_implement_approved_issue', return_value=0):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    baseline = list(runner.issue_comments)
    actor, actor_id = runner.authenticated_actor or ('coding-review-agent-loop', 424242)
    publications = len(runner.comments)
    commands = len(runner.commands)
    for restart in range(2):
        with patch.object(planning, 'select_plan_reviewers', side_effect=AssertionError('scheduled')), \
             patch.object(execution, '_implement_approved_issue', return_value=0) as implement:
            with monkeypatch.context() as broken:
                if failure in {'malformed', 'sidecar'}:
                    body = BAD if failure == 'malformed' else spilled()[-1]
                    runner.issue_comments = [*baseline, {**raw(body, actor, actor_id, 990 + restart),
                        'createdAt': '2026-01-03T00:00:00Z'}]
                elif failure == 'page':
                    broken.setattr(ingestion, 'read_complete_comment_listing',
                        lambda *a, **k: github.UnknownCommentListing('issue:56', 'truncated page'))
                else:
                    broken.setattr(ingestion, 'resolve_authenticated_github_actor',
                        lambda *a, **k: (_ for _ in ()).throw(AgentLoopError('unknown identity')))
                with pytest.raises(AgentLoopError):
                    run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
                assert implement.call_count == 0
                assert len(runner.comments) == publications
            runner.issue_comments = list(baseline)
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert implement.call_count == (mode == 'auto')
    assert not any(cmd[0] in {'claude', 'codex', 'gemini', 'agy'}
                   for cmd, _ in runner.commands[commands:])



def _assert_refusal_diagnostics_only(bodies):
    """The only publication allowed after a refused refresh is the existing
    explicit scheduler refusal diagnostic; it never carries round authority."""
    from types import SimpleNamespace
    for body in bodies:
        assert 'scheduling diagnostic' in body
        assert not _extract_round_metadata_records([SimpleNamespace(body=body)], flow='plan')


@pytest.mark.parametrize('mode', ['plan-only', 'auto'])
@pytest.mark.parametrize('failure', ['foreign', 'malformed', 'sidecar', 'page'])
def test_plan_history_changed_inside_one_invocation_is_reacquired(tmp_path, monkeypatch, capsys, mode, failure):
    """History changes after the invocation's first successful acquisition,
    so a startup snapshot cannot be reused at the later refreshed readers."""
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    from test_pr_round_ingestion import BAD, spilled
    runner, config, _ = _approved_closed_history(tmp_path)
    config = replace(config, plan_execution_mode=mode, quiet=False)
    baseline = list(runner.issue_comments)
    actor = baseline[0]['author']['login']
    actor_id = (runner.authenticated_actor or (actor, 424242))[1]
    commands = len(runner.commands)
    if failure == 'foreign':
        bodies = [raw(BAD, 'outsider', 99, 90), raw(str(baseline[0]['body']), 'outsider', 99, 91),
                  raw('Ordinary operator discussion', 'outsider', 99, 92)]
    elif failure in {'malformed', 'sidecar'}:
        bodies = [raw(BAD if failure == 'malformed' else spilled()[-1], actor, actor_id, 90)]
    else:
        bodies = []
    for body in bodies:
        body['createdAt'] = '2026-01-03T00:00:00Z'
    # Each authenticated acquisition goes through the complete listing read;
    # change remote history right after the first one completes.
    state = {'reads': 0, 'published_at_refresh': None}
    listing = ingestion.read_complete_comment_listing
    def observed_listing(*args, **kwargs):
        state['reads'] += 1
        if state['reads'] == 2:
            state['published_at_refresh'] = len(runner.comments)
        if failure == 'page' and state['reads'] > 1:
            return github.UnknownCommentListing('issue:56', 'truncated page')
        result = listing(*args, **kwargs)
        if state['reads'] == 1:
            runner.issue_comments = [*baseline, *bodies]
        return result
    monkeypatch.setattr(ingestion, 'read_complete_comment_listing', observed_listing)
    with patch.object(planning, 'select_plan_reviewers', side_effect=AssertionError('scheduled')), \
         patch.object(execution, '_implement_approved_issue', return_value=0) as implement:
        if failure == 'foreign':
            # Foreign records arriving mid-run stay opaque: the run finishes on
            # trusted state with no foreign authority.
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert state['reads'] >= 2, 'no refreshed acquisition inside the invocation'
            assert implement.call_count == (mode == 'auto')
            monkeypatch.undo()
            reread = acquire(runner, config, 56)
            assert 'Ordinary operator discussion' in [c.body for c in reread]
            records = _extract_round_metadata_records(reread, flow='plan')
            assert records and all(reread[r.index].author != 'outsider' for r in records)
            diagnostics = [line for line in capsys.readouterr().err.splitlines()
                           if 'ignored loop record' in line]
            assert diagnostics and all(len(line) < 500 for line in diagnostics)
        else:
            with pytest.raises(AgentLoopError):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
            assert state['published_at_refresh'] is not None, 'no refreshed acquisition inside the invocation'
            assert implement.call_count == 0
            _assert_refusal_diagnostics_only(runner.comments[state['published_at_refresh']:])
            # A restart against the persisting failure refuses as well, with
            # no stale-baseline fallback, implementation or round publication.
            stopped = len(runner.comments)
            with pytest.raises(AgentLoopError):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
            assert implement.call_count == 0
            _assert_refusal_diagnostics_only(runner.comments[stopped:])
            monkeypatch.undo()
            runner.issue_comments = [*baseline, *runner.issue_comments[len(baseline) + len(bodies):]]
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert implement.call_count == (mode == 'auto')
    assert not any(cmd[0] in {'claude', 'codex', 'gemini', 'agy'}
                   for cmd, _ in runner.commands[commands:])


# --- #1984: issue-thread admission is independent of count and candidates ---

def _issue_raw(body, login, user_id, index):
    return dict(body=body, author={'login': login}, _rest_author_id=user_id,
                createdAt=f'2026-01-01T{index // 60:02d}:{index % 60:02d}:00Z', id=100 + index)


def _padded(history, total=105):
    return [*history, *(_issue_raw(f'Ordinary note {i}', 'loop', 42, i)
                        for i in range(len(history), total))]


def _acquire_issue(tmp_path, comments):
    from coding_review_agent_loop.pr_round_ingestion import AdmittedPrComment
    runner = FakeRunner(authenticated_actor=('loop', 42), issue_comments=comments,
                        serve_rest_issue_comments=True)
    config = make_config(tmp_path)
    admitted = acquire(runner, config, 56)
    assert all(isinstance(c, AdmittedPrComment) for c in admitted)
    assert any(cmd[:2] == ['gh', 'api'] and 'comments' in ' '.join(cmd)
               for cmd, _ in runner.commands)
    return admitted


def _projection(comments, count):
    from coding_review_agent_loop.pr_round_ingestion import record_body
    return [(c.comment_id, c.body, c.invocation_actor_id, c.observation_surface,
             record_body(c), c.protocol_body) for c in comments[:count]]


def test_candidate_free_small_issue_history_is_admitted_like_padded_history(tmp_path):
    from coding_review_agent_loop.pr_round_ingestion import protocol_candidate
    history = [
        _issue_raw('Please implement this.', 'loop', 42, 0),
        _issue_raw('Foreign prose that must not gain protocol authority.', 'outsider', 99, 1),
        _issue_raw('Follow-up note.', 'loop', 42, 2),
    ]
    snapshot = github._parse_issue_comments(history)
    assert not any(protocol_candidate(c) for c in snapshot)
    small = _acquire_issue(tmp_path / 'small', history)
    padded = _acquire_issue(tmp_path / 'padded', _padded(history))
    assert len(small) == 3 and len(padded) == 105
    assert _projection(small, 3) == _projection(padded, 3)
    foreign = small[1]
    assert foreign.body != foreign.protocol_body and foreign.protocol_body == ''
    assert {c.invocation_actor_id for c in small} == {42}


def test_plan_round_metadata_selection_is_unchanged_by_padding(tmp_path):
    from coding_review_agent_loop.pr_round_ingestion import extract_admitted_round_records
    history = [
        _issue_raw(_m936_round('Plan one.', 1), 'loop', 42, 0),
        _issue_raw(_m936_round('Foreign plan.', 2), 'outsider', 99, 1),
        _issue_raw('Ordinary discussion.', 'operator', 81, 2),
    ]
    small = _acquire_issue(tmp_path / 'small', history)
    padded = _acquire_issue(tmp_path / 'padded', _padded(history))
    small_records = extract_admitted_round_records(small, flow='plan')
    padded_records = extract_admitted_round_records(padded, flow='plan')
    assert [(r.index, r.metadata) for r in small_records] == [
        (r.index, r.metadata) for r in padded_records]
    assert [r.metadata.round_number for r in small_records] == [1]
    assert _projection(small, 3) == _projection(padded, 3)
