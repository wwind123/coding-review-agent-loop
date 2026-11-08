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
def test_plan_start_refresh_and_two_restarts_keep_foreign_payloads_opaque(tmp_path, mode):
    runner, config, _ = _approved_closed_history(tmp_path)
    config = replace(config, plan_execution_mode=mode)
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
    for _ in range(2):
        with patch.object(planning, 'select_plan_reviewers', side_effect=AssertionError('scheduled')), \
             patch.object(execution, '_implement_approved_issue', return_value=0) as implement:
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert implement.call_count == (mode == 'auto')
        records = _extract_round_metadata_records(acquire(runner, config, 56), flow='plan')
        assert [(r.metadata.subject, r.metadata.phase) for r in records] == [
            (r.metadata.subject, r.metadata.phase) for r in expected]
        # New foreign records on refresh remain opaque too.
        runner.issue_comments.append({**foreign, 'id': 90000 + _, 'createdAt': '2026-01-02T00:00:00Z'})


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
