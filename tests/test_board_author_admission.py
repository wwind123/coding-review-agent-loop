"""Family admission through live issue acquisition and read-only board entry."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import github
from coding_review_agent_loop.board_amendment import (
    collect_reviewer_board_amendments, format_reviewer_board_amendment_comment,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.plan_board_state import resolve_plan_board_state
from coding_review_agent_loop.pr_board_entry import _enforce_pr_board_floor
from coding_review_agent_loop.pr_round_ingestion import admit_pr_comments, protocol_comments

ACTOR = ('loop', 42)
HUMAN = github.TrustedHumanActor('Operator', 81)


def amendment(flow, *, target=None, malformed=False):
    body = format_reviewer_board_amendment_comment(
        flow=flow, issue=(target or 56) if flow == 'plan' else None,
        pr_number=(target or 77) if flow == 'pr' else None,
        original_required_reviewers=('Codex', 'Gemini', 'Antigravity'),
        policy='all-reviewers', primary_reviewer=None,
        removed_reviewers=('Antigravity',), effective_from_round=1,
        rationale='Backend unavailable.',
    )
    return body.replace('"effective_from_round": 1', '"effective_from_round": "bad"') if malformed else body


def raw(body, login='operator', user_id=81, index=0):
    return dict(body=body, author={'login': login}, _rest_author_id=user_id,
                createdAt=f'2026-01-01T00:00:{index:02d}Z', id=100 + index)


def fixture(tmp_path, flow, comments, *, trusted=(), dry_run=False):
    runner = FakeRunner(authenticated_actor=ACTOR, serve_rest_issue_comments=True,
                        issue_comments=comments, pr_payload={'comments': comments})
    config = make_config(tmp_path, reviewer=('codex', 'gemini', 'antigravity'),
                         plan_review_policy='all-reviewers',
                         human_reviewer_trusted_actors=trusted, dry_run=dry_run, quiet=False)
    return runner, config


def acquire(runner, config, flow):
    if flow == 'plan':
        return github.get_issue_context(runner, config=config, issue_number=56).comments
    return github.get_pr_review_context(runner, config=config, pr_number=77).comments


@pytest.mark.parametrize('flow', ['plan', 'pr'])
@pytest.mark.parametrize('login,user_id,trusted,expected', [
    ('operator', 81, (), False), ('OPERATOR', 81, (HUMAN,), True),
    ('operator', 82, (HUMAN,), False), ('other', 81, (HUMAN,), False),
    ('loop', 42, (), True), ('outsider', 99, (HUMAN,), False),
])
def test_board_family_partition_at_live_acquisition(tmp_path, capsys, flow, login, user_id, trusted, expected):
    # A malformed loop payload co-located with a human amendment must stay opaque.
    body = amendment(flow).replace('\n-- Human Reviewer', '\n<!-- AGENT_LOOP_META: invalid -->\n-- Human Reviewer')
    runner, config = fixture(tmp_path, flow, [raw(body, login, user_id)], trusted=trusted)
    observations = [acquire(runner, config, flow) for _ in range(3)]
    assert all([(c.comment_id, c.body) for c in current] ==
               [(c.comment_id, c.body) for c in observations[0]] for current in observations)
    assert observations[0][0] is not observations[-1][0]
    comments = observations[-1]
    kwargs = {'issue_number': 56} if flow == 'plan' else {'pr_number': 77}
    records = collect_reviewer_board_amendments(comments, flow=flow, **kwargs)
    assert bool(records) == expected
    assert comments[0].body == body and comments[0].comment_id == 100
    if user_id != ACTOR[1]:
        assert protocol_comments(comments)[0].body == ''
    if not expected:
        diagnostic = capsys.readouterr().err
        assert '100' in diagnostic and 'ignored reviewer-board amendment' in diagnostic
    assert len(runner.issue_comments) == 1  # acquisition never posts
    assert not runner.codex_outputs


@pytest.mark.parametrize('flow', ['plan', 'pr'])
@pytest.mark.parametrize('trusted', [False, True])
def test_misplaced_guard_runs_only_after_board_admission(tmp_path, flow, trusted):
    body = amendment(flow, target=999)
    runner, config = fixture(tmp_path, flow, [raw(body), raw(amendment(flow, malformed=True), index=1)],
                             trusted=(HUMAN,) if trusted else ())
    if flow == 'plan':
        def enter():
            comments = acquire(runner, config, flow)
            return resolve_plan_board_state(runner, config=config, issue_number=56,
                comments=comments, drift_error=lambda _, detail: AgentLoopError(detail))
    else:
        def enter():
            context = github.get_pr_review_context(runner, config=config, pr_number=77)
            return _enforce_pr_board_floor(runner, config=config, pr_number=77, pr_context=context)
    if trusted:
        with pytest.raises(AgentLoopError, match='names (issue|PR) #999'):
            enter()
    else:
        enter()
    assert not any(cmd[0] in {'claude', 'codex', 'agy'} or cmd[:3] == ['gh', 'api', '--method']
                   for cmd, _ in runner.commands)
    assert len(runner.issue_comments) == 2


@pytest.mark.parametrize('flow', ['plan', 'pr'])
@pytest.mark.parametrize('dry_run', [False, True])
@pytest.mark.parametrize('failure', ['actor', 'author', 'login', 'id', 'body', 'page'])
def test_board_preflight_identity_failure_is_read_only(tmp_path, monkeypatch, flow, dry_run, failure):
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    from coding_review_agent_loop.plan_board_state import admitted_issue_comments
    runner, config = fixture(tmp_path, flow, [raw(amendment(flow))], dry_run=dry_run)
    source = github._parse_issue_comments(runner.issue_comments)
    if failure == 'actor':
        monkeypatch.setattr(ingestion, 'resolve_authenticated_github_actor',
                            lambda *a, **k: (_ for _ in ()).throw(AgentLoopError('missing actor ID')))
    elif failure == 'page':
        monkeypatch.setattr(ingestion, 'read_complete_comment_listing',
                            lambda *a, **k: github.UnknownCommentListing('surface', 'page failed'))
    elif failure == 'author':
        runner.issue_comments[0]['_rest_author_id'] = None
    else:
        changes = {'login': {'author': 'changed'}, 'id': {'author_id': 82},
                   'body': {'body': amendment(flow) + ' drift'}}
        source = (replace(source[0], **changes[failure]),)
    with pytest.raises(AgentLoopError):
        if flow == 'plan':
            admitted_issue_comments(runner, config=config, issue_number=56, comments=source)
        else:
            context = SimpleNamespace(comments=source)
            _enforce_pr_board_floor(runner, config=config, pr_number=77, pr_context=context)
    assert len(runner.issue_comments) == 1
    assert all(cmd[:2] == ['gh', 'api'] for cmd, _ in runner.commands)


def test_rest_only_board_candidate_joins_full_observation(tmp_path):
    runner, config = fixture(tmp_path, 'pr', [raw('ordinary discussion'), raw(amendment('pr'), index=1)], trusted=(HUMAN,))
    comments, _ = admit_pr_comments(runner, config, pr_number=77, comments=(), force_probe=True)
    (record,) = collect_reviewer_board_amendments(comments, flow='pr', pr_number=77)
    assert len(comments) == 2 and comments[0].body == 'ordinary discussion'
    assert record.comment_index == 1


def test_bound_observation_rechecks_actor_surface_and_human_configuration(tmp_path):
    runner, config = fixture(tmp_path, 'pr', [raw(amendment('pr'))], trusted=(HUMAN,))
    comments = acquire(runner, config, 'pr')
    unconfigured, _ = admit_pr_comments(runner, replace(config, human_reviewer_trusted_actors=()),
                                       pr_number=77, comments=comments)
    assert not collect_reviewer_board_amendments(unconfigured, flow='pr', pr_number=77)
    with pytest.raises(AgentLoopError, match='different surface'):
        admit_pr_comments(runner, config, pr_number=78, comments=comments)
    runner._agent_loop_authenticated_actor = ('renamed', ACTOR[1])
    with pytest.raises(AgentLoopError, match='different invocation actor'):
        admit_pr_comments(runner, config, pr_number=77, comments=comments)


@pytest.mark.parametrize('trusted', [False, True])
def test_retirement_redaction_preserves_family_partition(tmp_path, trusted):
    from coding_review_agent_loop.execution_retirement import ExecutionHistory
    body = 'retired prose\n' + amendment('plan')
    runner, config = fixture(tmp_path, 'plan', [raw(body)], trusted=(HUMAN,) if trusted else ())
    context = github.get_issue_context(runner, config=config, issue_number=56)
    comment = context.comments[0]
    identity = (comment.comment_id, comment.created_at, comment.body)
    history = ExecutionHistory(56, frozenset(), frozenset(), frozenset(),
                               ((identity, 0, len('retired prose\n')),))
    view = history.view(context)
    records = collect_reviewer_board_amendments(view.comments, flow='plan', issue_number=56)
    assert bool(records) == trusted
    assert view.comments[0].body == amendment('plan')
    assert protocol_comments(view.comments)[0].body == ''


@pytest.mark.parametrize('flow', ['plan', 'pr'])
@pytest.mark.parametrize('trusted', [False, True])
def test_escaped_json_record_kind_cannot_bypass_candidate_admission(tmp_path, flow, trusted):
    body = amendment(flow).replace('reviewer-board-amendment', r'\u0072eviewer-board-amendment')
    runner, config = fixture(tmp_path, flow, [raw(body)], trusted=(HUMAN,) if trusted else ())
    comments = acquire(runner, config, flow)
    kwargs = {'issue_number': 56} if flow == 'plan' else {'pr_number': 77}
    assert bool(collect_reviewer_board_amendments(comments, flow=flow, **kwargs)) == trusted


@pytest.mark.parametrize('flow', ['plan', 'pr'])
def test_ordinary_signed_json_keeps_the_existing_requirement_policy(tmp_path, flow):
    body = '```json\n{"kind": "artifact-setting", "enabled": true}\n```\n-- Human Reviewer'
    runner, config = fixture(tmp_path, flow, [raw(body)])
    comments = acquire(runner, config, flow)
    assert comments[0].body == body
    assert not any(cmd[:2] == ['gh', 'api'] for cmd, _ in runner.commands)


def test_scope_only_issue_history_still_requires_complete_acquisition(tmp_path, monkeypatch):
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    from coding_review_agent_loop.plan_board_state import admitted_issue_comments
    runner, config = fixture(tmp_path, 'plan', [raw('<!-- AGENT_PLAN_SCOPE_BOUNDARY: invalid -->')])
    source = github._parse_issue_comments(runner.issue_comments)
    monkeypatch.setattr(ingestion, 'read_complete_comment_listing',
                        lambda *a, **k: github.UnknownCommentListing('issue#56', 'incomplete'))
    with pytest.raises(AgentLoopError, match='incomplete'):
        admitted_issue_comments(runner, config=config, issue_number=56, comments=source)
