"""Staged parent/child recovery across fresh authenticated observations."""

import pytest

from agent_loop_helpers import (
    FakeRunner, approved_plan_comments, child_pr_handoff_comment, make_config,
    phase_handoff_comment, pr_payload_for_state, staged_legacy_plan_records,
)
from coding_review_agent_loop.cli import run_issue_loop
from coding_review_agent_loop import github, issue_implementation
from coding_review_agent_loop.decomposition import (
    approved_plan_hash, _encode_json_payload, _decode_json_payload, DECOMPOSITION_MARKER_RE,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from test_pr_round_ingestion import BAD, spilled
from test_board_author_admission import raw


def staged_remote(tmp_path):
    plan, created, summary = staged_legacy_plan_records(stage_count=2)
    handoff = child_pr_handoff_comment(99, 912)
    handoff['body'] = format_issue_pr_handoff_comment(
        issue_number=99, pr_number=912, pr_url='https://github.com/OWNER/REPO/pull/912',
        pr_head_sha='6367c48dd193d56ea7b0baad25b19455e529f5ee',
        flow='approved-plan-implementation', plan_hash=approved_plan_hash(plan))
    runner = FakeRunner(
        issue_comments=[*approved_plan_comments(plan),
                        {'author': {'login': 'bot'}, 'body': summary},
                        phase_handoff_comment(plan, created, 1)],
        issue_comments_by_number={99: [handoff], 100: []},
        pr_payloads_by_number={912: {**pr_payload_for_state(912, 'OPEN'), 'body': 'Fixes #99'}})
    return runner, make_config(tmp_path, plan_execution_mode='implement-by-phase', quiet=False)


@pytest.mark.parametrize('surface,failure', [(surface, failure)
    for surface in (56, 99)
    for failure in ('none', 'malformed', 'sidecar', 'page', 'identity', 'topology')
    if failure != 'topology' or surface == 56])
def test_staged_parent_child_two_restarts_and_transport_recovery(tmp_path, monkeypatch, capsys, failure, surface):
    import coding_review_agent_loop.pr_round_ingestion as ingestion
    runner, config = staged_remote(tmp_path)
    dispatched = []
    def pr_boundary(_runner, **kwargs):
        dispatched.append((kwargs['issue_context'].number,
                           kwargs['parent_issue_context'].number,
                           kwargs['approved_plan_context'].plan_hash))
        return 0
    # Stop at the downstream PR loop; all issue acquisition, topology,
    # child routing and canonical handoff recovery above it are real.
    monkeypatch.setattr(issue_implementation, 'run_pr_loop', pr_boundary)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    expected = dispatched[-1]
    assert expected[:2] == (99, 56)
    baseline = list(runner.issue_comments if surface == 56 else runner.issue_comments_by_number[surface])
    def remote(comments):
        if surface == 56:
            runner.issue_comments = comments
        else:
            runner.issue_comments_by_number[surface] = comments
    for restart in range(2):
        foreign = raw(BAD, 'outsider', 99, 800 + restart)
        foreign['createdAt'] = '2026-01-02T00:00:00Z'
        copied = raw(str(baseline[0]['body']), 'outsider', 99, 1800 + restart)
        prose = raw('Ordinary operator discussion', 'outsider', 99, 2800 + restart)
        copied['createdAt'] = prose['createdAt'] = foreign['createdAt']
        valid = [*baseline, copied, prose, foreign]
        remote(valid)
        comments = github.get_issue_context(runner, config=config, issue_number=surface).comments
        assert any(c.body == BAD and c.author == 'outsider' for c in comments)
        identities = [(c.comment_id, c.body) for c in comments]
        before = len(dispatched)
        publications = len(runner.comments)
        with monkeypatch.context() as broken:
            if failure in {'malformed', 'sidecar'}:
                body = BAD if failure == 'malformed' else spilled()[-1]
                remote([*valid, {**raw(body, 'coding-review-agent-loop', 424242, 990 + restart),
                                'createdAt': '2026-01-03T00:00:00Z'}])
            elif failure == 'topology':
                incomplete = []
                for comment in valid:
                    match = DECOMPOSITION_MARKER_RE.search(comment['body'])
                    if match:
                        payload = _decode_json_payload(match.group('payload'), marker_name='AGENT_PLAN_DECOMPOSITION')
                        payload['children'] = payload['children'][:-1]
                        comment = {**comment, 'body': comment['body'].replace(
                            match.group('payload'), _encode_json_payload(payload))}
                    incomplete.append(comment)
                remote(incomplete)
            elif failure == 'identity':
                remote([{**valid[0], '_rest_author_id': None}, *valid[1:]])
            elif failure == 'page':
                original = ingestion.read_complete_comment_listing
                def listing(*args, **kwargs):
                    if kwargs.get('surface') == f'issue#{surface}':
                        return github.UnknownCommentListing(f'issue#{surface}', 'incomplete identity/page')
                    return original(*args, **kwargs)
                broken.setattr(ingestion, 'read_complete_comment_listing', listing)
            if failure != 'none':
                with pytest.raises(AgentLoopError, match='incomplete' if failure == 'topology' else None):
                    run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
                assert len(dispatched) == before
                assert len(runner.comments) == publications
        remote(valid)
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
        assert dispatched[-1] == expected
        assert len(runner.comments) == publications
        reread = github.get_issue_context(runner, config=config, issue_number=surface).comments
        assert [(c.comment_id, c.body) for c in reread] == identities
        baseline = valid
    diagnostics = [line for line in capsys.readouterr().err.splitlines() if 'ignored loop record' in line]
    assert diagnostics and all(len(line) < 500 for line in diagnostics)
    assert not runner.issues
    assert not any(cmd[0] in {'claude', 'codex', 'gemini', 'agy'} for cmd, _ in runner.commands)
