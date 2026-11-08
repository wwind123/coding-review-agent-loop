"""Invocation-only recovery families through live GitHub acquisition and restart."""
from dataclasses import replace

import pytest

from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop import github, pr_loop
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.pr_contract import (
    find_latest_pr_contract, format_pr_contract_comment, make_pr_contract,
)
from coding_review_agent_loop.pr_loop_support import admitted_capture_history
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from test_board_author_admission import raw, HUMAN
from test_pr_round_ingestion import anchor


def contract(**kwargs):
    return make_pr_contract(repository='OWNER/REPO', pr_number=77,
        origin_flow='issue-implementation', primary_issue_number=56,
        expected_closing_issue_ids=(56,), **kwargs)


@pytest.mark.parametrize('trusted_human', [False, True])
@pytest.mark.parametrize('owned', [False, True])
def test_legacy_contract_live_acquisition_and_owning_issue_lookup(
    tmp_path, capsys, monkeypatch, trusted_human, owned,
):
    initial = contract()
    extension = replace(initial, expected_closing_issue_ids=(56, 57),
        contract_hash=make_pr_contract(repository='OWNER/REPO', pr_number=77,
            origin_flow='issue-implementation', primary_issue_number=56,
            expected_closing_issue_ids=(56, 57)).contract_hash,
        supersedes_hash=initial.contract_hash)
    foreign = replace(initial, primary_issue_number=999)
    comments = ([raw(format_pr_contract_comment(initial), 'loop', 42, 0),
                 raw(format_pr_contract_comment(extension), 'loop', 42, 1)] if owned else [])
    comments += [raw(format_pr_contract_comment(foreign), 'operator', 81, 2),
        raw('<!-- AGENT_PR_EXPECTED_CLOSING_ISSUES: invalid -->', 'outsider', 99, 3)]
    runner = FakeRunner(authenticated_actor=('loop', 42), pr_payload={'comments': comments})
    config = make_config(tmp_path, quiet=False,
        human_reviewer_trusted_actors=(HUMAN,) if trusted_human else ())
    for _ in range(2):
        context = github.get_pr_review_context(runner, config=config, pr_number=77)
        assert find_latest_pr_contract(context.comments, repository=config.repo, pr_number=77) == (extension if owned else None)
    # Drive the startup owning-issue callback; the verified contract supplies
    # the primary, while foreign-only history cannot initiate an issue lookup.
    seen = []
    def resolve(_config, **kwargs):
        seen.append(kwargs['recorded_primary_issue_number']())
        from types import SimpleNamespace
        return SimpleNamespace(fresh_scope_number=None, branch_issue_number=None,
            ordinary_managed_recovery=False, owning_issue_number=None)
    monkeypatch.setattr(pr_loop, 'resolve_pr_owning_issue', resolve)
    assert pr_loop._entry_plan_handoff_board(runner, config=config, pr_context=context,
        pr_number=77, issue_context=None, parent_issue_context=None,
        approved_plan_context=None, managed_ci_issue_number=None) is None
    assert seen == [56 if owned else None]
    assert 'ignored loop record' in capsys.readouterr().err
    # Authentic malformed history retains its failure instead of being ignored.
    runner.pr_payload['comments'] = [raw('<!-- AGENT_PR_EXPECTED_CLOSING_ISSUES: invalid -->', 'loop', 42, 4)]
    context = github.get_pr_review_context(runner, config=config, pr_number=77)
    with pytest.raises(AgentLoopError):
        find_latest_pr_contract(context.comments, repository=config.repo, pr_number=77)


@pytest.mark.parametrize('trusted_human', [False, True])
def test_capture_history_has_no_reviewer_role_exemption(tmp_path, trusted_human):
    runner = FakeRunner(authenticated_actor=('loop', 42), pr_payload={'comments': [
        raw(anchor(), 'loop', 42, 0),
        raw(anchor(round_number=4), 'operator', 81, 1),
        raw(anchor(round_number=5, role='coder'), 'outsider', 99, 2),
    ]})
    config = make_config(tmp_path, human_reviewer_trusted_actors=(HUMAN,) if trusted_human else ())
    for _ in range(2):
        context = github.get_pr_review_context(runner, config=config, pr_number=77)
        records, complete = admitted_capture_history(runner, config, pr_number=77, comments=context.comments)
        assert complete
        assert [r.metadata.round_number for r in records] == [3]
        assert records == _extract_round_metadata_records(context.comments, flow='pr')
