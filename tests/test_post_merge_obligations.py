"""Strict classification, deterministic operator authority and durable contract tests."""
import json
from dataclasses import replace

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.post_merge_obligations import (
    authorize, filter_items, make_admission, operator_directive, parse_request,
    parse_requests, preserve_fields, render_context, scheduled_rows, source_for_item,
    validate_registry, validate_tracker,
)
from coding_review_agent_loop.protocol import parse_structured_pr_review
from coding_review_agent_loop.round_state import PostedRoundMetadata, _encode_round_metadata, _decode_round_metadata
from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation
from agent_loop_helpers import structured_pr_review


REQUEST = {'requirement': 'Verify live production alert firing.', 'rationale': 'Production deploys merged main only.',
           'owner': 'Operations on call', 'tracker_url': 'https://github.com/OWNER/REPO/issues/9'}


def evidence():
    return _upsert_evidence_obligation([], item_number=1, reviewer='OpenAI Codex',
        text=REQUEST['requirement'], source_round=1, current_head_sha='a' * 40)[0][0]


def transition(**changes):
    value = dict(REQUEST, target_kind='evidence-item', target_id=evidence().obligation_identity,
                 source_requirement_text=REQUEST['requirement'])
    value.update(changes)
    return value


def signed(value=None, **changes):
    body = '```post-merge-schedule\n' + json.dumps({'obligations': [value or transition()]}) + '\n```'
    params = dict(source_type='PR comment', author='operator', author_id=7,
        created_at='2026-01-01T00:00:00Z', url='https://github.com/OWNER/REPO/pull/1#issuecomment-10',
        body=body, author_verification='verified')
    params.update(changes)
    return HumanReviewRequirement(**params)


def admission():
    request = parse_request(transition())
    return make_admission(request, repo='OWNER/REPO', pr=1, source=source_for_item(evidence()),
                          provenance={'kind': 'reviewer', 'reviewers': ['OpenAI Codex']})


def test_typed_declarations_legacy_and_round_trip():
    raw = structured_pr_review()
    assert parse_structured_pr_review(raw, reviewer='OpenAI Codex').post_merge_obligations == ()
    payload = json.loads(raw.split('\n')[0])
    payload['post_merge_obligations'] = [REQUEST]
    parsed = parse_structured_pr_review(json.dumps(payload) + raw[raw.index("\n"):], reviewer='OpenAI Codex')
    assert parsed.post_merge_obligations[0].payload() == REQUEST
    metadata = PostedRoundMetadata(flow='pr', role='summary', agent='Orchestrator', round_number=1,
        subject='a' * 40, post_merge_registry=(admission(),))
    assert _decode_round_metadata(_encode_round_metadata(metadata)).post_merge_registry == (admission(),)


@pytest.mark.parametrize('change', [{'owner': ''}, {'tracker_url': 'https://github.com/OWNER/REPO/pull/9'},
    {'operator': True}, {'target_id': 'orphan'}, {'requirement': ''}])
def test_reject_incomplete_or_unknown_contract(change):
    with pytest.raises(AgentLoopError):
        parse_request(dict(REQUEST, **change))


def test_repair_cannot_invent_or_change_classification():
    preserve_fields({}, {})
    preserve_fields(None, {})
    preserve_fields({'post_merge_obligations': [REQUEST]}, {'post_merge_obligations': [REQUEST]})
    for other in ({}, {'post_merge_obligations': [dict(REQUEST, owner='Someone else')]}):
        with pytest.raises(AgentLoopError):
            preserve_fields({'post_merge_obligations': [REQUEST]}, other)
    with pytest.raises(AgentLoopError):
        preserve_fields({}, {'post_merge_obligations': [REQUEST]})


def test_duplicate_targets_rejected():
    with pytest.raises(AgentLoopError, match='duplicate'):
        parse_requests([transition(), transition()])


def test_directive_exact_content_and_surrounding_prose():
    control = signed()
    requests, prose = operator_directive(control)
    assert requests[0].payload() == transition()
    assert prose == ''
    _, prose = operator_directive(replace(control, body='Keep the privacy rule.\n\n' + control.body))
    assert prose == 'Keep the privacy rule.'
    assert operator_directive(replace(control, body='Schedule the alert after merge please.')) == ((), 'Schedule the alert after merge please.')


@pytest.mark.parametrize('mutation', [
    lambda c: replace(c, author_verification='unverified'),
    lambda c: replace(c, author_id=None),
    lambda c: replace(c, url=None),
    lambda c: replace(c, source_type='Issue comment'),
    lambda c: replace(c, body=c.body + '\n' + c.body),
    lambda c: replace(c, body=c.body.replace('"obligations":', '"obligations": [], "obligations":')),
    lambda c: replace(c, body=c.body.replace('"obligations":', '"unknown":')),
    lambda c: replace(c, body=c.body.replace('Verify live production alert firing.', 'Changed.', 1)),
    lambda c: replace(c, body=c.body.replace('"owner":', '"authority": "operator", "owner":')),
])
def test_invalid_directive_grants_nothing(mutation):
    with pytest.raises(AgentLoopError):
        operator_directive(mutation(signed()))


def test_shared_owner_authority_and_operator_only_targets():
    item = replace(evidence(), resolution_owners=('A', 'B'))
    source = source_for_item(item)
    request = parse_request(transition())
    sources = {('evidence-item', request.target_id): source}
    with pytest.raises(AgentLoopError, match='every resolution owner'):
        authorize(request, sources=sources, reviewer='A', voters=['A'], board=['A', 'B'])
    assert authorize(request, sources=sources, reviewer='A', voters=['A', 'B'], board=['A', 'B']) == source
    with pytest.raises(AgentLoopError):
        authorize(request, sources=sources, reviewer='new', voters=['new'], board=['new', 'other'])
    assert authorize(request, sources=sources, reviewer='new', voters=['new', 'other'], board=['new', 'other']) == source
    row = parse_request(transition(target_kind='converted-row', target_id='row', matrix_identity='matrix'))
    source = {'target_kind': 'converted-row', 'target_id': 'row', 'matrix_identity': 'matrix', 'requirement': row.requirement}
    sources = {('converted-row', 'row'): source}
    with pytest.raises(AgentLoopError, match='verified operator'):
        authorize(row, sources=sources, reviewer='A', voters=['A'], board=['A'])
    assert authorize(row, sources=sources, operator=signed()) == source
    with pytest.raises(AgentLoopError):
        authorize(replace(row, matrix_identity='other'), sources=sources, operator=signed())


@pytest.mark.parametrize('issue,closing,url', [
    ({'number': 9, 'state': 'closed'}, (), REQUEST['tracker_url']),
    ({'number': 9, 'state': 'open', 'pull_request': {}}, (), REQUEST['tracker_url']),
    ({}, (), REQUEST['tracker_url']),
    ({'number': 9, 'state': 'open'}, (9,), REQUEST['tracker_url']),
    ({'number': 9, 'state': 'open'}, (), 'https://github.com/OTHER/REPO/issues/9'),
])
def test_tracker_admission_boundary(issue, closing, url):
    with pytest.raises(AgentLoopError):
        validate_tracker(parse_request(dict(REQUEST, tracker_url=url)), repo='OWNER/REPO',
            closing_issues=closing, read_issue=lambda n: issue)


def test_head_independent_exact_source_filter_and_truthful_row_timing():
    original = evidence()
    record = admission()
    h2 = replace(original, candidate_head_sha='b' * 40)
    assert filter_items([h2], [record]) == []
    assert filter_items([replace(h2, text='Changed demand')], [record])
    assert filter_items([replace(h2, reviewer='Another reviewer')], [record])
    row = replace(original, converted_row_id='row-1', converted_matrix_identity='matrix')
    request = parse_request(transition(target_kind='converted-row', target_id='row-1', matrix_identity='matrix'))
    row_record = make_admission(request, repo='OWNER/REPO', pr=1, source=source_for_item(row), provenance={'kind': 'operator'})
    assert scheduled_rows([row_record], 'matrix') == {'row-1'}
    assert scheduled_rows([row_record], 'different') == set()
    assert row.lifecycle == 'evidence_deferred'
    context = render_context([record])
    for text in (record['identity'], REQUEST['owner'], REQUEST['tracker_url'], 'scheduled', 'Do not re-raise'):
        assert text in context


def test_mutated_registry_stops_instead_of_releasing():
    record = admission()
    validate_registry([record], repo='OWNER/REPO', pr=1)
    with pytest.raises(AgentLoopError):
        validate_registry([dict(record, request=dict(record['request'], owner='changed'))])
    with pytest.raises(AgentLoopError):
        validate_registry([dict(record, state='tracked')])


def test_nested_example_and_unclosed_directive_are_not_control_input():
    control = signed()
    for body in ('````markdown\n' + control.body + '\n````', control.body.rsplit('\n', 1)[0]):
        with pytest.raises(AgentLoopError):
            operator_directive(replace(control, body=body))


@pytest.mark.parametrize('field', ['blocking_items', 'same_pr_followups', 'future_followups'])
def test_contradictory_classifications_rejected(field):
    from coding_review_agent_loop.post_merge_obligations import parse_review_fields
    with pytest.raises(AgentLoopError, match='contradictory'):
        parse_review_fields({'post_merge_obligations': [REQUEST], field: [REQUEST['requirement']]})


def test_converted_row_restoration_changes_no_timing_authority():
    row = replace(evidence(), converted_row_id='row-a', converted_matrix_identity='matrix')
    request = parse_request(transition(target_kind='converted-row', target_id='row-a', matrix_identity='matrix'))
    record = make_admission(request, repo='OWNER/REPO', pr=1, source=source_for_item(row), provenance={'kind': 'operator'})
    reconstructed = replace(row, text='Generated explanation restored from the original conversion audit.')
    assert filter_items([reconstructed], [record]) == []
    changed_matrix = replace(reconstructed, converted_matrix_identity='different')
    assert filter_items([changed_matrix], [record]) == [changed_matrix]
