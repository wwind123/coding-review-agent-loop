"""Stub-only lifecycle reduction and head-independent capture identities."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from coding_review_agent_loop import review_images_state as state
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.review_images_config import CaptureConfig, delivery_limits
from coding_review_agent_loop.review_images_manifest import publish
from coding_review_agent_loop.review_scope import authority_records, is_complete_code_review
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    PostedRoundRecord,
    _encode_round_metadata,
    _decode_round_metadata,
)
from review_images_helpers import manifest, png

HEAD = 'a' * 40
OTHER = 'b' * 40
CONTRACT = dict(seat='seat', provider='codex', model='model', effort='medium',
    model_chain=['model'], executable_digest='executable', permission_mode='default',
    contract_id='contract', version='version', transport=1, proof=1, prompt=1, schema=1)


def binding(head=HEAD, **changes):
    value = dict(version=1, kind='intent', repo='OWNER/REPO', pr=77, head=head,
                 attempt=head[:32], seat='seat', contract=CONTRACT, capture='c' * 64,
                 configuration='d' * 64, comparison='e' * 64)
    return dict(value, **changes)


def record(value, index=0, **changes):
    head = value.get('head', HEAD) if isinstance(value, dict) else HEAD
    kind = value.get('kind') if isinstance(value, dict) else None
    metadata = PostedRoundMetadata(flow='pr', role='reviewer' if kind == 'result' else 'visual-' + str(kind),
        agent='seat', round_number=1, subject=head, visual_lifecycle=value, visual_lifecycle_present=True,
        review_scope={'version':1, 'kind':'visible-defects', 'head':head, 'seat':'seat'},
        state='approved' if kind == 'result' else None,
        acquisition_outcome='success', acquisition_returncode=0)
    return PostedRoundRecord(index, replace(metadata, **changes), '')


def history(*records, complete=True):
    return state.History(records, identities={r.index: r.index + 100 for r in records}, complete=complete)


def reduce(records, head=HEAD):
    return state.reduce_history(records, repo='OWNER/REPO', pr=77, head=head)


def success(head=HEAD, offset=0, **changes):
    intent = binding(head, **changes)
    return (record(intent, offset), record(dict(intent, kind='result', intent_id=offset+100, proof='f'*64), offset+1))


@pytest.mark.parametrize('raw', [None, [], 'bad', {}, {'version':99}, {'head':None}])
def test_raw_lifecycle_round_trip_has_no_other_authority(raw):
    m = record(raw).metadata
    decoded = _decode_round_metadata(_encode_round_metadata(m))
    assert decoded.visual_lifecycle_present and decoded.visual_lifecycle == raw
    assert not is_complete_code_review(decoded)
    assert authority_records((record(raw),)) == ()
    with pytest.raises(HumanDecisionRequiredError):
        reduce(history(record(raw)))


def test_local_capture_without_intent_does_not_spend_resume():
    assert not reduce(history()).spent
    assert reduce(history(record(binding()))).spent
    assert reduce(history(record(binding()))).terminal is None


@pytest.mark.parametrize('corruption', ['orphan', 'duplicate', 'binding', 'proof', 'version', 'null', 'head', 'identity'])
def test_malformed_history_never_reopens_spend(corruption):
    intent, result = success()
    records = [intent, result]
    if corruption == 'orphan': records = [result]
    elif corruption == 'duplicate': records.append(replace(result, index=2))
    elif corruption == 'binding': records[1] = record(dict(result.metadata.visual_lifecycle, comparison='0'*64), 1)
    elif corruption == 'proof': records[1] = record(dict(result.metadata.visual_lifecycle, proof=None), 1)
    elif corruption == 'version': records[0] = record(dict(binding(), version=9))
    elif corruption == 'null': records[0] = record(None)
    elif corruption == 'head': records[0] = record(binding(), subject=OTHER)
    h = history(*records, complete=corruption != 'identity')
    with pytest.raises(HumanDecisionRequiredError): reduce(h)


def test_older_invalid_history_disables_reuse_without_stopping_fresh_head():
    invalid = record(dict(binding(), version=2))
    admission = reduce(history(invalid), OTHER)
    assert not admission.spent and admission.latest is None
    with pytest.raises(HumanDecisionRequiredError): reduce(history(invalid))


@pytest.mark.parametrize('broken', ['missing', 'cycle', 'contract', 'uncovered'])
def test_skip_provenance_fails_closed(broken):
    records = list(success())
    value = binding(OTHER, kind='skip', source_id=101, original_id=101, assigned=[])
    if broken == 'missing': value['original_id'] = 999
    elif broken == 'cycle': value['source_id'] = 102
    elif broken == 'contract': value['contract'] = dict(CONTRACT, effort='high')
    else: value['assigned'] = [['item-9','seat']]
    records.append(record(value, 2))
    with pytest.raises(HumanDecisionRequiredError): reduce(history(*records), OTHER)


def test_latest_source_only_and_skip_chain_preserves_source_raised_coverage():
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item
    item = replace(_next_unresolved_item(item_number=1, reviewer='seat', source_round=1,
                   text='Clipped title', status='blocking'), origin_scope='visible-defects')
    a, b = success()
    b = replace(b, metadata=replace(b.metadata, new_items=(item,), state='blocking'))
    admitted = reduce(history(a,b), OTHER)
    assert state.can_reuse(admitted, binding(OTHER), (item,))
    uncovered = replace(item, item_id='item-2')
    assert not state.can_reuse(admitted, binding(OTHER), (uncovered,))
    skip = record(binding(OTHER, kind='skip', source_id=101, original_id=101,
                          assigned=[['item-1','seat']]), 2)
    resumed = reduce(history(a,b,skip), OTHER)
    assert resumed.terminal == skip and resumed.original[0] == 101
    assert state.can_reuse(resumed, binding('c'*40), (item,))
    changed = success(OTHER, offset=2, comparison='0'*64)
    latest = reduce(history(a,b,*changed), 'c'*40)
    assert latest.latest[0] == 103
    assert not state.can_reuse(latest, binding('c'*40), ())


@pytest.mark.parametrize('change', ['none', 'metadata', 'association', 'bytes', 'settings', 'inventory-order'])
def test_comparison_identity_separates_head_and_settings(change):
    settings = CaptureConfig(paths=('ui/*',), command='capture', limits=delivery_limits())
    config_digest = state.configuration_identity(settings)
    values = [manifest(HEAD), manifest(OTHER)]
    contents = [{'one.png':png(), 'base.png':png()} for _ in values]
    if change == 'metadata': values[1]['images'][0]['locale'] = 'fr'
    elif change == 'association': values[1]['images'][0]['path'], values[1]['images'][0]['baseline_path'] = 'base.png', 'one.png'
    elif change == 'bytes': contents[1]['one.png'] = png(width=3)
    identities = []
    snapshots = []
    for index, value in enumerate(values):
        snap = publish(json.dumps(value).encode(), value['head_sha'], settings.limits, contents[index],
                       {'one.png':(2,2), 'base.png':(2,2)}, value['head_sha'])
        snapshots.append(snap)
        with snap:
            cfg = state.configuration_identity(replace(settings, capture_timeout=7)) if index and change == 'settings' else config_digest
            if change == 'inventory-order': snap = replace(snap, files=tuple(reversed(snap.files)))
            identities.append(state.comparison_identity(snap, cfg))
    assert snapshots[0].digest != snapshots[1].digest
    assert (identities[0] == identities[1]) == (change in {'none','inventory-order'})


@pytest.mark.parametrize('field', list(CONTRACT))
def test_incompatible_or_incomplete_contract_cannot_reuse(field):
    value = dict(CONTRACT)
    value.pop(field)
    assert not state.compatible_contract(value)
    a,b = success()
    admission = reduce(history(a,b), OTHER)
    current = binding(OTHER, contract=dict(CONTRACT, **{field: 'changed'}))
    assert not state.can_reuse(admission, current, ())


@pytest.mark.parametrize('failure', ['none','anchor','sidecar'])
def test_verified_pr_round_transport_returns_final_anchor_and_requires_sidecars(tmp_path, monkeypatch, failure):
    from agent_loop_helpers import FakeRunner, make_config
    from test_round_transport import _random_text
    from coding_review_agent_loop import github
    from coding_review_agent_loop.round_metadata import (
        _attach_round_metadata,
    )
    from coding_review_agent_loop.round_state import (
        _extract_round_metadata_records,
    )
    from coding_review_agent_loop.errors import AgentLoopError
    config=make_config(tmp_path)
    runner=FakeRunner()
    m=replace(success()[1].metadata, canonical_reviewer_response=_random_text(70000))
    body=_attach_round_metadata('Verified visual result.',m)
    original=github.post_verified_trusted_pr_protocol_comment_observed
    posted=[]
    def write(*a,**kw):
        if failure=='anchor' and posted and 'Visual result' in str(kw['body']):
            raise AgentLoopError('anchor unavailable')
        result=original(*a,**kw)
        posted.append(result.comment_id)
        return result
    # Fail specifically at the final prepared anchor, after every sidecar write.
    prepared=github.prepare_round_comment(body)
    assert len(prepared)>1
    def counted(*a,**kw):
        if failure=='anchor' and len(posted)==len(prepared)-1: raise AgentLoopError('anchor unavailable')
        return write(*a,**kw)
    monkeypatch.setattr(github,'post_verified_trusted_pr_protocol_comment_observed',counted)
    if failure=='anchor':
        with pytest.raises(AgentLoopError): github.post_verified_trusted_pr_round_comment(runner,config=config,pr_number=77,
            body=body,expected_author_login='coding-review-agent-loop',expected_author_id=424242)
        assert len(posted)==len(prepared)-1
        assert _extract_round_metadata_records([SimpleNamespace(body=c['body']) for c in runner.pr_payload['comments']],flow='pr')==()
        return
    anchor=github.post_verified_trusted_pr_round_comment(runner,config=config,pr_number=77,
        body=body,expected_author_login='coding-review-agent-loop',expected_author_id=424242)
    assert anchor.comment_id==posted[-1] and len(posted)==len(prepared)
    comments=[SimpleNamespace(body=c['body']) for c in runner.pr_payload['comments']]
    if failure=='sidecar':
        with pytest.raises(AgentLoopError): _extract_round_metadata_records(comments[1:],flow='pr')
    else:
        restored=_extract_round_metadata_records(comments,flow='pr')
        assert len(restored)==1 and restored[0].metadata.visual_lifecycle==m.visual_lifecycle
        assert restored[0].metadata.canonical_reviewer_response==m.canonical_reviewer_response


def test_verified_pr_round_writer_refuses_untrusted_body_before_transport(tmp_path):
    from agent_loop_helpers import FakeRunner, make_config
    from coding_review_agent_loop.github import post_verified_trusted_pr_round_comment
    from coding_review_agent_loop.errors import AgentLoopError
    runner=FakeRunner()
    with pytest.raises(AgentLoopError,match='TrustedBody'):
        post_verified_trusted_pr_round_comment(runner,config=make_config(tmp_path),pr_number=77,
            body='untrusted text',expected_author_login='coding-review-agent-loop',expected_author_id=424242)
    assert runner.commands==[]
