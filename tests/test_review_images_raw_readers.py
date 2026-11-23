"""#2070: raw REST protocol readers ignore visual records in a no-option run.

Managed-CI continuity, legacy repair reconciliation and board-lineage
validation decode owned REST bodies without going through PR comment
admission. Each must apply the same no-option visual mask, while an opted-in
run keeps today's strict decoding.
"""
import base64
import json
from dataclasses import replace

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_transport import ROUND_RESUME_MARKER_RE, decode_mapping
from coding_review_agent_loop.unresolved_items import _next_unresolved_item

KINDS = ['malformed-coverage', 'missing-coverage-sidecar', 'oversized-item-id']


def visual_body(kind, head='a' * 40):
    """An owned visual record that today's strict codec cannot decode."""
    item = replace(_next_unresolved_item(item_number=7, reviewer='Codex', source_round=1,
        text='Old visual blocker', status='blocking'), origin_scope='visible-defects')
    body = _attach_round_metadata('Visual assessment.', PostedRoundMetadata(
        flow='pr', role='reviewer', agent='Codex', round_number=1, subject=head, state='blocking',
        phase='visual', acquisition_outcome='success', new_items=(item,),
        review_scope={'version': 1, 'kind': 'visible-defects', 'head': head, 'seat': 'Codex'}))
    match = ROUND_RESUME_MARKER_RE.search(body)
    payload = decode_mapping(match.group('payload'))
    if kind == 'oversized-item-id':
        payload['new_items'][0]['item_id'] = 'item-' + '9' * 4301
    else:
        payload['screenshot_coverage'] = (
            {'$round_transport_spill': 'gone', 'parts': 1, 'sha256': '0' * 64, 'spill': '0' * 64}
            if kind == 'missing-coverage-sidecar' else {'version': 99})
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    return body[:match.start('payload')] + encoded + body[match.end('payload'):]


def _opted_in(config):
    return replace(config, review_images_requested=True)


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('opted_in', [False, True])
def test_managed_ci_continuity_reader_ignores_visual_record_without_options(tmp_path, kind, opted_in):
    """find_actor_round_metadata_comment_ids is what run_pr_loop calls after a managed head change."""
    from test_managed_ci import AuthorizationCommentRunner, label_event, _conflict_round_comment
    from coding_review_agent_loop import managed_ci_authorization as authorization
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    runner.intent_comments.append(_conflict_round_comment(61, subject='merged-head', round_number=13))
    runner.intent_comments.append({'id': 62, 'user': {'login': 'agent-loop', 'id': 1},
                                   'body': visual_body(kind, head='merged-head')})
    config = make_config(tmp_path)
    if opted_in:
        config = _opted_in(config)
    def select():
        return authorization.find_actor_round_metadata_comment_ids(
            runner, config=config, pr_number=7, actor_login='agent-loop', actor_id=1,
            predecessor_head='6367c48dd193d56ea7b0baad25b19455e529f5ee', new_head='merged-head',
            round_number=12, after_comment_id=50)
    if not opted_in:
        assert select() == (61,)
    elif kind == 'missing-coverage-sidecar':
        # Today's continuity codec refuses a missing sidecar; opted-in runs keep that.
        with pytest.raises(AgentLoopError, match='continuity round metadata is incomplete'):
            select()


@pytest.mark.parametrize('kind', KINDS)
def test_legacy_repair_reconciliation_ignores_visual_record_without_options(tmp_path, kind):
    from test_managed_ci_reconciliation import recovery
    from coding_review_agent_loop import managed_ci as ci
    from coding_review_agent_loop.managed_ci_reconciliation import resolve_legacy_reference
    config, runner, metadata, contract, checkpoint, item, ref = recovery(tmp_path, legacy=True)
    page = ci._read_intent_comment_page(runner, config, 7)
    baseline = resolve_legacy_reference(page, contract, checkpoint, config)
    assert baseline is not None
    page.append({'id': 9999, 'user': {'login': contract.trusted_actor_login, 'id': contract.trusted_actor_id},
                 'body': visual_body(kind)})
    assert resolve_legacy_reference(page, contract, checkpoint, config) == baseline
    if kind != 'oversized-item-id':
        with pytest.raises(AgentLoopError):
            resolve_legacy_reference(page, contract, checkpoint, _opted_in(config))


@pytest.mark.parametrize('kind', KINDS)
def test_board_lineage_validation_ignores_visual_record_without_options(tmp_path, kind):
    from test_board_change import _identity, _link_body, _record_body, _ACTOR_ID
    from coding_review_agent_loop.managed_ci_authorization import _historical_bindings_match
    root = _identity(('codex', 'codex'), ('gemini', 'gemini'))
    link, _digest = _link_body(root, root.digest, root, _identity(('codex', 'codex')), anchor=1, advance=True)
    actor = {'login': 'orchestrator', 'id': _ACTOR_ID}
    comments = [{'id': 1, 'body': _record_body(1, root, root), 'user': actor},
                {'id': 2, 'body': link, 'user': actor}]
    by_index = {0: {'seat_binding_valid': True, 'round_number': 1, 'seat_binding': root.seat_binding()}}
    config = make_config(tmp_path)
    def check(cfg):
        return _historical_bindings_match(comments, [1], by_index, actor_id=_ACTOR_ID, config=cfg)
    baseline = check(config)
    comments.append({'id': 3, 'body': visual_body(kind), 'user': actor})
    assert check(config) == baseline
    if kind != 'oversized-item-id':
        assert check(_opted_in(config)) is False
