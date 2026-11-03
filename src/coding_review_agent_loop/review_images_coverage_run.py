"""Additive text-only requester coverage admission and evidence-gate consumers."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, replace

from .errors import AgentLoopError
from .review_images_coverage import (KIND, fail, upsert, decisions, validate_vote,
                                     render_context, request_identity)
from .review_images_coverage_history import envelope, records, report
from .review_images_state import digest


@dataclass(frozen=True)
class CoverageTransition:
    action: str
    context: object = None
    round_number: int | None = None

    def __bool__(self):
        return self.action == 'coverage_satisfied'


def boundary(runner, *, config, pr, head, baseline, collect_requirement_ids):
    from .pr_head_sync import get_observed_pr_review_context
    context = get_observed_pr_review_context(runner, config=config, pr_number=pr)
    if context.metadata.head_sha != head:
        return CoverageTransition('head_changed', context)
    if collect_requirement_ids is not None and set(collect_requirement_ids(context)) - set(baseline):
        return CoverageTransition('refresh', context)
    return CoverageTransition('coverage_satisfied', context)


def stage_transition(result, *, refresh, head_changed):
    if not isinstance(result, CoverageTransition) or result.action not in {'refresh', 'head_changed'}:
        return False
    (refresh if result.action == 'refresh' else head_changed)(result.context)
    return True


def board(config):
    from .config import reviewers
    from .agents.registry import agent_display_name
    return {agent_display_name(agent): agent for agent in reviewers(config)}


def history_for(runner, config, pr):
    from .review_images import load_history
    return load_history(runner, config, pr)


def contract_for(config, agent):
    from .config import resolve_invocation
    from .reviewer_seats import seat_backend, reviewer_seat_binding
    invocation = resolve_invocation(config, provider=agent, role='reviewer')
    return digest({'board': reviewer_seat_binding(config), 'seat': str(agent),
        'backend': seat_backend(agent), 'model': invocation.configured_model,
        'effort': invocation.resolved_effort})


def required_voters(item, current_board):
    requester = item.screenshot_request['requester']
    return (requester,) if requester in current_board else tuple(current_board)


def restore_requests(runner, *, config, pr, head, items, number, clearances=(), comments=None):
    if comments is not None and (not config.review_images_paths or config.dry_run):
        # Reuse the observed snapshot; disabled acquisition adds no head polling.
        from .review_images import admitted_visual_history
        history = admitted_visual_history(runner, config, pr, comments)
    else:
        history = history_for(runner, config, pr)
    return restore(history, config=config, pr=pr, head=head, items=items, number=number, clearances=clearances)


def restore(history, *, config, pr, head, items, number, clearances=()):
    from .review_scope import is_complete_code_review
    from .capture_acceptance import reviewer_contract
    if (not config.review_images_paths or config.dry_run) and not any(item.screenshot_request for item in items) and not any(
            r.metadata.screenshot_requests or any(item.screenshot_request
                for item in (*r.metadata.prior_items, *r.metadata.new_items)) for r in history):
        return list(items), number
    clearances = human_clearances(history, head, clearances)
    result = list(items)
    # Consume the complete history, including reserved/resolved visual IDs.
    from .round_state import _max_unresolved_item_number_from_records
    number = max(number, _max_unresolved_item_number_from_records(history) + 1)
    historical = {}
    for record in history:
        for item in (*record.metadata.prior_items, *record.metadata.new_items):
            if item.screenshot_request is not None:
                key = item.obligation_identity
                if key in historical and historical[key].screenshot_request != item.screenshot_request:
                    fail('historical request provenance changed')
                historical.setdefault(key, item)
    known = {item.obligation_identity for item in result}
    result.extend(item for key, item in historical.items() if key not in known)
    for record in history:
        metadata = record.metadata
        if not is_complete_code_review(metadata):
            continue
        for request in metadata.screenshot_requests:
            result, consumed = upsert(result, repo=config.repo, pr=pr, requester=metadata.agent,
                request=request, contract=reviewer_contract(metadata), source=history.identities.get(record.index),
                number=number, round_number=metadata.round_number, head=head, clearances=clearances)
            number += consumed
    if not any(item.screenshot_request for item in result):
        return result, number
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    receipts = verified_receipts(history, attachment, result, config=config, pr=pr, head=head)
    result = apply_receipts(result, receipts, board(config), head,
                            human_clearances=clearances)
    return result, number


def vote_candidates(history, attachment, items, *, config, pr, head):
    """Derive decisions from original authenticated sources, never from receipts."""
    from .review_scope import is_complete_code_review
    if attachment is None:
        return []
    by_identity = {item.obligation_identity: item for item in items if item.screenshot_request}
    candidates = []
    current_board = board(config)
    intents = {}
    auxiliary = records(history, repo=config.repo, pr=pr, head=head)
    for record, value in auxiliary:
        payload = value['payload']
        identity = history.identities.get(record.index)
        if value['kind'] == 'intent':
            if record.metadata.agent != payload['scope']['requester'] or record.metadata.review_scope != payload['scope']:
                continue
            intents[identity] = (record, payload)
        elif value['kind'] == 'outcome':
            intent = intents.get(payload['intent_id'])
            if intent is None or payload['scope'] != intent[1]['scope']:
                continue
            scope = payload['scope']
            try:
                decisions(payload['response'], scope, attachment)
            except AgentLoopError:
                continue
            if record.metadata.agent != scope['requester'] or record.metadata.review_scope != scope:
                continue
            for decision in payload['response']['decisions']:
                if not decision['accepted']:
                    continue
                assigned = next((a for a in intent[1]['assignments']
                                 if a['request_id'] == decision['request_id']), None)
                item = by_identity.get(assigned['identity']) if assigned else None
                if item is None or scope['requester'] not in required_voters(item, current_board):
                    continue
                candidates.append({'identity': item.obligation_identity, 'head': head,
                    'voter': scope['requester'], 'assessment_id': attachment['assessment_id'],
                    'snapshot_digest': attachment['snapshot_digest'], 'image_ids': decision['image_ids'],
                    'source_id': identity})
    for record in history:
        m = record.metadata
        if m.subject != head or not is_complete_code_review(m):
            continue
        for vote in m.screenshot_coverage_acceptances:
            try:
                validate_vote(vote, attachment)
            except AgentLoopError:
                continue
            identity = request_identity(config.repo, pr, m.agent, vote['request_id'])
            item = by_identity.get(identity)
            if item is not None and m.agent in required_voters(item, current_board):
                candidates.append({'identity': identity, 'head': head, 'voter': m.agent,
                    'assessment_id': vote['assessment_id'], 'snapshot_digest': vote['snapshot_digest'],
                    'image_ids': vote['image_ids'], 'source_id': history.identities.get(record.index)})
    return candidates


def verified_receipts(history, attachment, items, *, config, pr, head):
    candidates = vote_candidates(history, attachment, items, config=config, pr=pr, head=head)
    return [value['payload'] for _, value in records(history, repo=config.repo, pr=pr, head=head)
            if value['kind'] == 'receipt' and value['payload'] in candidates]


def human_clearances(history, head, clearances=()):
    """Combine trusted live clearances with authenticated terminal carry state."""
    carried = {pair for r in history
               if (r.metadata.evidence_freeze is not None and r.metadata.evidence_freeze.valid
                   or r.metadata.evidence_release is not None and r.metadata.evidence_release.valid)
               for pair in r.metadata.evidence_clearances if pair[1] == head}
    return carried | {pair for pair in clearances if pair[1] == head}


def apply_receipts(items, receipts, current_board, head, *, human_clearances=()):
    result = []
    for item in items:
        if item.screenshot_request is None:
            result.append(item)
            continue
        voters = required_voters(item, current_board)
        received = {r['voter'] for r in receipts if r['identity'] == item.obligation_identity and r['head'] == head}
        satisfied = ((item.obligation_identity, head) in human_clearances
                     or bool(voters) and set(voters) <= received)
        result.append(replace(item, status='resolved' if satisfied else 'blocking',
            lifecycle='cleared' if satisfied else (item.lifecycle if item.lifecycle == 'evidence_frozen' and item.candidate_head_sha == head else 'evidence_deferred'),
            candidate_head_sha=head if satisfied or item.lifecycle == 'evidence_frozen' and item.candidate_head_sha == head else None))
    return result


def publish(runner, config, pr, head, round_number, kind, payload, *, agent='Orchestrator', scope=None):
    from .github import resolve_authenticated_github_actor, post_verified_trusted_pr_protocol_comment_observed
    from .round_state import PostedRoundMetadata, _attach_round_metadata
    actor = resolve_authenticated_github_actor(runner, config=config)
    metadata = PostedRoundMetadata(flow='pr', role='coverage-' + kind, agent=agent,
        round_number=round_number, subject=head, phase='screenshot-coverage-' + kind,
        review_scope=scope, screenshot_coverage=envelope(kind, config, pr, head, payload))
    return post_verified_trusted_pr_protocol_comment_observed(runner, config=config, pr_number=pr,
        body=_attach_round_metadata('Screenshot coverage audit: ' + kind, metadata),
        expected_author_login=actor[0], expected_author_id=actor[1])


def prompt_context(runner, *, config, pr, head, requester, items):
    if not config.review_images_paths or config.dry_run:
        return ''
    history = history_for(runner, config, pr)
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    assignments = [item.screenshot_request['request'] for item in items
                   if item.screenshot_request and item.screenshot_request['requester'] == requester
                   and item.status in {'blocking', 'same-pr'}]
    guidance = ('\nScreenshot evidence requests may use the optional screenshot_requests list '
                'with request_id, kind="screenshot", description. Each ID belongs to your seat and '
                'must keep its original description. Legacy requests remain human-only.\n')
    return guidance + render_context(attachment, assignments)


def validate_ordinary(parsed, *, runner, config, pr, head, requester, items, matrix=False):
    typed = parsed.screenshot_requests or parsed.screenshot_coverage_acceptances
    if typed and (not config.review_images_paths or matrix):
        fail('typed authority is disabled in this turn')
    if not typed:
        return parsed
    known = {item.screenshot_request['request']['request_id']: item.screenshot_request['request']
             for item in items if item.screenshot_request and item.screenshot_request['requester'] == requester}
    for request in parsed.screenshot_requests:
        if request['request_id'] in known and known[request['request_id']] != request:
            fail('changed request identity')
        known[request['request_id']] = request
    if parsed.screenshot_coverage_acceptances:
        history = history_for(runner, config, pr)
        attachment = report(history, repo=config.repo, pr=pr, head=head)
        if attachment is None:
            fail('no available authenticated current-head report')
        for vote in parsed.screenshot_coverage_acceptances:
            if vote['request_id'] not in known:
                fail('acceptance names an unassigned request')
            validate_vote(vote, attachment)
    return parsed


def gate(runner, *, config, pr, head, items, round_number, usage_context=None, clearances=()):
    """Run only after the caller's machine and signed-input gates.

    Reservation is the admission boundary. A restart may publish original
    outcomes/receipts, but can never repeat a reserved acquisition.
    """
    from .pr_head_sync import get_observed_pr_review_context
    from .validated_agent import _run_validated_agent
    from .logging import log
    def live_head():
        return get_observed_pr_review_context(runner, config=config, pr_number=pr).metadata.head_sha
    history = history_for(runner, config, pr)
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    if not config.review_images_paths or config.dry_run:
        # Activation controls acquisition, never the validity of carried evidence.
        receipts = verified_receipts(history, attachment, items, config=config, pr=pr, head=head)
        updated = apply_receipts(items, receipts, board(config), head,
            human_clearances=human_clearances(history, head, clearances))
        from .unresolved_items import _pending_evidence_obligations
        return updated, 'pending' if _pending_evidence_obligations(updated) else 'coverage_satisfied'
    if attachment is None:
        return apply_receipts(items, (), board(config), head, human_clearances=human_clearances(history, head, clearances)), 'pending'
    current_board = board(config)
    updated = list(items)
    for seat, agent in current_board.items():
        if live_head() != head:
            return updated, 'head_changed'
        history = history_for(runner, config, pr)
        # Revalidate availability before every acquisition/publication.
        attachment = report(history, repo=config.repo, pr=pr, head=head)
        if attachment is None:
            return apply_receipts(updated, (), current_board, head, human_clearances=human_clearances(history, head, clearances)), 'pending'
        candidates = vote_candidates(history, attachment, updated, config=config, pr=pr, head=head)
        existing = verified_receipts(history, attachment, updated, config=config, pr=pr, head=head)
        for candidate in candidates:
            if candidate not in existing:
                publish(runner, config, pr, head, round_number, 'receipt', candidate)
                existing.append(candidate)
        updated = apply_receipts(updated, existing, current_board, head, human_clearances=human_clearances(history, head, clearances))
        pending = [item for item in updated if item.screenshot_request
                   and item.status in {'blocking', 'same-pr'} and seat in required_voters(item, current_board)
                   and not any(r['identity'] == item.obligation_identity and r['voter'] == seat for r in existing)]
        if not pending:
            continue
        auxiliary = records(history, repo=config.repo, pr=pr, head=head)
        if any(value['kind'] == 'intent' and value['payload']['scope']['requester'] == seat
               for _, value in auxiliary):
            continue
        # The response schema's bound is also the batching bound; overflow
        # cannot create a second turn or silently truncate assignments.
        if len(pending) > 16:
            log(config, 'Screenshot coverage remains pending: more than 16 assignments')
            continue
        assignments = [{'identity': item.obligation_identity,
                        'request_id': item.obligation_identity}
                       for item in pending]
        ids = [a['request_id'] for a in assignments]
        if len(set(ids)) != len(ids):
            continue  # Ambiguous removed-requester IDs cannot authorize a vote.
        scope = dict(version=1, kind=KIND, head=head, assessment_id=attachment['assessment_id'],
                     snapshot_digest=attachment['snapshot_digest'], requester=seat,
                     request_ids=ids, contract=contract_for(config, agent))
        try:
            prompt = render_context(attachment, [dict(item.screenshot_request['request'],
                request_id=item.obligation_identity) for item in pending])
            prompt += ('\nDedicated text-only coverage. Do not inspect files, run tools, review code, '
                'create findings or request evidence. Return only {"version":1,"kind":"screenshot-coverage",'
                '"decisions":[{"request_id":"assigned ID","accepted":false,"image_ids":[]}]}. '
                'Return one explicit decision per assigned request. Acceptance must name manifest image IDs.\n'
                + json.dumps(scope))
        except AgentLoopError as exc:
            log(config, str(exc))
            continue
        intent = publish(runner, config, pr, head, round_number, 'intent',
                         {'scope': scope, 'assignments': assignments}, agent=seat, scope=scope)
        if live_head() != head:
            return updated, 'head_changed'
        try:
            response = _run_validated_agent(runner, agent=agent, config=config, prompt=prompt,
                role='reviewer', session_id=None, timeout_seconds=300, label=KIND,
                coverage_scope=scope, coverage_report=attachment, usage_context=usage_context,
                coverage_acquisition_id=digest([config.repo, pr, head, seat, intent.comment_id]))
            if live_head() != head:
                return updated, 'head_changed'
            publish(runner, config, pr, head, round_number, 'outcome',
                {'intent_id': intent.comment_id, 'scope': scope, 'response': json.loads(response.text)},
                agent=seat, scope=scope)
        except (AgentLoopError, ValueError, OSError, TimeoutError, subprocess.SubprocessError) as exc:
            log(config, 'Screenshot coverage remains pending; reserved attempt is spent: ' + str(exc)[:512])
    history = history_for(runner, config, pr)
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    if live_head() != head:
        return updated, 'head_changed'
    candidates = vote_candidates(history, attachment, updated, config=config, pr=pr, head=head)
    existing = verified_receipts(history, attachment, updated, config=config, pr=pr, head=head)
    for candidate in candidates:
        if candidate not in existing:
            publish(runner, config, pr, head, round_number, 'receipt', candidate)
            existing.append(candidate)
    updated = apply_receipts(updated, existing, current_board, head, human_clearances=human_clearances(history, head, clearances))
    from .unresolved_items import _pending_evidence_obligations
    return updated, 'pending' if _pending_evidence_obligations(updated) else 'coverage_satisfied'


def terminal_fields(runner, *, config, pr, head, items, reason, refs=()):
    if reason not in {'refresh-clean', 'evidence-cleared'} or not config.review_images_paths:
        return reason, refs
    from .unresolved_items import _pending_evidence_obligations
    if _pending_evidence_obligations(items) or not any(item.screenshot_request and item.status == 'resolved' for item in items):
        return reason, refs
    history = history_for(runner, config, pr)
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    verified = verified_receipts(history, attachment, items, config=config, pr=pr, head=head)
    identities = {item.obligation_identity for item in items if item.screenshot_request and item.status == 'resolved'}
    selected = tuple(history.identities[r.index] for r, value in records(history, repo=config.repo, pr=pr, head=head)
                     if value['kind'] == 'receipt' and value['payload'] in verified
                     and value['payload']['identity'] in identities)
    return (KIND, selected) if selected else (reason, refs)


def release(runner, *, config, pr, head, items, round_number, baseline, clearances=(), references=()):
    from .pr_loop_support import _publish_evidence_release
    from .round_state import EVIDENCE_RELEASE_PHASE
    history = history_for(runner, config, pr)
    attachment = report(history, repo=config.repo, pr=pr, head=head)
    verified = verified_receipts(history, attachment, items, config=config, pr=pr, head=head)
    refs = tuple(history.identities[r.index] for r, value in records(history, repo=config.repo, pr=pr, head=head)
                 if value['kind'] == 'receipt' and value['payload'] in verified)
    prior = next((r for r in history if r.metadata.phase == EVIDENCE_RELEASE_PHASE
                  and r.metadata.subject == head and r.metadata.round_number == round_number
                  and r.metadata.evidence_release is not None
                  and r.metadata.evidence_release.reason == KIND
                  and set(r.metadata.evidence_release.signed_requirement_ids_surfaced) == set(baseline.signed_requirement_ids_at_freeze)
                  and r.metadata.screenshot_receipt_refs == refs), None)
    if prior is not None:
        return list(items)
    if not refs:
        return list(items)  # Existing signed-human clearance remains independent.
    return _publish_evidence_release(runner, config=config, pr_number=pr, head_sha=head,
        items=items, round_number=round_number, reason=KIND,
        surfaced_requirement_ids=baseline.signed_requirement_ids_at_freeze,
        allowed_rounds=baseline.allowed_rounds,
        watch_failure_extension_used=baseline.watch_failure_extension_used,
        watch_head_extension_used=baseline.watch_head_extension_used,
        clearances=clearances, signed_evidence_references=references, screenshot_receipt_refs=refs)


def resume_frozen(runner, *, config, pr, head, items, round_number, baseline, clearances, references,
                  collect_requirement_ids=None):
    if not any(item.screenshot_request for item in items):
        return CoverageTransition('pending')
    updated, action = gate(runner, config=config, pr=pr, head=head, items=items,
        round_number=round_number, usage_context=getattr(runner, '_screenshot_coverage_usage', None),
        clearances=clearances)
    items[:] = updated
    refreshed = boundary(runner, config=config, pr=pr, head=head,
        baseline=baseline.signed_requirement_ids_at_freeze, collect_requirement_ids=collect_requirement_ids)
    if refreshed.action in {'refresh', 'head_changed'}:
        return refreshed
    if action != 'coverage_satisfied':
        return CoverageTransition('pending')
    history = history_for(runner, config, pr)
    terminal_round = max([round_number, *(r.metadata.round_number + 1 for r in history
        if r.metadata.subject == head and (r.metadata.evidence_freeze or r.metadata.evidence_release))])
    items[:] = release(runner, config=config, pr=pr, head=head, items=updated, round_number=terminal_round,
        baseline=baseline, clearances=clearances, references=references)
    return CoverageTransition('coverage_satisfied', refreshed.context, terminal_round)


def final_gate(runner, *, config, pr, head, items, round_number, live_ids, allowed_rounds,
               failure_extension, head_extension, clearances, references, collect_requirement_ids=None):
    """Return the explicit transition; the existing caller owns final validation."""
    if not any(item.screenshot_request for item in items):
        return CoverageTransition('pending')
    updated, action = gate(runner, config=config, pr=pr, head=head, items=items,
        round_number=round_number, usage_context=getattr(runner, '_screenshot_coverage_usage', None),
        clearances=clearances)
    items[:] = updated
    refreshed = boundary(runner, config=config, pr=pr, head=head, baseline=live_ids,
                         collect_requirement_ids=collect_requirement_ids)
    if refreshed.action in {'refresh', 'head_changed'}:
        return refreshed
    if action == 'coverage_satisfied':
        from types import SimpleNamespace
        baseline = SimpleNamespace(signed_requirement_ids_at_freeze=tuple(live_ids), allowed_rounds=allowed_rounds,
            watch_failure_extension_used=failure_extension, watch_head_extension_used=head_extension)
        items[:] = release(runner, config=config, pr=pr, head=head, items=items, round_number=round_number,
            baseline=baseline, clearances=clearances, references=references)
    return CoverageTransition(action, refreshed.context)
