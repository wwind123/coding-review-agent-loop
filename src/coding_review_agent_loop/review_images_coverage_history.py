"""Optional coverage codecs and authenticated current-head report projection."""
from __future__ import annotations

from .review_images_coverage import FIELDS, fail, requests, acceptances, validate_scope, image_ids, bounded_text
from .review_images_state import sha, reduce_history, History
from .errors import HumanDecisionRequiredError


def validate_record(value):
    if (not isinstance(value, dict) or not {'version', 'kind', 'repo', 'pr', 'head', 'payload'} == set(value)
            or type(value['version']) is not int or value['version'] != 1
            or value['kind'] not in {'attachment', 'intent', 'outcome', 'receipt'}
            or not isinstance(value['repo'], str) or type(value['pr']) is not int or value['pr'] <= 0
            or not sha(value['head']) or not isinstance(value['payload'], dict)):
        fail('invalid auxiliary record')
    payload = value['payload']
    kind = value['kind']
    keys = {'attachment': {'attachment'}, 'intent': {'scope', 'assignments'},
            'outcome': {'intent_id', 'scope', 'response'},
            'receipt': {'identity', 'head', 'voter', 'assessment_id', 'snapshot_digest', 'image_ids', 'source_id'}}[kind]
    if set(payload) != keys:
        fail('invalid auxiliary payload')
    if kind in {'intent', 'outcome'}:
        validate_scope(payload['scope'])
        if payload['scope']['head'] != value['head']:
            fail('scope head mismatch')
    if kind == 'intent':
        assignments = payload['assignments']
        if not isinstance(assignments, list) or not 0 < len(assignments) <= 16:
            fail('invalid coverage assignments')
        seen = set()
        for assigned in assignments:
            if (not isinstance(assigned, dict) or set(assigned) != {'identity', 'request_id'}
                    or not isinstance(assigned['identity'], str) or not assigned['identity'].startswith('screenshot:v1:')
                    or not sha(assigned['identity'][14:], 64) or assigned['identity'] in seen
                    or assigned['request_id'] not in payload['scope']['request_ids']):
                fail('invalid coverage assignment identity')
            seen.add(assigned['identity'])
        if {a['request_id'] for a in assignments} != set(payload['scope']['request_ids']):
            fail('incomplete coverage assignments')
    if kind == 'outcome':
        if type(payload['intent_id']) is not int or payload['intent_id'] <= 0:
            fail('invalid outcome source')
        from .review_images_coverage import decisions
        response = payload['response']
        # Shape validation has no image authority; actual IDs are checked
        # against the live admitted report at consumption.
        if not isinstance(response, dict) or not isinstance(response.get('decisions'), list):
            fail('invalid outcome response')
        ids = []
        for decision in response['decisions']:
            if isinstance(decision, dict) and isinstance(decision.get('image_ids'), list):
                ids.extend(v for v in decision['image_ids'] if isinstance(v, str))
        scope = payload['scope']
        synthetic = {'head': value['head'], 'assessment_id': scope['assessment_id'],
            'snapshot_digest': scope['snapshot_digest'], 'images': [{'id': v} for v in ids]}
        decisions(response, scope, synthetic)
    if kind == 'receipt':
        if (not isinstance(payload['identity'], str) or not payload['identity'].startswith('screenshot:v1:')
                or not sha(payload['identity'][14:], 64) or payload['head'] != value['head']
                or not sha(payload['assessment_id'], 32) or not sha(payload['snapshot_digest'], 64)
                or type(payload['source_id']) is not int or payload['source_id'] <= 0):
            fail('invalid resolution receipt')
        bounded_text(payload['voter'], 256)
        image_ids(payload['image_ids'])
    return value


def envelope(kind, config, pr, head, payload):
    return validate_record(dict(version=1, kind=kind, repo=config.repo, pr=pr, head=head, payload=payload))


def encode_fields(metadata):
    result = {}
    if metadata.screenshot_receipt_refs:
        result["screenshot_receipt_refs"] = list(metadata.screenshot_receipt_refs)
    if metadata.screenshot_requests:
        result[FIELDS[0]] = list(requests(list(metadata.screenshot_requests)))
    if metadata.screenshot_coverage_acceptances:
        result[FIELDS[1]] = list(acceptances(list(metadata.screenshot_coverage_acceptances)))
    if metadata.screenshot_coverage is not None:
        result['screenshot_coverage'] = validate_record(metadata.screenshot_coverage)
    return result


def decode_fields(payload):
    result = {}
    if "screenshot_receipt_refs" in payload:
        refs = payload["screenshot_receipt_refs"]
        if not isinstance(refs, list) or not refs or any(type(ref) is not int or ref <= 0 for ref in refs) or len(set(refs)) != len(refs):
            fail("invalid receipt references")
        result["screenshot_receipt_refs"] = tuple(refs)
    if FIELDS[0] in payload:
        result[FIELDS[0]] = requests(payload[FIELDS[0]])
    if FIELDS[1] in payload:
        result[FIELDS[1]] = acceptances(payload[FIELDS[1]])
    if 'screenshot_coverage' in payload:
        result['screenshot_coverage'] = validate_record(payload['screenshot_coverage'])
    return result


def report(history, *, repo, pr, head):
    """No implicit legacy upgrade, private-file authority or stale acceptance."""
    from .review_images_coverage_cache import available
    if not isinstance(history, History) or not history.complete:
        return None
    try:
        admission = reduce_history(history, repo=repo, pr=pr, head=head)
    except HumanDecisionRequiredError:
        return None
    terminal = admission.terminal
    if terminal is None:
        return None
    # Later heads must not alter this terminal's original-assessment lineage.
    prefix = History((r for r in history if r.index <= terminal.index), identities=history.identities,
                     repo=repo, pr=pr, complete=True)
    admitted = reduce_history(prefix, repo=repo, pr=pr, head=head)
    value = terminal.metadata.screenshot_coverage
    if not isinstance(value, dict):
        return None
    validate_record(value)
    if value['kind'] != 'attachment' or (value['repo'], value['pr'], value['head']) != (repo, pr, head):
        return None
    attachment = value['payload']['attachment']
    lifecycle = terminal.metadata.visual_lifecycle
    original = admitted.original
    if (not isinstance(attachment, dict) or original is None or attachment.get('head') != head
            or attachment.get('assessment_id') != lifecycle['attempt']
            or attachment.get('snapshot_digest') != lifecycle['capture']
            or attachment.get('proof') != original[1]['proof']
            or attachment.get('source_id') != (original[0] if lifecycle['kind'] == 'skip' else None)
            or not available(attachment)):
        return None
    return attachment


def records(history, *, repo, pr, head=None):
    if not isinstance(history, History) or not history.complete:
        fail('complete authenticated history is required')
    result = []
    for record in history:
        value = record.metadata.screenshot_coverage
        if value is None:
            continue
        validate_record(value)
        if (value['repo'], value['pr'], value['head']) != (repo, pr, record.metadata.subject):
            fail('foreign auxiliary binding')
        if head is None or value['head'] == head:
            result.append((record, value))
    return result
