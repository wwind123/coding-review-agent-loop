"""Strict, head-independent post-merge scheduling contracts.

Parsing is deliberately separate from admission. Model payloads are proposals;
only the caller's authenticated reviewer or live-verified operator can admit one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .errors import AgentLoopError

MAX_OBLIGATIONS = 64
FIELDS = frozenset({'requirement', 'rationale', 'owner', 'tracker_url'})
TARGET_FIELDS = frozenset({'target_kind', 'target_id', 'source_requirement_text'})
KINDS = frozenset({'evidence-item', 'converted-row', 'signed-requirement'})


def fail(message):
    raise AgentLoopError('Post-merge scheduling: ' + message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail('duplicate JSON key ' + key)
        result[key] = value
    return result


@dataclass(frozen=True)
class ObligationRequest:
    requirement: str
    rationale: str
    owner: str
    tracker_url: str
    target_kind: str | None = None
    target_id: str | None = None
    source_requirement_text: str | None = None
    matrix_identity: str | None = None

    def payload(self):
        return {key: value for key, value in asdict(self).items() if value is not None}


def parse_request(value, *, operator=False):
    if not isinstance(value, dict):
        fail('each obligation must be an object')
    target = bool(TARGET_FIELDS & value.keys())
    required = FIELDS | (TARGET_FIELDS if operator or target else frozenset())
    if value.get('target_kind') == 'converted-row':
        required |= {'matrix_identity'}
    if set(value) != required:
        fail('incomplete or unknown obligation fields')
    if any(not isinstance(v, str) or not v.strip() or len(v) > 20000 for v in value.values()):
        fail('all obligation fields must be non-empty bounded strings')
    if target and value['target_kind'] not in KINDS:
        fail('unsupported target kind')
    if target and value['source_requirement_text'] != value['requirement']:
        fail('requirement must exactly equal the full source requirement')
    if not re.fullmatch(r'https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/issues/[1-9][0-9]*', value['tracker_url']):
        fail('tracker must be a canonical existing GitHub issue URL')
    return ObligationRequest(**value)


def parse_requests(value, *, operator=False):
    if not isinstance(value, list) or len(value) > MAX_OBLIGATIONS or (operator and not value):
        fail('obligations must be a bounded list')
    requests = tuple(parse_request(entry, operator=operator) for entry in value)
    keys = [(r.target_kind, r.target_id) if r.target_id else ('new', r.requirement) for r in requests]
    if len(keys) != len(set(keys)):
        fail('duplicate obligation targets')
    return requests


def parse_review_fields(payload):
    requests = parse_requests(payload.get('post_merge_obligations', []))
    other = list(payload.get('exact_head_evidence_requests', []))
    for field in ('blocking_items', 'same_pr_followups', 'future_followups'):
        for item in payload.get(field, []):
            other.append(item if isinstance(item, str) else item.get('text') if isinstance(item, dict) else None)
    if any(r.requirement in other for r in requests):
        fail('contradictory pre-merge and post-merge classifications')
    return requests


def preserve_fields(source, target):
    if (source or {}).get('post_merge_obligations', []) != (target or {}).get('post_merge_obligations', []):
        fail('repair must preserve obligation classification, targets and payload exactly')


def operator_directive(requirement):
    """Only github.py's live-record verification supplies verified requirements."""
    from markdown_it import MarkdownIt
    body = requirement.body
    if 'post-merge-schedule' not in body:
        return (), body
    matches = [token for token in MarkdownIt('commonmark').parse(body)
               if token.type == 'fence' and token.info.strip() == 'post-merge-schedule']
    if requirement.author_verification != 'verified' or type(requirement.author_id) is not int or requirement.author_id <= 0 or not requirement.url:
        fail('operator directive requires a live-verified login, numeric actor ID and comment locator')
    if requirement.source_type.lower() not in {'pr comment', 'pr-comment', 'comment'}:
        fail('operator directive must be a signed PR comment')
    if len(matches) != 1 or not matches[0].markup.startswith('`'):
        fail('exactly one unambiguous post-merge-schedule block is required')
    start, end = matches[0].map
    lines = body.splitlines()
    if not re.fullmatch(r'`{' + str(len(matches[0].markup)) + r',}[ \t]*', lines[end - 1].lstrip()):
        fail('directive fence must have a closing delimiter')
    try:
        payload = json.loads(matches[0].content, object_pairs_hook=strict_object)
    except (ValueError, TypeError) as exc:
        fail('malformed directive JSON: ' + str(exc))
    if not isinstance(payload, dict) or set(payload) != {'obligations'}:
        fail('directive must contain only obligations')
    prose = '\n'.join((*lines[:start], *lines[end:])).strip()
    return parse_requests(payload['obligations'], operator=True), prose


def tracker_number(url, repo):
    prefix = f'https://github.com/{repo}/issues/'
    if not url.startswith(prefix) or not re.fullmatch(r'[1-9][0-9]*', url[len(prefix):]):
        fail('tracker must be an issue in the same repository')
    return int(url[len(prefix):])


def validate_tracker(request, *, repo, closing_issues, read_issue, allow_closed=False):
    number = tracker_number(request.tracker_url, repo)
    if number in closing_issues:
        fail(f'{request.target_id or request.requirement}: tracker #{number} is in the PR closing set')
    issue = read_issue(number)
    if not isinstance(issue, dict) or issue.get('number') != number or 'pull_request' in issue:
        fail(f'tracker #{number} is missing or is a pull request')
    if issue.get('state') not in ({'open', 'closed'} if allow_closed else {'open'}):
        fail(f'tracker #{number} must be an existing open issue')
    return number, issue.get('state')


def source_for_item(item):
    if item.converted_row_id:
        return {'target_kind': 'converted-row', 'target_id': item.converted_row_id,
                'matrix_identity': item.converted_matrix_identity, 'requirement': item.text}
    return {'target_kind': 'evidence-item', 'target_id': item.obligation_identity or item.item_id,
            'requirement': item.text, 'requester': item.reviewer,
            'owners': list(item.resolution_owners or (item.reviewer,))}


def authorize(request, *, sources, reviewer=None, voters=(), board=(), operator=None):
    """Resolve exact sources, without fuzzy matching or model asserted authority."""
    if request.target_id is None:
        if reviewer is None or reviewer not in board or operator is not None:
            fail('new declarations require a selected validated reviewer')
        return {'target_kind': 'declaration', 'target_id': digest([reviewer, request.requirement]),
                'requirement': request.requirement, 'requester': reviewer}
    key = (request.target_kind, request.target_id)
    source = sources.get(key)
    if source is None or source['requirement'] != request.requirement:
        fail(f'{request.target_id}: unknown/stale target or source text mismatch')
    if request.matrix_identity != source.get('matrix_identity'):
        fail(f'{request.target_id}: matrix identity mismatch')
    if operator is None:
        if reviewer not in board or not set(voters) <= set(board):
            fail('scheduling requires selected validated reviewers')
        if request.target_kind != 'evidence-item':
            fail(f'{request.target_id}: only a verified operator may schedule this target')
        owners = set(source.get('owners', (source.get('requester'),)))
        required = owners if owners <= set(board) else set(board)
        if not required or not required <= set(voters):
            fail(f'{request.target_id}: every resolution owner must approve the same payload')
    return dict(source)


def make_admission(request, *, repo, pr, source, provenance, supersedes=None):
    payload = request.payload()
    identity_payload = [repo, pr, source, payload]
    if supersedes is not None:
        identity_payload.extend([provenance, supersedes])
    return {'identity': 'post-merge:' + digest(identity_payload),
            'repo': repo, 'pr': pr, 'source': source, 'request': payload,
            'provenance': provenance, 'supersedes': supersedes,
            'state': 'scheduled', 'receipt': None}


def validate_registry(value, *, repo=None, pr=None):
    if not isinstance(value, list) or len(value) > MAX_OBLIGATIONS:
        fail('invalid cumulative registry')
    seen = set()
    for admission in value:
        if not isinstance(admission, dict) or set(admission) != {
                'identity', 'repo', 'pr', 'source', 'request', 'provenance', 'supersedes', 'state', 'receipt'}:
            fail('malformed registry admission')
        request = parse_request(admission['request'])
        source = admission['source']
        provenance = admission['provenance']
        if not isinstance(source, dict) or not isinstance(provenance, dict) or not provenance:
            fail('unreconstructible admission provenance')
        if source.get('requirement') != request.requirement:
            fail('admission source text mismatch')
        # Retain authenticated legacy payload identities while binding new
        # supersessions to their authority and predecessor (including A-B-A).
        identities = {make_admission(request, repo=admission['repo'], pr=admission['pr'],
                source=source, provenance=provenance, supersedes=predecessor)['identity']
                for predecessor in (None, admission['supersedes'])}
        if admission['identity'] not in identities or admission['identity'] in seen:
            fail('mutated or duplicated admission identity')
        if admission['supersedes'] is not None and not any(
                old['identity'] == admission['supersedes'] and old['source'] == source
                for old in value[:len(seen)]):
            fail('supersession requires an earlier admission for the same source')
        if repo is not None and (admission['repo'] != repo or admission['pr'] != pr):
            fail('foreign PR admission')
        if provenance.get('kind') == 'operator':
            from .github import HumanReviewRequirement
            try:
                original = HumanReviewRequirement(source_type='PR comment',
                    author=provenance['actor'], author_id=provenance['actor_id'],
                    created_at=None, url=provenance['locator'], body=provenance['body'],
                    author_verification='verified')
                requests, _ = operator_directive(original)
            except (KeyError, TypeError):
                fail('unreconstructible operator provenance')
            if request not in requests or not provenance.get('source_id'):
                fail('admission does not match its original operator directive')
        elif provenance.get('kind') == 'reviewer':
            if not isinstance(provenance.get('reviewers'), list) or not provenance['reviewers']:
                fail('missing reviewer admission provenance')
            if source['target_kind'] not in {'declaration', 'evidence-item'}:
                fail('reviewer provenance cannot schedule operator-only targets')
        else:
            fail('unknown admission authority')
        if admission['state'] not in {'scheduled', 'tracking-pending', 'tracked'}:
            fail('unknown publication state')
        if (admission['state'] == 'tracked') != (type(admission['receipt']) is int and admission['receipt'] > 0):
            fail('tracked state requires a verified publication receipt')
        seen.add(admission['identity'])
    return tuple(value)


def active_registry(registry):
    superseded = {a['supersedes'] for a in registry if a['supersedes']}
    return tuple(a for a in registry if a['identity'] not in superseded)


def scheduled_item(item, registry):
    # Exact source binding survives a head change; neither prose nor a new item
    # from another requester can inherit scheduling by semantic similarity.
    if item.obligation_kind != 'human-exact-head-evidence':
        return False
    source = source_for_item(item)
    if source['target_kind'] == 'converted-row':
        return any(a['source']['target_kind'] == 'converted-row'
                   and a['source']['target_id'] == source['target_id']
                   and a['source'].get('matrix_identity') == source['matrix_identity']
                   for a in active_registry(registry))
    if set(item.resolution_owners or (item.reviewer,)) == {item.reviewer} and any(a['source']['target_kind'] == 'declaration' and a['source'].get('requester') == item.reviewer
           and a['request']['requirement'] == item.text for a in active_registry(registry)):
        return True
    return any(a['source'] == source for a in active_registry(registry))


def filter_items(items, registry):
    return [item for item in items if not scheduled_item(item, registry)]


def scheduled_rows(registry, matrix_identity):
    return {a['source']['target_id'] for a in active_registry(registry)
            if a['source']['target_kind'] == 'converted-row'
            and a['source'].get('matrix_identity') == matrix_identity}


def render_context(registry, diagnostics=()):
    if not registry and not diagnostics:
        return ''
    lines = ['\nAdmitted post-merge obligations (required, not fulfilled):']
    for a in active_registry(registry):
        r = a['request']
        lines.append(f"- {a['identity']}; source={json.dumps(a['source'], sort_keys=True)}; "
                     f"requirement={r['requirement']}; owner={r['owner']}; tracker={r['tracker_url']}; status={a['state']}")
    lines.append('Scheduling survives head changes. Do not re-raise an unchanged admitted source as a pre-merge or exact-head demand. Code approval and CI still require the current head.')
    lines.extend(diagnostics)
    return '\n'.join(lines)


def diagnostic(target, error):
    return (f'Post-merge request {target} rejected: {error}. Original requirement remains blocking. '
            'Supply a non-empty owner and an existing open same-repository tracking issue outside the PR closing set; '
            'manually create/select a tracker and retry through the requesting reviewers or a verified signed post-merge-schedule directive. No coder is dispatched to obtain trackers.')


def retained_diagnostic(target, error):
    return (f'Admitted post-merge obligation {target} retained: {error}. '
            'The obligation remains required after merge and does not block merge. '
            'If needed, select an eligible tracker through a new authorized scheduling request; '
            'use the requesting reviewers or a verified signed post-merge-schedule directive.')
