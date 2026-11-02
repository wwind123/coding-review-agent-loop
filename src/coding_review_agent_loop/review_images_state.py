"""Authenticated visual lifecycle reduction; shared transport stays fail closed.

Callers authenticate candidates before constructing History. No local capture
or model response is durable authority. A persisted intent spends the head even
when no process was started. Independent simultaneous orchestrators are outside
this comment-based protocol's coordination boundary.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
import uuid

from .errors import HumanDecisionRequiredError
from .review_scope import lifecycle_present, is_visual_review, visual_candidate


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def configuration_identity(settings):
    return digest(asdict(settings))


def comparison_identity(snapshots, configuration):
    manifest = json.loads(snapshots.manifest_json)
    manifest.pop('head_sha')
    manifest['images'] = sorted(manifest['images'], key=lambda image: image['id'])
    files = sorted((file.path, file.digest, file.width, file.height, file.size)
                   for file in snapshots.files)
    # File digests were computed by the parent from sealed bytes, not supplied
    # by the capture worker. Associations and multiplicity remain in manifest.
    return digest({'manifest': manifest, 'files': files, 'configuration': configuration})


def reviewer_contract(capability):
    fields = ('seat', 'provider', 'model', 'effort', 'model_chain',
              'executable_digest', 'permission_mode', 'contract_id', 'version')
    result = {key: getattr(capability, key, None) for key in fields}
    result.update(transport=1, proof=1, prompt=1, schema=1)
    return json.loads(json.dumps(result))


def compatible_contract(contract):
    return (isinstance(contract, dict) and set(contract) == {
        'seat', 'provider', 'model', 'effort', 'model_chain', 'executable_digest',
        'permission_mode', 'contract_id', 'version', 'transport', 'proof', 'prompt', 'schema'}
        and all(isinstance(contract[key], str) and contract[key] for key in
                ('seat', 'provider', 'model', 'effort', 'executable_digest',
                 'permission_mode', 'contract_id', 'version'))
        and isinstance(contract['model_chain'], list) and bool(contract['model_chain'])
        and all(isinstance(model, str) and model for model in contract['model_chain'])
        and all(type(contract[key]) is int and contract[key] == 1
                for key in ('transport', 'proof', 'prompt', 'schema')))


class History(tuple):
    def __new__(cls, records=(), *, identities=None, complete=True, diagnostic='', repo=None, pr=None):
        result = super().__new__(cls, records)
        result.identities = identities or {}
        result.repo, result.pr = repo, pr
        result.complete = complete
        result.diagnostic = diagnostic
        return result


def sha(value, length=40):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None


def reliable_head(metadata):
    """Use the independent round subject; reject contradictory head claims."""
    head = metadata.subject
    if not sha(head):
        return None
    value = metadata.visual_lifecycle
    if isinstance(value, dict) and 'head' in value and value['head'] != head:
        return None
    scope = metadata.review_scope
    if isinstance(scope, dict) and 'head' in scope and scope['head'] != head:
        return None
    return head


def owner_shares(items, seat=None):
    return sorted({(item.item_id, owner) for item in items
        for owner in (item.resolution_owners or (item.reviewer,))
        if (seat is None or owner == seat) and dict(item.owner_states).get(owner, 'pending') != 'cleared'})


COMMON = {'version', 'kind', 'repo', 'pr', 'head', 'attempt', 'seat', 'contract',
          'capture', 'configuration', 'comparison'}


def validate(record, *, repo, pr):
    m = record.metadata
    value = m.visual_lifecycle
    if not isinstance(value, dict) or len(json.dumps(value, allow_nan=False)) > 16384:
        raise ValueError('malformed lifecycle')
    kind = value.get('kind')
    extra = {'intent': set(), 'result': {'intent_id', 'proof'},
             'skip': {'source_id', 'original_id', 'assigned'}}.get(kind)
    if (extra is None or set(value) != COMMON | extra or type(value['version']) is not int
            or value['version'] != 1 or value['repo'] != repo or type(value['pr']) is not int
            or value['pr'] != pr or not sha(value['head']) or value['head'] != m.subject
            or not isinstance(value['attempt'], str) or not sha(value['attempt'], 32)
            or not isinstance(value['seat'], str) or value['seat'] != m.agent
            or not compatible_contract(value['contract'])
            or any(not sha(value[key], 64) for key in ('capture', 'configuration', 'comparison'))
            or m.review_scope != {'version': 1, 'kind': 'visible-defects',
                                  'head': value['head'], 'seat': value['seat']}):
        raise ValueError('invalid lifecycle contract or binding')
    for key in extra - {'proof', 'assigned'}:
        if type(value[key]) is not int or value[key] <= 0:
            raise ValueError('invalid lifecycle provenance')
    if kind == 'result':
        if (m.role != 'reviewer' or m.acquisition_outcome != 'success'
                or m.acquisition_returncode != 0 or m.state not in {'approved', 'blocking'}
                or not sha(value['proof'], 64)):
            raise ValueError('incomplete successful result')
        ids = [d.item_id for d in m.dispositions]
        if (len(set(ids)) != len(ids) or set(ids) != {item.item_id for item in m.prior_items}
                or any(item.origin_scope != 'visible-defects' for item in (*m.prior_items, *m.new_items))):
            raise ValueError('incomplete result assignment')
    elif kind == 'skip' and (not isinstance(value['assigned'], list)
            or any(not isinstance(share, list) or len(share) != 2
                   or any(not isinstance(part, str) or not part for part in share)
                   for share in value['assigned'])):
        raise ValueError('invalid skip assignment')
    if kind != 'result' and (m.prior_items or m.new_items or m.dispositions or m.state is not None):
        raise ValueError('non-assessment carries finding authority')
    return value


@dataclass(frozen=True)
class Admission:
    spent: bool = False
    terminal: object = None
    latest: object = None
    original: object = None
    diagnostic: str = ''


def reduce_history(history, *, repo, pr, head):
    """Strict validation happens only on authenticated, decodable candidates."""
    if not getattr(history, 'complete', True):
        raise HumanDecisionRequiredError('Visual history identity coverage is incomplete: ' + history.diagnostic)
    spent = False
    terminal = None
    latest = None
    original = None
    by_id = {}
    attempts = {}
    heads = set()
    invalid_heads = set()
    for record in history:
        m = record.metadata
        if not visual_candidate(m):
            continue
        bound = reliable_head(m)
        identity = history.identities.get(record.index)
        where = f'comment {identity or record.index}'
        if bound is None:
            raise HumanDecisionRequiredError('Visual lifecycle has no reliable head binding at ' + where)
        if bound == head:
            spent = True
        if not lifecycle_present(m):
            # Legacy publications spend their head but cannot authorize reuse.
            latest = original = None
            heads.add(bound)
            if not is_visual_review(m) and bound == head:
                raise HumanDecisionRequiredError('Malformed legacy visual scope at ' + where)
            continue
        try:
            value = validate(record, repo=repo, pr=pr)
            if type(identity) is not int or identity <= 0:
                raise ValueError('missing authenticated anchor identity')
            kind = value['kind']
            if kind in {'intent', 'skip'}:
                if bound in heads:
                    raise ValueError('conflicting per-head dispatch')
                heads.add(bound)
            if bound in invalid_heads:
                raise ValueError('earlier invalid lifecycle at this head')
            if kind == 'intent':
                if value['attempt'] in attempts:
                    raise ValueError('duplicate attempt')
                attempts[value['attempt']] = (identity, value)
                # An unknown latest attempt prevents fallback to older success.
                latest = original = None
            elif kind == 'result':
                intent = attempts.get(value['attempt'])
                if (intent is None or intent[0] != value['intent_id']
                        or any(intent[1][key] != value[key] for key in COMMON - {'kind'})
                        or any(v['kind'] == 'result' and v['attempt'] == value['attempt']
                               for _, v, _ in by_id.values())):
                    raise ValueError('orphan or conflicting result')
                latest = original = (identity, value, record)
                if bound == head:
                    terminal = record
            else:
                if latest is None or original is None or value['source_id'] != latest[0] or value['original_id'] != original[0]:
                    raise ValueError('broken skip provenance')
                source = latest[1]
                if any(source[key] != value[key] for key in ('comparison', 'configuration', 'contract', 'seat')):
                    raise ValueError('incompatible skip')
                coverage = set(owner_shares((*original[2].metadata.prior_items, *original[2].metadata.new_items)))
                # Skip has no ledger mutation; the original's coverage never expands.
                if not coverage.issuperset({tuple(share) for share in value['assigned']}):
                    raise ValueError('invalid source ownership')
                latest = (identity, value, record)
                if bound == head:
                    terminal = record
            by_id[identity] = (identity, value, record)
        except (ValueError, TypeError, KeyError) as exc:
            invalid_heads.add(bound)
            latest = original = None
            if bound == head:
                raise HumanDecisionRequiredError(f'Visual lifecycle refused at {where}: {exc}') from exc
    return Admission(spent, terminal, latest, original,
        'durable visual opportunity is spent; completion is unknown' if spent and terminal is None else '')


def can_reuse(admission, value, assigned):
    if admission.latest is None or admission.original is None:
        return False
    source = admission.latest[1]
    original = admission.original[2].metadata
    covered = set(owner_shares((*original.prior_items, *original.new_items)))
    return (all(source[key] == value[key] for key in ('comparison', 'configuration', 'contract', 'seat'))
            and set(owner_shares(assigned, value['seat'])) <= covered)


def new_binding(config, metadata, snapshots, settings, capability, seat):
    configuration = configuration_identity(settings)
    return {'version': 1, 'kind': 'intent', 'repo': config.repo, 'pr': metadata.number,
            'head': metadata.head_sha, 'attempt': uuid.uuid4().hex, 'seat': seat,
            'contract': reviewer_contract(capability), 'capture': snapshots.digest,
            'configuration': configuration, 'comparison': comparison_identity(snapshots, configuration)}
