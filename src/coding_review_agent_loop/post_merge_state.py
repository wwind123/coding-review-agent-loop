"""Optional round metadata extension. Malformed scheduling never grants timing authority."""
from .post_merge_obligations import fail, parse_requests, validate_registry

FIELDS = ('post_merge_registry', 'post_merge_diagnostics', 'post_merge_publication', 'post_merge_requests')


def encode_fields(metadata):
    return {key: list(getattr(metadata, key)) if isinstance(getattr(metadata, key), tuple)
            else getattr(metadata, key) for key in FIELDS if getattr(metadata, key)}


def decode_fields(payload):
    result = {}
    if 'post_merge_registry' in payload:
        result['post_merge_registry'] = validate_registry(payload['post_merge_registry'])
    if 'post_merge_diagnostics' in payload:
        diagnostics = payload['post_merge_diagnostics']
        if not isinstance(diagnostics, list) or not all(isinstance(v, str) for v in diagnostics):
            fail('invalid admission diagnostics')
        result['post_merge_diagnostics'] = tuple(diagnostics)
    if 'post_merge_publication' in payload:
        value = payload['post_merge_publication']
        if not isinstance(value, dict) or set(value) != {'version', 'identity', 'repo', 'pr',
                'request', 'source', 'provenance', 'merged_source_head', 'merge_commit'}:
            fail('malformed publication record')
        result['post_merge_publication'] = value
    if 'post_merge_requests' in payload:
        result['post_merge_requests'] = tuple(r.payload() for r in parse_requests(payload['post_merge_requests']))
    return result
