"""Extraction-bounded copy of this repository's production v2 validator.

Source repository: wwind123/coding-review-agent-loop
Source commit: 544b7659cda865d3f436e5b884b6d77d54a05e67
Source path: .github/workflows/ci.yml
Extraction boundary: the ``MANAGED_CI_V2_VALIDATOR`` block.
The contract test compares this fixture to the workflow block after YAML
indentation is removed.  The independent current and historical consumers
remain pinned in their own fixtures.
"""

import json
import re


# BEGIN MANAGED_CI_V2_VALIDATOR
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/ci.yml
# Extraction boundary: validate() through its closing assertion.
# This block is copied verbatim into tests/fixtures/managed_ci/current_router.py.
def validate(
    pr, pages, repo, num, sha, nonce, actor, revision, actor_id=None
):
    def fail(reason):
        raise ValueError(reason)

    def exact_int(value, name, positive=False):
        if type(value) is not int or (positive and value <= 0):
            fail('invalid ' + name)

    def exact_string(value, name, pattern=None):
        if type(value) is not str or (pattern and not re.fullmatch(pattern, value)):
            fail('invalid ' + name)

    if not isinstance(pr, dict):
        fail('live PR response is not an object')
    labels = {
        item.get('name') for item in pr.get('labels', [])
        if isinstance(item, dict)
    }
    if pr.get('state') != 'open' or pr.get('draft') is not True:
        fail('live PR is not an open draft')
    if pr.get('base', {}).get('ref') != 'main':
        fail('live PR base is not main')
    if pr.get('head', {}).get('sha') != sha:
        fail('live PR head drifted')
    if (pr.get('head', {}).get('repo') or {}).get('full_name') != repo:
        fail('live PR repository drifted')
    pr_user = pr.get('user', {})
    if pr_user.get('login') != actor or (actor_id is not None and type(pr_user.get('id')) is not int):
        fail('live PR author identity drifted')
    if actor_id is not None and pr_user.get('id') != actor_id:
        fail('live PR author ID drifted')
    if type(pr.get('number')) is not int or pr.get('number') != int(num):
        fail('live PR number drifted')
    if not pr.get('head', {}).get('ref', '').startswith('agent-loop/managed-'):
        fail('live PR branch is not reserved')
    if 'agent-loop-managed' not in labels:
        fail('live PR managed label is absent')
    if not isinstance(pages, list):
        fail('comments response is not paginated JSON')

    required = {
        'version', 'repository', 'pr', 'expected_head_sha', 'base_ref',
        'workflow_revision', 'generation', 'nonce', 'created_at', 'state',
        'run_id', 'run_attempt', 'terminal_run_id', 'terminal_run_attempt',
        'terminal_outcome', 'terminal_attempts',
    }
    allowed_states = {'prepared', 'dispatch-requested', 'attached', 'completed'}
    envelope = re.compile(r'^<!-- AGENT_MANAGED_CI_INTENT_V2 (?P<payload>.*?) -->$')
    sha_re = r'[0-9a-f]{40}'
    token_re = r'[A-Za-z0-9_-]{32}'
    validated = []
    for page in pages:
        if not isinstance(page, list):
            fail('malformed comments page')
        for comment in page:
            if not isinstance(comment, dict):
                fail('malformed comment object')
            user = comment.get('user')
            if not isinstance(user, dict) or type(user.get('login')) is not str:
                fail('malformed comment author')
            if user['login'] != actor:
                continue
            if actor_id is not None and user.get('id') != actor_id:
                fail('trusted comment author ID drifted')
            body = comment.get('body')
            if type(body) is not str:
                fail('trusted comment body is not text')
            match = envelope.fullmatch(body.strip())
            if match is None:
                continue
            try:
                record = json.loads(match.group('payload'))
            except json.JSONDecodeError:
                fail('trusted intent envelope contains malformed JSON')
            if not isinstance(record, dict):
                fail('trusted intent payload is not an object')
            if record.get('version') != 2:
                fail('unsupported intent version')
            if record.get('nonce') != nonce:
                continue
            if set(record) != required:
                fail('trusted intent has an invalid schema')
            exact_string(record.get('repository'), 'repository', r'[^/\s]+/[^/\s]+')
            exact_int(record.get('pr'), 'pr', True)
            exact_string(record.get('expected_head_sha'), 'expected_head_sha', sha_re)
            exact_string(record.get('base_ref'), 'base_ref', r'[^\s]+')
            exact_string(record.get('workflow_revision'), 'workflow_revision', sha_re)
            if actor_id is not None or record.get('generation') is not None:
                exact_string(record.get('generation'), 'generation', r'[A-Za-z0-9_-]+')
            exact_string(record.get('nonce'), 'nonce', token_re)
            exact_int(record.get('created_at'), 'created_at', True)
            exact_string(record.get('state'), 'state', r'(?:prepared|dispatch-requested|attached|completed)')
            if record['state'] not in allowed_states:
                fail('invalid lifecycle state')
            run_id, run_attempt = record.get('run_id'), record.get('run_attempt')
            if record['state'] in {'prepared', 'dispatch-requested'}:
                if run_id is not None or run_attempt is not None:
                    fail('early lifecycle state has run fields')
            elif type(run_id) is not int or run_id <= 0 or type(run_attempt) is not int or run_attempt <= 0:
                fail('attached lifecycle state lacks paired run fields')
            terminal_id = record.get('terminal_run_id')
            terminal_attempt = record.get('terminal_run_attempt')
            if terminal_id is not None and (type(terminal_id) is not int or terminal_id <= 0):
                fail('invalid terminal run ID')
            if terminal_id is None and terminal_attempt is not None:
                fail('terminal attempt has no terminal run ID')
            if terminal_attempt is not None and (type(terminal_attempt) is not int or terminal_attempt <= 0):
                fail('invalid terminal run attempt')
            if record.get('terminal_outcome') not in {None, 'no-status'}:
                fail('invalid terminal outcome')
            attempts = record.get('terminal_attempts')
            if not isinstance(attempts, list):
                fail('terminal attempt history is not a list')
            seen_attempts = set()
            for item in attempts:
                if not isinstance(item, dict) or set(item) != {'run_id', 'run_attempt'}:
                    fail('malformed terminal attempt history')
                item_id, item_attempt = item['run_id'], item['run_attempt']
                if type(item_id) is not int or item_id <= 0:
                    fail('invalid terminal history run ID')
                if item_attempt is not None and (type(item_attempt) is not int or item_attempt <= 0):
                    fail('invalid terminal history run attempt')
                key = (item_id, item_attempt)
                if key in seen_attempts:
                    fail('duplicate terminal attempt history')
                seen_attempts.add(key)
            expected = {
                'repository': repo,
                'pr': int(num),
                'expected_head_sha': sha,
                'base_ref': 'main',
                'workflow_revision': revision,
                'nonce': nonce,
            }
            if any(record[key] != value for key, value in expected.items()):
                fail('trusted intent binding drifted')
            validated.append(record)
    if len(validated) != 1:
        fail('expected exactly one fresh intent for requested nonce')
    if validated[0]['state'] == 'prepared':
        fail('prepared intent is not a dispatch authorization')
    return validated[0]
# END MANAGED_CI_V2_VALIDATOR
