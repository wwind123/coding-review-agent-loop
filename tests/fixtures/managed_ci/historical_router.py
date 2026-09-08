"""Verbatim lifecycle validator extracted from wwind123/llm-dialectic.

Source repository: wwind123/llm-dialectic
Source commit: 91c606e4a46a4c466d7ad86400b2d3c9f897d630
Source path: .github/workflows/ci.yml
Extraction boundary: route job's embedded Python validator, lines 252-315
of the fetched workflow.  This historical consumer validates every trusted
intent envelope before filtering by the requested nonce.
"""

import json
import re


def validate(
    pr: object,
    pages: object,
    repo: str,
    num: str,
    sha: str,
    nonce: str,
    actor: str,
    revision: str,
) -> None:
    """Run the embedded historical validator without changing its decisions."""
    def fail(reason):
        raise ValueError(reason)

    def exact_int(value, name, positive=False):
        if type(value) is not int or (positive and value <= 0): fail('invalid ' + name)

    def exact_string(value, name, pattern=None):
        if type(value) is not str or (pattern and not re.fullmatch(pattern, value)): fail('invalid ' + name)

    if not isinstance(pr, dict): fail('live PR response is not an object')
    labels={x.get('name') for x in pr.get('labels',[]) if isinstance(x,dict)}
    if pr.get('state') != 'open' or pr.get('draft') is not True: fail('live PR is not an open draft')
    if pr.get('base',{}).get('ref') != 'main' or pr.get('head',{}).get('sha') != sha: fail('live PR head or base drifted')
    if (pr.get('head',{}).get('repo') or {}).get('full_name') != repo: fail('live PR repository drifted')
    if pr.get('user',{}).get('login') != actor or type(pr.get('number')) is not int or pr.get('number') != int(num): fail('live PR actor or number drifted')
    if not pr.get('head',{}).get('ref','').startswith('agent-loop/managed-') or 'agent-loop-managed' not in labels: fail('live PR managed tuple is invalid')
    if not isinstance(pages,list): fail('comments response is not paginated JSON')
    required={'version','repository','pr','expected_head_sha','base_ref','workflow_revision','nonce','created_at','state','run_id','run_attempt'}
    envelope=re.compile(r'^<!-- AGENT_MANAGED_CI_INTENT_V2 (?P<payload>.*?) -->$')
    sha_re=r'[0-9a-f]{40}'; token_re=r'[A-Za-z0-9_-]{32}'
    validated=[]
    for page in pages:
        if not isinstance(page,list): fail('malformed comments page')
        for comment in page:
            if not isinstance(comment,dict): fail('malformed comment object')
            user=comment.get('user')
            if not isinstance(user,dict) or type(user.get('login')) is not str: fail('malformed comment author')
            # An untrusted author is never allowed to influence parsing;
            # in particular, malformed marker-like prose is harmless.
            if user['login'] != actor: continue
            body=comment.get('body')
            if type(body) is not str: fail('trusted comment body is not text')
            body=body.strip()
            match=envelope.fullmatch(body)
            if match is None: continue
            try: record=json.loads(match.group('payload'))
            except json.JSONDecodeError: fail('trusted intent envelope contains malformed JSON')
            if not isinstance(record,dict): fail('trusted intent payload is not an object')
            if not required.issubset(record): fail('trusted intent is missing required fields')
            exact_int(record.get('version'),'version')
            if record['version'] != 2: fail('unsupported intent version')
            exact_string(record.get('repository'),'repository',r'[^/\s]+/[^/\s]+')
            exact_int(record.get('pr'),'pr',True)
            exact_string(record.get('expected_head_sha'),'expected_head_sha',sha_re)
            exact_string(record.get('base_ref'),'base_ref',r'[^\s]+')
            exact_string(record.get('workflow_revision'),'workflow_revision',sha_re)
            exact_string(record.get('nonce'),'nonce',token_re)
            exact_int(record.get('created_at'),'created_at',True)
            exact_string(record.get('state'),'state',r'(?:prepared|dispatch-requested|attached|completed)')
            run_id, run_attempt=record.get('run_id'), record.get('run_attempt')
            if record['state'] in ('prepared','dispatch-requested'):
                if run_id is not None or run_attempt is not None: fail('early lifecycle state has run fields')
            elif type(run_id) is not int or run_id <= 0 or type(run_attempt) is not int or run_attempt <= 0:
                fail('attached lifecycle state lacks paired positive run fields')
            if 'generation' in record:
                generation=record['generation']
                if generation is not None and (type(generation) is not str or not re.fullmatch(r'[A-Za-z0-9_-]+', generation)):
                    fail('malformed known generation field')
            if record['nonce'] != nonce:
                continue
            expected={'repository':repo,'pr':int(num),'expected_head_sha':sha,'base_ref':'main','workflow_revision':revision,'nonce':nonce}
            if any(record[key] != value for key,value in expected.items()): fail('trusted intent binding drifted')
            validated.append((body,record))
    distinct={body: record for body,record in validated}
    qualifying={body: record for body,record in distinct.items() if record['state'] != 'prepared'}
    if len(qualifying) != 1: fail('expected exactly one distinct qualifying intent for requested nonce')
