"""Extraction-bounded copy of the workflow's v2 dispatch validator.

Source repository: wwind123/coding-review-agent-loop
Source commit: 544b7659cda865d3f436e5b884b6d77d54a05e67
Source path: .github/workflows/ci.yml
Extraction boundary: ``validate_dispatch`` through its return value.
The contract tests invoke this copy with offline API callbacks and compare its
source block to the workflow so identity and input checks remain production-linked.
"""

import re


# BEGIN MANAGED_CI_V2_DISPATCH_VALIDATOR
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/ci.yml
# Extraction boundary: validate_dispatch() through its return value.
def validate_dispatch(
    *, protocol, pr_number_text, expected_head, nonce, repo, ref,
    configured_actor, initiating_actor, rerun_actor, api_json, api_pages,
    validate,
):
    if not (
        protocol == '2'
        and re.fullmatch(r'[1-9][0-9]*', pr_number_text or '')
        and re.fullmatch(r'[0-9a-f]{40}', expected_head or '')
        and re.fullmatch(r'[A-Za-z0-9_-]{32}', nonce or '')
    ):
        raise ValueError('managed dispatch inputs must be complete protocol-v2 values')
    if ref != 'refs/heads/main':
        raise ValueError('managed dispatch must execute the base workflow from main')
    trusted_actor = (configured_actor or '').strip()
    if not re.fullmatch(r'[A-Za-z0-9-]+', trusted_actor):
        raise ValueError('managed dispatch trusted actor configuration is invalid')
    live_user = api_json('users/' + trusted_actor)
    live_login = live_user.get('login') if isinstance(live_user, dict) else None
    live_id = live_user.get('id') if isinstance(live_user, dict) else None
    if (
        not isinstance(live_login, str)
        or type(live_id) is not int
        or not isinstance(initiating_actor, str)
        or not isinstance(rerun_actor, str)
        or trusted_actor.casefold() != live_login.casefold()
        or initiating_actor.casefold() != live_login.casefold()
        or rerun_actor.casefold() != live_login.casefold()
    ):
        raise ValueError('managed dispatch actors do not match the configured live identity')
    live_repo = api_json('repos/' + repo)
    if (
        not isinstance(live_repo, dict)
        or live_repo.get('full_name', '').casefold() != repo.casefold()
    ):
        raise ValueError('managed dispatch repository identity could not be validated')
    pr = api_json('repos/' + repo + '/pulls/' + pr_number_text)
    base_commit = api_json('repos/' + repo + '/commits/main')
    revision = base_commit.get('sha') if isinstance(base_commit, dict) else None
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('managed dispatch base workflow revision is unavailable')
    pages = api_pages('repos/' + repo + '/issues/' + pr_number_text + '/comments?per_page=100')
    record = validate(
        pr, pages, repo, pr_number_text, expected_head, nonce,
        live_login, revision, live_id,
    )
    return {
        'target_sha': expected_head,
        'pr_number': pr_number_text,
        'managed_nonce': nonce,
        'record': record,
    }
# END MANAGED_CI_V2_DISPATCH_VALIDATOR
