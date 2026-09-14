"""Extraction-bounded copy of the workflow's v2 dispatch validator.

Source repository: wwind123/coding-review-agent-loop
Source commit: 544b7659cda865d3f436e5b884b6d77d54a05e67
Source path: .github/workflows/ci.yml
Extraction boundary: ``validate_dispatch`` through its return value.
The contract tests invoke this copy with offline API callbacks and compare its
source block to the workflow so identity and input checks remain production-linked.
"""

import math
import re


# BEGIN MANAGED_CI_V2_DISPATCH_VALIDATOR
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/ci.yml
# Extraction boundary: validate_dispatch() through its return value.
# A dispatch intent must be no more than 15 minutes old and may be at most 5
# minutes ahead of the runner clock. The producer creates the record
# immediately before dispatch; bounding both directions prevents replay while
# tolerating normal clock skew.
MAX_INTENT_AGE_SECONDS = 15 * 60
MAX_INTENT_FUTURE_SKEW_SECONDS = 5 * 60
def validate_dispatch(
    *, protocol, pr_number_text, expected_head, nonce, repo, ref,
    configured_actor, initiating_actor, rerun_actor, current_run_id,
    current_run_attempt, current_time, api_json, api_pages, validate,
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
    if not re.fullmatch(r'[1-9][0-9]*', current_run_id or ''):
        raise ValueError('managed dispatch run ID is invalid')
    if not re.fullmatch(r'[1-9][0-9]*', current_run_attempt or ''):
        raise ValueError('managed dispatch run attempt is invalid')
    if (
        type(current_time) not in {int, float}
        or not math.isfinite(current_time)
        or current_time <= 0
    ):
        raise ValueError('managed dispatch current time is invalid')
    executing_run = (int(current_run_id), int(current_run_attempt))
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
    intent_age = current_time - record['created_at']
    if intent_age > MAX_INTENT_AGE_SECONDS:
        raise ValueError('managed intent record is stale')
    if intent_age < -MAX_INTENT_FUTURE_SKEW_SECONDS:
        raise ValueError('managed intent record is too far in the future')
    record_run = (record.get('run_id'), record.get('run_attempt'))
    if record['state'] == 'completed' and record.get('terminal_outcome') == 'no-status':
        terminal_key = (record.get('terminal_run_id'), record.get('terminal_run_attempt'))
        history = record.get('terminal_attempts')
        same_run_attempts = [
            item['run_attempt'] for item in history
            if item['run_id'] == record_run[0]
        ]
        # The driver records a no-status terminal attempt before a human or
        # operator reruns this same Actions run. Accept only the next attempt
        # of that same run; this rejects foreign, repeated, skipped, and
        # incoherent transitions.
        if (
            terminal_key != record_run
            or record_run[0] != executing_run[0]
            or not same_run_attempts
            or max(same_run_attempts) != record_run[1]
            or executing_run[1] != record_run[1] + 1
        ):
            raise ValueError('completed no-status retry run pair is incoherent')
    elif record['state'] in {'attached', 'completed'} and record_run != executing_run:
        raise ValueError('managed intent run pair does not match executing Actions run')
    return {
        'target_sha': expected_head,
        'pr_number': pr_number_text,
        'managed_nonce': nonce,
        'record': record,
    }
# END MANAGED_CI_V2_DISPATCH_VALIDATOR
