"""Extraction-bounded copy of the workflow's terminal publisher decision.

Source repository: wwind123/coding-review-agent-loop
Source commit: 278699233efee2aebe5346e89e7daebaa0bb8cd6
Source path: .github/workflows/ci.yml
Extraction boundary: ``build_status_payload`` through its return value.
The contract tests compare this fixture to the production workflow and invoke
the extracted decision without making a GitHub request.
"""

import re


# BEGIN MANAGED_CI_V2_PUBLISHER
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/ci.yml
# Extraction boundary: build_status_payload() through its return value.
def build_status_payload(
    *, target_sha, validation_result, test_result, nonce, run_id,
    run_attempt, server_url, repository,
):
    if (
        validation_result != 'success'
        or not isinstance(target_sha, str)
        or not re.fullmatch(r'[0-9a-f]{40}', target_sha)
        or not re.fullmatch(r'[A-Za-z0-9_-]{32}', nonce or '')
        or not re.fullmatch(r'[1-9][0-9]*', run_id or '')
        or not re.fullmatch(r'[1-9][0-9]*', run_attempt or '')
        or not isinstance(server_url, str)
        or not server_url
        or not isinstance(repository, str)
        or not repository
    ):
        return None
    state = 'success' if test_result == 'success' else 'failure'
    description = (
        'nonce=' + nonce + ';run_id=' + run_id +
        ';attempt=' + run_attempt + ';result=' + state
    )
    return {
        'target_sha': target_sha,
        'payload': {
            'state': state,
            'context': 'final-ci/exact-head',
            'description': description,
            'target_url': server_url.rstrip('/') + '/' + repository +
                          '/actions/runs/' + run_id,
        },
    }
# END MANAGED_CI_V2_PUBLISHER
