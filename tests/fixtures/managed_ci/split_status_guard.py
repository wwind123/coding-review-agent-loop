"""Extraction-bounded copy of the split publish entry point's correlation guard.

Source repository: wwind123/coding-review-agent-loop
Source path: .github/workflows/managed-ci-publish.yml
Extraction boundary: ``correlation_matches`` through its return value.
The contract tests compare this fixture to the production workflow and drive
the guard together with the extracted status builder without a GitHub request.
"""


# BEGIN MANAGED_CI_SPLIT_STATUS_GUARD
# Source repository: wwind123/coding-review-agent-loop
# Source path: .github/workflows/managed-ci-publish.yml
# Extraction boundary: correlation_matches() through its return value.
# The status job trusts nothing it was handed: every forwarded input and the
# supplied target and nonce must equal the values the dispatch event carried,
# and validation must have succeeded. Any mismatch means no status is written.
FORWARDED_KEYS = ('protocol_version', 'pr_number', 'expected_head_sha', 'managed_nonce')
def correlation_matches(*, forwarded, event, target_sha, nonce, validation_result):
    if validation_result != 'success':
        return False
    if not isinstance(forwarded, dict) or not isinstance(event, dict):
        return False
    for key in FORWARDED_KEYS:
        supplied = forwarded.get(key)
        dispatched = event.get(key)
        if not isinstance(supplied, str) or not isinstance(dispatched, str):
            return False
        if supplied != dispatched:
            return False
    return (
        isinstance(target_sha, str)
        and isinstance(nonce, str)
        and target_sha == event['expected_head_sha']
        and nonce == event['managed_nonce']
    )
# END MANAGED_CI_SPLIT_STATUS_GUARD
