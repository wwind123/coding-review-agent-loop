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
def correlation_matches(
    *, forwarded, event, target_sha, nonce, validation_result,
    validation_run_id, validation_attempt, run_id, run_attempt,
):
    if validation_result != 'success':
        return False
    # Validation must have run in this very run attempt: a validate job carried
    # over from an earlier attempt (for example by "Re-run failed jobs") never
    # re-checked the completed audit record, freshness or rerun actor.
    if not (
        isinstance(validation_run_id, str) and isinstance(validation_attempt, str)
        and isinstance(run_id, str) and isinstance(run_attempt, str)
        and run_id != '' and run_attempt != ''
        and validation_run_id == run_id and validation_attempt == run_attempt
    ):
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
