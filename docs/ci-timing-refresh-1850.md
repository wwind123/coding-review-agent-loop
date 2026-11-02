# Phase-one timing refresh verification

Refs #1551; post-merge verification and manual closure belong to the operator
in shard-trial. This report does not change the three-shard selection.

The replacement recording reported in PR #1852 produced the committed
`tests/.test_durations` at a004890fcc3c692c9ca6b451163d0275ea3aaaaf:
SHA-256 `ee1d52b868c217a92f052102a44bd667c10e87abf755bb99b42a2d4eb0957b64`,
17,327 selected IDs, including measured setup/teardown for selected skips.
The original run reported 17,314 passed and 13 skipped. The 14 opt-in
`image_cli_contract` cases require provider CLI prerequisites and are outside
the default collection.

The original recording intentionally mutated a tracked output. Its execution
receipt therefore cannot alone be cited as a stable-tree current-head pass.
An ordinary full-suite pass also does not prove that recording ran.
For independent current-input verification, run the same unsharded replacement
recorder with an output outside the checkout, preserving the committed store. Clear
inherited shard selectors, manifest/duration inputs and `PYTEST_ADDOPTS` in the
invoking shell first; keep the wrapper target as the recognized `env KEY=value`
launcher (its evidence parser does not recognize `env -u` options):

```bash
agent-loop run-tests --timeout-seconds 1800 -- env PYTHONPATH=ci/managed CI_SHARD_STORE_DURATIONS=/tmp/coding-review-agent-loop/scratch/1852-evidence/durations.json CI_SHARD_STORE_DURATIONS_MODE=replace /home/wwind123/tools/coding-review-agent-loop/.venv/bin/python -m pytest -p ci_shard_plugin -n 6 tests/ -q
```

Use the managed receipt from that invocation, not this procedure, as execution
evidence. Compare both stores with the default full collection; require finite
nonnegative values and report missing and stale IDs. Fresh measurements will
differ numerically, so do not compare their floating-point values for equality.
The external recording is an independent reproduction, not a claim that the
original receipt had stable attribution or that fresh values equal old values.

Source inspection: `pytest_configure` registers a session-local recorder;
`_owns_durations` gives the xdist controller sole ownership;
`_DurationRecorder` sums setup, call and teardown reports and atomically writes
the replacement output. With no shard selectors, collection is not partitioned.
The output destination is not a scheduling input, and the committed store is
not read when no duration input is selected. Nested sessions scrub recording
configuration. Existing parametrized recorder tests exercise plain and xdist
ownership, replacement of stale entries, and nested-session isolation.

Keep the subsequent ordinary final-state full-suite check separate from this
recording verification. Record actual results, receipt selectors, collection
counts and caveats in the PR follow-up after completion. Do not broaden the
evidence protocol or refresh the repository-wide store again merely to change
its receipt attribution.
