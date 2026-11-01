# coding-review-agent-loop — Claude Code Instructions

## Running the CLI

Use the installed `agent-loop` CLI for issue, task, plan-first, and PR runs.
Gather the repository, issue or PR number, reviewers, and flow type, then
consult `agent-loop --help` and `docs/local_agent_loop.md` for options and recovery.

## Development

The main package lives in `src/coding_review_agent_loop/`.  Tests are in
`tests/`.  Run the full test suite with:

```bash
.venv/bin/python -m pytest tests/ -q
```

Do not refresh `tests/.test_durations` in a feature or fix PR; full refreshes
belong in a dedicated maintenance PR, and if the coverage gate in
`tests/test_ci_sharding.py` fails, add only the missing entries with
`PYTHONPATH=ci/managed${PYTHONPATH:+:$PYTHONPATH} CI_SHARD_STORE_DURATIONS=tests/.test_durations CI_SHARD_STORE_DURATIONS_MODE=add-missing python -m pytest -p ci_shard_plugin -n auto`.
