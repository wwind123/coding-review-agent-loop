Read `SKILL.md` in full, then run the coding-review-agent-loop orchestration.

Arguments provided by the user: $ARGUMENTS

Parse the arguments to extract:
- **repo** (`OWNER/REPO`) — required
- **flow** — `issue <N>` (maps to `--flow plan`) or `pr <N>` (maps to `--flow pr`)
- **reviewers** — `--reviewers codex`, `gemini`, or both (default: gemini)
- **plan-first** — present if the user passes `--plan-first` (only relevant for issue flow)
- **coder** — `--coder claude` (default), `--coder codex`, `--coder gemini`, or
  `--coder antigravity`

If any required argument is missing, ask the user before proceeding.

Then follow the orchestration steps in `SKILL.md` from Step 1.

When an external `--coder` is parsed, follow the **Reversed roles** section of
`SKILL.md` instead of the default Claude-as-coder flow. If an external-coder PR
is blocked by `run-pr-round`, use `run-pr-fix --pr N --coder X --reviewers ...`
with the same reviewer set and a push-capable PR-branch workdir, then re-run
`run-pr-round` on the new head.

Note: `task "<text>"` is not supported in skill mode. Direct the user to the
headless CLI (`agent-loop task "..." --repo OWNER/REPO`) for task-based flows.
