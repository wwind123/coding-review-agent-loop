# coding-review-agent-loop

`coding-review-agent-loop` is a local command-line orchestrator for GitHub code
review. One coding agent creates or updates a pull request, one or more other
agents review it, and the loop sends blocking feedback back to the coder until
the reviewers approve or the run reaches a clear stopping condition.

```text
GitHub issue, task, or PR
          |
          v
      coding agent  <-------+
          |                 |
          v                 |
     pull request           |
          |                 |
          v                 |
      reviewers ---- feedback
          |
          v
       approved  ->  optional CI wait and merge
```

The loop runs on your machine and uses the local `claude`, `codex`, `agy`,
`gemini`, and `gh` programs you have already authenticated. It does not require
you to put model API keys into this project. You only need the agent CLIs used
for the roles you select; you do not need to install every supported backend.

The project is alpha software. It can let coding agents edit repositories, run
commands, push branches, and write to GitHub. Start with a repository where you
can inspect and revert the results.

## Why Use It?

- Replace the manual cycle of copying reviewer feedback between agent sessions.
- Assign coding and review to different models or providers.
- Start from a GitHub issue, an existing pull request, or a free-form task.
- Review an implementation plan before allowing code changes.
- Require multiple independent reviewers to approve the same PR head.
- Resume interrupted work from durable metadata recorded on GitHub.
- Optionally run local tests, wait for CI, and merge after approval.

The default workflow is deliberately conservative: Claude is the coder, Codex
is the reviewer, the review limit is 10 rounds, and automatic merge is off.

## Requirements

- Python 3.11 or newer.
- Git and [GitHub CLI](https://cli.github.com/) with `gh auth status` succeeding.
- Repository access sufficient for the requested issue, branch, PR, and comment
  operations.
- A local CLI for the coder and each reviewer you select:
  [Claude Code](https://docs.anthropic.com/en/docs/claude-code),
  [OpenAI Codex CLI](https://github.com/openai/codex),
  Antigravity CLI (`agy`), or the legacy Gemini CLI backend.

Each agent CLI has its own authentication, quota, terms, and model
availability. Confirm those with the provider; this tool does not combine or
replace provider subscriptions.

## Install

Clone the repository and install it into a virtual environment:

```bash
gh repo clone wwind123/coding-review-agent-loop
cd coding-review-agent-loop
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
agent-loop --help
```

Check the CLIs for your chosen roles before the first run:

```bash
gh auth status
claude --version
codex --version
```

Substitute `agy --version` or `gemini --version` when using those backends.

## Quick Start

### Review an existing PR

This is the smallest useful first run. The reviewers inspect the current PR;
if they find blockers, the coder updates that same PR and review continues.

```bash
agent-loop pr 456 \
  --repo OWNER/REPO \
  --coder claude \
  --reviewer codex
```

### Implement a GitHub issue

Issue mode gives the issue title, body, and comments to the coder, validates the
resulting PR, and then enters the same review loop.

```bash
agent-loop issue 123 \
  --repo OWNER/REPO \
  --coder claude \
  --reviewer codex
```

Without `--plan-first`, issue mode asks the coder to implement immediately.

### Review a plan, then implement it

Use plan-first mode for work whose design should be challenged before files are
changed. `--plan-first` alone stops after plan approval. Add
`--implement-after-approval` to continue into implementation and PR review.

```bash
agent-loop issue 123 \
  --repo OWNER/REPO \
  --coder codex \
  --reviewer claude \
  --plan-first \
  --implement-after-approval
```

### Implement a task without an issue

```bash
agent-loop task "Add a health-check endpoint" \
  --repo OWNER/REPO \
  --coder codex \
  --reviewer claude
```

## Choose a Workflow

| Command | Use it when |
| --- | --- |
| `agent-loop issue` | A GitHub issue defines the work to implement or plan. |
| `agent-loop pr` | The implementation PR already exists. |
| `agent-loop task` | You have a direct task and do not need an issue first. |
| `agent-loop discuss` | You want agents to evaluate an open question without writing code. |
| `agent-loop managed-pr` | Code is pushed but no PR exists, and the repository uses managed exact-head CI. |
| `agent-loop managed-ci preflight` | You want a read-only readiness report for managed CI. |

Run `agent-loop <command> --help` for the complete options for one workflow.
The [full CLI guide](docs/local_agent_loop.md#usage) covers lifecycle and resume
behavior in detail.

## How the Review Loop Behaves

1. The coder implements the issue or updates the existing PR.
2. Every configured reviewer reviews the same PR head.
3. Blocking findings return to the coder as an explicit work ledger.
4. The updated head is reviewed again.
5. The run stops on unanimous approval, a terminal blocker, a clarification
   request, unavailable required input, or the round limit.

Repeat `--reviewer` to require multiple approvals. Use `--review-parallel` when
the reviewers have distinct workdirs and may run concurrently:

```bash
agent-loop pr 456 \
  --repo OWNER/REPO \
  --coder codex \
  --reviewer claude \
  --reviewer agy \
  --review-parallel
```

Agent-loop creates separate repo-scoped temporary checkouts for active agents
unless you provide workdirs. GitHub comments carry durable round and handoff
metadata, so a later run can reconstruct the active review state. When the PR
number is known, resume with `agent-loop pr <number>` instead of starting issue
implementation again.

Signed comments ending in `-- Human Reviewer` are treated as explicit human
requirements and remain approval-critical. See
[Human requirements](docs/local_agent_loop.md#human-requirements) for the exact
contract.

## Current Limitations

- Run only one active `agent-loop` invocation per repository per machine. The
  default workdirs and repo-scoped local state are shared, and the tool does not
  currently enforce a repository-wide process lock. Separate concurrent runs
  against the same repository can interfere with each other's checkouts and
  artifacts. `--review-parallel` is supported within one orchestrator run; it
  does not make multiple same-repository invocations safe.
- Agent-loop is a local process, not a hosted service. The machine must remain
  available for the run, and an interrupted in-flight agent turn may need to be
  repeated. Once a PR exists, resume with `agent-loop pr <number>`.
- GitHub is the only supported forge, and every selected agent backend must be
  installed and authenticated locally.
- Agent CLIs can exhaust quota, time out, update themselves, or return malformed
  structured output. Retries, repair passes, and salvage reduce lost work but
  cannot guarantee unattended completion.
- Default agent checkouts live under the system temporary directory (`/tmp` on
  Linux) and may disappear after a reboot or system cleanup. Configure explicit
  workdirs for long-lived installations.

## Planning and Decomposition

Plan-first mode supports four post-approval choices:

| Mode | Result after plan approval |
| --- | --- |
| `plan-only` | Post the approved plan and stop. This is the default. |
| `implement-one-shot` | Implement the approved plan in one PR. |
| `decompose-only` | Create detailed child issues for the approved phases and stop. |
| `implement-by-phase` | Create the phase issues and implement only the first phase. |

Example:

```bash
agent-loop issue 123 --repo OWNER/REPO \
  --plan-first \
  --plan-execution-mode decompose-only
```

`--plan-execution-mode decompose-only` and `--materialize-split-issues` are
different mechanisms. Do not combine them for the same decomposition: doing so
can create duplicate children. Use the former for detailed approved phases and
the latter for discuss-mode split proposals or eligible plan-only deferred
work. Read
[Phased decomposition versus split materialization](docs/local_agent_loop.md#phased-decomposition-versus-split-materialization)
before filing child issues.

## Discuss Mode

Discuss mode asks agents to evaluate an issue without modifying the repository.
Use it for architecture choices, product decisions, feasibility questions, or
whether work should be implemented or split.

```bash
agent-loop discuss 123 \
  --repo OWNER/REPO \
  --reviewer claude \
  --reviewer codex
```

The default result contract is implementation triage: `implement`,
`do-not-implement`, `needs-human`, or `split`. For an open-ended recommendation
instead of an implementation vote, use `--discuss-result-mode answer`.

Useful optional controls include:

- `--discuss-analyzer AGENT` for a structured consensus/disagreement agenda.
- `--discuss-research auto|required|none` for current external facts.
- `--discuss-parallel` for concurrent independent positions.
- `--materialize-split-issues` to file agreed split proposals.

See [Discuss mode](docs/local_agent_loop.md#open-ended-answer-results) for result
semantics, research evidence, deadlocks, and resume behavior.

Answer-mode summaries now put a bounded executive state before the audit
transcript. With a configured `--discuss-analyzer`, completed non-final rounds
reuse the analyzer's enriched agenda when available and show cumulative current
consensus, active disagreements, changes, missing facts, and next-round focus.
The final comment similarly leads with outcome, agreed conclusions, residual
decisions, and the next action. Exact, semantic-equivalent, and debater-confirmed
results reuse their mechanically validated artifacts without a redundant final
synthesis call; only the configured discuss analyzer may perform the explicitly
bounded fallback or final synthesis call. Analyzer-less or invalid synthesis
falls back to the existing fail-closed result. Per-agent comments remain the
authoritative raw audit, and resume metadata carries only the latest validated
snapshot with bounded, spillable excerpts.

## Agent Backends

| Backend | CLI | Notes |
| --- | --- | --- |
| Claude | `claude` | Default coder. Select a model with `--claude-model`. |
| Codex | `codex` | Default reviewer. Select a model with `--codex-model`. |
| Antigravity | `agy` | Accepted as `agy` or `antigravity`; supports coder and reviewer roles. |
| Gemini | `gemini` | Legacy, best-effort path for accounts that still have CLI access. |

The default Antigravity model chain is `Gemini 3.7 Flash (High)`, then
`Gemini 3.6 Flash (High)`, then `Gemini 3.1 Pro (High)` for eligible capacity
failures. Override it with `--antigravity-model` or
`--antigravity-models`. Antigravity turns are single-shot and its usage totals
are estimated because `agy` does not expose token counts.

Backend-specific authentication, model selection, fallback, timeout, and
executable-replacement behavior are documented under
[Agent backends](docs/local_agent_loop.md#agent-backends).

## Safety and Permissions

Agents can run commands and change code. Keep their normal permission prompts
unless you understand and accept the repository and machine-level risk.

For a trusted local environment, this flag supplies each backend's permission
bypass option:

```bash
agent-loop pr 456 --repo OWNER/REPO \
  --coder codex --reviewer claude \
  --dangerous-agent-permissions
```

The flag is intentionally explicit. It does not make agent output, fetched
issue text, dependencies, shell commands, or generated code trustworthy.

Other important boundaries:

- Automatic merge is off unless `--auto-merge` is present.
- Reviewer approval is not a substitute for project tests or human judgment.
- `--test-command` adds a local gate before review and again before auto-merge. The
  finite watchdog defaults to 1,800 seconds and is configurable with
  `--coder-test-command-timeout-seconds SECONDS`.
- The tool validates assigned workdirs and reported test locations, but agent
  CLIs may still consume substantial CPU, memory, network, and provider quota.
- Raw subprocess logs and salvage artifacts can contain sensitive repository
  context. Protect access to the configured log directories and review their
  retention settings.

Read [Workdirs](docs/local_agent_loop.md#workdirs),
[Agent permission flags](docs/local_agent_loop.md#agent-permission-flags), and
[Logs](docs/local_agent_loop.md#logs) before unattended use.

### Runtime-aware local test timeouts

Coder test commands may be run through the backend-neutral wrapper:

```bash
agent-loop run-tests --memory-dir /path/to/memory -- pytest tests/test_app.py -q
```

The `--timeout-seconds` value is the watchdog for that one whole command. When
omitted, the wrapper uses the inherited run ceiling, or 1,800 seconds when run
outside agent-loop. A positive finite override may be smaller than the ceiling;
values above it are rejected before the child starts. Agent backends inherit the
ceiling through `AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS`.

When agent memory is enabled, the wrapper records measured outcomes, elapsed
time, the attempted cap, a privacy-preserving environment fingerprint, and
cheap lockfile/configuration hashes in `test-runtime.json`. Recent successful
runs produce advisory median/p95 recommendations with headroom; timeouts remain
lower-bound evidence and are never treated as successful durations. Data is
best-effort, retained to 20 samples per command/fingerprint cohort and 200
cohorts, and becomes stale after 30 days or when relevant inputs change.
Remembered commands are suggestions only: agents must inspect the checkout and
select focused tests. Framework per-test limits, the wrapper whole-command
watchdog, and the backend whole-turn timeout are separate. The backend turn
must leave headroom for analysis, edits, and reporting; split or shard healthy
browser/integration matrices when that improves diagnosis and retry cost.

## CI and Merge

Use `--auto-merge` only when the repository's CI and branch protections are
appropriate for unattended merging:

```bash
agent-loop pr 456 --repo OWNER/REPO --auto-merge
```

For ordinary CI, auto-merge waits for a reliable, non-empty check board on the
current head. Without auto-merge, `--watch-pending-ci` can wait and report that
an approved PR is merge-ready without merging. Set the total watcher budget
with `--ci-timeout-seconds` (default 1200) and its polling interval with
`--ci-poll-interval-seconds` (default 30). GitHub runner stalls are bounded by
`--ci-queued-grace-seconds`; see
[External CI infrastructure stalls](docs/local_agent_loop.md#external-ci-infrastructure-stalls).

### Managed exact-head CI

Managed CI is an advanced, repository-integrated workflow that suppresses
expensive intermediate CI and qualifies one reviewed SHA at the end. Do not
enable it from a README example alone. First read
[Managed exact-head CI](docs/local_agent_loop.md#managed-exact-head-ci) and run
the read-only preflight:

```bash
agent-loop managed-ci preflight \
  --repo OWNER/REPO \
  --base main \
  --trusted-actor LOGIN
```

For code already pushed without an open PR, the pre-creation form begins with
`agent-loop managed-pr --head BRANCH`. Managed issue and PR recovery relies on
a canonical issue handoff, explicit `--managed-ci` intent, documented
draft/labeled and ready/unlabeled lifecycle states, immutable actor evidence,
and preserved base provenance. Historical records are audit evidence only and
never grant fresh authority.

Qualification and merge remain bound to the live head. The final merge uses
`--match-head-commit` and merges only that qualified SHA. `--watch-pending-ci`
and `--no-watch-pending-ci` do not alter managed exact-head qualification.

## Claude Code Skill Mode

The repository also contains a Claude Code skill for running the orchestration
inside an attended Claude Code session. In skill mode, Claude acts in the
current interactive session while external agents still run through their
local CLIs.

Use the standalone CLI for predictable or unattended runs. Use skill mode when
you want conversational setup, active steering, and interactive recovery.
Skill mode never auto-merges.

See [`SKILL.md`](SKILL.md) for invocation instructions and
[`docs/skill_mode.md`](docs/skill_mode.md) for its design and limitations.

## Documentation

- [`agent-loop --help`](docs/local_agent_loop.md#usage): full command and option reference.
- [`docs/local_agent_loop.md`](docs/local_agent_loop.md): architecture, lifecycle, protocol, recovery, CI, memory, and safety reference.
- [`docs/skill_mode.md`](docs/skill_mode.md): Claude Code skill architecture and operation.
- [`SKILL.md`](SKILL.md): executable instructions for Claude Code skill mode.

The detailed guide is intentionally the source for protocol schemas, durable
markers, repair passes, fallback ladders, CI provenance, and compatibility
behavior. Those internals are not required for a first successful run.

## Development

Install the development dependency and run the tests:

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

Use focused tests while changing one subsystem, for example:

```bash
python -m pytest tests/test_docs_guidance.py
python -m pytest tests/test_protocol.py
python -m pytest tests/test_orchestrator_pr.py
```

Tests use fake subprocess runners and do not invoke real agent CLIs or GitHub.
Browse the focused test modules in [`tests/`](tests/) and see the architecture
diagram in [`docs/local_agent_loop.md`](docs/local_agent_loop.md#architecture).

## Related Tools

This project is a standalone local GitHub lifecycle orchestrator. Projects such
as [claude-review-loop](https://github.com/hamelsmu/claude-review-loop),
[codex-review](https://github.com/boyand/codex-review), and
[codex-plugin-cc](https://github.com/openai/codex-plugin-cc) integrate review or
delegation into a particular agent host. Here, the orchestrator stays outside
the agent hosts and can reverse coder/reviewer roles.

## License

[MIT](LICENSE)
