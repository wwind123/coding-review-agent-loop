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

### Risk-based mode and transition matrices

Fresh stateful or multi-mode plans may carry a bounded generation-1 risk matrix.
Each applicable row has a matrix-local ID, entry path or mode, initial state,
event, expected outcome, forbidden side effects, proposed test location, and one
execution owner. Simple local work can instead use a non-empty, proportionate
not-applicable rationale; important exclusions remain explicit. Rows are
planning proposals until implementation evidence maps them to actual tests.

The structured matrix payload is the authority, not a copied markdown table.
Round metadata and the existing bounded sidecar persist its canonical identity,
so a renderer change does not invalidate an otherwise authentic plan. A corrupt
or unfittable matrix closes only the matrix channel and reports a diagnostic;
it does not withdraw a separately valid plan hash and subject. Historical plans
without a matrix remain resumable without invented rows. The same contract is
available to skill helpers: use `helpers.validate_response` or
`helpers.render_response` with `--require-risk-test-matrix-contract` when a fresh
generation-1 matrix is required.

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

### Managed CI in this repository

This repository's `.github/workflows/ci.yml` installs the managed-CI v2
contract with the literal `AGENT_LOOP_MANAGED_CI_V2` and
`AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1` declarations. Pull requests use
exactly four activities: `opened`, `synchronize`, `reopened`, and `unlabeled`.
Only a trusted, same-repository draft on the reserved
`agent-loop/managed-*` branch can suppress intermediate CI: opening is
recognized before the label write, while later synchronization and reopening
require `agent-loop-managed`. Unlabeled events always run the ordinary Python
3.12 suite, and forks, malformed payloads, trust mismatches, and all ordinary
PRs fail open to that suite. Pushes to `main` and all-empty manual dispatches
also run the complete suite.

After the workflow change is merged, the sole-maintainer rollout is ordered:

1. Set the repository Actions variable to the trusted actor:
   `gh variable set AGENT_LOOP_MANAGED_ACTOR --repo wwind123/coding-review-agent-loop --body wwind123`.
2. Authenticate `gh` as `wwind123`, then run the read-only preflight:
   `agent-loop managed-ci preflight --repo wwind123/coding-review-agent-loop --base main --trusted-actor wwind123`.
3. While `main` remains unprotected, pass
   `--managed-ci-trusted-actor wwind123 --allow-unprotected-managed-ci` on
   every applicable issue-created invocation. The waiver is explicit per run;
   it does not change defaults for other repositories, enable arbitrary
   existing-PR adoption, or replace a head-guarded merge.
   For example:

   ```bash
   agent-loop issue <issue-number> --repo wwind123/coding-review-agent-loop \
     --plan-first --implement-after-approval --auto-merge --managed-ci \
     --managed-ci-trusted-actor wwind123 --allow-unprotected-managed-ci
   ```

For an approved head, the base-branch workflow validates the live actor,
repository, PR tuple, workflow revision, and one fresh generation-scoped
handoff record before exposing `expected_head_sha`. Freshness is measured on
the base runner: `created_at` must be at most 15 minutes old and no more than
5 minutes ahead of the runner clock. A no-status retry may advance only from
the recorded terminal attempt to the immediate next attempt of that same run.
A separate job checks out
that exact SHA, installs the editable development dependencies, and runs the
full `python -m pytest` suite once. The always-evaluated publisher writes
`final-ci/exact-head` only for that validated target, with the nonce, run ID,
attempt, and Actions URL correlated in the status. Authorization failures
before target validation write no status; checkout or test failures publish a
terminal non-success result. Keep queued work on ordinary CI until this
post-merge live qualification is complete.

### Process-tree containment

Agent subprocesses and repository test gates use a shared per-user containment
policy by default. On Linux with systemd 253+ and delegated cgroup v2,
agent-loop creates a foreground child scope inside `agent-loop.slice` and
applies aggregate plus role-specific `MemoryHigh`, `MemoryMax`,
`MemorySwapMax`, and `TasksMax` limits. The aggregate protects host headroom
across independent agent-loop processes; a child scope cannot exceed it. The
portable fallback uses process-group TERM/KILL for deterministic termination
but provides no memory ceiling. `--containment-mode required` fails closed
when cgroup preflight is unavailable, while `auto` reports the fallback.

Inspect the resolved policy and capabilities with
`agent-loop containment-preflight --containment-mode auto`. The systemd
launcher is the synchronous foreground form
`systemd-run --user --scope --quiet`; it deliberately does not use `--wait`,
`--service`, or `--pipe`. A target-start shim report is authoritative when
distinguishing launcher failures from target exit statuses. Optional cgroup
telemetry (peak memory, PSI, and swap counters) is capability-aware: missing
files are reported as not collected rather than treated as lost evidence.
OOM, hard memory/swap failures, and task-limit failures are
`resource-exhausted`; `MemoryHigh`/PSI pressure is diagnostic only.

The same-command test wrapper has a per-invocation lane lock. A duplicate is
rejected before spawn until the earlier managed attempt exits or is explicitly
terminated. Standalone commands not run through the wrapper cannot be
deduplicated from free-form agent logs, but their complete agent tree remains
resource-bounded when launched by agent-loop. The aggregate is per user
manager, not cross-user host isolation. A skill host's in-session Claude turn
and arbitrary descendants remain part of that host session; use the managed
wrapper for test gates and external skill agents for the mechanical boundary.

In skill mode, the test-gate policy is selected with
`AGENT_LOOP_CONTAINMENT_MODE=auto|required|off` (default `auto`).

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

Plan-first implementation carries the approved plan through a dedicated,
lossless PR-bound context. The PR handoff binds reviewers and coder follow-ups
to the recorded plan hash and subject, independently of truncated issue
comments or compact PR history. Issue/PR resume validates that handoff (and any
staged parent/child topology) before invoking an agent; an ordinary PR with no
planning provenance continues through the normal no-plan path.

Signed human instructions are rendered as `Requirement hr-<digest>` and use the
same content-derived ID in acknowledgements, repair, round metadata, and resume.
Insertion or chronological reordering cannot change an existing instruction's
ID, while an edited body receives a new ID. Historical positional
acknowledgements require a fresh acknowledgement against the currently surfaced
stable IDs rather than being reinterpreted by current ordering.
Original issue requirements, later valid human instructions, and safety
constraints outrank an approved plan; a defective plan must be raised as a
scope/plan decision, not silently replaced by review prose.

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

For an issue-created managed-CI PR whose original authorization comment is
missing, recovery is deliberately explicit. On a voluntary or plan-limited
base, use `--managed-ci-fresh` with `--managed-ci` and the unprotected waiver;
PR mode must also provide `--managed-ci-issue <issue-number>`. This creates a
new operator grant after validating the live PR tuple, fetching the issue and
canonical plan scope, and confirming a server-observed issue-to-PR association.
It reuses any valid exact-head authorization and does not adopt arbitrary
existing PRs. If structured response validation rejected the implementation
before accepting its PR number, strict protection instead uses ordinary
same-PR issue/PR discovery and resume; the fresh unprotected grant is not
available or required for a strict base. If a strict draft was left unlabeled,
that resume requires the authenticated strict PR tuple and an actor-owned
historical `agent-loop-managed` label event, then reapplies the label without
creating a waiver record.

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

### Advisory architecture context

By default, workflows read the conventional repository-local `ARCHITECTURE.md`
from committed Git objects and include a bounded, revision- and hash-labeled
overview in planning, implementation, and review prompts. Use
`--architecture-path docs/ARCHITECTURE.md` for a validated repository-relative
POSIX override, or `--no-architecture-context` to opt out. Missing, unsafe,
binary, oversized, or unavailable documents preserve the legacy prompt path.
The read and rendering budgets are configurable with `--architecture-read-size`,
`--architecture-snapshot-max-chars`, and `--architecture-aggregate-max-chars`;
`--managed-context-max-chars` bounds active managed prompt context and fails
closed when an explicitly restrictive cap cannot retain protected requirements
or plan context.
The overview is untrusted orientation only: source inspection and full-diff
review remain required, and it is not a whole-codebase audit or correctness
guarantee. Architecture-impact assessments distinguish contract changes from
meaningfully unchanged work and guide updates to the canonical document.

### Machine obligations and CI repair

Selective review keeps reviewer-owned findings separate from machine-owned
obligations. Managed exact-head CI, ordinary PR checks, migration validation,
mergeability, and human-requirement acknowledgement have stable authority kinds
and are upserted by kind rather than accumulated as duplicate reviewer items.
After a machine failure, the failed head is permanently ineligible for
requalification: the coder must produce a strictly different head, and the
selective scheduler takes the broad full-board path for that repair transition.

Reviewer dispositions keep machine records complete in the review ledger, but
approval is evidence about the correction only; it is never CI success. Each
authority clears only from its own fresh, correctly correlated result for the
current head and base. Skipped, absent, intermediate-filtered, stale, or
uncorrelated checks cannot satisfy the final gate. Qualification checkpoints
record the obligation, heads, review/plan/requirement identities, scheduler
inputs, attached attempt, and the two independent one-shot allowances (one
failure repair and one head-change re-review, for at most `max_rounds + 2`).
Restarts resume a correlated attempt when possible and otherwise fail closed.
Terminal diagnostics distinguish reviewer objections, repair-required CI,
qualification pending, migration/mergeability validation, unknown persisted
state, and external infrastructure stops.

Signed comments ending in `-- Human Reviewer` are treated as explicit human
requirements and remain approval-critical. See
[Human requirements](docs/local_agent_loop.md#human-requirements) for the exact
contract.

### What "review" and "approval" mean

Agent-loop publishes model reviews as structured comments in the pull
request's conversation. An `Approved` verdict means that the named model
approved the recorded PR head under agent-loop's protocol. It is not a native
GitHub pull-request review, does not populate GitHub's **Reviewers** or
**Reviews** panels, and does not satisfy a branch-protection rule that requires
approving GitHub reviews.

The agent CLIs normally share the GitHub identity authenticated through `gh`,
so their model signatures identify protocol participants rather than distinct
GitHub accounts. If a repository requires native GitHub approvals, obtain them
separately from an eligible human, bot, or GitHub App identity. GitHub's own
merge protections remain in force in addition to agent-loop's reviewer and CI
gates.

When a coder and reviewer maintain an evidence-backed disagreement after the
reviewer's reconsideration turn, agent-loop posts a **Human decision required**
comment and exits with status `4`. This is an intentional operator boundary,
not an orchestration failure. Respond on the PR with the decision and required
action in a comment ending with `-- Human Reviewer`, then resume the PR. An
issue-created managed PR keeps its suppression label while waiting, so ordinary
CI is not released before approval.

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

Plan-first mode supports five post-approval choices:

| Mode | Result after plan approval |
| --- | --- |
| `plan-only` | Post the approved plan and stop. This is the default. |
| `implement-one-shot` | Implement the approved plan in one PR. |
| `decompose-only` | Create detailed child issues for the approved phases and stop. |
| `implement-by-phase` | Create the phase issues and implement only the first phase. |
| `auto` | After approval, select one-shot or by-phase, then route each staged child from its reviewed disposition. |

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
work. `auto` cannot be combined with either `--materialize-split-issues` or
`--split-stage`, because its topology is unknown until approval. Read
[Phased decomposition versus split materialization](docs/local_agent_loop.md#phased-decomposition-versus-split-materialization)
before filing child issues.

For fresh staged plans, the planner declares each child as
`direct-implementation`, `requires-child-planning`, or `human-owned`, and plan
reviewers approve or block that semantic choice. The orchestrator checks typed
readiness fields and approved-parent provenance; it does not infer readiness
from issue size or file count. A direct child uses the parent stage contract
without duplicate planning. A planning-required child records its route first,
then must run with `--plan-first` and complete its own reviewed plan before
implementation. Recorded routes survive reruns and cannot be switched with CLI
flags. See the detailed
[child execution disposition contract](docs/local_agent_loop.md#child-execution-dispositions).

### Approved follow-up dedupe

When approved future follow-ups are summarized or filed, semantic reuse is
enabled by default after deterministic narrowing. Use
`--no-semantic-followup-dedupe` for deterministic-only operation. The provider
and its bounds are configurable with `--semantic-followup-backend`,
`--semantic-followup-model`, `--semantic-followup-timeout-seconds`,
`--semantic-followup-max-calls`, `--semantic-followup-max-candidates`, and
`--semantic-followup-prompt-char-limit`. Only high-confidence matches suppress
or merge work; uncertain matches are filed with a possible-duplicate note.

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

The default Antigravity model chain is `Gemini 3.8 Flash (High)`, then
`Gemini 3.7 Flash (High)`, then `Gemini 3.6 Flash (High)`, then `Gemini 3.1 Pro (High)` for eligible capacity
failures. Override it with `--antigravity-model` or
`--antigravity-models`. Antigravity turns are single-shot and its usage totals
are estimated because `agy` does not expose token counts.

Malformed structured responses get a format-repair pass. By default it uses
Antigravity with the repair chain `Gemini 3.8 Flash (Medium)`, then
`Gemini 3.7 Flash (Medium)`, followed by the Antigravity chain above. Use
`--repair-backend` and repeatable `--repair-model` to change it. Models that
`agy models` does not list are skipped. When `agy` reports a transient
`model-access validation errors` failure on its own output (stdout with no
response artifact and no structured JSON response), that repair model is
retried once, and then the next model in the chain is tried. If the chain ends
on that transient failure, the run stops with a resumable
`repair-provider-failure` and a suggestion to re-run the same command. It is
not reported as a deterministic plan-validation failure. A valid repaired
response is always accepted, and model-authored text that only quotes the
phrase is still treated as invalid output.

Backend-specific authentication, model selection, fallback, timeout, and
executable-replacement behavior are documented under
[Agent backends](docs/local_agent_loop.md#agent-backends).

### Reasoning-effort defaults

Agent-loop owns the effort setting for Codex and Claude. When no effort option
is supplied, every invocation explicitly receives `medium`; a local CLI config
file or inherited environment value cannot silently change that selection.
The precedence is the matching role override (reviewer or implementation), agent-wide option,
then the tool default. Use `--codex-reasoning-effort xhigh` (or the matching
implementation option) for an explicit higher-effort Codex/Luna run. Claude
accepts `low`, `medium`, `high`, `xhigh`, and `max` through `--claude-effort`.
Antigravity's selected model already carries its tier, such as
`Gemini 3.8 Flash (High)`, and is not rewritten by this setting. Startup logs,
signatures, usage records, and new round metadata distinguish configured effort
from verified runtime observations; older records retain unknown values.

### Separate Coder and Reviewer Models

When Codex or Claude occupies both seats, select its reviewer independently:

```bash
agent-loop pr 123 --repo OWNER/REPO \
  --coder codex --reviewer claude --reviewer codex \
  --codex-model gpt-5.6-luna --codex-reasoning-effort xhigh \
  --reviewer-codex-model gpt-5.6-sol --reviewer-codex-reasoning-effort medium
```

Claude has matching `--reviewer-claude-model` and `--reviewer-claude-effort`
options. These overrides apply to plan reviews, PR reviews, and discussion
participants, not the coder or discussion analyzer. Model and effort are
independent: omitting either retains that setting's existing fallback.
Reviewer overrides survive the issue-to-PR implementation handoff; repeat
them when starting a separate resume command. See the
[issue-mode example and precedence](docs/local_agent_loop.md#independent-reviewer-selection).

### Selective intermediate PR review

PR review keeps the historical `all-reviewers` policy by default. To pause
reviewers who have already approved while a narrow fix is checked, opt in with:

```bash
agent-loop pr 123 --repo OWNER/REPO \
  --pr-review-policy selective-intermediate
```

The initial candidate always runs the full configured board. A later transition
is considered narrow only when repository-observed Git history is available and
the complete diff contains ordinary text additions or modifications within the
exact reviewer-provided `fix_scope` paths. Workflow, dependency/lock,
build/packaging, schema/migration, and repository-policy/configuration paths
use deterministic broad rules; customize the rule list with repeated
`--pr-review-broad-rule` options. Missing, disputed, invalid, out-of-scope, or
uncertain scope is conservative and reactivates the full board. The
`--pr-review-force-full` latch remains active for the rest of the run.

For example, with Claude, Codex, and Antigravity required, a Codex-owned fix
scoped to `src/worker.py` can invoke Codex for the intermediate head while the
other two approvals are retained as historical evidence. Once the obligation
is cleared, the final exact-head sweep invokes only Claude and Antigravity if
they lack qualifying approvals for that head. A same-head approval is reusable
only when its approved-plan/handoff identity, surfaced requirements, and
acquisition contract still match. Every required reviewer must still approve
the exact final head; selective scheduling never changes CI or merge gates.

Resolution ownership is durable: a non-owner blocking disposition adds that
reviewer as an owner, while a non-owner resolved disposition is evidence only.
Every owner must independently clear the obligation. An unavailable owner is
not an approval; when all remaining owners are unavailable, the run stops as an
incomplete review without another coder turn. Scheduler decisions are recorded
in round audit metadata and recover conservatively after interruption or a
configuration change. Issue-mode implementation handoffs carry the same PR
policy, while planning and discussion remain outside this policy.

### Primary-then-panel PR review

The opt-in staged policy lets one configured primary work to exact-head approval
before an independent secondary audit:

```bash
agent-loop pr 123 --repo OWNER/REPO \
  --pr-review-policy primary-then-panel \
  --primary-reviewer codex \
  --reviewer codex --reviewer claude --reviewer gemini
```

The primary is the first and only reviewer in the normal opening phase, and it
alone rechecks its own findings on each narrow fix until it approves the exact
head. Once it approves, every secondary reviews the complete current
base-to-head diff from a common snapshot in the `secondary-audit` phase;
secondary prompts are independent and are not finding-check prompts. A scoped
secondary fix enters `remediation`, which rechecks every active finding owner
and the primary, then performs a complete exact-head `final-secondary-sweep`
for every secondary missing approval.

Before the primary's first exact-head approval the phase is strictly
primary-only. A broad or out-of-scope change, missing or ambiguous fix scope,
recovery ambiguity (a stale qualification checkpoint, an architecture identity
change, obligation-digest drift), or legacy/malformed/contradictory scheduler
metadata re-invokes only the primary with full context. The audit reason starts
with `strict pre-panel fallback:` and nothing is latched. Secondaries are never
treated as finding owners before their first legitimate panel invocation. The
panel is opened only by a *qualified panel opening*: a `secondary-audit` record
preceded by the primary's approval of the same exact head, or an
operator-attributed force-full record. Secondary reviews, `full-board`
checkpoints, and unattributed force-full latches written before such an opening
(for example by the pre-#840 automatic escalation) are premature-panel
artifacts. They are ignored, and a secondary approval counts only when it was
recorded after the qualified opening.

After the panel has opened, the existing conservative rules apply, with audit
reasons prefixed `post-panel fallback:`. An active finding with a broad,
ambiguous, out-of-scope, or unreconstructible change, and any unsafe head change,
selects the complete board. That full-board decision, like any automatic
recovery reason, raises a durable latch recorded with source `automatic`, so
later heads stay on the complete board instead of returning to owner-scoped
remediation. Missing input is never treated as approval.

If safety cannot be established without the panel, the run stops with a
`PR review scheduling diagnostic` instead of silently spending it. That happens
when a finding is pending on a configured secondary, or a premature secondary
review in an interrupted round is blocking, and there is no qualified opening.
No reviewer runs. Rerun with `--pr-review-force-full` to authorize the complete
board. Undecodable scheduler history (for example a missing round-metadata
sidecar) also stops with the diagnostic, at startup or at a round boundary. It
stops even with the flag, because resume, approval, and qualification
accounting all depend on that history; restore the missing records and rerun.
The operator latch
is durable, is recorded with source `operator`, and is itself a qualified
opening. Under the override, a premature blocking secondary review is
superseded rather than consumed: it is listed in the operator audit comment,
excluded from the ledger and approvals, and replayed to that reviewer only as
non-authoritative context for its fresh review.

The tradeoff is deliberate. Pre-panel uncertainty costs one extra primary turn
instead of N secondary turns. Safety is kept because the panel's first
invocation is always a complete, independent review of the whole diff, and
every configured reviewer must still approve the exact final head. Public audit
comments and logs name the phase, exact head, primary, active owners,
force-full state and source (`operator`, `automatic`, or `none`), and
policy-neutral cumulative calls avoided. The policy is
opt-in and does not change the default or the CI/merge gates. Use
`review-evaluation` with frozen local artifacts to compare latency/cost against
independent severity-weighted coverage before any proposal to change the
default; measurements without verified provenance are reported as unavailable.

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

Before a managed wrapper is recommended, agent-loop runs a non-mutating
`run-tests --preflight` probe. It checks at most the absolute `agent-loop`
console entry and the current-interpreter module fallback, once per invocation,
with a five-second watchdog (and no more than six distinct recognized inner
launcher candidates per invocation). The inner probe uses the effective target
environment, including ambient values merged with a partial overlay. The probe
never runs remembered test arguments,
installs dependencies, contacts a database, or executes an arbitrary shell.
Recognized inner launchers receive the same bounded `--version` probe: direct
`pytest`/`py.test` and exactly `<python> -m pytest`; other commands remain
unknown rather than being judged from text or exit codes. The runtime result
keeps independent `wrapper_bootstrap`, `inner_exec`, and `suite_start` states,
so a missing executable or import is launcher health rather than a suite
failure. Collection/configuration errors, failing tests, timeouts, and
interruptions after startup remain ordinary suite evidence.
Probe cleanup is process-tree aware: POSIX probes use a dedicated process
group, and Windows probes use a kill-on-close Job Object. If Windows cannot
provide that owning boundary, the probe is not launched and its result remains
unknown rather than running an uncontained child.
Python identity is checked without execution: the running interpreter and
symlinks to it are trusted, as are byte-for-byte copies in a conventional
`pyvenv.cfg` environment; name-only scripts or native binaries are unknown.

Schema-v1 `test-runtime.json` persistence adds a separate `launcher_health`
collection. It records only bounded diagnostics, repository/checkout and
environment identities, candidate identity, timestamp, and provenance; it
does not store environment dumps or external paths (the exact canonical
managed-wrapper path is retained for that wrapper). Health failures expire
after 24 hours, are capped independently at eight records per identity and 100
identities, deduplicate, and are cleared immediately by a matching successful
probe. Reinstalling a wrapper, changing its interpreter/dependencies,
recreating a virtualenv, or changing the checkout/environment changes the
identity and permits a fresh probe. Health is advisory: the configured test
command is always shown, verified alternatives are scoped to the current
checkout/environment, and direct agent-shell failures cannot be claimed as
comprehensively captured.

## CI and Merge

When a review round requires coder changes, agent-loop also checks for CI
failures already reported on the reviewed commit and includes them with the
reviewer findings. It does not wait for queued or running checks at this handoff.
Missing checks and recognized runner-infrastructure stalls are not added as
code defects. Managed CI still defers its final qualification until approval.

A blocking review is treated as CI-wait-only only when its findings and summary
are unambiguously simple CI-status statements (or the summary is boilerplate).
Mentioning a check name such as `test` is not sufficient. Mixed or ambiguous
findings remain blocking; the tool must not erase code concerns to avoid a CI wait.

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

- [`ARCHITECTURE.md`](ARCHITECTURE.md): canonical component map, execution flows, state, and trust boundaries.
- [`agent-loop --help`](docs/local_agent_loop.md#usage): full command and option reference.
- [`docs/local_agent_loop.md`](docs/local_agent_loop.md): detailed lifecycle, protocol, recovery, CI, memory, and safety reference.
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
Browse the focused test modules in [`tests/`](tests/) and see the component map
and diagrams in [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Related Tools

This project is a standalone local GitHub lifecycle orchestrator. Projects such
as [claude-review-loop](https://github.com/hamelsmu/claude-review-loop),
[codex-review](https://github.com/boyand/codex-review), and
[codex-plugin-cc](https://github.com/openai/codex-plugin-cc) integrate review or
delegation into a particular agent host. Here, the orchestrator stays outside
the agent hosts and can reverse coder/reviewer roles.

## License

[MIT](LICENSE)
