# Local Coding Review Agent Loop

`coding-review-agent-loop` is a local CLI that orchestrates coding agents through a GitHub pull request review loop. Its main advantage is account reuse: it shells out to locally authenticated `claude`, `codex`, `gemini`, and `gh` CLIs instead of calling model APIs directly. If your local agent CLIs are backed by existing AI subscriptions or authenticated developer accounts, the review loop can use those existing entitlements rather than requiring separate model API keys.

**Claude billing note:** Anthropic had announced that non-interactive `claude` usage — including `claude -p` as used by this tool — would move from your subscription's rate limits to a separate monthly Agent SDK credit. As of June 15, 2026 that change has been **postponed**: `claude -p` / Agent SDK usage continues to draw from your existing Claude subscription as before, with no separate credit, and Anthropic has said it will give advance notice before any future change. See [Anthropic's support article](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan) for the latest.

The default flow is:

1. A coder agent creates or updates a PR.
2. One or more reviewer agents review the PR.
3. If any reviewer finds blockers, the coder fixes the PR.
4. The loop repeats until every reviewer approves in the same round or `--max-rounds` is reached (default: 10).

### Reviewed execution strategy contract

New planning and plan-revision turns must emit generation 1 with an
`execution_recommendation`. Its scope ledger gives every requirement a stable
ID and acceptance criteria; coupling constraints keep inseparable work in one
delivery. A `one-shot` recommendation has one exact-once delivery and explicit
`none` retained-parent and final-integration allocations. A `staged`
recommendation has ordered enriched child stages plus independent retained and
final-integration allocations, with exact-once coverage and at least two real
allocations. It also records automation, dependencies, rollout risk,
compatibility constraints, and caveats.

The recommendation is executable only through an explicit policy compatible
with its canonical strategy. `one-shot` permits `implement-one-shot`; `staged`
permits `decompose-only` or `implement-by-phase`. The opt-in `auto` policy is
resolved only after approval: `one-shot` selects `implement-one-shot`, while
`staged` selects `implement-by-phase`. Legacy-undecided plans stop with an
actionable request for a reviewed revision, and incompatible explicit pairs
fail closed before approval-bound writes. A compact execution decision records the approved
plan identity and full recommendation digest, while the complete recommendation
and any bounded transport sidecars remain the recovery source. Fresh topology
summaries, child identities, and handoffs are keyed by strategy `staged`, source
`approved-plan-v1`, contract version, digest, ordinal, and stable stage ID;
legacy records keep their exact historical mode/source lookup and serializer.
Plan-only retains its existing approved-follow-up audit record, explicit split
materialization, and unfiled-scope warning, but creates no fresh decision,
canonical topology, phase-child handoff, or dispatch. Old unversioned comments and two-field
typed stages remain legacy-undecided. A bounded repair can only reformat a
complete recoverable v1 source; missing or partial strategy data requires a new
planner turn.

### Child execution dispositions

Every child in a fresh staged generation-1 recommendation has a reviewed
execution disposition. The planner makes the semantic choice and plan
reviewers approve or block it. `agent-pr` stages use
`direct-implementation` when the approved parent slice is a complete contract,
or `requires-child-planning` when design choices remain. `human-action` and
`manual-close` stages use `human-owned`. The orchestrator validates only the
typed declaration, internal consistency, direct-readiness fields, and
provenance; it does not infer readiness from issue size, file count, or prose.

Direct readiness requires no unresolved design decisions and non-empty
non-goals, dependency notes, compatibility constraints, and rationale. A
direct child records its handoff and implements from the approved parent phase
without a duplicate plan review. A planning-required child records a planning
handoff before any planning agent runs, then runs
`agent-loop issue <child> --plan-first --plan-execution-mode auto`. Only after
that child plan is reviewed may a one-shot child recommendation reach an
implementation coder. If the child itself recommends a staged topology, the
run stops for human handling under #720 before persisting nested topology work.

The effective route is deterministic. A recorded handoff wins and is
reconciled first. Without one, one valid signed override may select the route;
otherwise the reviewed phase declaration is used. Missing legacy metadata or
an unsupported topology source fails closed to child planning. Direct routing
always reuses the reviewed parent phase's readiness evidence—override metadata
cannot supply it. Human-owned stages remain a stop. Plain issue mode cannot
switch a planning route, and `--plan-first` cannot switch a direct route.

Signed overrides are durable JSON records in a comment signed by a human
reviewer. They identify the parent issue, approved plan hash, stable stage ID,
new disposition, and rationale. Discovery first rejects misaddressed or
out-of-topology records, then scopes, deduplicates, and conflict-checks records
for the routed stage. Conflicting records never use latest-wins ordering. The
handoff stores the canonical record digest; every resume and PR validation
must rediscover that exact record. Deleting, editing, adding after dispatch, or
superseding an applied record causes a fail-closed human-repair stop.

PR provenance follows the reconciled route. A direct child's PR binds to the
approved parent phase plan hash. A child-planned PR binds to the child's own
reviewed plan hash while retaining and validating its parent identity,
decomposition summary, and inherited risk-matrix obligations. Reruns reuse the
checkpoint, handoff, child plan, implementation handoff, and canonical PR, so
the route cannot change because context was truncated or the command was run
again.

A child plan does not have to reproduce its inherited rows byte for byte. For
each inherited row it must keep the row ID, never lower applicability
(`not-applicable` < `applicable` < `required`), copy every parent forbidden side
effect exactly (case and whitespace included; additions and reordering are
fine), and keep the text of the entry path or mode, initial state, event, and
expected outcome verbatim, adding refinements after it. The label, scope links,
and execution owner are free, child-local rows may be added, and the proposed
test level and location may change. Every admissible difference other than
label, scope links, and ordering is shown to the child plan reviewers as a
parent-versus-child coverage delta, and reviewers block any extension or test
placement change that narrows the inherited coverage.

The child planner and reviewers are shown the inherited rows in full in every
plan prompt form, in exactly the form the validator compares. The check runs
during child planning, before a candidate plan is posted: a candidate that
weakens an inherited row is never published or reviewed, and the planner is
re-invoked with a diagnostic naming each row and field, at most two times per
planning round. If every candidate is rejected, one authenticated diagnostic
record is posted and the run stops before any reviewer or implementation turn;
rerunning the same `--plan-first` command feeds that record to the next planner
turn. Neither prompt block is ever truncated: inherited rows too large to show
losslessly stop child planning with a sizing diagnostic that points at the
parent plan, and a child whose deltas are too large to show is replanned. PR
validation applies the same comparison on parent dispatch, direct child
invocation, and skill mode, so an already approved child plan that satisfies
these rules resumes PR review without replanning.

### Re-planning an approved child plan

The planning-time check above prevents a *new* child plan from being approved
while it weakens inherited rows. A plan approved earlier, for example before the
contract tightened, is a historical artifact that can still fail the check. It
is never grandfathered and the comparison is never relaxed; instead the failure
names one of two supported routes.

**Before any implementation handoff.** When a resumed, fully approved child
plan fails the check, the loop posts one plain audit comment (keyed by the plan
hash, so a rerun does not repeat it) and runs an enforced revision turn: the
planner receives the field-level diagnostic attributed to the orchestrator, no
reviewer item is invented, the revision is mechanically rechecked before it is
published (at most two replans, with the usual persisted diagnostic on
exhaustion), and the next round runs the complete reviewer board with no
approval carried from the superseded plan. If the round budget is already
spent, the run stops with a message naming `--max-rounds`. No signed record is
needed here, because nothing downstream is bound to the plan.

**After a handoff, with an open PR.** Issue mode judges the handed-off plan
before it resolves the canonical PR. If the plan is inadmissible and no signed
record matches, the run stops before any agent turn or write, printing the
weakening diagnostic, this record pre-filled for the child, and the rerun
command:

````markdown
Child plan supersession:

```json
{
  "child_issue": 925,
  "kind": "child-plan-supersession",
  "parent_issue": 924,
  "rationale": "Approved before the inherited-matrix contract tightened.",
  "schema_version": 1,
  "stage_id": "stage-1",
  "superseded_plan_hash": "<16-hex plan hash printed by the diagnostic>"
}
```
-- Human Reviewer
````

The record is read only from the child issue and only from a comment carrying
the standalone human reviewer signature. Malformed or unsigned records are
reported and ignored. A signed record naming another child, parent, or stage
stops for a human decision. Identical duplicates collapse; records for different
superseded hashes coexist (one per historical re-plan); two distinct records for
the same superseded hash always stop for a human decision.

The record is consulted for every handed-off plan, not only an inadmissible
one: admissibility asks whether the plan is broken, the signed record asks
whether a human authorized replacing it. An admissible plan with no matching
record resumes its PR unchanged; an admissible plan with a matching record (a
deliberate re-plan such as an authorized scope reduction) reopens planning
exactly like an inadmissible one. In that case the planner receives the
record's rationale as the revision instruction, and a resumed approval of the
superseded plan runs the revision instead of approving and rebinding the plan
the human asked to replace.

With exactly one matching record and an open canonical PR, planning reopens:

- The record's digest is written into the round metadata of every planner round
  of the re-plan. Only a plan whose planner rounds start from the superseded
  plan, chain to each other without a gap, all carry that digest, were posted
  after the signed comment, and resolve to one discoverable signed record is
  treated as produced by the authorized re-plan. A later plan that predates or
  bypasses the authorization, a round bound to another record, a gap, or a
  deleted record fails closed before any agent runs.
- When the revision is approved, the existing PR is re-authenticated (open, same
  number) and rebound by **one** issue comment containing the superseding
  issue-to-PR handoff record and a rebind audit record (child, PR, both plan
  hashes, digest, first re-plan round, approved round). The PR-side closing
  contract is never rewritten, so an interruption leaves either the old binding
  (the rerun performs the rebind only) or the new one (the rerun resumes the
  PR). The write is skipped when it already exists. A plan-only invocation
  stops after approval; the next rerun performs only the rebind.
- The PR-side closing contract keeps authenticating against the closing-contract
  lineage base: the most recent issue-side handoff record that is not a same-PR
  equal-ID plan replacement. The plan hash comes from the latest record. The
  closing-ID contract digest never identifies a plan replacement, since every
  unchanged-ID rebind shares it.
- Every path on which a replacement plan can become a PR's plan context, issue
  entry, `agent-loop pr <n>` entry, and mid-run adoption at a qualification
  gate, runs one verifier: the rebind audit record must be in the same comment
  and agree with the handoff, the digest-bound rounds, and a discoverable signed
  record, and the replacement plan must itself pass the inherited check.
  Otherwise the run stops with a human-repair diagnostic and posts nothing.
  The verifier follows the most recent plan-changing handoff record, so a
  closing-issue superset recorded after a rebind never hides it. How the
  current plan became the binding is verified before a new signed record for
  that plan is accepted: an unverified replacement cannot be re-planned away,
  while a verified plan that a later contract tightening made inadmissible can
  be superseded again with its own signed record.
- The round that reviews the revision runs the complete reviewer board under
  staged planning, including when the run restarted right after the revised
  plan round was posted; the requirement is recomputed from the issue history.
  PR reviewer approvals are keyed by plan hash and subject, so none recorded
  before the rebind counts under the new plan.
- The execution decision record the original planning run left under the
  superseded plan hash stays on the issue as history. After a rebind, issue
  mode walks the live PR's plan-changing handoff transitions back from the
  latest one. Each transition must carry its own rebind audit record and a
  verified signed re-plan lineage, and a standalone audit record with no
  transition counts for nothing. A decision bound to a plan hash that the
  chain replaced is skipped, and the resumed run
  records the decision for the rebound plan. A decision under any other plan
  hash is still a competing topology and fails closed. An issue that got
  stuck on this conflict before the fix recovers by rerunning the same
  issue-mode command, with no manual edit.
- The managed-CI authorization records on the PR that name the superseded
  plan hash likewise stay as history. The explicit fresh authorization
  (`--managed-ci-fresh`) uses the same verified chain: an actor-owned record
  whose only difference is a plan hash the chain replaced is not a conflict,
  and the new grant is bound to the rebound plan. A record that differs in any
  other field, or names a plan hash no verified transition replaced, still
  refuses. A PR stuck on this conflict recovers by rerunning the fresh
  authorization command the ordinary path recommends.

`agent-loop pr <n>` never re-plans or rebinds; it prints the issue-mode route.
That includes an admissible bound plan with a pending matching signed record:
PR mode stops before any reviewer runs rather than reviewing the PR against a
scope no approved plan contains, since a PR-mode coder turn cannot produce the
replacement plan. The same check runs again on the freshly fetched child issue
at qualification and before every approval or merge, under every review
policy and with or without `--auto-merge`. A record posted while reviewers
were running therefore stops the run before it approves or merges.
Keep the signed record on the child issue after the rebind. Re-planning
continues the child's round numbering, so a higher `--max-rounds` may be needed.
Not supported: abandoning or replacing the PR, re-planning direct-implementation
or non-child issues, parent-plan revision, rewriting the PR-side contract, and
any unsigned or flag-only bypass once a handoff exists.

**Migration note.** A tightened inherited-matrix contract makes children
approved under the looser contract inadmissible the next time the loop touches
them. Both routes above are the intended migration path.

### Removing an unavailable reviewer from an in-flight run

A persisted scheduler contract (the required reviewer board, the policy, and
the primary) is immutable for the run. If one reviewer's backend becomes
unavailable, for example because its quota is exhausted, rerunning without that
reviewer stops with the contract-drift error. That error now also prints a
filled-in **signed reviewer-board amendment** record. A human operator posts it
to record a deliberate, audited reduction of the board:

````markdown
Reviewer board amendment:

```json
{
  "effective_from_round": 3,
  "flow": "plan",
  "issue": 942,
  "kind": "reviewer-board-amendment",
  "original_required_reviewers": [
    "Codex",
    "Claude",
    "Antigravity"
  ],
  "policy": "primary-then-panel",
  "pr_number": null,
  "primary_reviewer": "Codex",
  "rationale": "Antigravity weekly quota exhausted.",
  "reason": "backend-unavailable",
  "removed_reviewers": [
    "Antigravity"
  ],
  "schema_version": 1
}
```
-- Human Reviewer
````

- **Where to post it.** Post a `plan` record (`issue` set, `pr_number` null)
  on the issue being planned. Post a `pr` record (`pr_number` set, `issue`
  null) on the PR itself, including standalone `agent-loop pr` runs. A record
  on the wrong surface, a record naming another issue or PR, and a `pr` record
  on the owning issue all stop for a human decision.
- **Choosing `effective_from_round`.** Use the round number the drift error
  prints. That is the round the resume re-enters, whether the round is only
  partly recorded or already reconciled, and it is not always the latest
  visible round number. Any other value stops before any agent turn or comment
  and prints the corrected template.
- **Retroactive use.** The record rescues runs whose contract was persisted
  before this feature existed. Earlier rounds are never rewritten. Inside the
  re-entered round, reviews already posted by the remaining reviewers are
  reused, and the removed reviewer is never invoked again.
- **What stays immutable.** `policy` and `primary_reviewer` must match the
  persisted contract. The primary cannot be removed, and a `primary-then-panel`
  board must keep at least one secondary. Re-adding a removed reviewer is
  contract drift. Every scheduler record posted after the amendment carries the
  amended board and the record's digest; any other contract fails closed, and
  at PR qualification it refuses the merge. Amendments can chain, with each
  record's `original_required_reviewers` equal to the previous amended board.
  Two different records that amend the same board always stop for a human
  decision.
- **Findings and approvals.** An active finding whose only pending owner was
  removed is reassigned to the primary, or to every remaining reviewer when
  there is no primary. It stays blocking until a new owner clears it; nothing
  is auto-cleared. Approvals the removed reviewer already gave remain in the
  history but are no longer required. The run posts one audit comment naming
  the record, the activation round, and every reassignment. Scheduler audits,
  completion messages, and (for PR runs, on every completion path including
  managed CI) one plain completion comment on the PR note that the run
  finished on a reduced board. The activation round always posts a fresh
  scheduler checkpoint carrying the amended board and digest, even when every
  remaining reviewer's review is reused.
- Unsigned or malformed records are ignored with a logged diagnostic. A comment
  that contains only the record is not treated as a signed human requirement.
  All-reviewers PR runs and the non-staged plan path persist no contract, so
  there you change the reviewer flags directly and must not post a record.

**Lineage rules.** Only scheduler records that carry a contract are compared:
the plan `scheduler-prelaunch` checkpoint and PR scheduler records written under
a selective policy. Coder, reviewer, and phase-advance records are never
compared and never carry the digest. The base contract is the one on the
earliest contract-bearing record. Each record is judged by exactly one link of
the amendment chain: the latest amendment whose comment precedes the record and
whose `effective_from_round` is at or before the record's round. The record must
carry that link's board and digest exactly. A record that precedes every
amendment carries the original board and no digest. Plan resume, PR startup, and
the PR qualification gate all use this one resolver, and the gate re-reads the
amendments from the same fresh PR comment fetch as the scheduler records.

**Ledger view.** Persisted `prior_items` inside a round are never rewritten.
The reassignment is a derived view that the scheduler, disposition
reconciliation, and completion check recompute from the record on every run.
The ledger handed to the next round is built from that view, so from then on
the persisted items carry explicit owners that exclude the removed reviewer.

### Risk-based mode and transition matrices

For planning work involving multiple modes, lifecycle transitions,
persistence/restart, authorization, or recovery/failure paths, the generation-1
planning contract can require a bounded structured matrix. Applicable rows use
stable matrix-local IDs and record the entry path or mode, initial state, event,
expected outcome, forbidden side effects, proposed test level/location, related
scope IDs, and one execution owner. The planner also records important
exclusions. Narrow local work may provide a non-empty not-applicable rationale
instead of rows; the matrix is not a mandatory Cartesian product.

The matrix is a review aid until implementation evidence maps each delivered row
to an actual workflow test and existing receipt. Evidence keeps missing,
not-run, failed, timed-out, stale/unverified, and helper-only or earlier-guard
caveats visible. The structured payload and its identity are authoritative in
round metadata and the bounded sidecar; rendered tables are derived output.
Recovery tolerates renderer wording changes, while payload corruption or a
matrix-only rendering/hydration problem reports zero enforceable matrix rows
without invalidating an otherwise verified plan hash and subject. Legacy records
without a matrix remain resumable without fabrication. Both CLI and skill-mode
helpers accept `--require-risk-test-matrix-contract` for fresh contract checks.

The default coder is Claude and the default reviewer is Codex. Reverse the direction with `--coder codex --reviewer claude`, or use Gemini with `--coder gemini` / `--reviewer gemini`. Repeat `--reviewer` to require multiple reviewer approvals.

### Machine obligations and resumable qualification

The review ledger distinguishes reviewer-owned findings from machine-owned
obligations. Managed exact-head CI, ordinary GitHub checks, Alembic migration
validation, merge conflicts, and human-requirement acknowledgement each have a
stable authority kind. Repeated failures for one kind update the existing
obligation, while authoritative success clears only obligations of that same
kind.

After any CI failure, its exact failed head remains ineligible for
requalification. The coder must produce a strictly different repair head.
That transition is scope-less and therefore selects the broad full reviewer
board under `selective-intermediate`; reviewer approval of the correction is
advisory evidence and cannot substitute for CI success. Only the source-specific
fresh result, correlated to the required head/base and provenance, clears the
obligation. Failed, pending, skipped, absent, stale, intermediate-filtered, or
uncorrelated checks never satisfy the final gate.

Before dispatch or a watcher, the loop records a qualification checkpoint in
round metadata. It contains the obligation identity and lifecycle, failed and
candidate heads, base and review/plan/requirement/scheduler identities,
qualification attempt identity, and the independent one-shot allowances: one
failure-repair extension and one head-change re-review extension. The effective
ceiling is therefore at most `max_rounds + 2`. A restart attaches to a
correlated in-flight attempt when possible; absent, malformed, stale, or
contradictory checkpoints force safe reconstruction and full review.

At the round limit, diagnostics identify the actual next action: a named
reviewer-owned blocker, a CI obligation awaiting a repair head, a reviewed
correction awaiting qualification, migration or mergeability validation, an
unknown persisted obligation, or external infrastructure recovery. The loop
does not describe a unanimous reviewer board as blocking when only CI remains.

Gemini CLI consumer access (free / Google AI Pro / Ultra) is retiring on June 18, 2026; personal-account `gemini` users should migrate to the Antigravity CLI (`agy`) with `--coder antigravity` / `--reviewer antigravity` (pick a single model via `--antigravity-model`, or an ordered fallback chain via `--antigravity-models`; default chain `Gemini 3.8 Flash (High)` → `Gemini 3.7 Flash (High)` → `Gemini 3.6 Flash (High)` → `Gemini 3.1 Pro (High)`). Enterprise / API-key Gemini CLI paths may remain available for organizations that still have access, so the `gemini` backend is retained for those users. Direct Gemini CLI support is best-effort: maintainers without enterprise Gemini CLI access need reporter-provided `.agent-loop-logs/*gemini.log` output, response-file contents, CLI version, and any sharable account/access context to debug live `gemini` failures. Antigravity turns are single-shot (no cross-round session resume) and report estimated usage.

Every `agy --print` call passes `--print-timeout` from
`--antigravity-print-timeout-seconds` (default `600`, i.e. ten minutes), which
overrides `agy`'s own five-minute print-mode default so long turns are not
cut short. Override it per run, e.g. `--antigravity-print-timeout-seconds
1800`.

## Architecture

The canonical [architecture overview](../ARCHITECTURE.md) describes component
ownership, lifecycle and CI flows, GitHub/local persistence, trust boundaries,
and the CLI/skill distinction. This guide remains the detailed reference for
commands, protocol schemas, and recovery contracts.

The tool shells out to authenticated agent and GitHub CLIs. Durable round,
plan, handoff, and CI metadata is recorded on GitHub; response files, subprocess
logs, salvage, and advisory memory are local. The orchestrator validates agent
responses, owns bounded format repair, and revalidates live state before
qualification or merge. See [state and recovery](../ARCHITECTURE.md#state-and-recovery).

### Architecture context

The default architecture context discovers `ARCHITECTURE.md`, or the path
specified by `--architecture-path`. Lookup is literal and repository-local:
the implementation reads a regular blob from committed Git objects, enforces
size and UTF-8/binary limits, and never follows worktree symlinks or remote
document links. `--no-architecture-context` keeps legacy prompt behavior.
Per-snapshot and aggregate caps bound the overview; omissions and truncations
are labeled and do not claim complete coverage. PR reviews receive separate
immutable base and candidate snapshots; candidate edits are proposals and
cannot replace an unavailable or deleted baseline. Architecture text is
advisory and untrusted, not a permission channel or correctness waiver.
Use `--architecture-read-size`, `--architecture-snapshot-max-chars`, and
`--architecture-aggregate-max-chars` to tune bounded reads and rendering.
`--managed-context-max-chars` applies the active architecture prompt budget;
an explicitly restrictive value fails before invocation if protected context
cannot fit. The default expands as needed for the bounded architecture block.

## Agent Backends

Currently supported local agent CLIs:

- Claude Code via `claude`
- OpenAI Codex CLI via `codex`
- Gemini CLI via `gemini` (best-effort support for users whose organization or API-key setup still has access)
- Antigravity CLI via `agy` (first-class backend; also the Gemini CLI migration path)

### Startup replacement evidence and replay

The Gemini and Antigravity backends classify executable replacement only from
attempt-local evidence. The launcher identity must change between the pre-spawn
and post-exit observations, the exit must be an integer, capture must not report
an interruption, timeout, or PTY read failure, and no non-empty public response
file, public-response marker, structured payload, or ordinary narration may be
present. Gemini's residual output is limited to empty output and recognized
startup, Node-loader, or updater diagnostics. Antigravity additionally permits
recognized `agy` version/model startup chrome; unknown lines are treated as
progress and suppress replay. A valid response-file artifact is accepted before
this gate, even for a failed or timed-out command; a malformed artifact follows
normal validation/repair.

Gemini takes a read-only `HEAD` and exact `git status --porcelain` snapshot
immediately before `run_with_log`, then lazily takes the after snapshot only for
an otherwise eligible candidate. Antigravity takes the before snapshot after
acquiring both the settings lock and the exclusive `GEMINI.md` lock, before
tool-owned injection. It invokes the backend, removes the injected prefix in a
`finally` path, parses the candidate, and takes the after snapshot before
releasing either lock. Reviewer and coder paths therefore cannot observe a peer's
injected prompt as worktree activity. Isolated `role=repair` / `run_repair`
invocations skip Git probes, replacement metadata, and replay eligibility. The
snapshots are read-only evidence gates and cannot guarantee that a backend had no
other external side effects.

When the gate succeeds, Gemini and Antigravity wait for one bounded executable
stability window and may make one fresh full-timeout replay using the
`executable-replacement-attempt2` log suffix. Gemini allows
`agent_max_retries + 2` total slots; Antigravity allows
`len(antigravity_models) + agent_max_retries + 1`. The dedicated replay is outside
ordinary retry accounting. Antigravity replays the same singleton model without
advancing `retries_remaining`, `model_index`, or `attempts`; only a later ordinary
failure may retry or fall back. A failed stability wait keeps ordinary retry and
fallback eligibility, while changed or unavailable snapshots retain diagnostic
refusal metadata without triggering stability waiting.

## Prerequisites

- `gh` is installed and authenticated for the target GitHub repository.
- `claude` is installed and authenticated if either side uses Claude.
- `codex` is installed and authenticated if either side uses Codex.
- `gemini` is installed and authenticated if either side uses Gemini. For individual Google accounts after the consumer cutoff, prefer `agy`; direct Gemini CLI support is best-effort for enterprise/API-key users who can provide logs when issues are not locally reproducible.
- Use separate clones or worktrees for each active agent to avoid local file conflicts. If you omit `--claude-dir`, `--codex-dir`, or `--gemini-dir` for an active agent, the tool uses a repo-scoped temporary checkout under `/tmp/coding-review-agent-loop/OWNER-REPO/{agent}/repo`.

## Usage

### Process-tree containment

Every standalone CLI agent invocation and every `agent-loop run-tests` gate is
admitted through the same per-user aggregate policy. The default `auto` mode
preflights a systemd 253+ user manager with unified cgroup v2 and, when all
requested properties are accepted, launches a unique foreground scope under
`agent-loop.slice`. It uses the exact form
`systemd-run --user --scope --quiet` and intentionally omits `--wait`,
`--service`, and `--pipe`. `MemoryHigh`, `MemoryMax`, `MemorySwapMax`, and
`TasksMax` are applied to the aggregate and to the `coder`, `reviewer`,
`repair`, or `test-gate` child profile. A child profile is never above the
effective aggregate, including when another process holds a stricter lease.

Use `containment-preflight` before a live run to see the resolved policy,
systemd version, scope probe result, and supported cgroup counters. Values may
be bytes (`512MiB`), percentages of physical memory (`70%`), or `max` where a
finite limit is not requested. The defaults reserve 25% of physical memory
for the OS and unrelated services. Independent agent-loop processes coordinate
through crash-recoverable runtime leases keyed by boot ID and PID start
identity, so one process cannot weaken another's live slice ceiling.

`auto` visibly falls back to process-group TERM/KILL on macOS, non-systemd
Linux, unavailable user managers, non-unified cgroups, or failed property
probes. That fallback guarantees deterministic termination but makes no memory
ceiling claim. `required` fails closed instead. Optional files such as
`memory.peak`, PSI, and swap counters are capability-tracked and rendered as
`not collected on this host` when absent; only previously observed evidence
that disappears unexpectedly is indeterminate. OOM, hard memory/swap, and
`TasksMax` termination is `resource-exhausted`, while `MemoryHigh` and PSI are
pressure diagnostics and do not by themselves claim to have killed a process.

When the Claude skill invokes its local test gate, the mode can also be selected
with `AGENT_LOOP_CONTAINMENT_MODE=auto|required|off` (default `auto`).

Each scope contains a target-start/target-exec-error shim. A missing shim
report means launcher or unit creation failure regardless of the numeric
systemd exit status. A target-start report binds numeric statuses, including 1
and 203, to the target. Timeouts retain the `returncode is None` wall-clock
convention, salvage a valid response file when available, terminate the full
scope, confirm emptiness, and only then permit a retry or replacement replay.

The managed test wrapper also locks a lane by canonical cwd, normalized argv,
and `AGENT_LOOP_INVOCATION_ID`, using a non-inherited advisory descriptor.
Same-lane duplicates are rejected before spawn with exit 125 and do not enter
runtime learning. Intentional parallel reviewer workers have distinct lanes.
An unwrapped command cannot be deduplicated from free-form logs when no
structured tool hook exists; it is still contained when it is a descendant of
an agent scope. In skill mode, the host's in-session Claude coder and arbitrary
descendants remain part of the host session and receive prompt guidance plus
the managed wrapper when used. The aggregate is per user manager, not a
cross-user host isolation boundary.

Parallel test workers are budgeted against these limits; see
[Parallel test-worker budget](#parallel-test-worker-budget).

### Parallel test-worker budget

Parallel test workers multiply memory, so every loop derives an explicit
test-worker budget from the containment that actually applies, exports it to
coder and repair agents as `AGENT_LOOP_TEST_WORKERS` (with the enforcement mode
in `AGENT_LOOP_TEST_WORKER_ENFORCEMENT`), and enforces it for pytest-xdist in
`agent-loop run-tests`.

**Derivation.** `workers = min(cpu_term, mem_term)`, never below 1.

- The CPU term uses `os.sched_getaffinity` where it exists and returns a
  non-empty set, then `os.cpu_count()`, then 1, so derivation never raises on
  macOS or other hosts without affinity support. It is further capped by
  `ceil(quota / period)` of every `cpu.max` quota in the process's own cgroup
  ancestry. `CPUWeight` is proportional sharing, not a cap, and is not used.
- The memory ceiling is the minimum of a candidate set: usable host memory
  (physical memory minus `--containment-os-headroom-percent`, always a
  candidate when readable), the admitted child and aggregate `MemoryHigh`
  (else `MemoryMax`) when the backend is the managed `systemd-cgroup-v2`
  scope, and every `memory.high`/`memory.max` in the orchestrator's own cgroup
  ancestry (for example an operator's outer
  `systemd-run --user --scope -p MemoryHigh=9G`). A limit above physical
  memory therefore never raises the ceiling.
  `mem_term = floor((ceiling - 1 GiB reserve) / per-worker estimate)`, with a
  1 GiB default per-worker estimate (`--test-worker-memory SIZE`).
- With containment `off`, a process-group fallback, macOS, or non-systemd
  hosts, the finite default policy values are advisory and ignored. If no
  memory candidate is readable, the budget is `max(1, cpu_term // 2)` with
  limiting factor `memory-unknown`. Preflight and prompts say
  `no agent-loop memory ceiling enforced` unless a managed or ancestry limit
  exists.
- Only `/proc/self/cgroup`, host memory, and `cpu.max`, `memory.high` and
  `memory.max` in the process's own cgroup ancestry are read; missing or racing
  files are skipped.

**Per-path sizing and timing.** Coder and repair budgets are derived *after*
containment admission from the admitted handle, including a stricter live
aggregate lease held by another agent-loop process, and the same value is
exported to the agent and handed to its test broker. Broker tests attach to
the requesting coder/repair scope, so they use that budget. A broker-unavailable
`run-tests` inside a coder/repair scope inherits the exported value and may only
lower it (child flags, ancestry limits). Only a genuinely standalone managed
`run-tests` (no invocation ID) sizes from the admitted `test-gate` limits: it
admits the scope first and binds the launcher to the effective command under
the same lease. The prompt shows a preliminary pre-admission estimate; the
launch-time environment value is authoritative. `containment-preflight` prints
a preliminary budget with its source, limiting factor, backend and whether a
ceiling is enforced.

**Precedence.** In loop flows (`issue`, `pr`, ...) and standalone `run-tests`,
`--test-workers N` replaces the derivation (raising or lowering it) and
`--test-worker-enforcement {clamp,refuse,off}` sets the mode (default `clamp`).
A `run-tests` with a parent (an inherited `AGENT_LOOP_TEST_WORKERS` or a broker)
may only lower the budget and only tighten the mode (`off < clamp < refuse`);
unparseable inherited values fail closed to 1 worker. The three flags are also
accepted by the managed-command parser in both `--flag value` and
`--flag=value` forms, so flagged wrappers keep workdir validation, evidence
extraction and normalization.

**Enforcement.** In `clamp` and `refuse` the wrapper injects a stdlib-only,
in-process pytest plugin (`_agent_loop_worker_cap`, shipped as package data)
through `PYTHONPATH` and `PYTEST_PLUGINS`, plus a wrapper-private
`AGENT_LOOP_WORKER_CAP_SPEC`. A direct pytest command (after `env`, `timeout`,
`nice`, `stdbuf`, `nohup`, `time` or `command` prefixes) also gets
`-p _agent_loop_worker_cap` after its entry point; that later argv token
re-enables the plugin even if config or `PYTEST_ADDOPTS` disables it. Every
other command (make, scripts, tox, nox, npm, go, cargo, ...) keeps its argv
unchanged, including any `env -i`/`-u` prefix, and receives the same inert
environment; it is enforced whenever that environment reaches a pytest process
and is recorded `not-observed` otherwise (for example tox without `passenv` for
both variables, or an `env -i make test` prefix). `off` injects nothing. For a
direct pytest, inline `env` segments cannot remove or forge the controlled
variables: `-i`, `-u` and assignments are rewritten so the plugin, spec and (in
clamp) the auto cap survive. The plugin requires pytest >= 8 (new-style pluggy wrappers); on
older pluggy it defines no hooks and the run is reported unverified.

The plugin acts on pytest's own final resolved options, whatever supplied them
(any config file chosen by pytest's rootdir rules, `-c`, TOML or INI,
`-o addopts`, `PYTEST_ADDOPTS`, argv, or a repository
`pytest_xdist_auto_num_workers` hook):

1. A pre-yield `pytest_cmdline_main` wrapper runs before xdist resolves
   `-n`. Clamp lowers an over-budget integer (budget 1 becomes serial) and caps
   `--maxprocesses` for `auto`/`logical`. Refuse judges
   `min(value, maxprocesses)`, so `pytest -n 8 --maxprocesses=2` under budget 2
   runs.
2. A `pytest_xdist_auto_num_workers` wrapper takes the winning result
   (xdist's or the repository's) and returns `min(result, budget)` in clamp; a
   lower result is kept exactly. In clamp the wrapper also lowers the caller's
   `PYTEST_XDIST_AUTO_NUM_WORKERS`; refuse leaves it untouched so the plugin can
   refuse the real request.
3. A tryfirst `pytest_xdist_setupnodes` wrapper trims the expanded spec list
   (and `--tx`) to the budget, or refuses, after session-start and inner
   setupnodes changes. Every gateway type counts (`popen`, `socket`, `ssh`,
   ...); clamp keeps the first specs in order. An explicit `--tx` list needs
   `--dist`/`-d` to distribute at all. It then writes the pre-creation
   **decision** record: the enforced plan, not an observed count.
4. `pytest_xdist_newgateway` counts the gateways NodeManager actually creates.
5. A `pytest_runtestloop` wrapper writes the **confirmation** record before any
   test runs, with the observed gateway count (`effective`), remote gateways,
   and action `clamped`, `unchanged` or `exceeded`. Worker-restart gateways are
   not counted. A first-test **executed** marker is informational only.

Every refusal writes its refused record first and then raises a usage error,
so pytest exits 4 before any gateway or test exists. Report writes are best
effort: a write failure prints one stderr line and never changes pytest's exit
status. xdist worker processes never enforce or write. Each top-level
`pytest.main` in a process claims the spec for its own Config and reports under
its own session; configs nested inside an active claim (pytester,
`pytest.main` in a test) and child processes that inherit the environment are
never enforced and write only a best-effort nested marker.

The enforcement contract is deliberately narrow. A repository tryfirst
`pytest_xdist_setupnodes` wrapper registered after the plugin can add gateways
after the trim; the plugin cannot prevent that, reports `exceeded` with a
warning, and never labels such a run enforced. A repository can also unregister
the plugin. The plugin is a safety net against prompt slips, **not a sandbox**:
cgroup memory limits remain the hard boundary.

**Verification and evidence.** After a clamp/refuse run the wrapper reads the
report. Evidence is suppressed only when it proves no test ran: a direct pytest
whose single session wrote only a refused record (and no nested marker) exits
with the wrapper configuration status 2, records no runtime observation or
launcher-health row, and the broker returns `worker-budget-refused`. The only
pre-spawn refusal is an argv `-p no:_agent_loop_worker_cap` in refuse mode (in
clamp it is stripped with a notice). Every other report containing a refused
record, for example a script running several pytest sessions, keeps the
command's own exit status and evidence with caveat
`worker-budget-refused-session`, because a missing report line never proves that
no test ran. A direct pytest with no valid report is recorded with
`worker-budget-unverified` whatever its exit status, including 4. Other caveats
are `worker-budget-not-observed`, `-mixed`, `-partial-observation`, `-nested-session`,
`-remote-gateways`, `-exceeded` and `-descendants-terminated`.

**Broker boundary.** The parent-owned budget passed to the broker is
authoritative. The broker takes the stricter of the parent and the client's
requested values, overwrites the spec, control variables, plugin environment
and (in clamp) the auto cap in the spawned environment, and applies its own
effective mode to a plugin-disabling argv token. Results carry the executed
argv, the report-derived cohort and the enforcement status; the journal keeps
the requested argv plus a worker-budget caveat. The local fallback trusts the
inherited environment and is advisory by comparison.

**One command at a time.** In clamp and refuse the budget is a simultaneous
ceiling for the whole invocation. Only the process that spawns the target (the
broker handler or the local fallback) takes the command lane and a
non-blocking per-invocation worker-budget lock, never the `run-tests` client.
A second concurrent test command in the same invocation, including a nested
`run-tests` launched from inside a running test command, gets
`worker-budget-busy` (exit 125, nothing recorded). The lock lives in a
uid-owned directory (`/run/user/<uid>/agent-loop/worker-budget`, else
`/tmp/coding-review-agent-loop-<uid>/worker-budget`) that no caller environment
can choose; an unsafe directory fails closed. The command lane is keyed on the
requested command with worker-selection tokens removed and no budget or mode,
so every worker spelling of one command shares a lane in every mode. After the
target exits, its process group is terminated before the lock is released:
the group is always signalled with `killpg` (SIGTERM, then SIGKILL after a
short grace), independently of `/proc`, and the lock is released only once no
live member remains. If a member survives the bounded wait after SIGKILL, a
warning is printed and a small detached watcher inherits the lock and holds it
until the group is gone, so the next command of the invocation keeps getting
`worker-budget-busy` until then. Processes that escape into a new session (`setsid`, double fork) are outside
the ceiling and bounded only by cgroup limits. Off mode takes no worker-budget
lock and terminates nothing. Test suites that exercise `run-tests` itself must
give the inner call its own environment (clear `AGENT_LOOP_INVOCATION_ID` and
run from its own cwd), as this repository's `tests/conftest.py` does.

**Runtime cohorts.** `test-runtime.json` cohorts are keyed on
`(normalized command, environment fingerprint, workers)`. The normalized
command never includes the injected plugin token. In clamp and refuse the
`workers` label comes only from the plugin report: `serial` or a count needs a
direct pytest with exactly one confirmed session with no remote gateways, action
`clamped` or `unchanged`, and no nested marker. It describes the top-level
runner only: a child pytest that a test launches with a filtered environment is
unobservable test-body workload. Everything else, including every `other`
command, is `unknown`. Off-mode runs and legacy rows are `serial` only when the
direct-pytest argv ends xdist loading with `-p no:xdist`, and `unknown`
otherwise (including `-n 0`, `-n N` and `--tx`), because `--maxprocesses` from
any source or a repository `pytest_configure` can change the gateway count.
Unknown rows are retained but never feed serial or explicit-count
recommendations, so after upgrade most legacy commands restart timeout
learning. Use clamp or refuse with direct pytest for learned parallel timings.

**Concurrent loops and containment tiers.** Different invocations (for example
two loops on different repositories in separate capped user scopes) never
contend for the worker-budget lock; each sizes from its own scope's limits and
memory governs. Coder and repair budgets come from their own admitted child
profiles, reviewer turns get no budget, and the `test-gate` profile only sizes a
standalone managed `run-tests`.

### Local test evidence

Test commands are also subject to the
[parallel test-worker budget](#parallel-test-worker-budget).

Coder and repair turns start an invocation-local test broker when the managed
wrapper is available. The broker uses a fresh mode-0700 runtime directory and
an authenticated Unix socket. Requests use the `local-test-broker-v1` schema
with a bound turn ID, fresh nonce, argv, timeout, absolute cwd, and complete
environment snapshot. Frames are capped at 128 KiB; argv is capped at 256
entries/32 KiB total/8 KiB per entry, cwd at 4 KiB, and the environment at 256
entries/64 KiB total/128 bytes per name/8 KiB per value. Invalid UTF-8, NULs,
duplicate JSON keys, unknown fields, unsafe timeouts, bad capabilities, and
replayed nonces with different requests are rejected. Raw requests,
capabilities, and environments are not logged.

The client snapshots its actual cwd and environment after startup. The broker
removes exactly the transport control variables
`AGENT_LOOP_TEST_BROKER_ENDPOINT`, `AGENT_LOOP_TEST_BROKER_CAPABILITY`,
and `AGENT_LOOP_TEST_BROKER_PROTOCOL` before launching the target, then supplies the
authenticated `AGENT_LOOP_INVOCATION_ID`. All other variables, including
`PATH`, virtual-environment, locale, inline-assignment, and test variables,
remain comparison-bearing. Environment identity bytes are retained only in
trusted memory within one orchestrator process. A restart therefore renders
old observations `identity-unknown`; protected cross-process environment
comparison is intentionally not supported by this same-user implementation.

The broker pins the assigned checkout, opens a validated directory handle, and
uses handle-based `fchdir` confinement. Traversal, outside paths, final
symlinks, root replacement, and intermediate symlink escapes are context
failures. Safe subdirectories and intermediate symlinks that remain inside
the checkout are permitted. Managed process groups and cgroup handles are
registered to the requesting turn for timeout, interruption, and teardown;
containment-off mode still tears down the process group but makes no cgroup
claim. Target output is streamed live, only the evidence copy is bounded, and
the target exit code is preserved; wrapper statuses are reserved for
infrastructure or context failures.

If the broker channel is unavailable or rejects a request, `run-tests` reports
that telemetry is unverified and executes the command locally in the coder's
inherited scope. Both broker and fallback executions continue feeding the
bounded runtime-timing sidecar when `--memory-dir` is supplied.

Each observation records a typed outcome (`passed`, `failed`, `timed_out`,
`interrupted`, `incomplete`, or overlap rejection), provenance, scope, opaque
receipt/turn identities, and tracked-tree attribution. A later pass supersedes
a failure only when it is parent-observed and has the same normalized command,
scope, stable tracked tree, and live in-process environment. A passing subset,
superset mismatch, changed head, environment difference, unknown identity,
partial output, and unsupported or unverified baseline remain visible. Legacy
`tests_run` entries are parsed with shell-aware tokenization and labeled
self-reported/capture-limited; direct shell execution cannot claim complete
capture. Base reproduction requires a clean, stable checkout at the base
commit and never performs a dangerous checkout mutation or live-service test.

Durable metadata contains only bounded, sanitized projections: at most 32
detailed observations and 16 KiB, with unresolved failures retained longest.
Paths, credentials, DSNs, environment-assignment values, unknown positional
arguments, parametrized identifiers, ANSI/control data, and diagnostics are
redacted or hashed before persistence. Oversized or failed journal/sidecar
capture degrades to explicit incomplete-capture caveats. Evidence is advisory
history for coders and reviewers, including same-head and changed-head resume;
it cannot replace GitHub CI, override configured gates, or turn every
historical intermediate failure into a permanent blocker.

#### Semantic matrix claims and derived evidence

When an approved plan has an applicable risk matrix, implementation and
follow-up responses submit only bounded `risk_test_matrix_claims`. Each claim
uses an approved `row_id`, one or more opaque `execution_ref` selectors from
the current broker turn, actual test identifiers and locations, semantic
workflow/outcome/forbidden-effect assertions, and optional caveats. Selectors
are unique to the live invocation and repeated commands receive different
selectors; they are not receipt IDs and are never durable authority.

After the issue PR is authenticated at its exact pushed head, or after a
follow-up head is fetched and reconciled, the orchestrator's shared builder
derives the canonical `risk_test_matrix_evidence`. It supplies the approved
matrix identity and rows, deterministic ordering, receipt citations, and
statuses from the closed selector catalog plus the complete authoritative
journal. Only an authoritative passing observation bound to the current
invocation and exact eventual head/tree can verify a row. Failed, timed-out,
stale, cross-turn, wrong-tree, unbound, and restart-limited observations stay
uncited with explicit non-verified caveats; a passing subset cannot erase an
unsuperseded broader failure. Every enforceable row is retained, while
not-applicable rows create no coder obligation.

The derived object and bounded diagnostics are carried through public comments,
review context, round metadata, retry, and idempotent replay. Execution
selectors are not persisted as receipt authority. Missing or malformed claims
trigger bounded semantic correction that preserves the committed work and
authenticated PR; it never asks repair to recreate rows, identities, receipts,
mappings, statuses, or an evidence envelope. If correction is exhausted or a
head race occurs, the same PR remains resumable with complete non-verified
evidence. Planned tests and unverified command text remain non-evidence.

The claim-row keys come from one machine-owned schema: `row_id`,
`execution_refs`, the five semantic facts (`test_identifiers`,
`test_locations`, `workflow_path_claim`, `outcome_assertions`,
`forbidden_effect_assertions`), and optional `caveats`. The fresh
implementation prompt, the follow-up prompt, the post-authentication correction
prompt, and repair all show the same key list and complete example row. Before
authentication, a claim whose semantic facts are missing, `null`, or empty
(including a blank `workflow_path_claim`) is accepted rather than rejecting the
whole envelope, so an otherwise valid pushed PR still reaches authentication and
handoff. After authentication such a row is recorded as unverified with empty
facts (never the approved row's planned text), carries an
`incomplete-semantic-claim` diagnostic naming the missing fields, and triggers
the single bounded correction continuation. A correction that is still
incomplete leaves the row unverified; there is no second correction.

Each fact list is bounded to twelve items. A longer list is a claim defect
rather than an envelope defect (#913): the first twelve entries are retained,
the claim gains one bounded caveat naming every truncated field and how many
items it listed, and after authentication the row is unverified with a
`truncated-semantic-claim` diagnostic naming those fields, which triggers the
same single bounded correction continuation. The shared claim schema text
states the bound on every producer surface, so coverage that does not fit in
one row should be split across additional approved rows.

`execution_refs` must be the opaque selectors printed by `agent-loop run-tests`
in the same turn. Tests run directly, outside the wrapper, belong in `tests_run`
only and cannot verify a row; with no selector, omit the claim rather than
inventing one. Refs that cannot resolve to a current-turn handle are a claim
defect, not an envelope defect (#859). Command strings, cross-turn or restored
handles, and refs between 1,024 and 16,384 bytes are dropped before
authentication. The claim keeps its facts and gains one bounded caveat naming
the dropped refs (sanitized and truncated). After authentication each dropped
ref yields an `unknown-execution-ref` diagnostic, the row is unverified with no
citations, and the same single correction continuation runs. The same dropped
command string may appear in several rows.

One admissible selector may also appear in several rows (#865): a single
wrapper run commonly executes the tests for more than one row, and each row is
still verified only on its own semantic facts and the shared passing receipt.
Within one row a selector may be listed at most once.

These cases still reject the envelope, because they forge or corrupt execution
authority, are unbounded input, or are owned elsewhere:

- unknown keys, and ill-typed or oversize facts;
- a missing, invalid, unapproved, or duplicate `row_id`;
- a missing `execution_refs` key or an empty list (#855);
- more than eight refs, non-string or blank refs, or a ref over the
  16,384-byte hard cap;
- an admissible selector listed twice within one row, or a colliding catalog;
- an in-catalog selector that is not a passing parent-observed observation, or
  whose supplied launch-integrity state is failing or unknown (wrapper
  bootstrap, inner exec, and suite start must all be authoritative). It is a
  real broker handle, so selecting it is an authority decision, not a format
  defect;
- legacy canonical evidence fields in a fresh response.

Broader handoff atomicity for other envelope failures after a PR is pushed is
owned by #827 and #828.

Fix a GitHub issue:

```bash
agent-loop issue 56 --repo OWNER/REPO
```

Issue mode includes the issue title, body, and comments in the coder prompt and
issue-origin review prompts. Comments are ordered oldest to newest so later
discussion can refine or supersede the original body.

The issue implementation coder uses the structured `issue_implementation`
contract. It reports a non-blank summary, a positive `pr_number` or `null`, an
exact signed-human-requirement disposition ledger, and optional `tests_run`
command strings. The orchestrator validates supplied commands inside the
assigned checkout. A null-PR result is posted as a readable issue-level
terminal comment and stops before PR operations. If a signed requirement is
blocked after a PR was opened, the coder must retain the real PR number or URL
in the summary or disposition evidence while setting `pr_number` to `null`;
the conflict is posted once without retrying or entering handoff, review, or
merge gates. Accepted created-PR results are rendered for GitHub and retain the
raw structured payload in round metadata so resume can restore the typed result.

Run issue mode as plan-first discussion before implementation:

```bash
agent-loop issue 56 --repo OWNER/REPO --plan-first
```

With `--plan-first`, the coder writes an implementation plan without editing
code, pushing a branch, or opening a PR. Reviewers critique that plan on the
issue using `AGENT_PLAN_STATE` markers until every reviewer approves in the
same planning round. Plan reviews use explicit sections:

```md
### Blocking plan issues
### Same-plan follow-ups
### Future follow-ups
```

### Persisted planning-validation diagnostics

An exhausted deterministic validator failure is represented by a bounded
planning-validation audit record, not by rewriting the rejected plan into
canonical state. The immutable record carries only pre-post values: issue and
repository, planning generation, target coder round, prior plan subject when
revising, contract versions, expected producer login and immutable ID, failure
attempt, candidate digest, category, and a sanitized diagnostic. The diagnostic
is capped at 4096 characters, including the truncation suffix, and excludes
candidate bodies, provider output, credentials, environment data, and active
reserved syntax.

The issue API response and live recovery provide a separate transport wrapper
with the numeric comment ID, authoritative creation time, exact live body, and
producer identity. The writer verifies those fields against one invocation-
scoped authenticated actor before accepting the record. On a later run, each
wrapper is checked again; same-context records select the unique highest
payload attempt, never the newest timestamp or comment. Duplicate transport
objects are deduplicated, conflicting highest attempts fail closed, and stale
or actor-mismatched records cannot constrain planning.

The selected diagnostic is rendered as a clearly labeled trusted orchestration
correction block in fresh, full-revision, compact-revision, and parallel-review
planner prompts. It is not issue prose, reviewer feedback, or a signed human
requirement, and validation remains authoritative. A verified canonical coder
plan comment semantically supersedes matching diagnostics; history is never
deleted or edited. A failed diagnostic write preserves the original validator
cause and adds only a bounded not-persisted note. Non-deterministic provider,
timeout, quota, marker-safety, and containment failures do not create this
record.

If earlier blocking or same-plan items are still open, reviewers encode prior
item dispositions in the JSON `prior_plan_item_dispositions` array using
`"resolved"`, `"blocking"`, `"same-plan"`, or `"future"` (with a `"note"`).
The orchestrator renders a `### Prior unresolved plan item dispositions` section
in the public GitHub comment; reviewers do not add that section themselves. Use
`"same-plan"` for required current-plan refinements; `"future"` is accepted only
in approved plan reviews and the approved future follow-ups are reconciled with
the final approved plan instead of reopening planning. In `--approved-followups=issue` and `fix-and-issue` modes,
when implementation will continue after approval, plan-stage future follow-ups
are filed as separate issues before implementation starts. If implementation
continues but issue filing is disabled, they are summarized inline with a note
that they are not carried into PR review. Planning `item-*` IDs visible in issue
history are not PR prior review items unless they appear in the active PR
unresolved-item ledger. By default the loop posts an approved consensus summary
to the issue and stops without filing follow-up issues. Add
`--implement-after-approval` to continue into the normal
implementation and PR review loop using the approved plan:

```bash
agent-loop issue 56 --repo OWNER/REPO --plan-first --implement-after-approval
```

For larger plans, choose the post-approval behavior explicitly with
`--plan-execution-mode`:

```bash
agent-loop issue 56 --repo OWNER/REPO --plan-first --plan-execution-mode plan-only
agent-loop issue 56 --repo OWNER/REPO --plan-first --plan-execution-mode decompose-only
agent-loop issue 56 --repo OWNER/REPO --plan-first --plan-execution-mode implement-one-shot
agent-loop issue 56 --repo OWNER/REPO --plan-first --plan-execution-mode implement-by-phase
agent-loop issue 56 --repo OWNER/REPO --plan-first --plan-execution-mode auto
```

The modes are:

- `plan-only`: post the approved plan summary and stop. This is the default.
- `decompose-only`: use typed `child_stages` directly when present; otherwise
  ask the coder to decompose the approved plan. Create one child issue per
  selected phase, post a parent summary, and stop. Typed stages are the
  approved plan's remainder; its primary scope remains owned by the parent.
- `implement-one-shot`: keep the existing post-approval implementation handoff.
  This is also what `--implement-after-approval` selects for compatibility.
- `implement-by-phase`: create/link every phase issue, then implement the
  *current* phase - the first phase that is not yet complete - and stop after
  that PR review loop. A parent rerun advances: once a phase's child issue is
  closed and its canonical implementation PR is merged, the next rerun
  dispatches the following phase with its own `phase_index` and handoff record.
  While a phase's child is still open the rerun names it and prints the exact
  command to resume it. Once every phase is complete the rerun reports a
  terminal state instead - one line per stage, plus any retained-parent and
  final-integration obligations - and dispatches nothing. Completion evidence
  is authenticated: a closed child whose PR evidence is missing, unreadable or
  unmerged, an open child whose canonical PR is merged or closed, and a handoff
  recorded for a later phase while an earlier one is incomplete all stop the run
  with a diagnostic rather than skipping or re-dispatching a phase. A
  `human-action` or `manual-close` phase completes when its child issue is
  closed: that closure is the operator's attestation that the required work and
  remark are done. Older decomposition summaries without a handoff marker are
  treated as not yet handed off, so the first child handoff is recorded once.
- `auto`: after approval, resolve a fresh reviewed recommendation to
  `implement-one-shot` or `implement-by-phase`. A legacy-undecided plan is
  refused, and the CLI rejects `--materialize-split-issues` and `--split-stage`
  because neither topology is known before approval. With `--dry-run`, the
  resolved action and normalized staged topology are previewed without a
  decision record, child issue, handoff, coder, PR, or follow-up mutation.

`decompose-only` and `implement-by-phase` already select one detailed child
topology. The CLI rejects `--materialize-split-issues` with either mode before
any GitHub write, and rejects both split flags with `auto` before planning.

At the approval boundary, routing follows this matrix:

| Requested policy | Fresh one-shot recommendation | Fresh staged recommendation | Legacy-undecided plan |
| --- | --- | --- | --- |
| `plan-only` | Stop without execution | Stop without execution | Stop without execution |
| `implement-one-shot` | Implement one shot | Reject before mutation | Use the historical explicit path only |
| `decompose-only` | Reject before mutation | Decompose and stop | Use the historical explicit path only |
| `implement-by-phase` | Reject before mutation | Create topology and dispatch the current (first incomplete) phase | Use the historical explicit path only |
| `auto` | Resolve to `implement-one-shot` | Resolve to `implement-by-phase` | Stop and request a reviewed revision |

For a non-dry run, read-only recovery and expected-closing validation complete
before the durable execution decision is posted. Follow-ups, split or phase
children, handoffs, coder dispatch, and PR work occur only after that decision.
The same identity checks apply when resuming a decision, partial topology,
handoff, or PR association; a mismatch fails closed. `--dry-run` resolves and
reports `auto` using the same checks, but persists no decision and performs no
approval-bound mutation.

Before invoking a coder for an issue — in direct `agent-loop issue <n>` mode or
approved-plan implementation alike — the orchestrator resolves the canonical
`AGENT_ISSUE_PR_HANDOFF` record: an authoritative, machine-readable comment
posted once an implementation PR passes validation, recording the PR number,
URL, head SHA, flow (`issue-implementation` or `approved-plan-implementation`),
and (for plan-first runs) the approved-plan hash. If a valid record names a
still-open PR, the rerun resumes PR review on that PR directly instead of
invoking the coder again. A record whose PR is closed, merged, or otherwise
unresolvable fails safely with an actionable message rather than falling back
to a fresh implementation; fix or select the correct PR and rerun
`agent-loop pr <number>` directly.

Approved-plan propagation is a separate PR-bound channel. The implementation
handoff's plan hash selects the canonical raw plan record (including its
subject, source locator, current scope, and deferred work); it is never
reconstructed from bounded issue history or compact PR context. Full and
compact reviewer prompts, merge-conflict prompts, and coder follow-ups receive
the same plan context. A missing, ambiguous, or mismatched record stops with a
diagnostic instead of silently reviewing stale prose. Direct `agent-loop pr`
resume uses the PR contract's primary issue and requires the matching
issue-side handoff; ordinary PRs without planning provenance remain supported.

Approved-plan reconciliation is explicit in both full and compact PR review
prompts. Reviewers classify a concern as implementation noncompliance or an
ordinary defect to fix within the plan, an evidence-backed correctness,
security, compatibility, or test defect in an approved decision that may block
and proposes a plan correction, or a discretionary scope/policy request that
names the incompatible approved decision, evidence, and proposed change. The
existing finding text and carried-item disposition notes carry the explanation;
no protocol fields, model call, semantic classifier, or automatic arbitration
is added. Plan conformance never defeats a legitimate defect, signed human
instruction, original issue authority, or safety constraint.

Coder follow-ups use `disputed_items` and `dispute_evidence` for a factually
incorrect claim or a verified plan conflict, with decision-bearing evidence for
the latter. They must fix ordinary defects and evidence-backed defects, and may
not park a verified plan conflict in `remaining_items`. The existing behavior
re-reviews a disputed item once and terminates visibly for human resolution if
the reviewer maintains it as blocking. When canonical plan text is omitted for
provider budget, the source locator must be fetched and verified before a plan
decision is enforced or challenged; if that fails, no approved decision is
enforceable or disputable, but ordinary defects remain reviewable. Unavailable,
mismatched, legacy, and direct-PR contexts must not cause a decision to be
invented. Host skill review requests include the same reconciliation guidance
artifact as external reviewer prompts.

For staged work, child and authoritative parent issue contexts are labeled
separately. Resume validates the generated split/decomposition identity and
the parent handoff selecting the exact child before recovering the plan. Parent
and child signed instructions are merged by chronological precedence under
stable IDs; planning `item-*` records and plan future-work entries never enter
the PR unresolved-item ledger.

### Durable marker trust boundary

All durable protocol records are registered in
`src/coding_review_agent_loop/protocol_markers.py`. The registry owns each
record's outer grammar, canonical codec, safe historical label, strictness, and
allowed GitHub surfaces. Trusted producers compose immutable `TrustedBody`
segments, and writers verify a one-to-one match between every visible
occurrence and its authorized canonical segment before posting.

The workflow transaction record (#827) is registered as a PR-comment-only
record. Nothing reads or writes it yet; the visible effect today is that
look-alike text in untrusted prose or agent output is neutralized like any
other reserved record.

Naming a reserved record in issue or pull-request prose is safe. Untrusted
GitHub text — an issue title or body, a pull-request title or body, a comment,
or the body of a signed human requirement, which is itself an ordinary
comment — that merely names a token is rendered into prompts with every
reserved name replaced by the registry's stable descriptive label, for every
strictness class. The run logs once which surface named it. This matters
because the cases where naming a record is legitimate are exactly the ones
where the work concerns the protocol. A name in untrusted text never carries
authority, is never parsed as a record, and cannot satisfy provenance: prompt
text is prose for a model, not an input to any record parser.

Rejection is deliberately narrower than neutralization, and applies at the
surfaces that actually gate authority:

- The **pull-request body** is checked before an unauthenticated resumption or
  handoff is accepted. A span there that claims the record grammar — shaped
  like the record rather than naming it — is rejected as a forgery attempt,
  and the diagnostic names the surface that carried it and what to do.
- A **current untrusted agent response** that emits a complete, parseable
  record is rejected the same way.
- A **tool-owned publication** is validated against its authorized canonical
  segments, so an unexpected token in a body the tool is about to publish
  fails closed.

Issue bodies and ordinary issue or pull-request comments are *not* rejected
for containing record-shaped text; they are neutralized for prompts. Authority
on those surfaces comes only from the typed record parsers, which apply their
own canonical-encoding, scope, and surface checks before a record is believed,
and which never read the rendered prompt text.

The loop distinguishes current prose from orchestrator-re-rendered history.
When a prior ledger item is projected into a later public comment, only
reserved spans are replaced with stable descriptive labels before the item is
normalized or truncated. Item IDs, reviewer, round, source status,
disposition, and disposition notes remain separate fields. The same
transformation is used during initial rendering and resume reconstruction.

Encoded or compressed payload values such as salvage patches and round-state
copies are opaque at their outer occurrence: codec, size, integrity,
decompression, secret/binary, and hydration checks still apply. If such a
value is later projected into visible current prose it goes through current
marker validation; if it becomes historical ledger prose it goes through the
deterministic sanitizer.

Surface policy is fail-closed. Issue comments carry issue-flow records, PR
comments carry PR contracts and managed-CI audits, child issue bodies carry
only the split-child record, and managed PR bodies carry only the managed
origin record plus the invocation-correlated override trailer. Missing,
duplicate, malformed, non-canonical, wrong-surface, stale, or uncorrelated
records are never inferred from finished-body text or payload equality.

Issues that predate this marker (or crashed before it could be posted) fall
back to the legacy check: searching GitHub directly for an already-open PR
that already references the target issue, independent of any handoff marker.
This closes the same crash window as before: if implementation created a PR
but the run aborted afterward — before any handoff comment could be posted —
a rerun still resumes PR review on that PR instead of invoking the coder again
and creating a duplicate, and backfills a canonical handoff record so later
reruns take the fast path. If more than one open PR references the issue, the
orchestrator raises an error instead of guessing which one to resume; close or
merge the extra PR and rerun `agent-loop pr <number>` directly.

`--implement-after-approval` is a compatibility alias that requests
`--plan-execution-mode implement-one-shot`. It requires `--plan-first`; fresh
approval-bound compatibility is resolved against the reviewed recommendation,
so a staged recommendation stops actionably before mutation rather than being
silently routed as one-shot. If `--plan-execution-mode` is supplied as well, the
alias is accepted only with `implement-one-shot`; combining it with
`plan-only`, `decompose-only`, or `implement-by-phase` is rejected before
recommendation recovery or any approval-bound write.

Approved-plan implementation can switch to a different coder after planning:

```bash
agent-loop issue 56 --repo OWNER/REPO --plan-first --implement-after-approval \
  --coder claude \
  --implementation-coder codex \
  --implementation-coder-model gpt-5.5 \
  --implementation-codex-reasoning-effort high
```

Planning and plan revisions always use `--coder`; the `--implementation-*`
flags apply only after reviewers approve the plan and the run enters
implementation. `--implementation-coder` accepts the same values as `--coder`.
`--implementation-coder-model` sets that implementation coder's model for
the approved-plan implementation and subsequent PR follow-ups. When `--implementation-coder-model` is set
without `--implementation-coder`, implementation keeps using `--coder` but with
the implementation model. `--implementation-codex-reasoning-effort` can
only be used when the implementation coder is Codex, either explicitly via
`--implementation-coder codex` or implicitly via `--coder codex`. The matching
`--implementation-claude-effort` option is restricted to Claude. An effort
override does not require an explicit model: an observed model may be unknown,
and the signature will report `unknown model` with the selected effort.

Codex and Claude effort are resolved independently for the provider executing
each turn. The precedence is the matching reviewer or implementation role override, provider-wide
option (`--codex-reasoning-effort` or `--claude-effort`), then agent-loop's
explicit `medium` default. The default is passed to the provider CLI rather
than inherited from local CLI configuration. Codex accepts `minimal`, `low`,
`medium`, `high`, and `xhigh`; Claude accepts `low`, `medium`, `high`, `xhigh`,
and `max`. Use `xhigh` explicitly when a demanding Codex/Luna implementation
run needs more effort. A Claude invocation receives both `--effort` and the
matching `CLAUDE_CODE_EFFORT_LEVEL` environment value. If an installed Claude
CLI rejects the explicit flag, agent-loop reports a non-retryable tooling
diagnostic and never retries without the requested effort. Antigravity model
tiers remain embedded in model selection and are not replaced with `medium`.

#### Independent Reviewer Selection

Codex and Claude can use different models and efforts as coder and reviewer,
in both `issue` and `pr` commands. For example, plan with Sol, implement and
address PR feedback with Luna/xhigh, and review with Sol/medium:

```bash
agent-loop issue 123 --repo OWNER/REPO \
  --coder codex --reviewer claude --reviewer codex \
  --plan-first --implement-after-approval \
  --codex-model gpt-5.6-sol --codex-reasoning-effort medium \
  --implementation-coder-model gpt-5.6-luna \
  --implementation-codex-reasoning-effort xhigh \
  --reviewer-codex-model gpt-5.6-sol \
  --reviewer-codex-reasoning-effort medium
```

Use `--reviewer-claude-model` and `--reviewer-claude-effort` for the equivalent
Claude reviewer overrides. Models and efforts are selected independently:

- Reviewer model: reviewer option, then the current provider-wide model.
- Reviewer effort: reviewer option, provider-wide effort, then `medium`.
- Coder: existing provider-wide settings, with implementation overrides activated
  after plan approval. Reviewer options never change the coder's selection.

Reviewer options also apply to discussion participants, but not the discussion
analyzer or repair backend. They do not change workdir or permission policy.
Without reviewer overrides, behavior is unchanged: an implementation model
switch also becomes that provider's default reviewer model during the handoff.
Effort-only overrides do not switch models, and model-only overrides do not
reset efforts. Raw provider model/effort arguments cannot override these options.

The configuration survives an in-process issue-to-PR handoff. A separate `pr`
resume command must repeat the desired settings (see the README PR example);
the tool does not reconstruct them from previous signatures. In `pr` mode,
use the normal provider-wide options for the coder, not issue-only
`--implementation-*` flags.

If the plan narrows scope (via `deferred_stages` or a prior discuss `split`
consensus), see [Split issue materialization](#split-issue-materialization)
below for how those follow-up stages are filed (or warned about) and how
`implement-one-shot` targets the correct stage instead of the whole parent. If
you are deciding between that mechanism and `decompose-only` /
`implement-by-phase`, read [Phased decomposition versus split
materialization](#phased-decomposition-versus-split-materialization) first.

Generated child issues are self-contained: each body includes the parent issue
link, the relevant approved parent-plan slice, constraints/invariants,
dependency notes, scope and non-goals, rollout risk, validation/soak
requirements, automation classification, and explicit instructions for either
`agent-loop issue <N>` execution or human remark/closure. Dependency links are
filled in after earlier child issue numbers or URLs are known.

Automation classification is required for every phase. `agent-pr` means the
phase is expected to be implemented through a child issue and PR.
`human-action` and `manual-close` phases are still created as child issues, but
their titles, bodies, and parent summary call out that a human must perform the
work or checkpoint, add the required remark/update, and close the issue. If
`implement-by-phase` selects a human-owned phase whose child issue is still
open, it stops instead of recording an implementation handoff; closing that
child is the attestation that lets a later rerun advance to the next phase.

Flat child topology allows 15 children by default; override it with
`--flat-child-limit`. The count is shared by decomposition and split
materialization, preflighted before a checkpoint or create, and never silently
truncated. An over-limit result creates no children and returns a structured
human decision to consolidate or use the hierarchical design tracked in #720.
This limit is independent of `--approved-followups`, whose issue mode still
caps approved-review future follow-up issues separately. Decomposition also rejects
duplicate phase titles, invalid automation classes, unknown dependencies,
self-dependencies, and forward dependencies; `depends_on` may reference only
earlier phase titles. Parent decomposition metadata
(`AGENT_PLAN_DECOMPOSITION`) and phase handoff metadata
(`AGENT_PLAN_PHASE_IMPLEMENTATION`) make reruns idempotent. The decomposition
metadata payload is `v1_`-prefixed zlib-compressed URL-safe base64, like the
topology checkpoint; summaries published in the earlier plain-base64 form
remain readable. While an
`implement-by-phase` child is still open, rerun that child issue directly
instead of expecting the parent to restart it; once it is closed with a merged
PR, rerun the parent to advance to the next phase.

Implement a free-form task:

```bash
agent-loop task "Add a /healthz endpoint that returns 200 OK." \
  --repo OWNER/REPO
```

Review an existing PR:

```bash
agent-loop pr 123 --repo OWNER/REPO
```

Evaluate a GitHub issue without writing any code:

```bash
agent-loop discuss 123 --repo OWNER/REPO
```

### Open-ended answer results

The legacy implementation-triage contract remains the default. For system and
design questions, use `--discuss-result-mode answer`:

```bash
agent-loop discuss 123 --repo OWNER/REPO \
  --reviewer codex --reviewer claude \
  --discuss-result-mode answer
```

Debaters return `kind: "discuss_answer"` with `position: "answer"` or
`"needs-human"`, plus `answer`, `rationale`, `confidence`, and
`unresolved_items`. Each item has exactly a non-empty `text` and a `status` of
`blocker`, `human-decision`, or `follow-up`. `position: "answer"` requires an
answer and may include any classification; for example, an otherwise useful
answer with a pricing `blocker` is valid but cannot conclude successfully.
`position: "needs-human"` omits `answer` and requires at least one
`human-decision` item.

The final complete round is authoritative: later rounds can clear or
reclassify earlier concerns. Final precedence is `human-decision` (Needs Human
Decision), then `blocker` (Deadlock), then answer convergence; follow-ups alone
remain non-blocking. Summaries list Blockers, Human decisions, and Non-blocking
follow-ups separately. Legacy persisted transcripts that use `open_questions`
are accepted only while resuming: old `needs-human` questions map to
`human-decision`, while old asserted-answer questions map conservatively to
`blocker`. New live responses must use `unresolved_items`.
Analyzer observations remain non-authoritative and cited research remains a
separate sourced-facts section. Repair and resume preserve the selected mode;
transcripts cannot mix answer and triage responses.

Answer-mode presentation is intentionally executive-first. When a configured
discuss analyzer returns the optional enriched agenda, each completed interim
comment leads with `Current consensus`, `Active disagreements`, `Changes this
round`, `Missing facts`, and `Next-round focus`; the final comment leads with
`Outcome`, `Agreed conclusions`, `Remaining disagreements`, and `Next action`.
The state is cumulative: a resolved item appears in the change record and does
not return to the active list unless a later response explicitly reopens it.
The raw table, analyzer audit, and bounded excerpts are collapsed below the
state, while each per-agent comment remains the complete authoritative audit.

Only the configured discuss analyzer can synthesize. A normal enriched agenda
adds no call, a valid legacy agenda permits one bounded same-analyzer fallback,
and exact-text, semantic-equivalent, or debater-confirmed final artifacts are
adapted directly without another synthesis call. Other complete or partial
final paths use at most one bounded final synthesis call; partial finalization
is always mechanically `material_deadlock`. The mechanical classification is
never changed by advisory text. Analyzer absence, malformed or unsupported
claims, failed fidelity checks, marker-like text, corrupt resume metadata, and
sidecar hydration problems fail closed to the existing rendering. Canonical
snapshots and visible synthesis text are bounded to 16,000 UTF-8 bytes; answer
excerpts are limited per responder and in aggregate, and large audit payloads
spill to transport sidecars. Resume consumes a stored snapshot only while a
discuss analyzer is configured, preserving analyzer gating when configuration
changes.

To enable semantic comparison for differently worded final answers, configure
an independent `--discuss-analyzer`. There is no reviewer fallback when it is
unset. The analyzer receives final-round answers only and is prohibited from
research or repository work; its output is advisory and recorded separately in
the summary. Equivalent answers may converge. Compatible answers receive one
bounded, budget-exempt confirmation phase: each original debater confirms the
canonical recommendation or supplies a refinement, and only exact normalized
agreement among those effective answers finalizes. Invalid comparator output,
comparison failure, failed confirmation, or material conflict safely remains a
deadlock. A successful debater-confirmed recommendation reuses the recorded
semantic comparison audit and does not invoke a second final-observations pass
over the same final answers; that advisory comparison remains distinct from the
authoritative debater confirmation. This makes resumed finalization idempotent
once a final summary is recorded.

Discuss mode sends the issue title, body, and comments to all configured
reviewers and asks each to return a `discuss_review` response with a single
outcome vote. Instead of collapsing a round into one orchestrator comment,
discuss mode posts a transcript to the issue, similar to how PR review mode
posts a comment per reviewer per round:

1. Each configured reviewer ("debater") posts its own issue comment with its
   structured vote and rationale, tagged with round metadata.
2. Once every reviewer has posted for the round, the orchestrator posts a
   separate round-summary comment identifying consensus or disagreement.
3. If the round is not the final one, the summary comment also lists the
   agenda for the next round (each reviewer's held position). Unresolved
   disagreement carries this agenda plus the prior round's comments into the
   next round's prompts, and requires each reviewer to include a non-empty
   `rebuttal` that engages the disagreement.
4. The final consensus/deadlock result is always its own round-summary
   comment, separate from the per-reviewer comments and from any interim
   round summaries.

A two-round debate transcript looks like:

```text
Round 1: Codex position
Round 1: Antigravity position
Round 1: Orchestrator summary (agenda for round 2)
Round 2: Codex rebuttal
Round 2: Antigravity rebuttal
Round 2: Orchestrator final consensus/deadlock
```

The four possible votes are:

| Vote | Meaning |
|------|---------|
| `implement` | Reviewer recommends proceeding with the issue as written. |
| `do-not-implement` | Reviewer recommends not implementing the issue. |
| `needs-human` | Reviewer cannot decide without more information from a human. |
| `split` | Reviewer recommends breaking the issue into smaller sub-issues and may include sub-issue proposals. |

After each round, the orchestrator checks for same-round unanimity. Agreement in
round 1 is marked `unanimous`; agreement after debate is marked `converged`.
By default, the orchestrator runs up to two debate rounds after round 1. Set
`--discuss-max-rounds 0` to post a human-needed deadlock immediately after an
initial disagreement, or increase the value to allow more debate.

If no consensus is reached after the configured debate rounds, the final
round-summary comment is a `deadlock` comment with the `needs-human` outcome,
each final reviewer position, and the core disagreement. `split` proposals
from multiple reviewers are merged in first-seen order only when all reviewers
in the same round agree on `split`. Discuss runs are idempotent and resumable:
the final round-summary comment includes an
`<!-- AGENT_DISCUSS_CONSENSUS: <subject-hash> -->` marker derived from the issue
title, body, and non-round comment bodies, and every posted comment carries an
`<!-- AGENT_LOOP_META: ... -->` marker the orchestrator decodes to reconstruct
completed rounds — including a round that only partially posted before a
crash — from the public comment thread on the next run, without relying on one
aggregate comment. A re-run on an unchanged issue with a final result posts no
second transcript; posting a new human comment on the issue invalidates the
cached result and triggers a fresh evaluation from round 1. If a resumed run's
next round would exceed a `--discuss-max-rounds` value that was lowered since
the prior run, the orchestrator immediately posts a final `deadlock` summary
from the last completed round instead of silently exiting without a result;
if no completed round exists to finalize from, it raises instead of exiting
silently.

`AGENT_LOOP_META` uses a `v1_` prefix followed by zlib-compressed URL-safe
base64 data (legacy plain-base64 markers remain readable). If metadata cannot
fit in one GitHub comment, the loop posts `AGENT_LOOP_SIDECAR` transport comments
before the metadata-bearing anchor. Sidecars carry machine-readable overflow
data for the following agent-loop comment; they are not independent agent
responses. Each sidecar starts with a deterministic visible label giving its
position and total, for example `Agent-loop review attachment 1/3
(machine-readable overflow: canonical_reviewer_response)`. The label says plan,
plan review, or review only when the round metadata's flow and role establish
it, and uses neutral `Agent-loop attachment` wording otherwise. The hidden
marker and its payload are unchanged, and older marker-only sidecars (which
GitHub renders as "No description provided.") remain valid. Keep those
sidecars with the anchor: resume fails loudly if one is missing or corrupt, at
which point restore the sidecars or remove the incomplete anchor and rerun.

Two review-state fields grow with use rather than with the change under
review: the accumulated review items (`prior_items`) and the canonical
risk/test-matrix evidence (`risk_test_matrix_evidence`). They are the last
fields considered for spilling, so they move into sidecars only when the anchor
would still overflow after every other spillable field has moved; a comment
that fits is posted byte-for-byte as before. Matrix evidence never spills
without the review items also spilling, and neither field ever spills in a
discuss round. Resume needs these sidecars like any other.

An older agent-loop binary never treats one of these spill references as data,
but it does not always fail with an error either:

- Resume and round-record extraction raise an invalid round-metadata error.
- The plan-candidate existence check skips the record as absent.
- Managed-CI continuity cannot see a merge-conflict obligation carried in
  spilled review items, so it declines a conflict-only head transition (one with
  no blocking-review record) instead of granting it. Transitions with a
  reviewer pair are unaffected.
- Discuss comments never carry these spills, so older discuss classification is
  unchanged.

Downgrading across such a comment is unsupported: upgrade the binary before
resuming. Before this change such a comment could not be posted at all.

The visible risk/test-matrix evidence section of coder comments is collapsed
under a `<details>` summary and, on PR coder follow-ups, shows only the rows
whose status, assertions, citations (ignoring receipt IDs) or caveats changed
since the previous coder round, followed by a line such as `21 rows unchanged
since round 26; full matrix in round 23.` The full row list is rendered in the
issue-implementation comment, in the first follow-up that carries evidence,
whenever the matrix identity or row set changes, and whenever the previous
round's evidence or its recorded full-matrix round cannot be used. That round is
kept in the optional `risk_test_matrix_evidence_full_round` round-metadata field;
records written before this change are treated as having rendered the full list
themselves. The canonical `risk_test_matrix_evidence` in `AGENT_LOOP_META` and
the matrix sidecar still carry every row, so resume and reviewer prompts are
unaffected.

Sidecars carry metadata only. When the visible text of a structured plan comment
(`plan_state` or `plan_revision`) would still push the comment over the
60,000-character budget, the loop posts a **compact plan digest** instead of the
full plan prose. This typically affects a separately planned child that must
reproduce many inherited risk-matrix rows verbatim. The digest is chosen only
for that size overflow and logs one line when selected; a plan that fits is
posted exactly as before, and any other transport error still aborts
publication.

- The digest starts with a "Compact plan digest" notice and shows a bounded
  summary: the plan summary, then plan steps, prior item dispositions, additional
  closing issues, deferred stages and structured scope categories, each cut off
  at a fixed share with a line such as `... 212 more of 300 omitted; complete
  list in the authenticated attachments`. These sections together never exceed
  12,000 characters.
- Still visible in the comment: the risk/test matrix and execution
  recommendation sections with their records (their payloads may be carried as
  authenticated references, as before), the signed-requirements acknowledgement,
  the plan-state footer, the signature and `AGENT_LOOP_META`. Signed requirement
  IDs are never omitted or count-summarized; only their evidence text is
  shortened, and the digest is revalidated against the surfaced IDs before it is
  posted.
- Only in authenticated metadata: the deferred-stage, typed-stage and
  expected-closing records, along with every omitted entry. The complete plan
  lives in `canonical_plan` and the assembled plan sidecar inside
  `AGENT_LOOP_META` and its attachments. Resume, reviewer prompts, plan hashes
  and child-plan validation read that complete plan, never the digest, so keep
  the attachments with the anchor.

Free-form (unstructured) plans cannot be compacted. If a comment still cannot
fit, nothing is posted and the error explains where the size is, for example:

```text
Round comment exceeds 60000 characters even after metadata spill; shorten the
visible response or metadata. Size attribution: visible body outside round
metadata 52120 characters; residual encoded round metadata 9544 characters;
largest unspilled metadata fields (encoded characters): architecture_impact=891, ...
```

When the round-metadata marker alone, with its framing and an empty visible
body, would exceed the budget, the message says so instead: `the derived round
metadata alone needs N characters, so shortening the visible response cannot
fix it.` The same size attribution follows. The attribution lists field names
and sizes only, never their content.

Discuss mode accepts `--reviewer` the same way as PR mode — repeat the flag to
require multiple reviewers:

```bash
agent-loop discuss 123 --repo OWNER/REPO \
  --reviewer codex --reviewer antigravity \
  --discuss-max-rounds 2
```

### Optional analyzer-guided debate agenda

Pass `--discuss-analyzer <agent>` (`claude`, `codex`, `gemini`, or
`antigravity`; it may coincide with a `--reviewer`) to add an analyzer agent
borrowed from the analyzer/debater pattern:

```bash
agent-loop discuss 123 --repo OWNER/REPO \
  --reviewer codex --reviewer antigravity \
  --discuss-analyzer claude
```

After each non-final round, the analyzer receives the complete multi-round
vote history (every completed round's outcomes, rationales, rebuttals, and any
framing corrections, oldest first) plus its own previous agenda, and returns a
structured `discuss_agenda` response:

```json
{
  "schema_version": 1,
  "kind": "discuss_agenda",
  "consensus": ["The issue is well-motivated."],
  "disagreements": [
    {
      "topic": "Scope of the change",
      "positions": {"Codex": "Narrow enough.", "Antigravity": "Too broad; split it."},
      "question_for_next_round": "Would splitting the API boundary resolve the scope objection?"
    }
  ],
  "missing_facts": ["Whether the API boundary is already specified."]
}
```

In analyzer mode, the next debate round's prompt is agenda-focused: it renders
only the structured agenda plus the target debater's own prior position
verbatim. Other debaters' full rationales and rebuttals are omitted and reach
each debater only through the analyzer's summarized `positions`. Each debater
must concede, defend with evidence, refine its position, or set
`analyzer_framing: "misframed"` with a `framing_note` correcting the agenda;
framing corrections are rendered in the debater's public comment.

Guardrails — the analyzer is never authoritative:

- Consensus detection stays vote-only; the analyzer never decides the outcome.
  An agenda claiming consensus while the votes differ is forwarded, but the
  votes rule and the divergence stays visible in the summary.
- The agenda is rendered in the non-final round summary ("Agenda for round
  N+1 (analyzer: ...)"), so it is auditable on the issue.
- After the final debater responses arrive, the analyzer gets a separate
  best-effort pass over only those successful final-round responses. A valid
  result is rendered as "Final analyzer observations (not debater-confirmed)"
  after the authoritative debater vote table. It is rejected if it names a
  non-debater or asserts a position, topic, consensus, disagreement, or missing
  fact unsupported by the final-round text. The exception is a successful
  answer-mode compatible-answer confirmation: its semantic comparison already
  analyzed those final answers, so the recorded comparison is reused instead
  of making a duplicate advisory pass. Debater confirmation is still the
  authoritative finalization step.
- If an agenda from the preceding non-final round exists, the final summary
  renders it separately as "Agenda before final round." This is explicitly
  historical and is never presented as current disagreements. If final analysis
  fails validation, the observations are omitted while the answer/vote table
  and mechanical outcome remain available. For a partial round, any retained
  observations are grounded only in its successful final-round responses.
- If the analyzer invocation fails even after the malformed-response repair
  pass, the orchestrator logs a warning and falls back to the plain mechanical
  agenda for that round (and full prior positions in the next debate prompt)
  instead of aborting the run.

The raw agenda rides in the round summary's `AGENT_LOOP_META` metadata, so a
resumed run restores the structured agenda for the next debate round. Legacy
summaries without an analyzer payload resume in plain mode. With
`--discuss-max-rounds 0`, there is no non-final agenda pass, but a configured
analyzer still receives the final-only pass after the initial final round.
Omitting `--discuss-analyzer` keeps plain #465-style direct deliberation
unchanged.

### Discuss research policy

`--discuss-research none|required|auto` (default: `none`) controls whether
debaters may use current external facts:

```bash
agent-loop discuss 123 --repo OWNER/REPO \
  --reviewer codex --reviewer antigravity \
  --discuss-analyzer claude \
  --discuss-research auto
```

For design issues, round-one research focuses on the decision under debate
(solution design, prior art, cost/latency, feasibility, or guardrails), not just
verification of an illustrative incident. Example validation is appropriate
when the example is disputed or outcome-critical. Active research records a
target and concrete questions. Allowed targets are `example-validation`,
`solution-design`, `cost-latency`, `implementation-feasibility`, and
`policy/legal/current-facts`.
New active responses and research-required agendas should include these fields;
older transcripts may omit them and remain resumable as legacy, unclassified
research.

- `none`: prompts explicitly forbid online research; plain discuss mode and
  analyzer mode remain usable without network-dependent behavior. Best for
  internal design questions.
- `required`: every debater must research before answering. The structured
  `discuss_review` must include a `research` object; its `status` must be
  `sourced`, `unavailable`, or `inconclusive` (`not-needed` is rejected), and a
  `sourced` status requires non-empty `sourced_facts` of `{"fact", "source"}`
  pairs. Validation enforces this (with the malformed-response repair pass as
  fallback), so the user can force research instead of relying on automatic
  detection.
- `auto`: debaters self-decide using conservative triggers — current
  vendor/product behavior, pricing, quotas, model availability, laws/policies,
  dependency behavior, or market/tool comparisons — and report
  `status: "not-needed"` when no trigger applies.

With an analyzer and a non-`none` policy, the analyzer's `discuss_agenda` may
add a shared research brief:

```json
{
  "research_required": true,
  "research_questions": ["Is Gemini CLI still available for enterprise users?"],
  "research_question_targets": ["policy/legal/current-facts"]
}
```

The orchestrator forwards those questions and their aligned classifications to the next round's debater prompts
("Shared research brief") so parallel or repeated debater turns do not
duplicate work, and carries unresolved questions forward between rounds. In
`auto` mode the analyzer is told to set `research_required: true` only when a
conservative trigger applies, so it can decide research is unnecessary.

Rendering keeps sourced facts distinct from judgment:

- Each debater comment shows a `Research:` status line and a "Sourced facts"
  list (`fact — source`).
- The final summary adds a "Research" section with the policy, each debater's
  research status, and all cited sourced facts. Gap cases are explicit: it
  states when all debaters deemed research unnecessary, when a debater
  reported research `unavailable`/`inconclusive`, and when a debater reported
  no research status — in each case telling the reader to treat the related
  claims as judgment, not sourced fact.
- The research policy in effect also rides in each posted comment's
  `AGENT_LOOP_META` metadata. Resume decoding of already-posted votes is
  lenient, so rerunning a transcript that was started under a different
  research policy never fails on old comments; enforcement applies only to
  newly invoked debaters.

### Reconciled final evidence

New discuss responses include an `evidence` object with `claims` and `updates`
(both arrays may be empty). Claims use `verified`, `reported-but-unverified`,
or `missing`. A verified claim requires an agent attestation that it inspected
the exact supporting source: `external-source-inspected` with a non-empty
reference, or `checkout-inspected` with repository-relative `path:line`.
`missing` cannot carry a citation. Legacy `research.sourced_facts` are retained
as reported-but-unverified; citations alone never promote a claim.

For later rounds, prompts show stable observation IDs. Debaters retract or
replace an earlier claim only through `updates`, for example
`{"action":"supersede","target_observation_id":"issue-123-r1-Codex-c0","reason":"direct inspection disproved it","replacement_claim_index":0}`.
This works without `--discuss-analyzer`; exact normalized fact/source matches
combine attribution deterministically. Paraphrase grouping is optional
evidence-reconciler behavior and never changes a status, invents an unknown, or
revives an agenda. The existing #529 rule still applies: the final analyzer
gets final-round debater text only.

Final comments render separate Verified evidence, Reported but unverified,
Missing facts, and Retracted or superseded history sections. Reconciliation
input is bounded to 64 observations / 24,000 UTF-8 bytes (fact/reason fields
clip to 512 bytes and sources to 256); the final ledger is bounded to 50
entries / 16,000 bytes. Selection is newest-first within updates/targets,
final-round claims, verified, reported, missing, and old history. Omitted
counts and a digest point to the complete per-round comments and replay
metadata, which remain the audit trail.

A `checkout-inspected` claim's `path:line` is mechanically cross-checked
against the reviewer's assigned checkout: the path must resolve inside that
checkout (no absolute paths, `..` traversal, or symlink escapes), exist as a
file, and the line number must fall within its current line count, or the
loop raises and fails the turn/resume/recovery outright. This is a bounded,
structural guarantee only — it confirms the referenced line exists, not that
it actually supports the claimed fact. The check always runs against
whatever is on disk in the assigned checkout *right now*, at live, repair,
resume, and legacy split-proposal-recovery time alike; there is no persisted
historical snapshot to compare against, so a claim can stop resolving later
if the checkout's contents have since changed. A default (tool-managed)
reviewer checkout is kept at the current base-branch tip by the same
`ensure_agent_workdirs` sync used everywhere else, so it can drift past a
debater's persisted claim as the base branch advances. An explicit reviewer
checkout (e.g. `--codex-dir`) is only checked for cleanliness and remote by
`validate_explicit_workdir` when it happens to already be a Git work tree —
a non-Git explicit directory receives no such check and is used exactly as
the user left it — but its contents are still validated live against the
persisted claim like any other checkout.

### Parallel debater execution

`--discuss-parallel` runs same-round debaters concurrently instead of one
after another:

```bash
agent-loop discuss 123 --repo OWNER/REPO \
  --reviewer codex --reviewer antigravity --reviewer claude \
  --discuss-analyzer claude \
  --discuss-parallel \
  --discuss-debater-timeout 1800 \
  --discuss-on-debater-failure partial
```

Execution model:

- Every pending debater's prompt is built up front from shared pre-round state
  (issue context, prior-round votes, the analyzer agenda), then all pending
  turns are submitted to a thread pool. Debater comments are posted only after
  every turn settles — from the main thread, in configured `--reviewer`
  order — so same-round debaters never see each other's in-progress output.
- The analyzer, consensus detection, and the round summary run only after that
  debater synchronization point.
- Resume works unchanged: already-posted round votes are reused without
  re-invoking their debaters, and when every configured debater's vote resumes
  from comments, no thread pool is constructed at all.
- Log files are isolated per turn: debater logs end in
  `-<agent>-discuss-r<N>.log` and analyzer logs in
  `-<agent>-discuss-analyzer-r<N>.log`. Claude logs additionally include an
  attempt suffix such as `-attempt1` or `-self-update-attempt2`; response
  files already use a per-invocation UUID.
- Parallel mode requires a distinct workdir per debater and rejects the run
  otherwise — deliberately not bypassed by `--allow-shared-dir`, because
  concurrent git/tool activity in a single worktree can corrupt it. The
  analyzer (or the coder) may still share a debater's directory since it runs
  only after the synchronization point.
- Ctrl-C kills all in-flight debater process groups, waits for the workers to
  settle, and re-raises, so no agent subprocesses are orphaned.
- Sequential execution remains the default; prefer it when concurrent
  quota/API pressure across providers is a concern.

Two companion flags apply in both sequential and parallel discuss runs:

- `--discuss-debater-timeout SECONDS` (default: none) bounds each debater
  turn's wall-clock time. On expiry the agent's whole process group is killed
  (SIGTERM, then SIGKILL after a short grace); the turn is classified with
  failure category `timeout` and is never retried as transient, since a kill
  deadline is not a provider hiccup.
- `--discuss-on-debater-failure fail|partial` (default: `fail`) is the
  failure/timeout policy:
  - `fail` aborts the run after in-flight debaters settle. Successful votes
    are posted first so a rerun resumes them instead of re-invoking.
  - `partial` continues the round when at least two debaters produced votes
    (otherwise the run aborts as with `fail`). The failed debater appears in
    the round summary under "Debater failures" with its failure category, is
    recorded in the summary's `AGENT_LOOP_META` metadata (so resume treats the
    missing comment as accounted for and reconstructs an internal `failed`
    placeholder in the round history), and gets a fresh turn in the next
    round. A partial round never declares final consensus — the placeholder
    vote can never match real outcomes — so a partial final round ends in a
    `needs-human` deadlock, with the failures noted in the summary.

### Parallel plan/PR reviewer execution

`--review-parallel` (#594) runs same-round plan or PR reviewers concurrently
instead of one after another. It is accepted by `issue`, `pr`, and `task`
only — `discuss` rejects it with a usage error since it has its own
`--discuss-parallel`, which this flag does not change:

```bash
agent-loop pr 456 --repo OWNER/REPO \
  --reviewer codex --reviewer antigravity \
  --review-parallel
```

Execution model:

- Every pending reviewer's prompt is built up front from the same pre-round
  plan/PR state (current plan or PR diff, prior unresolved items, PR checks
  snapshot), then all pending turns are submitted to a thread pool. Same-round
  reviewers never see each other's in-progress output.
- A validated healthy review is posted by the main thread as soon as its worker
  completes. Its provisional publication checkpoint is durable, so resume
  avoids duplicate comments even if the next run is sequential.
- The orchestrator still waits for every reviewer to settle before shared state
  changes: it aggregates outcomes, numbers unresolved items, and may begin
  coder work only in configured `--reviewer` order. It then posts a neutral
  reconciliation checkpoint; this summary is not a reviewer verdict and is
  excluded from reviewer/approval selection. On resume, settled `new_items`
  from that reconciliation checkpoint remain authoritative; provisional
  publication checkpoints do not cause those items to be numbered again.
- Only after every healthy outcome is applied does the orchestrator raise a
  fatal failure, if any: a quota-reset failure takes priority; otherwise the
  first failure in configured `--reviewer` order. Because healthy reviewers
  were already posted, a rerun resumes them instead of re-invoking them.
- Existing per-reviewer policies are unchanged and isolated per turn: retry,
  structured repair, the unavailable-reviewer / incomplete-review
  distinction, and the PR flow's single-reviewer-fatal rule. One reviewer's
  failure never cancels a healthy concurrent reviewer's turn.
- For PR review, a reviewer's pre-launch `sync_reviewer_pr_before_review` is
  attempted for every pending reviewer before any turn launches; a sync
  failure only removes that reviewer from the launch set and is classified
  (fatal or unavailable) alongside turn failures after the round settles, so
  the remaining reviewers still launch. A single shared PR-checks snapshot is
  used for every concurrently launched reviewer's prompt in the round,
  instead of one fetch per reviewer as sequential mode does.
- Resume works in either mode: already-posted parallel publication checkpoints
  are reused without re-invoking or reposting their reviewers, and when every configured reviewer's review
  resumes from comments (or a round is entirely skipped, e.g. head-advance
  recovery routing), no thread pool is constructed at all.
- Parallel mode requires a distinct workdir per reviewer and rejects the run
  otherwise — deliberately not bypassed by `--allow-shared-dir`, the same
  guardrail `--discuss-parallel` uses, because concurrent git/tool activity in
  a single worktree can corrupt it. The coder may still share a reviewer's
  directory since it only runs after the reviewer synchronization point.
- The coder is never parallelized with reviewers, and review rounds never
  overlap with each other.
- Sequential execution remains the default; prefer it when concurrent
  quota/API pressure across providers is a concern.

### Selective intermediate PR review

PR review defaults to `all-reviewers`, which preserves the historical behavior
of invoking every configured reviewer on every candidate head. The opt-in
`--pr-review-policy selective-intermediate` policy separates intermediate fix
verification from the final independent review:

1. The initial candidate runs the full required board.
2. For a repository-observed narrow transition, only pending resolution owners
   and conservative co-owners are invoked. Reviewers with a valid approval are
   paused, not waived; their approval remains bound to the old SHA.
3. After active obligations clear, a final exact-head sweep invokes each
   required reviewer missing a qualifying approval for the unchanged head.

The reviewer may attach an exact `fix_scope` path list to a blocking or Same-PR
finding. Paths must be normalized repository-relative POSIX files, with no
globs, directories, traversal, duplicates, or oversized payloads. The
classifier uses repository-observed ancestry and the complete Git diff. Any
missing/invalid/disputed scope, unavailable history, rename/copy/deletion,
binary or mode change, out-of-scope path, or configured broad path is
conservative and reactivates the full board. Default broad rules cover workflow,
dependency/lock, build/packaging, schema/migration, and repository-policy or
configuration files. Repeat `--pr-review-broad-rule` to provide an explicit
rule list. `--pr-review-force-full` latches the full board for the remainder
of the run.

Resolution owners are stored with the canonical finding ledger. A non-owner
blocking or Same-PR disposition adds that reviewer as a pending owner; a
non-owner resolved disposition is evidence only. Every owner must provide its
own valid clearing disposition. If all remaining owners are unavailable, the
run records an incomplete review and stops without another coder turn, CI,
qualification, or merge. Mixed-owner obligations continue only with available
owners while the unavailable reviewer remains required.

Scheduler checkpoints are written before reviewer launch and after
reconciliation. They persist the immutable reviewer/policy/rule contract,
head pair, selected/paused reviewers, reasons, final-sweep and force-full
state, phase/primary/owner identities, exact-head approval evidence, and
cumulative scheduler-policy calls avoided. Derived approvals and
obligations continue to come from reviewer records and the canonical ledger.
Missing, malformed, contradictory, or legacy scheduler metadata selects the
full board (under `primary-then-panel` before a qualified panel opening it
instead re-invokes only the primary; see below). A changed reviewer set, policy, or broad-rule digest stops rather
than weakening an in-flight run. Issue-mode implementation handoffs carry the
same PR policy; planning and discussion scheduling are intentionally outside
this feature.

#### Primary-then-panel

`primary-then-panel` is opt-in and requires `--primary-reviewer` plus at least
one other unique `--reviewer`. The primary must be on the configured board.
The scheduler phases are `primary`, `secondary-audit`, `remediation`,
`final-secondary-sweep`, and `full-board`:

1. `primary`: the primary is the only reviewer. When it blocks and the coder
   makes a narrow fix, the primary alone rechecks its own findings; this stays
   in the primary phase until it approves the exact head.
2. `secondary-audit`: after exact-head primary approval, every secondary
   receives the complete base-to-head diff and approved-plan/human context
   independently from a common snapshot. It does not merely validate the
   primary's findings.
3. `remediation`: after a secondary finding, a safe narrow descendant invokes
   every active finding owner together with the primary.
4. `final-secondary-sweep`: owner/primary clearance is followed by a mandatory
   complete-diff sweep of every secondary without qualifying approval on the
   new head, before any CI or merge gate.
5. `full-board`: after a qualified panel opening, any active finding whose
   change is broad, out of scope, non-textual, missing scope, disputed, or
   unreconstructible, any unsafe head change, or the automatic latch selects
   the complete board. The operator force-full latch selects it in any phase.

**Strict pre-panel fallback.** Until the primary first approves an exact head,
the phase is strictly primary-only. The same uncertainties that would reopen the
board after the panel instead re-invoke only the primary with full context:
broad or out-of-scope changes, missing or ambiguous scope, and scope-less
Orchestrator/CI/machine obligations. So do automatic recovery reasons: an invalid
or stale qualification checkpoint, an architecture identity change,
obligation-digest drift, and scheduler-metadata recovery for legacy,
phase-less, malformed, or subject-contradictory records. These reasons apply to
the current decision only. They are recorded with `scheduler_force_full: false`,
and the audit reason is prefixed `strict pre-panel fallback:`. A strict
fallback turn is a complete current-head review, and an older-head primary
approval is never carried. When the primary holds the exact-head approval, the
panel opens (`secondary-audit`) for every available secondary lacking a
qualified approval. It opens even if a CI or machine obligation remains; the
reason lists those obligations, and the later coder repair follows post-panel
rules.

**Qualified panel evidence.** Whether the panel has opened is derived from
comment-ordered history, never from a phase checkpoint alone. A qualified
opening is either a scheduler record with `scheduler_force_full: true` and
`scheduler_force_full_source: operator`, or a `secondary-audit` record for head
S that lists the primary as approved and is preceded by the primary's approved
review of S. Every record after the first qualified opening is post-panel
state. Anything that only looks like a panel before it is an unqualified
premature-panel artifact and is ignored (the audit notes it): secondary
reviews, `full-board`/`remediation` checkpoints, and automatic or legacy
unattributed force-full latches. A secondary approval counts toward carried
approvals, resume, and the exact-head barrier only when its record comes after
the first qualified opening. Premature secondary approvals are therefore neither
resumed nor carried, and they never shrink the first `secondary-audit`. An
operator opening qualifies only records written after it, for every reviewer.

**Post-panel fallback.** After a qualified opening, owner-scoped remediation and
the conservative full board behave as before, and their audit reasons are
prefixed `post-panel fallback:`. A post-panel full-board decision (an unsafe
broad, ambiguous, or out-of-scope transition) and every automatic recovery
reason raise the monotonic durable latch, persisted with
`scheduler_force_full_source: automatic`, so a later narrow head keeps the
complete board. On resume, automatic latches and legacy latches without a source are
restored only when they come after the qualified opening. A legacy latch that
predates any qualified opening is ambiguous between operator intent and the
pre-#840 escalation, so it is not honored. The audit reason says to rerun with
`--pr-review-force-full` to restore the complete board.

**Diagnostic stop and operator override.** The run stops with a plain
`PR review scheduling diagnostic` comment, and no reviewer, coder, CI,
qualification, or merge step runs, when pre-panel safety cannot be established
without the panel. That covers three cases: an active finding pending on a
configured secondary (required reviewers minus the primary) with no qualified
opening; an interrupted round's premature secondary review that is blocking or
carries new items; and scheduler history that cannot be decoded. The last case
is checked at startup and at every round boundary. It stops even with
`--pr-review-force-full`, because resume, approval, ledger, and qualification
accounting all depend on that history; restore the missing round-metadata
records or sidecars (or remove the incomplete record) and rerun. The diagnostic
never fires for Orchestrator, CI, machine, or human-requirement obligations.
For the first two cases `--pr-review-force-full` is the escape hatch. It is durable for the run and
every resume, recorded with source `operator`, and itself a qualified opening.
Under the override, a premature blocking secondary review is superseded. It is
excluded from resume, approval, ledger, and ownership accounting. The operator
opening's audit comment lists it by reviewer, round, head, and item IDs. The
same secondary is then freshly invoked with its earlier claims supplied only in
a non-authoritative "superseded pre-panel review context" prompt block. Only the
post-opening review establishes findings, ownership, and approval.

**Quota versus safety.** Pre-panel uncertainty costs one extra primary turn
rather than N secondary turns. No safety is lost, because the panel's first
invocation is always a complete, independent base-to-head review, and every
configured reviewer must still approve the exact final head. Post-panel
uncertainty stays conservative. The diagnostic stop trades availability for
quota in rare contradictory-history cases, and the operator override restores
the full board.

A secondary that has never reviewed is not a "returning" reviewer: its first
invocation always receives the complete diff, so only reviewers with a prior
record need reconstructible span history.
Any failure, timeout, unavailability, incomplete output, or head mutation
remains blocking and invalidates nonmatching approvals; the settled results of
healthy parallel reviewers are recorded while the failed reviewer stays
outstanding in the required barrier.

The offline `review-evaluation` command consumes local frozen JSON artifacts and
reports unique and severity-weighted marginal findings, process/call/token/time
metrics, CI/escape outcomes, false positives, withdrawals, disagreements, and
unavailable measurements. It never invokes a reviewer or mutates GitHub:

```bash
agent-loop review-evaluation docs/evaluation/frozen_review_artifacts.json \
  --format text
```

Each run in the artifact carries `flow`, `policy`, `run_id`, `findings`,
optional `metrics`, `rounds`, `primary_reviewer`, and approval-round fields, plus
provenance: a run-level `provenance` object (`{"source": ..., "verified":
true}`) covering metrics, round snapshots, and approval rounds; a
`label_provenance` object covering the finding `valid` labels; and optional
per-metric overrides in `metric_provenance`. Every finding must carry an
explicit boolean `valid` label. A measurement is reported as `verified` only
when its provenance is verified; unlabeled findings, missing provenance, or
`verified: false` are reported as `unavailable` with a reason rather than
estimated. Finding `severity` labels must be one of `critical`, `high`,
`medium`, `low`, or `info` (case-insensitive); any other label is rejected at
load time rather than silently weighted as zero, and a valid finding with no
severity label makes the severity-weighted row `unavailable`, naming the
finding, instead of dropping it from the comparison. Every finding must also
carry `contributors`: a non-empty array of nonblank reviewer identities that
raised it. Absent, wrong-typed, empty, or partially invalid contributor data is
rejected at load time and by direct evaluation, so a valid finding is never
silently dropped from unique, marginal, or severity-weighted coverage while the
row still reports `verified`. Finding IDs are namespaced
by run, so identical IDs across runs never collide; two records with the same
`flow`, `policy`, and `run_id` are rejected so distinct findings can never be
collapsed into one.

`flow` is `pr` or `plan` and defaults to `pr` only when the key is absent, so
PR-only artifacts written before the flow dimension keep loading unchanged and
produce byte-identical PR rows. An explicitly present `"flow": null` is a
labeled run whose label is missing, not a legacy record, and is rejected rather
than assigned to the PR rows. Run identity is unique per `(flow, policy, run_id)`,
and aggregation and report titling are per flow. The text report prints a
section titled `Frozen PR review policy evaluation` and one titled
`Frozen plan review policy evaluation`, and the JSON report carries
`flows.pr` and `flows.plan`
(with the historical `policies` key still naming the PR rows). PR review and
issue plan review share the `all-reviewers` and `primary-then-panel` policy
names, so this separation is what keeps a planning run's calls, tokens,
latency, reviewer overlap, findings, and escaped plan defects out of the PR
rows and the reverse. `selective-intermediate` is PR-only and is rejected on a
planning run rather than producing an always-empty planning row. Marginal-beyond-primary rows are `not-applicable` for policies whose
runs declare no primary; a historical full-board run may declare a hypothetical
`primary_reviewer` to measure what the other reviewers would have added; the
frozen full-board planning run does exactly that, so reviewer overlap and
severity-weighted marginal findings are comparable across both planning
policies. Runs
using `primary-then-panel` must declare their primary. Primary-to-panel
approval regressions are a whole-policy measurement: if any primary-bearing run
lacks either approval-round endpoint or verified run provenance, the row is
`unavailable` naming those runs rather than a partial list of the complete
runs. The checked-in
`docs/evaluation/frozen_review_report.json` is regenerated from the fixture
artifact and asserted by `tests/test_review_evaluation.py`.

#### Review contract comparison

Each run may also carry a `review_contract` label naming the reviewer prompt
contract every review round of that run used. There are exactly two values:
`first-finding-permitted`, the historical contract under which a reviewer could
return after substantiating a single blocking defect, and `exhaustive`, the
contract introduced with the reviewer exhaustiveness rule (see
[Protocol](#protocol)). An absent key
defaults to `first-finding-permitted`, mirroring the absent-`flow` default, so
artifacts frozen before this dimension keep loading with an unchanged artifact
hash. An explicitly present null, non-string, blank, or unknown value is
rejected at load time and by direct evaluation, naming the run and the field,
and the CLI exits non-zero without writing a report. The label is not part of
run identity: duplicates are still detected per `(flow, policy, run_id)`.

`review_contract_provenance` is an optional `{"source": ..., "verified": ...}`
object recording the evidence for the label. A malformed object is rejected
like any other provenance, and the object is rejected on a run that names no
`review_contract`, because evidence cannot vouch for a label nobody wrote down.

The JSON report gains an additive section per flow,
`flows.<flow>.review_contracts.<contract>.policies.<policy>`. Runs are
partitioned by flow, then review contract, then scheduling policy; both
contracts and every policy the flow can run are always enumerated, so the shape
is stable. Each `(flow, review_contract, policy)` cell carries `run_count`, the
summed `review_rounds`, `reviewer_calls`, `coder_followup_rounds`, and
`escaped_defects`, and the derived `review_rounds_per_run`,
`reviewer_calls_per_run`, and `escaped_defects_per_run`. A cell value is
`verified` only when every run in the cell has both the verified underlying
measurement and verified `review_contract_provenance`. A run whose label was
defaulted, or whose label provenance is absent or `verified: false`, makes
every value of its cell `unavailable` with a reason naming the runs, so a
mislabelled or unevidenced run can never yield a verified comparison. A cell
with no runs is `unavailable` with the reason `no frozen runs for this review
contract and policy`; nothing is estimated and an empty cell is never divided.
The existing `flows.*.policies` rows and the top-level `policies` alias are
unchanged by this section.

The text report prints a `Review contract comparison (within scheduling
policy)` block after each flow's policy rows. For every policy it lists the
`first-finding-permitted` and `exhaustive` per-run values side by side, and it
states that the before/after comparison is unavailable for that policy when
either contract lacks a fully verified cell under it.

Two reading rules apply:

- Read the effect of the contract only within the same flow and the same
  scheduling policy. Policies differ in reviewer calls, rounds, and escapes by
  design, so comparing an `exhaustive` cell under one policy with a
  `first-finding-permitted` cell under another would attribute a scheduling
  effect to the prompt contract. For the same reason no pooled contract figure
  is produced: there is no per-flow or cross-policy contract rollup.
- Read rounds per run and reviewer calls per run together with escaped defects
  per run. Fewer rounds are an improvement only if they were not bought with
  missed defects.

There are two checked-in artifact pairs, and they never mix:

- `docs/evaluation/frozen_review_artifacts.json` with
  `docs/evaluation/frozen_review_report.json` is the synthetic regression
  fixture. Its runs are unlabeled, so they count in the
  `first-finding-permitted` cell of their own policy with `unavailable` values.
  It is never extended with real runs, so fixture records can neither pool into
  a real baseline cell nor count toward its run minimum, and real data can
  never break the fixture's pinned test values.
- `docs/evaluation/review_contract_runs.json` with
  `docs/evaluation/review_contract_report.json` is reserved for real
  review-contract runs. It ships with an empty `runs` list, whose report has
  every contract cell `unavailable`. `tests/test_review_evaluation.py` asserts
  that the report equals regeneration and checks per-run invariants without
  pinning any metric value, so data-only additions need no test edit.

#### Freezing a real run

Follow this procedure when adding a completed agent-loop run to
`docs/evaluation/review_contract_runs.json`. It is a data-only change: do not
edit the regression fixture, its report, the evaluator, or the tests.

1. **Eligibility.** Freeze only completed runs. Every real run carries an
   explicit `review_contract`; never rely on the absent-key default. A run
   whose review rounds straddle the prompt change (for example a PR resumed
   after the tool was upgraded) belongs to neither contract and must not be
   frozen under either label.
2. **Metrics.** Take the figures from the run's own records, not from memory:
   `review_rounds` is the number of review rounds the orchestrator ran for the
   PR or plan, as recorded by its per-round review comments and round
   metadata; `reviewer_calls` is the number of reviewer invocations across
   those rounds, one per reviewer log under `.agent-loop-logs/` (see
   [Logs](#logs)), which under a selective policy is fewer than rounds times
   reviewers; `coder_followup_rounds` is the number of coder follow-up turns
   between reviews. Record `flow`, `policy`, and, for `primary-then-panel`,
   `primary_reviewer` as the run was configured.
3. **Metric and finding provenance.** Set the run-level `provenance` to
   `{"source": "<where the figures were read>", "verified": true}` only when
   the figures were checked against those records; otherwise set `verified:
   false` or leave the run out. Never upgrade a run that could not be
   verified. Set `label_provenance` the same way for the maintainer-triaged
   finding `valid` labels. Provenance sources starting with `frozen-fixture:`
   are reserved for the synthetic fixture and are rejected by the real-run
   test.
4. **Contract label evidence.** Record the evidence in
   `review_contract_provenance.source`: either the tool commit used for every
   review round of the run relative to the commit that merged the
   exhaustiveness rule, or the captured reviewer prompt in the per-run agent
   log showing the presence or absence of the rule in every round. Mark it
   `verified: true` only when that evidence covers every review round.
5. **Escaped defects.** Use a fixed escaped-defect observation window of 14
   days after the run's PR merge, identical for both contracts. For a plan-flow
   run the window is measured from the merge of the implementation PR produced
   from the plan. Count only defects reported inside the window and traced to
   the run's merged change, for historical baselines as well as new runs, so a
   longer-exposed baseline is truncated to the same window and a just-merged
   run is not credited with an unobserved zero. Always give the metric its own
   `metric_provenance.escaped_defects` entry, whose `source` records the merge
   date, the window end date, the observation date, and where defects were
   searched (issues and PRs referencing the merged change); the real-run test
   rejects a run that carries `escaped_defects` with only run-level
   provenance. `escaped_defects` may be frozen as `verified: true` only after
   the window has closed. Before that, either wait or freeze the entry with
   `verified: false`, which the per-metric override reports as `unavailable`
   and which makes only that cell's escaped-defect values unavailable while
   rounds and calls stay verified. Do not report a comparison while the window
   is still open for any run counted toward either contract.
6. **Regenerate and check.**

   ```bash
   agent-loop review-evaluation docs/evaluation/review_contract_runs.json \
     --output docs/evaluation/review_contract_report.json
   python3 -m pytest tests/test_review_evaluation.py -q -p no:cacheprovider
   ```

A frozen run looks like this:

```json
{
  "run_id": "pr-<number>",
  "flow": "pr",
  "policy": "primary-then-panel",
  "primary_reviewer": "<reviewer>",
  "review_contract": "exhaustive",
  "review_contract_provenance": {
    "source": "tool commit <sha> (after the rule merged in <sha>) for all review rounds",
    "verified": true
  },
  "provenance": {"source": "PR <number> round metadata and .agent-loop-logs", "verified": true},
  "label_provenance": {"source": "maintainer triage <date>", "verified": true},
  "metric_provenance": {
    "escaped_defects": {
      "source": "merged <date>; window end <date>; observed <date>; searched issues and PRs referencing the merge",
      "verified": true
    }
  },
  "metrics": {"review_rounds": 0, "reviewer_calls": 0, "coder_followup_rounds": 0, "escaped_defects": 0},
  "findings": []
}
```

The zeros above are placeholders for the shape only; never freeze an invented
or estimated measurement.

The policy remains non-default until a separate frozen-history review shows
severity-weighted marginal coverage justifies its latency and cost tradeoff.

Returning reviewers receive fresh full context and an orchestrator-computed
diff summary since their previous review, while still being instructed to
inspect the complete base-to-head diff themselves. Before migration
validation, managed CI, qualification, and merge, the live head and current
requirements/plan contract are re-read. Same-head approvals are reusable only
when they match the exact head, approved-plan or handoff identity, surfaced
requirements, and current acquisition contract. Thus selective scheduling
changes reviewer call selection only; CI health, branch protection,
qualification, auto-merge, and the all-reviewers compatibility behavior remain
unchanged.

Example: with Claude, Codex, and Antigravity required, Codex can own a
`src/worker.py` blocker. After a narrow fix, only Codex is called to clear it;
Claude and Antigravity are paused. Once Codex clears the obligation, the final
sweep calls Claude and Antigravity on the new exact head. Savings count only
the calls avoided by selective scheduling after same-head carries, recovery
skips, and unavailable reviewers have been removed from eligibility, so the
example does not double-count the paused approvals.

### Staged issue plan review

Issue plan review has its own scheduling policy, selected independently of
`--pr-review-policy`:

```bash
agent-loop issue 123 --repo OWNER/REPO --plan-first \
  --plan-review-policy primary-then-panel \
  --primary-plan-reviewer codex \
  --reviewer codex --reviewer claude --reviewer gemini \
  --max-rounds 10
```

`--plan-review-policy` accepts `all-reviewers` (the compatibility default) and
`primary-then-panel`. `--primary-plan-reviewer` is required by the staged
policy, must be on the configured `--reviewer` board, and needs at least one
secondary. `--plan-review-force-full` is the operator override. All three
validate independently of `--pr-review-policy`, `--primary-reviewer`, and
`--pr-review-force-full`, so a run may stage planning with full-board PR review
or the reverse. Omitting them preserves today's full-board planning behavior
byte-for-byte: no planning scheduler metadata is written and no reviewer is
paused.

#### Candidate key and generation-1 requirement

Every staged planning decision is bound to one canonical *exact-plan candidate
key*: the ordered tuple of the plan subject, aggregate plan identity,
execution-strategy identity, risk-test-matrix identity, and a deterministic
digest of the surfaced planning-requirement IDs in force for the round. The same
key is used by scheduler records, carried approvals, panel-opening evidence, and
resume. A legacy unversioned plan has no execution-strategy or risk-matrix
identity, so the key cannot be formed; staged planning is refused with an
actionable message instead of degrading silently.

#### Phases and the reviewer-only phase advance

1. `primary`: only the primary plan reviewer is invoked. It rechecks its own
   plan findings on each revision until it approves the exact candidate key.
2. `secondary-audit`: an exact-key primary approval opens the panel. Every
   available secondary without a qualified exact-key approval is invoked with
   the complete issue context and the byte-identical candidate plan the primary
   approved.
3. `remediation`: a narrow revision after the opening invokes the active finding
   owners from the canonical ledger plus the primary.
4. `final-secondary-sweep`: every required reviewer still missing a qualifying
   approval of the unchanged candidate key is invoked.
5. `full-board`: a broad revision after a qualified opening, the automatic
   latch, and the operator override.

`secondary-audit` and `final-secondary-sweep` run as *reviewer-only rounds*: the
loop posts a `plan-phase-advance` record, increments the round number, and
invokes the reviewers against a byte-identical candidate plan with no planner
turn. **Round budget:** each advance costs a round, so staged planning always
consumes more rounds than full-board planning for the same plan. Raise
`--max-rounds` when enabling it. Exhausting the budget while an advance is still
pending reports that cause distinctly from "reviewers still reported blocking
plan issues", naming the outstanding phase and reviewers. Both the advance
record and that diagnostic name the phase that is still *pending* — the panel
audit after a primary approval, the final sweep after a remediation round — not
the phase of the board that just finished.

#### Transition classifier

The planning flow has no diff, so the classifier uses authenticated data only
and never attributes a patch operation to a finding or an owner. It reports
`recheck` when the candidate key is unchanged; `narrow` when the revision is an
authenticated `semantic-patch-v1` bound to the immediately preceding base
identity whose operations touch only `summary`, `plan_steps`, `deferred_work`,
`plan_actions`, `external_dependencies`, and risk-matrix row add/edit
operations, while the execution-recommendation identity, human-requirement
dispositions, `additional_closing_issue_ids`, and architecture-impact status are
all unchanged; and `broad` for everything else, including a full-state rewrite,
a missing or unbindable sidecar, and an unreconstructible ledger. A routine
remediation revision that edits plan steps and matrix rows therefore stays
narrow and does not latch the complete board on the first revision. Ownership
always comes from the canonical finding ledger, never from the patch: an active
planning obligation is a `blocking` **or** `same-plan` finding, so a Same-plan
panel follow-up keeps its durable owner and routes the next round to
`remediation` rather than falling through to a final sweep.

#### Qualified panel evidence

A qualified panel opening is derived from comment order: an operator-sourced
planning force-full record, or a `secondary-audit` planning scheduler record for
candidate key K that lists the primary as approved and is preceded by the
primary's own approved plan review of K. Premature secondary plan reviews,
`full-board`/`remediation` checkpoints, and unattributed or automatic latches
recorded before an opening are unqualified artifacts: they are listed in the
audit, excluded from approval, ownership, and resume accounting, and supplied to
a re-invoked secondary only as non-authoritative superseded context.

#### Degraded planning history and the diagnostic stop

Degraded planning history is partitioned into exactly four disjoint classes with
one outcome each, so the same durable history can never both continue and stop:

| Class | Condition | Outcome |
| --- | --- | --- |
| A `absent` | The record carries no scheduler fields at all (every legacy and full-board planning comment). | Conservative fallback; continue. |
| B `invalid` | Scheduler fields extracted but are partial, malformed, or internally contradictory. | Conservative fallback; continue. |
| C `contradictory-key` | The record decodes valid but its persisted key components contradict the canonical plan. | Conservative fallback; continue. |
| D transport failure | The planning round-metadata record set cannot be extracted at all. | Stop with the planning diagnostic. |

Before a qualified opening the fallback re-invokes only the primary with full
context under a `strict pre-panel fallback:` reason and latches nothing; after
one it selects the complete board under a `post-panel fallback:` reason and
raises the durable `automatic` latch.

Degradation is scoped to the current *recovery boundary*: the latest valid
planning scheduler checkpoint. An invalid record written before that checkpoint
stays listed in the audit but grants no phase authority and no longer degrades
later rounds, so a class-B fallback costs one round and then the run advances on
its next exact-plan primary approval instead of repeating the primary-only turn
until `--max-rounds` is exhausted.

The persisted planning scheduler contract is immutable for the run, and that is
checked under **both** policies. Restarting an issue that already carries a
staged planning contract with the compatibility default, with a different
primary, or with a different reviewer board stops with an actionable message
naming the persisted contract instead of silently continuing on a different one.

The run stops with a plain `Plan review scheduling diagnostic` comment, and no
reviewer and no planner turn, in exactly three cases: an active plan finding
pending on a configured secondary with no qualified opening; an interrupted
round holding a premature blocking secondary plan review; and the class-D
transport extraction failure. `--plan-review-force-full` recovers the first two
only. It can never recover class D, because approval, ownership, and
qualification accounting all depend on a readable record set — restore the
missing planning round-metadata records and rerun instead.

When the override authorizes the complete board, each superseded pre-panel
secondary plan review is named in the posted audit, excluded from approval and
ownership accounting, and replayed to its own author — and to no other
reviewer — as an explicitly non-authoritative `Superseded pre-panel plan review
context` block in that reviewer's fresh plan review prompt. The block is
bounded, states that the earlier claims are not findings, not plan-item
dispositions, and not approvals, and instructs the reviewer to re-raise any
concern that still holds as a new finding of the fresh review rather than
dispositioning it as a prior plan item.

#### Carried approvals and signed human requirements

An exact-key approval is carried across rounds only when the stored record
matches every component of the current candidate key *and* itself carried
`HUMAN_REQUIREMENTS_RESOLVED` for exactly the currently surfaced
planning-requirement ID set. Plan reviewer records persist those IDs only when
the approval actually carried the acknowledgement, so a carried approval can
never satisfy the signed-requirement gate vacuously. Any plan revision, and any
added, edited, replaced, or withdrawn signed requirement, changes the
requirement-digest component of the key and invalidates every carried approval.
The final gate evaluates every required reviewer, carried and current alike; a
carried approval that fails the key or acknowledgement comparison blocks
approval and re-invokes that reviewer rather than being repaired in place. Plan
item numbering, disposition reconciliation, deferred stages, child-stage
topology, and decomposition decisions are never changed by scheduling.

At the approval-to-implementation boundary the loop re-reads the issue, and the
authoritative parent issue when there is one, and stops whenever the surfaced
signed requirement set differs at all from the set the plan was reviewed
against. An addition and a withdrawal are equally disqualifying: every plan
review and every carried approval was bound to the earlier requirement digest,
so the message names the added and withdrawn IDs and you re-run planning rather
than carrying the plan into implementation.

#### Exclusions

Discussion-mode scheduling and the child-planning cycle always invoke the full
board, enforced by configuration reset rather than by convention: the child
planning configuration and the semantic-dedupe isolated provider configuration
both reset the planning policy, primary, and force-full fields. PR-flow
scheduling, qualification, managed CI, branch protection, and merge behavior are
unchanged. The staged planning policy remains non-default until a flow-separated
frozen evaluation justifies the latency and cost tradeoff. That comparison is
now available: `review-evaluation` carries a `flow` dimension, and
`docs/evaluation/frozen_review_artifacts.json` contains full-board and staged
planning runs whose calls, tokens, latency, reviewer overlap, unique and
severity-weighted marginal findings, and escaped plan defects are reported in
the `plan` flow, separately from the PR rows. A measurement the frozen run
never recorded, such as escaped plan defects on the staged run or the
full-board run's absent primary-to-panel transition, is reported as
`unavailable` with a reason rather than borrowed from the other run, so
changing the default still needs runs whose provenance covers the measurement
the decision rests on.

### Phased decomposition versus split materialization

Decomposition modes and split materialization select one child-issue path.
They share a parent-wide flat cap; decomposition modes reject the split flag
before any write. Pick the row that matches your situation:

| Situation | Correct mechanism |
| --- | --- |
| Approved detailed staged plan with phase contracts | `--plan-execution-mode decompose-only` |
| Same plan, but implement the current phase now (rerun the parent to advance) | `--plan-execution-mode implement-by-phase` |
| Approved plan you want implemented as a single PR, no phase breakdown | `--plan-execution-mode implement-one-shot` (or `--implement-after-approval`) |
| Approved plan whose reviewed recommendation should choose the topology | `--plan-execution-mode auto` |
| Plan review only, no implementation, no detailed child issues | `--plan-execution-mode plan-only` (the default) |
| Discuss `split` consensus, or plan-only deferred work with no detailed phase decomposition | `--materialize-split-issues` |

**Do not combine `--materialize-split-issues` with `--plan-execution-mode
decompose-only`, `implement-by-phase`, or `auto`.** The CLI rejects the combination
before a checkpoint or child create. `decompose-only` uses typed child stages
directly when present and otherwise invokes one model decomposition; it never
materializes a competing topology.

What each mechanism produces and where the run stops:

- **`plan-only`** (default): posts the approved-plan summary and stops. No
  implementation, no detailed child issues. One nuance: if
  `--materialize-split-issues` is also passed, generic split children are
  still filed even in `plan-only`, because that materialization step runs
  before the mode is dispatched — `plan-only` only skips decomposition and
  implementation, not legacy split materialization. A fresh v1 staged
  recommendation owns its complete topology, so it remains inert at that
  legacy seam; a fresh v1 one-shot recommendation preserves the historical
  split-materialization behavior.
- **`decompose-only`**: uses typed `child_stages` directly when present;
  otherwise it validates one model decomposition. The complete topology is
  checked against the shared default cap of 15 (override with
  `--flat-child-limit`) before any checkpoint or child create. An over-limit
  result creates nothing and returns a structured decision to consolidate or
  use hierarchical decomposition tracked in #720. Typed stages remain the
  parent-owned plan remainder and are represented in the parent summary.
- **Fresh v1 recommendation**: uses the reviewed staged allocation directly;
  it preserves every enriched field and exact automation class, records a
  compact canonical execution decision before child creation, and reuses the
  same summary when a later explicit staged policy resumes. A one-shot
  recommendation cannot be sent through a decomposition policy, and a staged
  recommendation cannot be sent through one-shot implementation. `auto` maps
  these strategies to their compatible concrete actions after approval.
- **`auto` dry-run**: performs read-only reconciliation and prints the resolved
  action. Staged previews list the normalized children, first-phase eligibility,
  remaining work, and retained/final-integration obligations; one-shot previews
  state that implementation would be selected. It does not persist approval,
  file follow-ups or children, create handoffs, invoke a coder, or open/update a
  PR. A later non-dry run performs the persistence-first path once.
- **`implement-by-phase`**: creates every phase child issue, records a
  per-phase `AGENT_PLAN_PHASE_IMPLEMENTATION` handoff, then implements the
  current (first incomplete) `agent-pr` phase and stops after that phase's PR
  review loop. Rerunning the parent advances to the next phase once the current
  phase's child is closed with a merged PR, and reports a terminal delivery
  report once every phase is complete. If the selected phase is `human-action`
  or `manual-close` and its child is still open, the run stops without
  implementing anything. Resume an in-progress child with
  `agent-loop issue <child>`; rerun the parent to move on to the next phase.
- **`--materialize-split-issues`**: files one linked child issue for discuss
  `split` proposals or plan-only/one-shot deferred work. Use
  `external_dependencies` for existing `#N`, issue URL, or
  `owner/repo#N` references; `deferred_work` and `plan_actions` are recorded
  only. Legacy `deferred_stages` are record-only and are never auto-filed.
  idempotent (tracked by the parent's `AGENT_DISCUSS_SPLIT` marker), capped by
  the same 15-child parent budget, and files nothing from free-form prose
  narrowing alone — only from the two structured signals above. It never
  truncates; over-limit materialization returns the same structured decision.

Copyable commands, one per workflow, each stopping where noted:

```bash
# Detailed staged plan: create phase children, stop (review/resume each child separately)
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode decompose-only

# Same plan, but also implement the current phase now; stops after that phase's PR review loop.
# Rerun the same command after that phase's child closes with a merged PR to advance to the next
# phase, and again once every phase is delivered to get the terminal report.
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode implement-by-phase

# Discuss-mode split consensus: file generic linked children, stop (no implementation in discuss mode)
agent-loop discuss 123 --repo OWNER/REPO --materialize-split-issues

# Plan-only run whose deferred_stages should still be filed as generic children; stops after plan approval
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode plan-only --materialize-split-issues

# Pick work back up on a child that is still open
agent-loop issue <child-issue-number> --repo OWNER/REPO
```

Anti-example — do not run this; it duplicates children:

```bash
# WRONG: decompose-only already creates one child per phase; --materialize-split-issues
# creates an unrelated, overlapping set of generic children for the same deferred work.
agent-loop issue 123 --repo OWNER/REPO --plan-first \
  --plan-execution-mode decompose-only --materialize-split-issues
```

Two worked examples:

1. **Plan-first staged master issue.** Run `--plan-first` with reviewers
   until the plan is approved, then run `--plan-execution-mode decompose-only`
   to create the phase children and drive each one with
   `agent-loop issue <child>`. With `implement-by-phase` instead of
   `decompose-only`, the same parent command also runs the work: each parent
   rerun selects the first phase that is not yet complete, dispatches it, and
   stops after that phase's PR review loop. A phase counts as complete once its
   child issue is closed and its implementation PR is merged (for a
   `human-action` or `manual-close` phase, once its child issue is closed), so
   the sequence is: run the parent, finish the child it names, rerun the
   parent. When every phase is delivered the parent rerun prints a terminal
   report naming each stage and any operator-owned retained-parent or
   final-integration work, and dispatches nothing further; it does not close
   the parent for you. `--materialize-split-issues` is not used anywhere in
   this flow — the phase children already are the detailed decomposition.
2. **Discuss-mode split consensus.** Run
   `agent-loop discuss 123 --repo OWNER/REPO --materialize-split-issues` to
   get a `split` consensus filed as generic linked child issues, then plan or
   implement each child separately with `agent-loop issue <child>`. A later
   `agent-loop issue 123 --repo OWNER/REPO --plan-first --implement-after-approval`
   run on the parent resolves which specific child the approved plan covers
   via a unique title match, or `--split-stage <child>` when the match is
   ambiguous or missing.

`--implement-after-approval` (the `implement-one-shot` alias) combined with
`--materialize-split-issues` is not the failure mode above — it is the
supported split-stage handoff flow described in [Split issue
materialization](#split-issue-materialization) below. The warning here is
scoped to `decompose-only` and `implement-by-phase` specifically, since only
those two modes already create detailed per-phase children. It cannot be
combined with `auto`.

Skill mode's `run-decompose` and `run-implement-by-phase` helper commands
(see [`docs/skill_mode.md`](skill_mode.md)) drive the same
`decompose-only` / `implement-by-phase` modes and are subject to the same
rule.

### Split issue materialization

By default, a discuss `split` consensus or a plan-first plan that narrows
scope leaves its proposed follow-up stages as text in issue comments — easy to
miss, especially when `--implement-after-approval` proceeds straight into
implementing one stage. Pass `--materialize-split-issues` (on `discuss` or
`issue`) to file each remaining stage as its own linked child GitHub issue
instead:

```bash
agent-loop discuss 467 --repo OWNER/REPO --materialize-split-issues
agent-loop issue 467 --repo OWNER/REPO --plan-first --implement-after-approval \
  --materialize-split-issues
```

Default is off. Whether or not the flag is set, the orchestrator always warns
explicitly when split follow-ups would otherwise go unfiled — in the discuss
final summary, the plan-approval summary, and the CLI log — so the gap can't
hide in a neutral listing.

Two structured signals drive materialization; free-form prose narrowing is
never enough to auto-create issues:

- **Discuss `split` proposals.** When every debater's final vote is `split`,
  the merged `split_proposals` from that round are the remaining stages (a
  discuss run implements nothing, so every proposal is unfiled/unimplemented).
- **Plan `deferred_stages`.** A coder's structured `plan_revision` or initial
  `plan_state` response may declare an optional `deferred_stages` array of
  `{"title", "summary"}` objects for scope the plan intentionally leaves out.
  Declared stages render into the canonical plan under a `### Deferred stages
  (not in this plan)` heading, so they carry into subject hashing, stored plan
  state, reviewer prompts, and resume. In `--plan-first` mode, the stage the
  approved plan actually covers is never filed as a child — only the
  `deferred_stages` (plus any not-yet-covered discuss split proposals from an
  earlier `discuss` run on the same issue) are remaining stages.
- If neither signal is present but the approved plan's text still looks like
  it narrows scope (mentions "stage 1 of", "first stage", "out of scope",
  "separate issue", or "follow-up issue"), the orchestrator posts a
  heuristic-only warning. It never files issues from this signal alone.

Each child issue is created with a deterministic title
(`[#<parent> stage] <proposal title>`), a `Part of #<parent>` first line, the
proposal text, the split rationale from the debaters who voted `split` (when
available), links to sibling stages already materialized, and execution
instructions to run `agent-loop issue <child>` and never use a closing keyword
against the parent. Every child body carries a durable
`AGENT_SPLIT_CHILD: parent=<N> key=<hash>` HTML-comment marker.

Materialization is idempotent and crash-safe:

- The parent issue accumulates a single `<!-- AGENT_DISCUSS_SPLIT: ... -->`
  marker recording every known child (title, key, issue number/URL, and
  whether it was `created` or `adopted`); a rerun that finds every current
  proposal already covered by that marker performs zero GitHub writes.
- If a prior run crashed after creating some child issues but before posting
  the marker, the next run searches existing issues
  (`gh issue list --search '"[#<parent> stage]" in:title'`) before creating
  anything, adopts any match into the metadata instead of duplicating it, and
  files only the remaining stages.
- Proposals are deduplicated by a normalized-title key across the whole
  parent, not just within one run, so subject-hash drift between a discuss
  consensus and a later plan-first run never refiles the same stage twice.
- Materialization uses the shared default cap of 15 child issues per parent;
  every desired stage is counted before mutation, and an over-limit request is
  returned without a checkpoint, issue, warning, or partial topology.
- Exact child identities and decomposition checkpoints make reruns adopt open
  or closed children after a create-before-summary failure. Weak cross-workflow
  title matches are open-only and require an explicit parent link; authorship
  alone never makes a protocol record canonical.
- `--dry-run` previews `gh issue create`/`gh issue list --search` commands
  without writing any state.

When a parent's proposals were already fully materialized into child issues
(from an earlier `discuss` run, or a prior `--plan-first` run on the same
issue), a later `issue --plan-first --implement-after-approval` run on that
parent must resolve which child stage the approved plan implements before
handing off implementation — it never implements the parent as a monolith in
that case. Resolution is a unique normalized-title match between the plan and
a child's title, or an explicit `--split-stage <child-issue-number>` flag when
the match is ambiguous or missing:

```bash
agent-loop issue 467 --repo OWNER/REPO --plan-first --implement-after-approval \
  --split-stage 480
```

The resolution is recorded in an `<!-- AGENT_SPLIT_STAGE_HANDOFF: ... -->`
parent comment marker so reruns reuse it instead of re-resolving. The staged
implementation prompt then instructs the coder to use `Closes #<child>` plus
`Refs #<parent>` in the PR body — never a closing keyword against the parent —
and `validate_pr_body_does_not_close_issue` rejects a PR body that uses
`Fixes`/`Closes`/`Resolves` against the parent while other stages remain
unfiled or unimplemented, with an actionable error to edit the PR body and
rerun `agent-loop pr <n>`.

### Issue-to-PR association and recovery

Issue mode first trusts a validated `AGENT_ISSUE_PR_HANDOFF` marker. Its schema,
issue, case-insensitive repository/PR URL, PR number, and current open state are
checked against GitHub; the recorded head SHA is historical handoff evidence
and may change after review commits. Canonical records resume across direct and
plan-first invocations, including `implement-by-phase` child PRs, regardless of
the producer flow stored in the marker.

Without canonical metadata, crash-window recovery scans all open PR pages and
accepts only one same-repository closing reference such as `Fixes #123`,
`Closes owner/repo#123`, or `Resolves https://github.com/owner/repo/issues/123`.
Bare `#123`, `Refs #123`, contextual URLs, discussion prose, titles, branch
names, and cross-repository references are not candidates. Pagination covers
the full open-PR list. Multiple strong candidates stop with their PR numbers
and matched closing evidence, plus cleanup and `agent-loop pr <number>`
guidance. A sole candidate must also carry one or more identical, complete
`Agent-Issue-Provenance` Git commit trailers for the exact repository, issue,
flow, and (for plan-first recovery) approved-plan hash. Missing, malformed,
conflicting, stale, or unstable commit history fails closed. This trailer is
an unauthenticated convention that reduces accidental adoption, not an
authentication boundary, because contributors can copy it.

New direct and approved-plan PRs must contain a closing phrase for their issue
before a handoff marker is posted. A staged child closes the child, includes
non-closing `Refs #<parent>`, and never closes the parent. Existing canonical or
plan-handoff PRs retain their validated provenance and do not need a closing
phrase added retroactively; staged-parent no-close safety still applies.

Plan-first reruns reconstruct persisted plan state from comments before memory
preparation or agent calls. A matching approved-plan hash resumes without
planning; a definite mismatch stops with the PR and both hashes and directs the
operator to `agent-loop pr <number>` or handoff cleanup. If no plan round can be
reconstructed, a valid canonical record resumes with a warning and its recorded
hash. A metadata-free legacy candidate is stopped rather than assigned invented
plan provenance, while direct mode can backfill only a matching trailer-backed
candidate. Metadata-free recovery is retired: pre-trailer PRs and
`agent-loop managed-pr --head` PRs without a trailer require direct resume with
`agent-loop pr <number>`. Squash or rebase removal of the trailer matters only
before a canonical handoff exists; newly created PRs receive an advisory
warning and still complete their authoritative handoff.

Logs identify canonical marker versus legacy closing-reference evidence. To
recover from a false handoff, edit or delete the latest canonical marker and,
only if present, remove the unrelated PR's accidental closing phrase or close
that PR. Removing a marker alone is sufficient when the PR has only an
incidental mention or contextual URL: that text cannot recreate the handoff.
`agent-loop pr <number>` remains the explicit operator path for a known PR.

### Expected closing issue contracts

Use the repeatable `--expected-closing-issue POSITIVE_ID` option when a single
PR intentionally completes multiple issues. Issue mode starts with the actual
implementation issue and unions CLI additions with the approved plan's
`additional_closing_issue_ids`; the plan field is optional, and an explicit
empty list is different from omission. Direct `pr` and `managed-pr` use the
explicit CLI IDs as the complete contract. Direct `pr` without a declaration or
recovered contract remains contract-unknown and does not infer expected issues
from linked-issue prose, `Refs`, or related URLs.

On public `pr --managed-ci` recovery of an authenticated issue-created PR, the
issue scope (and any authenticated approved-plan additions) supplies the
immutable contract when no PR-side record exists. The current PR body is then
checked for unexpected same-repository closing references before managed-CI
activation can apply its suppression label; body text remains evidence for this
validation, never the source of the expected set.

The normalized set is persisted in the schema-version-1 issue handoff and in a
canonical PR-side `AGENT_PR_EXPECTED_CLOSING_ISSUES` record. Recovery requires
the issue and PR records to agree, while a validated one-sided write or
supersession crash window can be completed idempotently. Omission on a rerun
reuses the recovered set. A changed explicit declaration must match exactly;
`--supersede-expected-closing-contract` is available only on `issue` and `pr`,
requires a full declaration, and permits only a proper superset. The new record
stores its contract hash and supersession hash, so narrowing or replacement is
not silently accepted.

After PR creation, and again after every body-edit round and immediately before
reviewer dispatch, qualification, or merge, the loop fetches the current body
and checks every expected ID. Each ID needs its own case-insensitive GitHub
closing keyword (`Close(s|d)`, `Fix(es|ed)`, or `Resolve(s|d)`) paired with a
same-repository `#N`, `OWNER/REPO#N`, or canonical issue URL. A keyword does not
carry over to later bare targets, so `Closes #847, #848` satisfies only #847.
`Refs`, cross-repository references, incidental links, comments, and code
examples are not affirmative closure evidence. Blockquotes and nested list
items remain active Markdown evidence because GitHub linkifies them.

Staged and materialized topology is single-PR scoped: a child contract may
include the child and child-scoped additions, but excludes the unfinished
parent. The body must close the child and use non-closing `Refs #<parent>`; any
parent closing keyword is rejected. Parent-scoped additions are rejected before
child creation, stage handoff, or coder invocation, and the operator must rerun
the actual child with its contract.

If validation reports missing IDs, edit the existing PR description so every
listed issue has its own closing keyword/reference pair, then resume with
`agent-loop pr <number>`. No second PR is created. The reserved marker names
may appear in ordinary coder or reviewer prose, but only an exactly well-formed
canonical marker is trusted; such a forged marker aborts before the first
durable write.

If `--repo` is omitted, the tool runs `gh repo view` from the current working
directory, or from `--codex-dir` when that flag is provided, and uses the
detected `OWNER/REPO`. Pass `--repo` explicitly when running outside the target
repository.

Reverse the direction so Codex creates/fixes and Claude reviews:

```bash
agent-loop task "Refactor the cache layer" \
  --repo OWNER/REPO \
  --coder codex \
  --reviewer claude
```

Use Gemini as the coder. Gemini is invoked in headless mode with `gemini --prompt`:

```bash
agent-loop task "Improve validation errors" \
  --repo OWNER/REPO \
  --coder gemini \
  --reviewer codex
```

Use Gemini as one reviewer:

```bash
agent-loop pr 123 \
  --repo OWNER/REPO \
  --reviewer codex \
  --reviewer gemini
```

Require both reviewers to approve. The coder may also be listed as a reviewer
when you want the same agent to work in separate coding and review passes:

```bash
agent-loop pr 123 \
  --repo OWNER/REPO \
  --reviewer codex \
  --reviewer claude
```

## Workdirs

Explicit `--claude-dir`, `--codex-dir`, and `--gemini-dir` values are used
exactly as provided. Missing explicit directories are still created for
backwards compatibility.

When an active agent directory is omitted, the default checkout path is scoped
by repo and agent:

```text
/tmp/coding-review-agent-loop/OWNER-REPO/claude/repo
/tmp/coding-review-agent-loop/OWNER-REPO/codex/repo
/tmp/coding-review-agent-loop/OWNER-REPO/gemini/repo
```

The tool prints the selected default workdirs. If a default checkout does not
exist, it runs `gh repo clone OWNER/REPO <path>`. If it already exists and is a
clean checkout for the requested repo, it fetches origin and fast-forwards the
resolved base branch. In `pr` mode the base defaults to the PR's base branch,
then the repository default branch; in `issue` and `task` modes it defaults to
the repository default branch. An explicit `--base` overrides these defaults.
Default checkouts are tool-owned and disposable; if one
is dirty, the tool logs the cleanup, runs `git reset --hard` and `git clean -fd`,
then syncs the configured base branch. If a default checkout points at another
repo or is not a git checkout, the command fails clearly instead of overwriting
local work.

Explicit workdirs remain conservative. A dirty explicit git checkout fails
clearly, and an explicit checkout whose origin does not match `--repo` is
rejected.

Coder prompts name the active assigned checkout as an absolute path and set
`AGENT_LOOP_WORKDIR` to that same path for the agent subprocess. Implementation,
inspection, tests, commits, and pushes are expected to stay in that checkout
unless the user explicitly authorizes another path. Coder prompts also ask the
agent to run `pwd` and `git status --branch --short` before tests or commits,
and to avoid sibling, home, deployment, or duplicate clones such as `~/REPO` or
`~/claude-code/REPO`.
Every coder `gh` body must be written to a temporary file outside the checkout
and passed with `--body-file`.

The orchestrator validates coder-reported test commands before posting normal
coder progress. For `Tests:` reports and structured `tests_run` entries, it is
role-aware rather than pattern-only: an explicit working directory (`cd
<path>`, `-C <path>`, `--directory[=]<path>`, `--chdir[=]`, `--cwd[=]`,
`--rootdir[=]<path>`) is always validated, and so is any ordinary target,
checkout, or artifact path (a positional test path, a redirect target, an
`--rootdir` value, and so on). The values of the explicitly recognized report
flags `--output`, `--output-dir`, `--output-file`, `--report`, and
`--report-file` are the narrow exception: they are treated as report
destinations and may name paths outside the checkout. This exemption is
flag-gated only; a shell redirect to an outside path such as `pytest tests/ >
/tmp/out.log` is still rejected. An absolute interpreter, runner, or package-manager
executable in *program position* -- for example `/usr/bin/python3`, a venv's
`.venv/bin/pytest`, or a wrapper like `sudo`/`env` in front of one -- is not
treated as a test location, and neither is the value of a narrow set of
interpreter-valued flags/env-vars (`--python`, `PYTHONPATH=`, and similar).
The supported wrappers are parsed only far enough to locate their nested
program: `timeout`, `sudo`, `nohup`, `nice`, `time`, `stdbuf`, `command`,
`env`, and `xargs`. GNU `timeout` consumes one duration after its options
(including bare, suffixed, and fractional values); value-taking wrapper options
are likewise consumed only for their defined options such as `timeout -k`/`-s`,
`sudo -u`/`-g`, `nice -n`, `stdbuf -i`/`-o`/`-e`, `env -u`/`-C`, and
`xargs -n`/`-P`/`-I`. These exemptions
are gated on command position or on a specific interpreter-valued construct,
not on path components: wrapper operands, option values, workdirs,
test/script targets, unrecognized outputs, remote targets, malformed commands,
and arbitrary `/tmp` paths receive no blanket exemption, so a toolchain-shaped path
used as an ordinary argument (`pytest /outside/bin/tests/test_foo.py`) still
fails containment. In narrative execution phrases, the parser may skip one
optional determiner, leading `VAR=VALUE` assignments, and basename-normalized
wrapper paths before resolving the effective head; a direct HTTP(S) URL at that
head is treated as the target. If a recognized wrapper prefix is malformed,
recovery may promote a later command-shaped token only when it occurs before
the governing boundary, and never promotes an otherwise unattached URL.
Wrapper traversal stops only at the next execution verb. Malformed recovery in
both verb-adjacent and verbless narrative clauses, as well as prose
prepositional attachment, stops at governing negation or the next execution
verb. After a command head is successfully resolved or recovered, its target
span runs through the end of the clause. Package acquisition (`pip install ...`,
including the `python -m pip install ...` form) is exempted from the separate
live-target check described below, but never from path containment.
A URL is rejected as a live remote target when it appears in command syntax
(a structured entry, or backtick-quoted command text in a `Tests:` report) or
is reported as the target of an affirmative execution phrase in prose (`ran
curl https://...`, `hit https://...`, `ran the suite against https://...`).
The whole execution phrase is scanned for the attached URL, so an earlier
non-URL prepositional object does not hide the real target (`ran the suite
against the production environment at https://...` is still rejected); the
phrase ends at a negation word or at the next execution verb. A negated
execution phrase (`Did not run curl https://...`, `ran the suite against the
local stub and never against https://...`) and unattached URL-like prose
(deployment notes, session-cookie mentions) are accepted, and
this narrower prose latitude does not extend to command syntax.

The one exception, in command syntax as well as prose, is a URL whose host is
a loopback address: `localhost`, an IPv4 address in `127.0.0.0/8`, or the IPv6
loopback `::1` (for example `http://localhost:8765`, `http://127.0.0.1:8765`,
`http://127.42.0.9:8765`, `http://[::1]:8765`). Coders commonly need to stand
up a local server and drive an E2E client such as Playwright or Node against
it inside the assigned checkout; a loopback target reported as a test command
is accepted while every other host -- including non-loopback private/LAN
addresses such as `0.0.0.0` or `192.168.x.x` -- is still rejected. This
exemption is host-classification only, not a blanket exemption for the
clause: a loopback target reported alongside a live remote target in the same
command (`... && curl https://live.example`) still fails on the live target.
Authority syntax that a browser or Node client could parse differently than
the loop's own classifier -- for example a backslash inside the URL, which
WHATWG URL parsers treat as a path separator and could resolve to a different
host than a strict URL parse reports -- is never treated as loopback, even if
it superficially contains `localhost` or a loopback IP; it is rejected as an
unverifiable live target instead. If an explicit
test location is outside the assigned checkout, or a live remote target is
detected, the loop fails with an `AgentLoopError` naming the offending
command/URL and assigned checkout. When that failure happens after a PR was
already created or detected, the error also confirms the PR state and tells
the user to continue with `agent-loop pr <number>` instead of rerunning
implementation and creating a duplicate PR. For initial issue, task, and
approved-plan implementations, the loop also checks that the assigned checkout
`HEAD` advanced when the coder reports a PR; unchanged `HEAD` is rejected
before the coder PR comment is posted.

These temporary checkouts may disappear after reboot or `/tmp` cleanup. Large
projects and long-lived agent setups should use explicit persistent workdirs to
avoid repeated clone, dependency setup, and indexing costs.

## Agent Memory

Agent memory is enabled by default. Before an agent prompt is built, the loop
creates or refreshes advisory repo memory in a durable, repo-scoped user cache
directory. On Linux the default is:

```text
~/.cache/coding-review-agent-loop/repos/OWNER-REPO/memory
```

If `$XDG_CACHE_HOME` is set on Linux, the root is
`$XDG_CACHE_HOME/coding-review-agent-loop`. On macOS the default root is
`~/Library/Caches/coding-review-agent-loop`; on Windows it is
`%LOCALAPPDATA%/coding-review-agent-loop/Cache`.

The memory cache includes a repo summary, architecture map, module index,
execution/test profile, toolchain facts, and changed files since the previous
memory commit. This context is included in coder and reviewer prompts as
orientation only. The prompt explicitly tells agents that cached memory may be
stale and that correctness, security, and behavior claims must come from the
actual source files and PR diff. The cache is local-only, but it can contain
repo structure, local paths, test command notes, and advisory summaries.

Use these flags to control it:

```bash
agent-loop pr 123 --repo OWNER/REPO --no-agent-memory
agent-loop pr 123 --repo OWNER/REPO --refresh-agent-memory
agent-loop pr 123 --repo OWNER/REPO --refresh-test-profile
agent-loop pr 123 --repo OWNER/REPO --agent-memory-dir .cache/agent-loop-memory
```

Relative `--agent-memory-dir` values are resolved inside the active coder
checkout. Use `--no-agent-memory` or a custom short-lived
`--agent-memory-dir` for sensitive repositories where local cache retention is
undesirable. If a custom memory directory uses the repo-local `.agent-loop`
parent, that parent is ignored automatically so generated memory files are not
accidentally committed. If the previous memory commit cannot be diffed against
the current commit, the loop logs the git failure and treats all tracked files
as changed for that refresh.

## Real Example

This project uses `agent-loop` to improve itself. This command asked Codex to
review existing issue and PR feedback, with both Claude and Gemini reviewing
the result. The work became PR #13:
https://github.com/wwind123/coding-review-agent-loop/pull/13

```bash
~/tools/coding-review-agent-loop/.venv/bin/agent-loop task \
  "Please go over all issue and PR reviews again and see if any future follow-ups are still worth addressing but have not been addressed." \
  --repo wwind123/coding-review-agent-loop \
  --coder codex \
  --reviewer claude \
  --reviewer gemini \
  --dangerous-agent-permissions
```

Read a task from a file or stdin:

```bash
agent-loop task --task-file task.md --repo OWNER/REPO
cat task.md | agent-loop task --task-file - --repo OWNER/REPO
```

## Clarification

Task mode is non-interactive by default. If the coder agent decides the task is too ambiguous and emits `<!-- AGENT_CLARIFY -->`, the command exits with the agent's questions. You can add detail and rerun.

To allow interactive clarification:

```bash
agent-loop task "Add caching to the recent-debates endpoint." \
  --repo OWNER/REPO \
  --interactive \
  --max-clarification-rounds 3
```

In interactive mode, answer the questions on stdin. Finish with a single `.` line or Ctrl+D.

## Auto-Merge

Auto-merge is disabled by default. Enable it explicitly:

```bash
agent-loop pr 123 \
  --repo OWNER/REPO \
  --auto-merge
```

For repositories without the managed exact-head CI contract, auto-merge always
uses the full-board watcher. A merge permit requires a reliable, non-empty
current-head board: the check query and branch protection must be available,
there must be no pending or missing required checks, and all reported checks
must pass. Partial or unavailable snapshots remain fail-closed and are polled;
an otherwise reliable empty board is bounded startup, not success. The live
head is re-read immediately before an exact-head merge proof is sent to GitHub.

`--no-watch-pending-ci` remains parseable for compatibility, but does not
disable the ordinary auto-merge gate; an explicit use with auto-merge emits a
warning. One timeout and attempt budget is shared across watcher polls,
coder-failure rounds, and head-change rounds.
Auto-merge timeout, bounded-startup, and already-exhausted-budget stops retain
their resumable diagnostics and return a non-zero exit. An explicit manual
`--watch-pending-ci` run keeps the same watcher outcomes but stops cleanly
without merging.
Local `--test-command` is an additional local gate, not a replacement for CI.
By default, `--test-command` also runs after coder-created or coder-updated
changes before reviewer rounds, so reviewers are less likely to spend rounds
on code that already fails the configured local test command. Use
`--no-pre-review-tests` to keep `--test-command` as a post-approval gate only.

Failing GitHub checks always block approval and can route back to the coder. Pending or unavailable GitHub checks are treated as an external wait state rather than actionable coder feedback: if every reviewer approves the code and only GitHub checks are pending/unavailable, the loop posts a comment and stops with a clear message instead of erroring or starting another coder/reviewer round. If those checks later pass, manual merge is fine and rerunning is optional unless you want agent-loop to re-check or automate the final step. With `--auto-merge`, the loop instead keeps watching until checks resolve before merging. With active `--managed-ci`, the informational comment is retained but the loop falls through to dispatch the final exact-head workflow; ordinary pending/unavailable checks are not qualification.

Managed exact-head qualification is approval-gated. Reviewers approve a
repaired candidate when no code-level blocker remains even though the managed
qualification job has not run; that approval advances the orchestrator to the
authoritative dispatch. A reviewer disposition cannot itself clear the durable
machine obligation, and the missing post-approval dispatch is not a reason to
withhold code approval.

### Managed exact-head CI

#### Installed repository contract

The checked-in workflow for `wwind123/coding-review-agent-loop` is the
base-branch security boundary for managed CI. It declares the literal
`AGENT_LOOP_MANAGED_CI_V2`,
`AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1`, and
`AGENT_LOOP_MANAGED_CI_VISIBLE_INTENT_V1` capabilities and subscribes to
exactly `opened`, `synchronize`, `reopened`, and `unlabeled` pull-request
activities. A trusted same-repository draft on `main` with a reserved
`agent-loop/managed-*` head may suppress the opening matrix before its label is
visible; synchronization and reopening additionally require the active
`agent-loop-managed` label. `unlabeled` always selects ordinary CI. Forks,
non-drafts, wrong authors or branches, missing trust configuration, and
malformed or unavailable event data fail open to the Python 3.12 full suite.
No label-addition, readiness, draft-conversion, or arbitrary existing-PR
adoption event is part of this installed route.

The manual dispatch inputs are `protocol_version`, `pr_number`,
`expected_head_sha`, and `managed_nonce`. A managed dispatch is named
`managed-ci-v2 nonce=<managed_nonce>`; ordinary runs use the workflow name.
All four empty inputs mean ordinary manual CI; partial inputs are rejected. The base workflow resolves
`AGENT_LOOP_MANAGED_ACTOR` to the live authenticated identity and requires the
initiating and re-run actors to agree. It then validates the live repository,
open draft, trusted author, reserved branch, managed label, `main` base, exact
head, current workflow revision, and exactly one fresh generation-scoped
handoff record in one of the paired dispatch-requested, attached, or completed
states. Prepared-only, duplicate, ambiguous, stale, foreign, and drifted
records are rejected. The base runner checks `created_at` against its current
epoch time: the record may be at most 15 minutes old and at most 5 minutes in
the future for clock skew. A completed no-status retry is accepted only when
it names the immediately preceding terminal attempt of the same Actions run.

Validation alone exposes the expected SHA. The exact-head job checks out that
SHA, verifies `git rev-parse HEAD`, installs `.[dev]` on Python 3.12, and runs
`python -m pytest` once. The publisher writes `final-ci/exact-head` only for
that validated SHA and correlates its terminal description and Actions URL to
the managed nonce, run ID, and current attempt. It writes nothing when
authorization fails before a target exists, and publishes failure when
checkout or tests fail. Pushes to `main` and ordinary manual dispatch remain
full-suite paths.

For this sole-maintainer repository, install and qualify the workflow before
using managed CI: set `AGENT_LOOP_MANAGED_ACTOR` to `wwind123`, authenticate
`gh` as that account, run the read-only preflight, and pass
`--managed-ci-trusted-actor wwind123 --allow-unprotected-managed-ci` on every
applicable issue-created invocation while `main` is unprotected. The waiver is
never implicit, does not weaken other repositories' defaults, does not enable
existing-PR adoption, and does not replace a head-guarded merge. Keep the
serial queue on ordinary CI until live qualification proves suppression,
exact-head status publication, ordinary recovery, and post-merge `main` CI.

For a plan-first issue implementation, the complete invocation is:

```bash
agent-loop issue <issue-number> --repo wwind123/coding-review-agent-loop \
  --plan-first --implement-after-approval --auto-merge --managed-ci \
  --managed-ci-trusted-actor wwind123 --allow-unprotected-managed-ci
```

#### Read-only readiness preflight and unprotected override

Run this before starting agents or creating a PR:

```bash
agent-loop managed-ci preflight --repo OWNER/REPO --base main --trusted-actor LOGIN
```

The command only reads repository, identity, Actions-variable, workflow, branch
protection, and ruleset APIs. It reports visibility, the authenticated and
configured actors, workflow/recovery completeness, and whether GitHub can
independently enforce `final-ci/exact-head`. Its deterministic exit values are
`0` (strict-ready), `10` (known non-ready, ordinary fallback, invalid contract,
or override-eligible), and `11` (ambiguous API/probe failure). A private-GitHub
Free protection API response that says to upgrade or make the repository public
is reported as a plan limitation, not as a missing actor permission.

Strict means either classic required-status protection includes
`final-ci/exact-head` *and* enforces administrators, or an active applicable
ruleset requires that context without bypass actors. Context-only classic rules
with admin bypass, evaluate/disabled rulesets, and rulesets with bypass actors
are voluntary rather than strict. Existing-PR adoption always requires strict,
non-bypassable enforcement.

For an otherwise eligible authenticated plan-first issue-created or
pre-creation `managed-pr` v2 invocation, `--allow-unprotected-managed-ci` may
waive only that protection prerequisite.
It must be written on every invocation; an old label, audit trailer, or
`--dangerous-agent-permissions` never re-enables it. The coder records the
override nonce in the PR body and agent-loop warns locally. This is not suitable
for shared repositories or unattended automation: GitHub cannot prevent a
manual merge, another automation, a compromised credential, or an agent-loop
defect from bypassing the voluntary gate. The flag is rejected for adoption and
does not waive identity, workflow, nonce, exact-head qualification, or
`--match-head-commit` merge checks.

This intentionally tightens the issue-created v2 path for workflows that can
suppress `pull_request` CI. A repository that previously ran that v2 flow
without non-bypassable GitHub protection now uses ordinary CI unless the
operator supplies the explicit waiver for that invocation. If activation later
cannot prove readiness, a later waiver is omitted, or the authorization
checkpoint cannot be written, agent-loop removes `agent-loop-managed` and
waits for the workflow's `unlabeled` recovery CI instead of treating a
`no_checks` board as mergeable.

The explicit fresh issue-created command is printed only for an authenticated
issue-created missing or stale authorization whose issue scope is known and
whose invocation includes the unprotected waiver. It is not a remedy for an
unreadable or foreign label event, a missing waiver/protection prerequisite, a
tuple or publication race, an intent-ledger failure, or an adopted/source-managed
PR; those states report the failed prerequisite and require that state to be
restored.

Issue-created unprotected authorization is persisted independently of coder
test evidence. After the live PR number and opening tuple are authenticated,
agent-loop writes one versioned authorization record as an actor-authored PR
comment before validating `tests_run` or test-observation locations. The PR
body nonce remains only an opening-handshake input; PR-body copies, issue
comments, malformed records, coder-authored comments, and mismatched actor,
repository, issue, base, label event, or exact-head fields never authorize
managed CI. A rejected post-PR test report therefore stays rejected without a
handoff or approval, while the same authorized head remains resumable.

If structured response validation fails before a PR number is accepted, no
creation authorization is synthesized from the rejected response. Recovery
depends on the live protection assessment. For a voluntary or plan-limited
base, a later issue-mode recovery uses the explicit fresh grant:

```bash
agent-loop issue <issue-number> --managed-ci --managed-ci-fresh \
  --managed-ci-trusted-actor <login> --allow-unprotected-managed-ci
```

When entering from PR mode, the issue scope must be explicit:

```bash
agent-loop pr <number> --managed-ci --managed-ci-fresh \
  --managed-ci-issue <issue-number> \
  --managed-ci-trusted-actor <login> --allow-unprotected-managed-ci
```

This is a new operator authorization, not recovered creation provenance. It
requires the authenticated actor, same-repository open PR, reserved issue
branch, selected base, live exact head, expected issue association, an
actor-owned managed-label history, and the canonical approved-plan scope when
one applies. PR mode fetches the issue and its canonical plan comments from
GitHub and requires a server-observed issue timeline association to the PR;
PR-body closing text is corroboration, not authority. Identical retries reuse
an existing valid creation, fresh, or continuity authorization at that exact
head instead of publishing a competing grant; conflicting records,
ambiguous provenance, or changed live state fail closed before labels,
readiness, review dispatch, qualification, or merge writes.
The grant records the live voluntary or plan-limited protection assessment,
and the PR tuple, managed-label event, and authorization-comment set are read
again immediately before publication. Managed issue recovery from a legacy
association does not backfill a canonical handoff for a response whose report
was rejected; authorization and coder evidence remain separate checkpoints.

Both managed-CI recovery branches — the fresh authorization above and the
ordinary same-PR resume — refresh the child issue snapshot and the
authoritative in-process parent issue snapshot before reading either one's
comments, so a caller-supplied snapshot that predates plan approval cannot
defeat recovery. A staged decomposition child normally carries only its
issue-to-PR handoff record, with the approved plan round living on its parent
issue, so when the child recovery is unavailable and reports no matching
candidate the canonical plan may be recovered from that refreshed parent
snapshot. That predicate is exactly what the recovery model reports, and it is
broader than "the child has no record with the handoff hash": a legacy
free-form child record that does carry the hash but is rejected on its own
derived plan subject also leaves no surviving candidate, and the fallback is
deliberately permitted in that case because the subject-rejected record is
never adopted and the handoff plan hash still has to match whatever the parent
yields. Only divergent accepted child records — several records that match the
hash but disagree on the plan text — report a matching candidate, and those
keep failing closed without consulting the parent. The fallback is hash-only,
passes no expected subject, and reaches the parent solely because the child's
authenticated fresh-phase identity named it; the handoff plan hash remains the
only binding. Documentation never excuses an implementation defect; the source
remains authoritative.

For a strictly protected base, no unprotected authorization record is needed
and `--managed-ci-fresh` is not a valid remedy. If the response was rejected
before its PR number was accepted, use the ordinary same-PR issue/PR discovery
and managed-CI resume path instead; it reauthenticates the strict PR tuple and
does not rerun implementation. A strict draft/unlabeled re-entry additionally
requires an actor-owned historical `agent-loop-managed` label event before it
reapplies the label. The strict path never mints or accepts a waiver nonce
merely because `--allow-unprotected-managed-ci` was supplied.

To retry an interrupted issue-created managed draft on an unprotected
repository and preserve automatic merging, use the explicit per-invocation
PR-mode command:

```bash
agent-loop pr <number> --auto-merge \
  --managed-ci-trusted-actor <login> --allow-unprotected-managed-ci
```

For a manual-merge resume, replace `--auto-merge` with `--managed-ci`; that
mode publishes a fresh SHA-bound qualification and never calls the merge API.
This is supported resume, not retroactive adoption. The live PR must still be
open, a draft, same-repository, authored by the authenticated trusted actor
(login and immutable ID), on the reserved `agent-loop/managed-*` ref, with the
same live base/head and an active `agent-loop-managed` timeline event applied
by that actor. A prior actor-owned override audit is only provenance; its
editable body nonce never grants authority or gets reused. The audit's
repository/base fields must match, and missing, malformed, ambiguous, or raced
audit/timeline data fails closed to deliberate ordinary release.

The preferred recovery for a canonical issue handoff is to rerun the original
`agent-loop issue <number>` command. That preserves its planning and
implementation shape and reuses the authenticated issue-to-PR association;
`agent-loop pr <number>` is the direct fallback when the PR is already known.
An authenticated ready/unlabeled issue-created or `managed-pr` PR may be
reconstructed only when the new invocation has an explicit `--managed-ci`.
An implicit `--auto-merge` invocation leaves that state ready and unlabeled,
prints the exact flow-preserving retry, and performs no label, body, comment,
dispatch, or readiness write. Draft/labeled and ready/unlabeled are the normal
accepted lifecycle states; an explicit `--managed-ci` retry may also re-admit
the draft/unlabeled state left by a failed explicit managed run. On a strict
base, that state is reauthenticated from the strict PR tuple plus the
actor-owned historical label event; it does not require or create the
unprotected authorization record. Other mixed states stop before agents run.

Base resolution records whether the value came from an explicit `--base`, the
repository default, or live PR metadata. This base provenance crosses the
issue-to-PR boundary unchanged. If an inherited repository default differs
from the live PR base, the run stops before workdir setup and prints a retry
with the live base explicitly supplied. An operator-provided `--base` remains
authoritative. Recovery commands replay parser-valid invocation tokens,
including repeated common options and shell metacharacters, and remove only
options that the target subcommand cannot accept. Historical audit records and
old qualification ledgers are provenance only: recovery mints fresh authority
and never asks the operator to delete durable records.

A successful resume records a fresh audit, nonce, and invocation intent
generation. Earlier ledger entries and attached workflow runs remain history:
even a queued, successful, failed, or rejected prior run is logged as the
previous invocation's outcome and is never adopted by the new dispatch. If
safe managed resume is unavailable, agent-loop selects ordinary recovery only
when the base workflow proves an unlabeled pull-request route. It baselines
current-head run IDs before label release, retains the draft, and observes
startup for at most `--ci-startup-timeout-seconds` (default 120). A missing,
queued-only, or jobless run is never success: the draft remains unmerged and
the local terminal prints a deterministic shell-quoted PR resume command. A
new post-`unlabeled` run must complete successfully and the exact head must
have a non-empty complete passing board. It does not accept pre-release green
checks, an empty rollup, or a different SHA. Only the invocation-owned fallback draft
can be made ready, and only an auto-merge invocation can take this
deliberate ordinary-recovery path; explicit managed manual mode fails closed
instead. Unrelated or intentional drafts remain drafts. Agent-loop checks the
exact head before and after `gh pr ready`, then performs one final live-head
read before merging with `--match-head-commit`; if that guarded merge fails,
the PR remains ready and unmerged for a safe retry.

Suppression-capable v2 workflows must additionally advertise
`AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1` and subscribe their
`pull_request` trigger to `unlabeled`; when a managed label is released,
ordinary CI must emit a new check on the current head. Agent-loop refuses to
merge a managed-labeled PR on a `no_checks` board.

#### v2 authenticated opening and exact-head provenance

V2 is an explicit opt-in migration that prevents the opening-event matrix race.
Set the base repository Actions variable `AGENT_LOOP_MANAGED_ACTOR` and pass
the same login with `--managed-ci-trusted-actor`. Agent-loop verifies both that
variable and the authenticated `gh api user` login/ID, reads the workflow from
the resolved base ref, and requires its `AGENT_LOOP_MANAGED_CI_V2` marker.
Missing/mismatched trust disables v2 suppression; marker-free repositories use
ordinary CI and complete v1 workflows keep their post-open label handoff.

Participating repositories must make the v2 workflow contract literal and
complete. The committed `.github/workflows/ci.yml` must contain all of these
feature markers exactly: `AGENT_LOOP_MANAGED_CI_V2`, `workflow_dispatch`,
`managed_nonce`, and `final-ci/exact-head`. Its `workflow_dispatch` inputs must
be named `protocol_version`, `pr_number`, `expected_head_sha`, and
`managed_nonce`. The dispatched workflow must set this exact run name (where
`managed_nonce` is the input):

```yaml
run-name: managed-ci-v2 nonce=${{ inputs.managed_nonce }}
```

The final publisher must post the `final-ci/exact-head` commit status to
`expected_head_sha`, with a semicolon-delimited description containing these
exact tokens (in any order):

```text
nonce=<managed_nonce>;run_id=<github.run_id>;attempt=<github.run_attempt>
```

Its `target_url` path must end in `/actions/runs/<github.run_id>` (the host may
be github.com or a GitHub Enterprise Server hostname). Agent-loop
matches each token and the final URL path segment exactly; a matching prefix,
another run, or an earlier rerun attempt is not accepted. The publisher should
run under `github-actions[bot]` (or the configured trusted actor when posting
through that account).

For auto-merge issue work, a v2 preflight gives the coder an atomic creation
intent: create or verify `agent-loop-managed`, use the reserved
`agent-loop/managed-<issue>` branch, then run `gh pr create --draft --label
agent-loop-managed --body-file <path>`. The workflow can suppress later
reopened/synchronize matrices only for the complete tuple: same-repository
head, trusted REST author, reserved branch, draft, and label. The opening event
is necessarily evaluated before GitHub's separate label write and therefore
uses the same tuple without the label; agent-loop must apply the label before
continuing.
A contributor-editable label, branch, or body alone is never trusted. Forks
and ordinary PRs retain regular CI unless they are explicitly adopted as
described below.

For an unprotected override, the coder carries the preflight-minted nonce in
one canonical body record containing only that nonce; it must not add another
reserved body record. During the same issue-created handoff, agent-loop checks
the draft's repository/base/head/branch/label/author tuple and one issue
closing reference, re-reads it to catch races, then publishes the richer
actor-authored PR-comment authorization record before post-PR report checks.
That record has its own versioned schema and binds the repository, issue, PR,
actor login and immutable ID, base, exact head, waiver, nonce, and managed-label
event. It is the only durable source for unprotected issue-created resume.

When an orchestrator-dispatched coder repair advances the PR, automatic
authority movement is allowed only through one unique, gap-free chain of
actor-authored continuity comments. Each continuity record binds the
predecessor authorization comment, predecessor head, new exact head, and the
durable review/coder round metadata identities. The referenced records are
re-parsed, must postdate the predecessor authorization, and must order the
blocking review for the predecessor head before the immediately following
coder record for the new head; unrelated or
malformed metadata and authorization-comment fallbacks are rejected. A push,
branch name, PR body, author, label, draft state, missing link, fork, or race cannot extend
authority; recovery prints the explicit fresh-authorization command instead.

A merge-conflict resolution round is the single exception, because it runs no
reviewer. When the head conflicts with the base branch, agent-loop skips the
board and routes the round to the coder, so continuity accepts that head move on
the tool-owned merge-conflict obligation instead of a review pair: exactly one
actor-authored coder round metadata record for the new exact head, newer than
the predecessor authorization, carrying the orchestrator-minted merge-conflict
obligation that no agent response can add or classify. The continuity record
binds that one record, and resume re-parses and rechecks the same shape. A head
advance with neither an ordered review/coder pair nor that obligation still
fails closed, and the board must still approve the exact final head before
qualification or merge. This removes the second `--managed-ci-fresh` grant that
an ordinary base-branch move used to require, without widening what counts as an
approval.

#### Creating a managed PR from an existing branch

When code is already pushed to the repository but no PR exists, use the
pre-creation mode instead of creating an ordinary PR and adopting it later:

```bash
agent-loop managed-pr \
  --repo OWNER/REPO \
  --head fix/prepared-change \
  --base main \
  --title "Fix prepared change" \
  --body-file /path/to/pr-body.md \
  --managed-ci \
  --managed-ci-trusted-actor LOGIN \
  --reviewer claude \
  --reviewer agy \
  --review-parallel
```

The command resolves the exact SHA of `--head`, rejects a source SHA that
already has an open PR, and runs managed-CI readiness before creating anything.
It then creates a unique `agent-loop/managed-direct-*` alias at that SHA, opens
a draft, applies `agent-loop-managed`, and enters the ordinary PR review loop.
Use `--managed-ci` to leave a successfully qualified PR ready for the printed
head-guarded manual merge, or replace it with `--auto-merge` to merge the
qualified head automatically.
If the source moves during handoff or draft labeling fails, the partial draft
is closed and the reserved alias is removed. The original source branch is
never changed or deleted.

`--body-file -` reads the PR body from stdin; omitting `--body-file` creates an
otherwise empty body with only the hidden source audit marker. On a repository
whose preflight reports `override_eligible`, add
`--allow-unprotected-managed-ci` to this invocation. Its nonce is embedded in
the newly created body and correlated in memory exactly like issue-created v2.
The waiver is safe only under the same single-operator constraints documented
above. `managed-pr` never adopts an already-open PR and does not weaken the
strict-only `--managed-ci-adopt-existing-pr` path.

#### Optional adoption of an existing PR

Existing same-repository PRs, whether draft or ready and regardless of their
author, may be adopted only when all of the following are true:

```bash
agent-loop pr <number> --auto-merge \
  --managed-ci-trusted-actor <trusted-login> \
  --managed-ci-adopt-existing-pr
```

This is a distinct opt-in capability. The base-ref workflow must contain the
literal `AGENT_LOOP_MANAGED_CI_V2_PR_ADOPTION` marker in addition to the normal
complete v2 markers. Its absence never changes issue-created v2 drafts; an
adoption attempt simply retains ordinary CI. Agent-loop reads this workflow
from the live base ref, rejects forks, base/head races, closed PRs, and a PR
with `agent-loop-managed-opt-out`.

Before it applies or trusts `agent-loop-managed`, agent-loop must be able to
inspect the base branch's required-status-check protection and find
`final-ci/exact-head`. It publishes a pending `final-ci/exact-head` guard on
the live SHA before suppression and repeats that guard after each adopted head
change. Missing, inaccessible, malformed, or ambiguous protection is
unsupported: no label mutation occurs. This keeps every suppressed adopted
head non-mergeable until nonce/run/attempt-correlated qualification.

The workflow is the security boundary. For every adoption evaluation on
`opened`, `reopened`, `labeled`, `unlabeled`, `synchronize`,
`ready_for_review`, and `converted_to_draft`, it must query the complete issue
event timeline from the base workflow and suppress only if the currently active
managed-label application's actor login and immutable ID match the actor named
by `AGENT_LOOP_MANAGED_ACTOR`. It must also require no opt-out label. API,
pagination, or provenance failures, or a collaborator relabeling the PR, must
fail open to the ordinary matrix. This prevents triage/write collaborators from
suppressing CI by manipulating labels.

The handshake records the active label event ID. A trusted existing label is
reused and never removed by that invocation; a label created by the invocation
is removed only if its exact application is still current on a terminal
unqualified exit (max rounds, agent/reviewer error, interrupt, ordinary
non-zero exit, or head movement). A relabel race is left untouched and ordinary
CI resumes. After qualification/merge the label is retained.

To opt out durably, apply `agent-loop-managed-opt-out` and remove
`agent-loop-managed`: this immediately restores current-head CI and prevents a
later explicit adoption. Removing only `agent-loop-managed` also restores CI
immediately, but a later explicit adoption may apply it again.

V2 dispatches `ci.yml` at the base ref with protocol version, PR, exact SHA,
and a random nonce. The workflow run name, checkout verification, per-PR/SHA
concurrency group, and always-running publisher must carry those inputs. Its
`final-ci/exact-head` status description includes nonce, run ID, and attempt,
and targets that run. Agent-loop persists one actor-owned hidden
`AGENT_MANAGED_CI_INTENT_V2` comment, rediscovers it after interruption, and
converges same-nonce duplicate dispatches to the newest surviving run. It
accepts neither green nor red same-context statuses unless publisher, nonce,
attached run, and latest attempt all correlate. A failure is reported from the
validated run's failing jobs, not base-ref PR checks.

The workflow fullmatches the whole stripped intent comment body, so a trusted
comment can never carry prose or a second record alongside the authorization.
When the base workflow also advertises
`AGENT_LOOP_MANAGED_CI_VISIBLE_INTENT_V1`, its envelope additionally admits one
fixed visible line ahead of the hidden record, exactly
`Managed CI authorization for exact head <sha>.` followed by a blank line, and
the workflow rejects the record when that SHA differs from the payload's
`expected_head_sha`. Agent-loop emits that line only for a base workflow that
advertises the capability; against an older workflow, which accepts only the
bare record, the comment stays marker-only. The bare form remains accepted by
the current workflow for this transition.

The v2 intent lifecycle is deliberately limited to `prepared`,
`dispatch-requested`, `attached`, and `completed`. `prepared` and
`dispatch-requested` always serialize `run_id: null` and `run_attempt: null`,
including when an earlier invocation or round had an attachment. Minting a
fresh nonce clears all attachment and terminal fields while retaining only the
authenticated contract and invocation-generation provenance. Same-nonce
retries retain a non-excluded attachment even when a workflow-runs read is
temporarily empty; the waiter refreshes that run and no second dispatch is
issued. Only an explicitly excluded terminal attempt may be cleared for
replacement.

After correlated success, auto-merge applies the short-lived
`agent-loop-exact-head-qualified` label, marks the PR ready, rechecks its head,
and merges with `--match-head-commit`. Explicit `--managed-ci` never applies
that bare label and never calls the merge API: it releases the managed label,
marks an issue-created draft ready, writes a SHA-bearing
`AGENT_LOOP_MANAGED_CI_QUALIFIED_V2` audit comment, and prints:

```bash
gh pr merge <number> --repo OWNER/REPO --merge --match-head-commit <qualified-sha>
```

The PR stays open for a human. A later head change invalidates the result. A
rerun of a successful issue-created manual result first makes the PR draft
again, suspending the earlier manual command; if reconstruction fails, rerun
managed qualification or restore readiness manually and use the old guarded
command only after confirming that exact SHA is still live. For an explicitly
unprotected run, the audit and terminal warning state that GitHub cannot force
the human to use the qualified SHA after agent-loop exits. Explicit mode
requires complete v2; it rejects v1 instead of silently claiming qualification.

A v2 issue-created PR has zero billed routing jobs at opening, zero hosted
minutes per intermediate revision, one final matrix, and one rounded
publisher/aggregate minute—about 11–14 minutes for a 10–13 minute matrix.
`agent-loop pr <n>` has already paid the ordinary opening matrix and remains
roughly the earlier 20–25-minute shape plus recovery work. Keep routing,
aggregate, and qualification telemetry separate for billing comparisons.

For the legacy v1 contract only, under `--auto-merge`, agent-loop automatically detects a same-repository PR
whose `.github/workflows/ci.yml` advertises the `agent-loop-managed`,
`final-ci/exact-head`, and `expected_head_sha` contract. It applies the managed
label before iterative review, allowing that repository to suppress full
hosted test matrices on intermediate heads. Repositories without those markers
retain the ordinary CI behavior above; a partial contract fails closed.

During managed rounds, the intentionally pending final aggregate and missing
hosted matrix contexts are not presented as actionable review failures.
Actually observed non-final failures remain visible. If a configured
pre-review `--test-command` passes, agent-loop also publishes the non-required
`agent-loop/round-readiness` status on that head; it never publishes readiness
without running that command.

After every required reviewer approves one live head, agent-loop dispatches the
repository's `CI` workflow with the PR number and that exact expected SHA. In
v2 it polls the exact-head status and the validated nonce/run/attempt together.
A correlated terminal status is authoritative; a completed workflow without
that publisher status is confirmed by an immediate re-read and one bounded
subsequent poll, then stops resumably without synthesizing a status. This
applies to every workflow conclusion, including cancellation, timeout,
action-required, success, neutral, skipped, failure, and unknown values—workflow
success alone never qualifies the head. A real correlated failure routes back
to the coder, and a moved head restarts review. A passing aggregate is merged
with `--match-head-commit`, so a different head cannot inherit the approval or
CI result. The terminal-without-status stop records the router-compatible
`state=completed` plus a separate `terminal_outcome: "no-status"` round field,
the exact run attempt, and accumulated excluded-attempt history; it never
publishes the unsupported `terminal-no-status` lifecycle. A later
higher-attempt rerun is accepted when the excluded attempt is known; if the
attempt is missing, the entire run ID remains excluded and only a different
fresh run ID can be correlated. An unchanged head can also dispatch a fresh
same-nonce run. If an adopted PR stops this way, its
invocation-owned managed label is released and the next invocation
re-adopts/reapplies managed mode. A corrected head requires a fresh exact-head
review. The managed route is selected independently of `--watch-pending-ci`;
neither that flag nor `--no-watch-pending-ci` replaces exact-head
qualification with ordinary full-board watching.

Historical authorization records are not rewritten. A pre-existing record
with an unsupported lifecycle value can still block an older router that
validates every trusted record before nonce filtering; remediate that consumer
or preserve its strict validation when rolling out the compatible producer.
The supported nonce-scoped router validates only the requested nonce's
lifecycle, so unrelated historical generations cannot deadlock a fresh
dispatch. Neither behavior infers qualification from `completed` or workflow
success: only a correlated exact-head status with all required checks passing
can reach the merge path.

#### Managed-CI GitHub CLI compatibility

The managed-CI API calls use the GitHub CLI 2.45-compatible pagination
behavior. Paginated array endpoints are decoded as one flat JSON array;
concatenated object pages (such as workflow jobs) are decoded page by page. A
malformed page or entry is unavailable and is never treated as an empty or
absent timeline. The managed-label timeline is therefore unreadable-is-not-absent,
and every ownership, readiness, and merge gate fails closed when it cannot be
revalidated.

Ordinary fallback is authorized only by the centralized release path. Before
removing the managed label it records the IDs of existing workflow runs for
the exact head. Recovery accepts only a newly observed `pull_request` run for
that same head, together with a passing and complete ordinary check board.
There is deliberately no local-clock or timestamp ordering claim: GitHub run
IDs and the exact-head check are the available provenance, and no run observed
is not treated as a successful recovery.

Reconstructed reserved drafts require the authenticated user, the configured
CLI trusted actor, and the current `AGENT_LOOP_MANAGED_ACTOR` repository
variable to agree. Readiness and merge remain fail-closed, with `gh pr ready`
and `--match-head-commit` guarded by a fresh exact-head and inactive-label
provenance check. The workflow must advertise the `pull_request: unlabeled`
recovery trigger for this path.

### Watch pending CI

For repositories not using managed exact-head CI, ordinary `--auto-merge`
always enters the full-board watcher, regardless of the effective
`--watch-pending-ci` value. `--no-watch-pending-ci` remains parseable for
compatibility and produces a warning when explicitly supplied with auto-merge;
it does not weaken that gate. An explicit
`--watch-pending-ci` without `--auto-merge` watches ordinary checks after
approval and reports merge-ready without merging. It does not activate
suppression or replace managed exact-head qualification; when `--managed-ci` is
active it is inert for that managed route.

When active, it foreground-polls the full PR check/status/required-check board
using the existing timeout and poll interval controls. It does not call
reviewers or the coder while checks remain pending. A completed actionable
failure resumes the normal coder loop with the check name, conclusion, and
URL. A merge permit requires a reliable, non-empty current-head board with no
pending or missing required checks, and all reported checks passing. Unavailable
or partial queries, unavailable protection, pending checks, and missing
required checks remain fail-closed polling states; only a reliable empty board
uses the bounded `not_started` startup outcome. A new head is re-reviewed, and
the final-round CI-failure path receives one bounded extra round.

One deadline and attempt budget is shared across all watcher entries in an
invocation, including rounds caused by a CI failure or head change. If that
budget is already exhausted, the loop records that no fresh poll occurred.
Auto-merge timeout, `not_started`, and pre-poll exhaustion retain resumable
diagnostics and exit non-zero; explicit manual watch-only runs stop cleanly
with exit zero. A passing board is consumed immediately by the exact-head merge
proof, so it does not enter a second CI wait.

The watch runs synchronously with interruptible `sleep` subprocesses, so Ctrl-C
and restarts leave no hidden worker. Timeout and transient API snapshots remain
bounded and print a shell-quoted rerun command only locally; GitHub comments do
not repeat invocation arguments. Dry-run previews the watch without polling,
sleeping, resuming agents, or merging. `--no-watch-pending-ci` does not disable
the ordinary auto-merge watcher.

### External CI infrastructure stalls

GitHub-hosted runner capacity incidents can leave a check-run `queued` indefinitely with no job ever starting, or cause it to be cancelled before execution because a runner could not be acquired. Left unhandled, a coder round could otherwise run an unbounded `gh run watch` and consume an entire session without producing a result.

Two independent, complementary mechanisms bound this instead:

- **Detection and classification.** Every `get_pr_checks` fetch classifies each check-run against `--ci-queued-grace-seconds` (default 1200): a check still `queued`/`pending` with no job started past the grace period is `queued_too_long`; a check-run cancelled (or `startup_failure`) with no real start is `runner_unavailable`. A check is only ever treated as a full stop when the *whole* check board is wholly infrastructure-blocked — every failing/pending check is a classified stall, no required check is missing, and branch protection and the check query both succeeded. A single genuinely failing test, a never-reporting required check, or a partial API failure never takes this exit; it falls back to ordinary pending/failing handling instead. When the whole board is wholly blocked, the loop posts a comment explaining that no code change is required and no merge was attempted, and stops in a state that is safe to resume later — rerun the same command once GitHub Actions runners recover. With `--auto-merge`, the same predicate governs whether the CI wait loop exits early with that message instead of merging.
- **Bounded coder observation policy.** Independent of classification, every coder prompt forbids `gh run watch`, `gh pr checks --watch`, or any other unbounded CI wait. A coder may take at most 3 status snapshots, spaced at least 30 seconds apart, for at most 120 seconds of total CI observation per turn; past that bound it must return its terminal blocking response immediately, naming the affected check/run and noting that work should resume once GitHub Actions runners recover. This applies even if a reviewer's finding is not recognized as a canonical stall-only item, so a coder round can never wait indefinitely on GitHub Actions infrastructure.

Reviewers see the same classification (an "External CI infrastructure stalls" section in the PR checks context) and are instructed not to record a classified stall as a blocking code item; any other failing or never-reporting check remains ordinary review work.

### Merge conflicts

GitHub's own mergeability computation, not just the current-head CI check, is
evaluated before spending reviewer time or an auto-merge CI wait: a PR whose
branch has drifted out of sync with its base can otherwise pass every
reviewer round only to fail to merge, or run CI against a head that stops
being relevant the moment the base advances.

- **When it's checked.** The loop probes `gh pr view --json
  mergeable,mergeStateStatus,headRefOid,baseRefName` at the start of every
  review round (before any reviewer is invoked) and again right before
  fetching GitHub PR checks / attempting `--auto-merge` (the "merge gate").
  `--auto-merge`'s CI wait also re-checks mergeability on every poll, so a
  conflict that appears mid-wait stops the wait immediately instead of
  polling a check on a head that can no longer merge.
- **Classification.** A `mergeStateStatus` of `DIRTY` or a `mergeable` of
  `CONFLICTING` is a confirmed conflict, checked first so it wins even if the
  other field is null. A `mergeable` of `MERGEABLE` is mergeable. Everything
  else — including GitHub's own `UNKNOWN` (still computing), a non-zero `gh`
  exit, or unparsable output — is `unknown` and is treated exactly like
  `mergeable`: it never triggers a coder round on its own. An explicit
  `UNKNOWN` is retried up to `--mergeability-poll-attempts` times (default 3),
  `--mergeability-poll-interval-seconds` apart (default 5), before settling as
  `unknown`.
- **What happens on a confirmed conflict.** Reviewers are skipped for that
  round, no GitHub PR checks are fetched, and no CI wait or `gh pr merge` is
  attempted. The coder is dispatched with a dedicated prompt naming the
  observed base branch and head SHA, instructed to sync the branch, merge
  `origin/<base>`, resolve every conflict, run relevant tests, commit, and
  push to the same PR — never opening a new PR, never waiting on CI, and
  never force-pushing over unrelated work. Any other genuinely unresolved
  reviewer items are carried into the same round.
- **After a resolution push.** The next round re-probes mergeability from
  scratch; once GitHub reports `mergeable` (or `unknown`), the synthetic
  conflict item clears itself and normal reviewer/CI flow resumes against the
  new head — prior approvals and checks from the old head are never reused
  for merging. If the head is still unchanged the next time a conflict round
  would be dispatched (the coder made no progress), the loop stops cleanly
  with an explanatory comment instead of looping.

### Unchanged-head follow-ups

A PR follow-up coder turn that leaves the PR head
unchanged is counted. After two consecutive such turns the loop stops with a
human-review error instead of starting another review of the same diff, which
could only repeat the same verdict until `--max-rounds` ran out. A turn that
moves the head resets the count, and the count belongs to one head: a round
that starts on a different head, for example after an external push, starts
from zero. For a planning child, the error also names the
signed child-plan supersession route and the issue-mode rerun command, because
a finding that requires re-planning can never be satisfied by a PR-mode coder
turn.

### Focused, bounded local test selection

A same-PR follow-up scoped to a wording correction in two files does not
justify pulling in a `tests/test_server.py`-class suite (hundreds of
unmarked FastAPI/database/SSE tests) or a `pytest tests/ --ignore=...` list
that amounts to nearly the whole repository. A coder that does this and then
backgrounds the run and polls it — via a shell loop watching a process ID or
a task-output file — consumes the session for many minutes with no visible
progress and leaves manual interruption as the practical recovery path. This
is distinct from [External CI infrastructure
stalls](#external-ci-infrastructure-stalls): that section bounds waiting on
*GitHub Actions* infrastructure; this one bounds the *local* test command a
coder chooses to run and how it runs it.

Every coder prompt built by `coding_review_agent_loop.prompts` requires:

- **Proportionate selection.** Tests must be chosen for the files actually
  changed and the reviewer item being addressed, preferring the repository's
  verified focused test command from the execution profile when one covers
  the change. When the change is narrow, the coder must give a one-line
  rationale for each selected test module tying it to a changed file or
  reviewer item.
- **A breadth prohibition with an escape hatch.** No whole-`tests/` run, no
  `--ignore` list that is effectively the whole suite, and no broad
  server/database/integration/end-to-end suite — unless the change actually
  touches those surfaces, focused tests demonstrably do not cover it, or a
  human or the issue explicitly asked for full-suite verification. Normal
  full-suite verification for a genuinely broad change stays available; only
  the unnecessary or mis-targeted case is prohibited.
- **Foreground execution under a bounded timeout.** Required completion
  tests run in the foreground with visible output and a concrete stated cap
  no greater than the configured finite run-level ceiling (1,800 seconds by
  default). This is a maximum allowance for one individually justified
  command, not a default reason to choose a broad suite. Coders must not launch pytest in the background or
  spawn auxiliary shell loops that poll process IDs, `ps`/`kill -0`/`wait`,
  or task-output files to learn whether a test finished.
- **A valid terminal path on timeout.** If a required test exceeds its
  bound, the coder terminates the run and returns a valid terminal response
  immediately, naming the exact command and the timeout, rather than
  silently waiting or retrying with a broader selection. `build_task_prompt`
  and `build_task_clarification_prompt` document a no-PR `AGENT_STATE:
  blocking` result for exactly this case, so a free-form task turn that must
  stop after a bounded timeout has an ordinary terminal path instead of
  being forced into an `agent_unavailable` report reserved for genuine
  environment/tooling failure.

This per-command coder policy is separate from the three-snapshot,
120-second GitHub CI observation limit and from the orchestrator's optional
`--test-command` parsing and execution path; it does not change either one.
When `antigravity` is the coder, configure
`--antigravity-print-timeout-seconds` above the selected command watchdog plus
`max(300s, 20%)`, and leave additional budget for analysis, edits, reporting,
and other turn work. The run-level ceiling can be raised for a justified long
suite with `--coder-test-command-timeout-seconds`; the default 600-second
whole-invocation deadline may be too short for that selection. Apply the same
headroom principle to any other backend-imposed whole-turn deadline.

The completion-recovery prompt (sent once, when a prior implementation turn
ended without a valid terminal marker and its text suggested deferring to
background work) instructs the coder not to poll or wait on that old job —
no PID watching, no `ps`/`kill -0`/`wait` loop, no tailing its log or
task-output file — but to terminate a known process once if needed and
re-run the command it actually needs in the foreground under the bound.

## Agent Permission Flags

By default, this standalone package does not pass permission-bypass flags to either agent. This is safer for open-source use, but some CLIs may prompt or fail in non-interactive mode unless you provide suitable flags.

For trusted local automation, opt into permission bypasses explicitly:

```bash
agent-loop issue 56 \
  --repo OWNER/REPO \
  --dangerous-agent-permissions
```

This applies:

| Agent | Flag |
|-------|------|
| `claude` | `--dangerously-skip-permissions` |
| `codex exec` | `--dangerously-bypass-approvals-and-sandbox` |
| `gemini` | `--yolo --skip-trust` |

Dangerous permissions do not relax the assigned-checkout rule. They make it
more important: the CLI may allow cross-checkout mutation, but the prompt and
response validation still require coder work to remain in `AGENT_LOOP_WORKDIR`.

You can also provide exact per-agent replacements. Repeat once per token:

```bash
agent-loop issue 56 \
  --repo OWNER/REPO \
  --claude-arg=--permission-mode --claude-arg=acceptEdits \
  --codex-arg=--sandbox --codex-arg=workspace-write --codex-arg=--ask-for-approval --codex-arg=never \
  --gemini-arg=--approval-mode --gemini-arg=auto_edit
```

Providing any `--claude-arg`, `--codex-arg`, or `--gemini-arg` replaces that agent's default entirely. Claude and Gemini prompts include a tool-owned response-file path under `/tmp/coding-review-agent-loop/responses/`; when the file exists and is non-empty, the loop validates and posts that file instead of stdout so CLI diagnostics and tool narration do not leak into GitHub comments. Gemini still supports stdout marker filtering as a fallback. If you pass `--gemini-arg=--output-format --gemini-arg=json`, the loop extracts the JSON `response` field before parsing markers when no response file was written. Fallback stdout is never posted unless the required protocol marker validates.

## Protocol

Agent responses are parsed using HTML comment markers:

```text
<!-- AGENT_PR: 123 -->
<!-- AGENT_STATE: approved -->
<!-- AGENT_STATE: blocking -->
<!-- AGENT_PLAN_STATE: approved -->
<!-- AGENT_PLAN_STATE: blocking -->
<!-- AGENT_CLARIFY -->
```

`AGENT_PR` is required after a coder creates a PR and must be a positive base-10
integer. `0`, negative, empty, and malformed identifiers are rejected before
any GitHub lookup; a final explicit invalid marker is authoritative and is not
rescued by an incidental PR URL. An issue implementation that cannot safely
continue may instead end with `AGENT_STATE: blocking` or a final
`AGENT_CLARIFY`; it stops without PR review and surfaces that state. Review/fix
responses must include a final `AGENT_STATE` marker. Plan-first coder/reviewer
responses use `AGENT_PLAN_STATE` instead. If a response quotes older markers,
the final matching marker is treated as authoritative.

Structured-response runs follow this fallback order:

1. Structured JSON payloads in agent output are authoritative when present.
2. `AGENT_LOOP_META` on orchestrator-posted comments is the canonical resume
   source for the active structured-response round. It carries the current
   ledger, reviewer dispositions, and item-number allocation.
3. Markdown parsing is a compatibility fallback for comments that do not have a
   structured payload or active-round metadata.

Mixed histories are normal during rollout. A thread may contain old raw
markdown comments, newer orchestrator-rendered comments, or both. When
`AGENT_LOOP_META` exists for the current PR head or plan subject, resume uses
that metadata-backed ledger and ignores stale visible item IDs from older heads,
superseded plans, or replayed rounds.
If the PR head advanced without a current-head coder metadata comment, resume
uses metadata-backed active `blocking` and `same-pr` items from the latest
recorded head and sends them to the coder for a structured follow-up before
reviewers run again.

Reviewer responses should use structured JSON first. A PR review starts with:

```json
{
  "schema_version": 1,
  "kind": "pr_review",
  "state": "approved",
  "summary": "short reviewer summary",
  "blocking_items": [],
  "same_pr_followups": [],
  "future_followups": ["future work after approval"],
  "prior_item_dispositions": [
    {"item_id": "item-1", "disposition": "resolved"}
  ]
}
```

A plan review uses `kind: "plan_review"`, `blocking_plan_issues`,
`same_plan_followups`, `future_followups`, and
`prior_plan_item_dispositions`. The JSON state must match the final
`AGENT_STATE` or `AGENT_PLAN_STATE` footer. Blocking reviews must not hide
current-round work in `future_followups`; approved reviews must not contain
active blocking, Same-PR, Same-plan, or carried-forward active items.

Each plan-review finding entry is normally a string. A reviewer may instead
emit a finding object using only the keys `title`, `text`, `issue`, `finding`,
`description`, `summary`, `location`, `evidence`, `rationale`, `impact`,
`required_change`, `recommendation`, `suggested_fix`, and the reviewer-local
labels `item_id`/`id` (#957). The parser flattens such an object mechanically,
in that fixed key order, into one finding string that keeps every prose value
verbatim (labelled values such as `Evidence:` and `Required change:` keep their
label); the local labels are dropped. No repair model runs for this shape, so a
container-type mismatch can no longer discard a well-grounded review. Unknown
keys and non-string values are still rejected.

When a structured response is recognized but fails schema validation, the
terminal error leads with the validation reason and reports `Failure category:
schema-validation`: the rejection is deterministic for that output, but model
output varies between runs, so re-running the same command may succeed.

Reviews are exhaustive. The full and compact PR review prompts and the full and
compact plan review prompts share one static rule: report every defect that can
be independently substantiated on the reviewed head or plan, not only the
first; substantiating one blocking defect does not end the review; and when a
defect is found in a function, code path, or plan step, re-read that whole
function or path and enumerate every other independently evidenced defect
there as separate entries in the same response. Each entry still needs its own
evidence, and speculation or padding with items the reviewer cannot evidence
is forbidden. When a defect genuinely prevents the reviewer from evaluating
the code or plan content behind it, the reviewer says so in that entry's text
and in `summary`, naming what could not be evaluated, so the coder knows
another round is expected; masking must not be claimed merely to stop early.
The rule changes no response schema: masking is conveyed through the existing
finding text and `summary`. Coder, discuss, repair, and decomposition prompts
do not carry it. Its effect on rounds per run is measured with the
[review contract comparison](#review-contract-comparison).

Published prior-item dispositions put the current status immediately after the
item ID, before evidence and the original finding:

```text
- [item-3] RESOLVED
  - Original finding: Blocking issue from Anthropic Claude, round 1: Add coverage.
- [item-16] SAME-PR: The error diagnostic still needs correction.
  - Original finding: Same-PR follow-up from Anthropic Claude, round 2: Preserve diagnostics.
```

`BLOCKING`, `SAME-PR`, and `SAME-PLAN` remain active dispositions; `RESOLVED`
and `FUTURE FOLLOW-UP` have their existing meanings. These are the publishing
reviewer's dispositions, not a substitute for the aggregate reviewer gate.
The original finding retains its historical severity and attribution; it does
not override the status at the start of the bullet.
There is deliberately no colon immediately after the bracketed ID: Markdown
can interpret `[item-3]: RESOLVED` as a hidden link-reference definition.

Coder follow-up and plan-revision rounds are structured too. A PR follow-up
response must classify every carried reviewer item exactly once:

```json
{
  "schema_version": 1,
  "kind": "coder_followup",
  "state": "blocking",
  "summary": "Implemented the requested fix.",
  "addressed_items": ["item-1"],
  "remaining_items": [],
  "addressed_item_notes": {"item-1": "Updated the parser and added regression coverage."},
  "remaining_item_notes": {},
  "human_requirement_dispositions": [],
  "human_requirements": {
    "addressed_ids": [],
    "checked_discussion_directly": false
  },
  "tests_run": ["python -m pytest tests/test_agent_loop.py -k followup"]
}
```

`human_requirement_dispositions` is an auditable ledger: include exactly one
entry for each surfaced signed requirement, with disposition `addressed`,
`blocked`, or `not-applicable` and non-blank evidence. It must be empty when
no signed requirements were surfaced. In structured coder follow-ups,
`human_requirements.addressed_ids` contains exactly requirements whose
disposition is `addressed`; omit `blocked` and `not-applicable` requirements.
A blocked disposition requires `state: "blocking"`, while not-applicable may
appear in an approved response. Legacy markdown acknowledgements still list
every surfaced requirement.

A plan revision uses:

```json
{
  "schema_version": 1,
  "kind": "plan_revision",
  "state": "blocking",
  "summary": "Updated the plan for the reviewer feedback.",
  "prior_plan_item_dispositions": [
    {"item_id": "item-3", "disposition": "resolved", "note": "Covered in step 2."}
  ],
  "plan_steps": ["Update the parser.", "Add focused tests."]
}
```

Structured payloads must be the first content in the response, not fenced in
markdown, and may not have prose between the JSON and the required footer. The
only trailing content after the footer is the standalone agent signature. Plan
revisions are rendered into canonical markdown for stored plan state, reviewer
prompts, subject hashing, and resume; public comments render human-readable
sections and omit raw JSON.

Coder PR follow-ups use a deliberately narrower item namespace. Only the
human-requirements acknowledgement record and the merge-conflict record have
dedicated non-classifiable prompt paths. Ordinary reviewer findings and every
other repair-required machine obligation — including managed exact-head CI,
ordinary GitHub checks, migration validation, and conservative unknown
obligations — remain visible in `addressed_items`, `remaining_items`, or
`disputed_items`. The acknowledgement record is shown separately with its
stored validation diagnostic, and its internal ID must never be placed in
those reviewer-item fields. Merge conflicts retain their dedicated conflict
resolution prompt. Coder classifications are evidence of intent only: they
cannot clear a CI or other source-authoritative machine gate before that
authority revalidates the repaired head.

Signed human reviewer comments are requirements when the comment body ends with
a standalone `-- Human Reviewer` signature. Issue comments become signed
planning or implementation requirements; PR comments become signed PR-review
requirements. They override AI reviewer preferences unless they are unsafe,
impossible, or superseded by a later signed human instruction.

When signed requirements are present, legacy coder markdown acknowledgement
responses must include:

```md
<!-- HUMAN_REQUIREMENTS_ADDRESSED -->

### Human requirements
- Requirement hr-<digest>: explain how it was addressed or why it cannot be satisfied safely.
```

Structured coder follow-ups acknowledge requirements only in
`human_requirement_dispositions` and
`human_requirements.addressed_ids` / `checked_discussion_directly`; they do
not include the legacy marker or a Markdown section, and no prose may appear
between their JSON object and footer. If the prompt says detailed requirements
were omitted to stay bounded, the coder must check the GitHub discussion
directly and acknowledge that fact in those JSON fields instead of listing
requirement IDs. Initial and revised planning responses retain their distinct
post-JSON marker-and-section placement rule. Stable IDs are content-derived and
survive insertion or reordering; edited instructions receive new IDs, and
legacy positional replies require a fresh acknowledgement instead of being
mapped onto the current set. The orchestrator creates or retains a synthetic
`item-human-requirements-acknowledgement` obligation only for a dedicated
acknowledgement/disposition failure, not for a generic schema, footer, or
layout failure. Reviewers must include
`<!-- HUMAN_REQUIREMENTS_RESOLVED -->` in an approved review before the loop
treats signed requirements as resolved; otherwise the synthetic item is carried
into the next round even if the visible review says approved.

The loop validates required markers before posting agent output to GitHub. If
an agent exits unsuccessfully or returns only diagnostics, it first checks the
uniquely assigned public response file for that invocation. A non-empty
artifact that passes the normal schema and role validation is accepted and
posted even after a timeout or nonzero exit. Stdout remains diagnostics only
and is never salvaged as a public response. Empty, stale, malformed, and
wrong-role artifacts continue through the normal retry/failure path; accepted
failed exits record their outcome and return code in resume metadata.

Coder prompts for a direct issue implementation explicitly forbid launching
required tests or other completion work (builds, commits, pushes, PR creation)
in the background and ending the turn early; the coder must finish that work
in the foreground and wait for it before responding. If a Claude
implementation turn still ends this way — no valid `AGENT_PR`/`AGENT_STATE`/
`AGENT_CLARIFY` marker, and text like "I'll wait for the background test run
to finish" or "you'll be notified when it's done" — the loop performs one
bounded `claude --resume <session>` completion-recovery pass instead of
failing immediately. The resume turn is told to inspect the existing checkout,
finish only foreground work, and either complete the PR or end with a real
terminal marker. Its result is validated exactly like a normal implementation
response: a valid PR, a no-PR `AGENT_STATE: blocking`, or `AGENT_CLARIFY` is
accepted and posted like any other outcome. If the resume turn instead
declares `AGENT_UNAVAILABLE` itself, or if the resume command fails or still
does not produce a valid terminal response, the loop first applies the same
per-invocation response-file rule. A valid artifact is accepted with the
failed-exit diagnostic preserved in resume metadata; otherwise the loop renders
(or reuses the agent's own verbatim) protocol-valid `AGENT_UNAVAILABLE` text,
persists it to that attempt's own response file, and posts it to the GitHub
issue before failing locally with `AgentLoopError` and the usual salvage
artifacts — there is never more than one resume attempt, regardless of what
the agent's own response says about retrying. A genuine, non-recovery no-PR
`AGENT_STATE: blocking` or `AGENT_CLARIFY` result is likewise posted to the
issue, matching how a successful PR-creating implementation is already posted.

For structured plan reviews, plan revisions, PR reviews, and coder follow-ups,
a present but malformed structured response may get a repair pass before the
local failure is raised. By default the repair pass calls Antigravity through
the existing PTY backend with the default repair chain `Gemini 3.8 Flash (Medium)` →
`Gemini 3.7 Flash (Medium)`; explicit repair models replace that chain and are
followed by the configured coder/reviewer chain. Models missing from `agy models`
are skipped. When `agy` reports a transient `model-access validation errors`
failure on its own output channel (stdout with no response artifact and no
structured JSON response), that repair model is retried once and the chain then
continues with the next model; a valid response is always accepted, and
model-authored text quoting the phrase is still treated as invalid output. A
chain that ends in that transient failure is reported as a resumable
`repair-provider-failure` with a re-run suggestion rather than a deterministic
plan-validation failure. It uses a fresh temporary
workdir, empty tool permissions, and repair-only instructions forbidding file
inspection, tests, mutation, background work, and subagents. The format-repair
prompt asks it to preserve the agent's intent while emitting
only the required JSON object, matching footer marker, and standalone
signature. Repaired output is accepted only after it passes the same schema,
state, footer, follow-up, prior-item, and human-requirement validation as an
original response. If the repair CLI fails, returns empty output, or produces
invalid output, the original validation failure remains local and nothing is
posted to GitHub.

Reviewer repair is refused before any backend call when the source carries no
review substance. A `plan_review` or `pr_review` source is admitted only when a
JSON object is mechanically recoverable from it and that object either declares
the expected reviewer kind or, when it carries no `kind`, carries a field unique
to that review schema (`blocking_plan_issues`, `same_plan_followups`, or
`prior_plan_item_dispositions` for a plan review; `blocking_items`,
`same_pr_followups`, or `prior_item_dispositions` for a PR review). The
kind-unique-field fallback applies only to a source that carries no `kind` key at
all: a `kind` that is present but is not exactly the expected reviewer kind —
including an empty string, a null, or any non-string value — is refused, and the
grounding check below fails closed on the same rule rather than on a string-only
comparison. Fields shared
with other schemas — `summary`, `state`, `schema_version`, `future_followups`,
`human_requirement_dispositions`, `architecture_impact` — are never admission
evidence, so narration, tool-use diagnostics, a bare protocol state footer, an
explicitly different kind, and a kindless object carrying only a generic state
and summary are all refused. A refusal is classified as a reviewer
unavailability (`agent-unavailable`) with a bounded retry inside the configured
retry policy, never as a deterministic blocking verdict: no reviewer comment is
posted and no carried ledger item is numbered for that reviewer. An empty
reviewer response keeps its existing `empty-response` path and never reaches
repair, and a transient repair-backend provider failure keeps its resumable
`repair-provider-failure` classification. A reviewer output whose own
diagnostics already name a definitive provider condition — an auth, billing, or
credit failure, an unsupported model or effort, an exhausted resource, or an
indeterminate containment result — keeps that classification and its operator
suggestion: refusing to repair such a turn says nothing about reviewer
availability, and a rerun cannot fix it.

Whenever a repaired response's own kind is `plan_review` or `pr_review`, a
grounding check requires every repaired finding, summary, and carried
disposition to be supported by the reviewer's own source text; it fails closed
when the source payload is absent or declares a different kind. Support is token
coverage — not contiguous containment, so the documented title-plus-detail
concatenation stays legal — of the candidate against the whole normalized
source, after removing four exempt sets: a pinned closed stop list, the review
schema's own vocabulary, carried `item-<n>` identifiers, and the reserved-marker
neutralization labels derived from the marker registry. Findings in all three
buckets are matched injectively to source findings from any bucket, so a
promotion out of `future_followups` remains legal while the repaired finding
count may never exceed the number of source candidates. When the source payload
declares no finding in any bucket, the candidates come only from freeform prose
that sits *outside* the recovered JSON object: the object's own fields are
structured data, and `summary` in particular is not a finding, so splitting the
serialized payload into prose segments would let an approved source's summary be
copied into a current-scope finding and then ground an inverted blocking verdict.
Those freeform candidates are non-overlapping — a paragraph contributes either its
list items or its joined prose, never both, a wrapped bullet's continuation
lines join the item they belong to, and a lead-in line before the first bullet is
list structure that is dropped rather than a concern of its own, so it can neither
raise the ceiling nor let a repaired finding match on the heading alone — so
one trailing reviewer statement cannot be matched twice and raise the ceiling,
and protocol footer and signature lines are structural records rather than
candidates. A declared source finding that carries
no prose is a candidate only when it really was nothing but a reserved marker, and
then it corresponds solely to its own authorized neutralization: the target must
represent the COMPLETE source marker multiset, each occurrence either kept
verbatim or replaced by that occurrence's own safe label, compared by marker
identity and occurrence count over the registry's historical replacement spans —
the same occurrence set the stripping pass uses, so a malformed name-bearing-line
fallback cannot hide a second reserved token on its line — rather than by the
empty string that stripping any marker leaves behind, so an unrelated family, a dropped occurrence, a duplicated
one, and leftover prose are all refused. Every other exempt-only target is refused,
whichever candidate it is matched against: schema vocabulary, the stop list and
every registry safe label are exempt, so `blocking`, an unrelated marker or its
label, or stop-word-only prose has an empty content-token set that is trivially a
subset of any candidate and vacuously covered by the whole source, and a target
matched to a substantive candidate must therefore retain substantive content of
its own. A modifier-only candidate is bounded instead by the modifier-count
equality rule below, which forces the target to carry the same modifiers. A genuinely empty entry such as `{}` is dropped, so it can neither become
a wildcard for an exempt-token-only finding nor raise the ceiling.
Whole-source coverage alone cannot tell
a finding apart from the summary or any other global prose, so without that
restriction a source summary could be promoted into a fabricated finding and
ground a blocking verdict. Each matched finding pair — including one matched
against a freeform trailing-prose candidate — and each matched disposition note
pair, must carry equal per-modifier occurrence counts over a pinned set of
negations and limiting qualifiers, compared after a pinned
contraction normalization that runs before punctuation stripping over both
apostrophe forms — so deleting, adding, or substituting `not`, `isn't`, or
`only` is rejected even though subset coverage alone would accept it. The
verdict is grounded against the source rather than against what survives into
the target: a repaired `blocking` needs an unambiguous source blocking state, a
preserved source finding, or a carried disposition that is active in the source
*and* preserved as active in the target — a disposition completed from the repair
context's allowed IDs, or one re-stated out of a source `future`, is supplied by
the orchestrator or by the target's own state and can never be that state's own
support; a repaired
`approved` needs an unambiguous approved source state *and* a source that itself
carries no current-scope finding and no active disposition, so demoting a
current-scope finding into `future_followups` can never manufacture an approval.
Carried dispositions are matched by `item_id`, and only bounded normalizations
are authorized: the repair prompt's own enum aliases (`still blocking` →
`blocking` and its siblings); an active-to-`resolved` change only when the
source note satisfies a negation-safe coverage predicate (a pinned coverage
phrase that no pinned negation marker precedes inside its own sentence, with the
still-open phrase list generated from the coverage list so every phrase carries
its negated counterparts); re-stating a source `future` disposition as the
kind's non-resolving active value in a blocking review; and completing an ID
supplied in the repair context as that same non-resolving active value.
Deterministic removal of unknown IDs is unaffected. The check is a bounded
support check, not a certification that a genuine reviewer finding is correct.

Repair is lossless formatting, not summarization: findings must retain their full
evidence, code references, and requested tests; summaries and test reports must
retain failures, timeouts, skips, and partial-coverage caveats. Object-to-string
conversion must carry the complete substantive text. Reviewer item IDs must never
be invented as signed human requirements.

For parseable JSON in review, planning, and coder/implementation responses, a
local preservation guard additionally rejects dropped/rewritten summaries,
test-command entries, plan steps, coder evidence notes, and current-scope finding
text. A rejected candidate follows the existing repair-model fallback chain; it
is not posted as a successful repair. Whitespace, finding order, and allowed
current-scope bucket moves are tolerated. Schema-mandated removal of invalid
fields, forbidden future items, and reserved protocol syntax remains allowed.
This is a bounded loss check, not semantic-equivalence certification: malformed
JSON, unsupported response kinds, and individual strings containing reserved
protocol grammar are not compared by this guard. Those paths still rely on the
lossless prompt and the ordinary schema/context validators. No repair can certify
that an agent's underlying code or test claims are true.

Known transient agent/model failures are retried before local failure. A
structurally recognized response rejected by schema, item-classification,
acknowledgement, footer, or content validation is classified from that trusted
validation context, not from vocabulary inside the response's prose. Thus a
rejected response that happens to mention authentication, credit, billing, or a
dirty tree remains a deterministic protocol failure without credential or
billing advice. A repair timeout or invalid repair remains secondary to that
originating validation category; genuine provider and command diagnostics keep
their normal classifications. The
default is two retries with bounded backoff; tune this with
`--agent-max-retries` and `--agent-retry-backoff-seconds`. Retry matching is
narrow and intended for stream/tool-call failures, empty responses, network
timeouts/resets, and provider 5xx errors. Auth, credit, quota, dirty workdir,
and normal missing-marker responses are not retried.

Claude and Codex have a separate, evidence-gated post-spawn executable
replacement path. The runner samples the resolved command entry, symlink target,
and executable identity for the successful spawn attempt and again after exit.
A replay is considered only when direct identity-change or disappearance evidence
falls within the invocation window. Codex has no elapsed-time cap, but it may be
replayed only when its failed `--json` stream parsed as empty or as exactly one
setup-only `thread.started` dictionary. Any other dictionary event—including
item, tool, command, error, or `turn.completed` activity—means work may have
started and prevents a fresh replay. A public response file or last-message
artifact suppresses replacement classification even when metadata changes after
completion; a malformed present artifact remains on ordinary validation.

The Codex stability wait is independent and bounded to six seconds. Once stable,
the loop performs at most one fresh `codex exec` replay with the full configured
timeout (or no timeout when unconfigured), without consuming ordinary retry
budget. Ordinary Codex invocation and discuss-round log names are unchanged; only
the dedicated replay uses `executable-replacement-attempt2`. An unstable or
exhausted path retains a specific Codex executable-replacement diagnostic. Bare
commands compare PATH entry, symlink, and target identities; an absolute override
requires direct evidence that its exact entry or target changed or disappeared.
Claude's existing 30-second eligibility cap, updater diagnostics, deadline-
bounded stability wait, remaining-time replay, and `self-update-attempt2` suffix
remain unchanged, with one additional read-only workdir gate. Immediately before
each Claude invocation the backend runs exactly `git rev-parse HEAD` and `git
status --porcelain` in the assigned checkout. A zero exit with nonblank stripped
HEAD is available; a zero exit with blank HEAD is unavailable. Status is available
on zero exit even when stdout is empty, because empty porcelain output is the
valid clean-worktree state. Non-empty porcelain output is preserved exactly, so
unchanged dirty snapshots are replayable just like unchanged clean snapshots.

The after snapshot is lazy and is taken only after the existing failure, elapsed,
artifact, session, and parseable-JSON progress exclusions leave positive updater
or managed-command identity evidence. Claude replay is accepted only when both
snapshots are available and identical. Changed HEAD, changed status, and before
or after probe unavailability fail closed for replay. Claude tolerates an
`AgentLoopError` or `OSError` from these probes by recording structured
unavailability; the separate PR HEAD-advance guard uses strict exception
behavior, while ordinary nonzero or blank HEAD results still map to an
unavailable observation.

Replay refusal details are attempt-specific diagnostics only. They are recorded
with the failed call's log path, do not set accepted executable-replacement
evidence, do not trigger stability waiting or the dedicated replay, and do not
change provider-derived retry eligibility, reviewer availability, or failure
category. Terminal annotations are added once, with accepted replacement context
before the latest refusal context and the underlying failure last. This remains
a read-only guard: local HEAD and porcelain status cannot observe byte-identical
external effects such as pushes, pull-request creation, or comments, so it
reduces replay risk without eliminating it.

Codex remains unchanged because its JSONL stream already supplies setup-versus-
progress evidence and retains its separate fresh-timeout replacement replay. If
a genuine long quota reset occurs after replacement context was recorded, quota
remains primary and exit code 3 is preserved with the earlier replacement detail
appended.

Unsupported model/provider-auth compatibility errors are reported separately
with failure category `unsupported_model`. These diagnostics name the agent and
requested model when available and suggest choosing a compatible model or
provider/auth mode instead of treating the failure as a deterministic protocol
or code issue.

When `--approved-followups` is set to `summarize`, `issue`, or a `fix-and-*`
mode, approved reviewer responses may also include optional future-work items
under a dedicated heading:

```md
### Future follow-ups
- Add a follow-up test.
```

Reviewers should use this section only for substantial work that is better
handled in a separate issue or PR. The legacy heading
`### Non-blocking follow-ups` is still accepted as future work for
compatibility.

When `--approved-followups` is set to a `fix-and-*` mode, approved reviewers
can request small, localized, low-risk cleanup that should land in the current
PR by returning a blocking review and putting those items under:

```md
### Same-PR follow-ups
- Rename a helper before merge.
```

Same-PR follow-ups are sent back to the coder in the existing PR and require a
new review round before approval can finalize. They should stay narrowly scoped
to files already touched by the PR or directly adjacent code; larger redesigns
and independent work belong under Future follow-ups. Approved reviews may not
include Same-PR follow-ups, blocking items, or carried-forward items that remain
`blocking` or `same-pr`. Approved future follow-ups remain in the
round-to-round ledger so later reviewers can explicitly confirm they are still
future work, resolved, or should be promoted back to same-PR or blocking
status. The final summary or issue creation uses the remaining future items
from that ledger. Without a `fix-and-*` mode, reviewers should mark same-PR
cleanup blocking instead.

By default, `--approved-followups=ignore` asks reviewers not to include these
sections. Reviewers should mark the review blocking instead when cleanup should
be fixed before merge.

`--approved-followups` accepts:

- `ignore`: ignore approved follow-up sections. This is the default.
- `summarize`: post future follow-ups as a grouped PR comment.
- `issue`: create GitHub issues for up to three future follow-ups, then comment with the created issue links.
- `fix-and-summarize`: send same-PR follow-ups to the coder for another review round, then summarize future follow-ups after final approval.
- `fix-and-issue`: send same-PR follow-ups to the coder for another review round, then create issues for future follow-ups after final approval and comment with the created issue links.

Issue-filing modes use a conservative two-stage reconciliation. Deterministic
normalization, headings, identifiers, paths, and topic overlap run first. The
publisher then searches open follow-up trackers in the configured repository
with at most five focused queries, twenty results per query, and fifty unique
candidates per publication. Parent issue, approved-plan hash, PR, and related
links are preferred before a bounded repository-wide topic query. Search
results are repository/identity checked and the selected tracker is
revalidated as open immediately before reuse; closed or cross-repository
results never suppress a new issue. Search indexing is eventually consistent,
and there is no atomic repository-wide lock, so a create-then-interruption
window is recovered by bounded rediscovery on a later invocation.

For ambiguous in-batch groups and narrowed existing trackers, an optional cheap
semantic classifier receives only bounded excerpts and must return strict JSON
with `duplicate_of`, `confidence`, and a non-empty `reason`. Only `high`
confidence equivalence of the actual deliverable suppresses or merges work.
Medium/low confidence files normally with a sanitized possible-duplicate note;
provider failures, invalid output, timeouts, and local budget exhaustion fall
back to deterministic behavior. A quota-reset exhaustion is different: it is
propagated so the orchestration run can stop without filing or publishing a
success audit record. Configure the classifier with
`--semantic-followup-backend`, `--semantic-followup-model`,
`--semantic-followup-timeout-seconds`, `--semantic-followup-max-calls`,
`--semantic-followup-max-candidates`, and
`--semantic-followup-prompt-char-limit`; use
`--no-semantic-followup-dedupe` for deterministic-only operation.

Publication summaries distinguish created issues, reused trackers, uncertain
matches, and cap-skipped work. The three-new-issue cap is applied after all
groups have been checked for reuse, so reusing an existing tracker does not
consume a creation slot. Candidate titles, excerpts, retained reviewer/planning
context, and model reasons are sanitized before entering a GitHub body or
comment, while the expected publish-once audit record remains the only
authorized protocol record in those bodies.

For plan-first runs that continue into implementation, issue-filing modes apply
twice at different lifecycle points: planning-stage future follow-ups are filed
before implementation begins, while PR-stage approved-review future follow-ups
are filed only after final PR approval.

Bullets and prose paragraphs inside the `Same-PR follow-ups`, `Future follow-ups`,
and legacy `Non-blocking follow-ups` sections are parsed; each section ends at
the next heading, HTML marker, or agent signature. The same parsing is used
when creating follow-up issues. The issue cap keeps one approved review from
creating a large batch of low-value issues.

The remaining legacy compatibility surface is intentionally narrow:

- Markdown plan/review parsing stays enabled for agents that do not emit
  structured JSON.
- The legacy heading `### Non-blocking follow-ups` is still treated as future
  work.
- Marker-only markdown paths are still parsed where supported, but structured
  JSON is the documented format for new reviewer, coder follow-up, and plan
  revision responses.
- Resume reconstruction should rely on `AGENT_LOOP_META` instead of reparsing
  old prose whenever metadata exists for the active round.

### Tool-owned protocol records

Beyond the round transport, agent-loop publishes protocol records whose payload
lives in a hidden marker. Each of these comments now opens with one bounded,
deterministic visible line naming the record and its role, for example:

```text
Agent-loop managed-CI authorization record (machine-readable). Binds issue #868
to pull request #877 at head e75771c. Not an agent response; keep this comment.
```

The labelled record types are:

- the managed-CI issue authorization record, with distinct wording for the
  creation, fresh re-authorization, and head-continuity kinds;
- the managed-CI exact-head intent record, which also names its lifecycle
  state;
- the managed-CI unprotected-override audit and the resume-provenance audit;
- the managed-CI qualified-head record;
- the plan-validation diagnostic record.

The issue-to-PR handoff and the PR expected-closing contract already render
their own visible prose and are unchanged. All of these records are
machine-readable and must be kept: deleting one can cost resume continuity,
recovery context, or an audit trail. The label sits outside the marker span, so
the marker grammar, payload, trust boundary, parsing, and recovery are
unchanged, and older marker-only copies (which GitHub renders as "No
description provided.") remain valid and still parse. Because each label is a
pure function of the already-authenticated record, a retry produces a
byte-identical body; every one of these writes is then verified by comparing
the server's stored body and the producing identity against what was posted.

### Carried item identity

Every unresolved item has an immutable canonical claim (`text`) and accumulated
reviewer/coder evidence (`notes`). Review prompts show these separately as
**Original claim** and **Updates/evidence**. Legacy records that appended
`Update from ...` lines to the claim are split only when displayed; the stored
record and resume signatures remain unchanged.

When reviewing a carried item, reviewers must evaluate its original predicate.
More specific evidence for the same defect may keep the same ID. If the old
predicate is accepted but a materially different defect is found, resolve the
old item and add the different concern as a new `blocking_items` or
`same_pr_followups` entry (or `blocking_plan_issues` / `same_plan_followups`
for a plan review) in the same response. Those arrays contain new findings
only; an active carried claim appears only as a `blocking`, `same-pr`, or
`same-plan` disposition plus its note. Each explicit new finding receives a
fresh stable ID.

New PR review responses must supply a non-empty, actionable `note` for every
carried `blocking` or `same-pr` disposition. Explain what remains wrong on the
reviewed head, the relevant evidence, and the change or test needed. Missing
notes and bare status restatements are rejected; this structural check cannot
guarantee the quality of the explanation. Resolved dispositions may omit notes.
Older saved reviews still parse and resume; a missing active-item explanation
is explicitly labeled in the ledger rather than invented.

The CLI coder follow-up also includes the latest available reviewer summaries
as separate, attributed review-level context, including on resumed rounds and
when compact PR review prompts are used. Summaries do not receive item IDs,
replace original claims, or become another reviewer's item-specific evidence.
Repair may copy clearly attributed explanations from the original response,
but must not invent missing rationale or discard the original summary.

In the other direction, CLI reviewers receive the latest structured coder
summary, reported tests (including caveats), addressed/remaining item notes,
and dispute evidence as a separate block. It is labeled with the saved coder,
review round, and PR head, and is included in full and compact prompts for
serial, parallel, and resumed rounds. Compact prompts put this changing block
in the volatile tail. Old-head explanations are omitted with an explicit
notice; missing legacy structured details are not invented. These are coder
claims to verify, not resolutions or proof that tests passed. The original
item ledger and reviewer disposition requirements remain unchanged.

Reconciliation remains conservative for a genuine same-claim disagreement: a
valid active disposition still outweighs another reviewer's `resolved` vote.
An active carried disposition is actionable even without a new-finding array,
so it is neither classified as an incomplete review nor converted from summary
prose into a duplicate new item. Summary fallback remains available only when
there is no explicit new finding and no active carried disposition.

Fresh findings do not inherit coder-dispute lineage because semantic
equivalence cannot be inferred safely. If a disputed claim is improperly
re-filed as a fresh item, the coder gets one additional dispute on that new ID
before the existing continued-blocking escalation applies.

## Logs

Agent stdout/stderr is written to `.agent-loop-logs/` under the active coder
checkout by default. If that coder directory was omitted, the relative default
log path is also under the repo-scoped temporary checkout and may disappear
with `/tmp` cleanup. The CLI prints heartbeat messages with the log path while
agents run:

```text
[agent-loop 12:00:31] Claude still running (30s); log: /path/to/.agent-loop-logs/20260425-120001-claude-attempt1.log
```

Use `tail -f` on the displayed path to see live output. Logs are diagnostic
output and may include CLI status text or tool narration. For Claude and
Gemini, prompts also include a public response-file path under
`/tmp/coding-review-agent-loop/responses/`; when that file exists and is
non-empty, the loop validates and posts the file contents to GitHub instead of
stdout. Codex also receives a response-file instruction, and separately uses
`--output-last-message` so the loop can fall back to the last Codex message
instead of raw JSON event logs. Gemini response files live inside Gemini's git
directory because Gemini can only write trusted workspace paths; Gemini stdout
also supports the `=== AGENT_LOOP_PUBLIC_RESPONSE_BELOW ===` marker and, when
`--output-format json` is used, extraction of the JSON `response` field.
Fallback stdout is still validated before posting. The log directory gets its
own `.gitignore` on first use.

When an agent appears stuck, inspect the heartbeat log path first and then the
printed response-file path. The log shows CLI narration, tool output, provider
diagnostics, and whether the subprocess is still making progress; the response
file is the public answer the orchestrator will validate and post. Empty
response files, missing markers, or diagnostics-only stdout fail locally instead
of being posted to GitHub.

Active subprocess captures default to a unique directory under the agent-loop
cache, outside managed checkouts. `--subprocess-log-dir` selects an explicit
capture root; relative values are resolved from the primary agent directory,
and paths equal to or beneath a managed checkout are rejected.
Invocation directories hold a lifetime lease; cleanup removes only old,
unlocked directories on a best-effort basis. `--log-dir` remains the legacy
root for usage summaries and salvage artifacts, and old checkout-resident logs
are not migrated or rediscovered as active captures.

When a mutating coder implementation run reaches a terminal failure with a
non-empty tracked diff, the orchestrator writes local salvage artifacts under
`<log_dir>/salvage/<run>-<agent>-<scope>/`. That directory contains
`partial.patch`, `changed-files.txt`, `diff-stat.txt`, best-effort
`diff-check.txt`, `salvage-summary.md`, and `metadata.json`. The run log and
local error point to the artifact directory. These artifacts are incomplete
failure diagnostics only: they are not posted as a successful response, and a
later rerun injects only the latest matching salvage summary into the coder
prompt so the next attempt can cherry-pick or ignore it selectively. The
orchestrator never auto-applies the patch.

For `issue-implementation` and `approved-plan-implementation` scopes (the two
whose rerun prompts consume a salvage summary), the orchestrator also posts a
best-effort GitHub issue comment with a hidden `<!-- AGENT_SALVAGE: ... -->`
marker alongside the local artifacts. This makes salvage durable across a
different coder, workdir, or machine: a rerun with an empty local `--log-dir`
can still discover the latest matching comment (filtered by repo, issue,
scope, and approved-plan hash) and inject it into the coder prompt, noting
that the referenced local artifact paths may not exist on this machine. Local
and remote salvage are merged deterministically by timestamp, with ties
preferring the local copy. The partial patch is embedded in the comment only
when it is under `--salvage-comment-patch-max-bytes` (default 20000), has no
`GIT binary patch` section, and does not match a conservative secret scan
(private keys, AWS key ids, `ghp_`/`github_pat_` tokens, bearer/authorization
headers, `password=`/`secret=`-style assignments); otherwise the comment notes
the patch is local-only. `AGENT_SALVAGE` breadcrumb comments are excluded from
the raw issue-comment context sent to the coder (they are consumed only
through the parsed salvage summary above) so they cannot bloat or displace
real discussion when the issue-context prompt is truncated. Posting failures
are logged and never mask the
original agent failure. Use `--no-salvage-comments` to keep salvage entirely
local (matching prior behavior).

If strict structured-response validation fails, the log may show a repair pass:
`schema validation failed ... attempting repair pass`, followed by either
`repair pass recovered malformed response` or `repair pass produced invalid
output`. A recovered response is still revalidated before posting; a failed
repair leaves the run in local failure just like any other protocol error.

Use `--repair-backend`, repeatable `--repair-model`, and
`--repair-timeout-seconds` to configure this path. For Antigravity, the repair chain is the explicit
prefix followed by `antigravity_models`, with duplicates removed. It queries `agy
models` once through the same PTY-safe runner as `agy --print`, rejects stale names
with available-choice guidance, and attempts candidates directly when discovery is
unavailable. `--repair-backend gemini` retains the legacy CLI for
enterprise/API-key/Vertex authentication; personal OAuth may require an
interactive authorization and is not the default. Normal diagnostics include
backend, model, return code, a sanitized bounded combined-PTY diagnostic, log
path, and whether another configured model will run. Usage summaries record
prompt/output usage (provider-reported when available, otherwise estimated), repair outcome, and validation status for
every attempt, including failed attempts, so repair consumes visible quota.

Codex and Claude can also repair responses, independently of the original agent:

```bash
--repair-backend codex --repair-model gpt-5.6-luna --repair-reasoning-effort medium
```

Or select `--repair-backend claude --repair-model claude-sonnet-5`.
Both require an explicit model; repeat `--repair-model` for a same-backend fallback
chain. They do not append reviewer models or query Antigravity's catalog.
`--repair-reasoning-effort` defaults independently to `medium`, accepts the selected
provider's effort values, and is only valid with these two backends. It does not
inherit coder/reviewer effort. The configured `--codex-cmd` or `--claude-cmd` selects
the executable, but ordinary agent arguments and dangerous permission flags are
not forwarded. Selecting a repair backend does not change coder/reviewer models.

Each attempt uses stdin, a fresh temporary directory, and no resumed session.
Codex runs ephemeral with user config/rules ignored, a read-only sandbox, and
shell/web search/subagents disabled. Claude runs safe mode with tools and MCP
disabled and no session persistence. These paths require CLIs supporting those
flags; unsupported flags fail the attempt rather than weakening its restrictions.
CLI authentication remains available. Each attempt is bounded by
`--repair-timeout-seconds` (default 120). Nonzero exits and timeouts are not accepted
for Codex/Claude repairs. All successful output still passes schema validation and
the same content-preservation guard; repairs cannot drop findings or plan steps
just to make validation pass. The default backend remains Antigravity.

Long reset or quota responses can exit early with guidance to rerun after the
reset or switch keys/models. Narrower transient stream, tool-call, network
reset, timeout, empty-response, provider 5xx, and first-attempt marker
near-miss failures retry according to `--agent-max-retries` and
`--agent-retry-backoff-seconds`. Authentication, billing, dirty workdirs, and
normal missing-marker responses are treated as non-retryable configuration or
protocol failures.

Each top-level run also writes `<run-id>-usage-summary.json` in the same
directory. That sidecar aggregates usage by call, by agent, and for the full
run, including retries and marker-near-miss attempts. Backend-provided token
counts are normalized as `exact` when the counters are complete or `partial`
when only some counters are available. When a backend exposes no usable usage
data, the orchestrator records an `estimated` fallback based on prompt and
public-response size, along with raw character and byte counts. `--dry-run`
does not invent token usage records.

## Runtime-aware local test timeouts

The run-level command ceiling is configured with
`--coder-test-command-timeout-seconds SECONDS` and defaults to 1,800 seconds.
It is separate from a framework's per-test timeout and from the backend's
whole-turn timeout. A backend turn must exceed the selected whole-command
watchdog with headroom for analysis, edits, and reporting; Antigravity's print
timeout should exceed it by `max(300s, 20%)`.

The backend-neutral wrapper is `agent-loop run-tests [--timeout-seconds N]
[--memory-dir DIR] -- COMMAND...`. The scalar is the chosen watchdog for that
invocation. Omitted values inherit
`AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS`, falling back to the 1,800-second
default outside agent-loop. Positive finite values at or below the ceiling are
accepted; malformed or over-ceiling values are rejected before child spawn.
Agents may use a learned sub-ceiling recommendation when rendered, but must
continue to select focused tests and may split or shard long browser,
integration, or end-to-end matrices.

### Launcher preflight and health memory

`run-tests --preflight` is a bounded, non-mutating wrapper probe. It checks at
most two candidates—the absolute `agent-loop` console entry and the current
interpreter's `-m coding_review_agent_loop.cli` fallback—once per invocation,
with a five-second watchdog. Recognized inner launcher probes are separately
limited to six distinct candidate identities per invocation. Inner probes use
the effective target environment, including ambient values merged with a
partial overlay. Candidates are executed directly, including the
console script's own shebang interpreter. A fixed `agent-loop preflight:
verified` response proves startup; a non-start, timeout, or explicit import /
bootstrap failure is unhealthy, while ambiguous output remains unknown. The
probe does not run remembered test argv, install packages, access live
databases, cross the assigned checkout, or alter the operator's configured
command. Probe cleanup is process-tree aware: POSIX probes use a dedicated
process group, while Windows probes use a kill-on-close Job Object. If the
Windows owning boundary cannot be created or assigned, the probe is not
launched and remains unknown. The candidate cache includes executable,
shebang, interpreter,
package-origin, virtualenv, and invocation identity so repair or replacement
causes a fresh probe.

For the recognized inner forms direct `pytest`/`py.test` and exactly
`<python> -m pytest`, agent-loop performs a fixed `--version` bootstrap probe
under the same five-second bound. Other launchers are never classified from
stderr or an exit code. Results carry independent `wrapper_bootstrap`,
`inner_exec`, and `suite_start` states plus the existing suite outcome. Direct
exec failure and a containment `target-exec-error` are launch failures;
overlap rejection is coordination; collection/configuration errors, ordinary
failing tests, timeout, and interruption after startup are suite results.
Python identity is established without executing an untrusted candidate: the
running interpreter and symlinks to it are trusted, and a conventional
`pyvenv.cfg` interpreter is trusted only when it is a byte-for-byte copy of
the running interpreter. Name-only scripts and arbitrary ELF/MZ binaries stay
unknown and are never spawned by preflight.

The schema-v1 runtime sidecar keeps suite timing in `observations` and stores
bounded launcher health separately in `launcher_health`. Health rows include
repository slug, a SHA-256 assigned-checkout identity, environment fingerprint,
candidate identity, UTC timestamp, state, provenance, and a whitespace-collapsed
diagnostic of at most 240 characters. External executable/interpreter/source
paths are reduced to checkout-relative paths or basenames; only the exact
canonical managed-wrapper path is retained. Rows are independently validated,
locked, atomically written, capped at eight records per scoped identity and 100
identities, deduplicated, and expired after 24 hours. A successful matching
probe removes prior failures immediately. Scope or identity changes—including
wrapper reinstall, interpreter/dependency changes, virtualenv recreation,
checkout changes, and environment changes—permit a fresh probe.

Coder and PR-resume prompts load health separately from timing memory. Timing
rows remain the only source of timeout recommendations and remembered command
discovery. The configured command is always retained and a missing inner
launcher is annotated rather than removed. Managed invocation guidance is
emitted only from a wrapper verified in the current invocation; otherwise the
prompt gives an actionable repair diagnostic without claiming tests passed or
forbidding manual repair. Wrapper/parent observations are authoritative only
for their boundary, direct agent shell failures are not comprehensively
observable, and agent-reported diagnostics are labelled advisory and cannot
alone suppress a command. The finite foreground watchdog, containment,
assigned-cwd checks, and operator configuration remain in force.

Managed-wrapper recognition also traverses supported execution prefixes, using
the canonical managed-wrapper traversal shared by receipt citations and report
validation. Examples include
`env -u AGENT_LOOP_INVOCATION_ID /absolute/agent-loop run-tests -- pytest` and
`pwd && timeout 1800 env MODE=inline /absolute/agent-loop run-tests -- pytest`.
The conservative prefix contract supports `env` unsets/assignments/empty
environment, `timeout`, `nice`, `stdbuf`, `nohup`, `time -p`, and `command -p`.
Value-taking options must also have executable syntax: timeout durations and
signals, integer nice adjustments, valid stdbuf modes, and nonempty env names
without `=` are checked before the managed-launcher exemption is granted.
Numeric operands must use ASCII digits and fit conservative conversion bounds;
overflowing values do not qualify for the exemption.
Their executables and prefix values still undergo path and live-target checks;
only the managed launcher and its memory output receive the special exemption.
Unknown options, `env -S`, cwd-changing prefixes (`env -C`/`--chdir`), `sudo`,
`xargs`, and lookup-only `command -v`/`-V` do not qualify. This is not a general
shell interpreter and does not evaluate substitutions or infer rewritten argv.

Test-report location validation recognizes the managed wrapper both as a
standalone invocation and in parsed shell clauses such as
`pwd && git status --branch --short && /absolute/path/agent-loop run-tests -- python3 -m pytest tests/test_api.py`.
Only the exact wrapper executable and its memory-output option receive the
special exemption. The inner command, leading assignments, and surrounding
clauses still undergo the existing path and live-target checks. Malformed
wrapper options receive no special exemption. This validates reported command
text; it does not execute the report or provide a shell sandbox.

#### Recognized launcher spellings for reported command text

Agents report the launcher spelling they actually typed, which is often a bare
PATH command name rather than the absolute path the prompt renders. Report-side
recognition therefore accepts three launcher spellings: the absolute
`agent-loop` console entry, the absolute interpreter's
`-m coding_review_agent_loop.cli` fallback, and a separator-free PATH command
name in either of those positions (`agent-loop run-tests ...` and
`python3 -m coding_review_agent_loop.cli run-tests ...`). Accepting the bare
spelling is what keeps the wrapper's own `--memory-dir` and `--timeout-seconds`
values, which point at an agent-memory directory outside the checkout by
design, from being scanned as reported test locations.

The four report-side consumers that adopt this recognition are checkout
validation of reported test commands, risk-matrix citation-to-receipt
projection, public comment rendering of reported commands, and referenced-path
projection for local test evidence. They adopt it consistently so a spelling
accepted by one gate cannot fail a later one or leak wrapper plumbing into a
published comment.

Path-shaped spellings such as `../agent-loop`, `./agent-loop`, or
`bin/agent-loop` do not qualify: a bare command name cannot name a location,
whereas a relative path can, so extending the exemption to it would widen the
containment boundary. Recognition is opt-in per call site: the shared parser's
default remains absolute-only, and actual command execution is unchanged, since
`run-tests` parses its own argv through argparse and never consults this
parser. Malformed, duplicate, or unknown wrapper options still fail closed
under every launcher spelling, and the inner command after `--` is always
validated.

Explicit `sh`, `bash`, and `zsh` command-string invocations (`-c`, including
simple combinations such as `-lc`) are validated as nested commands. An
external virtualenv interpreter is checked separately from its test paths;
the quoted command string is not treated as one filesystem path. Launcher
arguments, inner test targets, working directories, and live URLs remain
subject to the existing checks. Nested shells are bounded to eight levels;
common strict-mode forms such as `-euo pipefail` and operand-free login/profile
options are parsed, while unsupported options before `-c` and multi-word shell
operands without `-c` are rejected because their contents cannot be validated
safely. Unsupported short-option clusters containing `c` are likewise rejected
rather than risking an uninspected command-string operand.

Command-string tokenization recognizes control operators adjacent to commands
or arguments, while quoted operator characters remain ordinary argument text.
This does not evaluate shell substitutions or certify arbitrary shell programs.

With writable agent memory, measured wrapper/gate outcomes are stored in the
versioned `test-runtime.json` sidecar. It records elapsed time, attempted cap,
outcome, commit, input hashes, and a privacy-preserving local environment
fingerprint. Successful samples produce conservative median/p95 recommendations;
timeouts are lower-bound evidence, never successes. Samples are retained up to
20 per command/fingerprint cohort and 200 cohorts, and stale after 30 days or
relevant lockfile, configuration, target, or fixture changes. Persistence is
best-effort and uses an advisory lock plus atomic replacement.

### Semantic revision assembly

The phase-1 semantic planning foundation separates model decisions from wire
serialization. A `plan_revision_patch` v1 carries only a response summary,
prior-item dispositions, an authenticated base round/identity, and bounded
semantic operations. Whole-field replacements require complete generation-1
values. Matrix edits require complete rows; additions declare their final
position; splits declare ordered targets; and merges declare their complete
lineage. Derived metadata, audit entries, ordering, identities, sidecars, and
protocol markers are not writable operation targets.

The deterministic assembler hydrates an authenticated blocking base, validates
all operations before mutation, and then emits the existing generation-1
`plan_revision` object. Matrix operations are simultaneous and independent of
patch-array order. Unchanged fields and rows remain sourced from the
authenticated canonical JSON, audits are normalized in fixed operation-class
order, and an empty complete matrix scope is represented by the `matrix`
sentinel. Metadata coverage does not hide precise row-operation entries.

The assembled canonical JSON, aggregate identity, response form, base binding,
and raw patch provenance can be carried in round metadata. The canonical
sidecar is the restart authority; raw patch text cannot be reparsed as state.
Fresh validated full-state planning and newly produced legacy-form full-state
revisions are seeded at publication with an authenticated sidecar and
aggregate identity. The first eligible unapproved revision is pinned to
`semantic-patch-v1`; the pin survives full/compact prompts, retries, and
restart. The model receives the canonical plan as read-only context and emits
only bounded semantic decisions. Publication hydrates the sidecar, assembles
the generation-1 object, validates it, and renders the existing Markdown
surface.

Historical and already-started legacy rounds remain pinned to their original
form. Matrix-less history does not acquire fabricated matrix obligations, and
old records are never backfilled with sidecars. Missing, partial, conflicting,
or identity-mismatched hydration fails closed before prompting, assembly, or
publication. Repair is limited to envelope presentation before assembly and
must preserve the semantic patch, rationale, operation ordering, and base
binding exactly.
