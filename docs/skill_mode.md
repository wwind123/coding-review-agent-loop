# Claude Code Native Skill Mode

## Overview

`coding-review-agent-loop` includes a Claude Code skill that lets you run the
multi-agent review loop directly inside an interactive Claude Code session
instead of through `claude -p` subprocesses.

| Aspect | Headless CLI mode | Skill mode |
|--------|-------------------|------------|
| Claude turns | `claude -p` subprocess | Active Claude Code session |
| Codex turns | `codex exec` subprocess | Same `codex exec` subprocess |
| Gemini turns | `gemini` subprocess | Same `gemini` subprocess |
| GitHub ops | Python `gh` wrapper | Same `gh` wrapper |
| Session resume | AGENT_LOOP_META in GitHub comments | Same markers + local session JSON |
| Claude token use | Usually lower because orchestration is mechanical Python control flow | Often higher because the host session also interprets and executes orchestration steps; actual usage varies with context, caching, and task shape |
| Configuration | Exact CLI flags supplied up front | Conversational; host can explain and select helper options |
| Quota reset | Can be rerun later by an external scheduler | User must return and resume the host session after reset |
| Unexpected failure | Exits unless code has a retry/recovery path | Host can inspect and sometimes recover interactively |
| Permissions | Can be configured for unattended trusted-environment execution | Host Claude Code policy may still require tool approval |
| Best for | Headless CI / unattended automation | Interactive development sessions |

Both modes require the `claude` executable to launch Claude. The distinction is
that CLI mode launches a fresh `claude -p` process for each Claude turn, while
skill mode performs host turns in the already-running Claude Code session.
Replacing or updating the executable on disk therefore affects a later CLI turn
but not the host turns already running in the current skill session.

### Containment and test gates

External skill agents and the skill-mode test gate use the same shared
containment policy as the standalone CLI. On a Linux host with systemd 253+
and delegated cgroup v2, a unique foreground child scope runs under the
per-user `agent-loop.slice`; aggregate and role-specific memory, swap, and
task limits protect host headroom across simultaneous processes. The exact
launcher is `systemd-run --user --scope --quiet` without `--wait`, `--service`,
or `--pipe`. Use the standalone `agent-loop containment-preflight` command to
inspect capabilities and resolved defaults before a live skill run.

Unsupported hosts visibly use process-group termination with no memory-ceiling
claim. `required` mode is available to fail closed. Optional peak, PSI, and
swap files are capability-aware. Resource-limit termination is reported as
`resource-exhausted`; pressure is not reported as a kill cause. The managed
test wrapper rejects an identical same-lane command until the prior attempt
exits or is explicitly terminated. The interactive Claude host session and
arbitrary descendants remain part of that host session, so prompt guidance and
the wrapper are the available boundaries for in-session work.

## Architecture

```
Claude Code (interactive session)
│
├── helpers/validate_response.py   ← validates structured protocol responses
├── helpers/state_manager.py       ← session state + GitHub comment resume
├── helpers/run_external.py        ← invokes codex / gemini / antigravity (agy) CLIs
├── helpers/gh_ops.py              ← GitHub issue/PR comment operations
└── helpers/demo_loop.py           ← standalone dry-run demo
```

Claude performs coder/plan turns by writing files directly (using its Write
tool or by producing structured JSON in its response).  External reviewers
(Codex, Gemini) are still invoked as subprocesses via `run_external.py`.  In
**reversed-roles** mode these swap — an external agent performs the coder/
implement turns via `run_external.py` while the host (Claude) reviews — see
[Reversed roles](#reversed-roles-external-coder--host-reviewer) below.

## Structured protocol compatibility

The skill helpers reuse the same library entry points used by the headless CLI:

- `_validate_plan_review_response` / `_validate_review_response` (unresolved_items)
- `_resume_plan_round` / `_resume_pr_round` (round_state)
- `parse_plan_state` / `validate_structured_plan_revision` /
  `validate_structured_issue_implementation` (protocol)

GitHub comment metadata markers (`AGENT_LOOP_META`) written by the skill are
identical to those written by the headless CLI, so mixed-mode operation (start
headless, resume in skill, or vice versa) is supported.

## Reversed roles (external coder / host reviewer)

Skill mode supports running the loop with the coder/reviewer roles reversed —
mirroring the CLI's `--coder` / `--reviewer` reversal — so an external agent
(Codex/Gemini) does the coder turns and the host (Claude) reviews:

- **Plan, reversed**: `run-plan-round --coder codex --reviewers gemini claude`
  has the external coder write the plan; the configured reviewers (external
  and/or the host) review it. `--coder claude` (the default) keeps the host as
  coder and requires `--plan-file`; an external `--coder` generates the plan
  instead.
- **Host as reviewer**: when `claude` is among `--reviewers`, the round runs the
  external reviewers first, then writes a review-request dir and reports
  `pending`. The host reads the posted plan/PR, writes its structured review
  there, and finalizes it with `complete-host-review --dir <dir>` (works for both
  plan and PR rounds); re-running the round then recomputes the final state.
- **Implement, reversed**: after a plan is approved, an external coder emits the
  typed `issue_implementation` result and may open a PR — see [Approved-plan
  execution helpers](#approved-plan-execution-helpers) — which the host then
  reviews with `run-pr-round`. A null-PR result is posted to the issue and stops
  before handoff. A positive PR combined with a blocked signed requirement is
  posted once as a rejected terminal conflict, preserving the real PR reference
  in the summary/evidence without retrying or entering review/merge gates. If
  that review blocks, `run-pr-fix --coder X --reviewers ... --workdir <push-capable
  clone>` sends the same external coder back to the open PR branch, posts a coder
  follow-up for the new head, and hands the PR back to `run-pr-round`.

End to end: external coder plans -> reviewers review -> external coder implements
(one-shot / decompose / by-phase) -> host + external reviewers review the PR ->
external coder fixes blocking PR review with `run-pr-fix` -> reviewers re-review.
Merge stays a human decision.

The external agent can be `codex`, `gemini`, or `antigravity` (the `agy` CLI —
the migration path for Gemini CLI consumer access, which Google retires on
2026-06-18). With no override, Antigravity uses the ordered fallback chain
`Gemini 3.7 Flash (High)` → `Gemini 3.6 Flash (High)` → `Gemini 3.1 Pro (High)`. Use `--model MODEL` for the
legacy single-model override or `--antigravity-models MODEL [MODEL ...]` for a
custom ordered chain; these options are mutually exclusive. Use
`--antigravity-quota-signatures SIGNATURE [SIGNATURE ...]` to customize the
output signatures that trigger fallback. Use
`--antigravity-print-timeout-seconds SECONDS` (default `600`, matching the
`agent-loop` CLI) to override the `--print-timeout` passed to each `agy
--print` call, which otherwise falls back to `agy`'s own five-minute
print-mode default. For a required local test that may run up to 1,800
seconds, configure this whole-invocation deadline above that cap with
additional budget for analysis, edits, reporting, and other turn work; the
default cannot accommodate it. These options are available on plan, task, PR-review,
implementation, decomposition, phased implementation, and PR-fix commands.
Antigravity turns are single-shot (no cross-round session resume) and report
estimated token usage.

Provider high-traffic and capacity failures retry the active model before
the ordered fallback chain advances. `--max-retries` is shared across the chain,
which caps a run at `models + retries` calls; custom quota signatures decide which
diagnostics qualify. Invalid settings, unsupported models, timeouts, and malformed
responses do not retry or fall back.

## Approved-plan execution helpers

Skill mode also exposes the external-coder execution helpers used after a plan
has already been approved:

- `run-implement` performs the existing one-shot reverse implementation. It
  validates and renders the external coder's `issue_implementation` result,
  validates structured test commands inside the assigned workdir, preserves the
  raw payload in accepted PR metadata, and stops before handoff for null-PR or
  rejected-conflict terminal results. It keeps using the durable
  `AGENT_PLAN_ONE_SHOT_IMPL` marker and is unchanged by by-phase support.
- `run-decompose` uses typed `child_stages` directly when present, otherwise
  decomposes an approved plan into child phase issues with mode
  `decompose-only`.
- `run-implement-by-phase` decomposes with mode `implement-by-phase`, then
  implements phase 1 only when that phase is `agent-pr`.

Both helpers use the shared parent-wide flat child cap (default 15, configurable
with `--flat-child-limit`), preflight every draft before mutation, and recover
from checkpoints and exact child identities. Over-limit requests return exit 2
with a structured consolidation-versus-hierarchy decision referencing #720.
Do not pass `--materialize-split-issues` to these commands; their topology is
selected once. See [Phased decomposition versus split materialization](local_agent_loop.md#phased-decomposition-versus-split-materialization).
- `run-pr-fix` addresses a settled blocking PR review for an externally opened
  PR. The target PR must be `OPEN`; `--reviewers` must exactly match the reviewer
  set used by the prior `run-pr-round`; and `--workdir` must be a clean,
  push-capable clone where the PR head branch can be checked out and pushed.

Example:

```bash
python -m helpers.skill_runner run-implement-by-phase \
  --issue 123 --repo owner/repo \
  --coder codex \
  --plan-file /tmp/approved-plan.md \
  --workdir /path/to/push-capable/clone \
  --base main
```

Live by-phase implementation creates or reuses child phase issues, posts the
parent decomposition summary, runs the external coder in the push-capable
workdir, and posts a phase implementation handoff only after the coder returns
an accepted positive PR. Null-PR and rejected-conflict terminal results stop
without handoff. Reruns with an existing phase handoff stop with a child
issue resume hint rather than invoking the coder again. If phase 1 is
`human-action` or `manual-close`, the command stops after decomposition and
reports the child issue that needs human work.

Dry-run by-phase execution validates the decomposition and implementation stubs
without creating issues, posting comments, pushing branches, or opening real
PRs. Because dry-run child issue numbers may be unavailable, its JSON output is
a preview of what would be handed off and implemented.

## Round states and reviewer resilience

Each `run-plan-round` / `run-pr-round` prints a JSON result whose `state` is one
of the following, in precedence order:

- `pending` — a host (`claude`) review handoff is outstanding (listed under
  `pending_reviewers`); complete it with `complete-host-review`, then re-run.
- `blocking` — a completed reviewer reported must-fix items (`blocking_items`).
- `incomplete` — a configured external reviewer was **unavailable** (listed under
  `unavailable_reviewers`); re-attempted on the next run.
- `approved` — all configured reviewers signed off.

**Reviewer resilience**: if an external reviewer's CLI fails to produce a usable
review — an agent/tooling failure such as an empty or malformed-tool-call
response, *not* a fixable malformed review — the round does **not** abort. That
reviewer is marked unavailable, the remaining reviewers still run, and a round is
never falsely reported `approved`. A malformed-but-content-bearing structured
review is recovered automatically when possible: safe skill normalization,
envelope normalization, deterministic unknown-prior-item stripping against the
complete carried ledger, and then model format repair. Local-loop repair now
defaults to isolated Antigravity with `Gemini 3.7 Flash (Medium)`; explicit repeatable
repair models are tried first, followed by the configured Antigravity chain. Legacy
Gemini CLI repair requires
`--repair-backend gemini` and suitable non-interactive authentication. The
classifier is conservative: only known tooling-failure signatures or
truly-empty output count as unavailable.

The same automatic recovery pipeline covers external-coder `plan_revision`
responses and `run-pr-fix` `coder_followup` responses. Plan revisions may also
recover one unique, independently valid signed-human-requirements
acknowledgement from the external agent's captured response evidence.
`skill_runner` requests that sidecar from `run_external` with
`--response-evidence-output PATH`, alongside `--usage-output PATH`. Use
`--gemini-cmd PATH` to configure both Gemini agent invocations and the final
repair pass. The original response is saved before recovery; `retry-validate`
repair directories and PR-fix debug directories remain the final fallback for
genuinely unrecoverable output.

## No unbounded CI waits

A queued GitHub check-run can sit indefinitely with no job started during a
hosted-runner capacity incident, or be cancelled before execution because a
runner could not be acquired. The orchestrating Claude session running this
skill (the host coder) must not run `gh run watch`, `gh pr checks --watch`, or
any other unbounded wait on such a check — including while fixing a
reviewer-reported check failure. Take at most 3 bounded status snapshots
(`gh run view <run-id> --json status,conclusion,startedAt` or `gh pr
checks`), spaced at least 30 seconds apart, for at most 120 seconds of total
CI observation, then stop with a resumable result instead of waiting further:
name the affected check/run and note that work should resume once GitHub
Actions runners recover, rather than treating the stall as a code defect. A
check that is actively running, or that fails due to a real repository test
failure, is unaffected — that remains real work to investigate and fix.

The `coding_review_agent_loop` CLI orchestrator (used directly, outside skill
mode) applies the equivalent bound automatically, plus board-wide stall
classification driven by `--ci-queued-grace-seconds`; see [External CI
infrastructure
stalls](local_agent_loop.md#external-ci-infrastructure-stalls). Every coder
prompt built by `coding_review_agent_loop.prompts` — including the ones
`helpers/prompt_builders.py` reuses for skill mode's `run-pr-fix` — carries
the same bounded-observation guidance, so this applies whether the coder turn
runs through the CLI orchestrator or through the skill's own external-coder
path.

## Focused, bounded local test runs

A same-PR follow-up or reviewer item scoped to a couple of files does not
justify running the whole `tests/` suite, an `--ignore` list that amounts to
the whole suite, or a broad server/database/integration suite — those add
minutes of runtime and hide progress when they are launched in the background
and polled from an auxiliary shell loop. The host coder must select tests
proportionate to the files actually changed and the reviewer item being
addressed, prefer the repository's verified focused test command when one
covers the change, and give a one-line rationale tying each selected test
module to a changed file or reviewer item. Broad suites remain available when
the change genuinely touches those surfaces, focused tests demonstrably do
not cover it, or a human or the issue explicitly asked for full-suite
verification. Required completion tests must run in the foreground with
visible output under an explicit bounded timeout no greater than the configured
finite run-level ceiling (1,800 seconds by default). That maximum allowance
requires an individually justified command; it is not a reason to choose a
broad suite. Never launch
pytest in the background or poll a process ID, `ps`/`kill -0`/`wait`, or a
task-output file to learn whether it finished. If a required test exceeds its
bound, terminate it and report the blocker immediately, naming the exact
command and the timeout, instead of waiting silently or retrying with a
broader selection.

Every coder prompt built by `coding_review_agent_loop.prompts` — including the
ones `helpers/prompt_builders.py` reuses for skill mode's `run-pr-fix` —
carries this policy, but the skill's host coder drives its own shell commands
directly and is not itself constrained by the CLI orchestrator's test
plumbing, so this section states the policy explicitly for that path.

## Session state

Local session state is stored at:

```
~/.local/state/coding-review-agent-loop/skill-sessions/{owner-repo}/{issue}.json
```

This path is outside any git checkout so it never dirties a working tree.
Fields written by `state_manager write-session`:

| Field | Description |
|-------|-------------|
| `last_completed_step` | Most recently completed orchestration step |
| `session_id` | Current skill session UUID prefix |
| `round_number` | Current plan/PR round number |
| `pending_comment_body` | Path to a comment body not yet posted |

The `pending_comment_body` field provides crash recovery: if the session ends
after writing the comment file but before posting it, the next `build-resume`
call includes the path so Claude can re-post it.

## Resume from existing round

`state_manager build-resume` reads GitHub issue comments, extracts all
`AGENT_LOOP_META` markers (versioned `v1_` zlib-compressed URL-safe base64,
with legacy plain-base64 support), calls `_resume_plan_round(comments,
configured_reviewers=...)` or `_resume_pr_round(comments, head_sha=...,
configured_reviewers=...)`, and outputs a JSON descriptor:

```json
{
  "round_number": 2,
  "prior_items": [...],
  "compact_prior_summaries": [...],
  "completed_reviewer_names": ["Codex"],
  "pending_comment_body": null,
  "current_plan": "..."
}
```

Oversized metadata is split into `AGENT_LOOP_SIDECAR` comments posted before its
anchor. These are protocol transport records rather than agent replies; retain
them. If resume reports missing sidecars, restore them or remove the incomplete
anchor and rerun.

The skill then skips already-completed reviewer turns and resumes from where
the last session ended.

**Important**: `--reviewers` must exactly match the configured reviewer list for
the current invocation.  For PR-flow sessions, `--head-sha` or `--pr` is also
required so `_resume_pr_round` can compare the current PR head SHA.

## Billing and terms

Running Claude turns inside an interactive Claude Code session may count
differently toward billing than `claude -p` / Agent SDK invocations.  Whether
this constitutes "interactive" or "programmatic" use depends on Anthropic's
current terms and product behavior at the time of use.

**Non-goals / constraints**:
- Do not use this skill to proxy one user's session to other users.
- Do not build unattended 24/7 automation that relies on pretending to be
  interactive use.
- Do not market this as free Claude access or billing bypass.
- The existing headless `agent-loop` CLI path is unchanged and unaffected.

## Install / setup for open-source users

1. Clone the repository and install in development mode:
   ```
   pip install -e ".[dev]"
   ```
2. Copy or symlink `helpers/` and `SKILL.md` into your working directory
   (or run from the repo root).
3. Authenticate `gh` and install `codex` / `gemini` CLIs as needed.
4. Run the demo to verify the install:
   ```
   python -m helpers.demo_loop --issue 123 --repo demo/repo
   ```

## Runtime-aware local test timeouts

Skill local test gates use the finite
`--coder-test-command-timeout-seconds SECONDS` ceiling (default 1,800) and
return `timed_out` with a bounded diagnostic tail in their JSON result. This
whole-command watchdog is distinct from framework per-test and backend
whole-turn limits. The optional `agent-loop run-tests [--timeout-seconds N] --
COMMAND...` wrapper inherits the ceiling from its environment, defaults to
1,800 outside agent-loop, rejects malformed or over-ceiling requests before
spawn, and records measured observations only when skill memory is enabled and
writable. Long browser/integration matrices remain finite and can be split or
sharded when useful.

## Known limitations

- Reviewer subprocess progress (Codex, Gemini) is not streamed to Claude's
  terminal while the subprocess runs.  Check logs in
  `/tmp/coding-review-agent-loop/skill-logs/`.
- If the Claude Code session ends mid-loop, the next session must call
  `build-resume` to reconstruct the round state from GitHub comments.
- The structured protocol versions must match; update both the library and
  the skill helpers together when the protocol evolves.
- Future Antigravity CLI migration (#215) may require updates to
  `run_external.py` when the `gemini` CLI name or interface changes.

## Related

- `SKILL.md` — step-by-step skill orchestration instructions for Claude.
- Issue #216 — original exploration proposal.
- Issue #215 — Antigravity CLI migration for Gemini CLI consumer users.
