# coding-review-agent-loop

Local command-line orchestration for a coding PR review loop.

Run a local Claude/Codex/Gemini PR review loop using your existing CLI subscriptions.

The main advantage is account reuse: the tool shells out to your
already-authenticated local CLIs (`claude`, `codex`, `gemini`, and `gh`) instead
of calling model APIs directly. If your local agent CLIs are backed by existing
AI subscriptions or authenticated developer accounts, the review loop can use
those existing entitlements rather than requiring separate model API keys.

**Claude billing note:** Anthropic had announced that non-interactive `claude`
usage — including `claude -p` as used by this tool — would move from your
subscription's rate limits to a separate monthly Agent SDK credit. As of
June 15, 2026 that change has been **postponed**: `claude -p` / Agent SDK usage
continues to draw from your existing Claude subscription as before, with no
separate credit, and Anthropic has said it will give advance notice before any
future change. See
[Anthropic's support article](https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan)
for the latest. Gemini CLI and Codex CLI have their own separate billing models.

## Who This Is For

This is for developers who already use Claude Code, OpenAI Codex CLI, Gemini
CLI, and GitHub, and want one local agent to implement or fix a PR while
another local agent reviews it before merge.

It is especially useful when you are already doing this manually by switching
between agent CLIs and copying review feedback back and forth.

## Why Not GitHub Actions?

GitHub Actions-based agent loops usually need model API keys, hosted workflow
permissions, and separate API billing. This tool keeps the loop on your local
machine and uses the CLI accounts you have already authenticated.

That makes it easier to experiment with agent-to-agent review loops before
committing to hosted automation. It also keeps local workspace setup,
credentials, and agent approval prompts under your direct control.

Note that Claude subscriptions have their own usage limits, and Anthropic's
terms for non-interactive (`claude -p` / Agent SDK) usage may change in the
future (see the billing note above) — so very high-volume automated use may
incur costs or hit limits depending on your plan.

## Compared To Similar Tools

Several related projects exist. `coding-review-agent-loop` is deliberately
positioned as a standalone local CLI for GitHub PR lifecycle orchestration:
one agent creates or fixes a PR, one or more reviewers review it, and the loop
continues until approval.

| Tool | Focus | How this project differs |
|------|-------|--------------------------|
| [claude-review-loop](https://github.com/hamelsmu/claude-review-loop) | Claude Code plugin that has Claude implement, then Codex review. | This project is not a Claude plugin; it is a standalone CLI that can start from an issue, task, or existing PR and can reverse the coder/reviewer direction. |
| [codex-review](https://github.com/boyand/codex-review) | Claude Code plugin for Codex review of plans and implementations. | This project focuses on GitHub PR creation, review, fix, and approval loops rather than plan/artifact review inside Claude Code. |
| [reviewd](https://pypi.org/project/reviewd/) | Local PR review assistant for GitHub/BitBucket using Claude, Gemini, or Codex CLI. | This project focuses on agent-to-agent implementation loops where the coder can create/fix the PR and reviewers gate approval. |
| [codex-plugin-cc](https://github.com/openai/codex-plugin-cc) | Use Codex from inside Claude Code for review or delegated tasks. | This project stays outside either agent host and orchestrates local CLIs plus GitHub directly. |

## Agent Backends

Currently supported local agent CLIs:

- Claude Code via `claude`
- OpenAI Codex CLI via `codex`
- Gemini CLI via `gemini`
- Antigravity CLI via `agy` (first-class backend; also the Gemini CLI migration path — see below)

The `agy` backend is supported in every role the other external agents support — `--coder antigravity` and `--reviewer antigravity` — and in skill mode (`--coder antigravity` / `--reviewers antigravity`). `agy` is also accepted as an alias for `antigravity` in these flags (e.g. `--coder agy`, `--reviewer agy`, skill `--reviewers agy`, `run_external --agent agy`); it is normalized to the canonical `antigravity` internally.

### Gemini CLI → Antigravity migration

Google is retiring **Gemini CLI consumer access** (free / Google AI Pro / Ultra)
on **June 18, 2026**; personal-account `gemini` usage stops working after that
(enterprise / API-key Gemini CLI paths remain supported). Use the Antigravity CLI
(`agy`) instead — it runs the same Google account plans with its own quota model:

```bash
# install agy, authenticate, then select it as a coder or reviewer:
agent-loop pr 123 --repo OWNER/REPO --reviewer antigravity
agent-loop task "Fix the flaky test" --repo OWNER/REPO --coder antigravity --reviewer codex
```

Pick the model with `--antigravity-model "<name>"` (as listed by `agy models`;
default `Gemini 3.1 Pro (High)`). When a `gemini` invocation fails with an
auth/quota error near or after the cutoff, the tool surfaces this migration
guidance. Notes: Antigravity turns are single-shot (no cross-round session
resume) and report estimated token usage (`agy` emits no token counts).

**Billing / quota note:**
- `agy` usage counts against a **separate Antigravity-specific quota**, not the same token pool as the Gemini app/chat in your subscription. The two meters are tracked independently and can diverge.
- Your Google AI subscription tier (Pro/Ultra) **raises** the Antigravity limits rather than sharing one pool; the free / Google One tier is small.
- When the included quota is exhausted, continued use draws on **Google AI credits** (pay-as-you-go), so monitor actual spend for the first runs.
- Exact limits are not officially well-documented and have changed since launch — treat numbers as fluid.

## Install / Use

Clone the repo first:

```bash
gh repo clone wwind123/coding-review-agent-loop
cd coding-review-agent-loop
```

Then install the CLI into a local virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
agent-loop --help
```

This installs the `agent-loop` command from your checkout. The tool still
requires local `gh`, `claude`, `codex`, and/or `gemini` authentication depending
on which agents you use.

## Claude Code skill mode

Besides the headless `agent-loop` CLI, the repo ships a **Claude Code skill** that
runs the same review loop directly inside an interactive Claude Code session
(host Claude turns use your session instead of `claude -p`). It supports the
reversed roles end to end: an external agent (Codex/Gemini/Antigravity) can
plan, implement (one-shot, decompose, or by-phase), address blocking PR review
with `run-pr-fix`, and hand the PR back for re-review; the host (Claude) can
review. See [`SKILL.md`](SKILL.md) for the step-by-step instructions and
[`docs/skill_mode.md`](docs/skill_mode.md) for the design overview.

### Skill vs CLI — which to use

Both drive the same review loop. External agents — Codex, Gemini, and
Antigravity — still run as subprocesses either way; the main difference is
whether Claude's turns run as isolated `claude -p` calls or in the active
Claude Code session.

| Concern | CLI (`agent-loop`) | Claude Code skill |
|---|---|---|
| Claude runtime | Each Claude turn starts a separate `claude -p` subprocess. The `claude` binary must be installed and available for every turn; an update or replacement can affect the next turn. | The skill also requires the `claude` binary to start Claude Code. Host turns then run in that already-active session, so replacing the binary on disk does not change the running session or require a new `claude -p` process. |
| Model selection | Uses the Claude CLI default unless `--claude-model` is supplied. | Uses the active session model. |
| Claude token use | Usually lower for equivalent work because round control, validation, retries, and state transitions are mechanical Python orchestration; Claude tokens are spent on the explicit Claude agent turns. | Can use more Claude quota because the host session must interpret state and execute the orchestration workflow as well as perform Claude's coder or reviewer work. The actual difference depends on task size, session context, caching, and model behavior. |
| Configuration | Flags and parameters must be supplied correctly up front. `--help` documents the available choices. | Configuration is conversational: Claude can explain options, remind you of parameters, and translate intent into helper commands even when you do not remember exact flag names. |
| Unattended operation | Best suited to scripts, cron, and fire-and-forget runs. With `--test-command` and `--auto-merge`, it can run test gates, wait for CI, and merge. Without those flags, it does not add those behaviors. | Intended for an attended session. It lets you watch and steer rounds, and it keeps merge as a human decision. It never auto-merges or waits for CI. |
| Quota exhaustion | The process stops, but durable GitHub metadata supports resume. You can arrange a shell scheduler or other external job to rerun the command after the reported reset time and leave it unattended. | If the host Claude session exhausts its quota, that session cannot schedule or perform its own later continuation. You must return after reset and resume or start a new session. |
| Unexpected failures | Failures outside the implemented retry/repair paths normally abort the command and require a later diagnostic or code change. | The host Claude can inspect logs and state, explain the failure, and sometimes perform a safe manual recovery or adapt the next step. This is useful but not guaranteed. |
| Permission prompts | With the appropriate trusted-environment flags, the loop can run without interactive approvals. | Claude Code's own security policy remains in force. The host may still request tool permission during a long run even when the skill instructions ask it not to prompt for particular commands. |

Rule of thumb: **use the CLI for predictable, unattended execution and
scheduled resume; use the skill for conversational setup, active oversight,
and hands-on recovery from unusual failures.**

Keeping the skill also **reduces reliance on programmatic `claude -p`** for Claude turns —
useful if `claude -p` is ever billed or restricted differently (such a change was announced
once, then reversed). Whether interactive-session usage is actually treated differently
from `claude -p` depends on Anthropic's current terms and product behavior; see the billing
note in [`SKILL.md`](SKILL.md). This is about reducing the `claude -p` dependency, not a
guaranteed billing outcome.

## Develop This Tool

Use this if you are changing `coding-review-agent-loop` itself:

```bash
gh repo clone wwind123/coding-review-agent-loop
cd coding-review-agent-loop
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

## Quick Start

Start from a GitHub issue when you want the agent loop to use the issue title,
body, and comments as the implementation task. Comments are included oldest to
newest, and prompts tell agents that later comments may refine or supersede the
original issue body:

```bash
agent-loop issue 123 --repo OWNER/REPO
```

For larger or ambiguous issues, add `--plan-first` to run a plan review on the
issue before code is written. The coder may inspect the checkout but must not
edit files, push, or open a PR during planning. Reviewers approve or block with
`AGENT_PLAN_STATE` markers using explicit plan-review sections:

```md
### Blocking plan issues
### Same-plan follow-ups
### Future follow-ups
```

When earlier plan issues remain open, reviewers encode prior item dispositions
in the JSON `prior_plan_item_dispositions` array using `"resolved"`,
`"blocking"`, `"same-plan"`, or `"future"` (with a `"note"`). The orchestrator
renders those as a `### Prior unresolved plan item dispositions` section in the
public GitHub comment; reviewers do not add that section themselves. `"future"`
dispositions are accepted only in approved plan reviews and are reconciled with
the final approved plan instead of reopening planning. If
`--approved-followups=issue` or `fix-and-issue` is enabled and implementation
will continue after approval, those plan-stage future follow-ups are filed as
separate issues before implementation starts. If implementation continues but
issue filing is disabled, they are summarized in the planning-complete comment
with an explicit note that they are not carried into PR review. Planning
`item-*` IDs visible in issue history are not PR prior review items unless they
are repeated in the active PR unresolved-item ledger. By default the loop posts
the approved plan summary and stops without filing follow-up issues; add
`--implement-after-approval` to continue into the normal PR flow:

```bash
agent-loop issue 123 --repo OWNER/REPO --plan-first --implement-after-approval
```

`--plan-first` also supports explicit post-approval modes:

```bash
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode plan-only
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode decompose-only
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode implement-one-shot
agent-loop issue 123 --repo OWNER/REPO --plan-first --plan-execution-mode implement-by-phase
```

`plan-only` is the default. `implement-one-shot` is the same behavior selected
by the backward-compatible `--implement-after-approval` flag. `decompose-only`
asks the coder to turn the approved plan into ordered phases, always creates
one GitHub child issue per phase, posts a parent summary table, and stops.
`implement-by-phase` creates every child issue, then implements only the first
`agent-pr` phase and stops after that PR review loop. Before entering that child
implementation, the parent issue records a one-time handoff marker. Parent
reruns after that marker do not re-run the child implementation; resume directly
with `agent-loop issue <child>`. If decomposition already exists without a
handoff marker, the first child is treated as not yet attempted and the handoff
is recorded once. If the first phase is `human-action` or `manual-close`, the
loop creates and reports all child issues but stops so a human can do the
required work, add a remark/update, and close that child issue.

Each generated child issue copies the relevant parent-plan slice, constraints
and invariants, dependency notes, scope and non-goals, rollout risk,
validation/soak requirements, automation classification, and instructions for
agent execution or human closure. Decomposition is capped at 8 phases; an
over-cap response is rejected and must be consolidated, not truncated. This cap
is separate from the approved-review follow-up issue cap used by
`--approved-followups`.

Provide a one-off task directly when there is no issue yet:

```bash
agent-loop task "Add a health check endpoint" --repo OWNER/REPO
```

Run the loop against an existing pull request when you want another review and
iteration pass:

```bash
agent-loop pr 456 --repo OWNER/REPO
```

When `--base` is omitted, `pr` mode uses the pull request's base branch.
`issue` and `task` modes use the repository default branch. If PR metadata does
not include a base branch, `pr` mode also falls back to the repository default.
An explicit `--base BRANCH` always takes precedence.

If `--repo` is omitted, the tool runs `gh repo view` from the current working
directory, or from `--codex-dir` when that flag is provided, and uses the
detected `OWNER/REPO`. Pass `--repo` explicitly when running outside the target
repository.

When `--claude-dir`, `--codex-dir`, or `--gemini-dir` is omitted for an active
agent, the tool creates or reuses a repo-scoped temporary checkout such as
`/tmp/coding-review-agent-loop/OWNER-REPO/codex/repo`. Existing clean temp
checkouts are fetched and fast-forwarded on the resolved base branch before the agent
runs. Default temp checkouts are tool-owned and disposable; if one is dirty,
the tool resets and cleans it before reuse. Explicit persistent directories are
kept conservative: dirty explicit workdirs fail clearly, and existing git
checkouts must point at the requested repository. Use explicit persistent
directories for large repositories, long-lived agent worktrees, or setups that
should survive `/tmp` cleanup or reboot.

Agent memory is enabled by default. Before invoking agents, the loop creates or
refreshes advisory repo memory in a durable, repo-scoped user cache directory
such as `~/.cache/coding-review-agent-loop/repos/OWNER-REPO/memory` on Linux:
repo summary, architecture map, module index, execution/test profile, toolchain
facts, and changed files since the previous memory commit. On macOS the default
root is `~/Library/Caches/coding-review-agent-loop`; on Windows it is
`%LOCALAPPDATA%/coding-review-agent-loop/Cache`. Agent prompts state that this
cache is stale-prone orientation only, and that agents must inspect source files
and PR diffs directly for correctness claims. The cache is local-only. Disable
it with `--no-agent-memory`, force a refresh with `--refresh-agent-memory`,
customize the location with `--agent-memory-dir PATH`, or refresh only test
command facts with `--refresh-test-profile`. Relative `--agent-memory-dir`
values are resolved inside the coder checkout. If you keep sensitive repo
details out of local cache retention, use `--no-agent-memory` or a custom
short-lived location. If the previous memory commit is unavailable or no longer
diffable, the loop logs the git failure and treats all tracked files as changed
for that refresh.

Use `--test-command` to add a local test gate:

```bash
agent-loop task "Fix the flaky test" --repo OWNER/REPO --test-command "python -m pytest"
```

By default, the command runs after coder-created or coder-updated changes before
reviewer rounds, and again after final reviewer approval before auto-merge. Add
`--no-pre-review-tests` if you only want the final post-approval local test
gate. The coder prompt also asks the coding agent to report the exact tests it
ran, or explain why it could not run tests.

By default Claude is the coder and Codex is the reviewer. Reverse that with:

```bash
agent-loop task "Fix the flaky test" --repo OWNER/REPO --coder codex --reviewer claude
```

Use Gemini as either side of the loop:

```bash
agent-loop task "Improve error handling" \
  --repo OWNER/REPO \
  --coder gemini \
  --reviewer codex

agent-loop pr 456 \
  --repo OWNER/REPO \
  --reviewer gemini
```

Repeat `--reviewer` to require approvals from multiple reviewers. The PR is
approved only after every configured reviewer approves in the same round. The
coder may also be listed as a reviewer when you want the same agent to work in
separate coding and review passes:

```bash
agent-loop pr 456 --repo OWNER/REPO --reviewer codex --reviewer claude
```

Structured-response runs use a three-level interpretation order:

1. Structured JSON payloads are authoritative when present in the agent output.
2. For resume/replay, `AGENT_LOOP_META` attached to orchestrator-posted comments is the canonical source of the active round ledger, carried `prior_items`, completed reviewer dispositions, and next `item-N` allocation for that structured-response round.
3. Markdown section parsing remains a compatibility fallback for interpreting comments that do not include a structured payload or metadata for the current round.

Mixed histories are expected during rollout. Old raw-markdown comments can
remain earlier in the issue or PR thread, while newer orchestrator-rendered
comments carry `AGENT_LOOP_META`. When metadata exists for the current head or
plan subject, resume reconstruction uses that metadata-backed ledger and ignores
stale visible item IDs from older heads, superseded plans, or replayed rounds.
If a PR head advances but no current-head coder metadata was recorded, the PR
loop recovers from metadata-backed active `blocking` and `same-pr` items on the
latest recorded head and routes them through a coder follow-up before reviewers
run again.

Structured JSON is also the preferred coder format for follow-up and plan
revision rounds. Coder follow-up responses use `kind: "coder_followup"` with
`state`, `summary`, `addressed_items`, `remaining_items`,
`human_requirements`, optional `addressed_item_notes` / `remaining_item_notes`,
and optional `tests_run`; every carried reviewer item ID must appear exactly
once in either `addressed_items` or `remaining_items`.
Plan-revision responses use `kind: "plan_revision"` with `state: "blocking"`,
`summary`, `prior_plan_item_dispositions`, and `plan_steps`. Structured
responses must start with one top-level JSON object, place the matching
`AGENT_STATE` or `AGENT_PLAN_STATE` footer immediately after it, and end with
only the standalone agent signature. The loop renders validated structured
payloads into normal public GitHub comments, so raw JSON is not posted.

When a structured plan review, plan revision, PR review, or coder follow-up is
present but malformed, the loop may run one Gemini-backed repair pass. The
repair pass is format-only: it asks Gemini to re-emit the agent's intent as the
required JSON object, footer marker, and signature. The repaired response is
accepted only if it passes the same strict validation as the original response;
failed repairs remain local protocol errors and are not posted to GitHub.

Signed human reviewer comments are approval-critical when they end with a
standalone `-- Human Reviewer` signature. The loop surfaces those requirements
to coders and reviewers. In markdown fallback paths, coders must include
`<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` plus a `### Human requirements` section
that covers every surfaced `Requirement N`; in structured coder follow-ups, the
same acknowledgement is carried in the `human_requirements` object. If details
were omitted to keep a prompt bounded, the coder must state that it checked the
GitHub discussion directly before responding. Reviewers cannot approve signed
requirements as resolved unless the approved review includes
`<!-- HUMAN_REQUIREMENTS_RESOLVED -->`; otherwise the loop carries a synthetic
human-requirements acknowledgement item into the next round.

When `--approved-followups` is set to `summarize`, `issue`, or a `fix-and-*`
mode, approved reviews may include future work under:

```md
### Future follow-ups
- Add a follow-up test.
```

Reviewers should use that section only for substantial work that is better
handled in a separate issue or PR. The legacy heading
`### Non-blocking follow-ups` is still parsed as future work for compatibility.
Approval means the review is fully complete for that round: no new blocking
work, and no carried-forward unresolved items left active in the reviewer’s
disposition section.

When `--approved-followups` uses a `fix-and-*` mode, blocking reviews may also
include small, localized, low-risk current-PR cleanup under:

```md
### Same-PR follow-ups
- Rename a helper before merge.
```

Same-PR follow-ups are sent back to the coder in the existing PR and require a
new review round. They may not appear in an approved review. They should stay
narrowly scoped to files already touched by the PR or directly adjacent code;
larger redesigns and independent work belong under Future follow-ups. Approved
future follow-ups remain in the round-to-round
ledger so later reviewers can explicitly confirm they are still future work,
resolved, or should be promoted back to same-PR or blocking status. The final
summary or issue creation uses the remaining future items from that reconciled
ledger, not only the final round's newly written Future follow-ups.

Before posting summaries or creating issues, the loop deduplicates the remaining
future items across reviewers using deterministic topic keys from headings,
code identifiers, docs/files, and normalized wording. The selected issue body
keeps the canonical wording plus an `Original reviewer notes` section so
reviewer provenance and later disposition notes are not lost. The issue modes
create at most three follow-up issues to avoid issue noise, and the final PR
comment reports how many items were filed or summarized, deduplicated, or
skipped by the cap.

Plan reviews follow the same rule: approved plan reviews may include Future
follow-ups only. Blocking plan issues, Same-plan follow-ups, or carried-forward
plan items left `still blocking` or `same-plan` keep the planning round
unapproved. Plan-stage future follow-up issues are filed before implementation
begins in issue-filing modes; PR-stage approved-review future follow-up issues
are filed after final PR approval.

By default, `--approved-followups=ignore` asks reviewers not to include
approved-review follow-up sections. Reviewers should mark the review blocking
instead when cleanup should be fixed before merge.

Each top-level `issue`, `task`, or `pr` run also writes a machine-readable
usage summary beside the normal agent logs in `--log-dir` as
`<run-id>-usage-summary.json`. The file aggregates per-call, per-agent, and
whole-run usage, including retries. When a backend exposes token counters, the
summary records them as `exact` or `partial`; when a backend exposes no usable
usage data, the loop falls back to a clearly labeled estimate based on prompt
and public-response size. `--dry-run` does not fabricate token usage.

`--approved-followups` accepts:

- `ignore`: ignore approved follow-up sections. This is the default.
- `summarize`: post future follow-ups as a grouped PR comment.
- `issue`: create GitHub issues for future follow-ups, then comment with the created issue links.
- `fix-and-summarize`: send same-PR follow-ups to the coder for another review round, then summarize future follow-ups after final approval.
- `fix-and-issue`: send same-PR follow-ups to the coder for another review round, then create issues for future follow-ups after final approval and comment with the created issue links.

To keep a grouped record on the PR or create follow-up issues, use:

```bash
agent-loop pr 456 --repo OWNER/REPO --approved-followups summarize
agent-loop pr 456 --repo OWNER/REPO --approved-followups issue
agent-loop pr 456 --repo OWNER/REPO --approved-followups fix-and-summarize
```

Bullets and prose paragraphs inside the `Same-PR follow-ups`, `Future follow-ups`,
and legacy `Non-blocking follow-ups` sections are parsed. Each section ends at
the next heading, HTML marker, or agent signature, so final protocol markers
are not mistaken for follow-up text.

The remaining legacy compatibility surface is intentional:

- Markdown review/plan parsing remains supported in the resume path for
  already-completed reviewer rounds that predate structured metadata.
- The legacy heading `### Non-blocking follow-ups` still maps to future work.
- Marker-only markdown paths remain compatibility fallbacks; new follow-up,
  review, and plan-revision examples should use structured JSON first.
- Resume reconstruction should not depend on reparsing old prose once
  `AGENT_LOOP_META` exists for the active structured-response round.

Agent subprocess logs are written under `.agent-loop-logs/` in the active
checkout, and long-running agents print heartbeat lines with the exact log
path. GitHub comments come from validated public response files under
`/tmp/coding-review-agent-loop/responses/...` or validated fallback stdout, not
from raw logs. If an agent looks stuck or returns diagnostics, inspect the
heartbeat log path and the response-file path; quota/reset failures may exit
early with rerun guidance, while narrower transient failures retry according to
`--agent-max-retries` and `--agent-retry-backoff-seconds`. Repair-pass attempts
are also visible in the log as schema-validation failure, repair attempt, and
recovered-or-invalid repair messages.

For trusted local automation that must run without approval prompts:

```bash
agent-loop issue 123 --repo OWNER/REPO --dangerous-agent-permissions
```

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

See [docs/local_agent_loop.md](docs/local_agent_loop.md) for the architecture diagram, full usage, and safety notes.

## Test

```bash
python -m pytest
```

Tests use fake subprocess runners. They do not call real `claude`, `codex`, `gemini`, or `gh`.
