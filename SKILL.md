# Agent Loop Skill — Claude Code Native Mode

This skill lets you run the `coding-review-agent-loop` orchestration directly inside
an interactive Claude Code session, without calling `claude -p` for Claude turns.

Claude (you, the host) performs coder/plan turns using your active session context.
External agents (Codex, Gemini) are invoked via their local CLIs as subprocesses.
GitHub operations go through `gh`.

## Prerequisites

- `gh` authenticated and configured.
- `codex` CLI installed (for Codex reviewer turns).
- `gemini` CLI installed (for Gemini reviewer turns).
- The `coding-review-agent-loop` package importable from `src/` (run from repo root).

## How to invoke this skill

Open a Claude Code session **in the `coding-review-agent-loop` repo root** (or
any directory where `helpers/` is on the Python path).  Then tell Claude what
you want in natural language, for example:

```
Run the agent-loop skill for issue #123 in myorg/myrepo with gemini as reviewer.
```

```
Plan and implement issue #42 in myorg/myrepo, reviewers: codex and gemini.
```

```
Run agent-loop on PR #99 in myorg/myrepo.
```

```
Resume the agent-loop skill for issue #123 in myorg/myrepo.
```

Claude will read this file, pick a mode, and follow the procedure below.  You do
not need to type any slash command; natural-language requests are enough.

---

## Choosing a mode

This is one skill with three modes; pick the procedure from what you're given:

- A **PR number** to review → **PR mode** (run the PR-loop).
- An **issue number**, or you want a plan first → **issue mode** (plan-loop, then
  *optionally* implement + PR-loop). Confirm whether to stop at an approved plan
  or continue into implementation.
- A **free-form task** with no issue yet → **task mode** (create a scratch issue,
  then proceed as issue mode).

All modes share the sub-procedures below, the same primitives, the same session/
resume model, and the same posture: **merge is always a human decision.** For any
mode, you need `OWNER/REPO` and the reviewer set (`codex`, `gemini`, and/or
`antigravity` — the `agy` CLI, the migration path for Gemini CLI consumer access
that Google retires on 2026-06-18). `antigravity` works wherever an external
coder/reviewer does (`--coder antigravity` / `--reviewers antigravity ...`).

---

## Sub-procedures

These are the building blocks; the modes sequence them.

### Plan-loop (for issue N)

1. Write the implementation plan to
   `/tmp/agent-loop-skill/{session-id}/plan-r{N}.md`, ending with:
   ```
   <!-- AGENT_PLAN_STATE: approved -->
   -- Anthropic Claude
   ```
2. Run one round. `skill_runner` handles resume, plan validation, attaching
   `AGENT_LOOP_META`, posting to GitHub, running each reviewer, rendering/
   validating responses, and writing session state:
   ```bash
   python -m helpers.skill_runner run-plan-round \
     --issue ISSUE --repo OWNER/REPO \
     --plan-file /tmp/agent-loop-skill/{session-id}/plan-r{N}.md \
     --reviewers codex gemini \
     [--workdir-codex /path/to/checkout] [--workdir-gemini /path/to/checkout]
   ```
   It prints a JSON result:
   ```json
   { "state": "approved" | "blocking", "round_number": N,
     "blocking_items": [...], "approved_reviewers": [...] }
   ```
3. Decide:
   - `"blocking"` → address `blocking_items`, post a change-summary comment
     (template below), write a revised plan, and re-run. The round number
     increments automatically.
   - `"approved"` with empty `blocking_items` → **planning is complete.**
   - Clarification needed → post an `<!-- AGENT_CLARIFY -->` comment and stop.

Change-summary template (`EOF` must be flush-left when run in a shell):
```bash
gh issue comment ISSUE --repo OWNER/REPO --body "$(cat <<'EOF'
Addressed round-N feedback:

- **item-X**: <what was changed>
- **item-Y**: <what was changed>

-- <Your Name>
EOF
)"
```

### Implement step (after an approved plan)

Run this only when the user asked to implement (see **issue mode**):

1. If a PR for this issue already exists (e.g. you were interrupted), resume it —
   do **not** open a second one.
2. Implement the approved plan in your working tree on a feature branch and commit.
3. Open a PR that references the issue, and note the PR number. Hand off to the
   PR-loop.

### External by-phase implementation (after an approved plan)

Use this when the user asks for skill-mode parity with
`plan_execution_mode="implement-by-phase"` and an external coder should perform
the first automated phase:

```bash
python -m helpers.skill_runner run-implement-by-phase \
  --issue ISSUE --repo OWNER/REPO \
  --coder codex \
  --plan-file /tmp/agent-loop-skill/{session-id}/approved-plan.md \
  --workdir /path/to/push-capable/clone \
  [--base main]
```

Live runs require a push-capable `--workdir` (or `--workdir-codex` /
`--workdir-gemini` / `--workdir-antigravity`). The command decomposes the
approved parent plan using an
`implement-by-phase` decomposition marker, creates or reuses child phase issues,
and then inspects phase 1:

- If phase 1 is `agent-pr`, it records a phase handoff marker on the parent and
  runs the external coder against the child issue, using that phase's
  `parent_context` as the approved implementation plan. It does not write the
  one-shot implementation marker used by `run-implement`.
- If phase 1 is `human-action` or `manual-close`, it prints JSON identifying the
  child issue and stops without posting a phase handoff or running the coder.
- If a matching phase handoff marker already exists, it prints a resume hint for
  the child issue and does not invoke the coder again.

`--dry-run` parses the decomposition and runs the implementation dry-run stub,
but it does not create child issues, post markers, push branches, or open a real
PR. Dry-run child issue numbers may be absent, so the JSON is a preview of the
phase that would be implemented.

### PR-loop (for PR N)

1. Run one round. The PR diff is fetched automatically — there is no plan-file
   step:
   ```bash
   python -m helpers.skill_runner run-pr-round \
     --pr PR_NUMBER --repo OWNER/REPO --reviewers codex gemini \
     [--head-sha SHA] [--workdir-codex /path/to/checkout] [--workdir-gemini /path/to/checkout] \
     [--test-command "pytest -q"] [--test-workdir .] \
     [--approved-followups summarize|issue]
   ```
   The result shape matches plan rounds, plus the optional `tests` and
   `approved_followups` fields (see **Gates & guardrails**).
2. Decide:
   - `"blocking"` with a host-owned PR → fix the code, push, post a
     change-summary comment (template below), and re-run. A new head SHA starts
     a new round automatically.
   - `"blocking"` with an external-coder PR → have that coder address the
     settled blocking review and push to the same PR branch:
     ```bash
     python -m helpers.skill_runner run-pr-fix \
       --pr PR_NUMBER --repo OWNER/REPO \
       --coder codex \
       --reviewers claude gemini \
       --workdir /path/to/push-capable/pr-clone
     ```
     The PR must be open, `--reviewers` must exactly match the reviewer set used
     by the previous `run-pr-round`, and `--workdir` must be a clean,
     push-capable clone where the PR head branch can be checked out by name.
     Re-run `run-pr-round` after the coder follow-up posts.
   - `"approved"` → check the test gate, then **stop at "ready to merge — human
     decision."** The skill never merges.

Change-summary template:
```bash
gh pr comment PR --repo OWNER/REPO --body "$(cat <<'EOF'
Addressed round-N feedback:

- **item-X**: <what was changed>
- **item-Y**: <what was changed>

-- <Your Name>
EOF
)"
```

---

## Modes

### PR mode

Run the **PR-loop** on the given PR. Done when it approves (ready to merge —
human decision).

### Issue mode

1. **Plan-loop** on the issue until approved.
2. Then, based on the user's intent:
   - **"plan and implement"** → **Implement step** → **PR-loop** → ready to merge.
   - **"just plan"** (the default) → report the approved plan and stop.

This is the skill's equivalent of the CLI's `issue --plan-first
[--implement-after-approval]`. Because the host **is** the coder, "implement after
approval" is your stated intent, not a code flag.

### Task mode

For a free-form task with no issue yet:
```bash
python -m helpers.skill_runner run-task-round \
  --task "Add a --verbose flag to the CLI" \
  --repo OWNER/REPO \
  --plan-file /tmp/agent-loop-skill/{session-id}/plan-r{N}.md \
  --reviewers codex gemini
```
This creates (or idempotently reuses) a scratch issue from the task text, then
runs the first plan round on it. Use `--task-file PATH` (or `--task-file -` for
stdin) for longer descriptions; `--dry-run` previews the issue it would create
without creating it. From there, continue **exactly as issue mode** from the
Plan-loop onward (including the optional Implement step + PR-loop).

---

## Gates & guardrails

- **Test gate** (PR-loop): `--test-command` runs after the reviewer turns in
  `--test-workdir` (default: the current directory, where you as the host coder
  have the PR branch checked out). The outcome is reported under `tests`:
  ```json
  "tests": { "command": "pytest -q", "passed": true, "exit_code": 0, "output_tail": "..." }
  ```
  A setup failure (empty/bad command, missing executable, or missing
  `--test-workdir`) is reported as
  `{"passed": false, "exit_code": null, "error": ...}` rather than crashing the
  round. **"Ready to merge" = `state == "approved"` AND `tests.passed`.** Treat a
  failing or errored gate as a hard stop, even when reviewers approved.
- **Approved-followups** (PR-loop): `--approved-followups summarize|issue`
  publishes reviewers' future follow-ups when (and only when) a round is
  **approved**. `summarize` posts one PR comment of the reconciled follow-ups;
  `issue` files up to three follow-up issues; `ignore` (default) discards them.
  The mode is also threaded into the reviewer prompt, so reviewers only surface
  future follow-ups when a non-`ignore` mode is set. Publishing is idempotent
  (one publish per PR head SHA + mode) and reported under `approved_followups`:
  ```json
  "approved_followups": { "mode": "summarize", "published": true, "count": 2 }
  ```
- **Agent memory** (plan-loop and PR-loop): repo-scoped orientation context
  (repo summary, architecture map, module index, test profile, toolchain,
  changed-file summaries) is generated deterministically (git + static analysis,
  no LLM) and included in the reviewer prompts so Codex/Gemini get the same
  context the CLI gives them. On by default; `--no-agent-memory` disables it,
  `--refresh-agent-memory` forces regeneration. The cache lives under the skill
  session dir and is incremental. Generation is advisory — if it fails, the round
  continues without memory.
- **Usage/cost** (plan-loop and PR-loop): the round result includes a `usage`
  field with token totals for the external reviewers (Codex/Gemini), aggregated
  per agent. It is **external-agents-only** — the host's Claude coder/plan turns
  run in the interactive session and have no programmatic token count, so the
  numbers are *not* a session total (the field's `scope`/`note` say so). Usage is
  persisted in `AGENT_LOOP_META`, so resumed rounds aggregate the full
  external-agent cost.
  ```json
  "usage": { "scope": "external-agents-only", "note": "...host turns not counted...",
             "totals": { "call_count": 2, "total_tokens": 250, ... },
             "per_agent": { "codex": {...}, "gemini": {...} } }
  ```
- **Reviewer resilience** (plan-loop and PR-loop): if an external reviewer's CLI
  fails to produce a usable review (an agent/tooling failure — e.g. an empty or
  malformed-tool-call response — *not* a fixable malformed review), the skill does
  **not** abort the round. It marks that reviewer **unavailable**, continues with
  the remaining reviewers, and lists it under `unavailable_reviewers`; the reviewer
  is re-attempted on the next run (or drop it from `--reviewers` to proceed). A
  round with an unavailable reviewer is never reported `approved`: its `state` is
  `incomplete` (or `blocking`/`pending` if those apply first). A genuinely
  malformed-but-content-bearing review still uses the `retry-validate` repair path.
- **Round states.** `approved` (all configured reviewers signed off) · `blocking`
  (a reviewer reported must-fix items) · `pending` (a host `claude` review handoff
  is outstanding — complete it with `complete-host-review`) · `incomplete` (a
  configured reviewer was unavailable; rerun). Precedence when several apply:
  `pending` > `blocking` > `incomplete` > `approved`.
- **Merge is always a human decision.** The skill never runs CI-wait or
  auto-merge; every mode stops at "ready to merge."

---

## Resuming

Every phase is re-runnable; if a session ends mid-arc, just re-invoke:

- **Plan-loop / PR-loop**: re-run the same round command. `build-resume` reads the
  GitHub comment history and skips reviewer turns already completed this round.
- **Task mode**: re-running with the same task text reuses the same scratch issue
  (tracked in a local task index), then resumes its plan-loop.
- **Implement step**: before implementing, check whether a PR already exists for
  the issue and resume it instead of opening a duplicate.

---

## Reversed roles (external coder, #307)

`run-plan-round` can run with an **external coder** (Codex or Gemini writes the
plan) instead of the host. Pass `--coder codex|gemini` and **omit** `--plan-file`
— the skill generates the plan via `run_external --role coder`, validates it,
attaches `--role coder --agent Codex|Gemini`, and posts it, then runs the
configured reviewers:

```bash
python -m helpers.skill_runner run-plan-round \
  --issue N --repo OWNER/REPO \
  --coder codex \
  --reviewers gemini
```

The skill drives the rounds from the posted ledger (no `--plan-file` to
discriminate them): you just re-run the command until it returns
`{"state": "approved"}`. Each invocation:

- **no coder record yet** → runs the coder for round 1 (`plan_state`);
- **plan posted, reviewers pending** → runs the remaining reviewers;
- **round complete, blocking/same-plan** → runs the coder for round N+1, which
  emits a structured `plan_revision` (rendered to a public comment; the canonical
  plan is carried forward for reviewers);
- **round complete, all approved** → returns `approved`.

If a coder turn produces a malformed plan, the raw response is saved to a repair
dir (manifest `role: coder`); fix `raw.md` and recover with
`retry-validate --repair-dir <dir>` (no re-run of the agent), then re-run
`run-plan-round`.

### Host-as-reviewer (Claude reviews the plan or PR)

Put `claude` in `--reviewers` to have **you (the host) review** — for both
`run-plan-round` (review the posted plan) and `run-pr-round` (review the PR diff):

```bash
python -m helpers.skill_runner run-plan-round \
  --issue N --repo OWNER/REPO --coder codex --reviewers claude gemini

python -m helpers.skill_runner run-pr-round \
  --pr N --repo OWNER/REPO --reviewers claude codex
```

External reviewers always run first. The host can only review *after* reading the
posted plan/PR, so a configured `claude` reviewer becomes a **pending handoff**:
the round returns `{"state": "pending", "pending_reviewers": ["Claude"], ...}`
(never `approved`/`blocking` while it's outstanding) and prints a review-request
dir. Read the material there (`{dir}/plan.md` for a plan, `{dir}/pr-diff.diff` for
a PR), write your `plan_review`/`pr_review` JSON to `{dir}/host-review.md`, then:

```bash
python -m helpers.skill_runner complete-host-review --dir <dir>
```

That validates, renders, attaches `--agent Claude --role reviewer`, and posts your
review. Re-run the same round command to recompute the round (it advances to
`approved`, or — for a plan — revises via the coder when there are blocking /
same-plan findings).

### Reverse implementation (external coder opens the PR)

`run-implement` has an **external coder** implement an approved plan and open a PR,
which you then review with `run-pr-round` (`claude` reviewer optional):

```bash
python -m helpers.skill_runner run-implement \
  --issue N --repo OWNER/REPO \
  --coder codex \
  --plan-file <approved-plan.md> \
  --workdir <push-capable-clone> [--base main]
# → prints {"pr": N, ...}; then:
python -m helpers.skill_runner run-pr-round --pr N --repo OWNER/REPO --reviewers claude gemini
# If that review blocks, let the same external coder push fixes:
python -m helpers.skill_runner run-pr-fix \
  --pr N --repo OWNER/REPO --coder codex --reviewers claude gemini \
  --workdir <push-capable-clone>
# Then re-review the new head:
python -m helpers.skill_runner run-pr-round --pr N --repo OWNER/REPO --reviewers claude gemini
```

This is **side-effecting**: the coder commits, pushes a branch, and opens a real
PR, so it needs a **push-capable** `--workdir`. The response is validated like the
CLI's implement path — a real PR marker, the signed-human-requirements
acknowledgement, in-workdir test reports, an advanced checkout HEAD, and that the
PR is open and references the issue — before a `role: coder` PR comment is posted.
It is **idempotent per plan** (the one-shot handoff marker): re-running returns the
existing PR instead of opening another one. For PR-review fixes, use
`run-pr-fix`; it is gated on a complete, current-head blocking `run-pr-round`
and posts a `role: coder` follow-up for the new PR head after validating that
the PR head and assigned checkout HEAD advanced.
A response that fails validation is non-retryable — fix the cause and re-run.
`--dry-run` does no pushes/PRs.

### Decompose an approved plan into phase issues

`run-decompose` has an **external coder** turn an approved plan into structured
phase child issues:

```bash
python -m helpers.skill_runner run-decompose \
  --issue N --repo OWNER/REPO \
  --coder codex \
  --plan-file <approved-plan.md> \
  [--workdir <checkout>]
```

Live runs are **side-effecting**: they create real GitHub child issues and post a
parent summary marker. The command is idempotent by parent issue + approved-plan
hash + `decompose-only` mode; re-running the same plan returns the recorded child
issues instead of creating duplicates. `--dry-run` still exercises the
decomposition parser and prints the child-issue preview JSON, but it does not
create issues or post comments.

`run-task-round` stays host-coder only.

---

## Billing and terms note

This skill runs Claude turns inside your active interactive Claude Code session.
Whether that counts as interactive or programmatic usage depends on Anthropic's
current terms and product behavior at the time you run it.
Do not use this skill to proxy one user's session to other users, to build
unattended 24/7 automation, or in any way that violates Anthropic's usage policies.

---

## Session state location

Session state is stored in:

```
~/.local/state/coding-review-agent-loop/skill-sessions/{owner-repo}/{issue}.json
```

This location is outside git checkouts, so it never dirties any working tree.

---

## Limitations

- If Claude Code's session ends mid-loop, resume by running `skill_runner` again
  with the same plan file — it reads GitHub comment history automatically via
  `build-resume` and skips already-completed reviewer turns.
- Long-running Codex/Gemini subprocess progress is not streamed; check the log
  file in `/tmp/coding-review-agent-loop/skill-logs/` if a reviewer hangs.
- The structured protocol (AGENT_LOOP_META markers, structured JSON responses)
  must match the versions expected by the existing library in `src/`.

---

## Demo

Run a minimal dry-run demo (no live GitHub or agent calls):

```bash
python -m helpers.demo_loop --issue 123 --repo demo/repo
```

Expected output includes:
```
validation passed: plan_state
validation passed: plan_review
demo_loop: all steps completed successfully
```
