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
Start agent-loop plan-first for issue #42 in myorg/myrepo, reviewers: codex and gemini.
```

```
Run agent-loop on PR #99 in myorg/myrepo.
```

```
Resume the agent-loop skill for issue #123 in myorg/myrepo.
```

Claude will read this file and follow the orchestration steps below.  You do
not need to type any slash command; natural-language requests are enough.

---

## How to start a plan loop for an issue

Provide the following information:

1. **Repository**: `OWNER/REPO`
2. **Issue number**: e.g. `123`
3. **Reviewers**: e.g. `codex`, `gemini`, or both

Then follow the steps below.

---

## Orchestration steps

### Step 1 — Write the plan (Claude host turn)

Write the implementation plan to a temp file, e.g.:

```
/tmp/agent-loop-skill/{session-id}/plan-r{N}.md
```

The file must end with:

```
<!-- AGENT_PLAN_STATE: approved -->
-- Anthropic Claude
```

### Step 2 — Run one review round

`skill_runner` handles everything else: session resume, plan validation, attaching
`AGENT_LOOP_META`, posting to GitHub, running each reviewer, validating/rendering
their responses, and writing session state.

```bash
python -m helpers.skill_runner run-plan-round \
  --issue ISSUE \
  --repo OWNER/REPO \
  --plan-file /tmp/agent-loop-skill/{session-id}/plan-r{N}.md \
  --reviewers codex gemini \
  [--workdir-codex /path/to/codex/checkout] \
  [--workdir-gemini /path/to/gemini/checkout]
```

It prints a JSON result to stdout:

```json
{
  "state": "approved" | "blocking",
  "round_number": N,
  "blocking_items": [...],
  "approved_reviewers": [...]
}
```

### Step 3 — Decision

- `"state": "approved"` and `blocking_items` is empty → implementation is complete.
- `"state": "blocking"` → address the `blocking_items`, write a revised plan, and
  loop back to Step 1 with the new plan file (the round number increments automatically).
- If clarification is needed: post an `<!-- AGENT_CLARIFY -->` comment and stop.

---

## PR review mode

Use `run-pr-round` instead of `run-plan-round`.  Pass `--pr PR_NUMBER` and
optionally `--head-sha SHA` (auto-fetched if omitted):

```bash
python -m helpers.skill_runner run-pr-round \
  --pr PR_NUMBER \
  --repo OWNER/REPO \
  --reviewers codex gemini \
  [--head-sha SHA] \
  [--workdir-codex /path/to/checkout] \
  [--workdir-gemini /path/to/checkout]
```

The JSON result shape is the same as for plan rounds.  There is no "write plan"
step — the PR diff is fetched automatically.

---

## Reversed roles (Codex as coder, Claude as reviewer)

Reversed-roles mode (Codex writes the plan, Claude reviews) is tracked in
issue #285 and not yet implemented in `skill_runner`.  Until then, use the
low-level helpers directly:

1. Invoke Codex to write the plan via `helpers.run_external --role coder`.
2. Validate the plan with `helpers.validate_response --kind plan_state`.
3. Attach metadata with `helpers.state_manager attach-metadata --agent Codex`.
4. Post the plan via `helpers.gh_ops post-issue-comment`.
5. Claude writes a `plan_review` JSON review, validates it, renders it, and posts it
   using `--agent Claude`.  Pass `--reviewers claude` to `build-resume` so that
   Claude's review is recognized on resume.

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
