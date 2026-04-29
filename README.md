# coding-review-agent-loop

Local command-line orchestration for a coding PR review loop.

Run a local Claude/Codex/Gemini PR review loop using your existing CLI subscriptions,
without paying separate model API costs.

The main advantage is cost and account reuse: the tool shells out to your
already-authenticated local CLIs (`claude`, `codex`, `gemini`, and `gh`) instead
of calling model APIs directly. If your local agent CLIs are backed by existing
AI subscriptions or authenticated developer accounts, the review loop can use
those existing entitlements rather than requiring separate model API
keys and per-token API billing.

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

Start from a GitHub issue when you want the agent loop to use the issue title
and body as the implementation task:

```bash
agent-loop issue 123 --repo OWNER/REPO
```

Provide a one-off task directly when there is no issue yet:

```bash
agent-loop task "Add a health check endpoint" --repo OWNER/REPO
```

Run the loop against an existing pull request when you want another review and
iteration pass:

```bash
agent-loop pr 456 --repo OWNER/REPO
```

If `--repo` is omitted, the tool runs `gh repo view` from the current working
directory, or from `--codex-dir` when that flag is provided, and uses the
detected `OWNER/REPO`. Pass `--repo` explicitly when running outside the target
repository.

When `--claude-dir`, `--codex-dir`, or `--gemini-dir` is omitted for an active
agent, the tool creates or reuses a repo-scoped temporary checkout such as
`/tmp/coding-review-agent-loop/OWNER-REPO/codex/repo`. Existing clean temp
checkouts are fetched and fast-forwarded on the base branch before the agent
runs; dirty temp checkouts fail clearly instead of being overwritten. Use
explicit persistent directories for large repositories, long-lived agent
worktrees, or setups that should survive `/tmp` cleanup or reboot.

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
  "Please go over all the issue and pr reviews again and see if there's any non-blocking issues still worth addressing but have not been addressed." \
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
