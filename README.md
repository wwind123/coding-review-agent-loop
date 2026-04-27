# coding-review-agent-loop

Local command-line orchestration for a coding PR review loop.

Run a local Claude/Codex PR review loop using your existing CLI subscriptions,
without paying separate model API costs.

The main advantage is cost and account reuse: the tool shells out to your
already-authenticated local CLIs (`claude`, `codex`, and `gh`) instead of
calling model APIs directly. If your local agent CLIs are backed by existing AI
subscriptions or authenticated developer accounts, the review loop can use
those existing entitlements rather than requiring separate Claude/OpenAI API
keys and per-token API billing.

## Who This Is For

This is for developers who already use Claude Code, OpenAI Codex CLI, and
GitHub, and want one local agent to implement or fix a PR while another local
agent reviews it before merge.

It is especially useful when you are already doing this manually by switching
between agent CLIs and copying review feedback back and forth.

## Why Not GitHub Actions?

GitHub Actions-based agent loops usually need model API keys, hosted workflow
permissions, and separate API billing. This tool keeps the loop on your local
machine and uses the CLI accounts you have already authenticated.

That makes it easier to experiment with agent-to-agent review loops before
committing to hosted automation. It also keeps local workspace setup,
credentials, and agent approval prompts under your direct control.

## Agent Backends

Currently supported local agent CLIs:

- Claude Code via `claude`
- OpenAI Codex CLI via `codex`

Gemini CLI support is planned. The architecture is intended to support
additional local agent CLIs over time while keeping the same local,
subscription-friendly workflow.

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
requires local `gh`, `claude`, and/or `codex` authentication depending on which
agents you use.

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
agent-loop issue 123 --repo OWNER/REPO --claude-dir /path/to/claude/repo --codex-dir /path/to/codex/repo
```

Provide a one-off task directly when there is no issue yet:

```bash
agent-loop task "Add a health check endpoint" --repo OWNER/REPO --claude-dir /path/to/claude/repo --codex-dir /path/to/codex/repo
```

Run the loop against an existing pull request when you want another review and
iteration pass:

```bash
agent-loop pr 456 --repo OWNER/REPO --claude-dir /path/to/claude/repo --codex-dir /path/to/codex/repo
```

By default Claude is the coder and Codex is the reviewer. Reverse that with:

```bash
agent-loop task "Fix the flaky test" --repo OWNER/REPO --coder codex --reviewer claude
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
implement multiple-reviewer support and Claude to review the result. The work
became PR #2:
https://github.com/wwind123/coding-review-agent-loop/pull/2

```bash
~/tools/coding-review-agent-loop/.venv/bin/agent-loop task \
  "Currently the tool allows only 1 reviewer. Can we add the feature to allow multiple reviewers? By default, all reviewers have to approve for the whole thing to be considered approved." \
  --repo wwind123/coding-review-agent-loop \
  --claude-dir ~/tools/coding-review-agent-loop/claude/repo \
  --codex-dir ~/tools/coding-review-agent-loop/codex/repo \
  --coder codex \
  --reviewer claude \
  --dangerous-agent-permissions
```

See [docs/local_agent_loop.md](docs/local_agent_loop.md) for full usage and safety notes.

## Test

```bash
python -m pytest
```

Tests use fake subprocess runners. They do not call real `claude`, `codex`, or `gh`.
