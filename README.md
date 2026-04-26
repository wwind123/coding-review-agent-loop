# coding-review-agent-loop

Local command-line orchestration for a two-agent coding PR review loop.

The tool shells out to your already-authenticated local CLIs (`claude`, `codex`, and `gh`) so it can run from your workstation without GitHub Actions model API keys.

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

For trusted local automation that must run without approval prompts:

```bash
agent-loop issue 123 --repo OWNER/REPO --dangerous-agent-permissions
```

See [docs/local_agent_loop.md](docs/local_agent_loop.md) for full usage and safety notes.

## Test

```bash
python -m pytest
```

Tests use fake subprocess runners. They do not call real `claude`, `codex`, or `gh`.
