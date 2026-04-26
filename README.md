# coding-review-agent-loop

Local command-line orchestration for a two-agent coding PR review loop.

The tool shells out to your already-authenticated local CLIs (`claude`, `codex`, and `gh`) so it can run from your workstation without GitHub Actions model API keys.

## Install for Development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Quick Start

```bash
agent-loop issue 123 --repo OWNER/REPO --claude-dir /path/to/claude/repo --codex-dir /path/to/codex/repo
agent-loop task "Add a health check endpoint" --repo OWNER/REPO --claude-dir /path/to/claude/repo --codex-dir /path/to/codex/repo
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
