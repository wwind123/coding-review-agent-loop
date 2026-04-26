# Local Coding Review Agent Loop

`coding-review-agent-loop` is a local CLI that orchestrates two coding agents through a GitHub pull request review loop. It shells out to locally authenticated `claude`, `codex`, and `gh` CLIs instead of calling model APIs directly.

The default flow is:

1. A coder agent creates or updates a PR.
2. A reviewer agent reviews the PR.
3. If the reviewer finds blockers, the coder fixes the PR.
4. The loop repeats until the reviewer approves or `--max-rounds` is reached.

The default coder is Claude and the default reviewer is Codex. Reverse the direction with `--coder codex --reviewer claude`.

## Prerequisites

- `gh` is installed and authenticated for the target GitHub repository.
- `claude` is installed and authenticated if either side uses Claude.
- `codex` is installed and authenticated if either side uses Codex.
- Use separate clones or worktrees for Claude and Codex to avoid local file conflicts. Missing `--claude-dir` / `--codex-dir` directories are created automatically; paths that already exist as files fail clearly.

## Usage

Fix a GitHub issue:

```bash
agent-loop issue 56 \
  --repo OWNER/REPO \
  --claude-dir /path/to/claude/worktree \
  --codex-dir /path/to/codex/worktree
```

Implement a free-form task:

```bash
agent-loop task "Add a /healthz endpoint that returns 200 OK." \
  --repo OWNER/REPO \
  --claude-dir /path/to/claude/worktree \
  --codex-dir /path/to/codex/worktree
```

Review an existing PR:

```bash
agent-loop pr 123 \
  --repo OWNER/REPO \
  --claude-dir /path/to/claude/worktree \
  --codex-dir /path/to/codex/worktree
```

Reverse the direction so Codex creates/fixes and Claude reviews:

```bash
agent-loop task "Refactor the cache layer" \
  --repo OWNER/REPO \
  --coder codex \
  --reviewer claude \
  --claude-dir /path/to/claude/worktree \
  --codex-dir /path/to/codex/worktree
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
  --auto-merge \
  --ci-check-name test
```

When enabled, the tool waits for the configured GitHub check-run to pass before merging. Local `--test-command` is an additional local gate, not a replacement for CI.

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

You can also provide exact per-agent replacements. Repeat once per token:

```bash
agent-loop issue 56 \
  --repo OWNER/REPO \
  --claude-arg=--permission-mode --claude-arg=acceptEdits \
  --codex-arg=--sandbox --codex-arg=workspace-write --codex-arg=--ask-for-approval --codex-arg=never
```

Providing any `--claude-arg` or `--codex-arg` replaces that agent's default entirely.

## Protocol

Agent responses are parsed using HTML comment markers:

```text
<!-- AGENT_PR: 123 -->
<!-- AGENT_STATE: approved -->
<!-- AGENT_STATE: blocking -->
<!-- AGENT_CLARIFY -->
```

`AGENT_PR` is required after a coder creates a PR. Review/fix responses must include exactly one `AGENT_STATE` marker. If a response quotes older markers, the final marker is treated as authoritative.

## Logs

Agent stdout/stderr is written to `.agent-loop-logs/` under the Codex checkout by default. The CLI prints heartbeat messages with the log path while agents run:

```text
[agent-loop 12:00:31] Claude still running (30s); log: /path/to/.agent-loop-logs/20260425-120001-claude.log
```

Use `tail -f` on the displayed path to see live output. The log directory gets its own `.gitignore` on first use.
