# Local Coding Review Agent Loop

`coding-review-agent-loop` is a local CLI that orchestrates coding agents through a GitHub pull request review loop. Its main advantage is cost and account reuse: it shells out to locally authenticated `claude`, `codex`, and `gh` CLIs instead of calling model APIs directly. If your local agent CLIs are backed by existing AI subscriptions or authenticated developer accounts, the review loop can use those existing entitlements rather than requiring separate Claude/OpenAI API keys and per-token API billing.

The default flow is:

1. A coder agent creates or updates a PR.
2. One or more reviewer agents review the PR.
3. If any reviewer finds blockers, the coder fixes the PR.
4. The loop repeats until every reviewer approves in the same round or `--max-rounds` is reached.

The default coder is Claude and the default reviewer is Codex. Reverse the direction with `--coder codex --reviewer claude`. Repeat `--reviewer` to require multiple reviewer approvals.

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

Require both reviewers to approve. The coder may also be listed as a reviewer
when you want the same agent to work in separate coding and review passes:

```bash
agent-loop pr 123 \
  --repo OWNER/REPO \
  --reviewer codex \
  --reviewer claude \
  --claude-dir /path/to/claude/worktree \
  --codex-dir /path/to/codex/worktree
```

## Real Example

This project uses `agent-loop` to improve itself. The following command asked
Codex to implement multiple-reviewer support and Claude to review the result.
That work became PR #2:
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
