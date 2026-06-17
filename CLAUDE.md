# coding-review-agent-loop — Claude Code Instructions

## Skill mode

When a user asks you to run the agent loop for an issue or PR, load and follow
the instructions in `SKILL.md`.

**Slash command** (preferred): the user can type `/coding-review-agent-loop`
followed by arguments — Claude Code loads `.claude/commands/coding-review-agent-loop.md`
which prompts you to read `SKILL.md` and start orchestration.

**Natural language** also works:
- "Run the agent-loop skill for issue #123 in OWNER/REPO with gemini as reviewer"
- "Start agent-loop plan-first for issue #42, reviewers: codex and gemini"
- "Run agent-loop pr 99 in OWNER/REPO"
- "Have codex fix the blocking review on PR #99 in OWNER/REPO"
- "Resume the agent-loop skill for issue #123"

In every case: read `SKILL.md`, gather the required inputs (repo, issue/PR
number, reviewers, flow type), and follow the orchestration steps from the top.

## Development

The main package lives in `src/coding_review_agent_loop/`.  Tests are in
`tests/`.  Run the full test suite with:

```bash
.venv/bin/python -m pytest tests/ -q
```
