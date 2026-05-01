"""Prompt builders for coder and reviewer agent turns."""

from __future__ import annotations

from typing import Sequence

from .agents.base import AgentName
from .agents.registry import agent_display_name, agent_signature
from .config import AgentLoopConfig, reviewers
from .github import PullRequestMetadata
from .memory import AgentMemoryContext, format_agent_memory_context


def format_agent_list(agents: Sequence[AgentName]) -> str:
    names = [agent_display_name(agent) for agent in agents]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _memory_block(memory: AgentMemoryContext | None) -> str:
    text = format_agent_memory_context(memory)
    if not text:
        return ""
    return f"Agent memory context:\n{text}\n"


def build_issue_prompt(
    issue_number: int,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder)
    return f"""Fix GitHub issue #{issue_number} in {config.repo}.

Use this local checkout as your workspace. Create a branch, implement the fix,
run relevant tests, commit, push, and open a pull request against {config.base}.
{_memory_block(memory)}

Do not wait for {reviewer_name} yourself; this local orchestrator will run {reviewer_name} after
you create the PR. In your final response, include the PR number using exactly
this marker:

<!-- AGENT_PR: <number> -->

Also include exactly one state marker:

<!-- AGENT_STATE: blocking -->

Use blocking here to hand the PR to {reviewer_name} for review. Sign the response as:
-- {coder_signature}
"""


def build_task_prompt(
    task_text: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder)
    return f"""You have been given a free-form task to implement in {config.repo}.

Task:
{task_text}
{_memory_block(memory)}

Use this local checkout as your workspace. Decide between two paths:

(a) If the task is clear enough to implement, create a branch, implement the
    change, run relevant tests, commit, push, and open a pull request against
    {config.base}. Do not wait for {reviewer_name}; this local orchestrator
    will run {reviewer_name} after you create the PR. End your final response
    with both markers:

    <!-- AGENT_PR: <number> -->
    <!-- AGENT_STATE: blocking -->

(b) If the task is genuinely ambiguous or missing information that would change
    the implementation, do NOT write code. Instead, ask focused clarifying
    questions and end your final response with exactly this marker:

    <!-- AGENT_CLARIFY -->

Prefer (a) when reasonable assumptions can be documented in the PR description;
choose (b) only for material ambiguity. Sign your response as:
-- {coder_signature}
"""


def build_task_clarification_prompt(
    task_text: str,
    history: Sequence[tuple[str, str]],
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    coder_signature = agent_signature(config.coder)
    qa_blocks = "\n\n".join(
        f"Round {idx + 1} questions from you:\n{questions}\n\n"
        f"Round {idx + 1} answers from the user:\n{answers}"
        for idx, (questions, answers) in enumerate(history)
    )
    return f"""Continuing the previous free-form task in {config.repo}.

Original task:
{task_text}
{_memory_block(memory)}

Clarification so far:

{qa_blocks}

Now proceed. Strongly prefer to implement the task and open a PR. Only ask
again if a critical detail is still missing. Use the same response markers as
before:

- For implementation: include both <!-- AGENT_PR: <number> --> and
  <!-- AGENT_STATE: blocking --> at the end of your final response.
- For another clarification round: end your final response with exactly
  <!-- AGENT_CLARIFY -->.

Sign your response as:
-- {coder_signature}
"""


def build_review_prompt(
    pr_number: int,
    round_number: int,
    config: AgentLoopConfig,
    *,
    reviewer: AgentName,
    pr_metadata: PullRequestMetadata | None = None,
    memory: AgentMemoryContext | None = None,
) -> str:
    coder_name = agent_display_name(config.coder)
    reviewer_signature = agent_signature(reviewer)
    reviewer_group = format_agent_list(reviewers(config))
    metadata = pr_metadata or PullRequestMetadata(
        number=pr_number,
        repo=config.repo,
        title=None,
        head_branch=None,
        base_branch=None,
        head_sha=None,
        url=None,
    )
    title = metadata.title or "(unknown)"
    head_branch = metadata.head_branch or "(unknown)"
    base_branch = metadata.base_branch or "(unknown)"
    head_sha = metadata.head_sha or "(unknown)"
    url_line = f"- URL: {metadata.url}\n" if metadata.url else ""
    if config.approved_followups == "ignore":
        followup_guidance = """Do not include Same-PR follow-ups, Future follow-ups, or legacy
Non-blocking follow-ups sections in approved reviews; this run is configured to
ignore approved-review follow-up sections. Mark the review blocking instead
when cleanup should be fixed before merge.
"""
    elif config.approved_followups.startswith("fix-and-"):
        followup_guidance = f"""If you approve but notice small, low-risk cleanup worth fixing before merge,
list those items under this exact heading:

### Same-PR follow-ups

If you approve but notice substantial work that is better handled separately in
a future issue or PR, list at most three highest-value items under this exact
heading:

### Future follow-ups

Same-PR follow-ups will be sent back to {coder_name} and require another review
round before final approval. Do not put trivial style nits in either follow-up
section.
"""
    else:
        followup_guidance = """If you approve but notice substantial work that is better handled separately in
a future issue or PR, list at most three highest-value items under this exact
heading:

### Future follow-ups

Do not use the Same-PR follow-ups section in this mode; mark the review blocking
instead when small or local cleanup should be fixed before merge.
The legacy heading `### Non-blocking follow-ups` is still accepted as future
follow-ups for compatibility, but prefer `### Future follow-ups`.
"""
    return f"""Review pull request #{pr_number} in {config.repo} (round {round_number}).

PR metadata:
- Repo: {metadata.repo}
- PR: #{metadata.number}
- Title: {title}
- Head branch: {head_branch}
- Base branch: {base_branch}
- Head SHA: {head_sha}
{url_line}
Use this PR metadata as authoritative. Do not spend time discovering the PR
branch.
{_memory_block(memory)}

Suggested commands:
- {config.gh_cmd} pr view {metadata.number} --repo {metadata.repo} --json title,body,headRefName,baseRefName,headRefOid,comments,reviews
- {config.gh_cmd} pr diff {metadata.number} --repo {metadata.repo}

If a shell/tool command requires confirmation in non-interactive mode, do not
retry repeatedly. Use the PR metadata above and the suggested GitHub CLI
commands, or produce a blocking review explaining the limitation.

Focus on correctness, security, test coverage, and maintainability. Review the
full diff and any existing PR discussion. Do not make code changes in this
review step; report blocking findings if {coder_name} needs to fix anything.
{followup_guidance}
Use blocking only for issues that should prevent merge.
All configured reviewers ({reviewer_group}) must approve in the same round for
the pull request to be considered approved.

End your final response with exactly one marker:

<!-- AGENT_STATE: approved -->

or:

<!-- AGENT_STATE: blocking -->

Use approved only if there are no blocking issues. Always sign your response:
-- {reviewer_signature}
"""


def build_followup_prompt(
    pr_number: int,
    round_number: int,
    review: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder)
    return f"""{reviewer_name} reviewed pull request #{pr_number} in {config.repo} and found blocking issues.

Address the review below in this local checkout. Pull/sync the PR branch if
needed, implement fixes, run relevant tests, commit, and push to the same PR.
Do not create a new PR.
{_memory_block(memory)}

{reviewer_name} review:

{review}

This is round {round_number}. End your final response with exactly one marker:

<!-- AGENT_STATE: blocking -->

Use blocking to hand the updated PR back to {reviewer_name}. If you cannot safely address
the review, explain why and still use the blocking marker so a human can
intervene. Sign the response as:
-- {coder_signature}
"""


def build_same_pr_followup_prompt(
    pr_number: int,
    round_number: int,
    review: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder)
    return f"""{reviewer_name} approved pull request #{pr_number} in {config.repo} with same-PR follow-ups.

Address the follow-up items below in this local checkout. Pull/sync the PR
branch if needed, implement fixes, run relevant tests, commit, and push to the
same PR. Do not create a new PR.
{_memory_block(memory)}

Same-PR follow-ups:

{review}

This is round {round_number}. End your final response with exactly one marker:

<!-- AGENT_STATE: blocking -->

Use blocking to hand the updated PR back to {reviewer_name}. If you cannot safely address
the follow-ups, explain why and still use the blocking marker so a human can
intervene. Sign the response as:
-- {coder_signature}
"""
