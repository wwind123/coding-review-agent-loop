"""
Thin wrappers around the existing prompt-building library for use in skill_runner.

These functions accept plain dicts and primitive types so callers need not
instantiate internal dataclasses directly.
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.agents.base import AgentName
from coding_review_agent_loop.config import AgentLoopConfig
from coding_review_agent_loop.github import IssueContext, IssueComment
from coding_review_agent_loop.memory import AgentMemoryContext
from coding_review_agent_loop.prompts import build_plan_review_prompt, build_review_prompt
from coding_review_agent_loop.round_state import _deserialize_unresolved_item


def make_minimal_config(
    repo: str,
    coder: AgentName,
    reviewer_names: Sequence[AgentName],
    *,
    reviewer: AgentName | None = None,
    workdir: str | None = None,
    approved_followups: str = "ignore",
    agent_memory: bool = False,
    agent_memory_dir: Path | None = None,
    refresh_agent_memory: bool = False,
) -> AgentLoopConfig:
    base = Path(tempfile.gettempdir()) / "coding-review-agent-loop"
    legacy = base / "skill-runner"

    def _agent_dir(agent: AgentName) -> Path:
        # The active reviewer's dir must match where the agent actually runs, so the
        # checkout path embedded in the prompt is real. The default mirrors
        # skill_runner._workdir_for_agent ("skill-runner-{agent}"); previously this
        # hardcoded the bare "skill-runner" path, which never exists and made
        # reviewers (notably Codex) block on a missing checkout (#297).
        if workdir and agent == reviewer:
            return Path(workdir)
        return base / f"skill-runner-{agent}"

    return AgentLoopConfig(
        repo=repo,
        claude_dir=_agent_dir("claude"),
        codex_dir=_agent_dir("codex"),
        gemini_dir=_agent_dir("gemini"),
        coder=coder,
        reviewer=tuple(reviewer_names),
        base="main",
        max_rounds=1,
        auto_merge=False,
        dry_run=False,
        allow_shared_dir=True,
        claude_cmd="claude",
        codex_cmd="codex",
        gemini_cmd="gemini",
        gh_cmd="gh",
        claude_args=(),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
        gemini_args=("--skip-trust",),
        test_command=None,
        pre_review_tests=False,
        ci_check_name="",
        ci_timeout_seconds=300,
        ci_poll_interval_seconds=30,
        quiet=False,
        log_dir=legacy,
        progress_interval_seconds=30,
        agent_max_retries=0,
        agent_retry_backoff_seconds=(30,),
        agent_memory=agent_memory,
        refresh_agent_memory=refresh_agent_memory,
        agent_memory_dir=agent_memory_dir if agent_memory_dir is not None else legacy,
        refresh_test_profile=False,
        auto_agent_dirs=tuple(reviewer_names),
        approved_followups=approved_followups,
    )


def _make_issue_context(issue_dict: dict) -> IssueContext:
    return IssueContext(
        number=int(issue_dict.get("number", 0)),
        repo=str(issue_dict.get("repo", "")),
        title=issue_dict.get("title"),
        body=issue_dict.get("body"),
        url=issue_dict.get("url"),
        comments=(),
    )


def build_plan_review_prompt_for_skill(
    issue_dict: dict,
    plan_text: str,
    prior_items_raw: list[dict],
    round_number: int,
    reviewer: AgentName,
    *,
    repo: str,
    coder: AgentName = "claude",
    all_reviewers: Sequence[AgentName] | None = None,
    workdir: str | None = None,
    memory: AgentMemoryContext | None = None,
) -> str:
    """Build a plan reviewer prompt from plain dicts.

    Args:
        issue_dict: Plain dict from gh issue view --json.
        plan_text: The current plan text.
        prior_items_raw: List of serialized UnresolvedReviewItem dicts.
        round_number: Current review round number.
        reviewer: Agent name of the reviewer.
        repo: Repository owner/name.
        coder: Agent name of the coder (default: "claude").
        all_reviewers: All configured reviewer names (defaults to [reviewer]).
        workdir: The reviewer's actual checkout path, embedded in the prompt so
            the agent inspects a directory that exists (#297).
        memory: Repo-scoped agent memory to include for reviewer orientation (#306).
    """
    reviewers_list: Sequence[AgentName] = all_reviewers or [reviewer]
    config = make_minimal_config(repo, coder, reviewers_list, reviewer=reviewer, workdir=workdir)
    issue_context = _make_issue_context(issue_dict)
    unresolved = [_deserialize_unresolved_item(item) for item in prior_items_raw]
    return build_plan_review_prompt(
        issue_context.number,
        round_number,
        plan_text,
        config,
        reviewer=reviewer,
        memory=memory,
        issue_context=issue_context,
        unresolved_items=unresolved,
    )


def build_review_prompt_for_skill(
    issue_dict: dict,
    pr_diff: str,
    prior_items_raw: list[dict],
    round_number: int,
    reviewer: AgentName,
    *,
    repo: str,
    pr_number: int,
    coder: AgentName = "claude",
    all_reviewers: Sequence[AgentName] | None = None,
    workdir: str | None = None,
    approved_followups: str = "ignore",
    memory: AgentMemoryContext | None = None,
) -> str:
    """Build a PR reviewer prompt from plain dicts.

    Args:
        issue_dict: Plain dict from gh issue/pr view --json.
        pr_diff: The PR diff text.
        prior_items_raw: List of serialized UnresolvedReviewItem dicts.
        round_number: Current review round number.
        reviewer: Agent name of the reviewer.
        repo: Repository owner/name.
        pr_number: Pull request number.
        coder: Agent name of the coder (default: "claude").
        all_reviewers: All configured reviewer names (defaults to [reviewer]).
        workdir: The reviewer's actual checkout path, embedded in the prompt so
            the agent inspects a directory that exists (#297).
        approved_followups: Approved-followups mode; when not "ignore" the prompt
            instructs the reviewer to surface future follow-ups so they can be
            published on approval (#300).
        memory: Repo-scoped agent memory to include for reviewer orientation (#306).
    """
    from coding_review_agent_loop.github import PullRequestMetadata

    reviewers_list: Sequence[AgentName] = all_reviewers or [reviewer]
    config = make_minimal_config(
        repo, coder, reviewers_list, reviewer=reviewer, workdir=workdir,
        approved_followups=approved_followups,
    )
    issue_context = _make_issue_context(issue_dict)
    unresolved = [_deserialize_unresolved_item(item) for item in prior_items_raw]
    pr_metadata = PullRequestMetadata(
        number=pr_number,
        repo=repo,
        title=issue_dict.get("title"),
        head_branch=None,
        base_branch=None,
        head_sha=None,
        url=issue_dict.get("url"),
    )
    prompt = build_review_prompt(
        pr_number,
        round_number,
        config,
        reviewer=reviewer,
        memory=memory,
        pr_metadata=pr_metadata,
        issue_context=issue_context,
        unresolved_items=unresolved,
    )
    if pr_diff:
        prompt += f"\n\n## PR diff\n\n```diff\n{pr_diff}\n```\n"
    return prompt
