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
from coding_review_agent_loop.prompts import (
    build_issue_implementation_prompt,
    build_issue_plan_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
    build_review_prompt,
)
from coding_review_agent_loop.round_state import _deserialize_unresolved_item


def make_minimal_config(
    repo: str,
    coder: AgentName,
    reviewer_names: Sequence[AgentName],
    *,
    reviewer: AgentName | None = None,
    workdir: str | None = None,
    base: str = "main",
    approved_followups: str = "ignore",
    agent_memory: bool = False,
    agent_memory_dir: Path | None = None,
    refresh_agent_memory: bool = False,
) -> AgentLoopConfig:
    tmp_base = Path(tempfile.gettempdir()) / "coding-review-agent-loop"
    legacy = tmp_base / "skill-runner"

    def _agent_dir(agent: AgentName) -> Path:
        # The active reviewer's dir must match where the agent actually runs, so the
        # checkout path embedded in the prompt is real. The default mirrors
        # skill_runner._workdir_for_agent ("skill-runner-{agent}"); previously this
        # hardcoded the bare "skill-runner" path, which never exists and made
        # reviewers (notably Codex) block on a missing checkout (#297).
        if workdir and agent == reviewer:
            return Path(workdir)
        return tmp_base / f"skill-runner-{agent}"

    return AgentLoopConfig(
        repo=repo,
        claude_dir=_agent_dir("claude"),
        codex_dir=_agent_dir("codex"),
        gemini_dir=_agent_dir("gemini"),
        coder=coder,
        reviewer=tuple(reviewer_names),
        base=base,
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


def build_plan_prompt_for_skill(
    issue_dict: dict,
    *,
    repo: str,
    coder: AgentName,
    reviewers: Sequence[AgentName],
    workdir: str,
    memory: AgentMemoryContext | None = None,
) -> str:
    """Build the round-1 coder (plan) prompt for an external coder (#307).

    Sets the coder's checkout dir to ``workdir`` so the embedded checkout guidance
    points at the real run directory.
    """
    config = make_minimal_config(
        repo, coder, tuple(reviewers), reviewer=coder, workdir=workdir,
    )
    issue_context = _make_issue_context(issue_dict)
    return build_issue_plan_prompt(issue_context.number, config, memory, issue_context)


def build_plan_revision_prompt_for_skill(
    issue_dict: dict,
    *,
    repo: str,
    coder: AgentName,
    reviewers: Sequence[AgentName],
    workdir: str,
    round_number: int,
    previous_plan: str,
    reviewer_feedback: str,
    prior_items_raw: list[dict],
    memory: AgentMemoryContext | None = None,
) -> str:
    """Build the round-N+1 coder (plan revision) prompt for an external coder (#307).

    ``reviewer_feedback`` is the aggregated prose feedback (each reviewer's summary
    + item texts) the caller assembles from the resume's completed reviewers.
    """
    config = make_minimal_config(
        repo, coder, tuple(reviewers), reviewer=coder, workdir=workdir,
    )
    issue_context = _make_issue_context(issue_dict)
    unresolved = [_deserialize_unresolved_item(item) for item in prior_items_raw]
    return build_plan_revision_prompt(
        issue_context.number,
        round_number,
        previous_plan,
        reviewer_feedback,
        config,
        memory,
        issue_context,
        unresolved_items=unresolved,
    )


def build_implementation_prompt_for_skill(
    issue_context: IssueContext,
    approved_plan: str,
    *,
    repo: str,
    coder: AgentName,
    workdir: str,
    base: str = "main",
    memory: AgentMemoryContext | None = None,
) -> str:
    """Build the external-coder implementation prompt (reversed roles, #316).

    Takes a full ``IssueContext`` (built by the caller via ``get_issue_context``)
    rather than a bare dict, so issue comments + signed human requirements reach the
    coder. Sets the coder's checkout dir to ``workdir`` and the config ``base`` so
    the embedded checkout guidance and "open a PR against {base}" instruction match
    the real run dir + requested base.
    """
    config = make_minimal_config(
        repo, coder, (coder,), reviewer=coder, workdir=workdir, base=base,
    )
    return build_issue_implementation_prompt(
        issue_context.number, approved_plan, config, memory, issue_context=issue_context,
    )
