"""Close a child issue after its PR merged into a non-default base (#1285).

GitHub closes a closing-referenced issue only when the PR merges into the
repository default branch.  A managed PR merged into a trusted integration
branch therefore leaves its child issue open, which staged-phase tracking
would report as an inconsistent phase.  This module closes the child after a
confirmed merge, and retries that closure without replaying the merge when a
later run finds the PR already MERGED.
"""

from __future__ import annotations

import json
from typing import Literal

from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import IssueContext, parse_strong_issue_reference_evidence
from .logging import log
from .runner import Runner
from .workdirs import github_api_cwd

CloseOutcome = Literal["skipped", "closed", "already-closed", "unresolved"]


def _api(runner: Runner, config: AgentLoopConfig, endpoint: str) -> dict[str, object] | None:
    result = runner.run([config.gh_cmd, "api", endpoint], cwd=github_api_cwd(), check=False)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def close_child_after_integration_merge(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext | None,
    pr_number: int,
) -> CloseOutcome:
    """Close the closing-referenced child of a MERGED non-default-base PR.

    ``skipped``: nothing to do (default-branch merge, no validated closing
    reference, or the PR is not confirmed merged: never close before a
    confirmed merge).  ``unresolved``: the merge is authenticated but the
    child could not be closed; the caller must not report the stage complete.
    """
    if issue_context is None or config.dry_run:
        return "skipped"
    pr = _api(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    if pr is None:
        return "skipped"
    base = pr.get("base") if isinstance(pr.get("base"), dict) else {}
    base_repo = base.get("repo") if isinstance(base.get("repo"), dict) else {}
    base_ref = base.get("ref")
    if (
        pr.get("merged") is not True
        or not isinstance(pr.get("merged_at"), str)
        or not isinstance(base_ref, str)
        or not base_ref
        or str(base_repo.get("full_name", "")).casefold() != config.repo.casefold()
    ):
        return "skipped"
    repo = _api(runner, config, f"repos/{config.repo}")
    default_branch = repo.get("default_branch") if repo else None
    if not isinstance(default_branch, str) or base_ref == default_branch:
        return "skipped"
    body = pr.get("body") if isinstance(pr.get("body"), str) else ""
    if not parse_strong_issue_reference_evidence(
        body, repo=config.repo, issue_number=issue_context.number
    ):
        return "skipped"
    issue = _api(runner, config, f"repos/{config.repo}/issues/{issue_context.number}")
    state = str(issue.get("state", "")).lower() if issue else ""
    if state == "closed":
        return "already-closed"
    if state != "open":
        return "unresolved"
    comment = (
        f"Closing: PR #{pr_number} merged into `{base_ref}`. GitHub only auto-closes a "
        f"closing-referenced issue on the default branch (`{default_branch}`), so "
        "agent-loop closes it after the confirmed merge.\n\n-- agent-loop"
    )
    result = runner.run(
        [
            config.gh_cmd, "issue", "close", str(issue_context.number),
            "--repo", config.repo, "--reason", "completed", "--comment", comment,
        ],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        log(config, f"Issue #{issue_context.number}: closing after merge into {base_ref} failed")
        return "unresolved"
    log(config, f"Issue #{issue_context.number}: closed after PR #{pr_number} merged into {base_ref}")
    return "closed"


def require_child_closed_or_report(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext | None,
    pr_number: int,
) -> bool:
    """Close the child and report unresolved closure; ``False`` means incomplete."""
    outcome = close_child_after_integration_merge(
        runner, config=config, issue_context=issue_context, pr_number=pr_number
    )
    if outcome == "unresolved":
        number = issue_context.number if issue_context is not None else "?"
        print(
            f"PR #{pr_number} merged, but child issue #{number} could not be closed. "
            "Rerun agent-loop on the issue (or its staged parent) to retry the closure "
            "without replaying the merge; the stage is not complete until it is closed."
        )
        return False
    return True
