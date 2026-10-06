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

CloseOutcome = Literal[
    "skipped",
    "default-branch",
    "not-merged",
    "closed",
    "already-closed",
    "unresolved",
]

# Outcomes after which the stage may be reported complete.  ``unresolved``
# covers every unknown or unauthenticated state (unreadable evidence, a base
# that is not the recorded/trusted one, no closing reference, a failed close),
# so uncertainty is never mistaken for a confirmed no-op.
COMPLETE_OUTCOMES = frozenset({"skipped", "default-branch", "closed", "already-closed"})


def _api(runner: Runner, config: AgentLoopConfig, endpoint: str) -> dict[str, object] | None:
    result = runner.run([config.gh_cmd, "api", endpoint], cwd=github_api_cwd(), check=False)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _recorded_base(config: AgentLoopConfig) -> str | None:
    """The base the operator explicitly recorded for this run, if any."""
    return config.base if config.base and config.base_provenance == "explicit" else None


def close_child_after_integration_merge(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext | None,
    pr_number: int,
    expected_base: str | None = None,
) -> CloseOutcome:
    """Close the closing-referenced child of a MERGED non-default-base PR.

    The merge is authenticated first: the PR must be MERGED into this
    repository, its base must equal the recorded base (``expected_base`` or the
    run's explicit ``--base``) and still be trusted by the repository
    variable, and its body must carry the closing reference to the child.
    Any mismatch or unreadable evidence is ``unresolved`` and mutates nothing.
    ``default-branch`` and ``not-merged`` are confirmed states, never inferred
    from a failed read.
    """
    if issue_context is None or config.dry_run:
        return "skipped"
    pr = _api(runner, config, f"repos/{config.repo}/pulls/{pr_number}")
    if pr is None:
        return "unresolved"
    if pr.get("merged") is False:
        return "not-merged"
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
        return "unresolved"
    repo = _api(runner, config, f"repos/{config.repo}")
    default_branch = repo.get("default_branch") if repo else None
    if not isinstance(default_branch, str) or not default_branch:
        return "unresolved"
    if base_ref == default_branch:
        return "default-branch"
    recorded = expected_base or _recorded_base(config)
    if recorded is not None and base_ref != recorded:
        log(config, f"Issue #{issue_context.number}: PR #{pr_number} merged into {base_ref}, not the recorded base {recorded}")
        return "unresolved"
    from .managed_ci import trusted_base_problem

    if trusted_base_problem(
        runner, gh_cmd=config.gh_cmd, repo=config.repo, cwd=github_api_cwd(),
        base=base_ref, default_branch=default_branch, require_marker=False,
    ) is not None:
        log(config, f"Issue #{issue_context.number}: PR #{pr_number} merged into untrusted base {base_ref}")
        return "unresolved"
    body = pr.get("body") if isinstance(pr.get("body"), str) else ""
    if not parse_strong_issue_reference_evidence(
        body, repo=config.repo, issue_number=issue_context.number
    ):
        return "unresolved"
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
    if outcome not in COMPLETE_OUTCOMES:
        number = issue_context.number if issue_context is not None else "?"
        print(
            f"PR #{pr_number} merged, but closure of child issue #{number} could not be "
            f"confirmed ({outcome}). Rerun agent-loop on the issue (or its staged parent) to "
            "retry the closure without replaying the merge; the stage is not complete until "
            "the child is closed."
        )
        return False
    return True


def reconcile_merged_integration_child(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
) -> bool:
    """Resume path: finish an interrupted closure of a MERGED integration PR.

    Returns ``True`` when the canonical PR is MERGED into a trusted non-default
    base and the child is now closed (nothing else to do, no review, no merge
    replay).  Raises when the merge is authenticated but closure is unresolved.
    Returns ``False`` when this is not that situation, so ordinary resume
    handling (including its MERGED rejection) proceeds unchanged.
    """
    if config.dry_run:
        return False
    from .issue_pr_handoff import authenticate_canonical_issue_pr

    authenticated = authenticate_canonical_issue_pr(
        runner, config=config, issue_number=issue_number, issue_context=issue_context
    )
    if authenticated is None or authenticated.state != "MERGED":
        return False
    outcome = close_child_after_integration_merge(
        runner, config=config, issue_context=issue_context, pr_number=authenticated.pr_number
    )
    if outcome in {"closed", "already-closed"}:
        print(
            f"Issue #{issue_number}: PR #{authenticated.pr_number} already merged; child issue "
            "closure completed without replaying the merge."
        )
        return True
    if outcome == "unresolved":
        raise AgentLoopError(
            f"Issue #{issue_number}: canonical PR #{authenticated.pr_number} is MERGED but the "
            "integration-base merge or the child closure could not be authenticated; the stage "
            "is not complete. Close the issue manually once the merge is verified."
        )
    return False
