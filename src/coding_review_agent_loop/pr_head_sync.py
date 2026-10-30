"""Bounded, read-only observation of PR-record and tracked-branch agreement."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import quote

from .errors import AgentLoopError
from .github import PullRequestReviewContext, get_pr_review_context as read_pr_context
from .github_retry import run_gh_read
from .managed_ci_recovery_commands import render_managed_ci_resume_command
from .workdirs import active_workdir

if TYPE_CHECKING:
    from .config import AgentLoopConfig
    from .runner import Runner


class PrHeadObservationError(AgentLoopError):
    """GitHub cannot supply a safe, coherent PR-head observation."""

    def __init__(self, message: str, *, desync: dict | None = None) -> None:
        super().__init__(message)
        self.desync = dict(desync) if desync is not None else None


def _sha(value: object) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{40}", value) is None:
        raise PrHeadObservationError("PR-head verification failed: expected a full hexadecimal commit SHA.")
    return value


def _object(value: object) -> dict:
    if not isinstance(value, dict):
        raise PrHeadObservationError("PR-head verification failed: missing or malformed GitHub object.")
    return value


def _read(runner: Runner, config: AgentLoopConfig, endpoint: str, cwd: Path) -> dict:
    try:
        result = run_gh_read(runner, [config.gh_cmd, "api", endpoint], cwd=cwd)
        if result.returncode:
            raise PrHeadObservationError(f"PR-head verification failed reading {endpoint}.")
        return _object(json.loads(result.stdout))
    except (AgentLoopError, ValueError) as exc:
        raise PrHeadObservationError(f"PR-head verification failed reading {endpoint}: {exc}") from exc


def desync_message(
    config: AgentLoopConfig, *, number: int, repository: str, ref: str,
    head: str, tip: str, origin: str | None = None, issue_number: int | None = None,
    managed_hint: bool = False,
) -> str:
    message = (
        f"GitHub PR-head desync for PR #{number}: PR record {head}, tracked branch "
        f"{repository}:{ref} at descendant {tip}, after three samples two seconds apart. "
        "Review, CI qualification and merge cannot proceed while GitHub disagrees. "
        "Recheck current remote PR and branch state first; the lag may have cleared. "
        f"Only if it persists, preserve commit {tip} under a safe reference and verify the remote "
        f"state before an operator rewinds the branch to {head}, then normally fast-forward "
        f"re-pushes the saved {tip}. This force-push recovery is operator-only. "
        "Rerun only after the PR record and tracked branch agree."
    )
    if origin == "issue-created":
        message += " Manual rewind invalidates managed continuity; existing stop cleanup may remove the label."
        if issue_number is not None:
            message += " Resume with fresh authorization: " + render_managed_ci_resume_command(
                config, pr_number=number, managed_ci=True, fresh_authorization=True,
                fresh_issue_number=issue_number,
            )
        else:
            message += " Supply the authenticated owning issue for fresh issue-scoped authorization."
    elif origin in {"source-managed", "adopted"}:
        message += " Existing origin-specific managed-CI recovery and cleanup still apply."
    elif origin is None and (config.managed_ci or managed_hint):
        message += (
            " If this is an issue-created managed PR, manual rewind invalidates continuity; "
            "cleanup may remove its label. Authenticate the owning issue before requesting fresh authorization."
        )
    return message


def observe_pr_head(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int,
    context: PullRequestReviewContext, cwd: Path | None = None,
    origin: str | None = None, issue_number: int | None = None,
) -> PullRequestReviewContext:
    """Return a complete coherent context; never substitute the branch tip."""
    if config.dry_run and context.metadata.head_sha is None:
        return context
    resolved_cwd = cwd or active_workdir(config)
    identity = None
    selected_identity = (context.metadata.repo, context.metadata.head_branch, context.metadata.base_branch)
    for attempt in range(3):
        metadata = context.metadata
        if (metadata.repo, metadata.head_branch, metadata.base_branch) != selected_identity:
            raise PrHeadObservationError("PR-head verification failed: PR repository/ref/base identity changed.")
        head = _sha(metadata.head_sha)
        pull = _read(runner, config, f"repos/{config.repo}/pulls/{pr_number}", resolved_cwd)
        rest_head, base = _object(pull.get("head")), _object(pull.get("base"))
        repository = _object(rest_head.get("repo")).get("full_name")
        ref = rest_head.get("ref")
        if not isinstance(repository, str) or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
            raise PrHeadObservationError("PR-head verification failed: invalid head repository.")
        if not isinstance(ref, str) or not ref or ref != metadata.head_branch or base.get("ref") != metadata.base_branch:
            raise PrHeadObservationError("PR-head verification failed: inconsistent repository/ref/base identity.")
        base_repository = _object(base.get("repo")).get("full_name")
        if not isinstance(base_repository, str) or any(part in {".", ".."} for part in repository.split("/")):
            raise PrHeadObservationError("PR-head verification failed: invalid repository identity.")
        current_identity = (repository.casefold(), ref, base.get("ref"), base_repository.casefold())
        if current_identity[-1] != config.repo.casefold() or (identity is not None and identity != current_identity):
            raise PrHeadObservationError("PR-head verification failed: repository/ref/base identity changed.")
        identity = current_identity
        rest_sha = _sha(rest_head.get("sha"))
        branch = _read(runner, config, f"repos/{repository}/git/ref/heads/{quote(ref, safe='')}", resolved_cwd)
        if branch.get("ref") != f"refs/heads/{ref}":
            raise PrHeadObservationError("PR-head verification failed: branch ref identity disagrees.")
        tip = _sha(_object(branch.get("object")).get("sha"))
        if head == rest_sha == tip:
            return context
        if head == rest_sha:
            comparison = _read(runner, config, f"repos/{repository}/compare/{head}...{tip}", resolved_cwd)
            status = comparison.get("status")
            compare_base = _sha(_object(comparison.get("base_commit")).get("sha"))
            merge_base = _sha(_object(comparison.get("merge_base_commit")).get("sha"))
            if compare_base != head or status not in {"ahead", "behind", "diverged"}:
                raise PrHeadObservationError("PR-head verification failed: inconsistent ancestry comparison.")
            if status == "ahead" and merge_base != head:
                raise PrHeadObservationError("PR-head verification failed: inconsistent descendant proof.")
            if (status == "behind" and merge_base != tip) or (status == "diverged" and merge_base in {head, tip}):
                raise PrHeadObservationError("PR-head verification failed: inconsistent non-descendant proof.")
            if status in {"behind", "diverged"}:
                return context
        if attempt == 2:
            if head != rest_sha:
                raise PrHeadObservationError("PR-head coherence verification failed after three samples: PR projections disagree.")
            labels = pull.get("labels")
            managed_hint = isinstance(labels, list) and any(
                isinstance(label, dict) and label.get("name") == "agent-loop-managed" for label in labels
            )
            details = dict(number=pr_number, repository=repository, ref=ref, head=head, tip=tip)
            raise PrHeadObservationError(desync_message(
                config, **details, origin=origin, issue_number=issue_number, managed_hint=managed_hint,
            ), desync=details)
        runner.run(["sleep", "2"], cwd=resolved_cwd)
        context = _context(runner, config=config, pr_number=pr_number, cwd=resolved_cwd)
    raise AssertionError("unreachable")


def get_observed_pr_review_context(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path | None = None,
    origin: str | None = None, issue_number: int | None = None,
) -> PullRequestReviewContext:
    """Lifecycle getter: observe every newly acquired complete PR snapshot."""
    context = _context(runner, config=config, pr_number=pr_number, cwd=cwd)
    return observe_pr_head(runner, config=config, pr_number=pr_number, context=context,
                           cwd=cwd, origin=origin, issue_number=issue_number)


def _context(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, cwd: Path | None = None,
) -> PullRequestReviewContext:
    try:
        return read_pr_context(runner, config=config, pr_number=pr_number, cwd=cwd)
    except AgentLoopError as exc:
        raise PrHeadObservationError(f"PR-head context verification failed: {exc}") from exc


def explain_pr_head_recovery(
    error: BaseException | None, config: AgentLoopConfig, *,
    origin: str | None, issue_number: int | None,
) -> None:
    """Enrich a stop from already authenticated scope without masking cleanup."""
    if not isinstance(error, PrHeadObservationError) or error.desync is None:
        return
    if origin not in {"issue-created", "source-managed", "adopted"}:
        return
    try:
        message = desync_message(config, **error.desync, origin=origin, issue_number=issue_number)
    except Exception:  # Diagnostics must preserve the primary stop and its cleanup.
        return
    error.args = (message,)
