"""Bounded, same-repository recovery for a missing or stale GitHub pull head ref."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import git_transport
from .errors import AgentLoopError
from .github_retry import run_gh_read
from .workdirs import github_api_cwd


def _same_repo(value: Any, repo: str) -> bool:
    return isinstance(value, str) and value.casefold() == repo.casefold()


def _fallback_branch(destination: Path, number: int, *, config: Any,
                     runner: Any, metadata: Any) -> tuple[str, str]:
    """Authorize from one fresh REST observation, never from metadata.repo alone."""
    refused = "PR head ref: same-repository branch recovery could not be verified."
    sha = getattr(metadata, "head_sha", None)
    branch = getattr(metadata, "head_branch", None)
    base_branch = getattr(metadata, "base_branch", None)
    if (getattr(metadata, "number", None) != number
            or not _same_repo(getattr(metadata, "repo", None), config.repo)
            or not isinstance(sha, str) or not git_transport._SHA.fullmatch(sha)
            or not isinstance(branch, str)
            or not isinstance(base_branch, str) or not base_branch):
        raise AgentLoopError(refused)
    source = f"refs/heads/{branch}"
    if (not git_transport._REF.fullmatch(source)
            or runner.run(("git", "check-ref-format", source), cwd=destination,
                          check=False).returncode):
        raise AgentLoopError(refused)
    try:
        result = run_gh_read(
            runner, [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{number}"],
            cwd=github_api_cwd(), check=False,
        )
        if result.returncode:
            raise AgentLoopError(refused)
        data = json.loads(result.stdout)
        if not isinstance(data, dict):
            raise ValueError()
        base, head = data.get("base"), data.get("head")
        if not isinstance(base, dict) or not isinstance(head, dict):
            raise ValueError()
        base_repo, head_repo = base.get("repo"), head.get("repo")
        if (type(data.get("number")) is not int or data["number"] != number
                or data.get("state") != "open"
                or not isinstance(base_repo, dict) or not isinstance(head_repo, dict)
                or not _same_repo(base_repo.get("full_name"), config.repo)
                or not _same_repo(head_repo.get("full_name"), config.repo)
                or head.get("ref") != branch or head.get("sha") != sha
                or base.get("ref") != base_branch):
            raise ValueError()
    except (AgentLoopError, OSError, ValueError, TypeError, RecursionError):
        raise AgentLoopError(refused) from None
    return source, sha


def fetch_pr_head(destination: Path, number: int, *, config: Any, runner: Any,
                  pr_metadata: Any = None) -> str:
    """Try the pull ref first; a missing or stale result permits one fallback.

    The caller retains its existing store/checkout lock and checkout ordering.
    Both transports use the configured origin, never an API-provided URL.
    """
    options = dict(repo=config.repo, runner=runner, gh_cmd=config.gh_cmd,
                   local_origin=config.trusted_local_origin,
                   ssh_identity=getattr(config, "git_ssh_identity", None),
                   forbidden_roots=getattr(config, "transport_forbidden_roots", ()))
    tracking = f"refs/remotes/origin/pr/{number}"
    pull_ref = f"refs/pull/{number}/head"
    stale = None
    try:
        fetched = git_transport.import_ref(destination, pull_ref, tracking, **options)
        expected = getattr(pr_metadata, "head_sha", None)
        if pr_metadata is None:
            return fetched
        if not isinstance(expected, str) or not git_transport._SHA.fullmatch(expected):
            raise AgentLoopError("PR head metadata requires a full hexadecimal expected SHA.")
        if fetched == expected:
            return fetched
        stale = f"stale pull ref: {pull_ref} at {fetched}, PR head {expected}"
    except git_transport.TransportFailure as exc:
        if exc.label != "remote ref missing":
            raise
    try:
        source, expected = _fallback_branch(destination, number, config=config,
                                            runner=runner, metadata=pr_metadata)
        return git_transport.import_ref(destination, source, tracking,
                                        expected_sha=expected, **options)
    except AgentLoopError as exc:
        if stale is None:
            raise
        # Once staleness is observed, rejection is fatal even if the recovery
        # transport had a transient category. Keep the safe cause, not that
        # category, so reviewer settlement cannot reduce it to unavailability.
        reason = exc.label if isinstance(exc, git_transport.TransportFailure) else str(exc)
        raise AgentLoopError(f"{stale}; branch recovery failed: {reason}") from None
