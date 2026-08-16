"""GitHub CLI operations used by the orchestrator."""

from __future__ import annotations

import datetime
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .ci_health import (
    CiInfrastructureStall,
    PullRequestCheck,
    PullRequestChecks,
    StalledCheck,
    _extract_run_id,
    classify_ci_infrastructure_stall,
    is_wholly_infrastructure_blocked,
)
from .errors import AgentLoopError
from .logging import log
from .round_transport import MAX_GITHUB_BODY_CHARS, prepare_round_comment
from .protocol import parse_signed_human_requirement_body
from .runner import Runner
from .workdirs import active_workdir

if TYPE_CHECKING:
    from .config import AgentLoopConfig

# PullRequestCheck/PullRequestChecks/StalledCheck/CiInfrastructureStall are
# defined in ci_health.py (kept dependency-free of this module's live GitHub
# API calls) and re-exported here for existing importers: checks.py,
# orchestrator.py, prompts.py, and tests/agent_loop_helpers.py.


@dataclass(frozen=True)
class PullRequestMetadata:
    number: int
    repo: str
    title: str | None
    head_branch: str | None
    base_branch: str | None
    head_sha: str | None
    url: str | None
    body: str | None = None
    author_login: str | None = None
    author_id: int | None = None
    head_repo: str | None = None
    is_draft: bool | None = None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class IssueComment:
    author: str | None
    created_at: str | None
    body: str | None


@dataclass(frozen=True)
class IssueContext:
    number: int
    repo: str
    title: str | None
    body: str | None
    url: str | None
    comments: tuple[IssueComment, ...]
    human_requirements: tuple[HumanReviewRequirement, ...] = ()


@dataclass(frozen=True)
class FoundIssue:
    number: int | None
    title: str | None
    url: str | None
    body: str | None


@dataclass(frozen=True)
class HumanReviewRequirement:
    source_type: str
    author: str | None
    created_at: str | None
    url: str | None
    body: str


@dataclass(frozen=True)
class PullRequestReviewContext:
    metadata: PullRequestMetadata
    comments: tuple[IssueComment, ...]
    human_requirements: tuple[HumanReviewRequirement, ...]


@dataclass(frozen=True)
class PullRequestMergeability:
    state: Literal["mergeable", "conflicted", "unknown"]
    mergeable_raw: str | None
    merge_state_raw: str | None
    head_sha: str | None
    base_branch: str | None


PR_METADATA_FIELDS = "number,title,headRefName,baseRefName,headRefOid,url,body"
PR_REVIEW_CONTEXT_FIELDS = f"{PR_METADATA_FIELDS},comments,reviews"
ISSUE_REFERENCE_RE_TEMPLATE = r"(?:#%d\b|/issues/%d\b)"

# A direct issue URL is meaningful without an auto-close keyword.  Shorthand
# references, on the other hand, are only linked here when they are part of a
# GitHub closing phrase so ordinary discussion of ``#123`` stays untouched.
_GITHUB_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<repo>[^/\s#]+/[^/\s#]+)/issues/(?P<number>[1-9]\d*)\b",
    re.IGNORECASE,
)
_CLOSING_ISSUE_REFERENCE_RE = re.compile(
    r"\b(?:close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\b"
    r"\s*:?[ \t]*(?:"
    r"(?P<unqualified>#[1-9]\d*)|"
    r"(?P<qualified>[^\s/#]+/[^\s/#]+#[1-9]\d*)|"
    r"(?P<url>https?://github\.com/[^/\s#]+/[^/\s#]+/issues/[1-9]\d*\b)"
    r")",
    re.IGNORECASE,
)


def parse_linked_issue_numbers(pr_body: str | None, *, repo: str) -> tuple[int, ...]:
    """Return same-repository issues linked by a PR body, in first-seen order.

    GitHub closing phrases accept unqualified and ``owner/repo#N`` forms.  A
    same-repository issue URL is also a link even without a closing phrase.
    Repository comparison is case-insensitive because GitHub repository names
    are case-insensitive.
    """
    if not pr_body:
        return ()

    normalized_repo = repo.casefold()
    matches: list[tuple[int, int]] = []
    for match in _CLOSING_ISSUE_REFERENCE_RE.finditer(pr_body):
        reference = match.group("unqualified") or match.group("qualified") or match.group("url")
        if reference is None:
            continue
        if reference.startswith("#"):
            matches.append((match.start(), int(reference[1:])))
            continue
        if reference.lower().startswith(("http://", "https://")):
            url_match = _GITHUB_ISSUE_URL_RE.fullmatch(reference)
            if url_match and url_match.group("repo").casefold() == normalized_repo:
                matches.append((match.start(), int(url_match.group("number"))))
            continue
        qualified_repo, number = reference.rsplit("#", 1)
        if qualified_repo.casefold() == normalized_repo:
            matches.append((match.start(), int(number)))

    for match in _GITHUB_ISSUE_URL_RE.finditer(pr_body):
        if match.group("repo").casefold() == normalized_repo:
            matches.append((match.start(), int(match.group("number"))))

    seen: set[int] = set()
    numbers: list[int] = []
    for _, number in sorted(matches, key=lambda item: item[0]):
        if number not in seen:
            seen.add(number)
            numbers.append(number)
    return tuple(numbers)


def detect_repo(runner: Runner, cwd: Path, gh_cmd: str) -> str:
    result = runner.run(
        [gh_cmd, "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        cwd=cwd,
    )
    repo = result.stdout.strip()
    if not repo:
        raise AgentLoopError("Unable to detect GitHub repo. Pass --repo owner/name.")
    return repo


def get_repo_default_branch(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    cwd: Path,
) -> str | None:
    result = runner.run(
        [
            config.gh_cmd,
            "repo",
            "view",
            config.repo,
            "--json",
            "defaultBranchRef",
            "--jq",
            ".defaultBranchRef.name",
        ],
        cwd=cwd,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_pr_state(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> str:
    """Return the PR state string ('OPEN', 'CLOSED', or 'MERGED').

    Raises AgentLoopError when the state cannot be determined (non-zero exit or
    absent state field), so the caller can wrap it with issue-level context.
    """
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "number,state,url",
        ],
        cwd=active_workdir(config),
    )
    if result.returncode != 0:
        raise AgentLoopError(f"Unable to determine state of PR #{pr_number}.")
    data = json.loads(result.stdout or "{}")
    state = _optional_str(data.get("state"))
    if not state:
        raise AgentLoopError(f"Unable to determine state of PR #{pr_number}.")
    return state


def validate_open_pr(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> None:
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "number,state,url",
        ],
        cwd=active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    if data.get("state") != "OPEN":
        raise AgentLoopError(
            f"PR #{pr_number} is {data.get('state', 'not open')}; provide an open PR number."
        )


def validate_pr_references_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
) -> None:
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "body,url",
        ],
        cwd=active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    body = _optional_str(data.get("body")) or ""
    reference_re = re.compile(ISSUE_REFERENCE_RE_TEMPLATE % (issue_number, issue_number), re.I)
    if reference_re.search(body):
        return
    raise AgentLoopError(
        f"PR #{pr_number} does not reference issue #{issue_number} in its body. "
        f"Edit the PR description on GitHub to include `Fixes #{issue_number}` or another direct "
        f"issue reference, then rerun the orchestrator as `agent-loop pr {pr_number}` to continue "
        "the review."
    )


# Matches a GitHub auto-close keyword immediately followed by an issue reference,
# e.g. "Closes #12", "fixed #12", "resolves owner/repo#12", or
# "closes https://github.com/owner/repo/issues/12" (GitHub also treats a full
# issue URL as a closing reference). Used to keep staged implementation PRs
# from accidentally auto-closing the parent issue while other split stages
# remain unimplemented (#476, #492 review).
_CLOSE_KEYWORD_ISSUE_RE_TEMPLATE = (
    r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s*:?\s*(?:%s#%d|#%d|\S*/issues/%d)\b"
)


def validate_pr_body_does_not_close_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int,
) -> None:
    """Reject a staged-implementation PR body that would auto-close `issue_number`.

    Used when split/deferred stages remain unfiled or unimplemented for the
    parent issue (#476): a PR implementing only one stage must not use
    `Fixes`/`Closes`/`Resolves` against the parent, or the parent would close
    while other stages remain outstanding. `Refs #N` (or any non-closing
    reference) is fine and is validated separately by
    `validate_pr_references_issue`.
    """
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "body,url",
        ],
        cwd=active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    body = _optional_str(data.get("body")) or ""
    close_re = re.compile(
        _CLOSE_KEYWORD_ISSUE_RE_TEMPLATE
        % (re.escape(config.repo), issue_number, issue_number, issue_number),
        re.I,
    )
    if not close_re.search(body):
        return
    raise AgentLoopError(
        f"PR #{pr_number} body uses a closing keyword (Closes/Fixes/Resolves) against parent "
        f"issue #{issue_number}, but other split stages remain unfiled or unimplemented. Edit the "
        f"PR description to use a non-closing reference (e.g. `Refs #{issue_number}`) instead, then "
        f"rerun `agent-loop pr {pr_number}` to continue the review."
    )


def find_open_pr_referencing_issue(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int
) -> int | None:
    """Find an open PR whose body already references `issue_number`.

    Used to resume PR review instead of re-invoking the coder when a prior
    implementation attempt created a PR but the run aborted before recording
    any handoff marker or comment (#495): the marker-based resume checks only
    help once the marker exists, so a rerun otherwise has no trace of the PR
    and would invoke the coder again, creating a duplicate (as happened with
    #492/#494).
    """
    if config.dry_run:
        return None
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "list",
            "--repo",
            config.repo,
            "--state",
            "open",
            "--json",
            "number,body",
            "--limit",
            "100",
        ],
        cwd=active_workdir(config),
    )
    raw = result.stdout.strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list):
        return None
    reference_re = re.compile(ISSUE_REFERENCE_RE_TEMPLATE % (issue_number, issue_number), re.I)
    matches: list[int] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        body = _optional_str(item.get("body")) or ""
        if not reference_re.search(body):
            continue
        number = item.get("number")
        if isinstance(number, int):
            matches.append(number)
    if not matches:
        return None
    if len(matches) > 1:
        joined = ", ".join(f"#{n}" for n in sorted(matches))
        raise AgentLoopError(
            f"Multiple open PRs ({joined}) reference issue #{issue_number}; cannot automatically "
            "determine which to resume. Close or merge the duplicate PR(s), then rerun "
            "`agent-loop pr <number>` directly to continue review on the correct one."
        )
    return matches[0]


def _parse_pr_metadata(
    data: dict[str, object], *, config: AgentLoopConfig, pr_number: int
) -> PullRequestMetadata:
    return PullRequestMetadata(
        number=int(data.get("number") or pr_number),
        repo=config.repo,
        title=_optional_str(data.get("title")),
        head_branch=_optional_str(data.get("headRefName")),
        base_branch=_optional_str(data.get("baseRefName")),
        head_sha=_optional_str(data.get("headRefOid")),
        url=_optional_str(data.get("url")),
        body=_optional_str(data.get("body")),
        author_login=_author_login(data.get("author")),
        author_id=_author_id(data.get("author")),
        head_repo=_optional_str((data.get("headRepository") or {}).get("nameWithOwner"))
        if isinstance(data.get("headRepository"), dict)
        else None,
        is_draft=data.get("isDraft") if isinstance(data.get("isDraft"), bool) else None,
        labels=tuple(
            label.get("name")
            for label in ((data.get("labels") or {}).get("nodes") or [])
            if isinstance(label, dict) and isinstance(label.get("name"), str)
        ) if isinstance(data.get("labels"), dict) else (),
    )


def _author_login(raw: object) -> str | None:
    if isinstance(raw, dict):
        login = raw.get("login")
        return str(login) if login else None
    return None


def _author_id(raw: object) -> int | None:
    if isinstance(raw, dict) and isinstance(raw.get("id"), int):
        return raw["id"]
    return None


def _human_requirement_sort_key(requirement: HumanReviewRequirement) -> str:
    return requirement.created_at or ""


def _optional_str(raw: object) -> str | None:
    return str(raw) if raw is not None else None


def get_pr_review_context(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    cwd: Path | None = None,
) -> PullRequestReviewContext:
    if config.dry_run:
        return PullRequestReviewContext(
            metadata=PullRequestMetadata(
                number=pr_number,
                repo=config.repo,
                title=None,
                head_branch=None,
                base_branch=None,
                head_sha=None,
                url=None,
            ),
            comments=(),
            human_requirements=(),
        )

    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            PR_REVIEW_CONTEXT_FIELDS,
        ],
        cwd=cwd or active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    comments = _parse_issue_comments(data.get("comments"))
    return PullRequestReviewContext(
        metadata=_parse_pr_metadata(data, config=config, pr_number=pr_number),
        comments=comments,
        human_requirements=_parse_pr_human_requirements(data),
    )


def _classify_mergeability(
    *, mergeable_raw: str | None, merge_state_raw: str | None
) -> Literal["mergeable", "conflicted", "unknown"]:
    # Explicit conflict evidence wins first, even when the other field is
    # null/missing: a DIRTY merge state or a CONFLICTING mergeable value both
    # mean GitHub cannot merge the branch as-is.
    if merge_state_raw == "DIRTY" or mergeable_raw == "CONFLICTING":
        return "conflicted"
    if mergeable_raw == "MERGEABLE":
        return "mergeable"
    return "unknown"


def get_pr_mergeability(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    cwd: Path | None = None,
) -> PullRequestMergeability:
    """Probe GitHub's computed mergeability for `pr_number`.

    A confirmed `mergeable`/`conflicted` state is authoritative. Anything
    else -- a non-zero `gh` exit, unparsable JSON, a null/missing `mergeable`
    with no conflict evidence, or an explicit `"UNKNOWN"` (GitHub is still
    computing it) -- settles as `unknown` so an old `gh`, a token without
    `mergeStateStatus` access, or a transient computation window is never
    mistaken for a real conflict. Only the explicit `"UNKNOWN"` case is worth
    a bounded re-poll, since it is the one case GitHub says will resolve on
    its own shortly.
    """
    if config.dry_run:
        return PullRequestMergeability(
            state="unknown",
            mergeable_raw=None,
            merge_state_raw=None,
            head_sha=None,
            base_branch=None,
        )

    resolved_cwd = cwd or active_workdir(config)
    attempts = max(1, config.mergeability_poll_attempts)
    for attempt in range(attempts):
        result = runner.run(
            [
                config.gh_cmd,
                "pr",
                "view",
                str(pr_number),
                "--repo",
                config.repo,
                "--json",
                "mergeable,mergeStateStatus,headRefOid,baseRefName",
            ],
            cwd=resolved_cwd,
            check=False,
        )
        if result.returncode != 0:
            log(config, f"PR #{pr_number}: mergeability probe failed (gh exit {result.returncode}); treating as unknown")
            return PullRequestMergeability(
                state="unknown",
                mergeable_raw=None,
                merge_state_raw=None,
                head_sha=None,
                base_branch=None,
            )
        try:
            data = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            log(config, f"PR #{pr_number}: mergeability probe returned invalid JSON; treating as unknown")
            return PullRequestMergeability(
                state="unknown",
                mergeable_raw=None,
                merge_state_raw=None,
                head_sha=None,
                base_branch=None,
            )
        mergeable_raw = _optional_str(data.get("mergeable"))
        merge_state_raw = _optional_str(data.get("mergeStateStatus"))
        head_sha = _optional_str(data.get("headRefOid"))
        base_branch = _optional_str(data.get("baseRefName"))
        state = _classify_mergeability(mergeable_raw=mergeable_raw, merge_state_raw=merge_state_raw)
        if state != "unknown" or mergeable_raw != "UNKNOWN":
            return PullRequestMergeability(
                state=state,
                mergeable_raw=mergeable_raw,
                merge_state_raw=merge_state_raw,
                head_sha=head_sha,
                base_branch=base_branch,
            )
        if attempt < attempts - 1:
            log(
                config,
                f"PR #{pr_number}: GitHub is still computing mergeability (UNKNOWN); "
                f"retrying in {config.mergeability_poll_interval_seconds}s "
                f"({attempt + 1}/{attempts})",
            )
            runner.run(
                ["sleep", str(config.mergeability_poll_interval_seconds)],
                cwd=resolved_cwd,
            )
    return PullRequestMergeability(
        state="unknown",
        mergeable_raw="UNKNOWN",
        merge_state_raw=merge_state_raw,
        head_sha=head_sha,
        base_branch=base_branch,
    )


def _parse_issue_comments(raw_comments: object) -> tuple[IssueComment, ...]:
    comments: list[IssueComment] = []
    if not isinstance(raw_comments, list):
        return ()
    for raw_comment in raw_comments:
        if not isinstance(raw_comment, dict):
            continue
        comments.append(
            IssueComment(
                author=_author_login(raw_comment.get("author")),
                created_at=raw_comment.get("createdAt") or raw_comment.get("created_at"),
                body=_optional_str(raw_comment.get("body")),
            )
        )
    return tuple(sorted(comments, key=_comment_sort_key))


def _normalize_check_run_status(raw_run: object) -> str:
    if not isinstance(raw_run, dict):
        return "unknown"
    status = _optional_str(raw_run.get("status"))
    conclusion = _optional_str(raw_run.get("conclusion"))
    if status != "completed":
        return status or "pending"
    return conclusion or "completed"


def _classify_check_status(status: str) -> Literal["passing", "pending", "failing"]:
    normalized = status.strip().lower()
    if normalized in {"success", "neutral", "skipped"}:
        return "passing"
    if normalized in {
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "startup_failure",
        "stale",
        "error",
    }:
        return "failing"
    return "pending"


def _dedupe_checks(checks: list[PullRequestCheck]) -> tuple[PullRequestCheck, ...]:
    deduped: dict[tuple[str, str], PullRequestCheck] = {}
    for check in checks:
        deduped.setdefault((check.kind, check.name), check)
    return tuple(deduped.values())


def _parse_check_runs_payload(payload: object) -> tuple[list[PullRequestCheck], list[str]]:
    """Parse the `commits/{sha}/check-runs` response into `PullRequestCheck`s.

    Shared by `get_pr_checks` (full board) and `get_check_record` (a single
    configured check's full timestamped record for the auto-merge wait loop).
    """
    checks: list[PullRequestCheck] = []
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("check-runs response was not a JSON object")
        return checks, errors
    for raw_check in payload.get("check_runs") or []:
        if not isinstance(raw_check, dict):
            continue
        name = _optional_str(raw_check.get("name"))
        if not name:
            continue
        url = _optional_str(raw_check.get("html_url") or raw_check.get("details_url"))
        raw_id = raw_check.get("id")
        checks.append(
            PullRequestCheck(
                name=name,
                kind="check_run",
                status=_normalize_check_run_status(raw_check),
                url=url,
                check_id=raw_id if isinstance(raw_id, int) else None,
                run_id=_extract_run_id(url),
                created_at=_optional_str(raw_check.get("created_at")),
                started_at=_optional_str(raw_check.get("started_at")),
                completed_at=_optional_str(raw_check.get("completed_at")),
                creator_login=_author_login(raw_check.get("app")),
                creator_id=_author_id(raw_check.get("app")),
                description=(
                    _optional_str(raw_check["output"].get("summary"))
                    if isinstance(raw_check.get("output"), dict)
                    else None
                ),
            )
        )
    return checks, errors


def _parse_commit_statuses_payload(payload: object) -> tuple[list[PullRequestCheck], list[str]]:
    checks: list[PullRequestCheck] = []
    errors: list[str] = []
    if not isinstance(payload, dict):
        errors.append("commit-status response was not a JSON object")
        return checks, errors
    for raw_status in payload.get("statuses") or []:
        if not isinstance(raw_status, dict):
            continue
        name = _optional_str(raw_status.get("context"))
        if not name:
            continue
        checks.append(
            PullRequestCheck(
                name=name,
                kind="status_context",
                status=_optional_str(raw_status.get("state")) or "pending",
                url=_optional_str(raw_status.get("target_url")),
                created_at=_optional_str(raw_status.get("created_at")),
                completed_at=_optional_str(raw_status.get("updated_at")),
                creator_login=_author_login(raw_status.get("creator")),
                creator_id=_author_id(raw_status.get("creator")),
                description=_optional_str(raw_status.get("description")),
            )
        )
    return checks, errors


def _fetch_branch_protection_required_checks(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    base_branch: str | None,
) -> tuple[Literal["configured", "not_found", "forbidden", "unavailable"], tuple[str, ...], str | None]:
    if not base_branch:
        return ("unavailable", (), "PR base branch is unavailable, so branch protection could not be checked.")

    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/branches/{base_branch}/protection/required_status_checks",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode == 0:
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return ("unavailable", (), "Branch protection response was not valid JSON.")
        required_checks: list[str] = []
        for context in payload.get("contexts") or []:
            if isinstance(context, str) and context:
                required_checks.append(context)
        for check in payload.get("checks") or []:
            if isinstance(check, dict):
                context = _optional_str(check.get("context"))
                if context:
                    required_checks.append(context)
        return ("configured", tuple(dict.fromkeys(required_checks)), None)

    stderr = (result.stderr or "").lower()
    stdout = (result.stdout or "").lower()
    combined = f"{stdout}\n{stderr}"
    if "404" in combined:
        return (
            "not_found",
            (),
            "Required status checks are not configured on the PR base branch.",
        )
    if "403" in combined or "forbidden" in combined:
        return (
            "forbidden",
            (),
            "Current GitHub token cannot inspect branch protection on the PR base branch.",
        )
    return (
        "unavailable",
        (),
        "GitHub branch protection could not be inspected due to an unexpected API failure.",
    )


def get_pr_checks(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    metadata: PullRequestMetadata,
    now: datetime.datetime | None = None,
) -> PullRequestChecks:
    if config.dry_run:
        return PullRequestChecks(
            state="no_checks",
            required_checks=(),
            passing=(),
            pending=(),
            failing=(),
            missing_required=(),
            branch_protection_status="unavailable",
            branch_protection_note="Dry run mode does not query live GitHub PR checks.",
            check_query_status="unavailable",
            check_query_errors=("Dry run mode does not query live GitHub PR checks.",),
        )

    if not metadata.head_sha:
        return PullRequestChecks(
            state="unavailable",
            required_checks=(),
            passing=(),
            pending=(),
            failing=(),
            missing_required=(),
            branch_protection_status="unavailable",
            branch_protection_note="PR head SHA is unavailable, so GitHub PR checks could not be queried.",
            check_query_status="unavailable",
            check_query_errors=("PR head SHA is unavailable.",),
        )

    branch_protection_status, required_checks, branch_protection_note = (
        _fetch_branch_protection_required_checks(
            runner,
            config=config,
            base_branch=metadata.base_branch,
        )
    )
    check_runs_result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{metadata.head_sha}/check-runs",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    statuses_result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{metadata.head_sha}/status",
        ],
        cwd=active_workdir(config),
        check=False,
    )

    check_errors: list[str] = []
    checks: list[PullRequestCheck] = []
    check_runs_ok = False
    statuses_ok = False

    if check_runs_result.returncode == 0:
        try:
            payload = json.loads(check_runs_result.stdout or "{}")
        except json.JSONDecodeError:
            check_errors.append("check-runs response was not valid JSON")
        else:
            check_runs_ok = True
            parsed_checks, parse_errors = _parse_check_runs_payload(payload)
            checks.extend(parsed_checks)
            check_errors.extend(parse_errors)
    else:
        check_errors.append("check-runs query failed")

    if statuses_result.returncode == 0:
        try:
            payload = json.loads(statuses_result.stdout or "{}")
        except json.JSONDecodeError:
            check_errors.append("commit-status response was not valid JSON")
        else:
            statuses_ok = True
            parsed_statuses, parse_errors = _parse_commit_statuses_payload(payload)
            checks.extend(parsed_statuses)
            check_errors.extend(parse_errors)
    else:
        check_errors.append("commit-status query failed")

    if check_runs_ok and statuses_ok:
        check_query_status: Literal["ok", "partial", "unavailable"] = "ok"
    elif check_runs_ok or statuses_ok:
        check_query_status = "partial"
    else:
        check_query_status = "unavailable"

    deduped_checks = _dedupe_checks(checks)
    passing = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "passing")
    pending = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "pending")
    failing = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "failing")
    observed_names = {check.name for check in deduped_checks}
    missing_required = tuple(name for name in required_checks if name not in observed_names)
    infrastructure_stalls = classify_ci_infrastructure_stall(
        deduped_checks,
        now=now or datetime.datetime.now(datetime.timezone.utc),
        grace_seconds=config.ci_queued_grace_seconds,
    ).checks

    state: Literal["passing", "failing", "pending", "no_checks", "unavailable"]
    if failing:
        state = "failing"
    elif pending or missing_required:
        state = "pending"
    elif passing:
        state = "passing"
    elif branch_protection_status == "configured" and required_checks:
        state = "pending"
    elif branch_protection_status in {"configured", "not_found", "forbidden"}:
        state = "no_checks"
    elif check_errors:
        state = "unavailable"
    else:
        state = "no_checks"

    if state == "unavailable" and not branch_protection_note and check_errors:
        branch_protection_note = "; ".join(check_errors)

    return PullRequestChecks(
        state=state,
        required_checks=required_checks,
        passing=passing,
        pending=pending,
        failing=failing,
        missing_required=missing_required,
        branch_protection_status=branch_protection_status,
        branch_protection_note=branch_protection_note,
        check_query_status=check_query_status,
        check_query_errors=tuple(check_errors),
        infrastructure_stalls=infrastructure_stalls,
    )


def _parse_pr_human_requirements(data: dict[str, object]) -> tuple[HumanReviewRequirement, ...]:
    requirements: list[HumanReviewRequirement] = []
    for raw_comment in data.get("comments") or []:
        if not isinstance(raw_comment, dict):
            continue
        body = parse_signed_human_requirement_body(raw_comment.get("body"))
        if body is None:
            continue
        requirements.append(
            HumanReviewRequirement(
                source_type="PR comment",
                author=_author_login(raw_comment.get("author")),
                created_at=raw_comment.get("createdAt") or raw_comment.get("created_at"),
                url=raw_comment.get("url"),
                body=body,
            )
        )
    for raw_review in data.get("reviews") or []:
        if not isinstance(raw_review, dict):
            continue
        body = parse_signed_human_requirement_body(raw_review.get("body"))
        if body is None:
            continue
        requirements.append(
            HumanReviewRequirement(
                source_type="PR review",
                author=_author_login(raw_review.get("author")),
                created_at=raw_review.get("submittedAt")
                or raw_review.get("submitted_at")
                or raw_review.get("createdAt")
                or raw_review.get("created_at"),
                url=raw_review.get("url"),
                body=body,
            )
        )
    return tuple(sorted(requirements, key=_human_requirement_sort_key))


def validate_open_issue(runner: Runner, *, config: AgentLoopConfig, issue_number: int) -> None:
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/issues/{issue_number}",
            "--jq",
            "{number:.number,state:.state,is_pr:has(\"pull_request\"),url:.html_url}",
        ],
        cwd=active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    if data.get("is_pr"):
        raise AgentLoopError(
            f"#{issue_number} is a pull request, not an issue. Use `agent-loop pr {issue_number}`."
        )
    if data.get("state") != "open":
        raise AgentLoopError(
            f"Issue #{issue_number} is {data.get('state', 'not open')}; provide an open issue number."
        )


def _comment_sort_key(comment: IssueComment) -> str:
    return comment.created_at or ""


def get_issue_context(runner: Runner, *, config: AgentLoopConfig, issue_number: int) -> IssueContext:
    if config.dry_run:
        return IssueContext(
            number=issue_number,
            repo=config.repo,
            title=None,
            body=None,
            url=None,
            comments=(),
            human_requirements=(),
        )

    result = runner.run(
        [
            config.gh_cmd,
            "issue",
            "view",
            str(issue_number),
            "--repo",
            config.repo,
            "--comments",
            "--json",
            "number,title,body,url,author,createdAt,comments",
        ],
        cwd=active_workdir(config),
    )
    data = json.loads(result.stdout or "{}")
    comments = _parse_issue_comments(data.get("comments"))
    return IssueContext(
        number=int(data.get("number") or issue_number),
        repo=config.repo,
        title=_optional_str(data.get("title")),
        body=_optional_str(data.get("body")),
        url=_optional_str(data.get("url")),
        comments=comments,
        human_requirements=_parse_issue_human_requirements(data),
    )


def _parse_issue_human_requirements(data: dict[str, object]) -> tuple[HumanReviewRequirement, ...]:
    requirements: list[HumanReviewRequirement] = []
    issue_body = parse_signed_human_requirement_body(_optional_str(data.get("body")))
    if issue_body is not None:
        requirements.append(
            HumanReviewRequirement(
                source_type="Issue body",
                author=_author_login(data.get("author")),
                created_at=_optional_str(data.get("createdAt")) or _optional_str(data.get("created_at")),
                url=_optional_str(data.get("url")),
                body=issue_body,
            )
        )
    for raw_comment in data.get("comments") or []:
        if not isinstance(raw_comment, dict):
            continue
        body = parse_signed_human_requirement_body(_optional_str(raw_comment.get("body")))
        if body is None:
            continue
        requirements.append(
            HumanReviewRequirement(
                source_type="Issue comment",
                author=_author_login(raw_comment.get("author")),
                created_at=_optional_str(raw_comment.get("createdAt"))
                or _optional_str(raw_comment.get("created_at")),
                url=_optional_str(raw_comment.get("url")),
                body=body,
            )
        )
    return tuple(sorted(requirements, key=_human_requirement_sort_key))


def post_pr_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: str,
) -> None:
    bodies = prepare_round_comment(body)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to PR #{pr_number}")
    log(config, f"Posting agent output to PR #{pr_number}")
    for prepared in bodies:
        _post_comment_body(runner, config=config, command=["pr", "comment", str(pr_number)], body=prepared)


def post_issue_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: str,
) -> None:
    bodies = prepare_round_comment(body)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to issue #{issue_number}")
    log(config, f"Posting agent output to issue #{issue_number}")
    for prepared in bodies:
        _post_comment_body(runner, config=config, command=["issue", "comment", str(issue_number)], body=prepared)


def _post_comment_body(runner: Runner, *, config: AgentLoopConfig, command: list[str], body: str) -> None:
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(f"GitHub comment body exceeds {MAX_GITHUB_BODY_CHARS} characters; shorten the response.")
    if config.dry_run:
        runner.run([config.gh_cmd, *command, "--repo", config.repo, "--body", body], cwd=active_workdir(config))
        return
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        runner.run([config.gh_cmd, *command, "--repo", config.repo, "--body-file", path], cwd=active_workdir(config))
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def create_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    title: str,
    body: str,
) -> str | None:
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(f"GitHub issue body exceeds {MAX_GITHUB_BODY_CHARS} characters; shorten the response.")
    log(config, f"Creating GitHub issue: {title}")
    if config.dry_run:
        result = runner.run(
            [
                config.gh_cmd,
                "issue",
                "create",
                "--repo",
                config.repo,
                "--title",
                title,
                "--body",
                body,
            ],
            cwd=active_workdir(config),
        )
        issue_url = result.stdout.strip() or None
        if issue_url:
            log(config, f"Created GitHub issue: {issue_url}")
        return issue_url

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        result = runner.run(
            [
                config.gh_cmd,
                "issue",
                "create",
                "--repo",
                config.repo,
                "--title",
                title,
                "--body-file",
                path,
            ],
            cwd=active_workdir(config),
        )
        issue_url = result.stdout.strip() or None
        if issue_url:
            log(config, f"Created GitHub issue: {issue_url}")
        return issue_url
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def search_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    search: str,
    state: str = "all",
) -> tuple[FoundIssue, ...]:
    """Search issues in `config.repo`, used to recover from a create-then-crash window (#476).

    Split-issue materialization posts its idempotency marker only after every
    child issue is created; if the process crashes between a `create_issue`
    call and posting that marker, a rerun must find already-created children
    instead of duplicating them. This wraps `gh issue list --search` for that
    recovery pass. Dry-run returns an empty tuple so the dry-run
    materialization path still previews creations instead of "adopting"
    nothing.
    """
    log(config, f"Searching GitHub issues in {config.repo}: {search}")
    if config.dry_run:
        runner.run(
            [
                config.gh_cmd,
                "issue",
                "list",
                "--repo",
                config.repo,
                "--search",
                search,
                "--state",
                state,
                "--json",
                "number,title,url,body",
            ],
            cwd=active_workdir(config),
        )
        return ()
    result = runner.run(
        [
            config.gh_cmd,
            "issue",
            "list",
            "--repo",
            config.repo,
            "--search",
            search,
            "--state",
            state,
            "--json",
            "number,title,url,body",
        ],
        cwd=active_workdir(config),
    )
    raw = result.stdout.strip()
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    found: list[FoundIssue] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        number = item.get("number")
        found.append(
            FoundIssue(
                number=int(number) if isinstance(number, int) else None,
                title=_optional_str(item.get("title")),
                url=_optional_str(item.get("url")),
                body=_optional_str(item.get("body")),
            )
        )
    return tuple(found)


def get_pr_head_sha(runner: Runner, config: AgentLoopConfig, pr_number: int) -> str:
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "headRefOid",
            "--jq",
            ".headRefOid",
        ],
        cwd=active_workdir(config),
    )
    sha = result.stdout.strip()
    if not sha:
        raise AgentLoopError(f"Unable to resolve head SHA for PR #{pr_number}.")
    return sha


def get_check_record(runner: Runner, config: AgentLoopConfig, head_sha: str) -> PullRequestCheck | None:
    """Full timestamped record for `config.ci_check_name`, or None if absent/unavailable."""
    result = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/commits/{head_sha}/check-runs"],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    checks, _errors = _parse_check_runs_payload(payload)
    for check in checks:
        if check.name == config.ci_check_name:
            return check
    return None


def get_check_status(runner: Runner, config: AgentLoopConfig, head_sha: str) -> str:
    record = get_check_record(runner, config, head_sha)
    return record.status if record is not None else "pending"


@dataclass(frozen=True)
class CiWaitOutcome:
    status: Literal["passed", "infrastructure_stall", "merge_conflict"]
    stall: CiInfrastructureStall | None = None
    pr_checks: PullRequestChecks | None = None
    mergeability: PullRequestMergeability | None = None


@dataclass(frozen=True)
class CiWatchOutcome:
    """Terminal result from the opt-in full-board post-approval watcher."""

    status: Literal[
        "passed",
        "no_checks",
        "failed",
        "timeout",
        "infrastructure_stall",
        "merge_conflict",
        "head_changed",
        "dry_run",
    ]
    pr_checks: PullRequestChecks | None = None
    failed_checks: tuple[PullRequestCheck, ...] = ()
    mergeability: PullRequestMergeability | None = None
    head_sha: str | None = None
    stall: CiInfrastructureStall | None = None
    attempts_used: int = 0


def watch_pr_checks(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    *,
    metadata: PullRequestMetadata,
    deadline: float | None = None,
    attempts: int | None = None,
) -> CiWatchOutcome:
    """Synchronously watch the complete current-head check board.

    This deliberately sits beside the legacy single-check ``wait_for_ci``.
    It has no worker or persisted process; interrupting the foreground runner
    stops it immediately and a later invocation simply fetches fresh state.
    """
    if config.dry_run:
        return CiWatchOutcome(status="dry_run", head_sha=metadata.head_sha)
    deadline = deadline if deadline is not None else time.monotonic() + config.ci_timeout_seconds
    limit = attempts if attempts is not None else max(
        1, config.ci_timeout_seconds // config.ci_poll_interval_seconds
    )
    latest: PullRequestChecks | None = None
    for attempt in range(limit):
        # A flaky head probe is diagnostic only.  The full-board and
        # mergeability probes already degrade transient API failures safely.
        try:
            current_head = get_pr_head_sha(runner, config, pr_number)
        except AgentLoopError as error:
            log(config, f"PR #{pr_number}: head probe failed while watching ({error}); retrying")
            current_head = metadata.head_sha
        if metadata.head_sha and current_head != metadata.head_sha:
            return CiWatchOutcome(
                status="head_changed", pr_checks=latest, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if mergeability.state == "conflicted":
            return CiWatchOutcome(
                status="merge_conflict", pr_checks=latest, mergeability=mergeability,
                head_sha=current_head, attempts_used=attempt + 1,
            )
        snapshot = get_pr_checks(runner, config=config, metadata=metadata)
        latest = snapshot
        if is_wholly_infrastructure_blocked(snapshot):
            return CiWatchOutcome(
                status="infrastructure_stall",
                pr_checks=snapshot,
                head_sha=current_head,
                stall=CiInfrastructureStall(checks=snapshot.infrastructure_stalls),
                attempts_used=attempt + 1,
            )
        if snapshot.state == "failing":
            return CiWatchOutcome(
                status="failed", pr_checks=snapshot, failed_checks=snapshot.failing,
                head_sha=current_head, attempts_used=attempt + 1,
            )
        reliable = snapshot.check_query_status == "ok" and snapshot.branch_protection_status in {
            "configured", "not_found", "forbidden",
        }
        if snapshot.state == "passing" and reliable and not snapshot.pending and not snapshot.missing_required:
            return CiWatchOutcome(
                status="passed", pr_checks=snapshot, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        if snapshot.state == "no_checks" and reliable and not snapshot.missing_required:
            return CiWatchOutcome(
                status="no_checks", pr_checks=snapshot, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        if time.monotonic() >= deadline or attempt == limit - 1:
            return CiWatchOutcome(
                status="timeout", pr_checks=latest, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AssertionError("CI watch loop must return a terminal outcome")


def wait_for_ci(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    *,
    metadata: PullRequestMetadata,
) -> CiWaitOutcome:
    """Poll the configured CI check before auto-merge.

    A queued-too-long or pre-execution-cancelled check is not, by itself,
    grounds to stop: a stall on the single configured check only ends the
    wait once a full `get_pr_checks` snapshot confirms the whole PR check
    board is wholly infrastructure-blocked (`is_wholly_infrastructure_blocked`).
    Otherwise this keeps today's contract exactly: keep polling and ultimately
    raise `AgentLoopError` on a genuine terminal failure or `ci_timeout_seconds`
    expiry.

    Each poll also re-checks GitHub mergeability first (#606): once the base
    advances or a rebase-required situation appears, the PR's current-head CI
    is no longer a reliable merge signal, so a confirmed conflict ends the
    wait immediately without attempting a merge.
    """
    log(config, f"Waiting for GitHub check '{config.ci_check_name}' before merge")
    head_sha = get_pr_head_sha(runner, config, pr_number)
    attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    terminal_failures = {
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "startup_failure",
        "skipped",
    }
    for attempt in range(attempts):
        mergeability = get_pr_mergeability(runner, config=config, pr_number=pr_number)
        if mergeability.state == "conflicted":
            log(config, f"PR #{pr_number}: GitHub reports a merge conflict; stopping CI wait")
            return CiWaitOutcome(status="merge_conflict", mergeability=mergeability)

        record = get_check_record(runner, config, head_sha)
        status = record.status if record is not None else "pending"
        log(config, f"GitHub check '{config.ci_check_name}' status: {status}")
        if status == "success":
            return CiWaitOutcome(status="passed")

        stall = classify_ci_infrastructure_stall(
            [record] if record is not None else [],
            now=datetime.datetime.now(datetime.timezone.utc),
            grace_seconds=config.ci_queued_grace_seconds,
        )
        if stall.is_stalled:
            snapshot = get_pr_checks(runner, config=config, metadata=metadata)
            if is_wholly_infrastructure_blocked(snapshot):
                return CiWaitOutcome(status="infrastructure_stall", stall=stall, pr_checks=snapshot)

        if status in terminal_failures:
            raise AgentLoopError(f"CI check '{config.ci_check_name}' failed with status: {status}")
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AgentLoopError(
        f"CI check '{config.ci_check_name}' did not pass within {config.ci_timeout_seconds}s"
    )


def merge_pr(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    *,
    expected_head_sha: str | None = None,
) -> None:
    log(config, f"Merging PR #{pr_number}")
    command = [config.gh_cmd, "pr", "merge", str(pr_number), "--repo", config.repo, "--merge"]
    if expected_head_sha:
        command.extend(["--match-head-commit", expected_head_sha])
    runner.run(
        command,
        cwd=active_workdir(config),
    )
