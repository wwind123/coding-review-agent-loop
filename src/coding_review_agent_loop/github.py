"""GitHub CLI operations used by the orchestrator."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .errors import AgentLoopError
from .logging import log
from .protocol import parse_signed_human_requirement_body
from .runner import Runner
from .workdirs import active_workdir

if TYPE_CHECKING:
    from .config import AgentLoopConfig


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
class PullRequestCheck:
    name: str
    kind: Literal["check_run", "status_context"]
    status: str
    url: str | None = None


@dataclass(frozen=True)
class PullRequestChecks:
    state: Literal["passing", "failing", "pending", "no_checks", "unavailable"]
    required_checks: tuple[str, ...]
    passing: tuple[PullRequestCheck, ...]
    pending: tuple[PullRequestCheck, ...]
    failing: tuple[PullRequestCheck, ...]
    missing_required: tuple[str, ...]
    branch_protection_status: Literal["configured", "not_found", "forbidden", "unavailable"]
    branch_protection_note: str | None = None


PR_METADATA_FIELDS = "number,title,headRefName,baseRefName,headRefOid,url,body"
PR_REVIEW_CONTEXT_FIELDS = f"{PR_METADATA_FIELDS},comments,reviews"
ISSUE_REFERENCE_RE_TEMPLATE = r"(?:#%d\b|/issues/%d\b)"


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
    )


def _author_login(raw: object) -> str | None:
    if isinstance(raw, dict):
        login = raw.get("login")
        return str(login) if login else None
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

    if check_runs_result.returncode == 0:
        try:
            payload = json.loads(check_runs_result.stdout or "{}")
        except json.JSONDecodeError:
            check_errors.append("check-runs response was not valid JSON")
        else:
            for raw_check in payload.get("check_runs") or []:
                if not isinstance(raw_check, dict):
                    continue
                name = _optional_str(raw_check.get("name"))
                if not name:
                    continue
                checks.append(
                    PullRequestCheck(
                        name=name,
                        kind="check_run",
                        status=_normalize_check_run_status(raw_check),
                        url=_optional_str(raw_check.get("html_url") or raw_check.get("details_url")),
                    )
                )
    else:
        check_errors.append("check-runs query failed")

    if statuses_result.returncode == 0:
        try:
            payload = json.loads(statuses_result.stdout or "{}")
        except json.JSONDecodeError:
            check_errors.append("commit-status response was not valid JSON")
        else:
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
                    )
                )
    else:
        check_errors.append("commit-status query failed")

    deduped_checks = _dedupe_checks(checks)
    passing = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "passing")
    pending = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "pending")
    failing = tuple(check for check in deduped_checks if _classify_check_status(check.status) == "failing")
    observed_names = {check.name for check in deduped_checks}
    missing_required = tuple(name for name in required_checks if name not in observed_names)

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
    log(config, f"Posting agent output to PR #{pr_number}")
    if config.dry_run:
        runner.run(
            [config.gh_cmd, "pr", "comment", str(pr_number), "--repo", config.repo, "--body", body],
            cwd=active_workdir(config),
        )
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        runner.run(
            [
                config.gh_cmd,
                "pr",
                "comment",
                str(pr_number),
                "--repo",
                config.repo,
                "--body-file",
                path,
            ],
            cwd=active_workdir(config),
        )
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def post_issue_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: str,
) -> None:
    log(config, f"Posting agent output to issue #{issue_number}")
    if config.dry_run:
        runner.run(
            [
                config.gh_cmd,
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                config.repo,
                "--body",
                body,
            ],
            cwd=active_workdir(config),
        )
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        runner.run(
            [
                config.gh_cmd,
                "issue",
                "comment",
                str(issue_number),
                "--repo",
                config.repo,
                "--body-file",
                path,
            ],
            cwd=active_workdir(config),
        )
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


def get_check_status(runner: Runner, config: AgentLoopConfig, head_sha: str) -> str:
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{head_sha}/check-runs",
            "--jq",
            (
                f"[.check_runs[] | select(.name == {json.dumps(config.ci_check_name)})] | "
                'if length == 0 then "pending" else .[0].conclusion // .[0].status end'
            ),
        ],
        cwd=active_workdir(config),
    )
    return result.stdout.strip() or "pending"


def wait_for_ci(runner: Runner, config: AgentLoopConfig, pr_number: int) -> None:
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
        status = get_check_status(runner, config, head_sha)
        log(config, f"GitHub check '{config.ci_check_name}' status: {status}")
        if status == "success":
            return
        if status in terminal_failures:
            raise AgentLoopError(f"CI check '{config.ci_check_name}' failed with status: {status}")
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AgentLoopError(
        f"CI check '{config.ci_check_name}' did not pass within {config.ci_timeout_seconds}s"
    )


def merge_pr(runner: Runner, config: AgentLoopConfig, pr_number: int) -> None:
    log(config, f"Merging PR #{pr_number}")
    runner.run(
        [config.gh_cmd, "pr", "merge", str(pr_number), "--repo", config.repo, "--merge"],
        cwd=active_workdir(config),
    )
