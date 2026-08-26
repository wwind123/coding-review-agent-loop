"""GitHub CLI operations used by the orchestrator."""

from __future__ import annotations

import base64
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
from .pr_contract import (
    PR_EXPECTED_CLOSING_MARKER,
    PR_EXPECTED_CLOSING_MARKER_RE,
    decode_pr_contract,
    encode_pr_contract,
)
from .logging import log
from .issue_pr_provenance import (
    IssuePrProvenanceScope,
    compare_issue_pr_provenance,
    parse_issue_pr_provenance_messages,
)
from .round_transport import MAX_GITHUB_BODY_CHARS, prepare_round_comment
from .protocol import parse_signed_human_requirement_body
from .protocol_markers import (
    ISSUE_BODY_SURFACE,
    ISSUE_COMMENT_SURFACE,
    PR_COMMENT_SURFACE,
    TrustedBody,
)
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

# This intentionally remains looser than the recovery parser below.  It is
# used by PR review context inference, where a same-repository issue URL is
# useful even when it is only quoted as background context.
_GITHUB_ISSUE_URL_RE = re.compile(
    r"https?://github\.com/(?P<repo>[^/\s#]+/[^/\s#]+)/issues/(?P<number>[1-9]\d*)(?![\w/-])",
    re.IGNORECASE,
)

# GitHub's supported auto-close grammar.  Recovery must be anchored to one of
# these keywords; a bare issue number, Refs sentence, URL, title, or branch
# name is not implementation provenance.
_CLOSING_ISSUE_REFERENCE_RE = re.compile(
    r"\b(?P<keyword>close|closes|closed|fix|fixes|fixed|resolve|resolves|resolved)\b"
    r"[ \t]*:?[ \t]*(?:"
    r"(?P<unqualified>#[1-9]\d*)|"
    r"(?P<qualified>[^\s/#]+/[^\s/#]+#[1-9]\d*(?![\w/-]))|"
    r"(?P<url>https?://github\.com/[^/\s#]+/[^/\s#]+/issues/[1-9]\d*(?![\w/-]))"
    r")",
    re.IGNORECASE,
)

# Non-closing references are intentionally parsed by the same target grammar,
# but are never returned as strong recovery evidence.  This narrow form is
# used only when a caller explicitly says it is validating a staged parent.
_NON_CLOSING_ISSUE_REFERENCE_RE = re.compile(
    r"\b(?P<keyword>refs?|references?)\b[ \t]*:?[ \t]*(?:"
    r"(?P<unqualified>#[1-9]\d*)|"
    r"(?P<qualified>[^\s/#]+/[^\s/#]+#[1-9]\d*(?![\w/-]))|"
    r"(?P<url>https?://github\.com/[^/\s#]+/[^/\s#]+/issues/[1-9]\d*(?![\w/-]))"
    r")",
    re.IGNORECASE,
)

_AGENT_ISSUE_PR_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*AGENT_ISSUE_PR_HANDOFF:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)


def affirmative_markdown_view(body: str | None) -> str:
    """Return active Markdown text that can count as affirmative issue evidence.

    GitHub does not interpret code samples, inline code, or HTML comments as
    closing instructions. List items are classified before indentation so a
    nested list item remains active evidence, and blockquotes remain active
    because GitHub linkifies their references.
    """
    if not body:
        return ""
    text = re.sub(r"<!--.*?-->", "", body, flags=re.DOTALL)
    output: list[str] = []
    fenced = False
    fence_char = ""
    fence_length = 0
    list_line_re = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
    for line in text.splitlines():
        stripped = line.lstrip(" \t")
        fence = re.match(r"(`{3,}|~{3,})", stripped)
        if fence:
            token = fence.group(1)
            if not fenced:
                fenced = True
                fence_char = token[0]
                fence_length = len(token)
            elif (
                token[0] == fence_char
                and len(token) >= fence_length
                and not stripped[len(token) :].strip()
            ):
                fenced = False
            continue
        if fenced:
            continue
        is_list_item = bool(list_line_re.match(line))
        if (line.startswith("    ") or line.startswith("\t")) and not is_list_item:
            continue
        # Inline code spans are non-rendered code, even when they are inside a
        # list item or blockquote. Preserve surrounding prose and references.
        line = re.sub(r"(`+)(.+?)\1", "", line)
        output.append(line)
    return "\n".join(output)


@dataclass(frozen=True)
class IssueReferenceEvidence:
    """One parsed issue reference, retaining enough detail for diagnostics."""

    keyword: str
    target_repo: str
    issue_number: int
    reference_form: Literal["unqualified", "qualified", "url"]
    matched_text: str
    closing: bool

    @property
    def is_closing(self) -> bool:
        return self.closing


class OpenPrClosingMatch(int):
    """An open PR number with its strong closing-reference evidence.

    This is an ``int`` subclass for compatibility with callers that compared
    the old discovery result directly to a PR number, while exposing the
    structured evidence required by the safer resolver.
    """

    def __new__(cls, pr_number: int, evidence: tuple[IssueReferenceEvidence, ...]):
        instance = int.__new__(cls, pr_number)
        instance.evidence = evidence
        return instance

    @property
    def pr_number(self) -> int:
        return int(self)


@dataclass(frozen=True)
class PullRequestCommitMetadata:
    """One commit in the provider-reported PR base-to-head connection."""

    oid: str
    message: str


def _graphql_repository_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise AgentLoopError(f"Repository {repository!r} is not an owner/repository identity.")
    return parts[0], parts[1]


_PR_COMMIT_CONNECTION_QUERY = """
query($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      headRefOid
      commits(first: 100, after: $after) {
        totalCount
        pageInfo { hasNextPage endCursor }
        nodes {
          commit { oid message }
        }
      }
    }
  }
}
""".strip()


def _query_pr_commit_connection(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    after: str | None,
) -> object:
    owner, name = _graphql_repository_parts(config.repo)
    args = [
        config.gh_cmd,
        "api",
        "graphql",
        "-f",
        f"query={_PR_COMMIT_CONNECTION_QUERY}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={name}",
        "-F",
        f"number={pr_number}",
    ]
    if after is not None:
        args.extend(("-f", f"after={after}"))
    result = runner.run(args, cwd=active_workdir(config), check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AgentLoopError(
            f"GitHub PR commit provenance query failed for PR #{pr_number}"
            + (f": {detail}" if detail else ".")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"GitHub returned malformed PR commit provenance JSON for PR #{pr_number}."
        ) from exc
    if not isinstance(payload, dict) or payload.get("errors"):
        raise AgentLoopError(f"GitHub returned an error for PR #{pr_number} commit provenance.")
    return payload


def _parse_pr_commit_connection_page(
    payload: object,
    *,
    pr_number: int,
) -> tuple[str, int, tuple[PullRequestCommitMetadata, ...], bool, str | None]:
    if not isinstance(payload, dict):
        raise AgentLoopError(f"GitHub returned a malformed commit page for PR #{pr_number}.")
    data = payload.get("data")
    repository = data.get("repository") if isinstance(data, dict) else None
    pull_request = repository.get("pullRequest") if isinstance(repository, dict) else None
    connection = pull_request.get("commits") if isinstance(pull_request, dict) else None
    head_oid = pull_request.get("headRefOid") if isinstance(pull_request, dict) else None
    if not isinstance(head_oid, str) or not head_oid:
        raise AgentLoopError(f"GitHub returned no current head OID for PR #{pr_number}.")
    if not isinstance(connection, dict):
        raise AgentLoopError(f"GitHub returned no complete commit connection for PR #{pr_number}.")
    total = connection.get("totalCount")
    page_info = connection.get("pageInfo")
    nodes = connection.get("nodes")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise AgentLoopError(f"GitHub returned an invalid commit total for PR #{pr_number}.")
    if not isinstance(page_info, dict):
        raise AgentLoopError(f"GitHub returned malformed commit pagination for PR #{pr_number}.")
    has_next = page_info.get("hasNextPage")
    end_cursor = page_info.get("endCursor")
    if not isinstance(has_next, bool) or (has_next and (not isinstance(end_cursor, str) or not end_cursor)):
        raise AgentLoopError(f"GitHub returned malformed commit pagination for PR #{pr_number}.")
    if end_cursor is not None and not isinstance(end_cursor, str):
        raise AgentLoopError(f"GitHub returned malformed commit pagination for PR #{pr_number}.")
    if not isinstance(nodes, list):
        raise AgentLoopError(f"GitHub returned malformed commit nodes for PR #{pr_number}.")
    commits: list[PullRequestCommitMetadata] = []
    for node in nodes:
        if not isinstance(node, dict):
            raise AgentLoopError(f"GitHub returned an incomplete commit node for PR #{pr_number}.")
        commit_node = node.get("commit") if isinstance(node.get("commit"), dict) else node
        oid = commit_node.get("oid") if isinstance(commit_node, dict) else None
        message = commit_node.get("message") if isinstance(commit_node, dict) else None
        if not isinstance(oid, str) or not oid or not isinstance(message, str):
            raise AgentLoopError(f"GitHub returned an incomplete commit node for PR #{pr_number}.")
        commits.append(PullRequestCommitMetadata(oid=oid, message=message))
    return head_oid, total, tuple(commits), has_next, end_cursor


def read_pull_request_commit_metadata(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
) -> tuple[PullRequestCommitMetadata, ...]:
    """Read the complete stable commit history for one PR.

    The connection is paginated independently of the provider's default page
    cap.  Both the head OID and provider-reported total are sampled again
    after traversal so a concurrent push or truncated response is unavailable
    for recovery rather than being mistaken for a clean provenance miss.
    """
    if config.dry_run:
        return ()
    first_payload = _query_pr_commit_connection(
        runner, config=config, pr_number=pr_number, after=None
    )
    before_head, expected_total, first_commits, has_next, cursor = _parse_pr_commit_connection_page(
        first_payload, pr_number=pr_number
    )
    commits = list(first_commits)
    seen_oids = {commit.oid for commit in commits}
    seen_cursors: set[str] = set()
    while has_next:
        assert cursor is not None
        if cursor in seen_cursors:
            raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} did not advance.")
        seen_cursors.add(cursor)
        payload = _query_pr_commit_connection(
            runner, config=config, pr_number=pr_number, after=cursor
        )
        head, total, page_commits, has_next, next_cursor = _parse_pr_commit_connection_page(
            payload, pr_number=pr_number
        )
        if head != before_head or total != expected_total:
            raise AgentLoopError(f"PR #{pr_number} commit history changed during provenance scan.")
        if not page_commits and has_next:
            raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} returned an empty advancing page.")
        for commit in page_commits:
            if commit.oid in seen_oids:
                raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} repeated a commit.")
            seen_oids.add(commit.oid)
            commits.append(commit)
        if has_next and next_cursor == cursor:
            raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} did not advance.")
        cursor = next_cursor
    if len(commits) != expected_total:
        raise AgentLoopError(
            f"GitHub PR #{pr_number} commit provenance was truncated: provider reported "
            f"{expected_total} commits but returned {len(commits)}."
        )
    final_payload = _query_pr_commit_connection(
        runner, config=config, pr_number=pr_number, after=None
    )
    final_head, final_total, _ignored, _ignored_next, _ignored_cursor = _parse_pr_commit_connection_page(
        final_payload, pr_number=pr_number
    )
    if final_head != before_head or final_total != expected_total:
        raise AgentLoopError(f"PR #{pr_number} commit history changed during provenance scan.")
    return tuple(commits)


get_pull_request_commit_metadata = read_pull_request_commit_metadata


def validate_pull_request_provenance(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_scope: IssuePrProvenanceScope,
) -> IssuePrProvenanceScope:
    try:
        commits = read_pull_request_commit_metadata(runner, config=config, pr_number=pr_number)
    except AgentLoopError as exc:
        raise AgentLoopError(f"PR #{pr_number} commit provenance is unavailable: {exc}") from exc
    try:
        claims = parse_issue_pr_provenance_messages(commit.message for commit in commits)
    except AgentLoopError as exc:
        raise AgentLoopError(f"PR #{pr_number} commit provenance is malformed: {exc}") from exc
    try:
        return compare_issue_pr_provenance(claims, expected=expected_scope)
    except AgentLoopError as exc:
        if not claims:
            reason = "is missing"
        elif len(set(claims)) != 1:
            reason = "contains conflicting claims"
        else:
            reason = "does not match the expected scope"
        raise AgentLoopError(f"PR #{pr_number} commit provenance {reason}: {exc}") from exc


def _issue_reference_evidence_from_match(
    match: re.Match[str], *, closing: bool, default_repo: str
) -> IssueReferenceEvidence:
    reference = match.group("unqualified") or match.group("qualified") or match.group("url")
    if reference is None:  # pragma: no cover - every grammar branch has a target
        raise AgentLoopError("Issue reference parser produced an empty target.")
    if match.group("unqualified"):
        target_repo = default_repo
        reference_form: Literal["unqualified", "qualified", "url"] = "unqualified"
        issue_number = int(reference[1:])
    elif match.group("qualified"):
        target_repo, raw_number = reference.rsplit("#", 1)
        reference_form = "qualified"
        issue_number = int(raw_number)
    else:
        url_match = _GITHUB_ISSUE_URL_RE.fullmatch(reference)
        if url_match is None:  # pragma: no cover - guarded by the shared grammar
            raise AgentLoopError("Issue reference parser produced an invalid URL target.")
        target_repo = url_match.group("repo")
        reference_form = "url"
        issue_number = int(url_match.group("number"))
    return IssueReferenceEvidence(
        keyword=match.group("keyword").casefold(),
        target_repo=target_repo.casefold(),
        issue_number=issue_number,
        reference_form=reference_form,
        matched_text=match.group(0),
        closing=closing,
    )


def parse_issue_reference_evidence(
    body: str | None,
    *,
    repo: str,
    include_non_closing: bool = True,
    affirmative: bool = True,
) -> tuple[IssueReferenceEvidence, ...]:
    """Parse closing and, optionally, explicit ``Refs`` issue evidence.

    The parser deliberately returns cross-repository matches too, with their
    normalized target repository, so callers can explain why a candidate was
    rejected.  Use :func:`parse_strong_issue_reference_evidence` for recovery.
    """
    if not body:
        return ()
    searchable_body = affirmative_markdown_view(body) if affirmative else body
    matches: list[tuple[int, IssueReferenceEvidence]] = [
        (
            match.start(),
            _issue_reference_evidence_from_match(match, closing=True, default_repo=repo),
        )
        for match in _CLOSING_ISSUE_REFERENCE_RE.finditer(searchable_body)
    ]
    if include_non_closing:
        matches.extend(
            (
                match.start(),
                _issue_reference_evidence_from_match(match, closing=False, default_repo=repo),
            )
            for match in _NON_CLOSING_ISSUE_REFERENCE_RE.finditer(searchable_body)
        )
    return tuple(evidence for _position, evidence in sorted(matches, key=lambda item: item[0]))


def parse_strong_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return only same-repository closing evidence for ``issue_number``."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(body, repo=repo, include_non_closing=False)
        if evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
    )


def parse_raw_strong_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return raw-body closing evidence for fail-closed safety prohibitions."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(
            body, repo=repo, include_non_closing=False, affirmative=False
        )
        if evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
    )


def parse_non_closing_issue_reference_evidence(
    body: str | None, *, repo: str, issue_number: int
) -> tuple[IssueReferenceEvidence, ...]:
    """Return explicit same-repository ``Refs`` evidence for a staged role."""
    normalized_repo = repo.casefold()
    return tuple(
        evidence
        for evidence in parse_issue_reference_evidence(body, repo=repo)
        if not evidence.closing
        and evidence.target_repo.casefold() == normalized_repo
        and evidence.issue_number == issue_number
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
    staged_parent_issue: int | None = None,
    body: str | None = None,
) -> None:
    if config.dry_run:
        return
    body = _get_pr_body(runner, config=config, pr_number=pr_number) if body is None else body
    strong_evidence = parse_strong_issue_reference_evidence(
        body, repo=config.repo, issue_number=issue_number
    )
    if strong_evidence:
        if staged_parent_issue is not None:
            _validate_staged_parent_reference(
                body, config=config, pr_number=pr_number, parent_issue=staged_parent_issue
            )
        return
    raise AgentLoopError(
        f"PR #{pr_number} does not reference issue #{issue_number} with strong closing evidence. "
        f"Edit the PR description on GitHub to include `Fixes #{issue_number}`, `Closes "
        f"#{issue_number}`, or `Resolves #{issue_number}`, then rerun the orchestrator as "
        f"`agent-loop pr {pr_number}` to continue the review. Bare `#{issue_number}`, `Refs`, "
        "issue URLs used as context, and branch names are not implementation evidence."
    )


def missing_expected_closing_issue_ids(
    body: str | None,
    *,
    repo: str,
    expected_issue_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return expected IDs without their own affirmative closing pair."""
    observed = {
        evidence.issue_number
        for evidence in parse_issue_reference_evidence(
            body, repo=repo, include_non_closing=False, affirmative=True
        )
        if evidence.closing and evidence.target_repo.casefold() == repo.casefold()
    }
    return tuple(sorted(set(expected_issue_ids) - observed))


def validate_pr_expected_closing_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_issue_ids: Sequence[int],
    body: str | None = None,
) -> tuple[int, ...]:
    """Validate a known contract against one freshly fetched PR body."""
    if config.dry_run:
        return ()
    current_body = _get_pr_body(runner, config=config, pr_number=pr_number) if body is None else body
    missing = missing_expected_closing_issue_ids(
        current_body, repo=config.repo, expected_issue_ids=expected_issue_ids
    )
    if missing:
        rendered = ", ".join(f"#{issue}" for issue in missing)
        expected = ", ".join(f"#{issue}" for issue in sorted(set(expected_issue_ids))) or "(none)"
        raise AgentLoopError(
            f"PR #{pr_number} is missing affirmative closing references for expected issue(s): {rendered}. "
            f"The immutable expected set is {{{expected}}}. Edit the existing PR description so every "
            "listed issue has its own `Closes`, `Fixes`, or `Resolves` keyword/reference pair, then "
            f"resume with `agent-loop pr {pr_number}`; do not create another PR."
        )
    return missing


def _get_pr_body(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> str:
    if config.dry_run:
        return ""
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
    return _optional_str(data.get("body")) or ""


def _validate_staged_parent_reference(
    body: str, *, config: AgentLoopConfig, pr_number: int, parent_issue: int
) -> None:
    if parse_raw_strong_issue_reference_evidence(
        body, repo=config.repo, issue_number=parent_issue
    ):
        raise AgentLoopError(
            f"PR #{pr_number} body uses a closing keyword against staged parent issue "
            f"#{parent_issue}; use an explicit non-closing `Refs #{parent_issue}` reference."
        )
    if not parse_non_closing_issue_reference_evidence(
        body, repo=config.repo, issue_number=parent_issue
    ):
        raise AgentLoopError(
            f"PR #{pr_number} must include an explicit non-closing `Refs #{parent_issue}` "
            "reference when implementing a staged child."
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
    body = _get_pr_body(runner, config=config, pr_number=pr_number)
    if not parse_raw_strong_issue_reference_evidence(
        body, repo=config.repo, issue_number=issue_number
    ):
        return
    raise AgentLoopError(
        f"PR #{pr_number} body uses a closing keyword (Closes/Fixes/Resolves) against parent "
        f"issue #{issue_number}, but other split stages remain unfiled or unimplemented. Edit the "
        f"PR description to use a non-closing reference (e.g. `Refs #{issue_number}`) instead, then "
        f"rerun `agent-loop pr {pr_number}` to continue the review."
    )


_UNSET_PROVENANCE_SCOPE = object()


def find_open_pr_closing_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    expected_scope: IssuePrProvenanceScope | None | object = _UNSET_PROVENANCE_SCOPE,
) -> OpenPrClosingMatch | None:
    """Find the unique open PR with strong closing evidence for an issue.

    This is the metadata-free crash-window recovery path.  A closing keyword
    tied to the configured repository and issue is required; incidental prose,
    bare issue references, and contextual URLs are deliberately ignored.
    """
    if config.dry_run:
        return None
    if expected_scope is _UNSET_PROVENANCE_SCOPE:
        # Preserve the direct helper's historical default while resolver call
        # sites pass an explicit scope (including ``None`` when a plan cannot
        # be reconstructed).
        expected_scope = IssuePrProvenanceScope(
            repository=config.repo, issue_number=issue_number, flow="direct"
        )
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
            # `gh pr list` follows GitHub pagination up to this explicit limit.
            # The former 100-item cap could silently miss the only candidate.
            "--limit",
            "100000",
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
    matches: list[OpenPrClosingMatch] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        body = _optional_str(item.get("body")) or ""
        evidence = parse_strong_issue_reference_evidence(
            body, repo=config.repo, issue_number=issue_number
        )
        if not evidence:
            continue
        number = item.get("number")
        if isinstance(number, int):
            matches.append(OpenPrClosingMatch(number, evidence))
    if not matches:
        return None
    if len(matches) > 1:
        def format_match(match: OpenPrClosingMatch) -> str:
            evidence_text = "; ".join(
                f"{evidence.keyword}: {evidence.matched_text}"
                for evidence in match.evidence
            )
            return f"#{match.pr_number} ({evidence_text})"

        sorted_matches = sorted(matches, key=lambda item: item.pr_number)
        numbers = ", ".join(f"#{match.pr_number}" for match in sorted_matches)
        joined = ", ".join(format_match(match) for match in sorted_matches)
        raise AgentLoopError(
            f"Multiple open PRs ({numbers}) "
            f"with strong closing evidence for issue #{issue_number}: {joined}; "
            "cannot automatically determine which to resume. Remove the accidental closing "
            "reference or close the unrelated PR(s), then rerun `agent-loop pr <number>` directly "
            "to continue review on the correct one."
        )
    match = matches[0]
    if expected_scope is None:
        raise _candidate_provenance_error(
            match.pr_number,
            "the issue-mode plan has no reconstructable approved-plan scope.",
        )
    try:
        validate_pull_request_provenance(
            runner,
            config=config,
            pr_number=match.pr_number,
            expected_scope=expected_scope,
        )
    except AgentLoopError as exc:
        raise _candidate_provenance_error(match.pr_number, str(exc)) from exc
    return match


def _candidate_provenance_error(pr_number: int, reason: str) -> AgentLoopError:
    return AgentLoopError(
        f"Open PR #{pr_number} is the sole closing-reference candidate, but it cannot be "
        f"safely adopted: {reason} Remove the closing reference from this unrelated PR or "
        f"close the unrelated PR; if PR #{pr_number} is the intended implementation, resume "
        f"it directly with `agent-loop pr {pr_number}`."
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
        login = raw.get("login") or raw.get("slug")
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
    body: str | TrustedBody,
) -> None:
    carrier = body if isinstance(body, TrustedBody) else TrustedBody.current_untrusted_visible(body)
    bodies = prepare_round_comment(carrier)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to PR #{pr_number}")
    log(config, f"Posting agent output to PR #{pr_number}")
    for prepared in bodies:
        prepared.validate_for_surface(PR_COMMENT_SURFACE)
        _post_comment_body(runner, config=config, command=["pr", "comment", str(pr_number)], body=prepared)


def post_issue_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: str | TrustedBody,
) -> None:
    carrier = body if isinstance(body, TrustedBody) else TrustedBody.current_untrusted_visible(body)
    bodies = prepare_round_comment(carrier)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to issue #{issue_number}")
    log(config, f"Posting agent output to issue #{issue_number}")
    for prepared in bodies:
        prepared.validate_for_surface(ISSUE_COMMENT_SURFACE)
        _post_comment_body(runner, config=config, command=["issue", "comment", str(issue_number)], body=prepared)


def reject_forged_protocol_markers(body: str) -> None:
    """Reject reserved records before any temp-file, runner, or remote mutation."""
    TrustedBody.current_untrusted_visible(body)


def _post_trusted_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    command: list[str],
    body: str,
) -> None:
    """Post canonical protocol output after validating its marker encoding."""
    _post_comment_body(runner, config=config, command=command, body=body)


def _validate_canonical_json_marker_payload(
    body: str,
    *,
    marker_re: re.Pattern[str],
    marker_name: str,
) -> None:
    matches = tuple(marker_re.finditer(body))
    if not matches:
        raise AgentLoopError(
            f"Trusted {marker_name} posting requires its canonical protocol marker."
        )
    for match in matches:
        encoded = match.group("payload")
        try:
            value = json.loads(
                base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8")
            )
        except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
            raise AgentLoopError(f"Trusted {marker_name} payload is not valid JSON.") from exc
        canonical = base64.urlsafe_b64encode(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).decode("ascii")
        if canonical != encoded:
            raise AgentLoopError(f"Trusted {marker_name} payload is not canonically encoded.")


def post_trusted_pr_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: TrustedBody,
) -> None:
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Trusted PR comment posting requires a TrustedBody.")
    bodies = prepare_round_comment(body)
    for prepared in bodies:
        prepared.validate_for_surface(PR_COMMENT_SURFACE)
        _post_trusted_protocol_comment(
            runner, config=config, command=["pr", "comment", str(pr_number)], body=prepared
        )


def post_trusted_pr_contract_record(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: TrustedBody,
) -> None:
    """Create a canonical PR contract record before issue-origin handoff."""
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Trusted PR contract posting requires a TrustedBody.")
    body.validate_for_surface(PR_COMMENT_SURFACE)
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/issues/{pr_number}/comments",
            "--method",
            "POST",
            "--input",
            "-",
        ],
        cwd=active_workdir(config),
        input_text=json.dumps({"body": str(body)}),
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AgentLoopError(
            f"Unable to persist the expected-closing PR contract for PR #{pr_number}."
            + (f" {detail}" if detail else "")
        )


def post_trusted_issue_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: TrustedBody,
) -> None:
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Trusted issue comment posting requires a TrustedBody.")
    bodies = prepare_round_comment(body)
    for prepared in bodies:
        prepared.validate_for_surface(ISSUE_COMMENT_SURFACE)
        _post_trusted_protocol_comment(
            runner, config=config, command=["issue", "comment", str(issue_number)], body=prepared
        )


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
    body: str | TrustedBody,
) -> str | None:
    if isinstance(body, TrustedBody):
        body.validate_for_surface(ISSUE_BODY_SURFACE)
        rendered_body = str(body)
    else:
        rendered_body = str(TrustedBody.current_untrusted_visible(body))
    if len(rendered_body) > MAX_GITHUB_BODY_CHARS:
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
                rendered_body,
            ],
            cwd=active_workdir(config),
        )
        issue_url = result.stdout.strip() or None
        if issue_url:
            log(config, f"Created GitHub issue: {issue_url}")
        return issue_url

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(rendered_body)
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
