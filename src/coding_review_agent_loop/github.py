"""GitHub CLI operations used by the orchestrator."""

from __future__ import annotations

import base64
import datetime
import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Sequence
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
from .round_transport import (
    MAX_GITHUB_BODY_CHARS,
    prepare_round_comment,
)
from .protocol import parse_signed_human_requirement_body
from .protocol_markers import (
    ISSUE_BODY_SURFACE,
    ISSUE_COMMENT_SURFACE,
    PR_COMMENT_SURFACE,
    TrustedBody,
    named_reserved_marker_tokens,
    record_shaped_untrusted_markers,
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
    comment_id: int | None = None
    author_id: int | None = None
    # Presentation-only permalink (GraphQL ``url`` / REST ``html_url``).  It
    # is excluded from equality so it never participates in record identity.
    url: str | None = field(default=None, compare=False)

    @property
    def id(self) -> int | None:
        return self.comment_id


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

    @property
    def requirement_id(self) -> str:
        """Stable identity for this exact signed instruction.

        The ID intentionally includes the signed body as well as source
        metadata.  A comment edit therefore creates a new requirement rather
        than silently inheriting an acknowledgement for the old meaning.
        """
        return human_requirement_id(self)

    @property
    def canonical_key(self) -> str:
        return canonical_human_requirement_key(self)

    # ``id`` is a convenient compatibility alias for callers that model the
    # requirement as an identified record.
    @property
    def id(self) -> str:
        return self.requirement_id


@dataclass(frozen=True)
class PullRequestReviewContext:
    metadata: PullRequestMetadata
    comments: tuple[IssueComment, ...]
    human_requirements: tuple[HumanReviewRequirement, ...]
    # Set by the orchestration qualification read when immutable architecture
    # context changed. It is a scheduling signal, not GitHub-supplied state.
    architecture_identity_changed: bool = False


@dataclass(frozen=True)
class PullRequestMergeability:
    state: Literal["mergeable", "conflicted", "unknown"]
    mergeable_raw: str | None
    merge_state_raw: str | None
    head_sha: str | None
    base_branch: str | None


PR_METADATA_FIELDS = "number,title,headRefName,baseRefName,headRefOid,url,body"
PR_REVIEW_CONTEXT_FIELDS = f"{PR_METADATA_FIELDS},comments,reviews"
# Recovery searches must not inherit gh's small default page size: a parent
# may have hundreds of unrelated issues before its child topology is found.
ISSUE_RECOVERY_SEARCH_LIMIT = 100_000

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
    data = _load_json_object(result, description=f"the state of PR #{pr_number}")
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


def unexpected_closing_issue_ids(
    body: str | None,
    *,
    repo: str,
    expected_issue_ids: Sequence[int],
) -> tuple[int, ...]:
    """Return same-repository closing IDs outside the durable contract."""
    observed = {
        evidence.issue_number
        for evidence in parse_issue_reference_evidence(
            body, repo=repo, include_non_closing=False, affirmative=True
        )
        if evidence.closing and evidence.target_repo.casefold() == repo.casefold()
    }
    return tuple(sorted(observed - set(expected_issue_ids)))


def validate_pr_expected_closing_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_issue_ids: Sequence[int],
    body: str | None = None,
    reject_unexpected: bool = False,
) -> tuple[int, ...]:
    """Validate a known contract against one freshly fetched PR body.

    Direct PR mode retains its historical subset-then-supersede behavior. The
    managed issue recovery seam can opt into exact validation so an existing
    PR cannot use an unapproved closing reference as recovery provenance.
    """
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
    if reject_unexpected:
        unexpected = unexpected_closing_issue_ids(
            current_body, repo=config.repo, expected_issue_ids=expected_issue_ids
        )
        if unexpected:
            rendered = ", ".join(f"#{issue}" for issue in unexpected)
            expected = ", ".join(f"#{issue}" for issue in sorted(set(expected_issue_ids))) or "(none)"
            raise AgentLoopError(
                f"PR #{pr_number} has affirmative closing references outside the expected contract: "
                f"{rendered}. The immutable expected set is {{{expected}}}. Remove each unapproved "
                "`Closes`, `Fixes`, or `Resolves` reference from the existing PR description, then "
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
    if (
        isinstance(raw, dict)
        and isinstance(raw.get("id"), int)
        and not isinstance(raw.get("id"), bool)
        and raw["id"] > 0
    ):
        return raw["id"]
    return None


def _normalize_requirement_body(body: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in body.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def _normalize_requirement_field(value: str | None) -> str:
    return " ".join((value or "").replace("\r", "").split())


def canonical_human_requirement_key(requirement: HumanReviewRequirement) -> str:
    """Return the canonical content used to identify one signed instruction."""
    fields = (
        _normalize_requirement_field(requirement.source_type).casefold(),
        _normalize_requirement_field(requirement.url),
        _normalize_requirement_field(requirement.author).casefold(),
        _normalize_requirement_field(requirement.created_at),
        _normalize_requirement_body(requirement.body),
    )
    return "\x1f".join(fields)


def human_requirement_id(requirement: HumanReviewRequirement) -> str:
    """Return the full digest-backed stable ID for a signed requirement."""
    return "hr-" + hashlib.sha256(canonical_human_requirement_key(requirement).encode("utf-8")).hexdigest()


def human_requirement_deduplication_key(requirement: HumanReviewRequirement) -> str:
    """Return a source-aware key used when the same signed record is surfaced twice.

    GitHub can expose one comment through more than one API surface.  A URL is
    the strongest cross-surface identity; body/source metadata is retained for
    older fixtures and providers that omit comment URLs.
    """
    body = _normalize_requirement_body(requirement.body)
    if requirement.url:
        return "url\x1f" + _normalize_requirement_field(requirement.url).casefold() + "\x1f" + body
    return "record\x1f" + canonical_human_requirement_key(requirement)


def deduplicate_human_requirements(
    requirements: Sequence[HumanReviewRequirement],
) -> tuple[HumanReviewRequirement, ...]:
    """Deduplicate identical signed records without changing their IDs."""
    found: dict[str, HumanReviewRequirement] = {}
    identity_content: dict[str, str] = {}
    for requirement in requirements:
        key = human_requirement_deduplication_key(requirement)
        identity = requirement.requirement_id
        canonical = requirement.canonical_key
        previous_canonical = identity_content.get(identity)
        if previous_canonical is not None and previous_canonical != canonical:
            raise AgentLoopError(
                "A stable signed human requirement ID maps to divergent canonical content "
                f"({identity})."
            )
        identity_content[identity] = canonical
        previous = found.get(key)
        if previous is None:
            found[key] = requirement
            continue
        # The same URL/body can be surfaced by issue and PR API payloads with
        # slightly different metadata. Keep the deterministic earliest record;
        # its body and source locator remain the cross-surface identity.
        if _human_requirement_sort_key(requirement) < _human_requirement_sort_key(previous):
            found[key] = requirement
    return tuple(sorted(found.values(), key=_human_requirement_sort_key))


def _human_requirement_sort_key(requirement: HumanReviewRequirement) -> tuple[float, str, str, str, str]:
    # GitHub normally returns RFC3339 timestamps.  Parse offsets instead of
    # relying on lexical order so chronological precedence is correct across
    # timezone representations.
    raw_created = _normalize_requirement_field(requirement.created_at)
    try:
        parsed = datetime.datetime.fromisoformat(raw_created.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        created_sort = parsed.timestamp()
    except (TypeError, ValueError, OverflowError):
        created_sort = float("inf")
    return (
        created_sort,
        _normalize_requirement_field(requirement.source_type).casefold(),
        _normalize_requirement_field(requirement.url).casefold(),
        _normalize_requirement_field(requirement.author).casefold(),
        requirement.requirement_id,
    )


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
    data = _load_json_object(result, description=f"pull request #{pr_number}")
    comments = _parse_issue_comments(data.get("comments"))
    metadata = _parse_pr_metadata(data, config=config, pr_number=pr_number)
    log_untrusted_marker_neutralization(
        config,
        surface=f"Pull request #{pr_number} text",
        texts=[
            metadata.title,
            metadata.body,
            *(comment.body for comment in comments),
        ],
    )
    return PullRequestReviewContext(
        metadata=metadata,
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
        # GraphQL issue views expose the producer as ``author`` while the
        # REST issue-comment endpoint exposes the same identity as ``user``.
        # Keep one parser for both projections so transport authentication is
        # based on the live REST identity rather than comment ordering.
        author = raw_comment.get("author") or raw_comment.get("user")
        raw_id = raw_comment.get("id")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raw_id = raw_comment.get("databaseId")
        comment_id = (
            raw_id
            if (
                isinstance(raw_id, int)
                and not isinstance(raw_id, bool)
                and raw_id > 0
            )
            else None
        )
        comments.append(
            IssueComment(
                author=_author_login(author),
                created_at=_optional_str(raw_comment.get("createdAt")) or _optional_str(raw_comment.get("created_at")),
                body=_optional_str(raw_comment.get("body")),
                comment_id=comment_id,
                author_id=_author_id(author),
                url=_optional_str(raw_comment.get("url"))
                or _optional_str(raw_comment.get("html_url")),
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

    The full-board `get_pr_checks` query and its watcher both rely on this
    parser; it deliberately does not select a named merge gate.
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


def _is_board_amendment_record(signed_body: str) -> bool:
    # A signed reviewer-board amendment is an orchestration record (#943),
    # not a requirement on the reviewed artifact.
    from .board_amendment import is_reviewer_board_amendment_only

    return is_reviewer_board_amendment_only(signed_body)


def _parse_pr_human_requirements(data: dict[str, object]) -> tuple[HumanReviewRequirement, ...]:
    requirements: list[HumanReviewRequirement] = []
    for raw_comment in data.get("comments") or []:
        if not isinstance(raw_comment, dict):
            continue
        body = parse_signed_human_requirement_body(raw_comment.get("body"))
        if body is None or _is_board_amendment_record(body):
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
    return deduplicate_human_requirements(requirements)


def _load_json_object(result, *, description: str) -> dict:
    """Decode a `gh --json` projection, failing closed on unreadable output.

    `json.loads` raises `JSONDecodeError`, and a non-object payload raises an
    attribute error on the first `.get`.  Neither is an `AgentLoopError`, so a
    caller that wraps GitHub reads with its own context would otherwise let an
    unreadable payload escape unlabelled (#918).
    """
    try:
        data = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"Unable to read {description}: GitHub CLI output is not JSON ({exc})."
        ) from exc
    if not isinstance(data, dict):
        raise AgentLoopError(
            f"Unable to read {description}: GitHub CLI output is not a JSON object."
        )
    return data


def _read_issue_state_projection(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int
) -> dict:
    """Read the shared `{number,state,is_pr,url}` issue projection, fail-closed.

    `validate_open_issue` and `get_issue_state` both authenticate a live issue
    state through this single reader so the two cannot drift: a nonzero `gh`
    exit, a non-object payload, an absent or non-string `state`, a number
    mismatch, or a payload that resolves to a pull request is an error rather
    than an implicitly open or closed issue.
    """
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/issues/{issue_number}",
            "--jq",
            "{number:.number,state:.state,is_pr:has(\"pull_request\"),url:.html_url}",
        ],
        cwd=active_workdir(config),
        # `check=False` so a nonzero `gh` exit reaches the contextual
        # diagnostic below instead of the Runner's generic command failure.
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Unable to read issue #{issue_number} from {config.repo}: "
            f"`gh` exited {result.returncode}."
        )
    data = _load_json_object(
        result, description=f"issue #{issue_number} from {config.repo}"
    )
    if data.get("is_pr"):
        raise AgentLoopError(
            f"#{issue_number} is a pull request, not an issue. Use `agent-loop pr {issue_number}`."
        )
    number = data.get("number")
    if number is not None and number != issue_number:
        raise AgentLoopError(
            f"GitHub returned issue #{number} for requested issue #{issue_number}."
        )
    state = data.get("state")
    if not isinstance(state, str) or not state.strip():
        raise AgentLoopError(
            f"Unable to determine the state of issue #{issue_number} in {config.repo}."
        )
    return data


def get_issue_state(runner: Runner, *, config: AgentLoopConfig, issue_number: int) -> str:
    """Return the live issue state normalized to `OPEN` or `CLOSED`.

    Any other value is an error: a staged parent authenticates phase
    completion from this state, so an unrecognized value must stop the run
    rather than be interpreted as either open or closed.
    """
    data = _read_issue_state_projection(runner, config=config, issue_number=issue_number)
    state = str(data["state"]).strip().upper()
    if state not in {"OPEN", "CLOSED"}:
        raise AgentLoopError(
            f"Issue #{issue_number} in {config.repo} reported unexpected state "
            f"{data['state']!r}; expected `open` or `closed`."
        )
    return state


def validate_open_issue(runner: Runner, *, config: AgentLoopConfig, issue_number: int) -> None:
    if config.dry_run:
        return
    data = _read_issue_state_projection(runner, config=config, issue_number=issue_number)
    if data.get("state") != "open":
        raise AgentLoopError(
            f"Issue #{issue_number} is {data.get('state', 'not open')}; provide an open issue number."
        )


def _comment_sort_key(comment: IssueComment) -> str:
    return comment.created_at or ""


def _merge_issue_comment_transport_identity(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    comments: tuple[IssueComment, ...],
) -> tuple[IssueComment, ...]:
    """Attach numeric REST identities only when a durable audit marker exists.

    ``gh issue view --comments`` supplies the convenient GraphQL projection,
    but it does not expose the numeric comment/user IDs required by the
    authenticated diagnostic protocol.  The REST endpoint is therefore read
    to completion and matched by immutable visible fields.  An incomplete
    REST read is a fail-closed recovery error; returning shape-only comments
    would make a real durable record silently disappear from resume selection.
    """
    marker_in_projection = any(
        isinstance(comment.body, str)
        and "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.body
        for comment in comments
    )
    # ``gh issue view --comments`` is itself a bounded projection.  A full
    # page without the marker may simply mean that the durable record is on a
    # later REST page, so probe REST at the projection boundary too.
    page_size = 100
    if not marker_in_projection and len(comments) < page_size:
        return comments
    page = 1
    raw_transport_comments: list[object] = []
    seen_ids: set[int] = set()
    # This is only a loop guard.  GitHub's endpoint is finite, but a broken
    # proxy must not turn recovery into an unbounded operation.
    max_pages = 10_000
    while page <= max_pages:
        result = runner.run(
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues/{issue_number}/comments?per_page={page_size}&page={page}",
            ],
            cwd=active_workdir(config),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} is incomplete; "
                "trusted planning diagnostics cannot be resumed safely."
            )
        try:
            raw_page = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} returned malformed JSON."
            ) from exc
        if not isinstance(raw_page, list):
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} returned a non-list page."
            )
        raw_transport_comments.extend(raw_page)
        for raw_comment in raw_page:
            if not isinstance(raw_comment, dict):
                raise AgentLoopError(
                    f"GitHub issue comment recovery for issue #{issue_number} returned an incomplete page."
                )
            comment_id = raw_comment.get("id")
            if not isinstance(comment_id, int) or isinstance(comment_id, bool) or comment_id < 1:
                raise AgentLoopError(
                    f"GitHub issue comment recovery for issue #{issue_number} returned a comment without a numeric ID."
                )
            if comment_id in seen_ids:
                raise AgentLoopError(
                    f"GitHub issue comment recovery for issue #{issue_number} repeated a comment ID."
                )
            seen_ids.add(comment_id)
        if len(raw_page) < page_size:
            break
        page += 1
    else:
        raise AgentLoopError(
            f"GitHub issue comment recovery for issue #{issue_number} exceeded its pagination bound."
        )
    transport_comments = _parse_issue_comments(raw_transport_comments)
    by_key: dict[tuple[str | None, str | None, str | None], list[IssueComment]] = {}
    for comment in transport_comments:
        by_key.setdefault((comment.author, comment.created_at, comment.body), []).append(comment)
    merged: list[IssueComment] = []
    matched_transport_ids: set[int] = set()
    for comment in comments:
        candidates = by_key.get((comment.author, comment.created_at, comment.body), [])
        transport = candidates.pop(0) if candidates else None
        if transport is not None and transport.comment_id is not None:
            matched_transport_ids.add(transport.comment_id)
        if (
            isinstance(comment.body, str)
            and "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.body
            and (
                transport is None
                or transport.comment_id is None
                or transport.author is None
                or transport.author_id is None
                or transport.created_at is None
            )
        ):
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} could not authenticate "
                "a planning diagnostic against the live REST record."
            )
        merged.append(
            replace(
                comment,
                comment_id=(transport.comment_id if transport is not None else comment.comment_id),
                author_id=(transport.author_id if transport is not None else comment.author_id),
            )
        )
    # The GraphQL projection can omit older comments once it reaches its
    # connection cap.  Add only authenticated protocol transport records
    # discovered by REST; ordinary comments remain sourced from the existing
    # projection and are not duplicated into prompt context.  Round anchors
    # and sidecars are needed together: a canonical plan may use sidecars, and
    # the authenticated canonical anchor is what semantically supersedes a
    # diagnostic during later recovery.  Issue-to-PR handoff records are added
    # too: canonical-PR authentication decides between the version-1 path and
    # transaction-era discovery from this snapshot, so a capped projection must
    # never hide a handoff (#827).
    transport_marker_names = (
        "AGENT_PLAN_VALIDATION_DIAGNOSTIC",
        "AGENT_LOOP_META",
        "AGENT_LOOP_SIDECAR",
        "AGENT_ISSUE_PR_HANDOFF",
    )
    for transport in transport_comments:
        if (
            transport.comment_id is not None
            and transport.comment_id not in matched_transport_ids
            and isinstance(transport.body, str)
            and any(marker in transport.body for marker in transport_marker_names)
        ):
            merged.append(transport)
    return tuple(sorted(merged, key=_comment_sort_key))


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
    data = _load_json_object(result, description=f"issue #{issue_number}")
    comments = _merge_issue_comment_transport_identity(
        runner,
        config=config,
        issue_number=issue_number,
        comments=_parse_issue_comments(data.get("comments")),
    )
    body = _optional_str(data.get("body"))
    title = _optional_str(data.get("title"))
    log_untrusted_marker_neutralization(
        config,
        surface=f"Issue #{issue_number} text",
        texts=[title, body, *(comment.body for comment in comments)],
    )
    return IssueContext(
        number=int(data.get("number") or issue_number),
        repo=config.repo,
        title=title,
        body=body,
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
        if body is None or _is_board_amendment_record(body):
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
    return deduplicate_human_requirements(requirements)


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


def reject_forged_protocol_markers(
    body: str, *, surface: str = "pull-request body"
) -> None:
    """Reject forged reserved records in untrusted GitHub text.

    Untrusted text that merely *names* a reserved token is ordinary prose: it
    never carries authority and is rendered defanged into prompts, so it must
    not stop the run (#891).  A span that claims the record grammar is still a
    forgery attempt and fails closed, and tool-owned publications keep their
    own stricter :class:`TrustedBody` checks.
    """
    occurrences = record_shaped_untrusted_markers(body)
    if not occurrences:
        return
    names = ", ".join(sorted({item.definition.token for item in occurrences}))
    raise AgentLoopError(
        f"The {surface} contains forged reserved protocol record syntax: {names}. "
        "Naming a reserved token in prose is allowed and is rendered as a "
        "descriptive label; remove the record-shaped span from that surface, "
        "or refer to the record by name instead."
    )


def log_untrusted_marker_neutralization(
    config: "AgentLoopConfig", *, surface: str, texts: Sequence[str | None]
) -> None:
    """Log once that untrusted GitHub text named reserved protocol records."""
    tokens: set[str] = set()
    for text in texts:
        tokens.update(named_reserved_marker_tokens(text or ""))
    if not tokens:
        return
    log(
        config,
        f"{surface} names reserved protocol marker(s) {', '.join(sorted(tokens))}; "
        "rendering them as descriptive labels in prompts. They carry no authority.",
    )


def resolve_authenticated_github_actor(
    runner: Runner,
    *,
    config: AgentLoopConfig,
) -> tuple[str, int]:
    """Resolve and cache the invocation actor used by trusted issue records."""
    cached = getattr(runner, "_agent_loop_authenticated_actor", None)
    if isinstance(cached, tuple) and len(cached) == 2:
        login, actor_id = cached
        if (
            isinstance(login, str)
            and bool(login)
            and isinstance(actor_id, int)
            and not isinstance(actor_id, bool)
            and actor_id > 0
        ):
            return login, actor_id
    if config.dry_run:
        raise AgentLoopError("Authenticated GitHub actor is unavailable in dry-run mode.")
    result = runner.run(
        [config.gh_cmd, "api", "user"],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError("Unable to resolve the authenticated GitHub actor.")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError("Authenticated GitHub actor response was not valid JSON.") from exc
    login = payload.get("login") if isinstance(payload, dict) else None
    actor_id = payload.get("id") if isinstance(payload, dict) else None
    if (
        not isinstance(login, str)
        or not login
        or not isinstance(actor_id, int)
        or isinstance(actor_id, bool)
        or actor_id < 1
    ):
        raise AgentLoopError(
            "Authenticated GitHub actor response lacked a login and immutable user ID."
        )
    setattr(runner, "_agent_loop_authenticated_actor", (login, actor_id))
    return login, actor_id


def reset_authenticated_github_actor(runner: Runner) -> None:
    """Start a fresh invocation-scoped actor cache.

    A ``Runner`` can be reused by tests and embedding callers across separate
    issue invocations.  Reusing its cached identity across those boundaries
    would make an actor change invisible and could incorrectly authenticate
    historical records.
    """
    if hasattr(runner, "_agent_loop_authenticated_actor"):
        delattr(runner, "_agent_loop_authenticated_actor")


ISSUE_THREAD_SURFACE = "issue"
PR_THREAD_SURFACE = "pr"
_THREAD_SURFACE_RE = re.compile(r"\A(?P<kind>issue|pr)#(?P<number>[1-9][0-9]*)\Z")


def comment_thread_surface(kind: str, number: int) -> str:
    """Name one issue or PR conversation thread, for example ``issue#827``."""
    if kind not in {ISSUE_THREAD_SURFACE, PR_THREAD_SURFACE}:
        raise AgentLoopError(f"Unknown comment thread surface kind {kind!r}.")
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise AgentLoopError("Comment thread surface number must be a positive integer.")
    return f"{kind}#{number}"


def parse_comment_thread_surface(surface: object) -> tuple[str, int]:
    match = _THREAD_SURFACE_RE.match(surface) if isinstance(surface, str) else None
    if match is None:
        raise AgentLoopError(f"Invalid comment thread surface {surface!r}.")
    return match.group("kind"), int(match.group("number"))


def parse_comment_timestamp(value: object) -> datetime.datetime:
    """Parse one immutable GitHub comment timestamp; never default it."""
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError("Comment timestamp is missing.")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError as exc:
        raise AgentLoopError(f"Comment timestamp {value!r} is not parseable.") from exc
    if parsed.tzinfo is None:
        raise AgentLoopError(f"Comment timestamp {value!r} carries no UTC offset.")
    return parsed


@dataclass(frozen=True)
class AuthenticatedComment:
    """One REST comment envelope with its immutable identity and ordering fields.

    The envelope is the only input the transaction-aware lineage entry points
    accept (#827), so a comment snapshot without a numeric ID, an immutable
    author user ID, or a parseable ``created_at`` cannot reach them.
    """

    surface: str
    comment_id: int
    author_login: str
    author_id: int
    created_at: str
    updated_at: str | None
    body: str

    def __post_init__(self) -> None:
        parse_comment_thread_surface(self.surface)
        for name in ("comment_id", "author_id"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise AgentLoopError(
                    f"Authenticated comment envelope {name} must be a positive integer."
                )
        if not isinstance(self.author_login, str) or not self.author_login:
            raise AgentLoopError("Authenticated comment envelope lacks an author login.")
        parse_comment_timestamp(self.created_at)
        if self.updated_at is not None:
            parse_comment_timestamp(self.updated_at)
        if not isinstance(self.body, str):
            raise AgentLoopError("Authenticated comment envelope body must be text.")

    @property
    def id(self) -> int:
        return self.comment_id

    @property
    def author(self) -> str:
        return self.author_login

    @property
    def created(self) -> datetime.datetime:
        return parse_comment_timestamp(self.created_at)


@dataclass(frozen=True)
class AuthenticatedCommentView:
    """A complete read of one thread, split by the invocation's actor.

    ``authored`` holds every comment written by the authenticated actor, in
    comment-ID order.  ``ignored_foreign`` holds comments from any other author
    that name a reserved protocol record; they grant nothing, are never
    adoptable, and exist only so diagnostics can list them.
    """

    surface: str
    actor_login: str
    actor_id: int
    authored: tuple[AuthenticatedComment, ...]
    ignored_foreign: tuple[AuthenticatedComment, ...] = ()

    def __post_init__(self) -> None:
        parse_comment_thread_surface(self.surface)
        if (
            isinstance(self.actor_id, bool)
            or not isinstance(self.actor_id, int)
            or self.actor_id < 1
            or not isinstance(self.actor_login, str)
            or not self.actor_login
        ):
            raise AgentLoopError("Authenticated comment view lacks an actor identity.")
        seen: set[int] = set()
        for group, own in ((self.authored, True), (self.ignored_foreign, False)):
            if not isinstance(group, tuple):
                raise AgentLoopError("Authenticated comment view groups must be tuples.")
            for comment in group:
                if not isinstance(comment, AuthenticatedComment):
                    raise AgentLoopError(
                        "Authenticated comment view accepts only authenticated envelopes."
                    )
                if comment.surface != self.surface:
                    raise AgentLoopError(
                        "Authenticated comment view mixes comment thread surfaces."
                    )
                if (comment.author_id == self.actor_id) != own:
                    raise AgentLoopError(
                        "Authenticated comment view author partition is inconsistent."
                    )
                if comment.comment_id in seen:
                    raise AgentLoopError("Authenticated comment view repeats a comment ID.")
                seen.add(comment.comment_id)
        if list(self.authored) != sorted(self.authored, key=lambda item: item.comment_id):
            raise AgentLoopError("Authenticated comment view is not in comment-ID order.")

    def comment(self, comment_id: int) -> AuthenticatedComment | None:
        for comment in self.authored:
            if comment.comment_id == comment_id:
                return comment
        return None

    def foreign_diagnostics(self) -> tuple[str, ...]:
        return tuple(
            f"ignored-foreign comment {comment.comment_id} on {comment.surface} by "
            f"{comment.author_login} (user ID {comment.author_id})"
            for comment in self.ignored_foreign
        )


def _authenticated_comment_from_rest(raw: object, *, surface: str) -> AuthenticatedComment:
    if not isinstance(raw, dict):
        raise AgentLoopError(f"Authenticated comment read of {surface} returned an incomplete page.")
    user = raw.get("user")
    if not isinstance(user, dict):
        raise AgentLoopError(
            f"Authenticated comment read of {surface} returned a comment without an author."
        )
    body = raw.get("body")
    return AuthenticatedComment(
        surface=surface,
        comment_id=raw.get("id"),  # type: ignore[arg-type]
        author_login=user.get("login"),  # type: ignore[arg-type]
        author_id=user.get("id"),  # type: ignore[arg-type]
        created_at=raw.get("created_at"),  # type: ignore[arg-type]
        updated_at=raw.get("updated_at") if raw.get("updated_at") is not None else None,  # type: ignore[arg-type]
        body=body if body is not None else "",
    )


def read_authenticated_protocol_comments(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    surface_kind: str,
    number: int,
) -> AuthenticatedCommentView:
    """Read one issue or PR conversation completely, bound to the invocation actor.

    Pagination is exhaustive and fail-closed: a failed, malformed, or
    self-contradictory page raises instead of returning a partial view.  Only
    comments whose immutable author user ID equals the authenticated actor are
    authoritative; a byte-exact record from anyone else is reported as
    ignored-foreign and can never be adopted (#827).
    """
    surface = comment_thread_surface(surface_kind, number)
    actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    page_size = 100
    # Loop guard only; see _merge_issue_comment_transport_identity.
    max_pages = 10_000
    page = 1
    envelopes: list[AuthenticatedComment] = []
    seen_ids: set[int] = set()
    while page <= max_pages:
        result = runner.run(
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues/{number}/comments?per_page={page_size}&page={page}",
            ],
            cwd=active_workdir(config),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"Authenticated comment read of {surface} failed on page {page}; "
                "a partial view is never used."
            )
        try:
            raw_page = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise AgentLoopError(
                f"Authenticated comment read of {surface} returned malformed JSON on page {page}."
            ) from exc
        if not isinstance(raw_page, list):
            raise AgentLoopError(
                f"Authenticated comment read of {surface} returned a non-list page {page}."
            )
        if len(raw_page) > page_size:
            raise AgentLoopError(
                f"Authenticated comment read of {surface} returned an oversized page {page}."
            )
        for raw_comment in raw_page:
            envelope = _authenticated_comment_from_rest(raw_comment, surface=surface)
            if envelope.comment_id in seen_ids:
                raise AgentLoopError(
                    f"Authenticated comment read of {surface} repeated comment ID "
                    f"{envelope.comment_id}."
                )
            seen_ids.add(envelope.comment_id)
            envelopes.append(envelope)
        if len(raw_page) < page_size:
            break
        page += 1
    else:
        raise AgentLoopError(
            f"Authenticated comment read of {surface} exceeded its pagination bound."
        )
    envelopes.sort(key=lambda item: item.comment_id)
    return AuthenticatedCommentView(
        surface=surface,
        actor_login=actor_login,
        actor_id=actor_id,
        authored=tuple(item for item in envelopes if item.author_id == actor_id),
        ignored_foreign=tuple(
            item
            for item in envelopes
            if item.author_id != actor_id and named_reserved_marker_tokens(item.body)
        ),
    )


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


def _fetch_protocol_comment_envelope(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    comment_id: int,
    context: str,
) -> dict[str, object]:
    """Re-read one stored comment when a write response omits its envelope."""
    result = runner.run(
        [config.gh_cmd, "api", f"repos/{config.repo}/issues/comments/{comment_id}"],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(f"{context} could not be read back from GitHub.")
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"{context} read-back returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(f"{context} read-back returned no comment envelope.")
    return payload


def verify_written_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    payload: dict[str, object],
    body: TrustedBody,
    expected_author_login: str | None,
    expected_author_id: int | None,
    context: str,
) -> int:
    """Apply the uniform read-back contract to one comment write response.

    Every tool-owned comment write compares the server's stored body for the
    just-written comment against the exact posted carrier and verifies the
    producing identity.  When the write response carries no body or author, the
    comment is fetched explicitly rather than trusted on its status code.
    """
    comment_id = payload.get("id")
    if not isinstance(comment_id, int) or isinstance(comment_id, bool) or comment_id < 1:
        raise AgentLoopError(f"{context} returned no comment ID.")
    envelope = payload
    if envelope.get("body") is None or not isinstance(
        envelope.get("user") or envelope.get("author"), dict
    ):
        envelope = _fetch_protocol_comment_envelope(
            runner, config=config, comment_id=comment_id, context=context
        )
    if envelope.get("body") != str(body):
        raise AgentLoopError(f"{context} returned a different body.")
    returned_user = envelope.get("user") or envelope.get("author")
    if not isinstance(returned_user, dict):
        raise AgentLoopError(f"{context} returned no author identity.")
    login = returned_user.get("login") or returned_user.get("slug")
    user_id = returned_user.get("id")
    if expected_author_login is not None and login != expected_author_login:
        raise AgentLoopError(f"{context} was authored by an unexpected actor.")
    if expected_author_id is not None and user_id != expected_author_id:
        raise AgentLoopError(f"{context} has an unexpected actor identity.")
    return comment_id


def post_verified_trusted_pr_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: TrustedBody,
    expected_author_login: str | None = None,
    expected_author_id: int | None = None,
) -> int:
    """Persist one trusted PR protocol record and return its server ID.

    Durable authorization records use the REST issue-comment endpoint so the
    returned comment identity is available to the caller.  The body is
    validated again at this seam, and the stored body and producing identity
    are read back through the shared envelope verifier before the record is
    adopted.
    """
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted PR protocol posting requires a TrustedBody.")
    body.validate_for_surface(PR_COMMENT_SURFACE)
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--method",
            "POST",
            f"repos/{config.repo}/issues/{pr_number}/comments",
            "-f",
            f"body={body}",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AgentLoopError(
            f"Unable to persist the trusted PR protocol record for PR #{pr_number}."
            + (f" {detail}" if detail else "")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"Trusted PR protocol record for PR #{pr_number} returned invalid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(
            f"Trusted PR protocol record for PR #{pr_number} returned no comment ID."
        )
    return verify_written_protocol_comment(
        runner,
        config=config,
        payload=payload,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        context=f"Trusted PR protocol record for PR #{pr_number}",
    )


def patch_verified_trusted_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    comment_id: int,
    body: TrustedBody,
    expected_author_login: str | None = None,
    expected_author_id: int | None = None,
    surface: str = PR_COMMENT_SURFACE,
) -> int:
    """Update one trusted protocol comment and verify the stored result.

    A successful exit status is not proof that the new body was persisted, so
    the update response is held to the same read-back contract as a create.
    """
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted protocol update requires a TrustedBody.")
    body.validate_for_surface(surface)
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--method",
            "PATCH",
            f"repos/{config.repo}/issues/comments/{comment_id}",
            "-f",
            f"body={body}",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    context = f"Trusted protocol comment update for comment {comment_id}"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AgentLoopError(
            f"Unable to persist the trusted protocol comment update for comment {comment_id}."
            + (f" {detail}" if detail else "")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(f"{context} returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError(f"{context} returned no comment envelope.")
    if payload.get("id") is None:
        payload = {**payload, "id": comment_id}
    updated_id = verify_written_protocol_comment(
        runner,
        config=config,
        payload=payload,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        context=context,
    )
    if updated_id != comment_id:
        raise AgentLoopError(f"{context} returned a different comment identity.")
    return updated_id


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


def post_verified_trusted_issue_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: TrustedBody,
    expected_author_login: str,
    expected_author_id: int,
) -> IssueComment:
    """Post one issue protocol record and verify the live server envelope."""
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted issue protocol posting requires a TrustedBody.")
    body.validate_for_surface(ISSUE_COMMENT_SURFACE)
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            "--method",
            "POST",
            f"repos/{config.repo}/issues/{issue_number}/comments",
            "--input",
            "-",
        ],
        cwd=active_workdir(config),
        input_text=json.dumps({"body": str(body)}, separators=(",", ":")),
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise AgentLoopError(
            f"Unable to persist the trusted issue protocol record for issue #{issue_number}."
            + (f" {detail}" if detail else "")
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned invalid JSON."
        ) from exc
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("id"), int)
        or isinstance(payload.get("id"), bool)
        or payload["id"] < 1
    ):
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned no numeric comment ID."
        )
    returned_body = payload.get("body")
    if returned_body != str(body):
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned a different body."
        )
    created_at = payload.get("created_at") or payload.get("createdAt")
    if not isinstance(created_at, str) or not created_at:
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned no server timestamp."
        )
    returned_user = payload.get("user") or payload.get("author")
    if not isinstance(returned_user, dict):
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned no author identity."
        )
    login = returned_user.get("login") or returned_user.get("slug")
    author_id = returned_user.get("id")
    if (
        not isinstance(login, str)
        or not login
        or not isinstance(author_id, int)
        or isinstance(author_id, bool)
        or author_id < 1
    ):
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned an invalid author identity."
        )
    if login != expected_author_login or author_id != expected_author_id:
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} was authored by an unexpected actor."
        )
    return IssueComment(
        author=login,
        created_at=created_at,
        body=returned_body,
        comment_id=payload["id"],
        author_id=author_id,
    )


def post_verified_trusted_issue_round_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: TrustedBody,
    expected_author_login: str,
    expected_author_id: int,
) -> IssueComment:
    """Post a round comment through the verified REST seam.

    Large canonical round comments may be split into transport sidecars and a
    final anchor.  Return the verified wrapper for that final anchor, which is
    the comment whose round metadata drives recovery and supersession.
    """
    bodies = prepare_round_comment(body)
    posted: IssueComment | None = None
    for prepared in bodies:
        posted = post_verified_trusted_issue_protocol_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=prepared,
            expected_author_login=expected_author_login,
            expected_author_id=expected_author_id,
        )
    if posted is None:  # pragma: no cover - prepare_round_comment always returns an anchor
        raise AgentLoopError("Verified issue round posting produced no comment wrapper.")
    return posted


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
    limit: int = ISSUE_RECOVERY_SEARCH_LIMIT,
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
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise AgentLoopError("search_issues limit must be a positive integer")
    log(config, f"Searching GitHub issues in {config.repo}: {search} (limit={limit})")
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
                "--limit",
                str(limit),
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
            "--limit",
            str(limit),
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


@dataclass(frozen=True)
class CiWatchOutcome:
    """Terminal result from the full-board post-approval watcher."""

    status: Literal[
        "passed",
        "not_started",
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
    empty_attempts = 0
    startup_attempt_limit = max(
        1,
        (config.ci_startup_timeout_seconds + config.ci_poll_interval_seconds - 1)
        // config.ci_poll_interval_seconds,
    )
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
        reliable = snapshot.check_query_status == "ok" and snapshot.branch_protection_status in {
            "configured", "not_found",
        }
        # A positively observed failure is actionable even when another query
        # was partial or a required check has not materialized yet. Those
        # conditions can delay a passing decision, but they must not hide a
        # concrete failing check from the coder.
        if snapshot.state == "failing":
            return CiWatchOutcome(
                status="failed", pr_checks=snapshot, failed_checks=snapshot.failing,
                head_sha=current_head, attempts_used=attempt + 1,
            )
        if snapshot.state == "passing" and reliable and not snapshot.pending and not snapshot.missing_required:
            return CiWatchOutcome(
                status="passed", pr_checks=snapshot, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        if snapshot.state == "no_checks" and reliable and not snapshot.missing_required:
            empty_attempts += 1
            # GitHub can expose an empty rollup while a pull_request workflow
            # is queued but no job/check has materialized.  It is a startup
            # state, not a passing board and never a merge permit.
            if empty_attempts >= startup_attempt_limit:
                return CiWatchOutcome(
                    status="not_started", pr_checks=snapshot, head_sha=current_head,
                    attempts_used=attempt + 1,
                )
        else:
            empty_attempts = 0
        if time.monotonic() >= deadline or attempt == limit - 1:
            return CiWatchOutcome(
                status="timeout", pr_checks=latest, head_sha=current_head,
                attempts_used=attempt + 1,
            )
        runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AssertionError("CI watch loop must return a terminal outcome")


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
