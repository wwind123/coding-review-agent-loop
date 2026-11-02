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
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Literal
from urllib.parse import quote

from .ci_health import (
    CiInfrastructureStall,
    PullRequestCheck,
    PullRequestChecks,
    StalledCheck,
    EXCLUSION_NOTE_SUFFIX,
    V2_RUN_NAME_PREFIX,
    _extract_run_id,
    WatchFailureClass,
    check_name_matches,
    classify_run_jobs,
    classify_watch_failure,
    decode_concatenated_json_objects,
    classify_ci_infrastructure_stall,
    is_wholly_infrastructure_blocked,
)
from .errors import AgentLoopError
from .github_issue_references import (
    _GITHUB_ISSUE_URL_RE,
    _CLOSING_ISSUE_REFERENCE_RE,
    _NON_CLOSING_ISSUE_REFERENCE_RE,
    affirmative_markdown_view,
    IssueReferenceEvidence,
    OpenPrClosingMatch,
    _issue_reference_evidence_from_match,
    parse_issue_reference_evidence,
    parse_strong_issue_reference_evidence,
    parse_raw_strong_issue_reference_evidence,
    parse_non_closing_issue_reference_evidence,
    parse_linked_issue_numbers,
)
from .github_transport import TransportConfigError, is_graphql_refusal, transport_mode
from .github_changed_paths import ChangedPath, PullRequestChangedPaths, get_pr_changed_paths
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
    sanitize_untrusted_prose,
    stored_body_matches_posted,
    strip_known_host_footer,
)
from . import github_retry
from .publication_resume import PreparedPublication, RoundPublication
from .review_spool import ReviewRoundSpool
from .github_retry import (
    DEFAULT_POLICY,
    GitHubAmbiguousWriteError,
    GitHubAttempt,
    GitHubRetryPolicy,
    GitHubTransientExhaustedError,
    attempt_from_result,
    classify_gh_failure,
    describe_gh_failure,
    format_attempt_history,
    run_gh_read,
)
from .runner import CommandResult, Runner
from .workdirs import active_workdir, github_api_cwd

if TYPE_CHECKING:
    from .config import AgentLoopConfig
    from .managed_ci import ManagedV1DispatchRecord

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


HumanRequirementVerification = Literal["unverified", "verified"]


@dataclass(frozen=True)
class TrustedHumanActor:
    """One configured identity whose signed comments count as requirements (#1022).

    Both the login and the immutable numeric user ID must match, as managed CI
    does for its trusted actor: a login alone can be renamed or reused.
    """

    login: str
    user_id: int


@dataclass(frozen=True)
class HumanReviewRequirement:
    source_type: str
    author: str | None
    created_at: str | None
    url: str | None
    body: str
    # Provenance only (#1022).  None of these fields participate in the
    # canonical key or stable requirement ID, so an acknowledgement recorded
    # before verification existed still names the same requirement.
    author_id: int | None = None
    author_verification: HumanRequirementVerification = "unverified"
    # GraphQL node ID of the source record.  gh's review projection has no
    # URL, so a PR review is located for REST verification by its node ID.
    node_id: str | None = None

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
    converted_capture_review_pending: bool = False


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

_AGENT_ISSUE_PR_HANDOFF_MARKER_RE = re.compile(
    r"<!--\s*AGENT_ISSUE_PR_HANDOFF:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)


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
    result = run_gh_read(runner, args, cwd=github_api_cwd(), check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        error_type = _GraphQLRefusedError if is_graphql_refusal(detail) else AgentLoopError
        history = describe_gh_failure(result)
        raise error_type(
            f"GitHub PR commit provenance query failed for PR #{pr_number}"
            + (f": {history}" if history else ".")
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
    try:
        mode = transport_mode()
    except TransportConfigError as exc:
        raise AgentLoopError(str(exc)) from exc
    if mode == "rest":
        return _read_pull_request_commit_metadata_rest(runner, config=config, pr_number=pr_number)
    try:
        first_payload = _query_pr_commit_connection(
            runner, config=config, pr_number=pr_number, after=None
        )
    except _GraphQLRefusedError:
        if mode != "auto":
            raise
        return _read_pull_request_commit_metadata_rest(runner, config=config, pr_number=pr_number)
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


class _GraphQLRefusedError(AgentLoopError):
    """The host refused GraphQL itself (not a GitHub-side query error)."""


# GitHub's REST commit listing for a pull request stops at 250 commits.
REST_PR_COMMIT_LIMIT = 250


def _rest_pr_object(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> tuple[str, int]:
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", f"repos/{config.repo}/pulls/{pr_number}"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"GitHub REST read of PR #{pr_number} failed during commit provenance."
            + _failure_detail(result)
        )
    data = _load_json_object(result, description=f"PR #{pr_number} for commit provenance")
    head = data.get("head")
    head_sha = head.get("sha") if isinstance(head, dict) else None
    total = data.get("commits")
    if not isinstance(head_sha, str) or not head_sha:
        raise AgentLoopError(f"GitHub returned no current head OID for PR #{pr_number}.")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        raise AgentLoopError(f"GitHub returned an invalid commit total for PR #{pr_number}.")
    return head_sha, total


def _read_pull_request_commit_metadata_rest(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
) -> tuple[PullRequestCommitMetadata, ...]:
    """REST equivalent of the GraphQL commit connection, with the same invariants.

    Used only where GraphQL is refused (#1029).  REST cannot list more than
    250 commits, so a longer history fails closed rather than returning a
    partial provenance set.
    """
    before_head, expected_total = _rest_pr_object(runner, config=config, pr_number=pr_number)
    if expected_total > REST_PR_COMMIT_LIMIT:
        raise AgentLoopError(
            f"GitHub PR #{pr_number} has {expected_total} commits; the REST API used where GraphQL is "
            f"unavailable lists at most {REST_PR_COMMIT_LIMIT}, so commit provenance cannot be read completely."
        )
    page_size = 100
    commits: list[PullRequestCommitMetadata] = []
    seen_oids: set[str] = set()
    for page in range(1, REST_PR_COMMIT_LIMIT // page_size + 2):
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/pulls/{pr_number}/commits?per_page={page_size}&page={page}",
            ],
            cwd=github_api_cwd(),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"GitHub REST commit page {page} for PR #{pr_number} failed."
                + _failure_detail(result)
            )
        try:
            items = json.loads(result.stdout or "null")
        except json.JSONDecodeError as exc:
            raise AgentLoopError(f"GitHub returned a malformed commit page for PR #{pr_number}.") from exc
        if not isinstance(items, list):
            raise AgentLoopError(f"GitHub returned a malformed commit page for PR #{pr_number}.")
        for item in items:
            commit = item.get("commit") if isinstance(item, dict) else None
            oid = item.get("sha") if isinstance(item, dict) else None
            message = commit.get("message") if isinstance(commit, dict) else None
            if not isinstance(oid, str) or not oid or not isinstance(message, str):
                raise AgentLoopError(f"GitHub returned an incomplete commit node for PR #{pr_number}.")
            if oid in seen_oids:
                raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} repeated a commit.")
            seen_oids.add(oid)
            commits.append(PullRequestCommitMetadata(oid=oid, message=message))
        if len(items) < page_size:
            break
    else:
        raise AgentLoopError(f"GitHub commit pagination for PR #{pr_number} did not terminate.")
    if len(commits) != expected_total:
        raise AgentLoopError(
            f"GitHub PR #{pr_number} commit provenance was truncated: provider reported "
            f"{expected_total} commits but returned {len(commits)}."
        )
    final_head, final_total = _rest_pr_object(runner, config=config, pr_number=pr_number)
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


def _failure_detail(result) -> str:
    """Render ``describe_gh_failure`` as a message suffix (empty when blank)."""
    detail = describe_gh_failure(result)
    return f"\n{detail}" if detail else ""


def detect_repo(runner: Runner, cwd: Path, gh_cmd: str) -> str:
    result = run_gh_read(
        runner,
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
    result = run_gh_read(
        runner,
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
    result = run_gh_read(
        runner,
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
    result = run_gh_read(
        runner,
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
    result = run_gh_read(
        runner,
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
    result = run_gh_read(
        runner,
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


def strip_bot_login_suffix(login: str | None) -> str | None:
    """Spell an app login as GitHub GraphQL does (REST appends ``[bot]``)."""
    if isinstance(login, str) and login.endswith("[bot]"):
        return login[: -len("[bot]")]
    return login


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
        requirement_sort = _human_requirement_sort_key(requirement)
        previous_sort = _human_requirement_sort_key(previous)
        if requirement_sort < previous_sort or (
            requirement_sort == previous_sort
            and requirement.author_verification == "verified"
            and previous.author_verification != "verified"
        ):
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

    result = run_gh_read(
        runner,
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
        human_requirements=verify_human_requirement_authors(
            runner,
            config=config,
            number=pr_number,
            candidates=_parse_pr_human_requirements(data),
        ),
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
        result = run_gh_read(
            runner,
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
                # REST carries both the API ``url`` and the web ``html_url``;
                # the permalink is ``html_url``.  GraphQL has only ``url``,
                # which is already the web permalink (#1022).
                url=_optional_str(raw_comment.get("html_url"))
                or _optional_str(raw_comment.get("url")),
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


def _shadowed_checks(checks: list[PullRequestCheck]) -> tuple[PullRequestCheck, ...]:
    seen: set[tuple[str, str]] = set()
    shadowed: list[PullRequestCheck] = []
    for check in checks:
        key = (check.kind, check.name)
        if key in seen:
            shadowed.append(check)
        else:
            seen.add(key)
    return tuple(shadowed)


def _listing_is_complete(payload: object, list_key: str, parsed_count: int) -> bool:
    """True when total_count == raw entries == parsed checks (no truncation, no skips)."""
    if not isinstance(payload, dict):
        return False
    total = payload.get("total_count")
    raw = payload.get(list_key)
    if isinstance(total, bool) or not isinstance(total, int) or not isinstance(raw, list):
        return False
    return total == len(raw) == parsed_count


def _check_suite_id(raw_check: dict) -> int | None:
    suite = raw_check.get("check_suite")
    suite_id = suite.get("id") if isinstance(suite, dict) else None
    if isinstance(suite_id, bool) or not isinstance(suite_id, int):
        return None
    return suite_id


def _check_app_slug(raw_check: dict) -> str | None:
    app = raw_check.get("app")
    return _optional_str(app.get("slug")) if isinstance(app, dict) else None


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
                check_suite_id=_check_suite_id(raw_check),
                app_slug=_check_app_slug(raw_check),
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

    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/branches/{base_branch}/protection/required_status_checks",
        ],
        cwd=github_api_cwd(),
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
        "GitHub branch protection could not be inspected due to an unexpected API failure."
        + _failure_detail(result),
    )


_ACTIONS_APP_SLUG = "github-actions"


def _fetch_head_dispatch_runs(
    runner: Runner, *, config: AgentLoopConfig, head_sha: str
) -> tuple[dict[int, tuple[int, bool]], list[str]]:
    """Map check_suite id -> (run id, is_managed) for workflow_dispatch runs on the head.

    Any problem returns an empty mapping plus a note, so nothing is excluded.
    """
    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/actions/runs?event=workflow_dispatch&head_sha={head_sha}&per_page=100",
        ],
        cwd=github_api_cwd(),
        check=False,
    )
    prefix = "workflow_dispatch run lookup failed, so no check was excluded: "
    if result.returncode != 0:
        return {}, [prefix + "query failed" + _failure_detail(result)]
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return {}, [prefix + "response was not valid JSON"]
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    total = payload.get("total_count") if isinstance(payload, dict) else None
    if not isinstance(runs, list):
        return {}, [prefix + "response had no workflow_runs list"]
    if isinstance(total, bool) or not isinstance(total, int) or total != len(runs):
        return {}, [prefix + "listing was truncated or inconsistent"]
    mapping: dict[int, tuple[int, bool]] = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        run_id = run.get("id")
        suite_id = run.get("check_suite_id")
        repo = run.get("repository")
        full_name = repo.get("full_name") if isinstance(repo, dict) else None
        names = [n for n in (run.get("name"), run.get("display_title")) if isinstance(n, str)]
        if (
            run.get("event") != "workflow_dispatch"
            or run.get("head_sha") != head_sha
            or not isinstance(full_name, str)
            or full_name.lower() != config.repo.lower()
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or isinstance(suite_id, bool)
            or not isinstance(suite_id, int)
            or not names
        ):
            continue
        if suite_id in mapping:
            return {}, [prefix + f"check suite {suite_id} was claimed by more than one run"]
        mapping[suite_id] = (run_id, any(n.startswith(V2_RUN_NAME_PREFIX) for n in names))
    return mapping, []


CODER_DISPATCH_AUDIT_SKEW_SECONDS = 30


@dataclass(frozen=True)
class CoderTurnDispatchRun:
    run_id: int
    head_sha: str
    url: str | None
    created_at: str


CoderTurnAmbiguityReason = Literal["multiple-matches", "shared-match", "incomplete-lookup"]
CODER_TURN_AMBIGUITY_REASONS: tuple[CoderTurnAmbiguityReason, ...] = (
    "multiple-matches", "shared-match", "incomplete-lookup",
)
_CODER_TURN_AMBIGUITY_CLAUSES: dict[str, str] = {
    "multiple-matches": "more than one run matches the same orchestrator dispatch",
    "shared-match": "this run matches more than one orchestrator dispatch",
    "incomplete-lookup": (
        "the run listing for the dispatched head was unavailable, so the match could not be confirmed"
    ),
}


@dataclass(frozen=True)
class CoderTurnAmbiguousRun:
    """A turn-window run that may be the orchestrator's own v1 dispatch (#1414)."""

    run: CoderTurnDispatchRun
    reasons: tuple[CoderTurnAmbiguityReason, ...]


@dataclass(frozen=True)
class CoderTurnDispatchAudit:
    """Non-managed workflow_dispatch runs created during one coder turn (#1366).

    Purely diagnostic: these runs are never qualification, and gating keeps
    excluding them through :func:`_fetch_head_dispatch_runs`.  Runs that may
    be the orchestrator's own v1 dispatch are kept apart in
    ``ambiguous_runs``; a run uniquely matched to a recorded orchestrator
    dispatch is only logged (``excluded_orchestrator_runs``) (#1414).
    """

    runs: tuple[CoderTurnDispatchRun, ...] = ()
    lookup_notes: tuple[str, ...] = ()
    ambiguous_runs: tuple[CoderTurnAmbiguousRun, ...] = ()
    excluded_orchestrator_runs: tuple[CoderTurnDispatchRun, ...] = ()


def _parse_github_timestamp(value: object) -> datetime.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=datetime.timezone.utc)


def _list_dispatch_runs(
    runner: Runner, *, config: AgentLoopConfig, query: str
) -> tuple[list[dict] | None, str | None]:
    """Every workflow_dispatch run for one listing query, or None plus a reason."""
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/actions/runs?{query}&per_page=100"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        return None, "query failed" + _failure_detail(result)
    pages = decode_concatenated_json_objects(result.stdout or "")
    if pages is None:
        return None, "response was not valid JSON"
    totals: set[int] = set()
    runs: list[dict] = []
    for page in pages:
        total = page.get("total_count")
        page_runs = page.get("workflow_runs")
        if isinstance(total, bool) or not isinstance(total, int) or not isinstance(page_runs, list):
            return None, "response had no workflow_runs list"
        totals.add(total)
        runs.extend(run for run in page_runs if isinstance(run, dict))
    if len(totals) > 1 or (totals and totals.pop() != len(runs)):
        return None, "listing was truncated or inconsistent"
    return runs, None


@dataclass(frozen=True)
class _DispatchCandidate:
    run: CoderTurnDispatchRun
    head_branch: str | None
    path: str | None
    created: datetime.datetime


def _candidate_matches_record(
    candidate: _DispatchCandidate, record: ManagedV1DispatchRecord
) -> bool:
    skew = datetime.timedelta(seconds=CODER_DISPATCH_AUDIT_SKEW_SECONDS)
    return (
        candidate.path is not None
        and candidate.path.endswith(f".github/workflows/{record.workflow_file}")
        and candidate.run.head_sha == record.expected_head_sha
        and (candidate.head_branch is None or candidate.head_branch == record.ref)
        and record.requested_at - skew <= candidate.created <= record.returned_at + skew
    )


def audit_coder_turn_dispatches(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    branch: str | None,
    head_shas: Sequence[str | None],
    turn_started_at: datetime.datetime,
    orchestrator_dispatches: Sequence[ManagedV1DispatchRecord] = (),
) -> CoderTurnDispatchAudit:
    """Find non-managed workflow_dispatch runs created since a coder turn started.

    Discovery is branch-scoped and time-bounded, so a dispatch on an
    intermediate head that a later push superseded is still found; it is
    unioned with the per-SHA listing of every known head and deduplicated by
    run id.  A lookup failure yields only a note and never raises.

    ``orchestrator_dispatches`` are this process's own managed v1 dispatches
    (#1414).  Discovery is widened to cover each record's correlation window
    and head, and records are associated with the complete deduplicated
    candidate set before the turn cutoff applies.  A record claims a run only
    when that run is its sole match, no other record matches the run, and the
    record head's listing succeeded; a claimed run is never reported.  Every
    other matched run inside the turn window is reported as ambiguous with
    typed reasons, never as coder-created.  Chronological order never decides
    ownership.
    """
    skew = datetime.timedelta(seconds=CODER_DISPATCH_AUDIT_SKEW_SECONDS)
    since = turn_started_at - skew
    records = tuple(orchestrator_dispatches)
    floor = min([since, *(record.requested_at - skew for record in records)])
    floor_text = floor.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    known_heads = tuple(dict.fromkeys(sha for sha in head_shas if sha))
    scope_heads = tuple(dict.fromkeys(
        (*known_heads, *(record.expected_head_sha for record in records if record.expected_head_sha))
    ))
    queries: list[tuple[str, str, str | None]] = []
    if branch:
        queries.append((
            f"branch `{branch}`",
            "event=workflow_dispatch"
            f"&branch={quote(branch, safe='')}&created={quote('>=' + floor_text, safe='')}",
            None,
        ))
    queries.extend(
        (f"head `{sha}`", f"event=workflow_dispatch&head_sha={quote(sha, safe='')}", sha)
        for sha in scope_heads
    )
    candidates: dict[int, _DispatchCandidate] = {}
    failed_heads: set[str] = set()
    notes: list[str] = []
    for scope, query, query_head in queries:
        try:
            runs, problem = _list_dispatch_runs(runner, config=config, query=query)
        except Exception as exc:  # diagnostic only: never block the round
            runs, problem = None, f"lookup raised {type(exc).__name__}"
        if runs is None:
            notes.append(f"workflow_dispatch audit for {scope} unavailable: {problem}")
            if query_head is not None:
                failed_heads.add(query_head)
            continue
        for run in runs:
            run_id = run.get("id")
            repo = run.get("repository")
            full_name = repo.get("full_name") if isinstance(repo, dict) else None
            names = [n for n in (run.get("name"), run.get("display_title")) if isinstance(n, str)]
            head_sha = run.get("head_sha")
            created = _parse_github_timestamp(run.get("created_at"))
            if (
                run.get("event") != "workflow_dispatch"
                or isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or not isinstance(full_name, str)
                or full_name.lower() != config.repo.lower()
                or not names
                or any(n.startswith(V2_RUN_NAME_PREFIX) for n in names)
                or not isinstance(head_sha, str)
                or created is None
            ):
                continue
            head_branch = _optional_str(run.get("head_branch"))
            if branch and head_branch != branch and head_sha not in scope_heads:
                continue
            path = _optional_str(run.get("path"))
            candidates.setdefault(run_id, _DispatchCandidate(
                run=CoderTurnDispatchRun(
                    run_id=run_id,
                    head_sha=head_sha,
                    url=_optional_str(run.get("html_url")),
                    created_at=str(run.get("created_at")),
                ),
                head_branch=head_branch,
                path=path.split("@", 1)[0] if path else None,
                created=created,
            ))

    record_matches = [
        [run_id for run_id, candidate in candidates.items() if _candidate_matches_record(candidate, record)]
        for record in records
    ]
    run_record_count: dict[int, int] = {}
    for matches in record_matches:
        for run_id in matches:
            run_record_count[run_id] = run_record_count.get(run_id, 0) + 1
    claimed: set[int] = set()
    reasons: dict[int, set[str]] = {}
    for record, matches in zip(records, record_matches):
        lookup_ok = record.expected_head_sha not in failed_heads
        if len(matches) == 1 and run_record_count[matches[0]] == 1 and lookup_ok:
            claimed.add(matches[0])
            continue
        for run_id in matches:
            run_reasons = reasons.setdefault(run_id, set())
            if len(matches) > 1:
                run_reasons.add("multiple-matches")
            if run_record_count[run_id] > 1:
                run_reasons.add("shared-match")
            if not lookup_ok:
                run_reasons.add("incomplete-lookup")

    reported: list[CoderTurnDispatchRun] = []
    ambiguous: list[CoderTurnAmbiguousRun] = []
    excluded: list[CoderTurnDispatchRun] = []
    for run_id in sorted(candidates):
        candidate = candidates[run_id]
        if run_id in claimed:
            excluded.append(candidate.run)
        elif candidate.created < since:
            continue
        elif reasons.get(run_id):
            ambiguous.append(CoderTurnAmbiguousRun(
                run=candidate.run,
                reasons=tuple(r for r in CODER_TURN_AMBIGUITY_REASONS if r in reasons[run_id]),
            ))
        else:
            reported.append(candidate.run)
    return CoderTurnDispatchAudit(
        runs=tuple(reported),
        lookup_notes=tuple(notes),
        ambiguous_runs=tuple(ambiguous),
        excluded_orchestrator_runs=tuple(excluded),
    )


def render_coder_dispatch_audit_note(audit: CoderTurnDispatchAudit) -> str:
    """The visible round audit note, or "" when there is nothing to report."""
    lines: list[str] = []
    if audit.runs:
        lines.append(
            "**Coder-turn CI dispatch audit:** ordinary workflow_dispatch created during coder "
            "turn before managed approval; non-authoritative diagnostic, not qualification."
        )
        for run in audit.runs:
            where = f" ({run.url})" if run.url else ""
            lines.append(f"- run {run.run_id} on `{run.head_sha}`{where}")
    if audit.ambiguous_runs:
        lines.append(
            "**Coder-turn CI dispatch audit (attribution undetermined):** these workflow_dispatch "
            "runs could be the orchestrator's own managed dispatch, but this audit could not "
            "confirm which run the orchestrator created; non-authoritative diagnostic, not "
            "qualification."
        )
        for entry in audit.ambiguous_runs:
            where = f" ({entry.run.url})" if entry.run.url else ""
            clauses = "; ".join(_CODER_TURN_AMBIGUITY_CLAUSES[r] for r in entry.reasons)
            lines.append(f"- run {entry.run.run_id} on `{entry.run.head_sha}`{where}: {clauses}")
    if audit.lookup_notes:
        lines.append(
            "**Coder-turn CI dispatch audit (diagnostic only, does not block this round):** "
            + "; ".join(audit.lookup_notes)
        )
    return "\n".join(lines)


def _log_orchestrator_dispatch_attribution(
    config: AgentLoopConfig, audit: CoderTurnDispatchAudit
) -> None:
    for run in audit.excluded_orchestrator_runs:
        log(
            config,
            f"Coder-turn dispatch audit: run {run.run_id} on {run.head_sha} matches the "
            "orchestrator's own managed v1 dispatch; not attributed to the coder",
        )
    for entry in audit.ambiguous_runs:
        log(
            config,
            f"Coder-turn dispatch audit: run {entry.run.run_id} on {entry.run.head_sha} may be "
            f"the orchestrator's own managed v1 dispatch ({', '.join(entry.reasons)}); "
            "attribution undetermined",
        )


def coder_turn_dispatch_audit_note(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    branch: str | None,
    head_shas: Sequence[str | None],
    turn_started_at: datetime.datetime,
    orchestrator_dispatches: Sequence[ManagedV1DispatchRecord] = (),
) -> str:
    """Run the post-turn audit, log its outcome, and return the visible note."""
    audit = audit_coder_turn_dispatches(
        runner, config, branch=branch, head_shas=head_shas, turn_started_at=turn_started_at,
        orchestrator_dispatches=orchestrator_dispatches,
    )
    for run in audit.runs:
        log(
            config,
            f"Ordinary workflow_dispatch run {run.run_id} on {run.head_sha} was created during "
            "a guarded coder turn; non-authoritative diagnostic, not qualification",
        )
    _log_orchestrator_dispatch_attribution(config, audit)
    for note in audit.lookup_notes:
        log(config, f"Coder-turn dispatch audit: {note}")
    return render_coder_dispatch_audit_note(audit)


def publish_rejected_coder_turn_dispatch_audit(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    branch: str | None,
    head_shas: Sequence[str | None],
    turn_started_at: datetime.datetime,
    orchestrator_dispatches: Sequence[ManagedV1DispatchRecord] = (),
) -> str:
    """Audit a guarded coder turn whose response was rejected (#1366).

    A rejected turn posts no coder comment, so any non-managed dispatch is
    published as its own visible orchestrator comment; a lookup failure is
    only logged.  Diagnostic only:
    this never raises, so the caller's original rejection is never masked.
    """
    try:
        live_head: str | None = None
        try:
            live_head = (get_pr_head_sha(runner, config, pr_number) or "").strip() or None
        except Exception:  # noqa: BLE001 - the branch query still covers it
            live_head = None
        audit = audit_coder_turn_dispatches(
            runner, config, branch=branch, head_shas=(*head_shas, live_head),
            turn_started_at=turn_started_at,
            orchestrator_dispatches=orchestrator_dispatches,
        )
        for problem in audit.lookup_notes:
            log(config, f"Coder-turn dispatch audit: {problem}")
        _log_orchestrator_dispatch_attribution(config, audit)
        # Only a found run is published: a lookup failure on a rejected turn
        # stays a logged diagnostic, so a rejection adds no PR noise.  A run
        # claimed as the orchestrator's own dispatch is never published.
        note = render_coder_dispatch_audit_note(
            CoderTurnDispatchAudit(runs=audit.runs, ambiguous_runs=audit.ambiguous_runs)
        )
        for run in audit.runs:
            log(
                config,
                f"Ordinary workflow_dispatch run {run.run_id} on {run.head_sha} was created during "
                "a rejected guarded coder turn; non-authoritative diagnostic, not qualification",
            )
        if note:
            post_pr_comment(
                runner,
                config=config,
                pr_number=pr_number,
                body=f"{note}\n\nThe coder response for this turn was rejected.\n\n-- Orchestrator",
            )
        return note
    except Exception as exc:  # noqa: BLE001 - never mask the rejection
        log(config, f"PR #{pr_number}: rejected coder-turn dispatch audit failed ({exc})")
        return ""


def _classify_dispatch_observation(
    check: PullRequestCheck, dispatch_map: dict[int, tuple[int, bool]]
) -> Literal["excluded", "conflict", "counted"]:
    """The single exclusion decision: only an Actions check run whose own suite
    is a non-managed dispatch run on the head, with no conflicting URL run id."""
    if (
        check.kind != "check_run"
        or check.app_slug != _ACTIONS_APP_SLUG
        or check.check_suite_id is None
    ):
        return "counted"
    entry = dispatch_map.get(check.check_suite_id)
    if entry is None or entry[1]:
        return "counted"
    if check.run_id is not None and check.run_id != str(entry[0]):
        return "conflict"
    return "excluded"


def _fetch_check_run_history(
    runner: Runner, *, config: AgentLoopConfig, head_sha: str
) -> list[PullRequestCheck] | None:
    """Every check-run observation (filter=all), or None when unusable."""
    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            "--paginate",
            f"repos/{config.repo}/commits/{head_sha}/check-runs?filter=all&per_page=100",
        ],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        return None
    pages = decode_concatenated_json_objects(result.stdout or "")
    if pages is None:
        return None
    totals: set[int] = set()
    raw_runs: list[object] = []
    for page in pages:
        total = page.get("total_count")
        page_runs = page.get("check_runs")
        if isinstance(total, bool) or not isinstance(total, int) or not isinstance(page_runs, list):
            return None
        totals.add(total)
        raw_runs.extend(page_runs)
    if len(totals) != 1:
        return None
    aggregate = {"total_count": totals.pop(), "check_runs": raw_runs}
    parsed, _errors = _parse_check_runs_payload(aggregate)
    if not _listing_is_complete(aggregate, "check_runs", len(parsed)):
        return None
    for raw in raw_runs:
        raw_id = raw.get("id") if isinstance(raw, dict) else None
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            return None
    ids = [check.check_id for check in parsed]
    if any(i is None for i in ids) or len(set(ids)) != len(ids):
        return None
    return parsed


def _rebuild_counted_check_runs(
    history: list[PullRequestCheck], dispatch_map: dict[int, tuple[int, bool]]
) -> tuple[list[PullRequestCheck], list[PullRequestCheck], list[PullRequestCheck]]:
    """Classify first, then collapse only fully identified same-suite reruns.

    Returns (counted check runs newest-first, excluded, conflicts).
    """
    excluded: list[PullRequestCheck] = []
    conflicts: list[PullRequestCheck] = []
    counted: list[PullRequestCheck] = []
    grouped: dict[tuple[str, int, str], PullRequestCheck] = {}
    for check in history:
        verdict = _classify_dispatch_observation(check, dispatch_map)
        if verdict == "excluded":
            excluded.append(check)
            continue
        if verdict == "conflict":
            conflicts.append(check)
            counted.append(check)
        elif check.app_slug is None or check.check_suite_id is None:
            counted.append(check)
        else:
            key = (check.app_slug, check.check_suite_id, check.name)
            held = grouped.get(key)
            if held is None or (check.check_id or 0) > (held.check_id or 0):
                grouped[key] = check
    counted.extend(grouped.values())
    counted.sort(key=lambda c: c.check_id or 0, reverse=True)
    return counted, excluded, conflicts


def _conflict_note(check: PullRequestCheck) -> str:
    return (
        f"{check.name} ({check.status.lower()}, check run {check.check_id}) names workflow run "
        f"{check.run_id} but its check suite {check.check_suite_id} belongs to a different "
        "dispatch run: kept counted"
    )


def _exclusion_note(check: PullRequestCheck, dispatch_map, head_sha: str) -> str:
    run_id, _managed = dispatch_map[check.check_suite_id]  # type: ignore[index]
    return (
        f"{check.name} ({check.status.lower()}) from ad hoc workflow_dispatch run {run_id} "
        f"(check suite {check.check_suite_id}) on {head_sha}{EXCLUSION_NOTE_SUFFIX}"
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
    check_runs_result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{metadata.head_sha}/check-runs",
        ],
        cwd=github_api_cwd(),
        check=False,
    )
    statuses_result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{metadata.head_sha}/status",
        ],
        cwd=github_api_cwd(),
        check=False,
    )

    check_errors: list[str] = []
    checks: list[PullRequestCheck] = []
    check_runs_ok = False
    statuses_ok = False
    check_runs_complete = False
    statuses_complete = False

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
            check_runs_complete = _listing_is_complete(payload, "check_runs", len(parsed_checks))
    else:
        check_errors.append("check-runs query failed" + _failure_detail(check_runs_result))

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
            statuses_complete = _listing_is_complete(payload, "statuses", len(parsed_statuses))
    else:
        check_errors.append("commit-status query failed" + _failure_detail(statuses_result))

    if check_runs_ok and statuses_ok:
        check_query_status: Literal["ok", "partial", "unavailable"] = "ok"
    elif check_runs_ok or statuses_ok:
        check_query_status = "partial"
    else:
        check_query_status = "unavailable"

    excluded_checks: list[PullRequestCheck] = []
    exclusion_notes: list[str] = []
    non_authoritative = False
    head_sha = metadata.head_sha
    if check_runs_ok and any(
        c.kind == "check_run" and c.app_slug == _ACTIONS_APP_SLUG and c.check_suite_id is not None
        for c in checks
    ):
        dispatch_map, lookup_notes = _fetch_head_dispatch_runs(
            runner, config=config, head_sha=head_sha
        )
        exclusion_notes.extend(lookup_notes)
        default_runs = [c for c in checks if c.kind == "check_run"]
        verdicts = [_classify_dispatch_observation(c, dispatch_map) for c in default_runs]
        for c, verdict in zip(default_runs, verdicts):
            if verdict == "conflict":
                exclusion_notes.append(_conflict_note(c))
        if "excluded" in verdicts:
            history = _fetch_check_run_history(runner, config=config, head_sha=head_sha)
            rebuilt = None
            if history is not None:
                rebuilt = _rebuild_counted_check_runs(history, dispatch_map)
                counted_ids = {c.check_id for c in rebuilt[0]}
                for c, verdict in zip(default_runs, verdicts):
                    if verdict != "excluded" and (
                        isinstance(c.check_id, bool)
                        or not isinstance(c.check_id, int)
                        or c.check_id not in counted_ids
                    ):
                        rebuilt = None
                        break
            if rebuilt is None:
                non_authoritative = True
                note = (
                    "complete check-run history was unavailable or inconsistent while an "
                    "ad hoc workflow_dispatch exclusion candidate exists, so nothing was "
                    "excluded and the check board is non-authoritative"
                )
                exclusion_notes.append(note)
                check_errors.append(note)
                if check_query_status == "ok":
                    check_query_status = "partial"
            else:
                counted_runs, history_excluded, conflicts = rebuilt
                seen_ids = {c.check_id for c in history_excluded}
                excluded_checks = history_excluded + [
                    c
                    for c, v in zip(default_runs, verdicts)
                    if v == "excluded" and c.check_id not in seen_ids
                ]
                checks = counted_runs + [c for c in checks if c.kind != "check_run"]
                for c in excluded_checks:
                    exclusion_notes.append(_exclusion_note(c, dispatch_map, head_sha))
                for c in conflicts:
                    note = _conflict_note(c)
                    if note not in exclusion_notes:
                        exclusion_notes.append(note)
    if excluded_checks or non_authoritative:
        log(config, "PR checks gate: " + "; ".join(exclusion_notes))

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

    if non_authoritative and state != "failing":
        state = "unavailable"

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
        shadowed=_shadowed_checks(checks),
        listing_complete=check_runs_complete and statuses_complete and not non_authoritative,
        excluded=tuple(excluded_checks),
        exclusion_notes=tuple(exclusion_notes),
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
                node_id=_optional_str(raw_review.get("id")),
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
    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/issues/{issue_number}",
            "--jq",
            "{number:.number,state:.state,is_pr:has(\"pull_request\"),url:.html_url}",
        ],
        cwd=github_api_cwd(),
        # `check=False` so a nonzero `gh` exit reaches the contextual
        # diagnostic below instead of the Runner's generic command failure.
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Unable to read issue #{issue_number} from {config.repo}: "
            f"`gh` exited {result.returncode}." + _failure_detail(result)
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

    ``gh issue view --json comments`` supplies the convenient GraphQL projection,
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
    # ``gh issue view --json comments`` is itself a bounded projection.  A full
    # page without the marker may simply mean that the durable record is on a
    # later REST page, so probe REST at the projection boundary too.
    page_size = _REST_ISSUE_COMMENT_PAGE_SIZE
    if not marker_in_projection and len(comments) < page_size:
        return comments
    transport_comments = read_rest_issue_comments(
        runner,
        config=config,
        issue_number=issue_number,
        purpose="trusted planning diagnostics cannot be resumed safely",
    )
    return _merge_transport_comments(
        comments, transport_comments, issue_number=issue_number
    )


_REST_ISSUE_COMMENT_PAGE_SIZE = 100


def read_rest_issue_comments(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    purpose: str,
    reject_empty_output: bool = False,
) -> tuple[IssueComment, ...]:
    """Read an issue's complete comment history from REST, with numeric IDs.

    Unlike the ``gh issue view --json comments`` projection, every comment carries
    its numeric comment ID and immutable author user ID, and no older comment
    is dropped at a connection cap.  Any incomplete read fails closed; the
    ``purpose`` clause names what cannot proceed safely without it.
    """
    page_size = _REST_ISSUE_COMMENT_PAGE_SIZE
    page = 1
    raw_transport_comments: list[object] = []
    seen_ids: set[int] = set()
    # This is only a loop guard.  GitHub's endpoint is finite, but a broken
    # proxy must not turn recovery into an unbounded operation.
    max_pages = 10_000
    while page <= max_pages:
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues/{issue_number}/comments?per_page={page_size}&page={page}",
            ],
            cwd=github_api_cwd(),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} is incomplete; "
                f"{purpose}." + _failure_detail(result)
            )
        if reject_empty_output and not (result.stdout or "").strip():
            # An empty successful response is not proof of an empty history;
            # callers that must fail closed treat it as an incomplete read.
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} returned an empty "
                f"response; {purpose}."
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
    return _parse_issue_comments(raw_transport_comments)


def merge_pr_comment_transport_identity(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    comments: tuple[IssueComment, ...],
) -> tuple[IssueComment, ...]:
    """Attach REST author IDs to a ``gh pr view`` comment projection (#1292).

    The GraphQL-shaped projection carries no numeric author IDs, so recovery
    records cannot be admitted from it.  An incomplete or empty REST read fails
    closed instead of reading as an empty history.
    """
    transport = read_rest_issue_comments(
        runner,
        config=config,
        issue_number=pr_number,
        purpose="PR recovery records cannot be authenticated",
        reject_empty_output=True,
    )
    return _merge_transport_comments(comments, transport, issue_number=pr_number)


def _merge_transport_comments(
    comments: tuple[IssueComment, ...],
    transport_comments: tuple[IssueComment, ...],
    *,
    issue_number: int,
) -> tuple[IssueComment, ...]:
    # GraphQL spells an app's login without the ``[bot]`` suffix REST uses, so
    # the match key normalizes it; a key shared by different raw logins is
    # ambiguous and never borrows either identity (#1029).
    by_key: dict[tuple[str | None, str | None, str | None], list[IssueComment]] = {}
    for comment in transport_comments:
        key = (strip_bot_login_suffix(comment.author), comment.created_at, comment.body)
        by_key.setdefault(key, []).append(comment)
    merged: list[IssueComment] = []
    matched_transport_ids: set[int] = set()
    for comment in comments:
        candidates = by_key.get(
            (strip_bot_login_suffix(comment.author), comment.created_at, comment.body), []
        )
        ambiguous = len({candidate.author for candidate in candidates}) > 1
        if ambiguous and isinstance(comment.body, str) and any(
            marker in comment.body
            for marker in ("AGENT_PLAN_VALIDATION_DIAGNOSTIC", "AGENT_LOOP_META", "AGENT_LOOP_SIDECAR")
        ):
            raise AgentLoopError(
                f"GitHub issue comment recovery for issue #{issue_number} found protocol comments from "
                "different authors that differ only by an app `[bot]` suffix; refusing to guess their identity."
            )
        transport = candidates.pop(0) if candidates and not ambiguous else None
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
        # A matched comment takes its whole identity from the live REST record
        # so login and numeric IDs agree for downstream authentication.
        merged.append(
            replace(
                comment,
                comment_id=(transport.comment_id if transport is not None else comment.comment_id),
                author_id=(transport.author_id if transport is not None else comment.author_id),
                author=(transport.author if transport is not None else comment.author),
            )
        )
    # The GraphQL projection can omit older comments once it reaches its
    # connection cap.  Add only authenticated protocol transport records
    # discovered by REST; ordinary comments remain sourced from the existing
    # projection and are not duplicated into prompt context.  Round anchors
    # and sidecars are needed together: a canonical plan may use sidecars, and
    # the authenticated canonical anchor is what semantically supersedes a
    # diagnostic during later recovery.
    transport_marker_names = (
        "AGENT_PLAN_VALIDATION_DIAGNOSTIC",
        "AGENT_LOOP_META",
        "AGENT_LOOP_SIDECAR",
        # Plan board-change links (#1377) are lineage elements too.
        "AGENT_PLAN_BOARD_CHANGE",
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

    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "issue",
            "view",
            str(issue_number),
            "--repo",
            config.repo,
            # gh >= 2.102 refuses ``--comments`` with ``--json``; the
            # ``comments`` field below already projects the comments.
            "--json",
            "number,title,body,url,author,createdAt,comments",
        ],
        cwd=github_api_cwd(),
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
    context = IssueContext(
        number=int(data.get("number") or issue_number),
        repo=config.repo,
        title=title,
        body=body,
        url=_optional_str(data.get("url")),
        comments=comments,
        human_requirements=verify_human_requirement_authors(
            runner,
            config=config,
            number=issue_number,
            candidates=_parse_issue_human_requirements(data),
        ),
    )
    from .execution_retirement import execution_view
    return execution_view(config, context)


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


_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE = (
    "signed human requirement authors cannot be verified against the configured "
    "trusted human reviewer set"
)


@dataclass(frozen=True)
class _RestSignedRecord:
    """The live REST identity and body of one requirement-bearing record."""

    login: str | None
    user_id: int | None
    body: str | None


def _identity_login_key(login: str | None) -> str:
    """Normalize a login the way GraphQL and REST spellings are reconciled.

    REST spells an app ``<slug>[bot]`` while GraphQL uses ``<slug>`` (or an
    ``app/<slug>`` form in some projections); those name one identity.
    """
    value = (login or "").strip()
    if value.casefold().startswith("app/"):
        value = value[len("app/"):]
    return (strip_bot_login_suffix(value) or "").casefold()


def _safe_diagnostic_text(value: object) -> str:
    text = "(none)" if value is None or value == "" else str(value)
    text = "".join(ch if ch.isprintable() else "?" for ch in text)
    return sanitize_untrusted_prose(text)[:200]


def _human_requirement_locator(
    requirement: HumanReviewRequirement,
) -> tuple[str, str] | None:
    if requirement.url:
        return ("url", requirement.url.strip())
    if requirement.node_id:
        return ("node", requirement.node_id.strip())
    return None


def _requirement_source_group(source_type: str) -> str:
    # PR conversation comments are issue comments in REST.
    if source_type in {"Issue comment", "PR comment"}:
        return "comments"
    if source_type == "Issue body":
        return "issue-body"
    if source_type == "PR review":
        return "reviews"
    return "unknown"


def _gh_api_json(
    runner: Runner, *, config: AgentLoopConfig, path: str, description: str
) -> object:
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", path],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0 or not (result.stdout or "").strip():
        raise AgentLoopError(
            f"GitHub {description} read is incomplete; {_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
            + _failure_detail(result)
        )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"GitHub {description} read returned malformed JSON; "
            f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
        ) from exc


def _index_rest_record(
    index: dict[tuple[str, str], _RestSignedRecord],
    key: tuple[str, str],
    record: _RestSignedRecord,
    *,
    description: str,
) -> None:
    if key in index:
        raise AgentLoopError(
            f"GitHub {description} read returned two records sharing one locator; "
            f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
        )
    index[key] = record


def _read_rest_issue_body_record(
    runner: Runner, *, config: AgentLoopConfig, number: int
) -> dict[tuple[str, str], _RestSignedRecord]:
    description = f"issue #{number} body"
    raw = _gh_api_json(
        runner, config=config, path=f"repos/{config.repo}/issues/{number}", description=description
    )
    if not isinstance(raw, dict):
        raise AgentLoopError(
            f"GitHub {description} read returned a non-object payload; "
            f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
        )
    record = _RestSignedRecord(
        login=_author_login(raw.get("user")),
        user_id=_author_id(raw.get("user")),
        body=_optional_str(raw.get("body")),
    )
    index: dict[tuple[str, str], _RestSignedRecord] = {}
    html_url = _optional_str(raw.get("html_url"))
    if html_url:
        index[("url", html_url.strip())] = record
    node_id = _optional_str(raw.get("node_id"))
    if node_id:
        index[("node", node_id.strip())] = record
    return index


def _read_rest_issue_comment_records(
    runner: Runner, *, config: AgentLoopConfig, number: int
) -> dict[tuple[str, str], _RestSignedRecord]:
    comments = read_rest_issue_comments(
        runner,
        config=config,
        issue_number=number,
        purpose=_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE,
        reject_empty_output=True,
    )
    index: dict[tuple[str, str], _RestSignedRecord] = {}
    for comment in comments:
        if not comment.url:
            continue
        _index_rest_record(
            index,
            ("url", comment.url.strip()),
            _RestSignedRecord(login=comment.author, user_id=comment.author_id, body=comment.body),
            description=f"issue #{number} comment",
        )
    return index


_REST_PR_REVIEW_PAGE_SIZE = 100


def _read_rest_pr_review_records(
    runner: Runner, *, config: AgentLoopConfig, number: int
) -> dict[tuple[str, str], _RestSignedRecord]:
    description = f"pull request #{number} review"
    index: dict[tuple[str, str], _RestSignedRecord] = {}
    page = 1
    max_pages = 10_000
    while page <= max_pages:
        raw_page = _gh_api_json(
            runner,
            config=config,
            path=(
                f"repos/{config.repo}/pulls/{number}/reviews"
                f"?per_page={_REST_PR_REVIEW_PAGE_SIZE}&page={page}"
            ),
            description=description,
        )
        if not isinstance(raw_page, list):
            raise AgentLoopError(
                f"GitHub {description} read returned a non-list page; "
                f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
            )
        for raw_review in raw_page:
            if not isinstance(raw_review, dict):
                raise AgentLoopError(
                    f"GitHub {description} read returned an incomplete page; "
                    f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
                )
            record = _RestSignedRecord(
                login=_author_login(raw_review.get("user")),
                user_id=_author_id(raw_review.get("user")),
                body=_optional_str(raw_review.get("body")),
            )
            html_url = _optional_str(raw_review.get("html_url"))
            if html_url:
                _index_rest_record(index, ("url", html_url.strip()), record, description=description)
            node_id = _optional_str(raw_review.get("node_id"))
            if node_id:
                _index_rest_record(index, ("node", node_id.strip()), record, description=description)
        if len(raw_page) < _REST_PR_REVIEW_PAGE_SIZE:
            break
        page += 1
    else:
        raise AgentLoopError(
            f"GitHub {description} read exceeded its pagination bound; "
            f"{_HUMAN_REQUIREMENT_VERIFICATION_PURPOSE}."
        )
    return index


def _trusted_actor_exclusion_reason(
    trusted: Sequence[TrustedHumanActor], *, login: str | None, user_id: int | None
) -> tuple[TrustedHumanActor | None, str | None]:
    """Return the matching trusted entry, or the reason the author is excluded."""
    login_key = _identity_login_key(login)
    if user_id is None:
        return None, "the REST record has no numeric author ID"
    by_login = [actor for actor in trusted if _identity_login_key(actor.login) == login_key]
    for actor in by_login:
        if actor.user_id == user_id:
            return actor, None
    if by_login:
        return None, (
            "the login matches a trusted entry but the numeric user ID differs "
            "(renamed or reused login)"
        )
    by_id = [actor for actor in trusted if actor.user_id == user_id]
    if by_id:
        return None, (
            "the numeric user ID matches a trusted entry but the login differs "
            f"(trusted login {_safe_diagnostic_text(by_id[0].login)})"
        )
    return None, "the author is not in the trusted human reviewer set"


def verify_human_requirement_authors(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    number: int,
    candidates: tuple[HumanReviewRequirement, ...],
) -> tuple[HumanReviewRequirement, ...]:
    """Admit signed requirements only from configured trusted identities (#1022).

    With no trusted human reviewer identities configured this is a no-op: every
    candidate is surfaced unverified and no GitHub call is made.  Otherwise each
    candidate is joined to its live REST record by exact source locator (URL,
    or node ID for a PR review) -- never by timestamp or body -- bound to that
    record's current signed content, and admitted only when both the author
    login and numeric user ID match one configured entry.  Excluded records are
    logged and remain ordinary untrusted discussion.  An incomplete REST read,
    a duplicate REST locator, or a projection/REST disagreement fails closed.
    """
    trusted = tuple(getattr(config, "human_reviewer_trusted_actors", ()) or ())
    if not trusted or not candidates:
        return candidates
    groups = {_requirement_source_group(candidate.source_type) for candidate in candidates}
    rest: dict[str, dict[tuple[str, str], _RestSignedRecord]] = {}
    if "issue-body" in groups:
        rest["issue-body"] = _read_rest_issue_body_record(runner, config=config, number=number)
    if "comments" in groups:
        rest["comments"] = _read_rest_issue_comment_records(runner, config=config, number=number)
    if "reviews" in groups:
        rest["reviews"] = _read_rest_pr_review_records(runner, config=config, number=number)

    admitted: list[HumanReviewRequirement] = []
    excluded = 0
    for candidate in candidates:
        locator = _human_requirement_locator(candidate)
        group = _requirement_source_group(candidate.source_type)
        record = rest.get(group, {}).get(locator) if locator is not None else None
        observed_login: str | None = candidate.author
        observed_id: int | None = None
        reason: str | None = None
        if locator is None:
            reason = "the record has no verifiable record locator"
        elif record is None:
            reason = "the record locator is absent from the complete REST read"
        else:
            rest_body = parse_signed_human_requirement_body(record.body)
            if (
                rest_body is None
                or _normalize_requirement_body(rest_body)
                != _normalize_requirement_body(candidate.body)
                or _identity_login_key(record.login) != _identity_login_key(candidate.author)
            ):
                raise AgentLoopError(
                    f"The signed {candidate.source_type.lower()} at "
                    f"{_safe_diagnostic_text(candidate.url or candidate.node_id)} changed during "
                    "author verification (its REST body, signature, or author no longer matches "
                    "the loaded context); rerun agent-loop to verify the current record."
                )
            observed_login = record.login
            observed_id = record.user_id
            match, reason = _trusted_actor_exclusion_reason(
                trusted, login=record.login, user_id=record.user_id
            )
            if match is not None:
                admitted.append(
                    replace(
                        candidate,
                        author_id=record.user_id,
                        author_verification="verified",
                    )
                )
                continue
        excluded += 1
        log(
            config,
            "Signed human requirement excluded: "
            f"{candidate.source_type} {_safe_diagnostic_text(candidate.url or candidate.node_id)} "
            f"by {_safe_diagnostic_text(observed_login)} "
            f"(user ID {_safe_diagnostic_text(observed_id)}): {reason}. "
            "It remains ordinary untrusted discussion.",
        )
    if excluded:
        log(
            config,
            f"Signed human requirement verification: {len(admitted)} admitted, "
            f"{excluded} excluded (not from the configured trusted human reviewer set).",
        )
    return deduplicate_human_requirements(admitted)


def _apply_publication(
    publication: RoundPublication | None,
    config: AgentLoopConfig,
    bodies: Sequence[object],
) -> tuple[Sequence[object], frozenset[int]]:
    """Freeze the composed bodies or substitute the frozen ones (#1258)."""
    if publication is None or config.dry_run:
        return bodies, frozenset()
    prepared = publication.prepare(bodies)
    return prepared.bodies, prepared.already_public


def round_publication(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    spool: ReviewRoundSpool,
    reviewer_name: str,
    flow: str,
    round_number: int,
    subject: str,
    surface_kind: str,
    number: int,
    validation_context: str,
) -> RoundPublication:
    """Build the freeze/resume hook for one spooled reviewer's round comment."""
    surface = comment_thread_surface(surface_kind, number)
    return RoundPublication(
        spool=spool,
        reviewer_name=reviewer_name,
        flow=flow,
        round_number=round_number,
        subject=subject,
        validation_context=validation_context,
        surface=surface,
        resolve_actor=lambda: resolve_authenticated_github_actor(runner, config=config),
        list_comments=lambda actor: read_complete_comment_listing(
            runner, config=config, surface=surface, actor=actor
        ),
        config=config,
    )


def post_frozen_round_bodies(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    surface_kind: str,
    number: int,
    plan: "PreparedPublication",
) -> int:
    """Post exactly the frozen bodies not yet public, in order; return how many."""
    command = [PR_THREAD_SURFACE if surface_kind == PR_THREAD_SURFACE else "issue", "comment", str(number)]
    claimed: set[int] = set()
    posted = 0
    for index, body in enumerate(plan.bodies, start=1):
        if index in plan.already_public:
            continue
        _post_comment_body(
            runner, config=config, command=command, body=body, claimed=claimed, body_index=index
        )
        posted += 1
    return posted


def post_pr_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: str | TrustedBody,
    publication: RoundPublication | None = None,
) -> None:
    carrier = body if isinstance(body, TrustedBody) else TrustedBody.current_untrusted_visible(body)
    bodies = prepare_round_comment(carrier)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to PR #{pr_number}")
    log(config, f"Posting agent output to PR #{pr_number}")
    claimed: set[int] = set()
    for prepared in bodies:
        prepared.validate_for_surface(PR_COMMENT_SURFACE)
    outgoing, already_public = _apply_publication(publication, config, bodies)
    for index, prepared in enumerate(outgoing, start=1):
        if index in already_public:
            continue
        _post_comment_body(
            runner, config=config, command=["pr", "comment", str(pr_number)], body=prepared,
            claimed=claimed, body_index=index,
        )


def post_issue_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: str | TrustedBody,
    publication: RoundPublication | None = None,
) -> None:
    carrier = body if isinstance(body, TrustedBody) else TrustedBody.current_untrusted_visible(body)
    bodies = prepare_round_comment(carrier)
    if len(bodies) > 1:
        log(config, f"Posting round transport with {len(bodies) - 1} sidecars to issue #{issue_number}")
    log(config, f"Posting agent output to issue #{issue_number}")
    claimed: set[int] = set()
    for prepared in bodies:
        prepared.validate_for_surface(ISSUE_COMMENT_SURFACE)
    outgoing, already_public = _apply_publication(publication, config, bodies)
    for index, prepared in enumerate(outgoing, start=1):
        if index in already_public:
            continue
        _post_comment_body(
            runner, config=config, command=["issue", "comment", str(issue_number)], body=prepared,
            claimed=claimed, body_index=index,
        )


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
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", "user"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            "Unable to resolve the authenticated GitHub actor." + _failure_detail(result)
        )
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


_host_footer_logged = False


def reset_host_footer_log_latch() -> None:
    """Start a new invocation's host-footer log latch.

    Called only from ``_new_usage_context``, whose callers are exactly the run
    entries that own a usage context, so every owning invocation logs the
    footer once and a nested run that received a usage context shares it.
    """
    global _host_footer_logged
    _host_footer_logged = False


def note_host_footer_observed(config: AgentLoopConfig | None, context: str) -> None:
    """Log once per invocation that the known host comment footer was seen."""
    global _host_footer_logged
    if _host_footer_logged or config is None:
        return
    _host_footer_logged = True
    log(
        config,
        "GitHub host appended its known comment footer to a stored protocol record "
        f"({context}); the exact footer is tolerated once and removed before parsing.",
    )


def _authenticated_comment_from_rest(
    raw: object,
    *,
    surface: str,
    config: AgentLoopConfig | None = None,
    require_body: bool = False,
) -> AuthenticatedComment:
    """Build one authenticated envelope from a raw REST comment.

    This is a comment-ingestion boundary: the known host footer is removed
    here exactly once (#1043), so downstream strict parsers never strip it.
    """
    if not isinstance(raw, dict):
        raise AgentLoopError(f"Authenticated comment read of {surface} returned an incomplete page.")
    user = raw.get("user")
    if not isinstance(user, dict):
        raise AgentLoopError(
            f"Authenticated comment read of {surface} returned a comment without an author."
        )
    body = raw.get("body")
    if require_body and not isinstance(body, str):
        raise AgentLoopError(
            f"Authenticated comment read of {surface} returned comment {raw.get('id')!r} "
            "without a text body."
        )
    if isinstance(body, str):
        body, footered = strip_known_host_footer(body)
        if footered:
            note_host_footer_observed(config, f"re-read on {surface}")
    return AuthenticatedComment(
        surface=surface,
        comment_id=raw.get("id"),  # type: ignore[arg-type]
        author_login=user.get("login"),  # type: ignore[arg-type]
        author_id=user.get("id"),  # type: ignore[arg-type]
        created_at=raw.get("created_at"),  # type: ignore[arg-type]
        updated_at=raw.get("updated_at") if raw.get("updated_at") is not None else None,  # type: ignore[arg-type]
        body=body if body is not None else "",
    )


def _read_comment_pages(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    surface: str,
    number: int,
    since: datetime.datetime | None = None,
    require_body: bool = False,
) -> list[AuthenticatedComment]:
    """Page one conversation exhaustively; any defect raises (never a partial list)."""
    page_size = 100
    # Loop guard only; see _merge_issue_comment_transport_identity.
    max_pages = 10_000
    since_query = (
        "&since=" + since.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if since is not None
        else ""
    )
    page = 1
    envelopes: list[AuthenticatedComment] = []
    seen_ids: set[int] = set()
    while page <= max_pages:
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues/{number}/comments?per_page={page_size}&page={page}"
                + since_query,
            ],
            cwd=github_api_cwd(),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"Authenticated comment read of {surface} failed on page {page}; "
                "a partial view is never used." + _failure_detail(result)
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
            envelope = _authenticated_comment_from_rest(
                raw_comment, surface=surface, config=config, require_body=require_body
            )
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
    return envelopes


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
    envelopes = _read_comment_pages(runner, config=config, surface=surface, number=number)
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


@dataclass(frozen=True)
class CompleteCommentListing:
    """An exhaustive, internally consistent read of one conversation (#510).

    ``actor_id`` is the author the listing is bound to, or ``None`` when it
    holds every author's comments.  Only a complete listing can show that a
    comment is absent.
    """

    surface: str
    actor_id: int | None
    comments: tuple[AuthenticatedComment, ...]


@dataclass(frozen=True)
class UnknownCommentListing:
    """The surface could not be read completely; absence is unprovable."""

    surface: str
    reason: str


def read_complete_comment_listing(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    surface: str,
    since: datetime.datetime | None = None,
    actor: bool = True,
) -> CompleteCommentListing | UnknownCommentListing:
    """Return a complete listing or ``Unknown``; never a partial list.

    ``actor=True`` binds the listing to the authenticated actor's immutable user
    ID; ``actor=False`` returns every author's comments with the same
    completeness rules.  A failed (after read retry), malformed, non-list or
    self-contradictory page, or an unresolvable actor, yields ``Unknown``.
    """
    kind, number = parse_comment_thread_surface(surface)
    del kind
    try:
        actor_id: int | None = None
        if actor:
            actor_id = resolve_authenticated_github_actor(runner, config=config)[1]
        envelopes = _read_comment_pages(
            runner, config=config, surface=surface, number=number, since=since,
            require_body=True,
        )
    except (AgentLoopError, OSError) as exc:
        # An unusable working directory (for example a not-yet-cloned checkout)
        # makes absence unprovable; it never authorizes a replay.
        return UnknownCommentListing(surface=surface, reason=str(exc))
    if actor_id is not None:
        envelopes = [item for item in envelopes if item.author_id == actor_id]
    return CompleteCommentListing(
        surface=surface, actor_id=actor_id, comments=tuple(envelopes)
    )


COMMENT_RECONCILIATION_WINDOW = datetime.timedelta(minutes=10)


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


@dataclass(frozen=True)
class ReconciledComment:
    """Outcome of a reconciled comment write.

    ``result`` is the last write command result (a failure only when it was
    non-transient, left for the caller's own message); ``adopted`` is the comment
    that proved an earlier ambiguous attempt had been accepted.
    """

    result: CommandResult | None
    adopted: AuthenticatedComment | None = None


def _ambiguity_message(
    *,
    what: str,
    surface: str,
    body_index: int | None,
    outcome: str,
    attempts: Sequence[GitHubAttempt],
    detail: str = "",
) -> str:
    position = f" (body {body_index})" if body_index is not None else ""
    return (
        f"{what} on {surface}{position} failed with a transient GitHub error and the "
        f"last attempt is {outcome}. {detail}".rstrip()
        + "\nGitHub attempt history:\n"
        + format_attempt_history(attempts)
        + "\nA rerun resumes from the local review spool without re-invoking reviewers; "
        "check the conversation for the comment before publishing it by hand."
    )


def post_comment_with_reconciliation(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    surface: str,
    body: str,
    write: Callable[[], CommandResult],
    claimed: set[int] | None = None,
    body_index: int | None = None,
    what: str = "Comment publication",
    policy: GitHubRetryPolicy | None = None,
) -> ReconciledComment:
    """Run one comment create with bounded retry and complete-listing reconciliation.

    ``write`` performs the create with ``check=False``.  Before the first attempt a
    baseline records the actor's comments that already carry this exact body, so
    a pre-existing identical comment is never adopted and an intentionally
    repeated publication is never suppressed.  After every transient failure
    (including the last) a complete listing must show exactly one new unclaimed
    exact-body comment to adopt it; zero authorize a replay; several, or an
    unreadable listing, fail closed.  ``claimed`` carries adopted IDs across the
    bodies of one publication sequence.  Dry-run bypasses all of it.
    """
    if config.dry_run:
        return ReconciledComment(result=write())
    claimed = claimed if claimed is not None else set()
    active = policy or DEFAULT_POLICY
    window_start = _utc_now() - COMMENT_RECONCILIATION_WINDOW
    baseline = read_complete_comment_listing(
        runner, config=config, surface=surface, since=window_start
    )
    baseline_ids: set[int] | None = None
    if isinstance(baseline, CompleteCommentListing):
        baseline_ids = {
            c.comment_id for c in baseline.comments if c.body == body
        }
    history: list[GitHubAttempt] = []
    for number in range(1, active.attempts + 1):
        started = time.monotonic()
        result = write()
        if result.returncode == 0:
            return ReconciledComment(result=result)
        attempt = attempt_from_result(number, result, started)
        history.append(attempt)
        if attempt.classification == "permanent":
            return ReconciledComment(result=result)
        if baseline_ids is None:
            raise GitHubAmbiguousWriteError(
                _ambiguity_message(
                    what=what, surface=surface, body_index=body_index, outcome="unknown",
                    attempts=history,
                    detail="The pre-write baseline could not be read, so a pre-existing "
                    "identical comment cannot be told apart from this write's. "
                    + baseline.reason  # type: ignore[union-attr]
                ),
                tuple(history),
            )
        listing = read_complete_comment_listing(
            runner, config=config, surface=surface, since=window_start
        )
        if isinstance(listing, UnknownCommentListing):
            raise GitHubAmbiguousWriteError(
                _ambiguity_message(
                    what=what, surface=surface, body_index=body_index, outcome="unknown",
                    attempts=history,
                    detail="The reconciliation listing was incomplete, so absence is "
                    "unprovable and the write is not replayed. " + listing.reason,
                ),
                tuple(history),
            )
        candidates = [
            c
            for c in listing.comments
            if c.body == body
            and c.created >= window_start
            and c.comment_id not in baseline_ids
            and c.comment_id not in claimed
        ]
        if len(candidates) == 1:
            claimed.add(candidates[0].comment_id)
            log(
                config,
                f"{what} on {surface}: GitHub accepted the write despite a transient "
                f"error; adopted comment {candidates[0].comment_id}",
            )
            return ReconciledComment(result=None, adopted=candidates[0])
        if len(candidates) > 1:
            ids = ", ".join(str(c.comment_id) for c in candidates)
            raise GitHubAmbiguousWriteError(
                _ambiguity_message(
                    what=what, surface=surface, body_index=body_index, outcome="unknown",
                    attempts=history,
                    detail=f"{len(candidates)} identical unclaimed comments ({ids}) exist; "
                    "none is chosen and the write is not replayed.",
                ),
                tuple(history),
            )
        if number < active.attempts:
            github_retry.sleep_before_retry(active, number)
    raise GitHubTransientExhaustedError(
        _ambiguity_message(
            what=what, surface=surface, body_index=body_index, outcome="verified absent",
            attempts=history,
            detail="The retry budget is exhausted.",
        ),
        tuple(history),
    )


def _post_trusted_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    command: list[str],
    body: str,
    claimed: set[int] | None = None,
    body_index: int | None = None,
) -> None:
    """Post canonical protocol output after validating its marker encoding."""
    _post_comment_body(
        runner, config=config, command=command, body=body, claimed=claimed, body_index=body_index
    )


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
    claimed: set[int] = set()
    for index, prepared in enumerate(bodies, start=1):
        prepared.validate_for_surface(PR_COMMENT_SURFACE)
        _post_trusted_protocol_comment(
            runner, config=config, command=["pr", "comment", str(pr_number)], body=prepared,
            claimed=claimed, body_index=index,
        )


def _fetch_protocol_comment_envelope(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    comment_id: int,
    context: str,
) -> dict[str, object]:
    """Re-read one stored comment when a write response omits its envelope."""
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", f"repos/{config.repo}/issues/comments/{comment_id}"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"{context} could not be read back from GitHub." + _failure_detail(result)
        )
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
    """Apply the uniform read-back contract and return the comment ID."""
    return verify_written_protocol_comment_observed(
        runner,
        config=config,
        payload=payload,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        context=context,
    ).comment_id


@dataclass(frozen=True)
class WrittenProtocolComment:
    """One verified comment write and what its read-back observed.

    ``server_updated_at`` is the epoch-second value of the read-back
    envelope's ``updated_at`` only.  It never falls back to ``created_at``,
    which is the comment's immutable creation time and would misdate a PATCH
    of an older comment.
    """

    comment_id: int
    host_footer_observed: bool
    server_updated_at: int | None


def _envelope_updated_at_epoch(envelope: dict[str, object]) -> int | None:
    value = envelope.get("updated_at")
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(parse_comment_timestamp(value).timestamp())
    except AgentLoopError:
        return None


def verify_written_protocol_comment_observed(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    payload: dict[str, object],
    body: TrustedBody,
    expected_author_login: str | None,
    expected_author_id: int | None,
    context: str,
) -> WrittenProtocolComment:
    """Apply the uniform read-back contract to one comment write response.

    Every tool-owned comment write compares the server's stored body for the
    just-written comment against the exact posted carrier and verifies the
    producing identity.  When the write response carries no body or author, the
    comment is fetched explicitly rather than trusted on its status code.  The
    stored body may carry exactly one known host footer (#1043); any other
    difference fails closed.
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
    matches, footer_observed = stored_body_matches_posted(envelope.get("body"), str(body))
    if not matches:
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
    if footer_observed:
        note_host_footer_observed(config, context)
    return WrittenProtocolComment(
        comment_id=comment_id,
        host_footer_observed=footer_observed,
        server_updated_at=_envelope_updated_at_epoch(envelope),
    )


def post_verified_trusted_pr_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: TrustedBody,
    expected_author_login: str | None = None,
    expected_author_id: int | None = None,
) -> int:
    """Persist one trusted PR protocol record and return its server ID."""
    return post_verified_trusted_pr_protocol_comment_observed(
        runner,
        config=config,
        pr_number=pr_number,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
    ).comment_id


def post_verified_trusted_pr_protocol_comment_observed(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    body: TrustedBody,
    expected_author_login: str | None = None,
    expected_author_id: int | None = None,
) -> WrittenProtocolComment:
    """Persist one trusted PR protocol record and report its read-back.

    Durable authorization records use the REST issue-comment endpoint so the
    returned comment identity is available to the caller.  The body is
    validated again at this seam, and the stored body and producing identity
    are read back through the shared envelope verifier before the record is
    adopted.
    """
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted PR protocol posting requires a TrustedBody.")
    body.validate_for_surface(PR_COMMENT_SURFACE)
    outcome = post_comment_with_reconciliation(
        runner,
        config=config,
        surface=comment_thread_surface(PR_THREAD_SURFACE, pr_number),
        body=str(body),
        write=lambda: runner.run(
            [
                config.gh_cmd,
                "api",
                "--method",
                "POST",
                f"repos/{config.repo}/issues/{pr_number}/comments",
                "-f",
                f"body={body}",
            ],
            cwd=github_api_cwd(),
            check=False,
        ),
        what="Trusted PR protocol record",
    )
    result = outcome.result
    if outcome.adopted is not None:
        # The verifier re-reads the adopted comment by ID, so adoption passes the
        # same stored-body and author checks as a response-carried envelope.
        payload: object = {"id": outcome.adopted.comment_id}
    else:
        assert result is not None
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
    return verify_written_protocol_comment_observed(
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
    """Update one trusted protocol comment and return its verified ID."""
    return patch_verified_trusted_protocol_comment_observed(
        runner,
        config=config,
        comment_id=comment_id,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        surface=surface,
    ).comment_id


def patch_verified_trusted_protocol_comment_observed(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    comment_id: int,
    body: TrustedBody,
    expected_author_login: str | None = None,
    expected_author_id: int | None = None,
    surface: str = PR_COMMENT_SURFACE,
) -> WrittenProtocolComment:
    """Update one trusted protocol comment and verify the stored result.

    A successful exit status is not proof that the new body was persisted, so
    the update response is held to the same read-back contract as a create.
    """
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted protocol update requires a TrustedBody.")
    body.validate_for_surface(surface)
    context = f"Trusted protocol comment update for comment {comment_id}"
    active = DEFAULT_POLICY
    history: list[GitHubAttempt] = []
    payload: object = None
    for number in range(1, active.attempts + 1):
        started = time.monotonic()
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
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError as exc:
                raise AgentLoopError(f"{context} returned invalid JSON.") from exc
            break
        attempt = attempt_from_result(number, result, started)
        history.append(attempt)
        if config.dry_run or attempt.classification == "permanent":
            detail = (result.stderr or result.stdout).strip()
            raise AgentLoopError(
                f"Unable to persist the trusted protocol comment update for comment {comment_id}."
                + (f" {detail}" if detail else "")
            )
        # An update is idempotent by comment ID: re-read the comment and accept
        # it when the stored body already equals the target.  An unreadable
        # comment proves nothing and is never replayed against.
        reread = run_gh_read(
            runner,
            [config.gh_cmd, "api", f"repos/{config.repo}/issues/comments/{comment_id}"],
            cwd=github_api_cwd(),
            check=False,
        )
        envelope: object = None
        if reread.returncode == 0:
            try:
                envelope = json.loads(reread.stdout or "{}")
            except json.JSONDecodeError:
                envelope = None
        if (
            not isinstance(envelope, dict)
            or envelope.get("id") != comment_id
            or not isinstance(envelope.get("body"), str)
        ):
            raise GitHubAmbiguousWriteError(
                _ambiguity_message(
                    what="Protocol comment update", surface=f"comment {comment_id}",
                    body_index=None, outcome="unknown", attempts=history,
                    detail="The comment could not be re-read as a valid envelope for this ID."
                    + _failure_detail(reread),
                ),
                tuple(history),
            )
        if stored_body_matches_posted(envelope.get("body"), str(body))[0]:
            log(config, f"{context}: GitHub applied the update despite a transient error; adopted")
            payload = envelope
            break
        if number < active.attempts:
            github_retry.sleep_before_retry(active, number)
    else:
        raise GitHubTransientExhaustedError(
            _ambiguity_message(
                what="Protocol comment update", surface=f"comment {comment_id}",
                body_index=None, outcome="verified unapplied", attempts=history,
                detail="The retry budget is exhausted.",
            ),
            tuple(history),
        )
    if not isinstance(payload, dict):
        raise AgentLoopError(f"{context} returned no comment envelope.")
    if payload.get("id") is None:
        payload = {**payload, "id": comment_id}
    written = verify_written_protocol_comment_observed(
        runner,
        config=config,
        payload=payload,
        body=body,
        expected_author_login=expected_author_login,
        expected_author_id=expected_author_id,
        context=context,
    )
    if written.comment_id != comment_id:
        raise AgentLoopError(f"{context} returned a different comment identity.")
    return written


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
    outcome = post_comment_with_reconciliation(
        runner,
        config=config,
        surface=comment_thread_surface(PR_THREAD_SURFACE, pr_number),
        body=str(body),
        write=lambda: runner.run(
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
        ),
        what="Expected-closing PR contract record",
    )
    result = outcome.result
    if outcome.adopted is not None:
        return
    assert result is not None
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
    claimed: set[int] = set()
    for index, prepared in enumerate(bodies, start=1):
        prepared.validate_for_surface(ISSUE_COMMENT_SURFACE)
        _post_trusted_protocol_comment(
            runner, config=config, command=["issue", "comment", str(issue_number)], body=prepared,
            claimed=claimed, body_index=index,
        )


def post_verified_trusted_issue_protocol_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: TrustedBody,
    expected_author_login: str,
    expected_author_id: int,
    claimed: set[int] | None = None,
    body_index: int | None = None,
) -> IssueComment:
    """Post one issue protocol record and verify the live server envelope."""
    if not isinstance(body, TrustedBody):
        raise AgentLoopError("Verified trusted issue protocol posting requires a TrustedBody.")
    body.validate_for_surface(ISSUE_COMMENT_SURFACE)
    outcome = post_comment_with_reconciliation(
        runner,
        config=config,
        surface=comment_thread_surface(ISSUE_THREAD_SURFACE, issue_number),
        body=str(body),
        write=lambda: runner.run(
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
        ),
        claimed=claimed,
        body_index=body_index,
        what="Trusted issue protocol record",
    )
    result = outcome.result
    if outcome.adopted is not None:
        payload = _fetch_protocol_comment_envelope(
            runner,
            config=config,
            comment_id=outcome.adopted.comment_id,
            context=f"Trusted issue protocol record for issue #{issue_number}",
        )
    else:
        assert result is not None
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
    matches, footer_observed = stored_body_matches_posted(payload.get("body"), str(body))
    if not matches:
        raise AgentLoopError(
            f"Trusted issue protocol record for issue #{issue_number} returned a different body."
        )
    # The wrapper carries the posted canonical body, never the host-footered
    # stored body, so downstream byte-exact checks stay exact (#1043).
    returned_body = str(body)
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
    if footer_observed:
        note_host_footer_observed(
            config, f"Trusted issue protocol record for issue #{issue_number}"
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
    claimed: set[int] = set()
    for index, prepared in enumerate(bodies, start=1):
        posted = post_verified_trusted_issue_protocol_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=prepared,
            expected_author_login=expected_author_login,
            expected_author_id=expected_author_id,
            claimed=claimed,
            body_index=index,
        )
    if posted is None:  # pragma: no cover - prepare_round_comment always returns an anchor
        raise AgentLoopError("Verified issue round posting produced no comment wrapper.")
    return posted


def _command_comment_surface(command: list[str]) -> str:
    """Name the conversation a ``gh pr|issue comment N`` command writes to."""
    return comment_thread_surface(
        PR_THREAD_SURFACE if command[0] == "pr" else ISSUE_THREAD_SURFACE, int(command[2])
    )


def _post_comment_body(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    command: list[str],
    body: str,
    claimed: set[int] | None = None,
    body_index: int | None = None,
) -> None:
    if len(body) > MAX_GITHUB_BODY_CHARS:
        raise AgentLoopError(f"GitHub comment body exceeds {MAX_GITHUB_BODY_CHARS} characters; shorten the response.")
    if config.dry_run:
        runner.run([config.gh_cmd, *command, "--repo", config.repo, "--body", body], cwd=active_workdir(config))
        return
    # One temp file serves every attempt; it is removed after success or terminal failure.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        outcome = post_comment_with_reconciliation(
            runner,
            config=config,
            surface=_command_comment_surface(command),
            body=str(body),
            write=lambda: runner.run(
                [config.gh_cmd, *command, "--repo", config.repo, "--body-file", path],
                cwd=active_workdir(config),
                check=False,
            ),
            claimed=claimed,
            body_index=body_index,
        )
        if outcome.result is not None and outcome.result.returncode != 0:
            raise AgentLoopError(github_retry._failure_message(outcome.result))
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

    window_start = issue_reconciliation_window_start()
    # One temp file serves every attempt; it is removed after success or terminal failure.
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(rendered_body)
        path = handle.name
    try:
        issue_url = _create_issue_reconciled(
            runner,
            config=config,
            title=title,
            rendered_body=rendered_body,
            path=path,
            window_start=window_start,
        )
    except AgentLoopError as exc:
        exc.window_start = window_start  # type: ignore[attr-defined]
        raise
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
    if issue_url:
        log(config, f"Created GitHub issue: {issue_url}")
    return issue_url


ISSUE_CREATION_IDENTITY_RE = re.compile(
    r"<!--\s*(?:AGENT_FOLLOWUP_CREATION_IDENTITY|AGENT_PLAN_PHASE_IDENTITY|AGENT_SPLIT_CHILD):"
    r"[^>]*?-->",
    re.I,
)


def issue_reconciliation_window_start() -> datetime.datetime:
    return _utc_now() - COMMENT_RECONCILIATION_WINDOW


def creation_identity_records(text: str | None) -> frozenset[str]:
    """Canonical creation-identity records (follow-up, phase or split-child) in ``text``."""
    return frozenset(
        " ".join(match.group(0).split()) for match in ISSUE_CREATION_IDENTITY_RE.finditer(text or "")
    )


def _read_creator_issue_pages(
    runner: Runner, *, config: AgentLoopConfig, creator: str, since: datetime.datetime
) -> list[dict]:
    """Page the creator's issues exhaustively via REST; any defect raises."""
    page_size = 100
    since_text = since.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    items: list[dict] = []
    seen: set[int] = set()
    for page in range(1, 10_001):
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues?creator={creator}&state=all&since={since_text}"
                f"&sort=created&direction=desc&per_page={page_size}&page={page}",
            ],
            cwd=github_api_cwd(),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"Creator issue listing failed on page {page}; a partial view is never used."
                + _failure_detail(result)
            )
        try:
            raw_page = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise AgentLoopError(
                f"Creator issue listing returned malformed JSON on page {page}."
            ) from exc
        if not isinstance(raw_page, list) or len(raw_page) > page_size:
            raise AgentLoopError(
                f"Creator issue listing returned a non-list or oversized page {page}."
            )
        for raw in raw_page:
            if not isinstance(raw, dict):
                raise AgentLoopError(f"Creator issue listing page {page} held a non-object.")
            number = raw.get("number")
            if isinstance(number, bool) or not isinstance(number, int) or number in seen:
                raise AgentLoopError(
                    f"Creator issue listing page {page} held a missing or repeated issue number."
                )
            seen.add(number)
            if "pull_request" not in raw:
                # Every non-PR issue must be fully classifiable: a malformed
                # envelope could be the accepted create, so it makes the whole
                # listing unusable instead of being skipped.
                user = raw.get("user")
                user_id = user.get("id") if isinstance(user, dict) else None
                if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id < 1:
                    raise AgentLoopError(
                        f"Creator issue listing issue #{number} lacks a creator id."
                    )
                parse_comment_timestamp(raw.get("created_at"))
                if not isinstance(raw.get("title"), str):
                    raise AgentLoopError(f"Creator issue listing issue #{number} lacks a title.")
                if "body" not in raw:
                    raise AgentLoopError(f"Creator issue listing issue #{number} omits its body.")
                if raw["body"] is not None and not isinstance(raw["body"], str):
                    raise AgentLoopError(f"Creator issue listing issue #{number} has a non-text body.")
            items.append(raw)
        if len(raw_page) < page_size:
            return items
    raise AgentLoopError("Creator issue listing exceeded its pagination bound.")


def recover_created_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    title: str,
    body: str | TrustedBody,
    window_start: datetime.datetime,
    attempts: Sequence[GitHubAttempt] = (),
) -> str | None:
    """Find the issue an ambiguous create produced, by exact identity (#510).

    Lists the authenticated creator's issues through the exhaustive REST pager
    (never the search index), drops pull-request objects, and keeps issues whose
    immutable creator ID is the actor, created at or after ``window_start`` and
    carrying one of the body's creation-identity records.  Exactly one such issue
    whose title and stored body also match exactly is returned (its URL).  Zero
    identity matches return ``None`` (replay is safe).  An identity match with a
    different title or body, several identity matches, or an unreadable listing
    raise :class:`GitHubAmbiguousWriteError` and nothing is adopted or replayed.
    """
    rendered = str(body)
    identities = creation_identity_records(rendered)
    if not identities:
        return None

    def ambiguous(detail: str) -> GitHubAmbiguousWriteError:
        return GitHubAmbiguousWriteError(
            f"Issue creation {title!r} in {config.repo} is ambiguous: {detail} The create is "
            "not replayed; check the repository for the issue before creating it by hand."
            + (
                "\nGitHub attempt history:\n" + format_attempt_history(attempts)
                if attempts
                else ""
            ),
            tuple(attempts),
        )

    try:
        actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
        listing = _read_creator_issue_pages(
            runner, config=config, creator=actor_login, since=window_start
        )
    except (AgentLoopError, OSError) as exc:
        raise ambiguous(f"the creator listing was incomplete ({exc}).") from exc
    matches: list[dict] = []
    for raw in listing:
        if "pull_request" in raw:
            continue
        user = raw.get("user")
        if not isinstance(user, dict) or user.get("id") != actor_id:
            continue
        created = parse_comment_timestamp(raw.get("created_at"))
        if created < window_start:
            continue
        if not (creation_identity_records(raw.get("body") or "") & identities):
            continue
        matches.append(raw)
    if not matches:
        return None
    if len(matches) > 1:
        numbers = ", ".join(f"#{m.get('number')}" for m in matches)
        raise ambiguous(f"{len(matches)} issues carry the same creation identity ({numbers}).")
    match = matches[0]
    exact_title = match.get("title") == title
    exact_body, _footer = stored_body_matches_posted(match.get("body"), rendered)
    if not (exact_title and exact_body):
        raise ambiguous(
            f"issue #{match.get('number')} carries the creation identity but its stored "
            "title or body differs from this create."
        )
    url = match.get("html_url")
    if not isinstance(url, str) or not url:
        raise ambiguous(f"issue #{match.get('number')} lacks an html_url.")
    return url


def _create_issue_reconciled(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    title: str,
    rendered_body: str,
    path: str,
    window_start: datetime.datetime,
) -> str | None:
    command = [
        config.gh_cmd, "issue", "create", "--repo", config.repo,
        "--title", title, "--body-file", path,
    ]
    active = DEFAULT_POLICY
    history: list[GitHubAttempt] = []
    for number in range(1, active.attempts + 1):
        started = time.monotonic()
        result = runner.run(command, cwd=active_workdir(config), check=False)
        if result.returncode == 0:
            return result.stdout.strip() or None
        attempt = attempt_from_result(number, result, started)
        history.append(attempt)
        if attempt.classification == "permanent":
            raise AgentLoopError(github_retry._failure_message(result))
        if not creation_identity_records(rendered_body):
            raise GitHubAmbiguousWriteError(
                f"Issue creation {title!r} failed with a transient GitHub error and the "
                "body carries no creation identity, so an accepted create cannot be told "
                "from an absent one; it is not replayed.\nGitHub attempt history:\n"
                + format_attempt_history(history),
                tuple(history),
            )
        recovered = recover_created_issue(
            runner, config=config, title=title, body=rendered_body,
            window_start=window_start, attempts=history,
        )
        if recovered is not None:
            log(
                config,
                f"Issue creation {title!r}: GitHub accepted the create despite a transient "
                f"error; adopted {recovered}",
            )
            return recovered
        if number < active.attempts:
            github_retry.sleep_before_retry(active, number)
    raise GitHubTransientExhaustedError(
        f"Issue creation {title!r} failed after {len(history)} transient attempt(s) and the "
        "issue is verified absent.\nGitHub attempt history:\n" + format_attempt_history(history),
        tuple(history),
    )


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
    result = run_gh_read(
        runner,
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
    history = getattr(config, "execution_history", None)
    if history is not None:
        found = [issue for issue in found if issue.number not in history.child_numbers]
    return tuple(found)


def get_issue_found(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
) -> FoundIssue | None:
    """Read one issue by number as a `FoundIssue`, or None when unavailable.

    Used to resolve child references recorded in a decomposition summary
    without depending on a title search. Dry-run returns None, matching
    `search_issues`; unreadable or malformed responses also return None.
    """
    if isinstance(issue_number, bool) or not isinstance(issue_number, int) or issue_number <= 0:
        raise AgentLoopError("get_issue_found issue_number must be a positive integer")
    if config.dry_run:
        return None
    try:
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "issue",
                "view",
                str(issue_number),
                "--repo",
                config.repo,
                "--json",
                "number,title,url,body",
            ],
            cwd=active_workdir(config),
        )
    except AgentLoopError:
        return None
    raw = (result.stdout or "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    number = payload.get("number")
    if isinstance(number, bool) or number != issue_number:
        return None
    return FoundIssue(
        number=number,
        title=_optional_str(payload.get("title")),
        url=_optional_str(payload.get("url")),
        body=_optional_str(payload.get("body")),
    )


def get_pr_head_sha(runner: Runner, config: AgentLoopConfig, pr_number: int) -> str:
    result = run_gh_read(
        runner,
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


def get_pr_merge_commit_sha(
    runner: Runner, config: AgentLoopConfig, pr_number: int
) -> str:
    """Return a merged PR's merge commit, failing closed when it cannot be read.

    A durable record that names the commit must not be published without it:
    a record written with a gap would be taken as complete by later runs.
    """
    result = run_gh_read(
        runner,
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "mergeCommit",
        ],
        cwd=active_workdir(config),
        check=False,
    )
    unreadable = f"Unable to read the merge commit of PR #{pr_number}."
    if result.returncode != 0:
        raise AgentLoopError(unreadable + _failure_detail(result))
    try:
        data = json.loads(result.stdout or "null")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(unreadable) from exc
    commit = data.get("mergeCommit") if isinstance(data, dict) else None
    oid = commit.get("oid") if isinstance(commit, dict) else None
    if isinstance(oid, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", oid):
        return oid
    raise AgentLoopError(unreadable)


@dataclass(frozen=True)
class CiWatchOutcome:
    """Terminal result from the full-board post-approval watcher."""

    status: Literal[
        "passed",
        "not_started",
        "failed",
        "plumbing_failure",
        "timeout",
        "infrastructure_stall",
        "merge_conflict",
        "head_changed",
        "protection_unreadable",
        "dry_run",
    ]
    pr_checks: PullRequestChecks | None = None
    failed_checks: tuple[PullRequestCheck, ...] = ()
    mergeability: PullRequestMergeability | None = None
    head_sha: str | None = None
    stall: CiInfrastructureStall | None = None
    attempts_used: int = 0
    # Plumbing classification (#1477).  ``stuck_checks`` are required statuses
    # confirmed pending after their producing run completed; ``hold_reason``
    # names why a timeout expired while a plumbing failure was being held.
    classification: WatchFailureClass | None = None
    stuck_checks: tuple[PullRequestCheck, ...] = ()
    hold_reason: Literal["plumbing-awaiting-tests", "stale-after-rerun"] | None = None
    run_urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlumbingRerunState:
    """Invocation-local record of the single plumbing rerun for one head."""

    head_sha: str
    run_id: int
    run_url: str | None
    prior_attempt: int
    # ("check", check_id) and ("status", context, run_id, created_at) keys of
    # the pre-rerun observations; they stay stale until a new one replaces them.
    stale_keys: frozenset[tuple[object, ...]] = frozenset()


@dataclass(frozen=True)
class OrdinaryRerunResult:
    status: Literal["rerun_started", "unresolvable", "head_mismatch", "command_failed", "dry_run", "expired"]
    run_id: int | None = None
    run_url: str | None = None
    prior_attempt: int | None = None
    stale_keys: frozenset[tuple[object, ...]] = frozenset()
    reason: str = ""

    def state(self, head_sha: str) -> PlumbingRerunState | None:
        if self.status != "rerun_started" or self.run_id is None or self.prior_attempt is None:
            return None
        return PlumbingRerunState(
            head_sha=head_sha,
            run_id=self.run_id,
            run_url=self.run_url,
            prior_attempt=self.prior_attempt,
            stale_keys=self.stale_keys,
        )


def plumbing_observation_key(check: PullRequestCheck) -> tuple[object, ...] | None:
    """Stable identity of one failed/stuck observation for stale filtering."""
    if check.kind == "check_run":
        return ("check", check.check_id) if check.check_id is not None else None
    return ("status", check.name, _extract_run_id(check.url), check.created_at)


def read_actions_run(runner: Runner, config: AgentLoopConfig, run_id: int | str) -> dict | None:
    """Read one Actions run (read-retried); ``None`` when unreadable."""
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", f"repos/{config.repo}/actions/runs/{run_id}"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or str(payload.get("id")) != str(run_id):
        return None
    return payload


def _run_id_for_check_suite(runner: Runner, config: AgentLoopConfig, suite_id: int) -> str | None:
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", f"repos/{config.repo}/actions/runs?check_suite_id={suite_id}"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError:
        return None
    runs = payload.get("workflow_runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or len(runs) != 1 or not isinstance(runs[0], dict):
        return None
    run_id = runs[0].get("id")
    return str(run_id) if isinstance(run_id, int) and not isinstance(run_id, bool) else None


def rerun_ordinary_failed_jobs(
    runner: Runner,
    config: AgentLoopConfig,
    observations: Sequence[PullRequestCheck],
    expected_head_sha: str | None,
    plumbing_names: frozenset[str] = frozenset(),
    deadline: float | None = None,
) -> OrdinaryRerunResult:
    """Re-run the failed jobs of the one ordinary run behind ``observations``.

    Every observation resolves through one path: a check run by its
    ``check_suite_id``, a status context by its target-URL Actions run.  The
    single resolved run must be completed, for the approved head, and not a
    ``workflow_dispatch`` run before ``gh run rerun --failed`` is issued.
    """
    if config.dry_run:
        return OrdinaryRerunResult(status="dry_run", reason="dry-run never re-runs CI")
    if not observations or not expected_head_sha:
        return OrdinaryRerunResult(status="unresolvable", reason="no plumbing observation to resolve")
    run_ids: set[str] = set()
    for check in observations:
        if check.kind == "check_run":
            run_id = (
                _run_id_for_check_suite(runner, config, check.check_suite_id)
                if check.check_suite_id is not None
                else None
            )
        else:
            run_id = _extract_run_id(check.url)
        if run_id is None:
            return OrdinaryRerunResult(
                status="unresolvable",
                reason=f"{check.name} has no resolvable GitHub Actions run",
            )
        run_ids.add(run_id)
    if len(run_ids) != 1:
        return OrdinaryRerunResult(
            status="unresolvable",
            reason="plumbing failures belong to different runs: " + ", ".join(sorted(run_ids)),
        )
    (run_id_text,) = run_ids
    run = read_actions_run(runner, config, run_id_text)
    if run is None:
        return OrdinaryRerunResult(status="unresolvable", reason=f"run {run_id_text} is unreadable")
    run_url = run.get("html_url") if isinstance(run.get("html_url"), str) else None
    run_id = int(run_id_text)
    attempt = run.get("run_attempt")
    if run.get("head_sha") != expected_head_sha or run.get("event") == "workflow_dispatch":
        return OrdinaryRerunResult(
            status="head_mismatch", run_id=run_id, run_url=run_url,
            reason=(
                f"run {run_id} is for {run.get('head_sha') or 'an unknown head'} "
                f"({run.get('event') or 'unknown event'}), not an ordinary run of {expected_head_sha}"
            ),
        )
    if run.get("status") != "completed" or not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        return OrdinaryRerunResult(
            status="unresolvable", run_id=run_id, run_url=run_url,
            reason=f"run {run_id} is not a completed run with a readable attempt",
        )
    # The run's own job listing must show a failed job for --failed to retry,
    # and every failed job must be plumbing; otherwise a rerun is unsafe.
    jobs_class = classify_run_jobs(read_run_jobs(runner, config, run_id), plumbing_names)
    if jobs_class != "plumbing-only":
        reason = {
            "no-failed-jobs": f"run {run_id} has no failed job for a rerun to retry",
            "test-failure": f"run {run_id} has a failed non-plumbing job",
            "not-classifiable": f"run {run_id} has no readable complete job listing",
        }[jobs_class]
        return OrdinaryRerunResult(
            status="unresolvable", run_id=run_id, run_url=run_url, prior_attempt=attempt, reason=reason,
        )
    if deadline is not None and time.monotonic() >= deadline:
        # The resolver and job reads consumed the shared watch budget.
        return OrdinaryRerunResult(
            status="expired", run_id=run_id, run_url=run_url, prior_attempt=attempt,
            reason="the shared CI watch deadline passed before the rerun could be issued",
        )
    result = runner.run(
        [config.gh_cmd, "run", "rerun", str(run_id), "--failed", "--repo", config.repo],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return OrdinaryRerunResult(
            status="command_failed", run_id=run_id, run_url=run_url, prior_attempt=attempt,
            reason=f"gh run rerun failed{_failure_detail(result)}",
        )
    stale = frozenset(
        key for key in (plumbing_observation_key(check) for check in observations) if key is not None
    )
    return OrdinaryRerunResult(
        status="rerun_started", run_id=run_id, run_url=run_url, prior_attempt=attempt,
        stale_keys=stale,
    )


def board_protection_is_reliable(
    checks: PullRequestChecks,
    mergeability: PullRequestMergeability | None = None,
    *,
    head_sha: str | None = None,
) -> bool:
    """Return whether the board's required-check set is known well enough.

    Readable classic protection (``configured``) or its confirmed absence
    (``not_found``) lets the observed board stand on its own.  A classic
    protection 403 (``forbidden``) hides the required contexts, so the board
    is trusted only when GitHub's own merge-state computation for the same
    head is ``CLEAN``: GitHub derives it from the real protection, including
    required contexts this token cannot read.  The merge itself still pins the
    head, and GitHub enforces protection again at merge time.
    """
    status = checks.branch_protection_status
    if status in {"configured", "not_found"}:
        return True
    if status != "forbidden" or mergeability is None:
        return False
    return (
        mergeability.state == "mergeable"
        and mergeability.merge_state_raw == "CLEAN"
        and head_sha is not None
        and mergeability.head_sha == head_sha
    )


def protection_awaits_readiness(
    checks: PullRequestChecks,
    mergeability: PullRequestMergeability | None,
    *,
    head_sha: str | None,
) -> bool:
    """Return whether an unreadable protection can only be judged after readiness.

    GitHub reports ``DRAFT`` rather than ``CLEAN`` for a draft PR, so a draft
    under a classic-protection 403 cannot prove its required checks yet.  This
    is never merge evidence by itself: the caller must mark the PR ready and
    then require ``CLEAN`` for the same head via
    :func:`board_protection_is_reliable` before merging.
    """
    return (
        checks.branch_protection_status == "forbidden"
        and mergeability is not None
        and mergeability.state == "mergeable"
        and mergeability.merge_state_raw == "DRAFT"
        and head_sha is not None
        and mergeability.head_sha == head_sha
    )


def _observed_board_complete(checks: PullRequestChecks) -> bool:
    return (
        checks.check_query_status == "ok"
        and checks.state == "passing"
        and not checks.pending
        and not checks.missing_required
    )


def watch_pr_checks(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    *,
    metadata: PullRequestMetadata,
    deadline: float | None = None,
    attempts: int | None = None,
    plumbing_names: frozenset[str] | None = None,
    rerun_state: PlumbingRerunState | None = None,
) -> CiWatchOutcome:
    """Synchronously watch the complete current-head check board.

    It has no worker or persisted process; interrupting the foreground runner
    stops it immediately and a later invocation simply fetches fresh state.

    With ``plumbing_names`` a failure is classified before it is reported
    (#1477): a fresh test failure returns ``failed``; a complete board whose
    failures are all plumbing returns ``plumbing_failure``; plumbing that
    failed while tests still run, and evidence from the attempt before
    ``rerun_state``'s rerun, are polled through under the same budget.
    """
    if config.dry_run:
        return CiWatchOutcome(status="dry_run", head_sha=metadata.head_sha)
    deadline = deadline if deadline is not None else time.monotonic() + config.ci_timeout_seconds
    limit = attempts if attempts is not None else max(
        1, config.ci_timeout_seconds // config.ci_poll_interval_seconds
    )
    latest: PullRequestChecks | None = None
    empty_attempts = 0
    unverified_green_attempts = 0
    startup_attempt_limit = max(
        1,
        (config.ci_startup_timeout_seconds + config.ci_poll_interval_seconds - 1)
        // config.ci_poll_interval_seconds,
    )
    stuck_tracker: dict[str, tuple[int, int]] = {}
    stuck: tuple[PullRequestCheck, ...] = ()
    hold_reason: Literal["plumbing-awaiting-tests", "stale-after-rerun"] | None = None
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
        reliable = snapshot.check_query_status == "ok" and board_protection_is_reliable(
            snapshot, mergeability, head_sha=current_head,
        )
        # A positively observed failure is actionable even when another query
        # was partial or a required check has not materialized yet. Those
        # conditions can delay a passing decision, but they must not hide a
        # concrete failing check from the coder.
        hold_reason = None
        if plumbing_names is not None:
            (
                stuck, stuck_without_failed_jobs, stuck_with_test_failures, stuck_attempts,
            ) = _confirm_stuck_statuses(
                runner, config, snapshot, plumbing_names, stuck_tracker
            )
            if snapshot.state == "failing" or stuck:
                fresh_failing = tuple(
                    check for check in snapshot.failing
                    if not observation_is_stale(check, rerun_state, None)
                )
                fresh_stuck = tuple(
                    check for check in stuck
                    if not observation_is_stale(check, rerun_state, stuck_attempts.get(check.name))
                )
                if rerun_state is not None and not fresh_failing and not fresh_stuck:
                    # Prior-attempt evidence: wait for the rerun's own result.
                    hold_reason = "stale-after-rerun"
                else:
                    classification = classify_watch_failure(
                        replace(snapshot, failing=fresh_failing),
                        plumbing_names,
                        fresh_stuck,
                        stuck_without_failed_jobs=tuple(stuck_without_failed_jobs),
                        stuck_with_test_failures=tuple(stuck_with_test_failures),
                    )
                    if classification == "test-failure" or (
                        classification == "not-classifiable" and fresh_failing
                    ):
                        return CiWatchOutcome(
                            status="failed", pr_checks=snapshot, failed_checks=snapshot.failing,
                            head_sha=current_head, attempts_used=attempt + 1,
                            classification=classification, stuck_checks=stuck,
                        )
                    if classification in {"plumbing-only", "stuck-infrastructure"}:
                        return CiWatchOutcome(
                            status="plumbing_failure", pr_checks=snapshot,
                            failed_checks=fresh_failing, head_sha=current_head,
                            attempts_used=attempt + 1, classification=classification,
                            stuck_checks=fresh_stuck,
                            run_urls=observation_run_urls((*fresh_failing, *fresh_stuck)),
                        )
                    if classification == "plumbing-awaiting-tests":
                        hold_reason = "plumbing-awaiting-tests"
        elif snapshot.state == "failing":
            return CiWatchOutcome(
                status="failed", pr_checks=snapshot, failed_checks=snapshot.failing,
                head_sha=current_head, attempts_used=attempt + 1,
            )
        if snapshot.state == "passing" and reliable and not snapshot.pending and not snapshot.missing_required:
            return CiWatchOutcome(
                status="passed", pr_checks=snapshot, mergeability=mergeability,
                head_sha=current_head, attempts_used=attempt + 1,
            )
        if snapshot.branch_protection_status == "forbidden" and _observed_board_complete(snapshot):
            # Every observed check passed, yet GitHub does not report a CLEAN
            # merge state and the required set is unreadable.  Allow the same
            # bounded window used for a board that has not materialized (an
            # unseen required check may still be starting, or GitHub may still
            # be recomputing), then stop instead of polling to the timeout.
            unverified_green_attempts += 1
            if unverified_green_attempts >= startup_attempt_limit:
                return CiWatchOutcome(
                    status="protection_unreadable", pr_checks=snapshot,
                    mergeability=mergeability, head_sha=current_head,
                    attempts_used=attempt + 1,
                )
        else:
            unverified_green_attempts = 0
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
            # A budget that expires on a complete green board which only the
            # unreadable protection keeps from passing reports that cause,
            # not a generic timeout.
            return CiWatchOutcome(
                status="protection_unreadable" if unverified_green_attempts else "timeout",
                pr_checks=latest,
                mergeability=mergeability if unverified_green_attempts else None,
                head_sha=current_head,
                attempts_used=attempt + 1,
                stuck_checks=stuck,
                hold_reason=hold_reason,
                run_urls=(
                    observation_run_urls((*snapshot.failing, *stuck))
                    if hold_reason is not None else ()
                ),
            )
        runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=active_workdir(config))
    raise AssertionError("CI watch loop must return a terminal outcome")


def _confirm_stuck_statuses(
    runner: Runner,
    config: AgentLoopConfig,
    snapshot: PullRequestChecks,
    plumbing_names: frozenset[str],
    tracker: dict[str, tuple[int, int]],
) -> tuple[tuple[PullRequestCheck, ...], set[str], set[str], dict[str, int]]:
    """Confirm required statuses left pending after their run completed.

    A candidate is a pending status context that is required (or a catalogued
    plumbing status) whose target URL names an Actions run.  The same
    (context, run, attempt) must read ``completed`` on two consecutive polls.
    An unreadable run, an unreadable job listing, or a status without a run URL
    stays ordinary pending.  The producing run's complete job listing (never
    its aggregate conclusion) decides whether a rerun has a failed plumbing job
    to retry, nothing to retry, or a failed test job that keeps precedence.
    """
    required = set(snapshot.required_checks)
    confirmed: list[PullRequestCheck] = []
    without_failed_jobs: set[str] = set()
    with_test_failures: set[str] = set()
    attempts: dict[str, int] = {}
    seen: dict[str, tuple[int, int]] = {}
    for check in snapshot.pending:
        if check.kind != "status_context":
            continue
        if check.name not in required and not check_name_matches(check.name, plumbing_names):
            continue
        run_id = _extract_run_id(check.url)
        if run_id is None:
            continue
        run = read_actions_run(runner, config, run_id)
        attempt = run.get("run_attempt") if run is not None else None
        if (
            run is None
            or run.get("status") != "completed"
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
        ):
            continue
        key = (int(run_id), attempt)
        seen[check.name] = key
        if tracker.get(check.name) != key:
            continue
        jobs_class = classify_run_jobs(read_run_jobs(runner, config, run_id), plumbing_names)
        if jobs_class == "not-classifiable":
            continue
        confirmed.append(replace(check, run_id=run_id))
        attempts[check.name] = attempt
        if jobs_class == "no-failed-jobs":
            without_failed_jobs.add(check.name)
        elif jobs_class == "test-failure":
            with_test_failures.add(check.name)
    tracker.clear()
    tracker.update(seen)
    return tuple(confirmed), without_failed_jobs, with_test_failures, attempts


def read_run_jobs(
    runner: Runner, config: AgentLoopConfig, run_id: int | str | None
) -> tuple[tuple[str, str, str], ...] | None:
    """Return a run's latest-attempt jobs as (name, conclusion, url).

    ``None`` when the listing is unreadable or incomplete (``total_count``
    disagrees with the jobs read), so callers fail closed.
    """
    if run_id is None:
        return None
    result = run_gh_read(
        runner,
        [config.gh_cmd, "api", "--paginate", f"repos/{config.repo}/actions/runs/{run_id}/jobs?filter=latest&per_page=100"],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        return None
    # Without --slurp, gh 2.45 concatenates object pages; decode every page.
    pages = decode_concatenated_json_objects(result.stdout or "")
    if pages is None:
        return None
    jobs: list[tuple[str, str, str]] = []
    total: int | None = None
    for payload in pages:
        page_jobs = payload.get("jobs")
        if not isinstance(page_jobs, list):
            return None
        page_total = payload.get("total_count")
        if isinstance(page_total, int) and not isinstance(page_total, bool):
            total = page_total
        for job in page_jobs:
            if not isinstance(job, dict):
                return None
            name = job.get("name") if isinstance(job.get("name"), str) else "unnamed job"
            conclusion = str(job.get("conclusion") or job.get("status") or "").lower()
            url = job.get("html_url") if isinstance(job.get("html_url"), str) else ""
            jobs.append((name, conclusion, url))
    if total is not None and total != len(jobs):
        return None
    return tuple(jobs)


def observation_is_stale(
    check: PullRequestCheck,
    rerun_state: PlumbingRerunState | None,
    stuck_attempt: int | None,
) -> bool:
    if rerun_state is None:
        return False
    if stuck_attempt is not None:
        return (
            _extract_run_id(check.url) == str(rerun_state.run_id)
            and stuck_attempt <= rerun_state.prior_attempt
        )
    key = plumbing_observation_key(check)
    return key is not None and key in rerun_state.stale_keys


def observation_run_urls(checks: Sequence[PullRequestCheck]) -> tuple[str, ...]:
    urls: list[str] = []
    for check in checks:
        run_id = check.run_id or _extract_run_id(check.url)
        if not check.url:
            continue
        url = check.url
        if run_id is not None and f"/actions/runs/{run_id}" in url:
            url = url[: url.index(f"/actions/runs/{run_id}") + len(f"/actions/runs/{run_id}")]
        if url not in urls:
            urls.append(url)
    return tuple(urls)


def _read_pr_view_json(
    runner: Runner, *, config: AgentLoopConfig, pr_number: int, fields: str
) -> tuple[dict | None, str]:
    """Read ``gh pr view --json`` (read-retried); ``(None, reason)`` when unusable."""
    result = run_gh_read(
        runner,
        [config.gh_cmd, "pr", "view", str(pr_number), "--repo", config.repo, "--json", fields],
        cwd=active_workdir(config),
        check=False,
    )
    if result.returncode != 0:
        return None, describe_gh_failure(result) or f"exit {result.returncode}"
    try:
        payload = json.loads(result.stdout or "")
    except json.JSONDecodeError:
        return None, "malformed JSON"
    if not isinstance(payload, dict):
        return None, "non-object payload"
    return payload, ""


def reconciled_pr_ready(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    expected_head_sha: str | None,
    undo: bool = False,
) -> CommandResult:
    """Run ``gh pr ready`` (or ``--undo``) with live-state reconciliation (#510).

    The caller supplies the head it verified before the write.  After a
    transient failure the draft state and head are re-read: the desired state
    at the expected head is success; a changed head or an unreadable PR leaves
    the failed result for the caller's existing fail-closed path (no replay);
    the undesired state at the same head is replayed.  Without an expected head
    or in dry-run the command is a single attempt, as before.
    """
    command = [config.gh_cmd, "pr", "ready"]
    if undo:
        command.append("--undo")
    command.extend([str(pr_number), "--repo", config.repo])
    cwd = active_workdir(config)
    if config.dry_run or not expected_head_sha:
        return runner.run(command, cwd=cwd, check=False)
    active = DEFAULT_POLICY
    history: list[GitHubAttempt] = []
    result: CommandResult | None = None

    def failed(final: CommandResult, exhausted: bool = False) -> CommandResult:
        return github_retry._with_history(final, history, exhausted=exhausted)

    for number in range(1, active.attempts + 1):
        started = time.monotonic()
        result = runner.run(command, cwd=cwd, check=False)
        if result.returncode == 0:
            return result
        if classify_gh_failure(result) == "permanent":
            return result
        history.append(attempt_from_result(number, result, started))
        state, reason = _read_pr_view_json(
            runner, config=config, pr_number=pr_number, fields="isDraft,headRefOid"
        )
        if state is None:
            log(config, f"PR #{pr_number}: ready transition is ambiguous and unreadable ({reason}); not replayed")
            return failed(result)
        is_draft, head = state.get("isDraft"), state.get("headRefOid")
        if not isinstance(is_draft, bool) or not isinstance(head, str):
            return failed(result)
        if head != expected_head_sha:
            log(config, f"PR #{pr_number}: head changed during an ambiguous ready transition; not replayed")
            return failed(result)
        if is_draft == undo:
            log(config, f"PR #{pr_number}: ready transition was accepted despite a transient error")
            return github_retry.as_success(result)
        if number < active.attempts:
            github_retry.sleep_before_retry(active, number)
    assert result is not None
    return failed(result, exhausted=True)


def merge_pr(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    *,
    expected_head_sha: str | None = None,
) -> None:
    """Merge with ``--match-head-commit``; a merge is never replayed in-process (#510).

    After a transient failure the PR is re-read.  Merged at the expected head is
    success.  Any other state is a terminal error naming it: an open PR at the
    same head says the merge did not happen and a rerun re-executes every
    source-specific gate before any new attempt.  Without an expected head, or in
    dry-run, the merge is a single attempt with no reconciliation.
    """
    log(config, f"Merging PR #{pr_number}")
    command = [config.gh_cmd, "pr", "merge", str(pr_number), "--repo", config.repo, "--merge"]
    if expected_head_sha:
        command.extend(["--match-head-commit", expected_head_sha])
    if config.dry_run or not expected_head_sha:
        runner.run(command, cwd=active_workdir(config))
        return
    started = time.monotonic()
    result = runner.run(command, cwd=active_workdir(config), check=False)
    if result.returncode == 0:
        return
    attempt = attempt_from_result(1, result, started)
    if attempt.classification == "permanent":
        raise AgentLoopError(github_retry._failure_message(result))
    state, reason = _read_pr_view_json(
        runner, config=config, pr_number=pr_number, fields="state,mergedAt,headRefOid,mergeCommit"
    )
    history = "\nGitHub attempt history:\n" + format_attempt_history([attempt])
    base = f"Merge of PR #{pr_number} (expected head {expected_head_sha}) failed with a transient GitHub error"
    if state is None:
        raise GitHubAmbiguousWriteError(
            f"{base} and the PR state could not be re-read ({reason}); the merge outcome is "
            "unknown and it is not replayed. Inspect the PR, then rerun." + history,
            (attempt,),
        )
    observed, head = state.get("state"), state.get("headRefOid")
    if observed == "MERGED" and head == expected_head_sha:
        log(config, f"PR #{pr_number}: merge was accepted at {expected_head_sha} despite a transient error; recovered")
        return
    if observed == "OPEN" and head == expected_head_sha:
        detail = (
            "the PR is still open at the expected head, so the merge did not happen. It is not "
            "replayed in-process; rerun to re-execute every merge gate (CI authority, "
            "supersession, evidence freeze, managed qualification, ordinary-recovery "
            "capability) before any new match-head-commit attempt."
        )
    else:
        detail = (
            f"the PR is now {observed} at head {head}; this is not the expected head "
            f"{expected_head_sha} merged, so the merge is refused."
        )
    raise AgentLoopError(f"{base}; {detail}" + history)


# ---------------------------------------------------------------------------
# Read-only outcome-history reads for ``agent-loop history export`` (#1433).
# Every function here is a read; none posts, edits or labels anything.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepositoryIssueSummary:
    """One issue (never a PR) from an exhaustive repository listing."""

    number: int
    title: str
    created_at: str
    closed_at: str | None
    state: str


@dataclass(frozen=True)
class LinkedPullRequest:
    """A PR GitHub links to an issue as closing it, with its final state."""

    number: int
    state: str
    merged_at: str | None
    closed_at: str | None


@dataclass(frozen=True)
class IssueOutcomeHistory:
    """An issue's text history and linked PRs.

    ``body_revisions`` is every body version, oldest first, as
    ``(effective_at, body)`` (the creation version included); ``()`` means the
    body was never edited and ``None`` means the edit history is missing or
    incomplete.  ``title_renames`` holds ``(at, previous, current)`` events,
    oldest first, with the same ``None`` meaning.
    """

    number: int
    title: str
    body: str
    state: str
    created_at: str
    closed_at: str | None
    body_revisions: tuple[tuple[str, str], ...] | None
    title_renames: tuple[tuple[str, str, str], ...] | None
    linked_prs: tuple[LinkedPullRequest, ...]


def list_repository_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    since: datetime.datetime | None = None,
) -> tuple[RepositoryIssueSummary, ...]:
    """Page every issue in the repository (PRs excluded); any defect raises.

    ``since`` is GitHub's updated-since filter, which only narrows the read.
    """
    page_size = 100
    since_query = (
        "&since=" + since.astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if since is not None
        else ""
    )
    issues: dict[int, RepositoryIssueSummary] = {}
    for page in range(1, 10_001):
        result = run_gh_read(
            runner,
            [
                config.gh_cmd,
                "api",
                f"repos/{config.repo}/issues?state=all&per_page={page_size}&page={page}"
                + since_query,
            ],
            cwd=github_api_cwd(),
            check=False,
        )
        if result.returncode != 0:
            raise AgentLoopError(
                f"Issue listing of {config.repo} failed on page {page}." + _failure_detail(result)
            )
        try:
            raw_page = json.loads(result.stdout or "")
        except json.JSONDecodeError as exc:
            raise AgentLoopError(
                f"Issue listing of {config.repo} returned malformed JSON on page {page}."
            ) from exc
        if not isinstance(raw_page, list) or len(raw_page) > page_size:
            raise AgentLoopError(
                f"Issue listing of {config.repo} returned an unusable page {page}."
            )
        for raw in raw_page:
            if not isinstance(raw, dict):
                raise AgentLoopError(f"Issue listing of {config.repo} returned a malformed item.")
            if "pull_request" in raw:
                continue
            number = raw.get("number")
            created_at = raw.get("created_at")
            if (
                isinstance(number, bool)
                or not isinstance(number, int)
                or number < 1
                or not isinstance(created_at, str)
            ):
                raise AgentLoopError(f"Issue listing of {config.repo} returned a malformed issue.")
            issues[number] = RepositoryIssueSummary(
                number=number,
                title=_optional_str(raw.get("title")) or "",
                created_at=created_at,
                closed_at=_optional_str(raw.get("closed_at")),
                state=_optional_str(raw.get("state")) or "",
            )
        if len(raw_page) < page_size:
            break
    else:
        raise AgentLoopError(f"Issue listing of {config.repo} exceeded its pagination bound.")
    return tuple(issues[number] for number in sorted(issues))


_ISSUE_OUTCOME_HISTORY_QUERY = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      number title body state createdAt closedAt lastEditedAt includesCreatedEdit
      userContentEdits(first: 100) {
        totalCount
        nodes { editedAt deletedAt diff }
      }
      timelineItems(first: 100, itemTypes: [RENAMED_TITLE_EVENT]) {
        totalCount
        nodes { ... on RenamedTitleEvent { createdAt previousTitle currentTitle } }
      }
      closedByPullRequestsReferences(first: 100, includeClosedPrs: true) {
        totalCount
        nodes { number state mergedAt closedAt }
      }
    }
  }
}
""".strip()


def _parse_body_revisions(issue: dict) -> tuple[tuple[str, str], ...] | None:
    edits = issue.get("userContentEdits")
    if not isinstance(edits, dict):
        return None
    nodes = edits.get("nodes")
    total = edits.get("totalCount")
    if not isinstance(nodes, list) or not isinstance(total, int):
        return None
    if total == 0 and not nodes:
        return () if issue.get("lastEditedAt") is None else None
    if total != len(nodes) or issue.get("includesCreatedEdit") is not True:
        return None
    revisions: list[tuple[str, str]] = []
    for node in nodes:
        if not isinstance(node, dict) or node.get("deletedAt") is not None:
            return None
        edited_at, diff = node.get("editedAt"), node.get("diff")
        if not isinstance(edited_at, str) or not isinstance(diff, str):
            return None
        revisions.append((edited_at, diff))
    revisions.sort(key=lambda item: parse_comment_timestamp(item[0]))
    return tuple(revisions)


def _parse_title_renames(issue: dict) -> tuple[tuple[str, str, str], ...] | None:
    timeline = issue.get("timelineItems")
    if not isinstance(timeline, dict):
        return None
    nodes = timeline.get("nodes")
    total = timeline.get("totalCount")
    if not isinstance(nodes, list) or not isinstance(total, int) or total != len(nodes):
        return None
    renames: list[tuple[str, str, str]] = []
    for node in nodes:
        if not isinstance(node, dict):
            return None
        at, previous, current = (
            node.get("createdAt"), node.get("previousTitle"), node.get("currentTitle")
        )
        if not all(isinstance(value, str) for value in (at, previous, current)):
            return None
        renames.append((at, previous, current))
    renames.sort(key=lambda item: parse_comment_timestamp(item[0]))
    return tuple(renames)


def read_issue_outcome_history(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
) -> IssueOutcomeHistory:
    """Read one issue's edit history and closing PRs in a single GraphQL read."""
    owner, name = _graphql_repository_parts(config.repo)
    result = run_gh_read(
        runner,
        [
            config.gh_cmd, "api", "graphql",
            "-f", f"query={_ISSUE_OUTCOME_HISTORY_QUERY}",
            "-f", f"owner={owner}",
            "-f", f"name={name}",
            "-F", f"number={issue_number}",
        ],
        cwd=github_api_cwd(),
        check=False,
    )
    if result.returncode != 0:
        raise AgentLoopError(
            f"Outcome-history read of issue #{issue_number} failed." + _failure_detail(result)
        )
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise AgentLoopError(
            f"Outcome-history read of issue #{issue_number} returned malformed JSON."
        ) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    repository = data.get("repository") if isinstance(data, dict) else None
    issue = repository.get("issue") if isinstance(repository, dict) else None
    if not isinstance(issue, dict) or (isinstance(payload, dict) and payload.get("errors")):
        raise AgentLoopError(f"Outcome-history read of issue #{issue_number} returned no issue.")
    created_at = issue.get("createdAt")
    if not isinstance(created_at, str):
        raise AgentLoopError(f"Outcome-history read of issue #{issue_number} lacks createdAt.")
    references = issue.get("closedByPullRequestsReferences")
    nodes = references.get("nodes") if isinstance(references, dict) else None
    total = references.get("totalCount") if isinstance(references, dict) else None
    if not isinstance(nodes, list) or not isinstance(total, int) or total != len(nodes):
        raise AgentLoopError(
            f"Outcome-history read of issue #{issue_number} returned an incomplete "
            "linked-PR list; a partial list is never used."
        )
    linked: dict[int, LinkedPullRequest] = {}
    for node in nodes:
        number = node.get("number") if isinstance(node, dict) else None
        if isinstance(number, bool) or not isinstance(number, int):
            raise AgentLoopError(
                f"Outcome-history read of issue #{issue_number} returned a malformed linked PR."
            )
        linked[number] = LinkedPullRequest(
            number=number,
            state=_optional_str(node.get("state")) or "",
            merged_at=_optional_str(node.get("mergedAt")),
            closed_at=_optional_str(node.get("closedAt")),
        )
    return IssueOutcomeHistory(
        number=issue_number,
        title=_optional_str(issue.get("title")) or "",
        body=_optional_str(issue.get("body")) or "",
        state=_optional_str(issue.get("state")) or "",
        created_at=created_at,
        closed_at=_optional_str(issue.get("closedAt")),
        body_revisions=_parse_body_revisions(issue),
        title_renames=_parse_title_renames(issue),
        linked_prs=tuple(linked[number] for number in sorted(linked)),
    )
