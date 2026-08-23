"""Canonical issue-to-PR handoff record for issue reruns (#589).

After an issue implementation creates a validated open PR, a rerun of
`agent-loop issue <n>` (direct or plan-first) should resume reviewing that PR
instead of invoking a coder again and creating a duplicate. This module
defines the `AGENT_ISSUE_PR_HANDOFF` marker that records which PR is the
authoritative implementation PR for an issue, and the resolver that consults
it (falling back to the legacy exactly-one-open-PR GitHub search for issues
predating this marker) before any coder invocation.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal
from urllib.parse import urlparse

from .config import AgentLoopConfig
from .errors import AgentLoopError
from .github import (
    IssueContext,
    OpenPrClosingMatch,
    PullRequestMetadata,
    find_open_pr_closing_issue,
    get_pr_state,
    get_pr_review_context,
    post_issue_comment,
)
from .runner import Runner

SCHEMA_VERSION = 1
_VALID_FLOWS = {"issue-implementation", "approved-plan-implementation"}

AGENT_ISSUE_PR_HANDOFF_RE = re.compile(
    r"<!--\s*AGENT_ISSUE_PR_HANDOFF:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)


@dataclass(frozen=True)
class IssuePrHandoffMetadata:
    schema_version: int
    issue_number: int
    pr_number: int
    pr_url: str
    pr_head_sha: str
    flow: str
    plan_hash: str | None


@dataclass(frozen=True)
class ResolvedIssuePr:
    pr_number: int
    source: Literal["canonical", "legacy-closing-reference"]
    evidence: IssuePrHandoffMetadata | OpenPrClosingMatch

    @property
    def metadata(self) -> IssuePrHandoffMetadata | None:
        return self.evidence if isinstance(self.evidence, IssuePrHandoffMetadata) else None

    @property
    def evidence_summary(self) -> str:
        if self.metadata is not None:
            return (
                "canonical marker "
                f"(flow={self.metadata.flow}, plan_hash={self.metadata.plan_hash or 'none'}, "
                f"pr_url={self.metadata.pr_url})"
            )
        match = self.evidence
        return "legacy-closing-reference " + "; ".join(
            f"keyword={item.keyword}, target={item.target_repo}#{item.issue_number}, "
            f"form={item.reference_form}, text={item.matched_text!r}"
            for item in match.evidence
        )


def _encode_json_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_json_payload(encoded: str) -> dict[str, object]:
    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, json.JSONDecodeError) as exc:
        raise AgentLoopError("Invalid AGENT_ISSUE_PR_HANDOFF payload.") from exc
    if not isinstance(payload, dict):
        raise AgentLoopError("Invalid AGENT_ISSUE_PR_HANDOFF payload.")
    return payload


def _encode_issue_pr_handoff_metadata(metadata: IssuePrHandoffMetadata) -> str:
    return _encode_json_payload(
        {
            "schema_version": metadata.schema_version,
            "issue_number": metadata.issue_number,
            "pr_number": metadata.pr_number,
            "pr_url": metadata.pr_url,
            "pr_head_sha": metadata.pr_head_sha,
            "flow": metadata.flow,
            "plan_hash": metadata.plan_hash,
        }
    )


def _require_non_empty_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError(
            f"Invalid AGENT_ISSUE_PR_HANDOFF payload: `{key}` must be a non-empty string."
        )
    return value


def _decode_issue_pr_handoff_metadata(encoded: str) -> IssuePrHandoffMetadata:
    payload = _decode_json_payload(encoded)
    raw_schema_version = payload.get("schema_version")
    if isinstance(raw_schema_version, bool) or not isinstance(raw_schema_version, int):
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `schema_version` must be an integer "
            "(not a bool or fractional value)."
        )
    schema_version = raw_schema_version
    if schema_version != SCHEMA_VERSION:
        raise AgentLoopError(
            f"Invalid AGENT_ISSUE_PR_HANDOFF payload: unsupported schema_version {schema_version}."
        )
    flow = payload.get("flow")
    if flow not in _VALID_FLOWS:
        raise AgentLoopError(f"Invalid AGENT_ISSUE_PR_HANDOFF payload: unknown flow {flow!r}.")
    raw_issue_number = payload.get("issue_number")
    raw_pr_number = payload.get("pr_number")
    if (
        isinstance(raw_issue_number, bool)
        or not isinstance(raw_issue_number, int)
        or isinstance(raw_pr_number, bool)
        or not isinstance(raw_pr_number, int)
    ):
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `issue_number`/`pr_number` must be integers."
        )
    issue_number = raw_issue_number
    pr_number = raw_pr_number
    if issue_number <= 0 or pr_number <= 0:
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `issue_number`/`pr_number` must be positive."
        )
    pr_url = _require_non_empty_str(payload, "pr_url")
    pr_head_sha = _require_non_empty_str(payload, "pr_head_sha")
    plan_hash = payload.get("plan_hash")
    if flow == "approved-plan-implementation":
        if not isinstance(plan_hash, str) or not plan_hash.strip():
            raise AgentLoopError(
                "Invalid AGENT_ISSUE_PR_HANDOFF payload: `plan_hash` is required for "
                "approved-plan-implementation flow."
            )
    elif plan_hash is not None:
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `plan_hash` must be absent for "
            "issue-implementation flow."
        )
    return IssuePrHandoffMetadata(
        schema_version=schema_version,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=pr_url,
        pr_head_sha=pr_head_sha,
        flow=str(flow),
        plan_hash=plan_hash if isinstance(plan_hash, str) else None,
    )


def _validate_issue_pr_handoff_url(url: str, *, repo: str, pr_number: int) -> None:
    parsed = urlparse(url)
    expected_path = f"/{repo}/pull/{pr_number}".casefold()
    if (
        parsed.scheme != "https"
        or parsed.netloc.casefold() != "github.com"
        or parsed.path.rstrip("/").casefold() != expected_path
        or parsed.query
        or parsed.fragment
    ):
        raise AgentLoopError(
            f"Invalid AGENT_ISSUE_PR_HANDOFF payload: `pr_url` {url!r} does not match "
            f"https://github.com/{repo}/pull/{pr_number}."
        )


def find_latest_issue_pr_handoff(
    comments: Sequence[object], *, issue_number: int, repo: str
) -> IssuePrHandoffMetadata | None:
    found: IssuePrHandoffMetadata | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(body):
            metadata = _decode_issue_pr_handoff_metadata(match.group("payload"))
            if metadata.issue_number != issue_number:
                continue
            _validate_issue_pr_handoff_url(metadata.pr_url, repo=repo, pr_number=metadata.pr_number)
            found = metadata
    return found


def resolve_canonical_pr_for_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
) -> ResolvedIssuePr | None:
    """Resolve the PR a rerun of `agent-loop issue <issue_number>` should resume.

    Consults the canonical `AGENT_ISSUE_PR_HANDOFF` record first; if none
    exists, falls back to strong closing-reference recovery for issues
    predating this marker. A canonical marker is authoritative: malformed,
    stale, or mismatched remote state raises instead of falling through to a
    potentially unrelated PR.
    """
    if config.dry_run:
        return None
    canonical = find_latest_issue_pr_handoff(
        issue_context.comments, issue_number=issue_number, repo=config.repo
    )
    if canonical is not None:
        try:
            pr_context = get_pr_review_context(
                runner, config=config, pr_number=canonical.pr_number
            )
            actual = pr_context.metadata
            if actual.number != canonical.pr_number:
                raise AgentLoopError(
                    f"GitHub returned PR #{actual.number} for canonical PR "
                    f"#{canonical.pr_number}."
                )
            if not actual.url:
                raise AgentLoopError("GitHub returned no PR URL.")
            _validate_issue_pr_handoff_url(
                actual.url, repo=config.repo, pr_number=canonical.pr_number
            )
            if actual.url.casefold() != canonical.pr_url.casefold():
                raise AgentLoopError(
                    f"recorded URL {canonical.pr_url!r} does not match GitHub URL "
                    f"{actual.url!r}"
                )
            state = get_pr_state(runner, config=config, pr_number=canonical.pr_number)
        except AgentLoopError as exc:
            raise AgentLoopError(
                f"Canonical handoff record for issue #{issue_number} references PR "
                f"#{canonical.pr_number}, but its state could not be determined in {config.repo} "
                f"({exc}). Verify the PR exists and rerun `agent-loop pr {canonical.pr_number}` "
                "directly to continue, or close/select the correct duplicate."
            ) from exc
        if state != "OPEN":
            raise AgentLoopError(
                f"Canonical handoff record for issue #{issue_number} references PR "
                f"#{canonical.pr_number}, which is {state}, not OPEN. Rerun "
                f"`agent-loop pr {canonical.pr_number}` directly if that PR should still be "
                "reviewed, or close/select the correct duplicate before rerunning the issue."
            )
        return ResolvedIssuePr(
            pr_number=canonical.pr_number, source="canonical", evidence=canonical
        )
    legacy_match = find_open_pr_closing_issue(
        runner, config=config, issue_number=issue_number
    )
    if legacy_match is None:
        return None
    return ResolvedIssuePr(
        pr_number=legacy_match.pr_number,
        source="legacy-closing-reference",
        evidence=legacy_match,
    )


def require_pr_metadata_for_handoff(metadata: PullRequestMetadata) -> tuple[str, str]:
    """Return `(pr_url, pr_head_sha)`, raising if either is unavailable.

    Guards every handoff-posting call site so a record is never posted with
    an incomplete PR URL or head SHA.
    """
    if not metadata.url:
        raise AgentLoopError(
            f"Cannot record issue-to-PR handoff for PR #{metadata.number}: PR URL is unavailable."
        )
    if not metadata.head_sha:
        raise AgentLoopError(
            f"Cannot record issue-to-PR handoff for PR #{metadata.number}: PR head SHA is unavailable."
        )
    return metadata.url, metadata.head_sha


def format_issue_pr_handoff_comment(
    *,
    issue_number: int,
    pr_number: int,
    pr_url: str,
    pr_head_sha: str,
    flow: str,
    plan_hash: str | None,
) -> str:
    metadata = IssuePrHandoffMetadata(
        schema_version=SCHEMA_VERSION,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=pr_url,
        pr_head_sha=pr_head_sha,
        flow=flow,
        plan_hash=plan_hash,
    )
    lines = [
        f"Issue #{issue_number} implementation handed off to PR #{pr_number}.",
        "",
        f"Flow: {flow}",
        f"PR: {pr_url}",
        f"PR head SHA: {pr_head_sha}",
    ]
    if plan_hash:
        lines.append(f"Plan hash: {plan_hash}")
    lines.extend(
        [
            "",
            "Reruns of `agent-loop issue` for this issue will resume review of this PR instead of "
            "invoking a coder again.",
            "",
            f"<!-- AGENT_ISSUE_PR_HANDOFF: {_encode_issue_pr_handoff_metadata(metadata)} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def post_issue_pr_handoff_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    pr_number: int,
    pr_url: str,
    pr_head_sha: str,
    flow: str,
    plan_hash: str | None,
) -> None:
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=format_issue_pr_handoff_comment(
            issue_number=issue_number,
            pr_number=pr_number,
            pr_url=pr_url,
            pr_head_sha=pr_head_sha,
            flow=flow,
            plan_hash=plan_hash,
        ),
    )
