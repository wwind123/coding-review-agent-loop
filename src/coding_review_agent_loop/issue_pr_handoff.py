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
from .expected_closure import contract_hash, normalize_issue_ids
from .github import (
    IssueContext,
    OpenPrClosingMatch,
    PullRequestMetadata,
    find_open_pr_closing_issue,
    get_pr_state,
    get_pr_review_context,
    post_issue_comment,
    post_trusted_issue_comment,
)
from .pr_contract import PrExpectedClosingContract, find_latest_pr_contract
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
    expected_closing_issue_ids: tuple[int, ...] = ()
    contract_hash: str | None = None
    supersedes_hash: str | None = None

    def __post_init__(self) -> None:
        # Handoff records are issue-origin records, so the primary issue is
        # always part of their contract. Preserve compatibility with callers
        # that construct the pre-contract dataclass without the new fields.
        if (
            not self.expected_closing_issue_ids
            and isinstance(self.issue_number, int)
            and not isinstance(self.issue_number, bool)
            and self.issue_number > 0
        ):
            object.__setattr__(self, "expected_closing_issue_ids", (self.issue_number,))
        if not self.expected_closing_issue_ids:
            return
        normalized = normalize_issue_ids(
            self.expected_closing_issue_ids,
            field_name="expected_closing_issue_ids",
        )
        assert normalized is not None
        if self.issue_number not in normalized:
            raise AgentLoopError(
                f"Issue handoff contract must retain primary issue #{self.issue_number}."
            )
        object.__setattr__(self, "expected_closing_issue_ids", normalized)
        if self.contract_hash is None:
            object.__setattr__(self, "contract_hash", contract_hash(normalized))


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
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
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
            "expected_closing_issue_ids": list(metadata.expected_closing_issue_ids),
            "contract_hash": metadata.contract_hash,
            "supersedes_hash": metadata.supersedes_hash,
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
    raw_expected = payload.get("expected_closing_issue_ids")
    if raw_expected is None:
        expected_ids = (issue_number,)
    else:
        expected_ids = normalize_issue_ids(
            raw_expected, field_name="AGENT_ISSUE_PR_HANDOFF.expected_closing_issue_ids"
        )
        assert expected_ids is not None
        if issue_number not in expected_ids:
            raise AgentLoopError(
                "Invalid AGENT_ISSUE_PR_HANDOFF payload: expected closing IDs must retain "
                f"the primary issue #{issue_number}."
            )
    raw_contract_hash = payload.get("contract_hash")
    expected_contract_hash = contract_hash(expected_ids)
    if raw_contract_hash is None:
        handoff_contract_hash = expected_contract_hash
    elif isinstance(raw_contract_hash, str) and raw_contract_hash == expected_contract_hash:
        handoff_contract_hash = raw_contract_hash
    else:
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `contract_hash` does not match "
            "expected_closing_issue_ids."
        )
    supersedes_hash = payload.get("supersedes_hash")
    if supersedes_hash is not None and (
        not isinstance(supersedes_hash, str) or not supersedes_hash.strip()
    ):
        raise AgentLoopError(
            "Invalid AGENT_ISSUE_PR_HANDOFF payload: `supersedes_hash` is invalid."
        )
    return IssuePrHandoffMetadata(
        schema_version=schema_version,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=pr_url,
        pr_head_sha=pr_head_sha,
        flow=str(flow),
        plan_hash=plan_hash if isinstance(plan_hash, str) else None,
        expected_closing_issue_ids=expected_ids,
        contract_hash=handoff_contract_hash,
        supersedes_hash=supersedes_hash if isinstance(supersedes_hash, str) else None,
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
            encoded = match.group("payload")
            metadata = _decode_issue_pr_handoff_metadata(encoded)
            canonical_encoded = _encode_issue_pr_handoff_metadata(metadata)
            if canonical_encoded != encoded:
                legacy_keys = {
                    "schema_version",
                    "issue_number",
                    "pr_number",
                    "pr_url",
                    "pr_head_sha",
                    "flow",
                    "plan_hash",
                }
                payload = _decode_json_payload(encoded)
                if set(payload) != legacy_keys:
                    raise AgentLoopError(
                        "AGENT_ISSUE_PR_HANDOFF record is not canonically encoded."
                    )
            if metadata.issue_number != issue_number:
                continue
            _validate_issue_pr_handoff_url(metadata.pr_url, repo=repo, pr_number=metadata.pr_number)
            if found is not None and found.pr_number == metadata.pr_number and found != metadata:
                if (
                    metadata.supersedes_hash == found.contract_hash
                    and set(found.expected_closing_issue_ids) < set(metadata.expected_closing_issue_ids)
                ):
                    found = metadata
                    continue
                raise AgentLoopError(
                    "Divergent AGENT_ISSUE_PR_HANDOFF records were found for "
                    f"issue #{issue_number}."
                )
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
            pr_contract = find_latest_pr_contract(
                pr_context.comments,
                repository=config.repo,
                pr_number=canonical.pr_number,
            )
            if pr_contract is not None and tuple(pr_contract.expected_closing_issue_ids) != tuple(
                canonical.expected_closing_issue_ids
            ):
                raise AgentLoopError(
                    "issue-side and PR-side expected closing contracts diverge: "
                    f"issue side {canonical.expected_closing_issue_ids!r}, PR side "
                    f"{pr_contract.expected_closing_issue_ids!r}."
                )
            if pr_contract is not None and (
                pr_contract.primary_issue_number != canonical.issue_number
                or pr_contract.origin_flow != canonical.flow
                or pr_contract.contract_hash != canonical.contract_hash
                or pr_contract.supersedes_hash != canonical.supersedes_hash
            ):
                raise AgentLoopError(
                    "issue-side and PR-side expected closing contract metadata diverge: "
                    "primary issue, origin flow, hash, or supersession lineage differs."
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
    expected_closing_issue_ids: Sequence[int] | None = None,
    supersedes_hash: str | None = None,
) -> str:
    expected_ids = normalize_issue_ids(
        expected_closing_issue_ids or (issue_number,),
        field_name="expected_closing_issue_ids",
    )
    assert expected_ids is not None
    if issue_number not in expected_ids:
        raise AgentLoopError(
            f"Issue handoff contract must retain primary issue #{issue_number}."
        )
    metadata = IssuePrHandoffMetadata(
        schema_version=SCHEMA_VERSION,
        issue_number=issue_number,
        pr_number=pr_number,
        pr_url=pr_url,
        pr_head_sha=pr_head_sha,
        flow=flow,
        plan_hash=plan_hash,
        expected_closing_issue_ids=expected_ids,
        contract_hash=contract_hash(expected_ids),
        supersedes_hash=supersedes_hash,
    )
    encoded_metadata = _encode_issue_pr_handoff_metadata(metadata)
    if _encode_issue_pr_handoff_metadata(_decode_issue_pr_handoff_metadata(encoded_metadata)) != encoded_metadata:
        raise AgentLoopError("Issue-to-PR handoff failed canonical rendering validation.")
    lines = [
        f"Issue #{issue_number} implementation handed off to PR #{pr_number}.",
        "",
        f"Flow: {flow}",
        f"PR: {pr_url}",
        f"PR head SHA: {pr_head_sha}",
    ]
    if plan_hash:
        lines.append(f"Plan hash: {plan_hash}")
    lines.append(
        "Expected closing issues: "
        + (", ".join(f"#{item}" for item in expected_ids) or "(none)")
        + "."
    )
    lines.extend(
        [
            "",
            "Reruns of `agent-loop issue` for this issue will resume review of this PR instead of "
            "invoking a coder again.",
            "",
            f"<!-- AGENT_ISSUE_PR_HANDOFF: {encoded_metadata} -->",
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
    expected_closing_issue_ids: Sequence[int] | None = None,
    supersedes_hash: str | None = None,
) -> None:
    post_trusted_issue_comment(
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
            expected_closing_issue_ids=expected_closing_issue_ids,
            supersedes_hash=supersedes_hash,
        ),
    )
