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
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlparse

from .config import AgentLoopConfig
from .errors import AgentLoopError, WorkflowTransactionError
from .expected_closure import contract_hash, normalize_issue_ids
from .issue_pr_provenance import IssuePrProvenanceScope
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
from .protocol_markers import TrustedBody

if TYPE_CHECKING:
    from .workflow_transaction_publication import CanonicalHandoffView

SCHEMA_VERSION = 1
_VALID_FLOWS = {"issue-implementation", "approved-plan-implementation"}

AGENT_ISSUE_PR_HANDOFF_RE = re.compile(
    r"<!--\s*AGENT_ISSUE_PR_HANDOFF:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)


_UNSET_FALLBACK_SCOPE = object()


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
    legacy_contract: bool = False

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
    evidence: "IssuePrHandoffMetadata | CanonicalHandoffView | OpenPrClosingMatch"

    @property
    def metadata(self) -> "IssuePrHandoffMetadata | CanonicalHandoffView | None":
        """The canonical handoff binding: the fields both record versions share."""
        return self.evidence if self.source == "canonical" else None

    @property
    def evidence_summary(self) -> str:
        if self.source == "canonical" and not isinstance(self.evidence, IssuePrHandoffMetadata):
            view = self.evidence
            return (
                "committed workflow transaction "
                f"(flow={view.flow}, plan_hash={view.plan_hash or 'none'}, "
                f"pr_url={view.pr_url}, transaction={view.transaction_id})"
            )
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
    legacy_contract = raw_expected is None and payload.get("contract_hash") is None
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
        legacy_contract=legacy_contract,
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


@dataclass(frozen=True)
class IssuePrHandoffLineage:
    """The latest handoff record plus the record cross-side checks compare against.

    A same-PR approved-plan replacement (#936) changes only the plan hash: it
    is an issue-side record with no PR-side counterpart, and every such
    replacement shares the closing-ID contract digest of the record it
    replaces.  ``closing_base`` is therefore the most recent record that is
    not such a replacement; the PR-side closing contract authenticates against
    it, while the plan hash always comes from ``latest``.  A plan replacement
    is identified only by its plan-changing edge (plan hash plus its audit
    record), never by the closing-ID contract digest.

    The most recent plan-changing edge for the current PR is tracked
    independently of the closing base: ``replaced`` is the record whose plan
    was replaced, ``replacement`` the record that first named the new plan,
    and ``replacement_comment_index`` its comment.  A later closing-ID
    superset moves the base but never erases that edge, so the rebind stays
    verifiable for as long as the PR is bound to the replacement plan.
    """

    latest: IssuePrHandoffMetadata
    closing_base: IssuePrHandoffMetadata
    replaced: IssuePrHandoffMetadata | None = None
    replacement: IssuePrHandoffMetadata | None = None
    replacement_comment_index: int = -1
    latest_comment_index: int = -1

    @property
    def is_plan_replacement(self) -> bool:
        return self.replaced is not None


def find_latest_issue_pr_handoff(
    comments: Sequence[object], *, issue_number: int, repo: str
) -> IssuePrHandoffMetadata | None:
    lineage = resolve_issue_pr_handoff_lineage(comments, issue_number=issue_number, repo=repo)
    return lineage.latest if lineage is not None else None


def resolve_issue_pr_handoff_lineage(
    comments: Sequence[object], *, issue_number: int, repo: str
) -> IssuePrHandoffLineage | None:
    found: IssuePrHandoffMetadata | None = None
    closing_base: IssuePrHandoffMetadata | None = None
    replaced: IssuePrHandoffMetadata | None = None
    replacement: IssuePrHandoffMetadata | None = None
    replacement_index = -1
    found_index = -1
    for comment_index, comment in enumerate(comments):
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
                if metadata.supersedes_hash == found.contract_hash and set(
                    found.expected_closing_issue_ids
                ) < set(metadata.expected_closing_issue_ids):
                    if found.plan_hash != metadata.plan_hash:
                        # A superset that also changes the plan is a
                        # plan-changing edge in its own right.
                        replaced, replacement = found, metadata
                        replacement_index = comment_index
                    found = closing_base = metadata
                    found_index = comment_index
                    continue
                if (
                    # An approved implementation plan may be replaced
                    # for the same PR without changing its closing
                    # issue contract.  The explicit supersession still
                    # prevents an unannotated divergent handoff from
                    # silently replacing the authoritative identity.
                    metadata.supersedes_hash == found.contract_hash
                    and found.flow == metadata.flow == "approved-plan-implementation"
                    and found.plan_hash != metadata.plan_hash
                    and set(found.expected_closing_issue_ids)
                    == set(metadata.expected_closing_issue_ids)
                ):
                    replaced, replacement = found, metadata
                    replacement_index = comment_index
                    found = metadata
                    found_index = comment_index
                    continue
                raise AgentLoopError(
                    "Divergent AGENT_ISSUE_PR_HANDOFF records were found for "
                    f"issue #{issue_number}."
                )
            if found != metadata:
                # First record, or a record for a different PR: a new lineage.
                closing_base = metadata
                replaced = replacement = None
                replacement_index = -1
                found_index = comment_index
            found = metadata
    if found is None:
        return None
    assert closing_base is not None
    return IssuePrHandoffLineage(
        latest=found,
        closing_base=closing_base,
        replaced=replaced,
        replacement=replacement,
        replacement_comment_index=replacement_index,
        latest_comment_index=found_index,
    )


@dataclass(frozen=True)
class AuthenticatedCanonicalPr:
    """A canonical issue-to-PR handoff whose identity and contract are verified.

    Identity, URL, and closing-contract authentication are separated here
    from the `OPEN`-only gate that resume applies, so a staged parent can
    authenticate a *merged* child PR as completion evidence with exactly the
    same checks resume uses.
    """

    # A version-1 record for a legacy-era PR; for a transaction-era PR the
    # ``CanonicalHandoffView`` of the committed workflow transaction (#827).
    record: "IssuePrHandoffMetadata | CanonicalHandoffView"
    state: str

    @property
    def pr_number(self) -> int:
        return self.record.pr_number

    @property
    def pr_url(self) -> str:
        return self.record.pr_url


def _names_version_2_handoff(comments: Sequence[object]) -> bool:
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(body):
            try:
                if issue_pr_handoff_payload_schema_version(match.group("payload")) == 2:
                    return True
            except AgentLoopError:
                continue
    return False


def _authenticate_transaction_era_issue_pr(
    runner: Runner, *, config: AgentLoopConfig, issue_number: int
) -> "AuthenticatedCanonicalPr | None":
    """Authority form for an issue whose handoff is version 2 (#827).

    Candidate-PR discovery and the committed-transaction gate replace the
    version-1 cross-surface checks; both are read-only and fail closed on a
    partial, pending, or contradictory transaction.  Returns ``None`` when the
    authenticated actor's records are all version 1.
    """
    # Imported here: the publication module imports this one at module level.
    from .managed_ci_bound_authorization import BoundAuthorizationCodec
    from .workflow_transaction import ERA_TRANSACTION
    from .workflow_transaction_publication import (
        discover_canonical_issue_pr,
        gate_canonical_issue_pr,
    )

    discovered = discover_canonical_issue_pr(runner, config, issue_number)
    if discovered is None or discovered.era != ERA_TRANSACTION:
        return None
    view = discovered.handoff
    try:
        actual = get_pr_review_context(
            runner, config=config, pr_number=discovered.pr_number
        ).metadata
        if actual.number != discovered.pr_number:
            raise AgentLoopError(
                f"GitHub returned PR #{actual.number} for canonical PR "
                f"#{discovered.pr_number}."
            )
        if not actual.url:
            raise AgentLoopError("GitHub returned no PR URL.")
        _validate_issue_pr_handoff_url(
            actual.url, repo=config.repo, pr_number=discovered.pr_number
        )
        if actual.url.casefold() != view.pr_url.casefold():
            raise AgentLoopError(
                f"recorded URL {view.pr_url!r} does not match GitHub URL {actual.url!r}"
            )
        if not actual.head_sha:
            raise AgentLoopError("GitHub returned no PR head SHA.")
        state = get_pr_state(runner, config=config, pr_number=discovered.pr_number)
    except WorkflowTransactionError:
        raise
    except AgentLoopError as exc:
        raise AgentLoopError(
            f"Canonical handoff record for issue #{issue_number} references PR "
            f"#{discovered.pr_number}, but its state could not be determined in {config.repo} "
            f"({exc}). Verify the PR exists and rerun `agent-loop pr {discovered.pr_number}` "
            "directly to continue, or close/select the correct duplicate."
        ) from exc
    gate_canonical_issue_pr(
        runner,
        config,
        discovered,
        issue_number=issue_number,
        live_head=actual.head_sha,
        pr_state=state,
        authorization_codec=BoundAuthorizationCodec(),
    )
    return AuthenticatedCanonicalPr(record=view, state=state)


def authenticate_canonical_issue_pr(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
) -> AuthenticatedCanonicalPr | None:
    """Authenticate the canonical PR recorded for an issue, whatever its state.

    Returns ``None`` when the issue carries no canonical
    `AGENT_ISSUE_PR_HANDOFF` record. Otherwise the recorded PR number, URL and
    expected-closing contract are reconciled against live GitHub state and the
    live PR state is returned; any mismatch raises, so an unauthenticatable
    record can never be read as evidence.
    """
    if _names_version_2_handoff(issue_context.comments):
        transaction_era = _authenticate_transaction_era_issue_pr(
            runner, config=config, issue_number=issue_number
        )
        if transaction_era is not None:
            return transaction_era
        # Only foreign-authored version-2 records exist: the version-1 path
        # below still rejects them, exactly as before.
    lineage = resolve_issue_pr_handoff_lineage(
        issue_context.comments, issue_number=issue_number, repo=config.repo
    )
    if lineage is None:
        return None
    canonical = lineage.latest
    # The PR-side closing contract is never rewritten by a same-PR plan
    # replacement, so it authenticates against the lineage base.
    closing_base = lineage.closing_base
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
            closing_base.expected_closing_issue_ids
        ):
            raise AgentLoopError(
                "issue-side and PR-side expected closing contracts diverge: "
                f"issue side {closing_base.expected_closing_issue_ids!r}, PR side "
                f"{pr_contract.expected_closing_issue_ids!r}."
            )
        if pr_contract is not None and (
            pr_contract.primary_issue_number != closing_base.issue_number
            or pr_contract.origin_flow != closing_base.flow
            or pr_contract.contract_hash != closing_base.contract_hash
            or pr_contract.supersedes_hash != closing_base.supersedes_hash
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
    return AuthenticatedCanonicalPr(record=canonical, state=state)


def resolve_canonical_pr_for_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
    expected_fallback_scope: IssuePrProvenanceScope | None | object = _UNSET_FALLBACK_SCOPE,
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
    authenticated = authenticate_canonical_issue_pr(
        runner, config=config, issue_number=issue_number, issue_context=issue_context
    )
    if authenticated is not None:
        if authenticated.state != "OPEN":
            raise AgentLoopError(
                f"Canonical handoff record for issue #{issue_number} references PR "
                f"#{authenticated.pr_number}, which is {authenticated.state}, not OPEN. Rerun "
                f"`agent-loop pr {authenticated.pr_number}` directly if that PR should still be "
                "reviewed, or close/select the correct duplicate before rerunning the issue."
            )
        return ResolvedIssuePr(
            pr_number=authenticated.pr_number, source="canonical", evidence=authenticated.record
        )
    if expected_fallback_scope is _UNSET_FALLBACK_SCOPE:
        expected_fallback_scope = IssuePrProvenanceScope(
            repository=config.repo, issue_number=issue_number, flow="direct"
        )
    legacy_match = find_open_pr_closing_issue(
        runner,
        config=config,
        issue_number=issue_number,
        expected_scope=expected_fallback_scope,
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
        body=TrustedBody.canonical(
            format_issue_pr_handoff_comment(
                issue_number=issue_number,
                pr_number=pr_number,
                pr_url=pr_url,
                pr_head_sha=pr_head_sha,
                flow=flow,
                plan_hash=plan_hash,
                expected_closing_issue_ids=expected_closing_issue_ids,
                supersedes_hash=supersedes_hash,
            ),
            expected_tokens=("AGENT_ISSUE_PR_HANDOFF",),
        ),
    )


# ---------------------------------------------------------------------------
# Version 2: transaction-bound handoff records (#827).
#
# A separate codec.  ``resolve_issue_pr_handoff_lineage`` and the version-1
# decoder are deliberately not taught version 2 and keep rejecting it, so a
# comment snapshot that was not read through the author-authenticated reader
# can never be interpreted as a version-2 record.
# ---------------------------------------------------------------------------

HANDOFF_V2_SCHEMA_VERSION = 2
_TRANSACTION_ID_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_V2_HEAD_SHA_RE = re.compile(r"\A(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
_V2_PLAN_HASH_RE = re.compile(r"\A[0-9a-f]{16}\Z")
_V2_PR_URL_RE = re.compile(
    r"\Ahttps://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/pull/(?P<number>[1-9][0-9]*)\Z"
)


@dataclass(frozen=True)
class IssuePrHandoffMetadataV2:
    issue_number: int
    pr_number: int
    pr_url: str
    pr_head_sha: str
    flow: str
    plan_hash: str | None
    expected_closing_issue_ids: tuple[int, ...]
    contract_hash: str
    transaction_id: str
    schema_version: int = HANDOFF_V2_SCHEMA_VERSION


def encode_issue_pr_handoff_v2(metadata: IssuePrHandoffMetadataV2) -> str:
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
            "transaction_id": metadata.transaction_id,
        }
    )


def decode_issue_pr_handoff_v2(encoded: str) -> IssuePrHandoffMetadataV2:
    payload = _decode_json_payload(encoded)
    required = {
        "schema_version",
        "issue_number",
        "pr_number",
        "pr_url",
        "pr_head_sha",
        "flow",
        "plan_hash",
        "expected_closing_issue_ids",
        "contract_hash",
        "transaction_id",
    }
    prefix = "Invalid AGENT_ISSUE_PR_HANDOFF v2 payload"
    if set(payload) != required:
        raise AgentLoopError(f"{prefix}: expected exactly {', '.join(sorted(required))}.")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != HANDOFF_V2_SCHEMA_VERSION:
        raise AgentLoopError(f"{prefix}: schema_version must be 2.")
    numbers: dict[str, int] = {}
    for key in ("issue_number", "pr_number"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AgentLoopError(f"{prefix}: `{key}` must be a positive integer.")
        numbers[key] = value
    pr_url = _require_non_empty_str(payload, "pr_url")
    pr_head_sha = _require_non_empty_str(payload, "pr_head_sha")
    url_match = _V2_PR_URL_RE.match(pr_url)
    if url_match is None or int(url_match.group("number")) != numbers["pr_number"]:
        raise AgentLoopError(f"{prefix}: `pr_url` is not this PR's canonical GitHub URL.")
    if _V2_HEAD_SHA_RE.match(pr_head_sha) is None:
        raise AgentLoopError(f"{prefix}: `pr_head_sha` must be a full lowercase commit SHA.")
    flow = payload["flow"]
    if flow not in _VALID_FLOWS:
        raise AgentLoopError(f"{prefix}: unknown flow {flow!r}.")
    plan_hash = payload["plan_hash"]
    if flow == "approved-plan-implementation":
        if not isinstance(plan_hash, str) or _V2_PLAN_HASH_RE.match(plan_hash) is None:
            raise AgentLoopError(
                f"{prefix}: `plan_hash` is required for approved-plan-implementation flow."
            )
    elif plan_hash is not None:
        raise AgentLoopError(
            f"{prefix}: `plan_hash` must be absent for issue-implementation flow."
        )
    expected_ids = normalize_issue_ids(
        payload["expected_closing_issue_ids"],
        field_name="AGENT_ISSUE_PR_HANDOFF.expected_closing_issue_ids",
    )
    assert expected_ids is not None
    if list(expected_ids) != payload["expected_closing_issue_ids"]:
        raise AgentLoopError(f"{prefix}: expected closing IDs are not canonical.")
    if numbers["issue_number"] not in expected_ids:
        raise AgentLoopError(
            f"{prefix}: expected closing IDs must retain the primary issue "
            f"#{numbers['issue_number']}."
        )
    digest = payload["contract_hash"]
    if not isinstance(digest, str) or digest != contract_hash(expected_ids):
        raise AgentLoopError(
            f"{prefix}: `contract_hash` does not match expected_closing_issue_ids."
        )
    transaction_id = payload["transaction_id"]
    if not isinstance(transaction_id, str) or _TRANSACTION_ID_RE.match(transaction_id) is None:
        raise AgentLoopError(f"{prefix}: `transaction_id` is invalid.")
    metadata = IssuePrHandoffMetadataV2(
        issue_number=numbers["issue_number"],
        pr_number=numbers["pr_number"],
        pr_url=pr_url,
        pr_head_sha=pr_head_sha,
        flow=str(flow),
        plan_hash=plan_hash if isinstance(plan_hash, str) else None,
        expected_closing_issue_ids=expected_ids,
        contract_hash=digest,
        transaction_id=transaction_id,
    )
    if encode_issue_pr_handoff_v2(metadata) != encoded:
        raise AgentLoopError(f"{prefix}: record is not canonically encoded.")
    return metadata


def issue_pr_handoff_payload_schema_version(encoded: str) -> object:
    """Peek at a payload's declared version without interpreting the record."""
    return _decode_json_payload(encoded).get("schema_version")


def issue_pr_handoff_record_hash(
    metadata: IssuePrHandoffMetadata | IssuePrHandoffMetadataV2,
) -> str:
    """Hash the full canonical payload of one handoff record."""
    encoded = (
        encode_issue_pr_handoff_v2(metadata)
        if isinstance(metadata, IssuePrHandoffMetadataV2)
        else _encode_issue_pr_handoff_metadata(metadata)
    )
    return hashlib.sha256(base64.urlsafe_b64decode(encoded.encode("ascii"))).hexdigest()


def format_issue_pr_handoff_v2_comment(metadata: IssuePrHandoffMetadataV2, *, repo: str) -> str:
    encoded = encode_issue_pr_handoff_v2(metadata)
    if encode_issue_pr_handoff_v2(decode_issue_pr_handoff_v2(encoded)) != encoded:
        raise AgentLoopError("Issue-to-PR handoff failed canonical rendering validation.")
    _validate_issue_pr_handoff_url(metadata.pr_url, repo=repo, pr_number=metadata.pr_number)
    lines = [
        f"Issue #{metadata.issue_number} implementation handed off to PR #{metadata.pr_number}.",
        "",
        f"Flow: {metadata.flow}",
        f"PR: {metadata.pr_url}",
        f"PR head SHA: {metadata.pr_head_sha}",
    ]
    if metadata.plan_hash:
        lines.append(f"Plan hash: {metadata.plan_hash}")
    lines.extend(
        [
            "Expected closing issues: "
            + ", ".join(f"#{item}" for item in metadata.expected_closing_issue_ids)
            + ".",
            f"Workflow transaction: {metadata.transaction_id}",
            "",
            f"<!-- AGENT_ISSUE_PR_HANDOFF: {encoded} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)
