"""Unauthenticated commit provenance for issue-origin PR recovery.

This is deliberately a conventional Git trailer, not a durable protocol
marker.  It helps avoid adopting an unrelated open PR after an interrupted
issue implementation, but contributors can copy it, so it is not an
authentication boundary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from .errors import AgentLoopError


ISSUE_PR_PROVENANCE_TRAILER_KEY = "Agent-Issue-Provenance"
ISSUE_PR_PROVENANCE_VERSION = 1
# Public aliases keep the convention easy to discover without introducing a
# reserved protocol-marker definition.
ISSUE_PR_PROVENANCE_TRAILER = ISSUE_PR_PROVENANCE_TRAILER_KEY
PROVENANCE_TRAILER_KEY = ISSUE_PR_PROVENANCE_TRAILER_KEY
IssuePrProvenanceFlow = Literal["direct", "approved"]

_REPOSITORY_RE = re.compile(r"^[^/\s]+/[^/\s]+$")
_PLAN_HASH_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_TRAILER_LINE_RE = re.compile(
    rf"^{re.escape(ISSUE_PR_PROVENANCE_TRAILER_KEY)}:\s+(?P<value>\S(?:.*\S)?)$"
)


def normalize_repository(repository: str) -> str:
    """Normalize a GitHub ``owner/repository`` identity for comparison."""
    if not isinstance(repository, str):
        raise AgentLoopError("Issue provenance repository must be a string.")
    normalized = repository.strip().casefold()
    if not _REPOSITORY_RE.fullmatch(normalized):
        raise AgentLoopError(
            "Issue provenance repository must be a non-empty owner/repository identity."
        )
    return normalized


def _normalize_plan_hash(plan_hash: str | None) -> str | None:
    if plan_hash is None:
        return None
    if not isinstance(plan_hash, str) or not plan_hash.strip():
        raise AgentLoopError("Approved issue provenance requires a non-empty plan hash.")
    normalized = plan_hash.strip()
    if not _PLAN_HASH_RE.fullmatch(normalized):
        raise AgentLoopError("Issue provenance plan hash contains invalid characters.")
    return normalized


@dataclass(frozen=True)
class IssuePrProvenanceScope:
    """The exact issue-origin scope a recovered PR must claim."""

    repository: str
    issue_number: int
    flow: IssuePrProvenanceFlow
    approved_plan_hash: str | None = None

    @property
    def repository_identity(self) -> str:
        return self.repository

    @property
    def plan_hash(self) -> str | None:
        return self.approved_plan_hash

    def __post_init__(self) -> None:
        repository = normalize_repository(self.repository)
        issue_number = self.issue_number
        if isinstance(issue_number, str) and re.fullmatch(r"[1-9]\d*", issue_number.strip()):
            issue_number = int(issue_number.strip())
        if (
            isinstance(issue_number, bool)
            or not isinstance(issue_number, int)
            or issue_number <= 0
        ):
            raise AgentLoopError("Issue provenance issue number must be a positive integer.")
        if self.flow not in {"direct", "approved"}:
            raise AgentLoopError(
                "Issue provenance flow must be exactly 'direct' or 'approved'."
            )
        plan_hash = _normalize_plan_hash(self.approved_plan_hash)
        if self.flow == "direct" and plan_hash is not None:
            raise AgentLoopError("Direct issue provenance cannot carry a plan hash.")
        if self.flow == "approved" and plan_hash is None:
            raise AgentLoopError("Approved issue provenance requires a plan hash.")
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "issue_number", issue_number)
        object.__setattr__(self, "approved_plan_hash", plan_hash)


# Short aliases make the model convenient for callers that use the term
# "scope" rather than the full record name.
IssueProvenanceScope = IssuePrProvenanceScope


def format_issue_pr_provenance(scope: IssuePrProvenanceScope) -> str:
    """Render the canonical complete trailer line for ``scope``."""
    if not isinstance(scope, IssuePrProvenanceScope):
        raise AgentLoopError("Cannot format an invalid issue provenance scope.")
    fields = [
        f"v{ISSUE_PR_PROVENANCE_VERSION}",
        f"repo={scope.repository}",
        f"issue={scope.issue_number}",
        f"flow={scope.flow}",
    ]
    if scope.approved_plan_hash is not None:
        fields.append(f"plan={scope.approved_plan_hash}")
    return f"{ISSUE_PR_PROVENANCE_TRAILER_KEY}: " + " ".join(fields)


format_issue_provenance_trailer = format_issue_pr_provenance


def _parse_fields(value: str) -> dict[str, str]:
    parts = value.split()
    if not parts or not parts[0].startswith("v"):
        raise AgentLoopError("Issue provenance trailer has no supported version.")
    try:
        version = int(parts[0][1:])
    except ValueError as exc:
        raise AgentLoopError("Issue provenance trailer has an invalid version.") from exc
    if version != ISSUE_PR_PROVENANCE_VERSION:
        raise AgentLoopError(f"Issue provenance trailer has unsupported version {version}.")
    fields: dict[str, str] = {}
    for part in parts[1:]:
        key, separator, raw_value = part.partition("=")
        if not separator or not key or not raw_value or key in fields:
            raise AgentLoopError("Issue provenance trailer has malformed fields.")
        fields[key] = raw_value
    if set(fields) not in ({"repo", "issue", "flow"}, {"repo", "issue", "flow", "plan"}):
        raise AgentLoopError("Issue provenance trailer has an incomplete or unknown scope.")
    try:
        issue_number = int(fields["issue"])
    except ValueError as exc:
        raise AgentLoopError("Issue provenance trailer issue is not an integer.") from exc
    flow = fields["flow"]
    plan_hash = fields.get("plan")
    return {
        "repo": fields["repo"],
        "issue": str(issue_number),
        "flow": flow,
        "plan": plan_hash or "",
    }


def parse_issue_pr_provenance_line(line: str) -> IssuePrProvenanceScope | None:
    """Parse one complete trailer line; unrelated lines return ``None``."""
    if not isinstance(line, str):
        raise AgentLoopError("Issue provenance commit message line must be text.")
    stripped = line.strip()
    if not stripped:
        return None
    if not (
        stripped == ISSUE_PR_PROVENANCE_TRAILER_KEY
        or stripped.startswith(f"{ISSUE_PR_PROVENANCE_TRAILER_KEY}:")
        or stripped.startswith(f"{ISSUE_PR_PROVENANCE_TRAILER_KEY} ")
    ):
        return None
    match = _TRAILER_LINE_RE.fullmatch(stripped)
    if match is None:
        raise AgentLoopError("Issue provenance trailer is malformed.")
    fields = _parse_fields(match.group("value"))
    try:
        return IssuePrProvenanceScope(
            repository=fields["repo"],
            issue_number=int(fields["issue"]),
            flow=fields["flow"],  # type: ignore[arg-type]
            approved_plan_hash=fields["plan"] or None,
        )
    except AgentLoopError:
        raise
    except (TypeError, ValueError) as exc:
        raise AgentLoopError("Issue provenance trailer has an invalid scope.") from exc


parse_issue_provenance_trailer = parse_issue_pr_provenance_line


def parse_issue_pr_provenance_messages(
    messages: Iterable[str],
) -> tuple[IssuePrProvenanceScope, ...]:
    """Parse complete provenance trailer lines from commit messages only.

    A line that starts with the trailer key is an occurrence even when it is
    malformed.  That distinction lets callers fail closed for valid-plus-
    malformed mixtures instead of silently ignoring the malformed claim.
    """
    claims: list[IssuePrProvenanceScope] = []
    for message in messages:
        if not isinstance(message, str):
            raise AgentLoopError("GitHub returned a non-text commit message.")
        for line in message.splitlines():
            stripped = line.strip()
            if (
                stripped == ISSUE_PR_PROVENANCE_TRAILER_KEY
                or stripped.startswith(f"{ISSUE_PR_PROVENANCE_TRAILER_KEY}:")
                or stripped.startswith(f"{ISSUE_PR_PROVENANCE_TRAILER_KEY} ")
            ):
                claim = parse_issue_pr_provenance_line(line)
                if claim is None:  # pragma: no cover - guarded by the prefix check
                    raise AgentLoopError("Issue provenance trailer is malformed.")
                claims.append(claim)
    return tuple(claims)


def compare_issue_pr_provenance(
    claims: Iterable[IssuePrProvenanceScope],
    *,
    expected: IssuePrProvenanceScope,
) -> IssuePrProvenanceScope:
    """Validate a non-empty set of identical claims against an exact scope."""
    claims = tuple(claims)
    if not claims:
        raise AgentLoopError("No issue provenance trailer was found in the PR commits.")
    unique = set(claims)
    if len(unique) != 1:
        raise AgentLoopError("PR commit provenance contains conflicting issue scopes.")
    found = unique.pop()
    if found != expected:
        raise AgentLoopError(
            "PR commit provenance does not match the expected repository, issue, flow, or plan scope."
        )
    return found


validate_issue_pr_provenance = compare_issue_pr_provenance
