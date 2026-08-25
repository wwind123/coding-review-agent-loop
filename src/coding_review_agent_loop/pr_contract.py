"""Canonical PR-side expected-closing contract records."""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError
from .expected_closure import contract_hash, normalize_issue_ids

PR_EXPECTED_CLOSING_MARKER = "AGENT_PR_EXPECTED_CLOSING_ISSUES"
PR_EXPECTED_CLOSING_MARKER_RE = re.compile(
    rf"<!--\s*{PR_EXPECTED_CLOSING_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->"
    ,
    re.IGNORECASE,
)

_VALID_ORIGINS = {
    "issue-implementation",
    "approved-plan-implementation",
    "direct-pr",
    "managed-pr",
}


@dataclass(frozen=True)
class PrExpectedClosingContract:
    schema_version: int
    repository: str
    pr_number: int
    origin_flow: str
    primary_issue_number: int | None
    expected_closing_issue_ids: tuple[int, ...]
    contract_hash: str
    supersedes_hash: str | None = None


def _encode_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def _decode_payload(encoded: str) -> dict[str, object]:
    try:
        value = json.loads(base64.urlsafe_b64decode(encoded.encode("ascii")).decode("utf-8"))
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload.") from exc
    if not isinstance(value, dict):
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload.")
    return value


def encode_pr_contract(contract: PrExpectedClosingContract) -> str:
    return _encode_payload(
        {
            "schema_version": contract.schema_version,
            "repository": contract.repository,
            "pr_number": contract.pr_number,
            "origin_flow": contract.origin_flow,
            "primary_issue_number": contract.primary_issue_number,
            "expected_closing_issue_ids": list(contract.expected_closing_issue_ids),
            "contract_hash": contract.contract_hash,
            "supersedes_hash": contract.supersedes_hash,
        }
    )


def decode_pr_contract(encoded: str) -> PrExpectedClosingContract:
    payload = _decode_payload(encoded)
    required = {
        "schema_version",
        "repository",
        "pr_number",
        "origin_flow",
        "primary_issue_number",
        "expected_closing_issue_ids",
        "contract_hash",
        "supersedes_hash",
    }
    if set(payload) != required:
        raise AgentLoopError(
            f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: expected exactly "
            f"{', '.join(sorted(required))}."
        )
    version = payload["schema_version"]
    pr_number = payload["pr_number"]
    primary = payload["primary_issue_number"]
    if isinstance(version, bool) or version != 1:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: schema_version must be 1.")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: pr_number must be positive.")
    if primary is not None and (
        isinstance(primary, bool) or not isinstance(primary, int) or primary <= 0
    ):
        raise AgentLoopError(
            f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: primary_issue_number is invalid."
        )
    repository = payload["repository"]
    origin = payload["origin_flow"]
    digest = payload["contract_hash"]
    supersedes = payload["supersedes_hash"]
    if not isinstance(repository, str) or not repository.strip():
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: repository is invalid.")
    if origin not in _VALID_ORIGINS:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: origin_flow is invalid.")
    if supersedes is not None and (not isinstance(supersedes, str) or not supersedes.strip()):
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: supersedes_hash is invalid.")
    ids = normalize_issue_ids(
        payload["expected_closing_issue_ids"],
        field_name=f"{PR_EXPECTED_CLOSING_MARKER}.expected_closing_issue_ids",
    )
    assert ids is not None
    if not isinstance(digest, str) or digest != contract_hash(ids):
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} payload: contract_hash is invalid.")
    return PrExpectedClosingContract(
        schema_version=1,
        repository=repository,
        pr_number=pr_number,
        origin_flow=str(origin),
        primary_issue_number=primary,
        expected_closing_issue_ids=ids,
        contract_hash=digest,
        supersedes_hash=supersedes,
    )


def make_pr_contract(
    *,
    repository: str,
    pr_number: int,
    origin_flow: str,
    expected_closing_issue_ids: Sequence[int],
    primary_issue_number: int | None = None,
    supersedes_hash: str | None = None,
) -> PrExpectedClosingContract:
    if origin_flow not in _VALID_ORIGINS:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} origin_flow.")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} pr_number.")
    if primary_issue_number is not None and (
        isinstance(primary_issue_number, bool)
        or not isinstance(primary_issue_number, int)
        or primary_issue_number <= 0
    ):
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} primary_issue_number.")
    if not isinstance(repository, str) or not repository.strip():
        raise AgentLoopError(f"Invalid {PR_EXPECTED_CLOSING_MARKER} repository.")
    ids = normalize_issue_ids(expected_closing_issue_ids, field_name="expected_closing_issue_ids")
    assert ids is not None
    return PrExpectedClosingContract(
        schema_version=1,
        repository=repository,
        pr_number=pr_number,
        origin_flow=origin_flow,
        primary_issue_number=primary_issue_number,
        expected_closing_issue_ids=ids,
        contract_hash=contract_hash(ids),
        supersedes_hash=supersedes_hash,
    )


def format_pr_contract_comment(contract: PrExpectedClosingContract) -> str:
    if encode_pr_contract(decode_pr_contract(encode_pr_contract(contract))) != encode_pr_contract(contract):
        raise AgentLoopError("PR expected-closing contract failed canonical rendering validation.")
    issue_text = ", ".join(f"#{item}" for item in contract.expected_closing_issue_ids) or "(none)"
    return "\n".join(
        [
            f"Expected closing issues for PR #{contract.pr_number}: {issue_text}.",
            "",
            f"Origin flow: {contract.origin_flow}",
            f"Contract hash: {contract.contract_hash}",
            f"<!-- {PR_EXPECTED_CLOSING_MARKER}: {encode_pr_contract(contract)} -->",
            "-- coding-review-agent-loop",
        ]
    )


def render_pr_contract_marker(contract: PrExpectedClosingContract) -> str:
    """Return only the canonical HTML marker for embedding in an existing post."""
    encoded = encode_pr_contract(contract)
    # Round-trip before exposing the marker so trusted callers cannot embed a
    # hand-edited payload.
    if encode_pr_contract(decode_pr_contract(encoded)) != encoded:
        raise AgentLoopError("PR expected-closing contract failed canonical encoding validation.")
    return f"<!-- {PR_EXPECTED_CLOSING_MARKER}: {encoded} -->"


def find_latest_pr_contract(
    comments: Sequence[object], *, repository: str, pr_number: int
) -> PrExpectedClosingContract | None:
    found: PrExpectedClosingContract | None = None
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PR_EXPECTED_CLOSING_MARKER_RE.finditer(body):
            contract = decode_pr_contract(match.group("payload"))
            if contract.repository.casefold() != repository.casefold() or contract.pr_number != pr_number:
                raise AgentLoopError(
                    f"{PR_EXPECTED_CLOSING_MARKER} record does not belong to {repository} PR #{pr_number}."
                )
            if encode_pr_contract(contract) != match.group("payload"):
                raise AgentLoopError(
                    f"{PR_EXPECTED_CLOSING_MARKER} record is not canonically encoded."
                )
            if found is not None and found != contract:
                if (
                    contract.supersedes_hash == found.contract_hash
                    and set(found.expected_closing_issue_ids) < set(contract.expected_closing_issue_ids)
                ):
                    found = contract
                    continue
                raise AgentLoopError(
                    f"Divergent {PR_EXPECTED_CLOSING_MARKER} records were found for PR #{pr_number}."
                )
            found = contract
    return found
