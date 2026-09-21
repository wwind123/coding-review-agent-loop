"""Canonical PR-side expected-closing contract records."""

from __future__ import annotations

import base64
import hashlib
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


# ---------------------------------------------------------------------------
# Version 2: transaction-bound contract records (#827).
#
# The functions below are a separate codec.  ``find_latest_pr_contract`` and
# ``decode_pr_contract`` are deliberately not taught version 2: they keep
# rejecting it, so a comment snapshot that was not read through the
# author-authenticated reader can never be interpreted as a version-2 record.
# ---------------------------------------------------------------------------

PR_CONTRACT_V2_SCHEMA_VERSION = 2
# One successor transaction can change both the origin flow and the closing
# scope (its kind is the higher-precedence one), and a record carries exactly
# one supersession kind, so the combined change has its own atomic kind.
PR_CONTRACT_SUPERSESSION_COMBINED = "flow-correction-with-closing-widening"
PR_CONTRACT_SUPERSESSION_KINDS = frozenset(
    {"closing-widening", "flow-correction", PR_CONTRACT_SUPERSESSION_COMBINED}
)
_HEX64_RE = re.compile(r"\A[0-9a-f]{64}\Z")
_V2_REPOSITORY_RE = re.compile(r"\A[A-Za-z0-9._-]+/[A-Za-z0-9._-]+\Z")
_V2_ISSUE_ORIGIN_FLOWS = frozenset({"issue-implementation", "approved-plan-implementation"})
_V2_APPROVED_PLAN_FLOW = "approved-plan-implementation"
_V2_PLAN_HASH_RE = re.compile(r"\A[0-9a-f]{16}\Z")


@dataclass(frozen=True)
class PrExpectedClosingContractV2:
    repository: str
    pr_number: int
    origin_flow: str
    primary_issue_number: int | None
    expected_closing_issue_ids: tuple[int, ...]
    contract_hash: str
    transaction_id: str
    supersession_kind: str | None = None
    supersedes_record_hash: str | None = None
    # Present exactly when the origin flow is the approved-plan flow, so the PR
    # surface carries the same plan identity as the issue-side handoff.
    approved_plan_hash: str | None = None
    schema_version: int = PR_CONTRACT_V2_SCHEMA_VERSION


def _v2_payload(contract: PrExpectedClosingContractV2) -> dict[str, object]:
    return {
        "schema_version": contract.schema_version,
        "repository": contract.repository,
        "pr_number": contract.pr_number,
        "origin_flow": contract.origin_flow,
        "approved_plan_hash": contract.approved_plan_hash,
        "primary_issue_number": contract.primary_issue_number,
        "expected_closing_issue_ids": list(contract.expected_closing_issue_ids),
        "contract_hash": contract.contract_hash,
        "transaction_id": contract.transaction_id,
        "supersession_kind": contract.supersession_kind,
        "supersedes_record_hash": contract.supersedes_record_hash,
    }


def make_pr_contract_v2(
    *,
    repository: str,
    pr_number: int,
    origin_flow: str,
    expected_closing_issue_ids: Sequence[int],
    transaction_id: str,
    primary_issue_number: int | None = None,
    supersession_kind: str | None = None,
    supersedes_record_hash: str | None = None,
    approved_plan_hash: str | None = None,
) -> PrExpectedClosingContractV2:
    ids = normalize_issue_ids(expected_closing_issue_ids, field_name="expected_closing_issue_ids")
    assert ids is not None
    contract = PrExpectedClosingContractV2(
        repository=repository,
        pr_number=pr_number,
        origin_flow=origin_flow,
        primary_issue_number=primary_issue_number,
        expected_closing_issue_ids=ids,
        contract_hash=contract_hash(ids),
        transaction_id=transaction_id,
        supersession_kind=supersession_kind,
        supersedes_record_hash=supersedes_record_hash,
        approved_plan_hash=approved_plan_hash,
    )
    # Validate through the strict decoder so construction and parsing agree.
    return decode_pr_contract_v2(encode_pr_contract_v2(contract))


def encode_pr_contract_v2(contract: PrExpectedClosingContractV2) -> str:
    return _encode_payload(_v2_payload(contract))


def decode_pr_contract_v2(encoded: str) -> PrExpectedClosingContractV2:
    payload = _decode_payload(encoded)
    required = {
        "schema_version",
        "repository",
        "pr_number",
        "origin_flow",
        "approved_plan_hash",
        "primary_issue_number",
        "expected_closing_issue_ids",
        "contract_hash",
        "transaction_id",
        "supersession_kind",
        "supersedes_record_hash",
    }
    prefix = f"Invalid {PR_EXPECTED_CLOSING_MARKER} v2 payload"
    if set(payload) != required:
        raise AgentLoopError(f"{prefix}: expected exactly {', '.join(sorted(required))}.")
    version = payload["schema_version"]
    if isinstance(version, bool) or version != PR_CONTRACT_V2_SCHEMA_VERSION:
        raise AgentLoopError(f"{prefix}: schema_version must be 2.")
    pr_number = payload["pr_number"]
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise AgentLoopError(f"{prefix}: pr_number must be positive.")
    primary = payload["primary_issue_number"]
    if primary is not None and (
        isinstance(primary, bool) or not isinstance(primary, int) or primary <= 0
    ):
        raise AgentLoopError(f"{prefix}: primary_issue_number is invalid.")
    repository = payload["repository"]
    if not isinstance(repository, str) or _V2_REPOSITORY_RE.match(repository) is None:
        raise AgentLoopError(f"{prefix}: repository must be OWNER/NAME.")
    origin = payload["origin_flow"]
    if origin not in _VALID_ORIGINS:
        raise AgentLoopError(f"{prefix}: origin_flow is invalid.")
    if origin in _V2_ISSUE_ORIGIN_FLOWS and primary is None:
        raise AgentLoopError(f"{prefix}: {origin} requires primary_issue_number.")
    plan_hash = payload["approved_plan_hash"]
    if origin == _V2_APPROVED_PLAN_FLOW:
        if not isinstance(plan_hash, str) or _V2_PLAN_HASH_RE.match(plan_hash) is None:
            raise AgentLoopError(
                f"{prefix}: approved_plan_hash is required for {_V2_APPROVED_PLAN_FLOW}."
            )
    elif plan_hash is not None:
        raise AgentLoopError(f"{prefix}: approved_plan_hash must be absent for {origin}.")
    ids = normalize_issue_ids(
        payload["expected_closing_issue_ids"],
        field_name=f"{PR_EXPECTED_CLOSING_MARKER}.expected_closing_issue_ids",
    )
    assert ids is not None
    if list(ids) != payload["expected_closing_issue_ids"]:
        raise AgentLoopError(f"{prefix}: expected_closing_issue_ids are not canonical.")
    if primary is not None and primary not in ids:
        raise AgentLoopError(
            f"{prefix}: expected_closing_issue_ids must retain primary issue #{primary}."
        )
    digest = payload["contract_hash"]
    if not isinstance(digest, str) or digest != contract_hash(ids):
        raise AgentLoopError(f"{prefix}: contract_hash is invalid.")
    transaction_id = payload["transaction_id"]
    if not isinstance(transaction_id, str) or _HEX64_RE.match(transaction_id) is None:
        raise AgentLoopError(f"{prefix}: transaction_id is invalid.")
    kind = payload["supersession_kind"]
    supersedes = payload["supersedes_record_hash"]
    if (kind is None) != (supersedes is None):
        raise AgentLoopError(
            f"{prefix}: supersession_kind and supersedes_record_hash must be set together."
        )
    if kind is not None and kind not in PR_CONTRACT_SUPERSESSION_KINDS:
        raise AgentLoopError(f"{prefix}: supersession_kind is invalid.")
    if supersedes is not None and (
        not isinstance(supersedes, str) or _HEX64_RE.match(supersedes) is None
    ):
        raise AgentLoopError(f"{prefix}: supersedes_record_hash is invalid.")
    contract = PrExpectedClosingContractV2(
        repository=repository,
        pr_number=pr_number,
        origin_flow=str(origin),
        primary_issue_number=primary,
        expected_closing_issue_ids=ids,
        contract_hash=digest,
        transaction_id=transaction_id,
        supersession_kind=kind if isinstance(kind, str) else None,
        supersedes_record_hash=supersedes if isinstance(supersedes, str) else None,
        approved_plan_hash=plan_hash if isinstance(plan_hash, str) else None,
    )
    if encode_pr_contract_v2(contract) != encoded:
        raise AgentLoopError(f"{prefix}: record is not canonically encoded.")
    return contract


def pr_contract_record_hash(
    contract: PrExpectedClosingContract | PrExpectedClosingContractV2,
) -> str:
    """Hash the full canonical payload, so the digest identifies flow too.

    ``expected_closure.contract_hash`` covers closing IDs only and therefore
    cannot distinguish two records that differ in origin flow.
    """
    encoded = (
        encode_pr_contract_v2(contract)
        if isinstance(contract, PrExpectedClosingContractV2)
        else encode_pr_contract(contract)
    )
    return hashlib.sha256(base64.urlsafe_b64decode(encoded.encode("ascii"))).hexdigest()


def pr_contract_payload_schema_version(encoded: str) -> object:
    """Peek at a payload's declared version without interpreting the record."""
    return _decode_payload(encoded).get("schema_version")


def format_pr_contract_v2_comment(contract: PrExpectedClosingContractV2) -> str:
    encoded = encode_pr_contract_v2(contract)
    if encode_pr_contract_v2(decode_pr_contract_v2(encoded)) != encoded:
        raise AgentLoopError("PR expected-closing contract failed canonical rendering validation.")
    issue_text = ", ".join(f"#{item}" for item in contract.expected_closing_issue_ids) or "(none)"
    lines = [
        f"Expected closing issues for PR #{contract.pr_number}: {issue_text}.",
        "",
        f"Origin flow: {contract.origin_flow}",
        f"Contract hash: {contract.contract_hash}",
        f"Workflow transaction: {contract.transaction_id}",
    ]
    if contract.approved_plan_hash is not None:
        lines.append(f"Plan hash: {contract.approved_plan_hash}")
    if contract.supersession_kind is not None:
        lines.append(
            f"Supersedes record {contract.supersedes_record_hash} ({contract.supersession_kind})."
        )
    lines.extend(
        [
            f"<!-- {PR_EXPECTED_CLOSING_MARKER}: {encoded} -->",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)
