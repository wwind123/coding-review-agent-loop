"""Construction and reconciliation of the expected issue-closing contract."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from collections.abc import Iterable

from .errors import AgentLoopError


def normalize_issue_ids(
    value: Iterable[object] | None,
    *,
    field_name: str = "expected_closing_issue_ids",
) -> tuple[int, ...] | None:
    """Normalize an optional issue-id declaration without treating bool as int."""
    if value is None:
        return None
    result: set[int] = set()
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise AgentLoopError(
                f"{field_name} item at index {index} must be a positive integer (not a bool)."
            )
        result.add(raw)
    return tuple(sorted(result))


def contract_hash(issue_ids: Iterable[int]) -> str:
    normalized = normalize_issue_ids(issue_ids, field_name="contract") or ()
    return hashlib.sha256(
        json.dumps(list(normalized), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ExpectedClosingContract:
    issue_ids: tuple[int, ...]
    hash: str
    supersedes_hash: str | None = None


def make_contract(
    issue_ids: Iterable[object], *, supersedes_hash: str | None = None
) -> ExpectedClosingContract:
    normalized = normalize_issue_ids(issue_ids, field_name="expected_closing_issue_ids")
    assert normalized is not None
    return ExpectedClosingContract(normalized, contract_hash(normalized), supersedes_hash)


def _contract_conflict(
    *,
    recovered: tuple[int, ...],
    current: tuple[int, ...],
    supersede: bool,
) -> AgentLoopError:
    recovered_text = "{" + ", ".join(f"#{item}" for item in recovered) + "}"
    current_text = "{" + ", ".join(f"#{item}" for item in current) + "}"
    if supersede:
        detail = (
            "--supersede-expected-closing-contract only permits a proper superset; "
            "narrowing, replacement, and equality require canonical-metadata correction "
            "before resume."
        )
    else:
        detail = (
            "omit the explicit declaration to reuse the recovered contract, or use "
            "--supersede-expected-closing-contract for a proper-superset widening."
        )
    return AgentLoopError(
        "Expected closing contract conflict: recovered "
        f"{recovered_text}, current {current_text}. No durable metadata changed; {detail}"
    )


def reconcile_contracts(
    recovered: Iterable[object],
    current: Iterable[object],
    *,
    supersede: bool = False,
) -> ExpectedClosingContract:
    """Reconcile a durable contract with a current explicit declaration."""
    recovered_ids = normalize_issue_ids(recovered, field_name="recovered contract") or ()
    current_ids = normalize_issue_ids(current, field_name="current contract") or ()
    if recovered_ids == current_ids:
        if supersede:
            raise _contract_conflict(
                recovered=recovered_ids, current=current_ids, supersede=True
            )
        return make_contract(current_ids)
    if supersede and set(recovered_ids) < set(current_ids):
        return make_contract(current_ids, supersedes_hash=contract_hash(recovered_ids))
    raise _contract_conflict(
        recovered=recovered_ids, current=current_ids, supersede=supersede
    )


def resolve_issue_contract(
    *,
    primary_issue: int,
    cli_additions: Iterable[object] | None,
    plan_additions: Iterable[object] | None,
    recovered: Iterable[object] | None = None,
    supersede: bool = False,
) -> ExpectedClosingContract:
    """Resolve issue-mode additions while retaining the primary issue."""
    primary = normalize_issue_ids((primary_issue,), field_name="primary issue") or ()
    cli = normalize_issue_ids(cli_additions, field_name="--expected-closing-issue")
    plan = normalize_issue_ids(
        plan_additions, field_name="additional_closing_issue_ids"
    )
    explicit = cli is not None or plan is not None
    current = tuple(sorted(set(primary) | set(cli or ()) | set(plan or ())))
    if recovered is None:
        if supersede:
            raise AgentLoopError(
                "--supersede-expected-closing-contract requires an existing recovered "
                "expected closing contract; no durable metadata changed."
            )
        return make_contract(current)
    recovered_ids = normalize_issue_ids(recovered, field_name="recovered contract") or ()
    if primary_issue not in recovered_ids:
        raise AgentLoopError(
            f"Recovered expected closing contract {recovered_ids!r} does not retain "
            f"the actual primary issue #{primary_issue}. No durable metadata changed."
        )
    if not explicit:
        if supersede:
            raise AgentLoopError(
                "--supersede-expected-closing-contract requires an explicit full "
                "expected-closing declaration; no durable metadata changed."
            )
        return make_contract(recovered_ids)
    return reconcile_contracts(recovered_ids, current, supersede=supersede)


def resolve_direct_contract(
    *,
    explicit: Iterable[object] | None,
    recovered: Iterable[object] | None = None,
    supersede: bool = False,
) -> ExpectedClosingContract | None:
    """Resolve direct/managed PR mode without inferring from PR prose."""
    if recovered is None:
        if supersede:
            raise AgentLoopError(
                "--supersede-expected-closing-contract requires an existing recovered "
                "expected closing contract; no durable metadata changed."
            )
        return make_contract(explicit) if explicit is not None else None
    recovered_ids = normalize_issue_ids(recovered, field_name="recovered contract") or ()
    if explicit is None:
        if supersede:
            raise AgentLoopError(
                "--supersede-expected-closing-contract requires an explicit full "
                "expected-closing declaration; no durable metadata changed."
            )
        return make_contract(recovered_ids)
    return reconcile_contracts(recovered_ids, explicit, supersede=supersede)


def reject_parent_from_contract(
    contract: ExpectedClosingContract,
    *,
    parent_issue: int | None,
) -> None:
    if parent_issue is not None and parent_issue in contract.issue_ids:
        raise AgentLoopError(
            f"Expected closing contract includes staged parent issue #{parent_issue}. "
            "Use a child-scoped contract containing the child and its additions only; "
            "the parent must remain a non-closing `Refs` reference."
        )
