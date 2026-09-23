"""Signed reviewer-board amendment records (#943).

A persisted scheduler contract is immutable for the run: a resume that
silently narrowed the required reviewer board would weaken the guarantee that
every configured reviewer approved the exact artifact.  This module adds the
one audited exception.  A human operator posts a signed
``reviewer-board-amendment`` record that removes unavailable non-primary
reviewers from the persisted board, starting at an explicit round.

The record is posted on the same comment surface as the round records it
amends: the owning issue for planning, and the PR itself for PR review
(including standalone PR runs).  Plan and PR resume, and the PR
qualification gate, all validate history through :func:`resolve_contract_lineage`,
so the three gates share one parser, one activation rule, and one lineage rule.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from .errors import AgentLoopError
from .protocol import UnresolvedReviewItem, parse_signed_human_requirement_body
from .round_state import PostedRoundRecord

REVIEWER_BOARD_AMENDMENT_KIND = "reviewer-board-amendment"
REVIEWER_BOARD_AMENDMENT_SCHEMA_VERSION = 1
REVIEWER_BOARD_AMENDMENT_REASONS = frozenset({"backend-unavailable"})
REVIEWER_BOARD_AMENDMENT_AUDIT_HEADING = "Reviewer board amendment applied."
_AMENDMENT_RECORD_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "flow",
        "issue",
        "pr_number",
        "original_required_reviewers",
        "policy",
        "primary_reviewer",
        "removed_reviewers",
        "effective_from_round",
        "reason",
        "rationale",
    }
)
_ACTIVE_LEDGER_STATUSES = frozenset({"blocking", "same-plan", "same-pr"})
_FENCED_JSON_RE = re.compile(r"```(?:json)?[ \t]*\n(?P<body>.*?)\n```", re.S | re.I)

ContractT = TypeVar("ContractT")


@dataclass(frozen=True)
class ReviewerBoardAmendment:
    """One signed human authorization to remove reviewers from a persisted board."""

    flow: str
    issue: int | None
    pr_number: int | None
    original_required_reviewers: tuple[str, ...]
    policy: str
    primary_reviewer: str | None
    removed_reviewers: tuple[str, ...]
    effective_from_round: int
    reason: str
    rationale: str
    digest: str
    comment_locator: str
    comment_index: int

    @property
    def amended_required_reviewers(self) -> tuple[str, ...]:
        removed = set(self.removed_reviewers)
        return tuple(
            name for name in self.original_required_reviewers if name not in removed
        )


@dataclass(frozen=True)
class ContractLineage:
    """The validated amendment chain C0 -> C1 -> ... -> Cn for one surface."""

    contracts: tuple[object, ...]
    amendments: tuple[ReviewerBoardAmendment, ...]
    # Amendments that no digest-bound contract-bearing record has used yet.
    # Their activation round must match the round the resume re-enters.
    pending_amendments: tuple[ReviewerBoardAmendment, ...]

    @property
    def active_amendment(self) -> ReviewerBoardAmendment | None:
        return self.amendments[-1] if self.amendments else None

    @property
    def active_digest(self) -> str | None:
        active = self.active_amendment
        return active.digest if active is not None else None

    @property
    def removed_reviewers(self) -> tuple[str, ...]:
        return tuple(
            name for amendment in self.amendments for name in amendment.removed_reviewers
        )


@dataclass(frozen=True)
class LedgerReassignment:
    item_id: str
    removed: tuple[str, ...]
    new_owners: tuple[str, ...]


def reviewer_board_amendment_digest(record: dict[str, object]) -> str:
    """Canonical digest shared by discovery, round metadata, and qualification."""
    canonical = json.dumps(
        {key: record[key] for key in sorted(_AMENDMENT_RECORD_KEYS)},
        separators=(",", ":"),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reviewer_board_amendment_payload(
    *,
    flow: str,
    issue: int | None,
    pr_number: int | None,
    original_required_reviewers: Sequence[str],
    policy: str,
    primary_reviewer: str | None,
    removed_reviewers: Sequence[str],
    effective_from_round: int,
    reason: str = "backend-unavailable",
    rationale: str,
) -> dict[str, object]:
    return {
        "kind": REVIEWER_BOARD_AMENDMENT_KIND,
        "schema_version": REVIEWER_BOARD_AMENDMENT_SCHEMA_VERSION,
        "flow": flow,
        "issue": issue,
        "pr_number": pr_number,
        "original_required_reviewers": list(original_required_reviewers),
        "policy": policy,
        "primary_reviewer": primary_reviewer,
        "removed_reviewers": list(removed_reviewers),
        "effective_from_round": effective_from_round,
        "reason": reason,
        "rationale": rationale,
    }


def format_reviewer_board_amendment_comment(
    *,
    flow: str,
    issue: int | None,
    pr_number: int | None,
    original_required_reviewers: Sequence[str],
    policy: str,
    primary_reviewer: str | None,
    removed_reviewers: Sequence[str],
    effective_from_round: int,
    reason: str = "backend-unavailable",
    rationale: str = "<why the removed reviewer cannot be reached>",
) -> str:
    """Render the signed amendment record format documented for operators."""
    record = reviewer_board_amendment_payload(
        flow=flow,
        issue=issue,
        pr_number=pr_number,
        original_required_reviewers=original_required_reviewers,
        policy=policy,
        primary_reviewer=primary_reviewer,
        removed_reviewers=removed_reviewers,
        effective_from_round=effective_from_round,
        reason=reason,
        rationale=rationale,
    )
    return (
        "Reviewer board amendment:\n\n```json\n"
        + json.dumps(record, indent=2, sort_keys=True)
        + "\n```\n-- Human Reviewer"
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _unique_name_list(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and item.strip() for item in value)
        and len(set(value)) == len(value)
    )


def _amendment_record_problem(payload: dict[str, object]) -> str | None:
    keys = set(payload)
    if keys != _AMENDMENT_RECORD_KEYS:
        missing = sorted(_AMENDMENT_RECORD_KEYS - keys)
        unknown = sorted(keys - _AMENDMENT_RECORD_KEYS)
        return f"missing keys {missing}, unknown keys {unknown}"
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != REVIEWER_BOARD_AMENDMENT_SCHEMA_VERSION
    ):
        return "schema_version must be 1"
    flow = payload.get("flow")
    if flow == "plan":
        if not _is_positive_int(payload.get("issue")):
            return "issue must be a positive integer for a plan amendment"
        if payload.get("pr_number") is not None:
            return "pr_number must be null for a plan amendment"
    elif flow == "pr":
        if not _is_positive_int(payload.get("pr_number")):
            return "pr_number must be a positive integer for a pr amendment"
        if payload.get("issue") is not None:
            return "issue must be null for a pr amendment"
    else:
        return "flow must be `plan` or `pr`"
    if not _unique_name_list(payload.get("original_required_reviewers")):
        return "original_required_reviewers must be a non-empty list of unique names"
    if not _unique_name_list(payload.get("removed_reviewers")):
        return "removed_reviewers must be a non-empty list of unique names"
    policy = payload.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        return "policy must be a non-empty string"
    primary = payload.get("primary_reviewer")
    if primary is not None and (not isinstance(primary, str) or not primary.strip()):
        return "primary_reviewer must be a non-empty string or null"
    if not _is_positive_int(payload.get("effective_from_round")):
        return "effective_from_round must be a positive integer"
    if payload.get("reason") not in REVIEWER_BOARD_AMENDMENT_REASONS:
        return f"reason must be one of {sorted(REVIEWER_BOARD_AMENDMENT_REASONS)}"
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return "rationale must be a non-empty string"
    return None


def parse_reviewer_board_amendment_records(
    body: str | None, *, comment_locator: str, comment_index: int = -1
) -> tuple[tuple[ReviewerBoardAmendment, ...], tuple[str, ...]]:
    """Return (signed valid records, ignored-record diagnostics) for one comment.

    Only a body with the standalone human reviewer signature counts, and a
    malformed record is reported and ignored so it can never be applied.
    """
    signed = parse_signed_human_requirement_body(body)
    if signed is None:
        return (), ()
    records: list[ReviewerBoardAmendment] = []
    ignored: list[str] = []
    for match in _FENCED_JSON_RE.finditer(signed):
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError as exc:
            # A fence that names the record kind but is not valid JSON is a
            # malformed amendment: report it rather than skip it silently.
            if REVIEWER_BOARD_AMENDMENT_KIND in match.group("body"):
                ignored.append(
                    f"{comment_locator}: malformed reviewer-board amendment record ignored "
                    f"(invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno})"
                )
            continue
        if not isinstance(payload, dict) or payload.get("kind") != REVIEWER_BOARD_AMENDMENT_KIND:
            continue
        problem = _amendment_record_problem(payload)
        if problem is not None:
            ignored.append(
                f"{comment_locator}: malformed reviewer-board amendment record ignored ({problem})"
            )
            continue
        records.append(
            ReviewerBoardAmendment(
                flow=str(payload["flow"]),
                issue=payload["issue"],  # type: ignore[arg-type]
                pr_number=payload["pr_number"],  # type: ignore[arg-type]
                original_required_reviewers=tuple(payload["original_required_reviewers"]),  # type: ignore[arg-type]
                policy=str(payload["policy"]),
                primary_reviewer=payload["primary_reviewer"],  # type: ignore[arg-type]
                removed_reviewers=tuple(payload["removed_reviewers"]),  # type: ignore[arg-type]
                effective_from_round=int(payload["effective_from_round"]),  # type: ignore[arg-type]
                reason=str(payload["reason"]),
                rationale=str(payload["rationale"]),
                digest=reviewer_board_amendment_digest(payload),
                comment_locator=comment_locator,
                comment_index=comment_index,
            )
        )
    return tuple(records), tuple(ignored)


def is_reviewer_board_amendment_only(signed_body: str | None) -> bool:
    """Whether a signed comment body carries nothing but amendment record(s).

    Such a comment is an orchestration record, not a requirement on the
    reviewed artifact, so it is never surfaced as a signed human requirement
    that every reviewer must disposition.  Any other text in the comment keeps
    the whole comment a signed requirement.
    """
    if not signed_body:
        return False
    found = False

    def strip(match: re.Match[str]) -> str:
        nonlocal found
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            return match.group(0)
        if isinstance(payload, dict) and payload.get("kind") == REVIEWER_BOARD_AMENDMENT_KIND:
            found = True
            return ""
        return match.group(0)

    remainder = _FENCED_JSON_RE.sub(strip, signed_body)
    remainder = re.sub(r"(?im)^\s*reviewer board amendment:?\s*$", "", remainder)
    return found and not remainder.strip()


def _surface_label(flow: str, number: int) -> str:
    return f"issue #{number}" if flow == "plan" else f"PR #{number}"


def collect_reviewer_board_amendments(
    comments: Sequence[object],
    *,
    flow: str,
    issue_number: int | None = None,
    pr_number: int | None = None,
    ignored_sink: list[str] | None = None,
) -> tuple[ReviewerBoardAmendment, ...]:
    """Signed amendment discovery on exactly one comment surface.

    ``flow="plan"`` reads the owning issue's comments; ``flow="pr"`` reads the
    PR's own comments.  A signed record for the other flow, or naming another
    issue or PR, fails closed rather than being silently ignored.  Identical
    duplicates collapse by digest (the earliest comment is kept); distinct
    records are returned in comment order and chained by
    :func:`resolve_contract_lineage`, which never picks one by comment order.
    """
    if flow == "plan":
        if issue_number is None:
            raise AgentLoopError("Plan reviewer-board amendment discovery needs an issue number.")
        surface = _surface_label("plan", issue_number)
    elif flow == "pr":
        if pr_number is None:
            raise AgentLoopError("PR reviewer-board amendment discovery needs a PR number.")
        surface = _surface_label("pr", pr_number)
    else:
        raise AgentLoopError(f"Unknown reviewer-board amendment flow {flow!r}.")
    by_digest: dict[str, ReviewerBoardAmendment] = {}
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        records, ignored = parse_reviewer_board_amendment_records(
            body, comment_locator=f"{surface} comment {index + 1}", comment_index=index
        )
        if ignored_sink is not None:
            ignored_sink.extend(ignored)
        for record in records:
            if record.flow != flow:
                target = (
                    _surface_label("pr", record.pr_number)
                    if record.flow == "pr" and record.pr_number is not None
                    else _surface_label("plan", record.issue)
                    if record.flow == "plan" and record.issue is not None
                    else "its own surface"
                )
                raise AgentLoopError(
                    "Human decision required: signed reviewer-board amendment at "
                    f"{record.comment_locator} is a `{record.flow}` record, but {surface} only "
                    f"accepts `{flow}` amendments. Post this record on {target} instead, and "
                    "remove it from here before rerunning."
                )
            if flow == "plan" and record.issue != issue_number:
                raise AgentLoopError(
                    "Human decision required: signed reviewer-board amendment at "
                    f"{record.comment_locator} names issue #{record.issue}, but it is posted "
                    f"on issue #{issue_number}. Post this record on issue #{record.issue}, "
                    "or correct it, before rerunning."
                )
            if flow == "pr" and record.pr_number != pr_number:
                raise AgentLoopError(
                    "Human decision required: signed reviewer-board amendment at "
                    f"{record.comment_locator} names PR #{record.pr_number}, but it is posted "
                    f"on PR #{pr_number}. Post this record on PR #{record.pr_number}, "
                    "or correct it, before rerunning."
                )
            by_digest.setdefault(record.digest, record)
    return tuple(sorted(by_digest.values(), key=lambda record: record.comment_index))


def reject_misplaced_pr_amendments(
    issue_comments: Sequence[object], *, issue_number: int
) -> None:
    """A signed ``pr`` amendment on the owning issue fails closed (#943).

    Issue and PR comments are never ordered against each other, so a PR
    amendment is only valid on the PR itself.
    """
    for index, comment in enumerate(issue_comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        records, _ignored = parse_reviewer_board_amendment_records(
            body,
            comment_locator=f"issue #{issue_number} comment {index + 1}",
            comment_index=index,
        )
        for record in records:
            if record.flow == "pr":
                raise AgentLoopError(
                    "Human decision required: signed `pr` reviewer-board amendment at "
                    f"{record.comment_locator} is posted on the owning issue. Post this "
                    f"record on PR #{record.pr_number} instead, and remove it from issue "
                    f"#{issue_number} before rerunning."
                )


def amend_contract(original: ContractT, amendment: ReviewerBoardAmendment) -> ContractT:
    """Apply one amendment to a plan or PR scheduler contract, fail-closed.

    Policy, primary, and (for PR contracts) broad-path rules stay unchanged;
    only ``removed_reviewers`` leave ``required_reviewers``.  The contracts'
    own ``__post_init__`` rules reject a primary-then-panel board left with no
    secondary.
    """
    required = tuple(getattr(original, "required_reviewers"))
    policy = getattr(original, "policy")
    primary = getattr(original, "primary_reviewer")

    def fail(problem: str) -> AgentLoopError:
        return AgentLoopError(
            "Human decision required: signed reviewer-board amendment at "
            f"{amendment.comment_locator} cannot be applied: {problem}. Correct or remove "
            "the record before rerunning."
        )

    if tuple(amendment.original_required_reviewers) != required:
        raise fail(
            f"its original_required_reviewers {list(amendment.original_required_reviewers)} "
            f"do not match the persisted board {list(required)}"
        )
    if amendment.policy != policy:
        raise fail(f"it names policy {amendment.policy!r} but the persisted policy is {policy!r}")
    if amendment.primary_reviewer != primary:
        raise fail(
            f"it names primary {amendment.primary_reviewer!r} but the persisted primary is "
            f"{primary!r}"
        )
    unknown = [name for name in amendment.removed_reviewers if name not in required]
    if unknown:
        raise fail(f"removed reviewer(s) {unknown} are not on the persisted board")
    if primary is not None and primary in amendment.removed_reviewers:
        raise fail(f"it removes the primary reviewer {primary}; the primary is immutable")
    remaining = amendment.amended_required_reviewers
    if not remaining:
        raise fail("it removes every reviewer")
    try:
        return dataclasses.replace(original, required_reviewers=remaining)  # type: ignore[type-var]
    except AgentLoopError as exc:
        raise fail(str(exc)) from exc


def build_amendment_chain(
    base_contract: ContractT, amendments: Sequence[ReviewerBoardAmendment]
) -> tuple[tuple[ContractT, ...], tuple[ReviewerBoardAmendment, ...]]:
    """Order amendments into one linear chain from ``base_contract``.

    Two distinct records amending the same board, a gap, a fork, a
    non-increasing effective round, or a link posted before its predecessor
    all fail closed.  An amendment that matches no link is unmatched.
    """
    contracts: list[ContractT] = [base_contract]
    chain: list[ReviewerBoardAmendment] = []
    remaining = list(amendments)
    while remaining:
        board = tuple(getattr(contracts[-1], "required_reviewers"))
        matches = [
            record for record in remaining
            if tuple(record.original_required_reviewers) == board
        ]
        if not matches:
            break
        if len(matches) > 1:
            raise AgentLoopError(
                "Human decision required: signed reviewer-board amendments at "
                + ", ".join(record.comment_locator for record in matches)
                + f" all amend the board {list(board)}. The orchestrator never chooses "
                "between them by comment order; keep exactly one and remove the others."
            )
        record = matches[0]
        if chain:
            previous = chain[-1]
            if record.effective_from_round <= previous.effective_from_round:
                raise AgentLoopError(
                    "Human decision required: chained reviewer-board amendment at "
                    f"{record.comment_locator} is effective from round "
                    f"{record.effective_from_round}, which does not follow round "
                    f"{previous.effective_from_round} of the amendment it chains from "
                    f"({previous.comment_locator})."
                )
            if record.comment_index <= previous.comment_index:
                raise AgentLoopError(
                    "Human decision required: chained reviewer-board amendment at "
                    f"{record.comment_locator} is posted before the amendment it chains "
                    f"from ({previous.comment_locator})."
                )
        contracts.append(amend_contract(contracts[-1], record))
        chain.append(record)
        remaining.remove(record)
    if remaining:
        raise AgentLoopError(
            "Human decision required: signed reviewer-board amendment(s) at "
            + ", ".join(record.comment_locator for record in remaining)
            + " match no persisted scheduler contract board in this run's amendment "
            "chain (original_required_reviewers must equal the persisted board, or the "
            "board produced by the previous amendment). Correct or remove the record "
            "before rerunning."
        )
    return tuple(contracts), tuple(chain)


def _describe_contract(contract: object) -> str:
    return (
        f"policy {getattr(contract, 'policy')} with primary "
        f"{getattr(contract, 'primary_reviewer') or '(none)'} and reviewer board "
        f"{', '.join(getattr(contract, 'required_reviewers'))}"
    )


def resolve_contract_lineage(
    records: Sequence[PostedRoundRecord],
    amendments: Sequence[ReviewerBoardAmendment],
    configured_contract: ContractT | None,
    *,
    contract_from_metadata: Callable[[object], ContractT | None],
    drift_error: Callable[[ContractT, str], AgentLoopError],
) -> ContractLineage:
    """Validate contract-bearing history against the signed amendment chain.

    Contract-bearing records are those whose decoded scheduler contract is
    non-null; every other record is contract-neutral and never compared.  C0
    is the contract on the earliest contract-bearing record.  Each record is
    judged by exactly one governing link: the latest amendment whose comment
    precedes the record and whose effective round is at or before the
    record's round.  It must carry exactly that link's contract and digest
    (link 0 carries C0 and no digest).  The configured contract must equal
    Cn.  ``drift_error(persisted_contract, detail)`` builds the fail-closed
    error for any mismatch.
    """
    ordered = sorted(records, key=lambda record: record.index)
    bearing: list[tuple[PostedRoundRecord, ContractT]] = []
    for record in ordered:
        contract = contract_from_metadata(record.metadata)
        if contract is not None:
            bearing.append((record, contract))
        elif record.metadata.reviewer_board_amendment_digest is not None:
            raise AgentLoopError(
                "A contract-neutral round record carries a reviewer-board amendment "
                f"digest (comment {record.index + 1}); the scheduler contract history is "
                "unexplained. Rerun is refused until the history is corrected."
            )
    if not bearing:
        if amendments:
            raise AgentLoopError(
                "Human decision required: signed reviewer-board amendment(s) at "
                + ", ".join(record.comment_locator for record in amendments)
                + " are unmatched: this run has no persisted scheduler contract to amend "
                "(all-reviewers PR runs and the non-staged plan path persist none). "
                "Change the reviewer flags directly and delete the record."
            )
        return ContractLineage(
            contracts=((configured_contract,) if configured_contract is not None else ()),
            amendments=(),
            pending_amendments=(),
        )
    base = bearing[0][1]
    contracts, chain = build_amendment_chain(base, amendments)
    digests = {record.digest: position + 1 for position, record in enumerate(chain)}
    used: set[str] = set()
    for record, contract in bearing:
        governing = 0
        for position, amendment in enumerate(chain, start=1):
            if (
                amendment.comment_index < record.index
                and record.metadata.round_number >= amendment.effective_from_round
            ):
                governing = position
        expected_contract = contracts[governing]
        expected_digest = chain[governing - 1].digest if governing else None
        digest = record.metadata.reviewer_board_amendment_digest
        if contract != expected_contract or digest != expected_digest:
            if digest is not None and digest not in digests:
                detail = (
                    f"round {record.metadata.round_number} scheduler record (comment "
                    f"{record.index + 1}) carries an unknown reviewer-board amendment digest"
                )
            elif governing:
                detail = (
                    f"round {record.metadata.round_number} scheduler record (comment "
                    f"{record.index + 1}) was posted under the amendment at "
                    f"{chain[governing - 1].comment_locator} but does not carry its amended "
                    "contract and digest"
                )
            else:
                detail = (
                    f"round {record.metadata.round_number} scheduler record (comment "
                    f"{record.index + 1}) does not match the run's original contract"
                )
            raise drift_error(contract, detail)
        if digest is not None:
            used.add(digest)
    if configured_contract != contracts[-1]:
        raise drift_error(contracts[-1], "the configured contract does not match it")
    return ContractLineage(
        contracts=tuple(contracts),
        amendments=chain,
        pending_amendments=tuple(record for record in chain if record.digest not in used),
    )


def require_amendment_activation(
    lineage: ContractLineage,
    *,
    start_round_number: int,
    template: Callable[[ReviewerBoardAmendment, int], str],
) -> None:
    """An amendment no digest-bound record has used must start at round N.

    ``N`` is the round the resume re-enters, partial or reconciled.  An
    earlier round would reinterpret scheduler decisions already final; a later
    round would force the reduced board into a round the record does not
    cover.  Both fail closed before any agent invocation or comment post.
    """
    for amendment in lineage.pending_amendments:
        if amendment.effective_from_round != start_round_number:
            raise AgentLoopError(
                "Human decision required: signed reviewer-board amendment at "
                f"{amendment.comment_locator} is effective from round "
                f"{amendment.effective_from_round}, but this resume re-enters round "
                f"{start_round_number}, the first round whose scheduler decision can use "
                "the amended board. Replace the record with one effective from round "
                f"{start_round_number}:\n\n{template(amendment, start_round_number)}"
            )


def _effective_ownership(
    item: UnresolvedReviewItem,
) -> tuple[tuple[str, ...], dict[str, str]]:
    # Exactly the fallback ``_scheduler_obligations`` and
    # ``_apply_unresolved_item_dispositions`` use for legacy items.
    owners = tuple(item.resolution_owners or (item.reviewer,))
    states = dict(item.owner_states or ((owner, "pending") for owner in owners))
    return owners, states


def apply_board_amendment_to_ledger(
    items: Iterable[UnresolvedReviewItem],
    *,
    removed_reviewers: Sequence[str],
    remaining_reviewers: Sequence[str],
    primary_reviewer: str | None,
) -> tuple[tuple[UnresolvedReviewItem, ...], tuple[LedgerReassignment, ...]]:
    """Derived ledger view with removed reviewers' ownership reassigned.

    Pure and idempotent.  An active item whose effective pending owners
    include a removed reviewer is normalized to explicit ownership, so the
    removed author cannot re-enter through the implicit owner fallback.  If
    another owner is still pending, the removed reviewer is simply dropped;
    if it was the only pending owner, the item is reassigned to the primary
    (or to every remaining reviewer without a primary).  Authors, notes, and
    text are unchanged, and nothing is auto-cleared.
    """
    removed = set(removed_reviewers)
    result: list[UnresolvedReviewItem] = []
    reassignments: list[LedgerReassignment] = []
    for item in items:
        if item.status not in _ACTIVE_LEDGER_STATUSES or not removed:
            result.append(item)
            continue
        owners, states = _effective_ownership(item)
        removed_owners = tuple(owner for owner in owners if owner in removed)
        if not removed_owners:
            result.append(item)
            continue
        kept = tuple(owner for owner in owners if owner not in removed)
        still_pending = [owner for owner in kept if states.get(owner) != "cleared"]
        removed_pending = [owner for owner in removed_owners if states.get(owner) != "cleared"]
        new_owners = kept
        new_states = {owner: states.get(owner, "pending") for owner in kept}
        if removed_pending and not still_pending:
            successors = (
                (primary_reviewer,) if primary_reviewer is not None else tuple(remaining_reviewers)
            )
            new_owners = tuple(dict.fromkeys((*kept, *successors)))
            for owner in successors:
                new_states[owner] = "pending"
        if not new_owners:
            # Every owner was removed and all had already cleared; keep the
            # item attributable to the remaining board rather than orphaned.
            new_owners = (
                (primary_reviewer,) if primary_reviewer is not None else tuple(remaining_reviewers)
            )
            new_states = {owner: "pending" for owner in new_owners}
        result.append(
            dataclasses.replace(
                item,
                resolution_owners=new_owners,
                owner_states=tuple((owner, new_states[owner]) for owner in new_owners),
                owner_evidence=tuple(
                    (owner, note) for owner, note in item.owner_evidence if owner not in removed
                ),
                owner_dispositions=tuple(
                    (owner, value)
                    for owner, value in item.owner_dispositions
                    if owner not in removed
                ),
            )
        )
        reassignments.append(
            LedgerReassignment(
                item_id=item.item_id,
                removed=removed_owners,
                new_owners=tuple(owner for owner in new_owners if owner not in kept),
            )
        )
    return tuple(result), tuple(reassignments)


def describe_reassignments(reassignments: Sequence[LedgerReassignment]) -> str:
    parts = []
    for entry in reassignments:
        if entry.new_owners:
            parts.append(
                f"{entry.item_id} ({', '.join(entry.removed)} -> {', '.join(entry.new_owners)})"
            )
        else:
            parts.append(f"{entry.item_id} (dropped {', '.join(entry.removed)})")
    return ", ".join(parts) or "none"


def amendment_summary_line(
    lineage: ContractLineage, reassignments: Sequence[LedgerReassignment] = ()
) -> str | None:
    """One-line round-summary note, or ``None`` when the board is unamended."""
    active = lineage.active_amendment
    if active is None:
        return None
    board = getattr(lineage.contracts[-1], "required_reviewers")
    return (
        f"Reviewer board amended from round {active.effective_from_round} by signed record "
        f"{active.digest[:16]}: removed {', '.join(lineage.removed_reviewers)} "
        f"({active.reason}); required board now {', '.join(board)}; reassigned findings "
        f"{describe_reassignments(reassignments)}."
    )


def amendment_audit_already_posted(comments: Sequence[object], digest: str) -> bool:
    return any(
        isinstance(getattr(comment, "body", None), str)
        and REVIEWER_BOARD_AMENDMENT_AUDIT_HEADING in comment.body  # type: ignore[attr-defined]
        and digest in comment.body  # type: ignore[attr-defined]
        for comment in comments
    )


def render_amendment_audit_comment(
    lineage: ContractLineage,
    *,
    start_round_number: int,
    reassignments: Sequence[LedgerReassignment],
) -> str:
    active = lineage.active_amendment
    assert active is not None
    board = getattr(lineage.contracts[-1], "required_reviewers")
    original = getattr(lineage.contracts[0], "required_reviewers")
    return "\n".join(
        [
            REVIEWER_BOARD_AMENDMENT_AUDIT_HEADING,
            "",
            f"- Signed record: {active.comment_locator} (digest `{active.digest}`)",
            f"- Activation round: {start_round_number}",
            f"- Removed reviewer(s): {', '.join(lineage.removed_reviewers)} ({active.reason})",
            f"- Original board: {', '.join(original)}",
            f"- Required board now: {', '.join(board)}",
            "- Approvals already given by removed reviewers stay in the history but are no "
            "longer required.",
            f"- Reassigned findings: {describe_reassignments(reassignments)}",
            "",
            "-- Orchestrator",
        ]
    )


def missing_from_config(persisted: object, configured: object | None) -> tuple[str, ...] | None:
    """Reviewers dropped by the configured contract, when a template fits.

    ``None`` when the configured contract is not a same-policy, same-primary
    strict subset of the persisted board (no amendment can explain it).
    """
    if configured is None:
        return None
    if (
        getattr(configured, "policy") != getattr(persisted, "policy")
        or getattr(configured, "primary_reviewer") != getattr(persisted, "primary_reviewer")
    ):
        return None
    persisted_board = tuple(getattr(persisted, "required_reviewers"))
    configured_board = tuple(getattr(configured, "required_reviewers"))
    removed = tuple(name for name in persisted_board if name not in configured_board)
    if not removed or any(name not in persisted_board for name in configured_board):
        return None
    if tuple(name for name in persisted_board if name in configured_board) != configured_board:
        return None
    return removed
