"""Signed reviewer-board amendment records (#943).

A persisted scheduler contract is immutable for the run: a resume that
silently narrowed the required reviewer board would weaken the guarantee that
every configured reviewer approved the exact artifact.  This module adds the
one audited exception.  A human operator posts a signed
``reviewer-board-amendment`` record that removes unavailable non-primary
reviewers from the persisted board, starting at an explicit round.

The same record can also restore a recovered reviewer (#984) through the
optional ``restored_reviewers`` field.  A restoration only strengthens the
guarantee, and is bounded by the validated chain: it may only re-add a
reviewer an earlier link removed, so the board never exceeds C0.  A restored
reviewer is required again from the effective round forward and must
re-approve the current artifact (its pre-restoration approvals never count);
findings reassigned while it was removed stay reassigned, so a restoration
never re-opens settled items.

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
REVIEWER_BOARD_REMOVAL_REASON = "backend-unavailable"
REVIEWER_SEAT_REMOVAL_REASON = "seat-unavailable"
REVIEWER_BOARD_RESTORATION_REASON = "backend-recovered"
REVIEWER_BOARD_AMENDMENT_REASONS = frozenset(
    {REVIEWER_BOARD_REMOVAL_REASON, REVIEWER_SEAT_REMOVAL_REASON, REVIEWER_BOARD_RESTORATION_REASON}
)
REVIEWER_BOARD_AMENDMENT_AUDIT_HEADING = "Reviewer board amendment applied."
# ``restored_reviewers`` (#984) is optional so every removal-only record
# posted before it keeps its exact key set and canonical digest.
_OPTIONAL_AMENDMENT_RECORD_KEYS = frozenset({"restored_reviewers"})
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
    """One signed human authorization to remove or restore board reviewers."""

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
    restored_reviewers: tuple[str, ...] = ()

    def amended_board(self, base_board: Sequence[str]) -> tuple[str, ...]:
        """original - removed + restored, in the order of the C0 ``base_board``.

        The record alone cannot know where a restored reviewer sat in C0, so
        there is deliberately no board property on the record: callers pass
        the chain's base board (#984).
        """
        kept = set(self.original_required_reviewers) - set(self.removed_reviewers)
        kept |= set(self.restored_reviewers)
        return tuple(name for name in base_board if name in kept)


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
    def current_board(self) -> tuple[str, ...]:
        if not self.contracts:
            return ()
        return tuple(getattr(self.contracts[-1], "required_reviewers"))

    @property
    def removed_reviewers(self) -> tuple[str, ...]:
        """Reviewers the chain removed that are still off the board, in removal order.

        A reviewer a later link restored is required again, so it is not
        removed: the ledger view stops reassigning its ownership (#984).
        """
        board = set(self.current_board)
        return tuple(
            dict.fromkeys(
                name
                for amendment in self.amendments
                for name in amendment.removed_reviewers
                if name not in board
            )
        )

    @property
    def restoration_rounds(self) -> dict[str, int]:
        """Currently required restored reviewers -> round their restoration took effect.

        A restored reviewer's review from an earlier round (before it was
        removed, or while it was off the board) never counts as a current
        approval, so it must re-approve the artifact before completion (#984).
        """
        board = set(self.current_board)
        rounds: dict[str, int] = {}
        for amendment in self.amendments:
            for name in amendment.restored_reviewers:
                if name in board:
                    rounds[name] = amendment.effective_from_round
        return rounds

    @property
    def restored_reviewers(self) -> tuple[str, ...]:
        return tuple(self.restoration_rounds)


@dataclass(frozen=True)
class LedgerReassignment:
    item_id: str
    removed: tuple[str, ...]
    new_owners: tuple[str, ...]


def reviewer_board_amendment_digest(record: dict[str, object]) -> str:
    """Canonical digest shared by discovery, round metadata, and qualification."""
    keys = _AMENDMENT_RECORD_KEYS | (_OPTIONAL_AMENDMENT_RECORD_KEYS & set(record))
    canonical = json.dumps(
        {key: record[key] for key in sorted(keys)},
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
    reason: str | None = None,
    rationale: str,
    restored_reviewers: Sequence[str] = (),
) -> dict[str, object]:
    if reason is None:
        reason = (
            REVIEWER_BOARD_RESTORATION_REASON
            if restored_reviewers and not removed_reviewers
            else REVIEWER_BOARD_REMOVAL_REASON
        )
    payload: dict[str, object] = {
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
    if restored_reviewers:
        # Omitted when empty so a removal-only record keeps its #943 shape.
        payload["restored_reviewers"] = list(restored_reviewers)
    return payload


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
    reason: str | None = None,
    rationale: str | None = None,
    restored_reviewers: Sequence[str] = (),
) -> str:
    """Render the signed amendment record format documented for operators."""
    if rationale is None:
        rationale = (
            "<why the restored reviewer is reachable again>"
            if restored_reviewers and not removed_reviewers
            else "<why the removed reviewer cannot be reached>"
        )
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
        restored_reviewers=restored_reviewers,
    )
    return (
        "Reviewer board amendment:\n\n```json\n"
        + json.dumps(record, indent=2, sort_keys=True)
        + "\n```\n-- Human Reviewer"
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _unique_name_list(value: object, *, allow_empty: bool = False) -> bool:
    return (
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and item.strip() for item in value)
        and len(set(value)) == len(value)
    )


def _amendment_record_problem(payload: dict[str, object]) -> str | None:
    keys = set(payload)
    allowed = _AMENDMENT_RECORD_KEYS | _OPTIONAL_AMENDMENT_RECORD_KEYS
    if not _AMENDMENT_RECORD_KEYS <= keys <= allowed:
        missing = sorted(_AMENDMENT_RECORD_KEYS - keys)
        unknown = sorted(keys - allowed)
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
    removed = payload.get("removed_reviewers")
    if not _unique_name_list(removed, allow_empty=True):
        return "removed_reviewers must be a list of unique names"
    restored = payload.get("restored_reviewers", [])
    if "restored_reviewers" in payload and not _unique_name_list(restored):
        return "restored_reviewers, when present, must be a non-empty list of unique names"
    assert isinstance(removed, list) and isinstance(restored, list)
    if not removed and not restored:
        return "removed_reviewers must be non-empty unless restored_reviewers is present"
    overlap = sorted(set(removed) & set(restored))
    if overlap:
        return f"reviewer(s) {overlap} are both removed and restored"
    policy = payload.get("policy")
    if not isinstance(policy, str) or not policy.strip():
        return "policy must be a non-empty string"
    primary = payload.get("primary_reviewer")
    if primary is not None and (not isinstance(primary, str) or not primary.strip()):
        return "primary_reviewer must be a non-empty string or null"
    if not _is_positive_int(payload.get("effective_from_round")):
        return "effective_from_round must be a positive integer"
    reason = payload.get("reason")
    if reason not in REVIEWER_BOARD_AMENDMENT_REASONS:
        return f"reason must be one of {sorted(REVIEWER_BOARD_AMENDMENT_REASONS)}"
    if reason == REVIEWER_SEAT_REMOVAL_REASON and flow != "pr":
        return "seat-unavailable is supported only for pr amendments"
    if reason == REVIEWER_SEAT_REMOVAL_REASON and len(removed) != 1:
        return "seat-unavailable must remove exactly one reviewer seat"
    if not restored and reason not in {REVIEWER_BOARD_REMOVAL_REASON, REVIEWER_SEAT_REMOVAL_REASON}:
        return "a removal-only record must give a removal reason"
    if not removed and reason != REVIEWER_BOARD_RESTORATION_REASON:
        return (
            f"a restoration-only record must give reason `{REVIEWER_BOARD_RESTORATION_REASON}`"
        )
    rationale = payload.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        return "rationale must be a non-empty string"
    return None


def _normalize_newlines(body: str | None) -> str | None:
    """CRLF -> LF; a lone CR stays unsupported (the signature parser ignores it)."""
    return body.replace("\r\n", "\n") if isinstance(body, str) else body


_AMENDMENT_HEADING_RE = re.compile(r"(?im)^[ \t]*reviewer board amendment:?[ \t]*$")
_ANY_FENCE_RE = re.compile(
    r"^[ \t]{0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)\n[ \t]{0,3}(?P=fence)[ \t]*$",
    re.S | re.M,
)


def _amendment_shaped_block(signed_body: str) -> str | None:
    """The block text when a comment is exactly one unreadable amendment fence.

    Matches only when the strict ``json`` fence path found no fence at all and
    the body, minus an optional heading line, is a single fenced block naming
    the record kind.  Records are never applied from such a block; the shape is
    only used to report the comment and keep it out of signed requirements.
    Prose alongside the block, or a strict fence anywhere, never matches.
    """
    if _FENCED_JSON_RE.search(signed_body):
        return None
    remainder = _AMENDMENT_HEADING_RE.sub("", signed_body, count=1).strip()
    match = _ANY_FENCE_RE.fullmatch(remainder)
    if match is None or REVIEWER_BOARD_AMENDMENT_KIND not in match.group("body"):
        return None
    return match.group("body")


def parse_reviewer_board_amendment_records(
    body: str | None, *, comment_locator: str, comment_index: int = -1
) -> tuple[tuple[ReviewerBoardAmendment, ...], tuple[str, ...]]:
    """Return (signed valid records, ignored-record diagnostics) for one comment.

    Only a body with the standalone human reviewer signature counts, and a
    malformed record is reported and ignored so it can never be applied.
    """
    # GitHub hands CRLF bodies back verbatim; the signature parser and the fence
    # regex only understand LF, so normalize before either runs (#1133).
    signed = parse_signed_human_requirement_body(_normalize_newlines(body))
    if signed is None:
        return (), ()
    records: list[ReviewerBoardAmendment] = []
    ignored: list[str] = []
    if _amendment_shaped_block(signed) is not None:
        ignored.append(
            f"{comment_locator}: reviewer-board amendment record found but its fence could "
            "not be read; use a ```json fence on its own lines, and delete and repost the "
            "record if it was posted under a different fence"
        )
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
                restored_reviewers=tuple(payload.get("restored_reviewers", ())),  # type: ignore[arg-type]
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
    signed_body = _normalize_newlines(signed_body) or ""
    if _amendment_shaped_block(signed_body) is not None:
        return True
    found = False

    def strip(match: re.Match[str]) -> str:
        nonlocal found
        try:
            payload = json.loads(match.group("body"))
        except json.JSONDecodeError:
            # A malformed amendment fence (the same shape discovery reports
            # as an ignored record) is still an orchestration record, never
            # a requirement on the reviewed artifact.
            if REVIEWER_BOARD_AMENDMENT_KIND in match.group("body"):
                found = True
                return ""
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


def amend_contract(
    original: ContractT,
    amendment: ReviewerBoardAmendment,
    *,
    base_board: Sequence[str] | None = None,
    previously_removed: Iterable[str] = (),
) -> ContractT:
    """Apply one amendment to a plan or PR scheduler contract, fail-closed.

    Policy, primary, and (for PR contracts) broad-path rules stay unchanged;
    only ``removed_reviewers`` leave ``required_reviewers`` and only
    ``restored_reviewers`` rejoin it.  A restoration must name a reviewer an
    earlier link of the chain removed (``previously_removed``) that is not
    already required, so the board never exceeds the C0 ``base_board``, whose
    order it keeps.  The contracts' own ``__post_init__`` rules reject a
    primary-then-panel board left with no secondary.
    """
    required = tuple(getattr(original, "required_reviewers"))
    c0_board = tuple(base_board) if base_board is not None else required
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
    already = [name for name in amendment.restored_reviewers if name in required]
    if already:
        raise fail(f"restored reviewer(s) {already} are already on the persisted board")
    removed_before = set(previously_removed)
    never_removed = [
        name
        for name in amendment.restored_reviewers
        if name not in removed_before or name not in c0_board
    ]
    if never_removed:
        raise fail(
            f"restored reviewer(s) {never_removed} were never removed by an earlier "
            "amendment in this run's chain; a restoration can only re-add a reviewer "
            "the chain removed"
        )
    remaining = amendment.amended_board(c0_board)
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

    Restorations (#984) let a board recur (remove -> restore -> remove), so
    two records may legitimately name the same original board at different
    chain positions.  Links must be posted in chain order, so only the
    earliest matching record can be the next link; any other record that
    matched the same board and is not consumed by a later link is a genuine
    conflict and fails closed without choosing between them.
    """
    base_board = tuple(getattr(base_contract, "required_reviewers"))
    contracts: list[ContractT] = [base_contract]
    chain: list[ReviewerBoardAmendment] = []
    remaining = list(amendments)
    contested: list[tuple[tuple[str, ...], list[ReviewerBoardAmendment]]] = []
    while remaining:
        board = tuple(getattr(contracts[-1], "required_reviewers"))
        matches = sorted(
            (
                record for record in remaining
                if tuple(record.original_required_reviewers) == board
            ),
            key=lambda record: record.comment_index,
        )
        if not matches:
            break
        if len(matches) > 1:
            contested.append((board, matches))
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
        contracts.append(
            amend_contract(
                contracts[-1],
                record,
                base_board=base_board,
                previously_removed=(
                    name for link in chain for name in link.removed_reviewers
                ),
            )
        )
        chain.append(record)
        remaining.remove(record)
    for board, matches in contested:
        if any(record in remaining for record in matches):
            raise AgentLoopError(
                "Human decision required: signed reviewer-board amendments at "
                + ", ".join(record.comment_locator for record in matches)
                + f" all amend the board {list(board)}. The orchestrator never chooses "
                "between them by comment order; keep exactly one and remove the others."
            )
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


# Tag carried by the drift detail when scheduler records were posted after an
# amendment comment without reading it; drift builders key their repost
# instruction on it (#1133).
STALE_UNREAD_AMENDMENT_HINT = "stale-unread-amendment-record"
_STALE_UNREAD_AMENDMENT_RE = re.compile(
    rf"\[{STALE_UNREAD_AMENDMENT_HINT} at (?P<locator>[^\]]+)\]"
)


def stale_unread_amendment_locator(detail: str) -> str | None:
    """The amendment comment locator tagged in a stale-record drift detail."""
    match = _STALE_UNREAD_AMENDMENT_RE.search(detail)
    return match.group("locator") if match else None


def resolve_contract_lineage(
    records: Sequence[PostedRoundRecord],
    amendments: Sequence[ReviewerBoardAmendment],
    configured_contract: ContractT | None,
    *,
    accept_base_configured: bool = False,
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
    Cn (``accept_base_configured`` also accepts C0, for PR runs that derive
    the amended board themselves).  ``drift_error(persisted_contract, detail)``
    builds the fail-closed error for any mismatch.
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
            elif (
                governing
                and contract == contracts[governing - 1]
                and digest == (chain[governing - 2].digest if governing > 1 else None)
            ):
                detail = (
                    f"round {record.metadata.round_number} scheduler record (comment "
                    f"{record.index + 1}) was posted under the amendment at "
                    f"{chain[governing - 1].comment_locator} but does not carry its amended "
                    "contract and digest: it was posted after that amendment comment without "
                    "the run reading it, and such records are never reinterpreted "
                    f"[{STALE_UNREAD_AMENDMENT_HINT} at {chain[governing - 1].comment_locator}]"
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
    if configured_contract != contracts[-1] and not (
        accept_base_configured and chain and configured_contract == contracts[0]
    ):
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
    board = lineage.current_board
    changes = []
    if lineage.removed_reviewers:
        changes.append(f"removed {', '.join(lineage.removed_reviewers)}")
    if lineage.restored_reviewers:
        changes.append(
            "restored "
            + ", ".join(
                f"{name} (from round {round_number}, must re-approve)"
                for name, round_number in lineage.restoration_rounds.items()
            )
        )
    return (
        f"Reviewer board amended from round {active.effective_from_round} by signed record "
        f"{active.digest[:16]}: {'; '.join(changes) or 'original board restored'} "
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
    board = lineage.current_board
    original = getattr(lineage.contracts[0], "required_reviewers")
    lines = [
        REVIEWER_BOARD_AMENDMENT_AUDIT_HEADING,
        "",
        f"- Signed record: {active.comment_locator} (digest `{active.digest}`)",
        f"- Activation round: {start_round_number}",
        f"- Removed reviewer(s): {', '.join(lineage.removed_reviewers) or 'none'} "
        f"({active.reason})",
    ]
    if lineage.restored_reviewers:
        lines.append(
            f"- Restored reviewer(s): {', '.join(lineage.restored_reviewers)}; required "
            "again from the restoration round and must re-approve the current artifact "
            "(earlier approvals do not count). Findings reassigned while removed stay "
            "reassigned and are not re-opened."
        )
    lines += [
        f"- Original board: {', '.join(original)}",
        f"- Required board now: {', '.join(board)}",
    ]
    if lineage.removed_reviewers:
        lines.append(
            "- Approvals already given by removed reviewers stay in the history but are no "
            "longer required."
        )
    lines += [
        f"- Reassigned findings: {describe_reassignments(reassignments)}",
        "",
        "-- Orchestrator",
    ]
    return "\n".join(lines)


def restoration_rounds_from_comments(
    comments: Sequence[object], *, flow: str
) -> dict[str, int]:
    """Restored reviewer -> latest restoration round, from signed records alone.

    For gates that check approvals without a resolved lineage (#984); the
    lineage itself is validated by the resume and qualification paths.  A
    stale entry for a reviewer later removed again is harmless: it is not on
    the required board, so its approvals are never consulted.
    """
    rounds: dict[str, int] = {}
    for index, comment in enumerate(comments):
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        records, _ignored = parse_reviewer_board_amendment_records(
            body, comment_locator=f"comment {index + 1}", comment_index=index
        )
        for record in records:
            if record.flow != flow:
                continue
            for name in record.restored_reviewers:
                rounds[name] = max(rounds.get(name, 0), record.effective_from_round)
    return rounds


def predates_restoration(
    restoration_rounds: dict[str, int], reviewer: str, round_number: int
) -> bool:
    """Whether a review by ``reviewer`` in ``round_number`` precedes its restoration."""
    floor = restoration_rounds.get(reviewer)
    return floor is not None and round_number < floor


def added_in_config(persisted: object, configured: object | None) -> tuple[str, ...] | None:
    """Reviewers the configured contract adds back, when a restore template fits.

    ``None`` unless the configured contract is a same-policy, same-primary
    strict superset of the persisted board.  Whether each name was removed
    earlier in the chain is checked when the signed record is applied.
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
    if any(name not in configured_board for name in persisted_board):
        return None
    added = tuple(name for name in configured_board if name not in persisted_board)
    return added or None


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
