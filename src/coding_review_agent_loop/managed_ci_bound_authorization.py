"""Transaction-bound managed-CI authorization record (#827, stage B).

The bound record is the managed-CI authorization entry of a workflow
transaction: the v1 authorization fields plus the transaction ID, the grant
anchor, and an optional upgrade source.  It is judged in exactly one place,
``validate_bound_authorization``, which the seam (pending), the gate
(committed), and the router all reach through ``BoundAuthorizationCodec``.

The v1 record and its parser in ``managed_ci`` are untouched.  This module
never calls the live-dependent resume audit: every check reads durable,
author-authenticated records only.

Not yet wired: no production call site builds a bound payload, so no bound
record is emitted.
"""

from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, replace

from .errors import AgentLoopError, WorkflowTransactionError
from .github import AuthenticatedComment, AuthenticatedCommentView
from .managed_ci import (
    ManagedCiIssueAuthorization,
    _continuity_round_metadata_is_valid,
    parse_issue_created_authorization_comment,
)
from .protocol_markers import (
    MARKER_BY_TOKEN,
    PR_COMMENT_SURFACE,
    TrustedBody,
    protocol_record_label,
    scan_reserved_markers,
)
from .workflow_transaction import (
    DISPOSITION_NOT_APPLICABLE,
    DISPOSITION_REISSUED,
    ENTRY_AUTHORIZATION,
    KIND_INITIAL,
    KIND_PLAN_REPLACEMENT,
    RECOVERY_OPERATOR_REVIEW,
    TransactionLineage,
    TransactionState,
    is_strictly_later,
)

BOUND_AUTHORIZATION_MARKER = "AGENT_MANAGED_CI_BOUND_AUTHORIZATION_V2"
BOUND_AUTHORIZATION_VERSION = 2

KIND_CREATION = "creation"
KIND_FRESH = "fresh"
KIND_CONTINUITY = "continuity"
KIND_PLAN_REBIND = "plan-rebind"
KIND_ORDINARY_RELEASE = "ordinary-release"
GRANTED_KINDS = (KIND_CREATION, KIND_FRESH, KIND_CONTINUITY, KIND_PLAN_REBIND)
# Kinds whose nonce is minted by the builder, so adoption takes the stored one.
_MINTED_NONCE_KINDS = frozenset({KIND_CREATION, KIND_FRESH, KIND_CONTINUITY})
_V1_KINDS = frozenset({KIND_CREATION, KIND_FRESH, KIND_CONTINUITY})

CODE_AUTHORIZATION_INVALID = "authorization-invalid"


def _digest(values: tuple[object, ...]) -> str:
    raw = json.dumps(list(values), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def granted_generation(
    *,
    repository: str,
    issue_number: int,
    pr_number: int,
    base_ref: str,
    actor_id: int,
    protection: str,
    waiver: str,
    grant_anchor_event_id: int,
    approved_plan_hash: str | None,
) -> str:
    """Generation of a granted authorization.

    It excludes the head, the authorization kind, minted nonces, and the live
    active label event, so an ordinary push is a head-advance and a plain rerun
    after the label was re-applied computes the committed generation again.
    """
    return _digest((
        "granted", repository.casefold(), issue_number, pr_number, base_ref, actor_id,
        protection, waiver, grant_anchor_event_id, approved_plan_hash,
    ))


def released_generation(
    *, repository: str, issue_number: int, pr_number: int, base_ref: str, actor_id: int
) -> str:
    """Generation of a PR released to ordinary recovery; it excludes the plan hash."""
    return _digest((
        repository.casefold(), issue_number, pr_number, base_ref, actor_id,
        KIND_ORDINARY_RELEASE,
    ))


@dataclass(frozen=True)
class BoundManagedCiAuthorization:
    """The managed-CI authorization entry of one workflow transaction."""

    kind: str
    transaction_id: str
    repository: str
    issue_number: int
    pr_number: int
    base_ref: str
    head_sha: str
    actor_login: str
    actor_id: int
    protection: str | None = None
    waiver: str | None = None
    nonce: str | None = None
    label_event_id: int | None = None
    grant_anchor_event_id: int | None = None
    predecessor_head: str | None = None
    predecessor_comment_id: int | None = None
    round_comment_ids: tuple[int, ...] = ()
    approved_plan_hash: str | None = None
    upgraded_from_comment_id: int | None = None

    @property
    def granted(self) -> bool:
        return self.kind in GRANTED_KINDS

    def generation(self) -> str:
        """The generation recomputed from the record's own canonical fields."""
        if not self.granted:
            return released_generation(
                repository=self.repository, issue_number=self.issue_number,
                pr_number=self.pr_number, base_ref=self.base_ref, actor_id=self.actor_id,
            )
        assert self.protection is not None and self.waiver is not None
        assert self.grant_anchor_event_id is not None
        return granted_generation(
            repository=self.repository, issue_number=self.issue_number,
            pr_number=self.pr_number, base_ref=self.base_ref, actor_id=self.actor_id,
            protection=self.protection, waiver=self.waiver,
            grant_anchor_event_id=self.grant_anchor_event_id,
            approved_plan_hash=self.approved_plan_hash,
        )

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": BOUND_AUTHORIZATION_VERSION,
            "kind": self.kind,
            "transaction_id": self.transaction_id,
            "repository": self.repository,
            "issue": self.issue_number,
            "pr": self.pr_number,
            "base": self.base_ref,
            "head": self.head_sha,
            "actor": self.actor_login,
            "actor_id": self.actor_id,
        }
        optional: dict[str, object | None] = {
            "protection": self.protection,
            "waiver": self.waiver,
            "nonce": self.nonce,
            "label_event_id": self.label_event_id,
            "grant_anchor_event_id": self.grant_anchor_event_id,
            "predecessor_head": self.predecessor_head,
            "predecessor_comment_id": self.predecessor_comment_id,
            "approved_plan_hash": self.approved_plan_hash,
            "upgraded_from_comment_id": self.upgraded_from_comment_id,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        if self.round_comment_ids:
            payload["round_comment_ids"] = list(self.round_comment_ids)
        return payload

    def as_v1(self) -> ManagedCiIssueAuthorization:
        """The v1 view of a granted v1-kind record, for today's unchanged checks."""
        if self.kind not in _V1_KINDS:
            raise AgentLoopError(f"A {self.kind} bound authorization has no v1 form.")
        assert self.protection and self.waiver and self.nonce and self.label_event_id
        return ManagedCiIssueAuthorization(
            kind=self.kind,  # type: ignore[arg-type]
            repository=self.repository,
            issue_number=self.issue_number,
            pr_number=self.pr_number,
            base_ref=self.base_ref,
            head_sha=self.head_sha,
            actor_login=self.actor_login,
            actor_id=self.actor_id,
            protection=self.protection,
            waiver=self.waiver,
            nonce=self.nonce,
            label_event_id=self.label_event_id,
            predecessor_head=self.predecessor_head,
            predecessor_comment_id=self.predecessor_comment_id,
            round_comment_ids=self.round_comment_ids,
            approved_plan_hash=self.approved_plan_hash,
        )


def bind_v1_authorization(
    authorization: ManagedCiIssueAuthorization,
    *,
    transaction_id: str = "",
    grant_anchor_event_id: int,
    upgraded_from_comment_id: int | None = None,
) -> BoundManagedCiAuthorization:
    """The bound payload that carries every v1 field of ``authorization`` unchanged."""
    return BoundManagedCiAuthorization(
        kind=authorization.kind,
        transaction_id=transaction_id,
        repository=authorization.repository,
        issue_number=authorization.issue_number,
        pr_number=authorization.pr_number,
        base_ref=authorization.base_ref,
        head_sha=authorization.head_sha,
        actor_login=authorization.actor_login,
        actor_id=authorization.actor_id,
        protection=authorization.protection,
        waiver=authorization.waiver,
        nonce=authorization.nonce,
        label_event_id=authorization.label_event_id,
        grant_anchor_event_id=grant_anchor_event_id,
        predecessor_head=authorization.predecessor_head,
        predecessor_comment_id=authorization.predecessor_comment_id,
        round_comment_ids=authorization.round_comment_ids,
        approved_plan_hash=authorization.approved_plan_hash,
        upgraded_from_comment_id=upgraded_from_comment_id,
    )


def ordinary_release_payload(
    *,
    repository: str,
    issue_number: int,
    pr_number: int,
    base_ref: str,
    head_sha: str,
    actor_login: str,
    actor_id: int,
    transaction_id: str = "",
) -> BoundManagedCiAuthorization:
    """The record of a PR released to ordinary recovery.  It grants nothing."""
    return BoundManagedCiAuthorization(
        kind=KIND_ORDINARY_RELEASE, transaction_id=transaction_id, repository=repository,
        issue_number=issue_number, pr_number=pr_number, base_ref=base_ref, head_sha=head_sha,
        actor_login=actor_login, actor_id=actor_id,
    )


def build_plan_rebind_payload(
    committed_record: BoundManagedCiAuthorization,
    *,
    committed_comment_id: int,
    new_plan_hash: str,
    live_head: str,
) -> BoundManagedCiAuthorization | None:
    """The deterministic ``plan-rebind`` reissue of a committed granted record.

    Mints nothing.  Returns ``None`` when the record grants nothing, the live
    head moved past it, or the plan did not change; the caller stops
    non-mutating and lets the PR loop commit the head successor first.
    """
    if (
        not committed_record.granted
        or committed_record.head_sha != live_head
        or not new_plan_hash
        or committed_record.approved_plan_hash == new_plan_hash
    ):
        return None
    return replace(
        committed_record,
        kind=KIND_PLAN_REBIND,
        transaction_id="",
        approved_plan_hash=new_plan_hash,
        predecessor_comment_id=committed_comment_id,
        predecessor_head=committed_record.head_sha,
        round_comment_ids=(),
        upgraded_from_comment_id=None,
    )


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def _malformed(reason: str) -> AgentLoopError:
    return AgentLoopError(f"Managed-CI bound authorization record {reason}.")


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is not None and (not isinstance(value, str) or not value):
        raise _malformed(f"has an invalid {key}")
    return value


def _optional_id(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise _malformed(f"has an invalid {key}")
    return value


_PAYLOAD_FIELDS = frozenset({
    "version", "kind", "transaction_id", "repository", "issue", "pr", "base", "head",
    "actor", "actor_id", "protection", "waiver", "nonce", "label_event_id",
    "grant_anchor_event_id", "predecessor_head", "predecessor_comment_id",
    "round_comment_ids", "approved_plan_hash", "upgraded_from_comment_id",
})
_TRANSACTION_ID_CHARS = frozenset("0123456789abcdef")


def decode_bound_authorization(payload: object) -> BoundManagedCiAuthorization:
    """Decode one payload, checking only what the record says about itself."""
    if not isinstance(payload, dict) or payload.get("version") != BOUND_AUTHORIZATION_VERSION:
        raise _malformed("has an invalid version")
    if set(payload) - _PAYLOAD_FIELDS:
        raise _malformed("has unknown fields")
    kind = payload.get("kind")
    if kind not in (*GRANTED_KINDS, KIND_ORDINARY_RELEASE):
        raise _malformed("has an invalid kind")
    tx_id = payload.get("transaction_id")
    if not isinstance(tx_id, str) or len(tx_id) != 64 or set(tx_id) - _TRANSACTION_ID_CHARS:
        raise _malformed("has an invalid transaction ID")
    for key in ("repository", "base", "head", "actor"):
        if not isinstance(payload.get(key), str) or not payload[key]:
            raise _malformed("is missing required fields")
    for key in ("issue", "pr", "actor_id"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise _malformed("has invalid identity fields")
    rounds = payload.get("round_comment_ids", [])
    if not isinstance(rounds, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in rounds
    ):
        raise _malformed("has invalid round metadata IDs")
    record = BoundManagedCiAuthorization(
        kind=kind,
        transaction_id=tx_id,
        repository=payload["repository"],
        issue_number=payload["issue"],
        pr_number=payload["pr"],
        base_ref=payload["base"],
        head_sha=payload["head"],
        actor_login=payload["actor"],
        actor_id=payload["actor_id"],
        protection=_optional_text(payload, "protection"),
        waiver=_optional_text(payload, "waiver"),
        nonce=_optional_text(payload, "nonce"),
        label_event_id=_optional_id(payload, "label_event_id"),
        grant_anchor_event_id=_optional_id(payload, "grant_anchor_event_id"),
        predecessor_head=_optional_text(payload, "predecessor_head"),
        predecessor_comment_id=_optional_id(payload, "predecessor_comment_id"),
        round_comment_ids=tuple(rounds),
        approved_plan_hash=_optional_text(payload, "approved_plan_hash"),
        upgraded_from_comment_id=_optional_id(payload, "upgraded_from_comment_id"),
    )
    if record.granted:
        if (
            record.protection is None or record.waiver is None or record.nonce is None
            or record.label_event_id is None or record.grant_anchor_event_id is None
        ):
            raise _malformed("is missing granted fields")
        # The same protection and waiver context the v1 parser accepts.
        if (
            record.protection not in {"voluntary", "plan_limited"}
            or record.waiver != "allow-unprotected-managed-ci"
        ):
            raise _malformed("has an invalid protection or waiver context")
    elif any(
        value is not None
        for value in (
            record.protection, record.waiver, record.nonce, record.label_event_id,
            record.grant_anchor_event_id, record.predecessor_head,
            record.predecessor_comment_id, record.approved_plan_hash,
            record.upgraded_from_comment_id,
        )
    ) or record.round_comment_ids:
        raise _malformed("is a release that carries grant fields")
    return record


def format_bound_authorization_comment(record: BoundManagedCiAuthorization) -> TrustedBody:
    """Render the complete bound record for a PR comment only.  Byte-deterministic."""
    raw = json.dumps(record.to_payload(), separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).decode("ascii")
    label = protocol_record_label(
        "managed_ci_authorization",
        kind=record.kind,
        issue_number=record.issue_number,
        pr_number=record.pr_number,
        head_sha=record.head_sha,
    )
    return TrustedBody.canonical(
        f"{label}\n\n<!-- {BOUND_AUTHORIZATION_MARKER}: {encoded} -->",
        surface=PR_COMMENT_SURFACE,
        expected_tokens=(BOUND_AUTHORIZATION_MARKER,),
    )


def parse_bound_authorization_comment(body: str) -> BoundManagedCiAuthorization | None:
    """Parse one strict bound record without accepting body copies."""
    if BOUND_AUTHORIZATION_MARKER not in body:
        return None
    matches = [
        occurrence for occurrence in scan_reserved_markers(body)
        if occurrence.definition.token == BOUND_AUTHORIZATION_MARKER
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise _malformed("is duplicated in one comment")
    TrustedBody.canonical(
        body, surface=PR_COMMENT_SURFACE, expected_tokens=(BOUND_AUTHORIZATION_MARKER,)
    )
    match = MARKER_BY_TOKEN[BOUND_AUTHORIZATION_MARKER].pattern.fullmatch(matches[0].text)
    if match is None:
        raise _malformed("is malformed")
    try:
        payload = json.loads(
            base64.urlsafe_b64decode(match.group("payload").encode("ascii")).decode("utf-8")
        )
    except (ValueError, UnicodeError) as exc:
        raise _malformed("is not valid JSON") from exc
    return decode_bound_authorization(payload)


# ---------------------------------------------------------------------------
# The one rule
# ---------------------------------------------------------------------------


def _unedited(comment: AuthenticatedComment) -> bool:
    return comment.updated_at is None or comment.updated_at == comment.created_at


def _bound_records(
    pr_view: AuthenticatedCommentView,
) -> list[tuple[AuthenticatedComment, BoundManagedCiAuthorization]]:
    found = []
    for comment in pr_view.authored:
        record = parse_bound_authorization_comment(comment.body)
        if record is not None:
            found.append((comment, record))
    return found


def effective_authorization(
    transaction: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> tuple[AuthenticatedComment, BoundManagedCiAuthorization] | None:
    """The canonical bound record of a committed transaction, through ``inherited``."""
    cursor: TransactionState | None = transaction
    while cursor is not None:
        entry = cursor.intent.entry(ENTRY_AUTHORIZATION)
        if entry.disposition == DISPOSITION_NOT_APPLICABLE:
            return None
        if entry.disposition == DISPOSITION_REISSUED:
            outcome = cursor.outcome(ENTRY_AUTHORIZATION)
            if outcome is None or outcome.comment_id is None:
                return None
            comment = pr_view.comment(outcome.comment_id)
            record = parse_bound_authorization_comment(comment.body) if comment else None
            return (comment, record) if comment is not None and record is not None else None
        parent = cursor.intent.predecessor_transaction_id
        cursor = lineage.state(parent) if parent is not None else None
    return None


def _predecessor_state(
    transaction: TransactionState, lineage: TransactionLineage
) -> TransactionState | None:
    parent = transaction.intent.predecessor_transaction_id
    return lineage.state(parent) if parent is not None else None


def nearest_granted_ancestor(
    transaction: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> tuple[AuthenticatedComment, BoundManagedCiAuthorization] | None:
    """First granted ``effective_authorization`` walking back from the predecessor."""
    cursor = _predecessor_state(transaction, lineage)
    while cursor is not None:
        effective = effective_authorization(cursor, lineage, pr_view)
        if effective is not None and effective[1].granted:
            return effective
        cursor = _predecessor_state(cursor, lineage)
    return None


def _unbound_records(
    pr_view: AuthenticatedCommentView, before: AuthenticatedComment
) -> list[tuple[AuthenticatedComment, ManagedCiIssueAuthorization]]:
    found = []
    for comment in pr_view.authored:
        if not is_strictly_later(comment, before):
            continue
        record = parse_issue_created_authorization_comment(comment.body)
        if record is not None:
            found.append((comment, record))
    return found


def select_upgrade_source(
    records: list[tuple[AuthenticatedComment, ManagedCiIssueAuthorization]],
    head: str,
    copied: BoundManagedCiAuthorization,
) -> AuthenticatedComment | None:
    """Durable upgrade-selection rule: a pure function of durable inputs only.

    The greatest-comment-ID unbound actor record at ``head`` whose tuple,
    protection, waiver, and plan hash equal the copied fields and which no
    other unbound record names as its predecessor.
    """
    named = {record.predecessor_comment_id for _comment, record in records}
    matching = [
        comment
        for comment, record in records
        if record.head_sha == head
        and comment.comment_id not in named
        and (
            record.repository.casefold(), record.issue_number, record.pr_number,
            record.base_ref, record.actor_id, record.protection, record.waiver,
            record.approved_plan_hash,
        ) == (
            copied.repository.casefold(), copied.issue_number, copied.pr_number,
            copied.base_ref, copied.actor_id, copied.protection, copied.waiver,
            copied.approved_plan_hash,
        )
    ]
    return max(matching, key=lambda item: item.comment_id, default=None)


def _round_dicts(pr_view: AuthenticatedCommentView) -> list[dict[str, object]]:
    return [
        {
            "id": comment.comment_id,
            "body": comment.body,
            "user": {"login": comment.author_login, "id": comment.author_id},
        }
        for comment in pr_view.authored
    ]


def _legacy_chain_problem(
    source: AuthenticatedComment,
    records: list[tuple[AuthenticatedComment, ManagedCiIssueAuthorization]],
    pr_view: AuthenticatedCommentView,
) -> str | None:
    """Today's legacy chain rule, walked to its root over unbound actor records."""
    by_id = {comment.comment_id: (comment, record) for comment, record in records}
    cursor: int | None = source.comment_id
    seen: set[int] = set()
    while cursor is not None:
        if cursor in seen or cursor not in by_id:
            return f"missing unbound authorization link {cursor}"
        seen.add(cursor)
        comment, record = by_id[cursor]
        if not _unedited(comment):
            return f"edited unbound authorization link {cursor}"
        if record.kind == KIND_CONTINUITY:
            parent = by_id.get(record.predecessor_comment_id or 0)
            if parent is None or parent[1].head_sha != record.predecessor_head:
                return f"unbound continuity link {cursor} names no record at its predecessor head"
            if not _continuity_round_metadata_is_valid(
                _round_dicts(pr_view), authorization=record
            ):
                return f"unbound continuity link {cursor} fails round-metadata correlation"
        if record.predecessor_comment_id is not None and (
            record.predecessor_comment_id >= cursor
        ):
            return f"unbound authorization link {cursor} names a later predecessor"
        cursor = record.predecessor_comment_id
    return None


def validate_bound_authorization(
    envelope: AuthenticatedComment,
    record: BoundManagedCiAuthorization,
    transaction: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
    *,
    committed: bool,
) -> None:
    """The only place a bound record is judged.

    Raises an integrity-class ``WorkflowTransactionError`` (operator review, no
    write) naming the transaction, the comment, and the failing field.  A
    record that fails is never skipped or treated as absent.
    """
    intent = transaction.intent

    def fail(field: str) -> WorkflowTransactionError:
        return WorkflowTransactionError(
            "A bound managed-CI authorization does not match the transaction it is bound to",
            transaction_ids=(transaction.transaction_id,),
            successor_kind=intent.successor_kind,
            expected_record_set=intent.expected_record_set,
            problems=(
                f"contradictory bound authorization in comment {envelope.comment_id}: {field}",
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code=CODE_AUTHORIZATION_INVALID,
        )

    # (1) envelope
    if envelope.author_id != pr_view.actor_id:
        raise fail("it was not written by the authenticated actor")
    if not _unedited(envelope):
        raise fail("the comment was edited")
    if not is_strictly_later(transaction.prepared_comment, envelope):
        raise fail("it is not later than the prepared record")
    same = [
        comment for comment, other in _bound_records(pr_view)
        if other.transaction_id == transaction.transaction_id
    ]
    if any(comment.body != envelope.body for comment in same):
        raise fail("another bound record of the same transaction differs")
    if committed:
        outcome = transaction.outcome(ENTRY_AUTHORIZATION)
        if outcome is None or outcome.comment_id != envelope.comment_id:
            raise fail("comment ID differs from the committed outcome")
        if same and min(item.comment_id for item in same) != envelope.comment_id:
            raise fail("it is not the earliest bound record of the transaction")
    # (2) scope
    if intent.managed_ci_generation is None:
        raise fail("the transaction is unmanaged and admits no bound record")
    if intent.entry(ENTRY_AUTHORIZATION).disposition != DISPOSITION_REISSUED:
        raise fail("the transaction does not reissue the authorization")
    scope = {
        "transaction_id": (record.transaction_id, transaction.transaction_id),
        "repository": (record.repository.casefold(), intent.repository.casefold()),
        "issue": (record.issue_number, intent.primary_issue),
        "pr": (record.pr_number, intent.pr_number),
        "base": (record.base_ref, intent.base),
        "head": (record.head_sha, intent.head_sha),
        "actor_id": (record.actor_id, pr_view.actor_id),
    }
    for field, (actual, expected) in scope.items():
        if actual != expected:
            raise fail(field)
    # (3) plan
    if record.granted and record.approved_plan_hash != intent.approved_plan_hash:
        raise fail("approved_plan_hash")
    # (4) generation digest, recomputed from the record's own fields
    if record.generation() != intent.managed_ci_generation:
        raise fail("generation")
    # (6) nonce
    if record.granted and not record.nonce:
        raise fail("nonce")
    # (5) kind shape: exactly one branch runs
    if record.upgraded_from_comment_id is not None:
        problem = _upgrade_problem(record, transaction, lineage, pr_view)
    else:
        problem = _native_problem(record, transaction, lineage, pr_view)
    if problem is not None:
        raise fail(problem)


def _upgrade_problem(
    record: BoundManagedCiAuthorization,
    transaction: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> str | None:
    intent = transaction.intent
    if (
        intent.successor_kind != KIND_INITIAL
        or intent.predecessor_transaction_id is not None
        or intent.legacy_root is not None
        or any(
            item.committed and item.transaction_id != transaction.transaction_id
            and is_strictly_later(item.prepared_comment, transaction.prepared_comment)
            for item in lineage.transactions
        )
    ):
        return "upgraded_from_comment_id on a transaction that is not a plain initial root"
    if record.kind not in _V1_KINDS:
        return "an upgraded record must keep a v1 kind"
    records = _unbound_records(pr_view, transaction.prepared_comment)
    source = select_upgrade_source(records, intent.head_sha, record)
    if source is None or source.comment_id != record.upgraded_from_comment_id:
        return "upgraded_from_comment_id is not the record the durable upgrade rule selects"
    unbound = next(item for comment, item in records if comment.comment_id == source.comment_id)
    if not _unedited(source):
        return "the upgrade source was edited"
    if record.as_v1() != unbound:
        return "a v1 field differs from the upgrade source"
    if record.grant_anchor_event_id != unbound.label_event_id:
        return "grant_anchor_event_id"
    return _legacy_chain_problem(source, records, pr_view)


def _native_problem(
    record: BoundManagedCiAuthorization,
    transaction: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> str | None:
    intent = transaction.intent
    has_predecessor_fields = (
        record.predecessor_head is not None or record.predecessor_comment_id is not None
    )
    if record.kind == KIND_ORDINARY_RELEASE:
        return None  # the decoder already refuses every grant field
    if record.kind == KIND_CREATION:
        if intent.successor_kind != KIND_INITIAL or intent.predecessor_transaction_id:
            return "a creation record is valid only on an initial root"
        if has_predecessor_fields or record.round_comment_ids:
            return "a creation record carries continuity fields"
        if record.grant_anchor_event_id != record.label_event_id:
            return "grant_anchor_event_id"
        return None
    ancestor = nearest_granted_ancestor(transaction, lineage, pr_view)
    if record.kind == KIND_FRESH:
        if record.round_comment_ids:
            return "a fresh record carries round metadata"
        if (record.predecessor_head is None) != (record.predecessor_comment_id is None):
            return "a fresh record names half a predecessor"
        if ancestor is None:
            if has_predecessor_fields:
                return "a fresh record names a predecessor but no granted ancestor exists"
        elif not has_predecessor_fields or (
            record.predecessor_comment_id, record.predecessor_head
        ) != (ancestor[0].comment_id, ancestor[1].head_sha):
            return "predecessor fields do not name the nearest granted ancestor"
        if record.grant_anchor_event_id not in {
            record.label_event_id,
            ancestor[1].grant_anchor_event_id if ancestor is not None else None,
        }:
            return "grant_anchor_event_id"
        return None
    parent = _predecessor_state(transaction, lineage)
    effective = effective_authorization(parent, lineage, pr_view) if parent is not None else None
    if parent is None or effective is None or not effective[1].granted:
        return f"a {record.kind} record needs a granted predecessor authorization"
    parent_comment, parent_record = effective
    if (record.predecessor_comment_id, record.predecessor_head) != (
        parent_comment.comment_id, parent_record.head_sha
    ):
        return "predecessor fields do not name the predecessor's effective authorization"
    carried = ("grant_anchor_event_id", "protection", "waiver")
    if record.kind == KIND_CONTINUITY:
        if record.predecessor_head != parent.intent.head_sha or (
            record.predecessor_head == record.head_sha
        ):
            return "predecessor_head"
        for field in (*carried, "approved_plan_hash"):
            if getattr(record, field) != getattr(parent_record, field):
                return field
        if not record.round_comment_ids or not _continuity_round_metadata_is_valid(
            _round_dicts(pr_view), authorization=record.as_v1()
        ):
            return "round_comment_ids"
        return None
    # plan-rebind
    if intent.successor_kind != KIND_PLAN_REPLACEMENT:
        return "a plan-rebind record is valid only on a plan-replacement successor"
    if record.predecessor_head != record.head_sha or record.round_comment_ids:
        return "a plan-rebind record must stay on the predecessor head with no round metadata"
    for field in (*carried, "label_event_id", "nonce", "actor_login"):
        if getattr(record, field) != getattr(parent_record, field):
            return field
    if record.approved_plan_hash == parent_record.approved_plan_hash:
        return "approved_plan_hash does not change"
    return None


# ---------------------------------------------------------------------------
# Transaction-era accessor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundAuthorizationRecord:
    """One validated bound record under its canonical comment ID."""

    comment_id: int
    record: BoundManagedCiAuthorization


@dataclass(frozen=True)
class BuilderAuthorizationView:
    """What a transaction-era builder may select, compare, and extend.

    ``superseded`` is for diagnostics only: builders never select, compare, or
    count it, so an earlier record at the same head never conflicts.
    """

    effective: BoundAuthorizationRecord | None = None
    nearest_granted: BoundAuthorizationRecord | None = None
    pending: BoundAuthorizationRecord | None = None
    superseded: tuple[BoundAuthorizationRecord, ...] = ()


def _validated(
    pair: tuple[AuthenticatedComment, BoundManagedCiAuthorization] | None,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> BoundAuthorizationRecord | None:
    """Judge a committed record against the transaction that issued it."""
    if pair is None:
        return None
    comment, record = pair
    issuing = lineage.state(record.transaction_id)
    if issuing is None or not issuing.committed:
        raise WorkflowTransactionError(
            "A bound managed-CI authorization names a transaction that did not commit",
            transaction_ids=(record.transaction_id,),
            problems=(f"contradictory bound authorization in comment {comment.comment_id}",),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code=CODE_AUTHORIZATION_INVALID,
        )
    validate_bound_authorization(comment, record, issuing, lineage, pr_view, committed=True)
    return BoundAuthorizationRecord(comment.comment_id, record)


def pending_bound_authorization(
    pr_view: AuthenticatedCommentView,
    lineage: TransactionLineage,
    transaction: TransactionState,
) -> BoundAuthorizationRecord | None:
    """Pending mode: the uncommitted record of one prepared-only transaction.

    It is a seam and selection input only, never authority.
    """
    if transaction.committed or transaction.aborted:
        return None
    bound = [
        (comment, record)
        for comment, record in _bound_records(pr_view)
        if record.transaction_id == transaction.transaction_id
    ]
    if not bound:
        return None
    comment, record = min(bound, key=lambda item: item[0].comment_id)
    validate_bound_authorization(comment, record, transaction, lineage, pr_view, committed=False)
    return BoundAuthorizationRecord(comment.comment_id, record)


def builder_authorization_view(
    pr_view: AuthenticatedCommentView, lineage: TransactionLineage
) -> BuilderAuthorizationView:
    """Builder mode: every committed canonical bound record, validated.

    Unbound records are never returned on a transaction-era PR, and there is no
    selection of a record by head.
    """
    committed = lineage.latest_committed
    effective = (
        _validated(effective_authorization(committed, lineage, pr_view), lineage, pr_view)
        if committed is not None
        else None
    )
    nearest = effective if effective is not None and effective.record.granted else None
    if nearest is None and committed is not None:
        nearest = _validated(
            nearest_granted_ancestor(committed, lineage, pr_view), lineage, pr_view
        )
    superseded: list[BoundAuthorizationRecord] = []
    for state in lineage.chain:
        if not state.committed:
            continue
        if state.intent.entry(ENTRY_AUTHORIZATION).disposition != DISPOSITION_REISSUED:
            continue
        item = _validated(effective_authorization(state, lineage, pr_view), lineage, pr_view)
        if item is not None and (effective is None or item.comment_id != effective.comment_id):
            superseded.append(item)
    pending_state = lineage.pending
    return BuilderAuthorizationView(
        effective=effective,
        nearest_granted=nearest,
        pending=(
            pending_bound_authorization(pr_view, lineage, pending_state)
            if pending_state is not None
            else None
        ),
        superseded=tuple(superseded),
    )


def consumer_bound_authorization(
    pr_view: AuthenticatedCommentView, lineage: TransactionLineage, *, live_head: str
) -> BoundAuthorizationRecord | None:
    """Consumer mode: managed authority for the live head, or nothing.

    Only the latest committed transaction's record, only at the live head, never
    while a prepared transaction is pending, and never an ``ordinary-release``
    record.  An invalid record raises; it is never skipped as if absent.
    """
    committed = lineage.latest_committed
    if committed is None or lineage.pending is not None:
        return None
    effective = _validated(effective_authorization(committed, lineage, pr_view), lineage, pr_view)
    if effective is None or committed.intent.head_sha != live_head:
        return None
    return effective if effective.record.granted else None


# ---------------------------------------------------------------------------
# Seam codec
# ---------------------------------------------------------------------------


class BoundAuthorizationCodec:
    """``AuthorizationEntryCodec`` over the real bound record.

    The seam, the gate, and the router reach the bound record only through this
    object, so neither this module nor ``managed_ci`` imports the publication
    module.
    """

    def transaction_ids(self, comment: AuthenticatedComment) -> tuple[str, ...]:
        record = parse_bound_authorization_comment(comment.body)
        return (record.transaction_id,) if record is not None else ()

    def expected_body(self, payload: object, transaction_id: str) -> TrustedBody:
        return format_bound_authorization_comment(self._bound(payload, transaction_id))

    def adoptable(
        self, comment: AuthenticatedComment, payload: object, transaction_id: str
    ) -> bool:
        record = parse_bound_authorization_comment(comment.body)
        if record is None:
            return False
        expected = self._bound(payload, transaction_id)
        minted = (
            expected.kind in _MINTED_NONCE_KINDS and expected.upgraded_from_comment_id is None
        )
        if minted and record.nonce:
            expected = replace(expected, nonce=record.nonce)
        return (
            record == expected
            and comment.body == str(format_bound_authorization_comment(expected))
        )

    def validate(
        self,
        comment: AuthenticatedComment,
        *,
        state: TransactionState,
        lineage: TransactionLineage,
        pr_view: AuthenticatedCommentView,
        committed: bool,
    ) -> None:
        try:
            record = parse_bound_authorization_comment(comment.body)
        except AgentLoopError as exc:
            raise WorkflowTransactionError(
                "A bound managed-CI authorization cannot be decoded",
                transaction_ids=(state.transaction_id,),
                problems=(f"contradictory bound authorization in comment {comment.comment_id}: {exc}",),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code=CODE_AUTHORIZATION_INVALID,
            ) from exc
        if record is None:
            raise WorkflowTransactionError(
                "The comment named as the bound managed-CI authorization carries no such record",
                transaction_ids=(state.transaction_id,),
                problems=(f"missing bound authorization in comment {comment.comment_id}",),
                recovery_action=RECOVERY_OPERATOR_REVIEW,
                code=CODE_AUTHORIZATION_INVALID,
            )
        validate_bound_authorization(
            comment, record, state, lineage, pr_view, committed=committed
        )

    def digest(self, comment: AuthenticatedComment) -> str:
        record = parse_bound_authorization_comment(comment.body)
        if record is None:
            raise AgentLoopError(
                f"Comment {comment.comment_id} carries no bound managed-CI authorization."
            )
        raw = json.dumps(record.to_payload(), separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _bound(payload: object, transaction_id: str) -> BoundManagedCiAuthorization:
        if not isinstance(payload, BoundManagedCiAuthorization):
            raise AgentLoopError("The managed transition input is not a bound authorization payload.")
        return replace(payload, transaction_id=transaction_id)
