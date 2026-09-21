"""Publication and reconciliation seam for workflow transactions (#827, stage B).

Stage A (:mod:`workflow_transaction`) is a pure model.  This module is the one
place that writes transaction records and the records a transaction binds:

* :func:`publish_transition` publishes one logical workflow transition as an
  append-only ``prepared`` record, its record-set entries, and one terminal
  record.  Every write is read-first (an ambiguous earlier write is adopted,
  never repeated) and verified by read-back.
* :func:`require_committed_transaction` is the fail-closed gate every authority
  consumer calls.  It never writes.
* :func:`read_pr_transaction_views` and :func:`discover_canonical_issue_pr` are
  the strict, read-only readers.  :func:`route_issue_publication` is the
  non-authority router that hands an interrupted publication back to the seam.

Sibling reconciliation runs over grouped transactions *below* the strict
lineage resolver, and only inside the seam: a reader or the gate never
reconciles.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from .config import AgentLoopConfig
from .errors import AgentLoopError, WorkflowTransactionError
from .github import (
    ISSUE_THREAD_SURFACE,
    PR_THREAD_SURFACE,
    AuthenticatedComment,
    AuthenticatedCommentView,
    comment_thread_surface,
    post_verified_trusted_issue_protocol_comment,
    post_verified_trusted_pr_protocol_comment,
    read_authenticated_protocol_comments,
)
from .issue_pr_handoff import (
    AGENT_ISSUE_PR_HANDOFF_RE,
    IssuePrHandoffMetadata,
    IssuePrHandoffMetadataV2,
    decode_issue_pr_handoff_v2,
    format_issue_pr_handoff_v2_comment,
    issue_pr_handoff_payload_schema_version,
)
from .plan_review_scheduling import PlanCandidateKey
from .pr_contract import (
    PR_CONTRACT_SUPERSESSION_COMBINED,
    PR_EXPECTED_CLOSING_MARKER_RE,
    PrExpectedClosingContract,
    PrExpectedClosingContractV2,
    decode_pr_contract_v2,
    format_pr_contract_v2_comment,
    pr_contract_payload_schema_version,
)
from .protocol_markers import TrustedBody, scan_reserved_markers
from .round_state import _extract_round_metadata_records
from .runner import Runner
from .workflow_transaction import (
    ABORT_SIBLING_CANONICAL,
    DISPOSITION_INHERITED,
    DISPOSITION_NOT_APPLICABLE,
    DISPOSITION_REISSUED,
    ENTRY_AUTHORIZATION,
    ENTRY_HANDOFF,
    ENTRY_INITIAL_CODER_ROUND,
    ENTRY_PR_CONTRACT,
    ERA_LEGACY,
    ERA_TRANSACTION,
    FLOW_APPROVED_PLAN,
    FLOW_DIRECT_PR,
    FLOW_ISSUE,
    FLOW_MANAGED_PR,
    KIND_INITIAL,
    PHASE_ABORTED,
    PHASE_COMMITTED,
    RECORD_SET_ENTRY_NAMES,
    RECOVERY_OPERATOR_REVIEW,
    RECOVERY_RERUN,
    STATUS_INHERITED,
    STATUS_NOT_APPLICABLE,
    STATUS_UNPUBLISHED,
    STATUS_WAIVED_CODER_RESPONSE,
    CommentRef,
    EntryOutcome,
    LegacyRootContext,
    ResolvedHandoff,
    ResolvedPrContract,
    SchedulerCheckpointRef,
    StagedIdentity,
    TransactionLineage,
    TransactionState,
    TransitionInputs,
    WorkflowTransactionRecord,
    WorkflowTransition,
    actor_change_error,
    classify_transaction_era,
    collect_transactions,
    compare_prepared_intent,
    derive_handoff_metadata,
    derive_pr_contract,
    format_transaction_record_comment,
    handoff_candidate_pr_numbers,
    is_strictly_later,
    not_applicable,
    plan_successor,
    prepared_record,
    reissued,
    resolve_approved_plan_anchor,
    resolve_handoff_lineage,
    resolve_pr_contract_lineage,
    resolve_transaction_lineage,
    round_metadata_digest,
    select_scheduler_checkpoint,
    verify_scheduler_checkpoint,
)

ORIGIN_DIRECT_ISSUE = "direct-issue"
ORIGIN_APPROVED_PLAN = "approved-plan"
ORIGIN_STAGED_CHILD = "staged-child"
ORIGIN_PR_RESUME = "pr-resume"
ORIGIN_CHILD_PLAN_REBIND = "child-plan-rebind"
ORIGIN_PATHS = frozenset(
    {
        ORIGIN_DIRECT_ISSUE,
        ORIGIN_APPROVED_PLAN,
        ORIGIN_STAGED_CHILD,
        ORIGIN_PR_RESUME,
        ORIGIN_CHILD_PLAN_REBIND,
    }
)

# Failures a rerun (or the next round) can finish.  Everything else is an
# integrity failure that stops the round.
CODE_WRITE_FAILED = "publication-write-failed"
CODE_PENDING = "transaction-pending"
CODE_UNCOMMITTED = "transaction-uncommitted"
CODE_HEAD_NOT_COMMITTED = "head-not-committed"
CODE_SIBLING_LOST = "sibling-lost"
CODE_MANAGED_UNAVAILABLE = "managed-input-unavailable"
CODE_PARTIAL_CANDIDATE = "partial-candidate"
RECOVERABLE_CODES = frozenset(
    {
        CODE_WRITE_FAILED,
        CODE_PENDING,
        CODE_UNCOMMITTED,
        CODE_HEAD_NOT_COMMITTED,
        CODE_SIBLING_LOST,
        CODE_MANAGED_UNAVAILABLE,
    }
)

_MAX_PUBLICATION_ATTEMPTS = 4


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------


class AuthorizationEntryCodec(Protocol):
    """How the seam reads, writes, and judges the managed-CI authorization entry.

    The bound authorization codec and its validation rule live with managed CI;
    the seam only needs these operations, so it never imports that module's
    activation logic.
    """

    def transaction_ids(self, comment: AuthenticatedComment) -> tuple[str, ...]:
        """Transaction IDs of every bound authorization the comment carries."""

    def expected_body(self, payload: object, transaction_id: str) -> TrustedBody:
        """The exact comment body that publishes ``payload`` under the transaction."""

    def adoptable(
        self, comment: AuthenticatedComment, payload: object, transaction_id: str
    ) -> bool:
        """Whether the comment equals the payload on every field but a minted nonce."""

    def validate(
        self,
        comment: AuthenticatedComment,
        *,
        state: TransactionState,
        lineage: TransactionLineage,
        pr_view: AuthenticatedCommentView,
        committed: bool,
    ) -> None:
        """Raise an integrity ``WorkflowTransactionError`` unless the record is valid."""

    def digest(self, comment: AuthenticatedComment) -> str:
        """SHA-256 payload digest used by an ``inherited`` reference."""


@dataclass(frozen=True)
class Unmanaged:
    pass


@dataclass(frozen=True)
class Granted:
    expected_payload: object
    generation: str


@dataclass(frozen=True)
class Released:
    expected_payload: object
    generation: str


@dataclass(frozen=True)
class Unavailable:
    reason: str


ManagedTransitionInput = Unmanaged | Granted | Released | Unavailable


@dataclass(frozen=True)
class ApprovedPlanInput:
    plan_hash: str
    plan_subject: str | None
    plan_candidate_key: PlanCandidateKey | None


@dataclass(frozen=True)
class InitialCoderRound:
    """The initial coder comment, rendered once the transaction ID is known."""

    render: Callable[[str], TrustedBody]


@dataclass(frozen=True)
class TransitionRequest:
    """Everything one publish site knows.  There is no flow-string parameter."""

    repository: str
    pr_number: int
    base: str
    head_sha: str
    origin_path: str
    expected_closing_issue_ids: tuple[int, ...]
    primary_issue: int | None = None
    approved_plan: ApprovedPlanInput | None = None
    staged: StagedIdentity | None = None
    managed: ManagedTransitionInput = Unmanaged()
    authorization_codec: AuthorizationEntryCodec | None = None
    initial_coder_round: InitialCoderRound | None = None
    # pr-resume of a PR that no issue owns: managed source PRs are `managed-pr`.
    unowned_managed_pr: bool = False
    # Rebind only: one extra trusted section carried by the reissued handoff.
    handoff_extra_section: str | None = None
    legacy_root_context: LegacyRootContext | None = None

    def __post_init__(self) -> None:
        if self.origin_path not in ORIGIN_PATHS:
            raise AgentLoopError(f"Unknown workflow transition origin path {self.origin_path!r}.")
        if self.origin_path != ORIGIN_PR_RESUME and self.primary_issue is None:
            raise AgentLoopError(f"{self.origin_path} publication requires a primary issue.")
        plan_paths = {ORIGIN_APPROVED_PLAN, ORIGIN_STAGED_CHILD, ORIGIN_CHILD_PLAN_REBIND}
        if self.origin_path in plan_paths and self.approved_plan is None:
            raise AgentLoopError(f"{self.origin_path} publication requires the approved plan.")
        if self.origin_path == ORIGIN_DIRECT_ISSUE and self.approved_plan is not None:
            raise AgentLoopError("direct-issue publication carries no approved plan.")
        if self.origin_path == ORIGIN_STAGED_CHILD and self.staged is None:
            raise AgentLoopError("staged-child publication requires the staged identity.")
        if self.origin_path == ORIGIN_CHILD_PLAN_REBIND and (
            self.staged is None or self.staged.plan_owner != "child"
        ):
            raise AgentLoopError(
                "A child-plan rebind applies only to a planning child whose plan owner is "
                "the child."
            )
        if self.handoff_extra_section is not None and (
            self.origin_path != ORIGIN_CHILD_PLAN_REBIND
        ):
            raise AgentLoopError("Only a child-plan rebind carries an extra handoff section.")
        if isinstance(self.managed, (Granted, Released)) and self.authorization_codec is None:
            raise AgentLoopError("A managed transition requires the authorization codec.")

    @property
    def origin_flow(self) -> str:
        """The origin flow, derived from the origin path and never chosen by a caller."""
        if self.approved_plan is not None:
            return FLOW_APPROVED_PLAN
        if self.primary_issue is not None:
            return FLOW_ISSUE
        return FLOW_MANAGED_PR if self.unowned_managed_pr else FLOW_DIRECT_PR

    @property
    def managed_ci_generation(self) -> str | None:
        if isinstance(self.managed, (Granted, Released)):
            return self.managed.generation
        return None

    @property
    def plan_owning_issue(self) -> int | None:
        if self.staged is not None and self.staged.plan_owner == "parent":
            return self.staged.parent_issue
        return self.primary_issue

    def inputs(self) -> TransitionInputs:
        return TransitionInputs(
            repository=self.repository,
            primary_issue=self.primary_issue,
            pr_number=self.pr_number,
            base=self.base,
            head_sha=self.head_sha,
            origin_flow=self.origin_flow,
            approved_plan_hash=self.approved_plan.plan_hash if self.approved_plan else None,
            expected_closing_issue_ids=tuple(self.expected_closing_issue_ids),
            staged=self.staged,
            managed_ci_generation=self.managed_ci_generation,
        )


@dataclass(frozen=True)
class PublicationViews:
    """One authenticated read of every surface a transaction touches."""

    pr_view: AuthenticatedCommentView
    issue_view: AuthenticatedCommentView | None = None
    plan_issue_view: AuthenticatedCommentView | None = None


@dataclass(frozen=True)
class CommittedTransaction:
    """The committed canonical transaction for a head: the only source of authority."""

    state: TransactionState
    lineage: TransactionLineage
    contract: ResolvedPrContract | None
    handoff: ResolvedHandoff | None

    @property
    def intent(self) -> WorkflowTransition:
        return self.state.intent

    @property
    def transaction_id(self) -> str:
        return self.state.transaction_id

    @property
    def chain(self) -> tuple[TransactionState, ...]:
        return self.lineage.chain

    def entry_comment_id(self, name: str) -> int | None:
        """The terminal outcome comment ID, or the inherited reference's comment ID."""
        entry = self.intent.entry(name)
        if entry.inherited is not None:
            return entry.inherited.comment_id
        outcome = self.state.outcome(name)
        return outcome.comment_id if outcome is not None else None


@dataclass(frozen=True)
class LegacyEra:
    """The PR has no authenticated version-2 record; today's v1 checks decide."""


@dataclass(frozen=True)
class NoLiveHeadAuthority:
    """A recoverable failure: the round continues but nothing may be reused."""

    diagnostic: WorkflowTransactionError


RoundAuthority = CommittedTransaction | LegacyEra | NoLiveHeadAuthority


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _error(
    summary: str,
    *,
    intent: WorkflowTransition | None = None,
    transaction_ids: Sequence[str] = (),
    problems: Sequence[str] = (),
    recovery: str = RECOVERY_OPERATOR_REVIEW,
    code: str | None = None,
) -> WorkflowTransactionError:
    ids = tuple(transaction_ids) or ((intent.transaction_id,) if intent is not None else ())
    return WorkflowTransactionError(
        summary,
        transaction_ids=ids,
        successor_kind=intent.successor_kind if intent is not None else None,
        expected_record_set=intent.expected_record_set if intent is not None else (),
        problems=tuple(problems),
        recovery_action=recovery,
        code=code,
    )


def is_recoverable(error: WorkflowTransactionError) -> bool:
    return error.code in RECOVERABLE_CODES


def _unedited(comment: AuthenticatedComment) -> bool:
    return comment.updated_at is None or comment.updated_at == comment.created_at


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def _read_views(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    issue_number: int | None,
    plan_issue_number: int | None = None,
) -> PublicationViews:
    pr_view = read_authenticated_protocol_comments(
        runner, config=config, surface_kind=PR_THREAD_SURFACE, number=pr_number
    )
    issue_view = (
        read_authenticated_protocol_comments(
            runner, config=config, surface_kind=ISSUE_THREAD_SURFACE, number=issue_number
        )
        if issue_number is not None
        else None
    )
    plan_view = issue_view
    if plan_issue_number is not None and plan_issue_number != issue_number:
        plan_view = read_authenticated_protocol_comments(
            runner, config=config, surface_kind=ISSUE_THREAD_SURFACE, number=plan_issue_number
        )
    return PublicationViews(pr_view, issue_view, plan_view)


@dataclass(frozen=True)
class PrTransactionViews:
    views: PublicationViews
    era: str
    lineage: TransactionLineage
    contract: ResolvedPrContract | None
    handoff: ResolvedHandoff | None


def _resolve_views(
    views: PublicationViews,
    *,
    repository: str,
    pr_number: int,
    issue_number: int | None,
    legacy_root_context: LegacyRootContext | None = None,
) -> PrTransactionViews:
    foreign = actor_change_error(views.pr_view)
    if foreign is not None:
        raise foreign
    lineage = resolve_transaction_lineage(
        views.pr_view,
        repository=repository,
        pr_number=pr_number,
        legacy_root_context=legacy_root_context,
    )
    contract = resolve_pr_contract_lineage(
        views.pr_view, lineage, repository=repository, pr_number=pr_number
    )
    handoff = (
        resolve_handoff_lineage(
            views.issue_view, lineage, repository=repository, issue_number=issue_number
        )
        if issue_number is not None and views.issue_view is not None
        else None
    )
    era = classify_transaction_era(views.pr_view, views.issue_view)
    if lineage.transactions:
        era = ERA_TRANSACTION
    return PrTransactionViews(views, era, lineage, contract, handoff)


def read_pr_transaction_views(
    runner: Runner,
    config: AgentLoopConfig,
    pr_number: int,
    issue_number: int | None,
    *,
    plan_issue_number: int | None = None,
) -> PrTransactionViews:
    """Strict read-only view of a PR's transaction state.  Never reconciles."""
    views = _read_views(
        runner,
        config,
        pr_number=pr_number,
        issue_number=issue_number,
        plan_issue_number=plan_issue_number,
    )
    return _resolve_views(
        views, repository=config.repo, pr_number=pr_number, issue_number=issue_number
    )


# ---------------------------------------------------------------------------
# Write primitive (one choke point per surface)
# ---------------------------------------------------------------------------


def _write_verified(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    surface: str,
    number: int,
    body: TrustedBody,
    actor: tuple[str, int],
    intent: WorkflowTransition | None,
    what: str,
) -> None:
    """Write one record through the read-back writers.

    The returned comment ID is deliberately discarded: the seam always re-reads
    the authenticated view, so an ID from a call that reported failure is never
    trusted and a write that landed but reported failure is adopted on rerun.
    """
    try:
        if surface == PR_THREAD_SURFACE:
            post_verified_trusted_pr_protocol_comment(
                runner,
                config=config,
                pr_number=number,
                body=body,
                expected_author_login=actor[0],
                expected_author_id=actor[1],
            )
        else:
            post_verified_trusted_issue_protocol_comment(
                runner,
                config=config,
                issue_number=number,
                body=body,
                expected_author_login=actor[0],
                expected_author_id=actor[1],
            )
    except WorkflowTransactionError:
        raise
    except AgentLoopError as exc:
        raise _error(
            f"Publishing the {what} of a workflow transaction failed or could not be verified",
            intent=intent,
            problems=(f"missing {what}: {exc}",),
            recovery=RECOVERY_RERUN,
            code=CODE_WRITE_FAILED,
        ) from exc


def _trusted(text: str) -> TrustedBody:
    expected = tuple(item.definition.token for item in scan_reserved_markers(text))
    return TrustedBody.canonical(text, expected_tokens=expected)


# ---------------------------------------------------------------------------
# Record discovery by decoded transaction ID (never by text matching)
# ---------------------------------------------------------------------------


def _v2_handoffs(
    issue_view: AuthenticatedCommentView | None,
) -> list[tuple[AuthenticatedComment, IssuePrHandoffMetadataV2]]:
    found: list[tuple[AuthenticatedComment, IssuePrHandoffMetadataV2]] = []
    if issue_view is None:
        return found
    for comment in issue_view.authored:
        for match in AGENT_ISSUE_PR_HANDOFF_RE.finditer(comment.body):
            encoded = match.group("payload")
            try:
                if issue_pr_handoff_payload_schema_version(encoded) != 2:
                    continue
            except AgentLoopError:
                continue
            found.append((comment, decode_issue_pr_handoff_v2(encoded)))
    return found


def _v2_contracts(
    pr_view: AuthenticatedCommentView,
) -> list[tuple[AuthenticatedComment, PrExpectedClosingContractV2]]:
    found: list[tuple[AuthenticatedComment, PrExpectedClosingContractV2]] = []
    for comment in pr_view.authored:
        for match in PR_EXPECTED_CLOSING_MARKER_RE.finditer(comment.body):
            encoded = match.group("payload")
            try:
                if pr_contract_payload_schema_version(encoded) != 2:
                    continue
            except AgentLoopError:
                continue
            found.append((comment, decode_pr_contract_v2(encoded)))
    return found


def _tagged_coder_rounds(pr_view: AuthenticatedCommentView):
    tagged = []
    for record in _extract_round_metadata_records(pr_view.authored, flow="pr"):
        if record.metadata.workflow_transaction_id is not None:
            tagged.append((pr_view.authored[record.index], record.metadata))
    return tagged


def _authorization_comments(
    pr_view: AuthenticatedCommentView, codec: AuthorizationEntryCodec | None
) -> list[tuple[AuthenticatedComment, str]]:
    if codec is None:
        return []
    return [
        (comment, tx_id)
        for comment in pr_view.authored
        for tx_id in codec.transaction_ids(comment)
    ]


def _published_entry_ids(
    state: TransactionState,
    views: PublicationViews,
    codec: AuthorizationEntryCodec | None,
) -> dict[str, int]:
    """Earliest comment ID of every entry already published under one transaction."""
    tx_id = state.transaction_id
    published: dict[str, int] = {}

    def note(name: str, comment: AuthenticatedComment) -> None:
        published[name] = min(published.get(name, comment.comment_id), comment.comment_id)

    for comment, record in _v2_handoffs(views.issue_view):
        if record.transaction_id == tx_id:
            note(ENTRY_HANDOFF, comment)
    for comment, contract in _v2_contracts(views.pr_view):
        if contract.transaction_id == tx_id:
            note(ENTRY_PR_CONTRACT, comment)
    for comment, bound in _authorization_comments(views.pr_view, codec):
        if bound == tx_id:
            note(ENTRY_AUTHORIZATION, comment)
    for comment, metadata in _tagged_coder_rounds(views.pr_view):
        if metadata.workflow_transaction_id == tx_id:
            note(ENTRY_INITIAL_CODER_ROUND, comment)
    return published


# ---------------------------------------------------------------------------
# Initial coder round: one rule for adoption and for the gate
# ---------------------------------------------------------------------------


def _coder_round_candidates(
    state_intent: WorkflowTransition,
    prepared_comment: AuthenticatedComment,
    pr_view: AuthenticatedCommentView,
    *,
    expected_body: str | None = None,
) -> list[AuthenticatedComment]:
    """Tagged initial-coder-round comments, all of them valid, earliest first.

    A comment that carries this transaction's tag with any contradictory field,
    an edited tagged comment, or two tagged comments with different bodies is a
    contradiction and stops non-mutating.
    """
    tx_id = state_intent.transaction_id
    tagged = [
        (comment, metadata)
        for comment, metadata in _tagged_coder_rounds(pr_view)
        if metadata.workflow_transaction_id == tx_id
    ]
    problems: list[str] = []
    for comment, metadata in tagged:
        reasons = []
        if metadata.role != "coder":
            reasons.append(f"role {metadata.role}")
        if metadata.round_number != 1:
            reasons.append(f"round {metadata.round_number}")
        if metadata.subject != state_intent.head_sha:
            reasons.append("head subject differs from the intent head")
        if not is_strictly_later(prepared_comment, comment):
            reasons.append("not strictly later than the prepared record")
        if not _unedited(comment):
            reasons.append("edited after creation")
        if expected_body is not None and comment.body != expected_body:
            reasons.append("body differs from the coder response held by this session")
        if reasons:
            problems.append(
                f"contradictory initial coder round in comment {comment.comment_id}: "
                + ", ".join(reasons)
            )
    if len({comment.body for comment, _metadata in tagged}) > 1:
        problems.append(
            "contradictory initial coder rounds in comments "
            + ", ".join(str(comment.comment_id) for comment, _metadata in tagged)
            + ": their bodies differ"
        )
    if problems:
        raise _error(
            "The transaction-tagged initial coder round contradicts its transaction",
            intent=state_intent,
            problems=problems,
            code="initial-coder-round-contradiction",
        )
    return sorted((comment for comment, _m in tagged), key=lambda item: item.comment_id)


# ---------------------------------------------------------------------------
# Sibling reconciliation (below the strict resolver; seam only)
# ---------------------------------------------------------------------------


def _abort_outcomes(
    state: TransactionState, published: Mapping[str, int]
) -> tuple[EntryOutcome, ...]:
    outcomes = []
    for entry in state.intent.record_set:
        if entry.disposition == DISPOSITION_INHERITED:
            outcomes.append(EntryOutcome(entry.name, status=STATUS_INHERITED))
        elif entry.disposition == DISPOSITION_NOT_APPLICABLE:
            outcomes.append(EntryOutcome(entry.name, status=STATUS_NOT_APPLICABLE))
        elif entry.name in published:
            outcomes.append(EntryOutcome(entry.name, comment_id=published[entry.name]))
        else:
            outcomes.append(EntryOutcome(entry.name, status=STATUS_UNPUBLISHED))
    return tuple(outcomes)


def _sibling_groups(
    states: Sequence[TransactionState],
) -> dict[str | None, list[TransactionState]]:
    groups: dict[str | None, list[TransactionState]] = {}
    for state in states:
        if not state.aborted:
            groups.setdefault(state.intent.predecessor_transaction_id, []).append(state)
    return groups


def _canonical_sibling(members: Sequence[TransactionState]) -> TransactionState:
    lowest = min(members, key=lambda item: item.prepared_comment.comment_id)
    committed = [item for item in members if item.committed]
    if len(committed) > 1 or (committed and committed[0] is not lowest):
        raise WorkflowTransactionError(
            "Sibling transactions cannot be reconciled: more than one is committed, or the "
            "committed one is not the lowest prepared record",
            transaction_ids=tuple(item.transaction_id for item in members),
            successor_kind=members[0].intent.successor_kind,
            expected_record_set=members[0].intent.expected_record_set,
            problems=tuple(
                f"contradictory {item.status} transaction {item.transaction_id} prepared in "
                f"comment {item.prepared_comment.comment_id}"
                for item in members
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code="divergent-transactions",
        )
    return lowest


# ---------------------------------------------------------------------------
# The seam
# ---------------------------------------------------------------------------


class _Seam:
    def __init__(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        request: TransitionRequest,
        refresh: Callable[[], TransitionRequest] | None,
    ) -> None:
        self.runner = runner
        self.config = config
        self.request = request
        self.refresh = refresh
        self.own_transaction_ids: set[str] = set()

    # -- reads -------------------------------------------------------------

    def read(self) -> PublicationViews:
        request = self.request
        views = _read_views(
            self.runner,
            self.config,
            pr_number=request.pr_number,
            issue_number=request.primary_issue,
            plan_issue_number=request.plan_owning_issue,
        )
        foreign = actor_change_error(views.pr_view)
        if foreign is not None:
            raise foreign
        return views

    @property
    def codec(self) -> AuthorizationEntryCodec | None:
        return self.request.authorization_codec

    def _actor(self, views: PublicationViews) -> tuple[str, int]:
        return views.pr_view.actor_login, views.pr_view.actor_id

    # -- (2) sibling reconciliation -----------------------------------------

    def reconcile_siblings(self, views: PublicationViews) -> PublicationViews:
        request = self.request
        states = collect_transactions(
            views.pr_view, repository=request.repository, pr_number=request.pr_number
        )
        losers: list[TransactionState] = []
        for members in _sibling_groups(states).values():
            if len(members) < 2:
                continue
            canonical = _canonical_sibling(members)
            for member in members:
                if member is canonical:
                    continue
                if member.intent.scope != canonical.intent.scope:
                    raise _error(
                        "Sibling transactions belong to different scopes",
                        transaction_ids=[item.transaction_id for item in members],
                        code="divergent-transactions",
                    )
                losers.append(member)
        if not losers:
            return views
        for loser in losers:
            # Re-read and adopt an abort another invocation already appended.
            views = self.read()
            current = next(
                (
                    item
                    for item in collect_transactions(
                        views.pr_view,
                        repository=request.repository,
                        pr_number=request.pr_number,
                    )
                    if item.transaction_id == loser.transaction_id
                ),
                None,
            )
            if current is None or current.aborted:
                continue
            if current.committed:
                raise _error(
                    "A non-canonical sibling transaction committed during reconciliation",
                    intent=current.intent,
                    code="divergent-transactions",
                )
            self._write_abort(views, current, ABORT_SIBLING_CANONICAL, ())
        views = self.read()
        lost = sorted(
            loser.transaction_id
            for loser in losers
            if loser.transaction_id in self.own_transaction_ids
        )
        if lost:
            raise _error(
                "This invocation's transaction lost sibling reconciliation and was aborted; "
                "the canonical sibling is finished by a rerun",
                transaction_ids=lost,
                problems=tuple(f"aborted sibling transaction {tx_id}" for tx_id in lost),
                recovery=RECOVERY_RERUN,
                code=CODE_SIBLING_LOST,
            )
        return views

    def _write_abort(
        self,
        views: PublicationViews,
        state: TransactionState,
        reason: str,
        differing: tuple[tuple[str, str, str], ...],
    ) -> None:
        record = WorkflowTransactionRecord(
            phase=PHASE_ABORTED,
            transaction_id=state.transaction_id,
            prepared_comment_id=state.prepared_comment.comment_id,
            outcomes=_abort_outcomes(state, _published_entry_ids(state, views, self.codec)),
            abort_reason=reason,
            differing_fields=differing,
        )
        _write_verified(
            self.runner,
            self.config,
            surface=PR_THREAD_SURFACE,
            number=self.request.pr_number,
            body=format_transaction_record_comment(record),
            actor=self._actor(views),
            intent=state.intent,
            what=f"aborted ({reason}) record",
        )

    # -- (3) stored intent --------------------------------------------------

    def _plan_anchor(self, views: PublicationViews, plan_hash: str | None):
        if plan_hash is None or views.plan_issue_view is None:
            return None
        try:
            return resolve_approved_plan_anchor(views.plan_issue_view, plan_hash=plan_hash).comment
        except WorkflowTransactionError:
            return None

    def compare(self, views: PublicationViews, stored: WorkflowTransition):
        fresh = self.request.inputs()
        return compare_prepared_intent(
            stored,
            fresh,
            stored_plan_anchor=self._plan_anchor(views, stored.approved_plan_hash),
            fresh_plan_anchor=self._plan_anchor(views, fresh.approved_plan_hash),
        )

    def _refuse_contradiction(self, stored: TransactionState, comparison) -> None:
        raise _error(
            "A prepared workflow transaction contradicts the current authenticated inputs; "
            "nothing was written",
            intent=stored.intent,
            problems=tuple(
                f"contradictory prepared record in comment "
                f"{stored.prepared_comment.comment_id}: {item}"
                for item in comparison.contradictions
            ),
            code="prepared-intent-contradiction",
        )

    # -- (4) build ----------------------------------------------------------

    def _scheduler_checkpoint(self, views: PublicationViews) -> SchedulerCheckpointRef:
        request = self.request
        plan = request.approved_plan
        return select_scheduler_checkpoint(
            views.plan_issue_view,
            origin_flow=request.origin_flow,
            plan_hash=plan.plan_hash if plan else None,
            plan_subject=plan.plan_subject if plan else None,
            plan_candidate_key=plan.plan_candidate_key if plan else None,
        )

    def _effective_records(
        self, resolved: PrTransactionViews, committed: TransactionState
    ) -> dict[str, CommentRef]:
        request = self.request
        records: dict[str, CommentRef] = {}
        handoff = resolved.handoff
        if (
            handoff is not None
            and handoff.comment_id is not None
            and handoff.record_hash is not None
            and request.primary_issue is not None
        ):
            records[ENTRY_HANDOFF] = CommentRef(
                comment_thread_surface(ISSUE_THREAD_SURFACE, request.primary_issue),
                handoff.comment_id,
                handoff.record_hash,
            )
        contract = resolved.contract
        if contract is not None and contract.comment_id is not None and contract.record_hash:
            records[ENTRY_PR_CONTRACT] = CommentRef(
                resolved.views.pr_view.surface, contract.comment_id, contract.record_hash
            )
        authorization = effective_authorization_comment(
            committed, resolved.lineage, resolved.views.pr_view
        )
        if authorization is not None and self.codec is not None:
            records[ENTRY_AUTHORIZATION] = CommentRef(
                resolved.views.pr_view.surface,
                authorization.comment_id,
                self.codec.digest(authorization),
            )
        return records

    def _legacy_needs_no_write(self, resolved: PrTransactionViews) -> bool:
        """A consistent legacy v1 PR that already records this request is left alone."""
        request = self.request
        if resolved.era != ERA_LEGACY or not isinstance(request.managed, Unmanaged):
            return False
        contract = resolved.contract.contract if resolved.contract is not None else None
        if not isinstance(contract, PrExpectedClosingContract):
            return False
        if tuple(contract.expected_closing_issue_ids) != tuple(
            sorted(request.expected_closing_issue_ids)
        ) and tuple(contract.expected_closing_issue_ids) != tuple(
            request.expected_closing_issue_ids
        ):
            return False
        if contract.origin_flow != request.origin_flow:
            return False
        if request.primary_issue is None:
            return True
        handoff = resolved.handoff.handoff if resolved.handoff is not None else None
        if not isinstance(handoff, IssuePrHandoffMetadata):
            return False
        plan_hash = request.approved_plan.plan_hash if request.approved_plan else None
        return (
            handoff.flow == request.origin_flow
            and handoff.plan_hash == plan_hash
            and set(handoff.expected_closing_issue_ids or (handoff.issue_number,))
            == set(request.expected_closing_issue_ids)
        )

    def _refuse_legacy_flow_disagreement(self, resolved: PrTransactionViews) -> None:
        request = self.request
        recorded: list[tuple[str, int | None, str]] = []
        contract = resolved.contract.contract if resolved.contract is not None else None
        if isinstance(contract, PrExpectedClosingContract):
            recorded.append(("PR contract", resolved.contract.comment_id, contract.origin_flow))
        handoff = resolved.handoff.handoff if resolved.handoff is not None else None
        if isinstance(handoff, IssuePrHandoffMetadata):
            recorded.append(("handoff", resolved.handoff.comment_id, handoff.flow))
        disagreeing = [item for item in recorded if item[2] != request.origin_flow]
        if disagreeing:
            # Repairing this state needs a legacy-root correction, which only
            # final integration builds.
            raise _error(
                "A legacy PR whose version-1 flow disagrees with the derived flow cannot be "
                "upgraded by an initial transaction; nothing was written",
                problems=tuple(
                    f"contradictory version-1 {what} in comment {comment_id}: flow {flow}, "
                    f"derived flow {request.origin_flow}"
                    for what, comment_id, flow in disagreeing
                ),
                code="legacy-flow-disagreement",
            )

    def build_intent(
        self, views: PublicationViews, resolved: PrTransactionViews
    ) -> WorkflowTransition | None:
        """The fresh intent, or ``None`` when the last committed transaction already holds."""
        request = self.request
        fresh = request.inputs()
        committed = resolved.lineage.latest_committed
        if committed is not None:
            names_plan = fresh.approved_plan_hash != committed.intent.approved_plan_hash
            return plan_successor(
                committed.intent,
                fresh,
                effective_records=self._effective_records(resolved, committed),
                scheduler_checkpoint=self._scheduler_checkpoint(views) if names_plan else None,
            )
        self._refuse_legacy_flow_disagreement(resolved)
        issue_origin = request.origin_flow in {FLOW_ISSUE, FLOW_APPROVED_PLAN}
        entries = {
            ENTRY_HANDOFF: reissued if issue_origin else not_applicable,
            ENTRY_PR_CONTRACT: reissued,
            ENTRY_AUTHORIZATION: (
                reissued if fresh.managed_ci_generation is not None else not_applicable
            ),
            ENTRY_INITIAL_CODER_ROUND: (
                reissued if request.initial_coder_round is not None else not_applicable
            ),
        }
        return WorkflowTransition(
            repository=request.repository,
            primary_issue=request.primary_issue,
            pr_number=request.pr_number,
            base=request.base,
            head_sha=request.head_sha,
            origin_flow=request.origin_flow,
            approved_plan_hash=fresh.approved_plan_hash,
            expected_closing_issue_ids=fresh.expected_closing_issue_ids,
            scheduler_checkpoint=self._scheduler_checkpoint(views),
            record_set=tuple(entries[name](name) for name in RECORD_SET_ENTRY_NAMES),
            successor_kind=KIND_INITIAL,
            staged=request.staged,
            managed_ci_generation=fresh.managed_ci_generation,
        )

    # -- (5)/(6) writes -----------------------------------------------------

    def _expected_handoff_body(self, intent: WorkflowTransition) -> TrustedBody:
        text = format_issue_pr_handoff_v2_comment(
            derive_handoff_metadata(intent), repo=intent.repository
        )
        extra = self.request.handoff_extra_section
        if extra:
            prefix, signature = text.rsplit("\n-- ", 1)
            text = f"{prefix}\n{extra.strip()}\n-- {signature}"
        return _trusted(text)

    def _expected_contract_body(
        self, intent: WorkflowTransition, previous: ResolvedPrContract | None
    ) -> TrustedBody:
        kind, record_hash = contract_supersession(intent, previous)
        return _trusted(
            format_pr_contract_v2_comment(
                derive_pr_contract(
                    intent, supersession_kind=kind, supersedes_record_hash=record_hash
                )
            )
        )

    def _adopt_exact(
        self,
        intent: WorkflowTransition,
        prepared: AuthenticatedComment,
        bound: Sequence[AuthenticatedComment],
        expected: str,
        *,
        what: str,
    ) -> int | None:
        """Adopt the earliest byte-exact record bound to this transaction, if any."""
        problems = [
            f"contradictory {what} in comment {comment.comment_id}: "
            + (
                "its body is not the expected payload"
                if comment.body != expected
                else "edited after creation"
                if not _unedited(comment)
                else "not strictly later than the prepared record"
            )
            for comment in bound
            if comment.body != expected
            or not _unedited(comment)
            or not is_strictly_later(prepared, comment)
        ]
        if problems:
            raise _error(
                f"A {what} bound to this transaction contradicts its intent; nothing was "
                "written",
                intent=intent,
                problems=problems,
                code="record-contradiction",
            )
        return min((comment.comment_id for comment in bound), default=None)

    def adopt_entry(
        self, name: str, state: TransactionState, resolved: PrTransactionViews
    ) -> int | None:
        """The canonical comment ID of an already-published entry, or ``None``.

        This is the one adoption rule.  It runs read-first before every entry
        write and again over the final re-read immediately before the terminal
        write, so a contradictory, edited, or divergent record bound to this
        transaction stops the seam instead of being committed over.
        """
        intent = state.intent
        views = resolved.views
        tx_id = intent.transaction_id
        if name == ENTRY_HANDOFF:
            bound = [c for c, r in _v2_handoffs(views.issue_view) if r.transaction_id == tx_id]
            return self._adopt_exact(
                intent, state.prepared_comment, bound,
                str(self._expected_handoff_body(intent)), what="handoff",
            )
        if name == ENTRY_PR_CONTRACT:
            bound = [c for c, r in _v2_contracts(views.pr_view) if r.transaction_id == tx_id]
            return self._adopt_exact(
                intent, state.prepared_comment, bound,
                str(self._expected_contract_body(intent, resolved.contract)),
                what="PR contract",
            )
        if name == ENTRY_AUTHORIZATION:
            return self._adopt_authorization(state, resolved)
        expected = self._expected_coder_round(tx_id)
        candidates = _coder_round_candidates(
            intent, state.prepared_comment, views.pr_view,
            expected_body=str(expected) if expected is not None else None,
        )
        return candidates[0].comment_id if candidates else None

    def _expected_coder_round(self, tx_id: str) -> TrustedBody | None:
        supplied = self.request.initial_coder_round
        return supplied.render(tx_id) if supplied is not None else None

    def publish_entry(
        self,
        name: str,
        state: TransactionState,
        resolved: PrTransactionViews,
    ) -> PublicationViews | None:
        """Read-first adoption or a verified write.  Returns fresh views after a write."""
        intent = state.intent
        tx_id = intent.transaction_id
        actor = self._actor(resolved.views)
        if self.adopt_entry(name, state, resolved) is not None:
            return None
        if name == ENTRY_HANDOFF:
            assert intent.primary_issue is not None
            _write_verified(
                self.runner, self.config, surface=ISSUE_THREAD_SURFACE,
                number=intent.primary_issue, body=self._expected_handoff_body(intent),
                actor=actor, intent=intent, what="issue-to-PR handoff",
            )
        elif name == ENTRY_PR_CONTRACT:
            _write_verified(
                self.runner, self.config, surface=PR_THREAD_SURFACE, number=intent.pr_number,
                body=self._expected_contract_body(intent, resolved.contract),
                actor=actor, intent=intent, what="PR expected-closing contract",
            )
        elif name == ENTRY_AUTHORIZATION:
            managed = self.request.managed
            assert isinstance(managed, (Granted, Released)) and self.codec is not None
            _write_verified(
                self.runner, self.config, surface=PR_THREAD_SURFACE, number=intent.pr_number,
                body=self.codec.expected_body(managed.expected_payload, tx_id),
                actor=actor, intent=intent, what="bound managed-CI authorization",
            )
        else:
            expected = self._expected_coder_round(tx_id)
            if expected is None:
                return None  # waived: the coder response is unavailable
            _write_verified(
                self.runner, self.config, surface=PR_THREAD_SURFACE, number=intent.pr_number,
                body=expected, actor=actor, intent=intent, what="initial coder round",
            )
        # A competing invocation may have interleaved: reconcile below the strict
        # resolver before anything resolves the lineage again.
        return self.reconcile_siblings(self.read())

    def _adopt_authorization(
        self, state: TransactionState, resolved: PrTransactionViews
    ) -> int | None:
        managed = self.request.managed
        codec = self.codec
        intent = state.intent
        if not isinstance(managed, (Granted, Released)) or codec is None:
            raise _error(
                "A managed transaction cannot be finished without its managed-CI input",
                intent=intent,
                problems=(
                    "missing bound managed-CI authorization: "
                    + (managed.reason if isinstance(managed, Unavailable) else "no managed input"),
                ),
                recovery=RECOVERY_RERUN,
                code=CODE_MANAGED_UNAVAILABLE,
            )
        bound = [
            comment
            for comment, tx_id in _authorization_comments(resolved.views.pr_view, codec)
            if tx_id == intent.transaction_id
        ]
        for comment in bound:
            if not codec.adoptable(comment, managed.expected_payload, intent.transaction_id):
                raise _error(
                    "A bound managed-CI authorization contradicts the expected payload; "
                    "nothing was written",
                    intent=intent,
                    problems=(
                        f"contradictory bound authorization in comment {comment.comment_id}",
                    ),
                    code="record-contradiction",
                )
            codec.validate(
                comment, state=state, lineage=resolved.lineage,
                pr_view=resolved.views.pr_view, committed=False,
            )
        return min((comment.comment_id for comment in bound), default=None)

    # -- (8) terminal -------------------------------------------------------

    def commit_outcomes(
        self, state: TransactionState, resolved: PrTransactionViews
    ) -> tuple[EntryOutcome, ...]:
        """Re-validate every reissued entry from the final view and state its outcome."""
        published = {
            entry.name: comment_id
            for entry in state.intent.record_set
            if entry.disposition == DISPOSITION_REISSUED
            and (comment_id := self.adopt_entry(entry.name, state, resolved)) is not None
        }
        outcomes = []
        for entry in state.intent.record_set:
            if entry.disposition == DISPOSITION_INHERITED:
                outcomes.append(EntryOutcome(entry.name, status=STATUS_INHERITED))
            elif entry.disposition == DISPOSITION_NOT_APPLICABLE:
                outcomes.append(EntryOutcome(entry.name, status=STATUS_NOT_APPLICABLE))
            elif entry.name in published:
                outcomes.append(EntryOutcome(entry.name, comment_id=published[entry.name]))
            elif entry.name == ENTRY_INITIAL_CODER_ROUND:
                outcomes.append(EntryOutcome(entry.name, status=STATUS_WAIVED_CODER_RESPONSE))
            else:
                raise _error(
                    "A required record is still unpublished; the transaction cannot commit",
                    intent=state.intent,
                    problems=(f"missing {entry.name} record",),
                    recovery=RECOVERY_RERUN,
                    code=CODE_WRITE_FAILED,
                )
        return tuple(outcomes)

    # -- driver -------------------------------------------------------------

    def resolve(self, views: PublicationViews) -> PrTransactionViews:
        request = self.request
        return _resolve_views(
            views,
            repository=request.repository,
            pr_number=request.pr_number,
            issue_number=request.primary_issue,
            legacy_root_context=request.legacy_root_context,
        )

    def _abort_obsolete(self, views: PublicationViews, pending: TransactionState, comparison):
        assert comparison.abort_reason is not None
        self._write_abort(views, pending, comparison.abort_reason, comparison.differing_fields)

    def run(self) -> CommittedTransaction | None:
        for _attempt in range(_MAX_PUBLICATION_ATTEMPTS):
            views = self.reconcile_siblings(self.read())
            resolved = self.resolve(views)
            pending = resolved.lineage.pending
            if pending is not None:
                if pending.intent.scope != (
                    self.request.repository.casefold(),
                    self.request.primary_issue,
                    self.request.pr_number,
                ):
                    raise _error(
                        "A prepared workflow transaction on this PR belongs to another scope",
                        intent=pending.intent,
                        problems=(
                            "contradictory prepared record in comment "
                            f"{pending.prepared_comment.comment_id}",
                        ),
                        code="prepared-intent-contradiction",
                    )
                comparison = self.compare(views, pending.intent)
                if comparison.outcome == "contradiction":
                    self._refuse_contradiction(pending, comparison)
                if comparison.outcome == "obsolete":
                    self._abort_obsolete(views, pending, comparison)
                    continue
                state = pending
            else:
                intent = self.build_intent(views, resolved)
                if intent is None:
                    committed = resolved.lineage.latest_committed
                    if committed is None:
                        return None
                    return self._verified_commit(committed, resolved)
                if resolved.lineage.latest_committed is None and self._legacy_needs_no_write(
                    resolved
                ):
                    return None
                if resolved.lineage.state(intent.transaction_id) is not None:
                    raise _error(
                        "The fresh intent equals a transaction that was already aborted on "
                        "this PR",
                        intent=intent,
                        code="aborted-intent-reused",
                    )
                actor = self._actor(views)
                self.own_transaction_ids.add(intent.transaction_id)
                _write_verified(
                    self.runner, self.config, surface=PR_THREAD_SURFACE,
                    number=intent.pr_number,
                    body=format_transaction_record_comment(
                        prepared_record(intent, writer_login=actor[0], writer_id=actor[1])
                    ),
                    actor=actor, intent=intent, what="prepared record",
                )
                continue  # re-read, reconcile, and adopt what is now stored
            finished = self._finish(state, resolved)
            if finished is not None:
                return finished
        raise _error(
            "The workflow transaction kept changing while it was being published",
            recovery=RECOVERY_RERUN,
            code=CODE_WRITE_FAILED,
        )

    def _verified_commit(
        self, committed: TransactionState, resolved: PrTransactionViews
    ) -> CommittedTransaction:
        """Never report a commit as a success unless the gate's entry rules hold.

        GitHub offers no compare-and-swap, so a record can still land between the
        final re-read and the terminal write; the same rules the gate applies
        refuse that state here, on this run and on every rerun.
        """
        _gate_authorization(committed, resolved, self.codec)
        _gate_initial_coder_round(resolved.lineage, resolved.views.pr_view)
        return CommittedTransaction(
            committed, resolved.lineage, resolved.contract, resolved.handoff
        )

    def _finish(
        self, state: TransactionState, resolved: PrTransactionViews
    ) -> CommittedTransaction | None:
        """Publish the adopted intent's entries and commit; ``None`` restarts the loop."""
        if state.intent.scheduler_checkpoint.reference is not None:
            plan = self.request.approved_plan
            verify_scheduler_checkpoint(
                state.intent,
                resolved.views.plan_issue_view,
                plan_candidate_key=plan.plan_candidate_key if plan else None,
            )
        for entry in state.intent.record_set:
            if entry.disposition != DISPOSITION_REISSUED:
                continue
            views = self.publish_entry(entry.name, state, resolved)
            if views is not None:
                resolved = self.resolve(views)
        # (7) full re-read, reconciliation again, second comparison.
        if self.refresh is not None:
            self.request = self.refresh()
        views = self.reconcile_siblings(self.read())
        resolved = self.resolve(views)
        current = resolved.lineage.state(state.transaction_id)
        if current is None or current.aborted:
            return None
        if current.committed:
            return self._verified_commit(current, resolved)
        comparison = self.compare(views, current.intent)
        if comparison.outcome == "contradiction":
            self._refuse_contradiction(current, comparison)
        if comparison.outcome == "obsolete":
            self._abort_obsolete(views, current, comparison)
            return None
        record = WorkflowTransactionRecord(
            phase=PHASE_COMMITTED,
            transaction_id=current.transaction_id,
            prepared_comment_id=current.prepared_comment.comment_id,
            outcomes=self.commit_outcomes(current, resolved),
        )
        _write_verified(
            self.runner, self.config, surface=PR_THREAD_SURFACE,
            number=current.intent.pr_number,
            body=format_transaction_record_comment(record),
            actor=self._actor(views), intent=current.intent, what="committed record",
        )
        resolved = self.resolve(self.reconcile_siblings(self.read()))
        final = resolved.lineage.state(state.transaction_id)
        if final is None or not final.committed:
            raise _error(
                "The committed record could not be read back",
                intent=state.intent,
                recovery=RECOVERY_RERUN,
                code=CODE_WRITE_FAILED,
            )
        return self._verified_commit(final, resolved)


def contract_supersession(
    intent: WorkflowTransition, previous: ResolvedPrContract | None
) -> tuple[str | None, str | None]:
    """The supersession a reissued PR contract declares: a pure function of the record
    resolved for the predecessor chain (or the v1 contract for an upgrade)."""
    prior = previous.contract if previous is not None else None
    if prior is None:
        return None, None
    assert previous is not None
    same_scope = prior.primary_issue_number == intent.primary_issue and tuple(
        prior.expected_closing_issue_ids
    ) == tuple(intent.expected_closing_issue_ids)
    same_flow = prior.origin_flow == intent.origin_flow
    widened = prior.primary_issue_number == intent.primary_issue and set(
        prior.expected_closing_issue_ids
    ) < set(intent.expected_closing_issue_ids)
    if same_scope and same_flow:
        return None, None
    if widened and same_flow:
        return "closing-widening", previous.record_hash
    if same_scope:
        return "flow-correction", previous.record_hash
    if widened:
        return PR_CONTRACT_SUPERSESSION_COMBINED, previous.record_hash
    raise _error(
        "The derived PR contract neither restates nor validly supersedes the recorded one",
        intent=intent,
        problems=(f"contradictory PR contract in comment {previous.comment_id}",),
        code="contract-supersession-invalid",
    )


def publish_transition(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    request: TransitionRequest,
    refresh: Callable[[], TransitionRequest] | None = None,
) -> CommittedTransaction | None:
    """Publish one workflow transition, or finish the one a previous run left behind.

    Returns ``None`` only for a consistent legacy v1 PR that needs no write.
    ``refresh`` re-derives the request from live authenticated inputs for the
    comparison that runs immediately before the terminal write.
    """
    if isinstance(request.managed, Unavailable):
        raise _error(
            "The managed-CI input for this transition is unavailable; nothing was written",
            problems=(f"missing managed-CI input: {request.managed.reason}",),
            recovery=RECOVERY_RERUN,
            code=CODE_MANAGED_UNAVAILABLE,
        )
    return _Seam(runner, config, request, refresh).run()


def ensure_head_transaction(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    request: TransitionRequest,
    refresh: Callable[[], TransitionRequest] | None = None,
) -> CommittedTransaction | None:
    """Commit the successor a new live head needs.  Writes only on a head difference
    (or to finish a pending transaction); a legacy-era PR is left untouched."""
    resolved = read_pr_transaction_views(
        runner, config, request.pr_number, request.primary_issue,
        plan_issue_number=request.plan_owning_issue,
    ) if _has_transaction_records(runner, config, request) else None
    if resolved is None:
        return None
    committed = resolved.lineage.latest_committed
    if (
        committed is not None
        and resolved.lineage.pending is None
        and committed.intent.head_sha == request.head_sha
    ):
        return CommittedTransaction(
            committed, resolved.lineage, resolved.contract, resolved.handoff
        )
    return publish_transition(runner, config=config, request=request, refresh=refresh)


def _has_transaction_records(
    runner: Runner, config: AgentLoopConfig, request: TransitionRequest
) -> bool:
    views = _read_views(
        runner, config, pr_number=request.pr_number, issue_number=request.primary_issue
    )
    return classify_transaction_era(views.pr_view, views.issue_view) == ERA_TRANSACTION


def managed_release_hook(
    runner: Runner,
    config: AgentLoopConfig,
    *,
    pr_number: int,
    issue_number: int,
) -> Callable[[str], None]:
    """The pre-deletion hook of a label-removing managed-CI release.

    ``managed_ci`` calls it immediately before it deletes the managed label, so
    the release is committed before the label goes and a seam failure leaves
    the label untouched.  On a transaction-era PR whose committed generation is
    granted it commits the released ``managed-ci-continuity`` successor for the
    head being released; on a legacy-era, null-generation, or already released PR it
    writes nothing.  ``managed_ci`` never imports this module: the PR loop
    stores the returned callable on the managed contract.
    """

    def commit_release(head: str) -> None:
        # Imported here: the bound codec module imports ``managed_ci``.
        from .managed_ci_bound_authorization import (
            BoundAuthorizationCodec,
            ordinary_release_payload,
            released_generation,
        )

        resolved = read_pr_transaction_views(runner, config, pr_number, issue_number)
        committed = resolved.lineage.latest_committed
        if resolved.era != ERA_TRANSACTION or committed is None:
            return
        intent = committed.intent
        if intent.managed_ci_generation is None or intent.primary_issue is None:
            return
        view = resolved.views.pr_view
        generation = released_generation(
            repository=intent.repository, issue_number=intent.primary_issue,
            pr_number=pr_number, base_ref=intent.base, actor_id=view.actor_id,
        )
        if intent.managed_ci_generation == generation:
            return
        plan = None
        if intent.approved_plan_hash is not None:
            plan_views = _read_views(
                runner, config, pr_number=pr_number, issue_number=issue_number,
                plan_issue_number=intent.plan_owning_issue,
            )
            plan = ApprovedPlanInput(
                intent.approved_plan_hash,
                None,
                recover_checkpoint_candidate_key(intent, plan_views.plan_issue_view),
            )
        payload = ordinary_release_payload(
            repository=intent.repository, issue_number=intent.primary_issue,
            pr_number=pr_number, base_ref=intent.base, head_sha=head,
            actor_login=view.actor_login, actor_id=view.actor_id,
        )
        publish_transition(
            runner,
            config=config,
            request=TransitionRequest(
                repository=intent.repository,
                pr_number=pr_number,
                base=intent.base,
                head_sha=head,
                origin_path=ORIGIN_PR_RESUME,
                expected_closing_issue_ids=tuple(intent.expected_closing_issue_ids),
                primary_issue=intent.primary_issue,
                approved_plan=plan,
                staged=intent.staged,
                managed=Released(payload, generation),
                authorization_codec=BoundAuthorizationCodec(),
            ),
        )

    return commit_release


# ---------------------------------------------------------------------------
# Effective authorization
# ---------------------------------------------------------------------------


def effective_authorization_comment(
    state: TransactionState,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
) -> AuthenticatedComment | None:
    """The canonical authorization comment of a committed transaction, following
    ``inherited`` references back to the transaction that issued it."""
    cursor: TransactionState | None = state
    while cursor is not None:
        entry = cursor.intent.entry(ENTRY_AUTHORIZATION)
        if entry.disposition == DISPOSITION_NOT_APPLICABLE:
            return None
        if entry.disposition == DISPOSITION_REISSUED:
            outcome = cursor.outcome(ENTRY_AUTHORIZATION)
            if outcome is None or outcome.comment_id is None:
                return None
            return pr_view.comment(outcome.comment_id)
        predecessor = cursor.intent.predecessor_transaction_id
        cursor = lineage.state(predecessor) if predecessor is not None else None
    return None


def _issuing_state(
    state: TransactionState, lineage: TransactionLineage
) -> TransactionState | None:
    cursor: TransactionState | None = state
    while cursor is not None:
        if cursor.intent.entry(ENTRY_AUTHORIZATION).disposition == DISPOSITION_REISSUED:
            return cursor
        predecessor = cursor.intent.predecessor_transaction_id
        cursor = lineage.state(predecessor) if predecessor is not None else None
    return None


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def require_committed_transaction(
    views: PublicationViews,
    *,
    repository: str,
    pr_number: int,
    issue_number: int | None,
    live_head: str,
    pr_state: str = "OPEN",
    allow_terminal_pr_state: bool = False,
    authorization_codec: AuthorizationEntryCodec | None = None,
    plan_candidate_key: PlanCandidateKey | None = None,
    legacy_root_context: LegacyRootContext | None = None,
) -> CommittedTransaction | None:
    """Fail closed unless the live head has a committed canonical transaction.

    Returns ``None`` for a legacy-era PR, whose callers keep today's v1 checks.
    Never writes and never reconciles.  For a merged or closed PR pass the PR's
    final head as ``live_head`` with ``allow_terminal_pr_state=True``.
    """
    foreign = actor_change_error(views.pr_view)
    if foreign is not None:
        raise foreign
    if pr_state != "OPEN" and not allow_terminal_pr_state:
        raise _error(
            f"PR #{pr_number} is {pr_state}; this consumer requires an open PR",
            code="pr-not-open",
        )
    if (
        classify_transaction_era(views.pr_view, views.issue_view) == ERA_LEGACY
        and not collect_transactions(views.pr_view, repository=repository, pr_number=pr_number)
    ):
        return None
    resolved = _resolve_views(
        views, repository=repository, pr_number=pr_number, issue_number=issue_number,
        legacy_root_context=legacy_root_context,
    )
    lineage = resolved.lineage
    pending = lineage.pending
    if pending is not None:
        raise _error(
            "A prepared workflow transaction has not committed; a partial transaction grants "
            "no approval, qualification, CI, phase-progress, or merge authority",
            intent=pending.intent,
            problems=(
                f"missing committed record for prepared comment "
                f"{pending.prepared_comment.comment_id}",
            ),
            recovery=RECOVERY_RERUN,
            code=CODE_PENDING,
        )
    committed = lineage.latest_committed
    if committed is None:
        raise _error(
            f"PR #{pr_number} carries version-2 records but no committed workflow transaction; "
            "it stays transaction-era and grants nothing",
            problems=("missing committed transaction record",) + lineage.ignored_foreign,
            recovery=RECOVERY_RERUN,
            code=CODE_UNCOMMITTED,
        )
    intent = committed.intent
    if intent.head_sha != live_head:
        raise _error(
            "The committed workflow transaction is bound to another head; nothing is "
            "authoritative for the live head until a head-advance successor commits",
            intent=intent,
            problems=(
                f"missing committed transaction for head {live_head} (committed head "
                f"{intent.head_sha})",
            ),
            recovery=RECOVERY_RERUN,
            code=CODE_HEAD_NOT_COMMITTED,
        )
    if intent.primary_issue != issue_number:
        raise _error(
            f"The committed transaction belongs to issue #{intent.primary_issue}, not "
            f"#{issue_number}",
            intent=intent,
            code="scope-mismatch",
        )
    contract_entry = intent.entry(ENTRY_PR_CONTRACT)
    if contract_entry.disposition != DISPOSITION_NOT_APPLICABLE and (
        resolved.contract is None or resolved.contract.transaction_id != committed.transaction_id
    ):
        raise _error(
            "The committed transaction's PR contract does not resolve",
            intent=intent, problems=("missing PR contract record",), code="record-missing",
        )
    if intent.entry(ENTRY_HANDOFF).disposition != DISPOSITION_NOT_APPLICABLE and (
        resolved.handoff is None or resolved.handoff.transaction_id != committed.transaction_id
    ):
        raise _error(
            "The committed transaction's issue-to-PR handoff does not resolve",
            intent=intent, problems=("missing handoff record",), code="record-missing",
        )
    _gate_authorization(committed, resolved, authorization_codec)
    _gate_initial_coder_round(lineage, views.pr_view)
    if intent.scheduler_checkpoint.reference is not None:
        verify_scheduler_checkpoint(
            intent, views.plan_issue_view, plan_candidate_key=plan_candidate_key
        )
    return CommittedTransaction(committed, lineage, resolved.contract, resolved.handoff)


def _gate_authorization(
    committed: TransactionState,
    resolved: PrTransactionViews,
    codec: AuthorizationEntryCodec | None,
) -> None:
    intent = committed.intent
    if intent.managed_ci_generation is None:
        return
    pr_view = resolved.views.pr_view
    issuing = _issuing_state(committed, resolved.lineage)
    comment = effective_authorization_comment(committed, resolved.lineage, pr_view)
    if codec is None or issuing is None or comment is None:
        raise _error(
            "The committed transaction's bound managed-CI authorization cannot be validated",
            intent=intent,
            problems=("missing bound managed-CI authorization record",),
            code="authorization-invalid",
        )
    entry = intent.entry(ENTRY_AUTHORIZATION)
    if entry.inherited is not None and codec.digest(comment) != entry.inherited.digest:
        raise _error(
            "An inherited bound managed-CI authorization no longer matches its recorded digest",
            intent=intent,
            problems=(f"contradictory bound authorization in comment {comment.comment_id}",),
            code="authorization-invalid",
        )
    # Presence at the recorded ID is not sufficient: the record is judged
    # against the transaction that issued it.
    codec.validate(
        comment, state=issuing, lineage=resolved.lineage, pr_view=pr_view, committed=True
    )


def _gate_initial_coder_round(
    lineage: TransactionLineage, pr_view: AuthenticatedCommentView
) -> None:
    root = lineage.chain[0] if lineage.chain else None
    if root is None or not root.committed:
        return
    outcome = root.outcome(ENTRY_INITIAL_CODER_ROUND)
    if outcome is None or outcome.comment_id is None:
        return  # not applicable, or explicitly waived
    assert root.bound_prepared_comment is not None
    candidates = _coder_round_candidates(root.intent, root.bound_prepared_comment, pr_view)
    if not candidates or candidates[0].comment_id != outcome.comment_id:
        raise _error(
            "The initial coder round named by the committed transaction no longer holds",
            intent=root.intent,
            problems=(f"missing initial coder round comment {outcome.comment_id}",),
            code="initial-coder-round-contradiction",
        )


def resolve_round_authority(
    ensure: Callable[[], CommittedTransaction | None],
    gate: Callable[[], CommittedTransaction | None],
) -> RoundAuthority:
    """Run the head-advance check then the gate for one round.

    A recoverable failure yields :class:`NoLiveHeadAuthority` so the round
    continues with fresh reviewers and no reuse; an integrity failure raises.
    """
    try:
        ensure()
        committed = gate()
    except WorkflowTransactionError as exc:
        if is_recoverable(exc):
            return NoLiveHeadAuthority(exc)
        raise
    return committed if committed is not None else LegacyEra()


# ---------------------------------------------------------------------------
# Canonical-PR discovery (authority form) and publication routing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalHandoffView:
    """The fields both handoff versions share, plus the era-specific ones."""

    issue_number: int
    pr_number: int
    pr_url: str
    pr_head_sha: str
    flow: str
    plan_hash: str | None
    expected_closing_issue_ids: tuple[int, ...]
    contract_hash: str | None
    era: str
    comment_id: int
    transaction_id: str | None = None
    legacy_record: IssuePrHandoffMetadata | None = None

    @classmethod
    def of(cls, resolved: ResolvedHandoff) -> "CanonicalHandoffView":
        record = resolved.handoff
        assert record is not None and resolved.comment_id is not None
        legacy = record if isinstance(record, IssuePrHandoffMetadata) else None
        return cls(
            issue_number=record.issue_number,
            pr_number=record.pr_number,
            pr_url=record.pr_url,
            pr_head_sha=record.pr_head_sha,
            flow=record.flow,
            plan_hash=record.plan_hash,
            expected_closing_issue_ids=tuple(
                record.expected_closing_issue_ids or (record.issue_number,)
            ),
            contract_hash=record.contract_hash,
            era=ERA_LEGACY if legacy is not None else ERA_TRANSACTION,
            comment_id=resolved.comment_id,
            transaction_id=resolved.transaction_id,
            legacy_record=legacy,
        )


@dataclass(frozen=True)
class DiscoveredCanonicalPr:
    pr_number: int
    handoff: CanonicalHandoffView
    transaction_id: str | None

    @property
    def era(self) -> str:
        return self.handoff.era


@dataclass(frozen=True)
class PredecessorBinding:
    """The last committed binding, for a writer's own preconditions.  Not authority."""

    plan_hash: str | None
    contract_hash: str | None
    expected_closing_issue_ids: tuple[int, ...]
    pr_head_sha: str
    flow: str
    transaction_id: str


@dataclass(frozen=True)
class Committed:
    canonical: DiscoveredCanonicalPr


@dataclass(frozen=True)
class Legacy:
    canonical: DiscoveredCanonicalPr


@dataclass(frozen=True)
class NoCandidate:
    pass


@dataclass(frozen=True)
class Recoverable:
    pr_number: int
    pending_transaction_ids: tuple[str, ...]


@dataclass(frozen=True)
class RecoverableSuccessor:
    pr_number: int
    pending_transaction_ids: tuple[str, ...]
    predecessor: PredecessorBinding


CanonicalRoute = Committed | Legacy | NoCandidate | Recoverable | RecoverableSuccessor


@dataclass(frozen=True)
class _Candidate:
    pr_number: int
    kind: str  # committed | legacy | partial | committed-with-pending-successor
    handoff: ResolvedHandoff | None = None
    pending: tuple[TransactionState, ...] = ()
    error: WorkflowTransactionError | None = None
    # Comment ID of the issue-side handoff that orders this candidate.
    order_comment_id: int | None = None


def _partial_error(issue_number: int, pr_number: int, detail: str, *, cause=None):
    problems = (f"missing committed handoff for candidate PR #{pr_number}: {detail}",)
    if isinstance(cause, WorkflowTransactionError):
        problems += cause.problems
    return WorkflowTransactionError(
        f"Issue #{issue_number} names PR #{pr_number} in a handoff whose workflow transaction "
        "is not committed; it is never skipped in favour of another candidate or the legacy "
        "path",
        transaction_ids=cause.transaction_ids if isinstance(cause, WorkflowTransactionError) else (),
        problems=problems,
        recovery_action=(
            cause.recovery_action
            if isinstance(cause, WorkflowTransactionError)
            else RECOVERY_RERUN
        ),
        code=CODE_PARTIAL_CANDIDATE,
    )


def _classify_candidate(
    issue_view: AuthenticatedCommentView,
    pr_view: AuthenticatedCommentView,
    *,
    repository: str,
    issue_number: int,
    pr_number: int,
) -> _Candidate:
    foreign = actor_change_error(pr_view)
    if foreign is not None:
        raise foreign
    try:
        lineage = resolve_transaction_lineage(
            pr_view, repository=repository, pr_number=pr_number
        )
        handoff = resolve_handoff_lineage(
            issue_view, lineage, repository=repository, issue_number=issue_number
        )
    except WorkflowTransactionError as exc:
        return _Candidate(pr_number, "partial", error=exc)
    pending = (lineage.pending,) if lineage.pending is not None else ()
    if handoff is None or handoff.handoff is None:
        return _Candidate(pr_number, "partial", handoff=handoff, pending=pending)
    if handoff.era == ERA_LEGACY:
        return _Candidate(pr_number, "legacy", handoff, order_comment_id=handoff.comment_id)
    if handoff.transaction_id is None:
        # A v1 handoff on a transaction-era PR with no committed transaction.
        return _Candidate(pr_number, "partial", handoff=handoff, pending=pending)
    kind = "committed-with-pending-successor" if pending else "committed"
    return _Candidate(pr_number, kind, handoff, pending, order_comment_id=handoff.comment_id)


def _candidates(
    runner: Runner, config: AgentLoopConfig, issue_number: int
) -> tuple[AuthenticatedCommentView, dict[int, AuthenticatedCommentView]]:
    issue_view = read_authenticated_protocol_comments(
        runner, config=config, surface_kind=ISSUE_THREAD_SURFACE, number=issue_number
    )
    pr_views = {
        number: read_authenticated_protocol_comments(
            runner, config=config, surface_kind=PR_THREAD_SURFACE, number=number
        )
        for number in handoff_candidate_pr_numbers(issue_view, issue_number=issue_number)
    }
    return issue_view, pr_views


def _select_winner(issue_number: int, candidates: Sequence[_Candidate]) -> _Candidate | None:
    ordered = [item for item in candidates if item.order_comment_id is not None]
    if not ordered:
        return None
    best = max(item.order_comment_id for item in ordered)  # type: ignore[type-var]
    winners = [item for item in ordered if item.order_comment_id == best]
    if len(winners) != 1:
        raise WorkflowTransactionError(
            f"Issue #{issue_number} has candidate PRs whose resolved handoffs cannot be ordered",
            problems=tuple(
                f"contradictory handoff for PR #{item.pr_number} in comment "
                f"{item.order_comment_id}"
                for item in winners
            ),
            recovery_action=RECOVERY_OPERATOR_REVIEW,
            code="divergent-candidates",
        )
    return winners[0]


def discover_canonical_issue_pr(
    runner: Runner, config: AgentLoopConfig, issue_number: int
) -> DiscoveredCanonicalPr | None:
    """Authority form: the canonical PR of an issue across handoff versions.

    Read-only.  Any partial candidate, and a committed chain with a pending
    successor, raises; nothing falls through to another candidate or to v1.
    """
    issue_view, pr_views = _candidates(runner, config, issue_number)
    classified = [
        _classify_candidate(
            issue_view, view, repository=config.repo, issue_number=issue_number, pr_number=number
        )
        for number, view in pr_views.items()
    ]
    for item in classified:
        if item.kind == "partial":
            raise _partial_error(
                issue_number, item.pr_number, "no committed transaction", cause=item.error
            )
        if item.kind == "committed-with-pending-successor":
            pending = item.pending[0]
            raise _error(
                f"PR #{item.pr_number} has a prepared successor transaction that has not "
                "committed; its committed predecessor is not consumed while it is pending",
                intent=pending.intent,
                problems=(
                    "missing committed record for prepared comment "
                    f"{pending.prepared_comment.comment_id}",
                ),
                recovery=RECOVERY_RERUN,
                code=CODE_PENDING,
            )
    winner = _select_winner(issue_number, classified)
    if winner is None:
        return None
    assert winner.handoff is not None
    return DiscoveredCanonicalPr(
        winner.pr_number, CanonicalHandoffView.of(winner.handoff), winner.handoff.transaction_id
    )


def recover_checkpoint_candidate_key(
    intent: WorkflowTransition, plan_issue_view: AuthenticatedCommentView | None
) -> PlanCandidateKey | None:
    """Recover the approved plan's candidate key from the hashed checkpoint reference.

    The intent hashes the checkpoint comment's round-metadata digest, and the
    seam verified that record against the approved key before the prepared
    write.  A comment that still matches the hashed digest therefore still
    carries exactly that key.  Anything else returns ``None`` and is refused by
    ``verify_scheduler_checkpoint``, which re-checks the digest, the subject,
    the anchor, and the ordering itself.
    """
    reference = intent.scheduler_checkpoint.reference
    if reference is None or plan_issue_view is None:
        return None
    comment = plan_issue_view.comment(reference.comment_id)
    if comment is None:
        return None
    try:
        if round_metadata_digest(comment) != reference.digest:
            return None
        for record in _extract_round_metadata_records(plan_issue_view.authored, flow="plan"):
            if plan_issue_view.authored[record.index] is comment:
                raw = record.metadata.plan_candidate_key
                return PlanCandidateKey.from_mapping(raw) if raw is not None else None
    except AgentLoopError:
        return None
    return None


def gate_canonical_issue_pr(
    runner: Runner,
    config: AgentLoopConfig,
    discovered: DiscoveredCanonicalPr,
    *,
    issue_number: int,
    live_head: str,
    pr_state: str,
    authorization_codec: AuthorizationEntryCodec | None = None,
    plan_candidate_key: PlanCandidateKey | None = None,
) -> CommittedTransaction:
    """Gate a discovered transaction-era canonical PR for an issue-only consumer.

    Read-only.  ``live_head`` is the PR's live head, or its final head when the
    PR is merged or closed (the terminal-state form staged phase progress uses).
    The scheduler checkpoint is re-verified on the plan-owning issue; a caller
    that holds no approved-plan session recovers the candidate key durably.
    """
    pr_number = discovered.pr_number
    resolved = read_pr_transaction_views(runner, config, pr_number, issue_number)
    committed = resolved.lineage.latest_committed
    views = resolved.views
    if committed is not None and committed.intent.plan_owning_issue != issue_number:
        views = _read_views(
            runner,
            config,
            pr_number=pr_number,
            issue_number=issue_number,
            plan_issue_number=committed.intent.plan_owning_issue,
        )
    if plan_candidate_key is None and committed is not None:
        plan_candidate_key = recover_checkpoint_candidate_key(
            committed.intent, views.plan_issue_view
        )
    gated = require_committed_transaction(
        views,
        repository=config.repo,
        pr_number=pr_number,
        issue_number=issue_number,
        live_head=live_head,
        pr_state=pr_state,
        allow_terminal_pr_state=True,
        authorization_codec=authorization_codec,
        plan_candidate_key=plan_candidate_key,
    )
    if gated is None or gated.transaction_id != discovered.transaction_id:
        raise _error(
            f"PR #{pr_number} no longer resolves the committed transaction its handoff for "
            f"issue #{issue_number} is bound to",
            transaction_ids=(discovered.transaction_id,) if discovered.transaction_id else (),
            problems=(f"contradictory handoff in comment {discovered.handoff.comment_id}",),
            code="record-missing",
        )
    return gated


def _binding(handoff: ResolvedHandoff) -> PredecessorBinding:
    record = handoff.handoff
    assert isinstance(record, IssuePrHandoffMetadataV2) and handoff.transaction_id is not None
    return PredecessorBinding(
        plan_hash=record.plan_hash,
        contract_hash=record.contract_hash,
        expected_closing_issue_ids=tuple(record.expected_closing_issue_ids),
        pr_head_sha=record.pr_head_sha,
        flow=record.flow,
        transaction_id=handoff.transaction_id,
    )


def _router_refusal(issue_number: int, pr_number: int, problem: str, *, ids=()):
    return WorkflowTransactionError(
        f"The interrupted publication on PR #{pr_number} for issue #{issue_number} cannot be "
        "routed back to the publication seam",
        transaction_ids=tuple(ids),
        problems=(problem,),
        recovery_action=RECOVERY_OPERATOR_REVIEW,
        code="unroutable-candidate",
    )


def _filtered_views(
    issue_view: AuthenticatedCommentView,
    pr_view: AuthenticatedCommentView,
    withheld: frozenset[str],
    codec: AuthorizationEntryCodec | None,
    *,
    repository: str,
    pr_number: int,
) -> tuple[
    AuthenticatedCommentView,
    AuthenticatedCommentView,
    list[tuple[AuthenticatedComment, str, str, object]],
]:
    """Router-private views without the withheld transactions' records.

    Removal is decided by each record's decoded transaction ID.  v1 records,
    unbound comments, and committed or aborted transactions' records stay.
    """
    # Every withheld record keeps its entry name and decoded payload so the
    # router can judge it against the stored intent it is bound to.
    removed: list[tuple[AuthenticatedComment, str, str, object]] = []

    def drop(comment: AuthenticatedComment, tx_id: str, name: str, record: object) -> None:
        if tx_id in withheld:
            removed.append((comment, tx_id, name, record))

    for comment, contract in _v2_contracts(pr_view):
        drop(comment, contract.transaction_id, ENTRY_PR_CONTRACT, contract)
    for comment, tx_id in _authorization_comments(pr_view, codec):
        drop(comment, tx_id, ENTRY_AUTHORIZATION, None)
    for comment, metadata in _tagged_coder_rounds(pr_view):
        drop(comment, metadata.workflow_transaction_id, ENTRY_INITIAL_CODER_ROUND, metadata)
    for comment, record in _v2_handoffs(issue_view):
        # Any handoff in this issue thread bound to a withheld transaction is
        # withheld, whatever PR or issue it names: the mismatch is judged below.
        drop(comment, record.transaction_id, ENTRY_HANDOFF, record)
    prepared_ids: set[int] = set()
    for state in collect_transactions(pr_view, repository=repository, pr_number=pr_number):
        if state.transaction_id in withheld:
            prepared_ids.add(state.prepared_comment.comment_id)
            prepared_ids.update(item.comment_id for item in state.prepared_duplicates)
    gone = {comment.comment_id for comment, _tx, _name, _record in removed} | prepared_ids
    filtered_pr = dataclasses.replace(
        pr_view, authored=tuple(c for c in pr_view.authored if c.comment_id not in gone)
    )
    filtered_issue = dataclasses.replace(
        issue_view, authored=tuple(c for c in issue_view.authored if c.comment_id not in gone)
    )
    return filtered_issue, filtered_pr, removed


def _removed_record_problem(
    comment: AuthenticatedComment,
    name: str,
    record: object,
    state: TransactionState,
    *,
    lineage: TransactionLineage,
    pr_view: AuthenticatedCommentView,
    codec: AuthorizationEntryCodec | None,
) -> str | None:
    """Why a withheld record contradicts the stored intent it is bound to, if it does.

    This replaces, for withheld records, what stage A's bound-state and
    derived-payload checks establish for records that stay in the views.
    """
    intent = state.intent
    where = f"contradictory {name} record in comment {comment.comment_id}"
    if intent.entry(name).disposition != DISPOSITION_REISSUED:
        return f"{where}: its transaction does not reissue that entry"
    if name == ENTRY_HANDOFF:
        if record != derive_handoff_metadata(intent):
            return f"{where}: it does not equal the handoff derived from its stored intent"
    elif name == ENTRY_PR_CONTRACT:
        assert isinstance(record, PrExpectedClosingContractV2)
        if record != derive_pr_contract(
            intent,
            supersession_kind=record.supersession_kind,
            supersedes_record_hash=record.supersedes_record_hash,
        ):
            return f"{where}: it does not equal the contract derived from its stored intent"
    elif name == ENTRY_AUTHORIZATION:
        assert codec is not None
        try:
            codec.validate(
                comment, state=state, lineage=lineage, pr_view=pr_view, committed=False
            )
        except WorkflowTransactionError as exc:
            return f"{where}: {exc.summary}"
    else:
        metadata = record
        if (
            metadata.role != "coder"  # type: ignore[attr-defined]
            or metadata.round_number != 1  # type: ignore[attr-defined]
            or metadata.subject != intent.head_sha  # type: ignore[attr-defined]
        ):
            return f"{where}: its round metadata contradicts the stored intent"
    return None


def _route_sibling_group(
    issue_view: AuthenticatedCommentView,
    pr_view: AuthenticatedCommentView,
    prepared_only: Sequence[TransactionState],
    *,
    repository: str,
    issue_number: int,
    pr_number: int,
    codec: AuthorizationEntryCodec | None,
) -> tuple[_Candidate, PredecessorBinding | None]:
    """Tier 2: classify a true prepared-only sibling group over filtered views."""
    withheld = frozenset(item.transaction_id for item in prepared_only)
    by_id = {item.transaction_id: item for item in prepared_only}
    filtered_issue, filtered_pr, removed = _filtered_views(
        issue_view, pr_view, withheld, codec, repository=repository, pr_number=pr_number
    )
    lineage = resolve_transaction_lineage(filtered_pr, repository=repository, pr_number=pr_number)
    handoff = resolve_handoff_lineage(
        filtered_issue, lineage, repository=repository, issue_number=issue_number
    )
    resolve_pr_contract_lineage(filtered_pr, lineage, repository=repository, pr_number=pr_number)
    committed = lineage.latest_committed
    expected_parent = committed.transaction_id if committed is not None else None
    ids = sorted(withheld)
    for state in prepared_only:
        intent = state.intent
        if intent.scope != (repository.casefold(), issue_number, pr_number):
            raise _router_refusal(
                issue_number, pr_number,
                f"contradictory prepared record in comment "
                f"{state.prepared_comment.comment_id}: it belongs to another scope", ids=ids,
            )
        if intent.predecessor_transaction_id != expected_parent:
            raise _router_refusal(
                issue_number, pr_number,
                f"contradictory prepared record in comment "
                f"{state.prepared_comment.comment_id}: its predecessor is not the last "
                "committed transaction", ids=ids,
            )
        for prepared in (state.prepared_comment, *state.prepared_duplicates):
            if not _unedited(prepared):
                raise _router_refusal(
                    issue_number, pr_number,
                    f"contradictory prepared record in comment {prepared.comment_id}: edited",
                    ids=ids,
                )
    order_comment_id = handoff.comment_id if handoff is not None else None
    for comment, tx_id, name, record in removed:
        state = by_id[tx_id]
        problem = _removed_record_problem(
            comment, name, record, state, lineage=lineage, pr_view=pr_view, codec=codec
        )
        if problem is not None:
            raise _router_refusal(issue_number, pr_number, problem, ids=ids)
        if not _unedited(comment):
            raise _router_refusal(
                issue_number, pr_number,
                f"contradictory record in comment {comment.comment_id}: edited", ids=ids,
            )
        if not is_strictly_later(state.prepared_comment, comment):
            raise _router_refusal(
                issue_number, pr_number,
                f"unordered record in comment {comment.comment_id}: it is not later than its "
                f"prepared comment {state.prepared_comment.comment_id}", ids=ids,
            )
        if committed is None and comment.surface == issue_view.surface:
            order_comment_id = max(order_comment_id or 0, comment.comment_id)
    pending = tuple(sorted(prepared_only, key=lambda item: item.prepared_comment.comment_id))
    if committed is None:
        return (
            _Candidate(pr_number, "partial", pending=pending, order_comment_id=order_comment_id),
            None,
        )
    if handoff is None or handoff.handoff is None or handoff.transaction_id is None:
        raise _router_refusal(
            issue_number, pr_number, "missing committed handoff for the sibling group", ids=ids
        )
    return (
        _Candidate(
            pr_number, "committed-with-pending-successor", handoff, pending,
            order_comment_id=handoff.comment_id,
        ),
        _binding(handoff),
    )


def route_issue_publication(
    runner: Runner,
    config: AgentLoopConfig,
    issue_number: int,
    *,
    authorization_codec: AuthorizationEntryCodec | None = None,
) -> CanonicalRoute:
    """Routing form, for issue-command writer paths only.  Never writes or reconciles.

    A recoverable route carries no handoff view, no record, and no committed
    transaction, so it cannot reach any authority consumer.
    """
    issue_view, pr_views = _candidates(runner, config, issue_number)
    classified: list[_Candidate] = []
    bindings: dict[int, PredecessorBinding] = {}
    for number, pr_view in pr_views.items():
        foreign = actor_change_error(pr_view)
        if foreign is not None:
            raise foreign
        states = collect_transactions(pr_view, repository=config.repo, pr_number=number)
        prepared_only = [item for item in states if item.terminal is None]
        groups: dict[str | None, int] = {}
        for item in prepared_only:
            key = item.intent.predecessor_transaction_id
            groups[key] = groups.get(key, 0) + 1
        if any(count > 1 for count in groups.values()):
            if len(groups) > 1:
                raise _router_refusal(
                    issue_number, number,
                    "contradictory prepared records: root and successor siblings are mixed",
                    ids=[item.transaction_id for item in prepared_only],
                )
            candidate, binding = _route_sibling_group(
                issue_view, pr_view, prepared_only, repository=config.repo,
                issue_number=issue_number, pr_number=number, codec=authorization_codec,
            )
            if binding is not None:
                bindings[number] = binding
        else:
            # Tier 1: the full views and the full strict path.
            candidate = _classify_candidate(
                issue_view, pr_view, repository=config.repo,
                issue_number=issue_number, pr_number=number,
            )
            if candidate.kind == "committed-with-pending-successor":
                assert candidate.handoff is not None
                bindings[number] = _binding(candidate.handoff)
            elif candidate.kind == "partial" and candidate.error is None:
                handoff_ids = [
                    comment.comment_id
                    for comment, record in _v2_handoffs(issue_view)
                    if record.pr_number == number
                    and any(record.transaction_id == s.transaction_id for s in candidate.pending)
                ]
                candidate = dataclasses.replace(
                    candidate, order_comment_id=max(handoff_ids, default=None)
                )
        classified.append(candidate)
    partial = [item for item in classified if item.kind == "partial"]
    if len(partial) > 1:
        raise _partial_error(issue_number, partial[0].pr_number, "two partial candidates")
    for item in partial:
        if item.error is not None or not item.pending or item.order_comment_id is None:
            raise _partial_error(
                issue_number, item.pr_number, "it is not a routable interrupted publication",
                cause=item.error,
            )
        for state in item.pending:
            if state.intent.scope != (config.repo.casefold(), issue_number, item.pr_number):
                raise _partial_error(
                    issue_number, item.pr_number, "its stored intent belongs to another scope"
                )
    if not classified:
        return NoCandidate()
    winner = _select_winner(issue_number, classified)
    if winner is None:
        return NoCandidate()
    if partial and winner is not partial[0]:
        raise _partial_error(
            issue_number, partial[0].pr_number, "it is older than another candidate's handoff"
        )
    pending_ids = tuple(item.transaction_id for item in winner.pending)
    if winner.kind == "partial":
        return Recoverable(winner.pr_number, pending_ids)
    if winner.kind == "committed-with-pending-successor":
        for state in winner.pending:
            if state.intent.scope != (config.repo.casefold(), issue_number, winner.pr_number):
                raise _router_refusal(
                    issue_number, winner.pr_number,
                    "contradictory pending successor: another scope", ids=pending_ids,
                )
        return RecoverableSuccessor(winner.pr_number, pending_ids, bindings[winner.pr_number])
    assert winner.handoff is not None
    canonical = DiscoveredCanonicalPr(
        winner.pr_number, CanonicalHandoffView.of(winner.handoff), winner.handoff.transaction_id
    )
    return Committed(canonical) if winner.kind == "committed" else Legacy(canonical)
