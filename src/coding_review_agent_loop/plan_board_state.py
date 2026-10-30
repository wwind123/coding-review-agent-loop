"""Read-only plan board resolution shared by the issue preflight and plan startup (#1377).

:func:`resolve_plan_board_state` is the plan-surface twin of
:func:`~coding_review_agent_loop.pr_board_state.resolve_pr_board_state`.  It
resolves the unified plan board lineage (root, orchestrator-authored plan
board-change links and signed plan amendments) from the issue that owns the
plan, applies the #1133 original-flags rebind for a signed segment, derives
the board-change candidate when the configured plan board differs from the
effective one, and checks the reviewer floor against the lineage that would
exist after posting that candidate.  It never posts and never invokes an
agent, so every caller can refuse before any write.

It is only consulted for a plan that is not provably approved: an approved
plan is never reopened, and its PR board is governed by the stage 1 handoff
rule alone.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .board_amendment import collect_reviewer_board_amendments, require_amendment_activation
from .board_change import (
    PLAN_SURFACE,
    BoardChangeCandidate,
    BoardIdentity,
    BoardLineage,
    LineageElement,
    build_board_change_candidate,
    comments_bear_board_change_links,
    comments_with_candidate,
    configured_differs,
    plan_identity_from_config,
    policy_differs,
    resolve_board_lineage,
)
from .errors import AgentLoopError
from .pr_board_state import _rebind_signed_segment, check_lineage_floor
from .round_state import _extract_round_metadata_records
from .plan_round_closure import plan_amendment_activation_round

# Placeholder actor used only to validate an unposted candidate when the issue
# has no link records yet, so no authenticated actor is needed.
_CANDIDATE_ACTOR = ("orchestrator-candidate", 0)


@dataclass(frozen=True)
class PlanBoardState:
    """The resolved plan board for one invocation."""

    lineage: BoardLineage
    config: object
    configured: BoardIdentity
    candidate: BoardChangeCandidate | None
    comments: tuple[object, ...]
    trusted_actor: tuple[str, int] | None
    rebound: bool

    @property
    def activation(self) -> LineageElement | None:
        """The pending link this run activates (a recovered run's last, or the candidate)."""
        return self.lineage.pending_links[-1] if self.lineage.pending_links else None

    @property
    def predecessor_board(self) -> BoardIdentity | None:
        """The board that governed the anchor-round records of a pending activation."""
        if not self.lineage.pending_links:
            return None
        first = self.lineage.pending_links[0].link
        assert first is not None
        return first.old_board


def admitted_issue_comments(
    runner: object,
    *,
    config: object,
    issue_number: int,
    comments: Sequence[object],
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Issue comments with REST author identities, plus the actor, when plan links exist.

    Plan board-change links and planning-scope boundary records (#1390) are
    admitted only from the authenticated orchestrator actor, so an issue
    carrying either record needs both.  An incomplete REST read fails closed
    rather than reading as an empty history.  Without such records nothing is
    read and no actor is needed.
    """
    from .github import (
        _merge_transport_comments,
        read_rest_issue_comments,
        resolve_authenticated_github_actor,
    )
    from .plan_issue_scope import comments_bear_scope_boundary
    from .pr_loop_support import _cached_trusted_actor

    comments = tuple(comments)
    links = comments_bear_board_change_links(comments, surface=PLAN_SURFACE)
    if not links and not comments_bear_scope_boundary(comments):
        return comments, None
    records = (
        "plan reviewer-board change records" if links else "planning-scope boundary records"
    )
    if runner is None:
        raise AgentLoopError(
            f"Issue #{issue_number} carries {records}, but no "
            "authenticated GitHub read is available to admit them."
        )
    if getattr(config, "dry_run", False):
        actor = _cached_trusted_actor(runner)
        if actor is None:
            raise AgentLoopError(
                f"Issue #{issue_number} carries {records}, but the "
                "authenticated GitHub actor is unavailable in dry-run mode, so they cannot be "
                "admitted."
            )
        return comments, actor
    if all(getattr(comment, "author_id", None) is not None for comment in comments):
        merged = comments
    else:
        transport = read_rest_issue_comments(
            runner,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            issue_number=issue_number,
            purpose=f"{records} cannot be authenticated",
            reject_empty_output=True,
        )
        merged = _merge_transport_comments(
            comments, transport, issue_number=issue_number  # type: ignore[arg-type]
        )
    return tuple(merged), resolve_authenticated_github_actor(runner, config=config)  # type: ignore[arg-type]


def plan_drift_error(issue_number: int) -> Callable[[object, str], AgentLoopError]:
    """A read-only planning contract drift error for preflight and verification callers."""

    def error(persisted: object, detail: str) -> AgentLoopError:
        return AgentLoopError(
            "Plan review scheduler contract changed during resume; the planning policy "
            f"must remain immutable for the run. Issue #{issue_number} already carries a "
            f"planning scheduler contract for policy {getattr(persisted, 'policy', '?')} with "
            f"primary {getattr(persisted, 'primary_reviewer', None) or '(none)'} and reviewer "
            f"board {', '.join(getattr(persisted, 'required_reviewers', ()))} ({detail}). "
            "Rerun with the persisted planning policy."
        )

    return error


def resolve_plan_lineage(
    comments: Sequence[object],
    *,
    issue_number: int,
    configured: BoardIdentity,
    trusted_actor: tuple[str, int] | None,
    drift_error: Callable[[object, str], AgentLoopError],
    diagnostics: list[str] | None = None,
) -> BoardLineage:
    """Resolve the plan board lineage of ``comments`` read-only."""
    from .panel_evidence import _plan_contract_or_none

    records = _extract_round_metadata_records(comments, flow="plan")
    amendments = collect_reviewer_board_amendments(
        comments, flow="plan", issue_number=issue_number,
        ignored_sink=diagnostics if diagnostics is not None else [],
    )
    return resolve_board_lineage(
        records,
        comments,
        amendments=amendments,
        configured=configured,
        pr_number=issue_number,
        trusted_actor=trusted_actor,
        contract_from_metadata=_plan_contract_or_none,
        drift_error=drift_error,
        surface=PLAN_SURFACE,
    )


def resolve_plan_board_state(
    runner: object,
    *,
    config: object,
    issue_number: int,
    comments: Sequence[object],
    drift_error: Callable[[object, str], AgentLoopError],
    floor_context: str | None = None,
    resume_round: Callable[[object, BoardLineage], int] | None = None,
    floor_refusal_hint: Callable[[object, object], str] | None = None,
    diagnostics: list[str] | None = None,
    admitted: tuple[Sequence[object], tuple[str, int] | None] | None = None,
) -> PlanBoardState:
    """Resolve the plan board read-only and refuse a below-floor board (#1377 steps 3, 9).

    ``resume_round(effective_config, lineage)`` returns the round a resume
    re-enters; it is consulted only to validate a pending signed amendment.
    ``admitted`` supplies comments and actor already admitted by the caller.
    Changing the planning policy against the historical root stays contract
    drift.
    """
    from .panel_evidence import _plan_contract_or_none

    if admitted is not None:
        comments, trusted_actor = tuple(admitted[0]), admitted[1]
    else:
        comments, trusted_actor = admitted_issue_comments(
            runner, config=config, issue_number=issue_number, comments=comments
        )
    records = _extract_round_metadata_records(comments, flow="plan")
    amendments = collect_reviewer_board_amendments(
        comments, flow="plan", issue_number=issue_number,
        ignored_sink=diagnostics if diagnostics is not None else [],
    )
    configured = plan_identity_from_config(config)

    def resolve(source: Sequence[object], actor: tuple[str, int] | None) -> BoardLineage:
        return resolve_board_lineage(
            records,
            source,
            amendments=amendments,
            configured=configured,
            pr_number=issue_number,
            trusted_actor=actor,
            contract_from_metadata=_plan_contract_or_none,
            drift_error=drift_error,
            surface=PLAN_SURFACE,
        )

    lineage = resolve(comments, trusted_actor)
    if policy_differs(configured, lineage.root):
        # Changing the planning policy stays contract drift, never a link.
        raise drift_error(lineage.current_contract, "the configured contract does not match it")
    effective_config = config
    rebound = False
    rebind = _rebind_signed_segment(config, lineage, configured)
    if rebind is not None:
        effective_config, rebound = rebind, True
    effective_identity = plan_identity_from_config(effective_config)
    candidate = None
    previous_contract = lineage.current_contract
    if configured_differs(effective_identity, lineage.current_identity):
        candidate = build_board_change_candidate(
            lineage, configured=effective_identity, records=records, pr_number=issue_number
        )
        actor = trusted_actor or _CANDIDATE_ACTOR
        lineage = resolve(
            comments_with_candidate(comments, candidate, actor_id=actor[1], surface=PLAN_SURFACE),
            actor,
        )
    elif lineage.pending_amendments and not lineage.pending_links and resume_round is not None:
        from .panel_evidence import _board_amendment_template
        from .plan_issue_scope import resolve_plan_scope

        start_round = resume_round(effective_config, lineage)
        scope_start = resolve_plan_scope(comments, trusted_actor)
        activation_records = tuple(
            record for record in records if record.metadata.round_number >= scope_start
        )
        require_amendment_activation(
            lineage,  # type: ignore[arg-type]
            start_round_number=plan_amendment_activation_round(activation_records, lineage, start_round),
            template=lambda amendment, round_number: _board_amendment_template(
                flow="plan",
                issue_number=issue_number,
                pr_number=None,
                persisted=lineage.contract_before(amendment),
                removed=amendment.removed_reviewers,
                restored=amendment.restored_reviewers,
                start_round_number=round_number,
            ),
        )
    try:
        check_lineage_floor(
            config,
            lineage,
            context=floor_context or f"Issue #{issue_number} plan reviewer board",
        )
    except AgentLoopError as exc:
        if candidate is None or floor_refusal_hint is None:
            raise
        # A refused plan board change posts nothing; a removal can still be
        # authorized by a signed plan amendment, whose template is printed here.
        raise AgentLoopError(
            str(exc)
            + floor_refusal_hint(
                previous_contract, PLAN_SURFACE.contract(effective_identity)
            )
        ) from exc
    return PlanBoardState(
        lineage=lineage,
        config=effective_config,
        configured=effective_identity,
        candidate=candidate,
        comments=comments,
        trusted_actor=trusted_actor,
        rebound=rebound,
    )


def plan_floor_refusal_hint(
    issue_number: int, start_round_number: Callable[[], int | None]
) -> Callable[[object, object], str]:
    """The filled-in signed plan amendment template appended to a floor refusal."""
    from .panel_evidence import _board_amendment_route_clause

    def hint(persisted: object, refused: object) -> str:
        return _board_amendment_route_clause(
            flow="plan",
            issue_number=issue_number,
            pr_number=None,
            persisted=persisted,
            configured=refused,
            start_round_number=start_round_number,
            amendments_recognized=True,
        )

    return hint


def plan_resume_round(
    comments: Sequence[object], config: object, *, min_round_number: int | None = None,
    trusted_actor: tuple[str, int] | None = None,
) -> int:
    """The round a plan resume of ``comments`` re-enters under ``config``'s board.

    ``min_round_number`` is the planning scope start (#1390); when omitted it
    is derived from the admitted scope-boundary record.  A scope with no plan
    round of its own re-enters at its start round.
    """
    from .board_change import board_reentry_round
    from .config import reviewers
    from .plan_issue_scope import resolve_plan_scope
    from .round_state import _resume_plan_round

    if min_round_number is None:
        min_round_number = resolve_plan_scope(comments, trusted_actor)
    resumed = _resume_plan_round(
        comments, configured_reviewers=reviewers(config), min_round_number=min_round_number,
    )
    reentry = board_reentry_round(
        comments,
        resumed[1].round_number if resumed is not None else None,
        signed_amendment=True,
        surface=PLAN_SURFACE,
    )
    return reentry if resumed is not None else max(reentry, min_round_number)
