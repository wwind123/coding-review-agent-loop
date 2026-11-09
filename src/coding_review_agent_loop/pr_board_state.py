"""Read-only PR board resolution shared by the entry preflight and startup (#1376).

:func:`resolve_pr_board_state` resolves the unified PR board lineage from PR
comments, applies the #1133 original-flags rebind for a signed segment,
derives the board-change candidate when the configured board differs from the
effective board, and checks the reviewer floor against the lineage that would
exist after posting that candidate.  It never posts and never invokes an
agent, so every caller can refuse before any write.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from dataclasses import replace as dataclasses_replace

from .agents.registry import agent_display_name
from .board_amendment import collect_reviewer_board_amendments, require_amendment_activation
from .board_change import (
    BoardChangeCandidate,
    BoardIdentity,
    BoardLineage,
    LineageElement,
    binding_compatible,
    build_board_change_candidate,
    comments_bear_board_change_links,
    comments_with_candidate,
    configured_differs,
    identity_from_config,
    lineage_floor_authorizations,
    policy_differs,
    resolve_board_lineage,
)
from .config import reviewers
from .errors import AgentLoopError
from .reviewer_floor import board_floor_enabled, check_reviewer_floor
from .round_state import _extract_round_metadata_records

# Placeholder actor used only to validate an unposted candidate when the PR
# has no link records yet, so no authenticated actor is needed.
_CANDIDATE_ACTOR = ("orchestrator-candidate", 0)


@dataclass(frozen=True)
class PrBoardState:
    """The resolved PR board for one invocation."""

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


def admitted_pr_comments(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Comments admitted before round or reviewer-board interpretation.

    Board-change links are admitted only from the authenticated orchestrator
    actor, so a PR carrying any link record needs both; an incomplete read
    fails closed rather than reading as an empty history.
    """
    from .pr_round_ingestion import admit_pr_comments
    return admit_pr_comments(runner, config, pr_number=pr_number, comments=comments)


def _rebind_signed_segment(config: object, lineage: BoardLineage, configured: BoardIdentity) -> object | None:
    """The #1133 rebind: original flags of a signed segment select its amended board.

    Applies when the latest element is a signed amendment and the configured
    board is the segment's base board (the root, or the latest link's new
    board) with compatible bindings.  Returns ``None`` when it does not apply.
    """
    from .reviewer_seats import reviewer_seat_binding

    latest = lineage.latest
    if latest is None or latest.kind != "amendment":
        return None
    base = lineage.segment_base
    from .board_change import apply_inits

    if not binding_compatible(apply_inits(base, lineage.inits), configured):
        return None
    if lineage.current_board == configured.names:
        # A restoration back to the configured board: nothing to rebind.
        return None
    names = set(lineage.current_board)
    return dataclasses_replace(
        config,
        reviewer=tuple(r for r in reviewers(config) if agent_display_name(r) in names),
        pr_seat_binding_override=reviewer_seat_binding(config),
    )


def resolve_pr_board_state(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
    drift_error: Callable[[object, str], AgentLoopError],
    handoff_authorizations: Sequence[object] = (),
    floor_context: str | None = None,
    resume_round: Callable[[object, BoardLineage], int] | None = None,
    floor_refusal_hint: Callable[[object, object], str] | None = None,
    diagnostics: list[str] | None = None,
    activation_template: Callable[[object, BoardLineage, int], str | None] | None = None,
) -> PrBoardState:
    """Resolve the PR board read-only and refuse a below-floor board (D4).

    ``resume_round(effective_config, lineage)`` returns the round a resume
    re-enters; it is consulted only to validate a pending signed amendment.
    ``activation_template(amendment, lineage, round)`` replaces the default
    replacement template; returning None marks the round unsafe (#1981).
    """
    from .pr_loop_support import _scheduler_contract_from_metadata
    from .panel_evidence import _board_amendment_template

    comments, trusted_actor = admitted_pr_comments(
        runner, config=config, pr_number=pr_number, comments=comments
    )
    records = _extract_round_metadata_records(comments, flow="pr")
    amendments = collect_reviewer_board_amendments(
        comments, flow="pr", pr_number=pr_number,
        ignored_sink=diagnostics if diagnostics is not None else [],
    )
    configured = identity_from_config(config)

    def resolve(source: Sequence[object], actor: tuple[str, int] | None) -> BoardLineage:
        return resolve_board_lineage(
            records,
            source,
            amendments=amendments,
            configured=configured,
            pr_number=pr_number,
            trusted_actor=actor,
            contract_from_metadata=_scheduler_contract_from_metadata,
            drift_error=drift_error,
        )

    lineage = resolve(comments, trusted_actor)
    if policy_differs(configured, lineage.root):
        # Changing the review policy or broad-path rules stays contract drift.
        raise drift_error(lineage.current_contract, "the configured contract does not match it")
    effective_config = config
    rebound = False
    rebind = _rebind_signed_segment(config, lineage, configured)
    if rebind is not None:
        effective_config, rebound = rebind, True
    effective_identity = identity_from_config(effective_config)
    candidate = None
    previous_contract = lineage.current_contract
    if configured_differs(effective_identity, lineage.current_identity):
        candidate = build_board_change_candidate(
            lineage, configured=effective_identity, records=records, pr_number=pr_number
        )
        actor = trusted_actor or _CANDIDATE_ACTOR
        lineage = resolve(
            comments_with_candidate(comments, candidate, actor_id=actor[1]), actor
        )
    elif lineage.pending_amendments and not lineage.pending_links and resume_round is not None:
        start_round = resume_round(effective_config, lineage)
        require_amendment_activation(
            lineage,  # type: ignore[arg-type]
            start_round_number=start_round,
            template=lambda amendment, round_number: (
                activation_template(amendment, lineage, round_number)
                if activation_template is not None
                else _board_amendment_template(
                    flow="pr",
                    issue_number=None,
                    pr_number=pr_number,
                    persisted=lineage.contract_before(amendment),
                    removed=amendment.removed_reviewers,
                    restored=amendment.restored_reviewers,
                    start_round_number=round_number,
                )
            ),
        )
    try:
        check_lineage_floor(
            config,
            lineage,
            context=floor_context or f"PR #{pr_number} reviewer board",
            handoff_authorizations=handoff_authorizations,
        )
    except AgentLoopError as exc:
        if candidate is None or floor_refusal_hint is None:
            raise
        # A refused board change posts nothing; a removal can still be
        # authorized by a signed amendment, whose template is printed here.
        raise AgentLoopError(
            str(exc) + floor_refusal_hint(previous_contract, effective_identity.contract())
        ) from exc
    return PrBoardState(
        lineage=lineage,
        config=effective_config,
        configured=effective_identity,
        candidate=candidate,
        comments=comments,
        trusted_actor=trusted_actor,
        rebound=rebound,
    )


def check_lineage_floor(
    config: object, lineage: BoardLineage, *, context: str,
    handoff_authorizations: Sequence[object] = (),
) -> None:
    """The per-round and qualification floor check (pure, no I/O)."""
    if not board_floor_enabled(config):
        return
    check_reviewer_floor(
        lineage.current_identity.seats,
        min_reviewers=getattr(config, "min_reviewers", None),
        min_distinct_providers=getattr(config, "min_distinct_providers", None),
        authorizations=(*handoff_authorizations, *lineage_floor_authorizations(lineage)),
        context=context,
    )


def qualification_board_check(
    runner: object,
    *,
    config: object,
    pr_number: int,
    expected: BoardIdentity,
    handoff_authorizations: Sequence[object] = (),
) -> Callable[[Sequence[object]], None]:
    """The qualification gate's board check over the same fresh PR comment fetch.

    Re-resolves the unified lineage for every policy, requires the run's
    board to equal the effective board, and rechecks the floor.  Any failure
    permits no qualification or merge.
    """
    from .logging import log
    from .pr_loop_support import _pr_contract_drift_error

    def check(comments: Sequence[object]) -> None:
        diagnostics: list[str] = []
        try:
            state = resolve_pr_board_state(
                runner,
                config=config,
                pr_number=pr_number,
                comments=comments,
                drift_error=lambda persisted, detail: _pr_contract_drift_error(
                    persisted,
                    detail,
                    pr_number=pr_number,
                    configured=None,
                    start_round_number=lambda: None,
                    during="qualification",
                ),
                handoff_authorizations=handoff_authorizations,
                floor_context=f"PR #{pr_number} qualification reviewer board",
                diagnostics=diagnostics,
            )
            # The run's config already reflects every element it activated, so
            # any difference is a change made after the run read the board: a
            # new board, a signed segment that rebinds the run's board, or a
            # signed amendment still waiting to take effect.  A pending link
            # whose board is the run's own board (a removal that needed no new
            # record) is the board the run reviewed with.
            if state.candidate is not None or state.rebound or state.lineage.pending_amendments:
                raise AgentLoopError(
                    "PR reviewer board changed during qualification: the effective board "
                    f"is {state.lineage.describe()}, but this run reviewed "
                    + ", ".join(expected.names)
                )
        except AgentLoopError as exc:
            if "no qualification or merge is permitted" in str(exc):
                raise
            raise AgentLoopError(f"{exc} No qualification or merge is permitted.") from exc
        finally:
            for diagnostic in diagnostics:
                log(config, f"PR #{pr_number}: {diagnostic}")

    return check


# Roles whose unanchored records seed a derived legacy root and are adopted as
# round history (#1365).
_LEGACY_ENTRY_ROLES = frozenset({"coder", "reviewer", "summary"})


def named_legacy_entry_required(config: object, comments: Sequence[object]) -> bool:
    """Whether a named board would adopt unanchored legacy PR history (#1365).

    True only when named seats are configured and the PR carries PR-flow
    coder, reviewer or summary records but no root anchor and no board-change
    link, so the lineage root would come from :func:`derive_legacy_root`.
    Plain boards, anchored history and empty history need no check.
    """
    from .reviewer_seats import reviewer_seat_binding

    if reviewer_seat_binding(config) is None:
        return False
    records = _extract_round_metadata_records(comments, flow="pr")
    if not records:
        return False
    if any(record.metadata.pr_board_root is not None for record in records):
        return False
    if comments_bear_board_change_links(comments):
        return False
    return any(record.metadata.role in _LEGACY_ENTRY_ROLES for record in records)


@dataclass(frozen=True)
class PrOwningIssue:
    """The issue that owns a PR's handoff for one invocation (#1418)."""

    owning_issue_number: int | None
    fresh_scope_number: int | None
    branch_issue_number: int | None
    ordinary_managed_recovery: bool


def resolve_pr_owning_issue(
    config: object,
    *,
    caller_issue_number: int | None,
    managed_ci_issue_number: int | None,
    head_branch: str | None,
    recorded_primary_issue_number: int | None | Callable[[], int | None],
    pr_body: str | None,
) -> PrOwningIssue:
    """Resolve the issue that owns a PR before its first write (#1418).

    The order is: the caller's issue; else the explicit managed-CI issue
    scope (the argument, else ``config.managed_ci_issue_number``); else the
    recorded PR contract's primary issue; else, only during ordinary
    managed-CI recovery (managed-CI PR mode or a fresh scope), the issue named
    by an ``agent-loop/managed-N`` head branch; else the single issue the PR
    body links.  ``recorded_primary_issue_number`` may be a callable so a
    caller reads the contract only when the earlier lookups resolve nothing.
    Performs no I/O.
    """
    from .github import parse_linked_issue_numbers
    from .managed_ci_authorization import (
        managed_branch_issue_number,
    )

    fresh_scope_number = managed_ci_issue_number or getattr(
        config, "managed_ci_issue_number", None
    )
    branch_number = managed_branch_issue_number(head_branch)
    ordinary_managed_recovery = bool(
        getattr(config, "managed_ci_pr_mode", False) or fresh_scope_number is not None
    )
    owning_number = caller_issue_number
    if owning_number is None:
        owning_number = fresh_scope_number
    if owning_number is None:
        owning_number = (
            recorded_primary_issue_number()
            if callable(recorded_primary_issue_number)
            else recorded_primary_issue_number
        )
    if owning_number is None and ordinary_managed_recovery:
        owning_number = branch_number
    if owning_number is None:
        linked = parse_linked_issue_numbers(pr_body, repo=getattr(config, "repo"))
        if len(linked) == 1:
            owning_number = linked[0]
    return PrOwningIssue(
        owning_issue_number=owning_number,
        fresh_scope_number=fresh_scope_number,
        branch_issue_number=branch_number,
        ordinary_managed_recovery=ordinary_managed_recovery,
    )


def authenticate_named_legacy_pr_entry(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
    owning_issue_number: int | None,
    pr_body: str | None = None,
    managed_ci_issue_number: int | None = None,
    head_branch: str | None = None,
) -> None:
    """Refuse unanchored named-board PR history not written by the actor (#1365).

    Before the PR loop adopts unanchored legacy round records under a named
    board, every coder, reviewer or summary record and the owning issue's
    canonical handoff record must be authored by the authenticated actor.
    The owning issue comes from :func:`resolve_pr_owning_issue`, the same
    resolver ``pr_loop._entry_plan_handoff_board`` uses: the caller's issue,
    else the explicit managed-CI issue scope, else the PR contract's primary
    issue, else the ``agent-loop/managed-N`` branch issue during managed-CI
    recovery, else the single issue the PR body links (#1418).  Reads only;
    any failure raises before a PR write or agent invocation.
    """
    from .github import read_authenticated_protocol_comments
    from .issue_pr_handoff import resolve_issue_pr_handoff_lineage
    from .pr_contract import find_latest_pr_contract
    from .pr_loop_support import _cached_trusted_actor

    if not named_legacy_entry_required(config, comments):
        return
    repo = getattr(config, "repo")
    if getattr(config, "dry_run", False) and _cached_trusted_actor(runner) is None:  # type: ignore[arg-type]
        raise AgentLoopError(
            f"PR #{pr_number} has unanchored round history under a named reviewer board, but the "
            "authenticated GitHub actor is unavailable in dry-run mode, so it cannot be admitted."
        )

    def refuse(detail: str) -> AgentLoopError:
        return AgentLoopError(
            f"Human decision required: PR #{pr_number} has unanchored round history under a "
            f"named reviewer board, and {detail}. Nothing was posted. Only records written by "
            "the authenticated orchestrator actor can seed a named board."
        )

    pr_view = read_authenticated_protocol_comments(
        runner, config=config, surface_kind="pr", number=pr_number  # type: ignore[arg-type]
    )
    pr_full = pr_view.authored
    for record in _extract_round_metadata_records(pr_full, flow="pr"):
        if record.metadata.role not in _LEGACY_ENTRY_ROLES:
            continue
        comment = pr_full[record.index]
        if comment.author_id != pr_view.actor_id:
            raise refuse(
                f"the {record.metadata.role} round record in comment {comment.comment_id} was "
                f"written by {comment.author_login}, not {pr_view.actor_login}"
            )
    recorded = find_latest_pr_contract(pr_full, repository=repo, pr_number=pr_number)
    if recorded != find_latest_pr_contract(pr_view.authored, repository=repo, pr_number=pr_number):
        raise refuse("its PR contract record was not written by the authenticated actor")
    owning_issue_number = resolve_pr_owning_issue(
        config,
        caller_issue_number=owning_issue_number,
        managed_ci_issue_number=managed_ci_issue_number,
        head_branch=head_branch,
        recorded_primary_issue_number=(
            recorded.primary_issue_number if recorded is not None else None
        ),
        pr_body=pr_body,
    ).owning_issue_number
    if owning_issue_number is None:
        # Task-origin PRs have no issue handoff record.
        return
    issue_view = read_authenticated_protocol_comments(
        runner, config=config, surface_kind="issue", number=owning_issue_number  # type: ignore[arg-type]
    )
    issue_full = sorted(
        (*issue_view.authored, *issue_view.ignored_foreign), key=lambda c: c.comment_id
    )
    unfiltered = resolve_issue_pr_handoff_lineage(
        issue_full, issue_number=owning_issue_number, repo=repo
    )
    authored = resolve_issue_pr_handoff_lineage(
        issue_view.authored, issue_number=owning_issue_number, repo=repo
    )
    names_pr = any(
        lineage is not None and lineage.latest.pr_number == pr_number
        for lineage in (unfiltered, authored)
    )
    if not names_pr:
        # No handoff names this PR, so no plan association is adopted.
        return
    if (
        unfiltered is None
        or authored is None
        or unfiltered.latest != authored.latest
        or issue_full[unfiltered.latest_comment_index].comment_id
        != issue_view.authored[authored.latest_comment_index].comment_id
    ):
        raise refuse(
            f"the canonical issue-to-PR handoff record on issue #{owning_issue_number} is not "
            "the latest one written by the authenticated actor"
        )
