"""Effective reviewer-board floor (#1373).

``--min-reviewers`` and ``--min-distinct-providers`` are per-invocation
guards.  Configuration only validates the option values; the board is checked
here, on the *effective* board after read-only history and signed amendment
lineage are resolved, and before any agent invocation or comment post.  A
signed reviewer-board amendment is the only thing that can authorize a board
below the floor, and only for a measure that its chain actually lowered.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from .errors import AgentLoopError


@dataclass(frozen=True)
class BoardSeat:
    """One reviewer on a board, with the binding that decides its provider.

    ``model_chain`` and ``effort`` are ``None`` when history does not record
    them (a plain plan board without a seat binding).
    """

    seat_id: str
    backend: str
    model_chain: tuple[str | None, ...] | None = None
    effort: str | None = None

    @property
    def binding_known(self) -> bool:
        return self.model_chain is not None


@dataclass(frozen=True)
class SignedBoardAuthorization:
    """A board reached through a validated, non-empty signed amendment chain.

    ``seats`` carry the *recorded* bindings of that board, never the current
    invocation's, so a binding change cannot inherit the signed exception.
    """

    seats: tuple[BoardSeat, ...]
    lowered_reviewers: bool
    lowered_providers: bool


def distinct_providers(seats: Sequence[BoardSeat]) -> int:
    """Providers are counted by backend: two seats on one backend count once."""
    return len({seat.backend for seat in seats})


def board_floor_enabled(config: object) -> bool:
    return (
        getattr(config, "min_reviewers", None) is not None
        or getattr(config, "min_distinct_providers", None) is not None
    )


def required_floor(
    *,
    min_reviewers: int | None,
    min_distinct_providers: int | None,
    authorizations: Sequence[SignedBoardAuthorization] = (),
) -> tuple[int, int]:
    """Return the (reviewers, providers) minimum after signed exceptions.

    Each measure is relaxed only by an authorization whose chain lowered that
    measure, and never below that authorized board's own count.
    """
    reviewers = min_reviewers or 0
    providers = min_distinct_providers or 0
    for authorization in authorizations:
        if authorization.lowered_reviewers:
            reviewers = min(reviewers, len(authorization.seats))
        if authorization.lowered_providers:
            providers = min(providers, distinct_providers(authorization.seats))
    return reviewers, providers


def check_reviewer_floor(
    seats: Sequence[BoardSeat],
    *,
    min_reviewers: int | None,
    min_distinct_providers: int | None,
    authorizations: Sequence[SignedBoardAuthorization] = (),
    context: str = "Reviewer board",
) -> None:
    """Refuse a board below the configured floor.

    The single floor check shared by every entry point.  It performs no I/O,
    so callers run it before any agent invocation or comment post.
    """
    if min_reviewers is None and min_distinct_providers is None:
        return
    required_reviewers, required_providers = required_floor(
        min_reviewers=min_reviewers,
        min_distinct_providers=min_distinct_providers,
        authorizations=authorizations,
    )
    board = ", ".join(f"{seat.seat_id} ({seat.backend})" for seat in seats) or "(empty)"
    problems: list[str] = []
    if len(seats) < required_reviewers:
        problems.append(
            f"{len(seats)} reviewer(s) is below the floor of {required_reviewers}"
        )
    providers = distinct_providers(seats)
    if providers < required_providers:
        problems.append(
            f"{providers} distinct provider(s) is below the floor of {required_providers}"
        )
    if problems:
        raise AgentLoopError(
            f"{context} {board}: {'; '.join(problems)}. No agent was invoked and no comment "
            "was posted. Configure a larger board, or post a signed reviewer-board "
            "amendment to authorize a smaller one."
        )


def board_seats_from_config(config: object) -> tuple[BoardSeat, ...]:
    """The configured board with each reviewer's current backend and models."""
    from .reviewer_seats import board_binding_entries

    return tuple(
        BoardSeat(
            seat_id=entry["id"],
            backend=entry["backend"],
            model_chain=tuple(entry["model_chain"]),
            effort=entry["effort"],
        )
        for entry in board_binding_entries(config)
    )


def enforce_configured_board_floor(
    config: object,
    *,
    authorizations: Sequence[SignedBoardAuthorization] = (),
    context: str = "Reviewer board",
) -> None:
    if not board_floor_enabled(config):
        return
    check_reviewer_floor(
        board_seats_from_config(config),
        min_reviewers=getattr(config, "min_reviewers", None),
        min_distinct_providers=getattr(config, "min_distinct_providers", None),
        authorizations=authorizations,
        context=context,
    )
