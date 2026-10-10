"""Plan verification context and the issue-to-PR board-change audit (#1373).

An approved plan is verified against the board that approved it, never the PR
invocation's board.  ``derive_plan_verification_context`` reads that board
from the plan's owning issue: its persisted plan contract and signed plan
amendment lineage (C0 -> Cn), its seat binding, and its planning policy and
primary.  Every plan-approval consumer (managed-CI fresh authorization,
ordinary resume without an issue-side handoff, the handoff seam, and strict
managed-CI qualification) uses it.

The PR board is a per-invocation choice.  When it differs from the plan's
effective board, the orchestrator posts one plain-text board-change audit
record on the PR, identified by a digest so a rerun posts nothing new.  The
record carries no protocol marker and no signature grammar: it is neither a
signed human requirement nor a signed reviewer-board amendment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Sequence

from .errors import AgentLoopError
from .reviewer_floor import BoardSeat, SignedBoardAuthorization, distinct_providers

BOARD_CHANGE_AUDIT_HEADING = "Reviewer board change recorded at issue-to-PR handoff."
BOARD_CHANGE_KIND = "reviewer-board-change"
BOARD_CHANGE_REASON = "operator-reconfigured"
BOARD_CHANGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlanVerificationContext:
    """The approving plan board, derived from the plan's owning issue only."""

    issue_number: int | None
    original_reviewers: tuple[str, ...]
    effective_reviewers: tuple[str, ...]
    original_seats: tuple[BoardSeat, ...]
    effective_seats: tuple[BoardSeat, ...]
    policy: str
    primary_reviewer: str | None
    seat_binding: dict | None
    signed_amendment_count: int

    @property
    def signed_authorization(self) -> SignedBoardAuthorization | None:
        """The signed board, only when a validated signed link exists.

        An unamended plan (C0 equal to Cn) never supplies a floor exception.
        Each measure is excepted only where the chain lowered it.
        """
        if self.signed_amendment_count < 1:
            return None
        return SignedBoardAuthorization(
            seats=self.effective_seats,
            lowered_reviewers=len(self.effective_seats) < len(self.original_seats),
            lowered_providers=(
                distinct_providers(self.effective_seats)
                < distinct_providers(self.original_seats)
            ),
        )

    def reviewer_agents(self) -> tuple[object, ...]:
        from .agents.registry import agent_display_name
        from .reviewer_seats import ReviewerSeat, SeatAgent

        agents: list[object] = []
        for seat in self.effective_seats:
            if seat.seat_id == agent_display_name(seat.backend):
                agents.append(seat.backend)
            else:
                agents.append(SeatAgent(
                    ReviewerSeat(
                        seat.seat_id, seat.backend,
                        tuple(seat.model_chain or ()), seat.effort,
                    ),
                    Path("."),
                ))
        return tuple(agents)

    def config_for(self, config: object) -> object:
        """``config`` re-pointed at the approving plan board for verification.

        Only the plan-approval readers consume the result; the PR invocation
        board stays the PR review board.
        """
        from .agents.registry import agent_display_name
        from .config import (
            DEFAULT_PLAN_PRIMARY_STALL_ROUNDS,
            DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS,
            DEFAULT_PLAN_STEP_BACK_ROUNDS,
        )
        from .reviewer_seats import SeatAgent

        agents = self.reviewer_agents()
        primary = None
        if self.primary_reviewer is not None:
            primary = next(
                (agent for agent in agents if agent_display_name(agent) == self.primary_reviewer),
                None,
            )
            if primary is None:
                raise AgentLoopError(
                    "Approved plan primary reviewer is not on the approving plan board."
                )
        staged = self.policy == "primary-then-panel"
        return replace(
            config,
            reviewer=agents,
            reviewer_seats=tuple(agent for agent in agents if isinstance(agent, SeatAgent)),
            active_reviewer_seat_id=None,
            pr_seat_binding_override=self.seat_binding,
            plan_review_policy=self.policy,
            primary_plan_reviewer=primary,
            plan_review_force_full=False,
            plan_reset_stall_streak=False,
            plan_primary_stall_rounds=(
                config.plan_primary_stall_rounds if staged else DEFAULT_PLAN_PRIMARY_STALL_ROUNDS
            ),
            plan_step_back_rounds=(
                config.plan_step_back_rounds if staged else DEFAULT_PLAN_STEP_BACK_ROUNDS
            ),
            plan_step_back_escalation_rounds=(
                config.plan_step_back_escalation_rounds
                if staged else DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS
            ),
            pr_review_policy="all-reviewers",
            primary_reviewer=None,
            pr_review_force_full=False,
        )


def _plain_backends() -> dict[str, str]:
    from .agents.registry import agent_display_name
    from .reviewer_seats import _BACKENDS

    return {agent_display_name(backend): backend for backend in sorted(_BACKENDS)}


def _board_seats(
    names: Sequence[str], binding_entries: dict[str, dict],
) -> tuple[BoardSeat, ...]:
    plain = _plain_backends()
    seats: list[BoardSeat] = []
    for name in names:
        entry = binding_entries.get(name)
        if entry is not None:
            seats.append(BoardSeat(
                seat_id=name,
                backend=entry["backend"],
                model_chain=tuple(entry["model_chain"]),
                effort=entry.get("effort"),
            ))
        elif name in plain and not binding_entries:
            # A plain plan board records no models; only membership and the
            # implied backend are verifiable.
            seats.append(BoardSeat(seat_id=name, backend=plain[name]))
        else:
            raise AgentLoopError(
                f"Approved plan reviewer {name!r} has no verifiable backend binding."
            )
    return tuple(seats)


def derive_plan_verification_context(
    comments: Sequence[object], *, issue_number: int | None,
) -> PlanVerificationContext | None:
    """Derive the approving plan board from the plan's owning issue history.

    Returns ``None`` when the issue carries no plan round records (or no plan
    reviewer at all), in which case no plan board can be verified.  Malformed
    binding or amendment lineage fails closed.
    """
    from .board_amendment import collect_reviewer_board_amendments, resolve_contract_lineage
    from .reviewer_seats import (
        _persisted_plan_contract,
        validate_backend_outage_amendments_for_binding,
        validated_binding_entries,
    )
    from .round_state import _extract_round_metadata_records

    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return None
    binding = next(
        (record.metadata.seat_binding for record in reversed(records)
         if record.metadata.seat_binding is not None),
        None,
    )
    binding_entries = validated_binding_entries(binding) if binding is not None else {}
    contract = next(
        (contract for record in records
         if (contract := _persisted_plan_contract(record.metadata)) is not None),
        None,
    )
    amendment_count = 0
    if contract is not None:
        amendments = collect_reviewer_board_amendments(
            comments, flow="plan", issue_number=issue_number,
        )
        validate_backend_outage_amendments_for_binding(amendments, binding)
        lineage = resolve_contract_lineage(
            records, amendments, contract, accept_base_configured=True,
            contract_from_metadata=_persisted_plan_contract,
            drift_error=lambda persisted, detail: AgentLoopError(
                f"Approved plan board history is inconsistent with its signed amendments: {detail}"
            ),
        )
        effective = lineage.contracts[-1]
        original_names = tuple(contract.required_reviewers)
        effective_names = tuple(effective.required_reviewers)
        policy = effective.policy
        primary = effective.primary_reviewer
        amendment_count = len(lineage.amendments)
    elif binding is not None:
        original_names = effective_names = tuple(binding_entries)
        policy, primary = "all-reviewers", None
    else:
        # Plain all-reviewers planning persists no contract or binding: every
        # reviewer that reviewed the plan is on the approving board.
        names: list[str] = []
        for record in records:
            if record.metadata.role == "reviewer" and record.metadata.agent not in names:
                names.append(record.metadata.agent)
        if not names:
            return None
        original_names = effective_names = tuple(names)
        policy, primary = "all-reviewers", None
    return PlanVerificationContext(
        issue_number=issue_number,
        original_reviewers=original_names,
        effective_reviewers=effective_names,
        original_seats=_board_seats(original_names, binding_entries),
        effective_seats=_board_seats(effective_names, binding_entries),
        policy=policy,
        primary_reviewer=primary,
        seat_binding=binding,
        signed_amendment_count=amendment_count,
    )


def plan_verification_config(
    config: object, comments: Sequence[object], *, issue_number: int | None,
) -> object:
    """The config every plan-approval reader uses: the approving plan board.

    Without a plan candidate there is nothing to verify, and every consumer
    already fails closed on a missing canonical plan.
    """
    from .round_state import _extract_round_metadata_records

    if not any(
        record.metadata.role == "coder"
        for record in _extract_round_metadata_records(comments, flow="plan")
    ):
        return config
    context = derive_plan_verification_context(comments, issue_number=issue_number)
    return config if context is None else context.config_for(config)


# --- Board-change audit record ------------------------------------------------


def _seat_payload(seat: BoardSeat) -> dict[str, object]:
    return {
        "id": seat.seat_id,
        "backend": seat.backend,
        "model_chain": list(seat.model_chain) if seat.model_chain is not None else None,
        "effort": seat.effort,
    }


def board_change_payload(
    handoff: object, *, repo: str, pr_number: int, plan_hash: str | None,
) -> dict[str, object]:
    plan = handoff.plan
    return {
        "kind": BOARD_CHANGE_KIND,
        "schema_version": BOARD_CHANGE_SCHEMA_VERSION,
        "repo": repo,
        "pr_number": pr_number,
        "plan_issue": plan.issue_number,
        "plan_hash": plan_hash,
        "old_board": [_seat_payload(seat) for seat in plan.effective_seats],
        "new_board": [_seat_payload(seat) for seat in handoff.pr_seats],
        "effective_round": 1,
        "reason": BOARD_CHANGE_REASON,
    }


def board_change_digest(payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _describe_seat(seat: BoardSeat) -> str:
    if seat.model_chain is None:
        return f"{seat.seat_id} ({seat.backend})"
    models = " -> ".join(model or "default model" for model in seat.model_chain)
    effort = f", effort {seat.effort}" if seat.effort else ""
    return f"{seat.seat_id} ({seat.backend}: {models}{effort})"


def render_board_change_audit(handoff: object, payload: dict[str, object], digest: str) -> str:
    plan = handoff.plan
    changes = [
        f"{_describe_seat(old)} -> {_describe_seat(new)}"
        for old, new in handoff.binding_changes
    ]
    lines = [
        BOARD_CHANGE_AUDIT_HEADING,
        "",
        f"- Record digest: `{digest}`",
        f"- Reason: {BOARD_CHANGE_REASON}",
        "- Effective round: 1",
        (
            f"- Approved plan `{payload['plan_hash']}`"
            + (f" from issue #{plan.issue_number}" if plan.issue_number is not None else "")
            + " stays verified against the plan board that approved it."
        ),
        "- Plan board: " + "; ".join(_describe_seat(seat) for seat in plan.effective_seats),
        "- PR board: " + "; ".join(_describe_seat(seat) for seat in handoff.pr_seats),
        f"- Added reviewer(s): {', '.join(handoff.added) or 'none'}",
        f"- Removed reviewer(s): {', '.join(handoff.removed) or 'none'}",
        f"- Binding change(s): {'; '.join(changes) or 'none'}",
        "- Every PR board reviewer must approve the PR's exact head; plan approvals do "
        "not count as PR approvals.",
        "",
        "-- Orchestrator",
    ]
    return "\n".join(lines)


def board_change_audit_already_posted(comments: Sequence[object], digest: str) -> bool:
    def body(comment: object) -> object:
        if isinstance(comment, dict):
            return comment.get("body")
        return getattr(comment, "body", None)

    return any(
        isinstance(text := body(comment), str)
        and BOARD_CHANGE_AUDIT_HEADING in text
        and digest in text
        for comment in comments
    )


def signed_lineage_authorization(
    original_names: Sequence[str],
    *,
    effective_config: object,
    known_configs: Sequence[object] = (),
    binding: dict | None = None,
) -> SignedBoardAuthorization:
    """The signed authorization of a validated, non-empty PR amendment chain.

    The effective seats keep their configured bindings (PR history refuses a
    backend change on a recorded seat).  A provider exception applies only
    when every original seat's backend is known and the chain lowered it.
    """
    from .reviewer_floor import board_seats_from_config
    from .reviewer_seats import validated_binding_entries

    effective_seats = board_seats_from_config(effective_config)
    backends: dict[str, str] = {}
    for known in known_configs:
        for seat in board_seats_from_config(known):
            backends.setdefault(seat.seat_id, seat.backend)
    if binding is not None:
        for seat_id, entry in validated_binding_entries(binding).items():
            backends.setdefault(seat_id, entry["backend"])
    plain = _plain_backends()
    original_backends = [backends.get(name) or plain.get(name) for name in original_names]
    return SignedBoardAuthorization(
        seats=effective_seats,
        lowered_reviewers=len(effective_seats) < len(original_names),
        lowered_providers=(
            None not in original_backends
            and distinct_providers(effective_seats) < len(set(original_backends))
        ),
    )


def enforce_issue_board_floor(
    config: object, comments: Sequence[object], *, issue_number: int,
) -> None:
    """Issue-mode floor preflight, before any planning or implementation turn.

    The effective board is the invocation's board after the handoff board rule
    against the issue's own plan lineage, so a signed plan amendment excuses
    only the measures its chain lowered.
    """
    from .reviewer_floor import board_floor_enabled, enforce_configured_board_floor
    from .reviewer_seats import resolve_plan_handoff_board

    if not board_floor_enabled(config):
        return
    plan = derive_plan_verification_context(comments, issue_number=issue_number)
    authorizations: tuple[SignedBoardAuthorization, ...] = ()
    effective = config
    if plan is not None:
        handoff = resolve_plan_handoff_board(config, plan)
        effective = handoff.config
        authorizations = handoff.signed_authorizations
    enforce_configured_board_floor(
        effective,
        authorizations=authorizations,
        context=f"Issue #{issue_number} reviewer board",
    )
