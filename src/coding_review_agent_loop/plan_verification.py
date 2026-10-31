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

Recorded plan board changes (#1377): once the plan history carries a plan
board root anchor or an orchestrator-authored plan board-change link, the
context is resolved through the plan board lineage (links and signed plan
amendments in one ordered chain).  The approving board is the board of the
latest lineage element that took effect, and approvals that the lineage
retired (a removed seat, an added or rebound seat before its effective round,
a pre-epoch approval after a primary change) never count.

:func:`classify_plan_approval` decides read-only, from history alone, whether
the current canonical plan is provably completely approved.  An approved plan
is never reopened by a later invocation's board or planning flags.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from collections.abc import Callable, Sequence

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
    # Comment positions of approvals bound to a superseded seat model; they
    # never count towards the approving board (#1373 review item-2).
    stale_approval_indices: frozenset[int] = frozenset()
    # How the approving board was derived: ``lineage`` (plan board root or
    # link, #1377), ``contract``, ``binding`` or ``posters`` (plain legacy
    # history, which proves membership but never completion by itself).
    derivation: str = "contract"
    # Lineage-derived history (#1377): the frozen authorization of every
    # signed segment that governed the approving prefix, kept through later
    # board-change links.  ``None`` for the other derivations.
    lineage_authorizations: tuple[SignedBoardAuthorization, ...] | None = None

    def verification_comments(self, comments: Sequence[object]) -> tuple[object, ...]:
        """``comments`` with superseded-model approvals neutralized in place.

        Positions are kept so sidecar hydration and ordering are unchanged.
        """
        from types import SimpleNamespace

        return tuple(
            SimpleNamespace(body="") if index in self.stale_approval_indices else comment
            for index, comment in enumerate(comments)
        )

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

    @property
    def signed_authorizations(self) -> tuple[SignedBoardAuthorization, ...]:
        """Every floor authorization the approving plan board carries.

        A lineage keeps each signed segment's frozen authorization, so a later
        board-change link never erases a signed reduction; other derivations
        supply the single trailing-amendment authorization.
        """
        if self.lineage_authorizations is not None:
            return self.lineage_authorizations
        authorization = self.signed_authorization
        return (authorization,) if authorization is not None else ()

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
            DEFAULT_PLAN_SWEEP_ROUNDS,
            DEFAULT_PROVENANCE_BASELINE_MAX_CHARS,
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
            plan_resume_keep_on_issue_edit=False,
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
            plan_sweep_rounds=(
                config.plan_sweep_rounds if staged else DEFAULT_PLAN_SWEEP_ROUNDS
            ),
            plan_sweep_reasoning_effort=(
                config.plan_sweep_reasoning_effort if staged else ""
            ),
            plan_provenance_baseline_max_chars=(
                config.plan_provenance_baseline_max_chars
                if staged else DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
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
    comments: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None = None,
) -> PlanVerificationContext | None:
    """Derive the approving plan board from the plan's owning issue history.

    Returns ``None`` when the issue carries no plan round records (or no plan
    reviewer at all), in which case no plan board can be verified.  Malformed
    binding or amendment lineage fails closed.  History carrying a plan board
    root anchor or a plan board-change link is resolved through the plan
    board lineage; its links are admitted only from ``trusted_actor``.
    """
    from .board_amendment import collect_reviewer_board_amendments, resolve_contract_lineage
    from .board_change import PLAN_SURFACE, comments_bear_board_change_links
    from .reviewer_seats import (
        _persisted_plan_contract,
        validate_backend_outage_amendments_for_binding,
        validated_binding_entries,
    )
    from .round_state import _extract_round_metadata_records

    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return None
    if plan_history_bears_lineage(comments, records):
        return _lineage_verification_context(
            comments, records, issue_number=issue_number, trusted_actor=trusted_actor,
        )
    binding, binding_entries, stale = _validated_plan_binding_history(records)
    contract = next(
        (contract for record in records
         if (contract := _persisted_plan_contract(record.metadata)) is not None),
        None,
    )
    if contract is None and any(
        record.metadata.scheduler_metadata_status == "invalid" for record in records
    ):
        # A malformed planning checkpoint hides the persisted policy; never
        # downgrade it to all-reviewers verification.
        raise AgentLoopError(
            "Approved plan scheduler history is malformed, so its planning policy "
            "cannot be verified."
        )
    amendment_count = 0
    derivation = "contract"
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
        derivation = "binding"
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
        derivation = "posters"
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
        stale_approval_indices=stale,
        derivation=derivation,
    )


def plan_history_bears_lineage(comments: Sequence[object], records: Sequence[object]) -> bool:
    """Whether plan history must be judged through its board lineage (#1377).

    A link record, a root anchor, or any record carrying a board-change
    digest all do: a change digest is validated against the lineage on every
    path, never ignored by the legacy derivations.
    """
    from .board_change import PLAN_SURFACE, comments_bear_board_change_links

    return comments_bear_board_change_links(comments, surface=PLAN_SURFACE) or any(
        record.metadata.plan_board_root is not None
        or record.metadata.reviewer_board_change_digest is not None
        for record in records
    )


def _lineage_verification_context(
    comments: Sequence[object],
    records: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None,
) -> PlanVerificationContext:
    """The approving plan board of history with a plan board lineage (#1377).

    Resolved through the plan surface resolver, never from the invocation's
    planning configuration.  The approving identity is the board of the
    latest lineage element any plan round record took effect under (pending
    elements never governed an approval).  Any lineage defect raises.
    """
    import dataclasses

    from .board_change import (
        PLAN_SURFACE,
        BoardIdentity,
        _segment_authorization,
        apply_inits,
        collect_board_change_links,
    )
    from .plan_board_state import resolve_plan_lineage

    if issue_number is None:
        raise AgentLoopError(
            "Approved plan board lineage cannot be verified without its owning issue."
        )
    anchored = next(
        (record.metadata.plan_board_root for record in records
         if record.metadata.plan_board_root is not None),
        None,
    )
    if anchored is not None:
        root = BoardIdentity.from_payload(anchored)
    else:
        links = collect_board_change_links(
            comments, pr_number=issue_number, trusted_actor=trusted_actor, surface=PLAN_SURFACE,
        )
        if not links:
            raise AgentLoopError(
                "Approved plan board history is inconsistent with its lineage: a plan round "
                "record carries a board-change digest, but the issue has neither a plan board "
                "root nor an admitted board-change record."
            )
        root = links[0].root
    lineage = resolve_plan_lineage(
        comments,
        issue_number=issue_number,
        configured=root,
        trusted_actor=trusted_actor,
        drift_error=lambda _persisted, detail: AgentLoopError(
            f"Approved plan board history is inconsistent with its lineage: {detail}"
        ),
    )
    # Only elements some plan round record took effect under can have
    # governed an approval.
    taken = [element for element in lineage.elements if element.digest in lineage.carried]
    last = lineage.elements.index(taken[-1]) + 1 if taken else 0
    view = dataclasses.replace(lineage, elements=lineage.elements[:last])
    effective = view.current_identity
    base = apply_inits(view.segment_base, view.inits)
    segment_amendments = 0
    for element in reversed(view.elements):
        if element.kind != "amendment":
            break
        segment_amendments += 1
    fresh = view.fresh_from_rounds
    epoch = view.primary_epoch_round
    effective_names = set(effective.names)
    stale: set[int] = set()
    for record in records:
        metadata = record.metadata
        if epoch is not None and metadata.round_number < epoch and metadata.role != "coder":
            # A primary change restarts the primary phase: pre-epoch reviews,
            # panel openings and approvals never count.
            stale.add(record.index)
            continue
        if metadata.role != "reviewer" or metadata.state != "approved":
            continue
        if metadata.agent not in effective_names or metadata.round_number < fresh.get(
            metadata.agent, 0
        ):
            stale.add(record.index)
    binding = base.seat_binding() if base.named else effective.seat_binding()
    # Each signed segment of the approving prefix keeps its frozen
    # authorization, also when a later link follows it.
    authorizations: list[SignedBoardAuthorization] = []
    previous = lineage.root
    position = 0
    while position < len(view.elements):
        if view.elements[position].kind != "amendment":
            previous = view.elements[position].identity
            position += 1
            continue
        pre = previous
        while position < len(view.elements) and view.elements[position].kind == "amendment":
            previous = view.elements[position].identity
            position += 1
        authorizations.append(_segment_authorization(pre, previous))
    return PlanVerificationContext(
        issue_number=issue_number,
        original_reviewers=base.names,
        effective_reviewers=effective.names,
        original_seats=base.seats,
        effective_seats=effective.seats,
        policy=effective.policy,
        primary_reviewer=effective.primary,
        seat_binding=binding,
        signed_amendment_count=segment_amendments,
        stale_approval_indices=frozenset(stale),
        derivation="lineage",
        lineage_authorizations=tuple(authorizations),
    )


def _validated_plan_binding_history(
    records: Sequence[object],
) -> tuple[dict | None, dict[str, dict], frozenset[int]]:
    """Validate every plan seat binding against the plan's own history.

    Returns the latest binding, its entries, and the positions of approvals
    that a later model chain or effort reconfiguration superseded: any
    approval recorded before the latest binding transition, including one
    that returns to an earlier model, plus any approval bound differently
    from the latest binding.  Mixed
    bound/unbound rounds, changed backends, a changed recorded board, and a
    reviewer outside its own binding fail closed.  The PR invocation board is
    never consulted.
    """
    from .reviewer_seats import validated_binding_entries

    bound = [record for record in records if record.metadata.seat_binding is not None]
    if not bound:
        return None, {}, frozenset()
    latest = bound[-1].metadata.seat_binding
    latest_entries = validated_binding_entries(latest)
    board = set(latest_entries)
    backends: dict[str, str] = {}
    stale: set[int] = set()
    approvals: list[tuple[int, int]] = []  # (record index, comment index)
    previous_models: dict[str, tuple[object, object]] | None = None
    last_transition = -1
    for position, record in enumerate(records):
        metadata = record.metadata
        recorded = metadata.seat_binding
        if recorded is None:
            if metadata.role in {"reviewer", "coder", "summary"}:
                raise AgentLoopError(
                    "Approved plan history mixes bound and unbound round records "
                    f"({metadata.role}, round {metadata.round_number}); its approving "
                    "board cannot be verified."
                )
            continue
        entries = validated_binding_entries(recorded)
        if set(entries) != board:
            raise AgentLoopError(
                "Approved plan seat binding changed its recorded board; its approving "
                "board cannot be verified."
            )
        for seat_id, entry in entries.items():
            if backends.setdefault(seat_id, entry["backend"]) != entry["backend"]:
                raise AgentLoopError(
                    f"Approved plan seat {seat_id!r} changed backend in plan history."
                )
        models = {
            seat_id: (tuple(entry["model_chain"]), entry.get("effort"))
            for seat_id, entry in entries.items()
        }
        if previous_models is not None and models != previous_models:
            # A model chain or effort reconfiguration: the live plan loop
            # clears approval and panel state here, so every approval recorded
            # before it is superseded, even when a later record restores the
            # earlier model (A -> B -> A).
            last_transition = position
        previous_models = models
        if metadata.role != "reviewer":
            continue
        own = entries.get(metadata.agent)
        if own is None:
            raise AgentLoopError(
                "Approved plan reviewer record is not represented in its seat binding."
            )
        current = latest_entries[metadata.agent]
        if metadata.state == "approved":
            approvals.append((position, record.index))
            if (
                own["model_chain"] != current["model_chain"]
                or own.get("effort") != current.get("effort")
            ):
                stale.add(record.index)
    stale.update(index for position, index in approvals if position < last_transition)
    return latest, latest_entries, frozenset(stale)


def plan_verification_inputs(
    config: object,
    comments: Sequence[object],
    *,
    issue_number: int | None,
    trusted_actor: tuple[str, int] | None = None,
    scope_start_round: int | None = None,
) -> tuple[object, tuple[object, ...]]:
    """The config and comments every plan-approval reader uses.

    The config is re-pointed at the approving plan board; the comments have
    approvals bound to a superseded seat model neutralized.  Without a plan
    candidate there is nothing to verify, and every consumer already fails
    closed on a missing canonical plan.

    The returned comments are the current planning scope (#1390): planning
    records of an earlier scope are blanked in place, so every candidate and
    obligation reader downstream sees only the current scope while the board
    is still derived from the full history.  ``scope_start_round`` defaults
    to the admitted scope-boundary record's.
    """
    from .plan_issue_scope import resolve_plan_scope, scope_plan_comments
    from .round_state import _extract_round_metadata_records

    comments = tuple(comments)
    if scope_start_round is None:
        scope_start_round = resolve_plan_scope(
            comments, trusted_actor, issue_number=issue_number
        )
    if not any(
        record.metadata.role == "coder"
        for record in _extract_round_metadata_records(comments, flow="plan")
    ):
        return config, scope_plan_comments(comments, scope_start_round)
    context = derive_plan_verification_context(
        comments, issue_number=issue_number, trusted_actor=trusted_actor,
    )
    if context is None:
        return config, scope_plan_comments(comments, scope_start_round)
    return context.config_for(config), scope_plan_comments(
        context.verification_comments(comments), scope_start_round
    )


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


# --- Plan approval classification (#1377) --------------------------------------

# ``empty-scope`` (#1390): an edited-issue rescope opened a planning scope that
# has no plan candidate yet.  Every consumer treats it like ``no-plan`` except
# the issue floor preflight, which floor-checks the plan board at the scope
# start instead of deriving the archived plan's handoff board.
PLAN_APPROVAL_STATES = ("no-plan", "empty-scope", "unfinished", "undetermined", "approved")


@dataclass(frozen=True)
class PlanApprovalClassification:
    """Whether the current canonical plan is provably completely approved.

    ``approved`` exempts a run from plan board-change candidacy, planning
    policy drift and the planning candidate floor; it never produces an
    approved outcome by itself.  The live approval boundary still runs, and
    any input it cannot confirm downgrades the run to normal resolution.
    """

    state: str
    reason: str
    context: PlanVerificationContext | None = None
    plan_hash: str | None = None

    @property
    def approved(self) -> bool:
        return self.state == "approved"


def plan_hash_has_issue_handoff(
    comments: Sequence[object], *, issue_number: int, repo: str | None, plan_hash: str,
) -> bool:
    """Whether an issue-side implementation handoff names exactly ``plan_hash``.

    Completion evidence for unanchored plain history, whose posters prove
    membership only (#1377 step 12a).  Malformed records never count.
    """
    from .decomposition import (
        ONE_SHOT_IMPL_HANDOFF_MARKER_RE,
        PHASE_IMPLEMENTATION_MARKER_RE,
        _decode_one_shot_impl_handoff_metadata,
        _decode_phase_implementation_handoff_metadata,
    )
    from .issue_pr_handoff import find_latest_issue_pr_handoff

    if repo is not None:
        try:
            handoff = find_latest_issue_pr_handoff(comments, issue_number=issue_number, repo=repo)
        except AgentLoopError:
            handoff = None
        if handoff is not None and handoff.plan_hash == plan_hash:
            return True
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for pattern, decode in (
            (ONE_SHOT_IMPL_HANDOFF_MARKER_RE, _decode_one_shot_impl_handoff_metadata),
            (PHASE_IMPLEMENTATION_MARKER_RE, _decode_phase_implementation_handoff_metadata),
        ):
            for match in pattern.finditer(body):
                try:
                    metadata = decode(match.group("payload"))
                except (AgentLoopError, ValueError, KeyError, TypeError):
                    continue
                if (
                    getattr(metadata, "parent_issue", None) == issue_number
                    and getattr(metadata, "plan_hash", None) == plan_hash
                ):
                    return True
    return False


def plan_open_obligations(plan_round: object, *, board: Sequence[str]) -> tuple[str, ...]:
    """Active blocking or same-plan items left after the resumed round reconciles.

    The ledger view reassigns removed reviewers' items to the remaining
    ``board``, so a removed reviewer's unresolved finding stays open.
    """
    from .board_amendment import apply_board_amendment_to_ledger
    from .round_state import _aggregate_record_dispositions
    from .unresolved_items import _apply_unresolved_item_dispositions

    prior = tuple(getattr(plan_round, "prior_items", ()))
    new = tuple(getattr(plan_round, "current_round_new_items", ()))
    if not new:
        new = tuple(
            item
            for record in getattr(plan_round, "completed_reviews", ())
            for item in record.metadata.new_items
        )
    owners = {
        owner
        for item in (*prior, *new)
        for owner in (tuple(item.resolution_owners or ()) or (item.reviewer,))
    }
    removed = tuple(sorted(name for name in owners if name and name not in set(board)))
    view, _ = apply_board_amendment_to_ledger(
        prior,
        removed_reviewers=removed,
        remaining_reviewers=tuple(board),
        primary_reviewer=None,
    )
    items, _future = _apply_unresolved_item_dispositions(
        view,
        _aggregate_record_dispositions(tuple(getattr(plan_round, "completed_reviews", ()))),
        same_status="same-plan",
        retain_future=False,
    )
    open_ids = [item.item_id for item in items if item.status in {"blocking", "same-plan"}]
    prior_ids = {item.item_id for item in prior}
    open_ids += [
        item.item_id for item in new
        if item.status in {"blocking", "same-plan"} and item.item_id not in prior_ids
    ]
    return tuple(dict.fromkeys(open_ids))


def plan_round_candidate_key(plan_round: object, *, plan_text: str, surfaced_ids: Sequence[str]):
    """The exact-plan candidate key of the resumed round under ``surfaced_ids``, or ``None``."""
    from .panel_evidence import _plan_candidate_key_for
    from .plan_assembly import decode_assembled_plan_sidecar
    from .round_state import _plan_subject

    coder_metadata = getattr(plan_round, "coder_metadata", None)
    sidecar = None
    if coder_metadata is not None and coder_metadata.assembled_plan_sidecar is not None:
        try:
            sidecar = decode_assembled_plan_sidecar(coder_metadata.assembled_plan_sidecar)
        except (AgentLoopError, ValueError, KeyError, TypeError):
            return None
    return _plan_candidate_key_for(
        plan_subject=_plan_subject(plan_text),
        sidecar=sidecar,
        surfaced_requirement_ids=surfaced_ids,
    )


def approval_record_binding(
    record: object,
    *,
    current_key: object | None,
    surfaced_ids: Sequence[str],
    parent_ids: Sequence[str] | None,
    legacy_fallback: Callable[[object], str] | None = None,
    check_acknowledgement: bool = True,
) -> str:
    """``bound``, ``stale`` or ``unproven`` for one counted plan approval (step 12c).

    A recorded exact-plan candidate key binds only when it equals the key
    recomputed against the current requirement set and its recorded
    requirement IDs equal the current surfaced set (``check_acknowledgement``;
    the live loop leaves an unacknowledged approval to its acknowledgement
    repair).  A keyless legacy approval is judged by ``legacy_fallback``.  For a staged child, the recorded authoritative-parent
    requirement IDs must equal the current parent set; a record without them
    binds only while that set is empty.
    """
    from .panel_evidence import _plan_key_from_payload

    metadata = record.metadata  # type: ignore[attr-defined]
    surfaced = {str(item) for item in surfaced_ids}
    if parent_ids is not None:
        recorded_parent = metadata.parent_requirement_ids
        if recorded_parent is None:
            if parent_ids:
                return "unproven"
        elif set(recorded_parent) != set(parent_ids):
            return "stale"
    key = _plan_key_from_payload(metadata.plan_candidate_key)
    if key is None:
        return legacy_fallback(record) if legacy_fallback is not None else "unproven"
    if current_key is None or not key.matches(current_key):
        return "stale"
    if (
        check_acknowledgement
        and surfaced
        and set(metadata.surfaced_reviewer_requirement_ids) != surfaced
    ):
        return "stale"
    return "bound"


def plan_coder_dispositions_complete(
    coder_output: str | None,
    surfaced_ids: Sequence[str],
    *,
    sidecar_payload: object = None,
    response_form: str | None = None,
) -> bool:
    """Whether the current coder plan dispositions name exactly ``surfaced_ids``.

    The live approval gate's coder attestation, shared with the
    classification.  A semantic patch inherits its dispositions from the
    authenticated assembled state.  An unparsable (unversioned) plan never
    proves its disposition set, even an empty one.
    """
    from .plan_assembly import decode_assembled_plan_sidecar, hydrate_authenticated_plan_state
    from .response_validation import _current_plan_has_complete_human_requirement_dispositions

    inherited = None
    if response_form == "semantic-patch-v1" and sidecar_payload is not None:
        try:
            inherited = hydrate_authenticated_plan_state(
                decode_assembled_plan_sidecar(sidecar_payload)
            ).plan.human_requirement_dispositions
        except (AgentLoopError, ValueError, KeyError, TypeError):
            return False
    return _current_plan_has_complete_human_requirement_dispositions(
        coder_output, surfaced_requirement_ids=tuple(surfaced_ids),
        inherited_dispositions=inherited,
    )


def legacy_requirement_fallback(
    dispositions_ok: bool, surfaced_ids: Sequence[str]
) -> Callable[[object], str]:
    """The authenticated fallback for a keyless legacy approval (step 12c2).

    The plan's own dispositions must name exactly the current surfaced set,
    empty or not (``dispositions_ok``); an unversioned plan cannot prove it.
    With requirements surfaced, the approval text must carry the
    acknowledgement.  With none surfaced, an approval text carrying it was
    given under a since-withdrawn requirement (the marker is prohibited when
    none are surfaced), so it is stale.
    """
    from .protocol import human_requirements_resolved

    def fallback(record: object) -> str:
        acknowledged = human_requirements_resolved(getattr(record, "body", "") or "")
        if not surfaced_ids and acknowledged:
            return "stale"
        if not dispositions_ok:
            return "unproven"
        if surfaced_ids and not acknowledged:
            return "unproven"
        return "bound"

    return fallback


def _round_dispositions_ok(plan_round: object, surfaced_ids: Sequence[str]) -> bool:
    metadata = getattr(plan_round, "coder_metadata", None)
    return plan_coder_dispositions_complete(
        getattr(plan_round, "coder_output", None),
        surfaced_ids,
        sidecar_payload=getattr(metadata, "assembled_plan_sidecar", None),
        response_form=getattr(metadata, "response_form", None),
    )


def plan_ledger_unreconstructible_items(
    records: Sequence[object], plan_round: object, *, ignored_ids: Sequence[str] = (),
) -> tuple[str, ...]:
    """Blocking or same-plan items published in history that the resumed ledger lost.

    Every such item must still be in the resumed round's ledger or provably
    cleared by the recorded reconciliation history; anything else means the
    obligations cannot be recovered, so completion fails closed.
    """
    from .round_state import _canonically_resolved_history_item_ids

    ledger_ids = {
        item.item_id
        for item in (
            *getattr(plan_round, "prior_items", ()),
            *getattr(plan_round, "current_round_new_items", ()),
        )
    }
    published = {
        item.item_id: item
        for record in records
        for item in record.metadata.new_items  # type: ignore[attr-defined]
        if item.status in {"blocking", "same-plan"} and item.item_id not in set(ignored_ids)
    }
    resolved = _canonically_resolved_history_item_ids(
        records, reconciliation_mode="aggregate", same_status="same-plan",
        current_carried_ids=tuple(sorted(ledger_ids)),
    )
    return tuple(sorted(
        item_id for item_id in published if item_id not in ledger_ids and item_id not in resolved
    ))


@dataclass(frozen=True)
class PlanApprovalInputs:
    """The live, read-only inputs of the plan approval classification (#1377 step 12).

    Built once by :func:`build_plan_approval_inputs` for the issue-entry
    preflight, managed-CI no-handoff recovery and strict qualification; plan
    startup supplies the same fields from its own live state.  ``derivable``
    is false when any input could not be read; the classification is then
    never approved.
    """

    human_requirements: tuple[object, ...] = ()
    # Authoritative staged-parent signed requirement IDs; ``None`` without a
    # staged parent.
    parent_requirement_ids: tuple[str, ...] | None = None
    inherited_check: Callable[[str], str | None] | None = None
    supersession_check: Callable[[str], bool] | None = None
    derivable: bool = True
    underivable_reason: str = ""
    # The planning scope start the classification judges (#1390); ``None``
    # derives it from the admitted scope-boundary record.
    scope_start_round: int | None = None


def build_plan_approval_inputs(
    runner: object | None,
    *,
    config: object,
    issue_context: object,
    parent_issue_context: object | None = None,
    scope_start_round: int | None = None,
) -> PlanApprovalInputs:
    """Derive every live classification input read-only from the issue and its parent.

    ``scope_start_round`` carries a planning scope the caller already decided
    (#1390), including one the plan loop is about to open.
    """
    inputs = _build_plan_approval_inputs(
        runner, config=config, issue_context=issue_context,
        parent_issue_context=parent_issue_context,
    )
    if scope_start_round is None:
        return inputs
    return replace(inputs, scope_start_round=scope_start_round)


def _build_plan_approval_inputs(
    runner: object | None,
    *,
    config: object,
    issue_context: object,
    parent_issue_context: object | None = None,
) -> PlanApprovalInputs:
    """Derive every live classification input read-only from the issue and its parent.

    A staged child's authoritative parent is read for its current signed
    requirement IDs; a fresh planning child additionally gets its
    inherited-matrix admissibility check and its signed supersession check,
    exactly as the issue router and the plan loop judge them.  Any read that
    cannot be completed makes the inputs underivable.
    """
    from .execution_policy import _infer_staged_parent_issue, _resolve_fresh_child_provenance
    from .github import get_issue_context

    requirements = tuple(getattr(issue_context, "human_requirements", ()) or ())
    try:
        parent_number = _infer_staged_parent_issue(issue_context)  # type: ignore[arg-type]
    except AgentLoopError as exc:
        return PlanApprovalInputs(requirements, derivable=False, underivable_reason=str(exc))
    if parent_number is None or parent_number == getattr(issue_context, "number", None):
        return PlanApprovalInputs(requirements)
    parent = parent_issue_context
    if parent is None or getattr(parent, "number", None) != parent_number:
        if runner is None:
            return PlanApprovalInputs(
                requirements, derivable=False,
                underivable_reason=f"parent issue #{parent_number} cannot be read",
            )
        try:
            parent = get_issue_context(runner, config=config, issue_number=parent_number)  # type: ignore[arg-type]
        except AgentLoopError as exc:
            return PlanApprovalInputs(requirements, derivable=False, underivable_reason=str(exc))
    parent_ids = tuple(sorted(
        requirement.requirement_id for requirement in getattr(parent, "human_requirements", ())
    ))
    if getattr(parent, "human_requirements", ()):
        from .github import deduplicate_human_requirements

        # The planning requirement set includes the authoritative parent's.
        requirements = deduplicate_human_requirements(
            (*parent.human_requirements, *requirements)  # type: ignore[union-attr]
        )
    try:
        fresh_child = _resolve_fresh_child_provenance(
            issue_context=issue_context, parent_issue_context=parent,  # type: ignore[arg-type]
        )
    except AgentLoopError as exc:
        return PlanApprovalInputs(
            requirements, parent_ids, derivable=False, underivable_reason=str(exc),
        )
    if fresh_child is None or not fresh_child.route.is_planning:
        return PlanApprovalInputs(requirements, parent_ids)

    from .child_plan_binding import _child_plan_admissibility_failure
    from .decomposition import collect_child_plan_supersessions
    from .round_state import make_approved_plan_context

    child_number = getattr(issue_context, "number", None)

    def inherited_check(plan_text: str) -> str | None:
        return _child_plan_admissibility_failure(
            fresh_child.parent_plan_context,
            make_approved_plan_context(plan_text, source_locator=f"issue #{child_number} plan"),
            stage_id=fresh_child.stage_id,
        )

    def supersession_check(plan_hash: str) -> bool:
        return any(
            record.superseded_plan_hash == plan_hash
            for record in collect_child_plan_supersessions(
                getattr(issue_context, "comments", ()),
                child_issue=child_number,  # type: ignore[arg-type]
                parent_issue=fresh_child.parent_issue,
                stage_id=fresh_child.stage_id,
                ignored_sink=[],
            )
        )

    return PlanApprovalInputs(
        requirements, parent_ids, inherited_check=inherited_check,
        supersession_check=supersession_check,
    )


def classify_plan_approval(
    comments: Sequence[object],
    *,
    config: object,
    issue_number: int,
    human_requirements: Sequence[object] = (),
    trusted_actor: tuple[str, int] | None = None,
    inherited_check: Callable[[str], str | None] | None = None,
    inherited_derivable: bool = True,
    supersession_pending: bool = False,
    supersession_check: Callable[[str], bool] | None = None,
    parent_requirement_ids: Sequence[str] | None = None,
    inputs: PlanApprovalInputs | None = None,
    poster_completion_proven: bool = False,
    execution_mode_guard: bool = True,
    scope_start_round: int | None = None,
    prove_poster_incompleteness: bool = False,
) -> PlanApprovalClassification:
    """Read-only plan approval classification from history alone (#1377 step 12).

    Shared by the issue-entry floor preflight, plan startup, managed-CI
    no-handoff recovery and strict qualification.  Every input comes from
    durable history: the approving plan identity (never the invocation's
    reviewers or planning flags), the reconstructed latest round and a
    reconstructible ledger, the current signed requirement set (issue plus
    authoritative parent) against each counted approval's requirement
    binding, the inherited-matrix admissibility, a pending signed
    supersession, the execution-mode guard and the plan-growth gate.  Any
    underivable input means not approved; a malformed lineage raises.

    ``poster_completion_proven`` is set only by managed-CI recovery and
    qualification, whose own complete-approval check over the plan board
    proves completion of unanchored plain history (as before #1377);
    ``execution_mode_guard`` is off there because no planning turn follows.

    Two inputs (#1390): the approving board is derived from the full history,
    while the plan candidate, its ledger and its approvals are read from the
    current planning scope (``scope_start_round``, derived from the admitted
    scope-boundary record when omitted).  A scope with no candidate yet is
    ``empty-scope``.

    ``prove_poster_incompleteness`` is set only by the plain-issue-mode guard
    (#1090).  Before unanchored poster-derived history is reported
    ``undetermined``, the latest round's completeness is checked first: a
    round that is provably incomplete (an approving poster beside another
    poster whose latest record does not approve the exact plan) is
    ``unfinished``.  Posters still never prove completion, so a round that
    looks complete stays ``undetermined``.
    """
    from .config import reviewers
    from .panel_evidence import _require_complete_canonical_plan_approval
    from .plan_review_scheduling import plan_policy_capabilities
    from .response_validation import _surfaced_reviewer_requirement_ids
    from .round_state import _extract_round_metadata_records, _resume_plan_round
    from .decomposition import approved_plan_hash

    from .board_change import PLAN_SURFACE, comments_bear_board_change_links
    from .config import phased_delivery_guard_active
    from .execution_policy import _current_execution_recommendation

    if inputs is not None:
        if not inputs.derivable:
            return PlanApprovalClassification(
                "undetermined", f"an approval input is not derivable: {inputs.underivable_reason}",
            )
        human_requirements = inputs.human_requirements
        parent_requirement_ids = inputs.parent_requirement_ids
        inherited_check = inputs.inherited_check
        supersession_check = inputs.supersession_check
        if scope_start_round is None:
            scope_start_round = inputs.scope_start_round
    from .plan_issue_scope import resolve_plan_scope, scope_plan_comments

    comments = tuple(comments)
    if scope_start_round is None:
        scope_start_round = resolve_plan_scope(
            comments, trusted_actor, issue_number=issue_number
        )
    records = _extract_round_metadata_records(comments, flow="plan")
    if not records:
        return PlanApprovalClassification("no-plan", "no plan round records")
    scoped_records = tuple(
        record for record in records if record.metadata.round_number >= scope_start_round
    )
    lineage_bearing = plan_history_bears_lineage(comments, records)
    try:
        context = derive_plan_verification_context(
            comments, issue_number=issue_number, trusted_actor=trusted_actor,
        )
    except AgentLoopError as exc:
        if lineage_bearing:
            # A malformed plan board lineage fails closed.
            raise
        # Degraded legacy history keeps today's handling in the live loop.
        return PlanApprovalClassification("undetermined", f"legacy plan history: {exc}")
    if scope_start_round > 1 and not any(
        record.metadata.role == "coder" for record in scoped_records
    ):
        return PlanApprovalClassification(
            "empty-scope",
            f"the planning scope starting at round {scope_start_round} has no plan candidate yet",
            context,
        )
    if context is None:
        return PlanApprovalClassification("unfinished", "no plan reviewer has reviewed the plan")
    plan_config = context.config_for(config)
    plan_comments = scope_plan_comments(
        context.verification_comments(comments), scope_start_round
    )
    try:
        resumed = _resume_plan_round(
            plan_comments, configured_reviewers=reviewers(plan_config),
            min_round_number=scope_start_round,
        )
    except (AgentLoopError, ValueError, KeyError, TypeError) as exc:
        return PlanApprovalClassification(
            "undetermined", f"the latest plan round cannot be reconstructed: {exc}", context,
        )
    if resumed is None:
        return PlanApprovalClassification("unfinished", "no reconstructible plan round", context)
    from .approved_plan_resume import approved_plan_resume_view

    resumed = approved_plan_resume_view(
        plan_comments, resumed, config=plan_config, human_requirements=human_requirements,
        min_round_number=scope_start_round,
    )
    plan_text, plan_round = resumed
    plan_hash = approved_plan_hash(plan_text)

    def result(state: str, reason: str) -> PlanApprovalClassification:
        return PlanApprovalClassification(state, reason, context, plan_hash)

    if execution_mode_guard and phased_delivery_guard_active(
        getattr(config, "plan_execution_mode", "")
    ):
        # The execution-mode guard (#1268): a staged plan under a guard-active
        # mode is narrowed or stopped by the live loop, never approved here.
        try:
            recommendation = _current_execution_recommendation(plan_text, comments)
        except AgentLoopError:
            return result("undetermined", "the execution recommendation cannot be read")
        if recommendation is not None and recommendation.strategy == "staged":
            return result("unfinished", "a staged plan under a phased-delivery execution mode")

    try:
        superseded = supersession_pending or (
            supersession_check is not None and supersession_check(plan_hash)
        )
    except AgentLoopError as exc:
        return result("undetermined", f"the signed supersession state cannot be read: {exc}")
    if superseded:
        return result("unfinished", "a validated signed plan supersession is pending")
    if not inherited_derivable:
        return result("undetermined", "the inherited-matrix admissibility is not derivable")
    if inherited_check is not None and inherited_check(plan_text) is not None:
        return result("unfinished", "the plan fails its inherited parent matrix rows")
    if context.derivation == "posters" and not poster_completion_proven and not plan_hash_has_issue_handoff(
        comments, issue_number=issue_number, repo=getattr(config, "repo", None),
        plan_hash=plan_hash,
    ):
        # Historical posters prove membership, never completion (step 12a),
        # but they can prove incompleteness (#1090).
        if prove_poster_incompleteness:
            try:
                _require_complete_canonical_plan_approval(
                    plan_comments,
                    config=plan_config,
                    plan_text=plan_text,
                    plan_round=plan_round,
                    human_requirements=human_requirements,
                    error_message="incomplete",
                )
            except AgentLoopError:
                return result("unfinished", "the latest legacy plan round is provably incomplete")
            except (ValueError, KeyError, TypeError):
                pass
        return result(
            "undetermined",
            "unanchored plain plan history has no issue-side handoff bound to the plan hash",
        )
    try:
        _require_complete_canonical_plan_approval(
            plan_comments,
            config=plan_config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=human_requirements,
            error_message="incomplete",
        )
    except AgentLoopError:
        return result("unfinished", "not every approving reviewer approved the exact plan")
    except (ValueError, KeyError, TypeError) as exc:
        return result("undetermined", f"the approval state cannot be reconstructed: {exc}")
    board = context.effective_reviewers
    # A reconstructible ledger: nothing published in history may be lost.
    # Recorded reconciliation evidence, not the resume reader's conservative
    # incompleteness flag, distinguishes cleared history from lost items.
    # Publication and reconciliation history is never filtered: a primary
    # epoch or stale approval neutralizes evidence, not obligations.  Only a
    # planning-scope boundary retires obligations: they belong to the archived
    # plan, which no longer binds the issue (#1390).
    verification_records = _extract_round_metadata_records(plan_comments, flow="plan")
    lost = plan_ledger_unreconstructible_items(scoped_records, plan_round)
    if lost:
        return result("undetermined", "unreconstructible plan obligation(s) " + ", ".join(lost))
    open_items = plan_open_obligations(plan_round, board=board)
    if open_items:
        return result("unfinished", "open plan obligation(s) " + ", ".join(open_items))
    surfaced_ids = _surfaced_reviewer_requirement_ids(
        human_requirements, requirement_scope="planning requirements",
    )
    current_key = plan_round_candidate_key(plan_round, plan_text=plan_text, surfaced_ids=surfaced_ids)
    staged = plan_policy_capabilities(context.policy).scheduler_enabled
    counted: list[object] = []
    if staged:
        latest: dict[str, object] = {}
        for record in verification_records:
            if record.metadata.role == "reviewer" and record.metadata.agent in board:
                latest[record.metadata.agent] = record
        counted = list(latest.values())
    else:
        counted = [
            record for record in plan_round.completed_reviews
            if record.metadata.agent in board and record.metadata.state == "approved"
        ]
    dispositions_ok = _round_dispositions_ok(plan_round, surfaced_ids)
    if surfaced_ids and not dispositions_ok:
        # The live approval gate's coder attestation (step 12): every
        # surfaced requirement needs a coder disposition before approval.
        return result("unfinished", "the plan lacks complete requirement dispositions")
    fallback = legacy_requirement_fallback(dispositions_ok, surfaced_ids)
    bindings = {
        approval_record_binding(
            record,
            current_key=current_key,
            surfaced_ids=surfaced_ids,
            parent_ids=parent_requirement_ids,
            legacy_fallback=None if staged else fallback,
        )
        for record in counted
    }
    if "stale" in bindings:
        return result("unfinished", "an approval is bound to a stale requirement set")
    if "unproven" in bindings:
        return result("undetermined", "an approval's requirement binding is unproven")
    return result("approved", "every approving reviewer approved the exact plan")


def enforce_issue_board_floor(
    config: object,
    comments: Sequence[object],
    *,
    issue_number: int,
    runner: object | None = None,
    human_requirements: Sequence[object] = (),
    planning: bool = False,
    inputs: PlanApprovalInputs | None = None,
    issue_digest: str | None = None,
) -> None:
    """Issue-mode floor preflight, before any planning or implementation turn.

    Two branches (#1377 step 9).  When the issue's current plan is provably
    approved, or there is no plan to resume, the effective board is the
    invocation's board after the stage 1 handoff board rule against the
    approving plan context, so a signed plan amendment excuses only the
    measures its chain lowered; the invocation's planning flags are never
    compared.  An approved plan resumed plan-only selects no board at all,
    so neither floor applies.  Otherwise a plan-first run resolves its plan
    board read-only, including any board-change candidate, and refuses a
    below-floor plan board before any agent invocation or comment post.

    ``issue_digest`` is the current issue title/body digest of a plan-first
    run (#1390).  The preflight then makes the same read-only edited-issue
    decision as plan startup: when the loop is about to open a fresh planning
    scope, the classification is ``empty-scope`` and the plan board is
    resolved from full lineage and floor-checked at that scope start, so a
    board the plan floor refuses is refused before the boundary is posted.
    """
    from .reviewer_floor import board_floor_enabled, enforce_configured_board_floor
    from .reviewer_seats import resolve_plan_handoff_board

    if not board_floor_enabled(config):
        return
    trusted_actor = None
    if runner is not None:
        from .plan_board_state import admitted_issue_comments

        comments, trusted_actor = admitted_issue_comments(
            runner, config=config, issue_number=issue_number, comments=comments,
        )
    if planning:
        from .plan_issue_scope import plan_scope_decision, resolve_plan_scope

        scope_start = (
            inputs.scope_start_round
            if inputs is not None and inputs.scope_start_round is not None
            else resolve_plan_scope(comments, trusted_actor, issue_number=issue_number)
        )
        classification = classify_plan_approval(
            comments,
            config=config,
            issue_number=issue_number,
            human_requirements=human_requirements,
            trusted_actor=trusted_actor,
            inputs=inputs,
            scope_start_round=scope_start,
        )
        if issue_digest is not None:
            decision = plan_scope_decision(
                comments,
                trusted_actor,
                issue_digest,
                bool(getattr(config, "plan_resume_keep_on_issue_edit", False)),
                plan_approved=classification.approved,
                issue_number=issue_number,
                repo=getattr(config, "repo", None),
            )
            if decision.reset_required:
                # The plan loop is about to open this scope: judge it as the
                # loop will, before anything is posted.
                scope_start = decision.scope_start_round
                classification = classify_plan_approval(
                    comments,
                    config=config,
                    issue_number=issue_number,
                    human_requirements=human_requirements,
                    trusted_actor=trusted_actor,
                    inputs=inputs,
                    scope_start_round=scope_start,
                )
        if classification.approved and getattr(config, "plan_execution_mode", None) == "plan-only":
            # Nothing is implemented and planning is not reopened: no board
            # is chosen by this invocation, so there is nothing to floor.
            return
        if classification.state not in {"approved", "no-plan"}:
            from .plan_board_state import (
                plan_drift_error,
                plan_floor_refusal_hint,
                plan_resume_round,
                resolve_plan_board_state,
            )

            def scope_resume_round(effective_config: object, _lineage: object = None) -> int:
                return plan_resume_round(
                    comments, effective_config, min_round_number=scope_start,
                    trusted_actor=trusted_actor,
                )

            resolve_plan_board_state(
                runner,
                config=config,
                issue_number=issue_number,
                comments=comments,
                admitted=(comments, trusted_actor),
                drift_error=plan_drift_error(issue_number),
                floor_context=f"Issue #{issue_number} plan reviewer board",
                # An empty fresh scope re-enters at its start round, where a
                # pending signed amendment must take effect (#1390).
                resume_round=(
                    scope_resume_round if classification.state == "empty-scope" else None
                ),
                floor_refusal_hint=plan_floor_refusal_hint(
                    issue_number, lambda: scope_resume_round(config),
                ),
            )
            return
    plan = derive_plan_verification_context(
        comments, issue_number=issue_number, trusted_actor=trusted_actor,
    )
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


def require_recoverable_plan_approval(
    runner: object | None,
    *,
    config: object,
    issue_context: object,
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    error_message: str,
) -> None:
    """Managed-CI no-handoff recovery and strict qualification refuse to bind
    a plan the shared approval classification does not approve (#1377).

    Uses the same read-only inputs as the issue preflight and plan startup:
    current issue and authoritative-parent requirements against each counted
    approval's binding, a reconstructible clear ledger, inherited
    admissibility and any pending signed supersession.
    """
    inputs = build_plan_approval_inputs(runner, config=config, issue_context=issue_context)
    classification = classify_plan_approval(
        comments,
        config=config,
        issue_number=getattr(issue_context, "number"),
        trusted_actor=trusted_actor,
        inputs=inputs,
        poster_completion_proven=True,
        execution_mode_guard=False,
    )
    if not classification.approved:
        raise AgentLoopError(f"{error_message} ({classification.reason})")
