"""Per-phase progress resolution for a staged (`implement-by-phase`) parent (#918).

A staged parent used to dispatch its first phase unconditionally, so a topology
larger than one phase never advanced: once stage-1's handoff existed the parent
printed a resume hint for that child forever, even after the child was closed
and its PR merged.

This module resolves the *current* phase instead.  It reconciles the
materialized child mapping against the parent's recorded phase handoff records,
authenticates every recorded phase against live child state, and reports an
ordered per-phase status the dispatcher selects from.  Every gap - a missing or
duplicated child number, a handoff recorded out of order, a closed child without
merged-PR evidence - stops the run with a diagnostic rather than skipping a
phase or re-dispatching an ambiguous one.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from .config import AgentLoopConfig
from .decomposition import (
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    RetainedParentScope,
    find_phase_implementation_handoffs_for_parent,
)
from .errors import AgentLoopError
from .github import get_issue_context, get_issue_state
from .issue_pr_handoff import authenticate_canonical_issue_pr
from .protocol import ExecutionAllocation
from .runner import Runner

AGENT_AUTOMATION = "agent-pr"

STATUS_NOT_DISPATCHED = "not-dispatched"
STATUS_IN_PROGRESS = "in-progress"
STATUS_COMPLETE = "complete"
STATUS_HUMAN_PENDING = "human-pending"


@dataclass(frozen=True)
class StagedTopologyOutcome:
    """What a staged decomposition resolved, beyond the materialized children.

    The dispatcher cannot obtain stage identity, plan identity, or the parent's
    own obligations from a bare child sequence, especially on the adopted and
    legacy paths where the recorded decomposition summary - not a normalized
    topology - is the only source.  Decomposition therefore returns this
    explicit contract.
    """

    created: tuple[CreatedPhaseIssue, ...]
    stage_ids: tuple[str, ...]
    automations: tuple[str, ...]
    plan_hash: str
    mode: str
    topology_source: str
    retained_parent_scope: RetainedParentScope | None = None
    final_integration_work: ExecutionAllocation | None = None

    @property
    def retained_parent_status(self) -> str:
        return self.retained_parent_scope.status if self.retained_parent_scope else "none"

    @property
    def final_integration_status(self) -> str:
        return self.final_integration_work.status if self.final_integration_work else "none"


@dataclass(frozen=True)
class PhaseProgress:
    """Resolved state of one phase of a staged topology."""

    phase_index: int
    stage_id: str
    automation: str
    created: CreatedPhaseIssue
    handoff: PhaseImplementationHandoffMetadata | None
    child_issue_number: int | None
    child_state: str | None
    pr_number: int | None
    pr_state: str | None
    status: str

    @property
    def is_human(self) -> bool:
        return self.automation != AGENT_AUTOMATION


def _filter_outcome_handoffs(
    handoffs: Sequence[PhaseImplementationHandoffMetadata],
    *,
    outcome: StagedTopologyOutcome,
) -> tuple[PhaseImplementationHandoffMetadata, ...]:
    """Keep only the handoffs bound to this run's plan identity and mode.

    ``find_phase_implementation_handoffs_for_parent`` is unfiltered across plan
    hashes and modes.  Binding to the outcome's own ``plan_hash``/``mode`` keeps
    a legacy rerun exactly as narrow as today's explicit single-phase lookup, so
    a record from an unrelated plan is invisible to progress instead of tripping
    the ordered-prefix invariant.
    """
    return tuple(
        handoff
        for handoff in handoffs
        if handoff.plan_hash == outcome.plan_hash and handoff.mode == outcome.mode
    )


def _reconcile(
    outcome: StagedTopologyOutcome,
    handoffs: Sequence[PhaseImplementationHandoffMetadata],
    *,
    parent_issue: int,
) -> dict[int, PhaseImplementationHandoffMetadata]:
    """Fail closed on any disagreement between the topology and its records."""
    phase_count = len(outcome.created)
    if phase_count == 0:
        raise AgentLoopError(
            f"Issue #{parent_issue} staged topology records no phases; "
            "repair the recorded decomposition before rerunning."
        )
    if len(outcome.stage_ids) != phase_count or len(outcome.automations) != phase_count:
        raise AgentLoopError(
            f"Issue #{parent_issue} staged topology is internally inconsistent: "
            f"{phase_count} children, {len(outcome.stage_ids)} stage ids, "
            f"{len(outcome.automations)} automations."
        )
    seen_children: dict[int, int] = {}
    for index, created in enumerate(outcome.created, start=1):
        stage_id = outcome.stage_ids[index - 1]
        if created.issue_number is None:
            raise AgentLoopError(
                f"Issue #{parent_issue} staged phase {index} (`{stage_id}`) has no child issue "
                "number recorded; repair the decomposition summary before rerunning."
            )
        previous = seen_children.get(created.issue_number)
        if previous is not None:
            raise AgentLoopError(
                f"Issue #{parent_issue} staged topology maps child issue "
                f"#{created.issue_number} to both phase {previous} "
                f"(`{outcome.stage_ids[previous - 1]}`) and phase {index} (`{stage_id}`); "
                "repair the decomposition summary before rerunning."
            )
        seen_children[created.issue_number] = index

    by_index: dict[int, PhaseImplementationHandoffMetadata] = {}
    for handoff in handoffs:
        index = handoff.phase_index
        if index < 1 or index > phase_count:
            raise AgentLoopError(
                f"Issue #{parent_issue} carries a phase handoff record for phase {index}, "
                f"which is outside its {phase_count}-phase topology; repair the recorded "
                "handoff before rerunning."
            )
        stage_id = outcome.stage_ids[index - 1]
        existing = by_index.get(index)
        if existing is not None:
            # At most one record may exist per index.  An identical duplicate is
            # rejected too: two durable records for one phase is an ambiguous
            # history the parent must not silently collapse.
            shape = "divergent" if existing != handoff else "duplicate"
            raise AgentLoopError(
                f"Issue #{parent_issue} carries {shape} phase handoff records for phase "
                f"{index} (`{stage_id}`): child issues #{existing.child_issue_number} and "
                f"#{handoff.child_issue_number}; repair the records before rerunning."
            )
        expected_child = outcome.created[index - 1].issue_number
        if handoff.child_issue_number != expected_child:
            raise AgentLoopError(
                f"Issue #{parent_issue} phase {index} (`{stage_id}`) is materialized as child "
                f"issue #{expected_child} but its handoff record names child issue "
                f"#{handoff.child_issue_number}; repair the records before rerunning."
            )
        if handoff.stage_id is not None and handoff.stage_id != stage_id:
            raise AgentLoopError(
                f"Issue #{parent_issue} phase {index} resolves stage id `{stage_id}` but its "
                f"handoff record records `{handoff.stage_id}`; repair the records before "
                "rerunning."
            )
        if outcome.automations[index - 1] != AGENT_AUTOMATION:
            raise AgentLoopError(
                f"Issue #{parent_issue} phase {index} (`{stage_id}`) is a "
                f"{outcome.automations[index - 1]} stage, but a phase handoff record exists for "
                f"child issue #{handoff.child_issue_number}; human-owned stages are never "
                "dispatched, so repair the recorded handoff before rerunning."
            )
        by_index[index] = handoff
    return by_index


@contextmanager
def _phase_read_context(
    *, parent_issue: int, phase_index: int, stage_id: str, child_issue_number: int
):
    """Attach parent/phase/stage/child context to any failed live read.

    The underlying readers only know the child issue or PR they were asked
    about, so an unreadable issue state or an unauthenticatable canonical PR
    record would otherwise reach the operator without naming which staged
    parent and stage is blocked.  The original diagnostic is preserved as the
    message tail and as the exception cause.
    """
    try:
        yield
    except AgentLoopError as exc:
        raise AgentLoopError(
            f"Issue #{parent_issue} could not authenticate phase {phase_index} "
            f"(`{stage_id}`) from child issue #{child_issue_number}: {exc} "
            "Repair that child's issue state or its canonical issue-to-PR handoff "
            "record, then rerun the parent."
        ) from exc
    except (ValueError, TypeError, AttributeError, KeyError) as exc:
        # Defence in depth: the GitHub readers translate unreadable payloads to
        # AgentLoopError, but a residual parsing failure must still name the
        # blocked parent and stage rather than escaping as a bare decode error.
        raise AgentLoopError(
            f"Issue #{parent_issue} could not authenticate phase {phase_index} "
            f"(`{stage_id}`) from child issue #{child_issue_number}: {exc} "
            "Repair that child's issue state or its canonical issue-to-PR handoff "
            "record, then rerun the parent."
        ) from exc


def _read_child_issue_state(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    phase_index: int,
    stage_id: str,
    child_issue_number: int,
) -> str:
    with _phase_read_context(
        parent_issue=parent_issue,
        phase_index=phase_index,
        stage_id=stage_id,
        child_issue_number=child_issue_number,
    ):
        return get_issue_state(runner, config=config, issue_number=child_issue_number)


def _resolve_agent_phase(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    phase_index: int,
    stage_id: str,
    child_issue_number: int,
) -> tuple[str, str, int | None, str | None]:
    """Authenticate one recorded agent phase; returns (status, child, pr, pr state)."""
    with _phase_read_context(
        parent_issue=parent_issue,
        phase_index=phase_index,
        stage_id=stage_id,
        child_issue_number=child_issue_number,
    ):
        child_state = get_issue_state(runner, config=config, issue_number=child_issue_number)
        child_context = get_issue_context(
            runner, config=config, issue_number=child_issue_number
        )
        authenticated = authenticate_canonical_issue_pr(
            runner,
            config=config,
            issue_number=child_issue_number,
            issue_context=child_context,
        )
    pr_number = authenticated.pr_number if authenticated is not None else None
    pr_state = authenticated.state if authenticated is not None else None
    if child_state == "OPEN":
        if authenticated is None or pr_state == "OPEN":
            return STATUS_IN_PROGRESS, child_state, pr_number, pr_state
        raise AgentLoopError(
            f"Issue #{parent_issue} phase {phase_index} (`{stage_id}`) child issue "
            f"#{child_issue_number} is OPEN but its canonical implementation PR "
            f"#{pr_number} is {pr_state}, not OPEN. Resuming that child would be rejected, "
            "so the staged parent stops here: reopen or supersede PR "
            f"#{pr_number}, or close child issue #{child_issue_number} once its work is "
            "merged, then rerun the parent."
        )
    if pr_state == "MERGED":
        return STATUS_COMPLETE, child_state, pr_number, pr_state
    observed = f"PR #{pr_number} is {pr_state}" if authenticated is not None else (
        "no canonical issue-to-PR handoff record exists"
    )
    raise AgentLoopError(
        f"Issue #{parent_issue} phase {phase_index} (`{stage_id}`) child issue "
        f"#{child_issue_number} is CLOSED but {observed}, so the phase cannot be "
        "authenticated as delivered. Record or repair the child's canonical "
        "issue-to-PR handoff and merge its implementation PR, or reopen child issue "
        f"#{child_issue_number}, then rerun the parent."
    )


def resolve_staged_phase_progress(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    parent_issue: int,
    parent_comments: Sequence[object],
    outcome: StagedTopologyOutcome,
) -> tuple[PhaseProgress, ...]:
    """Resolve per-phase progress for a staged parent, in topology order.

    Phases are resolved in order and resolution stops at the first phase that
    is not complete: a later phase has nothing to contribute to the selection
    and reading its child state would cost a GitHub call for no decision.  The
    ordered-prefix invariant is still enforced across every later phase from
    the recorded comments alone.
    """
    handoffs = _filter_outcome_handoffs(
        find_phase_implementation_handoffs_for_parent(
            parent_comments, parent_issue=parent_issue
        ),
        outcome=outcome,
    )
    by_index = _reconcile(outcome, handoffs, parent_issue=parent_issue)

    progress: list[PhaseProgress] = []
    settled = False
    first_incomplete: tuple[int, str] | None = None
    for index, created in enumerate(outcome.created, start=1):
        stage_id = outcome.stage_ids[index - 1]
        automation = outcome.automations[index - 1]
        handoff = by_index.get(index)
        if settled:
            if handoff is not None:
                assert first_incomplete is not None
                raise AgentLoopError(
                    f"Issue #{parent_issue} recorded a phase handoff for phase {index} "
                    f"(`{stage_id}`, child issue #{handoff.child_issue_number}) while phase "
                    f"{first_incomplete[0]} (`{first_incomplete[1]}`) is not complete. Staged "
                    "phases are delivered in order, so the parent stops instead of dispatching "
                    "or resuming either phase: finish or repair the earlier phase, or remove "
                    "the out-of-order handoff record, then rerun the parent."
                )
            progress.append(
                PhaseProgress(
                    phase_index=index,
                    stage_id=stage_id,
                    automation=automation,
                    created=created,
                    handoff=None,
                    child_issue_number=created.issue_number,
                    child_state=None,
                    pr_number=None,
                    pr_state=None,
                    status=STATUS_NOT_DISPATCHED,
                )
            )
            continue
        child_issue_number = created.issue_number
        assert child_issue_number is not None  # established by reconciliation
        if automation != AGENT_AUTOMATION:
            # Closing the human child is the attestation that the required
            # operator work and its recorded remark are done: the parent has no
            # other durable signal for operator-owned work, and without this a
            # mid-topology human stage would block the topology forever.
            child_state = _read_child_issue_state(
                runner,
                config=config,
                parent_issue=parent_issue,
                phase_index=index,
                stage_id=stage_id,
                child_issue_number=child_issue_number,
            )
            status = (
                STATUS_COMPLETE if child_state == "CLOSED" else STATUS_HUMAN_PENDING
            )
            progress.append(
                PhaseProgress(
                    phase_index=index,
                    stage_id=stage_id,
                    automation=automation,
                    created=created,
                    handoff=None,
                    child_issue_number=child_issue_number,
                    child_state=child_state,
                    pr_number=None,
                    pr_state=None,
                    status=status,
                )
            )
        elif handoff is None:
            progress.append(
                PhaseProgress(
                    phase_index=index,
                    stage_id=stage_id,
                    automation=automation,
                    created=created,
                    handoff=None,
                    child_issue_number=child_issue_number,
                    child_state=None,
                    pr_number=None,
                    pr_state=None,
                    status=STATUS_NOT_DISPATCHED,
                )
            )
        else:
            status, child_state, pr_number, pr_state = _resolve_agent_phase(
                runner,
                config=config,
                parent_issue=parent_issue,
                phase_index=index,
                stage_id=stage_id,
                child_issue_number=child_issue_number,
            )
            progress.append(
                PhaseProgress(
                    phase_index=index,
                    stage_id=stage_id,
                    automation=automation,
                    created=created,
                    handoff=handoff,
                    child_issue_number=child_issue_number,
                    child_state=child_state,
                    pr_number=pr_number,
                    pr_state=pr_state,
                    status=status,
                )
            )
        if progress[-1].status != STATUS_COMPLETE:
            settled = True
            first_incomplete = (index, stage_id)
    return tuple(progress)


def select_current_phase(progress: Sequence[PhaseProgress]) -> PhaseProgress | None:
    """Return the first phase that is not complete, or ``None`` when all are."""
    for phase in progress:
        if phase.status != STATUS_COMPLETE:
            return phase
    return None


def render_phase_status_line(phase: PhaseProgress) -> str:
    """Render one progress-derived staged child-work status line."""
    child = f"#{phase.child_issue_number}"
    if phase.status == STATUS_COMPLETE:
        if phase.is_human:
            return f"{phase.stage_id}: complete ({child}, human attestation)"
        return f"{phase.stage_id}: complete ({child}, PR #{phase.pr_number})"
    if phase.status == STATUS_IN_PROGRESS:
        return f"{phase.stage_id}: in progress ({child})"
    if phase.status == STATUS_HUMAN_PENDING:
        return f"{phase.stage_id}: pending human work ({child})"
    return f"{phase.stage_id}: pending"
