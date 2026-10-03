"""Inherited-matrix replan state, child-plan rebind/supersession and
structured plan-round assembly.

Extracted from ``orchestrator.py`` as a pure move (#1181, #1198); the
orchestrator re-exports every name defined here.
"""

from __future__ import annotations

import collections
import functools
import re
from collections.abc import (
    Mapping,
    Sequence,
)
from dataclasses import dataclass
from typing import Literal
from .config import AgentLoopConfig
from .decomposition import (
    approved_plan_hash,
    validate_separately_planned_child_matrix,
    AuthorizedReplanLineage,
    ChildPlanRebindRecord,
    authorized_replan_lineage,
    collect_child_plan_supersessions,
    find_child_plan_rebind_records,
    format_child_plan_rebind_section,
    format_child_plan_supersession_comment,
    InheritedMatrixBinding,
    risk_matrix_row_ids_for_owner,
)
from .protocol import EXECUTION_DISPOSITION_PLANNING
from .errors import (
    AgentInvocationError,
    AgentLoopError,
    DeterministicPlanValidationExhaustion,
)
from .github import (
    IssueContext,
    get_issue_context,
    post_issue_comment,
    post_trusted_issue_comment,
    post_verified_trusted_issue_round_comment,
    post_verified_trusted_issue_protocol_comment,
    note_host_footer_observed,
    resolve_authenticated_github_actor,
)
from .issue_pr_handoff import (
    authenticate_canonical_issue_pr,
    format_issue_pr_handoff_comment,
    resolve_issue_pr_handoff_lineage,
)
from .logging import log
from .protocol import (
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    StructuredPlanState,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    parse_human_requirements_acknowledgement,
    validate_human_requirements_acknowledgement,
)
from .runner import Runner
from .comment_rendering import (
    _extract_plan_human_requirements_block,
    render_public_agent_comment,
)
from .round_state import (
    ApprovedPlanContext,
    PostedRoundMetadata,
    _attach_round_metadata,
    _extract_round_metadata_records,
    recover_approved_plan_context,
    PlanValidationDiagnosticPayload,
    PlanValidationDiagnosticTransport,
    encode_plan_validation_diagnostic_body,
    has_plan_validation_diagnostic_marker,
    recover_plan_validation_diagnostic,
    sanitize_plan_validation_diagnostic,
)
from .round_transport import (
    prepare_round_comment,
    round_comment_fits,
)
from .plan_assembly import decode_assembled_plan_sidecar
from .plan_growth import (
    STRUCTURAL_SIGNALS,
    PlanGrowthAssessment,
    PlanGrowthThresholds,
    assess_plan_growth,
    plan_justification,
    plan_strategy,
)
from .protocol_markers import (
    TrustedBody,
    scan_reserved_markers,
)
from .panel_evidence import _plan_growth_verdict_for_hash
from .execution_policy import (
    _fresh_phase_marker_payload,
    _resolve_fresh_child_provenance,
    _child_resume_hint,
    _infer_staged_parent_issue,
)


def _plan_validation_contract_versions(
    *,
    require_execution_strategy_contract: bool,
    require_risk_test_matrix_contract: bool,
) -> tuple[int, int | None, int | None]:
    return (
        1,
        1 if require_execution_strategy_contract else None,
        1 if require_risk_test_matrix_contract else None,
    )


# Inherited-obligation replans per coder round: two replans, three candidates.
MAX_INHERITED_MATRIX_REPLANS = 2


@dataclass(frozen=True)
class _InheritedReplanDiagnostic:
    """In-memory correction context for an unpublished rejected candidate."""

    diagnostic: str
    failure_attempt: int
    candidate_digest: str


def _inherited_matrix_binding(
    *, parent_issue: int, stage_id: str, parent_plan_context: ApprovedPlanContext
) -> InheritedMatrixBinding | None:
    """Binding for a child plan-first cycle, or ``None`` when nothing is inherited."""
    payload = (
        parent_plan_context.risk_test_matrix_payload
        if parent_plan_context.matrix_available else None
    )
    if not isinstance(payload, dict) or not risk_matrix_row_ids_for_owner(payload, stage_id):
        return None
    return InheritedMatrixBinding(
        parent_issue=parent_issue, stage_id=stage_id, parent_matrix=payload
    )


@dataclass(frozen=True)
class _PlanningChildBinding:
    """Identity of a fresh planning child, captured once per entry path (#936)."""

    child_issue: int
    parent_issue: int
    stage_id: str
    parent_plan_context: ApprovedPlanContext


@dataclass(frozen=True)
class _PlanSupersessionBinding:
    """A signed re-plan authorization bound to the existing canonical PR (#936)."""

    digest: str
    superseded_hash: str
    pr_number: int
    parent_issue: int
    stage_id: str
    parent_plan_context: ApprovedPlanContext
    # The signed record's human rationale.  It is the instruction the forced
    # revision turn carries, so an admissible plan is re-planned for the
    # reason a human gave rather than re-approved unchanged (#985).
    rationale: str = ""


def _child_plan_admissibility_failure(
    parent_plan_context: ApprovedPlanContext,
    child_plan_context: ApprovedPlanContext,
    *,
    stage_id: str,
) -> str | None:
    """Judge a recorded approved child plan against its inherited parent rows.

    The one admissibility rule shared by issue routing, the plan-loop guard,
    PR-loop entry, and mid-run plan adoption.  Returns the sanitized
    weakening diagnostic, or ``None`` for an admissible plan.
    """
    try:
        validate_separately_planned_child_matrix(
            parent_plan_context.risk_test_matrix_payload
            if parent_plan_context.matrix_available else None,
            child_plan_context.risk_test_matrix_payload
            if child_plan_context.matrix_available else None,
            execution_owner=stage_id,
        )
    except AgentLoopError as exc:
        return sanitize_plan_validation_diagnostic(str(exc))
    return None


def _child_plan_supersession_route(
    *, child_issue: int, parent_issue: int, stage_id: str, superseded_plan_hash: str
) -> str:
    """Route-forward text naming the signed record and the issue-mode rerun."""
    template = format_child_plan_supersession_comment(
        child_issue=child_issue,
        parent_issue=parent_issue,
        stage_id=stage_id,
        superseded_plan_hash=superseded_plan_hash,
        rationale="<why this approved child plan must be re-planned>",
    )
    return (
        f"Supported route: approved child plan {superseded_plan_hash} is already bound to an "
        f"implementation PR, so it can only be re-planned under a signed human authorization. "
        f"Post this signed record as a comment on child issue #{child_issue} (fill in the "
        "rationale, keep the signature line, and leave the record in place afterwards):\n\n"
        f"{template}\n\n"
        f"Then rerun `{_child_resume_hint(child_issue, EXECUTION_DISPOSITION_PLANNING)}`. The "
        "child is re-planned, and after approval the same PR is rebound to the revised plan; "
        "`agent-loop pr` never re-plans or rebinds. Re-planning continues the child's existing "
        "round numbering, so a higher --max-rounds may be needed."
    )


def verify_child_plan_rebind(
    child_comments: Sequence[object],
    *,
    repo: str,
    parent_plan_context: ApprovedPlanContext,
    child_issue: int,
    parent_issue: int,
    stage_id: str,
    pr_number: int,
    require_admissible: bool = True,
) -> ApprovedPlanContext | None:
    """Verify a planning child's same-PR plan-replacement handoff.

    Returns ``None`` when the PR's bound plan never came from a plan
    replacement, and the verified replacement plan otherwise.  Every path on
    which a replacement plan can become a PR's plan context calls this; a
    missing, inconsistent, unauthorized, or inadmissible rebind raises a
    human-repair diagnostic and nothing is ever posted to correct it.

    The check follows the most recent plan-changing handoff edge, which a
    later closing-ID superset never erases.  Provenance and current-contract
    admissibility are separate: ``require_admissible=False`` verifies only how
    the plan became the binding, so a legitimately rebound plan that a later
    contract tightening made inadmissible can still be superseded, while an
    unverified replacement can never be laundered through a new authorization.
    """
    lineage = resolve_issue_pr_handoff_lineage(
        child_comments, issue_number=child_issue, repo=repo
    )
    if lineage is None or lineage.replaced is None or lineage.replacement is None:
        return None
    handoff = lineage.replacement
    replaced = lineage.replaced

    def fail(reason: str) -> AgentLoopError:
        return AgentLoopError(
            f"Human repair required: child issue #{child_issue} carries a same-PR approved-plan "
            f"replacement handoff ({replaced.plan_hash} -> {handoff.plan_hash}) for PR "
            f"#{handoff.pr_number} that cannot be verified: {reason}. No reviewer, coder, "
            "qualification, or merge step ran and no corrective record was posted. Restore the "
            "signed child-plan supersession record and the rebind comment, or remove the "
            "unverifiable handoff comment, then rerun."
        )

    if handoff.pr_number != pr_number:
        raise fail(f"it names PR #{handoff.pr_number}, not PR #{pr_number}")
    records = [
        record
        for record in find_child_plan_rebind_records(child_comments)
        if record.comment_index == lineage.replacement_comment_index
    ]
    if len(records) != 1:
        raise fail("its comment does not carry exactly one rebind audit record")
    record = records[0]
    if (
        record.child_issue != child_issue
        or record.pr_number != handoff.pr_number
        or record.new_plan_hash != handoff.plan_hash
        or record.superseded_plan_hash != replaced.plan_hash
    ):
        raise fail(
            "its rebind audit record disagrees with the handoff on child issue, PR number, or "
            "plan hashes"
        )
    supersessions = collect_child_plan_supersessions(
        child_comments, child_issue=child_issue, parent_issue=parent_issue, stage_id=stage_id
    )
    replan = authorized_replan_lineage(
        child_comments,
        superseded_hash=record.superseded_plan_hash,
        digest=record.plan_supersession_digest,
        supersessions=supersessions,
        through_round=record.approved_round,
    )
    if isinstance(replan, str):
        raise fail(replan)
    if (
        replan.first_round != record.first_replan_round
        or replan.latest_round != record.approved_round
        or replan.latest_plan_hash != record.new_plan_hash
    ):
        raise fail(
            "its rebind audit record disagrees with the digest-bound re-plan rounds on the "
            "first re-plan round, the approved round, or the approved plan hash"
        )
    replacement = recover_approved_plan_context(child_comments, expected_hash=record.new_plan_hash)
    if not replacement.is_available:
        raise fail(f"replacement plan {record.new_plan_hash} is not recoverable")
    if not require_admissible:
        return replacement
    inadmissible = _child_plan_admissibility_failure(
        parent_plan_context, replacement, stage_id=stage_id
    )
    if inadmissible is not None:
        raise fail(f"the replacement plan is itself inadmissible.\n{inadmissible}")
    return replacement


def verified_retired_child_plan_hashes(
    child_comments: Sequence[object],
    *,
    repo: str,
    parent_plan_context: ApprovedPlanContext,
    child_issue: int,
    parent_issue: int,
    stage_id: str,
    pr_number: int,
) -> frozenset[str]:
    """Approved plans that a verified signed supersession chain replaced (#988).

    A rebind moves the PR binding to the replacement plan, but the execution
    decision recorded under the superseded plan stays on the issue.  That
    decision is history, not a competing topology, exactly when the plan it
    names was replaced through a verified signed re-plan.

    The chain is the ordered sequence of plan-changing handoff edges of the
    live PR lineage, walked backwards from the live handoff.  The latest edge
    is verified by ``verify_child_plan_rebind`` (which raises on an
    unverifiable replacement).  Each earlier edge must be a real handoff
    transition whose own comment carries exactly one rebind audit record
    agreeing with it, and whose digest-bound re-plan lineage verifies.  A
    standalone audit record with no handoff transition retires nothing.  The
    walk stops at the first edge that does not verify, so an unexplained hash
    divergence keeps failing closed at the execution-decision check.
    """
    replacement = verify_child_plan_rebind(
        child_comments,
        repo=repo,
        parent_plan_context=parent_plan_context,
        child_issue=child_issue,
        parent_issue=parent_issue,
        stage_id=stage_id,
        pr_number=pr_number,
        require_admissible=False,
    )
    if replacement is None or not replacement.plan_hash:
        return frozenset()
    lineage = resolve_issue_pr_handoff_lineage(
        child_comments, issue_number=child_issue, repo=repo
    )
    if lineage is None or lineage.latest.pr_number != pr_number:
        return frozenset()
    supersessions = collect_child_plan_supersessions(
        child_comments, child_issue=child_issue, parent_issue=parent_issue, stage_id=stage_id
    )
    rebinds = find_child_plan_rebind_records(child_comments)
    retired: set[str] = set()
    current = lineage.latest.plan_hash
    for replaced, successor, comment_index in reversed(lineage.replacement_edges):
        if successor.plan_hash != current or successor.pr_number != pr_number:
            break
        records = [record for record in rebinds if record.comment_index == comment_index]
        if len(records) != 1:
            break
        record = records[0]
        if (
            record.child_issue != child_issue
            or record.pr_number != pr_number
            or record.new_plan_hash != successor.plan_hash
            or record.superseded_plan_hash != replaced.plan_hash
        ):
            break
        if record.superseded_plan_hash in retired or record.superseded_plan_hash == (
            lineage.latest.plan_hash
        ):
            break
        replan = authorized_replan_lineage(
            child_comments,
            superseded_hash=record.superseded_plan_hash,
            digest=record.plan_supersession_digest,
            supersessions=supersessions,
            through_round=record.approved_round,
        )
        if (
            isinstance(replan, str)
            or replan.first_round != record.first_replan_round
            or replan.latest_round != record.approved_round
            or replan.latest_plan_hash != record.new_plan_hash
        ):
            break
        retired.add(record.superseded_plan_hash)
        current = replaced.plan_hash
    return frozenset(retired)


def _managed_ci_retired_plan_hashes(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    parent_issue_context: IssueContext | None,
    pr_number: int,
) -> tuple[frozenset[str], IssueContext | None]:
    """Plans a verified signed re-plan retired, for a managed-CI resume (#993).

    A rebind leaves the managed-CI authorization recorded under the
    superseded plan on the PR, just as it leaves the execution decision on
    the issue (#988).  Both the explicit fresh grant and the ordinary resume
    carry this set on the handoff so those grants read as history.  Only a fresh decomposition child whose live handoff
    lineage carries a same-PR plan-replacement edge can retire anything; the
    retired set comes from ``verified_retired_child_plan_hashes``, so an
    unexplained plan divergence still retires nothing and keeps refusing.

    Returns the retired hashes and the (possibly newly fetched) parent issue
    context so the caller does not refetch it.
    """
    lineage = resolve_issue_pr_handoff_lineage(
        issue_context.comments, issue_number=issue_context.number, repo=config.repo
    )
    if lineage is None or lineage.replaced is None or lineage.latest.pr_number != pr_number:
        return frozenset(), parent_issue_context
    if _fresh_phase_marker_payload(issue_context) is None:
        return frozenset(), parent_issue_context
    staged_parent = _infer_staged_parent_issue(issue_context)
    if parent_issue_context is None and staged_parent is not None:
        parent_issue_context = get_issue_context(
            runner, config=config, issue_number=staged_parent
        )
    fresh_child = _resolve_fresh_child_provenance(
        issue_context=issue_context, parent_issue_context=parent_issue_context
    )
    if fresh_child is None or not fresh_child.route.is_planning:
        return frozenset(), parent_issue_context
    retired = verified_retired_child_plan_hashes(
        issue_context.comments,
        repo=config.repo,
        parent_plan_context=fresh_child.parent_plan_context,
        child_issue=issue_context.number,
        parent_issue=fresh_child.parent_issue,
        stage_id=fresh_child.stage_id,
        pr_number=pr_number,
    )
    return retired, parent_issue_context


def _require_authorized_replan_state(
    comments: Sequence[object],
    *,
    issue_number: int,
    plan_supersession: _PlanSupersessionBinding,
    latest_plan: str | None,
    latest_round: int | None,
) -> AuthorizedReplanLineage | None:
    """Fail closed unless the latest plan round belongs to the authorized re-plan.

    Returns ``None`` when the latest plan round is the superseded plan itself
    and no digest-bound round exists yet (the enforced revision runs next), or
    the valid digest-bound lineage.  Anything else is never revised,
    approved, or rebound.
    """
    supersessions = collect_child_plan_supersessions(
        comments,
        child_issue=issue_number,
        parent_issue=plan_supersession.parent_issue,
        stage_id=plan_supersession.stage_id,
    )
    lineage = authorized_replan_lineage(
        comments,
        superseded_hash=plan_supersession.superseded_hash,
        digest=plan_supersession.digest,
        supersessions=supersessions,
    )
    if isinstance(lineage, AuthorizedReplanLineage):
        return lineage
    bound_round_exists = any(
        record.metadata.role == "coder"
        and record.metadata.plan_supersession_superseded_hash
        == plan_supersession.superseded_hash
        for record in _extract_round_metadata_records(comments, flow="plan")
    )
    if (
        latest_plan is not None
        and not bound_round_exists
        and approved_plan_hash(latest_plan) == plan_supersession.superseded_hash
    ):
        return None
    raise AgentLoopError(
        f"Human repair required: issue #{issue_number} has a signed child-plan supersession for "
        f"approved plan {plan_supersession.superseded_hash}, but its latest plan round "
        f"({'round ' + str(latest_round) if latest_round is not None else 'none reconstructable'}"
        f"{', plan ' + approved_plan_hash(latest_plan) if latest_plan is not None else ''}) is "
        f"not part of the authorized re-plan: {lineage}. A plan outside the digest-bound "
        "lineage is never revised, approved, or rebound, and no agent was invoked. Remove the "
        "offending plan round comment(s) or restore the signed record, then rerun."
    )


@dataclass(frozen=True)
class _PlanContractShape:
    """The structural contract a plan asks a PR to satisfy (#1013)."""

    steps: tuple[str, ...]
    matrix_row_ids: tuple[str, ...]
    strategy: str | None
    # Every execution-recommendation field value, path-labelled: work or
    # topology the PR must satisfy even when steps and rows hold.
    commitments: tuple[str, ...] = ()
    # Each matrix row's path-labelled field values, keyed by row ID, so a
    # row strengthened under an unchanged ID is still visible.
    matrix_row_values: tuple[tuple[str, tuple[str, ...]], ...] = ()


# Bounds on the rebind expansion notice so a large re-plan cannot flood it.
_REBIND_EXPANSION_LIST_LIMIT = 5
_REBIND_EXPANSION_ITEM_CHARS = 160
_REBIND_EXPANSION_HEADING = "### Rebound PR contract expanded"


# Keys that name an element of a contract list; they become part of the
# element's path instead of a separate value.
_CONTRACT_ELEMENT_ID_KEYS = ("scope_item_id", "constraint_id", "stage_id", "row_id")


def _flatten_contract_values(value: object, path: str, *, skip: frozenset[str] = frozenset()) -> tuple[str, ...]:
    """Flatten a contract payload into path-labelled leaf values.

    Elements of object lists are addressed by their ID, so an added or
    changed value is a new label.  ``skip`` names top-level keys left out.
    """
    labels: list[str] = []

    def walk(node: object, node_path: str, top: bool) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node):
                if key in _CONTRACT_ELEMENT_ID_KEYS or (top and key in skip):
                    continue
                walk(node[key], f"{node_path}.{key}" if node_path else str(key), False)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                if isinstance(item, Mapping):
                    element = next(
                        (str(item[key]) for key in _CONTRACT_ELEMENT_ID_KEYS if key in item),
                        str(index),
                    )
                    walk(item, f"{node_path}[{element}]", False)
                else:
                    walk(item, node_path, False)
        else:
            labels.append(f"`{node_path}`: {node}")

    walk(value, path, True)
    return tuple(labels)


def _recommendation_commitments(recommendation: Mapping[str, object]) -> tuple[str, ...]:
    """Flatten every field of a recommendation into path-labelled values.

    The whole reviewed recommendation is the execution contract, so no field
    is singled out: scope, coupling, allocations, and every stage field
    (summary, order, dependencies, notes, risk, disposition and its
    rationale, ...) each become a label such as
    ``child_stages[stage-one].summary: ...``.  The strategy is reported on
    its own and is left out here.
    """
    return _flatten_contract_values(recommendation, "", skip=frozenset({"strategy"}))


def _matrix_row_values(rows: object) -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (
            str(row["row_id"]),
            _flatten_contract_values(row, f"risk_test_matrix[{row['row_id']}]"),
        )
        for row in (rows if isinstance(rows, list) else ())
        if isinstance(row, Mapping) and "row_id" in row
    )


def _plan_contract_shape(comments: Sequence[object], plan_hash: str) -> _PlanContractShape | None:
    """Recover a plan's structural contract from its plan round.

    Reads the authenticated assembled-state sidecar of the latest plan coder
    round whose canonical plan has ``plan_hash``.  ``None`` when no such round
    carries one; the caller treats that as "not comparable", never as a gap.
    """
    for record in reversed(_extract_round_metadata_records(comments, flow="plan")):
        metadata = record.metadata
        if (
            metadata.role != "coder"
            or metadata.canonical_plan is None
            or metadata.assembled_plan_sidecar is None
            or approved_plan_hash(metadata.canonical_plan) != plan_hash
        ):
            continue
        payload = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar).canonical_json
        steps = payload.get("plan_steps")
        matrix = payload.get("risk_test_matrix")
        row_values = _matrix_row_values(matrix.get("rows") if isinstance(matrix, Mapping) else None)
        recommendation = payload.get("execution_recommendation")
        if not isinstance(recommendation, Mapping):
            recommendation = {}
        strategy = recommendation.get("strategy")
        return _PlanContractShape(
            steps=tuple(str(step) for step in steps) if isinstance(steps, list) else (),
            matrix_row_ids=tuple(row_id for row_id, _values in row_values),
            strategy=strategy if isinstance(strategy, str) else None,
            commitments=_recommendation_commitments(recommendation),
            matrix_row_values=row_values,
        )
    return None


# Leading characters kept before an elision when the change sits deep in a
# long item, and characters of shared text kept just before the divergence.
_REBIND_EXPANSION_HEAD_CHARS = 60
_REBIND_EXPANSION_LEAD_CHARS = 30


def _flatten_contract_text(text: str) -> str:
    """Join non-empty lines so later lines of a multi-line item stay visible."""
    return " ⏎ ".join(part.strip() for part in text.splitlines() if part.strip())


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def _clip_contract_item(text: str, prior: Sequence[str] = ()) -> str:
    """Render one changed item within the bound, keeping the change visible.

    ``prior`` holds the superseded plan's items.  When the item shares a long
    prefix with its closest superseded counterpart, so that plain clipping
    would show only unchanged text, the rendering keeps a short head, elides
    the shared middle, and shows the text from just before the divergence.
    """
    line = _flatten_contract_text(text)
    limit = _REBIND_EXPANSION_ITEM_CHARS
    if len(line) > limit:
        divergence = max(
            (_common_prefix_length(line, _flatten_contract_text(item)) for item in prior),
            default=0,
        )
        head = _REBIND_EXPANSION_HEAD_CHARS
        if divergence >= limit - 1 - _REBIND_EXPANSION_LEAD_CHARS:
            tail = line[max(head, divergence - _REBIND_EXPANSION_LEAD_CHARS):]
            budget = limit - head - 3
            if len(tail) > budget:
                tail = tail[: budget - 1].rstrip() + "…"
            line = line[:head].rstrip() + " … " + tail
        else:
            line = line[: limit - 1].rstrip() + "…"
    # Plan text is quoted, never interpreted: an HTML comment opener would
    # otherwise let a quoted step pose as a record in the rebind comment.
    return line.replace("<!--", "&lt;!--")


def _added_contract_items(before: Sequence[str], after: Sequence[str]) -> list[str]:
    """Items of ``after`` beyond ``before``, compared on full normalized text."""
    remaining = collections.Counter(item.strip() for item in before)
    added: list[str] = []
    for item in after:
        key = item.strip()
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            added.append(item)
    return added


def _plan_contract_expansion(
    superseded: _PlanContractShape, replacement: _PlanContractShape
) -> list[tuple[str, list[str]]]:
    """Describe how the replacement plan materially expands the contract.

    Structural, not numeric on review findings: plan steps whose text is not
    in the superseded plan (even when another step was removed), new matrix
    rows, new or changed values in rows that keep their ID, new or changed
    execution-recommendation values, or a changed strategy.  Each entry is a
    summary line and the (unclipped) items it names.  An empty list means
    equivalent or narrower.  Detection compares full text; clipping is only
    for rendering.
    """
    expansion: list[tuple[str, list[str]]] = []
    new_steps = _added_contract_items(superseded.steps, replacement.steps)
    if new_steps:
        expansion.append(
            (
                f"{len(new_steps)} plan step(s) not in the superseded plan "
                f"(steps: {len(superseded.steps)} before, {len(replacement.steps)} after):",
                new_steps,
            )
        )
    known_rows = set(superseded.matrix_row_ids)
    new_rows = [row_id for row_id in replacement.matrix_row_ids if row_id not in known_rows]
    if new_rows:
        expansion.append(
            (
                f"{len(new_rows)} new risk/test matrix row(s):",
                [f"`{row_id}`" for row_id in new_rows],
            )
        )
    superseded_row_values = dict(superseded.matrix_row_values)
    changed_row_values = [
        label
        for row_id, values in replacement.matrix_row_values
        if row_id in superseded_row_values
        for label in _added_contract_items(superseded_row_values[row_id], values)
    ]
    if changed_row_values:
        expansion.append(
            (
                f"{len(changed_row_values)} new or changed value(s) in existing risk/test "
                "matrix rows:",
                changed_row_values,
            )
        )
    new_commitments = _added_contract_items(superseded.commitments, replacement.commitments)
    if new_commitments:
        expansion.append(
            (
                f"{len(new_commitments)} new or changed execution-recommendation value(s):",
                new_commitments,
            )
        )
    if superseded.strategy != replacement.strategy:
        expansion.append(
            (
                f"The execution recommendation changed from "
                f"`{superseded.strategy or 'none'}` to `{replacement.strategy or 'none'}`.",
                [],
            )
        )
    return expansion


def _render_contract_expansion_notice(
    expansion: Sequence[tuple[str, Sequence[str]]],
    *,
    pr_number: int,
    superseded_hash: str,
    plan_hash: str,
    include_items: bool,
    prior: Sequence[str] = (),
) -> str:
    lines = [
        _REBIND_EXPANSION_HEADING,
        "",
        f"PR #{pr_number} is rebound from approved plan `{superseded_hash}` to replacement "
        f"plan `{plan_hash}`. The replacement materially expands the contract the PR must "
        "satisfy, and the PR's implementation predates it:",
        "",
    ]
    for summary, items in expansion:
        lines.append(f"- {summary}")
        if not include_items:
            continue
        lines.extend(
            f"  - {_clip_contract_item(item, prior)}"
            for item in items[:_REBIND_EXPANSION_LIST_LIMIT]
        )
        if len(items) > _REBIND_EXPANSION_LIST_LIMIT:
            lines.append(f"  - …and {len(items) - _REBIND_EXPANSION_LIST_LIMIT} more")
    lines.extend(
        [
            "",
            "Review will now judge the existing diff against the replacement plan. If the "
            "remaining work no longer suits a single PR, consider decomposing it before "
            "continuing. This notice is informational and does not stop the run.",
        ]
    )
    return "\n".join(lines)


def _rebind_contract_expansion_notice(
    *,
    config: AgentLoopConfig,
    issue_number: int,
    comments: Sequence[object],
    pr_number: int,
    superseded_hash: str,
    plan_hash: str,
) -> str | None:
    """Informational notice text when a rebind materially expands the PR's contract (#1013).

    The PR's code was written against the superseded plan, so a larger
    replacement plan is a divergence reviewers would otherwise discover one
    finding at a time.  The text rides in the rebind comment itself, so an
    interruption can never leave a rebind without its notice.  This never
    blocks: any failure to compare is logged and ``None`` is returned, and
    the notice never carries a reserved record span.  Whether to decompose
    is left to the operator or the next planning cycle.
    """
    try:
        superseded = _plan_contract_shape(comments, superseded_hash)
        replacement = _plan_contract_shape(comments, plan_hash)
        if superseded is None or replacement is None:
            log(
                config,
                f"Issue #{issue_number}: rebind contract comparison skipped; the structured "
                f"state of plan {superseded_hash if superseded is None else plan_hash} is not "
                "recoverable from its plan round",
            )
            return None
        expansion = _plan_contract_expansion(superseded, replacement)
        if not expansion:
            return None
        log(
            config,
            f"Issue #{issue_number}: replacement plan {plan_hash} materially expands the "
            f"contract PR #{pr_number} must satisfy relative to superseded plan "
            f"{superseded_hash}: " + "; ".join(summary for summary, _items in expansion),
        )
        render = functools.partial(
            _render_contract_expansion_notice,
            expansion,
            pr_number=pr_number,
            superseded_hash=superseded_hash,
            plan_hash=plan_hash,
            prior=(
                *superseded.steps,
                *superseded.commitments,
                *(label for _row_id, values in superseded.matrix_row_values for label in values),
            ),
        )
        notice = render(include_items=True)
        if scan_reserved_markers(notice):
            # Plan text quoting a reserved record must not reach the trusted
            # rebind comment; the counts alone still name what grew.
            notice = render(include_items=False)
        return notice
    except Exception as exc:  # noqa: BLE001 - the notice must never block the rebind
        log(
            config,
            f"Issue #{issue_number}: rebind contract-expansion comparison failed: {exc}",
        )
        return None


_REBIND_GROWTH_HEADING = "### Rebound plan crossed plan-growth thresholds"
_REBIND_GROWTH_RATIONALE_CHARS = 400


def _render_rebind_growth_advisory(
    assessment: PlanGrowthAssessment,
    justification: Mapping[str, object] | None,
    *,
    pr_number: int,
    plan_hash: str,
    include_rationale: bool,
) -> str:
    lines = [
        _REBIND_GROWTH_HEADING,
        "",
        f"PR #{pr_number}'s code predates replacement plan `{plan_hash}`, a one-shot plan that "
        "crosses plan-growth threshold(s) "
        + ", ".join(f"`{signal}`" for signal in assessment.crossed)
        + f" ({assessment.describe()}). Check the remaining work against the one-shot choice.",
        "",
    ]
    if justification is None:
        lines.append(
            # Only the absence is authenticated: approval-time gate mode and
            # thresholds are not recorded, so no reason is asserted.
            "- No reviewed one-shot growth justification exists for this plan."
        )
    else:
        named = justification.get("crossed_signals")
        lines.append(
            "- Reviewed justification signals: "
            + ", ".join(f"`{signal}`" for signal in (named if isinstance(named, list) else []))
        )
        if include_rationale:
            rationale = _flatten_contract_text(str(justification.get("rationale", "")))
            if len(rationale) > _REBIND_GROWTH_RATIONALE_CHARS:
                rationale = (
                    rationale[: _REBIND_GROWTH_RATIONALE_CHARS - 1].rstrip()
                    + "… (complete rationale in the authenticated plan round)"
                )
            lines.append(f"- Rationale excerpt: {rationale.replace('<!--', '&lt;!--')}")
    lines.extend(
        [
            "",
            "This advisory is informational and does not stop the run.",
        ]
    )
    return "\n".join(lines)


def _rebind_growth_advisory(
    *,
    config: AgentLoopConfig,
    issue_number: int,
    comments: Sequence[object],
    pr_number: int,
    plan_hash: str,
) -> str | None:
    """Growth advisory for a rebound replacement plan (#886).

    Independent of contract expansion and of justification: a grown v1
    one-shot replacement gets the advisory whether it is justified (the
    normal case under the enforced gate) or not.  Size is taken from the
    authenticated coder round's stored canonical text, never a re-render.
    Revision count is not used.  Never blocks: failures are logged.
    """
    try:
        for record in reversed(_extract_round_metadata_records(comments, flow="plan")):
            metadata = record.metadata
            if (
                metadata.role != "coder"
                or metadata.canonical_plan is None
                or metadata.assembled_plan_sidecar is None
                or approved_plan_hash(metadata.canonical_plan) != plan_hash
            ):
                continue
            payload = decode_assembled_plan_sidecar(metadata.assembled_plan_sidecar).canonical_json
            if plan_strategy(payload) != "one-shot":
                return None
            assessment = assess_plan_growth(
                payload,
                rendered_chars=len(metadata.canonical_plan),
                revision_count=None,
                thresholds=PlanGrowthThresholds.from_config(config),
            )
            if not any(signal in STRUCTURAL_SIGNALS for signal in assessment.crossed):
                return None
            justification = plan_justification(payload)
            log(
                config,
                f"Issue #{issue_number}: rebound replacement plan {plan_hash} is one-shot and "
                f"crosses plan-growth signal(s) {', '.join(assessment.crossed)}",
            )
            render = functools.partial(
                _render_rebind_growth_advisory,
                assessment,
                justification,
                pr_number=pr_number,
                plan_hash=plan_hash,
            )
            advisory = render(include_rationale=True)
            if scan_reserved_markers(advisory):
                advisory = render(include_rationale=False)
            return advisory
        log(
            config,
            f"Issue #{issue_number}: rebind growth advisory skipped; the structured state of "
            f"plan {plan_hash} is not recoverable from its plan round",
        )
        return None
    except Exception as exc:  # noqa: BLE001 - the advisory must never block the rebind
        log(config, f"Issue #{issue_number}: rebind growth advisory failed: {exc}")
        return None


def _rebind_superseded_child_plan(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
    plan_supersession: _PlanSupersessionBinding,
    plan_hash: str,
) -> None:
    """Rebind the existing PR to the approved revision with one issue comment.

    The superseding handoff record and the rebind audit record share one
    comment, and no PR-side record is written, so an interruption leaves
    either the fully old or the fully new binding.  A contract-expansion
    notice (#1013), when the replacement plan grows the PR's contract, rides
    in the same comment so it can never be lost between two writes.  Idempotent: an existing
    verified rebind to ``plan_hash`` posts nothing.
    """
    replan = _require_authorized_replan_state(
        issue_context.comments,
        issue_number=issue_number,
        plan_supersession=plan_supersession,
        latest_plan=None,
        latest_round=None,
    )
    if replan is None or replan.latest_plan_hash != plan_hash:
        raise AgentLoopError(
            f"Human repair required: approved plan {plan_hash} on issue #{issue_number} is not "
            "the latest plan of the digest-bound re-plan lineage for superseded plan "
            f"{plan_supersession.superseded_hash}; PR #{plan_supersession.pr_number} was not "
            "rebound and nothing was posted."
        )
    handoff_lineage = resolve_issue_pr_handoff_lineage(
        issue_context.comments, issue_number=issue_number, repo=config.repo
    )
    current = handoff_lineage.latest if handoff_lineage is not None else None
    if current is None or current.pr_number != plan_supersession.pr_number:
        raise AgentLoopError(
            f"Human repair required: the issue-to-PR handoff for issue #{issue_number} no longer "
            f"names PR #{plan_supersession.pr_number}; the re-planned child plan was not rebound "
            "and no implementation turn was started."
        )
    authenticated = authenticate_canonical_issue_pr(
        runner, config=config, issue_number=issue_number, issue_context=issue_context
    )
    if (
        authenticated is None
        or authenticated.pr_number != plan_supersession.pr_number
        or authenticated.state != "OPEN"
    ):
        raise AgentLoopError(
            f"Human repair required: canonical PR #{plan_supersession.pr_number} for issue "
            f"#{issue_number} is "
            f"{authenticated.state if authenticated is not None else 'not recorded'}, not OPEN "
            "with the same number as the superseded handoff. The re-planned child plan was not "
            "rebound, no superseding handoff was posted, and no implementation turn was started; "
            "abandoning or replacing the existing PR is not supported."
        )
    if current.plan_hash == plan_hash:
        verify_child_plan_rebind(
            issue_context.comments,
            repo=config.repo,
            parent_plan_context=plan_supersession.parent_plan_context,
            child_issue=issue_number,
            parent_issue=plan_supersession.parent_issue,
            stage_id=plan_supersession.stage_id,
            pr_number=plan_supersession.pr_number,
        )
        return
    if current.plan_hash != plan_supersession.superseded_hash:
        raise AgentLoopError(
            f"Human repair required: the issue-to-PR handoff for issue #{issue_number} names plan "
            f"{current.plan_hash}, neither superseded plan {plan_supersession.superseded_hash} nor "
            f"approved revision {plan_hash}; nothing was posted."
        )
    handoff_lines = format_issue_pr_handoff_comment(
        issue_number=issue_number,
        pr_number=current.pr_number,
        pr_url=current.pr_url,
        pr_head_sha=current.pr_head_sha,
        flow="approved-plan-implementation",
        plan_hash=plan_hash,
        expected_closing_issue_ids=current.expected_closing_issue_ids,
        # The annotation the same-PR approved-plan replacement rule requires.
        # It never identifies the replacement: unchanged-ID rebinds all share it.
        supersedes_hash=current.contract_hash,
        plan_growth_verdict=_plan_growth_verdict_for_hash(
            config, plan_hash=plan_hash, comment_sources=(issue_context.comments,)
        ),
    ).split("\n")
    marker_position = next(
        index
        for index, line in enumerate(handoff_lines)
        if line.startswith("<!-- AGENT_ISSUE_PR_HANDOFF:")
    )
    rebind_section = format_child_plan_rebind_section(
        ChildPlanRebindRecord(
            child_issue=issue_number,
            pr_number=current.pr_number,
            superseded_plan_hash=plan_supersession.superseded_hash,
            new_plan_hash=plan_hash,
            plan_supersession_digest=plan_supersession.digest,
            first_replan_round=replan.first_round,
            approved_round=replan.latest_round,
        )
    )
    expansion_notice = _rebind_contract_expansion_notice(
        config=config,
        issue_number=issue_number,
        comments=issue_context.comments,
        pr_number=current.pr_number,
        superseded_hash=plan_supersession.superseded_hash,
        plan_hash=plan_hash,
    )
    growth_advisory = _rebind_growth_advisory(
        config=config,
        issue_number=issue_number,
        comments=issue_context.comments,
        pr_number=current.pr_number,
        plan_hash=plan_hash,
    )
    body = "\n".join(
        [
            *handoff_lines[:marker_position],
            *([expansion_notice, ""] if expansion_notice is not None else []),
            *([growth_advisory, ""] if growth_advisory is not None else []),
            rebind_section,
            *handoff_lines[marker_position:],
        ]
    )
    post_trusted_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=TrustedBody.canonical(
            body, expected_tokens=("AGENT_ISSUE_PR_HANDOFF", "AGENT_CHILD_PLAN_REBIND")
        ),
    )
    log(
        config,
        f"Issue #{issue_number}: rebound PR #{current.pr_number} from approved plan "
        f"{plan_supersession.superseded_hash} to {plan_hash} with one issue comment",
    )


def _route_child_plan_handoff(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    issue_context: IssueContext,
    fresh_child,
    handoff,
    child_plan_context: ApprovedPlanContext,
) -> _PlanSupersessionBinding | None:
    """Issue-mode routing for a planning child's handed-off plan (#936, #985).

    Admissibility and authorization are separate questions: admissibility asks
    whether the bound plan is broken, a signed supersession record asks whether
    a human authorized replacing it.  The signed records are therefore
    consulted for every bound plan.  Exactly one matching signed record:
    authenticate the canonical PR and return the binding that reopens planning,
    whether or not the plan is admissible.  No matching record: an admissible
    plan returns ``None`` so the PR resumes unchanged, and an inadmissible one
    fails closed with the record template, before any agent or write.
    """
    # How the handed-off plan became the binding is verified first and on
    # every branch: a signed authorization for the current hash never
    # excuses an unverified replacement that produced that hash.
    verify_child_plan_rebind(
        issue_context.comments,
        repo=config.repo,
        parent_plan_context=fresh_child.parent_plan_context,
        child_issue=issue_number,
        parent_issue=fresh_child.parent_issue,
        stage_id=fresh_child.stage_id,
        pr_number=handoff.pr_number,
        require_admissible=False,
    )
    failure = _child_plan_admissibility_failure(
        fresh_child.parent_plan_context, child_plan_context, stage_id=fresh_child.stage_id
    )
    ignored: list[str] = []
    supersessions = collect_child_plan_supersessions(
        issue_context.comments,
        child_issue=issue_number,
        parent_issue=fresh_child.parent_issue,
        stage_id=fresh_child.stage_id,
        ignored_sink=ignored,
    )
    for note in ignored:
        log(config, f"Issue #{issue_number}: {note}")
    matching = [
        record for record in supersessions if record.superseded_plan_hash == handoff.plan_hash
    ]
    if not matching:
        if failure is None:
            return None
        raise AgentLoopError(
            f"Approved child plan {handoff.plan_hash} on issue #{issue_number}, bound to PR "
            f"#{handoff.pr_number}, is inadmissible under the inherited-matrix contract. No agent "
            f"was invoked and nothing was posted.\n{failure}\n\n"
            + _child_plan_supersession_route(
                child_issue=issue_number,
                parent_issue=fresh_child.parent_issue,
                stage_id=fresh_child.stage_id,
                superseded_plan_hash=handoff.plan_hash,
            )
        )
    signed = matching[0]
    if config.dry_run:
        pr_number = handoff.pr_number
    else:
        authenticated = authenticate_canonical_issue_pr(
            runner, config=config, issue_number=issue_number, issue_context=issue_context
        )
        if (
            authenticated is None
            or authenticated.pr_number != handoff.pr_number
            or authenticated.state != "OPEN"
        ):
            raise AgentLoopError(
                f"Human repair required: issue #{issue_number} carries a signed child-plan "
                f"supersession for plan {handoff.plan_hash}, but canonical PR "
                f"#{handoff.pr_number} is "
                f"{authenticated.state if authenticated is not None else 'not recorded'}, not "
                "OPEN. Re-planning only rebinds the existing open PR; abandoning or replacing "
                "it is not supported. No agent was invoked."
            )
        pr_number = authenticated.pr_number
    log(
        config,
        f"Issue #{issue_number}: approved child plan {handoff.plan_hash} is "
        f"{'inadmissible' if failure is not None else 'admissible'}; "
        f"re-planning under signed supersession {signed.digest} ({signed.comment_locator}) "
        f"for PR #{pr_number}",
    )
    return _PlanSupersessionBinding(
        digest=signed.digest,
        superseded_hash=handoff.plan_hash,
        pr_number=pr_number,
        parent_issue=fresh_child.parent_issue,
        stage_id=fresh_child.stage_id,
        parent_plan_context=fresh_child.parent_plan_context,
        rationale=signed.rationale,
    )


def _require_admissible_pr_child_plan(
    child_comments: Sequence[object],
    *,
    config: AgentLoopConfig,
    binding: _PlanningChildBinding,
    child_plan_context: ApprovedPlanContext,
    pr_number: int,
) -> None:
    """PR-mode provenance gate for a planning child's bound plan (#936).

    PR mode never re-plans and never rebinds: an inadmissible binding fails
    closed naming the issue-mode supersession route, and a same-PR plan
    replacement must pass rebind verification before any reviewer runs.  A
    signed supersession naming the bound plan also fails closed (#985): the
    human authorized a re-plan, reviewers cannot judge the PR against a scope
    no plan approved, and a PR-mode coder turn can never produce the
    replacement, so reviewing would only repeat the same verdict.
    """
    verify_child_plan_rebind(
        child_comments,
        repo=config.repo,
        parent_plan_context=binding.parent_plan_context,
        child_issue=binding.child_issue,
        parent_issue=binding.parent_issue,
        stage_id=binding.stage_id,
        pr_number=pr_number,
        require_admissible=False,
    )
    failure = _child_plan_admissibility_failure(
        binding.parent_plan_context, child_plan_context, stage_id=binding.stage_id
    )
    if failure is not None:
        raise AgentLoopError(
            f"{failure}\n\nPR #{pr_number} is bound to approved child plan "
            f"{child_plan_context.plan_hash}, which is inadmissible; no reviewer ran. "
            + _child_plan_supersession_route(
                child_issue=binding.child_issue,
                parent_issue=binding.parent_issue,
                stage_id=binding.stage_id,
                superseded_plan_hash=child_plan_context.plan_hash or "",
            )
        )
    _reject_pending_child_plan_supersession(
        child_comments,
        binding=binding,
        plan_hash=child_plan_context.plan_hash,
        pr_number=pr_number,
        stopped="no reviewer ran",
    )


def _reject_pending_child_plan_supersession(
    child_comments: Sequence[object],
    *,
    binding: _PlanningChildBinding,
    plan_hash: str | None,
    pr_number: int,
    stopped: str,
) -> None:
    """Fail closed while a signed record authorizes replacing the bound plan (#985).

    Checked at PR entry and again on the freshly fetched child issue at
    qualification, so a record posted while reviewers ran cannot be bypassed.
    """
    pending = [
        record
        for record in collect_child_plan_supersessions(
            child_comments,
            child_issue=binding.child_issue,
            parent_issue=binding.parent_issue,
            stage_id=binding.stage_id,
        )
        if record.superseded_plan_hash == plan_hash
    ]
    if pending:
        raise AgentLoopError(
            f"PR #{pr_number} is bound to approved child plan {plan_hash}, "
            f"which the signed child-plan supersession at {pending[0].comment_locator} "
            f"authorizes replacing; {stopped}. The authorization permits a re-plan but is "
            "not itself an approved plan, and `agent-loop pr` never re-plans or rebinds. Rerun "
            f"`{_child_resume_hint(binding.child_issue, EXECUTION_DISPOSITION_PLANNING)}`: the "
            "child is re-planned, and after approval this PR is rebound to the revised plan."
        )


def _inadmissible_plan_audit_line(plan_hash: str) -> str:
    """The guard's audit sentence; also the durable key that stops a repeat."""
    return (
        f"Approved plan {plan_hash} is inadmissible under the inherited-matrix "
        "contract and is being revised."
    )


def _resumed_inherited_replan_force_full(comments: Sequence[object]) -> bool:
    """Reconstruct the complete-board latch after an inherited-matrix re-plan.

    The latch is set in memory just before the revision turn.  A run that
    stops after the revised plan round is durable, but before the next
    scheduling record, would otherwise lose it and let staged planning
    narrow the board that must review the revision.  It is recomputed here
    from durable state only: the latest plan coder round either carries a
    signed supersession binding, or revises a plan for which the guard's
    audit comment exists.  Once the next round has run, its scheduling
    record carries the automatic latch through the existing recovery.
    """
    coder_records = [
        record
        for record in _extract_round_metadata_records(comments, flow="plan")
        if record.metadata.role == "coder"
    ]
    if not coder_records:
        return False
    latest = coder_records[-1].metadata
    if latest.plan_supersession_digest is not None:
        return True
    if latest.prior_plan_subject is None:
        return False
    bodies = [
        body for comment in comments if isinstance((body := getattr(comment, "body", None)), str)
    ]
    for record in coder_records[:-1]:
        plan = record.metadata.canonical_plan
        if record.metadata.subject != latest.prior_plan_subject or plan is None:
            continue
        audit_line = _inadmissible_plan_audit_line(approved_plan_hash(plan))
        if any(audit_line in body for body in bodies):
            return True
    return False


def _recover_current_plan_validation_diagnostic(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    issue_number: int,
    target_coder_round: int,
    prior_plan_subject: str | None,
    candidate_kind: Literal["plan_state", "plan_revision"],
    require_execution_strategy_contract: bool,
    require_risk_test_matrix_contract: bool,
) -> PlanValidationDiagnosticTransport | None:
    if not has_plan_validation_diagnostic_marker(issue_context.comments):
        return None
    actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    architecture_version, execution_version, matrix_version = _plan_validation_contract_versions(
        require_execution_strategy_contract=require_execution_strategy_contract,
        require_risk_test_matrix_contract=require_risk_test_matrix_contract,
    )
    return recover_plan_validation_diagnostic(
        issue_context.comments,
        repository=config.repo,
        issue_number=issue_number,
        expected_author_login=actor_login,
        expected_author_id=actor_id,
        planning_generation=1,
        target_coder_round=target_coder_round,
        prior_plan_subject=prior_plan_subject,
        candidate_kind=candidate_kind,
        architecture_contract_version=architecture_version,
        execution_strategy_contract_version=execution_version,
        risk_test_matrix_contract_version=matrix_version,
        on_host_footer=lambda ctx: note_host_footer_observed(config, ctx),
    )


def _persist_exhausted_plan_validation_diagnostic(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_context: IssueContext,
    issue_number: int,
    original_error: AgentInvocationError,
    exhaustion: DeterministicPlanValidationExhaustion,
    target_coder_round: int,
    prior_plan_subject: str | None,
    candidate_kind: Literal["plan_state", "plan_revision"],
    require_execution_strategy_contract: bool,
    require_risk_test_matrix_contract: bool,
) -> None:
    """Write one authenticated bounded diagnostic, preserving the validator error."""
    try:
        actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
        architecture_version, execution_version, matrix_version = _plan_validation_contract_versions(
            require_execution_strategy_contract=require_execution_strategy_contract,
            require_risk_test_matrix_contract=require_risk_test_matrix_contract,
        )
        current = recover_plan_validation_diagnostic(
            issue_context.comments,
            repository=config.repo,
            issue_number=issue_number,
            expected_author_login=actor_login,
            expected_author_id=actor_id,
            planning_generation=1,
            target_coder_round=target_coder_round,
            prior_plan_subject=prior_plan_subject,
            candidate_kind=candidate_kind,
            architecture_contract_version=architecture_version,
            execution_strategy_contract_version=execution_version,
            risk_test_matrix_contract_version=matrix_version,
            on_host_footer=lambda ctx: note_host_footer_observed(config, ctx),
        )
        payload = PlanValidationDiagnosticPayload(
            repository=config.repo,
            issue_number=issue_number,
            planning_generation=1,
            target_coder_round=target_coder_round,
            prior_plan_subject=prior_plan_subject,
            candidate_kind=candidate_kind,
            architecture_contract_version=architecture_version,
            execution_strategy_contract_version=execution_version,
            risk_test_matrix_contract_version=matrix_version,
            expected_producer_login=actor_login,
            expected_producer_id=actor_id,
            failure_attempt=(current.failure_attempt + 1 if current is not None else 1),
            candidate_digest=exhaustion.candidate_digest,
            category="deterministic",
            diagnostic=sanitize_plan_validation_diagnostic(exhaustion.diagnostic),
        )
        body = encode_plan_validation_diagnostic_body(payload)
        posted = post_verified_trusted_issue_protocol_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=body,
            expected_author_login=actor_login,
            expected_author_id=actor_id,
        )
        if posted.body != str(body) or posted.comment_id is None or posted.author_id != actor_id:
            raise AgentLoopError("Verified diagnostic transport wrapper did not round-trip safely.")
        log(config, "Persisted an authenticated plan-validation diagnostic audit record.")
    except Exception as exc:
        detail = sanitize_plan_validation_diagnostic(str(exc))
        raise AgentInvocationError(
            f"{original_error}\nDiagnostic persistence note: the bounded handoff record was not persisted. {detail}",
            failure_category=original_error.failure_category,
            terminal_public_response=original_error.terminal_public_response,
            containment=original_error.containment,
            plan_validation_exhaustion=exhaustion,
        ) from exc


def _assemble_structured_plan_round_body(
    *,
    config: AgentLoopConfig,
    issue_number: int,
    kind: Literal["plan_state", "plan_revision"],
    parsed_plan: StructuredPlanState | StructuredPlanRevision,
    full_comment: str,
    metadata: PostedRoundMetadata,
    raw_text: str,
    prior_items: Sequence[UnresolvedReviewItem],
    model_used: str | None,
    surfaced_requirement_ids: Sequence[str],
    requires_direct_discussion_ack: bool,
) -> TrustedBody:
    """Assemble a structured plan coder round, compacting only on body overflow (#948).

    The full public comment is used whenever the transport can carry it.  Only
    the dedicated body-budget overflow selects the bounded visible digest; any
    other transport failure propagates and aborts publication.  Metadata is
    always the one derived from the full canonical plan, so the digest changes
    presentation only.
    """
    full_body = _attach_round_metadata(full_comment, metadata)
    if round_comment_fits(full_body):
        return full_body
    compact_comment = render_public_agent_comment(
        kind=kind,
        parsed=parsed_plan,
        agent=config.coder,
        prior_items=prior_items,
        raw_text=raw_text,
        config=config,
        model_used=model_used,
        compact=True,
    )
    if surfaced_requirement_ids or requires_direct_discussion_ack:
        # Never post a digest whose signed-requirement content differs from
        # what validated the raw response.
        if parse_human_requirements_acknowledgement(
            _extract_plan_human_requirements_block(raw_text)
        ).marker_present:
            validate_human_requirements_acknowledgement(
                _compact_digest_acknowledgement_text(compact_comment),
                surfaced_requirement_ids=surfaced_requirement_ids,
                requires_direct_discussion_ack=requires_direct_discussion_ack,
            )
        expected_ids = [item.requirement_id for item in parsed_plan.human_requirement_dispositions]
        rendered_ids = _compact_digest_disposition_ids(compact_comment)
        if rendered_ids != expected_ids:
            raise AgentLoopError(
                "Compact plan digest does not list exactly the plan's signed human "
                "requirement dispositions; refusing to post it."
            )
    log(
        config,
        f"Planning issue #{issue_number}: full plan comment exceeds the comment budget; "
        "posting the bounded visible digest (complete plan stays in authenticated round metadata)",
    )
    compact_body = _attach_round_metadata(compact_comment, metadata)
    # Final fit check (#886): a digest that still cannot be carried is
    # refused here with the transport's own overflow error, before posting.
    prepare_round_comment(compact_body)
    return compact_body


_COMPACT_DIGEST_DISPOSITION_LINE_RE = re.compile(r"^- \*\*(?P<id>[^*]+)\*\* — `")


def _compact_digest_disposition_ids(comment: str) -> list[str]:
    ids: list[str] = []
    active = False
    for line in comment.splitlines():
        if line.strip() == "### Human requirement dispositions":
            active = True
            continue
        if not active:
            continue
        match = _COMPACT_DIGEST_DISPOSITION_LINE_RE.match(line)
        if match is None:
            break
        ids.append(match.group("id"))
    return ids


def _compact_digest_acknowledgement_text(comment: str) -> str:
    """Acknowledgement text of a rendered digest: the block after the dispositions."""
    marker_index = comment.find(HUMAN_REQUIREMENTS_ADDRESSED_MARKER)
    return comment[marker_index:] if marker_index >= 0 else ""


def _post_plan_coder_round_comment(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    body: TrustedBody,
    diagnostic: PlanValidationDiagnosticTransport | None,
    target_coder_round: int,
    prior_plan_subject: str | None,
    candidate_kind: Literal["plan_state", "plan_revision"],
    require_execution_strategy_contract: bool,
    require_risk_test_matrix_contract: bool,
) -> bool:
    """Publish a plan coder round and report whether it superseded a diagnostic.

    A canonical coder round only clears an in-memory diagnostic when the
    diagnostic's complete payload context matches this candidate and the
    round has crossed the authenticated REST publication seam.  Unrelated or
    stale diagnostics retain their normal history and cannot be cleared by a
    successful plan.
    """
    architecture_version, execution_version, matrix_version = _plan_validation_contract_versions(
        require_execution_strategy_contract=require_execution_strategy_contract,
        require_risk_test_matrix_contract=require_risk_test_matrix_contract,
    )
    supersedes = diagnostic is not None and diagnostic.payload.matches_context(
        repository=config.repo,
        issue_number=issue_number,
        planning_generation=1,
        target_coder_round=target_coder_round,
        prior_plan_subject=prior_plan_subject,
        candidate_kind=candidate_kind,
        architecture_contract_version=architecture_version,
        execution_strategy_contract_version=execution_version,
        risk_test_matrix_contract_version=matrix_version,
    )
    if not supersedes:
        post_issue_comment(
            runner,
            config=config,
            issue_number=issue_number,
            body=body,
        )
        return False
    actor_login, actor_id = resolve_authenticated_github_actor(runner, config=config)
    posted = post_verified_trusted_issue_round_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=body,
        expected_author_login=actor_login,
        expected_author_id=actor_id,
    )
    if (
        posted.comment_id is None
        or posted.comment_id < 1
        or posted.author != actor_login
        or posted.author_id != actor_id
        or not isinstance(posted.created_at, str)
        or not posted.created_at
        or not isinstance(posted.body, str)
    ):
        raise AgentLoopError(
            "Verified canonical plan publication returned an incomplete transport wrapper."
        )
    return True
