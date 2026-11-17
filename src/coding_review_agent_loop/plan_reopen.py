"""Operator-requested reopening of an approved plan (#2010).

``agent-loop issue N --plan-first --reopen-approved-plan`` supersedes the
issue's completely approved plan and opens a fresh planning scope against the
current issue body.  It is the only way an approved plan is reopened; without
the option an approved plan is never reopened, whatever the issue edits.

One read-only gate, :func:`apply_operator_reopen`, runs in ``run_issue_loop``
before issue recovery, whether or not the reviewer-board floor is enabled.  It
classifies the current scope's plan, inventories every hand-off surface issue
recovery could act on (:func:`approved_plan_handoff_reason`) and refuses a
handed-off plan before anything is posted.  Otherwise it posts one
authenticated schema-version-2 planning-scope boundary naming the superseded
plan hash, so recovery and planning see only the new, empty scope.  The
option is per invocation: child planning and rendered recovery commands never
inherit it.
"""

from __future__ import annotations

from collections.abc import Sequence

from .errors import AgentLoopError
from .logging import log

_INVENTORY_ERRORS = (AgentLoopError, OSError, ValueError, KeyError, TypeError)


def _bodies(comments: Sequence[object]):
    from .pr_round_ingestion import record_body

    for comment in comments:
        body = record_body(comment)
        if isinstance(body, str):
            yield body


def comment_handoff_reason(
    comments: Sequence[object], *, issue_number: int, repo: str | None, plan_hash: str | None,
) -> str | None:
    """The hand-off recorded in the issue's own comments, or ``None``.

    Covers the canonical issue-PR handoff (any flow, any PR state), one-shot
    implementation handoffs (any mode), decomposition summaries, phase
    implementation handoffs, split-stage handoffs and split materializations
    bound to the issue.  It makes no GitHub read.  A malformed record raises
    ``AgentLoopError``; callers treat that as unverifiable, never as absent.
    """
    from .decomposition import (
        ONE_SHOT_IMPL_HANDOFF_MARKER_RE,
        _decode_one_shot_impl_handoff_metadata,
        find_decompositions_for_parent,
        find_phase_implementation_handoffs_for_parent,
    )
    from .split_materialization import (
        SPLIT_STAGE_HANDOFF_MARKER_RE,
        _decode_stage_handoff_metadata,
        find_existing_split_materialization,
    )

    if repo:
        from .issue_pr_handoff import find_latest_issue_pr_handoff

        handoff = find_latest_issue_pr_handoff(comments, issue_number=issue_number, repo=repo)
        if handoff is not None:
            return (
                f"an issue-PR handoff record names PR #{handoff.pr_number} "
                f"(flow {handoff.flow}"
                + (f", plan {handoff.plan_hash}" if handoff.plan_hash else "")
                + ")"
            )
    for body in _bodies(comments):
        for match in ONE_SHOT_IMPL_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_one_shot_impl_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == issue_number:
                return (
                    f"a one-shot implementation handoff record exists for plan "
                    f"{metadata.plan_hash}"
                )
    for summary in find_decompositions_for_parent(comments, parent_issue=issue_number):
        return f"a decomposition summary records child stages for plan {summary.plan_hash}"
    for phase in find_phase_implementation_handoffs_for_parent(
        comments, parent_issue=issue_number
    ):
        return (
            f"a phase implementation handoff names child issue #{phase.child_issue_number} "
            f"for plan {phase.plan_hash}"
        )
    for body in _bodies(comments):
        for match in SPLIT_STAGE_HANDOFF_MARKER_RE.finditer(body):
            metadata = _decode_stage_handoff_metadata(match.group("payload"))
            if metadata.parent_issue == issue_number:
                return f"a split-stage handoff record exists for plan {metadata.plan_hash}"
    materialized = find_existing_split_materialization(comments, parent_issue=issue_number)
    if materialized is not None and materialized.children:
        return "a split materialization record lists child issues for this issue"
    return None


def approved_shortcut_reopen_hint(
    comments: Sequence[object],
    *,
    issue_number: int,
    repo: str | None,
    plan_hash: str | None,
    recorded_digest: str | None,
    current_digest: str,
) -> str:
    """The suffix of the "planning is not reopened" log line (#2010).

    Empty unless the issue title/body digest changed since approval (an
    unknown recorded digest gives no hint).  Only the cheap comment-level
    hand-off check runs here; it makes no GitHub search.
    """
    if recorded_digest is None or recorded_digest == current_digest:
        return ""
    changed = (
        f"; the issue title/body changed since approval (recorded digest {recorded_digest}, "
        f"current digest {current_digest})"
    )
    try:
        handed_off = comment_handoff_reason(
            comments, issue_number=issue_number, repo=repo, plan_hash=plan_hash,
        )
    except AgentLoopError as exc:
        handed_off = f"its hand-off records cannot be read: {exc}"
    if handed_off is not None:
        return (
            f"{changed}, but the plan was already handed off ({handed_off}), so "
            "--reopen-approved-plan cannot reopen it"
        )
    return (
        f"{changed} — rerun with --reopen-approved-plan to start a fresh planning scope"
    )


def approved_plan_handoff_reason(
    runner: object,
    *,
    config: object,
    issue_number: int,
    comments: Sequence[object],
    plan_hash: str,
) -> str | None:
    """Why the approved plan ``plan_hash`` counts as handed off, or ``None``.

    The inventory covers every hand-off surface issue recovery could act on
    after a boundary is posted, using recovery's own read-only resolvers:

    (a) the issue's own hand-off records (:func:`comment_handoff_reason`);
    (b) a legacy, metadata-free open PR strongly closing the issue, through
        ``find_open_pr_closing_issue`` with the approved-plan provenance scope
        recovery would use for this plan, in its strict mode so an empty,
        non-JSON, non-list, malformed or possibly truncated listing refuses.
        A match refuses, and so does a
        raised ambiguity or provenance mismatch: after a reopen recovery would
        call the same resolver without a scope and raise on that candidate;
    (c) realized child issues in the create-before-summary window, found by
        the same discovery topology creation uses (in its strict mode, so an
        unreadable or malformed listing refuses), whose phase-identity record
        names this parent and ``plan_hash``.  A child for another plan hash
        does not count.

    Fail closed: any lookup or decode failure refuses the reopen ("cannot
    verify").  An infrastructure failure never counts as "not handed off".
    Any record, PR or identity found counts toward refusal even when it is
    not authenticated, because a forged artifact can only cause a refusal,
    never a reopen.
    """
    from .decomposition import (
        PHASE_IDENTITY_MARKER_RE,
        _decode_json_payload,
        discover_parent_child_issues,
    )
    from .github import find_open_pr_closing_issue
    from .issue_pr_provenance import IssuePrProvenanceScope

    repo = getattr(config, "repo", None)
    try:
        recorded = comment_handoff_reason(
            comments, issue_number=issue_number, repo=repo, plan_hash=plan_hash,
        )
    except _INVENTORY_ERRORS as exc:
        return f"cannot verify that plan {plan_hash} was not handed off: {exc}"
    if recorded is not None:
        return recorded
    try:
        legacy = find_open_pr_closing_issue(
            runner,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            issue_number=issue_number,
            expected_scope=IssuePrProvenanceScope(
                repository=repo, issue_number=issue_number, flow="approved",
                approved_plan_hash=plan_hash,
            ),
            strict=True,
        )
    except _INVENTORY_ERRORS as exc:
        return f"an open PR association for this issue cannot be resolved: {exc}"
    if legacy is not None:
        return (
            f"open PR #{legacy.pr_number} strongly closes this issue and carries plan "
            f"{plan_hash}"
        )
    try:
        found, _searched, _direct = discover_parent_child_issues(
            runner,  # type: ignore[arg-type]
            config=config,  # type: ignore[arg-type]
            parent_issue=issue_number,
            issue_comments=comments,
            strict=True,
        )
        for candidate in found:
            for marker in PHASE_IDENTITY_MARKER_RE.finditer(candidate.body or ""):
                payload = _decode_json_payload(
                    marker.group("payload"), marker_name="AGENT_PLAN_PHASE_IDENTITY",
                )
                if (
                    payload.get("parent_issue") == issue_number
                    and not isinstance(payload.get("parent_issue"), bool)
                    and payload.get("plan_hash") == plan_hash
                ):
                    return f"child issue #{candidate.number} already carries plan {plan_hash}"
    except _INVENTORY_ERRORS as exc:
        return f"cannot verify that plan {plan_hash} was not handed off: {exc}"
    return None


def reopen_refusal_error(issue_number: int, plan_hash: str | None, reason: str) -> AgentLoopError:
    return AgentLoopError(
        f"Issue #{issue_number}: --reopen-approved-plan refused: plan {plan_hash} was already "
        f"handed off ({reason}); nothing was posted. File a follow-up issue or re-plan the "
        "handed-off work through its own flow."
    )


def operator_reopen_decision(
    runner: object,
    *,
    config: object,
    issue_number: int,
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    issue_digest: str,
    classification: object,
):
    """The shared reopen decision of the gate and the issue floor preflight.

    Runs the hand-off inventory only for an approved plan, then makes the
    read-only :func:`~coding_review_agent_loop.plan_issue_scope.plan_scope_decision`.
    """
    from .plan_issue_scope import plan_scope_decision

    approved = bool(getattr(classification, "approved", False))
    plan_hash = getattr(classification, "plan_hash", None)
    handoff = (
        approved_plan_handoff_reason(
            runner, config=config, issue_number=issue_number, comments=comments,
            plan_hash=plan_hash,
        )
        if approved and plan_hash
        else None
    )
    return plan_scope_decision(
        comments,
        trusted_actor,
        issue_digest,
        False,
        plan_approved=approved,
        issue_number=issue_number,
        repo=getattr(config, "repo", None),
        reopen_approved=True,
        approved_plan_hash=plan_hash,
        handoff_reason=handoff,
    )


def apply_operator_reopen(
    runner: object,
    *,
    config: object,
    issue_number: int,
    issue_context: object,
    parent_issue_context: object | None = None,
):
    """The single pre-recovery ``--reopen-approved-plan`` gate; returns the issue context.

    Without the option this is the identity.  Under ``--dry-run`` the issue
    context is synthetic, so the gate logs a no-op notice and posts nothing.
    A handed-off plan raises before any post, state write, recovery routing
    or agent invocation.  Otherwise the boundary is posted, re-read and
    verified, and the refreshed issue context is returned so recovery runs
    only on the new, empty scope.
    """
    if not getattr(config, "reopen_approved_plan", False):
        return issue_context
    if getattr(config, "dry_run", False):
        log(
            config,
            f"Issue #{issue_number}: dry run; --reopen-approved-plan performs no reopen "
            "check and posts nothing",
        )
        return issue_context
    from dataclasses import replace as dataclasses_replace

    from .github import get_issue_context, post_issue_comment
    from .panel_evidence import plan_issue_text_digest
    from .plan_board_state import (
        admitted_issue_comments,
        plan_drift_error,
        plan_floor_refusal_hint,
        resolve_plan_board_state,
    )
    from .plan_issue_scope import latest_scope_boundary, render_scope_boundary
    from .plan_verification import build_plan_approval_inputs, classify_plan_approval

    comments, trusted_actor = admitted_issue_comments(
        runner, config=config, issue_number=issue_number,
        comments=getattr(issue_context, "comments", ()),
    )
    admitted_context = dataclasses_replace(issue_context, comments=comments)  # type: ignore[type-var]
    classification = classify_plan_approval(
        comments,
        config=config,
        issue_number=issue_number,
        trusted_actor=trusted_actor,
        inputs=build_plan_approval_inputs(
            runner, config=config, issue_context=admitted_context,
            parent_issue_context=parent_issue_context,
        ),
    )
    if not classification.approved:
        log(
            config,
            f"Issue #{issue_number}: --reopen-approved-plan has no effect: the current "
            f"planning scope has no completely approved plan ({classification.state})",
        )
        return issue_context
    decision = operator_reopen_decision(
        runner,
        config=config,
        issue_number=issue_number,
        comments=comments,
        trusted_actor=trusted_actor,
        issue_digest=plan_issue_text_digest(issue_context),  # type: ignore[arg-type]
        classification=classification,
    )
    if decision.reopen_refused_reason is not None:
        raise reopen_refusal_error(
            issue_number, classification.plan_hash, decision.reopen_refused_reason,
        )
    if not decision.reset_required:  # pragma: no cover - defensive
        raise AgentLoopError(
            f"Issue #{issue_number}: --reopen-approved-plan could not open a fresh planning "
            "scope; nothing was posted."
        )
    fresh_scope_start = decision.scope_start_round
    # The same read-only plan board floor check plan startup runs at a fresh
    # scope start: a refusal stops before anything is posted.
    resolve_plan_board_state(
        runner,
        config=config,
        issue_number=issue_number,
        comments=comments,
        admitted=(comments, trusted_actor),
        drift_error=plan_drift_error(issue_number),
        floor_context=f"Issue #{issue_number} plan reviewer board",
        resume_round=lambda _config, _lineage: fresh_scope_start,
        floor_refusal_hint=plan_floor_refusal_hint(issue_number, lambda: fresh_scope_start),
    )
    log(
        config,
        f"Issue #{issue_number}: superseding approved plan {classification.plan_hash} by "
        f"operator request; starting a fresh plan at round {fresh_scope_start}",
    )
    post_issue_comment(
        runner,  # type: ignore[arg-type]
        config=config,  # type: ignore[arg-type]
        issue_number=issue_number,
        body=render_scope_boundary(decision.boundary_payload(issue_number)),  # type: ignore[arg-type]
    )
    refreshed = get_issue_context(runner, config=config, issue_number=issue_number)  # type: ignore[arg-type]
    refreshed_comments, refreshed_actor = admitted_issue_comments(
        runner, config=config, issue_number=issue_number, comments=refreshed.comments,
    )
    boundary = latest_scope_boundary(
        refreshed_comments, refreshed_actor, issue_number=issue_number,
    )
    if (
        boundary is None
        or boundary.scope_start_round != fresh_scope_start
        or boundary.superseded_plan_hash != classification.plan_hash
    ):
        raise AgentLoopError(
            f"Issue #{issue_number}: the planning-scope boundary record posted for round "
            f"{fresh_scope_start} is not visible as an admitted record; no agent was "
            "invoked. Rerun once the record is readable."
        )
    return refreshed


def reopen_superseded_execution_hashes(
    runner: object,
    *,
    config: object,
    issue_number: int,
    comments: Sequence[object],
) -> tuple[str, ...]:
    """Plan hashes whose execution records an operator reopen superseded (#2010).

    An operator reopen is accepted only when its hand-off inventory found no
    realized work, so execution decisions and topology checkpoints posted
    before the newest admitted ``operator-reopen`` boundary are unrealized
    history.  Their hashes join the existing durable retirement of
    terminal or unrealized decisions (``execution_policy``), including the
    superseded hash itself when the new scope re-approves byte-identical
    text, so the next execution decision records the supersession and no
    archived decision or checkpoint is adopted.

    A hash is omitted once a later decision already retires that record: the
    retirement is position-aware from then on, so new-scope records stay
    current on resume.  Only a boundary authored by the authenticated
    orchestrator actor counts; without one nothing is retired.
    """
    from .decomposition import (
        EXECUTION_DECISION_MARKER_RE,
        TOPOLOGY_CHECKPOINT_MARKER_RE,
        _decode_checkpoint,
        _decode_execution_decision,
    )
    from .plan_issue_scope import comments_bear_scope_boundary, latest_operator_reopen_boundary

    comments = tuple(comments)
    if not comments_bear_scope_boundary(comments):
        return ()
    from .plan_board_state import admitted_issue_comments

    admitted, trusted_actor = admitted_issue_comments(
        runner, config=config, issue_number=issue_number, comments=comments,
    )
    boundary = latest_operator_reopen_boundary(
        admitted, trusted_actor, issue_number=issue_number,
    )
    if boundary is None:
        return ()
    decisions: list[tuple[int, object]] = []
    records: list[tuple[int, str]] = []
    for index, body in enumerate(_bodies_by_position(admitted)):
        for match in EXECUTION_DECISION_MARKER_RE.finditer(body):
            decision = _decode_execution_decision(match.group("payload"))
            if decision.parent_issue == issue_number:
                decisions.append((index, decision))
                records.append((index, decision.plan_hash))
        for match in TOPOLOGY_CHECKPOINT_MARKER_RE.finditer(body):
            checkpoint = _decode_checkpoint(match.group("payload"))
            if checkpoint.parent_issue == issue_number:
                records.append((index, checkpoint.plan_hash))
    superseded = [
        plan_hash
        for index, plan_hash in records
        if index < boundary.comment_index
        and not any(
            later > index and plan_hash in getattr(decision, "retires_plan_hashes", ())
            for later, decision in decisions
        )
    ]
    return tuple(dict.fromkeys(superseded))


def _bodies_by_position(comments: Sequence[object]):
    from .pr_round_ingestion import record_body

    for comment in comments:
        body = record_body(comment)
        yield body if isinstance(body, str) else ""
