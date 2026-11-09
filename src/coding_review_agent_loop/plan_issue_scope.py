"""Planning scopes across an edited issue (#1390).

A plan-first resume compares the normalized issue title/body digest the
current plan was written against with the issue's current digest.  When the
issue was edited under an unapproved plan, the run opens a new *planning
scope*: the orchestrator posts one authenticated scope-boundary audit record
and starts a fresh plan at the next monotonic round number, the scope start.
Old plan comments are never deleted; they stay in the issue as lineage.

Two inputs, kept apart on purpose:

* reviewer-board lineage (root anchor, binding initialization, board-change
  links, signed amendments) is always resolved from the *full* admitted
  history, so its round semantics are unchanged;
* planning candidates, the obligation ledger, panel evidence, scheduler
  replay, validation diagnostics and the prompt context read a *scoped*
  projection that excludes planning records from earlier scopes.

The scoped projection keeps every comment position: an archived planning
record is replaced by an empty-bodied copy rather than removed, so record and
link comment indices stay comparable with the full-history lineage.  Without
an admitted boundary the scope starts at round 1 and every projection is the
identity, so a never-edited issue behaves exactly as before.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .errors import AgentLoopError

PLAN_SCOPE_BOUNDARY_MARKER = "AGENT_PLAN_SCOPE_BOUNDARY"
PLAN_SCOPE_BOUNDARY_KIND = "plan-scope-boundary"
PLAN_SCOPE_BOUNDARY_SCHEMA_VERSION = 1
PLAN_SCOPE_BOUNDARY_HEADING = "Planning scope restarted."
PLAN_SCOPE_BOUNDARY_RE = re.compile(
    rf"<!--\s*{PLAN_SCOPE_BOUNDARY_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", re.I
)
# Any record-shaped span naming the boundary record, well-formed or not.
PLAN_SCOPE_BOUNDARY_PRESENCE_RE = re.compile(
    rf"<!--\s*{PLAN_SCOPE_BOUNDARY_MARKER}\b", re.I
)
_BOUNDARY_KEYS = frozenset(
    {
        "kind",
        "schema_version",
        "issue_number",
        "previous_issue_digest",
        "current_issue_digest",
        "archived_round_number",
        "archived_plan_subject",
        "scope_start_round",
    }
)
_HEX16 = re.compile(r"[0-9a-f]{16}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class PlanScopeBoundary:
    """One admitted planning-scope boundary record."""

    comment_index: int
    issue_number: int
    previous_issue_digest: str
    current_issue_digest: str
    archived_round_number: int
    archived_plan_subject: str | None
    scope_start_round: int

    def payload(self) -> dict[str, object]:
        return scope_boundary_payload(
            issue_number=self.issue_number,
            previous_issue_digest=self.previous_issue_digest,
            current_issue_digest=self.current_issue_digest,
            archived_round_number=self.archived_round_number,
            archived_plan_subject=self.archived_plan_subject,
        )


@dataclass(frozen=True)
class PlanScopeDecision:
    """The read-only edited-issue decision shared by the preflight and plan startup."""

    # The planning scope this invocation runs in: the admitted boundary's
    # scope start, or the start of the scope it is about to open.
    scope_start_round: int
    # True when this invocation must post a new boundary before planning.
    reset_required: bool
    recorded_digest: str | None
    current_digest: str
    # The latest round of the scope being archived (or the resumed round).
    archived_round: int | None = None
    archived_subject: str | None = None
    # The issue text changed since the recorded digest.
    edited: bool = False
    # ``--plan-resume-keep-on-issue-edit`` kept the existing plan.
    kept: bool = False
    # The plan was already handed off for implementation, so it is kept: an
    # authorized signed re-plan of a handed-off child plan (#936) revises that
    # plan, and a fresh scope would orphan it.
    handed_off: bool = False
    boundary: PlanScopeBoundary | None = None

    def boundary_payload(self, issue_number: int) -> dict[str, object]:
        assert self.reset_required and self.recorded_digest is not None
        assert self.archived_round is not None
        return scope_boundary_payload(
            issue_number=issue_number,
            previous_issue_digest=self.recorded_digest,
            current_issue_digest=self.current_digest,
            archived_round_number=self.archived_round,
            archived_plan_subject=self.archived_subject,
        )


def _body(comment: object) -> object:
    if isinstance(comment, dict):
        return comment.get("body")
    return getattr(comment, "body", None)


def _author_id(comment: object) -> object:
    if isinstance(comment, dict):
        return comment.get("author_id")
    return getattr(comment, "author_id", None)


def comments_bear_scope_boundary(comments: Sequence[object]) -> bool:
    """Whether any comment carries a record-shaped boundary span, well-formed or not."""
    return any(
        isinstance(body := _body(comment), str) and PLAN_SCOPE_BOUNDARY_PRESENCE_RE.search(body)
        for comment in comments
    )


def scope_boundary_payload(
    *,
    issue_number: int,
    previous_issue_digest: str,
    current_issue_digest: str,
    archived_round_number: int,
    archived_plan_subject: str | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": PLAN_SCOPE_BOUNDARY_KIND,
        "schema_version": PLAN_SCOPE_BOUNDARY_SCHEMA_VERSION,
        "issue_number": issue_number,
        "previous_issue_digest": previous_issue_digest,
        "current_issue_digest": current_issue_digest,
        "archived_round_number": archived_round_number,
        "archived_plan_subject": archived_plan_subject,
        "scope_start_round": archived_round_number + 1,
    }
    parse_scope_boundary(payload, comment_index=0)
    return payload


def _encode_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def render_scope_boundary(payload: dict[str, object]) -> object:
    """The orchestrator-authored boundary comment: audit text plus the reserved record."""
    from .protocol_markers import ISSUE_COMMENT_SURFACE, TrustedBody

    subject = payload["archived_plan_subject"]
    lines = [
        PLAN_SCOPE_BOUNDARY_HEADING,
        "",
        "The issue title or body changed since plan round "
        f"{payload['archived_round_number']}, so the earlier plan no longer binds this "
        "issue. A fresh plan starts in a new planning scope.",
        "",
        f"- Recorded issue digest: `{payload['previous_issue_digest']}`",
        f"- Current issue digest: `{payload['current_issue_digest']}`",
        f"- Archived plan round: {payload['archived_round_number']}"
        + (f" (plan subject `{str(subject)[:16]}`)" if subject else ""),
        f"- Fresh plan starts at round {payload['scope_start_round']}; the planning round "
        "budget restarts there.",
        "- The earlier plan and its review discussion stay in this issue's history. Their "
        "unresolved items are not carried into the new plan.",
        "",
        f"<!-- {PLAN_SCOPE_BOUNDARY_MARKER}: {_encode_payload(payload)} -->",
        "",
        "-- Orchestrator",
    ]
    return TrustedBody.canonical(
        "\n".join(lines),
        surface=ISSUE_COMMENT_SURFACE,
        expected_tokens=(PLAN_SCOPE_BOUNDARY_MARKER,),
    )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def parse_scope_boundary(
    payload: object, *, comment_index: int, issue_number: int | None = None
) -> PlanScopeBoundary:
    """Validate every field of one boundary payload; raise ``AgentLoopError`` otherwise."""
    if not isinstance(payload, dict) or set(payload) != _BOUNDARY_KEYS:
        raise AgentLoopError("planning-scope boundary record has missing or unknown fields")
    if (
        payload["kind"] != PLAN_SCOPE_BOUNDARY_KIND
        or payload["schema_version"] != PLAN_SCOPE_BOUNDARY_SCHEMA_VERSION
        or isinstance(payload["schema_version"], bool)
    ):
        raise AgentLoopError("planning-scope boundary record has an unsupported kind or version")
    if not _is_positive_int(payload["issue_number"]) or (
        issue_number is not None and payload["issue_number"] != issue_number
    ):
        raise AgentLoopError("planning-scope boundary record names another issue")
    for key in ("previous_issue_digest", "current_issue_digest"):
        if not isinstance(payload[key], str) or not _HEX16.match(payload[key]):
            raise AgentLoopError(f"planning-scope boundary record has an invalid {key}")
    if payload["previous_issue_digest"] == payload["current_issue_digest"]:
        raise AgentLoopError("planning-scope boundary record names an unchanged issue digest")
    archived = payload["archived_round_number"]
    start = payload["scope_start_round"]
    if not _is_positive_int(archived) or not _is_positive_int(start) or start <= archived:
        raise AgentLoopError("planning-scope boundary record has inconsistent rounds")
    if start != archived + 1:
        raise AgentLoopError("planning-scope boundary record has inconsistent rounds")
    subject = payload["archived_plan_subject"]
    if subject is not None and (not isinstance(subject, str) or not _HEX64.match(subject)):
        raise AgentLoopError("planning-scope boundary record has an invalid plan subject")
    return PlanScopeBoundary(
        comment_index=comment_index,
        issue_number=payload["issue_number"],
        previous_issue_digest=payload["previous_issue_digest"],
        current_issue_digest=payload["current_issue_digest"],
        archived_round_number=archived,
        archived_plan_subject=subject,
        scope_start_round=start,
    )


def latest_scope_boundary(
    admitted_comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    *,
    issue_number: int | None = None,
    ignored_sink: list[str] | None = None,
) -> PlanScopeBoundary | None:
    """The newest boundary authored by the authenticated orchestrator actor.

    A boundary-shaped record by any other author, or one whose author cannot be
    authenticated, is ignored and reported to ``ignored_sink``: it never
    retires planning history or obligations.  A malformed record is ignored the
    same way, which keeps today's resume behaviour.
    """
    latest: PlanScopeBoundary | None = None
    for index, comment in enumerate(admitted_comments):
        body = _body(comment)
        if not isinstance(body, str) or not PLAN_SCOPE_BOUNDARY_PRESENCE_RE.search(body):
            continue
        locator = f"issue comment {index + 1}"
        author_id = _author_id(comment)
        if trusted_actor is None or author_id is None or author_id != trusted_actor[1]:
            if ignored_sink is not None:
                ignored_sink.append(
                    f"ignoring the planning-scope boundary record at {locator}: it was not "
                    "authored by the authenticated orchestrator actor"
                )
            continue
        matches = list(PLAN_SCOPE_BOUNDARY_RE.finditer(body))
        try:
            if len(matches) != 1 or len(PLAN_SCOPE_BOUNDARY_PRESENCE_RE.findall(body)) != 1:
                raise AgentLoopError("planning-scope boundary record is malformed")
            try:
                raw = json.loads(
                    base64.urlsafe_b64decode(matches[0].group("payload").encode("ascii")).decode(
                        "utf-8"
                    )
                )
            except (ValueError, UnicodeDecodeError) as exc:
                raise AgentLoopError("planning-scope boundary record cannot be decoded") from exc
            boundary = parse_scope_boundary(
                raw, comment_index=index, issue_number=issue_number
            )
        except AgentLoopError as exc:
            if ignored_sink is not None:
                ignored_sink.append(f"ignoring the planning-scope boundary record at {locator}: {exc}")
            continue
        latest = boundary
    return latest


def resolve_plan_scope(
    comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    *,
    issue_number: int | None = None,
    ignored_sink: list[str] | None = None,
) -> int:
    """The current planning scope start: the admitted boundary's, or 1.

    This never acquires history itself.  Callers admit the comments first,
    through :func:`~coding_review_agent_loop.plan_board_state.admitted_issue_comments`,
    and pass the authenticated actor that admission returned.  With no
    boundary-shaped record this returns 1 without needing an actor.  A
    boundary-bearing history with no authenticated actor fails closed rather
    than silently resolving to scope 1, which would resurrect archived rounds.
    """
    comments = tuple(comments)
    if not comments_bear_scope_boundary(comments):
        return 1
    if trusted_actor is None:
        raise AgentLoopError(
            "The issue carries a planning-scope boundary record, but no authenticated "
            "GitHub read is available to admit it."
        )
    boundary = latest_scope_boundary(
        comments, trusted_actor, issue_number=issue_number, ignored_sink=ignored_sink,
    )
    return boundary.scope_start_round if boundary is not None else 1


def _blank(comment: object) -> object:
    if dataclasses.is_dataclass(comment) and not isinstance(comment, type):
        from .pr_round_ingestion import redact_comment_body
        return redact_comment_body(comment, "")
    if isinstance(comment, dict):
        return {**comment, "body": ""}
    clone = copy.copy(comment)
    try:
        setattr(clone, "body", "")
    except (AttributeError, TypeError):
        from types import SimpleNamespace

        clone = SimpleNamespace(**{**getattr(comment, "__dict__", {}), "body": ""})
    return clone


def _archived_indexes(comments: Sequence[object], scope_start_round: int) -> frozenset[int]:
    from .protocol_markers import strip_known_host_footer
    from .round_state import (
        _extract_round_metadata_records,
    )
    from .round_metadata import (
        decode_plan_validation_diagnostic_body,
    )
    from .round_transport import PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE

    archived = {
        record.index
        for record in _extract_round_metadata_records(comments, flow="plan")
        if record.metadata.round_number < scope_start_round
    }
    for index, comment in enumerate(comments):
        body = _body(comment)
        if not isinstance(body, str):
            continue
        stripped, _footered = strip_known_host_footer(body)
        if not PLAN_VALIDATION_DIAGNOSTIC_MARKER_RE.search(stripped):
            continue
        try:
            payload = decode_plan_validation_diagnostic_body(stripped)
        except AgentLoopError:
            continue
        if payload.target_coder_round < scope_start_round:
            archived.add(index)
    return frozenset(archived)


def scope_plan_comments(
    comments: Sequence[object], scope_start_round: int
) -> tuple[object, ...]:
    """``comments`` with earlier-scope planning records blanked, positions kept.

    Only plan-flow round-metadata comments whose round precedes the scope
    start, and plan-validation diagnostics that target an earlier round, are
    blanked.  Human comments, board-change links and amendments, round
    transport sidecars and boundary records are kept verbatim.
    """
    comments = tuple(comments)
    if scope_start_round <= 1:
        return comments
    archived = _archived_indexes(comments, scope_start_round)
    if not archived:
        return comments
    return tuple(
        _blank(comment) if index in archived else comment
        for index, comment in enumerate(comments)
    )


def scope_prompt_comments(
    comments: Sequence[object], scope_start_round: int
) -> tuple[object, ...]:
    """The prompt view: earlier-scope planning records dropped, not just blanked.

    Accepts full or already-scoped comments: a blanked archived record (an
    empty body) is dropped as well.
    """
    comments = tuple(comments)
    if scope_start_round <= 1:
        return comments
    archived = _archived_indexes(comments, scope_start_round)
    return tuple(
        comment
        for index, comment in enumerate(comments)
        if index not in archived and _body(comment) != ""
    )


def recorded_plan_issue_digest(
    scoped_records: Sequence[object], boundary: PlanScopeBoundary | None
) -> str | None:
    """The digest the current scope's plan was written against, or ``None`` (legacy).

    Prefers the newest in-scope planner ``plan_issue_digest``, then the newest
    in-scope scheduler checkpoint ``scheduler_issue_digest``, then the admitted
    boundary's current digest.
    """
    ordered = sorted(scoped_records, key=lambda record: record.index)  # type: ignore[attr-defined]
    for record in reversed(ordered):
        metadata = record.metadata  # type: ignore[attr-defined]
        if metadata.role == "coder" and metadata.plan_issue_digest is not None:
            return metadata.plan_issue_digest
    for record in reversed(ordered):
        digest = record.metadata.scheduler_issue_digest  # type: ignore[attr-defined]
        if digest is not None:
            return digest
    return boundary.current_issue_digest if boundary is not None else None


def plan_handed_off(
    comments: Sequence[object], *, issue_number: int | None, repo: str | None
) -> bool:
    """Whether the issue's latest handoff record hands an approved plan to implementation."""
    if issue_number is None or not repo:
        return False
    from .issue_pr_handoff import find_latest_issue_pr_handoff

    handoff = find_latest_issue_pr_handoff(comments, issue_number=issue_number, repo=repo)
    return bool(
        handoff is not None
        and handoff.flow == "approved-plan-implementation"
        and handoff.plan_hash
    )


def plan_scope_decision(
    admitted_comments: Sequence[object],
    trusted_actor: tuple[str, int] | None,
    current_digest: str,
    keep_on_edit: bool,
    *,
    plan_approved: bool = False,
    issue_number: int | None = None,
    repo: str | None = None,
    ignored_sink: list[str] | None = None,
) -> PlanScopeDecision:
    """Decide, read-only, whether an edited issue opens a new planning scope.

    ``plan_approved`` is the approval classification of the plan in the scope
    derived from the current boundary: an approved plan is never reopened.
    Neither is a plan the issue has already handed off for implementation
    (read from the issue's own handoff record when ``issue_number`` and
    ``repo`` are given).  The issue floor preflight and plan startup both call
    this one function, so they always agree on the scope.
    """
    from .round_state import _extract_round_metadata_records

    admitted_comments = tuple(admitted_comments)
    boundary = latest_scope_boundary(
        admitted_comments, trusted_actor, issue_number=issue_number, ignored_sink=ignored_sink,
    )
    base_start = boundary.scope_start_round if boundary is not None else 1
    all_records = _extract_round_metadata_records(admitted_comments, flow="plan")
    scoped_records = tuple(
        record for record in all_records if record.metadata.round_number >= base_start
    )
    recorded = recorded_plan_issue_digest(scoped_records, boundary)
    unchanged = PlanScopeDecision(
        scope_start_round=base_start,
        reset_required=False,
        recorded_digest=recorded,
        current_digest=current_digest,
        boundary=boundary,
    )
    if plan_approved or recorded is None or recorded == current_digest:
        return unchanged
    latest_round = max(
        (record.metadata.round_number for record in all_records), default=base_start - 1
    )
    archived_round = max(latest_round, base_start - 1, 1)
    latest_coder = next(
        (record for record in reversed(scoped_records) if record.metadata.role == "coder"),
        None,
    )
    archived_subject = latest_coder.metadata.subject if latest_coder is not None else None
    if archived_subject is not None and not _HEX64.match(archived_subject):
        archived_subject = None
    if keep_on_edit or plan_handed_off(admitted_comments, issue_number=issue_number, repo=repo):
        return dataclasses.replace(
            unchanged, edited=True, kept=True, handed_off=not keep_on_edit,
            archived_round=archived_round, archived_subject=archived_subject,
        )
    return PlanScopeDecision(
        scope_start_round=archived_round + 1,
        reset_required=True,
        recorded_digest=recorded,
        current_digest=current_digest,
        archived_round=archived_round,
        archived_subject=archived_subject,
        edited=True,
        boundary=boundary,
    )
