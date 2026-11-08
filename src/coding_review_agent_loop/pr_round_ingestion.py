"""Author admission and family projections for live issue and PR observations."""
from __future__ import annotations
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentLoopConfig
    from .runner import Runner
    from .round_metadata import PostedRoundRecord
from .errors import AgentLoopError
from .pr_round_errors import pr_history_boundary
from .github import (IssueComment, CompleteCommentListing, PR_THREAD_SURFACE,
    ISSUE_THREAD_SURFACE, comment_thread_surface, read_complete_comment_listing,
    resolve_authenticated_github_actor)


# Match literal or JSON-unicode-escaped kind bytes without decoding any payload.
_BOARD_CONTROL_CANDIDATE_RE = re.compile("".join(
    rf"(?:{re.escape(char)}|\\u{ord(char):04x})" for char in "reviewer-board-amendment"
), re.I)


def board_control_candidate(comment: object) -> bool:
    """Recognize amendment-shaped content before authoritative JSON parsing."""
    body = getattr(comment, "body", None)
    return isinstance(body, str) and _BOARD_CONTROL_CANDIDATE_RE.search(body) is not None


def protocol_candidate(comment: object) -> bool:
    """Recognize malformed record-shaped content without decoding payloads."""
    body = getattr(comment, "body", None)
    return board_control_candidate(comment) or isinstance(body, str) and re.search(
        r"<!--\s*AGENT_(?:LOOP_META|LOOP_SIDECAR|PR_BOARD_CHANGE|PLAN_BOARD_CHANGE)\b|reviewer-board-amendment", body, re.I) is not None



@dataclass(frozen=True)
class AdmittedPrComment(IssueComment):
    """Raw prose and identity, with separately bound protocol eligibility."""
    invocation_actor_id: int = 0
    invocation_actor_login: str = ""
    observation_surface: str = ""
    admitted_body: str | None = None
    admitted_protocol_body: str | None = None

    amendment_eligible: bool = False
    human_trust: tuple[object, ...] = ()

    @property
    def amendment_body(self) -> str | None:
        if self.body != self.admitted_body:
            raise AgentLoopError(f"Comment {self.comment_id} changed after admission.")
        return self.admitted_protocol_body if self.amendment_eligible else ""

    @property
    def protocol_body(self) -> str | None:
        if self.body != self.admitted_body:
            raise AgentLoopError(f"PR comment {self.comment_id} changed after admission.")
        return self.admitted_protocol_body if self.author_id == self.invocation_actor_id else ""


def redact_comment_body(comment: IssueComment, body: str) -> IssueComment:
    """Preserve family exclusions when a trusted scope/retirement reducer redacts prose."""
    from dataclasses import replace
    if not isinstance(comment, AdmittedPrComment):
        return replace(comment, body=body)
    comment.protocol_body  # Validate the original binding before deriving a view.
    from .protocol_markers import strip_known_host_footer
    return replace(comment, body=body, admitted_body=body,
                   admitted_protocol_body=strip_known_host_footer(body)[0])


def human_control_eligibility(comment: object, *, actor: tuple[str, int],
                              config: object) -> tuple[bool, str | None]:
    """The existing login-and-ID human policy, without its unconfigured fallback."""
    from .github import _identity_login_key, _trusted_actor_exclusion_reason
    login, user_id = getattr(comment, "author", None), getattr(comment, "author_id", None)
    if user_id == actor[1]:
        if _identity_login_key(login) != _identity_login_key(actor[0]):
            raise AgentLoopError("Protocol history invocation actor identity drift.")
        return True, None
    match, reason = _trusted_actor_exclusion_reason(
        tuple(getattr(config, "human_reviewer_trusted_actors", ()) or ()), login=login, user_id=user_id)
    return match is not None, reason


def _canonical_body(comment: object) -> str | None:
    from .protocol_markers import strip_known_host_footer
    body = getattr(comment, "body", None)
    return strip_known_host_footer(body)[0] if isinstance(body, str) else body


def _project(comment: object, actor_id: int, *, raw_body: str | None,
             amendment_eligible: bool = False, human_trust: tuple[object, ...] = (),
             actor_login: str = "", surface: str = "") -> AdmittedPrComment:
    return AdmittedPrComment(
        author=getattr(comment, "author", None), created_at=getattr(comment, "created_at", None),
        body=raw_body,
        comment_id=getattr(comment, "comment_id", None), author_id=getattr(comment, "author_id", None),
        url=getattr(comment, "url", None), invocation_actor_id=actor_id,
        admitted_body=raw_body, admitted_protocol_body=getattr(comment, "body", None),
        amendment_eligible=amendment_eligible, human_trust=human_trust,
        invocation_actor_login=actor_login, observation_surface=surface)


@pr_history_boundary
def admit_pr_comments(
    runner: Runner, config: AgentLoopConfig, *, pr_number: int,
    comments: Sequence[object], force_probe: bool = False,
    surface_kind: str = PR_THREAD_SURFACE,
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Bind candidates to exhaustive REST history; never cache across reads.

    Reuse is limited to immutable objects from this acquisition. Original raw
    prose stays intact; REST-only protocol records join the ordered observation.
    Its indices belong to the full raw list, never a compacted actor-only list.
    """
    comments = tuple(comments)
    human_trust = tuple(getattr(config, "human_reviewer_trusted_actors", ()) or ())
    surface = comment_thread_surface(surface_kind, pr_number)
    from .github import _identity_login_key
    if comments and all(isinstance(c, AdmittedPrComment) for c in comments) and not force_probe and all(c.human_trust == human_trust for c in comments):
        for comment in comments:
            comment.protocol_body
        if len({c.invocation_actor_id for c in comments}) != 1:
            raise AgentLoopError("PR observation mixes invocation actors.")
        actor = resolve_authenticated_github_actor(runner, config=config, allow_dry_run_read=True)
        if any(c.observation_surface != surface for c in comments):
            raise AgentLoopError("Protocol observation belongs to a different surface.")
        if any(c.invocation_actor_id != actor[1] or
               _identity_login_key(c.invocation_actor_login) != _identity_login_key(actor[0])
               for c in comments):
            raise AgentLoopError("PR observation belongs to a different invocation actor.")
        return comments, actor
    if not force_probe and len(comments) < 100 and not any(map(protocol_candidate, comments)):
        return comments, None
    if runner is None:
        raise AgentLoopError(f"{surface} candidates require authenticated GitHub acquisition.")
    actor = resolve_authenticated_github_actor(runner, config=config, allow_dry_run_read=True)
    listing = read_complete_comment_listing(runner, config=config,
        surface=surface, actor=False)
    if not isinstance(listing, CompleteCommentListing):
        raise AgentLoopError("PR round history cannot be authenticated: " + listing.reason)
    from .logging import log

    def project(comment: object, raw_body: str | None) -> AdmittedPrComment:
        owned = (getattr(comment, "author_id", None) == actor[1]
                 and _identity_login_key(getattr(comment, "author", None)) == _identity_login_key(actor[0]))
        eligible, reason = human_control_eligibility(comment, actor=actor, config=config)
        body = getattr(comment, "body", None) or ""
        if board_control_candidate(comment) and not eligible:
            log(config, f"{surface_kind} #{pr_number} comment {getattr(comment, 'comment_id', None)}: "
                f"ignored reviewer-board amendment candidate: {reason}.")
        if protocol_candidate(comment) and not owned and re.search(r"<!--\s*AGENT_", body, re.I):
            log(config, f"{surface_kind} #{pr_number} comment {getattr(comment, 'comment_id', None)}: "
                "ignored loop record: author is not the invocation actor.")
        return _project(comment, actor[1], raw_body=raw_body,
                        amendment_eligible=eligible, human_trust=human_trust,
                        actor_login=actor[0], surface=surface)

    by_id = {c.comment_id: c for c in listing.comments}
    used, result = set(), []
    for index, comment in enumerate(comments):
        found = by_id.get(getattr(comment, "comment_id", None))
        if found is None and getattr(comment, "comment_id", None) is None:
            matches = [c for c in listing.comments if c.comment_id not in used
                and _identity_login_key(c.author) == _identity_login_key(getattr(comment, "author", None))
                and c.created_at == getattr(comment, "created_at", None) and c.body == _canonical_body(comment)]
            if len(matches) == 1:
                found = matches[0]
        if (protocol_candidate(comment) or protocol_candidate(found)) and (
                found is None or found.body != _canonical_body(comment)
                or found.created_at != getattr(comment, "created_at", None)
                or _identity_login_key(found.author) != _identity_login_key(getattr(comment, "author", None))
                or (getattr(comment, "author_id", None) is not None
                    and found.author_id != comment.author_id)):
            raise AgentLoopError(f"PR round history cannot authenticate comment index {index} "
                f"(comment {getattr(comment, 'comment_id', None)}): content or identity drift.")
        if found is not None:
            if found.comment_id in used:
                raise AgentLoopError("PR round history repeats a comment identity.")
            used.add(found.comment_id)
        result.append(project(found if found is not None else comment, getattr(comment, "body", None)))
    result.extend(project(c, c.body) for c in listing.comments
                  if c.comment_id not in used)
    result.sort(key=lambda c: (c.created_at or "", c.comment_id or 0))
    return tuple(result), actor


def validate_owned_round_envelope(body: str | None, *, index: int, comment_id: object) -> None:
    """Reject malformed owned anchors before any codec can silently skip them.

    Call only after author partitioning; foreign payloads must remain opaque.
    """
    from .round_transport import ROUND_RESUME_MARKER_RE
    if body and re.search(r"<!--\s*AGENT_LOOP_META\b", body, re.I) and not ROUND_RESUME_MARKER_RE.search(body):
        raise AgentLoopError(f"Invalid round metadata envelope at comment index {index} "
                             f"(comment {comment_id}).")


def protocol_comments(comments: Sequence[object]) -> tuple[object, ...]:
    """Codec projection retaining original indices and excluding foreign bytes."""
    from types import SimpleNamespace
    result = []
    for index, comment in enumerate(comments):
        if not isinstance(comment, AdmittedPrComment):
            result.append(comment)
            continue
        body = comment.protocol_body
        validate_owned_round_envelope(body, index=index, comment_id=comment.comment_id)
        result.append(SimpleNamespace(body=body, comment_id=comment.comment_id))
    return tuple(result)


@pr_history_boundary
def extract_admitted_round_records(
    comments: Sequence[object], *, flow: str = "pr",
) -> tuple[PostedRoundRecord, ...]:
    """The sole live PR extraction boundary; masks anchors and chunks together."""
    from .round_state import _decode_round_metadata_records
    if any(protocol_candidate(c) and not isinstance(c, AdmittedPrComment) for c in comments):
        raise AgentLoopError("PR protocol candidates lack an admission binding.")
    return _decode_round_metadata_records(protocol_comments(comments), flow=flow)


def owned_rest_protocol_bodies(
    comments: Sequence[dict[str, object]], *, actor_id: int,
) -> tuple[str, ...]:
    """Mask a complete REST-only codec input without changing its index joins.

    Managed continuity has a deliberately smaller field codec, but must use
    the same pre-decode author partition for anchors and chunks. Its caller
    obtains the complete REST list and invocation actor before entering here.
    """
    from types import SimpleNamespace
    if not isinstance(actor_id, int) or isinstance(actor_id, bool) or actor_id < 1:
        raise AgentLoopError("PR REST protocol history lacks an invocation actor.")
    bodies = []
    for index, comment in enumerate(comments):
        body = comment.get("body")
        user = comment.get("user") if isinstance(comment.get("user"), dict) else {}
        author_id = user.get("id")
        comment_id = comment.get("id")
        if protocol_candidate(SimpleNamespace(body=body)) and (
                not isinstance(author_id, int) or isinstance(author_id, bool) or author_id < 1
                or not isinstance(comment_id, int) or isinstance(comment_id, bool) or comment_id < 1):
            raise AgentLoopError(f"PR round history cannot authenticate REST comment index {index}.")
        owned_body = body if author_id == actor_id and isinstance(body, str) else ""
        validate_owned_round_envelope(owned_body, index=index, comment_id=comment_id)
        bodies.append(owned_body)
    return tuple(bodies)


def admit_issue_board_comments(runner: Runner, config: AgentLoopConfig, *,
                               issue_number: int, comments: Sequence[object],
                               force_probe: bool = False):
    """Shared complete acquisition for issue board-control projections."""
    from .board_change import PLAN_SURFACE, comments_bear_board_change_links
    from .plan_issue_scope import comments_bear_scope_boundary
    if not force_probe and len(comments) < 100 and not (
        any(map(board_control_candidate, comments))
        or comments_bear_board_change_links(comments, surface=PLAN_SURFACE)
        or comments_bear_scope_boundary(comments)
    ):
        return tuple(comments), None
    return admit_pr_comments(runner, config, pr_number=issue_number,
        comments=comments, force_probe=force_probe, surface_kind=ISSUE_THREAD_SURFACE)
