"""Author admission for one live PR observation, before any transport decoding."""
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
    comment_thread_surface, read_complete_comment_listing,
    resolve_authenticated_github_actor, strip_bot_login_suffix)


def protocol_candidate(comment: object) -> bool:
    """Recognize malformed record-shaped content without decoding payloads."""
    body = getattr(comment, "body", None)
    return isinstance(body, str) and re.search(
        r"<!--\s*AGENT_(?:LOOP_META|LOOP_SIDECAR|PR_BOARD_CHANGE)\b", body, re.I) is not None



@dataclass(frozen=True)
class AdmittedPrComment(IssueComment):
    """Raw prose and identity, with separately bound protocol eligibility."""
    invocation_actor_id: int = 0
    admitted_body: str | None = None
    admitted_protocol_body: str | None = None

    @property
    def protocol_body(self) -> str | None:
        if self.body != self.admitted_body:
            raise AgentLoopError(f"PR comment {self.comment_id} changed after admission.")
        return self.admitted_protocol_body if self.author_id == self.invocation_actor_id else ""


def _canonical_body(comment: object) -> str | None:
    from .protocol_markers import strip_known_host_footer
    body = getattr(comment, "body", None)
    return strip_known_host_footer(body)[0] if isinstance(body, str) else body


def _project(comment: object, actor_id: int, *, raw_body: str | None) -> AdmittedPrComment:
    return AdmittedPrComment(
        author=getattr(comment, "author", None), created_at=getattr(comment, "created_at", None),
        body=raw_body,
        comment_id=getattr(comment, "comment_id", None), author_id=getattr(comment, "author_id", None),
        url=getattr(comment, "url", None), invocation_actor_id=actor_id,
        admitted_body=raw_body, admitted_protocol_body=getattr(comment, "body", None))


@pr_history_boundary
def admit_pr_comments(
    runner: Runner, config: AgentLoopConfig, *, pr_number: int,
    comments: Sequence[object], force_probe: bool = False,
) -> tuple[tuple[object, ...], tuple[str, int] | None]:
    """Bind candidates to exhaustive REST history; never cache across reads.

    Reuse is limited to immutable objects from this acquisition. Original raw
    prose stays intact; REST-only protocol records join the ordered observation.
    Its indices belong to the full raw list, never a compacted actor-only list.
    """
    comments = tuple(comments)
    if comments and all(isinstance(c, AdmittedPrComment) for c in comments) and not force_probe:
        for comment in comments:
            comment.protocol_body
        if len({c.invocation_actor_id for c in comments}) != 1:
            raise AgentLoopError("PR observation mixes invocation actors.")
        actor = resolve_authenticated_github_actor(runner, config=config, allow_dry_run_read=True)
        if comments[0].invocation_actor_id != actor[1]:
            raise AgentLoopError("PR observation belongs to a different invocation actor.")
        return comments, actor
    if not force_probe and len(comments) < 100 and not any(map(protocol_candidate, comments)):
        return comments, None
    actor = resolve_authenticated_github_actor(runner, config=config, allow_dry_run_read=True)
    listing = read_complete_comment_listing(runner, config=config,
        surface=comment_thread_surface(PR_THREAD_SURFACE, pr_number), actor=False)
    if not isinstance(listing, CompleteCommentListing):
        raise AgentLoopError("PR round history cannot be authenticated: " + listing.reason)
    by_id = {c.comment_id: c for c in listing.comments}
    used, result = set(), []
    for index, comment in enumerate(comments):
        found = by_id.get(getattr(comment, "comment_id", None))
        if found is None and getattr(comment, "comment_id", None) is None:
            matches = [c for c in listing.comments if c.comment_id not in used
                and strip_bot_login_suffix(c.author) == strip_bot_login_suffix(getattr(comment, "author", None))
                and c.created_at == getattr(comment, "created_at", None) and c.body == _canonical_body(comment)]
            if len(matches) == 1:
                found = matches[0]
        if protocol_candidate(comment) and (found is None or found.body != _canonical_body(comment)
                or found.created_at != getattr(comment, "created_at", None)
                or strip_bot_login_suffix(found.author) != strip_bot_login_suffix(getattr(comment, "author", None))):
            raise AgentLoopError(f"PR round history cannot authenticate comment index {index} "
                f"(comment {getattr(comment, 'comment_id', None)}): content or identity drift.")
        if found is not None:
            if found.comment_id in used:
                raise AgentLoopError("PR round history repeats a comment identity.")
            used.add(found.comment_id)
        result.append(_project(found if found is not None else comment, actor[1], raw_body=getattr(comment, "body", None)))
    result.extend(_project(c, actor[1], raw_body=c.body) for c in listing.comments
                  if c.comment_id not in used and protocol_candidate(c))
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
