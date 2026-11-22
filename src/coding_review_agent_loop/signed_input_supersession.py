"""Close a PR round superseded by new signed input, then open the next (#2054).

When the operator follows the conflicting-classification hint and posts a
new signed comment, every review already published in the unreconciled round
was formed against the old signed set.  Resume used to reject them and then
refuse to run the round again beside its public peers, asking the operator to
delete them.  This module instead closes the round with a tool-owned
supersession record, so the existing closed-round rollover (#1553) opens
round N+1 against the new set.  The published reviews stay on GitHub as
history and never count toward approval, the ledger, panel evidence or
evidence obligations.

Rollover applies only when every rejected review was rejected for pure growth
(a strict digest superset at the same head, round, seat and reviewer
identity) and every such review proves it is unedited: either its frozen
publication carrier on this host matches the public comments completely, or,
without a carrier, GitHub's edit history proves that its anchor and sidecar
comments were never edited.  Anything else, including a withdrawn or edited
signed source, keeps today's partial-round refusal unchanged.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable, Iterable, Mapping, Sequence

from .errors import AgentLoopError
from .logging import log
from .publication_resume import PublicationCarrier, classify_frozen_sequence
from .pr_round_ingestion import record_body
from .response_validation import (
    _reviewer_requirement_identity_ids,
    signed_input_growth_rejection,
)
from .review_scope import scope_present
from .round_metadata import PostedRoundRecord, _attach_round_metadata
from .round_transport import (
    attachment_keys,
    is_round_transport_sidecar,
    referenced_attachment_keys,
)
from .signed_input_record import SIGNED_INPUT_SUPERSESSION_PHASE, decode_supersession_payload

EDIT_HISTORY_QUERY = """
query SignedInputEditHistory($owner: String!, $name: String!, $number: Int!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      comments(first: 100, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          databaseId lastEditedAt includesCreatedEdit
          userContentEdits(first: 1) { totalCount nodes { editedAt deletedAt diff } }
        }
      }
    }
  }
}
""".strip()

_MAX_EDIT_HISTORY_PAGES = 50
# Publication states of a reviewer that never published this round.
_UNPUBLISHED_STATES = frozenset({"absent", "fresh"})


def read_pr_comment_edit_states(
    runner: object, *, config: object, pr_number: int, wanted: Iterable[int]
) -> dict[int, tuple[tuple[str, str], ...] | None]:
    """Each wanted comment's body revisions: ``()`` never edited, ``None`` unknown.

    A failed, malformed or unbounded read leaves the comment out, which the
    caller treats as unproven (fail closed).
    """
    from .github import _graphql_repository_parts, _parse_body_revisions
    from .github_retry import run_gh_read
    from .workdirs import github_api_cwd

    remaining = {int(item) for item in wanted}
    states: dict[int, tuple[tuple[str, str], ...] | None] = {}
    owner, name = _graphql_repository_parts(config.repo)  # type: ignore[attr-defined]
    after: str | None = None
    for _page in range(_MAX_EDIT_HISTORY_PAGES):
        if not remaining:
            break
        command = [
            config.gh_cmd, "api", "graphql",  # type: ignore[attr-defined]
            "-f", f"query={EDIT_HISTORY_QUERY}",
            "-f", f"owner={owner}", "-f", f"name={name}", "-F", f"number={pr_number}",
        ]
        if after is not None:
            command += ["-f", f"after={after}"]
        try:
            result = run_gh_read(runner, command, cwd=github_api_cwd(), check=False)
            payload = json.loads(result.stdout or "{}") if result.returncode == 0 else None
        except (AgentLoopError, OSError, ValueError):
            return states
        if not isinstance(payload, dict) or payload.get("errors"):
            return states
        try:
            comments = payload["data"]["repository"]["pullRequest"]["comments"]
            nodes, page_info = comments["nodes"], comments["pageInfo"]
        except (KeyError, TypeError):
            return states
        if not isinstance(nodes, list) or not isinstance(page_info, dict):
            return states
        for node in nodes:
            comment_id = node.get("databaseId") if isinstance(node, dict) else None
            if isinstance(comment_id, int) and comment_id in remaining:
                remaining.discard(comment_id)
                states[comment_id] = _parse_body_revisions(node)
        if page_info.get("hasNextPage") is not True:
            break
        after = page_info.get("endCursor")
        if not isinstance(after, str):
            break
    return states


def review_comment_ids(comments: Sequence[object], record: PostedRoundRecord) -> tuple[int, ...] | None:
    """The record's anchor and sidecar comment IDs, anchor last; ``None`` if unknown."""
    if not 0 <= record.index < len(comments):
        return None
    anchor = comments[record.index]
    anchor_id = getattr(anchor, "comment_id", None)
    sidecars: list[int] = []
    try:
        # A comment changed after admission raises here: unproven, no rollover.
        body = record_body(anchor)
        if not isinstance(anchor_id, int) or not isinstance(body, str):
            return None
        refs = set(referenced_attachment_keys(body))
        for index, comment in enumerate(comments):
            text = record_body(comment)
            if index == record.index or not isinstance(text, str) or not is_round_transport_sidecar(text):
                continue
            if any(key in refs or (key[0], "*") in refs for key in attachment_keys(text)):
                comment_id = getattr(comment, "comment_id", None)
                if not isinstance(comment_id, int):
                    return None
                sidecars.append(comment_id)
    except AgentLoopError:
        return None
    return (*sorted(sidecars), anchor_id)


def carrier_proves_unedited(
    raw_carrier: Mapping[str, object] | None,
    listing_comments: Sequence[object] | None,
    comment_ids: Sequence[int],
) -> bool:
    """A frozen carrier matches the public comments completely and exactly.

    ``prefix`` with ``matched_count == 0`` while the record's anchor is public
    means a published body no longer matches its frozen bytes: tampering, never
    an unpublished carrier.  Every kind other than ``complete`` fails.
    """
    if raw_carrier is None or listing_comments is None:
        return False
    try:
        carrier = PublicationCarrier.from_dict(dict(raw_carrier))
    except (ValueError, TypeError):
        return False
    result = classify_frozen_sequence(carrier, listing_comments)
    if result.kind != "complete":
        return False
    matched = [comment_id for comment_id in result.matched if comment_id is not None]
    return (
        len(matched) == len(comment_ids)
        and set(matched) == set(comment_ids)
        and matched[-1] == comment_ids[-1]
    )


def edit_history_proves_unedited(
    states: Mapping[int, tuple[tuple[str, str], ...] | None], comment_ids: Sequence[int]
) -> bool:
    """Only a proven never-edited history, ``()``, for every comment passes."""
    return bool(comment_ids) and all(
        comment_id in states and states[comment_id] == () for comment_id in comment_ids
    )


def supersede_for_signed_growth(
    runner: object,
    *,
    config: object,
    pr_number: int,
    comments: Sequence[object],
    current_resume: object,
    round_number: int,
    allowed_rounds: int,
    head_sha: str,
    human_requirements: Sequence[object],
    approved_plan_context: object,
    acquisition_contract: Mapping[str, tuple[object, ...]] | None,
    seat_binding: object,
    prior_items: Sequence[object],
    configured_reviewer_names: Sequence[str],
    completed_by_name: dict[str, PostedRoundRecord],
    resumed_by_name: Mapping[str, PostedRoundRecord],
    rejection_reasons: dict[str, str],
    spool: object,
    bound_metadata: Callable[..., object],
    post_comment: Callable[..., None],
    unresolved_items: Sequence[object],
    blocked: bool,
) -> object:
    """Return ``current_resume`` closed by a posted supersession record, or unchanged.

    Nothing is posted unless every condition holds; the caller's existing
    refusal then fires with its existing cause and recovery listing.
    """
    if (
        blocked
        or current_resume is None
        or getattr(current_resume, "round_number", None) != round_number
        or getattr(current_resume, "reconciliation_record", None) is not None
        or getattr(current_resume, "evidence_boundary", None) is not None
        # At least one published review must be rejected for growth.
        or not completed_by_name
        or set(completed_by_name) & set(resumed_by_name)
        or set(completed_by_name) - set(rejection_reasons)
    ):
        return current_resume
    for record in completed_by_name.values():
        if not signed_input_growth_rejection(
            record, human_requirements, approved_plan_context, acquisition_contract,
            seat_binding=seat_binding, round_number=round_number, head_sha=head_sha,
            reconciled=False,
        ):
            return current_resume
    # A reviewer that did not publish must provably never have begun: a carrier
    # partly public without its anchor keeps its PublicationResumeStop.
    states: dict[str, tuple[str, Mapping[str, object] | None]] = {}
    for name in configured_reviewer_names:
        state = spool.publication_state(name)  # type: ignore[attr-defined]
        states[name] = state
        if name not in completed_by_name and state[0] not in _UNPUBLISHED_STATES:
            return current_resume
    # A one-reviewer board has no same-round peer: without a retained carrier
    # today's fresh same-round turn already works (and keeps the reviewer's
    # own durable evidence requests in its round).  A retained carrier cannot
    # be resumed against the grown signed set, so that case rolls over too.
    # On a larger board even one public review is a peer for every reviewer
    # still to launch, so today's path would refuse.
    if len(configured_reviewer_names) < 2 and not any(
        states.get(name, ("absent", None))[0] == "carrier" for name in completed_by_name
    ):
        return current_resume
    ids_by_name: dict[str, tuple[int, ...]] = {}
    for name, record in completed_by_name.items():
        ids = review_comment_ids(comments, record)
        if ids is None:
            return current_resume
        ids_by_name[name] = ids
    routes: dict[str, str] = {}
    edit_states: dict[int, tuple[tuple[str, str], ...] | None] | None = None
    for name in completed_by_name:
        state, raw = states.get(name, ("malformed", None))
        if state == "carrier":
            from .github import comment_thread_surface, read_complete_comment_listing

            listing = read_complete_comment_listing(
                runner, config=config, surface=comment_thread_surface("pr", pr_number), actor=True,  # type: ignore[arg-type]
            )
            if not carrier_proves_unedited(raw, getattr(listing, "comments", None), ids_by_name[name]):
                return current_resume
            routes[name] = "carrier"
        elif state == "absent":
            if edit_states is None:
                edit_states = read_pr_comment_edit_states(
                    runner, config=config, pr_number=pr_number,
                    wanted=[i for ids in ids_by_name.values() for i in ids],
                )
            if not edit_history_proves_unedited(edit_states, ids_by_name[name]):
                return current_resume
            routes[name] = "edit-history"
        else:
            # fresh/legacy/malformed beside a public anchor is unproven.
            return current_resume
    previous_ids = sorted({
        str(item)
        for record in completed_by_name.values()
        for item in record.metadata.surfaced_reviewer_requirement_ids
    })
    current_ids = sorted(_reviewer_requirement_identity_ids(human_requirements))  # type: ignore[arg-type]
    added = sorted(set(current_ids) - set(previous_ids))
    if round_number >= allowed_rounds:
        # No round is left: stop through the ordinary round limit, post nothing.
        from .pr_loop_support import _round_limit_diagnostic

        raise AgentLoopError(_round_limit_diagnostic(
            pr_number=pr_number, round_number=allowed_rounds, items=tuple(unresolved_items),
            current_head_sha=head_sha,
            sub_item_stall_rounds=getattr(config, "sub_item_stall_rounds", None),
        ))
    high_water = max(int(getattr(current_resume, "next_unresolved_item_number", 1)) - 1, 0)
    payload = decode_supersession_payload({
        "round": round_number,
        "head": head_sha,
        "superseded": [
            {"agent": name, "comment_id": ids_by_name[name][-1], "route": routes[name]}
            for name in sorted(completed_by_name)
        ],
        "previous_requirement_ids": previous_ids,
        "requirement_ids": current_ids,
        "item_number_high_water": high_water,
    }, round_number=round_number, subject=head_sha)
    if payload is None:
        return current_resume
    metadata = bound_metadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=round_number,
        # Round N's input ledger as resume rebuilt it, including the durable
        # evidence requests it restores for reviews invalidated by new input
        # (#1922), so a signed answer can clear them; superseded reviews add
        # no dispositions, votes or new findings.
        subject=head_sha, prior_items=tuple(prior_items),
        phase=SIGNED_INPUT_SUPERSESSION_PHASE,
        surfaced_reviewer_requirement_ids=tuple(current_ids),
        signed_input_supersession=payload,
    )
    if scope_present(metadata):
        return current_resume
    body = _attach_round_metadata(
        f"Review round {round_number} is superseded by new signed input "
        f"(added {', '.join(added)}). Its published reviews of "
        f"{', '.join(sorted(completed_by_name))} stay as history and do not count toward "
        f"approval; round {round_number + 1} reviews the head again against the current "
        "signed sources. No comment needs to be deleted.",
        metadata,
    )
    # Save before emit: the boundary is durable before any round-N+1 launch.
    post_comment(runner, config=config, pr_number=pr_number, body=body)
    log(
        config,  # type: ignore[arg-type]
        f"PR #{pr_number}: round {round_number} superseded by new signed input "
        f"(added {', '.join(added)}); opening round {round_number + 1}",
    )
    completed_by_name.clear()
    rejection_reasons.clear()
    return dataclasses.replace(
        current_resume,  # type: ignore[arg-type]
        reconciliation_record=PostedRoundRecord(index=len(comments), metadata=metadata, body=body),
        completed_reviews=(),
        reconciled=True,
    )
