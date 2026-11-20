"""Release of carried coder disputes by newer signed requirements (#2025).

A coder dispute that the reviewer still blocks after reconsideration stops the
PR loop for a human decision. A human answers by posting a signed
requirement. This module observes, once per PR round and from freshly read
GitHub records, whether such an answer exists:

* the dispute time is the ``created_at`` of the newest admitted coder
  follow-up round record that lists the item in ``disputed_items``;
* a dispute is released when the loop's effective signed-requirement set has a
  requirement created strictly later than that time.

An unknown dispute time never releases (fail closed). Release only routes the
item back to the coder and reviewers; it never approves or merges, and a
later re-dispute re-arms the stop because its newer record moves the time.
Nothing here is persisted or cached across rounds.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .errors import HumanDecisionRequiredError
from .github import _parse_github_timestamp
from .protocol import StructuredCoderFollowup, UnresolvedReviewItem
from .round_metadata import PostedRoundMetadata, rebuild_resumed_coder_carrier
from .unresolved_items import (
    _raise_if_maintained_disputed_items,
    released_dispute_requirement_ids,
)


@dataclass(frozen=True)
class DisputeRelease:
    """One round's dispute observation: dispute times and released items."""

    dispute_times: Mapping[str, datetime.datetime | None] = field(default_factory=dict)
    released: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


def _carrier_disputes(metadata: object) -> tuple[str, ...]:
    if getattr(metadata, "role", None) != "coder":
        return ()
    raw = getattr(metadata, "raw_structured_coder_response", None)
    if not raw:
        return ()
    carrier = rebuild_resumed_coder_carrier(raw, None)
    if not isinstance(carrier, StructuredCoderFollowup):
        return ()
    return tuple(carrier.disputed_items)


def coder_dispute_times(
    comments: Sequence[object],
    *,
    latest_coder: PostedRoundMetadata | None = None,
) -> dict[str, datetime.datetime | None]:
    """Dispute time per item from admitted coder follow-up round records.

    The newest record that disputes an item wins, so a re-dispute moves the
    time forward. A missing or unparseable ``created_at`` yields ``None``.

    ``latest_coder`` is this invocation's newest coder publication. A round
    may observe a snapshot read before that publication was posted; its
    disputed items then have an unknown time (fail closed), so an older
    record can never release a newer re-dispute.
    """
    from .round_state import _extract_round_metadata_records

    times: dict[str, datetime.datetime | None] = {}
    records = sorted(_extract_round_metadata_records(comments, flow="pr"), key=lambda r: r.index)
    latest_published = latest_coder is None
    for record in records:
        metadata = record.metadata
        if (
            latest_coder is not None
            and metadata.role == "coder"
            and metadata.round_number == latest_coder.round_number
            and metadata.subject == latest_coder.subject
        ):
            latest_published = True
        disputed = _carrier_disputes(metadata)
        if not disputed:
            continue
        created_at = None
        if 0 <= record.index < len(comments):
            created_at = _parse_github_timestamp(getattr(comments[record.index], "created_at", None))
        for item_id in disputed:
            times[item_id] = created_at
    if not latest_published:
        for item_id in _carrier_disputes(latest_coder):
            times[item_id] = None
    return times


def round_dispute_release(
    comments: Sequence[object],
    items: Sequence[UnresolvedReviewItem],
    effective_requirements: Sequence[object],
    *,
    latest_coder: PostedRoundMetadata | None = None,
) -> DisputeRelease:
    """Observe this round's released disputes from fresh GitHub records."""
    times = coder_dispute_times(comments, latest_coder=latest_coder)
    return DisputeRelease(
        dispute_times=times,
        released=released_dispute_requirement_ids(items, times, effective_requirements),
    )


def stop_for_held_disputes(
    must_fix_items: Sequence[UnresolvedReviewItem],
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    release: DisputeRelease,
    post_notice: Callable[[str], object],
    log_failure: Callable[[str], object],
) -> None:
    """Raise for held disputes after a best-effort IDs-only notice.

    The notice never echoes item text or dispute evidence. A failure to post
    it is logged and the original ``HumanDecisionRequiredError`` is raised.
    """
    try:
        _raise_if_maintained_disputed_items(
            must_fix_items,
            prior_items=prior_items,
            released_item_ids=tuple(release.released),
            dispute_times=release.dispute_times,
        )
    except HumanDecisionRequiredError as exc:
        try:
            post_notice(
                "## Human decision required\n\n"
                f"{exc}\n\n"
                "Post a PR comment that states the decision and required action, "
                "and end it with the standalone signature `-- Human Reviewer`. "
                "Then rerun this PR; no coder, CI, qualification, or merge was "
                "started from this decision boundary.\n\n"
                "-- coding-review-agent-loop"
            )
        except Exception as post_error:  # best-effort: never mask the decision error
            log_failure(f"Could not post the human-decision notice: {post_error}")
        raise exc
