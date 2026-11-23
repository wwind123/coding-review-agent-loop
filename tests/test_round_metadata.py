"""Legacy step-back fields in PR round metadata stay decodable and inert (#2097)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coding_review_agent_loop.no_progress import pr_no_progress_observations
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from coding_review_agent_loop.round_transport import (
    ROUND_RESUME_MARKER_RE,
    decode_mapping,
    encode_mapping,
)

HEAD = "c" * 40
ENTRY = {
    "phase": "pr", "reviewer": "Codex", "trigger_round": 1, "trigger_head": HEAD,
    "anchor": {"path": "src/a.py", "start": 1, "end": 5},
}


def _bodies():
    item = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Fix it.", status="blocking",
    )
    review = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=1, subject=HEAD,
        state="blocking", new_items=(item,),
    )
    coder = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=HEAD,
        followup_dispatch_head=HEAD, prior_items=(item,),
    )
    return [_attach_round_metadata("review", review), _attach_round_metadata("coder", coder)]


def _rewrite_coder(body, value):
    match = list(ROUND_RESUME_MARKER_RE.finditer(body))[-1]
    payload = dict(decode_mapping(match.group("payload")))
    payload["step_back_entries"] = value
    return body[: match.start("payload")] + encode_mapping(payload) + body[match.end("payload"):]


@pytest.mark.parametrize(
    "value,status",
    [(None, "absent"), ([ENTRY], "valid"), ("not-a-list", "invalid"), ([{"phase": "pr"}], "invalid")],
    ids=["absent", "valid", "malformed-type", "malformed-entry"],
)
def test_legacy_step_back_entries_parse_and_never_change_observations(value, status):
    """`pr-resume-legacy`: no parse error; the detector's input is identical."""
    reference_bodies = _bodies()
    bodies = list(reference_bodies)
    if value is not None:
        bodies[1] = _rewrite_coder(bodies[1], value)
    records = _extract_round_metadata_records(
        [SimpleNamespace(body=body) for body in bodies], flow="pr"
    )
    coder = next(r.metadata for r in records if r.metadata.role == "coder")
    assert coder.step_back_status == status
    reference = _extract_round_metadata_records(
        [SimpleNamespace(body=body) for body in reference_bodies], flow="pr"
    )
    assert pr_no_progress_observations(records) == pr_no_progress_observations(reference)
    assert [o.round_number for o in pr_no_progress_observations(records)] == [1]
