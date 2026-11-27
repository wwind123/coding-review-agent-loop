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


# --- Plan review implementation notes (#2145) --------------------------------


def _plan_reviewer_metadata(**fields):
    return PostedRoundMetadata(
        flow="plan", role="reviewer", agent="Codex", round_number=2,
        subject="a" * 64, state="approved", **fields,
    )


def test_implementation_notes_round_trip_through_round_metadata():
    from coding_review_agent_loop.round_metadata import _encode_round_metadata
    from coding_review_agent_loop.round_metadata import _decode_round_metadata

    metadata = _plan_reviewer_metadata(implementation_notes=("Update the fixture.", "Refresh SIBLINGS."))
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.implementation_notes == ("Update the fixture.", "Refresh SIBLINGS.")


def test_legacy_record_without_notes_is_byte_identical_and_decodes_empty():
    """Matrix row `legacy-record`."""
    from coding_review_agent_loop.round_metadata import (
        _decode_round_metadata,
        _encode_round_metadata,
        decode_mapping,
    )

    legacy = _plan_reviewer_metadata()
    explicit_empty = _plan_reviewer_metadata(implementation_notes=())
    encoded = _encode_round_metadata(legacy)
    assert _encode_round_metadata(explicit_empty) == encoded
    payload = decode_mapping(encoded)
    assert "implementation_notes" not in payload
    decoded = _decode_round_metadata(encoded)
    assert decoded.implementation_notes == ()
    assert decoded == legacy
    # A record written before this change, re-encoded, keeps its exact bytes.
    assert _encode_round_metadata(decoded) == encoded


@pytest.mark.parametrize("value", [None, []])
def test_null_or_empty_notes_decode_as_none(value):
    from coding_review_agent_loop.round_metadata import decode_implementation_notes

    assert decode_implementation_notes(value) == ()


@pytest.mark.parametrize("value", ["one note", [1], [""], ["x" * 5000]])
def test_malformed_notes_fail_closed(value):
    from coding_review_agent_loop.round_metadata import decode_implementation_notes

    with pytest.raises(ValueError):
        decode_implementation_notes(value)


# --- #2156: prior_plan_content_identity on planner records ------------------

from coding_review_agent_loop.round_metadata import (  # noqa: E402
    _decode_round_metadata,
    _encode_round_metadata,
)
from coding_review_agent_loop.round_transport import (  # noqa: E402
    decode_mapping as _decode_mapping_2156,
    encode_mapping as _encode_mapping_2156,
)


def _planner_2156(**changes):
    return PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=3,
        subject="b" * 64, prior_plan_subject="a" * 64, **changes,
    )


def test_2156_a_valid_content_identity_round_trips():
    """`metadata-strict`: a lowercase 64-hex identity survives encode/decode."""
    metadata = _planner_2156(prior_plan_content_identity="c" * 64)
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.prior_plan_content_identity == "c" * 64
    assert decoded.prior_plan_subject == "a" * 64


def test_2156_an_absent_identity_decodes_as_none_and_is_not_encoded():
    encoded = _encode_round_metadata(_planner_2156())
    assert "prior_plan_content_identity" not in _decode_mapping_2156(encoded)
    assert _decode_round_metadata(encoded).prior_plan_content_identity is None


@pytest.mark.parametrize(
    "value", [None, "", "C" * 64, "c" * 63, "g" * 64, 7, ["c" * 64]]
)
def test_2156_a_malformed_identity_is_rejected(value):
    """`metadata-strict`: present-but-malformed never decodes as legacy absence."""
    payload = dict(_decode_mapping_2156(_encode_round_metadata(_planner_2156())))
    payload["prior_plan_content_identity"] = value
    from coding_review_agent_loop.errors import AgentLoopError

    with pytest.raises((ValueError, AgentLoopError), match="prior.plan.content.identity"):
        _decode_round_metadata(_encode_mapping_2156(payload))
    if isinstance(value, str):
        with pytest.raises(ValueError):
            _planner_2156(prior_plan_content_identity=value)
