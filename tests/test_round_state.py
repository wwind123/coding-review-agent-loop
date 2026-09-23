"""Round metadata compatibility for the reviewer-board amendment digest (#943)."""

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_scheduling import make_contract
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
)
from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping


def _checkpoint(**overrides):
    values = dict(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=2,
        subject="abc123",
        phase="scheduler-prelaunch",
        scheduler_contract=make_contract(
            ("Codex", "Claude"), "selective-intermediate", None
        ).as_dict(),
        scheduler_previous_sha=None,
        scheduler_current_sha="abc123",
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("Codex", "Claude"),
        scheduler_reasons=("full board",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=0,
    )
    values.update(overrides)
    return PostedRoundMetadata(**values)


def test_legacy_record_encoding_is_unchanged_and_decodes_without_the_field():
    legacy = _checkpoint()
    encoded = _encode_round_metadata(legacy)
    assert "reviewer_board_amendment_digest" not in decode_mapping(encoded)
    decoded = _decode_round_metadata(encoded)
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.reviewer_board_amendment_digest is None
    assert _decode_round_metadata(_encode_round_metadata(decoded)) == decoded


def test_amendment_digest_round_trips():
    digest = "a" * 64
    encoded = _encode_round_metadata(_checkpoint(reviewer_board_amendment_digest=digest))
    assert decode_mapping(encoded)["reviewer_board_amendment_digest"] == digest
    decoded = _decode_round_metadata(encoded)
    assert decoded.reviewer_board_amendment_digest == digest
    assert _decode_round_metadata(_encode_round_metadata(decoded)) == decoded


def test_invalid_amendment_digest_is_rejected():
    with pytest.raises(ValueError, match="reviewer board amendment digest"):
        _checkpoint(reviewer_board_amendment_digest="not-a-digest")
    tampered = _encode_round_metadata(_checkpoint(reviewer_board_amendment_digest="b" * 64))
    # Re-encode a payload whose digest is present but malformed.
    payload = decode_mapping(tampered)
    payload["reviewer_board_amendment_digest"] = "XYZ"
    with pytest.raises(AgentLoopError):
        _decode_round_metadata(encode_mapping(payload))
    payload["reviewer_board_amendment_digest"] = ""
    with pytest.raises(AgentLoopError):
        _decode_round_metadata(encode_mapping(payload))
