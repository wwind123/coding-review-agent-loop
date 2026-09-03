import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.round_transport as transport
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _decode_round_metadata,
    _encode_round_metadata,
    _extract_round_metadata_records,
    _prior_item_ledger_signature,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem


def _random_text(size: int) -> str:
    return base64.urlsafe_b64encode(os.urandom(size)).decode("ascii")


def _comment(payload: dict[str, object], body: str = "Visible response") -> str:
    return f"{body}\n<!-- AGENT_LOOP_META: {transport.encode_mapping(payload)} -->"


def _anchor_payload(anchor: str) -> dict[str, object]:
    match = list(transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    return transport.decode_mapping(match.group("payload"))


def test_encode_decode_mapping_round_trip_and_legacy_base64() -> None:
    payload = {"emoji": "✓", "items": ["one", 2]}

    assert transport.decode_mapping(transport.encode_mapping(payload)) == payload
    legacy = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    assert transport.decode_mapping(legacy) == payload


def test_is_round_transport_sidecar() -> None:
    sidecar = transport._sidecar({"field": "canonical_plan"})
    assert transport.is_round_transport_sidecar(sidecar)
    assert not transport.is_round_transport_sidecar("ordinary agent output")


def test_prepare_round_comment_spills_only_until_anchor_fits() -> None:
    review = _random_text(46_000)
    payload = {
        "canonical_reviewer_response": review,
        "raw_structured_coder_response": _random_text(500),
        "canonical_plan": _random_text(500),
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    assert len(prepared) > 1
    anchor_payload = _anchor_payload(prepared[-1])
    assert isinstance(anchor_payload["canonical_reviewer_response"], dict)
    assert anchor_payload["raw_structured_coder_response"] == payload["raw_structured_coder_response"]
    assert anchor_payload["canonical_plan"] == payload["canonical_plan"]
    hydrated, missing = transport.hydrate_mapping(anchor_payload, prepared)
    assert missing == set()
    assert hydrated == payload


def test_prepare_round_comment_spills_multiple_fields_in_fixed_order() -> None:
    payload = {
        "canonical_reviewer_response": _random_text(50_000),
        "raw_structured_coder_response": _random_text(50_000),
        "canonical_plan": _random_text(50_000),
    }

    prepared = transport.prepare_round_comment(_comment(payload))

    anchor_payload = _anchor_payload(prepared[-1])
    expected_fields = tuple(field for field in transport._SPILL_FIELDS if field in payload)
    assert all(isinstance(anchor_payload[field], dict) for field in expected_fields)
    sidecar_fields = []
    for sidecar in prepared[:-1]:
        match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(sidecar)
        assert match is not None
        sidecar_fields.append(json.loads(base64.urlsafe_b64decode(match.group("payload")))["field"])
    assert list(dict.fromkeys(sidecar_fields)) == list(expected_fields)


def test_hydrate_mapping_reports_missing_duplicate_and_corrupt_sidecars() -> None:
    payload = {"canonical_reviewer_response": _random_text(46_000)}
    prepared = transport.prepare_round_comment(_comment(payload))
    sidecars, anchor = prepared[:-1], prepared[-1]
    anchor_payload = _anchor_payload(anchor)

    hydrated, missing = transport.hydrate_mapping(anchor_payload, (*sidecars, sidecars[0]))
    assert missing == set()
    assert hydrated == payload

    hydrated, missing = transport.hydrate_mapping(anchor_payload, sidecars[1:])
    assert missing == {"canonical_reviewer_response"}
    assert hydrated["canonical_reviewer_response"] is None

    match = transport.ROUND_TRANSPORT_SIDECAR_RE.search(sidecars[0])
    assert match is not None
    corrupt = json.loads(base64.urlsafe_b64decode(match.group("payload")))
    corrupt["data"] = "corrupt"
    corrupt_sidecar = transport._sidecar(corrupt)
    hydrated, missing = transport.hydrate_mapping(anchor_payload, (corrupt_sidecar, *sidecars[1:]))
    assert missing == {"canonical_reviewer_response"}
    assert hydrated["canonical_reviewer_response"] is None


def test_hydrate_mapping_bounds_decompression(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transport, "_MAX_DECOMPRESSED", 10)
    raw = b"a" * 100
    packed = transport.zlib.compress(raw)
    reference = {
        "$round_transport_spill": "anchor",
        "field": "canonical_plan",
        "parts": 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "spill": hashlib.sha256(packed).hexdigest(),
    }
    sidecar = transport._sidecar(
        {
            "anchor": "anchor",
            "field": "canonical_plan",
            "index": 0,
            "count": 1,
            "sha256": reference["sha256"],
            "spill": reference["spill"],
            "data": transport._b64(packed),
        }
    )

    hydrated, missing = transport.hydrate_mapping({"canonical_plan": reference}, (sidecar,))

    assert missing == {"canonical_plan"}
    assert hydrated["canonical_plan"] is None


def test_hydrate_mapping_reports_invalid_compressed_sidecar() -> None:
    packed = b"not a zlib stream"
    reference = {
        "$round_transport_spill": "anchor",
        "field": "canonical_plan",
        "parts": 1,
        "sha256": hashlib.sha256(b"expected raw payload").hexdigest(),
        "spill": hashlib.sha256(packed).hexdigest(),
    }
    sidecar = transport._sidecar(
        {
            "anchor": "anchor",
            "field": "canonical_plan",
            "index": 0,
            "count": 1,
            "sha256": reference["sha256"],
            "spill": reference["spill"],
            "data": transport._b64(packed),
        }
    )

    hydrated, missing = transport.hydrate_mapping({"canonical_plan": reference}, (sidecar,))

    assert missing == {"canonical_plan"}
    assert hydrated["canonical_plan"] is None


@pytest.mark.parametrize(
    ("field", "phase"),
    (
        ("canonical_reviewer_response", "provisional"),
        ("raw_structured_coder_response", "authoritative"),
        ("canonical_plan", "authoritative"),
    ),
)
def test_resume_rejects_missing_spilled_canonical_metadata(field: str, phase: str) -> None:
    values = {field: _random_text(46_000)}
    metadata = PostedRoundMetadata(
        flow="pr",
        role="reviewer",
        agent="codex",
        round_number=1,
        subject="head",
        phase=phase,
        **values,
    )
    body = _attach_round_metadata("Visible review", metadata)
    anchor = transport.prepare_round_comment(body)[-1]

    with pytest.raises(AgentLoopError, match="Incomplete round metadata"):
        _extract_round_metadata_records((SimpleNamespace(body=anchor),), flow="pr")


def test_round_metadata_decode_uses_mapping_without_reencoding() -> None:
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="codex", round_number=1, subject="head"
    )

    assert _decode_round_metadata(_encode_round_metadata(metadata)) == metadata


def test_round_metadata_preserves_mixed_legacy_claim_and_modern_evidence_exactly() -> None:
    legacy_and_modern_item = UnresolvedReviewItem(
        item_id="item-4",
        reviewer="Anthropic Claude",
        source_round=3,
        text=(
            "Only three of fourteen locales were updated.\n\n"
            "Update from Anthropic Claude: Original scope evidence was rechecked."
        ),
        status="blocking",
        source_status="blocking",
        notes=("OpenAI Codex: all fourteen locales and 49 keys were evaluated.",),
    )
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=4, subject="head",
        prior_items=(legacy_and_modern_item,),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.prior_items == (legacy_and_modern_item,)
    assert _prior_item_ledger_signature(decoded.prior_items) == _prior_item_ledger_signature(
        (legacy_and_modern_item,)
    )
