import base64
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.round_transport as transport
import coding_review_agent_loop.comment_rendering as comment_rendering
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    QualificationCheckpoint,
    _attach_round_metadata,
    _decode_round_metadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
    _extract_round_metadata_records,
    _prior_item_ledger_signature,
)
from coding_review_agent_loop.protocol import (
    MACHINE_AUTHORITY,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    UNKNOWN_MACHINE_AUTHORITY,
)
from coding_review_agent_loop.review_scheduling import ReviewSchedulingContract
from coding_review_agent_loop.unresolved_items import _apply_unresolved_item_dispositions
from coding_review_agent_loop.protocol_markers import TrustedBody


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


def test_prepare_round_comment_preserves_trusted_carrier_without_spill() -> None:
    carrier = TrustedBody.join(
        TrustedBody.current_untrusted_visible("first"),
        TrustedBody.current_untrusted_visible("second"),
    )

    prepared = transport.prepare_round_comment(carrier)

    assert prepared == (carrier,)
    assert prepared[0] is carrier
    assert prepared[0].segments == carrier.segments


def test_prepare_round_comment_spill_keeps_existing_segment_provenance() -> None:
    recommendation = {
        "strategy": "one-shot",
        "rationale": _random_text(50_000),
        "staging_feasibility": "inseparable",
        "scope_items": [{
            "scope_item_id": "scope-1",
            "requirement": "Implement the requested behavior.",
            "acceptance_criteria": ["The focused regression passes."],
        }],
        "coupling_constraints": [],
        "one_shot_delivery": {
            "deliverables": ["Implementation."],
            "acceptance_criteria": ["The regression passes."],
            "covered_scope_item_ids": ["scope-1"],
        },
        "child_stages": [],
        "retained_parent_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "final_integration_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "caveats": [],
    }
    encoded = transport._b64(
        json.dumps(recommendation, separators=(",", ":"), sort_keys=True).encode()
    )
    marker_text = f"<!-- AGENT_EXECUTION_RECOMMENDATION: {encoded} -->"
    carrier = TrustedBody.join(
        TrustedBody.current_untrusted_visible("first"),
        TrustedBody.current_untrusted_visible("second"),
        TrustedBody.marker("AGENT_EXECUTION_RECOMMENDATION", marker_text),
    )

    prepared = transport.prepare_round_comment(carrier)

    assert len(prepared) > 1
    assert prepared[-1].segments[0] == ("first", None)
    assert prepared[-1].segments[1] == ("second", None)
    assert prepared[-1].segments[2][1] == "AGENT_EXECUTION_RECOMMENDATION"


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


def test_oversized_execution_recommendation_uses_bounded_lossless_sidecar() -> None:
    recommendation = {
        "strategy": "one-shot",
        "rationale": _random_text(50_000),
        "staging_feasibility": "inseparable",
        "scope_items": [{
            "scope_item_id": "scope-1",
            "requirement": "Implement the requested behavior.",
            "acceptance_criteria": ["The focused regression passes."],
        }],
        "coupling_constraints": [],
        "one_shot_delivery": {
            "deliverables": ["Implementation and tests."],
            "acceptance_criteria": ["The focused regression passes."],
            "covered_scope_item_ids": ["scope-1"],
        },
        "child_stages": [],
        "retained_parent_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "final_integration_work": {
            "status": "none", "deliverables": [], "acceptance_criteria": [],
            "covered_scope_item_ids": [],
        },
        "caveats": [],
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(recommendation, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    body = _random_text(5_000) + f"\n<!-- AGENT_EXECUTION_RECOMMENDATION: {encoded} -->"

    prepared = transport.prepare_round_comment(body)
    anchor = str(prepared[-1])

    assert len(anchor) <= transport.MAX_GITHUB_BODY_CHARS
    marker = list(comment_rendering.EXECUTION_RECOMMENDATION_MARKER_RE.finditer(anchor))[-1]
    assert comment_rendering.decode_execution_recommendation_marker(
        marker.group("payload"), bodies=tuple(map(str, prepared))
    ) == recommendation


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


def test_round_metadata_round_trips_architecture_identity_and_impact() -> None:
    identity = {
        "repository": "OWNER/REPO",
        "path": "ARCHITECTURE.md",
        "revision": "a" * 40,
        "blob_oid": "b" * 40,
        "sha256": "c" * 64,
        "availability": "available",
        "size": 128,
    }
    impact = {
        "status": "unchanged",
        "rationale": "Only an internal test helper changed.",
        "affected_components": [],
        "dependencies": [],
        "execution_data_flows": [],
        "persistence": [],
        "public_contracts": [],
        "security_boundaries": [],
        "canonical_document_action": "no-change",
        "canonical_document_path": None,
        "canonical_document_rationale": "No canonical update is needed.",
    }
    metadata = PostedRoundMetadata(
        flow="pr", role="reviewer", agent="codex", round_number=1, subject="head",
        architecture_identity=identity, architecture_impact=impact,
        architecture_contract_version=1,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.architecture_identity == identity
    assert decoded.architecture_impact == impact
    assert decoded.architecture_contract_version == 1


def test_round_metadata_round_trips_bounded_local_test_evidence() -> None:
    from coding_review_agent_loop.local_test_evidence import bounded_evidence_for_round

    evidence = bounded_evidence_for_round({
        "observations": [{
            "command": ["python", "-m", "pytest", "tests/test_protocol.py", "-q"],
            "outcome": "failed", "provenance": "parent-observed",
            "receipt_id": "receipt-1", "environment": "unknown",
        }]
    })
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="codex", round_number=2,
        subject="head", local_test_evidence=evidence,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.local_test_evidence is not None
    payload = json.loads(decoded.local_test_evidence)
    assert payload["observations"][0]["receipt_id"] == "receipt-1"
    assert payload["observations"][0]["environment"] == "identity-unknown"


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


def test_round_metadata_preserves_owner_future_disposition_across_resume() -> None:
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="cleanup remains incomplete",
        status="blocking",
        source_status="blocking",
        resolution_owners=("Codex", "Claude"),
        owner_states=(("Codex", "cleared"), ("Claude", "pending")),
        owner_dispositions=(("Codex", "future"),),
    )
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Codex", round_number=2, subject="head",
        prior_items=(item,),
    )

    decoded = _decode_round_metadata(_encode_round_metadata(metadata))

    assert decoded.prior_items == (item,)
    assert decoded.prior_items[0].owner_dispositions == (("Codex", "future"),)


def test_machine_obligation_and_qualification_checkpoint_round_trip() -> None:
    item = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="Managed exact-head CI failed.",
        status="blocking",
        source_status="blocking",
        authority=MACHINE_AUTHORITY,
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualifying",
        failed_head_sha="oldhead123",
        candidate_head_sha="newhead123",
        obligation_identity="managed-exact-head-ci:item-30",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha=item.failed_head_sha,
        candidate_head_sha=item.candidate_head_sha,
        base_branch="main",
        approval_digest="approval",
        plan_digest="plan",
        requirements_digest="requirements",
        acquisition_digest="acquisition",
        scheduler_digest="scheduler",
        qualification_attempt_id="123/1",
        watch_failure_extension_used=True,
        watch_head_extension_used=False,
        allowed_rounds=3,
    )
    metadata = PostedRoundMetadata(
        flow="pr",
        role="summary",
        agent="Orchestrator",
        round_number=14,
        subject="newhead123",
        prior_items=(item,),
        qualification_checkpoint=checkpoint,
    )

    encoded = _encode_round_metadata(metadata)
    decoded = _decode_round_metadata(encoded)

    assert decoded.prior_items == (item,)
    assert decoded.qualification_checkpoint == checkpoint
    assert transport.decode_mapping(encoded)["prior_items"][0]["authority"] == MACHINE_AUTHORITY


@pytest.mark.parametrize(
    "machine_fields",
    [
        {"authority": "bogus"},
        {"lifecycle": "repair_required"},
        {"failed_head_sha": "oldhead123"},
        {"candidate_head_sha": "newhead123"},
        {"obligation_identity": "machine:item-30"},
        {"authority": None, "obligation_kind": None, "lifecycle": None},
    ],
)
def test_partial_or_invalid_machine_item_decodes_as_non_bypassable_unknown(
    machine_fields,
) -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "prior_items": [
            {
                "item_id": "item-30",
                "reviewer": "GitHub managed exact-head CI",
                "source_round": 13,
                "text": "Persisted machine obligation.",
                "status": "blocking",
                **machine_fields,
            }
        ],
    }

    item = _decode_round_metadata_mapping(payload).prior_items[0]

    assert item.is_machine_obligation
    assert item.authority == UNKNOWN_MACHINE_AUTHORITY
    assert item.obligation_kind == "unknown"
    assert item.lifecycle == "repair_required"
    retained, _ = _apply_unresolved_item_dispositions(
        [item],
        {item.item_id: [ReviewItemDisposition(item.item_id, "Codex", "resolved")]},
        reconciliation_mode="owner-scoped",
    )
    assert retained == [item]


def test_invalid_qualification_checkpoint_decodes_fail_closed() -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "qualification_checkpoint": {
            "obligation_kind": "managed-exact-head-ci",
            "obligation_identity": "managed-exact-head-ci:item-30",
            "lifecycle": "qualifying",
            "failed_head_sha": "samehead",
            "candidate_head_sha": "samehead",
            "base_branch": "main",
            "approval_digest": None,
            "plan_digest": None,
            "requirements_digest": None,
            "acquisition_digest": None,
            "scheduler_digest": None,
            "qualification_attempt_id": None,
            "watch_failure_extension_used": False,
            "watch_head_extension_used": False,
            "allowed_rounds": 1,
        },
    }

    decoded = _decode_round_metadata_mapping(payload)

    assert decoded.qualification_checkpoint is not None
    assert not decoded.qualification_checkpoint.valid
    assert decoded.qualification_checkpoint.obligation_kind == "unknown"


@pytest.mark.parametrize(
    "missing",
    [
        "approval_digest",
        "requirements_digest",
        "acquisition_digest",
        "scheduler_digest",
        "qualification_attempt_id",
    ],
)
def test_qualification_checkpoint_missing_resume_identity_decodes_fail_closed(missing):
    checkpoint = {
        "obligation_kind": "managed-exact-head-ci",
        "obligation_identity": "managed-exact-head-ci:item-30",
        "lifecycle": "qualifying",
        "failed_head_sha": "oldhead123",
        "candidate_head_sha": "newhead123",
        "base_branch": "main",
        "approval_digest": "approval",
        "plan_digest": None,
        "requirements_digest": "requirements",
        "acquisition_digest": "acquisition",
        "scheduler_digest": "scheduler",
        "qualification_attempt_id": "run/1",
        "watch_failure_extension_used": False,
        "watch_head_extension_used": False,
        "allowed_rounds": 1,
    }
    checkpoint[missing] = None

    decoded = QualificationCheckpoint.from_mapping(checkpoint)

    assert not decoded.valid
    assert decoded.obligation_kind == "unknown"


@pytest.mark.parametrize(
    ("kind", "lifecycle", "candidate", "failed"),
    [
        ("unknown", "qualifying", "newhead123", "oldhead123"),
        ("managed-exact-head-ci", "cleared", "newhead123", "oldhead123"),
        ("managed-exact-head-ci", "qualifying", None, "oldhead123"),
    ],
)
def test_contradictory_qualification_checkpoint_payloads_decode_invalid(
    kind, lifecycle, candidate, failed
) -> None:
    payload = {
        "flow": "pr",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 14,
        "subject": "newhead123",
        "qualification_checkpoint": {
            "obligation_kind": kind,
            "obligation_identity": "machine:item-30",
            "lifecycle": lifecycle,
            "failed_head_sha": failed,
            "candidate_head_sha": candidate,
            "base_branch": "main",
            "approval_digest": None,
            "plan_digest": None,
            "requirements_digest": None,
            "acquisition_digest": None,
            "scheduler_digest": None,
            "qualification_attempt_id": None,
            "watch_failure_extension_used": False,
            "watch_head_extension_used": False,
            "allowed_rounds": 1,
        },
    }

    decoded = _decode_round_metadata_mapping(payload)

    assert decoded.qualification_checkpoint is not None
    assert not decoded.qualification_checkpoint.valid


def test_legacy_machine_item_requires_orchestrator_lineage_for_promotion() -> None:
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="CI failed at head `oldhead123`.",
        status="blocking",
        source_status="blocking",
    )
    trusted = _attach_round_metadata(
        "machine failure",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=13,
            subject="oldhead123",
            new_items=(legacy,),
        ),
    )
    ambiguous = _attach_round_metadata(
        "reviewer prose",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=13,
            subject="oldhead123",
            new_items=(legacy,),
        ),
    )

    trusted_item = _extract_round_metadata_records(
        [SimpleNamespace(body=trusted)], flow="pr"
    )[0].metadata.new_items[0]
    ambiguous_item = _extract_round_metadata_records(
        [SimpleNamespace(body=ambiguous)], flow="pr"
    )[0].metadata.new_items[0]

    assert trusted_item.authority == MACHINE_AUTHORITY
    assert trusted_item.obligation_kind == "managed-exact-head-ci"
    assert trusted_item.failed_head_sha == "oldhead123"
    assert ambiguous_item.authority == UNKNOWN_MACHINE_AUTHORITY
    assert ambiguous_item.obligation_kind == "unknown"


def test_legacy_coder_checkpoint_recovers_failed_head_from_scheduler_provenance() -> None:
    legacy = UnresolvedReviewItem(
        item_id="item-30",
        reviewer="GitHub managed exact-head CI",
        source_round=13,
        text="The managed check failed; repair the code.",
        status="blocking",
        source_status="blocking",
    )
    coder_checkpoint = _attach_round_metadata(
        "Coder repaired the PR.",
        PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=14,
            subject="newhead123",
            prior_items=(legacy,),
            scheduler_contract=ReviewSchedulingContract(
                required_reviewers=("Claude",),
                policy="selective-intermediate",
                broad_rules=("src/**",),
            ).as_dict(),
            scheduler_previous_sha="oldhead123",
            scheduler_current_sha="newhead123",
            scheduler_obligation_digest="0123456789abcdef",
            scheduler_selected_reviewers=("Claude",),
            scheduler_reasons=("full board",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )

    record = _extract_round_metadata_records(
        [SimpleNamespace(body=coder_checkpoint)], flow="pr"
    )[0].metadata.prior_items[0]

    assert record.authority == MACHINE_AUTHORITY
    assert record.obligation_kind == "managed-exact-head-ci"
    assert record.failed_head_sha == "oldhead123"
