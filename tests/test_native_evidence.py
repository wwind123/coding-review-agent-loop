"""Native request matching, aliases, protocol compatibility and boundary replay."""
from dataclasses import replace
from types import SimpleNamespace
import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.native_evidence import (consolidate_ledger, known_pr_heads,
    request_key, provisional_clearances)
from coding_review_agent_loop.protocol import ReviewItemDisposition, parse_pr_review
from coding_review_agent_loop.round_state import (PostedRoundMetadata, PostedRoundRecord,
    _append_recorded_evidence_requests, _encode_round_metadata, _decode_round_metadata)
from coding_review_agent_loop.unresolved_items import (_upsert_evidence_obligation,
    _apply_unresolved_item_dispositions, freeze_evidence_obligations)
from pr_loop_test_helpers import _evidence_review, _resolve

H1 = "1234567" + "a" * 33
H2 = "abcdef0" + "b" * 33


def upsert(items=(), text="Accept screenshot " + H1, *, head=H1, owner="Codex", number=1, heads=(H1, H2), clearances=()):
    return _upsert_evidence_obligation(items, item_number=number, reviewer=owner, text=text,
        source_round=1, current_head_sha=head, known_heads=heads, clearances=clearances)


def record(index, head, requests=(), **kwargs):
    return PostedRoundRecord(index=index, body="Review", metadata=PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=index + 1, subject=head,
        evidence_requests=tuple(requests), **kwargs))


@pytest.mark.parametrize("label", ["", "head "])
@pytest.mark.parametrize("width", [7, 12, 40])
def test_head_only_repetition_and_legacy_coalescing(label, width):
    first = f"Accept screenshot {label}{H1[:width]}"
    second = f"Accept screenshot {label}{H2[:width].upper()}"
    items, _ = upsert(text=first)
    identity = items[0].obligation_identity
    updated, consumed = upsert(items, second, head=H2, number=2)
    assert not consumed and len(updated) == 1
    assert updated[0].item_id == "item-1" and updated[0].obligation_identity == identity
    assert updated[0].text == second
    legacy, _ = upsert(text=second, head=H2, number=2)
    merged = consolidate_ledger([items[0], legacy[0]], (H1, H2))
    assert len(merged) == 1 and merged[0].alias_item_ids == ("item-2",)
    assert legacy[0].obligation_identity in merged[0].alias_obligation_identities


@pytest.mark.parametrize("suffix", [
    " manifest " + H1 + "a" * 24, " image " + "deadbee", " mode dark", " item-99"
])
def test_preserve_digests_and_semantic_text(suffix):
    first = "Accept screenshot " + H1[:7] + suffix
    second = "Accept screenshot " + H2[:7] + suffix + "b"
    items, _ = upsert(text=first)
    updated, consumed = upsert(items, second, head=H2, number=2)
    assert consumed and len(updated) == 2
    assert H1 + "a" * 24 in repr(request_key(first, (H1, H2))) if "manifest" in suffix else True


def test_unknown_ambiguous_and_other_owners_never_merge():
    heads = (H1, "1234567" + "b" * 33, H2)
    items, _ = upsert(text="Accept 1234567", heads=heads)
    updated, consumed = upsert(items, "Accept abcdef0", head=H2, heads=heads, number=2)
    assert consumed and len(updated) == 2
    items, _ = upsert(text="Accept deadbee")
    assert upsert(items, "Accept abcdef0", head=H2, number=2)[1]
    items, _ = upsert()
    assert upsert(items, "Accept screenshot " + H2, owner="Gemini", head=H2, number=2)[1]


def test_explicit_consolidation_alias_routing_and_conflicting_votes():
    first, _ = upsert(text="Screenshot A")
    items, _ = upsert(first, "Screenshot B", number=2)
    merged, consumed = upsert(items, "Consolidate item-1 and item-2 into this existing human-only obligation.", number=3)
    assert not consumed and len(merged) == 1 and merged[0].item_id == "item-1"
    assert merged[0].alias_item_ids == ("item-2",)
    frozen = freeze_evidence_obligations(merged, head_sha=H1)
    clearances = []
    kept, _ = _apply_unresolved_item_dispositions(frozen, {
        "item-2": [ReviewItemDisposition("item-2", "Codex", "resolved")],
        "item-1": [ReviewItemDisposition("item-1", "Codex", "blocking")],
    }, evidence_response_head=H1, configured_reviewers=("Codex",), evidence_clearances=clearances)
    assert kept and not clearances
    kept, _ = _apply_unresolved_item_dispositions(frozen, {
        "item-2": [ReviewItemDisposition("item-2", "Codex", "resolved")],
    }, evidence_response_head=H1, configured_reviewers=("Codex",), evidence_clearances=clearances)
    assert not kept and len(clearances) >= 2
    for text in ("Screenshot A", "Screenshot B", merged[0].text):
        assert upsert((), text, clearances=clearances)[0] == []
        assert upsert((), text, head=H2, clearances=clearances)[1]


@pytest.mark.parametrize("clause", [
    "Do not consolidate item-1 and item-2 into this obligation.",
    "Consolidate item-1 and item-99 into this obligation.",
    "Consolidate item-1 and item-2 and item-2 into this obligation.",
])
def test_invalid_consolidation_never_retires_targets(clause):
    items, _ = upsert(text="A")
    items, _ = upsert(items, "B", number=2)
    updated, consumed = upsert(items, clause, number=3)
    assert consumed and [i.item_id for i in updated] == ["item-1", "item-2", "item-3"]


@pytest.mark.parametrize("target", ["foreign", "converted"])
def test_foreign_and_converted_targets_are_not_consolidated(target):
    items, _ = upsert(text="A")
    items, _ = upsert(items, "B", number=2, owner="Gemini" if target == "foreign" else "Codex")
    if target == "converted":
        items[1] = replace(items[1], converted_row_id="row", converted_matrix_identity="m")
    updated, consumed = upsert(items, "Consolidate item-1 and item-2 into this obligation.", number=3)
    assert consumed and len(updated) == 3


def test_live_replay_equivalence_is_boundary_scoped():
    a = record(0, H1, ["Accept screenshot " + H1[:7]])
    b = record(1, H2, ["Accept screenshot " + H2[:7]])
    # A later ambiguous head must not change earlier matching retroactively.
    future = record(2, H1[:7] + "b" * 33)
    records = [a, b, future]
    live = []
    replay = []
    for r in (a, b):
        live, _ = upsert(live, r.metadata.evidence_requests[0], head=r.metadata.subject,
            number=2 if live else 1, heads=known_pr_heads([v for v in records if v.index <= r.index], r.metadata.subject))
        _append_recorded_evidence_requests(replay, [r], records=records)
        assert replay == live
    assert len(replay) == 1
    consolidated = record(3, H2, ["Consolidate item-1 into this existing acceptance obligation."])
    _append_recorded_evidence_requests(replay, [consolidated], records=[*records, consolidated])
    assert len(replay) == 1 and replay[0].item_id == "item-1"


def test_optional_fields_roundtrip_and_legacy_defaults():
    items, _ = upsert(text="A")
    items, _ = upsert(items, "B", number=2)
    items, _ = upsert(items, "Consolidate item-1 and item-2 into this obligation.", number=3)
    vote = {**_resolve(), "evidence_requirement_ids": ["signed-id"]}
    parsed = parse_pr_review(_evidence_review(dispositions=[vote]), reviewer="Codex")
    assert parsed.dispositions[0].evidence_requirement_ids == ("signed-id",)
    metadata = record(0, H1, prior_items=tuple(items), dispositions=parsed.dispositions).metadata
    assert _decode_round_metadata(_encode_round_metadata(metadata)).prior_items == tuple(items)
    assert _decode_round_metadata(_encode_round_metadata(metadata)).dispositions == parsed.dispositions
    legacy = parse_pr_review(_evidence_review(dispositions=[_resolve()]), reviewer="Codex")
    assert legacy.dispositions[0].evidence_requirement_ids == ()
    assert upsert()[0][0].alias_item_ids == ()


@pytest.mark.parametrize("value", ["id", ["id", "id"], [1], [""]])
def test_citation_shape_is_strict(value):
    with pytest.raises(AgentLoopError, match="evidence_requirement_ids"):
        parse_pr_review(_evidence_review(dispositions=[{**_resolve(), "evidence_requirement_ids": value}]), reviewer="Codex")


def test_removed_owner_requires_all_current_reviewers_with_exposure():
    items, _ = upsert()
    req = SimpleNamespace(requirement_id="r", body="Accepted " + H1[:7])
    vote = ReviewItemDisposition("item-1", "Gemini", "resolved", evidence_requirement_ids=("r",))
    m = record(0, H1, dispositions=(vote,), surfaced_reviewer_requirement_ids=("r",)).metadata
    one = PostedRoundRecord(index=0, body="", metadata=replace(m, agent="Gemini"))
    kept, clear = provisional_clearances(items, [one], [req], head=H1, round_number=1, board=("Gemini", "Claude"))
    assert kept and not clear
    second = PostedRoundRecord(index=1, body="", metadata=replace(m, agent="Claude", dispositions=(replace(vote, reviewer="Claude"),)))
    kept, clear = provisional_clearances(items, [one, second], [req], head=H1, round_number=1, board=("Gemini", "Claude"))
    assert not kept and clear
    # Another reviewer's surfaced set never grants exposure to this requester.
    kept, clear = provisional_clearances(items, [replace(one, metadata=replace(one.metadata, surfaced_reviewer_requirement_ids=())), second],
        [req], head=H1, round_number=1, board=("Gemini", "Claude"))
    assert kept and not clear


def test_literal_placeholder_and_foreign_subjects_are_not_commit_references():
    items, _ = upsert()
    updated, consumed = upsert(items, "Accept screenshot <commit>", number=2)
    assert consumed and len(updated) == 2
    trusted = record(0, H1, ["Accept screenshot " + H1[:7]])
    foreign = record(1, H2, ["Accept screenshot " + H2[:7]])
    replay = []
    _append_recorded_evidence_requests(replay, [trusted], records=[trusted, foreign], admitted_records=[trusted])
    _append_recorded_evidence_requests(replay, [foreign], records=[trusted, foreign], admitted_records=[trusted])
    assert len(replay) == 2
