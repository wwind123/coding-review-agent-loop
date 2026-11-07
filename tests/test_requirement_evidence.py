"""Signed requirement routing shape, ownership, replay and repair boundaries."""
import dataclasses
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.protocol import UnresolvedReviewItem, parse_structured_pr_review
from coding_review_agent_loop.requirement_evidence import parse_associations, project, historical_projection, classify_review
from coding_review_agent_loop.repair_preservation import validate_repair_preservation
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    _encode_round_metadata,
    _decode_round_metadata,
)
from coding_review_agent_loop.unresolved_items import _upsert_evidence_obligation
from pr_loop_test_helpers import _association, _classified_review, _evidence_review

SOURCE = HumanReviewRequirement("PR comment", "maintainer", None, "https://example.test/1", "Accept the run. Also fix retries.")


def setup_role(role="mixed", reviewer="Codex"):
    items, _ = _upsert_evidence_obligation([], item_number=1, reviewer=reviewer, text="Accept the run", source_round=1, current_head_sha="a" * 40)
    entry = _association(SOURCE.requirement_id, role, answers=["item-1"], leftover="Fix retries." if role == "mixed" else "")
    text = _classified_review([entry])
    parsed = parse_structured_pr_review(text, reviewer="Codex", architecture_status_mode="legacy")
    return items, entry, text, parsed


def test_mixed_projection_preserves_original_identity_and_source():
    items, entry, text, parsed = setup_role()
    (action,) = project([SOURCE], {"Codex": parsed}, items)
    assert action.requirement_id == SOURCE.requirement_id
    assert action.body == "Fix retries."
    assert SOURCE.body == "Accept the run. Also fix retries."
    assert action.url == SOURCE.url


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fault", ["unsurfaced", "foreign", "missing", "nonnative", "ambiguous", "unlinked", "conflict", "wording"])
def test_associations_fail_closed(fault, replay):
    items, entry, text, parsed = setup_role()
    requirements = [SOURCE]
    reviews = {"Codex": parsed}
    if fault == "unsurfaced":
        requirements = []
    elif fault == "foreign":
        items = [dataclasses.replace(items[0], reviewer="Gemini")]
    elif fault == "missing":
        items = []
    elif fault == "nonnative":
        items = [UnresolvedReviewItem(item_id="item-1", reviewer="Codex", text="Code work", source_round=1, status="blocking")]
    elif fault == "ambiguous":
        items = [*items, dataclasses.replace(items[0], item_id="item-2", alias_item_ids=("item-1",))]
    elif fault == "unlinked":
        reviews["Codex"] = dataclasses.replace(parsed, requirement_evidence=(
            _association(SOURCE.requirement_id, "human-only", request="Unemitted request"),))
    else:
        other_entry = _association(SOURCE.requirement_id, "code") if fault == "conflict" else {**entry, "leftover_instructions": "Repair retries."}
        other = dataclasses.replace(parsed, requirement_evidence=(other_entry,))
        if fault == "wording":
            items = [*items, dataclasses.replace(items[0], item_id="item-2", reviewer="Gemini")]
            other = dataclasses.replace(other, requirement_evidence=({**other_entry, "answered_item_ids": ["item-2"]},))
        reviews["Gemini"] = other
    if replay:
        records = [SimpleNamespace(metadata=PostedRoundMetadata(flow="pr", role="reviewer", agent=name,
            round_number=index + 1, subject=("a" if index else "b") * 40, prior_items=tuple(items),
            requirement_evidence=review.requirement_evidence, evidence_requests=review.exact_head_evidence_requests))
            for index, (name, review) in enumerate(reviews.items())]
        assert historical_projection(requirements, records) == tuple(requirements)
    else:
        assert project(requirements, reviews, items) == tuple(requirements)


@pytest.mark.parametrize("change", ["add", "remove", "role", "leftover", "target"])
def test_format_repair_cannot_invent_or_change_classification(change):
    _, entry, text, _ = setup_role()
    modified = dict(entry)
    if change == "role":
        modified["role"] = "evidence-only"
        modified["leftover_instructions"] = ""
    elif change == "leftover":
        modified["leftover_instructions"] = "Invent another code instruction."
    elif change == "target":
        modified["answered_item_ids"] = ["item-2"]
    target = _evidence_review() if change == "remove" else _classified_review([modified])
    source = _evidence_review() if change == "add" else text
    with pytest.raises(AgentLoopError, match="classification"):
        validate_repair_preservation(source, target)
    validate_repair_preservation(text, text)


@pytest.mark.parametrize("role", ["mixed", "evidence-only"])
def test_round_trip_and_trusted_replay_keep_scope(role):
    items, entry, text, parsed = setup_role(role)
    meta = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1,
        subject="a" * 40, prior_items=tuple(items), requirement_evidence=(entry,))
    decoded = _decode_round_metadata(_encode_round_metadata(meta))
    assert decoded.requirement_evidence == (entry,)
    projected = historical_projection([SOURCE], [SimpleNamespace(metadata=decoded)])
    assert [r.requirement_id for r in projected] == ([SOURCE.requirement_id] if role == "mixed" else [])
    if projected:
        assert projected[0].body == "Fix retries."
    legacy = dataclasses.replace(meta, requirement_evidence=())
    assert historical_projection([SOURCE], [SimpleNamespace(metadata=legacy)]) == (SOURCE,)
    edited = dataclasses.replace(SOURCE, body="New instruction")
    assert historical_projection([edited], [SimpleNamespace(metadata=meta)]) == (edited,)


def test_missing_classification_is_bounded_semantic_turn():
    items, _, _, parsed = setup_role()
    missing = dataclasses.replace(parsed, requirement_evidence=())
    calls = []
    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(marker_value=missing, text=_evidence_review())
    with pytest.raises(HumanDecisionRequiredError, match="classification remained absent"):
        classify_review(missing, _evidence_review(), requirements=[SOURCE], items=items,
            reviewer_name="Codex", invoke=invoke)
    assert len(calls) == 1
    assert "real semantic reviewer turn" in calls[0]


@pytest.mark.parametrize("field,value", [("role", "unknown"), ("role", []), ("answered_item_ids", ["item-1", "item-1"]),
    ("leftover_instructions", ""), ("requirement_id", ""), ("request", "unlinked")])
def test_invalid_association_shape(field, value):
    _, entry, _, _ = setup_role()
    entry[field] = value
    with pytest.raises(AgentLoopError):
        parse_associations([entry])


@pytest.mark.parametrize("resolved", [False, True])
def test_conflicting_classification_gets_one_real_continuation(resolved):
    items, entry, text, parsed = setup_role()
    previous = {"Gemini": dataclasses.replace(parsed, requirement_evidence=(_association(SOURCE.requirement_id, "code"),))}
    calls = []
    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(marker_value=previous["Gemini"] if resolved else parsed, text=text)
    def run():
        return classify_review(parsed, text, requirements=[SOURCE], items=items,
            reviewer_name="Codex", invoke=invoke, previous=previous)
    if resolved:
        result, _ = run()
        assert project([SOURCE], {**previous, "Codex": result}, items) == (SOURCE,)
    else:
        with pytest.raises(HumanDecisionRequiredError, match="classification remained conflicting"):
            run()
    assert len(calls) == 1


@pytest.mark.parametrize("acknowledged", [False, True])
@pytest.mark.parametrize("classified", [False, True])
def test_ordinary_code_requirement_classification_matches_prompt(acknowledged, classified):
    from agent_loop_helpers import structured_pr_review
    from pr_loop_test_helpers import _with_code_classification
    text = structured_pr_review(human_requirements_resolved=acknowledged)
    code_text = _with_code_classification(text, [SOURCE])
    if classified:
        text = code_text
    parse = lambda value: parse_structured_pr_review(value, reviewer="Codex", architecture_status_mode="legacy")
    calls = []
    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(marker_value=parse(code_text), text=code_text)
    parsed, _ = classify_review(parse(text), text, requirements=[SOURCE], items=[],
                               reviewer_name="Codex", invoke=invoke)
    assert len(calls) == (0 if classified or acknowledged else 1)
    # Classification routes ordinary instructions to the coder without waiving them.
    assert project([SOURCE], {"Codex": parsed}, []) == (SOURCE,)
