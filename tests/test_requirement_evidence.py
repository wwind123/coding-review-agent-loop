"""Signed requirement routing shape, ownership and replay boundaries."""
import dataclasses
from copy import deepcopy
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.protocol import UnresolvedReviewItem, parse_structured_pr_review
from coding_review_agent_loop.requirement_evidence import parse_associations, project, historical_projection, classify_review
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
    assert action.body == (
        "Signed source (verbatim):\nAccept the run. Also fix retries.\n\n"
        "Reviewer leftover instructions (classification only; the signed source is authoritative):\n"
        "- Codex: Fix retries.")
    assert SOURCE.body == "Accept the run. Also fix retries."
    assert action.url == SOURCE.url


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fault", ["unsurfaced", "foreign", "missing", "nonnative", "ambiguous", "unlinked", "conflict", "mixed-invalid"])
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
    elif fault == "conflict":
        reviews["Gemini"] = dataclasses.replace(parsed, requirement_evidence=(_association(SOURCE.requirement_id, "code"),))
    else:
        # Same mixed role, but Gemini answers Codex's item: role agreement never
        # overrides an invalid per-requester linkage.
        reviews["Gemini"] = dataclasses.replace(parsed, requirement_evidence=(
            {**entry, "leftover_instructions": "Repair retries."},))
    if replay:
        records = [SimpleNamespace(metadata=PostedRoundMetadata(flow="pr", role="reviewer", agent=name,
            round_number=index + 1, subject=("a" if index else "b") * 40, prior_items=tuple(items),
            requirement_evidence=review.requirement_evidence, evidence_requests=review.exact_head_evidence_requests))
            for index, (name, review) in enumerate(reviews.items())]
        assert historical_projection(requirements, records) == tuple(requirements)
    else:
        assert project(requirements, reviews, items) == tuple(requirements)


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
        assert projected[0].requirement_id == SOURCE.requirement_id
        assert SOURCE.body in projected[0].body
        assert "- Codex: Fix retries." in projected[0].body
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
    assert 'request (a string:' in calls[0]
    assert 'leftover_instructions (a string:' in calls[0]
    assert 'otherwise ""' in calls[0]
    assert 'answered_item_ids (a list:' in calls[0]
    assert 'otherwise []' in calls[0]


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
        with pytest.raises(HumanDecisionRequiredError, match="classification remained conflicting") as error:
            run()
        message = str(error.value)
        assert f"{SOURCE.requirement_id}: Codex: mixed, Gemini: code" in message
        assert "Post a new follow-up signed comment (do not edit the existing signed comment) that separates exact-head evidence answers from code requirements" in message
        assert "opens a fresh review round against the added source" in message
        assert "no coder was dispatched" in message
    assert len(calls) == 1
    assert "Agreement is decided by role" in calls[0]
    assert '"reviewer": "Gemini", "role": "code"' in calls[0]


def _two_mixed_reviewers(order=("Codex", "Gemini")):
    items, entry, _, parsed = setup_role()
    items = [*items, dataclasses.replace(items[0], item_id="item-2", reviewer="Gemini")]
    gemini = dataclasses.replace(parsed, requirement_evidence=(
        {**entry, "answered_item_ids": ["item-2"], "leftover_instructions": "Repair retries."},))
    reviews = {"Codex": parsed, "Gemini": gemini}
    return items, {name: reviews[name] for name in order}


def _records(reviews, items):
    return [SimpleNamespace(metadata=PostedRoundMetadata(flow="pr", role="reviewer", agent=name,
        round_number=1, subject="a" * 40, prior_items=tuple(items),
        requirement_evidence=review.requirement_evidence, evidence_requests=review.exact_head_evidence_requests))
        for name, review in reviews.items()]


EXPECTED_MERGED = (
    "Signed source (verbatim):\nAccept the run. Also fix retries.\n\n"
    "Reviewer leftover instructions (classification only; the signed source is authoritative):\n"
    "- Codex: Fix retries.\n- Gemini: Repair retries.")


@pytest.mark.parametrize("order", [("Codex", "Gemini"), ("Gemini", "Codex")])
@pytest.mark.parametrize("replay", [False, True])
def test_mixed_wording_difference_merges_attributed(replay, order):
    items, reviews = _two_mixed_reviewers(order)
    live = project([SOURCE], reviews, items)
    result = historical_projection([SOURCE], _records(reviews, items)) if replay else live
    (action,) = result
    assert action.requirement_id == SOURCE.requirement_id
    assert action.url == SOURCE.url
    assert action.body == EXPECTED_MERGED
    # Replay parity: identical tuples, independent of posting order.
    assert historical_projection([SOURCE], _records(reviews, items)) == live
    assert project([SOURCE], dict(reversed(list(reviews.items()))), items) == live


@pytest.mark.parametrize("replay", [False, True])
def test_role_conflict_projection_retains_source(replay):
    items, reviews = _two_mixed_reviewers()
    reviews["Gemini"] = dataclasses.replace(reviews["Gemini"], requirement_evidence=(
        _association(SOURCE.requirement_id, "code"),))
    result = historical_projection([SOURCE], _records(reviews, items)) if replay else project([SOURCE], reviews, items)
    assert result == (SOURCE,)
    assert type(result[0]) is HumanReviewRequirement


def test_classify_review_wording_difference_is_not_conflict():
    items, reviews = _two_mixed_reviewers()
    text = _classified_review(list(reviews["Codex"].requirement_evidence))
    def invoke(prompt):
        raise AssertionError("wording difference must not trigger the classification re-ask")
    result, result_text = classify_review(reviews["Codex"], text, requirements=[SOURCE], items=items,
        reviewer_name="Codex", invoke=invoke, previous={"Gemini": reviews["Gemini"]})
    assert result is reviews["Codex"]
    assert result_text == text


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


@pytest.mark.parametrize("request_value", ["", [], None])
@pytest.mark.parametrize("leftover", ["", [], None])
def test_empty_text_live_review(request_value, leftover):
    answers = ["item-8", "item-9", "item-11", "item-12", "item-13", "item-14"]
    entry = _association(SOURCE.requirement_id, "evidence-only", answers=answers)
    canonical = deepcopy(entry)
    entry.update(request=request_value, leftover_instructions=leftover)
    original = deepcopy(entry)
    assert parse_associations([entry]) == (canonical,)
    parsed = parse_structured_pr_review(_classified_review([entry]), reviewer="Codex",
                                        architecture_status_mode="legacy")
    assert parsed.state == "approved"
    assert parsed.requirement_evidence == (canonical,)
    assert entry == original


@pytest.mark.parametrize("field", ["request", "leftover_instructions", "requirement_id"])
@pytest.mark.parametrize("value", [["text"], 7, True, {}, ()])
def test_text_type_errors_name_index_and_field(field, value):
    first = _association("first", "code")
    entry = _association("second", "code")
    entry[field] = value
    with pytest.raises(AgentLoopError) as error:
        parse_associations([first, entry])
    assert str(error.value) == (
        f"requirement_evidence[1].{field} must be a string (got {type(value).__name__})")


@pytest.mark.parametrize("value", [[], None])
def test_identity_is_not_normalized(value):
    entry = _association(value, "code")
    with pytest.raises(AgentLoopError, match=r"requirement_evidence\[0\].requirement_id must be a string"):
        parse_associations([entry])


@pytest.mark.parametrize("value", ["", "  ", [], None])
@pytest.mark.parametrize("role,field,message,live_message", [
    ("human-only", "request", "human-only must name its exact emitted evidence request",
     # A live review that emitted no request is offered only the valid choices (#2101).
     "this role needs a request linked to an emitted evidence request; no exact_head_evidence_requests emitted"),
    ("mixed", "leftover_instructions", "mixed requires explicit leftover_instructions",
     "mixed requires explicit leftover_instructions")])
def test_normalized_empty_text_retains_required_content(role, field, message, live_message, value):
    entry = _association("source", role, answers=["item-1"] if role == "mixed" else [])
    entry[field] = value
    with pytest.raises(AgentLoopError, match=message):
        parse_associations([entry])
    with pytest.raises(AgentLoopError, match=live_message):
        parse_structured_pr_review(_classified_review([entry]), reviewer="Codex",
                                   architecture_status_mode="legacy")


@pytest.mark.parametrize("request_value", ["", [], None])
@pytest.mark.parametrize("leftover", ["", [], None])
def test_empty_text_metadata_replay(request_value, leftover):
    items, entry, _, _ = setup_role("evidence-only")
    canonical = deepcopy(entry)
    entry.update(request=request_value, leftover_instructions=leftover)
    original = deepcopy(entry)
    meta = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1,
        subject="a" * 40, prior_items=tuple(items), requirement_evidence=(entry,))
    decoded = _decode_round_metadata(_encode_round_metadata(meta))
    assert decoded.requirement_evidence == (canonical,)
    again = _decode_round_metadata(_encode_round_metadata(decoded))
    assert again.requirement_evidence == decoded.requirement_evidence
    assert historical_projection([SOURCE], [SimpleNamespace(metadata=again)]) == ()
    foreign = dataclasses.replace(again, prior_items=(dataclasses.replace(items[0], reviewer="Gemini"),))
    assert historical_projection([SOURCE], [SimpleNamespace(metadata=foreign)]) == (SOURCE,)
    assert entry == original


@pytest.mark.parametrize("role", ["code", "human-only", "evidence-only"])
def test_nonmixed_leftovers_live_review_equivalence(role):
    from coding_review_agent_loop.requirement_evidence import normalize_empty_text
    canonical = _association(SOURCE.requirement_id, role,
        request="Accept the run" if role == "human-only" else "",
        answers=["item-1"] if role == "evidence-only" else [])
    entry = {**canonical, "leftover_instructions": "Fix retries. Preserve ownership checks."}
    before = deepcopy(entry)
    assert normalize_empty_text(entry) == canonical
    assert normalize_empty_text(normalize_empty_text(entry)) == canonical
    assert parse_associations([entry]) == (canonical,)
    kwargs = dict(state="blocking", blocking=["Fix retries.", "Preserve ownership checks."],
                  evidence=["Accept the run"] if role == "human-only" else [])
    parse = lambda e: parse_structured_pr_review(_classified_review([e], **kwargs),
        reviewer="Codex", architecture_status_mode="legacy")
    parsed = parse(entry)
    assert parsed == parse(canonical)
    assert len(parsed.blocking_items) == 2
    assert entry == before


@pytest.mark.parametrize("role", ["code", "human-only", "evidence-only"])
def test_nonmixed_leftovers_metadata_replay(role):
    items, _, _, _ = setup_role("evidence-only")
    canonical = _association(SOURCE.requirement_id, role,
        request="Accept the run" if role == "human-only" else "",
        answers=["item-1"] if role == "evidence-only" else [])
    entry = {**canonical, "leftover_instructions": "Redundant instructions"}
    meta = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1,
        subject="a" * 40, prior_items=tuple(items), requirement_evidence=(entry,),
        evidence_requests=("Accept the run",) if role == "human-only" else ())
    before = deepcopy(meta)
    encoded = _encode_round_metadata(meta)
    decoded = _decode_round_metadata(encoded)
    assert decoded.requirement_evidence == (canonical,)
    again = _decode_round_metadata(_encode_round_metadata(decoded))
    assert again == decoded
    replay = lambda m: historical_projection([SOURCE], [SimpleNamespace(metadata=m)])
    assert replay(again) == ((SOURCE,) if role == "code" else ())
    if role == "evidence-only":
        foreign = dataclasses.replace(again, prior_items=(dataclasses.replace(items[0], reviewer="Gemini"),))
        assert replay(foreign) == (SOURCE,)
    elif role == "human-only":
        assert replay(dataclasses.replace(again, evidence_requests=())) == (SOURCE,)
    assert meta == before
    assert _encode_round_metadata(meta) == encoded


def test_valid_mixed_leftovers_preserved_in_complete_review():
    _, entry, text, parsed = setup_role()
    before = deepcopy(entry)
    assert parse_associations([entry]) == (entry,)
    assert parsed.requirement_evidence == (entry,)
    assert entry == before


# --- Reviewer-owned routing helper (#2020) ---

from coding_review_agent_loop.requirement_evidence import reviewer_owned_ids  # noqa: E402

CODE_SOURCE = HumanReviewRequirement("PR comment", "maintainer", None, "https://example.test/2", "Fix the retry bound.")


@pytest.mark.parametrize("replay", [False, True])
def test_reviewer_owned_ids_evidence_only(replay):
    items, entry, text, parsed = setup_role("evidence-only")
    reviews = {"Codex": parsed}
    projected = historical_projection([SOURCE], _records(reviews, items)) if replay else project([SOURCE], reviews, items)
    assert projected == ()
    assert reviewer_owned_ids([SOURCE], projected) == (SOURCE.requirement_id.lower(),)


@pytest.mark.parametrize("replay", [False, True])
def test_reviewer_owned_ids_human_only(replay):
    request = "Operator must accept the policy at this exact head."
    review = SimpleNamespace(
        requirement_evidence=({"requirement_id": SOURCE.requirement_id, "role": "human-only",
            "request": request, "answered_item_ids": [], "leftover_instructions": ""},),
        exact_head_evidence_requests=(request,))
    reviews = {"Codex": review}
    projected = historical_projection([SOURCE], _records(reviews, [])) if replay else project([SOURCE], reviews, [])
    assert projected == ()
    assert reviewer_owned_ids([SOURCE], projected) == (SOURCE.requirement_id,)


@pytest.mark.parametrize("replay", [False, True])
def test_reviewer_owned_ids_mixed_and_code_stay_coder_owned(replay):
    items, entry, text, parsed = setup_role("mixed")
    parsed = dataclasses.replace(parsed, requirement_evidence=(
        entry, {"requirement_id": CODE_SOURCE.requirement_id, "role": "code", "request": "",
                "answered_item_ids": [], "leftover_instructions": ""}))
    reviews = {"Codex": parsed}
    sources = [SOURCE, CODE_SOURCE]
    projected = historical_projection(sources, _records(reviews, items)) if replay else project(sources, reviews, items)
    assert [r.requirement_id for r in projected] == [SOURCE.requirement_id, CODE_SOURCE.requirement_id]
    assert reviewer_owned_ids(sources, projected) == ()


@pytest.mark.parametrize("replay", [False, True])
@pytest.mark.parametrize("fault", ["conflict", "unclassified"])
def test_reviewer_owned_ids_conflict_or_unclassified_stay_coder_owned(replay, fault):
    items, entry, text, parsed = setup_role("evidence-only")
    if fault == "conflict":
        items = [*items, dataclasses.replace(items[0], item_id="item-2", reviewer="Gemini")]
        reviews = {"Codex": parsed, "Gemini": dataclasses.replace(parsed, requirement_evidence=(
            _association(SOURCE.requirement_id, "code"),))}
    else:
        reviews = {"Codex": dataclasses.replace(parsed, requirement_evidence=())}
    projected = historical_projection([SOURCE], _records(reviews, items)) if replay else project([SOURCE], reviews, items)
    assert projected == (SOURCE,)
    assert reviewer_owned_ids([SOURCE], projected) == ()


def test_reviewer_owned_ids_mixed_set_lists_only_routed_sources():
    items, entry, text, parsed = setup_role("evidence-only")
    parsed = dataclasses.replace(parsed, requirement_evidence=(
        entry, _association(CODE_SOURCE.requirement_id, "code")))
    projected = project([SOURCE, CODE_SOURCE], {"Codex": parsed}, items)
    assert projected == (CODE_SOURCE,)
    assert reviewer_owned_ids([SOURCE, CODE_SOURCE], projected) == (SOURCE.requirement_id,)
    assert reviewer_owned_ids([], ()) == ()


# --- #2035: role/field consistency is a reviewer judgement, shape stays fatal ---

from coding_review_agent_loop.errors import MissingJudgementFieldError
from coding_review_agent_loop.requirement_evidence import decode_associations


@pytest.mark.parametrize("role,request_text,answers,leftover,rule", [
    ("mixed", "", [], "Fix retries.",
     "mixed requires at least one answered_item_id; if no prior evidence item of yours is answered, use code or human-only"),
    ("evidence-only", "", [], "", "evidence-only requires at least one answered_item_id"),
    ("code", "", ["item-1"], "", "code uses answered_item_ids []"),
    ("human-only", "", [], "", "human-only must name its exact emitted evidence request"),
    ("code", "Run the suite", [], "", "only human-only names a request"),
    ("mixed", "", ["item-1"], "", "mixed requires explicit leftover_instructions"),
])
def test_consistency_rules_raise_typed_judgement_error(role, request_text, answers, leftover, rule):
    valid = _association("hr-first", "code")
    entry = _association("hr-second", role, request=request_text, answers=answers, leftover=leftover)
    with pytest.raises(MissingJudgementFieldError) as error:
        parse_associations([valid, entry])
    exc = error.value
    assert exc.field_path == "requirement_evidence"
    message = str(exc)
    assert "requirement_evidence[1]" in message
    assert "hr-second" in message
    assert rule in message
    assert exc.allowed_values and rule in exc.allowed_values[0]
    assert exc.observed_preview == message
    # Through the structured PR review parser the error keeps its type.
    with pytest.raises(MissingJudgementFieldError):
        parse_structured_pr_review(_classified_review([valid, entry]), reviewer="Codex",
                                   architecture_status_mode="legacy")


def test_consistency_rule_order_reports_first_violation():
    # human-only without request also has answers: the request rule still wins.
    entry = _association("hr-x", "human-only", answers=["item-1"])
    with pytest.raises(MissingJudgementFieldError, match="human-only must name"):
        parse_associations([entry])


@pytest.mark.parametrize("value", [
    "not-an-array",
    [{"requirement_id": "hr-x", "role": "code"}],
    [_association("hr-x", "unknown")],
    [{**_association("hr-x", "code"), "request": 3}],
    [{**_association("hr-x", "code"), "answered_item_ids": "item-1"}],
    [_association("hr-x", "evidence-only", answers=["item-1", "item-1"])],
    [_association("", "code")],
    [_association("hr-x", "code"), _association("hr-x", "code")],
])
def test_shape_errors_stay_plain_agent_loop_error(value):
    with pytest.raises(AgentLoopError) as error:
        parse_associations(value)
    assert not isinstance(error.value, MissingJudgementFieldError)
    with pytest.raises(ValueError):
        decode_associations({"requirement_evidence": value})


def test_decode_associations_maps_consistency_errors_to_value_error():
    with pytest.raises(ValueError, match="mixed requires at least one answered_item_id"):
        decode_associations({"requirement_evidence": [_association("hr-x", "mixed", leftover="x")]})


# --- #2101: superseded signed sources ---

from coding_review_agent_loop.requirement_evidence import (  # noqa: E402
    VALID_CHOICES_WITHOUT_REQUESTS,
    historical_superseded_ids,
    normalize_empty_text,
    reviewer_owned_ids,
    superseded_ids,
    validate_associations,
)


def _signed_source(body, minute, *, author="maintainer", author_id=7, created_at=None):
    stamp = created_at if created_at is not None else f"2026-11-22T05:{minute:02d}:00Z"
    return HumanReviewRequirement("PR comment", author, stamp, f"https://example.test/{minute}", body,
                                  author_id=author_id)


OLD_APPROVAL = _signed_source("Approved for head 13fd8b50.", 0)
LIMITING_NOTE = _signed_source("The approval above does not cover later heads.", 5)
NEW_APPROVAL = _signed_source("Approved for head 46ba82d9.", 12)
SUPERSEDED_SET = (OLD_APPROVAL, LIMITING_NOTE, NEW_APPROVAL)


def _carrier(entries, requests=()):
    return SimpleNamespace(requirement_evidence=tuple(entries), exact_head_evidence_requests=tuple(requests))


def _superseded_review(requests=()):
    return _carrier([_association(OLD_APPROVAL.requirement_id, "superseded"),
                     _association(LIMITING_NOTE.requirement_id, "superseded"),
                     _association(NEW_APPROVAL.requirement_id, "code")], requests)


@pytest.mark.parametrize("value", ["", None, [], "   "])
def test_superseded_parses_with_empty_fields(value):
    entry = _association("hr-old", "superseded")
    entry["request"] = value if value != "   " else ""
    entry["leftover_instructions"] = value
    (parsed,) = parse_associations([entry])
    assert parsed["role"] == "superseded"
    assert parsed["request"] == "" and parsed["leftover_instructions"] == "" and parsed["answered_item_ids"] == []


@pytest.mark.parametrize("field,value,rule", [
    ("request", "Run the suite", "code, evidence-only, mixed and superseded use request \"\""),
    ("answered_item_ids", ["item-1"], "superseded uses answered_item_ids []; a replaced source answers nothing"),
    ("leftover_instructions", "Also fix retries.",
     'superseded uses leftover_instructions ""; if leftover code work remains, use mixed or code'),
])
def test_superseded_shape_violation_takes_judgement_reask(field, value, rule):
    entry = _association("hr-old", "superseded")
    entry[field] = value
    with pytest.raises(MissingJudgementFieldError) as error:
        parse_associations([entry])
    assert rule in str(error.value)
    assert error.value.field_path == "requirement_evidence"
    assert rule in error.value.allowed_values[0]
    with pytest.raises(ValueError, match="superseded"):
        decode_associations({"requirement_evidence": [entry]})


def test_superseded_leftover_is_never_normalized_away():
    entry = _association("hr-old", "superseded", leftover="Also fix retries.")
    assert normalize_empty_text(entry)["leftover_instructions"] == "Also fix retries."
    # Existing non-mixed normalization (#1954) is unchanged.
    for role in ("code", "human-only", "evidence-only"):
        assert normalize_empty_text({**entry, "role": role})["leftover_instructions"] == ""


def test_superseded_validates_with_newer_same_author_source():
    validate_associations(_superseded_review(), SUPERSEDED_SET, (), "Codex")
    # Login match when either side lacks an author_id.
    by_login = [dataclasses.replace(s, author_id=None, author=s.author.upper()) for s in SUPERSEDED_SET]
    validate_associations(_superseded_review(), by_login, (), "Codex")


@pytest.mark.parametrize("fault", ["newest", "other-author-id", "other-login", "equal-time", "missing-time",
                                   "missing-newer-time", "missing-author", "alone"])
def test_superseded_precondition_fails_closed(fault):
    old, newer = OLD_APPROVAL, NEW_APPROVAL
    if fault == "newest":
        old, newer = NEW_APPROVAL, OLD_APPROVAL
    elif fault == "other-author-id":
        newer = dataclasses.replace(newer, author="maintainer", author_id=8)
    elif fault == "other-login":
        old, newer = (dataclasses.replace(old, author_id=None),
                      dataclasses.replace(newer, author="someone-else", author_id=None))
    elif fault == "equal-time":
        newer = dataclasses.replace(newer, created_at=old.created_at)
    elif fault == "missing-time":
        old = dataclasses.replace(old, created_at=None)
    elif fault == "missing-newer-time":
        newer = dataclasses.replace(newer, created_at="not a time")
    elif fault == "missing-author":
        old, newer = (dataclasses.replace(old, author=None, author_id=None),
                      dataclasses.replace(newer, author=None, author_id=None))
    requirements = [old] if fault == "alone" else [old, newer]
    review = _carrier([_association(old.requirement_id, "superseded")])
    with pytest.raises(MissingJudgementFieldError, match="superseded requires another surfaced signed source"):
        validate_associations(review, requirements, (), "Codex")
    # Project and replay retain the full current source.
    assert project(requirements, {"Codex": review}, ()) == tuple(requirements)
    assert superseded_ids(requirements, {"Codex": review}, ()) == ()


def test_superseded_project_drops_source_and_lists_it_reviewer_owned():
    items, entry, _, parsed = setup_role("evidence-only")
    evidence_source = SOURCE
    requirements = (*SUPERSEDED_SET, evidence_source)
    review = _carrier([*_superseded_review().requirement_evidence, entry])
    projected = project(requirements, {"Codex": review}, items)
    assert projected == (NEW_APPROVAL,)
    owned = reviewer_owned_ids(requirements, projected)
    assert owned == (OLD_APPROVAL.requirement_id, LIMITING_NOTE.requirement_id, evidence_source.requirement_id)
    superseded = superseded_ids(requirements, {"Codex": review}, items)
    assert superseded == (OLD_APPROVAL.requirement_id, LIMITING_NOTE.requirement_id)
    assert set(superseded) <= set(owned)


def test_two_reviewers_agree_on_superseded_without_conflict():
    """`superseded-two-reviewer-agreement`: agreement is by role (#2101)."""
    gemini = _superseded_review()
    codex_text = _classified_review(list(_superseded_review().requirement_evidence), hr=True)
    codex = parse_structured_pr_review(codex_text, reviewer="Codex", architecture_status_mode="legacy")
    def invoke(prompt):
        raise AssertionError("agreeing superseded roles must not trigger a continuation")
    result, _ = classify_review(codex, codex_text, requirements=list(SUPERSEDED_SET), items=(),
                                reviewer_name="Codex", invoke=invoke, previous={"Gemini": gemini})
    assert result is codex
    assert project(SUPERSEDED_SET, {"Gemini": gemini, "Codex": result}, ()) == (NEW_APPROVAL,)
    assert superseded_ids(SUPERSEDED_SET, {"Gemini": gemini, "Codex": result}, ()) == (
        OLD_APPROVAL.requirement_id, LIMITING_NOTE.requirement_id)


def test_superseded_versus_code_still_conflicts():
    """`superseded-vs-code-conflict`: a role split stops and retains the source."""
    gemini = _superseded_review()
    code_entries = [_association(s.requirement_id, "code") for s in SUPERSEDED_SET]
    codex_text = _classified_review(code_entries, hr=True)
    codex = parse_structured_pr_review(codex_text, reviewer="Codex", architecture_status_mode="legacy")
    calls = []
    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(marker_value=codex, text=codex_text)
    with pytest.raises(HumanDecisionRequiredError, match="classification remained conflicting") as error:
        classify_review(codex, codex_text, requirements=list(SUPERSEDED_SET), items=(),
                        reviewer_name="Codex", invoke=invoke, previous={"Gemini": gemini})
    assert len(calls) == 1
    assert f"{OLD_APPROVAL.requirement_id}: Codex: code, Gemini: superseded" in str(error.value)
    assert project(SUPERSEDED_SET, {"Gemini": gemini, "Codex": codex}, ()) == SUPERSEDED_SET
    assert superseded_ids(SUPERSEDED_SET, {"Gemini": gemini, "Codex": codex}, ()) == ()


def test_classification_prompt_explains_superseded_routing():
    codex_text = _classified_review([], hr=False)
    codex = parse_structured_pr_review(codex_text, reviewer="Codex", architecture_status_mode="legacy")
    calls = []
    def invoke(prompt):
        calls.append(prompt)
        return SimpleNamespace(marker_value=dataclasses.replace(
            codex, requirement_evidence=_superseded_review().requirement_evidence), text=codex_text)
    classify_review(codex, codex_text, requirements=list(SUPERSEDED_SET), items=(), reviewer_name="Codex",
                    invoke=invoke)
    (prompt,) = calls
    assert "role (code, human-only, evidence-only, mixed, superseded)" in prompt
    assert 'Use superseded (request "", answered_item_ids [], leftover_instructions "")' in prompt
    assert "later signed source from the same author replaces" in prompt
    assert "Use human-only only when you emit its linked exact_head_evidence_request" in prompt
    assert '"created_at": "2026-11-22T05:00:00Z"' in prompt and '"author": "maintainer"' in prompt


def _rendered_reask(error):
    """The judgement re-ask appendix the reviewer actually receives."""
    from coding_review_agent_loop.architecture_contract import _missing_judgement_field_reask_prompt
    exc = error.value
    return _missing_judgement_field_reask_prompt("PROMPT", exc.field_path, exc.allowed_values,
                                                 exc.observed_preview)


_PARSE_BRANCHES = [
    ("evidence-only", "", [], ""),
    ("mixed", "", [], "Fix retries."),
    ("mixed", "", ["item-1"], ""),
    ("human-only", "", [], ""),
    ("human-only", "", ["item-1"], ""),
    ("code", "Run the suite", [], ""),
    ("code", "", ["item-1"], ""),
    ("superseded", "Run the suite", [], ""),
    ("superseded", "", ["item-1"], ""),
    ("superseded", "", [], "Also fix retries."),
]


def _validate_branch(branch, requests):
    """Each validate_associations rejection: unsurfaced, unlinked, precondition, foreign answer."""
    items, _, _, _ = setup_role("evidence-only")
    if branch == "unsurfaced":
        entry, requirements = _association("hr-unknown", "code"), SUPERSEDED_SET
    elif branch == "unlinked-human-only":
        entry, requirements = _association(OLD_APPROVAL.requirement_id, "human-only", request="Unemitted"), SUPERSEDED_SET
    elif branch == "superseded-precondition":
        entry, requirements = _association(NEW_APPROVAL.requirement_id, "superseded"), SUPERSEDED_SET
    else:
        items = [dataclasses.replace(items[0], reviewer="Gemini")]
        entry, requirements = _association(SOURCE.requirement_id, "evidence-only", answers=["item-1"]), [SOURCE]
    with pytest.raises(MissingJudgementFieldError) as error:
        validate_associations(_carrier([entry], requests), requirements, items, "Codex")
    return error


@pytest.mark.parametrize("role,request_text,answers,leftover", _PARSE_BRANCHES)
def test_reask_without_emitted_requests_never_offers_human_only(role, request_text, answers, leftover):
    """`reask-no-emitted-requests`: every parse rejection names only choices that can validate (#2101)."""
    entry = _association("hr-x", role, request=request_text, answers=answers, leftover=leftover)
    with pytest.raises(MissingJudgementFieldError) as error:
        parse_associations([entry], emitted_requests=())
    assert error.value.allowed_values[1] == VALID_CHOICES_WITHOUT_REQUESTS
    assert all("human-only" not in value for value in error.value.allowed_values)
    rendered = _rendered_reask(error)
    assert VALID_CHOICES_WITHOUT_REQUESTS in rendered
    for choice in ("use superseded", "evidence-only with your own prior evidence item IDs", "or code"):
        assert choice in rendered
    # With emitted requests, or in replay (unknown), the existing wording is unchanged.
    for emitted in (None, ("Run the suite",)):
        with pytest.raises(MissingJudgementFieldError) as legacy:
            parse_associations([entry], emitted_requests=emitted)
        assert len(legacy.value.allowed_values) == 1
        assert VALID_CHOICES_WITHOUT_REQUESTS not in str(legacy.value)


@pytest.mark.parametrize("branch", ["unsurfaced", "unlinked-human-only", "superseded-precondition", "foreign-answer"])
def test_validation_reask_without_emitted_requests_names_valid_choices(branch):
    """`reask-no-emitted-requests`: every validation rejection names the choices intact (#2101)."""
    error = _validate_branch(branch, ())
    assert error.value.allowed_values[1] == VALID_CHOICES_WITHOUT_REQUESTS
    rendered = _rendered_reask(error)
    assert VALID_CHOICES_WITHOUT_REQUESTS in rendered
    emitted = _validate_branch(branch, ("Other request",))
    assert len(emitted.value.allowed_values) == 1
    assert VALID_CHOICES_WITHOUT_REQUESTS not in str(emitted.value)


def test_valid_choices_fit_the_rendered_reask_bound():
    from coding_review_agent_loop.architecture_contract import _JUDGEMENT_REASK_MAX_CHARS
    assert len(VALID_CHOICES_WITHOUT_REQUESTS) <= _JUDGEMENT_REASK_MAX_CHARS


def test_live_review_without_requests_reasks_with_valid_choices():
    entry = _association("hr-x", "evidence-only")
    with pytest.raises(MissingJudgementFieldError) as error:
        parse_structured_pr_review(_classified_review([entry]), reviewer="Codex", architecture_status_mode="legacy")
    assert "use code or human-only" not in str(error.value)
    assert VALID_CHOICES_WITHOUT_REQUESTS in _rendered_reask(error)


def test_unlinked_human_only_validation_names_valid_choices_without_requests():
    entry = _association(OLD_APPROVAL.requirement_id, "human-only", request="Unemitted request")
    with pytest.raises(MissingJudgementFieldError) as error:
        validate_associations(_carrier([entry]), SUPERSEDED_SET, (), "Codex")
    assert "Human-only association must name this requester's emitted evidence request" in str(error.value)
    assert VALID_CHOICES_WITHOUT_REQUESTS in str(error.value)
    with pytest.raises(MissingJudgementFieldError) as emitted:
        validate_associations(_carrier([entry], ["Other request"]), SUPERSEDED_SET, (), "Codex")
    assert VALID_CHOICES_WITHOUT_REQUESTS not in str(emitted.value)


def test_superseded_metadata_round_trip_and_historical_replay():
    """`superseded-resume-replay` helper coverage: decode, revalidate and replay (#2101)."""
    review = _superseded_review()
    meta = PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1, subject="a" * 40,
                               requirement_evidence=review.requirement_evidence)
    decoded = _decode_round_metadata(_encode_round_metadata(meta))
    assert decoded.requirement_evidence == review.requirement_evidence
    # The resumable-classification predicate's parse and validation accept it.
    validate_associations(_carrier(parse_associations(list(decoded.requirement_evidence)),
                                   decoded.evidence_requests), SUPERSEDED_SET, (), "Codex")
    records = [SimpleNamespace(metadata=decoded)]
    assert historical_projection(SUPERSEDED_SET, records) == project(SUPERSEDED_SET, {"Codex": review}, ())
    assert historical_projection(SUPERSEDED_SET, records) == (NEW_APPROVAL,)
    assert historical_superseded_ids(SUPERSEDED_SET, records) == (
        OLD_APPROVAL.requirement_id, LIMITING_NOTE.requirement_id)
    # Replay rechecks the precondition: without the newer source the old one is retained.
    assert historical_projection(SUPERSEDED_SET[:1], records) == SUPERSEDED_SET[:1]
    assert historical_superseded_ids(SUPERSEDED_SET[:1], records) == ()
