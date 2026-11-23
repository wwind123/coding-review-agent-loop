"""Harmless extras in agent responses are dropped or aliased, not rejected (#2114).

Each section maps to one row of the approved risk matrix.  Normalization runs
on the parsed value only: the response text is never rewritten, so a resume
re-parses the same text to the same result.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest

from agent_loop_helpers import (
    structured_coder_followup,
    structured_issue_implementation,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
    structured_v1_plan_state,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.prompt_support import render_coder_human_requirements_prompt_context
from coding_review_agent_loop.protocol import (
    HumanRequirementDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    UnresolvedReviewItem,
    parse_agent_unavailable,
    parse_historical_structured_coder_followup,
    parse_structured_plan_review,
    parse_structured_pr_review,
    validate_human_requirement_dispositions,
    validate_structured_coder_followup,
    validate_structured_human_requirements_acknowledgement,
    validate_structured_issue_implementation,
    validate_structured_plan_revision,
    validate_structured_plan_state,
    validate_structured_task_result,
)
from coding_review_agent_loop.protocol_common import (
    NOTE_KEY_RENAMES,
    _expect_known_keys_lenient,
    is_protected_response_key,
    partition_item_id_buckets,
)
from coding_review_agent_loop.protocol_risk import (
    parse_risk_test_matrix,
    parse_risk_test_matrix_evidence,
    parse_risk_test_matrix_row,
    risk_test_matrix_identity,
    risk_test_matrix_prompt_examples,
)
from coding_review_agent_loop.pr_loop_support import (
    _coder_followup_review_context,
    _extract_structured_coder_summary,
    _extract_structured_coder_tests_run,
)
from coding_review_agent_loop.response_validation import _validate_issue_implementation_response
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, rebuild_resumed_coder_carrier
from coding_review_agent_loop.architecture_contract import _TerminalIssueImplementationConflict
from coding_review_agent_loop.unresolved_items import (
    _validate_coder_followup_response,
    _validate_review_response,
)

LOGGER = "coding_review_agent_loop"
EXTRAS = {"confidence": "high", "notes": "Looked at everything.", "pr_url": "https://github.com/o/r/pull/1"}
REQUIREMENT = HumanReviewRequirement(
    source_type="PR comment",
    author="reviewer",
    created_at="2026-05-18T10:00:00Z",
    url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
    body="Please use the absolute URL.",
)
SURFACED_ID = render_coder_human_requirements_prompt_context((REQUIREMENT,)).surfaced_requirement_ids[0]
UNSURFACED_ID = "hr-" + "e" * 64
HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"


def _item(item_id: str) -> UnresolvedReviewItem:
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="OpenAI Codex", source_round=1, text=f"Fix {item_id}.", status="blocking",
    )


ITEMS = (_item("item-1"), _item("item-2"))


def _with(text: str, **extra) -> str:
    """Add top-level keys to a structured response, keeping its footer."""
    payload, end = json.JSONDecoder().raw_decode(text)
    payload.update(extra)
    return json.dumps(payload) + text[end:]


def _edit(text: str, edit) -> str:
    payload, end = json.JSONDecoder().raw_decode(text)
    edit(payload)
    return json.dumps(payload) + text[end:]


def _dropped(caplog, key: str) -> bool:
    return any(f"dropped unknown field {key}" in record.getMessage() for record in caplog.records)


def _task_result(**extra) -> str:
    payload = {
        "schema_version": 1, "kind": "task_result", "state": "blocking",
        "outcome": "blocking", "summary": "Could not finish.", **extra,
    }
    return json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"


# --- extra-string-key ---------------------------------------------------------

_PARSERS = {
    "plan_state": (structured_plan_state, lambda text: validate_structured_plan_state(text, architecture_status_mode="legacy")),
    "plan_revision": (structured_plan_revision, lambda text: validate_structured_plan_revision(text, architecture_status_mode="legacy")),
    "plan_review": (structured_plan_review, lambda text: parse_structured_plan_review(text, reviewer="Codex")),
    "pr_review": (structured_pr_review, lambda text: parse_structured_pr_review(text, reviewer="Codex")),
    "coder_followup": (structured_coder_followup, validate_structured_coder_followup),
    "issue_implementation": (structured_issue_implementation, validate_structured_issue_implementation),
    "task_result": (_task_result, validate_structured_task_result),
}


@pytest.mark.parametrize("kind", sorted(_PARSERS))
def test_top_level_string_extras_are_dropped_with_a_warning(kind, caplog):
    build, parse = _PARSERS[kind]
    baseline = parse(build())
    text = _with(build(), **EXTRAS)
    original = str(text)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse(text)
    assert parsed == baseline
    for key in EXTRAS:
        assert _dropped(caplog, key)
    assert text == original  # the stored text is never rewritten


def test_requirement_disposition_note_and_stage_estimated_size_are_dropped(caplog):
    review = structured_plan_review(
        human_requirement_dispositions=[
            {"requirement_id": SURFACED_ID, "disposition": "addressed", "evidence": "Covered.", "note": "Extra."}
        ],
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_structured_plan_review(review, reviewer="Codex")
    assert parsed.human_requirement_dispositions == (
        HumanRequirementDisposition(SURFACED_ID, "addressed", "Covered."),
    )
    assert _dropped(caplog, "note")

    def stage(number, depends, extra):
        return {
            "stage_id": f"stage-{number}", "position": number, "title": f"Stage {number}",
            "summary": f"Stage {number}.", "deliverables": ["A."], "non_goals": [],
            "acceptance_criteria": ["Done."], "depends_on_stage_ids": depends,
            "dependency_notes": "None.", "automation": "agent-pr", "rollout_risk": "low",
            "compatibility_constraints": [], "covered_scope_item_ids": [f"scope-{number}"], **extra,
        }

    def add_stage(payload):
        recommendation = payload["execution_recommendation"]
        recommendation.update(
            strategy="staged", staging_feasibility="safe",
            scope_items=[
                {"scope_item_id": "scope-1", "requirement": "First.", "acceptance_criteria": ["Done."]},
                {"scope_item_id": "scope-2", "requirement": "Second.", "acceptance_criteria": ["Done."]},
            ],
            child_stages=[stage(1, [], {"estimated_size": "small"}), stage(2, ["stage-1"], {})],
        )
        recommendation.pop("one_shot_delivery")

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = validate_structured_plan_state(_edit(structured_v1_plan_state(), add_stage), architecture_status_mode="legacy")
    assert plan.execution_recommendation.child_stages[0].stage_id == "stage-1"
    assert _dropped(caplog, "estimated_size")


# --- protected-key-strict -----------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {"confidence": 0.8},
        {"notes": ["a"]},
        {"item_id": "item-1"},
        {"disposition": "resolved"},
        {"head_sha": HEAD},
        {"<!-- AGENT_STATE: approved -->": "x"},
        {"agent_signature": "-- Orchestrator"},
        {"managed_ci_dispatch": "x"},
        {"provenance": "x"},
    ],
    ids=lambda extra: next(iter(extra)),
)
@pytest.mark.parametrize("kind", ["pr_review", "coder_followup", "plan_state"])
def test_protected_or_non_string_extras_still_reject(kind, extra):
    build, parse = _PARSERS[kind]
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse(_with(build(), **extra))


def test_risk_test_matrix_evidence_without_the_historical_flag_rejects():
    text = _with(structured_coder_followup(), risk_test_matrix_evidence="forged")
    with pytest.raises(AgentLoopError, match="unknown field\\(s\\): risk_test_matrix_evidence"):
        validate_structured_coder_followup(text)


def test_protected_name_policy():
    for name in ("item_id", "id", "head_sha", "risk_test_matrix_evidence", "AGENT_PR", "x<!--",
                 "managed_ci_run", "reviewer_signature", "topology_source", 7):
        assert is_protected_response_key(name), name
    for name in ("confidence", "notes", "pr_url", "estimated_size", "reason"):
        assert not is_protected_response_key(name), name


# --- strict-sites-unchanged ---------------------------------------------------


def test_trust_boundary_sites_stay_strict():
    unavailable = json.dumps({
        "schema_version": 1, "kind": "agent_unavailable", "retryable": False, "category": "tooling",
        "summary": "Broken.", "suggested_action": "Fix it.", "notes": "extra",
    }) + "\n<!-- AGENT_UNAVAILABLE -->\n-- Anthropic Claude"
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_agent_unavailable(unavailable)
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_risk_test_matrix_evidence({"matrix_identity": "a" * 64, "rows": [], "notes": "extra"})
    row = copy.deepcopy(risk_test_matrix_prompt_examples()["applicable"]["rows"][0])
    row["notes"] = "extra"
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_risk_test_matrix_row(row)  # the semantic-patch matrix_add/matrix_edit row parser


# --- matrix-row-notes ---------------------------------------------------------


def test_complete_matrix_row_notes_are_dropped_and_identity_uses_parsed_rows(caplog):
    matrix = copy.deepcopy(risk_test_matrix_prompt_examples()["applicable"])
    baseline = parse_risk_test_matrix(matrix)
    noisy = copy.deepcopy(matrix)
    noisy["rows"][0]["notes"] = "Reviewer asked for this row."
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_risk_test_matrix(noisy)
    assert parsed == baseline
    assert risk_test_matrix_identity(parsed) == risk_test_matrix_identity(baseline)
    assert _dropped(caplog, "notes")
    noisy["rows"][0]["notes"] = ["not", "a", "string"]
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse_risk_test_matrix(noisy)


def test_plan_state_with_matrix_row_notes_is_accepted(caplog):
    examples = risk_test_matrix_prompt_examples()

    def use_applicable(payload, notes):
        matrix = copy.deepcopy(examples["applicable"])
        matrix["rows"][0]["notes"] = notes
        payload["risk_test_matrix"] = matrix
        payload["risk_test_matrix_changes"] = copy.deepcopy(examples["changes"])

    def parse(notes):
        return validate_structured_plan_state(
            _edit(structured_v1_plan_state(), lambda payload: use_applicable(payload, notes)),
            architecture_status_mode="legacy",
        )

    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = parse("Reviewer asked for this row.")
    assert plan.risk_test_matrix == parse_risk_test_matrix(examples["applicable"])
    assert _dropped(caplog, "notes")
    with pytest.raises(AgentLoopError, match="unknown field"):
        parse(42)


# --- plan-review-blocking-items -----------------------------------------------


def test_plan_review_blocking_items_keep_their_findings(caplog):
    text = _edit(
        structured_plan_review(state="blocking"),
        lambda payload: (
            payload.pop("blocking_plan_issues"),
            payload.pop("same_plan_followups"),
            payload.update(
                blocking_items=[{"text": "Step 3 misses the resume path.", "fix_scope": ["src/a.py"]}],
                same_pr_followups=["Name the test file."],
            ),
        ),
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_structured_plan_review(text, reviewer="Codex")
    assert parsed.state == "blocking"
    assert [item.text for item in parsed.items.blocking] == ["Step 3 misses the resume path."]
    assert [item.text for item in parsed.items.same_plan] == ["Name the test file."]
    assert "renamed field blocking_items -> blocking_plan_issues" in caplog.text


def test_plan_review_alias_beside_canonical_key_rejects():
    text = _with(structured_plan_review(state="blocking", blocking_plan_issues=["A."]), blocking_items=["B."])
    with pytest.raises(AgentLoopError, match="both `blocking_items` and `blocking_plan_issues`"):
        parse_structured_plan_review(text, reviewer="Codex")


# --- plan-disposition-sub-items -----------------------------------------------


def test_plan_review_disposition_sub_item_dispositions_degrade(caplog):
    text = structured_plan_review(
        prior_plan_item_dispositions=[{
            "item_id": "item-1", "disposition": "resolved", "note": "Fixed.",
            "sub_item_dispositions": {"item-1.s1": "resolved"},
        }],
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_structured_plan_review(text, reviewer="Codex")
    (disposition,) = parsed.dispositions
    assert (disposition.item_id, disposition.disposition, disposition.note) == ("item-1", "resolved", "Fixed.")
    assert disposition.sub_item_dispositions == ()
    (record,) = parsed.disposition_degradations
    assert record.rule == "sub_item_dispositions-not-supported-in-plan-review"
    assert "sub_item_dispositions" in caplog.text


# --- note-rename --------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "note", "messages"),
    [
        ({"reason": "R."}, "R.", ["renamed field reason -> note"]),
        ({"rationale": "Q."}, "Q.", ["renamed field rationale -> note"]),
        ({"reason": "R.", "rationale": "Q."}, "R.", ["renamed field reason -> note", "dropped field rationale"]),
        ({"note": "N.", "reason": "R."}, "N.", ["dropped field reason"]),
        ({"note": "N.", "rationale": "Q."}, "N.", ["dropped field rationale"]),
        ({"note": "N.", "reason": "R.", "rationale": "Q."}, "N.", ["dropped field reason", "dropped field rationale"]),
    ],
)
def test_review_disposition_note_aliases_follow_deterministic_precedence(extra, note, messages, caplog):
    text = structured_pr_review(
        state="blocking", blocking_items=["Gap."],
        prior_item_dispositions=[{"item_id": "item-1", "disposition": "blocking", **extra}],
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = parse_structured_pr_review(text, reviewer="Codex")
    assert parsed.dispositions[0].note == note
    for message in messages:
        assert message in caplog.text


@pytest.mark.parametrize("alias", ["reason", "rationale"])
@pytest.mark.parametrize("value", [None, ["Fixed."], 3], ids=["null", "singleton-list", "number"])
@pytest.mark.parametrize("with_note", [False, True], ids=["no-note", "beside-note"])
def test_non_string_note_alias_always_rejects(alias, value, with_note):
    disposition = {"item_id": "item-1", "disposition": "resolved", alias: value}
    if with_note:
        disposition["note"] = "N."
    with pytest.raises(AgentLoopError, match=f"{alias} must be a string to be read as `note`"):
        parse_structured_pr_review(
            structured_pr_review(prior_item_dispositions=[disposition]), reviewer="Codex",
        )


def test_list_valued_cross_kind_finding_alias_is_still_renamed():
    text = _edit(
        structured_plan_review(state="blocking"),
        lambda payload: (payload.pop("blocking_plan_issues"), payload.update(blocking_items=["A.", "B."])),
    )
    parsed = parse_structured_plan_review(text, reviewer="Codex")
    assert [item.text for item in parsed.items.blocking] == ["A.", "B."]


def test_known_rationale_is_never_renamed():
    payload = {"disposition": "direct-implementation", "rationale": "Kept.", "unresolved_design_decisions": []}
    result = _expect_known_keys_lenient(
        payload, context="execution_disposition",
        required={"disposition", "rationale", "unresolved_design_decisions"}, renames=NOTE_KEY_RENAMES,
    )
    assert result == payload and "note" not in result


# --- surplus-direct-ack -------------------------------------------------------


def test_surplus_checked_discussion_directly_is_accepted_with_a_warning(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_coder_followup_response(
            structured_coder_followup(addressed_items=["item-1", "item-2"], checked_discussion_directly=True),
            unresolved_items=ITEMS, human_requirements=(), architecture_status_mode="legacy",
        )
        implementation = _validate_issue_implementation_response(
            structured_issue_implementation(checked_discussion_directly=True),
            human_requirements=(), architecture_status_mode="legacy",
        )
    assert parsed.human_requirements.checked_discussion_directly is True
    assert implementation.pr_number == 77
    assert "ignored a surplus `true`" in caplog.text


def test_required_but_missing_direct_ack_still_rejects():
    with pytest.raises(AgentLoopError, match="must acknowledge"):
        validate_structured_human_requirements_acknowledgement(
            (), dispositions=(), checked_discussion_directly=False,
            surfaced_requirement_ids=(), requires_direct_discussion_ack=True,
        )


# --- addressed-ids-derived ----------------------------------------------------


@pytest.mark.parametrize("submitted", [[], [SURFACED_ID, UNSURFACED_ID]], ids=["omitted", "extra"])
def test_addressed_ids_are_derived_from_dispositions(submitted, caplog):
    text = structured_coder_followup(
        addressed_items=["item-1", "item-2"], human_requirement_ids=submitted,
        human_requirement_dispositions=[
            {"requirement_id": SURFACED_ID, "disposition": "addressed", "evidence": "Done."},
        ],
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_coder_followup_response(
            text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
        )
    assert parsed.human_requirements.addressed_ids == (SURFACED_ID,)
    assert "derived from `addressed` dispositions" in caplog.text


def test_reviewer_owned_and_legacy_addressed_labels_still_reject():
    owned = "hr-" + "c" * 64
    text = structured_coder_followup(
        addressed_items=["item-1", "item-2"], human_requirement_ids=[SURFACED_ID, owned],
        human_requirement_dispositions=[
            {"requirement_id": SURFACED_ID, "disposition": "addressed", "evidence": "Done."},
        ],
    )
    with pytest.raises(AgentLoopError, match="outside coder"):
        _validate_coder_followup_response(
            text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
            reviewer_owned_requirement_ids=(owned,),
        )
    legacy = _edit(text, lambda payload: payload["human_requirements"].update(addressed_ids=["Requirement 1"]))
    with pytest.raises(AgentLoopError, match="legacy positional"):
        _validate_coder_followup_response(
            legacy, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
        )


# --- unsurfaced-requirement-disposition ---------------------------------------


def _blocked(requirement_id: str) -> dict[str, str]:
    return {"requirement_id": requirement_id, "disposition": "blocked", "evidence": "Waiting on access."}


def _addressed(requirement_id: str) -> dict[str, str]:
    return {"requirement_id": requirement_id, "disposition": "addressed", "evidence": "Done."}


@pytest.mark.parametrize("surfaced", [False, True], ids=["none-surfaced", "some-surfaced"])
def test_approved_coder_followup_with_unsurfaced_blocked_disposition_is_accepted(surfaced, caplog):
    dispositions = [_blocked(UNSURFACED_ID), *([_addressed(SURFACED_ID)] if surfaced else [])]
    text = structured_coder_followup(
        state="approved", addressed_items=["item-1", "item-2"],
        human_requirement_ids=[SURFACED_ID] if surfaced else [],
        human_requirement_dispositions=dispositions,
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_coder_followup_response(
            text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,) if surfaced else (),
            architecture_status_mode="legacy",
        )
    assert parsed.state == "approved"
    assert [item.requirement_id for item in parsed.human_requirement_dispositions] == ([SURFACED_ID] if surfaced else [])
    assert f"unsurfaced requirement ID(s) {UNSURFACED_ID}" in caplog.text


def test_surfaced_blocked_requirement_still_rejects_an_approved_followup():
    text = structured_coder_followup(
        state="approved", addressed_items=["item-1", "item-2"], human_requirement_dispositions=[_blocked(SURFACED_ID)],
    )
    with pytest.raises(AgentLoopError, match="must be `blocking` when a signed human requirement is blocked"):
        _validate_coder_followup_response(
            text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
        )


def test_issue_implementation_with_unsurfaced_blocked_disposition_is_not_a_conflict():
    accepted = _validate_issue_implementation_response(
        structured_issue_implementation(human_requirement_dispositions=[_blocked(UNSURFACED_ID)]),
        human_requirements=(), architecture_status_mode="legacy",
    )
    assert isinstance(accepted, StructuredIssueImplementation)
    assert accepted.pr_number == 77 and accepted.human_requirement_dispositions == ()
    conflict = _validate_issue_implementation_response(
        structured_issue_implementation(human_requirement_dispositions=[_blocked(SURFACED_ID)]),
        human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
    )
    assert isinstance(conflict, _TerminalIssueImplementationConflict)


def test_missing_coverage_conflicting_duplicates_and_reviewer_owned_ids_still_reject():
    with pytest.raises(AgentLoopError, match="is missing requirement ID"):
        validate_human_requirement_dispositions(
            (HumanRequirementDisposition(UNSURFACED_ID, "addressed", "Done."),),
            surfaced_requirement_ids=(SURFACED_ID,),
        )
    with pytest.raises(AgentLoopError, match="duplicate requirement ID"):
        validate_human_requirement_dispositions(
            (
                HumanRequirementDisposition(SURFACED_ID, "addressed", "Done."),
                HumanRequirementDisposition(SURFACED_ID, "blocked", "Waiting."),
            ),
            surfaced_requirement_ids=(SURFACED_ID,),
        )
    with pytest.raises(AgentLoopError, match="outside coder"):
        validate_human_requirement_dispositions(
            (HumanRequirementDisposition(UNSURFACED_ID, "addressed", "Done."),),
            surfaced_requirement_ids=(),
            reviewer_owned_requirement_ids=(UNSURFACED_ID,),
        )
    kept = validate_human_requirement_dispositions(
        (
            HumanRequirementDisposition(SURFACED_ID, "addressed", "Done."),
            HumanRequirementDisposition(UNSURFACED_ID, "blocked", "Waiting."),
        ),
        surfaced_requirement_ids=(SURFACED_ID,),
    )
    assert kept == (HumanRequirementDisposition(SURFACED_ID, "addressed", "Done."),)


# --- coder-unknown-id ---------------------------------------------------------


def test_unknown_coder_item_id_is_dropped_with_its_notes(caplog):
    text = structured_coder_followup(
        addressed_items=["item-1", "item-99"], remaining_items=["item-2"],
        addressed_item_notes={"item-1": "Fixed.", "item-99": "Stale."},
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_coder_followup_response(
            text, unresolved_items=ITEMS, human_requirements=(), architecture_status_mode="legacy",
        )
    assert parsed.addressed_items == ("item-1",)
    assert parsed.addressed_item_notes == {"item-1": "Fixed."}
    assert parsed.dropped_unknown_items == ("item-99",)
    assert "item-99" in caplog.text
    with pytest.raises(AgentLoopError, match="did not classify all unresolved reviewer items"):
        _validate_coder_followup_response(
            structured_coder_followup(addressed_items=["item-1", "item-99"]),
            unresolved_items=ITEMS, human_requirements=(), architecture_status_mode="legacy",
        )


def test_shared_bucket_partition_keeps_order_and_non_strings():
    kept, dropped = partition_item_id_buckets(
        {"addressed_items": ["item-1", "x", 3, "x"], "remaining_items": "not-a-list"}, {"item-1"},
    )
    assert kept == {"addressed_items": ["item-1", 3], "remaining_items": "not-a-list"}
    assert dropped == ("x",)


# --- duplicates ---------------------------------------------------------------


def test_review_dispositions_collapse_only_when_fully_equal(caplog):
    same = {"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_review_response(
            structured_pr_review(prior_item_dispositions=[same, same]),
            reviewer="OpenAI Codex", unresolved_items=(_item("item-1"),), architecture_status_mode="legacy",
        )
    assert len(parsed.dispositions) == 1
    assert "collapsed a duplicate disposition" in caplog.text
    for other in (
        {**same, "note": "Different."},
        {**same, "evidence_requirement_ids": ["hr-" + "d" * 64]},
    ):
        with pytest.raises(AgentLoopError, match="more than once: item-1"):
            _validate_review_response(
                structured_pr_review(prior_item_dispositions=[same, other]),
                reviewer="OpenAI Codex", unresolved_items=(_item("item-1"),), architecture_status_mode="legacy",
            )


def test_coder_bucket_repeats_collapse_and_cross_bucket_repeats_reject():
    parsed = validate_structured_coder_followup(structured_coder_followup(addressed_items=["item-1", "item-1"]))
    assert parsed.addressed_items == ("item-1",)
    with pytest.raises(AgentLoopError, match="more than once: item-1"):
        validate_structured_coder_followup(
            structured_coder_followup(addressed_items=["item-1"], remaining_items=["item-1"])
        )


def test_identical_requirement_dispositions_collapse():
    text = structured_coder_followup(
        addressed_items=["item-1", "item-2"], human_requirement_ids=[SURFACED_ID],
        human_requirement_dispositions=[_addressed(SURFACED_ID), _addressed(SURFACED_ID)],
    )
    parsed = _validate_coder_followup_response(
        text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
    )
    assert len(parsed.human_requirement_dispositions) == 1
    conflicting = _edit(text, lambda payload: payload["human_requirement_dispositions"][1].update(evidence="Other."))
    with pytest.raises(AgentLoopError, match="duplicate requirement ID"):
        _validate_coder_followup_response(
            conflicting, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,), architecture_status_mode="legacy",
        )


# --- unlisted-note ------------------------------------------------------------


def test_note_keyed_to_unlisted_id_is_dropped(caplog):
    text = structured_coder_followup(addressed_items=["item-1"], remaining_item_notes={"item-1": "Wrong map."})
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = validate_structured_coder_followup(text)
    assert parsed.remaining_item_notes == {}
    assert "not listed in coder_followup.remaining_items" in caplog.text
    with pytest.raises(AgentLoopError, match="Missing evidence for: item-2"):
        validate_structured_coder_followup(
            structured_coder_followup(disputed_items=["item-2"], dispute_evidence={"item-1": "Wrong key."})
        )


# --- resume-reparse -----------------------------------------------------------


def test_resume_reparses_normalized_responses_deterministically():
    texts = [
        _with(structured_coder_followup(addressed_items=["item-1", "item-1"], addressed_item_notes={"item-9": "x"}), **EXTRAS),
        _with(structured_plan_review(prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "reason": "Fixed."},
        ]), **EXTRAS),
    ]
    first = [validate_structured_coder_followup(texts[0]), parse_structured_plan_review(texts[1], reviewer="Codex")]
    second = [validate_structured_coder_followup(texts[0]), parse_structured_plan_review(texts[1], reviewer="Codex")]
    assert first == second
    assert first[1].dispositions[0].note == "Fixed."


# --- historical-decode-roundtrip ----------------------------------------------


def test_historical_readers_decode_a_newly_accepted_coder_response():
    text = _with(
        structured_coder_followup(
            state="approved", summary="Fixed both items.", addressed_items=["item-1", "item-2"],
            tests_run=["pytest tests/test_a.py -q"],
            human_requirement_dispositions=[_blocked(UNSURFACED_ID)],
        ),
        **EXTRAS,
    )
    live = _validate_coder_followup_response(
        text, unresolved_items=ITEMS, human_requirements=(), architecture_status_mode="legacy",
    )
    assert isinstance(live, StructuredCoderFollowup) and live.state == "approved"

    carrier = rebuild_resumed_coder_carrier(text, None)
    assert isinstance(carrier, StructuredCoderFollowup)
    assert carrier.addressed_items == ("item-1", "item-2")
    assert carrier.summary == "Fixed both items."
    assert parse_historical_structured_coder_followup(text) is not None
    assert _extract_structured_coder_summary(text) == "Fixed both items."
    assert _extract_structured_coder_tests_run(text) == ("pytest tests/test_a.py -q",)
    metadata = PostedRoundMetadata(flow="pr", role="coder", agent="Claude", round_number=2, subject=HEAD)
    context = _coder_followup_review_context(text, metadata, head_sha=HEAD)
    assert "Fixed both items." in context

    # The same text with that requirement surfaced is still refused live.
    surfaced_text = text.replace(UNSURFACED_ID, SURFACED_ID)
    with pytest.raises(AgentLoopError, match="must be `blocking`"):
        _validate_coder_followup_response(
            surfaced_text, unresolved_items=ITEMS, human_requirements=(REQUIREMENT,),
            architecture_status_mode="legacy",
        )


# --- callers keep the filtered requirement dispositions (review item-1) -------
#
# Generalization: every caller of validate_human_requirement_dispositions that
# keeps a parsed record carries the returned (kept) tuple, and the strict
# semantic-patch path still rejects an unsurfaced ID.


@pytest.mark.parametrize("surfaced", [False, True], ids=["none-surfaced", "some-surfaced"])
def test_live_plan_review_validator_keeps_only_surfaced_dispositions(surfaced, caplog):
    from coding_review_agent_loop.unresolved_items import _validate_plan_review_response

    surfaced_ids = (SURFACED_ID,) if surfaced else ()
    dispositions = [_blocked(UNSURFACED_ID), *([_addressed(SURFACED_ID)] if surfaced else [])]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        parsed = _validate_plan_review_response(
            structured_plan_review(human_requirement_dispositions=dispositions),
            reviewer="OpenAI Codex", unresolved_items=(), surfaced_requirement_ids=surfaced_ids,
            architecture_status_mode="legacy",
        )
    assert [item.requirement_id for item in parsed.human_requirement_dispositions] == list(surfaced_ids)
    assert UNSURFACED_ID in caplog.text


def test_plan_state_acceptance_keeps_only_surfaced_dispositions():
    from coding_review_agent_loop.response_validation import _validate_response_with_human_requirements

    parsed = _validate_response_with_human_requirements(
        structured_plan_state(human_requirement_dispositions=[_blocked(UNSURFACED_ID)]),
        marker_validator=lambda text: validate_structured_plan_state(text, architecture_status_mode="legacy"),
        human_requirements=(), requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly.",
    )
    assert parsed.human_requirement_dispositions == ()


def test_semantic_patch_dispositions_stay_strict_for_unsurfaced_ids():
    from types import SimpleNamespace

    from coding_review_agent_loop.errors import SemanticPatchPayloadRejection
    from coding_review_agent_loop.response_validation import (
        _check_plan_revision_patch_human_requirement_dispositions,
    )

    with pytest.raises(SemanticPatchPayloadRejection, match=f"unknown requirement ID\\(s\\): {UNSURFACED_ID}"):
        _check_plan_revision_patch_human_requirement_dispositions(
            SimpleNamespace(operations=()),
            surfaced_requirement_ids=(SURFACED_ID,),
            inherited_human_requirement_dispositions=(
                HumanRequirementDisposition(SURFACED_ID, "addressed", "Done."),
                HumanRequirementDisposition(UNSURFACED_ID, "blocked", "Waiting."),
            ),
        )


def _plan_revision_with_matrix(notes):
    examples = risk_test_matrix_prompt_examples()

    def edit(payload):
        payload["kind"] = "plan_revision"
        payload["prior_plan_item_dispositions"] = []
        matrix = copy.deepcopy(examples["applicable"])
        if notes is not None:
            matrix["rows"][0]["notes"] = notes
        payload["risk_test_matrix"] = matrix
        payload["risk_test_matrix_changes"] = copy.deepcopy(examples["changes"])

    return _edit(structured_v1_plan_state(), edit)


def test_plan_revision_with_matrix_row_notes_is_accepted(caplog):
    baseline = validate_structured_plan_revision(_plan_revision_with_matrix(None), architecture_status_mode="legacy")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        plan = validate_structured_plan_revision(
            _plan_revision_with_matrix("Reviewer asked for this row."), architecture_status_mode="legacy"
        )
    assert plan.risk_test_matrix == baseline.risk_test_matrix
    assert risk_test_matrix_identity(plan.risk_test_matrix) == risk_test_matrix_identity(baseline.risk_test_matrix)
    assert _dropped(caplog, "notes")
    with pytest.raises(AgentLoopError, match="unknown field"):
        validate_structured_plan_revision(_plan_revision_with_matrix(["not", "text"]), architecture_status_mode="legacy")


# --- semantic-patch operations stay strict (review item-5, item-6) ------------
#
# Generalization: every parser a semantic plan-patch operation reaches runs
# inside strict_response_keys, so no #2114 drop, rename or collapse applies to
# a replacement value, while the same nested extras stay lenient in complete
# plan_state/plan_revision responses.


def _staged_recommendation(stage_extra):
    def stage(number, depends, extra):
        return {
            "stage_id": f"stage-{number}", "position": number, "title": f"Stage {number}",
            "summary": f"Stage {number}.", "deliverables": ["A."], "non_goals": [],
            "acceptance_criteria": ["Done."], "depends_on_stage_ids": depends,
            "dependency_notes": "None.", "automation": "agent-pr", "rollout_risk": "low",
            "compatibility_constraints": [], "covered_scope_item_ids": [f"scope-{number}"], **extra,
        }

    recommendation = copy.deepcopy(
        json.JSONDecoder().raw_decode(structured_v1_plan_state())[0]["execution_recommendation"]
    )
    recommendation.update(
        strategy="staged", staging_feasibility="safe",
        scope_items=[
            {"scope_item_id": "scope-1", "requirement": "First.", "acceptance_criteria": ["Done."]},
            {"scope_item_id": "scope-2", "requirement": "Second.", "acceptance_criteria": ["Done."]},
        ],
        child_stages=[stage(1, [], stage_extra), stage(2, ["stage-1"], {})],
    )
    recommendation.pop("one_shot_delivery")
    return recommendation


def _patch(field, value):
    return {
        "schema_version": 1, "kind": "plan_revision_patch", "semantic_patch_contract_version": 1,
        "state": "blocking", "summary": "Revise.", "prior_plan_item_dispositions": [],
        "base_round_number": 2, "base_state_identity": "a" * 64,
        "operations": [{"op": "replace", "field": field, "value": value}],
    }


_PATCH_CASES = {
    "requirement-disposition-extra": (
        "human_requirement_dispositions",
        lambda extra: [{**_addressed(SURFACED_ID), **({"notes": "extra"} if extra else {})}],
        "unknown field",
    ),
    "child-stage-extra": (
        "execution_recommendation",
        lambda extra: _staged_recommendation({"estimated_size": "small"} if extra else {}),
        "unknown field",
    ),
}


@pytest.mark.parametrize("case", sorted(_PATCH_CASES))
def test_semantic_patch_replacement_values_stay_strict(case, caplog):
    from coding_review_agent_loop.protocol import parse_plan_revision_patch

    field, build, message = _PATCH_CASES[case]
    parse_plan_revision_patch(_patch(field, build(False)))  # the clean value parses
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        with pytest.raises(AgentLoopError, match=message):
            parse_plan_revision_patch(_patch(field, build(True)))
    assert "dropped unknown field" not in caplog.text


def test_semantic_patch_duplicate_requirement_dispositions_are_not_collapsed():
    from coding_review_agent_loop.errors import SemanticPatchPayloadRejection
    from coding_review_agent_loop.protocol import parse_plan_revision_patch
    from coding_review_agent_loop.response_validation import (
        _check_plan_revision_patch_human_requirement_dispositions,
    )

    patch = parse_plan_revision_patch(
        _patch("human_requirement_dispositions", [_addressed(SURFACED_ID), _addressed(SURFACED_ID)])
    )
    assert len(patch.operations[0].value) == 2
    with pytest.raises(SemanticPatchPayloadRejection, match="duplicate requirement ID"):
        _check_plan_revision_patch_human_requirement_dispositions(
            patch, surfaced_requirement_ids=(SURFACED_ID,), inherited_human_requirement_dispositions=(),
        )


def test_strict_mode_is_scoped_to_the_patch_operations():
    from coding_review_agent_loop.protocol_common import response_keys_strict, strict_response_keys

    with strict_response_keys():
        assert response_keys_strict()
    assert not response_keys_strict()
