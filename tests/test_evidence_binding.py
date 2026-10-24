"""Deterministic risk-matrix row binding for untagged evidence-only findings (#1622)."""

from __future__ import annotations

import dataclasses

import pytest

from agent_loop_helpers import structured_coder_followup
from coding_review_agent_loop.evidence_binding import (
    BINDING_BOUND,
    BINDING_CODE_DEMAND,
    BINDING_NEEDS_REPAIR,
    BINDING_NOT_EVIDENCE,
    REASON_AMBIGUOUS_COLLECTIVE,
    REASON_MISSING_ROW_REFERENCE,
    REASON_UNKNOWN_ROW_REFERENCE,
    EvidenceBindingContext,
    derive_evidence_row_binding,
    derived_evidence_row_ids,
    flagged_finding_texts,
)
from coding_review_agent_loop.pr_loop_support import (
    HEAD_CHANGE_CITATION_ONLY,
    HEAD_CHANGE_CODE_FINDING,
    HEAD_CHANGE_EMPTY_ADDRESSED,
    HEAD_CHANGE_MACHINE_OBLIGATION,
    HEAD_CHANGE_NO_CITATION_ROWS,
    HEAD_CHANGE_UNKNOWN_ITEM,
    HEAD_CHANGE_UNTAGGED_EVIDENCE_REQUEST,
    FollowupHeadChange,
    _classify_followup_head_change,
    _followup_reports_code_changes,
    bind_carried_evidence_findings,
    untagged_evidence_stop_cause,
)
from coding_review_agent_loop.protocol import (
    EVIDENCE_OBLIGATION_KIND,
    PREDECESSOR_HEAD_UNMOVED_MESSAGE,
    PostAuthClaimDiagnostic,
    ReviewSubItem,
    UnresolvedReviewItem,
    validate_structured_coder_followup,
)
from coding_review_agent_loop.round_state import (
    _deserialize_unresolved_item,
    _serialize_unresolved_item,
)

ROWS = ("share-page-locale", "row-a", "row-b")
ALL_UNSATISFIED = EvidenceBindingContext(ROWS, frozenset(ROWS))
SOME_SATISFIED = EvidenceBindingContext(ROWS, frozenset({"row-a"}))
UNKNOWN_SET = EvidenceBindingContext(ROWS, None)

ITEM_9_TEXT = (
    "Provide admissible passing test observations for the 13 risk-matrix rows; "
    "do not change code just to move the SHA."
)
MIXED_TEXT = (
    "Fix the locale fallback defect for share-page-locale and provide passing test "
    "observations through run-tests"
)
MISSING_ROW_TEXT = "Provide passing test observations for the risk-matrix row."


def _derive(text, context=ALL_UNSATISFIED):
    return derive_evidence_row_binding(text, context)


# --- derivation --------------------------------------------------------------

def test_cited_row_ids_bind_in_order_of_first_appearance():
    binding = _derive("Provide passing run-tests observations for row-b and row-a.")
    assert binding == derive_evidence_row_binding(
        "Provide passing run-tests observations for row-b and row-a.", ALL_UNSATISFIED
    )
    assert binding.kind == BINDING_BOUND
    assert binding.rows == ("row-b", "row-a")


def test_backticked_row_ids_bind_and_duplicates_collapse():
    binding = _derive("Capture test observations for rows `row-a`, `row-a` and `row-b`.")
    assert (binding.kind, binding.rows) == (BINDING_BOUND, ("row-a", "row-b"))


def test_cited_rows_bind_even_when_the_unsatisfied_set_is_unknown():
    binding = _derive("Provide test observations for risk-matrix row `row-a`.", UNKNOWN_SET)
    assert (binding.kind, binding.rows) == (BINDING_BOUND, ("row-a",))


def test_a_substring_that_is_not_a_whole_token_is_not_a_citation():
    # `row-ab` contains `row-a` but is not that row; it is not a matrix reference.
    assert _derive("Capture an admissible passing test observation for row-ab").kind == (
        BINDING_NOT_EVIDENCE
    )
    assert _derive("Capture test observations for risk-matrix row `row-ab`").reason == (
        REASON_UNKNOWN_ROW_REFERENCE
    )


def test_collective_reference_binds_every_row_only_when_all_are_unsatisfied():
    """Matrix row `bind-collective-all-unsatisfied`."""
    bound = _derive(ITEM_9_TEXT, ALL_UNSATISFIED)
    assert (bound.kind, bound.rows) == (BINDING_BOUND, ROWS)
    for context in (SOME_SATISFIED, UNKNOWN_SET):
        binding = _derive(ITEM_9_TEXT, context)
        assert (binding.kind, binding.reason, binding.rows) == (
            BINDING_NEEDS_REPAIR, REASON_AMBIGUOUS_COLLECTIVE, ()
        )


@pytest.mark.parametrize(
    "text",
    [
        "Add tests for the loader and provide passing test observations.",
        "Capture an admissible passing test observation for the rows",
        "Please run the suite again.",
        "A real code defect in the loader",
    ],
)
def test_text_without_evidence_vocabulary_or_matrix_reference_is_not_evidence(text):
    assert _derive(text).kind == BINDING_NOT_EVIDENCE


def test_the_mixed_share_page_locale_finding_is_a_code_demand():
    """Matrix row `mixed-finding-not-bound` (derivation)."""
    for context in (ALL_UNSATISFIED, SOME_SATISFIED, UNKNOWN_SET):
        binding = _derive(MIXED_TEXT, context)
        assert (binding.kind, binding.rows) == (BINDING_CODE_DEMAND, ())


@pytest.mark.parametrize(
    "negation",
    [
        "do not change code",
        "don't change the code",
        "never change code",
        "no code changes",
        "without changing the code",
        "without modifying code",
        "you are not to change code",
    ],
)
def test_negated_code_change_phrases_do_not_count_as_code_demands(negation):
    text = f"Provide passing test observations for row-a; {negation}."
    assert _derive(text).kind == BINDING_BOUND


@pytest.mark.parametrize(
    "text",
    [
        "Provide passing test observations for row-a and update the fixture.",
        "Provide passing test observations for row-a; the guard is wrong.",
        "Provide passing test observations for row-a after the regression is fixed.",
        "Provide passing test observations for row-a; missing handling for None.",
        "Remove the code path and provide passing test observations for row-a.",
    ],
)
def test_any_other_code_demand_wording_blocks_binding(text):
    assert _derive(text).kind == BINDING_CODE_DEMAND


def test_unknown_and_missing_row_references_need_repair():
    """Matrix row `unknown-row-reference-repair` (derivation)."""
    unknown = _derive("Provide passing test observations for risk-matrix row `unknown-row`")
    assert (unknown.kind, unknown.reason, unknown.rows) == (
        BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()
    )
    mixed = _derive("Provide passing test observations for row-a and risk-matrix row `ghost-row`")
    assert (mixed.kind, mixed.reason, mixed.rows) == (
        BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()
    )
    missing = _derive(MISSING_ROW_TEXT)
    assert (missing.kind, missing.reason, missing.rows) == (
        BINDING_NEEDS_REPAIR, REASON_MISSING_ROW_REFERENCE, ()
    )


def test_rows_outside_the_enforceable_universe_are_never_bound():
    context = EvidenceBindingContext(("row-a",), frozenset({"row-a"}))
    binding = derive_evidence_row_binding(
        "Provide passing test observations for rows row-a and row-b.", context
    )
    assert binding.kind == BINDING_NEEDS_REPAIR
    assert "row-b" not in binding.rows


def test_flagged_finding_texts_lists_only_untagged_sub_item_free_needs_repair_entries():
    entries = [
        MISSING_ROW_TEXT,
        {"text": MISSING_ROW_TEXT + " again", "evidence_row_ids": ["row-a"]},
        {"text": MISSING_ROW_TEXT + " with parts", "sub_items": ["one", "two"]},
        {"text": "A real code defect in the loader"},
        {"text": "Provide passing test observations for row-a."},
        {"text": ITEM_9_TEXT},
    ]
    assert flagged_finding_texts(entries, SOME_SATISFIED) == (MISSING_ROW_TEXT, ITEM_9_TEXT)
    assert flagged_finding_texts(entries, ALL_UNSATISFIED) == (MISSING_ROW_TEXT,)
    assert flagged_finding_texts(entries, None) == ()


def test_minted_tag_keeps_reviewer_tags_and_skips_sub_items_and_missing_context():
    """Matrix row `explicit-tag-and-exclusions` (mint helper)."""
    text = "Provide passing test observations for row-a."
    assert derived_evidence_row_ids(
        text, reviewer_tag=("row-b",), has_sub_items=False, context=ALL_UNSATISFIED
    ) == ("row-b",)
    assert derived_evidence_row_ids(
        text, reviewer_tag=(), has_sub_items=True, context=ALL_UNSATISFIED
    ) == ()
    assert derived_evidence_row_ids(text, reviewer_tag=(), has_sub_items=False, context=None) == ()
    assert derived_evidence_row_ids(
        text, reviewer_tag=(), has_sub_items=False, context=ALL_UNSATISFIED
    ) == ("row-a",)
    assert derived_evidence_row_ids(
        MIXED_TEXT, reviewer_tag=(), has_sub_items=False, context=ALL_UNSATISFIED
    ) == ()


# --- head-change classification ---------------------------------------------

def _finding(item_id, text, rows=(), sub_items=()):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="codex", source_round=1, text=text,
        status="blocking", evidence_row_ids=tuple(rows), sub_items=tuple(sub_items),
    )


def _obligation(item_id, kind):
    evidence = kind == EVIDENCE_OBLIGATION_KIND
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="orchestrator", source_round=1, text=f"obligation {kind}",
        status="blocking", authority="machine", obligation_kind=kind,
        lifecycle="evidence_deferred" if evidence else "repair_required",
        failed_head_sha=None if evidence else "abc123",
        obligation_identity=f"{kind}:identity" if evidence else None,
    )


_LEDGER = (
    _finding("item-1", "Capture the rows", rows=("row-a",)),
    _finding("item-2", "A real code defect in the loader"),
    _finding("item-3", MISSING_ROW_TEXT),
    _finding("item-4", MIXED_TEXT),
    _finding("item-5", MISSING_ROW_TEXT, sub_items=(ReviewSubItem("item-5.s1", "sub"),)),
    _obligation("item-6", EVIDENCE_OBLIGATION_KIND),
    _obligation("item-7", "github-pr-checks"),
)


def _followup(addressed=()):
    return validate_structured_coder_followup(
        structured_coder_followup(addressed_items=list(addressed))
    )


@pytest.mark.parametrize(
    ("addressed", "rows", "expected", "reason", "item_ids"),
    [
        pytest.param(["item-1"], {"row-a"}, False, HEAD_CHANGE_CITATION_ONLY, (), id="citation-only"),
        pytest.param(["item-1"], set(), True, HEAD_CHANGE_NO_CITATION_ROWS, (), id="no-rows"),
        pytest.param([], {"row-a"}, True, HEAD_CHANGE_EMPTY_ADDRESSED, (), id="empty-addressed"),
        pytest.param(["item-99"], {"row-a"}, True, HEAD_CHANGE_UNKNOWN_ITEM, (), id="unknown-item"),
        pytest.param(["item-7"], {"row-a"}, True, HEAD_CHANGE_MACHINE_OBLIGATION, (), id="machine"),
        pytest.param(["item-2"], {"row-a"}, True, HEAD_CHANGE_CODE_FINDING, (), id="code-finding"),
        pytest.param(["item-4"], {"row-a"}, True, HEAD_CHANGE_CODE_FINDING, (), id="mixed-finding"),
        pytest.param(["item-5"], {"row-a"}, True, HEAD_CHANGE_CODE_FINDING, (), id="sub-items"),
        pytest.param(
            ["item-3"], {"row-a"}, True, HEAD_CHANGE_UNTAGGED_EVIDENCE_REQUEST, ("item-3",),
            id="untagged-evidence",
        ),
        pytest.param(
            ["item-1", "item-3", "item-6"], {"row-a"}, True,
            HEAD_CHANGE_UNTAGGED_EVIDENCE_REQUEST, ("item-3",), id="untagged-with-citations",
        ),
        pytest.param(
            ["item-2", "item-3"], {"row-a"}, True, HEAD_CHANGE_CODE_FINDING, (),
            id="code-outranks-untagged",
        ),
    ],
)
def test_head_change_classifier_reasons(addressed, rows, expected, reason, item_ids):
    """Matrix rows `code-finding-unchanged-head` and `empty-addressed-unchanged` (classifier)."""
    result = _classify_followup_head_change(_followup(addressed), _LEDGER, rows, ROWS)
    assert result == FollowupHeadChange(expected, reason, item_ids)
    # The boolean wrapper keeps the #1338 semantics exactly.
    assert _followup_reports_code_changes(
        _followup(addressed), prior_items=_LEDGER, citation_row_ids=rows
    ) is expected


def test_classifier_without_enforceable_rows_reports_untagged_items_as_code_findings():
    result = _classify_followup_head_change(_followup(["item-3"]), _LEDGER, {"row-a"})
    assert result == FollowupHeadChange(True, HEAD_CHANGE_CODE_FINDING)


# --- stop cause ---------------------------------------------------------------

_UNMOVED = PostAuthClaimDiagnostic("row-a", "checkout-head-mismatch", PREDECESSOR_HEAD_UNMOVED_MESSAGE)
_OTHER_MISMATCH = PostAuthClaimDiagnostic(
    "row-a", "checkout-head-mismatch",
    "The assigned checkout was not authenticated at the exact current PR head.",
)
_UNTAGGED = FollowupHeadChange(True, HEAD_CHANGE_UNTAGGED_EVIDENCE_REQUEST, ("item-9",))


def test_stop_cause_names_the_items_and_the_rejection():
    """Matrix row `stop-cause-diagnosis` (unit)."""
    cause = untagged_evidence_stop_cause(_UNTAGGED, (_UNMOVED,), head_moved=False)
    assert "item-9" in cause
    assert "checkout-head-mismatch" in cause
    assert "evidence_row_ids" in cause


@pytest.mark.parametrize(
    ("classification", "diagnostics", "head_moved"),
    [
        (_UNTAGGED, (_UNMOVED,), True),
        (_UNTAGGED, (_OTHER_MISMATCH,), False),
        (_UNTAGGED, (), False),
        (FollowupHeadChange(True, HEAD_CHANGE_CODE_FINDING), (_UNMOVED,), False),
        (FollowupHeadChange(True, HEAD_CHANGE_EMPTY_ADDRESSED), (_UNMOVED,), False),
        (None, (_UNMOVED,), False),
    ],
)
def test_stop_cause_is_empty_in_every_other_shape(classification, diagnostics, head_moved):
    assert untagged_evidence_stop_cause(classification, diagnostics, head_moved=head_moved) == ""


# --- carried binding ------------------------------------------------------------

def test_carried_binding_binds_only_derivable_untagged_carried_findings():
    """Matrix row `carried-legacy-bind-resume` (helper)."""
    cited = _finding("item-1", "Provide passing test observations for row-a; do not change code.")
    tagged = _finding("item-2", "Provide passing test observations for row-a.", rows=("row-b",))
    mixed = _finding("item-3", MIXED_TEXT)
    missing = _finding("item-4", MISSING_ROW_TEXT)
    new = _finding("item-5", "Provide passing test observations for row-b.")
    same_pr = dataclasses.replace(
        _finding("item-6", "Provide passing test observations for row-b."),
        status="same-pr", source_status="same-pr",
    )
    machine = _obligation("item-7", EVIDENCE_OBLIGATION_KIND)
    ledger = [cited, tagged, mixed, missing, new, same_pr, machine]
    result, bound = bind_carried_evidence_findings(
        ledger, ALL_UNSATISFIED,
        carried_item_ids=("item-1", "item-2", "item-3", "item-4", "item-6", "item-7"),
    )
    assert bound == (("item-1", ("row-a",)),)
    assert result[0] == dataclasses.replace(cited, evidence_row_ids=("row-a",))
    assert result[1:] == ledger[1:]
    assert bind_carried_evidence_findings(ledger, None, carried_item_ids=("item-1",)) == (
        ledger, ()
    )


def test_bound_tag_round_trips_through_round_state_serialization():
    item = dataclasses.replace(
        _finding("item-1", "Provide passing test observations for row-a."),
        evidence_row_ids=("row-a",),
    )
    assert _deserialize_unresolved_item(_serialize_unresolved_item(item)) == item


# --- keyword case never hides a row reference (#1622 review item-1) -----------

@pytest.mark.parametrize(
    ("text", "kind", "reason", "rows"),
    [
        # Mixed known and unknown rows behind capitalized list keywords.
        ("Provide passing test observations for risk-matrix Rows `row-a` and `ghost-row`.",
         BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()),
        ("Provide passing test observations for ROWS row-a AND ghost-row.",
         BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()),
        ("Provide passing test observations for Row: row-a Or `ghost-row`.",
         BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()),
        # Unknown-only behind a capitalized keyword must not fall through to a
        # collective binding of every row.
        ("Provide passing test observations for the Risk-Matrix Rows `ghost-row`.",
         BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()),
        ("Provide passing test observations for Row `ghost-row`.",
         BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()),
        # Capitalized vocabulary, collective and matrix phrases still classify.
        ("Provide PASSING TEST OBSERVATIONS for the RISK-MATRIX ROWS.", BINDING_BOUND, None, ROWS),
        ("Provide Passing Test Observations for the Risk-Matrix Row.",
         BINDING_NEEDS_REPAIR, REASON_MISSING_ROW_REFERENCE, ()),
        ("FIX the defect and provide passing test observations for row-a.",
         BINDING_CODE_DEMAND, None, ()),
        ("Provide passing test observations for row-a; Do Not Change Code.",
         BINDING_BOUND, None, ("row-a",)),
    ],
)
def test_keyword_case_never_hides_a_row_reference(text, kind, reason, rows):
    binding = _derive(text, ALL_UNSATISFIED)
    assert (binding.kind, binding.reason, binding.rows) == (kind, reason, rows)


def test_row_id_identity_stays_case_sensitive():
    # `ROW-A` is not the approved `row-a`: after a row keyword it is unknown.
    assert _derive("Provide passing test observations for Rows `ROW-A`.").reason == (
        REASON_UNKNOWN_ROW_REFERENCE
    )
    # A bare differently-cased token is not the approved ID and is never bound.
    bare = _derive("Provide passing test observations for ROW-A.")
    assert (bare.kind, bare.rows) == (BINDING_NOT_EVIDENCE, ())


# --- a row list is bound only when fully interpreted (#1622 review item-2) -----

_SEPARATORS = [", and", ", or", " and/or", "; and", " &", " plus", " as well as", ",", " and"]


@pytest.mark.parametrize("separator", _SEPARATORS)
def test_unknown_final_row_after_any_separator_prevents_partial_binding(separator):
    text = (
        "Provide passing test observations for risk-matrix rows `row-a`, `row-b`"
        f"{separator} `ghost-row`."
    )
    binding = _derive(text)
    assert (binding.kind, binding.reason, binding.rows) == (
        BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()
    )


@pytest.mark.parametrize("separator", _SEPARATORS)
def test_fully_known_list_after_any_separator_binds_every_listed_row(separator):
    text = (
        "Provide passing test observations for risk-matrix rows `row-a`, `row-b`"
        f"{separator} `share-page-locale`."
    )
    binding = _derive(text)
    assert (binding.kind, binding.rows) == (BINDING_BOUND, ("row-a", "row-b", "share-page-locale"))


@pytest.mark.parametrize(
    "tail",
    [
        " then `ghost-row`",
        " (and also `ghost-row`)",
        " with `ghost-row` too",
    ],
)
def test_unparsed_backticked_token_in_the_list_clause_forces_repair(tail):
    binding = _derive(f"Provide passing test observations for rows `row-a`, `row-b`{tail}.")
    assert (binding.kind, binding.reason, binding.rows) == (
        BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()
    )


def test_backticked_tokens_in_a_later_clause_do_not_affect_the_list():
    binding = _derive(
        "Provide passing test observations for rows `row-a` and `row-b`. "
        "Run them with `agent-loop run-tests`."
    )
    assert (binding.kind, binding.rows) == (BINDING_BOUND, ("row-a", "row-b"))


# --- the clause after a row list: row candidates vs command spans (items 3, 4) --

@pytest.mark.parametrize(
    "text",
    [
        # Bare unknown rows the list grammar did not reach (item-3).
        "Provide passing test observations for risk-matrix rows row-a, row-b, and also ghost-row.",
        "Provide passing test observations for rows row-a and row-b, then ghost-row.",
        "Provide passing test observations for rows row-a, row-b plus also ghost_row.",
        # Backticked unknown rows the grammar did not reach (item-2).
        "Provide passing test observations for rows `row-a`, `row-b` then `ghost-row`.",
        "Provide passing test observations for rows `row-a`, `row-b` with `ghost-row` too.",
        # An unknown row after a command span still counts.
        "Provide passing test observations for rows row-a via `agent-loop run-tests`, and also ghost-row.",
    ],
)
def test_unparsed_row_candidate_after_a_list_forces_repair(text):
    binding = _derive(text)
    assert (binding.kind, binding.reason, binding.rows) == (
        BINDING_NEEDS_REPAIR, REASON_UNKNOWN_ROW_REFERENCE, ()
    )


_COMMAND_TAILS = [
    " using `agent-loop run-tests`",
    " via `run-tests`",
    " through agent-loop run-tests",
    " by running `pytest tests/test_row_a.py -q`",
    " with `agent-loop run-tests -- python -m pytest`",
    " (rerun with run-tests)",
]


@pytest.mark.parametrize("tail", _COMMAND_TAILS)
def test_command_span_after_a_row_list_is_not_a_row_reference(tail):
    binding = _derive(f"Provide passing test observations for risk-matrix row `row-a`{tail}.")
    assert (binding.kind, binding.rows) == (BINDING_BOUND, ("row-a",))


@pytest.mark.parametrize("tail", _COMMAND_TAILS)
def test_carried_finding_with_a_command_span_is_bound(tail):
    item = _finding("item-1", f"Provide passing test observations for risk-matrix row `row-a`{tail}.")
    result, bound = bind_carried_evidence_findings(
        [item], ALL_UNSATISFIED, carried_item_ids=("item-1",)
    )
    assert bound == (("item-1", ("row-a",)),)
    assert result[0].evidence_row_ids == ("row-a",)
