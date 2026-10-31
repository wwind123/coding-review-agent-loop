"""Pure evidence conversion (#1325, stage 2 of #1317)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.evidence_conversion import (
    ConversionAuditRow,
    EvidenceConversionAudit,
    attempt_counts_for_row,
    auto_conversion_rows,
    capture_classification_payload,
    decode_capture_classification,
    reconcile_converted_obligations,
    validate_operator_conversion_rows,
)
from coding_review_agent_loop.evidence_stall import (
    StallRoundSnapshot,
    UnsatisfiedRows,
    classify_round,
)
from coding_review_agent_loop.protocol import (
    CAPTURE_LIMITED,
    NOT_CAPTURE_LIMITED,
    SemanticRiskCoverageClaim,
    SemanticRiskCoverageClaims,
    SignedEvidenceReference,
    UnresolvedReviewItem,
    derive_risk_test_matrix_evidence,
    disqualify_capture_classification,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
)
from coding_review_agent_loop.unresolved_items import (
    _upsert_converted_evidence_obligation,
    apply_residual_row_conversion,
    converted_evidence_obligation_identity,
    converted_row_owners,
    evidence_obligation_identity,
)

HEAD = "a1b2c3d4"
TREE = "tree-current"


def _row(row_id: str) -> dict:
    return {
        "row_id": row_id,
        "label": f"Behaviour {row_id}",
        "entry_path_or_mode": "agent-loop pr",
        "initial_state": "approved plan",
        "event": "coder reports",
        "expected_outcome": f"outcome {row_id}",
        "forbidden_side_effects": [f"no side effect {row_id}"],
        "proposed_test_level": "integration",
        "proposed_test_location": f"tests/test_{row_id}.py",
        "applicability": "applicable",
        "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }


def _matrix(*row_ids: str):
    return parse_risk_test_matrix(
        {"applicability": "applicable", "rows": [_row(r) for r in row_ids], "important_exclusions": []}
    )


def _observation(ref: str, *, outcome: str = "incomplete", head: str = HEAD, tree: str = TREE, **overrides):
    values = dict(
        execution_ref=ref,
        receipt_id=f"receipt-{ref}",
        command=("python3", "-m", "pytest", "tests/test_x.py", "-q"),
        normalized_command="python3 -m pytest tests/test_x.py -q",
        outcome=outcome,
        provenance="parent-observed",
        turn_id="turn-current",
        attribution={
            "state": "current-head",
            "head": head,
            "tracked_digest": tree,
            "stable": True,
            "untracked_input": False,
            "caveats": [],
        },
        environment_state="not-compared",
        superseded_by=None,
        caveats=(),
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _claim(row_id: str, *refs: str, **overrides) -> SemanticRiskCoverageClaim:
    values = dict(
        row_id=row_id,
        execution_refs=tuple(refs),
        test_identifiers=(f"tests/test_{row_id}.py::test_it",),
        test_locations=(f"tests/test_{row_id}.py",),
        workflow_path_claim="agent-loop pr",
        outcome_assertions=("The outcome holds.",),
        forbidden_effect_assertions=("No forbidden effect.",),
    )
    values.update(overrides)
    return SemanticRiskCoverageClaim(**values)


def _derive(matrix, claims, observations, *, journal=None, checkout_head=HEAD, tree_clean=True):
    return derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=SemanticRiskCoverageClaims(tuple(claims)),
        observations=tuple(observations if journal is None else journal),
        execution_catalog=tuple(observations),
        invocation_id="turn-current",
        current_head=HEAD,
        current_tree_digest=TREE,
        authenticated_checkout_head=checkout_head,
        authenticated_tree_clean=tree_clean,
        expected_identity=risk_test_matrix_identity(matrix),
    )


def _codes(result, row_id):
    return {diagnostic.code for diagnostic in result.diagnostics if diagnostic.row_id == row_id}


# --- structured classification (no-convert-test-failure) ---------------------


@pytest.mark.parametrize(
    "outcome,disqualifier",
    [("failed", "authoritative-failure"), ("timed_out", "authoritative-timeout")],
)
def test_same_aggregate_code_classifies_capture_limitation_apart_from_a_real_failure(outcome, disqualifier):
    """`no-convert-test-failure`: one aggregate code, two structured classifications."""
    matrix = _matrix("row-limited", "row-failed")
    result = _derive(
        matrix,
        [_claim("row-limited", "ref-1"), _claim("row-failed", "ref-2")],
        [_observation("ref-1", outcome="incomplete"), _observation("ref-2", outcome=outcome)],
        journal=[_observation("ref-1", outcome="incomplete")],
    )
    # Both rows carry the same aggregate rejection code ...
    assert "inadmissible-execution" in _codes(result, "row-limited")
    assert "inadmissible-execution" in _codes(result, "row-failed")
    # ... but only the capture limitation is eligible.
    limited = result.row_capture_classifications["row-limited"]
    failed = result.row_capture_classifications["row-failed"]
    assert limited["class"] == CAPTURE_LIMITED
    assert limited["observation_classes"] == ["outcome-incomplete"]
    assert failed["class"] == NOT_CAPTURE_LIMITED
    assert disqualifier in failed["disqualifiers"]


def test_passing_receipt_with_unverified_launch_is_a_capture_limitation():
    matrix = _matrix("row-a")
    result = _derive(
        matrix, [_claim("row-a", "ref-1")],
        [_observation("ref-1", outcome="passed", wrapper_bootstrap="unknown")],
    )
    assert "inadmissible-execution" in _codes(result, "row-a")
    assert result.row_capture_classifications["row-a"] == {
        "class": CAPTURE_LIMITED,
        "observation_classes": ["pass-unauthoritative"],
        "disqualifiers": [],
    }


def test_sole_unverified_suite_start_is_disqualified_not_capture_limited_1659():
    """`no-authority-granted`: such a pass stays unauthoritative and never converts.

    The coder can capture the check as a pytest test, so automatic conversion
    must not hand the row to a human.  A different unauthoritative pass (an
    unverified wrapper bootstrap) keeps its capture-limited classification.
    """
    matrix = _matrix("row-a", "row-b")
    claims = [_claim("row-a", "ref-1"), _claim("row-b", "ref-2")]
    observations = [
        _observation("ref-1", outcome="passed", suite_start="unknown"),
        _observation("ref-2", outcome="passed", wrapper_bootstrap="unknown", suite_start="unknown"),
    ]
    result = _derive(matrix, claims, observations)
    row_a = result.evidence.rows[0]
    assert row_a.status != "verified"
    assert row_a.evidence_citations == ()
    assert result.row_capture_classifications["row-a"] == {
        "class": NOT_CAPTURE_LIMITED,
        "observation_classes": ["pass-unauthoritative"],
        "disqualifiers": ["suite-start-unverified"],
    }
    assert result.row_capture_classifications["row-b"] == {
        "class": CAPTURE_LIMITED,
        "observation_classes": ["pass-unauthoritative"],
        "disqualifiers": [],
    }
    # N consecutive coder rounds, each at its own head, never convert row-a,
    # and every persisted record passes the closed-vocabulary validator.
    identity = risk_test_matrix_identity(matrix)
    classifications = {}
    for round_number in (1, 2, 3):
        payload = capture_classification_payload(result, head_sha=f"head-{round_number}")
        assert decode_capture_classification(
            payload, subject=f"head-{round_number}", evidence_identity=identity
        ) == payload
        classifications[round_number] = payload
    prior = {
        1: {**_snapshot(1, 1, ["row-a", "row-b"]), "matrix_identity": identity},
        2: {**_snapshot(2, 2, ["row-a", "row-b"]), "matrix_identity": identity},
    }
    current = dataclasses.replace(_current(3, 3, ["row-a", "row-b"]), matrix_identity=identity)
    converted = auto_conversion_rows(current, prior, classifications, 2, identity)
    assert "row-a" not in converted
    # The contrasting capture-limited row still converts under the same rounds.
    assert "row-b" in converted


def test_unknown_and_cross_turn_selectors_are_capture_limitations():
    matrix = _matrix("row-unknown", "row-cross")
    result = _derive(
        matrix,
        [_claim("row-unknown", "ref-missing"), _claim("row-cross", "ref-old")],
        [_observation("ref-old", outcome="passed", turn_id="turn-earlier")],
    )
    assert result.row_capture_classifications["row-unknown"]["class"] == CAPTURE_LIMITED
    assert result.row_capture_classifications["row-unknown"]["observation_classes"] == ["unknown-ref"]
    cross = result.row_capture_classifications["row-cross"]
    assert cross["class"] == CAPTURE_LIMITED
    assert "unknown-ref" in cross["observation_classes"]


@pytest.mark.parametrize(
    "observation_kwargs,derive_kwargs,disqualifier",
    [
        ({"head": "5e6741348108b11e3f3ef936c335b1967d17ac6c"}, {}, "head-mismatch"),
        ({"tree": "other-tree"}, {}, "tree-mismatch"),
        ({}, {"checkout_head": "elsewhere"}, "checkout-head-mismatch"),
        ({}, {"tree_clean": False}, "checkout-tree-unavailable"),
    ],
)
def test_wrong_head_or_tree_evidence_never_classifies_capture_limited(
    observation_kwargs, derive_kwargs, disqualifier
):
    matrix = _matrix("row-a")
    result = _derive(
        matrix, [_claim("row-a", "ref-1")],
        [_observation("ref-1", outcome="incomplete", **observation_kwargs)],
        **derive_kwargs,
    )
    classification = result.row_capture_classifications["row-a"]
    assert classification["class"] == NOT_CAPTURE_LIMITED
    assert disqualifier in classification["disqualifiers"]


def test_semantic_claim_defects_are_not_capture_limited():
    matrix = _matrix("row-missing", "row-incomplete")
    result = _derive(
        matrix,
        [_claim("row-incomplete", "ref-1", outcome_assertions=())],
        [_observation("ref-1")],
    )
    assert "missing-claim" in result.row_capture_classifications["row-missing"]["disqualifiers"]
    assert "incomplete-semantic-claim" in result.row_capture_classifications["row-incomplete"]["disqualifiers"]
    assert all(entry["class"] == NOT_CAPTURE_LIMITED for entry in result.row_capture_classifications.values())


def test_journal_disqualifier_covers_only_unsuperseded_failures_not_incomplete_runs():
    """Paired journal case: an unsuperseded incomplete entry stays eligible, a failed one does not."""
    matrix = _matrix("row-a")
    selected = _observation("ref-1", outcome="incomplete")
    incomplete_journal = _derive(
        matrix, [_claim("row-a", "ref-1")], [selected],
        journal=[selected, _observation("ref-other", outcome="incomplete")],
    )
    failed_journal = _derive(
        matrix, [_claim("row-a", "ref-1")], [selected],
        journal=[selected, _observation("ref-other", outcome="failed")],
    )
    superseded_failure = _derive(
        matrix, [_claim("row-a", "ref-1")], [selected],
        journal=[selected, _observation("ref-other", outcome="failed", superseded_by="ref-new")],
    )
    assert incomplete_journal.row_capture_classifications["row-a"]["class"] == CAPTURE_LIMITED
    assert failed_journal.row_capture_classifications["row-a"]["class"] == NOT_CAPTURE_LIMITED
    assert "journal-failure" in failed_journal.row_capture_classifications["row-a"]["disqualifiers"]
    assert superseded_failure.row_capture_classifications["row-a"]["class"] == CAPTURE_LIMITED


def test_a_verified_row_is_never_capture_limited():
    matrix = _matrix("row-a")
    result = _derive(matrix, [_claim("row-a", "ref-1")], [_observation("ref-1", outcome="passed")])
    assert result.evidence.rows[0].status == "verified"
    assert result.row_capture_classifications["row-a"]["disqualifiers"] == ["verified"]


def test_classification_replays_identically_through_round_metadata():
    """Per-attempt classification persists on the coder record and decodes identically."""
    matrix = _matrix("row-a", "row-b")
    result = _derive(
        matrix, [_claim("row-a", "ref-1"), _claim("row-b", "ref-2")],
        [_observation("ref-1"), _observation("ref-2", outcome="failed")],
    )
    payload = capture_classification_payload(result, head_sha=HEAD)
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=HEAD,
        risk_test_matrix_evidence=result.evidence.to_payload(),
        risk_test_matrix_capture_classification=payload,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.risk_test_matrix_capture_classification == payload
    assert decode_capture_classification(
        payload, subject=HEAD, evidence_identity=risk_test_matrix_identity(matrix)
    ) == payload


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {**p, "head": "ffff0000"},
        lambda p: {**p, "matrix_identity": "0" * 64},
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "class": "maybe"}}},
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "disqualifiers": ["made-up"]}}},
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "disqualifiers": ["verified"]}}},
        lambda p: {**p, "version": 2},
    ],
)
def test_malformed_or_misbound_classification_decodes_as_no_classification(mutate):
    matrix = _matrix("row-a")
    result = _derive(matrix, [_claim("row-a", "ref-1")], [_observation("ref-1")])
    payload = capture_classification_payload(result, head_sha=HEAD)
    assert decode_capture_classification(
        mutate(payload), subject=HEAD, evidence_identity=risk_test_matrix_identity(matrix)
    ) is None


def test_a_raced_correction_disqualifies_every_row():
    matrix = _matrix("row-a")
    result = _derive(matrix, [_claim("row-a", "ref-1")], [_observation("ref-1")])
    raced = disqualify_capture_classification(result, "head-changed-during-correction")
    assert raced.row_capture_classifications["row-a"]["class"] == NOT_CAPTURE_LIMITED


# --- attempt windows (convert-attempt-reset) -----------------------------------

IDENTITY = "a" * 64


def _snapshot(review_round, bound, rows, *, qualifies=True, head=None):
    return {
        "version": 1,
        "review_round": review_round,
        "review_head": head or f"head-{bound}",
        "matrix_identity": IDENTITY,
        "qualifies": qualifies,
        "unsatisfied_row_ids": sorted(rows),
        "reasons": [] if qualifies else ["untagged-finding"],
        **({"bound_attempt_round": bound} if bound is not None else {}),
    }


def _current(review_round, bound, rows, **kwargs):
    payload = _snapshot(review_round, bound, rows, **kwargs)
    return StallRoundSnapshot(
        review_round=review_round,
        review_head=payload["review_head"],
        matrix_identity=IDENTITY,
        qualifies=payload["qualifies"],
        unsatisfied_row_ids=tuple(payload["unsatisfied_row_ids"]),
        reasons=tuple(payload["reasons"]),
        bound_attempt_round=bound,
    )


def _classification(record_round, rows, *, limited=True, head=None):
    return {
        "version": 1,
        "matrix_identity": IDENTITY,
        "head": head or f"head-{record_round}",
        "rows": {
            row: {
                "class": CAPTURE_LIMITED if limited else NOT_CAPTURE_LIMITED,
                "observation_classes": ["outcome-incomplete"] if limited else ["failed"],
                "disqualifiers": [] if limited else ["authoritative-failure"],
            }
            for row in rows
        },
    }


def test_two_bound_capture_limited_attempts_convert_the_row():
    current = _current(2, 2, ["row-a"])
    prior = {1: _snapshot(1, 1, ["row-a"])}
    classifications = {1: _classification(1, ["row-a"]), 2: _classification(2, ["row-a"])}
    assert auto_conversion_rows(current, prior, classifications, 2, IDENTITY) == {"row-a": (1, 2)}
    assert auto_conversion_rows(current, prior, classifications, 0, IDENTITY) == {}
    assert auto_conversion_rows(current, prior, classifications, 2, IDENTITY, {"row-a"}) == {}
    assert auto_conversion_rows(current, prior, classifications, 2, "b" * 64) == {}


@pytest.mark.parametrize(
    "earlier",
    [
        "verified",  # the row was verified, so absent from the unsatisfied set
        "not-limited",
        "legacy-snapshot",
        "non-qualifying",
        "repeat-attempt",
        "5e6741348108b11e3f3ef936c335b1967d17ac6c-classification",
    ],
)
def test_an_ineligible_earlier_attempt_resets_the_count(earlier):
    """`convert-attempt-reset`: the count restarts at the next capture-limited attempt."""
    rows_1 = [] if earlier == "verified" else ["row-a"]
    prior = {
        1: _snapshot(
            1,
            None if earlier == "legacy-snapshot" else (2 if earlier == "repeat-attempt" else 1),
            rows_1 or ["row-b"],
            qualifies=earlier != "non-qualifying",
        ),
    }
    classifications = {
        1: _classification(
            1,
            ["row-a"],
            limited=earlier != "not-limited",
            head="elsewhere" if earlier == "5e6741348108b11e3f3ef936c335b1967d17ac6c-classification" else None,
        ),
        2: _classification(2, ["row-a"]),
        3: _classification(3, ["row-a"]),
    }
    current = _current(2, 2, ["row-a"])
    assert auto_conversion_rows(current, prior, classifications, 2, IDENTITY) == {}
    # Two fresh qualifying attempts after the reset convert it.
    later_prior = {**prior, 2: _snapshot(2, 2, ["row-a"])}
    assert auto_conversion_rows(_current(3, 3, ["row-a"]), later_prior, classifications, 2, IDENTITY) == {
        "row-a": (2, 3)
    }


def test_attempt_counts_only_for_the_bound_head_and_matrix():
    classification = _classification(1, ["row-a"])
    assert attempt_counts_for_row(classification, "row-a", matrix_identity=IDENTITY, head_sha="head-1")
    assert not attempt_counts_for_row(classification, "row-a", matrix_identity=IDENTITY, head_sha="head-2")
    assert not attempt_counts_for_row(classification, "row-b", matrix_identity=IDENTITY, head_sha="head-1")
    assert not attempt_counts_for_row(None, "row-a", matrix_identity=IDENTITY, head_sha="head-1")


def test_a_non_qualifying_current_review_never_converts():
    current = _current(2, 2, ["row-a"], qualifies=False)
    prior = {1: _snapshot(1, 1, ["row-a"])}
    classifications = {1: _classification(1, ["row-a"]), 2: _classification(2, ["row-a"])}
    assert auto_conversion_rows(current, prior, classifications, 2, IDENTITY) == {}


# --- residual tags (convert-partial-multirow) ---------------------------------


def _finding(item_id="item-1", rows=("row-a", "row-b"), reviewer="Codex"):
    return UnresolvedReviewItem(
        item_id=item_id, reviewer=reviewer, source_round=1,
        text="Capture admissible evidence", status="blocking", evidence_row_ids=tuple(rows),
    )


def test_partial_conversion_keeps_a_residual_tag_and_only_full_conversion_closes_the_finding():
    ledger = [_finding()]
    ledger, converted = apply_residual_row_conversion(ledger, {"row-a": "item-2"})
    assert converted == {"row-a": ("item-1",)}
    finding = ledger[0]
    assert finding.status == "blocking"
    assert finding.evidence_row_ids == ("row-a", "row-b")
    assert finding.residual_evidence_row_ids == ("row-b",)
    # The residual finding still classifies as evidence-only for row-b.
    snapshot = classify_round(
        review_round=2, review_head=HEAD, approved_identity=IDENTITY, open_items=ledger,
        unsatisfied=UnsatisfiedRows(rows={"row-b": "canonical-status:missing"}),
        checks=None, freeze_active=False,
    )
    assert "tag-names-satisfied-row" not in snapshot.reasons
    ledger, converted = apply_residual_row_conversion(ledger, {"row-b": "item-3"})
    finding = ledger[0]
    assert finding.status == "converted"
    assert finding.converted_row_ids == ("row-a", "row-b")
    assert finding.is_inert_ledger_record
    assert any("item-3" in note for note in finding.notes)


def test_a_new_tag_on_a_converted_row_is_an_ordinary_finding():
    retagged = _finding(item_id="item-5", rows=("row-a",))
    snapshot = classify_round(
        review_round=3, review_head=HEAD, approved_identity=IDENTITY, open_items=[retagged],
        # row-a was converted, so it is excluded from the unsatisfied set.
        unsatisfied=UnsatisfiedRows(rows={"row-b": "canonical-status:missing"}),
        checks=None, freeze_active=False,
    )
    assert "tag-names-satisfied-row" in snapshot.reasons


def test_converted_owners_are_tagging_reviewers_or_the_whole_board():
    ledger = [_finding(reviewer="Codex"), _finding(item_id="item-2", rows=("row-a",), reviewer="Gemini")]
    assert converted_row_owners(ledger, "row-a", configured_reviewers=("Codex", "Gemini", "Claude")) == (
        "Codex", "Gemini",
    )
    assert converted_row_owners(ledger, "row-z", configured_reviewers=("Codex", "Claude")) == ("Codex", "Claude")


def test_minting_is_idempotent_per_row_and_matrix_identity():
    ledger, item, consumed = _upsert_converted_evidence_obligation(
        [], item_number=4, row_id="row-a", matrix_identity=IDENTITY, head_sha=HEAD,
        owners=("Codex",), source_round=2, explanation="canonical-status:stale/unverified",
    )
    assert consumed and item.item_id == "item-4"
    assert item.obligation_identity == converted_evidence_obligation_identity("row-a", IDENTITY)
    assert item.obligation_identity.startswith("human-exact-head-evidence:row:")
    again, same, consumed_again = _upsert_converted_evidence_obligation(
        ledger, item_number=5, row_id="row-a", matrix_identity=IDENTITY, head_sha=HEAD,
        owners=("Codex",), source_round=3, explanation="",
    )
    assert not consumed_again and same.item_id == "item-4" and len(again) == 1
    # Reviewer-requested identities keep their text-and-reviewer namespace.
    assert ":row:" not in evidence_obligation_identity("Run the live suite", "Codex")


# --- operator validation (operator-convert) -----------------------------------


def test_operator_rows_must_all_be_currently_unsatisfied():
    assert validate_operator_conversion_rows(["row-a"], unsatisfied_row_ids=["row-a", "row-b"]) == ("row-a",)
    with pytest.raises(AgentLoopError, match="rejected: row-z"):
        validate_operator_conversion_rows(["row-a", "row-z"], unsatisfied_row_ids=["row-a"])
    with pytest.raises(AgentLoopError, match="more than once"):
        validate_operator_conversion_rows(["row-a", "row-a"], unsatisfied_row_ids=["row-a"])
    with pytest.raises(AgentLoopError, match="cannot be applied"):
        validate_operator_conversion_rows(["row-a"], unsatisfied_row_ids=None)
    # An already-converted row (pending or cleared) is rejected, alone or mixed.
    for requested in (["row-a"], ["row-a", "row-b"]):
        with pytest.raises(AgentLoopError, match="rejected: row-a"):
            validate_operator_conversion_rows(
                requested, unsatisfied_row_ids=["row-a", "row-b"], already_converted=["row-a"]
            )


# --- audit record and reconciliation ------------------------------------------


def _audit(**overrides):
    values = dict(
        source="auto",
        matrix_identity=IDENTITY,
        head=HEAD,
        rows=(
            ConversionAuditRow(
                row_id="row-a",
                obligation_identity=converted_evidence_obligation_identity("row-a", IDENTITY),
                obligation_item_id="item-4",
                owners=("Codex",),
                converted_item_ids=("item-1",),
                attempt_record_rounds=(1, 2),
            ),
        ),
    )
    values.update(overrides)
    return EvidenceConversionAudit(**values)


def test_audit_round_trips_and_rejects_forged_identities():
    audit = _audit()
    assert EvidenceConversionAudit.from_payload(audit.to_payload()) == audit
    forged = audit.to_payload()
    forged["rows"][0]["obligation_identity"] = "human-exact-head-evidence:row:" + "0" * 24
    assert EvidenceConversionAudit.from_payload(forged) is None
    no_attempts = audit.to_payload()
    no_attempts["rows"][0]["attempt_record_rounds"] = []
    assert EvidenceConversionAudit.from_payload(no_attempts) is None
    operator = _audit(source="operator").to_payload()
    operator["rows"][0]["attempt_record_rounds"] = []
    assert EvidenceConversionAudit.from_payload(operator) is not None


def _obligation(**overrides):
    ledger, item, _ = _upsert_converted_evidence_obligation(
        [], item_number=4, row_id="row-a", matrix_identity=IDENTITY, head_sha=HEAD,
        owners=("Codex",), source_round=2, explanation="",
    )
    return dataclasses.replace(item, **overrides)


def _reference(head=HEAD, row_id="row-a"):
    return SignedEvidenceReference.build(
        matrix_identity=IDENTITY, row_id=row_id, head=head,
        obligation_identity=converted_evidence_obligation_identity(row_id, IDENTITY),
        obligation_item_id="item-4", owners=("Codex",), signed_requirement_ids=("hr-1",),
        clearing_reviewers=("Codex",), cleared_round=3,
    )


def test_reconciliation_reopens_a_tombstone_at_a_new_head_with_the_same_identity():
    tombstone = _obligation(status="resolved", lifecycle="cleared", candidate_head_sha=HEAD)
    # Kept at its head while its admitted reference stands.
    same = reconcile_converted_obligations(
        [tombstone], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[_audit()], admitted_references=[_reference()]
    )
    assert same == [tombstone]
    # Without the reference a tombstone alone never satisfies the row.
    reopened = reconcile_converted_obligations([tombstone], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[_audit()])
    assert reopened[0].lifecycle == "evidence_deferred" and reopened[0].status == "blocking"
    moved = reconcile_converted_obligations(
        [tombstone], live_head="ffff0001", matrix_identity=IDENTITY, admitted_audits=[_audit()], admitted_references=[_reference()]
    )
    assert moved[0].item_id == "item-4"
    assert moved[0].obligation_identity == tombstone.obligation_identity
    assert (moved[0].status, moved[0].lifecycle, moved[0].candidate_head_sha) == (
        "blocking", "evidence_deferred", None,
    )
    # Idempotent.
    assert reconcile_converted_obligations(
        moved, live_head="ffff0001", matrix_identity=IDENTITY, admitted_audits=[_audit()], admitted_references=[_reference()]
    ) == moved


def test_reconciliation_supersedes_by_tool_evidence_and_reopens_at_the_next_head():
    pending = _obligation()
    superseded = reconcile_converted_obligations(
        [pending], live_head="ffff0002", matrix_identity=IDENTITY, admitted_audits=[_audit()],
        tool_verified_rows_at_live_head=["row-a"],
    )[0]
    assert (superseded.status, superseded.lifecycle, superseded.candidate_head_sha) == (
        "resolved", "cleared", "ffff0002",
    )
    assert any("tool-captured" in note for note in superseded.notes)
    reopened = reconcile_converted_obligations(
        [superseded], live_head="ffff0003", matrix_identity=IDENTITY, admitted_audits=[_audit()]
    )[0]
    assert (reopened.item_id, reopened.lifecycle) == ("item-4", "evidence_deferred")


def test_reconciliation_releases_a_freeze_at_another_head_and_restores_from_an_audit():
    frozen = _obligation(lifecycle="evidence_frozen", candidate_head_sha=HEAD)
    assert reconcile_converted_obligations(
        [frozen], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[_audit()]
    ) == [frozen]
    released = reconcile_converted_obligations([frozen], live_head="ffff0004", matrix_identity=IDENTITY, admitted_audits=[_audit()])
    assert released[0].lifecycle == "evidence_deferred"
    restored = reconcile_converted_obligations(
        [], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[_audit()]
    )
    assert [(item.item_id, item.resolution_owners, item.lifecycle) for item in restored] == [
        ("item-4", ("Codex",), "evidence_deferred")
    ]
    # Another matrix identity's audit is ignored.
    assert reconcile_converted_obligations(
        [], live_head=HEAD, matrix_identity="b" * 64, admitted_audits=[_audit()]
    ) == []


# --- review round 1 regressions -------------------------------------------------


@pytest.mark.parametrize(
    "derive_kwargs,disqualifier",
    [
        ({"checkout_head": "elsewhere"}, "checkout-head-mismatch"),
        ({"tree_clean": False}, "checkout-tree-unavailable"),
    ],
)
@pytest.mark.parametrize("selector", ["unknown", "dropped"])
def test_checkout_disqualifiers_apply_even_without_a_resolved_selector(derive_kwargs, disqualifier, selector):
    """item-2: a claim selecting only unknown or dropped refs from a wrong or
    unauthenticated checkout is never capture-limited."""
    matrix = _matrix("row-a")
    claim = (
        _claim("row-a", "ref-missing")
        if selector == "unknown"
        else _claim("row-a", dropped_execution_refs=("ref-dropped",))
    )
    result = _derive(matrix, [claim], [], **derive_kwargs)
    classification = result.row_capture_classifications["row-a"]
    assert classification["class"] == NOT_CAPTURE_LIMITED
    assert disqualifier in classification["disqualifiers"]
    assert classification["observation_classes"] == ["unknown-ref"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "class": []}}},
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "observation_classes": [{}]}}},
        lambda p: {**p, "rows": {"row-a": {**p["rows"]["row-a"], "disqualifiers": [[]]}}},
        lambda p: {**p, "rows": {"row-a": []}},
        lambda p: {**p, "head": ["x"]},
    ],
)
def test_ill_typed_classification_values_decode_as_none_without_raising(mutate):
    """item-4: tolerant decoders never raise on ill-typed JSON values."""
    matrix = _matrix("row-a")
    result = _derive(matrix, [_claim("row-a", "ref-1")], [_observation("ref-1")])
    payload = mutate(capture_classification_payload(result, head_sha=HEAD))
    assert decode_capture_classification(
        payload, subject=HEAD, evidence_identity=risk_test_matrix_identity(matrix)
    ) is None
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=HEAD,
        risk_test_matrix_evidence=result.evidence.to_payload(),
        risk_test_matrix_capture_classification=payload,
    )
    assert _decode_round_metadata(_encode_round_metadata(metadata)).risk_test_matrix_capture_classification is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: {**p, "source": []},
        lambda p: {**p, "matrix_identity": {}},
        lambda p: {**p, "rows": [{**p["rows"][0], "owners": [[]]}]},
        lambda p: {**p, "rows": [{**p["rows"][0], "row_id": []}]},
    ],
)
def test_ill_typed_audits_decode_as_unconverted_without_raising(mutate):
    from coding_review_agent_loop.round_state import _decode_round_metadata as decode, _encode_round_metadata as encode

    payload = mutate(_audit().to_payload())
    assert EvidenceConversionAudit.from_payload(payload) is None
    from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

    stored = decode_mapping(encode(PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=HEAD,
        phase="evidence-conversion", evidence_conversion=_audit(),
    )))
    stored["evidence_conversion"] = payload
    assert decode(encode_mapping(stored)).evidence_conversion is None


def test_conversion_state_without_an_admitted_audit_is_dropped_and_the_finding_reopens():
    """item-3: a ledger adopted from a record whose audit was rejected grants no conversion."""
    obligation = _obligation()
    finding = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Capture evidence",
        status="converted", evidence_row_ids=("row-a",), converted_row_ids=("row-a",),
    )
    unbacked = reconcile_converted_obligations(
        [finding, obligation], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[],
    )
    assert [item.item_id for item in unbacked] == ["item-1"]
    assert unbacked[0].status == "blocking" and unbacked[0].converted_row_ids == ()
    backed = reconcile_converted_obligations(
        [finding, obligation], live_head=HEAD, matrix_identity=IDENTITY, admitted_audits=[_audit()],
    )
    assert [item.item_id for item in backed] == ["item-1", "item-4"]
    assert backed[0].status == "converted"


def test_a_board_change_never_reassigns_converted_owners_to_a_primary_alone():
    """item-1: the board-amendment ledger view keeps stored owners, so the
    departed-owner rule (every current reviewer) applies, not primary-only."""
    from coding_review_agent_loop.board_amendment import apply_board_amendment_to_ledger
    from coding_review_agent_loop.protocol import ReviewItemDisposition
    from coding_review_agent_loop.unresolved_items import _apply_unresolved_item_dispositions

    frozen = _obligation(lifecycle="evidence_frozen", candidate_head_sha=HEAD)  # owned by Codex
    view, reassignments = apply_board_amendment_to_ledger(
        [frozen], removed_reviewers=("Codex",), remaining_reviewers=("Gemini", "Claude"),
        primary_reviewer="Gemini",
    )
    assert view[0].resolution_owners == ("Codex",) and reassignments == ()

    def clears(resolvers):
        clearances: list = []
        kept, _ = _apply_unresolved_item_dispositions(
            list(view),
            {"item-4": [ReviewItemDisposition("item-4", name, "resolved") for name in resolvers]},
            evidence_response_head=HEAD, configured_reviewers=("Gemini", "Claude"),
            evidence_clearances=clearances,
        )
        return bool(clearances) and kept[0].lifecycle == "cleared"

    assert not clears(["Gemini"])  # the new primary alone cannot clear it
    assert clears(["Gemini", "Claude"])
