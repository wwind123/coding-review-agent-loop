"""Pure evidence-only stall classification (#1324)."""

from __future__ import annotations

from coding_review_agent_loop.evidence_stall import (
    CheckBoardSummary,
    StallRoundSnapshot,
    UnsatisfiedRows,
    approved_matrix_row_ids,
    classify_round,
    stall_window,
    unsatisfied_rows,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem

IDENTITY = "a" * 64
OTHER_IDENTITY = "b" * 64
HEAD = "c" * 40
ROWS = ("row-a", "row-b")


def _row(row_id: str, status: str = "stale/unverified") -> dict[str, object]:
    verified = status == "verified"
    return {
        "row_id": row_id,
        "status": status,
        "test_identifiers": ["tests/test_x.py::t"] if verified else [],
        "test_locations": ["tests/test_x.py"] if verified else [],
        "workflow_path_claim": "path",
        "outcome_assertions": ["ok"],
        "forbidden_effect_assertions": ["none"],
        "evidence_citations": (
            [{"command": "pytest tests/test_x.py", "receipt_id": "exec-1", "claim": "current-result"}]
            if verified else []
        ),
    }


def _evidence(statuses: dict[str, str], identity: str = IDENTITY) -> dict[str, object]:
    return {"matrix_identity": identity, "rows": [_row(r, s) for r, s in statuses.items()]}


def _item(item_id="item-1", *, tags=(), status="blocking", **extra) -> UnresolvedReviewItem:
    return UnresolvedReviewItem(
        item_id=item_id, reviewer="Codex", source_round=1, text="evidence needed",
        status=status, evidence_row_ids=tuple(tags), **extra,
    )


def _machine(kind: str, lifecycle: str = "repair_required") -> UnresolvedReviewItem:
    return UnresolvedReviewItem(
        item_id=f"item-{kind}", reviewer="Orchestrator", source_round=1, text="m",
        status="blocking", authority="machine", obligation_kind=kind, lifecycle=lifecycle,
        obligation_identity=kind, failed_head_sha=HEAD,
        **({"candidate_head_sha": HEAD} if lifecycle == "evidence_frozen" else {}),
    )


def _unsat(*row_ids: str) -> UnsatisfiedRows:
    return UnsatisfiedRows(rows={r: "canonical-status:stale/unverified" for r in row_ids})


GREEN = CheckBoardSummary()


def _classify(items, unsat=None, checks=GREEN, freeze=False, identity=IDENTITY, round_=2):
    return classify_round(
        review_round=round_, review_head=HEAD, approved_identity=identity,
        open_items=items, unsatisfied=_unsat("row-a") if unsat is None else unsat,
        checks=checks, freeze_active=freeze,
    )


# --- unsatisfied_rows -------------------------------------------------------

def test_unsatisfied_rows_orders_codes_and_uses_canonical_status_fallback():
    evidence = _evidence({"row-a": "stale/unverified", "row-b": "verified"})
    diagnostics = [
        {"row_id": "row-a", "code": "launch-integrity", "message": "m"},
        {"row_id": "row-a", "code": "capture_incomplete", "message": "m"},
        {"row_id": "row-b", "code": "ignored-for-verified-row", "message": "m"},
    ]
    result = unsatisfied_rows(evidence, diagnostics, ROWS, IDENTITY)
    assert result.reason is None
    assert dict(result.rows) == {
        "row-a": "canonical-status:stale/unverified, capture_incomplete, launch-integrity"
    }


def test_unsatisfied_row_without_any_diagnostic_still_has_an_explanation():
    # Correction-time head race: the row is downgraded but no diagnostic names it.
    result = unsatisfied_rows(_evidence({"row-a": "stale/unverified", "row-b": "incomplete"}), [], ROWS, IDENTITY)
    assert dict(result.rows) == {
        "row-a": "canonical-status:stale/unverified",
        "row-b": "canonical-status:incomplete",
    }


def test_diagnostic_of_another_row_is_never_copied():
    diagnostics = [{"row_id": "row-b", "code": "only-b", "message": "m"}]
    result = unsatisfied_rows(_evidence({"row-a": "missing", "row-b": "missing"}), diagnostics, ROWS, IDENTITY)
    assert result.rows["row-a"] == "canonical-status:missing"
    assert "only-b" in result.rows["row-b"]


def test_absent_evidence_is_not_every_row_unverified():
    result = unsatisfied_rows(None, [], ROWS, IDENTITY)
    assert result.reason == "evidence-absent" and not result.rows


def test_matrix_identity_mismatch_unbinds_evidence():
    result = unsatisfied_rows(_evidence({"row-a": "missing", "row-b": "missing"}, OTHER_IDENTITY), [], ROWS, IDENTITY)
    assert result.reason == "evidence-matrix-mismatch" and not result.rows


def test_evidence_missing_an_enforceable_row_is_a_mismatch():
    result = unsatisfied_rows(_evidence({"row-a": "missing"}), [], ROWS, IDENTITY)
    assert result.reason == "evidence-matrix-mismatch"


# --- classify_round ---------------------------------------------------------

def test_tagged_evidence_only_round_qualifies():
    snap = _classify([_item(tags=["row-a"])])
    assert snap.qualifies and snap.reasons == () and snap.unsatisfied_row_ids == ("row-a",)
    assert snap.to_payload()["version"] == 1


def test_evidence_obligation_does_not_disqualify():
    items = [_item(tags=["row-a"]), _machine("human-exact-head-evidence", "evidence_deferred")]
    assert _classify(items).qualifies


def test_reason_table():
    tagged = _item(tags=["row-a"])
    cases = {
        "untagged-finding": _classify([_item()]),
        "tag-names-satisfied-row": _classify([_item(tags=["row-b"])]),
        "machine-obligation-open": _classify([tagged, _machine("github-pr-checks")]),
        "merge-conflict-obligation": _classify([tagged, _machine("merge-conflict")]),
        "checks-failing": _classify([tagged], checks=CheckBoardSummary(failing=("check-run:ci",))),
        "checks-stalled": _classify([tagged], checks=CheckBoardSummary(infrastructure_stalls=("check-run:q",))),
        "checks-unavailable": _classify([tagged], checks=None),
        "evidence-freeze-active": _classify([tagged], freeze=True),
        "no-open-items": _classify([]),
        "no-applicable-matrix": _classify([tagged], identity=None),
        "evidence-head-unbound": classify_round(
            review_round=2, review_head=HEAD, approved_identity=IDENTITY, open_items=[tagged],
            unsatisfied=None, checks=GREEN, freeze_active=False,
        ),
        "evidence-absent": _classify([tagged], unsat=UnsatisfiedRows(reason="evidence-absent")),
        "evidence-matrix-mismatch": _classify([tagged], unsat=UnsatisfiedRows(reason="evidence-matrix-mismatch")),
        "no-unsatisfied-rows": _classify([tagged], unsat=UnsatisfiedRows()),
    }
    for name, snap in cases.items():
        assert not snap.qualifies, name
        expected = {
            "merge-conflict-obligation": "machine-obligation-open",
            "checks-stalled": "checks-failing",
        }.get(name, name)
        assert expected in snap.reasons, (name, snap.reasons)


def test_tag_on_verified_row_is_an_ordinary_finding():
    # row-b is verified, so it is absent from the unsatisfied set.
    snap = _classify([_item(tags=["row-a", "row-b"])])
    assert not snap.qualifies and "tag-names-satisfied-row" in snap.reasons


def test_same_pr_untagged_item_blocks_and_future_item_is_ignored():
    assert "untagged-finding" in _classify([_item(tags=["row-a"]), _item("item-2", status="same-pr")]).reasons
    assert _classify([_item(tags=["row-a"]), _item("item-3", status="future")]).qualifies


# --- stall_window -----------------------------------------------------------

def _snap(review_round, *, qualifies=True, rows=("row-a",), identity=IDENTITY):
    return StallRoundSnapshot(review_round, HEAD, identity, qualifies, rows, () if qualifies else ("untagged-finding",))


def _prior(*snaps):
    return {s.review_round: s.to_payload() for s in snaps}


def test_window_stops_at_k_and_not_before():
    current = _snap(2)
    assert stall_window(current, _prior(_snap(1)), 2).stop
    assert not stall_window(current, {}, 2).stop
    assert not stall_window(_snap(3), _prior(_snap(2)), 3).stop
    assert stall_window(_snap(3), _prior(_snap(2), _snap(1)), 3).stop


def test_window_resets_on_non_qualifying_missing_invalid_or_changed():
    current = _snap(3)
    assert not stall_window(current, _prior(_snap(2, qualifies=False)), 2).stop
    assert not stall_window(current, {2: None}, 2).stop
    assert not stall_window(current, {2: {"invalid": True}}, 2).stop
    assert not stall_window(current, _prior(_snap(2, rows=("row-b",))), 2).stop
    assert not stall_window(current, _prior(_snap(2, identity=OTHER_IDENTITY)), 2).stop
    # A snapshot filed under the wrong round does not count.
    assert not stall_window(current, {2: _snap(7).to_payload()}, 2).stop


def test_window_reset_by_code_finding_round_then_one_evidence_round():
    prior = _prior(_snap(1), _snap(2, qualifies=False))
    assert not stall_window(_snap(3), prior, 2).stop


def test_window_k_zero_never_stops_and_non_qualifying_current_never_stops():
    assert not stall_window(_snap(5), _prior(_snap(4), _snap(3)), 0).stop
    assert not stall_window(_snap(2, qualifies=False), _prior(_snap(1)), 2).stop


def test_approved_matrix_row_ids_none_without_matrix():
    class Ctx:
        matrix_available = False
        risk_test_matrix_expected_row_ids = ("row-a",)

    assert approved_matrix_row_ids(None) is None
    assert approved_matrix_row_ids(Ctx()) is None
    Ctx.matrix_available = True
    assert approved_matrix_row_ids(Ctx()) == ("row-a",)


def _payload(**overrides):
    payload = {
        "review_round": 2, "review_head": "h" * 40, "matrix_identity": IDENTITY,
        "qualifies": True, "unsatisfied_row_ids": ["row-a"], "reasons": [],
    }
    payload.update(overrides)
    return payload


import pytest  # noqa: E402

from coding_review_agent_loop.evidence_stall import stall_snapshot_is_consistent  # noqa: E402


@pytest.mark.parametrize(
    ("overrides", "consistent"),
    [
        ({}, True),
        ({"qualifies": False, "reasons": ["untagged-finding"]}, True),
        ({"qualifies": False, "reasons": ["checks-failing", "machine-obligation-open"], "unsatisfied_row_ids": []}, True),
        ({"reasons": ["checks-failing"]}, False),  # qualifies with a reason
        ({"qualifies": False, "reasons": []}, False),  # non-qualifying without a reason
        ({"qualifies": False, "reasons": ["not-a-reason"]}, False),
        ({"qualifies": False, "reasons": ["checks-failing", "checks-failing"]}, False),
        ({"unsatisfied_row_ids": ["row-a", "row-a"]}, False),
        ({"unsatisfied_row_ids": [""]}, False),
        ({"unsatisfied_row_ids": []}, False),  # qualifying without rows
        ({"review_head": ""}, False),
        ({"matrix_identity": " "}, False),
        ({"qualifies": "true"}, False),
        ({"reasons": "checks-failing"}, False),
    ],
)
def test_stall_snapshot_consistency_table(overrides, consistent):
    """`stall-window-reset` / `stall-legacy-compat`: only consistent history can count toward K."""
    assert stall_snapshot_is_consistent(_payload(**overrides)) is consistent
