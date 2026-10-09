from coding_review_agent_loop.evidence_reconciliation import (
    MAX_RENDERED_EVIDENCE_ENTRIES,
    reconcile_evidence,
)
from coding_review_agent_loop.protocol import (
    DiscussEvidenceClaim,
    DiscussEvidenceUpdate,
    ParsedDiscussReview,
    parse_structured_discuss_review,
)


def _vote(reviewer, claims=(), updates=()):
    return ParsedDiscussReview(
        outcome="implement", rationale="test", split_proposals=(), reviewer=reviewer,
        evidence_claims=tuple(claims), evidence_updates=tuple(updates),
    )


def test_reconciliation_retracts_old_claim_and_combines_exact_contributors():
    subject = "issue-535"
    first = _vote("Codex", [DiscussEvidenceClaim("Same fact", "reported-but-unverified", "https://x")])
    duplicate = _vote("Gemini", [DiscussEvidenceClaim(" same   fact ", "reported-but-unverified", "https://x")])
    retraction = _vote("Codex", [], [DiscussEvidenceUpdate("retract", "issue-535-r1-Codex-c0", "later inspection disproved it")])
    ledger = reconcile_evidence(subject, [[first, duplicate], [retraction]])
    assert len(ledger["entries"]) == 1
    assert ledger["entries"][0]["contributors"] == ["Gemini"]
    assert ledger["history"][0]["action"] == "retract"


def test_reconciliation_combines_semantic_paraphrases_and_contributors():
    subject = "issue-535"
    first = _vote("Codex", [DiscussEvidenceClaim(
        "The renderer adds each sourced fact as a separate ledger entry.",
        "reported-but-unverified", "src/comment_rendering.py:130",
    )])
    paraphrase = _vote("Antigravity", [DiscussEvidenceClaim(
        "Every sourced fact is shown in its own final evidence ledger item.",
        "reported-but-unverified", "src/comment_rendering.py:131",
    )])
    ledger = reconcile_evidence(
        subject,
        [[first, paraphrase]],
        semantic_groups=[
            ("issue-535-r1-Codex-c0", "issue-535-r1-Antigravity-c0"),
        ],
    )

    assert len(ledger["entries"]) == 1
    entry = ledger["entries"][0]
    assert entry["contributors"] == ["Codex", "Antigravity"]
    assert entry["ids"] == ["issue-535-r1-Codex-c0", "issue-535-r1-Antigravity-c0"]


def test_reconciliation_keeps_reported_separate_from_missing_and_bounds_output():
    claims = [DiscussEvidenceClaim(f"fact {index}", "reported-but-unverified") for index in range(60)]
    claims.append(DiscussEvidenceClaim("implementation assertion", "missing"))
    ledger = reconcile_evidence("issue-535", [[_vote("Codex", claims)]])
    assert any(item["status"] == "missing" for item in ledger["entries"])
    assert len(ledger["rendered"]) <= MAX_RENDERED_EVIDENCE_ENTRIES
    assert ledger["omitted_entries"] > 0


def test_protocol_accepts_verified_attestation_and_rejects_missing_citation():
    good = '''{"schema_version":1,"kind":"discuss_review","outcome":"implement","rationale":"r","evidence":{"claims":[{"fact":"checked","status":"verified","source":"src/x.py:4","verification_basis":"checkout-inspected"}],"updates":[]}}
<!-- AGENT_PLAN_STATE: approved -->
-- Codex'''
    parsed = parse_structured_discuss_review(good, reviewer="Codex")
    assert parsed and parsed.evidence_claims[0].status == "verified"
    bad = good.replace('"status":"verified","source":"src/x.py:4","verification_basis":"checkout-inspected"', '"status":"missing","source":"src/x.py:4"')
    try:
        parse_structured_discuss_review(bad, reviewer="Codex")
    except Exception as exc:
        assert "missing evidence claims" in str(exc)
    else:
        raise AssertionError("missing evidence with a citation was accepted")


# --- #1329: obsolete-tree failures through reconcile and the protocol gate ------

import sys
from dataclasses import replace

import pytest

from coding_review_agent_loop.local_test_evidence import (
    TREE_CHANGE_SUPERSESSION,
    EnvironmentIdentityRegistry,
    EvidenceScope,
    LocalTestObservation,
    TrackedTreeSnapshot,
    TreeAttribution,
    reconcile_test_observations,
)
import coding_review_agent_loop.local_test_evidence as evidence_module
from coding_review_agent_loop.protocol import (
    SemanticRiskCoverageClaim,
    SemanticRiskCoverageClaims,
    _receipt_expected_status,
    derive_risk_test_matrix_evidence,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)

_SNAPSHOT_B = TrackedTreeSnapshot(
    root="/checkout", head="head-b", digest="all", tracked_digest="tree-b",
    status_clean=True, complete=True, stable=True,
)


def _journal_row(registry, *, receipt, outcome, digest, head, minute, state="current-head",
                 claim=None):
    return LocalTestObservation(
        command=(sys.executable, "-m", "pytest", "tests/test_protocol.py", "-q"),
        outcome=outcome,
        provenance="parent-observed",
        scope=EvidenceScope("suite", ("tests/test_protocol.py",)),
        receipt_id=receipt,
        execution_ref=f"turn-current:{receipt}",
        turn_id="turn-current",
        timestamp=f"2026-10-09T10:{minute:02d}:00+00:00",
        cwd="/checkout",
        normalized_command="python -m pytest tests/test_protocol.py -q",
        returncode=0 if outcome == "passed" else 1,
        attribution=TreeAttribution(state=state, head=head, tracked_digest=digest, stable=True),
        environment_state="not-compared",
        environment_identity=registry.capture({"PATH": "/usr/bin"}),
        claim=claim,
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )


def _matrix():
    return parse_risk_test_matrix({
        "applicability": "applicable",
        "rows": [{
            "row_id": "row-gate",
            "label": "Gate",
            "entry_path_or_mode": "agent-loop pr",
            "initial_state": "journal",
            "event": "reviewer validation",
            "expected_outcome": "row verifies",
            "forbidden_side_effects": ["no hidden failure"],
            "proposed_test_level": "integration",
            "proposed_test_location": "tests/test_evidence_reconciliation.py",
            "applicability": "applicable",
            "related_scope_item_ids": ["scope-1"],
            "execution_owner": "one-shot",
        }],
        "important_exclusions": [],
    })


def _derive(observations):
    matrix = _matrix()
    claims = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-gate",
        execution_refs=("turn-current:pass-b",),
        test_identifiers=("tests/test_protocol.py::test_gate",),
        test_locations=("tests/test_protocol.py",),
        workflow_path_claim="The reviewer validation path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No failure was hidden.",),
    ),))
    return derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claims,
        observations=observations,
        execution_catalog=observations,
        invocation_id="turn-current",
        current_head="head-b",
        current_tree_digest="tree-b",
        authenticated_checkout_head="head-b",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )


@pytest.mark.parametrize("failure_digest", ["tree-a", "tree-b"])
def test_protocol_gate_ignores_only_historical_journal_failures(failure_digest):
    registry = EnvironmentIdentityRegistry()
    failure = _journal_row(
        registry, receipt="fail", outcome="failed", digest=failure_digest,
        head="head-a" if failure_digest == "tree-a" else "head-b", minute=0,
    )
    passing = _journal_row(
        registry, receipt="pass-b", outcome="passed", digest="tree-b", head="head-b", minute=5,
    )
    # A different-scope pass cannot supersede the current-digest failure.
    passing = replace(passing, scope=EvidenceScope("suite", ("tests/test_protocol.py::test_gate",)))

    reconciled = reconcile_test_observations(
        [failure, passing], current_head="head-b", current_snapshot=_SNAPSHOT_B,
        registry=registry,
    )
    result = _derive(reconciled.observations)

    codes = {diagnostic.code for diagnostic in result.diagnostics}
    if failure_digest == "tree-a":
        assert reconciled.observations[0].superseded_by == TREE_CHANGE_SUPERSESSION
        assert "unsuperseded-journal-failure" not in codes
        assert result.evidence.rows[0].status == "verified"
    else:
        assert reconciled.observations[0].superseded_by is None
        assert "unsuperseded-journal-failure" in codes
        assert result.evidence.rows[0].status == "incomplete"


def test_base_reproduction_citation_authority_is_unchanged(monkeypatch):
    registry = EnvironmentIdentityRegistry()
    rows = [
        # Base head differs from the current head.
        _journal_row(registry, receipt="base-other-head", outcome="passed", digest="tree-a",
                     head="head-a", minute=0, state="base-reproduction"),
        # Tracked digest equals the current digest.
        _journal_row(registry, receipt="base-current-digest", outcome="passed", digest="tree-b",
                     head="head-b", minute=1, state="base-reproduction"),
        _journal_row(registry, receipt="base-claim-failure", outcome="failed", digest="tree-a",
                     head="head-a", minute=2, claim="base-reproduction"),
        _journal_row(registry, receipt="base-state-failure", outcome="failed", digest="tree-a",
                     head="head-a", minute=3, state="base-reproduction"),
    ]
    kwargs = dict(current_head="head-b", current_snapshot=_SNAPSHOT_B, registry=registry)

    now = reconcile_test_observations(rows, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(evidence_module, "_classify_tree_change_failures", lambda *a, **k: None)
        today = reconcile_test_observations(rows, **kwargs)

    assert now.observations == today.observations
    assert not any(row.superseded_by for row in now.observations)
    for current, previous in zip(now.observations, today.observations):
        for claim in ("base-reproduction", "current-result"):
            assert _receipt_expected_status(current, claim=claim) == _receipt_expected_status(
                previous, claim=claim
            )


# --- #1329: the authenticated response-validation seam --------------------------

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import coding_review_agent_loop.response_validation as validation_module
from coding_review_agent_loop.protocol import validate_structured_coder_followup
from coding_review_agent_loop.round_state import make_approved_plan_context
from agent_loop_helpers import structured_coder_followup


def _seam_row(registry, workdir, *, receipt, outcome, digest, head, minute,
              state="current-head", claim=None):
    return replace(
        _journal_row(registry, receipt=receipt, outcome=outcome, digest=digest, head=head,
                     minute=minute, state=state, claim=claim),
        cwd=str(workdir),
    )


def _restored_failure(receipt, *, state="current-head", claim=None):
    row = {
        "command": ["python", "-m", "pytest", "tests/test_protocol.py", "-q"],
        "outcome": "failed", "provenance": "parent-observed", "receipt_id": receipt,
        "turn_id": "turn-current", "timestamp": "2026-10-09T09:59:00+00:00",
        "attribution": {"state": state, "head": "head-a", "stable": True,
                        "tracked_digest": "tree-a"},
    }
    if claim is not None:
        row["claim"] = claim
    return row


def _validate_at_seam(monkeypatch, workdir, journal, *, selected, disable_classification=False):
    """Run the real post-auth validation seam: snapshot, admissibility, both reconciles."""
    matrix = _matrix()
    identity = risk_test_matrix_identity(matrix)
    plan_context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    parsed = validate_structured_coder_followup(structured_coder_followup())
    parsed = dataclasses.replace(parsed, risk_test_matrix_claims=SemanticRiskCoverageClaims((
        SemanticRiskCoverageClaim(
            row_id="row-gate",
            execution_refs=tuple(f"turn-current:{receipt}" for receipt in selected),
            test_identifiers=("tests/test_protocol.py::test_gate",),
            test_locations=("tests/test_protocol.py",),
            workflow_path_claim="The reviewer validation path ran.",
            outcome_assertions=("The selected test passed.",),
            forbidden_effect_assertions=("No failure was hidden.",),
        ),
    )))
    snapshot = TrackedTreeSnapshot(
        root=str(workdir), head="head-b", digest="all", tracked_digest="tree-b",
        status_clean=True, complete=True, stable=True,
    )
    live = tuple(row for row in journal if isinstance(row, LocalTestObservation))
    runner = SimpleNamespace(
        latest_test_turn_id="turn-current",
        current_test_turn_observations=lambda: live,
        local_test_observations=lambda: tuple(journal),
    )
    reconciled = []

    def capture(*args, **kwargs):
        evidence = reconcile_test_observations(*args, **kwargs)
        reconciled.append(evidence)
        return evidence

    with monkeypatch.context() as patch:
        patch.setattr(validation_module, "stable_tracked_tree_snapshot", lambda _cwd: snapshot)
        patch.setattr(validation_module, "reconcile_test_observations", capture)
        if disable_classification:
            patch.setattr(
                evidence_module, "_classify_tree_change_failures", lambda *a, **k: None
            )
        derived, result = validation_module._derive_authenticated_risk_evidence_for_coder(
            parsed,
            approved_plan_context=plan_context,
            runner=runner,
            assigned_workdir=Path(workdir),
            head_sha="head-b",
            invocation_id="turn-current",
            assigned_worktree_head="head-b",
        )
    assert result is not None
    # The seam reconciles the journal first, then the selectable catalog.
    journal_evidence = reconciled[0]
    return (
        derived.risk_test_matrix_evidence.rows[0],
        {d.code for d in result.diagnostics},
        journal_evidence,
    )


@pytest.mark.parametrize("failure_kind", ["obsolete-live", "obsolete-restored", "current-digest"])
def test_validation_seam_journal_gate_ignores_only_historical_failures(
    monkeypatch, tmp_path, failure_kind
):
    registry = EnvironmentIdentityRegistry()
    passing = replace(
        _seam_row(registry, tmp_path, receipt="pass-b", outcome="passed", digest="tree-b",
                  head="head-b", minute=5),
        scope=EvidenceScope("suite", ("tests/test_protocol.py::test_gate",)),
    )
    if failure_kind == "obsolete-live":
        failure = _seam_row(registry, tmp_path, receipt="fail", outcome="failed",
                            digest="tree-a", head="head-a", minute=0)
    elif failure_kind == "obsolete-restored":
        failure = _restored_failure("fail")
    else:
        failure = _seam_row(registry, tmp_path, receipt="fail", outcome="failed",
                            digest="tree-b", head="head-b", minute=0)

    row, codes, journal_evidence = _validate_at_seam(
        monkeypatch, tmp_path, [failure, passing], selected=["pass-b"]
    )
    labels = {item.receipt_id: item.superseded_by for item in journal_evidence.observations}
    assert labels["fail"] == (None if failure_kind == "current-digest" else TREE_CHANGE_SUPERSESSION)

    if failure_kind == "current-digest":
        assert "unsuperseded-journal-failure" in codes
        assert row.status == "incomplete"
    else:
        assert "unsuperseded-journal-failure" not in codes
        assert row.status == "verified"


@pytest.mark.parametrize("selected", ["base-other-head", "base-current-digest"])
def test_validation_seam_base_reproduction_status_is_unchanged(monkeypatch, tmp_path, selected):
    registry = EnvironmentIdentityRegistry()
    journal = [
        _seam_row(registry, tmp_path, receipt="base-other-head", outcome="passed",
                  digest="tree-a", head="head-a", minute=0, state="base-reproduction"),
        _seam_row(registry, tmp_path, receipt="base-current-digest", outcome="passed",
                  digest="tree-b", head="head-b", minute=1, state="base-reproduction"),
        # Live and restored base failures, by state and by claim.
        _seam_row(registry, tmp_path, receipt="base-live-state", outcome="failed",
                  digest="tree-a", head="head-a", minute=2, state="base-reproduction"),
        _seam_row(registry, tmp_path, receipt="base-live-claim", outcome="failed",
                  digest="tree-a", head="head-a", minute=3, claim="base-reproduction"),
        _restored_failure("base-restored-state", state="base-reproduction"),
        _restored_failure("base-restored-claim", claim="base-reproduction"),
    ]

    now_row, now_codes, now_journal = _validate_at_seam(
        monkeypatch, tmp_path, journal, selected=[selected]
    )
    today_row, today_codes, today_journal = _validate_at_seam(
        monkeypatch, tmp_path, journal, selected=[selected], disable_classification=True
    )

    assert now_row == today_row
    assert now_codes == today_codes
    assert now_row.status != "verified"
    assert now_journal.observations == today_journal.observations
    assert not any(item.superseded_by for item in now_journal.observations)
    assert set(now_journal.authoritative_failures) == {
        "base-live-state", "base-live-claim", "base-restored-state", "base-restored-claim",
    }


def _restored_base_pass(receipt, *, digest, head):
    return {
        "command": ["python", "-m", "pytest", "tests/test_protocol.py", "-q"],
        "normalized_command": "python -m pytest tests/test_protocol.py -q",
        "outcome": "passed", "provenance": "parent-observed", "receipt_id": receipt,
        "turn_id": "turn-current", "timestamp": "2026-10-09T09:58:00+00:00",
        "claim": "base-reproduction",
        "wrapper_bootstrap": "verified", "inner_exec": "started", "suite_start": "verified",
        "attribution": {"state": "base-reproduction", "head": head, "stable": True,
                        "tracked_digest": digest},
    }


def _base_citation_outcome(observations, receipt, status):
    """Validate a ``base-reproduction`` citation the way published evidence is checked."""
    from coding_review_agent_loop.protocol import parse_risk_test_matrix_evidence

    matrix = _matrix()
    payload = {
        "matrix_identity": risk_test_matrix_identity(matrix),
        "rows": [{
            "row_id": "row-gate",
            "status": status,
            "test_identifiers": ["tests/test_protocol.py::test_gate"],
            "test_locations": ["tests/test_protocol.py"],
            "workflow_path_claim": "Base reproduction ran.",
            "outcome_assertions": ["The base reproduction passed."],
            "forbidden_effect_assertions": ["No failure was hidden."],
            "evidence_citations": [{
                "command": "python -m pytest tests/test_protocol.py -q",
                "receipt_id": receipt,
                "claim": "base-reproduction",
            }],
        }],
    }
    try:
        parsed = parse_risk_test_matrix_evidence(
            payload, matrix=matrix, authoritative_test_observations=observations,
        )
    except Exception as exc:  # the rejection text is the citation outcome
        return ("rejected", str(exc))
    return ("accepted", parsed.rows[0].status, parsed.rows[0].caveats)


_BASE_PASSES = ("live-base-other-head", "live-base-current-digest",
                "restored-base-other-head", "restored-base-current-digest")


def test_validation_seam_base_reproduction_citations_are_unchanged(monkeypatch, tmp_path):
    registry = EnvironmentIdentityRegistry()
    journal = [
        _seam_row(registry, tmp_path, receipt="live-base-other-head", outcome="passed",
                  digest="tree-a", head="head-a", minute=0, state="base-reproduction",
                  claim="base-reproduction"),
        _seam_row(registry, tmp_path, receipt="live-base-current-digest", outcome="passed",
                  digest="tree-b", head="head-b", minute=1, state="base-reproduction",
                  claim="base-reproduction"),
        _restored_base_pass("restored-base-other-head", digest="tree-a", head="head-a"),
        _restored_base_pass("restored-base-current-digest", digest="tree-b", head="head-b"),
        _seam_row(registry, tmp_path, receipt="base-live-claim", outcome="failed",
                  digest="tree-a", head="head-a", minute=3, claim="base-reproduction"),
        _restored_failure("base-restored-state", state="base-reproduction"),
    ]

    _row, _codes, now_journal = _validate_at_seam(
        monkeypatch, tmp_path, journal, selected=["live-base-other-head"]
    )
    _row, _codes, today_journal = _validate_at_seam(
        monkeypatch, tmp_path, journal, selected=["live-base-other-head"],
        disable_classification=True,
    )

    assert {row.receipt_id for row in now_journal.observations} >= set(_BASE_PASSES)
    assert not any(row.superseded_by for row in now_journal.observations)
    assert set(now_journal.authoritative_failures) == {"base-live-claim", "base-restored-state"}
    for receipt in _BASE_PASSES:
        for status in ("verified", "stale/unverified"):
            now = _base_citation_outcome(now_journal.observations, receipt, status)
            today = _base_citation_outcome(today_journal.observations, receipt, status)
            assert now == today, (receipt, status)
        # The citation genuinely matches its receipt (not a vacuous rejection).
        assert _base_citation_outcome(
            now_journal.observations, receipt, "stale/unverified"
        )[:2] == ("accepted", "stale/unverified")
        # No base pass becomes verified at an obsolete or rewritten attribution.
        assert _base_citation_outcome(now_journal.observations, receipt, "verified")[0] == "rejected"
