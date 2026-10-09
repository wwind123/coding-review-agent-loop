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
