from __future__ import annotations

import json
import hashlib
import dataclasses
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.comment_rendering import (
    decode_risk_test_matrix_marker,
    render_canonical_plan_state,
    render_public_agent_comment,
    render_risk_test_matrix_section,
)
import coding_review_agent_loop.orchestrator as orchestrator_module
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.local_test_evidence import LocalTestObservation, TreeAttribution
from coding_review_agent_loop.protocol import (
    RiskTestMatrixChange,
    SemanticRiskCoverageClaim,
    SemanticRiskCoverageClaims,
    derive_risk_test_matrix_evidence,
    parse_risk_test_matrix,
    parse_risk_test_matrix_evidence,
    risk_test_matrix_identity,
    validate_structured_coder_followup,
    validate_structured_plan_state,
    validate_risk_test_matrix_revision,
)
from coding_review_agent_loop.decomposition import validate_risk_matrix_ownership
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _decode_round_metadata,
    _encode_round_metadata,
    _attach_round_metadata,
    make_approved_plan_context,
    recover_approved_plan_context,
    scope_approved_plan_matrix,
)
from coding_review_agent_loop.round_transport import (
    prepare_round_comment,
    risk_test_matrix_section_boundary,
)
from agent_loop_helpers import structured_coder_followup


def _row(row_id: str = "row-ordinary") -> dict[str, object]:
    return {
        "row_id": row_id,
        "label": "Ordinary recovery reaches completion",
        "entry_path_or_mode": "ordinary / review-only",
        "initial_state": "review complete, CI obligation pending",
        "event": "repaired head passes",
        "expected_outcome": "Complete without requiring a watcher",
        "forbidden_side_effects": ["Do not merge a stale head"],
        "proposed_test_level": "orchestrator",
        "proposed_test_location": "tests/test_orchestrator_pr.py::test_review_only_recovery",
        "applicability": "applicable",
        "related_scope_item_ids": ["scope-matrix-evidence-review"],
        "execution_owner": "one-shot",
    }


def _matrix() -> dict[str, object]:
    return {
        "applicability": "applicable",
        "rows": [_row()],
        "important_exclusions": ["No Cartesian product of unrelated feature flags."],
    }


def _not_applicable() -> dict[str, object]:
    return {
        "applicability": "not-applicable",
        "rows": [],
        "important_exclusions": [],
        "not_applicable_rationale": "This is a local formatting-only change with no stateful entry path.",
    }


def _derived_observation(
    *,
    execution_ref: str,
    receipt_id: str,
    outcome: str = "passed",
    turn_id: str = "turn-current",
    command: tuple[str, ...] = ("python3", "-m", "pytest", "tests/test_protocol.py", "-q"),
) -> SimpleNamespace:
    return SimpleNamespace(
        execution_ref=execution_ref,
        receipt_id=receipt_id,
        command=command,
        normalized_command="python3 -m pytest tests/test_protocol.py -q",
        outcome=outcome,
        provenance="parent-observed",
        turn_id=turn_id,
        attribution={
            "state": "current-head",
            "head": "head-current",
            "tracked_digest": "tree-current",
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


def test_derived_matrix_evidence_is_complete_and_selector_citations_are_tool_owned() -> None:
    matrix = parse_risk_test_matrix({
        **_matrix(),
        "rows": [_row("row-first"), _row("row-second")],
    })
    observation = _derived_observation(execution_ref="invocation:observation-1", receipt_id="receipt-1")
    claims = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-first",
        execution_refs=("invocation:observation-1",),
        test_identifiers=("test_first",),
        test_locations=("tests/test_protocol.py::test_first",),
        workflow_path_claim="The first workflow path ran.",
        outcome_assertions=("The first test passed.",),
        forbidden_effect_assertions=("No unauthorized evidence was accepted.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claims,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert [row.row_id for row in result.evidence.rows] == ["row-first", "row-second"]
    assert result.evidence.rows[0].status == "verified"
    assert result.evidence.rows[0].evidence_citations[0].receipt_id == "receipt-1"
    assert result.evidence.rows[1].status == "missing"
    assert result.diagnostics[0].code == "missing-claim"
    assert "execution_ref" not in result.evidence.to_payload()["rows"][0]["evidence_citations"][0]


def test_one_shared_selector_verifies_every_row_it_covers() -> None:
    # Regression for #865: a single passing wrapper run may evidence several rows.
    matrix = parse_risk_test_matrix({
        **_matrix(),
        "rows": [_row("row-first"), _row("row-second")],
    })
    observation = _derived_observation(execution_ref="invocation:observation-1", receipt_id="receipt-1")
    claims = SemanticRiskCoverageClaims(tuple(
        SemanticRiskCoverageClaim(
            row_id=row_id,
            execution_refs=("invocation:observation-1",),
            test_identifiers=(f"test_{row_id}",),
            test_locations=(f"tests/test_protocol.py::test_{row_id}",),
            workflow_path_claim="The shared run covered this workflow path.",
            outcome_assertions=("The row's test passed.",),
            forbidden_effect_assertions=("No unauthorized evidence was accepted.",),
        )
        for row_id in ("row-first", "row-second")
    ))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claims,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert [row.status for row in result.evidence.rows] == ["verified", "verified"]
    assert [row.evidence_citations[0].receipt_id for row in result.evidence.rows] == [
        "receipt-1", "receipt-1"
    ]
    assert not result.diagnostics


@pytest.mark.parametrize(
    "field",
    [
        "test_identifiers",
        "test_locations",
        "outcome_assertions",
        "forbidden_effect_assertions",
    ],
)
def test_builder_downgrades_incomplete_semantic_facts(field) -> None:
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="invocation:observation-1", receipt_id="receipt-1"
    )
    values = {
        "test_identifiers": ("test_ordinary",),
        "test_locations": ("tests/test_risk_test_matrix.py::test_ordinary",),
        "workflow_path_claim": "The workflow path ran.",
        "outcome_assertions": ("The selected test passed.",),
        "forbidden_effect_assertions": ("No stale head was merged.",),
    }
    values[field] = ()
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        **values,
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert result.evidence.rows[0].status == "stale/unverified"
    assert not result.evidence.rows[0].evidence_citations
    assert any(diagnostic.code == "incomplete-semantic-claim" for diagnostic in result.diagnostics)


def _derive_for_claims(matrix, claims):
    observation = _derived_observation(
        execution_ref="invocation:observation-1", receipt_id="receipt-1"
    )
    return derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claims,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )


def test_builder_keeps_empty_facts_for_present_incomplete_claim() -> None:
    """#849: an all-empty present claim fails closed without copying approved-row text."""
    from coding_review_agent_loop.protocol import SEMANTIC_RISK_CLAIM_FACT_KEYS

    matrix = parse_risk_test_matrix(_matrix())
    approved = matrix.rows[0]
    claims = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        test_identifiers=(),
        test_locations=(),
        workflow_path_claim="",
        outcome_assertions=(),
        forbidden_effect_assertions=(),
    ),))

    result = _derive_for_claims(matrix, claims)

    row = result.evidence.rows[0]
    assert row.status != "verified"
    assert row.evidence_citations == ()
    assert row.test_identifiers == ()
    assert row.test_locations == ()
    assert row.workflow_path_claim == ""
    assert row.outcome_assertions == ()
    assert row.forbidden_effect_assertions == ()
    assert row.workflow_path_claim != approved.entry_path_or_mode
    assert approved.expected_outcome not in row.outcome_assertions
    assert not set(approved.forbidden_side_effects) & set(row.forbidden_effect_assertions)
    incomplete = [d for d in result.diagnostics if d.code == "incomplete-semantic-claim"]
    assert len(incomplete) == 1
    assert incomplete[0].message == (
        "Semantic coverage is missing required facts: "
        + ", ".join(SEMANTIC_RISK_CLAIM_FACT_KEYS)
        + "."
    )


def test_builder_absent_claim_keeps_missing_status_and_approved_fallback() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    approved = matrix.rows[0]

    result = _derive_for_claims(matrix, None)

    row = result.evidence.rows[0]
    assert row.status == "missing"
    assert row.workflow_path_claim == approved.entry_path_or_mode
    assert row.outcome_assertions == (approved.expected_outcome,)
    assert row.forbidden_effect_assertions == approved.forbidden_side_effects
    assert [d.code for d in result.diagnostics] == ["missing-claim"]


def test_derived_matrix_evidence_preserves_unsuperseded_failure_caveat() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    passing = _derived_observation(execution_ref="invocation:observation-1", receipt_id="receipt-pass")
    failed = _derived_observation(
        execution_ref="invocation:observation-2",
        receipt_id="receipt-fail",
        outcome="failed",
    )
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_protocol.py::test_ordinary",),
        workflow_path_claim="The workflow path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No stale head was merged.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        observations=(passing, failed),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    row = result.evidence.rows[0]
    assert row.status == "incomplete"
    assert any("unsuperseded" in caveat for caveat in row.caveats)
    assert any(diagnostic.code == "unsuperseded-journal-failure" for diagnostic in result.diagnostics)


def test_orchestrator_derivation_ignores_prior_turn_failure(monkeypatch, tmp_path) -> None:
    matrix = parse_risk_test_matrix(_matrix())
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
    prior_failure = _derived_observation(
        execution_ref="old-turn:observation-1",
        receipt_id="receipt-old-failure",
        outcome="failed",
        turn_id="turn-old",
    )
    current_pass = _derived_observation(
        execution_ref="current-turn:observation-1",
        receipt_id="receipt-current-pass",
        turn_id="turn-current",
    )
    parsed = validate_structured_coder_followup(structured_coder_followup())
    parsed = dataclasses.replace(parsed, risk_test_matrix_claims=SemanticRiskCoverageClaims((
        SemanticRiskCoverageClaim(
            row_id="row-ordinary",
            execution_refs=("current-turn:observation-1",),
            test_identifiers=("test_ordinary",),
            test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
            workflow_path_claim="The current coder turn ran the workflow.",
            outcome_assertions=("The selected test passed.",),
            forbidden_effect_assertions=("No stale head was merged.",),
        ),
    )))

    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _cwd: SimpleNamespace(
            head="head-current", tracked_digest="tree-current", complete=True,
            stable=True, status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    runner = SimpleNamespace(
        local_test_observations=lambda: (prior_failure, current_pass),
    )

    derived, result = orchestrator_module._derive_authenticated_risk_evidence_for_coder(
        parsed,
        approved_plan_context=plan_context,
        runner=runner,
        assigned_workdir=tmp_path,
        head_sha="head-current",
        invocation_id="turn-current",
        _closed_execution_catalog=(current_pass,),
    )

    assert result is not None
    assert derived.risk_test_matrix_evidence.rows[0].status == "verified"
    assert not any(
        diagnostic.code == "unsuperseded-journal-failure"
        for diagnostic in result.diagnostics
    )


@pytest.mark.parametrize("outside_first", [True, False])
def test_orchestrator_derivation_excludes_out_of_checkout_broker_runs(
    monkeypatch, tmp_path, outside_first
) -> None:
    """Issue #991: a passing broker run of an outside test path is never evidence."""
    matrix = parse_risk_test_matrix(_matrix())
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
    baseline = _derived_observation(
        execution_ref="current-turn:observation-1",
        receipt_id="receipt-baseline",
        command=("python3", "-m", "pytest", "/tmp/scratch-main-991/tests/", "-q"),
    )
    failing_baseline = _derived_observation(
        execution_ref="current-turn:observation-2",
        receipt_id="receipt-baseline-failure",
        outcome="failed",
        command=("python3", "-m", "pytest", "/tmp/scratch-main-991/tests/", "-q"),
    )
    in_checkout = _derived_observation(
        execution_ref="current-turn:observation-3",
        receipt_id="receipt-in-checkout",
    )
    parsed = validate_structured_coder_followup(structured_coder_followup())
    parsed = dataclasses.replace(parsed, risk_test_matrix_claims=SemanticRiskCoverageClaims((
        SemanticRiskCoverageClaim(
            row_id="row-ordinary",
            execution_refs=(
                "current-turn:observation-1" if outside_first else "current-turn:observation-3",
            ),
            test_identifiers=("test_ordinary",),
            test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
            workflow_path_claim="The current coder turn ran the workflow.",
            outcome_assertions=("The selected test passed.",),
            forbidden_effect_assertions=("No stale head was merged.",),
        ),
    )))
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _cwd: SimpleNamespace(
            head="head-current", tracked_digest="tree-current", complete=True,
            stable=True, status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    catalog = (baseline, failing_baseline, in_checkout)
    runner = SimpleNamespace(local_test_observations=lambda: catalog)

    derived, result = orchestrator_module._derive_authenticated_risk_evidence_for_coder(
        parsed,
        approved_plan_context=plan_context,
        runner=runner,
        assigned_workdir=tmp_path,
        head_sha="head-current",
        invocation_id="turn-current",
        _closed_execution_catalog=catalog,
    )

    assert result is not None
    row = derived.risk_test_matrix_evidence.rows[0]
    if outside_first:
        assert row.status != "verified"
        assert not any(
            "receipt-baseline" in str(citation) for citation in row.evidence_citations
        )
    else:
        # The failing outside baseline is context, so it neither blocks the
        # in-checkout run nor surfaces as an unsuperseded journal failure.
        assert row.status == "verified"
        assert not any(
            diagnostic.code == "unsuperseded-journal-failure"
            for diagnostic in result.diagnostics
        )


@pytest.mark.parametrize("failure_command", [
    ("pytest", "tests/", "https://live.example"),
    ("python3", "-m", "pytest", "tests/test_protocol.py", "/tmp/scratch-main-991/tests/"),
    ("python3", "-m", "pytest", "tests", "/tmp/scratch-main-991/tests"),
    ("pytest", "--rootdir=/tmp/scratch-main-991", "tests/test_protocol.py"),
    ("python3", "-m", "pytest", "-q", "--junit-xml", "/tmp/scratch-main-991/r.xml"),
])
def test_orchestrator_derivation_keeps_unvalidatable_failures_in_journal(
    monkeypatch, tmp_path, failure_command
) -> None:
    """Issue #991: only pure out-of-checkout runs leave the failure journal.

    Unvalidatable runs and mixed runs that also name in-checkout tests keep
    degrading rows, though neither is ever a selectable execution ref.
    """
    matrix = parse_risk_test_matrix(_matrix())
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
    unvalidatable_failure = _derived_observation(
        execution_ref="current-turn:observation-1",
        receipt_id="receipt-url-failure",
        outcome="failed",
        command=failure_command,
    )
    in_checkout_pass = _derived_observation(
        execution_ref="current-turn:observation-2",
        receipt_id="receipt-in-checkout",
    )
    parsed = validate_structured_coder_followup(structured_coder_followup())
    parsed = dataclasses.replace(parsed, risk_test_matrix_claims=SemanticRiskCoverageClaims((
        SemanticRiskCoverageClaim(
            row_id="row-ordinary",
            execution_refs=("current-turn:observation-2",),
            test_identifiers=("test_ordinary",),
            test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
            workflow_path_claim="The current coder turn ran the workflow.",
            outcome_assertions=("The selected test passed.",),
            forbidden_effect_assertions=("No stale head was merged.",),
        ),
    )))
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _cwd: SimpleNamespace(
            head="head-current", tracked_digest="tree-current", complete=True,
            stable=True, status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    catalog = (unvalidatable_failure, in_checkout_pass)
    runner = SimpleNamespace(local_test_observations=lambda: catalog)

    derived, result = orchestrator_module._derive_authenticated_risk_evidence_for_coder(
        parsed,
        approved_plan_context=plan_context,
        runner=runner,
        assigned_workdir=tmp_path,
        head_sha="head-current",
        invocation_id="turn-current",
        _closed_execution_catalog=catalog,
    )

    assert result is not None
    assert derived.risk_test_matrix_evidence.rows[0].status == "incomplete"
    assert any(
        diagnostic.code == "unsuperseded-journal-failure" for diagnostic in result.diagnostics
    )


def test_derived_matrix_evidence_rejects_a_matching_tree_from_the_wrong_checkout_head() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="invocation:observation-1",
        receipt_id="receipt-1",
    )
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_protocol.py::test_ordinary",),
        workflow_path_claim="The workflow path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No stale head was merged.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-other",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert result.evidence.rows[0].status == "stale/unverified"
    assert any(diagnostic.code == "checkout-head-mismatch" for diagnostic in result.diagnostics)
    assert result.evidence.rows[0].evidence_citations == ()


def test_derived_matrix_evidence_requires_explicit_post_authentication_proof() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="invocation:observation-1",
        receipt_id="receipt-1",
    )
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
        workflow_path_claim="The workflow path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No stale head was merged.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert result.evidence.rows[0].status == "stale/unverified"
    assert any(diagnostic.code == "checkout-head-mismatch" for diagnostic in result.diagnostics)
    assert any(diagnostic.code == "checkout-tree-unavailable" for diagnostic in result.diagnostics)


def test_current_turn_catalog_filters_a_misbehaving_cumulative_provider() -> None:
    old = SimpleNamespace(turn_id="turn-old", execution_ref="old:observation-1")
    current = SimpleNamespace(turn_id="turn-current", execution_ref="current:observation-1")
    runner = SimpleNamespace(
        latest_test_turn_id="turn-current",
        current_test_turn_observations=lambda: (old, current),
    )

    assert orchestrator_module._current_test_turn_observations(runner) == (current,)


def test_derived_matrix_evidence_cannot_resolve_a_cross_turn_selector_from_the_journal() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    old_observation = _derived_observation(
        execution_ref="old-turn:observation-1",
        receipt_id="receipt-old",
        turn_id="turn-old",
    )
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("old-turn:observation-1",),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
        workflow_path_claim="The workflow path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No stale head was merged.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        # The bounded journal may retain this observation, but the current
        # closed catalog is empty, so its selector must remain unresolved.
        observations=(old_observation,),
        execution_catalog=(),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert result.evidence.rows[0].status == "stale/unverified"
    assert any(diagnostic.code == "unknown-execution-ref" for diagnostic in result.diagnostics)


@pytest.mark.parametrize("with_admissible", [False, True])
def test_derived_matrix_evidence_turns_dropped_selectors_into_unknown_ref_diagnostics(
    with_admissible: bool,
) -> None:
    """#859: parser-dropped refs derive a non-verified row with claim facts kept."""
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="turn-current:observation-1", receipt_id="receipt-1"
    )
    command = "python3 -m pytest tests/test_round_transport.py -q"
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("turn-current:observation-1",) if with_admissible else (),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
        workflow_path_claim="The claimed workflow path.",
        outcome_assertions=("The claimed outcome.",),
        forbidden_effect_assertions=("The claimed forbidden effect.",),
        caveats=("Dropped execution_refs ...",),
        dropped_execution_refs=(command, "x" * 2_000),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claim,
        observations=(observation,),
        execution_catalog=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    row = result.evidence.rows[0]
    assert row.status == "stale/unverified"
    assert row.evidence_citations == ()
    assert row.workflow_path_claim == "The claimed workflow path."
    assert row.outcome_assertions == ("The claimed outcome.",)
    assert row.forbidden_effect_assertions == ("The claimed forbidden effect.",)
    assert "A claimed execution selector was unknown or cross-turn." in row.caveats
    unknown = [d for d in result.diagnostics if d.code == "unknown-execution-ref"]
    assert len(unknown) == 2
    assert command in unknown[0].message
    assert len(unknown[1].message) < 300
    # Dropped refs are not selectors, so they never reach the duplicate check
    # or the orchestrator's non-actionable diagnostic set.
    assert all(d.code not in {"missing-claim", "unsuperseded-journal-failure"} for d in unknown)


def test_post_authentication_head_race_downgrades_correction_output(monkeypatch, tmp_path) -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    observation = LocalTestObservation(
        command=("python3", "-m", "pytest", "tests/test_protocol.py", "-q"),
        outcome="passed",
        provenance="parent-observed",
        receipt_id="receipt-1",
        execution_ref="turn-current:observation-1",
        turn_id="turn-current",
        normalized_command="python3 -m pytest tests/test_protocol.py -q",
        attribution=TreeAttribution(
            state="current-head",
            head="head-current",
            tracked_digest="tree-current",
            stable=True,
        ),
        environment_state="not-compared",
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )
    claim = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("turn-current:observation-1",),
        test_identifiers=("test_ordinary",),
        test_locations=("tests/test_risk_test_matrix.py::test_ordinary",),
        workflow_path_claim="The workflow path ran.",
        outcome_assertions=("The selected test passed.",),
        forbidden_effect_assertions=("No stale head was merged.",),
    ),))
    from coding_review_agent_loop.protocol import (
        StructuredHumanRequirementsPayload,
        StructuredIssueImplementation,
    )

    parsed = StructuredIssueImplementation(
        schema_version=1,
        kind="issue_implementation",
        state="blocking",
        summary="The implementation is complete.",
        pr_number=77,
        human_requirements=StructuredHumanRequirementsPayload((), False),
        human_requirement_dispositions=(),
        risk_test_matrix_claims=claim,
    )
    corrected_payload = {
        "schema_version": 1,
        "kind": "issue_implementation",
        "state": "blocking",
        "summary": "The implementation is complete.",
        "pr_number": 77,
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [],
        "risk_test_matrix_claims": [claim.claims[0].to_payload()],
    }

    class FakeRunner:
        latest_test_turn_id = "turn-current"

        def local_test_observations(self):
            return (observation,)

    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="head-other",
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_agent_result",
        lambda *_args, **_kwargs: SimpleNamespace(
            text=json.dumps(corrected_payload)
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ),
    )

    corrected, result = orchestrator_module._derive_authenticated_risk_evidence_for_coder(
        parsed,
        approved_plan_context=context,
        runner=FakeRunner(),
        assigned_workdir=tmp_path,
        head_sha="head-current",
        config=SimpleNamespace(coder="codex", coder_test_command_timeout_seconds=1),
        session_id="coder-session",
        reauthenticate_head=lambda: "head-new",
    )

    assert result is not None
    assert result.evidence.rows[0].status != "verified"
    assert result.evidence.rows[0].evidence_citations == ()
    assert any(
        diagnostic.code == "head-changed-during-correction"
        for diagnostic in result.diagnostics
    )
    assert corrected.risk_test_matrix_evidence == result.evidence


def test_m780_01_matrix_is_bounded_and_rendered_from_structured_payload() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    changes = (RiskTestMatrixChange("change", ("row-ordinary",), "Clarified the recovery outcome."),)
    section = render_risk_test_matrix_section(matrix, changes)
    assert "### Risk-based mode and transition test matrix" in section
    assert section.index("| ID |") < section.index("row-ordinary")
    assert "Clarified the recovery outcome." in section
    marker = section.rsplit("<!-- AGENT_RISK_TEST_MATRIX: ", 1)[1].split(" -->", 1)[0]
    decoded = decode_risk_test_matrix_marker(marker)
    assert decoded["identity"] == risk_test_matrix_identity(matrix, changes)
    assert risk_test_matrix_section_boundary(decoded["identity"]) in section


def test_m780_02_trivial_work_accepts_only_a_non_empty_not_applicable_rationale() -> None:
    matrix = parse_risk_test_matrix(_not_applicable())
    assert not matrix.is_applicable
    with pytest.raises(AgentLoopError, match="not_applicable_rationale|not-applicable matrices require"):
        parse_risk_test_matrix({**_not_applicable(), "not_applicable_rationale": "   "})
    with pytest.raises(AgentLoopError, match="at least one applicable or required row"):
        parse_risk_test_matrix({**_matrix(), "rows": [{**_row(), "applicability": "not-applicable"}]})


def test_m780_03_draft_changes_are_explicit_and_approved_rows_are_immutable() -> None:
    changed = {**_matrix(), "rows": [{**_row(), "expected_outcome": "A different outcome"}]}
    changes = [{"operation": "change", "row_ids": ["row-ordinary"], "rationale": "The state transition was clarified."}]
    validate_risk_test_matrix_revision(_matrix(), changed, changes)
    with pytest.raises(AgentLoopError, match="Approved risk matrix"):
        validate_risk_test_matrix_revision(_matrix(), changed, changes, approved=True)
    with pytest.raises(AgentLoopError, match="omit audit operations"):
        validate_risk_test_matrix_revision(_matrix(), changed, [])


@pytest.mark.parametrize(
    ("operation", "current", "message"),
    [
        (
            "add",
            {**_matrix(), "rows": [{**_row(), "expected_outcome": "A different outcome"}]},
            "newly introduced",
        ),
        (
            "change",
            {**_matrix(), "rows": [{**_row("row-new")}]},
            "retained",
        ),
        (
            "retire",
            {**_matrix(), "rows": [{**_row(), "expected_outcome": "A different outcome"}]},
            "removed",
        ),
    ],
)
def test_m780_03_audit_operation_must_match_actual_row_transition(
    operation: str, current: dict[str, object], message: str
) -> None:
    with pytest.raises(AgentLoopError, match=message):
        validate_risk_test_matrix_revision(
            _matrix(),
            current,
            [{"operation": operation, "row_ids": ["row-ordinary"], "rationale": "Mislabelled change."}],
        )


def test_m780_03_split_and_merge_require_removed_added_cardinality() -> None:
    split_current = {
        **_matrix(),
        "rows": [{**_row("row-new-one")}, {**_row("row-new-two")}],
    }
    with pytest.raises(AgentLoopError, match="exactly one prior row and at least two"):
        validate_risk_test_matrix_revision(
            _matrix(),
            split_current,
            [{
                "operation": "split",
                "row_ids": ["row-ordinary", "row-new-one"],
                "rationale": "Invalid split cardinality.",
            }],
        )

    merge_current = {
        **_matrix(),
        "rows": [{**_row("row-merged")}, {**_row("row-other-new")}],
    }
    previous = {**_matrix(), "rows": [_row("row-old-one"), _row("row-old-two")]}
    with pytest.raises(AgentLoopError, match="at least two prior rows and exactly one"):
        validate_risk_test_matrix_revision(
            previous,
            merge_current,
            [{
                "operation": "merge",
                "row_ids": ["row-old-one", "row-old-two", "row-merged", "row-other-new"],
                "rationale": "Invalid merge membership.",
            }],
        )


def test_m780_03_overlapping_row_audits_are_rejected() -> None:
    previous = {**_matrix(), "rows": [_row("row-old"), _row("row-other")]}
    current = {**_matrix(), "rows": [{**_row("row-new-one")}, {**_row("row-new-two")}]}
    with pytest.raises(AgentLoopError, match="overlap"):
        validate_risk_test_matrix_revision(
            previous,
            current,
            [
                {
                    "operation": "split",
                    "row_ids": ["row-old", "row-new-one", "row-new-two"],
                    "rationale": "Split the old scenario.",
                },
                {
                    "operation": "retire",
                    "row_ids": ["row-old"],
                    "rationale": "Overlapping audit subject.",
                },
            ],
        )


def test_approved_matrix_accepts_the_final_revision_change_audit() -> None:
    changes = [{
        "operation": "change",
        "row_ids": ["row-ordinary"],
        "rationale": "Clarified the post-review recovery transition.",
    }]
    parsed = validate_risk_test_matrix_revision(_matrix(), _matrix(), changes, approved=True)
    assert parsed[0].rationale == changes[0]["rationale"]


def test_draft_revision_ignores_an_exactly_replayed_historical_audit() -> None:
    historical = [{
        "operation": "change",
        "row_ids": ["row-ordinary"],
        "rationale": "Clarified the post-review recovery transition.",
    }]
    parsed = validate_risk_test_matrix_revision(
        _matrix(),
        _matrix(),
        historical,
        historical_changes=historical,
    )
    assert parsed[0].row_ids == ("row-ordinary",)


def test_authoritative_caveats_have_headroom_across_multiple_receipts() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    evidence["rows"][0]["evidence_citations"] = [
        {
            "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
            "receipt_id": f"receipt-{index}",
            "claim": "current-result",
        }
        for index in range(8)
    ]
    observations = [
        _rich_receipt(
            receipt_id=f"receipt-{index}",
            attribution_caveats=(f"authoritative head caveat {index}",),
            caveats=(f"broker caveat {index}",),
        )
        for index in range(8)
    ]

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=observations,
    )

    assert len(parsed.rows[0].caveats) == 16
    assert "broker caveat 7" in parsed.rows[0].caveats


def test_matrix_level_draft_changes_cannot_hide_behind_one_row_operation() -> None:
    previous = {**_matrix(), "rows": [_row("row-one"), _row("row-two")]}
    current = {**previous, "important_exclusions": ["A newly explicit exclusion."]}
    with pytest.raises(AgentLoopError, match="complete matrix scope") as exc_info:
        validate_risk_test_matrix_revision(
            previous,
            current,
            [{"operation": "change", "row_ids": ["row-one"], "rationale": "Clarified one row."}],
        )
    assert "Changed matrix fields: important_exclusions" in str(exc_info.value)
    assert "Required row_ids: [row-one, row-two]" in str(exc_info.value)
    validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": ["row-one", "row-two"],
            "rationale": "Clarified the matrix exclusions.",
        }],
    )


def test_matrix_level_audit_can_cover_the_full_matrix_row_bound() -> None:
    row_ids = [f"row-{index:02d}" for index in range(24)]
    previous = {**_matrix(), "rows": [_row(row_id) for row_id in row_ids]}
    current = {**previous, "important_exclusions": ["A newly explicit exclusion."]}

    parsed = validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": row_ids,
            "rationale": "Clarified exclusions across the complete matrix scope.",
        }],
    )

    assert parsed[0].row_ids == tuple(row_ids)


def test_matrix_level_audit_can_cover_a_full_replacement_transition_union() -> None:
    old_ids = [f"old-{index:02d}" for index in range(24)]
    new_ids = [f"new-{index:02d}" for index in range(24)]
    previous = {**_matrix(), "rows": [_row(row_id) for row_id in old_ids]}
    current = {
        **previous,
        "rows": [_row(row_id) for row_id in new_ids],
        "important_exclusions": ["A newly explicit exclusion."],
    }
    union = old_ids + new_ids
    assert len(union) == 48

    parsed = validate_risk_test_matrix_revision(
        previous,
        current,
        [
            {
                "operation": "change",
                "row_ids": union,
                "rationale": "Clarified exclusions across the complete old/new matrix scope.",
            },
            {"operation": "retire", "row_ids": old_ids, "rationale": "Replaced the prior row set."},
            {"operation": "add", "row_ids": new_ids, "rationale": "Introduced the revised row set."},
        ],
    )

    assert parsed[0].row_ids == tuple(union)


def test_matrix_audit_row_ids_remain_bounded_by_the_transition_union_limit() -> None:
    previous = _matrix()
    current = {**previous, "important_exclusions": ["A newly explicit exclusion."]}

    with pytest.raises(AgentLoopError, match="exceeds the 48-item bound"):
        validate_risk_test_matrix_revision(
            previous,
            current,
            [{
                "operation": "change",
                "row_ids": [f"row-{index:02d}" for index in range(49)],
                "rationale": "An invalid oversized matrix audit.",
            }],
        )


def test_one_row_matrix_level_and_row_change_share_one_complete_scope_audit() -> None:
    previous = _matrix()
    current = {
        **previous,
        "rows": [{**_row(), "expected_outcome": "A revised outcome"}],
        "important_exclusions": ["The revised exclusion boundary."],
    }

    parsed = validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": ["row-ordinary"],
            "rationale": "Updated the only row and the matrix exclusion boundary.",
        }],
    )

    assert parsed[0].row_ids == ("row-ordinary",)


def test_matrix_level_error_distinguishes_duplicate_complete_scope_audits() -> None:
    previous = _matrix()
    current = {**previous, "important_exclusions": ["A newly explicit exclusion."]}
    change = {
        "operation": "change",
        "row_ids": ["row-ordinary"],
        "rationale": "Repeated complete-scope audit.",
    }

    with pytest.raises(AgentLoopError, match="found 2; remove duplicate complete-scope entries"):
        validate_risk_test_matrix_revision(previous, current, [change, change])


def test_rowless_not_applicable_matrix_uses_matrix_sentinel_for_rationale_change() -> None:
    previous = _not_applicable()
    current = {
        **previous,
        "not_applicable_rationale": "The revised scope remains formatting-only.",
    }

    parsed = validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": ["matrix"],
            "rationale": "Refined why the matrix remains not applicable.",
        }],
    )

    assert parsed[0].row_ids == ("matrix",)


def test_matrix_level_change_also_audits_a_retained_row_change() -> None:
    previous = _matrix()
    current = {
        **previous,
        "rows": [{**_row(), "expected_outcome": "A revised outcome"}],
        "important_exclusions": ["The revised exclusion boundary."],
    }

    parsed = validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": ["row-ordinary"],
            "rationale": "Clarified the retained transition and its matrix exclusion.",
        }],
    )

    assert parsed[0].row_ids == ("row-ordinary",)


def test_matrix_level_change_audits_every_retained_row_changed_with_matrix_fields() -> None:
    previous = {
        **_matrix(),
        "rows": [_row("row-one"), _row("row-two"), _row("row-three")],
    }
    current = {
        **previous,
        "rows": [
            {**_row("row-one"), "expected_outcome": "Outcome one revised."},
            {**_row("row-two"), "expected_outcome": "Outcome two revised."},
            {**_row("row-three"), "expected_outcome": "Outcome three revised."},
        ],
        "important_exclusions": ["The complete-scope exclusion was revised."],
    }

    validate_risk_test_matrix_revision(
        previous,
        current,
        [{
            "operation": "change",
            "row_ids": ["row-one", "row-two", "row-three"],
            "rationale": "Updated every retained row and the matrix exclusions together.",
        }],
    )


@pytest.mark.parametrize("matrix_audit_first", [True, False])
def test_matrix_and_retained_row_audits_are_order_independent(matrix_audit_first: bool) -> None:
    previous = {
        **_matrix(),
        "rows": [_row("row-one"), _row("row-two")],
    }
    current = {
        **previous,
        "rows": [
            {**_row("row-one"), "expected_outcome": "Outcome one revised."},
            _row("row-two"),
        ],
        "important_exclusions": ["The complete-scope exclusion was revised."],
    }
    matrix_change = {
        "operation": "change",
        "row_ids": ["row-one", "row-two"],
        "rationale": "Updated the matrix exclusion boundary.",
    }
    row_change = {
        "operation": "change",
        "row_ids": ["row-one"],
        "rationale": "Clarified the retained transition.",
    }
    changes = [matrix_change, row_change] if matrix_audit_first else [row_change, matrix_change]

    validate_risk_test_matrix_revision(previous, current, changes)


def test_non_prefixed_stage_id_is_a_valid_parser_owner_but_topology_membership_is_checked() -> None:
    matrix = parse_risk_test_matrix({**_matrix(), "rows": [{**_row(), "execution_owner": "api"}]})
    recommendation = {
        "strategy": "staged",
        "rationale": "Split by lifecycle boundary.",
        "staging_feasibility": "safe",
        "scope_items": [{
            "scope_item_id": "scope-1", "requirement": "Do it.", "acceptance_criteria": ["Done."],
        }, {
            "scope_item_id": "scope-2", "requirement": "Integrate it.", "acceptance_criteria": ["Integrated."],
        }],
        "coupling_constraints": [],
        "child_stages": [{
            "stage_id": "api", "position": 1, "title": "API", "summary": "API.",
            "deliverables": ["API."], "non_goals": [], "acceptance_criteria": ["Done."],
            "depends_on_stage_ids": [], "dependency_notes": "None.", "automation": "agent-pr",
            "rollout_risk": "low", "compatibility_constraints": [], "covered_scope_item_ids": ["scope-1"],
        }, {
            "stage_id": "later", "position": 2, "title": "Integration", "summary": "Integration.",
            "deliverables": ["Integration."], "non_goals": [], "acceptance_criteria": ["Integrated."],
            "depends_on_stage_ids": ["api"], "dependency_notes": "After API.", "automation": "agent-pr",
            "rollout_risk": "low", "compatibility_constraints": [], "covered_scope_item_ids": ["scope-2"],
        }],
        "retained_parent_work": {"status": "none", "deliverables": [], "acceptance_criteria": [], "covered_scope_item_ids": []},
        "final_integration_work": {"status": "none", "deliverables": [], "acceptance_criteria": [], "covered_scope_item_ids": []},
        "caveats": [],
    }
    from coding_review_agent_loop.protocol import parse_execution_recommendation_payload

    parsed_recommendation = parse_execution_recommendation_payload(recommendation)
    validate_risk_matrix_ownership(matrix, parsed_recommendation)


def test_m780_04_not_applicable_and_matrix_less_plan_payloads_remain_compatible() -> None:
    base = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": "blocking",
        "summary": "A narrow plan.",
        "plan_steps": ["Make the local change."],
    }
    text = json.dumps(base) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex"
    assert validate_structured_plan_state(text).risk_test_matrix is None
    matrix_payload = {
        **base,
        "risk_test_matrix_contract_version": 1,
        "risk_test_matrix": _not_applicable(),
        "risk_test_matrix_changes": [],
    }
    parsed = validate_structured_plan_state(
        json.dumps(matrix_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        require_risk_test_matrix_contract=1,
    )
    assert parsed is not None and parsed.risk_test_matrix is not None


def test_m780_05_evidence_requires_exact_row_identity_and_preserves_incomplete_status() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = parse_risk_test_matrix_evidence(
        {
            "matrix_identity": identity,
            "rows": [{
                "row_id": "row-ordinary",
                "status": "timed-out",
                "test_identifiers": ["test_review_only_recovery"],
                "test_locations": ["tests/test_orchestrator_pr.py:100"],
                "workflow_path_claim": "Reached the post-review recovery branch.",
                "outcome_assertions": ["Completion was not observed before timeout."],
                "forbidden_effect_assertions": ["No merge call occurred."],
                "evidence_citations": [{"command": "python -m pytest tests/test_orchestrator_pr.py -q", "receipt_id": "receipt-1", "claim": "current-result"}],
                "caveats": ["The managed test gate timed out."],
            }],
        },
        matrix=matrix,
    )
    assert evidence.rows[0].status == "timed-out"
    with pytest.raises(AgentLoopError):
        parse_risk_test_matrix_evidence(
            {"matrix_identity": identity, "rows": [{"row_id": "row-ordinary"}]},
            matrix=matrix,
        )


def test_m780_06_metadata_round_trip_authenticates_payload_not_rendered_words() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    metadata = PostedRoundMetadata(
        flow="plan",
        role="coder",
        agent="Codex",
        round_number=1,
        subject="subject",
        canonical_plan="A canonical plan\n\n" + render_risk_test_matrix_section(matrix),
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.risk_test_matrix_payload == matrix.to_payload()
    assert decoded.risk_test_matrix_identity == identity
    corrupted = _encode_round_metadata(
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Codex", round_number=1, subject="subject",
            canonical_plan="A canonical plan", risk_test_matrix_contract_version=1,
            risk_test_matrix_payload={**matrix.to_payload(), "rows": []},
            risk_test_matrix_identity=identity,
        )
    )
    closed = _decode_round_metadata(corrupted)
    assert closed.risk_test_matrix_payload is None
    assert closed.risk_test_matrix_diagnostic


def test_legacy_plan_recovery_ignores_empty_default_matrix_audit() -> None:
    canonical = "Legacy approved plan\n\n### Scope\n- Preserve the API."
    comment = SimpleNamespace(
        body=_attach_round_metadata(
            canonical,
            PostedRoundMetadata(
                flow="plan",
                role="coder",
                agent="Codex",
                round_number=1,
                subject="legacy",
                canonical_plan=canonical,
            ),
        )
    )

    expected = make_approved_plan_context(canonical)
    recovered = recover_approved_plan_context(
        (comment,),
        expected_hash=expected.plan_hash,
        expected_subject=expected.plan_subject,
    )

    assert recovered.is_available
    assert recovered.risk_test_matrix_availability == "not-planned"
    assert recovered.risk_test_matrix_diagnostic is None


def test_m780_11_payload_bound_boundary_mismatch_closes_only_matrix_channel() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "Canonical plan prose",
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest="0" * 64,
    )
    assert context.availability == "available"
    assert not context.matrix_available
    assert context.risk_test_matrix_diagnostic


def test_m780_07_approved_context_has_matrix_channel_even_if_renderer_wording_changes() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "Summary\n\n" + render_risk_test_matrix_section(matrix).replace(
            "### Risk-based mode and transition test matrix",
            "### Risk-based mode and transition test matrix (renderer v2)",
        ),
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    assert context.availability == "available"
    assert context.matrix_available
    assert context.risk_test_matrix_payload == matrix.to_payload()


def test_marker_only_approved_context_preserves_non_empty_change_audit() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    changes = (
        RiskTestMatrixChange(
            "change",
            ("row-ordinary",),
            "Clarified the post-review recovery outcome.",
        ),
    )
    canonical = "Approved plan\n\n" + render_risk_test_matrix_section(matrix, changes)

    context = make_approved_plan_context(canonical)

    assert context.availability == "available"
    assert context.matrix_available
    assert context.risk_test_matrix_expected_row_ids == ("row-ordinary",)
    assert context.risk_test_matrix_changes_payload == tuple(
        change.to_payload() for change in changes
    )


def test_m780_08a_matrix_semantics_remain_available_without_canonical_prose() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    assert context.is_available
    assert context.canonical_text is None
    assert context.matrix_available


def test_m780_09_public_evidence_renders_outstanding_rows_first() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = parse_risk_test_matrix_evidence(
        {
            "matrix_identity": identity,
            "rows": [{
                "row_id": "row-ordinary", "status": "missing", "test_identifiers": [],
                "test_locations": [], "workflow_path_claim": "Not exercised.",
                "outcome_assertions": [], "forbidden_effect_assertions": [], "evidence_citations": [],
            }],
        },
        matrix=matrix,
    )
    from coding_review_agent_loop.protocol import StructuredIssueImplementation, StructuredHumanRequirementsPayload

    text = render_public_agent_comment(
        kind="issue_implementation",
        parsed=StructuredIssueImplementation(
            schema_version=1, kind="issue_implementation", state="blocking",
            summary="Implementation is complete but the workflow row is still open.", pr_number=1,
            human_requirements=StructuredHumanRequirementsPayload((), False),
            human_requirement_dispositions=(), risk_test_matrix_evidence=evidence,
        ),
        agent="Codex",
    )
    assert text.index("row-ordinary") < text.index("AGENT_STATE")


def test_m780_05_verified_evidence_requires_a_passing_authoritative_receipt() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = {
        "matrix_identity": identity,
        "rows": [{
            "row_id": "row-ordinary", "status": "verified",
            "test_identifiers": ["test_review_only_recovery"],
            "test_locations": ["tests/test_orchestrator_pr.py:1"],
            "workflow_path_claim": "The orchestrator recovery path was exercised.",
            "outcome_assertions": ["The workflow completes."],
            "forbidden_effect_assertions": ["No stale head was merged."],
            "evidence_citations": [{
                "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
                "receipt_id": "receipt-1",
                "claim": "current-result",
            }],
        }],
    }
    authoritative = SimpleNamespace(
        receipt_id="receipt-1",
        claim=None,
        normalized_command="python3 -m pytest tests/test_orchestrator_pr.py -q",
        outcome="passed",
        provenance="parent-observed",
        public_projection=lambda: {"command": "python3 -m pytest tests/test_orchestrator_pr.py -q"},
    )
    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[authoritative],
    )
    assert parsed.rows[0].status == "verified"
    with pytest.raises(AgentLoopError, match="authoritative.*receipts"):
        parse_risk_test_matrix_evidence(
            evidence,
            matrix=matrix,
            authoritative_test_observations=[],
        )


def test_managed_test_wrapper_citation_matches_broker_inner_command() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "/home/test/.local/bin/agent-loop run-tests --timeout-seconds 1800 "
        "--memory-dir /home/test/.cache/agent-loop -- "
        "python3 -m pytest tests/test_orchestrator_pr.py -q"
    )

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt()],
    )

    assert parsed.rows[0].status == "verified"


def test_module_managed_test_wrapper_citation_matches_broker_inner_command() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "/opt/venv/bin/python -m coding_review_agent_loop.cli run-tests "
        "--timeout-seconds 1800 --memory-dir /home/test/.cache/agent-loop -- "
        "python3 -m pytest tests/test_orchestrator_pr.py -q"
    )

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt()],
    )

    assert parsed.rows[0].status == "verified"


def test_prefixed_managed_test_wrapper_citation_matches_broker_inner_command() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "MODE=inline timeout 1800 env -u AGENT_LOOP_INVOCATION_ID "
        "/home/test/.local/bin/agent-loop run-tests --timeout-seconds 1800 "
        "--memory-dir /home/test/.cache/agent-loop -- "
        "python3 -m pytest tests/test_orchestrator_pr.py -q"
    )

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt()],
    )

    assert parsed.rows[0].status == "verified"


def test_bare_launcher_citation_matches_broker_inner_command() -> None:
    """Issue #892: the launcher spelling agents actually report must project."""
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "agent-loop run-tests --timeout-seconds 1800 "
        "--memory-dir /home/test/.cache/agent-loop -- "
        "python3 -m pytest tests/test_orchestrator_pr.py -q"
    )

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt()],
    )

    assert parsed.rows[0].status == "verified"


def test_bare_module_launcher_citation_matches_broker_inner_command() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "python3 -m coding_review_agent_loop.cli run-tests "
        "--memory-dir /home/test/.cache/agent-loop -- "
        "python3 -m pytest tests/test_orchestrator_pr.py -q"
    )

    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt()],
    )

    assert parsed.rows[0].status == "verified"


def test_managed_test_wrapper_citation_rejects_different_inner_command() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    citation = evidence["rows"][0]["evidence_citations"][0]
    citation["command"] = (
        "agent-loop run-tests --timeout-seconds 1800 -- "
        "python3 -m pytest tests/test_protocol.py -q"
    )

    with pytest.raises(AgentLoopError, match="passing.*receipts"):
        parse_risk_test_matrix_evidence(
            evidence,
            matrix=matrix,
            authoritative_test_observations=[_rich_receipt()],
        )


@pytest.mark.parametrize("command", [
    "/opt/venv/bin/agent-loop run-tests --unknown -- "
    "python3 -m pytest tests/test_orchestrator_pr.py -q",
    "/opt/venv/bin/agent-loop run-tests --timeout-seconds 1 "
    "--timeout-seconds 2 -- python3 -m pytest tests/test_orchestrator_pr.py -q",
    "agent-loop run-tests --unknown -- "
    "python3 -m pytest tests/test_orchestrator_pr.py -q",
    "../agent-loop run-tests --timeout-seconds 1800 -- "
    "python3 -m pytest tests/test_orchestrator_pr.py -q",
])
def test_managed_test_wrapper_citation_rejects_noncanonical_or_malformed_command(
    command: str,
) -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "verified")
    evidence["rows"][0]["evidence_citations"][0]["command"] = command

    with pytest.raises(AgentLoopError, match="passing.*receipts"):
        parse_risk_test_matrix_evidence(
            evidence,
            matrix=matrix,
            authoritative_test_observations=[_rich_receipt()],
        )


def _rich_receipt(
    *,
    outcome: str = "passed",
    attribution_state: str = "current-head",
    receipt_id: str = "receipt-rich",
    stable: bool | None = True,
    attribution_caveats: tuple[str, ...] = (),
    caveats: tuple[str, ...] = (),
    wrapper_bootstrap: str = "verified",
    inner_exec: str = "started",
    suite_start: str = "verified",
    environment_state: str = "not-compared",
    superseded_by: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        receipt_id=receipt_id,
        claim=None,
        normalized_command="python3 -m pytest tests/test_orchestrator_pr.py -q",
        outcome=outcome,
        provenance="parent-observed",
        attribution=SimpleNamespace(
            state=attribution_state,
            stable=stable,
            untracked_input=False,
            caveats=attribution_caveats,
        ),
        environment_state=environment_state,
        superseded_by=superseded_by,
        caveats=caveats,
        wrapper_bootstrap=wrapper_bootstrap,
        inner_exec=inner_exec,
        suite_start=suite_start,
        public_projection=lambda: {
            "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
        },
    )


def _evidence_for_status(identity: str, status: str) -> dict[str, object]:
    return {
        "matrix_identity": identity,
        "rows": [{
            "row_id": "row-ordinary", "status": status,
            "test_identifiers": ["test_review_only_recovery"],
            "test_locations": ["tests/test_orchestrator_pr.py:1"],
            "workflow_path_claim": "The recovery branch was exercised.",
            "outcome_assertions": ["The observed outcome is retained."],
            "forbidden_effect_assertions": ["No forbidden side effect occurred."],
            "evidence_citations": [{
                "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
                "receipt_id": "receipt-rich", "claim": "current-result",
            }],
        }],
    }


def test_receipt_semantics_reject_stale_verified_and_relabelled_failure() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    with pytest.raises(AgentLoopError, match="passing.*receipts"):
        parse_risk_test_matrix_evidence(
            _evidence_for_status(identity, "verified"),
            matrix=matrix,
            authoritative_test_observations=[_rich_receipt(attribution_state="stale")],
        )
    with pytest.raises(AgentLoopError, match="passing.*receipts"):
        parse_risk_test_matrix_evidence(
            _evidence_for_status(identity, "verified"),
            matrix=matrix,
            authoritative_test_observations=[_rich_receipt(outcome="failed")],
        )
    parsed = parse_risk_test_matrix_evidence(
        _evidence_for_status(identity, "failed"),
        matrix=matrix,
        authoritative_test_observations=[_rich_receipt(outcome="failed")],
    )
    assert parsed.rows[0].status == "failed"


@pytest.mark.parametrize(
    "field, value",
    [
        ("wrapper_bootstrap", "unknown"),
        ("inner_exec", "not-attempted"),
        ("suite_start", "not-started"),
    ],
)
def test_verified_receipts_require_affirmative_launch_and_suite_boundaries(field: str, value: str) -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    receipt = _rich_receipt(**{field: value})
    with pytest.raises(AgentLoopError, match="passing.*receipts"):
        parse_risk_test_matrix_evidence(
            _evidence_for_status(identity, "verified"),
            matrix=matrix,
            authoritative_test_observations=[receipt],
        )


def test_receipt_caveats_and_mixed_outcomes_are_retained_as_incomplete() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    evidence = _evidence_for_status(identity, "incomplete")
    evidence["rows"][0]["evidence_citations"] = [
        {
            "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
            "receipt_id": "receipt-rich",
            "claim": "current-result",
        },
        {
            "command": "python3 -m pytest tests/test_orchestrator_pr.py -q",
            "receipt_id": "receipt-failed",
            "claim": "current-result",
        },
    ]
    parsed = parse_risk_test_matrix_evidence(
        evidence,
        matrix=matrix,
        authoritative_test_observations=[
            _rich_receipt(
                receipt_id="receipt-rich",
                attribution_caveats=("head comparison caveat",),
                caveats=("wrapper caveat",),
            ),
            _rich_receipt(receipt_id="receipt-failed", outcome="failed"),
        ],
    )
    assert parsed.rows[0].status == "incomplete"
    assert "head comparison caveat" in parsed.rows[0].caveats
    assert "wrapper caveat" in parsed.rows[0].caveats


def test_m780_05_row_not_applicable_is_not_an_evidence_obligation() -> None:
    payload = _matrix()
    payload["rows"] = [
        _row("row-required"),
        {**_row("row-excluded"), "applicability": "not-applicable"},
    ]
    matrix = parse_risk_test_matrix(payload)
    identity = risk_test_matrix_identity(matrix)
    evidence = parse_risk_test_matrix_evidence(
        {
            "matrix_identity": identity,
            "rows": [{
                "row_id": "row-required", "status": "missing",
                "test_identifiers": [], "test_locations": [],
                "workflow_path_claim": "Not run.", "outcome_assertions": [],
                "forbidden_effect_assertions": [], "evidence_citations": [],
            }],
        },
        matrix=matrix,
    )
    assert [row.row_id for row in evidence.rows] == ["row-required"]


def test_m780_08c_oversized_matrix_is_omitted_atomically_from_prompt_context() -> None:
    large = {**_row(), "label": "x" * 900, "expected_outcome": "y" * 900}
    matrix = parse_risk_test_matrix({**_matrix(), "rows": [large]})
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "Approved plan prose\n\n" + render_risk_test_matrix_section(matrix),
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    from coding_review_agent_loop.prompts import format_approved_plan_context

    rendered = format_approved_plan_context(context, max_chars=2_000)
    assert "matrix: unavailable" in rendered
    assert "zero matrix rows are enforceable" in rendered
    assert "Row row-ordinary" not in rendered


def test_complete_fitting_matrix_context_keeps_matrix_enforceable() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    from coding_review_agent_loop.prompts import format_approved_plan_context

    rendered = format_approved_plan_context(context, max_chars=5_000)
    assert "Row row-ordinary" in rendered
    assert "matrix: unavailable" not in rendered


def test_m780_08b_fitting_identity_only_context_is_diagnostic_only() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        None,
        expected_hash="a" * 16,
        expected_subject="b" * 64,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=None,
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    from coding_review_agent_loop.prompts import format_approved_plan_context

    rendered = format_approved_plan_context(context, max_chars=5_000)
    assert not context.matrix_available
    assert context.risk_test_matrix_diagnostic
    assert "matrix: unavailable" in rendered
    assert "Row row-ordinary" not in rendered


def test_matrix_priority_omits_large_canonical_prose_before_small_matrix() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        "large approved body\n" + "x" * 70_000 + "\n" + render_risk_test_matrix_section(matrix),
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    from coding_review_agent_loop.prompts import format_approved_plan_context

    rendered = format_approved_plan_context(context, max_chars=5_000)
    assert "Row row-ordinary" in rendered
    assert "Canonical approved plan text: omitted" in rendered
    assert "matrix: unavailable" not in rendered


def test_matching_plan_records_with_divergent_matrix_metadata_close_only_matrix_channel() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    other_matrix = parse_risk_test_matrix({**_matrix(), "rows": [{**_row(), "expected_outcome": "Different."}]})
    other_identity = risk_test_matrix_identity(other_matrix)
    canonical = "Approved plan\n\n" + render_risk_test_matrix_section(matrix)
    comments = [
        SimpleNamespace(body=_attach_round_metadata(canonical, PostedRoundMetadata(
            flow="plan", role="coder", agent="Codex", round_number=1,
            subject="subject", canonical_plan=canonical,
            risk_test_matrix_contract_version=1, risk_test_matrix_payload=matrix.to_payload(),
            risk_test_matrix_identity=identity, risk_test_matrix_boundary_digest=identity,
        ))),
        SimpleNamespace(body=_attach_round_metadata(canonical, PostedRoundMetadata(
            flow="plan", role="coder", agent="Codex", round_number=2,
            subject="subject", canonical_plan=canonical,
            risk_test_matrix_contract_version=1, risk_test_matrix_payload=other_matrix.to_payload(),
            risk_test_matrix_identity=other_identity, risk_test_matrix_boundary_digest=other_identity,
        ))),
    ]
    recovered = recover_approved_plan_context(
        comments,
        expected_hash=hashlib.sha256(canonical.strip().encode()).hexdigest()[:16],
    )
    assert recovered.availability == "available"
    assert not recovered.matrix_available
    assert recovered.risk_test_matrix_diagnostic


def test_m780_12_staged_context_keeps_pending_rows_read_only() -> None:
    payload = _matrix()
    payload["rows"] = [
        {**_row("row-owned"), "execution_owner": "stage-first"},
        {**_row("row-later"), "execution_owner": "stage-later"},
    ]
    matrix = parse_risk_test_matrix(payload)
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        render_risk_test_matrix_section(matrix),
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    scoped = scope_approved_plan_matrix(
        context,
        execution_owner="stage-first",
        valid_stage_ids=("stage-first", "stage-later"),
    )
    assert scoped.risk_test_matrix_expected_row_ids == ("row-owned",)
    assert scoped.risk_test_matrix_pending_row_ids == ("row-later",)


def test_scope_not_applicable_matrix_skips_stage_owner_validation() -> None:
    matrix = parse_risk_test_matrix(_not_applicable())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        render_risk_test_matrix_section(matrix),
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )

    scoped = scope_approved_plan_matrix(
        context,
        execution_owner="stage-not-in-plan",
        valid_stage_ids=("stage-first",),
    )

    assert scoped is context
    assert scoped.risk_test_matrix_expected_row_ids == ()


def test_scope_applicable_matrix_rejects_unknown_stage_owner() -> None:
    matrix = parse_risk_test_matrix(_matrix())
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        render_risk_test_matrix_section(matrix),
        expected_hash=None,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )

    with pytest.raises(AgentLoopError, match="not an approved stage ID"):
        scope_approved_plan_matrix(
            context,
            execution_owner="stage-not-in-plan",
            valid_stage_ids=("stage-first",),
        )


def test_truncated_semantic_claim_parses_but_never_derives_as_verified() -> None:
    """#913: accepted truncation is content loss, so the row must not verify."""
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="invocation:observation-1", receipt_id="receipt-1"
    )
    identifiers = [f"tests/test_mod.py::test_case_{index}" for index in range(13)]
    payload = {
        "schema_version": 1,
        "kind": "coder_followup",
        "state": "blocking",
        "summary": "Updated the PR.",
        "addressed_items": [],
        "remaining_items": [],
        "human_requirement_dispositions": [],
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "risk_test_matrix_claims": [{
            "row_id": "row-ordinary",
            "execution_refs": ["invocation:observation-1"],
            "test_identifiers": identifiers,
            "test_locations": ["tests/test_mod.py"],
            "workflow_path_claim": "ordinary / review-only",
            "outcome_assertions": ["Every listed test passed."],
            "forbidden_effect_assertions": ["No unauthorized evidence was accepted."],
        }],
    }
    parsed = validate_structured_coder_followup(
        json.dumps(payload) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        delivered_risk_test_matrix_row_ids=["row-ordinary"],
        execution_catalog=[{
            "execution_ref": "invocation:observation-1",
            "outcome": "passed",
            "provenance": "parent-observed",
        }],
    )

    claim = parsed.risk_test_matrix_claims.claims[0]
    assert claim.truncated_fact_fields == ("test_identifiers",)
    assert claim.test_identifiers == tuple(identifiers[:12])

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=parsed.risk_test_matrix_claims,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    row = result.evidence.rows[0]
    assert row.status == "stale/unverified"
    assert row.evidence_citations == ()
    truncation = [d for d in result.diagnostics if d.code == "truncated-semantic-claim"]
    assert len(truncation) == 1
    assert "test_identifiers" in truncation[0].message
    assert any("lost listed facts" in caveat for caveat in row.caveats)


def test_untruncated_semantic_claim_still_verifies() -> None:
    """#913 guard: the truncation diagnostic must not fire on a bounded claim."""
    matrix = parse_risk_test_matrix(_matrix())
    observation = _derived_observation(
        execution_ref="invocation:observation-1", receipt_id="receipt-1"
    )
    claims = SemanticRiskCoverageClaims((SemanticRiskCoverageClaim(
        row_id="row-ordinary",
        execution_refs=("invocation:observation-1",),
        test_identifiers=tuple(f"tests/test_mod.py::test_case_{index}" for index in range(12)),
        test_locations=("tests/test_mod.py",),
        workflow_path_claim="ordinary / review-only",
        outcome_assertions=("Every listed test passed.",),
        forbidden_effect_assertions=("No unauthorized evidence was accepted.",),
    ),))

    result = derive_risk_test_matrix_evidence(
        matrix=matrix,
        claims=claims,
        observations=(observation,),
        invocation_id="turn-current",
        current_head="head-current",
        current_tree_digest="tree-current",
        authenticated_checkout_head="head-current",
        authenticated_tree_clean=True,
        expected_identity=risk_test_matrix_identity(matrix),
    )

    assert result.evidence.rows[0].status == "verified"
    assert not [d for d in result.diagnostics if d.code == "truncated-semantic-claim"]
