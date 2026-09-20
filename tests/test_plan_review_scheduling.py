"""Unit tests for the pure planning-review scheduling decision module (#904).

These tests deliberately import no GitHub, agent, or orchestrator module: the
planning scheduler is a pure decision surface and must stay replayable offline.
"""

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.review_scheduling import (
    POST_PANEL_PREFIX,
    STRICT_PRE_PANEL_PREFIX,
    ReviewObligation,
)
from coding_review_agent_loop.plan_review_scheduling import (
    OPERATOR_PLAN_FORCE_FULL_REASON,
    PLAN_HISTORY_ABSENT,
    PLAN_HISTORY_CONTRADICTORY_KEY,
    PLAN_HISTORY_INTACT,
    PLAN_HISTORY_INVALID,
    PLAN_HISTORY_TRANSPORT_FAILURE,
    PLAN_REVIEW_POLICIES,
    PlanCandidateKey,
    PlanCrossCuttingContracts,
    PlanPrePanelSafetyError,
    PlanReviewSchedulingContract,
    PlanRevisionDescriptor,
    PlanSchedulerSnapshot,
    PlanTransitionClassification,
    classify_plan_history,
    classify_plan_transition,
    make_plan_contract,
    plan_history_continues,
    plan_history_fallback_reason,
    plan_policy_capabilities,
    select_plan_reviewers,
    surfaced_requirement_id_digest,
)

PRIMARY = "Codex"
SECONDARIES = ("Claude", "Antigravity")
BOARD = (PRIMARY,) + SECONDARIES


def _contract(policy: str = "primary-then-panel", *, primary: str | None = PRIMARY):
    return PlanReviewSchedulingContract(
        required_reviewers=BOARD,
        policy=policy,
        primary_reviewer=primary,
    )


def _key(
    *,
    subject: str = "a" * 64,
    plan: str = "plan-1",
    strategy: str = "strategy-1",
    matrix: str = "matrix-1",
    requirements: str = "req-1",
    version: int | None = 1,
) -> PlanCandidateKey:
    return PlanCandidateKey(
        subject=subject,
        aggregate_plan_identity=plan,
        execution_strategy_identity=strategy,
        risk_test_matrix_identity=matrix,
        surfaced_requirement_id_digest=requirements,
        execution_strategy_contract_version=version,
    )


def _contracts(**overrides) -> PlanCrossCuttingContracts:
    values = {
        "execution_recommendation_identity": "rec-1",
        "human_requirement_disposition_digest": "disp-1",
        "additional_closing_issue_ids": (),
        "architecture_impact_status": "unchanged",
    }
    values.update(overrides)
    return PlanCrossCuttingContracts(**values)


def _revision(base: str = "plan-1", **overrides) -> PlanRevisionDescriptor:
    values = {
        "response_form": "semantic-patch-v1",
        "semantic_patch_contract_version": 1,
        "base_state_identity": base,
        "sidecar_bound": True,
        "operation_fields": ("plan_steps",),
        "matrix_operations": ("matrix_edit",),
    }
    values.update(overrides)
    return PlanRevisionDescriptor(**values)


def _obligation(item_id="item-1", owners=(PRIMARY,), status="blocking"):
    return ReviewObligation(
        item_id=item_id,
        status=status,
        scope=None,
        resolution_owners=tuple(owners),
        pending_owners=tuple(owners),
    )


def _snapshot(**overrides) -> PlanSchedulerSnapshot:
    values = {
        "contract": _contract(),
        "previous_key": None,
        "current_key": _key(),
    }
    values.update(overrides)
    return PlanSchedulerSnapshot(**values)


def _recheck() -> PlanTransitionClassification:
    return PlanTransitionClassification("recheck", "the candidate plan key is unchanged")


def _paused(decision) -> dict[str, str]:
    return dict(decision.paused_reviewers)


# --------------------------------------------------------------------------
# Contract validation
# --------------------------------------------------------------------------


def test_plan_policies_exclude_the_pr_only_selective_policy():
    assert PLAN_REVIEW_POLICIES == frozenset({"all-reviewers", "primary-then-panel"})
    with pytest.raises(AgentLoopError):
        plan_policy_capabilities("selective-intermediate")


def test_contract_carries_no_pr_only_broad_rule_state():
    contract = _contract()
    assert contract.as_dict() == {
        "required_reviewers": list(BOARD),
        "policy": "primary-then-panel",
        "primary_reviewer": PRIMARY,
    }
    assert not hasattr(contract, "broad_rules")
    assert contract.secondary_reviewers == SECONDARIES
    assert PlanReviewSchedulingContract.from_mapping(contract.as_dict()) == contract


def test_contract_rejects_staged_policy_without_a_primary():
    with pytest.raises(AgentLoopError, match="must be a member"):
        PlanReviewSchedulingContract(
            required_reviewers=BOARD, policy="primary-then-panel"
        )


def test_contract_rejects_staged_policy_with_a_single_reviewer_board():
    with pytest.raises(AgentLoopError, match="at least one secondary"):
        PlanReviewSchedulingContract(
            required_reviewers=(PRIMARY,),
            policy="primary-then-panel",
            primary_reviewer=PRIMARY,
        )


def test_contract_rejects_a_primary_outside_the_board():
    with pytest.raises(AgentLoopError, match="must be a member"):
        PlanReviewSchedulingContract(
            required_reviewers=BOARD,
            policy="primary-then-panel",
            primary_reviewer="Gemini",
        )


def test_contract_rejects_a_primary_under_the_compatibility_default():
    with pytest.raises(AgentLoopError, match="may only be configured"):
        PlanReviewSchedulingContract(
            required_reviewers=BOARD,
            policy="all-reviewers",
            primary_reviewer=PRIMARY,
        )


def test_make_plan_contract_matches_direct_construction():
    assert make_plan_contract(BOARD, "primary-then-panel", PRIMARY) == _contract()


# --------------------------------------------------------------------------
# Candidate key
# --------------------------------------------------------------------------


def test_candidate_key_is_one_ordered_tuple_and_round_trips():
    key = _key()
    assert key.components == ("a" * 64, "plan-1", "strategy-1", "matrix-1", "req-1")
    assert key.complete
    assert PlanCandidateKey.from_mapping(key.as_dict()) == key


def test_candidate_key_requires_generation_one():
    key = _key(version=None)
    assert not key.complete
    assert "generation 1" in key.incompleteness_reason()
    assert not key.matches(_key())


def test_candidate_key_missing_component_names_it():
    key = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="plan-1",
        execution_strategy_identity=None,
        risk_test_matrix_identity="matrix-1",
        surfaced_requirement_id_digest="req-1",
    )
    assert key.missing_components == ("execution_strategy_identity",)
    assert "execution_strategy_identity" in key.incompleteness_reason()


def test_requirement_digest_is_order_insensitive_and_set_sensitive():
    assert surfaced_requirement_id_digest(("hr-2", "hr-1")) == (
        surfaced_requirement_id_digest(("hr-1", "hr-2"))
    )
    assert surfaced_requirement_id_digest(("hr-1",)) != (
        surfaced_requirement_id_digest(("hr-1", "hr-2"))
    )
    assert surfaced_requirement_id_digest(()) == surfaced_requirement_id_digest(None)


def test_requirement_digest_rejects_a_bare_string():
    with pytest.raises(AgentLoopError):
        surfaced_requirement_id_digest("hr-1")


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------


def test_unchanged_candidate_key_is_a_recheck():
    classification = classify_plan_transition(_key(), _key())
    assert classification.kind == "recheck"
    assert classification.owner_scoped


def test_plan_step_and_matrix_row_remediation_is_narrow():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2", matrix="matrix-2"),
        _revision(operation_fields=("plan_steps", "summary"), matrix_operations=("matrix_add", "matrix_edit")),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.kind == "narrow"
    assert classification.owner_scoped


def test_cross_cutting_contract_change_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2", strategy="strategy-2"),
        _revision(),
        previous_contracts=_contracts(),
        current_contracts=_contracts(execution_recommendation_identity="rec-2"),
    )
    assert classification.broad
    assert "execution_recommendation_identity" in classification.reason


@pytest.mark.parametrize(
    "overrides",
    [
        {"human_requirement_disposition_digest": "disp-2"},
        {"additional_closing_issue_ids": ("905",)},
        {"architecture_impact_status": "changed"},
    ],
)
def test_every_cross_cutting_contract_latches_broad(overrides):
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=_contracts(),
        current_contracts=_contracts(**overrides),
    )
    assert classification.broad


def test_full_state_rewrite_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(response_form="legacy-full-state"),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad
    assert "semantic-patch-v1" in classification.reason


def test_unbindable_sidecar_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(sidecar_bound=False),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad


def test_patch_not_bound_to_the_preceding_identity_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(base="plan-0"),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad
    assert "immediately preceding" in classification.reason


def test_operation_outside_narrow_fields_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(operation_fields=("architecture_impact",)),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad
    assert "architecture_impact" in classification.reason


def test_matrix_retire_operation_is_broad():
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(matrix_operations=("matrix_retire",)),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad
    assert "matrix_retire" in classification.reason


def test_defaulted_cross_cutting_identities_are_broad_not_narrow():
    """Two unobserved identity sets compare equal; that is not evidence."""
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=PlanCrossCuttingContracts(),
        current_contracts=PlanCrossCuttingContracts(),
    )
    assert classification.broad
    assert "could not be observed" in classification.reason
    for name in (
        "execution_recommendation_identity",
        "human_requirement_disposition_digest",
        "architecture_impact_status",
    ):
        assert name in classification.reason


@pytest.mark.parametrize(
    "missing",
    [
        "execution_recommendation_identity",
        "human_requirement_disposition_digest",
        "architecture_impact_status",
    ],
)
def test_one_unobserved_identity_on_either_side_is_broad(missing):
    incomplete = _contracts(**{missing: None})
    assert incomplete.missing_identities == (missing,)
    assert not incomplete.complete
    for previous, current in ((incomplete, _contracts()), (_contracts(), incomplete)):
        classification = classify_plan_transition(
            _key(),
            _key(plan="plan-2"),
            _revision(),
            previous_contracts=previous,
            current_contracts=current,
        )
        assert classification.broad
        assert missing in classification.reason


@pytest.mark.parametrize(
    "missing",
    [
        "execution_recommendation_identity",
        "human_requirement_disposition_digest",
    ],
)
@pytest.mark.parametrize("blank", ["", "   ", "\t"])
def test_blank_identities_are_unobserved_and_classify_broad(missing, blank):
    """A blank identity is not an observation; two blanks must not compare equal."""
    incomplete = _contracts(**{missing: blank})
    assert incomplete.missing_identities == (missing,)
    assert not incomplete.complete
    both_blank = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=_contracts(**{missing: blank}),
        current_contracts=_contracts(**{missing: blank}),
    )
    assert both_blank.broad
    assert missing in both_blank.reason
    for previous, current in ((incomplete, _contracts()), (_contracts(), incomplete)):
        one_side = classify_plan_transition(
            _key(),
            _key(plan="plan-2"),
            _revision(),
            previous_contracts=previous,
            current_contracts=current,
        )
        assert one_side.broad
        assert missing in one_side.reason


@pytest.mark.parametrize(
    "field",
    [
        "execution_recommendation_identity",
        "human_requirement_disposition_digest",
        "architecture_impact_status",
    ],
)
@pytest.mark.parametrize("wrong", [1, 1.5, True, object(), ("rec-1",), ["rec-1"], {"a": 1}])
def test_wrong_typed_identities_are_rejected_outright(field, wrong):
    with pytest.raises(AgentLoopError, match="must be a string or None"):
        _contracts(**{field: wrong})


def test_blank_architecture_status_is_unobserved_and_classifies_broad():
    incomplete = _contracts(architecture_impact_status="  ")
    assert incomplete.missing_identities == ("architecture_impact_status",)
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=incomplete,
        current_contracts=_contracts(architecture_impact_status=""),
    )
    assert classification.broad
    assert "architecture_impact_status" in classification.reason


def test_architecture_status_domain_is_enforced():
    for value in ("Unchanged", "modified", "no-change", "unknown"):
        with pytest.raises(AgentLoopError, match="architecture_impact_status"):
            _contracts(architecture_impact_status=value)
    for value in ("changed", "unchanged"):
        assert _contracts(architecture_impact_status=value).complete


def test_identities_are_stripped_before_comparison():
    padded = _contracts(execution_recommendation_identity="  rec-1  ")
    assert padded.execution_recommendation_identity == "rec-1"
    assert padded == _contracts()
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=padded,
        current_contracts=_contracts(),
    )
    assert classification.narrow


def test_an_empty_closing_issue_set_is_a_complete_declaration():
    contracts = _contracts(additional_closing_issue_ids=())
    assert contracts.complete
    assert contracts.missing_identities == ()
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(),
        previous_contracts=contracts,
        current_contracts=contracts,
    )
    assert classification.narrow


def test_an_unobserved_operation_set_is_broad_not_narrow():
    """An authenticated patch always carries an operation; empty means unobserved."""
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(operation_fields=(), matrix_operations=()),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.broad
    assert "reports no authenticated patch operations" in classification.reason


@pytest.mark.parametrize(
    "fields,matrix",
    [
        (("plan_steps",), ()),
        ((), ("matrix_edit",)),
        (("summary", "deferred_work"), ()),
        ((), ("matrix_add",)),
    ],
)
def test_one_observed_operation_side_is_enough_to_classify_narrow(fields, matrix):
    classification = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(operation_fields=fields, matrix_operations=matrix),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert classification.narrow


def test_an_unobserved_operation_set_cannot_smuggle_an_excluded_operation():
    """The empty-set guard protects the excluded field and matrix operations."""
    for fields, matrix in (
        (("deferred_stages",), ()),
        ((), ("matrix_retire",)),
        ((), ("matrix_split",)),
        ((), ("matrix_merge",)),
    ):
        observed = classify_plan_transition(
            _key(),
            _key(plan="plan-2"),
            _revision(operation_fields=fields, matrix_operations=matrix),
            previous_contracts=_contracts(),
            current_contracts=_contracts(),
        )
        assert observed.broad
    unobserved = classify_plan_transition(
        _key(),
        _key(plan="plan-2"),
        _revision(operation_fields=(), matrix_operations=()),
        previous_contracts=_contracts(),
        current_contracts=_contracts(),
    )
    assert unobserved.broad


def test_unreconstructible_ledger_is_broad():
    classification = classify_plan_transition(
        _key(), _key(), ledger_reconstructible=False
    )
    assert classification.broad
    assert "ledger" in classification.reason


def test_legacy_unversioned_current_key_is_broad():
    classification = classify_plan_transition(_key(), _key(version=None))
    assert classification.broad


def test_key_change_without_a_revision_is_broad():
    classification = classify_plan_transition(_key(), _key(plan="plan-2"))
    assert classification.broad
    assert "no authenticated revision" in classification.reason


def test_classification_kind_is_validated():
    with pytest.raises(AgentLoopError):
        PlanTransitionClassification("owner-scoped", "nope")


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def test_compatibility_default_selects_every_reviewer_and_avoids_nothing():
    decision = select_plan_reviewers(
        _snapshot(contract=_contract("all-reviewers", primary=None)), _recheck()
    )
    assert decision.selected_reviewers == BOARD
    assert decision.phase == "full-board"
    assert decision.paused_reviewers == ()
    assert decision.calls_avoided == 0
    assert not decision.records_panel_opening


def test_primary_phase_invokes_only_the_primary():
    decision = select_plan_reviewers(_snapshot(), _recheck())
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.calls_avoided == len(SECONDARIES)
    assert not decision.records_panel_opening
    for secondary in SECONDARIES:
        assert "waits for an exact-plan primary approval" in _paused(decision)[secondary]


def test_primary_rechecks_its_own_findings_without_latching_the_board():
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key(), obligations=(_obligation(),)), _recheck()
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.active_owners == (PRIMARY,)
    assert "primary rechecks its plan findings" in decision.reason
    assert STRICT_PRE_PANEL_PREFIX not in decision.reason


def test_exact_key_primary_approval_opens_the_independent_panel():
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key()), _recheck(), qualifying_approvals=(PRIMARY,)
    )
    assert decision.selected_reviewers == SECONDARIES
    assert decision.phase == "secondary-audit"
    assert decision.records_panel_opening
    assert "exact-plan primary approval" in decision.reason


def test_premature_secondary_approval_never_shrinks_the_first_panel():
    """A secondary cannot hold a qualified approval before the panel opens."""
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key()),
        _recheck(),
        # "Claude" holds a stored exact-key approval predating any opening.
        qualifying_approvals=(PRIMARY, "Claude"),
    )
    assert decision.phase == "secondary-audit"
    assert set(decision.selected_reviewers) == set(SECONDARIES)
    assert decision.records_panel_opening
    # The premature approval is named as unqualified, never silently honored.
    assert "Claude" in decision.reason
    assert "do not shrink this audit" in decision.reason
    # Only the primary is paused, on its own approval that opened the panel.
    assert tuple(name for name, _ in decision.paused_reviewers) == (PRIMARY,)
    assert "carried" in _paused(decision)[PRIMARY]
    # The premature approval is not counted as a call avoided either.
    assert decision.calls_avoided == 0


@pytest.mark.parametrize("premature", [("Claude",), ("Antigravity",), SECONDARIES])
def test_premature_secondary_approvals_are_ignored_in_the_primary_phase(premature):
    """Before an opening they change neither the board nor the pause reasons."""
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key()), _recheck(), qualifying_approvals=premature
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert not decision.records_panel_opening
    for name in premature:
        assert "waits for an exact-plan primary approval" in _paused(decision)[name]
        assert "carried" not in _paused(decision)[name]


def test_a_post_opening_secondary_approval_is_still_honored():
    """The exclusion is narrow: after a qualified opening the carry applies."""
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            panel_evidence=True,
        ),
        _recheck(),
        qualifying_approvals=(PRIMARY, "Claude"),
    )
    assert decision.selected_reviewers == ("Antigravity",)
    assert decision.phase == "final-secondary-sweep"
    assert "qualifying exact-plan approval carried" in _paused(decision)["Claude"]


def test_narrow_remediation_after_the_panel_routes_to_owners_plus_primary():
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(plan="plan-2"),
            panel_evidence=True,
            obligations=(_obligation(owners=("Claude",)),),
        ),
        PlanTransitionClassification("narrow", "narrow plan remediation"),
    )
    assert set(decision.selected_reviewers) == {PRIMARY, "Claude"}
    assert decision.phase == "remediation"
    assert decision.reason.startswith(POST_PANEL_PREFIX)
    assert decision.active_owners == ("Claude",)
    assert "ledger owner" in _paused(decision)["Antigravity"]
    assert not decision.latches_force_full


def test_broad_revision_after_the_panel_selects_the_complete_board():
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(plan="plan-2"),
            panel_evidence=True,
        ),
        PlanTransitionClassification("broad", "cross-cutting plan contract(s) changed"),
    )
    assert decision.selected_reviewers == BOARD
    assert decision.phase == "full-board"
    assert decision.reason.startswith(POST_PANEL_PREFIX)
    assert decision.latches_force_full


def test_post_panel_broad_latch_survives_a_later_narrow_transition():
    """The post-panel broad board must not narrow back on the next transition."""
    broad = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(plan="plan-2"),
            panel_evidence=True,
        ),
        PlanTransitionClassification("broad", "cross-cutting plan contract(s) changed"),
    )
    assert broad.selected_reviewers == BOARD
    assert broad.latches_force_full

    # A stage-2 consumer persists the advertised latch and replays it on the
    # next round, whose revision is a clean narrow remediation with an active
    # secondary-owned finding that would otherwise select owners plus primary.
    latched = PlanSchedulerSnapshot(
        contract=_contract(),
        previous_key=_key(plan="plan-2"),
        current_key=_key(plan="plan-3"),
        panel_evidence=True,
        force_full=broad.latches_force_full,
        force_full_source="automatic",
        obligations=(_obligation(owners=("Claude",)),),
    )
    following = select_plan_reviewers(
        latched, PlanTransitionClassification("narrow", "narrow plan remediation")
    )
    assert following.selected_reviewers == BOARD
    assert following.phase == "full-board"
    assert following.latches_force_full
    assert following.reason.startswith(POST_PANEL_PREFIX)


def test_final_exact_plan_sweep_selects_every_reviewer_missing_an_approval():
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(), previous_key=_key(), current_key=_key(), panel_evidence=True
        ),
        _recheck(),
        qualifying_approvals=(PRIMARY, "Claude"),
    )
    assert decision.selected_reviewers == ("Antigravity",)
    assert decision.phase == "final-secondary-sweep"
    assert decision.final_sweep
    assert "qualifying exact-plan approval carried" in _paused(decision)["Claude"]


def test_outstanding_primary_approval_after_the_panel_returns_to_remediation():
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(), previous_key=_key(), current_key=_key(), panel_evidence=True
        ),
        _recheck(),
        qualifying_approvals=("Claude",),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "remediation"


def test_unavailable_reviewer_stays_required_and_is_never_waived():
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(), previous_key=_key(), current_key=_key(), panel_evidence=True
        ),
        _recheck(),
        qualifying_approvals=(PRIMARY,),
        unavailable_reviewers=("Antigravity",),
    )
    assert decision.selected_reviewers == ("Claude",)
    assert "required plan approval remains outstanding" in _paused(decision)["Antigravity"]


def test_unavailable_primary_selects_no_reviewer_in_the_primary_phase():
    decision = select_plan_reviewers(
        _snapshot(), _recheck(), unavailable_reviewers=(PRIMARY,)
    )
    assert decision.selected_reviewers == ()
    assert decision.phase == "primary"


def test_phase_checkpoint_is_validated_but_never_opens_the_panel():
    with pytest.raises(AgentLoopError, match="phase checkpoint"):
        select_plan_reviewers(_snapshot(), _recheck(), phase="panel-open")
    decision = select_plan_reviewers(_snapshot(), _recheck(), phase="secondary-audit")
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"


# --------------------------------------------------------------------------
# Degraded-history partition
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,kwargs,expected",
    [
        ("valid", {}, PLAN_HISTORY_INTACT),
        ("absent", {}, PLAN_HISTORY_ABSENT),
        (None, {}, PLAN_HISTORY_ABSENT),
        ("invalid", {}, PLAN_HISTORY_INVALID),
        ("valid", {"key_contradiction": True}, PLAN_HISTORY_CONTRADICTORY_KEY),
        ("valid", {"transport_failure": True}, PLAN_HISTORY_TRANSPORT_FAILURE),
        ("invalid", {"transport_failure": True}, PLAN_HISTORY_TRANSPORT_FAILURE),
    ],
)
def test_history_classification_is_a_disjoint_partition(status, kwargs, expected):
    assert classify_plan_history(status, **kwargs) == expected


def test_history_classification_rejects_an_unknown_status():
    with pytest.raises(AgentLoopError):
        classify_plan_history("stale")


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_falls_back_to_the_primary_before_an_opening(history_class):
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            degraded_history_class=history_class,
        ),
        _recheck(),
        qualifying_approvals=(PRIMARY,),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert plan_history_fallback_reason(history_class) in decision.reason
    assert not decision.latches_force_full
    assert not decision.records_panel_opening
    assert plan_history_continues(history_class)


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_after_an_opening_latches_the_complete_board(history_class):
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            panel_evidence=True,
            degraded_history_class=history_class,
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == BOARD
    assert decision.phase == "full-board"
    assert decision.reason.startswith(POST_PANEL_PREFIX)
    assert decision.latches_force_full


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_outranks_a_secondary_owned_finding(history_class):
    """A readable degraded class continues; it never stops for ownership doubt."""
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            degraded_history_class=history_class,
            obligations=(_obligation(owners=("Claude",)),),
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert plan_history_fallback_reason(history_class) in decision.reason
    assert "not authoritative ownership" in decision.reason
    assert not decision.latches_force_full
    assert not decision.records_panel_opening


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_outranks_a_premature_secondary_review(history_class):
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            degraded_history_class=history_class,
            premature_secondary_reviews=("Claude",),
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert plan_history_fallback_reason(history_class) in decision.reason
    assert "unqualified artifacts" in decision.reason
    assert not decision.records_panel_opening


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_outranks_both_ownership_ambiguities_together(history_class):
    decision = select_plan_reviewers(
        PlanSchedulerSnapshot(
            contract=_contract(),
            previous_key=_key(),
            current_key=_key(),
            degraded_history_class=history_class,
            premature_secondary_reviews=("Antigravity",),
            obligations=(_obligation(owners=("Claude",)),),
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert plan_history_continues(history_class)
    assert "unqualified artifacts" in decision.reason
    assert "not authoritative ownership" in decision.reason


def test_intact_history_still_stops_for_both_ownership_ambiguities():
    """Precedence is narrow: intact history keeps both diagnostic stops."""
    with pytest.raises(PlanPrePanelSafetyError, match="pending on"):
        select_plan_reviewers(
            _snapshot(
                previous_key=_key(), obligations=(_obligation(owners=("Claude",)),)
            ),
            _recheck(),
        )
    with pytest.raises(PlanPrePanelSafetyError, match="premature blocking plan review"):
        select_plan_reviewers(
            _snapshot(previous_key=_key(), premature_secondary_reviews=("Claude",)),
            _recheck(),
        )


@pytest.mark.parametrize(
    "history_class",
    [PLAN_HISTORY_ABSENT, PLAN_HISTORY_INVALID, PLAN_HISTORY_CONTRADICTORY_KEY],
)
def test_degraded_history_never_both_continues_and_stops(history_class):
    """One durable history, one outcome, whatever else the snapshot carries."""
    for extra in (
        {},
        {"obligations": (_obligation(owners=("Claude",)),)},
        {"premature_secondary_reviews": ("Claude",)},
        {
            "obligations": (_obligation(owners=("Claude",)),),
            "premature_secondary_reviews": ("Antigravity",),
        },
        {"force_full": True, "force_full_source": "automatic"},
    ):
        decision = select_plan_reviewers(
            PlanSchedulerSnapshot(
                contract=_contract(),
                previous_key=_key(),
                current_key=_key(),
                degraded_history_class=history_class,
                **extra,
            ),
            _recheck(),
        )
        assert decision.selected_reviewers == (PRIMARY,)
        assert decision.phase == "primary"


def test_intact_history_has_no_fallback_reason():
    assert plan_history_fallback_reason(PLAN_HISTORY_INTACT) is None
    with pytest.raises(AgentLoopError):
        plan_history_fallback_reason("unknown-class")


def test_automatic_latch_is_monotonic_once_the_panel_is_open():
    snapshot = PlanSchedulerSnapshot(
        contract=_contract(),
        previous_key=_key(),
        current_key=_key(),
        panel_evidence=True,
        force_full=True,
        force_full_source="automatic",
    )
    first = select_plan_reviewers(snapshot, _recheck())
    assert first.selected_reviewers == BOARD
    assert first.latches_force_full
    # Replaying the latched snapshot on a clean narrow transition keeps the
    # complete board rather than silently narrowing again.
    second = select_plan_reviewers(
        snapshot, PlanTransitionClassification("narrow", "narrow plan remediation")
    )
    assert second.selected_reviewers == BOARD
    assert second.phase == "full-board"
    assert second.latches_force_full


def test_automatic_force_full_before_an_opening_stays_primary_only():
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key(), force_full=True, force_full_source="automatic"),
        _recheck(),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"
    assert decision.reason.startswith(STRICT_PRE_PANEL_PREFIX)
    assert not decision.latches_force_full


def test_snapshot_rejects_an_unknown_force_full_source():
    with pytest.raises(AgentLoopError, match="force-full source"):
        PlanSchedulerSnapshot(contract=_contract(), force_full_source="reviewer")


def test_snapshot_rejects_an_unknown_history_class():
    with pytest.raises(AgentLoopError, match="planning history class"):
        PlanSchedulerSnapshot(contract=_contract(), degraded_history_class="stale")


# --------------------------------------------------------------------------
# Stop conditions and the operator override
# --------------------------------------------------------------------------


def test_secondary_owned_finding_without_an_opening_stops():
    with pytest.raises(PlanPrePanelSafetyError) as excinfo:
        select_plan_reviewers(
            _snapshot(
                previous_key=_key(), obligations=(_obligation(owners=("Claude",)),)
            ),
            _recheck(),
        )
    message = str(excinfo.value)
    assert "item-1" in message and "Claude" in message
    assert "--plan-review-force-full" in message


def test_orchestrator_authored_finding_with_no_reviewer_owner_does_not_stop():
    decision = select_plan_reviewers(
        _snapshot(
            previous_key=_key(),
            obligations=(_obligation(owners=("Orchestrator",)),),
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == (PRIMARY,)
    assert decision.phase == "primary"


def test_premature_blocking_secondary_review_without_an_opening_stops():
    with pytest.raises(PlanPrePanelSafetyError, match="premature blocking plan review"):
        select_plan_reviewers(
            _snapshot(previous_key=_key(), premature_secondary_reviews=("Claude",)),
            _recheck(),
        )


def test_transport_extraction_failure_always_stops():
    with pytest.raises(PlanPrePanelSafetyError) as excinfo:
        select_plan_reviewers(
            _snapshot(
                degraded_history_class=PLAN_HISTORY_TRANSPORT_FAILURE,
                transport_error=RuntimeError("comment 7 is truncated"),
            ),
            _recheck(),
        )
    message = str(excinfo.value)
    assert "could not be extracted" in message
    assert "comment 7 is truncated" in message
    assert "cannot authorize" in message
    assert not plan_history_continues(PLAN_HISTORY_TRANSPORT_FAILURE)


def test_operator_override_does_not_recover_a_transport_failure():
    with pytest.raises(PlanPrePanelSafetyError, match="cannot authorize"):
        select_plan_reviewers(
            _snapshot(
                degraded_history_class=PLAN_HISTORY_TRANSPORT_FAILURE,
                operator_force_full=True,
            ),
            _recheck(),
        )


def test_operator_override_recovers_the_secondary_owned_finding():
    decision = select_plan_reviewers(
        _snapshot(
            previous_key=_key(),
            obligations=(_obligation(owners=("Claude",)),),
            operator_force_full=True,
        ),
        _recheck(),
    )
    assert decision.selected_reviewers == BOARD
    assert decision.phase == "full-board"
    assert decision.reason.startswith(OPERATOR_PLAN_FORCE_FULL_REASON)
    assert decision.records_panel_opening


def test_operator_override_recovers_and_supersedes_a_premature_review():
    decision = select_plan_reviewers(
        _snapshot(previous_key=_key(), premature_secondary_reviews=("Claude",), operator_force_full=True),
        _recheck(),
    )
    assert decision.selected_reviewers == BOARD
    assert "Claude" in decision.reason
    assert "non-authoritative context" in decision.reason
    assert decision.records_panel_opening


def test_module_is_pure_and_imports_no_github_or_agent_module():
    """The decision surface may only depend on errors and the shared primitives."""
    import ast
    import pathlib

    import coding_review_agent_loop.plan_review_scheduling as module

    source = pathlib.Path(module.__file__).read_text()
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                imported.add(node.module or "")
            else:
                imported.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".")[0])
    assert {"errors", "review_scheduling"} <= imported
    assert not imported & {
        "github",
        "agents",
        "orchestrator",
        "round_state",
        "subprocess",
        "requests",
    }
