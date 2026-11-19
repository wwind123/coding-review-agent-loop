import json

import pytest

from agent_loop_helpers import structured_coder_followup
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.protocol import (
    validate_human_requirement_dispositions,
    validate_structured_plan_state,
)
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.response_validation as response_validation


def _plan(dispositions):
    payload = {
        "schema_version": 1,
        "kind": "plan_state",
        "state": "blocking",
        "summary": "Plan the requested integration.",
        "plan_steps": ["Add the Grafana dashboard provisioning artifact."],
        "human_requirement_dispositions": dispositions,
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- coder"


def test_signed_requirement_requires_exact_structured_disposition():
    parsed = validate_structured_plan_state(
        _plan(
            [{
                "requirement_id": "Requirement 1",
                "disposition": "addressed",
                "evidence": "The plan names Grafana dashboard provisioning.",
            }]
        )
    )
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=("Requirement 1",),
    )


@pytest.mark.parametrize(
    "dispositions, message",
    [
        ([], "missing"),
        ([{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": "x"},
          {"requirement_id": "Requirement 1", "disposition": "blocked", "evidence": "y"}], "duplicate"),
        ([{"requirement_id": "Requirement 2", "disposition": "addressed", "evidence": "x"}], "unknown"),
    ],
)
def test_invalid_requirement_coverage_is_rejected(dispositions, message):
    parsed = validate_structured_plan_state(_plan(dispositions))
    with pytest.raises(AgentLoopError, match=message):
        validate_human_requirement_dispositions(
            parsed.human_requirement_dispositions,
            surfaced_requirement_ids=("Requirement 1",),
        )


def test_invalid_status_and_empty_evidence_are_rejected_at_schema_boundary():
    with pytest.raises(AgentLoopError, match="disposition"):
        validate_structured_plan_state(
            _plan([{"requirement_id": "Requirement 1", "disposition": "maybe", "evidence": "x"}])
        )
    with pytest.raises(AgentLoopError, match="evidence"):
        validate_structured_plan_state(
            _plan([{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": ""}])
        )


def test_no_signed_requirements_requires_empty_collection():
    parsed = validate_structured_plan_state(_plan([]))
    validate_human_requirement_dispositions(
        parsed.human_requirement_dispositions,
        surfaced_requirement_ids=(),
    )


def test_initial_plan_validator_checks_coder_dispositions():
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at=None,
        url=None,
        body="Provide Grafana.",
    )
    with pytest.raises(AgentLoopError, match="missing requirement ID"):
        response_validation._validate_response_with_human_requirements(
            _plan([]).replace(
                "\n<!-- AGENT_PLAN_STATE: blocking -->",
                "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
                f"- Requirement {requirement.requirement_id}: Grafana is planned.\n"
                "<!-- AGENT_PLAN_STATE: blocking -->",
            ),
            marker_validator=lambda text: response_validation._require_plan_state_or_clarification(
                text, architecture_status_mode="legacy"
            ),
            human_requirements=(requirement,),
            requirement_scope="planning requirements",
            full_omission_fallback="Fetch the discussion.",
        )


def test_current_plan_gate_rejects_missing_coder_dispositions():
    assert not response_validation._current_plan_has_complete_human_requirement_dispositions(
        _plan([]),
        surfaced_requirement_ids=("Requirement 1",),
    )


def test_ack_gate_requires_dedicated_coder_disposition_evidence():
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="reviewer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Please use the absolute URL.",
    )
    output = structured_coder_followup(
        addressed_items=[],
        human_requirement_ids=[],
        human_requirement_dispositions=[],
    )

    with pytest.raises(AgentLoopError, match="missing"):
        validate_human_requirement_dispositions(
            [],
            surfaced_requirement_ids=(requirement.requirement_id,),
            context="coder_followup.human_requirement_dispositions",
        )
    assert '"human_requirement_dispositions": []' in output


# ---------------------------------------------------------------------------
# #905 (from #841): a carried plan approval cannot satisfy the gate vacuously


def _carried_plan_record(*, agent, state, key, requirement_ids, index):
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata,
        PostedRoundRecord,
    )

    return PostedRoundRecord(
        index=index,
        body=f"{agent} plan review.",
        metadata=PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent=agent,
            round_number=1,
            subject=key.subject,
            state=state,
            plan_candidate_key=key.as_dict(),
            surfaced_reviewer_requirement_ids=tuple(requirement_ids),
        ),
    )


def _plan_key(requirement_ids):
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanCandidateKey,
        surfaced_requirement_id_digest,
    )

    return PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity="c" * 32,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest=surfaced_requirement_id_digest(requirement_ids),
    )


def test_carried_plan_approval_requires_the_current_requirement_acknowledgement():
    """`carried-approval-requires-current-ack`."""
    evidence = panel_evidence.PlanPanelEvidence(opening_index=0, opening_source="operator")
    key = _plan_key(("req-1",))

    acknowledged = panel_evidence._carried_plan_approvals(
        (
            _carried_plan_record(
                agent="Gemini", state="approved", key=key,
                requirement_ids=("req-1",), index=1,
            ),
        ),
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=("req-1",),
        panel_evidence=evidence,
        primary_reviewer="Codex",
    )
    assert acknowledged == ("Gemini",)

    # The stored approval carried no acknowledgement: it cannot be repaired in
    # place, so the reviewer must be re-invoked.
    unacknowledged = panel_evidence._carried_plan_approvals(
        (
            _carried_plan_record(
                agent="Gemini", state="approved", key=key,
                requirement_ids=(), index=1,
            ),
        ),
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=("req-1",),
        panel_evidence=evidence,
        primary_reviewer="Codex",
    )
    assert unacknowledged == ()


def test_a_changed_surfaced_requirement_set_invalidates_every_carried_plan_approval():
    """`plan-identity-change-invalidates-approvals` and
    `human-requirements-and-decomposition`: an added, edited, replaced, or
    withdrawn requirement changes the key's requirement digest."""
    evidence = panel_evidence.PlanPanelEvidence(opening_index=0, opening_source="operator")
    stored_key = _plan_key(("req-1",))
    records = (
        _carried_plan_record(
            agent="Gemini", state="approved", key=stored_key,
            requirement_ids=("req-1",), index=1,
        ),
    )

    for surfaced in (("req-1", "req-2"), ("req-2",), ()):
        current_key = _plan_key(surfaced)
        assert current_key.components != stored_key.components
        assert (
            panel_evidence._carried_plan_approvals(
                records,
                current_key=current_key,
                required_reviewers=("Codex", "Gemini"),
                surfaced_requirement_ids=surfaced,
                panel_evidence=evidence,
                primary_reviewer="Codex",
            )
            == ()
        )


def test_a_premature_secondary_plan_approval_is_never_carried():
    """A secondary approval before any qualified opening is unqualified."""
    key = _plan_key(())
    records = (
        _carried_plan_record(
            agent="Gemini", state="approved", key=key, requirement_ids=(), index=1
        ),
        _carried_plan_record(
            agent="Codex", state="approved", key=key, requirement_ids=(), index=2
        ),
    )

    carried = panel_evidence._carried_plan_approvals(
        records,
        current_key=key,
        required_reviewers=("Codex", "Gemini"),
        surfaced_requirement_ids=(),
        panel_evidence=panel_evidence.PlanPanelEvidence(),
        primary_reviewer="Codex",
    )

    # Only the primary can hold a pre-opening approval.
    assert carried == ("Codex",)


# --- #1241: surfaced-gated `requirement_label` alias on plan_review ---------

import logging

from coding_review_agent_loop.protocol import parse_plan_review, parse_structured_plan_review
from coding_review_agent_loop.unresolved_items import _validate_plan_review_response

_HR_ID = "hr-" + "ab12" * 16
_OTHER_HR_ID = "hr-" + "cd34" * 16
_ALIAS_LOG = "normalized disposition key alias requirement_label -> requirement_id"


def _plan_review(disposition_entry):
    payload = {
        "schema_version": 1,
        "kind": "plan_review",
        "state": "approved",
        "summary": "Plan looks good.",
        "prior_plan_item_dispositions": [],
        "human_requirement_dispositions": [disposition_entry],
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- reviewer"


def _aliased(value):
    return {"requirement_label": value, "disposition": "addressed", "evidence": "Step 2 covers it."}


def _validate(text, surfaced):
    return _validate_plan_review_response(
        text,
        reviewer="Codex",
        unresolved_items=(),
        surfaced_requirement_ids=surfaced,
        architecture_status_mode="legacy",
    )


def test_plan_review_alias_with_surfaced_id_is_accepted_and_logged(caplog):
    text = _plan_review(_aliased(_HR_ID))
    with caplog.at_level(logging.WARNING):
        parsed = _validate(text, (_HR_ID,))
    assert [d.requirement_id for d in parsed.human_requirement_dispositions] == [_HR_ID]
    assert parsed.human_requirement_dispositions[0].evidence == "Step 2 covers it."
    assert _ALIAS_LOG in caplog.text


def test_parse_structured_plan_review_alias_with_surfaced_ids():
    parsed = parse_structured_plan_review(
        _plan_review(_aliased(_HR_ID)),
        reviewer="Codex",
        architecture_status_mode="legacy",
        surfaced_requirement_ids=(_HR_ID,),
    )
    assert parsed.human_requirement_dispositions[0].requirement_id == _HR_ID


@pytest.mark.parametrize(
    "value, surfaced",
    [
        (_OTHER_HR_ID, (_HR_ID,)),
        (f" {_HR_ID} ", (_HR_ID,)),
        (f"{_HR_ID}\n", (_HR_ID,)),
        ("hr-" + ("AB12" * 16), (_HR_ID,)),
        ("Requirement 1", (_HR_ID,)),
        ("Requirement 1", ("Requirement 1",)),
        (123, (_HR_ID,)),
    ],
)
def test_plan_review_alias_rejected_for_inexact_or_unsurfaced_value(value, surfaced, caplog):
    with caplog.at_level(logging.WARNING):
        with pytest.raises(AgentLoopError, match="requirement_id"):
            _validate(_plan_review(_aliased(value)), surfaced)
    assert _ALIAS_LOG not in caplog.text


def test_plan_review_alias_with_both_keys_is_rejected():
    entry = _aliased(_HR_ID)
    entry["requirement_id"] = _HR_ID
    with pytest.raises(AgentLoopError, match="requirement_label"):
        _validate(_plan_review(entry), (_HR_ID,))


def test_plan_review_alias_not_honored_without_surfaced_set():
    with pytest.raises(AgentLoopError, match="requirement_id"):
        parse_plan_review(
            _plan_review(_aliased(_HR_ID)), reviewer="Codex", architecture_status_mode="legacy"
        )


def test_plan_state_alias_is_not_honored():
    with pytest.raises(AgentLoopError, match="requirement_id"):
        validate_structured_plan_state(_plan([_aliased(_HR_ID)]))


# --- Reviewer-owned signed requirements are outside coder scope (#2020) ---

from coding_review_agent_loop.protocol import (  # noqa: E402
    HumanRequirementDisposition,
    validate_human_requirements_acknowledgement,
    validate_structured_human_requirements_acknowledgement,
)

_CODER_ID = "hr-" + "a" * 64
_OWNED_ID = "hr-" + "b" * 64
_UNKNOWN_ID = "hr-" + "c" * 64


def _disposition(requirement_id):
    return HumanRequirementDisposition(requirement_id, "addressed", "Covered.")


@pytest.mark.parametrize("surfaced", [(_CODER_ID,), ()])
def test_disposition_naming_reviewer_owned_id_gets_out_of_scope_message(surfaced):
    dispositions = [_disposition(item) for item in (*surfaced, _OWNED_ID)]
    with pytest.raises(AgentLoopError) as error:
        validate_human_requirement_dispositions(
            dispositions,
            surfaced_requirement_ids=surfaced,
            context="coder_followup.human_requirement_dispositions",
            reviewer_owned_requirement_ids=(_OWNED_ID.upper().replace("HR-", "hr-"),),
        )
    message = str(error.value)
    assert _OWNED_ID in message
    assert "outside coder scope" in message
    assert "reviewer-owned exact-head evidence" in message
    assert "unknown requirement ID" not in message
    assert "must be empty when no signed human requirements" not in message


def test_truly_unknown_disposition_id_keeps_unknown_message():
    with pytest.raises(AgentLoopError, match="contains unknown requirement ID") as error:
        validate_human_requirement_dispositions(
            [_disposition(_CODER_ID), _disposition(_UNKNOWN_ID)],
            surfaced_requirement_ids=(_CODER_ID,),
            context="coder_followup.human_requirement_dispositions",
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )
    assert "outside coder scope" not in str(error.value)
    assert _UNKNOWN_ID in str(error.value)


def test_mixed_reviewer_owned_and_unknown_disposition_ids_report_both():
    with pytest.raises(AgentLoopError) as error:
        validate_human_requirement_dispositions(
            [_disposition(_CODER_ID), _disposition(_OWNED_ID), _disposition(_UNKNOWN_ID)],
            surfaced_requirement_ids=(_CODER_ID,),
            context="coder_followup.human_requirement_dispositions",
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )
    message = str(error.value)
    assert "outside coder scope" in message
    assert f"also contains unknown requirement ID(s): {_UNKNOWN_ID}" in message
    assert message.index(_OWNED_ID) < message.index("also contains unknown")


def test_coder_scoped_dispositions_still_validate_with_reviewer_owned_ids():
    validate_human_requirement_dispositions(
        [_disposition(_CODER_ID)],
        surfaced_requirement_ids=(_CODER_ID,),
        reviewer_owned_requirement_ids=(_OWNED_ID,),
    )
    validate_human_requirement_dispositions(
        [], surfaced_requirement_ids=(), reviewer_owned_requirement_ids=(_OWNED_ID,),
    )


def test_all_reviewer_owned_addressed_ids_get_out_of_scope_message():
    """Early-return branch: nothing surfaced, no direct-discussion acknowledgement."""
    with pytest.raises(AgentLoopError) as error:
        validate_structured_human_requirements_acknowledgement(
            [_OWNED_ID],
            dispositions=(),
            checked_discussion_directly=False,
            surfaced_requirement_ids=(),
            requires_direct_discussion_ack=False,
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )
    message = str(error.value)
    assert "outside coder scope" in message
    assert _OWNED_ID in message
    assert "no signed human requirements were surfaced" not in message
    assert "arbitrary labels" not in message
    assert "unknown" not in message


def test_all_reviewer_owned_empty_acknowledgement_validates():
    validate_structured_human_requirements_acknowledgement(
        [],
        dispositions=(),
        checked_discussion_directly=False,
        surfaced_requirement_ids=(),
        requires_direct_discussion_ack=False,
        reviewer_owned_requirement_ids=(_OWNED_ID,),
    )


def test_early_return_branch_keeps_generic_text_for_non_owned_labels():
    with pytest.raises(AgentLoopError, match="even though no signed human requirements were surfaced"):
        validate_structured_human_requirements_acknowledgement(
            [_UNKNOWN_ID],
            dispositions=(),
            checked_discussion_directly=False,
            surfaced_requirement_ids=(),
            requires_direct_discussion_ack=False,
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )


def test_partially_surfaced_reviewer_owned_addressed_id_uses_general_branch():
    with pytest.raises(AgentLoopError) as error:
        validate_structured_human_requirements_acknowledgement(
            [_CODER_ID, _OWNED_ID],
            dispositions=(_disposition(_CODER_ID),),
            checked_discussion_directly=False,
            surfaced_requirement_ids=(_CODER_ID,),
            requires_direct_discussion_ack=False,
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )
    message = str(error.value)
    assert "outside coder scope" in message
    assert "referenced unknown signed human requirement IDs" not in message
    with pytest.raises(AgentLoopError, match="referenced unknown signed human requirement IDs"):
        validate_structured_human_requirements_acknowledgement(
            [_CODER_ID, _UNKNOWN_ID],
            dispositions=(_disposition(_CODER_ID),),
            checked_discussion_directly=False,
            surfaced_requirement_ids=(_CODER_ID,),
            requires_direct_discussion_ack=False,
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )


def test_legacy_markdown_acknowledgement_forwards_reviewer_owned_ids():
    text = (
        "Updated.\n\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
        f"- {_CODER_ID}: done.\n- {_OWNED_ID}: done.\n"
    )
    with pytest.raises(AgentLoopError, match="outside coder scope"):
        validate_human_requirements_acknowledgement(
            text,
            surfaced_requirement_ids=(_CODER_ID,),
            requires_direct_discussion_ack=False,
            reviewer_owned_requirement_ids=(_OWNED_ID,),
        )
