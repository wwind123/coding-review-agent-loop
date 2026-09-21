"""Stage A unit tests for the cross-surface workflow transaction model (#827)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent_loop_helpers import make_config
from workflow_transaction_helpers import (
    ACTOR,
    FOREIGN,
    HEAD_1,
    HEAD_2,
    ISSUE,
    PLAN,
    PLAN_HASH,
    PLAN_SUBJECT,
    PR,
    REPO,
    comment,
    direct_intent,
    issue_view,
    plan_intent,
    plan_key,
    v2_contract_comment,
    v2_handoff_comment,
    plan_record_comment,
    plan_reviewer_comment,
    pr_review_comment,
    pr_view,
    prepared_comment,
    record_set,
    scheduler_comment,
    terminal_comment,
    v1_contract,
    v1_contract_comment,
    v1_handoff_comment,
    with_created_at,
)

from coding_review_agent_loop.errors import AgentLoopError, WorkflowTransactionError
from coding_review_agent_loop.expected_closure import contract_hash
from coding_review_agent_loop.github import (
    AuthenticatedComment,
    AuthenticatedCommentView,
    read_authenticated_protocol_comments,
)
from coding_review_agent_loop.issue_pr_handoff import (
    decode_issue_pr_handoff_v2,
    encode_issue_pr_handoff_v2,
    issue_pr_handoff_record_hash,
    resolve_issue_pr_handoff_lineage,
)
from coding_review_agent_loop.pr_contract import (
    decode_pr_contract_v2,
    encode_pr_contract_v2,
    pr_contract_record_hash,
)
from coding_review_agent_loop.round_state import _approved_plan_hash, _plan_subject
from coding_review_agent_loop.workflow_transaction import (
    ABORT_STALE_HEAD,
    ABORT_SUPERSEDED_INTENT,
    ENTRY_AUTHORIZATION,
    ENTRY_HANDOFF,
    ENTRY_INITIAL_CODER_ROUND,
    ENTRY_PR_CONTRACT,
    ERA_LEGACY,
    ERA_TRANSACTION,
    FLOW_APPROVED_PLAN,
    KIND_CLOSING_WIDENING,
    KIND_FLOW_CORRECTION,
    KIND_HEAD_ADVANCE,
    KIND_LEGACY_ROOT_CORRECTION,
    KIND_MANAGED_CI_CONTINUITY,
    KIND_PLAN_REPLACEMENT,
    PHASE_ABORTED,
    CommentRef,
    LegacyRoot,
    LegacyRootContext,
    OriginEvidence,
    SchedulerCheckpointRef,
    StagedIdentity,
    TransitionInputs,
    WorkflowTransactionRecord,
    WorkflowTransition,
    actor_change_error,
    classify_transaction_era,
    collect_transactions,
    comment_order,
    compare_prepared_intent,
    decode_transaction_record,
    derive_handoff_metadata,
    derive_pr_contract,
    encode_transaction_record,
    find_origin_evidence,
    format_transaction_record_comment,
    inherited,
    not_applicable,
    plan_successor,
    prepared_record,
    reissued,
    resolve_approved_plan_anchor,
    resolve_handoff_lineage,
    resolve_pr_contract_lineage,
    resolve_transaction_lineage,
    round_metadata_digest,
    select_scheduler_checkpoint,
    successor_kind_for,
    validate_legacy_root,
    verify_scheduler_checkpoint,
)

DIGEST = "d" * 64
PR_SURFACE = f"pr#{PR}"
ISSUE_SURFACE = f"issue#{ISSUE}"


def _resolve(*comments, **kwargs):
    return resolve_transaction_lineage(
        pr_view(*comments), repository=REPO, pr_number=PR, **kwargs
    )


def _committed_initial(intent=None, *, prepared_id=10, terminal_id=20):
    intent = intent or direct_intent()
    published = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
    return intent, [
        prepared_comment(prepared_id, intent),
        terminal_comment(terminal_id, intent, prepared_id=prepared_id, published=published),
    ]


def _head_advance(predecessor, **overrides):
    fields = dict(
        head_sha=HEAD_2,
        successor_kind=KIND_HEAD_ADVANCE,
        predecessor_transaction_id=predecessor.transaction_id,
        record_set=record_set(
            handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
            contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, DIGEST)),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    fields.update(overrides)
    return replace(predecessor, **fields)


# --- derive-single-source ---------------------------------------------------


@pytest.mark.parametrize("intent", [direct_intent(), plan_intent(expected_closing_issue_ids=(ISSUE, 900))])
def test_handoff_and_contract_derive_from_one_intent_and_cannot_disagree(intent):
    handoff = derive_handoff_metadata(intent)
    contract = derive_pr_contract(intent)

    assert handoff.flow == contract.origin_flow == intent.origin_flow
    assert handoff.issue_number == contract.primary_issue_number == intent.primary_issue
    assert handoff.plan_hash == intent.approved_plan_hash
    assert (
        handoff.expected_closing_issue_ids
        == contract.expected_closing_issue_ids
        == intent.expected_closing_issue_ids
    )
    assert handoff.contract_hash == contract.contract_hash == contract_hash(
        intent.expected_closing_issue_ids
    )
    assert handoff.transaction_id == contract.transaction_id == intent.transaction_id
    assert handoff.pr_number == contract.pr_number == intent.pr_number
    assert handoff.pr_head_sha == intent.head_sha
    # Both payloads round-trip through the strict version-2 codecs.
    assert decode_issue_pr_handoff_v2(encode_issue_pr_handoff_v2(handoff)) == handoff
    assert decode_pr_contract_v2(encode_pr_contract_v2(contract)) == contract


def test_derivation_accepts_no_flow_string_from_the_call_site():
    with pytest.raises(TypeError):
        derive_handoff_metadata(direct_intent(), flow="approved-plan-implementation")
    with pytest.raises(TypeError):
        derive_pr_contract(direct_intent(), origin_flow="approved-plan-implementation")


# --- lineage-id-binding -----------------------------------------------------


def _legacy_root(contract_id=5, handoff=None, evidence_id=7):
    return LegacyRoot(
        contract=CommentRef(PR_SURFACE, contract_id, DIGEST),
        handoff=handoff,
        origin_evidence=OriginEvidence(
            kind="pr-round-metadata",
            plan_hash=PLAN_HASH,
            plan_subject=PLAN_SUBJECT,
            reference=CommentRef(PR_SURFACE, evidence_id, DIGEST),
        ),
    )


def _legacy_intent(**root_overrides):
    return plan_intent(
        successor_kind=KIND_LEGACY_ROOT_CORRECTION,
        legacy_root=_legacy_root(**root_overrides),
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )


def test_transaction_id_binds_every_lineage_field():
    base, _ = _committed_initial()
    successor = _head_advance(base)
    changed = [
        replace(successor, predecessor_transaction_id="f" * 64),
        replace(
            successor,
            successor_kind=KIND_MANAGED_CI_CONTINUITY,
            managed_ci_generation="g1",
            record_set=record_set(
                handoff=successor.entry(ENTRY_HANDOFF),
                contract=successor.entry(ENTRY_PR_CONTRACT),
                authorization=reissued(ENTRY_AUTHORIZATION),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        ),
        replace(successor, schema_version=2),
        replace(successor, head_sha="c" * 40),
    ]
    ids = {item.transaction_id for item in changed} | {successor.transaction_id}
    assert len(ids) == len(changed) + 1


def test_transaction_id_binds_comment_ids_inside_preexisting_references():
    checkpoint = plan_intent(
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 3, DIGEST))
    )
    moved_checkpoint = replace(
        checkpoint,
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST)),
    )
    assert checkpoint.transaction_id != moved_checkpoint.transaction_id

    legacy = _legacy_intent()
    for other in (
        _legacy_intent(contract_id=6),
        _legacy_intent(evidence_id=8),
        _legacy_intent(handoff=CommentRef(ISSUE_SURFACE, 2, DIGEST)),
    ):
        assert other.transaction_id != legacy.transaction_id
    asserted = replace(
        legacy,
        legacy_root=replace(
            legacy.legacy_root,
            origin_evidence=OriginEvidence(
                kind="operator-asserted",
                plan_hash=PLAN_HASH,
                plan_subject=PLAN_SUBJECT,
                operator_login=ACTOR[0],
                operator_id=ACTOR[1],
            ),
        ),
    )
    assert asserted.transaction_id != legacy.transaction_id

    base, _ = _committed_initial()
    successor = _head_advance(base)
    moved_inherited = _head_advance(
        base,
        record_set=record_set(
            handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 111, DIGEST)),
            contract=successor.entry(ENTRY_PR_CONTRACT),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    assert moved_inherited.transaction_id != successor.transaction_id


def test_transaction_id_excludes_publication_results_phase_status_and_writer():
    intent = direct_intent()
    first = prepared_record(intent, writer_login="one", writer_id=1)
    second = prepared_record(intent, writer_login="two", writer_id=2)
    assert first.transaction_id == second.transaction_id == intent.transaction_id

    published_a = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
    published_b = {ENTRY_HANDOFF: 91, ENTRY_PR_CONTRACT: 92, ENTRY_INITIAL_CODER_ROUND: 93}
    committed = terminal_comment(20, intent, prepared_id=10, published=published_a)
    recommitted = terminal_comment(21, intent, prepared_id=15, published=published_b)
    aborted = terminal_comment(
        22, intent, prepared_id=10, phase=PHASE_ABORTED, abort_reason=ABORT_STALE_HEAD
    )
    for body in (committed.body, recommitted.body, aborted.body):
        assert intent.transaction_id in body

    # The canonical intent serialization has no slot for a publication result.
    payload = json.dumps(intent.to_payload())
    assert "prepared_comment_id" not in payload and "writer" not in payload
    assert "phase" not in intent.to_payload() and "status" not in intent.to_payload()
    for entry in intent.to_payload()["record_set"]:
        if entry["disposition"] == "reissued":
            assert entry["inherited"] is None
    with pytest.raises(AgentLoopError, match="exactly when it is inherited"):
        WorkflowTransition.from_payload(
            {
                **intent.to_payload(),
                "record_set": [
                    {**entry, "inherited": {"surface": PR_SURFACE, "comment_id": 9, "digest": DIGEST}}
                    if entry["name"] == ENTRY_PR_CONTRACT
                    else entry
                    for entry in intent.to_payload()["record_set"]
                ],
            }
        )


def test_intent_round_trips_and_rejects_non_canonical_payloads():
    for intent in (direct_intent(), _legacy_intent(), _head_advance(direct_intent())):
        assert WorkflowTransition.from_payload(intent.to_payload()) == intent
    payload = direct_intent().to_payload()
    with pytest.raises(AgentLoopError):
        WorkflowTransition.from_payload({**payload, "comment_id": 5})
    with pytest.raises(AgentLoopError, match="not canonical"):
        WorkflowTransition.from_payload({**payload, "expected_closing_issue_ids": [ISSUE, ISSUE]})


# --- lineage shapes and intent validation -----------------------------------


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"predecessor_transaction_id": "f" * 64}, "neither predecessor nor legacy root"),
        ({"successor_kind": KIND_HEAD_ADVANCE}, "has a predecessor and no legacy root"),
        ({"successor_kind": KIND_LEGACY_ROOT_CORRECTION}, "legacy root and no predecessor"),
        ({"successor_kind": "sideways"}, "unknown successor kind"),
        ({"head_sha": "abc123"}, "full lowercase commit SHA"),
        ({"head_sha": "A" * 40}, "full lowercase commit SHA"),
        ({"approved_plan_hash": PLAN_HASH}, "must not carry an approved-plan hash"),
        ({"expected_closing_issue_ids": (900,)}, "retain primary issue"),
        (
            {"scheduler_checkpoint": SchedulerCheckpointRef(absence_reason="no-plan-scheduler-records")},
            "flow-without-plan-review",
        ),
        (
            {"staged": StagedIdentity(ISSUE + 1, ISSUE, "parent"), "expected_closing_issue_ids": (ISSUE, ISSUE + 1)},
            "staged parent must not be part",
        ),
    ],
)
def test_invalid_intents_are_rejected(overrides, match):
    with pytest.raises(AgentLoopError, match=match):
        direct_intent(**overrides)


def test_approved_plan_flow_requires_plan_hash_and_a_true_checkpoint_shape():
    with pytest.raises(AgentLoopError, match="requires an approved-plan hash"):
        plan_intent(approved_plan_hash=None)
    with pytest.raises(AgentLoopError, match="cannot claim it had no plan review"):
        plan_intent(
            scheduler_checkpoint=SchedulerCheckpointRef(absence_reason="flow-without-plan-review")
        )
    with pytest.raises(AgentLoopError, match="exactly one of a reference or an absence reason"):
        SchedulerCheckpointRef()
    with pytest.raises(AgentLoopError, match="unknown scheduler checkpoint absence reason"):
        SchedulerCheckpointRef(absence_reason="legacy-unversioned")
    with pytest.raises(AgentLoopError, match="SHA-256 digest"):
        CommentRef(ISSUE_SURFACE, 3, "")
    with pytest.raises(AgentLoopError, match="only name an issue-side record"):
        SchedulerCheckpointRef(CommentRef(PR_SURFACE, 3, DIGEST))
    with pytest.raises(AgentLoopError, match="plan-owning issue"):
        plan_intent(
            scheduler_checkpoint=SchedulerCheckpointRef(CommentRef("issue#1", 3, DIGEST))
        )


def test_legacy_root_requires_one_of_the_two_origin_evidence_kinds():
    with pytest.raises(AgentLoopError, match="unknown origin evidence kind"):
        OriginEvidence(kind="issue-phase-handoff", plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT)
    with pytest.raises(AgentLoopError, match="needs a comment reference"):
        OriginEvidence(kind="pr-round-metadata", plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT)
    with pytest.raises(AgentLoopError, match="needs the operator actor"):
        OriginEvidence(kind="operator-asserted", plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT)
    with pytest.raises(AgentLoopError, match="requires origin evidence"):
        LegacyRoot(CommentRef(PR_SURFACE, 5, DIGEST), None, None)
    payload = _legacy_intent().to_payload()
    payload["legacy_root"]["origin_evidence"] = None
    with pytest.raises(AgentLoopError):
        WorkflowTransition.from_payload(payload)
    with pytest.raises(AgentLoopError, match="origin-evidence plan hash"):
        plan_intent(
            successor_kind=KIND_LEGACY_ROOT_CORRECTION,
            approved_plan_hash="0" * 16,
            legacy_root=_legacy_root(),
            record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
        )


def test_record_set_typing_is_validated_per_successor_kind():
    base = direct_intent()
    ref = CommentRef(PR_SURFACE, 12, DIGEST)
    with pytest.raises(AgentLoopError, match="cannot inherit"):
        direct_intent(record_set=record_set(contract=inherited(ENTRY_PR_CONTRACT, ref)))
    with pytest.raises(AgentLoopError, match="cannot inherit"):
        plan_intent(
            successor_kind=KIND_LEGACY_ROOT_CORRECTION,
            legacy_root=_legacy_root(),
            record_set=record_set(
                contract=inherited(ENTRY_PR_CONTRACT, ref),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )
    with pytest.raises(AgentLoopError, match="must not reissue"):
        _head_advance(base, record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)))
    with pytest.raises(AgentLoopError, match="must reissue the managed-CI authorization"):
        _head_advance(
            base,
            managed_ci_generation="g1",
            record_set=record_set(
                handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
                contract=inherited(ENTRY_PR_CONTRACT, ref),
                authorization=inherited(ENTRY_AUTHORIZATION, CommentRef(PR_SURFACE, 14, DIGEST)),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )
    with pytest.raises(AgentLoopError, match="only a first transaction"):
        _head_advance(
            base,
            record_set=record_set(
                handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
                contract=inherited(ENTRY_PR_CONTRACT, ref),
            ),
        )
    with pytest.raises(AgentLoopError, match="wrong comment thread"):
        _head_advance(
            base,
            record_set=record_set(
                handoff=inherited(ENTRY_HANDOFF, CommentRef(PR_SURFACE, 11, DIGEST)),
                contract=inherited(ENTRY_PR_CONTRACT, ref),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )
    with pytest.raises(AgentLoopError, match="declare exactly"):
        direct_intent(record_set=record_set()[:2])


# --- transaction record codec -----------------------------------------------


def test_prepared_record_round_trips_the_complete_intent():
    intent = _legacy_intent()
    record = prepared_record(intent, writer_login=ACTOR[0], writer_id=ACTOR[1])
    decoded = decode_transaction_record(encode_transaction_record(record))
    assert decoded == record
    assert decoded.intent == intent
    assert decoded.intent.transaction_id == decoded.transaction_id


def test_prepared_record_whose_intent_does_not_hash_to_its_id_is_rejected():
    intent = direct_intent()
    with pytest.raises(AgentLoopError, match="does not hash to transaction"):
        WorkflowTransactionRecord(
            phase="prepared",
            transaction_id="0" * 64,
            intent=intent,
            writer_login=ACTOR[0],
            writer_id=ACTOR[1],
        )
    # Tampering with the encoded payload is caught by the same rule.
    import base64

    payload = prepared_record(intent, writer_login=ACTOR[0], writer_id=ACTOR[1]).to_payload()
    payload["intent"]["head_sha"] = HEAD_2
    forged = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    with pytest.raises(AgentLoopError, match="does not hash to transaction"):
        decode_transaction_record(forged)


def test_terminal_record_parsing_is_strict():
    intent = direct_intent()
    with pytest.raises(AgentLoopError, match="outcome for every record-set entry"):
        WorkflowTransactionRecord(
            phase="committed", transaction_id=intent.transaction_id, prepared_comment_id=10
        )
    with pytest.raises(AgentLoopError, match="binds its prepared comment ID"):
        WorkflowTransactionRecord(phase="committed", transaction_id=intent.transaction_id)
    with pytest.raises(AgentLoopError, match="unknown abort reason"):
        terminal_comment(20, intent, prepared_id=10, phase=PHASE_ABORTED, abort_reason="bored")
    with pytest.raises(AgentLoopError, match="unexpected fields"):
        import base64

        decode_transaction_record(
            base64.urlsafe_b64encode(
                json.dumps(
                    {"schema_version": 1, "phase": "committed", "transaction_id": "0" * 64}
                ).encode()
            ).decode()
        )


def test_transaction_record_comment_is_a_trusted_pr_comment_only_body():
    body = format_transaction_record_comment(
        prepared_record(direct_intent(), writer_login=ACTOR[0], writer_id=ACTOR[1])
    )
    body.validate_for_surface("pr_comment")
    with pytest.raises(AgentLoopError, match="not allowed on the issue_comment surface"):
        body.validate_for_surface("issue_comment")


# --- lineage resolution -----------------------------------------------------


def test_lineage_resolves_prepared_committed_and_successor_chain():
    base, comments = _committed_initial()
    pending = _resolve(comments[0])
    assert pending.latest_committed is None
    assert pending.pending.transaction_id == base.transaction_id

    committed = _resolve(*comments)
    assert committed.pending is None
    assert committed.latest_committed.transaction_id == base.transaction_id

    successor = _head_advance(base)
    lineage = _resolve(*comments, prepared_comment(30, successor))
    assert [item.transaction_id for item in lineage.chain] == [
        base.transaction_id,
        successor.transaction_id,
    ]
    assert lineage.latest_committed.transaction_id == base.transaction_id
    assert lineage.pending.transaction_id == successor.transaction_id


def test_divergent_same_scope_transactions_fail_closed_naming_both_ids():
    first = direct_intent()
    second = direct_intent(head_sha=HEAD_2)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(prepared_comment(10, first), prepared_comment(11, second))
    message = str(excinfo.value)
    assert first.transaction_id in message and second.transaction_id in message
    assert excinfo.value.code == "divergent-transactions"
    assert set(excinfo.value.transaction_ids) == {first.transaction_id, second.transaction_id}
    assert "Recovery:" in message and "Expected record set:" in message


def test_aborted_sibling_is_inert_and_does_not_make_the_lineage_divergent():
    loser = direct_intent(head_sha=HEAD_2)
    winner, comments = _committed_initial()
    lineage = _resolve(
        *comments,
        prepared_comment(15, loser),
        terminal_comment(
            25, loser, prepared_id=15, phase=PHASE_ABORTED, abort_reason="sibling-canonical"
        ),
    )
    assert [item.transaction_id for item in lineage.chain] == [winner.transaction_id]
    assert lineage.state(loser.transaction_id).aborted


def test_two_live_children_and_reparenting_are_rejected():
    base, comments = _committed_initial()
    first = _head_advance(base)
    second = _head_advance(base, head_sha="c" * 40)
    with pytest.raises(WorkflowTransactionError, match="more than one non-aborted successor"):
        _resolve(*comments, prepared_comment(30, first), prepared_comment(31, second))
    # A successor cannot be re-parented: its ID binds its predecessor, and a
    # predecessor that is not a committed transaction of this PR is refused.
    orphan = _head_advance(base, predecessor_transaction_id="f" * 64)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(30, orphan))
    assert excinfo.value.code == "uncommitted-predecessor"


def test_successor_over_uncommitted_predecessor_is_rejected():
    base, comments = _committed_initial()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(comments[0], prepared_comment(30, _head_advance(base)))
    assert excinfo.value.code == "uncommitted-predecessor"


def test_terminal_without_prepared_record_fails_closed():
    base, comments = _committed_initial()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(comments[1])
    assert excinfo.value.code == "terminal-without-prepared"
    with pytest.raises(WorkflowTransactionError):
        _resolve(
            comments[0],
            terminal_comment(
                20,
                base,
                prepared_id=999,
                published={ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
            ),
        )


def test_contradictory_terminal_records_fail_closed_and_duplicates_canonicalize():
    base, comments = _committed_initial()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(
            *comments,
            terminal_comment(
                21, base, prepared_id=10, phase=PHASE_ABORTED, abort_reason=ABORT_STALE_HEAD
            ),
        )
    assert excinfo.value.code == "contradictory-terminal"

    duplicate = prepared_comment(14, base)
    states = collect_transactions(
        pr_view(*comments, duplicate), repository=REPO, pr_number=PR
    )
    assert len(states) == 1
    assert states[0].prepared_comment.comment_id == 10
    assert [item.comment_id for item in states[0].prepared_duplicates] == [14]


def test_two_legacy_roots_and_a_legacy_root_beside_a_committed_transaction_fail_closed():
    _base, comments = _committed_initial()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(30, _legacy_intent()))
    assert excinfo.value.code == "divergent-transactions"
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(
            prepared_comment(30, _legacy_intent()),
            prepared_comment(31, _legacy_intent(contract_id=6)),
        )
    assert excinfo.value.code == "divergent-transactions"


def test_inherited_entry_must_name_the_predecessor_chains_canonical_record():
    base, comments = _committed_initial()
    wrong = _head_advance(
        base,
        record_set=record_set(
            handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
            contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 999, DIGEST)),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(30, wrong))
    assert excinfo.value.code == "inherited-mismatch"


def test_prepared_record_from_another_pr_or_with_a_foreign_writer_name_is_rejected():
    other_pr = direct_intent(pr_number=PR + 1)
    with pytest.raises(WorkflowTransactionError, match="belongs to another PR"):
        _resolve(prepared_comment(10, other_pr))
    body = format_transaction_record_comment(
        prepared_record(direct_intent(), writer_login="someone-else", writer_id=1)
    )
    with pytest.raises(WorkflowTransactionError, match="writer other than its comment author"):
        _resolve(comment(10, body))


def test_forged_exact_records_are_ignored_and_actor_change_names_the_prior_writer():
    base, comments = _committed_initial()
    forged = [replace(item, author_login=FOREIGN[0], author_id=FOREIGN[1]) for item in comments]
    view = pr_view(*forged)
    lineage = resolve_transaction_lineage(view, repository=REPO, pr_number=PR)
    assert lineage.transactions == () and lineage.chain == ()
    assert len(lineage.ignored_foreign) == 2
    assert classify_transaction_era(view) == ERA_LEGACY

    error = actor_change_error(view)
    assert error is not None and error.code == "actor-change"
    assert FOREIGN[0] in str(error) and str(FOREIGN[1]) in str(error)
    assert "rerun under the original GitHub actor" in str(error)
    assert base.transaction_id in str(error)
    assert actor_change_error(pr_view(*comments)) is None


def test_lineage_entry_points_reject_unauthenticated_snapshots():
    snapshot = [SimpleNamespace(body=prepared_comment(10, direct_intent()).body)]
    for call in (
        lambda: resolve_transaction_lineage(snapshot, repository=REPO, pr_number=PR),
        lambda: classify_transaction_era(snapshot),
        lambda: resolve_approved_plan_anchor(snapshot, plan_hash=PLAN_HASH),
        lambda: resolve_handoff_lineage(snapshot, None, repository=REPO, issue_number=ISSUE),
        lambda: find_origin_evidence(snapshot),
    ):
        with pytest.raises(AgentLoopError, match="only an authenticated comment view"):
            call()
    with pytest.raises(AgentLoopError, match="not the expected"):
        resolve_transaction_lineage(issue_view(), repository=REPO, pr_number=PR)


# --- era classification -----------------------------------------------------


def test_any_authenticated_v2_record_makes_the_pr_transaction_era():
    from workflow_transaction_helpers import v2_contract_comment, v2_handoff_comment

    intent = direct_intent()
    assert classify_transaction_era(pr_view(v1_contract_comment(5))) == ERA_LEGACY
    # The transaction records were deleted or hidden: still transaction-era.
    assert classify_transaction_era(pr_view(v2_contract_comment(12, intent))) == ERA_TRANSACTION
    assert (
        classify_transaction_era(pr_view(), issue_view(v2_handoff_comment(11, intent)))
        == ERA_TRANSACTION
    )
    assert classify_transaction_era(pr_view(prepared_comment(10, intent))) == ERA_TRANSACTION
    # A forged version-2 record from another author never changes the era.
    assert (
        classify_transaction_era(pr_view(v2_contract_comment(12, intent, author=FOREIGN)))
        == ERA_LEGACY
    )


# --- ordering helper --------------------------------------------------------


def test_ordering_rule_orders_same_second_pairs_by_id_and_fails_on_reversal():
    first = comment(10, "a", second=5)
    same_second = comment(11, "b", second=5)
    later = comment(12, "c", second=9)
    assert comment_order(first, same_second) == "later"
    assert comment_order(same_second, first) == "earlier"
    assert comment_order(first, later) == "later"
    assert comment_order(first, first) == "same"
    assert comment_order(first, replace(first, body="edited")) == "same"
    # Greater ID with an earlier created_at is a reversal, never "later".
    assert comment_order(later, comment(13, "d", second=1)) == "unordered"
    for broken in (
        SimpleNamespace(comment_id=None, created_at="2026-09-21T10:00:00Z"),
        SimpleNamespace(comment_id=14, created_at=None),
        SimpleNamespace(comment_id=14, created_at="yesterday"),
        SimpleNamespace(comment_id=True, created_at="2026-09-21T10:00:00Z"),
        SimpleNamespace(comment_id=14, created_at="2026-09-21T10:00:00"),
    ):
        assert comment_order(first, broken) == "unordered"
        assert comment_order(broken, first) == "unordered"


@pytest.mark.parametrize(
    "overrides",
    [
        {"comment_id": None},
        {"comment_id": "12"},
        {"comment_id": True},
        {"created_at": None},
        {"created_at": "not-a-time"},
        {"author_id": None},
        {"author_login": ""},
        {"surface": "discussion#1"},
    ],
)
def test_envelope_without_identity_or_ordering_fields_is_rejected(overrides):
    fields = dict(
        surface=PR_SURFACE,
        comment_id=12,
        author_login="bot",
        author_id=1,
        created_at="2026-09-21T10:00:00Z",
        updated_at=None,
        body="",
    )
    fields.update(overrides)
    with pytest.raises(AgentLoopError):
        AuthenticatedComment(**fields)


def test_view_cannot_present_a_foreign_comment_as_authored():
    foreign = comment(10, "x", author=FOREIGN)
    with pytest.raises(AgentLoopError, match="author partition"):
        AuthenticatedCommentView(PR_SURFACE, ACTOR[0], ACTOR[1], authored=(foreign,))
    with pytest.raises(AgentLoopError, match="only authenticated envelopes"):
        AuthenticatedCommentView(
            PR_SURFACE, ACTOR[0], ACTOR[1], authored=(SimpleNamespace(body="x"),)
        )


# --- approved-plan anchor ---------------------------------------------------


def test_anchor_is_the_earliest_matching_plan_record_and_duplicates_resolve_to_it():
    view = issue_view(plan_record_comment(3), plan_record_comment(6), plan_reviewer_comment(4))
    anchor = resolve_approved_plan_anchor(view, plan_hash=PLAN_HASH)
    assert anchor.comment.comment_id == 3
    assert anchor.plan_subject == PLAN_SUBJECT and anchor.canonical_text == PLAN
    assert (
        resolve_approved_plan_anchor(view, plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT)
        == anchor
    )


def test_anchor_fails_closed_when_missing_foreign_or_unordered():
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve_approved_plan_anchor(issue_view(plan_record_comment(3, "Other plan.")), plan_hash=PLAN_HASH)
    assert excinfo.value.code == "approved-plan-anchor-missing"
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve_approved_plan_anchor(
            issue_view(plan_record_comment(3, author=FOREIGN)), plan_hash=PLAN_HASH
        )
    assert "ignored-foreign comment 3" in str(excinfo.value)
    # Duplicate records whose mutual order is a timestamp reversal.
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve_approved_plan_anchor(
            issue_view(plan_record_comment(3, second=30), plan_record_comment(6, second=2)),
            plan_hash=PLAN_HASH,
        )
    assert excinfo.value.code == "approved-plan-anchor-unordered"
    with pytest.raises(WorkflowTransactionError):
        resolve_approved_plan_anchor(
            issue_view(plan_record_comment(3)), plan_hash=PLAN_HASH, plan_subject="0" * 64
        )


def test_anchor_fails_closed_when_same_hash_records_carry_differing_text(monkeypatch):
    import coding_review_agent_loop.workflow_transaction as module

    monkeypatch.setattr(module, "_approved_plan_hash", lambda text: PLAN_HASH)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve_approved_plan_anchor(
            issue_view(plan_record_comment(3), plan_record_comment(6, "Differing text.")),
            plan_hash=PLAN_HASH,
        )
    assert excinfo.value.code == "approved-plan-anchor-divergent"


# --- scheduler checkpoint selector ------------------------------------------

EARLIER_PLAN = "Earlier candidate that reviewers sent back."


def _select(view, **overrides):
    fields = dict(
        origin_flow=FLOW_APPROVED_PLAN,
        plan_hash=PLAN_HASH,
        plan_subject=PLAN_SUBJECT,
        plan_candidate_key=plan_key(),
    )
    fields.update(overrides)
    return select_scheduler_checkpoint(view, **fields)


def test_first_round_approval_binds_that_rounds_checkpoint_not_an_absence_reason():
    view = issue_view(plan_record_comment(3), scheduler_comment(4), plan_reviewer_comment(5))
    chosen = _select(view)
    assert chosen.absence_reason is None
    assert chosen.reference == CommentRef(
        ISSUE_SURFACE, 4, round_metadata_digest(view.comment(4))
    )


def test_revised_candidate_binds_its_own_checkpoint_never_the_preceding_candidates():
    view = issue_view(
        plan_record_comment(1, EARLIER_PLAN),
        scheduler_comment(2, EARLIER_PLAN),
        plan_record_comment(3),
        scheduler_comment(4, round_number=2),
    )
    assert _select(view).reference.comment_id == 4


def test_several_same_subject_checkpoints_bind_the_earliest_and_later_records_change_nothing():
    base = [plan_record_comment(3), scheduler_comment(4)]
    before = _select(issue_view(*base))
    after = _select(
        issue_view(
            *base,
            scheduler_comment(8, round_number=2),
            scheduler_comment(9, round_number=3),
            plan_record_comment(10, EARLIER_PLAN),
            scheduler_comment(11, EARLIER_PLAN),
        )
    )
    assert before == after and before.reference.comment_id == 4
    intent_before = plan_intent(scheduler_checkpoint=before)
    assert intent_before.transaction_id == plan_intent(scheduler_checkpoint=after).transaction_id


def test_duplicate_plan_records_anchor_on_the_earliest_so_its_checkpoint_is_bound():
    view = issue_view(plan_record_comment(3), scheduler_comment(4), plan_record_comment(6), scheduler_comment(7))
    assert _select(view).reference.comment_id == 4


def test_checkpoint_in_the_same_second_as_its_candidate_is_bound_by_id_order():
    view = issue_view(plan_record_comment(3, second=7), scheduler_comment(4, second=7))
    assert _select(view).reference.comment_id == 4


def test_records_at_or_before_the_anchor_and_other_subjects_are_never_selected():
    view = issue_view(
        scheduler_comment(2),  # same subject, but before the anchor
        plan_record_comment(3),
        scheduler_comment(4, EARLIER_PLAN),
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _select(view)
    assert excinfo.value.code == "scheduler-checkpoint-unmatched"
    assert "rerun plan review" in str(excinfo.value)


def test_checkpoint_with_greater_id_but_earlier_timestamp_is_unordered_failure():
    view = issue_view(plan_record_comment(3, second=50), scheduler_comment(4, second=1))
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _select(view)
    assert excinfo.value.code == "scheduler-checkpoint-unordered"


def test_absence_reasons_are_mechanical_and_identical_for_both_schedulerless_histories():
    assert (
        select_scheduler_checkpoint(None, origin_flow="issue-implementation").absence_reason
        == "flow-without-plan-review"
    )
    assert (
        select_scheduler_checkpoint(None, origin_flow="managed-pr").absence_reason
        == "flow-without-plan-review"
    )
    # An all-reviewers run and genuinely old metadata both decode with
    # scheduler metadata absent; nothing durable distinguishes them.
    all_reviewers = _select(issue_view(plan_record_comment(3), plan_reviewer_comment(4)))
    old_metadata = _select(issue_view(plan_record_comment(3)))
    assert all_reviewers == old_metadata
    assert all_reviewers.absence_reason == "no-plan-scheduler-records"
    assert (
        plan_intent(scheduler_checkpoint=all_reviewers).transaction_id
        == plan_intent(scheduler_checkpoint=old_metadata).transaction_id
    )


def test_invalid_scheduler_metadata_stops_the_selector(monkeypatch):
    import coding_review_agent_loop.workflow_transaction as module

    real = module._plan_records

    def invalid(view):
        return tuple(
            (
                replace(record, metadata=replace(record.metadata, scheduler_metadata_status="invalid"))
                if record.metadata.role == "summary"
                else record,
                item,
            )
            for record, item in real(view)
        )

    monkeypatch.setattr(module, "_plan_records", invalid)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _select(issue_view(plan_record_comment(3), scheduler_comment(4)))
    assert excinfo.value.code == "scheduler-metadata-invalid"


def test_staged_direct_child_resolves_anchor_and_checkpoint_on_the_parent_issue():
    parent = 827
    staged = StagedIdentity(parent_issue=parent, child_issue=ISSUE, plan_owner="parent")
    parent_view = issue_view(
        plan_record_comment(3, number=parent), scheduler_comment(4, number=parent), number=parent
    )
    chosen = _select(parent_view)
    assert chosen.reference.surface == f"issue#{parent}"
    intent = plan_intent(staged=staged, scheduler_checkpoint=chosen)
    assert intent.plan_owning_issue == parent
    verify_scheduler_checkpoint(intent, parent_view, plan_candidate_key=plan_key())
    # The child issue is not where a direct-implementation child's plan lives.
    with pytest.raises(AgentLoopError, match="plan-owning issue"):
        plan_intent(
            staged=staged,
            scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST)),
        )
    own_plan = plan_intent(staged=replace(staged, plan_owner="child"))
    assert own_plan.plan_owning_issue == ISSUE


def test_pr_side_scheduler_record_cannot_be_the_reference():
    with pytest.raises(AgentLoopError, match="only name an issue-side record"):
        SchedulerCheckpointRef(CommentRef(PR_SURFACE, 4, DIGEST))
    with pytest.raises(AgentLoopError, match="only an authenticated comment view|not the expected"):
        _select(pr_view(scheduler_comment(4, surface=PR_SURFACE)))


def test_gate_reverification_fails_closed_on_missing_foreign_altered_or_wrong_checkpoint():
    comments = [plan_record_comment(3), scheduler_comment(4), scheduler_comment(8, EARLIER_PLAN)]
    view = issue_view(*comments)
    intent = plan_intent(scheduler_checkpoint=_select(view))
    verify_scheduler_checkpoint(intent, view, plan_candidate_key=plan_key())

    def code(bad_view, bad_intent=intent):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            verify_scheduler_checkpoint(bad_intent, bad_view, plan_candidate_key=plan_key())
        return str(excinfo.value)

    assert "missing scheduler checkpoint comment 4" in code(issue_view(comments[0]))
    foreign_copy = replace(comments[1], author_login=FOREIGN[0], author_id=FOREIGN[1])
    assert "ignored-foreign scheduler checkpoint comment 4" in code(
        issue_view(comments[0], foreign_copy)
    )
    altered = replace(scheduler_comment(4, round_number=9))
    assert "contradictory digest" in code(issue_view(comments[0], altered))
    wrong_subject = replace(
        intent,
        scheduler_checkpoint=SchedulerCheckpointRef(
            CommentRef(ISSUE_SURFACE, 8, round_metadata_digest(comments[2]))
        ),
    )
    assert "not a scheduler-prelaunch record for the approved plan's subject" in code(
        view, wrong_subject
    )
    before_anchor = issue_view(scheduler_comment(2), plan_record_comment(3))
    early = replace(
        intent,
        scheduler_checkpoint=SchedulerCheckpointRef(
            CommentRef(ISSUE_SURFACE, 2, round_metadata_digest(before_anchor.comment(2)))
        ),
    )
    assert "not strictly later than anchor" in code(before_anchor, early)



KEY_VARIANTS = (
    {"aggregate_plan_identity": "other-aggregate"},
    {"execution_strategy_identity": "other-strategy"},
    {"risk_test_matrix_identity": "other-matrix"},
    {"surfaced_requirement_id_digest": "other-requirements"},
)


@pytest.mark.parametrize("variant", KEY_VARIANTS)
def test_same_subject_checkpoint_with_a_differing_key_component_is_never_bound(variant):
    other = scheduler_comment(4, key=plan_key(**variant))
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _select(issue_view(plan_record_comment(3), other))
    assert excinfo.value.code == "scheduler-checkpoint-unmatched"
    # The exact candidate's own later checkpoint is bound instead of the earlier
    # same-subject one.
    view = issue_view(plan_record_comment(3), other, scheduler_comment(5, round_number=2))
    assert _select(view).reference.comment_id == 5
    assert _select(view, plan_candidate_key=plan_key(**variant)).reference.comment_id == 4


@pytest.mark.parametrize("variant", KEY_VARIANTS)
def test_gate_reverification_rejects_a_same_subject_checkpoint_with_a_differing_key(variant):
    view = issue_view(plan_record_comment(3), scheduler_comment(4, key=plan_key(**variant)))
    intent = plan_intent(
        scheduler_checkpoint=SchedulerCheckpointRef(
            CommentRef(ISSUE_SURFACE, 4, round_metadata_digest(view.comment(4)))
        )
    )
    verify_scheduler_checkpoint(intent, view, plan_candidate_key=plan_key(**variant))
    with pytest.raises(WorkflowTransactionError) as excinfo:
        verify_scheduler_checkpoint(intent, view, plan_candidate_key=plan_key())
    assert excinfo.value.code == "scheduler-checkpoint-invalid"
    assert "not the approved plan's complete candidate key" in str(excinfo.value)


def test_candidate_key_must_be_supplied_complete_and_for_the_anchor_subject():
    view = issue_view(plan_record_comment(3), scheduler_comment(4))
    with pytest.raises(AgentLoopError, match="requires the approved plan's candidate key"):
        _select(view, plan_candidate_key=None)
    with pytest.raises(AgentLoopError, match="incomplete"):
        _select(view, plan_candidate_key=plan_key(risk_test_matrix_identity=None))
    with pytest.raises(AgentLoopError, match="incomplete"):
        _select(view, plan_candidate_key=plan_key(execution_strategy_contract_version=None))
    with pytest.raises(AgentLoopError, match="different plan subject"):
        _select(view, plan_candidate_key=plan_key(EARLIER_PLAN))
    intent = plan_intent(scheduler_checkpoint=_select(view))
    with pytest.raises(AgentLoopError, match="requires the approved plan's candidate key"):
        verify_scheduler_checkpoint(intent, view)


def test_checkpoint_without_a_decodable_candidate_key_is_never_bound(monkeypatch):
    import coding_review_agent_loop.workflow_transaction as module

    real = module._plan_records
    for broken in (None, {"subject": 7}):
        monkeypatch.setattr(
            module,
            "_plan_records",
            lambda view, broken=broken: tuple(
                (
                    replace(record, metadata=replace(record.metadata, plan_candidate_key=broken))
                    if record.metadata.role == "summary"
                    else record,
                    item,
                )
                for record, item in real(view)
            ),
        )
        with pytest.raises(WorkflowTransactionError) as excinfo:
            _select(issue_view(plan_record_comment(3), scheduler_comment(4)))
        assert excinfo.value.code == "scheduler-checkpoint-unmatched"


# --- legacy root validation --------------------------------------------------


def _legacy_state(*, evidence=None, extra_pr=(), plan_comments=None, contract=None):
    contract_comment = contract or v1_contract_comment(105)
    evidence_comment = evidence or pr_review_comment(107)
    pr = pr_view(contract_comment, evidence_comment, *extra_pr)
    plan = issue_view(*(plan_comments or [plan_record_comment(3)]))
    found = find_origin_evidence(pr)
    intent = plan_intent(
        successor_kind=KIND_LEGACY_ROOT_CORRECTION,
        legacy_root=LegacyRoot(
            contract=CommentRef(PR_SURFACE, 105, pr_contract_record_hash(v1_contract())),
            handoff=None,
            origin_evidence=found,
        ),
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )
    return intent, pr, plan


def _validate(intent, pr, plan, commits=(HEAD_1, HEAD_2)):
    validate_legacy_root(intent, pr_view=pr, plan_issue_view=plan, pr_commit_shas=commits)


def test_legacy_root_with_authenticated_pr_round_metadata_evidence_validates():
    intent, pr, plan = _legacy_state()
    _validate(intent, pr, plan)
    assert intent.legacy_root.origin_evidence.plan_hash == PLAN_HASH
    lineage = resolve_transaction_lineage(
        pr_view(*pr.authored, prepared_comment(110, intent)),
        repository=REPO,
        pr_number=PR,
        legacy_root_context=LegacyRootContext(plan, (HEAD_1,)),
    )
    assert lineage.pending.transaction_id == intent.transaction_id
    # Without the authenticated context the legacy root is never accepted.
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*pr.authored, prepared_comment(110, intent))
    assert excinfo.value.code == "legacy-root-unvalidated"


def test_reviewer_records_for_different_heads_that_agree_on_the_plan_still_match():
    intent, pr, plan = _legacy_state(extra_pr=(pr_review_comment(108, head=HEAD_2),))
    _validate(intent, pr, plan)


def test_same_second_evidence_with_a_greater_comment_id_matches():
    intent, pr, plan = _legacy_state(
        evidence=pr_review_comment(107, second=9),
        plan_comments=[plan_record_comment(3, second=9)],
    )
    _validate(intent, pr, plan)


def test_legacy_root_evidence_rejections():
    intent, pr, plan = _legacy_state()

    def refused(**kwargs):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            _validate(**{"intent": intent, "pr": pr, "plan": plan, **kwargs})
        assert excinfo.value.code in {"legacy-root-invalid", "origin-evidence-conflict"}
        return str(excinfo.value)

    assert "is not a commit of PR" in refused(commits=("c" * 40,))
    # Evidence not later than the plan record: timestamp reversal and lower ID.
    assert "not strictly later" in refused(
        plan=issue_view(plan_record_comment(3, second=3000))
    )
    assert "not strictly later" in refused(plan=issue_view(plan_record_comment(300)))
    assert "does not authenticate" in refused(
        plan=issue_view(plan_record_comment(3, author=FOREIGN))
    )
    foreign_evidence = replace(
        pr.comment(107), author_login=FOREIGN[0], author_id=FOREIGN[1]
    )
    assert "ignored-foreign origin-evidence comment 107" in refused(
        pr=pr_view(pr.comment(105), foreign_evidence)
    )
    foreign_contract = replace(pr.comment(105), author_login=FOREIGN[0], author_id=FOREIGN[1])
    assert "ignored-foreign version-1 PR contract comment 105" in refused(
        pr=pr_view(foreign_contract, pr.comment(107))
    )
    changed_contract = v1_contract_comment(105, v1_contract(ids=(ISSUE, 900)))
    assert "no longer matches its recorded hash" in refused(
        pr=pr_view(changed_contract, pr.comment(107))
    )
    other = "Another plan entirely."
    disagreeing = pr_review_comment(
        108, plan_hash=_approved_plan_hash(other), plan_subject=_plan_subject(other)
    )
    assert "contradictory approved-plan identity" in refused(
        pr=pr_view(*pr.authored, disagreeing)
    )
    subject_only = pr_review_comment(108, plan_subject=_plan_subject(other))
    assert "contradictory approved-plan identity" in refused(
        pr=pr_view(*pr.authored, subject_only)
    )


def test_no_evidence_twin_yields_no_origin_evidence_and_conflicts_raise():
    # A legitimate direct-issue PR: reviewer rounds never carried a plan hash.
    twin = pr_view(v1_contract_comment(105), pr_review_comment(107, plan_hash=None, plan_subject=None))
    assert find_origin_evidence(twin) is None
    other = "Another plan entirely."
    with pytest.raises(WorkflowTransactionError) as excinfo:
        find_origin_evidence(
            pr_view(
                pr_review_comment(107),
                pr_review_comment(
                    108, plan_hash=_approved_plan_hash(other), plan_subject=_plan_subject(other)
                ),
            )
        )
    assert excinfo.value.code == "origin-evidence-conflict"


def test_operator_asserted_evidence_needs_the_current_actor_and_an_authenticating_hash():
    _intent, pr, plan = _legacy_state()

    def asserted(plan_hash=PLAN_HASH, subject=PLAN_SUBJECT, operator=ACTOR):
        return plan_intent(
            approved_plan_hash=plan_hash,
            successor_kind=KIND_LEGACY_ROOT_CORRECTION,
            legacy_root=LegacyRoot(
                contract=CommentRef(PR_SURFACE, 105, pr_contract_record_hash(v1_contract())),
                handoff=None,
                origin_evidence=OriginEvidence(
                    kind="operator-asserted",
                    plan_hash=plan_hash,
                    plan_subject=subject,
                    operator_login=operator[0],
                    operator_id=operator[1],
                ),
            ),
            record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
        )

    _validate(asserted(), pr, plan)
    with pytest.raises(WorkflowTransactionError, match="does not authenticate"):
        _validate(asserted(plan_hash="0" * 16), pr, plan)
    with pytest.raises(WorkflowTransactionError, match="contradictory operator assertion"):
        _validate(asserted(operator=FOREIGN), pr, plan)


def test_legacy_root_handoff_reference_is_checked_against_the_issue_conversation():
    handoff_comment = v1_handoff_comment(50)
    v1 = resolve_issue_pr_handoff_lineage(
        (handoff_comment,), issue_number=ISSUE, repo=REPO
    ).latest
    intent, pr, _plan = _legacy_state()
    with_handoff = replace(
        intent,
        legacy_root=replace(
            intent.legacy_root,
            handoff=CommentRef(ISSUE_SURFACE, 50, issue_pr_handoff_record_hash(v1)),
        ),
    )
    plan = issue_view(plan_record_comment(3), handoff_comment)
    _validate(with_handoff, pr, plan)
    with pytest.raises(WorkflowTransactionError, match="missing version-1 handoff comment 50"):
        _validate(with_handoff, pr, issue_view(plan_record_comment(3)))
    wrong = replace(
        with_handoff,
        legacy_root=replace(with_handoff.legacy_root, handoff=CommentRef(ISSUE_SURFACE, 50, DIGEST)),
    )
    with pytest.raises(WorkflowTransactionError, match="no longer matches its recorded hash"):
        _validate(wrong, pr, plan)


# --- obsolete prepared intent and successor planning -------------------------


def _inputs(intent, **overrides):
    return replace(TransitionInputs.of(intent), **overrides)


def test_each_legitimate_difference_makes_the_prepared_intent_obsolete():
    stored = plan_intent(managed_ci_generation="g1", record_set=record_set(authorization=reissued(ENTRY_AUTHORIZATION)))
    assert compare_prepared_intent(stored, _inputs(stored)).outcome == "current"

    head = compare_prepared_intent(stored, _inputs(stored, head_sha=HEAD_2))
    assert (head.outcome, head.abort_reason) == ("obsolete", ABORT_STALE_HEAD)
    assert head.differing_fields == (("head_sha", repr(HEAD_1), repr(HEAD_2)),)

    generation = compare_prepared_intent(stored, _inputs(stored, managed_ci_generation="g2"))
    assert (generation.outcome, generation.abort_reason) == ("obsolete", ABORT_SUPERSEDED_INTENT)

    widened = compare_prepared_intent(
        stored, _inputs(stored, expected_closing_issue_ids=(ISSUE, 900))
    )
    assert (widened.outcome, widened.abort_reason) == ("obsolete", ABORT_SUPERSEDED_INTENT)

    old_anchor, new_anchor = plan_record_comment(3), plan_record_comment(9, "Replacement plan.")
    replaced = compare_prepared_intent(
        stored,
        _inputs(stored, approved_plan_hash="1" * 16),
        stored_plan_anchor=old_anchor,
        fresh_plan_anchor=new_anchor,
    )
    assert (replaced.outcome, replaced.abort_reason) == ("obsolete", ABORT_SUPERSEDED_INTENT)

    both = compare_prepared_intent(
        stored, _inputs(stored, head_sha=HEAD_2, managed_ci_generation="g2")
    )
    assert both.abort_reason == ABORT_SUPERSEDED_INTENT
    assert {name for name, _b, _a in both.differing_fields} == {"head_sha", "managed_ci_generation"}


def test_every_other_difference_is_a_contradiction_that_writes_nothing():
    stored = plan_intent()
    old_anchor, new_anchor = plan_record_comment(3), plan_record_comment(9, "Replacement plan.")
    cases = [
        _inputs(stored, base="release"),
        _inputs(stored, repository="OTHER/REPO"),
        _inputs(stored, pr_number=PR + 1),
        _inputs(stored, staged=StagedIdentity(827, ISSUE, "parent")),
        _inputs(stored, expected_closing_issue_ids=(900,)),
        _inputs(stored, origin_flow="issue-implementation", approved_plan_hash=None),
        # A plan hash that does not authenticate (no anchor supplied).
        _inputs(stored, approved_plan_hash="1" * 16),
    ]
    for fresh in cases:
        result = compare_prepared_intent(stored, fresh)
        assert result.outcome == "contradiction" and result.abort_reason is None
        assert result.contradictions
    # An older plan is not a legitimate replacement.
    older = compare_prepared_intent(
        stored,
        _inputs(stored, approved_plan_hash="1" * 16),
        stored_plan_anchor=new_anchor,
        fresh_plan_anchor=old_anchor,
    )
    assert older.outcome == "contradiction"


def test_flow_correction_to_or_from_the_approved_plan_flow_over_a_predecessor_is_obsolete():
    # A correction across the approved-plan boundary always moves the plan
    # hash between absent and present; with a committed predecessor that is one
    # legitimate difference, the same delta plan_successor represents.
    anchor = plan_record_comment(3)
    planless = _head_advance(direct_intent())
    to_plan = _inputs(planless, origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=PLAN_HASH)
    gained = compare_prepared_intent(planless, to_plan, fresh_plan_anchor=anchor)
    assert (gained.outcome, gained.abort_reason) == ("obsolete", ABORT_SUPERSEDED_INTENT)
    assert {name for name, _b, _a in gained.differing_fields} == {
        "origin_flow",
        "approved_plan_hash",
    }
    assert successor_kind_for(direct_intent(), to_plan) == KIND_PLAN_REPLACEMENT
    # The plan that appears must still authenticate.
    unauthenticated = compare_prepared_intent(planless, to_plan)
    assert unauthenticated.outcome == "contradiction"
    assert "does not authenticate" in " ".join(unauthenticated.contradictions)
    assert (
        compare_prepared_intent(planless, to_plan, fresh_plan_anchor=object()).outcome
        == "contradiction"
    )

    planned = _head_advance(plan_intent())
    from_plan = _inputs(planned, origin_flow="issue-implementation", approved_plan_hash=None)
    lost = compare_prepared_intent(planned, from_plan)
    assert (lost.outcome, lost.abort_reason) == ("obsolete", ABORT_SUPERSEDED_INTENT)
    # Together with a head change it is still one superseded intent.
    moved = compare_prepared_intent(planned, replace(from_plan, head_sha=HEAD_1))
    assert moved.abort_reason == ABORT_SUPERSEDED_INTENT

    # Without a committed predecessor both directions stay contradictions,
    # even when the appearing plan authenticates.
    for stored, fresh in (
        (direct_intent(), _inputs(direct_intent(), origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=PLAN_HASH)),
        (plan_intent(), _inputs(plan_intent(), origin_flow="issue-implementation", approved_plan_hash=None)),
    ):
        result = compare_prepared_intent(stored, fresh, fresh_plan_anchor=anchor)
        assert result.outcome == "contradiction" and result.abort_reason is None
    # A predecessor does not excuse a plan-hash flip without a flow change
    # (the intent model itself forbids that pairing) nor a real replacement
    # that fails to order.
    swapped = compare_prepared_intent(
        planned,
        _inputs(planned, approved_plan_hash="1" * 16),
        stored_plan_anchor=plan_record_comment(9, "Replacement plan."),
        fresh_plan_anchor=anchor,
    )
    assert swapped.outcome == "contradiction"


def test_successor_kind_follows_the_fixed_precedence():
    committed = plan_intent(managed_ci_generation="g1", record_set=record_set(authorization=reissued(ENTRY_AUTHORIZATION)))
    assert successor_kind_for(committed, _inputs(committed)) is None
    assert successor_kind_for(committed, _inputs(committed, head_sha=HEAD_2)) == KIND_HEAD_ADVANCE
    assert (
        successor_kind_for(committed, _inputs(committed, head_sha=HEAD_2, managed_ci_generation="g2"))
        == KIND_MANAGED_CI_CONTINUITY
    )
    assert (
        successor_kind_for(
            committed,
            _inputs(committed, managed_ci_generation="g2", expected_closing_issue_ids=(ISSUE, 900)),
        )
        == KIND_CLOSING_WIDENING
    )
    assert (
        successor_kind_for(
            committed,
            _inputs(committed, approved_plan_hash="1" * 16, expected_closing_issue_ids=(ISSUE, 900)),
        )
        == KIND_PLAN_REPLACEMENT
    )
    direct = direct_intent()
    assert (
        successor_kind_for(
            direct,
            _inputs(direct, origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=PLAN_HASH, head_sha=HEAD_2),
        )
        == KIND_PLAN_REPLACEMENT
    )
    with pytest.raises(AgentLoopError, match="no successor kind can change base"):
        successor_kind_for(committed, _inputs(committed, base="release"))


def test_planned_successor_reissues_affected_entries_and_inherits_the_rest():
    committed = plan_intent(
        managed_ci_generation="g1",
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST)),
        record_set=record_set(authorization=reissued(ENTRY_AUTHORIZATION)),
    )
    effective = {
        ENTRY_HANDOFF: CommentRef(ISSUE_SURFACE, 11, DIGEST),
        ENTRY_PR_CONTRACT: CommentRef(PR_SURFACE, 12, DIGEST),
        ENTRY_AUTHORIZATION: CommentRef(PR_SURFACE, 14, DIGEST),
    }
    assert plan_successor(committed, _inputs(committed), effective_records=effective) is None

    advance = plan_successor(
        committed, _inputs(committed, head_sha=HEAD_2), effective_records=effective
    )
    assert advance.successor_kind == KIND_HEAD_ADVANCE
    assert advance.predecessor_transaction_id == committed.transaction_id
    assert advance.scheduler_checkpoint == committed.scheduler_checkpoint
    assert advance.entry(ENTRY_HANDOFF).inherited == effective[ENTRY_HANDOFF]
    assert advance.entry(ENTRY_PR_CONTRACT).inherited == effective[ENTRY_PR_CONTRACT]
    assert advance.entry(ENTRY_AUTHORIZATION).disposition == "reissued"
    assert advance.entry(ENTRY_INITIAL_CODER_ROUND).disposition == "not-applicable"

    widening = plan_successor(
        committed,
        _inputs(committed, expected_closing_issue_ids=(ISSUE, 900)),
        effective_records=effective,
    )
    assert widening.successor_kind == KIND_CLOSING_WIDENING
    assert widening.entry(ENTRY_HANDOFF).disposition == "reissued"
    assert widening.entry(ENTRY_PR_CONTRACT).disposition == "reissued"
    assert widening.entry(ENTRY_AUTHORIZATION).inherited == effective[ENTRY_AUTHORIZATION]

    with pytest.raises(AgentLoopError, match="must recompute its scheduler checkpoint"):
        plan_successor(
            committed, _inputs(committed, approved_plan_hash="1" * 16), effective_records=effective
        )
    fresh_checkpoint = SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 40, DIGEST))
    replacement = plan_successor(
        committed,
        _inputs(committed, approved_plan_hash="1" * 16),
        effective_records=effective,
        scheduler_checkpoint=fresh_checkpoint,
    )
    assert replacement.successor_kind == KIND_PLAN_REPLACEMENT
    assert replacement.scheduler_checkpoint == fresh_checkpoint
    assert replacement.entry(ENTRY_HANDOFF).disposition == "reissued"
    assert replacement.entry(ENTRY_PR_CONTRACT).inherited == effective[ENTRY_PR_CONTRACT]

    flow = plan_successor(
        direct_intent(),
        _inputs(direct_intent(), origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=PLAN_HASH),
        effective_records=effective,
        scheduler_checkpoint=SchedulerCheckpointRef(absence_reason="no-plan-scheduler-records"),
    )
    assert flow.successor_kind == KIND_PLAN_REPLACEMENT
    assert flow.entry(ENTRY_HANDOFF).disposition == "reissued"
    assert flow.entry(ENTRY_PR_CONTRACT).disposition == "reissued"

    direct_pr = direct_intent(
        origin_flow="direct-pr", record_set=record_set(handoff=not_applicable(ENTRY_HANDOFF))
    )
    corrected = plan_successor(
        direct_pr,
        _inputs(direct_pr, origin_flow="managed-pr"),
        effective_records={ENTRY_PR_CONTRACT: effective[ENTRY_PR_CONTRACT]},
    )
    assert corrected.successor_kind == KIND_FLOW_CORRECTION
    assert corrected.entry(ENTRY_HANDOFF).disposition == "not-applicable"
    assert corrected.entry(ENTRY_PR_CONTRACT).disposition == "reissued"


def _mixed_successor_chain(contract_entry):
    """Committed initial, then a plan replacement that also widens closing."""
    base = plan_intent(
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST))
    )
    mixed = replace(
        base,
        approved_plan_hash="1" * 16,
        expected_closing_issue_ids=(ISSUE, 900),
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 40, DIGEST)),
        successor_kind=KIND_PLAN_REPLACEMENT,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(
            contract=contract_entry, coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)
        ),
    )
    first_contract = v2_contract_comment(12, base)
    pr_comments = [
        prepared_comment(10, base),
        first_contract,
        terminal_comment(
            20,
            base,
            prepared_id=10,
            published={ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
        ),
        prepared_comment(30, mixed),
    ]
    published = {ENTRY_HANDOFF: 31}
    if contract_entry.disposition == "reissued":
        published[ENTRY_PR_CONTRACT] = 32
        pr_comments.append(
            v2_contract_comment(
                32,
                mixed,
                supersession_kind="closing-widening",
                supersedes_record_hash=pr_contract_record_hash(
                    derive_pr_contract(base)
                ),
            )
        )
    pr_comments.append(terminal_comment(40, mixed, prepared_id=30, published=published))
    issue_comments = [v2_handoff_comment(11, base), v2_handoff_comment(31, mixed)]
    return base, mixed, pr_view(*pr_comments), issue_view(*issue_comments)


def test_mixed_successor_cannot_inherit_an_entry_a_lower_precedence_change_affects():
    # The winning kind only obliges the handoff; the closing change still
    # obliges the PR contract, which would otherwise keep the old closing IDs.
    _base, mixed, prs, issues = _mixed_successor_chain(
        inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, DIGEST))
    )
    assert successor_kind_for(_base, TransitionInputs.of(mixed)) == KIND_PLAN_REPLACEMENT
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve_transaction_lineage(prs, repository=REPO, pr_number=PR)
    assert excinfo.value.code == "successor-disposition-mismatch"
    assert "pr-expected-closing-contract" in str(excinfo.value)
    assert "expected_closing_issue_ids changed" in str(excinfo.value)


def test_mixed_successor_reissuing_every_affected_entry_agrees_on_both_surfaces():
    _base, mixed, prs, issues = _mixed_successor_chain(reissued(ENTRY_PR_CONTRACT))
    lineage = resolve_transaction_lineage(prs, repository=REPO, pr_number=PR)
    assert lineage.latest_committed.transaction_id == mixed.transaction_id
    contract = resolve_pr_contract_lineage(prs, lineage, repository=REPO, pr_number=PR)
    handoff = resolve_handoff_lineage(issues, lineage, repository=REPO, issue_number=ISSUE)
    assert contract.comment_id == 32 and handoff.comment_id == 31
    assert contract.transaction_id == handoff.transaction_id == mixed.transaction_id
    assert (
        tuple(contract.contract.expected_closing_issue_ids)
        == tuple(handoff.handoff.expected_closing_issue_ids)
        == (ISSUE, 900)
    )
    assert handoff.handoff.plan_hash == "1" * 16


def test_successor_cannot_reissue_a_publicly_visible_entry_no_changed_field_affects():
    base, comments = _committed_initial(
        plan_intent(scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST)))
    )
    replacement = replace(
        base,
        approved_plan_hash="1" * 16,
        scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 40, DIGEST)),
        successor_kind=KIND_PLAN_REPLACEMENT,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(30, replacement))
    assert excinfo.value.code == "successor-disposition-mismatch"
    assert "no field it carries changed" in str(excinfo.value)


def test_successor_must_inherit_the_predecessors_scheduler_checkpoint_unchanged():
    base, comments = _committed_initial(
        plan_intent(scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 4, DIGEST)))
    )
    drifted = _head_advance(
        base, scheduler_checkpoint=SchedulerCheckpointRef(CommentRef(ISSUE_SURFACE, 8, DIGEST))
    )
    with pytest.raises(WorkflowTransactionError, match="scheduler checkpoint reference unchanged"):
        _resolve(*comments, prepared_comment(30, drifted))


# --- authenticated reader ----------------------------------------------------


class _PagedRunner:
    def __init__(self, pages, *, actor=ACTOR, fail_page=None):
        self.pages = pages
        self.actor = actor
        self.fail_page = fail_page
        self.calls = []

    def run(self, args, *, cwd, check=True, **_kwargs):
        self.calls.append(list(args))
        endpoint = args[-1]
        if endpoint == "user":
            return SimpleNamespace(
                returncode=0, stdout=json.dumps({"login": self.actor[0], "id": self.actor[1]})
            )
        page = int(endpoint.rsplit("page=", 1)[1])
        if page == self.fail_page:
            return SimpleNamespace(returncode=1, stdout="")
        payload = self.pages[page - 1] if page <= len(self.pages) else []
        return SimpleNamespace(
            returncode=0, stdout=payload if isinstance(payload, str) else json.dumps(payload)
        )


def _rest(comment_id, body="hello", author=ACTOR, **overrides):
    raw = {
        "id": comment_id,
        "body": body,
        "user": {"login": author[0], "id": author[1]},
        "created_at": "2026-09-21T10:00:00Z",
        "updated_at": "2026-09-21T10:00:01Z",
    }
    raw.update(overrides)
    return raw


def _read(tmp_path, runner, kind="pr", number=PR):
    return read_authenticated_protocol_comments(
        runner, config=make_config(tmp_path), surface_kind=kind, number=number
    )


def test_reader_paginates_exhaustively_and_returns_full_envelopes(tmp_path):
    first = [_rest(index) for index in range(1, 101)]
    runner = _PagedRunner([first, [_rest(101), _rest(102)]])
    view = _read(tmp_path, runner)
    assert [item.comment_id for item in view.authored] == list(range(1, 103))
    assert view.surface == PR_SURFACE and (view.actor_login, view.actor_id) == ACTOR
    envelope = view.authored[0]
    assert (envelope.surface, envelope.author_login, envelope.author_id) == (PR_SURFACE, *ACTOR)
    assert envelope.created_at == "2026-09-21T10:00:00Z"
    assert envelope.updated_at == "2026-09-21T10:00:01Z" and envelope.body == "hello"
    pages = [call[-1] for call in runner.calls if "comments" in call[-1]]
    assert pages == [
        f"repos/OWNER/REPO/issues/{PR}/comments?per_page=100&page=1",
        f"repos/OWNER/REPO/issues/{PR}/comments?per_page=100&page=2",
    ]


def test_reader_never_returns_a_byte_exact_forged_record_as_authoritative(tmp_path):
    base, comments = _committed_initial()
    raw = [_rest(item.comment_id, item.body, author=FOREIGN) for item in comments]
    raw.append(_rest(99, "ordinary human remark", author=FOREIGN))
    view = _read(tmp_path, _PagedRunner([raw]))
    assert view.authored == ()
    assert [item.comment_id for item in view.ignored_foreign] == [10, 20]
    assert any("ignored-foreign comment 10" in line for line in view.foreign_diagnostics())
    lineage = resolve_transaction_lineage(view, repository=REPO, pr_number=PR)
    assert lineage.chain == () and base.transaction_id in " ".join(lineage.ignored_foreign)


@pytest.mark.parametrize(
    "pages,fail_page,match",
    [
        ([[_rest(index) for index in range(1, 101)]], 2, "failed on page 2"),
        (["{not json"], None, "malformed JSON"),
        ([{"message": "rate limited"}], None, "non-list page"),
        ([[_rest(1), "truncated"]], None, "incomplete page"),
        ([[_rest(1), _rest(1)]], None, "repeated comment ID"),
        ([[_rest(1, created_at=None)]], None, "timestamp is missing"),
        ([[_rest(1, created_at="soon")]], None, "not parseable"),
        ([[_rest("1")]], None, "comment_id must be a positive integer"),
        ([[_rest(1, user=None)]], None, "without an author"),
        ([[_rest(1, user={"login": "bot"})]], None, "author_id must be a positive integer"),
        ([[""]], None, "incomplete page"),
    ],
)
def test_reader_fails_closed_instead_of_returning_a_partial_view(tmp_path, pages, fail_page, match):
    with pytest.raises(AgentLoopError, match=match):
        _read(tmp_path, _PagedRunner(pages, fail_page=fail_page))


def test_reader_under_a_different_actor_grants_no_authority(tmp_path):
    _base, comments = _committed_initial()
    raw = [_rest(item.comment_id, item.body) for item in comments]
    view = _read(tmp_path, _PagedRunner([raw], actor=("new-bot", 9001)))
    assert view.authored == ()
    error = actor_change_error(view)
    assert error is not None
    assert ACTOR[0] in str(error) and "new-bot" in str(error)
    assert "rerun under the original GitHub actor" in str(error)


# --- review round 1 regressions ------------------------------------------------


def test_same_phase_terminal_records_that_differ_fail_closed():
    base, comments = _committed_initial()
    # Same phase, different entry outcomes.
    other_outcomes = terminal_comment(
        21,
        base,
        prepared_id=10,
        published={ENTRY_HANDOFF: 91, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, other_outcomes)
    assert excinfo.value.code == "contradictory-terminal"
    assert "comment 20" in str(excinfo.value) and "comment 21" in str(excinfo.value)
    # Same phase, different prepared binding (a duplicate prepared record).
    rebound = terminal_comment(
        21,
        base,
        prepared_id=14,
        published={ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(14, base), rebound)
    assert excinfo.value.code == "contradictory-terminal"
    # Two aborted records with different reasons.
    intent = direct_intent()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(
            prepared_comment(10, intent),
            terminal_comment(
                20, intent, prepared_id=10, phase=PHASE_ABORTED, abort_reason=ABORT_STALE_HEAD
            ),
            terminal_comment(
                21, intent, prepared_id=10, phase=PHASE_ABORTED,
                abort_reason=ABORT_SUPERSEDED_INTENT,
            ),
        )
    assert excinfo.value.code == "contradictory-terminal"
    # A byte-identical duplicate still canonicalizes to the earliest comment.
    duplicate = terminal_comment(
        21,
        base,
        prepared_id=10,
        published={ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
    )
    lineage = _resolve(*comments, duplicate)
    assert lineage.latest_committed.terminal_comment.comment_id == 20


def test_issue_origin_intent_cannot_omit_the_issue_side_handoff():
    for build in (direct_intent, plan_intent):
        with pytest.raises(AgentLoopError, match="requires the issue-to-PR handoff entry"):
            build(record_set=record_set(handoff=not_applicable(ENTRY_HANDOFF)))
    base, _ = _committed_initial()
    with pytest.raises(AgentLoopError, match="requires the issue-to-PR handoff entry"):
        _head_advance(
            base,
            record_set=record_set(
                handoff=not_applicable(ENTRY_HANDOFF),
                contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, DIGEST)),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )
    # Such a payload is equally unreadable from a prepared record.
    payload = direct_intent().to_payload()
    payload["record_set"][0] = {
        "name": ENTRY_HANDOFF, "disposition": "not-applicable", "inherited": None
    }
    with pytest.raises(AgentLoopError, match="requires the issue-to-PR handoff entry"):
        WorkflowTransition.from_payload(payload)
    # A staged child may still be handoff-only on the PR side.
    staged = direct_intent(
        staged=StagedIdentity(827, ISSUE, "parent"),
        record_set=record_set(contract=not_applicable(ENTRY_PR_CONTRACT)),
    )
    assert staged.entry(ENTRY_HANDOFF).disposition == "reissued"


def _pr_round_comment(comment_id, *, role, plan_hash, plan_subject):
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    body = _attach_round_metadata(
        f"Round {comment_id}.\n-- Claude",
        PostedRoundMetadata(
            flow="pr",
            role=role,
            agent="Claude",
            round_number=1,
            subject=HEAD_1,
            approved_plan_hash=plan_hash,
            approved_plan_subject=plan_subject,
        ),
    )
    return comment(comment_id, body)


def test_only_complete_reviewer_records_are_origin_evidence():
    coder = _pr_round_comment(107, role="coder", plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        find_origin_evidence(pr_view(v1_contract_comment(105), coder))
    assert excinfo.value.code == "origin-evidence-conflict"
    assert "role coder" in str(excinfo.value)

    for partial in (
        _pr_round_comment(108, role="reviewer", plan_hash=PLAN_HASH, plan_subject=None),
        _pr_round_comment(108, role="reviewer", plan_hash=None, plan_subject=PLAN_SUBJECT),
    ):
        # Alone, and beside otherwise valid reviewer evidence.
        for others in ((), (pr_review_comment(107),)):
            with pytest.raises(WorkflowTransactionError) as excinfo:
                find_origin_evidence(pr_view(*others, partial))
            assert excinfo.value.code == "origin-evidence-conflict"

    intent, pr, plan = _legacy_state()
    hash_only = _pr_round_comment(108, role="reviewer", plan_hash=PLAN_HASH, plan_subject=None)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _validate(intent, pr_view(*pr.authored, hash_only), plan)
    assert excinfo.value.code == "legacy-root-invalid"
    assert "contradictory approved-plan identity in comment 108" in str(excinfo.value)
    coder_beside = _pr_round_comment(
        108, role="coder", plan_hash=PLAN_HASH, plan_subject=PLAN_SUBJECT
    )
    with pytest.raises(WorkflowTransactionError, match="role coder"):
        _validate(intent, pr_view(*pr.authored, coder_beside), plan)
