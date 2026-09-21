"""Version-2 PR contract codec and the envelope-only contract lineage (#827)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from workflow_transaction_helpers import (
    FOREIGN,
    HEAD_2,
    ISSUE,
    PLAN_HASH,
    PLAN_SUBJECT,
    PR,
    REPO,
    comment,
    direct_intent,
    plan_intent,
    pr_view,
    prepared_comment,
    record_set,
    terminal_comment,
    v1_contract,
    v1_contract_comment,
    v2_contract_comment,
)

from coding_review_agent_loop.errors import AgentLoopError, WorkflowTransactionError
from coding_review_agent_loop.pr_contract import (
    PR_CONTRACT_SUPERSESSION_COMBINED,
    PrExpectedClosingContractV2,
    decode_pr_contract,
    decode_pr_contract_v2,
    encode_pr_contract,
    encode_pr_contract_v2,
    find_latest_pr_contract,
    format_pr_contract_v2_comment,
    make_pr_contract_v2,
    pr_contract_record_hash,
)
from coding_review_agent_loop.protocol_markers import PR_COMMENT_SURFACE, TrustedBody
from coding_review_agent_loop.workflow_transaction import (
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
    KIND_PLAN_REPLACEMENT,
    PHASE_ABORTED,
    CommentRef,
    LegacyRoot,
    OriginEvidence,
    derive_pr_contract,
    inherited,
    not_applicable,
    resolve_pr_contract_lineage,
    resolve_transaction_lineage,
)

TX = "a" * 64
PR_SURFACE = f"pr#{PR}"
ISSUE_SURFACE = f"issue#{ISSUE}"
DIGEST = "d" * 64
V1_HASH = pr_contract_record_hash(v1_contract())


def _v2(**overrides):
    fields = dict(
        repository=REPO,
        pr_number=PR,
        origin_flow="issue-implementation",
        expected_closing_issue_ids=(ISSUE,),
        transaction_id=TX,
        primary_issue_number=ISSUE,
    )
    fields.update(overrides)
    return make_pr_contract_v2(**fields)


def test_v2_contract_round_trips_and_renders_one_trusted_pr_comment_record():
    contract = _v2(supersession_kind="flow-correction", supersedes_record_hash=V1_HASH)
    assert decode_pr_contract_v2(encode_pr_contract_v2(contract)) == contract
    body = TrustedBody.canonical(
        format_pr_contract_v2_comment(contract),
        surface=PR_COMMENT_SURFACE,
        expected_tokens=("AGENT_PR_EXPECTED_CLOSING_ISSUES",),
    )
    assert TX in body and "flow-correction" in body


@pytest.mark.parametrize(
    "overrides",
    [
        {"transaction_id": "short"},
        {"transaction_id": None},
        {"supersession_kind": "flow-correction"},
        {"supersedes_record_hash": V1_HASH},
        {"supersession_kind": "rename", "supersedes_record_hash": V1_HASH},
        {"supersession_kind": "closing-widening", "supersedes_record_hash": "zz"},
        {"origin_flow": "sideways"},
        {"pr_number": 0},
        {"primary_issue_number": True},
    ],
)
def test_v2_contract_codec_is_strict(overrides):
    with pytest.raises(AgentLoopError):
        _v2(**overrides)


def test_v2_decoder_rejects_v1_payloads_extra_keys_and_a_wrong_contract_hash():
    with pytest.raises(AgentLoopError, match="expected exactly"):
        decode_pr_contract_v2(encode_pr_contract(v1_contract()))
    with pytest.raises(AgentLoopError, match="contract_hash is invalid"):
        encode_pr_contract_v2(
            decode_pr_contract_v2(encode_pr_contract_v2(replace(_v2(), contract_hash="0" * 64)))
        )


def test_record_hash_identifies_flow_not_only_closing_ids():
    direct, planned = v1_contract(), v1_contract(origin_flow=FLOW_APPROVED_PLAN)
    assert direct.contract_hash == planned.contract_hash
    assert pr_contract_record_hash(direct) != pr_contract_record_hash(planned)
    assert pr_contract_record_hash(_v2()) != pr_contract_record_hash(_v2(transaction_id="b" * 64))


def test_existing_entry_points_still_reject_a_v2_payload_exactly_as_before():
    body = format_pr_contract_v2_comment(_v2())
    with pytest.raises(AgentLoopError, match="expected exactly"):
        decode_pr_contract(encode_pr_contract_v2(_v2()))
    with pytest.raises(AgentLoopError, match="expected exactly"):
        find_latest_pr_contract([SimpleNamespace(body=body)], repository=REPO, pr_number=PR)
    # Version-1 behavior is untouched, including the divergence outcome for a
    # same-scope flow change.
    v1_direct = v1_contract_comment(5)
    v1_planned = v1_contract_comment(6, v1_contract(origin_flow=FLOW_APPROVED_PLAN))
    assert find_latest_pr_contract([v1_direct], repository=REPO, pr_number=PR) == v1_contract()
    with pytest.raises(AgentLoopError, match="Divergent"):
        find_latest_pr_contract([v1_direct, v1_planned], repository=REPO, pr_number=PR)


# --- flow-correction-reader ---------------------------------------------------


def _resolve(*comments):
    view = pr_view(*comments)
    lineage = resolve_transaction_lineage(view, repository=REPO, pr_number=PR)
    return resolve_pr_contract_lineage(view, lineage, repository=REPO, pr_number=PR)


def _resolve_legacy(*comments):
    """Resolve without re-validating origin evidence (covered in the model tests)."""
    import coding_review_agent_loop.workflow_transaction as module

    view = pr_view(*comments)
    states = module.collect_transactions(view, repository=REPO, pr_number=PR)
    lineage = module.TransactionLineage(
        repository=REPO,
        pr_number=PR,
        surface=view.surface,
        transactions=states,
        chain=tuple(item for item in states if not item.aborted),
    )
    return resolve_pr_contract_lineage(view, lineage, repository=REPO, pr_number=PR)


PUBLISHED = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}


def _initial(intent=None):
    intent = intent or direct_intent()
    return intent, [
        prepared_comment(10, intent),
        v2_contract_comment(12, intent),
        terminal_comment(20, intent, prepared_id=10, published=PUBLISHED),
    ]


def _legacy_correction(contract_ref=None):
    return plan_intent(
        successor_kind=KIND_LEGACY_ROOT_CORRECTION,
        legacy_root=LegacyRoot(
            contract=contract_ref or CommentRef(PR_SURFACE, 5, V1_HASH),
            handoff=None,
            origin_evidence=OriginEvidence(
                kind="pr-round-metadata",
                plan_hash=PLAN_HASH,
                plan_subject=PLAN_SUBJECT,
                reference=CommentRef(PR_SURFACE, 7, DIGEST),
            ),
        ),
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )


def _correction_comments(intent, **supersession):
    supersession = supersession or {
        "supersession_kind": "flow-correction",
        "supersedes_record_hash": V1_HASH,
    }
    return [
        v1_contract_comment(5),
        prepared_comment(30, intent),
        v2_contract_comment(32, intent, **supersession),
        terminal_comment(
            40, intent, prepared_id=30, published={ENTRY_HANDOFF: 31, ENTRY_PR_CONTRACT: 32}
        ),
    ]


def test_legacy_pr_resolves_exactly_like_the_version_1_entry_point():
    resolved = _resolve(v1_contract_comment(5))
    assert resolved.contract == v1_contract() and resolved.era == ERA_LEGACY
    assert (resolved.comment_id, resolved.record_hash) == (5, V1_HASH)
    assert _resolve() is None
    with pytest.raises(AgentLoopError, match="Divergent"):
        _resolve(
            v1_contract_comment(5),
            v1_contract_comment(6, v1_contract(origin_flow=FLOW_APPROVED_PLAN)),
        )


def test_committed_legacy_root_correction_accepts_the_flow_changing_v2_contract():
    intent = _legacy_correction()
    resolved = _resolve_legacy(*_correction_comments(intent))
    assert resolved.contract.origin_flow == FLOW_APPROVED_PLAN
    assert resolved.contract.transaction_id == resolved.transaction_id == intent.transaction_id
    assert resolved.comment_id == 32 and resolved.era == ERA_TRANSACTION


def test_flow_change_without_a_committed_transaction_is_not_accepted():
    intent = _legacy_correction()
    comments = _correction_comments(intent)
    # Prepared only: the version-1 record stays authoritative and nothing the
    # pending transaction published counts yet.
    pending = _resolve_legacy(*comments[:3])
    assert pending.contract == v1_contract() and pending.transaction_id is None
    assert pending.pending_transaction_id == intent.transaction_id
    # Aborted: its records are inert.
    aborted = _resolve_legacy(
        *comments[:3],
        terminal_comment(
            40, intent, prepared_id=30, phase=PHASE_ABORTED, abort_reason="superseded-intent",
            published={ENTRY_PR_CONTRACT: 32},
        ),
    )
    assert aborted.contract == v1_contract() and aborted.pending_transaction_id is None
    # Transaction records deleted or hidden: fail closed, never the legacy path.
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve_legacy(comments[0], comments[2])
    assert excinfo.value.code == "transaction-record-missing"


def test_flow_change_with_a_wrong_record_hash_is_rejected():
    intent = _legacy_correction()
    with pytest.raises(WorkflowTransactionError, match="record hash does not match comment 5"):
        _resolve_legacy(
            *_correction_comments(
                intent, supersession_kind="flow-correction", supersedes_record_hash="0" * 64
            )
        )
    with pytest.raises(WorkflowTransactionError, match="without declaring a supersession"):
        comments = _correction_comments(intent)
        comments[2] = v2_contract_comment(32, intent)
        _resolve_legacy(*comments)


def test_flow_change_whose_legacy_root_names_a_different_record_is_rejected():
    for reference in (CommentRef(PR_SURFACE, 4, V1_HASH), CommentRef(PR_SURFACE, 5, DIGEST)):
        with pytest.raises(WorkflowTransactionError, match="names exactly the superseded"):
            _resolve_legacy(*_correction_comments(_legacy_correction(reference)))


def test_flow_change_is_never_accepted_on_an_initial_or_widening_transaction():
    intent = plan_intent()
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(
            v1_contract_comment(5),
            prepared_comment(30, intent),
            v2_contract_comment(
                32, intent, supersession_kind="flow-correction", supersedes_record_hash=V1_HASH
            ),
            terminal_comment(40, intent, prepared_id=30, published=PUBLISHED | {ENTRY_PR_CONTRACT: 32}),
        )
    assert excinfo.value.code == "contract-supersession-invalid"


def test_committed_flow_correction_over_a_v2_predecessor_is_accepted():
    base = direct_intent(
        origin_flow="direct-pr", record_set=record_set(handoff=not_applicable(ENTRY_HANDOFF))
    )
    published = {ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
    first_hash = pr_contract_record_hash(derive_pr_contract(base))
    corrected = replace(
        base,
        origin_flow="managed-pr",
        successor_kind=KIND_FLOW_CORRECTION,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(
            handoff=not_applicable(ENTRY_HANDOFF),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    comments = [
        prepared_comment(10, base),
        v2_contract_comment(12, base),
        terminal_comment(20, base, prepared_id=10, published=published),
        prepared_comment(30, corrected),
        v2_contract_comment(
            32, corrected, supersession_kind="flow-correction", supersedes_record_hash=first_hash
        ),
        terminal_comment(40, corrected, prepared_id=30, published={ENTRY_PR_CONTRACT: 32}),
    ]
    resolved = _resolve(*comments)
    assert resolved.contract.origin_flow == "managed-pr" and resolved.comment_id == 32
    # Until the correction commits, the predecessor's contract stays authoritative.
    before_commit = _resolve(*comments[:5])
    assert before_commit.contract.origin_flow == "direct-pr"
    assert before_commit.pending_transaction_id == corrected.transaction_id


def _flow_and_closing_chain(case, supersession_kind, *, ids=(ISSUE, 900), flow_changes=True):
    """Committed initial, then one successor changing flow and closing scope."""
    if case == "flow-correction":
        base = direct_intent(
            origin_flow="direct-pr", record_set=record_set(handoff=not_applicable(ENTRY_HANDOFF))
        )
        successor = replace(
            base,
            origin_flow="managed-pr" if flow_changes else "direct-pr",
            expected_closing_issue_ids=ids,
            successor_kind=KIND_FLOW_CORRECTION if flow_changes else KIND_CLOSING_WIDENING,
            predecessor_transaction_id=base.transaction_id,
            record_set=record_set(
                handoff=not_applicable(ENTRY_HANDOFF),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )
        first = {ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
        second = {ENTRY_PR_CONTRACT: 32}
    else:
        # Plan replacement outranks both: issue flow -> approved-plan flow,
        # a plan hash appears, and the closing scope widens, all at once.
        base = direct_intent()
        successor = plan_intent(
            expected_closing_issue_ids=ids,
            successor_kind=KIND_PLAN_REPLACEMENT,
            predecessor_transaction_id=base.transaction_id,
            record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
        )
        first = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
        second = {ENTRY_HANDOFF: 31, ENTRY_PR_CONTRACT: 32}
    comments = [
        prepared_comment(10, base),
        v2_contract_comment(12, base),
        terminal_comment(20, base, prepared_id=10, published=first),
        prepared_comment(30, successor),
        v2_contract_comment(
            32,
            successor,
            supersession_kind=supersession_kind,
            supersedes_record_hash=pr_contract_record_hash(derive_pr_contract(base)),
        ),
        terminal_comment(40, successor, prepared_id=30, published=second),
    ]
    return successor, comments


@pytest.mark.parametrize("case", ["flow-correction", "plan-replacement"])
def test_committed_successor_changing_flow_and_closing_scope_resolves_its_contract(case):
    successor, comments = _flow_and_closing_chain(case, PR_CONTRACT_SUPERSESSION_COMBINED)
    resolved = _resolve(*comments)
    assert resolved.comment_id == 32
    assert resolved.transaction_id == successor.transaction_id
    assert resolved.contract.origin_flow == successor.origin_flow
    assert tuple(resolved.contract.expected_closing_issue_ids) == (ISSUE, 900)
    assert resolved.contract.supersession_kind == PR_CONTRACT_SUPERSESSION_COMBINED
    # Neither single-change kind can describe the combined change.
    for single in ("flow-correction", "closing-widening"):
        _successor, mislabeled = _flow_and_closing_chain(case, single)
        with pytest.raises(WorkflowTransactionError) as excinfo:
            _resolve(*mislabeled)
        assert excinfo.value.code == "contract-supersession-invalid"
    # The existing entry point still rejects the version-2 payload.
    with pytest.raises(AgentLoopError):
        find_latest_pr_contract(
            [SimpleNamespace(body=item.body) for item in comments], repository=REPO, pr_number=PR
        )


def test_combined_supersession_kind_requires_both_changes():
    # Only the flow changes.
    _successor, comments = _flow_and_closing_chain(
        "flow-correction", PR_CONTRACT_SUPERSESSION_COMBINED, ids=(ISSUE,)
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments)
    assert excinfo.value.code == "contract-supersession-invalid"
    # Only the closing scope changes.
    _successor, comments = _flow_and_closing_chain(
        "flow-correction", PR_CONTRACT_SUPERSESSION_COMBINED, flow_changes=False
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments)
    assert excinfo.value.code == "contract-supersession-invalid"


def test_combined_supersession_kind_round_trips_and_unknown_kinds_are_rejected():
    successor, _comments = _flow_and_closing_chain(
        "flow-correction", PR_CONTRACT_SUPERSESSION_COMBINED
    )
    contract = derive_pr_contract(
        successor,
        supersession_kind=PR_CONTRACT_SUPERSESSION_COMBINED,
        supersedes_record_hash="a" * 64,
    )
    assert decode_pr_contract_v2(encode_pr_contract_v2(contract)) == contract
    with pytest.raises(AgentLoopError):
        derive_pr_contract(
            successor,
            supersession_kind="closing-widening+flow-correction",
            supersedes_record_hash="a" * 64,
        )


def test_closing_widening_and_inherited_contract_resolve_through_the_successor_chain():
    base, comments = _initial()
    base_hash = pr_contract_record_hash(derive_pr_contract(base))
    advance = replace(
        base,
        head_sha=HEAD_2,
        successor_kind=KIND_HEAD_ADVANCE,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(
            handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
            contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, base_hash)),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    advanced = comments + [
        prepared_comment(30, advance),
        terminal_comment(40, advance, prepared_id=30),
    ]
    resolved = _resolve(*advanced)
    # The inherited record is bound to the successor chain, not only to the
    # transaction ID it literally carries.
    assert resolved.contract.transaction_id == base.transaction_id
    assert resolved.transaction_id == advance.transaction_id and resolved.comment_id == 12

    wrong = replace(
        advance,
        record_set=record_set(
            handoff=advance.entry(ENTRY_HANDOFF),
            contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, DIGEST)),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*comments, prepared_comment(30, wrong), terminal_comment(40, wrong, prepared_id=30))
    assert excinfo.value.code == "inherited-mismatch"

    widened = replace(
        base,
        expected_closing_issue_ids=(ISSUE, 900),
        successor_kind=KIND_CLOSING_WIDENING,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )
    widening = comments + [
        prepared_comment(30, widened),
        v2_contract_comment(
            32, widened, supersession_kind="closing-widening", supersedes_record_hash=base_hash
        ),
        terminal_comment(40, widened, prepared_id=30, published={ENTRY_HANDOFF: 31, ENTRY_PR_CONTRACT: 32}),
    ]
    assert _resolve(*widening).contract.expected_closing_issue_ids == (ISSUE, 900)


def test_forged_and_contradictory_v2_contracts_never_count():
    base, comments = _initial()
    forged = v2_contract_comment(
        50, replace(base, expected_closing_issue_ids=(ISSUE, 900)), author=FOREIGN
    )
    assert _resolve(*comments, forged).contract.expected_closing_issue_ids == (ISSUE,)
    # A contract that disagrees with its own transaction intent is refused.
    tampered = comment(
        12,
        format_pr_contract_v2_comment(
            replace(
                derive_pr_contract(base),
                origin_flow=FLOW_APPROVED_PLAN,
                approved_plan_hash=PLAN_HASH,
            )
        ),
    )
    with pytest.raises(WorkflowTransactionError, match="disagrees with its transaction intent"):
        _resolve(comments[0], tampered, comments[2])
    # A version-1 record appended after the PR became transaction-era is refused.
    with pytest.raises(WorkflowTransactionError, match="appended after the PR became transaction-era"):
        _resolve(*comments, v1_contract_comment(60))
    # Exact duplicates from interleaved writers canonicalize to the earliest.
    assert _resolve(*comments, v2_contract_comment(14, base)).comment_id == 12


def test_contract_lineage_rejects_an_unauthenticated_snapshot():
    snapshot = [SimpleNamespace(body=format_pr_contract_v2_comment(_v2()))]
    with pytest.raises(AgentLoopError, match="only an authenticated comment view"):
        resolve_pr_contract_lineage(snapshot, None, repository=REPO, pr_number=PR)
    assert isinstance(_v2(), PrExpectedClosingContractV2)


def test_v2_contract_carries_the_plan_hash_exactly_for_the_approved_plan_flow():
    planned = _v2(origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=PLAN_HASH)
    assert decode_pr_contract_v2(encode_pr_contract_v2(planned)) == planned
    assert planned.approved_plan_hash == PLAN_HASH
    assert f"Plan hash: {PLAN_HASH}" in format_pr_contract_v2_comment(planned)
    assert _v2().approved_plan_hash is None
    assert "Plan hash" not in format_pr_contract_v2_comment(_v2())
    # The hash is part of the full record hash, so a replacement is a new record.
    assert pr_contract_record_hash(planned) != pr_contract_record_hash(
        _v2(origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash="1" * 16)
    )
    for bad in (
        dict(origin_flow=FLOW_APPROVED_PLAN),
        dict(origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash="B" * 16),
        dict(origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash="b" * 15),
        dict(origin_flow=FLOW_APPROVED_PLAN, approved_plan_hash=7),
        dict(approved_plan_hash=PLAN_HASH),
        dict(origin_flow="direct-pr", primary_issue_number=None, approved_plan_hash=PLAN_HASH),
    ):
        with pytest.raises(AgentLoopError, match="approved_plan_hash"):
            _v2(**bad)
    # A payload without the field is not a version-2 contract at all.
    import base64
    import json

    encoded = encode_pr_contract_v2(planned)
    payload = json.loads(base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)))
    del payload["approved_plan_hash"]
    stripped = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    with pytest.raises(AgentLoopError):
        decode_pr_contract_v2(stripped)


def _plan_replacement_chain(*, contract_intent=None, kind=None):
    """Committed approved-plan initial, then a hash-to-hash plan replacement."""
    base = plan_intent()
    successor = plan_intent(
        approved_plan_hash="1" * 16,
        successor_kind=KIND_PLAN_REPLACEMENT,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )
    comments = [
        prepared_comment(10, base),
        v2_contract_comment(12, base),
        terminal_comment(
            20,
            base,
            prepared_id=10,
            published={ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13},
        ),
        prepared_comment(30, successor),
        v2_contract_comment(32, contract_intent or successor),
        terminal_comment(
            40, successor, prepared_id=30, published={ENTRY_HANDOFF: 31, ENTRY_PR_CONTRACT: 32}
        ),
    ]
    return base, successor, comments


def test_plan_replacement_reissues_the_pr_contract_with_the_new_plan_hash():
    base, successor, comments = _plan_replacement_chain()
    assert _resolve(*comments[:3]).contract.approved_plan_hash == PLAN_HASH
    resolved = _resolve(*comments)
    assert resolved.comment_id == 32 and resolved.transaction_id == successor.transaction_id
    assert resolved.contract.approved_plan_hash == "1" * 16
    assert resolved.contract.supersession_kind is None
    # The model itself refuses a plan replacement that keeps the old contract.
    with pytest.raises(AgentLoopError, match="must reissue pr-expected-closing-contract"):
        replace(
            successor,
            record_set=record_set(
                contract=inherited(ENTRY_PR_CONTRACT, CommentRef(PR_SURFACE, 12, DIGEST)),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )


# --- review round 1: canonical wire form ---------------------------------------


def _reencode(encoded, **dumps_kwargs):
    import base64
    import json

    value = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
    return base64.urlsafe_b64encode(json.dumps(value, **dumps_kwargs).encode()).decode()


def test_v2_contract_decoder_and_lineage_reject_a_noncanonical_wire_record():
    canonical = encode_pr_contract_v2(_v2())
    assert decode_pr_contract_v2(canonical) == _v2()
    for noncanonical in (
        _reencode(canonical, sort_keys=True),  # default separators add whitespace
        _reencode(canonical, separators=(",", ":"), sort_keys=False),
        _reencode(canonical, indent=1, sort_keys=True),
    ):
        if noncanonical == canonical:
            continue
        with pytest.raises(AgentLoopError, match="not canonically encoded"):
            decode_pr_contract_v2(noncanonical)

    base, comments = _initial()
    spaced = _reencode(encode_pr_contract_v2(derive_pr_contract(base)), sort_keys=True)
    body = comments[1].body.replace(encode_pr_contract_v2(derive_pr_contract(base)), spaced)
    assert body != comments[1].body
    with pytest.raises(AgentLoopError, match="not canonically encoded"):
        _resolve(comments[0], comment(12, body), comments[2])


# --- review round 2 -------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"repository": "x"},
        {"repository": "OWNER/REPO/extra"},
        {"repository": " OWNER/REPO"},
        {"primary_issue_number": None},
        {"origin_flow": "approved-plan-implementation", "primary_issue_number": None},
        {"expected_closing_issue_ids": (900,)},
        {"origin_flow": "direct-pr", "expected_closing_issue_ids": ()},
    ],
)
def test_v2_contract_codec_enforces_the_typed_intent_invariants(overrides):
    with pytest.raises(AgentLoopError):
        _v2(**overrides)


def test_v2_contract_codec_still_accepts_a_direct_pr_without_primary_issue():
    contract = _v2(origin_flow="direct-pr", primary_issue_number=None, expected_closing_issue_ids=())
    assert decode_pr_contract_v2(encode_pr_contract_v2(contract)) == contract
    # The version-1 codec keeps its historical permissiveness.
    assert decode_pr_contract(encode_pr_contract(replace(v1_contract(), repository="x")))


def _successor(base, **overrides):
    fields = dict(
        head_sha=HEAD_2,
        successor_kind=KIND_HEAD_ADVANCE,
        predecessor_transaction_id=base.transaction_id,
        record_set=record_set(
            handoff=inherited(ENTRY_HANDOFF, CommentRef(ISSUE_SURFACE, 11, DIGEST)),
            contract=inherited(
                ENTRY_PR_CONTRACT,
                CommentRef(PR_SURFACE, 12, pr_contract_record_hash(derive_pr_contract(base))),
            ),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    fields.update(overrides)
    return replace(base, **fields)


def _committed_successor(base, comments, successor):
    return comments + [
        prepared_comment(30, successor),
        terminal_comment(40, successor, prepared_id=30),
    ]


def test_committed_successor_whose_kind_disagrees_with_its_delta_never_resolves():
    base, comments = _initial()

    def refused(successor):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            _resolve(*_committed_successor(base, comments, successor))
        assert excinfo.value.code == "successor-kind-mismatch"
        return str(excinfo.value)

    # A head-advance that also widens the closing scope while inheriting the
    # predecessor's handoff and PR contract.
    assert "require closing-widening" in refused(
        _successor(base, expected_closing_issue_ids=(ISSUE, 900))
    )
    # An identical no-change successor.
    assert "changes nothing" in refused(_successor(base, head_sha=base.head_sha))
    # A closing change that is not a strict superset.
    wide, wide_comments = _initial(direct_intent(expected_closing_issue_ids=(ISSUE, 900)))
    narrowed = replace(
        wide,
        expected_closing_issue_ids=(ISSUE, 901),
        successor_kind=KIND_CLOSING_WIDENING,
        predecessor_transaction_id=wide.transaction_id,
        record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
    )
    with pytest.raises(WorkflowTransactionError, match="strict superset") as excinfo:
        _resolve(
            *wide_comments,
            prepared_comment(30, narrowed),
            terminal_comment(
                40, narrowed, prepared_id=30,
                published={ENTRY_HANDOFF: 31, ENTRY_PR_CONTRACT: 32},
            ),
        )
    assert excinfo.value.code == "successor-kind-mismatch"

    # A head-advance that silently changes the origin flow.
    direct_pr = direct_intent(
        origin_flow="direct-pr", record_set=record_set(handoff=not_applicable(ENTRY_HANDOFF))
    )
    _b, direct_comments = _initial(direct_pr)
    flipped = replace(
        direct_pr,
        origin_flow="managed-pr",
        head_sha=HEAD_2,
        successor_kind=KIND_HEAD_ADVANCE,
        predecessor_transaction_id=direct_pr.transaction_id,
        record_set=record_set(
            handoff=not_applicable(ENTRY_HANDOFF),
            contract=inherited(
                ENTRY_PR_CONTRACT,
                CommentRef(PR_SURFACE, 12, pr_contract_record_hash(derive_pr_contract(direct_pr))),
            ),
            coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
        ),
    )
    with pytest.raises(WorkflowTransactionError, match="require flow-correction"):
        _resolve(*_committed_successor(direct_pr, direct_comments, flipped))

    # The correctly labeled head-advance still resolves through the chain.
    good = _resolve(*_committed_successor(base, comments, _successor(base)))
    assert good.transaction_id == _successor(base).transaction_id


def _timed_initial(*, prepared=(10, 10), contract=(12, 12), terminal=(20, 20), bind=10):
    intent = direct_intent()
    published = {**PUBLISHED, ENTRY_PR_CONTRACT: contract[0]}
    return intent, [
        comment(prepared[0], prepared_comment(prepared[0], intent).body, second=prepared[1]),
        comment(contract[0], v2_contract_comment(contract[0], intent).body, second=contract[1]),
        comment(
            terminal[0],
            terminal_comment(terminal[0], intent, prepared_id=bind, published=published).body,
            second=terminal[1],
        ),
    ]


def test_contract_must_be_published_between_the_bound_prepared_and_terminal_records():
    # Same-second publication ordered by comment ID is normal and accepted.
    _intent, same_second = _timed_initial(prepared=(10, 7), contract=(12, 7), terminal=(20, 7))
    assert _resolve(*same_second).comment_id == 12

    def refused(comments):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            _resolve(*comments)
        assert excinfo.value.code == "record-unordered"

    refused(_timed_initial(contract=(9, 9))[1])  # posted before preparation
    refused(_timed_initial(contract=(25, 25))[1])  # posted after the terminal record
    refused(_timed_initial(contract=(12, 3))[1])  # greater ID, earlier timestamp
    refused(_timed_initial(contract=(12, 50))[1])  # later than the terminal timestamp

    # A contract carried inside the terminal comment itself.
    intent = direct_intent()
    terminal = terminal_comment(
        20, intent, prepared_id=10, published={**PUBLISHED, ENTRY_PR_CONTRACT: 20}
    )
    merged = comment(20, terminal.body + "\n\n" + v2_contract_comment(20, intent).body)
    refused([prepared_comment(10, intent), merged])


def test_terminal_must_be_later_than_the_exact_prepared_comment_it_binds():
    intent = direct_intent()
    comments = [
        prepared_comment(10, intent),
        v2_contract_comment(12, intent),
        # The terminal names a duplicate prepared comment posted after it.
        terminal_comment(20, intent, prepared_id=25, published=PUBLISHED),
        prepared_comment(25, intent),
    ]
    with pytest.raises(WorkflowTransactionError, match="not later than its prepared record"):
        _resolve(*comments)
    # Binding a duplicate that precedes the terminal is fine, and the named
    # contract must then be later than that bound duplicate.
    bound_late = [
        prepared_comment(10, intent),
        v2_contract_comment(12, intent),
        prepared_comment(14, intent),
        terminal_comment(20, intent, prepared_id=14, published=PUBLISHED),
    ]
    with pytest.raises(WorkflowTransactionError) as excinfo:
        _resolve(*bound_late)
    assert excinfo.value.code == "record-unordered"


def test_non_staged_transition_cannot_omit_the_pr_contract():
    import base64
    import hashlib
    import json

    from coding_review_agent_loop.workflow_transaction import StagedIdentity

    for build in (direct_intent, plan_intent):
        with pytest.raises(AgentLoopError, match="requires the PR expected-closing contract"):
            build(record_set=record_set(contract=not_applicable(ENTRY_PR_CONTRACT)))
    with pytest.raises(AgentLoopError, match="requires the PR expected-closing contract"):
        direct_intent(
            origin_flow="direct-pr",
            record_set=record_set(
                handoff=not_applicable(ENTRY_HANDOFF), contract=not_applicable(ENTRY_PR_CONTRACT)
            ),
        )
    staged = direct_intent(
        staged=StagedIdentity(827, ISSUE, "parent"),
        record_set=record_set(contract=not_applicable(ENTRY_PR_CONTRACT)),
    )
    assert staged.entry(ENTRY_PR_CONTRACT).disposition == "not-applicable"

    # A hand-built committed lineage whose stored intent omits the contract on
    # a non-staged transition never resolves.
    payload = staged.to_payload()
    payload["staged"] = None
    tx_id = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()

    def encoded(record):
        return base64.urlsafe_b64encode(
            json.dumps(record, separators=(",", ":"), sort_keys=True).encode()
        ).decode()

    prepared = {
        "schema_version": 1, "phase": "prepared", "transaction_id": tx_id, "intent": payload,
        "writer": {"login": "agent-loop-bot", "id": 4242},
    }
    terminal = {
        "schema_version": 1, "phase": "committed", "transaction_id": tx_id,
        "prepared_comment_id": 10,
        "entries": [
            {"name": ENTRY_HANDOFF, "comment_id": 11, "status": None},
            {"name": ENTRY_PR_CONTRACT, "comment_id": None, "status": "not-applicable"},
            {"name": "managed-ci-authorization", "comment_id": None, "status": "not-applicable"},
            {"name": ENTRY_INITIAL_CODER_ROUND, "comment_id": 13, "status": None},
        ],
    }
    marker = "AGENT_" + "WORKFLOW_TRANSACTION"
    with pytest.raises(AgentLoopError, match="requires the PR expected-closing contract"):
        _resolve(
            comment(10, f"<!-- {marker}: {encoded(prepared)} -->"),
            comment(20, f"<!-- {marker}: {encoded(terminal)} -->"),
        )
