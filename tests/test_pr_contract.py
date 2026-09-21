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
            replace(derive_pr_contract(base), origin_flow=FLOW_APPROVED_PLAN)
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
