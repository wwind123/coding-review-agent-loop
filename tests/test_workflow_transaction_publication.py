"""Workflow tests for the transaction publication seam, gate, and readers (#827 stage B)."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from agent_loop_helpers import make_config
from workflow_transaction_helpers import (
    ACTOR,
    FAIL_BEFORE_WRITE,
    FOREIGN,
    HEAD_1,
    HEAD_2,
    ISSUE,
    PLAN,
    PLAN_HASH,
    PLAN_SUBJECT,
    PR,
    REPO,
    WRITE_THEN_REPORT_FAILURE,
    TransactionGitHub,
    direct_intent,
    plan_key,
    plan_record_comment,
    pr_review_comment,
    prepared_comment,
    scheduler_comment,
    v1_contract,
    v1_contract_comment,
    v1_handoff_comment,
)

from coding_review_agent_loop.errors import WorkflowTransactionError
from coding_review_agent_loop.protocol_markers import TrustedBody
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.workflow_transaction import (
    ENTRY_AUTHORIZATION,
    ENTRY_HANDOFF,
    ENTRY_INITIAL_CODER_ROUND,
    ENTRY_PR_CONTRACT,
    FLOW_APPROVED_PLAN,
    FLOW_ISSUE,
    KIND_CLOSING_WIDENING,
    KIND_HEAD_ADVANCE,
    KIND_INITIAL,
    KIND_MANAGED_CI_CONTINUITY,
    RECOVERY_ORIGINAL_ACTOR,
    RECOVERY_RERUN,
    STATUS_WAIVED_CODER_RESPONSE,
    ResolvedPrContract,
    StagedIdentity,
    collect_transactions,
    pr_contract_record_hash,
)
from coding_review_agent_loop.github import (
    read_authenticated_protocol_comments,
    reset_authenticated_github_actor,
)
from coding_review_agent_loop.workflow_transaction_publication import (
    CODE_HEAD_NOT_COMMITTED,
    CODE_PARTIAL_CANDIDATE,
    CODE_PENDING,
    CODE_SIBLING_LOST,
    CODE_UNCOMMITTED,
    CODE_WRITE_FAILED,
    ORIGIN_APPROVED_PLAN,
    ORIGIN_DIRECT_ISSUE,
    ORIGIN_PR_RESUME,
    ORIGIN_STAGED_CHILD,
    ApprovedPlanInput,
    Committed,
    CommittedTransaction,
    Granted,
    InitialCoderRound,
    Legacy,
    LegacyEra,
    NoCandidate,
    NoLiveHeadAuthority,
    PublicationViews,
    Recoverable,
    RecoverableSuccessor,
    Released,
    TransitionRequest,
    Unavailable,
    contract_supersession,
    discover_canonical_issue_pr,
    ensure_head_transaction,
    publish_transition,
    read_pr_transaction_views,
    require_committed_transaction,
    resolve_round_authority,
    route_issue_publication,
)

PARENT = 700
MODES = (FAIL_BEFORE_WRITE, WRITE_THEN_REPORT_FAILURE)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def coder_round(head=HEAD_1, text="Implemented the change."):
    def render(transaction_id: str) -> TrustedBody:
        body = _attach_round_metadata(
            f"{text}\n-- Anthropic Claude",
            PostedRoundMetadata(
                flow="pr", role="coder", agent="Claude", round_number=1, subject=head,
                workflow_transaction_id=transaction_id,
            ),
        )
        return TrustedBody.canonical(body, expected_tokens=("AGENT_LOOP_META",))

    return InitialCoderRound(render)


def direct_request(**overrides) -> TransitionRequest:
    fields = dict(
        repository=REPO, pr_number=PR, base="main", head_sha=HEAD_1,
        origin_path=ORIGIN_DIRECT_ISSUE, expected_closing_issue_ids=(ISSUE,),
        primary_issue=ISSUE, initial_coder_round=coder_round(),
    )
    fields.update(overrides)
    return TransitionRequest(**fields)


def plan_request(**overrides) -> TransitionRequest:
    fields = dict(
        origin_path=ORIGIN_APPROVED_PLAN,
        approved_plan=ApprovedPlanInput(PLAN_HASH, PLAN_SUBJECT, plan_key()),
    )
    fields.update(overrides)
    return direct_request(**fields)


def seed_plan(github: TransactionGitHub, number: int = ISSUE, plan: str = PLAN) -> int:
    github.seed(number, plan_record_comment(1, plan, number=number).body)
    return github.seed(number, scheduler_comment(2, plan, number=number).body)


def publish(github, request, tmp_path, **kwargs):
    return publish_transaction(github, request, tmp_path, **kwargs)


def publish_transaction(github, request, tmp_path, **kwargs):
    return publish_transition(github, config=make_config(tmp_path), request=request, **kwargs)


def states(github, tmp_path, number=PR):
    view = read_authenticated_protocol_comments(
        github, config=make_config(tmp_path), surface_kind="pr", number=number
    )
    return collect_transactions(view, repository=REPO, pr_number=number)


def gate(github, tmp_path, *, live_head=HEAD_1, issue=ISSUE, plan_issue=None, **kwargs):
    resolved_views = _views(github, tmp_path, issue=issue, plan_issue=plan_issue)
    return require_committed_transaction(
        resolved_views, repository=REPO, pr_number=PR, issue_number=issue,
        live_head=live_head, **kwargs,
    )


def _views(github, tmp_path, *, issue=ISSUE, plan_issue=None) -> PublicationViews:
    config = make_config(tmp_path)
    pr = read_authenticated_protocol_comments(github, config=config, surface_kind="pr", number=PR)
    issue_view = (
        read_authenticated_protocol_comments(
            github, config=config, surface_kind="issue", number=issue
        )
        if issue is not None else None
    )
    plan_view = issue_view
    if plan_issue is not None:
        plan_view = read_authenticated_protocol_comments(
            github, config=config, surface_kind="issue", number=plan_issue
        )
    return PublicationViews(pr, issue_view, plan_view)


def assert_converged(github, tmp_path, *, committed: CommittedTransaction, entries: int):
    """Exactly one prepared, one committed, and one canonical record per entry."""
    found = [item for item in states(github, tmp_path) if not item.aborted]
    assert len(found) == 1
    state = found[0]
    assert state.committed and state.transaction_id == committed.transaction_id
    assert not state.prepared_duplicates
    tx = committed.transaction_id
    assert sum(tx in body and "AGENT_ISSUE_PR_HANDOFF" in body for body in github.bodies(ISSUE)) == 1
    pr_bodies = github.bodies(PR)
    assert sum("AGENT_PR_EXPECTED_CLOSING_ISSUES" in body for body in pr_bodies) == 1
    assert sum("AGENT_WORKFLOW_TRANSACTION" in body for body in pr_bodies) == 2
    published = [item for item in state.terminal.outcomes if item.comment_id is not None]
    assert len(published) == entries


# ---------------------------------------------------------------------------
# Each-boundary convergence per origin path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5])
def test_direct_issue_each_boundary(tmp_path, boundary, mode):
    github = TransactionGitHub()
    github.fail_write(boundary, mode)
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, direct_request(), tmp_path)
    assert raised.value.code == CODE_WRITE_FAILED
    assert raised.value.recovery_action == RECOVERY_RERUN
    # Uncommitted: nothing qualifies.  (A commit that landed but reported failure is
    # genuinely committed.)
    landed_commit = boundary == 5 and mode == WRITE_THEN_REPORT_FAILURE
    if github.bodies(PR) and not landed_commit:
        with pytest.raises(WorkflowTransactionError) as refused:
            gate(github, tmp_path)
        assert refused.value.code in {CODE_PENDING, CODE_UNCOMMITTED}
    # A new session without the coder response still finishes the first four
    # boundaries' state; with it, the round is written.
    committed = publish(github, direct_request(), tmp_path)
    assert committed.intent.origin_flow == FLOW_ISSUE
    assert committed.intent.successor_kind == KIND_INITIAL
    assert_converged(github, tmp_path, committed=committed, entries=3)
    # The initial coder comment never embeds the PR contract.
    coder = [body for body in github.bodies(PR) if "Implemented the change." in body]
    assert len(coder) == 1 and "AGENT_PR_EXPECTED_CLOSING_ISSUES" not in coder[0]
    assert gate(github, tmp_path).transaction_id == committed.transaction_id
    # A further rerun writes nothing.
    before = github.write_count
    assert publish(github, direct_request(), tmp_path).transaction_id == committed.transaction_id
    assert github.write_count == before


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("scheduler_fields", [True, False])
def test_approved_plan_each_boundary(tmp_path, boundary, mode, scheduler_fields):
    github = TransactionGitHub()
    checkpoint_id = seed_plan(github)
    github.fail_write(boundary, mode)
    with pytest.raises(WorkflowTransactionError):
        publish(github, plan_request(), tmp_path)
    prepared = [item.transaction_id for item in states(github, tmp_path)]
    # An authenticated PR reviewer round, and a later same-subject checkpoint,
    # are posted between the failure and the rerun.
    github.seed(PR, pr_review_comment(9).body)
    if scheduler_fields:
        github.seed(ISSUE, scheduler_comment(10, round_number=2).body)
    committed = publish(github, plan_request(), tmp_path)
    if prepared:
        assert committed.transaction_id == prepared[0]
    assert committed.intent.origin_flow == FLOW_APPROVED_PLAN
    assert committed.intent.approved_plan_hash == PLAN_HASH
    assert committed.intent.scheduler_checkpoint.reference.comment_id == checkpoint_id
    assert committed.handoff.handoff.flow == committed.contract.contract.origin_flow
    assert committed.handoff.handoff.plan_hash == PLAN_HASH
    assert committed.contract.contract.transaction_id == committed.transaction_id
    assert_converged(github, tmp_path, committed=committed, entries=3)
    assert gate(github, tmp_path, plan_candidate_key=plan_key()) is not None


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("boundary", [1, 2, 3, 4])
def test_staged_child_each_boundary(tmp_path, boundary, mode):
    github = TransactionGitHub()
    seed_plan(github, number=PARENT)
    request = plan_request(
        origin_path=ORIGIN_STAGED_CHILD,
        staged=StagedIdentity(PARENT, ISSUE, "parent"),
        initial_coder_round=None,
    )
    github.fail_write(boundary, mode)
    with pytest.raises(WorkflowTransactionError):
        publish(github, request, tmp_path)
    committed = publish(github, request, tmp_path)
    intent = committed.intent
    assert intent.staged == StagedIdentity(PARENT, ISSUE, "parent")
    assert intent.expected_closing_issue_ids == (ISSUE,)
    # Both the handoff and the PR contract are reissued; the checkpoint names the parent.
    assert intent.entry(ENTRY_HANDOFF).disposition == "reissued"
    assert intent.entry(ENTRY_PR_CONTRACT).disposition == "reissued"
    assert intent.scheduler_checkpoint.reference.surface == f"issue#{PARENT}"
    assert_converged(github, tmp_path, committed=committed, entries=2)
    assert not any("AGENT_ISSUE_PR_HANDOFF" in body for body in github.bodies(PARENT))


def test_staged_parent_may_never_join_the_closing_scope(tmp_path):
    github = TransactionGitHub()
    seed_plan(github, number=PARENT)
    request = plan_request(
        origin_path=ORIGIN_STAGED_CHILD, staged=StagedIdentity(PARENT, ISSUE, "parent"),
        expected_closing_issue_ids=(ISSUE, PARENT), initial_coder_round=None,
    )
    with pytest.raises(Exception, match="staged parent"):
        publish(github, request, tmp_path)
    assert github.write_count == 0


# ---------------------------------------------------------------------------
# Adoption, forgery, actor change, contradictions
# ---------------------------------------------------------------------------


def test_forged_exact_records_are_never_adopted_and_are_listed(tmp_path):
    reference = TransactionGitHub()
    committed = publish(reference, direct_request(), tmp_path)
    github = TransactionGitHub()
    for number, body in reference.writes:
        if "AGENT_WORKFLOW_TRANSACTION" not in body:
            github.seed(number, body, author=FOREIGN)
    # Forged handoff/contract/coder copies do not make the PR transaction-era
    # and are not adopted: a full, own publication happens.
    assert gate(github, tmp_path) is None
    own = publish(github, direct_request(), tmp_path)
    assert own.transaction_id == committed.transaction_id
    assert github.write_count == 5
    # Forged transaction records from another author fail closed as an actor change.
    for number, body in reference.writes:
        if "AGENT_WORKFLOW_TRANSACTION" in body:
            github.seed(number, body, author=FOREIGN)
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    assert raised.value.code == "actor-change"
    assert any("ignored-foreign" in item for item in raised.value.problems)


def test_actor_change_fails_closed_without_a_competing_transaction(tmp_path):
    github = TransactionGitHub()
    publish(github, direct_request(), tmp_path)
    github.actor = ("other-bot", 9999)
    reset_authenticated_github_actor(github)
    writes = github.write_count
    for call in (lambda: publish(github, direct_request(), tmp_path), lambda: gate(github, tmp_path)):
        with pytest.raises(WorkflowTransactionError) as raised:
            call()
        assert raised.value.code == "actor-change"
        assert raised.value.recovery_action == RECOVERY_ORIGINAL_ACTOR
        assert ACTOR[0] in str(raised.value)
    assert github.write_count == writes


def test_contradictory_same_scope_record_stops_before_any_write(tmp_path):
    github = TransactionGitHub()
    github.fail_write(2, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    writes = github.write_count
    bodies = (github.bodies(PR), github.bodies(ISSUE))
    # A changed base is never a legitimate difference: no abort, no write.
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, direct_request(base="release"), tmp_path)
    error = raised.value
    assert error.code == "prepared-intent-contradiction"
    assert error.transaction_ids and error.expected_record_set and error.problems
    assert "base" in " ".join(error.problems)
    assert github.write_count == writes
    assert (github.bodies(PR), github.bodies(ISSUE)) == bodies


def test_edited_bound_record_is_a_contradiction_not_adopted(tmp_path):
    github = TransactionGitHub()
    github.fail_write(3, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    handoff_id = github.threads[ISSUE][-1]["id"]
    github.edit(handoff_id, github.threads[ISSUE][-1]["body"])
    writes = github.write_count
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, direct_request(), tmp_path)
    assert raised.value.code == "record-contradiction"
    assert github.write_count == writes


# ---------------------------------------------------------------------------
# Interleaved writers
# ---------------------------------------------------------------------------


def test_same_id_duplicates_canonicalize_to_the_earliest_comment(tmp_path):
    github = TransactionGitHub()
    intent_holder = {}

    def inject(fake):
        # A competing invocation writes the byte-identical prepared record first.
        reference = TransactionGitHub()
        publish(reference, direct_request(), tmp_path)
        intent_holder["body"] = reference.writes[0][1]
        fake.seed(PR, reference.writes[0][1])

    github.before_write(1, inject)
    committed = publish(github, direct_request(), tmp_path)
    state = next(iter(states(github, tmp_path)))
    assert len(state.prepared_duplicates) == 1
    assert state.prepared_comment.comment_id < state.prepared_duplicates[0].comment_id
    assert committed.state.terminal.prepared_comment_id == state.prepared_comment.comment_id


def _competitor_prepared(request, tmp_path) -> str:
    reference = TransactionGitHub()
    publish(reference, request, tmp_path)
    return reference.writes[0][1]


@pytest.mark.parametrize("competitor_first", [True, False])
def test_root_siblings_lowest_prepared_id_wins_even_if_the_competitor_never_resumes(
    tmp_path, competitor_first
):
    github = TransactionGitHub()
    other = direct_request(expected_closing_issue_ids=(ISSUE, 901))
    competitor = _competitor_prepared(other, tmp_path)
    if competitor_first:
        github.before_write(1, lambda fake: fake.seed(PR, competitor))
        with pytest.raises(WorkflowTransactionError) as raised:
            publish(github, direct_request(), tmp_path)
        # Our own transaction lost: it is aborted and a rerun finishes the winner.
        assert raised.value.code == CODE_SIBLING_LOST
        live = [item for item in states(github, tmp_path) if not item.aborted]
        assert [item.intent.expected_closing_issue_ids for item in live] == [(ISSUE, 901)]
        committed = publish(github, other, tmp_path)
        assert committed.intent.expected_closing_issue_ids == (ISSUE, 901)
    else:
        github.before_write(2, lambda fake: fake.seed(PR, competitor))
        committed = publish(github, direct_request(), tmp_path)
        assert committed.intent.expected_closing_issue_ids == (ISSUE,)
    found = states(github, tmp_path)
    aborted = [item for item in found if item.aborted]
    assert len(aborted) == 1 and aborted[0].terminal.abort_reason == "sibling-canonical"
    assert [item.committed for item in found if not item.aborted] == [True]
    assert gate(github, tmp_path).transaction_id == committed.transaction_id


@pytest.mark.parametrize("competitor_first", [True, False])
def test_successor_siblings_are_reconciled_below_the_strict_resolver(tmp_path, competitor_first):
    github = TransactionGitHub()
    publish(github, direct_request(), tmp_path)
    ours = direct_request(head_sha=HEAD_2, initial_coder_round=None)
    theirs = direct_request(
        expected_closing_issue_ids=(ISSUE, 901), initial_coder_round=None
    )
    reference = TransactionGitHub()
    reference.threads = {k: [dict(c) for c in v] for k, v in github.threads.items()}
    reference.next_id = github.next_id + 500
    publish(reference, theirs, tmp_path)
    competitor = reference.writes[0][1]
    index = github.write_count + (1 if competitor_first else 2)
    github.before_write(index, lambda fake: fake.seed(PR, competitor))
    if competitor_first:
        with pytest.raises(WorkflowTransactionError) as raised:
            publish(github, ours, tmp_path)
        assert raised.value.code == CODE_SIBLING_LOST
    else:
        committed = publish(github, ours, tmp_path)
        assert committed.intent.successor_kind == KIND_HEAD_ADVANCE
        assert gate(github, tmp_path, live_head=HEAD_2) is not None
    aborted = [item for item in states(github, tmp_path) if item.aborted]
    assert [item.terminal.abort_reason for item in aborted] == ["sibling-canonical"]


def test_gate_and_reader_never_reconcile_siblings(tmp_path):
    github = TransactionGitHub()
    github.seed(PR, prepared_comment(1, direct_intent()).body)
    github.seed(PR, prepared_comment(2, direct_intent(expected_closing_issue_ids=(ISSUE, 901))).body)
    for call in (
        lambda: gate(github, tmp_path),
        lambda: read_pr_transaction_views(github, make_config(tmp_path), PR, ISSUE),
    ):
        with pytest.raises(WorkflowTransactionError) as raised:
            call()
        assert raised.value.code == "divergent-transactions"
    assert github.write_count == 0


# ---------------------------------------------------------------------------
# Obsolete prepared intent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("boundary", [2, 3, 4, 5])
def test_stale_head_aborts_and_prepares_a_fresh_initial_transaction(tmp_path, boundary):
    github = TransactionGitHub()
    github.fail_write(boundary, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    fresh = direct_request(head_sha=HEAD_2, initial_coder_round=coder_round(HEAD_2))
    committed = publish(github, fresh, tmp_path)
    found = states(github, tmp_path)
    aborted = [item for item in found if item.aborted]
    assert [item.terminal.abort_reason for item in aborted] == ["stale-head"]
    assert aborted[0].terminal.differing_fields[0][0] == "head_sha"
    # Lineage is computed against the last committed transaction, never the aborted one.
    assert committed.intent.successor_kind == KIND_INITIAL
    assert committed.intent.predecessor_transaction_id is None
    assert committed.intent.head_sha == HEAD_2
    assert gate(github, tmp_path, live_head=HEAD_2).transaction_id == committed.transaction_id
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path, live_head=HEAD_1)


def test_abort_then_prepare_is_itself_replayable(tmp_path):
    github = TransactionGitHub()
    github.fail_write(2, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    fresh = direct_request(head_sha=HEAD_2, initial_coder_round=coder_round(HEAD_2))
    # Interrupted after the abort, then after the fresh prepare.
    for _ in range(2):
        github.fail_write(github.write_count + 2, FAIL_BEFORE_WRITE)
        with pytest.raises(WorkflowTransactionError):
            publish(github, fresh, tmp_path)
    committed = publish(github, fresh, tmp_path)
    assert committed.intent.head_sha == HEAD_2
    assert sum(item.aborted for item in states(github, tmp_path)) == 1


def test_closing_widening_before_commit_is_superseded_intent(tmp_path):
    github = TransactionGitHub()
    github.fail_write(3, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    committed = publish(github, direct_request(expected_closing_issue_ids=(ISSUE, 901)), tmp_path)
    aborted = [item for item in states(github, tmp_path) if item.aborted]
    assert [item.terminal.abort_reason for item in aborted] == ["superseded-intent"]
    # The aborted transaction's handoff is listed and inert.
    assert aborted[0].outcome(ENTRY_HANDOFF).comment_id is not None
    assert committed.handoff.handoff.expected_closing_issue_ids == (ISSUE, 901)


def test_change_between_adoption_and_the_terminal_write_aborts(tmp_path):
    github = TransactionGitHub()
    requests = iter([direct_request(head_sha=HEAD_2, initial_coder_round=coder_round(HEAD_2))])
    latest = {"request": direct_request()}

    def refresh():
        latest["request"] = next(requests, latest["request"])
        return latest["request"]

    committed = publish_transaction(github, direct_request(), tmp_path, refresh=refresh)
    assert committed.intent.head_sha == HEAD_2
    assert [i.terminal.abort_reason for i in states(github, tmp_path) if i.aborted] == ["stale-head"]


# ---------------------------------------------------------------------------
# Successors on a committed chain
# ---------------------------------------------------------------------------


def test_head_advance_inherits_the_handoff_and_contract(tmp_path):
    github = TransactionGitHub()
    first = publish(github, direct_request(), tmp_path)
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path, live_head=HEAD_2)
    assert raised.value.code == CODE_HEAD_NOT_COMMITTED
    before = github.write_count
    successor = ensure_head_transaction(
        github, config=make_config(tmp_path),
        request=direct_request(head_sha=HEAD_2, initial_coder_round=None),
    )
    assert successor.intent.successor_kind == KIND_HEAD_ADVANCE
    assert successor.intent.predecessor_transaction_id == first.transaction_id
    # Only the prepared and terminal records are written.
    assert github.write_count == before + 2
    assert successor.entry_comment_id(ENTRY_HANDOFF) == first.entry_comment_id(ENTRY_HANDOFF)
    assert successor.entry_comment_id(ENTRY_PR_CONTRACT) == first.entry_comment_id(ENTRY_PR_CONTRACT)
    assert gate(github, tmp_path, live_head=HEAD_2).transaction_id == successor.transaction_id
    # Equal heads: no write at all.
    ensure_head_transaction(
        github, config=make_config(tmp_path),
        request=direct_request(head_sha=HEAD_2, initial_coder_round=None),
    )
    assert github.write_count == before + 2


@pytest.mark.parametrize("boundary", [1, 2])
def test_head_advance_each_boundary_and_a_further_push(tmp_path, boundary):
    github = TransactionGitHub()
    publish(github, direct_request(), tmp_path)
    request = direct_request(head_sha=HEAD_2, initial_coder_round=None)
    github.fail_write(github.write_count + boundary, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, request, tmp_path)
    head_3 = "c" * 40
    committed = publish(github, replace(request, head_sha=head_3), tmp_path)
    assert committed.intent.head_sha == head_3
    assert committed.intent.successor_kind == KIND_HEAD_ADVANCE


def test_closing_widening_successor_declares_its_supersession(tmp_path):
    github = TransactionGitHub()
    first = publish(github, direct_request(), tmp_path)
    widened = publish(
        github,
        direct_request(expected_closing_issue_ids=(ISSUE, 901), initial_coder_round=None),
        tmp_path,
    )
    assert widened.intent.successor_kind == KIND_CLOSING_WIDENING
    contract = widened.contract.contract
    assert contract.supersession_kind == "closing-widening"
    assert contract.supersedes_record_hash == first.contract.record_hash


def test_ensure_head_transaction_leaves_a_legacy_pr_untouched(tmp_path):
    github = TransactionGitHub()
    github.seed(PR, v1_contract_comment(1).body)
    assert ensure_head_transaction(
        github, config=make_config(tmp_path), request=direct_request(head_sha=HEAD_2)
    ) is None
    assert github.write_count == 0


# ---------------------------------------------------------------------------
# Initial coder round
# ---------------------------------------------------------------------------


def test_new_session_without_the_coder_response_waives_the_round(tmp_path):
    github = TransactionGitHub()
    github.fail_write(4, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    committed = publish(github, direct_request(initial_coder_round=None), tmp_path)
    outcome = committed.state.outcome(ENTRY_INITIAL_CODER_ROUND)
    assert outcome.status == STATUS_WAIVED_CODER_RESPONSE and outcome.comment_id is None
    assert not any("Implemented the change." in body for body in github.bodies(PR))
    assert gate(github, tmp_path) is not None


def test_new_session_adopts_a_round_that_landed(tmp_path):
    github = TransactionGitHub()
    github.fail_write(4, WRITE_THEN_REPORT_FAILURE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    committed = publish(github, direct_request(initial_coder_round=None), tmp_path)
    assert committed.entry_comment_id(ENTRY_INITIAL_CODER_ROUND) is not None
    assert sum("Implemented the change." in body for body in github.bodies(PR)) == 1


@pytest.mark.parametrize(
    "metadata",
    [dict(role="reviewer"), dict(round_number=2), dict(subject=HEAD_2)],
)
def test_contradictory_tagged_round_stops_non_mutating(tmp_path, metadata):
    github = TransactionGitHub()
    github.fail_write(4, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    tx = states(github, tmp_path)[0].transaction_id
    fields = dict(flow="pr", role="coder", agent="Claude", round_number=1, subject=HEAD_1)
    fields.update(metadata)
    github.seed(PR, _attach_round_metadata(
        "Forged.\n-- Claude", PostedRoundMetadata(workflow_transaction_id=tx, **fields)
    ))
    writes = github.write_count
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, direct_request(), tmp_path)
    assert raised.value.code == "initial-coder-round-contradiction"
    assert github.write_count == writes


def test_gate_rechecks_the_committed_initial_coder_round(tmp_path):
    github = TransactionGitHub()
    committed = publish(github, direct_request(), tmp_path)
    round_id = committed.entry_comment_id(ENTRY_INITIAL_CODER_ROUND)
    original = next(c["body"] for c in github.threads[PR] if c["id"] == round_id)
    github.edit(round_id, original)
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    assert raised.value.code == "initial-coder-round-contradiction"
    github.delete(round_id)
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path)


# ---------------------------------------------------------------------------
# Gate: partial transactions, deletions, legacy
# ---------------------------------------------------------------------------


def test_partial_transaction_grants_nothing(tmp_path):
    github = TransactionGitHub()
    github.fail_write(5, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    # Prepared record, handoff, contract and coder round all exist and agree.
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    error = raised.value
    assert error.code == CODE_PENDING and error.recovery_action == RECOVERY_RERUN
    assert error.transaction_ids and error.expected_record_set and error.successor_kind
    authority = resolve_round_authority(lambda: None, lambda: gate(github, tmp_path))
    assert isinstance(authority, NoLiveHeadAuthority)


@pytest.mark.parametrize("marker_index", [0, 1])
def test_deleted_transaction_record_never_selects_the_legacy_path(tmp_path, marker_index):
    github = TransactionGitHub()
    publish(github, direct_request(), tmp_path)
    records = [c["id"] for c in github.threads[PR] if "AGENT_WORKFLOW_TRANSACTION" in c["body"]]
    github.delete(records[marker_index])
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    assert raised.value.code != CODE_HEAD_NOT_COMMITTED
    # An integrity failure stops the round unless a rerun can finish it.
    if marker_index == 0:
        with pytest.raises(WorkflowTransactionError):
            resolve_round_authority(lambda: None, lambda: gate(github, tmp_path))
        writes = github.write_count
        with pytest.raises(WorkflowTransactionError):
            publish(github, direct_request(), tmp_path)
        assert github.write_count == writes
    else:
        # The prepared record survives: a rerun re-appends only the terminal record.
        writes = github.write_count
        publish(github, direct_request(), tmp_path)
        assert github.write_count == writes + 1
        assert gate(github, tmp_path) is not None


def test_deleted_scheduler_checkpoint_fails_closed(tmp_path):
    github = TransactionGitHub()
    checkpoint_id = seed_plan(github)
    publish(github, plan_request(), tmp_path)
    assert gate(github, tmp_path, plan_candidate_key=plan_key()) is not None
    github.delete(checkpoint_id)
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path, plan_candidate_key=plan_key())
    assert raised.value.code == "scheduler-checkpoint-invalid"


def test_merged_pr_needs_the_terminal_state_form(tmp_path):
    github = TransactionGitHub()
    publish(github, direct_request(), tmp_path)
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path, pr_state="MERGED")
    assert gate(github, tmp_path, pr_state="MERGED", allow_terminal_pr_state=True) is not None


def test_legacy_pr_is_unchanged_and_receives_no_write(tmp_path):
    github = TransactionGitHub()
    github.seed(PR, v1_contract_comment(1).body)
    github.seed(ISSUE, v1_handoff_comment(2).body)
    assert gate(github, tmp_path) is None
    assert isinstance(
        resolve_round_authority(lambda: None, lambda: gate(github, tmp_path)), LegacyEra
    )
    assert publish(github, direct_request(initial_coder_round=None), tmp_path) is None
    assert github.write_count == 0


def test_legacy_pr_that_needs_a_write_is_upgraded_by_one_initial_transaction(tmp_path):
    github = TransactionGitHub()
    contract_id = github.seed(PR, v1_contract_comment(1).body)
    github.seed(ISSUE, v1_handoff_comment(2).body)
    committed = publish(
        github,
        direct_request(expected_closing_issue_ids=(ISSUE, 901), initial_coder_round=None),
        tmp_path,
    )
    assert committed.intent.successor_kind == KIND_INITIAL
    contract = committed.contract.contract
    assert contract.supersession_kind == "closing-widening"
    assert contract.supersedes_record_hash == pr_contract_record_hash(v1_contract())
    assert contract_id < committed.contract.comment_id
    # No v1 record is ever appended to the now transaction-era PR.
    assert gate(github, tmp_path) is not None


def test_legacy_pr_whose_flow_disagrees_is_not_upgradable(tmp_path):
    github = TransactionGitHub()
    seed_plan(github)
    github.seed(PR, v1_contract_comment(3).body)  # issue-implementation on the PR side
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, plan_request(initial_coder_round=None), tmp_path)
    assert raised.value.code == "legacy-flow-disagreement"
    assert github.write_count == 0


# ---------------------------------------------------------------------------
# Supersession table and diagnostics
# ---------------------------------------------------------------------------


def _resolved(contract):
    return ResolvedPrContract(contract, 5, pr_contract_record_hash(contract), "legacy")


@pytest.mark.parametrize(
    "prior, overrides, expected",
    [
        (None, {}, None),
        (v1_contract(), {}, None),
        (v1_contract(), {"expected_closing_issue_ids": (ISSUE, 901)}, "closing-widening"),
        (v1_contract(FLOW_APPROVED_PLAN), {}, "flow-correction"),
        (
            v1_contract(FLOW_APPROVED_PLAN),
            {"expected_closing_issue_ids": (ISSUE, 901)},
            "flow-correction-with-closing-widening",
        ),
    ],
)
def test_contract_supersession_is_a_pure_function_of_the_resolved_record(prior, overrides, expected):
    intent = direct_intent(**overrides)
    kind, record_hash = contract_supersession(intent, _resolved(prior) if prior else None)
    assert kind == expected
    assert record_hash == (pr_contract_record_hash(prior) if expected else None)


def test_a_narrowed_contract_is_a_contradiction():
    with pytest.raises(WorkflowTransactionError) as raised:
        contract_supersession(direct_intent(), _resolved(v1_contract(ids=(ISSUE, 901))))
    assert raised.value.code == "contract-supersession-invalid"


def test_request_has_no_flow_parameter_and_derives_it():
    assert "flow" not in {f for f in TransitionRequest.__dataclass_fields__ if "origin_flow" in f}
    assert direct_request().origin_flow == FLOW_ISSUE
    assert plan_request().origin_flow == FLOW_APPROVED_PLAN
    unowned = direct_request(origin_path=ORIGIN_PR_RESUME, primary_issue=None)
    assert unowned.origin_flow == "direct-pr"
    assert replace(unowned, unowned_managed_pr=True).origin_flow == "managed-pr"


# ---------------------------------------------------------------------------
# Managed-CI authorization entry through a stub codec
# ---------------------------------------------------------------------------


class StubAuthorizationCodec:
    """A minimal bound-authorization codec: payload JSON plus the transaction ID."""

    MARKER = "STUB-BOUND-AUTHORIZATION "

    def __init__(self):
        self.validated: list[tuple[int, bool]] = []
        self.invalid = False

    def _payloads(self, comment):
        return [
            json.loads(base64.b64decode(line[len(self.MARKER):]))
            for line in comment.body.splitlines() if line.startswith(self.MARKER)
        ]

    def transaction_ids(self, comment):
        return tuple(item["transaction_id"] for item in self._payloads(comment))

    def expected_body(self, payload, transaction_id):
        raw = json.dumps({**payload, "transaction_id": transaction_id}, sort_keys=True)
        return TrustedBody.canonical(
            "Bound authorization.\n" + self.MARKER + base64.b64encode(raw.encode()).decode(),
            expected_tokens=(),
        )

    def adoptable(self, comment, payload, transaction_id):
        return self._payloads(comment) == [{**payload, "transaction_id": transaction_id}]

    def validate(self, comment, *, state, lineage, pr_view, committed):
        self.validated.append((comment.comment_id, committed))
        if self.invalid:
            raise WorkflowTransactionError("invalid bound authorization", code="authorization-invalid")

    def digest(self, comment):
        return hashlib.sha256(comment.body.encode()).hexdigest()


def managed_request(codec, *, generation="gen-1", head=HEAD_1, kind=Granted, **overrides):
    return direct_request(
        head_sha=head, managed=kind({"kind": "creation", "head": head}, generation),
        authorization_codec=codec, **overrides,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5, 6])
def test_managed_activation_each_boundary_yields_one_bound_authorization(tmp_path, boundary, mode):
    github, codec = TransactionGitHub(), StubAuthorizationCodec()
    github.fail_write(boundary, mode)
    with pytest.raises(WorkflowTransactionError):
        publish(github, managed_request(codec), tmp_path)
    # No authority before the commit, even with the bound record read back.
    landed_commit = boundary == 6 and mode == WRITE_THEN_REPORT_FAILURE
    if github.bodies(PR) and not landed_commit:
        with pytest.raises(WorkflowTransactionError):
            gate(github, tmp_path, authorization_codec=codec)
    committed = publish(github, managed_request(codec), tmp_path)
    assert committed.intent.managed_ci_generation == "gen-1"
    assert sum(codec.MARKER in body for body in github.bodies(PR)) == 1
    assert gate(github, tmp_path, authorization_codec=codec) is not None
    assert (committed.entry_comment_id(ENTRY_AUTHORIZATION), True) in codec.validated


def test_gate_refuses_a_bound_authorization_that_fails_validation(tmp_path):
    github, codec = TransactionGitHub(), StubAuthorizationCodec()
    publish(github, managed_request(codec), tmp_path)
    codec.invalid = True
    with pytest.raises(WorkflowTransactionError) as raised:
        resolve_round_authority(
            lambda: None, lambda: gate(github, tmp_path, authorization_codec=codec)
        )
    assert raised.value.code == "authorization-invalid"
    # Presence at the recorded ID without a codec is never enough either.
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path)


def test_managed_head_advance_reissues_and_release_is_managed_ci_continuity(tmp_path):
    github, codec = TransactionGitHub(), StubAuthorizationCodec()
    publish(github, managed_request(codec), tmp_path)
    pushed = publish(
        github, managed_request(codec, head=HEAD_2, initial_coder_round=None), tmp_path
    )
    assert pushed.intent.successor_kind == KIND_HEAD_ADVANCE
    assert pushed.intent.entry(ENTRY_AUTHORIZATION).disposition == "reissued"
    released = publish(
        github,
        managed_request(
            codec, head=HEAD_2, generation="released", kind=Released, initial_coder_round=None
        ),
        tmp_path,
    )
    assert released.intent.successor_kind == KIND_MANAGED_CI_CONTINUITY
    assert released.intent.entry(ENTRY_HANDOFF).disposition == "inherited"


def test_unavailable_managed_input_stops_before_any_write(tmp_path):
    github, codec = TransactionGitHub(), StubAuthorizationCodec()
    publish(github, managed_request(codec), tmp_path)
    writes = github.write_count
    request = direct_request(
        head_sha=HEAD_2, managed=Unavailable("managed head has no continuity provenance"),
        initial_coder_round=None,
    )
    authority = resolve_round_authority(
        lambda: publish(github, request, tmp_path), lambda: gate(github, tmp_path)
    )
    assert isinstance(authority, NoLiveHeadAuthority)
    assert github.write_count == writes


# ---------------------------------------------------------------------------
# Canonical-PR discovery and publication routing
# ---------------------------------------------------------------------------


def test_v2_only_handoff_resolves_without_a_pr_number(tmp_path):
    github = TransactionGitHub()
    committed = publish(github, direct_request(), tmp_path)
    writes, reads_only = github.write_count, make_config(tmp_path)
    found = discover_canonical_issue_pr(github, reads_only, ISSUE)
    assert found.pr_number == PR and found.transaction_id == committed.transaction_id
    assert found.handoff.era == "transaction" and found.handoff.flow == FLOW_ISSUE
    route = route_issue_publication(github, reads_only, ISSUE)
    assert isinstance(route, Committed) and route.canonical.pr_number == PR
    assert github.write_count == writes


def test_all_legacy_issue_gives_the_v1_result_and_no_candidate_is_none(tmp_path):
    github = TransactionGitHub()
    assert discover_canonical_issue_pr(github, make_config(tmp_path), ISSUE) is None
    assert isinstance(route_issue_publication(github, make_config(tmp_path), ISSUE), NoCandidate)
    github.seed(ISSUE, v1_handoff_comment(1).body)
    found = discover_canonical_issue_pr(github, make_config(tmp_path), ISSUE)
    assert found.pr_number == PR and found.era == "legacy"
    assert found.handoff.legacy_record is not None
    assert isinstance(route_issue_publication(github, make_config(tmp_path), ISSUE), Legacy)


@pytest.mark.parametrize("boundary", [3, 4, 5])
def test_interrupted_issue_publication_routes_as_recoverable(tmp_path, boundary):
    github = TransactionGitHub()
    github.fail_write(boundary, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    config = make_config(tmp_path)
    # The authority form refuses; it never falls back.
    with pytest.raises(WorkflowTransactionError) as raised:
        discover_canonical_issue_pr(github, config, ISSUE)
    assert raised.value.code == CODE_PARTIAL_CANDIDATE
    writes = github.write_count
    route = route_issue_publication(github, config, ISSUE)
    assert isinstance(route, Recoverable) and route.pr_number == PR
    assert len(route.pending_transaction_ids) == 1
    # A recoverable route carries no record, handoff view, or committed transaction.
    assert not hasattr(route, "canonical") and not hasattr(route, "record")
    assert github.write_count == writes
    publish(github, direct_request(initial_coder_round=None), tmp_path)
    assert discover_canonical_issue_pr(github, config, ISSUE).pr_number == PR


def test_failure_before_the_handoff_leaves_no_candidate(tmp_path):
    github = TransactionGitHub()
    github.fail_write(2, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    assert isinstance(route_issue_publication(github, make_config(tmp_path), ISSUE), NoCandidate)


def test_pending_successor_routes_with_a_predecessor_binding(tmp_path):
    github = TransactionGitHub()
    first = publish(github, direct_request(), tmp_path)
    github.fail_write(github.write_count + 2, FAIL_BEFORE_WRITE)
    widened = direct_request(expected_closing_issue_ids=(ISSUE, 901), initial_coder_round=None)
    with pytest.raises(WorkflowTransactionError):
        publish(github, widened, tmp_path)
    config = make_config(tmp_path)
    with pytest.raises(WorkflowTransactionError) as raised:
        discover_canonical_issue_pr(github, config, ISSUE)
    assert raised.value.code == CODE_PENDING
    route = route_issue_publication(github, config, ISSUE)
    assert isinstance(route, RecoverableSuccessor)
    assert route.predecessor.transaction_id == first.transaction_id
    assert route.predecessor.expected_closing_issue_ids == (ISSUE,)
    # The single pending successor stays routable after it published its handoff.
    github.fail_write(github.write_count + 2, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, widened, tmp_path)
    assert isinstance(route_issue_publication(github, config, ISSUE), RecoverableSuccessor)
    assert publish(github, widened, tmp_path).intent.successor_kind == KIND_CLOSING_WIDENING


@pytest.mark.parametrize("published", ["neither", "one", "both"])
def test_successor_siblings_route_together_over_filtered_views(tmp_path, published):
    github = TransactionGitHub()
    first = publish(github, direct_request(), tmp_path)
    requests = [
        direct_request(expected_closing_issue_ids=(ISSUE, 901), initial_coder_round=None),
        direct_request(expected_closing_issue_ids=(ISSUE, 902), initial_coder_round=None),
    ]
    publish_count = {"neither": 0, "one": 1, "both": 2}[published]
    base_threads = {k: [dict(c) for c in v] for k, v in github.threads.items()}
    for index, request in enumerate(requests):
        scratch = TransactionGitHub()
        scratch.threads = {k: [dict(c) for c in v] for k, v in base_threads.items()}
        scratch.next_id = github.next_id
        stop = 3 if index < publish_count else 2
        scratch.fail_write(stop, FAIL_BEFORE_WRITE)
        with pytest.raises(WorkflowTransactionError):
            publish(scratch, request, tmp_path)
        for number, body in scratch.writes:
            github.seed(number, body)
    config = make_config(tmp_path)
    with pytest.raises(WorkflowTransactionError):
        discover_canonical_issue_pr(github, config, ISSUE)
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path)
    writes = github.write_count
    route = route_issue_publication(github, config, ISSUE)
    assert isinstance(route, RecoverableSuccessor)
    assert len(route.pending_transaction_ids) == 2
    assert route.predecessor.transaction_id == first.transaction_id
    assert github.write_count == writes
    # The seam picks the lowest prepared ID and commits without the competitor.
    committed = publish(github, requests[0], tmp_path)
    assert committed.intent.expected_closing_issue_ids == (ISSUE, 901)
    aborted = [item for item in states(github, tmp_path) if item.aborted]
    assert [item.terminal.abort_reason for item in aborted] == ["sibling-canonical"]
    if published == "both":
        assert aborted[0].outcome(ENTRY_HANDOFF).comment_id is not None
    assert discover_canonical_issue_pr(github, config, ISSUE).pr_number == PR


def test_root_siblings_that_published_handoffs_route_as_recoverable(tmp_path):
    github = TransactionGitHub()
    for ids in ((ISSUE, 901), (ISSUE, 902)):
        scratch = TransactionGitHub()
        scratch.next_id = github.next_id
        scratch.fail_write(3, FAIL_BEFORE_WRITE)
        with pytest.raises(WorkflowTransactionError):
            publish(scratch, direct_request(expected_closing_issue_ids=ids), tmp_path)
        for number, body in scratch.writes:
            github.seed(number, body)
    route = route_issue_publication(github, make_config(tmp_path), ISSUE)
    assert isinstance(route, Recoverable) and len(route.pending_transaction_ids) == 2


def test_non_routable_partials_raise_from_the_router(tmp_path):
    github = TransactionGitHub()
    github.fail_write(3, FAIL_BEFORE_WRITE)
    with pytest.raises(WorkflowTransactionError):
        publish(github, direct_request(), tmp_path)
    prepared_id = next(
        c["id"] for c in github.threads[PR] if "AGENT_WORKFLOW_TRANSACTION" in c["body"]
    )
    github.delete(prepared_id)
    with pytest.raises(WorkflowTransactionError) as raised:
        route_issue_publication(github, make_config(tmp_path), ISSUE)
    assert raised.value.code == CODE_PARTIAL_CANDIDATE


# ---------------------------------------------------------------------------
# Import isolation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module",
    ["workflow_transaction_publication", "issue_pr_handoff", "phase_progress", "managed_ci"],
)
def test_module_imports_in_isolation(module):
    source = Path(__file__).resolve().parents[1] / "src"
    result = subprocess.run(
        [sys.executable, "-c", f"import coding_review_agent_loop.{module}"],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(source)},
    )
    assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Round-metadata transaction tag
# ---------------------------------------------------------------------------


def test_round_metadata_tag_is_encoded_only_when_set_and_rejects_malformed_values():
    from coding_review_agent_loop.errors import AgentLoopError
    from coding_review_agent_loop.round_state import (
        _decode_round_metadata,
        _encode_round_metadata,
    )
    from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

    untagged = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=1, subject=HEAD_1
    )
    encoded = _encode_round_metadata(untagged)
    assert "workflow_transaction_id" not in decode_mapping(encoded)
    assert _encode_round_metadata(_decode_round_metadata(encoded)) == encoded
    tagged = replace(untagged, workflow_transaction_id="a" * 64)
    assert _decode_round_metadata(_encode_round_metadata(tagged)).workflow_transaction_id == "a" * 64
    malformed = encode_mapping({**decode_mapping(encoded), "workflow_transaction_id": "nope"})
    with pytest.raises(AgentLoopError):
        _decode_round_metadata(malformed)
