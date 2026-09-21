"""The transaction-bound managed-CI authorization: codec, rule, and seam codec (#946)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from agent_loop_helpers import make_config
from workflow_transaction_helpers import (
    ACTOR,
    FAIL_BEFORE_WRITE,
    HEAD_1,
    HEAD_2,
    ISSUE,
    PR,
    REPO,
    WRITE_THEN_REPORT_FAILURE,
    TransactionGitHub,
)

from coding_review_agent_loop.errors import AgentLoopError, WorkflowTransactionError
from coding_review_agent_loop.github import read_authenticated_protocol_comments
from coding_review_agent_loop.managed_ci import (
    ManagedCiIssueAuthorization,
    format_issue_created_authorization_comment,
    parse_issue_created_authorization_comment,
)
from coding_review_agent_loop.managed_ci_bound_authorization import (
    BOUND_AUTHORIZATION_MARKER,
    KIND_ORDINARY_RELEASE,
    KIND_PLAN_REBIND,
    BoundAuthorizationCodec,
    bind_v1_authorization,
    build_plan_rebind_payload,
    decode_bound_authorization,
    format_bound_authorization_comment,
    granted_generation,
    ordinary_release_payload,
    parse_bound_authorization_comment,
    released_generation,
)
from coding_review_agent_loop.protocol_markers import PR_BODY_SURFACE, TrustedBody
from coding_review_agent_loop.workflow_transaction import (
    ENTRY_AUTHORIZATION,
    KIND_HEAD_ADVANCE,
    KIND_MANAGED_CI_CONTINUITY,
    collect_transactions,
)
from coding_review_agent_loop.workflow_transaction_publication import (
    ORIGIN_DIRECT_ISSUE,
    Granted,
    PublicationViews,
    Released,
    TransitionRequest,
    publish_transition,
    require_committed_transaction,
)

TX = "c" * 64
EVENT = 9001


def v1(kind="creation", head=HEAD_1, **overrides) -> ManagedCiIssueAuthorization:
    fields = dict(
        kind=kind, repository=REPO, issue_number=ISSUE, pr_number=PR, base_ref="main",
        head_sha=head, actor_login=ACTOR[0], actor_id=ACTOR[1], protection="voluntary",
        waiver="allow-unprotected-managed-ci", nonce="nonce-1", label_event_id=EVENT,
    )
    fields.update(overrides)
    return ManagedCiIssueAuthorization(**fields)


def creation(head=HEAD_1, **overrides):
    return bind_v1_authorization(v1(head=head, **overrides), grant_anchor_event_id=EVENT)


def fresh(head=HEAD_1, *, predecessor=None, **overrides):
    fields = dict(overrides)
    if predecessor is not None:
        fields.update(predecessor_comment_id=predecessor[0], predecessor_head=predecessor[1])
    return bind_v1_authorization(v1("fresh", head=head, **fields), grant_anchor_event_id=EVENT)


def release(head=HEAD_1):
    return ordinary_release_payload(
        repository=REPO, issue_number=ISSUE, pr_number=PR, base_ref="main", head_sha=head,
        actor_login=ACTOR[0], actor_id=ACTOR[1],
    )


def request(payload, *, head=None, **overrides) -> TransitionRequest:
    managed = (
        Released(payload, payload.generation())
        if payload.kind == KIND_ORDINARY_RELEASE
        else Granted(payload, payload.generation())
    )
    fields = dict(
        repository=REPO, pr_number=PR, base="main", head_sha=head or payload.head_sha,
        origin_path=ORIGIN_DIRECT_ISSUE, expected_closing_issue_ids=(ISSUE,),
        primary_issue=ISSUE, managed=managed, authorization_codec=BoundAuthorizationCodec(),
    )
    fields.update(overrides)
    return TransitionRequest(**fields)


def publish(github, payload, tmp_path, **overrides):
    return publish_transition(
        github, config=make_config(tmp_path), request=request(payload, **overrides)
    )


def views(github, tmp_path) -> PublicationViews:
    config = make_config(tmp_path)
    pr = read_authenticated_protocol_comments(github, config=config, surface_kind="pr", number=PR)
    issue = read_authenticated_protocol_comments(
        github, config=config, surface_kind="issue", number=ISSUE
    )
    return PublicationViews(pr, issue, issue)


def gate(github, tmp_path, *, live_head=HEAD_1):
    return require_committed_transaction(
        views(github, tmp_path), repository=REPO, pr_number=PR, issue_number=ISSUE,
        live_head=live_head, authorization_codec=BoundAuthorizationCodec(),
    )


def bound_comments(github):
    return [
        item for item in github.threads.get(PR, []) if BOUND_AUTHORIZATION_MARKER in item["body"]
    ]


def rewrite_bound(github, comment, **changes) -> None:
    """Replace a bound comment in place, keeping the envelope unedited."""
    record = parse_bound_authorization_comment(comment["body"])
    comment["body"] = str(format_bound_authorization_comment(replace(record, **changes)))


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload", [
    creation(), fresh(predecessor=(1234, HEAD_2)), release(),
    bind_v1_authorization(
        v1("continuity", predecessor_comment_id=7, predecessor_head=HEAD_2,
           round_comment_ids=(8, 9), approved_plan_hash="p" * 16),
        grant_anchor_event_id=EVENT, upgraded_from_comment_id=11,
    ),
])
def test_bound_record_round_trips_byte_identically(payload):
    record = replace(payload, transaction_id=TX)
    body = format_bound_authorization_comment(record)
    assert parse_bound_authorization_comment(str(body)) == record
    assert str(format_bound_authorization_comment(record)) == str(body)
    # The v1 parser and its substring scans never see a bound record.
    assert parse_issue_created_authorization_comment(str(body)) is None
    assert "AGENT_MANAGED_CI_ISSUE_AUTHORIZATION_V1" not in str(body)


def test_v1_record_is_untouched_and_is_not_a_bound_record():
    body = str(format_issue_created_authorization_comment(v1()))
    assert parse_bound_authorization_comment(body) is None
    assert parse_issue_created_authorization_comment(body) == v1()


def test_bound_record_is_pr_comment_only():
    body = str(format_bound_authorization_comment(replace(creation(), transaction_id=TX)))
    marker = body.split("\n\n", 1)[1]
    with pytest.raises(AgentLoopError, match="not allowed on the pr_body surface"):
        TrustedBody.canonical(
            marker, surface=PR_BODY_SURFACE, expected_tokens=(BOUND_AUTHORIZATION_MARKER,)
        )


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(version=1),
    lambda p: p.update(kind="adopted"),
    lambda p: p.update(transaction_id="short"),
    lambda p: p.update(extra=1),
    lambda p: p.pop("grant_anchor_event_id"),
    lambda p: p.pop("nonce"),
    lambda p: p.update(waiver="other"),
    lambda p: p.update(actor_id=True),
    lambda p: p.update(round_comment_ids=[0]),
])
def test_decoder_refuses_malformed_granted_payloads(mutate):
    payload = replace(creation(), transaction_id=TX).to_payload()
    mutate(payload)
    with pytest.raises(AgentLoopError):
        decode_bound_authorization(payload)


@pytest.mark.parametrize("field,value", [
    ("nonce", "n"), ("label_event_id", 1), ("grant_anchor_event_id", 1),
    ("approved_plan_hash", "p"), ("predecessor_head", HEAD_2), ("protection", "voluntary"),
])
def test_a_release_that_carries_any_grant_field_is_refused(field, value):
    payload = replace(release(), transaction_id=TX).to_payload()
    payload[field] = value
    with pytest.raises(AgentLoopError):
        decode_bound_authorization(payload)


def test_generation_excludes_head_kind_nonce_and_live_label_event():
    base = creation()
    assert base.generation() == creation(head=HEAD_2, nonce="other").generation()
    assert base.generation() == fresh().generation()
    # The record's own label event is not the anchor.
    assert base.generation() == replace(base, label_event_id=EVENT + 5).generation()
    assert base.generation() != replace(base, grant_anchor_event_id=EVENT + 5).generation()
    assert base.generation() != replace(base, approved_plan_hash="p" * 16).generation()
    assert base.generation() != replace(base, protection="plan_limited").generation()
    assert base.generation() == granted_generation(
        repository=REPO.lower(), issue_number=ISSUE, pr_number=PR, base_ref="main",
        actor_id=ACTOR[1], protection="voluntary", waiver="allow-unprotected-managed-ci",
        grant_anchor_event_id=EVENT, approved_plan_hash=None,
    )
    assert release().generation() == released_generation(
        repository=REPO, issue_number=ISSUE, pr_number=PR, base_ref="main", actor_id=ACTOR[1]
    )
    assert release().generation() == release(HEAD_2).generation() != base.generation()


def test_plan_rebind_payload_is_deterministic_and_mints_nothing():
    committed = replace(creation(approved_plan_hash="p1"), transaction_id=TX)
    rebind = build_plan_rebind_payload(
        committed, committed_comment_id=55, new_plan_hash="p2", live_head=HEAD_1
    )
    assert rebind == build_plan_rebind_payload(
        committed, committed_comment_id=55, new_plan_hash="p2", live_head=HEAD_1
    )
    assert (rebind.kind, rebind.nonce, rebind.label_event_id, rebind.grant_anchor_event_id) == (
        KIND_PLAN_REBIND, committed.nonce, EVENT, EVENT
    )
    assert (rebind.predecessor_comment_id, rebind.predecessor_head) == (55, HEAD_1)
    assert rebind.approved_plan_hash == "p2" and not rebind.round_comment_ids
    unavailable = dict(committed_comment_id=55, new_plan_hash="p2", live_head=HEAD_1)
    assert build_plan_rebind_payload(committed, **{**unavailable, "live_head": HEAD_2}) is None
    assert build_plan_rebind_payload(committed, **{**unavailable, "new_plan_hash": "p1"}) is None
    assert build_plan_rebind_payload(replace(release(), transaction_id=TX), **unavailable) is None


# ---------------------------------------------------------------------------
# The real codec through the real seam and gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [FAIL_BEFORE_WRITE, WRITE_THEN_REPORT_FAILURE])
@pytest.mark.parametrize("boundary", [1, 2, 3, 4, 5])
def test_managed_publication_each_boundary_yields_one_bound_record(tmp_path, boundary, mode):
    github = TransactionGitHub()
    github.fail_write(boundary, mode)
    with pytest.raises(WorkflowTransactionError):
        publish(github, creation(), tmp_path)
    landed_commit = boundary == 5 and mode == WRITE_THEN_REPORT_FAILURE
    if github.bodies(PR) and not landed_commit:
        with pytest.raises(WorkflowTransactionError):
            gate(github, tmp_path)
    # A new session mints a different nonce; a record that landed is adopted.
    committed = publish(github, creation(nonce="nonce-2"), tmp_path)
    assert len(bound_comments(github)) == 1
    record = parse_bound_authorization_comment(bound_comments(github)[0]["body"])
    assert record.transaction_id == committed.intent.transaction_id
    assert committed.intent.managed_ci_generation == record.generation()
    assert committed.entry_comment_id(ENTRY_AUTHORIZATION) == bound_comments(github)[0]["id"]
    assert gate(github, tmp_path) is not None
    writes = github.write_count
    publish(github, creation(nonce="nonce-3"), tmp_path)
    assert github.write_count == writes


def test_push_release_and_regrant_walk_the_lineage(tmp_path):
    github = TransactionGitHub()
    first = publish(github, creation(), tmp_path)
    first_id = first.entry_comment_id(ENTRY_AUTHORIZATION)
    # An ordinary push on a granted PR keeps the generation: head-advance.
    pushed = publish(github, fresh(HEAD_2, predecessor=(first_id, HEAD_1)), tmp_path)
    assert pushed.intent.successor_kind == KIND_HEAD_ADVANCE
    pushed_id = pushed.entry_comment_id(ENTRY_AUTHORIZATION)
    released = publish(github, release(HEAD_2), tmp_path)
    assert released.intent.successor_kind == KIND_MANAGED_CI_CONTINUITY
    assert gate(github, tmp_path, live_head=HEAD_2) is not None
    # A same-head regrant names the nearest granted ancestor, past the release.
    regrant = publish(github, fresh(HEAD_2, predecessor=(pushed_id, HEAD_2)), tmp_path)
    assert regrant.intent.successor_kind == KIND_MANAGED_CI_CONTINUITY
    assert regrant.intent.managed_ci_generation == first.intent.managed_ci_generation
    assert gate(github, tmp_path, live_head=HEAD_2) is not None
    assert len(bound_comments(github)) == 4


def test_fresh_record_must_name_the_nearest_granted_ancestor(tmp_path):
    github = TransactionGitHub()
    publish(github, creation(), tmp_path)
    writes = github.write_count
    with pytest.raises(WorkflowTransactionError) as raised:
        publish(github, fresh(HEAD_2, predecessor=(1, HEAD_1)), tmp_path)
    assert raised.value.code == "authorization-invalid"
    assert "nearest granted ancestor" in str(raised.value)
    # The bound record landed but the transaction never committed: no authority.
    assert github.write_count == writes + 2
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path, live_head=HEAD_2)


VALIDATION_TABLE = [
    ("head", dict(head_sha=HEAD_2), "head"),
    ("issue", dict(issue_number=ISSUE + 1), "issue"),
    ("base", dict(base_ref="release"), "base"),
    ("plan", dict(approved_plan_hash="p" * 16), "approved_plan_hash"),
    ("anchor", dict(grant_anchor_event_id=EVENT + 1), "generation"),
    ("protection", dict(protection="plan_limited"), "generation"),
    ("creation-with-rounds", dict(round_comment_ids=(5,)), "continuity fields"),
    ("kind-continuity", dict(kind="continuity"), "granted predecessor"),
    ("kind-rebind", dict(kind="plan-rebind"), "granted predecessor"),
    ("upgrade-on-native", dict(upgraded_from_comment_id=1), "durable upgrade rule"),
]


@pytest.mark.parametrize("name,changes,field", VALIDATION_TABLE, ids=[r[0] for r in VALIDATION_TABLE])
def test_gate_judges_the_committed_record_against_its_transaction(tmp_path, name, changes, field):
    github = TransactionGitHub()
    publish(github, creation(), tmp_path)
    assert gate(github, tmp_path) is not None
    # Substitute the payload under the committed comment ID and transaction ID.
    rewrite_bound(github, bound_comments(github)[0], **changes)
    writes = github.write_count
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    assert raised.value.code == "authorization-invalid"
    assert field in str(raised.value)
    assert github.write_count == writes


def test_gate_refuses_an_edited_or_duplicated_bound_record(tmp_path):
    github = TransactionGitHub()
    publish(github, creation(), tmp_path)
    comment = bound_comments(github)[0]
    github.seed(PR, str(format_bound_authorization_comment(
        replace(parse_bound_authorization_comment(comment["body"]), nonce="other")
    )))
    with pytest.raises(WorkflowTransactionError, match="same transaction differs"):
        gate(github, tmp_path)
    github.delete(github.threads[PR][-1]["id"])
    assert gate(github, tmp_path) is not None
    github.edit(comment["id"], comment["body"])
    with pytest.raises(WorkflowTransactionError, match="edited"):
        gate(github, tmp_path)


def test_unmanaged_transaction_admits_no_bound_record(tmp_path):
    github = TransactionGitHub()
    committed = publish_transition(
        github, config=make_config(tmp_path),
        request=TransitionRequest(
            repository=REPO, pr_number=PR, base="main", head_sha=HEAD_1,
            origin_path=ORIGIN_DIRECT_ISSUE, expected_closing_issue_ids=(ISSUE,),
            primary_issue=ISSUE,
        ),
    )
    pr_view = views(github, tmp_path).pr_view
    state = next(
        item for item in collect_transactions(pr_view, repository=REPO, pr_number=PR)
        if item.transaction_id == committed.intent.transaction_id
    )
    forged = github.seed(PR, str(format_bound_authorization_comment(
        replace(creation(), transaction_id=state.transaction_id)
    )))
    pr_view = views(github, tmp_path).pr_view
    with pytest.raises(WorkflowTransactionError, match="unmanaged"):
        BoundAuthorizationCodec().validate(
            pr_view.comment(forged), state=state, lineage=None, pr_view=pr_view, committed=False
        )


# ---------------------------------------------------------------------------
# Legacy upgrade
# ---------------------------------------------------------------------------


def upgraded(source_id, source=None):
    return bind_v1_authorization(
        source or v1(), grant_anchor_event_id=EVENT, upgraded_from_comment_id=source_id
    )


def test_upgrade_copies_the_unbound_record_and_keeps_validating(tmp_path):
    github = TransactionGitHub()
    source_id = github.seed(PR, str(format_issue_created_authorization_comment(v1())))
    publish(github, upgraded(source_id), tmp_path)
    assert gate(github, tmp_path) is not None
    # Later comments, including a later unbound record, cannot change the rule's result.
    github.seed(PR, str(format_issue_created_authorization_comment(v1(nonce="later"))))
    assert gate(github, tmp_path) is not None


@pytest.mark.parametrize("break_it,field", [
    (lambda github, source_id: github.delete(source_id), "durable upgrade rule"),
    (lambda github, source_id: github.edit(
        source_id, str(format_issue_created_authorization_comment(v1()))
    ), "edited"),
    (lambda github, source_id: rewrite_bound(
        github, bound_comments(github)[0], nonce="substituted"
    ), "v1 field differs"),
])
def test_upgrade_with_a_missing_edited_or_divergent_source_is_an_integrity_error(
    tmp_path, break_it, field
):
    github = TransactionGitHub()
    source_id = github.seed(PR, str(format_issue_created_authorization_comment(v1())))
    publish(github, upgraded(source_id), tmp_path)
    break_it(github, source_id)
    with pytest.raises(WorkflowTransactionError) as raised:
        gate(github, tmp_path)
    assert raised.value.code == "authorization-invalid" and field in str(raised.value)


def test_upgrade_must_name_the_record_the_durable_rule_selects(tmp_path):
    github = TransactionGitHub()
    older = github.seed(PR, str(format_issue_created_authorization_comment(v1())))
    github.seed(PR, str(format_issue_created_authorization_comment(v1(nonce="newer"))))
    with pytest.raises(WorkflowTransactionError, match="durable upgrade rule"):
        publish(github, upgraded(older), tmp_path)
    with pytest.raises(WorkflowTransactionError):
        gate(github, tmp_path)


def test_upgrade_is_refused_on_a_successor(tmp_path):
    github = TransactionGitHub()
    source_id = github.seed(PR, str(format_issue_created_authorization_comment(v1())))
    publish(github, upgraded(source_id), tmp_path)
    second = github.seed(
        PR, str(format_issue_created_authorization_comment(v1(head=HEAD_2, nonce="n2")))
    )
    with pytest.raises(WorkflowTransactionError, match="not a plain initial root"):
        publish(github, upgraded(second, v1(head=HEAD_2, nonce="n2")), tmp_path)
