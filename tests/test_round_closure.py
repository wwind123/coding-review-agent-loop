"""The closed-incomplete round record and its dormant readers (#2064, stage 1).

Rows ``closure-record-reader`` and ``untrusted-closure-record``: a trusted
closed-incomplete record closes its PR or plan round, its reviewer records are
inert history, the round's ledger is reconstructed from the record (including
evidence obligations at their reserved item numbers), and a record from an
untrusted author is never admitted.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.plan_round_closure import (
    _plan_round_is_reconciled,
    project_plan_history,
    select_plan_round_anchor,
)
from coding_review_agent_loop.pr_round_ingestion import AdmittedPrComment
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_closure import (
    ROUND_CLOSED_INCOMPLETE_PHASE,
    build_round_closure_payload,
    closure_item_number_high_water,
    decode_round_closure_payload,
    is_round_closed_incomplete,
    is_round_closure_boundary,
    restore_closure_evidence_obligations,
)
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    PostedRoundRecord,
    _attach_round_metadata,
    _plan_subject,
)
from coding_review_agent_loop.round_state import (
    _extract_round_metadata_records,
    _resume_plan_round,
    _resume_pr_round,
    closed_round_review_indexes,
    drop_inert_post_reconciliation_checkpoints,
    inert_post_reconciliation_checkpoint_indexes,
)
from coding_review_agent_loop.unresolved_items import evidence_obligation_identity

HEAD = "c" * 40
EVIDENCE = "Attach a live run of the migration at this exact head."
PRIOR = UnresolvedReviewItem(
    item_id="item-3", reviewer="Codex", source_round=1, text="Guard the retry bound.",
    status="blocking", source_status="blocking",
)
NEW = UnresolvedReviewItem(
    item_id="item-4", reviewer="Gemini", source_round=2, text="Close the file handle.",
    status="blocking", source_status="blocking",
)


def _payload(flow="pr", round_number=2, subject=HEAD, *, obligations=True, high_water=5):
    return build_round_closure_payload(
        flow=flow, round_number=round_number, subject=subject,
        published=[("Codex", 11), ("Gemini", 12)],
        unreplayable=[("Antigravity", "spool outcome missing")],
        evidence_obligations=[{
            "reviewer": "Codex", "source_round": round_number, "head": subject,
            "text": EVIDENCE, "item_number": 5,
        }] if obligations else [],
        item_number_high_water=high_water,
    )


def _closure(flow="pr", round_number=2, subject=HEAD, *, new_items=(NEW,), payload="default", **extra):
    return PostedRoundMetadata(
        flow=flow, role="summary", agent="Orchestrator", round_number=round_number,
        subject=subject, prior_items=(PRIOR,), new_items=new_items,
        phase=ROUND_CLOSED_INCOMPLETE_PHASE,
        round_closure=_payload(flow, round_number, subject) if payload == "default" else payload,
        **extra,
    )


def _review(agent, state, *, flow="pr", round_number=2, subject=HEAD):
    return PostedRoundMetadata(
        flow=flow, role="reviewer", agent=agent, round_number=round_number, subject=subject,
        prior_items=(PRIOR,), state=state,
        **({"phase": "publication"} if flow == "pr" else {}),
    )


def _pr_comments(*, closure=None, tail=()):
    bodies = [
        _attach_round_metadata("Codex approves.", _review("Codex", "approved")),
        _attach_round_metadata("Gemini approves.", _review("Gemini", "approved")),
    ]
    if closure is not None:
        bodies.append(_attach_round_metadata("Round 2 closed, incomplete.", closure))
    bodies.extend(tail)
    return [SimpleNamespace(body=body) for body in bodies]


# --- strict codec -------------------------------------------------------------


def test_payload_decodes_strictly_and_round_trips_through_round_metadata():
    payload = _payload()
    assert payload is not None
    assert payload["published"] == [{"agent": "Codex", "comment_id": 11}, {"agent": "Gemini", "comment_id": 12}]
    (record,) = _extract_round_metadata_records(_pr_comments(closure=_closure())[2:], flow="pr")
    assert record.metadata.round_closure == payload
    assert is_round_closed_incomplete(record.metadata)
    assert is_round_closure_boundary(record.metadata)
    assert closure_item_number_high_water(record.metadata) == 5


@pytest.mark.parametrize("mutate", [
    lambda raw: raw.update(round=3),
    lambda raw: raw.update(subject="d" * 40),
    lambda raw: raw.update(flow="plan"),
    lambda raw: raw.update(extra=1),
    lambda raw: raw.update(published=[]),
    lambda raw: raw.update(unreplayable=[]),
    lambda raw: raw.update(published=[{"agent": "Codex", "comment_id": 1}, {"agent": "Codex", "comment_id": 2}]),
    lambda raw: raw.update(unreplayable=[{"agent": "Codex", "cause": " "}]),
    lambda raw: raw.update(item_number_high_water=4),  # below a reserved obligation number
    lambda raw: raw.update(item_number_high_water=True),
    lambda raw: raw["evidence_obligations"][0].update(item_number=0),
    lambda raw: raw["evidence_obligations"][0].update(source_round=3),
    lambda raw: raw["evidence_obligations"][0].pop("head"),
], ids=lambda _: "")
def test_malformed_or_misbound_payload_decodes_as_none(mutate):
    raw = {key: (list(map(dict, value)) if isinstance(value, list) else value)
           for key, value in _payload().items()}
    mutate(raw)
    assert decode_round_closure_payload(raw, flow="pr", round_number=2, subject=HEAD) is None


def _legacy_malformed():
    from coding_review_agent_loop.signed_input_record import SIGNED_INPUT_SUPERSESSION_PHASE
    return PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=HEAD,
        prior_items=(PRIOR,), phase=SIGNED_INPUT_SUPERSESSION_PHASE,
        signed_input_supersession={"round": 2},
    )


def _colliding():
    # The reserved obligation number 5 names an unrelated ordinary finding.
    other = UnresolvedReviewItem(
        item_id="item-5", reviewer="Gemini", source_round=2, text="Unrelated finding.",
        status="blocking", source_status="blocking",
    )
    return _closure(new_items=(NEW, other))


@pytest.mark.parametrize("closure", [
    lambda: _closure(payload={"round": 2}),
    _legacy_malformed,
    _colliding,
], ids=["malformed-payload", "malformed-legacy-supersession", "obligation-number-collision"])
def test_unvalidated_closure_summary_never_closes_or_settles_the_round(closure):
    """Item-1/item-3 class: a closure-phase summary that does not validate is
    neither a closure nor a reconciliation; the round stays open for settlement."""
    from coding_review_agent_loop.round_closure import is_unvalidated_closure_summary
    comments = _pr_comments(closure=closure())
    (_, _, record) = _extract_round_metadata_records(comments, flow="pr")
    assert not is_round_closure_boundary(record.metadata)
    assert is_unvalidated_closure_summary(record.metadata)
    records = _extract_round_metadata_records(comments, flow="pr")
    assert closed_round_review_indexes(records) == frozenset()
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert resumed.reconciliation_record is None
    # Ordinary settlement stays available: the round is not read as reconciled.
    assert resumed.reconciled is False
    assert {r.metadata.agent for r in resumed.completed_reviews} == {"Codex", "Gemini"}


def test_malformed_record_never_closes_the_round():
    comments = _pr_comments(closure=_closure(payload={"round": 2}))
    (_, _, record) = _extract_round_metadata_records(comments, flow="pr")
    assert record.metadata.round_closure is None
    assert not is_round_closure_boundary(record.metadata)
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert resumed.reconciliation_record is None and resumed.reconciled is False
    assert {r.metadata.agent for r in resumed.completed_reviews} == {"Codex", "Gemini"}


def test_already_materialized_obligation_at_its_reserved_number_restores_idempotently():
    """Item-3: the reserved number may name the same obligation, never another item."""
    seed = _closure(new_items=())
    materialized, _ = restore_closure_evidence_obligations([], seed, next_item_number=6)
    (obligation,) = materialized
    closure = _closure(new_items=(NEW, obligation))
    assert is_round_closed_incomplete(closure)
    items, next_number = restore_closure_evidence_obligations(
        [PRIOR, NEW, obligation], closure, next_item_number=6)
    assert [item.item_id for item in items] == ["item-3", "item-4", "item-5"]
    assert next_number == 6
    assert not is_round_closed_incomplete(_colliding())


@pytest.mark.parametrize("change", [{"agent": "Codex"}, {"role": "reviewer"}])
def test_only_an_orchestrator_summary_is_a_closure_record(change):
    import dataclasses
    metadata = dataclasses.replace(_closure(), **change)
    assert not is_round_closed_incomplete(metadata)


# --- closure-record-reader: PR flow ------------------------------------------


def test_pr_closure_record_closes_round_and_its_reviews_are_inert():
    """Row closure-record-reader (PR): round N is closed, its reviews never resume."""
    comments = _pr_comments(closure=_closure())
    records = _extract_round_metadata_records(comments, flow="pr")
    assert closed_round_review_indexes(records) == {0, 1}
    assert [r.index for r in drop_inert_post_reconciliation_checkpoints(records)] == [2]

    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert resumed.round_number == 2 and resumed.reconciled
    # Round-N reviews are never admitted as resumed or completed reviews.
    assert resumed.completed_reviews == ()
    assert resumed.reconciliation_record.index == 2
    assert resumed.reconciliation_record.metadata.phase == ROUND_CLOSED_INCOMPLETE_PHASE
    # The record's ledger is the resumed ledger; the reserved obligation number
    # raises the item-number high-water mark.
    assert resumed.prior_items == (PRIOR,)
    assert resumed.reconciliation_record.metadata.new_items == (NEW,)
    assert resumed.next_unresolved_item_number == 6


def test_pr_closure_record_closes_round_for_scheduler_checkpoints():
    """A later scheduler checkpoint posted into the closed round is inert (#1553 rule)."""
    stray = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=HEAD,
        prior_items=(PRIOR,), phase="scheduler-prelaunch",
    )
    comments = _pr_comments(closure=_closure(), tail=[_attach_round_metadata("Stray.", stray)])
    records = _extract_round_metadata_records(comments, flow="pr")
    assert inert_post_reconciliation_checkpoint_indexes(records) == {3}


def test_closure_of_another_head_or_round_is_a_freshness_boundary_for_earlier_reviews():
    """#2064 item-7: a closed-incomplete record of another round or head closes
    no review of its own round here, but round N+1 needs the full board, so
    every earlier reviewer record is history.  The same-round view used by the
    no-progress window leaves them live."""
    from coding_review_agent_loop.round_closure import closed_round_reviewer_indexes

    for closure in (_closure(subject="d" * 40), _closure(round_number=3)):
        records = _extract_round_metadata_records(_pr_comments(closure=closure), flow="pr")
        reviewers = {r.index for r in records if r.metadata.role == "reviewer"}
        assert reviewers and closed_round_review_indexes(records) == reviewers
        assert closed_round_reviewer_indexes(records, flow="pr", freshness=False) == frozenset()


def test_pr_closure_blocks_approval_carry_unit():
    from coding_review_agent_loop.code_approval import _latest_pr_approved_reviews_for_head
    open_round = _latest_pr_approved_reviews_for_head(
        _pr_comments(), head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert set(open_round) == {"Codex", "Gemini"}
    closed = _latest_pr_approved_reviews_for_head(
        _pr_comments(closure=_closure()), head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert closed == {}


def test_restart_restores_evidence_obligation_with_reserved_number_even_without_new_items():
    """Ledger semantics: obligations come from the record, not the inert reviews."""
    comments = _pr_comments(closure=_closure(new_items=()))
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    closure = resumed.reconciliation_record.metadata
    assert closure.new_items == () and resumed.completed_reviews == ()
    items, next_number = restore_closure_evidence_obligations(
        list(closure.prior_items), closure, next_item_number=resumed.next_unresolved_item_number,
    )
    assert [item.item_id for item in items] == ["item-3", "item-5"]
    obligation = items[-1]
    assert obligation.reviewer == "Codex" and obligation.lifecycle == "evidence_deferred"
    assert obligation.obligation_identity == evidence_obligation_identity(EVIDENCE, "Codex")
    assert next_number == 6
    # Restoring twice is idempotent: same items, never a new number.
    again, again_next = restore_closure_evidence_obligations(items, closure, next_item_number=next_number)
    assert [item.item_id for item in again] == ["item-3", "item-5"] and again_next == 6


def test_restore_is_a_no_op_for_ordinary_reconciliation():
    reconciliation = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject=HEAD,
        prior_items=(PRIOR,), phase="reconciliation",
    )
    assert restore_closure_evidence_obligations([PRIOR], reconciliation, next_item_number=4) == ([PRIOR], 4)


# --- untrusted-closure-record ---------------------------------------------------


def _admitted(body, index, *, author_id):
    return AdmittedPrComment(
        author="bot" if author_id == 1 else "mallory", created_at=f"2026-11-22T00:00:{index:02d}Z",
        body=body, comment_id=100 + index, author_id=author_id,
        invocation_actor_id=1, invocation_actor_login="bot",
        admitted_body=body, admitted_protocol_body=body,
    )


def test_untrusted_closure_record_is_ignored_by_author_authentication():
    """Row untrusted-closure-record: pr_round_ingestion masks the forged record."""
    bodies = [comment.body for comment in _pr_comments(closure=_closure())]
    trusted = [_admitted(body, i, author_id=1) for i, body in enumerate(bodies)]
    forged = [*trusted[:2], _admitted(bodies[2], 2, author_id=2)]

    trusted_resume = _resume_pr_round(trusted, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert trusted_resume.reconciliation_record is not None

    records = _extract_round_metadata_records(forged, flow="pr")
    assert [record.index for record in records] == [0, 1]
    assert closed_round_review_indexes(records) == frozenset()
    resumed = _resume_pr_round(forged, head_sha=HEAD, configured_reviewers=("codex", "gemini"))
    assert resumed.reconciliation_record is None
    assert {r.metadata.agent for r in resumed.completed_reviews} == {"Codex", "Gemini"}


# --- closure-record-reader: plan flow -------------------------------------------


PLAN = "Plan text for the closure test."
PLAN_SUBJECT = _plan_subject(PLAN)


def _plan_records(*, closure=True):
    metas = [
        PostedRoundMetadata(flow="plan", role="coder", agent="Claude", round_number=2,
                            subject=PLAN_SUBJECT, prior_items=(PRIOR,), canonical_plan=PLAN),
        _review("Codex", "approved", flow="plan", subject=PLAN_SUBJECT),
        _review("Gemini", "approved", flow="plan", subject=PLAN_SUBJECT),
    ]
    if closure:
        metas.append(_closure("plan", subject=PLAN_SUBJECT,
                              payload=_payload("plan", subject=PLAN_SUBJECT, obligations=False, high_water=4)))
    return metas


def _plan_comments(metas):
    bodies = []
    for metadata in metas:
        text = PLAN if metadata.role == "coder" else f"{metadata.agent} {metadata.role}."
        bodies.append(_attach_round_metadata(text + "\n<!-- AGENT_PLAN_STATE: blocking -->", metadata))
    return [IssueComment(author="bot", created_at=f"2026-11-22T00:0{i}:00Z", body=b) for i, b in enumerate(bodies)]


def test_plan_closure_record_settles_its_round():
    """Row closure-record-reader (plan): reconciled, projected and anchored at the record."""
    records = tuple(PostedRoundRecord(index=i, metadata=m, body="") for i, m in enumerate(_plan_records()))
    assert _plan_round_is_reconciled(records)
    assert _plan_round_is_reconciled((records[-1],))
    # Closed-round plan reviews leave every projected reader (panel evidence,
    # carried approvals, visibility, resume) and the record anchors the ledger.
    projected = project_plan_history(records)
    assert [record.index for record in projected] == [0, 3]
    assert select_plan_round_anchor(projected).index == 3

    unclosed = records[:3]
    assert not _plan_round_is_reconciled(unclosed)
    assert [record.index for record in project_plan_history(unclosed)] == [0, 1, 2]


def test_plan_resume_rolls_into_next_round_without_closed_reviews():
    comments = _plan_comments(_plan_records())
    plan_text, resumed = _resume_plan_round(comments, configured_reviewers=("codex", "gemini"))
    assert plan_text == PLAN
    assert resumed.round_number == 2 and resumed.reconciled
    # No closed-round plan review resumes or can approve the plan.
    assert resumed.completed_reviews == ()
    assert resumed.prior_items == (PRIOR,)
    assert resumed.current_round_new_items == (NEW,)
    assert resumed.next_unresolved_item_number == 5

    # Control: without the record the same reviews resume in round 2.
    _, open_round = _resume_plan_round(_plan_comments(_plan_records(closure=False)),
                                       configured_reviewers=("codex", "gemini"))
    assert not open_round.reconciled
    assert {r.metadata.agent for r in open_round.completed_reviews} == {"Codex", "Gemini"}


def test_legacy_supersession_predicate_stays_pr_only():
    from coding_review_agent_loop.signed_input_record import SIGNED_INPUT_SUPERSESSION_PHASE
    legacy_plan = PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", round_number=2, subject=PLAN_SUBJECT,
        phase=SIGNED_INPUT_SUPERSESSION_PHASE, signed_input_supersession={"round": 2},
    )
    assert not is_round_closure_boundary(legacy_plan)


# --- head advance after a closure boundary (item-2) ---------------------------


NEXT_HEAD = "e" * 40


@pytest.mark.parametrize("new_items", [(NEW,), ()], ids=["with-new-items", "obligation-only"])
def test_unrecorded_head_advance_after_closure_recovers_the_closure_ledger(new_items):
    """A coder record, published reviews and a closure at the old head, then a
    push: recovery takes the closure's ledger, including its reserved
    evidence obligation, never only the coder's prior items."""
    coder = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=2, subject=HEAD,
        prior_items=(PRIOR,), raw_structured_coder_response="Coder follow-up.",
    )
    comments = [SimpleNamespace(body=_attach_round_metadata("Coder follow-up.", coder)),
                *_pr_comments(closure=_closure(new_items=new_items))]
    resumed = _resume_pr_round(comments, head_sha=NEXT_HEAD, configured_reviewers=("codex", "gemini"))
    assert resumed is not None and resumed.unrecorded_head_advance
    assert resumed.round_number == 2
    assert resumed.completed_reviews == ()
    ids = [item.item_id for item in resumed.prior_items]
    assert ids == ["item-3", *(["item-4"] if new_items else []), "item-5"]
    obligation = resumed.prior_items[-1]
    assert obligation.obligation_identity == evidence_obligation_identity(EVIDENCE, "Codex")
    assert resumed.next_unresolved_item_number == 6


@pytest.mark.parametrize("closure", [
    lambda: PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD,
        phase=ROUND_CLOSED_INCOMPLETE_PHASE,
        round_closure=build_round_closure_payload(
            flow="pr", round_number=1, subject=HEAD,
            published=[("Codex", 11), ("Gemini", 12)],
            unreplayable=[("Antigravity", "spool outcome missing")],
            item_number_high_water=0,
        ),
    ),
    lambda: PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD,
        phase=__import__("coding_review_agent_loop.signed_input_record", fromlist=["x"]).SIGNED_INPUT_SUPERSESSION_PHASE,
        surfaced_reviewer_requirement_ids=("hr-" + "a" * 64, "hr-" + "b" * 64),
        signed_input_supersession=__import__(
            "coding_review_agent_loop.signed_input_record", fromlist=["x"]
        ).decode_supersession_payload({
            "round": 1, "head": HEAD,
            "superseded": [{"agent": "Codex", "comment_id": 11, "route": "carrier"}],
            "previous_requirement_ids": ["hr-" + "a" * 64],
            "requirement_ids": ["hr-" + "a" * 64, "hr-" + "b" * 64],
            "item_number_high_water": 0,
        }, round_number=1, subject=HEAD),
    ),
], ids=["closed-incomplete", "legacy-supersession"])
def test_empty_ledger_closure_without_coder_is_a_coherent_head_advance_handoff(closure):
    """Item-5: round 1 with no coder record, approving reviews and a closure
    carrying no ledger, then a push: resume proceeds as an ordinary fresh
    review (no refusal) and never admits the closed reviews."""
    metadata = closure()
    assert is_round_closure_boundary(metadata)
    review = lambda agent: PostedRoundMetadata(
        flow="pr", role="reviewer", agent=agent, round_number=1, subject=HEAD,
        state="approved", phase="publication",
    )
    comments = [SimpleNamespace(body=_attach_round_metadata(text, meta)) for text, meta in (
        ("Codex approves.", review("Codex")),
        ("Gemini approves.", review("Gemini")),
        ("Round 1 closed.", metadata),
    )]
    assert _resume_pr_round(comments, head_sha=NEXT_HEAD, configured_reviewers=("codex", "gemini")) is None
    # Control: without the closure the same reviews are the coherence proof.
    assert _resume_pr_round(comments[:2], head_sha=NEXT_HEAD, configured_reviewers=("codex", "gemini")) is None
    from coding_review_agent_loop.code_approval import _latest_pr_approved_reviews_for_head
    assert _latest_pr_approved_reviews_for_head(
        comments, head_sha=HEAD, configured_reviewers=("codex", "gemini")) == {}
