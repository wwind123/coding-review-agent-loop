"""Signed-input growth now closes a published PR round under the one rule (#2054, #2064).

The sequence from llm-dialectic PR #2310: every review of a round is published,
the conflicting-classification stop tells the operator to post a follow-up
signed comment, and the resume after that comment must open a fresh round
rather than ask for the round's reviews to be deleted.  Since #2064 stage 2 the
round-wide closure rule handles this and every other rejection reason: one
closed-incomplete record closes the round, its published findings carry
forward as prior items, and no new signed-input supersession record is
posted.  A legacy supersession record still closes its round.
"""

import dataclasses
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.pr_loop as pr_loop
from coding_review_agent_loop import response_validation, signed_input_supersession as sis
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.github import _parse_pr_human_requirements
from coding_review_agent_loop.publication_resume import PublicationCarrier, classify_frozen_sequence
from coding_review_agent_loop.review_spool import ReviewRoundSpool
from coding_review_agent_loop.round_closure import ROUND_CLOSED_INCOMPLETE_PHASE
from coding_review_agent_loop.signed_input_record import (
    SIGNED_INPUT_SUPERSESSION_PHASE,
    decode_supersession_payload,
)

from agent_loop_helpers import FakeRunner, make_config
from pr_loop_test_helpers import (
    _EVIDENCE_TEXT,
    _association,
    _classified_review,
    _pr_records,
    _rewrite_pr_metadata,
    _signed,
)

HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"
_ACTOR = ("coding-review-agent-loop", 424242)


def _rids(runner):
    return [r.requirement_id for r in _parse_pr_human_requirements(runner.pr_payload)]


def _conflict_stop(tmp_path, monkeypatch, *, carrier=False, max_rounds=None, policy=None):
    """Run A: both reviews publish, then the role-conflict stop fires (#2310 shape)."""
    runner = FakeRunner()
    if carrier:
        # The real freeze/resume hook needs the REST actor surface.
        runner.authenticated_actor = _ACTOR
        runner.serve_rest_issue_comments = True
    runner.pr_payload["comments"].append(_signed("A human must accept the policy run.", 1))
    (rid,) = _rids(runner)
    extra = {"max_rounds": max_rounds} if max_rounds is not None else {}
    if policy is not None:
        extra["pr_review_policy"] = policy
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, **extra)
    runner.codex_outputs = [_classified_review(
        [_association(rid, "human-only", request=_EVIDENCE_TEXT)], hr=True, evidence=[_EVIDENCE_TEXT])] * 2
    runner.gemini_outputs = [_classified_review(
        [_association(rid, "code")], reviewer="Google Gemini", hr=True,
        state="blocking", blocking=["Fix the policy check."])] * 2
    with monkeypatch.context() as patch:
        if carrier:
            # A crash between publication and spool cleanup leaves the carriers.
            patch.setattr(ReviewRoundSpool, "remove", lambda self, name: None)
            patch.setattr(ReviewRoundSpool, "discard", lambda self: None)
        with pytest.raises(HumanDecisionRequiredError, match="remained conflicting") as stop:
            run_pr_loop(runner, pr_number=77, config=config)
    # The hint routes the operator to a new comment, never an edit.
    assert "Post a new follow-up signed comment (do not edit the existing signed comment)" in str(stop.value)
    publications = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"]
    assert sorted(r.metadata.agent for r in publications) == ["Codex", "Gemini"]
    assert not [r for r in _pr_records(runner) if r.metadata.phase == "reconciliation"]
    return runner, config


def _follow_hint(runner):
    """The operator posts a new follow-up signed comment, as the hint says."""
    runner.pr_payload["comments"].append(
        _signed(f"The policy run is accepted for {HEAD}; keep the policy check code-actionable.", 2))


def _queue_fresh_round(runner, *, carried=("item-2",)):
    """Round 2: Codex accepts its carried evidence request (item-1) by citing the
    added signed answer; both reviewers classify against the grown set and
    dispose of the round-1 findings the closure carried forward."""
    old, new = _rids(runner)[0], _rids(runner)[-1]
    answer = {"item_id": "item-1", "disposition": "resolved", "note": "The signed answer accepts the run.",
              "evidence_requirement_ids": [new]}
    fixed = [{"item_id": item, "disposition": "resolved", "note": "The policy check is fixed."} for item in carried]
    runner.codex_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], hr=True, dispositions=[answer, *fixed])]
    runner.gemini_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], reviewer="Google Gemini", hr=True,
        dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Requester-owned."}, *fixed])]


def _agent_commands(runner, start=0):
    return [cmd[0] for cmd, _ in runner.commands[start:] if cmd[:1] in (["codex"], ["gemini"], ["claude"])]


def _no_deletes(runner):
    return not any(cmd[:2] == ["gh", "api"] and "DELETE" in cmd for cmd, _ in runner.commands)


def _supersessions(runner):
    return [r for r in _pr_records(runner) if r.metadata.phase == SIGNED_INPUT_SUPERSESSION_PHASE]


def _closures(runner):
    return [r for r in _pr_records(runner) if r.metadata.phase == ROUND_CLOSED_INCOMPLETE_PHASE]


def _spy_closure_post(monkeypatch, runner):
    """Record how many reviewer agents had run when the closure record was posted."""
    seen = []
    real = pr_loop.post_pr_comment

    def spy(*args, **kwargs):
        if "closed, incomplete" in str(kwargs.get("body", "")):
            seen.append(len(_agent_commands(runner)))
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "post_pr_comment", spy)
    return seen


class _Interrupted(Exception):
    pass


def _interrupt_after_closure(runner, config, monkeypatch):
    """Post the closed-incomplete record, then stop before any round-2 work."""
    real = pr_loop.post_pr_comment

    def post(*args, **kwargs):
        real(*args, **kwargs)
        if "closed, incomplete" in str(kwargs.get("body", "")):
            raise _Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(pr_loop, "post_pr_comment", post)
        with pytest.raises(_Interrupted):
            run_pr_loop(runner, pr_number=77, config=config)
    (record,) = _closures(runner)
    assert _pr_records(runner)[-1].index == record.index
    return record


def _post_legacy_supersession(runner, config):
    """A legacy #2054 supersession record, as an older release wrote it."""
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from coding_review_agent_loop.round_state import _resume_pr_round

    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config))
    reviews = [r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.round_number == 1]
    previous = sorted({str(i) for r in reviews for i in r.metadata.surfaced_reviewer_requirement_ids})
    current = sorted(_rids(runner))
    payload = decode_supersession_payload({
        "round": 1, "head": HEAD,
        "superseded": [{"agent": r.metadata.agent, "comment_id": None, "route": "edit-history"} for r in reviews],
        "previous_requirement_ids": previous, "requirement_ids": current,
        "item_number_high_water": resumed.next_unresolved_item_number - 1,
    }, round_number=1, subject=HEAD)
    assert payload is not None
    metadata = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD,
        prior_items=resumed.prior_items, phase=SIGNED_INPUT_SUPERSESSION_PHASE,
        surfaced_reviewer_requirement_ids=tuple(current), signed_input_supersession=payload,
    )
    pr_loop.post_pr_comment(runner, config=config, pr_number=77, body=_attach_round_metadata(
        "Review round 1 is superseded by new signed input. No comment needs to be deleted.", metadata))
    (record,) = _supersessions(runner)
    return record


def _cid(comment):
    return comment.get("id", comment.get("databaseId"))


def _round_spool(tmp_path, config, round_number=1):
    return pr_loop._review_round_spool(
        config, surface="pr", number=77, round_number=round_number, subject=HEAD)


def _assert_closed_and_rolled_over(runner, before_commands, *, published=("Codex", "Gemini")):
    """One closed-incomplete record closed round 1 and round 2 ran the full board."""
    (record,) = _closures(runner)
    payload = record.metadata.round_closure
    assert record.metadata.round_number == 1 and record.metadata.subject == HEAD
    assert [e["agent"] for e in payload["published"]] == sorted(published)
    assert all(isinstance(e["comment_id"], int) for e in payload["published"])
    assert {e["agent"] for e in payload["unreplayable"]} == set(published)
    assert not _supersessions(runner)
    later = [r for r in _pr_records(runner) if r.index > record.index]
    assert later and all(r.metadata.round_number >= 2 for r in later)
    round_two = [r for r in later if r.metadata.role == "reviewer"]
    assert sorted({r.metadata.agent for r in round_two}) == ["Codex", "Gemini"]
    assert sorted(_agent_commands(runner, before_commands))[:2] == ["codex", "gemini"]
    assert _no_deletes(runner)
    assert not any("Delete these" in c or "partial-round recovery" in c for c in runner.comments)
    return record


# --- signed-growth-closure ----------------------------------------------------


@pytest.mark.parametrize("carrier", [False, True], ids=["no-carrier", "carrier"])
def test_growth_after_conflict_stop_closes_round_one_without_deletion(tmp_path, monkeypatch, carrier):
    """`signed-growth-closure` (#2064): a closed-incomplete record, not a supersession record."""
    runner, config = _conflict_stop(tmp_path, monkeypatch, carrier=carrier)
    round_one = [c["body"] for c in runner.pr_payload["comments"]]
    _follow_hint(runner)
    _queue_fresh_round(runner)
    before = len(runner.commands)
    seen = _spy_closure_post(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    record = _assert_closed_and_rolled_over(runner, before)
    # The carried ledger is round 1's, plus Codex's durable evidence request
    # that resume restores for a review invalidated by new input (#1922).
    assert [(i.item_id, i.reviewer, i.obligation_kind) for i in record.metadata.prior_items] == [
        ("item-1", "Codex", "human-exact-head-evidence")]
    # Gemini's published blocker carries forward as a fresh, never-reused item;
    # Codex's evidence request is already item-1, so no duplicate obligation.
    assert [(i.item_id, i.reviewer, i.text) for i in record.metadata.new_items] == [
        ("item-2", "Gemini", "Fix the policy check.")]
    assert record.metadata.round_closure["evidence_obligations"] == []
    assert record.metadata.round_closure["item_number_high_water"] == 2
    # Save before emit: posted before any round-2 reviewer launch.
    assert seen == [len(_agent_commands(runner)) - len(_agent_commands(runner, before))]
    reconciliation = next(r for r in _pr_records(runner)
                          if r.index > record.index and r.metadata.phase == "reconciliation")
    assert reconciliation.metadata.round_number == 2
    assert {i.item_id for i in reconciliation.metadata.prior_items} == {"item-1", "item-2"}
    # Round-1 comments are untouched and nothing was deleted or asked for.
    assert [c["body"] for c in runner.pr_payload["comments"][:len(round_one)]] == round_one


def test_interrupt_after_closure_record_resumes_into_round_two(tmp_path, monkeypatch):
    """`closure-idempotent` (PR): a rerun after the record never posts a second one."""
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    _interrupt_after_closure(runner, config, monkeypatch)
    _queue_fresh_round(runner)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_closures(runner)) == 1
    round_two = [r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.round_number == 2]
    assert sorted(r.metadata.agent for r in round_two) == ["Codex", "Gemini"]
    assert _no_deletes(runner)
    assert not any("Delete these" in c for c in runner.comments)


def test_closed_reviews_are_inert_for_approval_panel_and_resume(tmp_path, monkeypatch):
    """`closed-review-no-approval` on the live writer: closed reviews never approve or resume."""
    from coding_review_agent_loop.code_approval import _latest_pr_approved_reviews_for_head
    from coding_review_agent_loop.round_closure import closed_round_reviewer_indexes
    from coding_review_agent_loop.round_state import _resume_pr_round, drop_inert_post_reconciliation_checkpoints

    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    _interrupt_after_closure(runner, config, monkeypatch)
    history = _pr_records(runner)
    closed = closed_round_reviewer_indexes(history, flow="pr")
    assert closed == {r.index for r in history if r.metadata.role == "reviewer"}
    assert not closed & {r.index for r in drop_inert_post_reconciliation_checkpoints(history)}
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config))
    assert resumed.round_number == 1 and resumed.reconciled and resumed.completed_reviews == ()
    assert resumed.reconciliation_record.metadata.phase == ROUND_CLOSED_INCOMPLETE_PHASE
    # Withdrawing the added source makes Codex's approval cover the live set
    # again, yet the closed round's approval never carries.
    added = next(i for i, c in enumerate(runner.pr_payload["comments"])
                 if "keep the policy check code-actionable" in c["body"])
    del runner.pr_payload["comments"][added]
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    codex = next(r for r in _pr_records(runner) if r.metadata.agent == "Codex" and r.metadata.role == "reviewer")
    assert codex.metadata.state == "approved"
    assert response_validation._resumed_pr_reviewer_matches_requirements(codex, requirements)
    assert _latest_pr_approved_reviews_for_head(
        comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config),
        human_requirements=requirements) == {}


def test_legacy_supersession_record_still_closes_its_round(tmp_path, monkeypatch):
    """Migration (#2064): an existing #2054 record keeps closing its round for one release."""
    from coding_review_agent_loop.round_state import _resume_pr_round

    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    record = _post_legacy_supersession(runner, config)
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config))
    assert resumed.round_number == 1 and resumed.reconciled and resumed.completed_reviews == ()
    assert resumed.reconciliation_record.metadata.phase == SIGNED_INPUT_SUPERSESSION_PHASE
    # The legacy record settles no findings: round 2 starts from its ledger.
    _queue_fresh_round(runner, carried=())
    before = len(runner.commands)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_supersessions(runner)) == 1 and not _closures(runner)
    later = [r for r in _pr_records(runner) if r.index > record.index]
    assert sorted({r.metadata.agent for r in later if r.metadata.role == "reviewer"}) == ["Codex", "Gemini"]
    assert all(r.metadata.round_number >= 2 for r in later)
    assert sorted(_agent_commands(runner, before)) == ["codex", "gemini"]
    assert _no_deletes(runner)


def test_single_published_review_on_two_reviewer_board_closes(tmp_path, monkeypatch):
    """Only Codex published before the stop; Gemini never began.  Gemini's turn
    would sit beside Codex's public review, so the round closes."""
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    gemini = next(r for r in _pr_records(runner) if r.metadata.agent == "Gemini")
    del runner.pr_payload["comments"][gemini.index]
    assert [r.metadata.agent for r in _pr_records(runner) if r.metadata.role == "reviewer"] == ["Codex"]
    _follow_hint(runner)
    _queue_fresh_round(runner, carried=())
    before = len(runner.commands)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (record,) = _closures(runner)
    payload = record.metadata.round_closure
    assert [e["agent"] for e in payload["published"]] == ["Codex"]
    assert {e["agent"] for e in payload["unreplayable"]} == {"Codex", "Gemini"}
    round_two = [r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.round_number == 2]
    assert sorted(r.metadata.agent for r in round_two) == ["Codex", "Gemini"]
    assert sorted(_agent_commands(runner, before)) == ["codex", "gemini"]
    assert _no_deletes(runner)
    assert not any("Delete these" in c for c in runner.comments)


# --- every former refusal reason closes the round -----------------------------


def _edit_review_body(runner, agent):
    record = next(r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.agent == agent)
    comment = runner.pr_payload["comments"][record.index]
    assert "Reviewed." in comment["body"]
    # The visible body changes; the round metadata (IDs, head, seat,
    # acquisition) is kept exactly.
    comment["body"] = comment["body"].replace("Reviewed.", "Reviewed, then quietly edited.", 1)
    return record.index


def test_edited_review_with_retained_carrier_closes_the_round(tmp_path, monkeypatch):
    runner, config = _conflict_stop(tmp_path, monkeypatch, carrier=True)
    index = _edit_review_body(runner, "Codex")
    _follow_hint(runner)
    _queue_fresh_round(runner)
    spool = _round_spool(tmp_path, config)
    state, raw = spool.publication_state("Codex")
    assert state == "carrier"
    # The edited public body no longer matches its frozen bytes.
    listing = SimpleNamespace(
        comment_id=_cid(runner.pr_payload["comments"][index]),
        body=runner.pr_payload["comments"][index]["body"],
        created=PublicationCarrier.from_dict(raw).prepared_at)
    import datetime
    listing.created = datetime.datetime.fromisoformat(listing.created)
    result = classify_frozen_sequence(PublicationCarrier.from_dict(raw), [listing])
    assert (result.kind, result.matched_count) == ("prefix", 0)
    before = len(runner.commands)

    # Formerly a carrier stop pointing at the recovery list (#1258).
    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    _assert_closed_and_rolled_over(runner, before)
    assert spool.publication_state("Codex")[0] == "absent"  # the round spool is discarded


@pytest.mark.parametrize("variant", ["edited-history", "missing-history", "metadata-mismatch",
                                     "edited-signed-source", "edited-and-added-signed-source"])
def test_former_refusals_close_the_round_and_roll_over(tmp_path, monkeypatch, variant):
    """Edits, unproven histories, acquisition mismatches and withdrawn sources all
    used to keep the partial-round refusal; the one rule closes the round for any reason."""
    policy = "selective-intermediate" if variant == "metadata-mismatch" else None
    runner, config = _conflict_stop(tmp_path, monkeypatch, policy=policy)
    if variant in {"edited-history", "missing-history"}:
        record = next(r for r in _pr_records(runner) if r.metadata.agent == "Gemini")
        comment = runner.pr_payload["comments"][record.index]
        comment["_edit_history"] = None if variant == "missing-history" else {
            "lastEditedAt": "2026-11-21T18:00:00Z", "includesCreatedEdit": True,
            "userContentEdits": {"totalCount": 1, "nodes": [
                {"editedAt": "2026-11-21T18:00:00Z", "deletedAt": None, "diff": "edited"}]}}
        _follow_hint(runner)
    elif variant == "metadata-mismatch":
        _rewrite_pr_metadata(runner, lambda payload: (
            {**payload, "configured_effort": "a-different-effort"}
            if payload.get("role") == "reviewer" and payload.get("agent") == "Codex" else None))
        _follow_hint(runner)
    else:
        signed = next(c for c in runner.pr_payload["comments"] if c["body"].endswith("-- Human Reviewer"))
        signed["body"] = "A human must accept the policy run, edited.\n\n-- Human Reviewer"
        if variant == "edited-and-added-signed-source":
            _follow_hint(runner)
    if variant.startswith("edited-signed") or variant.startswith("edited-and-added"):
        rids = _rids(runner)
        fixed = [{"item_id": "item-1", "disposition": "resolved", "note": "The request was withdrawn."},
                 {"item_id": "item-2", "disposition": "resolved", "note": "The policy check is fixed."}]
        runner.codex_outputs = [_classified_review([_association(r, "code") for r in rids], hr=True,
                                                   dispositions=fixed)]
        runner.gemini_outputs = [_classified_review([_association(r, "code") for r in rids],
                                                    reviewer="Google Gemini", hr=True, dispositions=fixed)]
    else:
        _queue_fresh_round(runner)
    before = len(runner.commands)

    if variant in {"edited-signed-source", "edited-and-added-signed-source"}:
        # Without the signed answer, Codex's evidence request survives the
        # rollover and freezes round 2 for human evidence.
        with pytest.raises(HumanDecisionRequiredError, match="awaiting human-only exact-head evidence: item-1"):
            run_pr_loop(runner, pr_number=77, config=config)
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    _assert_closed_and_rolled_over(runner, before)


def test_budget_exhausted_posts_the_closure_then_the_round_limit_stop(tmp_path, monkeypatch):
    """`closure-at-max-rounds` (PR): the record is posted, then the ordinary round limit names round 2."""
    runner, config = _conflict_stop(tmp_path, monkeypatch, max_rounds=1)
    _follow_hint(runner)
    _queue_fresh_round(runner)
    before = len(runner.commands)

    with pytest.raises(AgentLoopError) as stop:
        run_pr_loop(runner, pr_number=77, config=config)

    (record,) = _closures(runner)
    assert record.metadata.round_number == 1
    message = str(stop.value)
    assert message.startswith("PR #77")
    assert "after round 1" in message  # the existing blocker detail is kept
    assert "Round 1 was closed incomplete" in message
    assert "require round 2, beyond --max-rounds" in message
    assert "Delete these" not in str(stop.value)
    assert not _supersessions(runner)
    assert _agent_commands(runner, before) == []
    assert not any("Delete these" in c for c in runner.comments)
    # A rerun at the same limit resumes the closed round from its record and
    # stops with the same named round, posting nothing.
    comments_before = len(runner.comments)
    with pytest.raises(AgentLoopError, match="require round 2, beyond --max-rounds"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(runner.comments) == comments_before
    # Resuming with a higher limit proceeds in round 2 without a second record.
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3)) == 0
    assert len(_closures(runner)) == 1
    round_two = [r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.round_number == 2]
    assert sorted(r.metadata.agent for r in round_two) == ["Codex", "Gemini"]


# --- unit: round-level decision ---------------------------------------------


class _Spool:
    def __init__(self, states):
        self.states = states
        self.removed = []

    def publication_state(self, name):
        return self.states.get(name, ("absent", None))

    def remove(self, name):  # pragma: no cover - must never be called
        self.removed.append(name)


def _decision_inputs(tmp_path, monkeypatch):
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    records = {r.metadata.agent: r for r in _pr_records(runner) if r.metadata.role == "reviewer"}
    comments = [SimpleNamespace(body=c["body"], comment_id=_cid(c) or 900 + i)
                for i, c in enumerate(runner.pr_payload["comments"])]
    from coding_review_agent_loop.round_state import _resume_pr_round
    resume = _resume_pr_round([SimpleNamespace(body=c.body) for c in comments], head_sha=HEAD,
                              configured_reviewers=pr_loop.reviewers(config))
    assert resume.round_number == 1 and resume.reconciliation_record is None
    assert {r.metadata.agent for r in resume.completed_reviews} == {"Codex", "Gemini"}
    posted = []

    def call(completed, *, spool=None, seat_binding=None, edit_states=None, **overrides):
        monkeypatch.setattr(sis, "read_pr_comment_edit_states",
                            lambda *a, **k: edit_states if edit_states is not None
                            else {c.comment_id: () for c in comments})
        kwargs = dict(
            config=config, pr_number=77, comments=comments, current_resume=resume, round_number=1,
            allowed_rounds=3, head_sha=HEAD, human_requirements=requirements, approved_plan_context=None,
            acquisition_contract=None, seat_binding=seat_binding, prior_items=(),
            configured_reviewer_names=["Codex", "Gemini"], completed_by_name=dict(completed),
            resumed_by_name={}, rejection_reasons={name: "growth" for name in completed},
            spool=spool or _Spool({}), bound_metadata=lambda **f: pr_loop.PostedRoundMetadata(**f),
            post_comment=lambda *a, **k: posted.append(k["body"]), unresolved_items=(), blocked=False,
        )
        kwargs.update(overrides)
        return sis.supersede_for_signed_growth(runner, **kwargs)

    return records, resume, posted, call


def test_unit_pure_growth_closes_the_round(tmp_path, monkeypatch):
    records, resume, posted, call = _decision_inputs(tmp_path, monkeypatch)
    closed = call(records)
    assert len(posted) == 1 and closed is not resume
    assert closed.completed_reviews == () and closed.reconciled
    assert closed.reconciliation_record.metadata.phase == SIGNED_INPUT_SUPERSESSION_PHASE


def test_unit_mixed_rejection_reasons_keep_the_refusal(tmp_path, monkeypatch):
    records, resume, posted, call = _decision_inputs(tmp_path, monkeypatch)
    seat = {"seat": "current"}
    rebound = {
        "Codex": dataclasses.replace(records["Codex"], metadata=dataclasses.replace(
            records["Codex"].metadata, seat_binding=seat)),
        "Gemini": dataclasses.replace(records["Gemini"], metadata=dataclasses.replace(
            records["Gemini"].metadata, seat_binding={"seat": "old"})),
    }
    assert call(rebound, seat_binding=seat) is resume and posted == []
    # A blocked route (recovery, conflict, evidence or conversion pass) never rolls over.
    assert call(records, blocked=True) is resume and posted == []
    # No published review at all: nothing was rejected, so nothing rolls over.
    assert call({}) is resume and posted == []
    # A one-reviewer board has no same-round peer: today's fresh turn applies.
    assert call({"Codex": records["Codex"]}, configured_reviewer_names=["Codex"]) is resume and posted == []
    # A reviewer not rejected at all (resumed) keeps the ordinary path.
    assert call(records, resumed_by_name={"Codex": records["Codex"]}) is resume and posted == []


@pytest.mark.parametrize("state", ["carrier", "malformed", "legacy"])
def test_unit_partly_public_or_unknown_spool_of_unpublished_reviewer_keeps_stop(tmp_path, monkeypatch, state):
    records, resume, posted, call = _decision_inputs(tmp_path, monkeypatch)
    board = ["Codex", "Gemini", "Claude"]
    spool = _Spool({"Claude": (state, {"bodies": ["sidecar"]} if state == "carrier" else None)})
    # Codex and Gemini are public; Claude's carrier may be partly public
    # without its anchor, so its PublicationResumeStop must stay in charge.
    assert call(records, spool=spool, configured_reviewer_names=board) is resume
    assert posted == [] and spool.removed == []
    # Control: a provably unpublished third reviewer does not block rollover.
    assert call(records, spool=_Spool({"Claude": ("fresh", None)}), configured_reviewer_names=board) is not resume
    assert len(posted) == 1


def test_unit_unproven_publications_and_budget(tmp_path, monkeypatch):
    records, resume, posted, call = _decision_inputs(tmp_path, monkeypatch)
    # Unreadable history (missing comment) and a fresh spool beside a public anchor.
    assert call(records, edit_states={}) is resume
    assert call(records, spool=_Spool({"Codex": ("fresh", None)})) is resume
    assert call(records, spool=_Spool({"Codex": ("malformed", None)})) is resume
    assert posted == []
    with pytest.raises(AgentLoopError, match="after round 1"):
        call(records, allowed_rounds=1)
    assert posted == []


# --- unit: growth classifier and integrity gate ------------------------------


def test_unit_growth_classifier_cases(tmp_path, monkeypatch):
    records, _resume, _posted, _call = _decision_inputs(tmp_path, monkeypatch)
    record = records["Codex"]

    def classify(rec, reqs, **kw):
        args = dict(seat_binding=None, round_number=1, head_sha=HEAD, reconciled=False)
        args.update(kw)
        return response_validation.signed_input_growth_rejection(rec, reqs, **args)

    live = _live_requirements(tmp_path)
    old_ids = tuple(record.metadata.surfaced_reviewer_requirement_ids)
    assert len(old_ids) == 1
    superset = list(live)
    assert classify(record, superset)
    equal = [r for r in live if r.requirement_id in old_ids]
    assert not classify(record, equal)
    removed = [r for r in live if r.requirement_id not in old_ids]
    assert not classify(record, removed)
    legacy = dataclasses.replace(record, metadata=dataclasses.replace(
        record.metadata, surfaced_reviewer_requirement_ids=("1",)))
    assert not classify(legacy, superset)
    assert not classify(record, superset, reconciled=True)
    assert not classify(record, superset, round_number=2)
    assert not classify(record, superset, head_sha="f" * 40)
    assert not classify(record, superset, seat_binding={"seat": "x"})
    assert not response_validation.signed_input_growth_rejection(
        record, superset, None, {"Codex": ("other-model", "high")},
        seat_binding=None, round_number=1, head_sha=HEAD, reconciled=False)


def _live_requirements(tmp_path):
    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy run.", 1))
    runner.pr_payload["comments"].append(
        _signed("The policy run is accepted; keep the policy check code-actionable.", 2))
    return _parse_pr_human_requirements(runner.pr_payload)


def test_unit_integrity_gate_cases():
    import datetime
    body = "frozen body"
    carrier = PublicationCarrier(
        bodies=("sidecar body", body), body_sha256=(
            __import__("hashlib").sha256(b"sidecar body").hexdigest(),
            __import__("hashlib").sha256(body.encode()).hexdigest()),
        prepared_at="2026-11-21T10:00:00+00:00", actor_id=1, actor_login="bot",
        baseline_ids=(), response_sha256="x", validation_context_digest="y").to_dict()
    at = datetime.datetime(2026, 11, 21, 10, 1, tzinfo=datetime.timezone.utc)
    sidecar = SimpleNamespace(comment_id=11, body="sidecar body", created=at)
    anchor = SimpleNamespace(comment_id=12, body=body, created=at)
    edited = SimpleNamespace(comment_id=12, body="edited", created=at)
    assert sis.carrier_proves_unedited(carrier, [sidecar, anchor], (11, 12))
    # Edited anchor: not complete.
    assert not sis.carrier_proves_unedited(carrier, [sidecar, edited], (11, 12))
    # Edited sidecar (prefix-zero shape) and disorder.
    assert not sis.carrier_proves_unedited(carrier, [anchor], (11, 12))
    swapped = [SimpleNamespace(comment_id=12, body="sidecar body", created=at),
               SimpleNamespace(comment_id=11, body=body, created=at)]
    assert not sis.carrier_proves_unedited(carrier, swapped, (11, 12))
    # Matched comments must be exactly the record's own comments.
    assert not sis.carrier_proves_unedited(carrier, [sidecar, anchor], (12,))
    assert not sis.carrier_proves_unedited({"bodies": []}, [sidecar, anchor], (11, 12))
    assert not sis.carrier_proves_unedited(carrier, None, (11, 12))
    # Edit history: only a proven never-edited history passes.
    assert sis.edit_history_proves_unedited({11: (), 12: ()}, (11, 12))
    assert not sis.edit_history_proves_unedited({11: (), 12: None}, (11, 12))
    assert not sis.edit_history_proves_unedited({11: ()}, (11, 12))
    assert not sis.edit_history_proves_unedited({11: (), 12: (("t", "d"),)}, (11, 12))
    assert not sis.edit_history_proves_unedited({}, ())


def test_unit_changed_after_admission_is_unproven():
    from coding_review_agent_loop.pr_round_ingestion import AdmittedPrComment
    from coding_review_agent_loop.round_metadata import PostedRoundRecord
    comment = AdmittedPrComment(author="bot", created_at=None, body="changed", comment_id=5, author_id=1,
                                invocation_actor_id=1, admitted_body="original", admitted_protocol_body="original")
    record = PostedRoundRecord(index=0, metadata=None, body="original")
    assert sis.review_comment_ids([comment], record) is None


def test_unit_supersession_payload_decoding():
    a, b = "hr-" + "a" * 64, "hr-" + "b" * 64
    good = {"round": 2, "head": HEAD, "superseded": [{"agent": "Codex", "comment_id": 9, "route": "carrier"}],
            "previous_requirement_ids": [a], "requirement_ids": [a, b], "item_number_high_water": 3}
    assert decode_supersession_payload(good, round_number=2, subject=HEAD)["requirement_ids"] == [a, b]
    for bad in (
        {**good, "round": 3},
        {**good, "head": "f" * 40},
        {**good, "requirement_ids": [a]},  # not growth
        {**good, "previous_requirement_ids": [b], "requirement_ids": [a]},
        {**good, "superseded": []},
        {**good, "superseded": [{"agent": "Codex", "comment_id": 9, "route": "guess"}]},
        {**good, "item_number_high_water": -1},
        "not a mapping",
    ):
        assert decode_supersession_payload(bad, round_number=2, subject=HEAD) is None


# --- managed-ci-binding-survives ---------------------------------------------


def _managed_round_comment(comment_id, metadata):
    from coding_review_agent_loop.round_metadata import _attach_round_metadata
    return {"id": comment_id, "user": {"login": "agent-loop", "id": 1},
            "body": _attach_round_metadata(f"{metadata.role} round", metadata)}


def test_managed_ci_authorization_survives_rollover_and_correlates_next_round(tmp_path):
    """The authorization is admitted after rollover; continuity pairs the round-2
    block with the round-3 coder, and the superseded round-1 block adds no match."""
    from coding_review_agent_loop import managed_ci_authorization
    from coding_review_agent_loop.managed_ci import _find_resume_audit
    from coding_review_agent_loop.managed_ci_authorization import (
        publish_issue_created_authorization,
        publish_issue_created_continuity_authorization,
    )
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata
    from test_managed_ci import AuthorizationCommentRunner, _authorization_handoff, label_event, metadata

    new_head = "e9bff449de155dd0be5fd76276214538b0dfd315"
    a, b = "hr-" + "a" * 64, "hr-" + "b" * 64
    runner = AuthorizationCommentRunner(issue_events=[label_event()])
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
                         allow_unprotected_managed_ci=True)
    initial = publish_issue_created_authorization(
        runner, config=config, handoff=_authorization_handoff(), metadata=metadata())
    review = lambda round_number: PostedRoundMetadata(
        flow="pr", role="reviewer", agent="Codex", round_number=round_number, subject=HEAD,
        state="blocking", surfaced_reviewer_requirement_ids=(a,) if round_number == 1 else (a, b))
    supersession = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD,
        phase=SIGNED_INPUT_SUPERSESSION_PHASE, surfaced_reviewer_requirement_ids=(a, b),
        signed_input_supersession=decode_supersession_payload({
            "round": 1, "head": HEAD, "superseded": [{"agent": "Codex", "comment_id": 87, "route": "edit-history"}],
            "previous_requirement_ids": [a], "requirement_ids": [a, b], "item_number_high_water": 0,
        }, round_number=1, subject=HEAD))
    assert supersession.signed_input_supersession is not None
    coder = PostedRoundMetadata(flow="pr", role="coder", agent="agent-loop", round_number=3, subject=new_head)
    runner.intent_comments.extend([
        _managed_round_comment(87, review(1)),
        _managed_round_comment(88, supersession),
        _managed_round_comment(89, review(2)),
        _managed_round_comment(90, coder),
    ])
    runner.rest_pr["head"]["sha"] = new_head
    find = lambda round_number: managed_ci_authorization.find_actor_round_metadata_comment_ids(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        predecessor_head=HEAD, new_head=new_head, round_number=round_number, after_comment_id=50)
    assert find(2) == (89, 90)
    # The superseded round-1 block has no round-2 coder and pairs with nothing.
    with pytest.raises(AgentLoopError, match="correlated blocking-review and coder"):
        find(1)
    continued = publish_issue_created_continuity_authorization(
        runner, config=config, handoff=initial, predecessor_head=HEAD, new_head=new_head,
        round_comment_ids=(89, 90))
    audit = _find_resume_audit(
        runner, config=config, pr_number=7, actor_login="agent-loop", actor_id=1,
        base_ref="main", issue_number=643, live_head=new_head)
    # Admitted without a fresh authorization; the original record is not retired.
    assert continued.authorization_kind == "continuity"
    assert audit is not None and audit[0] == continued.authorization_comment_id
    assert any(c.get("id") == initial.authorization_comment_id for c in runner.intent_comments)


class _ContinuityCaptured(Exception):
    pass


def test_managed_ci_rollover_workflow_keeps_authorization_and_correlates_new_records(tmp_path, monkeypatch):
    """Review item-3: a managed issue-created run takes the conflict stop, the
    operator adds a signed source, and the resume rolls over, blocks in round 2 and
    pushes a coder fix.  The original authorization handoff is reused (no fresh
    grant), the real continuity correlation binds only the newly generated
    round-2 block and round-3 coder records, and nothing is deleted."""
    import coding_review_agent_loop.pr_loop_support as pr_loop_support
    from coding_review_agent_loop.managed_ci_types import ManagedCiContract
    from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
    from agent_loop_helpers import structured_coder_followup

    actor_login, actor_id = _ACTOR
    handoff = pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha=HEAD, branch="agent-loop/managed-56",
        trusted_actor_login=actor_login, trusted_actor_id=actor_id,
        protection_mode="voluntary", override_nonce="root",
        authorization_kind="creation", authorization_comment_id=17,
    )
    fresh, correlated, published = [], [], []
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(pr_loop, "activate_managed_ci",
                        lambda *_a, **_k: ManagedCiContract(protocol_version=2, issue_created_pr=True))
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "authorize_fresh_issue_created_resume",
                        lambda *a, **k: fresh.append(k) or (_ for _ in ()).throw(AssertionError("fresh grant")))
    real_find = pr_loop.find_actor_round_metadata_comment_ids

    def find(*args, **kwargs):
        result = real_find(*args, **kwargs)
        correlated.append((kwargs, result))
        return result

    def publish(*_args, **kwargs):
        published.append(kwargs)
        raise _ContinuityCaptured

    monkeypatch.setattr(pr_loop, "find_actor_round_metadata_comment_ids", find)
    monkeypatch.setattr(pr_loop, "publish_issue_created_continuity_authorization", publish)

    runner = FakeRunner(pr_payload={"headRefName": "agent-loop/managed-56", "headRefOid": HEAD,
                                    "baseRefName": "main", "body": "Fixes #56"})
    runner.authenticated_actor = _ACTOR
    # An existing qualification record for the head: rollover must keep it.
    checkpoint = _attach_round_metadata("Qualification was dispatched.", PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD, prior_items=(),
        state="blocking", phase="qualification-checkpoint", architecture_contract_version=1))
    runner.pr_payload["comments"].append({"author": {"login": actor_login}, "createdAt": "2026-05-23T00:00:00Z",
                                          "body": checkpoint})
    runner.pr_payload["comments"].append(_signed("A human must accept the policy run.", 1))
    (rid,) = _rids(runner)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, managed_ci=True,
                         managed_ci_trusted_actor=actor_login, allow_unprotected_managed_ci=True, max_rounds=4)
    run = lambda: run_pr_loop(runner, pr_number=77, config=config, managed_ci_handoff=handoff,
                              managed_ci_issue_number=56)
    runner.codex_outputs = [_classified_review(
        [_association(rid, "human-only", request=_EVIDENCE_TEXT)], hr=True, evidence=[_EVIDENCE_TEXT])] * 2
    runner.gemini_outputs = [_classified_review(
        [_association(rid, "code")], reviewer="Google Gemini", hr=True,
        state="blocking", blocking=["Fix the policy check."])] * 2
    with pytest.raises(HumanDecisionRequiredError, match="remained conflicting"):
        run()
    _follow_hint(runner)
    old, new = _rids(runner)
    answer = {"item_id": "item-1", "disposition": "resolved", "note": "Accepted.", "evidence_requirement_ids": [new]}
    # Gemini's round-1 blocker carries forward as item-2 through the closure (#2064).
    fixed = {"item_id": "item-2", "disposition": "resolved", "note": "The policy check is fixed."}
    runner.codex_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], hr=True, dispositions=[answer, fixed],
        state="blocking", blocking=["Bound the policy retries."])]
    runner.gemini_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], reviewer="Google Gemini", hr=True,
        dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Requester-owned."}, fixed])]
    runner.claude_outputs = [structured_coder_followup(
        addressed_items=["item-1"], human_requirement_ids=[old, new],
        human_requirement_dispositions=[{"requirement_id": r, "disposition": "addressed", "evidence": "Done."}
                                        for r in (old, new)])]
    kept = [c["body"] for c in runner.pr_payload["comments"]]

    with pytest.raises(_ContinuityCaptured):
        run()

    assert fresh == []
    (record,) = _closures(runner)
    assert not _supersessions(runner)
    ((kwargs, ids),) = correlated
    assert kwargs["round_number"] == 2 and kwargs["predecessor_head"] == HEAD
    assert kwargs["after_comment_id"] == 17  # the original authorization, not a fresh one
    by_id = {_cid(c): c for c in runner.pr_payload["comments"]}
    records = {r.index: r for r in _pr_records(runner)}
    index_of = {_cid(c): i for i, c in enumerate(runner.pr_payload["comments"])}
    bound = [records[index_of[i]].metadata for i in ids]
    assert [(m.role, m.agent, m.round_number, m.state) for m in bound] == [
        ("reviewer", "Codex", 2, "blocking"), ("coder", "Claude", 3, None)]
    assert all(index_of[i] > record.index for i in ids)
    assert published and published[0]["round_comment_ids"] == ids
    assert published[0]["handoff"] is handoff
    # The checkpoint, round-1 reviews and every other earlier comment are retained.
    assert [c["body"] for c in runner.pr_payload["comments"][:len(kept)]] == kept
    assert by_id and _no_deletes(runner)


def _single_reviewer_publication(tmp_path, monkeypatch, *, carrier):
    """Run A on a one-reviewer board: Codex publishes, then its classification
    continuation fails, leaving the round unreconciled."""
    from pr_loop_test_helpers import _evidence_review
    runner = FakeRunner()
    if carrier:
        runner.authenticated_actor = _ACTOR
        runner.serve_rest_issue_comments = True
    runner.pr_payload["comments"].append(_signed("A human must accept the policy run.", 1))
    (rid,) = _rids(runner)
    config = make_config(tmp_path, reviewer=("codex",), review_parallel=True)
    bad = _classified_review([_association(rid, "mixed", leftover="Fix retries.")])
    runner.codex_outputs = [_evidence_review(), bad, bad]
    with monkeypatch.context() as patch:
        if carrier:
            patch.setattr(ReviewRoundSpool, "remove", lambda self, name: None)
            patch.setattr(ReviewRoundSpool, "discard", lambda self: None)
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=config)
    assert [r.metadata.agent for r in _pr_records(runner) if r.metadata.role == "reviewer"] == ["Codex"]
    return runner, config


@pytest.mark.parametrize("carrier", [True, False], ids=["carrier", "no-carrier"])
def test_single_reviewer_board_whose_own_review_is_rejected_closes_the_round(tmp_path, monkeypatch, carrier):
    """`single-reviewer-own-rejected` (#2064): with one configured reviewer whose own
    published review resume rejects, the round closes and the reviewer runs fresh in
    round 2, never in round 1; retained content-valid spool data never replays it."""
    runner, config = _single_reviewer_publication(tmp_path, monkeypatch, carrier=carrier)
    spool = _round_spool(tmp_path, config)
    if carrier:
        assert spool.publication_state("Codex")[0] == "carrier"
    _follow_hint(runner)
    rids = _rids(runner)
    runner.codex_outputs = [_classified_review([_association(r, "code") for r in rids], hr=True)]
    before = len(_agent_commands(runner))
    kept = [c["body"] for c in runner.pr_payload["comments"]]

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (record,) = _closures(runner)
    payload = record.metadata.round_closure
    assert [e["agent"] for e in payload["published"]] == ["Codex"]
    assert [e["agent"] for e in payload["unreplayable"]] == ["Codex"]
    assert "rejected by resume" in payload["unreplayable"][0]["cause"]
    assert not _supersessions(runner)
    reviews = [r for r in _pr_records(runner) if r.metadata.role == "reviewer"]
    assert [r.metadata.round_number for r in reviews if r.index > record.index] == [2]
    assert [r.metadata.round_number for r in reviews if r.index < record.index] == [1]
    assert _agent_commands(runner)[before:] == ["codex"]
    assert [c["body"] for c in runner.pr_payload["comments"][:len(kept)]] == kept
    assert spool.publication_state("Codex")[0] == "absent"
    assert _no_deletes(runner)
    assert not any("Delete these" in c or "partial-round recovery" in c for c in runner.comments)
