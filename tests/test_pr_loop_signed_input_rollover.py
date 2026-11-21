"""Signed-input growth rolls a published PR round over instead of refusing (#2054).

The sequence from llm-dialectic PR #2310: every review of a round is published,
the conflicting-classification stop tells the operator to post a follow-up
signed comment, and the resume after that comment must open a fresh round
rather than ask for the round's reviews to be deleted.  Edits, withdrawals and
every other rejection keep the partial-round refusal.
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
from coding_review_agent_loop.review_rounds import PartialReviewRoundError
from coding_review_agent_loop.review_spool import ReviewRoundSpool
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


def _queue_fresh_round(runner):
    """Round 2: Codex accepts its carried evidence request (item-1) by citing the
    added signed answer; both reviewers classify against the grown set."""
    old, new = _rids(runner)[0], _rids(runner)[-1]
    answer = {"item_id": "item-1", "disposition": "resolved", "note": "The signed answer accepts the run.",
              "evidence_requirement_ids": [new]}
    runner.codex_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], hr=True, dispositions=[answer])]
    runner.gemini_outputs = [_classified_review(
        [_association(old, "code"), _association(new, "code")], reviewer="Google Gemini", hr=True,
        dispositions=[{"item_id": "item-1", "disposition": "resolved", "note": "Requester-owned."}])]


def _agent_commands(runner, start=0):
    return [cmd[0] for cmd, _ in runner.commands[start:] if cmd[:1] in (["codex"], ["gemini"], ["claude"])]


def _no_deletes(runner):
    return not any(cmd[:2] == ["gh", "api"] and "DELETE" in cmd for cmd, _ in runner.commands)


def _supersessions(runner):
    return [r for r in _pr_records(runner) if r.metadata.phase == SIGNED_INPUT_SUPERSESSION_PHASE]


def _spy_supersession_post(monkeypatch, runner):
    """Record how many reviewer agents had run when the supersession record was posted."""
    seen = []
    real = pr_loop.post_pr_comment

    def spy(*args, **kwargs):
        if SIGNED_INPUT_SUPERSESSION_PHASE in str(kwargs.get("body", "")) or (
            "superseded by new signed input" in str(kwargs.get("body", ""))
        ):
            seen.append(len(_agent_commands(runner)))
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "post_pr_comment", spy)
    return seen


class _Interrupted(Exception):
    pass


def _interrupt_after_supersession(runner, config, monkeypatch):
    """Post the supersession record, then stop before any round-2 work."""
    real = pr_loop.post_pr_comment

    def post(*args, **kwargs):
        real(*args, **kwargs)
        if "superseded by new signed input" in str(kwargs.get("body", "")):
            raise _Interrupted()

    with monkeypatch.context() as patch:
        patch.setattr(pr_loop, "post_pr_comment", post)
        with pytest.raises(_Interrupted):
            run_pr_loop(runner, pr_number=77, config=config)
    (record,) = _supersessions(runner)
    assert _pr_records(runner)[-1].index == record.index
    return record


def _cid(comment):
    return comment.get("id", comment.get("databaseId"))


def _round_spool(tmp_path, config, round_number=1):
    return pr_loop._review_round_spool(
        config, surface="pr", number=77, round_number=round_number, subject=HEAD)


def _assert_refused_without_rollover(runner, config, before_commands, match="Delete these",
                                     error=PartialReviewRoundError):
    with pytest.raises(error, match=match) as refusal:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not _supersessions(runner)
    assert _agent_commands(runner, before_commands) == []
    return str(refusal.value)


# --- growth-after-conflict-stop ---------------------------------------------


@pytest.mark.parametrize("carrier", [False, True], ids=["edit-history", "carrier"])
def test_growth_after_conflict_stop_opens_round_two_without_deletion(tmp_path, monkeypatch, carrier):
    runner, config = _conflict_stop(tmp_path, monkeypatch, carrier=carrier)
    round_one = [c["body"] for c in runner.pr_payload["comments"]]
    _follow_hint(runner)
    _queue_fresh_round(runner)
    before = len(_agent_commands(runner))
    seen = _spy_supersession_post(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (record,) = _supersessions(runner)
    payload = record.metadata.signed_input_supersession
    assert record.metadata.round_number == 1 and record.metadata.subject == HEAD
    # The carried ledger is round 1's, plus Codex's durable evidence request
    # that resume restores for a review invalidated by new input (#1922).
    assert [(i.item_id, i.reviewer, i.obligation_kind) for i in record.metadata.prior_items] == [
        ("item-1", "Codex", "human-exact-head-evidence")]
    assert {e["agent"] for e in payload["superseded"]} == {"Codex", "Gemini"}
    assert {e["route"] for e in payload["superseded"]} == {"carrier" if carrier else "edit-history"}
    assert set(payload["previous_requirement_ids"]) < set(payload["requirement_ids"]) == set(_rids(runner))
    # Save before emit: posted before any round-2 reviewer launch.
    assert seen == [before]
    later = [r for r in _pr_records(runner) if r.index > record.index]
    assert later and all(r.metadata.round_number >= 2 for r in later)
    round_two = [r for r in later if r.metadata.role == "reviewer"]
    assert sorted(r.metadata.agent for r in round_two) == ["Codex", "Gemini"]
    assert all(set(r.metadata.surfaced_reviewer_requirement_ids) == set(_rids(runner)) for r in round_two)
    assert sorted(_agent_commands(runner, 0)[before:]) == ["codex", "gemini"]
    # Round-1 comments are untouched and nothing was deleted or asked for.
    assert [c["body"] for c in runner.pr_payload["comments"][:len(round_one)]] == round_one
    assert _no_deletes(runner)
    assert not any("Delete these" in c for c in runner.comments)
    # Superseded reviews add no dispositions, votes or new findings: round 2
    # starts from the carried ledger and settles it from its own reviews.
    reconciliation = next(r for r in later if r.metadata.phase == "reconciliation")
    assert reconciliation.metadata.round_number == 2
    assert reconciliation.metadata.prior_items == record.metadata.prior_items
    assert not any("Fix the policy check." in i.text for i in reconciliation.metadata.prior_items)
    if carrier:
        # The round-1 carriers stay as integrity evidence.
        spool = _round_spool(tmp_path, config)
        assert spool.publication_state("Codex")[0] == "carrier"
        assert spool.publication_state("Gemini")[0] == "carrier"
    else:
        assert getattr(runner, "edit_history_calls", 0) >= 1


def test_superseded_reviews_are_inert_for_approval_panel_and_resume(tmp_path, monkeypatch):
    from coding_review_agent_loop.round_state import (
        _resume_pr_round, drop_inert_post_reconciliation_checkpoints, superseded_review_indexes,
    )
    from pr_loop_test_helpers import _pr_records as records
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    _interrupt_after_supersession(runner, config, monkeypatch)
    history = records(runner)
    (record,) = _supersessions(runner)
    superseded = superseded_review_indexes(history)
    assert superseded == {r.index for r in history if r.metadata.role == "reviewer"}
    assert not superseded & {r.index for r in drop_inert_post_reconciliation_checkpoints(history)}
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config))
    assert resumed.round_number == 1 and resumed.reconciled
    assert resumed.completed_reviews == ()
    assert resumed.reconciliation_record.metadata.phase == SIGNED_INPUT_SUPERSESSION_PHASE
    # Approval carry never admits a review that misses the grown signed set.
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    approved = [r for r in history if r.metadata.role == "reviewer" and r.metadata.state == "approved"]
    assert approved and not any(
        response_validation._resumed_pr_reviewer_matches_requirements(r, requirements) for r in approved)


# --- interrupt-after-boundary -----------------------------------------------


def test_interrupt_after_supersession_record_resumes_into_round_two(tmp_path, monkeypatch):
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    _follow_hint(runner)
    # Interrupted after the record, before any round-2 record or launch.
    _interrupt_after_supersession(runner, config, monkeypatch)
    _queue_fresh_round(runner)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_supersessions(runner)) == 1
    round_two = [r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.round_number == 2]
    assert sorted(r.metadata.agent for r in round_two) == ["Codex", "Gemini"]
    assert _no_deletes(runner)
    assert not any("Delete these" in c for c in runner.comments)


# --- tamper and non-growth refusals -----------------------------------------


def _edit_review_body(runner, agent):
    record = next(r for r in _pr_records(runner) if r.metadata.role == "reviewer" and r.metadata.agent == agent)
    comment = runner.pr_payload["comments"][record.index]
    assert "Reviewed." in comment["body"]
    # The visible body changes; the round metadata (IDs, head, seat,
    # acquisition) is kept exactly.
    comment["body"] = comment["body"].replace("Reviewed.", "Reviewed, then quietly edited.", 1)
    return record.index


def test_edited_review_with_retained_carrier_is_refused(tmp_path, monkeypatch):
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

    # Today's refusal for a carrier that no longer matches: the spool preflight
    # stops before any launch and points at the partial-round recovery list.
    from coding_review_agent_loop.publication_resume import PublicationResumeStop
    message = _assert_refused_without_rollover(
        runner, config, before, match="partial-round recovery", error=PublicationResumeStop)

    assert "Codex" in message and "spool record was kept" in message
    assert spool.publication_state("Codex") == (state, raw)
    assert not any("superseded by new signed input" in c for c in runner.comments)


@pytest.mark.parametrize("history", ["edited", "missing"])
def test_edited_or_unproven_review_without_carrier_is_refused(tmp_path, monkeypatch, history):
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    record = next(r for r in _pr_records(runner) if r.metadata.agent == "Gemini")
    comment = runner.pr_payload["comments"][record.index]
    comment["_edit_history"] = None if history == "missing" else {
        "lastEditedAt": "2026-11-21T18:00:00Z", "includesCreatedEdit": True,
        "userContentEdits": {"totalCount": 1, "nodes": [
            {"editedAt": "2026-11-21T18:00:00Z", "deletedAt": None, "diff": "edited"}]}}
    _follow_hint(runner)
    _queue_fresh_round(runner)

    _assert_refused_without_rollover(runner, config, len(runner.commands))

    assert getattr(runner, "edit_history_calls", 0) >= 1


def test_metadata_no_longer_pure_growth_match_is_refused(tmp_path, monkeypatch):
    """An altered acquisition identity is not pure growth, even with an added source."""
    runner, config = _conflict_stop(tmp_path, monkeypatch, policy="selective-intermediate")

    def alter(payload):
        if payload.get("role") == "reviewer" and payload.get("agent") == "Codex":
            return {**payload, "configured_effort": "a-different-effort"}
        return None

    _rewrite_pr_metadata(runner, alter)
    _follow_hint(runner)
    _queue_fresh_round(runner)

    message = _assert_refused_without_rollover(runner, config, len(runner.commands))

    assert "acquisition contract no longer matches" in message


@pytest.mark.parametrize("added", [False, True], ids=["edited", "edited-and-added"])
def test_edited_or_withdrawn_signed_source_is_refused(tmp_path, monkeypatch, added):
    runner, config = _conflict_stop(tmp_path, monkeypatch)
    signed = next(c for c in runner.pr_payload["comments"] if c["body"].endswith("-- Human Reviewer"))
    signed["body"] = "A human must accept the policy run, edited.\n\n-- Human Reviewer"
    if added:
        _follow_hint(runner)
    _queue_fresh_round(runner)

    message = _assert_refused_without_rollover(runner, config, len(runner.commands))

    assert "signed requirement set changed since publication" in message
    assert "withdrawn hr-" in message


def test_budget_exhausted_takes_round_limit_stop_without_record(tmp_path, monkeypatch):
    runner, config = _conflict_stop(tmp_path, monkeypatch, max_rounds=1)
    _follow_hint(runner)
    _queue_fresh_round(runner)
    before = len(runner.commands)

    with pytest.raises(AgentLoopError) as stop:
        run_pr_loop(runner, pr_number=77, config=config)

    assert not isinstance(stop.value, PartialReviewRoundError)
    # The ordinary round-limit diagnostic for the exhausted budget.
    assert str(stop.value).startswith("PR #77") and "after round 1" in str(stop.value)
    assert "Delete these" not in str(stop.value)
    assert not _supersessions(runner)
    assert _agent_commands(runner, before) == []
    assert not any("Delete these" in c for c in runner.comments)


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
    # A lone rejected review has no public peer: today's fresh turn applies.
    assert call({"Codex": records["Codex"]}) is resume and posted == []
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
