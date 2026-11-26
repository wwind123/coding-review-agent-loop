"""One rollover rule for partially published PR rounds (#2064 stage 2, #2080).

When a round holds any published review and any reviewer whose outcome cannot
be replayed, the round-wide decision posts one closed-incomplete record that
carries the round's ledger and the loop rolls over to round N+1 with the full
board.  Published reviews stay on GitHub as history, their findings enter the
ledger with fresh IDs, and no comment is deleted or named for deletion.
"""

import dataclasses
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.issue_loop as issue_loop
from coding_review_agent_loop import response_validation
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.round_closure import (
    ROUND_CLOSED_INCOMPLETE_PHASE,
    restore_closure_evidence_obligations,
)
from coding_review_agent_loop.round_closure_writer import (
    RoundClosureDecision,
    close_incomplete_round,
    mint_pr_review_items,
    pr_published_review,
    pr_records_blocking_summary,
)
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    PostedRoundRecord,
    ResumedReviewRound,
    _attach_round_metadata,
)
from coding_review_agent_loop.round_state import _resume_pr_round

from agent_loop_helpers import FakeRunner, make_config, patch_attr_in, patch_object_in, structured_pr_review
from test_review_parallel import (
    _PartialPublicationProbeRunner,
    _comment_records,
    _published_count,
    _spool_files,
)

HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"
NAMES = {"codex": "Codex", "gemini": "Gemini", "claude": "Claude"}


def _reviewer_runs(runner):
    """Every reviewer CLI invocation, including Claude as a reviewer."""
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["codex"], ["gemini"], ["claude"])]


def _with_evidence(review, requests):
    """Add exact-head evidence requests to a structured PR review fixture."""
    text, marker = review.split("\n", 1)
    payload = json.loads(text)
    payload["exact_head_evidence_requests"] = list(requests)
    return json.dumps(payload) + "\n" + marker


def _closures(runner):
    return [r for r in _comment_records(runner, surface="pr", flow="pr") if r.phase == ROUND_CLOSED_INCOMPLETE_PHASE]


def _no_deletion_advice(runner):
    assert not any("Delete these" in body or "partial-round recovery" in body for body in runner.comments)
    assert not any(cmd[:2] == ["gh", "api"] and "DELETE" in cmd for cmd, _cwd in runner.commands)


def _interrupt_on_publication(real_post, markers, *, at):
    """Interrupt the ``at``-th same-round reviewer publication."""
    state = {"posted": 0}

    def post(*args, **kwargs):
        body = str(kwargs["body"])
        if any(marker in body for marker in markers):
            state["posted"] += 1
            if state["posted"] == at:
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    return post


def _interrupt_pr(runner, config, markers, *, at):
    with patch_object_in((pr_loop_support, pr_loop, issue_loop), "post_pr_comment",
                         side_effect=_interrupt_on_publication(pr_loop_support.post_pr_comment, markers, at=at)):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)


def _spy_closure(monkeypatch, runner):
    """Record the reviewer launches seen when the closure record is posted."""
    seen = []
    real = pr_loop.post_pr_comment

    def spy(*args, **kwargs):
        if "closed, incomplete" in str(kwargs.get("body", "")):
            seen.append(_reviewer_runs(runner))
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "post_pr_comment", spy)
    return seen


# --- pr-partial-rollover --------------------------------------------------------


def _three_reviewer_runner():
    markers = ("Codex found a blocker.", "Gemini approves independently.", "Claude reviews last.")
    resolved = [{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        slow_reviewer="claude",
        codex_outputs=[
            structured_pr_review(state="blocking", summary=markers[0], blocking_items=["Bound the retry loop."]),
            structured_pr_review(summary="Codex approves round two.", prior_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_pr_review(summary=markers[1], reviewer="Google Gemini"),
            structured_pr_review(summary="Gemini approves round two.", reviewer="Google Gemini",
                                 prior_item_dispositions=resolved),
        ],
        claude_outputs=[
            structured_pr_review(summary=markers[2], reviewer="Anthropic Claude"),
            structured_pr_review(summary="Claude approves round two.", reviewer="Anthropic Claude",
                                 prior_item_dispositions=resolved),
        ],
    )
    return runner, markers


@pytest.mark.parametrize("review_parallel", [True, False], ids=["parallel", "sequential"])
@pytest.mark.parametrize("spool_state", ["missing", "invalid"])
def test_two_published_reviews_and_one_unreplayable_reviewer_roll_over(
    tmp_path, monkeypatch, review_parallel, spool_state
):
    """`pr-partial-rollover`: one closed-incomplete record, round N+1 runs the full board."""
    runner, markers = _three_reviewer_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini", "claude"), coder="codex", review_parallel=True)
    _interrupt_pr(runner, config, markers, at=3)
    published = sorted(NAMES[cmd] for cmd, marker in zip(("codex", "gemini", "claude"), markers)
                       if _published_count(runner, marker))
    assert len(published) == 2
    (unpublished,) = set(NAMES.values()) - set(published)
    for path in _spool_files(config):
        if spool_state == "missing":
            path.unlink()
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["response"]["text"] = "not a structured PR review"
            path.write_text(json.dumps(payload), encoding="utf-8")
    launches_before = _reviewer_runs(runner)
    kept = list(runner.pr_payload["comments"])
    seen = _spy_closure(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, review_parallel=review_parallel)) == 0

    (closure,) = _closures(runner)
    payload = closure.round_closure
    assert closure.round_number == 1 and closure.subject == HEAD
    assert [e["agent"] for e in payload["published"]] == published
    assert all(isinstance(e["comment_id"], int) for e in payload["published"])
    assert [e["agent"] for e in payload["unreplayable"]] == [unpublished]
    assert payload["unreplayable"][0]["cause"] == (
        "no spooled outcome for this round exists on this host" if spool_state == "missing"
        else "its spooled outcome no longer validates against this run")
    # Decided before any launch: no reviewer ran in round 1 beside public peers.
    assert seen == [launches_before]
    # Codex's published blocker is a prior item of round 2 under a fresh ID.
    if "Codex" in published:
        assert [(i.item_id, i.text) for i in closure.new_items] == [("item-1", "Bound the retry loop.")]
    records = _comment_records(runner, surface="pr", flow="pr")
    assert sorted(r.agent for r in records if r.role == "reviewer" and r.round_number == 2) == sorted(NAMES.values())
    assert sorted(_reviewer_runs(runner)[len(launches_before):]) == ["claude", "codex", "gemini"]
    assert runner.pr_payload["comments"][:len(kept)] == kept
    _no_deletion_advice(runner)


# --- own-publication-rollover ----------------------------------------------------


def _two_reviewer_runner():
    markers = ("Codex approves independently.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        slow_reviewer="gemini",
        codex_outputs=[structured_pr_review(summary=markers[0]),
                       structured_pr_review(summary="Codex approves round two.")],
        gemini_outputs=[structured_pr_review(summary=markers[1], reviewer="Google Gemini"),
                        structured_pr_review(summary="Gemini approves round two.", reviewer="Google Gemini")],
    )
    return runner, markers


@pytest.mark.parametrize("review_parallel", [True, False], ids=["parallel", "sequential"])
def test_own_rejected_publication_closes_the_round_and_discards_the_replayable_peer(
    tmp_path, monkeypatch, review_parallel
):
    """`own-publication-rollover`: only Codex's own review is public and resume rejects it."""
    runner, markers = _two_reviewer_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr(runner, config, markers, at=2)
    assert [_published_count(runner, m) for m in markers] == [1, 0]
    assert len(_spool_files(config)) == 1  # Gemini's withheld outcome is still replayable
    real_matches = response_validation._resumed_pr_reviewer_matches_requirements
    closed = {"done": False}

    def reject_round_one(record, *args, **kwargs):
        if record.metadata.round_number == 1 and not closed["done"]:
            return False
        return real_matches(record, *args, **kwargs)

    patch_attr_in(monkeypatch, (response_validation, pr_loop), "_resumed_pr_reviewer_matches_requirements",
                  reject_round_one)
    launches_before = _reviewer_runs(runner)
    seen = _spy_closure(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, review_parallel=review_parallel)) == 0

    (closure,) = _closures(runner)
    assert [e["agent"] for e in closure.round_closure["published"]] == ["Codex"]
    assert [e["agent"] for e in closure.round_closure["unreplayable"]] == ["Codex"]
    assert "rejected by resume" in closure.round_closure["unreplayable"][0]["cause"]
    assert seen == [launches_before]  # Codex was not re-run in round 1
    # Gemini's spooled round-1 outcome was discarded with the round spool, never replayed.
    assert _published_count(runner, markers[1]) == 0
    assert sorted(_reviewer_runs(runner)[len(launches_before):]) == ["codex", "gemini"]
    records = _comment_records(runner, surface="pr", flow="pr")
    assert sorted(r.agent for r in records if r.role == "reviewer" and r.round_number == 2) == ["Codex", "Gemini"]
    _no_deletion_advice(runner)



# --- migration-stopped-partial ----------------------------------------------------


def test_pr_stopped_by_an_old_release_on_a_partial_round_rolls_over_on_first_resume(tmp_path):
    """`migration-stopped-partial`: published peers, missing spool, an incomplete-status notice."""
    runner, markers = _two_reviewer_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr(runner, config, markers, at=2)
    for path in _spool_files(config):
        path.unlink()
    # The old release's stop left a plain incomplete-status notice behind.
    pr_loop.post_pr_comment(runner, config=config, pr_number=77, body=(
        "Review round 1 on PR #77 is partially published (Codex already posted) but Gemini's "
        "withheld same-round outcome is missing. Delete these 1 comments so the whole round runs "
        "again independently, then rerun."))
    notices = len(runner.comments)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (closure,) = _closures(runner)
    assert [e["agent"] for e in closure.round_closure["published"]] == ["Codex"]
    assert [e["agent"] for e in closure.round_closure["unreplayable"]] == ["Gemini"]
    records = _comment_records(runner, surface="pr", flow="pr")
    assert sorted(r.agent for r in records if r.role == "reviewer" and r.round_number == 2) == ["Codex", "Gemini"]
    # No new comment repeats the old deletion advice; nothing was deleted.
    assert not any("Delete these" in body for body in runner.comments[notices:])
    assert not any(cmd[:2] == ["gh", "api"] and "DELETE" in cmd for cmd, _cwd in runner.commands)


# --- ledger-carry-ids (loop, then restart from the closure record alone) -----------


def test_published_findings_and_evidence_requests_survive_rollover_and_restart(tmp_path, monkeypatch):
    """`ledger-carry-ids`: publication-phase records with empty new_items are re-parsed and minted."""
    evidence = "Run the live retry test at this exact head and attach the output"
    markers = ("Codex found a blocker with evidence.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        slow_reviewer="gemini",
        codex_outputs=[_with_evidence(structured_pr_review(
            state="blocking", summary=markers[0], blocking_items=["Bound the retry loop."],
        ), [evidence])],
        gemini_outputs=[structured_pr_review(summary=markers[1], reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr(runner, config, markers, at=2)
    codex = next(r for r in _comment_records(runner, surface="pr", flow="pr") if r.agent == "Codex")
    assert codex.phase == "publication" and codex.new_items == ()  # provisional checkpoint
    for path in _spool_files(config):
        path.unlink()

    class _Stop(Exception):
        pass

    real = pr_loop.post_pr_comment

    def stop_after_closure(*args, **kwargs):
        result = real(*args, **kwargs)
        if "closed, incomplete" in str(kwargs.get("body", "")):
            raise _Stop
        return result

    monkeypatch.setattr(pr_loop, "post_pr_comment", stop_after_closure)
    with pytest.raises(_Stop):
        run_pr_loop(runner, pr_number=77, config=config)

    (closure,) = _closures(runner)
    assert [(i.item_id, i.text, i.status) for i in closure.new_items] == [("item-1", "Bound the retry loop.", "blocking")]
    obligations = closure.round_closure["evidence_obligations"]
    assert obligations == [{"head": HEAD, "item_number": 2, "reviewer": "Codex", "source_round": 1, "text": evidence}]
    assert closure.round_closure["item_number_high_water"] == 2
    # A restart reads only the closure record: the ledger and the obligation come back.
    comments = [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]]
    resumed = _resume_pr_round(comments, head_sha=HEAD, configured_reviewers=pr_loop.reviewers(config))
    assert resumed.round_number == 1 and resumed.reconciled and resumed.completed_reviews == ()
    record = resumed.reconciliation_record.metadata
    assert [i.item_id for i in record.new_items] == ["item-1"]
    items, next_number = restore_closure_evidence_obligations(list(record.new_items), record, next_item_number=2)
    assert next_number == 3
    restored = [i for i in items if getattr(i, "is_machine_obligation", False)]
    assert [(i.item_id, i.reviewer) for i in restored] == [("item-2", "Codex")]


# --- ledger-carry-ids (writer: authoritative IDs, provisional checkpoints, normalization)


def _record(index, agent, *, phase="publication", state="blocking", response="", new_items=(), evidence=()):
    metadata = PostedRoundMetadata(
        flow="pr", role="reviewer", agent=agent, round_number=3, subject=HEAD, state=state,
        phase=phase, new_items=tuple(new_items), evidence_requests=tuple(evidence),
        canonical_reviewer_response=response or None,
    )
    return PostedRoundRecord(index=index, metadata=metadata, body=response or f"{agent} review")


def _mint(record, next_number, *, prior=()):
    parsed, _output = pr_published_review(record, reviewer=record.metadata.agent)
    return mint_pr_review_items(
        parsed, reviewer=record.metadata.agent, round_number=record.metadata.round_number,
        next_item_number=next_number, approved_followups="record",
        has_blocking_summary=pr_records_blocking_summary(parsed, had_prior_items=bool(prior)))


def test_closure_ledger_keeps_authoritative_ids_mints_provisional_ones_and_follows_published_state():
    """Mixed case: an already-minted record keeps its IDs; a provisional one is minted above
    the high-water mark; a raw blocking response published as approved mints no blocking item."""
    from coding_review_agent_loop.unresolved_items import _next_unresolved_item

    prior = (_next_unresolved_item(item_number=4, reviewer="Codex", source_round=2,
                                   text="Earlier open finding.", status="blocking"),)
    authoritative = _next_unresolved_item(item_number=7, reviewer="Codex", source_round=3,
                                          text="Already minted.", status="blocking")
    records = {
        "Codex": _record(10, "Codex", phase="authoritative", new_items=(authoritative,)),
        "Gemini": _record(11, "Gemini", response=structured_pr_review(
            state="blocking", summary="Gemini blocks.", blocking_items=["Fix the cache race."],
            same_pr_followups=["Add a regression test."], reviewer="Google Gemini"),
            evidence=("Attach the live cache run.",)),
        # Raw response blocks, but publication normalized it to approved (#2064 item-5).
        "Claude": _record(12, "Claude", state="approved", response=structured_pr_review(
            state="blocking", summary="Only a pending CI note.", blocking_items=["Wait for CI."],
            future_followups=["Consider a dashboard."], reviewer="Anthropic Claude")),
    }
    posted = []
    resume = ResumedReviewRound(round_number=3, prior_items=prior, coder_output=None, completed_reviews=(),
                                next_unresolved_item_number=8)
    decision = RoundClosureDecision(round_number=3, published=(("Codex", 1010), ("Gemini", 1011), ("Claude", 1012)),
                                    unreplayable=(("Antigravity", "no spooled outcome"),))
    closed = close_incomplete_round(
        None, config=make_config_stub(), decision=decision, flow="pr", subject=HEAD, number=77,
        comments_length=13, current_resume=resume, prior_items=prior, published_records=records,
        next_item_number=8, mint_record=lambda record, n: _mint(record, n, prior=prior),
        bound_metadata=lambda **fields: PostedRoundMetadata(**fields), post_comment=posted.append,
    )
    metadata = closed.reconciliation_record.metadata
    items = [(i.item_id, i.reviewer, i.text, i.status) for i in metadata.new_items]
    assert items == [
        ("item-7", "Codex", "Already minted.", "blocking"),
        ("item-8", "Gemini", "Fix the cache race.", "blocking"),
        ("item-9", "Gemini", "Add a regression test.", "same-pr"),
        ("item-10", "Claude", "Consider a dashboard.", "future"),
    ]
    assert not any(text == "Wait for CI." for _id, _r, text, _s in items)
    assert metadata.prior_items == prior  # prior items stay open
    assert metadata.round_closure["evidence_obligations"] == [{
        "head": HEAD, "item_number": 11, "reviewer": "Gemini", "source_round": 3,
        "text": "Attach the live cache run."}]
    assert metadata.round_closure["item_number_high_water"] == 11
    assert closed.next_unresolved_item_number == 12 and closed.reconciled and closed.completed_reviews == ()
    (body,) = posted
    assert "No comment needs to be deleted." in body and "Delete these" not in body


def make_config_stub():
    return SimpleNamespace(log_file=None, verbose=False, quiet=True)


# --- single-reviewer-own-rejected (parallel and sequential) ------------------------


@pytest.mark.parametrize("review_parallel", [True, False], ids=["parallel", "sequential"])
def test_single_reviewer_whose_own_publication_is_rejected_runs_fresh_in_the_next_round(
    tmp_path, monkeypatch, review_parallel
):
    """`single-reviewer-own-rejected`: retained content-valid spool data never replays it."""
    from coding_review_agent_loop.review_spool import ReviewRoundSpool

    marker = "Codex approves alone."
    runner = _PartialPublicationProbeRunner(
        round_markers=(marker,),
        codex_outputs=[structured_pr_review(summary=marker), structured_pr_review(summary="Codex approves round two.")],
    )
    config = make_config(tmp_path, reviewer=("codex",), review_parallel=True)
    # Publish, then stop before the spool cleanup and reconciliation.
    with monkeypatch.context() as patch:
        patch.setattr(ReviewRoundSpool, "remove", lambda self, name: None)
        patch.setattr(ReviewRoundSpool, "discard", lambda self: None)
        real = pr_loop_support.post_pr_comment

        def stop_at_reconciliation(*args, **kwargs):
            if "reconciliation" in str(kwargs.get("body", "")).lower() and marker not in str(kwargs.get("body", "")):
                raise KeyboardInterrupt
            return real(*args, **kwargs)

        patch_attr_in(patch, (pr_loop_support, pr_loop, issue_loop), "post_pr_comment", stop_at_reconciliation)
        try:
            run_pr_loop(runner, pr_number=77, config=config)
        except (KeyboardInterrupt, AgentLoopError):
            pass
    records = _comment_records(runner, surface="pr", flow="pr")
    if any(r.phase == "reconciliation" for r in records):
        pytest.skip("the one-reviewer round reconciled before the interruption point")
    assert _published_count(runner, marker) == 1 and _spool_files(config)  # content-valid spool kept
    patch_attr_in(monkeypatch, (response_validation, pr_loop), "_resumed_pr_reviewer_matches_requirements",
                  lambda record, *a, **k: record.metadata.round_number != 1)
    launches_before = _reviewer_runs(runner)
    seen = _spy_closure(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, review_parallel=review_parallel)) == 0

    (closure,) = _closures(runner)
    assert [e["agent"] for e in closure.round_closure["unreplayable"]] == ["Codex"]
    assert seen == [launches_before]
    assert _reviewer_runs(runner)[len(launches_before):] == ["codex"]
    assert _published_count(runner, marker) == 1
    reviews = [r for r in _comment_records(runner, surface="pr", flow="pr") if r.role == "reviewer"]
    assert [r.round_number for r in reviews] == [1, 2]
    _no_deletion_advice(runner)


# --- prepanel-unqualified-rollover -------------------------------------------------


def test_published_prepanel_unqualified_blocking_review_closes_an_unreconciled_round(tmp_path):
    """`prepanel-unqualified-rollover`: the round closes before any pre-panel stop; its finding
    is a prior item; round 2 needs a new qualified primary opening."""
    from pr_loop_test_helpers import (
        _agent_sequence, _posted_scheduler_metadata, _rewrite_pr_metadata, _staged_config, _staged_review,
    )
    from test_pr_loop_review_scheduling import _legacyize_operator_latches
    from test_review_parallel import _interrupt_on_phase
    from agent_loop_helpers import FakeRunner

    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(
            reviewer="Google Gemini", state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        )],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # A pre-#840 full board on the head, interrupted before its reconciliation.
    with patch_object_in((pr_loop_support, pr_loop, issue_loop), "post_pr_comment",
                         side_effect=_interrupt_on_phase(pr_loop_support.post_pr_comment, "reconciliation")):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.pr_payload["comments"])

    class _Stop(Exception):
        pass

    real = pr_loop.post_pr_comment

    def stop_after_closure(*args, **kwargs):
        result = real(*args, **kwargs)
        if "closed, incomplete" in str(kwargs.get("body", "")):
            raise _Stop
        return result

    with patch_object_in((pr_loop,), "post_pr_comment", side_effect=stop_after_closure):
        with pytest.raises(_Stop):
            run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))

    assert len(_agent_sequence(runner)) == seeded_agents
    new = runner.pr_payload["comments"][seeded:]
    assert len(new) == 1 and "closed, incomplete" in new[0]["body"]
    assert not any("pre-panel safety" in c["body"] for c in new)
    (closure,) = _closures(runner)
    assert "Gemini" in {e["agent"] for e in closure.round_closure["published"]}
    causes = {e["agent"]: e["cause"] for e in closure.round_closure["unreplayable"]}
    assert "before any qualified panel opening" in causes["Gemini"]
    assert [i.text for i in closure.new_items if i.reviewer == "Gemini"] == ["premature cache race"]
    # Round 2 proceeds without the operator override: the carried secondary-owned
    # finding is a carried obligation, so a fresh primary phase opens and the panel
    # rechecks the finding once the primary approves.
    from agent_loop_helpers import structured_coder_followup
    carried = [i.item_id for i in closure.new_items]
    resolved = [{"item_id": item, "disposition": "resolved", "note": "Fixed."} for item in carried]
    runner.claude_outputs.append(structured_coder_followup(summary="Fixed the cache race.", addressed_items=carried))
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=resolved))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=resolved))
    agents_before = len(_agent_sequence(runner))
    try:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    except AgentLoopError as exc:
        assert not isinstance(exc, pr_loop.PrePanelSafetyError), exc
    assert len(_closures(runner)) == 1  # the rerun never closes round 1 again
    later = [m for m in _posted_scheduler_metadata(runner)
             if m.phase == "scheduler-prelaunch" and m.round_number >= 2]
    assert later and later[0].scheduler_phase == "primary"
    assert later[0].scheduler_selected_reviewers == ("Codex",)
    assert not any(m.scheduler_force_full for m in later)
    assert all("Gemini" not in m.scheduler_approved_reviewers for m in later[:1])
    # The configured board reviews after the closure: the primary, then the panel.
    new_agents = _agent_sequence(runner)[agents_before:]
    assert {"codex", "gemini", "agy"} <= set(new_agents), new_agents
    assert any(m.scheduler_phase == "secondary-audit" for m in later)
    _no_deletion_advice(runner)


# --- item-1 (round 1 review): every completed review enters the closure ledger -------


def test_closure_ledger_keeps_an_authoritative_sequential_review_findings(tmp_path):
    """A sequential (authoritative) review's minted finding survives a closure that another
    publication triggers (here a malformed spool record of the unpublished reviewer)."""
    from coding_review_agent_loop import review_rounds

    markers = ("Codex blocks sequentially.", "Gemini reviews second.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        codex_outputs=[
            structured_pr_review(state="blocking", summary=markers[0], blocking_items=["Bound the retry loop."]),
            structured_pr_review(summary="Codex approves round two.",
                                 prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            structured_pr_review(summary=markers[1], reviewer="Google Gemini"),
            structured_pr_review(summary="Gemini approves round two.", reviewer="Google Gemini",
                                 prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=False)
    _interrupt_pr(runner, config, markers, at=2)  # Codex posted; Gemini's turn is lost
    codex = next(r for r in _comment_records(runner, surface="pr", flow="pr") if r.agent == "Codex")
    assert codex.phase == "authoritative" and [i.item_id for i in codex.new_items] == ["item-1"]
    spool = review_rounds._review_round_spool(config, surface="pr", number=77, round_number=1, subject=HEAD)
    spool.store("Gemini", {"text": "x"})
    spool._path("Gemini").write_text("{broken", encoding="utf-8")  # publication cannot be excluded

    class _Stop(Exception):
        pass

    real = pr_loop.post_pr_comment

    def stop_after_closure(*args, **kwargs):
        result = real(*args, **kwargs)
        if "closed, incomplete" in str(kwargs.get("body", "")):
            raise _Stop
        return result

    with patch_object_in((pr_loop,), "post_pr_comment", side_effect=stop_after_closure):
        with pytest.raises(_Stop):
            run_pr_loop(runner, pr_number=77, config=config)

    (closure,) = _closures(runner)
    assert [e["agent"] for e in closure.round_closure["unreplayable"]] == ["Gemini"]
    # Codex's authoritative finding keeps its minted ID in the closure ledger.
    assert [(i.item_id, i.text) for i in closure.new_items] == [("item-1", "Bound the retry loop.")]
    assert closure.round_closure["item_number_high_water"] == 1
    _no_deletion_advice(runner)


# --- managed-ci-binding-kept ------------------------------------------------------


def _managed_checkpoint_comment(config):
    """A managed-CI qualifying checkpoint opening round 1 at HEAD (as in the #2035 fixture)."""
    from coding_review_agent_loop.qualification_checkpoint import QualificationCheckpoint
    from coding_review_agent_loop.unresolved_items import UnresolvedReviewItem

    acquisition = {
        name: (inv.configured_model, inv.resolved_effort, provider)
        for name, provider in (("Codex", "codex"), ("Gemini", "gemini"))
        for inv in [pr_loop.resolve_invocation(config, provider=provider, role="reviewer")]
    }
    item = UnresolvedReviewItem(
        item_id="item-1", reviewer="GitHub managed exact-head CI", source_round=1,
        text="Managed CI failed on the previous head.", status="blocking",
        source_status="blocking", authority="machine",
        obligation_kind="managed-exact-head-ci", lifecycle="qualifying",
        failed_head_sha="old-head", candidate_head_sha=HEAD,
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci", obligation_identity=item.obligation_identity,
        lifecycle="qualifying", failed_head_sha="old-head", candidate_head_sha=HEAD,
        base_branch="main",
        requirements_digest=pr_loop_support._qualification_digest(tuple()),
        acquisition_digest=pr_loop_support._qualification_digest(acquisition),
        approval_digest=pr_loop_support._qualification_digest(("Codex", "Gemini")),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature((item,))),
        qualification_attempt_id="123/1", allowed_rounds=config.max_rounds,
    )
    body = _attach_round_metadata(
        "Qualification was dispatched.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=HEAD,
            prior_items=(item,), state="blocking", phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint, architecture_contract_version=1,
        ),
    )
    return body, checkpoint


def test_closure_keeps_the_managed_ci_qualification_checkpoint_readable(tmp_path, monkeypatch):
    """`managed-ci-binding-kept`: round 1 opens on a managed-CI qualification
    checkpoint and is partially published.  The closure posts one record and
    deletes nothing: the checkpoint comment stays verbatim and still decodes
    with the same binding, before and after the rollover into round 2."""
    from coding_review_agent_loop.managed_ci_types import ManagedCiContract

    markers = ("Codex reviews the qualifying head.", "Gemini reviews the qualifying head.")
    done = [{"item_id": "item-1", "disposition": "resolved", "note": "Fixed."}]
    runner = _PartialPublicationProbeRunner(
        round_markers=markers, slow_reviewer="gemini",
        codex_outputs=[structured_pr_review(summary=summary, prior_item_dispositions=done)
                       for summary in (markers[0], "Codex approves round two.")],
        gemini_outputs=[structured_pr_review(summary=summary, reviewer="Google Gemini", prior_item_dispositions=done)
                        for summary in (markers[1], "Gemini approves round two.")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True,
                         managed_ci=True, max_rounds=3)
    body, checkpoint = _managed_checkpoint_comment(config)
    runner.pr_payload["comments"].insert(0, {"author": {"login": "coding-review-agent-loop"}, "body": body})
    contract = ManagedCiContract(protocol_version=2, attached_run_id=123, run_attempt=1, expected_head_sha=HEAD)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)
    _interrupt_pr(runner, config, markers, at=2)
    assert sum(_published_count(runner, marker) for marker in markers) == 1
    for path in _spool_files(config):
        path.unlink()
    seen = _spy_closure(monkeypatch, runner)

    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError:
        pass  # managed qualification after round 2 is out of this row's scope

    assert len(seen) == 1
    (closure,) = _closures(runner)
    assert closure.round_number == 1
    assert runner.pr_payload["comments"][0]["body"] == body
    records = _comment_records(runner, surface="pr", flow="pr")
    checkpoints = [r.qualification_checkpoint for r in records if r.phase == "qualification-checkpoint"]
    assert checkpoints[0] == checkpoint
    assert sorted(r.agent for r in records if r.role == "reviewer" and r.round_number == 2) == ["Codex", "Gemini"]
    _no_deletion_advice(runner)


# --- #2064 stage 3: the removed resume paths now roll the round over ---------


def test_review_formerly_admitted_by_classification_only_resume_rolls_the_round_over(tmp_path):
    """`former-resume-paths-rollover` (classification-only resume removed, #2081).

    Every review of round 1 is published, then Codex's signed-source
    classification exhausts its re-ask.  The old release admitted Codex's
    published review for a classification-only continuation; now resume
    rejects it, so the reviewer is unreplayable: round 1 closes with one
    closed-incomplete record, Codex is not re-run in round 1, and round 2
    reviews the full board.  No stop, no deletion advice.
    """
    from coding_review_agent_loop.github import _parse_pr_human_requirements
    from coding_review_agent_loop.protocol import human_requirements_resolved
    from pr_loop_test_helpers import _association, _classified_review, _pr_records
    from test_pr_loop_resume import _CLASSIFY, _classification_failed_round

    assert not hasattr(response_validation, "classification_resume_kind")
    assert not hasattr(response_validation, "admit_classification_resumes")
    runner, config, rids = _classification_failed_round(tmp_path)
    requirements = _parse_pr_human_requirements(runner.pr_payload)
    codex = [r for r in _pr_records(runner) if r.metadata.agent == "Codex"][-1]
    assert codex.metadata.phase == "publication" and not human_requirements_resolved(codex.body)
    assert not response_validation._resumed_pr_reviewer_matches_requirements(codex, requirements)
    before = len(runner.commands)
    runner.codex_outputs = [_classified_review([_association(rid, "code") for rid in rids], hr=True)]
    runner.gemini_outputs = [_classified_review([_association(rid, "code") for rid in rids],
                                                reviewer="Google Gemini", hr=True)]

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    (closure,) = _closures(runner)
    assert closure.round_number == 1
    assert [e["agent"] for e in closure.round_closure["published"]] == ["Codex", "Gemini"]
    assert [e["agent"] for e in closure.round_closure["unreplayable"]] == ["Codex"]
    rerun = runner.commands[before:]
    codex_prompts = [cmd[-1] for cmd, _cwd in rerun if cmd[:1] == ["codex"]]
    assert len(codex_prompts) == 1 and _CLASSIFY not in codex_prompts[0]  # a fresh round-2 turn
    assert sum(1 for cmd, _cwd in rerun if cmd[:1] == ["gemini"]) == 1
    reviewer_rounds = [r.metadata.round_number for r in _pr_records(runner)
                       if r.metadata.role == "reviewer" and r.metadata.agent == "Codex"]
    assert reviewer_rounds == [1, 2]
    assert not runner.codex_outputs and not runner.gemini_outputs
    _no_deletion_advice(runner)


@pytest.mark.parametrize("review_parallel", [True, False], ids=["parallel", "sequential"])
def test_partly_public_publication_rolls_the_round_over_without_posting_its_suffix(tmp_path, review_parallel):
    """`former-resume-paths-rollover` (publication-carrier resume removed, #2081).

    Codex's overflowing round-1 review publishes its sidecars and the outage
    hits before its anchor.  The old release resumed the frozen carrier and
    posted the missing suffix; now the begun publication is unreplayable: the
    rerun posts one closed-incomplete record, never the anchor, and launches no
    reviewer in round 1 -- even a sequential rerun of the withheld round.
    """
    import coding_review_agent_loop.github as github_module
    from coding_review_agent_loop.round_transport import attachment_keys
    from test_review_parallel import _overflow_pr_runner, _run_until_closure

    runner, markers = _overflow_pr_runner((11,))
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    real = github_module._post_comment_body
    state = {"anchor": None}

    def fail_codex_anchor(runner_, *, config, command, body, claimed=None, body_index=None):
        if markers[0] in str(body) and not attachment_keys(str(body)):
            state["anchor"] = str(body)
            raise KeyboardInterrupt
        return real(runner_, config=config, command=command, body=body, claimed=claimed,
                    body_index=body_index)

    with patch_object_in((github_module,), "_post_comment_body", side_effect=fail_codex_anchor):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert state["anchor"] is not None
    sidecars = [c["body"] for c in runner.pr_payload["comments"] if attachment_keys(c["body"])]
    assert sidecars  # the publication is partly public
    marks = [json.loads(p.read_text(encoding="utf-8")).get("publication") for p in _spool_files(config)]
    assert [set(m) for m in marks if m is not None] == [{"began_at"}]
    launches_before = len(runner.reviewer_launches)
    comments_before = len(runner.pr_payload["comments"])
    rerun_config = dataclasses.replace(config, review_parallel=review_parallel)

    _run_until_closure(lambda: run_pr_loop(runner, pr_number=77, config=rerun_config))

    assert runner.reviewer_launches[launches_before:] == []
    assert len(runner.pr_payload["comments"]) == comments_before + 1  # only the closure record
    assert not any(c["body"] == state["anchor"] for c in runner.pr_payload["comments"])
    assert [c["body"] for c in runner.pr_payload["comments"] if attachment_keys(c["body"])] == sidecars
    (closure,) = _closures(runner)
    assert closure.round_number == 1
    unreplayable = {e["agent"]: e["cause"] for e in closure.round_closure["unreplayable"]}
    assert "Codex" in unreplayable and "began" in unreplayable["Codex"]
    _no_deletion_advice(runner)


def _seat_board(tmp_path, flash_model, **overrides):
    """A named-seat board (flash, opus); ``flash_model`` sets flash's seat binding."""
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    flash = SeatAgent(ReviewerSeat("flash", "antigravity", (flash_model,)), tmp_path / "flash")
    opus = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model B",)), tmp_path / "opus")
    flash.workdir.mkdir(exist_ok=True)
    opus.workdir.mkdir(exist_ok=True)
    return make_config(tmp_path, reviewer=(flash, opus), reviewer_seats=(flash, opus),
                       pre_review_tests=False, agent_max_retries=0, **overrides)


def _seat_review(seat, model, summary, **kwargs):
    return (structured_pr_review(summary=summary, reviewer=f"{seat} (Google Antigravity: {model})", **kwargs), 0)


def _old_seat_binding(tmp_path):
    """flash's binding under the previous model, taken from a real board."""
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding

    return reviewer_seat_binding(_seat_board(tmp_path, "Model A"))


def _seed_old_seat_record(spool, state, old):
    """Store flash's round-1 record under ``old`` in publication ``state``."""
    spool.store("flash", {"text": structured_pr_review(summary="Stale under the old seat."),
                          "seat_binding": old})
    if state != "fresh":
        path = spool._path("flash")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if state == "began":
            payload["publication"] = {"began_at": "2026-11-26T00:00:00+00:00"}
        elif state == "legacy":
            del payload["carrier_protocol"]
        else:  # malformed: a non-mapping publication mark on a loadable record
            payload["publication"] = None
        path.write_text(json.dumps(payload), encoding="utf-8")
    assert spool.publication_state("flash") == state
    # The record still loads, so it reaches the seat-reconfiguration check.
    assert spool.load("flash") is not None and spool.load("flash")["seat_binding"] == old


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
@pytest.mark.parametrize("state", ["fresh", "began", "legacy", "malformed"])
def test_seat_reconfiguration_reads_the_spool_publication_state_string(tmp_path, monkeypatch, state, policy):
    """A spooled response saved under another seat binding (#2081 item-1, #2150).

    flash's seat moved from Model A to Model C.  A never-published (fresh)
    record is discarded and flash reviews round 1 under the new binding, with
    no closure.  A began, legacy or malformed record is no longer a terminal
    stop: before any reviewer launches, round 1 closes with one
    closed-incomplete record naming flash as unreplayable, the stale text is
    never posted, and both seats review round 2 under the new binding.  The
    same holds for the non-selective and the selective board.
    """
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_closure_writer import _POSSIBLY_PUBLIC_STATES

    old = _old_seat_binding(tmp_path)
    config = _seat_board(tmp_path, "Model C", review_parallel=True, pr_review_policy=policy)
    new = reviewer_seat_binding(config)
    assert new is not None and old is not None and new != old
    runner = FakeRunner(antigravity_outputs=[
        _seat_review("flash", "Model C", "flash reviews under the new seat."),
        _seat_review("opus", "Model B", "opus approves."),
    ])
    spool = pr_loop._review_round_spool(config, surface="pr", number=77, round_number=1, subject=HEAD)
    _seed_old_seat_record(spool, state, old)
    seen = _spy_closure(monkeypatch, runner)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not any("Stale under the old seat." in body for body in runner.comments)
    assert not any("refusing recovery" in body for body in runner.comments)
    _no_deletion_advice(runner)
    reviews = [r for r in _comment_records(runner, surface="pr", flow="pr") if r.role == "reviewer"]
    review_rounds = {r.agent: r.round_number for r in reviews}
    assert all(r.seat_binding == new for r in reviews)
    if state == "fresh":
        assert spool.publication_state("flash") == "absent"
        assert _closures(runner) == [] and seen == []
        assert review_rounds == {"flash": 1, "opus": 1}
        return
    (closure,) = _closures(runner)
    assert closure.round_number == 1
    assert seen == [[]]  # no reviewer launched before the round-1 closure
    unreplayable = {e["agent"]: e["cause"] for e in closure.round_closure["unreplayable"]}
    assert unreplayable["flash"] == _POSSIBLY_PUBLIC_STATES[state]
    assert review_rounds == {"flash": 2, "opus": 2}
    assert not runner.antigravity_outputs

@pytest.mark.parametrize("review_parallel", [True, False], ids=["parallel", "sequential"])
def test_stale_seat_record_beside_an_admitted_public_anchor_is_removed_without_closing(tmp_path, review_parallel):
    """`seat-public-anchor` (#2150): flash's round-1 anchor is public under the
    current seat binding and resume admits it, while a stale begun record
    under flash's old binding remains in the spool.

    The seat-reconfiguration check keeps the record; the preflight then
    removes it because flash's public anchor was admitted.  opus's withheld
    outcome replays, so round 1 completes with no closure, no duplicate
    anchor, and the stale text is never posted.
    """
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent, reviewer_seat_binding

    def board(flash_model):
        flash = SeatAgent(ReviewerSeat("flash", "codex", (flash_model,)), tmp_path / "flash")
        opus = SeatAgent(ReviewerSeat("opus", "gemini", ("Model B",)), tmp_path / "opus")
        flash.workdir.mkdir(exist_ok=True)
        opus.workdir.mkdir(exist_ok=True)
        config = make_config(tmp_path, reviewer=(flash, opus), reviewer_seats=(flash, opus),
                             review_parallel=True, pre_review_tests=False, agent_max_retries=0)
        return config, flash, opus

    old = reviewer_seat_binding(board("Model A")[0])
    config, flash, opus = board("Model C")
    assert reviewer_seat_binding(config) != old
    markers = ("flash approves independently.", "opus approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers, slow_reviewer="gemini",
        codex_outputs=[structured_pr_review(summary=markers[0], reviewer="flash (OpenAI Codex: Model C)")],
        gemini_outputs=[structured_pr_review(summary=markers[1], reviewer="opus (Google Gemini: Model B)")],
    )
    _interrupt_pr(runner, config, markers, at=2)
    assert [_published_count(runner, m) for m in markers] == [1, 0]
    spool = pr_loop._review_round_spool(config, surface="pr", number=77, round_number=1, subject=HEAD)
    assert spool.publication_state("flash") == "absent"
    assert spool.publication_state("opus") == "fresh"  # opus's withheld outcome is replayable
    spool.store("flash", {"text": structured_pr_review(summary="Stale under the old seat."),
                          "seat_binding": old})
    spool.mark_publication_began("flash")
    assert spool.publication_state("flash") == "began"
    launches_before = list(runner.reviewer_launches)

    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, review_parallel=review_parallel)) == 0

    assert _closures(runner) == []
    assert spool.publication_state("flash") == "absent"  # the stale record was removed
    assert [_published_count(runner, m) for m in markers] == [1, 1]  # no duplicate anchor
    assert runner.reviewer_launches == launches_before  # nobody re-ran round 1
    assert not any("Stale under the old seat." in body for body in runner.comments)
    reviews = [r for r in _comment_records(runner, surface="pr", flow="pr") if r.role == "reviewer"]
    assert sorted((r.agent, r.round_number) for r in reviews) == [("flash", 1), ("opus", 1)]
    _no_deletion_advice(runner)
