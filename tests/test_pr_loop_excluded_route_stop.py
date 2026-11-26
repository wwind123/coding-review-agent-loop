"""Routes that keep their own records name a working next step (#2163).

The evidence pass, the unrecorded-head-advance recovery route and the merge
conflict route launch no withheld-round reviewer, so no former refusal site
runs there and a begun spool record is neither replayed nor refused.  Their
defensive stop names one next step: rerun the same ``agent-loop pr <N>``
command, which resumes the route from its own records; the next ordinary
round's start then closes a round that still holds an incomplete publication.
"""

from types import SimpleNamespace

import pytest

import coding_review_agent_loop.pr_loop as pr_loop
from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.round_closure_writer import (
    RoundIncompleteSignal,
    excluded_incomplete_round_route,
    excluded_route_stop,
    round_incomplete_signal,
)
from coding_review_agent_loop.round_metadata import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.round_state import _extract_round_metadata_records

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup, structured_pr_review
from pr_loop_test_helpers import _EVIDENCE_TEXT, _evidence_review, _resolve, _signed

HEAD = "6367c48dd193d56ea7b0baad25b19455e529f5ee"
NEW = "fd2d39cf710f84ca537091424fc212fa5211885b"
WITHHELD = "Withheld spooled outcome that must never be replayed."
ROUTES = {
    "evidence": dict(evidence_pass=True, recovery=True, conflict=False),
    "recovery": dict(evidence_pass=False, recovery=True, conflict=False),
    "conflict": dict(evidence_pass=False, recovery=False, conflict=True),
}


def _records(runner):
    return _extract_round_metadata_records(
        [SimpleNamespace(body=comment["body"]) for comment in runner.pr_payload["comments"]], flow="pr",
    )


def _closures(runner):
    return [r for r in _records(runner) if r.metadata.phase == "round-closed-incomplete"]


def _agents(runner):
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]


def _began(config, round_number, subject, name="Codex", reviewer="OpenAI Codex"):
    """A spool record whose same-round publication began (it may be partly public)."""
    spool = pr_loop._review_round_spool(config, surface="pr", number=77, round_number=round_number, subject=subject)
    spool.store(name, {"text": structured_pr_review(summary=WITHHELD, reviewer=reviewer)})
    spool.mark_publication_began(name)
    assert spool.publication_state(name) == "began"
    return spool


def _route_stop(route_key, round_number):
    """The stop the loop's handler raises on this route, from the same helpers."""
    route = excluded_incomplete_round_route(**ROUTES[route_key])
    signal = round_incomplete_signal(
        surface="pr", number=77, round_number=round_number, published=("Codex",),
        unreplayable=(("Gemini", "its withheld outcome is missing or no longer validates"),),
    )
    return excluded_route_stop(signal, route=route, pr_number=77)


def _assert_rerun_step(message, route):
    assert f"inside its {route}" in message
    assert "Nothing was posted and no reviewer was launched" in message
    assert "Next step: rerun `agent-loop pr 77` with the same options" in message
    assert f"the {route} resumes from its own records" in message
    assert "delet" not in message.lower() and "--resume" not in message


def _inject_route_stop(monkeypatch, builder, route_key, round_number):
    """The route's first seam raises the route's defensive stop once, before any post."""
    real = getattr(pr_loop, builder)
    armed = [True]

    def build(*args, **kwargs):
        if armed[0]:
            armed[0] = False
            raise _route_stop(route_key, round_number)
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop, builder, build)
    return armed


# --- unit -----------------------------------------------------------------


def test_route_labels_name_the_conflict_route_correctly():
    assert excluded_incomplete_round_route(**ROUTES["evidence"]) == "evidence pass"
    assert excluded_incomplete_round_route(**ROUTES["recovery"]) == "recovery route"
    assert excluded_incomplete_round_route(**ROUTES["conflict"]) == "conflict route"
    assert excluded_incomplete_round_route(evidence_pass=False, recovery=True, conflict=True) == "conflict route"
    # A conversion repeat is not an excluded route: it closes and rolls over.
    assert excluded_incomplete_round_route(evidence_pass=False, recovery=False, conflict=False) is None


@pytest.mark.parametrize("route_key", sorted(ROUTES))
def test_the_defensive_stop_names_the_rerun_step_and_no_removed_remedy(route_key):
    stop = _route_stop(route_key, 4)
    message = str(stop)
    route = excluded_incomplete_round_route(**ROUTES[route_key])
    _assert_rerun_step(message, route)
    assert "if round 4 still holds an incomplete publication, the next ordinary round's start closes it" in message
    assert "review continues in round 5" in message
    assert "writes sequential or coder records, not withheld-round publications" in message
    assert isinstance(stop, RoundIncompleteSignal) and stop.before_publication is False
    assert stop.published == ("Codex",)
    if route_key == "conflict":
        assert "recovery route" not in message


# --- conflict route -------------------------------------------------------


_CONFLICT = {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": HEAD, "baseRefName": "main"}


def _conflict_runner(conflicts=1):
    return FakeRunner(
        claude_outputs=["Resolved the conflict.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        mergeability_payloads=[dict(_CONFLICT) for _ in range(conflicts)],
    )


def test_the_conflict_route_never_reaches_a_refusal_site_with_a_begun_spool(tmp_path):
    """`excluded-gating`, conflict: no reviewer, no closure, the begun record untouched."""
    runner = _conflict_runner()
    config = make_config(tmp_path)
    spool = _began(config, 1, HEAD)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agents(runner) == ["claude", "codex"]  # the conflict round launched no reviewer
    assert _closures(runner) == []
    assert not any(r.metadata.round_number == 1 and r.metadata.role == "reviewer" for r in _records(runner))
    assert spool.publication_state("Codex") == "began"  # neither replayed nor refused
    assert not any(WITHHELD in comment for comment in runner.comments)


def test_a_conflict_route_stop_is_resumed_by_rerunning_the_same_command(tmp_path, monkeypatch):
    """`excluded-stop-rerun`, conflict: the rerun resumes the coder route and advances."""
    runner = _conflict_runner(conflicts=2)
    config = make_config(tmp_path)
    _inject_route_stop(monkeypatch, "build_merge_conflict_prompt", "conflict", 1)
    with pytest.raises(RoundIncompleteSignal) as stopped:
        run_pr_loop(runner, pr_number=77, config=config)
    _assert_rerun_step(str(stopped.value), "conflict route")
    assert list(_records(runner)) == [] and _agents(runner) == []
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agents(runner) == ["claude", "codex"]
    assert [r.metadata.role for r in _records(runner)] == ["coder", "reviewer"]


# --- recovery route -------------------------------------------------------


def _recovery_runner():
    """An unrecorded head advance: the old head's records, a new live head."""
    old_item = UnresolvedReviewItem(
        item_id="item-2", reviewer="Google Gemini", source_round=1,
        text="Preserve the metadata-backed unresolved item on rerun.", status="blocking",
    )
    coder = _attach_round_metadata(
        "Opened the PR.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(flow="pr", role="coder", agent="Claude", round_number=1, subject="old-sha", prior_items=()),
    )
    review = _attach_round_metadata(
        "Blocked.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1, subject="old-sha",
                            prior_items=(), new_items=(old_item,), state="blocking"),
    )
    return FakeRunner(
        claude_outputs=[structured_coder_followup(
            summary="Addressed the recovered prior item.", addressed_items=["item-2"],
            tests_run=["python -m pytest tests/test_agent_loop.py -k unrecorded_head"],
        )],
        codex_outputs=[structured_pr_review(
            state="approved", summary="Recovered item is resolved.",
            prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        )],
        pr_payload={"headRefOid": NEW, "comments": [
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:01:00Z", "body": review},
        ]},
        advance_pr_head_on_coder_followup=False,
    )


def _assert_recovery_then_ordinary_closure(runner, spool):
    """The route runs the coder only; the next ordinary round closes the stale round."""
    assert _agents(runner) == ["claude", "codex"]
    records = [r for r in _records(runner) if r.metadata.subject == NEW]
    assert [(r.metadata.round_number, r.metadata.role, r.metadata.phase) for r in records] == [
        (2, "coder", "authoritative"),
        (2, "summary", "round-closed-incomplete"),
        (3, "reviewer", "authoritative"),
    ]
    closure = records[1].metadata.round_closure
    assert [entry["agent"] for entry in closure["published"]] == ["Codex"]
    assert spool.publication_state("Codex") == "absent"  # discarded after the closure was saved
    assert not any(WITHHELD in comment for comment in runner.comments)


def test_the_recovery_route_never_reaches_a_refusal_site_with_a_begun_spool(tmp_path):
    """`excluded-gating`, recovery: the coder runs first; no reviewer launches on the route."""
    runner = _recovery_runner()
    config = make_config(tmp_path, reviewer="codex")
    spool = _began(config, 2, NEW)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    _assert_recovery_then_ordinary_closure(runner, spool)


def test_a_recovery_route_stop_is_resumed_by_rerunning_the_same_command(tmp_path, monkeypatch):
    """`excluded-stop-rerun`, recovery: the rerun resumes the route; round 2 then closes."""
    runner = _recovery_runner()
    config = make_config(tmp_path, reviewer="codex")
    spool = _began(config, 2, NEW)
    comments_before = len(runner.pr_payload["comments"])
    _inject_route_stop(monkeypatch, "build_followup_prompt", "recovery", 2)
    with pytest.raises(RoundIncompleteSignal) as stopped:
        run_pr_loop(runner, pr_number=77, config=config)
    _assert_rerun_step(str(stopped.value), "recovery route")
    assert len(runner.pr_payload["comments"]) == comments_before and _agents(runner) == []
    assert spool.publication_state("Codex") == "began"
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    _assert_recovery_then_ordinary_closure(runner, spool)


# --- evidence pass --------------------------------------------------------


def _frozen_two_reviewer_runner(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(_signed(f"Live suite at {HEAD}: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", hr=True, dispositions=[_resolve("item-1", "Codex owns this request.")],
    ))
    return runner, config


def _assert_sequential_evidence_pass(runner, *, before):
    assert _agents(runner)[before:] == ["codex", "gemini"]
    phases = [(r.metadata.agent, r.metadata.phase) for r in _records(runner)]
    assert ("Codex", "evidence-response") in phases and ("Gemini", "evidence-response") in phases
    assert phases[-1] == ("Orchestrator", "evidence-release")
    assert _closures(runner) == []
    assert not any(WITHHELD in comment for comment in runner.comments)


def test_the_evidence_pass_never_reaches_a_refusal_site_with_a_begun_spool(tmp_path):
    """`excluded-gating`, evidence pass: only its sequential evidence reviewers run."""
    runner, config = _frozen_two_reviewer_runner(tmp_path)
    _began(config, 1, HEAD, name="Gemini", reviewer="Google Gemini")
    before = len(_agents(runner))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    # No parallel or withheld-round launch: Gemini is invoked fresh, its begun
    # record is neither replayed nor refused, and no closure record is posted.
    _assert_sequential_evidence_pass(runner, before=before)


def test_an_evidence_pass_stop_is_resumed_by_rerunning_the_same_command(tmp_path, monkeypatch):
    """`excluded-stop-rerun`, evidence pass: the rerun runs the evidence reviewers."""
    runner, config = _frozen_two_reviewer_runner(tmp_path)
    before, comments_before = len(_agents(runner)), len(runner.pr_payload["comments"])
    _inject_route_stop(monkeypatch, "build_review_prompt", "evidence", 1)
    with pytest.raises(RoundIncompleteSignal) as stopped:
        run_pr_loop(runner, pr_number=77, config=config)
    _assert_rerun_step(str(stopped.value), "evidence pass")
    assert len(runner.pr_payload["comments"]) == comments_before and len(_agents(runner)) == before
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    _assert_sequential_evidence_pass(runner, before=before)
