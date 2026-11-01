"""Recorded in-run PR reviewer-board changes (#1376, stage 2 of #1373).

Integration tests drive the real ``run_pr_loop`` through ``FakeRunner``; the
unit tests pin the link record, the resolver and the floor authorization.
"""

import base64
import dataclasses
import json
from types import SimpleNamespace

import pytest

from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
from coding_review_agent_loop.board_change import (
    BOARD_CHANGE_LINK_HEADING,
    BOARD_CHANGE_LINK_MARKER,
    BOARD_CHANGE_LINK_RE,
    BoardIdentity,
    board_anchor,
    board_change_link_digest,
    board_change_link_payload,
    identity_from_config,
    lineage_floor_authorizations,
    render_board_change_link,
    resolve_board_lineage,
)
from coding_review_agent_loop.cli import run_pr_loop
from coding_review_agent_loop.errors import AgentInvocationError, AgentLoopError
from coding_review_agent_loop.pr_loop_support import _scheduler_contract_from_metadata
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.reviewer_floor import BoardSeat
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _extract_round_metadata_records,
)
import coding_review_agent_loop.child_plan_binding as child_plan_binding_module
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy_module
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.pr_loop as pr_loop_module
import coding_review_agent_loop.pr_loop_support as pr_loop_support_module
from agent_loop_helpers import patch_attr_in

_AGENTS = {"codex", "claude", "agy", "gemini"}
_ACTOR_ID = 424242  # FakeRunner's default authenticated PR actor
_RESOLVED_1 = [{"item_id": "item-1", "disposition": "resolved"}]


# --- helpers ---------------------------------------------------------------------


def _comments(runner):
    return [
        SimpleNamespace(body=c["body"], author_id=c.get("_rest_author_id"))
        for c in runner.pr_payload.get("comments", [])
    ]


def _records(runner):
    return _extract_round_metadata_records(_comments(runner), flow="pr")


def _links(runner):
    return [
        c["body"] for c in runner.pr_payload.get("comments", [])
        if c["body"].startswith(BOARD_CHANGE_LINK_HEADING)
    ]


def _link_payloads(runner):
    payloads = []
    for body in _links(runner):
        match = BOARD_CHANGE_LINK_RE.search(body)
        payloads.append(json.loads(base64.urlsafe_b64decode(match.group("payload"))))
    return payloads


def _agents(runner, start=0):
    return [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd and cmd[0] in _AGENTS]


def _seat(tmp_path, name, backend, model, effort="medium"):
    workdir = tmp_path / name
    workdir.mkdir(exist_ok=True)
    return SeatAgent(ReviewerSeat(name, backend, (model,), effort), workdir)


def _named(tmp_path, *seats, **overrides):
    named = tuple(seat for seat in seats if isinstance(seat, SeatAgent))
    return make_config(
        tmp_path, reviewer=tuple(seats), reviewer_seats=named, pre_review_tests=False,
        agent_max_retries=0, **overrides,
    )


def _append(runner, body, *, login="operator", author_id=None):
    comments = runner.pr_payload.setdefault("comments", [])
    entry = {
        "author": {"login": login},
        "createdAt": f"2026-05-23T00:00:{len(comments):02d}Z",
        "body": body,
    }
    if author_id is not None:
        entry["_rest_author_id"] = author_id
    comments.append(entry)


def _legacy(role, agent, round_number, *, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", state=None, new_items=(),
            prior_items=(), dispositions=(), **extra):
    """A pre-upgrade round record: no board identity fields."""
    if role == "reviewer":
        body = structured_pr_review(
            state=state or "approved", reviewer=agent,
            blocking_items=[item.text for item in new_items],
        )
    elif role == "coder":
        body = structured_coder_followup()
    else:
        body = f"{agent} record.\n-- Orchestrator"
    return _attach_round_metadata(
        body,
        PostedRoundMetadata(
            flow="pr", role=role, agent=agent, round_number=round_number, subject=subject,
            state=state, new_items=tuple(new_items), prior_items=tuple(prior_items),
            dispositions=tuple(dispositions), **extra,
        ),
    )


def _item(number, reviewer, *, status="blocking", round_number=1, owners=(), states=()):
    return UnresolvedReviewItem(
        item_id=f"item-{number}", reviewer=reviewer, source_round=round_number,
        text=f"finding {number} by {reviewer}", status=status,
        resolution_owners=tuple(owners), owner_states=tuple(states),
    )


def _legacy_runner(bodies, **outputs):
    return FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"},
                 "createdAt": f"2026-05-20T09:{index:02d}:00Z",
                 "body": body, "_rest_author_id": _ACTOR_ID}
                for index, body in enumerate(bodies)
            ]
        },
        **outputs,
    )


def _board_state(runner, config):
    """The read-only board resolution over the PR's live comment projection."""
    from coding_review_agent_loop.github import get_pr_review_context
    from coding_review_agent_loop.pr_board_state import resolve_pr_board_state

    return resolve_pr_board_state(
        runner, config=config, pr_number=77,
        comments=get_pr_review_context(runner, config=config, pr_number=77).comments,
        drift_error=lambda persisted, detail: AgentLoopError(detail),
    )


def _qualify(runner, config):
    """The qualification gate's fresh-comment snapshot (every policy)."""
    return pr_loop_support_module._fresh_pr_qualification_snapshot(
        runner, config=config, pr_number=77, issue_context=None, parent_issue_context=None,
    )


# --- pr-swap-midrun / pr-add-midrun ------------------------------------------------


def _approved_two_reviewer_run(tmp_path, reviewers=("codex", "gemini"), **config):
    outputs = {
        "codex": "codex_outputs", "gemini": "gemini_outputs",
        "antigravity": "antigravity_outputs", "claude": "claude_outputs",
    }
    reviews = {
        "codex": structured_pr_review(), "gemini": structured_pr_review(reviewer="Google Gemini"),
        "antigravity": structured_pr_review(reviewer="Antigravity"),
        "claude": structured_pr_review(reviewer="Anthropic Claude"),
    }
    runner = FakeRunner(**{outputs[name]: [reviews[name]] for name in reviewers})
    cfg = make_config(tmp_path, reviewer=tuple(reviewers), **config)
    assert run_pr_loop(runner, pr_number=77, config=cfg) == 0
    return runner, cfg


def test_all_reviewers_seat_swap_posts_one_link_and_requires_the_new_seat(tmp_path):
    """Row pr-swap-midrun: plain all-reviewers board, Gemini replaced by Antigravity."""
    runner, _ = _approved_two_reviewer_run(tmp_path)
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["removed"] == ["Gemini"]
    assert (payload["anchor_round"], payload["advance"], payload["effective_from_round"]) == (1, True, 2)
    # No signed amendment was needed, and only the new seat reviewed.
    assert not any("Reviewer board amendment" in c for c in runner.comments)
    assert _agents(runner, start) == ["agy"]
    round_two = [r.metadata for r in _records(runner) if r.metadata.round_number == 2]
    assert round_two and all(
        m.reviewer_board_change_digest == board_change_link_digest(payload) for m in round_two
    )
    # Qualification revalidates the chain from fresh comments; the run's board passes.
    _qualify(runner, swapped)
    # A rerun with the same board derives the same lineage and posts nothing.
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1


def test_selective_named_seat_model_change_requires_fresh_approval(tmp_path):
    """Row pr-swap-midrun (selective): same seat ID moved to another model."""
    opus = _seat(tmp_path, "opus", "antigravity", "Model A")
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        antigravity_outputs=[(structured_pr_review(reviewer="opus (Google Antigravity: Model A)"), 0)],
    )
    config = _named(tmp_path, "codex", opus, pr_review_policy="selective-intermediate")
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    moved = _seat(tmp_path, "opus", "antigravity", "Model B")
    rebound = _named(tmp_path, "codex", moved, pr_review_policy="selective-intermediate")
    runner.antigravity_outputs.append(
        (structured_pr_review(reviewer="opus (Google Antigravity: Model B)"), 0)
    )
    # A named board's approvals are bound to the whole recorded seat binding,
    # so every seat re-approves the head under the rebound board.
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["opus"] and not payload["added"] and not payload["removed"]
    # The rebound seat reviewed again on its new model.
    commands = [cmd for cmd, _cwd in runner.commands[start:] if cmd[0] == "agy"]
    assert len(commands) == 1 and "Model B" in " ".join(commands[0])
    assert sorted(_agents(runner, start)) == ["agy", "codex"]
    scheduler = [
        r.metadata for r in _records(runner)
        if r.metadata.scheduler_contract is not None and r.metadata.round_number >= 2
    ]
    assert scheduler and all(m.reviewer_board_change_digest for m in scheduler)


def test_added_reviewer_is_required_from_the_effective_round(tmp_path):
    """Row pr-add-midrun: the added reviewer must approve the current head."""
    runner, _ = _approved_two_reviewer_run(tmp_path)
    larger = make_config(tmp_path, reviewer=("codex", "gemini", "antigravity"))
    start = len(runner.commands)
    # Without the added reviewer's approval the run cannot finish.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=larger)
    assert _agents(runner, start) == ["agy"]
    (payload,) = _link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["effective_from_round"] == 2
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert len(_links(runner)) == 1
    approvals = {
        r.metadata.agent for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.state == "approved"
        and r.metadata.round_number >= 2
    }
    assert "Antigravity" in approvals


# --- pr-unchanged-board-noop ---------------------------------------------------------


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_unchanged_board_posts_nothing_and_adds_only_identity_fields(tmp_path, policy):
    """Row pr-unchanged-board-noop: legacy and identity-bearing history."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="blocking", blocking_items=["Fix it."])],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    before = _records(runner)
    assert before and all(r.metadata.pr_board_root is not None for r in before)
    root = before[0].metadata.pr_board_root
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved))
    runner.gemini_outputs.append(
        structured_pr_review(reviewer="Google Gemini", prior_item_dispositions=resolved)
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    after = _records(runner)
    assert all(r.metadata.pr_board_root == root for r in after)
    assert all(r.metadata.reviewer_board_change_digest is None for r in after)
    _qualify(runner, config)


def test_legacy_history_upgrades_without_any_record(tmp_path):
    """Row pr-unchanged-board-noop: a no-change legacy resume posts no link."""
    runner = _legacy_runner(
        [_legacy("reviewer", "Codex", 1, state="approved"),
         _legacy("reviewer", "Gemini", 1, state="blocking",
                 new_items=(_item(1, "Gemini"),))],
        codex_outputs=[structured_pr_review(prior_item_dispositions=_RESOLVED_1)],
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", prior_item_dispositions=_RESOLVED_1)],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    upgraded = [r for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded
    root = BoardIdentity.from_payload(upgraded[0].metadata.pr_board_root)
    assert root.names == ("Codex", "Gemini")
    # Plain legacy history records no models: the first upgraded record
    # initializes them from configuration, with no link of its own.
    assert all(seat.model_chain is None for seat in root.seats)
    assert upgraded[0].metadata.pr_board_binding_init
    assert all(
        seat.model_chain is not None
        for seat in BoardIdentity.from_payload(upgraded[0].metadata.pr_board_identity).seats
    )


# --- pr-policy-change-refused ----------------------------------------------------------


@pytest.mark.parametrize(
    "change",
    [
        {"pr_review_policy": "all-reviewers"},
        {"pr_review_broad_rules": ("docs/**",)},
        {"pr_review_policy": "all-reviewers", "reviewer": ("codex",)},
    ],
    ids=["policy", "broad-rules", "policy-and-board"],
)
def test_policy_or_broad_rule_change_stays_contract_drift(tmp_path, change):
    """Row pr-policy-change-refused: no link, no agent, no comment."""
    runner, _ = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    config = make_config(
        tmp_path, **{"reviewer": ("codex", "gemini"), "pr_review_policy": "selective-intermediate", **change}
    )
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == []


# --- pr-link-marker-agent-forgery / pr-qualification-forged-record -------------------------


def _forged_link_body(runner, config, **overrides):
    """A well-formed link for the current lineage, optionally corrupted."""
    from coding_review_agent_loop.board_change import build_board_change_candidate
    from coding_review_agent_loop.pr_board_state import resolve_pr_board_state

    state = resolve_pr_board_state(
        runner, config=config, pr_number=77, comments=_comments(runner),
        drift_error=lambda persisted, detail: AgentLoopError(detail),
    )
    candidate = build_board_change_candidate(
        state.lineage, configured=identity_from_config(config),
        records=_records(runner), pr_number=77,
    )
    payload = {**candidate.payload, **overrides}
    return str(render_board_change_link(payload))


def test_reviewer_text_naming_the_link_marker_is_neutralized(tmp_path):
    """Row pr-link-marker-agent-forgery: agent text cannot mint a link."""
    runner, config = _approved_two_reviewer_run(tmp_path)
    forged = _forged_link_body(runner, make_config(tmp_path, reviewer=("codex",)))
    marker = BOARD_CHANGE_LINK_RE.search(forged).group(0)
    # The reviewer's response embeds a complete, well-formed link record.
    runner.codex_outputs.append(
        structured_pr_review(summary=f"Board update {marker} applied.")
    )
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    try:
        run_pr_loop(runner, pr_number=77, config=config)
    except AgentLoopError:
        pass
    assert not any(BOARD_CHANGE_LINK_RE.search(c["body"]) for c in runner.pr_payload["comments"])
    assert _links(runner) == []


@pytest.mark.parametrize("corruption", [
    "foreign-author", "malformed", "broken-predecessor", "wrong-old-board",
    "empty-payload", "illegal-characters", "foreign-illegal-characters", "wrong-old-seat-binding",
])
def test_inconsistent_link_records_fail_closed_at_startup_and_qualification(tmp_path, corruption):
    """Rows pr-qualification-forged-record and pr-link-marker-agent-forgery."""
    runner, config = _approved_two_reviewer_run(tmp_path)
    smaller = make_config(tmp_path, reviewer=("codex",))
    if corruption == "foreign-author":
        _append(runner, _forged_link_body(runner, smaller), login="mallory", author_id=9)
    elif corruption == "malformed":
        body = _forged_link_body(runner, smaller)
        match = BOARD_CHANGE_LINK_RE.search(body)
        bad = base64.urlsafe_b64encode(b'{"kind": "reviewer-board-change"}').decode()
        _append(runner, body.replace(match.group("payload"), bad), author_id=_ACTOR_ID)
    elif corruption == "broken-predecessor":
        _append(runner, _forged_link_body(runner, smaller, predecessor="0" * 64), author_id=_ACTOR_ID)
    elif corruption == "wrong-old-board":
        wrong = identity_from_config(make_config(tmp_path, reviewer=("gemini",))).payload()
        _append(runner, _forged_link_body(runner, smaller, old_board=wrong), author_id=_ACTOR_ID)
    elif corruption == "wrong-old-seat-binding":
        forged_binding = {"version": 1, "seats": [
            {"id": "x", "backend": "codex", "model_chain": ["m"], "effort": None},
        ]}
        _append(runner, _forged_link_body(runner, smaller, old_seat_binding=forged_binding),
                author_id=_ACTOR_ID)
    else:
        # Record-shaped spans the strict grammar cannot match are still records.
        payload = {"empty-payload": "", "illegal-characters": "!!not*base64!!",
                   "foreign-illegal-characters": "!!not*base64!!"}[corruption]
        author = 9 if corruption.startswith("foreign") else _ACTOR_ID
        _append(runner, f"Board change.\n\n<!-- {BOARD_CHANGE_LINK_MARKER}: {payload} -->",
                login="mallory" if author == 9 else "operator", author_id=author)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent|not authored"):
        run_pr_loop(runner, pr_number=77, config=smaller)
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == []
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)


def test_unknown_change_digest_on_a_scheduler_record_fails_closed(tmp_path):
    """Row pr-qualification-forged-record: a record names an unknown link digest."""
    runner, config = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    record = _records(runner)[-1]
    forged = dataclasses.replace(
        record.metadata, round_number=2, reviewer_board_change_digest="e" * 64,
    )
    _append(runner, _attach_round_metadata("Forged.", forged), author_id=_ACTOR_ID)
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)
    with pytest.raises(AgentLoopError, match="unknown board-change or amendment digest"):
        run_pr_loop(runner, pr_number=77, config=config)


# --- pr-remove-with-blocker ------------------------------------------------------------


def _dispositions(reviewer, *item_ids, disposition="resolved"):
    from coding_review_agent_loop.protocol import ReviewItemDisposition

    return tuple(
        ReviewItemDisposition(item_id=item_id, reviewer=reviewer, disposition=disposition)
        for item_id in item_ids
    )


def _gemini_blocker_history(*, evidence=()):
    """Round 2 on head 6367c48dd193d56ea7b0baad25b19455e529f5ee: item-1 resolved by both, Gemini alone raises item-2."""
    item1 = _item(1, "Gemini")
    item2 = _item(2, "Gemini", round_number=2)
    return [
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item1,)),
        _legacy("coder", "Claude", 2, prior_items=(item1,)),
        _legacy("reviewer", "Codex", 2, state="approved", prior_items=(item1,),
                dispositions=_dispositions("Codex", "item-1")),
        _legacy("reviewer", "Gemini", 2, state="blocking", prior_items=(item1,),
                dispositions=_dispositions("Gemini", "item-1"), new_items=(item2,),
                evidence_requests=tuple(evidence)),
    ]


def _crash_after_link(monkeypatch):
    real_post = pr_loop_support_module.post_pr_comment

    def post(*args, **kwargs):
        real_post(*args, **kwargs)
        if str(kwargs["body"]).startswith(BOARD_CHANGE_LINK_HEADING):
            raise KeyboardInterrupt

    patch_attr_in(monkeypatch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", post)


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
def test_removed_reviewer_blocker_is_carried_reassigned_and_settled_items_stay_settled(
    tmp_path, monkeypatch, activation
):
    """Row pr-remove-with-blocker: Gemini is the sole publisher of item-2."""
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
        assert len(_links(runner)) == 1 and _agents(runner) == []
    # Neither remaining reviewer resolves item-2 yet: it blocks completion.
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(structured_pr_review(
        reviewer="Antigravity", state="blocking",
        prior_item_dispositions=[{
            "item_id": "item-2", "disposition": "blocking",
            "note": "The worker still retries forever in src/worker.py; bound it and add a test.",
        }],
    ))
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["effective_from_round"]) == (2, 3)
    round_three = [r.metadata for r in _records(runner) if r.metadata.round_number == 3]
    reviews = [m for m in round_three if m.role == "reviewer"]
    assert {m.agent for m in reviews} == {"Codex", "Antigravity"}
    # item-1 was resolved in the anchor round and stays settled; item-2 is
    # carried, and both remaining reviewers had to disposition it.
    assert all([item.item_id for item in m.prior_items] == ["item-2"] for m in reviews)
    assert all(
        {d.item_id for d in m.dispositions} == {"item-2"} for m in reviews
    )
    assert not any(m.agent == "Gemini" for m in round_three)


def test_removed_reviewer_evidence_request_is_rebuilt_and_blocks(tmp_path):
    """Row pr-remove-with-blocker: an outgoing reviewer's exact-head evidence request."""
    from coding_review_agent_loop.board_change import carry_over_anchor_round

    records = _extract_round_metadata_records(
        [SimpleNamespace(body=body) for body in _gemini_blocker_history(
            evidence=("Attach the production deploy log.",)
        )],
        flow="pr",
    )
    anchor = [r for r in records if r.metadata.round_number == 2]
    reviews = [r for r in anchor if r.metadata.role == "reviewer"]
    carried = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=(),
        next_item_number=3,
    )
    ids = [item.item_id for item in carried.items]
    assert ids[:1] == ["item-2"] and "item-1" not in ids
    evidence = [item for item in carried.items if item.item_id not in {"item-2"}]
    assert len(evidence) == 1 and evidence[0].item_id == "item-3"
    assert "deploy log" in evidence[0].text
    # A request already cleared at this head is not recreated...
    cleared = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=tuple(
            (item.obligation_identity, "6367c48dd193d56ea7b0baad25b19455e529f5ee") for item in evidence if item.obligation_identity
        ),
        next_item_number=3,
    )
    assert [item.item_id for item in cleared.items] == ["item-2"]
    # ...but at a different head it is a new obligation.
    moved = carry_over_anchor_round(
        prior_items=anchor[0].metadata.prior_items,
        completed_reviews=reviews,
        settled_new_items=(),
        records=records,
        round_number=2,
        reconciliation_mode="aggregate",
        evidence_clearances=tuple(
            (item.obligation_identity, "f" * 40) for item in evidence if item.obligation_identity
        ),
        next_item_number=3,
    )
    assert len(moved.items) == 2


# --- pr-legacy-partial-round-noop --------------------------------------------------------


@pytest.mark.parametrize("missing", ["crash", "unavailable-summary", "coder-only"])
def test_unchanged_plain_board_with_missing_reviewer_outcome_posts_nothing(tmp_path, missing):
    """Row pr-legacy-partial-round-noop: membership is never inferred from completeness."""
    history = [
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
        _legacy("coder", "Claude", 2),
    ]
    if missing != "coder-only":
        history.append(_legacy("reviewer", "Codex", 2, state="approved"))
    if missing == "unavailable-summary":
        history.append(_legacy("summary", "Orchestrator", 2, phase="reconciliation"))
    runner = _legacy_runner(
        history,
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    # The round is never advanced for an unchanged board.  A reconciled round
    # is closed, though (#1553): the missing reviewer's turn opens round 3.
    expected_round = 3 if missing == "unavailable-summary" else 2
    new = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert new and all(m.round_number == expected_round for m in new)


def test_historical_poster_missing_from_the_board_is_one_removal_link(tmp_path):
    """Row pr-legacy-partial-round-noop: B posted in an earlier legacy round."""
    item = _item(1, "Gemini")
    runner = _legacy_runner(
        [
            _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
            _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
            _legacy("coder", "Claude", 2, prior_items=(item,)),
        ],
        codex_outputs=[structured_pr_review(prior_item_dispositions=_RESOLVED_1)],
    )
    config = make_config(tmp_path, reviewer=("codex",))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    (payload,) = _link_payloads(runner)
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    # Gemini's open item was carried and reassigned to the remaining reviewer.
    (review,) = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.pr_board_root is not None
    ]
    assert [item.item_id for item in review.prior_items] == ["item-1"]
    assert [d.item_id for d in review.dispositions] == ["item-1"]


# --- pr-floor-unsigned-vs-signed ----------------------------------------------------------


def _amendment_from(message):
    start = message.index("Reviewer board amendment:")
    return message[start:].replace(
        "<why the removed reviewer cannot be reached>", "Gemini quota exhausted."
    )


def _two_reviewer_history(policy):
    """Round 1: Codex approved, Gemini raised item-1; the coder answered in round 2."""
    if policy == "legacy-plain":
        item = _item(1, "Gemini")
        return _legacy_runner([
            _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
            _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
            _legacy("coder", "Claude", 2, prior_items=(item,)),
        ])
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(
            reviewer="Google Gemini", state="blocking", blocking_items=["Bound the retry."]
        )],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=make_config(
            tmp_path_placeholder(), reviewer=("codex", "gemini"), pr_review_policy=policy,
        ))
    return runner


_TMP = {}


def tmp_path_placeholder():
    return _TMP["path"]


@pytest.mark.parametrize("policy", ["legacy-plain", "selective-intermediate"])
@pytest.mark.parametrize(
    ("floor", "refusal"),
    [
        ({"min_reviewers": 2}, "1 reviewer\\(s\\) is below the floor of 2"),
        ({"min_distinct_providers": 2}, "1 distinct provider\\(s\\) is below the floor of 2"),
    ],
    ids=["reviewer-floor", "provider-floor"],
)
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_below_floor_shrink_needs_a_signed_amendment(tmp_path, policy, floor, refusal, flags):
    """Row pr-floor-unsigned-vs-signed, both flag forms and both floor measures."""
    _TMP["path"] = tmp_path
    runner = _two_reviewer_history(policy)
    config_policy = "all-reviewers" if policy == "legacy-plain" else policy
    reduced = make_config(tmp_path, reviewer=("codex",), pr_review_policy=config_policy, **floor)
    original = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy=config_policy, **floor
    )
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match=refusal) as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    # Refused after read-only lineage validation: nothing posted, no agent.
    assert len(runner.pr_payload["comments"]) == comments
    assert _agents(runner, start) == [] and _links(runner) == []
    _append(runner, _amendment_from(str(refused.value)))
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    config = reduced if flags == "reduced" else original
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    assert "gemini" not in _agents(runner, start)
    # The qualification gate accepts the signed board from fresh comments; the
    # run's own config is the effective (rebound) board whichever flags it got.
    _qualify(runner, reduced)


# --- pr-floor-candidate-link --------------------------------------------------------------


def _signed_three_to_two(tmp_path, **floor):
    """A selective PR approved by three reviewers, then a signed 3 -> 2 reduction."""
    runner, _ = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )
    reduced = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
        min_reviewers=3, **floor,
    )
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _append(runner, _amendment_from(str(refused.value)).replace("Gemini quota", "Antigravity quota"))
    runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    return runner, reduced


def test_signed_reduction_stays_authorized_through_later_links_but_never_further(tmp_path):
    """Row pr-floor-candidate-link: frozen signed authorization, links never enlarge it."""
    runner, reduced = _signed_three_to_two(tmp_path)
    # A binding-only change of a remaining seat posts one link and stays admissible.
    rebound = dataclasses.replace(reduced, reviewer_codex_model="gpt-alternate")
    runner.pr_payload["headRefOid"] = "59e9fef4edb549af995ec1617c78fd9f54405f29"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (first,) = _link_payloads(runner)
    assert first["rebound"] == ["Codex"] and not first["added"] and not first["removed"]
    _qualify(runner, rebound)
    # An above-floor membership change (re-adding the signed-removed seat) is a link too.
    larger = make_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate", min_reviewers=3,
        reviewer_codex_model="gpt-alternate",
    )
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert [p["added"] for p in _link_payloads(runner)] == [[], ["Antigravity"]]
    # A further unsigned reduction below the signed board is refused before any post.
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    smaller = make_config(
        tmp_path, reviewer=("codex",), pr_review_policy="selective-intermediate",
        min_reviewers=3, reviewer_codex_model="gpt-alternate",
    )
    with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
        run_pr_loop(runner, pr_number=77, config=smaller)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, start) == []


def test_link_never_creates_a_provider_exception(tmp_path):
    """Row pr-floor-candidate-link: a provider-reducing swap after a signed 3 -> 2."""
    runner, reduced = _signed_three_to_two(tmp_path, min_distinct_providers=2)
    comments = len(runner.pr_payload["comments"])
    # Gemini -> a second Codex-backed seat: one provider, and the signed chain
    # never lowered the provider count, so no authorization applies.
    seat = _seat(tmp_path, "second", "codex", "gpt-second")
    swapped = make_config(
        tmp_path, reviewer=("codex", seat), reviewer_seats=(seat,),
        pr_review_policy="selective-intermediate", min_reviewers=3, min_distinct_providers=2,
        reviewer_codex_model="gpt-main",
    )
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        run_pr_loop(runner, pr_number=77, config=swapped)
    assert len(runner.pr_payload["comments"]) == comments


# --- resolver-level helpers -----------------------------------------------------------


def _identity(*seats, policy="all-reviewers", primary=None, models=None):
    from coding_review_agent_loop.review_scheduling import normalize_broad_rules

    models = models or {}
    return BoardIdentity(
        policy=policy, primary=primary, broad_rules=normalize_broad_rules(None),
        seats=tuple(
            BoardSeat(seat_id, backend, (models.get(seat_id, f"{seat_id}-model"),), None)
            for seat_id, backend in seats
        ),
    )


def _record_body(round_number, root, identity, *, change=None, amendment=None,
                 agent=None, binding="identity", **extra):
    seat_binding = identity.seat_binding() if binding == "identity" else binding
    return _attach_round_metadata(
        structured_pr_review(reviewer=agent or identity.names[0]),
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent=agent or identity.names[0],
            round_number=round_number, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", state="approved",
            pr_board_root=root.payload(), pr_board_identity=identity.payload(),
            reviewer_board_change_digest=change, reviewer_board_amendment_digest=amendment,
            seat_binding=seat_binding, **extra,
        ),
    )


def _link_body(root, predecessor, old, new, *, anchor, advance, inits=None, old_base=None):
    """``old_base`` is the predecessor segment's base board (its seat binding)."""
    payload = board_change_link_payload(
        pr_number=77, root=root, predecessor=predecessor, old_board=old, new_board=new,
        old_seat_binding=(old_base or old).seat_binding(), binding_init=inits or {},
        anchor_round=anchor, advance=advance,
    )
    return str(render_board_change_link(payload)), board_change_link_digest(payload)


def _signed_body(original, removed=(), round_number=2, *, restored=(), policy="all-reviewers"):
    return format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=77, original_required_reviewers=original,
        policy=policy, primary_reviewer=None, removed_reviewers=removed,
        restored_reviewers=restored, effective_from_round=round_number,
        reason=None if restored and not removed else "seat-unavailable",
        rationale="Seat quota changed.",
    )


def _amendment_digest(body):
    from coding_review_agent_loop.board_amendment import parse_reviewer_board_amendment_records

    (record,), _ = parse_reviewer_board_amendment_records(body, comment_locator="c")
    return record.digest


def _resolve(bodies, *, configured=None, authors=None):
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments

    comments = [
        SimpleNamespace(body=body, author_id=(authors or {}).get(index, _ACTOR_ID))
        for index, body in enumerate(bodies)
    ]
    records = _extract_round_metadata_records(comments, flow="pr")
    amendments = collect_reviewer_board_amendments(comments, flow="pr", pr_number=77)
    return resolve_board_lineage(
        records, comments, amendments=amendments,
        configured=configured or _identity(("a", "codex")), pr_number=77,
        trusted_actor=("orchestrator", _ACTOR_ID),
        contract_from_metadata=_scheduler_contract_from_metadata,
        drift_error=lambda persisted, detail: AgentLoopError(f"drift: {detail}"),
    )


def test_frozen_signed_bindings_come_from_the_predecessor_link():
    """Row pr-floor-candidate-link: a signed removal after a backend-rebinding link.

    Root a/codex, b/claude, c/antigravity; a link rebinds b to codex; a signed
    removal of c then leaves a/codex, b/codex, which lowered the provider count
    from the link's board, so it authorizes one provider -- identically while
    pending and after activation.
    """
    root = _identity(("a", "codex"), ("b", "claude"), ("c", "antigravity"))
    rebound = _identity(("a", "codex"), ("b", "codex"), ("c", "antigravity"))
    link, link_digest = _link_body(root, root.digest, root, rebound, anchor=1, advance=True)
    signed = _signed_body(("a", "b", "c"), removed=("c",), round_number=3)
    history = [
        _record_body(1, root, root),
        link,
        _record_body(2, root, rebound, change=link_digest),
        signed,
    ]
    pending = _resolve(history)
    activated = _resolve([
        *history,
        _record_body(3, root, rebound.with_seats(rebound.seats[:2]),
                     amendment=_amendment_digest(signed), binding=rebound.seat_binding()),
    ])
    assert pending.pending_amendments and not activated.pending_amendments
    for lineage in (pending, activated):
        (authorization,) = lineage_floor_authorizations(lineage)
        assert [seat.backend for seat in authorization.seats] == ["codex", "codex"]
        assert authorization.lowered_providers and authorization.lowered_reviewers


# --- pr-link-crash-recovery ------------------------------------------------------------------


def test_pending_link_recovery_re_enters_the_advanced_round_exactly_once(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: a partially published reviewer round."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    new_rounds = {r.metadata.round_number for r in _records(runner) if r.metadata.pr_board_root}
    assert new_rounds == {3}
    # Another rerun neither reposts nor advances again.
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    assert {r.metadata.round_number for r in _records(runner) if r.metadata.pr_board_root} == {3}


def test_coder_only_anchor_round_advances_once(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: the anchor round holds only a coder record."""
    item = _item(1, "Gemini")
    runner = _legacy_runner([
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="blocking", new_items=(item,)),
        _legacy("coder", "Claude", 2, prior_items=(item,)),
    ])
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["advance"], payload["effective_from_round"]) == (2, True, 3)
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=_RESOLVED_1)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    reviews = [r.metadata for r in _records(runner) if r.metadata.pr_board_root]
    assert {m.round_number for m in reviews} == {3}
    # The anchor round's coder output is the artifact the new board reviews.
    assert all(m.subject == "6367c48dd193d56ea7b0baad25b19455e529f5ee" for m in reviews)


def test_prelaunch_only_anchor_round_keeps_its_round_and_ledger(tmp_path, monkeypatch):
    """Row pr-link-crash-recovery: only a scheduler-prelaunch record exists at R > 1."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="blocking", blocking_items=["Bound the retry."])],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
    )
    real_post = pr_loop_support_module.post_pr_comment
    prelaunches = []

    def interrupt_round_two_prelaunch(*args, **kwargs):
        real_post(*args, **kwargs)
        if "PR review scheduling audit:" in str(kwargs["body"]):
            prelaunches.append(True)
            if len(prelaunches) == 2:
                raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", interrupt_round_two_prelaunch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    highest = board_anchor(_records(runner))
    assert highest.round_number == 2
    assert [r.metadata.phase for r in _records(runner) if r.metadata.round_number == 2][-1] == (
        "scheduler-prelaunch"
    )
    assert [item.item_id for item in highest.ledger] == ["item-1"]
    swapped = make_config(
        tmp_path, reviewer=("codex", "antigravity"), pr_review_policy="selective-intermediate",
    )
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=_RESOLVED_1)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert (payload["anchor_round"], payload["effective_from_round"]) == (2, 3)
    reviews = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.round_number == 3
    ]
    assert {m.agent for m in reviews} >= {"Antigravity"}
    assert all([item.item_id for item in m.prior_items] == ["item-1"] for m in reviews)


def test_superseding_a_pending_link_copies_its_anchor_and_activates_only_the_last(
    tmp_path, monkeypatch
):
    """Row pr-link-crash-recovery: a second change while the first link is pending."""
    runner = _legacy_runner(_gemini_blocker_history())
    first = make_config(tmp_path, reviewer=("codex", "antigravity"))
    second = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex")
    for config in (first, second):
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=config)
    one, two = _link_payloads(runner)
    assert two["predecessor"] == board_change_link_digest(one)
    for key in ("anchor_round", "advance", "effective_from_round"):
        assert two[key] == one[key]
    assert two["old_board"] == one["new_board"]
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.claude_outputs.append(
        structured_pr_review(reviewer="Anthropic Claude", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    assert len(_links(runner)) == 2
    governed = {
        r.metadata.reviewer_board_change_digest for r in _records(runner) if r.metadata.pr_board_root
    }
    assert governed == {board_change_link_digest(two)}
    assert "agy" not in _agents(runner)
    _qualify(runner, second)


def test_superseding_link_with_mismatched_rounds_fails_closed():
    """Row pr-link-crash-recovery: a superseding link must copy anchor and advance."""
    root = _identity(("a", "codex"), ("b", "claude"))
    first_board = _identity(("a", "codex"), ("c", "gemini"))
    second_board = _identity(("a", "codex"), ("d", "antigravity"))
    link_one, digest_one = _link_body(root, root.digest, root, first_board, anchor=1, advance=True)
    link_two, _ = _link_body(root, digest_one, first_board, second_board, anchor=2, advance=False)
    with pytest.raises(AgentLoopError, match="does not copy its anchor round"):
        _resolve([_record_body(1, root, root), link_one, link_two])


@pytest.mark.parametrize("state", ["consistent-empty", "records-after-link", "wrong-anchor"])
def test_pending_link_activation_follows_durable_history(tmp_path, state):
    """Row pr-link-crash-recovery: consistent and inconsistent pending links."""
    root = _identity(("a", "codex"), ("b", "claude"))
    new = _identity(("a", "codex"), ("c", "gemini"))
    if state == "consistent-empty":
        link, _ = _link_body(root, root.digest, root, new, anchor=1, advance=False)
        lineage = _resolve([link])
        assert [e.effective_from_round for e in lineage.pending_links] == [1]
        return
    if state == "records-after-link":
        link, _ = _link_body(root, root.digest, root, new, anchor=1, advance=True)
        bodies = [_record_body(1, root, root), link, _record_body(1, root, root)]
    else:
        link, _ = _link_body(root, root.digest, root, new, anchor=3, advance=True)
        bodies = [_record_body(1, root, root), link]
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        _resolve(bodies)


# --- pr-primary-change -------------------------------------------------------------------


@pytest.mark.parametrize("change", ["new-primary", "primary-model", "primary-effort"])
def test_primary_change_restarts_the_primary_phase_and_carries_secondary_obligations(
    tmp_path, monkeypatch, change
):
    """Row pr-primary-change: a new primary, or a model or effort change on the
    same primary seat, restarts the primary phase; the panel's open item
    survives it and is dispositioned by its owner when the panel reopens."""
    from pr_loop_test_helpers import _audit_phases, _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)
    assert _agents(runner)[:3] == ["codex", "gemini", "agy"]
    overrides = {
        "new-primary": {"primary_reviewer": "antigravity"},
        "primary-model": {"reviewer_codex_model": "gpt-primary-next"},
        "primary-effort": {"reviewer_codex_reasoning_effort": "high"},
    }[change]
    changed = _staged_config(tmp_path, pre_review_tests=False, **overrides)
    primary, primary_agent = (
        ("Antigravity", "agy") if change == "new-primary" else ("Codex", "codex")
    )
    # A non-owner's disposition is a note under owner-scoped reconciliation:
    # item-1 stays pending on its owner, Gemini.
    carried = [{"item_id": "item-1", "disposition": "resolved",
                "note": "Looks addressed to me; Gemini owns the finding."}]
    # The restarted primary alone first, then the reopened panel at the same
    # head, where Gemini dispositions its own carried finding.
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=carried))
    runner.gemini_outputs.append(_staged_review(
        reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=carried))
    phases_before = len(_audit_phases(runner))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["new_board"]["primary"] == primary
    if change == "new-primary":
        assert payload["old_board"]["primary"] == "Codex"
    else:
        assert payload["rebound"] == ["Codex"]
    # The restarted primary runs alone first -- no pre-panel stop for the
    # carried secondary obligation -- then the panel reopens.
    agents = _agents(runner, start)
    assert agents[0] == primary_agent and "gemini" in agents[1:] and "claude" not in agents
    phases = [phase for phase, _head in _audit_phases(runner)][phases_before:]
    assert phases[0] == "primary" and "secondary-audit" in phases
    restarted = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == primary
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
    ]
    # item-1 was still pending on Gemini when the restarted primary reviewed.
    assert restarted and [item.item_id for item in restarted[0].prior_items] == ["item-1"]
    assert not any("pre-panel safety" in c for c in runner.comments)
    _qualify(runner, changed)


def test_only_carried_pre_epoch_obligations_are_exempt_from_the_pre_panel_stop():
    """Row pr-primary-change: a secondary-owned item created after the epoch still stops."""
    from coding_review_agent_loop.review_scheduling import (
        PrePanelSafetyError,
        ReviewObligation,
        SchedulerSnapshot,
        TransitionClassification,
        make_contract,
        select_reviewers,
    )

    contract = make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Antigravity")
    obligation = ReviewObligation(
        item_id="item-1", status="blocking", scope=("src/worker.py",),
        resolution_owners=("Gemini",), pending_owners=("Gemini",),
    )
    narrow = TransitionClassification("narrow", "scoped fix")

    def decide(carried):
        return select_reviewers(
            SchedulerSnapshot(
                previous_sha="a", current_sha="b", contract=contract,
                obligations=(obligation,), carried_obligations=carried,
            ),
            narrow,
        )

    assert decide(("item-1",)).selected_reviewers == ("Antigravity",)
    with pytest.raises(PrePanelSafetyError, match="item-1"):
        decide(())


# --- pr-resume-chain ---------------------------------------------------------------------


def test_link_then_signed_amendment_then_link_resolves_one_ordered_chain(tmp_path):
    """Row pr-resume-chain: link -> signed amendment of the link's board -> link."""
    runner, _ = _approved_two_reviewer_run(tmp_path, pr_review_policy="selective-intermediate")
    larger = make_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate", min_reviewers=3,
    )
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    # The signed amendment after the link amends the link's new board.
    reduced = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate",
        min_reviewers=3,
    )
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    signed = _amendment_from(str(refused.value))
    assert '"Antigravity"' in signed.split('"original_required_reviewers"')[1].split("]")[0]
    _append(runner, signed)
    runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    # Original flags of the link's board take the #1133 rebind.
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert len(_links(runner)) == 1
    def lineage(config=larger):
        return _board_state(runner, config).lineage

    first = lineage()
    assert [element.kind for element in first.elements] == ["link", "amendment"]
    assert first.current_board == ("Codex", "Gemini")
    # A rerun derives the identical lineage and posts nothing.
    assert run_pr_loop(runner, pr_number=77, config=larger) == 0
    assert lineage().elements == first.elements and len(_links(runner)) == 1
    # A further differing board adds exactly one link after the amendment.
    swapped = make_config(
        tmp_path, reviewer=("codex", "claude"), coder="codex",
        pr_review_policy="selective-intermediate", min_reviewers=3,
    )
    runner.claude_outputs.append(structured_pr_review(reviewer="Anthropic Claude"))
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 2
    after = lineage(swapped)
    assert [element.kind for element in after.elements] == ["link", "amendment", "link"]
    _qualify(runner, swapped)


def test_signed_restoration_after_a_link_only_restores_seats_signed_off_since_it():
    """Row pr-resume-chain: restorations are bounded by the latest link."""
    root = _identity(("a", "codex"), ("b", "claude"), ("c", "antigravity"))
    abd = _identity(("a", "codex"), ("b", "claude"), ("d", "gemini"))
    ab = abd.with_seats(abd.seats[:2])
    # c was removed by a signed amendment before the link: a restoration after
    # the link cannot bring it back.
    removal = _signed_body(("a", "b", "c"), removed=("c",), round_number=2)
    link, _ = _link_body(root, _amendment_digest(removal), root.with_seats(root.seats[:2]), abd,
                         anchor=2, advance=True, old_base=root)
    restore_c = _signed_body(("a", "b", "d"), restored=("c",), round_number=4)
    history = [
        _record_body(1, root, root),
        removal,
        _record_body(2, root, root.with_seats(root.seats[:2]), amendment=_amendment_digest(removal),
                     binding=root.seat_binding()),
        link,
    ]
    with pytest.raises(AgentLoopError, match="restored reviewer|not on the persisted board|already"):
        _resolve([*history, restore_c])
    # d was added by the link and signed off after it: its restoration is valid.
    remove_d = _signed_body(("a", "b", "d"), removed=("d",), round_number=4)
    restore_d = _signed_body(("a", "b"), restored=("d",), round_number=5)
    lineage = _resolve([*history, remove_d, restore_d])
    assert lineage.current_board == ("a", "b", "d")
    assert lineage.restoration_rounds == {"d": 5}
    del ab


# --- pr-legacy-root-anchor ----------------------------------------------------------------


def _strip_board_identity(runner):
    """Simulate pre-upgrade history: drop every board identity field."""
    from pr_loop_test_helpers import _rewrite_pr_metadata

    keys = ("pr_board_root", "pr_board_identity", "pr_board_binding_init",
            "reviewer_board_change_digest")

    def strip(payload):
        if not any(key in payload for key in keys):
            return None
        return {key: value for key, value in payload.items() if key not in keys}

    _rewrite_pr_metadata(runner, strip)


def test_legacy_signed_chain_keeps_its_original_root_after_upgrade(tmp_path):
    """Row pr-legacy-root-anchor: legacy C0 -> signed C1, then upgraded records."""
    runner, reduced = _signed_three_to_two(tmp_path)
    _strip_board_identity(runner)
    assert all(r.metadata.pr_board_root is None for r in _records(runner))
    runner.pr_payload["headRefOid"] = "59e9fef4edb549af995ec1617c78fd9f54405f29"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded
    roots = {BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded}
    assert roots == {("Codex", "Gemini", "Antigravity")}
    assert all(
        BoardIdentity.from_payload(m.pr_board_identity).names == ("Codex", "Gemini")
        for m in upgraded
    )
    assert all(m.reviewer_board_amendment_digest for m in upgraded)
    assert _links(runner) == []
    _qualify(runner, reduced)
    runner.pr_payload["headRefOid"] = "4a0cde71aee7158542d013fc0c9f5acfc735c612"
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0


def test_legacy_link_root_and_unknown_bindings_initialize_once(tmp_path, monkeypatch):
    """Row pr-legacy-root-anchor: A -> link B keeps root A; later model change is one link."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    # The link carries the derived legacy root: plain history records no board,
    # so the root is the configured board plus every historical poster, with
    # membership-only bindings.  Antigravity is adopted; only Gemini is removed.
    root = BoardIdentity.from_payload(payload["root"])
    assert root.names == ("Codex", "Antigravity", "Gemini")
    assert all(s.model_chain is None for s in root.seats)
    assert (payload["added"], payload["removed"]) == ([], ["Gemini"])
    # The seats that stay are initialized by the link, with no link of their own.
    assert set(payload["binding_init"]) == {"Codex", "Antigravity"}
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved_2))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=resolved_2)
    )
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert {BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded} == {
        ("Codex", "Antigravity", "Gemini")
    }
    # A later model change of an initialized seat is exactly one link and
    # needs Codex's fresh approval.
    moved = dataclasses.replace(swapped, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    assert len(_links(runner)) == 2
    assert _link_payloads(runner)[1]["rebound"] == ["Codex"]
    assert _agents(runner, start) == ["codex"]
    _qualify(runner, moved)


def test_unchanged_upgrade_then_model_change_posts_one_link(tmp_path):
    """Row pr-legacy-root-anchor: an unchanged plain upgrade, then a model change."""
    runner = _legacy_runner(
        [_legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
         _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
         _legacy("coder", "Claude", 2)],
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    first = next(r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None)
    assert set(first.pr_board_binding_init) == {"Codex", "Gemini"}
    # A resume after the upgrade reads the initialization: no link.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _links(runner) == []
    moved = dataclasses.replace(config, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"] and payload["binding_init"] == {}
    assert _agents(runner, start) == ["codex"]


# --- pr-all-reviewers-board-change ------------------------------------------------------------


def test_all_reviewers_plain_to_named_and_back(tmp_path):
    """Row pr-all-reviewers-board-change: plain -> named -> plain, one link each."""
    runner, plain = _approved_two_reviewer_run(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    named = _named(tmp_path, "codex", flash)
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(
        (structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0)
    )
    assert run_pr_loop(runner, pr_number=77, config=named) == 0
    (to_named,) = _link_payloads(runner)
    assert (to_named["added"], to_named["removed"]) == (["flash"], ["Gemini"])
    assert to_named["old_seat_binding"] is None and to_named["new_seat_binding"]["version"] == 1
    # Records under the named board carry its seat binding.
    named_records = [
        r.metadata for r in _records(runner)
        if r.metadata.reviewer_board_change_digest == board_change_link_digest(to_named)
    ]
    assert named_records and all(m.seat_binding == to_named["new_seat_binding"] for m in named_records)
    _qualify(runner, named)
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    assert run_pr_loop(runner, pr_number=77, config=plain) == 0
    back = _link_payloads(runner)[1]
    assert (back["added"], back["removed"]) == (["Gemini"], ["flash"])
    assert back["new_seat_binding"] is None
    _qualify(runner, plain)
    assert run_pr_loop(runner, pr_number=77, config=plain) == 0
    assert len(_links(runner)) == 2


def test_all_reviewers_same_seat_backend_change(tmp_path):
    """Row pr-all-reviewers-board-change: a named seat moves to another backend."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        antigravity_outputs=[(structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0)],
    )
    config = _named(tmp_path, "codex", flash)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    moved = _seat(tmp_path, "flash", "gemini", "gemini-pro")
    rebound = _named(tmp_path, "codex", moved)
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="flash (Google Gemini: gemini-pro)"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=rebound) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["flash"]
    assert "agy" not in _agents(runner, start) and "gemini" in _agents(runner, start)
    _qualify(runner, rebound)


@pytest.mark.parametrize(
    "defect", ["identity-without-root", "foreign-root", "wrong-identity", "mixed-bound-unbound"],
)
def test_malformed_board_history_fails_closed(defect):
    """Row pr-all-reviewers-board-change: malformed board history is never accepted.

    Row mixed-legacy-refused (#1365): unanchored history whose unbound coder
    record is followed by a bound reviewer record is never read as a named board.
    """
    root = _identity(("a", "codex"), ("b", "claude"))
    other = _identity(("a", "codex"), ("c", "gemini"))
    if defect == "mixed-bound-unbound":
        bound_review = _attach_round_metadata(
            structured_pr_review(reviewer="a"),
            PostedRoundMetadata(
                flow="pr", role="reviewer", agent="a", round_number=1, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
                state="approved", seat_binding=root.seat_binding(),
            ),
        )
        with pytest.raises(AgentLoopError, match="PR history mixes bound and unbound round records"):
            _resolve([_legacy("coder", "Claude", 1), bound_review], configured=root)
        return
    if defect == "identity-without-root":
        bad = _attach_round_metadata(
            "x", PostedRoundMetadata(
                flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
                reviewer_board_change_digest="a" * 64,
            ),
        )
    elif defect == "foreign-root":
        bad = _record_body(2, other, other)
    else:
        bad = _record_body(2, root, other)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        _resolve([_record_body(1, root, root), bad])


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_removal_with_carried_approvals_qualifies_and_reruns_cleanly(tmp_path, policy):
    """Rows pr-all-reviewers-board-change and pr-link-crash-recovery.

    Removing a reviewer whose peers already approved the current head may need
    no new reviewer turn; the link is then the board the run reviewed with.
    """
    runner, _ = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"), pr_review_policy=policy,
    )
    smaller = make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy)
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=smaller) == 0
    assert _agents(runner, start) == []
    (payload,) = _link_payloads(runner)
    assert payload["removed"] == ["Antigravity"]
    _qualify(runner, smaller)
    assert run_pr_loop(runner, pr_number=77, config=smaller) == 0
    assert len(_links(runner)) == 1 and _agents(runner, start) == []


def test_signed_amendment_posted_during_the_run_blocks_qualification(tmp_path):
    """Row pr-qualification-forged-record: the gate re-reads the lineage from fresh comments."""
    runner, config = _approved_two_reviewer_run(
        tmp_path, reviewers=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
    )
    _append(runner, _signed_body(
        ("Codex", "Gemini", "Antigravity"), removed=("Antigravity",), round_number=2,
        policy="selective-intermediate",
    ).replace('"seat-unavailable"', '"backend-unavailable"'))
    with pytest.raises(AgentLoopError, match="No qualification or merge is permitted"):
        _qualify(runner, config)


# --- carried-finding provenance (D4b) ---------------------------------------------------


def _review_record(index, agent, new_items, *, round_number=2):
    return SimpleNamespace(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role="reviewer", agent=agent, round_number=round_number,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", state="blocking", new_items=tuple(new_items),
        ),
    )


@pytest.mark.parametrize("case", [
    "unknown-owner", "other-seat-owner", "record-collision", "ledger-collision",
])
def test_carry_over_refuses_unverifiable_or_conflicting_findings(case):
    """Row pr-remove-with-blocker: the publisher must own its findings, and a
    repeated item ID must carry identical content wherever it repeats."""
    from coding_review_agent_loop.board_change import carry_over_anchor_round

    prior = ()
    if case == "unknown-owner":
        reviews = [_review_record(1, "Gemini", [_item(2, "Somebody Else")])]
    elif case == "other-seat-owner":
        reviews = [_review_record(1, "Gemini", [_item(2, "Codex")])]
    elif case == "record-collision":
        first = _item(2, "Gemini")
        reviews = [
            _review_record(1, "Gemini", [first]),
            _review_record(2, "Gemini", [dataclasses.replace(first, text="another claim")]),
        ]
    else:
        prior = (_item(2, "Gemini"),)
        reviews = [_review_record(1, "Gemini", [dataclasses.replace(prior[0], text="changed")])]
    with pytest.raises(AgentLoopError, match="cannot verify the owner|conflicting content"):
        carry_over_anchor_round(
            prior_items=prior, completed_reviews=reviews, settled_new_items=(),
            records=reviews, round_number=2, reconciliation_mode="aggregate",
            evidence_clearances=(), next_item_number=3,
        )


# --- signed original board as the legacy root (D1a) ---------------------------------------


@pytest.mark.parametrize("shape", ["coder-only", "never-published", "non-tail"])
@pytest.mark.parametrize("floor", ["reviewers", "providers"])
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_signed_original_board_is_the_legacy_plain_root(tmp_path, shape, floor, flags):
    """Rows pr-floor-unsigned-vs-signed and pr-legacy-root-anchor.

    Plain legacy history with signed amendments takes the earliest signed
    original board as its root, so a reduction of a never-published seat or a
    non-tail seat resolves in both flag forms and for both floor measures.
    """
    if shape == "non-tail":
        board = ("Codex", "Gemini", "Antigravity")
        history = [_legacy("reviewer", name, 1, subject="abc000", state="approved") for name in board]
        removed, kept = ("Gemini",), ("codex", "antigravity")
    else:
        board = ("Codex", "Gemini")
        history = (
            [] if shape == "coder-only"
            else [_legacy("reviewer", "Codex", 1, subject="abc000", state="approved")]
        )
        removed, kept = ("Gemini",), ("codex",)
    history.append(_legacy("coder", "Claude", 2 if history else 1))
    reentry = 2 if len(history) > 1 else 1
    runner = _legacy_runner(history)
    minimum = len(board)
    floor_kwargs = (
        {"min_reviewers": minimum} if floor == "reviewers" else {"min_distinct_providers": minimum}
    )
    plain = {"Codex": "codex", "Gemini": "gemini", "Antigravity": "antigravity"}
    original = make_config(tmp_path, reviewer=tuple(plain[name] for name in board), **floor_kwargs)
    reduced = make_config(tmp_path, reviewer=kept, **floor_kwargs)
    comments = len(runner.pr_payload["comments"])
    with pytest.raises(AgentLoopError, match="below the floor"):
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner) == []
    _append(runner, _signed_body(board, removed=removed, round_number=reentry))
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    config = reduced if flags == "reduced" else original
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert "gemini" not in _agents(runner) and _links(runner) == []
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded and all(
        BoardIdentity.from_payload(m.pr_board_root).names == board for m in upgraded
    )
    _qualify(runner, reduced)


# --- evidence requests across a board change (item-1, row pr-remove-with-blocker) ----------


_DEPLOY_LOG = "Attach the production deploy log for this head."
_CANARY = "Attach the canary dashboard for this head."


def _evidence_history(release_head):
    """Gemini's deploy-log request was cleared at ``release_head`` by a durable
    release; in round 2 Gemini (outgoing) re-emits it and Codex (retained)
    emits a new request, at head 6367c48dd193d56ea7b0baad25b19455e529f5ee."""
    from coding_review_agent_loop.round_state import EvidenceReleaseRecord
    from coding_review_agent_loop.unresolved_items import evidence_obligation_identity

    release = _attach_round_metadata(
        "## Exact-head evidence accepted\n\n-- coding-review-agent-loop",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", state="blocking", phase="evidence-release",
            evidence_release=EvidenceReleaseRecord(
                released_head="6367c48dd193d56ea7b0baad25b19455e529f5ee", reason="evidence-cleared",
                signed_requirement_ids_surfaced=(), allowed_rounds=5,
                watch_failure_extension_used=False, watch_head_extension_used=False,
            ),
            evidence_clearances=((evidence_obligation_identity(_DEPLOY_LOG, "Gemini"), release_head),),
        ),
    )
    return [
        _legacy("reviewer", "Codex", 1, state="approved"),
        _legacy("reviewer", "Gemini", 1, state="approved", evidence_requests=(_DEPLOY_LOG,)),
        release,
        _legacy("reviewer", "Codex", 2, state="approved", evidence_requests=(_CANARY,)),
        _legacy("reviewer", "Gemini", 2, state="approved", evidence_requests=(_DEPLOY_LOG,)),
    ]


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
@pytest.mark.parametrize("cleared_at", ["same-head", "5e6741348108b11e3f3ef936c335b1967d17ac6c"])
def test_evidence_requests_survive_a_board_change_with_their_head_scoped_clearances(
    tmp_path, monkeypatch, activation, cleared_at
):
    """Item-1: durable clearances accumulated before the anchor round are kept.

    The outgoing reviewer's request cleared at this head stays cleared; at
    another head it is a new obligation.  The retained reviewer's request is
    rebuilt.  Open evidence obligations block qualification.
    """
    runner = _legacy_runner(_evidence_history("6367c48dd193d56ea7b0baad25b19455e529f5ee" if cleared_at == "same-head" else "f" * 40))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    # Every reviewer dispositions the carried evidence obligations (item-1,
    # and item-2 at another head); on a human-only evidence obligation that is
    # only a note, so the obligation stays owed.
    owed = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The code looks right; the human evidence is still owed."}
        for item_id in (("item-1", "item-2") if cleared_at == "5e6741348108b11e3f3ef936c335b1967d17ac6c" else ("item-1",))
    ]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=owed))
    runner.antigravity_outputs.append(
        structured_pr_review(reviewer="Antigravity", prior_item_dispositions=owed)
    )
    try:
        outcome = run_pr_loop(runner, pr_number=77, config=swapped)
    except AgentLoopError:
        outcome = None
    (payload,) = _link_payloads(runner)
    assert payload["effective_from_round"] == 3
    round_three = [
        r.metadata for r in _records(runner)
        if r.metadata.round_number == 3 and r.metadata.role == "reviewer"
    ]
    assert round_three
    texts = {item.text for item in round_three[0].prior_items if item.obligation_kind}
    assert _CANARY in texts
    assert (_DEPLOY_LOG in texts) is (cleared_at == "5e6741348108b11e3f3ef936c335b1967d17ac6c")
    # Human-only evidence is still owed: the run neither approves nor merges.
    assert outcome != 0
    assert not any(c.startswith("PR #77 approved") for c in runner.comments)


_LIVE_SUITE = "Run the live suite at this exact head and attach the output."
_LIVE_LOG = "Attach the live deploy log for this exact head."


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
def test_frozen_evidence_is_answered_by_the_new_board_after_a_change(
    tmp_path, monkeypatch, activation
):
    """Item-9.s1 (row pr-remove-with-blocker), driven by the live loop.

    Round 1 freezes on two evidence requests: one from Gemini (the outgoing
    reviewer) and one from Codex (retained).  A human signs the evidence.  The
    evidence pass then stops after Gemini resolved both and re-emitted its own
    request, before reconciliation.  Replay never clears evidence from an
    unreconciled pass, so carry-over keeps both obligations frozen, and the
    freeze stays in force across the change.  The new board answers the
    signed input in an evidence-response pass whose reconciliation clears
    both at 6367c48dd193d56ea7b0baad25b19455e529f5ee.  Gemini's re-emission reopens nothing, and the PR
    qualifies.
    """
    from pr_loop_test_helpers import _evidence_review, _signed

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_LIVE_LOG])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", evidence=[_LIVE_SUITE])],
    )
    original = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(pr_loop_support_module.HumanDecisionRequiredError, match="frozen at head 6367c48dd193d56ea7b0baad25b19455e529f5ee"):
        run_pr_loop(runner, pr_number=77, config=original)
    runner.pr_payload["comments"].append(
        _signed("Live suite and deploy log at 6367c48dd193d56ea7b0baad25b19455e529f5ee: 11 passed, deploy healthy.", 1)
    )
    resolved = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The signed output covers this exact head."}
        for item_id in ("item-1", "item-2")
    ]
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", dispositions=resolved, evidence=[_LIVE_SUITE], hr=True,
    ))
    real_post = pr_loop_support_module.post_pr_comment

    def stop_after_gemini(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.role == "reviewer" and record.metadata.dispositions:
            if record.metadata.agent == "Gemini":
                raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", stop_after_gemini)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=original)
    anchor = board_anchor(_records(runner))
    in_anchor = [r.metadata for r in _records(runner) if r.metadata.round_number == anchor.round_number]
    # The anchor round's dispositions were never reconciled.
    assert not any(m.phase == "reconciliation" for m in in_anchor)
    assert any(m.agent == "Gemini" and m.evidence_requests == (_LIVE_SUITE,) for m in in_anchor)
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.antigravity_outputs.append(
        _evidence_review(reviewer="Antigravity", dispositions=resolved, hr=True)
    )
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    (payload,) = _link_payloads(runner)
    assert payload["anchor_round"] == anchor.round_number and payload["advance"] is True
    assert "gemini" not in _agents(runner, start) and "claude" not in _agents(runner, start)
    after = [
        r.metadata for r in _records(runner)
        if r.metadata.round_number == payload["effective_from_round"] and r.metadata.role == "reviewer"
    ]
    assert after and {m.reviewer_board_change_digest for m in after} == {
        board_change_link_digest(payload)
    }
    # The freeze stayed in force across the change: the new board answered the
    # signed input in an evidence-response pass, and its reconciliation
    # cleared both requests at 6367c48dd193d56ea7b0baad25b19455e529f5ee (Gemini's, whose requester left the
    # board, needed every configured reviewer).  Gemini's re-emission at that
    # head reopened nothing: one freeze, one release, nothing owed.
    assert {m.phase for m in after} == {"evidence-response"}
    releases = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-release"]
    assert [m.evidence_release.reason for m in releases] == ["evidence-cleared"]
    assert {identity for identity, head in releases[0].evidence_clearances if head == "6367c48dd193d56ea7b0baad25b19455e529f5ee"} >= {
        item.obligation_identity for item in after[0].prior_items if item.obligation_kind
    }
    assert len([r for r in _records(runner) if r.metadata.phase == "evidence-freeze"]) == 1
    _qualify(runner, swapped)


@pytest.mark.parametrize("activation", ["immediate", "after-crash"])
@pytest.mark.parametrize("head", ["same-head", "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"])
def test_evidence_released_in_the_anchor_round_stays_cleared_at_that_head(
    tmp_path, monkeypatch, activation, head
):
    """Item-9.s1 (row pr-remove-with-blocker), driven by the live loop.

    Round 1 freezes on Codex's (retained) and Gemini's (outgoing) evidence
    requests.  A human signs the evidence; in the evidence pass each requester
    resolves its own request and Gemini re-emits its request at the same head.
    The anchor round's reconciliation clears both and the release (with its
    clearances) is recorded, then the run stops.  After the board change --
    immediate or recovered after a crash -- carry-over keeps that
    post-reconciliation ledger as it is (D4b step 2), so the re-emission at
    6367c48dd193d56ea7b0baad25b19455e529f5ee reopens nothing: no obligation, no second freeze, and the PR
    qualifies.  (Rebuilding an unreconciled anchor round's requests against
    the accumulated clearances is pinned by
    test_evidence_requests_survive_a_board_change_with_their_head_scoped_clearances.)  At a different head the retained requester's same request is a
    new obligation and freezes that head.
    """
    from pr_loop_test_helpers import _evidence_review, _signed

    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_LIVE_LOG])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini", evidence=[_LIVE_SUITE])],
    )
    original = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(pr_loop_support_module.HumanDecisionRequiredError, match="frozen at head 6367c48dd193d56ea7b0baad25b19455e529f5ee"):
        run_pr_loop(runner, pr_number=77, config=original)
    runner.pr_payload["comments"].append(
        _signed("Live suite and deploy log at 6367c48dd193d56ea7b0baad25b19455e529f5ee: 11 passed, deploy healthy.", 1)
    )
    resolved = [
        {"item_id": item_id, "disposition": "resolved",
         "note": "The signed output covers this exact head."}
        for item_id in ("item-1", "item-2")
    ]
    runner.codex_outputs.append(_evidence_review(dispositions=resolved, hr=True))
    runner.gemini_outputs.append(_evidence_review(
        reviewer="Google Gemini", dispositions=resolved, evidence=[_LIVE_SUITE], hr=True,
    ))
    real_post = pr_loop_support_module.post_pr_comment

    def stop_after_release(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.phase == "evidence-release":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", stop_after_release)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=original)
    (release,) = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-release"]
    assert release.evidence_release.reason == "evidence-cleared"
    cleared = {identity for identity, at in release.evidence_clearances if at == "6367c48dd193d56ea7b0baad25b19455e529f5ee"}
    assert len(cleared) == 2
    anchor = board_anchor(_records(runner))
    assert release.round_number == anchor.round_number
    assert any(
        r.metadata.agent == "Gemini" and r.metadata.round_number == anchor.round_number
        and r.metadata.dispositions and r.metadata.evidence_requests == (_LIVE_SUITE,)
        for r in _records(runner)
    )
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    if head == "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6":
        runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    if activation == "after-crash":
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=swapped)
    if head == "same-head":
        runner.codex_outputs.append(_evidence_review(hr=True))
        runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True))
        start = len(runner.commands)
        assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
        assert "gemini" not in _agents(runner, start)
    else:
        # At 0b3d8b29493059afd7f9912106279c4643ac4939 Codex, the retained requester, asks for its evidence again.
        runner.codex_outputs.append(_evidence_review(evidence=[_LIVE_LOG], hr=True))
        runner.antigravity_outputs.append(_evidence_review(reviewer="Antigravity", hr=True))
        with pytest.raises(pr_loop_support_module.HumanDecisionRequiredError, match="frozen at head 0b3d8b29493059afd7f9912106279c4643ac4939"):
            run_pr_loop(runner, pr_number=77, config=swapped)
    (payload,) = _link_payloads(runner)
    assert payload["anchor_round"] == anchor.round_number and payload["advance"] is True
    after = [r.metadata for r in _records(runner)
             if r.metadata.round_number >= payload["effective_from_round"]]
    assert after and all(
        m.reviewer_board_change_digest == board_change_link_digest(payload) for m in after
    )
    # Nothing the anchor round cleared is carried as an obligation.
    carried = [item for m in after if m.role == "reviewer" for item in m.prior_items
               if item.obligation_kind and item.obligation_identity in cleared]
    assert carried == []
    freezes = [r.metadata for r in _records(runner) if r.metadata.phase == "evidence-freeze"]
    if head == "same-head":
        assert [m.subject for m in freezes] == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
        assert not any(m.phase == "evidence-response" for m in after)
        _qualify(runner, swapped)
    else:
        assert [m.subject for m in freezes] == ["6367c48dd193d56ea7b0baad25b19455e529f5ee", "0b3d8b29493059afd7f9912106279c4643ac4939"]
        owed = [item for item in freezes[-1].prior_items if item.obligation_kind]
        # The same obligation identity, cleared only at 6367c48dd193d56ea7b0baad25b19455e529f5ee, is owed at 0b3d8b29493059afd7f9912106279c4643ac4939.
        assert [(item.text, item.reviewer, item.candidate_head_sha) for item in owed] == [
            (_LIVE_LOG, "Codex", "0b3d8b29493059afd7f9912106279c4643ac4939")
        ]
        assert owed[0].obligation_identity in cleared


def _staged_panel_history(tmp_path, monkeypatch):
    """Codex (primary) approves, the panel opens, Gemini raises item-1; then stop."""
    from pr_loop_test_helpers import _staged_config, _staged_review
    from coding_review_agent_loop.review_scheduling import TransitionClassification

    patch_attr_in(monkeypatch, (pr_loop_support_module, pr_loop_module), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(
            reviewer="Google Gemini", state="blocking",
            blocking_items=[{"text": "panel gap", "fix_scope": ["src/worker.py"]}],
        )],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pre_review_tests=False))
    return runner


@pytest.mark.parametrize("opening", ["qualified", "never-opened"])
def test_carried_obligations_require_a_qualified_pre_epoch_opening(tmp_path, monkeypatch, opening):
    """Item-4 (row pr-primary-change): only items created under a qualified
    panel opening before the epoch are exempt from the pre-panel stop."""
    from pr_loop_test_helpers import _staged_config
    from coding_review_agent_loop.board_change import (
        carried_secondary_obligations,
        pre_epoch_panel_items,
    )

    runner = _staged_panel_history(tmp_path, monkeypatch)
    changed = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    lineage = _board_state(runner, changed).lineage
    assert lineage.primary_epoch_round == 4
    records = _records(runner)
    if opening == "never-opened":
        # The same secondary-owned item without the qualified opening record.
        records = [
            r for r in records
            if not (r.metadata.scheduler_phase == "secondary-audit" and r.metadata.role == "summary")
        ]
    eligible = pre_epoch_panel_items(records, lineage)
    ledger = [item for r in records for item in r.metadata.prior_items][-1:]
    carried = carried_secondary_obligations(
        ledger, epoch_round=4, primary="Antigravity", eligible=eligible,
    )
    assert carried == (("item-1",) if opening == "qualified" else ())


def test_unqualified_secondary_item_still_stops_after_a_primary_change(tmp_path, monkeypatch):
    """Item-4: a secondary-owned item that never had a qualified opening keeps
    the pre-panel safety stop, before any reviewer runs."""
    from pr_loop_test_helpers import _rewrite_pr_metadata, _staged_config

    runner = _staged_panel_history(tmp_path, monkeypatch)

    def drop_opening(payload):
        # Demote the secondary-audit decision so history holds no qualified opening.
        if payload.get("scheduler_phase") == "secondary-audit":
            return {**payload, "scheduler_phase": "primary"}
        return None

    _rewrite_pr_metadata(runner, drop_opening)
    changed = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    start = len(runner.commands)
    with pytest.raises(pr_loop_module.PrePanelSafetyError, match="item-1"):
        run_pr_loop(runner, pr_number=77, config=changed)
    assert _agents(runner, start) == []


@pytest.mark.parametrize("path", ["activated", "superseded"])
def test_carried_secondary_finding_survives_successive_primary_changes(
    tmp_path, monkeypatch, path
):
    """Item-11 (row pr-primary-change): a secondary finding published under the
    original qualified opening stays a carried obligation through a second
    primary change -- after the first change activated (later windows hold it
    only in their ledgers), and when the second change supersedes the first
    while it is still pending (both epochs share one round)."""
    from pr_loop_test_helpers import _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)
    first = _staged_config(tmp_path, primary_reviewer="antigravity", pre_review_tests=False)
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; Gemini owns the finding."}]
    if path == "activated":
        # The restarted primary approves; the run stops before the panel answers.
        runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=note))
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=first)
    else:
        with monkeypatch.context() as patch:
            _crash_after_link(patch)
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=first)
    second = _staged_config(
        tmp_path, primary_reviewer="codex", reviewer_codex_model="gpt-primary-next",
        pre_review_tests=False, max_rounds=8,
    )
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=note))
    runner.gemini_outputs.append(_staged_review(
        reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=note))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    first_link, second_link = _link_payloads(runner)
    assert second_link["new_board"]["primary"] == "Codex"
    assert second_link["old_board"]["primary"] == "Antigravity"
    if path == "superseded":
        assert second_link["effective_from_round"] == first_link["effective_from_round"]
    else:
        assert second_link["effective_from_round"] > first_link["effective_from_round"]
    # No pre-panel stop: the restarted primary runs first, then Gemini
    # dispositions its own carried finding when the panel reopens.
    agents = _agents(runner, start)
    assert agents[0] == "codex" and "gemini" in agents[1:]
    assert not any("pre-panel safety" in c for c in runner.comments)
    restarted = [
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "Codex"
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(second_link)
    ]
    assert restarted and [item.item_id for item in restarted[0].prior_items] == ["item-1"]
    _qualify(runner, second)


@pytest.mark.parametrize("antigravity_state", ["pending", "cleared"])
def test_co_owned_carried_obligation_through_primary_restart_and_panel(
    tmp_path, monkeypatch, antigravity_state
):
    """Item-9.s3 (row pr-primary-change): a pre-epoch item co-owned by two
    secondaries is carried through a same-seat primary model change.  Its
    owners keep their states; each still-pending owner dispositions it when
    the panel reopens, and the run qualifies."""
    from pr_loop_test_helpers import _rewrite_pr_metadata, _staged_config, _staged_review

    runner = _staged_panel_history(tmp_path, monkeypatch)

    def co_own(payload):
        changed = False
        for key in ("new_items", "prior_items"):
            items = []
            for item in payload.get(key) or ():
                if item.get("item_id") == "item-1":
                    item = {
                        **item, "resolution_owners": ["Gemini", "Antigravity"],
                        "owner_states": [["Gemini", "pending"], ["Antigravity", antigravity_state]],
                    }
                    changed = True
                items.append(item)
            if items:
                payload[key] = items
        return payload if changed else None

    _rewrite_pr_metadata(runner, co_own)
    changed = _staged_config(
        tmp_path, reviewer_codex_model="gpt-primary-next", pre_review_tests=False, max_rounds=8,
    )
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; the panel owners decide."}]
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=note))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(
        reviewer="Antigravity", dispositions=resolved if antigravity_state == "pending" else note,
    ))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"]
    agents = _agents(runner, start)
    assert agents[0] == "codex" and "gemini" in agents[1:]
    assert not any("pre-panel safety" in c for c in runner.comments)
    restarted = next(
        r.metadata for r in _records(runner)
        if r.metadata.role == "reviewer" and r.metadata.agent == "Codex"
        and r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
    )
    (carried,) = restarted.prior_items
    # The restarted primary saw the obligation with both owners' states intact.
    assert carried.resolution_owners == ("Gemini", "Antigravity")
    assert dict(carried.owner_states) == {"Gemini": "pending", "Antigravity": antigravity_state}
    _qualify(runner, changed)


def test_primary_seat_backend_change_restarts_the_primary_phase(tmp_path, monkeypatch):
    """Item-9.s3 (row pr-primary-change): named primary seat ``p`` moves from
    Codex to Claude under the same seat ID.  The primary phase restarts on the
    rebound seat, the carried secondary finding is exempt from the pre-panel
    stop, its owner dispositions it when the panel reopens, and the run
    qualifies."""
    from pr_loop_test_helpers import _audit_phases, _staged_review
    from coding_review_agent_loop.review_scheduling import TransitionClassification

    patch_attr_in(monkeypatch, (pr_loop_support_module, pr_loop_module), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    p = _seat(tmp_path, "p", "codex", "gpt-p")
    g = _seat(tmp_path, "g", "gemini", "gem-g")
    a = _seat(tmp_path, "a", "antigravity", "Model A")
    staged = {"pr_review_policy": "primary-then-panel", "max_rounds": 8}
    label = {"p": "p (OpenAI Codex: gpt-p)", "g": "g (Google Gemini: gem-g)",
             "a": "a (Google Antigravity: Model A)", "p2": "p (Anthropic Claude: claude-p)"}
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer=label["p"])],
        gemini_outputs=[_staged_review(
            reviewer=label["g"], state="blocking",
            blocking_items=[{"text": "panel gap", "fix_scope": ["src/worker.py"]}],
        )],
        antigravity_outputs=[(_staged_review(reviewer=label["a"]), 0)],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
    )
    # The run stops when the next round's scripted reviews run out.
    with pytest.raises((AgentLoopError, AgentInvocationError)):
        run_pr_loop(runner, pr_number=77, config=_named(tmp_path, p, g, a, primary_reviewer=p, **staged))
    moved = _seat(tmp_path, "p", "claude", "claude-p")
    note = [{"item_id": "item-1", "disposition": "resolved",
             "note": "Looks addressed to me; g owns the finding."}]
    # The rebound primary (on Claude) reviews alone first, then g at the reopened panel.
    runner.claude_outputs.append(_staged_review(reviewer=label["p2"], dispositions=note))
    runner.gemini_outputs.append(_staged_review(
        reviewer=label["g"], dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    ))
    runner.antigravity_outputs.append((_staged_review(reviewer=label["a"], dispositions=note), 0))
    changed = _named(tmp_path, moved, g, a, primary_reviewer=moved, **staged)
    phases_before = len(_audit_phases(runner))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=changed) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["p"] and payload["new_board"]["primary"] == "p"
    assert [seat["backend"] for seat in payload["new_board"]["seats"] if seat["id"] == "p"] == ["claude"]
    assert _agents(runner, start)[0] == "claude" and "codex" not in _agents(runner, start)
    phases = [phase for phase, _head in _audit_phases(runner)][phases_before:]
    assert phases[0] == "primary" and "secondary-audit" in phases
    assert not any("pre-panel safety" in c for c in runner.comments)
    _qualify(runner, changed)


def test_co_owned_pre_epoch_item_is_carried_while_an_owner_is_pending():
    """Row pr-primary-change: co-owned items follow the carry rule per owner."""
    from coding_review_agent_loop.board_change import carried_secondary_obligations

    def co_owned(gemini_state):
        return _item(
            1, "Gemini", owners=("Gemini", "Antigravity"),
            states=(("Gemini", gemini_state), ("Antigravity", "cleared")),
        )

    assert carried_secondary_obligations(
        [co_owned("pending")], epoch_round=4, primary="Codex", eligible=frozenset({"item-1"}),
    ) == ("item-1",)
    assert carried_secondary_obligations(
        [co_owned("cleared")], epoch_round=4, primary="Codex", eligible=frozenset({"item-1"}),
    ) == ()


# --- prelaunch-only anchors and successive changes (row pr-link-crash-recovery) -------------


def _selective_prelaunch_history():
    """Round 1 reviewed at 6367c48dd193d56ea7b0baad25b19455e529f5ee (Gemini raised item-1); round 2 holds only its
    scheduler-prelaunch record (the run stopped right after posting it)."""
    from coding_review_agent_loop.review_scheduling import make_contract

    contract = make_contract(("Codex", "Gemini"), "selective-intermediate", None)
    item = _item(1, "Gemini")

    def prelaunch(round_number, prior):
        return _legacy(
            "summary", "Orchestrator", round_number, phase="scheduler-prelaunch",
            prior_items=prior, scheduler_contract=contract.as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            scheduler_obligation_digest="0" * 16, scheduler_selected_reviewers=("Codex", "Gemini"),
            scheduler_reasons=("full board",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
        )

    return [
        prelaunch(1, ()),
        _legacy("reviewer", "Codex", 1, state="approved"),
        _legacy("reviewer", "Gemini", 1, state="blocking", new_items=(item,)),
        prelaunch(2, (item,)),
    ]


@pytest.mark.parametrize("path", ["recover", "supersede"])
def test_prelaunch_only_anchor_survives_a_crash_and_a_superseding_change(
    tmp_path, monkeypatch, path
):
    """Row pr-link-crash-recovery: a genuinely prelaunch-only round R > 1."""
    runner = _legacy_runner(_selective_prelaunch_history())
    anchor = board_anchor(_records(runner))
    in_round = [r.metadata for r in _records(runner) if r.metadata.round_number == 2]
    assert anchor.round_number == 2 and [m.phase for m in in_round] == ["scheduler-prelaunch"]
    assert not any(m.role in {"coder", "reviewer"} for m in in_round)
    first = make_config(tmp_path, reviewer=("codex", "antigravity"),
                        pr_review_policy="selective-intermediate")
    final = first
    with monkeypatch.context() as patch:
        _crash_after_link(patch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=first)
        if path == "supersede":
            final = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex",
                                pr_review_policy="selective-intermediate")
            with pytest.raises(KeyboardInterrupt):
                run_pr_loop(runner, pr_number=77, config=final)
    payloads = _link_payloads(runner)
    assert all((p["anchor_round"], p["advance"], p["effective_from_round"]) == (2, True, 3)
               for p in payloads)
    assert len(payloads) == (2 if path == "supersede" else 1)
    reviews = {
        "antigravity": "antigravity_outputs", "claude": "claude_outputs", "codex": "codex_outputs",
    }
    for backend in ("codex", "antigravity" if path == "recover" else "claude"):
        getattr(runner, reviews[backend]).append(structured_pr_review(
            reviewer={"codex": "OpenAI Codex", "antigravity": "Antigravity",
                      "claude": "Anthropic Claude"}[backend],
            prior_item_dispositions=_RESOLVED_1,
        ))
    assert run_pr_loop(runner, pr_number=77, config=final) == 0
    assert len(_link_payloads(runner)) == len(payloads)
    new = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    # Re-entered round 3 exactly once, with round 2's ledger carried.
    assert {m.round_number for m in new} == {3}
    reviewer = next(m for m in new if m.role == "reviewer")
    assert [item.item_id for item in reviewer.prior_items] == ["item-1"]
    assert {m.reviewer_board_change_digest for m in new} == {
        board_change_link_digest(payloads[-1])
    }
    _qualify(runner, final)


def test_further_change_after_prelaunch_only_activation_takes_effect_next_round(
    tmp_path, monkeypatch
):
    """Item-9.s4 (row pr-link-crash-recovery): the first link is activated at
    round 3 by nothing but its scheduler-prelaunch record (the run stops right
    after posting it); a second board change then takes effect at round 4."""
    runner = _legacy_runner(_selective_prelaunch_history())
    first = make_config(tmp_path, reviewer=("codex", "antigravity"),
                        pr_review_policy="selective-intermediate")
    real_post = pr_loop_support_module.post_pr_comment

    def stop_after_prelaunch(*args, **kwargs):
        real_post(*args, **kwargs)
        (record,) = _extract_round_metadata_records(
            [SimpleNamespace(body=str(kwargs["body"]))], flow="pr"
        ) or (None,)
        if record is not None and record.metadata.phase == "scheduler-prelaunch":
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", stop_after_prelaunch)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=first)
    (link_one,) = _link_payloads(runner)
    assert link_one["effective_from_round"] == 3
    round_three = [r.metadata for r in _records(runner) if r.metadata.round_number == 3]
    assert [m.phase for m in round_three] == ["scheduler-prelaunch"]
    assert round_three[0].reviewer_board_change_digest == board_change_link_digest(link_one)
    second = make_config(tmp_path, reviewer=("codex", "claude"), coder="codex",
                         pr_review_policy="selective-intermediate")
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=_RESOLVED_1))
    runner.claude_outputs.append(structured_pr_review(
        reviewer="Anthropic Claude", prior_item_dispositions=_RESOLVED_1,
    ))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=second) == 0
    _, link_two = _link_payloads(runner)
    assert link_two["predecessor"] == board_change_link_digest(link_one)
    assert (link_two["anchor_round"], link_two["advance"], link_two["effective_from_round"]) == (3, True, 4)
    assert "agy" not in _agents(runner, start)
    new = [r.metadata for r in _records(runner) if r.metadata.round_number >= 4]
    assert new and {m.round_number for m in new} == {4}
    assert {m.reviewer_board_change_digest for m in new} == {board_change_link_digest(link_two)}
    # Round 2's carried ledger survives both changes.
    reviewer = next(m for m in new if m.role == "reviewer")
    assert [item.item_id for item in reviewer.prior_items] == ["item-1"]
    _qualify(runner, second)


def test_empty_anchor_link_recovery_re_enters_the_anchor_round(tmp_path):
    """Item-9.s4 (row pr-link-crash-recovery): a durable link whose anchor round
    holds no record (advance clear) is recovered by the loop at the anchor
    round itself, without a second link, and the run qualifies."""
    root = identity_from_config(make_config(tmp_path, reviewer=("codex", "gemini")))
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    link, digest = _link_body(
        root, root.digest, root, identity_from_config(swapped), anchor=1, advance=False,
    )
    runner = _legacy_runner([link])
    runner.codex_outputs.append(structured_pr_review())
    runner.antigravity_outputs.append(structured_pr_review(reviewer="Antigravity"))
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=swapped) == 0
    assert len(_links(runner)) == 1
    assert "gemini" not in _agents(runner, start)
    new = _records(runner)
    assert new and {r.metadata.round_number for r in new} == {1}
    assert {r.metadata.reviewer_board_change_digest for r in new} == {digest}
    _qualify(runner, swapped)


def test_further_change_after_coder_only_activation_takes_effect_next_round(tmp_path):
    """Row pr-link-crash-recovery: a link activated by a coder-only round, then another change."""
    runner = _legacy_runner(_gemini_blocker_history())
    swapped = make_config(tmp_path, reviewer=("codex", "antigravity"))
    keep_open = [{"item_id": "item-2", "disposition": "blocking",
                  "note": "The retry is still unbounded in src/worker.py; bound it and test it."}]
    runner.codex_outputs.append(structured_pr_review(state="blocking", prior_item_dispositions=keep_open))
    runner.antigravity_outputs.append(structured_pr_review(
        reviewer="Antigravity", state="blocking", prior_item_dispositions=keep_open,
    ))
    runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-2"]))
    # Round 3 reviews, the coder answers in round 4, and round 4's reviews run out.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=swapped)
    (first,) = _link_payloads(runner)
    round_four = [r.metadata for r in _records(runner) if r.metadata.round_number == 4]
    assert round_four and {m.role for m in round_four} <= {"coder", "summary"}
    assert all(m.reviewer_board_change_digest == board_change_link_digest(first) for m in round_four)
    again = make_config(tmp_path, reviewer=("codex", "gemini"))
    resolved = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_pr_review(prior_item_dispositions=resolved))
    runner.gemini_outputs.append(
        structured_pr_review(reviewer="Google Gemini", prior_item_dispositions=resolved)
    )
    assert run_pr_loop(runner, pr_number=77, config=again) == 0
    _, second = _link_payloads(runner)
    assert second["predecessor"] == board_change_link_digest(first)
    assert (second["anchor_round"], second["advance"], second["effective_from_round"]) == (4, True, 5)
    _qualify(runner, again)


# --- pending signed removal after a rebinding or seat-adding link (row pr-floor-candidate-link)


_LABELS = {"codex": "OpenAI Codex", "gemini": "Google Gemini", "antigravity": "Google Antigravity"}
_OUTPUTS = {"codex": "codex_outputs", "gemini": "gemini_outputs", "antigravity": "antigravity_outputs"}


def _queue_reviews(runner, seats):
    for seat in seats:
        review = structured_pr_review(
            reviewer=f"{seat} ({_LABELS[seat.backend]}: {seat.model_chain[0]})"
        )
        getattr(runner, _OUTPUTS[seat.backend]).append(
            (review, 0) if seat.backend == "antigravity" else review
        )


@pytest.mark.parametrize("link", ["backend-rebinding", "seat-adding"])
def test_pending_signed_removal_after_a_link_is_authorized_from_the_link_board(tmp_path, link):
    """Row pr-floor-candidate-link, end to end: preflight refusal, signed
    activation, resume and qualification all use the frozen authorization
    taken from the predecessor link's board."""
    a = _seat(tmp_path, "a", "codex", "gpt-a")
    b = _seat(tmp_path, "b", "gemini", "gem-b")
    c = _seat(tmp_path, "c", "antigravity", "Model C")
    selective = {"pr_review_policy": "selective-intermediate"}
    runner = FakeRunner()
    _queue_reviews(runner, (a, b, c))
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, a, b, c, **selective)) == 0
    if link == "backend-rebinding":
        changed = (a, _seat(tmp_path, "b", "codex", "gpt-b"), c)
        floor = {"min_distinct_providers": 2}
        refusal = "1 distinct provider\\(s\\) is below the floor of 2"
    else:
        changed = (a, b, c, _seat(tmp_path, "d", "codex", "gpt-d"))
        floor = {"min_reviewers": 4}
        refusal = "3 reviewer\\(s\\) is below the floor of 4"
    runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    _queue_reviews(runner, changed)
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, *changed, **selective, **floor)) == 0
    (first,) = _link_payloads(runner)
    reduced_seats = tuple(seat for seat in changed if str(seat) != "c")
    reduced = _named(tmp_path, *reduced_seats, **selective, **floor)
    comments = len(runner.pr_payload["comments"])
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match=refusal) as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    # Refused at preflight: no record, link or agent.
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, start) == []
    template = refused.value.args[0][refused.value.args[0].index("Reviewer board amendment:"):]
    assert '"c"' in template.split('"removed_reviewers"')[1].split("]")[0]
    _append(runner, template.replace("<why the removed reviewer cannot be reached>", "c is out of quota."))
    runner.pr_payload["headRefOid"] = "59e9fef4edb549af995ec1617c78fd9f54405f29"
    _queue_reviews(runner, reduced_seats)
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert len(_link_payloads(runner)) == 1
    assert any("Reviewer board amendment applied." in c for c in runner.comments)
    assert "agy" not in _agents(runner, start)
    lineage = _board_state(runner, reduced).lineage
    assert [element.kind for element in lineage.elements] == ["link", "amendment"]
    (authorization,) = lineage.authorizations
    assert [seat.seat_id for seat in authorization.seats] == [str(seat) for seat in reduced_seats]
    _qualify(runner, reduced)
    # A resume with the same board derives the same lineage and posts nothing.
    # (A named Gemini seat is re-asked on every rerun, as on main.)
    _queue_reviews(runner, [seat for seat in reduced_seats if seat.backend == "gemini"])
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert len(_link_payloads(runner)) == 1
    assert first == _link_payloads(runner)[0]


_LABELS["claude"] = "Anthropic Claude"
_OUTPUTS["claude"] = "claude_outputs"


@pytest.mark.parametrize("floor", ["reviewers", "providers"])
@pytest.mark.parametrize("flags", ["reduced", "original"])
def test_named_legacy_all_reviewers_signed_reduction(tmp_path, floor, flags):
    """Rows pr-floor-unsigned-vs-signed and pr-legacy-root-anchor: a legacy
    named all-reviewers PR (recorded seat binding, no scheduler contract)."""
    sol = _seat(tmp_path, "sol", "codex", "gpt-sol")
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    opus = _seat(tmp_path, "opus", "claude", "claude-opus")
    board = (sol, flash, opus)
    runner = FakeRunner()
    _queue_reviews(runner, board)
    assert run_pr_loop(runner, pr_number=77, config=_named(tmp_path, *board)) == 0
    _strip_board_identity(runner)
    assert all(r.metadata.pr_board_root is None and r.metadata.seat_binding for r in _records(runner))
    floor_kwargs = {"min_reviewers": 3} if floor == "reviewers" else {"min_distinct_providers": 3}
    kept = (sol, opus)
    reduced = _named(tmp_path, *kept, **floor_kwargs)
    original = _named(tmp_path, *board, **floor_kwargs)
    runner.pr_payload["headRefOid"] = "0b3d8b29493059afd7f9912106279c4643ac4939"
    comments = len(runner.pr_payload["comments"])
    before = len(runner.commands)
    with pytest.raises(AgentLoopError, match="below the floor of 3") as refused:
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert len(runner.pr_payload["comments"]) == comments and _agents(runner, before) == []
    # REST attributes the operator's signed amendment to its own user ID; the
    # named legacy entry check (#1365) reads every comment's author.
    _append(runner, _amendment_from(str(refused.value)), author_id=81)
    start = len(runner.commands)
    _queue_reviews(runner, kept)
    assert run_pr_loop(runner, pr_number=77, config=reduced if flags == "reduced" else original) == 0
    assert "agy" not in _agents(runner, start) and _links(runner) == []
    upgraded = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert upgraded and {
        BoardIdentity.from_payload(m.pr_board_root).names for m in upgraded
    } == {("sol", "flash", "opus")}
    _qualify(runner, reduced)


def test_signed_first_upgrade_initializes_bindings_through_a_crash_then_links_a_model_change(
    tmp_path, monkeypatch
):
    """Row pr-legacy-root-anchor: a plain signed reduction under reduced flags is
    the first upgrade; the crash after its first record keeps the initialization,
    and a later model change is exactly one link needing fresh approval."""
    runner = _legacy_runner([
        _legacy("reviewer", "Codex", 1, subject="abc000", state="approved"),
        _legacy("reviewer", "Gemini", 1, subject="abc000", state="approved"),
        _legacy("coder", "Claude", 2),
    ])
    _append(runner, _signed_body(("Codex", "Gemini"), removed=("Gemini",), round_number=2))
    reduced = make_config(tmp_path, reviewer=("codex",), min_reviewers=2)
    real_post = pr_loop_support_module.post_pr_comment

    def crash_after_first_record(*args, **kwargs):
        real_post(*args, **kwargs)
        if "AGENT_LOOP_META" in str(kwargs["body"]):
            raise KeyboardInterrupt

    runner.codex_outputs.append(structured_pr_review())
    with monkeypatch.context() as patch:
        patch_attr_in(patch, (pr_loop_support_module, pr_loop_module, issue_loop), "post_pr_comment", crash_after_first_record)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=reduced)
    (first,) = [r.metadata for r in _records(runner) if r.metadata.pr_board_root is not None]
    assert first.reviewer_board_amendment_digest is not None
    # Gemini (signed off) stays unknown; Codex is initialized by this record.
    assert set(first.pr_board_binding_init) == {"Codex"}
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert _links(runner) == []
    moved = dataclasses.replace(reduced, reviewer_codex_model="gpt-later")
    runner.codex_outputs.append(structured_pr_review())
    start = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=moved) == 0
    (payload,) = _link_payloads(runner)
    assert payload["rebound"] == ["Codex"] and payload["binding_init"] == {}
    assert _agents(runner, start) == ["codex"]
    _qualify(runner, moved)


# =====================================================================================
# Recorded plan board changes (#1377): the same lineage on the issue owning a plan.
# =====================================================================================

from agent_loop_helpers import structured_plan_review, structured_v1_plan_state  # noqa: E402
from coding_review_agent_loop.board_change import (  # noqa: E402
    PLAN_BOARD_CHANGE_LINK_HEADING,
    PLAN_BOARD_CHANGE_LINK_MARKER,
    PLAN_SURFACE,
)
from coding_review_agent_loop.cli import run_issue_loop  # noqa: E402

_PLAN_ACTOR = ("agent-loop", 4242)
_PLAN_AGENTS = {"claude", "codex", "gemini", "agy"}


def _plan_runner(**outputs):
    return FakeRunner(
        authenticated_actor=_PLAN_ACTOR, serve_rest_issue_comments=True, **outputs
    )


def _plan_cfg(tmp_path, reviewers, **overrides):
    values = dict(
        reviewer=tuple(reviewers), max_rounds=6, plan_execution_mode="plan-only",
        agent_max_retries=0,
    )
    values.update(overrides)
    return make_config(tmp_path, **values)


def _plan_comments(runner):
    return [
        SimpleNamespace(body=c["body"], author_id=c.get("_rest_author_id"))
        for c in runner.issue_comments
    ]


def _plan_records(runner):
    return _extract_round_metadata_records(_plan_comments(runner), flow="plan")


def _plan_links(runner):
    return [
        (index, c["body"]) for index, c in enumerate(runner.issue_comments)
        if c["body"].startswith(PLAN_BOARD_CHANGE_LINK_HEADING)
    ]


def _plan_link_payloads(runner):
    pattern = PLAN_SURFACE.link_re
    return [
        json.loads(base64.urlsafe_b64decode(pattern.search(body).group("payload")))
        for _index, body in _plan_links(runner)
    ]


def _plan_agents(runner, start=0):
    return [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd and cmd[0] in _PLAN_AGENTS]


def _gemini_blocks_round_one(tmp_path, reviewers=("codex", "gemini")):
    """Round 1: Codex approves, Gemini raises item-1; the planner revision never starts."""
    reviews = {
        "codex": ("codex_outputs", structured_plan_review(state="approved")),
        "gemini": ("gemini_outputs", structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="Missing owner.",
            blocking_plan_issues=["Name the rollout owner."],
        )),
        "antigravity": ("antigravity_outputs", structured_plan_review(
            state="approved", reviewer="Google Antigravity",
        )),
    }
    outputs = {"claude_outputs": [structured_v1_plan_state()]}
    for name in reviewers:
        key, review = reviews[name]
        outputs[key] = [review]
    runner = _plan_runner(**outputs)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=_plan_cfg(tmp_path, reviewers), plan_first=True,
        )
    return runner


def _append_issue_comment(runner, body, *, login="operator", author_id=81):
    """An operator comment created now, after every comment the run posted."""
    import datetime

    runner.issue_comments.append({
        "body": body, "author": {"login": login}, "_rest_author_id": author_id,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })


def _review(name, state="approved", dispositions=_RESOLVED_1, **extra):
    signature = {
        "codex": "OpenAI Codex", "gemini": "Google Gemini", "antigravity": "Google Antigravity",
    }[name]
    return structured_plan_review(
        state=state, reviewer=signature, prior_plan_item_dispositions=dispositions, **extra,
    )


def test_plan_midrun_swap_posts_one_link_and_requires_the_new_reviewer(tmp_path):
    """Rows plan-board-change-midrun / plan-all-reviewers-board-change: Gemini -> Antigravity."""
    runner = _gemini_blocks_round_one(tmp_path)
    before_comments = len(runner.issue_comments)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    start = len(runner.commands)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0

    (payload,) = _plan_link_payloads(runner)
    assert payload["flow"] == "plan" and payload["issue_number"] == 56
    assert "pr_number" not in payload
    assert payload["added"] == ["Antigravity"] and payload["removed"] == ["Gemini"]
    assert payload["reason"] == "operator-reconfigured"
    assert payload["anchor_round"] == 1 and payload["advance"] is True
    assert payload["effective_from_round"] == 2
    ((link_index, _body),) = _plan_links(runner)
    # The link is the first comment of the rerun: posted before any agent turn.
    assert link_index == before_comments
    # Round 2 re-enters reviewer-only on the byte-identical plan: no planner turn.
    assert _plan_agents(runner, start) == ["codex", "agy"]
    records = _plan_records(runner)
    round_two = [r for r in records if r.metadata.round_number == 2]
    assert {r.metadata.role for r in round_two} == {"reviewer"}
    assert {r.metadata.subject for r in records} == {records[0].metadata.subject}
    # The removed reviewer's open finding was carried, never dropped.
    assert all(
        [item.item_id for item in r.metadata.prior_items] == ["item-1"] for r in round_two
    )
    assert all(
        r.metadata.reviewer_board_change_digest == board_change_link_digest(payload)
        and r.metadata.plan_board_root is not None
        for r in round_two
    )
    # Idempotent resume: the approved plan is never reopened and no link repeats.
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    assert _plan_agents(runner, start) == []
    assert len(_plan_links(runner)) == 1


def test_plan_swapped_out_reviewer_approval_never_counts(tmp_path):
    """Row plan-board-change-midrun: the new seat must approve; the old one's approval is gone."""
    runner = _gemini_blocks_round_one(tmp_path)
    # Swap Codex (which approved round 1) for Antigravity, which then blocks.
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review(
        "antigravity", state="blocking", dispositions=[
            {"item_id": "item-1", "disposition": "resolved"},
        ],
        summary="Still unclear.", blocking_plan_issues=["Clarify the rollback."],
    ))
    swapped = _plan_cfg(tmp_path, ("antigravity", "gemini"))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    (payload,) = _plan_link_payloads(runner)
    assert payload["removed"] == ["Codex"] and payload["added"] == ["Antigravity"]
    # Codex's round-1 approval did not approve the plan: a planner revision followed.
    assert not any("plan approved" in c["body"] for c in runner.issue_comments)
    assert _plan_agents(runner)[-1] == "claude"


def test_plan_removed_reviewer_finding_stays_open_and_blocks_approval(tmp_path):
    """Row plan-all-reviewers-board-change: removing the finding's owner reassigns it."""
    runner = _gemini_blocks_round_one(tmp_path, reviewers=("codex", "gemini", "antigravity"))
    # Gemini (owner of item-1) leaves; its finding stays open for the remaining
    # board, which keeps it blocking, so the plan is not approved.
    still_open = [{"item_id": "item-1", "disposition": "blocking"}]
    runner.codex_outputs.append(_review(
        "codex", state="blocking", dispositions=still_open, summary="Owner still missing.",
    ))
    runner.antigravity_outputs.append(_review(
        "antigravity", state="blocking", dispositions=still_open, summary="Owner still missing.",
    ))
    reduced = _plan_cfg(tmp_path, ("codex", "antigravity"))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True)
    (payload,) = _plan_link_payloads(runner)
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    assert not any("plan approved" in c["body"] for c in runner.issue_comments)
    round_two = [r for r in _plan_records(runner) if r.metadata.round_number == 2]
    assert round_two and all(
        [item.item_id for item in r.metadata.prior_items] == ["item-1"] for r in round_two
    )
    # The next planner revision is asked to address the carried item.
    assert _plan_agents(runner)[-1] == "claude"


def test_plan_added_reviewer_is_required_from_the_effective_round(tmp_path):
    """Row plan-board-change-midrun: an added reviewer reviews the current plan."""
    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review("antigravity"))
    widened = _plan_cfg(tmp_path, ("codex", "gemini", "antigravity"))
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["added"] == ["Antigravity"] and payload["removed"] == []
    assert sorted(_plan_agents(runner, start)) == ["agy", "codex", "gemini"]


def test_plan_named_model_rebind_is_a_rebound_seat_link(tmp_path):
    """Row plan-board-change-midrun: a same-ID seat on another model is a swap."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        antigravity_outputs=[structured_plan_review(
            state="blocking", reviewer="flash (Google Antigravity: Model A)", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
    )
    config = _plan_cfg(tmp_path, ("codex", flash), reviewer_seats=(flash,))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    rebound = _seat(tmp_path, "flash", "antigravity", "Model B")
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model B)",
        prior_plan_item_dispositions=_RESOLVED_1,
    ))
    rebound_config = _plan_cfg(tmp_path, ("codex", rebound), reviewer_seats=(rebound,))
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=rebound_config, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["rebound"] == ["flash"] and payload["added"] == [] == payload["removed"]
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands[start:]
              if cmd[0] == "agy" and "--model" in cmd]
    assert models == ["Model B"]


def test_plan_plain_to_named_and_back_each_post_one_link(tmp_path):
    """Row plan-all-reviewers-board-change: plain-to-named, then named-to-plain."""
    runner = _gemini_blocks_round_one(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner.codex_outputs.append(_review("codex", state="blocking", summary="More.",
                                        blocking_plan_issues=["Add rollback steps."]))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model A)",
        prior_plan_item_dispositions=_RESOLVED_1,
    ))
    named = _plan_cfg(tmp_path, ("codex", flash), reviewer_seats=(flash,))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=named, plan_first=True)
    first = _plan_link_payloads(runner)
    assert len(first) == 1 and first[0]["new_seat_binding"] is not None
    assert first[0]["old_seat_binding"] is None
    # Back to the plain board while the planner revision is still pending: a
    # second link at the re-entry round; nothing is posted twice.
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "gemini")),
                       plan_first=True)
    payloads = _plan_link_payloads(runner)
    assert len(payloads) == 2
    assert payloads[1]["new_seat_binding"] is None
    assert payloads[1]["predecessor"] == board_change_link_digest(payloads[0])


def test_plan_link_crash_before_effective_round_is_recovered_once(tmp_path, monkeypatch):
    """Row plan-link-crash-resume: the link is recovered, never reposted."""
    import coding_review_agent_loop.plan_first_loop as plan_first_loop

    runner = _gemini_blocks_round_one(tmp_path)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    real = plan_first_loop.set_plan_board_stamp

    def crash(stamp):
        if stamp is not None:
            raise AgentLoopError("simulated crash after the link post")
        return real(stamp)

    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", crash)
    with pytest.raises(AgentLoopError, match="simulated crash"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", real)
    assert len(_plan_links(runner)) == 1
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    assert len(_plan_links(runner)) == 1
    rounds = sorted({r.metadata.round_number for r in _plan_records(runner)})
    assert rounds == [1, 2]


def test_plan_further_change_while_a_link_is_pending_supersedes_it(tmp_path, monkeypatch):
    """Row plan-link-crash-resume: a further differing board adds exactly one link."""
    import coding_review_agent_loop.plan_first_loop as plan_first_loop

    runner = _gemini_blocks_round_one(tmp_path)
    real = plan_first_loop.set_plan_board_stamp

    def crash(stamp):
        if stamp is not None:
            raise AgentLoopError("simulated crash after the link post")
        return real(stamp)

    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", crash)
    with pytest.raises(AgentLoopError, match="simulated crash"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
                       plan_first=True)
    monkeypatch.setattr(plan_first_loop, "set_plan_board_stamp", real)
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    runner.antigravity_outputs.append(_review("antigravity"))
    widened = _plan_cfg(tmp_path, ("codex", "gemini", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    first, second = _plan_link_payloads(runner)
    assert second["predecessor"] == board_change_link_digest(first)
    assert (second["anchor_round"], second["advance"]) == (first["anchor_round"], first["advance"])
    assert second["effective_from_round"] == 2


def _plan_link_body(runner, **overrides):
    payload = dict(_plan_link_payloads(runner)[0])
    payload.update(overrides)
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    return f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: {encoded} -->"


@pytest.mark.parametrize("defect", ["foreign-author", "malformed", "wrong-issue", "unknown-digest"])
def test_forged_or_malformed_plan_link_fails_closed(tmp_path, defect):
    """Row plan-link-forged: resume refuses before any agent invocation or post."""
    runner = _gemini_blocks_round_one(tmp_path)
    swapped = _plan_cfg(tmp_path, ("codex", "antigravity"))
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True) == 0
    link = next(c for c in runner.issue_comments if c["body"].startswith(PLAN_BOARD_CHANGE_LINK_HEADING))
    if defect == "foreign-author":
        link["_rest_author_id"] = 99
        link["author"] = {"login": "someone-else"}
    elif defect == "malformed":
        link["body"] = f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: e30= -->"
    elif defect == "wrong-issue":
        link["body"] = _plan_link_body(runner, issue_number=57)
    else:
        _rewrite_last_plan_record(runner, reviewer_board_change_digest="f" * 64)
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        run_issue_loop(runner, issue_number=56, config=swapped, plan_first=True)
    assert _plan_agents(runner, start) == []
    assert len(runner.issue_comments) == comments


def _planned_round_one(tmp_path, reviewers=("codex", "gemini")):
    """Round 1 holds only the published plan: the first reviewer turn never ran."""
    runner = _plan_runner(claude_outputs=[structured_v1_plan_state()])
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56,
            config=_plan_cfg(tmp_path, reviewers, review_parallel=False), plan_first=True,
        )
    return runner


def test_plan_floor_shrink_refused_unsigned_and_allowed_signed(tmp_path):
    """Rows plan-floor-shrink / plan-all-reviewers-board-change (signed floor case)."""
    runner = _planned_round_one(tmp_path)
    one = _plan_cfg(tmp_path, ("codex",), min_reviewers=2)
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="below the floor of 2") as refusal:
        run_issue_loop(runner, issue_number=56, config=one, plan_first=True)
    # The refusal prints the filled-in signed plan amendment template and posts nothing.
    assert '"kind": "reviewer-board-amendment"' in str(refusal.value)
    assert '"flow": "plan"' in str(refusal.value)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments
    assert _plan_links(runner) == []

    signed = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    )
    _append_issue_comment(runner, signed)
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    # Explicitly reduced flags are accepted under the signed reduction.
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=one, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["codex"]
    assert _plan_links(runner) == []


def test_plan_signed_reduction_with_original_flags_rebinds_to_the_signed_board(tmp_path):
    """Row plan-all-reviewers-board-change: original flags are rebound (#1133) under the floor."""
    runner = _planned_round_one(tmp_path)
    signed = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    )
    _append_issue_comment(runner, signed)
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    start = len(runner.commands)
    original = _plan_cfg(tmp_path, ("codex", "gemini"), min_reviewers=2, min_distinct_providers=2)
    assert run_issue_loop(runner, issue_number=56, config=original, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["codex"]
    assert _plan_links(runner) == []


def test_plan_provider_floor_refuses_an_unsigned_same_provider_board(tmp_path):
    """Row plan-all-reviewers-board-change: the distinct-provider floor on a plan link."""
    runner = _gemini_blocks_round_one(tmp_path)
    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    opus = _seat(tmp_path, "opus", "antigravity", "Model B")
    same_provider = _plan_cfg(
        tmp_path, (flash, opus), reviewer_seats=(flash, opus), min_distinct_providers=2,
    )
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="distinct provider"):
        run_issue_loop(runner, issue_number=56, config=same_provider, plan_first=True)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments


def _rewrite_last_plan_record(runner, **changes):
    """Rewrite the newest plan round record carrying a board-change digest."""
    import re

    from coding_review_agent_loop.round_state import _decode_round_metadata

    for comment in reversed(runner.issue_comments):
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow != "plan" or metadata.reviewer_board_change_digest is None:
            continue
        body = comment["body"][: match.start()].rstrip("\n")
        comment["body"] = _attach_round_metadata(body, dataclasses.replace(metadata, **changes))
        return
    raise AssertionError("no stamped plan record")


def _strip_plan_stamps(runner):
    """Rewrite every plan record as pre-upgrade (unstamped) history."""
    from coding_review_agent_loop.round_state import _decode_round_metadata

    import re

    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow != "plan":
            continue
        stripped = dataclasses.replace(
            metadata, plan_board_root=None, plan_board_identity=None,
            plan_board_binding_init=None, reviewer_board_change_digest=None,
            reviewer_board_amendment_digest=None,
        )
        body = comment["body"][: match.start()].rstrip("\n")
        comment["body"] = _attach_round_metadata(body, stripped)


def test_plan_legacy_plain_history_upgrades_without_any_record(tmp_path):
    """Row plan-legacy-regression: unchanged board on unstamped history posts nothing."""
    runner = _gemini_blocks_round_one(tmp_path)
    _strip_plan_stamps(runner)
    assert all(r.metadata.plan_board_root is None for r in _plan_records(runner))
    runner.claude_outputs.append(structured_plan_review())  # unused guard
    runner.claude_outputs.clear()
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "gemini")),
                       plan_first=True)
    # Exactly today's continuation: the planner revision, no link, no new record.
    assert _plan_agents(runner, start) == ["claude"]
    assert _plan_links(runner) == []
    assert len(runner.issue_comments) == comments


def test_plan_legacy_plain_history_migrates_its_root_from_posters(tmp_path):
    """Row plan-all-reviewers-board-change: legacy root = configured board plus posters."""
    runner = _gemini_blocks_round_one(tmp_path)
    _strip_plan_stamps(runner)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    (payload,) = _plan_link_payloads(runner)
    root = BoardIdentity.from_payload(payload["root"])
    # The configured board plus every historical plan reviewer poster, read-only.
    assert root.names == ("Codex", "Antigravity", "Gemini")
    assert root.policy == "all-reviewers" and root.primary is None
    assert payload["removed"] == ["Gemini"] and payload["added"] == []
    # Persisted legacy rounds are never rewritten.
    assert all(
        r.metadata.plan_board_root is None
        for r in _plan_records(runner) if r.metadata.round_number == 1
    )


def test_plan_policy_change_stays_drift_including_legacy_history(tmp_path):
    """Row plan-policy-drift: legacy plain history roots at all-reviewers, no primary."""
    runner = _gemini_blocks_round_one(tmp_path)
    for strip in (False, True):
        if strip:
            _strip_plan_stamps(runner)
        staged = _plan_cfg(
            tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer="codex",
        )
        start, comments = len(runner.commands), len(runner.issue_comments)
        with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
            run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
        assert _plan_agents(runner, start) == []
        assert len(runner.issue_comments) == comments
        assert _plan_links(runner) == []


def test_plan_legacy_partial_approval_is_undetermined_and_resumes_as_today(tmp_path):
    """Row plan-legacy-partial-approval: posters alone never prove completion."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"), review_parallel=False)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    _strip_plan_stamps(runner)
    # Only Codex approved; Gemini never posted.
    assert {r.metadata.agent for r in _plan_records(runner) if r.metadata.role == "reviewer"} == {
        "Codex"
    }
    classification = classify_plan_approval(
        _plan_comments(runner), config=config, issue_number=56, human_requirements=(),
    )
    assert classification.state == "undetermined"
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    start = len(runner.commands)
    floor = _plan_cfg(tmp_path, ("codex", "gemini"), min_reviewers=2, review_parallel=False)
    assert run_issue_loop(runner, issue_number=56, config=floor, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["gemini"]
    assert _plan_links(runner) == []


def test_approved_plan_is_never_reopened_by_a_different_board(tmp_path):
    """Row plan-approved-board-change: approved by B, resumed with C and below the floor."""
    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    board_b = _plan_cfg(tmp_path, ("codex", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=board_b, plan_first=True) == 0
    links = len(_plan_links(runner))
    for board_c in (
        _plan_cfg(tmp_path, ("gemini", "antigravity")),
        _plan_cfg(tmp_path, ("gemini", "codex"), plan_review_policy="primary-then-panel",
                  primary_plan_reviewer="gemini", min_reviewers=2),
    ):
        start, comments = len(runner.commands), len(runner.issue_comments)
        assert run_issue_loop(runner, issue_number=56, config=board_c, plan_first=True) == 0
        assert _plan_agents(runner, start) == []
        assert len(_plan_links(runner)) == links
        assert len(runner.issue_comments) == comments
    # An approved plan-only resume with a below-floor invocation board selects
    # no board: no planning candidate floor, no handoff floor, no link.
    start, comments = len(runner.commands), len(runner.issue_comments)
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("gemini",), min_reviewers=2),
        plan_first=True,
    ) == 0
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments


def test_approval_classification_is_independent_of_the_invocation_board(tmp_path):
    """Row plan-approved-board-change (unit parity): the approving identity governs."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    comments = _plan_comments(runner)
    results = set()
    for invocation in (
        _plan_cfg(tmp_path, ("codex", "antigravity")),
        _plan_cfg(tmp_path, ("gemini",)),
        _plan_cfg(tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
                  primary_plan_reviewer="gemini"),
    ):
        classification = classify_plan_approval(
            comments, config=invocation, issue_number=56, human_requirements=(),
            trusted_actor=_PLAN_ACTOR,
        )
        results.add((classification.state, classification.context.effective_reviewers))
    assert results == {("approved", ("Codex", "Antigravity"))}


def test_unanimous_approvals_with_an_open_carried_obligation_are_not_approved():
    """Row plan-approval-gate-open-obligation (unit): a reassigned item blocks completion."""
    from coding_review_agent_loop.plan_verification import plan_open_obligations
    from coding_review_agent_loop.round_state import PostedRoundRecord, ResumedReviewRound

    gemini_item = UnresolvedReviewItem(
        item_id="item-1", reviewer="Gemini", source_round=1, text="Name the owner.",
        status="blocking",
    )

    def approval(index, agent, dispositions=()):
        return PostedRoundRecord(
            index=index, body="",
            metadata=PostedRoundMetadata(
                flow="plan", role="reviewer", agent=agent, round_number=2, subject="s",
                state="approved", prior_items=(gemini_item,), dispositions=tuple(dispositions),
            ),
        )

    unanimous = ResumedReviewRound(
        round_number=2, prior_items=(gemini_item,), coder_output=None,
        completed_reviews=(approval(1, "Codex"), approval(2, "Antigravity")),
        next_unresolved_item_number=2,
    )
    # Every remaining reviewer approved, but the removed reviewer's finding was
    # never dispositioned: it is reassigned to the remaining board and stays open.
    assert plan_open_obligations(unanimous, board=("Codex", "Antigravity")) == ("item-1",)
    from coding_review_agent_loop.protocol import ReviewItemDisposition

    resolved = ResumedReviewRound(
        round_number=2, prior_items=(gemini_item,), coder_output=None,
        completed_reviews=(
            approval(1, "Codex", [ReviewItemDisposition(item_id="item-1", disposition="resolved",
                                                        reviewer="Codex")]),
            approval(2, "Antigravity"),
        ),
        next_unresolved_item_number=2,
    )
    assert plan_open_obligations(resolved, board=("Codex", "Antigravity")) == ()


def test_requirement_evidence_is_recorded_under_all_reviewers_and_binds_approval(tmp_path):
    """Row plan-approval-requirement-evidence: recorded keys bind; a change unbinds."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    reviews = [r for r in _plan_records(runner) if r.metadata.role == "reviewer"]
    # Generation-1 reviewer records carry the exact-plan key as evidence only.
    assert all(r.metadata.plan_candidate_key is not None for r in reviews)
    assert all(r.metadata.scheduler_contract is None for r in _plan_records(runner))
    comments = _plan_comments(runner)
    unchanged = classify_plan_approval(
        comments, config=config, issue_number=56, human_requirements=(),
        trusted_actor=_PLAN_ACTOR,
    )
    assert unchanged.state == "approved"
    from coding_review_agent_loop.github import HumanReviewRequirement

    added = HumanReviewRequirement(
        source_type="issue_comment", author="operator", created_at="2026-05-23T01:00:00Z",
        url=None, body="Keep the rollout reversible.",
    )
    # A signed requirement added after the approvals: the recorded key no longer
    # matches, so the approval cannot be shortcut.
    changed = classify_plan_approval(
        comments, config=config, issue_number=56, human_requirements=(added,),
        trusted_actor=_PLAN_ACTOR,
    )
    assert changed.state in {"unfinished", "undetermined"}


def test_plan_verification_resolves_the_lineage_and_judges_the_approving_board(tmp_path):
    """Row plan-lineage-to-pr-verification (unit): A -> B approved by B verifies against B."""
    from coding_review_agent_loop.plan_verification import derive_plan_verification_context

    runner = _gemini_blocks_round_one(tmp_path)
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(_review("antigravity"))
    assert run_issue_loop(
        runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex", "antigravity")),
        plan_first=True,
    ) == 0
    context = derive_plan_verification_context(
        _plan_comments(runner), issue_number=56, trusted_actor=_PLAN_ACTOR,
    )
    assert context.derivation == "lineage"
    assert context.effective_reviewers == ("Codex", "Antigravity")
    # Gemini's round-1 record is not an approving-board approval; nothing else is stale.
    with pytest.raises(AgentLoopError, match="not authored by the authenticated orchestrator"):
        derive_plan_verification_context(
            _plan_comments(runner), issue_number=56, trusted_actor=("intruder", 7),
        )


@pytest.mark.parametrize("finding", ["blocking_plan_issues", "same_plan_followups"])
def test_plan_primary_change_restarts_the_primary_phase_and_carries_the_panel_item(
    tmp_path, finding
):
    """Row plan-primary-change-midrun: Codex -> Antigravity as primary-then-panel primary.

    Both blocking and same-plan secondary findings are carried obligations
    (item-6): neither raises the pre-panel safety stop.
    """
    def staged(primary, **extra):
        return _plan_cfg(
            tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary, **extra,
        )

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            **{finding: ["Name the owner."]},
        )],
        antigravity_outputs=[structured_plan_review(state="approved", reviewer="Google Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    records = _plan_records(runner)
    gemini_items = [
        item.item_id for r in records if r.metadata.agent == "Gemini" for item in r.metadata.new_items
    ]
    assert gemini_items == ["item-1"]
    anchor = max(r.metadata.round_number for r in records)
    # The new primary approves the unchanged plan; the panel then needs Codex and
    # Gemini, whose outputs are not scripted, so the run stops there.
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="Google Antigravity", prior_plan_item_dispositions=_RESOLVED_1,
    ))
    start = len(runner.commands)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=staged("antigravity", plan_primary_stall_rounds=1),
            plan_first=True,
        )
    (payload,) = _plan_link_payloads(runner)
    assert payload["old_board"]["primary"] == "Codex"
    assert payload["new_board"]["primary"] == "Antigravity"
    assert payload["effective_from_round"] == anchor + 1
    # The next round is primary-only on the new primary: no pre-panel safety
    # stop for Gemini's carried item, and the old primary's opening is not reused.
    assert _plan_agents(runner, start)[0] == "agy"
    prelaunch = [
        r.metadata for r in _plan_records(runner)
        if r.metadata.phase == "scheduler-prelaunch" and r.metadata.round_number == anchor + 1
    ]
    assert prelaunch and prelaunch[0].scheduler_phase == "primary"
    assert prelaunch[0].scheduler_selected_reviewers == ("Antigravity",)
    assert prelaunch[0].scheduler_primary_reviewer == "Antigravity"
    assert [item.item_id for item in prelaunch[0].prior_items] == ["item-1"]
    assert not any("Plan review scheduling diagnostic" in c["body"] for c in runner.issue_comments)


def test_supersession_inherited_failure_and_mode_guard_are_never_approved(tmp_path):
    """Rows plan-supersession-not-shortcut / plan-approval-gate-live-inputs (unit)."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    comments = _plan_comments(runner)

    def classify(**kwargs):
        return classify_plan_approval(
            comments, config=kwargs.pop("config", config), issue_number=56,
            human_requirements=(), trusted_actor=_PLAN_ACTOR, **kwargs,
        )

    assert classify().state == "approved"
    # A pending validated signed supersession is never shortcut as approved.
    assert classify(supersession_pending=True).state == "unfinished"
    # An inadmissible inherited matrix routes to the revision path, never approval.
    assert classify(inherited_check=lambda _text: "inherited row weakened").state == "unfinished"
    assert classify(inherited_check=lambda _text: None).state == "approved"
    # An underivable inherited input is not approved.
    assert classify(inherited_derivable=False).state == "undetermined"


# --- Round 2: shared approval inputs, stale bindings, ledger, carried same-plan items


def _patch_requirements(monkeypatch, requirements):
    real = discuss_loop.get_issue_context

    def patched(runner_arg, *, config, issue_number):
        context = real(runner_arg, config=config, issue_number=issue_number)
        if issue_number != 56:
            return context
        return dataclasses.replace(context, human_requirements=tuple(requirements))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy_module, child_plan_binding_module, pr_loop_support_module, pr_loop_module, plan_first_loop_module, issue_loop), "get_issue_context", patched)


def _signed_requirement():
    from coding_review_agent_loop.github import HumanReviewRequirement

    return HumanReviewRequirement(
        source_type="Issue body", author="maintainer", created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56", body="Keep the public API unchanged.",
    )


def _approved_plain_plan(tmp_path, *, requirement=None):
    from coding_review_agent_loop.protocol import HUMAN_REQUIREMENTS_ADDRESSED_MARKER

    plan = structured_v1_plan_state()
    reviews = {"codex": structured_plan_review(state="approved"),
               "gemini": structured_plan_review(state="approved", reviewer="Google Gemini")}
    if requirement is not None:
        dispositions = [{"requirement_id": requirement.requirement_id, "disposition": "addressed",
                         "evidence": "The plan preserves the public API."}]
        plan = plan.replace(
            '"human_requirement_dispositions": []',
            '"human_requirement_dispositions": ' + json.dumps(dispositions), 1,
        ).replace(
            "\n<!-- AGENT_PLAN_STATE: blocking -->",
            f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n"
            f"- {requirement.requirement_id}: the plan preserves the public API.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->", 1,
        )
        reviews = {
            name: structured_plan_review(
                state="approved", reviewer=signature, human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
            for name, signature in (("codex", "OpenAI Codex"), ("gemini", "Google Gemini"))
        }
    runner = _plan_runner(
        claude_outputs=[plan], codex_outputs=[reviews["codex"]], gemini_outputs=[reviews["gemini"]],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    return runner, config


def test_added_signed_requirement_reopens_review_of_an_approved_plan(tmp_path, monkeypatch):
    """Rows plan-approval-gate-live-inputs / plan-approval-requirement-evidence (public entry)."""
    runner, config = _approved_plain_plan(tmp_path)
    _patch_requirements(monkeypatch, [_signed_requirement()])
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    start = len(runner.commands)
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    assert classify_plan_approval(
        _plan_comments(runner), config=config, issue_number=56,
        human_requirements=(_signed_requirement(),), trusted_actor=_PLAN_ACTOR,
    ).state == "unfinished"
    with pytest.raises(AgentLoopError) as stopped:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    # The stale approvals are never reused: the reviewers review again (the
    # first one now meets the unacknowledged requirement), and no approved
    # outcome or handoff follows.
    assert _plan_agents(runner, start)[:1] == ["codex"]
    assert "requirement" in str(stopped.value).lower() or "exhausted" in str(stopped.value)
    assert _plan_links(runner) == []


def test_withdrawn_signed_requirement_reopens_review_of_an_approved_plan(tmp_path, monkeypatch):
    """Item-2: with the last requirement withdrawn, old approvals are not reused."""
    requirement = _signed_requirement()
    _patch_requirements(monkeypatch, [requirement])
    runner, config = _approved_plain_plan(tmp_path, requirement=requirement)
    _patch_requirements(monkeypatch, [])
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    assert classify_plan_approval(
        _plan_comments(runner), config=config, issue_number=56, human_requirements=(),
        trusted_actor=_PLAN_ACTOR,
    ).state == "unfinished"
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    start = len(runner.commands)
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass
    assert {"codex", "gemini"} <= set(_plan_agents(runner, start))


def test_parent_requirement_binding_is_part_of_the_approval_classification(tmp_path):
    """Item-3: authoritative-parent requirements bind every counted approval."""
    from coding_review_agent_loop.plan_verification import (
        approval_record_binding,
        classify_plan_approval,
    )

    runner, config = _approved_plain_plan(tmp_path)
    comments = _plan_comments(runner)

    def classify(parent_ids):
        return classify_plan_approval(
            comments, config=config, issue_number=56, trusted_actor=_PLAN_ACTOR,
            parent_requirement_ids=parent_ids,
        ).state

    assert classify(None) == "approved"
    assert classify(()) == "approved"
    # Records without parent evidence cannot prove a non-empty parent set.
    assert classify(("hr-" + "b" * 64,)) == "undetermined"
    review = next(r for r in _plan_records(runner) if r.metadata.role == "reviewer")
    recorded = SimpleNamespace(
        metadata=dataclasses.replace(review.metadata, parent_requirement_ids=("hr-a",)), body="",
    )
    key = None
    assert approval_record_binding(
        recorded, current_key=key, surfaced_ids=(), parent_ids=("hr-b",),
    ) == "stale"


def test_recovery_refuses_an_unreconstructible_ledger(tmp_path):
    """Items 1/8: a published obligation the ledger lost fails closed, also in recovery."""
    import re

    from coding_review_agent_loop.github import IssueComment, IssueContext
    from coding_review_agent_loop.plan_verification import (
        classify_plan_approval,
        require_recoverable_plan_approval,
    )
    from coding_review_agent_loop.round_state import _decode_round_metadata

    runner, config = _approved_plain_plan(tmp_path)
    lost = UnresolvedReviewItem(
        item_id="item-7", reviewer="Gemini", source_round=1, text="Lost finding.",
        status="blocking",
    )
    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.role == "reviewer" and metadata.agent == "Gemini":
            body = comment["body"][: match.start()].rstrip("\n")
            comment["body"] = _attach_round_metadata(
                body, dataclasses.replace(metadata, new_items=(lost,)),
            )
    comments = _plan_comments(runner)
    classification = classify_plan_approval(
        comments, config=config, issue_number=56, trusted_actor=_PLAN_ACTOR,
    )
    assert not classification.approved
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="t", body="b", url=None,
        comments=tuple(
            IssueComment(author=c["author"]["login"], created_at=c["createdAt"], body=c["body"],
                          author_id=c.get("_rest_author_id"))
            for c in runner.issue_comments
        ),
    )
    with pytest.raises(AgentLoopError, match="not complete"):
        require_recoverable_plan_approval(
            None, config=config, issue_context=issue, comments=issue.comments,
            trusted_actor=_PLAN_ACTOR, error_message="not complete",
        )


def test_planning_child_inputs_derive_inherited_admissibility(tmp_path, monkeypatch):
    """Item-5: a planning child's admissibility is derived, so approval is recognized."""
    import coding_review_agent_loop.child_plan_binding as child_plan_binding
    import coding_review_agent_loop.execution_policy as execution_policy
    from coding_review_agent_loop.github import IssueComment, IssueContext
    from coding_review_agent_loop.plan_verification import (
        build_plan_approval_inputs,
        classify_plan_approval,
        enforce_issue_board_floor,
    )

    runner, _config = _approved_plain_plan(tmp_path)
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="t", body="b", url=None,
        comments=tuple(
            IssueComment(author=c["author"]["login"], created_at=c["createdAt"], body=c["body"],
                          author_id=c.get("_rest_author_id"))
            for c in runner.issue_comments
        ),
    )
    parent = IssueContext(number=55, repo="OWNER/REPO", title="p", body="p", url=None, comments=())
    monkeypatch.setattr(execution_policy, "_infer_staged_parent_issue", lambda _ctx: 55)
    monkeypatch.setattr(
        execution_policy, "_resolve_fresh_child_provenance",
        lambda **_kw: SimpleNamespace(
            route=SimpleNamespace(is_planning=True), parent_plan_context=None,
            stage_id="api", parent_issue=55,
        ),
    )
    verdict = {"failure": None}
    monkeypatch.setattr(
        child_plan_binding, "_child_plan_admissibility_failure",
        lambda *_a, **_k: verdict["failure"],
    )
    inputs = build_plan_approval_inputs(
        runner, config=_config, issue_context=issue, parent_issue_context=parent,
    )
    assert inputs.derivable and inputs.parent_requirement_ids == ()
    staged_flags = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex", min_reviewers=2,
    )
    assert classify_plan_approval(
        issue.comments, config=staged_flags, issue_number=56, trusted_actor=_PLAN_ACTOR,
        inputs=inputs,
    ).approved
    # Approved: the differing planning flags are never compared at preflight.
    enforce_issue_board_floor(
        staged_flags, issue.comments, issue_number=56, runner=runner, planning=True,
        inputs=inputs,
    )
    verdict["failure"] = "inherited row weakened"
    assert classify_plan_approval(
        issue.comments, config=staged_flags, issue_number=56, trusted_actor=_PLAN_ACTOR,
        inputs=inputs,
    ).state == "unfinished"


def test_approved_staged_plan_resumed_under_default_flags_is_not_reopened(tmp_path):
    """Row plan-approved-board-change: approved primary-then-panel plan, default planning flags."""
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex", quiet=False,
    )
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    assert run_issue_loop(runner, issue_number=56, config=staged, plan_first=True) == 0
    start, comments = len(runner.commands), len(runner.issue_comments)
    default_flags = _plan_cfg(tmp_path, ("codex", "gemini"), min_reviewers=2)
    assert run_issue_loop(runner, issue_number=56, config=default_flags, plan_first=True) == 0
    assert _plan_agents(runner, start) == []
    assert _plan_links(runner) == []
    assert not any(
        "scheduler contract changed" in c["body"] for c in runner.issue_comments[comments:]
    )


def test_coder_only_legacy_history_keeps_the_all_reviewers_root(tmp_path):
    """Item-7: coder-only legacy history resumed as primary-then-panel is drift."""
    runner = _planned_round_one(tmp_path)
    _strip_plan_stamps(runner)
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
    )
    start, comments = len(runner.commands), len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
    assert _plan_agents(runner, start) == [] and len(runner.issue_comments) == comments


def test_signed_provider_floor_reduction_with_explicitly_reduced_flags(tmp_path):
    """Item-9.s6: signed provider-floor reduction accepted with explicitly reduced flags."""
    runner = _planned_round_one(tmp_path)
    _append_issue_comment(runner, format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    ))
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    start = len(runner.commands)
    reduced = _plan_cfg(tmp_path, ("codex",), min_distinct_providers=2)
    assert run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["codex"]
    assert _plan_links(runner) == []


def test_signed_amendment_then_link_resolves_one_chain_through_resume(tmp_path):
    """Item-9.s6 / row plan-link-crash-resume: a signed amendment followed by a link."""
    runner = _planned_round_one(tmp_path)
    _append_issue_comment(runner, format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    ))
    runner.codex_outputs.append(structured_plan_review(
        state="blocking", summary="x", blocking_plan_issues=["Name the owner."],
    ))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(tmp_path, ("codex",)),
                       plan_first=True)
    amendment_digest = next(
        r.metadata.reviewer_board_amendment_digest for r in _plan_records(runner)
        if r.metadata.reviewer_board_amendment_digest is not None
    )
    still_open = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(structured_plan_review(prior_plan_item_dispositions=still_open))
    runner.antigravity_outputs.append(structured_plan_review(
        reviewer="Google Antigravity", prior_plan_item_dispositions=still_open,
    ))
    widened = _plan_cfg(tmp_path, ("codex", "antigravity"))
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["predecessor"] == amendment_digest
    assert payload["added"] == ["Antigravity"] and payload["removed"] == []
    # A rerun rebuilds the identical lineage and posts nothing new.
    comments = len(runner.issue_comments)
    assert run_issue_loop(runner, issue_number=56, config=widened, plan_first=True) == 0
    assert len(runner.issue_comments) == comments


# --- Round 2 review: keyless fallback, coder dispositions, cleared history


def _issue_context(runner, *, requirements=()):
    from coding_review_agent_loop.github import IssueComment, IssueContext

    return IssueContext(
        number=56, repo="OWNER/REPO", title="t", body="b", url=None,
        comments=tuple(
            IssueComment(author=c["author"]["login"], created_at=c["createdAt"], body=c["body"],
                         author_id=c.get("_rest_author_id"))
            for c in runner.issue_comments
        ),
        human_requirements=tuple(requirements),
    )


def _rewrite_plan_records(runner, predicate, *, body_suffix="", **changes):
    """Rewrite every plan round record whose metadata matches ``predicate``."""
    import re

    from coding_review_agent_loop.round_state import _decode_round_metadata

    rewritten = 0
    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment["body"])
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow != "plan" or not predicate(metadata):
            continue
        body = comment["body"][: match.start()].rstrip("\n") + body_suffix
        comment["body"] = _attach_round_metadata(body, dataclasses.replace(metadata, **changes))
        rewritten += 1
    assert rewritten


def _strip_requirement_keys(runner, *, acknowledged=False):
    """Rewrite the reviewer records as keyless legacy approvals.

    A legacy approval body carried its acknowledgement as the rendered marker.
    """
    _rewrite_plan_records(
        runner, lambda metadata: metadata.role == "reviewer",
        body_suffix="\n\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if acknowledged else "",
        plan_candidate_key=None, surfaced_reviewer_requirement_ids=(),
    )


def _classify(runner, config, requirements=()):
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    return classify_plan_approval(
        _plan_comments(runner), config=config, issue_number=56,
        human_requirements=tuple(requirements), trusted_actor=_PLAN_ACTOR,
    )


def _recover(runner, config, requirements=()):
    """Managed-CI no-handoff recovery and strict qualification's shared gate."""
    from coding_review_agent_loop.plan_verification import require_recoverable_plan_approval

    issue = _issue_context(runner, requirements=requirements)
    require_recoverable_plan_approval(
        None, config=config, issue_context=issue, comments=issue.comments,
        trusted_actor=_PLAN_ACTOR, error_message="not recoverable",
    )


def test_keyless_acknowledged_approval_is_not_reused_after_withdrawal(tmp_path, monkeypatch):
    """Item-2: a keyless legacy approval acknowledging a withdrawn requirement is stale."""
    requirement = _signed_requirement()
    _patch_requirements(monkeypatch, [requirement])
    runner, config = _approved_plain_plan(tmp_path, requirement=requirement)
    _strip_requirement_keys(runner, acknowledged=True)
    # While the requirement stands, the legacy fallback binds the acknowledged approvals.
    assert _classify(runner, config, [requirement]).state == "approved"
    _recover(runner, config, [requirement])
    _patch_requirements(monkeypatch, [])
    withdrawn = _classify(runner, config)
    assert withdrawn.state == "unfinished"
    assert "stale requirement set" in withdrawn.reason
    with pytest.raises(AgentLoopError, match="not recoverable"):
        _recover(runner, config)
    # A public rerun never reuses them: both reviewers review again.
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    start = len(runner.commands)
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass
    assert {"codex", "gemini"} <= set(_plan_agents(runner, start))


def test_missing_coder_dispositions_refuse_keyed_approvals_and_recovery(tmp_path, monkeypatch):
    """Item-11: keyed approvals never stand in for the coder's requirement dispositions."""
    requirement = _signed_requirement()
    _patch_requirements(monkeypatch, [requirement])
    runner, config = _approved_plain_plan(tmp_path, requirement=requirement)
    assert _classify(runner, config, [requirement]).state == "approved"
    _recover(runner, config, [requirement])
    # An interrupted run left a plan record without the requirement disposition;
    # the reviewers' keyed approvals are unchanged.
    _rewrite_plan_records(
        runner, lambda metadata: metadata.role == "coder",
        raw_structured_coder_response=structured_v1_plan_state(),
    )
    interrupted = _classify(runner, config, [requirement])
    assert interrupted.state == "unfinished"
    assert "lacks complete requirement dispositions" in interrupted.reason
    with pytest.raises(AgentLoopError, match="not recoverable"):
        _recover(runner, config, [requirement])
    # A restart is never shortcut as approved, so no approved outcome follows.
    start = len(runner.commands)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _plan_agents(runner, start)


def _semantic_patch(runner, *, round_number, summary, item_id):
    """A plan revision patch against the round's authenticated plan state."""
    from coding_review_agent_loop.plan_assembly import (
        AuthenticatedPlanState,
        decode_assembled_plan_sidecar,
        hydrate_authenticated_plan_state,
    )
    from coding_review_agent_loop.response_validation import validate_structured_plan_state

    if round_number == 1:
        base = AuthenticatedPlanState.from_plan(
            validate_structured_plan_state(structured_v1_plan_state()), round_number=1,
        )
    else:
        coder = [r for r in _plan_records(runner) if r.metadata.role == "coder"][-1].metadata
        assert coder.round_number == round_number
        base = hydrate_authenticated_plan_state(
            decode_assembled_plan_sidecar(coder.assembled_plan_sidecar)
        )
    patch = {
        "schema_version": 1, "kind": "plan_revision_patch", "semantic_patch_contract_version": 1,
        "state": "blocking", "summary": summary,
        "prior_plan_item_dispositions": [
            {"item_id": item_id, "disposition": "resolved", "rationale": "Addressed."},
        ],
        "base_round_number": round_number, "base_state_identity": base.state_identity,
        "operations": [{"op": "replace", "field": "summary", "value": summary}],
    }
    return json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


def test_completed_staged_plan_with_cleared_history_is_approved(tmp_path):
    """Item-13: an earlier finding resolved and absent from the final ledger is cleared
    history, not a lost obligation, live, at restart and in recovery.  The semantic
    patches inherit their (empty) requirement dispositions from the assembled state."""
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
    )
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
    # Round 3 resolves item-1; Gemini raises same-plan item-2.
    runner.claude_outputs.append(
        _semantic_patch(runner, round_number=1, summary="Name the owner.", item_id="item-1")
    )
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review(
        "gemini", state="blocking", same_plan_followups=["Mention the owner in the summary."],
    ))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
    # Round 4 carries only item-2: item-1 was cleared by recorded history.
    resolved_2 = [{"item_id": "item-2", "disposition": "resolved"}]
    runner.claude_outputs.append(
        _semantic_patch(runner, round_number=3, summary="Mention the owner.", item_id="item-2")
    )
    runner.codex_outputs.append(_review("codex", dispositions=resolved_2))
    runner.gemini_outputs.append(_review("gemini", dispositions=resolved_2))
    assert run_issue_loop(runner, issue_number=56, config=staged, plan_first=True) == 0
    final_coder = [r for r in _plan_records(runner) if r.metadata.role == "coder"][-1].metadata
    assert final_coder.response_form == "semantic-patch-v1"
    assert [item.item_id for item in final_coder.prior_items] == ["item-2"]
    assert _classify(runner, staged).state == "approved"
    _recover(runner, staged)
    # An unchanged restart does not reopen the plan.
    start, comments = len(runner.commands), len(runner.issue_comments)
    assert run_issue_loop(runner, issue_number=56, config=staged, plan_first=True) == 0
    assert _plan_agents(runner, start) == []


def test_staged_child_public_entry_merges_parent_requirements(tmp_path, monkeypatch):
    """Item-3: the authoritative parent's signed requirements join the planning set at
    the public entry, bind every approval, and gate restart and recovery."""
    import coding_review_agent_loop.execution_policy as execution_policy
    import coding_review_agent_loop.plan_first_loop as plan_first_loop
    from coding_review_agent_loop.github import HumanReviewRequirement, IssueContext
    from coding_review_agent_loop.plan_verification import (
        build_plan_approval_inputs,
        classify_plan_approval,
        require_recoverable_plan_approval,
    )

    parent_requirement = HumanReviewRequirement(
        source_type="Issue body", author="maintainer", created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/55", body="Preserve the parent audit trail.",
    )
    parent_requirements = [parent_requirement]
    for module in (execution_policy, plan_first_loop):
        monkeypatch.setattr(
            module, "_infer_staged_parent_issue",
            lambda context: 55 if context.number == 56 else None,
        )
    monkeypatch.setattr(execution_policy, "_resolve_fresh_child_provenance", lambda **_kw: None)
    real = discuss_loop.get_issue_context

    def with_parent(runner_arg, *, config, issue_number):
        if issue_number == 55:
            return IssueContext(
                number=55, repo="OWNER/REPO", title="Parent", body="Parent body", url=None,
                comments=(), human_requirements=tuple(parent_requirements),
            )
        return real(runner_arg, config=config, issue_number=issue_number)

    import coding_review_agent_loop.github as github

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy_module, child_plan_binding_module, pr_loop_support_module, pr_loop_module, plan_first_loop_module, issue_loop), "get_issue_context", with_parent)
    monkeypatch.setattr(github, "get_issue_context", with_parent)
    runner, config = _approved_plain_plan(tmp_path, requirement=parent_requirement)
    parent_id = parent_requirement.requirement_id
    assert any(parent_requirement.body in prompt for prompt in (
        cmd[-1] for cmd, _cwd in runner.commands if cmd[0] == "claude"
    ))
    reviews = [r.metadata for r in _plan_records(runner) if r.metadata.role == "reviewer"]
    assert reviews and all(m.parent_requirement_ids == (parent_id,) for m in reviews)
    assert all(m.surfaced_reviewer_requirement_ids == (parent_id,) for m in reviews)

    def classify():
        issue = _issue_context(runner)
        inputs = build_plan_approval_inputs(runner, config=config, issue_context=issue)
        assert inputs.parent_requirement_ids == tuple(r.requirement_id for r in parent_requirements)
        return classify_plan_approval(
            issue.comments, config=config, issue_number=56, trusted_actor=_PLAN_ACTOR,
            inputs=inputs,
        )

    assert classify().state == "approved"
    def recover():
        issue = _issue_context(runner)
        require_recoverable_plan_approval(
            runner, config=config, issue_context=issue, comments=issue.comments,
            trusted_actor=_PLAN_ACTOR, error_message="not recoverable",
        )

    recover()
    # An unchanged restart is not reopened.
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert _plan_agents(runner, start) == []
    # The parent requirement changes: every approval is stale, recovery refuses,
    # and a restart reviews again instead of reusing them.
    parent_requirements[:] = [dataclasses.replace(parent_requirement, body="Changed parent rule.")]
    assert classify().state in {"unfinished", "undetermined"}
    with pytest.raises(AgentLoopError, match="not recoverable"):
        recover()
    start = len(runner.commands)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _plan_agents(runner, start)


# --- Round 3 review: frozen signed authorizations, refreshed lineage, epoch history


def test_signed_reduction_survives_a_later_model_link_into_implementation(tmp_path):
    """Item-14: amendment -> model-only link -> approval keeps the frozen signed
    authorization for both floor measures at the implementation and PR handoff."""
    from coding_review_agent_loop.plan_verification import (
        derive_plan_verification_context,
        enforce_issue_board_floor,
    )
    from coding_review_agent_loop.reviewer_seats import resolve_plan_handoff_board

    floor = dict(min_reviewers=3, min_distinct_providers=3)
    flash_a = _seat(tmp_path, "flash", "antigravity", "Model A")
    runner = _plan_runner(claude_outputs=[structured_v1_plan_state()])
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=_plan_cfg(
            tmp_path, ("codex", "gemini", flash_a), reviewer_seats=(flash_a,),
            review_parallel=False, **floor,
        ), plan_first=True)
    _append_issue_comment(runner, format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini", "flash"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    ))
    runner.codex_outputs.append(structured_plan_review(
        state="blocking", summary="x", blocking_plan_issues=["Name the owner."],
    ))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model A)",
    ))
    signed = _plan_cfg(
        tmp_path, ("codex", flash_a), reviewer_seats=(flash_a,), review_parallel=False, **floor,
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=signed, plan_first=True)
    # A model-only link on the signed two-seat board, then approval by it.
    flash_b = _seat(tmp_path, "flash", "antigravity", "Model B")
    runner.codex_outputs.append(_review("codex"))
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="flash (Google Antigravity: Model B)",
        prior_plan_item_dispositions=_RESOLVED_1,
    ))
    rebound = _plan_cfg(
        tmp_path, ("codex", flash_b), reviewer_seats=(flash_b,), review_parallel=False, **floor,
    )
    assert run_issue_loop(runner, issue_number=56, config=rebound, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    assert payload["rebound"] == ["flash"]
    comments = _issue_context(runner).comments
    context = derive_plan_verification_context(
        comments, issue_number=56, trusted_actor=_PLAN_ACTOR,
    )
    # The trailing element is the link, yet the signed segment's frozen
    # authorization (three seats and providers to two) is kept.
    assert context.signed_amendment_count == 0
    (authorization,) = context.signed_authorizations
    assert len(authorization.seats) == 2
    assert authorization.lowered_reviewers and authorization.lowered_providers
    handoff = resolve_plan_handoff_board(rebound, context)
    assert handoff.inherited and handoff.signed_authorizations == (authorization,)
    # The implementation preflight accepts the inherited signed board under both
    # floor measures...
    enforce_issue_board_floor(rebound, comments, issue_number=56, runner=runner)
    # ...but an operator-chosen two-seat board gets no signed exception.
    other = _plan_cfg(tmp_path, ("codex", "gemini"), **floor)
    with pytest.raises(AgentLoopError, match="below the floor of 3"):
        enforce_issue_board_floor(other, comments, issue_number=56, runner=runner)


def _inject_during_first(runner, agent, body):
    """Append an operator comment right after ``agent``'s first turn of the run."""
    real = runner._run_with_log_locked
    injected = []

    def run_with_log_locked(args, **kwargs):
        result = real(args, **kwargs)
        if not injected and [str(arg) for arg in args][:1] == [agent]:
            injected.append(True)
            _append_issue_comment(runner, body)
        return result

    runner._run_with_log_locked = run_with_log_locked
    return injected


def _inject_after_record(runner, predicate, body):
    """Append an operator comment right after a plan record matching ``predicate`` posts."""
    real = runner._run_locked
    injected = []

    def run_locked(args, **kwargs):
        result = real(args, **kwargs)
        if not injected and any(predicate(r.metadata) for r in _plan_records(runner)):
            injected.append(True)
            _append_issue_comment(runner, body)
        return result

    runner._run_locked = run_locked
    return injected


def test_forged_link_arriving_mid_plan_refuses_approval(tmp_path):
    """Item-16: a forged plan link posted while reviewers run never lets the
    approval boundary report approval."""
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    injected = _inject_during_first(
        runner, "codex",
        f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: e30= -->",
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"), review_parallel=False)
    # Refused once the round's reviews are in, before any next-round record or
    # approval.
    with pytest.raises(AgentLoopError, match="history is inconsistent before the next round"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert injected
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    with pytest.raises(AgentLoopError, match="lineage is inconsistent"):
        classify_plan_approval(
            _plan_comments(runner), config=config, issue_number=56, trusted_actor=_PLAN_ACTOR,
        )


def test_signed_amendment_arriving_mid_plan_stops_before_the_next_round(tmp_path):
    """Item-16: a signed amendment recorded during round 1 stops the run before the
    next round's planner turn or record; a clean rerun applies it."""
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(
            state="blocking", summary="x", blocking_plan_issues=["Name the owner."],
        )],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    # Posted right after round 1's last review record: nothing the run writes
    # later can be published under the old board.
    injected = _inject_after_record(
        runner, lambda m: m.role == "reviewer" and m.agent == "Gemini",
        format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("Codex", "Gemini"),
        policy="all-reviewers", primary_reviewer=None,
        removed_reviewers=("Gemini",), effective_from_round=1,
        rationale="Gemini quota exhausted.",
    ))
    config = _plan_cfg(tmp_path, ("codex", "gemini"), review_parallel=False)
    with pytest.raises(
        AgentLoopError,
        match="board change was recorded while this run was active .*before the next round",
    ):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert injected
    # Round 1's reviews ran; no planner turn and no round-2 record followed.
    assert _plan_agents(runner) == ["claude", "codex", "gemini"]
    assert max(r.metadata.round_number for r in _plan_records(runner)) == 1
    # A clean rerun resolves and activates the amendment: the approving Gemini
    # is removed, and Codex's finding is revised and resolved by Codex alone.
    runner.claude_outputs.append(
        _semantic_patch(runner, round_number=1, summary="Name the owner.", item_id="item-1")
    )
    runner.codex_outputs.append(_review("codex"))
    start = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert _plan_agents(runner, start) == ["claude", "codex"]
    round_two = [r.metadata for r in _plan_records(runner) if r.metadata.round_number == 2]
    assert round_two and all(m.reviewer_board_amendment_digest is not None for m in round_two)
    assert not any(m.agent == "Gemini" for m in round_two)


def _epoch_plan_losing_a_pre_epoch_finding(tmp_path):
    """A primary change (Codex -> Antigravity) whose post-epoch ledger lost the
    secondary's pre-epoch finding: it is neither carried nor dispositioned."""
    def staged(primary):
        return _plan_cfg(
            tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary,
        )

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
        antigravity_outputs=[structured_plan_review(state="approved", reviewer="Google Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    runner.antigravity_outputs.append(structured_plan_review(
        state="approved", reviewer="Google Antigravity", prior_plan_item_dispositions=_RESOLVED_1,
    ))
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    config = staged("antigravity")
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    (payload,) = _plan_link_payloads(runner)
    epoch = payload["effective_from_round"]
    assert _classify(runner, config).state == "approved"
    # Every post-epoch record drops item-1 from its ledger and dispositions.
    _rewrite_plan_records(
        runner, lambda metadata: metadata.round_number >= epoch and metadata.role != "coder",
        prior_items=(), dispositions=(),
    )
    published = [
        record.metadata for record in _plan_records(runner)
        if any(item.item_id == "item-1" for item in record.metadata.new_items)
    ]
    assert [(m.agent, m.round_number < epoch) for m in published] == [("Gemini", True)]
    return runner, config


def test_pre_epoch_finding_lost_after_a_primary_change_is_never_approved(tmp_path):
    """Item-8: the lost-obligation rule reads complete publication history, so a
    primary epoch never hides an undispositioned pre-epoch finding."""
    runner, config = _epoch_plan_losing_a_pre_epoch_finding(tmp_path)
    lost = _classify(runner, config)
    assert lost.state == "undetermined"
    assert "unreconstructible plan obligation(s) item-1" in lost.reason
    with pytest.raises(AgentLoopError, match="not recoverable"):
        _recover(runner, config)
    # A public restart never reports approval over the lost finding.
    runner.codex_outputs.append(_review("codex", dispositions=()))
    runner.gemini_outputs.append(_review("gemini", dispositions=()))
    runner.antigravity_outputs.append(_review("antigravity", dispositions=()))
    with pytest.raises(AgentLoopError, match="item-1"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_pre_epoch_finding_lost_after_a_primary_change_refuses_managed_recovery(
    tmp_path, monkeypatch, entry
):
    """Item-8: managed-CI recovery of the same history refuses before any PR turn."""
    from test_issue_pr_handoff import _agent_commands, _lineage_strict_run

    runner, _config = _epoch_plan_losing_a_pre_epoch_finding(tmp_path)
    pr_runner, pr_config, merges, run_pr_loop = _lineage_strict_run(
        tmp_path, monkeypatch, list(runner.issue_comments), entry=entry,
    )
    with pytest.raises(AgentLoopError, match="unreconstructible plan obligation\\(s\\) item-1"):
        run_pr_loop(pr_runner, pr_number=77, config=pr_config)
    assert merges == []
    assert _agent_commands(pr_runner) == []


def _unanimous_with_open_same_plan_item(tmp_path, kind="same-plan"):
    """Round 2's reviewers both approve, but neither dispositions Gemini's carried
    same-plan item-1 (their resolutions are dropped from the records).

    ``kind="machine"`` records item-1 as a machine obligation instead, which no
    planning participant can clear."""
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            same_plan_followups=["Mention the owner in the summary."],
        )],
    )
    config = _plan_cfg(tmp_path, ("codex", "gemini"), review_parallel=False)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    runner.claude_outputs.append(
        _semantic_patch(runner, round_number=1, summary="Mention the owner.", item_id="item-1")
    )
    runner.codex_outputs.append(_review("codex"))
    runner.gemini_outputs.append(_review("gemini"))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert _classify(runner, config).state == "approved"
    _rewrite_plan_records(
        runner,
        lambda metadata: metadata.role == "reviewer" and metadata.round_number == 2,
        dispositions=(),
    )
    if kind == "machine":
        def machine(items):
            return tuple(
                dataclasses.replace(
                    item, authority="machine", obligation_kind="github-pr-checks",
                    lifecycle="qualification_ready", failed_head_sha="old-head",
                    candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", obligation_identity="github-pr-checks:item-1",
                )
                if item.item_id == "item-1" else item
                for item in items
            )

        for metadata in [r.metadata for r in _plan_records(runner)]:
            if any(i.item_id == "item-1" for i in (*metadata.prior_items, *metadata.new_items)):
                _rewrite_plan_records(
                    runner,
                    lambda m, target=(metadata.role, metadata.agent, metadata.round_number, metadata.phase): (
                        (m.role, m.agent, m.round_number, m.phase) == target
                    ),
                    prior_items=machine(metadata.prior_items),
                    new_items=machine(metadata.new_items),
                )
    reviews = [
        r.metadata for r in _plan_records(runner)
        if r.metadata.role == "reviewer" and r.metadata.round_number == 2
    ]
    assert {m.state for m in reviews} == {"approved"} and len(reviews) == 2
    assert all([item.status for item in m.prior_items] == ["same-plan"] for m in reviews)
    assert all(
        item.is_machine_obligation == (kind == "machine")
        for m in reviews for item in m.prior_items
    )
    return runner, config


@pytest.mark.parametrize("kind", ["same-plan", "machine"])
def test_unanimous_approvals_with_an_open_same_plan_item_resume_planning(tmp_path, kind):
    """Item-9.s4: the open same-plan or machine obligation keeps the plan
    unfinished at classification and recovery.  A restart resumes into the
    revision for a same-plan item, and stops with the machine-obligation
    diagnostic (no planning participant can clear it) for a machine one."""
    runner, config = _unanimous_with_open_same_plan_item(tmp_path, kind)
    # The shared complete-approval check finds the undispositioned carried item.
    assert _classify(runner, config).state == "unfinished"
    with pytest.raises(AgentLoopError, match="not recoverable"):
        _recover(runner, config)
    start, comments = len(runner.commands), len(runner.issue_comments)
    refusal = (
        "scripted agent output exhausted" if kind == "same-plan"
        else "item-1 .*are machine obligations that no planning participant can clear"
    )
    with pytest.raises(AgentLoopError, match=refusal):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    # No reviewer is re-run and nothing is approved: the planner revises a
    # same-plan item; a machine obligation stops before any agent turn.
    assert _plan_agents(runner, start) == (["claude"] if kind == "same-plan" else [])
    assert _plan_links(runner) == []


@pytest.mark.parametrize("kind", ["same-plan", "machine"])
@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_unanimous_approvals_with_an_open_same_plan_item_refuse_managed_recovery(
    tmp_path, monkeypatch, entry, kind
):
    """Item-9.s4: managed-CI recovery never binds the plan with an open obligation."""
    from test_issue_pr_handoff import _agent_commands, _lineage_strict_run

    runner, _config = _unanimous_with_open_same_plan_item(tmp_path, kind)
    pr_runner, pr_config, merges, run_pr_loop = _lineage_strict_run(
        tmp_path, monkeypatch, list(runner.issue_comments), entry=entry,
    )
    with pytest.raises(
        AgentLoopError,
        match="incomplete canonical plan approval|without a complete canonical reviewer approval",
    ):
        run_pr_loop(pr_runner, pr_number=77, config=pr_config)
    assert merges == []
    assert _agent_commands(pr_runner) == []


def test_signed_amendment_after_the_primary_approval_stops_before_the_phase_advance(tmp_path):
    """Item-16 (phase-advance branch): an amendment recorded after the primary's
    approval stops the run before the reviewer-only phase-advance record."""
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    injected = _inject_after_record(
        runner, lambda m: m.role == "reviewer" and m.agent == "Codex",
        format_reviewer_board_amendment_comment(
            flow="plan", issue=56, pr_number=None,
            original_required_reviewers=("Codex", "Gemini", "Antigravity"),
            policy="primary-then-panel", primary_reviewer="Codex",
            removed_reviewers=("Antigravity",), effective_from_round=2,
            rationale="Antigravity quota exhausted.",
        ),
    )
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex", review_parallel=False,
    )
    with pytest.raises(
        AgentLoopError,
        match="board change was recorded while this run was active .*before the next round",
    ):
        run_issue_loop(runner, issue_number=56, config=staged, plan_first=True)
    assert injected
    assert _plan_agents(runner) == ["claude", "codex"]
    assert not any(r.metadata.phase == "plan-phase-advance" for r in _plan_records(runner))
    assert max(r.metadata.round_number for r in _plan_records(runner)) == 1


def _legacy_keyless_plan_with_handoff(tmp_path, monkeypatch, requirement, *, acknowledged=True):
    """Unstamped legacy plain history: keyless approvals of a plan whose dispositions
    name ``requirement``, plus an issue-side handoff bound to the plan hash.  Without
    the acknowledgement the fallback cannot prove the binding."""
    from coding_review_agent_loop.decomposition import format_one_shot_impl_handoff_comment
    from coding_review_agent_loop.round_state import _plan_subject

    _patch_requirements(monkeypatch, [requirement])
    runner, config = _approved_plain_plan(tmp_path, requirement=requirement)
    _strip_plan_stamps(runner)
    _strip_requirement_keys(runner, acknowledged=acknowledged)
    assert all(r.metadata.plan_board_root is None for r in _plan_records(runner))
    unanchored = _classify(runner, config, [requirement])
    # Posters prove membership only: without the handoff it is undetermined.
    assert unanchored.state == "undetermined"
    plan = next(r.metadata.canonical_plan for r in _plan_records(runner) if r.metadata.role == "coder")
    _append_issue_comment(runner, format_one_shot_impl_handoff_comment(
        parent_issue=56, mode="implement-one-shot", plan_hash=unanchored.plan_hash,
        plan_subject=_plan_subject(plan), pr_number=77, pr_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
    ), login="agent-loop", author_id=_PLAN_ACTOR[1])
    classified = _classify(runner, config, [requirement])
    if acknowledged:
        assert classified.state == "approved"
    else:
        assert (classified.state, classified.reason) == (
            "undetermined", "an approval's requirement binding is unproven",
        )
    return runner, config


@pytest.mark.parametrize(
    ("evidence", "mode"),
    [("matching", "plan-only"), ("unprovable", "plan-only"), ("unprovable", "implement-one-shot")],
)
def test_legacy_keyless_fallback_through_the_public_resume(tmp_path, monkeypatch, evidence, mode):
    """Item-9.s3: a public ``run_issue_loop`` resume of legacy keyless history.

    Matching evidence (versioned dispositions naming the surfaced requirement and
    acknowledged approvals) completes with no agent turn and no new record.
    Unprovable evidence (approvals without the acknowledgement) never takes the
    approval shortcut or implements, in either execution mode: the run asks the
    planner for an unapproved revision before any reviewer turn.  Full-board
    scheduling never reads the carried approval evidence on either path.
    """
    import coding_review_agent_loop.plan_first_loop as plan_first_loop

    requirement = _signed_requirement()
    runner, config = _legacy_keyless_plan_with_handoff(
        tmp_path, monkeypatch, requirement, acknowledged=evidence == "matching",
    )
    config = dataclasses.replace(config, plan_execution_mode=mode)

    def evidence_read(*_args, **_kwargs):
        raise AssertionError("full-board scheduling read carried approval evidence")

    monkeypatch.setattr(plan_first_loop, "_carried_plan_approvals", evidence_read)
    start, records_before = len(runner.commands), len(_plan_records(runner))
    comments_before = len(runner.issue_comments)
    if evidence == "matching":
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
        assert _plan_agents(runner, start) == []
        assert len(_plan_records(runner)) == records_before
        assert _plan_links(runner) == []
        return
    # The approvals cannot prove the binding: the run asks the planner for an
    # unapproved revision before any reviewer turn, approval or implementation.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _plan_agents(runner, start) == ["claude"]
    planner_prompt = [cmd[-1] for cmd, _cwd in runner.commands[start:] if cmd[0] == "claude"][0]
    assert "This is an eligible unapproved planning revision." in planner_prompt
    assert len(_plan_records(runner)) == records_before
    new_comments = [c["body"] for c in runner.issue_comments[comments_before:]]
    assert not any("plan approved" in body.lower() for body in new_comments)
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands[start:])
    assert _plan_links(runner) == []


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
@pytest.mark.parametrize("requirements", ["matching", "withdrawn"])
def test_legacy_keyless_fallback_through_managed_recovery(
    tmp_path, monkeypatch, entry, requirements
):
    """Item-9.s3: legacy keyless history with matching dispositions and acknowledged
    approvals binds and qualifies through managed-CI recovery; after the requirement
    is withdrawn, the same recovery refuses before any PR reviewer or merge."""
    import coding_review_agent_loop.pr_loop as pr_loop
    import coding_review_agent_loop.pr_loop_support as pr_loop_support
    from test_issue_pr_handoff import _agent_commands, _lineage_strict_run

    requirement = _signed_requirement()
    runner, _config = _legacy_keyless_plan_with_handoff(tmp_path, monkeypatch, requirement)
    current = [requirement] if requirements == "matching" else []
    for module in (pr_loop, pr_loop_support):
        real = module.get_issue_context

        def with_current(*args, _real=real, **kwargs):
            context = _real(*args, **kwargs)
            if context.number != 56:
                return context
            return dataclasses.replace(context, human_requirements=tuple(current))

        monkeypatch.setattr(module, "get_issue_context", with_current)
    pr_runner, pr_config, merges, run_pr_loop = _lineage_strict_run(
        tmp_path, monkeypatch, list(runner.issue_comments), entry=entry,
    )
    if requirements == "matching":
        # Recovery binds the plan and PR review starts on board C.  (The signed
        # requirement also applies to PR review, whose acknowledgement turns
        # are not scripted here, so the run stops there.)
        with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
            run_pr_loop(pr_runner, pr_number=77, config=pr_config)
        assert {cmd[0] for cmd in _agent_commands(pr_runner)} >= {"claude"}
        assert merges == []
        return
    with pytest.raises(
        AgentLoopError,
        match="incomplete canonical plan approval|without a complete canonical reviewer approval"
        "|not completely approved",
    ):
        run_pr_loop(pr_runner, pr_number=77, config=pr_config)
    assert merges == []
    assert _agent_commands(pr_runner) == []



def test_primary_change_after_an_old_primary_streak_reopens_the_panel_for_the_owner(tmp_path):
    """Item-9.s5 / row plan-primary-change-midrun: the old primary blocks two rounds,
    approves, and the panel opens with a secondary finding; the primary then moves to
    Antigravity under --plan-primary-stall-rounds 3.  The new primary's first block
    never trips the stall stop (2 + 1 would), the carried finding keeps its owner and
    blocks approval, and the panel reopens after the new primary's approval."""
    def staged(primary):
        return _plan_cfg(
            tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary, plan_primary_stall_rounds=3, review_parallel=False,
            max_rounds=10,
        )

    def last_coder_round():
        return [r for r in _plan_records(runner) if r.metadata.role == "coder"][-1].metadata.round_number

    def resolved(*item_ids):
        return [{"item_id": item_id, "disposition": "resolved"} for item_id in item_ids]

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(
            state="blocking", summary="x", blocking_plan_issues=["Name the owner."],
        )],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    runner.claude_outputs.append(_semantic_patch(
        runner, round_number=last_coder_round(), summary="Owner.", item_id="item-1",
    ))
    runner.codex_outputs.append(_review(
        "codex", state="blocking", blocking_plan_issues=["Name the deadline."],
    ))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    runner.claude_outputs.append(_semantic_patch(
        runner, round_number=last_coder_round(), summary="Deadline.", item_id="item-2",
    ))
    runner.codex_outputs.append(_review("codex", dispositions=resolved("item-2")))
    runner.gemini_outputs.append(_review(
        "gemini", dispositions=[], state="blocking", blocking_plan_issues=["Name the rollback."],
    ))
    runner.antigravity_outputs.append(_review("antigravity", dispositions=[]))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    records = [r.metadata for r in _plan_records(runner)]
    # The old primary's blocking streak, its approval, and the panel opening
    # with Gemini's post-opening finding.
    assert [(m.round_number, m.state) for m in records if m.agent == "Codex"] == [
        (1, "blocking"), (2, "blocking"), (3, "approved"),
    ]
    assert any(m.phase == "plan-phase-advance" and m.round_number == 4 for m in records)
    assert [i.item_id for m in records if m.agent == "Gemini" for i in m.new_items] == ["item-3"]

    # Primary change: the new primary blocks once.  Counting the old primary's
    # streak would reach the stall threshold of 3; the run revises instead.
    start = len(runner.commands)
    runner.antigravity_outputs.append(_review(
        "antigravity", state="blocking", blocking_plan_issues=["Name the metric."],
        dispositions=[{"item_id": "item-3", "disposition": "blocking", "note": "Gemini owns it."}],
    ))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("antigravity"), plan_first=True)
    assert _plan_agents(runner, start) == ["agy", "claude"]
    (payload,) = _plan_link_payloads(runner)
    assert payload["old_board"]["primary"] == "Codex"
    assert payload["new_board"]["primary"] == "Antigravity"
    epoch = payload["effective_from_round"]
    prelaunch = [
        r.metadata for r in _plan_records(runner)
        if r.metadata.phase == "scheduler-prelaunch" and r.metadata.round_number == epoch
    ]
    assert prelaunch[0].scheduler_selected_reviewers == ("Antigravity",)
    # The carried secondary finding keeps its owner into the primary-only round.
    assert [(i.item_id, i.reviewer) for i in prelaunch[0].prior_items] == [("item-3", "Gemini")]

    # The new primary approves; approval still waits for the reopened panel.
    start = len(runner.commands)
    runner.claude_outputs.append(_semantic_patch(
        runner, round_number=last_coder_round(), summary="Metric.", item_id="item-4",
    ))
    runner.antigravity_outputs.append(_review("antigravity", dispositions=resolved("item-4", "item-3")))
    runner.gemini_outputs.append(_review("gemini", dispositions=[]))
    runner.codex_outputs.append(_review("codex", dispositions=[]))
    assert run_issue_loop(runner, issue_number=56, config=staged("antigravity"), plan_first=True) == 0
    assert _plan_agents(runner, start) == ["claude", "agy", "codex", "gemini"]
    revision = [r.metadata for r in _plan_records(runner) if r.metadata.role == "coder"][-1]
    assert {(i.item_id, i.reviewer) for i in revision.prior_items} >= {("item-3", "Gemini")}
    final = [r.metadata for r in _plan_records(runner)]
    reopened = max(m.round_number for m in final)
    assert any(m.phase == "plan-phase-advance" and m.round_number == reopened for m in final)
    assert {m.agent for m in final if m.round_number == reopened and m.role == "reviewer"} == {
        "Codex", "Gemini",
    }
    assert len(_plan_links(runner)) == 1


@pytest.mark.parametrize("next_primary", ["codex", "antigravity"])
def test_primary_change_mid_streak_restarts_the_stall_count(tmp_path, next_primary):
    """Item-9.s5: Codex blocks two consecutive primary rounds under
    --plan-primary-stall-rounds 2.  Resuming with Codex trips the stall stop before
    round 3; after a primary change, the new primary starts a fresh streak, reviews
    and the run revises instead."""
    def staged(primary):
        return _plan_cfg(
            tmp_path, ("codex", "gemini", "antigravity"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary, plan_primary_stall_rounds=2, review_parallel=False,
            max_rounds=10,
        )

    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(
            state="blocking", summary="x", blocking_plan_issues=["Name the owner."],
        )],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    coder_round = [r for r in _plan_records(runner) if r.metadata.role == "coder"][-1].metadata.round_number
    runner.claude_outputs.append(_semantic_patch(
        runner, round_number=coder_round, summary="Owner.", item_id="item-1",
    ))
    runner.codex_outputs.append(_review(
        "codex", state="blocking", blocking_plan_issues=["Name the deadline."],
    ))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    assert [
        (r.metadata.round_number, r.metadata.state)
        for r in _plan_records(runner) if r.metadata.agent == "Codex"
    ] == [(1, "blocking"), (2, "blocking")]
    still_open = [{"item_id": "item-2", "disposition": "blocking", "note": "Still open."}]
    third_block = dict(
        state="blocking", blocking_plan_issues=["Name the metric."], dispositions=still_open,
    )
    start = len(runner.commands)
    if next_primary == "codex":
        # The pending round-2 revision runs; round 3 then stops on the streak
        # before Codex is invoked again.
        coder_round = [
            r for r in _plan_records(runner) if r.metadata.role == "coder"
        ][-1].metadata.round_number
        runner.claude_outputs.append(_semantic_patch(
            runner, round_number=coder_round, summary="Deadline.", item_id="item-2",
        ))
        runner.codex_outputs.append(_review("codex", **third_block))
        with pytest.raises(AgentLoopError, match="stall"):
            run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
        assert _plan_agents(runner, start) == ["claude"]
        return
    runner.antigravity_outputs.append(_review("antigravity", **third_block))
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("antigravity"), plan_first=True)
    # One link, a primary-only round on the new primary, and a planner revision:
    # the old primary's two blocks did not count towards the new streak.
    (payload,) = _plan_link_payloads(runner)
    assert payload["new_board"]["primary"] == "Antigravity"
    assert _plan_agents(runner, start) == ["agy", "claude"]


@pytest.mark.parametrize("next_primary", ["codex", "gemini"])
def test_primary_change_after_a_step_back_episode_resets_its_counters(tmp_path, next_primary):
    """Item-9.s5: Codex's crossed-plan blocks earn a step-back revision at round 5 and
    Codex rejects it, so --plan-step-back-escalation-rounds 1 stops for a human
    decision.  Resuming with Codex stops again before any agent turn.  After a
    primary change, the episode and its escalation count belong to the old primary:
    Gemini reviews the candidate, its single block neither escalates nor earns a
    step-back, and the run revises without approving."""
    from coding_review_agent_loop.review_step_back import derive_plan_step_back_state
    from issue_loop_test_helpers import (
        _STEP_BACK_MARKER,
        _m1103_blocking_chain,
        _m1103_fresh_base,
        _m1251_justify_patch,
    )

    def staged(primary):
        return _plan_cfg(
            tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
            primary_plan_reviewer=primary, max_rounds=8, plan_growth_max_chars=4500,
            plan_growth_max_revisions=3, plan_step_back_escalation_rounds=1,
            review_parallel=False,
        )

    fresh, base0 = _m1103_fresh_base()
    codex1, claude1, base2 = _m1103_blocking_chain(1, 1, base=base0)
    codex2, _claude2, _base = _m1103_blocking_chain(2, 2, base=base2)
    justify, base3 = _m1251_justify_patch(base2, resolved_item="item-2")
    codex_rest, claude_rest, _base = _m1103_blocking_chain(3, 5, base=base3)
    runner = _plan_runner(
        claude_outputs=[fresh, *claude1, justify, *claude_rest[:-1]],
        codex_outputs=[*codex1, *codex2, *codex_rest],
    )
    with pytest.raises(AgentLoopError, match="human decision required"):
        run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
    records = _plan_records(runner)
    (step_back,) = [r.metadata for r in records if r.metadata.step_back_entries]
    assert (step_back.round_number, step_back.step_back_entries) == (
        5, ({"phase": "plan", "reviewer": "Codex", "trigger_round": 4},),
    )
    old = derive_plan_step_back_state(records, primary="Codex")
    assert old.episode is not None and old.escalation_count == 1
    start, comments_before = len(runner.commands), len(runner.issue_comments)
    if next_primary == "codex":
        with pytest.raises(AgentLoopError, match="human decision required"):
            run_issue_loop(runner, issue_number=56, config=staged("codex"), plan_first=True)
        assert _plan_agents(runner, start) == []
        return

    runner.gemini_outputs.append(_review(
        "gemini", state="blocking", blocking_plan_issues=["Name the metric."],
        dispositions=[{"item_id": "item-5", "disposition": "blocking", "note": "Still open."}],
    ))
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=staged("gemini"), plan_first=True)
    # One link moves the primary; Gemini reviews and the planner is asked for an
    # ordinary revision: no human-decision stop and no second step-back.
    (payload,) = _plan_link_payloads(runner)
    assert (payload["old_board"]["primary"], payload["new_board"]["primary"]) == ("Codex", "Gemini")
    assert _plan_agents(runner, start) == ["gemini", "claude"]
    planner_prompt = [cmd[-1] for cmd, _cwd in runner.commands[start:] if cmd[0] == "claude"][-1]
    assert _STEP_BACK_MARKER not in planner_prompt
    records = _plan_records(runner)
    new = derive_plan_step_back_state(records, primary="Gemini")
    assert new.episode is None and new.escalation_count == 0
    assert [r.classification for r in new.reviews] == ["new-finding"]
    assert sum(1 for r in records if r.metadata.step_back_entries) == 1
    assert not any(r.metadata.phase == "plan-phase-advance" for r in records)
    assert not any(
        "human decision required" in c["body"]
        for c in runner.issue_comments[comments_before:]
    )


def _legacy_plan_with_one_shot_handoff(tmp_path, monkeypatch, requirement):
    """Approved legacy keyless plain history (board A = Codex, Gemini) with matching
    requirement evidence and a complete one-shot handoff to PR #77 bound to its hash."""
    from coding_review_agent_loop.decomposition import (
        approved_plan_hash,
        format_one_shot_impl_handoff_comment,
    )
    from coding_review_agent_loop.execution_policy import _current_execution_recommendation
    from coding_review_agent_loop.round_state import _plan_subject

    _patch_requirements(monkeypatch, [requirement])
    runner, _config = _approved_plain_plan(tmp_path, requirement=requirement)
    _strip_plan_stamps(runner)
    _strip_requirement_keys(runner, acknowledged=True)
    plan = next(r.metadata.canonical_plan for r in _plan_records(runner) if r.metadata.role == "coder")
    identity = _current_execution_recommendation(plan, []).identity()
    _append_issue_comment(runner, format_one_shot_impl_handoff_comment(
        parent_issue=56, mode="implement-one-shot", plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan), pr_number=77, pr_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        strategy="one-shot", topology_source=identity["topology_source"],
        execution_strategy_contract_version=1,
        recommendation_digest=identity["recommendation_sha256"],
    ), login="agent-loop", author_id=_PLAN_ACTOR[1])
    return runner


@pytest.mark.parametrize("implement", [False, True], ids=["plan-only", "implement-after-approval"])
@pytest.mark.parametrize("board", ["a-plus-b", "named-replacements"])
def test_legacy_hash_bound_plan_resumed_with_another_board_is_not_reopened(
    tmp_path, monkeypatch, board, implement
):
    """Item-9.s2 / row plan-approved-board-change: legacy plain history approved by
    A (Codex, Gemini) with a hash-bound one-shot handoff record is resumed with A plus
    Antigravity, or with named replacement seats (sol on Codex, flash on Antigravity).
    No plan link is posted and no planner or plan reviewer runs.  Plan-only reports
    the approved plan; implement-after-approval resumes PR #77 review on the
    invocation board, whose handoff audit record compares it with A."""
    from test_issue_pr_handoff import _audits

    requirement = _signed_requirement()
    runner = _legacy_plan_with_one_shot_handoff(tmp_path, monkeypatch, requirement)
    assert all(r.metadata.plan_board_root is None for r in _plan_records(runner))
    if board == "a-plus-b":
        config = _plan_cfg(tmp_path, ("codex", "gemini", "antigravity"))
    else:
        sol = _seat(tmp_path, "sol", "codex", "gpt-sol")
        flash = _seat(tmp_path, "flash", "antigravity", "Model A")
        config = _plan_cfg(tmp_path, (sol, flash), reviewer_seats=(sol, flash))
    assert _classify(runner, config, [requirement]).state == "approved"
    if implement:
        config = dataclasses.replace(config, plan_execution_mode="implement-one-shot")
    start, records = len(runner.commands), len(_plan_records(runner))
    if not implement:
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    else:
        # PR review starts on the invocation board; its acknowledgement turns are
        # not scripted here, so the run stops inside PR review.
        with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
            run_issue_loop(
                runner, issue_number=56, config=config, plan_first=True,
                implement_after_approval=True,
            )
    agents = [cmd[0] for cmd, _cwd in runner.commands[start:] if cmd[0] in _PLAN_AGENTS]
    # Never a planner turn, and no plan review: plan records are unchanged.
    assert "claude" not in agents
    assert len(_plan_records(runner)) == records
    assert _plan_links(runner) == []
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands[start:])
    if not implement:
        assert agents == []
        assert _audits(runner) == []
        return
    # The existing handoff routes to PR #77 review: a PR reviewer of the invocation
    # board runs, and the handoff audit record compares it with the approving A.
    assert agents and agents[0] == "codex"
    assert any(
        cmd[:2] == ["gh", "pr"] and "77" in cmd for cmd, _cwd in runner.commands[start:]
    )
    (audit,) = _audits(runner)
    if board == "a-plus-b":
        assert "- Added reviewer(s): Antigravity" in audit
        assert "- Removed reviewer(s): none" in audit
    else:
        assert "- Added reviewer(s): sol, flash" in audit
        assert "- Removed reviewer(s): Codex, Gemini" in audit


def test_approved_staged_plan_implemented_under_default_flags_is_not_reopened(tmp_path):
    """Item-9.s2 / row plan-approved-board-change: an approved primary-then-panel plan
    resumed for implementation under default (all-reviewers) planning flags and a
    floor proceeds to implementation: no planner revision, no plan review, no link."""
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
    )
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    assert run_issue_loop(runner, issue_number=56, config=staged, plan_first=True) == 0
    start, records = len(runner.commands), len(_plan_records(runner))
    default_flags = _plan_cfg(
        tmp_path, ("codex", "gemini"), min_reviewers=2, plan_execution_mode="implement-one-shot",
    )
    # The implementation coder's turn is not scripted, so the run stops there.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=default_flags, plan_first=True,
            implement_after_approval=True,
        )
    turns = [cmd for cmd, _cwd in runner.commands[start:] if cmd[0] in _PLAN_AGENTS]
    # Exactly one turn: the implementation coder.  No planner revision, no plan
    # reviewer, and no drift refusal of the primary-then-panel history.
    assert [cmd[0] for cmd in turns] == ["claude"]
    assert turns[0][-1].startswith("Implement the approved plan for GitHub issue #56")
    new = [r.metadata for r in _plan_records(runner)[records:]]
    # Completed approval goes straight to the live execution gate without
    # publishing a redundant scheduling decision under the approving board.
    assert new == []
    assert _plan_links(runner) == []


# --- Authenticated named-board legacy PR entry (#1365) ---------------------------------


def _entry_check(runner, config, bodies, *, owning_issue_number=None):
    from coding_review_agent_loop.pr_board_state import authenticate_named_legacy_pr_entry

    authenticate_named_legacy_pr_entry(
        runner, config=config, pr_number=77,
        comments=[SimpleNamespace(body=body) for body in bodies],
        owning_issue_number=owning_issue_number,
    )


def _rest_reads(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["gh", "api"]]


@pytest.mark.parametrize("case", ["plain-board", "anchored-history", "empty-history", "link-history"])
def test_named_legacy_entry_check_is_a_no_op_without_unanchored_named_history(tmp_path, case):
    """Row entry-check-noop: no authenticated read and no refusal."""
    from coding_review_agent_loop.pr_board_state import named_legacy_entry_required

    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    named = _named(tmp_path, "codex", flash)
    root = identity_from_config(named)
    config, bodies = {
        "plain-board": (make_config(tmp_path, reviewer=("codex",)), [_legacy("coder", "Claude", 1)]),
        "anchored-history": (named, [_legacy("coder", "Claude", 1), _record_body(1, root, root)]),
        "empty-history": (named, []),
        "link-history": (named, [
            _legacy("coder", "Claude", 1),
            _link_body(root, root.digest, root, root, anchor=1, advance=True)[0],
        ]),
    }[case]
    assert named_legacy_entry_required(
        config, [SimpleNamespace(body=body) for body in bodies]
    ) is False
    # A runner that would refuse any authenticated read: nothing is read.
    runner = FakeRunner(serve_rest_issue_comments=False)
    runner.default_pr_actor = False
    _entry_check(runner, config, bodies, owning_issue_number=56)
    assert runner.commands == []


def test_named_legacy_entry_check_requires_actor_authored_records(tmp_path):
    """The check reads the PR through REST and refuses a foreign coder record."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    config = _named(tmp_path, "codex", flash)
    body = _legacy("coder", "Claude", 1)
    runner = _legacy_runner([body])
    runner.pr_payload["body"] = "Task PR."
    _entry_check(runner, config, [body])
    assert ["gh", "api", "repos/OWNER/REPO/issues/77/comments?per_page=100&page=1"] in _rest_reads(runner)

    foreign = _legacy_runner([body])
    foreign.pr_payload["comments"][0].update(_rest_author_id=999, author={"login": "intruder"})
    with pytest.raises(AgentLoopError, match="Human decision required: PR #77 has unanchored"):
        _entry_check(foreign, config, [body])


# --- Shared owning-issue resolution for the named legacy entry (#1418) -----------------


def _owning_handoff(*, issue_number=56, login="coding-review-agent-loop", author_id=_ACTOR_ID):
    from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment

    return {
        "author": {"login": login},
        "createdAt": "2026-05-19T09:00:00Z",
        "body": format_issue_pr_handoff_comment(
            issue_number=issue_number, pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77", pr_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            flow="issue-implementation", plan_hash=None,
        ),
        "_rest_author_id": author_id,
    }


def _managed_entry_runner(body, issue_comments_by_number, *, head_branch="agent-loop/managed-56"):
    runner = _legacy_runner([body])
    runner.pr_payload.update(body="Task PR.", headRefName=head_branch)
    runner.serve_rest_issue_comments = True
    runner.issue_comments_by_number = {
        int(number): list(comments) for number, comments in issue_comments_by_number.items()
    }
    return runner


def _owning_entry_check(runner, config, body, **kwargs):
    from coding_review_agent_loop.pr_board_state import authenticate_named_legacy_pr_entry

    authenticate_named_legacy_pr_entry(
        runner, config=config, pr_number=77, comments=[SimpleNamespace(body=body)],
        owning_issue_number=None, pr_body="Task PR.", **kwargs,
    )


def _issue_rest_reads(runner, number):
    return [
        cmd for cmd in _rest_reads(runner)
        if cmd[2].startswith(f"repos/OWNER/REPO/issues/{number}/comments?")
    ]


def test_named_legacy_entry_check_refuses_a_foreign_handoff_on_the_managed_branch_issue(tmp_path):
    """Row managed-branch-foreign-handoff: the branch-named issue is authenticated."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    config = _named(tmp_path, "codex", flash, managed_ci_pr_mode=True)
    body = _legacy("coder", "Claude", 1)
    runner = _managed_entry_runner(
        body, {56: [_owning_handoff(login="intruder", author_id=999)]},
    )
    with pytest.raises(AgentLoopError, match="handoff record on issue #56 is not the latest"):
        _owning_entry_check(runner, config, body, head_branch="agent-loop/managed-56")
    assert _issue_rest_reads(runner, 56)
    assert len(runner.pr_payload["comments"]) == 1


def test_named_legacy_entry_check_admits_an_actor_handoff_on_the_managed_branch_issue(tmp_path):
    """Row managed-branch-authored-handoff: the same setup with an actor-authored handoff passes."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    config = _named(tmp_path, "codex", flash, managed_ci_pr_mode=True)
    body = _legacy("coder", "Claude", 1)
    runner = _managed_entry_runner(body, {56: [_owning_handoff()]})
    _owning_entry_check(runner, config, body, head_branch="agent-loop/managed-56")
    assert _issue_rest_reads(runner, 56)


@pytest.mark.parametrize("scope_source", ["argument", "config"])
def test_named_legacy_entry_check_authenticates_the_explicit_managed_ci_scope(tmp_path, scope_source):
    """Row explicit-scope-wins: the scope issue, not the branch issue, is authenticated."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    overrides = {"managed_ci_issue_number": 58} if scope_source == "config" else {}
    config = _named(tmp_path, "codex", flash, **overrides)
    body = _legacy("coder", "Claude", 1)
    issues = {
        56: [_owning_handoff()],
        58: [_owning_handoff(issue_number=58, login="intruder", author_id=999)],
    }
    kwargs = {"managed_ci_issue_number": 58} if scope_source == "argument" else {}
    runner = _managed_entry_runner(body, issues)
    with pytest.raises(AgentLoopError, match="handoff record on issue #58 is not the latest"):
        _owning_entry_check(runner, config, body, head_branch="agent-loop/managed-56", **kwargs)
    assert _issue_rest_reads(runner, 58)
    assert not _issue_rest_reads(runner, 56)

    issues[58] = [_owning_handoff(issue_number=58)]
    admitted = _managed_entry_runner(body, issues)
    _owning_entry_check(admitted, config, body, head_branch="agent-loop/managed-56", **kwargs)
    assert _issue_rest_reads(admitted, 58)
    assert not _issue_rest_reads(admitted, 56)


def test_named_legacy_entry_check_skips_the_managed_branch_outside_managed_ci(tmp_path):
    """Row non-managed-branch-no-fallback: no new issue read or refusal."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    config = _named(tmp_path, "codex", flash)
    body = _legacy("coder", "Claude", 1)
    runner = _managed_entry_runner(
        body, {56: [_owning_handoff(login="intruder", author_id=999)]},
    )
    _owning_entry_check(runner, config, body, head_branch="agent-loop/managed-56")
    assert not _issue_rest_reads(runner, 56)




@pytest.mark.parametrize(
    "caller, argument_scope, config_scope, contract, branch, managed_mode, body, expected",
    [
        # (owning, fresh_scope, branch_issue, ordinary_managed_recovery)
        (41, 42, None, 43, "agent-loop/managed-44", True, "Fixes #45", (41, 42, 44, True)),
        (None, 42, 49, 43, "agent-loop/managed-44", False, "Fixes #45", (42, 42, 44, True)),
        (None, None, 49, 43, "agent-loop/managed-44", False, "Fixes #45", (49, 49, 44, True)),
        (None, None, None, 43, "agent-loop/managed-44", True, "Fixes #45", (43, None, 44, True)),
        (None, None, None, None, "agent-loop/managed-44", True, "Fixes #45", (44, None, 44, True)),
        (None, None, None, None, "agent-loop/managed-44", False, "Fixes #45", (45, None, 44, False)),
        (None, None, None, None, "agent-loop/managed-44", False, "Task PR.", (None, None, 44, False)),
        (None, None, None, None, "agent-loop/managed-0", True, "Fixes #45", (45, None, None, True)),
        (None, None, None, None, "feature/x", True, "Fixes #45", (45, None, None, True)),
        (None, None, None, None, None, True, "Fixes #45 and fixes #46", (None, None, None, True)),
        (None, None, None, None, "agent-loop/managed-44-x", True, "Task PR.", (None, None, None, True)),
    ],
)
def test_resolve_pr_owning_issue_order_and_recovery_fields(
    tmp_path, caller, argument_scope, config_scope, contract, branch, managed_mode, body, expected,
):
    """Row resolver-parity: the documented order and the recovery-scope fields."""
    from coding_review_agent_loop.pr_board_state import resolve_pr_owning_issue

    config = make_config(
        tmp_path, managed_ci_pr_mode=managed_mode, managed_ci_issue_number=config_scope,
    )
    reads = []

    def lazy_contract():
        reads.append(1)
        return contract

    resolved = resolve_pr_owning_issue(
        config, caller_issue_number=caller, managed_ci_issue_number=argument_scope,
        head_branch=branch, recorded_primary_issue_number=lazy_contract, pr_body=body,
    )
    assert (
        resolved.owning_issue_number, resolved.fresh_scope_number,
        resolved.branch_issue_number, resolved.ordinary_managed_recovery,
    ) == expected
    # The contract is consulted only when the caller and the scope resolve nothing.
    assert reads == ([] if caller is not None or resolved.fresh_scope_number else [1])
    # A precomputed contract value resolves identically.
    assert resolve_pr_owning_issue(
        config, caller_issue_number=caller, managed_ci_issue_number=argument_scope,
        head_branch=branch, recorded_primary_issue_number=contract, pr_body=body,
    ) == resolved


@pytest.mark.parametrize(
    "branch", ["agent-loop/managed-1513", "agent-loop/managed-01", "feature/x", None],
)
def test_resolve_pr_owning_issue_uses_the_shared_managed_branch_parser(tmp_path, branch):
    """#1513: the owning-issue branch number is the shared parser's result."""
    from coding_review_agent_loop.managed_ci import managed_branch_issue_number
    from coding_review_agent_loop.pr_board_state import resolve_pr_owning_issue

    config = make_config(tmp_path, managed_ci_pr_mode=True)
    resolved = resolve_pr_owning_issue(
        config, caller_issue_number=None, managed_ci_issue_number=None,
        head_branch=branch, recorded_primary_issue_number=None, pr_body="Task PR.",
    )
    assert resolved.branch_issue_number == managed_branch_issue_number(branch)
    assert resolved.owning_issue_number == managed_branch_issue_number(branch)
    assert (resolved.branch_issue_number is not None) == (branch == "agent-loop/managed-1513")


def test_run_pr_loop_refuses_a_foreign_managed_branch_handoff_before_any_write(tmp_path):
    """Row managed-branch-foreign-handoff: run_pr_loop refuses with nothing written."""
    flash = _seat(tmp_path, "flash", "antigravity", "Model F")
    config = _named(tmp_path, "codex", flash, managed_ci_pr_mode=True)
    body = _legacy("coder", "Claude", 1)
    runner = _managed_entry_runner(
        body, {56: [_owning_handoff(login="intruder", author_id=999)]},
    )
    with pytest.raises(AgentLoopError, match="handoff record on issue #56 is not the latest"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _issue_rest_reads(runner, 56)
    assert len(runner.pr_payload["comments"]) == 1
    assert _agents(runner) == []
    assert not [
        cmd for cmd, _cwd in runner.commands
        if cmd[:3] in (["gh", "pr", "comment"], ["gh", "pr", "edit"], ["gh", "workflow", "run"])
        or (cmd[:2] == ["gh", "api"] and any(arg in ("POST", "PATCH", "DELETE") for arg in cmd))
    ]


def test_approved_all_reviewers_plan_resumed_with_a_baseline_budget_is_not_reopened(
    tmp_path, monkeypatch
):
    """#1516 row flag-validation-inheritance (workflow): a primary-then-panel invocation
    with a non-default --plan-provenance-baseline-max-chars resumes a completely
    approved all-reviewers plan.  The shortcut completes without a config validation
    error or a planning turn, and implementation gets the invocation's budget back."""
    from coding_review_agent_loop import plan_first_loop
    from coding_review_agent_loop.review_step_back import DEFAULT_PROVENANCE_BASELINE_MAX_CHARS

    full_board = _plan_cfg(tmp_path, ("codex", "gemini"))
    runner = _plan_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    assert run_issue_loop(runner, issue_number=56, config=full_board, plan_first=True) == 0
    start = len(runner.commands)

    approving_configs, restored_configs = [], []
    real_approving = plan_first_loop._approving_plan_board_config
    real_restore = plan_first_loop._restore_invocation_board

    def spy_approving(config, approving):
        result = real_approving(config, approving)
        approving_configs.append(result)
        return result

    def spy_restore(config, invocation):
        result = real_restore(config, invocation)
        restored_configs.append(result)
        return result

    monkeypatch.setattr(plan_first_loop, "_approving_plan_board_config", spy_approving)
    monkeypatch.setattr(plan_first_loop, "_restore_invocation_board", spy_restore)
    staged = _plan_cfg(
        tmp_path, ("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex", plan_provenance_baseline_max_chars=50000,
        plan_execution_mode="implement-one-shot",
    )
    # The implementation coder's turn is not scripted, so the run stops there.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(
            runner, issue_number=56, config=staged, plan_first=True,
            implement_after_approval=True,
        )
    turns = [cmd for cmd, _cwd in runner.commands[start:] if cmd[0] in _PLAN_AGENTS]
    # Exactly one turn, the implementation coder: no planner, no plan reviewer.
    assert [cmd[0] for cmd in turns] == ["claude"]
    assert turns[0][-1].startswith("Implement the approved plan for GitHub issue #56")
    (approving,) = approving_configs
    assert approving.plan_review_policy == "all-reviewers"
    assert approving.plan_provenance_baseline_max_chars == DEFAULT_PROVENANCE_BASELINE_MAX_CHARS
    (restored,) = restored_configs
    assert restored.plan_review_policy == "primary-then-panel"
    assert restored.plan_provenance_baseline_max_chars == 50000
    assert _plan_links(runner) == []


# Neutral staged rollover compatibility (#1827).
def _neutral_lineage_fixture(*, pending=False):
    root = dataclasses.replace(_identity(("a", "codex"), ("b", "claude")), broad_rules=())
    board = dataclasses.replace(_identity(("a", "codex"), ("c", "antigravity")), broad_rules=())
    payload = board_change_link_payload(
        pr_number=56, surface=PLAN_SURFACE, root=root, predecessor=root.digest,
        old_board=root, new_board=board, old_seat_binding=root.seat_binding(),
        binding_init={}, anchor_round=1, advance=True,
    )
    def stamped(round_number, identity, digest=None, **extra):
        return PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", subject="candidate",
            round_number=round_number, phase="reconciliation",
            plan_board_root=root.payload(), plan_board_identity=identity.payload(),
            plan_board_binding_init={}, seat_binding=identity.seat_binding(),
            reviewer_board_change_digest=digest, **extra,
        )
    settled = stamped(1, root)
    active = stamped(2, board, board_change_link_digest(payload))
    advance = PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", subject="candidate",
        round_number=2 if pending else 3, phase="plan-phase-advance",
        prior_items=(_item(1, "a"),), seat_binding=board.seat_binding(),
    )
    bodies = [_attach_round_metadata("Settled root", settled),
              str(render_board_change_link(payload, surface=PLAN_SURFACE))]
    if not pending:
        bodies.append(_attach_round_metadata("Settled linked round", active))
    return bodies, advance, board


def _resolve_neutral_plan(bodies, board):
    from coding_review_agent_loop.panel_evidence import _plan_contract_or_none
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments
    comments = [SimpleNamespace(body=body, author_id=_PLAN_ACTOR[1]) for body in bodies]
    return resolve_board_lineage(
        _extract_round_metadata_records(comments, flow="plan"), comments,
        amendments=collect_reviewer_board_amendments(comments, flow="plan", issue_number=56),
        configured=board, pr_number=56, trusted_actor=_PLAN_ACTOR,
        contract_from_metadata=_plan_contract_or_none,
        drift_error=lambda persisted, detail: AgentLoopError(detail), surface=PLAN_SURFACE,
    )


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("retained", [False, True])
def test_neutral_link_advance_preserves_lineage_authority(legacy, retained):
    bodies, advance, board = _neutral_lineage_fixture()
    if legacy:
        from coding_review_agent_loop.round_state import _decode_round_metadata
        import re
        meta = _decode_round_metadata(re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", bodies[-1]).group(1))
        bodies[-1] = _attach_round_metadata("Legacy settlement", dataclasses.replace(meta, phase="authoritative"))
    if retained:
        from coding_review_agent_loop.round_state import _decode_round_metadata
        import re
        meta = _decode_round_metadata(re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", bodies[-1]).group(1))
        bodies.append(_attach_round_metadata("Retained reviewer", dataclasses.replace(
            meta, role="reviewer", agent="a", phase="authoritative", new_items=(_item(1, "a"),),
        )))
    before = _resolve_neutral_plan(bodies, board)
    bodies.append(_attach_round_metadata("Historical neutral advance", advance))
    after = _resolve_neutral_plan(bodies, board)
    assert after == before
    assert advance.prior_items[0].status == "blocking"


@pytest.mark.parametrize("changes", [
    {"role": "reviewer", "agent": "a"}, {"role": "coder"}, {"agent": "a"},
    {"phase": "authoritative"}, {"phase": "scheduler-prelaunch"},
    {"scheduler_contract": {"invalid": True}},
    {"reviewer_board_amendment_digest": "a" * 64},
    {"new_items": (_item(2, "a"),)}, {"evidence_clearances": (("head", "item-1"),)},
    {"subject": "another"}, {"round_number": 4},
])
def test_neutral_link_advance_rejects_substantive_or_unexplained_shapes(changes):
    bodies, advance, board = _neutral_lineage_fixture()
    bodies.append(_attach_round_metadata("Invalid advance", dataclasses.replace(advance, **changes)))
    with pytest.raises(AgentLoopError):
        _resolve_neutral_plan(bodies, board)


def test_neutral_advance_does_not_activate_pending_link():
    bodies, advance, board = _neutral_lineage_fixture(pending=True)
    bodies.append(_attach_round_metadata("Neutral advance", advance))
    with pytest.raises(AgentLoopError, match="after the pending board-change"):
        _resolve_neutral_plan(bodies, board)


@pytest.mark.parametrize("defect", ["reviewer", "scheduler", "board"])
def test_neutral_advance_cannot_shelter_invalid_retained_work(defect):
    bodies, advance, board = _neutral_lineage_fixture()
    if defect == "board":
        invalid = dataclasses.replace(advance, round_number=2, phase="reconciliation",
                                      plan_board_root=board.payload(), plan_board_identity=board.payload())
    else:
        invalid = dataclasses.replace(advance, round_number=2,
            phase="scheduler-prelaunch" if defect == "scheduler" else "authoritative",
            role="summary" if defect == "scheduler" else "reviewer",
            agent="Orchestrator" if defect == "scheduler" else "a")
    bodies.append(_attach_round_metadata("Invalid retained work", invalid))
    bodies.append(_attach_round_metadata("Neutral advance", advance))
    with pytest.raises(AgentLoopError):
        _resolve_neutral_plan(bodies, board)


@pytest.mark.parametrize("older_unbound", [False, True])
@pytest.mark.parametrize("successive", [False, True])
def test_neutral_advances_do_not_compare_seat_bindings_across_boards(older_unbound, successive):
    from coding_review_agent_loop.round_state import _decode_round_metadata
    import re
    bodies, advance, board = _neutral_lineage_fixture()
    root_meta = _decode_round_metadata(re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", bodies[0]).group(1))
    root = BoardIdentity.from_payload(root_meta.plan_board_root)
    root_advance = dataclasses.replace(advance, round_number=2, seat_binding=root.seat_binding())
    bodies.insert(1, _attach_round_metadata("Neutral root advance", root_advance))
    if older_unbound:
        bodies.insert(0, _attach_round_metadata("Older unbound history", dataclasses.replace(
            root_meta, plan_board_root=None, plan_board_identity=None,
            plan_board_binding_init=None, seat_binding=None,
        )))
    bodies.append(_attach_round_metadata("Neutral linked advance", advance))
    lineage = _resolve_neutral_plan(bodies, board)
    assert not lineage.pending_links
    if successive:
        next_board = dataclasses.replace(board, seats=(board.seats[1], board.seats[0]))
        payload = board_change_link_payload(
            pr_number=56, surface=PLAN_SURFACE, root=root,
            predecessor=lineage.elements[-1].digest, old_board=board, new_board=next_board,
            old_seat_binding=board.seat_binding(), binding_init={}, anchor_round=2, advance=True,
        )
        bodies.append(str(render_board_change_link(payload, surface=PLAN_SURFACE)))
        bodies.append(_attach_round_metadata("Genuine next-board reconciliation", dataclasses.replace(
            root_meta, round_number=3, plan_board_identity=next_board.payload(),
            seat_binding=next_board.seat_binding(), reviewer_board_change_digest=board_change_link_digest(payload),
        )))
        bodies.append(_attach_round_metadata("Neutral next-board advance", dataclasses.replace(
            advance, round_number=4, seat_binding=next_board.seat_binding(),
        )))
        assert len(_resolve_neutral_plan(bodies, next_board).elements) == 2


@pytest.mark.parametrize("binding", [None, {"broken": {}}, "changed-board"])
def test_substantive_legacy_binding_validation_remains_strict(binding):
    bodies, advance, board = _neutral_lineage_fixture()
    if binding == "changed-board":
        binding = _identity(("a", "codex"), ("b", "claude")).seat_binding()
    bodies.insert(0, _attach_round_metadata("Legacy substantive reviewer", dataclasses.replace(
        advance, round_number=1, role="reviewer", agent="a", phase="publication", seat_binding=binding,
    )))
    # A neutral summary cannot shelter structural binding problems on itself,
    # nor differences or bound/unbound mixing on substantive legacy records.
    bodies.insert(1, _attach_round_metadata("Bound legacy reviewer", dataclasses.replace(
        advance, round_number=1, role="reviewer", agent="a", phase="publication",
    )))
    bodies.append(_attach_round_metadata("Neutral advance", advance))
    with pytest.raises(AgentLoopError, match="binding|bound and unbound"):
        _resolve_neutral_plan(bodies, board)


def test_neutral_summary_still_validates_its_own_binding_structure():
    bodies, advance, board = _neutral_lineage_fixture()
    bodies.append(_attach_round_metadata("Malformed neutral binding", dataclasses.replace(
        advance, seat_binding={"a": {}},
    )))
    with pytest.raises(AgentLoopError, match="malformed"):
        _resolve_neutral_plan(bodies, board)



def test_neutral_summary_does_not_consume_pending_amendment():
    bodies, advance, board = _neutral_lineage_fixture()
    bodies.append(format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None, original_required_reviewers=board.names,
        policy="all-reviewers", primary_reviewer=None, removed_reviewers=("c",),
        effective_from_round=3, reason="backend-unavailable", rationale="Quota unavailable.",
    ))
    before = _resolve_neutral_plan(bodies, board)
    bodies.append(_attach_round_metadata("Neutral advance", advance))
    assert _resolve_neutral_plan(bodies, board) == before
    assert len(before.pending_amendments) == 1
    amended = dataclasses.replace(board, seats=(board.seats[0],))
    element = before.elements[-1]
    bodies.append(_attach_round_metadata("Genuine stamped work", dataclasses.replace(
        advance, phase="reconciliation", plan_board_root=before.root.payload(),
        plan_board_identity=amended.payload(), plan_board_binding_init={},
        seat_binding=amended.seat_binding(), reviewer_board_amendment_digest=element.digest,
    )))
    assert not _resolve_neutral_plan(bodies, amended).pending_amendments


@pytest.mark.parametrize("claim", ["digest", "identity", "binding"])
def test_identity_bearing_advance_keeps_normal_validation(claim):
    bodies, advance, board = _neutral_lineage_fixture()
    before = _resolve_neutral_plan(bodies, board)
    stamped = dataclasses.replace(
        advance, plan_board_root=before.root.payload(), plan_board_identity=board.payload(),
        plan_board_binding_init={}, reviewer_board_change_digest=before.elements[-1].digest,
    )
    if claim == "digest":
        stamped = dataclasses.replace(stamped, reviewer_board_change_digest=None)
    elif claim == "identity":
        stamped = dataclasses.replace(stamped, plan_board_identity=before.root.payload())
    else:
        stamped = dataclasses.replace(stamped, seat_binding=before.root.seat_binding())
    bodies.append(_attach_round_metadata("Stamped advance", stamped))
    with pytest.raises(AgentLoopError):
        _resolve_neutral_plan(bodies, board)



def test_pr_phase_advance_does_not_receive_plan_lineage_exception():
    root = _identity(("a", "codex"), ("b", "claude"))
    board = _identity(("a", "codex"), ("c", "antigravity"))
    link, digest = _link_body(root, root.digest, root, board, anchor=1, advance=True)
    bodies = [_record_body(1, root, root), link, _record_body(2, root, board, change=digest)]
    bodies.append(_legacy("summary", "Orchestrator", 3, phase="plan-phase-advance",
                          seat_binding=board.seat_binding()))
    with pytest.raises(AgentLoopError, match="carries no board identity"):
        _resolve(bodies, configured=board)



def test_neutral_summary_cannot_seed_legacy_root_bindings():
    board = dataclasses.replace(_identity(("Codex", "codex"), ("Gemini", "gemini")), broad_rules=())
    settled = PostedRoundMetadata(flow="plan", role="summary", agent="Orchestrator",
        subject="candidate", round_number=1, phase="reconciliation")
    bodies = [_attach_round_metadata("Legacy unbound settlement", settled)]
    before = _resolve_neutral_plan(bodies, board)
    advance = dataclasses.replace(settled, round_number=2, phase="plan-phase-advance",
        seat_binding=_identity(("injected", "antigravity")).seat_binding())
    bodies.append(_attach_round_metadata("Neutral bound advance", advance))
    after = _resolve_neutral_plan(bodies, board)
    assert after.root == before.root
    assert after.inits == before.inits == {}
