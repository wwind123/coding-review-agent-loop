"""Closed reconciled PR rounds on resume (#1553).

A PR round closed by its reconciliation summary never receives another
scheduling or reviewer record.  A later decision that needs fresh reviewer
turns opens the next round; stray checkpoints already posted into a closed
round are inert for resume and every scheduler reader.
"""

import dataclasses
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.review_rounds as review_rounds
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.protocol import ReviewItemDisposition, UnresolvedReviewItem
from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
from coding_review_agent_loop.unresolved_items import _upsert_converted_evidence_obligation

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from pr_loop_test_helpers import (
    _agent_sequence,
    _m943_amendment_from_error,
    _m943_append,
    _rewrite_pr_metadata,
    _staged_config,
    _staged_review,
    _unavailable,
)

_ACTOR_ID = 424242  # FakeRunner's default authenticated PR actor


def _records(runner):
    return architecture_contract._extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
    )


def _last_reconciliation(runner, round_number):
    return max(
        (
            record for record in _records(runner)
            if record.metadata.phase == "reconciliation"
            and record.metadata.round_number == round_number
        ),
        key=lambda record: record.index,
    )


def _reconciled_secondary_audit(tmp_path, **overrides):
    """Round 1 primary approval; round 2 secondary audit with Codex paused (carried)."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, **overrides)) == 0
    assert _primary_then_panel(_agent_sequence(runner))
    checkpoint = next(
        record.metadata for record in _records(runner)
        if record.metadata.round_number == 2 and record.metadata.phase == "scheduler-prelaunch"
    )
    assert checkpoint.scheduler_selected_reviewers == ("Gemini", "Antigravity")
    assert "Codex" in dict(checkpoint.scheduler_paused_reviewers)
    return runner


def _sign(runner, body="Keep the worker cleanup idempotent."):
    runner.pr_payload["comments"].append(
        {
            "author": {"login": "maintainer"},
            "createdAt": f"2026-05-23T00:00:{len(runner.pr_payload['comments']):02d}Z",
            "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1553",
            "body": f"{body}\n\n-- Human Reviewer",
        }
    )


def _queue_resolved_board(runner):
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", resolved=True))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", resolved=True))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", resolved=True))


def _append_bot_record(runner, body):
    template = next(
        comment for comment in runner.pr_payload["comments"]
        if comment.get("author", {}).get("login") != "maintainer"
    )
    comments = runner.pr_payload["comments"]
    comments.append(
        dict(template, body=body, createdAt=f"2026-05-23T00:00:{len(comments):02d}Z")
    )


def _append_stray_checkpoint(runner, **changes):
    """The #1607 shape: a round-2 scheduler checkpoint posted after round 2's reconciliation."""
    original = next(
        record.metadata for record in _records(runner)
        if record.metadata.round_number == 2 and record.metadata.phase == "scheduler-prelaunch"
    )
    values = dict(
        scheduler_selected_reviewers=("Codex",),
        scheduler_paused_reviewers=(
            ("Gemini", "carried exact-head approval"),
            ("Antigravity", "carried exact-head approval"),
        ),
        scheduler_phase="remediation",
        scheduler_reasons=(
            "remediation: exact-head primary approval is outstanding before the final "
            "secondary sweep",
            "same exact candidate head; recover missing reviewer work",
        ),
    )
    values.update(changes)
    stray = dataclasses.replace(original, **values)
    _append_bot_record(
        runner,
        _attach_round_metadata(
            "PR review scheduling audit: selected Codex; paused Gemini, Antigravity; "
            "reason: remediation; phase: remediation; head: abc123.",
            stray,
        ),
    )
    return stray


def _primary_then_panel(sequence):
    # Parallel panel reviewers launch in completion-independent order.
    return sequence[:1] == ["codex"] and sorted(sequence[1:]) == ["agy", "gemini"]


def _records_after(runner, index):
    return [record for record in _records(runner) if record.index > index]


@pytest.mark.parametrize("parallel", [False, True])
def test_1553_reconciled_round_rolls_over_after_signed_change_request(tmp_path, parallel):
    """Row reconciled-human-change-rollover: the #1607 transcript opens round 3."""
    runner = _reconciled_secondary_audit(tmp_path, review_parallel=parallel)
    reconciliation = _last_reconciliation(runner, 2).index
    _sign(runner)
    _queue_resolved_board(runner)

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, review_parallel=parallel)
    ) == 0

    later = _records_after(runner, reconciliation)
    # Nothing of any role is posted into the reconciled round.
    assert later and all(record.metadata.round_number >= 3 for record in later)
    first_checkpoint = next(
        record.metadata for record in later if record.metadata.phase == "scheduler-prelaunch"
    )
    assert first_checkpoint.round_number == 3
    assert first_checkpoint.scheduler_selected_reviewers == ("Codex",)
    # The primary remediates first, then the panel follows as usual.
    assert _primary_then_panel(_agent_sequence(runner)[3:])
    assert not any("partially published" in comment for comment in runner.comments)
    assert not any("Delete these" in comment for comment in runner.comments)


@pytest.mark.parametrize("parallel", [False, True])
def test_1553_stray_post_reconciliation_checkpoint_resumes_into_the_next_round(tmp_path, parallel):
    """Row stray-checkpoint-recovery: the #1607 PR resumes without deleting anything."""
    runner = _reconciled_secondary_audit(tmp_path, review_parallel=parallel)
    _sign(runner)
    _append_stray_checkpoint(runner)
    kept = [comment["body"] for comment in runner.pr_payload["comments"]]
    stray_index = _records(runner)[-1].index
    _queue_resolved_board(runner)

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, review_parallel=parallel)
    ) == 0

    # Every round-2 review, the reconciliation and the stray record remain.
    assert [c["body"] for c in runner.pr_payload["comments"][: len(kept)]] == kept
    later = _records_after(runner, stray_index)
    assert later and all(record.metadata.round_number >= 3 for record in later)
    assert later[0].metadata.round_number == 3
    checkpoint = next(record.metadata for record in later if record.metadata.phase == "scheduler-prelaunch")
    assert checkpoint.scheduler_selected_reviewers == ("Codex",)
    assert _primary_then_panel(_agent_sequence(runner)[3:])
    assert not any("partially published" in comment for comment in runner.comments)


def test_1553_stray_checkpoint_does_not_alter_reconstructed_scheduler_inputs(tmp_path):
    """Row stray-checkpoint-scheduler-inputs: phase, latch and calls-avoided are unchanged."""
    def resumed_checkpoint(path, *, stray):
        runner = _reconciled_secondary_audit(path)
        _sign(runner)
        if stray:
            _append_stray_checkpoint(
                runner,
                scheduler_phase="full-board",
                scheduler_force_full=True,
                scheduler_force_full_source="operator",
                scheduler_calls_avoided=99,
            )
        _queue_resolved_board(runner)
        assert run_pr_loop(runner, pr_number=77, config=_staged_config(path)) == 0
        checkpoints = [
            record.metadata for record in _records(runner)
            if record.metadata.phase == "scheduler-prelaunch" and record.metadata.round_number >= 3
        ]
        return [
            (
                item.round_number, item.scheduler_phase, item.scheduler_selected_reviewers,
                tuple(name for name, _reason in item.scheduler_paused_reviewers),
                item.scheduler_force_full, item.scheduler_force_full_source,
                item.scheduler_calls_avoided,
            )
            for item in checkpoints
        ], _agent_sequence(runner)

    baseline = resumed_checkpoint(tmp_path / "baseline", stray=False)
    with_stray = resumed_checkpoint(tmp_path / "stray", stray=True)
    assert with_stray == baseline
    assert not any(row[5] == "operator" for row in with_stray[0])
    assert with_stray[0][0][1] == "remediation"


@pytest.mark.parametrize("staged", [True, False])
def test_1553_reconciled_round_without_fresh_turns_resumes_unchanged(tmp_path, staged):
    """Row reconciled-no-fresh-turns: no rollover and no new reviewer records."""
    if staged:
        runner = _reconciled_secondary_audit(tmp_path)
        config = _staged_config(tmp_path)
    else:
        runner = FakeRunner(
            codex_outputs=[structured_pr_review()],
            gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        )
        config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    before_records = len(_records(runner))
    before_agents = len(_agent_sequence(runner))
    last_round = _records(runner)[-1].metadata.round_number

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert len(_agent_sequence(runner)) == before_agents
    assert len(_records(runner)) == before_records
    assert all(record.metadata.round_number <= last_round for record in _records(runner))


def test_1553_reconciled_round_with_all_resumed_selection_posts_no_checkpoint(tmp_path):
    """Row reconciled-selected-resumed-no-checkpoint: operator force-full on a closed round."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert sorted(_agent_sequence(runner)) == ["agy", "codex", "gemini"]
    reconciliation = _last_reconciliation(runner, 1).index
    before_agents = len(_agent_sequence(runner))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # The scheduler re-selects the complete (resumed) board, but nothing is
    # posted into the closed round and no reviewer is relaunched.
    assert _records_after(runner, reconciliation) == []
    assert len(_agent_sequence(runner)) == before_agents


def test_1553_parallel_reconciliation_only_items_survive_rollover(tmp_path):
    """Row parallel-reconciliation-only-items: IDs, owners and text are unchanged."""
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[
            structured_pr_review(
                reviewer="Google Gemini", state="blocking",
                blocking_items=["Guard the cache eviction race."],
            )
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    # The coder turn has no scripted output: the run stops after round 1 is
    # reconciled and before any coder follow-up exists.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    reconciliation = _last_reconciliation(runner, 1)
    settled = reconciliation.metadata.new_items
    assert [item.item_id for item in settled] == ["item-1"]
    gemini_publication = next(
        record.metadata for record in _records(runner)
        if record.metadata.agent == "Gemini" and record.metadata.role == "reviewer"
    )
    # Parallel publications omit numbered items; only the summary carries them.
    assert gemini_publication.new_items == ()
    # Simulate an interruption after the reconciliation and before the coder
    # dispatch was recorded: no coder recovery handoff exists.
    dispatch_indexes = {
        record.index for record in _records(runner) if record.metadata.phase == "coder-dispatch"
    }
    runner.pr_payload["comments"] = [
        comment for index, comment in enumerate(runner.pr_payload["comments"])
        if index not in dispatch_indexes
    ]
    _sign(runner)
    runner.codex_outputs.append(
        structured_pr_review(
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            human_requirements_resolved=True,
        )
    )
    runner.gemini_outputs.append(
        structured_pr_review(
            reviewer="Google Gemini", state="blocking",
            blocking_items=["Guard the cache eviction race at the second call site too."],
            prior_item_dispositions=[
                {
                    "item_id": "item-1", "disposition": "blocking",
                    "note": "The eviction path in src/cache.py still races; add a lock and a test.",
                }
            ],
            human_requirements_resolved=True,
        )
    )

    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)

    later = [
        record for record in _records_after(runner, reconciliation.index)
        if record.metadata.phase != "coder-dispatch"
    ]
    assert later and all(record.metadata.round_number == 2 for record in later)
    round_two = [record.metadata for record in later if record.metadata.role == "reviewer"]
    assert round_two
    for metadata in round_two:
        carried = {item.item_id: item for item in metadata.prior_items}
        assert carried["item-1"] == settled[0]
    new_reconciliation = _last_reconciliation(runner, 2).metadata
    # A new finding gets a fresh ID; item-1 is never reused.
    assert [item.item_id for item in new_reconciliation.new_items] == ["item-2"]


def test_1553_rollover_keeps_inert_ledger_records(tmp_path):
    """Row rollover-preserves-inert-records: converted finding and cleared tombstone."""
    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=2, row_id="row-a", matrix_identity="ab" * 32, head_sha="abc123",
        owners=("Codex", "Gemini"), source_round=1, explanation="canonical-status:missing",
    )
    tombstone = dataclasses.replace(obligation, lifecycle="cleared")
    finding = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Capture evidence",
        status="converted", evidence_row_ids=("row-a",), converted_row_ids=("row-a",),
    )
    assert finding.is_inert_ledger_record and tombstone.is_inert_ledger_record
    inert = (finding, tombstone)

    def record(role, agent, **extra):
        body = (
            structured_pr_review(reviewer=agent) if role == "reviewer"
            else f"{agent} record.\n-- Orchestrator"
        )
        return _attach_round_metadata(
            body,
            PostedRoundMetadata(
                flow="pr", role=role, agent=agent, round_number=1, subject="abc123",
                prior_items=inert, **extra,
            ),
        )

    # The cleared tombstone is still shown to reviewers as a carried machine
    # record; a disposition on an inert record never changes it.
    noted = [{"item_id": tombstone.item_id, "disposition": "resolved", "note": "Already cleared."}]
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": f"2026-05-20T09:{index:02d}:00Z",
                    "body": body,
                    "_rest_author_id": _ACTOR_ID,
                }
                for index, body in enumerate(
                    [
                        record("reviewer", "Codex", state="approved"),
                        record("reviewer", "Gemini", state="approved"),
                        record("summary", "Orchestrator", phase="reconciliation"),
                    ]
                )
            ]
        },
        codex_outputs=[
            structured_pr_review(
                prior_item_dispositions=noted, human_requirements_resolved=True,
            )
        ],
        gemini_outputs=[
            structured_pr_review(
                reviewer="Google Gemini", prior_item_dispositions=noted,
                human_requirements_resolved=True,
            )
        ],
    )
    _sign(runner)
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    # Without an approved matrix this fixture cannot finalize past the
    # tombstone; that stop comes after round 2 is fully recorded.
    with pytest.raises(AgentLoopError, match="human-only exact-head evidence"):
        run_pr_loop(runner, pr_number=77, config=config)

    round_two = [
        r.metadata for r in _records(runner)
        if r.metadata.round_number == 2 and r.metadata.role == "reviewer"
    ]
    assert len(round_two) == 2
    for metadata in round_two:
        carried = {item.item_id: item for item in metadata.prior_items}
        assert carried["item-1"] == finding
        assert carried[tombstone.item_id] == tombstone
    assert not any(r.metadata.round_number == 1 for r in _records(runner)[3:])


def _amendment_ready_round(tmp_path):
    """Round 2 reconciled with Antigravity unavailable; a signed removal is posted."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    reconciliation = _last_reconciliation(runner, 2).index
    template = _m943_amendment_from_error(str(excinfo.value))
    assert '"effective_from_round": 2' in template
    _m943_append(runner, template)
    return runner, reconciliation


def test_1553_amendment_at_reconciled_round_without_fresh_turns_activates_in_place(tmp_path):
    """Row amendment-reconciled-no-fresh-turns: the activation checkpoint stays in round 2."""
    runner, reconciliation = _amendment_ready_round(tmp_path)
    before_agents = len(_agent_sequence(runner))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    later = _records_after(runner, reconciliation)
    activation = [r.metadata for r in later if r.metadata.phase == "scheduler-prelaunch"]
    assert len(activation) == 1
    assert activation[0].round_number == 2
    assert activation[0].reviewer_board_amendment_digest is not None
    assert all(r.metadata.round_number == 2 for r in later)
    assert len(_agent_sequence(runner)) == before_agents


def test_1553_amendment_at_reconciled_round_with_fresh_turn_rolls_over(tmp_path):
    """Row amendment-reconciled-rollover: activation-only record, then round 3."""
    runner, reconciliation = _amendment_ready_round(tmp_path)
    _sign(runner)
    _queue_resolved_board(runner)

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    later = [r.metadata for r in _records_after(runner, reconciliation)]
    in_round_two = [m for m in later if m.round_number == 2]
    assert len(in_round_two) == 1
    activation = in_round_two[0]
    assert activation.phase == "scheduler-prelaunch"
    assert activation.scheduler_selected_reviewers == ()
    assert activation.reviewer_board_amendment_digest is not None
    assert "amendment activation only" in activation.scheduler_reasons[0]
    round_three = [m for m in later if m.round_number == 3]
    assert round_three and round_three[0].phase == "scheduler-prelaunch"
    assert round_three[0].scheduler_selected_reviewers == ("Codex",)
    assert round_three[0].reviewer_board_amendment_digest == activation.reviewer_board_amendment_digest
    assert tuple(round_three[0].scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
    # Reviewers ran only in round 3, on the amended board.
    assert "agy" not in _agent_sequence(runner)[3:]
    before = len(runner.pr_payload["comments"])

    # The second resume sees the amendment as used: no effective-round refusal.
    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0
    assert len(runner.pr_payload["comments"]) >= before


@pytest.mark.parametrize("ledger_differs", [False, True])
def test_1553_existing_post_reconciliation_activation_is_lineage_only(tmp_path, ledger_differs):
    """Rows existing-activation-checkpoint-kept and activation-latest-ledger-differs."""
    runner, reconciliation = _amendment_ready_round(tmp_path)
    # An earlier tool version posted the activation into the closed round.
    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0
    activation = next(
        r for r in _records_after(runner, reconciliation)
        if r.metadata.phase == "scheduler-prelaunch"
    )
    if ledger_differs:
        ack = UnresolvedReviewItem(
            item_id="human-requirements-ack", reviewer="Human requirements",
            source_round=2, text="Acknowledge the signed requirements.", status="blocking",
        )
        encoded = PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=2,
            subject="abc123", prior_items=(ack,),
        )
        from coding_review_agent_loop.round_transport import ROUND_RESUME_MARKER_RE, decode_mapping

        ack_payload = dict(
            decode_mapping(ROUND_RESUME_MARKER_RE.search(
                _attach_round_metadata("x", encoded)
            ).group("payload"))
        )["prior_items"]
        target_index = activation.index

        def add_ack(payload, state={"seen": -1}):
            state["seen"] += 1
            if payload.get("phase") != "scheduler-prelaunch" or payload.get("round_number") != 2:
                return None
            if payload.get("reviewer_board_amendment_digest") is None:
                return None
            return dict(payload, prior_items=ack_payload)

        _rewrite_pr_metadata(runner, add_ack)
        rewritten = next(r for r in _records(runner) if r.index == target_index)
        assert [item.item_id for item in rewritten.metadata.prior_items] == ["human-requirements-ack"]
        assert [item.item_id for item in rewritten.metadata.prior_items] != [
            item.item_id for item in _last_reconciliation(runner, 2).metadata.prior_items
        ]
    _sign(runner)
    _queue_resolved_board(runner)

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    later = [r.metadata for r in _records_after(runner, activation.index)]
    # No effective-round refusal, no second activation and no record in round 2:
    # the reconciled transcript stayed the anchor and the fresh turns opened round 3.
    assert later and all(m.round_number >= 3 for m in later)
    assert later[0].round_number == 3
    assert later[0].phase == "scheduler-prelaunch"
    assert later[0].scheduler_selected_reviewers == ("Codex",)


def test_1553_conflict_resume_keeps_the_coder_route_in_the_closed_round(tmp_path):
    """Row recovery-skip-resume-not-rolled-over: a pending conflict is never rolled over."""
    runner = _reconciled_secondary_audit(tmp_path)
    reconciliation = _last_reconciliation(runner, 2).index
    _sign(runner)
    runner.pr_payload["mergeable"] = "CONFLICTING"
    runner.pr_payload["mergeStateStatus"] = "DIRTY"
    before_agents = len(_agent_sequence(runner))

    # The coder has no scripted output; the point is where the run goes.
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))

    assert not isinstance(excinfo.value, review_rounds.PartialReviewRoundError)
    assert _agent_sequence(runner)[before_agents:] == ["claude"]
    # Only the coder's own dispatch record follows: no scheduler checkpoint,
    # reviewer record or reconciliation, in round 2 or a rolled-over round.
    later = _records_after(runner, reconciliation)
    assert later and all(r.metadata.phase == "coder-dispatch" for r in later)


def test_1553_rollover_at_the_round_limit_stops_with_the_round_limit_diagnostic(tmp_path):
    """Row rollover-at-round-limit: no round is minted beyond the bound."""
    runner = _reconciled_secondary_audit(tmp_path, max_rounds=2)
    reconciliation = _last_reconciliation(runner, 2).index
    _sign(runner)
    _queue_resolved_board(runner)
    before_agents = len(_agent_sequence(runner))

    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, max_rounds=2))

    assert not isinstance(excinfo.value, review_rounds.PartialReviewRoundError)
    assert "Delete these" not in str(excinfo.value)
    expected = pr_loop._round_limit_diagnostic(
        pr_number=77, round_number=2, items=[], current_head_sha="abc123",
    )
    assert str(excinfo.value).startswith(expected.split("\n")[0][:40])
    assert _records_after(runner, reconciliation) == []
    assert len(_agent_sequence(runner)) == before_agents


def test_1553_settlement_helper_matches_round_end_semantics():
    """Shared round-end settlement keeps inert records and appends new items."""
    blocking = UnresolvedReviewItem(
        item_id="item-1", reviewer="Codex", source_round=1, text="Fix it", status="blocking",
    )
    converted = UnresolvedReviewItem(
        item_id="item-2", reviewer="Codex", source_round=1, text="Evidence", status="converted",
        evidence_row_ids=("row-a",), converted_row_ids=("row-a",),
    )
    new = UnresolvedReviewItem(
        item_id="item-3", reviewer="Gemini", source_round=2, text="New", status="blocking",
    )
    resolved = ReviewItemDisposition("item-1", "Codex", "resolved")
    settled, future = pr_loop._settle_pr_round_ledger(
        (blocking, converted),
        {"item-1": [resolved], "item-2": []},
        (new,),
        compact=False,
        reconciliation_mode="aggregate",
        ledger_view=tuple,
    )
    assert [item.item_id for item in settled] == ["item-2", "item-3"]
    assert future == []
    metadata = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="abc123",
        prior_items=(blocking, converted), dispositions=(resolved,), phase="reconciliation",
    )
    assert pr_loop._reconciliation_dispositions_by_item(metadata) == {
        "item-1": [resolved], "item-2": [],
    }
    unknown = dataclasses.replace(
        metadata, dispositions=(ReviewItemDisposition("item-9", "Codex", "resolved"),)
    )
    with pytest.raises(AgentLoopError, match="refusing to settle the closed round"):
        pr_loop._reconciliation_dispositions_by_item(unknown)


def test_1553_prelaunch_classification_helper():
    approval = SimpleNamespace(
        metadata=SimpleNamespace(round_number=1, surfaced_reviewer_requirement_ids=()),
        body="approved",
    )
    common = dict(
        unavailable=False, resumed=False, selective_policy=True,
        selected_reviewer_names={"Codex"}, prior_approval=None, full_board_pass=False,
        round_number=2, human_requirements=(),
    )
    classify = pr_loop._classify_pr_reviewer_prelaunch
    assert classify("Codex", **dict(common, unavailable=True)) == "skip"
    assert classify("Codex", **dict(common, resumed=True)) == "resumed"
    assert classify("Gemini", **common) == "skip"
    assert classify("Codex", **common) == "turn"
    assert classify("Codex", **dict(common, prior_approval=approval)) == "carried"
    assert classify("Codex", **dict(common, prior_approval=approval, full_board_pass=True)) == "turn"
    # A new signed requirement the carried approval never covered needs a turn.
    assert classify(
        "Codex", **dict(common, prior_approval=approval, human_requirements=(object(),))
    ) == "turn"
