"""Unit tests for the deterministic step-back bookkeeping (#1251)."""

from __future__ import annotations

import json

import pytest

from agent_loop_helpers import make_config, structured_v1_plan_state
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.plan_assembly import make_assembled_plan_sidecar
from coding_review_agent_loop.review_step_back import (
    CLASS_APPROVED,
    CLASS_NEW_FINDING,
    CLASS_OTHER,
    CLASS_REPEAT_ONLY,
    PlanStepBackContext,
    classify_review,
    derive_plan_step_back_state,
    entry_payload_for_plan,
    mandatory_plan_findings_since,
    plan_growth_crossing_round,
    plan_step_back_candidate_rounds,
    render_plan_step_back_guidance,
    render_step_back_human_decision,
    step_back_alternative_summary,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    PostedRoundRecord,
    UnresolvedReviewItem,
    _decode_round_metadata,
    _decode_round_metadata_mapping,
    _encode_round_metadata,
)
from coding_review_agent_loop.round_transport import decode_mapping

PRIMARY = "Codex"
DIGEST = "0123456789abcdef"


def _item(item_id, status, *, reviewer=PRIMARY, round_number=1, text="finding"):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer=reviewer,
        source_round=round_number,
        text=text,
        status=status,
    )


def _review(index, round_number, *, state="blocking", items=(), agent=PRIMARY):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent=agent,
            round_number=round_number,
            subject="plan",
            state=state,
            new_items=tuple(items),
        ),
        body="",
    )


def _checkpoint(index, round_number, *, digest=None, reset=False, phase="primary"):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject="plan",
            phase="scheduler-prelaunch",
            scheduler_metadata_status="valid",
            scheduler_phase=phase,
            scheduler_issue_digest=digest,
            scheduler_stall_reset=reset,
        ),
        body="",
    )


def _coder(index, round_number, *, entries=(), raw=None, status=None):
    kwargs = {}
    if status is not None:
        kwargs["step_back_status"] = status
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=round_number,
            subject="plan",
            step_back_entries=tuple(entries),
            raw_structured_coder_response=raw,
            **kwargs,
        ),
        body="",
    )


def _qualify(records):
    """Give every primary review a same-round primary checkpoint just before it."""
    have = {
        (r.metadata.round_number, r.index)
        for r in records
        if r.metadata.role == "summary" and r.metadata.phase == "scheduler-prelaunch"
    }
    rounds_with_checkpoint = {number for number, _index in have}
    extra = []
    for record in records:
        metadata = record.metadata
        if metadata.role == "reviewer" and metadata.round_number not in rounds_with_checkpoint:
            rounds_with_checkpoint.add(metadata.round_number)
            extra.append(_checkpoint(record.index - 0.5, metadata.round_number, digest=None))
    return [*records, *extra]


def _derive(records, **kwargs):
    return derive_plan_step_back_state(_qualify(list(records)), **kwargs)


def _new(n):
    return _item(f"item-{n}", "blocking", round_number=n, text=f"Gap {n}\nmore detail")


def _chain(rounds):
    """Primary new-finding reviews for each round, each with its checkpoint."""
    records = []
    index = 0
    for n in rounds:
        records.append(_checkpoint(index, n, digest=DIGEST))
        records.append(_review(index + 1, n, items=[_new(n)]))
        index += 2
    return records


# --- classification --------------------------------------------------------


def test_new_blocking_or_same_plan_item_is_a_new_finding():
    assert classify_review(_review(0, 1, items=[_new(1)]).metadata) == CLASS_NEW_FINDING
    same = _review(0, 1, items=[_item("item-2", "same-plan")]).metadata
    assert classify_review(same) == CLASS_NEW_FINDING


def test_old_blocker_plus_new_future_item_is_repeat_only():
    """`plan-repeat-only-blocks`: classified from the historical review record."""
    review = _review(0, 2, items=[_item("item-9", "future", round_number=2)]).metadata
    assert classify_review(review) == CLASS_REPEAT_ONLY


def test_blocking_review_with_no_new_items_is_repeat_only():
    assert classify_review(_review(0, 2).metadata) == CLASS_REPEAT_ONLY


def test_item_owned_by_another_reviewer_does_not_make_a_new_finding():
    review = _review(0, 2, items=[_item("item-3", "blocking", reviewer="Gemini")]).metadata
    assert classify_review(review) == CLASS_REPEAT_ONLY


def test_approval_and_other_states_are_not_blocks():
    assert classify_review(_review(0, 1, state="approved").metadata) == CLASS_APPROVED
    assert classify_review(_review(0, 1, state="unavailable").metadata) == CLASS_OTHER


def test_pr_phase_counts_same_pr_items():
    metadata = _review(0, 1, items=[_item("item-1", "same-pr")]).metadata
    assert classify_review(metadata, phase="pr") == CLASS_NEW_FINDING
    assert classify_review(metadata, phase="plan") == CLASS_REPEAT_ONLY


# --- trigger streak --------------------------------------------------------


def test_trigger_streak_counts_consecutive_new_findings():
    state = _derive(_chain([1, 2, 3]), primary=PRIMARY)
    assert state.streak_since(1) == 3
    assert state.streak_since(2) == 2
    # Reviews before the crossing never count.
    assert state.streak_since(4) == 0


def test_repeat_only_review_ends_the_streak():
    records = [*_chain([1]), _checkpoint(10, 2), _review(11, 2), *_chain([3])[0:0]]
    records += [_checkpoint(12, 3), _review(13, 3, items=[_new(3)])]
    state = _derive(records, primary=PRIMARY)
    assert state.streak_since(1) == 1


def test_approval_ends_the_streak():
    records = [*_chain([1]), _review(5, 2, state="approved"), _review(6, 3, items=[_new(3)])]
    state = _derive(records, primary=PRIMARY)
    assert state.streak_since(1) == 1


def test_other_reviewers_never_count():
    records = [*_chain([1, 2]), _review(9, 3, items=[_new(3)], agent="Gemini")]
    state = _derive(records, primary=PRIMARY)
    assert state.streak_since(1) == 2
    assert [review.round_number for review in state.reviews] == [1, 2]


def test_a_later_record_for_a_round_supersedes_an_earlier_one():
    records = [_review(0, 1, items=[_new(1)]), _review(1, 1)]
    state = _derive(records, primary=PRIMARY)
    assert state.reviews[0].classification == CLASS_REPEAT_ONLY


# --- episode ---------------------------------------------------------------


def _episode_records(*extra_reviews):
    """Step-back published at round 3 (triggered by round 2's review)."""
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    return [*_chain([1, 2]), _coder(10, 3, entries=[entry]), *extra_reviews]


def test_no_entry_means_no_episode():
    state = _derive(_chain([1, 2]), primary=PRIMARY)
    assert state.episode is None
    assert state.escalation_count == 0


def test_episode_counts_the_step_back_candidates_own_block_first():
    records = _episode_records(_review(11, 3, items=[_new(3)]))
    state = _derive(records, primary=PRIMARY)
    assert state.episode is not None
    assert (state.episode.candidate_round, state.episode.trigger_round) == (3, 2)
    assert state.escalation_count == 1


def test_new_repeat_new_after_a_step_back_keeps_one_episode_and_counts_all():
    """`plan-episode-new-repeat-new`: repeat-only never closes the episode."""
    records = _episode_records(
        _review(11, 3, items=[_new(3)]),
        _review(12, 4),
        _review(13, 5, items=[_new(5)]),
    )
    state = _derive(records, primary=PRIMARY)
    assert state.episode is not None
    assert state.escalation_count == 3
    # The streak never spans the start of an episode.
    assert state.streak_since(1) == 1


def test_reviews_before_the_step_back_candidate_do_not_count_toward_escalation():
    state = _derive(_episode_records(), primary=PRIMARY)
    assert state.episode is not None
    assert state.escalation_count == 0


def test_primary_approval_closes_the_episode():
    records = _episode_records(_review(11, 3, state="approved"))
    state = _derive(records, primary=PRIMARY)
    assert state.episode is None
    assert state.escalation_count == 0


def test_panel_opening_closes_the_episode():
    records = _episode_records(_review(11, 3, items=[_new(3)]))
    state = _derive(records, primary=PRIMARY, panel_opening_index=11)
    assert state.episode is None


def test_entry_owned_by_another_reviewer_does_not_open_the_primarys_episode():
    entry = entry_payload_for_plan(reviewer="Gemini", trigger_round=2)
    records = [*_chain([1, 2]), _coder(10, 3, entries=[entry])]
    assert _derive(records, primary=PRIMARY).episode is None


def test_the_newest_entry_wins():
    first = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    second = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=5)
    records = [
        *_chain([1, 2]),
        _coder(10, 3, entries=[first]),
        _coder(20, 6, entries=[second]),
        _review(21, 6, items=[_new(6)]),
    ]
    state = _derive(records, primary=PRIMARY)
    assert state.episode.candidate_round == 6
    assert state.escalation_count == 1


# --- reset and issue edit (`plan-episode-reset`) -----------------------------


def test_operator_reset_retires_earlier_rounds_and_the_episode():
    records = [
        *_episode_records(_review(11, 3, items=[_new(3)]), _review(12, 4, items=[_new(4)])),
        _checkpoint(30, 5, digest=DIGEST, reset=True),
        _review(31, 5, items=[_new(5)]),
    ]
    state = _derive(records, primary=PRIMARY, current_issue_digest=DIGEST)
    assert state.episode is None
    assert state.escalation_count == 0
    assert [review.round_number for review in state.reviews] == [5]
    assert state.streak_since(1) == 1


def test_issue_digest_change_retires_rounds_reviewed_against_other_text():
    records = _episode_records(_review(11, 3, items=[_new(3)]))
    # Rounds 1 and 2 were checkpointed against the old digest; round 3's was too.
    records.insert(-1, _checkpoint(10, 3, digest=DIGEST))
    state = _derive(
        records, primary=PRIMARY, current_issue_digest="fedcba9876543210"
    )
    assert state.episode is None
    assert state.reviews == ()


def test_rounds_without_a_recorded_digest_are_not_retired_by_an_edit():
    records = _episode_records(_review(11, 3, items=[_new(3)]))
    records = [r for r in records if r.metadata.role != "summary"]
    state = _derive(
        records, primary=PRIMARY, current_issue_digest="fedcba9876543210"
    )
    assert state.episode is not None


def test_phase_advance_record_is_a_history_boundary():
    advance = PostedRoundRecord(
        index=11,
        metadata=PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=3,
            subject="plan", phase="plan-phase-advance",
        ),
        body="",
    )
    records = _episode_records(advance, _review(12, 4, items=[_new(4)]))
    state = _derive(records, primary=PRIMARY)
    assert state.episode is None
    assert [review.round_number for review in state.reviews] == [4]


# --- malformed entries (`plan-malformed-marker`) -----------------------------


@pytest.mark.parametrize(
    "value",
    [
        "nope",
        [],
        [{"phase": "plan"}],
        [{"phase": "plan", "reviewer": "Codex", "trigger_round": 0}],
        [{"phase": "plan", "reviewer": "Codex", "trigger_round": True}],
        [{"phase": "plan", "reviewer": " ", "trigger_round": 2}],
        [{"phase": "other", "reviewer": "Codex", "trigger_round": 2}],
        [{"phase": "plan", "reviewer": "Codex", "trigger_round": 2, "anchor": {}}],
        [{"phase": "plan", "reviewer": "Codex", "trigger_round": 2, "extra": 1}],
        [{"phase": "pr", "reviewer": "Codex", "trigger_round": 2}],
        [{"phase": [], "reviewer": "Codex", "trigger_round": 2}],
        [{"phase": {}, "reviewer": "Codex", "trigger_round": 2}],
        [{"phase": ["plan"], "reviewer": "Codex", "trigger_round": 2}],
    ],
)
def test_malformed_entries_decode_as_degraded_not_as_an_error(value):
    payload = dict(decode_mapping(_encode_round_metadata(_coder(0, 3).metadata)))
    payload["step_back_entries"] = value
    metadata = _decode_round_metadata_mapping(payload)
    assert metadata.step_back_status == "invalid"
    assert metadata.step_back_entries == ()


def test_degraded_history_suppresses_the_state_with_a_flag():
    bad = _coder(10, 3, status="invalid")
    state = _derive([*_chain([1, 2]), bad], primary=PRIMARY)
    assert state.degraded
    assert state.episode is None


def test_entry_validation_rejects_a_malformed_entry_at_construction():
    with pytest.raises(ValueError):
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=3, subject="plan",
            step_back_entries=({"phase": "plan", "reviewer": "", "trigger_round": 2},),
        )


def test_entries_round_trip_and_legacy_records_decode_unchanged():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    record = _coder(0, 3, entries=[entry]).metadata
    decoded = _decode_round_metadata(_encode_round_metadata(record))
    assert decoded.step_back_entries == (dict(entry),)
    assert decoded.step_back_status == "valid"
    legacy = _coder(0, 3).metadata
    assert "step_back_entries" not in decode_mapping(_encode_round_metadata(legacy))
    assert _decode_round_metadata(_encode_round_metadata(legacy)).step_back_status == "absent"


def test_pr_shaped_entries_validate():
    entry = {
        "phase": "pr",
        "reviewer": "Codex",
        "trigger_round": 4,
        "trigger_head": "abcdef1",
        "anchor": {"path": "a/b.py", "start": 10, "end": 12},
    }
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent="Claude", round_number=5, subject="x",
        step_back_entries=(entry,),
    )
    assert metadata.step_back_status == "valid"


def test_candidate_rounds_reads_only_valid_plan_entries():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    records = [_coder(1, 3, entries=[entry]), _coder(2, 4), _coder(3, 5, status="invalid")]
    assert plan_step_back_candidate_rounds(records) == frozenset({3})


# --- growth crossing -------------------------------------------------------


def _candidate(index, round_number, *, chars):
    payload = json.loads(structured_v1_plan_state().split("\n<!--")[0])
    text = "x" * chars
    sidecar = make_assembled_plan_sidecar(
        payload, round_number=round_number, response_form="fresh-plan-state",
        rendered_plan=text,
    )
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=round_number,
            subject=f"s{round_number}", canonical_plan=text,
            assembled_plan_sidecar=sidecar.to_payload(),
            response_form="fresh-plan-state",
            aggregate_plan_identity=sidecar.aggregate_identity,
        ),
        body="",
    )


def test_crossing_round_is_the_earliest_of_the_current_contiguous_crossed_run(tmp_path):
    config = make_config(tmp_path, plan_growth_max_chars=1000)
    records = [
        _candidate(0, 1, chars=100),
        _candidate(1, 2, chars=1500),
        _candidate(2, 3, chars=1600),
        _candidate(3, 4, chars=1700),
    ]
    assert plan_growth_crossing_round(records, config=config, current_round=4) == 2


def test_an_uncrossed_newest_candidate_has_no_crossing(tmp_path):
    config = make_config(tmp_path, plan_growth_max_chars=1000)
    records = [_candidate(0, 1, chars=1500), _candidate(1, 2, chars=100)]
    assert plan_growth_crossing_round(records, config=config, current_round=2) is None


def test_a_dip_below_the_threshold_restarts_the_run(tmp_path):
    config = make_config(tmp_path, plan_growth_max_chars=1000)
    records = [
        _candidate(0, 1, chars=1500),
        _candidate(1, 2, chars=100),
        _candidate(2, 3, chars=1500),
    ]
    assert plan_growth_crossing_round(records, config=config, current_round=3) == 3


def test_growth_gate_off_disables_the_crossing(tmp_path):
    config = make_config(tmp_path, plan_growth_max_chars=1000, plan_growth_gate="off")
    assert plan_growth_crossing_round(
        [_candidate(0, 1, chars=1500)], config=config, current_round=1
    ) is None


def test_a_stale_current_round_has_no_crossing(tmp_path):
    config = make_config(tmp_path, plan_growth_max_chars=1000)
    records = [_candidate(0, 1, chars=1500), _candidate(1, 2, chars=1500)]
    assert plan_growth_crossing_round(records, config=config, current_round=3) is None


# --- rendering -------------------------------------------------------------


def test_findings_since_lists_mandatory_items_by_id_and_first_line():
    records = _chain([1, 2, 3])
    lines = mandatory_plan_findings_since(records, primary=PRIMARY, first_round=2)
    assert lines == (
        "[item-2] (round 2) Gap 2",
        "[item-3] (round 3) Gap 3",
    )


def test_alternative_summary_is_read_from_the_stored_structured_response():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    raw = json.dumps({"summary": "simpler design: round boundaries only"}) + "\n<!-- X -->"
    records = [*_chain([1, 2]), _coder(10, 3, entries=[entry], raw=raw)]
    state = _derive(records, primary=PRIMARY)
    assert step_back_alternative_summary(records, state.episode) == (
        "simpler design: round boundaries only"
    )


def test_alternative_summary_tolerates_missing_or_unreadable_responses():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    for raw in (None, "not json"):
        records = [*_chain([1, 2]), _coder(10, 3, entries=[entry], raw=raw)]
        state = _derive(records, primary=PRIMARY)
        assert step_back_alternative_summary(records, state.episode) is None


def test_human_decision_message_has_measurements_round_excerpt_and_options():
    message = render_step_back_human_decision(
        phase="plan",
        measurements="Measurements: 130744 chars.",
        step_back_round=3,
        blocks=2,
        threshold=2,
        alternative="A" * 5000,
    )
    assert "Measurements: 130744 chars." in message
    assert "round 3" in message
    assert "--plan-step-back-escalation-rounds 2" in message
    assert "--plan-step-back-rounds 0" in message and "--plan-reset-stall-streak" in message
    assert "narrow the issue text" in message
    assert "--plan-execution-mode auto" in message
    # The excerpt is bounded.
    assert message.count("A") <= 2000


def test_guidance_offers_staging_only_outside_implement_one_shot():
    def render(mode):
        return render_plan_step_back_guidance(
            PlanStepBackContext(
                measurements="M.", findings=("[item-1] x",), execution_mode=mode, streak=2
            )
        )

    one_shot = render("implement-one-shot")
    assert "re-filed as staged work" in one_shot
    assert "Scope-ledger preservation" not in one_shot
    for mode in ("auto", "plan-only", "implement-by-phase"):
        assert "Scope-ledger preservation still applies" in render(mode)
    assert "[item-1] x" in one_shot and "Patching the newest finding" in one_shot


def test_unknown_phase_is_rejected_by_the_message_renderer():
    with pytest.raises(ValueError):
        render_step_back_human_decision(
            phase="pr", measurements="", step_back_round=1, blocks=1, threshold=1,
            alternative=None,
        )


def test_agent_loop_error_is_the_config_error_type(tmp_path):
    with pytest.raises(AgentLoopError):
        make_config(
            tmp_path,
            reviewer=("codex", "gemini"),
            plan_step_back_rounds=-1,
        )


def test_reset_checkpoint_in_the_step_back_candidates_own_round_closes_the_episode():
    """A reset recorded after the step-back turn closes it, even in the same round."""
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    records = [
        *_chain([1, 2]),
        _coder(10, 3, entries=[entry]),
        _checkpoint(11, 3, digest=DIGEST, reset=True),
        _review(12, 3, items=[_new(3)]),
    ]
    state = _derive(records, primary=PRIMARY, current_issue_digest=DIGEST)
    assert state.episode is None
    assert state.escalation_count == 0
    # The reset round's own review still counts toward a fresh streak.
    assert [review.round_number for review in state.reviews] == [3]


def test_a_non_reset_checkpoint_in_the_candidates_round_keeps_the_episode():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=2)
    records = [
        *_chain([1, 2]),
        _coder(10, 3, entries=[entry]),
        _checkpoint(11, 3, digest=DIGEST),
        _review(12, 3, items=[_new(3)]),
    ]
    state = _derive(records, primary=PRIMARY, current_issue_digest=DIGEST)
    assert state.episode is not None and state.escalation_count == 1


def test_reviews_without_a_qualifying_primary_checkpoint_are_not_counted():
    """Legacy or full-board history ends the streak instead of lengthening it."""
    records = [
        _review(1, 1, items=[_new(1)]),  # no checkpoint at all (legacy)
        _checkpoint(2, 2, digest=DIGEST, phase="full-board"),
        _review(3, 2, items=[_new(2)]),  # non-primary checkpoint
        _checkpoint(4, 3, digest=DIGEST),
        _review(5, 3, items=[_new(3)]),
    ]
    state = derive_plan_step_back_state(records, primary=PRIMARY)
    assert [review.round_number for review in state.reviews] == [3]
    assert state.streak_since(1) == 1


def test_a_checkpoint_recorded_after_its_review_does_not_qualify_it():
    records = [_review(1, 1, items=[_new(1)]), _checkpoint(2, 1, digest=DIGEST)]
    assert derive_plan_step_back_state(records, primary=PRIMARY).reviews == ()
