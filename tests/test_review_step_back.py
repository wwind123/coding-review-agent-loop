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


OLD_DIGEST = "aaaaaaaaaaaaaaaa"
NEW_DIGEST = "bbbbbbbbbbbbbbbb"


def test_a_same_round_re_checkpoint_under_the_edited_issue_keeps_its_review():
    records = [
        _checkpoint(1, 3, digest=OLD_DIGEST),
        _review(2, 3, items=[_new(3)]),
        _checkpoint(3, 4, digest=OLD_DIGEST),  # obsolete: interrupted before its review
        _checkpoint(4, 4, digest=NEW_DIGEST),
        _review(5, 4, items=[_new(4)]),
        _checkpoint(6, 5, digest=NEW_DIGEST),
        _review(7, 5, items=[_new(5)]),
    ]
    state = derive_plan_step_back_state(records, primary=PRIMARY, current_issue_digest=NEW_DIGEST)
    assert [review.round_number for review in state.reviews] == [4, 5]
    assert state.streak_since(1) == 2


def test_an_edit_closes_an_episode_published_before_its_first_checkpoint():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=4)
    records = [
        _checkpoint(1, 4, digest=OLD_DIGEST),
        _review(2, 4, items=[_new(4)]),
        _coder(3, 5, entries=[entry]),
        _checkpoint(4, 5, digest=NEW_DIGEST),
        _review(5, 5, items=[_new(5)]),
    ]
    state = derive_plan_step_back_state(records, primary=PRIMARY, current_issue_digest=NEW_DIGEST)
    assert state.episode is None and state.escalation_count == 0
    assert [review.round_number for review in state.reviews] == [5]
    # Without an edit the same history keeps the episode and counts the block.
    same = derive_plan_step_back_state(records[:3], primary=PRIMARY, current_issue_digest=OLD_DIGEST)
    assert same.episode is not None


def test_an_episode_published_after_the_edit_is_not_closed_by_it():
    entry = entry_payload_for_plan(reviewer=PRIMARY, trigger_round=4)
    records = [
        _checkpoint(1, 3, digest=OLD_DIGEST),
        _review(2, 3, items=[_new(3)]),
        _checkpoint(3, 4, digest=NEW_DIGEST),
        _review(4, 4, items=[_new(4)]),
        _coder(5, 5, entries=[entry]),
        _checkpoint(6, 5, digest=NEW_DIGEST),
        _review(7, 5, items=[_new(5)]),
    ]
    state = derive_plan_step_back_state(records, primary=PRIMARY, current_issue_digest=NEW_DIGEST)
    assert state.episode is not None and state.escalation_count == 1


# ---------------------------------------------------------------------------
# PR fix-loop step-back (#1251, stage 2)
# ---------------------------------------------------------------------------

from coding_review_agent_loop.protocol import ReviewItemDisposition, ReviewSubItem
from coding_review_agent_loop import review_step_back as sb

HEAD_A = "a" * 40
HEAD_B = "b" * 40
HEAD_C = "c" * 40
PR_REVIEWER = "OpenAI Codex"


def _pr_item(item_id, text, *, status="blocking", round_number=1, sub_items=(), fix_scope=None):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer=PR_REVIEWER,
        source_round=round_number,
        text=text,
        status=status,
        sub_items=tuple(sub_items),
        fix_scope=fix_scope,
    )


def _pr_review(index, round_number, head, *, state="blocking", new=(), prior=(), dispositions=()):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent=PR_REVIEWER,
            round_number=round_number,
            subject=head,
            state=state,
            new_items=tuple(new),
            prior_items=tuple(prior),
            dispositions=tuple(dispositions),
        ),
        body="",
    )


def _pr_coder(index, round_number, head, *, entries=(), raw=None):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr",
            role="coder",
            agent="Claude",
            round_number=round_number,
            subject=head,
            step_back_entries=tuple(entries),
            raw_structured_coder_response=raw,
        ),
        body="",
    )


def _pr_entry(*, trigger_round=3, head=HEAD_A, path="src/spool.py", start=100, end=105):
    return {
        "phase": "pr",
        "reviewer": PR_REVIEWER,
        "trigger_round": trigger_round,
        "trigger_head": head,
        "anchor": {"path": path, "start": start, "end": end},
    }


def _identity_mapper(from_head, to_head, path, start, end):
    return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)


def test_project_finding_locations_reads_parent_fix_scope_and_sub_items():
    item = _pr_item(
        "item-7",
        "Undecodable file classified absent at `src/spool.py:124-131` and see also "
        "https://example.com/x.py:9 and src/readme.md without a line",
        fix_scope=("src/other.py:40",),
        sub_items=(
            ReviewSubItem("sub-1", "invalid identity at tests/test_spool.py:12-13"),
            ReviewSubItem("sub-2", "already fixed at src/gone.py:5", status="resolved"),
        ),
    )
    locations = sb.project_finding_locations(item)
    assert [(loc.path, loc.start, loc.end, loc.item_id, loc.sub_item_id) for loc in locations] == [
        ("src/spool.py", 124, 131, "item-7", None),
        ("src/other.py", 40, 40, "item-7", None),
        ("tests/test_spool.py", 12, 13, "item-7", "sub-1"),
    ]


def test_map_anchor_insertion_above_is_shifted():
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 40,
        name_status="M\tsrc/a.py\n",
        diff_text="@@ -10,0 +11,100 @@\n+x\n",
    )
    assert (mapping.outcome, mapping.start, mapping.end) == (sb.OUTCOME_SHIFTED, 200, 205)


def test_map_anchor_in_window_rewrite_is_rewritten_without_widening_to_the_hunk():
    # Old lines 98-110 replaced by 20 lines; anchor 100-105 lies inside the hunk.
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 40,
        name_status="M\tsrc/a.py\n",
        diff_text="@@ -98,13 +98,20 @@\n",
    )
    assert mapping.outcome == sb.OUTCOME_REWRITTEN
    assert (mapping.start, mapping.end) == (98, 117)


def test_map_anchor_rewrite_spilling_into_unrelated_code_is_unmappable():
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 10,
        name_status="M\tsrc/a.py\n",
        diff_text="@@ -60,80 +60,90 @@\n",
    )
    assert mapping.outcome == sb.OUTCOME_UNMAPPABLE
    assert "unrelated" in mapping.reason


def test_map_anchor_pure_deletion_or_move_is_unmappable():
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 40,
        name_status="M\tsrc/a.py\n",
        diff_text="@@ -99,8 +98,0 @@\n",
    )
    assert mapping.outcome == sb.OUTCOME_UNMAPPABLE
    deleted = sb.map_anchor(
        "src/a.py", 100, 105, 40, name_status="D\tsrc/a.py\n", diff_text=""
    )
    assert deleted.outcome == sb.OUTCOME_UNMAPPABLE and "deleted" in deleted.reason
    assert sb.map_anchor(
        "src/a.py", 1, 2, 40, name_status=None, diff_text=None
    ).outcome == sb.OUTCOME_UNMAPPABLE


def test_map_anchor_pairs_a_rename_to_a_different_directory():
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 40,
        name_status="R095\tsrc/a.py\tlib/core/a.py\n",
        diff_text="@@ -1,0 +2,3 @@\n",
    )
    assert (mapping.outcome, mapping.path, mapping.start, mapping.end) == (
        sb.OUTCOME_SHIFTED, "lib/core/a.py", 103, 108,
    )
    assert mapping.original_path == "src/a.py"


def test_in_cluster_uses_the_window_and_the_unmappable_fallback():
    mapped = sb.AnchorMapping(sb.OUTCOME_SHIFTED, "src/a.py", "src/a.py", 100, 105)
    near = sb.FindingLocation("src/a.py", 135, 135, "item-9")
    far = sb.FindingLocation("src/a.py", 146, 146, "item-9")
    other = sb.FindingLocation("src/b.py", 100, 100, "item-9")
    assert sb.in_cluster(near, mapped, 40)
    assert not sb.in_cluster(far, mapped, 40)
    assert not sb.in_cluster(other, mapped, 40)
    lost = sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, "src/a.py", "lib/a.py", 100, 105, "moved")
    assert sb.in_cluster(far, lost, 40)
    assert sb.in_cluster(sb.FindingLocation("lib/a.py", 900, 900, "x"), lost, 40)
    assert not sb.in_cluster(other, lost, 40)


def _loc(path, start, end, item="item-1"):
    return sb.MappedLocation(path, start, end, item)


def test_find_cluster_adjacent_overlapping_unrelated_and_missing():
    adjacent = [[_loc("a.py", 124, 131)], [_loc("a.py", 133, 136)], [_loc("a.py", 150, 160)]]
    cluster = sb.find_cluster(adjacent, 40)
    assert (cluster.path, cluster.start, cluster.end) == ("a.py", 124, 160)
    assert sb.find_cluster([[_loc("a.py", 1, 5)], [_loc("a.py", 3, 9)]], 0).end == 9
    assert sb.find_cluster([[_loc("a.py", 1, 5)], [_loc("b.py", 1, 5)]], 40) is None
    assert sb.find_cluster([[_loc("a.py", 1, 5)], [_loc("a.py", 500, 505)]], 40) is None
    assert sb.find_cluster([[_loc("a.py", 1, 5)], []], 40) is None


def _trigger_records(texts, *, heads=None):
    heads = heads or [HEAD_A] * len(texts)
    return [
        _pr_review(
            10 + n, n + 1, heads[n], new=[_pr_item(f"item-{n + 1}", text, round_number=n + 1)]
        )
        for n, text in enumerate(texts)
    ]


def test_cluster_trigger_needs_k_consecutive_clustered_new_findings():
    records = _trigger_records(
        ["gap src/spool.py:124-131", "gap src/spool.py:131-136", "gap src/spool.py:140"]
    )
    trigger = sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=_identity_mapper
    )
    assert trigger is not None
    assert (trigger.cluster.path, trigger.cluster.start, trigger.cluster.end) == (
        "src/spool.py", 124, 140,
    )
    assert trigger.trigger_head == HEAD_A and trigger.trigger_round == 3
    assert sb.pr_step_back_entry_payload(trigger)["anchor"] == {
        "path": "src/spool.py", "start": 124, "end": 140,
    }
    assert sb.find_pr_cluster_trigger(
        records[1:], PR_REVIEWER, k=3, window=40, current_round=3, mapper=_identity_mapper
    ) is None


@pytest.mark.parametrize(
    "texts",
    [
        ["src/a.py:10", "src/b.py:10", "src/a.py:12"],  # unrelated files
        ["src/a.py:10", "src/a.py:900", "src/a.py:12"],  # beyond the window
        ["src/a.py:10", "no reference here", "src/a.py:12"],  # no line reference
    ],
)
def test_cluster_trigger_ignores_unrelated_unparseable_and_distant_findings(texts):
    assert sb.find_pr_cluster_trigger(
        _trigger_records(texts), PR_REVIEWER, k=3, window=40, current_round=3,
        mapper=_identity_mapper,
    ) is None


def test_cluster_trigger_ignores_repeat_only_and_unmappable_locations():
    records = _trigger_records(["src/a.py:10", "src/a.py:11", "src/a.py:12"])
    repeat = _pr_review(
        40, 3, HEAD_A,
        prior=[_pr_item("item-2", "src/a.py:11")],
        dispositions=[ReviewItemDisposition("item-2", PR_REVIEWER, "blocking")],
    )
    assert sb.find_pr_cluster_trigger(
        [*records[:2], repeat], PR_REVIEWER, k=3, window=40, current_round=3,
        mapper=_identity_mapper,
    ) is None

    def lost(from_head, to_head, path, start, end):
        return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, path, start, end, "moved")

    mixed = _trigger_records(["src/a.py:10", "src/a.py:11", "src/a.py:12"], heads=[HEAD_B, HEAD_B, HEAD_A])
    assert sb.find_pr_cluster_trigger(
        mixed, PR_REVIEWER, k=3, window=40, current_round=3, mapper=lost
    ) is None


def test_cluster_trigger_maps_older_locations_to_the_newest_head():
    def shift(from_head, to_head, path, start, end):
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start + 100, end + 100)

    records = _trigger_records(
        ["src/a.py:10", "src/a.py:110", "src/a.py:111"], heads=[HEAD_B, HEAD_A, HEAD_A]
    )
    trigger = sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=0, current_round=3, mapper=shift
    )
    assert trigger is not None and (trigger.cluster.start, trigger.cluster.end) == (110, 111)


def test_cluster_trigger_rearms_only_from_reviews_after_a_step_back():
    records = [
        *_trigger_records(["src/a.py:10", "src/a.py:11", "src/a.py:12"]),
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]),
        _pr_review(60, 4, HEAD_B, new=[_pr_item("item-9", "src/a.py:13", round_number=4)]),
    ]
    assert sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=4, mapper=_identity_mapper
    ) is None


def test_malformed_pr_step_back_entries_suppress_trigger_and_episode():
    bad = PostedRoundRecord(
        index=45,
        metadata=PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=4, subject=HEAD_B,
            step_back_status="invalid",
        ),
        body="",
    )
    records = [*_trigger_records(["src/a.py:10", "src/a.py:11", "src/a.py:12"]), bad]
    assert sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=_identity_mapper
    ) is None
    assert sb.derive_pr_episode(records, PR_REVIEWER, window=40, mapper=_identity_mapper).entry is None


def _episode(*extra, window=40):
    records = [_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]), *extra]
    return sb.derive_pr_episode(records, PR_REVIEWER, window=window, mapper=_identity_mapper)


def test_sweep_sibling_in_the_cluster_escalates_even_when_located_only_in_a_sub_item():
    sibling = _pr_item(
        "item-12",
        "Another spool branch is misclassified",
        round_number=4,
        sub_items=(
            ReviewSubItem("sub-1", "branch at src/spool.py:130 treats it as absent"),
            ReviewSubItem("sub-2", "elsewhere at src/other.py:5"),
        ),
    )
    result = _episode(_pr_review(60, 4, HEAD_B, new=[sibling]))
    assert result.entry is not None and result.sibling_round == 4
    assert [(loc.item_id, loc.sub_item_id) for loc in result.siblings] == [("item-12", "sub-1")]


def test_carried_item_does_not_escalate_but_keeps_the_episode_open():
    carried = _pr_item("item-3", "old gap src/spool.py:130", round_number=3)
    result = _episode(
        _pr_review(
            60, 4, HEAD_B, prior=[carried],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        )
    )
    assert result.entry is not None and not result.siblings


def test_findings_outside_the_cluster_neither_escalate_nor_close_the_episode():
    carried = _pr_item("item-3", "old gap src/spool.py:130", round_number=3)
    unrelated = _pr_item("item-12", "unrelated src/other.py:5", round_number=4)
    result = _episode(
        _pr_review(
            60, 4, HEAD_B, new=[unrelated], prior=[carried],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        )
    )
    assert result.entry is not None and not result.siblings


def test_clearance_and_escalation_share_the_window_membership_rule():
    carried = _pr_item("item-3", "old gap src/spool.py:130", round_number=3)
    keep_open = _pr_review(
        60, 4, HEAD_B, prior=[carried],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
    )
    sibling = _pr_review(
        70, 5, HEAD_B,
        new=[_pr_item("item-13", "src/spool.py:135", round_number=5)],
        prior=[carried],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
    )
    live = _episode(keep_open, sibling)
    assert live.entry is not None and live.sibling_round == 5
    # A resume replays the same records and reaches the same decision.
    assert _episode(keep_open, sibling).sibling_round == 5
    # Once the carried in-window item is resolved the episode closes, so a later
    # sibling is an ordinary finding that can re-arm the trigger.
    resolved = _pr_review(
        61, 4, HEAD_B, prior=[carried],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "resolved")],
    )
    later = _pr_review(
        71, 5, HEAD_B, new=[_pr_item("item-13", "src/spool.py:135", round_number=5)]
    )
    assert _episode(resolved, later).entry is None


def test_approval_closes_the_pr_episode():
    assert _episode(_pr_review(60, 4, HEAD_B, state="approved")).entry is None


def test_unmappable_anchor_escalates_on_any_same_path_finding():
    def lost(from_head, to_head, path, start, end):
        return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, path, start, end, "moved")

    records = [
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]),
        _pr_review(60, 4, HEAD_B, new=[_pr_item("item-12", "src/spool.py:900", round_number=4)]),
    ]
    result = sb.derive_pr_episode(records, PR_REVIEWER, window=40, mapper=lost)
    assert result.siblings and not result.mapping.mappable
    message = sb.render_pr_step_back_human_decision(
        reviewer=PR_REVIEWER, entry=result.entry, mapping=result.mapping,
        siblings=result.siblings, window=40, generalization=None, sibling_round=4,
    )
    assert "UNMAPPABLE" in message and "--pr-step-back-rounds 0" in message


def test_sweep_entries_bind_to_the_resulting_head_and_round_only():
    records = [_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()])]
    assert set(sb.pr_sweep_entries(records, round_number=4, head_sha=HEAD_B)) == {PR_REVIEWER}
    assert sb.pr_sweep_entries(records, round_number=4, head_sha=HEAD_C) == {}
    assert sb.pr_sweep_entries(records, round_number=5, head_sha=HEAD_B) == {}
    assert sb.pr_sweep_entries(records, round_number=4, head_sha=None) == {}


def test_generalization_is_reconstructed_from_the_stored_structured_response():
    raw = json.dumps({"kind": "coder_followup", "summary": "Generalization: any malformed record"})
    records = [_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()], raw=raw)]
    entry = sb.pr_step_back_history(records, PR_REVIEWER)[0][0]
    assert sb.step_back_generalization(records, entry) == "Generalization: any malformed record"
    assert sb.step_back_generalization([_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()])], entry) is None


def test_pr_step_back_entry_round_trips_through_round_metadata():
    metadata = _pr_coder(1, 4, HEAD_B, entries=[_pr_entry()]).metadata
    decoded = _decode_round_metadata(_encode_round_metadata(metadata))
    assert decoded.step_back_status == "valid"
    assert decoded.step_back_entries[0]["anchor"] == {"path": "src/spool.py", "start": 100, "end": 105}
