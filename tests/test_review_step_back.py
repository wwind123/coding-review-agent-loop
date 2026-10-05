"""Unit tests for the deterministic step-back bookkeeping (#1251)."""

from __future__ import annotations

import json

import pytest

from agent_loop_helpers import make_config, structured_v1_plan_state
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.plan_assembly import make_assembled_plan_sidecar
from coding_review_agent_loop.review_step_back import (
    CLASS_APPROVED,
    CLASS_BLOCKING_UNRESOLVED,
    CLASS_NEW_FINDING,
    CLASS_OTHER,
    CLASS_REPEAT_ONLY,
    PlanStepBackContext,
    classify_review,
    derive_plan_step_back_state,
    effective_new_items,
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


# --- review round 1 fixes (#1265) -------------------------------------------


def test_clearance_applies_the_reviews_own_sub_item_dispositions():
    sub_items = (
        ReviewSubItem("sub-1", "in cluster at src/spool.py:130"),
        ReviewSubItem("sub-2", "distant at src/spool.py:900"),
    )
    parent = _pr_item("item-3", "conjunctive gap", round_number=3, sub_items=sub_items)

    def episode(outcomes):
        review = _pr_review(
            60, 4, HEAD_B, prior=[parent],
            dispositions=[
                ReviewItemDisposition("item-3", PR_REVIEWER, "blocking", sub_item_dispositions=outcomes)
            ],
        )
        return _episode(review)

    # Resolving the in-cluster sub-item while the distant one stays open clears the episode.
    assert episode((("sub-1", "resolved"), ("sub-2", "unresolved"))).entry is None
    # Reopening an in-cluster sub-item (previously resolved) keeps it open.
    closed_first = _pr_item(
        "item-3", "conjunctive gap", round_number=3,
        sub_items=(
            ReviewSubItem("sub-1", "in cluster at src/spool.py:130", status="resolved"),
            ReviewSubItem("sub-2", "distant at src/spool.py:900"),
        ),
    )
    review = _pr_review(
        60, 4, HEAD_B, prior=[closed_first],
        dispositions=[
            ReviewItemDisposition(
                "item-3", PR_REVIEWER, "blocking",
                sub_item_dispositions=(("sub-1", "unresolved"), ("sub-2", "resolved")),
            )
        ],
    )
    assert _episode(review).entry is not None


def _shifting_mapper(from_head, to_head, path, start, end):
    # The coder inserted 100 lines between the anchor (100-105) and later code.
    if (from_head, to_head) == (HEAD_A, HEAD_B) and start >= 110:
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start + 100, end + 100)
    if (from_head, to_head) == (HEAD_B, HEAD_A) and start >= 210:
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start - 100, end - 100)
    return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)


def test_clearance_maps_carried_locations_into_the_current_heads_coordinates():
    carried = _pr_item("item-3", "old gap src/spool.py:130", round_number=3)
    records = [
        _pr_review(40, 3, HEAD_A, new=[carried]),
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry(head=HEAD_A)]),
        _pr_review(
            60, 4, HEAD_B, prior=[carried],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        ),
    ]
    # Line 130 moved to 230, outside 100-105 +/- 40: no member remains.
    result = sb.derive_pr_episode(records, PR_REVIEWER, window=40, mapper=_shifting_mapper)
    assert result.entry is None
    # Without the shift the carried item is still a member (control).
    kept = sb.derive_pr_episode(records, PR_REVIEWER, window=40, mapper=_identity_mapper)
    assert kept.entry is not None


def test_a_formerly_distant_carried_item_that_moves_into_the_window_keeps_the_episode_open():
    carried = _pr_item("item-3", "old gap src/spool.py:300", round_number=3)

    def pulled_in(from_head, to_head, path, start, end):
        if (from_head, to_head) == (HEAD_A, HEAD_B) and start >= 290:
            return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start - 190, end - 190)
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)

    records = [
        _pr_review(40, 3, HEAD_A, new=[carried]),
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry(head=HEAD_A)]),
        _pr_review(
            60, 4, HEAD_B, prior=[carried],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        ),
    ]
    assert sb.derive_pr_episode(records, PR_REVIEWER, window=40, mapper=pulled_in).entry is not None


def test_find_cluster_ignores_a_distant_finding_on_the_same_path():
    cluster = sb.find_cluster(
        [[_loc("a.py", 100, 100), _loc("a.py", 900, 900)], [_loc("a.py", 110, 110)], [_loc("a.py", 120, 120)]],
        40,
    )
    assert (cluster.path, cluster.start, cluster.end) == ("a.py", 100, 120)
    # A component missing one review's finding is not a cluster.
    assert sb.find_cluster(
        [[_loc("a.py", 100, 100)], [_loc("a.py", 110, 110)], [_loc("a.py", 900, 900)]], 40
    ) is None


def test_map_anchor_uses_only_the_matching_file_section_of_a_rename_chain():
    name_status = "R100\ta.py\tb.py\nR100\tb.py\tc.py\n"
    diff = (
        "diff --git a/b.py b/c.py\nsimilarity index 90%\n@@ -1,0 +2,50 @@\n"
        "diff --git a/a.py b/b.py\nsimilarity index 100%\n"
    )
    mapping = sb.map_anchor("a.py", 100, 105, 40, name_status=name_status, diff_text=diff)
    assert (mapping.outcome, mapping.path, mapping.start, mapping.end) == (
        sb.OUTCOME_SHIFTED, "b.py", 100, 105,
    )
    other = sb.map_anchor("b.py", 100, 105, 40, name_status=name_status, diff_text=diff)
    assert (other.path, other.start) == ("c.py", 150)


def test_a_failed_hunk_read_keeps_the_known_rename_destination():
    mapping = sb.map_anchor(
        "src/a.py", 100, 105, 40, name_status="R100\tsrc/a.py\tlib/a.py\n", diff_text=None
    )
    assert mapping.outcome == sb.OUTCOME_UNMAPPABLE and mapping.path == "lib/a.py"
    assert sb.in_cluster(sb.FindingLocation("lib/a.py", 7, 7, "x"), mapping, 40)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("see Dockerfile:20", ("Dockerfile", 20, 20)),
        ("see Makefile:30-35", ("Makefile", 30, 35)),
        ("see scripts/run:40", ("scripts/run", 40, 40)),
        ("see src/a.py:7", ("src/a.py", 7, 7)),
        ("see run:40", ("run", 40, 40)),
        ("see configure:20-22", ("configure", 20, 22)),
    ],
)
def test_extensionless_repository_files_project_from_parent_and_sub_items(text, expected):
    parent = sb.project_finding_locations(_pr_item("item-1", text))
    assert [(l.path, l.start, l.end) for l in parent] == [expected]
    sub = sb.project_finding_locations(
        _pr_item("item-1", "plain", sub_items=(ReviewSubItem("sub-1", text), ReviewSubItem("sub-2", "x")))
    )
    assert [(l.path, l.start, l.end, l.sub_item_id) for l in sub] == [(*expected, "sub-1")]


@pytest.mark.parametrize(
    "text",
    ["https://example.com/x.py:9", "http://host/run:40", "/etc/passwd:3"],
)
def test_urls_absolute_paths_and_plain_words_do_not_project(text):
    assert sb.project_finding_locations(_pr_item("item-1", text)) == ()


# --- review round 2 fixes (#1265) -------------------------------------------


def test_a_parent_whose_sub_items_are_all_completed_no_longer_keeps_the_episode_open():
    parent = _pr_item(
        "item-3", "spool gap at src/spool.py:130", round_number=3,
        sub_items=(ReviewSubItem("sub-1", "a"), ReviewSubItem("sub-2", "b")),
    )
    completing = _pr_review(
        60, 4, HEAD_B, prior=[parent],
        dispositions=[
            ReviewItemDisposition(
                "item-3", PR_REVIEWER, "blocking",
                sub_item_dispositions=(("sub-1", "resolved"), ("sub-2", "resolved")),
            )
        ],
    )
    later = _pr_review(
        70, 5, HEAD_B, new=[_pr_item("item-9", "sibling src/spool.py:135", round_number=5)]
    )
    # The completing entry derives to resolved: the episode closed, so the later
    # in-window finding is an ordinary finding rather than a sibling.
    assert _episode(completing, later).entry is None


def test_a_future_item_promoted_by_the_sweep_keeps_the_episode_open_and_a_sibling_escalates():
    future = _pr_item("item-3", "retained gap src/spool.py:130", status="future", round_number=3)
    promoted = _pr_review(
        60, 4, HEAD_B, prior=[future],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
    )
    sibling = _pr_review(
        70, 5, HEAD_B, new=[_pr_item("item-9", "sibling src/spool.py:135", round_number=5)],
        prior=[future],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
    )
    assert _episode(promoted).entry is not None
    assert _episode(promoted, sibling).sibling_round == 5
    # A still-future item stays out of clearance.
    still_future = _pr_review(
        60, 4, HEAD_B, prior=[future],
        dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "future")],
    )
    assert _episode(still_future).entry is None


def test_nul_delimited_name_status_and_non_ascii_paths_map_through_the_runner(tmp_path):
    from coding_review_agent_loop.pr_loop_support import _git_anchor_mapper as _git_anchor_mapper_
    from coding_review_agent_loop.runner import CommandResult

    config = make_config(tmp_path)
    commands = []

    class Runner:
        def run(self, args, *, cwd, check=True, **_kw):
            args = list(args)
            commands.append(args)
            if "--name-status" in args:
                out = "M\0src/café.py\0R100\0src/old é.py\0lib/new é.py\0"
                return CommandResult(args, cwd, out, "", 0)
            diff = (
                "diff --git a/src/café.py b/src/café.py\n@@ -10,0 +11,100 @@\n"
                "diff --git a/src/old é.py b/lib/new é.py\n@@ -1,0 +2,3 @@\n"
            )
            return CommandResult(args, cwd, diff, "", 0)

    mapper = _git_anchor_mapper_(Runner(), config, checkout=tmp_path, window=40)
    moved = mapper(HEAD_A, HEAD_B, "src/café.py", 100, 105)
    assert (moved.outcome, moved.start, moved.end) == (sb.OUTCOME_SHIFTED, 200, 205)
    renamed = mapper(HEAD_A, HEAD_B, "src/old é.py", 100, 105)
    assert (renamed.path, renamed.start) == ("lib/new é.py", 103)
    assert all("core.quotePath=false" in c for c in commands)
    assert all("-z" in c for c in commands if "--name-status" in c)


def test_a_changed_file_whose_diff_section_is_missing_is_unmappable_not_unchanged():
    mapping = sb.map_anchor(
        "src/café.py", 100, 105, 40,
        name_status="M\tsrc/café.py\n",
        diff_text='diff --git "a/src/caf\\303\\251.py" "b/src/caf\\303\\251.py"\n@@ -10,0 +11,100 @@\n',
    )
    assert mapping.outcome == sb.OUTCOME_UNMAPPABLE
    # An unchanged file (not in the listing) with unrelated sections stays SHIFTED.
    other = sb.map_anchor(
        "src/a.py", 100, 105, 40,
        name_status="M\tsrc/b.py\n",
        diff_text="diff --git a/src/b.py b/src/b.py\n@@ -1,0 +2,3 @@\n",
    )
    assert (other.outcome, other.start) == (sb.OUTCOME_SHIFTED, 100)


def test_a_sibling_from_an_earlier_round_does_not_escalate_again_but_keeps_the_episode_open():
    sibling = _pr_item("item-12", "another branch src/spool.py:134", round_number=4)
    stale = _pr_review(60, 4, HEAD_B, new=[sibling])
    later = _pr_review(
        70, 5, HEAD_C,
        new=[_pr_item("item-13", "unrelated src/other.py:5", round_number=5)],
        prior=[sibling],
        dispositions=[ReviewItemDisposition("item-12", PR_REVIEWER, "blocking")],
    )
    records = [_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]), stale, later]
    # Dispatching round 5: the round-4 sibling already stopped the run once.
    result = sb.derive_pr_episode(
        records, PR_REVIEWER, window=40, mapper=_identity_mapper, current_round=5
    )
    assert result.entry is not None and not result.siblings
    # Dispatching round 4 itself still escalates, as does an unscoped replay.
    assert sb.derive_pr_episode(
        records, PR_REVIEWER, window=40, mapper=_identity_mapper, current_round=4
    ).sibling_round == 4
    assert sb.derive_pr_episode(
        records, PR_REVIEWER, window=40, mapper=_identity_mapper
    ).sibling_round == 4
    # A new sibling introduced by the dispatched round's own review escalates.
    newer = _pr_review(
        70, 5, HEAD_C, new=[_pr_item("item-13", "another src/spool.py:136", round_number=5)],
        prior=[sibling],
        dispositions=[ReviewItemDisposition("item-12", PR_REVIEWER, "blocking")],
    )
    assert sb.derive_pr_episode(
        [records[0], stale, newer], PR_REVIEWER, window=40, mapper=_identity_mapper, current_round=5
    ).sibling_round == 5


def test_a_sibling_reviewed_on_a_head_the_code_has_moved_past_does_not_escalate_again():
    sibling = _pr_item("item-12", "another branch src/spool.py:134", round_number=4)
    records = [
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]),
        _pr_review(60, 4, HEAD_B, new=[sibling]),
    ]
    same = sb.derive_pr_episode(
        records, PR_REVIEWER, window=40, mapper=_identity_mapper,
        current_round=4, current_head=HEAD_B,
    )
    assert same.sibling_round == 4  # an unchanged rerun stops again
    pushed = sb.derive_pr_episode(
        records, PR_REVIEWER, window=40, mapper=_identity_mapper,
        current_round=4, current_head=HEAD_C,
    )
    assert pushed.entry is not None and not pushed.siblings


# --- review round 6 fixes (#1265) -------------------------------------------


def test_trigger_locations_are_mapped_to_the_dispatch_head_not_the_newest_reviews_head():
    records = _trigger_records(["src/a.py:124", "src/a.py:130", "src/a.py:136"], heads=[HEAD_A] * 3)

    def deleted_since(from_head, to_head, path, start, end):
        if to_head == HEAD_B:
            return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, path, start, end, "deleted")
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)

    # Reviewed on HEAD_A, dispatched on HEAD_B where the code is gone: no trigger.
    assert sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=deleted_since,
        current_head=HEAD_B,
    ) is None
    assert sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=deleted_since,
    ) is not None  # without the dispatch head the stale review head was used

    def shifted(from_head, to_head, path, start, end):
        if to_head == HEAD_B:
            return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start + 100, end + 100)
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)

    trigger = sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=shifted,
        current_head=HEAD_B,
    )
    assert trigger.trigger_head == HEAD_B
    assert (trigger.cluster.start, trigger.cluster.end) == (224, 236)


def test_a_moved_distant_carried_item_does_not_keep_a_mappable_anchors_episode_open():
    near = _pr_item("item-3", "gap src/spool.py:130", round_number=3)
    far = _pr_item("item-6", "elsewhere src/spool.py:900", round_number=3)

    def mapper(from_head, to_head, path, start, end):
        if start >= 900:  # that code moved away: it cannot be placed on the new head
            return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, path, start, end, "moved")
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)

    sweep = _pr_review(
        60, 4, HEAD_B, prior=[near, far],
        dispositions=[
            ReviewItemDisposition("item-3", PR_REVIEWER, "resolved"),
            ReviewItemDisposition("item-6", PR_REVIEWER, "blocking"),
        ],
    )
    later = _pr_review(
        70, 5, HEAD_B, new=[_pr_item("item-9", "ordinary src/spool.py:135", round_number=5)],
        prior=[far], dispositions=[ReviewItemDisposition("item-6", PR_REVIEWER, "blocking")],
    )
    # Round 3's review (on HEAD_A) is the carried items' source head.
    source = _pr_review(40, 3, HEAD_A, new=[near, far])
    records = [source, _pr_coder(50, 4, HEAD_B, entries=[_pr_entry(head=HEAD_A)]), sweep, later]
    # The only in-window item was resolved, so the episode closed at the sweep; the
    # later in-window finding is ordinary.  Replaying the same records (a resume) agrees.
    for _ in range(2):
        result = sb.derive_pr_episode(
            records, PR_REVIEWER, window=40, mapper=mapper, current_round=5, current_head=HEAD_B
        )
        assert result.entry is None and not result.siblings
    # An unmappable ANCHOR still uses the same-path fallback for a carried item.
    lost = lambda *a: sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, a[2], a[2], a[3], a[4], "moved")
    kept = sb.derive_pr_episode(
        records[:3], PR_REVIEWER, window=40, mapper=lost, current_round=4, current_head=HEAD_B
    )
    assert kept.entry is not None


# --- review round 7 fix (#1265) ---------------------------------------------


def test_a_carried_locations_known_rename_destination_keeps_an_unmappable_anchors_episode_open():
    near = _pr_item("item-3", "gap src/a.py:130", round_number=3)
    far = _pr_item("item-6", "elsewhere src/b.py:900", round_number=4)

    def mapper(from_head, to_head, path, start, end):
        if (from_head, to_head) == (HEAD_A, HEAD_B) and path == "src/a.py":
            return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, "src/b.py", start, end)
        if (from_head, to_head) == (HEAD_A, HEAD_C) and path == "src/a.py":
            return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, "src/c.py", start, end, "rewritten")
        if (from_head, to_head) == (HEAD_B, HEAD_C) and path == "src/b.py":
            return sb.AnchorMapping(sb.OUTCOME_UNMAPPABLE, path, "src/c.py", start, end, "moved")
        return sb.AnchorMapping(sb.OUTCOME_SHIFTED, path, path, start, end)

    records = [
        _pr_review(40, 3, HEAD_A, new=[near]),
        _pr_coder(50, 4, HEAD_B, entries=[_pr_entry(head=HEAD_A, path="src/a.py")]),
        _pr_review(
            60, 4, HEAD_B, new=[far], prior=[near],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        ),
        # The near item is resolved; the distant one is carried onto a head where both
        # files were renamed to src/c.py and the anchored code was rewritten.
        _pr_review(
            70, 5, HEAD_C, prior=[near, far],
            dispositions=[
                ReviewItemDisposition("item-3", PR_REVIEWER, "resolved"),
                ReviewItemDisposition("item-6", PR_REVIEWER, "blocking"),
            ],
        ),
        _pr_review(
            80, 6, HEAD_C, new=[_pr_item("item-9", "sibling src/c.py:50", round_number=6)],
            prior=[far], dispositions=[ReviewItemDisposition("item-6", PR_REVIEWER, "blocking")],
        ),
    ]
    for _ in range(2):  # live, then a replay as a resume would see it
        result = sb.derive_pr_episode(
            records, PR_REVIEWER, window=40, mapper=mapper, current_round=6, current_head=HEAD_C
        )
        assert result.entry is not None and result.sibling_round == 6
    # Before the sibling arrives the episode is still open (the rename destination counts).
    assert sb.derive_pr_episode(
        records[:4], PR_REVIEWER, window=40, mapper=mapper, current_round=5, current_head=HEAD_C
    ).entry is not None


# ---------------------------------------------------------------------------
# --review-parallel: publication reviewer records resolve via reconciliation (#1271)
# ---------------------------------------------------------------------------


def _publication(index, round_number, *, state="blocking", agent=PRIMARY, flow="plan",
                 subject="plan", phase="publication", prior=(), dispositions=()):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow=flow, role="reviewer", agent=agent, round_number=round_number,
            subject=subject, state=state, phase=phase, new_items=(),
            prior_items=tuple(prior), dispositions=tuple(dispositions),
        ),
        body="",
    )


def _recon(index, round_number, items, *, flow="plan", subject="plan"):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow=flow, role="summary", agent="Orchestrator", round_number=round_number,
            subject=subject, phase="reconciliation", new_items=tuple(items),
        ),
        body="",
    )


def _parallel_chain(rounds):
    records = []
    index = 0
    for n in rounds:
        records.append(_checkpoint(index, n, digest=DIGEST))
        records.append(_publication(index + 1, n))
        records.append(_recon(index + 2, n, [_new(n)]))
        index += 3
    return records


def test_parallel_publication_records_classify_new_finding_and_build_a_streak():
    state = _derive(_parallel_chain([6, 7]), primary=PRIMARY)
    assert [r.classification for r in state.reviews] == [CLASS_NEW_FINDING] * 2
    assert state.streak_since(6) == 2


def test_publication_without_reconciliation_is_unresolved_and_ends_the_streak():
    records = _parallel_chain([6]) + [
        _checkpoint(10, 7, digest=DIGEST),
        _publication(11, 7),
    ]
    state = _derive(records, primary=PRIMARY)
    assert state.reviews[-1].classification == CLASS_BLOCKING_UNRESOLVED
    assert state.streak_since(6) == 0


def test_reconciliation_binding_ignores_earlier_other_round_and_other_subject():
    review = _publication(5, 7)
    records = [
        _recon(1, 7, [_new(7)]),  # before the reviewer record
        _recon(6, 8, [_new(8)]),  # other round
        _recon(7, 7, [_new(7)], subject="other"),  # other subject
        _recon(8, 7, [_new(7)], flow="pr"),  # other flow
        review,
    ]
    assert effective_new_items(records, review) is None


def test_latest_reconciliation_after_the_record_wins():
    review = _publication(5, 7)
    records = [review, _recon(6, 7, [_new(7)]), _recon(9, 7, [])]
    assert effective_new_items(records, review) == ()
    assert classify_review(review.metadata, new_items=()) == CLASS_REPEAT_ONLY


def test_only_the_reviewers_own_reconciliation_items_count():
    review = _publication(5, 7)
    other = _item("item-7", "blocking", reviewer="Claude", round_number=7)
    items = effective_new_items([review, _recon(6, 7, [other])], review)
    assert items == ()


def test_findings_since_lists_reconciliation_items():
    lines = mandatory_plan_findings_since(
        _parallel_chain([6, 7]), primary=PRIMARY, first_round=6
    )
    assert lines == ("[item-6] (round 6) Gap 6", "[item-7] (round 7) Gap 7")


def test_non_publication_records_use_their_own_items_and_ignore_reconciliations():
    review = _review(5, 7, items=[_new(7)])
    assert effective_new_items([review, _recon(6, 7, [])], review) == (_new(7),)
    assert classify_review(review.metadata) == CLASS_NEW_FINDING


def test_escalation_count_includes_an_unresolved_block():
    records = _parallel_chain([6]) + [
        _coder(10, 7, entries=[entry_payload_for_plan(reviewer=PRIMARY, trigger_round=6)]),
        _checkpoint(11, 7, digest=DIGEST),
        _publication(12, 7),
    ]
    state = _derive(records, primary=PRIMARY)
    assert state.episode is not None
    assert state.escalation_count == 1


def test_approved_publication_closes_the_episode_without_a_reconciliation():
    records = _parallel_chain([6]) + [
        _coder(10, 7, entries=[entry_payload_for_plan(reviewer=PRIMARY, trigger_round=6)]),
        _checkpoint(11, 7, digest=DIGEST),
        _publication(12, 7, state="approved"),
    ]
    state = _derive(records, primary=PRIMARY)
    assert state.reviews[-1].classification == CLASS_APPROVED
    assert state.episode is None


def _pr_publication(index, round_number, head, **kwargs):
    return _publication(index, round_number, agent=PR_REVIEWER, flow="pr", subject=head, **kwargs)


def _pr_recon(index, round_number, head, items):
    return _recon(index, round_number, items, flow="pr", subject=head)


def test_pr_cluster_trigger_fires_on_parallel_publication_plus_reconciliation_rounds():
    texts = ["gap src/spool.py:124-131", "gap src/spool.py:131-136", "gap src/spool.py:140"]
    records = []
    for n, text in enumerate(texts):
        records.append(_pr_publication(10 + 2 * n, n + 1, HEAD_A))
        records.append(
            _pr_recon(11 + 2 * n, n + 1, HEAD_A,
                      [_pr_item(f"item-{n + 1}", text, round_number=n + 1)])
        )
    trigger = sb.find_pr_cluster_trigger(
        records, PR_REVIEWER, k=3, window=40, current_round=3, mapper=_identity_mapper
    )
    assert trigger is not None and len(trigger.findings) == 3
    # An unresolved newest round breaks the K-tail.
    assert sb.find_pr_cluster_trigger(
        records[:-1], PR_REVIEWER, k=3, window=40, current_round=3, mapper=_identity_mapper
    ) is None


def test_pr_episode_detects_a_sibling_found_only_on_the_reconciliation():
    sibling = _pr_item("item-12", "spool branch src/spool.py:130", round_number=4)
    result = _episode(_pr_publication(60, 4, HEAD_B), _pr_recon(61, 4, HEAD_B, [sibling]))
    assert result.entry is not None and result.sibling_round == 4


def test_pr_unresolved_post_entry_review_neither_escalates_nor_closes_the_episode():
    carried = _pr_item("item-3", "old gap src/spool.py:130", round_number=3)
    result = _episode(
        _pr_review(
            55, 4, HEAD_B, prior=[carried],
            dispositions=[ReviewItemDisposition("item-3", PR_REVIEWER, "blocking")],
        ),
        _pr_publication(60, 5, HEAD_B),
    )
    assert result.entry is not None and not result.siblings
    only = _episode(_pr_publication(60, 4, HEAD_B))
    assert only.entry is not None and not only.siblings


def test_pr_approved_publication_closes_the_episode():
    assert _episode(_pr_publication(60, 4, HEAD_B, state="approved")).entry is None


def test_pr_reconciliation_member_item_keeps_the_episode_open_until_resolved():
    member = _pr_item("item-12", "spool branch src/spool.py:130", round_number=4)

    def replay(*extra):
        records = [
            _pr_publication(60, 4, HEAD_B),
            _pr_recon(61, 4, HEAD_B, [member]),
            *extra,
        ]
        # current_round=5: round 4's sibling is stale, so escalation is suppressed
        # and the open member is evaluated through _remaining_mandatory_items.
        return sb.derive_pr_episode(
            [_pr_coder(50, 4, HEAD_B, entries=[_pr_entry()]), *records],
            PR_REVIEWER, window=40, mapper=_identity_mapper,
            current_round=5, current_head=HEAD_B,
        )

    kept = replay()
    assert kept.entry is not None and not kept.siblings
    still_open = replay(
        _pr_publication(
            70, 5, HEAD_B, prior=[member],
            dispositions=[ReviewItemDisposition("item-12", PR_REVIEWER, "blocking")],
        ),
        _pr_recon(71, 5, HEAD_B, []),
    )
    assert still_open.entry is not None and not still_open.siblings
    closed = replay(
        _pr_publication(
            70, 5, HEAD_B, prior=[member],
            dispositions=[ReviewItemDisposition("item-12", PR_REVIEWER, "resolved")],
        ),
        _pr_recon(71, 5, HEAD_B, []),
    )
    assert closed.entry is None
