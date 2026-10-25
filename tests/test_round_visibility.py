"""Same-round visibility with a reopening boundary (#1664)."""

from __future__ import annotations

from coding_review_agent_loop.pr_loop_support import _visibility_snapshot
from coding_review_agent_loop.round_state import PostedRoundMetadata, PostedRoundRecord
from coding_review_agent_loop.round_visibility import (
    same_round_public_publications,
    visible_peer_names,
)

REVIEWERS = ("Codex", "Gemini", "Antigravity")


def _record(index, *, role="reviewer", agent="Codex", phase="publication", round_number=3):
    return PostedRoundRecord(
        index=index,
        metadata=PostedRoundMetadata(
            flow="pr", role=role, agent=agent, round_number=round_number, subject="h", phase=phase,
        ),
        body="",
    )


def _history():
    return [
        _record(0, agent="Gemini"),  # before the round's coder record
        _record(1, role="coder", agent="Claude", phase="authoritative"),
        _record(2, agent="Antigravity"),  # pre-conversion pass
        _record(3, role="summary", agent="Orchestrator", phase="reconciliation"),
        _record(4, role="summary", agent="Orchestrator", phase="evidence-conversion"),
        _record(5, agent="Codex"),  # repeat pass
    ]


def _agents(records):
    return [record.metadata.agent for record in records]


def _peers(records, boundary=None):
    return visible_peer_names(
        records, opening_source=None, flow="pr", round_number=3, subject="h",
        reviewer_names=REVIEWERS, primary_reviewer=None, launch_phase=None, launching=(),
        checkpoint_index=None, reopening_boundary_index=boundary,
    )


def test_without_a_boundary_publications_are_unchanged():
    """`visibility-unit-boundary`: the default keeps today's coder-bounded output."""
    records = _history()
    default = same_round_public_publications(
        records, flow="pr", round_number=3, subject="h", reviewer_names=REVIEWERS
    )
    explicit = same_round_public_publications(
        records, flow="pr", round_number=3, subject="h", reviewer_names=REVIEWERS,
        reopening_boundary_index=None,
    )
    assert _agents(default) == _agents(explicit) == ["Antigravity", "Codex"]
    assert _peers(records) == {"Antigravity", "Codex"}


def test_a_boundary_excludes_publications_at_or_before_it():
    records = _history()
    assert _agents(same_round_public_publications(
        records, flow="pr", round_number=3, subject="h", reviewer_names=REVIEWERS,
        reopening_boundary_index=4,
    )) == ["Codex"]
    assert _peers(records, boundary=4) == {"Codex"}
    # A boundary at a publication's own index excludes it too.
    assert _peers(records, boundary=5) == set()


def test_coder_and_reopening_boundaries_combine_by_max():
    records = _history()
    # A reopening boundary below the coder record does not widen the window.
    assert _peers(records, boundary=-1) == _peers(records, boundary=0) == {"Antigravity", "Codex"}
    # A later coder record wins over an earlier reopening boundary.
    later_coder = [*records, _record(6, role="coder", agent="Claude", phase="authoritative"),
                   _record(7, agent="Gemini")]
    assert _peers(later_coder, boundary=4) == {"Gemini"}


def test_the_fallback_snapshot_carries_this_process_s_own_conversion_before_the_checkpoint():
    """`operator-conversion-visibility` (unit): the own record precedes the checkpoint."""
    base = [_record(0, role="coder", agent="Claude", phase="authoritative"), _record(1, agent="Antigravity")]
    checkpoint = _record(0, role="summary", agent="Orchestrator", phase="scheduler-prelaunch").metadata
    own = _record(0, role="summary", agent="Orchestrator", phase="evidence-conversion").metadata
    records, present = _visibility_snapshot(
        fresh_records=None, base_records=base, base_length=2, checkpoint=checkpoint, own_conversion=own,
    )
    assert present
    assert [record.metadata.phase for record in records[-2:]] == ["evidence-conversion", "scheduler-prelaunch"]
    conversion_index = records[-2].index
    assert conversion_index == 2 and records[-1].index == 3
    assert _peers(records, boundary=conversion_index) == set()
    # Without an own record, nothing is synthesized but the checkpoint.
    plain, _present = _visibility_snapshot(
        fresh_records=None, base_records=base, base_length=2, checkpoint=checkpoint,
    )
    assert [record.metadata.phase for record in plain[2:]] == ["scheduler-prelaunch"]
    # A readable fresh history never receives a synthesized conversion.
    fresh, _present = _visibility_snapshot(
        fresh_records=base, base_records=base, base_length=2, checkpoint=checkpoint, own_conversion=own,
    )
    assert "evidence-conversion" not in [record.metadata.phase for record in fresh]
