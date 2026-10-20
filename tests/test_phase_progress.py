"""Read-only, prefix-bounded staged progress resolution (#1331)."""

import pytest

import coding_review_agent_loop.phase_progress as phase_progress
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.execution_policy import staged_outcome_from_summary
from coding_review_agent_loop.integration_close import IntegrationMergeVerification
from coding_review_agent_loop.phase_progress import (
    STATUS_COMPLETE,
    STATUS_IN_PROGRESS,
    STATUS_NOT_DISPATCHED,
    resolve_staged_phase_progress,
    unmet_phase_prerequisites,
)

from agent_loop_helpers import FakeRunner, make_config
from dependency_guard_helpers import (
    GuardScenario,
    handoff_for,
    install,
    plan_phase,
    summary_for,
)

PHASES = (plan_phase("s1"), plan_phase("s2", depends_on=("s1",)), plan_phase("s3"))


def _outcome():
    return staged_outcome_from_summary(summary_for(PHASES), mode="implement-by-phase")


def _merged_open_stage_one(monkeypatch, verification):
    scenario = GuardScenario(
        phases=PHASES, stage_id="s2", handoffs=(handoff_for("s1"),),
        child_states={61: "OPEN"}, canonical_prs={61: (71, "MERGED")},
        verifications={61: verification},
    )
    install(monkeypatch, scenario)
    return scenario


def _resolve(tmp_path, **kwargs):
    return resolve_staged_phase_progress(
        FakeRunner(), config=make_config(tmp_path), parent_issue=55, parent_comments=(),
        outcome=_outcome(), **kwargs,
    )


def test_verified_integration_merge_counts_complete_without_closing(tmp_path, monkeypatch):
    """Row dep-integration-verified: the closure write is never reached."""
    scenario = _merged_open_stage_one(
        monkeypatch, IntegrationMergeVerification("integration-verified", base_ref="integration/x")
    )
    progress = _resolve(tmp_path, read_only=True, through_phase_index=2)
    assert [phase.status for phase in progress] == [STATUS_COMPLETE]
    assert progress[0].child_state == "OPEN"
    assert progress[0].integration_outcome == "integration-verified"
    assert ("verify", 61) in scenario.reads
    unmet = unmet_phase_prerequisites(
        FakeRunner(), config=make_config(tmp_path), parent_issue=55, parent_comments=(),
        outcome=_outcome(), phase_index=2, declared_stage_ids=("s1",),
    )
    assert unmet == ()


@pytest.mark.parametrize(
    "verification",
    [
        IntegrationMergeVerification("unresolved", base_ref="integration/x",
                                      reason="base integration/x is not trusted"),
        IntegrationMergeVerification("unresolved", reason="the PR could not be read"),
        IntegrationMergeVerification("default-branch", base_ref="main"),
    ],
    ids=["untrusted-base", "unreadable", "default-branch-not-closed"],
)
def test_unverified_integration_merge_is_incomplete_not_an_error(
    tmp_path, monkeypatch, verification
):
    """Row dep-integration-unverified: a MERGED PR alone is not delivery."""
    _merged_open_stage_one(monkeypatch, verification)
    progress = _resolve(tmp_path, read_only=True, through_phase_index=2)
    assert [phase.status for phase in progress] == [STATUS_IN_PROGRESS]
    assert progress[0].integration_outcome == verification.outcome
    assert progress[0].integration_reason == verification.reason
    unmet = unmet_phase_prerequisites(
        FakeRunner(), config=make_config(tmp_path), parent_issue=55, parent_comments=(),
        outcome=_outcome(), phase_index=2, declared_stage_ids=("s1",),
    )
    assert [(item.progress.stage_id, item.declared) for item in unmet] == [("s1", True)]


def test_through_phase_index_ignores_later_handoffs(tmp_path, monkeypatch):
    # Stage 1 not dispatched while stage 3 already holds a handoff: the
    # default path rejects the out-of-order record, the bounded one ignores it.
    scenario = GuardScenario(phases=PHASES, stage_id="s2", handoffs=(handoff_for("s3"),))
    install(monkeypatch, scenario)
    with pytest.raises(AgentLoopError, match="delivered in order"):
        _resolve(tmp_path)
    progress = _resolve(tmp_path, read_only=True, through_phase_index=2)
    assert [(phase.stage_id, phase.status) for phase in progress] == [
        ("s1", STATUS_NOT_DISPATCHED)
    ]
    assert scenario.reads == []


def test_default_arguments_keep_the_closure_capable_parent_path(tmp_path, monkeypatch):
    """Row parent-path-unchanged: the parent closes; it never only verifies."""
    scenario = _merged_open_stage_one(
        monkeypatch, IntegrationMergeVerification("integration-verified", base_ref="integration/x")
    )
    closed = []
    monkeypatch.setattr(
        phase_progress, "close_child_after_integration_merge",
        lambda *_a, **kw: closed.append(kw["pr_number"]) or "closed",
    )
    progress = _resolve(tmp_path)
    assert closed == [71]
    assert progress[0].status == STATUS_COMPLETE and progress[0].child_state == "CLOSED"
    assert ("verify", 61) not in scenario.reads
    assert [phase.stage_id for phase in progress] == ["s1", "s2", "s3"]
