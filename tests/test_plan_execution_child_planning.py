"""Parent-driven child planning applies the edited-approved-plan gate (#2138).

``plan_execution._run_child_planning_cycle`` dispatches child recovery and the
child plan loop itself, bypassing ``run_issue_loop``.  The shared gate runs
there too, before either, on the fresh and the resume path, with the child
config (which never inherits the parent's ``--keep-approved-plan``).
"""

from __future__ import annotations

import pytest

import coding_review_agent_loop.issue_recovery as issue_recovery
import coding_review_agent_loop.plan_execution as plan_execution
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.issue_pr_handoff import AGENT_ISSUE_PR_HANDOFF_RE
from coding_review_agent_loop.validated_agent import _new_usage_context
from test_child_plan_provenance import _ChildPlanningRunner, _M936World

_NOTE_MARK = "changed after plan `"


def _notes(runner):
    return [
        str(item["body"]) for item in runner.issue_comments
        if _NOTE_MARK in str(item["body"])
    ]


def _world(tmp_path, monkeypatch, *, handed_off):
    # The world's durable plan history was approved under other issue text, so
    # the served child reads as edited after approval.
    world = _M936World(tmp_path, monkeypatch, weak=False, signed=False)
    if not handed_off:
        world.comments = [
            c for c in world.comments if not AGENT_ISSUE_PR_HANDOFF_RE.search(c.body)
        ]
    world.runner = _ChildPlanningRunner(pr_payload={
        "number": 77, "body": "Fixes #56", "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
    })
    return world


def _install_spies(monkeypatch):
    """Spy on child recovery, the child plan loop and the PR route; returns the calls."""
    recoveries, plan_loops, pr_loops = [], [], []
    real_recover = issue_recovery.recover_issue_execution

    def recover_spy(runner, **kwargs):
        recoveries.append(kwargs)
        return real_recover(runner, **kwargs)

    def plan_loop_spy(_runner, **kwargs):
        plan_loops.append(kwargs)
        return 0

    def pr_loop_spy(_runner, **kwargs):
        pr_loops.append(kwargs)
        return 0

    monkeypatch.setattr(issue_recovery, "recover_issue_execution", recover_spy)
    monkeypatch.setattr(issue_recovery, "run_pr_loop", pr_loop_spy)
    return recoveries, plan_loops, pr_loops, plan_loop_spy


def _run_cycle(world, *, resume, plan_loop):
    # The parent run carries --keep-approved-plan; the child must not inherit it.
    config = world.config(keep_approved_plan=True)
    return plan_execution._run_child_planning_cycle(
        world.runner, run_plan_first_loop=plan_loop, config=config, memory=None,
        usage_context=_new_usage_context(config), parent_issue=55, child_issue_number=56,
        resume=resume, parent_issue_context=world.parent,
    )


@pytest.mark.parametrize("resume", [True, False])
def test_edited_approved_child_plan_stops_child_planning_despite_the_parent_keep_flag(
    tmp_path, monkeypatch, resume
):
    """`child-resume-edited-stop`: stop before child recovery or the child plan loop."""
    world = _world(tmp_path, monkeypatch, handed_off=False)
    recoveries, plan_loops, pr_loops, plan_loop = _install_spies(monkeypatch)
    with pytest.raises(HumanDecisionRequiredError) as raised:
        _run_cycle(world, resume=resume, plan_loop=plan_loop)

    message = str(raised.value)
    assert raised.value.EXIT_CODE == 4
    assert "Issue #56:" in message and world.old_hash in message
    assert "--keep-approved-plan" in message and "--reopen-approved-plan" in message
    assert "agent-loop issue 56 --plan-first" in message
    assert "recorded digest" in message and "current digest" in message
    assert recoveries == [] and plan_loops == [] and pr_loops == []
    assert world.agent_calls("claude") == [] and world.agent_calls("codex") == []
    notes = _notes(world.runner)
    assert len(notes) == 1 and "--keep-approved-plan" in notes[0]


def test_handed_off_child_plan_resume_notes_once_and_continues_into_the_pr_route(
    tmp_path, monkeypatch
):
    """`child-resume-handed-off-note`: one note on the child; recovery continues."""
    world = _world(tmp_path, monkeypatch, handed_off=True)
    recoveries, plan_loops, pr_loops, plan_loop = _install_spies(monkeypatch)
    for _attempt in range(2):
        assert _run_cycle(world, resume=True, plan_loop=plan_loop) == 0

    assert len(recoveries) == 2
    assert plan_loops == []
    assert [call["pr_number"] for call in pr_loops] == [77, 77]
    notes = _notes(world.runner)
    assert len(notes) == 1
    assert "already handed off" in notes[0] and "PR #77" in notes[0]
    assert world.agent_calls("claude") == [] and world.agent_calls("codex") == []
