"""Execution-policy tests: execution-policy resolution and collection extraction (#1543)."""

import json
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.execution_policy import _infer_staged_parent_issue
from coding_review_agent_loop.protocol import validate_structured_plan_state
import coding_review_agent_loop.execution_policy as execution_policy
from issue_loop_test_helpers import _M948_FOOTER, _m948_stages
from agent_loop_helpers import (
    make_config,
    structured_plan_state,
    structured_v1_plan_state,
)
from coding_review_agent_loop.comment_rendering import render_public_agent_comment  # noqa: E402,F811
from coding_review_agent_loop.protocol import (
    validate_structured_plan_state as _m948_validate_state,
)


def test_auto_execution_resolves_one_shot_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation

    resolved = execution_policy._resolve_execution_policy(
        config,
        recommendation=recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-one-shot"
    assert resolved.strategy == "one-shot"


def test_auto_execution_resolves_staged_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation
    staged_recommendation = replace(recommendation, strategy="staged")

    resolved = execution_policy._resolve_execution_policy(
        config,
        recommendation=staged_recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-by-phase"
    assert resolved.strategy == "staged"


def test_auto_execution_rejects_legacy_undecided_plan(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")

    with pytest.raises(AgentLoopError, match="legacy-undecided"):
        execution_policy._resolve_execution_policy(
            config,
            recommendation=None,
        )


def test_infer_staged_parent_requires_generated_marker_or_body_header():
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Ordinary issue text.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Quoted AGENT_SPLIT_CHILD: parent=12 in review prose.",
            ),
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #13: quoted comment text.",
            ),
        ),
    )

    assert _infer_staged_parent_issue(issue_context) is None

    marked_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="<!-- AGENT_SPLIT_CHILD: parent=12 key=" + "a" * 64 + " -->",
        url=issue_context.url,
        comments=(),
    )
    assert _infer_staged_parent_issue(marked_context) == 12

    decomposed_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child phase issue for parent #13: https://github.com/OWNER/REPO/issues/13\n\nDetails.",
        url=issue_context.url,
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #14: quoted comment text.",
            ),
        ),
    )
    assert _infer_staged_parent_issue(decomposed_context) == 13


def test_m948_resumed_collections_equal_the_full_form_publication():
    payload = json.loads(structured_plan_state(summary="Collections plan.").split("\n", 1)[0])
    payload["deferred_stages"] = _m948_stages("Deferred", 5)
    payload["plan_actions"] = _m948_stages("Action", 5)
    payload["additional_closing_issue_ids"] = [101, 102]
    parsed = _m948_validate_state(json.dumps(payload) + _M948_FOOTER)
    canonical_plan = plan_first_loop_module.render_canonical_plan_state(parsed)

    # Stage and closing collections are read from the canonical plan only; the
    # digest carries none of their records.
    assert execution_policy._extract_current_deferred_stages(canonical_plan) == parsed.deferred_stages
    assert execution_policy._extract_current_expected_closing_issue_ids(canonical_plan) == (101, 102)
    digest = render_public_agent_comment(
        kind="plan_state", parsed=parsed, agent="claude", compact=True
    )
    assert execution_policy._extract_current_deferred_stages(digest) == ()
    assert execution_policy._extract_current_expected_closing_issue_ids(digest) is None


# ---------------------------------------------------------------------------
# Decomposition dependency guard on direct child entry (#1331)
# ---------------------------------------------------------------------------

from dataclasses import replace as _dc_replace  # noqa: E402
from types import SimpleNamespace as _NS  # noqa: E402

import coding_review_agent_loop.phase_progress as _phase_progress  # noqa: E402
from coding_review_agent_loop.errors import DependencyNotMergedError  # noqa: E402
from coding_review_agent_loop.phase_progress import StagedTopologyOutcome  # noqa: E402
from dependency_guard_helpers import (  # noqa: E402
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
    GuardScenario,
    IntegrationMergeVerification,
    comment_writes,
    handoff_for,
    install,
    issue_context,
    plan_phase,
    run_cli,
    summary_for,
)


def _two_stage(*, s1_disposition=EXECUTION_DISPOSITION_DIRECT, s1_automation="agent-pr",
               s2_depends=("s1",), s2_disposition=EXECUTION_DISPOSITION_DIRECT):
    return (
        plan_phase("s1", disposition=s1_disposition, automation=s1_automation),
        plan_phase("s2", disposition=s2_disposition, depends_on=s2_depends),
    )


@pytest.mark.parametrize(
    ("s1_disposition", "s1_command"),
    [
        (EXECUTION_DISPOSITION_DIRECT, "`agent-loop issue 61`"),
        (EXECUTION_DISPOSITION_PLANNING,
         "`agent-loop issue 61 --plan-first --plan-execution-mode auto`"),
    ],
)
def test_unstarted_stage_one_blocks_direct_stage_two_with_exit_5(
    tmp_path, monkeypatch, capsys, s1_disposition, s1_command
):
    """Row dep-open-direct-blocks: the issue's primary scenario."""
    scenario = GuardScenario(phases=_two_stage(s1_disposition=s1_disposition), stage_id="s2")
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "agent-loop: DEPENDENCY NOT MERGED" in err
    assert "dependency #61 (stage s1) is not merged" in err
    assert "declared dependency" in err
    assert f"stage s1 has not been started; start it with {s1_command}" in err
    assert err.rstrip().endswith("Then rerun: `agent-loop issue 62`")
    # Not-dispatched: no stage-1 child or PR state is read, and no PR merge
    # is recommended for a PR that does not exist.
    assert scenario.reads == []
    assert "merge PR" not in err
    assert scenario.events == []
    assert comment_writes(runner) == []


@pytest.mark.parametrize(
    ("setup", "expected"),
    [
        (
            dict(handoffs=(handoff_for("s1"),)),
            "stage s1 is in progress with no PR yet; resume it with `agent-loop issue 61`",
        ),
        (
            dict(handoffs=(handoff_for("s1"),), canonical_prs={61: (71, "OPEN")}),
            "PR #71 is OPEN; finish and merge PR #71 for #61",
        ),
        (
            dict(handoffs=(handoff_for("s1"),), canonical_prs={61: (71, "MERGED")},
                 verifications={61: IntegrationMergeVerification("default-branch", base_ref="main")}),
            "PR #71 merged but GitHub has not closed #61 yet; rerun shortly",
        ),
        (
            dict(handoffs=(handoff_for("s1"),), canonical_prs={61: (71, "MERGED")},
                 verifications={61: IntegrationMergeVerification(
                     "unresolved", base_ref="integration/x",
                     reason="base integration/x is not trusted")}),
            "PR #71 merged into integration/x but the integration merge could not be verified "
            "(base integration/x is not trusted); repair the recorded base or trust, or close "
            "#61 once verified",
        ),
    ],
    ids=["in-progress-no-pr", "open-pr", "default-branch-pending", "unresolved-integration"],
)
def test_each_unmet_status_renders_its_own_recovery_line(
    tmp_path, monkeypatch, capsys, setup, expected
):
    scenario = GuardScenario(phases=_two_stage(), stage_id="s2", **setup)
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "dependency #61 (stage s1) is not merged" in err
    assert expected in err
    assert scenario.events == []
    assert comment_writes(runner) == []


def test_open_human_dependency_blocks_and_names_the_closure(tmp_path, monkeypatch, capsys):
    """Row dep-human-open-blocks."""
    scenario = GuardScenario(
        phases=_two_stage(s1_automation="human-action", s1_disposition=None), stage_id="s2"
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "#61 is a human-owned human-action stage; close it once its work is done" in err
    assert ("canonical-pr", 61) not in scenario.reads
    assert scenario.events == []


def test_undeclared_earlier_stage_still_blocks_in_recorded_order(tmp_path, monkeypatch, capsys):
    """Row dep-undeclared-independent: stage 2 without a declared edge."""
    scenario = GuardScenario(phases=_two_stage(s2_depends=()), stage_id="s2")
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "dependency #61 (stage s1) is not merged" in err
    assert "earlier stage in the recorded order (staged phases are delivered in order)" in err
    assert scenario.events == []
    assert comment_writes(runner) == []


def test_stage_three_blocked_by_incomplete_undeclared_stage_two(tmp_path, monkeypatch, capsys):
    """Row dep-undeclared-independent: stage 3 depends only on a completed stage 1."""
    phases = (
        plan_phase("s1"),
        plan_phase("s2"),
        plan_phase("s3", depends_on=("s1",)),
    )
    scenario = GuardScenario(
        phases=phases, stage_id="s3", handoffs=(handoff_for("s1"),),
        child_states={61: "CLOSED"}, canonical_prs={61: (71, "MERGED")},
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "Issue #63 (stage s3) cannot start: dependency #62 (stage s2) is not merged" in err
    assert "#61 (stage s1" not in err
    assert "earlier stage in the recorded order" in err
    assert scenario.events == []


def test_completed_earlier_stages_proceed_to_dispatch(tmp_path, monkeypatch):
    """Row dep-merged-proceeds: agent stage MERGED and human stage CLOSED."""
    phases = (
        plan_phase("s1"),
        plan_phase("s2", automation="human-action", disposition=None),
        plan_phase("s3", depends_on=("s1", "s2")),
    )
    scenario = GuardScenario(
        phases=phases, stage_id="s3", handoffs=(handoff_for("s1"),),
        child_states={61: "CLOSED", 62: "CLOSED"}, canonical_prs={61: (71, "MERGED")},
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 23
    assert scenario.events == ["memory", "dispatch"]
    # Human stage: closure only, no PR evidence read.
    assert ("canonical-pr", 62) not in scenario.reads
    assert comment_writes(runner) == []


def test_closed_agent_dependency_without_merged_pr_fails_closed(tmp_path, monkeypatch, capsys):
    """Row dep-closed-agent-unmerged (agent case)."""
    scenario = GuardScenario(
        phases=_two_stage(), stage_id="s2", handoffs=(handoff_for("s1"),),
        child_states={61: "CLOSED"},
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 1
    err = capsys.readouterr().err
    assert "is CLOSED but no canonical issue-to-PR handoff record exists" in err
    assert "DEPENDENCY NOT MERGED" not in err
    assert scenario.events == []


def test_closed_human_dependency_proceeds_without_pr_evidence(tmp_path, monkeypatch):
    """Row dep-closed-agent-unmerged (human case)."""
    scenario = GuardScenario(
        phases=_two_stage(s1_automation="human-action", s1_disposition=None), stage_id="s2",
        child_states={61: "CLOSED"},
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 23
    assert scenario.reads == [("state", 61)]
    assert scenario.events == ["memory", "dispatch"]


def test_dependent_childs_own_out_of_order_handoff_still_exits_5(tmp_path, monkeypatch, capsys):
    """Row dep-own-handoff-out-of-order."""
    own = handoff_for("s2")
    scenario = GuardScenario(phases=_two_stage(), stage_id="s2", handoffs=(own,))
    runner = install(monkeypatch, scenario, handoff=own)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 5
    err = capsys.readouterr().err
    assert "dependency #61 (stage s1) is not merged" in err
    assert "Staged phases are delivered in order, so the parent stops" not in err
    assert scenario.events == []


def test_decompose_only_sibling_dispatched_directly_counts_complete(tmp_path, monkeypatch):
    """Row dep-decompose-only-handoff-mode."""
    scenario = GuardScenario(
        phases=_two_stage(), stage_id="s2", summary_mode="decompose-only",
        handoffs=(handoff_for("s1"),), child_states={61: "CLOSED"},
        canonical_prs={61: (71, "MERGED")},
    )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 23
    assert scenario.events == ["memory", "dispatch"]


@pytest.mark.parametrize("fault", ["missing-stage", "no-child-number", "not-earlier"])
def test_unresolvable_dependency_provenance_fails_closed_with_exit_1(
    tmp_path, monkeypatch, capsys, fault
):
    """Row dep-provenance-broken."""
    phases = _two_stage(
        s2_depends=("ghost",) if fault == "missing-stage" else
        ("s2",) if fault == "not-earlier" else ("s1",)
    )
    scenario = GuardScenario(phases=phases, stage_id="s2")
    if fault == "no-child-number":
        summary = summary_for(phases)
        title, url, _number = summary.children[0]
        scenario.summary = _dc_replace(
            summary, children=((title, url, None), *summary.children[1:])
        )
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 1
    err = capsys.readouterr().err
    assert "DEPENDENCY NOT MERGED" not in err
    assert "repair the decomposition summary" in err
    assert scenario.events == []
    assert comment_writes(runner) == []


def test_stage_one_is_never_blocked(tmp_path, monkeypatch):
    scenario = GuardScenario(phases=_two_stage(), stage_id="s1")
    runner = install(monkeypatch, scenario)
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 23
    assert scenario.reads == []


def test_parent_implement_by_phase_path_keeps_resume_hint(tmp_path, monkeypatch, capsys):
    """Row parent-path-unchanged: the parent never raises the dependency error."""
    phases = _two_stage()
    scenario = GuardScenario(
        phases=phases, stage_id="s1", handoffs=(handoff_for("s1"),),
        canonical_prs={61: (71, "OPEN")},
    )
    install(monkeypatch, scenario)
    fresh = scenario.fresh()
    monkeypatch.setattr(
        plan_first_loop_module, "find_existing_phase_implementation_handoff",
        lambda *_a, **_k: handoff_for("s1"),
    )
    outcome = execution_policy.staged_outcome_from_summary(
        summary_for(phases), mode="implement-by-phase"
    )
    result = plan_first_loop_module._dispatch_current_decomposition_phase(
        _FakeRunnerForParent(), config=make_config(tmp_path), memory=None,
        usage_context=_NS(), issue_number=55, current_plan="Approved parent plan",
        plan_subject="plan-subject", outcome=outcome,
        recommendation=fresh.recommendation, approved_plan_context=fresh.parent_plan_context,
        issue_context=issue_context(55), mode="implement-by-phase", coder_session_id=None,
    )
    assert result == 0
    assert "agent-loop issue 61" in capsys.readouterr().out
    assert "dispatch" not in scenario.events
    assert ("verify", 61) not in scenario.reads


def _FakeRunnerForParent():
    from agent_loop_helpers import FakeRunner

    return FakeRunner()


def test_staged_outcome_from_summary_pairs_children_with_recorded_automation():
    phases = _two_stage(s1_automation="human-action", s1_disposition=None)
    outcome = execution_policy.staged_outcome_from_summary(
        summary_for(phases, mode="decompose-only"), mode="implement-by-phase"
    )
    assert isinstance(outcome, StagedTopologyOutcome)
    assert outcome.mode == "implement-by-phase"
    assert outcome.stage_ids == ("s1", "s2")
    assert outcome.automations == ("human-action", "agent-pr")
    assert [item.issue_number for item in outcome.created] == [61, 62]


def test_dependency_error_message_lists_each_prerequisite():
    from coding_review_agent_loop.errors import UnmetDependency

    error = DependencyNotMergedError(
        child_issue=63, stage_id="s3", relaunch_command="`agent-loop issue 63`",
        prerequisites=(
            UnmetDependency(61, "s1", 1, True, "not-dispatched", None, None, None, "start it"),
            UnmetDependency(62, "s2", 2, False, "human-pending", "OPEN", None, None, "close it"),
        ),
    )
    text = str(error)
    assert text.startswith(
        "Issue #63 (stage s3) cannot start: dependency #61 (stage s1) is not merged."
    )
    assert "- #61 (stage s1, phase 1; declared dependency): start it" in text
    assert "- #62 (stage s2, phase 2; earlier stage in the recorded order" in text
    assert error.EXIT_CODE == 5
    assert _phase_progress.STATUS_NOT_DISPATCHED == "not-dispatched"


def test_open_dependency_with_verified_integration_merge_proceeds_to_dispatch(
    tmp_path, monkeypatch
):
    """Row dep-integration-verified through child entry: dispatch proceeds, no close."""
    scenario = GuardScenario(
        phases=_two_stage(), stage_id="s2", handoffs=(handoff_for("s1"),),
        child_states={61: "OPEN"}, canonical_prs={61: (71, "MERGED")},
        verifications={61: IntegrationMergeVerification(
            "integration-verified", base_ref="integration/x")},
    )
    runner = install(monkeypatch, scenario)
    # install() makes close_child_after_integration_merge raise if called.
    assert run_cli(monkeypatch, tmp_path, runner, scenario) == 23
    assert ("verify", 61) in scenario.reads
    assert scenario.events == ["memory", "dispatch"]
    assert comment_writes(runner) == []
