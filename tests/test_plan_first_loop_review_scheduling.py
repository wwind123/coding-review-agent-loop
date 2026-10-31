"""Plan-first loop tests: staged plan-review scheduling and finding provenance (#1543)."""

import json
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import ExitStack

import pytest

pytestmark = pytest.mark.usefixtures("private_response_tempdir")

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.comment_rendering import render_canonical_plan_state
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
)
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata
from coding_review_agent_loop.prompts import (
    COMPACT_PLANNING_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
    validate_structured_plan_state,
    risk_test_matrix_identity,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    _STEP_BACK_MARKER,
    _invalid_plan_scheduler_comment,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1251_prompts,
    _plan_round_records,
    _remediation_plan_fixtures,
    _repaired_acknowledgement_fixtures,
    _staged_plan_config,
)
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
from coding_review_agent_loop.plan_assembly import assemble_plan_revision  # noqa: E402
from coding_review_agent_loop.round_state import PostedRoundRecord  # noqa: E402
import coding_review_agent_loop.validated_agent as validated_agent_module


def _plan_audit_body(runner, *, round_number):
    """The posted planning scheduler audit body for one round."""
    for comment in runner.issue_comments:
        body = comment["body"]
        if not body.startswith("Plan review scheduling audit."):
            continue
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if (
            metadata.flow == "plan"
            and metadata.phase == "scheduler-prelaunch"
            and metadata.round_number == round_number
        ):
            return body
    raise AssertionError(f"no planning scheduler audit posted for round {round_number}")


def _staged_plan_history(
    tmp_path, *, reviewer_records=(), scheduler_record=True, reconciliation_record=False
):
    """A generation-1 planning round-1 coder record plus reviewer records.

    ``scheduler_record`` writes the round-1 planning scheduler checkpoint, so
    the history decodes as intact rather than as the ``absent`` degraded class.
    """
    config = _staged_plan_config(tmp_path)
    structured = validate_structured_plan_state(structured_v1_plan_state())
    canonical = render_canonical_plan_state(structured, config)
    sidecar = plan_first_loop_module.make_assembled_plan_sidecar(
        structured, round_number=1, response_form="fresh-plan-state", rendered_plan=canonical
    )
    subject = _plan_subject(canonical)
    comments = [
        {
            "author": {"login": "bot"},
            "createdAt": "2026-01-01T00:00:00Z",
            "body": _attach_round_metadata(
                canonical + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Anthropic Claude",
                    round_number=1,
                    subject=subject,
                    canonical_plan=canonical,
                    raw_structured_coder_response=structured_v1_plan_state(),
                    state="blocking",
                    execution_strategy_contract_version=1,
                    execution_strategy_identity=structured.execution_recommendation.identity(),
                    risk_test_matrix_contract_version=1,
                    risk_test_matrix_payload=structured.risk_test_matrix.to_payload(),
                    risk_test_matrix_changes_payload=(),
                    risk_test_matrix_identity=risk_test_matrix_identity(
                        structured.risk_test_matrix, structured.risk_test_matrix_changes
                    ),
                    risk_test_matrix_boundary_digest=risk_test_matrix_identity(
                        structured.risk_test_matrix, structured.risk_test_matrix_changes
                    ),
                    response_form="fresh-plan-state",
                    aggregate_plan_identity=sidecar.aggregate_identity,
                    assembled_plan_sidecar=sidecar.to_payload(),
                ),
            ),
        }
    ]
    if scheduler_record:
        contract = panel_evidence.make_plan_contract(
            ("Codex", "Gemini"), "primary-then-panel", "Codex"
        )
        key = panel_evidence._plan_candidate_key_for(
            plan_subject=subject, sidecar=sidecar, surfaced_requirement_ids=()
        )
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": "2026-01-01T00:00:00Z",
                "body": _attach_round_metadata(
                    "Plan review scheduling audit.\n\n-- Orchestrator",
                    PostedRoundMetadata(
                        flow="plan",
                        role="summary",
                        agent="Orchestrator",
                        round_number=1,
                        subject=subject,
                        phase="scheduler-prelaunch",
                        scheduler_contract=contract.as_dict(),
                        scheduler_obligation_digest="0" * 16,
                        scheduler_selected_reviewers=("Codex",),
                        scheduler_paused_reviewers=(("Gemini", "primary phase"),),
                        scheduler_reasons=("primary phase",),
                        scheduler_final_sweep=False,
                        scheduler_force_full=False,
                        scheduler_calls_avoided=1,
                        scheduler_phase="primary",
                        scheduler_primary_reviewer="Codex",
                        plan_candidate_key=key.as_dict(),
                    ),
                ),
            }
        )
    for index, (agent, state, items) in enumerate(reviewer_records, start=1):
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": f"2026-01-01T00:00:{index:02d}Z",
                "body": _attach_round_metadata(
                    f"{agent} plan review.\n<!-- AGENT_PLAN_STATE: {state} -->\n-- {agent}",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent=agent,
                        round_number=1,
                        subject=subject,
                        state=state,
                        new_items=items,
                    ),
                ),
            }
        )
    if reconciliation_record:
        # A reconciled round: the summary checkpoint repeats the round's items,
        # so a resume rehydrates them into the current-round ledger.
        reconciled_items = tuple(
            item for _agent, _state, items in reviewer_records for item in items
        )
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": "2026-01-01T00:01:00Z",
                "body": _attach_round_metadata(
                    "Plan round reconciliation.\n\n-- Orchestrator",
                    PostedRoundMetadata(
                        flow="plan",
                        role="summary",
                        agent="Orchestrator",
                        round_number=1,
                        subject=subject,
                        phase="reconciliation",
                        new_items=reconciled_items,
                    ),
                ),
            }
        )
    return comments, canonical


def test_default_full_board_planning_writes_no_scheduler_metadata(tmp_path):
    """`default-full-board-planning`: the compatibility path is unchanged."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    records = _plan_round_records(runner)
    assert records, "the planning flow must still post durable round records"
    assert all(record.scheduler_metadata_status == "absent" for record in records)
    # Reviewer records of a generation-1 plan carry the exact-plan key as
    # requirement-binding evidence only (#1377); no other record carries it.
    assert all(
        record.plan_candidate_key is None for record in records if record.role != "reviewer"
    )
    assert all(
        record.plan_candidate_key is not None for record in records if record.role == "reviewer"
    )
    assert all(record.phase != "scheduler-prelaunch" for record in records)
    reviewer_rounds = {
        (record.agent, record.round_number)
        for record in records
        if record.role == "reviewer"
    }
    # Every configured reviewer is invoked in the same planning round.
    assert reviewer_rounds == {("Codex", 1), ("Gemini", 1)}


def test_staged_planning_primary_gate_then_reviewer_only_panel_round(tmp_path):
    """`primary-blocking-revision`, `panel-opens-on-primary-approval`,
    `reviewer-only-phase-advance`, and `final-exact-plan-gate` together."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    assert run_issue_loop(
        runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
    ) == 0

    records = _plan_round_records(runner)
    prelaunch = [record for record in records if record.phase == "scheduler-prelaunch"]
    assert [record.scheduler_phase for record in prelaunch] == [
        "primary",
        "secondary-audit",
    ]
    # Round 1 is strictly primary-only; the secondary is paused, not dropped.
    assert prelaunch[0].scheduler_selected_reviewers == ("Codex",)
    assert [name for name, _why in prelaunch[0].scheduler_paused_reviewers] == ["Gemini"]
    assert prelaunch[0].scheduler_metadata_status == "valid"
    # The panel opens on the exact-plan primary approval, against the same key.
    assert prelaunch[1].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[1].scheduler_approved_reviewers
    assert prelaunch[0].plan_candidate_key == prelaunch[1].plan_candidate_key
    # The panel round is reviewer-only: one planner turn in the whole run.
    advance = [record for record in records if record.phase == "plan-phase-advance"]
    assert [record.round_number for record in advance] == [2]
    # The advance names the phase that is still pending, not the primary round
    # that just finished.
    advance_body = next(
        comment["body"]
        for comment in runner.issue_comments
        if "Plan review phase advance to round 2." in comment["body"]
    )
    assert "Outstanding phase after this advance: `secondary-audit`" in advance_body
    assert "`primary`" not in advance_body
    assert len([record for record in records if record.role == "coder"]) == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "claude") == 1
    # Each reviewer ran exactly once, in its own round.
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1
    assert [
        record.round_number
        for record in records
        if record.role == "reviewer" and record.agent == "Gemini"
    ] == [2]


def test_staged_planning_withdrawn_requirement_blocks_carried_approvals(
    tmp_path, monkeypatch
):
    """`human-requirements-and-decomposition`, `carried-approval-requires-current-ack`.

    The primary's carried exact-key approval and the panel's fresh approval are
    both bound to the earlier surfaced requirement digest, so withdrawing a
    signed requirement at the live approval boundary must stop the run instead
    of letting those acknowledgements satisfy the gate for a requirement set
    that no longer exists.
    """
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )
    dispositions = [
        {
            "requirement_id": requirement_1.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        },
        {
            "requirement_id": requirement_2.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the audit trail.",
        },
    ]
    plan_output = structured_v1_plan_state().replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        f"- {requirement_1.requirement_id}: the plan preserves the public API.\n"
        f"- {requirement_2.requirement_id}: the plan preserves the audit trail.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
        gemini_outputs=[
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        # The withdrawal lands only after the panel round has settled, so the
        # primary already holds a carried approval for the earlier digest.
        panel_ran = any(cmd[0] == "gemini" for cmd, _cwd in runner.commands)
        requirements = (
            (requirement_1,) if panel_ran else (requirement_1, requirement_2)
        )
        return replace(context, human_requirements=requirements)

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"withdrawn hr-",
    ):
        run_issue_loop(
            runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
        )

    # Both reviewers approved, yet the plan is not carried into implementation.
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_staged_planning_persists_a_repaired_primary_acknowledgement(tmp_path, monkeypatch):
    """`carried-approval-requires-current-ack`, `reviewer-only-phase-advance`.

    A repaired acknowledgement must reach the durable record before the phase
    advance. The record posted with the original text stores no surfaced
    requirement IDs, so without the amendment the next round would reject the
    primary's approval as unacknowledged and invoke the primary again instead
    of opening the independent panel.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[unacknowledged],
        gemini_outputs=[panel],
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)

    with patch(
        "coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired
    ):
        assert run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path),
            plan_first=True,
        ) == 0

    records = _plan_round_records(runner)
    codex_records = [
        record
        for record in records
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    # The amended record supersedes the one written before the repair, and it
    # is the one that carries the acknowledgement for the surfaced set.
    assert codex_records[-1].round_number == 1
    assert codex_records[-1].state == "approved"
    assert codex_records[-1].surfaced_reviewer_requirement_ids == (
        requirement.requirement_id,
    )
    prelaunch = [record for record in records if record.phase == "scheduler-prelaunch"]
    assert [record.scheduler_phase for record in prelaunch] == [
        "primary",
        "secondary-audit",
    ]
    # The panel opens instead of re-running the primary.
    assert prelaunch[1].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[1].scheduler_approved_reviewers
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1


def test_staged_planning_resume_reads_the_repaired_primary_acknowledgement(
    tmp_path, monkeypatch
):
    """`resume-no-duplicate-calls`: the repaired approval survives a restart.

    The amendment is durable, so a process that stops after the phase-advance
    record and restarts still carries the primary's acknowledged approval and
    invokes only the secondary panel.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[unacknowledged],
        gemini_outputs=[panel],
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
    config = _staged_plan_config(tmp_path)
    real_post = architecture_contract.post_issue_comment

    def interrupt_after_the_advance(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if kwargs["body"].startswith("Plan review phase advance to round 2."):
            raise KeyboardInterrupt
        return result

    with patch(
        "coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired
    ):
        with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
            side_effect=interrupt_after_the_advance,
        ):
            with pytest.raises(KeyboardInterrupt):
                run_issue_loop(
                    runner, issue_number=56, config=config, plan_first=True
                )

        calls_before = [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"claude", "codex", "gemini"}
        ]
        assert calls_before.count("codex") == 1

        assert run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True
        ) == 0

    records = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in records
        if record.phase == "scheduler-prelaunch"
    }
    assert prelaunch[2].scheduler_phase == "secondary-audit"
    assert prelaunch[2].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[2].scheduler_approved_reviewers
    # The restart adds the panel reviewer and nothing else.
    after = [
        cmd[0]
        for cmd, _cwd in runner.commands
        if cmd[0] in {"claude", "codex", "gemini"}
    ]
    assert after == [*calls_before, "gemini"]


def _named_repaired_acknowledgement_run_1363(tmp_path, monkeypatch, *, interrupt_repost):
    """A named Antigravity primary approves on fallback Model B without the ack."""
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    requirement, plan_output, _unack, _repaired, _panel = _repaired_acknowledgement_fixtures()
    dispositions = [{
        "requirement_id": requirement.requirement_id,
        "disposition": "addressed",
        "evidence": "The plan preserves the public API.",
    }]
    flash = SeatAgent(
        ReviewerSeat("flash", "antigravity", ("Model A", "Model B")), tmp_path / "flash",
    )
    opus = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model C",)), tmp_path / "opus")
    for seat in (flash, opus):
        seat.workdir.mkdir()

    def review(seat_signature, *, resolved):
        return structured_plan_review(
            state="approved", reviewer=seat_signature,
            human_requirements_resolved=resolved,
            human_requirement_dispositions=dispositions,
        )

    repaired = review("flash (Google Antigravity: Model B)", resolved=True)
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        antigravity_outputs=[
            ("quota exceeded", 1),
            review("flash (Google Antigravity: Model B)", resolved=False),
            review("opus (Google Antigravity: Model C)", resolved=True),
        ],
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
    config = make_config(
        tmp_path, reviewer=(flash, opus), reviewer_seats=(flash, opus),
        plan_review_policy="primary-then-panel", primary_plan_reviewer=flash,
        review_parallel=True, agent_max_retries=0, max_rounds=6,
    )
    real_post = architecture_contract.post_issue_comment
    state = {"interrupted": False}

    def interrupt_before_the_repost(*args, **kwargs):
        body = kwargs["body"]
        if (
            interrupt_repost and not state["interrupted"]
            and "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in body
            and "\n-- flash (" in body
        ):
            state["interrupted"] = True
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired):
        with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=interrupt_before_the_repost,
        ):
            if interrupt_repost:
                with pytest.raises(KeyboardInterrupt):
                    run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
                assert state["interrupted"]
                restored = [r for r in _plan_round_records(runner) if r.agent == "flash"]
                assert [(r.model_used, r.surfaced_reviewer_requirement_ids) for r in restored] == [
                    ("Model B", ()),
                ]
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    return runner


@pytest.mark.parametrize("interrupt_repost", [False, True])
def test_named_seat_repaired_plan_approval_keeps_the_fallback_model(
    tmp_path, monkeypatch, interrupt_repost
):
    """Row repaired-approval-model-preserved (#1363), live and on resume.

    The repaired repost re-renders a completed review, so it keeps the model
    that produced it rather than the chain head or no model at all.
    """
    runner = _named_repaired_acknowledgement_run_1363(
        tmp_path, monkeypatch, interrupt_repost=interrupt_repost,
    )

    flash_records = [r for r in _plan_round_records(runner) if r.agent == "flash"]
    assert flash_records[-1].round_number == 1
    assert flash_records[-1].surfaced_reviewer_requirement_ids != ()
    assert {r.model_used for r in flash_records} == {"Model B"}
    flash_bodies = [
        c["body"] for c in runner.issue_comments
        if "<!-- AGENT_PLAN_STATE: approved -->" in c["body"] and "\n-- flash (" in c["body"]
    ]
    assert len(flash_bodies) == 2
    reposted = flash_bodies[-1]
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in reposted
    assert reposted.rstrip().endswith("-- flash (Google Antigravity: Model B)")
    assert "Model A" not in reposted.rstrip().splitlines()[-1]
    # The repair never re-invokes the reviewer: one failed Model A attempt,
    # one Model B review, and the panel's Model C review.
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands if cmd[0] == "agy"]
    assert models == ["Model A", "Model B", "Model C"]


def test_staged_planning_round_budget_diagnostic_is_distinct(tmp_path):
    """`round-budget-diagnostic`: exhaustion during a pending phase advance."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path, max_rounds=1),
            plan_first=True,
        )

    message = str(excinfo.value)
    assert "reviewer-only plan phase advance was still pending" in message
    # The diagnostic names the outstanding phase, not the primary round that ran.
    assert "Outstanding phase: secondary-audit" in message
    assert "Gemini" in message
    assert "--max-rounds" in message
    assert "still reported blocking plan issues" not in message


def test_staged_planning_stops_on_premature_secondary_plan_review(tmp_path):
    """`pre-panel-safety-diagnostic`: no reviewer and no planner turn."""
    comments, _canonical = _staged_plan_history(
        tmp_path,
        reviewer_records=[
            (
                "Gemini",
                "blocking",
                (
                    UnresolvedReviewItem(
                        item_id="item-1",
                        reviewer="Gemini",
                        source_round=1,
                        text="Secondary-owned plan finding.",
                        status="blocking",
                    ),
                ),
            )
        ],
    )
    runner = _FakeRunner(issue_comments=comments)

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(
            runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
        )

    assert "no qualified panel opening exists" in str(excinfo.value)
    assert any(
        "Plan review scheduling diagnostic" in comment["body"]
        for comment in runner.issue_comments
    )
    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )


def test_plan_review_force_full_recovers_the_premature_secondary_review(tmp_path):
    """`operator-force-full-override`: the override authorizes the board."""
    comments, _canonical = _staged_plan_history(
        tmp_path,
        # The interrupted round is reconciled, so a resume rehydrates every
        # current-round item, including the superseded secondary's claim.
        reconciliation_record=True,
        reviewer_records=[
            (
                "Gemini",
                "blocking",
                (
                    UnresolvedReviewItem(
                        item_id="item-1",
                        reviewer="Gemini",
                        source_round=1,
                        text="Secondary-owned plan finding.",
                        status="blocking",
                    ),
                ),
            )
        ],
    )
    runner = _FakeRunner(
        issue_comments=comments,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    # The override selects the complete board rather than stopping with the
    # planning diagnostic, and the superseded review establishes no obligation,
    # so the freshly invoked board settles the round.
    assert run_issue_loop(
        runner,
        issue_number=56,
        config=_staged_plan_config(tmp_path, plan_review_force_full=True, max_rounds=2),
        plan_first=True,
    ) == 0

    prelaunch = [
        record for record in _plan_round_records(runner) if record.phase == "scheduler-prelaunch"
    ]
    # The last checkpoint is this run's decision; the first is seeded history.
    decision = prelaunch[-1]
    assert decision.round_number == 2
    assert decision.scheduler_phase == "full-board"
    assert set(decision.scheduler_selected_reviewers) == {"Codex", "Gemini"}
    assert decision.scheduler_force_full is True
    assert decision.scheduler_force_full_source == "operator"
    assert "superseded premature secondary plan review" in " ".join(
        decision.scheduler_reasons
    )
    # The superseded review establishes no approval and no ownership: the same
    # secondary is freshly invoked, and its earlier blocking claim never
    # becomes a durable obligation that outlives the override.
    assert any(cmd[0] == "gemini" for cmd, _cwd in runner.commands)
    fresh_reviews = [
        record
        for record in _plan_round_records(runner)
        if record.role == "reviewer" and not record.new_items
    ]
    assert {record.agent for record in fresh_reviews} == {"Codex", "Gemini"}
    assert {record.round_number for record in fresh_reviews} == {2}
    # The superseded review reaches its own author as explicitly
    # non-authoritative context, and no other reviewer.
    gemini_prompt = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]
    ][-1]
    flat = " ".join(gemini_prompt.split())
    assert (
        "Superseded pre-panel plan review context (non-authoritative; context only):"
        in gemini_prompt
    )
    assert "not a finding, not a plan-item disposition, not an approval" in flat
    assert (
        "none of its claims entered the unresolved plan-item ledger "
        "(superseded plan item IDs: item-1)"
    ) in flat
    assert "- Earlier claim: Secondary-owned plan finding." in gemini_prompt
    codex_prompt = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]
    ][-1]
    assert "Superseded pre-panel plan review context" not in codex_prompt
    # The reconciled resume rehydrates the round's items, but the superseded
    # claim establishes no obligation: no planner turn is needed, and the
    # fresh approvals settle the round with no surviving must-fix item.
    assert not any(cmd[0] == "claude" for cmd, _cwd in runner.commands)
    posted = _plan_round_records(runner)[len(comments):]
    assert posted, "the resumed round must post its own durable records"
    for record in posted:
        assert "item-1" not in {item.item_id for item in record.new_items}
        assert "item-1" not in {item.item_id for item in record.prior_items}


def test_staged_planning_stops_when_planning_history_cannot_be_extracted(tmp_path):
    """`undecodable-planning-history-stop`: only class D stops the run, and the
    operator override cannot recover it."""
    config = _staged_plan_config(tmp_path, plan_review_force_full=True)
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="bot",
                created_at="2026-01-01T00:00:30Z",
                body="Plan round record.\n<!-- AGENT_LOOP_META: not-a-valid-payload -->",
            ),
        ),
    )
    runner = _FakeRunner()

    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError) as excinfo:
        plan_first_loop_module._run_plan_first_loop(
            runner,
            issue_number=56,
            config=config,
            memory=None,
            issue_context=issue_context,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    message = str(excinfo.value)
    assert "could not be extracted" in message
    assert "--plan-review-force-full cannot authorize" in message
    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )
    assert any(
        "Plan review scheduling diagnostic (startup)" in comment["body"]
        for comment in runner.issue_comments
    )


def test_staged_planning_panel_blocker_routes_to_owner_plus_primary(tmp_path):
    """`panel-blocker-remediation`: a `same-plan` panel finding keeps its owner.

    Regression for the planning obligation status set: `same-plan` is the
    planning counterpart of `same-pr`, so it must stay an active obligation or
    the scheduler falls through to a final sweep instead of remediation.
    """
    fresh = structured_v1_plan_state()
    base = plan_first_loop_module.AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Name the rollout owner in the plan steps.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan step names the rollout owner.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "plan_steps",
                "value": ["Implement the reviewed scope and name the rollout owner."],
            }
        ],
    }
    patch_text = (
        json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner = _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved", prior_plan_item_dispositions=resolved
            ),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="One plan step omits the rollout owner.",
                same_plan_followups=["Name the rollout owner in the plan steps."],
            ),
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=resolved,
            ),
        ],
        antigravity_outputs=[
            structured_plan_review(state="approved", reviewer="Google Antigravity"),
            structured_plan_review(
                state="approved",
                reviewer="Google Antigravity",
                prior_plan_item_dispositions=resolved,
            ),
        ],
    )
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    assert phases[:2] == ["primary", "secondary-audit"]
    assert "remediation" in phases, phases
    remediation = prelaunch[phases.index("remediation")]
    # Owner-scoped: the `same-plan` finding owner plus the primary, never the
    # complete board, and no automatic full-board latch is raised.
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert [name for name, _why in remediation.scheduler_paused_reviewers] == [
        "Antigravity"
    ]
    assert remediation.scheduler_active_owners == ("Gemini",)
    assert remediation.scheduler_force_full is False
    assert remediation.scheduler_force_full_source is None
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    assert "full-board" not in phases
    # The posted audit must describe the ordinary owner-scoped remediation as
    # what it is, not as the complete-board post-panel fallback, so its decision
    # kind agrees with the phase, board, and force-full fields beside it.
    remediation_body = _plan_audit_body(runner, round_number=remediation.round_number)
    assert (
        "owner-scoped remediation decision (partial board, no automatic latch)"
        in remediation_body
    )
    assert "post-panel fallback (complete board" not in remediation_body
    assert "- Phase: `remediation`" in remediation_body
    assert "- Force-full: False (source: none)" in remediation_body


def _remediation_runner():
    """Primary gate, panel blocker, narrow remediation, then the final sweep."""
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved", prior_plan_item_dispositions=resolved
            ),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="One plan step omits the rollout owner.",
                blocking_plan_issues=["Name the rollout owner in the plan steps."],
            ),
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=resolved,
            ),
        ],
        antigravity_outputs=[
            structured_plan_review(state="approved", reviewer="Google Antigravity"),
            structured_plan_review(
                state="approved",
                reviewer="Google Antigravity",
                prior_plan_item_dispositions=resolved,
            ),
        ],
    )


def test_staged_planning_advance_after_remediation_names_the_final_sweep(tmp_path):
    """`reviewer-only-phase-advance`: the advance names the pending phase.

    After an owner-scoped remediation round the decision in hand still reads
    `remediation`, but the outstanding reviewer-only round is the final
    exact-plan sweep for the secondary that has no approval of the new key.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    assert phases[:3] == ["primary", "secondary-audit", "remediation"]
    assert "final-secondary-sweep" in phases, phases
    advance_bodies = {
        comment["body"].splitlines()[0]: comment["body"]
        for comment in runner.issue_comments
        if "Plan review phase advance to round" in comment["body"]
    }
    # The panel advance names the audit; the post-remediation advance names the
    # sweep, never the remediation round that just ran.
    assert (
        "Outstanding phase after this advance: `secondary-audit`"
        in advance_bodies["Plan review phase advance to round 2."]
    )
    post_remediation = advance_bodies["Plan review phase advance to round 4."]
    assert (
        "Outstanding phase after this advance: `final-secondary-sweep`"
        in post_remediation
    )
    assert "`remediation`" not in post_remediation
    assert "Antigravity" in post_remediation


def test_staged_planning_resumed_remediation_stays_owner_scoped(tmp_path):
    """`panel-blocker-remediation`: resume keeps the narrow classification.

    The classifier decides `narrow` from the authenticated semantic patch and
    the cross-cutting contracts of the state it was bound to, and the ledger
    stays reconstructible because the run has already accounted for the items
    recorded under the earlier plan subject. An interruption between the
    remediation coder turn and its scheduler checkpoint must therefore resume
    into the same owner-scoped board and the same final sweep an uninterrupted
    run produces, with no complete-board re-invocation and no latch.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = architecture_contract.post_issue_comment
    audits = {"count": 0}

    def interrupt_before_remediation_checkpoint(*args, **kwargs):
        # Rounds 1 and 2 post their own scheduler checkpoints first, so the
        # third one is the remediation round's, posted after the round-3
        # planner turn has already been recorded.
        if kwargs["body"].startswith("Plan review scheduling audit."):
            audits["count"] += 1
            if audits["count"] == 3:
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
        side_effect=interrupt_before_remediation_checkpoint,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    interrupted = _plan_round_records(runner)
    # The remediation planner turn is durable; its scheduler checkpoint is not.
    assert any(
        record.round_number == 3 and record.role == "coder" for record in interrupted
    )
    assert not any(
        record.round_number == 3 and record.phase == "scheduler-prelaunch"
        for record in interrupted
    )

    def reviewer_calls():
        return [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"codex", "gemini", "agy"}
        ]

    calls_before = reviewer_calls()

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in resumed
        if record.phase == "scheduler-prelaunch"
    }
    remediation = prelaunch[3]
    # Owner-scoped: the ledger owner plus the primary, and no latch.
    assert remediation.scheduler_phase == "remediation"
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert [name for name, _why in remediation.scheduler_paused_reviewers] == [
        "Antigravity"
    ]
    assert remediation.scheduler_active_owners == ("Gemini",)
    assert remediation.scheduler_force_full is False
    assert remediation.scheduler_force_full_source is None
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    # The round-3 approvals are carried: the sweep invokes only the reviewer
    # that still lacks an exact-plan approval, with no post-panel latch.
    sweep = prelaunch[4]
    assert sweep.scheduler_phase == "final-secondary-sweep"
    assert sweep.scheduler_selected_reviewers == ("Antigravity",)
    assert sweep.scheduler_force_full is False
    assert sweep.scheduler_force_full_source is None
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )
    reviewer_rounds = [
        (record.agent, record.round_number)
        for record in resumed
        if record.role == "reviewer"
    ]
    assert reviewer_rounds.count(("Antigravity", 4)) == 1
    assert ("Codex", 4) not in reviewer_rounds
    assert ("Gemini", 4) not in reviewer_rounds
    # The resumed run re-invokes nothing already settled: it adds exactly the
    # remediation owner plus primary and the single sweep reviewer.
    assert reviewer_calls() == [*calls_before, "codex", "gemini", "agy"]


def test_staged_planning_resume_on_the_phase_advance_seam_keeps_the_sweep_narrow(tmp_path):
    """`reviewer-only-phase-advance`: the cleared ledger survives a restart.

    A reviewer-only advance carries no unresolved item, so its durable record
    persists an empty ledger. A process that stops on exactly that seam and
    restarts must still read the ledger as reconstructible -- the remediated
    item is cleared by recorded history, not merely by the interrupted run's
    memory -- so the final sweep stays narrow and invokes only the reviewer
    that still lacks an exact-plan approval.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = architecture_contract.post_issue_comment

    def interrupt_after_the_sweep_advance(*args, **kwargs):
        # Post the advance first: the seam under test is a restart with that
        # record durable and nothing after it.
        result = real_post(*args, **kwargs)
        if kwargs["body"].startswith("Plan review phase advance to round 4."):
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
        side_effect=interrupt_after_the_sweep_advance,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    interrupted = _plan_round_records(runner)
    advance = [
        record
        for record in interrupted
        if record.phase == "plan-phase-advance" and record.round_number == 4
    ]
    assert len(advance) == 1
    # The seam: the advance record carries no ledger at all, because
    # remediation cleared the only item before it was written.
    assert advance[0].prior_items == ()
    assert not any(
        record.round_number == 4 and record.phase == "scheduler-prelaunch"
        for record in interrupted
    )

    def reviewer_calls():
        return [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"codex", "gemini", "agy"}
        ]

    calls_before = reviewer_calls()

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in resumed
        if record.phase == "scheduler-prelaunch"
    }
    sweep = prelaunch[4]
    assert sweep.scheduler_phase == "final-secondary-sweep"
    assert sweep.scheduler_selected_reviewers == ("Antigravity",)
    assert sweep.scheduler_force_full is False
    assert sweep.scheduler_force_full_source is None
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )
    reviewer_rounds = [
        (record.agent, record.round_number)
        for record in resumed
        if record.role == "reviewer"
    ]
    assert reviewer_rounds.count(("Antigravity", 4)) == 1
    assert ("Codex", 4) not in reviewer_rounds
    assert ("Gemini", 4) not in reviewer_rounds
    assert reviewer_calls() == [*calls_before, "agy"]


def test_staged_planning_round_budget_names_the_outstanding_final_sweep(tmp_path):
    """`round-budget-diagnostic`: exhaustion after remediation names the sweep."""
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=3
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "reviewer-only plan phase advance was still pending" in message
    assert "Outstanding phase: final-secondary-sweep" in message
    assert "Antigravity" in message
    assert "still reported blocking plan issues" not in message


def test_staged_planning_contract_drift_stops_a_default_policy_restart(tmp_path):
    """A persisted staged planning contract is immutable across a restart.

    Restarting with the compatibility default must stop rather than silently
    continue on a different contract.
    """
    comments, _canonical = _staged_plan_history(tmp_path)
    runner = _FakeRunner(issue_comments=comments)

    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3),
            plan_first=True,
        )

    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )

    # A restart that changes only the primary is a recorded plan board change
    # (#1377), not contract drift: exactly one plan board-change link is posted
    # before the new primary's primary-only round.
    drifted = _FakeRunner(issue_comments=comments)
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(
            drifted,
            issue_number=56,
            config=_staged_plan_config(tmp_path, primary_plan_reviewer="gemini"),
            plan_first=True,
        )
    links = [
        comment for comment in drifted.issue_comments
        if "AGENT_PLAN_BOARD_CHANGE" in comment["body"]
    ]
    assert len(links) == 1
    assert [cmd[0] for cmd, _cwd in drifted.commands if cmd[0] in {"claude", "codex", "gemini"}] == [
        "gemini"
    ]


def test_invalid_planning_history_recovers_after_its_fallback_round(tmp_path):
    """`legacy-planning-metadata-fallback`: class B continues and progresses.

    An invalid scheduler record before the current recovery boundary must not
    pin every later round to the strict pre-panel fallback, which would suppress
    each fresh exact-key primary approval until the round budget ran out.
    """
    comments, canonical = _staged_plan_history(tmp_path, scheduler_record=False)
    comments.append(_invalid_plan_scheduler_comment(_plan_subject(canonical)))
    runner = _FakeRunner(
        issue_comments=comments,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=_staged_plan_config(tmp_path, max_rounds=6),
        plan_first=True,
    ) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
        and record.scheduler_metadata_status == "valid"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    # The invalid record degrades exactly one round, then the valid checkpoint
    # it wrote becomes the recovery boundary and the run advances.
    assert phases == ["primary", "secondary-audit"]
    assert "strict pre-panel fallback:" in " ".join(prelaunch[0].scheduler_reasons)
    assert "strict pre-panel fallback:" not in " ".join(prelaunch[1].scheduler_reasons)
    # No planner turn was fabricated and no reviewer was invoked twice.
    assert not any(cmd[0] == "claude" for cmd, _cwd in runner.commands)
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1


# --- Plan sweep trigger and finding provenance (#1505) ------------------------

_SWEEP_HEADING = "PRIMARY SWEEP REQUEST"


def _m1505_patch(base, *, resolved_items, summary):
    payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": summary,
        "prior_plan_item_dispositions": [
            {"item_id": item_id, "disposition": "resolved"} for item_id in resolved_items
        ],
        "base_round_number": base.round_number,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "plan_steps", "value": [f"{summary} Implement it."]}
        ],
    }
    next_base = AuthenticatedPlanState.from_plan(
        assemble_plan_revision(base, payload), round_number=base.round_number + 1
    )
    return (
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        next_base,
    )


def _m1505_run(tmp_path, *, claude, codex, expect=AgentLoopError, **overrides):
    values = {"max_rounds": 3}
    values.update(overrides)
    runner = _FakeRunner(claude_outputs=list(claude), codex_outputs=list(codex))
    config = _staged_plan_config(tmp_path, **values)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(expect):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, [str(call.args[-1]) for call in logged.call_args_list]


def _m1505_primary_records(runner):
    return [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == "Codex"
    ]


def _m1505_three_blocking_rounds():
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 3, base=base)
    return [fresh, *claude], codex


def test_primary_sweep_fires_after_k_counting_blocks(tmp_path):
    """`sweep-fires-at-k`, `provenance-baseline`: live fake-runner plan loop."""
    claude, codex = _m1505_three_blocking_rounds()
    runner, messages = _m1505_run(tmp_path, claude=claude, codex=codex)

    reviews = _m1251_prompts(runner, "codex")
    assert [_SWEEP_HEADING in prompt for prompt in reviews] == [False, False, True]
    assert "- [item-1] (round 1) Close gap 1." in reviews[2]
    assert "- [item-2] (round 2) Close gap 2." in reviews[2]
    assert "one exhaustive pass over the whole current plan" in reviews[2]
    # The orchestrator-owned marker is on the swept primary record only.
    assert [record.plan_sweep for record in _m1505_primary_records(runner)] == [
        False, False, True,
    ]
    assert any("primary sweep requested (streak 2 >= 2)" in m for m in messages)
    # Never the planner, never a step-back turn, never a stop.
    planner = _m1251_prompts(runner, "claude")
    assert not any(_SWEEP_HEADING in prompt for prompt in planner)
    assert not any(_STEP_BACK_MARKER in prompt for prompt in planner)
    assert all(record.step_back_entries == () for record in _plan_round_records(runner))
    # The durable previous-plan baseline: first-review notice, then a diff.
    assert "this is your first plan review of this issue" in reviews[0]
    assert "you last reviewed the plan of round 1" in reviews[1]
    assert "----- BEGIN BASELINE DIFF -----" in reviews[1]
    assert "you last reviewed the plan of round 2" in reviews[2]
    assert any(
        "Codex new mandatory finding provenance: pre_existing=0 revision_introduced=0 "
        "untagged=1 repeats=0; first_review=True; sweep=False" in m
        for m in messages
    )
    assert any("first_review=False; sweep=True" in m for m in messages)


@pytest.mark.parametrize("compact", [False, True])
def test_resumed_run_reaches_the_same_sweep_decision(tmp_path, compact):
    """`resume-parity`, `provenance-baseline`: durable records only, fresh sessions."""
    claude, codex = _m1505_three_blocking_rounds()
    overrides = {"planning_context_mode": "compact"} if compact else {}
    # The first process dies at the planner revision after round 2's review.
    first, _messages = _m1505_run(tmp_path, claude=claude[:2], codex=codex[:2], **overrides)
    assert len(_m1505_primary_records(first)) == 2
    rerun = _FakeRunner(
        issue_comments=list(first.issue_comments),
        claude_outputs=claude[2:],
        codex_outputs=codex[2:],
    )
    config = _staged_plan_config(tmp_path, max_rounds=3, **overrides)
    with pytest.raises(AgentLoopError):
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    # Round 2's recorded review is not re-run; round 3 gets the sweep.
    reviews = _m1251_prompts(rerun, "codex")
    assert len(reviews) == 1
    assert _SWEEP_HEADING in reviews[0]
    assert "- [item-2] (round 2) Close gap 2." in reviews[0]
    assert "you last reviewed the plan of round 2" in reviews[0]
    assert "+++ current plan" in reviews[0]
    if compact:
        prefix, tail = reviews[0].split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
        assert _SWEEP_HEADING not in prefix and _SWEEP_HEADING in tail
        assert "Finding provenance baseline" in tail
    assert _m1505_primary_records(rerun)[-1].plan_sweep is True

    live, _messages = _m1505_run(tmp_path / "live", claude=claude, codex=codex, **overrides)
    live_sweep = _m1251_prompts(live, "codex")[2]
    assert live_sweep.split(_SWEEP_HEADING, 1)[1].split("Finding provenance baseline")[0] == (
        reviews[0].split(_SWEEP_HEADING, 1)[1].split("Finding provenance baseline")[0]
    )


@pytest.mark.parametrize(
    "overrides",
    [{"plan_sweep_rounds": 0}, {"plan_review_policy": "all-reviewers", "primary_plan_reviewer": None}],
)
def test_disabled_or_full_board_planning_never_sweeps(tmp_path, overrides):
    """`disabled-and-policy`: K=0 or the full-board policy keeps today's prompts."""
    claude, codex = _m1505_three_blocking_rounds()
    values = dict(overrides)
    full_board = values.get("plan_review_policy") == "all-reviewers"
    if full_board:
        values["reviewer"] = ("codex",)
    runner, _messages = _m1505_run(tmp_path, claude=claude, codex=codex, **values)
    reviews = _m1251_prompts(runner, "codex")
    assert len(reviews) == 3
    assert not any(_SWEEP_HEADING in prompt for prompt in reviews)
    assert not any(record.plan_sweep for record in _plan_round_records(runner))
    if full_board:
        assert not any("Finding provenance" in prompt for prompt in reviews)


def _m1505_repeat_script(*, mixed):
    fresh, base1 = _m1103_fresh_base()
    round1 = structured_plan_review(
        state="blocking", summary="Gap 1 remains.", blocking_plan_issues=["Close gap 1."],
    )
    patch1, base2 = _m1505_patch(base1, resolved_items=["item-1"], summary="Close gap 1.")
    # Round 2 keeps item-1 and also re-emits its text as a "new" finding.
    round2 = structured_plan_review(
        state="blocking",
        summary="Gap 1 still open.",
        blocking_plan_issues=["Close gap 1.", *(["Close gap 2b."] if mixed else [])],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "blocking"}],
    )
    resolved = ["item-1", "item-2", *(["item-3"] if mixed else [])]
    patch2, base3 = _m1505_patch(base2, resolved_items=resolved, summary="Close gap 1 again.")
    round3 = structured_plan_review(
        state="blocking",
        summary="Gap 3 remains.",
        blocking_plan_issues=["Close gap 3."],
        prior_plan_item_dispositions=[
            {"item_id": item_id, "disposition": "resolved"} for item_id in resolved
        ],
    )
    patch3, _base = _m1505_patch(base3, resolved_items=[f"item-{len(resolved) + 1}"], summary="Close gap 3.")
    return [fresh, patch1, patch2, patch3], [round1, round2, round3]


@pytest.mark.parametrize("mixed", [False, True])
def test_a_re_emitted_carried_finding_is_marked_and_not_counted(tmp_path, mixed):
    """`repeat-reemission`: minted with repeat_of, kept on the ledger, never counted."""
    claude, codex = _m1505_repeat_script(mixed=mixed)
    runner, messages = _m1505_run(tmp_path, claude=claude, codex=codex)

    round_two = _m1505_primary_records(runner)[1]
    duplicate = next(item for item in round_two.new_items if item.text == "Close gap 1.")
    assert duplicate.item_id == "item-2"
    assert duplicate.repeat_of == "item-1"
    assert duplicate.status == "blocking"
    if mixed:
        fresh_item = next(item for item in round_two.new_items if item.text == "Close gap 2b.")
        assert fresh_item.repeat_of is None
    # The duplicate stays an ordinary finding for the planner.
    assert "[item-2]" in _m1251_prompts(runner, "claude")[2]
    reviews = _m1251_prompts(runner, "codex")
    # The repeat-only review ends the streak; the mixed one counts through item-3.
    assert (_SWEEP_HEADING in reviews[2]) is mixed
    assert any(
        f"repeats=1; first_review=False" in m and "Planning round 2: Codex" in m
        for m in messages
    )


def test_a_revision_introduced_only_review_breaks_the_live_streak(tmp_path):
    """`streak-breakers` (live): later-review revision tags do not count."""
    fresh, base1 = _m1103_fresh_base()
    round1 = structured_plan_review(
        state="blocking", summary="Gap 1.", blocking_plan_issues=["Close gap 1."],
    )
    patch1, base2 = _m1505_patch(base1, resolved_items=["item-1"], summary="Close gap 1.")
    round2 = structured_plan_review(
        state="blocking", summary="The revision broke step 2.",
        blocking_plan_issues=[{"finding": "Step 2 regressed.", "provenance": "revision_introduced"}],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
    )
    patch2, base3 = _m1505_patch(base2, resolved_items=["item-2"], summary="Fix step 2.")
    round3 = structured_plan_review(
        state="blocking", summary="Gap 3.",
        blocking_plan_issues=[{"finding": "Close gap 3.", "provenance": "pre_existing"}],
        prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
    )
    patch3, _base = _m1505_patch(base3, resolved_items=["item-3"], summary="Close gap 3.")
    runner, messages = _m1505_run(
        tmp_path, claude=[fresh, patch1, patch2, patch3], codex=[round1, round2, round3],
    )
    assert not any(_SWEEP_HEADING in prompt for prompt in _m1251_prompts(runner, "codex"))
    tagged = [item.provenance for record in _m1505_primary_records(runner) for item in record.new_items]
    assert tagged == [None, "revision_introduced", "pre_existing"]
    assert any("pre_existing=0 revision_introduced=1 untagged=0" in m for m in messages)


def _m1505_codex_argv(runner):
    return [argv for argv, _cwd in runner.argv_commands if argv[:1] == ["codex"]]


def test_sweep_effort_applies_to_the_primary_sweep_turn_only(tmp_path):
    """`sweep-effort-turn`: invocation, signature and session of the sweep turn only."""
    claude, codex = _m1505_three_blocking_rounds()
    runner, messages = _m1505_run(
        tmp_path, claude=claude, codex=codex,
        plan_sweep_reasoning_effort="high", codex_model="gpt-test",
    )
    efforts = [
        next(arg for arg in argv if arg.startswith("model_reasoning_effort="))
        for argv in _m1505_codex_argv(runner)
    ]
    assert efforts == [
        'model_reasoning_effort="medium"', 'model_reasoning_effort="medium"',
        'model_reasoning_effort="high"',
    ]
    # The sweep turn starts a fresh provider session.
    assert "resume" not in _m1505_codex_argv(runner)[2]
    assert any("sweep turn at effort high in a fresh session" in m for m in messages)
    primary_comments = [
        comment["body"] for comment in runner.issue_comments
        if "AGENT_PLAN_STATE" in comment["body"]
        and "-- OpenAI Codex: gpt-test" in comment["body"]
    ]
    assert len(primary_comments) == 3
    assert primary_comments[-1].rstrip().endswith("gpt-test (high)")
    assert all(body.rstrip().endswith("gpt-test (medium)") for body in primary_comments[:-1])
    # The planner's effort is untouched.
    assert not any(
        "--effort" in argv and "high" in argv
        for argv, _cwd in runner.argv_commands if argv[:1] == ["claude"]
    )


def test_unsupported_sweep_effort_for_the_primary_backend_fails_before_any_turn(tmp_path):
    """`sweep-effort-turn`: `max` is a Claude effort, not a Codex one."""
    claude, codex = _m1505_three_blocking_rounds()
    runner, _messages = _m1505_run(
        tmp_path, claude=claude, codex=codex, plan_sweep_reasoning_effort="max",
    )
    assert _m1103_agent_calls(runner) == []


def test_sweep_effort_is_ignored_with_a_log_line_for_a_backend_without_effort(tmp_path):
    """`sweep-effort-turn`: a Gemini primary runs unchanged."""
    fresh, base = _m1103_fresh_base()
    review = structured_plan_review(
        state="blocking", summary="Gap 1.", blocking_plan_issues=["Close gap 1."],
        reviewer="Google Gemini",
    )
    runner = _FakeRunner(claude_outputs=[fresh], gemini_outputs=[review])
    config = _staged_plan_config(
        tmp_path, max_rounds=1, primary_plan_reviewer="gemini",
        plan_sweep_reasoning_effort="high",
    )
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    messages = [str(call.args[-1]) for call in logged.call_args_list]
    assert any(
        "--plan-sweep-reasoning-effort high is ignored: the primary plan reviewer's "
        "gemini backend has no effort control" in m
        for m in messages
    )


@pytest.mark.parametrize("backend", ["claude", "codex"])
def test_sweep_turn_inputs_override_bare_and_seat_efforts(tmp_path, backend):
    """`sweep-effort-turn`: bare agents get a turn config, seats a turn-local copy."""
    from coding_review_agent_loop.agents.registry import agent_signature
    from coding_review_agent_loop.config import resolve_invocation
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    config = make_config(tmp_path, claude_model="claude-test", codex_model="gpt-test")
    agent, turn_config = plan_first_loop_module._plan_sweep_turn_inputs(backend, config, "high")
    assert agent == backend
    assert resolve_invocation(turn_config, provider=backend, role="reviewer").resolved_effort == "high"
    assert resolve_invocation(config, provider=backend, role="reviewer").resolved_effort == "medium"
    assert plan_first_loop_module._plan_sweep_turn_inputs(backend, config, None) == (backend, config)

    seat = SeatAgent(ReviewerSeat("primary-seat", backend, ("model-x",), "low"), tmp_path)
    turn_seat, seat_config = plan_first_loop_module._plan_sweep_turn_inputs(seat, config, "high")
    assert seat_config is config
    assert isinstance(turn_seat, SeatAgent) and turn_seat == seat
    assert turn_seat.effort == "high" and seat.effort == "low"
    assert turn_seat.backend == backend and turn_seat.workdir == tmp_path
    assert agent_signature(turn_seat).endswith("model-x (high))")
    assert resolve_invocation(config, provider=turn_seat, role="reviewer").resolved_effort == "high"


def _m1505_effort_arg(argv):
    """The effort a recorded claude or codex invocation ran at."""
    if argv[0] == "claude":
        return argv[argv.index("--effort") + 1]
    value = next(arg for arg in argv if arg.startswith("model_reasoning_effort="))
    return value.split("=", 1)[1].strip('"')


@pytest.mark.parametrize("primary_kind", ["claude", "codex-seat", "claude-seat"])
def test_sweep_effort_workflow_for_claude_and_seat_primaries(tmp_path, primary_kind):
    """`sweep-effort-turn`: bare Claude and named-seat primaries through the plan loop."""
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    primary_backend = "codex" if primary_kind == "codex-seat" else "claude"
    planner_backend = "claude" if primary_backend == "codex" else "codex"
    planner_signature = "Anthropic Claude" if planner_backend == "claude" else "OpenAI Codex"
    claude_plan, _reviews = _m1505_three_blocking_rounds()
    planner = [text.replace("-- Anthropic Claude", f"-- {planner_signature}") for text in claude_plan]
    overrides: dict = {"coder": planner_backend, "plan_sweep_reasoning_effort": "high"}
    if primary_kind == "claude":
        signature = "Anthropic Claude"
        overrides.update(
            reviewer=("claude", "gemini"), primary_plan_reviewer="claude",
            claude_model="claude-test",
        )
        base_effort = "medium"
    else:
        seat_dir, panel_dir = tmp_path / "primary-seat", tmp_path / "panel-seat"
        seat_dir.mkdir()
        panel_dir.mkdir()
        seat = SeatAgent(
            ReviewerSeat("primary-seat", primary_backend, ("seat-model",), "low"), seat_dir
        )
        panel = SeatAgent(ReviewerSeat("panel-seat", "gemini", ("gemini-model",)), panel_dir)
        backend_name = "OpenAI Codex" if primary_backend == "codex" else "Anthropic Claude"
        signature = f"primary-seat ({backend_name}: seat-model (low))"
        overrides.update(
            reviewer=(seat, panel), reviewer_seats=(seat, panel), primary_plan_reviewer=seat,
        )
        base_effort = "low"
    reviews = [
        structured_plan_review(
            state="blocking", summary=f"Gap {n} remains.",
            blocking_plan_issues=[f"Close gap {n}."],
            prior_plan_item_dispositions=(
                [{"item_id": f"item-{n - 1}", "disposition": "resolved"}] if n > 1 else None
            ),
            reviewer=signature,
        )
        for n in (1, 2, 3)
    ]
    outputs = {
        f"{planner_backend}_outputs": planner,
        f"{primary_backend}_outputs": reviews,
    }
    runner = _FakeRunner(**outputs)
    config = _staged_plan_config(tmp_path, max_rounds=3, **overrides)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    messages = [str(call.args[-1]) for call in logged.call_args_list]

    primary_argv = [argv for argv, _cwd in runner.argv_commands if argv[:1] == [primary_backend]]
    planner_argv = [argv for argv, _cwd in runner.argv_commands if argv[:1] == [planner_backend]]
    assert len(primary_argv) == 3
    assert [_m1505_effort_arg(argv) for argv in primary_argv] == [base_effort, base_effort, "high"]
    # The planner's effort never changes; no gemini panel turn runs.
    planner_efforts = {_m1505_effort_arg(argv) for argv in planner_argv}
    assert planner_efforts == {"medium"}
    assert not any(argv[:1] == ["gemini"] for argv, _cwd in runner.argv_commands)
    # The sweep turn starts a fresh provider session.
    assert "--resume" not in primary_argv[2] and "resume" not in primary_argv[2]
    assert any("sweep turn at effort high in a fresh session" in m for m in messages)
    sweep_prompt = _m1251_prompts(runner, primary_backend)[2]
    assert _SWEEP_HEADING in sweep_prompt
    # Published signatures report the effort that ran.
    primary_bodies = [
        comment["body"].rstrip() for comment in runner.issue_comments
        if "AGENT_PLAN_STATE" in comment["body"]
        and comment["body"].rstrip().splitlines()[-1].startswith(
            "-- primary-seat" if primary_kind != "claude" else "-- Anthropic Claude"
        )
    ]
    if primary_kind == "claude":
        # The planner is Codex here, so every Claude-signed plan comment is a review.
        assert [body.endswith("(high)") for body in primary_bodies] == [False, False, True]
        assert all(body.endswith("claude-test (medium)") for body in primary_bodies[:2])
    else:
        assert [body.endswith("seat-model (high))") for body in primary_bodies] == [
            False, False, True,
        ]
        assert all(body.endswith("seat-model (low))") for body in primary_bodies[:2])
    records = _m1505_primary_records_for(runner, "primary-seat" if primary_kind != "claude" else "Claude")
    assert [record.plan_sweep for record in records] == [False, False, True]


def _m1505_primary_records_for(runner, agent):
    return [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == agent
    ]


@pytest.mark.parametrize("mixed", [False, True])
def test_a_repaired_re_emission_is_marked_and_not_counted(tmp_path, monkeypatch, mixed):
    """`repeat-reemission` (repaired path): the acknowledgement repair returns blocking."""
    requirement, plan_output, _unack, _repaired, _panel = _repaired_acknowledgement_fixtures()
    dispositions = [{
        "requirement_id": requirement.requirement_id,
        "disposition": "addressed",
        "evidence": "The plan preserves the public API.",
    }]
    base1 = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(plan_output), round_number=1
    )
    round1 = structured_plan_review(
        state="blocking", summary="Gap 1 remains.", blocking_plan_issues=["Close gap 1."],
        human_requirement_dispositions=dispositions,
    )
    patch1, _base2 = _m1505_patch(base1, resolved_items=["item-1"], summary="Close gap 1.")
    patch1 = patch1.replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n### Human requirements\n"
        f"- {requirement.requirement_id}: the plan preserves the public API.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    # Round 2 approves without the acknowledgement; its repair blocks and
    # re-emits the carried item-1 text as a "new" finding.
    # Repair may only recover findings grounded in the reviewer's own source
    # text, so the approval lists them as future follow-ups.
    round2 = structured_plan_review(
        state="approved", summary="Gap 1 closed.",
        future_followups=["Close gap 1.", *(["Close gap 2b."] if mixed else [])],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirement_dispositions=dispositions,
    )
    repaired = structured_plan_review(
        state="blocking", summary="Gap 1 closed.",
        blocking_plan_issues=["Close gap 1.", *(["Close gap 2b."] if mixed else [])],
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        human_requirements_resolved=True,
        human_requirement_dispositions=dispositions,
    )
    runner = _FakeRunner(claude_outputs=[plan_output, patch1], codex_outputs=[round1, round2])
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired):
        with patch.object(
            plan_first_loop_module, "log", wraps=plan_first_loop_module.log
        ) as logged:
            with pytest.raises(AgentLoopError):
                run_issue_loop(
                    runner, issue_number=56, config=_staged_plan_config(tmp_path, max_rounds=2),
                    plan_first=True,
                )
    messages = [str(call.args[-1]) for call in logged.call_args_list]
    assert any("repair returned blocking for Codex" in m for m in messages)
    assert any(
        "Codex repaired blocking outcome new mandatory finding provenance: "
        f"pre_existing=0 revision_introduced=0 untagged={int(mixed)} repeats=1" in m
        for m in messages
    )
    from coding_review_agent_loop.history_export import _plan_provenance_counts
    from coding_review_agent_loop.review_step_back import derive_plan_sweep_state

    # Everything below is derived from the posted durable records only.
    records = [
        PostedRoundRecord(index=index, metadata=metadata, body="")
        for index, metadata in enumerate(_plan_round_records(runner))
    ]
    round_two = [
        record.metadata for record in records
        if record.metadata.role == "reviewer" and record.metadata.agent == "Codex"
        and record.metadata.round_number == 2
    ]
    # The repaired blocking outcome supersedes the approval posted before it.
    assert [metadata.state for metadata in round_two] == ["approved", "blocking"]
    repaired_items = [item for item in round_two[-1].new_items if item.status == "blocking"]
    assert len(repaired_items) == (2 if mixed else 1)
    duplicate = next(item for item in repaired_items if item.text == "Close gap 1.")
    assert duplicate.repeat_of == "item-1"
    if mixed:
        fresh_item = next(item for item in repaired_items if item.text == "Close gap 2b.")
        assert fresh_item.repeat_of is None
    # The repeat-only repair ends the streak; the mixed one extends it.
    state = derive_plan_sweep_state(records, primary="Codex")
    assert [review.classification for review in state.reviews][-1] == (
        "new-finding" if mixed else "repeat-only"
    )
    assert state.streak_before(3) == (2 if mixed else 0)
    # The export counts the genuinely new item and never the duplicate.
    counts = _plan_provenance_counts(records)
    assert counts.resolvable
    assert (counts.pre_existing, counts.revision_introduced, counts.untagged) == (0, 0, int(mixed))


# --- Tiered finding-provenance baseline (#1516) -------------------------------


def _m1516_long_plan_outputs():
    """A fresh #1590-sized plan, then a one-word-edit patch of its long steps."""
    from agent_loop_helpers import long_plan_revision_pair

    previous, current = long_plan_revision_pair()
    fresh_payload = json.loads(structured_v1_plan_state().split("\n<!--", 1)[0])
    fresh_payload["plan_steps"] = previous.splitlines()
    fresh = (
        json.dumps(fresh_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch_payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Close gap 1.",
        "prior_plan_item_dispositions": [{"item_id": "item-1", "disposition": "resolved"}],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "plan_steps", "value": current.splitlines()}
        ],
    }
    revision = (
        json.dumps(patch_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    reviews = [
        structured_plan_review(
            state="blocking", summary=f"Gap {n} remains.",
            blocking_plan_issues=[f"Close gap {n}."],
            prior_plan_item_dispositions=(
                [{"item_id": "item-1", "disposition": "resolved"}] if n > 1 else None
            ),
        )
        for n in (1, 2)
    ]
    return [fresh, revision], reviews


def _m1516_baseline_lines(messages):
    return [m for m in messages if "Codex provenance baseline:" in m]


def _m1516_primary_records(runner):
    """The primary's plan reviewer records, hydrating large-plan sidecars."""
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    records = _extract_round_metadata_records(
        [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments],
        flow="plan",
    )
    return [
        record.metadata for record in records
        if record.metadata.role == "reviewer" and record.metadata.agent == "Codex"
    ]


def test_resumed_round_renders_and_records_the_same_baseline_form(tmp_path):
    """`resume-same-form`: prompt, log line and record agree, fresh or resumed."""
    claude, codex = _m1516_long_plan_outputs()
    live, live_messages = _m1505_run(
        tmp_path / "live", claude=claude, codex=codex, max_rounds=2
    )
    live_reviews = _m1251_prompts(live, "codex")
    assert len(live_reviews) == 2
    assert "this is your first plan review of this issue" in live_reviews[0]
    assert "+++ current plan (compact: changed sentences only)" in live_reviews[1]
    assert "the comparison baseline is unavailable" not in live_reviews[1]
    live_records = _m1516_primary_records(live)
    assert [record.provenance_baseline_form for record in live_records] == [
        "first-review", "compact-diff",
    ]
    compact_chars = live_records[1].provenance_baseline_chars
    assert compact_chars > 0
    live_lines = _m1516_baseline_lines(live_messages)
    assert len(live_lines) == 2
    assert "form=first-review chars=0" in live_lines[0]
    assert f"form=compact-diff chars={compact_chars} " in live_lines[1]
    assert "from_round=1" in live_lines[1]

    # The first process dies after the planner revision, before round 2's review.
    first, _messages = _m1505_run(
        tmp_path / "resumed", claude=claude, codex=codex[:1], max_rounds=2
    )
    assert len(_m1516_primary_records(first)) == 1
    rerun = _FakeRunner(issue_comments=list(first.issue_comments), codex_outputs=codex[1:])
    config = _staged_plan_config(tmp_path / "resumed", max_rounds=2)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        with pytest.raises(AgentLoopError):
            run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    rerun_messages = [str(call.args[-1]) for call in logged.call_args_list]
    resumed_reviews = _m1251_prompts(rerun, "codex")
    assert len(resumed_reviews) == 1
    baseline_block = live_reviews[1].split("Finding provenance baseline", 1)[1].split(
        "----- END BASELINE DIFF -----", 1
    )[0]
    assert "REVISED" in baseline_block
    assert baseline_block in resumed_reviews[0]
    resumed_record = _m1516_primary_records(rerun)[-1]
    assert (resumed_record.provenance_baseline_form, resumed_record.provenance_baseline_chars) == (
        "compact-diff", compact_chars,
    )
    resumed_lines = _m1516_baseline_lines(rerun_messages)
    assert resumed_lines == [live_lines[1]]


def _closed_rereview_history(tmp_path, *, opened=True, secondary_state="approved"):
    """Generation-1 closed history with a contradictory last scheduling key."""
    comments, canonical = _staged_plan_history(
        tmp_path, reviewer_records=[("Codex", "approved", ()), ("Gemini", secondary_state, ())],
        reconciliation_record=True,
    )
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    records = _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in comments], flow="plan"
    )
    checkpoint = records[1].metadata
    key = {**checkpoint.plan_candidate_key, "surfaced_requirement_id_digest": "old-requirements"}
    comments[3]["body"] = _attach_round_metadata(
        "Old secondary review", replace(records[3].metadata, plan_candidate_key=key)
    )
    primary = replace(records[2].metadata, plan_candidate_key=key)
    comments[2]["body"] = _attach_round_metadata("Old primary approval", primary)
    if opened:
        opening = replace(
            checkpoint, plan_candidate_key=key, scheduler_phase="secondary-audit",
            scheduler_approved_reviewers=("Codex",),
            scheduler_selected_reviewers=("Gemini",),
            scheduler_paused_reviewers=(("Codex", "carried approval"),),
        )
        comments.insert(3, {"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:02Z",
                            "body": _attach_round_metadata("Qualified opening", opening)})
    # A used contradictory checkpoint precedes the secondary, so it remains
    # historical scheduling evidence rather than an idle post-settlement stray.
    contradictory = replace(checkpoint, plan_candidate_key={**key, "aggregate_plan_identity": "old"})
    comments.insert(-2, {"author": {"login": "bot"}, "createdAt": "2026-01-01T00:00:03Z",
                         "body": _attach_round_metadata("Contradictory key", contradictory)})
    return comments, canonical, checkpoint


@pytest.mark.parametrize("stray_source", [None, "automatic", "operator"])
@pytest.mark.parametrize("ledger_differs", [False, True])
def test_closed_key_rereview_rolls_over_before_publication(tmp_path, stray_source, ledger_differs):
    comments, canonical, checkpoint = _closed_rereview_history(tmp_path)
    if stray_source:
        item = UnresolvedReviewItem(item_id="item-17", reviewer="Codex", source_round=1,
                                    text="Not the settled ledger", status="blocking")
        comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T00:02:00Z",
                         "body": _attach_round_metadata("Idle checkpoint", replace(
                             checkpoint, scheduler_force_full=True,
                             scheduler_force_full_source=stray_source,
                             prior_items=(item,) if ledger_differs else (),
                         ))})
    runner = _FakeRunner(issue_comments=comments,
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    before = len(_plan_round_records(runner))
    assert run_issue_loop(runner, issue_number=56,
                          config=_staged_plan_config(tmp_path, max_rounds=2), plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    assert {record.round_number for record in fresh} == {2}
    assert {record.agent for record in fresh if record.role == "reviewer"} == {"Codex", "Gemini"}
    decision = next(record for record in fresh if record.phase == "scheduler-prelaunch")
    assert decision.scheduler_force_full and decision.scheduler_force_full_source == "automatic"
    assert decision.scheduler_selected_reviewers == ("Codex", "Gemini")
    assert not any(record.role == "coder" for record in fresh)
    assert all(record.subject == _plan_subject(canonical) for record in fresh)
    assert not any(record.prior_items for record in fresh)
    assert not any(command[0] == "claude" for command, _ in runner.commands)


@pytest.mark.parametrize("blocking", [False, True])
@pytest.mark.parametrize("idle_operator", [None, "plain", "first-amendment-use"])
def test_unopened_closed_key_rereview_stays_primary_only_without_latch(tmp_path, blocking, idle_operator):
    comments, canonical, checkpoint = _closed_rereview_history(
        tmp_path, opened=False, secondary_state="blocking" if blocking else "approved"
    )
    runner = _FakeRunner(issue_comments=comments,
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    digest = None
    if idle_operator == "first-amendment-use":
        from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments
        from issue_loop_test_helpers import _m943_post_amendment
        contract = panel_evidence.make_plan_contract(
            ("Codex", "Gemini", "Antigravity"), "primary-then-panel", "Codex"
        )
        for comment in runner.issue_comments:
            match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"])
            metadata = _decode_round_metadata(match.group(1))
            if metadata.scheduler_contract is not None:
                comment["body"] = _attach_round_metadata("Historical scheduling", replace(
                    metadata, scheduler_contract=contract.as_dict(),
                    scheduler_paused_reviewers=metadata.scheduler_paused_reviewers + (("Antigravity", "paused"),),
                ))
        _m943_post_amendment(runner, effective_from_round=1)
        digest = collect_reviewer_board_amendments(
            [SimpleNamespace(body=c["body"], author_login=c["author"]["login"])
             for c in runner.issue_comments], flow="plan", issue_number=56,
        )[0].digest
    if idle_operator:
        runner.issue_comments.append({"author": {"login": "bot"}, "createdAt": "2026-06-01T00:02:00Z",
                         "body": _attach_round_metadata("Idle operator checkpoint", replace(
                             checkpoint, scheduler_force_full=True, scheduler_force_full_source="operator",
                             reviewer_board_amendment_digest=digest))})
    before = len(_plan_round_records(runner))
    config = _staged_plan_config(tmp_path, max_rounds=3)
    if blocking:
        with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    else:
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    checkpoint = next(record for record in fresh if record.phase == "scheduler-prelaunch")
    assert checkpoint.round_number == 2
    assert checkpoint.reviewer_board_amendment_digest == digest
    assert checkpoint.scheduler_selected_reviewers == ("Codex",)
    assert not checkpoint.scheduler_force_full
    assert checkpoint.scheduler_force_full_source is None
    assert all(record.round_number > 1 for record in fresh)
    assert all(record.subject == _plan_subject(canonical) for record in fresh)
    assert not any(record.role == "coder" for record in fresh)
    assert not any(record.agent == "Gemini" for record in fresh if record.round_number == 2)
    if not blocking:
        assert any(record.agent == "Gemini" and record.round_number == 3 for record in fresh)


def test_closed_operator_rollover_respects_round_bound(tmp_path):
    finding = UnresolvedReviewItem(item_id="item-1", reviewer="Gemini", source_round=1,
                                   text="Superseded", status="blocking")
    comments, _ = _staged_plan_history(tmp_path, reconciliation_record=True,
                                      reviewer_records=[("Gemini", "blocking", (finding,))])
    runner = _FakeRunner(issue_comments=comments)
    before = len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="beyond --max-rounds"):
        run_issue_loop(runner, issue_number=56, config=_staged_plan_config(
            tmp_path, max_rounds=1, plan_review_force_full=True), plan_first=True)
    fresh = runner.issue_comments[before:]
    assert not any("Plan review scheduling audit." in comment["body"] for comment in fresh)
    assert not any(command[0] in {"claude", "codex", "gemini"} for command, _ in runner.commands)


@pytest.mark.parametrize("boundary", ["plan-phase-advance", "scheduler-prelaunch"])
@pytest.mark.parametrize("amended", [False, True])
@pytest.mark.parametrize("idle_checkpoint", [False, True])
def test_closed_rereview_restart_uses_next_round_and_activation_boundary(tmp_path, boundary, amended, idle_checkpoint):
    from coding_review_agent_loop.plan_board_state import plan_resume_round
    from coding_review_agent_loop.github import get_issue_context
    from issue_loop_test_helpers import _m943_post_amendment

    comments, canonical, historical_checkpoint = _closed_rereview_history(tmp_path)
    if idle_checkpoint:
        comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T00:02:00Z",
                         "body": _attach_round_metadata("Idle historical checkpoint", historical_checkpoint)})
    runner = _FakeRunner(issue_comments=comments,
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    config = _staged_plan_config(tmp_path, max_rounds=2)
    if amended:
        contract = panel_evidence.make_plan_contract(
            ("Codex", "Gemini", "Antigravity"), "primary-then-panel", "Codex"
        )
        for comment in runner.issue_comments:
            match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"])
            metadata = _decode_round_metadata(match.group(1))
            if metadata.scheduler_contract is not None:
                comment["body"] = _attach_round_metadata("Historical scheduling", replace(
                    metadata, scheduler_contract=contract.as_dict(),
                    scheduler_paused_reviewers=metadata.scheduler_paused_reviewers + (("Antigravity", "paused"),),
                ))
        _m943_post_amendment(runner, effective_from_round=1)

    real_post = plan_first_loop_module.post_issue_comment

    def interrupt(*args, **kwargs):
        result = real_post(*args, **kwargs)
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", kwargs["body"])
        if match:
            metadata = _decode_round_metadata(match.group(1))
            if metadata.phase == boundary and metadata.round_number == 2:
                raise KeyboardInterrupt
        return result

    with patch.object(plan_first_loop_module, "post_issue_comment", side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    records = _plan_round_records(runner)
    advance = next(record for record in records if record.phase == "plan-phase-advance")
    assert advance.round_number == 2
    assert advance.scheduler_contract is None
    assert advance.reviewer_board_amendment_digest is None
    context = get_issue_context(runner, config=config, issue_number=56)
    assert plan_resume_round(context.comments, config) == 2
    before = len(records)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    assert {record.round_number for record in fresh} == {2}
    assert all(record.subject == _plan_subject(canonical) for record in fresh)
    assert not any(record.role == "coder" for record in fresh)
    assert not any(command[0] == "claude" for command, _ in runner.commands)
    checkpoint = next(record for record in _plan_round_records(runner)
                      if record.round_number == 2 and record.phase == "scheduler-prelaunch")
    assert checkpoint.scheduler_force_full
    assert checkpoint.scheduler_force_full_source == "automatic"
    assert bool(checkpoint.reviewer_board_amendment_digest) == amended
    if amended:
        assert checkpoint.scheduler_contract["required_reviewers"] == ["Codex", "Gemini"]


@pytest.mark.parametrize("operator", [False, True])
@pytest.mark.parametrize("blocking", [False, True])
def test_closed_history_preserves_reviewer_backed_post_reconciliation_opening(tmp_path, operator, blocking):
    comments, canonical, _ = _closed_rereview_history(tmp_path)
    # Move the used opening and its secondary after reconciliation, as older
    # tool versions sometimes did; reconciliation still owns the settled ledger.
    opening = comments.pop(3)
    secondary = comments.pop(-2)
    match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", opening["body"])
    metadata = _decode_round_metadata(match.group(1))
    if operator:
        metadata = replace(metadata, scheduler_force_full=True, scheduler_force_full_source="operator")
    opening["body"] = _attach_round_metadata("Used historical opening", metadata)
    match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", secondary["body"])
    review = _decode_round_metadata(match.group(1))
    finding = UnresolvedReviewItem(item_id="item-9", reviewer="Gemini", source_round=1,
                                   text="Qualified finding", status="blocking")
    secondary["body"] = _attach_round_metadata("Qualified secondary", replace(
        review, state="blocking" if blocking else "approved", new_items=(finding,) if blocking else (),
    ))
    comments.extend([opening, secondary])
    resolved = [{"item_id": "item-9", "disposition": "resolved"}] if blocking else []
    runner = _FakeRunner(issue_comments=comments,
                         codex_outputs=[structured_plan_review(state="approved", prior_plan_item_dispositions=resolved)],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini",
                                                               prior_plan_item_dispositions=resolved)])
    before = len(_plan_round_records(runner))
    assert run_issue_loop(runner, issue_number=56, config=_staged_plan_config(tmp_path, max_rounds=2),
                          plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    assert {record.round_number for record in fresh} == {2}
    advance = next(record for record in fresh if record.phase == "plan-phase-advance")
    assert [item.item_id for item in advance.prior_items] == (["item-9"] if blocking else [])
    assert not any(record.role == "coder" for record in fresh)
    assert all(record.subject == _plan_subject(canonical) for record in fresh)


@pytest.mark.parametrize("staged", [False, True])
def test_legacy_closed_plan_with_stale_bindings_rolls_over(tmp_path, staged):
    comments, canonical, _ = _closed_rereview_history(tmp_path, opened=False)
    rewritten = []
    for comment in comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"])
        metadata = _decode_round_metadata(match.group(1))
        if not staged and metadata.phase == "scheduler-prelaunch":
            continue
        if metadata.phase == "reconciliation":
            comment["body"] = _attach_round_metadata("Legacy authoritative summary", replace(metadata, phase="authoritative"))
        rewritten.append(comment)
    runner = _FakeRunner(issue_comments=rewritten,
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    config = (_staged_plan_config(tmp_path, max_rounds=3) if staged else
              make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2))
    before = len(_plan_round_records(runner))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    assert all(record.round_number > 1 for record in fresh)
    assert all(record.subject == _plan_subject(canonical) for record in fresh)
    assert not any(record.role == "coder" for record in fresh)


def test_completed_plan_shortcut_ignores_force_flag(tmp_path):
    runner = _FakeRunner(claude_outputs=[structured_v1_plan_state()],
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    config = _staged_plan_config(tmp_path, max_rounds=2)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    calls = [command for command, _ in runner.commands if command[0] in {"claude", "codex", "gemini"}]
    assert run_issue_loop(runner, issue_number=56, config=replace(config, plan_review_force_full=True),
                          plan_first=True) == 0
    assert calls == [command for command, _ in runner.commands if command[0] in {"claude", "codex", "gemini"}]


def _pending_amendment_rereview(tmp_path):
    from issue_loop_test_helpers import _m943_post_amendment

    comments, canonical, _ = _closed_rereview_history(tmp_path)
    runner = _FakeRunner(issue_comments=comments,
                         codex_outputs=[structured_plan_review(state="approved")],
                         gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    contract = panel_evidence.make_plan_contract(
        ("Codex", "Gemini", "Antigravity"), "primary-then-panel", "Codex"
    )
    for comment in runner.issue_comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"])
        metadata = _decode_round_metadata(match.group(1))
        if metadata.scheduler_contract is not None:
            comment["body"] = _attach_round_metadata("Historical scheduling", replace(
                metadata, scheduler_contract=contract.as_dict(),
                scheduler_paused_reviewers=metadata.scheduler_paused_reviewers + (("Antigravity", "paused"),),
            ))
    _m943_post_amendment(runner, effective_from_round=1)
    return runner, canonical, _staged_plan_config(tmp_path, max_rounds=2)


def test_closed_amendment_rollover_schedules_same_run_without_clearing_contradiction(tmp_path):
    runner, canonical, config = _pending_amendment_rereview(tmp_path)
    before = len(_plan_round_records(runner))
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    records = _plan_round_records(runner)[before:]
    advance, checkpoint = records[:2]
    assert advance.phase == "plan-phase-advance" and advance.round_number == 2
    assert advance.scheduler_contract is None and advance.reviewer_board_amendment_digest is None
    assert checkpoint.phase == "scheduler-prelaunch" and checkpoint.round_number == 2
    assert checkpoint.scheduler_selected_reviewers == ("Codex", "Gemini")
    assert checkpoint.scheduler_force_full_source == "automatic"
    assert checkpoint.reviewer_board_amendment_digest
    assert all(record.subject == _plan_subject(canonical) for record in records)
    assert not any(record.role == "coder" for record in records)


@pytest.mark.parametrize("defect", ["absent", "wrong-subject", "wrong-phase", "skipped", "intervening-review", "intervening-scheduler"])
def test_invalid_amendment_rollover_boundary_refused_in_preflight_and_startup(tmp_path, defect):
    from coding_review_agent_loop.github import get_issue_context
    from coding_review_agent_loop.plan_board_state import (
        resolve_plan_board_state, plan_drift_error, plan_resume_round,
    )

    runner, canonical, config = _pending_amendment_rereview(tmp_path)
    for comment in list(runner.issue_comments):
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", comment["body"])
        if not match:
            continue
        metadata = _decode_round_metadata(match.group(1))
        if metadata.phase == "reconciliation":
            if defect == "absent":
                runner.issue_comments.remove(comment)
            elif defect == "wrong-subject":
                comment["body"] = _attach_round_metadata("Wrong predecessor", replace(metadata, subject="other"))
            elif defect == "wrong-phase":
                comment["body"] = _attach_round_metadata("Not settled", replace(metadata, phase="plan-phase-advance"))
    advance = PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", subject=_plan_subject(canonical),
        round_number=3 if defect == "skipped" else 2, phase="plan-phase-advance",
    )
    if defect.startswith("intervening"):
        runner.issue_comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T00:03:00Z",
                                      "body": _attach_round_metadata("Intervening work", replace(
                                          advance, role="reviewer" if defect.endswith("review") else "summary",
                                          agent="Codex", state="approved",
                                          phase="publication" if defect.endswith("review") else "scheduler-prelaunch"))})
    runner.issue_comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T00:04:00Z",
                                  "body": _attach_round_metadata("Reviewer-only advance", advance)})
    context = get_issue_context(runner, config=config, issue_number=56)
    before = len(runner.issue_comments)
    with pytest.raises(AgentLoopError, match="Replace the record with one effective from round"):
        resolve_plan_board_state(
            runner, config=config, issue_number=56, comments=context.comments,
            admitted=(context.comments, None), drift_error=plan_drift_error(56),
            resume_round=lambda effective, _lineage: plan_resume_round(context.comments, effective),
        )
    with pytest.raises(AgentLoopError, match="Replace the record with one effective from round"):
        plan_first_loop_module._run_plan_first_loop(
            runner, issue_number=56, config=config, memory=None, issue_context=context,
            usage_context=validated_agent_module._new_usage_context(config),
        )
    assert len(runner.issue_comments) == before
    assert not any(command[0] in {"claude", "codex", "gemini"} for command, _ in runner.commands)


@pytest.mark.parametrize("gate", ["growth", "narrow-staged"])
def test_resumed_primary_approval_skips_panel_for_required_revision(tmp_path, gate):
    from issue_loop_test_helpers import _m1268_staged_raw

    initial = _FakeRunner(
        claude_outputs=[_m1268_staged_raw() if gate == "narrow-staged" else structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _staged_plan_config(tmp_path, max_rounds=1, plan_execution_mode="decompose-only")
    with pytest.raises(AgentLoopError, match="phase advance was still pending"):
        run_issue_loop(initial, issue_number=56, config=config, plan_first=True)
    resumed = _FakeRunner(issue_comments=list(initial.issue_comments))
    before = len(_plan_round_records(resumed))
    config = replace(config, max_rounds=3, plan_execution_mode="plan-only",
                     plan_narrow_staged=gate == "narrow-staged",
                     **({"plan_growth_max_chars": 1000} if gate == "growth" else {}))
    prompts = []

    def revision_boundary(*args, **kwargs):
        assert kwargs["agent"] == config.coder
        prompts.append(kwargs["prompt"])
        raise KeyboardInterrupt

    with patch.object(plan_first_loop_module, "_run_validated_agent", side_effect=revision_boundary):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)
    assert len(prompts) == 1
    assert ("Orchestrator plan-growth notice" if gate == "growth" else "Operator directive (orchestrator") in prompts[0]
    fresh = _plan_round_records(resumed)[before:]
    assert not any(record.role == "reviewer" or record.phase == "scheduler-prelaunch" for record in fresh)
    assert not any(record.phase == "plan-phase-advance" for record in fresh)


@pytest.mark.parametrize("source", [None, "automatic", "operator"])
@pytest.mark.parametrize("ledger_differs", [False, True])
@pytest.mark.parametrize("restart", [None, "plan-phase-advance", "scheduler-prelaunch"])
def test_idle_only_contradiction_requires_fresh_closed_round_review(
    tmp_path, source, ledger_differs, restart
):
    """Idle identity evidence invalidates exact approvals without gaining authority."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path, max_rounds=3)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    records = _plan_round_records(runner)
    checkpoint = next(r for r in reversed(records) if r.phase == "scheduler-prelaunch")
    canonical = next(r.canonical_plan for r in records if r.role == "coder")
    closed_round = checkpoint.round_number
    assert closed_round == 2
    # Serial staged approvals need not publish a reconciliation themselves.
    runner.issue_comments.append({"author": {"login": "bot"},
        "createdAt": f"2026-05-23T00:00:{len(runner.issue_comments):02d}Z",
        "body": _attach_round_metadata("Settled planning round", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=closed_round,
            subject=_plan_subject(canonical), phase="reconciliation"))})
    item = UnresolvedReviewItem(item_id="item-17", reviewer="Codex", source_round=closed_round,
                                text="Idle ledger has no authority", status="blocking")
    idle = replace(
        checkpoint,
        plan_candidate_key={**checkpoint.plan_candidate_key, "aggregate_plan_identity": "old"},
        scheduler_force_full=source is not None, scheduler_force_full_source=source,
        prior_items=(item,) if ledger_differs else (),
    )
    for metadata in (idle, replace(idle, plan_candidate_key=checkpoint.plan_candidate_key)):
        # Even a later idle matching key must not erase the negative evidence.
        runner.issue_comments.append({"author": {"login": "bot"},
            "createdAt": f"2026-05-23T00:00:{len(runner.issue_comments):02d}Z",
            "body": _attach_round_metadata("Idle historical checkpoint", metadata)})
    before = len(_plan_round_records(runner))
    calls_before = len(runner.commands)
    runner.codex_outputs.append(structured_plan_review(state="approved"))
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    real_post = plan_first_loop_module.post_issue_comment

    def interrupt(*args, **kwargs):
        result = real_post(*args, **kwargs)
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", kwargs["body"])
        if match:
            metadata = _decode_round_metadata(match.group(1))
            if metadata.round_number == closed_round + 1 and metadata.phase == restart:
                raise KeyboardInterrupt
        return result

    if restart:
        with patch.object(plan_first_loop_module, "post_issue_comment", side_effect=interrupt):
            with pytest.raises(KeyboardInterrupt):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    assert {r.round_number for r in fresh} == {closed_round + 1}
    assert {r.agent for r in fresh if r.role == "reviewer"} == {"Codex", "Gemini"}
    decision = next(r for r in fresh if r.phase == "scheduler-prelaunch")
    assert decision.scheduler_selected_reviewers == ("Codex", "Gemini")
    assert decision.scheduler_force_full_source == "automatic"
    assert not any(r.prior_items or r.role == "coder" for r in fresh)
    assert all(r.subject == _plan_subject(canonical) for r in fresh)
    assert sorted(command[0] for command, _ in runner.commands[calls_before:]
                  if command[0] in {"claude", "codex", "gemini"}) == ["codex", "gemini"]
    # The real current-key N+1 prelaunch clears degradation and permits completion.
    calls_after = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert not any(command[0] in {"claude", "codex", "gemini"}
                   for command, _ in runner.commands[calls_after:])


@pytest.mark.parametrize("opened,boundary", [
    (False, "scheduler-prelaunch"), (False, "first-review"),
    (True, "scheduler-prelaunch"), (True, "first-review"), (True, "all-reviews"),
])
def test_recovery_prelaunch_retires_only_older_round_approvals(tmp_path, opened, boundary):
    """Recovery retires invalid old approvals but keeps fresh same-round reuse."""
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    if opened:
        comments, canonical, checkpoint = _closed_rereview_history(tmp_path)
    else:
        comments, canonical = _staged_plan_history(
            tmp_path, reviewer_records=[("Codex", "approved", ())], reconciliation_record=True
        )
        records = plan_first_loop_module._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in comments], flow="plan"
        )
        checkpoint = records[1].metadata
        comments[2]["body"] = _attach_round_metadata("Exact primary approval", replace(
            records[2].metadata, plan_candidate_key=checkpoint.plan_candidate_key))
        comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T00:02:00Z",
            "body": _attach_round_metadata("Idle contradictory checkpoint", replace(
                checkpoint, plan_candidate_key={**checkpoint.plan_candidate_key,
                                                "aggregate_plan_identity": "old"}))})
    runner = _FakeRunner(issue_comments=comments,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")])
    config = _staged_plan_config(tmp_path, max_rounds=3, review_parallel=True)
    before = len(_plan_round_records(runner))
    real_post = plan_first_loop_module.post_issue_comment
    published = 0

    def interrupt(*args, **kwargs):
        nonlocal published
        result = real_post(*args, **kwargs)
        match = re.search(r"<!-- AGENT_LOOP_META: (\S+) -->", kwargs["body"])
        if match:
            metadata = _decode_round_metadata(match.group(1))
            if metadata.round_number == 2:
                published += metadata.role == "reviewer"
                if (metadata.phase == boundary
                    or (boundary == "first-review" and published == 1)
                    or (boundary == "all-reviews" and published == 2)):
                    raise KeyboardInterrupt
        return result

    with patch.object(plan_first_loop_module, "post_issue_comment", side_effect=interrupt):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    fresh = _plan_round_records(runner)[before:]
    decisions = [r for r in fresh if r.phase == "scheduler-prelaunch"]
    assert decisions[0].scheduler_force_full is opened
    assert decisions[0].scheduler_selected_reviewers == (("Codex", "Gemini") if opened else ("Codex",))
    reviews = [r for r in fresh if r.role == "reviewer"]
    assert sorted((r.agent, r.round_number) for r in reviews) == [
        ("Codex", 2), ("Gemini", 3 if not opened and boundary == "scheduler-prelaunch" else 2)
    ]
    if not opened:
        assert reviews[0].agent == "Codex"
    assert all(r.round_number > 1 and r.subject == _plan_subject(canonical) for r in fresh)
    assert not any(r.role == "coder" for r in fresh)
    assert sorted(command[0] for command, _ in runner.commands
                  if command[0] in {"claude", "codex", "gemini"}) == ["codex", "gemini"]
    comments = [SimpleNamespace(body=c["body"]) for c in runner.issue_comments]
    plan_text, plan_round = pr_loop_support._resume_plan_round(
        comments, configured_reviewers=panel_evidence.reviewers(config))
    panel_evidence._require_complete_canonical_plan_approval(
        comments, config=config, plan_text=plan_text, plan_round=plan_round,
        human_requirements=(), error_message="recovery approvals were lost")
    classification = classify_plan_approval(comments, config=config, issue_number=56)
    assert classification.approved, classification
    calls_before = len(runner.commands)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert not any(command[0] in {"claude", "codex", "gemini"}
                   for command, _ in runner.commands[calls_before:])


def _approved_closed_history(tmp_path, *, future=False):
    """Real generation-1 approval across primary/panel rounds, then settlement."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini",
                                              future_followups=["Retain predecessor follow-up."] if future else [])],
    )
    config = _staged_plan_config(tmp_path, max_rounds=2)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    records = _plan_round_records(runner)
    checkpoint = next(r for r in reversed(records) if r.phase == "scheduler-prelaunch")
    new_items = tuple(item for r in records if r.round_number == 2 for item in r.new_items)
    runner.issue_comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T01:00:00Z",
                                 "body": _attach_round_metadata("Settled", PostedRoundMetadata(
                                     flow="plan", role="summary", agent="Orchestrator",
                                     round_number=checkpoint.round_number, subject=checkpoint.subject,
                                     phase="reconciliation", new_items=new_items,
                                     plan_board_root=checkpoint.plan_board_root,
                                     plan_board_identity=checkpoint.plan_board_identity,
                                     reviewer_board_change_digest=checkpoint.reviewer_board_change_digest))})
    return runner, config, checkpoint


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("mode", ["plan-only", "auto"])
def test_approved_reconciled_resume_reaches_execution_without_scheduling(tmp_path, neutral, mode):
    runner, config, checkpoint = _approved_closed_history(tmp_path, future=True)
    if neutral:
        runner.issue_comments.append({"author": {"login": "bot"}, "createdAt": "2026-01-01T01:00:01Z",
                                     "body": _attach_round_metadata("Unused advance", PostedRoundMetadata(
                                         flow="plan", role="summary", agent="Orchestrator",
                                         round_number=3, subject=checkpoint.subject,
                                         phase="plan-phase-advance"))})
    config = replace(config, plan_execution_mode=mode)
    before = len(_plan_round_records(runner))
    calls = len(runner.commands)
    for _ in range(2):
        with patch.object(plan_first_loop_module, "select_plan_reviewers", side_effect=AssertionError("scheduled")), \
             patch.object(plan_first_loop_module, "_implement_approved_issue", return_value=0) as implement:
            assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
            assert implement.call_count == (mode == "auto")
    assert len(_plan_round_records(runner)) == before
    assert not any(command[0] in {"claude", "codex", "gemini"}
                   for command, _ in runner.commands[calls:])
    summaries = [c["body"] for c in runner.issue_comments if "Approved plan future follow-ups:" in c["body"]]
    assert summaries and all("Retain predecessor follow-up." in body for body in summaries)


@pytest.mark.parametrize("exclusion", ["seat", "restoration", "parent"])
@pytest.mark.parametrize("round_limit", [2, 3])
@pytest.mark.parametrize("reconciled", [False, True])
@pytest.mark.parametrize("parallel", [False, True])
def test_approved_shortcut_live_binding_disagreement_refuses_before_advance(tmp_path, round_limit, reconciled, parallel, exclusion):
    runner, config, checkpoint = _approved_closed_history(tmp_path)
    if not reconciled:
        runner.issue_comments.pop()
        if parallel:
            from coding_review_agent_loop.round_state import _extract_round_metadata_records
            records = _extract_round_metadata_records([SimpleNamespace(body=c["body"]) for c in runner.issue_comments], flow="plan")
            secondary = next(r for r in records if r.metadata.role == "reviewer" and r.metadata.agent == "Gemini")
            runner.issue_comments[secondary.index]["body"] = _attach_round_metadata(secondary.body, replace(
                secondary.metadata, phase="publication", canonical_reviewer_response=structured_plan_review(
                    state="approved", reviewer="Google Gemini")))
    records = plan_first_loop_module._extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.issue_comments], flow="plan")
    primary = next(r for r in records if r.metadata.role == "reviewer" and r.metadata.agent == "Codex")
    if exclusion == "parent":
        primary = replace(primary, metadata=replace(primary.metadata, parent_requirement_ids=("old-parent",)))
    runner.issue_comments[primary.index]["body"] = _attach_round_metadata(primary.body, primary.metadata)
    recorded_binding = primary.metadata.seat_binding
    config = replace(config, review_parallel=parallel)
    before = len(runner.issue_comments)
    calls = len(runner.commands)
    messages = []
    real_lineage = plan_first_loop_module.resolve_plan_lineage
    real_classify = plan_first_loop_module.classify_plan_approval

    def classify_before_parent_refresh(*args, **kwargs):
        # Simulate startup's older parent context, then exercise the real
        # live carried-approval predicate against the current empty parent set.
        kwargs["parent_requirement_ids"] = None
        return real_classify(*args, **kwargs)

    def restored_lineage(*args, **kwargs):
        lineage = real_lineage(*args, **kwargs)
        fields = {name: getattr(lineage, name) for name in dir(lineage) if not name.startswith("_")}
        fields.update(restoration_rounds={"Codex": 2}, fresh_from_rounds={})
        return SimpleNamespace(**fields)

    for _ in range(2):
        with ExitStack() as stack:
            if exclusion == "seat":
                stack.enter_context(patch("coding_review_agent_loop.reviewer_seats.reviewer_seat_binding", return_value={
                    "version": 1, "seats": [{"id": "Codex", "backend": "codex",
                    "model_chain": ["obsolete-seat"], "effort": None}]}))
            elif exclusion == "restoration":
                stack.enter_context(patch.object(plan_first_loop_module, "resolve_plan_lineage", side_effect=restored_lineage))
            else:
                stack.enter_context(patch.object(plan_first_loop_module, "_infer_staged_parent_issue", return_value=99))
                stack.enter_context(patch.object(plan_first_loop_module, "classify_plan_approval", side_effect=classify_before_parent_refresh))
            with pytest.raises(AgentLoopError) as exc:
                run_issue_loop(runner, issue_number=56, config=replace(config, max_rounds=round_limit), plan_first=True)
        messages.append(str(exc.value))
    assert messages[0] == messages[1]
    assert f"Codex: status=approved, approval round=1, seat binding={recorded_binding!r}" in messages[0]
    assert "resumed round 2" in messages[0] and checkpoint.subject.rsplit("-", 1)[-1][:16] in messages[0]
    assert "restore qualifying exact-plan" in messages[0]
    if exclusion == "seat":
        assert "obsolete-seat" in messages[0]
    elif exclusion == "restoration":
        assert "restoration round=2" in messages[0]
    else:
        assert "parent requirement binding=('old-parent',) (required ())" in messages[0]
    assert len(runner.issue_comments) == before
    assert not any(command[0] in {"claude", "codex", "gemini"}
                   for command, _ in runner.commands[calls:])


@pytest.mark.parametrize("tail", ["blocking-review", "changed-candidate", "contradictory-checkpoint", "advance-with-finding"])
def test_approved_resume_preserves_substantive_tail(tmp_path, tail):
    from coding_review_agent_loop.round_state import _resume_plan_round
    from coding_review_agent_loop.approved_plan_resume import approved_plan_resume_view
    runner, config, checkpoint = _approved_closed_history(tmp_path)
    meta = PostedRoundMetadata(flow="plan", role="summary", agent="Orchestrator",
                               round_number=3, subject=checkpoint.subject, phase="plan-phase-advance")
    runner.issue_comments.append({"body": _attach_round_metadata("Unused advance", meta)})
    if tail == "advance-with-finding":
        runner.issue_comments.pop()
        later = replace(meta, new_items=(UnresolvedReviewItem(
            item_id="item-99", reviewer="Codex", source_round=3, text="Substantive obligation", status="blocking"),))
    elif tail == "blocking-review":
        later = replace(meta, role="reviewer", agent="Codex", state="blocking", phase="authoritative")
    elif tail == "changed-candidate":
        # A later candidate is never mistaken for a summary-only neutral tail.
        later = PostedRoundMetadata(flow="plan", role="coder", agent="Claude", round_number=3,
                                    subject=_plan_subject("Changed candidate"), canonical_plan="Changed candidate")
    else:
        later = replace(checkpoint, round_number=3, plan_candidate_key={
            **checkpoint.plan_candidate_key, "aggregate_plan_identity": "contradictory",
        })
    runner.issue_comments.append({"body": _attach_round_metadata("Substantive later activity", later)})
    comments = [SimpleNamespace(body=c["body"]) for c in runner.issue_comments]
    resumed = _resume_plan_round(comments, configured_reviewers=("codex", "gemini"))
    view = approved_plan_resume_view(comments, resumed, config=config, human_requirements=())
    assert view == resumed
    assert view[1].round_number == 3
    from coding_review_agent_loop.plan_verification import classify_plan_approval
    classification = classify_plan_approval(
        comments, config=config, issue_number=56, trusted_actor=None, human_requirements=(),
    )
    assert not classification.approved


@pytest.mark.parametrize("neutral", [False, True])
@pytest.mark.parametrize("mode", ["plan-only", "auto"])
def test_approved_resume_with_historical_board_link_keeps_approving_board(tmp_path, neutral, mode):
    runner = _FakeRunner(
        serve_rest_issue_comments=True, authenticated_actor=("agent-loop", 4242),
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    original = _staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=1)
    with pytest.raises(AgentLoopError, match="phase advance was still pending"):
        run_issue_loop(runner, issue_number=56, config=original, plan_first=True)
    reduced = _staged_plan_config(tmp_path, max_rounds=3)
    assert run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True) == 0
    checkpoint = next(r for r in reversed(_plan_round_records(runner)) if r.phase == "scheduler-prelaunch")
    runner.issue_comments.append({"body": _attach_round_metadata("Settled", PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", round_number=checkpoint.round_number,
        subject=checkpoint.subject, phase="reconciliation", plan_board_root=checkpoint.plan_board_root,
        plan_board_identity=checkpoint.plan_board_identity,
        reviewer_board_change_digest=checkpoint.reviewer_board_change_digest)),
        "author": {"login": "agent-loop"}, "_rest_author_id": 4242,
        "createdAt": runner.issue_comments[-1]["createdAt"]})
    if neutral:
        runner.issue_comments.append({"body": _attach_round_metadata("Unused advance", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=checkpoint.round_number + 1,
            subject=checkpoint.subject, phase="plan-phase-advance",
            plan_board_root=checkpoint.plan_board_root, plan_board_identity=checkpoint.plan_board_identity,
            reviewer_board_change_digest=checkpoint.reviewer_board_change_digest)),
            "author": {"login": "agent-loop"}, "_rest_author_id": 4242,
            "createdAt": runner.issue_comments[-1]["createdAt"]})
    before = len(_plan_round_records(runner))
    links = [c for c in runner.issue_comments if "AGENT_PLAN_BOARD_CHANGE" in c["body"]]
    assert len(links) == 1
    logs = []
    with patch.object(plan_first_loop_module, "log", side_effect=lambda _config, message: logs.append(message)), \
         patch.object(plan_first_loop_module, "_implement_approved_issue", return_value=0) as implement, \
         patch.object(plan_first_loop_module, "select_plan_reviewers", side_effect=AssertionError("scheduled")):
        try:
            result = run_issue_loop(runner, issue_number=56, config=replace(
                original, max_rounds=checkpoint.round_number, plan_execution_mode=mode), plan_first=True)
        except AgentLoopError as exc:
            pytest.fail(f"{exc}; startup logs: {logs}")
        assert result == 0
        assert implement.call_count == (mode == "auto")
        if mode == "auto":
            assert implement.call_args.kwargs["config"].reviewer == original.reviewer
    assert len(_plan_round_records(runner)) == before
    assert len([c for c in runner.issue_comments if "AGENT_PLAN_BOARD_CHANGE" in c["body"]]) == 1
    assert any("no new plan board-change link is recorded" in message for message in logs)
    assert any("changed by recorded board-change link" in message for message in logs)


@pytest.mark.parametrize("mode,policy", [("auto", "issue"), ("auto", "ignore"), ("plan-only", "issue")])
def test_approved_neutral_resume_preserves_lifecycle_and_publication_parity(tmp_path, mode, policy):
    from coding_review_agent_loop.protocol import ReviewItemDisposition
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    seed, config, checkpoint = _approved_closed_history(tmp_path, future=True)
    records = _extract_round_metadata_records([SimpleNamespace(body=c["body"]) for c in seed.issue_comments], flow="plan")
    primary = next(r for r in records if r.metadata.role == "reviewer" and r.metadata.agent == "Codex")
    secondary = next(r for r in records if r.metadata.role == "reviewer" and r.metadata.agent == "Gemini")
    earlier = tuple(UnresolvedReviewItem(item_id=f"item-{number}", reviewer="Codex", source_round=1,
                                        text=text, status="future", source_status="future")
                    for number, text in [(11, "Retain earlier follow-up."), (12, "Resolved earlier follow-up."),
                                         (13, "Promoted then resolved follow-up.")])
    carried = (*earlier[:2], replace(earlier[2], status="same-plan"))
    dispositions = tuple(ReviewItemDisposition(item_id=f"item-{number}", disposition="resolved", reviewer="Gemini") for number in (12, 13))
    # Preserve a real generation-1 plan/approval key and add durable lifecycle
    # records: early deferral, later promotion, and reviewer clearances.
    for record in records:
        meta = record.metadata
        if record.index == primary.index:
            meta = replace(meta, new_items=earlier)
        elif meta.round_number == 2:
            meta = replace(meta, prior_items=carried)
            if record.index == secondary.index:
                meta = replace(meta, dispositions=dispositions)
        seed.issue_comments[record.index]["body"] = _attach_round_metadata(record.body, meta)
    # Remove the previous publication announcement so both paths must derive
    # and publish their sources afresh. Durable planning records stay intact.
    history = [c for c in seed.issue_comments if "Planning complete for issue" not in c["body"]]
    results = []
    for neutral in (False, True):
        comments = list(history)
        if neutral:
            comments.append({"body": _attach_round_metadata("Unused active-only advance", PostedRoundMetadata(
                flow="plan", role="summary", agent="Orchestrator", round_number=3,
                subject=checkpoint.subject, phase="plan-phase-advance")),
                "author": {"login": "bot"}, "createdAt": "2026-01-01T02:00:00Z"})
        runner = _FakeRunner(issue_comments=comments, issue_urls=["https://github.com/OWNER/REPO/issues/99",
                                                                 "https://github.com/OWNER/REPO/issues/100"])
        selected = replace(config, plan_execution_mode=mode, approved_followups=policy)
        before = len(_plan_round_records(runner))
        for _ in range(2):
            with patch.object(plan_first_loop_module, "_implement_approved_issue", return_value=0):
                assert run_issue_loop(runner, issue_number=56, config=selected, plan_first=True) == 0
        assert len(_plan_round_records(runner)) == before
        assert not any(command[0] in {"claude", "codex", "gemini"} for command, _ in runner.commands)
        assert len(runner.issues) == (2 if mode == "auto" and policy == "issue" else 0)
        output = "\n".join(issue["body"] for issue in runner.issues) if runner.issues else "\n".join(
            c["body"] for c in runner.issue_comments if c["body"].startswith("Planning complete for issue"))
        assert "Retain earlier follow-up." in output and "Retain predecessor follow-up." in output
        assert "Resolved earlier follow-up." not in output and "Promoted then resolved follow-up." not in output
        if runner.issues:
            assert any("item-11" in issue["body"] and "Codex" in issue["body"] for issue in runner.issues)
        results.append((runner.issues, output))
    assert results[0] == results[1]


@pytest.mark.parametrize("round_limit", [2, 3])
def test_unapproved_reconciled_resume_rolls_over_only_with_budget(tmp_path, round_limit):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    runner, config, _ = _approved_closed_history(tmp_path)
    records = _extract_round_metadata_records([SimpleNamespace(body=c["body"]) for c in runner.issue_comments], flow="plan")
    secondary = next(r for r in records if r.metadata.role == "reviewer" and r.metadata.agent == "Gemini")
    runner.issue_comments[secondary.index]["body"] = "Review was never published."
    runner.gemini_outputs.append(structured_plan_review(state="approved", reviewer="Google Gemini"))
    before = len(_plan_round_records(runner))
    selected = replace(config, max_rounds=round_limit)
    if round_limit == 2:
        with pytest.raises(AgentLoopError, match="fresh reviews require round 3, beyond --max-rounds"):
            run_issue_loop(runner, issue_number=56, config=selected, plan_first=True)
        assert len(_plan_round_records(runner)) == before
    else:
        assert run_issue_loop(runner, issue_number=56, config=selected, plan_first=True) == 0
        new = _plan_round_records(runner)[before:]
        assert new[0].phase == "plan-phase-advance" and new[0].round_number == 3
        assert all(r.round_number == 3 for r in new)
        assert [(r.agent, r.round_number) for r in new if r.role == "reviewer"] == [("Gemini", 3)]
        assert not any(r.role == "coder" for r in new)


def test_approved_resume_does_not_activate_pending_post_approval_link(tmp_path):
    from coding_review_agent_loop.board_change import (
        BoardIdentity, PLAN_SURFACE, board_change_link_payload, render_board_change_link,
        plan_identity_from_config,
    )
    runner, config, checkpoint = _approved_closed_history(tmp_path)
    runner.authenticated_actor = ("agent-loop", 4242)
    runner.serve_rest_issue_comments = True
    for comment in runner.issue_comments:
        comment.update(author={"login": "agent-loop"}, _rest_author_id=4242,
                       createdAt="2026-05-23T00:00:00Z")
    root = BoardIdentity.from_payload(checkpoint.plan_board_root)
    wider = plan_identity_from_config(replace(config, reviewer=("codex", "gemini", "antigravity")))
    payload = board_change_link_payload(
        pr_number=56, root=root, predecessor=root.digest, old_board=root, new_board=wider,
        old_seat_binding=root.seat_binding(), binding_init={}, anchor_round=2,
        advance=True, surface=PLAN_SURFACE,
    )
    runner.issue_comments.append({"body": str(render_board_change_link(payload, surface=PLAN_SURFACE)),
                                 "author": {"login": "agent-loop"}, "_rest_author_id": 4242,
                                 "createdAt": "2026-05-23T00:00:00Z"})
    before = len(runner.issue_comments)
    logs = []
    with patch.object(plan_first_loop_module, "log", side_effect=lambda _config, message: logs.append(message)):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert len(runner.issue_comments) == before
    assert any("does not reopen planning; it is not activated" in message for message in logs)


@pytest.mark.parametrize("policy", ["issue", "ignore"])
@pytest.mark.parametrize("mode", ["plan-only", "auto"])
@pytest.mark.parametrize("review_policy", ["primary-then-panel", "all-reviewers"])
def test_approved_publication_resume_preserves_followups(tmp_path, policy, mode, review_policy):
    runner, config, _ = _approved_closed_history(tmp_path, future=True)
    if review_policy == "all-reviewers":
        runner = _FakeRunner(claude_outputs=[structured_v1_plan_state()],
            codex_outputs=[structured_plan_review(state="approved")],
            gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini",
                future_followups=["Retain predecessor follow-up."])])
        config = replace(config, plan_review_policy=review_policy, primary_plan_reviewer=None)
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    records = plan_first_loop_module._extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.issue_comments], flow="plan")
    history = []
    for record in records:
        if record.metadata.phase == "reconciliation":
            continue
        meta = record.metadata
        if meta.role == "reviewer" and (review_policy == "all-reviewers" or meta.round_number == 2):
            meta = replace(meta, phase="publication", new_items=(),
                canonical_reviewer_response=structured_plan_review(state="approved", reviewer="Google Gemini" if meta.agent == "Gemini" else "Codex",
                    future_followups=["Retain predecessor follow-up."] if meta.agent == "Gemini" else []))
        history.append({"body": _attach_round_metadata(record.body, meta),
                        "author": {"login": "bot"}, "createdAt": "2026-01-01T01:00:00Z"})
    runner = _FakeRunner(issue_comments=history, issue_urls=["https://github.com/OWNER/REPO/issues/99"])
    selected = replace(config, plan_execution_mode=mode, approved_followups=policy,
                       plan_review_policy=review_policy, primary_plan_reviewer=None if review_policy == "all-reviewers" else "codex", review_parallel=True)
    before = len(_plan_round_records(runner))
    for _ in range(2):
        with patch.object(plan_first_loop_module, "_implement_approved_issue", return_value=0):
            assert run_issue_loop(runner, issue_number=56, config=selected, plan_first=True) == 0
    assert len(_plan_round_records(runner)) == before
    assert not any(command[0] in {"claude", "codex", "gemini"} for command, _ in runner.commands)
    assert len(runner.issues) == (1 if mode == "auto" and policy == "issue" else 0)
    output = "\n".join(issue["body"] for issue in runner.issues) if runner.issues else "\n".join(
        c["body"] for c in runner.issue_comments if c["body"].startswith("Planning complete for issue"))
    assert "Retain predecessor follow-up." in output
