"""Tests for opt-in parallel plan/PR reviewer execution (#594)."""
import dataclasses
import json
import re
import threading
import time
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.cli import AgentLoopError, build_parser, run_issue_loop, run_pr_loop
from coding_review_agent_loop.errors import AgentInvocationError, QuotaResetExceededError
from coding_review_agent_loop.round_state import (
    PlanValidationDiagnosticPayload,
    encode_plan_validation_diagnostic_body,
)
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    malformed_plan_review_source,
    structured_coder_followup,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)


def _initial_plan() -> str:
    return structured_plan_state(
        state="blocking", summary="Initial plan.", plan_steps=["Make the change."]
    )


class _PlanDiagnosticParallelRunner(FakeRunner):
    """Expose the REST identity seam while retaining the normal issue fixture."""

    def __init__(
        self,
        *,
        diagnostic_body,
        claude_outputs,
        codex_outputs,
        gemini_outputs,
        actor_login="agent",
        actor_id=7,
    ):
        self.actor_login = actor_login
        self.actor_id = actor_id
        super().__init__(
            claude_outputs=claude_outputs,
            codex_outputs=codex_outputs,
            gemini_outputs=gemini_outputs,
            issue_comments=(
                [
                    {
                        "author": {"login": "agent", "id": 7},
                        "createdAt": "2026-09-17T05:30:00Z",
                        "body": diagnostic_body,
                        "id": 700,
                    }
                ]
                if diagnostic_body is not None
                else []
            ),
        )
        self.verified_round_bodies = []

    def _rest_comment(self, raw_comment, index):
        author = raw_comment.get("author") or raw_comment.get("user") or {}
        login = author.get("login") if isinstance(author, dict) else None
        author_id = author.get("id") if isinstance(author, dict) else None
        return {
            "id": raw_comment.get("id") or 1000 + index,
            "created_at": raw_comment.get("createdAt") or raw_comment.get("created_at"),
            "body": raw_comment.get("body"),
            "user": {
                "login": login or "coding-review-agent-loop",
                "id": author_id if isinstance(author_id, int) else 99,
            },
        }

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded,
                cwd_path,
                json.dumps({"login": self.actor_login, "id": self.actor_id}),
                "",
                0,
            )
        if command[:4] == ["gh", "api", "--method", "POST"]:
            endpoint = command[4] if len(command) > 4 else ""
            if endpoint == "repos/OWNER/REPO/issues/56/comments":
                recorded, cwd_path = self._record_command(args, cwd)
                body = json.loads(input_text or "{}")["body"]
                self.verified_round_bodies.append(body)
                comment = {
                    "author": {"login": self.actor_login, "id": self.actor_id},
                    "createdAt": "2026-09-17T05:31:00Z",
                    "body": body,
                    "id": 701 + len(self.verified_round_bodies),
                }
                self.issue_comments.append(comment)
                return CommandResult(
                    recorded,
                    cwd_path,
                    json.dumps(
                        {
                            "id": comment["id"],
                            "created_at": comment["createdAt"],
                            "body": body,
                            "user": {
                                "login": self.actor_login,
                                "id": self.actor_id,
                            },
                        }
                    ),
                    "",
                    0,
                )
        if command[:2] == ["gh", "api"] and len(command) > 2 and command[2].startswith(
            "repos/OWNER/REPO/issues/56/comments?"
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            query = dict(part.split("=", 1) for part in command[2].split("?", 1)[1].split("&"))
            page = int(query["page"])
            raw_comments = [self._rest_comment(comment, index) for index, comment in enumerate(self.issue_comments)]
            start = (page - 1) * int(query["per_page"])
            end = start + int(query["per_page"])
            return CommandResult(recorded, cwd_path, json.dumps(raw_comments[start:end]), "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


@pytest.mark.parametrize("context_mode", ["compact", "full"])
def test_plan_parallel_revision_supersedes_diagnostic_without_leaking_it_to_next_prompt(
    tmp_path, context_mode
):
    payload = PlanValidationDiagnosticPayload(
        repository="OWNER/REPO",
        issue_number=56,
        planning_generation=1,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None,
        expected_producer_login="agent",
        expected_producer_id=7,
        failure_attempt=1,
        candidate_digest="a" * 64,
        category="deterministic",
        diagnostic="missing matrix-level audit operation",
    )
    diagnostic_body = str(encode_plan_validation_diagnostic_body(payload))
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=diagnostic_body,
        claude_outputs=[
            structured_plan_state(summary="Replacement initial plan."),
            structured_plan_revision(summary="Revision after the blocking review."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Codex found one plan issue.",
                blocking_plan_issues=["Add the missing verification step."],
            ),
            structured_plan_review(
                summary="Codex approves the revised plan.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        gemini_outputs=[
            structured_plan_review(
                summary="Gemini approves the initial plan.",
                reviewer="Google Gemini",
            ),
            structured_plan_review(
                summary="Gemini approves the revised plan.",
                reviewer="Google Gemini",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        review_parallel=True,
        planning_context_mode=context_mode,
        max_rounds=2,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompts = [
        command[-1]
        for command, _cwd in runner.commands
        if command[:1] == ["claude"]
    ]
    assert len(planner_prompts) == 2
    assert "missing matrix-level audit operation" in planner_prompts[0]
    assert "missing matrix-level audit operation" not in planner_prompts[1]
    assert len(runner.verified_round_bodies) == 1
    assert all("901" not in body for body in runner.verified_round_bodies)


def test_plan_validation_diagnostic_survives_a_new_invocation_and_is_superseded(
    tmp_path, monkeypatch
):
    invalid_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    invalid_payload.pop("architecture_impact")
    invalid_candidate = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=None,
        claude_outputs=[invalid_candidate],
        codex_outputs=[],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, agent_max_retries=0, max_rounds=1)

    with patch.object(
        orchestrator,
        "_run_structured_repair",
        return_value=(None, None, []),
    ):
        with pytest.raises(AgentInvocationError) as error:
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.plan_validation_exhaustion is not None
    assert len(runner.verified_round_bodies) == 1

    runner.claude_outputs = [structured_plan_state(summary="Recovered plan.")]
    runner.codex_outputs = [structured_plan_review(summary="The recovered plan is approved.")]
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompts = [
        command[-1]
        for command, _cwd in runner.commands
        if command[:1] == ["claude"]
    ]
    assert len(planner_prompts) == 2
    assert "Trusted orchestration correction record" in planner_prompts[1]
    assert "Failed validation attempt: 1" in planner_prompts[1]
    assert "Exact bounded validator diagnostic" in planner_prompts[1]
    assert planner_prompts[1].count(
        "plan_state must include architecture_impact for this fresh contract turn."
    ) == 1
    # The re-prompt names the field and its accepted values (#925).
    assert "`changed` or `unchanged`" in planner_prompts[1]
    assert "AGENT_PLAN_VALIDATION_DIAGNOSTIC" not in planner_prompts[1]
    assert "2026-09-17T05:31:00Z" not in planner_prompts[1]
    assert len(runner.verified_round_bodies) == 2


def test_unsatisfied_plan_repair_is_deterministic_and_the_rerun_is_accepted(tmp_path):
    """An uncorroborated near miss plus a repairable defect (#925).

    Repair fixes the envelope but may not supply the assessment, so the repair
    outcome is a deterministic contract refusal; the planner re-prompt names
    the field and the next planner turn is accepted.
    """
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    payload["architecture_impact"] = {"status": "modified", "rationale": "Something changed."}
    payload["unexpected_key"] = True
    invalid_candidate = (
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    repaired_payload.pop("architecture_impact")
    repaired_candidate = (
        json.dumps(repaired_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=None,
        claude_outputs=[invalid_candidate],
        codex_outputs=[],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, agent_max_retries=0, max_rounds=1)

    with patch.object(orchestrator, "attempt_repair", lambda raw, cmd, **kw: repaired_candidate):
        with pytest.raises(AgentInvocationError) as error:
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.failure_category == "deterministic"
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None and exhaustion.candidate_text == repaired_candidate
    assert [r.outcome for r in error.value.preserved_unsatisfied_response.architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]
    assert len(runner.verified_round_bodies) == 1

    runner.claude_outputs = [structured_plan_state(summary="Recovered plan.")]
    runner.codex_outputs = [structured_plan_review(summary="The recovered plan is approved.")]
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    planner_prompts = [command[-1] for command, _cwd in runner.commands if command[:1] == ["claude"]]
    assert len(planner_prompts) == 2
    assert "plan_state must include architecture_impact" in planner_prompts[1]
    assert "`changed` or `unchanged`" in planner_prompts[1]


@pytest.mark.parametrize("context_mode", ["compact", "full"])
def test_plan_parallel_revision_validation_exhaustion_survives_resume(
    tmp_path, context_mode
):
    invalid_payload = json.loads(
        structured_plan_revision(summary="Rejected revision.").split("\n", 1)[0]
    )
    invalid_payload.pop("architecture_impact")
    invalid_revision = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=None,
        claude_outputs=[structured_plan_state(summary="Initial plan."), invalid_revision],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="The plan needs one correction.",
                blocking_plan_issues=["Add the missing verification step."],
            )
        ],
        gemini_outputs=[],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        review_parallel=True,
        planning_context_mode=context_mode,
        max_rounds=2,
        agent_max_retries=0,
    )

    with patch.object(orchestrator, "_run_structured_repair", return_value=(None, None, [])):
        with pytest.raises(AgentInvocationError) as error:
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.plan_validation_exhaustion is not None
    diagnostic_comments = [
        comment for comment in runner.issue_comments
        if "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.get("body", "")
    ]
    assert len(diagnostic_comments) == 1

    runner.claude_outputs = [structured_plan_revision(summary="Recovered revision.")]
    runner.codex_outputs = [
        structured_plan_review(
            state="approved",
            summary="The recovered revision is approved.",
            prior_plan_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"}
            ],
        )
    ]
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompts = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["claude"]
    ]
    assert len(planner_prompts) == 3
    assert "Trusted orchestration correction record" in planner_prompts[-1]
    assert "Add the missing verification step." in planner_prompts[-1]
    assert len([command for command, _cwd in runner.commands if command[:2] == ["codex", "exec"]]) == 2
    assert len(runner.verified_round_bodies) == 2


def test_repeated_exhausted_plan_failures_select_highest_attempt_across_invocations(
    tmp_path,
):
    invalid_payload = json.loads(
        structured_plan_state(summary="Rejected plan.").split("\n", 1)[0]
    )
    invalid_payload.pop("architecture_impact")
    invalid_candidate = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=None,
        claude_outputs=[invalid_candidate],
        codex_outputs=[],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, agent_max_retries=0, max_rounds=1)

    with patch.object(orchestrator, "_run_structured_repair", return_value=(None, None, [])):
        with pytest.raises(AgentInvocationError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    runner.claude_outputs = [invalid_candidate]
    with patch.object(orchestrator, "_run_structured_repair", return_value=(None, None, [])):
        with pytest.raises(AgentInvocationError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    runner.claude_outputs = [structured_plan_state(summary="Recovered plan.")]
    runner.codex_outputs = [structured_plan_review(summary="The recovered plan is approved.")]
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    diagnostic_bodies = [
        comment["body"]
        for comment in runner.issue_comments
        if "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.get("body", "")
    ]
    assert len(diagnostic_bodies) == 2
    planner_prompts = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["claude"]
    ]
    assert "Failed validation attempt: 2" in planner_prompts[-1]


def test_plan_validation_diagnostic_is_ignored_after_actor_change(tmp_path):
    payload = PlanValidationDiagnosticPayload(
        repository="OWNER/REPO",
        issue_number=56,
        planning_generation=1,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        architecture_contract_version=1,
        execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None,
        expected_producer_login="agent",
        expected_producer_id=7,
        failure_attempt=1,
        candidate_digest="a" * 64,
        category="deterministic",
        diagnostic="old actor diagnostic",
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=str(encode_plan_validation_diagnostic_body(payload)),
        claude_outputs=[structured_plan_state(summary="Recovered by the new actor.")],
        codex_outputs=[structured_plan_review(summary="The recovered plan is approved.")],
        gemini_outputs=[],
    )
    runner.actor_login = "different-agent"
    runner.actor_id = 8
    config = make_config(tmp_path, max_rounds=1)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompt = next(
        command[-1] for command, _cwd in runner.commands if command[:1] == ["claude"]
    )
    assert "old actor diagnostic" not in planner_prompt
    assert sum(
        "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.get("body", "")
        for comment in runner.issue_comments
    ) == 1


def test_plan_validation_diagnostic_with_stale_context_is_ignored_in_workflow(tmp_path):
    payload = PlanValidationDiagnosticPayload(
        repository="OWNER/REPO",
        issue_number=56,
        planning_generation=1,
        target_coder_round=2,
        prior_plan_subject="b" * 64,
        candidate_kind="plan_revision",
        architecture_contract_version=1,
        execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None,
        expected_producer_login="agent",
        expected_producer_id=7,
        failure_attempt=4,
        candidate_digest="b" * 64,
        category="deterministic",
        diagnostic="stale revision diagnostic",
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=str(encode_plan_validation_diagnostic_body(payload)),
        claude_outputs=[structured_plan_state(summary="Fresh plan ignores stale record.")],
        codex_outputs=[structured_plan_review(summary="The fresh plan is approved.")],
        gemini_outputs=[],
    )
    config = make_config(tmp_path, max_rounds=1)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompt = next(
        command[-1] for command, _cwd in runner.commands if command[:1] == ["claude"]
    )
    assert "stale revision diagnostic" not in planner_prompt


@pytest.mark.parametrize("context_mode", ["compact", "full"])
def test_revision_success_supersedes_diagnostic_before_a_later_same_invocation_prompt(
    tmp_path, context_mode
):
    """A verified replacement must not leak its old correction into round N+1."""
    initial_plan = structured_plan_state(summary="Existing plan.")
    initial_subject = orchestrator._plan_subject(initial_plan)
    initial_comment = orchestrator._attach_round_metadata(
        initial_plan,
        orchestrator.PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Anthropic Claude",
            round_number=1,
            subject=initial_subject,
            prior_plan_subject=None,
            canonical_plan=initial_plan,
            raw_structured_coder_response=initial_plan,
            state="blocking",
            architecture_contract_version=1,
        ),
    )
    diagnostic = PlanValidationDiagnosticPayload(
        repository="OWNER/REPO",
        issue_number=56,
        planning_generation=1,
        target_coder_round=2,
        prior_plan_subject=initial_subject,
        candidate_kind="plan_revision",
        architecture_contract_version=1,
        execution_strategy_contract_version=None,
        risk_test_matrix_contract_version=None,
        expected_producer_login="agent",
        expected_producer_id=7,
        failure_attempt=1,
        candidate_digest="c" * 64,
        category="deterministic",
        diagnostic="the round-two correction must not survive its replacement",
    )
    runner = _PlanDiagnosticParallelRunner(
        diagnostic_body=str(encode_plan_validation_diagnostic_body(diagnostic)),
        claude_outputs=[
            structured_plan_revision(summary="Replacement revision."),
            structured_plan_revision(
                summary="Later revision.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="The existing plan needs one correction.",
                blocking_plan_issues=["Add the missing verification step."],
            ),
            structured_plan_review(
                state="blocking",
                summary="The replacement still needs one correction.",
                blocking_plan_issues=["Clarify the rollback step."],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
            structured_plan_review(
                summary="The later revision is approved.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        gemini_outputs=[],
    )
    runner.issue_comments.insert(
        0,
        {
            "author": {"login": "history", "id": 99},
            "createdAt": "2026-09-17T05:00:00Z",
            "body": str(initial_comment),
            "id": 699,
        },
    )
    config = make_config(
        tmp_path,
        reviewer="codex",
        planning_context_mode=context_mode,
        max_rounds=3,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    planner_prompts = [
        command[-1]
        for command, _cwd in runner.commands
        if command[:1] == ["claude"]
    ]
    assert len(planner_prompts) == 2
    assert "the round-two correction must not survive its replacement" in planner_prompts[0]
    assert "the round-two correction must not survive its replacement" not in planner_prompts[1]
    assert len(runner.verified_round_bodies) == 1


# ---------------------------------------------------------------------------
# CLI / config plumbing
# ---------------------------------------------------------------------------

def test_review_parallel_flag_scoped_to_issue_pr_task_not_discuss():
    for command, extra_args in (
        ("issue", ["56"]),
        ("pr", ["77"]),
        ("task", ["do the thing"]),
    ):
        args = build_parser().parse_args(
            [command, *extra_args, "--repo", "OWNER/REPO", "--review-parallel"]
        )
        assert args.review_parallel is True
        plain = build_parser().parse_args([command, *extra_args, "--repo", "OWNER/REPO"])
        assert plain.review_parallel is False

    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["discuss", "56", "--repo", "OWNER/REPO", "--review-parallel"]
        )


# ---------------------------------------------------------------------------
# Concurrency probes
# ---------------------------------------------------------------------------

class _ReviewConcurrencyProbeRunner(FakeRunner):
    """Each reviewer blocks until the other has started, so a timeout means
    same-round reviewers did not truly overlap."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.codex_started = threading.Event()
        self.gemini_started = threading.Event()
        self.overlap_confirmed = True

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:2] == ["codex", "exec"]:
            self.codex_started.set()
            if not self.gemini_started.wait(timeout=10):
                self.overlap_confirmed = False
        elif cmd[:1] == ["gemini"]:
            self.gemini_started.set()
            if not self.codex_started.wait(timeout=10):
                self.overlap_confirmed = False
        return super().run_with_log(args, cwd=cwd, **kwargs)


def test_plan_first_parallel_runs_same_round_reviewers_concurrently(tmp_path):
    runner = _ReviewConcurrencyProbeRunner(
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan review complete.")],
        gemini_outputs=[
            structured_plan_review(summary="Gemini plan review complete.", reviewer="Google Gemini")
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.overlap_confirmed, "same-round plan reviewers did not run concurrently"
    assert any("Codex plan review complete." in comment for comment in runner.comments)
    assert any("Gemini plan review complete." in comment for comment in runner.comments)
    assert any("reconciliation" in comment for comment in runner.comments)


def test_pr_loop_parallel_runs_same_round_reviewers_concurrently(tmp_path):
    runner = _ReviewConcurrencyProbeRunner(
        codex_outputs=[structured_pr_review(summary="Codex PR review complete.")],
        gemini_outputs=[
            structured_pr_review(summary="Gemini PR review complete.", reviewer="Google Gemini")
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.overlap_confirmed, "same-round PR reviewers did not run concurrently"
    assert any("Codex PR review complete." in comment for comment in runner.comments)
    assert any("Gemini PR review complete." in comment for comment in runner.comments)
    assert "reconciliation" in runner.comments[-1]


@pytest.mark.parametrize("context_mode", ["compact", "full"])
def test_selective_parallel_pr_loop_rechecks_owner_then_only_missing_sweep_reviewer(
    tmp_path, monkeypatch, context_mode
):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    monkeypatch.setattr(
        orchestrator,
        "_observe_pr_transition",
        lambda *args, **kwargs: orchestrator.TransitionClassification("narrow", "scoped fix"),
    )
    observed_reviewer_sessions = []
    original_run_validated_agent = orchestrator._run_validated_agent

    def run_validated_agent_with_session_observation(*args, **kwargs):
        if kwargs.get("role") == "reviewer":
            observed_reviewer_sessions.append((kwargs["agent"], kwargs.get("session_id")))
        response = original_run_validated_agent(*args, **kwargs)
        if kwargs.get("role") == "reviewer":
            return dataclasses.replace(
                response,
                session_id=f"{kwargs['agent']}-session",
            )
        return response

    monkeypatch.setattr(
        orchestrator,
        "_run_validated_agent",
        run_validated_agent_with_session_observation,
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[review(reviewer="Google Gemini"), review(reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        review_parallel=True,
        pr_review_policy="selective-intermediate",
        pr_review_context_mode=context_mode,
        max_rounds=4,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini"}
    ]
    assert [command[0] for command in reviewer_commands].count("codex") == 2
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    gemini_prompts = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]
    ]
    assert any("Returning reviewer handoff context" in prompt for prompt in gemini_prompts)
    assert [
        session_id for agent, session_id in observed_reviewer_sessions if agent == "gemini"
    ] == [None, None]


# ---------------------------------------------------------------------------
# Parity with sequential
# ---------------------------------------------------------------------------

def test_plan_first_parallel_matches_sequential_comments(tmp_path):
    claude_outputs = [_initial_plan()]
    codex_outputs = [structured_plan_review(summary="Codex approves the plan.")]
    gemini_outputs = [
        structured_plan_review(summary="Gemini approves the plan.", reviewer="Google Gemini")
    ]

    sequential_runner = FakeRunner(
        claude_outputs=list(claude_outputs),
        codex_outputs=list(codex_outputs),
        gemini_outputs=list(gemini_outputs),
    )
    sequential_config = make_config(
        tmp_path / "seq", reviewer=("codex", "gemini"), log_dir=tmp_path / "seq" / "logs"
    )
    parallel_runner = FakeRunner(
        claude_outputs=list(claude_outputs),
        codex_outputs=list(codex_outputs),
        gemini_outputs=list(gemini_outputs),
    )
    parallel_config = make_config(
        tmp_path / "par",
        reviewer=("codex", "gemini"),
        log_dir=tmp_path / "par" / "logs",
        review_parallel=True,
    )

    assert run_issue_loop(sequential_runner, issue_number=56, config=sequential_config, plan_first=True) == 0
    assert run_issue_loop(parallel_runner, issue_number=56, config=parallel_config, plan_first=True) == 0

    reviewer_comments = lambda runner: {
        comment for comment in runner.comments if comment.startswith("**Review verdict:")
    }
    assert reviewer_comments(parallel_runner) == reviewer_comments(sequential_runner)
    assert "reconciliation" in parallel_runner.comments[-2]


def test_pr_loop_parallel_matches_sequential_comments(tmp_path):
    codex_outputs = [structured_pr_review(summary="Codex approves the PR.")]
    gemini_outputs = [
        structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")
    ]

    sequential_runner = FakeRunner(codex_outputs=list(codex_outputs), gemini_outputs=list(gemini_outputs))
    sequential_config = make_config(
        tmp_path / "seq", reviewer=("codex", "gemini"), log_dir=tmp_path / "seq" / "logs"
    )
    parallel_runner = FakeRunner(codex_outputs=list(codex_outputs), gemini_outputs=list(gemini_outputs))
    parallel_config = make_config(
        tmp_path / "par",
        reviewer=("codex", "gemini"),
        log_dir=tmp_path / "par" / "logs",
        review_parallel=True,
    )

    assert run_pr_loop(sequential_runner, pr_number=77, config=sequential_config) == 0
    assert run_pr_loop(parallel_runner, pr_number=77, config=parallel_config) == 0

    reviewer_comments = lambda runner: {
        comment for comment in runner.comments if comment.startswith("**Review verdict:")
    }
    assert reviewer_comments(parallel_runner) == reviewer_comments(sequential_runner)
    assert "reconciliation" in parallel_runner.comments[-1]


def test_pr_parallel_resolves_disputed_scope_claim_and_tracks_translation_defect_as_fresh_item(tmp_path):
    """PR #802 shape: accept the old premise, then file the distinct defect anew."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Scope completeness is not established.",
                blocking_items=["Only 3 of 14 locale catalogs appear to be evaluated."],
            ),
            structured_pr_review(
                summary="The scope-completeness evidence is sufficient.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(
                summary="Codex approves the translation correction.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves the initial diff.", reviewer="Google Gemini"),
            structured_pr_review(
                state="blocking",
                summary="The scope premise is accepted, but a retained translation is wrong.",
                blocking_items=["The retained German translation reverses the confirmation action."],
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                reviewer="Google Gemini",
            ),
            structured_pr_review(
                summary="Gemini approves the translation correction.",
                prior_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
                reviewer="Google Gemini",
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                summary="All 14 locales and 49 keys were evaluated.",
                disputed_items=["item-1"],
                dispute_evidence={"item-1": "The approved no-churn audit covers every locale/key pair."},
            ),
            structured_coder_followup(
                summary="Corrected the retained German translation.",
                addressed_items=["item-2"],
            ),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, max_rounds=3)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    reviewer_metadata = [
        orchestrator._decode_round_metadata(match["payload"])
        for comment in runner.pr_payload["comments"]
        if (match := orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"]))
    ]
    gemini_round_two = next(
        record for record in reviewer_metadata
        if record.agent == "Gemini" and record.round_number == 2
    )
    # Parallel reviewer comments are published before deterministic allocation;
    # reconciliation metadata is the authoritative fresh/resume ledger.
    reconciliation = next(
        record for record in reviewer_metadata
        if record.role == "summary" and record.round_number == 2
    )
    assert gemini_round_two.dispositions[0].disposition == "resolved"
    assert [item.item_id for item in reconciliation.new_items] == ["item-2"]
    assert reconciliation.new_items[0].text == (
        "The retained German translation reverses the confirmation action."
    )
    assert all(
        item.item_id != "item-1" for item in reconciliation.new_items
    )


class _PeerVisibilityBarrierRunner(FakeRunner):
    """Makes Gemini's first turn slow and records what it could read mid-turn (#1025).

    The observation window is the first parallel review round: from the first
    reviewer launch until Gemini's first turn returns.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.review_round_started = threading.Event()
        self.gemini_finished = threading.Event()
        self.comment_posted_before_gemini_finished = False
        self.coder_started_before_gemini_finished = False
        self.surface_bodies_seen_by_gemini: list[str] = []

    def _in_first_review_window(self) -> bool:
        return self.review_round_started.is_set() and not self.gemini_finished.is_set()

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:1] == ["claude"] and self._in_first_review_window():
            self.coder_started_before_gemini_finished = True
        if cmd[:1] in (["codex"], ["gemini"]):
            self.review_round_started.set()
        if cmd[:1] == ["gemini"] and not self.gemini_finished.is_set():
            time.sleep(0.15)
            # Everything a tool-enabled reviewer could read from the shared
            # PR/issue surface at the end of its turn.
            self.surface_bodies_seen_by_gemini.extend(self.comments)
            result = super().run_with_log(args, cwd=cwd, **kwargs)
            self.gemini_finished.set()
            return result
        return super().run_with_log(args, cwd=cwd, **kwargs)

    def run(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:3] in (["gh", "pr", "comment"], ["gh", "issue", "comment"]) and self._in_first_review_window():
            self.comment_posted_before_gemini_finished = True
        return super().run(args, cwd=cwd, **kwargs)


def test_pr_parallel_withholds_fast_review_until_round_settles(tmp_path):
    runner = _PeerVisibilityBarrierRunner(
        codex_outputs=[structured_pr_review(
            state="blocking", summary="Codex found two blockers.",
            blocking_items=["First blocker.", "Second blocker."],
        ), structured_pr_review(
            summary="Codex approves after the fix.",
            prior_item_dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "resolved"},
            ],
        )],
        gemini_outputs=[structured_pr_review(summary="Gemini approves.", reviewer="Google Gemini"),
                        structured_pr_review(
                            summary="Gemini approves after the fix.", reviewer="Google Gemini",
                            prior_item_dispositions=[
                                {"item_id": "item-1", "disposition": "resolved"},
                                {"item_id": "item-2", "disposition": "resolved"},
                            ],
                        )],
        claude_outputs=[structured_coder_followup(
            summary="Fixed both blockers.", addressed_items=["item-1", "item-2"]
        )],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # The fast Codex review is not published while Gemini is still running,
    # so the slow reviewer cannot read (and echo) its peer's findings.
    assert not runner.comment_posted_before_gemini_finished
    assert not any(
        "First blocker." in body or "Codex found two blockers." in body
        for body in runner.surface_bodies_seen_by_gemini
    )
    assert not runner.coder_started_before_gemini_finished
    # The review is still published once the round settles.
    assert any("Codex found two blockers." in body for body in runner.comments)
    coder_prompt = next("\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "[item-1]" in coder_prompt and "[item-2]" in coder_prompt


def test_plan_parallel_withholds_fast_review_until_round_settles(tmp_path):
    runner = _PeerVisibilityBarrierRunner(
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan approval with a unique note.")],
        gemini_outputs=[structured_plan_review(summary="Gemini approves the plan.", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert not runner.comment_posted_before_gemini_finished
    assert not any(
        "Codex plan approval with a unique note." in body
        for body in runner.surface_bodies_seen_by_gemini
    )
    assert any("Codex plan approval with a unique note." in body for body in runner.comments)


def test_pr_parallel_resume_after_publication_does_not_duplicate_or_lose_items(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Codex found a blocker.", blocking_items=["Persist this item."]
            ),
            structured_pr_review(
                summary="Codex approves after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves.", reviewer="Google Gemini"),
            structured_pr_review(
                summary="Gemini approves after the fix.", reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(
            summary="Fixed the persisted item.", addressed_items=["item-1"]
        )],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    real_post = orchestrator.post_pr_comment

    def interrupt_before_reconciliation(*args, **kwargs):
        if "reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(orchestrator, "post_pr_comment", side_effect=interrupt_before_reconciliation):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)

    codex_publications = lambda: sum(
        (metadata := orchestrator._decode_round_metadata(
            orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])["payload"]
        )).agent == "Codex"
        and metadata.phase == "publication"
        and metadata.round_number == 1
        for comment in runner.pr_payload["comments"]
        if orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])
    )
    assert codex_publications() == 1
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert codex_publications() == 1
    coder_prompt = next("\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "[item-1]" in coder_prompt
    summary_block = coder_prompt.split(
        "Latest reviewer summaries (review-level context):", 1
    )[1].split("Codex unresolved blocking item", 1)[0]
    assert "Codex found a blocker." in summary_block
    assert "blocking_items" not in summary_block
    assert "Persist this item." not in summary_block


class _PartialPublicationProbeRunner(FakeRunner):
    """Records every reviewer launch and the same-round bodies public at that moment."""

    def __init__(self, *, round_markers, slow_reviewer=None, **kwargs):
        super().__init__(**kwargs)
        self.round_markers = round_markers
        # Delaying one reviewer's first turn fixes the completion (and so
        # publication) order of the first round.
        self.slow_reviewer = slow_reviewer
        self.reviewer_launches: list[str] = []
        self.peer_body_visible_at_launch = False

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:1] in (["codex"], ["gemini"]):
            self.reviewer_launches.append(cmd[0])
            if any(marker in body for marker in self.round_markers for body in self.comments):
                self.peer_body_visible_at_launch = True
            if cmd[0] == self.slow_reviewer and self.reviewer_launches.count(cmd[0]) == 1:
                time.sleep(0.2)
        return super().run_with_log(args, cwd=cwd, **kwargs)


def _interrupt_on_second_review_post(real_post, markers):
    """Interrupt the second same-round reviewer publication, whichever finished first."""
    state = {"posted": 0, "fired": False}

    def post(*args, **kwargs):
        body = str(kwargs["body"])
        if not state["fired"] and any(marker in body for marker in markers):
            state["posted"] += 1
            if state["posted"] == 2:
                state["fired"] = True
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    return post


def _published_count(runner, marker):
    return sum(marker in body for body in runner.comments)


def _spool_files(config):
    root = orchestrator.review_spool_root(config.agent_memory_dir)
    return sorted(root.rglob("*.json")) if root.exists() else []


def test_pr_parallel_interruption_between_publications_replays_withheld_review(tmp_path):
    markers = ("Codex found a blocker.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Codex found a blocker.", blocking_items=["Persist this item."]
            ),
            structured_pr_review(
                summary="Codex approves after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves independently.", reviewer="Google Gemini"),
            structured_pr_review(
                summary="Gemini approves after the fix.", reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(
            summary="Fixed the persisted item.", addressed_items=["item-1"]
        )],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    real_post = orchestrator.post_pr_comment

    # Stop after the first reviewer body is public but before the second one.
    with patch.object(
        orchestrator, "post_pr_comment",
        side_effect=_interrupt_on_second_review_post(real_post, markers),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert sorted(_published_count(runner, marker) for marker in markers) == [0, 1]
    assert _spool_files(config), "withheld review was not persisted before publication"
    launches_before_rerun = list(runner.reviewer_launches)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    rerun_launches = runner.reviewer_launches[len(launches_before_rerun):]
    # Round 1 is completed from the spool: neither round-1 reviewer runs again.
    # The next launches belong to round 2, after the coder follow-up.
    assert rerun_launches.count("gemini") == 1 and rerun_launches.count("codex") == 1
    first_claude = next(
        index for index, (cmd, _cwd) in enumerate(runner.commands) if cmd[:1] == ["claude"]
    )
    round_one_reviewers = [
        cmd[0] for cmd, _cwd in runner.commands[:first_claude] if cmd[:1] in (["codex"], ["gemini"])
    ]
    assert sorted(round_one_reviewers) == ["codex", "gemini"]
    assert [_published_count(runner, marker) for marker in markers] == [1, 1]
    assert _spool_files(config) == []


def test_plan_parallel_interruption_between_publications_replays_withheld_review(tmp_path):
    markers = ("Codex plan approval with a unique note.", "Gemini independent plan approval.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan approval with a unique note.")],
        gemini_outputs=[structured_plan_review(
            summary="Gemini independent plan approval.", reviewer="Google Gemini"
        )],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    real_post = orchestrator.post_issue_comment

    with patch.object(
        orchestrator, "post_issue_comment",
        side_effect=_interrupt_on_second_review_post(real_post, markers),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert sorted(_published_count(runner, marker) for marker in markers) == [0, 1]
    assert _spool_files(config)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    # The withheld review was replayed, never re-run against its peer's
    # already-public body.
    assert sorted(runner.reviewer_launches) == ["codex", "gemini"]
    assert not runner.peer_body_visible_at_launch
    assert [_published_count(runner, marker) for marker in markers] == [1, 1]
    assert _spool_files(config) == []


def _pr_partial_round_runner(slow_reviewer=None):
    markers = ("Codex found a blocker.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        slow_reviewer=slow_reviewer,
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Codex found a blocker.", blocking_items=["Persist this item."]
            ),
            structured_pr_review(
                summary="Codex approves after the fix.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves independently.", reviewer="Google Gemini"),
            structured_pr_review(
                summary="Gemini approves after the fix.", reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(
            summary="Fixed the persisted item.", addressed_items=["item-1"]
        )],
    )
    return runner, markers


def _interrupt_pr_round_between_publications(runner, config, markers):
    with patch.object(
        orchestrator, "post_pr_comment",
        side_effect=_interrupt_on_second_review_post(orchestrator.post_pr_comment, markers),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert sorted(_published_count(runner, marker) for marker in markers) == [0, 1]


def _plan_partial_round_runner():
    markers = ("Codex plan approval with a unique note.", "Gemini independent plan approval.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan approval with a unique note.")],
        gemini_outputs=[structured_plan_review(
            summary="Gemini independent plan approval.", reviewer="Google Gemini"
        )],
    )
    return runner, markers


def _interrupt_plan_round_between_publications(runner, config, markers):
    with patch.object(
        orchestrator, "post_issue_comment",
        side_effect=_interrupt_on_second_review_post(orchestrator.post_issue_comment, markers),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert sorted(_published_count(runner, marker) for marker in markers) == [0, 1]


def test_pr_partial_round_sequential_resume_replays_withheld_review(tmp_path):
    runner, markers = _pr_partial_round_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    launches_before = len(runner.reviewer_launches)

    sequential = dataclasses.replace(config, review_parallel=False)
    assert run_pr_loop(runner, pr_number=77, config=sequential) == 0

    first_claude = next(
        index for index, (cmd, _cwd) in enumerate(runner.commands) if cmd[:1] == ["claude"]
    )
    round_one = [
        cmd[0] for cmd, _cwd in runner.commands[:first_claude] if cmd[:1] in (["codex"], ["gemini"])
    ]
    # Neither round-1 reviewer is re-invoked by the sequential resume.
    assert sorted(round_one) == ["codex", "gemini"]
    assert len(runner.reviewer_launches) - launches_before == 2  # round 2 only
    assert [_published_count(runner, marker) for marker in markers] == [1, 1]
    assert _spool_files(config) == []


def test_plan_partial_round_sequential_resume_replays_withheld_review(tmp_path):
    runner, markers = _plan_partial_round_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_plan_round_between_publications(runner, config, markers)

    sequential = dataclasses.replace(config, review_parallel=False)
    assert run_issue_loop(runner, issue_number=56, config=sequential, plan_first=True) == 0

    assert sorted(runner.reviewer_launches) == ["codex", "gemini"]
    assert not runner.peer_body_visible_at_launch
    assert [_published_count(runner, marker) for marker in markers] == [1, 1]
    assert _spool_files(config) == []


@pytest.mark.parametrize("review_parallel", [True, False])
def test_pr_partial_round_without_spool_record_stops_before_invoking(tmp_path, review_parallel):
    runner, markers = _pr_partial_round_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    with pytest.raises(orchestrator.PartialReviewRoundError, match="partially published"):
        run_pr_loop(
            runner, pr_number=77,
            config=dataclasses.replace(config, review_parallel=review_parallel),
        )

    assert len(runner.reviewer_launches) == launches_before
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("review_parallel", [True, False])
def test_plan_partial_round_with_invalid_spool_record_stops_before_invoking(tmp_path, review_parallel):
    runner, markers = _plan_partial_round_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_plan_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["response"]["text"] = "not a structured plan review"
        path.write_text(json.dumps(payload), encoding="utf-8")
    launches_before = len(runner.reviewer_launches)

    with pytest.raises(orchestrator.PartialReviewRoundError, match="partially published"):
        run_issue_loop(
            runner, issue_number=56,
            config=dataclasses.replace(config, review_parallel=review_parallel), plan_first=True,
        )

    assert len(runner.reviewer_launches) == launches_before


def test_pr_partial_round_replays_unavailable_reviewer_without_reinvoking(tmp_path):
    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    markers = ("Gemini approves the PR.", "Claude approves the PR.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        codex_outputs=[unavailable],
        gemini_outputs=[structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")],
        claude_outputs=[structured_pr_review(summary="Claude approves the PR.", reviewer="Anthropic Claude")],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini", "claude"), coder="codex", review_parallel=True
    )
    with patch.object(
        orchestrator, "post_pr_comment",
        side_effect=_interrupt_on_second_review_post(orchestrator.post_pr_comment, markers),
    ):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert sorted(_published_count(runner, marker) for marker in markers) == [0, 1]
    spooled = [json.loads(path.read_text(encoding="utf-8")) for path in _spool_files(config)]
    assert any(record.get("failure") for record in spooled)
    launches_before = list(runner.reviewer_launches)

    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)

    # The unavailable Codex outcome is replayed, not re-run against public peers.
    assert runner.reviewer_launches == launches_before
    assert [_published_count(runner, marker) for marker in markers] == [1, 1]


def test_pr_held_round_sequential_rerun_never_shows_withheld_peer_to_retry(tmp_path):
    runner = _PartialPublicationProbeRunner(
        round_markers=("Codex approves the PR first.",),
        codex_outputs=[structured_pr_review(summary="Codex approves the PR first.")],
        gemini_outputs=[("gemini exploded", 1)],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )
    with pytest.raises(AgentLoopError, match="Gemini"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _published_count(runner, "Codex approves the PR first.") == 0
    assert _spool_files(config)

    runner.gemini_outputs.append(
        structured_pr_review(summary="Gemini approves on retry.", reviewer="Google Gemini")
    )
    launches_before = list(runner.reviewer_launches)
    sequential = dataclasses.replace(config, review_parallel=False)
    assert run_pr_loop(runner, pr_number=77, config=sequential) == 0

    # Codex, first in configured order, is replayed rather than posted before
    # Gemini's retry, so the retry cannot read it.
    assert runner.reviewer_launches[len(launches_before):] == ["gemini"]
    assert not runner.peer_body_visible_at_launch
    assert _published_count(runner, "Codex approves the PR first.") == 1
    assert _published_count(runner, "Gemini approves on retry.") == 1
    assert _spool_files(config) == []


def test_plan_held_round_sequential_rerun_never_shows_withheld_peer_to_retry(tmp_path):
    runner = _PartialPublicationProbeRunner(
        round_markers=("Codex plan approval with a unique note.",),
        claude_outputs=[_initial_plan()],
        codex_outputs=[structured_plan_review(summary="Codex plan approval with a unique note.")],
        gemini_outputs=[structured_plan_review(
            state="blocking",
            summary="Review incomplete: could not confirm the prior finding.",
            reviewer="Google Gemini",
        )],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    with pytest.raises(AgentLoopError, match="Gemini"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert _published_count(runner, "Codex plan approval with a unique note.") == 0

    runner.gemini_outputs.append(
        structured_plan_review(summary="Gemini approves on retry.", reviewer="Google Gemini")
    )
    sequential = dataclasses.replace(config, review_parallel=False)
    assert run_issue_loop(runner, issue_number=56, config=sequential, plan_first=True) == 0

    assert sorted(runner.reviewer_launches) == ["codex", "gemini", "gemini"]
    assert not runner.peer_body_visible_at_launch
    assert _published_count(runner, "Codex plan approval with a unique note.") == 1
    assert _spool_files(config) == []


@pytest.mark.parametrize("review_parallel", [True, False])
@pytest.mark.parametrize("spool_state", ["missing", "invalid"])
@pytest.mark.parametrize("slow_reviewer", ["codex", "gemini"])
def test_pr_partial_round_with_rejected_posted_record_stops_before_invoking(
    tmp_path, monkeypatch, review_parallel, spool_state, slow_reviewer
):
    runner, markers = _pr_partial_round_runner(slow_reviewer=slow_reviewer)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        if spool_state == "missing":
            path.unlink()
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["response"]["text"] = "not a structured PR review"
            path.write_text(json.dumps(payload), encoding="utf-8")
    # Resume rejects the posted record (e.g. requirements changed), but its
    # body is still public on the PR.
    monkeypatch.setattr(
        orchestrator, "_resumed_pr_reviewer_matches_requirements", lambda *args, **kwargs: False
    )
    launches_before = len(runner.reviewer_launches)

    with pytest.raises(orchestrator.PartialReviewRoundError, match="partially published"):
        run_pr_loop(
            runner, pr_number=77,
            config=dataclasses.replace(config, review_parallel=review_parallel),
        )

    assert len(runner.reviewer_launches) == launches_before


def _held_pr_round_with_invalid_gemini_spool(tmp_path):
    runner = _PartialPublicationProbeRunner(
        round_markers=("Gemini approves the PR.",),
        codex_outputs=[("codex exploded", 1)],
        gemini_outputs=[structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )
    with pytest.raises(AgentLoopError, match="Codex"):
        run_pr_loop(runner, pr_number=77, config=config)
    # No same-round body is public; Gemini's withheld review no longer validates.
    assert runner.comments == []
    for path in _spool_files(config):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["response"]["text"] = "not a structured PR review"
        path.write_text(json.dumps(payload), encoding="utf-8")
    runner.codex_outputs.append(structured_pr_review(summary="Codex approves after rerun."))
    return runner, config


def test_pr_spooled_reviewer_falling_back_to_fresh_turn_is_synced_first(tmp_path):
    runner, config = _held_pr_round_with_invalid_gemini_spool(tmp_path)
    runner.gemini_outputs.append(
        structured_pr_review(summary="Gemini approves on a fresh turn.", reviewer="Google Gemini")
    )
    real_sync = orchestrator.sync_reviewer_pr_before_review
    gemini_launches_at_sync = []

    def recording_sync(config, runner_arg, reviewer, pr_number, pr_metadata):
        if reviewer == "gemini":
            gemini_launches_at_sync.append(runner.reviewer_launches.count("gemini"))
        return real_sync(config, runner_arg, reviewer, pr_number, pr_metadata)

    launches_before = runner.reviewer_launches.count("gemini")
    with patch.object(orchestrator, "sync_reviewer_pr_before_review", recording_sync):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # Gemini's replay failed, so it ran a fresh turn -- after being synced.
    assert runner.reviewer_launches.count("gemini") == launches_before + 1
    assert gemini_launches_at_sync and gemini_launches_at_sync[0] == launches_before


def test_pr_spooled_reviewer_fallback_sync_failure_is_not_launched(tmp_path):
    runner, config = _held_pr_round_with_invalid_gemini_spool(tmp_path)
    real_sync = orchestrator.sync_reviewer_pr_before_review

    def failing_sync(config, runner_arg, reviewer, pr_number, pr_metadata):
        if reviewer == "gemini":
            raise AgentLoopError("Gemini checkout is desynced from the PR head.")
        return real_sync(config, runner_arg, reviewer, pr_number, pr_metadata)

    launches_before = runner.reviewer_launches.count("gemini")
    with patch.object(orchestrator, "sync_reviewer_pr_before_review", failing_sync):
        with pytest.raises(AgentLoopError, match="desynced"):
            run_pr_loop(runner, pr_number=77, config=config)

    assert runner.reviewer_launches.count("gemini") == launches_before
    assert not runner.peer_body_visible_at_launch


def test_review_round_spool_rejects_foreign_or_malformed_records(tmp_path):
    from coding_review_agent_loop.review_spool import ReviewRoundSpool

    spool = ReviewRoundSpool(
        root=tmp_path, repo="o/r", surface="pr", number=7, round_number=1, subject="abc"
    )
    spool.store("Codex", {"text": "body", "model_used": "m"})
    assert spool.load("Codex")["text"] == "body"
    assert spool.load("Google Gemini") is None
    other_round = dataclasses.replace(spool, round_number=2)
    assert other_round.load("Codex") is None
    other_subject = dataclasses.replace(spool, subject="def")
    assert other_subject.load("Codex") is None
    path = next(spool.directory.glob("*.json"))
    path.write_text("{not json", encoding="utf-8")
    assert spool.load("Codex") is None
    spool.store_failure("Codex", message="unavailable", failure_category="environment")
    assert spool.load("Codex") == {
        "failure": {"message": "unavailable", "failure_category": "environment"}
    }
    spool.discard()
    assert not spool.directory.exists()


# ---------------------------------------------------------------------------
# Collect-then-apply-then-raise, with resume
# ---------------------------------------------------------------------------

def test_plan_first_parallel_withholds_healthy_review_until_failed_peer_retries(tmp_path):
    runner = _PartialPublicationProbeRunner(
        round_markers=("Gemini approves the plan.",),
        claude_outputs=[_initial_plan()],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Review incomplete: could not confirm the prior finding.",
            )
        ],
        gemini_outputs=[
            structured_plan_review(summary="Gemini approves the plan.", reviewer="Google Gemini")
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    with pytest.raises(AgentLoopError, match="Codex"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # Codex must be re-invoked, so Gemini's healthy review is withheld
    # privately instead of posted (comment[0] is the coder's plan) (#1025).
    assert len(runner.comments) == 1
    assert _published_count(runner, "Gemini approves the plan.") == 0
    assert _spool_files(config)

    # A rerun replays Gemini's withheld review instead of re-invoking it, and
    # Codex's retry runs while no same-round peer body is public.
    commands_before_rerun = len(runner.commands)
    runner.codex_outputs.append(
        structured_plan_review(summary="Codex approves after rerun.")
    )
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    new_commands = runner.commands[commands_before_rerun:]
    gemini_calls_after = [cmd for cmd, _cwd in new_commands if cmd[:1] == ["gemini"]]
    assert len(gemini_calls_after) == 0
    assert any(cmd[:1] == ["codex"] for cmd, _cwd in new_commands)
    assert not runner.peer_body_visible_at_launch
    assert _published_count(runner, "Gemini approves the plan.") == 1
    assert _spool_files(config) == []


def test_pr_loop_parallel_withholds_healthy_review_until_failed_peer_retries(tmp_path):
    runner = _PartialPublicationProbeRunner(
        round_markers=("Gemini approves the PR.",),
        codex_outputs=[("codex exploded", 1)],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")
        ],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )

    with pytest.raises(AgentLoopError, match="Codex"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _published_count(runner, "Gemini approves the PR.") == 0
    assert _spool_files(config)

    commands_before_rerun = len(runner.commands)
    runner.codex_outputs.append(structured_pr_review(summary="Codex approves after rerun."))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    new_commands = runner.commands[commands_before_rerun:]
    gemini_calls_after = [cmd for cmd, _cwd in new_commands if cmd[:1] == ["gemini"]]
    assert len(gemini_calls_after) == 0
    assert any(cmd[:1] == ["codex"] for cmd, _cwd in new_commands)
    assert not runner.peer_body_visible_at_launch
    assert _published_count(runner, "Gemini approves the PR.") == 1
    assert _spool_files(config) == []


# ---------------------------------------------------------------------------
# PR pre-launch sync failure isolation
# ---------------------------------------------------------------------------

def test_pr_loop_parallel_sync_failure_isolated_from_healthy_reviewer(tmp_path):
    real_sync = orchestrator.sync_reviewer_pr_before_review

    def fake_sync(config, runner, reviewer, pr_number, pr_metadata):
        if reviewer == "codex":
            raise AgentLoopError("Codex checkout is desynced from the PR head.")
        return real_sync(config, runner, reviewer, pr_number, pr_metadata)

    runner = FakeRunner(
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    with patch("coding_review_agent_loop.orchestrator.sync_reviewer_pr_before_review", fake_sync):
        with pytest.raises(AgentLoopError, match="desynced"):
            run_pr_loop(runner, pr_number=77, config=config)

    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    # Codex will be re-invoked by a rerun, so Gemini's review stays withheld.
    assert _published_count(runner, "Gemini approves the PR.") == 0
    assert _spool_files(config)


# ---------------------------------------------------------------------------
# Unavailable (non-deterministic) reviewer failure policy
# ---------------------------------------------------------------------------

def test_pr_loop_parallel_unavailable_reviewer_alongside_healthy_reviewer(tmp_path):
    import json

    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[unavailable],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves the PR.", reviewer="Google Gemini")
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    with pytest.raises(AgentLoopError, match="missing required input from Codex"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert "**Review status: Incomplete**" in runner.comments[-1]
    assert "Codex" in runner.comments[-1]
    # Gemini's healthy approval is recorded even though the round ultimately
    # reports an incomplete review because of Codex.
    assert any("Gemini approves the PR." in comment for comment in runner.comments)


def test_pr_loop_parallel_unavailable_reviewer_with_blocking_peer_stops_in_round(tmp_path):
    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    runner = FakeRunner(
        codex_outputs=[unavailable],
        gemini_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Needs a test.",
                blocking_items=["Add a regression test."],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True, max_rounds=3)

    with pytest.raises(AgentLoopError, match="missing required input from Codex") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)

    assert "Detected in round 1" in str(excinfo.value)
    assert "--reviewer flags" in str(excinfo.value)
    assert sum(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands) == 0
    assert sum(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands) == 1
    assert any("Finalization stops after reconciliation" in c for c in runner.comments)
    assert not any("Finalization continues" in c for c in runner.comments)
    assert any("Needs a test." in c for c in runner.comments)
    assert "**Review status: Incomplete**" in runner.comments[-1]


# ---------------------------------------------------------------------------
# Quota precedence
# ---------------------------------------------------------------------------

class _OtherFatalError(AgentLoopError):
    pass


def test_pr_loop_parallel_quota_error_takes_precedence_over_other_fatal_error(tmp_path):
    # Configured order puts the plain fatal failure (Gemini) BEFORE the quota
    # failure (Codex), so this only passes if the raise phase scans every
    # captured failure for a quota error instead of just raising the first
    # one encountered in configured order.
    config = make_config(
        tmp_path, reviewer=("gemini", "codex"), review_parallel=True, agent_max_retries=0
    )
    runner = FakeRunner()

    def fake_run_validated_agent(runner_arg, *, agent, **kwargs):
        if agent == "codex":
            raise QuotaResetExceededError("Codex quota exhausted; resets in 2h.")
        raise _OtherFatalError("Gemini failed deterministically.")

    with patch.object(orchestrator, "_run_validated_agent", side_effect=fake_run_validated_agent):
        with pytest.raises(QuotaResetExceededError):
            run_pr_loop(runner, pr_number=77, config=config)


# ---------------------------------------------------------------------------
# Mixed resumed/pending: zero-pending resume constructs no executor
# ---------------------------------------------------------------------------

def test_plan_first_parallel_zero_pending_resume_constructs_no_executor(tmp_path):
    from coding_review_agent_loop.orchestrator import (
        PostedRoundMetadata,
        _attach_round_metadata,
        _plan_subject,
    )

    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=2,
            subject=_plan_subject(current_plan), prior_items=(),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(current_plan), state="approved",
        ),
    )
    gemini_comment = _attach_round_metadata(
        "Plan looks sound too.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Gemini", round_number=2,
            subject=_plan_subject(current_plan), state="approved",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:06:00Z", "body": gemini_comment},
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)

    def _boom(*args, **kwargs):
        raise AssertionError("no reviewer turn was pending; a thread pool should not be constructed")

    with patch("coding_review_agent_loop.orchestrator.ThreadPoolExecutor", side_effect=_boom):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == []


# ---------------------------------------------------------------------------
# Shared reviewer workdir rejection
# ---------------------------------------------------------------------------

def test_plan_first_parallel_rejects_shared_reviewer_workdirs(tmp_path):
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
        allow_shared_dir=True,
        review_parallel=True,
    )
    runner = FakeRunner(claude_outputs=[], codex_outputs=[], gemini_outputs=[])

    with pytest.raises(AgentLoopError, match="distinct workdir per reviewer"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert len(runner.comments) == 0


def test_pr_loop_parallel_rejects_shared_reviewer_workdirs(tmp_path):
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
        allow_shared_dir=True,
        review_parallel=True,
    )
    runner = FakeRunner(codex_outputs=[], gemini_outputs=[])

    with pytest.raises(AgentLoopError, match="distinct workdir per reviewer"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert len(runner.comments) == 0


def test_pr_loop_parallel_rejects_shared_reviewer_workdirs_with_workdirs_ready_handoff(tmp_path):
    shared = tmp_path / "shared"
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        codex_dir=shared,
        gemini_dir=shared,
        allow_shared_dir=True,
        review_parallel=True,
        create_dirs=False,
    )
    shared.mkdir(parents=True)
    config.claude_dir.mkdir(parents=True)
    runner = FakeRunner(codex_outputs=[], gemini_outputs=[])

    with pytest.raises(AgentLoopError, match="distinct workdir per reviewer"):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)

    assert len(runner.comments) == 0


# ---------------------------------------------------------------------------
# Repair/retry isolation
# ---------------------------------------------------------------------------

def test_plan_first_parallel_repair_isolated_between_reviewers(tmp_path):
    # A repairable source must carry the reviewer's own verdict; narration alone
    # is refused before any repair backend call (#871).
    malformed_codex = malformed_plan_review_source(summary="Codex approves after repair.")
    repaired_codex = structured_plan_review(summary="Codex approves after repair.")
    gemini_ok = structured_plan_review(summary="Gemini approves the plan.", reviewer="Google Gemini")

    lock = threading.Lock()
    repair_calls = []

    def fake_attempt_repair(raw, gemini_cmd, *, expected_kind=None, **kwargs):
        with lock:
            repair_calls.append(raw)
        if raw == malformed_codex:
            return repaired_codex
        return None

    runner = FakeRunner(
        claude_outputs=[_initial_plan()],
        codex_outputs=[malformed_codex],
        gemini_outputs=[gemini_ok],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(repair_calls) == 1
    assert any("Codex approves after repair." in comment for comment in runner.comments)
    assert any("Gemini approves the plan." in comment for comment in runner.comments)


def _staged_review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pr_review",
                "state": state,
                "summary": f"{reviewer} review",
                "blocking_items": blocking_items or [],
                "same_pr_followups": [],
                "future_followups": [],
                "prior_item_dispositions": dispositions or [],
            }
        )
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


class _SecondaryPanelConcurrencyProbeRunner(FakeRunner):
    """The two secondaries block until each other has started; the primary
    must have finished before either one begins."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.gemini_started = threading.Event()
        self.antigravity_started = threading.Event()
        self.primary_finished = threading.Event()
        self.overlap_confirmed = True
        self.panel_started_before_primary = False

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:1] == ["gemini"]:
            if not self.primary_finished.is_set():
                self.panel_started_before_primary = True
            self.gemini_started.set()
            if not self.antigravity_started.wait(timeout=10):
                self.overlap_confirmed = False
        elif cmd[:1] == ["agy"] and "catalog" not in cmd:
            if not self.primary_finished.is_set():
                self.panel_started_before_primary = True
            self.antigravity_started.set()
            if not self.gemini_started.wait(timeout=10):
                self.overlap_confirmed = False
        result = super().run_with_log(args, cwd=cwd, **kwargs)
        if cmd[:2] == ["codex", "exec"]:
            self.primary_finished.set()
        return result


def test_primary_then_panel_parallel_panel_runs_from_one_snapshot_after_primary_approval(tmp_path):
    runner = _SecondaryPanelConcurrencyProbeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        review_parallel=True,
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [
        command[0]
        for command, _cwd in runner.commands
        if command and command[0] in {"claude", "codex", "gemini", "agy"}
    ]
    # The primary runs alone first; the two secondaries only start afterwards
    # and run concurrently from the same frozen head.
    assert agent_commands[0] == "codex"
    assert sorted(agent_commands[1:]) == ["agy", "gemini"]
    assert runner.overlap_confirmed, "secondary panel reviewers did not run concurrently"
    assert not runner.panel_started_before_primary
    assert any("phase: secondary-audit" in comment for comment in runner.comments)


def test_primary_then_panel_parallel_panel_failure_keeps_healthy_result_and_blocks_finalization(
    tmp_path, monkeypatch
):
    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The review checkout cannot access the diff.",
            "suggested_action": "Repair the reviewer sandbox before retrying it.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- Antigravity"
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[unavailable],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        review_parallel=True,
        pr_review_policy="primary-then-panel",
        primary_reviewer="codex",
        auto_merge=True,
        max_rounds=3,
    )
    monkeypatch.setattr(
        orchestrator, "merge_pr", lambda *args, **kwargs: pytest.fail("merged without panel approval")
    )

    with pytest.raises(AgentLoopError, match="missing required input from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert "**Review status: Incomplete**" in runner.comments[-1]
    assert any("Google Gemini review" in comment for comment in runner.comments)
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


# --- Issue #871: a narration-only reviewer never blocks the healthy one ------

_NARRATION_ONLY_REVIEW = (
    "I have launched the test command in the background and will wait for it "
    "to complete.\nterminating 1 background task(s) on exit"
)


def test_parallel_plan_review_settles_healthy_reviewer_when_peer_returns_narration(tmp_path):
    runner = FakeRunner(
        claude_outputs=[_initial_plan()],
        codex_outputs=[_NARRATION_ONLY_REVIEW] * 4,
        gemini_outputs=[
            structured_plan_review(
                summary="Gemini plan review complete.", reviewer="Google Gemini"
            )
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        review_parallel=True,
        agent_max_retries=0,
        agent_retry_backoff_seconds=0,
    )

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: structured_plan_review(
            state="blocking",
            summary="Plan review incomplete: the test command was terminated.",
            blocking_plan_issues=["Plan review incomplete: the test command was terminated."],
        ),
    ):
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # The healthy reviewer's settled turn is kept -- privately, because the
    # refused reviewer is retried and must not read it (#1025).
    assert not any("Gemini plan review complete." in comment for comment in runner.comments)
    spooled = [path.read_text(encoding="utf-8") for path in _spool_files(config)]
    assert any("Gemini plan review complete." in text for text in spooled)
    # No publication checkpoint or fabricated verdict for the refused reviewer.
    assert not any("Plan review incomplete" in comment for comment in runner.comments)


# ---------------------------------------------------------------------------
# #905 (from #841): staged planning resume and reviewer failure


def _staged_parallel_config(tmp_path, **overrides):
    values = {
        "reviewer": ("codex", "gemini"),
        "review_parallel": True,
        "plan_review_policy": "primary-then-panel",
        "primary_plan_reviewer": "codex",
        "max_rounds": 6,
    }
    values.update(overrides)
    return make_config(tmp_path, **values)


def test_staged_planning_reviewer_failure_stays_required(tmp_path):
    """`reviewer-failure-remains-required`: no approval is waived."""
    from agent_loop_helpers import structured_v1_plan_state

    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[("codex exploded", 1)],
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ],
    )
    config = _staged_parallel_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="Codex"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    # The primary is the only selected reviewer in the primary phase, so its
    # failure stops the round without ever approving the plan.
    assert not any(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands)
    assert not any("plan approved" in comment.lower() for comment in runner.comments)


def test_staged_planning_resume_after_publication_keeps_the_panel_finding(tmp_path):
    """`resume-no-duplicate-calls`: a staged checkpoint is not a reconciliation.

    Staged planning writes a `scheduler-prelaunch` summary every round and a
    `plan-phase-advance` summary before a reviewer-only round.  Neither may be
    read as reviewer reconciliation, or a published blocking panel review would
    resume without its numbered item, owner, and durable obligation.
    """
    from agent_loop_helpers import structured_v1_plan_state
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    fresh = structured_v1_plan_state()
    base = orchestrator.AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch_payload = {
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
        json.dumps(patch_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner = FakeRunner(
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
    )
    config = _staged_parallel_config(tmp_path)
    real_post = orchestrator.post_issue_comment

    def interrupt_before_reconciliation(*args, **kwargs):
        if "Plan review round 2 reconciliation" in kwargs["body"]:
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    # Round 1 settles the primary; round 2 is the reviewer-only panel round.
    # The panel publishes its blocking review and the run is then interrupted
    # at the reconciliation boundary.
    with patch.object(
        orchestrator, "post_issue_comment", side_effect=interrupt_before_reconciliation
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    def plan_records():
        records = []
        for comment in runner.issue_comments:
            match = orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])
            if match is None:
                continue
            metadata = orchestrator._decode_round_metadata(match["payload"])
            if metadata.flow == "plan":
                records.append(metadata)
        return records

    interrupted = plan_records()
    # The panel round holds its staged checkpoints and the publication record,
    # but no reconciliation record: the round is genuinely unsettled.
    assert [record.phase for record in interrupted if record.round_number == 2] == [
        "plan-phase-advance",
        "scheduler-prelaunch",
        "publication",
    ]
    assert not any(
        record.new_items for record in interrupted if record.round_number == 2
    )

    gemini_calls_before = sum(
        1 for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]
    )
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed = plan_records()
    reconciliation = [
        record
        for record in resumed
        if record.round_number == 2 and record.phase == "reconciliation"
    ]
    assert len(reconciliation) == 1
    # The published panel finding keeps its ID, its owner, and its obligation.
    items = reconciliation[0].new_items
    assert [item.item_id for item in items] == ["item-1"]
    assert items[0].reviewer == "Gemini"
    assert items[0].status == "blocking"
    # The settled panel reviewer is not re-invoked for the resumed round: its
    # next call is the round-3 remediation turn.
    gemini_rounds = [
        record.round_number
        for record in resumed
        if record.role == "reviewer" and record.agent == "Gemini"
    ]
    assert gemini_rounds == [2, 3]
    assert (
        sum(1 for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
        == gemini_calls_before + 1
    )
    # The planner is given the reconstructed obligation, and the round-3
    # remediation reviews disposition it rather than minting a duplicate.
    planner_prompts = [
        "\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]
    ]
    assert "item-1" in planner_prompts[-1]
    assert not any(
        item.item_id == "item-1"
        for record in resumed
        if record.round_number == 3
        for item in record.new_items
    )


def test_staged_planning_resume_does_not_re_invoke_a_settled_reviewer(tmp_path):
    """`resume-no-duplicate-calls`: settled work is reconstructed, not redone."""
    from agent_loop_helpers import structured_v1_plan_state

    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _staged_parallel_config(tmp_path, max_rounds=1)

    # Round 1 settles the primary, then the run stops on the round budget while
    # the reviewer-only panel advance is still pending.
    with pytest.raises(AgentLoopError, match="phase advance was still pending"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    commands_before = len(runner.commands)
    runner.gemini_outputs.append(
        structured_plan_review(state="approved", reviewer="Google Gemini")
    )
    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_parallel_config(tmp_path, max_rounds=6),
            plan_first=True,
        )
        == 0
    )

    resumed_commands = runner.commands[commands_before:]
    # The primary holds a qualifying exact-key approval, so it is carried; only
    # the outstanding secondary is invoked, and no planner turn is fabricated.
    assert not any(cmd[:1] == ["codex"] for cmd, _cwd in resumed_commands)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in resumed_commands)
    assert len([cmd for cmd, _cwd in resumed_commands if cmd[:1] == ["gemini"]]) == 1


# --- #1142: the refusal names the comments to delete and why replay is unavailable ---


def _approving_pr_runner():
    markers = ("Codex approves independently.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        codex_outputs=[
            structured_pr_review(summary=markers[0]),
            structured_pr_review(summary="Codex approves on the rerun."),
        ],
        gemini_outputs=[
            structured_pr_review(summary=markers[1], reviewer="Google Gemini"),
            structured_pr_review(summary="Gemini approves on the rerun.", reviewer="Google Gemini"),
        ],
    )
    runner.serve_rest_issue_comments = True
    return runner, markers


def _refusal_message(runner, config, *, review_parallel=True):
    with pytest.raises(orchestrator.PartialReviewRoundError) as excinfo:
        run_pr_loop(
            runner, pr_number=77,
            config=dataclasses.replace(config, review_parallel=review_parallel),
        )
    return str(excinfo.value)


def _listed_comment_ids(message):
    return [int(value) for value in re.findall(r"comment (\d+) https://", message)]


def _delete_listed_comments(runner, ids, *, surface):
    """Delete exactly the advertised ids from every fake comment projection."""
    from agent_loop_helpers import _strip_round_metadata

    store = runner.pr_payload["comments"] if surface == "pr" else runner.issue_comments
    doomed = {comment_id - 10_000 - 1 for comment_id in ids}
    deleted = [comment["body"] for index, comment in enumerate(store) if index in doomed]
    kept = [comment for index, comment in enumerate(store) if index not in doomed]
    store[:] = kept
    for body in deleted:
        stripped = _strip_round_metadata(body)
        if stripped in runner.comments:
            runner.comments.remove(stripped)


@pytest.mark.parametrize("review_parallel", [True, False])
def test_pr_partial_round_refusal_lists_comments_and_recovers(tmp_path, review_parallel):
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    message = _refusal_message(runner, config, review_parallel=review_parallel)

    assert len(runner.reviewer_launches) == launches_before
    assert "No review spool for this round exists on this host" in message
    assert "Rerun from the host" not in message
    assert "Delete these 1 comments so the whole round runs again independently" in message
    assert "need not be deleted" in message
    ids = _listed_comment_ids(message)
    assert len(ids) == 1

    _delete_listed_comments(runner, ids, surface="pr")
    # Deleting only the advertised ids must have removed every peer body.
    assert not any(m in body for m in markers for body in runner.comments)
    comments_before = len(runner.comments)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not runner.peer_body_visible_at_launch
    assert len(runner.reviewer_launches) - launches_before == 2

    # Effective transition equals a never-started fixture: same reviewers
    # launched and the same number of records published (verdicts plus a
    # fresh reconciliation).  The raw resume result is not compared.
    baseline, _markers = _approving_pr_runner()
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True
    )
    assert run_pr_loop(baseline, pr_number=77, config=baseline_config) == 0
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(baseline.reviewer_launches)
    assert len(runner.comments) - comments_before == len(baseline.comments)


def test_pr_partial_round_refusal_without_rest_ids_is_provisional(tmp_path):
    runner, markers = _approving_pr_runner()
    runner.serve_rest_issue_comments = False
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()

    message = _refusal_message(runner, config)

    assert "provisional list" in message
    assert "comment ids could not be read completely" in message
    assert "Delete these" not in message
    assert "need not be deleted" not in message
    assert "id could not be determined" in message


def _unpublished_reviewer_spool_file(config, markers, runner):
    published = {marker for marker in markers if _published_count(runner, marker)}
    for path in _spool_files(config):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not any(marker in payload["response"]["text"] for marker in published):
            return path
    raise AssertionError("no unpublished reviewer spool file")


def test_pr_partial_round_spool_missing_record_cause(tmp_path):
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    path = _unpublished_reviewer_spool_file(config, markers, runner)
    path.unlink()
    (path.parent / "another-reviewer.json").write_text("{}", encoding="utf-8")

    message = _refusal_message(runner, config)

    assert "holds no outcome for" in message
    assert "No review spool for this round exists" not in message
    assert "Rerun from the host" not in message


def test_pr_partial_round_spool_unusable_file_cause(tmp_path):
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    path = _unpublished_reviewer_spool_file(config, markers, runner)
    path.write_text("{not json", encoding="utf-8")

    message = _refusal_message(runner, config)

    assert f"spool file {path} exists but is unreadable or does not belong to this round" in message
    assert "holds no outcome" not in message


def test_pr_partial_round_invalid_spool_record_cause(tmp_path):
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_pr_round_between_publications(runner, config, markers)
    path = _unpublished_reviewer_spool_file(config, markers, runner)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["response"]["text"] = "not a structured PR review"
    path.write_text(json.dumps(payload), encoding="utf-8")

    message = _refusal_message(runner, config)

    assert "spooled outcome no longer validates against this run" in message


def test_replay_unavailable_cause_for_rejected_own_record_ignores_spool(tmp_path):
    spool = orchestrator.ReviewRoundSpool(
        root=tmp_path, repo="OWNER/REPO", surface="pr", number=77, round_number=1, subject="abc"
    )
    spool.store("Codex", {"text": "valid"})

    cause = orchestrator._replay_unavailable_cause(spool, "Codex", own_posted=True)

    assert "already posted but no longer accepted" in cause
    assert "fresh turn" in cause
    assert "spool" not in cause
    # Without the own-record fact the same intact file is a validation failure.
    assert "no longer validates" in orchestrator._replay_unavailable_cause(
        spool, "Codex", loaded_unreplayable=True
    )


def test_plan_partial_round_refusal_lists_posted_review(tmp_path):
    runner, markers = _plan_partial_round_runner()
    runner.serve_rest_issue_comments = True
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_plan_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["response"]["text"] = "not a structured plan review"
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(orchestrator.PartialReviewRoundError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "no longer validates against this run" in message
    assert "Rerun from the host" not in message
    assert re.search(r"comment \d+ https://", message)


# --- #1142 review: recovery-list unit coverage (boundaries, authors, attachments) ---

import random as _random
import string as _string

from coding_review_agent_loop.github import IssueComment as _IssueComment
from coding_review_agent_loop.partial_round_recovery import compute_partial_round_recovery
from coding_review_agent_loop.protocol_markers import TrustedBody as _TrustedBody
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata as _Metadata,
    _attach_round_metadata,
    _resume_pr_round as _resume_pr,
)
from coding_review_agent_loop.round_transport import prepare_round_comment as _prepare


def _big_text(seed=1):
    rnd = _random.Random(seed)
    return "".join(rnd.choice(_string.ascii_letters + " ") for _ in range(300_000))


def _meta(role="reviewer", agent="Codex", phase="publication", scheduler_phase=None, **extra):
    scheduler = {}
    if scheduler_phase is not None or phase == "scheduler-prelaunch":
        # Scheduler fields decode only as a complete, valid set.
        from coding_review_agent_loop.review_scheduling import make_contract

        scheduler = dict(
            scheduler_contract=make_contract(
                ("Codex", "Gemini"), "selective-intermediate", None
            ).as_dict(),
            scheduler_previous_sha=None, scheduler_current_sha="h",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex", "Gemini"), scheduler_paused_reviewers=(),
            scheduler_reasons=("full board",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=0,
            scheduler_phase=scheduler_phase,
        )
    return _Metadata(
        flow="pr", role=role, agent=agent, round_number=1, subject="h",
        state="approved" if role == "reviewer" else None, phase=phase,
        **scheduler, **extra,
    )


class _History:
    def __init__(self, author="bot"):
        self.comments: list[_IssueComment] = []
        self.author = author

    def add(self, body, *, author="__default__"):
        number = len(self.comments) + 1
        self.comments.append(_IssueComment(
            author=self.author if author == "__default__" else author,
            created_at=f"2026-01-01T00:00:{number:02d}Z", body=str(body),
            comment_id=1000 + number, url=f"https://example.test/c/{1000 + number}",
        ))
        return number - 1

    def add_record(self, metadata, text="body", **kwargs):
        parts = _prepare(_attach_round_metadata(text, metadata))
        indexes = [self.add(part, **kwargs) for part in parts]
        return indexes[:-1], indexes[-1]  # attachments, anchor

    def orphan(self, metadata, **kwargs):
        parts = _prepare(_attach_round_metadata("orphan", metadata))
        return [self.add(part, **kwargs) for part in parts[:-1]]

    def recover(self, *, scheduler_phase=None, fingerprint=None):
        rest = tuple(self.comments)
        return compute_partial_round_recovery(
            fingerprint=fingerprint,
            snapshot=rest, read_rest=lambda: rest, flow="pr", round_number=1, subject="h",
            scheduler_phase=scheduler_phase, reviewer_names=("Codex", "Gemini"),
            resume=lambda remaining: _resume_pr(
                remaining, head_sha="h", configured_reviewers=("codex", "gemini")
            ),
        )

    def ids(self, indexes):
        return {1000 + i + 1 for i in indexes}


def _listed(recovery):
    return {target.comment_id for target in recovery.targets}


def test_recovery_lists_overflow_anchor_with_all_attachments_and_orphans():
    history = _History()
    history.add_record(_Metadata(
        flow="pr", role="reviewer", agent="Codex", round_number=0, subject="old",
        state="approved", phase="publication",
    ))
    orphans = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(2)))
    attachments, anchor = history.add_record(_meta(canonical_reviewer_response=_big_text(1)))
    recovery = history.recover()
    assert recovery.verified, recovery.reason
    assert _listed(recovery) == history.ids([*orphans, *attachments, anchor])


def test_recovery_does_not_list_older_orphan_below_the_round_boundary():
    history = _History()
    older = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(3)))
    # A durable record from an earlier round bounds the interval.
    history.add_record(_Metadata(
        flow="pr", role="reviewer", agent="Codex", round_number=0, subject="old",
        state="approved", phase="publication",
    ))
    _attachments, anchor = history.add_record(_meta())
    recovery = history.recover()
    assert recovery.verified, recovery.reason
    assert _listed(recovery) == history.ids([anchor])
    assert not history.ids(older) & _listed(recovery)


def test_recovery_with_unknown_attachment_author_is_provisional():
    history = _History()
    history.add_record(_Metadata(
        flow="pr", role="reviewer", agent="Codex", round_number=0, subject="old",
        state="approved", phase="publication",
    ))
    orphans = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(4)), author=None)
    _attachments, anchor = history.add_record(_meta(), author=None)
    recovery = history.recover()
    assert not recovery.verified
    assert history.ids(orphans) <= {t.comment_id for t in recovery.uncertain}
    assert history.ids(orphans).isdisjoint(_listed(recovery))


def test_recovery_keeps_retained_artifact_attachments():
    history = _History()
    coder_parts, coder = history.add_record(
        _meta(role="coder", agent="Claude", phase="authoritative",
              raw_structured_coder_response=_big_text(5))
    )
    _a, anchor = history.add_record(_meta())
    recovery = history.recover()
    assert recovery.verified, recovery.reason
    assert _listed(recovery) == history.ids([anchor])
    assert not history.ids([*coder_parts, coder]) & _listed(recovery)


def test_recovery_lists_orphan_of_first_attempt_despite_later_prelaunch_checkpoint():
    """A rerun's later checkpoint must not hide the first attempt's orphan."""
    history = _History()
    history.add_record(_meta(agent="Codex", scheduler_phase="primary"))
    history.add_record(_meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch"))
    orphans = history.orphan(
        _meta(agent="Gemini", scheduler_phase="secondary-audit", canonical_reviewer_response=_big_text(6))
    )
    history.add_record(_meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch"))
    _a, verdict = history.add_record(_meta(agent="Codex", scheduler_phase="secondary-audit"))
    recovery = history.recover(scheduler_phase="secondary-audit")
    assert recovery.verified, recovery.reason
    assert history.ids(orphans) <= _listed(recovery)
    assert 1000 + verdict + 1 in _listed(recovery)
    # The primary review and both checkpoints are retained.
    assert len(recovery.targets) == len(orphans) + 1


def test_recovery_boundary_counts_durable_records_of_other_flows():
    """An other-flow durable record between an old record and the round bounds it."""
    history = _History()
    history.add_record(_Metadata(
        flow="pr", role="reviewer", agent="Codex", round_number=0, subject="old",
        state="approved", phase="publication",
    ))
    older = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(8)))
    history.add_record(_Metadata(
        flow="plan", role="reviewer", agent="Codex", round_number=3, subject="other",
        state="approved", phase="publication",
    ))
    _a, anchor = history.add_record(_meta())
    recovery = history.recover()
    assert history.ids(older).isdisjoint(_listed(recovery))
    assert _listed(recovery) == history.ids([anchor])


def test_recovery_without_a_prior_boundary_is_provisional_for_gap_attachments():
    history = _History()
    orphans = history.orphan(_meta(agent="Gemini", canonical_reviewer_response=_big_text(7)))
    history.add_record(_meta())
    recovery = history.recover()
    assert not recovery.verified
    assert history.ids(orphans) <= {t.comment_id for t in recovery.uncertain}


def test_recovery_reconciliation_summary_is_listed_and_prelaunch_retained():
    history = _History()
    history.add_record(_meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch"))
    _a, verdict = history.add_record(_meta())
    _b, reconciliation = history.add_record(
        _meta(role="summary", agent="Orchestrator", phase="reconciliation")
    )
    recovery = history.recover()
    assert recovery.verified, recovery.reason
    assert _listed(recovery) == history.ids([verdict, reconciliation])


def test_recovery_is_provisional_when_retained_state_would_shift():
    """A fingerprint of retained state (e.g. the panel opening) must survive deletion."""
    history = _History()
    history.add_record(_meta(role="summary", agent="Orchestrator", phase="scheduler-prelaunch"))
    history.add_record(_meta())
    stable = history.recover(fingerprint=lambda comments: sum(
        "scheduler-prelaunch" in str(c.body) or "AGENT_LOOP_META" in str(c.body) and False
        for c in comments
    ))
    assert stable.verified, stable.reason
    shifting = history.recover(fingerprint=lambda comments: len(comments))
    assert not shifting.verified
    assert "could not be confirmed" in shifting.reason


def test_recovery_requires_anchored_round_to_resume_as_the_same_round():
    history = _History()
    history.add_record(_meta(role="coder", agent="Claude", phase="authoritative"))
    history.add_record(_meta())
    # The resume stub reports a different round once the verdict is gone.
    rest = tuple(history.comments)

    class _State:
        def __init__(self, round_number):
            self.round_number = round_number
            self.reconciled = False
            self.completed_reviews = ()
            self.coder_output = "x"
            self.coder_metadata = None
            self.compact_prior_summaries = ()

    calls = []

    def resume(comments):
        calls.append(len(comments))
        return _State(1 if len(comments) == len(rest) else 0)

    recovery = compute_partial_round_recovery(
        snapshot=rest, read_rest=lambda: rest, flow="pr", round_number=1, subject="h",
        scheduler_phase=None, reviewer_names=("Codex", "Gemini"), resume=resume,
    )
    assert not recovery.verified


def test_plan_partial_round_recovers_from_the_message_alone(tmp_path):
    """Delete exactly the advertised ids, rerun: no refusal, no peer body visible."""
    runner, markers = _plan_partial_round_runner()
    runner.serve_rest_issue_comments = True
    # The rerun needs a fresh plan and fresh reviews for the deleted round.
    runner.claude_outputs.append(_initial_plan())
    runner.codex_outputs.append(structured_plan_review(summary="Codex plan approval rerun note."))
    runner.gemini_outputs.append(structured_plan_review(
        summary="Gemini independent plan approval rerun.", reviewer="Google Gemini"
    ))
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_plan_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    with pytest.raises(orchestrator.PartialReviewRoundError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    message = str(excinfo.value)
    assert len(runner.reviewer_launches) == launches_before
    assert "No review spool for this round exists on this host" in message
    assert "Delete these" in message, message
    ids = _listed_comment_ids(message)
    assert ids

    _delete_listed_comments(runner, ids, surface="issue")
    # Deleting only the advertised ids must have removed every peer body.
    assert not any(m in body for m in markers for body in runner.comments)
    runner.peer_body_visible_at_launch = False

    comments_before = len(runner.comments)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert not runner.peer_body_visible_at_launch
    assert len(runner.reviewer_launches) - launches_before == 2

    # Same effective transition as a never-started plan round.
    baseline, _markers = _plan_partial_round_runner()
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True
    )
    assert run_issue_loop(baseline, issue_number=56, config=baseline_config, plan_first=True) == 0
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(baseline.reviewer_launches)
    # The retained plan record plus the fresh records equal the baseline's.
    assert len(runner.comments) == len(baseline.comments)


def test_provisional_target_without_id_but_with_url_never_prints_none():
    from coding_review_agent_loop.partial_round_recovery import RecoveryTarget

    line = orchestrator._format_recovery_target(
        RecoveryTarget(
            label="Codex review (round 1)", comment_id=None,
            url="https://example.test/c/1", author="bot", created_at="2026-01-01T00:00:01Z",
        )
    )
    assert "comment None" not in line
    assert "id could not be determined" in line
    assert "https://example.test/c/1" in line
    with_id = orchestrator._format_recovery_target(
        RecoveryTarget(label="x", comment_id=7, url="https://example.test/c/7", author=None, created_at=None)
    )
    assert with_id == "- x: comment 7 https://example.test/c/7"


def test_recovery_keeps_known_targets_when_a_retained_reference_marker_is_undecodable():
    history = _History()
    history.add_record(_Metadata(
        flow="pr", role="reviewer", agent="Codex", round_number=0, subject="old",
        state="approved", phase="publication",
    ))
    # A retained comment carries a reference-bearing marker that cannot be decoded.
    history.add("Plan\n<!-- AGENT_EXECUTION_RECOMMENDATION: AAAA -->")
    _a, verdict = history.add_record(_meta())

    recovery = history.recover()

    assert not recovery.verified
    assert recovery.reason == "attachment references could not be fully decoded"
    assert 1000 + verdict + 1 in {t.comment_id for t in recovery.targets}
    message = str(orchestrator._partial_round_refusal(
        surface="pr", number=77, round_number=1, reviewer_name="Gemini",
        public_peers=["Codex"], recovery=lambda: recovery,
    ))
    assert "provisional list" in message
    assert "attachment references could not be fully decoded" in message
    assert "Delete these" not in message


def test_reconciled_round_with_unavailable_reviewer_recovers_from_the_message(tmp_path):
    """#1124 shape: peer verdict and reconciliation public, one reviewer's verdict gone."""
    runner, markers = _approving_pr_runner()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    posted = runner.pr_payload["comments"]
    assert len(posted) == 3  # two verdicts and a reconciliation
    # The unavailable reviewer's verdict is lost (second comment, REST id 10002).
    gemini_id = next(
        10_000 + index for index, comment in enumerate(posted, start=1)
        if markers[1] in comment["body"]
    )
    _delete_listed_comments(runner, [gemini_id], surface="pr")
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    message = _refusal_message(runner, config)

    assert len(runner.reviewer_launches) == launches_before
    assert "Rerun from the host" not in message
    assert "No review spool for this round exists on this host" in message
    assert "Delete these 2 comments" in message
    assert "reconciliation" in message
    ids = _listed_comment_ids(message)
    assert len(ids) == 2
    _delete_listed_comments(runner, ids, surface="pr")
    assert runner.pr_payload["comments"] == []
    assert not any(m in body for m in markers for body in runner.comments)
    runner.peer_body_visible_at_launch = False

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not runner.peer_body_visible_at_launch
    baseline, _ = _approving_pr_runner()
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True
    )
    assert run_pr_loop(baseline, pr_number=77, config=baseline_config) == 0
    # Same reviewers launched and the same records (verdicts plus a fresh
    # reconciliation) posted as a never-started round.
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(baseline.reviewer_launches)
    assert len(runner.comments) == len(baseline.comments)
    assert any("reconciliation" in body for body in runner.comments)


def _overflow_pr_runner(seeds, slow_reviewer=None):
    """A PR runner whose Codex reviews overflow into attachment comments."""
    from coding_review_agent_loop.round_transport import attachment_keys  # noqa: F401

    rnd_text = []
    for seed in seeds:
        rnd = _random.Random(seed)
        rnd_text.append("".join(rnd.choice(_string.ascii_letters + " ") for _ in range(45_000)))
    markers = ("Codex approves independently.", "Gemini approves independently.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        slow_reviewer=slow_reviewer,
        codex_outputs=[
            structured_pr_review(summary=f"Codex approves independently. {text}") for text in rnd_text
        ],
        gemini_outputs=[
            structured_pr_review(summary=markers[1], reviewer="Google Gemini"),
            structured_pr_review(summary="Gemini approves on the rerun.", reviewer="Google Gemini"),
        ],
    )
    runner.serve_rest_issue_comments = True
    return runner, markers


def test_overflow_peer_verdict_attachments_are_listed_and_recovered(tmp_path):
    """A peer verdict published as an anchor plus attachments is listed whole."""
    from coding_review_agent_loop.round_transport import attachment_keys

    runner, markers = _overflow_pr_runner((11, 12))
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    posted = runner.pr_payload["comments"]
    attachment_ids = [
        10_000 + i for i, c in enumerate(posted, start=1) if attachment_keys(c["body"])
    ]
    assert len(attachment_ids) == 2
    gemini_id = next(10_000 + i for i, c in enumerate(posted, start=1) if markers[1] in c["body"])
    _delete_listed_comments(runner, [gemini_id], surface="pr")
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    message = _refusal_message(runner, config)

    assert len(runner.reviewer_launches) == launches_before
    ids = _listed_comment_ids(message)
    # Anchor, both attachments and the reconciliation are all advertised.
    assert len(ids) == 4, message
    assert "Delete these 4 comments" in message
    _delete_listed_comments(runner, ids, surface="pr")
    # No comment carrying the peer's response (anchor or attachment) remains.
    assert runner.pr_payload["comments"] == []
    runner.peer_body_visible_at_launch = False

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not runner.peer_body_visible_at_launch
    baseline, _ = _overflow_pr_runner((12,))
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True
    )
    assert run_pr_loop(baseline, pr_number=77, config=baseline_config) == 0
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(baseline.reviewer_launches)
    assert len(runner.pr_payload["comments"]) == len(baseline.pr_payload["comments"])


def test_orphan_attachments_before_a_lost_anchor_are_listed_and_recovered(tmp_path):
    """Interruption between a peer's attachments and its anchor leaves orphans."""
    import coding_review_agent_loop.github as github_module
    from coding_review_agent_loop.round_transport import attachment_keys

    runner, markers = _overflow_pr_runner((11, 12), slow_reviewer="codex")
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    real_post = github_module._post_comment_body
    state = {"marked": 0}

    def interrupting_post(*args, **kwargs):
        body = str(kwargs["body"])
        if any(marker in body for marker in markers):
            state["marked"] += 1
            if state["marked"] == 2:  # Codex's anchor, after its attachments landed
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(github_module, "_post_comment_body", side_effect=interrupting_post):
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    posted = runner.pr_payload["comments"]
    orphans = [10_000 + i for i, c in enumerate(posted, start=1) if attachment_keys(c["body"])]
    assert len(orphans) == 2 and len(posted) == 3  # Gemini's verdict plus two orphans
    for path in _spool_files(config):
        path.unlink()
    launches_before = len(runner.reviewer_launches)

    message = _refusal_message(runner, config)

    assert len(runner.reviewer_launches) == launches_before
    ids = _listed_comment_ids(message)
    assert set(orphans) <= set(ids) and len(ids) == 3, message
    assert "Delete these 3 comments" in message
    _delete_listed_comments(runner, ids, surface="pr")
    assert runner.pr_payload["comments"] == []  # no comment carries any peer response
    runner.peer_body_visible_at_launch = False

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not runner.peer_body_visible_at_launch
    baseline, _ = _overflow_pr_runner((12,))
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True
    )
    assert run_pr_loop(baseline, pr_number=77, config=baseline_config) == 0
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(baseline.reviewer_launches)
    assert len(runner.pr_payload["comments"]) == len(baseline.pr_payload["comments"])


# --- #1142 review: real-loop deletion-and-rerun for the scheduler, plan and panel paths ---


def _comment_records(runner, *, surface, flow):
    """Round metadata of ``flow``, decoded the way resume does (spills assembled)."""
    from coding_review_agent_loop.github import IssueComment
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    store = runner.pr_payload["comments"] if surface == "pr" else runner.issue_comments
    comments = [
        IssueComment(author="bot", created_at=f"2026-01-01T00:00:{i:02d}Z", body=c["body"])
        for i, c in enumerate(store)
    ]
    return [record.metadata for record in _extract_round_metadata_records(comments, flow=flow)]


def _transition(records, *, round_number):
    """Round, phase and record kinds a rerun published (the effective transition)."""
    return sorted(
        (r.role, r.agent, r.phase, r.scheduler_phase)
        for r in records
        if r.round_number == round_number
    )


def _selective_followup_runner():
    markers = ("Codex approves after the fix.", "Gemini approves after the fix.")
    runner = _PartialPublicationProbeRunner(
        round_markers=markers,
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Codex found a blocker.", blocking_items=["Persist this item."]
            ),
            structured_pr_review(
                summary=markers[0],
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(
                summary="Codex rerun approval.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves round one.", reviewer="Google Gemini"),
            structured_pr_review(
                summary=markers[1], reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(
                summary="Gemini rerun approval.", reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[structured_coder_followup(
            summary="Fixed the persisted item.", addressed_items=["item-1"]
        )],
    )
    runner.serve_rest_issue_comments = True
    return runner, markers


def test_scheduler_enabled_round_recovers_with_retained_prelaunch_and_coder(tmp_path):
    """A coder-anchored scheduler round: prelaunch and coder retained, rerun reconciles."""
    runner, markers = _selective_followup_runner()
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True,
        pr_review_policy="selective-intermediate", max_rounds=4,
    )
    _interrupt_pr_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()
    retained = list(runner.pr_payload["comments"])
    launches_before = len(runner.reviewer_launches)

    message = _refusal_message(runner, config)

    assert len(runner.reviewer_launches) == launches_before
    assert "Rerun from the host" not in message
    ids = _listed_comment_ids(message)
    assert ids and "Delete these" in message, message
    # The prelaunch summaries and the coder record are never advertised.
    doomed = {i - 10_001 for i in ids}
    for index, comment in enumerate(retained):
        match = orchestrator.ROUND_RESUME_MARKER_RE.search(comment["body"])
        if match is None:
            continue
        metadata = orchestrator._decode_round_metadata(match["payload"])
        if index in doomed:
            assert metadata.role != "coder" and metadata.phase != "scheduler-prelaunch"
    _delete_listed_comments(runner, ids, surface="pr")
    assert not any(m in body for m in markers for body in runner.comments)
    assert any(
        r.phase == "scheduler-prelaunch" and r.round_number == 2
        for r in _comment_records(runner, surface="pr", flow="pr")
    )
    assert any(
        r.role == "coder" for r in _comment_records(runner, surface="pr", flow="pr")
    )
    runner.peer_body_visible_at_launch = False

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert not runner.peer_body_visible_at_launch
    after = _comment_records(runner, surface="pr", flow="pr")
    # The rerun stayed in round 2 and posted a fresh reconciliation.
    assert any(r.role == "summary" and r.phase == "reconciliation" and r.round_number == 2 for r in after)
    assert launches_before + 2 <= len(runner.reviewer_launches)
    assert not any(m in body for m in markers for body in runner.comments)

    baseline, _ = _selective_followup_runner()
    baseline_config = make_config(
        tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True,
        pr_review_policy="selective-intermediate", max_rounds=4,
    )
    assert run_pr_loop(baseline, pr_number=77, config=baseline_config) == 0
    def effective(records):
        # Checkpoints are retained history (a rerun may add one); compare the
        # records the round itself publishes: coder, verdicts, reconciliation.
        return [
            entry for entry in _transition(records, round_number=2)
            if entry[2] != "scheduler-prelaunch"
        ]

    assert effective(after) == effective(_comment_records(baseline, surface="pr", flow="pr"))
    assert sorted(runner.reviewer_launches[launches_before:]) == sorted(
        baseline.reviewer_launches[2:]
    )


def test_plan_round_with_spilled_coder_artifact_recovers_and_keeps_its_attachments(tmp_path):
    """The coder record's attachments are referenced by a retained comment: never listed."""
    from coding_review_agent_loop.round_transport import attachment_keys

    def build():
        runner, markers = _plan_partial_round_runner()
        runner.serve_rest_issue_comments = True
        runner.claude_outputs[:] = [structured_plan_state(
            state="blocking", summary="Initial plan.", plan_steps=["Make the change.", _big_text(21)],
        )]
        runner.codex_outputs.append(structured_plan_review(summary="Codex rerun note."))
        runner.gemini_outputs.append(structured_plan_review(
            summary="Gemini rerun note.", reviewer="Google Gemini"
        ))
        return runner, markers

    runner, markers = build()
    config = make_config(tmp_path, reviewer=("codex", "gemini"), review_parallel=True)
    _interrupt_plan_round_between_publications(runner, config, markers)
    for path in _spool_files(config):
        path.unlink()
    spilled = [c["body"] for c in runner.issue_comments if attachment_keys(c["body"])]
    assert spilled, "the coder artifact did not spill into attachments"
    launches_before = len(runner.reviewer_launches)

    with pytest.raises(orchestrator.PartialReviewRoundError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    message = str(excinfo.value)

    assert len(runner.reviewer_launches) == launches_before
    ids = _listed_comment_ids(message)
    assert ids and "Delete these" in message, message
    _delete_listed_comments(runner, ids, surface="issue")
    # Every attachment of the retained coder artifact survives.
    assert [c["body"] for c in runner.issue_comments if attachment_keys(c["body"])] == spilled
    assert not any(m in body for m in markers for body in runner.comments)
    runner.peer_body_visible_at_launch = False

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert not runner.peer_body_visible_at_launch
    assert sorted(runner.reviewer_launches[launches_before:]) == ["codex", "gemini"]
    after = _comment_records(runner, surface="issue", flow="plan")
    assert {r.round_number for r in after} == {1}
    assert any(r.role == "summary" and r.phase == "reconciliation" for r in after)
    baseline, _ = build()
    assert run_issue_loop(
        baseline, issue_number=56,
        config=make_config(tmp_path / "baseline", reviewer=("codex", "gemini"), review_parallel=True),
        plan_first=True,
    ) == 0
    assert _transition(after, round_number=1) == _transition(
        _comment_records(baseline, surface="issue", flow="plan"), round_number=1
    )


def test_snapshot_cover_requires_sidecars_and_multiplicity():
    from coding_review_agent_loop import partial_round_recovery as prr

    def comment(body, n):
        return _IssueComment(
            author="bot", body=body, created_at="2026-01-01T00:00:00Z",
            url=f"https://x/{n}", comment_id=n,
        )

    plain = comment("no marker", 1)
    sidecar_body = "<!-- agent-loop round transport attachment -->"
    with patch.object(
        prr, "is_round_transport_sidecar", lambda body: body == sidecar_body
    ):
        side = comment(sidecar_body, 2)
        assert prr._snapshot_covered([plain, side], [plain, side])
        # A sidecar missing from REST must not pass.
        assert not prr._snapshot_covered([plain, side], [plain])
        # Duplicates in the snapshot need duplicates in REST.
        dup = comment(sidecar_body, 3)
        assert not prr._snapshot_covered([side, dup], [side])
        assert prr._snapshot_covered([side, dup], [side, dup])


def test_recovery_is_provisional_when_rest_history_drops_a_sidecar():
    history = _History()
    attachments, anchor = history.add_record(_meta(canonical_reviewer_response=_big_text(7)))
    assert attachments
    snapshot = tuple(history.comments)
    rest = tuple(c for i, c in enumerate(snapshot) if i not in attachments)
    recovery = compute_partial_round_recovery(
        snapshot=snapshot, read_rest=lambda: rest, flow="pr", round_number=1, subject="h",
        scheduler_phase=None, reviewer_names=("Codex", "Gemini"),
        resume=lambda remaining: _resume_pr(
            remaining, head_sha="h", configured_reviewers=("codex", "gemini")
        ),
    )
    assert not recovery.verified
    assert recovery.reason == "comment ids could not be read completely"
    # Control: the complete REST history verifies and lists the sidecars.
    assert history.recover().verified
