"""Plan-first loop tests: plan validation and repair (#1543)."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError
import coding_review_agent_loop.pr_loop as pr_loop
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import _PlanDiagnosticRunner, _TRANSIENT_REPAIR_SUGGESTION
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_state,
    structured_v1_plan_state,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_exhausted_plan_validation_persists_from_the_planning_orchestration_path(
    tmp_path, monkeypatch
):
    invalid_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    invalid_payload.pop("architecture_impact")
    invalid_candidate = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [invalid_candidate]
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *args, **kwargs: (None, None, []),
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.plan_validation_exhaustion is not None
    assert len(runner.diagnostic_posts) == 1
    assert "not persisted" not in str(error.value)


@pytest.mark.parametrize(
    ("missing_contract", "agent_max_retries", "expected_planner_calls"),
    [
        ("execution", 0, 1),
        ("matrix", 0, 2),
        ("execution", 1, 2),
        ("matrix", 1, 2),
        ("matrix-malformed", 0, 2),
        ("matrix-malformed", 1, 2),
    ],
)
def test_exhausted_fresh_contract_integrity_persists_validation_diagnostic(
    tmp_path, missing_contract, agent_max_retries, expected_planner_calls
):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    if missing_contract == "execution":
        payload.pop("execution_recommendation")
        expected_diagnostic = "execution_recommendation"
    elif missing_contract == "matrix":
        payload.pop("risk_test_matrix")
        payload.pop("risk_test_matrix_changes")
        payload.pop("risk_test_matrix_contract_version")
        expected_diagnostic = "risk_test_matrix"
    else:
        payload["risk_test_matrix"].pop("important_exclusions")
        expected_diagnostic = "important_exclusions"
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate] * expected_planner_calls
    config = make_config(
        tmp_path,
        agent_max_retries=agent_max_retries,
        execution_strategy_contract_required=True,
    )

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert expected_diagnostic in exhaustion.diagnostic
    assert exhaustion.candidate_digest == hashlib.sha256(candidate.encode()).hexdigest()
    assert len(runner.diagnostic_posts) == 1
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == expected_planner_calls
    assert error.value.failure_category == "fresh-contract-integrity"


def test_invalid_terminal_plan_repair_replaces_persisted_candidate_provenance(
    tmp_path, monkeypatch
):
    source_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    # An unrelated repairable defect: a sole missing assessment is refused
    # without repair, because repair may not supply one (#925).
    source_payload["unexpected_key"] = True
    source_candidate = (
        json.dumps(source_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    repair_payload.pop("summary")
    repair_candidate = (
        json.dumps(repair_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_attempt = SimpleNamespace(
        backend="gemini",
        model="repair-model",
        prompt="",
        output=repair_candidate,
        returncode=0,
        outcome="invalid_output",
        diagnostic="combined backend text that must not be persisted",
        log_path=None,
        fallback_planned=False,
    )
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *args, **kwargs: (repair_candidate, None, [repair_attempt]),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [source_candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert exhaustion.candidate_text == repair_candidate
    assert exhaustion.candidate_digest == hashlib.sha256(repair_candidate.encode()).hexdigest()
    assert "plan_state is missing required field(s): summary" in exhaustion.diagnostic
    assert "combined backend text" not in exhaustion.diagnostic
    assert len(runner.diagnostic_posts) == 1


@pytest.mark.parametrize(
    ("outcome", "returncode", "expected_category"),
    [("timeout", None, "timeout"), ("nonzero_exit", 1, "repair-provider-failure")],
)
def test_terminal_plan_repair_provider_failure_does_not_persist_stale_diagnostic(
    tmp_path, monkeypatch, outcome, returncode, expected_category
):
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    # An unrelated repairable defect: a sole missing assessment is refused
    # without repair, because repair may not supply one (#925).
    payload["unexpected_key"] = True
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_attempt = SimpleNamespace(
        backend="gemini",
        model="repair-model",
        prompt="",
        output="",
        returncode=returncode,
        outcome=outcome,
        diagnostic="repair transport or provider failed",
        log_path=None,
        fallback_planned=False,
    )
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *args, **kwargs: (None, None, [repair_attempt]),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    assert error.value.failure_category == expected_category
    assert error.value.plan_validation_exhaustion is None
    assert runner.diagnostic_posts == []


def _repair_attempt(outcome, *, output="", returncode=0, model="Gemini 3.8 Flash (Medium)"):
    return SimpleNamespace(
        backend="antigravity",
        model=model,
        prompt="",
        output=output,
        returncode=returncode,
        outcome=outcome,
        diagnostic=(
            "model-access validation errors" if outcome == "transient_provider_error" else ""
        ),
        log_path=None,
        fallback_planned=False,
    )


@pytest.mark.parametrize(
    ("chain", "expected_category"),
    [
        (["transient_provider_error", "transient_provider_error"], "repair-provider-failure"),
        (
            ["invalid_output", "transient_provider_error", "transient_provider_error"],
            "repair-provider-failure",
        ),
        (["transient_provider_error", "timeout"], "timeout"),
    ],
)
def test_terminal_transient_plan_repair_is_resumable_provider_failure(
    tmp_path, monkeypatch, chain, expected_category
):
    """Issue #846: a chain ending in a transient agy failure is not deterministic."""
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    # An unrelated repairable defect: a sole missing assessment is refused
    # without repair, because repair may not supply one (#925).
    payload["unexpected_key"] = True
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    attempts = [
        _repair_attempt(
            outcome,
            output=candidate if outcome == "invalid_output" else "",
            returncode=None if outcome == "timeout" else 0,
        )
        for outcome in chain
    ]
    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair",
        lambda *args, **kwargs: (None, None, attempts),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    assert error.value.failure_category == expected_category
    assert error.value.plan_validation_exhaustion is None
    assert runner.diagnostic_posts == []
    if expected_category == "repair-provider-failure":
        assert _TRANSIENT_REPAIR_SUGGESTION in str(error.value)
    else:
        assert _TRANSIENT_REPAIR_SUGGESTION not in str(error.value)


# --- Issue #871: a reviewer turn with no review is never repaired into one ---

_NARRATION_ONLY_PLAN_REVIEW = (
    "I have launched the test command for tests/test_test_runtime.py in the "
    "background and will wait for it to complete.\n"
    "root agent idle; waiting up to 5s for 1 background task(s)\n"
    "terminating 1 background task(s) on exit"
)


def _fabricated_plan_review():
    return structured_plan_review(
        state="blocking",
        summary="Plan review incomplete: the test command was terminated.",
        blocking_plan_issues=["Plan review incomplete: the test command was terminated."],
        reviewer="OpenAI Codex",
    )


def test_narration_only_plan_reviewer_is_unavailable_and_posts_no_verdict(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
    )
    repair_calls = []

    def fake_repair(raw, gemini_cmd, **kwargs):
        repair_calls.append(raw)
        return _fabricated_plan_review()

    config = make_config(tmp_path, agent_max_retries=1, agent_retry_backoff_seconds=0)
    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair):
        with pytest.raises(AgentLoopError, match="review_substance_integrity") as excinfo:
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert "agent-unavailable" in str(excinfo.value)
    # The refusal happens before any repair backend call.
    assert repair_calls == []
    # No fabricated verdict is posted and no ledger item is numbered.
    assert not any("Plan review incomplete" in body for body in runner.comments)
    assert not any("[item-1]" in body for body in runner.comments)
    # The reviewer is retried within the configured policy before the run stops.
    codex_turns = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_turns) >= 2


def test_empty_plan_reviewer_response_never_reaches_repair(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=["", "", "", ""],
    )
    repair_calls = []

    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)
    with patch(
        "coding_review_agent_loop.validated_agent.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: repair_calls.append(raw) or _fabricated_plan_review(),
    ):
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert repair_calls == []
    assert not any("Plan review incomplete" in body for body in runner.comments)


def test_refused_plan_reviewer_leaves_nothing_for_a_resume_to_replay(tmp_path):
    """A refused reviewer must not leave a checkpoint a rerun could replay."""
    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)

    def run_once():
        runner = _FakeRunner(
            claude_outputs=[structured_v1_plan_state()],
            codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
        )
        with patch(
            "coding_review_agent_loop.validated_agent.attempt_repair",
            lambda raw, gemini_cmd, **kwargs: _fabricated_plan_review(),
        ):
            with pytest.raises(AgentLoopError):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return runner

    first = run_once()
    second = run_once()
    for runner in (first, second):
        assert not any("Plan review incomplete" in body for body in runner.comments)
        assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
