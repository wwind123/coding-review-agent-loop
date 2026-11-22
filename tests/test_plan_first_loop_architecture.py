"""Plan-first loop tests: architecture-impact contract in planning rounds (#1543)."""

import coding_review_agent_loop.plan_execution as plan_execution
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.round_metadata import (
    _decode_round_metadata,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    _DEG_CORROBORATED,
    _DEG_UNCORROBORATED,
    _deg_with,
    _plan_round_records,
    _repaired_acknowledgement_fixtures,
    _staged_plan_config,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_state,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def test_plan_first_round_metadata_carries_a_corroborated_plan_state_record(tmp_path):
    plan = _deg_with(structured_plan_state(summary="Add schema helpers."), _DEG_CORROBORATED)
    runner = _FakeRunner(
        claude_outputs=[plan],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only", max_rounds=1)
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except (AgentLoopError, StopIteration, IndexError):
        pass  # only the posted planning round matters here
    bodies = [comment["body"] for comment in runner.issue_comments]
    plan_rounds = [c for c in bodies if "AGENT_LOOP_META" in c and "### Parse degradations" in c]
    assert plan_rounds, "the planning round must surface its degradation record"
    decoded = _decode_round_metadata(
        re.search(r"AGENT_LOOP_META: ([A-Za-z0-9+/=_-]+)", plan_rounds[0]).group(1)
    )
    assert [r.outcome for r in decoded.architecture_impact_degradations] == ["normalized-to-changed"]
    assert decoded.architecture_impact["status"] == "changed"


# --- #925 round 5: acknowledgement repairs through the real planning flow ----


def _ack_staged_run(
    tmp_path, monkeypatch, *, repaired_impact, repaired_status=None, max_rounds=None
):
    """Run staged planning whose primary approves with a degraded assessment.

    The primary's review carries an uncorroborated `modified`, so acceptance
    removes the assessment and records the degradation; it also lacks the
    signed-human acknowledgement, which the post-acceptance repair adds.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    unacknowledged = _deg_with(unacknowledged, _DEG_UNCORROBORATED)
    if repaired_status is not None:
        repaired_impact = {"status": repaired_status, "rationale": "No architectural change."}
    repaired = _deg_with(repaired, repaired_impact)
    runner = _FakeRunner(
        claude_outputs=[plan_output] * 4,
        codex_outputs=[unacknowledged] * 4,
        gemini_outputs=[panel] * 4,
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop, plan_execution), "get_issue_context", _patched)
    repair_calls = []

    def fake_repair(raw, cmd, **kwargs):
        repair_calls.append(raw)
        return repaired

    outcome = None
    repair_results = []
    real_structured_repair = validated_agent_module._run_structured_repair

    def spy_repair(*args, **kwargs):
        result = real_structured_repair(*args, **kwargs)
        repair_results.append(result)
        return result

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", spy_repair
    ):
        try:
            outcome = run_issue_loop(
                runner, issue_number=56,
                config=(
                    _staged_plan_config(tmp_path)
                    if max_rounds is None
                    else _staged_plan_config(tmp_path, max_rounds=max_rounds)
                ),
                plan_first=True,
            )
        except AgentLoopError as exc:
            outcome = exc
    codex_records = [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    _ack_staged_run.repair_results = repair_results
    return outcome, codex_records, repair_calls, requirement


_ACK_REFUSAL = {"unchanged": "must not introduce one", "none": "must be `changed` or `unchanged`"}


def _assert_refused_by(repair_results, status):
    assert repair_results
    for _text, validated, attempts in repair_results:
        assert validated is None
        assert _ACK_REFUSAL[status] in attempts[-1].diagnostic


def _ack_nonstaged_run(tmp_path, monkeypatch, *, repaired_status=None):
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    unacknowledged = _deg_with(unacknowledged, _DEG_UNCORROBORATED)
    repaired = _deg_with(
        repaired,
        None if repaired_status is None
        else {"status": repaired_status, "rationale": "No architectural change."},
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output] * 4,
        codex_outputs=[unacknowledged] * 4,
        gemini_outputs=[panel] * 4,
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop, plan_execution), "get_issue_context", _patched)
    repair_calls = []

    def fake_repair(raw, cmd, **kwargs):
        repair_calls.append(raw)
        return repaired

    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=1,
        agent_max_retries=0, agent_retry_backoff_seconds=0,
    )
    repair_results = []
    real_structured_repair = validated_agent_module._run_structured_repair

    def spy_repair(*args, **kwargs):
        result = real_structured_repair(*args, **kwargs)
        repair_results.append(result)
        return result

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", spy_repair
    ):
        try:
            outcome = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        except AgentLoopError as exc:
            outcome = exc
    codex_records = [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    _ack_nonstaged_run.repair_results = repair_results
    _ack_nonstaged_run.state = (runner, config, requirement)
    return outcome, codex_records, repair_calls


def _ack_plan_resume_run(tmp_path, monkeypatch, *, parallel, codex_impact, repaired_impact):
    """Interrupt a plan round at its acknowledgement repair, then resume it.

    ``parallel`` selects the early-publication path, whose record stores the
    canonical JSON response; otherwise the record is an authoritative
    rendered-prose post.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    unacknowledged = _deg_with(unacknowledged, codex_impact)
    repaired = _deg_with(repaired, repaired_impact)
    runner = _FakeRunner(
        claude_outputs=[plan_output] * 4,
        codex_outputs=[unacknowledged] * 4,
        gemini_outputs=[panel] * 4,
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop, plan_execution), "get_issue_context", _patched)
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=1,
        agent_max_retries=0, agent_retry_backoff_seconds=0, review_parallel=parallel,
    )

    def interrupt(raw, cmd, **kwargs):
        raise KeyboardInterrupt

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", interrupt):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    agents_before = [cmd[0] for cmd, _cwd in runner.commands if cmd[0] in {"claude", "codex", "gemini"}]

    carriers, pinned, repair_results, repair_calls = [], [], [], []
    real_pin = architecture_contract._pin_acknowledgement_repair
    real_structured_repair = validated_agent_module._run_structured_repair

    def spy_pin(accepted, repaired_value, **kwargs):
        carriers.append(accepted)
        result = real_pin(accepted, repaired_value, **kwargs)
        pinned.append((repaired_value, result))
        return result

    def spy_repair(*args, **kwargs):
        result = real_structured_repair(*args, **kwargs)
        repair_results.append(result)
        return result

    def fake_repair(raw, cmd, **kwargs):
        repair_calls.append(raw)
        return repaired

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((architecture_contract,), "_pin_acknowledgement_repair", spy_pin
    ), patch_object_in((validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", spy_repair):
        try:
            outcome = run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        except AgentLoopError as exc:
            outcome = exc
    agents_after = [cmd[0] for cmd, _cwd in runner.commands if cmd[0] in {"claude", "codex", "gemini"}]
    records = [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    return SimpleNamespace(
        outcome=outcome, carriers=carriers, pinned=pinned, repair_results=repair_results,
        repair_calls=repair_calls, records=records,
        resumed_without_reinvocation=agents_before == agents_after,
    )


def _assert_degraded_carrier(carrier):
    assert carrier.architecture_impact is None
    assert [r.outcome for r in carrier.architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]


