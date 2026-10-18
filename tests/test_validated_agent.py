"""Validated-agent tests: architecture-contract seam and completion recovery (#1543)."""

import hashlib
import json
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.protocol import (
    validate_structured_plan_state,
    validate_structured_issue_implementation,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.response_validation as response_validation
from issue_loop_test_helpers import (
    _DEG_CORROBORATED,
    _DEG_UNCORROBORATED,
    _deg_task_text,
    _deg_with,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_revision,
    structured_plan_state,
    structured_issue_implementation,
)
from coding_review_agent_loop.errors import AgentInvocationError as _DegInvocationError  # noqa: E402
from coding_review_agent_loop.protocol import (
    validate_structured_plan_revision as _deg_validate_plan_revision,
)
from coding_review_agent_loop.validated_agent import CompletionRecoveryPolicy as _r2_CompletionRecoveryPolicy, _attempt_claude_completion_recovery as _r2_attempt_recovery, _new_usage_context as _r2_new_usage_context
from agent_loop_helpers import (
    FakeRunner as _r2_FakeRunner,
    structured_plan_review as _r2_plan_review,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _deg_issue_validators():
    """The opt-in degradable validate and its strict re-parse, as call sites build them."""
    return architecture_contract._architecture_mode_validators(
        lambda mode: lambda text: response_validation._validate_issue_implementation_response(
            text, human_requirements=(), require_architecture_impact=True,
            architecture_status_mode=mode,
        )
    )


@pytest.mark.parametrize(
    "impact,records",
    [(None, []), (_DEG_UNCORROBORATED, ["degraded-to-undetermined"])],
    ids=["omitted", "uncorroborated"],
)
def test_seam_refuses_and_retains_an_unsatisfied_response(tmp_path, impact, records):
    text = _deg_with(structured_issue_implementation(), impact)
    runner = _FakeRunner(claude_outputs=[text])
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(_DegInvocationError) as error:
        validated_agent_module._run_validated_agent(
            runner, agent="claude", config=config, prompt="Implement.",
            marker_description="structured issue_implementation result",
            **_deg_issue_validators(),
            require_architecture_impact_contract=True,
        )

    preserved = error.value.preserved_unsatisfied_response
    assert preserved is not None and preserved.text == text
    assert "issue_implementation must include architecture_impact" in preserved.diagnostic
    assert "`changed` or `unchanged`" in preserved.diagnostic
    assert [r.outcome for r in preserved.architecture_impact_degradations] == records
    assert error.value.failure_category == "deterministic"
    assert "architecture_impact" in str(error.value)
    # The same retry budget as the former raise: one agent turn, no extras.
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1


def test_seam_accepts_a_corroborated_near_miss_with_its_record(tmp_path):
    text = _deg_with(structured_issue_implementation(), _DEG_CORROBORATED)
    runner = _FakeRunner(claude_outputs=[text])
    response = validated_agent_module._run_validated_agent(
        runner, agent="claude", config=make_config(tmp_path, agent_max_retries=0),
        prompt="Implement.", marker_description="structured issue_implementation result",
        **_deg_issue_validators(), require_architecture_impact_contract=True,
    )
    assert response.marker_value.architecture_impact.status == "changed"
    # The accepted text is canonical: it re-parses strictly and no longer
    # carries the near miss.
    assert '"modified"' not in response.text
    assert validate_structured_issue_implementation(
        response.text, required_architecture_impact_contract=1
    ).architecture_impact.status == "changed"
    assert [r.outcome for r in response.marker_value.architecture_impact_degradations] == [
        "normalized-to-changed"
    ]


def test_unsatisfied_blocking_task_is_not_hidden_by_the_no_pr_wrapper(tmp_path):
    result = response_validation._require_task_implementation_result(
        _deg_task_text(None, outcome="blocking"), required_architecture_impact_contract=1, architecture_status_mode="legacy"
    )
    assert not isinstance(result, architecture_contract._TerminalNoPrImplementation)
    assert architecture_contract.architecture_impact_contract_unsatisfied(result)
    satisfied = response_validation._require_task_implementation_result(
        _deg_task_text({"status": "unchanged", "rationale": "No change."}, outcome="blocking"),
        required_architecture_impact_contract=1, architecture_status_mode="legacy",
    )
    assert isinstance(satisfied, architecture_contract._TerminalNoPrImplementation)

    runner = _FakeRunner(claude_outputs=[_deg_task_text(None, outcome="blocking")])
    with pytest.raises(_DegInvocationError) as error:
        validated_agent_module._run_validated_agent(
            runner, agent="claude", config=make_config(tmp_path, agent_max_retries=0),
            prompt="Task.", marker_description="structured task_result JSON",
            validate=lambda text: response_validation._require_task_implementation_result(
                text, required_architecture_impact_contract=1, architecture_status_mode="legacy"
            ),
            require_architecture_impact_contract=True,
        )
    assert "task_result must include architecture_impact" in (
        error.value.preserved_unsatisfied_response.diagnostic
    )


def test_unknown_result_type_fails_closed(tmp_path):
    with pytest.raises(AgentLoopError, match="not an enumerated"):
        architecture_contract.architecture_impact_contract_unsatisfied(object())
    with pytest.raises(AgentLoopError, match="not an enumerated"):
        architecture_contract._architecture_result_fields(object())
    runner = _FakeRunner(claude_outputs=[structured_issue_implementation()])
    with pytest.raises(_DegInvocationError, match="not an enumerated"):
        validated_agent_module._run_validated_agent(
            runner, agent="claude", config=make_config(tmp_path, agent_max_retries=0),
            prompt="Implement.", marker_description="result",
            validate=lambda text: object(), require_architecture_impact_contract=True,
        )
    # Contract-free results pass through.
    for value in ("clarification", 77, architecture_contract._TerminalNoPrImplementation("clarification")):
        assert architecture_contract.architecture_impact_contract_unsatisfied(value) is False


@pytest.mark.parametrize("kind", ["plan_state", "plan_revision"])
def test_unsatisfied_repair_outcome_is_dispatched_as_deterministic(tmp_path, kind):
    factory = structured_plan_state if kind == "plan_state" else structured_plan_revision
    source = _deg_with(factory(), _DEG_UNCORROBORATED, unexpected_key=True)
    repaired = _deg_with(factory(), None)
    validators = architecture_contract._architecture_mode_validators(
        (lambda mode: lambda text: validate_structured_plan_state(
            text, required_architecture_impact_contract=1, architecture_status_mode=mode))
        if kind == "plan_state"
        else (lambda mode: lambda text: _deg_validate_plan_revision(
            text, required_architecture_impact_contract=1, architecture_status_mode=mode))
    )
    persisted = []
    runner = _FakeRunner(claude_outputs=[source])
    with patch.object(validated_agent_module, "attempt_repair", lambda raw, cmd, **kw: repaired):
        with pytest.raises(_DegInvocationError) as error:
            validated_agent_module._run_validated_agent(
                runner, agent="claude", config=make_config(tmp_path, agent_max_retries=0),
                prompt="Plan.", marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                **validators, use_repair=True, repair_expected_kind=kind,
                require_architecture_impact_contract=True,
                plan_validation_failure_handler=lambda exhaustion, err: persisted.append(exhaustion),
            )

    assert error.value.failure_category == "deterministic"
    assert error.value.failure_category != "repair-provider-failure"
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert exhaustion.candidate_kind == kind
    assert exhaustion.candidate_text == repaired
    assert exhaustion.candidate_digest == hashlib.sha256(repaired.encode()).hexdigest()
    assert f"{kind} must include architecture_impact" in exhaustion.diagnostic
    assert persisted == [exhaustion]
    preserved = error.value.preserved_unsatisfied_response
    # The retained candidate keeps its out-of-band record.
    assert preserved.text == repaired
    assert [r.outcome for r in preserved.architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]
    assert "undetermined" not in preserved.text


# ---------------------------------------------------------------------------
# #925 review round 2: contract retry, artifact and recovery routes, the
# acceptance boundary, per-candidate repair refusal, and carriers
# ---------------------------------------------------------------------------



def _r2_claude_prompts(runner):
    return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def _r2_run(runner, tmp_path, *, retries, **extra):
    return validated_agent_module._run_validated_agent(
        runner, agent="claude", config=make_config(tmp_path, agent_max_retries=retries),
        prompt="Implement the issue.",
        marker_description="structured issue_implementation result",
        **_deg_issue_validators(), require_architecture_impact_contract=True, **extra,
    )


@pytest.mark.parametrize("impact", [None, _DEG_UNCORROBORATED], ids=["omitted", "uncorroborated"])
def test_unsatisfied_contract_retries_once_with_a_field_naming_reprompt(tmp_path, impact):
    unsatisfied = _deg_with(structured_issue_implementation(), impact)
    valid = structured_issue_implementation()
    runner = _r2_FakeRunner(claude_outputs=[unsatisfied, valid])

    response = _r2_run(runner, tmp_path, retries=1)

    assert response.marker_value.architecture_impact.status in {"changed", "unchanged"}
    first, second = _r2_claude_prompts(runner)
    assert "Previous response not accepted: architecture_impact" not in first
    assert "Previous response not accepted: architecture_impact" in second
    for token in ("architecture_impact", "`changed`", "`unchanged`"):
        assert token in second
    section = architecture_contract._architecture_contract_retry_prompt(
        "P", "<!-- AGENT_STATE: approved --> " + "x" * 5000
    )
    assert "<!--" not in section and len(section) < 2000


def test_always_unsatisfied_contract_stops_after_the_existing_budget(tmp_path):
    unsatisfied = _deg_with(structured_issue_implementation(), None)
    runner = _r2_FakeRunner(claude_outputs=[unsatisfied, unsatisfied, unsatisfied, unsatisfied])

    with pytest.raises(_DegInvocationError) as error:
        _r2_run(runner, tmp_path, retries=2)

    assert len(_r2_claude_prompts(runner)) == 3  # agent_max_retries + 1
    assert error.value.failure_category == "deterministic"
    assert error.value.preserved_unsatisfied_response.text == unsatisfied


def test_contract_only_failure_makes_no_repair_invocation(tmp_path):
    calls = []
    unsatisfied = _deg_with(structured_issue_implementation(), None)
    runner = _r2_FakeRunner(claude_outputs=[unsatisfied])
    with patch.object(
        validated_agent_module, "attempt_repair", lambda raw, cmd, **kw: calls.append(raw)
    ):
        with pytest.raises(_DegInvocationError):
            _r2_run(runner, tmp_path, retries=0, use_repair=True,
                    repair_expected_kind="issue_implementation")
    assert calls == []


@pytest.mark.parametrize("returncode", [1, None], ids=["nonzero", "timeout"])
def test_unsatisfied_response_file_artifact_is_a_deterministic_contract_retry(tmp_path, returncode):
    artifact = _deg_with(structured_issue_implementation(), None)
    valid = structured_issue_implementation()
    runner = _r2_FakeRunner(
        claude_outputs=[("provider diagnostics only", returncode), valid],
        public_response_outputs=[artifact],
    )
    usage = _r2_new_usage_context(make_config(tmp_path))

    response = _r2_run(runner, tmp_path, retries=1, usage_context=usage)

    assert response.marker_value.architecture_impact is not None
    prompts = _r2_claude_prompts(runner)
    assert len(prompts) == 2
    assert "Previous response not accepted: architecture_impact" in prompts[1]
    first_record = usage.records[0]
    assert first_record.validation_status == "invalid"


@pytest.mark.parametrize("returncode", [1, None], ids=["nonzero", "timeout"])
def test_unsatisfied_artifact_exhaustion_preserves_the_artifact(tmp_path, returncode):
    artifact = _deg_with(structured_issue_implementation(), _DEG_UNCORROBORATED)
    runner = _r2_FakeRunner(
        claude_outputs=[("provider diagnostics only", returncode)],
        public_response_outputs=[artifact],
    )
    with pytest.raises(_DegInvocationError) as error:
        _r2_run(runner, tmp_path, retries=0)
    assert error.value.failure_category == "deterministic"
    preserved = error.value.preserved_unsatisfied_response
    assert preserved.text == artifact
    assert [r.outcome for r in preserved.architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]


@pytest.mark.parametrize("returncode", [1, None], ids=["nonzero", "timeout"])
def test_artifact_canonicalization_failure_is_deterministic_not_transport(tmp_path, returncode):
    # A corroborated `modified` artifact parses with a record, but its strict
    # re-parse fails the canonical comparison: the attempt keeps the
    # canonicalization diagnostic and is never classified as a timeout or
    # command failure, nor as an unsatisfied architecture contract.
    artifact = _deg_with(structured_issue_implementation(), _DEG_CORROBORATED)
    mismatched = validate_structured_issue_implementation(
        _deg_with(structured_issue_implementation(pr_number=99), dict(_DEG_CORROBORATED, status="changed"))
    )
    runner = _r2_FakeRunner(
        claude_outputs=[("provider diagnostics only", returncode)],
        public_response_outputs=[artifact],
    )
    usage = _r2_new_usage_context(make_config(tmp_path))
    validators = dict(_deg_issue_validators(), strict_revalidate=lambda _text: mismatched)
    with pytest.raises(_DegInvocationError) as error:
        validated_agent_module._run_validated_agent(
            runner, agent="claude", config=make_config(tmp_path, agent_max_retries=1),
            prompt="Implement the issue.",
            marker_description="structured issue_implementation result",
            **validators, require_architecture_impact_contract=True, usage_context=usage,
        )
    assert error.value.failure_category == "deterministic"
    reason = str(error.value).split("Reason: ", 1)[1].split(". Required marker", 1)[0]
    assert reason.startswith("Accepted-text canonicalization failed")
    assert "timed out" not in reason and "exited with" not in reason
    assert "architecture_impact" not in reason
    assert error.value.preserved_unsatisfied_response is None
    assert len(_r2_claude_prompts(runner)) == 1
    assert usage.records[0].validation_status == "invalid"


@pytest.mark.parametrize("form", ["artifact", "text"])
def test_completion_recovery_routes_an_unsatisfied_contract_to_the_ordinary_retry(tmp_path, form):
    unsatisfied = _deg_with(structured_issue_implementation(), None)
    if form == "artifact":
        runner = _r2_FakeRunner(
            claude_outputs=[("still running", 1)], public_response_outputs=[unsatisfied]
        )
    else:
        runner = _r2_FakeRunner(claude_outputs=[unsatisfied])
    validators = _deg_issue_validators()

    def refusing(text):
        parsed = validators["validate"](text)
        if architecture_contract.architecture_impact_contract_unsatisfied(parsed):
            raise architecture_contract._ArchitectureImpactContractUnsatisfied(
                architecture_contract._architecture_contract_diagnostic(parsed)
            )
        return parsed

    config = make_config(tmp_path, coder="claude")
    outcome = _r2_attempt_recovery(
        runner, config=config, completion_recovery=_r2_CompletionRecoveryPolicy(issue_number=56),
        session_id="sess-1", validate=refusing, usage_context=_r2_new_usage_context(config),
        run_id="run-1", role=None, label=None, timeout_seconds=None,
    )
    assert outcome.validated is None
    assert outcome.contract_unsatisfied is True
    assert outcome.failure_category == "deterministic"
    assert outcome.terminal_public_response is None
    assert "architecture_impact" in outcome.error
    assert runner.comments == []


def test_accepted_uncorroborated_review_text_drops_the_assessment(tmp_path):
    text = _deg_with(_r2_plan_review(), _DEG_UNCORROBORATED)
    runner = _r2_FakeRunner(codex_outputs=[text])
    validators = architecture_contract._architecture_mode_validators(
        lambda mode: lambda candidate: plan_first_loop_module._validate_plan_review_response(
            candidate, reviewer="OpenAI Codex", unresolved_items=(),
            architecture_status_mode=mode,
        )
    )
    response = validated_agent_module._run_validated_agent(
        runner, agent="codex", config=make_config(tmp_path, agent_max_retries=0),
        prompt="Review.", marker_description="plan review", **validators,
    )
    assert response.marker_value.architecture_impact is None
    (record,) = response.marker_value.architecture_impact_degradations
    assert record.outcome == "degraded-to-undetermined"
    assert "architecture_impact" not in json.loads(response.text.split("\n<!--", 1)[0])
    assert response.text.rstrip().endswith("-- OpenAI Codex")
    # The accepted text passes every strict re-parse.
    strict = plan_first_loop_module._validate_plan_review_response(
        response.text, reviewer="OpenAI Codex", unresolved_items=(),
        architecture_status_mode="strict",
    )
    assert strict.architecture_impact is None


def test_degrade_flag_without_a_strict_reparse_fails_closed(tmp_path):
    with pytest.raises(AgentLoopError, match="strict re-parse"):
        validated_agent_module._run_validated_agent(
            _r2_FakeRunner(), agent="claude", config=make_config(tmp_path),
            prompt="x", marker_description="x", validate=lambda text: text,
            degrade_architecture_impact=True,
        )


def test_nested_pr_phase_carries_pr_number_without_a_second_window(tmp_path):
    from types import SimpleNamespace

    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = _FakeRunner()
    config = make_config(tmp_path)
    context = SimpleNamespace(run_id="outer-run")
    outer = validated_agent_module._begin_run_telemetry(runner, config, context, True, issue_number=56)
    inner = validated_agent_module._begin_run_telemetry(runner, config, context, False, pr_number=99)
    assert runner.telemetry_attribution["run_id"] == "outer-run"
    assert (runner.telemetry_attribution["issue_number"], runner.telemetry_attribution["pr_number"]) == (56, 99)
    validated_agent_module._end_run_telemetry(runner, inner)
    assert runner.telemetry_attribution["pr_number"] is None
    validated_agent_module._end_run_telemetry(runner, outer)
    assert runner.telemetry_attribution is None
    kinds = [r["record"] for r in load_records(telemetry_log_path())]
    assert kinds == ["run-start", "run-end"]
