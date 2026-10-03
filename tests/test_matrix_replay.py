"""Fresh risk-matrix integrity replay and string-list normalization (#1229)."""
import hashlib
import json
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.errors import AgentInvocationError, AgentLoopError
from coding_review_agent_loop.protocol import (
    validate_structured_plan_revision,
    validate_structured_plan_state,
)
from agent_loop_helpers import FakeRunner, make_config, structured_plan_revision, structured_v1_plan_state

HEADING = "## Previous response not accepted: risk_test_matrix"
PLAN_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


def _row(**overrides):
    row = {
        "row_id": "row-a", "label": "L", "entry_path_or_mode": "m", "initial_state": "i",
        "event": "e", "expected_outcome": "o", "forbidden_side_effects": ["Must not refuse"],
        "proposed_test_level": "unit", "proposed_test_location": "tests/x.py",
        "applicability": "required", "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }
    row.update(overrides)
    return row


def _state(row=None, exclusions=None):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["risk_test_matrix"] = {
        "applicability": "applicable",
        "rows": [row or _row()],
        "important_exclusions": [] if exclusions is None else exclusions,
    }
    return json.dumps(payload) + PLAN_FOOTER


def _revision(row=None):
    payload = json.loads(structured_plan_revision().split("\n", 1)[0])
    payload.update(
        execution_strategy_contract_version=1,
        risk_test_matrix_contract_version=1,
        risk_test_matrix_changes=[],
    )
    state = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["execution_recommendation"] = state["execution_recommendation"]
    payload["execution_strategy_contract_version"] = state["execution_strategy_contract_version"]
    payload["risk_test_matrix"] = {
        "applicability": "applicable", "rows": [row or _row()], "important_exclusions": [],
    }
    return json.dumps(payload) + PLAN_FOOTER


def _required(value):
    if value is None:
        raise AgentLoopError("response is not a structured plan")
    return value


def _validate_state(text):
    return _required(validate_structured_plan_state(
        text, require_execution_strategy_contract=1, require_risk_test_matrix_contract=1
    ))


def _validate_revision(text):
    return _required(validate_structured_plan_revision(
        text, require_execution_strategy_contract=1, require_risk_test_matrix_contract=1
    ))


def _run(runner, config, texts_kind="plan_state"):
    logs: list[str] = []
    exhaustions = []
    with patch.object(orchestrator, "_run_structured_repair", wraps=orchestrator._run_structured_repair) as spy, \
            patch.object(orchestrator, "log", lambda _c, m: logs.append(m)), \
            patch("coding_review_agent_loop.validated_agent.log", lambda _c, m: logs.append(m)), \
            patch.object(orchestrator.time, "sleep", lambda *_a: None):
        try:
            response = orchestrator._run_validated_agent(
                runner,
                agent=_AGENT[0],
                config=config,
                prompt="ORIGINAL PROMPT",
                session_id=None,
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_validate_state if texts_kind == "plan_state" else _validate_revision,
                use_repair=True,
                repair_expected_kind=texts_kind,
                role="planner",
                operation_description="planning",
                require_execution_strategy_contract=True,
                require_risk_test_matrix_contract=True,
                plan_validation_failure_handler=lambda ex, err: exhaustions.append(ex),
            )
            error = None
        except AgentInvocationError as exc:
            response, error = None, exc
    return response, error, logs, spy, exhaustions


_AGENT = ["codex"]


def _runner(agent, outputs):
    _AGENT[0] = agent
    if agent == "codex":
        return FakeRunner(codex_outputs=list(outputs))
    return FakeRunner(antigravity_outputs=[(o, 0) for o in outputs])


def _cmds(runner, agent):
    binary = "agy" if agent == "antigravity" else "codex"
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == [binary]]


def _config(tmp_path, agent, retries):
    extra = {"antigravity_models": ("ModelA", "ModelB", "ModelC")} if agent == "antigravity" else {}
    return make_config(
        tmp_path, agent_max_retries=retries, execution_strategy_contract_required=True, **extra
    )


def _models(cmds):
    return [c[c.index("--model") + 1] for c in cmds if "--model" in c]


@pytest.mark.parametrize("kind", ["plan_state", "plan_revision"])
def test_bare_strings_are_normalized_without_repair_or_replay(tmp_path, kind):
    row = _row(forbidden_side_effects="Must not refuse", related_scope_item_ids="scope-1")
    if kind == "plan_state":
        text = _state(row, exclusions="Not loosened")
    else:
        text = _revision(row)
    runner = _runner("codex", [text])
    response, error, logs, spy, _ = _run(runner, _config(tmp_path, "codex", 0), kind)
    assert error is None and response is not None
    assert len(_cmds(runner, "codex")) == 1
    spy.assert_not_called()
    assert any(
        "normalized risk test matrix string field(s)" in m
        and "forbidden_side_effects" in m and "related_scope_item_ids" in m
        for m in logs
    )
    stored = json.loads(response.text.split("\n<!--", 1)[0])
    assert stored["risk_test_matrix"]["rows"][0]["forbidden_side_effects"] == ["Must not refuse"]
    assert stored["risk_test_matrix"]["rows"][0]["related_scope_item_ids"] == ["scope-1"]


@pytest.mark.parametrize("kind", ["plan_state", "plan_revision"])
def test_unrecoverable_matrix_replays_once_with_quoted_error(tmp_path, kind):
    make = _state if kind == "plan_state" else _revision
    runner = _runner("codex", [make(_row(forbidden_side_effects="")), make()])
    response, error, logs, spy, _ = _run(runner, _config(tmp_path, "codex", 0), kind)
    assert error is None and response is not None
    cmds = _cmds(runner, "codex")
    assert len(cmds) == 2
    first, second = ("\n".join(c) for c in cmds)
    assert HEADING not in first
    assert HEADING in second and "forbidden_side_effects" in second
    assert any("retrying planner turn once" in m for m in logs)


def test_remaining_defect_after_normalization_is_quoted_and_persisted(tmp_path):
    bad_first = _state(_row(forbidden_side_effects="First slip", related_scope_item_ids=""))
    bad_replay = _state(_row(forbidden_side_effects="Replay slip", related_scope_item_ids=""))
    runner = _runner("codex", [bad_first, bad_replay])
    response, error, logs, _spy, exhaustions = _run(runner, _config(tmp_path, "codex", 1))
    assert response is None and error is not None
    cmds = _cmds(runner, "codex")
    assert len(cmds) == 2
    replay_prompt = "\n".join(cmds[1])
    assert HEADING in replay_prompt and "related_scope_item_ids" in replay_prompt
    assert "forbidden_side_effects must be a JSON array" not in replay_prompt
    assert any("normalized risk test matrix string field(s)" in m for m in logs)
    assert error.failure_category == "fresh-contract-integrity"
    assert "one automatic planner replay" in str(error)
    expected = _state(_row(forbidden_side_effects=["Replay slip"], related_scope_item_ids=""))
    exhaustion = error.plan_validation_exhaustion
    assert exhaustion is not None and len(exhaustions) == 1
    assert "related_scope_item_ids" in exhaustion.diagnostic
    assert "forbidden_side_effects" not in exhaustion.diagnostic
    assert json.loads(exhaustion.candidate_text.split("\n<!--", 1)[0]) == json.loads(
        expected.split("\n<!--", 1)[0]
    )
    assert exhaustion.candidate_digest == hashlib.sha256(exhaustion.candidate_text.encode()).hexdigest()


@pytest.mark.parametrize("agent", ["codex", "antigravity"])
def test_replay_does_not_consume_retry_budget_or_fallback(tmp_path, agent):
    bad = _state(_row(forbidden_side_effects=""))
    # retries=0: the replay still runs, on the same model.
    runner = _runner(agent, [bad, _state()])
    response, error, _logs, spy, _ = _run(runner, _config(tmp_path, agent, 0))
    assert error is None and response is not None
    cmds = _cmds(runner, agent)
    assert len(cmds) == 2
    if agent == "antigravity":
        assert _models(cmds) == ["ModelA", "ModelA"]
    # retries=1: replay, then an ordinary retryable failure, still gets its retry.
    runner = _runner(agent, [bad, "Error: server is overloaded, try again later", _state()])
    response, error, _logs, _spy, _ = _run(runner, _config(tmp_path, agent, 1))
    assert error is None and response is not None
    cmds = _cmds(runner, agent)
    assert len(cmds) == 3
    if agent == "antigravity":
        assert _models(cmds) == ["ModelA"] * 3


def test_execution_integrity_refusal_keeps_ordinary_budget(tmp_path):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload.pop("execution_recommendation")
    bad = json.dumps(payload) + PLAN_FOOTER
    runner = _runner("codex", [bad, bad, bad])
    response, error, _l, _s, _e = _run(runner, _config(tmp_path, "codex", 1))
    assert response is None and error.failure_category == "fresh-contract-integrity"
    assert len(_cmds(runner, "codex")) == 2
    assert "one automatic planner replay" not in str(error)


def test_repair_guard_logs_its_own_normalization(tmp_path):
    logs: list[str] = []
    with patch("coding_review_agent_loop.validated_agent.log", lambda _c, m: logs.append(m)):
        _, _, attempts = orchestrator._run_structured_repair(
            _state(_row(forbidden_side_effects="Slip", related_scope_item_ids="")),
            runner=FakeRunner(),
            config=_config(tmp_path, "codex", 0),
            usage_context=None,
            validate=_validate_state,
            repair_kwargs={"expected_kind": "plan_state", "require_risk_test_matrix_contract": True},
        )
    assert attempts and attempts[-1].outcome == "fresh_contract_integrity"
    assert any("normalized risk test matrix string field(s)" in m and "forbidden_side_effects" in m for m in logs)


def _order(runner, binary):
    """Ordered ('agent'|'sleep') events from the runner's recorded commands."""
    events = []
    for cmd, _cwd in runner.commands:
        if cmd[:1] == [binary]:
            events.append("agent")
        elif cmd[:1] == ["sleep"]:
            events.append("sleep")
    return events


def _spy_transitions():
    from coding_review_agent_loop.agents.antigravity import AntigravityAttemptState

    transitions: list[int] = []
    real = AntigravityAttemptState.next_after_failure

    def spy(self, **kwargs):
        transitions.append(self.retries_remaining)
        return real(self, **kwargs)

    return AntigravityAttemptState, transitions, spy


def test_replay_takes_no_transition_or_delay_with_zero_budget(tmp_path):
    state_cls, transitions, spy = _spy_transitions()
    runner = _runner("antigravity", [_state(_row(forbidden_side_effects="")), _state()])
    with patch.object(state_cls, "next_after_failure", spy):
        response, error, _l, _s, _e = _run_noskip(runner, _config(tmp_path, "antigravity", 0))
    assert error is None and response is not None
    assert transitions == []
    assert _order(runner, "agy") == ["agent", "agent"]
    assert _models(_cmds(runner, "antigravity")) == ["ModelA", "ModelA"]


def test_replay_transitions_are_not_taken_for_the_dedicated_replay(tmp_path):
    state_cls, transitions, spy = _spy_transitions()
    bad = _state(_row(forbidden_side_effects=""))
    runner = _runner("antigravity", [bad, "Error: server is overloaded, try again later", _state()])
    with patch.object(state_cls, "next_after_failure", spy):
        response, error, _logs, _spy, _ = _run_noskip(runner, _config(tmp_path, "antigravity", 1))
    assert error is None and response is not None
    # Only the later ordinary failure takes a transition, with its full allowance.
    assert transitions == [1]
    # No delay between the refusal and the replay; exactly one before the ordinary retry.
    assert _order(runner, "agy") == ["agent", "agent", "sleep", "agent"]
    assert _models(_cmds(runner, "antigravity")) == ["ModelA"] * 3


def _run_noskip(runner, config):
    logs: list[str] = []
    with patch("coding_review_agent_loop.validated_agent.log", lambda _c, m: logs.append(m)):
        try:
            response = orchestrator._run_validated_agent(
                runner, agent=_AGENT[0], config=config, prompt="ORIGINAL PROMPT",
                session_id=None,
                marker_description="<!-- AGENT_PLAN_STATE: approved|blocking -->",
                validate=_validate_state, use_repair=True, repair_expected_kind="plan_state",
                role="planner", operation_description="planning",
                require_execution_strategy_contract=True,
                require_risk_test_matrix_contract=True,
            )
            error = None
        except AgentInvocationError as exc:
            response, error = None, exc
    return response, error, logs, None, []


def test_mixed_contract_failure_after_matrix_replay_reports_the_real_contract(tmp_path):
    payload = json.loads(_state().rsplit("\n<!--", 1)[0])
    payload.pop("execution_recommendation")
    exec_bad = json.dumps(payload) + PLAN_FOOTER
    runner = _runner("codex", [_state(_row(forbidden_side_effects="")), exec_bad])
    response, error, _logs, _spy, exhaustions = _run(runner, _config(tmp_path, "codex", 0))
    assert response is None and error.failure_category == "fresh-contract-integrity"
    assert len(_cmds(runner, "codex")) == 2
    text = str(error)
    assert "also failed the fresh risk-test-matrix contract" not in text
    assert "execution_recommendation" in text
    assert len(exhaustions) == 1
