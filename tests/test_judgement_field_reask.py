"""A missing or invalid reviewer judgement field is re-asked once (#1185)."""
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.errors import (
    AgentLoopError,
    MissingJudgementFieldError,
    MissingPriorItemDispositionError,
)
from coding_review_agent_loop.protocol import parse_architecture_impact
from coding_review_agent_loop.repair_preservation import validate_repair_preservation
from agent_loop_helpers import FakeRunner, make_config, structured_pr_review

HEADING = "## Previous response not accepted: required judgement field"
CTX = "plan_review.architecture_impact"
BAD = structured_pr_review(summary="BAD-STATUS")
GOOD = structured_pr_review(summary="GOOD")
OMITTED = structured_pr_review(summary="OMITTED")


def _impact(**overrides):
    payload = {"status": "unchanged", "rationale": "No architectural change."}
    payload.update(overrides)
    return payload


def _parse(payload):
    return parse_architecture_impact(payload, context=CTX)


@pytest.mark.parametrize("status", ["update", None, "", "  ", 7])
def test_sole_defect_status_is_typed_with_unchanged_message(status):
    with pytest.raises(MissingJudgementFieldError) as info:
        _parse(_impact(status=status))
    assert info.value.field_path == f"{CTX}.status"
    assert info.value.allowed_values == ("changed", "unchanged")
    assert len(info.value.observed_preview or "") <= 40
    assert str(info.value).startswith(f"{CTX}.status must be")


def test_absent_status_is_typed_when_sole_defect():
    payload = _impact()
    del payload["status"]
    with pytest.raises(MissingJudgementFieldError) as info:
        _parse(payload)
    assert str(info.value) == f"{CTX} is missing required field(s): status"
    assert info.value.observed_preview is None


@pytest.mark.parametrize(
    "extra",
    [
        {"unknown_key": 1},
        {"rationale": ""},
        {"dependencies": 123},
        {"canonical_document_path": ""},
    ],
)
@pytest.mark.parametrize("status", ["update", None, "ABSENT"])
def test_mixed_defects_stay_plain(status, extra):
    payload = _impact(**extra)
    if status == "ABSENT":
        del payload["status"]
    else:
        payload["status"] = status
    with pytest.raises(AgentLoopError) as info:
        _parse(payload)
    assert not isinstance(info.value, MissingJudgementFieldError)


def test_absent_status_and_rationale_keeps_combined_diagnostic():
    with pytest.raises(AgentLoopError, match="rationale, status") as info:
        _parse({})
    assert not isinstance(info.value, MissingJudgementFieldError)


def test_legacy_mode_is_not_typed():
    with pytest.raises(AgentLoopError) as info:
        parse_architecture_impact(_impact(status=7), context=CTX, architecture_status_mode="legacy")
    assert not isinstance(info.value, MissingJudgementFieldError)


def _validate(text):
    if "BAD-STATUS" in text:
        raise MissingJudgementFieldError(
            "x.status must be `changed` or `unchanged`.",
            field_path="architecture_impact.status",
            allowed_values=("changed", "unchanged"),
            observed_preview="update",
        )
    if "OMITTED" in text:
        raise MissingPriorItemDispositionError(("item-4",))
    return text


def _run(runner, config, *, judgement=True, omission=False, kind="pr_review"):
    return orchestrator._run_validated_agent(
        runner,
        agent="codex",
        config=config,
        prompt="ORIGINAL PROMPT",
        session_id="sess-1",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        validate=_validate,
        use_repair=True,
        repair_expected_kind=kind,
        role="reviewer",
        operation_description="PR review",
        reask_on_missing_judgement_field=judgement,
        reask_on_prior_disposition_omission=omission,
    )


def _codex(runner):
    return ["\n".join(cmd) for cmd, _ in runner.commands if cmd[:1] == ["codex"]]


def _repair_recorder(calls):
    def fake_repair(raw, **kwargs):
        calls.append(raw)
        return None, None, []

    return fake_repair


def test_invalid_status_is_reasked_once_and_accepted(tmp_path):
    runner = FakeRunner(codex_outputs=[BAD, GOOD])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        response = _run(runner, config)
    assert "GOOD" in response.text
    commands = _codex(runner)
    assert len(commands) == 2
    assert HEADING not in commands[0]
    assert "ORIGINAL PROMPT" in commands[1] and HEADING in commands[1]
    assert "architecture_impact.status" in commands[1] and "changed | unchanged" in commands[1]
    assert repairs == []


def test_second_invalid_status_stops_without_repair(tmp_path):
    runner = FakeRunner(codex_outputs=[BAD, BAD, BAD, BAD])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        with pytest.raises(AgentLoopError, match="status"):
            _run(runner, config)
    assert len(_codex(runner)) == 2
    assert repairs == []


def test_flag_off_has_no_reask_and_review_skips_repair(tmp_path):
    runner = FakeRunner(codex_outputs=[BAD, BAD, BAD, BAD])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        with pytest.raises(AgentLoopError):
            _run(runner, config, judgement=False)
    assert all(HEADING not in cmd for cmd in _codex(runner))
    assert repairs == []


@pytest.mark.parametrize("first,second", [(OMITTED, BAD), (BAD, OMITTED)])
def test_single_slot_is_shared(tmp_path, first, second):
    runner = FakeRunner(codex_outputs=[first, second, GOOD, GOOD])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder([])):
        with pytest.raises(AgentLoopError):
            _run(runner, config, omission=True)
    assert len(_codex(runner)) == 2


def _review(status, kind="pr_review", drop_kind=False):
    import json

    payload = {
        "schema_version": 1,
        "kind": kind,
        "state": "approved",
        "summary": "ok",
        "blocking_items": [],
        "same_pr_followups": [],
        "future_followups": [],
        "prior_item_dispositions": [],
        "architecture_impact": {"status": status, "rationale": "r"},
    }
    if status is ...:
        del payload["architecture_impact"]
    if drop_kind:
        del payload["kind"]
    return json.dumps(payload)


@pytest.mark.parametrize("source_status", ["update", None, "", 7])
@pytest.mark.parametrize("drop_kind", [False, True])
def test_repair_cannot_invent_status(source_status, drop_kind):
    source = _review(source_status, drop_kind=drop_kind)
    candidate = _review("unchanged")
    with pytest.raises(AgentLoopError, match="architecture_impact.status"):
        validate_repair_preservation(source, candidate)


@pytest.mark.parametrize("drop_kind", [False, True])
def test_repair_must_preserve_declared_status(drop_kind):
    source = _review("changed", drop_kind=drop_kind)
    validate_repair_preservation(source, _review("changed"))
    for bad in (_review("unchanged"), _review(...)):
        with pytest.raises(AgentLoopError, match="architecture_impact.status"):
            validate_repair_preservation(source, bad)
