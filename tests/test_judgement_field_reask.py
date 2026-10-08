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


@pytest.mark.parametrize("source_status", ["update", None, "", 7, ...])
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


# --- Workflow-path tests with the real validators and real repair preservation ---

import json
import threading

from coding_review_agent_loop.cli import run_issue_loop, run_pr_loop
from coding_review_agent_loop.architecture_contract import _architecture_mode_validators
from coding_review_agent_loop.unresolved_items import _validate_review_response
from agent_loop_helpers import (
    structured_coder_followup,
    structured_plan_review,
    structured_plan_state,
)

PR_HEADING = HEADING
_VALID = {"status": "unchanged", "rationale": "No architectural change."}
_ABSENT = object()


def _with_impact(text, impact):
    """Return ``text`` with an architecture_impact (or none when ``impact`` is None)."""
    head, sep, tail = text.partition("\n")
    payload = json.loads(head)
    if impact is not None:
        payload["architecture_impact"] = impact
    return json.dumps(payload) + sep + tail


def _imp(status, **extra):
    impact = dict(_VALID, **extra)
    if status is _ABSENT:
        del impact["status"]
    else:
        impact["status"] = status
    return impact


def _real_pr_validators(reviewer="OpenAI Codex"):
    return _architecture_mode_validators(
        lambda mode: lambda text: _validate_review_response(
            text, reviewer=reviewer, unresolved_items=(), current_round_items=(),
            architecture_status_mode=mode,
        )
    )


def _run_real(runner, config, *, judgement=True, agent="codex"):
    return orchestrator._run_validated_agent(
        runner,
        agent=agent,
        config=config,
        prompt="ORIGINAL PROMPT",
        session_id="sess-1",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        **_real_pr_validators(),
        use_repair=True,
        repair_expected_kind="pr_review",
        role="reviewer",
        operation_description="PR review",
        reask_on_missing_judgement_field=judgement,
    )


def _pr(summary, impact):
    return _with_impact(structured_pr_review(summary=summary), impact)


@pytest.mark.parametrize("status", ["update", None, "", 7, _ABSENT])
@pytest.mark.parametrize("judgement", [True, False])
def test_terminal_diagnostic_names_field_and_repair_refusal(tmp_path, status, judgement):
    bad = _pr("BAD", _imp(status))
    runner = FakeRunner(codex_outputs=[bad, bad, bad, bad])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        with pytest.raises(AgentLoopError) as info:
            _run_real(runner, config, judgement=judgement)
    message = str(info.value)
    assert "architecture_impact.status" in message
    assert "changed | unchanged" in message and "repair skipped" in message
    assert repairs == []
    assert len(_codex(runner)) == (2 if judgement else 1)


def test_real_validator_reasks_out_of_enum_status_once(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_pr("BAD", _imp("update")), _pr("GOOD", _imp("unchanged"))]
    )
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        response = _run_real(runner, config)
    assert "GOOD" in response.text
    commands = _codex(runner)
    assert len(commands) == 2 and HEADING in commands[1] and HEADING not in commands[0]
    assert repairs == []


@pytest.mark.parametrize(
    "extra",
    [
        {"rationale": ""},
        {"dependencies": 123},
        {"canonical_document_path": ""},
        {"unknown_key": 1},
    ],
)
@pytest.mark.parametrize("status", ["update", _ABSENT])
def test_mixed_defect_goes_to_repair_without_spending_the_reask(tmp_path, status, extra):
    bad = _pr("MIXED", _imp(status, **extra))
    runner = FakeRunner(codex_outputs=[bad, bad, bad, bad])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        with pytest.raises(AgentLoopError):
            _run_real(runner, config)
    assert len(repairs) >= 1
    assert all(HEADING not in cmd for cmd in _codex(runner))


@pytest.mark.parametrize("drop_kind", [False, True])
@pytest.mark.parametrize("source_impact", [_imp("update", unknown_key=1), None, ...])
def test_invented_status_from_repair_is_never_accepted(tmp_path, drop_kind, source_impact):
    # A structural defect masks the status defect so the response reaches repair;
    # the repair model then fabricates a decision the reviewer never made.
    source = structured_pr_review(summary="Masked.")
    head, _, tail = source.partition("\n")
    payload = json.loads(head)
    if source_impact is not None and source_impact is not ...:
        payload["architecture_impact"] = source_impact
    elif source_impact is ...:
        payload["architecture_impact"] = None
    payload["blocking_items"] = "not-a-list"
    if drop_kind:
        del payload["kind"]
    bad = json.dumps(payload) + "\n" + tail
    fabricated = structured_pr_review(summary="Masked.")
    fabricated = _with_impact(fabricated, _imp("unchanged"))
    runner = FakeRunner(codex_outputs=[bad, bad, bad, bad])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    calls: list[str] = []

    def fake_attempt(raw, cmd, **kw):
        calls.append(raw)
        return fabricated

    with patch.object(orchestrator, "attempt_repair", fake_attempt):
        with pytest.raises(AgentLoopError) as info:
            _run_real(runner, config)
    assert calls, "repair was never attempted"
    assert (
        "Repair content preservation failed for architecture_impact.status" in str(info.value)
    )


@pytest.mark.parametrize("drop_kind", [False, True])
def test_control_masked_source_with_declared_status_is_repaired(tmp_path, drop_kind):
    # Same masked structural defect, but the reviewer declared the status: the
    # guard must let a status-preserving format repair through.
    head, _, tail = structured_pr_review(summary="Masked.").partition("\n")
    payload = json.loads(head)
    payload["architecture_impact"] = _imp("unchanged")
    payload["blocking_items"] = "not-a-list"
    if drop_kind:
        del payload["kind"]
    bad = json.dumps(payload) + "\n" + tail
    fixed = _with_impact(structured_pr_review(summary="Masked."), _imp("unchanged"))
    runner = FakeRunner(codex_outputs=[bad])
    config = make_config(tmp_path, reviewer=("codex",), agent_max_retries=0)
    with patch.object(orchestrator, "attempt_repair", lambda raw, cmd, **kw: fixed):
        response = _run_real(runner, config)
    assert "Masked." in response.text


def _pr_panel(target_round2):
    first = structured_pr_review(
        state="blocking", summary="Found a blocker.", blocking_items=["Blocker."]
    )
    disposed = [{"item_id": "item-1", "disposition": "resolved"}]
    return FakeRunner(
        codex_outputs=[first, *target_round2],
        gemini_outputs=[
            structured_pr_review(summary="Gemini approves.", reviewer="Google Gemini"),
            structured_pr_review(
                summary="Gemini approves after the fix.", reviewer="Google Gemini",
                prior_item_dispositions=disposed,
            ),
        ],
        claude_outputs=[structured_coder_followup(summary="Fixed it.", addressed_items=["item-1"])],
    )


def test_pr_loop_parallel_worker_reasks_invalid_status_without_repair(tmp_path):
    disposed = [{"item_id": "item-1", "disposition": "resolved"}]
    round2_bad = _with_impact(
        structured_pr_review(summary="Bad status.", prior_item_dispositions=disposed),
        _imp("update"),
    )
    round2_good = _with_impact(
        structured_pr_review(summary="Good status.", prior_item_dispositions=disposed),
        _imp("unchanged"),
    )
    runner = _pr_panel([round2_bad, round2_good])
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    commands = _codex(runner)
    assert len(commands) == 3
    assert HEADING not in commands[1] and HEADING in commands[2]
    assert repairs == []


def test_antigravity_reask_stays_on_the_same_model(tmp_path):
    runner = FakeRunner(
        antigravity_outputs=[(_pr("BAD", _imp("update")), 0), (_pr("GOOD", _imp("unchanged")), 0)]
    )
    config = make_config(
        tmp_path, reviewer=("antigravity",), agent_max_retries=0,
        antigravity_models=("ModelA", "ModelB", "ModelC"),
    )
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)), \
            patch.object(orchestrator.time, "sleep", side_effect=AssertionError("slept")):
        response = _run_real(runner, config, agent="antigravity")
    assert "GOOD" in response.text
    commands = [cmd for cmd, _ in runner.commands if cmd[:1] == ["agy"]]
    models = [cmd[cmd.index("--model") + 1] for cmd in commands]
    assert models == ["ModelA", "ModelA"]
    assert repairs == []


class _PlanReaskRunner(FakeRunner):
    """Codex's first plan review has an invalid status; gemini finishes in between."""

    def __init__(self, *, parallel, **kwargs):
        super().__init__(**kwargs)
        self.parallel = parallel
        self.codex_prompts: list[str] = []
        self.codex_calls = 0
        self.gemini_done = threading.Event()
        self.comments_at_reask: list[str] | None = None

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        if cmd[:2] == ["codex", "exec"]:
            self.codex_calls += 1
            self.codex_prompts.append(kwargs.get("input_text") or "")
            if self.codex_calls == 2:
                if self.parallel:
                    assert self.gemini_done.wait(10)
                self.comments_at_reask = list(self.comments)
            return super().run_with_log(args, cwd=cwd, **kwargs)
        if cmd[:1] == ["gemini"] and self.parallel:
            result = super().run_with_log(args, cwd=cwd, **kwargs)
            self.gemini_done.set()
            return result
        return super().run_with_log(args, cwd=cwd, **kwargs)


@pytest.mark.parametrize("parallel", [True, False])
def test_plan_review_reask_keeps_frozen_prompt_and_publishes_nothing_between_turns(
    tmp_path, parallel
):
    plan = structured_plan_state(
        state="blocking", summary="Initial plan.", plan_steps=["Make the change."]
    )
    runner = _PlanReaskRunner(
        parallel=parallel,
        claude_outputs=[plan],
        codex_outputs=[
            _with_impact(structured_plan_review(summary="CODEX-BAD"), _imp("update")),
            _with_impact(structured_plan_review(summary="CODEX-GOOD"), _imp("unchanged")),
        ],
        gemini_outputs=[
            structured_plan_review(summary="GEMINI-UNIQUE approves.", reviewer="Google Gemini")
        ],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=parallel, agent_max_retries=0
    )
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    first, reask = runner.codex_prompts[0], runner.codex_prompts[1]
    marker = "PUBLIC RESPONSE FILE"
    first_body = first.split(marker)[0]
    assert reask.startswith(first_body)
    addendum = reask[len(first_body):].split(marker)[0]
    assert addendum.lstrip().startswith(HEADING) and "architecture_impact.status" in addendum
    assert "GEMINI-UNIQUE" not in reask
    assert runner.comments_at_reask is not None
    assert not any("GEMINI-UNIQUE" in c or "CODEX-BAD" in c for c in runner.comments_at_reask)
    assert repairs == []


# --- #1330: object list items are accepted at the seam without repair ---

from coding_review_agent_loop.errors import AgentInvocationError  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    validate_structured_issue_implementation,
)
from agent_loop_helpers import structured_issue_implementation  # noqa: E402

_OBJ_CHANGED_1330 = {
    "status": "changed",
    "rationale": "A typed registry replaces the ad hoc lookup.",
    "affected_components": [{"area": "x", "change": "y"}],
    "dependencies": [],
    "execution_data_flows": [],
    "persistence": [],
    "public_contracts": [],
    "security_boundaries": [],
    "canonical_document_action": "update",
    "canonical_document_path": "ARCHITECTURE.md",
    "canonical_document_rationale": "Document the registry.",
}
_OBJ_MODIFIED_1330 = {
    "status": "modified",
    "rationale": "The parser gains a flattening step.",
    "affected_components": [{"name": "protocol parser", "change": "modified"}],
    "dependencies": [{"from": "repair_preservation.py", "to": "protocol.py", "kind": "import"}],
    "execution_data_flows": [{"flow": "response -> parser -> seam"}],
    "persistence": [{"record": "round metadata degradation records"}],
    "public_contracts": [{"contract": "architecture_impact list fields"}],
    "security_boundaries": [{"boundary": "agent payload trust boundary"}],
    "canonical_document_action": "update",
    "canonical_document_path": "ARCHITECTURE.md",
    "canonical_document_rationale": "Document the flattening.",
}
_EMPTY_MODIFIED_1330 = {
    **{key: [] for key in (
        "affected_components", "dependencies", "execution_data_flows",
        "persistence", "public_contracts", "security_boundaries",
    )},
    "status": "modified",
    "rationale": "Something changed.",
    "canonical_document_action": "update",
    "canonical_document_path": "ARCHITECTURE.md",
    "canonical_document_rationale": "Document it.",
}


def _issue_validators_1330():
    return _architecture_mode_validators(
        lambda mode: lambda text: validate_structured_issue_implementation(
            text, required_architecture_impact_contract=1, architecture_status_mode=mode
        )
    )


def _run_issue_1330(runner, config):
    return orchestrator._run_validated_agent(
        runner,
        agent="claude",
        config=config,
        prompt="Implement.",
        session_id="sess-1",
        marker_description="structured issue_implementation result",
        **_issue_validators_1330(),
        use_repair=True,
        repair_expected_kind="issue_implementation",
        role="coder",
        operation_description="issue implementation",
        require_architecture_impact_contract=True,
    )


def _claude(runner):
    return ["\n".join(cmd) for cmd, _ in runner.commands if cmd[:1] == ["claude"]]


def test_seam_accepts_declared_changed_with_object_entries_without_repair_1330(tmp_path):
    text = _with_impact(structured_issue_implementation(), _OBJ_CHANGED_1330)
    runner = FakeRunner(claude_outputs=[text])
    config = make_config(tmp_path, agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        response = _run_issue_1330(runner, config)
    assert repairs == []
    assert len(_claude(runner)) == 1
    parsed = response.marker_value
    assert parsed.architecture_impact.status == "changed"
    assert parsed.architecture_impact.affected_components == ("area: x; change: y",)
    assert parsed.architecture_impact_degradations == ()
    strict = validate_structured_issue_implementation(
        response.text, required_architecture_impact_contract=1
    )
    assert strict.architecture_impact.affected_components == ("area: x; change: y",)


def test_seam_accepts_corroborated_modified_with_object_entries_1330(tmp_path):
    text = _with_impact(structured_issue_implementation(), _OBJ_MODIFIED_1330)
    runner = FakeRunner(claude_outputs=[text])
    config = make_config(tmp_path, agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        response = _run_issue_1330(runner, config)
    assert repairs == []
    assert len(_claude(runner)) == 1
    parsed = response.marker_value
    assert parsed.architecture_impact.status == "changed"
    assert parsed.architecture_impact.affected_components == (
        "name: protocol parser; change: modified",
    )
    assert [r.outcome for r in parsed.architecture_impact_degradations] == ["normalized-to-changed"]
    head = json.loads(response.text.partition("\n")[0])
    assert head["architecture_impact"]["status"] == "changed"
    assert '"status": "modified"' not in response.text
    strict = validate_structured_issue_implementation(
        response.text, required_architecture_impact_contract=1
    )
    assert strict.architecture_impact.status == "changed"
    assert strict.architecture_impact.dependencies == (
        "from: repair_preservation.py; to: protocol.py; kind: import",
    )


def test_seam_still_refuses_modified_with_empty_evidence_1330(tmp_path):
    text = _with_impact(structured_issue_implementation(), _EMPTY_MODIFIED_1330)
    runner = FakeRunner(claude_outputs=[text])
    config = make_config(tmp_path, agent_max_retries=0)
    repairs: list[str] = []
    with patch.object(orchestrator, "_run_structured_repair", _repair_recorder(repairs)):
        with pytest.raises(AgentInvocationError) as error:
            _run_issue_1330(runner, config)
    preserved = error.value.preserved_unsatisfied_response
    assert preserved is not None and preserved.text == text
    assert [r.outcome for r in preserved.architecture_impact_degradations] == [
        "degraded-to-undetermined"
    ]
    assert "architecture_impact" in str(error.value)
