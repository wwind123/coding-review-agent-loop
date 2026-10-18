"""Advisory issue triage (#1434, stage 2 of #1430).

Every test uses fixtures and fake backends; no live model or external service
is called.
"""

from __future__ import annotations

import datetime
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.issue_loop as issue_loop_module
import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.triage as triage_module
from agent_loop_helpers import FakeRunner, make_config
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    PlanPhase,
)
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.protocol import EXECUTION_DISPOSITION_DIRECT, EXECUTION_DISPOSITION_PLANNING
from coding_review_agent_loop.response_validation import parse_triage_response
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent
from coding_review_agent_loop.triage import (
    SIGNAL_EDGE_CASES,
    SIGNAL_LOCAL_CHANGE,
    SIGNAL_OPEN_QUESTION,
    SIGNAL_PRIOR_STALL,
    SIGNAL_SPECIFIED_BEHAVIOR,
    TRIAGE_REASON_SIGNALS,
    extract_signals,
    parse_triage_record,
    resolve_triage_backend,
    retrieval_exclusion,
    retrieve_similar_issues,
    run_issue_triage,
    triage_invocation,
)
from coding_review_agent_loop.usage import RunUsageContext
from coding_review_agent_loop.validated_agent import _new_usage_context
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in

FIXTURES = Path(__file__).parent / "fixtures" / "triage"
UTC = datetime.timezone.utc
TRIAGE_TOKEN = "AGENT_ISSUE_TRIAGE"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _triage_output(route="discuss-first", *, unsure=False, signal="open-design-question"):
    payload = {
        "schema_version": 1,
        "kind": "issue_triage",
        "route": route,
        "unsure": unsure,
        "reasons": [{"signal": signal, "explanation": "The issue asks how something should work."}],
    }
    return json.dumps(payload) + "\n-- OpenAI Codex"


def _issue(**overrides) -> IssueContext:
    values = dict(
        number=56, repo="OWNER/REPO", title="Fix issue-mode context",
        body="How should degraded payloads be normalized?", url="issue-url", comments=(),
    )
    values.update(overrides)
    return IssueContext(**values)


def _triage_comments(runner) -> list[str]:
    return [body for body in runner.comments if TRIAGE_TOKEN in body]


def _codex_prompts(runner) -> list[str]:
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]


def _codex_calls(runner) -> int:
    return len(_codex_prompts(runner))


class _StopAtFirstTurn(Exception):
    """Ends a run at its first role turn, after the dispatch seam recorded it."""


def _patch_issue_startup(monkeypatch, issue_context):
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop_module), "validate_open_issue", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop_module), "get_issue_context", lambda *a, **k: issue_context)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop_module), "resolve_canonical_pr_for_issue", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop_module), "prepare_agent_memory", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop_module), "sync_coder_base_before_implementation", lambda *a, **k: None
    )


def _record_dispatch(seen, runner, *, raise_stop: bool):
    """A first role turn: note the dispatch exactly like the shared validated turn."""

    def turn(*_args, **kwargs):
        kwargs["usage_context"].note_agent_dispatch()
        seen.append({**kwargs, "comments_before_turn": list(runner.comments)})
        if raise_stop:
            raise _StopAtFirstTurn()
        return 0

    return turn


def _run_direct(tmp_path, monkeypatch, *, triage, codex_outputs=(), **config_overrides):
    runner = FakeRunner(codex_outputs=list(codex_outputs))
    config = make_config(tmp_path, triage=triage, **config_overrides)
    _patch_issue_startup(monkeypatch, _issue())
    seen: list[dict] = []
    monkeypatch.setattr(
        issue_loop_module, "_run_validated_agent", _record_dispatch(seen, runner, raise_stop=True)
    )
    usage_context = _new_usage_context(config)
    with pytest.raises(_StopAtFirstTurn):
        run_issue_loop(runner, issue_number=56, config=config, usage_context=usage_context)
    return runner, seen, usage_context


def _run_plan_first(tmp_path, monkeypatch, *, triage, codex_outputs=(), **config_overrides):
    runner = FakeRunner(codex_outputs=list(codex_outputs))
    config = make_config(tmp_path, triage=triage, **config_overrides)
    _patch_issue_startup(monkeypatch, _issue())
    seen: list[dict] = []
    monkeypatch.setattr(
        issue_loop_module, "_run_plan_first_loop", _record_dispatch(seen, runner, raise_stop=False)
    )
    usage_context = _new_usage_context(config)
    assert run_issue_loop(
        runner, issue_number=56, config=config, plan_first=True, usage_context=usage_context
    ) == 0
    return runner, seen, usage_context


# ---------------------------------------------------------------------------
# triage-fixture-signals
# ---------------------------------------------------------------------------


def _hints(issue):
    return {signal.name: signal.route_hint for signal in extract_signals(issue["title"], issue["body"])}


def test_fixture_signals_mark_1337_discuss_1404_direct_and_ambiguous_neither():
    hints_1337 = _hints(_fixture("issue_1337"))
    assert {SIGNAL_OPEN_QUESTION, SIGNAL_EDGE_CASES, SIGNAL_PRIOR_STALL} <= set(hints_1337)
    assert set(hints_1337.values()) == {"discuss-first"}

    hints_1404 = _hints(_fixture("issue_1404"))
    assert {SIGNAL_LOCAL_CHANGE, SIGNAL_SPECIFIED_BEHAVIOR} <= set(hints_1404)
    assert set(hints_1404.values()) == {"direct"}

    assert _hints(_fixture("issue_ambiguous")) == {}


def test_closed_acceptance_criteria_suppress_the_edge_case_signal():
    body = "Handle degraded and malformed payloads.\n\n## Acceptance\n- The nine listed cases pass."
    names = {signal.name for signal in extract_signals("Title", body)}
    assert SIGNAL_EDGE_CASES not in names
    assert "closed-acceptance-criteria" in names


def test_fixture_route_and_reasons_round_trip_into_the_comment(tmp_path):
    issue = _fixture("issue_1337")
    runner = FakeRunner(codex_outputs=[_triage_output("discuss-first", signal=SIGNAL_EDGE_CASES)])
    config = make_config(tmp_path, triage=True)
    outcome = run_issue_triage(
        runner, config=config, usage_context=_new_usage_context(config),
        issue_context=_issue(title=issue["title"], body=issue["body"]), operator_route="plan-first",
    )
    assert outcome is not None and outcome.comment_posted
    (comment,) = _triage_comments(runner)
    assert "Recommended route: **discuss-first**" in comment
    assert f"- `{SIGNAL_EDGE_CASES}`: The issue asks how something should work." in comment
    assert "continues with the route the operator selected: `plan-first`" in comment
    record = parse_triage_record(comment)
    assert record["route"] == "discuss-first"
    assert record["operator_route"] == "plan-first"
    assert record["reason_signals"] == [SIGNAL_EDGE_CASES]
    assert set(record["signals"]) >= {SIGNAL_OPEN_QUESTION, SIGNAL_EDGE_CASES}


# ---------------------------------------------------------------------------
# triage-unsure-plan-first
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "unsure", "reason"),
    [
        ("direct", True, "answered unsure"),
        ("discuss-first", True, "answered unsure"),
        ("rewrite-everything", False, "unrecognized route"),
    ],
    ids=["unsure-direct", "unsure-discuss", "unknown-route"],
)
def test_unsure_or_unknown_route_is_reported_as_plan_first(tmp_path, capsys, route, unsure, reason):
    answer = parse_triage_response(
        _triage_output(route, unsure=unsure), allowed_signals=TRIAGE_REASON_SIGNALS
    )
    assert answer.route == "plan-first" and answer.unsure and reason in answer.unsure_reason

    runner = FakeRunner(codex_outputs=[_triage_output(route, unsure=unsure)])
    config = make_config(tmp_path, triage=True, quiet=False)
    run_issue_triage(
        runner, config=config, usage_context=_new_usage_context(config),
        issue_context=_issue(), operator_route="direct",
    )
    (comment,) = _triage_comments(runner)
    assert "Recommended route: **plan-first** (unsure:" in comment
    assert reason in comment
    assert "Recommended route: **direct**" not in comment
    assert "Recommended route: **discuss-first**" not in comment
    assert parse_triage_record(comment)["route"] == "plan-first"
    log_text = capsys.readouterr().err
    assert "advisory triage recommends `plan-first` (unsure:" in log_text


def test_triage_validator_accepts_only_a_bare_object_or_one_signature():
    for text in (_triage_output(), _triage_output().split("\n", 1)[0], _triage_output() + "\n"):
        assert parse_triage_response(text, allowed_signals=TRIAGE_REASON_SIGNALS).route == "discuss-first"


@pytest.mark.parametrize(
    "text",
    [
        "Route: direct",
        json.dumps({"schema_version": 1, "kind": "issue_triage", "route": "direct", "unsure": False}),
        _triage_output(signal="made-up-signal"),
        json.dumps({
            "schema_version": 1, "kind": "issue_triage", "route": "direct", "unsure": "no",
            "reasons": [{"signal": "no-signal", "explanation": "x"}],
        }),
        _triage_output().replace("\n-- OpenAI Codex", '\n["direct"]\n-- OpenAI Codex'),
        _triage_output().replace("\n-- OpenAI Codex", "\n42"),
        _triage_output().replace("\n-- OpenAI Codex", "\nActually the route is direct.\n-- OpenAI Codex"),
        _triage_output().replace("\n-- OpenAI Codex", "\n-- OpenAI Codex\n-- OpenAI Codex"),
    ],
    ids=[
        "prose", "missing-reasons", "unknown-signal", "non-boolean-unsure",
        "trailing-json-array", "trailing-scalar", "trailing-prose", "two-signatures",
    ],
)
def test_triage_validator_is_strict(text):
    with pytest.raises(AgentLoopError):
        parse_triage_response(text, allowed_signals=TRIAGE_REASON_SIGNALS)


# ---------------------------------------------------------------------------
# triage-retrieval-bounds
# ---------------------------------------------------------------------------

CUTOFF = datetime.datetime(2026, 10, 1, 12, 0, tzinfo=UTC)


def _row(number, *, repo="OWNER/REPO", complete=True, available="2026-09-01T00:00:00+00:00",
         status="never-edited", as_of="2026-08-01T00:00:00+00:00", title="normalize degraded payloads",
         stops=0):
    return {
        "schema_version": 1, "repo": repo, "issue_number": number, "route": "plan-first",
        "outcome_complete": complete, "outcome_available_at": available,
        "snapshot": {"status": status, "title": title, "body": "degraded payload normalization", "as_of": as_of},
        "plan_rounds": {"value": 3, "evidence": "verified"},
        "step_backs": {"value": 0, "evidence": "verified"},
        "human_decision_stops": {"value": stops, "evidence": "verified"},
        "pr_rounds": {"value": None, "evidence": "unavailable"},
        "merged": {"value": False, "evidence": "verified"},
        "final_plan_size": {"value": 4200, "evidence": "verified"},
        "infrastructure_failures": [{"category": "auth-expiry", "round": 2}],
        "infrastructure_coverage": "partial",
        "missing_reviewer_records": [],
        "terminal_states": ["issue-closed"],
    }


def test_retrieval_is_same_repo_complete_before_cutoff_and_leak_free():
    rows = [
        _row(10),  # completed before T
        _row(11, available="2026-10-02T00:00:00+00:00"),  # started before T, completed after
        _row(12, available="2026-10-01T12:00:00+00:00"),  # PR merged exactly at T
        _row(13, complete=False, available=None),  # still open
        _row(14, status="unavailable", as_of=None),  # unavailable snapshot
        _row(56),  # the issue itself
        _row(15, repo="OTHER/REPO"),  # another repository
        _row(16, as_of="2026-10-05T00:00:00+00:00"),  # snapshot dated after T
    ]
    expected = {
        10: None, 11: "outcome not available before the cutoff",
        12: "outcome not available before the cutoff", 13: "outcome not complete",
        14: "no usable run-start snapshot", 56: "the issue itself", 15: "another repository",
        16: "snapshot not dated before the cutoff",
    }
    for row in rows:
        assert retrieval_exclusion(
            row, repo="OWNER/REPO", issue_number=56, cutoff=CUTOFF
        ) == expected[row["issue_number"]]
    similar = retrieve_similar_issues(
        rows, repo="OWNER/REPO", issue_number=56, cutoff=CUTOFF, query="normalize degraded payloads"
    )
    assert [item.issue_number for item in similar] == [10]


def test_retrieval_selects_at_most_five_deterministically():
    rows = [_row(number, title=f"normalize degraded payloads {number}") for number in range(30, 38)]
    first = retrieve_similar_issues(
        list(reversed(rows)), repo="OWNER/REPO", issue_number=56, cutoff=CUTOFF,
        query="normalize degraded payloads",
    )
    second = retrieve_similar_issues(
        rows, repo="OWNER/REPO", issue_number=56, cutoff=CUTOFF, query="normalize degraded payloads"
    )
    assert len(first) == 5
    assert first == second
    assert [item.issue_number for item in first] == sorted(item.issue_number for item in first)


def test_retrieved_stall_feeds_the_prior_stalled_plan_signal(tmp_path):
    history = tmp_path / "history.json"
    history.write_text(json.dumps({
        "schema_version": 1, "repo": "OWNER/REPO",
        "rows": [_row(10, stops=1), _row(11, available="2099-01-01T00:00:00+00:00")],
    }))
    runner = FakeRunner(codex_outputs=[_triage_output()])
    config = make_config(tmp_path, triage=True, triage_history=history)
    outcome = run_issue_triage(
        runner, config=config, usage_context=_new_usage_context(config),
        issue_context=_issue(title="Normalize degraded payloads", body="Normalize degraded payloads."),
        operator_route="direct", now=lambda: CUTOFF,
    )
    assert [item.issue_number for item in outcome.similar] == [10]
    stall = next(signal for signal in outcome.signals if signal.name == SIGNAL_PRIOR_STALL)
    assert any("#10" in item for item in stall.evidence)
    (prompt,) = _codex_prompts(runner)
    assert "#10 (similarity" in prompt and "#11" not in prompt
    # The model sees the retrieved snapshot and the complete outcome row.
    assert "Run-start body:\ndegraded payload normalization" in prompt
    for evidence in (
        '"final_plan_size"', '"infrastructure_failures"', '"infrastructure_coverage": "partial"',
        '"missing_reviewer_records"', '"terminal_states"', '"outcome_available_at"',
        '"snapshot_status": "never-edited"', "auth-expiry",
    ):
        assert evidence in prompt, evidence
    assert parse_triage_record(_triage_comments(runner)[0])["similar_issues"] == [10]


def test_without_a_history_file_zero_similar_issues_and_the_prompt_says_so(tmp_path):
    runner = FakeRunner(codex_outputs=[_triage_output()])
    config = make_config(tmp_path, triage=True)
    outcome = run_issue_triage(
        runner, config=config, usage_context=_new_usage_context(config),
        issue_context=_issue(), operator_route="direct",
    )
    assert outcome.similar == ()
    (prompt,) = _codex_prompts(runner)
    assert "No outcome history file was given (--triage-history), so zero similar past issues" in prompt


# ---------------------------------------------------------------------------
# triage-default-backend
# ---------------------------------------------------------------------------


def _seat(tmp_path, seat_id="seat-a", backend="codex", model="gpt-seat"):
    workdir = tmp_path / seat_id
    workdir.mkdir(exist_ok=True)
    return SeatAgent(ReviewerSeat(seat_id=seat_id, backend=backend, model_chain=(model,)), workdir)


def test_default_backend_primary_then_codex_then_first_reviewer(tmp_path):
    primary = make_config(
        tmp_path, reviewer=("codex", "claude"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="claude", reviewer_claude_model="claude-review",
    )
    backend = resolve_triage_backend(primary)
    assert (backend.agent, backend.model) == ("claude", "claude-review")
    assert backend.reason == "the primary plan reviewer"

    with_codex = make_config(tmp_path, reviewer=("claude", "codex"), reviewer_codex_model="gpt-review")
    backend = resolve_triage_backend(with_codex)
    assert (backend.agent, backend.model) == ("codex", "gpt-review")
    assert backend.reason == "the first configured codex reviewer"

    without_codex = make_config(tmp_path, reviewer=("gemini", "claude"), gemini_model="gemini-x")
    backend = resolve_triage_backend(without_codex)
    assert (backend.agent, backend.model) == ("gemini", "gemini-x")
    assert backend.reason == "the first configured reviewer"


def test_named_seat_keeps_its_seat_model_and_triage_model_overrides_only_the_model(tmp_path):
    seat = _seat(tmp_path)
    config = make_config(tmp_path, reviewer=(seat,))
    backend = resolve_triage_backend(config)
    assert backend.agent is seat and backend.model == "gpt-seat"

    overridden = replace(config, triage_model="gpt-triage")
    backend = resolve_triage_backend(overridden)
    assert str(backend.agent) == "seat-a" and backend.model == "gpt-triage"
    assert backend.reason.endswith("with the model from --triage-model")
    agent, invocation_config = triage_invocation(overridden, backend)
    assert isinstance(agent, SeatAgent) and agent.backend == "codex"
    assert agent.model_chain == ("gpt-triage",) and agent.workdir == seat.workdir
    assert invocation_config is overridden
    assert overridden.reviewer == (seat,) and seat.model_chain == ("gpt-seat",)


@pytest.mark.parametrize(
    ("reviewer", "field", "expected"),
    [
        (("claude", "codex"), "reviewer_codex_model", "gpt-triage"),
        (("claude", "gemini"), "reviewer_claude_model", "gpt-triage"),
    ],
    ids=["codex", "claude"],
)
def test_triage_model_override_leaves_reviewer_and_coder_configuration(tmp_path, reviewer, field, expected):
    config = make_config(tmp_path, reviewer=reviewer, triage_model="gpt-triage")
    backend = resolve_triage_backend(config)
    _agent, invocation_config = triage_invocation(config, backend)
    assert getattr(invocation_config, field) == expected
    assert invocation_config.reviewer == config.reviewer
    assert invocation_config.coder == config.coder
    assert getattr(config, field) == ""


def test_backend_choice_and_reason_are_logged_and_shown(tmp_path, capsys):
    runner = FakeRunner(codex_outputs=[_triage_output()])
    config = make_config(tmp_path, triage=True, quiet=False, reviewer_codex_model="gpt-review")
    run_issue_triage(
        runner, config=config, usage_context=_new_usage_context(config),
        issue_context=_issue(), operator_route="direct",
    )
    assert "triage backend Codex model=gpt-review (the first configured codex reviewer)" in capsys.readouterr().err
    (comment,) = _triage_comments(runner)
    assert "Triage backend: Codex, model `gpt-review` (the first configured codex reviewer)." in comment


# ---------------------------------------------------------------------------
# triage-route-unchanged
# ---------------------------------------------------------------------------


def _turn_identity(call):
    return (
        call["agent"], call["role"], call["prompt"], call["operation_description"],
        call["config"].plan_execution_mode, call["config"].coder, call["config"].reviewer,
    )


def test_direct_route_is_dispatched_unchanged_after_the_triage_comment(tmp_path, monkeypatch):
    runner_off, seen_off, _ = _run_direct(tmp_path, monkeypatch, triage=False)
    runner_on, seen_on, usage = _run_direct(
        tmp_path, monkeypatch, triage=True, codex_outputs=[_triage_output("discuss-first")]
    )
    assert _codex_calls(runner_off) == 0 and _triage_comments(runner_off) == []
    assert _codex_calls(runner_on) == 1
    assert len(seen_on) == len(seen_off) == 1
    assert _turn_identity(seen_on[0]) == _turn_identity(seen_off[0])
    # The advisory comment was posted before the coder turn was dispatched.
    assert len(_triage_comments(SimpleNamespace(comments=seen_on[0]["comments_before_turn"]))) == 1
    assert "Recommended route: **discuss-first**" in _triage_comments(runner_on)[0]
    assert "operator selected: `direct`" in _triage_comments(runner_on)[0]
    assert not any("gh issue edit" in " ".join(map(str, cmd)) for cmd in runner_on.commands)
    assert [record.turn_role for record in usage.records] == ["reviewer"]


def test_plan_first_route_is_dispatched_unchanged_after_the_triage_comment(tmp_path, monkeypatch):
    _runner_off, seen_off, _ = _run_plan_first(tmp_path, monkeypatch, triage=False)
    runner_on, seen_on, _ = _run_plan_first(
        tmp_path, monkeypatch, triage=True, codex_outputs=[_triage_output("direct")]
    )
    keys = ("issue_number", "requested_policy", "implement_after_approval", "inherited_matrix_binding")
    assert [seen_on[0][key] for key in keys] == [seen_off[0][key] for key in keys]
    assert seen_on[0]["config"].plan_execution_mode == seen_off[0]["config"].plan_execution_mode
    assert len(_triage_comments(SimpleNamespace(comments=seen_on[0]["comments_before_turn"]))) == 1
    assert "operator selected: `plan-first`" in _triage_comments(runner_on)[0]


# ---------------------------------------------------------------------------
# triage-failure-continues
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "codex_outputs",
    [[], ["This is not JSON at all."]],
    ids=["backend-unavailable", "invalid-after-repair"],
)
def test_triage_failure_is_logged_and_the_route_runs_unchanged(tmp_path, monkeypatch, capsys, codex_outputs):
    _runner_off, seen_off, _ = _run_direct(tmp_path, monkeypatch, triage=False)
    runner, seen, _ = _run_direct(
        tmp_path, monkeypatch, triage=True, codex_outputs=codex_outputs, quiet=False
    )
    assert _turn_identity(seen[0]) == _turn_identity(seen_off[0])
    assert _triage_comments(runner) == []
    assert "advisory triage failed" in capsys.readouterr().err


def test_comment_post_failure_is_logged_and_the_route_runs_unchanged(tmp_path, monkeypatch, capsys):
    def failing_post(*_args, **_kwargs):
        raise AgentLoopError("GitHub refused the comment")

    monkeypatch.setattr(triage_module, "post_trusted_issue_comment", failing_post)
    _runner_off, seen_off, _ = _run_direct(tmp_path, monkeypatch, triage=False)
    runner, seen, _ = _run_direct(
        tmp_path, monkeypatch, triage=True, codex_outputs=[_triage_output()], quiet=False
    )
    assert _turn_identity(seen[0]) == _turn_identity(seen_off[0])
    assert _triage_comments(runner) == []
    assert "posting the triage comment failed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# triage-first-dispatch-hook
# ---------------------------------------------------------------------------


def test_dry_runs_never_arm_the_hook(tmp_path, monkeypatch):
    for run in (_run_direct, _run_plan_first):
        runner, seen, usage = run(tmp_path, monkeypatch, triage=True, dry_run=True)
        assert len(seen) == 1
        assert usage._dispatch_hook_state == "unarmed"
        assert _codex_calls(runner) == 0 and _triage_comments(runner) == []


def test_runs_without_triage_never_arm_the_hook(tmp_path, monkeypatch):
    runner, _seen, usage = _run_direct(tmp_path, monkeypatch, triage=False)
    assert usage._dispatch_hook_state == "unarmed" and usage.records == []
    assert runner.comments == []


def test_hook_fires_once_with_the_real_nested_validated_turn(tmp_path, monkeypatch):
    runner, seen, usage = _run_plan_first(
        tmp_path, monkeypatch, triage=True, codex_outputs=[_triage_output()]
    )
    # A second dispatch on the same context never fires triage again.
    usage.note_agent_dispatch()
    assert usage._dispatch_hook_state == "done"
    assert _codex_calls(runner) == 1 and len(_triage_comments(runner)) == 1
    assert [(record.agent, record.turn_role) for record in usage.records] == [("codex", "reviewer")]
    assert len(seen) == 1


def _parallel_batch(tmp_path, monkeypatch, *, codex_outputs):
    """Arm the real hook, block it mid-triage, and race a sibling dispatch against it."""
    runner = FakeRunner(codex_outputs=list(codex_outputs))
    config = make_config(tmp_path, triage=True)
    usage = _new_usage_context(config)
    in_triage, release = threading.Event(), threading.Event()
    lock_free_during_triage: list[bool] = []
    real_post = triage_module.post_trusted_issue_comment
    real_log = triage_module.log

    def hold(*_args):
        acquired = usage._lock.acquire(blocking=False)
        if acquired:
            usage._lock.release()
        lock_free_during_triage.append(acquired)
        in_triage.set()
        assert release.wait(10)

    # Appended inside the hook, before it returns and the completion Event is set.
    triage_finished: list[str] = []

    def blocking_post(*args, **kwargs):
        hold()
        result = real_post(*args, **kwargs)
        triage_finished.append("posted")
        return result

    def blocking_failure_log(config_arg, message):
        if "advisory triage failed" in message:
            hold()
            result = real_log(config_arg, message)
            triage_finished.append("failed")
            return result
        return real_log(config_arg, message)

    monkeypatch.setattr(triage_module, "post_trusted_issue_comment", blocking_post)
    monkeypatch.setattr(triage_module, "log", blocking_failure_log)
    assert triage_module.arm_issue_triage(
        runner, config=config, usage_context=usage, issue_context=_issue(), operator_route="plan-first"
    )
    started: dict[str, tuple[bool, int]] = {}

    def role_turn(name):
        usage.note_agent_dispatch()
        # What this role turn observes when it would invoke its backend.
        started[name] = (usage._dispatch_hook_done.is_set(), len(triage_finished))

    owner = threading.Thread(target=role_turn, args=("owner",))
    owner.start()
    assert in_triage.wait(10)
    sibling = threading.Thread(target=role_turn, args=("sibling",))
    sibling.start()
    time.sleep(0.2)
    assert "sibling" not in started, "a sibling dispatched before triage completed"
    release.set()
    owner.join(10)
    sibling.join(10)
    assert not owner.is_alive() and not sibling.is_alive(), "deadlock"
    assert started["sibling"] == (True, 1), "the sibling dispatched before triage finished"
    assert started["owner"] == (True, 1)
    assert lock_free_during_triage == [True]
    return runner, usage


def test_parallel_sibling_waits_for_triage_and_the_nested_turn_records_usage(tmp_path, monkeypatch):
    runner, usage = _parallel_batch(tmp_path, monkeypatch, codex_outputs=[_triage_output()])
    assert usage._dispatch_hook_state == "done" and usage._dispatch_hook_done.is_set()
    assert [record.turn_role for record in usage.records] == ["reviewer"]
    assert len(_triage_comments(runner)) == 1


def test_parallel_sibling_proceeds_after_a_failed_triage(tmp_path, monkeypatch):
    runner, usage = _parallel_batch(tmp_path, monkeypatch, codex_outputs=[])
    assert usage._dispatch_hook_state == "done" and usage._dispatch_hook_done.is_set()
    assert _triage_comments(runner) == []


def test_a_raising_hook_still_releases_waiting_siblings():
    usage = RunUsageContext(run_id="run", summary_path=Path("unused.json"))
    entered, release = threading.Event(), threading.Event()

    def hook():
        entered.set()
        release.wait(10)
        raise RuntimeError("triage backend crashed")

    assert usage.arm_pre_first_dispatch_hook(hook)
    assert not usage.arm_pre_first_dispatch_hook(lambda: None)
    errors: list[BaseException] = []
    sibling_started = threading.Event()

    def owner():
        try:
            usage.note_agent_dispatch()
        except RuntimeError as exc:
            errors.append(exc)

    def sibling():
        usage.note_agent_dispatch()
        sibling_started.set()

    first = threading.Thread(target=owner)
    first.start()
    assert entered.wait(10)
    second = threading.Thread(target=sibling)
    second.start()
    assert not sibling_started.wait(0.2)
    release.set()
    first.join(10)
    second.join(10)
    assert sibling_started.is_set() and len(errors) == 1
    assert usage._dispatch_hook_state == "done"


def test_owner_thread_nested_dispatch_passes_through():
    usage = RunUsageContext(run_id="run", summary_path=Path("unused.json"))
    nested: list[str] = []

    def hook():
        usage.note_agent_dispatch()  # the nested validated turn's own dispatch
        nested.append("dispatched")

    usage.arm_pre_first_dispatch_hook(hook)
    usage.note_agent_dispatch()
    usage.note_agent_dispatch()
    assert nested == ["dispatched"]


# ---------------------------------------------------------------------------
# triage-staged-child-recorded
# ---------------------------------------------------------------------------


def _fresh_child(disposition, *, handoff=None):
    phase = PlanPhase(
        title="Child", scope="Implement child.", non_goals="No unrelated work.",
        dependency_notes="No dependencies.", rollout_risk="low",
        validation="Run focused tests.", parent_context="Approved slice.",
        automation="agent-pr", stage_id="stage-one", position=1,
        deliverables=("Child",), non_goals_items=("Unrelated work",),
        acceptance_criteria=("Tests pass",), compatibility_constraints=("Preserve API",),
        covered_scope_item_ids=("scope-1",), execution_disposition=disposition,
        disposition_rationale="Reviewed route.",
    )
    route = execution_policy.resolve_child_execution_route(
        phase, topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    issue = _issue(title="Child", body="How should the child work?", url="child-url")
    parent = replace(issue, number=55, title="Parent", url="parent-url")
    fresh = SimpleNamespace(
        parent_issue=55, plan_hash="plan-hash", plan_subject="plan-subject",
        approved_plan="Approved parent plan", parent_plan_context=SimpleNamespace(
            matrix_available=False, risk_test_matrix_payload=None
        ), recommendation=SimpleNamespace(
            strategy="staged", identity=lambda: {"recommendation_sha256": "digest"},
            child_stages=(phase,),
        ), decomposition=SimpleNamespace(phases=(phase,)),
        created=CreatedPhaseIssue(phase=phase, issue_url=issue.url, issue_number=56),
        phase_index=1, stage_id="stage-one", handoff=handoff, overrides=(), route=route,
    )
    return fresh, issue, parent


def _recorded_handoff(disposition):
    return PhaseImplementationHandoffMetadata(
        parent_issue=55, plan_hash="plan-hash", mode="implement-by-phase",
        phase_index=1, phase_title="Child", automation="agent-pr",
        child_issue_number=56, child_issue_url="child-url", strategy="staged",
        topology_source="approved-plan-v1", execution_strategy_contract_version=1,
        recommendation_digest="digest", stage_id="stage-one", plan_subject="plan-subject",
        execution_disposition=disposition,
    )


@pytest.mark.parametrize(
    ("disposition", "plan_first", "seam"),
    [
        (EXECUTION_DISPOSITION_DIRECT, False, "_dispatch_decomposition_child"),
        (EXECUTION_DISPOSITION_PLANNING, True, "_run_plan_first_loop"),
    ],
    ids=["direct-implementation", "requires-child-planning"],
)
def test_recorded_staged_child_route_wins_without_a_triage_call(
    tmp_path, monkeypatch, capsys, disposition, plan_first, seam
):
    fresh, issue, parent = _fresh_child(disposition, handoff=_recorded_handoff(disposition))
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop_module), "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop_module), "_resolve_fresh_child_provenance", lambda **_: fresh)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop_module), "prepare_agent_memory", lambda *a, **k: None)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop_module), "resolve_canonical_pr_for_issue", lambda *a, **k: None)
    runner = FakeRunner(codex_outputs=[_triage_output("discuss-first")])
    seen: list[dict] = []
    monkeypatch.setattr(issue_loop_module, seam, _record_dispatch(seen, runner, raise_stop=False))
    config = make_config(tmp_path, triage=True, quiet=False)
    usage = _new_usage_context(config)
    assert run_issue_loop(
        runner, issue_number=56, config=config, plan_first=plan_first, usage_context=usage
    ) == 0
    assert len(seen) == 1
    assert _codex_calls(runner) == 0 and _triage_comments(runner) == []
    assert usage._dispatch_hook_state == "unarmed"
    assert f"--triage skipped; the recorded child route `{disposition}` wins" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# triage-startup-guards-first
# ---------------------------------------------------------------------------


def _refuse(name):
    def refusal(*_args, **_kwargs):
        raise AgentLoopError(f"refused by {name}")

    return refusal


def _assert_no_triage(runner, usage):
    assert _codex_calls(runner) == 0
    assert _triage_comments(runner) == []
    assert usage._dispatch_hook_state in {"unarmed", "armed"}
    assert usage.records == []


def _guarded_run(tmp_path, monkeypatch, *, plan_first=False, **config_overrides):
    runner = FakeRunner(codex_outputs=[_triage_output()])
    config = make_config(tmp_path, triage=True, **config_overrides)
    _patch_issue_startup(monkeypatch, _issue())
    seen: list[dict] = []
    monkeypatch.setattr(
        issue_loop_module, "_run_validated_agent", _record_dispatch(seen, runner, raise_stop=True)
    )
    usage = _new_usage_context(config)
    return runner, config, usage, seen


@pytest.mark.parametrize(
    ("plan_first", "patch_targets", "patch_name"),
    [
        (False, (discuss_loop, pr_loop, issue_loop_module), "validate_open_issue"),
        (False, (execution_policy, issue_loop_module), "_refuse_plain_mode_over_planning"),
        (False, (issue_loop_module,), "preflight_managed_ci_creation"),
        (True, (plan_first_loop_module,), "_ensure_parallel_reviewer_workdirs"),
    ],
    ids=["closed-issue", "plain-mode-over-planning", "managed-ci-preflight", "review-parallel-workdirs"],
)
def test_refusals_precede_triage(tmp_path, monkeypatch, plan_first, patch_targets, patch_name):
    overrides = {}
    if patch_name == "preflight_managed_ci_creation":
        monkeypatch.setattr(issue_loop_module, "_freeze_prompt_architecture", lambda _runner, config: config)
        overrides = {"managed_ci": True}
    if patch_name == "_ensure_parallel_reviewer_workdirs":
        overrides = {"review_parallel": True}
    runner, config, usage, seen = _guarded_run(tmp_path, monkeypatch, plan_first=plan_first, **overrides)
    patch_attr_in(monkeypatch, patch_targets, patch_name, _refuse(patch_name))
    with pytest.raises(AgentLoopError, match=f"refused by {patch_name}"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=plan_first, usage_context=usage)
    assert seen == []
    _assert_no_triage(runner, usage)


def test_board_floor_refusal_precedes_triage(tmp_path, monkeypatch):
    import coding_review_agent_loop.plan_verification as plan_verification
    import coding_review_agent_loop.reviewer_floor as reviewer_floor

    runner, config, usage, seen = _guarded_run(tmp_path, monkeypatch)
    monkeypatch.setattr(reviewer_floor, "board_floor_enabled", lambda _config: True)
    monkeypatch.setattr(issue_loop_module, "_issue_floor_plan_owner", lambda *a, **k: _issue())
    monkeypatch.setattr(plan_verification, "enforce_issue_board_floor", _refuse("board floor"))
    with pytest.raises(AgentLoopError, match="refused by board floor"):
        run_issue_loop(runner, issue_number=56, config=config, usage_context=usage)
    assert seen == []
    _assert_no_triage(runner, usage)


def test_reconciled_integration_child_returns_before_triage(tmp_path, monkeypatch):
    runner, config, usage, seen = _guarded_run(tmp_path, monkeypatch)
    monkeypatch.setattr(issue_loop_module, "reconcile_merged_integration_child", lambda *a, **k: True)
    assert run_issue_loop(runner, issue_number=56, config=config, usage_context=usage) == 0
    assert seen == []
    _assert_no_triage(runner, usage)


def test_direct_closing_contract_refusals_precede_triage(tmp_path, monkeypatch):
    # An unsupported supersede: no recovered contract to widen.
    runner, config, usage, seen = _guarded_run(
        tmp_path, monkeypatch, supersede_expected_closing_contract=True,
        expected_closing_issue_ids=(56, 57),
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, usage_context=usage)
    assert seen == []
    _assert_no_triage(runner, usage)

    # A contract that names the staged parent.
    runner, config, usage, seen = _guarded_run(tmp_path, monkeypatch, expected_closing_issue_ids=(55, 56))
    monkeypatch.setattr(issue_loop_module, "_infer_staged_parent_issue", lambda _issue: 55)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, usage_context=usage)
    assert seen == []
    _assert_no_triage(runner, usage)


def _resolved_pr(plan_hash):
    return SimpleNamespace(
        pr_number=77, source="canonical", evidence_summary="handoff",
        metadata=SimpleNamespace(
            flow="approved-plan-implementation", plan_hash=plan_hash, expected_closing_issue_ids=(56,),
        ),
    )


def test_plan_first_plan_hash_mismatch_refuses_before_triage(tmp_path, monkeypatch):
    runner, config, usage, seen = _guarded_run(tmp_path, monkeypatch)
    monkeypatch.setattr(issue_loop_module, "resolve_canonical_pr_for_issue", lambda *a, **k: _resolved_pr("b" * 16))
    monkeypatch.setattr(
        issue_loop_module, "find_latest_issue_pr_handoff",
        lambda *a, **k: SimpleNamespace(flow="approved-plan-implementation", plan_hash="a" * 16),
    )
    monkeypatch.setattr(
        issue_loop_module, "recover_approved_plan_context",
        lambda *a, **k: SimpleNamespace(is_available=False, canonical_text=None),
    )
    with pytest.raises(AgentLoopError, match="reconstructable approved plan has hash"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, usage_context=usage)
    _assert_no_triage(runner, usage)


def test_existing_pr_resumption_skips_triage_and_logs_the_recorded_route(tmp_path, monkeypatch, capsys):
    runner, config, usage, _seen = _guarded_run(tmp_path, monkeypatch, quiet=False)
    resumed: list[dict] = []
    monkeypatch.setattr(issue_loop_module, "resolve_canonical_pr_for_issue", lambda *a, **k: _resolved_pr(None))
    monkeypatch.setattr(issue_loop_module, "run_pr_loop", _record_dispatch(resumed, runner, raise_stop=False))
    assert run_issue_loop(runner, issue_number=56, config=config, usage_context=usage) == 0
    assert len(resumed) == 1
    assert usage._dispatch_hook_state == "unarmed"
    _assert_no_triage(runner, usage)
    assert "--triage skipped; the recorded route of existing PR #77 wins" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# CLI and recovery-command wiring
# ---------------------------------------------------------------------------


def test_triage_flags_parse_into_config_and_require_triage(tmp_path, capsys):
    from coding_review_agent_loop.cli import build_parser, config_from_args, main

    parser = build_parser()
    base = ["issue", "56", "--repo", "OWNER/REPO"]
    default = config_from_args(parser.parse_args(base), FakeRunner())
    assert (default.triage, default.triage_model, default.triage_history) == (False, "", None)
    history = tmp_path / "history.json"
    args = parser.parse_args(
        [*base, "--triage", "--triage-model", "gpt-triage", "--triage-history", str(history)]
    )
    config = config_from_args(args, FakeRunner())
    assert (config.triage, config.triage_model, config.triage_history) == (True, "gpt-triage", history)
    assert main([*base, "--triage-model", "gpt-triage"]) == 1
    assert "--triage-model and --triage-history require --triage" in capsys.readouterr().err


def test_recovery_commands_never_replay_the_one_shot_triage_flags(tmp_path):
    from coding_review_agent_loop.managed_ci import _render_recovery_command

    config = make_config(
        tmp_path, triage=True, triage_model="gpt-triage",
        invocation_argv=(
            "agent-loop", "issue", "56", "--triage", "--triage-model", "gpt-triage",
            "--triage-history", "history.json", "--reviewer", "codex",
        ),
    )
    for target in ("issue", "pr"):
        rendered = _render_recovery_command(
            config, target=target, identifier=7, managed_ci=True, preserve_managed_options=True,
        )
        assert "--triage" not in rendered and "history.json" not in rendered
        assert "gpt-triage" not in rendered and "--reviewer codex" in rendered


# ---------------------------------------------------------------------------
# triage-first-dispatch-hook: a real plan-first resume whose first dispatch is
# the parallel reviewer batch
# ---------------------------------------------------------------------------

TRIAGE_PROMPT_PREFIX = "Triage GitHub issue #56"


class _BarrierProbeRunner(FakeRunner):
    """Records, at every backend spawn, whether the triage hook had completed."""

    def __init__(self, *, raise_triage=False, **kwargs):
        super().__init__(**kwargs)
        self.usage_context = None
        self.raise_triage = raise_triage
        self.spawns: list[tuple[str, bool, bool]] = []
        self._spawn_lock = threading.Lock()

    def run_with_log(self, args, *, cwd, **kwargs):
        cmd = [str(arg) for arg in args]
        backend = {"codex": "codex", "gemini": "gemini", "claude": "claude"}.get(cmd[0] if cmd else "")
        if backend is not None and self.usage_context is not None:
            text = " ".join(cmd) + str(kwargs.get("input_text") or "")
            is_triage = TRIAGE_PROMPT_PREFIX in text
            with self._spawn_lock:
                self.spawns.append(
                    (backend, is_triage, self.usage_context._dispatch_hook_done.is_set())
                )
            if is_triage and self.raise_triage:
                raise RuntimeError("triage backend crashed")
            if not is_triage:
                time.sleep(0.05)  # widen the window in which an early sibling would show
        return super().run_with_log(args, cwd=cwd, **kwargs)


def _resume_parallel_plan_with_triage(tmp_path, *, raise_triage):
    from agent_loop_helpers import structured_plan_review, structured_plan_state

    runner = _BarrierProbeRunner(
        raise_triage=raise_triage,
        claude_outputs=[structured_plan_state()], codex_outputs=[], gemini_outputs=[],
        authenticated_actor=("agent-loop", 4242), serve_rest_issue_comments=True,
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True, agent_max_retries=0
    )
    # First invocation: the planner posts round 1, then both reviewers fail.
    with pytest.raises(Exception):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert runner.claude_outputs == []
    planner_comments = len(runner.comments)

    # Resume with --triage: the first dispatch is the parallel reviewer batch.
    runner.codex_outputs = (
        [] if raise_triage else [_triage_output("discuss-first")]
    ) + [structured_plan_review(summary="Codex resumed review.")]
    runner.gemini_outputs = [
        structured_plan_review(summary="Gemini resumed review.", reviewer="Google Gemini")
    ]
    resumed = replace(config, triage=True)
    usage = _new_usage_context(resumed)
    runner.usage_context = usage
    assert run_issue_loop(
        runner, issue_number=56, config=resumed, plan_first=True, usage_context=usage
    ) == 0
    return runner, usage, planner_comments


@pytest.mark.parametrize("raise_triage", [False, True], ids=["triage-succeeds", "triage-backend-raises"])
def test_plan_first_resume_parallel_batch_waits_for_the_real_triage_turn(tmp_path, raise_triage):
    runner, usage, planner_comments = _resume_parallel_plan_with_triage(
        tmp_path, raise_triage=raise_triage
    )
    triage_spawns = [spawn for spawn in runner.spawns if spawn[1]]
    reviewer_spawns = [spawn for spawn in runner.spawns if not spawn[1]]
    # Triage ran exactly once, as the first backend spawn, before completion was signalled.
    assert triage_spawns == [("codex", True, False)]
    assert runner.spawns[0] == triage_spawns[0]
    # Both reviewers of the parallel batch ran, and only after triage completed.
    assert sorted(backend for backend, _triage, _done in reviewer_spawns) == ["codex", "gemini"]
    assert all(done for _backend, _triage, done in reviewer_spawns)
    assert usage._dispatch_hook_state == "done"
    reviews = [comment for comment in runner.comments[planner_comments:] if "resumed review" in comment]
    assert len(reviews) == 2
    triage_comments = _triage_comments(runner)
    if raise_triage:
        # The raising backend leaves no triage record and no comment; reviewers still ran.
        assert triage_comments == []
        assert len(usage.records) == 2
    else:
        # Exactly one nested triage usage record, ahead of the two reviewer records.
        assert len(triage_comments) == 1
        assert runner.comments.index(triage_comments[0]) < runner.comments.index(reviews[0])
        assert len(usage.records) == 3 and usage.records[0].agent == "codex"
