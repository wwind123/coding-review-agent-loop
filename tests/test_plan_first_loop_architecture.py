"""Plan-first loop tests: architecture-impact contract in planning rounds (#1543)."""

import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.round_state import _decode_round_metadata
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

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
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


def test_staged_acknowledgement_repair_keeps_the_degraded_assessment_and_record(
    tmp_path, monkeypatch
):
    outcome, codex_records, repair_calls, requirement = _ack_staged_run(
        tmp_path, monkeypatch, repaired_impact=None
    )
    assert outcome == 0
    assert repair_calls, "the acknowledgement repair must run through the flow"
    # The repair source is the canonical accepted text: the degraded
    # assessment was already removed at acceptance.
    assert '"modified"' not in repair_calls[0]
    # Both the first record and the superseding re-post carry a null
    # assessment plus the original degradation record.
    assert len(codex_records) >= 2
    for record in codex_records:
        assert record.architecture_impact is None
        assert [r.outcome for r in record.architecture_impact_degradations] == [
            "degraded-to-undetermined"
        ]
    assert codex_records[-1].surfaced_reviewer_requirement_ids == (requirement.requirement_id,)


_ACK_REFUSAL = {"unchanged": "must not introduce one", "none": "must be `changed` or `unchanged`"}


def _assert_refused_by(repair_results, status):
    assert repair_results
    for _text, validated, attempts in repair_results:
        assert validated is None
        assert _ACK_REFUSAL[status] in attempts[-1].diagnostic


@pytest.mark.parametrize("status", ["unchanged", "none"])
def test_staged_acknowledgement_repair_cannot_add_an_assessment(tmp_path, monkeypatch, status):
    outcome, codex_records, repair_calls, _requirement = _ack_staged_run(
        tmp_path, monkeypatch, repaired_impact=None, repaired_status=status, max_rounds=1
    )
    assert repair_calls, "the acknowledgement repair must run through the flow"
    # The refusal comes from the absence pin or strict validation.
    _assert_refused_by(_ack_staged_run.repair_results, status)
    # The refused repair is never published: no record asserts an assessment,
    # and the primary's approval is never carried as acknowledged.
    assert outcome != 0
    for record in codex_records:
        assert record.architecture_impact is None
        assert record.surfaced_reviewer_requirement_ids == ()


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

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
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


def test_nonstaged_acknowledgement_repair_persists_the_bound_record(
    tmp_path, monkeypatch
):
    outcome, codex_records, repair_calls = _ack_nonstaged_run(tmp_path, monkeypatch)
    assert outcome == 0, outcome
    assert repair_calls, "the acknowledgement repair must run through the flow"
    # The comment posted before the gate comes from the accepted carrier.  The
    # repaired approval is then persisted with its requirement binding
    # (#1377), so a restart or qualification can bind it; both records keep
    # the carrier's degraded assessment.
    assert len(codex_records) == 2
    carrier, repaired = codex_records
    assert carrier.surfaced_reviewer_requirement_ids == ()
    assert len(repaired.surfaced_reviewer_requirement_ids) == 1
    assert repaired.plan_candidate_key is not None
    for record in codex_records:
        assert record.state == "approved"
        assert record.architecture_impact is None
        assert [r.outcome for r in record.architecture_impact_degradations] == [
            "degraded-to-undetermined"
        ]
    # The persisted binding is what a restart and strict qualification read.
    from coding_review_agent_loop.github import IssueComment
    from coding_review_agent_loop.plan_verification import (
        classify_plan_approval,
        require_recoverable_plan_approval,
    )

    runner, config, requirement = _ack_nonstaged_run.state
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="Issue", body="Issue body", url=None,
        comments=tuple(
            IssueComment(author=c["author"]["login"], created_at=c["createdAt"], body=c["body"])
            for c in runner.issue_comments
        ),
        human_requirements=(requirement,),
    )
    assert classify_plan_approval(
        issue.comments, config=config, issue_number=56, human_requirements=(requirement,),
    ).state == "approved"
    require_recoverable_plan_approval(
        None, config=config, issue_context=issue, comments=issue.comments,
        trusted_actor=None, error_message="not recoverable",
    )
    # An unchanged restart reuses the bound approvals: no reviewer runs again.
    start = len(runner.commands)
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass
    assert not any(
        cmd[0] in {"codex", "gemini"} for cmd, _cwd in runner.commands[start:]
    )


@pytest.mark.parametrize("status", ["unchanged", "none"])
def test_nonstaged_acknowledgement_repair_cannot_add_an_assessment(
    tmp_path, monkeypatch, status
):
    outcome, codex_records, repair_calls = _ack_nonstaged_run(
        tmp_path, monkeypatch, repaired_status=status
    )
    assert repair_calls
    _assert_refused_by(_ack_nonstaged_run.repair_results, status)
    assert outcome != 0
    assert len(codex_records) == 1
    assert codex_records[0].architecture_impact is None


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

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)
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

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair), patch_object_in((architecture_contract, pr_loop, plan_first_loop_module), "_pin_acknowledgement_repair", spy_pin
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


def test_resumed_authoritative_plan_review_keeps_metadata_carrier_and_is_not_repaired(
    tmp_path, monkeypatch
):
    run = _ack_plan_resume_run(
        tmp_path, monkeypatch, parallel=False,
        codex_impact=_DEG_UNCORROBORATED, repaired_impact=None,
    )
    assert run.resumed_without_reinvocation
    assert [record.phase for record in run.records] == ["authoritative"]
    # The carrier is rebuilt from round metadata, never from rendered prose.
    (carrier,) = run.carriers
    _assert_degraded_carrier(carrier)
    # The existing substance gate refuses the prose source before any repair
    # model runs, and nothing new is posted for the reviewer.
    assert run.repair_calls == []
    assert run.outcome != 0
    assert len(run.records) == 1


def test_resumed_publication_plan_review_accepts_acknowledgement_only_repair(
    tmp_path, monkeypatch
):
    run = _ack_plan_resume_run(
        tmp_path, monkeypatch, parallel=True,
        codex_impact=_DEG_UNCORROBORATED, repaired_impact=None,
    )
    assert run.resumed_without_reinvocation
    # The repaired approval is persisted with its requirement binding (#1377).
    assert [record.phase for record in run.records] == ["publication", "authoritative"]
    assert run.records[0].surfaced_reviewer_requirement_ids == ()
    assert len(run.records[-1].surfaced_reviewer_requirement_ids) == 1
    _assert_degraded_carrier(run.records[-1])
    (carrier,) = run.carriers
    _assert_degraded_carrier(carrier)
    assert len(run.repair_calls) == 1
    assert '"modified"' not in run.repair_calls[0]
    ((_repaired, pinned),) = run.pinned
    _assert_degraded_carrier(pinned)
    assert run.outcome == 0


@pytest.mark.parametrize(
    "status,diagnostic",
    [("unchanged", "must not introduce one"), ("none", "must be `changed` or `unchanged`")],
)
def test_resumed_publication_plan_review_repair_cannot_add_an_assessment(
    tmp_path, monkeypatch, status, diagnostic
):
    run = _ack_plan_resume_run(
        tmp_path, monkeypatch, parallel=True, codex_impact=_DEG_UNCORROBORATED,
        repaired_impact={"status": status, "rationale": "No architectural change."},
    )
    assert run.resumed_without_reinvocation
    (carrier,) = run.carriers
    _assert_degraded_carrier(carrier)
    # The refusal comes from the absence pin (or, for `none`, from strict
    # validation of the new repair output), not from an unrelated failure.
    ((_text, validated, attempts),) = run.repair_results
    assert validated is None
    assert diagnostic in attempts[-1].diagnostic
    assert run.outcome != 0
    assert all(record.architecture_impact is None for record in run.records)


def test_resumed_publication_corroborated_review_refuses_a_changed_assessment(
    tmp_path, monkeypatch
):
    # A corroborated `modified` is accepted as `changed` with its record.  A
    # repair that rewrites the accepted assessment is refused; the accepted
    # status is pinned in preservation, and the equality check backs it up.
    changed = dict(_DEG_CORROBORATED, status="unchanged")
    run = _ack_plan_resume_run(
        tmp_path, monkeypatch, parallel=True,
        codex_impact=_DEG_CORROBORATED, repaired_impact=changed,
    )
    (carrier,) = run.carriers
    assert carrier.architecture_impact.status == "changed"
    assert [r.outcome for r in carrier.architecture_impact_degradations] == [
        "normalized-to-changed"
    ]
    ((_text, validated, attempts),) = run.repair_results
    assert validated is None
    assert "architecture_impact.status" in attempts[-1].diagnostic
    ((_repaired, pinned),) = run.pinned
    assert pinned is None
    assert run.outcome != 0
    assert all(
        record.architecture_impact is None or record.architecture_impact["status"] == "changed"
        for record in run.records
    )


def test_resumed_publication_genuine_omission_cannot_gain_an_assessment(
    tmp_path, monkeypatch
):
    run = _ack_plan_resume_run(
        tmp_path, monkeypatch, parallel=True, codex_impact=None,
        repaired_impact={"status": "unchanged", "rationale": "No architectural change."},
    )
    (carrier,) = run.carriers
    assert carrier.architecture_impact is None
    assert carrier.architecture_impact_degradations == ()
    ((_text, validated, attempts),) = run.repair_results
    assert validated is None
    assert "must not introduce one" in attempts[-1].diagnostic
    assert run.outcome != 0
