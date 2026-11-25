"""Plan-first loop tests: primary-phase stall stop and streak retirement (#1543)."""

import coding_review_agent_loop.issue_recovery as issue_recovery
import coding_review_agent_loop.plan_execution as plan_execution
import hashlib
import json
import re
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.errors import AgentInvocationError
from coding_review_agent_loop.github import HumanReviewRequirement
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.protocol import validate_structured_plan_state
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.integration_close as integration_close
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.triage as triage
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    FakeRunner,
    _PlanDiagnosticRunner,
    _invalid_plan_scheduler_comment,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1103_stalled_runner,
    _m943_decoded,
    _plan_round_records,
    _plan_with_requirement_disposition,
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
import coding_review_agent_loop.validated_agent as validated_agent_module


# --- One plan no-progress detector (#2098) ---------------------------------
#
# The plan step-back escalation and primary-stall stops are gone.  The shared
# detector stops with exit 4 at round start, before any record or agent turn,
# when K completed rounds share the plan subject and the non-empty blocking
# IDs and the live plan and IDs still equal them.

from coding_review_agent_loop.agents.registry import agent_display_name  # noqa: E402,F401
from coding_review_agent_loop.errors import HumanDecisionRequiredError  # noqa: E402
from coding_review_agent_loop.no_progress import (  # noqa: E402
    NO_PROGRESS_PREFIX_TEXT,
    plan_no_progress_observations as _real_plan_observations,
)
from agent_loop_helpers import structured_plan_revision, structured_plan_state  # noqa: E402

_NP_KEEP_ITEM = [{"item_id": "item-1", "disposition": "blocking"}]


def _np_reviews(rounds, *, reviewer="OpenAI Codex"):
    """Round 1 blocks with item-1; every later review keeps item-1 blocking."""
    first = structured_plan_review(
        state="blocking", blocking_plan_issues=["Close the gap."], reviewer=reviewer
    )
    keep = structured_plan_review(
        state="blocking", prior_plan_item_dispositions=_NP_KEEP_ITEM, reviewer=reviewer
    )
    return [first] + [keep] * (rounds - 1)


def _np_revision(steps=("Update the code.", "Run the relevant tests.")):
    """A legacy full-state planner revision that keeps item-1 open."""
    return structured_plan_revision(
        plan_steps=list(steps),
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "blocking", "note": "Out of scope."}
        ],
    )


def _np_all_reviewers_run(tmp_path, *, revisions, reviews, max_rounds=8, comments=()):
    runner = _FakeRunner(
        issue_comments=list(comments),
        claude_outputs=[structured_plan_state(), *revisions],
        codex_outputs=list(reviews),
    )
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=max_rounds)
    error = None
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except (AgentLoopError, HumanDecisionRequiredError) as exc:
        error = exc
    return runner, config, error


def _np_diagnostics(runner, round_number):
    return [
        comment["body"]
        for comment in runner.issue_comments
        if comment["body"].startswith(f"Plan review scheduling diagnostic (round {round_number})")
    ]


def _np_subject(runner, round_number):
    return next(
        record.subject
        for record in _plan_round_records(runner)
        if record.role == "coder" and record.round_number == round_number
    )


def _np_assert_no_progress(error, *, subject, rounds, k=3):
    assert isinstance(error, HumanDecisionRequiredError), error
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)
    assert HumanDecisionRequiredError.EXIT_CODE == 4
    message = str(error)
    assert message.startswith(NO_PROGRESS_PREFIX_TEXT)
    assert f"rounds {rounds[0]}-{rounds[-1]}" in message
    assert f"the same plan `{subject}`" in message
    assert f"--no-progress-rounds {k}" in message
    assert "- [item-1] Close the gap." in message
    assert "Plan size: " in message and " characters" in message
    # Not one of the retired plan stops.
    assert "primary plan reviewer has blocked" not in message
    assert "stepped back at round" not in message


def test_plan_no_progress_stops_after_k_unchanged_rounds_under_all_reviewers(tmp_path):
    """`plan-k-unchanged`: ordinary all-reviewers records (no scheduler, no key)."""
    runner, _config, error = _np_all_reviewers_run(
        tmp_path, revisions=[_np_revision()] * 4, reviews=_np_reviews(4)
    )

    subject = _np_subject(runner, 2)
    _np_assert_no_progress(error, subject=subject, rounds=(2, 4))
    # Four reviews and four planner revisions; round 5 starts no agent turn.
    assert _m1103_agent_calls(runner) == ["claude", "codex"] * 4 + ["claude"]
    records = _plan_round_records(runner)
    assert not any(record.round_number == 5 and record.role != "coder" for record in records)
    assert not any(record.scheduler_contract for record in records)
    assert not any(record.plan_candidate_key for record in records if record.role == "reviewer")
    diagnostics = _np_diagnostics(runner, 5)
    assert len(diagnostics) == 1
    assert "AGENT_LOOP_META" not in diagnostics[0]


def test_plan_no_progress_resume_stops_again_without_any_turn(tmp_path):
    """`plan-k-unchanged` on resume: a pure derivation from the persisted rounds."""
    runner, config, _error = _np_all_reviewers_run(
        tmp_path, revisions=[_np_revision()] * 4, reviews=_np_reviews(4)
    )
    history = list(runner.issue_comments)
    rerun = _FakeRunner(issue_comments=list(history))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    _np_assert_no_progress(excinfo.value, subject=_np_subject(runner, 2), rounds=(2, 4))
    assert _m1103_agent_calls(rerun) == []
    assert len(_plan_round_records(rerun)) == len(_plan_round_records(runner))
    # A larger window continues to review (option 1 of the message).
    proceeding = _FakeRunner(
        issue_comments=list(history),
        codex_outputs=_np_reviews(2)[1:],
    )
    with pytest.raises(AgentLoopError, match="still reported blocking plan issues"):
        run_issue_loop(
            proceeding, issue_number=56,
            config=replace(config, no_progress_rounds=4, max_rounds=5), plan_first=True,
        )
    assert _m1103_agent_calls(proceeding) == ["codex"]


def test_plan_no_progress_reviews_a_changed_live_plan_first(tmp_path):
    """`plan-live-hash-changed`: P,P,P completed and the live plan is P2."""
    changed = _np_revision(("Update the code differently.", "Run the relevant tests."))
    runner, _config, error = _np_all_reviewers_run(
        tmp_path,
        revisions=[_np_revision()] * 3 + [changed],
        reviews=_np_reviews(5),
        max_rounds=5,
    )
    assert not isinstance(error, HumanDecisionRequiredError), error
    assert "still reported blocking plan issues after round 5" in str(error)
    assert _m1103_agent_calls(runner).count("codex") == 5
    assert _np_diagnostics(runner, 5) == []


def test_plan_max_rounds_is_reached_first_when_the_plan_keeps_changing(tmp_path):
    """`plan-max-rounds`: the existing round-limit exit, no no-progress message."""
    revisions = [
        _np_revision((f"Revision {n}.", "Run the relevant tests.")) for n in range(1, 4)
    ]
    runner, _config, error = _np_all_reviewers_run(
        tmp_path, revisions=revisions, reviews=_np_reviews(4), max_rounds=4
    )
    assert not isinstance(error, HumanDecisionRequiredError), error
    assert "still reported blocking plan issues after round 4" in str(error)
    assert NO_PROGRESS_PREFIX_TEXT not in str(error)
    assert not any(
        NO_PROGRESS_PREFIX_TEXT in comment["body"] for comment in runner.issue_comments
    )


def test_plan_newly_signed_requirement_restarts_a_keyless_window(tmp_path):
    """`plan-requirement-change`: keyless legacy rounds stop counting."""
    runner, config, _error = _np_all_reviewers_run(
        tmp_path, revisions=[_np_revision()] * 4, reviews=_np_reviews(4)
    )
    signed = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:00:00Z",
        "body": "Also preserve the audit trail.\n\n-- Human Reviewer",
    }
    history = [*runner.issue_comments, signed]
    rerun = _FakeRunner(issue_comments=list(history))
    try:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    except (AgentLoopError, HumanDecisionRequiredError, IndexError) as exc:
        assert not str(exc).startswith(NO_PROGRESS_PREFIX_TEXT), exc
    # The requirement set changed, so the live plan goes to review first.
    assert "codex" in _m1103_agent_calls(rerun)
    assert not any(
        NO_PROGRESS_PREFIX_TEXT in comment["body"]
        for comment in rerun.issue_comments[len(history):]
    )


# Under primary-then-panel every planner revision is a semantic patch, and the
# assembler rejects a payload-identical patch, so a live staged run can never
# return a byte-identical plan.  These tests keep every window rule real
# (reviews, planner records, closures, issue digest, requirement binding and
# the reset floor) and only normalize the observed subject to the live plan,
# which isolates the staged wiring: the history-integrity gate, the reset
# window and the placement before the prelaunch checkpoint.


def _np_patch_observations(monkeypatch):
    def observations(records, **kwargs):
        live = next(
            record.metadata.subject
            for record in sorted(records, key=lambda item: item.index, reverse=True)
            if record.metadata.role == "coder"
        )
        return tuple(
            replace(observation, subject=live)
            for observation in _real_plan_observations(records, **kwargs)
        )

    monkeypatch.setattr(plan_first_loop_module, "plan_no_progress_observations", observations)


def _np_staged_chain(first_round, last_round, *, base):
    """Staged primary reviews keeping item-1 blocking, and chained planner patches."""
    codex, claude = [], []
    for n in range(first_round, last_round + 1):
        codex.append(
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Close the gap."] if n == 1 else None,
                prior_plan_item_dispositions=None if n == 1 else _NP_KEEP_ITEM,
            )
        )
        patch_payload = {
            "schema_version": 1,
            "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1,
            "state": "blocking",
            "summary": f"Keep the gap open, revision {n}.",
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "blocking", "note": "Out of scope."}
            ],
            "base_round_number": n,
            "base_state_identity": base.state_identity,
            "operations": [
                {
                    "op": "replace",
                    "field": "plan_steps",
                    "value": [f"Implement the reviewed scope, revision {n}."],
                }
            ],
        }
        base = AuthenticatedPlanState.from_plan(
            assemble_plan_revision(base, patch_payload), round_number=n + 1
        )
        claude.append(
            json.dumps(patch_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
        )
    return codex, claude, base


def _np_prelaunch_rounds(runner):
    return [
        record.round_number
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]


def test_plan_no_progress_stops_before_the_staged_prelaunch_checkpoint(tmp_path, monkeypatch):
    """`plan-k-unchanged` under primary-then-panel: no checkpoint, no turn, posted once."""
    _np_patch_observations(monkeypatch)
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _np_staged_chain(1, 3, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=8)
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    _np_assert_no_progress(excinfo.value, subject=_np_subject(runner, 4), rounds=(1, 3))
    assert _m1103_agent_calls(runner) == ["claude"] + ["codex", "claude"] * 3
    assert _np_prelaunch_rounds(runner) == [1, 2, 3]
    assert len(_np_diagnostics(runner, 4)) == 1


def test_plan_no_progress_is_skipped_over_degraded_staged_history(tmp_path, monkeypatch):
    """`plan-empty-or-degraded`: a non-intact history never stops and names the class."""
    _np_patch_observations(monkeypatch)
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _np_staged_chain(1, 4, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=list(codex))
    config = _staged_plan_config(tmp_path, max_rounds=3)
    with pytest.raises(AgentLoopError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    history = list(runner.issue_comments)
    from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

    invalid = _invalid_plan_scheduler_comment(
        _np_subject(runner, 3), created_at=f"2026-05-23T00:00:{len(history):02d}Z"
    )
    payload_text = re.search(r"AGENT_LOOP_META: (\S+) -->", invalid["body"]).group(1)
    invalid["body"] = invalid["body"].replace(
        payload_text, encode_mapping(dict(decode_mapping(payload_text), round_number=4))
    )
    history.append(invalid)
    rerun = _FakeRunner(issue_comments=history, codex_outputs=codex[3:], claude_outputs=claude[2:])
    with patch_object_in((agent_failure, integration_close, architecture_contract, validated_agent_module, response_validation, panel_evidence, review_rounds, discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop, issue_recovery, plan_execution,), "log",
        wraps=agent_failure.log,
    ) as logged:
        try:
            run_issue_loop(
                rerun, issue_number=56, config=replace(config, max_rounds=4), plan_first=True
            )
        except (AgentLoopError, IndexError) as exc:
            assert not str(exc).startswith(NO_PROGRESS_PREFIX_TEXT), exc
    assert any(
        "no-progress detector skipped: degraded planning history (invalid)"
        in str(call.args[-1])
        for call in logged.call_args_list
    )
    assert not any(
        NO_PROGRESS_PREFIX_TEXT in comment["body"] for comment in rerun.issue_comments
    )
    # The degraded round still goes to review.
    assert _m1103_agent_calls(rerun)[:1] == ["codex"]


def test_plan_reset_flag_restarts_the_window_and_the_floor_persists(tmp_path, monkeypatch):
    """`plan-reset-window`: the reset round never stops; later resumes keep the floor."""
    _np_patch_observations(monkeypatch)
    fresh, base = _m1103_fresh_base()
    codex, claude, base = _np_staged_chain(1, 2, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=3)
    # Rounds 1 and 2 completed (K-1 identical rounds), round 3 reached the cap.
    codex_three, claude_three, base = _np_staged_chain(3, 3, base=base)
    runner.codex_outputs.extend(codex_three)
    with pytest.raises(AgentLoopError, match="still reported blocking plan issues"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    history = list(runner.issue_comments)

    # Without the reset, rounds 1-3 stop round 4 once round 3 is revised.
    later_codex, later_claude, _end = _np_staged_chain(4, 6, base=base)
    no_reset = _FakeRunner(
        issue_comments=list(history), claude_outputs=list(claude_three),
        codex_outputs=list(later_codex),
    )
    with pytest.raises(HumanDecisionRequiredError, match="rounds 1-3"):
        run_issue_loop(no_reset, issue_number=56, config=replace(config, max_rounds=10), plan_first=True)
    assert "codex" not in _m1103_agent_calls(no_reset)
    reset_run = _FakeRunner(
        issue_comments=list(history),
        claude_outputs=[*claude_three, *later_claude],
        codex_outputs=list(later_codex),
    )
    reset_config = replace(config, max_rounds=10, plan_reset_stall_streak=True)
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_issue_loop(reset_run, issue_number=56, config=reset_config, plan_first=True)
    reset_checkpoints = [
        record.round_number
        for record in _plan_round_records(reset_run)
        if record.phase == "scheduler-prelaunch" and record.scheduler_stall_reset
    ]
    # The resumed round 3 is the invocation's first staged round, so it carries
    # the reset and never stops itself.
    assert reset_checkpoints == [3]
    # Rounds before the reset checkpoint's round are dropped: rounds 3-5 stop 6.
    _np_assert_no_progress(excinfo.value, subject=_np_subject(reset_run, 6), rounds=(3, 5))
    assert _m1103_agent_calls(reset_run).count("codex") == 2

    # A later resume without the flag keeps the persisted floor.
    flagless = _FakeRunner(issue_comments=list(reset_run.issue_comments))
    with pytest.raises(HumanDecisionRequiredError) as again:
        run_issue_loop(flagless, issue_number=56, config=replace(config, max_rounds=10), plan_first=True)
    assert "rounds 3-5" in str(again.value)
    assert _m1103_agent_calls(flagless) == []


def _m1103_narrative_runner():
    """Primary gate, panel blocker, then a narrative-only recommendation patch."""
    fresh = structured_v1_plan_state()
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    recommendation = json.loads(fresh.split("\n<!--", 1)[0])["execution_recommendation"]
    recommendation = dict(
        recommendation,
        rationale="The reviewed scope is one coherent delivery; reworded.",
        caveats=["A narrative caveat."],
    )
    patch_payload = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Reword the recommendation rationale.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "execution_recommendation", "value": recommendation}
        ],
    }
    patch_text = (
        json.dumps(patch_payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="The rationale is unclear.",
                blocking_plan_issues=["Reword the recommendation rationale."],
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


def _m1103_assert_narrative_remediation(runner):
    prelaunch = {
        record.round_number: record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    }
    remediation = prelaunch[3]
    assert remediation.scheduler_phase == "remediation"
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert remediation.scheduler_force_full is False
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    # The candidate key still changed, so no approval was carried across it:
    # the final sweep re-covers the secondary lacking an exact-key approval.
    assert remediation.plan_candidate_key != prelaunch[2].plan_candidate_key
    assert prelaunch[4].scheduler_phase == "final-secondary-sweep"
    assert prelaunch[4].scheduler_selected_reviewers == ("Antigravity",)
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )


def test_narrative_only_recommendation_patch_stays_owner_scoped_after_the_panel(tmp_path):
    """`postpanel-prose-remediation`: no full board and no automatic latch."""
    runner = _m1103_narrative_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    _m1103_assert_narrative_remediation(runner)


def test_resume_after_a_narrative_only_patch_reconstructs_the_narrow_classification(tmp_path):
    """`resume-reconstructs-narrow`: the restart classifies as the live run does."""
    runner = _m1103_narrative_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = architecture_contract.post_issue_comment
    audits = {"count": 0}

    def interrupt_before_remediation_checkpoint(*args, **kwargs):
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

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    _m1103_assert_narrative_remediation(runner)


# --- Stall streak retirement: issue-text edit and reset flag (#1112) ---------


def test_all_reviewers_policy_writes_no_digest_or_reset_fields(tmp_path):
    fresh = structured_v1_plan_state()
    runner = _FakeRunner(
        claude_outputs=[fresh],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass
    records = _plan_round_records(runner)
    assert not any(
        r.scheduler_issue_digest is not None or r.scheduler_stall_reset for r in records
    )


def _matrix_plan_candidate_1229(**row_overrides):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    row = {
        "row_id": "row-a", "label": "L", "entry_path_or_mode": "m", "initial_state": "i",
        "event": "e", "expected_outcome": "o",
        "forbidden_side_effects": ["Must not refuse"],
        "proposed_test_level": "unit", "proposed_test_location": "tests/x.py",
        "applicability": "required", "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }
    row.update(row_overrides)
    payload["risk_test_matrix"] = {
        "applicability": "applicable", "rows": [row], "important_exclusions": [],
    }
    return json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"


def _claude_prompts_1229(runner):
    return [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


def test_normalized_remaining_defect_is_quoted_and_persisted(tmp_path):
    bad = _matrix_plan_candidate_1229(forbidden_side_effects="Must not refuse", related_scope_item_ids="")
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [bad, bad]
    config = make_config(tmp_path, agent_max_retries=1, execution_strategy_contract_required=True)
    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert "related_scope_item_ids" in exhaustion.diagnostic
    assert "forbidden_side_effects" not in exhaustion.diagnostic
    assert "response rejected after 2 same-agent re-asks" in str(error.value)
    assert '"forbidden_side_effects": ["Must not refuse"]' in exhaustion.candidate_text
    independent = _matrix_plan_candidate_1229(
        forbidden_side_effects=["Must not refuse"], related_scope_item_ids=""
    )
    assert exhaustion.candidate_text == independent
    assert exhaustion.candidate_digest == hashlib.sha256(independent.encode()).hexdigest()
    assert len(runner.diagnostic_posts) == 1
    # The primary turn and both same-agent re-asks (#2066).
    assert len(_claude_prompts_1229(runner)) == 3
    replay_prompt = "\n".join(_claude_prompts_1229(runner)[1])
    assert "Previous response rejected (re-ask 1 of 2)" in replay_prompt
    assert "related_scope_item_ids" in replay_prompt
    assert "forbidden_side_effects must be a JSON array" not in replay_prompt
    assert error.value.failure_category == "deterministic"


def _issue_level_run_1229(tmp_path, outputs, **config_overrides):
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = list(outputs)
    config = make_config(
        tmp_path, execution_strategy_contract_required=True, quiet=False, **config_overrides
    )
    repair_calls = []
    real_repair = validated_agent_module._run_structured_repair

    def spy(*args, **kwargs):
        repair_calls.append(args)
        return real_repair(*args, **kwargs)

    return runner, config, repair_calls, spy


def _run_issue_level_1229(tmp_path, monkeypatch, outputs, retries=0):
    runner, config, repair_calls, spy = _issue_level_run_1229(
        tmp_path, outputs, agent_max_retries=retries
    )
    posted = []
    real_post = architecture_contract.post_issue_comment

    def capture(_runner, *, config, issue_number, body):
        posted.append(str(body))
        return real_post(_runner, config=config, issue_number=issue_number, body=body)

    patch_attr_in(monkeypatch, (validated_agent_module, pr_loop, plan_first_loop_module), "_run_structured_repair", spy)
    patch_attr_in(monkeypatch, (architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", capture)
    # Later stages have no scripted output; only the planning turn is under test.
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner, repair_calls, posted


def _posted_matrix_1229(body):
    import re
    from coding_review_agent_loop.comment_rendering import decode_risk_test_matrix_marker

    match = re.search(r"<!-- AGENT_RISK_TEST_MATRIX: ([A-Za-z0-9+/=_-]+) -->", body)
    assert match, "published plan carries no canonical matrix"
    return decode_risk_test_matrix_marker(match.group(1), bodies=[body])["matrix"]


def test_issue_level_bare_strings_are_accepted_on_first_planner_call(tmp_path, monkeypatch, capsys):
    good = _matrix_plan_candidate_1229(
        forbidden_side_effects="Slip one", related_scope_item_ids="scope-1"
    )
    payload = json.loads(good.split("\n<!--", 1)[0])
    payload["risk_test_matrix"]["important_exclusions"] = "Lone exclusion"
    good = json.dumps(payload) + good[len(good.split("\n<!--", 1)[0]):]
    runner, repair_calls, posted = _run_issue_level_1229(tmp_path, monkeypatch, [good])
    assert len(_claude_prompts_1229(runner)) == 1
    assert repair_calls == []
    assert posted
    matrix = _posted_matrix_1229(posted[0])
    assert matrix["rows"][0]["forbidden_side_effects"] == ["Slip one"]
    assert matrix["rows"][0]["related_scope_item_ids"] == ["scope-1"]
    assert matrix["important_exclusions"] == ["Lone exclusion"]
    err = capsys.readouterr().err
    assert "normalized risk test matrix string field(s)" in err
    for path in ("forbidden_side_effects", "related_scope_item_ids", "important_exclusions"):
        assert path in err


def test_issue_level_unrecoverable_matrix_is_reasked_and_accepted(tmp_path, monkeypatch, capsys):
    from coding_review_agent_loop.protocol import validate_structured_plan_state

    bad = _matrix_plan_candidate_1229(forbidden_side_effects="")
    with pytest.raises(AgentLoopError) as expected:
        validate_structured_plan_state(
            bad, require_execution_strategy_contract=1, require_risk_test_matrix_contract=1
        )
    good = _matrix_plan_candidate_1229(forbidden_side_effects=["Replay accepted"])
    runner, repair_calls, posted = _run_issue_level_1229(tmp_path, monkeypatch, [bad, good])
    prompts = ["\n".join(c) for c in _claude_prompts_1229(runner)]
    assert len(prompts) == 2
    heading = "Previous response rejected (re-ask 1 of 2)"
    assert heading not in prompts[0] and heading in prompts[1]
    appended = prompts[1].split(heading, 1)[1]
    assert "forbidden_side_effects" in appended
    err = capsys.readouterr().err
    # The planner itself resent the matrix (#2066); no repair model ran.
    assert "repair backend=" not in err
    assert repair_calls == []
    assert posted
    assert _posted_matrix_1229(posted[0])["rows"][0]["forbidden_side_effects"] == ["Replay accepted"]


def test_issue_loop_plan_first_resume_reparses_alias_keyed_requirement_disposition(tmp_path):
    """#1241: a stored review accepted through the `requirement_label` alias resumes."""
    issue_body = "Keep the public API unchanged.\n\n-- Human Reviewer"
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author=None,
        created_at=None,
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_id = requirement.requirement_id
    current_plan = _plan_with_requirement_disposition(
        kind="plan_state",
        summary="Keep the public API unchanged.",
        plan_steps=["Keep the public API unchanged."],
        evidence="The plan keeps the public API unchanged.",
    ).replace('"Requirement 1"', f'"{requirement_id}"').replace("- Requirement 1:", f"- Requirement {requirement_id}:")
    subject = _plan_subject(current_plan)
    aliased_review = (
        structured_plan_review(state="approved", summary="Plan covers the requirement.").replace(
            '"human_requirement_dispositions": []',
            json.dumps(
                {
                    "human_requirement_dispositions": [{
                        "requirement_label": requirement_id,
                        "disposition": "addressed",
                        "evidence": "The plan keeps the public API unchanged.",
                    }]
                }
            )[1:-1],
        )
    )
    aliased_review = aliased_review.replace(
        "\n<!-- AGENT_PLAN_STATE: approved -->",
        "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_PLAN_STATE: approved -->",
    )
    assert "requirement_label" in aliased_review
    assert "HUMAN_REQUIREMENTS_RESOLVED" in aliased_review
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1, subject=subject, prior_items=()
        ),
    )
    codex_comment = _attach_round_metadata(
        aliased_review,
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=subject,
            state="approved",
            canonical_reviewer_response=aliased_review,
        ),
    )
    runner = FakeRunner(
        issue_payload={"body": issue_body},
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), plan_execution_mode="plan-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == []


# --- staged resumes over persisted repeated-hash history (real extractor) -----


def _np_payload_rewrite(comments, transform):
    """Apply ``transform(mapping) -> mapping`` to every round metadata payload."""
    from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping

    rewritten = []
    for comment in comments:
        body = comment["body"]
        match = re.search(r"AGENT_LOOP_META: (\S+) -->", body)
        if match is not None:
            payload = transform(decode_mapping(match.group(1)))
            body = body.replace(match.group(1), encode_mapping(payload))
        rewritten.append(dict(comment, body=body))
    return rewritten


def _np_repeated_staged_history(tmp_path, *, rounds=3, live_changed=False):
    """A valid staged history whose ``rounds`` completed rounds all reviewed one plan.

    A live staged planner cannot return a byte-identical plan (a no-op semantic
    patch is rejected), so a real run is stopped at the next round's prelaunch
    checkpoint and every earlier plan identity in the persisted round metadata
    (subject, prior subject and candidate-key components) is mapped to the live
    candidate's.  The result is the record shape a repeated plan would persist;
    the detector then reads it with the real observation extractor.

    With ``live_changed`` the earlier identities are mapped to round
    ``rounds``'s reviewed candidate instead, so the completed rounds repeat one
    plan P while the live candidate is the changed P2.
    """
    from coding_review_agent_loop.panel_evidence import _plan_candidate_key_for
    from coding_review_agent_loop.plan_assembly import decode_assembled_plan_sidecar

    fresh, base = _m1103_fresh_base()
    codex, claude, base = _np_staged_chain(1, rounds, base=base)
    runner = _FakeRunner(claude_outputs=[fresh, *claude], codex_outputs=list(codex))
    config = _staged_plan_config(tmp_path, max_rounds=8)
    real_post = architecture_contract.post_issue_comment

    def stop_at_next_checkpoint(*args, **kwargs):
        metadata = _m943_decoded(kwargs["body"])
        if (
            metadata is not None
            and metadata.phase == "scheduler-prelaunch"
            and metadata.round_number == rounds + 1
        ):
            raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment",
        side_effect=stop_at_next_checkpoint,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    records = _plan_round_records(runner)
    live = next(r for r in reversed(records) if r.role == "coder")
    assert live.round_number == rounds + 1
    if live_changed:
        target = next(
            r for r in records
            if r.role == "reviewer" and r.round_number == rounds and r.plan_candidate_key
        )
        target_subject, target_key = target.subject, dict(target.plan_candidate_key)
    else:
        target_subject = live.subject
        target_key = _plan_candidate_key_for(
            plan_subject=live.subject,
            sidecar=decode_assembled_plan_sidecar(live.assembled_plan_sidecar),
            surfaced_requirement_ids=(),
        ).as_dict()
    mapping: dict[str, str] = {}
    for record in records:
        if live_changed and record.role == "coder" and record.round_number == rounds + 1:
            continue
        for value in (record.subject, record.prior_plan_subject):
            if value and value != target_subject:
                mapping[value] = target_subject
        for key in (record.plan_candidate_key, record.scheduler_plan_previous_key):
            for name, value in (key or {}).items():
                if isinstance(value, str) and value != target_key.get(name):
                    mapping[value] = target_key[name]
    if live_changed:
        # The live candidate's own identity is never rewritten.
        mapping.pop(live.subject, None)

    # Self-authenticating candidate content (the sidecar checks its own
    # aggregate identity) keeps its bytes; only identity references move.
    kept = {"assembled_plan_sidecar", "canonical_plan", "raw_structured_coder_response", "raw_patch_provenance"}

    def remap(payload):
        result = {}
        planner = payload.get("role") == "coder"
        for name, value in payload.items():
            # A planner record's own identity fields are bound to its sidecar;
            # only the subject it was asked to revise is an identity reference.
            if name in kept or (planner and name != "prior_plan_subject"):
                result[name] = value
                continue
            text = json.dumps(value)
            for old, new in mapping.items():
                text = text.replace(old, new)
            result[name] = json.loads(text)
        return result

    history = _np_payload_rewrite(runner.issue_comments, remap)
    return history, config, (target_subject if live_changed else live.subject)


def test_plan_no_progress_stops_a_staged_resume_over_repeated_history(tmp_path):
    """`plan-k-unchanged` under primary-then-panel with the real observation extractor."""
    history, config, subject = _np_repeated_staged_history(tmp_path)
    rerun = _FakeRunner(issue_comments=list(history))
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)

    _np_assert_no_progress(excinfo.value, subject=subject, rounds=(1, 3))
    assert _m1103_agent_calls(rerun) == []
    # No new checkpoint, reviewer record or planner record for round 4.
    assert not any(
        record.round_number == 4 and record.role != "coder"
        for record in _plan_round_records(rerun)
    )
    assert len(_plan_round_records(rerun)) == len(_plan_round_records(_FakeRunner(issue_comments=history)))
    assert len(_np_diagnostics(rerun, 4)) == 1


def test_plan_staged_resume_after_a_new_signed_requirement_reviews_first(tmp_path):
    """`plan-requirement-change` under primary-then-panel: the issue digest is
    unchanged, the surfaced requirement set changed, so the plan is reviewed."""
    history, config, _subject = _np_repeated_staged_history(tmp_path)
    signed = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:00:00Z",
        "body": "Also preserve the audit trail.\n\n-- Human Reviewer",
    }
    seeded = [*history, signed]
    rerun = _FakeRunner(issue_comments=list(seeded))
    try:
        run_issue_loop(rerun, issue_number=56, config=config, plan_first=True)
    except (AgentLoopError, HumanDecisionRequiredError, IndexError, StopIteration) as exc:
        assert not str(exc).startswith(NO_PROGRESS_PREFIX_TEXT), exc
    assert _m1103_agent_calls(rerun)[:1] == ["codex"]
    assert not any(
        NO_PROGRESS_PREFIX_TEXT in comment["body"] for comment in rerun.issue_comments[len(seeded):]
    )


def test_plan_staged_resume_reviews_a_changed_live_plan_first(tmp_path):
    """`plan-live-hash-changed` (resume, primary-then-panel, real extractor)."""
    history, config, repeated = _np_repeated_staged_history(tmp_path, live_changed=True)
    # Control: the completed rounds really repeat one plan and one ID set,
    # and only the live candidate differs.
    persisted = _plan_round_records(_FakeRunner(issue_comments=history))
    assert {r.subject for r in persisted if r.role == "reviewer"} == {repeated}
    planners = {r.round_number: r for r in persisted if r.role == "coder"}
    assert [planners[n].prior_plan_subject for n in (2, 3, 4)] == [repeated] * 3
    assert all(
        {item.item_id for item in planners[n].prior_items} == {"item-1"} for n in (2, 3, 4)
    )
    assert planners[4].subject != repeated

    _fresh, base = _m1103_fresh_base()
    _c, _cl, base = _np_staged_chain(1, 3, base=base)
    codex, claude, _base = _np_staged_chain(4, 4, base=base)
    rerun = _FakeRunner(issue_comments=list(history), codex_outputs=codex, claude_outputs=claude)
    with pytest.raises(AgentLoopError, match="still reported blocking plan issues"):
        run_issue_loop(rerun, issue_number=56, config=replace(config, max_rounds=4), plan_first=True)
    # P2 went to review at round 4; no no-progress stop before it.
    assert _m1103_agent_calls(rerun) == ["codex"]
    assert not any(NO_PROGRESS_PREFIX_TEXT in c["body"] for c in rerun.issue_comments)


def test_plan_resume_reviews_a_changed_live_plan_first_under_all_reviewers(tmp_path):
    """`plan-live-hash-changed` (resume, all-reviewers): P,P,P persisted, then the
    resumed planner turn publishes P2, which is reviewed before any stop."""
    runner, config, error = _np_all_reviewers_run(
        tmp_path, revisions=[_np_revision()] * 3, reviews=_np_reviews(4), max_rounds=4
    )
    assert "still reported blocking plan issues after round 4" in str(error)
    changed = _np_revision(("Update the code differently.", "Run the relevant tests."))
    rerun = _FakeRunner(
        issue_comments=list(runner.issue_comments),
        claude_outputs=[changed], codex_outputs=_np_reviews(2)[1:],
    )
    with pytest.raises(AgentLoopError, match="still reported blocking plan issues after round 5"):
        run_issue_loop(rerun, issue_number=56, config=replace(config, max_rounds=5), plan_first=True)
    assert _m1103_agent_calls(rerun) == ["claude", "codex"]
    assert not any(NO_PROGRESS_PREFIX_TEXT in c["body"] for c in rerun.issue_comments)


def test_plan_empty_live_blocking_set_never_stops(tmp_path):
    """`plan-empty-or-degraded`: K identical plan rounds whose ledger holds only
    future follow-ups never stop; the run reaches the round limit."""
    future = structured_plan_review(
        state="blocking", future_followups=["Consider a later cleanup."]
    )
    keep_future = structured_plan_review(
        state="blocking",
        prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "future"}],
    )
    runner, _config, error = _np_all_reviewers_run(
        tmp_path,
        revisions=[structured_plan_revision(plan_steps=["Update the code.", "Run the relevant tests."])] * 5,
        reviews=[future] + [keep_future] * 5,
        max_rounds=6,
    )
    assert not isinstance(error, HumanDecisionRequiredError), error
    assert NO_PROGRESS_PREFIX_TEXT not in str(error)
    records = _plan_round_records(runner)
    # Rounds 2-5 reviewed one repeated plan (round 1 reviewed the fresh plan).
    planners = [r for r in records if r.role == "coder" and r.round_number >= 3]
    # The plan really repeats and the persisted ledgers carry no blocking item.
    assert len({r.prior_plan_subject for r in planners}) == 1 and len(planners) >= 3
    assert all(
        all(item.status not in {"blocking", "same-plan"} for item in r.prior_items)
        for r in planners
    )
    assert _m1103_agent_calls(runner).count("codex") == 6
    assert not any(NO_PROGRESS_PREFIX_TEXT in c["body"] for c in runner.issue_comments)


def _np_panel_patches(base_rounds, *, base, item_id="item-1"):
    """Planner patches keeping ``item_id`` open; each publishes the next listed round."""
    outputs = []
    for n, published in zip(base_rounds, base_rounds[1:]):
        payload = {
            "schema_version": 1,
            "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1,
            "state": "blocking",
            "summary": f"Keep the panel gap open, revision {n}.",
            "prior_plan_item_dispositions": [
                {"item_id": item_id, "disposition": "blocking", "note": "Out of scope."}
            ],
            "base_round_number": n,
            "base_state_identity": base.state_identity,
            "operations": [{
                "op": "replace", "field": "plan_steps",
                "value": [f"Implement the reviewed scope, panel revision {n}."],
            }],
        }
        base = AuthenticatedPlanState.from_plan(
            assemble_plan_revision(base, payload), round_number=published
        )
        outputs.append(json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude")
    return outputs


def test_plan_reset_received_after_the_panel_opens_restarts_the_window(tmp_path, monkeypatch):
    """`plan-reset-window` after the panel opens: the reset is stamped on a
    post-panel checkpoint, consumed once, and its floor persists on a flagless resume."""
    _np_patch_observations(monkeypatch)
    fresh, base = _m1103_fresh_base()
    # Round 1: the primary approves; round 2 is the reviewer-only panel round,
    # where Gemini blocks; every later revision keeps that item open.
    gemini_first = structured_plan_review(
        state="blocking", reviewer="Google Gemini", blocking_plan_issues=["Close the gap."]
    )
    gemini_keep = structured_plan_review(
        state="blocking", reviewer="Google Gemini", prior_plan_item_dispositions=_NP_KEEP_ITEM
    )
    patches = _np_panel_patches([1, 3, 4, 5, 6, 7, 8], base=base)
    # The first approval is of the fresh plan; later primary reviews (when the
    # scheduler selects the primary) must account for the panel's item too.
    approvals = [structured_plan_review(state="approved")] + [
        structured_plan_review(state="blocking", prior_plan_item_dispositions=_NP_KEEP_ITEM)
    ] * 10
    runner = _FakeRunner(
        claude_outputs=[fresh, *patches], codex_outputs=list(approvals),
        gemini_outputs=[gemini_first] + [gemini_keep] * 10,
    )
    config = _staged_plan_config(tmp_path, max_rounds=3)
    with pytest.raises(AgentLoopError, match="still reported blocking plan issues"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    history = list(runner.issue_comments)
    used = sum(1 for cmd, _cwd in runner.commands if cmd[0] == "claude") - 1

    def resumed(*, reset):
        return _FakeRunner(
            issue_comments=list(history), claude_outputs=patches[used:],
            codex_outputs=list(approvals), gemini_outputs=[gemini_keep] * 10,
        ), replace(config, max_rounds=10, plan_reset_stall_streak=reset)

    # Without the reset the window through rounds 2-4 stops round 5.
    no_reset, no_reset_config = resumed(reset=False)
    with pytest.raises(HumanDecisionRequiredError, match="rounds 2-4"):
        run_issue_loop(no_reset, issue_number=56, config=no_reset_config, plan_first=True)

    reset_run, reset_config = resumed(reset=True)
    with pytest.raises(HumanDecisionRequiredError) as excinfo:
        run_issue_loop(reset_run, issue_number=56, config=reset_config, plan_first=True)
    checkpoints = [
        r for r in _plan_round_records(reset_run)
        if r.phase == "scheduler-prelaunch" and r.scheduler_stall_reset
    ]
    # Consumed once, on a post-panel (non-primary) checkpoint that decodes valid.
    assert len(checkpoints) == 1
    assert checkpoints[0].scheduler_metadata_status == "valid"
    assert checkpoints[0].scheduler_phase != "primary"
    floor = checkpoints[0].round_number
    assert f"rounds {floor}-{floor + 2}" in str(excinfo.value)
    assert floor > 2

    flagless = _FakeRunner(issue_comments=list(reset_run.issue_comments))
    with pytest.raises(HumanDecisionRequiredError, match=f"rounds {floor}-{floor + 2}"):
        run_issue_loop(flagless, issue_number=56, config=replace(config, max_rounds=10), plan_first=True)
    assert _m1103_agent_calls(flagless) == []
