"""Plan-first loop tests: execution strategy and execution decisions (#1543)."""

import json
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.phase_progress as phase_progress_module
from coding_review_agent_loop.phase_progress import StagedTopologyOutcome
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.comment_rendering import render_canonical_plan_state
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    RecordedPhase,
    approved_plan_hash,
    format_decomposition_parent_summary,
)
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
    validate_structured_plan_state,
)
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    FakeRunner,
    _fresh_child_route_fixture,
    _fresh_staged_plan_for_recovery,
)
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_state,
    structured_v1_plan_state,
)
from coding_review_agent_loop.decomposition import (  # noqa: E402
    ExecutionDecision as _M1088ExecutionDecision,
    format_execution_decision as _m1088_format_execution_decision,
)
from coding_review_agent_loop.decomposition import (  # noqa: E402
    EXECUTION_DECISION_MARKER_RE as _M1087_DECISION_RE,
    _decode_execution_decision as _m1087_decode_decision,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


@pytest.mark.parametrize(
    ("disposition", "plan_first", "message"),
    [
        (EXECUTION_DISPOSITION_PLANNING, False, "--plan-first --plan-execution-mode auto"),
        (EXECUTION_DISPOSITION_DIRECT, True, "without `--plan-first`"),
    ],
)
def test_fresh_child_issue_flags_cannot_switch_reviewed_route(
    tmp_path, monkeypatch, disposition, plan_first, message
):
    fresh, issue, parent = _fresh_child_route_fixture(disposition)
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop), "_resolve_fresh_child_provenance", lambda **_: fresh)
    with pytest.raises(AgentLoopError, match=re.escape(message)):
        run_issue_loop(
            _FakeRunner(), issue_number=56, config=make_config(tmp_path), plan_first=plan_first
        )


def test_direct_child_entry_posts_planning_handoff_before_planner_and_reuses_it(
    tmp_path, monkeypatch
):
    fresh, issue, parent = _fresh_child_route_fixture(EXECUTION_DISPOSITION_PLANNING)
    override_digest = "d" * 64
    fresh.route = replace(fresh.route, override_digest=override_digest)
    events = []
    handoff_calls = []
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop), "_resolve_fresh_child_provenance", lambda **_: fresh)
    patch_attr_in(monkeypatch, (execution_policy, plan_first_loop_module, issue_loop), "_post_child_planning_handoff",
        lambda *_args, **kwargs: (handoff_calls.append(kwargs), events.append("handoff")),
    )
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_run_plan_first_loop",
        lambda *_args, **_kwargs: events.append("planner") or 19,
    )
    assert run_issue_loop(
        _FakeRunner(), issue_number=56, config=make_config(tmp_path), plan_first=True
    ) == 19
    assert events == ["handoff", "planner"]
    assert len(handoff_calls) == 1
    assert handoff_calls[0]["parent_issue"] == 55
    assert handoff_calls[0]["plan_hash"] == "plan-hash"
    assert handoff_calls[0]["plan_subject"] == "plan-subject"
    assert handoff_calls[0]["phase_index"] == 1
    assert handoff_calls[0]["created"] == fresh.created
    assert handoff_calls[0]["recommendation"] == fresh.recommendation
    assert handoff_calls[0]["inherited_matrix_row_ids"] == ()
    assert handoff_calls[0]["override_digest"] == override_digest

    recorded = PhaseImplementationHandoffMetadata(
        parent_issue=55, plan_hash="plan-hash", mode="implement-by-phase",
        phase_index=1, phase_title="Child", automation="agent-pr",
        child_issue_number=56, child_issue_url="child-url", strategy="staged",
        topology_source="approved-plan-v1", execution_strategy_contract_version=1,
        recommendation_digest="digest", stage_id="stage-one", plan_subject="plan-subject",
        execution_disposition=EXECUTION_DISPOSITION_PLANNING,
    )
    resumed, _, _ = _fresh_child_route_fixture(
        EXECUTION_DISPOSITION_PLANNING, handoff=recorded
    )
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop), "_resolve_fresh_child_provenance", lambda **_: resumed)
    assert run_issue_loop(
        _FakeRunner(), issue_number=56, config=make_config(tmp_path), plan_first=True
    ) == 19
    assert events == ["handoff", "planner", "planner"]


def test_direct_child_entry_dispatches_through_shared_child_dispatch(tmp_path, monkeypatch):
    fresh, issue, parent = _fresh_child_route_fixture(EXECUTION_DISPOSITION_DIRECT)
    calls = []
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    patch_attr_in(monkeypatch, (execution_policy, child_plan_binding, issue_loop), "_resolve_fresh_child_provenance", lambda **_: fresh)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "prepare_agent_memory", lambda *_args, **_kwargs: None)
    patch_attr_in(monkeypatch, (plan_first_loop_module, issue_loop), "_dispatch_decomposition_child",
        lambda *_args, **kwargs: calls.append(kwargs) or 23,
    )
    assert run_issue_loop(_FakeRunner(), issue_number=56, config=make_config(tmp_path)) == 23
    assert len(calls) == 1
    assert calls[0]["route"].is_direct
    assert calls[0]["existing_handoff"] is None


@pytest.mark.parametrize(
    ("disposition", "expected_hint"),
    [
        (
            EXECUTION_DISPOSITION_PLANNING,
            "agent-loop issue 56 --plan-first --plan-execution-mode auto",
        ),
        (EXECUTION_DISPOSITION_DIRECT, "agent-loop issue 56`"),
    ],
)
def test_parent_rerun_resume_hint_follows_recorded_disposition(
    tmp_path, monkeypatch, capsys, disposition, expected_hint
):
    fresh, child, parent = _fresh_child_route_fixture(disposition)
    handoff = PhaseImplementationHandoffMetadata(
        parent_issue=55, plan_hash="plan-hash", mode="implement-by-phase",
        phase_index=1, phase_title="Child", automation="agent-pr",
        child_issue_number=56, child_issue_url="child-url", strategy="staged",
        topology_source="approved-plan-v1", execution_strategy_contract_version=1,
        recommendation_digest="digest", stage_id="stage-one", plan_subject="plan-subject",
        execution_disposition=disposition,
    )
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        plan_first_loop_module, "find_existing_phase_implementation_handoff",
        lambda *_args, **_kwargs: handoff,
    )
    monkeypatch.setattr(
        phase_progress_module, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        phase_progress_module,
        "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: (handoff,),
    )
    result = plan_first_loop_module._dispatch_current_decomposition_phase(
        _FakeRunner(), config=make_config(tmp_path), memory=None,
        usage_context=SimpleNamespace(), issue_number=55,
        current_plan="Approved parent plan", plan_subject="plan-subject",
        outcome=StagedTopologyOutcome(
            created=(fresh.created,), stage_ids=("stage-one",),
            automations=("agent-pr",), plan_hash="plan-hash",
            mode="implement-by-phase", topology_source="approved-plan-v1",
        ),
        recommendation=fresh.recommendation,
        approved_plan_context=fresh.parent_plan_context, issue_context=parent,
        mode="implement-by-phase", coder_session_id=None,
    )
    assert result == 0
    assert expected_hint in capsys.readouterr().out


def test_auto_legacy_plan_round_reaches_review_before_legacy_refusal(tmp_path):
    """An in-flight legacy round must be revisable into the fresh contract."""
    runner = _FakeRunner(
        claude_outputs=[structured_plan_state(summary="Historical plan without a strategy.")],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    with pytest.raises(AgentLoopError, match="legacy-undecided"):
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, plan_execution_mode="auto"),
            plan_first=True,
        )

    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("include_matrix", [False, True])
def test_legacy_plan_revision_gate_runs_through_issue_loop(tmp_path, monkeypatch, include_matrix):
    """A resumed legacy round keeps execution rules but rejects matrix opt-in."""
    legacy_plan = "Historical execution-v1 plan."
    legacy_comment = _attach_round_metadata(
        legacy_plan + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Anthropic Claude",
            round_number=1,
            subject=_plan_subject(legacy_plan),
            canonical_plan=legacy_plan,
            state="blocking",
        ),
    )
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload.update(
        {
            "kind": "plan_revision",
            "summary": "Revised the historical plan with the execution contract.",
            "prior_plan_item_dispositions": [
                {"item_id": "item-1", "disposition": "resolved"}
            ],
        }
    )
    if not include_matrix:
        for key in (
            "risk_test_matrix_contract_version",
            "risk_test_matrix",
            "risk_test_matrix_changes",
        ):
            payload.pop(key, None)
    revision = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-09-14T00:00:00Z",
                "body": legacy_comment,
            }
        ],
        claude_outputs=[revision],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="The legacy plan needs a revision.",
                blocking_plan_issues=["Add the reviewed execution contract."],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        execution_strategy_contract_required=True,
        plan_execution_mode="plan-only",
        max_rounds=3,
    )
    captured_gate_values = []
    original_validator = response_validation._validate_plan_revision_response

    def capture_validator(*args, **kwargs):
        captured_gate_values.append(
            kwargs["reject_unsolicited_risk_test_matrix_contract"]
        )
        return original_validator(*args, **kwargs)

    patch_attr_in(monkeypatch, (response_validation, plan_first_loop_module), "_validate_plan_revision_response", capture_validator
    )
    if include_matrix:
        with patch.object(validated_agent_module, "attempt_repair", return_value=None):
            with pytest.raises(AgentLoopError):
                run_issue_loop(
                    runner,
                    issue_number=56,
                    config=config,
                    plan_first=True,
                )
        assert captured_gate_values and all(captured_gate_values)
    else:
        assert run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
        ) == 0
        assert captured_gate_values == [True]


@pytest.mark.parametrize(
    "requested_policy, expected_error",
    [
        ("auto", "existing implementation PR"),
        ("implement-one-shot", "incompatible"),
        ("plan-only", "existing implementation PR"),
    ],
)
def test_plan_first_pr_recovery_reconciles_fresh_strategy_before_resume(
    tmp_path, requested_policy, expected_error
):
    raw_plan, canonical_plan = _fresh_staged_plan_for_recovery()
    parsed = validate_structured_plan_state(raw_plan, require_execution_strategy_contract=1)
    recommendation = parsed.execution_recommendation
    assert recommendation is not None
    plan_comment = _attach_round_metadata(
        canonical_plan + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(canonical_plan),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=raw_plan,
            execution_strategy_contract_version=1,
            execution_strategy_identity=recommendation.identity(),
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=approved_plan_hash(canonical_plan),
    )
    runner = _FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:00Z", "body": plan_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:01Z", "body": handoff},
        ],
        pr_payload={"body": "Fixes #56"},
    )

    with pytest.raises(AgentLoopError, match=expected_error):
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, plan_execution_mode=requested_policy),
            plan_first=True,
        )

    assert not any(cmd[:1] == ["claude"] or cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any("AGENT_PLAN_EXECUTION_DECISION" in comment for comment in runner.comments)


def test_plan_first_plan_only_legacy_pr_recovery_still_reviews_without_plan_round(tmp_path):
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
    )
    runner = _FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:00Z", "body": handoff}
        ],
        pr_payload={"body": "Fixes #56"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_plan_first_plan_only_reconciles_one_shot_against_staged_state(tmp_path):
    canonical_plan = render_canonical_plan_state(
        validate_structured_plan_state(structured_v1_plan_state())
    )
    plan_comment = _attach_round_metadata(
        canonical_plan + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude",
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(canonical_plan),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=structured_v1_plan_state(),
            execution_strategy_contract_version=1,
            execution_strategy_identity=(
                validate_structured_plan_state(structured_v1_plan_state())
                .execution_recommendation.identity()
            ),
        ),
    )
    staged_summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="decompose-only",
        plan_hash="old-plan-hash",
        created=(
            CreatedPhaseIssue(
                phase=RecordedPhase(title="Existing stage", automation="agent-pr"),
                issue_url="https://github.com/OWNER/REPO/issues/101",
                issue_number=101,
            ),
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=approved_plan_hash(canonical_plan),
    )
    runner = _FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:00Z", "body": plan_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:01Z", "body": staged_summary},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        pr_payload={"body": "Fixes #56"},
    )

    with pytest.raises(AgentLoopError, match="existing decomposition summary"):
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, plan_execution_mode="plan-only"),
            plan_first=True,
        )

    assert not any("AGENT_PLAN_EXECUTION_DECISION" in comment for comment in runner.comments)


# --- #1087: an unrealized execution decision is superseded on re-approval ---


_M1087_STALE_HASH = "de76758b0293d117"


_M1087_BRANCH_ENDPOINT = "repos/OWNER/REPO/branches/agent-loop%2Fmanaged-56"


class _M1087Runner(FakeRunner):
    """Serves the reserved managed branch as absent unless told otherwise."""

    def __init__(self, *, branch_exists=False, unreadable_child_search=False, **kwargs):
        super().__init__(**kwargs)
        self.branch_exists = branch_exists
        self.unreadable_child_search = unreadable_child_search

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        cmd = [str(arg) for arg in args]
        if cmd[:2] == ["gh", "api"] and len(cmd) > 2 and "/branches/" in cmd[2]:
            # Like GitHub, only the encoded branch path resolves; the raw
            # `agent-loop/managed-56` path 404s even when the branch exists.
            cmd, cwd_path = self._record_command(cmd, cwd)
            if cmd[2] == _M1087_BRANCH_ENDPOINT and self.branch_exists:
                return CommandResult(
                    cmd, cwd_path, json.dumps({"name": "agent-loop/managed-56"}), "", 0
                )
            return CommandResult(cmd, cwd_path, "", "gh: Not Found (HTTP 404)", 1)
        if (
            self.unreadable_child_search
            and cmd[:3] == ["gh", "issue", "list"]
            and cmd[cmd.index("--json") + 1] == "number,title,body,state"
        ):
            # The realization inventory's own search; adoption searches differ.
            cmd, cwd_path = self._record_command(cmd, cwd)
            return CommandResult(cmd, cwd_path, "not json", "", 0)
        return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)


def _m1087_stale_decision_comment(plan_hash=_M1087_STALE_HASH, retires=(), second=0):
    decision = _M1088ExecutionDecision(
        parent_issue=56,
        plan_hash=plan_hash,
        plan_subject="An earlier approval of the same issue",
        execution_strategy_contract_version=1,
        strategy="one-shot",
        topology_source="approved-plan-v1",
        recommendation_digest="f" * 64,
        requested_policy="implement-one-shot",
        current_action="implement-one-shot",
        retires_plan_hashes=tuple(retires),
    )
    return {
        "author": {"login": "coding-review-agent-loop"},
        "createdAt": f"2026-05-22T00:00:{second:02d}Z",
        "body": _m1088_format_execution_decision(decision),
    }


def _m1087_runner(**kwargs):
    kwargs.setdefault("issue_comments", [_m1087_stale_decision_comment()])
    return _M1087Runner(
        claude_outputs=[
            structured_v1_plan_state(),
            "Implemented the approved fresh plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        **kwargs,
    )


def _m1087_config(tmp_path):
    return make_config(
        tmp_path,
        plan_execution_mode="implement-one-shot",
        execution_strategy_contract_required=True,
    )


def _m1087_posted_decisions(runner):
    return [
        _m1087_decode_decision(match.group("payload"))
        for body in runner.comments
        for match in _M1087_DECISION_RE.finditer(body)
    ]


def _m1087_implementation_dispatched(runner):
    # The scripted plan is consumed by planning; the second output only by
    # the implementation turn.
    return runner.claude_outputs == []


def test_1087_reapproval_supersedes_an_unrealized_decision(tmp_path):
    runner = _m1087_runner()

    assert run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True) == 0

    posted = _m1087_posted_decisions(runner)
    assert len(posted) == 1
    assert posted[0].plan_hash != _M1087_STALE_HASH
    # The superseding decision names the hash it retires; the old record stays.
    assert posted[0].retires_plan_hashes == (_M1087_STALE_HASH,)
    thread_hashes = [
        _m1087_decode_decision(match.group("payload")).plan_hash
        for comment in runner.issue_comments
        for match in _M1087_DECISION_RE.finditer(comment["body"])
    ]
    assert thread_hashes == [_M1087_STALE_HASH, posted[0].plan_hash]
    assert sum("AGENT_PLAN_ONE_SHOT_IMPL" in body for body in runner.comments) == 1
    assert _m1087_implementation_dispatched(runner)


def test_1087_superseded_decision_stays_retired_after_the_pr_exists(tmp_path):
    runner = _m1087_runner()
    config = _m1087_config(tmp_path)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert _m1087_implementation_dispatched(runner)

    # The rerun resumes the PR bound to the superseding decision (scripted
    # coder output is exhausted, so a second implementation would fail); the
    # old decision does not come back as a conflict.
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(_m1087_posted_decisions(runner)) == 1


def test_1087_decision_with_an_open_managed_pr_still_refuses(tmp_path):
    runner = _m1087_runner(
        open_prs_payload=[
            {"number": 81, "state": "OPEN", "body": "Work in progress.", "headRefName": "agent-loop/managed-56"}
        ]
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True)

    message = str(excinfo.value)
    assert "Conflicting execution decision exists" in message
    assert _M1087_STALE_HASH in message
    assert "PR #81" in message
    assert _m1087_posted_decisions(runner) == []
    # Only the planning turn ran; no implementation was dispatched.
    assert not _m1087_implementation_dispatched(runner)


def test_1792_decision_with_a_closed_pr_is_superseded(tmp_path):
    # A PR opened before its handoff record was posted, then closed with its
    # branch deleted, still acted on the decision.
    runner = _m1087_runner(
        open_prs_payload=[
            {
                "number": 82,
                "state": "CLOSED",
                "body": "Abandoned attempt.",
                "headRefName": "agent-loop/managed-56",
            }
        ]
    )

    assert run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True) == 0

    inventory = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:3] == ["gh", "pr", "list"] and "headRefName" in cmd[cmd.index("--json") + 1]
    ]
    assert inventory and inventory[-1][inventory[-1].index("--state") + 1] == "all"
    assert _m1087_posted_decisions(runner)[0].retires_plan_hashes == (_M1087_STALE_HASH,)
    assert _m1087_implementation_dispatched(runner)


def test_1087_reapproved_retired_hash_is_live_again_and_still_checked(tmp_path):
    # A, B retiring A, then A retiring B: the last A is live.  A PR that acted
    # on it and remains open must still
    # block the re-approved plan C, even though both hashes were once retired.
    other_hash = "0123456789abcdef"
    runner = _m1087_runner(
        issue_comments=[
            _m1087_stale_decision_comment(second=0),
            _m1087_stale_decision_comment(other_hash, retires=[_M1087_STALE_HASH], second=1),
            _m1087_stale_decision_comment(retires=[other_hash], second=2),
        ],
        open_prs_payload=[
            {
                "number": 83,
                "state": "OPEN",
                "body": "Abandoned attempt.",
                "headRefName": "agent-loop/managed-56",
            }
        ],
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True)

    message = str(excinfo.value)
    assert _M1087_STALE_HASH in message and other_hash not in message
    assert "PR #83" in message
    assert _m1087_posted_decisions(runner) == []
    assert not _m1087_implementation_dispatched(runner)


def test_1087_unreadable_child_issue_search_still_refuses(tmp_path):
    runner = _m1087_runner(unreadable_child_search=True)

    with pytest.raises(AgentLoopError, match="child issues that could not be listed"):
        run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True)

    assert _m1087_posted_decisions(runner) == []
    assert not _m1087_implementation_dispatched(runner)


def test_1087_decision_with_its_managed_branch_still_refuses(tmp_path):
    runner = _m1087_runner(branch_exists=True)

    with pytest.raises(AgentLoopError, match="branch `agent-loop/managed-56`"):
        run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True)

    # The probe used the encoded path; the fake resolves nothing else.
    assert ["gh", "api", _M1087_BRANCH_ENDPOINT] in [cmd[:3] for cmd, _cwd in runner.commands]
    assert _m1087_posted_decisions(runner) == []
    assert not _m1087_implementation_dispatched(runner)


def test_1087_decision_with_a_child_issue_still_refuses(tmp_path):
    runner = _m1087_runner(
        search_issues_payload=[
            {
                "number": 90,
                "title": "Phase 1: Earlier stage (from #56)",
                "url": "https://github.com/OWNER/REPO/issues/90",
                "body": "",
            }
        ]
    )

    with pytest.raises(AgentLoopError, match="child issue #90"):
        run_issue_loop(runner, issue_number=56, config=_m1087_config(tmp_path), plan_first=True)

    assert _m1087_posted_decisions(runner) == []
