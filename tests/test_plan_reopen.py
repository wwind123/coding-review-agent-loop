"""Operator-requested reopening of an approved plan (#2010)."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.decomposition as decomposition
import coding_review_agent_loop.github as github
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_board_state as plan_board_state
import coding_review_agent_loop.plan_execution as plan_execution
import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
import coding_review_agent_loop.plan_reopen as plan_reopen
from agent_loop_helpers import (
    make_config,
    patch_object_in,
    structured_plan_review,
    structured_v1_plan_state,
)
from issue_loop_test_helpers import _m1103_agent_calls, _plan_round_records
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    EXECUTION_TOPOLOGY_SOURCE,
    DecompositionMetadata,
    ExecutionDecision,
    OneShotImplementationHandoffMetadata,
    _encode_json_payload,
    _encode_metadata,
    _encode_one_shot_impl_handoff_metadata,
    approved_plan_hash,
    format_execution_decision,
)
from coding_review_agent_loop.github import IssueComment, get_issue_context
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.plan_issue_scope import (
    PLAN_SCOPE_BOUNDARY_MARKER,
    PLAN_SCOPE_BOUNDARY_REASON_OPERATOR_REOPEN,
    latest_scope_boundary,
)
from test_plan_first_loop_issue_scope import (
    _M1390Runner,
    _M1390_ACTOR,
    _M1390_EDIT,
    _m1390_config,
    _m1390_digest,
    _m1390_runner,
)

HEAD_SHA = "6367c48dd193d56ea7b0baad25b19455e529f5ee"


def _orchestrator(body: str) -> dict[str, object]:
    return {
        "author": {"login": _M1390_ACTOR[0]},
        "_rest_author_id": _M1390_ACTOR[1],
        "createdAt": "2026-05-23T02:00:00Z",
        "body": body,
    }


def _approved_history(tmp_path) -> tuple[list[dict[str, object]], str]:
    """An issue whose round-1 plan was completely approved (plan-only), no hand-off."""
    runner = _m1390_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_m1390_config(tmp_path), plan_first=True,
    ) == 0
    history = list(runner.issue_comments)
    coder = [
        record for record in _plan_round_records(SimpleNamespace(issue_comments=history))
        if record.role == "coder"
    ]
    return history, approved_plan_hash(coder[-1].canonical_plan)


def _execution_decision(plan_hash: str) -> dict[str, object]:
    return _orchestrator(format_execution_decision(ExecutionDecision(
        parent_issue=56, plan_hash=plan_hash, plan_subject="e" * 64,
        execution_strategy_contract_version=1, strategy="one-shot",
        topology_source=EXECUTION_TOPOLOGY_SOURCE, recommendation_digest="d" * 64,
        requested_policy="implement-one-shot", current_action="implement-one-shot",
    )))


def _pr_handoff(plan_hash: str) -> dict[str, object]:
    return _orchestrator(format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha=HEAD_SHA, flow="approved-plan-implementation", plan_hash=plan_hash,
    ))


def _one_shot_handoff(plan_hash: str) -> dict[str, object]:
    encoded = _encode_one_shot_impl_handoff_metadata(OneShotImplementationHandoffMetadata(
        parent_issue=56, plan_hash=plan_hash, plan_subject="e" * 64,
        mode="implement-one-shot", pr_number=77, pr_head_sha=HEAD_SHA,
    ))
    return _orchestrator(f"One-shot handoff.\n\n<!-- AGENT_PLAN_ONE_SHOT_IMPL: {encoded} -->")


def _decomposition_summary(plan_hash: str) -> dict[str, object]:
    encoded = _encode_metadata(DecompositionMetadata(
        parent_issue=56, plan_hash=plan_hash, mode="decompose-only", phase_count=1,
        phase_titles=("Stage one",), automation=("agent",),
        children=(("Stage one (from #56)", "https://github.com/OWNER/REPO/issues/90", 90),),
    ))
    return _orchestrator(f"Decomposition.\n\n<!-- AGENT_PLAN_DECOMPOSITION: {encoded} -->")


def _child_issue(plan_hash: str, *, number: int = 90, parent: int = 56) -> dict[str, object]:
    """One `gh issue list` entry for a realized child issue."""
    marker = _encode_json_payload({
        "identity": "x" * 16, "parent_issue": parent, "plan_hash": plan_hash,
        "source": "model", "stage_id": 1,
    })
    return {
        "number": number, "title": f"Stage one (from #{parent})",
        "url": f"https://github.com/OWNER/REPO/issues/{number}",
        "body": f"Child phase issue for parent #{parent}.\n\n"
        f"<!-- AGENT_PLAN_PHASE_IDENTITY: {marker} -->",
    }


_LISTING_FAULTS = (
    "failure", "invalid-json", "empty", "not-a-list", "malformed-entry", "missing-body",
)


class _SearchFaultRunner(_M1390Runner):
    """Serves one unreadable inventory listing (#2010 items 3 and 4).

    ``listing`` selects the child-issue search or the open-PR listing; the
    other listing stays readable.
    """

    def __init__(self, *, search_fault, listing="child-search", **kwargs):
        super().__init__(**kwargs)
        self.search_fault = search_fault
        self.listing = listing

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = [str(arg) for arg in args]
        child_search = command[:3] == ["gh", "issue", "list"] and "--search" in command
        pr_list = command[:3] == ["gh", "pr", "list"]
        if (child_search and self.listing == "child-search") or (
            pr_list and self.listing == "pr-list"
        ):
            from coding_review_agent_loop.runner import CommandResult

            recorded, cwd_path = self._record_command(args, cwd)
            stdout, returncode = {
                "failure": ("", 1),
                "invalid-json": ("not json", 0),
                "empty": ("", 0),
                "not-a-list": ("{}", 0),
                "malformed-entry": ('[{"number": "90", "body": "x"}]', 0),
                "missing-body": ('[{"number": 90}]', 0),
            }[self.search_fault]
            return CommandResult(recorded, cwd_path, stdout, "denied" if returncode else "", returncode)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _boundaries(comments):
    return [c for c in comments if PLAN_SCOPE_BOUNDARY_MARKER in str(c["body"])]


def _writes(runner):
    return [
        cmd for cmd, _cwd in runner.commands
        if cmd[:4] == ["gh", "api", "--method", "POST"]
        or cmd[:3] == ["gh", "issue", "comment"]
        or cmd[:3] == ["gh", "issue", "create"]
    ]


def _rerun(
    tmp_path, history, *, recovery=None, config_overrides=None, search_fault=None,
    listing="child-search", **runner_kwargs,
):
    """Rerun issue #56 with ``--reopen-approved-plan``; spy on recovery and logs."""
    values = {"max_rounds": 5, "reopen_approved_plan": True}
    values.update(config_overrides or {})
    if search_fault is None:
        runner = _m1390_runner(history, **runner_kwargs)
    else:
        runner = _SearchFaultRunner(
            search_fault=search_fault, listing=listing, issue_comments=list(history),
            authenticated_actor=_M1390_ACTOR, serve_rest_issue_comments=True, **runner_kwargs,
        )
    calls: list[dict[str, object]] = []
    real_recover = issue_loop.recover_issue_execution

    def spy(runner_arg, **kwargs):
        calls.append({
            "boundaries_posted": len(_boundaries(runner.issue_comments)),
            "comments": tuple(kwargs["issue_context"].comments),
        })
        if recovery == "forbidden":
            raise AssertionError("issue recovery must not run")
        return real_recover(runner_arg, **kwargs)

    with patch_object_in(
        (plan_first_loop_module, plan_execution, plan_reopen), "log",
        wraps=plan_first_loop_module.log,
    ) as logged, pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(issue_loop, "recover_issue_execution", spy)
        try:
            result = run_issue_loop(
                runner, issue_number=56, config=_m1390_config(tmp_path, **values),
                plan_first=True,
            )
        except AgentLoopError as exc:
            result = exc
    return runner, result, calls, [str(call.args[-1]) for call in logged.call_args_list]


# --- reopen-approved-unhanded -------------------------------------------------


@pytest.mark.parametrize("floor", [False, True])
@pytest.mark.parametrize("edited", [True, False])
def test_reopen_supersedes_an_approved_unhanded_plan_before_recovery(tmp_path, floor, edited):
    """`reopen-approved-unhanded`, `identical-plan-hash` (approval from new rounds only)."""
    history, plan_hash = _approved_history(tmp_path)
    history = [*history, _execution_decision(plan_hash)]
    runner, result, calls, messages = _rerun(
        tmp_path, history,
        config_overrides={"min_reviewers": 1} if floor else None,
        issue_payload=_M1390_EDIT if edited else None,
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    boundaries = _boundaries(runner.issue_comments)
    assert len(boundaries) == 1
    body = str(boundaries[0]["body"])
    assert f"The approved plan `{plan_hash}` was superseded by operator request" in body
    assert "Fresh plan starts at round 2" in body
    comments = [IssueComment(
        author=_M1390_ACTOR[0], created_at="t", body=str(c["body"]),
        author_id=c.get("_rest_author_id"),
    ) for c in runner.issue_comments]
    admitted = latest_scope_boundary(comments, _M1390_ACTOR, issue_number=56)
    assert admitted is not None
    assert admitted.reason == PLAN_SCOPE_BOUNDARY_REASON_OPERATOR_REOPEN
    assert admitted.superseded_plan_hash == plan_hash
    expected_current = _m1390_digest(**_M1390_EDIT) if edited else _m1390_digest()
    assert admitted.current_issue_digest == expected_current
    # The boundary was posted before recovery ran, and recovery saw it.
    assert len(calls) == 1 and calls[0]["boundaries_posted"] == 1
    assert any(PLAN_SCOPE_BOUNDARY_MARKER in str(getattr(c, "body", c)) for c in calls[0]["comments"])
    assert any(
        f"superseding approved plan {plan_hash} by operator request; starting a fresh plan "
        "at round 2" in message
        for message in messages
    )
    assert not any("planning is not reopened" in message for message in messages)
    # The planner ran at round 2 against the current issue body, and the new
    # round's own reviewer approval was needed (the identical plan re-emitted
    # under the superseded hash does not inherit the archived approval).
    new_records = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
    )
    coder = [record for record in new_records if record.role == "coder"]
    assert [record.round_number for record in coder] == [2]
    assert approved_plan_hash(coder[0].canonical_plan) == plan_hash
    assert coder[0].plan_issue_digest == expected_current
    assert [r.round_number for r in new_records if r.role == "reviewer"] == [2]
    assert _m1103_agent_calls(runner) == ["claude", "codex"]
    if edited:
        assert "Narrowed scope: a two-command design." in [
            cmd[-1] for cmd, _cwd in runner.commands if cmd[0] == "claude"
        ][0]


def test_reopen_crash_after_boundary_resumes_the_new_scope_without_the_flag(tmp_path):
    """`resume-after-reopen`: no second boundary; archived approval stays blanked."""
    history, plan_hash = _approved_history(tmp_path)
    crashed, result, _calls, _messages = _rerun(tmp_path, history, claude_outputs=[])
    assert isinstance(result, AgentLoopError)
    assert len(_boundaries(crashed.issue_comments)) == 1
    after_boundary = list(crashed.issue_comments)

    runner, result, _calls, messages = _rerun(
        tmp_path, after_boundary, config_overrides={"reopen_approved_plan": False},
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    assert len(_boundaries(runner.issue_comments)) == 1
    assert not any("planning is not reopened" in message for message in messages)
    assert any("empty-scope" in message for message in messages)
    new_coder = [
        record for record in _plan_round_records(
            SimpleNamespace(issue_comments=runner.issue_comments[len(after_boundary):])
        )
        if record.role == "coder"
    ]
    assert [record.round_number for record in new_coder] == [2]


def test_reopen_flag_on_the_new_unapproved_scope_posts_no_second_boundary(tmp_path):
    """`flag-without-approved-plan`: the flag is logged as having no effect."""
    history, _plan_hash = _approved_history(tmp_path)
    crashed, _result, _calls, _messages = _rerun(tmp_path, history, claude_outputs=[])
    runner, result, _calls, messages = _rerun(
        tmp_path, list(crashed.issue_comments),
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    assert len(_boundaries(runner.issue_comments)) == 1
    assert any(
        "--reopen-approved-plan has no effect" in message and "empty-scope" in message
        for message in messages
    )


# --- refusals -------------------------------------------------------------------


@pytest.mark.parametrize("record", ["pr-handoff", "one-shot"])
def test_reopen_refuses_a_plan_handed_off_to_a_pr_with_the_floor_disabled(tmp_path, record):
    """`refuse-handed-off-pr`: refusal before recovery; nothing posted, no agent."""
    history, plan_hash = _approved_history(tmp_path)
    extra = _pr_handoff(plan_hash) if record == "pr-handoff" else _one_shot_handoff(plan_hash)
    history = [*history, extra]
    runner, result, calls, _messages = _rerun(tmp_path, history, recovery="forbidden")

    assert isinstance(result, AgentLoopError)
    message = str(result)
    assert "--reopen-approved-plan refused" in message
    assert f"plan {plan_hash} was already handed off" in message
    assert "nothing was posted" in message
    assert ("PR #77" in message) if record == "pr-handoff" else ("one-shot" in message)
    assert calls == []
    assert runner.issue_comments == history
    assert _writes(runner) == []
    assert _m1103_agent_calls(runner) == []


@pytest.mark.parametrize("provenance", ["matches", "mismatched", "ambiguous"])
def test_reopen_refuses_a_legacy_closing_reference_pr(tmp_path, monkeypatch, provenance):
    """`refuse-legacy-pr`: the approved-plan scope lookup finds or cannot resolve the PR."""
    history, plan_hash = _approved_history(tmp_path)
    scopes = []

    def validate(runner, *, config, pr_number, expected_scope):
        scopes.append(expected_scope)
        if provenance == "mismatched":
            raise AgentLoopError("provenance names another plan")

    monkeypatch.setattr(github, "validate_pull_request_provenance", validate)
    prs = [{"number": 81, "body": "Fixes #56"}]
    if provenance == "ambiguous":
        prs.append({"number": 82, "body": "Closes #56"})
    runner, result, calls, _messages = _rerun(
        tmp_path, history, recovery="forbidden", open_prs_payload=prs,
    )

    assert isinstance(result, AgentLoopError)
    message = str(result)
    assert "--reopen-approved-plan refused" in message and "nothing was posted" in message
    if provenance == "matches":
        assert f"open PR #81 strongly closes this issue and carries plan {plan_hash}" in message
    else:
        assert "cannot be resolved" in message
    if provenance != "ambiguous":
        assert [(s.flow, s.approved_plan_hash) for s in scopes] == [("approved", plan_hash)]
    assert calls == []
    assert runner.issue_comments == history
    assert _writes(runner) == []
    assert _m1103_agent_calls(runner) == []


def test_reopen_refuses_a_plan_handed_off_into_child_stages(tmp_path):
    """`refuse-handed-off-children` (recorded decomposition summary)."""
    history, plan_hash = _approved_history(tmp_path)
    history = [*history, _decomposition_summary(plan_hash)]
    runner, result, calls, _messages = _rerun(tmp_path, history, recovery="forbidden")

    assert isinstance(result, AgentLoopError)
    assert "decomposition summary records child stages" in str(result)
    assert calls == [] and runner.issue_comments == history
    assert _writes(runner) == [] and _m1103_agent_calls(runner) == []


def test_reopen_refuses_a_realized_child_without_a_parent_summary(tmp_path):
    """`refuse-handed-off-children`: the create-before-summary window refuses."""
    history, plan_hash = _approved_history(tmp_path)
    runner, result, calls, _messages = _rerun(
        tmp_path, history, recovery="forbidden", search_issues_payload=[_child_issue(plan_hash)],
    )

    assert isinstance(result, AgentLoopError)
    assert f"child issue #90 already carries plan {plan_hash}" in str(result)
    assert calls == [] and runner.issue_comments == history
    assert _writes(runner) == [] and _m1103_agent_calls(runner) == []


def test_handoff_inventory_ignores_a_child_for_another_plan(tmp_path, monkeypatch):
    """`refuse-handed-off-children`: a child for a different plan hash does not refuse."""
    config = make_config(tmp_path)
    runner = _m1390_runner(search_issues_payload=[
        _child_issue("f" * 16), _child_issue("a" * 16, number=91, parent=57),
    ])
    monkeypatch.setattr(github, "find_open_pr_closing_issue", lambda *a, **k: None)
    assert plan_reopen.approved_plan_handoff_reason(
        runner, config=config, issue_number=56, comments=(), plan_hash="a" * 16,
    ) is None


def test_handoff_inventory_fails_closed_on_pr_lookup_errors(tmp_path, monkeypatch):
    """`child-discovery-failure`: a PR lookup failure is never 'not handed off'."""
    config = make_config(tmp_path)

    def broken(*_args, **_kwargs):
        raise AgentLoopError("GitHub search is unavailable")

    monkeypatch.setattr(github, "find_open_pr_closing_issue", broken)
    reason = plan_reopen.approved_plan_handoff_reason(
        _m1390_runner(), config=config, issue_number=56, comments=(), plan_hash="a" * 16,
    )
    assert reason is not None and "GitHub search is unavailable" in reason
    assert "cannot be resolved" in reason


@pytest.mark.parametrize("listing", ["child-search", "pr-list"])
@pytest.mark.parametrize("fault", _LISTING_FAULTS)
def test_reopen_refuses_and_posts_nothing_when_an_inventory_listing_is_unreadable(
    tmp_path, listing, fault,
):
    """`child-discovery-failure`, `refuse-legacy-pr` (items 3 and 4).

    Every live listing of the hand-off inventory, the child-issue search and
    the open-PR listing behind the legacy closing-reference lookup, runs its
    real `gh ... list` decoder: a failed command, output that is not JSON,
    empty output, a non-list or a malformed entry refuses before posting and
    is never read as "nothing handed off".
    """
    history, plan_hash = _approved_history(tmp_path)
    runner, result, calls, _messages = _rerun(
        tmp_path, history, recovery="forbidden", search_fault=fault, listing=listing,
    )
    assert isinstance(result, AgentLoopError)
    message = str(result)
    assert f"--reopen-approved-plan refused: plan {plan_hash}" in message
    assert "nothing was posted" in message
    if listing == "child-search":
        assert f"cannot verify that plan {plan_hash} was not handed off" in message
    else:
        assert "an open PR association for this issue cannot be resolved" in message
    assert calls == [] and runner.issue_comments == history and _writes(runner) == []
    assert _m1103_agent_calls(runner) == []


@pytest.mark.parametrize(
    "fault", ["invalid-json", "empty", "not-a-list", "malformed-entry", "missing-body"],
)
def test_strict_pr_lookup_raises_where_lenient_lookup_reads_none(tmp_path, fault):
    """`refuse-legacy-pr` (item-4): only the reopen inventory uses the strict PR lookup."""
    config = make_config(tmp_path)
    runner = _SearchFaultRunner(
        search_fault=fault, listing="pr-list", authenticated_actor=_M1390_ACTOR,
    )
    assert github.find_open_pr_closing_issue(runner, config=config, issue_number=56) is None
    with pytest.raises(AgentLoopError, match="open pull request listing is"):
        github.find_open_pr_closing_issue(runner, config=config, issue_number=56, strict=True)


def test_strict_child_discovery_raises_where_lenient_discovery_reads_empty(tmp_path):
    """`child-discovery-failure` (item-3): only the reopen inventory uses strict discovery."""
    config = make_config(tmp_path)
    runner = _SearchFaultRunner(
        search_fault="invalid-json", authenticated_actor=_M1390_ACTOR,
    )
    found, _searched, _direct = decomposition.discover_parent_child_issues(
        runner, config=config, parent_issue=56,
    )
    assert found == ()
    with pytest.raises(AgentLoopError, match="unreadable or possibly truncated"):
        decomposition.discover_parent_child_issues(
            runner, config=config, parent_issue=56, strict=True,
        )


@pytest.mark.parametrize(
    "make_record",
    [_pr_handoff, _one_shot_handoff, _decomposition_summary],
)
def test_comment_handoff_reason_names_each_parent_record_kind(make_record):
    """`refuse-handed-off-children`, `refuse-handed-off-pr` (comment inventory, no reads)."""
    comments = [SimpleNamespace(body=str(make_record("a" * 16)["body"]))]
    reason = plan_reopen.comment_handoff_reason(
        comments, issue_number=56, repo="OWNER/REPO", plan_hash="a" * 16,
    )
    assert reason is not None
    assert plan_reopen.comment_handoff_reason(
        comments, issue_number=57, repo="OWNER/REPO", plan_hash="a" * 16,
    ) is None


# --- preflight agreement ----------------------------------------------------------


def test_floor_preflight_refuses_with_the_gate_reason(tmp_path):
    """`preflight-agrees`: the floor preflight and the gate refuse identically."""
    history, plan_hash = _approved_history(tmp_path)
    history = [*history, _pr_handoff(plan_hash)]
    _runner, without_floor, _calls, _messages = _rerun(tmp_path, history, recovery="forbidden")
    runner, with_floor, calls, _messages = _rerun(
        tmp_path, history, recovery="forbidden", config_overrides={"min_reviewers": 1},
    )
    assert isinstance(with_floor, AgentLoopError) and isinstance(without_floor, AgentLoopError)
    assert str(with_floor) == str(without_floor)
    assert calls == [] and runner.issue_comments == history and _writes(runner) == []


def test_floor_preflight_judges_the_board_at_the_new_scope_start(tmp_path, monkeypatch):
    """`preflight-agrees`: no plan-only approved early return while a reopen is pending."""
    history, _plan_hash = _approved_history(tmp_path)
    real = plan_board_state.resolve_plan_board_state
    rounds: list[int] = []

    def spy(runner, **kwargs):
        rounds.append(kwargs["resume_round"](kwargs["config"], None))
        return real(runner, **kwargs)

    monkeypatch.setattr(plan_board_state, "resolve_plan_board_state", spy)
    monkeypatch.setattr(plan_first_loop_module, "resolve_plan_board_state", spy)
    runner, result, _calls, _messages = _rerun(
        tmp_path, history, config_overrides={"min_reviewers": 1},
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert result == 0
    # The preflight and the gate each floor-check round 2 before the boundary.
    assert rounds[:2] == [2, 2]
    assert len(_boundaries(runner.issue_comments)) == 1


def test_plan_startup_never_posts_an_operator_reopen_itself(tmp_path, monkeypatch):
    """`preflight-agrees`: a bypassed gate is an internal error, not a second posting site."""
    history, plan_hash = _approved_history(tmp_path)
    monkeypatch.setattr(
        plan_reopen, "apply_operator_reopen",
        lambda runner, *, config, issue_number, issue_context, parent_issue_context: issue_context,
    )
    runner, result, _calls, _messages = _rerun(tmp_path, history)
    assert isinstance(result, AgentLoopError)
    assert "internal error" in str(result) and plan_hash in str(result)
    assert runner.issue_comments == history and _writes(runner) == []


# --- dry run ------------------------------------------------------------------------


def test_reopen_under_dry_run_logs_a_no_op_and_posts_nothing(tmp_path):
    """`reopen-dry-run`: synthetic context; no check, no post, no internal error."""
    runner = _m1390_runner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    with patch_object_in(
        (plan_first_loop_module, plan_reopen), "log", wraps=plan_first_loop_module.log,
    ) as logged:
        result = run_issue_loop(
            runner, issue_number=56,
            config=_m1390_config(tmp_path, dry_run=True, reopen_approved_plan=True),
            plan_first=True,
        )
    messages = [str(call.args[-1]) for call in logged.call_args_list]
    assert result == 0
    assert any(
        "dry run; --reopen-approved-plan performs no reopen check and posts nothing" in m
        for m in messages
    )
    assert _boundaries(runner.issue_comments) == []
    assert not any("internal error" in m for m in messages)


# --- non-inheritance ----------------------------------------------------------------


def test_reopen_flag_is_reset_in_derived_configs_and_recovery_commands(tmp_path):
    """`non-inheritance`: approving-board, isolated-provider and rendered recovery commands."""
    from coding_review_agent_loop.managed_ci_recovery_commands import _render_recovery_command
    from coding_review_agent_loop.plan_verification import PlanVerificationContext
    from coding_review_agent_loop.reviewer_floor import BoardSeat
    from coding_review_agent_loop.semantic_dedupe import _isolated_provider_config

    config = make_config(
        tmp_path, reviewer=("codex", "claude"), reopen_approved_plan=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--plan-first", "--reopen-approved-plan",
            "--reviewer", "codex",
        ),
    )
    context = PlanVerificationContext(
        issue_number=56,
        original_reviewers=("Codex",),
        effective_reviewers=("Codex",),
        original_seats=(BoardSeat(seat_id="Codex", backend="codex"),),
        effective_seats=(BoardSeat(seat_id="Codex", backend="codex"),),
        policy="all-reviewers",
        primary_reviewer=None,
        seat_binding=None,
        signed_amendment_count=0,
    )
    assert context.config_for(config).reopen_approved_plan is False
    isolated, isolated_dir = _isolated_provider_config(config, "claude", "")
    try:
        assert isolated.reopen_approved_plan is False
    finally:
        import shutil

        shutil.rmtree(isolated_dir, ignore_errors=True)
    for target in ("issue", "pr"):
        rendered = _render_recovery_command(
            config, target=target, identifier=7, managed_ci=True,
            preserve_managed_options=True,
        )
        assert "--reopen-approved-plan" not in rendered


def test_child_planning_cycle_does_not_inherit_the_reopen_flag(tmp_path, monkeypatch, capsys):
    """`non-inheritance`: the parent's flag never reopens a child plan."""
    from test_child_plan_provenance import _capture_child_planning_config

    _parent, child_config, err = _capture_child_planning_config(
        tmp_path, monkeypatch, capsys, reopen_approved_plan=True,
    )
    assert child_config.reopen_approved_plan is False
    assert err.count("does not inherit --reopen-approved-plan") == 1


def test_reopen_flag_parses_only_with_issue_plan_first(capsys):
    """`non-inheritance` / scope-flag: CLI validation and mutual exclusion."""
    from agent_loop_helpers import FakeRunner
    from coding_review_agent_loop.cli import build_parser, main
    from coding_review_agent_loop.config import config_from_args

    parser = build_parser()
    base = ["issue", "56", "--repo", "OWNER/REPO"]
    assert parser.parse_args([*base, "--plan-first"]).reopen_approved_plan is False
    with_flag = parser.parse_args([*base, "--plan-first", "--reopen-approved-plan"])
    assert with_flag.reopen_approved_plan is True
    assert config_from_args(with_flag, FakeRunner()).reopen_approved_plan is True

    assert main([*base, "--reopen-approved-plan"]) == 1
    assert "--reopen-approved-plan requires --plan-first" in capsys.readouterr().err
    assert main(["pr", "7", "--repo", "OWNER/REPO", "--reopen-approved-plan"]) == 1
    assert "--reopen-approved-plan requires issue --plan-first" in capsys.readouterr().err
    assert main([
        *base, "--plan-first", "--reopen-approved-plan", "--plan-resume-keep-on-issue-edit",
    ]) == 1
    assert "cannot be combined with --plan-resume-keep-on-issue-edit" in (
        capsys.readouterr().err
    )


# --- execution history superseded by an operator reopen (review items 1 and 2) -------


def _execution_runner_class():
    from test_execution_decision_retirement import RetirementRunner

    class _ReopenExecutionRunner(RetirementRunner, _M1390Runner):
        """Execution-mode fixture that also serves the authenticated boundary POST."""

    return _ReopenExecutionRunner


def _matching_decision(history, plan_hash: str) -> dict[str, object]:
    """An archived decision whose identity equals what re-approving the same plan records."""
    from coding_review_agent_loop.execution_policy import (
        _current_execution_recommendation,
        _plan_subject,
    )

    plan = next(
        record.canonical_plan
        for record in _plan_round_records(SimpleNamespace(issue_comments=history))
        if record.role == "coder"
    )
    comments = [SimpleNamespace(body=str(c["body"])) for c in history]
    recommendation = _current_execution_recommendation(plan, comments)
    identity = recommendation.identity()
    return _orchestrator(format_execution_decision(ExecutionDecision(
        parent_issue=56, plan_hash=plan_hash, plan_subject=_plan_subject(plan),
        execution_strategy_contract_version=1, strategy=recommendation.strategy,
        topology_source=str(identity["topology_source"]),
        recommendation_digest=str(identity["recommendation_sha256"]),
        requested_policy="implement-one-shot", current_action="implement-one-shot",
    )))


def _checkpoint(plan_hash: str) -> dict[str, object]:
    phase = decomposition.PlanPhase(
        "Old", "Old scope", "None", "None", "low", "Tests", "Parent", "agent-pr",
    )
    return _orchestrator(decomposition.format_topology_checkpoint(decomposition.TopologyCheckpoint(
        56, plan_hash, "implement-by-phase", "model", (phase,),
    )))


@pytest.mark.parametrize(
    "archived",
    ["checkpoint-same-hash", "checkpoint-other-hash", "decision-same-hash",
     "decision-and-checkpoint-same-hash"],
)
def test_reopen_retires_unrealized_execution_history_in_execution_mode(tmp_path, archived):
    """`identical-plan-hash`, `reopen-approved-unhanded` (items 1 and 2, implement-one-shot).

    Execution decisions and topology checkpoints posted before the operator
    reopen are unrealized history: whatever their hash, and with or without a
    decision, the re-approved plan records a new decision that retires them
    and dispatches implementation, instead of adopting the archived decision
    or failing the checkpoint conflict check.
    """
    from test_plan_first_loop_execution_strategy import (
        _m1087_config,
        _m1087_implementation_dispatched,
        _m1087_posted_decisions,
    )

    history, plan_hash = _approved_history(tmp_path)
    other = "0123456789abcdef"
    extra = {
        "checkpoint-same-hash": [_checkpoint(plan_hash)],
        "checkpoint-other-hash": [_checkpoint(other)],
        "decision-same-hash": [_matching_decision(history, plan_hash)],
        "decision-and-checkpoint-same-hash": [
            _matching_decision(history, plan_hash), _checkpoint(plan_hash),
        ],
    }[archived]
    expected_retired = other if archived == "checkpoint-other-hash" else plan_hash
    runner = _execution_runner_class()(
        issue_comments=[*history, *extra], authenticated_actor=_M1390_ACTOR,
    )
    from dataclasses import replace

    result = run_issue_loop(
        runner, issue_number=56,
        config=replace(_m1087_config(tmp_path), reopen_approved_plan=True), plan_first=True,
    )

    assert result == 0
    assert len(_boundaries(runner.issue_comments)) == 1
    posted = _m1087_posted_decisions(runner)
    # A new decision is recorded, never adopted from the archived scope.
    assert len(posted) == 1
    assert posted[0].plan_hash == plan_hash
    assert posted[0].retires_plan_hashes == (expected_retired,)
    assert _m1087_implementation_dispatched(runner)
    # Archived records stay in the issue history, unedited.
    assert runner.issue_comments[: len(history) + len(extra)] == [*history, *extra]
    # On resume the superseding decision already retires the archived records:
    # nothing new is superseded, so new-scope records stay current.
    resume_config = _m1087_config(tmp_path)
    assert plan_reopen.reopen_superseded_execution_hashes(
        runner, config=resume_config, issue_number=56,
        comments=get_issue_context(runner, config=resume_config, issue_number=56).comments,
    ) == ()


def test_reopen_superseded_hashes_ignore_unadmitted_boundaries(tmp_path):
    """A forged operator-reopen boundary never retires execution history (items 1 and 2)."""
    from coding_review_agent_loop.plan_issue_scope import (
        render_scope_boundary,
        scope_boundary_payload,
    )

    history, plan_hash = _approved_history(tmp_path)
    forged = {
        "author": {"login": "mallory"}, "_rest_author_id": 999,
        "createdAt": "2026-05-23T03:00:00Z",
        "body": str(render_scope_boundary(scope_boundary_payload(
            issue_number=56, previous_issue_digest="a" * 16, current_issue_digest="a" * 16,
            archived_round_number=1, archived_plan_subject=None, reason="operator-reopen",
            superseded_plan_hash=plan_hash,
        ))),
    }
    comments = [*history, _checkpoint(plan_hash), forged]
    runner = _m1390_runner(comments)
    config = _m1390_config(tmp_path)
    live = get_issue_context(runner, config=config, issue_number=56).comments
    assert plan_reopen.reopen_superseded_execution_hashes(
        runner, config=config, issue_number=56, comments=live,
    ) == ()
    # The same history with the orchestrator-authored boundary does retire it.
    genuine = [*history, _checkpoint(plan_hash), _orchestrator(forged["body"])]
    runner = _m1390_runner(genuine)
    live = get_issue_context(runner, config=config, issue_number=56).comments
    assert plan_reopen.reopen_superseded_execution_hashes(
        runner, config=config, issue_number=56, comments=live,
    ) == (plan_hash,)


# --- #2138: --keep-approved-plan and the edited-approved-plan gate entries -----------


def test_keep_approved_plan_parses_only_with_issue_plan_first_and_not_with_reopen(capsys):
    """`cli-exclusivity-and-scope`: CLI validation; refusals read and post nothing."""
    from agent_loop_helpers import FakeRunner
    from coding_review_agent_loop.cli import build_parser, main
    from coding_review_agent_loop.config import config_from_args

    parser = build_parser()
    base = ["issue", "56", "--repo", "OWNER/REPO"]
    assert parser.parse_args([*base, "--plan-first"]).keep_approved_plan is False
    with_flag = parser.parse_args([*base, "--plan-first", "--keep-approved-plan"])
    assert with_flag.keep_approved_plan is True
    assert config_from_args(with_flag, FakeRunner()).keep_approved_plan is True

    assert main([*base, "--keep-approved-plan"]) == 1
    assert "--keep-approved-plan requires --plan-first" in capsys.readouterr().err
    assert main(["pr", "7", "--repo", "OWNER/REPO", "--keep-approved-plan"]) == 1
    assert "--keep-approved-plan requires issue --plan-first" in capsys.readouterr().err
    assert main([
        *base, "--plan-first", "--keep-approved-plan", "--reopen-approved-plan",
    ]) == 1
    err = capsys.readouterr().err
    assert "--keep-approved-plan" in err and "--reopen-approved-plan" in err
    assert "cannot be combined" in err
    # Combining it with the unapproved-plan opt-out is allowed.
    combined = parser.parse_args([
        *base, "--plan-first", "--keep-approved-plan", "--plan-resume-keep-on-issue-edit",
    ])
    assert config_from_args(combined, FakeRunner()).keep_approved_plan is True


@pytest.mark.parametrize("argv, expected", [
    (["issue", "56", "--plan-first", "--keep-approved-plan", "--reopen-approved-plan"],
     "--keep-approved-plan cannot be combined with --reopen-approved-plan"),
    (["issue", "56", "--keep-approved-plan"], "--keep-approved-plan requires --plan-first"),
    (["pr", "7", "--keep-approved-plan"], "--keep-approved-plan requires issue --plan-first"),
    (["issue", "56", "--reopen-approved-plan"], "--reopen-approved-plan requires --plan-first"),
    (["pr", "7", "--reopen-approved-plan"], "--reopen-approved-plan requires issue --plan-first"),
    (["issue", "56", "--plan-first", "--reopen-approved-plan",
      "--plan-resume-keep-on-issue-edit"],
     "cannot be combined with --plan-resume-keep-on-issue-edit"),
])
def test_approved_plan_option_refusals_happen_before_any_config_or_github_read(
    monkeypatch, capsys, argv, expected
):
    """`cli-exclusivity-and-scope`: no --repo, so config would detect the repo via gh."""
    import coding_review_agent_loop.cli as cli_module
    from coding_review_agent_loop.runner import Runner

    def forbidden_config(*_a, **_k):
        raise AssertionError("config_from_args must not run on a parse refusal")

    def forbidden_command(*_a, **_k):
        raise AssertionError("no GitHub read or post on a parse refusal")

    monkeypatch.setattr(cli_module, "config_from_args", forbidden_config)
    monkeypatch.setattr(Runner, "run", forbidden_command)
    assert cli_module.main(argv) == 1
    assert expected in capsys.readouterr().err


def test_keep_flag_is_reset_in_derived_configs_and_recovery_commands(tmp_path):
    """`cli-exclusivity-and-scope`: approving-board swap, isolated provider, rerun commands."""
    from coding_review_agent_loop.managed_ci_recovery_commands import _render_recovery_command
    from coding_review_agent_loop.semantic_dedupe import _isolated_provider_config

    config = make_config(
        tmp_path, reviewer=("codex", "claude"), keep_approved_plan=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--plan-first", "--keep-approved-plan",
            "--reviewer", "codex",
        ),
    )
    isolated, isolated_dir = _isolated_provider_config(config, "claude", "")
    try:
        assert isolated.keep_approved_plan is False
    finally:
        import shutil

        shutil.rmtree(isolated_dir, ignore_errors=True)
    for target in ("issue", "pr"):
        rendered = _render_recovery_command(
            config, target=target, identifier=7, managed_ci=True,
            preserve_managed_options=True,
        )
        assert "--keep-approved-plan" not in rendered
    # The invocation value survives the approving-board swap and its restore.
    assert "keep_approved_plan" in plan_execution._PLAN_BOARD_CONFIG_FIELDS
    swapped = replace(config, keep_approved_plan=False)
    assert plan_execution._restore_invocation_board(swapped, config).keep_approved_plan is True


def test_child_planning_cycle_does_not_inherit_the_keep_flag(tmp_path, monkeypatch, capsys):
    """`cli-exclusivity-and-scope`: the parent's flag never keeps a child plan."""
    from test_child_plan_provenance import _capture_child_planning_config

    _parent, child_config, err = _capture_child_planning_config(
        tmp_path, monkeypatch, capsys, keep_approved_plan=True,
    )
    assert child_config.keep_approved_plan is False
    assert err.count("does not inherit --keep-approved-plan") == 1


_DISPATCH_TARGETS = frozenset({
    "recover_issue_execution", "run_plan_first_loop", "_run_plan_first_loop",
})
_GATE = "apply_edited_approved_plan_gate"


def _called_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _dispatch_entry_inventory(src_dir: Path) -> dict[str, tuple[int, list[int], list[int]]]:
    """``module:function`` -> (gate call lines, dispatch call lines) outside issue_recovery."""
    found: dict[str, tuple[int, list[int], list[int]]] = {}
    for path in sorted(src_dir.rglob("*.py")):
        if path.name == "issue_recovery.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            gates: list[int] = []
            dispatches: list[int] = []
            for inner in ast.walk(node):
                if isinstance(inner, ast.Call):
                    name = _called_name(inner)
                    if name in _DISPATCH_TARGETS:
                        dispatches.append(inner.lineno)
                    elif name == _GATE:
                        gates.append(inner.lineno)
            if dispatches:
                found[node.name] = (node.lineno, gates, dispatches)
    return found


def test_every_plan_first_dispatch_entry_calls_the_edited_approved_plan_gate_first():
    """`gate-entry-inventory`: a new ungated plan-first entry fails this test."""
    src_dir = Path(plan_reopen.__file__).resolve().parent
    inventory = _dispatch_entry_inventory(src_dir)
    assert set(inventory) == {"run_issue_loop", "_run_child_planning_cycle"}
    for name, (_line, gates, dispatches) in inventory.items():
        assert gates, f"{name} dispatches plan-first work without {_GATE}"
        assert min(gates) < min(dispatches), (
            f"{name} must call {_GATE} before recovery or the plan loop"
        )


def test_dispatch_entry_inventory_flags_an_ungated_entry(tmp_path):
    """`gate-entry-inventory`: the scanner itself detects ungated and late-gated callers."""
    (tmp_path / "issue_recovery.py").write_text(
        "def recover_issue_execution():\n    run_plan_first_loop()\n", encoding="utf-8",
    )
    (tmp_path / "entry.py").write_text(
        "def gated():\n    apply_edited_approved_plan_gate()\n    recover_issue_execution()\n"
        "def ungated():\n    run_plan_first_loop()\n"
        "def late():\n    _run_plan_first_loop()\n    apply_edited_approved_plan_gate()\n",
        encoding="utf-8",
    )
    inventory = _dispatch_entry_inventory(tmp_path)
    assert set(inventory) == {"gated", "ungated", "late"}
    assert inventory["ungated"][1] == []
    late_gates, late_dispatches = inventory["late"][1], inventory["late"][2]
    assert min(late_gates) > min(late_dispatches)


def test_edited_approved_plan_notes_and_stop_carry_no_reserved_marker():
    """`scope-issue-note`: both note variants and the stop are plain prose."""
    from coding_review_agent_loop.errors import HumanDecisionRequiredError, parse_human_decision
    from coding_review_agent_loop.protocol_markers import scan_reserved_markers

    stop = plan_reopen.ApprovedShortcutEditDecision(
        kind=plan_reopen.EDIT_DECISION_NOT_HANDED_OFF, plan_hash="a" * 16,
        recorded_digest="1" * 16, current_digest="2" * 16,
    )
    handed = replace(
        stop, kind=plan_reopen.EDIT_DECISION_HANDED_OFF,
        handoff_reason="an issue-PR handoff record names PR #77",
    )
    for decision in (stop, handed):
        note = plan_reopen.render_edited_approved_plan_issue_note(56, decision)
        assert scan_reserved_markers(note) == ()
        assert "1" * 16 in note and "2" * 16 in note
    error = plan_reopen.render_edited_approved_plan_stop(56, stop)
    assert isinstance(error, HumanDecisionRequiredError)
    parts = parse_human_decision(str(error))
    assert parts is not None and parts.safe_to_automate is False
    assert "--keep-approved-plan" in parts.clear and "--reopen-approved-plan" in parts.clear
