"""Plan-first loop tests: edited-issue planning scope (#1543)."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import IssueContext, get_issue_context
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiCreationIntent,
    ManagedCiOutcome,
    UNPROTECTED_OVERRIDE_TRAILER,
)
from coding_review_agent_loop.protocol import validate_structured_plan_state
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in, patch_object_in
from issue_loop_test_helpers import (
    FakeRunner,
    _STEP_BACK_ANCHOR_PLANNER,
    _STEP_BACK_MARKER,
    _m1103_agent_calls,
    _m1103_blocking_chain,
    _m1103_fresh_base,
    _m1251_prompts,
    _m1251_run,
    _m943_decoded,
    _m943_interrupted_history,
    _m943_post_amendment,
    _m943_reduced_config,
    _m943_reviewer_calls,
    _managed_issue_handoff,
    _plan_round_records,
    _staged_plan_config,
)
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
    structured_pr_review,
    structured_issue_implementation,
)
import coding_review_agent_loop.validated_agent as validated_agent_module
from coding_review_agent_loop.panel_evidence import plan_issue_text_digest  # noqa: E402
from coding_review_agent_loop.plan_issue_scope import PLAN_SCOPE_BOUNDARY_MARKER


# --- Edited-issue planning scope (#1390) ---------------------------------------


_M1390_ACTOR = ("agent-loop", 4242)


_M1390_EDIT = {"body": "Narrowed scope: a two-command design."}


def _m1390_config(tmp_path, **overrides):
    values = {"reviewer": "codex", "max_rounds": 3}
    values.update(overrides)
    return make_config(tmp_path, **values)


class _M1390Runner(_FakeRunner):
    """FakeRunner that also serves the authenticated issue protocol-record POST."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command[:4] == ["gh", "api", "--method", "POST"] and command[4:5] == [
            "repos/OWNER/REPO/issues/56/comments"
        ]:
            from coding_review_agent_loop.runner import CommandResult

            recorded, cwd_path = self._record_command(args, cwd)
            body = json.loads(input_text or "{}")["body"]
            created = "2026-05-23T01:00:00Z"
            self.issue_comments.append(
                {
                    "author": {"login": _M1390_ACTOR[0]},
                    "_rest_author_id": _M1390_ACTOR[1],
                    "createdAt": created,
                    "body": body,
                }
            )
            return CommandResult(
                recorded, cwd_path,
                json.dumps({
                    "id": 10_000 + len(self.issue_comments), "created_at": created,
                    "body": body, "user": {"login": _M1390_ACTOR[0], "id": _M1390_ACTOR[1]},
                }),
                "", 0,
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def _m1390_runner(history=(), **kwargs):
    return _M1390Runner(
        issue_comments=list(history),
        authenticated_actor=_M1390_ACTOR,
        serve_rest_issue_comments=True,
        **kwargs,
    )


def _m1390_history(tmp_path, *, rounds=3, config=None):
    """An unapproved plan whose round ``rounds`` review blocked, under the original text."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, rounds, base=base)
    runner = _m1390_runner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            runner, issue_number=56,
            config=config or _m1390_config(tmp_path, max_rounds=rounds), plan_first=True,
        )
    return list(runner.issue_comments)


def _m1390_rerun(tmp_path, history, *, config=None, **runner_kwargs):
    runner = _m1390_runner(history, **runner_kwargs)
    with patch.object(
        plan_first_loop_module, "log", wraps=plan_first_loop_module.log
    ) as logged:
        try:
            result = run_issue_loop(
                runner, issue_number=56,
                config=config or _m1390_config(tmp_path, max_rounds=5), plan_first=True,
            )
        except AgentLoopError as exc:
            result = exc
    return runner, result, [str(call.args[-1]) for call in logged.call_args_list]


def _m1390_boundaries(runner):
    return [
        comment for comment in runner.issue_comments
        if PLAN_SCOPE_BOUNDARY_MARKER in comment["body"]
    ]


def _m1390_digest(**payload):
    issue = {"title": "Fix issue-mode context", "body": "Original issue body."}
    issue.update(payload)
    return plan_issue_text_digest(
        IssueContext(
            number=56, repo="OWNER/REPO", title=issue["title"], body=issue["body"],
            url=None, comments=(),
        )
    )


def test_m1390_edited_issue_starts_a_fresh_plan_in_a_new_scope(tmp_path):
    """`edited-issue-default-replan`, `prompt-excludes-archived-plan`, `fresh-round-identity`."""
    history = _m1390_history(tmp_path)
    old_digest, new_digest = _m1390_digest(), _m1390_digest(**_M1390_EDIT)
    assert {
        record.plan_issue_digest
        for record in _plan_round_records(SimpleNamespace(issue_comments=history))
        if record.role == "coder"
    } == {old_digest}

    runner, result, messages = _m1390_rerun(
        tmp_path, history,
        issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    assert any(
        f"issue title/body changed since plan round 3 (recorded digest {old_digest}, "
        f"current digest {new_digest})" in message
        for message in messages
    )
    assert any("starting a fresh plan at round 4" in message for message in messages)
    boundaries = _m1390_boundaries(runner)
    assert len(boundaries) == 1
    assert "Fresh plan starts at round 4" in boundaries[0]["body"]
    assert f"`{old_digest}`" in boundaries[0]["body"]
    assert f"`{new_digest}`" in boundaries[0]["body"]
    # Archived plan comments stay in the issue, unedited and undeleted.
    assert runner.issue_comments[: len(history)] == history
    assert not any(
        "DELETE" in cmd or any("PATCH" == part for part in cmd)
        for cmd, _cwd in runner.commands
    )
    assert any("Gap 3 remains." in comment["body"] for comment in history)
    new_records = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
    )
    coder = [record for record in new_records if record.role == "coder"]
    assert [record.round_number for record in coder] == [4]
    assert coder[0].prior_items == () and coder[0].prior_plan_subject is None
    assert coder[0].plan_issue_digest == new_digest
    sidecar = coder[0].assembled_plan_sidecar
    assert sidecar is not None and sidecar["round_number"] == 4
    assert {record.round_number for record in new_records} == {4}
    # The fresh planner prompt and the review prompt omit the archived plan
    # discussion, but keep the boundary audit text.
    planner_prompt = _m1251_prompts(runner, "claude")[0]
    review_prompt = _m1251_prompts(runner, "codex")[0]
    for prompt in (planner_prompt, review_prompt):
        assert "Close gap 2." not in prompt
        assert "Gap 3 remains." not in prompt
    assert "Planning scope restarted." in planner_prompt
    assert "Narrowed scope: a two-command design." in planner_prompt


def _m1390_legacy_history(tmp_path):
    """Planning history written before #1390: no planner digest, no checkpoint digest."""
    real = plan_first_loop_module.PostedRoundMetadata

    def legacy(**fields):
        fields.pop("plan_issue_digest", None)
        return real(**fields)

    with patch.object(plan_first_loop_module, "PostedRoundMetadata", legacy):
        return _m1390_history(tmp_path)


def _m1390_revision_outputs(first_round=3, last_round=3):
    """Round 3's planner revision after a resume (resolves the carried item-3)."""
    fresh, base = _m1103_fresh_base()
    _codex, _claude, base = _m1103_blocking_chain(1, first_round - 1, base=base)
    codex, claude, _base = _m1103_blocking_chain(first_round, last_round, base=base)
    return codex, claude


@pytest.mark.parametrize("legacy", [False, True])
def test_m1390_unchanged_or_legacy_history_resumes_the_recorded_round(tmp_path, legacy):
    """`no-reset-resume`: no boundary; the same round and carried items resume."""
    history = _m1390_legacy_history(tmp_path) if legacy else _m1390_history(tmp_path)
    coder_digests = {
        record.plan_issue_digest
        for record in _plan_round_records(SimpleNamespace(issue_comments=history))
        if record.role == "coder"
    }
    assert coder_digests == ({None} if legacy else {_m1390_digest()})
    _codex, claude = _m1390_revision_outputs()
    runner, _result, messages = _m1390_rerun(
        tmp_path, history,
        # An edit is invisible to a legacy history: it resumes as recorded.
        issue_payload=_M1390_EDIT if legacy else None,
        claude_outputs=claude,
    )

    assert _m1390_boundaries(runner) == []
    assert any("resuming round 3" in message for message in messages)
    assert not any("starting a fresh plan" in message for message in messages)
    no_digest = [m for m in messages if "no recorded issue digest is available" in m]
    assert bool(no_digest) is legacy
    # The carried item reaches the planner revision.
    assert "item-3" in _m1251_prompts(runner, "claude")[0]


def test_m1390_signed_comment_only_resumes_and_surfaces_the_requirement(tmp_path):
    """`signed-comment-only`: a signed comment never changes the issue digest."""
    history = _m1390_history(tmp_path)
    signed = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:00:00Z",
        "body": "Also preserve the audit trail.\n\n-- Human Reviewer",
    }
    _codex, claude = _m1390_revision_outputs()
    runner, _result, messages = _m1390_rerun(
        tmp_path, [*history, signed], claude_outputs=claude,
    )

    assert _m1390_boundaries(runner) == []
    assert any("resuming round 3" in message for message in messages)
    assert not any("issue title/body changed" in message for message in messages)
    revision_prompt = _m1251_prompts(runner, "claude")[0]
    assert "Also preserve the audit trail." in revision_prompt
    assert "item-3" in revision_prompt


def test_m1390_opt_out_keeps_the_existing_plan(tmp_path):
    """`opt-out-keeps-plan`: both digests are logged and round 3 resumes."""
    history = _m1390_history(tmp_path)
    _codex, claude = _m1390_revision_outputs()
    runner, _result, messages = _m1390_rerun(
        tmp_path, history,
        config=_m1390_config(tmp_path, max_rounds=5, plan_resume_keep_on_issue_edit=True),
        issue_payload=_M1390_EDIT,
        claude_outputs=claude,
    )

    assert _m1390_boundaries(runner) == []
    assert any(
        f"recorded digest {_m1390_digest()}, current digest {_m1390_digest(**_M1390_EDIT)}"
        in message
        for message in messages
    )
    assert any("keeping the existing plan" in message for message in messages)
    assert any("resuming round 3" in message for message in messages)
    assert "item-3" in _m1251_prompts(runner, "claude")[0]
    # The kept plan's next planner record carries the new digest.
    new_coder = [
        record
        for record in _plan_round_records(
            SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
        )
        if record.role == "coder"
    ]
    assert new_coder and new_coder[0].round_number == 4
    assert new_coder[0].plan_issue_digest == _m1390_digest(**_M1390_EDIT)


def test_m1390_restart_after_the_boundary_reopens_the_same_scope(tmp_path):
    """`crash-after-boundary`: no second boundary, fresh plan at the same start."""
    history = _m1390_history(tmp_path)
    crashed, result, _messages = _m1390_rerun(
        tmp_path, history, issue_payload=_M1390_EDIT, claude_outputs=[],
    )
    assert isinstance(result, AgentLoopError)
    boundary_index = next(
        index for index, comment in enumerate(crashed.issue_comments)
        if PLAN_SCOPE_BOUNDARY_MARKER in comment["body"]
    )
    assert boundary_index == len(history)
    # The crash happened before any fresh planner record was published.
    assert not any(
        record.round_number >= 4
        for record in _plan_round_records(
            SimpleNamespace(issue_comments=crashed.issue_comments)
        )
    )

    runner, result, messages = _m1390_rerun(
        tmp_path, crashed.issue_comments[: boundary_index + 1],
        issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    assert not any("issue title/body changed" in message for message in messages)
    assert not any("resuming round" in message for message in messages)
    new_records = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[boundary_index + 1:])
    )
    assert [r.round_number for r in new_records if r.role == "coder"] == [4]
    assert all(r.prior_items == () for r in new_records if r.role == "coder")
    # The archived plan is not resumed: its open item never reaches the planner.
    assert "item-3" not in _m1251_prompts(runner, "claude")[0]


@pytest.mark.parametrize("archived_rounds", [3, 15])
def test_m1390_round_budget_counts_from_the_scope_start(tmp_path, archived_rounds):
    """`budget-restarts-per-scope`: scope start 4 stops after 8, start 16 after 20."""
    history = _m1390_history(tmp_path, rounds=archived_rounds)
    start = archived_rounds + 1
    limit = start + 5 - 1
    fresh = structured_v1_plan_state()
    base = AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=start
    )
    codex, claude, _base = _m1103_blocking_chain(
        start, limit, base=base, item_offset=start - 1
    )
    runner, result, _messages = _m1390_rerun(
        tmp_path, history,
        config=_m1390_config(tmp_path, max_rounds=5),
        issue_payload=_M1390_EDIT,
        claude_outputs=[fresh, *claude],
        codex_outputs=codex,
    )

    assert isinstance(result, AgentLoopError)
    message = str(result)
    assert f"after round {limit}" in message
    assert f"counted from planning-scope start round {start}" in message
    rounds = {
        record.round_number
        for record in _plan_round_records(
            SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
        )
    }
    assert rounds == set(range(start, limit + 1))
    # Every review ran; no planner turn ran beyond the scope limit.
    assert _m1103_agent_calls(runner).count("codex") == 5
    assert _m1103_agent_calls(runner).count("claude") == 5


def test_m1390_fresh_scope_validation_diagnostic_survives_a_restart(tmp_path):
    """`diagnostic-recovery-after-reset` and `fresh-round-identity` (failed candidate)."""
    history = _m1390_history(tmp_path)
    tight = _m1390_config(tmp_path, max_rounds=5, plan_growth_max_chars=1000)
    failed, result, _messages = _m1390_rerun(
        tmp_path, history, config=tight, issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()] * 3,
    )
    assert isinstance(result, AgentLoopError)
    diagnostics = [
        comment["body"] for comment in failed.issue_comments
        if "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment["body"]
    ]
    assert len(diagnostics) == 1
    from coding_review_agent_loop.round_state import decode_plan_validation_diagnostic_body

    payload = decode_plan_validation_diagnostic_body(diagnostics[0])
    assert payload.target_coder_round == 4
    assert payload.candidate_kind == "plan_state"

    runner, _result, _messages = _m1390_rerun(
        tmp_path, failed.issue_comments, config=tight, issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()],
    )
    assert len(_m1390_boundaries(runner)) == 1
    planner_prompt = _m1251_prompts(runner, "claude")[0]
    assert "Trusted orchestration correction record" in planner_prompt


def _m1390_staged_approvals():
    return {
        "claude_outputs": [structured_v1_plan_state()],
        "codex_outputs": [structured_plan_review(state="approved")],
        "gemini_outputs": [
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ],
    }


def _m1390_staged_stalled_history(tmp_path):
    """A primary-then-panel plan whose primary blocked until the stall stop."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 2, base=base)
    runner = _m1390_runner(claude_outputs=[fresh, *claude], codex_outputs=codex)
    config = _staged_plan_config(tmp_path, max_rounds=8, plan_primary_stall_rounds=2)
    with pytest.raises(plan_first_loop_module.PlanPrePanelSafetyError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return list(runner.issue_comments), config


def test_m1390_rescope_isolates_staged_scheduler_history(tmp_path):
    """`boundary-filters-staged-history`: the archived stalled streak, panel
    state and calls-avoided count never bind the fresh scope."""
    history, config = _m1390_staged_stalled_history(tmp_path)
    archived = _plan_round_records(SimpleNamespace(issue_comments=history))
    archived_avoided = max(
        record.scheduler_calls_avoided or 0
        for record in archived
        if record.phase == "scheduler-prelaunch"
    )
    assert archived_avoided >= 2
    # Without the rescope, the archived streak trips the stall stop at once.
    stalled, error, _messages = _m1390_rerun(tmp_path, history, config=config)
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)

    runner, result, messages = _m1390_rerun(
        tmp_path, history, config=config, issue_payload=_M1390_EDIT,
        **_m1390_staged_approvals(),
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    assert not any("blocked 2 consecutive" in message for message in messages)
    fresh_scope = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
    )
    checkpoints = [r for r in fresh_scope if r.phase == "scheduler-prelaunch"]
    # The fresh scope opens in the primary phase at its start round; the panel
    # convenes only after the fresh primary approval.
    assert checkpoints[0].round_number == 4
    assert checkpoints[0].scheduler_phase == "primary"
    assert not checkpoints[0].scheduler_force_full
    assert not any(r.step_back_entries for r in fresh_scope)
    # Calls avoided restart from the scoped history, not the archived count.
    first_archived = next(r for r in archived if r.phase == "scheduler-prelaunch")
    assert checkpoints[0].scheduler_calls_avoided == first_archived.scheduler_calls_avoided
    assert _m1103_agent_calls(runner) == ["claude", "codex", "gemini"]


@pytest.mark.parametrize("operator_force_full", [False, True])
def test_m1390_archived_operator_force_full_does_not_leak(tmp_path, operator_force_full):
    """`force-full-replay-scoped`."""
    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 2, base=base)
    first = _m1390_runner(
        claude_outputs=[fresh, *claude],
        codex_outputs=codex,
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ] * 2,
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            first, issue_number=56,
            config=_staged_plan_config(tmp_path, max_rounds=2, plan_review_force_full=True),
            plan_first=True,
        )
    history = list(first.issue_comments)
    archived = [
        record for record in _plan_round_records(SimpleNamespace(issue_comments=history))
        if record.phase == "scheduler-prelaunch"
    ]
    assert archived and all(
        record.scheduler_force_full and record.scheduler_force_full_source == "operator"
        for record in archived
    )

    runner, _result, _messages = _m1390_rerun(
        tmp_path, history,
        config=_staged_plan_config(
            tmp_path, max_rounds=5, plan_review_force_full=operator_force_full
        ),
        issue_payload=_M1390_EDIT,
        **_m1390_staged_approvals(),
    )

    assert len(_m1390_boundaries(runner)) == 1
    checkpoint = next(
        record
        for record in _plan_round_records(
            SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
        )
        if record.phase == "scheduler-prelaunch"
    )
    assert checkpoint.round_number == 3
    if operator_force_full:
        # This invocation's own flag still applies.
        assert checkpoint.scheduler_force_full
        assert checkpoint.scheduler_phase == "full-board"
    else:
        assert not checkpoint.scheduler_force_full
        assert checkpoint.scheduler_phase == "primary"


def _m1390_link_comments(runner):
    return [
        comment for comment in runner.issue_comments
        if "AGENT_PLAN_BOARD_CHANGE" in comment["body"]
    ]


def _m1390_board_approvals():
    return {
        "claude_outputs": [structured_v1_plan_state()],
        "codex_outputs": [structured_plan_review(state="approved")],
        "gemini_outputs": [
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ],
    }


def test_m1390_edit_with_board_change_activates_from_the_scope_start(tmp_path):
    """`edit-with-board-change`: the link takes effect at the scope start; a
    restart before the fresh planner record skips anchor re-entry and posts
    no duplicate boundary or link."""
    from coding_review_agent_loop.board_change import PLAN_SURFACE

    history = _m1390_history(tmp_path)
    board = _m1390_config(tmp_path, max_rounds=5, reviewer=("codex", "gemini"))
    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=board, issue_payload=_M1390_EDIT,
        **_m1390_board_approvals(),
    )

    assert result == 0
    posted = runner.issue_comments[len(history):]
    assert PLAN_SCOPE_BOUNDARY_MARKER in posted[0]["body"]
    links = _m1390_link_comments(runner)
    assert len(links) == 1 and links[0] is posted[1]
    match = PLAN_SURFACE.link_re.search(links[0]["body"])
    import base64

    link = json.loads(base64.urlsafe_b64decode(match.group("payload")))
    assert link["effective_from_round"] == 4
    fresh_scope = _plan_round_records(SimpleNamespace(issue_comments=posted))
    assert {record.round_number for record in fresh_scope} == {4}
    assert {r.agent for r in fresh_scope if r.role == "reviewer"} == {"Codex", "Gemini"}
    assert all(r.reviewer_board_change_digest is not None for r in fresh_scope)
    # Anchor re-entry was skipped: the archived round-3 plan was never re-reviewed.
    assert _m1103_agent_calls(runner) == ["claude", "codex", "gemini"]

    # Restart after the boundary and the link, before the fresh planner record.
    crashed = [*history, posted[0], posted[1]]
    restarted, result, _messages = _m1390_rerun(
        tmp_path, crashed, config=board, issue_payload=_M1390_EDIT,
        **_m1390_board_approvals(),
    )
    assert result == 0
    assert len(_m1390_boundaries(restarted)) == 1
    assert len(_m1390_link_comments(restarted)) == 1
    assert _m1103_agent_calls(restarted) == ["claude", "codex", "gemini"]
    assert [
        record.round_number
        for record in _plan_round_records(
            SimpleNamespace(issue_comments=restarted.issue_comments[len(crashed):])
        )
        if record.role == "coder"
    ] == [4]


def test_m1390_below_floor_board_is_refused_before_the_boundary(tmp_path):
    """`edit-with-board-change` (floor variant): preflight and loop both refuse
    the plan board at the scope start before any boundary is published."""
    from coding_review_agent_loop.plan_verification import (
        classify_plan_approval,
        enforce_issue_board_floor,
    )

    fresh, base = _m1103_fresh_base()
    codex, claude, _base = _m1103_blocking_chain(1, 3, base=base)
    first = _m1390_runner(
        claude_outputs=[fresh, *claude], codex_outputs=codex,
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ] * 3,
    )
    with pytest.raises(AgentLoopError):
        run_issue_loop(
            first, issue_number=56,
            config=_m1390_config(tmp_path, max_rounds=3, reviewer=("codex", "gemini")),
            plan_first=True,
        )
    history = list(first.issue_comments)
    shrunk = _m1390_config(tmp_path, max_rounds=5, reviewer="codex", min_reviewers=2)

    # Entry preflight.
    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=shrunk, issue_payload=_M1390_EDIT,
        **_m1390_board_approvals(),
    )
    assert isinstance(result, AgentLoopError)
    assert "plan reviewer board" in str(result)
    assert _m1390_boundaries(runner) == []
    assert _m1103_agent_calls(runner) == []

    # The read-only classification the preflight routes on.
    issue = get_issue_context(runner, config=shrunk, issue_number=56)
    assert classify_plan_approval(
        issue.comments, config=shrunk, issue_number=56, scope_start_round=4,
    ).state == "empty-scope"
    # The same board passes once the plan floor is satisfied.
    enforce_issue_board_floor(
        replace(shrunk, reviewer=("codex", "gemini")), issue.comments, issue_number=56,
        runner=runner, planning=True, issue_digest=plan_issue_text_digest(issue),
    )

    # Plan startup refuses on its own too, before posting the boundary.
    loop_runner = _m1390_runner(history, issue_payload=_M1390_EDIT)
    with pytest.raises(AgentLoopError, match="plan reviewer board"):
        plan_first_loop_module._run_plan_first_loop(
            loop_runner,
            issue_number=56,
            config=shrunk,
            memory=None,
            issue_context=get_issue_context(loop_runner, config=shrunk, issue_number=56),
            usage_context=validated_agent_module._new_usage_context(shrunk),
        )
    assert _m1390_boundaries(loop_runner) == []
    assert _m1103_agent_calls(loop_runner) == []


def test_m1390_active_board_link_governs_the_fresh_scope(tmp_path):
    """`board-lineage-survives`: lineage resolved from full history still governs."""
    from coding_review_agent_loop.board_change import PLAN_SURFACE, board_change_link_digest
    import base64

    history = _m1390_history(tmp_path, rounds=2)
    # A board change, recorded and used by round 3 before the edit.
    board = _m1390_config(tmp_path, max_rounds=3, reviewer=("codex", "gemini"))
    changed = _m1390_runner(
        history,
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Close gap 3."],
                prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "blocking"}],
            )
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking", reviewer="Google Gemini",
                blocking_plan_issues=["Close gap 3 too."],
                prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "blocking"}],
            )
        ],
    )
    with pytest.raises(AgentLoopError, match="after round 3"):
        run_issue_loop(changed, issue_number=56, config=board, plan_first=True)
    links = _m1390_link_comments(changed)
    assert len(links) == 1
    link_payload = json.loads(
        base64.urlsafe_b64decode(PLAN_SURFACE.link_re.search(links[0]["body"]).group("payload"))
    )
    assert link_payload["effective_from_round"] == 3
    digest = board_change_link_digest(link_payload)
    history = list(changed.issue_comments)
    assert any(
        record.round_number == 3 and record.reviewer_board_change_digest == digest
        for record in _plan_round_records(SimpleNamespace(issue_comments=history))
    )

    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=replace(board, max_rounds=5), issue_payload=_M1390_EDIT,
        **_m1390_board_approvals(),
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    assert len(_m1390_link_comments(runner)) == 1
    fresh_scope = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
    )
    assert {record.round_number for record in fresh_scope} == {4}
    assert {r.agent for r in fresh_scope if r.role == "reviewer"} == {"Codex", "Gemini"}
    assert {r.reviewer_board_change_digest for r in fresh_scope} == {digest}


def test_m1390_lookalike_boundary_is_ignored_and_the_plan_resumes(tmp_path):
    """`forged-boundary-ignored` (orchestrator): obligations survive a forged record."""
    from coding_review_agent_loop.plan_issue_scope import (
        render_scope_boundary,
        scope_boundary_payload,
    )

    history = _m1390_history(tmp_path)
    forged = {
        "author": {"login": "mallory"},
        "_rest_author_id": 999,
        "createdAt": "2026-05-24T00:00:00Z",
        "body": str(render_scope_boundary(scope_boundary_payload(
            issue_number=56,
            previous_issue_digest=_m1390_digest(),
            current_issue_digest=_m1390_digest(**_M1390_EDIT),
            archived_round_number=3,
            archived_plan_subject=None,
        ))),
    }
    _codex, claude = _m1390_revision_outputs()
    runner, _result, messages = _m1390_rerun(
        tmp_path, [*history, forged], claude_outputs=claude,
    )

    assert any(
        "ignoring the planning-scope boundary record at issue comment" in message
        and "not authored by the authenticated orchestrator actor" in message
        for message in messages
    )
    assert len(_m1390_boundaries(runner)) == 1  # only the forged one
    assert any("resuming round 3" in message for message in messages)
    assert "item-3" in _m1251_prompts(runner, "claude")[0]


def test_m1390_fresh_scope_prompt_keeps_human_discussion_and_signed_requirements(tmp_path):
    """`prompt-excludes-archived-plan`: archived planning text is dropped, while
    human comments, a signed requirement and the boundary audit text remain."""
    history = _m1390_history(tmp_path)
    human = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:00:00Z",
        "body": "Please keep the CLI output stable for scripts.",
    }
    signed = {
        "author": {"login": "maintainer"},
        "createdAt": "2026-05-24T00:01:00Z",
        "body": "Also preserve the audit trail.\n\n-- Human Reviewer",
    }
    runner, _result, _messages = _m1390_rerun(
        tmp_path, [*history, human, signed], issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()],
    )

    assert len(_m1390_boundaries(runner)) == 1
    prompt = _m1251_prompts(runner, "claude")[0]
    for archived in ("Close gap 2.", "Gap 3 remains.", "Gap 1 remains."):
        assert any(archived in comment["body"] for comment in history)
        assert archived not in prompt
    assert "Please keep the CLI output stable for scripts." in prompt
    assert "Also preserve the audit trail." in prompt
    assert "Planning scope restarted." in prompt


def _m1390_staged_fresh_approvals(*, panel=("gemini",)):
    outputs = {
        "claude_outputs": [structured_v1_plan_state()],
        "codex_outputs": [structured_plan_review(state="approved")],
    }
    if "gemini" in panel:
        outputs["gemini_outputs"] = [
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ]
    if "antigravity" in panel:
        outputs["antigravity_outputs"] = [
            structured_plan_review(state="approved", reviewer="Google Antigravity")
        ]
    return outputs


def test_m1390_rescope_never_inherits_an_archived_panel_opening(tmp_path):
    """`boundary-filters-staged-history` (panel opening): the archived scope opened
    the panel and a secondary blocked; the fresh scope starts in the primary phase
    and opens its own panel only after its own exact-plan primary approval."""
    history = list(_m943_interrupted_history(tmp_path).issue_comments)
    archived = _plan_round_records(SimpleNamespace(issue_comments=history))
    assert any(
        record.phase == "scheduler-prelaunch" and record.scheduler_phase != "primary"
        for record in archived
    ), "the archived history must carry a panel opening"
    assert any(
        record.role == "reviewer" and record.agent == "Antigravity" and record.state == "blocking"
        for record in archived
    )
    config = _staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8)

    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=config, issue_payload=_M1390_EDIT,
        **_m1390_staged_fresh_approvals(panel=("gemini", "antigravity")),
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    fresh = _plan_round_records(SimpleNamespace(issue_comments=runner.issue_comments[len(history):]))
    checkpoints = [r for r in fresh if r.phase == "scheduler-prelaunch"]
    assert [(r.round_number, r.scheduler_phase) for r in checkpoints][:2] == [
        (4, "primary"), (5, "secondary-audit"),
    ]
    # Only the primary reviews round 4; the panel convenes at round 5.
    assert _m943_reviewer_calls(runner) == ["codex", "gemini", "agy"]
    assert {r.agent for r in fresh if r.role == "reviewer" and r.round_number == 4} == {"Codex"}
    # The archived Antigravity finding is not carried into the fresh scope.
    assert all(item.reviewer != "Antigravity" for r in fresh for item in r.prior_items)


def test_m1390_rescope_never_inherits_archived_step_back_stall_or_growth_count(tmp_path):
    """`boundary-filters-staged-history` (step-back, stall streak, growth count)."""
    first, _error = _m1251_run(
        tmp_path, last_round=5, max_rounds=5, plan_step_back_escalation_rounds=5
    )
    history = list(first.issue_comments)
    archived = _plan_round_records(SimpleNamespace(issue_comments=history))
    assert any(record.step_back_entries for record in archived)
    archived_candidates = [r for r in archived if r.role == "coder"]
    assert len(archived_candidates) == 5
    # Tight thresholds every archived signal would trip: an escalation after one
    # rejected step-back, a three-round stall streak, and three planner revisions.
    config = _staged_plan_config(
        tmp_path, max_rounds=8, plan_growth_max_chars=4500, plan_growth_max_revisions=3,
        plan_step_back_escalation_rounds=1, plan_primary_stall_rounds=3,
    )
    stopped, error, _messages = _m1390_rerun(tmp_path, history, config=config)
    assert isinstance(error, plan_first_loop_module.PlanPrePanelSafetyError)

    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=config, issue_payload=_M1390_EDIT,
        **_m1390_staged_fresh_approvals(),
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    fresh = _plan_round_records(SimpleNamespace(issue_comments=runner.issue_comments[len(history):]))
    assert [r.round_number for r in fresh if r.role == "coder"] == [6]
    assert not any(r.step_back_entries for r in fresh)
    assert _m1103_agent_calls(runner) == ["claude", "codex", "gemini"]
    planner_prompt = _m1251_prompts(runner, "claude")[0]
    assert _STEP_BACK_MARKER not in planner_prompt
    assert _STEP_BACK_ANCHOR_PLANNER not in planner_prompt
    # With five archived candidates counted, the fresh one-shot plan would cross
    # the three-revision growth signal and fail validation without a justification.
    assert not any(
        "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment["body"]
        for comment in runner.issue_comments
    )


def _m1390_amendment_audits(runner):
    return [c["body"] for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]


def test_m1390_pending_amendment_activates_at_the_fresh_scope_start(tmp_path):
    """`edit-with-board-change` (pending signed amendment) and the preflight: the
    amendment must name the scope start, activates there once, and a restart
    after the boundary and its audit duplicates neither."""
    from dataclasses import replace as dc_replace

    history_runner = _m943_interrupted_history(tmp_path)
    reduced = dc_replace(_m943_reduced_config(tmp_path), min_reviewers=2)

    # An amendment naming the archived re-entry round is refused by the shared
    # preflight decision before any boundary is published.
    stale = _m1390_runner(list(history_runner.issue_comments))
    _m943_post_amendment(stale, effective_from_round=3)
    stale_runner, result, _messages = _m1390_rerun(
        tmp_path, stale.issue_comments, config=reduced, issue_payload=_M1390_EDIT,
    )
    assert isinstance(result, AgentLoopError) and "re-enters round 4" in str(result)
    assert '"effective_from_round": 4' in str(result)
    assert _m1390_boundaries(stale_runner) == [] and _m943_reviewer_calls(stale_runner) == []

    pending = _m1390_runner(list(history_runner.issue_comments))
    _m943_post_amendment(pending, effective_from_round=4)
    history = list(pending.issue_comments)
    real_post = architecture_contract.post_issue_comment

    def stop_after_audit(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if "Reviewer board amendment applied." in str(kwargs["body"]):
            raise KeyboardInterrupt
        return result

    crashed = _m1390_runner(history, issue_payload=_M1390_EDIT)
    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=stop_after_audit):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(crashed, issue_number=56, config=reduced, plan_first=True)
    assert len(_m1390_boundaries(crashed)) == 1
    (audit,) = _m1390_amendment_audits(crashed)
    assert "Activation round: 4" in audit
    assert _m943_reviewer_calls(crashed) == []

    runner, result, _messages = _m1390_rerun(
        tmp_path, crashed.issue_comments, config=reduced, issue_payload=_M1390_EDIT,
        **_m1390_staged_fresh_approvals(),
    )
    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    assert len(_m1390_amendment_audits(runner)) == 1
    fresh = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(crashed.issue_comments):])
    )
    assert [r.round_number for r in fresh if r.role == "coder"] == [4]
    checkpoints = [r for r in fresh if r.phase == "scheduler-prelaunch"]
    assert checkpoints and all(r.reviewer_board_amendment_digest is not None for r in checkpoints)
    assert all(
        tuple(r.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for r in checkpoints
    )
    assert _m943_reviewer_calls(runner) == ["codex", "gemini"]


def test_m1390_amendment_used_before_the_edit_stays_active_and_is_not_pending(tmp_path):
    """`board-lineage-survives` (active signed amendment)."""
    runner = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(runner, effective_from_round=3)
    real_post = architecture_contract.post_issue_comment

    def stop_after_amended_checkpoint(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if metadata is not None and metadata.reviewer_board_amendment_digest is not None:
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=stop_after_amended_checkpoint):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    history = list(runner.issue_comments)
    (audit,) = _m1390_amendment_audits(runner)
    used = {
        r.reviewer_board_amendment_digest
        for r in _plan_round_records(runner) if r.reviewer_board_amendment_digest
    }
    assert len(used) == 1

    fresh_runner, result, messages = _m1390_rerun(
        tmp_path, history, config=_m943_reduced_config(tmp_path), issue_payload=_M1390_EDIT,
        **_m1390_staged_fresh_approvals(),
    )

    assert result == 0
    assert len(_m1390_boundaries(fresh_runner)) == 1
    # The used amendment is neither re-activated nor reported pending again.
    assert _m1390_amendment_audits(fresh_runner) == [audit]
    assert not any("re-enters round" in m for m in messages)
    fresh = _plan_round_records(SimpleNamespace(issue_comments=fresh_runner.issue_comments[len(history):]))
    assert [r.round_number for r in fresh if r.role == "coder"] == [4]
    checkpoints = [r for r in fresh if r.phase == "scheduler-prelaunch"]
    assert checkpoints and {r.reviewer_board_amendment_digest for r in checkpoints} == used
    assert all(
        tuple(r.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for r in checkpoints
    )
    # Antigravity, removed by the amendment, never reviews the fresh scope.
    assert _m943_reviewer_calls(fresh_runner) == ["codex", "gemini"]


def test_m1390_active_primary_changing_link_governs_the_fresh_scope(tmp_path):
    """`board-lineage-survives` (active primary-changing board link)."""
    import base64

    from coding_review_agent_loop.board_change import PLAN_SURFACE, board_change_link_digest

    history, config = _m1390_staged_stalled_history(tmp_path)
    swapped = replace(config, primary_plan_reviewer="gemini")
    changed = _m1390_runner(
        history,
        gemini_outputs=[
            structured_plan_review(
                state="blocking", reviewer="Google Gemini",
                blocking_plan_issues=["Name the migration owner."],
                prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            )
        ],
    )
    real_post = architecture_contract.post_issue_comment

    def stop_after_new_primary_review(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(str(kwargs["body"]))
        if metadata is not None and metadata.role == "reviewer" and metadata.agent == "Gemini":
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=stop_after_new_primary_review):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(changed, issue_number=56, config=swapped, plan_first=True)
    links = _m1390_link_comments(changed)
    assert len(links) == 1
    link_payload = json.loads(
        base64.urlsafe_b64decode(PLAN_SURFACE.link_re.search(links[0]["body"]).group("payload"))
    )
    assert link_payload["old_board"]["primary"] == "Codex"
    assert link_payload["new_board"]["primary"] == "Gemini"
    digest = board_change_link_digest(link_payload)
    history = list(changed.issue_comments)
    governed = [
        r for r in _plan_round_records(SimpleNamespace(issue_comments=history))
        if r.reviewer_board_change_digest == digest
    ]
    assert any(r.role == "reviewer" and r.agent == "Gemini" for r in governed)
    start = max(r.round_number for r in _plan_round_records(SimpleNamespace(issue_comments=history))) + 1

    runner, result, _messages = _m1390_rerun(
        tmp_path, history, config=replace(swapped, max_rounds=8), issue_payload=_M1390_EDIT,
        claude_outputs=[structured_v1_plan_state()],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    assert result == 0
    assert len(_m1390_boundaries(runner)) == 1
    assert len(_m1390_link_comments(runner)) == 1
    fresh = _plan_round_records(SimpleNamespace(issue_comments=runner.issue_comments[len(history):]))
    assert [r.round_number for r in fresh if r.role == "coder"] == [start]
    assert {r.reviewer_board_change_digest for r in fresh} == {digest}
    checkpoints = [r for r in fresh if r.phase == "scheduler-prelaunch"]
    assert checkpoints[0].round_number == start
    assert checkpoints[0].scheduler_phase == "primary"
    assert checkpoints[0].scheduler_primary_reviewer == "Gemini"
    # The changed primary reviews first; Codex audits only as the panel.
    assert _m1103_agent_calls(runner) == ["claude", "gemini", "codex"]


@pytest.mark.parametrize("max_rounds", [1, 2])
def test_m1390_staged_budget_exhausts_with_a_pending_phase_advance_at_the_scope_limit(
    tmp_path, max_rounds
):
    """`budget-restarts-per-scope` (primary-then-panel): the fresh scope's primary
    approves its candidate at the scope-relative limit, so the reviewer-only
    secondary-audit advance is still outstanding and the run stops there."""
    history, config = _m1390_staged_stalled_history(tmp_path)
    start = 4
    limit = start + max_rounds - 1
    fresh = structured_v1_plan_state()
    claude, codex = [fresh], []
    if max_rounds == 2:
        base = AuthenticatedPlanState.from_plan(
            validate_structured_plan_state(fresh), round_number=start
        )
        block_codex, block_claude, _base = _m1103_blocking_chain(
            start, start, base=base, item_offset=start - 1
        )
        claude += block_claude
        codex += block_codex
        codex.append(structured_plan_review(
            state="approved",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ))
    else:
        codex.append(structured_plan_review(state="approved"))
    runner, result, _messages = _m1390_rerun(
        tmp_path, history,
        config=replace(config, max_rounds=max_rounds, plan_primary_stall_rounds=0),
        issue_payload=_M1390_EDIT,
        claude_outputs=claude, codex_outputs=codex,
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    assert isinstance(result, AgentLoopError)
    message = str(result)
    assert "reviewer-only plan phase advance was still pending" in message
    assert f"counted from planning-scope start round {start}" in message
    assert "Gemini" in message
    assert len(_m1390_boundaries(runner)) == 1
    fresh_records = _plan_round_records(
        SimpleNamespace(issue_comments=runner.issue_comments[len(history):])
    )
    assert max(record.round_number for record in fresh_records) == limit
    # No secondary audit, planner turn, candidate or phase-advance record past the limit.
    assert not any(
        record.scheduler_phase not in (None, "primary") for record in fresh_records
    )
    assert [r.round_number for r in fresh_records if r.role == "coder"] == list(
        range(start, limit + 1)
    )
    assert _m1103_agent_calls(runner).count("claude") == max_rounds
    assert "gemini" not in _m1103_agent_calls(runner)
    assert not any("phase advance to round" in c["body"] for c in runner.issue_comments)


@pytest.mark.parametrize("reject", [False, True])
def test_direct_managed_issue_first_turn_is_guarded_and_bypass_dispatch_audited(
    tmp_path, monkeypatch, reject,
):
    """#1366: the creation intent guards the first coder turn, and a bypass
    ordinary dispatch on the reserved branch is reported after the created PR
    is authenticated, never as qualification -- also when the report is
    rejected after that authentication (item-11)."""
    import datetime
    import os
    import shutil
    import subprocess

    from coding_review_agent_loop import scratch
    from coding_review_agent_loop.runner import CommandResult

    bin_dir = tmp_path / "real-gh-bin"
    bin_dir.mkdir()
    record = bin_dir / "calls.txt"
    (bin_dir / "gh").write_text(f"#!/bin/sh\necho \"$@\" >> '{record}'\n", encoding="utf-8")
    (bin_dir / "gh").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setattr(scratch, "scratch_root", lambda: tmp_path / "scratch")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    bypass = {
        "id": 4242, "name": "CI", "display_title": "CI", "event": "workflow_dispatch",
        "head_sha": "abc123", "head_branch": "agent-loop/managed-56", "created_at": now,
        "html_url": "https://github.com/OWNER/REPO/actions/runs/4242",
        "repository": {"full_name": "OWNER/REPO"},
    }

    class IssueRunner(FakeRunner):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.attempts = []
            self.audit_queries = []
            self.launch_envs = []

        def run_with_log(self, args, *, cwd, env=None, **kwargs):
            self.launch_envs.append((str(args[0]), dict(env or {})))
            if str(args[0]) == "claude":
                merged = {**os.environ, **(env or {})}
                gh = shutil.which("gh", path=merged["PATH"])
                self.attempts.append(subprocess.run(
                    [gh, "workflow", "run", "ci.yml"], env=merged,
                    capture_output=True, text=True, timeout=60,
                ))
            return super().run_with_log(args, cwd=cwd, env=env, **kwargs)

        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            cmd = [str(arg) for arg in args]
            if cmd[:3] == ["gh", "api", "--paginate"] and "/actions/runs?" in cmd[3]:
                self.audit_queries.append(cmd[3])
                cmd, cwd_path = self._record_command(cmd, cwd)
                return CommandResult(
                    cmd, cwd_path, json.dumps({"total_count": 1, "workflow_runs": [bypass]}), "", 0
                )
            return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)

    nonce = "direct-managed-nonce"
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce=nonce,
    )
    handoff = _managed_issue_handoff(nonce=nonce)
    runner = IssueRunner(
        claude_outputs=[
            # A report citing an out-of-checkout receipt is rejected only after
            # the managed draft has been authenticated.
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest"],
                test_observations=[{
                    "command": "cd /outside && python -m pytest",
                    "receipt_id": "outside-receipt",
                    "claim": "current-result",
                }],
            )
            if reject else
            "Implemented the issue.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[structured_pr_review(state="approved", summary="Approved managed draft.")],
        pr_payload={
            "body": f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}",
            "headRefName": "agent-loop/managed-56",
            "headRefOid": "abc123",
        },
    )
    config = make_config(
        tmp_path, auto_merge=True, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    dispatches = []
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: intent)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(issue_implementation, "publish_issue_created_authorization", lambda *_a, **_k: handoff)
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci",
        lambda *_a, **_k: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "publish_round_readiness", lambda *_a, **_k: None)
    monkeypatch.setattr(
        pr_loop, "dispatch_final_qualification",
        lambda *_a, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification",
        lambda *_a, **_k: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(pr_loop, "prepare_v2_merge", lambda *_a, **_k: None)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *_a, **_k: None)

    if reject:
        with pytest.raises(AgentLoopError) as rejected:
            run_issue_loop(runner, issue_number=56, config=config)
        notes = [
            comment["body"] for comment in runner.pr_payload.get("comments", [])
            if "Coder-turn CI dispatch audit" in comment["body"]
        ]
        assert len(notes) == 1
        assert "run 4242 on `abc123`" in notes[0]
        assert "non-authoritative diagnostic, not qualification" in notes[0]
        assert "coder response for this turn was rejected" in notes[0]
        assert "Coder-turn" not in str(rejected.value)
        assert dispatches == []
        assert [attempt.returncode for attempt in runner.attempts] == [3]
        assert not record.exists()
        return
    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    # The guard refused the coder's gh trigger; the real gh was never reached.
    assert [attempt.returncode for attempt in runner.attempts] == [3]
    assert not record.exists()
    coder_env = next(env for name, env in runner.launch_envs if name == "claude")
    assert coder_env["AGENT_LOOP_CODER_CI_DISPATCH_GUARD"] == "managed"
    assert not any(
        "AGENT_LOOP_CODER_GUARD_DIR" in env for name, env in runner.launch_envs if name == "codex"
    )
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "This pull request uses managed CI." in coder_prompt
    # The bypass run on the reserved branch is reported once, as a diagnostic.
    assert any("branch=agent-loop%2Fmanaged-56" in query for query in runner.audit_queries)
    bodies = [comment["body"] for comment in runner.pr_payload.get("comments", [])]
    noted = [body for body in bodies if "run 4242 on `abc123`" in body]
    assert len(noted) == 1
    assert "non-authoritative diagnostic, not qualification" in noted[0]
    # Never qualification: the orchestrator's exact-head dispatch still runs once.
    assert [dispatch["expected_head_sha"] for dispatch in dispatches] == ["abc123"]
