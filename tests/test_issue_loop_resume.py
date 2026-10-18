"""Issue-loop tests: issue resume and approved-plan routing (#1543)."""

import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    approved_plan_hash,
    format_one_shot_impl_handoff_comment,
)
from coding_review_agent_loop.errors import AgentInvocationError
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    FakeRunner,
    _M1088_ACTOR,
    _initial_plan_state,
    _m1088_comment,
    _m1090_legacy_history,
    _provenance_pages,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    command_index,
    make_config,
    structured_plan_review,
    structured_plan_state,
    structured_v1_plan_state,
    structured_issue_implementation,
)
from coding_review_agent_loop.decomposition import (  # noqa: E402
    ExecutionDecision as _M1088ExecutionDecision,
    format_execution_decision as _m1088_format_execution_decision,
)


def test_issue_loop_plan_first_can_implement_after_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "Approved implementation plan" in claude_calls[1][-1]
    assert "include a GitHub closing phrase targeting issue #56" in claude_calls[1][-1]
    first_claude_index = command_index(runner.commands, ["claude", "--print"])
    fetch_index = command_index(runner.commands, ["git", "fetch", "origin"])
    switch_index = command_index(runner.commands, ["git", "switch", "main"])
    second_claude_index = command_index(runner.commands, ["claude", "--print"], start=first_claude_index + 1)
    assert first_claude_index < fetch_index < switch_index < second_claude_index
    assert len(runner.comments) == 7
    assert "<!-- AGENT_ISSUE_PR_HANDOFF:" in runner.comments[3]
    assert "<!-- AGENT_PLAN_ONE_SHOT_IMPL:" in runner.comments[4]
    assert runner.comments[5].startswith("## Issue implementation")
    assert runner.comments[6].startswith("**Review verdict:** Approved\n\nLGTM.")


def test_issue_loop_plan_first_can_override_implementation_model(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "result": structured_plan_state(
                        summary="Make the change.", plan_steps=["Make the change."]
                    ),
                    "session_id": "planning-session",
                }
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        claude_model="claude-fable-5",
        implementation_coder_model="claude-sonnet-5",
    )

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert claude_calls[0][claude_calls[0].index("--model") + 1] == "claude-fable-5"
    assert claude_calls[1][claude_calls[1].index("--model") + 1] == "claude-sonnet-5"
    assert "--resume" not in claude_calls[1]
    assert runner.comments[5].startswith("## Issue implementation")


def test_issue_loop_rejects_pr_without_issue_reference_in_body(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Summary only.",
        },
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="does not reference issue #56") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    assert "Edit the PR description on GitHub" in str(excinfo.value)
    assert "rerun the orchestrator as `agent-loop pr 77` to continue the review" in str(excinfo.value)


def test_issue_loop_plan_first_implementation_rejects_pr_without_issue_reference_in_body(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Summary only.",
        },
    )
    config = make_config(tmp_path, reviewer=("codex",))

    with pytest.raises(AgentLoopError, match="does not reference issue #56") as excinfo:
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )

    assert "Edit the PR description on GitHub" in str(excinfo.value)
    assert "rerun the orchestrator as `agent-loop pr 77` to continue the review" in str(excinfo.value)


def test_issue_loop_plan_first_one_shot_posts_handoff_after_pr_creation(tmp_path):
    plan = _initial_plan_state(summary="Make the change.")
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert (
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)
        == 0
    )

    handoff_comments = [c for c in runner.comments if "<!-- AGENT_PLAN_ONE_SHOT_IMPL:" in c]
    assert len(handoff_comments) == 1
    assert f"Plan hash: {approved_plan_hash(plan)}" in handoff_comments[0]
    assert "Plan subject:" in handoff_comments[0]
    assert "PR #77" in handoff_comments[0]


def test_issue_loop_fresh_one_shot_publishes_decision_before_handoff(tmp_path):
    plan = structured_v1_plan_state()
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented the approved fresh plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        plan_execution_mode="implement-one-shot",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    decision_indexes = [
        index for index, comment in enumerate(runner.comments)
        if "AGENT_PLAN_EXECUTION_DECISION" in comment
    ]
    handoff_indexes = [
        index for index, comment in enumerate(runner.comments)
        if "AGENT_PLAN_ONE_SHOT_IMPL" in comment
    ]
    assert len(decision_indexes) == 1
    assert len(handoff_indexes) == 1
    assert decision_indexes[0] < handoff_indexes[0]


def test_issue_loop_fresh_one_shot_rerun_reuses_decision_and_handoff(tmp_path):
    """A real issue entry-point rerun must not rematerialize fresh state."""
    plan = structured_v1_plan_state()
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented the approved fresh plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        plan_execution_mode="implement-one-shot",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    durable_counts = {
        "decision": sum("AGENT_PLAN_EXECUTION_DECISION" in body for body in runner.comments),
        "handoff": sum("AGENT_PLAN_ONE_SHOT_IMPL" in body for body in runner.comments),
    }
    agent_command_count = sum(
        command[:1] in (["claude"], ["gemini"], ["agy"])
        or command[:2] == ["codex", "exec"]
        for command, _cwd in runner.commands
    )
    issue_count = len(runner.issues)

    # The second invocation consumes the issue-side handoff and resumes the
    # existing PR review.  It must not ask the coder for another implementation
    # or publish another decision/handoff.
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert {
        "decision": sum("AGENT_PLAN_EXECUTION_DECISION" in body for body in runner.comments),
        "handoff": sum("AGENT_PLAN_ONE_SHOT_IMPL" in body for body in runner.comments),
    } == durable_counts == {"decision": 1, "handoff": 1}
    assert sum(
        command[:1] in (["claude"], ["gemini"], ["agy"])
        or command[:2] == ["codex", "exec"]
        for command, _cwd in runner.commands
    ) == agent_command_count
    assert len(runner.issues) == issue_count == 0


def test_issue_loop_plan_first_resume_uses_handoff_bound_plan_when_later_plan_exists(tmp_path):
    old_plan = "Approved old plan.\n\n### Scope\n- Preserve the old API."
    later_plan = "Unrelated later plan.\n\n### Scope\n- Replace the old API."
    old_plan_comment = _attach_round_metadata(
        old_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="old-plan",
            canonical_plan=old_plan,
            raw_structured_coder_response=old_plan,
        ),
    )
    later_plan_comment = _attach_round_metadata(
        later_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject="later-plan",
            canonical_plan=later_plan,
            raw_structured_coder_response=later_plan,
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=approved_plan_hash(old_plan),
    )
    runner = _FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:00:00Z", "body": old_plan_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:01:00Z", "body": handoff},
            {"author": {"login": "bot"}, "createdAt": "2026-05-01T00:02:00Z", "body": later_plan_comment},
        ],
        pr_payload={
            "number": 77,
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Fixes #56",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, plan_execution_mode="implement-one-shot"),
        plan_first=True,
    ) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
        "Target child/primary issue context", 1
    )[0]
    assert "Preserve the old API." in plan_block
    assert "Replace the old API." not in plan_block


def test_issue_loop_plan_first_staged_child_recovers_parent_owned_plan(tmp_path, monkeypatch):
    parent_plan = "Approved parent plan.\n\n### Scope\n- Preserve the child API."
    unrelated_child_plan = "Unrelated child plan.\n\n### Scope\n- Change a different API."
    unrelated_child_plan_comment = _attach_round_metadata(
        unrelated_child_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="unrelated-child-plan",
            canonical_plan=unrelated_child_plan,
            raw_structured_coder_response=unrelated_child_plan,
        ),
    )
    parent_plan_comment = _attach_round_metadata(
        parent_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject="parent-plan",
            canonical_plan=parent_plan,
            raw_structured_coder_response=parent_plan,
        ),
    )
    child = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child phase issue for parent #55: staged implementation.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="coding-review-agent-loop",
                created_at="2026-05-01T00:00:00Z",
                body=unrelated_child_plan_comment,
            ),
            IssueComment(
                author="coding-review-agent-loop",
                created_at="2026-05-01T00:01:00Z",
                body=format_issue_pr_handoff_comment(
                    issue_number=56,
                    pr_number=77,
                    pr_url="https://github.com/OWNER/REPO/pull/77",
                    pr_head_sha="abc123",
                    flow="approved-plan-implementation",
                    plan_hash=approved_plan_hash(parent_plan),
                ),
            ),
        ),
    )
    parent = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Parent issue",
        body="Parent issue body.",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(
            IssueComment(
                author="coding-review-agent-loop",
                created_at="2026-05-01T00:00:00Z",
                body=parent_plan_comment,
            ),
        ),
    )

    def issue_context_for(_runner, *, config, issue_number):
        assert config.repo == "OWNER/REPO"
        return child if issue_number == 56 else parent

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", issue_context_for)
    runner = _FakeRunner(
        pr_payload={
            "number": 77,
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Fixes #56",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, plan_execution_mode="implement-by-phase"),
        plan_first=True,
    ) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    plan_block = prompt.split("Approved implementation plan context", 1)[1].split(
        "Target child/primary issue context", 1
    )[0]
    assert "Preserve the child API." in plan_block


def test_issue_loop_plan_first_one_shot_rerun_with_closed_pr_stops(tmp_path, capsys):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        pr_payload={"state": "CLOSED"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    output = capsys.readouterr().out
    assert "PR #77" in output
    assert "closed" in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_one_shot_rerun_hash_mismatch_stops_safely(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_plan = "Plan:\n- Old approach that was replaced.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(old_plan),
        plan_subject=_plan_subject(old_plan),
        pr_number=99,
        pr_head_sha=None,
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": old_handoff},
        ],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match=r"older plan hash") as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)

    assert "agent-loop pr 99" in str(excinfo.value)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_one_shot_rerun_pr_missing_issue_reference(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=approved_plan_hash(plan),
        plan_subject=_plan_subject(plan),
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": handoff},
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "No issue reference here.",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_one_shot_resumes_existing_pr_after_crash_before_handoff_comment(tmp_path):
    """Reproduces #492/#494: implementation created a PR but the run aborted before
    posting any handoff marker/comment (e.g. the #493 test-report false positive). A
    rerun must resume PR review on the existing PR instead of invoking the coder again,
    which is what previously produced the duplicate PR #494 (#495).
    """
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=approved "
            f"plan={approved_plan_hash(plan)}"
        ),
    )
    config = make_config(tmp_path)

    assert (
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)
        == 0
    )

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    handoff_comments = [c for c in runner.comments if "<!-- AGENT_ISSUE_PR_HANDOFF:" in c]
    assert len(handoff_comments) == 1
    assert "Flow: approved-plan-implementation" in handoff_comments[0]
    assert "PR #77" in handoff_comments[0]


def test_issue_loop_plan_first_one_shot_resume_existing_pr_logs_clear_message(tmp_path, capsys):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        open_prs_payload=[{"number": 492, "body": "Fixes #56"}],
        pr_payload={
            "number": 492,
            "url": "https://github.com/OWNER/REPO/pull/492",
            "body": "Fixes #56",
        },
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=approved "
            f"plan={approved_plan_hash(plan)}"
        ),
    )
    config = make_config(tmp_path, quiet=False)

    assert (
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)
        == 0
    )

    output = capsys.readouterr().err
    assert "Issue #56: resuming PR #492 review instead of invoking Claude." in output


def test_issue_loop_plan_first_one_shot_rerun_raises_on_ambiguous_existing_prs(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
        ],
        open_prs_payload=[
            {"number": 492, "body": "Fixes #56"},
            {"number": 494, "body": "Closes #56"},
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match=r"Multiple open PRs \(#492, #494\)"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_direct_mode_resumes_existing_canonical_pr(tmp_path, capsys):
    """#589: direct `agent-loop issue <n>` (no --plan-first) must also consult
    the canonical AGENT_ISSUE_PR_HANDOFF record before invoking a coder, so a
    rerun after an interrupted PR review resumes the existing PR instead of
    invoking the coder again and creating a duplicate."""
    canonical_comment = {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_issue_pr_handoff_comment(
            issue_number=56,
            pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123",
            flow="issue-implementation",
            plan_hash=None,
        ),
    }
    runner = FakeRunner(
        issue_comments=[canonical_comment],
        pr_payload={"body": "Fixes #56"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, quiet=False)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any("<!-- AGENT_ISSUE_PR_HANDOFF:" in c for c in runner.comments)
    output = capsys.readouterr().err
    assert "Issue #56: resuming PR #77 review instead of invoking Claude." in output


def test_issue_loop_direct_mode_legacy_search_resumes_and_backfills_canonical_record(tmp_path):
    """Legacy issues with no canonical record yet must still recover through
    the exactly-one-open-PR GitHub search, and the resume should backfill a
    canonical record so later reruns hit the fast canonical path (#589)."""
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_payload={"body": "Fixes #56"},
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    handoff_comments = [c for c in runner.comments if "<!-- AGENT_ISSUE_PR_HANDOFF:" in c]
    assert len(handoff_comments) == 1
    assert "Flow: issue-implementation" in handoff_comments[0]
    assert "PR #77" in handoff_comments[0]


def test_rejected_legacy_evidence_response_resumes_same_pr_without_reimplementation(tmp_path):
    """A legacy mappings-only response cannot hide a PR created by the coder."""
    legacy = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(legacy)
    payload["risk_test_matrix_evidence"] = {
        "matrix_identity": "not-authoritative",
        "mappings": {"row-1": ["receipt-from-model"]},
    }
    rejected = json.dumps(payload) + legacy[end:]
    runner = FakeRunner(claude_outputs=[rejected, rejected])
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentInvocationError):
        run_issue_loop(runner, issue_number=56, config=config)
    first_coder_calls = sum(
        command[:1] == ["claude"] for command, _cwd in runner.commands
    )

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_payload["body"] = "Fixes #56"
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    runner.codex_outputs = ["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"]

    assert run_issue_loop(runner, issue_number=56, config=config) == 0
    assert first_coder_calls == 1
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == first_coder_calls
    assert any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    assert not any("PR #" in comment and "duplicate" in comment.lower() for comment in runner.comments)


def test_issue_loop_direct_mode_incidental_open_pr_invokes_coder(tmp_path):
    """An incidental issue mention must not be treated as crash recovery."""
    runner = FakeRunner(
        open_prs_payload=[
            {
                "number": 492,
                "body": "Not on auto-merge; #56's loop is running and this PR is unrelated.",
            }
        ],
        claude_outputs=[
            "Implemented issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert any(cmd[:2] == ["claude", "--print"] for cmd, _cwd in runner.commands)
    handoff_comments = [c for c in runner.comments if "<!-- AGENT_ISSUE_PR_HANDOFF:" in c]
    assert len(handoff_comments) == 1
    assert "PR #77" in handoff_comments[0]
    assert "PR #492" not in handoff_comments[0]


def test_issue_loop_direct_mode_stale_canonical_record_raises(tmp_path):
    """A canonical record pointing at a closed/merged PR must fail safely
    with an actionable message before any coder invocation, never falling
    back to a fresh implementation (#589)."""
    canonical_comment = {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_issue_pr_handoff_comment(
            issue_number=56,
            pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123",
            flow="issue-implementation",
            plan_hash=None,
        ),
    }
    runner = FakeRunner(
        issue_comments=[canonical_comment],
        pr_payload={
            "number": 77,
            "state": "CLOSED",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Fixes #56",
        },
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match=r"agent-loop pr 77"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:1] in (["claude"], ["codex"]) for cmd, _cwd in runner.commands)


def test_issue_loop_direct_mode_ambiguous_open_prs_raises(tmp_path):
    """No canonical record and multiple open PRs referencing the issue must
    stop before any coder invocation with an actionable message (#589)."""
    runner = FakeRunner(
        open_prs_payload=[
            {"number": 77, "body": "Fixes #56"},
            {"number": 78, "body": "Closes #56"},
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match=r"Multiple open PRs \(#77, #78\)"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_issue_loop_direct_mode_canonical_record_resumes_refs_only_pr(tmp_path):
    """Canonical provenance remains valid when an older PR has only Refs text."""
    canonical_comment = {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_issue_pr_handoff_comment(
            issue_number=56,
            pr_number=77,
            pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123",
            flow="issue-implementation",
            plan_hash=None,
        ),
    }
    runner = FakeRunner(
        issue_comments=[canonical_comment],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "body": "Refs #56",
        },
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def _m1088_approval_body(issue_number, plan_hash):
    return (
        f"Planning complete for issue #{issue_number}.\n\n"
        "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: "
        f"issue={issue_number} plan={plan_hash} mode=summarize -->\n"
        "-- coding-review-agent-loop"
    )


def _m1088_runner(issue_comments, **kwargs):
    return FakeRunner(
        claude_outputs=["Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        issue_comments=issue_comments,
        authenticated_actor=_M1088_ACTOR,
        serve_rest_issue_comments=True,
        **kwargs,
    )


def _m1088_assert_refused_without_coder(runner, config):
    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)
    message = str(excinfo.value)
    assert "agent-loop issue 56 --plan-first" in message
    assert "plain issue mode" in message
    assert not [cmd for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert runner.comments == []
    return message


def _m1088_assert_ran_directly(runner, config):
    assert run_issue_loop(runner, issue_number=56, config=config) == 0
    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names


def test_1088_plain_issue_mode_refuses_top_level_issue_with_approved_plan(tmp_path):
    plan_hash = approved_plan_hash("Plan:\n- Make the change.")
    runner = _m1088_runner([_m1088_comment(_m1088_approval_body(56, plan_hash))])

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert f"an approved plan (hash {plan_hash})" in message


def test_1088_plain_issue_mode_refuses_top_level_issue_with_execution_decision(tmp_path):
    decision = _M1088ExecutionDecision(
        parent_issue=56,
        plan_hash="sha256:" + "a" * 64,
        plan_subject="Make the change",
        execution_strategy_contract_version=1,
        strategy="one-shot",
        topology_source="approved-plan",
        recommendation_digest="sha256:" + "b" * 64,
        requested_policy="auto",
        current_action="implement",
    )
    runner = _m1088_runner([_m1088_comment(_m1088_format_execution_decision(decision))])

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "a recorded execution decision" in message


def _m1090_plan_history(tmp_path, **reviews):
    """Authenticated history of a real plan-first run with Codex and Gemini.

    The run's approval announcement and execution decision are dropped, as
    when the process stops between round approval and those records.
    """
    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        authenticated_actor=_M1088_ACTOR,
        serve_rest_issue_comments=True,
        **reviews,
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=6,
        plan_execution_mode="plan-only", agent_max_retries=0,
    )
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        pass  # A split or abandoned round stops when scripted outputs run out.
    return [
        comment
        for comment in runner.issue_comments
        if "AGENT_PLAN_APPROVED_FOLLOWUPS" not in comment["body"]
        and "AGENT_PLAN_EXECUTION_DECISION" not in comment["body"]
    ]


def _m1090_gemini_review(state):
    if state == "approved":
        return structured_plan_review(state="approved", reviewer="Google Gemini")
    return structured_plan_review(
        state="blocking", reviewer="Google Gemini", summary="Missing owner.",
        blocking_plan_issues=["Name the rollout owner."],
    )


def _m1090_complete_history(tmp_path):
    return _m1090_plan_history(
        tmp_path,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[_m1090_gemini_review("approved")],
    )


def _m1090_assert_no_approval_claim(message):
    assert "approved plan" not in message
    assert "reviewed plan" not in message
    assert "resume from the approved plan" not in message


def test_1088_plain_issue_mode_refuses_approved_review_without_announcement(tmp_path):
    # Approval can be durable in the planning rounds before the announcement
    # and decision are posted (a preflight stop or an interrupted process).
    # #1090: the round as a whole must have approved the plan.
    from coding_review_agent_loop.plan_verification import classify_plan_approval

    history = _m1090_complete_history(tmp_path)
    classification = classify_plan_approval(
        [
            SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c["_rest_author_id"])
            for c in history
        ],
        config=make_config(tmp_path), issue_number=56, trusted_actor=_M1088_ACTOR,
        execution_mode_guard=False,
    )
    assert classification.state == "approved"
    runner = _m1088_runner(history)

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert f"a planning round that approved plan (hash {classification.plan_hash})" in message
    assert "resume from the approved plan" in message


def test_1090_split_plan_round_lets_plain_issue_mode_run(tmp_path):
    history = _m1090_plan_history(
        tmp_path,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[_m1090_gemini_review("blocking")],
    )
    runner = _m1088_runner(history)

    _m1088_assert_ran_directly(runner, make_config(tmp_path))
    # Plain mode posted no planning record over the split round.
    assert not [
        c for c in runner.issue_comments[len(history):] if "AGENT_PLAN_STATE" in c["body"]
    ]


def test_1090_abandoned_partial_approval_round_lets_plain_issue_mode_run(tmp_path):
    # Codex approved; the run stopped before Gemini ever reviewed.
    history = _m1090_plan_history(
        tmp_path, codex_outputs=[structured_plan_review(state="approved")],
    )
    runner = _m1088_runner(history)

    _m1088_assert_ran_directly(runner, make_config(tmp_path))


def test_1090_legacy_split_round_lets_plain_issue_mode_run(tmp_path):
    runner = _m1088_runner(
        _m1090_legacy_history(
            ("Codex", "OpenAI Codex", "approved"),
            ("Gemini", "Google Gemini", "blocking"),
        )
    )

    _m1088_assert_ran_directly(runner, make_config(tmp_path))


def test_1090_undetermined_planning_state_refuses_without_claiming_approval(tmp_path):
    # Legacy poster-only history where the sole poster approved: posters
    # prove membership, never completion, so the guard fails closed.
    runner = _m1088_runner(_m1090_legacy_history(("Codex", "OpenAI Codex", "approved")))

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "approval state cannot be determined" in message
    assert "round-level approval cannot be determined" in message
    _m1090_assert_no_approval_claim(message)


def test_1090_malformed_plan_lineage_refuses_as_unreadable(tmp_path):
    from coding_review_agent_loop.board_change import (
        PLAN_BOARD_CHANGE_LINK_HEADING,
        PLAN_BOARD_CHANGE_LINK_MARKER,
    )

    history = _m1090_complete_history(tmp_path)
    history.append(
        _m1088_comment(
            f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: e30= -->",
            created_at="2099-01-01T00:00:00Z",
        )
    )
    runner = _m1088_runner(history)

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "planning round records that cannot be read" in message
    _m1090_assert_no_approval_claim(message)


def test_1090_unreadable_handoff_only_refuses_without_claiming_approval(tmp_path):
    runner = _m1088_runner(
        [_m1088_comment("<!-- AGENT_ISSUE_PR_HANDOFF: bm90LWpzb24 -->")]
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "an implementation handoff record that cannot be read" in message
    assert "approval state cannot be determined" in message
    _m1090_assert_no_approval_claim(message)


def test_1090_unreadable_execution_decision_only_refuses_without_claiming_approval(tmp_path):
    runner = _m1088_runner(
        [_m1088_comment("<!-- AGENT_PLAN_EXECUTION_DECISION: bm90LWpzb24 -->")]
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "an execution decision record that cannot be read" in message
    assert "approval state cannot be determined" in message
    _m1090_assert_no_approval_claim(message)


def test_1090_proven_approval_keeps_approval_wording_beside_unreadable_records(tmp_path):
    plan_hash = approved_plan_hash("Plan:\n- Make the change.")
    runner = _m1088_runner(
        [
            _m1088_comment(_m1088_approval_body(56, plan_hash), created_at="2026-05-23T00:00:01Z"),
            _m1088_comment("<!-- AGENT_ISSUE_PR_HANDOFF: bm90LWpzb24 -->"),
        ]
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert f"an approved plan (hash {plan_hash})" in message
    assert "an implementation handoff record that cannot be read" in message
    assert "resume from the approved plan" in message


def test_1088_plain_issue_mode_sees_approval_beyond_projection_cap(tmp_path):
    plan_hash = approved_plan_hash("Plan:\n- Make the change.")
    filler = [
        _m1088_comment(
            f"Ordinary discussion {index}.",
            created_at=f"2026-05-24T00:{index // 60:02d}:{index % 60:02d}Z",
            author=("someone", 900),
        )
        for index in range(100)
    ]
    runner = _m1088_runner(
        [
            _m1088_comment(
                _m1088_approval_body(56, plan_hash),
                created_at="2026-05-23T00:00:02Z",
                _rest_only=True,
            ),
            *filler,
        ]
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert f"an approved plan (hash {plan_hash})" in message


def test_1088_foreign_planning_records_do_not_block_direct_mode(tmp_path):
    forged = ("mallory", 777)
    runner = _m1088_runner(
        [
            _m1088_comment(
                _m1088_approval_body(56, approved_plan_hash("Plan:\n- Forged.")), author=forged
            ),
            _m1088_comment(
                "<!-- AGENT_PLAN_EXECUTION_DECISION: bm90LWpzb24 -->",
                created_at="2026-05-23T00:00:03Z",
                author=forged,
            ),
        ]
    )

    _m1088_assert_ran_directly(runner, make_config(tmp_path))


def test_1088_plain_issue_mode_still_runs_directly_without_plan_for_this_issue(tmp_path):
    # An approval recorded for a different issue does not belong to #56.
    runner = _m1088_runner(
        [_m1088_comment(_m1088_approval_body(99, approved_plan_hash("Plan:\n- Other.")))]
    )

    _m1088_assert_ran_directly(runner, make_config(tmp_path))


def test_1088_plain_issue_mode_refuses_handoff_only_planning_record(tmp_path):
    # The approved-plan handoff is the only planning record in view.
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=approved_plan_hash("Plan:\n- Make the change."),
    )
    runner = _m1088_runner([_m1088_comment(handoff)], pr_payload={"body": "Fixes #56"})

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "an approved-plan implementation handoff to PR #77" in message


def test_1088_later_direct_handoff_does_not_hide_approved_plan_handoff(tmp_path):
    approved = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=approved_plan_hash("Plan:\n- Make the change."),
    )
    direct = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=78,
        pr_url="https://github.com/OWNER/REPO/pull/78",
        pr_head_sha="def456",
        flow="issue-implementation",
        plan_hash=None,
    )
    runner = _m1088_runner(
        [
            _m1088_comment(approved, created_at="2026-05-23T00:00:01Z"),
            _m1088_comment(direct, created_at="2026-05-23T00:00:02Z"),
        ],
        pr_payload={"body": "Fixes #56"},
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "an approved-plan implementation handoff to PR #77" in message


class _M1088ApprovalAfterFirstSnapshotRunner(FakeRunner):
    """Posts a plan approval right after the first issue snapshot is read."""

    def __init__(self, approval, **kwargs):
        super().__init__(**kwargs)
        self._approval = approval
        self.issue_view_calls = 0

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        result = super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)
        if list(args[:3]) == ["gh", "issue", "view"]:
            self.issue_view_calls += 1
            if self.issue_view_calls == 1:
                if isinstance(self._approval, list):
                    self.issue_comments.extend(self._approval)
                else:
                    self.issue_comments.append(self._approval)
        return result


def test_1088_plain_issue_mode_rechecks_fresh_snapshot_before_dispatch(tmp_path):
    plan_hash = approved_plan_hash("Plan:\n- Make the change.")
    runner = _M1088ApprovalAfterFirstSnapshotRunner(
        _m1088_comment(_m1088_approval_body(56, plan_hash)),
        claude_outputs=["Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        authenticated_actor=_M1088_ACTOR,
        serve_rest_issue_comments=True,
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert f"an approved plan (hash {plan_hash})" in message
    # The first snapshot was clean; only the fresh pre-dispatch one refused.
    assert runner.issue_view_calls >= 2


def test_1090_fresh_snapshot_recheck_refuses_round_approval_completed_after_first_snapshot(
    tmp_path,
):
    # A concurrent plan-first run completes round approval (no announcement,
    # decision or handoff) after the first snapshot; the fresh guard refuses.
    history = _m1090_complete_history(tmp_path)
    runner = _M1088ApprovalAfterFirstSnapshotRunner(
        history,
        claude_outputs=["Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        authenticated_actor=_M1088_ACTOR,
        serve_rest_issue_comments=True,
    )

    message = _m1088_assert_refused_without_coder(runner, make_config(tmp_path))

    assert "a planning round that approved plan (hash " in message
    assert "resume from the approved plan" in message
    assert runner.issue_view_calls >= 2


def test_1090_foreign_complete_planning_round_does_not_block_direct_mode(tmp_path):
    # A complete, approving planning round authored by a foreign actor is
    # filtered out before classification.
    forged = ("mallory", 777)
    history = [
        {**comment, "author": {"login": forged[0]}, "_rest_author_id": forged[1]}
        for comment in _m1090_complete_history(tmp_path)
    ]
    runner = _m1088_runner(history)

    _m1088_assert_ran_directly(runner, make_config(tmp_path))


def test_1088_plain_issue_mode_without_planning_records_needs_no_rest_history(tmp_path):
    # No REST history or actor identity is served: an issue with no planning
    # record in its (uncapped) projection must not need either.
    runner = FakeRunner(
        claude_outputs=["Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    _m1088_assert_ran_directly(runner, make_config(tmp_path))
