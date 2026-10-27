"""PR-loop tests for review_rounds and review_scheduling: PR review scheduling (#1542)."""

import dataclasses
import json
import re
from types import SimpleNamespace

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestMetadata,
    PullRequestReviewContext,
)
from coding_review_agent_loop.issue_pr_handoff import (
    find_latest_issue_pr_handoff,
    format_issue_pr_handoff_comment,
)
from coding_review_agent_loop.migrations import MigrationValidationResult
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.review_scheduling import TransitionClassification
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.history_export as history_export
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _EVIDENCE_TEXT,
    _agent_sequence,
    _audit_phases,
    _evidence_review,
    _freeze_records,
    _m943_amendment_from_error,
    _m943_append,
    _posted_scheduler_metadata,
    _release_records,
    _resolve,
    _rewrite_pr_metadata,
    _signed,
    _staged_config,
    _staged_review,
    _unavailable,
)


def test_selective_pr_loop_only_rechecks_owner_then_final_missing_reviewers(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup is incomplete", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            review(reviewer="Google Gemini"),
            review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=4,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini", "agy"}
    ]
    # Initial full board, one owner-only remediation turn, then the exact-head
    # sweep for the two paused reviewers.
    assert [command[0] for command in reviewer_commands].count("codex") == 2
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert [command[0] for command in reviewer_commands].count("agy") == 2


def test_selective_pr_loop_keeps_multiple_consecutive_narrow_fixes_selective(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-1", "item-2"]),
        ],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "first cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "second cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
                dispositions=[
                    {"item_id": "item-1", "disposition": "blocking", "note": "still open"}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
        ],
        gemini_outputs=[
            review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
            review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=5,
        pr_review_context_mode="compact",
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini", "agy"}
    ]
    # Both approving reviewers remain paused across both narrow transitions;
    # they return once for the unchanged-head exact approval sweep.
    assert [command[0] for command in reviewer_commands].count("codex") == 3
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert [command[0] for command in reviewer_commands].count("agy") == 2


def test_selective_resume_with_malformed_scheduler_metadata_recovers_after_full_board(tmp_path):
    malformed_checkpoint = _attach_round_metadata(
        "stale scheduler checkpoint",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            scheduler_contract={"policy": "selective-intermediate"},
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "headRefOid": "abc123",
            "comments": [{"author": {"login": "bot"}, "body": malformed_checkpoint}],
        },
        codex_outputs=[structured_pr_review(summary="Codex reviewed the full board.")],
        gemini_outputs=[structured_pr_review(summary="Gemini reviewed the full board.", reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    reviewer_commands = [
        command for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini"}
    ]
    # The invalid checkpoint cannot make the scheduler select a subset.
    assert [command[0] for command in reviewer_commands] == ["codex", "gemini"]
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_selective_final_sweep_missing_approval_stops_before_migration_or_merge(
    tmp_path, monkeypatch
):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        pr_loop,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    managed_ci_calls = []
    qualification_calls = []
    optional_test_calls = []
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *args, **kwargs: managed_ci_calls.append(True),
    )
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: qualification_calls.append(True),
    )
    monkeypatch.setattr(
        pr_loop,
        "run_optional_tests",
        lambda *args, **kwargs: optional_test_calls.append(True),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
        antigravity_outputs=[review(reviewer="Antigravity")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert managed_ci_calls == []
    assert qualification_calls == []
    assert optional_test_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1


def test_selective_final_sweep_blocker_dispatches_coder_remediation(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
        ],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "first cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[
            review(reviewer="Google Gemini"),
            review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[
                    {"text": "final-sweep regression", "fix_scope": ["src/worker.py"]}
                ],
            ),
            review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
            review(reviewer="Antigravity"),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=4,
    )

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval"):
        run_pr_loop(runner, pr_number=77, config=config)

    coder_commands = [
        command for command, _cwd in runner.commands if command and command[0] == "claude"
    ]
    assert len(coder_commands) == 2
    assert "final-sweep regression" in coder_commands[1][-1]


def _run_selective_owner_unavailable(tmp_path, monkeypatch):
    def review(*, reviewer, state="approved", blocking_items=None, dispositions=None):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": state,
                    "summary": f"{reviewer} review",
                    "blocking_items": blocking_items or [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": dispositions or [],
                }
            )
            + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
        )

    unavailable = json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The reviewer cannot access the repository history.",
            "suggested_action": "Repair the reviewer environment before retrying.",
        }
    ) + "\n<!-- AGENT_UNAVAILABLE -->\n-- OpenAI Codex"
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[
                    {"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}
                ],
            ),
            unavailable,
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=3,
    )

    with pytest.raises(AgentLoopError, match="missing required input from Codex") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    return runner, str(excinfo.value)


def test_selective_owner_unavailability_stops_without_coder_redispatch(tmp_path, monkeypatch):
    runner, message = _run_selective_owner_unavailable(tmp_path, monkeypatch)

    assert "Detected in round 2" in message
    # Only the round-1 coder turn ran; the round-2 detection stops before a second.
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1
    # A validated amendment template goes to the error only, never the PR comment.
    assert "Signed amendment template" in message
    assert '"kind": "reviewer-board-amendment"' in message
    incomplete = [body for body in runner.comments if "**Review status: Incomplete**" in body]
    assert len(incomplete) == 1
    assert "effective from round 2" in incomplete[0]
    assert "reviewer-board-amendment" not in incomplete[0]
    assert "```" not in incomplete[0]


def test_unavailable_reviewer_amendment_rejected_prints_prose_only(tmp_path, monkeypatch):
    def reject(*args, **kwargs):
        raise AgentLoopError("cannot reduce")

    monkeypatch.setattr(review_rounds, "amend_contract", reject)
    runner, message = _run_selective_owner_unavailable(tmp_path, monkeypatch)

    assert "cannot be reduced for this run" in message
    assert "Signed amendment template" not in message
    incomplete = [body for body in runner.comments if "**Review status: Incomplete**" in body]
    assert len(incomplete) == 1
    assert "cannot be reduced for this run" in incomplete[0]


def test_unavailable_reviewer_advisory_lookup_failure_keeps_the_stop(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("lookup exploded")

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_pr_amendment_start_round", broken)
    runner, message = _run_selective_owner_unavailable(tmp_path, monkeypatch)

    assert "lookup exploded" not in message
    assert "Signed amendment template" not in message
    assert "the round the rerun resumes into" in message
    incomplete = [body for body in runner.comments if "**Review status: Incomplete**" in body]
    assert len(incomplete) == 1


def test_selective_resume_uses_a_valid_persisted_scheduler_checkpoint(tmp_path):
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="The worker cleanup still needs verification.",
        status="blocking",
        source_status="blocking",
        fix_scope=("src/worker.py",),
        resolution_owners=("Codex",),
        owner_states=(("Codex", "pending"),),
    )
    contract = pr_loop.make_contract(
        ("Codex", "Gemini", "Antigravity"),
        "selective-intermediate",
        None,
    )
    checkpoint = _attach_round_metadata(
        "Persisted selective reconciliation checkpoint.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="abc123",
            prior_items=(item,),
            phase="reconciliation",
            scheduler_contract=contract.as_dict(),
            scheduler_previous_sha="abc123",
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(
                ("Gemini", "approval is historical"),
                ("Antigravity", "approval is historical"),
            ),
            scheduler_reasons=("pending resolution owner",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=3,
        ),
    )

    def review(*, reviewer, dispositions=()):
        return (
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "pr_review",
                    "state": "approved",
                    "summary": f"{reviewer} review",
                    "blocking_items": [],
                    "same_pr_followups": [],
                    "future_followups": [],
                    "prior_item_dispositions": list(dispositions),
                }
            )
            + f"\n<!-- AGENT_STATE: approved -->\n-- {reviewer}"
        )

    runner = FakeRunner(
        pr_payload={
            "headRefOid": "abc123",
            "comments": [{"author": {"login": "bot"}, "body": checkpoint}],
        },
        codex_outputs=[
            review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        gemini_outputs=[review(reviewer="Google Gemini")],
        antigravity_outputs=[review(reviewer="Antigravity")],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="selective-intermediate",
        max_rounds=3,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    agent_commands = [
        command[0]
        for command, _cwd in runner.commands
        if command and command[0] in {"claude", "codex", "gemini", "agy"}
    ]
    assert agent_commands == ["codex", "gemini", "agy"]
    assert "Persisted selective reconciliation checkpoint." not in "\n".join(
        command[-1] for command, _cwd in runner.commands if command and command[0] == "claude"
    )


def test_selective_same_head_requirement_edit_invalidates_approval_before_merge(
    tmp_path, monkeypatch
):
    requirement_1 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="The edited instruction changes the required audit trail.",
    )
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Selective requirement edit",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    context_calls = 0

    def changing_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        return PullRequestReviewContext(
            metadata=metadata,
            comments=(),
            human_requirements=(requirement_1 if context_calls == 1 else requirement_2,),
        )

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", changing_context)
    runner = FakeRunner(
        claude_outputs=[],
        codex_outputs=[
            structured_pr_review(
                summary="Codex approves the original requirement.",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                summary="Codex approves the edited requirement.",
                reviewer="OpenAI Codex",
                human_requirements_resolved=True,
            ),
        ],
        gemini_outputs=[
            structured_pr_review(
                summary="Gemini approves the original requirement.",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                summary="Gemini approves the edited requirement.",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    reviewer_commands = [
        command
        for command, _cwd in runner.commands
        if command and command[0] in {"codex", "gemini"}
    ]
    assert [command[0] for command in reviewer_commands].count("codex") == 2
    assert [command[0] for command in reviewer_commands].count("gemini") == 2
    assert any(requirement_2.body in command[-1] for command in reviewer_commands)
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_selective_plan_handoff_change_stops_before_migration_or_merge(tmp_path, monkeypatch):
    plan = "Approved plan.\n\n## Scope\n- Preserve the current API."
    plan_context = execution_policy.make_approved_plan_context(plan, source_locator="test")
    handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=plan_context.plan_hash,
    )
    plan_record = _attach_round_metadata(
        plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=plan_context.plan_subject or "plan",
            canonical_plan=plan,
            raw_structured_coder_response=plan,
        ),
    )
    initial_issue = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Original issue.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(author="bot", created_at="2026-05-01T00:00:00Z", body=plan_record),
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=handoff),
        ),
        human_requirements=(),
    )
    changed_issue = dataclasses.replace(initial_issue, comments=(initial_issue.comments[0],))
    issue_fetches = 0

    def changing_issue(*args, **kwargs):
        nonlocal issue_fetches
        issue_fetches += 1
        return initial_issue if issue_fetches == 1 else changed_issue

    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context", changing_issue)
    migration_calls = []
    monkeypatch.setattr(
        pr_loop,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                summary="The approved plan is implemented.",
                reviewer="OpenAI Codex",
            )
        ]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="Approved-plan/handoff identity changed"):
        run_pr_loop(
            runner,
            pr_number=77,
            config=config,
            issue_context=initial_issue,
            approved_plan_context=plan_context,
        )
    assert migration_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_selective_valid_plan_handoff_replacement_sweeps_before_migration(
    tmp_path, monkeypatch
):
    old_plan = "Approved plan.\n\n## Scope\n- Preserve the current API."
    new_plan = "Approved replacement plan.\n\n## Scope\n- Preserve the current API and audit trail."
    old_plan_context = execution_policy.make_approved_plan_context(old_plan, source_locator="test")
    new_plan_context = execution_policy.make_approved_plan_context(new_plan, source_locator="test")
    old_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=old_plan_context.plan_hash,
    )
    old_handoff_metadata = find_latest_issue_pr_handoff(
        [IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff)],
        issue_number=56,
        repo="OWNER/REPO",
    )
    assert old_handoff_metadata is not None
    new_handoff = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="approved-plan-implementation",
        plan_hash=new_plan_context.plan_hash,
        supersedes_hash=old_handoff_metadata.contract_hash,
    )

    def plan_record(plan_text, timestamp):
        plan_context = execution_policy.make_approved_plan_context(plan_text, source_locator="test")
        return IssueComment(
            author="bot",
            created_at=timestamp,
            body=_attach_round_metadata(
                plan_text,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Claude",
                    round_number=1,
                    subject=plan_context.plan_subject or "plan",
                    canonical_plan=plan_text,
                    raw_structured_coder_response=plan_text,
                ),
            ),
        )

    old_plan_record = plan_record(old_plan, "2026-05-01T00:00:00Z")
    new_plan_record = plan_record(new_plan, "2026-05-01T00:02:00Z")
    initial_issue = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Original issue.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            old_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff),
        ),
        human_requirements=(),
    )
    replacement_issue = dataclasses.replace(
        initial_issue,
        comments=(
            old_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:01:00Z", body=old_handoff),
            new_plan_record,
            IssueComment(author="bot", created_at="2026-05-01T00:02:00Z", body=new_handoff),
        ),
    )
    issue_fetches = 0

    def changing_issue(*args, **kwargs):
        nonlocal issue_fetches
        issue_fetches += 1
        return initial_issue if issue_fetches == 1 else replacement_issue

    patch_attr_in(monkeypatch, (
        discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
        plan_first_loop, issue_loop,
    ), "get_issue_context", changing_issue)
    migration_calls = []

    def validate_migration(*args, **kwargs):
        migration_calls.append(True)
        reviewer_commands = [
            command
            for command, _cwd in runner.commands
            if command and command[0] == "codex"
        ]
        assert len(reviewer_commands) == 2
        return MigrationValidationResult(ok=True)

    monkeypatch.setattr(pr_loop, "validate_pr_migration_topology", validate_migration)
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(summary="Approval bound to the original plan."),
            structured_pr_review(summary="Approval bound to the replacement plan."),
        ]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        max_rounds=2,
    )

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        issue_context=initial_issue,
        approved_plan_context=old_plan_context,
    ) == 0

    assert migration_calls == [True]
    reviewer_commands = [
        command
        for command, _cwd in runner.commands
        if command and command[0] == "codex"
    ]
    assert any(new_plan in " ".join(command) for command in reviewer_commands)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_selective_scheduler_contract_change_during_managed_ci_blocks_merge(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(summary="Initial exact-head approval.")]
    )
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        auto_merge=True,
        max_rounds=1,
    )
    monkeypatch.setattr(
        pr_loop,
        "activate_managed_ci",
        lambda *args, **kwargs: ManagedCiContract(),
    )
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *args, **kwargs: None,
    )
    changed_contract = pr_loop.make_contract(
        ("Codex",), "all-reviewers", None
    )
    changed_audit = _attach_round_metadata(
        "A scheduler contract was changed while managed CI was running.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=2,
            subject="abc123",
            phase="reconciliation",
            scheduler_contract=changed_contract.as_dict(),
            scheduler_previous_sha=None,
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(),
            scheduler_reasons=("contract changed during managed CI",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
        ),
    )

    def wait_for_managed_ci(*args, **kwargs):
        runner.pr_payload.setdefault("comments", []).append(
            {"author": {"login": "bot"}, "body": changed_audit}
        )
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        wait_for_managed_ci,
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("scheduler contract changes must prevent merge"),
    )

    with pytest.raises(AgentLoopError, match="scheduler contract changed during qualification"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_primary_blocking_loop_gates_panel_then_independent_audit(tmp_path, monkeypatch):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="OpenAI Codex",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # Primary blocks (head abc123), coder fixes, primary re-checks alone
    # (still the primary phase), then both secondaries audit the new head.
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    phases = _audit_phases(runner)
    assert [phase for phase, _head in phases] == ["primary", "primary", "secondary-audit"]
    assert phases[0][1] == "abc123"
    assert phases[1][1] == phases[2][1] == "abc123-coder-1"
    assert not any("final-secondary-sweep" in comment for comment in runner.comments)
    assert not any("remediation" in phase for phase, _head in phases)
    # The secondary prompts carry the independent-audit instruction and the
    # phase audit record rather than a primary-finding validation task.
    for name in ("gemini", "agy"):
        prompt = next(command[-1] for command, _cwd in runner.commands if command[:1] == [name])
        assert "Selected phase: secondary-audit" in prompt
        assert "Inspect the complete base-to-head diff independently" in prompt
        assert "do not merely validate, repeat, or\ntriage findings attributed to the primary" in prompt
    metadata = _posted_scheduler_metadata(runner)
    assert {item.scheduler_primary_reviewer for item in metadata} == {"Codex"}
    assert all(item.scheduler_force_full is False for item in metadata)
    assert metadata[-1].scheduler_calls_avoided >= 2


def test_staged_secondary_scoped_remediation_rechecks_owner_and_primary_then_sweeps_others(
    tmp_path, monkeypatch
):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        pr_loop,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # primary -> panel (gemini, agy) -> coder -> remediation (codex + gemini)
    # -> final exact-head sweep (agy) -> migration gate.
    assert sequence[:3] == ["codex", "gemini", "agy"]
    assert sequence[3] == "claude"
    assert sorted(sequence[4:6]) == ["codex", "gemini"]
    assert sequence[6:] == ["agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "secondary-audit", "remediation", "final-secondary-sweep",
    ]
    remediation_audit = next(c for c in runner.comments if "phase: remediation" in c)
    assert "active owners: Gemini" in remediation_audit
    assert "paused Antigravity" in remediation_audit
    sweep_audit = next(c for c in runner.comments if "phase: final-secondary-sweep" in c)
    assert "selected Antigravity" in sweep_audit
    # Antigravity's old-head approval never counted as final; the migration
    # gate ran only after the sweep completed.
    assert len(migration_calls) == 1
    sweep_index = runner.comments.index(sweep_audit)
    assert not any(
        "Alembic" in comment for comment in runner.comments[:sweep_index]
    )


def test_staged_owner_and_primary_clearance_does_not_transition_directly_to_ci(tmp_path, monkeypatch):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    migration_calls = []
    monkeypatch.setattr(
        pr_loop,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("merged without the sweep")
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Rounds: 1 primary, 2 panel, 3 remediation; the sweep would be round 4.
    config = _staged_config(tmp_path, max_rounds=3, auto_merge=True)

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)
    assert _agent_sequence(runner).count("agy") == 1


def test_staged_broad_remediation_reactivates_complete_board(tmp_path, monkeypatch):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("broad", "diff path 'src/other.py' is outside obligation scopes"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    assert sequence[:4] == ["codex", "gemini", "agy", "claude"]
    # Every reviewer, including the non-owner Antigravity, reviews the broad head.
    assert sorted(sequence[4:]) == ["agy", "codex", "gemini"]
    assert [phase for phase, _head in _audit_phases(runner)] == [
        "primary", "secondary-audit", "full-board",
    ]
    full_board_audit = next(c for c in runner.comments if "phase: full-board" in c)
    assert "paused none" in full_board_audit
    assert "outside obligation scopes" in full_board_audit


def test_staged_ambiguous_ownership_reactivates_complete_board(tmp_path, monkeypatch):
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    observed_scopes = []

    def classify_from_ledger(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        observed_scopes.append(tuple(scopes))
        return classify_transition(
            previous_sha, current_sha, [GitChange("src/worker.py")],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_observe_pr_transition", classify_from_ledger)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            # No fix_scope: ownership scope is absent, so remediation is unsafe.
            _staged_review(reviewer="Google Gemini", state="blocking", blocking_items=["panel regression"]),
            _staged_review(
                reviewer="Google Gemini",
                dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    assert sorted(sequence[4:]) == ["agy", "codex", "gemini"]
    assert [phase for phase, _head in _audit_phases(runner)][-1] == "full-board"
    # The finding carried no reviewer scope, so the classifier had nothing to
    # match the change against and the orchestrator did not guess one.
    assert observed_scopes == [()]
    assert any("an obligation has no valid exact fix scope" in comment for comment in runner.comments)


def test_staged_force_full_selects_complete_board_from_the_primary_phase(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=True, max_rounds=1)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert _audit_phases(runner) == [("full-board", "abc123")]
    assert "force-full: True" in runner.comments[0]
    assert all(item.scheduler_force_full is True for item in _posted_scheduler_metadata(runner))


def test_staged_legacy_scheduler_payload_before_panel_is_strict_primary_only(tmp_path, monkeypatch, capsys):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    contract = pr_loop.make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex")
    # A pre-phase-aware record: complete legacy mandatory keys, no phase authority.
    legacy_checkpoint = _attach_round_metadata(
        "Legacy scheduler checkpoint.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="abc123",
            scheduler_contract=contract.as_dict(),
            scheduler_previous_sha=None,
            scheduler_current_sha="abc123",
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"), ("Antigravity", "primary phase")),
            scheduler_reasons=("primary phase",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=2,
        ),
    )
    decoded = architecture_contract._extract_round_metadata_records(
        [SimpleNamespace(body=legacy_checkpoint)], flow="pr"
    )[0].metadata
    assert decoded.scheduler_metadata_status == "valid"
    assert decoded.scheduler_phase is None
    runner = FakeRunner(
        pr_payload={"comments": [{"author": {"login": "bot"}, "body": legacy_checkpoint}]},
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "gap", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(reviewer="Google Gemini"),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = _staged_config(tmp_path, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # #840 (row prepanel-resume-incomplete-metadata): phase-less legacy data
    # before any panel evidence re-invokes only the primary with full context
    # and never latches; the panel starts only after exact-head approval.
    assert sequence == ["codex", "claude", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == ["primary", "primary", "secondary-audit"]
    audits = [c for c in runner.comments if c.startswith("PR review scheduling audit:")]
    assert "strict pre-panel fallback: scheduler metadata recovery: scheduler metadata lacks phase authority" in audits[0]
    assert "primary re-invoked with full context" in audits[1]
    assert "post-panel fallback" not in "".join(runner.comments)
    posted = _posted_scheduler_metadata(runner)
    assert posted[0].scheduler_force_full is False  # the seeded legacy record
    assert all(item.scheduler_force_full is False for item in posted[1:])
    assert all(item.scheduler_force_full_source is None for item in posted)
    assert "durable force-full latch" not in capsys.readouterr().err


def test_staged_legacy_selective_payload_is_contract_drift(tmp_path):
    contract = pr_loop.make_contract(("Codex", "Gemini"), "selective-intermediate", None)
    checkpoint = _attach_round_metadata(
        "Selective checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex", "Gemini"), scheduler_reasons=("full board",),
            scheduler_final_sweep=False, scheduler_force_full=False, scheduler_calls_avoided=0,
        ),
    )
    runner = FakeRunner(pr_payload={"comments": [{"author": {"login": "bot"}, "body": checkpoint}]})
    config = _staged_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == []


def test_staged_primary_change_is_a_recorded_board_change(tmp_path):
    """A primary change is a board-change link that restarts the primary phase (#1376).

    It was refused as contract drift before; now one link is posted and the
    next round runs primary-only on the new primary for the current head.
    """
    contract = pr_loop.make_contract(("Codex", "Gemini"), "primary-then-panel", None, "Gemini")
    checkpoint = _attach_round_metadata(
        "Staged checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Gemini",), scheduler_paused_reviewers=(("Codex", "primary phase"),),
            scheduler_reasons=("primary phase",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=1, scheduler_phase="primary",
            scheduler_primary_reviewer="Gemini",
        ),
    )
    runner = FakeRunner(
        pr_payload={"comments": [{"author": {"login": "bot"}, "body": checkpoint}]},
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
    )
    config = _staged_config(tmp_path, reviewer=("codex", "gemini"))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    links = [c for c in runner.comments if c.startswith("Reviewer board change recorded.")]
    assert len(links) == 1
    assert "- Primary reviewer: Gemini -> Codex" in links[0]
    assert "- Effective from round: 2" in links[0]
    # The restarted primary phase runs the new primary alone first.
    assert _agent_sequence(runner)[0] == "codex"
    assert [phase for phase, _head in _audit_phases(runner)][0] == "primary"


def test_staged_primary_failure_is_not_approval_and_resume_reuses_no_work(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_unavailable("OpenAI Codex"), _staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required input from Codex"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == ["codex"]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)

    # Resume from the durable comments: the primary is dispatched again, and
    # only after its approval does the panel run.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner) == ["codex", "codex", "gemini", "agy"]
    assert [phase for phase, _head in _audit_phases(runner)] == ["primary", "primary", "secondary-audit"]


def test_staged_secondary_failure_keeps_healthy_result_and_resume_dispatches_only_outstanding(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity"), _staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, auto_merge=True)
    merges = []

    with pytest.raises(AgentLoopError, match="missing required input from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert any("Healthy reviewers approved" in comment or "Google Gemini review" in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


    original_merge = pr_loop_support.merge_pr
    pr_loop_support.merge_pr = lambda *args, **kwargs: merges.append(kwargs)
    try:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
    finally:
        pr_loop_support.merge_pr = original_merge
    # Completed exact-head work (Codex, Gemini) is reused once; only the
    # outstanding secondary is dispatched.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy", "agy"]
    assert merges == [{"expected_head_sha": "abc123"}]


def test_staged_resume_after_remediation_checkpoint_dispatches_final_sweep_only(tmp_path, monkeypatch):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Stop right after the remediation round settles (round 3 of 3).
    first_config = _staged_config(tmp_path, max_rounds=3)
    with pytest.raises(AgentLoopError, match="missing reviewer approval from Antigravity"):
        run_pr_loop(runner, pr_number=77, config=first_config)
    assert _agent_sequence(runner).count("agy") == 1

    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
    resumed_config = _staged_config(tmp_path, max_rounds=6)
    assert run_pr_loop(runner, pr_number=77, config=resumed_config) == 0
    sequence = _agent_sequence(runner)
    # Codex and Gemini exact-head approvals are reused; only Antigravity sweeps.
    assert sequence.count("codex") == 2
    assert sequence.count("gemini") == 2
    assert sequence.count("agy") == 2
    assert sequence[-1] == "agy"
    assert [phase for phase, _head in _audit_phases(runner)][-1] == "final-secondary-sweep"


def test_staged_external_head_change_invalidates_primary_approval_before_panel(tmp_path, monkeypatch):
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("broad", "external change"),
    )
    runner = FakeRunner(
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    original = validated_agent._run_validated_agent
    calls = []

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "reviewer" and kwargs.get("agent") == "codex":
            calls.append(True)
            if len(calls) == 1:
                runner.pr_payload["headRefOid"] = "external-head"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    sequence = _agent_sequence(runner)
    # The stale primary approval on abc123 never opens the panel: the primary
    # must approve the external head first, then the secondaries audit it.
    assert sequence == ["codex", "codex", "gemini", "agy"]
    phases = _audit_phases(runner)
    assert [phase for phase, _head in phases][-1] == "secondary-audit"
    assert phases[-1][1] == "external-head"
    assert all(head == "external-head" for _phase, head in phases[1:])


def test_staged_head_change_during_final_sweep_invalidates_partial_sweep_evidence(tmp_path, monkeypatch):
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    def classify_from_ledger(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        # The scoped remediation touches only the owned path; the external
        # push after the sweep has no obligation scope to match against.
        return classify_transition(
            previous_sha, current_sha, [GitChange("src/worker.py")],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_observe_pr_transition", classify_from_ledger)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
            _staged_review(reviewer="OpenAI Codex"),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
            _staged_review(reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity"),
        ],
    )
    original = validated_agent._run_validated_agent
    agy_calls = []

    def run(*args, **kwargs):
        response = original(*args, **kwargs)
        if kwargs.get("role") == "reviewer" and kwargs.get("agent") == "antigravity":
            agy_calls.append(True)
            if len(agy_calls) == 2:
                # An external actor pushes while the final sweep is settling.
                runner.pr_payload["headRefOid"] = "pushed-during-sweep"
        return response

    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", run)
    config = _staged_config(tmp_path, max_rounds=8)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = _audit_phases(runner)
    # The sweep on the old head settles, but the pushed head has no obligation
    # scope to classify against, so every reviewer (including the primary and
    # the secondaries that had just approved) re-reviews the complete diff.
    # No partial sweep approval carries across the mutation.
    assert phases == [
        ("primary", "abc123"),
        ("secondary-audit", "abc123"),
        ("remediation", "abc123-coder-1"),
        ("final-secondary-sweep", "abc123-coder-1"),
        ("full-board", "pushed-during-sweep"),
    ]
    sequence = _agent_sequence(runner)
    assert sequence == [
        "codex", "gemini", "agy", "claude", "codex", "gemini", "agy", "codex", "gemini", "agy",
    ]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands[: -1])


def test_staged_exact_head_panel_gate_stops_before_migration_and_merge(tmp_path, monkeypatch):
    migration_calls = []
    monkeypatch.setattr(
        pr_loop,
        "validate_pr_migration_topology",
        lambda *args, **kwargs: migration_calls.append(True) or MigrationValidationResult(ok=True),
    )
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("merged on primary approval alone")
    )
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, max_rounds=1, auto_merge=True)

    with pytest.raises(AgentLoopError, match="exact-head final sweep is missing reviewer approval from Gemini, Antigravity"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert migration_calls == []
    assert _agent_sequence(runner) == ["codex"]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_unacknowledged_human_requirement_remains_blocking_after_panel_settlement(
    tmp_path, monkeypatch
):
    requirement = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the audit trail intact.",
    )
    original_context = pr_loop_support.get_pr_review_context

    def context_with_requirement(runner, *args, **kwargs):
        context = original_context(runner, *args, **kwargs)
        return dataclasses.replace(context, human_requirements=(requirement,))

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", context_with_requirement)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex", resolved=True)],
        # Gemini approves without acknowledging the signed requirement.
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity", resolved=True)],
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with an unacknowledged requirement")
    )

    with pytest.raises(AgentLoopError, match=r"blocking issues after round 2.*Orchestrator \(item-1\)"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _agent_sequence(runner)[:3] == ["codex", "gemini", "agy"]
    assert any("phase: secondary-audit" in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_ci_failure_remains_blocking_after_unanimous_panel_approval(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with CI failure")
    )

    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert any("GitHub PR checks are failing for PR #77." in comment for comment in runner.comments)
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_staged_policy_does_not_change_existing_policy_selection(tmp_path):
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=1)
    assert config.pr_review_policy == "all-reviewers"
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner) == ["codex", "gemini"]
    assert not any(comment.startswith("PR review scheduling audit:") for comment in runner.comments)
    assert _posted_scheduler_metadata(runner) == []


# ---------------------------------------------------------------------------
# #840: strict primary-only pre-panel phase for primary-then-panel
# ---------------------------------------------------------------------------


def _legacyize_operator_latches(payload):
    """Turn operator-attributed latches into pre-#840 unattributed latches."""
    if payload.get("scheduler_force_full_source") is None:
        return None
    payload = dict(payload)
    del payload["scheduler_force_full_source"]
    return payload


def _scheduling_audits(runner):
    return [c for c in runner.comments if c.startswith("PR review scheduling audit:")]


def _scheduling_diagnostics(runner):
    return [c for c in runner.comments if c.startswith("PR review scheduling diagnostic")]


def _comment_index(runner, predicate, *, start=0):
    return next(index for index, comment in enumerate(runner.comments) if index >= start and predicate(comment))


@pytest.mark.parametrize(
    "fix_scope, changed_path, expected_reason",
    [
        (["pyproject.toml"], "pyproject.toml", "matches a broad rule"),
        (["src/worker.py"], "src/other.py", "is outside obligation scopes"),
    ],
)
def test_840_broad_or_out_of_scope_change_before_primary_approval_is_primary_only(
    tmp_path, monkeypatch, fix_scope, changed_path, expected_reason
):
    """Row prepanel-broad-change / prepanel-ambiguous-scope (fresh run)."""
    from coding_review_agent_loop.review_scheduling import GitChange, classify_transition

    def classify(runner, *, checkout, previous_sha, current_sha, scopes, broad_rules, obligations):
        return classify_transition(
            previous_sha, current_sha, [GitChange(changed_path)],
            scopes=scopes, broad_rules=broad_rules, obligations=obligations,
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition", classify)
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": fix_scope}],
            ),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    # No secondary is invoked before the primary approves the exact head.
    assert _agent_sequence(runner) == ["codex", "claude", "codex", "gemini", "agy"]
    assert _audit_phases(runner) == [
        ("primary", "abc123"),
        ("primary", "abc123-coder-1"),
        ("secondary-audit", "abc123-coder-1"),
    ]
    strict_audit = _scheduling_audits(runner)[1]
    assert "reason: strict pre-panel fallback: " in strict_audit
    assert expected_reason in strict_audit
    assert "primary re-invoked with full context" in strict_audit
    assert "force-full: False (source: none)" in strict_audit
    codex_prompts = [command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]]
    assert "Scheduling reason: strict pre-panel fallback" in codex_prompts[1]
    assert "Inspect the complete base-to-head diff independently" in codex_prompts[1]
    metadata = _posted_scheduler_metadata(runner)
    assert all(item.scheduler_force_full is False for item in metadata)
    assert all(item.scheduler_force_full_source is None for item in metadata)
    assert not any("post-panel fallback" in comment for comment in runner.comments)


def test_840_panel_starts_after_primary_approval_and_post_panel_broad_change_latches_automatic(
    tmp_path, monkeypatch
):
    """Row panel-start-and-final-barrier: narrow, then broad, then narrow fixes after the panel opens."""
    transitions = iter(
        [
            TransitionClassification("narrow", "scoped fix"),
            TransitionClassification("broad", "diff path 'pyproject.toml' matches a broad rule"),
            TransitionClassification("narrow", "scoped fix"),
        ]
    )
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop,
    ), "_observe_pr_transition", lambda *args, **kwargs: next(transitions))

    def resolved(*item_ids):
        return [{"item_id": item_id, "disposition": "resolved"} for item_id in item_ids]

    def finding(text):
        return [{"text": text, "fix_scope": ["src/worker.py"]}]

    runner = FakeRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
            structured_coder_followup(addressed_items=["item-3"]),
        ],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-1")),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-2")),
            _staged_review(reviewer="OpenAI Codex", dispositions=resolved("item-3")),
        ],
        gemini_outputs=[
            _staged_review(reviewer="Google Gemini", state="blocking", blocking_items=finding("panel regression")),
            _staged_review(
                reviewer="Google Gemini", state="blocking",
                dispositions=resolved("item-1"), blocking_items=finding("second regression"),
            ),
            _staged_review(
                reviewer="Google Gemini", state="blocking",
                dispositions=resolved("item-2"), blocking_items=finding("third regression"),
            ),
            _staged_review(reviewer="Google Gemini", dispositions=resolved("item-3")),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=resolved("item-2")),
            _staged_review(reviewer="Antigravity", dispositions=resolved("item-3")),
        ],
    )
    merges = []
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))
    config = _staged_config(tmp_path, max_rounds=10, auto_merge=True)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = [phase for phase, _head in _audit_phases(runner)]
    # primary -> panel -> owner+primary remediation (narrow) -> full board
    # (broad) -> full board again on the later narrow head (monotonic latch).
    assert phases == ["primary", "secondary-audit", "remediation", "full-board", "full-board"]
    audits = _scheduling_audits(runner)
    assert "post-panel fallback: narrow remediation" in audits[2]
    assert "force-full: False (source: none)" in audits[2]
    assert "reason: post-panel fallback: full board required: diff path 'pyproject.toml'" in audits[3]
    assert "force-full: True (source: automatic)" in audits[3]
    assert "reason: post-panel fallback: force-full latch" in audits[4]
    assert "force-full: True (source: automatic)" in audits[4]
    assert "paused none" in audits[4]
    assert not any("strict pre-panel fallback" in audit for audit in audits[1:])
    sequence = _agent_sequence(runner)
    assert sequence[:3] == ["codex", "gemini", "agy"]
    assert sorted(sequence[-3:]) == ["agy", "codex", "gemini"]
    # The durable latch is persisted on every record from the broad head's
    # prelaunch on, including the coder checkpoint for the following narrow
    # head.  (A coder checkpoint carries the next round number but the state
    # of the round that produced it.)
    metadata = _posted_scheduler_metadata(runner)
    first_latched = next(
        index for index, item in enumerate(metadata)
        if item.phase == "scheduler-prelaunch" and item.scheduler_phase == "full-board"
    )
    assert all(
        item.scheduler_force_full is True and item.scheduler_force_full_source == "automatic"
        for item in metadata[first_latched:]
    )
    assert all(item.scheduler_force_full is False for item in metadata[:first_latched])
    assert any(item.role == "coder" for item in metadata[first_latched:])
    # Completion requires the exact final head approved by every reviewer.
    final_head = _audit_phases(runner)[-1][1]
    final_approvals = {
        record.metadata.agent
        for record in architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.role == "reviewer"
        and record.metadata.subject == final_head
        and record.metadata.state == "approved"
    }
    assert final_approvals == {"Codex", "Gemini", "Antigravity"}
    assert merges == [{"expected_head_sha": final_head}]


def test_840_post_panel_automatic_fallback_latches_with_automatic_source(tmp_path, monkeypatch):
    """An automatic recovery reason after a qualified opening keeps the durable latch."""
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(reviewer="OpenAI Codex"),
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
            _staged_review(reviewer="Google Gemini", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
        antigravity_outputs=[
            _staged_review(reviewer="Antigravity"),
            _staged_review(reviewer="Antigravity", dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    # The architecture identity observed for the coder's head is stale, which
    # is an automatic full-board reason raised after the qualified opening.
    stale_observations = []

    def observation(comments, *, head_sha):
        if head_sha == "abc123-coder-1" and not stale_observations:
            stale_observations.append(head_sha)
            return {"stale": "identity"}
        return None

    patch_attr_in(monkeypatch, (
        architecture_contract, pr_loop_support, pr_loop,
    ), "_latest_pr_architecture_observation", observation)
    config = _staged_config(tmp_path, max_rounds=6)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    phases = [phase for phase, _head in _audit_phases(runner)]
    assert phases[:2] == ["primary", "secondary-audit"]
    assert phases[2] == "full-board"
    full_board = _scheduling_audits(runner)[2]
    assert "post-panel fallback: force-full latch" in full_board
    assert "force-full: True (source: automatic)" in full_board
    latched = [
        item for item in _posted_scheduler_metadata(runner) if item.scheduler_force_full
    ]
    assert latched and all(item.scheduler_force_full_source == "automatic" for item in latched)


def test_840_resumed_primary_phase_with_incomplete_metadata_is_primary_only(tmp_path, monkeypatch):
    """Row prepanel-resume-incomplete-metadata: absent, invalid, phase-less, contradictory."""
    variants = {
        "absent": lambda payload: {
            key: value for key, value in payload.items() if not key.startswith("scheduler_")
        },
        "invalid": lambda payload: (
            {**payload, "scheduler_obligation_digest": "not-a-digest"}
            if "scheduler_contract" in payload else None
        ),
        "phase-less": lambda payload: (
            {
                key: value
                for key, value in payload.items()
                if key not in {"scheduler_phase", "scheduler_primary_reviewer"}
            }
            if "scheduler_contract" in payload else None
        ),
        "contradictory": lambda payload: (
            {**payload, "scheduler_current_sha": "some-other-head"}
            if "scheduler_contract" in payload else None
        ),
    }
    for name, transform in variants.items():
        variant_path = tmp_path / name
        variant_path.mkdir()
        runner = FakeRunner(
            codex_outputs=[
                _staged_review(
                    reviewer="OpenAI Codex",
                    state="blocking",
                    blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
                ),
            ],
        )
        # The first invocation stops after the primary's blocking review.
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path, max_rounds=1))
        assert _agent_sequence(runner) == ["codex"], name
        _rewrite_pr_metadata(runner, transform)
        runner.claude_outputs.append(structured_coder_followup(addressed_items=["item-1"]))
        runner.codex_outputs.append(
            _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}])
        )
        runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
        runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
        comments_before = len(runner.comments)

        if name in {"invalid", "contradictory"}:
            # The pre-existing qualification guard still refuses malformed or
            # contradictory history, but only after the strictly primary-only
            # phase: no secondary was spent on the uncertainty.
            with pytest.raises(AgentLoopError, match="scheduler (head )?metadata was observed during qualification"):
                run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path))
            assert set(_agent_sequence(runner)) == {"codex", "claude"}, name
            new_audits = [
                c for c in runner.comments[comments_before:] if c.startswith("PR review scheduling audit:")
            ]
            assert new_audits and all("phase: primary" in audit for audit in new_audits)
            assert "strict pre-panel fallback: scheduler metadata recovery" in new_audits[0]
            assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))
            continue

        assert run_pr_loop(runner, pr_number=77, config=_staged_config(variant_path)) == 0, name

        sequence = _agent_sequence(runner)[1:]
        first_secondary = min(
            index for index, agent in enumerate(sequence) if agent in {"gemini", "agy"}
        )
        # Only the primary (and coder) run until the primary approves the exact head.
        assert set(sequence[:first_secondary]) <= {"codex", "claude"}, (name, sequence)
        assert sequence[first_secondary - 1] == "codex", (name, sequence)
        new_audits = [
            c for c in runner.comments[comments_before:] if c.startswith("PR review scheduling audit:")
        ]
        assert all("phase: primary" in audit for audit in new_audits[:-1]), name
        assert "phase: secondary-audit" in new_audits[-1], name
        posted = _posted_scheduler_metadata(runner)
        assert not any(item.scheduler_force_full for item in posted), name
        assert not any("post-panel fallback" in c for c in runner.comments), name


def _seed_premature_full_board_round(tmp_path, *, gemini_output, parallel=False):
    """Simulate a pre-#840 run: an automatic pre-approval full board on abc123.

    The operator flag produces the full board; the operator attribution is then
    stripped so the history matches the unattributed latch pre-#840 runs wrote.
    Antigravity becomes unavailable, so the round is interrupted.
    """
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[gemini_output],
        antigravity_outputs=[_unavailable("Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=True, review_parallel=parallel)
    # An approving round stops on the unavailable reviewer; a blocking round
    # stops when the (unscripted) coder follow-up cannot run.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert sorted(_agent_sequence(runner)[:3]) == ["agy", "codex", "gemini"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    assert not any(item.scheduler_force_full_source for item in _posted_scheduler_metadata(runner))
    return runner


@pytest.mark.parametrize("parallel", [False, True])
def test_840_premature_secondary_approvals_are_not_resumed_carried_or_counted(tmp_path, parallel):
    """Row premature-secondary-approval-not-carried (same-round and historical)."""
    runner = _seed_premature_full_board_round(
        tmp_path, gemini_output=_staged_review(reviewer="Google Gemini"), parallel=parallel
    )
    seeded = len(runner.comments)
    if parallel:
        # Parallel reviewers publish completion-order records.
        records = architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        assert any(
            record.metadata.agent == "Gemini" and record.metadata.phase == "publication"
            for record in records
        )
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))
    if parallel:
        # #1156: the premature Gemini review is public, so Antigravity's fresh
        # post-opening turn would read it.  Visibility, not resume
        # qualification, decides: the round is refused before any launch.
        launches_before = len(_agent_sequence(runner))
        with pytest.raises(AgentLoopError, match="partially published") as excinfo:
            run_pr_loop(
                runner, pr_number=77,
                config=_staged_config(tmp_path, auto_merge=True, review_parallel=parallel),
            )
        assert "Gemini review" in str(excinfo.value)
        assert len(_agent_sequence(runner)) == launches_before
        return
    merges = []

    original_merge = pr_loop_support.merge_pr
    pr_loop_support.merge_pr = lambda *args, **kwargs: merges.append(kwargs)
    try:
        assert run_pr_loop(
            runner, pr_number=77, config=_staged_config(tmp_path, auto_merge=True, review_parallel=parallel)
        ) == 0
    finally:
        pr_loop_support.merge_pr = original_merge

    new_agents = _agent_sequence(runner)[3:]
    # The primary's own same-round approval is resumed; every secondary gets a
    # fresh post-opening turn, including Gemini whose premature approval exists.
    assert sorted(new_agents) == ["agy", "gemini"]
    new_comments = runner.comments[seeded:]
    audits = [c for c in new_comments if c.startswith("PR review scheduling audit:")]
    assert len(audits) == 1
    assert "phase: secondary-audit" in audits[0]
    assert "selected Gemini, Antigravity" in audits[0]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    gemini_review = _comment_index(runner, lambda c: "Google Gemini review" in c, start=seeded)
    assert gemini_review > opening
    posted = _posted_scheduler_metadata(runner)
    opening_record = next(item for item in posted if item.phase == "scheduler-prelaunch" and item.scheduler_phase == "secondary-audit")
    # The premature Gemini approval never entered the opening's approval set.
    assert opening_record.scheduler_approved_reviewers == ("Codex",)
    assert merges == [{"expected_head_sha": "abc123"}]


def test_840_premature_historical_secondary_approvals_are_not_carried(tmp_path):
    """Premature approvals from an earlier round on the same head are not carried."""
    runner = FakeRunner(
        codex_outputs=[_unavailable("OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="Codex"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded = len(runner.comments)
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex"))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    # Round 1 resumes: premature same-round secondary approvals are dropped and
    # the primary reviews alone.  Round 2 (same head): the round-1 secondary
    # approvals are older-round exact-head approvals, yet neither is carried.
    assert _agent_sequence(runner)[3:] == ["codex", "gemini", "agy"]
    audits = [c for c in runner.comments[seeded:] if c.startswith("PR review scheduling audit:")]
    assert [re.search(r"phase: ([a-z-]+);", audit).group(1) for audit in audits] == [
        "primary", "secondary-audit",
    ]
    assert "unqualified pre-approval panel history was ignored" in audits[0]
    assert "rerun with --pr-review-force-full to restore the complete board" in audits[0]
    assert "selected Gemini, Antigravity" in audits[1]
    assert not any("skipping Gemini; it approved unchanged" in c for c in runner.comments)


def test_840_blocking_premature_secondary_completion_stops_with_diagnostic(tmp_path):
    """Rows premature-panel-history-resume / prepanel-unsafe-stop (resume path)."""
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    seeded_agents = len(_agent_sequence(runner))
    seeded_comments = len(runner.pr_payload["comments"])

    with pytest.raises(pr_loop.PrePanelSafetyError, match="--pr-review-force-full"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))

    # No reviewer or coder ran, nothing was latched, and the only new comment
    # is the plain diagnostic (no scheduler metadata).
    assert len(_agent_sequence(runner)) == seeded_agents
    new_comments = runner.pr_payload["comments"][seeded_comments:]
    assert len(new_comments) == 1
    assert new_comments[0]["body"].startswith("PR review scheduling diagnostic (round 1): pre-panel safety")
    assert "AGENT_LOOP_META" not in new_comments[0]["body"]
    assert "Gemini (round 1, head abc123, state blocking" in new_comments[0]["body"]


def test_840_operator_override_supersedes_blocking_premature_completion(tmp_path):
    """Row premature-blocking-completion-operator-recovery (k3 with the flag)."""
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)
    premature_item_ids = {
        item.item_id
        for record in architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.agent == "Gemini"
        for item in record.metadata.new_items
    }
    assert premature_item_ids == {"item-1"}
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex"))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)
    ) == 0

    # Every configured reviewer is freshly invoked after the operator opening;
    # none is resumed or carried.
    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    opening_text = runner.comments[opening]
    assert "phase: full-board" in opening_text
    assert "force-full: True (source: operator)" in opening_text
    assert "Superseded premature panel reviews" in opening_text
    assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in opening_text
    for marker in ("OpenAI Codex review", "Google Gemini review", "Antigravity review"):
        assert _comment_index(runner, lambda c, marker=marker: marker in c, start=seeded) > opening
    gemini_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]][-1]
    assert "Superseded pre-panel review context (non-authoritative; context only):" in gemini_prompt
    assert "- Earlier claim: premature cache race" in gemini_prompt
    codex_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]][-1]
    assert "Superseded pre-panel review context" not in codex_prompt
    # The premature item never entered the ledger or ownership accounting.
    later_records = [
        record
        for record in architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.index >= len(runner.pr_payload["comments"]) - (len(runner.comments) - seeded)
    ]
    for record in later_records:
        assert not ({item.item_id for item in record.metadata.prior_items} & premature_item_ids)
        assert not ({item.item_id for item in record.metadata.new_items} & premature_item_ids)
    operator_records = [item for item in _posted_scheduler_metadata(runner) if item.scheduler_force_full_source == "operator"]
    assert operator_records


@pytest.mark.parametrize("operator", [False, True])
def test_840_stale_identity_premature_blocking_completion_is_not_silently_dropped(tmp_path, operator):
    """Row premature-blocking-completion-operator-recovery with a stale resume identity.

    A signed requirement surfaced after the premature blocking review makes
    that record ineligible for resume.  It must still reach the diagnostic
    (without the flag) or the audited, context-preserving supersession (with
    it) instead of being dropped before panel classification.
    """
    runner = _seed_premature_full_board_round(
        tmp_path,
        gemini_output=_staged_review(
            reviewer="Google Gemini",
            state="blocking",
            blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
        ),
    )
    runner.pr_payload["comments"].append(
        {
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-18T10:00:00Z",
            "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-840",
            "body": "Keep the worker cleanup idempotent.\n\n-- Human Reviewer",
        }
    )
    premature = next(
        record
        for record in architecture_contract._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
        )
        if record.metadata.agent == "Gemini"
    )
    from coding_review_agent_loop.github import _parse_pr_human_requirements

    requirements = _parse_pr_human_requirements(runner.pr_payload)
    assert requirements
    assert not response_validation._resumed_pr_reviewer_matches_requirements(premature, requirements)
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)

    if not operator:
        with pytest.raises(pr_loop.PrePanelSafetyError, match="--pr-review-force-full"):
            run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
        assert len(_agent_sequence(runner)) == seeded_agents
        new_comments = runner.comments[seeded:]
        assert len(new_comments) == 1
        assert new_comments[0].startswith("PR review scheduling diagnostic (round 1): pre-panel safety")
        assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in new_comments[0]
        return

    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", resolved=True))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", resolved=True))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", resolved=True))

    assert run_pr_loop(
        runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)
    ) == 0

    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    opening = _comment_index(runner, lambda c: c.startswith("PR review scheduling audit:"), start=seeded)
    opening_text = runner.comments[opening]
    assert "force-full: True (source: operator)" in opening_text
    assert "Superseded premature panel reviews" in opening_text
    assert "Gemini (round 1, head abc123, state blocking; items: item-1)" in opening_text
    assert _comment_index(runner, lambda c: "Google Gemini review" in c, start=seeded) > opening
    gemini_prompt = [command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]][-1]
    assert "Superseded pre-panel review context (non-authoritative; context only):" in gemini_prompt
    assert "- Earlier claim: premature cache race" in gemini_prompt


def test_840_operator_force_full_is_durable_across_resume(tmp_path):
    """Row operator-force-full: the latch survives a resume that omits the flag."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavailable("Antigravity"), _staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="Antigravity"):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    first_audit = _scheduling_audits(runner)[0]
    assert "reason: operator force-full" in first_audit
    assert "force-full: True (source: operator)" in first_audit

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    # Completed post-opening work is reused; the operator latch is not
    # downgraded to a primary-only round on resume.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy", "agy"]
    audits = _scheduling_audits(runner)
    assert "phase: full-board" in audits[-1]
    assert "force-full: True (source: operator)" in audits[-1]
    assert all(
        item.scheduler_force_full is True and item.scheduler_force_full_source == "operator"
        for item in _posted_scheduler_metadata(runner)
    )


def test_840_secondary_owned_ledger_item_before_panel_stops_unless_operator(tmp_path, monkeypatch):
    """Row premature-panel-history-resume: a premature secondary finding reached the ledger."""
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "premature cache race", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # Round 1 is a pre-#840 style full board; the coder then fixes the item and
    # the run stops when round 2's first reviewer has no scripted output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded_agents = len(_agent_sequence(runner))

    with pytest.raises(pr_loop.PrePanelSafetyError) as raised:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    message = str(raised.value)
    assert "item-1" in message and "Gemini" in message
    assert "--pr-review-force-full" in message
    assert len(_agent_sequence(runner)) == seeded_agents
    assert _scheduling_diagnostics(runner)

    # The operator override authorizes the complete board instead.  (The fake
    # runner would otherwise treat the replayed coder JSON in reviewer prompts
    # as a new coder push.)
    runner.advance_pr_head_on_coder_followup = False
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=resolved))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=resolved))
    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True)) == 0
    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    assert "force-full: True (source: operator)" in _scheduling_audits(runner)[-1]


def test_840_premature_approved_secondary_history_resumes_as_strict_primary_turn(tmp_path, monkeypatch):
    """Row premature-panel-history-resume variant: premature secondaries approved."""
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=True))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]
    _rewrite_pr_metadata(runner, _legacyize_operator_latches)
    seeded_agents = len(_agent_sequence(runner))
    seeded = len(runner.comments)
    runner.codex_outputs.append(
        _staged_review(reviewer="OpenAI Codex", dispositions=[{"item_id": "item-1", "disposition": "resolved"}])
    )
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity"))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert _agent_sequence(runner)[seeded_agents:] == ["codex", "gemini", "agy"]
    audits = [c for c in runner.comments[seeded:] if c.startswith("PR review scheduling audit:")]
    assert "phase: primary" in audits[0]
    assert "strict pre-panel fallback" in audits[0]
    assert "unqualified pre-approval panel history was ignored" in audits[0]
    assert "phase: secondary-audit" in audits[1]
    assert not any("post-panel fallback" in c for c in runner.comments[seeded:])
    new_metadata = _posted_scheduler_metadata(runner)[-4:]
    assert not any(item.scheduler_force_full for item in new_metadata)


def test_840_machine_obligation_before_panel_is_primary_only_then_panel_opens(tmp_path, monkeypatch):
    """Rows prepanel-machine-obligation and prepanel-primary-approved-with-machine-obligation."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    monkeypatch.setattr(
        pr_loop_support, "merge_pr", lambda *args, **kwargs: pytest.fail("merged with CI failure")
    )
    config = _staged_config(tmp_path, max_rounds=2, auto_merge=True)

    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=config)

    # The primary approved the exact head, so the panel opened on it even
    # though the CI obligation remains; no diagnostic was raised.
    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    assert not _scheduling_diagnostics(runner)
    audits = _scheduling_audits(runner)
    assert "phase: secondary-audit" in audits[1]

    # A resume derives qualified panel evidence from that prelaunch record.
    records = architecture_contract._extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
    )
    evidence = panel_evidence._derive_pr_panel_evidence(
        records, primary_reviewer="Codex", required_reviewers=("Codex", "Gemini", "Antigravity")
    )
    assert evidence.opened and evidence.opening_source == "primary-approval"


def test_840_ci_obligation_before_primary_approval_does_not_invoke_panel(tmp_path, monkeypatch):
    """Row prepanel-machine-obligation: CI item active while the primary lacks approval."""
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            _staged_review(
                reviewer="OpenAI Codex",
                state="blocking",
                blocking_items=[{"text": "worker cleanup gap", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]
        },
    )
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, max_rounds=3))
    # Round 2 only re-invokes the primary; no diagnostic, no secondary.
    assert "gemini" not in _agent_sequence(runner)
    assert "agy" not in _agent_sequence(runner)
    assert not _scheduling_diagnostics(runner)
    assert all("phase: primary" in audit for audit in _scheduling_audits(runner))
    assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))


def test_840_architecture_identity_change_before_panel_is_primary_only(tmp_path, monkeypatch):
    """Row prepanel-checkpoint-or-architecture-invalidation."""
    stale_observations = []

    def observation(comments, *, head_sha):
        if not stale_observations:
            stale_observations.append(head_sha)
            return {"stale": "identity"}
        return None

    patch_attr_in(monkeypatch, (
        architecture_contract, pr_loop_support, pr_loop,
    ), "_latest_pr_architecture_observation", observation)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert _agent_sequence(runner) == ["codex", "gemini", "agy"]
    audits = _scheduling_audits(runner)
    assert "phase: primary" in audits[0]
    assert "strict pre-panel fallback: architecture identity changed" in audits[0]
    assert "phase: secondary-audit" in audits[1]
    assert not any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))


def test_840_legacy_latch_after_qualified_opening_is_honored(tmp_path, monkeypatch):
    """Row legacy-latch-compatibility: a post-opening unattributed latch keeps the full board."""
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_observe_pr_transition",
        lambda *args, **kwargs: TransitionClassification("narrow", "scoped fix"),
    )
    runner = FakeRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(
                reviewer="Google Gemini",
                state="blocking",
                blocking_items=[{"text": "panel regression", "fix_scope": ["src/worker.py"]}],
            ),
        ],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    # primary -> panel -> coder; the remediation round then runs out of output.
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    assert _agent_sequence(runner)[:4] == ["codex", "gemini", "agy", "claude"]

    def latch_after_opening(payload):
        # Pretend the coder checkpoint (after the qualified opening) latched
        # the board under pre-#840 code, without an attribution source.
        if payload.get("role") != "coder" or "scheduler_contract" not in payload:
            return None
        return {**payload, "scheduler_force_full": True}

    _rewrite_pr_metadata(runner, latch_after_opening)
    seeded_agents = len(_agent_sequence(runner))
    runner.advance_pr_head_on_coder_followup = False
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner.codex_outputs.append(_staged_review(reviewer="OpenAI Codex", dispositions=resolved))
    runner.gemini_outputs.append(_staged_review(reviewer="Google Gemini", dispositions=resolved))
    runner.antigravity_outputs.append(_staged_review(reviewer="Antigravity", dispositions=resolved))

    assert run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path)) == 0

    assert sorted(_agent_sequence(runner)[seeded_agents:]) == ["agy", "codex", "gemini"]
    audit = _scheduling_audits(runner)[-1]
    assert "phase: full-board" in audit
    assert "post-panel fallback: force-full latch" in audit
    assert "force-full: True (source: automatic)" in audit


@pytest.mark.parametrize("operator", [False, True])
def test_840_round_boundary_undecodable_history_stops_with_diagnostic(tmp_path, monkeypatch, operator):
    """Row prepanel-unsafe-stop: panel state is unknowable when history cannot be decoded."""
    state = {"broken": False}
    original_extract = architecture_contract._extract_round_metadata_records
    original_mergeability = pr_loop_support.get_pr_mergeability

    def extract(comments, *, flow):
        if state["broken"]:
            raise AgentLoopError("Incomplete round metadata: sidecars are unavailable")
        return original_extract(comments, flow=flow)

    def mergeability(*args, **kwargs):
        # History becomes undecodable after startup, at the round boundary.
        state["broken"] = True
        return original_mergeability(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        architecture_contract, panel_evidence, review_rounds, discuss_loop, execution_policy,
        child_plan_binding, pr_loop_support, pr_loop, plan_first_loop, history_export,
    ), "_extract_round_metadata_records", extract)
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "get_pr_mergeability", mergeability)
    patch_attr_in(monkeypatch, (
        review_rounds, pr_loop, plan_first_loop,
    ), "_round_ledger_may_be_incomplete", lambda **kwargs: False)
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    config = _staged_config(tmp_path, pr_review_force_full=operator, max_rounds=1)

    # Resume, approval, ledger, and qualification accounting all depend on the
    # history, so neither flag state may spend reviewers over an undecodable
    # ledger: both stop through the same no-reviewer diagnostic.
    with pytest.raises(pr_loop.PrePanelSafetyError, match="could not be decoded"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert _agent_sequence(runner) == []
    diagnostics = _scheduling_diagnostics(runner)
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("PR review scheduling diagnostic (round 1): ")
    assert "Restore the missing round-metadata records or sidecars" in diagnostics[0]
    assert "--pr-review-force-full cannot authorize" in diagnostics[0]
    assert not _scheduling_audits(runner)


@pytest.mark.parametrize("operator", [False, True])
def test_840_startup_undecodable_history_stops_with_diagnostic(tmp_path, operator):
    """Row prepanel-unsafe-stop at startup: a real missing sidecar reference."""
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    contract = pr_loop.make_contract(("Codex", "Gemini", "Antigravity"), "primary-then-panel", None, "Codex")
    checkpoint = _attach_round_metadata(
        "Staged checkpoint.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
            scheduler_contract=contract.as_dict(), scheduler_previous_sha=None,
            scheduler_current_sha="abc123", scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_paused_reviewers=(("Gemini", "primary phase"), ("Antigravity", "primary phase")),
            scheduler_reasons=("primary phase",), scheduler_final_sweep=False,
            scheduler_force_full=False, scheduler_calls_avoided=2, scheduler_phase="primary",
            scheduler_primary_reviewer="Codex",
        ),
    )
    match = ROUND_RESUME_MARKER_RE.search(checkpoint)
    payload = decode_mapping(match.group("payload"))
    # Reference a spilled field whose sidecar comment does not exist.
    payload["canonical_reviewer_response"] = {
        "$round_transport_spill": "missing-anchor",
        "parts": 1,
        "sha256": "0" * 64,
        "spill": "0" * 64,
    }
    broken = checkpoint[: match.start("payload")] + encode_mapping(payload) + checkpoint[match.end("payload"):]
    with pytest.raises(AgentLoopError, match="sidecars are unavailable"):
        architecture_contract._extract_round_metadata_records([SimpleNamespace(body=broken)], flow="pr")
    runner = FakeRunner(
        pr_payload={"comments": [{"author": {"login": "bot"}, "body": broken}]},
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )

    with pytest.raises(pr_loop.PrePanelSafetyError, match="could not be decoded") as raised:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path, pr_review_force_full=operator))

    assert "sidecars are unavailable" in str(raised.value)
    assert _agent_sequence(runner) == []
    diagnostics = _scheduling_diagnostics(runner)
    assert len(diagnostics) == 1
    assert diagnostics[0].startswith("PR review scheduling diagnostic (startup): pre-panel safety")
    assert "--pr-review-force-full cannot authorize" in diagnostics[0]
    # The diagnostic carries no round metadata that could be mistaken for a checkpoint.
    assert "AGENT_LOOP_META" not in runner.pr_payload["comments"][-1]["body"]
    assert not _scheduling_audits(runner)


def test_840_startup_undecodable_history_keeps_legacy_error_for_other_policies(tmp_path):
    """Row other-policies-unchanged: non-staged policies still raise the decode error."""
    from coding_review_agent_loop.round_transport import (
        ROUND_RESUME_MARKER_RE,
        decode_mapping,
        encode_mapping,
    )

    body = _attach_round_metadata(
        "Reviewer record.",
        PostedRoundMetadata(flow="pr", role="reviewer", agent="Codex", round_number=1, subject="abc123", state="approved"),
    )
    match = ROUND_RESUME_MARKER_RE.search(body)
    payload = decode_mapping(match.group("payload"))
    payload["canonical_reviewer_response"] = {
        "$round_transport_spill": "missing-anchor", "parts": 1, "sha256": "0" * 64, "spill": "0" * 64,
    }
    broken = body[: match.start("payload")] + encode_mapping(payload) + body[match.end("payload"):]
    runner = FakeRunner(pr_payload={"comments": [{"author": {"login": "bot"}, "body": broken}]})
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), pr_review_policy="selective-intermediate"
    )
    with pytest.raises(AgentLoopError, match="sidecars are unavailable") as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert not isinstance(raised.value, pr_loop.PrePanelSafetyError)
    assert not _scheduling_diagnostics(runner)


_AGENT_UNAVAILABLE_ENV = (
    json.dumps(
        {
            "schema_version": 1,
            "kind": "agent_unavailable",
            "retryable": False,
            "category": "environment",
            "summary": "The reviewer sandbox is down.",
            "suggested_action": "Repair the reviewer sandbox.",
        }
    )
    + "\n<!-- AGENT_UNAVAILABLE -->\n-- Google Gemini"
)


def _incomplete_comments(runner):
    return [body for body in runner.comments if "**Review status: Incomplete**" in body]


def test_evidence_response_pass_with_unavailable_reviewer_stops_before_the_gate_and_reruns_in_full(
    tmp_path, monkeypatch
):
    """Rows evidence-response-pass-unavailable and evidence-pass-stop-then-restore-rerun."""
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)

    gate_calls = []
    real_gate = pr_loop_support._evidence_freeze_gate
    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop), "_evidence_freeze_gate",
        lambda *a, **k: gate_calls.append(1) or real_gate(*a, **k),
    )
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(_AGENT_UNAVAILABLE_ENV)
    with pytest.raises(AgentLoopError, match="missing required input from Gemini") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "evidence gate proceeded" not in str(excinfo.value)
    assert gate_calls == []
    assert _release_records(runner) == []
    assert len(_freeze_records(runner)) == 1
    assert len(_incomplete_comments(runner)) == 1
    assert "in full with the remaining board on rerun" in _incomplete_comments(runner)[0]

    # Backend restored: the unterminated pass is discarded and the FULL board reruns.
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(
        _evidence_review(
            reviewer="Google Gemini", hr=True,
            dispositions=[_resolve("item-1", "Codex owns this request.")],
        )
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    releases = _release_records(runner)
    assert len(releases) == 1 and releases[0].metadata.evidence_release.reason == "evidence-cleared"


def _unavail(label):
    return _AGENT_UNAVAILABLE_ENV.replace("Google Gemini", label)


def _scheduler_stop_runner(**overrides):
    values = dict(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_unavail("Antigravity")],
    )
    values.update(overrides)
    return FakeRunner(**values)


def _stop_commands_after(runner, mark):
    return [cmd for cmd, _cwd in runner.commands[mark:] if cmd]


def test_scheduler_stop_comment_carries_prose_and_error_carries_validated_template(tmp_path):
    """Rows selective-panel-unavailable-remediation and printed-amendment-accepted-on-rerun."""
    runner = _scheduler_stop_runner()
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    message = str(excinfo.value)
    assert "Reviewer board amendment:" in message
    incomplete = _incomplete_comments(runner)
    assert len(incomplete) == 1
    assert "effective from round" in incomplete[0]
    assert "original board" in incomplete[0]
    assert "reviewer-board-amendment" not in incomplete[0]
    assert "~~~" not in incomplete[0] and "```" not in incomplete[0] and "{" not in incomplete[0]
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any("pausing" in comment for comment in runner.comments)
    assert "Finalization stops" in " ".join(runner.comments)


def test_scheduler_stop_template_is_accepted_on_rerun_with_the_original_board(
    tmp_path, monkeypatch
):
    """Row printed-amendment-accepted-on-rerun: the template printed by the stop itself."""
    runner = _scheduler_stop_runner()
    original = _staged_config(tmp_path, auto_merge=True)
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=original)
    printed_round = int(re.search(r"effective from round (\d+)", _incomplete_comments(runner)[0]).group(1))
    template = _m943_amendment_from_error(str(excinfo.value))
    assert f'"effective_from_round": {printed_round}' in template
    # Post the first stop's template as-is and rerun with the ORIGINAL board.
    _m943_append(runner, template)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)
    mark = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=original) == 0
    assert "agy" not in [cmd[0] for cmd in _stop_commands_after(runner, mark)]
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)
    assert not any("scheduler contract changed" in comment for comment in runner.comments)


def test_scheduler_stop_with_primary_unavailable_is_prose_only_without_patching(tmp_path):
    """Row primary-unavailable-no-template with the real amend_contract dry-run."""
    runner = FakeRunner(
        codex_outputs=[_unavail("OpenAI Codex")],
        gemini_outputs=[_staged_review(reviewer="Google Gemini")],
        antigravity_outputs=[_staged_review(reviewer="Antigravity")],
    )
    with pytest.raises(AgentLoopError, match="missing required input from Codex") as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    message = str(excinfo.value)
    assert "cannot be reduced for this run" in message
    assert "Reviewer board amendment:" not in message
    incomplete = _incomplete_comments(runner)
    assert len(incomplete) == 1 and "cannot be reduced for this run" in incomplete[0]
    assert "signed reviewer-board amendment" not in incomplete[0]


def test_scheduler_stop_with_last_secondary_unavailable_is_prose_only(tmp_path):
    """Row primary-unavailable-no-template: 2-member board, only secondary unavailable."""
    runner = FakeRunner(
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[_unavail("Google Gemini")],
    )
    config = _staged_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(AgentLoopError, match="missing required input from Gemini") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    message = str(excinfo.value)
    assert "cannot be reduced for this run" in message
    assert "Reviewer board amendment:" not in message
    incomplete = _incomplete_comments(runner)
    assert len(incomplete) == 1 and "cannot be reduced for this run" in incomplete[0]


def test_scheduler_stop_survives_a_failing_fresh_context_fetch(tmp_path, monkeypatch):
    """Row advisory-lookup-failure-fallback: the fresh PR-context fetch raises."""
    stopping = []
    real_advisory = review_rounds._unavailable_reviewer_amendment_advisory
    real_fetch = pr_loop_support.get_pr_review_context

    def mark_advisory(**kwargs):
        stopping.append(True)
        return real_advisory(**kwargs)

    def fetch(*args, **kwargs):
        if stopping:
            raise RuntimeError("context fetch exploded")
        return real_fetch(*args, **kwargs)

    patch_attr_in(monkeypatch, (
        review_rounds, pr_loop,
    ), "_unavailable_reviewer_amendment_advisory", mark_advisory)
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context", fetch)
    runner = _scheduler_stop_runner()
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=_staged_config(tmp_path))
    assert stopping
    message = str(excinfo.value)
    assert "context fetch exploded" not in message
    assert "Reviewer board amendment:" not in message
    assert "the round the rerun resumes into" in message
    assert len(_incomplete_comments(runner)) == 1


def _all_reviewers_stop(tmp_path, *, codex_outputs, claude_outputs):
    runner = FakeRunner(
        codex_outputs=list(codex_outputs),
        claude_outputs=list(claude_outputs),
        antigravity_outputs=[_unavail("Antigravity")],
    )
    config = make_config(tmp_path, reviewer=("codex", "claude", "antigravity"), max_rounds=3)
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    return runner, str(excinfo.value)


def test_all_reviewers_flag_removal_rerun_proceeds_without_the_removed_reviewer(tmp_path):
    """Row all-reviewers-flag-removal-rerun."""
    runner, message = _all_reviewers_stop(
        tmp_path,
        codex_outputs=[structured_pr_review(state="approved", summary="ok", reviewer="OpenAI Codex")],
        claude_outputs=[structured_pr_review(state="approved", summary="ok", reviewer="Anthropic Claude")],
    )
    assert "--reviewer flags" in message
    assert "Reviewer board amendment:" not in message
    incomplete = _incomplete_comments(runner)
    assert len(incomplete) == 1
    assert "Open must-fix items (including carried): 0" in incomplete[0]
    mark = len(runner.commands)
    runner.codex_outputs.append(
        structured_pr_review(state="approved", summary="ok", reviewer="OpenAI Codex")
    )
    runner.claude_outputs.append(
        structured_pr_review(state="approved", summary="ok", reviewer="Anthropic Claude")
    )
    reduced = make_config(tmp_path, reviewer=("codex", "claude"), max_rounds=3)
    assert run_pr_loop(runner, pr_number=77, config=reduced) == 0
    assert "agy" not in [cmd[0] for cmd in _stop_commands_after(runner, mark)]


def test_incomplete_non_actionable_review_beside_a_blocking_peer_stops_at_detection(tmp_path):
    """Row incomplete-review-with-peer-blocking (markerless prose = incomplete review)."""
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="blocking", summary="Needs work.",
                blocking_items=["Add the missing test."], reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
        ],
        gemini_outputs=[
            structured_pr_review(
                state="blocking", summary="The review is incomplete; I could not verify the diff.",
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=3)
    runner.antigravity_outputs = [
        structured_pr_review(state="approved", summary="ok", reviewer="Antigravity")
    ]
    with pytest.raises(AgentLoopError, match="missing required input from Gemini") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "Blocking reviewers this round: Codex" in str(excinfo.value)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert len(_incomplete_comments(runner)) == 1


def test_strict_managed_all_reviewers_stop_suggests_restore_only(tmp_path, monkeypatch):
    """Row strict-managed-all-reviewers-restore-only on a real strict issue-created handoff."""
    plan_board = ("codex", "claude", "antigravity")
    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, plan_first_loop, issue_loop,
    ), "_resume_plan_round", lambda comments, **_kw: ("approved plan", object())
    )
    patch_attr_in(monkeypatch, (
        panel_evidence, pr_loop_support, pr_loop,
    ), "_require_complete_canonical_plan_approval", lambda *a, **k: None
    )
    # The fixture stubs the completion predicate, including the shared
    # classification recovery now applies to real plan history (#1377).
    import coding_review_agent_loop.plan_verification as plan_verification

    monkeypatch.setattr(plan_verification, "require_recoverable_plan_approval", lambda *a, **k: None)
    handoff = pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="strict", override_nonce=None,
    )
    assert pr_loop_support._managed_binding_protection_mode(handoff) == "strict"
    monkeypatch.setattr(
        pr_loop, "authorize_fresh_issue_created_resume", lambda *a, **k: handoff
    )
    monkeypatch.setattr(
        pr_loop, "revalidate_issue_created_handoff", lambda *a, **k: k["handoff"]
    )
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci",
        lambda *a, **k: ManagedCiContract(protocol_version=2, protection_mode="strict"),
    )
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="ok", reviewer="OpenAI Codex")],
        claude_outputs=[structured_pr_review(state="approved", summary="ok", reviewer="Anthropic Claude")],
        antigravity_outputs=[_unavail("Antigravity")],
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(
        tmp_path, reviewer=plan_board, max_rounds=3, managed_ci=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop",
    )
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    message = str(excinfo.value)
    assert "--reviewer flags" not in message
    assert "Reviewer board amendment:" not in message
    assert "Restore the reviewer backend" in message
    incomplete = _incomplete_comments(runner)
    assert len(incomplete) == 1
    assert "--reviewer flags" not in incomplete[0]
    assert "reviewer-board amendment" not in incomplete[0]


def test_carried_migration_obligation_is_not_revalidated_when_a_reviewer_is_unavailable(
    tmp_path, monkeypatch
):
    """Row carried-migration-obligation-unavailable."""
    def approve(name):
        return structured_pr_review(state="approved", summary="ok", reviewer=name)

    def approve_again(name):
        return structured_pr_review(
            state="approved", summary="ok", reviewer=name,
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        )

    runner = FakeRunner(
        codex_outputs=[approve("OpenAI Codex"), approve_again("OpenAI Codex")],
        claude_outputs=[approve("Anthropic Claude"), approve_again("Anthropic Claude")],
        antigravity_outputs=[approve("Antigravity"), _unavail("Antigravity")],
        gemini_outputs=[
            structured_coder_followup(
                state="blocking", summary="Fixed the migration.", addressed_items=["item-1"],
                reviewer="Google Gemini",
            )
        ],
    )
    config = make_config(
        tmp_path, coder="gemini", reviewer=("codex", "claude", "antigravity"), max_rounds=3
    )
    calls = []
    validations = iter(
        [
            MigrationValidationResult(ok=False, message="down_revision does not match head."),
            MigrationValidationResult(ok=True),
        ]
    )

    def validate(*args, **kwargs):
        calls.append("validate")
        return next(validations)

    monkeypatch.setattr(pr_loop, "validate_pr_migration_topology", validate)
    real_sync = pr_loop.sync_coder_pr_before_validation
    monkeypatch.setattr(
        pr_loop, "sync_coder_pr_before_validation",
        lambda *a, **k: calls.append("sync") or real_sync(*a, **k),
    )
    with pytest.raises(AgentLoopError, match="missing required input from Antigravity") as excinfo:
        run_pr_loop(runner, pr_number=77, config=config)
    # Round 1 validated once; the round-2 stop must not sync or validate again.
    assert calls.count("validate") == 1
    assert calls.count("sync") == 1
    message = str(excinfo.value)
    assert "Detected in round 2" in message
    assert "Open must-fix items (including carried): 1" in message
    assert sum(cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands) == 1


def test_refresh_pass_with_unavailable_reviewer_publishes_no_release_and_reruns_in_full(
    tmp_path, monkeypatch
):
    """Rows evidence-response-pass-unavailable and evidence-pass-stop-then-restore-rerun (refresh)."""
    runner = FakeRunner(
        codex_outputs=[_evidence_review(evidence=[_EVIDENCE_TEXT])],
        gemini_outputs=[_evidence_review(reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    runner.pr_payload["comments"].append(_signed("Live suite at abc123: 11 passed.", 1))
    runner.codex_outputs.append(_evidence_review(dispositions=[_resolve()], hr=True))
    runner.gemini_outputs.append(
        _evidence_review(
            reviewer="Google Gemini", hr=True,
            dispositions=[_resolve("item-1", "Codex owns this request.")],
        )
    )
    real_checks = pr_loop.get_pr_checks
    stops = []

    def stop_after_release(*args, **kwargs):
        if _release_records(runner) and not stops:
            stops.append(True)
            raise AgentLoopError("simulated process stop before finalization")
        return real_checks(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "get_pr_checks", stop_after_release)
    with pytest.raises(AgentLoopError, match="simulated process stop"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_release_records(runner)) == 1

    # New signed input forces a refresh pass; Gemini is unavailable in it.
    runner.pr_payload["comments"].append(_signed("One more requirement: keep the flag.", 2))
    runner.codex_outputs.append(_evidence_review(hr=True, summary="Checked the new requirement."))
    runner.gemini_outputs.append(_AGENT_UNAVAILABLE_ENV)
    with pytest.raises(AgentLoopError, match="missing required input from Gemini"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_release_records(runner)) == 1
    assert len(_incomplete_comments(runner)) == 1

    # Backend restored: the whole refresh pass repeats with both reviewers.
    runner.codex_outputs.append(_evidence_review(hr=True, summary="Checked the new requirement."))
    runner.gemini_outputs.append(_evidence_review(reviewer="Google Gemini", hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert runner.codex_outputs == [] and runner.gemini_outputs == []
    reasons = [record.metadata.evidence_release.reason for record in _release_records(runner)]
    assert reasons == ["evidence-cleared", "refresh-clean"]


@pytest.mark.parametrize("policy", ["selective-intermediate", "primary-then-panel"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("persistent_latch", [False, True])
def test_non_capture_digest_change_preserves_carry_after_restart(tmp_path, monkeypatch, policy, parallel, persistent_latch):
    """An ordinary same-head follow-up changes obligations, not approval eligibility."""
    runner = FakeRunner(
        advance_pr_head_on_coder_followup=False,
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[_staged_review(reviewer="OpenAI Codex")],
        gemini_outputs=[
            _staged_review(reviewer="Google Gemini", state="blocking",
                           blocking_items=[{"text": "Repair the cleanup defect."}]),
            _staged_review(reviewer="Google Gemini",
                           dispositions=[{"item_id": "item-1", "disposition": "resolved"}]),
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy,
                         primary_reviewer="codex" if policy == "primary-then-panel" else None,
                         review_parallel=parallel, max_rounds=8)
    real_post = pr_loop.post_pr_comment

    def post(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if pr_loop._extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr",
        )[-1].metadata.role == "coder":
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(pr_loop, "post_pr_comment", post)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config)
    monkeypatch.setattr(pr_loop, "post_pr_comment", real_post)

    def change(payload):
        if payload.get("role") == "reviewer" and payload.get("state") == "approved":
            # A legacy approval has no scheduler obligation digest at all.
            return {k: v for k, v in payload.items() if not k.startswith("scheduler_")}
        if payload.get("role") == "coder":
            return {**payload, "scheduler_obligation_digest": "0" * 16,
                    "scheduler_force_full": persistent_latch,
                    "scheduler_force_full_source": "automatic" if persistent_latch else None}
        return None

    _rewrite_pr_metadata(runner, change)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner).count("codex") == 1
    assert _agent_sequence(runner).count("gemini") == 2
    assert any(item.scheduler_force_full for item in _posted_scheduler_metadata(runner))
    before = list(_agent_sequence(runner))
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert _agent_sequence(runner) == before
