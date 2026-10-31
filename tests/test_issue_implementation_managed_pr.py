"""Issue-implementation tests: managed PR creation and requirement blocking (#1543)."""

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import HumanReviewRequirement, IssueContext
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiCreationIntent,
    ManagedCiOutcome,
    UNPROTECTED_OVERRIDE_TRAILER,
    parse_managed_ci_override_record,
)
from coding_review_agent_loop.issue_pr_provenance import IssuePrProvenanceScope
from coding_review_agent_loop.issue_implementation import _advisory_issue_pr_provenance
from coding_review_agent_loop.protocol_markers import PR_BODY_SURFACE
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import FakeRunner, _managed_issue_handoff
from agent_loop_helpers import (
    make_config,
    structured_plan_review,
    structured_plan_state,
    structured_pr_review,
    structured_issue_implementation,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _blocked_issue_implementation(pr_number: int = 77) -> str:
    return structured_issue_implementation(
        summary=f"PR #{pr_number} was opened, but Requirement 1 is blocked.",
        pr_number=pr_number,
        human_requirement_dispositions=[
            {
                "requirement_id": "Requirement 1",
                "disposition": "blocked",
                "evidence": "The required integration is unavailable.",
            }
        ],
    )


def _issue_context_with_blocked_requirement() -> IssueContext:
    return IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue comment",
                author="maintainer",
                created_at="2026-01-01T00:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                body="Preserve the required integration.",
            ),
        ),
    )


def test_advisory_issue_provenance_skips_commit_scan_in_dry_run(tmp_path):
    runner = FakeRunner()
    config = make_config(tmp_path, dry_run=True)

    _advisory_issue_pr_provenance(
        runner,
        config=config,
        pr_number=77,
        expected_scope=IssuePrProvenanceScope("OWNER/REPO", 56, "direct"),
    )

    assert runner.pr_commit_calls == 0


def test_direct_issue_managed_draft_nonce_reaches_review_and_exact_head_merge(tmp_path, monkeypatch):
    nonce = "direct-managed-nonce"
    body = f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce=nonce,
    )
    handoff = _managed_issue_handoff(nonce=nonce)
    runner = FakeRunner(
        claude_outputs=[
            "Implemented the issue.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[structured_pr_review(state="approved", summary="Approved managed draft.")],
        pr_payload={
            "body": body,
            "headRefName": "agent-loop/managed-56",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
        },
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        pre_review_tests=True,
        test_command=("verify-managed-head",),
    )
    authentication_calls = []
    publication_calls = []
    readiness_heads = []
    dispatches = []
    prepared_heads = []
    merged_heads = []

    def authenticate(*_args, **kwargs):
        record = parse_managed_ci_override_record(
            kwargs["metadata"].body or "",
            surface=PR_BODY_SURFACE,
            schema="body",
            required=True,
            expected_nonce=nonce,
        )
        assert record is not None
        assert kwargs["intent"] == intent
        # Handoff authentication must precede every durable PR/issue write.
        assert runner.comments == []
        authentication_calls.append(kwargs)
        return handoff

    def revalidate(*_args, **kwargs):
        assert kwargs["config"].managed_ci_expected_override_nonce == nonce
        assert kwargs["handoff"] == handoff
        return handoff

    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_args, **_kwargs: intent)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", authenticate)
    monkeypatch.setattr(
        issue_implementation,
        "publish_issue_created_authorization",
        lambda *_args, **kwargs: publication_calls.append(kwargs) or handoff,
    )
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", revalidate)
    monkeypatch.setattr(
        pr_loop,
        "activate_managed_ci",
        lambda *_args, **_kwargs: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(pr_loop, "managed_label_present", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        pr_loop,
        "publish_round_readiness",
        lambda *_args, **kwargs: readiness_heads.append(kwargs["head_sha"]),
    )
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *_args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *_args, **_kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(
        pr_loop,
        "prepare_v2_merge",
        lambda *_args, **kwargs: prepared_heads.append(kwargs["expected_head_sha"]),
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *_args, **kwargs: merged_heads.append(kwargs["expected_head_sha"]),
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(authentication_calls) == 1
    assert len(publication_calls) == 1
    assert ["verify-managed-head"] in [command for command, _cwd in runner.commands]
    assert readiness_heads == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert [dispatch["expected_head_sha"] for dispatch in dispatches] == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert prepared_heads == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert merged_heads == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]


def test_direct_issue_conflict_posts_once_and_stops_before_pr_handoff_gates(tmp_path, monkeypatch):
    runner = FakeRunner(claude_outputs=[_blocked_issue_implementation()])
    config = make_config(tmp_path)
    issue_context = _issue_context_with_blocked_requirement()
    gate_calls = []

    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "validate_open_issue", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", lambda *args, **kwargs: issue_context)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "prepare_agent_memory", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *args, **kwargs: None)

    def unexpected_gate(name):
        def _fail(*args, **kwargs):
            gate_calls.append(name)
            raise AssertionError(f"{name} must not run after an implementation conflict")

        return _fail

    for name, owners in (
        ("validate_open_pr", (response_validation, pr_loop, issue_implementation, issue_loop)),
        ("get_observed_pr_review_context", (pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
        ("post_issue_pr_handoff_comment", (pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
        ("run_pr_loop", (pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
    ):
        patch_attr_in(monkeypatch, owners, name, unexpected_gate(name))

    with pytest.raises(AgentLoopError, match="not accepted for handoff"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert gate_calls == []
    assert len(runner.comments) == 1
    assert runner.comments[0].count("Rejected for handoff: reported PR #77") == 1


def test_approved_plan_implementation_conflict_posts_once_and_stops_before_pr_gates(
    tmp_path, monkeypatch
):
    runner = FakeRunner(claude_outputs=[_blocked_issue_implementation()])
    config = make_config(tmp_path)
    issue_context = _issue_context_with_blocked_requirement()
    gate_calls = []

    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *args, **kwargs: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *args, **kwargs: None)

    def unexpected_gate(name):
        def _fail(*args, **kwargs):
            gate_calls.append(name)
            raise AssertionError(f"{name} must not run after an implementation conflict")

        return _fail

    for name, owners in (
        ("validate_open_pr", (response_validation, pr_loop, issue_implementation, issue_loop)),
        ("get_observed_pr_review_context", (pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
        ("post_issue_pr_handoff_comment", (pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
        ("run_pr_loop", (pr_loop, issue_implementation, plan_first_loop_module, issue_loop)),
    ):
        patch_attr_in(monkeypatch, owners, name, unexpected_gate(name))

    with pytest.raises(AgentLoopError, match="not accepted for handoff"):
        issue_implementation._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Approved implementation plan.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    assert gate_calls == []
    assert len(runner.comments) == 1
    assert runner.comments[0].count("Rejected for handoff: reported PR #77") == 1


def test_approved_plan_non_managed_rejects_forged_pr_body_before_handoff(
    tmp_path, monkeypatch,
):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\nTests: python3 -m pytest tests/test_orchestrator_issue.py\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        pr_payload={"body": f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce=forged"},
    )
    config = make_config(tmp_path)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None)

    with pytest.raises(
        AgentLoopError, match="forged reserved protocol record syntax"
    ) as excinfo:
        issue_implementation._implement_approved_issue(
            runner, issue_number=56, approved_plan="Approved implementation plan.",
            config=config, memory=None,
            issue_context=IssueContext(
                number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
                url="https://github.test/issues/56", comments=(), human_requirements=(),
            ),
            coder_session_id=None,
            usage_context=validated_agent_module._new_usage_context(config),
        )

    # The diagnostic names the surface that carried the span, not just the token.
    assert "pull-request #77 body" in str(excinfo.value)
    assert runner.comments == []
    assert not any(command[:1] == ["codex"] for command, _cwd in runner.commands)


def test_approved_plan_non_managed_allows_pr_body_naming_reserved_token(
    tmp_path, monkeypatch,
):
    """Issue #891: a PR body that names a record must not stop the run.

    The token carries no authority here; the handoff gate must let the run
    continue instead of refusing before any review work happens.
    """
    runner = FakeRunner(
        claude_outputs=[
            "Implemented.\nTests: python3 -m pytest tests/test_orchestrator_issue.py\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        pr_payload={
            "body": "Fixes #56\n\nThis PR renames the AGENT_PLAN_APPROVED_FOLLOWUPS record label."
        },
    )
    config = make_config(tmp_path)
    patch_attr_in(monkeypatch, (execution_policy, issue_implementation, plan_first_loop_module, issue_loop), "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_a, **_k: None)
    reviewed = []
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", lambda *_a, **kwargs: reviewed.append(kwargs["pr_number"]) or 0
    )

    assert issue_implementation._implement_approved_issue(
        runner, issue_number=56, approved_plan="Approved implementation plan.",
        config=config, memory=None,
        issue_context=IssueContext(
            number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
            url="https://github.test/issues/56", comments=(), human_requirements=(),
        ),
        coder_session_id=None,
        usage_context=validated_agent_module._new_usage_context(config),
    ) == 0

    assert reviewed == [77]


def test_plan_first_issue_managed_draft_nonce_is_authenticated_before_pr_review(tmp_path, monkeypatch):
    nonce = "plan-managed-nonce"
    body = f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}"
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce=nonce,
    )
    handoff = _managed_issue_handoff(nonce=nonce)
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Implement the managed draft."),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[structured_plan_review(state="approved", summary="Plan approved.")],
        pr_payload={
            "body": body,
            "headRefName": "agent-loop/managed-56",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
        },
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    reviewed = []

    def authenticate(*_args, **kwargs):
        record = parse_managed_ci_override_record(
            kwargs["metadata"].body or "",
            surface=PR_BODY_SURFACE,
            schema="body",
            required=True,
            expected_nonce=nonce,
        )
        assert record is not None
        assert kwargs["intent"] == intent
        return handoff

    def review_entry(*_args, **kwargs):
        assert kwargs["config"].managed_ci_expected_override_nonce == nonce
        assert kwargs["managed_ci_handoff"] == handoff
        reviewed.append(kwargs)
        return 0

    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "preflight_managed_ci_creation", lambda *_args, **_kwargs: intent)
    patch_attr_in(monkeypatch, (issue_implementation, issue_loop), "authenticate_issue_created_handoff", authenticate)
    monkeypatch.setattr(
        issue_implementation,
        "publish_issue_created_authorization",
        lambda *_args, **_kwargs: handoff,
    )
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", review_entry)

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

    assert len(reviewed) == 1


@pytest.mark.parametrize('approved', [False, True])
def test_new_managed_pr_observation_precedes_authorization(tmp_path, monkeypatch, approved):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    from agent_loop_helpers import structured_issue_implementation
    from coding_review_agent_loop.github import IssueContext
    nonce = 'observation-order'
    intent = ManagedCiCreationIntent(branch='agent-loop/managed-56', trusted_actor='agent-loop', protection_mode='voluntary', audit_nonce=nonce)
    runner = FakeRunner(claude_outputs=[structured_issue_implementation(pr_number=77)],
        pr_payload={'body': f'Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} nonce={nonce}', 'headRefName': intent.branch})
    runner.branch_tip = 'b' * 40
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor='agent-loop', allow_unprotected_managed_ci=True)
    calls = []
    for module in (issue_loop, issue_implementation):
        monkeypatch.setattr(module, 'preflight_managed_ci_creation', lambda *a, **k: intent)
        monkeypatch.setattr(module, 'authenticate_issue_created_handoff', lambda *a, **k: calls.append('authorization'))
    with pytest.raises(PrHeadObservationError):
        if approved:
            issue = IssueContext(number=56, repo='OWNER/REPO', title='Issue', body='Issue', url='https://github.com/OWNER/REPO/issues/56', comments=())
            issue_implementation._implement_approved_issue(runner, issue_number=56, approved_plan='Approved plan.',
                config=config, memory=None, issue_context=issue, coder_session_id=None,
                usage_context=validated_agent_module._new_usage_context(config))
        else:
            run_issue_loop(runner, issue_number=56, config=config)
    assert calls == []
    assert not runner.pr_payload.get('comments')
