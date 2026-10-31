"""PR-loop tests for managed_ci: exact-head qualification in the PR loop (#1542)."""

import dataclasses

import pytest

# A private response root per test: the shared $TMPDIR one races across
# xdist workers (#1653 CI).
pytestmark = pytest.mark.usefixtures("private_response_tempdir")

from coding_review_agent_loop.ci_health import CiInfrastructureStall, StalledCheck
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.github import (
    PullRequestCheck,
    PullRequestMergeability,
)
from coding_review_agent_loop.managed_ci import (
    ManagedCiContract,
    ManagedCiOutcome,
)
from coding_review_agent_loop.round_state import (
    QualificationCheckpoint,
    PostedRoundMetadata,
    _attach_round_metadata,
)
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
)
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.triage as triage
import coding_review_agent_loop.validated_agent as validated_agent
from agent_loop_helpers import patch_attr_in
from pr_loop_test_helpers import (
    _watch_check_board,
)
import json
from types import SimpleNamespace

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.managed_ci import (
    ManagedCiIssueAuthorization,
    UNPROTECTED_OVERRIDE_TRAILER,
    format_issue_created_authorization_comment,
)
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.execution_policy as execution_policy
from issue_loop_test_helpers import _IssueRecoveryWorkflowRunner
from coding_review_agent_loop.runner import CommandResult
from agent_loop_helpers import FakeRunner as _FakeRunner
import coding_review_agent_loop.managed_ci as _m1047_managed_ci  # noqa: E402
from coding_review_agent_loop.config import github_bootstrap_cwd as _m1047_bootstrap_cwd  # noqa: E402
from coding_review_agent_loop.pr_loop import run_pr_loop as _m1047_run_pr_loop  # noqa: E402


def test_auto_merge_supported_repo_dispatches_and_merges_exact_approved_head(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(tmp_path, watch_pending_ci=True, auto_merge=True)
    monkeypatch.setattr(
        pr_loop,
        "activate_managed_ci",
        lambda *args, **kwargs: ManagedCiContract(),
    )
    dispatches = []
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    merges = []
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: merges.append(kwargs),
    )
    monkeypatch.setattr(
        pr_loop,
        "watch_pr_checks",
        lambda *args, **kwargs: pytest.fail("managed CI must bypass the ordinary watcher"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert dispatches[0]["expected_head_sha"] == "6367c48dd193d56ea7b0baad25b19455e529f5ee"
    assert merges == [{"expected_head_sha": "6367c48dd193d56ea7b0baad25b19455e529f5ee"}]


def test_managed_ci_failure_routes_back_to_coder_and_uses_failure_extension(
    tmp_path, monkeypatch
):
    failed_check = PullRequestCheck(
        name="final-ci/exact-head",
        kind="check_run",
        status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/555",
    )
    failed_checks = _watch_check_board("failing", failing=(failed_check,))
    outcomes = iter(
        [
            ManagedCiOutcome(status="failed", checks=failed_checks, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
            ManagedCiOutcome(status="passed", head_sha="525753a4b04bd57d874211ba1fade250c3ff6d82"),
        ]
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved."),
            structured_pr_review(
                state="approved",
                summary="Approved after managed CI fix.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking",
                summary="Fixed managed CI.",
                addressed_items=["item-1"],
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification", lambda *args, **kwargs: next(outcomes)
    )
    merges = []
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "GitHub managed exact-head CI unresolved blocking item [item-1]" in coder_prompt
    assert "final-ci/exact-head (failure)" in coder_prompt
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert merges == [{"expected_head_sha": "525753a4b04bd57d874211ba1fade250c3ff6d82"}]


def test_managed_qualification_resume_attaches_without_redispatch(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path, managed_ci=True, auto_merge=True, max_rounds=1)
    invocation = pr_loop.resolve_invocation(
        config, provider="codex", role="reviewer"
    )
    acquisition_contract = {
        "Codex": (invocation.configured_model, invocation.resolved_effort, "codex")
    }
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed CI failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        base_branch="main",
        requirements_digest=pr_loop_support._qualification_digest(tuple()),
        acquisition_digest=pr_loop_support._qualification_digest(acquisition_contract),
        approval_digest=pr_loop_support._qualification_digest(("Codex",)),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id="123/1",
        allowed_rounds=1,
    )
    reviewer_comment = _attach_round_metadata(
        "Approved the repaired head.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(dataclasses.replace(item, lifecycle="awaiting_current_head_review"),),
            state="approved",
            architecture_contract_version=1,
        ),
    )
    checkpoint_comment = _attach_round_metadata(
        "Qualification was dispatched.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
            architecture_contract_version=1,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": reviewer_comment},
                {"author": {"login": "coding-review-agent-loop"}, "body": checkpoint_comment},
            ]
        },
    )
    contract = ManagedCiContract(
        protocol_version=2, attached_run_id=123, run_attempt=1, expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    dispatches = []
    merges = []
    waited = []
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *a, **k: dispatches.append(k),
    )

    def wait(*a, **k):
        c = k["contract"]
        waited.append((c.attached_run_id, c.run_attempt, c.expected_head_sha))
        return ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: merges.append(k))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert dispatches == []
    assert waited == [(123, 1, "6367c48dd193d56ea7b0baad25b19455e529f5ee")]
    assert merges == [{"expected_head_sha": "6367c48dd193d56ea7b0baad25b19455e529f5ee"}]
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def _managed_qualifying_resume(tmp_path, *, managed=True, attempt_id="123/1", **config_kwargs):
    """A fresh-process resume after a stop while managed qualification was in flight (#1177)."""
    config = make_config(
        tmp_path, managed_ci=managed, auto_merge=True, max_rounds=1, **config_kwargs
    )
    invocation = pr_loop.resolve_invocation(config, provider="codex", role="reviewer")
    acquisition_contract = {
        "Codex": (invocation.configured_model, invocation.resolved_effort, "codex")
    }
    item = UnresolvedReviewItem(
        item_id="item-1", reviewer="GitHub managed exact-head CI", source_round=1,
        text="Managed CI failed on the previous head.", status="blocking",
        source_status="blocking", authority="machine",
        obligation_kind="managed-exact-head-ci", lifecycle="qualifying",
        failed_head_sha="old-head", candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        base_branch="main",
        requirements_digest=pr_loop_support._qualification_digest(tuple()),
        acquisition_digest=pr_loop_support._qualification_digest(acquisition_contract),
        approval_digest=pr_loop_support._qualification_digest(("Codex",)),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id=attempt_id,
        allowed_rounds=1,
    )
    reviewer_comment = _attach_round_metadata(
        "Approved the repaired head.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="pr", role="reviewer", agent="Codex", round_number=1, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(dataclasses.replace(item, lifecycle="awaiting_current_head_review"),),
            state="approved", architecture_contract_version=1,
        ),
    )
    checkpoint_comment = _attach_round_metadata(
        "Qualification was dispatched.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(item,), state="blocking", phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint, architecture_contract_version=1,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": reviewer_comment},
                {"author": {"login": "coding-review-agent-loop"}, "body": checkpoint_comment},
            ]
        },
    )
    return config, runner


def _posted_attempt_checkpoints(runner):
    return [
        line
        for comment in runner.pr_payload["comments"]
        for line in comment["body"].splitlines()
        if "qualification checkpoint: attached attempt" in line
    ]


def test_managed_qualification_fresh_process_resume_dispatches_a_fresh_attempt(
    tmp_path, monkeypatch, capsys
):
    """#1177: a fresh unattached contract resumes a qualifying checkpoint without reviewers."""
    config, runner = _managed_qualifying_resume(tmp_path, quiet=False)
    # What real activation returns: a fresh generation, no attachment, no head binding.
    contract = ManagedCiContract(protocol_version=2, intent_generation="fresh-gen")
    dispatches = []
    merges = []
    waited = []
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)

    def dispatch(*a, **k):
        c = k["contract"]
        dispatches.append((k["expected_head_sha"], c.intent_generation))
        c.attached_run_id, c.run_attempt, c.expected_head_sha = 999, 1, k["expected_head_sha"]

    def wait(*a, **k):
        c = k["contract"]
        waited.append((c.attached_run_id, c.run_attempt, c.expected_head_sha))
        return ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee")

    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", dispatch)
    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: merges.append(k))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert dispatches == [("6367c48dd193d56ea7b0baad25b19455e529f5ee", "fresh-gen")]
    assert waited == [(999, 1, "6367c48dd193d56ea7b0baad25b19455e529f5ee")]
    assert merges == [{"expected_head_sha": "6367c48dd193d56ea7b0baad25b19455e529f5ee"}]
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    attempts = _posted_attempt_checkpoints(runner)
    assert attempts == ["PR #77 qualification checkpoint: attached attempt 999/1."]
    err = capsys.readouterr().err
    assert "without its prior attempt 123/1" in err
    assert "without redispatching completed reviewer work" in err
    assert "force_full=True" not in err


@pytest.mark.parametrize(
    "contract",
    [
        pytest.param(
            ManagedCiContract(
                protocol_version=2, attached_run_id=456, run_attempt=1,
                expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            ),
            id="mismatched-same-head-attachment",
        ),
        pytest.param(
            ManagedCiContract(protocol_version=2, dispatch_issued=True),
            id="dispatch-issued-without-attachment",
        ),
        pytest.param(
            ManagedCiContract(protocol_version=2, nonce="owned-nonce"),
            id="nonce-without-attachment",
        ),
        pytest.param(
            ManagedCiContract(protocol_version=2, expected_head_sha="old-head"),
            id="unattached-stale-head-contract",
        ),
        pytest.param(None, id="managed-contract-missing"),
    ],
)
def test_managed_qualifying_checkpoint_without_fresh_attempt_path_forces_review(
    tmp_path, monkeypatch, contract
):
    """#1177: only an unattached, unbound, unowned contract may skip the board."""
    # The selective scheduler exposes the automatic full-board fallback.
    config, runner = _managed_qualifying_resume(
        tmp_path, managed=contract is not None,
        reviewer=("codex",), pr_review_policy="selective-intermediate",
    )
    if contract is not None:
        monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
        monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)
    force_full_values = []
    reviewer_calls = []
    dispatches = []
    downgrades = []
    real_select = pr_loop.select_reviewers
    real_run_validated_agent = validated_agent._run_validated_agent
    import coding_review_agent_loop.pr_loop as pr_loop_module

    real_helper = pr_loop_module._managed_checkpoint_needs_fresh_attempt

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return real_select(snapshot, *args, **kwargs)

    def stop_at_reviewer(*args, **kwargs):
        if kwargs.get("role") == "reviewer":
            reviewer_calls.append(kwargs)
            raise AgentLoopError("review board invoked before qualification")
        return real_run_validated_agent(*args, **kwargs)

    def helper(*args, **kwargs):
        result = real_helper(*args, **kwargs)
        downgrades.append(result)
        return result

    monkeypatch.setattr(pr_loop, "select_reviewers", capture_force_full)
    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", stop_at_reviewer)
    monkeypatch.setattr(pr_loop_module, "_managed_checkpoint_needs_fresh_attempt", helper)
    monkeypatch.setattr(
        pr_loop, "dispatch_final_qualification", lambda *a, **k: dispatches.append(k)
    )

    with pytest.raises(AgentLoopError, match="review board invoked"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert downgrades and not any(downgrades)
    assert reviewer_calls
    assert force_full_values == [True]
    assert dispatches == []
    assert _posted_attempt_checkpoints(runner) == []


def test_managed_checkpoint_needs_fresh_attempt_helper_covers_each_conjunct():
    """#1177: the fresh-process downgrade predicate, one conjunct at a time."""
    from coding_review_agent_loop.pr_loop_support import (
        _managed_checkpoint_needs_fresh_attempt as needs_fresh,
    )

    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity="managed-exact-head-ci:item-1",
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        qualification_attempt_id="555/1",
    )
    fresh = ManagedCiContract(protocol_version=2, intent_generation="fresh-gen")

    assert needs_fresh(checkpoint, fresh, "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert needs_fresh(checkpoint, dataclasses.replace(fresh, expected_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"), "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(checkpoint, dataclasses.replace(fresh, expected_head_sha="old-head"), "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(None, fresh, "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(QualificationCheckpoint.invalid("stale"), fresh, "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(
        dataclasses.replace(checkpoint, lifecycle="qualification_ready"), fresh, "6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    assert not needs_fresh(
        dataclasses.replace(
            checkpoint, obligation_kind="github-pr-checks",
            obligation_identity="github-pr-checks:item-1",
        ),
        fresh,
        "6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    assert not needs_fresh(checkpoint, fresh, "5e6741348108b11e3f3ef936c335b1967d17ac6c")
    assert not needs_fresh(checkpoint, fresh, None)
    assert not needs_fresh(checkpoint, fresh, "")
    assert not needs_fresh(checkpoint, None, "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(checkpoint, dataclasses.replace(fresh, protocol_version=1), "6367c48dd193d56ea7b0baad25b19455e529f5ee")
    assert not needs_fresh(
        checkpoint, dataclasses.replace(fresh, activation_path="ordinary_fallback"), "6367c48dd193d56ea7b0baad25b19455e529f5ee"
    )
    for owned in (
        {"attached_run_id": 555},
        {"run_attempt": 1},
        {"nonce": "owned-nonce"},
        {"dispatch_issued": True},
    ):
        assert not needs_fresh(checkpoint, dataclasses.replace(fresh, **owned), "6367c48dd193d56ea7b0baad25b19455e529f5ee"), owned


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("approval_digest", None),
        ("requirements_digest", None),
        ("acquisition_digest", None),
        ("scheduler_digest", None),
        ("qualification_attempt_id", None),
        ("approval_digest", "stale-approval"),
        ("requirements_digest", "stale-requirements"),
        ("acquisition_digest", "stale-acquisition"),
        ("scheduler_digest", "stale-scheduler"),
        ("qualification_attempt_id", "stale-attempt"),
    ],
)
def test_resume_qualification_identity_gap_forces_review_before_qualification(
    tmp_path, monkeypatch, field, value
):
    config = make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        managed_ci=False,
        auto_merge=True,
        max_rounds=1,
    )
    invocation = pr_loop.resolve_invocation(
        config, provider="codex", role="reviewer"
    )
    item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub PR checks",
        source_round=1,
        text="GitHub PR checks failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="github-pr-checks",
        lifecycle="qualification_ready",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        obligation_identity="github-pr-checks:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="github-pr-checks",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        base_branch="main",
        requirements_digest=pr_loop_support._qualification_digest(tuple()),
        acquisition_digest=pr_loop_support._qualification_digest(
            {"Codex": (invocation.configured_model, invocation.resolved_effort, "codex")}
        ),
        approval_digest=pr_loop_support._qualification_digest(("Codex",)),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id="github-checks:6367c48dd193d56ea7b0baad25b19455e529f5ee",
        allowed_rounds=1,
    )
    checkpoint = dataclasses.replace(checkpoint, **{field: value})
    summary = _attach_round_metadata(
        "Persisted qualifying checkpoint without reviewer records.",
        PostedRoundMetadata(
            flow="pr",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            prior_items=(item,),
            state="blocking",
            phase="qualification-checkpoint",
            qualification_checkpoint=checkpoint,
        ),
    )
    runner = FakeRunner(
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": summary}
            ]
        }
    )
    force_full_values = []
    reviewer_calls = []
    original_select_reviewers = pr_loop.select_reviewers
    original_run_validated_agent = validated_agent._run_validated_agent

    def capture_force_full(snapshot, *args, **kwargs):
        force_full_values.append(snapshot.force_full)
        return original_select_reviewers(snapshot, *args, **kwargs)

    def stop_at_reviewer(*args, **kwargs):
        if kwargs.get("role") == "reviewer":
            reviewer_calls.append(kwargs)
            raise AgentLoopError("review board invoked before qualification")
        return original_run_validated_agent(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "select_reviewers", capture_force_full)
    patch_attr_in(monkeypatch, (
        validated_agent, discuss_loop, pr_loop, issue_implementation, plan_first_loop, triage,
        issue_loop,
    ), "_run_validated_agent", stop_at_reviewer)
    dispatches = []
    monkeypatch.setattr(
        pr_loop,
        "dispatch_final_qualification",
        lambda *args, **kwargs: dispatches.append(kwargs),
    )

    with pytest.raises(AgentLoopError, match="review board invoked"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert reviewer_calls
    assert force_full_values == [True]
    assert dispatches == []
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_success_publishes_result_without_merge_even_with_pending_intermediate_ci(
    tmp_path, monkeypatch, capsys
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "test", "status": "completed", "conclusion": "success"},
                {"name": "lint", "status": "in_progress"},
            ]
        },
        pr_status_payload={"state": "pending", "statuses": []},
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    contract = ManagedCiContract(protocol_version=2, protection_mode="strict")
    published = []

    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: contract)
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(
        pr_loop,
        "publish_manual_v2_qualification",
        lambda *args, **kwargs: published.append(kwargs["expected_head_sha"]) or "6367c48dd193d56ea7b0baad25b19455e529f5ee",
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("manual managed-CI success must not merge"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert published == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    out = capsys.readouterr().out
    assert "approved and qualified; manual merge required" in out
    assert "line containing exactly `-- Human Reviewer`, then rerun `agent-loop pr 77`" in out
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_issue_created_label_loss_cancels_qualification(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    contract = ManagedCiContract(protocol_version=2, issue_created_pr=True)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: contract)

    with pytest.raises(AgentLoopError, match="lost its authenticated"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_manual_adopted_provenance_loss_cancels_qualification(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(
        tmp_path,
        managed_ci=True,
        managed_ci_adopt_existing_pr=True,
        managed_ci_trusted_actor="agent-loop",
        max_rounds=1,
    )
    contract = ManagedCiContract(
        protocol_version=2,
        base_ref="main",
        adopted_existing_pr=True,
    )
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *args, **kwargs: contract)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *args, **kwargs: False)

    with pytest.raises(AgentLoopError, match="adoption provenance changed"):
        run_pr_loop(runner, pr_number=77, config=config)

    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_managed_ci_head_change_restarts_review_without_coder(tmp_path, monkeypatch):
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Old head approved."),
            structured_pr_review(state="approved", summary="New head approved."),
        ]
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    call_count = 0

    def wait_for_managed_ci(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            runner.pr_payload["headRefOid"] = "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"
            return ManagedCiOutcome(status="head_changed", head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6")
        return ManagedCiOutcome(status="passed", head_sha="4ebf8a1679f50e9e49fc965072f996bfe0dc18c6")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait_for_managed_ci)
    merges = []
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert call_count == 2
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert merges == [{"expected_head_sha": "4ebf8a1679f50e9e49fc965072f996bfe0dc18c6"}]


def test_managed_ci_merge_conflict_routes_through_reconciliation(tmp_path, monkeypatch):
    conflict = PullRequestMergeability(
        state="conflicted",
        mergeable_raw="CONFLICTING",
        merge_state_raw="DIRTY",
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        base_branch="main",
    )
    outcomes = iter(
        [
            ManagedCiOutcome(
                status="merge_conflict",
                mergeability=conflict,
                head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            ),
            ManagedCiOutcome(status="passed", head_sha="525753a4b04bd57d874211ba1fade250c3ff6d82"),
        ]
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved before conflict."),
            structured_pr_review(state="approved", summary="Approved after rebase."),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Rebased and resolved the conflict."
            )
        ],
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=2)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop, "wait_for_final_qualification", lambda *args, **kwargs: next(outcomes)
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "merge conflict with its base branch `main`" in coder_prompt
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2


def test_managed_ci_infrastructure_stall_stops_cleanly(tmp_path, monkeypatch):
    stalled_check = StalledCheck(
        name="final-ci/exact-head",
        kind="check_run",
        reason="queued_too_long",
        check_id=99,
        run_id="555",
        url="https://github.com/OWNER/REPO/actions/runs/555",
        age_seconds=1800,
    )
    stall = CiInfrastructureStall(checks=(stalled_check,))
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(tmp_path, auto_merge=True, max_rounds=1)
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(
            status="infrastructure_stall", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", stall=stall
        ),
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("infrastructure stall must not merge"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert any(
        comment.startswith("External GitHub Actions infrastructure is blocking PR #77.")
        for comment in runner.comments
    )
    assert runner.comments[-1].startswith(
        "PR #77 is blocked on external GitHub Actions infrastructure, not a code defect."
    )


@pytest.mark.parametrize(
    ("pre_review_tests", "test_command"),
    [(False, ("python", "-m", "pytest", "focused")), (True, None)],
)
def test_managed_ci_does_not_publish_readiness_without_configured_pre_review_gate(
    tmp_path, monkeypatch, pre_review_tests, test_command
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        pre_review_tests=pre_review_tests,
        test_command=test_command,
    )
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    monkeypatch.setattr(
        pr_loop,
        "publish_round_readiness",
        lambda *args, **kwargs: pytest.fail("readiness must not be published"),
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        pre_review_test_pending=True,
    ) == 0


def test_managed_ci_publishes_readiness_after_configured_pre_review_gate(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")]
    )
    config = make_config(
        tmp_path,
        auto_merge=True,
        pre_review_tests=True,
        test_command=("verify-managed-head",),
    )
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", lambda *args, **kwargs: ManagedCiContract()
    )
    published_heads = []
    monkeypatch.setattr(
        pr_loop,
        "publish_round_readiness",
        lambda *args, **kwargs: published_heads.append(kwargs["head_sha"]),
    )
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        pr_loop,
        "wait_for_final_qualification",
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee"),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        pre_review_test_pending=True,
    ) == 0

    assert published_heads == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert ["verify-managed-head"] in [cmd for cmd, _cwd in runner.commands]


def _stale_attachment_stubs(monkeypatch, *, stale_contract, outcomes):
    """Managed-CI stubs that record dispatches, waits and the attachment each wait saw."""
    state = {"contract": stale_contract, "dispatches": [], "waited": [], "merges": [], "fresh": 0}
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: state["contract"])
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *a, **k: True)

    def dispatch(*args, **kwargs):
        state["dispatches"].append(kwargs["expected_head_sha"])
        state["fresh"] += 1
        contract = kwargs["contract"]
        contract.attached_run_id = 999 + state["fresh"] - 1
        contract.run_attempt = 1
        contract.expected_head_sha = kwargs["expected_head_sha"]
        contract.nonce = f"fresh-nonce-{state['fresh']}"

    def wait(*args, **kwargs):
        contract = kwargs["contract"]
        state["waited"].append((contract.attached_run_id, contract.expected_head_sha))
        return outcomes(contract, kwargs["metadata"])

    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", dispatch)
    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: state["merges"].append(k))
    return state


def test_stale_older_head_managed_attachment_is_released_and_fresh_attempt_dispatched(
    tmp_path, monkeypatch, capsys
):
    """#1168: an attachment made for an older head is never attached or checkpointed."""
    seeded = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="GitHub managed exact-head CI",
        source_round=1,
        text="Managed CI failed on the previous head.",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind="managed-exact-head-ci",
        lifecycle="awaiting_current_head_review",
        failed_head_sha="old-head",
        candidate_head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    coder_output = structured_coder_followup(
        summary="The failed managed-CI head was repaired.", addressed_items=[seeded.item_id]
    )
    prior_round = _attach_round_metadata(
        coder_output,
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee", prior_items=(seeded,), state="blocking",
            raw_structured_coder_response=coder_output,
        ),
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Approved the repaired head.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
        pr_payload={
            "comments": [{"author": {"login": "coding-review-agent-loop"}, "body": prior_round}]
        },
    )
    stale = ManagedCiContract(
        protocol_version=2, attached_run_id=37061662125, run_attempt=1,
        expected_head_sha="old-head", nonce="old-nonce",
    )
    state = _stale_attachment_stubs(
        monkeypatch,
        stale_contract=stale,
        outcomes=lambda c, m: ManagedCiOutcome(status="passed", head_sha=m.head_sha),
    )
    config = make_config(tmp_path, managed_ci=True, auto_merge=True, max_rounds=2, quiet=False)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    bodies = [str(c) for c in runner.comments]
    assert any(c[:2] == ["codex", "exec"] for c, _cwd in runner.commands)
    assert any("authoritative exact-head validation is in progress" in b for b in bodies)
    assert state["dispatches"] == ["6367c48dd193d56ea7b0baad25b19455e529f5ee"]
    assert not any("37061662125" in b for b in bodies)
    assert [b for b in bodies if "attached attempt" in b and "999/1" in b]
    assert len([b for b in bodies if "attached attempt" in b]) == 1
    assert state["waited"] == [(999, "6367c48dd193d56ea7b0baad25b19455e529f5ee")]
    assert "resuming attached qualification attempt 37061662125" not in capsys.readouterr().err
    assert state["merges"] == [{"expected_head_sha": "6367c48dd193d56ea7b0baad25b19455e529f5ee"}]


def test_pr1166_stale_attempt_failure_does_not_end_run_or_partial_the_approved_round(
    tmp_path, monkeypatch, capsys
):
    """#1168: attempt A (head-1) fails late; round N+1 approves head-2 and uses a fresh attempt B."""
    failed_check = PullRequestCheck(
        name="final-ci/exact-head", kind="check_run", status="failure",
        url="https://github.com/OWNER/REPO/actions/runs/37061662125",
    )
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(state="approved", summary="Approved head-1."),
            structured_pr_review(
                state="approved",
                summary="Approved head-2.",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        gemini_outputs=[
            structured_pr_review(state="approved", summary="Approved head-1.", reviewer="Google Gemini"),
            structured_pr_review(
                state="approved",
                summary="Approved head-2.",
                reviewer="Google Gemini",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        claude_outputs=[
            structured_coder_followup(
                state="blocking", summary="Fixed managed CI.", addressed_items=["item-1"]
            )
        ],
    )

    def outcomes(contract, metadata):
        if contract.attached_run_id == 999:
            # Attempt A: bound to the first head and failing late.
            return ManagedCiOutcome(
                status="failed", head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
                checks=_watch_check_board("failing", failing=(failed_check,)),
            )
        return ManagedCiOutcome(status="passed", head_sha=metadata.head_sha)

    state = _stale_attachment_stubs(
        monkeypatch,
        stale_contract=ManagedCiContract(protocol_version=2),
        outcomes=outcomes,
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), managed_ci=True, auto_merge=True,
        max_rounds=1, quiet=False,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    bodies = [str(c) for c in runner.comments]
    # A was dispatched for 6367c48dd193d56ea7b0baad25b19455e529f5ee; B for the coder's new head. A's failure is never
    # reported as B's verdict, and the run does not stop on it.
    assert state["dispatches"] == ["6367c48dd193d56ea7b0baad25b19455e529f5ee", "525753a4b04bd57d874211ba1fade250c3ff6d82"]
    assert state["waited"] == [(999, "6367c48dd193d56ea7b0baad25b19455e529f5ee"), (1000, "525753a4b04bd57d874211ba1fade250c3ff6d82")]
    assert state["merges"] == [{"expected_head_sha": "525753a4b04bd57d874211ba1fade250c3ff6d82"}]
    attached = [b for b in bodies if "qualification checkpoint: attached attempt" in b]
    # Only the repair round carries a CI obligation, so only attempt B is checkpointed.
    assert [("999/1" in b, "1000/1" in b) for b in attached] == [(False, True)]
    assert not any("terminal state without publishing" in b for b in bodies)
    assert "resuming attached qualification attempt 999" not in capsys.readouterr().err

    # Resume over the posted history while the PR is still OPEN at the approved head.
    def reviewer_runs():
        return sum(c[:2] == ["codex", "exec"] or c[:1] == ["gemini"] for c, _cwd in runner.commands)

    reviewer_before = reviewer_runs()
    assert reviewer_before == 4  # both reviewers approved both heads in run 1
    resumed = ManagedCiContract(
        protocol_version=2, attached_run_id=1000, run_attempt=1,
        expected_head_sha="525753a4b04bd57d874211ba1fade250c3ff6d82", nonce="fresh-nonce-2",
    )
    state2 = _stale_attachment_stubs(
        monkeypatch, stale_contract=resumed,
        outcomes=lambda c, m: ManagedCiOutcome(status="passed", head_sha=m.head_sha),
    )
    runner.pr_payload["state"] = "OPEN"
    assert runner.pr_payload["headRefOid"] == "525753a4b04bd57d874211ba1fade250c3ff6d82"

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert reviewer_runs() == reviewer_before
    assert state2["dispatches"] == []
    assert state2["waited"] == [(1000, "525753a4b04bd57d874211ba1fade250c3ff6d82")]
    assert state2["merges"] == [{"expected_head_sha": "525753a4b04bd57d874211ba1fade250c3ff6d82"}]


def test_managed_pr_recovery_keeps_closing_reference_gate_without_pr_contract(
    tmp_path,
):
    body = (
        "Fixes #56\nCloses #999\n\n"
        f"{UNPROTECTED_OVERRIDE_TRAILER} nonce=opening-nonce"
    )
    authorization = ManagedCiIssueAuthorization(
        kind="creation",
        repository="OWNER/REPO",
        issue_number=56,
        pr_number=77,
        base_ref="main",
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        actor_login="agent-loop",
        actor_id=1,
        protection="voluntary",
        waiver="allow-unprotected-managed-ci",
        nonce="opening-nonce",
        label_event_id=101,
    )
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        authorization_comments=[{
            "id": 41,
            "user": {"login": "agent-loop", "id": 1},
            "body": str(format_issue_created_authorization_comment(authorization)),
        }],
    )
    runner.pr_payload["body"] = body
    runner.rest_pr["body"] = body
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_pr_mode=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        pre_review_tests=False,
        invocation_argv=(
            "agent-loop", "pr", "77", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )

    with pytest.raises(AgentLoopError, match="outside the expected contract"):
        pr_loop.run_pr_loop(
            runner,
            pr_number=77,
            config=config,
            workdirs_ready=True,
        )

    assert runner.comments == []
    assert runner.labels_posted is False
    assert not any(
        command[:3] == ["gh", "api", "--method"]
        and "issues/77/labels" in " ".join(command)
        and "POST" in command
        for command, _cwd in runner.commands
    )
    assert runner.dispatch_count == 0
    assert not any(command[:1] in (["claude"], ["codex"]) for command, _cwd in runner.commands)


# --- #1047: retained managed label after manual qualification -----------------


_M1047_DELETE = [
    "gh", "api", "--method", "DELETE",
    "repos/OWNER/REPO/issues/77/labels/agent-loop-managed",
]


class _M1047Stop(Exception):
    """Sentinel that ends a run once the path under test was reached."""


class _M1047LivePrRunner(_FakeRunner):
    """Serve the live REST PR, its label events and label writes for PR #77."""

    def __init__(
        self,
        *,
        draft=False,
        labels=("agent-loop-managed",),
        delete_returncode=0,
        events_fail_from=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.live_pr = {
            "state": "open",
            "draft": draft,
            "labels": [{"name": name} for name in labels],
            "head": {"sha": self.pr_payload.get("headRefOid"), "ref": self.pr_payload.get("headRefName"), "repo": {"full_name": "OWNER/REPO"}},
            "base": {"ref": "main", "repo": {"full_name": "OWNER/REPO"}},
        }
        self.events = [{
            "id": 101, "event": "labeled", "label": {"name": "agent-loop-managed"},
            "actor": {"login": "agent-loop", "id": 1},
        }]
        self.delete_returncode = delete_returncode
        self.events_fail_from = events_fail_from
        self.event_reads = 0

    def _run_locked(self, args, *, cwd, check, input_text=None):
        cmd = [str(arg) for arg in args]
        endpoint = next((part for part in cmd if part.startswith("repos/")), "")
        if cmd == ["gh", "api", "repos/OWNER/REPO/pulls/77"]:
            cmd, cwd_path = self._record_command(args, cwd)
            return CommandResult(cmd, cwd_path, json.dumps(self.live_pr), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/77/events?"):
            cmd, cwd_path = self._record_command(args, cwd)
            index = self.event_reads
            self.event_reads += 1
            if self.events_fail_from is not None and index >= self.events_fail_from:
                return CommandResult(cmd, cwd_path, "", "events unavailable", 1)
            return CommandResult(cmd, cwd_path, json.dumps(self.events), "", 0)
        if cmd[:5] == _M1047_DELETE:
            cmd, cwd_path = self._record_command(args, cwd)
            if self.delete_returncode:
                return CommandResult(cmd, cwd_path, "", "gh: Server Error (HTTP 500)", 1)
            self.live_pr["labels"] = [
                item for item in self.live_pr["labels"] if item["name"] != "agent-loop-managed"
            ]
            self.events.append({
                "id": 102, "event": "unlabeled", "label": {"name": "agent-loop-managed"},
                "actor": {"login": "agent-loop", "id": 1},
            })
            return CommandResult(cmd, cwd_path, "", "", 0)
        if cmd[:3] == ["gh", "pr", "ready"]:
            cmd, cwd_path = self._record_command(args, cwd)
            self.live_pr["draft"] = False
            return CommandResult(cmd, cwd_path, "", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    def managed_deletes(self):
        return [command for command, _cwd in self.commands if command[:5] == _M1047_DELETE]

    def has_managed_label(self):
        return any(item["name"] == "agent-loop-managed" for item in self.live_pr["labels"])


def _m1047_recorder(runner, seen, name):
    def record(*args, **kwargs):
        seen.append((name, runner.has_managed_label(), runner.live_pr["draft"]))
        raise _M1047Stop(name)
    return record


def _m1047_contract(origin):
    return ManagedCiContract(
        protocol_version=2,
        issue_created_pr=origin == "issue-created",
        origin=None if origin == "adopted" else origin,
        adopted_existing_pr=origin == "adopted",
        active_label_event_id=101,
        invocation_applied_label=True,
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        protection_mode="strict",
        base_ref="main",
    )


def _m1047_release_spy(monkeypatch):
    calls = []
    real = pr_loop.release_adopted_managed_ci

    def spy(*args, **kwargs):
        calls.append(kwargs.get("contract"))
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "release_adopted_managed_ci", spy)
    return calls


@pytest.mark.parametrize("path", ["activation", "revalidation", "recovery", "fresh", "source-managed"])
def test_m1047_entry_normalization_runs_before_every_managed_resume_path(
    tmp_path, monkeypatch, path,
):
    runner = _M1047LivePrRunner()
    overrides = {"managed_ci": True, "max_rounds": 1}
    if path == "fresh":
        overrides.update(managed_ci_fresh_authorization=True, managed_ci_issue_number=5)
    config = make_config(tmp_path, **overrides)
    seen = []
    kwargs = {}
    if path == "activation":
        monkeypatch.setattr(pr_loop, "recover_issue_created_handoff", lambda *a, **k: None)
        monkeypatch.setattr(
            pr_loop, "activate_managed_ci", _m1047_recorder(runner, seen, path)
        )
    elif path == "revalidation":
        kwargs["managed_ci_handoff"] = object()
        monkeypatch.setattr(
            pr_loop, "revalidate_issue_created_handoff",
            _m1047_recorder(runner, seen, path),
        )
    elif path == "recovery":
        monkeypatch.setattr(
            pr_loop, "recover_issue_created_handoff",
            _m1047_recorder(runner, seen, path),
        )
    elif path == "fresh":
        patch_attr_in(monkeypatch, (discuss_loop, pr_loop, issue_loop), "validate_open_issue", lambda *a, **k: None)
        patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
            lambda *a, **k: SimpleNamespace(number=5, comments=[], human_requirements=()),
        )
        monkeypatch.setattr(
            pr_loop, "authorize_fresh_issue_created_resume",
            _m1047_recorder(runner, seen, path),
        )
    else:
        monkeypatch.setattr(
            pr_loop, "recover_managed_pr_origin",
            lambda *a, **k: ("feature", "6367c48dd193d56ea7b0baad25b19455e529f5ee", "agent-loop/managed-77", None),
        )
        monkeypatch.setattr(pr_loop, "validate_managed_pr_body", lambda *a, **k: None)
        monkeypatch.setattr(
            pr_loop, "authenticate_source_managed_resume",
            _m1047_recorder(runner, seen, path),
        )

    with pytest.raises(_M1047Stop):
        _m1047_run_pr_loop(runner, pr_number=77, config=config, **kwargs)

    # The path saw ready/unlabeled, exactly as for a historical qualified PR.
    assert seen == [(path, False, False)]
    assert len(runner.managed_deletes()) == 1


def test_m1047_implicit_invocation_also_releases_retained_label(tmp_path, monkeypatch):
    runner = _M1047LivePrRunner()
    config = make_config(tmp_path, max_rounds=1)
    seen = []
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff", _m1047_recorder(runner, seen, "recovery"),
    )

    with pytest.raises(_M1047Stop):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert seen == [("recovery", False, False)]
    assert len(runner.managed_deletes()) == 1


@pytest.mark.parametrize(
    "state",
    [
        {"draft": True},
        {"draft": True, "labels": ()},
        {"labels": ()},
    ],
    ids=["draft-labeled", "draft-unlabeled", "ready-unlabeled"],
)
def test_m1047_entry_normalization_leaves_other_pr_states_untouched(tmp_path, monkeypatch, state):
    runner = _M1047LivePrRunner(**state)
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    seen = []
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff", _m1047_recorder(runner, seen, "recovery"),
    )

    with pytest.raises(_M1047Stop):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert runner.managed_deletes() == []
    assert seen == [("recovery", "labels" not in state, state.get("draft", False))]


def test_m1047_failed_automerge_ready_labeled_pr_is_normalized_on_retry(tmp_path, monkeypatch):
    qualified = _m1047_managed_ci.QUALIFIED_LABEL
    runner = _M1047LivePrRunner(labels=("agent-loop-managed", qualified))
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    seen = []
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff", _m1047_recorder(runner, seen, "recovery"),
    )

    with pytest.raises(_M1047Stop):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert seen == [("recovery", False, False)]
    assert runner.live_pr["labels"] == [{"name": qualified}]
    assert not any(command[:3] == ["gh", "pr", "merge"] for command, _cwd in runner.commands)


def test_m1047_entry_release_failure_stops_before_any_other_work(tmp_path, monkeypatch):
    runner = _M1047LivePrRunner(delete_returncode=1)
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    for name in (
        "recover_issue_created_handoff", "revalidate_issue_created_handoff",
        "authenticate_source_managed_resume", "authorize_fresh_issue_created_resume",
        "activate_managed_ci",
    ):
        monkeypatch.setattr(
            pr_loop, name,
            lambda *a, _name=name, **k: pytest.fail(f"{_name} ran after a failed entry release"),
        )

    with pytest.raises(AgentLoopError, match="Remove the label manually, then rerun"):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    writes = [
        command for command, _cwd in runner.commands
        if command[:1] in (["claude"], ["codex"])
        or command[:3] in (["gh", "pr", "ready"], ["gh", "pr", "merge"], ["gh", "pr", "comment"])
        or (command[:2] == ["gh", "api"] and any(m in command for m in ("POST", "PATCH", "DELETE")))
    ]
    assert writes == runner.managed_deletes()
    assert runner.live_pr["draft"] is False


@pytest.mark.parametrize("delete_returncode", [0, 1], ids=["released", "delete-fails"])
@pytest.mark.parametrize("labels", [("agent-loop-managed",), ()], ids=["labeled", "unlabeled"])
def test_m1047_entry_normalization_uses_bootstrap_cwd_without_coder_checkout(
    tmp_path, monkeypatch, delete_returncode, labels,
):
    config = make_config(tmp_path, create_dirs=False, managed_ci=True, max_rounds=1)
    config.codex_dir.mkdir(parents=True)
    assert not config.claude_dir.exists()
    bootstrap = _m1047_bootstrap_cwd(config)
    assert bootstrap == config.codex_dir
    runner = _M1047LivePrRunner(labels=labels, delete_returncode=delete_returncode)
    seen = []
    monkeypatch.setattr(
        pr_loop, "recover_issue_created_handoff", _m1047_recorder(runner, seen, "recovery"),
    )

    expected = (
        pytest.raises(AgentLoopError, match="Remove the label manually")
        if labels and delete_returncode
        else pytest.raises(_M1047Stop)
    )
    with expected:
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    normalization = [
        (command, cwd) for command, cwd in runner.commands
        if command == ["gh", "api", "repos/OWNER/REPO/pulls/77"] or command[:5] == _M1047_DELETE
    ]
    assert normalization
    assert {cwd for _command, cwd in normalization} == {bootstrap}
    assert not config.claude_dir.exists()
    assert len(runner.managed_deletes()) == (1 if labels else 0)


def _m1047_reach_publication(monkeypatch, runner, contract, *, dispatch_error=None):
    monkeypatch.setattr(pr_loop, "recover_issue_created_handoff", lambda *a, **k: None)
    monkeypatch.setattr(pr_loop, "activate_managed_ci", lambda *a, **k: contract)
    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", lambda *a, **k: None)

    def wait(*args, **kwargs):
        if dispatch_error is not None:
            raise dispatch_error
        return ManagedCiOutcome(status="passed", head_sha=runner.pr_payload.get("headRefOid"))

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
@pytest.mark.parametrize("state", ["draft", "ready"])
def test_m1047_failed_publication_releases_label_despite_preservation(
    tmp_path, monkeypatch, origin, state,
):
    # Draft: the pre-readiness provenance read fails.  Ready: the first read
    # succeeds, then the pre-record read and cleanup's read fail.
    runner = _M1047LivePrRunner(
        draft=True,
        events_fail_from=0 if state == "draft" else 1,
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop", max_rounds=1)
    contract = _m1047_contract(origin)
    assert pr_loop_support._preserve_issue_created_managed_suppression(
        contract, active_exception=AgentLoopError("any")
    )
    _m1047_reach_publication(monkeypatch, runner, contract)
    release_calls = _m1047_release_spy(monkeypatch)

    with pytest.raises(AgentLoopError, match="the head is not qualified"):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert not runner.has_managed_label()
    assert len(runner.managed_deletes()) == 1
    assert runner.live_pr["draft"] is (state == "draft")
    # Preservation still applies in the finally block, so no second release.
    assert release_calls == []


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_m1047_interruption_before_publication_still_preserves_label(tmp_path, monkeypatch, origin):
    runner = _M1047LivePrRunner(
        draft=True, codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    _m1047_reach_publication(
        monkeypatch, runner, _m1047_contract(origin),
        dispatch_error=AgentLoopError("dispatch lost"),
    )
    monkeypatch.setattr(
        pr_loop, "publish_manual_v2_qualification",
        lambda *a, **k: pytest.fail("publication must not run"),
    )
    release_calls = _m1047_release_spy(monkeypatch)

    with pytest.raises(AgentLoopError, match="dispatch lost"):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert runner.has_managed_label() and runner.live_pr["draft"] is True
    assert runner.managed_deletes() == []
    assert release_calls == []


def test_m1047_successful_publication_keeps_label_and_finally_does_not_release(
    tmp_path, monkeypatch,
):
    runner = _M1047LivePrRunner(
        draft=True, codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    _m1047_reach_publication(monkeypatch, runner, _m1047_contract("issue-created"))

    def publish(*args, **kwargs):
        runner.live_pr["draft"] = False
        return kwargs["expected_head_sha"]

    monkeypatch.setattr(pr_loop, "publish_manual_v2_qualification", publish)
    release_calls = _m1047_release_spy(monkeypatch)

    assert _m1047_run_pr_loop(runner, pr_number=77, config=config) == 0

    assert runner.has_managed_label() and runner.live_pr["draft"] is False
    assert runner.managed_deletes() == []
    assert release_calls == []


@pytest.mark.parametrize("origin", ["issue-created", "source-managed"])
def test_m1047_reentry_that_aborts_before_publication_is_preserved_for_exact_resume(
    tmp_path, monkeypatch, origin,
):
    runner = _M1047LivePrRunner(
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, max_rounds=1)
    contract = _m1047_contract(origin)
    _m1047_reach_publication(
        monkeypatch, runner, contract, dispatch_error=AgentLoopError("review aborted"),
    )
    observed_at_activation = []

    def reenter(*args, **kwargs):
        # Existing ready/unlabeled re-entry: `ready --undo` and reapply the label.
        observed_at_activation.append((runner.has_managed_label(), runner.live_pr["draft"]))
        runner.live_pr["draft"] = True
        runner.live_pr["labels"] = [{"name": "agent-loop-managed"}]
        return contract

    monkeypatch.setattr(pr_loop, "activate_managed_ci", reenter)
    release_calls = _m1047_release_spy(monkeypatch)

    with pytest.raises(AgentLoopError, match="review aborted"):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert observed_at_activation == [(False, False)]
    assert len(runner.managed_deletes()) == 1
    assert release_calls == []
    assert runner.has_managed_label() and runner.live_pr["draft"] is True

    # The next invocation resumes through the draft/labeled path untouched.
    seen = []
    monkeypatch.setattr(
        pr_loop, "activate_managed_ci", _m1047_recorder(runner, seen, "activation"),
    )
    with pytest.raises(_M1047Stop):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)
    assert seen == [("activation", True, True)]
    assert len(runner.managed_deletes()) == 1


def test_pr_loop_writes_run_end_when_managed_label_cleanup_raises(tmp_path, monkeypatch):
    """Reservation telemetry (#1107): cleanup failure must not skip run-end."""
    from coding_review_agent_loop.worker_telemetry import load_records, telemetry_log_path

    runner = _M1047LivePrRunner(
        draft=True,
        events_fail_from=0,
        codex_outputs=[structured_pr_review(state="approved", summary="Approved.")],
    )
    config = make_config(tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop", max_rounds=1)
    import dataclasses

    contract = dataclasses.replace(_m1047_contract("issue-created"), origin=None, issue_created_pr=False)
    _m1047_reach_publication(monkeypatch, runner, contract)
    calls = []

    def boom(*_a, **_k):
        calls.append(1)
        raise RuntimeError("label cleanup interrupted")

    monkeypatch.setattr(pr_loop, "release_adopted_managed_ci", boom)

    with pytest.raises(RuntimeError, match="label cleanup interrupted"):
        _m1047_run_pr_loop(runner, pr_number=77, config=config)

    assert calls
    assert [r["record"] for r in load_records(telemetry_log_path())] == ["run-start", "run-end"]
    assert runner.telemetry_attribution is None


# --- PR step-back approach alternative on a managed PR (#1628) ---------------


def test_approach_alternative_stop_publishes_continuity_first_and_resume_stops(
    tmp_path, monkeypatch,
):
    """`approach-alt-managed-continuity`."""
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    from pr_loop_test_helpers import SB_HEAD, _StepBackRunner

    def review(items=(), resolved=()):
        return structured_pr_review(
            state="blocking" if items else "approved",
            summary="review",
            blocking_items=list(items),
            prior_item_dispositions=[
                {"item_id": item, "disposition": "resolved"} for item in resolved
            ],
        )

    runner = _StepBackRunner(
        claude_outputs=[
            structured_coder_followup(addressed_items=["item-1"]),
            structured_coder_followup(addressed_items=["item-2"]),
            structured_coder_followup(
                addressed_items=["item-3"],
                summary="Approach alternative: drop the aa4ef5c453dd923aedb5e86e4c3b67aa90d1646c disqualifier.",
            ),
        ],
        codex_outputs=[
            review(["gap one at src/spool.py:124-131"]),
            review(["gap two at src/spool.py:131-136"], resolved=["item-1"]),
            review(["gap three at src/spool.py:140"], resolved=["item-2"]),
        ],
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": SB_HEAD,
            "baseRefName": "main", "body": "Fixes #56",
        },
    )
    handoff = pr_loop_support.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha=SB_HEAD, branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="root",
        authorization_kind="creation", authorization_comment_id=17,
    )
    events = []
    activated_heads = []
    monkeypatch.setattr(pr_loop, "revalidate_issue_created_handoff", lambda *_a, **_k: handoff)

    def activate(*_args, **_kwargs):
        activated_heads.append(runner.pr_payload["headRefOid"])
        return ManagedCiContract(protocol_version=2, issue_created_pr=True)

    monkeypatch.setattr(pr_loop, "activate_managed_ci", activate)
    monkeypatch.setattr(pr_loop, "revalidate_adopted_managed_ci", lambda *_a, **_k: True)
    monkeypatch.setattr(pr_loop, "managed_label_present", lambda *_a, **_k: True)
    monkeypatch.setattr(
        pr_loop, "find_actor_round_metadata_comment_ids", lambda *_a, **_k: (88, 89),
    )

    def publish(*_args, **kwargs):
        events.append(("publish", kwargs["new_head"]))
        return handoff

    monkeypatch.setattr(pr_loop, "publish_issue_created_continuity_authorization", publish)
    original_stop = pr_loop._pr_step_back_approach_alternative_stop

    def stop(*args, **kwargs):
        events.append(("stop-check", kwargs["head_sha"]))
        return original_stop(*args, **kwargs)

    monkeypatch.setattr(pr_loop, "_pr_step_back_approach_alternative_stop", stop)
    real_post = pr_loop.post_pr_comment

    def post(*args, body, **kwargs):
        # The durable human-decision diagnostic (#1640) is plain text.
        if isinstance(body, str) and body.startswith("PR review scheduling diagnostic ("):
            events.append(("diagnostic", runner.pr_payload["headRefOid"]))
        return real_post(*args, body=body, **kwargs)

    monkeypatch.setattr(pr_loop, "post_pr_comment", post)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, reviewer=("codex",), max_rounds=8,
    )

    with pytest.raises(HumanDecisionRequiredError, match="replacing the approved approach"):
        run_pr_loop(
            runner, pr_number=77, config=config, managed_ci_handoff=handoff,
            managed_ci_issue_number=56,
        )
    new_head = f"{3:040x}"
    # The continuity grant for the proposal head precedes the live stop.
    assert events.index(("publish", new_head)) < events.index(("stop-check", new_head))
    # The durable diagnostic follows the coder record and the continuity grant.
    assert events[-2:] == [("stop-check", new_head), ("diagnostic", new_head)]
    codex_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]]
    assert len(codex_calls) == 3
    assert not any(cmd[:3] == ["gh", "workflow", "run"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)

    # Resume: managed CI activates for the live head, then the round-start stop fires.
    events.clear()
    with pytest.raises(HumanDecisionRequiredError, match="replacing the approved approach"):
        run_pr_loop(
            runner, pr_number=77, config=config, managed_ci_handoff=handoff,
            managed_ci_issue_number=56,
        )
    assert activated_heads[-1] == new_head
    assert events == [("stop-check", new_head), ("diagnostic", new_head)]
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["codex"]]) == 3
    assert not any(cmd[:3] == ["gh", "workflow", "run"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize('origin', ['issue-created', 'source-managed', 'adopted'])
def test_observation_stop_keeps_existing_origin_cleanup(tmp_path, monkeypatch, origin):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    runner = _M1047LivePrRunner(draft=True,
        codex_outputs=[structured_pr_review(state='blocking', blocking_items=['Fix it.'])],
        claude_outputs=[structured_coder_followup(addressed_items=['item-1'])],
        advance_pr_head_on_coder_followup=False)
    contract = _m1047_contract(origin)
    monkeypatch.setattr(pr_loop, 'revalidate_adopted_managed_ci', lambda *a, **k: True)
    _m1047_reach_publication(monkeypatch, runner, contract)
    original = runner._maybe_advance_pr_head_for_coder_followup
    def push(cmd):
        original(cmd)
        if '"kind": "coder_followup"' in '\n'.join(cmd): runner.branch_tip = 'b' * 40
    runner._maybe_advance_pr_head_for_coder_followup = push
    release_calls = _m1047_release_spy(monkeypatch)
    initial_messages = []
    real_explain = pr_loop.explain_pr_head_recovery
    def explain(error, config, **kwargs):
        initial_messages.append(str(error))
        assert kwargs['origin'] == origin
        real_explain(error, config, **kwargs)
    monkeypatch.setattr(pr_loop, 'explain_pr_head_recovery', explain)
    with pytest.raises(PrHeadObservationError, match='desync') as error:
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, managed_ci=True))
    assert bool(release_calls) is (origin == 'adopted')
    assert runner.has_managed_label() is (origin != 'adopted')
    if origin == 'adopted':
        assert contract.origin is None and contract.adopted_existing_pr
    for message in [*initial_messages, str(error.value)]:
        if origin in {'adopted', 'source-managed'}:
            assert 'Existing origin-specific managed-CI recovery and cleanup still apply' in message
            assert 'If this is an issue-created managed PR' not in message
            assert 'Authenticate the owning issue' not in message
            assert '--managed-ci-fresh' not in message
        else:
            assert 'Manual rewind invalidates managed continuity' in message
    assert len(initial_messages) == 1
