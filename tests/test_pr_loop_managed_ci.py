"""PR-loop tests for managed_ci: exact-head qualification in the PR loop (#1542)."""

import dataclasses

import pytest

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
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
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

    assert dispatches[0]["expected_head_sha"] == "abc123"
    assert merges == [{"expected_head_sha": "abc123"}]


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
            ManagedCiOutcome(status="failed", checks=failed_checks, head_sha="abc123"),
            ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
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
    assert merges == [{"expected_head_sha": "abc123-coder-1"}]


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
        candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
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
            subject="abc123",
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
            subject="abc123",
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
        protocol_version=2, attached_run_id=123, run_attempt=1, expected_head_sha="abc123"
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
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: merges.append(k))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert dispatches == []
    assert waited == [(123, 1, "abc123")]
    assert merges == [{"expected_head_sha": "abc123"}]
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
        failed_head_sha="old-head", candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="managed-exact-head-ci",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
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
            flow="pr", role="reviewer", agent="Codex", round_number=1, subject="abc123",
            prior_items=(dataclasses.replace(item, lifecycle="awaiting_current_head_review"),),
            state="approved", architecture_contract_version=1,
        ),
    )
    checkpoint_comment = _attach_round_metadata(
        "Qualification was dispatched.",
        PostedRoundMetadata(
            flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
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
        return ManagedCiOutcome(status="passed", head_sha="abc123")

    monkeypatch.setattr(pr_loop, "dispatch_final_qualification", dispatch)
    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait)
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *a, **k: merges.append(k))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert dispatches == [("abc123", "fresh-gen")]
    assert waited == [(999, 1, "abc123")]
    assert merges == [{"expected_head_sha": "abc123"}]
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
                expected_head_sha="abc123",
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
        candidate_head_sha="abc123",
        qualification_attempt_id="555/1",
    )
    fresh = ManagedCiContract(protocol_version=2, intent_generation="fresh-gen")

    assert needs_fresh(checkpoint, fresh, "abc123")
    assert needs_fresh(checkpoint, dataclasses.replace(fresh, expected_head_sha="abc123"), "abc123")
    assert not needs_fresh(checkpoint, dataclasses.replace(fresh, expected_head_sha="old-head"), "abc123")
    assert not needs_fresh(None, fresh, "abc123")
    assert not needs_fresh(QualificationCheckpoint.invalid("stale"), fresh, "abc123")
    assert not needs_fresh(
        dataclasses.replace(checkpoint, lifecycle="qualification_ready"), fresh, "abc123"
    )
    assert not needs_fresh(
        dataclasses.replace(
            checkpoint, obligation_kind="github-pr-checks",
            obligation_identity="github-pr-checks:item-1",
        ),
        fresh,
        "abc123",
    )
    assert not needs_fresh(checkpoint, fresh, "other-head")
    assert not needs_fresh(checkpoint, fresh, None)
    assert not needs_fresh(checkpoint, fresh, "")
    assert not needs_fresh(checkpoint, None, "abc123")
    assert not needs_fresh(checkpoint, dataclasses.replace(fresh, protocol_version=1), "abc123")
    assert not needs_fresh(
        checkpoint, dataclasses.replace(fresh, activation_path="ordinary_fallback"), "abc123"
    )
    for owned in (
        {"attached_run_id": 555},
        {"run_attempt": 1},
        {"nonce": "owned-nonce"},
        {"dispatch_issued": True},
    ):
        assert not needs_fresh(checkpoint, dataclasses.replace(fresh, **owned), "abc123"), owned


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
        candidate_head_sha="abc123",
        obligation_identity="github-pr-checks:item-1",
    )
    checkpoint = QualificationCheckpoint(
        obligation_kind="github-pr-checks",
        obligation_identity=item.obligation_identity,
        lifecycle="qualifying",
        failed_head_sha="old-head",
        candidate_head_sha="abc123",
        base_branch="main",
        requirements_digest=pr_loop_support._qualification_digest(tuple()),
        acquisition_digest=pr_loop_support._qualification_digest(
            {"Codex": (invocation.configured_model, invocation.resolved_effort, "codex")}
        ),
        approval_digest=pr_loop_support._qualification_digest(("Codex",)),
        scheduler_digest=pr_loop_support._qualification_digest(
            pr_loop_support._prior_item_ledger_signature((item,))
        ),
        qualification_attempt_id="github-checks:abc123",
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
            subject="abc123",
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
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(
        pr_loop,
        "publish_manual_v2_qualification",
        lambda *args, **kwargs: published.append(kwargs["expected_head_sha"]) or "abc123",
    )
    monkeypatch.setattr(
        pr_loop_support,
        "merge_pr",
        lambda *args, **kwargs: pytest.fail("manual managed-CI success must not merge"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert published == ["abc123"]
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
            runner.pr_payload["headRefOid"] = "new-head"
            return ManagedCiOutcome(status="head_changed", head_sha="new-head")
        return ManagedCiOutcome(status="passed", head_sha="new-head")

    monkeypatch.setattr(pr_loop, "wait_for_final_qualification", wait_for_managed_ci)
    merges = []
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: merges.append(kwargs))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert call_count == 2
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]) == 2
    assert merges == [{"expected_head_sha": "new-head"}]


def test_managed_ci_merge_conflict_routes_through_reconciliation(tmp_path, monkeypatch):
    conflict = PullRequestMergeability(
        state="conflicted",
        mergeable_raw="CONFLICTING",
        merge_state_raw="DIRTY",
        head_sha="abc123",
        base_branch="main",
    )
    outcomes = iter(
        [
            ManagedCiOutcome(
                status="merge_conflict",
                mergeability=conflict,
                head_sha="abc123",
            ),
            ManagedCiOutcome(status="passed", head_sha="abc123-coder-1"),
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
            status="infrastructure_stall", head_sha="abc123", stall=stall
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
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
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
        lambda *args, **kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(pr_loop_support, "merge_pr", lambda *args, **kwargs: None)

    assert run_pr_loop(
        runner,
        pr_number=77,
        config=config,
        pre_review_test_pending=True,
    ) == 0

    assert published_heads == ["abc123"]
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
        candidate_head_sha="abc123",
        obligation_identity="managed-exact-head-ci:item-1",
    )
    coder_output = structured_coder_followup(
        summary="The failed managed-CI head was repaired.", addressed_items=[seeded.item_id]
    )
    prior_round = _attach_round_metadata(
        coder_output,
        PostedRoundMetadata(
            flow="pr", role="coder", agent="Claude", round_number=1,
            subject="abc123", prior_items=(seeded,), state="blocking",
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
    assert state["dispatches"] == ["abc123"]
    assert not any("37061662125" in b for b in bodies)
    assert [b for b in bodies if "attached attempt" in b and "999/1" in b]
    assert len([b for b in bodies if "attached attempt" in b]) == 1
    assert state["waited"] == [(999, "abc123")]
    assert "resuming attached qualification attempt 37061662125" not in capsys.readouterr().err
    assert state["merges"] == [{"expected_head_sha": "abc123"}]


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
                status="failed", head_sha="abc123",
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
    # A was dispatched for abc123; B for the coder's new head. A's failure is never
    # reported as B's verdict, and the run does not stop on it.
    assert state["dispatches"] == ["abc123", "abc123-coder-1"]
    assert state["waited"] == [(999, "abc123"), (1000, "abc123-coder-1")]
    assert state["merges"] == [{"expected_head_sha": "abc123-coder-1"}]
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
        expected_head_sha="abc123-coder-1", nonce="fresh-nonce-2",
    )
    state2 = _stale_attachment_stubs(
        monkeypatch, stale_contract=resumed,
        outcomes=lambda c, m: ManagedCiOutcome(status="passed", head_sha=m.head_sha),
    )
    runner.pr_payload["state"] = "OPEN"
    assert runner.pr_payload["headRefOid"] == "abc123-coder-1"

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert reviewer_runs() == reviewer_before
    assert state2["dispatches"] == []
    assert state2["waited"] == [(1000, "abc123-coder-1")]
    assert state2["merges"] == [{"expected_head_sha": "abc123-coder-1"}]
