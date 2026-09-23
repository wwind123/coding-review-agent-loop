import hashlib
import json
import re
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.orchestrator as orchestrator_module
import coding_review_agent_loop.phase_progress as phase_progress_module
from coding_review_agent_loop.phase_progress import StagedTopologyOutcome
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.comment_rendering import (
    _render_public_issue_implementation_comment,
    render_canonical_plan_state,
    render_risk_test_matrix_section,
)
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    PhaseImplementationHandoffMetadata,
    PlanPhase,
    RecordedPhase,
    RetainedParentScope,
    approved_plan_hash,
    format_decomposition_parent_summary,
    format_one_shot_impl_handoff_comment,
)
from coding_review_agent_loop.errors import (
    AgentInvocationError,
    DeterministicPlanValidationExhaustion,
    QuotaResetExceededError,
)
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    get_issue_context,
    get_issue_state,
)
from coding_review_agent_loop.managed_ci import (
    AuthenticatedIssueCreatedHandoff,
    ManagedCiContract,
    ManagedCiCreationIntent,
    ManagedCiIssueAuthorization,
    ManagedCiOutcome,
    UNPROTECTED_OVERRIDE_TRAILER,
    format_issue_created_authorization_comment,
    parse_issue_created_authorization_comment,
    parse_managed_ci_override_record,
)
from coding_review_agent_loop.issue_pr_handoff import (
    format_issue_pr_handoff_comment,
    resolve_canonical_pr_for_issue,
)
from coding_review_agent_loop.issue_pr_provenance import IssuePrProvenanceScope
from coding_review_agent_loop.memory import AgentMemoryContext
import coding_review_agent_loop.prompts as prompts_module
from coding_review_agent_loop.plan_review_scheduling import PlanCandidateKey
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    ValidatedAgentResponse,
    _advisory_issue_pr_provenance,
    _attach_round_metadata,
    _decode_round_metadata,
    _infer_staged_parent_issue,
    _plan_subject,
    _resume_plan_round,
    _strip_round_metadata,
)
from coding_review_agent_loop.prompts import (
    build_completion_recovery_prompt,
    build_issue_prompt,
    COMPACT_PLANNING_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
import coding_review_agent_loop.test_runtime as runtime
from coding_review_agent_loop.protocol_markers import PR_BODY_SURFACE, TrustedBody
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    ExecutionAllocation,
    EXECUTION_DISPOSITION_PLANNING,
    ApprovedFollowup,
    ParsedPlanReview,
    PlanReviewItems,
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_plan_revision_patch,
    validate_structured_plan_state,
    validate_structured_issue_implementation,
    risk_test_matrix_prompt_examples,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)
from coding_review_agent_loop.round_state import make_approved_plan_context


def _fresh_child_route_fixture(disposition, *, handoff=None):
    phase = PlanPhase(
        title="Child", scope="Implement child.", non_goals="No unrelated work.",
        dependency_notes="No dependencies.", rollout_risk="low",
        validation="Run focused tests.", parent_context="Approved slice.",
        automation="agent-pr", stage_id="stage-one", position=1,
        deliverables=("Child",), non_goals_items=("Unrelated work",),
        acceptance_criteria=("Tests pass",), compatibility_constraints=("Preserve API",),
        covered_scope_item_ids=("scope-1",), execution_disposition=disposition,
        disposition_rationale="Reviewed route.",
    )
    route = orchestrator_module.resolve_child_execution_route(
        phase, topology_source="approved-plan-v1", recorded_handoff=handoff
    )
    issue = IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body="Child", url="child-url", comments=()
    )
    parent = replace(issue, number=55, title="Parent", url="parent-url")
    return SimpleNamespace(
        parent_issue=55, plan_hash="plan-hash", plan_subject="plan-subject",
        approved_plan="Approved parent plan", parent_plan_context=SimpleNamespace(
            matrix_available=False, risk_test_matrix_payload=None
        ), recommendation=SimpleNamespace(
            strategy="staged", identity=lambda: {"recommendation_sha256": "digest"},
            child_stages=(phase,),
        ), decomposition=SimpleNamespace(phases=(phase,)),
        created=CreatedPhaseIssue(phase=phase, issue_url=issue.url, issue_number=56),
        phase_index=1, stage_id="stage-one", handoff=handoff, overrides=(), route=route,
    ), issue, parent


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
    monkeypatch.setattr(
        orchestrator_module, "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    monkeypatch.setattr(orchestrator_module, "_resolve_fresh_child_provenance", lambda **_: fresh)
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
    monkeypatch.setattr(
        orchestrator_module, "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    monkeypatch.setattr(orchestrator_module, "_resolve_fresh_child_provenance", lambda **_: fresh)
    monkeypatch.setattr(
        orchestrator_module, "_post_child_planning_handoff",
        lambda *_args, **kwargs: (handoff_calls.append(kwargs), events.append("handoff")),
    )
    monkeypatch.setattr(
        orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        orchestrator_module, "_run_plan_first_loop",
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
    monkeypatch.setattr(orchestrator_module, "_resolve_fresh_child_provenance", lambda **_: resumed)
    assert run_issue_loop(
        _FakeRunner(), issue_number=56, config=make_config(tmp_path), plan_first=True
    ) == 19
    assert events == ["handoff", "planner", "planner"]


def test_direct_child_entry_dispatches_through_shared_child_dispatch(tmp_path, monkeypatch):
    fresh, issue, parent = _fresh_child_route_fixture(EXECUTION_DISPOSITION_DIRECT)
    calls = []
    monkeypatch.setattr(
        orchestrator_module, "get_issue_context",
        lambda _runner, *, config, issue_number: issue if issue_number == 56 else parent,
    )
    monkeypatch.setattr(orchestrator_module, "_resolve_fresh_child_provenance", lambda **_: fresh)
    monkeypatch.setattr(orchestrator_module, "prepare_agent_memory", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        orchestrator_module, "_dispatch_decomposition_child",
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
    monkeypatch.setattr(
        orchestrator_module, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(
        orchestrator_module, "find_existing_phase_implementation_handoff",
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
    result = orchestrator_module._dispatch_current_decomposition_phase(
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
from coding_review_agent_loop.salvage import (
    SalvageContext,
    capture_salvage_artifacts,
    latest_salvage_summary,
    post_salvage_comment,
)
from coding_review_agent_loop.runner import CommandResult
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    child_pr_handoff_comment,
    command_index,
    make_config,
    pr_payload_for_state,
    prior_item_dispositions,
    prior_plan_item_dispositions,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_v1_plan_state,
    structured_pr_review,
    structured_issue_implementation,
)


def test_auto_execution_resolves_one_shot_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation

    resolved = orchestrator_module._resolve_execution_policy(
        config,
        recommendation=recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-one-shot"
    assert resolved.strategy == "one-shot"


def test_auto_execution_resolves_staged_after_approval(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")
    recommendation = validate_structured_plan_state(
        structured_v1_plan_state()
    ).execution_recommendation
    staged_recommendation = replace(recommendation, strategy="staged")

    resolved = orchestrator_module._resolve_execution_policy(
        config,
        recommendation=staged_recommendation,
    )

    assert resolved.requested_policy == "auto"
    assert resolved.action == "implement-by-phase"
    assert resolved.strategy == "staged"


def test_auto_execution_rejects_legacy_undecided_plan(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="auto")

    with pytest.raises(AgentLoopError, match="legacy-undecided"):
        orchestrator_module._resolve_execution_policy(
            config,
            recommendation=None,
        )


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
    original_validator = orchestrator_module._validate_plan_revision_response

    def capture_validator(*args, **kwargs):
        captured_gate_values.append(
            kwargs["reject_unsolicited_risk_test_matrix_contract"]
        )
        return original_validator(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module, "_validate_plan_revision_response", capture_validator
    )
    if include_matrix:
        with patch.object(orchestrator_module, "attempt_repair", return_value=None):
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


def _fresh_staged_plan_for_recovery() -> tuple[str, str]:
    raw = structured_v1_plan_state()
    payload, end = json.JSONDecoder().raw_decode(raw.lstrip())
    recommendation = payload["execution_recommendation"]
    recommendation.update(
        {
            "strategy": "staged",
            "staging_feasibility": "safe",
            "scope_items": [
                {
                    "scope_item_id": "scope-1",
                    "requirement": "Deliver the first stage.",
                    "acceptance_criteria": ["The first stage passes."],
                },
                {
                    "scope_item_id": "scope-2",
                    "requirement": "Deliver the second stage.",
                    "acceptance_criteria": ["The second stage passes."],
                },
            ],
            "child_stages": [
                {
                    "stage_id": "stage-1",
                    "position": 1,
                    "title": "First stage",
                    "summary": "Deliver the first stage.",
                    "deliverables": ["First stage implementation."],
                    "non_goals": [],
                    "acceptance_criteria": ["The first stage passes."],
                    "depends_on_stage_ids": [],
                    "dependency_notes": "No dependencies.",
                    "automation": "agent-pr",
                    "rollout_risk": "low",
                    "compatibility_constraints": [],
                    "covered_scope_item_ids": ["scope-1"],
                },
                {
                    "stage_id": "stage-2",
                    "position": 2,
                    "title": "Second stage",
                    "summary": "Deliver the second stage.",
                    "deliverables": ["Second stage implementation."],
                    "non_goals": [],
                    "acceptance_criteria": ["The second stage passes."],
                    "depends_on_stage_ids": ["stage-1"],
                    "dependency_notes": "After stage-1.",
                    "automation": "agent-pr",
                    "rollout_risk": "low",
                    "compatibility_constraints": [],
                    "covered_scope_item_ids": ["scope-2"],
                },
            ],
        }
    )
    recommendation.pop("one_shot_delivery", None)
    staged_raw = json.dumps(payload) + raw.lstrip()[end:]
    parsed = validate_structured_plan_state(staged_raw, require_execution_strategy_contract=1)
    canonical = render_canonical_plan_state(parsed)
    return staged_raw, canonical


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


def test_issue_resume_prompt_retains_configured_command_when_wrapper_probe_fails(
    tmp_path, monkeypatch
):
    config = make_config(
        tmp_path,
        test_command=("pytest", "tests/test_protocol.py", "-q"),
    )
    memory = AgentMemoryContext(
        memory_dir=tmp_path / "memory",
        current_commit=None,
        last_analyzed_commit=None,
        changed_files=(),
        repo_summary=None,
        architecture_map=None,
        test_profile=None,
        toolchain=None,
    )
    monkeypatch.setattr(
        prompts_module,
        "preflight_wrapper_candidates",
        lambda **_kwargs: (runtime.LauncherProbeResult(("/opt/agent-loop", "run-tests"), "failed", "import failed"),),
    )
    prompt = build_issue_prompt(764, config, memory=memory)
    assert "pytest tests/test_protocol.py -q" in prompt
    assert "no wrapper candidate verified" in prompt


def test_issue_handoff_receipt_must_belong_to_current_coder_turn():
    from coding_review_agent_loop.local_test_evidence import bounded_evidence_for_round

    parsed = validate_structured_issue_implementation(
        json.dumps({
            "schema_version": 1,
            "kind": "issue_implementation",
            "state": "blocking",
            "summary": "Implemented the issue.",
            "pr_number": 77,
            "human_requirement_dispositions": [],
            "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
            "test_observations": [{
                "command": "python -m pytest tests/test_protocol.py -q",
                "receipt_id": "old-receipt",
                "claim": "current-result",
            }],
            "tests_run": [],
        }) + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )
    evidence = bounded_evidence_for_round({"observations": [{
        "command": ["python", "-m", "pytest", "tests/test_protocol.py", "-q"],
        "outcome": "passed",
        "provenance": "parent-observed",
        "receipt_id": "old-receipt",
        "turn_id": "prior-turn",
        "environment": "unknown",
        "attribution": {"state": "current-head"},
    }]})

    rendered = _render_public_issue_implementation_comment(
        parsed,
        agent="Claude",
        local_test_evidence=evidence,
        current_test_turn_id="current-turn",
    )

    assert "unverified: unknown or cross-turn receipt" in rendered


def _provenance_pages(message: str):
    page = {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": "abc123",
                    "commits": {
                        "totalCount": 1,
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                        "nodes": [{"commit": {"oid": "commit-1", "message": message}}],
                    },
                }
            }
        }
    }
    return [page, page]


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


def _implementation_matrix_context():
    matrix = parse_risk_test_matrix(
        {
            "applicability": "applicable",
            "rows": [{
                "row_id": "implementation-derived-evidence",
                "label": "Implementation evidence reaches the public handoff",
                "entry_path_or_mode": "issue implementation",
                "initial_state": "approved plan and new implementation PR",
                "event": "the authenticated head is reconciled",
                "expected_outcome": "canonical evidence is persisted",
                "forbidden_side_effects": ["Do not lose the created PR."],
                "proposed_test_level": "orchestrator",
                "proposed_test_location": "tests/test_orchestrator_issue.py",
                "applicability": "required",
                "related_scope_item_ids": ["scope-orchestration-integration"],
                "execution_owner": "one-shot",
            }],
            "important_exclusions": ["Planned tests are not evidence."],
        }
    )
    approved_plan = (
        "Approved implementation plan.\n\n"
        + render_risk_test_matrix_section(matrix)
    )
    identity = risk_test_matrix_identity(matrix)
    context = make_approved_plan_context(
        approved_plan,
        source_locator="test approved implementation plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    return approved_plan, context


def _metadata_from_public_comment(body: str):
    encoded = body.split("AGENT_LOOP_META: ", 1)[1].split(" -->", 1)[0]
    return _decode_round_metadata(encoded)


@pytest.mark.parametrize(
    ("checkout_head", "expected_status", "expected_diagnostic"),
    [
        ("abc123", "verified", None),
        ("checkout-mismatch", "stale/unverified", "checkout-head-mismatch"),
    ],
)
def test_issue_implementation_keeps_pr_and_persists_derived_evidence_after_head_authentication(
    tmp_path, monkeypatch, checkout_head, expected_status, expected_diagnostic
):
    approved_plan, plan_context = _implementation_matrix_context()
    observation = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="abc123",
    )
    implementation_text = _semantic_issue_implementation_text(observation.execution_ref)
    parsed = validate_structured_issue_implementation(
        implementation_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(observation,),
    )
    assert parsed is not None
    coder_response = ValidatedAgentResponse(
        text=implementation_text,
        session_id=None,
        marker_value=parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(observation,),
    )
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56", "headRefOid": "abc123"},
    )
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_run_validated_agent",
        lambda *_args, **_kwargs: (
            setattr(runner, "git_head", "abc123-agent-1") or coder_response
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "resolve_canonical_pr_for_issue",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "sync_coder_base_before_implementation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "preflight_managed_ci_creation",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_pr_loop",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=checkout_head,
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )

    assert (
        orchestrator_module._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan=approved_plan,
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
            approved_plan_context=plan_context,
        )
        == 0
    )

    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ] + [
        item["body"] for item in runner.issue_comments
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = next(
        _metadata_from_public_comment(comment)
        for comment in raw_comments
        if "AGENT_LOOP_META: " in comment
        and _metadata_from_public_comment(comment).role == "coder"
    )
    assert metadata.risk_test_matrix_evidence is not None
    assert metadata.risk_test_matrix_evidence["rows"][0]["status"] == expected_status
    # #959: the establishing comment renders the full list and anchors to itself.
    assert metadata.risk_test_matrix_evidence_full_round == metadata.round_number
    assert metadata.risk_test_matrix_evidence_full_round_status == "valid"
    coder_comment = next(
        comment for comment in raw_comments
        if "AGENT_LOOP_META: " in comment
        and _metadata_from_public_comment(comment).role == "coder"
    )
    assert "<summary>Full matrix evidence (1 row)</summary>" in coder_comment
    assert "unchanged since round" not in coder_comment
    if expected_diagnostic is None:
        assert not metadata.risk_test_matrix_diagnostics
    else:
        assert any(
            item["code"] == expected_diagnostic
            for item in metadata.risk_test_matrix_diagnostics
        )
    assert any("AGENT_ISSUE_PR_HANDOFF" in comment for comment in runner.comments)
    assert any(
        "implementation-derived-evidence" in comment for comment in runner.comments
    )


def _semantic_issue_implementation_text(execution_ref: str) -> str:
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "test_locations": ["tests/test_orchestrator_issue.py"],
        "workflow_path_claim": "The issue implementation handoff reached post-head derivation.",
        "outcome_assertions": ["The selected managed observation passed."],
        "forbidden_effect_assertions": ["The PR handoff was not discarded."],
        "caveats": [],
    }]
    return json.dumps(payload) + raw[end:]


def _workflow_observation(*, execution_ref: str, receipt_id: str, head: str):
    return SimpleNamespace(
        execution_ref=execution_ref,
        receipt_id=receipt_id,
        command=("python3", "-m", "pytest", "tests/test_orchestrator_issue.py", "-q"),
        normalized_command="python3 -m pytest tests/test_orchestrator_issue.py -q",
        outcome="passed",
        provenance="parent-observed",
        turn_id="coder-turn",
        attribution={
            "state": "current-head",
            "head": head,
            "tracked_digest": "tree-current",
            "stable": True,
            "untracked_input": False,
            "caveats": [],
        },
        environment_state="not-compared",
        superseded_by=None,
        caveats=(),
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )


@pytest.mark.parametrize(
    ("mode", "expected_status", "expected_diagnostic"),
    [
        ("success", "verified", None),
        ("exhausted", "stale/unverified", "semantic-correction-exhausted"),
        ("head-race", "stale/unverified", "head-changed-during-correction"),
    ],
)
def test_issue_implementation_runs_bounded_post_auth_correction_without_losing_pr(
    tmp_path, monkeypatch, mode, expected_status, expected_diagnostic
):
    approved_plan, plan_context = _implementation_matrix_context()
    wrong = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-wrong-head",
        head="old-head",
    )
    current = _workflow_observation(
        execution_ref="coder-turn:observation-2",
        receipt_id="receipt-current-head",
        head="abc123",
    )
    initial_text = _semantic_issue_implementation_text(wrong.execution_ref)
    corrected_text = _semantic_issue_implementation_text(current.execution_ref)
    initial_parsed = validate_structured_issue_implementation(
        initial_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(wrong, current),
    )
    assert initial_parsed is not None
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "abc123"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(wrong, current),
    )
    run_pr_calls = []
    monkeypatch.setattr(
        orchestrator_module, "_run_validated_agent", lambda *_a, **_k: coder_response
    )
    monkeypatch.setattr(
        orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    real_get_pr_review_context = orchestrator_module.get_pr_review_context
    review_context_calls = 0

    def get_pr_review_context_with_optional_race(*args, **kwargs):
        nonlocal review_context_calls
        review_context_calls += 1
        if mode == "head-race" and review_context_calls >= 2:
            runner.pr_payload["headRefOid"] = "raced-head"
        return real_get_pr_review_context(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module,
        "get_pr_review_context",
        get_pr_review_context_with_optional_race,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "sync_coder_base_before_implementation",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "validate_assigned_head_advanced", lambda **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    snapshot_heads = iter(
        ("abc123", "abc123") if mode != "head-race" else ("abc123", "raced-head")
    )
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head=next(snapshot_heads),
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_agent_result",
        lambda *_a, **_k: SimpleNamespace(
            text=("not a structured response" if mode == "exhausted" else corrected_text)
        ),
    )
    result = orchestrator_module._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=orchestrator_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    assert evidence_row["status"] == expected_status
    if expected_diagnostic is not None:
        assert any(
            item["code"] == expected_diagnostic
            for item in metadata.risk_test_matrix_diagnostics
        )
    else:
        assert any(
            item["code"] == "wrong-head"
            for item in metadata.risk_test_matrix_diagnostics
        ) is False


def _selector_only_issue_implementation_text(execution_ref: str) -> str:
    """The #1187 shape: a claim with only row_id and execution_refs."""
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
    }]
    return json.dumps(payload) + raw[end:]


def _still_incomplete_issue_implementation_text(execution_ref: str) -> str:
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [execution_ref],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "workflow_path_claim": None,
        "outcome_assertions": [],
    }]
    return json.dumps(payload) + raw[end:]


@pytest.mark.parametrize("mode", ["completes", "still-incomplete"])
def test_issue_implementation_accepts_selector_only_claim_and_corrects_once_after_handoff(
    tmp_path, monkeypatch, mode
):
    """#849: a claim missing all semantic facts must not reject the envelope.

    The PR contract and issue handoff are recorded before the single
    post-authentication correction runs, implementation is not re-run, and
    repair is never invoked.
    """
    approved_plan, plan_context = _implementation_matrix_context()
    current = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="abc123",
    )
    initial_text = _selector_only_issue_implementation_text(current.execution_ref)
    corrected_text = (
        _semantic_issue_implementation_text(current.execution_ref)
        if mode == "completes"
        else _still_incomplete_issue_implementation_text(current.execution_ref)
    )
    initial_parsed = validate_structured_issue_implementation(
        initial_text,
        delivered_risk_test_matrix_row_ids=("implementation-derived-evidence",),
        execution_catalog=(current,),
    )
    assert initial_parsed is not None
    assert initial_parsed.risk_test_matrix_claims.claims[0].workflow_path_claim == ""
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "abc123"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(current,),
    )
    implementation_calls = []
    recorded = []
    correction_prompts = []
    run_pr_calls = []

    def fake_validated_agent(*_a, **_k):
        implementation_calls.append(1)
        return coder_response

    real_contract_record = orchestrator_module.post_trusted_pr_contract_record
    real_handoff = orchestrator_module.post_issue_pr_handoff_comment

    def recording_contract_record(*args, **kwargs):
        result = real_contract_record(*args, **kwargs)
        recorded.append("pr-contract")
        return result

    def recording_handoff(*args, **kwargs):
        result = real_handoff(*args, **kwargs)
        recorded.append(("issue-handoff", kwargs.get("pr_number")))
        return result

    def fake_correction(*_a, **kwargs):
        assert kwargs.get("label") == "semantic-evidence-correction"
        # Inspect the durable records at call time, not after the flow ends.
        assert "pr-contract" in recorded
        assert ("issue-handoff", 77) in recorded
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=corrected_text)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for an incomplete semantic claim")

    monkeypatch.setattr(orchestrator_module, "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(orchestrator_module, "post_trusted_pr_contract_record", recording_contract_record)
    monkeypatch.setattr(orchestrator_module, "post_issue_pr_handoff_comment", recording_handoff)
    monkeypatch.setattr(orchestrator_module, "run_agent_result", fake_correction)
    monkeypatch.setattr(orchestrator_module, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(orchestrator_module, "execute_repair", forbidden_repair)
    monkeypatch.setattr(
        orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "sync_coder_base_before_implementation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "validate_assigned_head_advanced", lambda **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="abc123",
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )

    result = orchestrator_module._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=orchestrator_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert implementation_calls == [1]
    assert len(correction_prompts) == 1
    assert "forbidden_effect_assertions" in correction_prompts[0]
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    codes = [item["code"] for item in metadata.risk_test_matrix_diagnostics]
    if mode == "completes":
        assert evidence_row["status"] == "verified"
        assert "incomplete-semantic-claim" not in codes
    else:
        assert evidence_row["status"] == "stale/unverified"
        assert evidence_row["evidence_citations"] == []
        assert evidence_row["workflow_path_claim"] == ""
        assert evidence_row["outcome_assertions"] == []
        assert "incomplete-semantic-claim" in codes
        assert "semantic-correction-exhausted" in codes


_COMMAND_STRING_REF = (
    "python3 -m pytest tests/test_round_transport.py -q -p no:cacheprovider"
)


def _command_string_issue_implementation_text() -> str:
    """The #859 shape: raw commands in execution_refs, no broker selector."""
    raw = structured_issue_implementation(pr_number=77)
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "implementation-derived-evidence",
        "execution_refs": [_COMMAND_STRING_REF, "python3 -m pytest tests/ -q"],
        "test_identifiers": ["tests/test_orchestrator_issue.py::test_workflow"],
        "test_locations": ["tests/test_orchestrator_issue.py"],
        "workflow_path_claim": "The issue implementation handoff reached post-head derivation.",
        "outcome_assertions": ["The command-string claim was dropped."],
        "forbidden_effect_assertions": ["The PR handoff was not discarded."],
        "caveats": [],
    }]
    return json.dumps(payload) + raw[end:]


@pytest.mark.parametrize("mode", ["corrected", "still-invalid"])
def test_issue_implementation_with_command_string_refs_is_handed_off_and_unverified(
    tmp_path, monkeypatch, mode
):
    """#859: command-string execution_refs must not strand a tested PR.

    The envelope parses with the refs dropped, the PR is authenticated and
    handed off, implementation is not re-run, repair is never invoked, and at
    most one post-authentication correction runs.
    """
    approved_plan, plan_context = _implementation_matrix_context()
    current = _workflow_observation(
        execution_ref="coder-turn:observation-1",
        receipt_id="receipt-current-head",
        head="abc123",
    )
    initial_text = _command_string_issue_implementation_text()
    corrected_text = (
        _semantic_issue_implementation_text(current.execution_ref)
        if mode == "corrected"
        else _command_string_issue_implementation_text()
    )
    # Real orchestrator validator with the live current-turn catalog.
    initial_parsed = orchestrator_module._validate_issue_implementation_response(
        initial_text,
        human_requirements=(),
        delivered_risk_test_matrix=plan_context.risk_test_matrix_payload,
        delivered_risk_test_matrix_identity=plan_context.risk_test_matrix_identity,
        require_risk_test_matrix_contract=True,
        authoritative_test_observations=(current,),
        delivered_risk_test_matrix_row_ids=plan_context.risk_test_matrix_expected_row_ids,
        execution_catalog=(current,),
    )
    assert initial_parsed.pr_number == 77
    initial_claim = initial_parsed.risk_test_matrix_claims.claims[0]
    assert initial_claim.execution_refs == ()
    assert initial_claim.dropped_execution_refs == (
        _COMMAND_STRING_REF, "python3 -m pytest tests/ -q",
    )
    runner = FakeRunner(pr_payload={"body": "Fixes #56", "headRefOid": "abc123"})
    config = make_config(tmp_path, coder="claude")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(),
    )
    coder_response = ValidatedAgentResponse(
        text=initial_text,
        session_id="coder-session",
        marker_value=initial_parsed,
        acquisition_test_turn_id="coder-turn",
        acquisition_test_observations=(current,),
    )
    implementation_calls = []
    recorded = []
    correction_prompts = []
    run_pr_calls = []

    def fake_validated_agent(*_a, **_k):
        implementation_calls.append(1)
        return coder_response

    real_handoff = orchestrator_module.post_issue_pr_handoff_comment

    def recording_handoff(*args, **kwargs):
        result = real_handoff(*args, **kwargs)
        recorded.append(("issue-handoff", kwargs.get("pr_number")))
        return result

    def fake_correction(*_a, **kwargs):
        assert kwargs.get("label") == "semantic-evidence-correction"
        assert ("issue-handoff", 77) in recorded
        correction_prompts.append(kwargs["prompt"])
        return SimpleNamespace(text=corrected_text)

    def forbidden_repair(*_a, **_k):
        raise AssertionError("repair must not run for dropped execution_refs")

    monkeypatch.setattr(orchestrator_module, "_run_validated_agent", fake_validated_agent)
    monkeypatch.setattr(orchestrator_module, "post_issue_pr_handoff_comment", recording_handoff)
    monkeypatch.setattr(orchestrator_module, "run_agent_result", fake_correction)
    monkeypatch.setattr(orchestrator_module, "attempt_repair", forbidden_repair)
    monkeypatch.setattr(orchestrator_module, "execute_repair", forbidden_repair)
    monkeypatch.setattr(
        orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "sync_coder_base_before_implementation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module, "validate_assigned_head_advanced", lambda **_k: None
    )
    monkeypatch.setattr(
        orchestrator_module,
        "run_pr_loop",
        lambda *_a, **kwargs: run_pr_calls.append(kwargs) or 0,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "reconcile_test_observations",
        lambda observations, **_kwargs: SimpleNamespace(observations=tuple(observations)),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "stable_tracked_tree_snapshot",
        lambda _workdir: SimpleNamespace(
            head="abc123",
            tracked_digest="tree-current",
            complete=True,
            stable=True,
            status_clean=True,
        ),
    )

    result = orchestrator_module._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=orchestrator_module._new_usage_context(config),
        approved_plan_context=plan_context,
    )

    assert result == 0
    assert implementation_calls == [1]
    assert len(correction_prompts) == 1
    assert "unknown-execution-ref" in correction_prompts[0]
    assert "Command strings and handles outside this catalog are dropped" in correction_prompts[0]
    assert ("issue-handoff", 77) in recorded
    assert any("AGENT_ISSUE_PR_HANDOFF" in comment for comment in runner.comments)
    assert run_pr_calls and run_pr_calls[0]["pr_number"] == 77
    raw_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
    ]
    metadata = _metadata_from_public_comment(
        next(comment for comment in raw_comments if "AGENT_LOOP_META: " in comment)
    )
    evidence_row = metadata.risk_test_matrix_evidence["rows"][0]
    codes = [item["code"] for item in metadata.risk_test_matrix_diagnostics]
    if mode == "corrected":
        assert evidence_row["status"] == "verified"
        assert "unknown-execution-ref" not in codes
    else:
        assert evidence_row["status"] == "stale/unverified"
        assert evidence_row["evidence_citations"] == []
        assert evidence_row["outcome_assertions"] == ["The command-string claim was dropped."]
        assert any(_COMMAND_STRING_REF in caveat for caveat in evidence_row["caveats"])
        assert "unknown-execution-ref" in codes
        assert "semantic-correction-exhausted" in codes


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


def _managed_issue_handoff(*, nonce: str) -> AuthenticatedIssueCreatedHandoff:
    return AuthenticatedIssueCreatedHandoff(
        pr_number=77,
        issue_number=56,
        repository="OWNER/REPO",
        base_ref="main",
        head_sha="abc123",
        branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop",
        trusted_actor_id=1,
        protection_mode="voluntary",
        override_nonce=nonce,
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
            "headRefOid": "abc123",
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

    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(orchestrator_module, "authenticate_issue_created_handoff", authenticate)
    monkeypatch.setattr(
        orchestrator_module,
        "publish_issue_created_authorization",
        lambda *_args, **kwargs: publication_calls.append(kwargs) or handoff,
    )
    monkeypatch.setattr(orchestrator_module, "revalidate_issue_created_handoff", revalidate)
    monkeypatch.setattr(
        orchestrator_module,
        "activate_managed_ci",
        lambda *_args, **_kwargs: ManagedCiContract(protocol_version=2, issue_created_pr=True),
    )
    monkeypatch.setattr(orchestrator_module, "revalidate_adopted_managed_ci", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(orchestrator_module, "managed_label_present", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        orchestrator_module,
        "publish_round_readiness",
        lambda *_args, **kwargs: readiness_heads.append(kwargs["head_sha"]),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "dispatch_final_qualification",
        lambda *_args, **kwargs: dispatches.append(kwargs),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "wait_for_final_qualification",
        lambda *_args, **_kwargs: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "prepare_v2_merge",
        lambda *_args, **kwargs: prepared_heads.append(kwargs["expected_head_sha"]),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "merge_pr",
        lambda *_args, **kwargs: merged_heads.append(kwargs["expected_head_sha"]),
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(authentication_calls) == 1
    assert len(publication_calls) == 1
    assert ["verify-managed-head"] in [command for command, _cwd in runner.commands]
    assert readiness_heads == ["abc123"]
    assert [dispatch["expected_head_sha"] for dispatch in dispatches] == ["abc123"]
    assert prepared_heads == ["abc123"]
    assert merged_heads == ["abc123"]


def test_direct_issue_conflict_posts_once_and_stops_before_pr_handoff_gates(tmp_path, monkeypatch):
    runner = FakeRunner(claude_outputs=[_blocked_issue_implementation()])
    config = make_config(tmp_path)
    issue_context = _issue_context_with_blocked_requirement()
    gate_calls = []

    monkeypatch.setattr(orchestrator_module, "validate_open_issue", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module, "get_issue_context", lambda *args, **kwargs: issue_context)
    monkeypatch.setattr(orchestrator_module, "resolve_canonical_pr_for_issue", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module, "prepare_agent_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module, "sync_coder_base_before_implementation", lambda *args, **kwargs: None)

    def unexpected_gate(name):
        def _fail(*args, **kwargs):
            gate_calls.append(name)
            raise AssertionError(f"{name} must not run after an implementation conflict")

        return _fail

    for name in (
        "validate_open_pr",
        "get_pr_review_context",
        "post_issue_pr_handoff_comment",
        "run_pr_loop",
    ):
        monkeypatch.setattr(orchestrator_module, name, unexpected_gate(name))

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

    monkeypatch.setattr(orchestrator_module, "resolve_canonical_pr_for_issue", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module, "sync_coder_base_before_implementation", lambda *args, **kwargs: None)
    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *args, **kwargs: None)

    def unexpected_gate(name):
        def _fail(*args, **kwargs):
            gate_calls.append(name)
            raise AssertionError(f"{name} must not run after an implementation conflict")

        return _fail

    for name in (
        "validate_open_pr",
        "get_pr_review_context",
        "post_issue_pr_handoff_comment",
        "run_pr_loop",
    ):
        monkeypatch.setattr(orchestrator_module, name, unexpected_gate(name))

    with pytest.raises(AgentLoopError, match="not accepted for handoff"):
        orchestrator_module._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Approved implementation plan.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
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
    monkeypatch.setattr(orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator_module, "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None)

    with pytest.raises(
        AgentLoopError, match="forged reserved protocol record syntax"
    ) as excinfo:
        orchestrator_module._implement_approved_issue(
            runner, issue_number=56, approved_plan="Approved implementation plan.",
            config=config, memory=None,
            issue_context=IssueContext(
                number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
                url="https://github.test/issues/56", comments=(), human_requirements=(),
            ),
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
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
    monkeypatch.setattr(orchestrator_module, "resolve_canonical_pr_for_issue", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator_module, "sync_coder_base_before_implementation", lambda *_a, **_k: None)
    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: None)
    reviewed = []
    monkeypatch.setattr(
        orchestrator_module, "run_pr_loop", lambda *_a, **kwargs: reviewed.append(kwargs["pr_number"]) or 0
    )

    assert orchestrator_module._implement_approved_issue(
        runner, issue_number=56, approved_plan="Approved implementation plan.",
        config=config, memory=None,
        issue_context=IssueContext(
            number=56, repo="OWNER/REPO", title="Issue", body="Issue body",
            url="https://github.test/issues/56", comments=(), human_requirements=(),
        ),
        coder_session_id=None,
        usage_context=orchestrator_module._new_usage_context(config),
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
            "headRefOid": "abc123",
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

    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_args, **_kwargs: intent)
    monkeypatch.setattr(orchestrator_module, "authenticate_issue_created_handoff", authenticate)
    monkeypatch.setattr(
        orchestrator_module,
        "publish_issue_created_authorization",
        lambda *_args, **_kwargs: handoff,
    )
    monkeypatch.setattr(orchestrator_module, "run_pr_loop", review_entry)

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


class FakeRunner(_FakeRunner):
    """Issue-loop fixtures model the PR created for issue #56 explicitly."""

    def __init__(self, **kwargs):
        # Most of these long-lived fixtures predate the initial plan_state
        # contract. Upgrade their two common initial-plan shorthands while
        # leaving malformed marker-only cases with non-blocking states intact
        # for their explicit rejection tests below.
        legacy_initial_plans = {
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
        }
        for output_key in ("claude_outputs", "codex_outputs", "gemini_outputs", "antigravity_outputs"):
            outputs = kwargs.get(output_key)
            if outputs is None:
                continue
            kwargs[output_key] = [
                structured_plan_state(summary="Initial plan.", plan_steps=["Make the change."])
                if output in legacy_initial_plans
                else output
                for output in outputs
            ]
        # Signed-requirement tests written before the typed disposition
        # contract focus on their named acknowledgement behavior. Supply the
        # otherwise-irrelevant complete attestation to their structured plan
        # fixtures, while preserving intentionally missing acknowledgement
        # markers for the tests that reject those.
        issue_body = str((kwargs.get("issue_payload") or {}).get("body") or "")
        if "-- Human Reviewer" in issue_body:
            issue_payload = kwargs.get("issue_payload") or {}
            requirement = HumanReviewRequirement(
                source_type="Issue body",
                author=str((issue_payload.get("author") or {}).get("login") or "") or None,
                created_at=issue_payload.get("createdAt"),
                url=issue_payload.get("url") or "https://github.com/OWNER/REPO/issues/56",
                body=issue_body.split("-- Human Reviewer", 1)[0].strip(),
            )
            for output_key in ("claude_outputs", "codex_outputs", "gemini_outputs", "antigravity_outputs"):
                outputs = kwargs.get(output_key)
                if outputs is None:
                    continue
                kwargs[output_key] = [
                    _add_default_requirement_disposition(
                        output,
                        requirement_id=requirement.requirement_id,
                    )
                    for output in outputs
                ]
        kwargs.setdefault("pr_payload", {"body": "Fixes #56"})
        super().__init__(**kwargs)


class _PlanDiagnosticRunner(_FakeRunner):
    def __init__(self, *, post_returncode=0, issue_number=813):
        super().__init__()
        self.post_returncode = post_returncode
        self.issue_number = issue_number
        self.diagnostic_posts = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"login": "agent", "id": 7}), "", 0
            )
        if command[:4] == ["gh", "api", "--method", "POST"]:
            endpoint = command[4] if len(command) > 4 else ""
            if endpoint == f"repos/OWNER/REPO/issues/{self.issue_number}/comments":
                recorded, cwd_path = self._record_command(args, cwd)
                body = json.loads(input_text or "{}")["body"]
                self.diagnostic_posts.append(body)
                if self.post_returncode:
                    return CommandResult(recorded, cwd_path, "", "post failed", self.post_returncode)
                return CommandResult(
                    recorded,
                    cwd_path,
                    json.dumps(
                        {
                            "id": 901,
                            "created_at": "2026-09-17T05:30:00Z",
                            "body": body,
                            "user": {"login": "agent", "id": 7},
                        }
                    ),
                    "",
                    0,
                )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _PaginatedIssueCommentsRunner(_FakeRunner):
    def __init__(self, pages):
        super().__init__(issue_comments=[])
        self.pages = pages
        self.rest_comment_requests = []

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        if command[:2] == ["gh", "api"] and len(command) > 2 and command[2].startswith(
            "repos/OWNER/REPO/issues/56/comments?"
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            query = command[2].split("?", 1)[1]
            self.rest_comment_requests.append(query)
            page = int(dict(part.split("=", 1) for part in query.split("&"))["page"])
            return CommandResult(
                recorded,
                cwd_path,
                json.dumps(self.pages[page - 1]),
                "",
                0,
            )
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


def test_issue_comment_recovery_parses_rest_user_identity_across_all_pages(tmp_path):
    marker_body = "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: payload -->"
    first_page = [
        {"id": index + 1, "body": f"ordinary {index}", "user": {"login": "agent", "id": 7}}
        for index in range(100)
    ]
    second_page = [
        {
            "id": 101,
            "created_at": "2026-09-17T05:30:00Z",
            "body": marker_body,
            "user": {"login": "agent", "id": 7},
        }
    ]
    runner = _PaginatedIssueCommentsRunner([first_page, second_page])
    runner.issue_comments = [
        {
            "author": {"login": "agent", "id": 7},
            "createdAt": "2026-09-17T05:30:00Z",
            "body": marker_body,
        }
    ]
    context = get_issue_context(
        runner,
        config=make_config(tmp_path),
        issue_number=56,
    )

    recovered = next(comment for comment in context.comments if comment.body == marker_body)
    assert recovered.author == "agent"
    assert recovered.author_id == 7
    assert recovered.comment_id == 101
    assert runner.rest_comment_requests == ["per_page=100&page=1", "per_page=100&page=2"]


def test_issue_comment_recovery_discovers_diagnostic_beyond_graphql_projection(tmp_path):
    marker_body = "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: payload -->"
    first_page = [
        {
            "id": index + 1,
            "body": f"ordinary {index}",
            "created_at": f"2026-09-17T04:00:{index:02d}Z",
            "user": {"login": "agent", "id": 7},
        }
        for index in range(100)
    ]
    second_page = [
        {
            "id": 101,
            "created_at": "2026-09-17T05:30:00Z",
            "body": marker_body,
            "user": {"login": "agent", "id": 7},
        }
    ]
    runner = _PaginatedIssueCommentsRunner([first_page, second_page])
    runner.issue_comments = [
        {
            "author": {"login": "agent", "id": 7},
            "createdAt": f"2026-09-17T04:00:{index:02d}Z",
            "body": f"ordinary {index}",
        }
        for index in range(100)
    ]

    context = get_issue_context(runner, config=make_config(tmp_path), issue_number=56)

    recovered = next(comment for comment in context.comments if comment.body == marker_body)
    assert recovered.comment_id == 101
    assert recovered.author == "agent"
    assert recovered.author_id == 7
    assert runner.rest_comment_requests == ["per_page=100&page=1", "per_page=100&page=2"]


def test_issue_comment_recovery_discovers_canonical_round_transport(
    tmp_path,
):
    canonical_plan = structured_plan_state(summary="Historical canonical plan.")
    canonical_body = _attach_round_metadata(
        canonical_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(canonical_plan),
            canonical_plan=canonical_plan,
            raw_structured_coder_response=canonical_plan,
            architecture_contract_version=1,
            state="approved",
        ),
    )
    first_page = [
        {
            "id": index + 1,
            "body": f"ordinary {index}",
            "created_at": f"2026-09-17T04:00:{index:02d}Z",
            "user": {"login": "agent", "id": 7},
        }
        for index in range(100)
    ]
    second_page = [
        {
            "id": 101,
            "created_at": "2026-09-17T05:30:00Z",
            "body": str(canonical_body),
            "user": {"login": "agent", "id": 7},
        }
    ]
    runner = _PaginatedIssueCommentsRunner([first_page, second_page])
    runner.issue_comments = [
        {
            "author": {"login": "agent", "id": 7},
            "createdAt": f"2026-09-17T04:00:{index:02d}Z",
            "body": f"ordinary {index}",
        }
        for index in range(100)
    ]

    context = get_issue_context(runner, config=make_config(tmp_path), issue_number=56)

    recovered = next(comment for comment in context.comments if comment.body == str(canonical_body))
    assert recovered.comment_id == 101
    assert recovered.author == "agent"
    assert recovered.author_id == 7
    assert runner.rest_comment_requests == ["per_page=100&page=1", "per_page=100&page=2"]


def test_issue_comment_recovery_fails_closed_when_a_later_rest_page_is_unavailable(tmp_path):
    marker_body = "<!-- AGENT_PLAN_VALIDATION_DIAGNOSTIC: payload -->"

    class IncompleteRunner(_PaginatedIssueCommentsRunner):
        def _run_locked(self, args, *, cwd, check, input_text=None):
            command = list(args)
            if command[:2] == ["gh", "api"] and len(command) > 2 and command[2].startswith(
                "repos/OWNER/REPO/issues/56/comments?"
            ):
                recorded, cwd_path = self._record_command(args, cwd)
                page = int(command[2].rsplit("=", 1)[1])
                if page == 2:
                    return CommandResult(recorded, cwd_path, "", "network failure", 1)
            return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)

    runner = IncompleteRunner(
        [[
            {"id": index + 1, "body": f"ordinary {index}", "user": {"login": "agent", "id": 7}}
            for index in range(100)
        ]]
    )
    runner.issue_comments = [
        {
            "author": {"login": "agent", "id": 7},
            "createdAt": "2026-09-17T05:30:00Z",
            "body": marker_body,
        }
    ]
    with pytest.raises(AgentLoopError, match="incomplete"):
        get_issue_context(runner, config=make_config(tmp_path), issue_number=56)


def test_exhausted_fresh_plan_validation_diagnostic_survives_resume(tmp_path):
    runner = _PlanDiagnosticRunner()
    config = make_config(tmp_path, execution_strategy_contract_required=True)
    issue_context = IssueContext(
        number=813,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/813",
        comments=(),
    )
    original_error = AgentInvocationError(
        "deterministic validator rejected the plan",
        failure_category="deterministic",
    )
    exhaustion = DeterministicPlanValidationExhaustion(
        candidate_kind="plan_state",
        candidate_text='{"kind":"plan_state"}',
        diagnostic="missing complete-scope audit operation",
        candidate_digest="a" * 64,
    )
    orchestrator_module._persist_exhausted_plan_validation_diagnostic(
        runner,
        config=config,
        issue_context=issue_context,
        issue_number=813,
        original_error=original_error,
        exhaustion=exhaustion,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        require_execution_strategy_contract=True,
        require_risk_test_matrix_contract=True,
    )
    assert len(runner.diagnostic_posts) == 1
    body = runner.diagnostic_posts[0]
    assert "901" not in body
    assert "2026-09-17T05:30:00Z" not in body
    resumed_context = replace(
        issue_context,
        comments=(
            IssueComment(
                author="agent",
                author_id=7,
                comment_id=901,
                created_at="2026-09-17T05:30:00Z",
                body=body,
            ),
        ),
    )
    diagnostic = orchestrator_module._recover_current_plan_validation_diagnostic(
        runner,
        config=config,
        issue_context=resumed_context,
        issue_number=813,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        require_execution_strategy_contract=True,
        require_risk_test_matrix_contract=True,
    )
    assert diagnostic is not None
    assert diagnostic.failure_attempt == 1
    assert diagnostic.diagnostic == "missing complete-scope audit operation"


def test_validation_record_post_failure_preserves_original_error(tmp_path):
    runner = _PlanDiagnosticRunner(post_returncode=1)
    config = make_config(tmp_path, execution_strategy_contract_required=True)
    issue_context = IssueContext(
        number=813, repo="OWNER/REPO", title="Issue", body="Issue", url=None, comments=()
    )
    original_error = AgentInvocationError(
        "original deterministic validator cause",
        failure_category="deterministic",
    )
    exhaustion = DeterministicPlanValidationExhaustion(
        candidate_kind="plan_revision",
        candidate_text='{"kind":"plan_revision"}',
        diagnostic="revision validation failed",
        candidate_digest="b" * 64,
    )
    with pytest.raises(AgentInvocationError) as raised:
        orchestrator_module._persist_exhausted_plan_validation_diagnostic(
            runner,
            config=config,
            issue_context=issue_context,
            issue_number=813,
            original_error=original_error,
            exhaustion=exhaustion,
            target_coder_round=2,
            prior_plan_subject="c" * 64,
            candidate_kind="plan_revision",
            require_execution_strategy_contract=True,
            require_risk_test_matrix_contract=True,
        )
    assert "original deterministic validator cause" in str(raised.value)
    assert "not persisted" in str(raised.value)
    assert runner.diagnostic_posts


def test_exhausted_plan_validation_persists_from_the_planning_orchestration_path(
    tmp_path, monkeypatch
):
    invalid_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    invalid_payload.pop("architecture_impact")
    invalid_candidate = (
        json.dumps(invalid_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [invalid_candidate]
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *args, **kwargs: (None, None, []),
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert error.value.plan_validation_exhaustion is not None
    assert len(runner.diagnostic_posts) == 1
    assert "not persisted" not in str(error.value)


@pytest.mark.parametrize(
    ("missing_contract", "agent_max_retries", "expected_planner_calls"),
    [
        ("execution", 0, 1),
        ("matrix", 0, 1),
        ("execution", 1, 2),
        ("matrix", 1, 1),
        ("matrix-malformed", 0, 1),
        ("matrix-malformed", 1, 1),
    ],
)
def test_exhausted_fresh_contract_integrity_persists_validation_diagnostic(
    tmp_path, missing_contract, agent_max_retries, expected_planner_calls
):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    if missing_contract == "execution":
        payload.pop("execution_recommendation")
        expected_diagnostic = "execution_recommendation"
    elif missing_contract == "matrix":
        payload.pop("risk_test_matrix")
        payload.pop("risk_test_matrix_changes")
        payload.pop("risk_test_matrix_contract_version")
        expected_diagnostic = "risk_test_matrix"
    else:
        payload["risk_test_matrix"].pop("important_exclusions")
        expected_diagnostic = "important_exclusions"
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate] * expected_planner_calls
    config = make_config(
        tmp_path,
        agent_max_retries=agent_max_retries,
        execution_strategy_contract_required=True,
    )

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert expected_diagnostic in exhaustion.diagnostic
    assert exhaustion.candidate_digest == hashlib.sha256(candidate.encode()).hexdigest()
    assert len(runner.diagnostic_posts) == 1
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == expected_planner_calls
    assert error.value.failure_category == "fresh-contract-integrity"


def test_invalid_terminal_plan_repair_replaces_persisted_candidate_provenance(
    tmp_path, monkeypatch
):
    source_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    source_payload.pop("architecture_impact")
    source_candidate = (
        json.dumps(source_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_payload = json.loads(structured_plan_state().split("\n", 1)[0])
    repair_payload.pop("summary")
    repair_candidate = (
        json.dumps(repair_payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_attempt = SimpleNamespace(
        backend="gemini",
        model="repair-model",
        prompt="",
        output=repair_candidate,
        returncode=0,
        outcome="invalid_output",
        diagnostic="combined backend text that must not be persisted",
        log_path=None,
        fallback_planned=False,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *args, **kwargs: (repair_candidate, None, [repair_attempt]),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [source_candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    exhaustion = error.value.plan_validation_exhaustion
    assert exhaustion is not None
    assert exhaustion.candidate_text == repair_candidate
    assert exhaustion.candidate_digest == hashlib.sha256(repair_candidate.encode()).hexdigest()
    assert "plan_state is missing required field(s): summary" in exhaustion.diagnostic
    assert "combined backend text" not in exhaustion.diagnostic
    assert len(runner.diagnostic_posts) == 1


@pytest.mark.parametrize(
    ("outcome", "returncode", "expected_category"),
    [("timeout", None, "timeout"), ("nonzero_exit", 1, "repair-provider-failure")],
)
def test_terminal_plan_repair_provider_failure_does_not_persist_stale_diagnostic(
    tmp_path, monkeypatch, outcome, returncode, expected_category
):
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    payload.pop("architecture_impact")
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repair_attempt = SimpleNamespace(
        backend="gemini",
        model="repair-model",
        prompt="",
        output="",
        returncode=returncode,
        outcome=outcome,
        diagnostic="repair transport or provider failed",
        log_path=None,
        fallback_planned=False,
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *args, **kwargs: (None, None, [repair_attempt]),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    assert error.value.failure_category == expected_category
    assert error.value.plan_validation_exhaustion is None
    assert runner.diagnostic_posts == []



_TRANSIENT_REPAIR_SUGGESTION = (
    "Suggestion: re-run the same command — the Antigravity repair model "
    "reported a transient model-access failure; the round is resumable "
    "and a retry may succeed."
)


def _repair_attempt(outcome, *, output="", returncode=0, model="Gemini 3.8 Flash (Medium)"):
    return SimpleNamespace(
        backend="antigravity",
        model=model,
        prompt="",
        output=output,
        returncode=returncode,
        outcome=outcome,
        diagnostic=(
            "model-access validation errors" if outcome == "transient_provider_error" else ""
        ),
        log_path=None,
        fallback_planned=False,
    )


@pytest.mark.parametrize(
    ("chain", "expected_category"),
    [
        (["transient_provider_error", "transient_provider_error"], "repair-provider-failure"),
        (
            ["invalid_output", "transient_provider_error", "transient_provider_error"],
            "repair-provider-failure",
        ),
        (["transient_provider_error", "timeout"], "timeout"),
    ],
)
def test_terminal_transient_plan_repair_is_resumable_provider_failure(
    tmp_path, monkeypatch, chain, expected_category
):
    """Issue #846: a chain ending in a transient agy failure is not deterministic."""
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    payload.pop("architecture_impact")
    candidate = (
        json.dumps(payload)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    attempts = [
        _repair_attempt(
            outcome,
            output=candidate if outcome == "invalid_output" else "",
            returncode=None if outcome == "timeout" else 0,
        )
        for outcome in chain
    ]
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *args, **kwargs: (None, None, attempts),
    )
    runner = _PlanDiagnosticRunner(issue_number=56)
    runner.claude_outputs = [candidate]

    with pytest.raises(AgentInvocationError) as error:
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, agent_max_retries=0),
            plan_first=True,
        )

    assert error.value.failure_category == expected_category
    assert error.value.plan_validation_exhaustion is None
    assert runner.diagnostic_posts == []
    if expected_category == "repair-provider-failure":
        assert _TRANSIENT_REPAIR_SUGGESTION in str(error.value)
    else:
        assert _TRANSIENT_REPAIR_SUGGESTION not in str(error.value)


def test_repair_provider_failure_suggestion_matches_transient_reason_only():
    reason = (
        "plan_review is invalid; repair invocation failure: "
        "antigravity/Gemini 3.8 Flash (Medium): transient_provider_error "
        "(model-access validation errors)"
    )
    assert (
        orchestrator_module._failure_suggestion("repair-provider-failure", reason, "Codex")
        == _TRANSIENT_REPAIR_SUGGESTION
    )
    assert orchestrator_module._failure_suggestion(
        "repair-provider-failure",
        "repair invocation failure: antigravity/model: nonzero_exit (boom)",
        "Codex",
    ) == ""


def _add_default_requirement_disposition(
    output: str,
    *,
    requirement_id: str = "Requirement 1",
) -> str:
    if requirement_id.startswith("hr-"):
        output = output.replace("Requirement 1", f"Requirement {requirement_id}")
        output = output.replace(f'"Requirement {requirement_id}"', f'"{requirement_id}"')
    try:
        payload, end = json.JSONDecoder().raw_decode(output.lstrip())
    except (json.JSONDecodeError, ValueError):
        return output
    if not isinstance(payload, dict) or payload.get("kind") not in {
        "plan_state", "plan_revision", "plan_review"
    }:
        return output
    if payload.get("human_requirement_dispositions"):
        return output
    payload["human_requirement_dispositions"] = [{
        "requirement_id": requirement_id,
        "disposition": "addressed",
        "evidence": "The structured plan covers the signed requirement.",
    }]
    prefix = output[: len(output) - len(output.lstrip())]
    return prefix + json.dumps(payload) + output.lstrip()[end:]


def _initial_plan_state(*, summary: str = "Initial plan.", human_requirements: str = "") -> str:
    return structured_plan_state(summary=summary).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        human_requirements + "\n<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )


def _plan_with_requirement_disposition(
    *,
    kind: str,
    summary: str,
    plan_steps: list[str],
    evidence: str,
    state: str = "blocking",
) -> str:
    payload = {
        "schema_version": 1,
        "kind": kind,
        "state": state,
        "summary": summary,
        "plan_steps": plan_steps,
        "architecture_impact": {
            "status": "unchanged",
            "rationale": "No architectural contract changed.",
            "affected_components": [], "dependencies": [],
            "execution_data_flows": [], "persistence": [],
            "public_contracts": [], "security_boundaries": [],
            "canonical_document_action": "no-change",
            "canonical_document_path": None,
            "canonical_document_rationale": "",
        },
        "human_requirement_dispositions": [{
            "requirement_id": "Requirement 1",
            "disposition": "addressed",
            "evidence": evidence,
        }],
    }
    if kind == "plan_revision":
        payload["prior_plan_item_dispositions"] = []
    return (
        json.dumps(payload)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
        + f"- Requirement 1: {evidence}\n"
        + f"<!-- AGENT_PLAN_STATE: {state} -->\n-- Anthropic Claude"
    )


def _plan_review_with_requirement_disposition(
    *, state: str, summary: str, marker: bool = False, prior_dispositions: list[dict[str, str]] | None = None
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": state,
                "summary": summary,
                "blocking_plan_issues": [summary] if state == "blocking" else [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": prior_dispositions or [],
                "human_requirement_dispositions": [{
                    "requirement_id": "Requirement 1",
                    "disposition": "addressed",
                    "evidence": summary,
                }],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if marker else "")
        + f"\n<!-- AGENT_PLAN_STATE: {state} -->\n-- OpenAI Codex"
    )


def test_issue_loop_creates_pr_then_alternates_until_codex_approval(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["claude", "--print"] in command_names
    assert ["codex", "exec"] in command_names
    assert len(runner.comments) == 5
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")
    assert list((tmp_path / "logs").glob("*-claude-attempt1.log"))
    assert list((tmp_path / "logs").glob("*-codex.log"))
    assert (tmp_path / "logs" / ".gitignore").read_text(encoding="utf-8") == "*\n!.gitignore\n"

def test_issue_loop_syncs_coder_base_after_memory_before_coder(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)
    config.agent_memory_dir.mkdir(parents=True)
    (config.agent_memory_dir / "last-analyzed-commit").write_text("base123\n", encoding="utf-8")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = runner.commands
    issue_context_index = command_index(commands, ["gh", "issue", "view"])
    memory_index = command_index(commands, ["git", "diff", "--name-only"])
    fetch_index = command_index(commands, ["git", "fetch", "origin"])
    switch_index = command_index(commands, ["git", "switch", "main"])
    pull_index = command_index(commands, ["git", "pull", "--ff-only", "origin", "main"])
    coder_index = command_index(commands, ["claude", "--print"])

    assert issue_context_index < memory_index < fetch_index < switch_index < pull_index < coder_index

def test_get_issue_context_parses_signed_issue_body_and_comments(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "number": 56,
            "title": "Support signed issue requirements",
            "body": "Use the stable API path.\n\n-- Human Reviewer",
            "url": "https://github.com/OWNER/REPO/issues/56",
            "author": {"login": "issue-author"},
            "createdAt": "2026-05-17T08:00:00Z",
        },
        issue_comments=[
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-17T09:00:00Z",
                "url": "https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                "body": "Unsigned discussion remains normal context.",
            },
            {
                "author": {"login": "lead"},
                "createdAt": "2026-05-17T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/issues/56#issuecomment-2",
                "body": "Add a regression test.\n\n-- Human Reviewer",
            },
        ],
    )
    config = make_config(tmp_path)

    issue_context = get_issue_context(runner, config=config, issue_number=56)

    assert [item.source_type for item in issue_context.human_requirements] == [
        "Issue body",
        "Issue comment",
    ]
    assert [item.author for item in issue_context.human_requirements] == ["issue-author", "lead"]
    assert [item.created_at for item in issue_context.human_requirements] == [
        "2026-05-17T08:00:00Z",
        "2026-05-17T10:00:00Z",
    ]
    assert issue_context.human_requirements[0].body == "Use the stable API path."
    assert issue_context.human_requirements[1].body == "Add a regression test."
    assert issue_context.comments[0].body == "Unsigned discussion remains normal context."


def test_infer_staged_parent_requires_generated_marker_or_body_header():
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Ordinary issue text.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Quoted AGENT_SPLIT_CHILD: parent=12 in review prose.",
            ),
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #13: quoted comment text.",
            ),
        ),
    )

    assert _infer_staged_parent_issue(issue_context) is None

    marked_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="<!-- AGENT_SPLIT_CHILD: parent=12 key=" + "a" * 64 + " -->",
        url=issue_context.url,
        comments=(),
    )
    assert _infer_staged_parent_issue(marked_context) == 12

    decomposed_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child phase issue for parent #13: https://github.com/OWNER/REPO/issues/13\n\nDetails.",
        url=issue_context.url,
        comments=(
            IssueComment(
                author="bot",
                created_at=None,
                body="Child phase issue for parent #14: quoted comment text.",
            ),
        ),
    )
    assert _infer_staged_parent_issue(decomposed_context) == 13


def test_issue_loop_can_use_codex_as_coder_and_claude_as_reviewer(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\n<!-- AGENT_STATE: blocking -->",
        ],
        claude_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == [
        ["codex", "exec"],
        ["claude", "--print"],
        ["codex", "exec"],
        ["claude", "--print"],
    ]
    assert len(runner.comments) == 5
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")

def test_issue_loop_runs_pre_review_tests_after_coder_changes(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\nTests: pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
            "Fixed review.\nTests: pytest passed.\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "Finding: bug remains.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    first_coder = command_index(runner.commands, ["claude", "--print"])
    first_test = commands.index(["pytest", "tests/test_agent_loop.py"])
    first_review = command_index(runner.commands, ["codex", "exec"])
    assert first_coder < first_test < first_review
    assert commands.count(["pytest", "tests/test_agent_loop.py"]) == 3

def test_issue_loop_requires_claude_to_report_pr_number(tmp_path):
    runner = FakeRunner(claude_outputs=["Created something.\n<!-- AGENT_STATE: blocking -->"])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="valid PR"):
        run_issue_loop(runner, issue_number=56, config=config)

def test_issue_loop_rejects_missing_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python -m pytest passed.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing requirement ID"):
        run_issue_loop(runner, issue_number=56, config=config)

def test_issue_loop_accepts_initial_issue_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the legacy flag.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            "Created PR.\nTests: python3 -m pytest passed.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: kept the legacy flag path.\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->"
        ],
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

def test_issue_loop_rejects_pr_number_before_running_claude(tmp_path):
    runner = FakeRunner(issue_payload={
        "number": 62,
        "state": "closed",
        "is_pr": True,
        "url": "https://github.com/OWNER/REPO/pull/62",
    })
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="pull request, not an issue"):
        run_issue_loop(runner, issue_number=62, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_stops_after_approved_plan(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                summary="Update the CLI and cover it with regression tests.",
                plan_steps=["Update the CLI.", "Add tests."],
            ),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert any(cmd[:3] == ["claude", "--print", "--output-format"] for cmd, _cwd in runner.commands)
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert len(runner.comments) == 3
    assert runner.comments[0].startswith("## Plan")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nPlan looks sound.")
    assert "Outcome: implement" in runner.comments[2]
    assert not any(cmd[:2] == ["git", "fetch"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_rejects_missing_initial_plan_human_requirements_acknowledgement(
    tmp_path,
):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            structured_plan_state()
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

def test_issue_loop_plan_first_accepts_initial_plan_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            structured_plan_state().replace(
                "\n<!-- AGENT_PLAN_STATE: blocking -->",
                "\n"
                f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan keeps the public API unchanged.\n"
                "<!-- AGENT_PLAN_STATE: blocking -->",
                1,
            )
        ],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.", human_requirements_resolved=True)],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_issue_loop_plan_first_fails_closed_when_requirements_change_after_approval(
    tmp_path, monkeypatch
):
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )
    initial = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(requirement_1,),
    )
    refreshed = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(requirement_1, requirement_2),
    )
    contexts = iter((initial, refreshed))
    monkeypatch.setattr(
        orchestrator_module,
        "get_issue_context",
        lambda *args, **kwargs: next(contexts),
    )
    plan_output = structured_plan_state(summary="Plan the compatibility fix.").replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": [{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": "The plan preserves the public API."}]',
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the plan preserves the public API.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                summary="Plan approved.",
                human_requirements_resolved=True,
                human_requirement_dispositions=[
                    {
                        "requirement_id": "Requirement 1",
                        "disposition": "addressed",
                        "evidence": "The plan preserves the public API.",
                    }
                ],
            )
        ],
    )

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"added hr-",
    ):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path), plan_first=True)

    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_fails_closed_when_a_requirement_is_withdrawn(
    tmp_path, monkeypatch
):
    """A withdrawal at the live approval boundary is a changed requirement set.

    Every plan review and every carried exact-key approval was bound to the
    earlier surfaced requirement digest, so a withdrawn ID must stop and
    re-enter planning rather than proceed to implementation (#905, from #841).
    """
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )

    def _context(*requirements):
        return IssueContext(
            number=56,
            repo="OWNER/REPO",
            title="Issue",
            body="Issue body",
            url="https://github.com/OWNER/REPO/issues/56",
            comments=(),
            human_requirements=requirements,
        )

    contexts = iter(
        (_context(requirement_1, requirement_2), _context(requirement_1))
    )
    monkeypatch.setattr(
        orchestrator_module,
        "get_issue_context",
        lambda *args, **kwargs: next(contexts),
    )
    dispositions = [
        {
            "requirement_id": "Requirement 1",
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        },
        {
            "requirement_id": "Requirement 2",
            "disposition": "addressed",
            "evidence": "The plan preserves the audit trail.",
        },
    ]
    plan_output = structured_plan_state(summary="Plan the compatibility fix.").replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the plan preserves the public API.\n"
        "- Requirement 2: the plan preserves the audit trail.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                summary="Plan approved.",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
    )

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"withdrawn hr-",
    ):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path), plan_first=True)

    # The run stops before implementation: no PR and no coder implementation turn.
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_approved_plan_completion_recovery_carries_parent_and_child_requirements(
    tmp_path, monkeypatch
):
    parent_requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/55",
        body="Preserve the parent audit trail.",
    )
    child_requirement = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Preserve the child API.",
    )
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Child issue",
        body="Child issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(child_requirement,),
    )
    parent_context = IssueContext(
        number=55,
        repo="OWNER/REPO",
        title="Parent issue",
        body="Parent issue body",
        url="https://github.com/OWNER/REPO/issues/55",
        comments=(),
        human_requirements=(parent_requirement,),
    )
    captured = {}

    class _StopAfterCapture(Exception):
        pass

    def capture_policy(*args, **kwargs):
        captured["policy"] = kwargs["completion_recovery"]
        raise _StopAfterCapture

    monkeypatch.setattr(orchestrator_module, "_run_validated_agent", capture_policy)
    config = make_config(tmp_path)

    with pytest.raises(_StopAfterCapture):
        orchestrator_module._implement_approved_issue(
            FakeRunner(),
            issue_number=56,
            approved_plan="Approved plan.\n\n### Scope\n- Preserve both APIs.",
            config=config,
            memory=None,
            issue_context=issue_context,
            parent_issue_context=parent_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
        )

    policy = captured["policy"]
    assert policy.human_requirements == (parent_requirement, child_requirement)
    recovery_prompt = build_completion_recovery_prompt(
        config,
        approved_plan_context=policy.approved_plan_context,
        human_requirements=policy.human_requirements,
    )
    assert "Preserve the parent audit trail." in recovery_prompt
    assert "Preserve the child API." in recovery_prompt


def test_issue_loop_plan_first_revises_until_all_reviewers_approve(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan."),
            structured_plan_revision(summary="Revised plan with tests."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

def test_issue_loop_plan_first_stops_on_incomplete_review_without_coder_followup(tmp_path):
    """An explicit inability to review is an agent failure, not a new blocker."""
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Review incomplete; unable to verify the retry path without further inspection.",
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "Review incomplete" not in "\n".join(runner.comments)

def test_issue_loop_plan_first_resume_ignores_incomplete_wording_outside_summary(tmp_path):
    """A resumed structured review must be scanned the same way a fresh one is.

    Before the resumed-branch summary reconstruction fix, `summary` was
    rebuilt from the raw (unrendered) response text even when a structured
    review had already been parsed, so incomplete-review wording living in an
    unrelated field (here, `future_followups`) could leak into the summary
    candidate text and falsely trip the incomplete-review heuristic only
    after a resume.
    """
    round1_plan = "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_round1_comment = _attach_round_metadata(
        round1_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(round1_plan),
        ),
    )
    resumed_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            summary="Reviewed the updated plan; still recommend revisiting the retry approach later.",
            future_followups=["Could not confirm the retry path in CI end-to-end; revisit later."],
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(round1_plan),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_round1_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resumed_reviewer_comment},
        ],
        claude_outputs=[structured_plan_revision(summary="Revised plan addressing round 1 feedback.")],
        codex_outputs=[structured_plan_review(summary="Plan looks sound now.")],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1

def test_issue_loop_plan_first_resume_stops_on_incomplete_review_in_summary(tmp_path):
    """Genuine incomplete-review wording must still be caught after a resume."""
    round1_plan = "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_round1_comment = _attach_round_metadata(
        round1_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(round1_plan),
        ),
    )
    resumed_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            summary="Review incomplete; unable to confirm the retry path without further inspection.",
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(round1_plan),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_round1_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resumed_reviewer_comment},
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",), max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 0

@pytest.mark.parametrize(
    ("state", "blocking", "same_plan", "summary", "expected"),
    [
        ("blocking", (), (), "Unable to verify the plan without more context.", True),
        (
            "blocking",
            (ApprovedFollowup(reviewer="Codex", text="Add a migration step."),),
            (),
            "Unable to verify the plan without more context.",
            False,
        ),
        ("approved", (), (), "Unable to verify the plan without more context.", False),
    ],
)
def test_is_incomplete_plan_review(state, blocking, same_plan, summary, expected):
    parsed_review = ParsedPlanReview(
        state=state,
        summary=summary,
        items=PlanReviewItems(blocking=blocking, same_plan=same_plan, future=()),
        dispositions=(),
    )
    assert orchestrator_module._is_incomplete_plan_review(parsed_review) is expected

def test_is_incomplete_plan_review_matches_disposition_note():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="still blocking",
                note="Resolution could not be confirmed; plan and referenced files need review.",
            ),
        ),
    )
    assert orchestrator_module._is_incomplete_plan_review(parsed_review) is True


def test_is_incomplete_plan_review_allows_active_carried_disposition():
    parsed_review = ParsedPlanReview(
        state="blocking",
        summary="",
        items=PlanReviewItems(blocking=(), same_plan=(), future=()),
        dispositions=(
            ReviewItemDisposition(
                item_id="item-1",
                reviewer="Codex",
                disposition="blocking",
                note="Could not confirm the revised plan covers migration ordering.",
            ),
        ),
    )

    assert orchestrator_module._is_incomplete_plan_review(parsed_review) is False


def test_issue_loop_plan_review_allows_active_carried_blocking_disposition(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan.", plan_steps=["Document migration ordering."]),
            structured_plan_revision(
                summary="Added migration ordering.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                plan_steps=["Document migration ordering."],
            ),
            structured_plan_revision(
                summary="Clarified migration ordering.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                plan_steps=["Document migration ordering."],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Document migration ordering."],
            ),
            structured_plan_review(
                state="blocking",
                summary="Could not confirm the revised plan covers migration ordering.",
                prior_plan_item_dispositions=[{
                    "item_id": "item-1",
                    "disposition": "blocking",
                    "note": "Could not confirm the revised plan covers migration ordering.",
                }],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert any("Could not confirm the revised plan" in comment for comment in runner.comments)


def test_issue_loop_structured_plan_state_public_comment_renders_markdown_and_preserves_metadata(tmp_path):
    raw_structured_plan = structured_plan_state(
        summary="Plan the issue fix.",
        plan_steps=["Update the renderer.", "Add regression tests."],
        reviewer="Google Antigravity",
    )
    runner = FakeRunner(
        antigravity_outputs=[raw_structured_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, coder="antigravity", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    public_comment = runner.comments[0]
    assert public_comment.startswith("## Plan")
    assert "### Plan steps\n1. Update the renderer.\n2. Add regression tests." in public_comment
    assert '"kind": "plan_state"' not in _strip_round_metadata(public_comment)

    raw_comment = runner.issue_comments[0]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.canonical_plan == raw_structured_plan
    assert metadata.raw_structured_coder_response == raw_structured_plan


def test_issue_loop_accepts_fresh_v1_plan_without_recommendation_driven_routing(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=783, config=config, plan_first=True) == 0
    assert runner.issues == []
    assert any("AGENT_EXECUTION_RECOMMENDATION" in comment for comment in runner.comments)


def test_issue_loop_accepts_fresh_v1_applicable_matrix(tmp_path):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["risk_test_matrix"] = risk_test_matrix_prompt_examples()["applicable"]
    candidate = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = _FakeRunner(
        claude_outputs=[candidate],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=783, config=config, plan_first=True) == 0
    assert runner.issues == []


def test_fresh_v1_plan_host_resume_reuses_posted_round_without_new_agent_turn(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    initial_agent_commands = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] or cmd[:2] == ["codex", "exec"]
    ]
    coder_metadata = next(
        _decode_round_metadata(match.group("payload"))
        for comment in runner.issue_comments
        if (match := re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
            comment["body"],
        ))
        and _decode_round_metadata(match.group("payload")).role == "coder"
    )
    assert coder_metadata.execution_strategy_contract_version == 1

    # The host rerun sees the durable canonical plan, raw response, sidecar,
    # and metadata identity. It must resume the approved plan without asking
    # either agent to produce a competing round.
    runner.claude_outputs = []
    runner.codex_outputs = []
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed_agent_commands = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] or cmd[:2] == ["codex", "exec"]
    ]
    assert resumed_agent_commands == initial_agent_commands
    assert runner.issues == []


def test_plan_review_accepts_valid_response_file_after_nonzero_exit(tmp_path):
    artifact = structured_plan_review(
        state="approved", summary="The plan is sound.", reviewer="Anthropic Claude"
    )
    runner = FakeRunner(
        codex_outputs=[structured_plan_state(summary="Plan the issue fix.")],
        claude_outputs=[("Error: timeout waiting for response", 1)],
        public_response_outputs=["", artifact],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude", agent_max_retries=0)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1
    metadata = next(
        _decode_round_metadata(match.group("payload"))
        for comment in runner.issue_comments
        if (match := re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", comment["body"]))
        and _decode_round_metadata(match.group("payload")).role == "reviewer"
    )
    assert metadata.acquisition_outcome == "accepted_nonzero_exit"
    assert metadata.acquisition_returncode == 1

def test_issue_loop_plan_first_rejects_marker_only_plan_before_posting_or_reviewer_dispatch(tmp_path):
    markdown_plan = "Initial markdown plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    runner = FakeRunner(
        claude_outputs=[markdown_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, reviewer="codex")

    with pytest.raises(AgentLoopError, match="structured `plan_state` JSON object"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert runner.comments == []
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def test_issue_loop_plan_first_rejects_generic_implementation_plan_before_posting(tmp_path):
    generic_plan = json.dumps({
        "summary": "Plan the fix.",
        "implementation_plan": ["Update the parser.", "Add tests."],
    }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(claude_outputs=[generic_plan])
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="kind mismatch|missing required field"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert runner.comments == []
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)

def test_issue_loop_plan_revision_stores_raw_structured_metadata(tmp_path):
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan with tests.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "note": "Added the missing test step."}
                ],
                "plan_steps": ["Add the regression test.", "Run the focused suite."],
                "architecture_impact": {
                    "status": "unchanged",
                    "rationale": "No architectural contract changed.",
                    "affected_components": [], "dependencies": [],
                    "execution_data_flows": [], "persistence": [],
                    "public_contracts": [], "security_boundaries": [],
                    "canonical_document_action": "no-change",
                    "canonical_document_path": None,
                    "canonical_document_rationale": "",
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[
            _initial_plan_state(),
            raw_structured_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[2].startswith("## Revised plan")
    assert '"kind": "plan_revision"' not in _strip_round_metadata(runner.comments[2])
    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.raw_structured_coder_response == raw_structured_revision


def test_issue_loop_activates_semantic_revision_from_fresh_authenticated_base(tmp_path):
    fresh = structured_v1_plan_state()
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Apply the reviewed semantic decision.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan addresses the original finding.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "summary", "value": "Revised semantic plan."}
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.base_state_identity == base.state_identity
    assert metadata.raw_patch_provenance == {
        **patch,
        "prior_plan_item_dispositions": [{
            "item_id": "item-1",
            "disposition": "resolved",
            "note": "The revised plan addresses the original finding.",
        }],
    }
    assert metadata.assembled_plan_sidecar is not None
    assert "Revised semantic plan." in runner.comments[2]



def test_semantic_revision_replacing_architecture_impact_records_and_resumes(tmp_path):
    """#879: an architecture_impact replacement must round-trip its provenance."""
    fresh = structured_v1_plan_state()
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    architecture_impact = {
        "status": "changed",
        "rationale": "The revision adds a publication seam.",
        "affected_components": ["round_state.py"],
        "dependencies": [],
        "execution_data_flows": ["plan round -> metadata"],
        "persistence": ["round metadata"],
        "public_contracts": [],
        "security_boundaries": [],
        "canonical_document_action": "update",
        "canonical_document_path": "ARCHITECTURE.md",
        "canonical_document_rationale": "Record the new seam.",
    }
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Apply the reviewed semantic decision.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "architecture_impact",
                "value": architecture_impact,
            }
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.raw_patch_provenance is not None
    # The stored provenance must be JSON, and must equal a freshly
    # re-serialized patch: restart compares exactly these two values.
    assert metadata.raw_patch_provenance == json.loads(
        json.dumps(metadata.raw_patch_provenance)
    )
    reparsed = parse_plan_revision_patch(metadata.raw_patch_provenance)
    assert reparsed.to_payload() == metadata.raw_patch_provenance

    # The recorded round must resume without tripping the integrity check.
    resumed = _resume_plan_round(
        [
            type("Comment", (), {"body": comment["body"]})()
            for comment in runner.issue_comments[:3]
        ],
        configured_reviewers=("codex",),
    )
    assert resumed is not None


def test_semantic_revision_inherits_signed_requirement_dispositions(tmp_path):
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )
    fresh_payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    fresh_payload["human_requirement_dispositions"] = [{
        "requirement_id": requirement.requirement_id,
        "disposition": "addressed",
        "evidence": "The authenticated base preserves backward compatibility.",
    }]
    fresh = (
        json.dumps(fresh_payload)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the base preserves backward compatibility.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Change only the plan summary.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "summary", "value": "Summary changed."}
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the authenticated disposition remains valid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    direct_patch_text = (
        json.dumps({**patch, "prior_plan_item_dispositions": []})
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the authenticated disposition remains valid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    validated_patch = orchestrator_module._validate_plan_revision_patch_response(
        direct_patch_text,
        human_requirements=(requirement,),
        inherited_human_requirement_dispositions=base.plan.human_requirement_dispositions,
    )
    assert validated_patch.operations[0].field == "summary"
    assert orchestrator_module._current_plan_has_complete_human_requirement_dispositions(
        direct_patch_text,
        surfaced_requirement_ids=(requirement.requirement_id,),
        inherited_dispositions=base.plan.human_requirement_dispositions,
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
                human_requirements_resolved=True,
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.assembled_plan_sidecar["canonical_json"]["human_requirement_dispositions"] == [
        {
            "requirement_id": requirement.requirement_id,
            "disposition": "addressed",
            "evidence": "The authenticated base preserves backward compatibility.",
        }
    ]

def test_issue_loop_plan_revision_rejects_missing_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            structured_plan_revision(summary="Revised plan."),
            "Revised plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan still preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

def test_issue_loop_plan_revision_accepts_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            structured_plan_revision(
                summary="Revised plan.",
                human_requirements=(
                    f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                    "### Human requirements\n"
                    "- Requirement 1: the revised plan still preserves backward compatibility.\n"
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "Missing a regression test." in claude_calls[1][-1]
    assert len(runner.comments) == 5
    assert runner.comments[2].startswith("## Revised plan")

def test_issue_loop_plan_revision_repair_preserves_signed_human_requirements(tmp_path):
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )
    malformed_revision = (
        "### Prior plan review item dispositions\n"
        "- item-1: resolved by adding compatibility tests.\n\n"
        "### Revised plan\n"
        "- Preserve backward compatibility.\n"
        "- Add regression tests.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_revision = _add_default_requirement_disposition(structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        prior_plan_item_dispositions=[
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "note": "Added compatibility tests.",
            }
        ],
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
        human_requirements=(
            f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan preserves backward compatibility.\n"
        ),
    ), requirement_id=requirement.requirement_id)
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append((raw, expected_kind))
        return repaired_revision

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(captured_repairs) == 1
    assert captured_repairs[0][1] == "plan_revision"
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in captured_repairs[0][0]
    public_revision = _strip_round_metadata(runner.comments[2])
    assert '"kind": "plan_revision"' not in public_revision
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in public_revision
    assert "### Human requirements" in public_revision

def test_issue_loop_plan_revision_repair_rejects_wrong_kind_from_human_requirements_text(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    wrong_kind_repair = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Revised the plan.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirement_dispositions": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_kinds = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_kinds.append(expected_kind)
        return wrong_kind_repair

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_attempt_repair):
        with pytest.raises(AgentLoopError, match="expected `plan_revision`"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert captured_kinds == ["plan_revision"]

def test_issue_loop_plan_revision_repair_without_human_ack_fails_clearly(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_without_ack = structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with patch("coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired_without_ack):
        with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

def test_issue_loop_plan_first_requires_reviewers_to_disposition_prior_items(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n- Add parser validation tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs the test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved plan items"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

def test_issue_loop_plan_first_carries_same_plan_item_across_reviewers_and_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs one plan refinement."
            + prior_plan_item_dispositions("[item-1] same-plan: still need the mixed-reviewer case")
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Plan looks sound now."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Final pass."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("item-1" in call[-1] for call in claude_calls[1:])
    assert "Approved plan:" in runner.comments[-1]

def test_issue_loop_plan_first_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n"
            "- Keep plan-review wording distinct from PR wording.\n"
            "### Same-plan follow-ups\n"
            "- Add one carry-forward plan test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=2)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[1] == (
        "**Review verdict:** Blocking\n\n"
        "### Blocking plan issues\n"
        "- Keep plan-review wording distinct from PR wording.\n"
        "\n"
        "### Same-plan follow-ups\n"
        "- Add one carry-forward plan test.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    assert runner.comments[3] == (
        "**Review verdict:** Approved\n\n"
        "Plan looks sound.\n\n"
        "### Prior unresolved plan item dispositions\n"
        "- [item-1] RESOLVED\n"
        "  - Original finding: Blocking issue from OpenAI Codex, round 1: Keep plan-review wording distinct from PR wording.\n"
        "- [item-2] RESOLVED\n"
        "  - Original finding: Same-plan follow-up from OpenAI Codex, round 1: Add one carry-forward plan test.\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )

def test_issue_loop_plan_first_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "### Same-plan follow-ups\n"
            "- Add the carry-forward orchestration test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
        ],
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer=("gemini", "claude"),
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking plan issues after round 1"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "planning round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved plan items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[1]

def test_issue_loop_plan_first_uses_compact_context_after_round_one(tmp_path, capsys):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Resolve item one.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round one."],
            ),
            structured_plan_revision(
                summary="Resolve item two.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round two."],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round one issue."],
            ),
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round two issue."],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=3,
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    round_one_review = next(prompt for prompt in prompts if "planning round 1" in prompt)
    round_two_review = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: reviewer" in prompt)
    round_two_revision = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: coder" in prompt)
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in round_one_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_revision

    captured = capsys.readouterr()
    assert "Planning issue #56: invoking Claude (context mode: full)" in captured.err
    assert "Planning round 2: Codex reviewing issue #56 (context mode: compact)" in captured.err
    assert "Planning round 2: Claude revising the plan (context mode: compact)" in captured.err

def test_issue_loop_plan_first_requires_reviewer_human_requirements_resolution(tmp_path, capsys):
    runner = FakeRunner(
        issue_payload={
            "body": "Keep compact context cache-aware.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(
                summary="Initial plan covers cache-aware compact context.",
                human_requirements=(
                    "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
                    "### Human requirements\n"
                    "- Requirement 1: The plan keeps compact context cache-aware."
                ),
            ),
            structured_plan_revision(
                summary="Revised plan requires explicit reviewer acknowledgement.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Reviewer must acknowledge."}
                ],
                plan_steps=["Keep the compact context cache-aware and require reviewer acknowledgement."],
                human_requirements=(
                    "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
                    "### Human requirements\n"
                    "- Requirement 1: The revised plan covers the cache-aware compact context requirement."
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Acknowledged."}
                ],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=2,
        plan_execution_mode="plan-only",
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert "Approved plan:" in runner.comments[-1]
    assert any(
        "approved without acknowledging the signed human requirements" in comment
        for comment in runner.comments
    )
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in runner.comments[-2]
    captured = capsys.readouterr()
    assert "approved without acknowledging signed human requirements" in captured.err


def test_plan_first_grafana_requirement_blocks_admin_html_only_plan_then_accepts_integration(tmp_path):
    """A reviewer must reject a lookalike local UI until the named integration is planned."""
    runner = FakeRunner(
        issue_payload={
            "body": "Provision a Grafana dashboard for verification attribution.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _plan_with_requirement_disposition(
                kind="plan_state",
                summary="Add verification attribution to the operations UI.",
                plan_steps=["Add verification attribution to admin.html."],
                evidence="admin.html will display verification attribution.",
            ),
            _plan_with_requirement_disposition(
                kind="plan_revision",
                summary="Provision the requested Grafana dashboard.",
                plan_steps=[
                    "Add Grafana dashboard provisioning for verification attribution.",
                    "Test the dashboard provisioning artifact.",
                ],
                evidence="The plan names Grafana dashboard provisioning and its artifact.",
            ),
        ],
        codex_outputs=[
            _plan_review_with_requirement_disposition(
                state="blocking",
                summary="admin.html does not cover the requested Grafana dashboard provisioning.",
            ),
            _plan_review_with_requirement_disposition(
                state="approved",
                summary="The revised canonical plan covers the Grafana provisioning artifact.",
                marker=True,
                prior_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Grafana is now planned."}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=2,
        plan_execution_mode="plan-only",
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    review_prompts = [
        command[-1]
        for command, _cwd in runner.commands
        if command[:2] == ["codex", "exec"]
    ]
    assert any("admin.html" in prompt and "Grafana" in prompt for prompt in review_prompts)
    assert "admin.html does not cover the requested Grafana" in runner.comments[1]
    assert "Provision the requested Grafana dashboard" in runner.comments[-1]

def test_issue_loop_plan_first_uses_full_context_when_plan_ledger_incomplete(tmp_path, capsys):
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item that could be missed.",
        status="blocking",
        source_status="blocking",
    )
    old_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=["Old subject item that could be missed."],
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(old_plan),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    latest_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(new_plan),
            prior_items=(),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_reviewer_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:10:00Z", "body": latest_coder_comment},
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "planning round 2" in cmd[-1]
    ][0]
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in review_prompt
    captured = capsys.readouterr()
    assert "Planning round 2: Codex reviewing issue #56 (context mode: full (ledger incomplete))" in captured.err

def test_issue_loop_plan_review_strips_resolved_history_disposition_under_incomplete_ledger(
    tmp_path, monkeypatch
):
    # #862 plan-flow analog: an earlier plan subject raised and resolved
    # item-1, the carried set is empty, and the reviewer repeats item-1 as
    # resolved.  The no-op entry is stripped instead of failing the run.
    # This is the compatibility-default policy, so the resolved-only history
    # must still read as an incomplete ledger and still select full context:
    # staged planning's durable cleared-item proof does not apply here (#905).
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    mid_plan = "Mid plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item.",
        status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(state="blocking", blocking_plan_issues=["Old subject item."]),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(old_plan), new_items=(old_item,), state="blocking",
        ),
    )
    resolved = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(mid_plan), prior_items=(old_item,),
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "resolved"),),
            state="blocking",
        ),
    )
    latest_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=3,
            subject=_plan_subject(new_plan), prior_items=(),
        ),
    )
    ledger_flags = []
    real_ledger_check = orchestrator_module._round_ledger_may_be_incomplete

    def ledger_spy(**kwargs):
        result = real_ledger_check(**kwargs)
        ledger_flags.append(result)
        return result

    monkeypatch.setattr(orchestrator_module, "_round_ledger_may_be_incomplete", ledger_spy)
    logged = []
    real_log = orchestrator_module.log
    monkeypatch.setattr(
        orchestrator_module,
        "log",
        lambda config, message, *a, **k: (logged.append(message), real_log(config, message, *a, **k))[1],
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": raised},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resolved},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:10:00Z", "body": latest_coder_comment},
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",))

    with patch("coding_review_agent_loop.orchestrator.attempt_repair") as repair_mock:
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    repair_mock.assert_not_called()
    assert ledger_flags and ledger_flags[0] is True
    assert any(
        "removed canonically resolved historical prior-item disposition ID(s) item-1 "
        "despite incomplete ledger" in message
        for message in logged
    )
    # The default path still takes the conservative full-context branch it
    # took before staged planning existed.
    assert any("(context mode: full (ledger incomplete))" in message for message in logged)

def test_round_resolved_history_ids_available_when_plan_ledger_is_complete():
    # #872 follow-up: a lossless semantic patch needs resolved-history proof in
    # every ledger state, so the history is computed for complete rounds too.
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    mid_plan = "Mid plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item.",
        status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(state="blocking", blocking_plan_issues=["Old subject item."]),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(old_plan), new_items=(old_item,), state="blocking",
        ),
    )
    resolved = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(mid_plan), prior_items=(old_item,),
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "resolved"),),
            state="blocking",
        ),
    )
    comments = [SimpleNamespace(body=raised), SimpleNamespace(body=resolved)]

    assert orchestrator_module._round_resolved_history_item_ids(
        prior_unresolved_items=(),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-1",)


def test_post_round_resolved_history_ids_cover_the_round_still_in_progress():
    # #874: the snapshot a round holds was fetched before its review turns, so
    # it cannot record the dispositions that round just applied. The proof for
    # the revision therefore also reads the round's in-process dispositions.
    plan = "Current plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    blocking = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Migration ordering is unspecified.", status="blocking",
        source_status="blocking",
    )
    cleared = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="Rollback path is uncovered.", status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=[blocking.text, cleared.text],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(plan), new_items=(blocking, cleared), state="blocking",
        ),
    )
    # Only round 1 is in the snapshot; round 2's reviewer comment is posted
    # after it was fetched, exactly as the live loop sees it.
    comments = [SimpleNamespace(body=raised)]
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "blocking", "Still unspecified.")],
        "item-2": [ReviewItemDisposition("item-2", "OpenAI Codex", "resolved")],
    }

    assert orchestrator_module._round_resolved_history_item_ids(
        prior_unresolved_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()
    assert orchestrator_module._post_round_resolved_history_item_ids(
        prior_unresolved_items=(blocking, cleared),
        dispositions_by_item=dispositions,
        carried_items=(blocking,),
        comments=comments,
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ("item-2",)


def test_post_round_resolved_history_ids_exclude_actively_disputed_and_carried_items():
    # The whitelist stays fail-closed: a mixed verdict, a still-carried item and
    # an undispositioned item never earn resolved-history proof.
    carried = UnresolvedReviewItem(
        item_id="item-1", reviewer="OpenAI Codex", source_round=1,
        text="Still blocking.", status="blocking", source_status="blocking",
    )
    mixed = UnresolvedReviewItem(
        item_id="item-2", reviewer="OpenAI Codex", source_round=1,
        text="One reviewer still blocks.", status="blocking", source_status="blocking",
    )
    untouched = UnresolvedReviewItem(
        item_id="item-3", reviewer="OpenAI Codex", source_round=1,
        text="Nobody dispositioned this.", status="blocking", source_status="blocking",
    )
    dispositions = {
        "item-1": [ReviewItemDisposition("item-1", "OpenAI Codex", "resolved")],
        "item-2": [
            ReviewItemDisposition("item-2", "OpenAI Codex", "resolved"),
            ReviewItemDisposition("item-2", "Anthropic Claude", "blocking", "Not covered."),
        ],
    }

    assert orchestrator_module._post_round_resolved_history_item_ids(
        prior_unresolved_items=(carried, mixed, untouched),
        dispositions_by_item=dispositions,
        # item-1 survived reconciliation, so it is still carried.
        carried_items=(carried,),
        comments=[],
        flow="plan",
        reconciliation_mode="aggregate",
        same_status="same-plan",
    ) == ()


def test_issue_loop_plan_revision_proof_covers_items_resolved_in_the_same_round(
    tmp_path, monkeypatch
):
    # #874 orchestration regression: the initial issue context lacks the review
    # record that resolves item-2, because that comment is posted during the
    # round. The revision turn issued at the end of that round must still prove
    # item-2 is resolved history.
    revision_history_ids = []
    revision_snapshot_sizes = []
    snapshots = []
    real_run_validated_agent = orchestrator_module._run_validated_agent

    def run_validated_agent_spy(*args, **kwargs):
        if kwargs.get("operation_description") == "plan revision":
            revision_history_ids.append(
                tuple(kwargs.get("repair_resolved_history_item_ids") or ())
            )
            revision_snapshot_sizes.append(
                tuple(len(context.comments) for context in snapshots)
            )
        return real_run_validated_agent(*args, **kwargs)

    monkeypatch.setattr(
        orchestrator_module, "_run_validated_agent", run_validated_agent_spy
    )
    real_get_issue_context = orchestrator_module.get_issue_context

    def get_issue_context_spy(*args, **kwargs):
        context = real_get_issue_context(*args, **kwargs)
        snapshots.append(context)
        return context

    monkeypatch.setattr(
        orchestrator_module, "get_issue_context", get_issue_context_spy
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                summary="Initial plan.", plan_steps=["Document migration ordering."]
            ),
            structured_plan_revision(
                summary="Addressed both findings.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_plan_revision(
                summary="Kept the still-blocking finding open.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=[
                    "Document migration ordering.",
                    "Cover the rollback path.",
                ],
            ),
            structured_plan_review(
                state="blocking",
                summary="Rollback path is covered; migration ordering still is not.",
                prior_plan_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "blocking",
                        "note": "Migration ordering is still unspecified.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    # Both revisions saw only the empty pre-loop fetch, so the recorded history
    # alone could not have supplied the item-2 proof.
    assert revision_snapshot_sizes == [(0,), (0,)]
    assert revision_history_ids[0] == ()
    assert revision_history_ids[1] == ("item-2",)


def test_issue_loop_plan_first_resumes_with_only_missing_reviewer_for_current_plan(tmp_path):
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(current_plan),
            prior_items=(),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=_plan_subject(current_plan),
            state="approved",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
        ],
        gemini_outputs=["Plan looks sound too.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    assert runner.comments[-1].startswith("Planning complete for issue #56.")

@pytest.mark.parametrize("persisted_as_machine", [False, True])
def test_issue_loop_plan_first_resume_clears_orchestrator_item_on_unanimous_approval(
    tmp_path, persisted_as_machine
):
    """An orchestrator-authored plan item stays reviewer-clearable after resume (#1005).

    Recovery used to promote it to an ``unknown`` machine obligation, which no
    planning participant can clear, so every unanimous approval was followed by
    another revision forever.  A ledger that already persisted the promoted
    form must be recoverable too.
    """
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    orchestrator_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Orchestrator",
        source_round=1,
        text="Reviewer(s) Codex approved without acknowledging the signed human requirements.",
        status="blocking",
        source_status="blocking",
    )
    if persisted_as_machine:
        orchestrator_item = replace(
            orchestrator_item,
            authority="unknown",
            obligation_kind="unknown",
            lifecycle="repair_required",
            obligation_identity="unknown:item-1",
        )
    summary_comment = _attach_round_metadata(
        "Orchestrator plan review.",
        PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject=subject,
            new_items=(orchestrator_item,),
        ),
    )
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(orchestrator_item,),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": summary_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
        ],
        codex_outputs=[
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved: acknowledged in the revised plan")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == ["codex"]
    assert runner.comments[-1].startswith("Planning complete for issue #56.")


def test_issue_loop_plan_first_stops_when_only_unclearable_items_block_approval(
    tmp_path, monkeypatch
):
    """Defense in depth (#1005): never revise an approved plan for an item no one can clear."""
    import coding_review_agent_loop.round_state as round_state_module

    monkeypatch.setattr(round_state_module, "_demote_plan_machine_items", tuple)
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    machine_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Orchestrator",
        source_round=1,
        text="Synthetic orchestrator blocker.",
        status="blocking",
        source_status="blocking",
        authority="unknown",
        obligation_kind="unknown",
        lifecycle="repair_required",
        obligation_identity="unknown:item-1",
    )
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(machine_item,),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
        ],
        codex_outputs=[
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved: nothing remains")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=3)

    with pytest.raises(AgentLoopError, match=r"item-1 \(owner Orchestrator, kind unknown\)"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-plan: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_issue_loop_plan_first_rejects_contradictory_disposition_before_extra_revision(
    tmp_path, line
):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound now."
            + prior_plan_item_dispositions(line)
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2

def test_issue_loop_plan_first_plan_only_does_not_publish_approved_future_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Tighten the prompt wording.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_plan_item_dispositions("[item-1] future follow-up: document parser helper reuse separately")
            + "\n### Future follow-ups\n- Add a later cleanup to dedupe shared prompt rendering.\n"
            + "<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    summary = runner.comments[-1]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in summary
    assert "document parser helper reuse separately" in summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in summary
    assert "not carried into PR review" in summary
    assert "not PR prior review items" in summary
    assert "Filed future follow-up issues:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary
    assert "mode=summarize" in summary

def test_issue_loop_plan_first_files_approved_future_followups_before_implementation(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(tmp_path, approved_followups="issue")

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

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == (
        "Follow up future plan-review note: Add a later cleanup to dedupe shared prompt rendering."
    )
    issue_body = runner.issues[0]["body"]
    assert "Parent issue: #56" in issue_body
    assert "Approved plan hash:" in issue_body
    assert "Planning round(s): 1" in issue_body
    assert "Original plan item ID(s): item-1" in issue_body
    assert "Codex" in issue_body
    assert "outside the current implementation scope" in issue_body
    assert "not a PR-review prior item" in issue_body
    summary = runner.comments[2]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Filed future follow-up issues:" in summary
    assert "https://github.com/OWNER/REPO/issues/99" in summary
    assert "Approved plan future follow-ups:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    second_claude_index = command_index(
        runner.commands,
        ["claude", "--print"],
        start=command_index(runner.commands, ["claude", "--print"]) + 1,
    )
    assert issue_create_index < second_claude_index

def test_issue_loop_plan_first_does_not_allocate_new_items_for_repeated_carried_future_followups(tmp_path):
    future_text = "Factor the shared follow-up guidance into a reusable helper."
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Address the blocking test gap.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Added the test."}
                ],
                plan_steps=["Make the change.", "Add the missing regression test."],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "future",
                        "note": "Keep this as confirmed post-plan cleanup.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add a regression test for the plan-review ledger."],
                reviewer="Google Gemini",
            ),
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Google Gemini",
            ),
            structured_pr_review(
                state="approved",
                summary="LGTM.",
                reviewer="Google Gemini",
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=2,
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

    coder_revision_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
    )
    assert "item-2" in coder_revision_prompt
    assert "item-1" not in coder_revision_prompt
    assert future_text not in coder_revision_prompt

    round_two_reviewer_prompts = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert len(round_two_reviewer_prompts) == 2
    assert all("item-1" in prompt for prompt in round_two_reviewer_prompts)

    assert len(runner.issues) == 1
    issue_body = runner.issues[0]["body"]
    assert "Planning round(s): 1" in issue_body
    assert "Reviewer: Codex" in issue_body
    assert "Original plan item ID(s): item-1" in issue_body
    assert "Keep this as confirmed post-plan cleanup." in issue_body
    assert any(
        "Reconciliation: 1 filed, 0 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    round_two_review_indexes = [
        index
        for index, (cmd, _cwd) in enumerate(runner.commands)
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert issue_create_index > max(round_two_review_indexes)

@pytest.mark.parametrize("later_disposition", ["resolved", "same-plan", "blocking"])
def test_issue_loop_plan_first_does_not_file_future_item_after_later_lifecycle_change(
    tmp_path, later_disposition
):
    future_text = "Extract the shared plan-review formatting helper."
    promoted = later_disposition in {"same-plan", "blocking"}
    promotion_state = "blocking" if promoted else "approved"
    final_plan_dispositions = [
        {"item_id": "item-1", "disposition": later_disposition},
        {"item_id": "item-2", "disposition": "resolved"},
    ]
    claude_outputs = [
        "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        structured_plan_revision(
            summary="Address the original blocker.",
            prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        ),
    ]
    if promoted:
        claude_outputs.append(
            structured_plan_revision(
                summary="Address the promoted future item.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            )
        )
    claude_outputs.append(
        "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    def reviewer_outputs(reviewer):
        outputs = [
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                reviewer=reviewer,
            ),
            structured_plan_review(
                state=promotion_state,
                prior_plan_item_dispositions=final_plan_dispositions,
                reviewer=reviewer,
            ),
        ]
        if promoted:
            outputs.append(
                structured_plan_review(
                    state="approved",
                    prior_plan_item_dispositions=[
                        {"item_id": "item-1", "disposition": "resolved"}
                    ],
                    reviewer=reviewer,
                )
            )
        outputs.append(structured_pr_review(state="approved", reviewer=reviewer))
        return outputs

    runner = FakeRunner(
        claude_outputs=claude_outputs,
        codex_outputs=reviewer_outputs("OpenAI Codex"),
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add the initial ledger regression."],
                reviewer="Google Gemini",
            ),
            *reviewer_outputs("Google Gemini")[1:],
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=3,
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

    assert runner.issues == []
    assert not any(future_text in comment for comment in runner.comments[-2:])
    if promoted:
        coder_revision_prompts = [
            cmd[-1]
            for cmd, _cwd in runner.commands
            if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
        ]
        assert len(coder_revision_prompts) == 2
        assert "item-1" in coder_revision_prompts[1]
        assert future_text in coder_revision_prompts[1]
        final_round_review_indexes = [
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if "Planning round: 3" in cmd[-1] and "Role: reviewer" in cmd[-1]
        ]
        implementation_index = next(
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if cmd[:1] == ["claude"] and "Implement the approved plan" in cmd[-1]
        )
        assert implementation_index > max(final_round_review_indexes)

def test_issue_loop_plan_first_ignore_mode_keeps_pr_prior_ledger_clean(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Track a separate planning cleanup later."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
    )
    config = make_config(tmp_path, approved_followups="ignore")

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

    assert runner.issues == []
    planning_summary = runner.comments[2]
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Track a separate planning cleanup later." in planning_summary
    assert "not carried into PR review" in planning_summary
    pr_review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and '"kind": "pr_review"' in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in pr_review_prompt
    ledger_start = pr_review_prompt.index("Prior unresolved review items from earlier rounds")
    assert "Track a separate planning cleanup later." not in pr_review_prompt[ledger_start:]
    assert "planning-stage `item-*` IDs and approved\nplan future follow-ups" in pr_review_prompt
    assert "prior_plan_item_dispositions" in pr_review_prompt

def test_issue_loop_plan_first_deduplicates_plan_followup_issues_across_reviewers(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        claude_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(state="approved", summary="LGTM.", reviewer="Anthropic Claude"),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="issue",
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

    assert len(runner.issues) == 1
    body = runner.issues[0]["body"]
    assert "Reviewers: Codex, Claude" in body
    assert "Original plan item ID(s): item-1, item-2" in body
    assert body.count("**Remote validation**") == 3
    assert any(
        "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )

def test_issue_loop_plan_first_plan_followup_marker_prevents_duplicate_issue_creation(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    plan_hash = approved_plan_hash(plan)
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a later cleanup to dedupe shared prompt rendering.",
        status="future",
        source_status="future",
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_pr_review(state="approved", summary="LGTM."),
        ],
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
                    structured_plan_review(
                        state="approved",
                        future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
                    ),
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        new_items=(future_item,),
                        state="approved",
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:02Z",
                "body": (
                    "Planning complete for issue #56.\n\n"
                    "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: "
                    f"issue=56 plan={plan_hash} mode=issue -->\n"
                    "-- coding-review-agent-loop"
                ),
            },
        ],
    )
    config = make_config(tmp_path, approved_followups="issue")

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

    assert runner.issues == []
    assert not any(
        "Filed future follow-up issues:" in comment or "Approved plan future follow-ups:" in comment
        for comment in runner.comments
    )

def test_issue_loop_plan_first_keeps_blocking_review_when_future_followups_are_misclassified(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan with focused tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n\n"
            "### Blocking plan issues\n"
            "- Add parser coverage for blocking reviews with stray future follow-ups.\n\n"
            "### Same-plan follow-ups\n"
            "- Tighten the plan-review prompt wording.\n\n"
            "### Future follow-ups\n"
            "- Consider a later prompt dedupe cleanup.\n\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n"
            "-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert "Add parser coverage for blocking reviews with stray future follow-ups." in claude_calls[1][-1]
    assert "Tighten the plan-review prompt wording." in claude_calls[1][-1]
    assert runner.comments[1].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "### Future follow-ups" not in runner.comments[1]

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

    monkeypatch.setattr(orchestrator_module, "get_issue_context", issue_context_for)
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


def test_removed_canonical_handoff_does_not_recreate_from_incidental_pr(tmp_path):
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
        open_prs_payload=[
            {"number": 492, "body": "#56's loop is running; this is only a sequencing note."}
        ],
    )
    config = make_config(tmp_path)

    first_context = get_issue_context(runner, config=config, issue_number=56)
    first = resolve_canonical_pr_for_issue(
        runner, config=config, issue_number=56, issue_context=first_context
    )
    assert first is not None
    assert first.source == "canonical"
    assert first.pr_number == 77

    runner.issue_comments.clear()
    second_context = get_issue_context(runner, config=config, issue_number=56)
    assert (
        resolve_canonical_pr_for_issue(
            runner, config=config, issue_number=56, issue_context=second_context
        )
        is None
    )


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


def test_codex_issue_loop_creates_pr_then_claude_approves(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    command_names = [cmd[:2] for cmd, _cwd in runner.commands]
    assert ["codex", "exec"] in command_names
    assert ["claude", "--print"] in command_names
    assert len(runner.comments) == 3
    assert "<!-- AGENT_ISSUE_PR_HANDOFF:" in runner.comments[0]
    assert runner.comments[1].startswith("## Issue implementation")
    assert runner.comments[2].startswith("**Review verdict:** Approved\n\nLooks good.")

def test_codex_issue_loop_alternates_until_claude_approval(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Implemented fix.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Addressed Claude's review.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Missing test.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    assert len(runner.comments) == 5
    assert runner.comments[-1].startswith("**Review verdict:** Approved\n\nLGTM.")

def test_codex_issue_loop_requires_codex_to_report_pr_number(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Did some work.\n<!-- AGENT_STATE: blocking -->"],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="valid PR"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("terminal_marker, expected_state", [
    ("<!-- AGENT_STATE: blocking -->", "blocking"),
    ("<!-- AGENT_CLARIFY -->", "clarification"),
])
def test_issue_loop_stops_before_pr_lookup_for_invalid_pr_terminal_result(
    tmp_path, terminal_marker, expected_state
):
    runner = FakeRunner(codex_outputs=[f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}"])
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match=f"implementation is {expected_state}"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    # A genuine coder-declared no-PR terminal is posted to the issue like a
    # PR-success implementation result already is (#588).
    if expected_state == "blocking":
        assert runner.comments[0].startswith("## Issue implementation")
        assert "Cannot proceed." in runner.comments[0]
        assert "No pull request was accepted for handoff." in runner.comments[0]
    else:
        assert runner.comments == [
            f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}\n-- OpenAI Codex: unknown model (medium)"
        ]


def test_issue_loop_invalid_pr_without_terminal_state_is_protocol_error(tmp_path):
    runner = FakeRunner(codex_outputs=["Cannot proceed.\n<!-- AGENT_PR: malformed -->"])
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(
        AgentLoopError,
        match="required structured `issue_implementation` format",
    ):
        run_issue_loop(runner, issue_number=56, config=config)

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)

def test_issue_loop_rejects_outside_workdir_tests_before_posting_pr_comment(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: cd ~/llm-dialectic && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="outside the assigned checkout"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_outside_workdir_after_reported_pr_mentions_confirmed_resume(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: cd /outside && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "outside the assigned checkout" in message
    assert "PR #77 was confirmed open" in message
    assert "handoff/reviewer comments were not posted" in message
    assert "agent-loop pr 77" in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_managed_issue_invalid_post_pr_report_persists_authorization_before_rejection(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest"],
                test_observations=[{
                    "command": "cd /outside && python -m pytest",
                    "receipt_id": "outside-receipt",
                    "claim": "current-result",
                }],
                reviewer="OpenAI Codex",
            )
        ],
        claude_outputs=[
            structured_pr_review(
                state="approved",
                summary="Reviewed the post-report recovery head.",
                reviewer="Anthropic Claude",
            )
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module, "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []

    # A later ordinary managed issue invocation discovers and enters the real
    # recovery/activation seam for the same head instead of invoking the
    # implementation coder again.
    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
    )
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    reviewer_command = next(
        command for command, _cwd in runner.commands if command[:1] == ["claude"]
    )
    assert "abc123" in " ".join(reviewer_command)
    assert any(
        "Reviewed the post-report recovery head." in comment
        for comment in runner.comments
    )
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF",
            "AGENT_TEST_OBSERVATION", "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_approved_plan_invalid_post_pr_observation_keeps_authorization_resumable(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            structured_issue_implementation(
                pr_number=77,
                tests_run=["python3 -m pytest"],
                test_observations=[{
                    "command": "cd /outside && python -m pytest",
                    "receipt_id": "outside-receipt",
                    "claim": "current-result",
                }],
                reviewer="OpenAI Codex",
            )
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
    )
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="voluntary",
        audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module, "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError) as exc_info:
        orchestrator_module._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Plan:\n- Preserve managed recovery.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
        )

    message = str(exc_info.value)
    assert "structured test-observation report was invalid" in message
    assert "PR #77 was confirmed open" in message
    assert "authorization checkpoint" in message
    assert "agent-loop pr 77" in message
    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0


_RECOVERY_WORKFLOW = """
# agent-loop-managed
# expected_head_sha
# AGENT_LOOP_MANAGED_CI_V2
# AGENT_LOOP_MANAGED_CI_UNLABELED_RECOVERY_V1
name: CI
on:
  pull_request:
    types: [opened, unlabeled]
  workflow_dispatch:
    inputs:
      protocol_version: {required: true}
      pr_number: {required: true}
      expected_head_sha: {required: true}
      managed_nonce: {required: true}
jobs:
  aggregate:
    name: final-ci/exact-head
"""


class _IssueRecoveryWorkflowRunner(FakeRunner):
    """Model the server-backed issue/PR tuple used by real recovery seams."""

    def __init__(self, *, labeled, authorization_comments=None, **kwargs):
        body = (
            f"Fixes #56\n\n{UNPROTECTED_OVERRIDE_TRAILER} "
            "nonce=opening-nonce"
        )
        pr_payload = {
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Managed recovery",
            "body": body,
            "headRefName": "agent-loop/managed-56",
            "baseRefName": "main",
            "headRefOid": "abc123",
            "comments": [],
            "reviews": [],
        }
        pr_payload.update(kwargs.pop("pr_payload", {}))
        super().__init__(pr_payload=pr_payload, **kwargs)
        self.rest_pr = {
            "state": "open",
            "draft": True,
            "labels": [{"name": "agent-loop-managed"}] if labeled else [],
            "body": body,
            "head": {
                "repo": {"full_name": "OWNER/REPO"},
                "sha": "abc123",
                "ref": "agent-loop/managed-56",
            },
            "base": {"ref": "main"},
            "user": {"login": "agent-loop", "id": 1},
        }
        self.authorization_comments = list(authorization_comments or [])
        self.issue_events = [{
            "id": 101,
            "event": "labeled",
            "label": {"name": "agent-loop-managed"},
            "actor": {"login": "agent-loop", "id": 1},
        }]
        self.labels_posted = False
        self.dispatch_count = 0

    @staticmethod
    def _form_value(command, name):
        prefix = f"{name}="
        return next(
            (part[len(prefix):] for part in command if part.startswith(prefix)),
            None,
        )

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        endpoint = next(
            (part for part in command if part.startswith("repos/")), ""
        )
        if command == ["gh", "api", "user"]:
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"login": "agent-loop", "id": 1}), "", 0
            )
        if endpoint.endswith("/actions/variables/AGENT_LOOP_MANAGED_ACTOR"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"value": "agent-loop"}), "", 0
            )
        if endpoint.startswith(
            "repos/OWNER/REPO/contents/.github/workflows/ci.yml"
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, _RECOVERY_WORKFLOW, "", 0)
        if endpoint == "repos/OWNER/REPO/pulls/77":
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, json.dumps(self.rest_pr), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/77/events?"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps(self.issue_events), "", 0
            )
        if endpoint.startswith("repos/OWNER/REPO/issues/77/comments?"):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps(self.authorization_comments), "", 0
            )
        if endpoint == "repos/OWNER/REPO/issues/77/comments" and "POST" in command:
            recorded, cwd_path = self._record_command(args, cwd)
            body = self._form_value(command, "body") or ""
            comment_id = max(
                (comment["id"] for comment in self.authorization_comments),
                default=40,
            ) + 1
            comment = {
                "id": comment_id,
                "body": body,
                "user": {"login": "agent-loop", "id": 1},
            }
            self.authorization_comments.append(comment)
            self.pr_payload.setdefault("comments", []).append({
                "author": {"login": "agent-loop"},
                "createdAt": f"2026-05-23T00:00:{comment_id:02d}Z",
                "body": body,
            })
            return CommandResult(recorded, cwd_path, json.dumps(comment), "", 0)
        if endpoint.startswith("repos/OWNER/REPO/issues/56/timeline?"):
            recorded, cwd_path = self._record_command(args, cwd)
            timeline = [{
                "event": "cross-referenced",
                "source": {
                    "issue": {
                        "number": 77,
                        "repository_url": "https://api.github.test/repos/OWNER/REPO",
                        "pull_request": {"url": "https://api.github.test/pulls/77"},
                    }
                },
            }]
            return CommandResult(recorded, cwd_path, json.dumps(timeline), "", 0)
        if endpoint == "repos/OWNER/REPO/issues/77/labels" and "POST" in command:
            recorded, cwd_path = self._record_command(args, cwd)
            self.labels_posted = True
            self.rest_pr["labels"] = [{"name": "agent-loop-managed"}]
            return CommandResult(recorded, cwd_path, "{}", "", 0)
        if endpoint == "repos/OWNER/REPO/commits/main":
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(
                recorded, cwd_path, json.dumps({"sha": "base-sha"}), "", 0
            )
        if endpoint.endswith("/actions/workflows/ci.yml/dispatches"):
            self.dispatch_count += 1
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _MalformedAuthorizationResponseRunner(_IssueRecoveryWorkflowRunner):
    """Return a successful but unverifiable response for the auth comment POST."""

    def _run_locked(self, args, *, cwd, check, input_text=None):
        command = list(args)
        endpoint = next(
            (part for part in command if part.startswith("repos/")), ""
        )
        body = self._form_value(command, "body") or ""
        if (
            endpoint == "repos/OWNER/REPO/issues/77/comments"
            and "POST" in command
            and "ISSUE_AUTHORIZATION" in body
        ):
            recorded, cwd_path = self._record_command(args, cwd)
            return CommandResult(recorded, cwd_path, "{}", "", 0)
        return super()._run_locked(args, cwd=cwd, check=check, input_text=input_text)


class _RealManagedActivationReached(Exception):
    pass


def _stop_issue_resume_after_real_activation(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_activate_managed_ci = orchestrator_module.activate_managed_ci

    def activate_then_stop(*args, **kwargs):
        result = real_activate_managed_ci(*args, **kwargs)
        raise _RealManagedActivationReached(result)

    monkeypatch.setattr(
        orchestrator_module,
        "activate_managed_ci",
        activate_then_stop,
    )


class _RealManagedReviewReached(Exception):
    pass


def _stop_issue_resume_after_reviewer(monkeypatch):
    monkeypatch.setattr(
        orchestrator_module,
        "_freeze_prompt_architecture",
        lambda _runner, config, **_kwargs: config,
    )
    real_post_pr_comment = orchestrator_module.post_pr_comment

    def post_and_stop(*args, **kwargs):
        result = real_post_pr_comment(*args, **kwargs)
        if str(kwargs.get("body") or "").startswith("**Review verdict:**"):
            raise _RealManagedReviewReached
        return result

    monkeypatch.setattr(orchestrator_module, "post_pr_comment", post_and_stop)


def test_managed_issue_resume_reviews_same_head_after_post_pr_report_rejection(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            "Fixed issue.\nTests: cd /outside && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex"
        ],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed the resumed exact head.")
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        pre_review_tests=False,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module, "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
    )
    coder_calls = sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)
    _stop_issue_resume_after_reviewer(monkeypatch)

    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands) == coder_calls
    assert sum(command[:1] == ["claude"] for command, _cwd in runner.commands) == 1
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert any("Reviewed the resumed exact head." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF", "AGENT_TEST_OBSERVATION",
            "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_invalid_post_pr_report_then_issue_resume_runs_real_activation_without_reimplementation(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[
            "Fixed issue.\nTests: cd /outside && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex"
        ],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed the durable recovery head.")
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module, "authenticate_issue_created_handoff", lambda *_a, **_k: handoff
    )

    with pytest.raises(AgentLoopError, match="authorization checkpoint.*persisted"):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "creation"
    assert runner.comments == []
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    recovery_start = len(runner.commands)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    recovery_commands = runner.commands[recovery_start:]
    assert any("repos/OWNER/REPO/pulls/77" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/77/comments?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("actions/variables/AGENT_LOOP_MANAGED_ACTOR" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("contents/.github/workflows/ci.yml" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/commits/main" in " ".join(command) for command, _cwd in recovery_commands)
    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert any("Reviewed the durable recovery head." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF",
            "AGENT_TEST_OBSERVATION", "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_pre_pr_number_response_rejection_then_fresh_issue_recovery_uses_real_activation(
    tmp_path, monkeypatch,
):
    valid = structured_issue_implementation(
        pr_number=77,
        tests_run=["python3 -m pytest tests/test_managed_ci.py -q"],
    )
    payload, end = json.JSONDecoder().raw_decode(valid)
    payload.pop("architecture_impact")
    rejected = json.dumps(payload) + valid[end:]
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        codex_outputs=[rejected, rejected],
        claude_outputs=[
            structured_pr_review(state="approved", summary="Reviewed after explicit recovery.")
        ],
    )
    ordinary_config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        agent_max_retries=0,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *_a, **_k: (None, None, ()),
    )

    with pytest.raises(AgentLoopError, match="architecture_impact"):
        run_issue_loop(runner, issue_number=56, config=ordinary_config)

    assert runner.authorization_comments == []
    assert runner.comments == []
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    fresh_config = replace(
        ordinary_config,
        managed_ci_fresh_authorization=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    recovery_start = len(runner.commands)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=fresh_config)

    recovery_commands = runner.commands[recovery_start:]
    assert any("repos/OWNER/REPO/pulls/77" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/77/comments?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("repos/OWNER/REPO/issues/56/timeline?" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("actions/variables/AGENT_LOOP_MANAGED_ACTOR" in " ".join(command) for command, _cwd in recovery_commands)
    assert any("contents/.github/workflows/ci.yml" in " ".join(command) for command, _cwd in recovery_commands)
    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "fresh"
    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    reviewer_command = next(command for command, _cwd in runner.commands if command[:1] == ["claude"])
    assert "abc123" in " ".join(reviewer_command)
    assert any("Reviewed after explicit recovery." in comment for comment in runner.comments)
    assert not any(
        marker in comment
        for comment in runner.comments
        for marker in (
            "AGENT_ISSUE_PR_HANDOFF", "AGENT_TEST_OBSERVATION",
            "AGENT_MANAGED_CI_READINESS",
        )
    )


def test_strict_pre_pr_number_rejection_directs_ordinary_discovery_without_reimplementation(
    tmp_path, monkeypatch,
):
    valid = structured_issue_implementation(
        pr_number=77,
        tests_run=["python3 -m pytest tests/test_managed_ci.py -q"],
    )
    payload, end = json.JSONDecoder().raw_decode(valid)
    payload.pop("architecture_impact")
    rejected = json.dumps(payload) + valid[end:]
    runner = _IssueRecoveryWorkflowRunner(
        labeled=True,
        codex_outputs=[rejected],
        claude_outputs=[
            structured_pr_review(
                state="approved", summary="Reviewed the strict recovered PR."
            )
        ],
        pr_branch_protection_payload={"contexts": ["final-ci/exact-head"]},
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer="claude",
        managed_ci=True,
        managed_ci_trusted_actor="agent-loop",
        agent_max_retries=0,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
        ),
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56",
        trusted_actor="agent-loop",
        protection_mode="strict",
    )
    monkeypatch.setattr(
        orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent
    )
    monkeypatch.setattr(
        orchestrator_module,
        "_run_structured_repair",
        lambda *_a, **_k: (None, None, ()),
    )

    with pytest.raises(AgentLoopError, match="ordinary managed-CI") as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "--managed-ci-fresh" not in message
    assert "unprotected fresh-authorization path is unavailable" in message
    coder_calls = sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    )

    # The PR was created before the structured response was rejected. Ordinary
    # issue discovery can now reach that same strict draft without invoking the
    # implementation coder again.
    runner.open_prs_payload = [{"number": 77, "body": "Fixes #56"}]
    runner.pr_payload["body"] = "Fixes #56"
    runner.rest_pr["body"] = "Fixes #56"
    runner.pr_commit_pages = _provenance_pages(
        "Implement issue.\n\nAgent-Issue-Provenance: v1 "
        "repo=owner/repo issue=56 flow=direct"
    )
    _stop_issue_resume_after_reviewer(monkeypatch)
    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    assert sum(
        command[:2] == ["codex", "exec"] for command, _cwd in runner.commands
    ) == coder_calls


def test_managed_issue_legacy_recovery_rejects_unexpected_closing_reference(
    tmp_path,
):
    body = "Fixes #56\nCloses #999"
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        open_prs_payload=[{"number": 77, "body": body}],
        pr_payload={"body": body},
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )

    with pytest.raises(AgentLoopError, match="outside the expected contract"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert runner.labels_posted is False
    assert runner.dispatch_count == 0
    assert not any(
        command[:1] in (["claude"], ["codex"])
        for command, _cwd in runner.commands
    )


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
        head_sha="abc123",
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
        orchestrator_module.run_pr_loop(
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


def test_managed_issue_authorization_publication_failure_prints_fresh_recovery(
    tmp_path, monkeypatch,
):
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    monkeypatch.setattr(
        orchestrator_module, "publish_issue_created_authorization",
        lambda *_a, **_k: (_ for _ in ()).throw(AgentLoopError("comment returned no ID")),
    )
    with pytest.raises(AgentLoopError) as exc_info:
        orchestrator_module._publish_issue_authorization_with_recovery(
            FakeRunner(), config=config,
            handoff=_managed_issue_handoff(nonce="nonce"),
            metadata=orchestrator_module.PullRequestMetadata(
                number=77, repo="OWNER/REPO", title="PR",
                head_branch="agent-loop/managed-56", base_branch="main",
                head_sha="abc123", url="https://github.test/pull/77",
            ),
            issue_number=56,
        )
    message = str(exc_info.value)
    assert "publication was interrupted" in message
    assert "--managed-ci-fresh" in message
    assert "agent-loop issue 56" in message


def test_managed_issue_publication_malformed_response_stops_before_handoff_or_review(
    tmp_path, monkeypatch,
):
    runner = _MalformedAuthorizationResponseRunner(
        labeled=True,
        codex_outputs=[
            "Implemented.\nTests: python3 -m pytest tests/test_managed_ci.py\n"
            "<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(
        tmp_path, coder="codex", reviewer="claude", managed_ci=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    intent = ManagedCiCreationIntent(
        branch="agent-loop/managed-56", trusted_actor="agent-loop",
        protection_mode="voluntary", audit_nonce="opening-nonce",
    )
    handoff = _managed_issue_handoff(nonce="opening-nonce")
    monkeypatch.setattr(orchestrator_module, "preflight_managed_ci_creation", lambda *_a, **_k: intent)
    monkeypatch.setattr(orchestrator_module, "authenticate_issue_created_handoff", lambda *_a, **_k: handoff)

    with pytest.raises(AgentLoopError, match="publication was interrupted"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert any(
        endpoint in " ".join(command)
        for command, _cwd in runner.commands
        for endpoint in ("repos/OWNER/REPO/issues/77/comments",)
        if "POST" in command and "ISSUE_AUTHORIZATION" in " ".join(command)
    )
    assert runner.authorization_comments == []
    assert runner.comments == []
    assert not any(command[:1] == ["claude"] for command, _cwd in runner.commands)


def test_issue_fresh_recovery_discovers_pre_handoff_pr_without_reimplementing(
    tmp_path, monkeypatch,
):
    runner = _IssueRecoveryWorkflowRunner(
        labeled=False,
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct"
        ),
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Reviewed the fresh-recovery exact head.",
                reviewer="OpenAI Codex",
            )
        ],
    )
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_fresh_authorization=True,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        invocation_argv=(
            "agent-loop", "issue", "56", "--managed-ci", "--managed-ci-fresh",
            "--managed-ci-trusted-actor", "agent-loop",
            "--allow-unprotected-managed-ci",
        ),
    )
    _stop_issue_resume_after_reviewer(monkeypatch)

    with pytest.raises(_RealManagedReviewReached):
        run_issue_loop(runner, issue_number=56, config=config)

    records = [
        parsed
        for comment in runner.authorization_comments
        if (parsed := parse_issue_created_authorization_comment(comment["body"]))
        is not None
    ]
    assert len(records) == 1 and records[0].kind == "fresh"
    assert runner.labels_posted is True
    assert runner.dispatch_count == 0
    assert sum(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands) == 1
    assert any(
        "Reviewed the fresh-recovery exact head." in comment
        for comment in runner.comments
    )


def test_issue_loop_outside_workdir_after_reported_pr_hedges_unconfirmed_pr(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: cd /outside && python -m pytest\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        pr_payload={"state": "CLOSED"},
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "PR #77 is CLOSED" in message
    assert "agent-loop pr 77" not in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_accepts_absolute_interpreter_test_command_through_response_path(tmp_path):
    # Regression for #584: an absolute system-interpreter path in program
    # position (the only path outside the assigned checkout) must not be
    # misclassified as an external test location when reported through the
    # freeform `Tests:` response path (origin='response', orchestrator.py
    # line ~3286/6223), matching the PR #484 follow-up reproducer.
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: `/usr/bin/python3 -m pytest tests/test_durable_jobs.py -q` "
            "- 12 passed, run from the assigned checkout.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0
    assert len(runner.comments) >= 1

def test_issue_loop_live_target_after_reported_pr_mentions_confirmed_resume(tmp_path):
    # Same post-PR guidance wrapper as the outside-workdir case above, but
    # triggered by a live-remote-target rejection instead of a path
    # rejection, proving the URL pass also reaches
    # _validate_response_tests_with_post_pr_context.
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: ran curl https://live.example/health to verify.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError) as exc_info:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(exc_info.value)
    assert "live remote target" in message
    assert "PR #77 was confirmed open" in message
    assert "handoff/reviewer comments were not posted" in message
    assert "agent-loop pr 77" in message
    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_rejects_reported_pr_when_assigned_head_unchanged(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Fixed issue.\n"
            "Tests: python -m pytest passed.\n"
            "<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n"
            "-- OpenAI Codex",
        ],
        claude_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        advance_git_head_on_pr=False,
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude")

    with pytest.raises(AgentLoopError, match="HEAD did not advance"):
        run_issue_loop(runner, issue_number=56, config=config)

    assert runner.comments == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_gemini_issue_loop_creates_pr_then_codex_approves(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Looks good.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="gemini", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    agent_commands = [cmd[:2] for cmd, _cwd in runner.commands if cmd[:1] in (["gemini"], ["codex"])]
    assert agent_commands == [["gemini", "--prompt"], ["codex", "exec"]]
    assert len(runner.comments) == 3
    assert "<!-- AGENT_ISSUE_PR_HANDOFF:" in runner.comments[0]
    assert runner.comments[1].startswith("## Issue implementation")
    assert runner.comments[2].startswith("**Review verdict:** Approved\n\nLooks good.")

def test_gemini_issue_loop_resumes_session_for_followup(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps({
                "response": "Fixed issue.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
                "session_id": "gemini-session-1",
            }),
            # Plain-text output intentionally clears the tracked session; a third
            # Gemini turn would start without --resume.
            "Addressed review.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            "Needs a regression test.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer="codex",
        gemini_args=("--output-format", "json"),
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    gemini_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"]]
    assert len(gemini_calls) == 2
    assert "--resume" not in gemini_calls[0]
    assert gemini_calls[1][-2:] == ["--resume", "gemini-session-1"]

def test_issue_loop_plan_first_one_shot_rerun_resumes_pr_loop(tmp_path):
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
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 0
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


def _salvage_dirs(config):
    salvage_root = config.log_dir / "salvage"
    if not salvage_root.exists():
        return []
    return sorted(path for path in salvage_root.iterdir() if path.is_dir())


def test_failed_issue_implementation_with_diff_writes_salvage_artifacts(tmp_path):
    patch_text = (
        "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
        "--- a/src/coding_review_agent_loop/cli.py\n"
        "+++ b/src/coding_review_agent_loop/cli.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n?? scratch-note.md\n",
        post_agent_git_diff=patch_text,
        post_agent_git_diff_stat=" src/coding_review_agent_loop/cli.py | 2 +-\n",
        post_agent_git_diff_check="src/coding_review_agent_loop/cli.py:1: trailing whitespace.\n",
        post_agent_git_diff_check_returncode=2,
    )
    config = make_config(tmp_path)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage artifacts were written to" in message
    assert "patch:" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    salvage_dir = _salvage_dirs(config)[0]
    assert str(salvage_dir / "salvage-summary.md") in message
    assert (salvage_dir / "partial.patch").read_text(encoding="utf-8") == patch_text
    assert "?? scratch-note.md" in (salvage_dir / "changed-files.txt").read_text(
        encoding="utf-8"
    )
    assert "2 +-" in (salvage_dir / "diff-stat.txt").read_text(encoding="utf-8")
    assert "trailing whitespace" in (salvage_dir / "diff-check.txt").read_text(
        encoding="utf-8"
    )

    summary = (salvage_dir / "salvage-summary.md").read_text(encoding="utf-8")
    assert "No\nsuccessful response, review result, or pull request should be inferred" in summary
    assert "Public response file: missing" in summary
    assert "Required marker status: missing or invalid" in summary
    assert "Untracked files appear in `changed-files.txt`" in summary

    metadata = json.loads((salvage_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["repo"] == "OWNER/REPO"
    assert metadata["issue_number"] == 56
    assert metadata["scope"] == "issue-implementation"
    assert metadata["agent"] == "claude"
    assert metadata["failure_category"] == "transient"
    assert metadata["response_file_missing"] is True
    assert metadata["diff_check_returncode"] == 2


def test_failed_issue_implementation_salvage_oserror_preserves_original_failure(
    tmp_path, capsys, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_diff="diff --git a/file.txt b/file.txt\n",
    )
    config = make_config(tmp_path, quiet=False)

    def fail_capture(*args, **kwargs):
        raise OSError("simulated disk full")

    monkeypatch.setattr(orchestrator_module, "capture_salvage_artifacts", fail_capture)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "quota exhausted" in message
    assert "Rerun when quota resets" in message
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "capture failed (simulated disk full)" in message
    assert "preserving the original agent failure" in message
    assert _salvage_dirs(config) == []
    assert "salvage capture failed (simulated disk full)" in capsys.readouterr().err


def test_failed_issue_implementation_without_diff_writes_no_salvage_patch(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "Implementation salvage was attempted for issue implementation" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert "no patch artifacts were created" in message
    assert _salvage_dirs(config) == []


def test_failed_issue_implementation_with_untracked_only_diff_reports_untracked_only(tmp_path):
    runner = FakeRunner(
        claude_outputs=["Created local notes but forgot the required marker."],
        post_agent_git_status="?? scratch-note.md\n",
        post_agent_git_diff="",
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "only untracked files were present" in message
    assert "no tracked/staged `git diff HEAD --binary` existed" in message
    assert _salvage_dirs(config) == []


def test_plan_revision_quota_failure_reports_response_file_without_recording(tmp_path):
    valid_revision = structured_plan_revision(
        summary="Revised plan with a regression test.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "note": "Added the test step."}
        ],
        plan_steps=["Add the regression test.", "Run the focused suite."],
    )
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            ("quota exceeded; reset in 10m", 1),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
        # A failed exit may only be salvaged when its current-attempt artifact
        # passes the normal revision validator.
        public_response_outputs=["", "", "partial revision"],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan revision" in message
    assert "not a mutating implementation attempt" in message
    assert "A public response file exists at" in message
    assert "no result was recorded because the agent command exited with quota/session-limit status" in message
    assert "Revised plan with a regression test" not in "".join(runner.comments)


def test_plan_review_failure_without_response_file_reports_non_mutating_skip(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["Plan review without the required structured response."],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "No implementation salvage was attempted because this was plan review" in message
    assert "not a mutating implementation attempt" in message
    assert "No non-empty public response file was produced at expected path" in message
    assert "no result was recorded because the public response failed validation" in message


def _write_salvage_summary(
    config,
    *,
    name,
    summary,
    created_at_ns,
    issue_number=56,
    scope="issue-implementation",
    approved_plan_hash_value=None,
):
    salvage_dir = config.log_dir / "salvage" / name
    salvage_dir.mkdir(parents=True)
    summary_path = salvage_dir / "salvage-summary.md"
    summary_path.write_text(summary, encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "created_at_ns": created_at_ns,
        "repo": config.repo,
        "issue_number": issue_number,
        "scope": scope,
        "agent": "claude",
        "approved_plan_hash": approved_plan_hash_value,
        "summary": str(summary_path),
    }
    (salvage_dir / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_issue_implementation_rerun_prompt_includes_latest_salvage_summary(tmp_path):
    config = make_config(tmp_path)
    _write_salvage_summary(
        config,
        name="old",
        summary="old failed attempt summary",
        created_at_ns=1,
    )
    _write_salvage_summary(
        config,
        name="new",
        summary="new failed attempt summary\nPartial patch: `/tmp/new.patch`",
        created_at_ns=2,
    )
    runner = FakeRunner(
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous failed implementation attempt salvage:" in coder_prompt
    assert "new failed attempt summary" in coder_prompt
    assert "old failed attempt summary" not in coder_prompt
    assert "Do not auto-apply the patch" in coder_prompt
    assert "cherry-pick or ignore it" in coder_prompt
    assert "selectively" in coder_prompt


def test_latest_salvage_summary_filters_approved_plan_hash(tmp_path):
    config = make_config(tmp_path)
    plan_hash = approved_plan_hash("Plan:\n- Current.")
    _write_salvage_summary(
        config,
        name="old-plan",
        summary="stale approved-plan summary",
        created_at_ns=5,
        scope="approved-plan-implementation",
        approved_plan_hash_value=approved_plan_hash("Plan:\n- Old."),
    )
    _write_salvage_summary(
        config,
        name="current-plan",
        summary="current approved-plan summary",
        created_at_ns=4,
        scope="approved-plan-implementation",
        approved_plan_hash_value=plan_hash,
    )

    summary = latest_salvage_summary(
        config.log_dir,
        repo=config.repo,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash=plan_hash,
    )

    assert summary == "current approved-plan summary"


def test_failed_issue_implementation_posts_github_salvage_comment(tmp_path):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff=(
            "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
            "--- a/src/coding_review_agent_loop/cli.py\n"
            "+++ b/src/coding_review_agent_loop/cli.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
        post_agent_git_diff_stat=" src/coding_review_agent_loop/cli.py | 2 +-\n",
    )
    config = make_config(tmp_path)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "A GitHub salvage comment was posted to issue #56." in message
    assert len(runner.issue_comments) == 1
    posted_body = runner.issue_comments[0]["body"]
    assert "<!-- AGENT_SALVAGE:" in posted_body
    assert "Implementation salvage breadcrumb" in posted_body


def test_failed_issue_implementation_salvage_comment_disabled_posts_nothing(tmp_path):
    runner = FakeRunner(
        claude_outputs=[("quota exceeded; reset in 10m", 1)],
        post_agent_git_status=" M src/coding_review_agent_loop/cli.py\n",
        post_agent_git_diff=(
            "diff --git a/src/coding_review_agent_loop/cli.py b/src/coding_review_agent_loop/cli.py\n"
            "--- a/src/coding_review_agent_loop/cli.py\n"
            "+++ b/src/coding_review_agent_loop/cli.py\n"
            "@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    config = make_config(tmp_path, salvage_comments=False)

    with pytest.raises(QuotaResetExceededError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config)

    message = str(excinfo.value)
    assert "No GitHub salvage comment was posted." in message
    assert runner.issue_comments == []


def _build_remote_salvage_comment(
    tmp_path,
    *,
    issue_number=56,
    scope="issue-implementation",
    approved_plan_hash_value=None,
    patch_text=None,
    failure_reason="remote-only failure summary",
    patch_max_bytes=20000,
):
    patch_text = patch_text or (
        "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    )
    checkout = tmp_path / "remote-source-checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    seed_runner = FakeRunner(
        git_diff=patch_text,
        git_status=" M file.txt\n",
        git_diff_stat=" file.txt | 2 +-\n",
    )
    context = SalvageContext(
        repo="OWNER/REPO",
        issue_number=issue_number,
        scope=scope,
        agent="claude",
        approved_plan_hash=approved_plan_hash_value,
    )
    artifacts = capture_salvage_artifacts(
        seed_runner,
        checkout=checkout,
        log_dir=tmp_path / "remote-source-logs",
        context=context,
        failure_category="transient",
        failure_reason=failure_reason,
        required_marker="<!-- AGENT_PR: <number> -->",
        result=None,
    )
    post_config = make_config(tmp_path, salvage_comment_patch_max_bytes=patch_max_bytes)
    posted = post_salvage_comment(
        seed_runner,
        config=post_config,
        artifacts=artifacts,
        context=context,
        failure_category="transient",
        failure_reason=failure_reason,
    )
    assert posted
    return seed_runner.issue_comments[-1]


def test_issue_implementation_rerun_discovers_remote_salvage_when_local_log_dir_is_empty(tmp_path):
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="issue-implementation",
        failure_reason="remote-only failure summary",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous failed implementation attempt salvage:" in coder_prompt
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "remote-only failure summary" in coder_prompt
    assert "```diff" in coder_prompt
    assert "-old" in coder_prompt and "+new" in coder_prompt
    # The raw AGENT_SALVAGE breadcrumb comment must not also be rendered
    # verbatim via the ordinary issue-context comment block (#507 follow-up):
    # it is consumed only through the parsed salvage_summary injection above.
    assert "AGENT_SALVAGE" not in coder_prompt
    assert "Implementation salvage breadcrumb" not in coder_prompt


def test_issue_implementation_rerun_remote_salvage_with_omitted_patch_renders_local_only_note(tmp_path):
    oversized_patch = (
        "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
    ) + ("+padding\n" * 5000)
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="issue-implementation",
        patch_text=oversized_patch,
        patch_max_bytes=100,
        failure_reason="remote failure with an oversized patch",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "local-only" in coder_prompt
    assert "```diff" not in coder_prompt


def test_issue_implementation_rerun_ignores_remote_salvage_for_a_different_issue(tmp_path):
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=999,
        scope="issue-implementation",
        failure_reason="unrelated issue failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "Fix GitHub issue #56" in cmd[-1]
    )
    assert "Previous failed implementation attempt salvage:" not in coder_prompt


def test_approved_plan_implementation_rerun_discovers_remote_salvage_with_matching_plan_hash(tmp_path):
    approved_plan = "Plan:\n- Do the thing."
    plan_hash = approved_plan_hash(approved_plan)
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash_value=plan_hash,
        failure_reason="remote approved-plan failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    usage_context = orchestrator_module._new_usage_context(config)

    result = orchestrator_module._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=usage_context,
    )

    assert result == 0
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Previous failed implementation attempt salvage:" in coder_prompt
    assert "recovered from a GitHub issue comment" in coder_prompt
    assert "remote approved-plan failure" in coder_prompt


@pytest.mark.parametrize("terminal_marker, expected_state", [
    ("<!-- AGENT_STATE: blocking -->", "blocking"),
    ("<!-- AGENT_CLARIFY -->", "clarification"),
])
def test_approved_plan_no_pr_terminal_result_bypasses_human_requirements_and_pr_checks(
    tmp_path, terminal_marker, expected_state
):
    runner = FakeRunner(
        issue_payload={"body": "Keep the public API stable."},
        claude_outputs=[f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)

    with pytest.raises(AgentLoopError, match=f"implementation is {expected_state}"):
        orchestrator_module._implement_approved_issue(
            runner,
            issue_number=56,
            approved_plan="Plan:\n- Do the thing.",
            config=config,
            memory=None,
            issue_context=issue_context,
            coder_session_id=None,
            usage_context=orchestrator_module._new_usage_context(config),
        )

    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:1] == ["codex"] for cmd, _cwd in runner.commands)
    # A genuine coder-declared no-PR terminal is posted to the issue like a
    # PR-success implementation result already is (#588).
    if expected_state == "blocking":
        assert runner.comments[0].startswith("## Issue implementation")
        assert "Cannot proceed." in runner.comments[0]
        assert "No pull request was accepted for handoff." in runner.comments[0]
    else:
        assert runner.comments == [
            f"Cannot proceed.\n<!-- AGENT_PR: 0 -->\n{terminal_marker}\n-- Anthropic Claude: unknown model (medium)"
        ]


def test_approved_plan_implementation_rerun_ignores_remote_salvage_on_plan_hash_mismatch(tmp_path):
    approved_plan = "Plan:\n- Do the current thing."
    stale_plan_hash = approved_plan_hash("Plan:\n- Do the old thing.")
    remote_comment = _build_remote_salvage_comment(
        tmp_path,
        issue_number=56,
        scope="approved-plan-implementation",
        approved_plan_hash_value=stale_plan_hash,
        failure_reason="stale plan failure",
    )
    runner = FakeRunner(
        issue_comments=[remote_comment],
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path)
    issue_context = get_issue_context(runner, config=config, issue_number=56)
    usage_context = orchestrator_module._new_usage_context(config)

    result = orchestrator_module._implement_approved_issue(
        runner,
        issue_number=56,
        approved_plan=approved_plan,
        config=config,
        memory=None,
        issue_context=issue_context,
        coder_session_id=None,
        usage_context=usage_context,
    )

    assert result == 0
    coder_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    assert "Previous failed implementation attempt salvage:" not in coder_prompt


def test_task_and_pr_followup_salvage_scopes_post_no_github_comment(tmp_path):
    config = make_config(tmp_path)
    runner = FakeRunner(
        post_agent_git_status=" M file.txt\n",
        post_agent_git_diff=(
            "diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n"
        ),
    )
    runner._mark_agent_command_seen()

    for scope in (
        orchestrator_module.TASK_IMPLEMENTATION_SALVAGE_SCOPE,
        orchestrator_module.PR_FOLLOWUP_SALVAGE_SCOPE,
    ):
        diagnostic = orchestrator_module._capture_failed_run_salvage_diagnostic(
            runner=runner,
            config=config,
            agent_name="claude",
            salvage_context=SalvageContext(
                repo=config.repo,
                issue_number=56,
                scope=scope,
                agent="claude",
            ),
            operation_description="task implementation",
            failure_category="transient",
            failure_reason="agent failed",
            classification_text="",
            marker_description="<!-- AGENT_PR: <number> -->",
            result=None,
        )
        assert diagnostic.artifacts is not None
        assert "No GitHub salvage comment was posted." in diagnostic.line

    assert runner.comments == []
    assert runner.issue_comments == []


# --- Visible sidecar labels at the issue posting seams (#842) ----------------


class _SidecarPostingRunner:
    """Record posted bodies; echo REST bodies back (optionally altered)."""

    def __init__(self, *, alter_sidecar_label: bool = False) -> None:
        self.bodies: list[str] = []
        self.alter_sidecar_label = alter_sidecar_label

    def run(self, args, *, cwd, input_text=None, check=True, env=None):
        from pathlib import Path

        if "--body-file" in args:
            self.bodies.append(Path(args[args.index("--body-file") + 1]).read_text())
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        assert input_text is not None
        body = json.loads(input_text)["body"]
        self.bodies.append(body)
        returned = body
        if self.alter_sidecar_label and "AGENT_LOOP_SIDECAR" in body:
            returned = body.replace("Agent-loop plan attachment", "Agent-loop attachment", 1)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "id": len(self.bodies),
                    "body": returned,
                    "created_at": "2026-09-19T00:00:00Z",
                    "user": {"login": "agent-bot", "id": 7},
                }
            ),
            stderr="",
        )


def _sidecar_seam_config():
    return SimpleNamespace(quiet=True, dry_run=False, gh_cmd="gh", repo="owner/repo")


def _oversized_round_body(flow: str, role: str) -> TrustedBody:
    import base64
    import os

    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    large = base64.urlsafe_b64encode(os.urandom(70_000)).decode("ascii")
    field = "canonical_plan" if flow == "plan" and role == "coder" else "canonical_reviewer_response"
    text = _attach_round_metadata(
        "Visible round response",
        PostedRoundMetadata(
            flow=flow,
            role=role,
            agent="codex",
            round_number=1,
            subject="sidecar-label-seam",
            **{field: large},
        ),
    )
    return TrustedBody.canonical(text, expected_tokens=("AGENT_LOOP_META",))


def test_verified_plan_round_posts_labeled_sidecars_and_accepts_exact_readback(monkeypatch):
    import coding_review_agent_loop.github as github_module
    from coding_review_agent_loop.round_transport import is_round_transport_sidecar

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    runner = _SidecarPostingRunner()

    posted = github_module.post_verified_trusted_issue_round_comment(
        runner,
        config=_sidecar_seam_config(),
        issue_number=842,
        body=_oversized_round_body("plan", "coder"),
        expected_author_login="agent-bot",
        expected_author_id=7,
    )

    sidecars = runner.bodies[:-1]
    assert sidecars
    for position, body in enumerate(sidecars, start=1):
        assert is_round_transport_sidecar(body)
        assert body.startswith(f"Agent-loop plan attachment {position}/{len(sidecars)} ")
        assert "see the following plan comment." in body
    assert posted.body == runner.bodies[-1]
    assert not is_round_transport_sidecar(posted.body)


def test_verified_plan_round_rejects_readback_with_altered_sidecar_label(monkeypatch):
    import coding_review_agent_loop.github as github_module

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    runner = _SidecarPostingRunner(alter_sidecar_label=True)

    with pytest.raises(AgentLoopError, match="returned a different body"):
        github_module.post_verified_trusted_issue_round_comment(
            runner,
            config=_sidecar_seam_config(),
            issue_number=842,
            body=_oversized_round_body("plan", "coder"),
            expected_author_login="agent-bot",
            expected_author_id=7,
        )
    assert len(runner.bodies) == 1


@pytest.mark.parametrize("role", ["debater", "summary"])
def test_discuss_round_sidecars_posted_to_issue_use_neutral_wording(monkeypatch, role):
    import coding_review_agent_loop.github as github_module

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    runner = _SidecarPostingRunner()

    github_module.post_issue_comment(
        runner,
        config=_sidecar_seam_config(),
        issue_number=842,
        body=_oversized_round_body("discuss", role),
    )

    sidecars = runner.bodies[:-1]
    assert sidecars
    for body in sidecars:
        assert body.startswith("Agent-loop attachment ")
        assert "see the following agent-loop comment." in body
        assert "plan attachment" not in body and "review attachment" not in body


def test_rest_recovery_merges_labeled_sidecars_missing_from_projection(monkeypatch):
    import coding_review_agent_loop.github as github_module
    from coding_review_agent_loop.round_transport import prepare_round_comment

    monkeypatch.setattr(github_module, "active_workdir", lambda config: None)
    prepared = [str(item) for item in prepare_round_comment(_oversized_round_body("plan", "coder"))]
    sidecars = prepared[:-1]
    projection = tuple(
        IssueComment(author="human", created_at=f"2026-09-19T00:{index // 60:02d}:{index % 60:02d}Z", body=f"note {index}")
        for index in range(100)
    )
    rest_page_1 = [
        {"id": index + 1, "body": comment.body, "created_at": comment.created_at, "user": {"login": "human", "id": 3}}
        for index, comment in enumerate(projection)
    ]
    rest_page_2 = [
        {"id": 1000 + index, "body": body, "created_at": "2026-09-19T01:00:00Z", "user": {"login": "agent-bot", "id": 7}}
        for index, body in enumerate(sidecars)
    ]

    class _RestRunner:
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            page = rest_page_1 if args[-1].endswith("page=1") else rest_page_2
            return SimpleNamespace(returncode=0, stdout=json.dumps(page), stderr="")

    merged = github_module._merge_issue_comment_transport_identity(
        _RestRunner(), config=_sidecar_seam_config(), issue_number=842, comments=projection
    )

    recovered = [comment for comment in merged if comment.comment_id and comment.comment_id >= 1000]
    assert [comment.body for comment in recovered] == sidecars


def test_plan_round_metadata_failure_raises_diagnosed_error(tmp_path, monkeypatch):
    """#879: a contradictory metadata record must not escape as a bare ValueError."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    real_metadata = orchestrator_module.PostedRoundMetadata

    def _exploding(**kwargs):
        if kwargs.get("flow") == "plan" and kwargs.get("role") == "coder":
            raise ValueError("semantic patch metadata is incomplete")
        return real_metadata(**kwargs)

    monkeypatch.setattr(orchestrator_module, "PostedRoundMetadata", _exploding)

    with pytest.raises(AgentLoopError) as error:
        run_issue_loop(
            runner, issue_number=56, config=make_config(tmp_path), plan_first=True
        )

    assert "Could not record the plan round metadata" in str(error.value)
    assert "semantic patch metadata is incomplete" in str(error.value)


# --- Issue #871: a reviewer turn with no review is never repaired into one ---

_NARRATION_ONLY_PLAN_REVIEW = (
    "I have launched the test command for tests/test_test_runtime.py in the "
    "background and will wait for it to complete.\n"
    "root agent idle; waiting up to 5s for 1 background task(s)\n"
    "terminating 1 background task(s) on exit"
)


def _fabricated_plan_review():
    return structured_plan_review(
        state="blocking",
        summary="Plan review incomplete: the test command was terminated.",
        blocking_plan_issues=["Plan review incomplete: the test command was terminated."],
        reviewer="OpenAI Codex",
    )


def test_narration_only_plan_reviewer_is_unavailable_and_posts_no_verdict(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
    )
    repair_calls = []

    def fake_repair(raw, gemini_cmd, **kwargs):
        repair_calls.append(raw)
        return _fabricated_plan_review()

    config = make_config(tmp_path, agent_max_retries=1, agent_retry_backoff_seconds=0)
    with patch("coding_review_agent_loop.orchestrator.attempt_repair", fake_repair):
        with pytest.raises(AgentLoopError, match="review_substance_integrity") as excinfo:
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert "agent-unavailable" in str(excinfo.value)
    # The refusal happens before any repair backend call.
    assert repair_calls == []
    # No fabricated verdict is posted and no ledger item is numbered.
    assert not any("Plan review incomplete" in body for body in runner.comments)
    assert not any("[item-1]" in body for body in runner.comments)
    # The reviewer is retried within the configured policy before the run stops.
    codex_turns = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_turns) >= 2


def test_empty_plan_reviewer_response_never_reaches_repair(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=["", "", "", ""],
    )
    repair_calls = []

    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)
    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair",
        lambda raw, gemini_cmd, **kwargs: repair_calls.append(raw) or _fabricated_plan_review(),
    ):
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert repair_calls == []
    assert not any("Plan review incomplete" in body for body in runner.comments)


def test_refused_plan_reviewer_leaves_nothing_for_a_resume_to_replay(tmp_path):
    """A refused reviewer must not leave a checkpoint a rerun could replay."""
    config = make_config(tmp_path, agent_max_retries=0, agent_retry_backoff_seconds=0)

    def run_once():
        runner = _FakeRunner(
            claude_outputs=[structured_v1_plan_state()],
            codex_outputs=[_NARRATION_ONLY_PLAN_REVIEW] * 4,
        )
        with patch(
            "coding_review_agent_loop.orchestrator.attempt_repair",
            lambda raw, gemini_cmd, **kwargs: _fabricated_plan_review(),
        ):
            with pytest.raises(AgentLoopError):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        return runner

    first = run_once()
    second = run_once()
    for runner in (first, second):
        assert not any("Plan review incomplete" in body for body in runner.comments)
        assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)


# --- Staged primary-then-panel plan review (#905, from #841) --------------


def _staged_plan_config(tmp_path, **overrides):
    values = {
        "reviewer": ("codex", "gemini"),
        "plan_review_policy": "primary-then-panel",
        "primary_plan_reviewer": "codex",
        "max_rounds": 6,
    }
    values.update(overrides)
    return make_config(tmp_path, **values)


def _plan_round_records(runner):
    """Decode every posted planning round record in comment order."""
    records = []
    for comment in runner.issue_comments:
        body = comment["body"]
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow == "plan":
            records.append(metadata)
    return records


def _plan_audit_body(runner, *, round_number):
    """The posted planning scheduler audit body for one round."""
    for comment in runner.issue_comments:
        body = comment["body"]
        if not body.startswith("Plan review scheduling audit."):
            continue
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if (
            metadata.flow == "plan"
            and metadata.phase == "scheduler-prelaunch"
            and metadata.round_number == round_number
        ):
            return body
    raise AssertionError(f"no planning scheduler audit posted for round {round_number}")


def _staged_plan_history(
    tmp_path, *, reviewer_records=(), scheduler_record=True, reconciliation_record=False
):
    """A generation-1 planning round-1 coder record plus reviewer records.

    ``scheduler_record`` writes the round-1 planning scheduler checkpoint, so
    the history decodes as intact rather than as the ``absent`` degraded class.
    """
    config = _staged_plan_config(tmp_path)
    structured = validate_structured_plan_state(structured_v1_plan_state())
    canonical = render_canonical_plan_state(structured, config)
    sidecar = orchestrator_module.make_assembled_plan_sidecar(
        structured, round_number=1, response_form="fresh-plan-state", rendered_plan=canonical
    )
    subject = _plan_subject(canonical)
    comments = [
        {
            "author": {"login": "bot"},
            "createdAt": "2026-01-01T00:00:00Z",
            "body": _attach_round_metadata(
                canonical + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Anthropic Claude",
                    round_number=1,
                    subject=subject,
                    canonical_plan=canonical,
                    raw_structured_coder_response=structured_v1_plan_state(),
                    state="blocking",
                    execution_strategy_contract_version=1,
                    execution_strategy_identity=structured.execution_recommendation.identity(),
                    risk_test_matrix_contract_version=1,
                    risk_test_matrix_payload=structured.risk_test_matrix.to_payload(),
                    risk_test_matrix_changes_payload=(),
                    risk_test_matrix_identity=risk_test_matrix_identity(
                        structured.risk_test_matrix, structured.risk_test_matrix_changes
                    ),
                    risk_test_matrix_boundary_digest=risk_test_matrix_identity(
                        structured.risk_test_matrix, structured.risk_test_matrix_changes
                    ),
                    response_form="fresh-plan-state",
                    aggregate_plan_identity=sidecar.aggregate_identity,
                    assembled_plan_sidecar=sidecar.to_payload(),
                ),
            ),
        }
    ]
    if scheduler_record:
        contract = orchestrator_module.make_plan_contract(
            ("Codex", "Gemini"), "primary-then-panel", "Codex"
        )
        key = orchestrator_module._plan_candidate_key_for(
            plan_subject=subject, sidecar=sidecar, surfaced_requirement_ids=()
        )
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": "2026-01-01T00:00:00Z",
                "body": _attach_round_metadata(
                    "Plan review scheduling audit.\n\n-- Orchestrator",
                    PostedRoundMetadata(
                        flow="plan",
                        role="summary",
                        agent="Orchestrator",
                        round_number=1,
                        subject=subject,
                        phase="scheduler-prelaunch",
                        scheduler_contract=contract.as_dict(),
                        scheduler_obligation_digest="0" * 16,
                        scheduler_selected_reviewers=("Codex",),
                        scheduler_paused_reviewers=(("Gemini", "primary phase"),),
                        scheduler_reasons=("primary phase",),
                        scheduler_final_sweep=False,
                        scheduler_force_full=False,
                        scheduler_calls_avoided=1,
                        scheduler_phase="primary",
                        scheduler_primary_reviewer="Codex",
                        plan_candidate_key=key.as_dict(),
                    ),
                ),
            }
        )
    for index, (agent, state, items) in enumerate(reviewer_records, start=1):
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": f"2026-01-01T00:00:{index:02d}Z",
                "body": _attach_round_metadata(
                    f"{agent} plan review.\n<!-- AGENT_PLAN_STATE: {state} -->\n-- {agent}",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent=agent,
                        round_number=1,
                        subject=subject,
                        state=state,
                        new_items=items,
                    ),
                ),
            }
        )
    if reconciliation_record:
        # A reconciled round: the summary checkpoint repeats the round's items,
        # so a resume rehydrates them into the current-round ledger.
        reconciled_items = tuple(
            item for _agent, _state, items in reviewer_records for item in items
        )
        comments.append(
            {
                "author": {"login": "bot"},
                "createdAt": "2026-01-01T00:01:00Z",
                "body": _attach_round_metadata(
                    "Plan round reconciliation.\n\n-- Orchestrator",
                    PostedRoundMetadata(
                        flow="plan",
                        role="summary",
                        agent="Orchestrator",
                        round_number=1,
                        subject=subject,
                        phase="reconciliation",
                        new_items=reconciled_items,
                    ),
                ),
            }
        )
    return comments, canonical


def test_default_full_board_planning_writes_no_scheduler_metadata(tmp_path):
    """`default-full-board-planning`: the compatibility path is unchanged."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    records = _plan_round_records(runner)
    assert records, "the planning flow must still post durable round records"
    assert all(record.scheduler_metadata_status == "absent" for record in records)
    assert all(record.plan_candidate_key is None for record in records)
    assert all(record.phase != "scheduler-prelaunch" for record in records)
    reviewer_rounds = {
        (record.agent, record.round_number)
        for record in records
        if record.role == "reviewer"
    }
    # Every configured reviewer is invoked in the same planning round.
    assert reviewer_rounds == {("Codex", 1), ("Gemini", 1)}


def test_staged_planning_primary_gate_then_reviewer_only_panel_round(tmp_path):
    """`primary-blocking-revision`, `panel-opens-on-primary-approval`,
    `reviewer-only-phase-advance`, and `final-exact-plan-gate` together."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    assert run_issue_loop(
        runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
    ) == 0

    records = _plan_round_records(runner)
    prelaunch = [record for record in records if record.phase == "scheduler-prelaunch"]
    assert [record.scheduler_phase for record in prelaunch] == [
        "primary",
        "secondary-audit",
    ]
    # Round 1 is strictly primary-only; the secondary is paused, not dropped.
    assert prelaunch[0].scheduler_selected_reviewers == ("Codex",)
    assert [name for name, _why in prelaunch[0].scheduler_paused_reviewers] == ["Gemini"]
    assert prelaunch[0].scheduler_metadata_status == "valid"
    # The panel opens on the exact-plan primary approval, against the same key.
    assert prelaunch[1].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[1].scheduler_approved_reviewers
    assert prelaunch[0].plan_candidate_key == prelaunch[1].plan_candidate_key
    # The panel round is reviewer-only: one planner turn in the whole run.
    advance = [record for record in records if record.phase == "plan-phase-advance"]
    assert [record.round_number for record in advance] == [2]
    # The advance names the phase that is still pending, not the primary round
    # that just finished.
    advance_body = next(
        comment["body"]
        for comment in runner.issue_comments
        if "Plan review phase advance to round 2." in comment["body"]
    )
    assert "Outstanding phase after this advance: `secondary-audit`" in advance_body
    assert "`primary`" not in advance_body
    assert len([record for record in records if record.role == "coder"]) == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "claude") == 1
    # Each reviewer ran exactly once, in its own round.
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1
    assert [
        record.round_number
        for record in records
        if record.role == "reviewer" and record.agent == "Gemini"
    ] == [2]


def _completed_staged_plan_comments(tmp_path):
    """Posted comments of a primary-then-panel plan approved across two rounds."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    return config, [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments]


def test_managed_ci_plan_approval_accepts_staged_approvals_across_rounds(tmp_path):
    """#962: the primary approves in round 1 and the panel in round 2."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )
    # The resumed round alone holds only the panel's approval.
    assert {
        record.metadata.agent
        for record in plan_round.completed_reviews
        if record.metadata.state == "approved"
    } == {"Gemini"}

    orchestrator_module._require_complete_canonical_plan_approval(
        comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=(),
        error_message="incomplete",
    )


def test_managed_ci_plan_approval_rejects_superseded_staged_approval(tmp_path):
    """A later non-approving record for the exact plan revokes the carry."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    revoked = []
    for comment in comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment.body)
        if match is not None:
            metadata = _decode_round_metadata(match.group("payload"))
            if metadata.flow == "plan" and metadata.role == "reviewer" and metadata.agent == "Codex":
                revoked.append(
                    SimpleNamespace(
                        body=_attach_round_metadata(
                            "Codex plan review.\n-- Codex",
                            replace(metadata, state="blocking"),
                        )
                    )
                )
    assert len(revoked) == 1
    comments = [*comments, *revoked]
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )

    with pytest.raises(AgentLoopError, match="incomplete"):
        orchestrator_module._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


def test_managed_ci_plan_approval_staged_full_board_round_requires_exact_key(tmp_path):
    """A staged full-board round still needs approvals bound to the current key.

    Surfacing a signed planning requirement after the approval changes the
    candidate key, so the same-round full set must not satisfy recovery.
    """
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    config = _staged_plan_config(tmp_path, plan_review_force_full=True)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    comments = [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments]
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )
    # The operator force-full run put the whole board in one round.
    assert {
        record.metadata.agent
        for record in plan_round.completed_reviews
        if record.metadata.state == "approved"
    } == {"Codex", "Gemini"}

    orchestrator_module._require_complete_canonical_plan_approval(
        comments,
        config=config,
        plan_text=plan_text,
        plan_round=plan_round,
        human_requirements=(),
        error_message="incomplete",
    )
    requirement = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
        body="Keep the public API unchanged.",
    )
    with pytest.raises(AgentLoopError, match="incomplete"):
        orchestrator_module._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(requirement,),
            error_message="incomplete",
        )


def _assert_managed_ci_plan_recovery_fails_closed(config, comments):
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )
    with pytest.raises(AgentLoopError, match="incomplete"):
        orchestrator_module._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


def test_managed_ci_plan_approval_rejects_post_approval_contradictory_key(tmp_path):
    """A later checkpoint contradicting the current key voids carried approvals."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    latest_checkpoint = None
    for comment in comments:
        match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", comment.body)
        if match is None:
            continue
        metadata = _decode_round_metadata(match.group("payload"))
        if metadata.flow == "plan" and metadata.scheduler_metadata_status == "valid":
            latest_checkpoint = metadata
    assert latest_checkpoint is not None
    stored_key = orchestrator_module._plan_key_from_payload(
        latest_checkpoint.plan_candidate_key
    )
    contradictory = replace(stored_key, aggregate_plan_identity="f" * 64)
    assert contradictory.subject == stored_key.subject
    comments = [
        *comments,
        SimpleNamespace(
            body=_attach_round_metadata(
                "Plan review scheduling audit.\n\n-- Orchestrator",
                replace(latest_checkpoint, plan_candidate_key=contradictory.as_dict()),
            )
        ),
    ]

    _assert_managed_ci_plan_recovery_fails_closed(config, comments)


def test_managed_ci_plan_approval_rejects_post_boundary_invalid_checkpoint(tmp_path):
    """An invalid checkpoint after the recovery boundary degrades the history."""
    config, comments = _completed_staged_plan_comments(tmp_path)
    plan_text, _plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )
    invalid = _invalid_plan_scheduler_comment(_plan_subject(plan_text))
    comments = [*comments, SimpleNamespace(body=invalid["body"])]

    _assert_managed_ci_plan_recovery_fails_closed(config, comments)


def test_managed_ci_plan_approval_all_reviewers_still_requires_one_round(tmp_path):
    """The compatibility policy keeps requiring the full set in one round."""
    _config, comments = _completed_staged_plan_comments(tmp_path)
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=6)
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        comments, configured_reviewers=orchestrator_module.reviewers(config)
    )

    with pytest.raises(AgentLoopError, match="incomplete"):
        orchestrator_module._require_complete_canonical_plan_approval(
            comments,
            config=config,
            plan_text=plan_text,
            plan_round=plan_round,
            human_requirements=(),
            error_message="incomplete",
        )


def test_staged_planning_withdrawn_requirement_blocks_carried_approvals(
    tmp_path, monkeypatch
):
    """`human-requirements-and-decomposition`, `carried-approval-requires-current-ack`.

    The primary's carried exact-key approval and the panel's fresh approval are
    both bound to the earlier surfaced requirement digest, so withdrawing a
    signed requirement at the live approval boundary must stop the run instead
    of letting those acknowledgements satisfy the gate for a requirement set
    that no longer exists.
    """
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )
    dispositions = [
        {
            "requirement_id": requirement_1.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        },
        {
            "requirement_id": requirement_2.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the audit trail.",
        },
    ]
    plan_output = structured_v1_plan_state().replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        f"- {requirement_1.requirement_id}: the plan preserves the public API.\n"
        f"- {requirement_2.requirement_id}: the plan preserves the audit trail.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
        gemini_outputs=[
            structured_plan_review(
                state="approved",
                reviewer="Google Gemini",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
    )
    real_get_issue_context = orchestrator_module.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        # The withdrawal lands only after the panel round has settled, so the
        # primary already holds a carried approval for the earlier digest.
        panel_ran = any(cmd[0] == "gemini" for cmd, _cwd in runner.commands)
        requirements = (
            (requirement_1,) if panel_ran else (requirement_1, requirement_2)
        )
        return replace(context, human_requirements=requirements)

    monkeypatch.setattr(orchestrator_module, "get_issue_context", _patched)

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"withdrawn hr-",
    ):
        run_issue_loop(
            runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
        )

    # Both reviewers approved, yet the plan is not carried into implementation.
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def _repaired_acknowledgement_fixtures():
    """A staged plan whose primary approves without acknowledging the requirement.

    Repair recovers the acknowledgement, so the reviewer's verdict is unchanged
    and only the signed-requirement marker and dispositions are added.
    """
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    dispositions = [
        {
            "requirement_id": requirement.requirement_id,
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        }
    ]
    plan_output = structured_v1_plan_state().replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        f"- {requirement.requirement_id}: the plan preserves the public API.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    # Structurally complete, so it passes reviewer validation and is posted as
    # written; only the signed-requirement marker is missing, which is exactly
    # what the post-reconciliation acknowledgement repair recovers.
    unacknowledged = structured_plan_review(
        state="approved",
        human_requirement_dispositions=dispositions,
    )
    repaired = structured_plan_review(
        state="approved",
        human_requirements_resolved=True,
        human_requirement_dispositions=dispositions,
    )
    panel = structured_plan_review(
        state="approved",
        reviewer="Google Gemini",
        human_requirements_resolved=True,
        human_requirement_dispositions=dispositions,
    )
    return requirement, plan_output, unacknowledged, repaired, panel


def test_staged_planning_persists_a_repaired_primary_acknowledgement(tmp_path, monkeypatch):
    """`carried-approval-requires-current-ack`, `reviewer-only-phase-advance`.

    A repaired acknowledgement must reach the durable record before the phase
    advance. The record posted with the original text stores no surfaced
    requirement IDs, so without the amendment the next round would reject the
    primary's approval as unacknowledged and invoke the primary again instead
    of opening the independent panel.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[unacknowledged],
        gemini_outputs=[panel],
    )
    real_get_issue_context = orchestrator_module.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        return replace(context, human_requirements=(requirement,))

    monkeypatch.setattr(orchestrator_module, "get_issue_context", _patched)

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired
    ):
        assert run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path),
            plan_first=True,
        ) == 0

    records = _plan_round_records(runner)
    codex_records = [
        record
        for record in records
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    # The amended record supersedes the one written before the repair, and it
    # is the one that carries the acknowledgement for the surfaced set.
    assert codex_records[-1].round_number == 1
    assert codex_records[-1].state == "approved"
    assert codex_records[-1].surfaced_reviewer_requirement_ids == (
        requirement.requirement_id,
    )
    prelaunch = [record for record in records if record.phase == "scheduler-prelaunch"]
    assert [record.scheduler_phase for record in prelaunch] == [
        "primary",
        "secondary-audit",
    ]
    # The panel opens instead of re-running the primary.
    assert prelaunch[1].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[1].scheduler_approved_reviewers
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1


def test_staged_planning_resume_reads_the_repaired_primary_acknowledgement(
    tmp_path, monkeypatch
):
    """`resume-no-duplicate-calls`: the repaired approval survives a restart.

    The amendment is durable, so a process that stops after the phase-advance
    record and restarts still carries the primary's acknowledged approval and
    invokes only the secondary panel.
    """
    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[unacknowledged],
        gemini_outputs=[panel],
    )
    real_get_issue_context = orchestrator_module.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(
            runner_arg, config=config, issue_number=issue_number
        )
        return replace(context, human_requirements=(requirement,))

    monkeypatch.setattr(orchestrator_module, "get_issue_context", _patched)
    config = _staged_plan_config(tmp_path)
    real_post = orchestrator_module.post_issue_comment

    def interrupt_after_the_advance(*args, **kwargs):
        result = real_post(*args, **kwargs)
        if kwargs["body"].startswith("Plan review phase advance to round 2."):
            raise KeyboardInterrupt
        return result

    with patch(
        "coding_review_agent_loop.orchestrator.attempt_repair", return_value=repaired
    ):
        with patch.object(
            orchestrator_module,
            "post_issue_comment",
            side_effect=interrupt_after_the_advance,
        ):
            with pytest.raises(KeyboardInterrupt):
                run_issue_loop(
                    runner, issue_number=56, config=config, plan_first=True
                )

        calls_before = [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"claude", "codex", "gemini"}
        ]
        assert calls_before.count("codex") == 1

        assert run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True
        ) == 0

    records = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in records
        if record.phase == "scheduler-prelaunch"
    }
    assert prelaunch[2].scheduler_phase == "secondary-audit"
    assert prelaunch[2].scheduler_selected_reviewers == ("Gemini",)
    assert "Codex" in prelaunch[2].scheduler_approved_reviewers
    # The restart adds the panel reviewer and nothing else.
    after = [
        cmd[0]
        for cmd, _cwd in runner.commands
        if cmd[0] in {"claude", "codex", "gemini"}
    ]
    assert after == [*calls_before, "gemini"]


def test_staged_planning_round_budget_diagnostic_is_distinct(tmp_path):
    """`round-budget-diagnostic`: exhaustion during a pending phase advance."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path, max_rounds=1),
            plan_first=True,
        )

    message = str(excinfo.value)
    assert "reviewer-only plan phase advance was still pending" in message
    # The diagnostic names the outstanding phase, not the primary round that ran.
    assert "Outstanding phase: secondary-audit" in message
    assert "Gemini" in message
    assert "--max-rounds" in message
    assert "still reported blocking plan issues" not in message


def test_staged_planning_stops_on_premature_secondary_plan_review(tmp_path):
    """`pre-panel-safety-diagnostic`: no reviewer and no planner turn."""
    comments, _canonical = _staged_plan_history(
        tmp_path,
        reviewer_records=[
            (
                "Gemini",
                "blocking",
                (
                    UnresolvedReviewItem(
                        item_id="item-1",
                        reviewer="Gemini",
                        source_round=1,
                        text="Secondary-owned plan finding.",
                        status="blocking",
                    ),
                ),
            )
        ],
    )
    runner = _FakeRunner(issue_comments=comments)

    with pytest.raises(orchestrator_module.PlanPrePanelSafetyError) as excinfo:
        run_issue_loop(
            runner, issue_number=56, config=_staged_plan_config(tmp_path), plan_first=True
        )

    assert "no qualified panel opening exists" in str(excinfo.value)
    assert any(
        "Plan review scheduling diagnostic" in comment["body"]
        for comment in runner.issue_comments
    )
    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )


def test_plan_review_force_full_recovers_the_premature_secondary_review(tmp_path):
    """`operator-force-full-override`: the override authorizes the board."""
    comments, _canonical = _staged_plan_history(
        tmp_path,
        # The interrupted round is reconciled, so a resume rehydrates every
        # current-round item, including the superseded secondary's claim.
        reconciliation_record=True,
        reviewer_records=[
            (
                "Gemini",
                "blocking",
                (
                    UnresolvedReviewItem(
                        item_id="item-1",
                        reviewer="Gemini",
                        source_round=1,
                        text="Secondary-owned plan finding.",
                        status="blocking",
                    ),
                ),
            )
        ],
    )
    runner = _FakeRunner(
        issue_comments=comments,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )

    # The override selects the complete board rather than stopping with the
    # planning diagnostic, and the superseded review establishes no obligation,
    # so the freshly invoked board settles the round.
    assert run_issue_loop(
        runner,
        issue_number=56,
        config=_staged_plan_config(tmp_path, plan_review_force_full=True, max_rounds=1),
        plan_first=True,
    ) == 0

    prelaunch = [
        record for record in _plan_round_records(runner) if record.phase == "scheduler-prelaunch"
    ]
    # The last checkpoint is this run's decision; the first is seeded history.
    decision = prelaunch[-1]
    assert decision.scheduler_phase == "full-board"
    assert set(decision.scheduler_selected_reviewers) == {"Codex", "Gemini"}
    assert decision.scheduler_force_full is True
    assert decision.scheduler_force_full_source == "operator"
    assert "superseded premature secondary plan review" in " ".join(
        decision.scheduler_reasons
    )
    # The superseded review establishes no approval and no ownership: the same
    # secondary is freshly invoked, and its earlier blocking claim never
    # becomes a durable obligation that outlives the override.
    assert any(cmd[0] == "gemini" for cmd, _cwd in runner.commands)
    fresh_reviews = [
        record
        for record in _plan_round_records(runner)
        if record.role == "reviewer" and not record.new_items
    ]
    assert {record.agent for record in fresh_reviews} == {"Codex", "Gemini"}
    # The superseded review reaches its own author as explicitly
    # non-authoritative context, and no other reviewer.
    gemini_prompt = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["gemini"]
    ][-1]
    flat = " ".join(gemini_prompt.split())
    assert (
        "Superseded pre-panel plan review context (non-authoritative; context only):"
        in gemini_prompt
    )
    assert "not a finding, not a plan-item disposition, not an approval" in flat
    assert (
        "none of its claims entered the unresolved plan-item ledger "
        "(superseded plan item IDs: item-1)"
    ) in flat
    assert "- Earlier claim: Secondary-owned plan finding." in gemini_prompt
    codex_prompt = [
        command[-1] for command, _cwd in runner.commands if command[:1] == ["codex"]
    ][-1]
    assert "Superseded pre-panel plan review context" not in codex_prompt
    # The reconciled resume rehydrates the round's items, but the superseded
    # claim establishes no obligation: no planner turn is needed, and the
    # fresh approvals settle the round with no surviving must-fix item.
    assert not any(cmd[0] == "claude" for cmd, _cwd in runner.commands)
    posted = _plan_round_records(runner)[len(comments):]
    assert posted, "the resumed round must post its own durable records"
    for record in posted:
        assert "item-1" not in {item.item_id for item in record.new_items}
        assert "item-1" not in {item.item_id for item in record.prior_items}


def test_staged_planning_stops_when_planning_history_cannot_be_extracted(tmp_path):
    """`undecodable-planning-history-stop`: only class D stops the run, and the
    operator override cannot recover it."""
    config = _staged_plan_config(tmp_path, plan_review_force_full=True)
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="bot",
                created_at="2026-01-01T00:00:30Z",
                body="Plan round record.\n<!-- AGENT_LOOP_META: not-a-valid-payload -->",
            ),
        ),
    )
    runner = _FakeRunner()

    with pytest.raises(orchestrator_module.PlanPrePanelSafetyError) as excinfo:
        orchestrator_module._run_plan_first_loop(
            runner,
            issue_number=56,
            config=config,
            memory=None,
            issue_context=issue_context,
            usage_context=orchestrator_module._new_usage_context(config),
        )

    message = str(excinfo.value)
    assert "could not be extracted" in message
    assert "--plan-review-force-full cannot authorize" in message
    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )
    assert any(
        "Plan review scheduling diagnostic (startup)" in comment["body"]
        for comment in runner.issue_comments
    )


def test_staged_planning_panel_blocker_routes_to_owner_plus_primary(tmp_path):
    """`panel-blocker-remediation`: a `same-plan` panel finding keeps its owner.

    Regression for the planning obligation status set: `same-plan` is the
    planning counterpart of `same-pr`, so it must stay an active obligation or
    the scheduler falls through to a final sweep instead of remediation.
    """
    fresh = structured_v1_plan_state()
    base = orchestrator_module.AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Name the rollout owner in the plan steps.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan step names the rollout owner.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "plan_steps",
                "value": ["Implement the reviewed scope and name the rollout owner."],
            }
        ],
    }
    patch_text = (
        json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    runner = _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved", prior_plan_item_dispositions=resolved
            ),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="One plan step omits the rollout owner.",
                same_plan_followups=["Name the rollout owner in the plan steps."],
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
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    assert phases[:2] == ["primary", "secondary-audit"]
    assert "remediation" in phases, phases
    remediation = prelaunch[phases.index("remediation")]
    # Owner-scoped: the `same-plan` finding owner plus the primary, never the
    # complete board, and no automatic full-board latch is raised.
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert [name for name, _why in remediation.scheduler_paused_reviewers] == [
        "Antigravity"
    ]
    assert remediation.scheduler_active_owners == ("Gemini",)
    assert remediation.scheduler_force_full is False
    assert remediation.scheduler_force_full_source is None
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    assert "full-board" not in phases
    # The posted audit must describe the ordinary owner-scoped remediation as
    # what it is, not as the complete-board post-panel fallback, so its decision
    # kind agrees with the phase, board, and force-full fields beside it.
    remediation_body = _plan_audit_body(runner, round_number=remediation.round_number)
    assert (
        "owner-scoped remediation decision (partial board, no automatic latch)"
        in remediation_body
    )
    assert "post-panel fallback (complete board" not in remediation_body
    assert "- Phase: `remediation`" in remediation_body
    assert "- Force-full: False (source: none)" in remediation_body


def _remediation_plan_fixtures():
    """A fresh generation-1 plan plus a narrow plan-step remediation patch."""
    fresh = structured_v1_plan_state()
    base = orchestrator_module.AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(fresh), round_number=1
    )
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Name the rollout owner in the plan steps.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan step names the rollout owner.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "plan_steps",
                "value": ["Implement the reviewed scope and name the rollout owner."],
            }
        ],
    }
    patch_text = (
        json.dumps(patch) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    return fresh, patch_text


def _remediation_runner():
    """Primary gate, panel blocker, narrow remediation, then the final sweep."""
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved", prior_plan_item_dispositions=resolved
            ),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Gemini",
                summary="One plan step omits the rollout owner.",
                blocking_plan_issues=["Name the rollout owner in the plan steps."],
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


def test_staged_planning_advance_after_remediation_names_the_final_sweep(tmp_path):
    """`reviewer-only-phase-advance`: the advance names the pending phase.

    After an owner-scoped remediation round the decision in hand still reads
    `remediation`, but the outstanding reviewer-only round is the final
    exact-plan sweep for the secondary that has no approval of the new key.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    assert phases[:3] == ["primary", "secondary-audit", "remediation"]
    assert "final-secondary-sweep" in phases, phases
    advance_bodies = {
        comment["body"].splitlines()[0]: comment["body"]
        for comment in runner.issue_comments
        if "Plan review phase advance to round" in comment["body"]
    }
    # The panel advance names the audit; the post-remediation advance names the
    # sweep, never the remediation round that just ran.
    assert (
        "Outstanding phase after this advance: `secondary-audit`"
        in advance_bodies["Plan review phase advance to round 2."]
    )
    post_remediation = advance_bodies["Plan review phase advance to round 4."]
    assert (
        "Outstanding phase after this advance: `final-secondary-sweep`"
        in post_remediation
    )
    assert "`remediation`" not in post_remediation
    assert "Antigravity" in post_remediation


def test_staged_planning_resumed_remediation_stays_owner_scoped(tmp_path):
    """`panel-blocker-remediation`: resume keeps the narrow classification.

    The classifier decides `narrow` from the authenticated semantic patch and
    the cross-cutting contracts of the state it was bound to, and the ledger
    stays reconstructible because the run has already accounted for the items
    recorded under the earlier plan subject. An interruption between the
    remediation coder turn and its scheduler checkpoint must therefore resume
    into the same owner-scoped board and the same final sweep an uninterrupted
    run produces, with no complete-board re-invocation and no latch.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = orchestrator_module.post_issue_comment
    audits = {"count": 0}

    def interrupt_before_remediation_checkpoint(*args, **kwargs):
        # Rounds 1 and 2 post their own scheduler checkpoints first, so the
        # third one is the remediation round's, posted after the round-3
        # planner turn has already been recorded.
        if kwargs["body"].startswith("Plan review scheduling audit."):
            audits["count"] += 1
            if audits["count"] == 3:
                raise KeyboardInterrupt
        return real_post(*args, **kwargs)

    with patch.object(
        orchestrator_module,
        "post_issue_comment",
        side_effect=interrupt_before_remediation_checkpoint,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    interrupted = _plan_round_records(runner)
    # The remediation planner turn is durable; its scheduler checkpoint is not.
    assert any(
        record.round_number == 3 and record.role == "coder" for record in interrupted
    )
    assert not any(
        record.round_number == 3 and record.phase == "scheduler-prelaunch"
        for record in interrupted
    )

    def reviewer_calls():
        return [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"codex", "gemini", "agy"}
        ]

    calls_before = reviewer_calls()

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in resumed
        if record.phase == "scheduler-prelaunch"
    }
    remediation = prelaunch[3]
    # Owner-scoped: the ledger owner plus the primary, and no latch.
    assert remediation.scheduler_phase == "remediation"
    assert set(remediation.scheduler_selected_reviewers) == {"Gemini", "Codex"}
    assert [name for name, _why in remediation.scheduler_paused_reviewers] == [
        "Antigravity"
    ]
    assert remediation.scheduler_active_owners == ("Gemini",)
    assert remediation.scheduler_force_full is False
    assert remediation.scheduler_force_full_source is None
    assert "narrow plan remediation" in " ".join(remediation.scheduler_reasons)
    # The round-3 approvals are carried: the sweep invokes only the reviewer
    # that still lacks an exact-plan approval, with no post-panel latch.
    sweep = prelaunch[4]
    assert sweep.scheduler_phase == "final-secondary-sweep"
    assert sweep.scheduler_selected_reviewers == ("Antigravity",)
    assert sweep.scheduler_force_full is False
    assert sweep.scheduler_force_full_source is None
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )
    reviewer_rounds = [
        (record.agent, record.round_number)
        for record in resumed
        if record.role == "reviewer"
    ]
    assert reviewer_rounds.count(("Antigravity", 4)) == 1
    assert ("Codex", 4) not in reviewer_rounds
    assert ("Gemini", 4) not in reviewer_rounds
    # The resumed run re-invokes nothing already settled: it adds exactly the
    # remediation owner plus primary and the single sweep reviewer.
    assert reviewer_calls() == [*calls_before, "codex", "gemini", "agy"]


def test_staged_planning_resume_on_the_phase_advance_seam_keeps_the_sweep_narrow(tmp_path):
    """`reviewer-only-phase-advance`: the cleared ledger survives a restart.

    A reviewer-only advance carries no unresolved item, so its durable record
    persists an empty ledger. A process that stops on exactly that seam and
    restarts must still read the ledger as reconstructible -- the remediated
    item is cleared by recorded history, not merely by the interrupted run's
    memory -- so the final sweep stays narrow and invokes only the reviewer
    that still lacks an exact-plan approval.
    """
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8
    )
    real_post = orchestrator_module.post_issue_comment

    def interrupt_after_the_sweep_advance(*args, **kwargs):
        # Post the advance first: the seam under test is a restart with that
        # record durable and nothing after it.
        result = real_post(*args, **kwargs)
        if kwargs["body"].startswith("Plan review phase advance to round 4."):
            raise KeyboardInterrupt
        return result

    with patch.object(
        orchestrator_module,
        "post_issue_comment",
        side_effect=interrupt_after_the_sweep_advance,
    ):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    interrupted = _plan_round_records(runner)
    advance = [
        record
        for record in interrupted
        if record.phase == "plan-phase-advance" and record.round_number == 4
    ]
    assert len(advance) == 1
    # The seam: the advance record carries no ledger at all, because
    # remediation cleared the only item before it was written.
    assert advance[0].prior_items == ()
    assert not any(
        record.round_number == 4 and record.phase == "scheduler-prelaunch"
        for record in interrupted
    )

    def reviewer_calls():
        return [
            cmd[0]
            for cmd, _cwd in runner.commands
            if cmd[0] in {"codex", "gemini", "agy"}
        ]

    calls_before = reviewer_calls()

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed = _plan_round_records(runner)
    prelaunch = {
        record.round_number: record
        for record in resumed
        if record.phase == "scheduler-prelaunch"
    }
    sweep = prelaunch[4]
    assert sweep.scheduler_phase == "final-secondary-sweep"
    assert sweep.scheduler_selected_reviewers == ("Antigravity",)
    assert sweep.scheduler_force_full is False
    assert sweep.scheduler_force_full_source is None
    assert not any(
        record.scheduler_phase == "full-board" for record in prelaunch.values()
    )
    reviewer_rounds = [
        (record.agent, record.round_number)
        for record in resumed
        if record.role == "reviewer"
    ]
    assert reviewer_rounds.count(("Antigravity", 4)) == 1
    assert ("Codex", 4) not in reviewer_rounds
    assert ("Gemini", 4) not in reviewer_rounds
    assert reviewer_calls() == [*calls_before, "agy"]


def test_staged_planning_round_budget_names_the_outstanding_final_sweep(tmp_path):
    """`round-budget-diagnostic`: exhaustion after remediation names the sweep."""
    runner = _remediation_runner()
    config = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=3
    )

    with pytest.raises(AgentLoopError) as excinfo:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    message = str(excinfo.value)
    assert "reviewer-only plan phase advance was still pending" in message
    assert "Outstanding phase: final-secondary-sweep" in message
    assert "Antigravity" in message
    assert "still reported blocking plan issues" not in message


def test_staged_planning_contract_drift_stops_a_default_policy_restart(tmp_path):
    """A persisted staged planning contract is immutable across a restart.

    Restarting with the compatibility default must stop rather than silently
    continue on a different contract.
    """
    comments, _canonical = _staged_plan_history(tmp_path)
    runner = _FakeRunner(issue_comments=comments)

    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(
            runner,
            issue_number=56,
            config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3),
            plan_first=True,
        )

    assert not any(
        cmd[0] in {"claude", "codex", "gemini"} for cmd, _cwd in runner.commands
    )

    # A restart that changes only the primary is rejected the same way.
    drifted = _FakeRunner(issue_comments=comments)
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(
            drifted,
            issue_number=56,
            config=_staged_plan_config(tmp_path, primary_plan_reviewer="gemini"),
            plan_first=True,
        )


def _invalid_plan_scheduler_comment(subject, *, created_at="2026-01-01T00:00:30Z"):
    """A planning scheduler record whose contract cannot be reconstructed."""
    from coding_review_agent_loop.round_transport import encode_mapping

    payload = {
        "flow": "plan",
        "role": "summary",
        "agent": "Orchestrator",
        "round_number": 1,
        "subject": subject,
        "phase": "scheduler-prelaunch",
        # Missing `required_reviewers`: the planning branch reports `invalid`.
        "scheduler_contract": {"policy": "primary-then-panel"},
        "scheduler_obligation_digest": "0" * 16,
        "scheduler_selected_reviewers": ["Codex"],
        "scheduler_paused_reviewers": [["Gemini", "primary phase"]],
        "scheduler_reasons": ["primary phase"],
        "scheduler_final_sweep": False,
        "scheduler_force_full": False,
        "scheduler_calls_avoided": 1,
    }
    return {
        "author": {"login": "bot"},
        "createdAt": created_at,
        "body": (
            "Plan review scheduling audit.\n\n-- Orchestrator\n"
            f"<!-- AGENT_LOOP_META: {encode_mapping(payload)} -->"
        ),
    }


def test_invalid_planning_history_recovers_after_its_fallback_round(tmp_path):
    """`legacy-planning-metadata-fallback`: class B continues and progresses.

    An invalid scheduler record before the current recovery boundary must not
    pin every later round to the strict pre-panel fallback, which would suppress
    each fresh exact-key primary approval until the round budget ran out.
    """
    comments, canonical = _staged_plan_history(tmp_path, scheduler_record=False)
    comments.append(_invalid_plan_scheduler_comment(_plan_subject(canonical)))
    runner = _FakeRunner(
        issue_comments=comments,
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini")
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=_staged_plan_config(tmp_path, max_rounds=6),
        plan_first=True,
    ) == 0

    prelaunch = [
        record
        for record in _plan_round_records(runner)
        if record.phase == "scheduler-prelaunch"
        and record.scheduler_metadata_status == "valid"
    ]
    phases = [record.scheduler_phase for record in prelaunch]
    # The invalid record degrades exactly one round, then the valid checkpoint
    # it wrote becomes the recovery boundary and the run advances.
    assert phases == ["primary", "secondary-audit"]
    assert "strict pre-panel fallback:" in " ".join(prelaunch[0].scheduler_reasons)
    assert "strict pre-panel fallback:" not in " ".join(prelaunch[1].scheduler_reasons)
    # No planner turn was fabricated and no reviewer was invoked twice.
    assert not any(cmd[0] == "claude" for cmd, _cwd in runner.commands)
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "codex") == 1
    assert sum(1 for cmd, _cwd in runner.commands if cmd[0] == "gemini") == 1


def test_operator_opening_requires_a_complete_candidate_key(tmp_path):
    """A partial-key record can never establish a qualified panel opening."""
    from coding_review_agent_loop.round_state import PostedRoundRecord

    partial = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity=None,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest="e" * 16,
    )
    complete = replace(partial, execution_strategy_identity="c" * 32)

    def _force_full_record(index, key):
        return PostedRoundRecord(
            index=index,
            body="Plan review scheduling audit.",
            metadata=PostedRoundMetadata(
                flow="plan",
                role="summary",
                agent="Orchestrator",
                round_number=1,
                subject="a" * 64,
                phase="scheduler-prelaunch",
                scheduler_contract=orchestrator_module.make_plan_contract(
                    ("Codex", "Gemini"), "primary-then-panel", "Codex"
                ).as_dict(),
                scheduler_obligation_digest="0" * 16,
                scheduler_selected_reviewers=("Codex", "Gemini"),
                scheduler_paused_reviewers=(),
                scheduler_reasons=("operator force-full",),
                scheduler_final_sweep=False,
                scheduler_force_full=True,
                scheduler_force_full_source="operator",
                scheduler_calls_avoided=0,
                scheduler_phase="full-board",
                scheduler_primary_reviewer="Codex",
                plan_candidate_key=key.as_dict(),
            ),
        )

    partial_evidence = orchestrator_module._derive_plan_panel_evidence(
        (_force_full_record(1, partial),),
        primary_reviewer="Codex",
        required_reviewers=("Codex", "Gemini"),
    )
    complete_evidence = orchestrator_module._derive_plan_panel_evidence(
        (_force_full_record(1, complete),),
        primary_reviewer="Codex",
        required_reviewers=("Codex", "Gemini"),
    )

    assert partial_evidence.opened is False
    assert complete_evidence.opened is True
    assert complete_evidence.opening_source == "operator"


# --- Staged parent terminal state and human stages (#918) -----------------


def _staged_created(automations):
    return tuple(
        CreatedPhaseIssue(
            phase=RecordedPhase(title=f"Stage {index}", automation=automation),
            issue_url=f"https://github.com/OWNER/REPO/issues/{98 + index}",
            issue_number=98 + index,
        )
        for index, automation in enumerate(automations, start=1)
    )


def _staged_outcome(created, *, retained=None, final=None, plan_hash="staged-plan-hash"):
    return StagedTopologyOutcome(
        created=created,
        stage_ids=tuple(f"stage-{index}" for index in range(1, len(created) + 1)),
        automations=tuple(item.phase.automation for item in created),
        plan_hash=plan_hash,
        mode="implement-by-phase",
        topology_source="model",
        retained_parent_scope=retained,
        final_integration_work=final,
    )


def _staged_handoff(phase_index, child_issue_number, *, plan_hash="staged-plan-hash"):
    return PhaseImplementationHandoffMetadata(
        parent_issue=55,
        plan_hash=plan_hash,
        mode="implement-by-phase",
        phase_index=phase_index,
        phase_title=f"Stage {phase_index}",
        automation="agent-pr",
        child_issue_number=child_issue_number,
        child_issue_url=f"https://github.com/OWNER/REPO/issues/{child_issue_number}",
        stage_id=f"stage-{phase_index}",
    )


def _merged_child_runner(states, *, pr_states=None):
    """FakeRunner presenting per-child issue states and canonical merged PRs."""
    pr_states = pr_states or {}
    issue_payloads = {}
    issue_comments = {}
    pr_payloads = {}
    for number, state in states.items():
        issue_payloads[number] = {"state": state}
        pr_number = pr_states.get(number)
        if pr_number is None:
            issue_comments[number] = []
            continue
        issue_comments[number] = [child_pr_handoff_comment(number, pr_number)]
        pr_payloads[pr_number] = pr_payload_for_state(pr_number, "MERGED")
    return _FakeRunner(
        issue_payloads_by_number=issue_payloads,
        issue_comments_by_number=issue_comments,
        pr_payloads_by_number=pr_payloads,
    )


def _dispatch_staged(runner, tmp_path, outcome, handoffs, monkeypatch, **overrides):
    monkeypatch.setattr(
        phase_progress_module,
        "find_phase_implementation_handoffs_for_parent",
        lambda *_args, **_kwargs: tuple(handoffs),
    )
    parent = IssueContext(
        number=55, repo="OWNER/REPO", title="Parent", body="Parent",
        url="https://github.com/OWNER/REPO/issues/55", comments=(),
    )
    monkeypatch.setattr(
        orchestrator_module, "get_issue_context",
        lambda _runner, *, config, issue_number: parent,
    )
    kwargs = dict(
        config=make_config(tmp_path), memory=None, usage_context=SimpleNamespace(),
        issue_number=55, current_plan="Approved staged plan", plan_subject="plan-subject",
        outcome=outcome, recommendation=None,
        approved_plan_context=SimpleNamespace(matrix_available=False, risk_test_matrix_payload=None),
        issue_context=parent, mode="implement-by-phase", coder_session_id=None,
    )
    kwargs.update(overrides)
    return orchestrator_module._dispatch_current_decomposition_phase(runner, **kwargs)


def test_staged_terminal_report_names_required_parent_obligations(tmp_path, monkeypatch, capsys):
    """Matrix row `terminal-required-obligations`."""
    created = _staged_created(("agent-pr", "agent-pr"))
    outcome = _staged_outcome(
        created,
        retained=RetainedParentScope(
            plan_subject="plan-subject", plan_hash="staged-plan-hash",
            excerpt="Parent keeps the rollout note.", status="required",
            deliverables=("The rollout note.",),
            acceptance_criteria=("The rollout note is published.",),
        ),
        final=ExecutionAllocation(
            status="required", deliverables=("Wire the stages together.",),
            acceptance_criteria=("End-to-end behavior passes.",),
            covered_scope_item_ids=("scope-3",),
        ),
    )
    runner = _merged_child_runner(
        {99: "closed", 100: "closed"}, pr_states={99: 912, 100: 913}
    )
    monkeypatch.setattr(
        orchestrator_module, "_dispatch_decomposition_child",
        lambda *_a, **_k: pytest.fail("a delivered topology must not dispatch a child"),
    )

    result = _dispatch_staged(
        runner, tmp_path, outcome,
        (_staged_handoff(1, 99), _staged_handoff(2, 100)), monkeypatch,
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "all 2 staged phases are delivered" in output
    assert "Retained-parent obligations: required; this is operator-owned parent work." in output
    assert "The rollout note." in output
    assert "Final-integration obligations: required" in output
    assert "Wire the stages together." in output
    assert "End-to-end behavior passes." in output
    assert "remains open pending that operator-owned parent work" in output
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)


def test_staged_terminal_report_renders_a_human_stage_without_a_pr(tmp_path, monkeypatch, capsys):
    """Matrix row `terminal-human-stage-line`."""
    created = _staged_created(("agent-pr", "human-action"))
    outcome = _staged_outcome(created)
    runner = _merged_child_runner({99: "closed", 100: "closed"}, pr_states={99: 912})

    result = _dispatch_staged(runner, tmp_path, outcome, (_staged_handoff(1, 99),), monkeypatch)

    assert result == 0
    output = capsys.readouterr().out
    assert "stage-1: delivered by child issue #99 with merged PR #912" in output
    assert (
        "stage-2: delivered by human work on child issue #100; "
        "its closure is the operator attestation." in output
    )
    assert "PR #None" not in output
    # No PR evidence is read for the human-owned stage.
    assert not any(
        cmd[:3] == ["gh", "pr", "view"] and "913" in cmd for cmd, _cwd in runner.commands
    )


def test_staged_pending_human_stage_stops_then_advances_when_closed(tmp_path, monkeypatch, capsys):
    """Matrix rows `human-stage-pending-stops` and `closed-human-stage-advances`."""
    created = _staged_created(("agent-pr", "human-action", "agent-pr"))
    outcome = _staged_outcome(created)
    handoffs = (_staged_handoff(1, 99),)
    monkeypatch.setattr(
        orchestrator_module, "_dispatch_decomposition_child",
        lambda *_a, **_k: pytest.fail("a human-owned stage must not dispatch an agent"),
    )
    pending = _merged_child_runner({99: "closed", 100: "open"}, pr_states={99: 912})

    assert _dispatch_staged(pending, tmp_path, outcome, handoffs, monkeypatch) == 0
    output = capsys.readouterr().out
    assert "phase 2 (`stage-2`) requires human work (human-action) on child issue #100" in output
    assert not any(
        "AGENT_PLAN_PHASE_IMPLEMENTATION" in comment for comment in pending.comments
    )

    dispatched = []
    monkeypatch.setattr(
        orchestrator_module, "_dispatch_decomposition_child",
        lambda *_a, **kwargs: dispatched.append(kwargs) or 0,
    )
    advanced = _merged_child_runner(
        {99: "closed", 100: "closed", 101: "open"}, pr_states={99: 912}
    )

    assert _dispatch_staged(advanced, tmp_path, outcome, handoffs, monkeypatch) == 0
    assert len(dispatched) == 1
    assert dispatched[0]["phase_index"] == 3
    assert dispatched[0]["created"].issue_number == 101


def test_get_issue_state_normalizes_lowercase_states(tmp_path):
    """Matrix row `issue-state-read-fails-closed`: the accepted case."""
    runner = _FakeRunner(issue_payloads_by_number={99: {"state": "closed"}})
    assert get_issue_state(runner, config=make_config(tmp_path), issue_number=99) == "CLOSED"


@pytest.mark.parametrize(
    ("payload", "returncode", "message"),
    [
        ({"state": None}, 0, "Unable to determine the state of issue #99"),
        ({"state": 7}, 0, "Unable to determine the state of issue #99"),
        ({"state": "open"}, 1, "`gh` exited 1"),
        ({"state": "open", "is_pr": True}, 0, "is a pull request, not an issue"),
        ({"state": "merged"}, 0, "reported unexpected state"),
    ],
)
def test_get_issue_state_fails_closed(tmp_path, payload, returncode, message):
    """Matrix row `issue-state-read-fails-closed`: every rejected case.

    The stub reproduces the real `Runner.run` check semantics, so a reader that
    left `check` at its default would raise the generic command failure instead
    of the contextual diagnostic and this test would fail.
    """

    class _Runner:
        def run(self, cmd, *, check=True, **_kwargs):
            if check and returncode != 0:
                raise AgentLoopError(
                    f"Command failed with exit {returncode}: {' '.join(cmd)}"
                )
            return CommandResult(cmd, None, json.dumps(payload), "", returncode)

    with pytest.raises(AgentLoopError, match=re.escape(message)):
        get_issue_state(_Runner(), config=make_config(tmp_path), issue_number=99)


def test_get_issue_state_reads_with_check_disabled(tmp_path):
    """A nonzero `gh` exit must reach the contextual diagnostic, not the generic one."""
    seen = {}

    class _Runner:
        def run(self, cmd, *, check=True, **_kwargs):
            seen["check"] = check
            if check:
                raise AgentLoopError("Command failed with exit 1: " + " ".join(cmd))
            return CommandResult(cmd, None, "", "not found", 1)

    with pytest.raises(AgentLoopError, match=re.escape("`gh` exited 1")):
        get_issue_state(_Runner(), config=make_config(tmp_path), issue_number=99)
    assert seen["check"] is False


# ---------------------------------------------------------------------------
# Approved child plan that fails the inherited check before any handoff (#936)
# ---------------------------------------------------------------------------

import test_child_plan_provenance as _m936_cpp  # noqa: E402


def _m936_patch(base_plan_text, row, *, summary):
    """Semantic revision with no reviewer item: the guard mints none."""
    patch = json.loads(_m936_cpp._semantic_patch(base_plan_text, row, summary=summary).split("\n<!--", 1)[0])
    patch["prior_plan_item_dispositions"] = []
    return json.dumps(patch) + _m936_cpp.PLAN_FOOTER


def _m936_historical_approved_plan(tmp_path, *, reviewers_kwargs=None):
    """A weakened child plan approved before the inherited check existed."""
    weak = _m936_cpp._child_plan_state(_m936_cpp._weak_child_row())
    first = _m936_cpp._ChildPlanningRunner(
        claude_outputs=[weak],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
    )
    assert orchestrator_module.run_issue_loop(
        first, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
    ) == 0
    return weak, list(first.issue_comments)


def test_m936_approved_inadmissible_plan_is_revised_instead_of_dead_ending(tmp_path, monkeypatch):
    weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    good = _m936_patch(weak, _m936_cpp._child_row(), summary="Inherited rows restored.")
    resumed = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(history),
        claude_outputs=[good],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
    )
    assert orchestrator_module.run_issue_loop(
        resumed, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
    ) == 0

    posted = [str(item["body"]) for item in resumed.issue_comments[len(history):]]
    audits = [body for body in posted if "is inadmissible under the inherited-matrix contract" in body]
    assert len(audits) == 1 and "AGENT_LOOP_META" not in audits[0]
    planner_prompts = _m936_cpp._agent_prompts(resumed, "claude")
    assert len(planner_prompts) == 1
    assert "Trusted orchestration correction record" in planner_prompts[0]
    assert "row-owned: forbidden_side_effects dropped" in planner_prompts[0]
    assert "Orchestrator inherited-matrix check (not a reviewer finding)" in planner_prompts[0]
    # No synthetic reviewer item and no signed record or digest is involved.
    revised = [body for body in posted if "Inherited rows restored." in body]
    assert revised
    from coding_review_agent_loop.round_state import _decode_round_metadata
    from coding_review_agent_loop.round_transport import ROUND_RESUME_MARKER_RE

    metadata = _decode_round_metadata(ROUND_RESUME_MARKER_RE.search(revised[0]).group("payload"))
    assert metadata.round_number == 2 and metadata.prior_items == ()
    assert metadata.plan_supersession_digest is None
    # The revised plan was reviewed again by the board and only then approved.
    assert len(_m936_cpp._agent_prompts(resumed, "codex")) == 1
    assert "Inherited coverage check FAILED" not in _m936_cpp._agent_prompts(resumed, "codex")[0]


def test_m936_guard_audit_comment_is_not_duplicated_and_recheck_exhaustion_persists(
    tmp_path, monkeypatch
):
    weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    still_weak = _m936_patch(weak, None, summary=_m936_cpp.WEAK_SUMMARY)
    resumed = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(history), claude_outputs=[still_weak] * 3,
    )
    with pytest.raises(AgentLoopError, match="consecutive plan candidates that weaken"):
        orchestrator_module.run_issue_loop(
            resumed, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
        )
    assert len(_m936_cpp._agent_prompts(resumed, "claude")) == 3
    assert _m936_cpp.WEAK_SUMMARY not in _m936_cpp._published(resumed)
    assert len(resumed.diagnostic_posts) == 1
    # A rerun finds the audit comment and does not post it again.
    rerun = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(resumed.issue_comments), claude_outputs=[still_weak] * 3,
    )
    with pytest.raises(AgentLoopError):
        orchestrator_module.run_issue_loop(
            rerun, issue_number=56, config=_m936_cpp._plan_config(tmp_path), plan_first=True
        )
    audits = [
        item for item in rerun.issue_comments
        if "is inadmissible under the inherited-matrix contract" in str(item["body"])
    ]
    assert len(audits) == 1


def test_m936_guard_at_max_rounds_names_the_round_budget(tmp_path, monkeypatch):
    _weak, history = _m936_historical_approved_plan(tmp_path)
    _m936_cpp._bind_child_planning(monkeypatch)
    resumed = _m936_cpp._ChildPlanningRunner(issue_comments=list(history))
    config = make_config(
        tmp_path, max_rounds=1, plan_execution_mode="plan-only",
        execution_strategy_contract_required=True,
    )
    with pytest.raises(AgentLoopError, match="Raise --max-rounds"):
        orchestrator_module.run_issue_loop(resumed, issue_number=56, config=config, plan_first=True)
    assert _m936_cpp._agent_prompts(resumed, "claude") == []
    assert len(resumed.issue_comments) == len(history)


_M936_STAGED = dict(
    reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
    primary_plan_reviewer="codex",
)


def test_m936_guard_full_board_latch_survives_a_restart_after_the_revised_round(
    tmp_path, monkeypatch
):
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    def staged_config():
        return make_config(
            tmp_path, max_rounds=8, plan_execution_mode="plan-only",
            execution_strategy_contract_required=True, **_M936_STAGED,
        )

    gemini_approval = _m936_cpp.structured_plan_review(state="approved", reviewer="Google Gemini")
    weak = _m936_cpp._child_plan_state(_m936_cpp._weak_child_row())
    first = _m936_cpp._ChildPlanningRunner(
        claude_outputs=[weak],
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[gemini_approval],
    )
    assert orchestrator_module.run_issue_loop(
        first, issue_number=56, config=staged_config(), plan_first=True
    ) == 0
    _m936_cpp._bind_child_planning(monkeypatch)
    good = _m936_patch(weak, _m936_cpp._child_row(), summary="Inherited rows restored.")
    # The run stops right after the revised round: no reviewer output is scripted.
    interrupted = _m936_cpp._ChildPlanningRunner(
        issue_comments=list(first.issue_comments), claude_outputs=[good],
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        orchestrator_module.run_issue_loop(
            interrupted, issue_number=56, config=staged_config(), plan_first=True
        )
    comments = [
        _m936_cpp.comment(str(item["body"])) for item in interrupted.issue_comments
    ]
    last_coder = max(
        record.index for record in _extract_round_metadata_records(comments, flow="plan")
        if record.metadata.role == "coder"
    )
    durable = list(interrupted.issue_comments[: last_coder + 1])
    assert orchestrator_module._resumed_inherited_replan_force_full(comments[: last_coder + 1])
    # No digest is involved on this path: the audit comment is the durable key.
    assert _extract_round_metadata_records(
        comments, flow="plan"
    )[-1].metadata.plan_supersession_digest is None

    restarted = _m936_cpp._ChildPlanningRunner(
        issue_comments=durable,
        codex_outputs=[_m936_cpp.structured_plan_review(state="approved")],
        gemini_outputs=[gemini_approval],
    )
    assert orchestrator_module.run_issue_loop(
        restarted, issue_number=56, config=staged_config(), plan_first=True
    ) == 0
    assert _m936_cpp._agent_prompts(restarted, "claude") == []
    audits = [
        str(item["body"]) for item in restarted.issue_comments[len(durable):]
        if "Plan review scheduling audit" in str(item["body"])
    ]
    assert "Force-full: True (source: automatic)" in audits[0]
    assert "Selected reviewers: Codex, Gemini" in audits[0]
    # An ordinary revision round reconstructs no latch.
    assert not orchestrator_module._resumed_inherited_replan_force_full(
        [_m936_cpp.comment(str(item["body"])) for item in first.issue_comments]
    )


# --- #948: digest-on-overflow selection at the structured plan publication sites ---

import coding_review_agent_loop.comment_rendering as _m948_rendering  # noqa: E402
import coding_review_agent_loop.round_transport as _m948_transport  # noqa: E402
from coding_review_agent_loop.comment_rendering import render_public_agent_comment  # noqa: E402,F811
from coding_review_agent_loop.github import post_issue_comment as _m948_post_issue_comment  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    validate_human_requirements_acknowledgement as _m948_validate_ack,
    validate_structured_plan_revision as _m948_validate_revision,
    validate_structured_plan_state as _m948_validate_state,
)
from coding_review_agent_loop.round_state import PostedRoundMetadata as _M948Metadata  # noqa: E402

_M948_FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
_M948_NOTICE = _m948_rendering.COMPACT_PLAN_DIGEST_NOTICE
_M948_DIRECT = "I checked the relevant GitHub discussion directly before responding."
_M948_CANONICAL_ONLY_RECORDS = (
    "AGENT_DEFERRED_STAGES:",
    "AGENT_TYPED_PLAN_STAGES:",
    "AGENT_PLAN_EXPECTED_CLOSING_ISSUES:",
)


def _m948_noise(seed, chars=256):
    text = ""
    while len(text) < chars:
        text += hashlib.sha256(f"{seed}-{len(text)}".encode()).hexdigest()
    return text[:chars]


def _m948_stages(prefix, count=300):
    return [
        {"title": f"{prefix} {index}", "summary": _m948_noise(f"{prefix}-{index}")}
        for index in range(count)
    ]


def _m948_plan_config(tmp_path):
    return make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only")


def _m948_bodies(runner):
    return [str(item["body"]) for item in runner.issue_comments]


def _m948_digest_anchor(runner):
    anchors = [body for body in _m948_bodies(runner) if _M948_NOTICE in body]
    assert len(anchors) == 1
    return anchors[0]


def _m948_hydrated_metadata(anchor, bodies):
    match = list(_m948_transport.ROUND_RESUME_MARKER_RE.finditer(anchor))[-1]
    hydrated, missing = _m948_transport.hydrate_mapping(
        _m948_transport.decode_mapping(match.group("payload")), bodies
    )
    assert missing == set()
    return hydrated


@pytest.mark.parametrize(
    "record_class", ["deferred-stages", "typed-stages", "expected-closing"]
)
def test_m948_oversized_collection_posts_digest_and_resume_recovers_it_from_metadata(
    tmp_path, record_class
):
    payload = json.loads(structured_plan_state(summary="Collections plan.").split("\n", 1)[0])
    if record_class == "deferred-stages":
        payload["deferred_stages"] = _m948_stages("Deferred")
    elif record_class == "typed-stages":
        payload["external_dependencies"] = _m948_stages("External", 100)
        payload["deferred_work"] = _m948_stages("Later", 100)
        payload["plan_actions"] = _m948_stages("Action", 100)
    else:
        payload["additional_closing_issue_ids"] = list(range(100, 140))
        payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    plan = json.dumps(payload) + _M948_FOOTER
    parsed = _m948_validate_state(plan)
    # The full visible body alone is over budget, embedded records included.
    assert len(render_public_agent_comment(kind="plan_state", parsed=parsed, agent="claude")) > 60_000

    # No reviewer output: the run stops with the plan published and unreviewed.
    first = FakeRunner(claude_outputs=[plan], codex_outputs=[])
    with pytest.raises(Exception):
        run_issue_loop(first, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True)

    bodies = _m948_bodies(first)
    assert all(len(body) <= _m948_transport.MAX_GITHUB_BODY_CHARS for body in bodies)
    anchor = _m948_digest_anchor(first)
    assert not any(record in anchor for record in _M948_CANONICAL_ONLY_RECORDS)
    assert "omitted; complete list in the authenticated attachments" in anchor
    assert "<!-- AGENT_PLAN_STATE: blocking -->" in anchor

    metadata = _m948_hydrated_metadata(anchor, bodies)
    canonical_plan = metadata["canonical_plan"]
    assert canonical_plan == orchestrator_module.render_canonical_plan_state(parsed) or (
        parsed.execution_recommendation is None and canonical_plan == plan
    )
    assert metadata["subject"] == orchestrator_module._plan_subject(canonical_plan)
    assert _M948_NOTICE not in canonical_plan

    second = FakeRunner(
        issue_comments=first.issue_comments,
        claude_outputs=[],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(
        second, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0
    assert [cmd for cmd, _cwd in second.commands if cmd[:1] == ["claude"]] == []
    # Large prompts are delivered on stdin; the reviewer turn is the only agent
    # call of the resumed run.  The reviewer is prompted with the complete plan,
    # whose tail the digest omitted.
    codex_calls = [cmd for cmd, _cwd in second.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_calls) == 1
    review_prompt = second.last_input_text or codex_calls[0][-1]
    if record_class == "deferred-stages":
        assert _m948_noise("Deferred-299") in review_prompt
        assert _m948_noise("Deferred-299") not in anchor
    elif record_class == "typed-stages":
        assert _m948_noise("Action-99") in review_prompt
        assert _m948_noise("Action-99") not in anchor
    else:
        assert _m948_noise(299) in review_prompt
        assert _m948_noise(299) not in anchor
    assert len([body for body in _m948_bodies(second) if _M948_NOTICE in body]) == 1


def test_m948_resumed_collections_equal_the_full_form_publication():
    payload = json.loads(structured_plan_state(summary="Collections plan.").split("\n", 1)[0])
    payload["deferred_stages"] = _m948_stages("Deferred", 5)
    payload["plan_actions"] = _m948_stages("Action", 5)
    payload["additional_closing_issue_ids"] = [101, 102]
    parsed = _m948_validate_state(json.dumps(payload) + _M948_FOOTER)
    canonical_plan = orchestrator_module.render_canonical_plan_state(parsed)

    # Stage and closing collections are read from the canonical plan only; the
    # digest carries none of their records.
    assert orchestrator_module._extract_current_deferred_stages(canonical_plan) == parsed.deferred_stages
    assert orchestrator_module._extract_current_expected_closing_issue_ids(canonical_plan) == (101, 102)
    digest = render_public_agent_comment(
        kind="plan_state", parsed=parsed, agent="claude", compact=True
    )
    assert orchestrator_module._extract_current_deferred_stages(digest) == ()
    assert orchestrator_module._extract_current_expected_closing_issue_ids(digest) is None


def test_m948_fitting_plan_posts_the_full_form_byte_identically(tmp_path, monkeypatch):
    compact_calls = []
    real_render = orchestrator_module.render_public_agent_comment

    full_renders = {}

    def spy(**kwargs):
        if kwargs.get("compact"):
            compact_calls.append(kwargs["kind"])
        rendered = real_render(**kwargs)
        full_renders[kwargs["kind"]] = rendered
        return rendered

    monkeypatch.setattr(orchestrator_module, "render_public_agent_comment", spy)
    plan = structured_plan_state(summary="Small plan.")
    revision = structured_plan_revision(
        summary="Small revision.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ],
    )
    runner = FakeRunner(
        claude_outputs=[plan, revision],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0

    assert compact_calls == []
    bodies = _m948_bodies(runner)
    assert not any(_M948_NOTICE in body for body in bodies)
    posted_plan = next(body for body in bodies if body.startswith("## Plan"))
    posted_revision = next(body for body in bodies if body.startswith("## Revised plan"))
    # The posted bodies are exactly the default renderings plus round metadata.
    # Attaching round metadata replaces the blank line before the trailing
    # footer with the metadata line; undo exactly that.
    def without_metadata(body):
        return re.sub(r"\n<!-- AGENT_LOOP_META: [^\n]* -->\n", "\n\n", body, count=1)

    assert without_metadata(posted_plan) == full_renders["plan_state"]
    assert without_metadata(posted_revision) == full_renders["plan_revision"]


def test_m948_non_size_transport_failure_aborts_without_selecting_the_digest(
    tmp_path, monkeypatch
):
    compact_calls = []
    real_render = orchestrator_module.render_public_agent_comment

    def spy(**kwargs):
        if kwargs.get("compact"):
            compact_calls.append(kwargs["kind"])
        return real_render(**kwargs)

    def malformed(_body_text):
        raise AgentLoopError("Risk matrix marker is not a recoverable JSON object.")

    monkeypatch.setattr(orchestrator_module, "render_public_agent_comment", spy)
    monkeypatch.setattr(_m948_transport, "_prepare_risk_test_matrix_transport", malformed)
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    runner = FakeRunner(claude_outputs=[json.dumps(payload) + _M948_FOOTER], codex_outputs=[])

    with pytest.raises(AgentLoopError, match="not a recoverable JSON object"):
        run_issue_loop(runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True)

    assert compact_calls == []
    assert runner.issue_comments == []


def _m948_requirement():
    return HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )


def _m948_ack_block(requirement_ids, *, direct=False):
    lines = ["", HUMAN_REQUIREMENTS_ADDRESSED_MARKER, "### Human requirements"]
    if direct:
        lines.append(f"- {_M948_DIRECT}")
    lines.extend(f"- {item}: covered because " + "EVIDENCE " * 300 for item in requirement_ids)
    return "\n".join(lines) + "\n"


def _m948_signed_payload(kind, requirement_ids):
    payload = json.loads(structured_plan_state(summary="Signed plan.").split("\n", 1)[0])
    payload["kind"] = kind
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    payload["human_requirement_dispositions"] = [
        {"requirement_id": item, "disposition": "addressed", "evidence": "EVIDENCE " * 300}
        for item in requirement_ids
    ]
    if kind == "plan_revision":
        payload["prior_plan_item_dispositions"] = [
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ]
    return payload


def _m948_assert_posted_acknowledgement(anchor, requirement_ids, *, direct):
    # Re-parse the POSTED anchor body with the existing validator.
    _m948_validate_ack(
        anchor, surfaced_requirement_ids=requirement_ids, requires_direct_discussion_ack=direct
    )
    assert anchor.count(HUMAN_REQUIREMENTS_ADDRESSED_MARKER) == 1
    assert (_M948_DIRECT in anchor) is direct
    for item in requirement_ids:
        assert f"- **{item}** — `addressed`: " in anchor
    assert "EVIDENCE " * 100 not in anchor
    assert len(anchor) <= _m948_transport.MAX_GITHUB_BODY_CHARS


def test_m948_signed_requirements_fresh_and_revision_digests_revalidate_end_to_end(tmp_path):
    requirement_id = _m948_requirement().requirement_id
    ids = (requirement_id,)
    fresh = json.dumps(_m948_signed_payload("plan_state", ids)) + _m948_ack_block(ids) + _M948_FOOTER
    revision = (
        json.dumps({**_m948_signed_payload("plan_revision", ids), "summary": "Signed revision."})
        + _m948_ack_block(ids) + _M948_FOOTER
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[fresh, revision],
        codex_outputs=[
            _add_default_requirement_disposition(
                structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
                requirement_id=requirement_id,
            ),
            _add_default_requirement_disposition(
                structured_plan_review(
                    state="approved",
                    prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                    human_requirements_resolved=True,
                ),
                requirement_id=requirement_id,
            ),
        ],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0

    anchors = [body for body in _m948_bodies(runner) if _M948_NOTICE in body]
    assert [anchor.split("\n", 1)[0] for anchor in anchors] == ["## Plan", "## Revised plan"]
    for anchor in anchors:
        _m948_assert_posted_acknowledgement(anchor, ids, direct=False)


def _m948_assemble(kind, parsed, raw_text, *, config, ids, direct, metadata=None):
    full_comment = render_public_agent_comment(
        kind=kind, parsed=parsed, agent=config.coder, raw_text=raw_text, config=config
    )
    canonical_plan = (
        orchestrator_module.render_canonical_plan_state(parsed)
        if kind == "plan_state"
        else orchestrator_module.render_canonical_plan_revision(parsed, (), config)
    )
    return orchestrator_module._assemble_structured_plan_round_body(
        config=config,
        issue_number=56,
        kind=kind,
        parsed_plan=parsed,
        full_comment=full_comment,
        metadata=metadata or _M948Metadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject=orchestrator_module._plan_subject(canonical_plan),
            canonical_plan=canonical_plan, raw_structured_coder_response=raw_text,
        ),
        raw_text=raw_text,
        prior_items=(),
        model_used=None,
        surfaced_requirement_ids=ids,
        requires_direct_discussion_ack=direct,
    )


@pytest.mark.parametrize("form", ["fresh-plan-state", "legacy-full-state", "semantic-patch-v1"])
@pytest.mark.parametrize("direct", [False, True], ids=["surfaced-ids", "direct-discussion"])
def test_m948_signed_requirements_digest_posted_body_revalidates(tmp_path, form, direct):
    config = _m948_plan_config(tmp_path)
    # Direct-discussion mode surfaces no IDs: the prompt omitted them all.
    ids = () if direct else tuple(f"hr-{index:064x}" for index in range(1, 7))
    kind = "plan_state" if form == "fresh-plan-state" else "plan_revision"
    payload = _m948_signed_payload(kind, ids)
    payload["prior_plan_item_dispositions"] = []
    if kind == "plan_state":
        payload.pop("prior_plan_item_dispositions")
    structured = json.dumps(payload) + _M948_FOOTER
    parsed = (_m948_validate_state if kind == "plan_state" else _m948_validate_revision)(structured)
    block = _m948_ack_block(ids, direct=direct)
    if form == "semantic-patch-v1":
        # The semantic site renders the assembled plan but passes the raw patch
        # response, which is where the acknowledgement block lives.
        raw_text = json.dumps({"kind": "plan_revision_patch", "operations": []}) + block + _M948_FOOTER
    else:
        raw_text = json.dumps(payload) + block + _M948_FOOTER
    _m948_validate_ack(raw_text, surfaced_requirement_ids=ids, requires_direct_discussion_ack=direct)

    body = _m948_assemble(kind, parsed, raw_text, config=config, ids=ids, direct=direct)
    runner = FakeRunner()
    _m948_post_issue_comment(runner, config=config, issue_number=56, body=body)

    anchor = _m948_digest_anchor(runner)
    _m948_assert_posted_acknowledgement(anchor, ids, direct=direct)
    metadata = _m948_hydrated_metadata(anchor, _m948_bodies(runner))
    assert metadata["raw_structured_coder_response"] == raw_text


def test_m948_fresh_plan_without_acknowledgement_block_posts_every_disposition_id(tmp_path):
    config = _m948_plan_config(tmp_path)
    ids = tuple(f"hr-{index:064x}" for index in range(1, 4))
    raw_text = json.dumps(_m948_signed_payload("plan_state", ids)) + _M948_FOOTER
    parsed = _m948_validate_state(raw_text)

    body = _m948_assemble("plan_state", parsed, raw_text, config=config, ids=ids, direct=False)

    assert _M948_NOTICE in body
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER not in body
    assert all(f"- **{item}** — `addressed`: " in body for item in ids)


def test_m948_digest_failing_revalidation_is_never_posted(tmp_path, monkeypatch):
    config = _m948_plan_config(tmp_path)
    ids = tuple(f"hr-{index:064x}" for index in range(1, 4))
    raw_text = (
        json.dumps(_m948_signed_payload("plan_state", ids)) + _m948_ack_block(ids) + _M948_FOOTER
    )
    parsed = _m948_validate_state(raw_text)
    monkeypatch.setattr(
        _m948_rendering, "_compact_human_requirements_block",
        lambda block: block.replace(ids[1], "hr-" + "f" * 64),
    )
    with pytest.raises(AgentLoopError):
        _m948_assemble("plan_state", parsed, raw_text, config=config, ids=ids, direct=False)


def test_m948_overflow_the_digest_cannot_fix_fails_closed_and_posts_nothing(tmp_path):
    config = _m948_plan_config(tmp_path)
    raw_text = json.dumps(_m948_signed_payload("plan_state", ())) + _M948_FOOTER
    parsed = _m948_validate_state(raw_text)
    canonical_plan = orchestrator_module.render_canonical_plan_state(parsed)
    metadata = _M948Metadata(
        flow="plan", role="coder", agent="Claude", round_number=1,
        subject=orchestrator_module._plan_subject(canonical_plan),
        canonical_plan=canonical_plan,
        # Not in the transport spill set, so no presentation change can help.
        compact_prior_summaries=(_m948_noise("unspillable", 120_000),),
    )
    body = _m948_assemble(
        "plan_state", parsed, raw_text, config=config, ids=(), direct=False, metadata=metadata
    )
    runner = FakeRunner()
    with pytest.raises(_m948_transport.RoundCommentOverflowError) as excinfo:
        _m948_post_issue_comment(runner, config=config, issue_number=56, body=body)
    assert "compact_prior_summaries=" in str(excinfo.value)
    assert _m948_noise("unspillable", 64) not in str(excinfo.value)
    assert runner.issue_comments == []


# --- #959: issue-to-PR coder metadata persists the full-matrix anchor -------

from coding_review_agent_loop.protocol import (  # noqa: E402
    parse_risk_test_matrix_evidence as _parse_evidence_959,
)


def _issue_evidence_959():
    return _parse_evidence_959({
        "matrix_identity": "c" * 64,
        "rows": [{
            "row_id": f"row-{index}",
            "status": "missing",
            "test_identifiers": [],
            "test_locations": [],
            "workflow_path_claim": f"Direct flow path {index}.",
            "outcome_assertions": [],
            "forbidden_effect_assertions": [],
            "evidence_citations": [],
            "caveats": [],
        } for index in range(2)],
    })


@pytest.mark.parametrize("with_evidence", [True, False], ids=["evidence", "no-evidence"])
def test_959_direct_issue_implementation_metadata_anchors_to_own_round(
    tmp_path, monkeypatch, with_evidence
):
    runner = FakeRunner(claude_outputs=[structured_issue_implementation(pr_number=77)])
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    real_derive = orchestrator_module._derive_authenticated_risk_evidence_for_coder

    def derive(result, **kwargs):
        derived, extra = real_derive(result, **kwargs)
        if with_evidence:
            derived = replace(derived, risk_test_matrix_evidence=_issue_evidence_959())
        return derived, extra

    monkeypatch.setattr(orchestrator_module, "_derive_authenticated_risk_evidence_for_coder", derive)
    monkeypatch.setattr(orchestrator_module, "run_pr_loop", lambda *_a, **_k: 0)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _metadata_from_public_comment(item["body"]).role == "coder"
    ]
    assert len(coder_comments) == 1
    metadata = _metadata_from_public_comment(coder_comments[0])
    visible = coder_comments[0].split("AGENT_LOOP_META", 1)[0]
    if with_evidence:
        assert metadata.risk_test_matrix_evidence is not None
        assert metadata.risk_test_matrix_evidence_full_round == metadata.round_number
        assert metadata.risk_test_matrix_evidence_full_round_status == "valid"
        assert "<summary>Full matrix evidence (2 rows)</summary>" in visible
        assert "**row-0**" in visible and "**row-1**" in visible
    else:
        assert metadata.risk_test_matrix_evidence is None
        assert metadata.risk_test_matrix_evidence_full_round is None
        assert metadata.risk_test_matrix_evidence_full_round_status == "absent"
        assert "matrix evidence" not in visible
        assert "<details>" not in visible


def test_959_structured_no_pr_terminal_comment_omits_matrix_evidence(tmp_path):
    raw = structured_issue_implementation(pr_number=None, summary="Blocked before a PR.")
    payload, end = json.JSONDecoder().raw_decode(raw)
    payload["risk_test_matrix_claims"] = [{
        "row_id": "row-0",
        "execution_refs": ["turn:observation-1"],
        "test_identifiers": ["tests/test_x.py::test_y"],
        "test_locations": ["tests/test_x.py"],
        "workflow_path_claim": "Semantic claim only.",
        "outcome_assertions": ["outcome"],
        "forbidden_effect_assertions": ["forbidden"],
        "caveats": [],
    }]
    parsed = validate_structured_issue_implementation(json.dumps(payload) + raw[end:])
    assert parsed is not None and parsed.pr_number is None
    assert parsed.risk_test_matrix_evidence is None
    runner = FakeRunner()
    config = make_config(tmp_path, coder="claude", reviewer="codex")

    orchestrator_module._post_structured_issue_implementation_terminal_comment(
        runner, config=config, issue_number=56, parsed=parsed, model_used=None
    )

    assert runner.comments
    body = runner.comments[-1]
    assert "Blocked before a PR." in body
    assert "matrix evidence" not in body
    assert "<details>" not in body


# --- Signed reviewer-board amendment (#943) -------------------------------

from coding_review_agent_loop.board_amendment import (  # noqa: E402
    format_reviewer_board_amendment_comment as _m943_amendment_comment,
)

_M943_BOARD = ("Codex", "Gemini", "Antigravity")


def _m943_runner():
    """Primary gate, then an Antigravity blocker whose remediation it never rechecks."""
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]
    return _FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
        ],
        gemini_outputs=[
            structured_plan_review(state="approved", reviewer="Google Gemini"),
            structured_plan_review(state="approved", reviewer="Google Gemini"),
        ],
        antigravity_outputs=[
            structured_plan_review(
                state="blocking",
                reviewer="Google Antigravity",
                summary="One plan step omits the rollout owner.",
                blocking_plan_issues=["Name the rollout owner in the plan steps."],
            ),
        ],
    )


def _m943_reviewer_calls(runner):
    return [cmd[0] for cmd, _cwd in runner.commands if cmd[0] in {"codex", "gemini", "agy"}]


def _m943_decoded(body):
    match = re.search(r"<!-- AGENT_LOOP_META: (?P<payload>\S+) -->", body)
    return _decode_round_metadata(match.group("payload")) if match else None


def _m943_interrupted_history(tmp_path):
    """Round 3 is partial: Codex's remediation review is posted, Antigravity never ran."""
    runner = _m943_runner()
    config = _staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8)
    real_post = orchestrator_module.post_issue_comment

    def stop_after_codex_round_3(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if (
            metadata is not None
            and metadata.role == "reviewer"
            and metadata.round_number == 3
            and metadata.agent == "Codex"
        ):
            raise KeyboardInterrupt
        return result

    with patch.object(orchestrator_module, "post_issue_comment", side_effect=stop_after_codex_round_3):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner


def _m943_strip_ownership(runner, item_id="item-1"):
    """Rewrite persisted records so ``item_id`` has genuinely pre-change ownership.

    Every record that carries the item gets ``resolution_owners``,
    ``owner_states``, ``owner_evidence``, and ``owner_dispositions`` emptied,
    so the finding falls back to its author exactly as a legacy ledger does.
    """
    from coding_review_agent_loop.round_state import _encode_round_metadata

    def legacy(items):
        return tuple(
            replace(
                item,
                resolution_owners=(),
                owner_states=(),
                owner_evidence=(),
                owner_dispositions=(),
            )
            if item.item_id == item_id
            else item
            for item in items
        )

    rewritten = 0
    for comment in runner.issue_comments:
        metadata = _m943_decoded(comment["body"])
        if metadata is None or not any(
            item.item_id == item_id for item in (*metadata.prior_items, *metadata.new_items)
        ):
            continue
        stripped = replace(
            metadata,
            prior_items=legacy(metadata.prior_items),
            new_items=legacy(metadata.new_items),
        )
        comment["body"] = re.sub(
            r"<!-- AGENT_LOOP_META: \S+ -->",
            lambda _match: f"<!-- AGENT_LOOP_META: {_encode_round_metadata(stripped)} -->",
            comment["body"],
        )
        rewritten += 1
    assert rewritten
    return runner


def _m943_post_amendment(runner, *, effective_from_round, removed=("Antigravity",), **overrides):
    values = dict(
        flow="plan",
        issue=56,
        pr_number=None,
        original_required_reviewers=_M943_BOARD,
        policy="primary-then-panel",
        primary_reviewer="Codex",
        removed_reviewers=removed,
        effective_from_round=effective_from_round,
        rationale="Antigravity weekly quota exhausted.",
    )
    values.update(overrides)
    runner.issue_comments.append(
        {
            "author": {"login": "operator"},
            "createdAt": f"2026-05-23T00:00:{len(runner.issue_comments):02d}Z",
            "body": _m943_amendment_comment(**values),
        }
    )


def _m943_reduced_config(tmp_path):
    return _staged_plan_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=8)


def test_board_amendment_partial_round_rescues_legacy_run_and_reassigns_finding(tmp_path):
    """Rows legacy-no-amendment, partial-round-activation, legacy-retroactive-amend,
    sole-owner-reassign, and activation-mismatch on one real history."""
    # The Antigravity finding is genuinely pre-change: no persisted record
    # carries explicit ownership, so it falls back to its removed author.
    runner = _m943_strip_ownership(_m943_interrupted_history(tmp_path))
    records = _plan_round_records(runner)
    old_round_3 = [
        record for record in records
        if record.round_number == 3 and record.phase == "scheduler-prelaunch"
    ]
    assert len(old_round_3) == 1
    assert tuple(old_round_3[0].scheduler_contract["required_reviewers"]) == _M943_BOARD
    # Every pre-change record lacks the new field.
    assert all(record.reviewer_board_amendment_digest is None for record in records)
    (finding,) = old_round_3[0].prior_items
    assert finding.item_id == "item-1" and finding.reviewer == "Antigravity"
    assert finding.resolution_owners == () and finding.owner_states == ()
    calls_before = _m943_reviewer_calls(runner)
    comments_before = len(runner.issue_comments)

    # legacy-no-amendment: the drift error prints a filled amendment template.
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume") as excinfo:
        run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    message = str(excinfo.value)
    assert '"kind": "reviewer-board-amendment"' in message
    assert '"effective_from_round": 3' in message
    assert '"removed_reviewers": [\n    "Antigravity"\n  ]' in message
    assert _m943_reviewer_calls(runner) == calls_before
    assert len(runner.issue_comments) == comments_before

    # activation-mismatch: an effective round other than the re-entered round 3.
    for wrong in (2, 4):
        mismatched = _m943_interrupted_history(tmp_path)
        _m943_post_amendment(mismatched, effective_from_round=wrong)
        before = (_m943_reviewer_calls(mismatched), len(mismatched.issue_comments))
        with pytest.raises(AgentLoopError, match="re-enters round 3") as excinfo:
            run_issue_loop(
                mismatched, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True
            )
        assert '"effective_from_round": 3' in str(excinfo.value)
        assert (_m943_reviewer_calls(mismatched), len(mismatched.issue_comments)) == before

    # partial-round-activation: the amendment starts inside round 3.  A first
    # resume is interrupted right after the fresh amended checkpoint.
    _m943_post_amendment(runner, effective_from_round=3)
    real_post = orchestrator_module.post_issue_comment

    def stop_after_fresh_checkpoint(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if metadata is not None and metadata.reviewer_board_amendment_digest is not None:
            raise KeyboardInterrupt
        return result

    with patch.object(orchestrator_module, "post_issue_comment", side_effect=stop_after_fresh_checkpoint):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    assert _m943_reviewer_calls(runner) == calls_before
    audits = [c["body"] for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]
    assert len(audits) == 1
    assert "Activation round: 3" in audits[0]
    assert "item-1 (Antigravity -> Codex)" in audits[0]
    fresh = [
        record for record in _plan_round_records(runner)
        if record.reviewer_board_amendment_digest is not None
    ]
    assert len(fresh) == 1
    assert fresh[0].round_number == 3 and fresh[0].phase == "scheduler-prelaunch"
    assert tuple(fresh[0].scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
    # Round 3's persisted ledger is carried verbatim, never rewritten in-round.
    assert fresh[0].prior_items == old_round_3[0].prior_items
    (fresh_body,) = [
        c["body"] for c in runner.issue_comments
        if (metadata := _m943_decoded(c["body"])) is not None
        and metadata.reviewer_board_amendment_digest is not None
    ]
    assert "Reviewer board amended from round 3" in fresh_body
    assert "required board now Codex, Gemini" in fresh_body
    assert "Required board now: Codex, Gemini" in audits[0]

    # Restart after the fresh checkpoint: Codex's round-3 review is reused,
    # the reassignment is re-derived, and only the Gemini sweep runs.
    assert run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True) == 0
    assert _m943_reviewer_calls(runner) == [*calls_before, "gemini"]
    assert "agy" not in _m943_reviewer_calls(runner)[len(calls_before):]
    assert len([c for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]) == 1
    records = _plan_round_records(runner)
    digest = fresh[0].reviewer_board_amendment_digest
    post_amendment_checkpoints = [
        record for record in records
        if record.phase == "scheduler-prelaunch" and record.round_number >= 3
        and record.reviewer_board_amendment_digest is not None
    ]
    assert post_amendment_checkpoints
    assert all(
        record.reviewer_board_amendment_digest == digest
        and tuple(record.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for record in post_amendment_checkpoints
    )
    # Contract-neutral records never carry the digest.
    assert all(
        record.reviewer_board_amendment_digest is None
        for record in records
        if record.scheduler_contract is None
    )
    # The Antigravity finding was cleared only by Codex's disposition.
    item_1_dispositions = [
        disposition
        for record in records
        if record.round_number == 3
        for disposition in record.dispositions
        if disposition.item_id == "item-1"
    ]
    assert item_1_dispositions
    assert {disposition.reviewer for disposition in item_1_dispositions} == {"Codex"}
    later = [record for record in records if record.round_number >= 4]
    assert later and all(
        item.item_id != "item-1" for record in later for item in record.prior_items
    )
    # Earlier rounds, including Antigravity's banked round-2 review, are history.
    assert any(
        record.agent == "Antigravity" and record.round_number == 2 and record.role == "reviewer"
        for record in records
    )

    # re-add-reviewer: widening the board again after the amendment is drift.
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(
            runner,
            issue_number=56,
            config=_staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8),
            plan_first=True,
        )


def test_board_amendment_rejects_an_old_board_checkpoint_after_the_record(tmp_path):
    """Row post-amend-old-board: a stale-board round after the record is drift."""
    runner = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(runner, effective_from_round=3)
    old_checkpoint = next(
        comment for comment in runner.issue_comments
        if (metadata := _m943_decoded(comment["body"])) is not None
        and metadata.phase == "scheduler-prelaunch"
        and metadata.round_number == 3
    )
    stale = replace(_m943_decoded(old_checkpoint["body"]), scheduler_reasons=("stale board",))
    runner.issue_comments.append(
        {
            "author": {"login": "bot"},
            "createdAt": "2026-05-23T00:59:00Z",
            "body": _attach_round_metadata("Plan review scheduling audit.\n\n-- Orchestrator", stale),
        }
    )
    calls_before = _m943_reviewer_calls(runner)
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    assert _m943_reviewer_calls(runner) == calls_before


def test_board_amendment_invalid_records_fail_closed_or_are_ignored(tmp_path):
    """Row record-invalid at the orchestrator: no partial amendment is applied."""
    unsigned = _m943_interrupted_history(tmp_path)
    unsigned.issue_comments.append(
        {
            "author": {"login": "operator"},
            "createdAt": "2026-05-23T00:59:00Z",
            "body": _m943_amendment_comment(
                flow="plan", issue=56, pr_number=None, original_required_reviewers=_M943_BOARD,
                policy="primary-then-panel", primary_reviewer="Codex",
                removed_reviewers=("Antigravity",), effective_from_round=3, rationale="quota",
            ).replace("-- Human Reviewer", "-- Anthropic Claude"),
        }
    )
    with pytest.raises(AgentLoopError, match="scheduler contract changed during resume"):
        run_issue_loop(unsigned, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)

    removes_primary = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(removes_primary, effective_from_round=3, removed=("Codex",))
    with pytest.raises(AgentLoopError, match="Human decision required"):
        run_issue_loop(
            removes_primary,
            issue_number=56,
            config=_staged_plan_config(
                tmp_path, reviewer=("gemini", "antigravity"), primary_plan_reviewer="gemini"
            ),
            plan_first=True,
        )

    other_issue = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(other_issue, effective_from_round=3, issue=57)
    with pytest.raises(AgentLoopError, match="names issue #57"):
        run_issue_loop(other_issue, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)


@pytest.mark.parametrize("legacy_owner", [False, True], ids=["explicit-owner", "legacy-implicit-owner"])
def test_board_amendment_reenters_a_reconciled_round_under_the_amended_board(tmp_path, legacy_owner):
    """Row reconciled-round-reentry: round 2 was reconciled under the old board.

    The legacy variant strips the Antigravity finding's persisted ownership,
    so the next round's ledger proves implicit ownership is normalized and
    reassigned rather than falling back to the removed author.
    """
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]

    def reconciled_history():
        runner = _FakeRunner(
            claude_outputs=[fresh],  # the round-3 planner turn has no output yet
            codex_outputs=[
                structured_plan_review(state="approved"),
                structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
            ],
            gemini_outputs=[
                structured_plan_review(state="approved", reviewer="Google Gemini"),
                structured_plan_review(state="approved", reviewer="Google Gemini"),
            ],
            antigravity_outputs=[
                structured_plan_review(
                    state="blocking",
                    reviewer="Google Antigravity",
                    summary="One plan step omits the rollout owner.",
                    blocking_plan_issues=["Name the rollout owner in the plan steps."],
                ),
            ],
        )
        config = _staged_plan_config(
            tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8, review_parallel=True
        )
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        records = _plan_round_records(runner)
        assert any(
            record.round_number == 2 and record.phase == "reconciliation" for record in records
        )
        assert not any(record.round_number == 3 for record in records)
        return runner

    reduced = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=8, review_parallel=True
    )

    # An amendment effective from round 3 fails closed naming N=2.
    early = reconciled_history()
    _m943_post_amendment(early, effective_from_round=3)
    before = (_m943_reviewer_calls(early), len(early.issue_comments))
    with pytest.raises(AgentLoopError, match="re-enters round 2"):
        run_issue_loop(early, issue_number=56, config=reduced, plan_first=True)
    assert (_m943_reviewer_calls(early), len(early.issue_comments)) == before

    runner = reconciled_history()
    if legacy_owner:
        _m943_strip_ownership(runner)
        (seeded,) = [
            item
            for record in _plan_round_records(runner)
            if record.round_number == 2 and record.phase == "reconciliation"
            for item in record.new_items
            if item.item_id == "item-1"
        ]
        assert seeded.resolution_owners == () and seeded.owner_states == ()
    _m943_post_amendment(runner, effective_from_round=2)
    runner.claude_outputs.append(patch_text)
    calls_before = _m943_reviewer_calls(runner)
    assert run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True) == 0

    records = _plan_round_records(runner)
    amended = [record for record in records if record.reviewer_board_amendment_digest is not None]
    # Round 2 is re-entered with a fresh digest-bound checkpoint, then the
    # blocking Antigravity finding moves the loop to coder round 3.
    assert amended[0].round_number == 2 and amended[0].phase == "scheduler-prelaunch"
    assert all(
        tuple(record.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for record in amended
    )
    assert any(record.round_number == 3 and record.role == "coder" for record in records)
    round_3 = [
        record for record in records
        if record.round_number == 3 and record.phase == "scheduler-prelaunch"
    ]
    assert round_3 and round_3[0].reviewer_board_amendment_digest == amended[0].reviewer_board_amendment_digest
    # The carried finding now has explicit ownership that excludes Antigravity.
    (carried,) = [item for item in round_3[0].prior_items if item.item_id == "item-1"]
    assert carried.reviewer == "Antigravity"
    assert carried.status == "blocking"
    assert carried.resolution_owners == ("Codex",)
    assert carried.owner_states == (("Codex", "pending"),)
    audits = [c["body"] for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]
    assert len(audits) == 1 and "item-1 (Antigravity -> Codex)" in audits[0]
    assert "agy" not in _m943_reviewer_calls(runner)[len(calls_before):]
