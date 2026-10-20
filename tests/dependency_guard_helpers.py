"""Shared fixtures for the decomposition dependency guard (#1331).

A staged parent #55 with three children: stage ``s1`` is child #61, ``s2`` is
#62 and ``s3`` is #63.  The scenarios drive direct child entry through
``cli.main`` with a fake runner and record every live read the read-only
progress resolution performs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.cli as cli_module
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.phase_progress as phase_progress
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    DecompositionMetadata,
    PhaseImplementationHandoffMetadata,
    PlanPhase,
)
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.integration_close import IntegrationMergeVerification
from coding_review_agent_loop.protocol import (
    EXECUTION_DISPOSITION_DIRECT,
    EXECUTION_DISPOSITION_PLANNING,
)

from agent_loop_helpers import FakeRunner, make_config, patch_attr_in

PARENT = 55
CHILDREN = {"s1": 61, "s2": 62, "s3": 63}
STAGES = ("s1", "s2", "s3")


def plan_phase(
    stage_id: str,
    *,
    disposition: str | None = EXECUTION_DISPOSITION_DIRECT,
    automation: str = "agent-pr",
    depends_on: tuple[str, ...] = (),
) -> PlanPhase:
    return PlanPhase(
        title=f"Stage {stage_id}", scope="Implement the stage.", non_goals="None.",
        dependency_notes="None.", rollout_risk="low", validation="Run tests.",
        parent_context="Approved slice.", automation=automation, stage_id=stage_id,
        position=STAGES.index(stage_id) + 1, deliverables=("Stage",),
        non_goals_items=("Unrelated work",), acceptance_criteria=("Tests pass",),
        depends_on_stage_ids=depends_on, compatibility_constraints=("Preserve API",),
        covered_scope_item_ids=("scope-1",), execution_disposition=disposition,
        disposition_rationale="Reviewed route." if disposition else None,
    )


def summary_for(phases, *, mode: str = "implement-by-phase") -> DecompositionMetadata:
    return DecompositionMetadata(
        parent_issue=PARENT,
        plan_hash="plan-hash",
        mode=mode,
        phase_count=len(phases),
        phase_titles=tuple(phase.title for phase in phases),
        automation=tuple(phase.automation for phase in phases),
        children=tuple(
            (phase.title, f"https://github.com/OWNER/REPO/issues/{CHILDREN[phase.stage_id]}",
             CHILDREN[phase.stage_id])
            for phase in phases
        ),
        topology_source="approved-plan-v1",
        strategy="staged",
        execution_strategy_contract_version=1,
        recommendation_digest="digest",
        plan_subject="plan-subject",
        stage_ids=tuple(phase.stage_id for phase in phases),
    )


def handoff_for(stage_id: str, *, disposition: str = EXECUTION_DISPOSITION_DIRECT):
    return PhaseImplementationHandoffMetadata(
        parent_issue=PARENT, plan_hash="plan-hash", mode="implement-by-phase",
        phase_index=STAGES.index(stage_id) + 1, phase_title=f"Stage {stage_id}",
        automation="agent-pr", child_issue_number=CHILDREN[stage_id],
        child_issue_url=None, strategy="staged", topology_source="approved-plan-v1",
        execution_strategy_contract_version=1, recommendation_digest="digest",
        stage_id=stage_id, plan_subject="plan-subject", execution_disposition=disposition,
    )


def issue_context(number: int, comments=()) -> IssueContext:
    return IssueContext(
        number=number, repo="OWNER/REPO", title=f"Issue {number}", body="",
        url=f"https://github.com/OWNER/REPO/issues/{number}", comments=tuple(comments),
    )


@dataclass
class GuardScenario:
    """Live state of the earlier stages, and a log of what was read."""

    phases: tuple
    stage_id: str
    summary_mode: str = "implement-by-phase"
    handoffs: tuple = ()
    child_states: dict = field(default_factory=dict)
    canonical_prs: dict = field(default_factory=dict)  # child -> (pr, state)
    verifications: dict = field(default_factory=dict)  # child -> verification
    reads: list = field(default_factory=list)
    events: list = field(default_factory=list)
    summary: DecompositionMetadata | None = None

    @property
    def phase_index(self) -> int:
        return STAGES.index(self.stage_id) + 1

    @property
    def child(self) -> int:
        return CHILDREN[self.stage_id]

    def fresh(self, *, handoff=None):
        phase = self.phases[self.phase_index - 1]
        route = execution_policy.resolve_child_execution_route(
            phase, topology_source="approved-plan-v1", recorded_handoff=handoff
        )
        return SimpleNamespace(
            parent_issue=PARENT, plan_hash="plan-hash", plan_subject="plan-subject",
            approved_plan="Approved parent plan",
            parent_plan_context=SimpleNamespace(
                matrix_available=False, risk_test_matrix_payload=None
            ),
            recommendation=SimpleNamespace(
                strategy="staged", identity=lambda: {"recommendation_sha256": "digest"},
                child_stages=self.phases,
            ),
            decomposition=SimpleNamespace(phases=self.phases),
            created=CreatedPhaseIssue(
                phase=phase, issue_url=issue_context(self.child).url, issue_number=self.child
            ),
            phase_index=self.phase_index, stage_id=self.stage_id, handoff=handoff,
            overrides=(), route=route,
            summary=self.summary or summary_for(self.phases, mode=self.summary_mode),
        )


def install(monkeypatch, scenario: GuardScenario, *, handoff=None):
    """Patch the child entry seams; returns the fake runner."""
    fresh = scenario.fresh(handoff=handoff)
    child = replace(
        issue_context(scenario.child),
        body=f"Child phase issue for parent #{PARENT}.\n\nStage {scenario.stage_id}.",
    )
    parent = issue_context(PARENT)

    def get_context(_runner, *, config, issue_number):
        return child if issue_number == scenario.child else (
            parent if issue_number == PARENT else issue_context(issue_number)
        )

    patch_attr_in(
        monkeypatch,
        (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop,
         plan_first_loop, issue_loop),
        "get_issue_context",
        get_context,
    )
    patch_attr_in(
        monkeypatch, (execution_policy, child_plan_binding, issue_loop),
        "_resolve_fresh_child_provenance", lambda **_: fresh,
    )

    def progress_state(_runner, *, config, issue_number):
        scenario.reads.append(("state", issue_number))
        return scenario.child_states.get(issue_number, "OPEN")

    def progress_context(_runner, *, config, issue_number):
        scenario.reads.append(("context", issue_number))
        return issue_context(issue_number)

    def authenticate(_runner, *, config, issue_number, issue_context):
        scenario.reads.append(("canonical-pr", issue_number))
        found = scenario.canonical_prs.get(issue_number)
        if found is None:
            return None
        return SimpleNamespace(pr_number=found[0], state=found[1])

    def verify(_runner, *, config, issue_context, pr_number, expected_base=None):
        scenario.reads.append(("verify", issue_context.number))
        return scenario.verifications[issue_context.number]

    def no_close(*_args, **_kwargs):
        raise AssertionError("the read-only dependency check must never close a child")

    monkeypatch.setattr(phase_progress, "get_issue_state", progress_state)
    monkeypatch.setattr(phase_progress, "get_issue_context", progress_context)
    monkeypatch.setattr(phase_progress, "authenticate_canonical_issue_pr", authenticate)
    monkeypatch.setattr(phase_progress, "verify_integration_merge", verify)
    monkeypatch.setattr(phase_progress, "close_child_after_integration_merge", no_close)
    monkeypatch.setattr(
        phase_progress, "find_phase_implementation_handoffs_for_parent",
        lambda *_a, **_k: tuple(scenario.handoffs),
    )

    def record(name, value=0):
        def _call(*_args, **_kwargs):
            scenario.events.append(name)
            return value
        return _call

    patch_attr_in(
        monkeypatch, (discuss_loop, pr_loop, issue_loop), "prepare_agent_memory",
        record("memory", None),
    )
    patch_attr_in(
        monkeypatch, (plan_first_loop, issue_loop), "_dispatch_decomposition_child",
        record("dispatch", 23),
    )
    patch_attr_in(
        monkeypatch, (execution_policy, plan_first_loop, issue_loop),
        "_post_child_planning_handoff", record("planning-handoff", None),
    )
    patch_attr_in(
        monkeypatch, (plan_first_loop, issue_loop), "_run_plan_first_loop", record("planner", 19)
    )
    patch_attr_in(
        monkeypatch, (execution_policy, plan_first_loop, issue_loop),
        "resolve_canonical_pr_for_issue", lambda *_a, **_k: None,
    )
    return FakeRunner()


def run_cli(monkeypatch, tmp_path, runner, scenario: GuardScenario, *, plan_first=False):
    """Run ``agent-loop issue <child>`` through ``cli.main`` with the fake runner."""
    config = make_config(tmp_path)
    monkeypatch.setattr(cli_module, "config_from_args", lambda *a, **k: config)
    monkeypatch.setattr(cli_module, "establish_sandboxed_run", lambda *a, **k: None)
    monkeypatch.setattr(cli_module, "resolve_base_branch", lambda c, r: c)
    monkeypatch.setattr(cli_module, "ensure_agent_workdirs", lambda c, r: None)
    monkeypatch.setattr(cli_module, "Runner", lambda **_kw: runner)
    argv = ["issue", str(scenario.child), "--repo", "OWNER/REPO"]
    if plan_first:
        argv += ["--plan-first", "--plan-execution-mode", "auto"]
    return cli_module.main(argv)


def comment_writes(runner) -> list:
    """Every recorded command that would write a GitHub comment."""
    writes = []
    for cmd, _cwd in runner.commands:
        text = " ".join(str(part) for part in cmd)
        if ("issue comment" in text or "pr comment" in text or "issue close" in text
                or ("/comments" in text and "POST" in text)):
            writes.append(text)
    return writes


__all__ = [
    "CHILDREN",
    "EXECUTION_DISPOSITION_DIRECT",
    "EXECUTION_DISPOSITION_PLANNING",
    "GuardScenario",
    "IntegrationMergeVerification",
    "PARENT",
    "comment_writes",
    "handoff_for",
    "install",
    "issue_context",
    "plan_phase",
    "run_cli",
    "summary_for",
]
