import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.config import DEFAULT_FLAT_CHILD_LIMIT
from coding_review_agent_loop.decomposition import (
    CreatedPhaseIssue,
    PlanDecomposition,
    PlanPhase,
    RecordedPhase,
    RetainedParentScope,
    TopologyCheckpoint,
    approved_plan_hash,
    format_phase_issue_body,
    find_existing_phase_implementation_handoff,
    find_existing_topology_checkpoint,
    format_decomposition_parent_summary,
    format_phase_implementation_handoff_comment,
    format_topology_checkpoint,
    phase_identity,
    parse_plan_decomposition,
    create_decomposition_child_issues,
)
from coding_review_agent_loop.github import IssueComment
from coding_review_agent_loop.child_topology import NeedsHumanDecision
from coding_review_agent_loop.orchestrator import PostedRoundMetadata, _attach_round_metadata, _plan_subject
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    plan_decomposition_json,
    structured_plan_review,
    structured_plan_state,
)


def test_parse_plan_decomposition_accepts_agent_and_human_phases():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["Internal schema utilities"],
            },
        )
    )

    assert [phase.title for phase in parsed.phases] == [
        "Internal schema utilities",
        "Manual rollout checkpoint",
    ]
    assert parsed.phases[1].automation == "human-action"
    assert parsed.phases[1].depends_on == ("Internal schema utilities",)

def test_parse_plan_decomposition_accepts_normalized_earlier_phase_dependency():
    parsed = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Internal schema utilities",
                "scope": "Add helpers.",
                "non_goals": "No live switch.",
                "dependency_notes": "First phase.",
                "rollout_risk": "low - internal only.",
                "validation": "Run python -m pytest.",
                "parent_context": "Approved plan slice and invariant details.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Manual rollout checkpoint",
                "scope": "Human validates the deployed behavior.",
                "non_goals": "No code changes.",
                "dependency_notes": "After Internal schema utilities.",
                "rollout_risk": "medium - live checkpoint.",
                "validation": "Human remark and closure required.",
                "parent_context": "Approved plan slice for the manual checkpoint.",
                "automation": "human-action",
                "depends_on": ["  internal   SCHEMA utilities  "],
            },
        )
    )

    assert parsed.phases[1].depends_on == ("internal   SCHEMA utilities",)

def test_parse_plan_decomposition_rejects_self_dependency():
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Internal schema utilities"],
    }

    with pytest.raises(AgentLoopError, match="cannot depend on itself"):
        parse_plan_decomposition(plan_decomposition_json(phase))

def test_parse_plan_decomposition_rejects_forward_dependency():
    first_phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": ["Manual rollout checkpoint"],
    }
    second_phase = {
        "title": "Manual rollout checkpoint",
        "scope": "Human validates the deployed behavior.",
        "non_goals": "No code changes.",
        "dependency_notes": "After Internal schema utilities.",
        "rollout_risk": "medium - live checkpoint.",
        "validation": "Human remark and closure required.",
        "parent_context": "Approved plan slice for the manual checkpoint.",
        "automation": "human-action",
        "depends_on": [],
    }

    with pytest.raises(AgentLoopError, match="dependencies must reference an earlier phase"):
        parse_plan_decomposition(plan_decomposition_json(first_phase, second_phase))

@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda phase: phase.pop("parent_context"), "parent_context"),
        (lambda phase: phase.pop("rollout_risk"), "rollout_risk"),
        (lambda phase: phase.pop("validation"), "validation"),
        (lambda phase: phase.__setitem__("automation", "robot"), "invalid automation"),
        (lambda phase: phase.__setitem__("depends_on", ["Missing phase"]), "unknown phase"),
    ],
)
def test_parse_plan_decomposition_rejects_invalid_phase_fields(mutate, message):
    phase = {
        "title": "Internal schema utilities",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    mutate(phase)

    with pytest.raises(AgentLoopError, match=message):
        parse_plan_decomposition(plan_decomposition_json(phase))

def test_parse_plan_decomposition_rejects_duplicates_but_leaves_cap_to_preflight():
    phase = {
        "title": "Repeated phase",
        "scope": "Add helpers.",
        "non_goals": "No live switch.",
        "dependency_notes": "First phase.",
        "rollout_risk": "low - internal only.",
        "validation": "Run python -m pytest.",
        "parent_context": "Approved plan slice and invariant details.",
        "automation": "agent-pr",
        "depends_on": [],
    }
    with pytest.raises(AgentLoopError, match="duplicate phase title"):
        parse_plan_decomposition(plan_decomposition_json(phase, dict(phase)))

    phases = [dict(phase, title=f"Phase {index}") for index in range(DEFAULT_FLAT_CHILD_LIMIT + 1)]
    parsed = parse_plan_decomposition(plan_decomposition_json(*phases))
    assert len(parsed.phases) == DEFAULT_FLAT_CHILD_LIMIT + 1


def _phase(title: str, *, depends_on: tuple[str, ...] = ()) -> PlanPhase:
    return PlanPhase(
        title=title,
        scope=f"Implement {title}.",
        non_goals="No unrelated changes.",
        dependency_notes="Follow the parent plan.",
        rollout_risk="low.",
        validation="Run focused tests.",
        parent_context="Approved parent constraints.",
        automation="agent-pr",
        depends_on=depends_on,
    )


def test_topology_checkpoint_stores_shared_context_once_and_round_trips(tmp_path):
    excerpt = "Approved parent constraints.\n" + ("constraint detail\n" * 500)
    phases = tuple(
        PlanPhase(
            title=f"Stage {index}",
            scope=f"Implement stage {index}.",
            non_goals="No unrelated changes.",
            dependency_notes="Follow the parent plan.",
            rollout_risk="low.",
            validation="Run focused tests.",
            parent_context=excerpt,
            automation="agent-pr",
            depends_on=(),
        )
        for index in range(13)
    )
    checkpoint = TopologyCheckpoint(
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
        topology_source="typed",
        phases=phases,
        retained_parent_scope=RetainedParentScope(
            plan_subject="Primary scope",
            plan_hash="plan-hash",
            excerpt=excerpt,
        ),
    )

    body = format_topology_checkpoint(checkpoint)
    restored = find_existing_topology_checkpoint(
        (IssueComment(author="bot", created_at=None, body=body),),
        parent_issue=56,
        plan_hash="plan-hash",
        mode="decompose-only",
    )

    assert len(body) < 60000
    assert "constraint detail" not in body
    assert restored == checkpoint


def test_dry_run_decomposition_previews_dependency_phases_without_issue_numbers(tmp_path):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "First phase",
                "scope": "First.",
                "non_goals": "None.",
                "dependency_notes": "First.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Second phase",
                "scope": "Second.",
                "non_goals": "None.",
                "dependency_notes": "After first.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": ["First phase"],
            },
        )
    )
    runner = FakeRunner()

    created = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path, dry_run=True),
        parent_issue=56,
        approved_plan="approved plan",
        decomposition=phases,
    )

    assert len(created) == 2
    assert all(item.issue_url is None and item.issue_number is None for item in created)
    assert len(runner.issues) == 2
    assert runner.comments == []


def test_decomposition_preflight_counts_split_children_toward_shared_limit(tmp_path):
    existing_children = [
        {
            "number": 100 + index,
            "title": f"[#56 stage] Existing {index}",
            "url": f"https://github.com/OWNER/REPO/issues/{100 + index}",
            "body": f"Part of #56\n<!-- AGENT_SPLIT_CHILD: parent=56 key={index + 1:064x} -->",
        }
        for index in range(DEFAULT_FLAT_CHILD_LIMIT)
    ]
    runner = FakeRunner(search_issues_payload=existing_children)

    decision = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan="approved plan",
        decomposition=PlanDecomposition(phases=(_phase("New phase"),)),
    )

    assert isinstance(decision, NeedsHumanDecision)
    assert decision.recognized_existing_count == DEFAULT_FLAT_CHILD_LIMIT
    assert decision.projected_total == DEFAULT_FLAT_CHILD_LIMIT + 1
    assert runner.issues == []
    assert runner.comments == []


def test_decomposition_adopts_closed_exact_identity(tmp_path):
    phase = _phase("Schema helpers")
    plan = "approved plan"
    identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phase,
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phase,
        created_so_far=(),
        phase_identity_value=identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner = FakeRunner(
        search_issues_payload=[
            {
                "number": 101,
                "title": "Phase 1: Schema helpers (from #56)",
                "url": "https://github.com/OWNER/REPO/issues/101",
                "body": body,
                "state": "closed",
            }
        ]
    )

    created = create_decomposition_child_issues(
        runner,
        config=make_config(tmp_path),
        parent_issue=56,
        approved_plan=plan,
        decomposition=PlanDecomposition(phases=(phase,)),
    )

    assert created[0].origin == "adopted"
    assert created[0].issue_number == 101
    assert runner.issues == []


def test_decomposition_rejects_ambiguous_exact_identity_before_checkpoint(tmp_path):
    phase = _phase("Schema helpers")
    plan = "approved plan"
    identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phase,
    )
    body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phase,
        created_so_far=(),
        phase_identity_value=identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner = FakeRunner(search_issues_payload=[
        {"number": 101, "title": "Phase 1: Schema helpers (from #56)", "url": "u101", "body": body},
        {"number": 102, "title": "Phase 1: Schema helpers (from #56)", "url": "u102", "body": body},
    ])

    with pytest.raises(AgentLoopError, match="Ambiguous decomposition recovery"):
        create_decomposition_child_issues(
            runner,
            config=make_config(tmp_path),
            parent_issue=56,
            approved_plan=plan,
            decomposition=PlanDecomposition(phases=(phase,)),
        )
    assert runner.issues == []
    assert runner.comments == []


def test_decomposition_partial_create_recovers_from_checkpoint_and_identity(tmp_path, monkeypatch):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "First phase",
                "scope": "First.",
                "non_goals": "None.",
                "dependency_notes": "First.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Second phase",
                "scope": "Second.",
                "non_goals": "None.",
                "dependency_notes": "Second.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
        )
    )
    plan = "approved plan"
    runner = FakeRunner(
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
        search_issues_payload=[[]],
    )
    config = make_config(tmp_path)
    original_create = __import__(
        "coding_review_agent_loop.decomposition", fromlist=["create_issue"]
    ).create_issue
    calls = {"count": 0}

    def fail_after_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise AgentLoopError("simulated create failure")
        return original_create(*args, **kwargs)

    import coding_review_agent_loop.decomposition as decomp

    monkeypatch.setattr(decomp, "create_issue", fail_after_first)
    with pytest.raises(AgentLoopError, match="simulated create failure"):
        create_decomposition_child_issues(
            runner,
            config=config,
            parent_issue=56,
            approved_plan=plan,
            decomposition=phases,
        )
    assert len(runner.issues) == 1
    checkpoint_comments = tuple(
        IssueComment(author="bot", created_at=None, body=comment["body"])
        for comment in runner.issue_comments
    )
    first_identity = phase_identity(
        parent_issue=56,
        plan_hash=approved_plan_hash(plan),
        topology_source="model",
        phase_index=1,
        phase=phases.phases[0],
    )
    first_body = format_phase_issue_body(
        repo="OWNER/REPO",
        parent_issue=56,
        approved_plan=plan,
        phase=phases.phases[0],
        created_so_far=(),
        phase_identity_value=first_identity,
        topology_source="model",
        phase_index=1,
        phase_plan_hash=approved_plan_hash(plan),
    )
    runner.search_issues_payload = [{
        "number": 101,
        "title": "Phase 1: First phase (from #56)",
        "url": "https://github.com/OWNER/REPO/issues/101",
        "body": first_body,
    }]
    monkeypatch.setattr(decomp, "create_issue", original_create)

    resumed = create_decomposition_child_issues(
        runner,
        config=config,
        parent_issue=56,
        approved_plan=plan,
        decomposition=phases,
        issue_comments=checkpoint_comments,
    )

    assert [item.origin for item in resumed] == ["adopted", "created"]
    assert [item.issue_number for item in resumed] == [101, 102]
    assert len(runner.issues) == 2
    assert sum("Topology checkpoint recorded" in comment for comment in runner.comments) == 1


def test_typed_decompose_only_materializes_one_thirteen_stage_topology(tmp_path):
    stages = [
        {"title": f"Backend stage {index}", "summary": f"Backend work {index}."}
        for index in range(8)
    ] + [
        {"title": f"Frontend stage {index}", "summary": f"Frontend work {index}."}
        for index in range(5)
    ]
    runner = FakeRunner(
        claude_outputs=[structured_plan_state(summary="Primary approved scope", child_stages=stages)],
        codex_outputs=[structured_plan_review(state="approved")],
        issue_urls=[f"https://github.com/OWNER/REPO/issues/{100 + index}" for index in range(13)],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert len(runner.issues) == 13
    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1
    assert not any("AGENT_TYPED_PLAN_STAGES" in issue["body"] for issue in runner.issues)
    assert "Retained parent scope" in runner.comments[-1]


def test_decomposition_preflights_last_unsafe_phase_before_any_write(tmp_path):
    phases = parse_plan_decomposition(
        plan_decomposition_json(
            {
                "title": "Safe phase",
                "scope": "Safe.",
                "non_goals": "None.",
                "dependency_notes": "None.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
            {
                "title": "Unsafe phase",
                "scope": "Contains <!-- AGENT_TYPED_PLAN_STAGES: historical -->",
                "non_goals": "None.",
                "dependency_notes": "None.",
                "rollout_risk": "low.",
                "validation": "Tests.",
                "parent_context": "Context.",
                "automation": "agent-pr",
                "depends_on": [],
            },
        )
    )
    runner = FakeRunner(issue_urls=["https://github.com/OWNER/REPO/issues/101"])
    with pytest.raises(AgentLoopError, match="marker set mismatch"):
        create_decomposition_child_issues(
            runner,
            config=make_config(tmp_path),
            parent_issue=56,
            approved_plan="approved plan",
            decomposition=phases,
        )
    assert runner.issues == []
    assert runner.comments == []

def test_issue_loop_plan_first_decompose_only_summarizes_instead_of_filing_plan_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Split the implementation into phases."),
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(
        tmp_path,
        approved_followups="issue",
        plan_execution_mode="decompose-only",
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert not any(
        issue["title"].startswith("Follow up future plan-review note:")
        for issue in runner.issues
    )
    planning_summary = runner.comments[2]
    assert planning_summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in planning_summary
    assert "Filed future follow-up issues:" not in planning_summary
    assert "mode=summarize" in planning_summary
    assert "mode=issue" not in planning_summary

def test_issue_loop_plan_first_decompose_only_creates_child_issues(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(
                {
                    "title": "Schema helpers",
                    "scope": "Add parser dataclasses and tests.",
                    "non_goals": "No live orchestrator switch.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "low - internal only.",
                    "validation": "Run python -m pytest tests/test_agent_loop.py.",
                    "parent_context": "Approved plan slice: add schema helpers and preserve behavior.",
                    "automation": "agent-pr",
                    "depends_on": [],
                },
                {
                    "title": "Human rollout checkpoint",
                    "scope": "Human validates rollout readiness.",
                    "non_goals": "No code changes.",
                    "dependency_notes": "Depends on Schema helpers.",
                    "rollout_risk": "medium - manual checkpoint.",
                    "validation": "Human must add a remark and close the issue.",
                    "parent_context": "Approved plan slice: stop for human validation.",
                    "automation": "manual-close",
                    "depends_on": ["Schema helpers"],
                },
            ),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 2
    assert runner.issues[0]["title"] == "Phase 1: Schema helpers (from #56)"
    assert "Run `agent-loop issue <this issue number>`" in runner.issues[0]["body"]
    assert "Approved plan slice: add schema helpers" in runner.issues[0]["body"]
    assert runner.issues[1]["title"] == "[Human] Phase 2: Human rollout checkpoint (from #56)"
    assert "depends on #101: Schema helpers" in runner.issues[1]["body"]
    assert "human should add the required remark/update and close this issue" in runner.issues[1]["body"]
    summary = runner.comments[-1]
    assert summary.startswith("Approved plan decomposed for issue #56.")
    assert "Every phase above has a GitHub child issue" in summary
    assert "<!-- AGENT_PLAN_DECOMPOSITION:" in summary
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_decompose_only_is_idempotent(tmp_path):
    plan = structured_plan_state(summary="Add schema helpers.")
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="decompose-only",
        plan_hash=approved_plan_hash(plan),
        created=(),
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
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="decompose-only")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_rerun_without_handoff_implements_once(tmp_path):
    plan = structured_plan_state(summary="Add schema helpers.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
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
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "GitHub issue #99" in claude_calls[0][-1]

def test_issue_loop_plan_first_implement_by_phase_rerun_with_handoff_stops(tmp_path, capsys):
    plan = structured_plan_state(summary="Add schema helpers.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Schema helpers", automation="agent-pr"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
    )
    handoff = format_phase_implementation_handoff_comment(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        phase_index=1,
        created=child,
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
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:03Z", "body": handoff},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    output = capsys.readouterr().out
    assert "already handed off to child issue #99" in output
    assert "agent-loop issue 99" in output
    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_human_first_rerun_does_not_handoff(tmp_path):
    plan = structured_plan_state(summary="Validate migration manually first.")
    child = CreatedPhaseIssue(
        phase=RecordedPhase(title="Manual readiness check", automation="human-action"),
        issue_url="https://github.com/OWNER/REPO/issues/99",
        issue_number=99,
    )
    summary = format_decomposition_parent_summary(
        parent_issue=56,
        mode="implement-by-phase",
        plan_hash=approved_plan_hash(plan),
        created=(child,),
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
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:02Z", "body": summary},
        ],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)

def test_issue_loop_plan_first_implement_by_phase_stops_on_human_first_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Validate migration manually first."),
            plan_decomposition_json(
                {
                    "title": "Manual readiness check",
                    "scope": "Human validates external readiness.",
                    "non_goals": "No agent PR.",
                    "dependency_notes": "First phase; no dependencies.",
                    "rollout_risk": "medium - manual readiness gate.",
                    "validation": "Human remark and closure required.",
                    "parent_context": "Approved plan slice: manual readiness gate.",
                    "automation": "human-action",
                    "depends_on": [],
                }
            ),
        ],
        codex_outputs=["Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"].startswith("[Human] Phase 1")
    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)


def test_issue_loop_dry_run_implement_by_phase_allows_dependency_preview(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(
                {
                    "title": "First phase",
                    "scope": "First.",
                    "non_goals": "None.",
                    "dependency_notes": "First.",
                    "rollout_risk": "low.",
                    "validation": "Tests.",
                    "parent_context": "Context.",
                    "automation": "agent-pr",
                    "depends_on": [],
                },
                {
                    "title": "Second phase",
                    "scope": "Second.",
                    "non_goals": "None.",
                    "dependency_notes": "After first.",
                    "rollout_risk": "low.",
                    "validation": "Tests.",
                    "parent_context": "Context.",
                    "automation": "agent-pr",
                    "depends_on": ["First phase"],
                },
            ),
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )

    result = run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(
            tmp_path,
            dry_run=True,
            plan_execution_mode="implement-by-phase",
        ),
        plan_first=True,
    )

    assert result == 0
    assert len(runner.issues) == 2
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2

def test_issue_loop_plan_first_implement_by_phase_implements_first_agent_phase(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(),
            "Implemented first phase.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
        pr_payload={"body": "Fixes #99"},
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(runner.issues) == 1
    decomposition_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_DECOMPOSITION:" in comment
    )
    implementation_index = next(
        index for index, comment in enumerate(runner.comments) if comment.startswith("## Issue implementation")
    )
    handoff_index = next(
        index for index, comment in enumerate(runner.comments) if "<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment
    )
    assert decomposition_index < implementation_index < handoff_index
    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 3
    assert "GitHub issue #99" in claude_calls[2][-1]
    assert "Approved implementation plan" in claude_calls[2][-1]

def test_issue_loop_plan_first_implement_by_phase_missing_child_number_does_not_handoff(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Add schema helpers."),
            plan_decomposition_json(),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        issue_urls=[None],
    )
    config = make_config(tmp_path, plan_execution_mode="implement-by-phase")

    with pytest.raises(AgentLoopError, match="child issue number"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not any("<!-- AGENT_PLAN_PHASE_IMPLEMENTATION:" in comment for comment in runner.comments)

def test_phase_implementation_handoff_rejects_malformed_marker():
    comment = IssueComment(
        author="bot",
        created_at="2026-05-23T00:00:00Z",
        body="<!-- AGENT_PLAN_PHASE_IMPLEMENTATION: not-valid-base64 -->",
    )

    with pytest.raises(AgentLoopError, match="Invalid AGENT_PLAN_PHASE_IMPLEMENTATION payload"):
        find_existing_phase_implementation_handoff(
            (comment,),
            parent_issue=56,
            plan_hash="abc123",
            mode="implement-by-phase",
            phase_index=1,
            child_issue_number=99,
        )
