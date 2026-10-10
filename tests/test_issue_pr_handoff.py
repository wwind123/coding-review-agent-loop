import base64

import pytest

from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import (
    IssueComment,
    IssueContext,
    PullRequestMetadata,
    find_open_pr_closing_issue,
    read_pull_request_commit_metadata,
)
from coding_review_agent_loop.issue_pr_provenance import (
    IssuePrProvenanceScope,
    compare_issue_pr_provenance,
    format_issue_pr_provenance,
    parse_issue_pr_provenance_messages,
)
from coding_review_agent_loop.issue_pr_handoff import (
    _decode_issue_pr_handoff_metadata,
    _encode_issue_pr_handoff_metadata,
    _validate_issue_pr_handoff_url,
    IssuePrHandoffMetadata,
    find_latest_issue_pr_handoff,
    format_issue_pr_handoff_comment,
    require_pr_metadata_for_handoff,
)
from agent_loop_helpers import FakeRunner, make_config


def _staged_plan_checkpoint(binding, names, primary, round_number=1):
    """A primary-then-panel planning checkpoint with complete scheduler metadata."""
    from coding_review_agent_loop.plan_review_scheduling import PlanCandidateKey, make_plan_contract
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    return _attach_round_metadata("Plan scheduling checkpoint.", PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", round_number=round_number,
        subject="plan", seat_binding=binding, phase="scheduler-prelaunch",
        scheduler_contract=make_plan_contract(names, "primary-then-panel", primary).as_dict(),
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=(primary,),
        scheduler_paused_reviewers=tuple((name, "panel") for name in names if name != primary),
        scheduler_reasons=("primary gate",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=0,
        scheduler_phase="primary",
        scheduler_primary_reviewer=primary,
        plan_candidate_key=PlanCandidateKey(
            subject="plan", aggregate_plan_identity="aggregate",
            execution_strategy_identity="strategy", risk_test_matrix_identity="matrix",
            surfaced_requirement_id_digest="requirements",
        ).as_dict(),
    ))


def test_named_plan_handoff_board_rule_records_changes_instead_of_refusing(tmp_path):
    """#1373: a changed model or seat set is an operator reconfiguration."""
    from dataclasses import replace
    from types import SimpleNamespace
    from coding_review_agent_loop.plan_verification import derive_plan_verification_context
    from coding_review_agent_loop.reviewer_seats import (
        ReviewerSeat, SeatAgent, resolve_plan_handoff_board, reviewer_seat_binding,
    )
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    flash = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    opus = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model B",)), tmp_path / "opus")
    config = make_config(
        tmp_path, reviewer=(flash, opus), reviewer_seats=(flash, opus),
        plan_review_policy="primary-then-panel", primary_plan_reviewer=opus,
    )
    binding = reviewer_seat_binding(config)
    comments = [
        SimpleNamespace(body=_attach_round_metadata("Plan.", PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject="plan", canonical_plan="Plan.", seat_binding=binding,
        ))),
        SimpleNamespace(body=_staged_plan_checkpoint(binding, ("flash", "opus"), "opus")),
    ]
    plan = derive_plan_verification_context(comments, issue_number=56)
    assert plan.effective_reviewers == ("flash", "opus")
    assert plan.policy == "primary-then-panel" and plan.primary_reviewer == "opus"
    assert plan.signed_authorization is None

    unchanged = resolve_plan_handoff_board(config, plan)
    assert unchanged.inherited and not unchanged.changed

    changed_opus = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model C",)), tmp_path / "opus")
    changed_model = resolve_plan_handoff_board(replace(
        config, reviewer=(flash, changed_opus), reviewer_seats=(flash, changed_opus),
        primary_plan_reviewer=changed_opus,
    ), plan)
    assert changed_model.inherited
    assert (changed_model.added, changed_model.removed) == ((), ())
    ((before, after),) = changed_model.binding_changes
    assert (before.seat_id, before.model_chain, after.model_chain) == (
        "opus", ("Model B",), ("Model C",),
    )

    other = SeatAgent(ReviewerSeat("other", "codex", ("gpt-6-sol",)), tmp_path / "other")
    added = resolve_plan_handoff_board(replace(
        config, reviewer=(flash, opus, other), reviewer_seats=(flash, opus, other),
    ), plan)
    assert not added.inherited
    assert added.added == ("other",) and added.removed == ()
    assert tuple(added.config.reviewer) == (flash, opus, other)

    plain = resolve_plan_handoff_board(replace(
        config, reviewer=("codex",), reviewer_seats=(), plan_review_policy="all-reviewers",
        primary_plan_reviewer=None,
    ), plan)
    assert not plain.inherited
    assert plain.added == ("Codex",) and plain.removed == ("flash", "opus")


def test_named_plan_handoff_applies_signed_outage_and_restoration(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace
    from coding_review_agent_loop.board_amendment import (
        collect_reviewer_board_amendments, format_reviewer_board_amendment_comment,
    )
    from coding_review_agent_loop.plan_review_scheduling import PlanCandidateKey, make_plan_contract
    from coding_review_agent_loop.reviewer_seats import (
        ReviewerSeat, SeatAgent, reconcile_plan_handoff_board, reviewer_seat_binding,
    )
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    seats = tuple(
        SeatAgent(ReviewerSeat(name, backend, (model,)), tmp_path / name)
        for name, backend, model in (
            ("primary", "codex", "gpt-6-sol"),
            ("flash", "antigravity", "Model A"),
            ("opus", "antigravity", "Model B"),
            ("other", "claude", "claude-sonnet"),
        )
    )
    config = make_config(
        tmp_path, reviewer=seats, reviewer_seats=seats,
        plan_review_policy="primary-then-panel", primary_plan_reviewer=seats[0],
    )
    board = tuple(str(seat) for seat in seats)
    binding = reviewer_seat_binding(config)

    def checkpoint(round_number, names, digest=None):
        return _attach_round_metadata("Plan scheduling checkpoint.", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=round_number,
            subject="plan", seat_binding=binding, phase="scheduler-prelaunch",
            scheduler_contract=make_plan_contract(
                names, "primary-then-panel", "primary",
            ).as_dict(),
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("primary",),
            scheduler_paused_reviewers=tuple((name, "panel") for name in names if name != "primary"),
            scheduler_reasons=("primary gate",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
            scheduler_phase="primary",
            scheduler_primary_reviewer="primary",
            plan_candidate_key=PlanCandidateKey(
                subject="plan", aggregate_plan_identity="aggregate",
                execution_strategy_identity="strategy", risk_test_matrix_identity="matrix",
                surfaced_requirement_id_digest="requirements",
            ).as_dict(),
            reviewer_board_amendment_digest=digest,
        ))

    removal = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=board, policy="primary-then-panel",
        primary_reviewer="primary", removed_reviewers=("flash", "opus"),
        effective_from_round=2, rationale="Shared account outage.",
    )
    (removed,) = collect_reviewer_board_amendments(
        [SimpleNamespace(body=removal)], flow="plan", issue_number=56,
    )
    comments = [
        SimpleNamespace(body=checkpoint(1, board)),
        SimpleNamespace(body=removal),
        SimpleNamespace(body=checkpoint(2, ("primary", "other"), removed.digest)),
    ]
    reduced = reconcile_plan_handoff_board(config, comments, 56)
    assert tuple(str(seat) for seat in reduced.reviewer) == ("primary", "other")
    assert reviewer_seat_binding(reduced) == binding
    # A direct PR invocation uses its default plan flags, even when the
    # approved issue plan selected a named primary.
    direct_pr = replace(config, plan_review_policy="all-reviewers", primary_plan_reviewer=None)
    reduced_from_pr = reconcile_plan_handoff_board(direct_pr, comments, 56)
    assert tuple(str(seat) for seat in reduced_from_pr.reviewer) == ("primary", "other")
    assert reviewer_seat_binding(reduced_from_pr) == binding

    restoration = format_reviewer_board_amendment_comment(
        flow="plan", issue=56, pr_number=None,
        original_required_reviewers=("primary", "other"),
        policy="primary-then-panel", primary_reviewer="primary",
        removed_reviewers=(), restored_reviewers=("flash", "opus"),
        effective_from_round=3, rationale="Shared account recovered.",
    )
    (restored,) = collect_reviewer_board_amendments(
        [SimpleNamespace(body=restoration)], flow="plan", issue_number=56,
    )
    comments.extend([
        SimpleNamespace(body=restoration),
        SimpleNamespace(body=checkpoint(3, board, restored.digest)),
    ])
    full = reconcile_plan_handoff_board(config, comments, 56)
    assert tuple(str(seat) for seat in full.reviewer) == board


def test_named_staged_handoff_uses_parent_board_when_child_has_unrelated_plan(tmp_path):
    from agent_loop_helpers import structured_pr_review
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.plan_review_scheduling import make_plan_contract
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent, reviewer_seat_binding
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata, _attach_round_metadata, make_approved_plan_context,
    )

    seats = tuple(
        SeatAgent(ReviewerSeat(name, "antigravity", (model,)), tmp_path / name)
        for name, model in (("flash", "Model A"), ("opus", "Model B"))
    )
    config = make_config(
        tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False,
    )
    approved = make_approved_plan_context("Parent approved plan.")
    binding = reviewer_seat_binding(config)

    def plan_comments(plan, board, record_binding):
        candidate = _attach_round_metadata(plan, PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject="plan", canonical_plan=plan, seat_binding=record_binding,
        ))
        checkpoint = _attach_round_metadata("Plan scheduling checkpoint.", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=1,
            subject="plan", seat_binding=record_binding,
            scheduler_contract=make_plan_contract(board, "all-reviewers", None).as_dict(),
        ))
        return [
            {"body": body, "author": {"login": "bot"},
             "createdAt": f"2026-05-23T00:00:0{index}Z", "id": index}
            for index, body in enumerate((candidate, checkpoint), start=1)
        ]

    parent_comments = plan_comments(approved.canonical_text, ("flash", "opus"), binding)
    child_comments = plan_comments("Unrelated child plan.", ("other",), {
        "version": 1, "seats": [{"id": "other", "backend": "codex",
                                "model_chain": ["gpt-6-sol"], "effort": None}],
    })
    runner = FakeRunner(
        issue_payloads_by_number={55: {"number": 55}, 56: {"number": 56}},
        issue_comments_by_number={55: parent_comments, 56: child_comments},
        antigravity_outputs=[
            structured_pr_review(reviewer=f"{name} (Google Antigravity: {model})")
            for name, model in (("flash", "Model A"), ("opus", "Model B"))
        ],
    )
    child = IssueContext(56, config.repo, "Child", "Child body", None, ())
    parent = IssueContext(55, config.repo, "Parent", "Parent body", None, ())
    assert run_pr_loop(
        runner, pr_number=77, config=config, issue_context=child,
        parent_issue_context=parent, approved_plan_context=approved,
    ) == 0
    assert len([command for command, _ in runner.commands if command[0] == "agy"]) == 2
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == 2
    assert any("flash" in comment for comment in reviews)
    assert any("opus" in comment for comment in reviews)

    assert run_pr_loop(
        runner, pr_number=77, config=config, issue_context=child,
        parent_issue_context=parent, approved_plan_context=approved,
    ) == 0
    assert len([command for command, _ in runner.commands if command[0] == "agy"]) == 2
    assert len([comment for comment in runner.comments if "**Review verdict:**" in comment]) == 2


# --- Per-invocation PR boards at issue-to-PR handoff (#1373) ----------------

_AGENT_COMMANDS = {"codex", "claude", "agy"}


def _seat(tmp_path, name, backend, model, effort=None):
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    return SeatAgent(ReviewerSeat(name, backend, (model,), effort), tmp_path / name)


def _named_config(tmp_path, seats, **overrides):
    overrides.setdefault("pre_review_tests", False)
    return make_config(tmp_path, reviewer=tuple(seats), reviewer_seats=tuple(seats), **overrides)


def _named_plan_comments(approved, plan_config, *, removed=(), restored=False):
    """An approved named plan: candidate, round-1 checkpoint, signed board links."""
    from types import SimpleNamespace
    from coding_review_agent_loop.agents.registry import agent_display_name
    from coding_review_agent_loop.board_amendment import (
        collect_reviewer_board_amendments, format_reviewer_board_amendment_comment,
    )
    from coding_review_agent_loop.plan_review_scheduling import make_plan_contract
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    binding = reviewer_seat_binding(plan_config)
    board = tuple(agent_display_name(seat) for seat in plan_config.reviewer)

    def checkpoint(round_number, names, digest=None):
        return _attach_round_metadata("Plan scheduling checkpoint.", PostedRoundMetadata(
            flow="plan", role="summary", agent="Orchestrator", round_number=round_number,
            subject="plan", seat_binding=binding,
            scheduler_contract=make_plan_contract(names, "all-reviewers", None).as_dict(),
            reviewer_board_amendment_digest=digest,
        ))

    def signed(text):
        (amendment,) = collect_reviewer_board_amendments(
            [SimpleNamespace(body=text)], flow="plan", issue_number=56,
        )
        return amendment.digest

    bodies = [
        _attach_round_metadata(approved.canonical_text, PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject="plan", canonical_plan=approved.canonical_text, seat_binding=binding,
        )),
        checkpoint(1, board),
    ]
    if removed:
        reduced = tuple(name for name in board if name not in removed)
        removal = format_reviewer_board_amendment_comment(
            flow="plan", issue=56, pr_number=None, original_required_reviewers=board,
            policy="all-reviewers", primary_reviewer=None, removed_reviewers=tuple(removed),
            effective_from_round=2, rationale="Backend quota exhausted.",
        )
        bodies += [removal, checkpoint(2, reduced, signed(removal))]
        if restored:
            restoration = format_reviewer_board_amendment_comment(
                flow="plan", issue=56, pr_number=None, original_required_reviewers=reduced,
                policy="all-reviewers", primary_reviewer=None, removed_reviewers=(),
                restored_reviewers=tuple(removed), effective_from_round=3,
                rationale="Backend quota recovered.",
            )
            bodies += [restoration, checkpoint(3, board, signed(restoration))]
    return [
        {"body": body, "author": {"login": "bot"},
         "createdAt": f"2026-05-23T00:00:{index:02d}Z", "id": index}
        for index, body in enumerate(bodies, start=1)
    ]


def _review(agent, config, **kwargs):
    from agent_loop_helpers import structured_pr_review
    from coding_review_agent_loop.agents.registry import agent_signature

    return structured_pr_review(reviewer=agent_signature(agent, config), **kwargs)


def _handoff_run(pr_config, approved, issue_comments, runner=None, **outputs):
    from coding_review_agent_loop.cli import run_pr_loop

    runner = runner or FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: issue_comments},
        **outputs,
    )
    issue = IssueContext(56, pr_config.repo, "Issue", "Body", None, ())
    result = run_pr_loop(
        runner, pr_number=77, config=pr_config, issue_context=issue,
        approved_plan_context=approved,
    )
    return result, runner


def _audits(runner):
    from coding_review_agent_loop.plan_verification import BOARD_CHANGE_AUDIT_HEADING

    return [comment for comment in runner.comments if BOARD_CHANGE_AUDIT_HEADING in comment]


def _agent_commands(runner):
    return [command for command, _cwd in runner.commands if command[0] in _AGENT_COMMANDS]


def _opus_plan(tmp_path):
    from coding_review_agent_loop.round_state import make_approved_plan_context

    seats = (
        _seat(tmp_path, "sol", "codex", "gpt-6.1-sol", "medium"),
        _seat(tmp_path, "flash", "gemini", "gemini-3.5-flash"),
        _seat(tmp_path, "opus-low", "antigravity", "Claude Opus 5.5 (Low)"),
    )
    approved = make_approved_plan_context("Approved named plan for #1373.")
    return seats, approved


def test_named_plan_hands_off_to_plain_board_with_one_audit_record(tmp_path):
    """handoff-named-to-plain and handoff-resume-idempotent."""
    seats, approved = _opus_plan(tmp_path)
    plan_comments = _named_plan_comments(approved, _named_config(tmp_path, seats))
    pr_config = make_config(
        tmp_path, reviewer=("codex", "claude", "antigravity"), pre_review_tests=False,
    )
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: plan_comments},
        codex_outputs=[_review("codex", pr_config)],
    )
    # The first attempt runs out of scripted reviewers after Codex: round 1 is
    # partially run with the audit record already posted.
    with pytest.raises(AgentLoopError):
        _handoff_run(pr_config, approved, plan_comments, runner=runner)
    (audit,) = _audits(runner)
    assert "- Reason: operator-reconfigured" in audit
    assert "- Effective round: 1" in audit
    assert "sol (codex: gpt-6.1-sol, effort medium)" in audit
    assert "opus-low (antigravity: Claude Opus 5.5 (Low))" in audit
    assert "- Added reviewer(s): Codex, Claude, Antigravity" in audit
    assert "- Removed reviewer(s): sol, flash, opus-low" in audit
    # The record is orchestrator text, not a signed amendment or requirement.
    from types import SimpleNamespace
    from coding_review_agent_loop.board_amendment import collect_reviewer_board_amendments
    assert collect_reviewer_board_amendments(
        [SimpleNamespace(body=audit)], flow="pr", pr_number=77,
    ) == ()
    assert "<!--" not in audit
    # Round 1 went to the new board; nothing was refused.
    assert [command[0] for command in _agent_commands(runner)][:1] == ["codex"]

    runner.claude_outputs = [_review("claude", pr_config)]
    runner.antigravity_outputs = [_review("antigravity", pr_config)]
    result, _ = _handoff_run(pr_config, approved, plan_comments, runner=runner)
    assert result == 0
    assert len(_audits(runner)) == 1
    # Every PR reviewer approved the PR itself; no plan approval stood in.
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == 3
    assert {command[0] for command in _agent_commands(runner)} == _AGENT_COMMANDS

    result, _ = _handoff_run(pr_config, approved, plan_comments, runner=runner)
    assert result == 0
    assert len(_audits(runner)) == 1


def test_named_plan_handoff_with_changed_seat_model_lists_only_the_binding_change(tmp_path):
    """handoff-model-change."""
    seats, approved = _opus_plan(tmp_path)
    plan_comments = _named_plan_comments(approved, _named_config(tmp_path, seats))
    remodeled = _seat(tmp_path, "opus-low", "antigravity", "Claude Opus 5.5 (High)")
    pr_config = _named_config(tmp_path, (seats[0], seats[1], remodeled))
    result, runner = _handoff_run(
        pr_config, approved, plan_comments,
        codex_outputs=[_review(seats[0], pr_config)],
        gemini_outputs=[_review(seats[1], pr_config)],
        antigravity_outputs=[_review(remodeled, pr_config)],
    )
    assert result == 0
    (audit,) = _audits(runner)
    assert (
        "- Binding change(s): opus-low (antigravity: Claude Opus 5.5 (Low)) -> "
        "opus-low (antigravity: Claude Opus 5.5 (High))"
    ) in audit
    assert "- Added reviewer(s): none" in audit and "- Removed reviewer(s): none" in audit
    # The re-modeled seat reviewed the PR head on its new model.
    agy = [command for command in _agent_commands(runner) if command[0] == "agy"]
    assert any("Claude Opus 5.5 (High)" in " ".join(command) for command in agy)
    assert not any("Claude Opus 5.5 (Low)" in " ".join(command) for command in agy)


def test_unchanged_handoff_board_posts_no_audit_record(tmp_path):
    """handoff-unchanged, for a named and a plain plan board."""
    from agent_loop_helpers import approved_plan_comments
    from coding_review_agent_loop.round_state import make_approved_plan_context

    seats, approved = _opus_plan(tmp_path)
    pr_config = _named_config(tmp_path, seats)
    result, runner = _handoff_run(
        pr_config, approved, _named_plan_comments(approved, pr_config),
        codex_outputs=[_review(seats[0], pr_config)],
        gemini_outputs=[_review(seats[1], pr_config)],
        antigravity_outputs=[_review(seats[2], pr_config)],
    )
    assert result == 0
    assert _audits(runner) == []

    plain_plan = "Approved plain plan."
    plain_config = make_config(tmp_path, reviewer=("codex",), pre_review_tests=False)
    result, runner = _handoff_run(
        plain_config, make_approved_plan_context(plain_plan),
        approved_plan_comments(plain_plan),
        codex_outputs=[_review("codex", plain_config)],
    )
    assert result == 0
    assert _audits(runner) == []


def test_plain_plan_hands_off_to_a_named_pr_board(tmp_path):
    from agent_loop_helpers import approved_plan_comments
    from coding_review_agent_loop.round_state import make_approved_plan_context

    plan = "Approved plain plan."
    seat = _seat(tmp_path, "sol", "codex", "gpt-6.1-sol", "medium")
    pr_config = _named_config(tmp_path, (seat,))
    result, runner = _handoff_run(
        pr_config, make_approved_plan_context(plan), approved_plan_comments(plan),
        codex_outputs=[_review(seat, pr_config)],
    )
    assert result == 0
    (audit,) = _audits(runner)
    assert "- Plan board: Codex (codex)" in audit
    assert "- Added reviewer(s): sol" in audit


@pytest.mark.parametrize("restored", [False, True])
def test_signed_plan_removal_carries_over_under_the_floor(tmp_path, restored):
    """handoff-signed-plan-carryover with --min-reviewers 2."""
    seats, approved = _opus_plan(tmp_path)
    plan_comments = _named_plan_comments(
        approved, _named_config(tmp_path, seats), removed=("opus-low",), restored=restored,
    )
    pr_config = _named_config(tmp_path, seats, min_reviewers=2)
    expected = seats if restored else seats[:2]
    result, runner = _handoff_run(
        pr_config, approved, plan_comments,
        codex_outputs=[_review(seats[0], pr_config)],
        gemini_outputs=[_review(seats[1], pr_config)],
        antigravity_outputs=[_review(seat, pr_config) for seat in expected[2:]],
    )
    assert result == 0
    assert _audits(runner) == []
    agy = " ".join(" ".join(command) for command in _agent_commands(runner) if command[0] == "agy")
    assert ("Claude Opus 5.5 (Low)" in agy) is restored
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == len(expected)

    # A different one-reviewer board is the operator's choice, not the signed
    # board: the floor refuses it before any post or agent.
    one = make_config(tmp_path, reviewer=("codex",), pre_review_tests=False, min_reviewers=2)
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: plan_comments},
    )
    with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
        _handoff_run(one, approved, plan_comments, runner=runner)
    assert runner.comments == []
    assert _agent_commands(runner) == []


def _three_provider_plan(tmp_path):
    from coding_review_agent_loop.round_state import make_approved_plan_context

    seats = (
        _seat(tmp_path, "a", "codex", "gpt-6.1-sol", "medium"),
        _seat(tmp_path, "b", "claude", "claude-opus-5-5", "medium"),
        _seat(tmp_path, "c", "antigravity", "Gemini 3.5 Pro"),
    )
    return seats, make_approved_plan_context("Approved three-provider plan.")


@pytest.mark.parametrize("repeat_original", [True, False])
def test_signed_plan_reduction_excuses_only_the_measures_it_lowered(tmp_path, repeat_original):
    """handoff-signed-exception-binding-swap."""
    seats, approved = _three_provider_plan(tmp_path)
    plan_comments = _named_plan_comments(
        approved, _named_config(tmp_path, seats), removed=("c",),
    )
    floor = {"min_reviewers": 3, "min_distinct_providers": 2}
    configured = seats if repeat_original else seats[:2]
    pr_config = _named_config(tmp_path, configured, **floor)
    result, runner = _handoff_run(
        pr_config, approved, plan_comments,
        codex_outputs=[_review(seats[0], pr_config)],
        claude_outputs=[_review(seats[1], pr_config)],
    )
    assert result == 0
    assert _audits(runner) == []
    assert {command[0] for command in _agent_commands(runner)} == {"codex", "claude"}

    swapped_b = _seat(tmp_path, "b", "codex", "gpt-6.1-flash", "medium")
    swapped = (seats[0], swapped_b, seats[2]) if repeat_original else (seats[0], swapped_b)
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: plan_comments},
    )
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        _handoff_run(_named_config(tmp_path, swapped, **floor), approved, plan_comments, runner=runner)
    assert runner.comments == []
    assert _agent_commands(runner) == []


@pytest.mark.parametrize(
    ("floor", "message"),
    [
        ({"min_reviewers": 3}, "2 reviewer\\(s\\) is below the floor of 3"),
        ({"min_distinct_providers": 3}, "2 distinct provider\\(s\\) is below the floor of 3"),
    ],
)
def test_unamended_plan_board_gets_no_floor_exception(tmp_path, floor, message):
    """handoff-unamended-floor-no-exception."""
    seats, approved = _three_provider_plan(tmp_path)
    plan_comments = _named_plan_comments(approved, _named_config(tmp_path, seats[:2]))
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: plan_comments},
    )
    with pytest.raises(AgentLoopError, match=message):
        _handoff_run(_named_config(tmp_path, seats[:2], **floor), approved, plan_comments, runner=runner)
    assert runner.comments == []
    assert _agent_commands(runner) == []


def _plain_plan_comments(plan, approvals):
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata, _attach_round_metadata, _plan_subject,
    )

    subject = _plan_subject(plan)
    bodies = [_attach_round_metadata(plan, PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1, subject=subject,
        canonical_plan=plan, raw_structured_coder_response=plan,
    ))]
    for agent, state in approvals:
        bodies.append(_attach_round_metadata(f"{state}.", PostedRoundMetadata(
            flow="plan", role="reviewer", agent=agent, round_number=1,
            subject=subject, state=state,
        )))
    return [
        {"author": {"login": "coding-review-agent-loop"},
         "createdAt": f"2026-05-01T00:00:{index:02d}Z", "body": body}
        for index, body in enumerate(bodies)
    ]


def _strict_verify(tmp_path, issue_comments, *, pr_reviewers, expected_hash):
    from types import SimpleNamespace
    from coding_review_agent_loop.github import IssueContext as Issue
    from coding_review_agent_loop.pr_loop_support import _verify_strict_managed_plan_binding

    issue = Issue(959, "OWNER/REPO", "t", "b", None, tuple(
        IssueComment(author="agent-loop", created_at=comment["createdAt"], body=comment["body"])
        for comment in issue_comments
    ))
    _verify_strict_managed_plan_binding(
        config=make_config(tmp_path, reviewer=pr_reviewers),
        pr_number=7,
        issue_context=issue,
        metadata=SimpleNamespace(head_branch="agent-loop/managed-959", head_sha="head-1"),
        expected_plan_hash=expected_hash,
    )


def test_strict_qualification_verifies_the_plan_against_its_own_board(tmp_path):
    """handoff-plan-binding-preserved: plan board A, PR board B."""
    from coding_review_agent_loop.decomposition import approved_plan_hash

    plan = "Approved plan.\n\n### Plan steps\n1. Keep plan identity bound."
    approved_by_a = _plain_plan_comments(plan, (("Codex", "approved"), ("Gemini", "approved")))
    # PR board B differs completely from A; the plan still verifies.
    _strict_verify(
        tmp_path, approved_by_a, pr_reviewers=("claude",),
        expected_hash=approved_plan_hash(plan),
    )
    # Board B's own verdicts never stand in for board A's.
    incomplete = _plain_plan_comments(plan, (("Codex", "approved"), ("Gemini", "blocking")))
    with pytest.raises(AgentLoopError, match="not completely approved"):
        _strict_verify(
            tmp_path, incomplete, pr_reviewers=("codex",),
            expected_hash=approved_plan_hash(plan),
        )
    with pytest.raises(AgentLoopError, match="canonical approved plan changed"):
        _strict_verify(
            tmp_path, approved_by_a, pr_reviewers=("claude",), expected_hash="0" * 16,
        )


def test_plan_verification_context_keeps_a_primary_then_panel_plan_policy(tmp_path):
    """handoff-plan-verification-recovery: the plan's own policy and primary."""
    from types import SimpleNamespace
    from coding_review_agent_loop.plan_verification import plan_verification_inputs
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata

    seats = (
        _seat(tmp_path, "primary", "codex", "gpt-6.1-sol", "medium"),
        _seat(tmp_path, "panel", "antigravity", "Gemini 3.5 Pro"),
    )
    plan_config = _named_config(
        tmp_path, seats, plan_review_policy="primary-then-panel",
        primary_plan_reviewer=seats[0],
    )
    binding = reviewer_seat_binding(plan_config)
    comments = [
        SimpleNamespace(body=_attach_round_metadata("Plan.", PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1, subject="plan",
            canonical_plan="Plan.", seat_binding=binding,
        ))),
        SimpleNamespace(body=_staged_plan_checkpoint(binding, ("primary", "panel"), "primary")),
    ]
    pr_config = make_config(
        tmp_path, reviewer=("claude", "gemini"), pr_review_policy="primary-then-panel",
        primary_reviewer="claude",
    )
    verified, _ = plan_verification_inputs(pr_config, comments, issue_number=56)
    assert tuple(str(reviewer) for reviewer in verified.reviewer) == ("primary", "panel")
    assert verified.plan_review_policy == "primary-then-panel"
    assert str(verified.primary_plan_reviewer) == "primary"
    assert verified.primary_reviewer is None
    # The PR config itself keeps its own board and policy.
    assert pr_config.reviewer == ("claude", "gemini")


def _managed_plan_issue_runner(plan, approvals):
    return FakeRunner(
        issue_comments=_plain_plan_comments(plan, approvals),
        pr_payload={
            "headRefName": "agent-loop/managed-56", "headRefOid": "abc123",
            "baseRefName": "main", "body": "Fixes #56",
        },
    )


class _ScopeCaptured(Exception):
    pass


def test_managed_ci_recovery_verifies_plan_board_not_pr_board(tmp_path, monkeypatch):
    """handoff-plan-verification-recovery: fresh authorization and ordinary resume."""
    import coding_review_agent_loop.orchestrator as orchestrator
    from coding_review_agent_loop.cli import run_pr_loop

    plan = "Approved plan.\n\n### Plan steps\n1. Preserve the trust boundary."
    managed = dict(
        managed_ci=True, managed_ci_pr_mode=True, managed_ci_trusted_actor="agent-loop",
        allow_unprotected_managed_ci=True, reviewer=("claude", "gemini"),
    )
    captured = {}

    def capture(*_args, **kwargs):
        captured.update(kwargs)
        raise _ScopeCaptured

    # Fresh authorization: plan approved by Codex; the PR board is different.
    monkeypatch.setattr(orchestrator, "authorize_fresh_issue_created_resume", capture)
    runner = _managed_plan_issue_runner(plan, (("Codex", "approved"),))
    with pytest.raises(_ScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=make_config(
            tmp_path, managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
            **managed,
        ))
    assert captured["approved_plan_hash"] == orchestrator.approved_plan_hash(plan)
    assert _agent_commands(runner) == []

    # Ordinary resume without an issue-side handoff.
    handoff = orchestrator.AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56",
        trusted_actor_login="agent-loop", trusted_actor_id=1,
        protection_mode="voluntary", override_nonce="opening-nonce",
    )
    monkeypatch.setattr(orchestrator, "recover_issue_created_handoff", lambda *_a, **_k: handoff)
    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", capture)
    captured.clear()
    runner = _managed_plan_issue_runner(plan, (("Codex", "approved"),))
    with pytest.raises(_ScopeCaptured):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **managed))
    assert captured["handoff"].approved_plan_hash == orchestrator.approved_plan_hash(plan)

    # The plan board's own blocking verdict still fails closed.
    runner = _managed_plan_issue_runner(plan, (("Codex", "blocking"),))
    with pytest.raises(AgentLoopError, match="incomplete canonical plan approval"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **managed))


def _comment(body: str) -> IssueComment:
    return IssueComment(author="bot", created_at="2026-05-23T00:00:00Z", body=body)


def _metadata(**overrides) -> IssuePrHandoffMetadata:
    defaults = dict(
        schema_version=1,
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
    )
    defaults.update(overrides)
    return IssuePrHandoffMetadata(**defaults)


def _commit_page(
    messages,
    *,
    head="abc123",
    total=None,
    has_next=False,
    cursor=None,
    oid_start=1,
):
    nodes = [
        {"commit": {"oid": f"commit-{index}", "message": message}}
        for index, message in enumerate(messages, start=oid_start)
    ]
    return {
        "data": {
            "repository": {
                "pullRequest": {
                    "headRefOid": head,
                    "commits": {
                        "totalCount": len(nodes) if total is None else total,
                        "pageInfo": {"hasNextPage": has_next, "endCursor": cursor},
                        "nodes": nodes,
                    },
                }
            }
        }
    }


def _provenance_pages(*, issue=56, flow="direct", plan=None):
    scope = IssuePrProvenanceScope("OWNER/REPO", issue, flow, plan)
    page = _commit_page([f"Change\n\n{format_issue_pr_provenance(scope)}"])
    return [page, page]


def test_round_trips_metadata_through_marker():
    metadata = _metadata()
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    assert _decode_issue_pr_handoff_metadata(encoded) == metadata


def test_round_trips_approved_plan_flow_with_plan_hash():
    metadata = _metadata(flow="approved-plan-implementation", plan_hash="deadbeef01234567")
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    assert _decode_issue_pr_handoff_metadata(encoded) == metadata


def test_round_trips_complete_expected_closing_set_and_supersession_lineage():
    metadata = _metadata(
        expected_closing_issue_ids=(56, 57),
        supersedes_hash="a" * 64,
    )

    encoded = _encode_issue_pr_handoff_metadata(metadata)

    assert _decode_issue_pr_handoff_metadata(encoded) == metadata
    rendered = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
        expected_closing_issue_ids=(56, 57),
        supersedes_hash="a" * 64,
    )
    assert "Expected closing issues: #56, #57." in rendered


def test_find_latest_issue_pr_handoff_returns_newest_when_multiple_markers_present():
    older = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="oldsha",
        flow="issue-implementation",
        plan_hash=None,
    )
    newer = format_issue_pr_handoff_comment(
        issue_number=56,
        pr_number=90,
        pr_url="https://github.com/OWNER/REPO/pull/90",
        pr_head_sha="newsha",
        flow="issue-implementation",
        plan_hash=None,
    )
    comments = [_comment(older), _comment(newer)]

    found = find_latest_issue_pr_handoff(comments, issue_number=56, repo="OWNER/REPO")

    assert found is not None
    assert found.pr_number == 90
    assert found.pr_head_sha == "newsha"


def test_find_latest_issue_pr_handoff_ignores_records_for_other_issues():
    comment = format_issue_pr_handoff_comment(
        issue_number=99,
        pr_number=77,
        pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha="abc123",
        flow="issue-implementation",
        plan_hash=None,
    )

    found = find_latest_issue_pr_handoff([_comment(comment)], issue_number=56, repo="OWNER/REPO")

    assert found is None


def test_find_latest_issue_pr_handoff_returns_none_with_no_matching_comments():
    assert find_latest_issue_pr_handoff([_comment("just talk")], issue_number=56, repo="OWNER/REPO") is None


def test_decode_raises_on_malformed_base64_payload():
    with pytest.raises(AgentLoopError, match="Invalid AGENT_ISSUE_PR_HANDOFF payload"):
        _decode_issue_pr_handoff_metadata("!!!not-base64!!!")


def test_decode_raises_on_malformed_json_payload():
    encoded = base64.urlsafe_b64encode(b"not json").decode("ascii")
    with pytest.raises(AgentLoopError, match="Invalid AGENT_ISSUE_PR_HANDOFF payload"):
        _decode_issue_pr_handoff_metadata(encoded)


def test_decode_raises_on_unsupported_schema_version():
    metadata = _metadata(schema_version=2)
    with pytest.raises(AgentLoopError, match="unsupported schema_version"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_fractional_schema_version():
    metadata = _metadata(schema_version=1.9)
    with pytest.raises(AgentLoopError, match="`schema_version` must be an integer"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_boolean_schema_version():
    # `True` coerces to `1` (== SCHEMA_VERSION) via a naive `int(...)` cast; the decoder
    # must reject bool outright rather than silently accepting it as a valid version.
    metadata = _metadata(schema_version=True)
    with pytest.raises(AgentLoopError, match="`schema_version` must be an integer"):
        _decode_issue_pr_handoff_metadata(_encode_issue_pr_handoff_metadata(metadata))


def test_decode_raises_on_unknown_flow():
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    payload = base64.urlsafe_b64decode(encoded)
    import json as _json

    data = _json.loads(payload)
    data["flow"] = "something-else"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="unknown flow"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("field,value", [("issue_number", 0), ("pr_number", -1)])
def test_decode_raises_on_non_positive_identifiers(field, value):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: value}))
    with pytest.raises(AgentLoopError, match="must be positive"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_non_coercible_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data[field] = "not-a-number"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_boolean_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: True}))
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("field", ["issue_number", "pr_number"])
def test_decode_raises_on_fractional_identifiers(field):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(**{field: 56.5}))
    with pytest.raises(AgentLoopError, match="must be integers"):
        _decode_issue_pr_handoff_metadata(encoded)


@pytest.mark.parametrize("pr_url", [None, ""])
def test_decode_raises_on_missing_or_empty_pr_url(pr_url):
    encoded = _encode_issue_pr_handoff_metadata(_metadata(pr_url=pr_url or ""))
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["pr_url"] = pr_url
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`pr_url` must be a non-empty string"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


@pytest.mark.parametrize("pr_head_sha", [None, ""])
def test_decode_raises_on_missing_or_empty_pr_head_sha(pr_head_sha):
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["pr_head_sha"] = pr_head_sha
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`pr_head_sha` must be a non-empty string"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


def test_decode_raises_when_plan_hash_present_for_issue_implementation_flow():
    encoded = _encode_issue_pr_handoff_metadata(_metadata())
    import json as _json

    data = _json.loads(base64.urlsafe_b64decode(encoded))
    data["plan_hash"] = "deadbeef01234567"
    bad_encoded = base64.urlsafe_b64encode(_json.dumps(data).encode("utf-8")).decode("ascii")
    with pytest.raises(AgentLoopError, match="`plan_hash` must be absent"):
        _decode_issue_pr_handoff_metadata(bad_encoded)


def test_decode_raises_when_plan_hash_absent_for_approved_plan_flow():
    metadata = _metadata(flow="approved-plan-implementation", plan_hash=None)
    encoded = _encode_issue_pr_handoff_metadata(metadata)
    with pytest.raises(AgentLoopError, match="`plan_hash` is required"):
        _decode_issue_pr_handoff_metadata(encoded)


def test_validate_url_accepts_matching_url():
    _validate_issue_pr_handoff_url(
        "https://github.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
    )


def test_validate_url_rejects_spoofed_host():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "https://evil.example.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
        )


def test_validate_url_rejects_prefix_collision():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "https://github.com/OWNER/REPO/pull/770", repo="OWNER/REPO", pr_number=77
        )


def test_validate_url_rejects_non_https_scheme():
    with pytest.raises(AgentLoopError, match="does not match"):
        _validate_issue_pr_handoff_url(
            "http://github.com/OWNER/REPO/pull/77", repo="OWNER/REPO", pr_number=77
        )


def test_require_pr_metadata_for_handoff_raises_on_missing_url():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="abc123",
        url=None,
    )
    with pytest.raises(AgentLoopError, match="PR URL is unavailable"):
        require_pr_metadata_for_handoff(metadata)


def test_require_pr_metadata_for_handoff_raises_on_missing_head_sha():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha=None,
        url="https://github.com/OWNER/REPO/pull/77",
    )
    with pytest.raises(AgentLoopError, match="PR head SHA is unavailable"):
        require_pr_metadata_for_handoff(metadata)


def test_require_pr_metadata_for_handoff_returns_tuple_when_present():
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Title",
        head_branch="feature",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    assert require_pr_metadata_for_handoff(metadata) == (
        "https://github.com/OWNER/REPO/pull/77",
        "abc123",
    )


def test_provenance_parser_collapses_identical_claims_across_commits():
    scope = IssuePrProvenanceScope("OWNER/REPO", 56, "direct")
    messages = [
        f"first\n\n{format_issue_pr_provenance(scope)}",
        f"second\n\n{format_issue_pr_provenance(scope)}",
    ]

    claims = parse_issue_pr_provenance_messages(messages)

    assert compare_issue_pr_provenance(claims, expected=scope) == scope


def test_legacy_recovery_requires_complete_matching_commit_provenance(tmp_path):
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(),
    )
    config = make_config(tmp_path)

    found = find_open_pr_closing_issue(runner, config=config, issue_number=56)

    assert found is not None
    assert found.pr_number == 77


@pytest.mark.parametrize(
    ("pages", "provenance_reason"),
    [
        ([_commit_page([]), _commit_page([])], "is missing"),
        (
            [
                _commit_page(
                    ["Change\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct\n"
                     "Agent-Issue-Provenance: malformed"]
                ),
                _commit_page(["ignored"]),
            ],
            "is malformed",
        ),
        (
            [
                _commit_page(
                    [
                        "Change\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=direct",
                        "Other\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=57 flow=direct",
                    ],
                    total=2,
                ),
                _commit_page(["ignored"], total=2),
            ],
            "contains conflicting claims",
        ),
    ],
)
def test_invalid_single_candidate_recovery_includes_both_safe_remedies(
    tmp_path, pages, provenance_reason
):
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=pages,
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError) as exc_info:
        find_open_pr_closing_issue(runner, config=config, issue_number=56)

    message = str(exc_info.value)
    assert provenance_reason in message
    assert "unavailable" not in message
    assert "Remove the closing reference" in message
    assert "close the unrelated PR" in message
    assert "agent-loop pr 77" in message


@pytest.mark.parametrize(
    ("claimed_scope", "expected_scope"),
    [
        (
            IssuePrProvenanceScope("OTHER/REPO", 56, "direct"),
            IssuePrProvenanceScope("OWNER/REPO", 56, "direct"),
        ),
        (
            IssuePrProvenanceScope("OWNER/REPO", 57, "direct"),
            IssuePrProvenanceScope("OWNER/REPO", 56, "direct"),
        ),
        (
            IssuePrProvenanceScope("OWNER/REPO", 56, "direct"),
            IssuePrProvenanceScope("OWNER/REPO", 56, "approved", "current-plan"),
        ),
        (
            IssuePrProvenanceScope("OWNER/REPO", 56, "approved", "current-plan"),
            IssuePrProvenanceScope("OWNER/REPO", 56, "direct"),
        ),
        (
            IssuePrProvenanceScope("OWNER/REPO", 56, "approved", "old-plan"),
            IssuePrProvenanceScope("OWNER/REPO", 56, "approved", "current-plan"),
        ),
    ],
    ids=["repository", "issue", "direct-to-approved", "approved-to-direct", "plan-hash"],
)
def test_single_candidate_recovery_rejects_one_well_formed_wrong_scope(
    tmp_path, claimed_scope, expected_scope
):
    page = _commit_page([format_issue_pr_provenance(claimed_scope)])
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=[page, page],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="does not match the expected scope") as exc_info:
        find_open_pr_closing_issue(
            runner,
            config=config,
            issue_number=56,
            expected_scope=expected_scope,
        )

    message = str(exc_info.value)
    assert "unavailable" not in message
    assert "Remove the closing reference" in message
    assert "close the unrelated PR" in message
    assert "agent-loop pr 77" in message


def test_single_candidate_recovery_without_reconstructable_scope_gives_safe_remedies(tmp_path):
    runner = FakeRunner(open_prs_payload=[{"number": 77, "body": "Fixes #56"}])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="reconstructable approved-plan scope\\. Remove") as exc_info:
        find_open_pr_closing_issue(
            runner,
            config=config,
            issue_number=56,
            expected_scope=None,
        )

    message = str(exc_info.value)
    assert "close the unrelated PR" in message
    assert "agent-loop pr 77" in message
    assert runner.pr_commit_calls == 0


def test_single_candidate_recovery_distinguishes_commit_query_unavailability(tmp_path, monkeypatch):
    # A transient transport failure is retried by the read policy, so the query
    # is unavailable only once the whole retry budget is exhausted.
    monkeypatch.setattr("coding_review_agent_loop.github_retry._sleep", lambda _seconds: None)
    runner = FakeRunner(
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_query_failures=["connection reset"] * 3,
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="commit provenance is unavailable.*query failed") as exc_info:
        find_open_pr_closing_issue(runner, config=config, issue_number=56)

    message = str(exc_info.value)
    assert "Remove the closing reference" in message
    assert "close the unrelated PR" in message
    assert "agent-loop pr 77" in message


@pytest.mark.parametrize(
    "changed_page",
    [
        _commit_page(["second"], head="def456", total=2, oid_start=2),
        _commit_page(["second"], total=3, oid_start=2),
    ],
    ids=["head", "total"],
)
def test_commit_connection_rejects_history_change_during_pagination(tmp_path, changed_page):
    first_page = _commit_page(["first"], total=2, has_next=True, cursor="cursor-1")
    runner = FakeRunner(pr_commit_pages=[first_page, changed_page])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="history changed during provenance scan"):
        read_pull_request_commit_metadata(runner, config=config, pr_number=77)


@pytest.mark.parametrize(
    "final_page",
    [
        _commit_page(["first"], head="def456"),
        _commit_page(["first"], total=2),
    ],
    ids=["head", "total"],
)
def test_commit_connection_rejects_history_change_after_pagination(tmp_path, final_page):
    first_page = _commit_page(["first"])
    runner = FakeRunner(pr_commit_pages=[first_page, final_page])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="history changed during provenance scan"):
        read_pull_request_commit_metadata(runner, config=config, pr_number=77)


@pytest.mark.parametrize(
    "pages",
    [
        [
            _commit_page(["first"], total=2, has_next=True, cursor="cursor-1"),
            _commit_page(["second"], total=2, has_next=True, cursor="cursor-1", oid_start=2),
        ],
        [
            _commit_page(["first"], total=4, has_next=True, cursor="cursor-1"),
            _commit_page(["second"], total=4, has_next=True, cursor="cursor-2", oid_start=2),
            _commit_page(["third"], total=4, has_next=True, cursor="cursor-1", oid_start=3),
        ],
    ],
    ids=["non-advancing", "repeated"],
)
def test_commit_connection_rejects_non_advancing_or_repeated_cursors(tmp_path, pages):
    runner = FakeRunner(pr_commit_pages=pages)
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="pagination.*did not advance"):
        read_pull_request_commit_metadata(runner, config=config, pr_number=77)


def test_commit_connection_sends_string_graphql_variables_with_raw_fields(tmp_path):
    first_page = _commit_page(["first"], total=2, has_next=True, cursor="cursor-1")
    second_page = _commit_page(["second"], total=2, oid_start=2)
    final_page = _commit_page([], total=2)
    runner = FakeRunner(pr_commit_pages=[first_page, second_page, final_page])
    config = make_config(tmp_path, repo="2048/null")

    read_pull_request_commit_metadata(runner, config=config, pr_number=77)

    graphql_commands = [cmd for cmd, _cwd in runner.commands if cmd[:3] == ["gh", "api", "graphql"]]
    first_command = graphql_commands[0]
    owner_index = first_command.index("owner=2048")
    number_index = first_command.index("number=77")
    assert first_command[owner_index - 1 : number_index + 1] == [
        "-f",
        "owner=2048",
        "-f",
        "name=null",
        "-F",
        "number=77",
    ]
    after_index = graphql_commands[1].index("after=cursor-1")
    assert graphql_commands[1][after_index - 1 : after_index + 1] == ["-f", "after=cursor-1"]


def test_commit_connection_rejects_count_truncation(tmp_path):
    scope = IssuePrProvenanceScope("OWNER/REPO", 56, "direct")
    page = _commit_page([format_issue_pr_provenance(scope)], total=2)
    runner = FakeRunner(pr_commit_pages=[page, page])
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="truncated"):
        read_pull_request_commit_metadata(runner, config=config, pr_number=77)


# ---------------------------------------------------------------------------
# Closing-contract lineage base across same-PR plan replacements (#936)
# ---------------------------------------------------------------------------

from coding_review_agent_loop.expected_closure import contract_hash as _m936_contract_hash  # noqa: E402
from coding_review_agent_loop.github import IssueContext as _M936IssueContext  # noqa: E402
from coding_review_agent_loop.issue_pr_handoff import (  # noqa: E402
    authenticate_canonical_issue_pr,
    resolve_issue_pr_handoff_lineage,
)
from coding_review_agent_loop.pr_contract import (  # noqa: E402
    format_pr_contract_comment,
    make_pr_contract,
)

_M936_URL = "https://github.com/OWNER/REPO/pull/77"


def _m936_handoff(plan_hash, *, ids=(56,), supersedes=None):
    return _comment(format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url=_M936_URL, pr_head_sha="abc123",
        flow="approved-plan-implementation", plan_hash=plan_hash,
        expected_closing_issue_ids=ids, supersedes_hash=supersedes,
    ))


def _m936_replacement(plan_hash, *, ids=(56,)):
    # Unchanged-ID replacements all carry the same closing-ID digest.
    return _m936_handoff(plan_hash, ids=ids, supersedes=_m936_contract_hash(ids))


def _m936_authenticate(tmp_path, comments, *, pr_ids=(56,), pr_supersedes=None):
    contract = make_pr_contract(
        repository="OWNER/REPO", pr_number=77, origin_flow="approved-plan-implementation",
        primary_issue_number=56, expected_closing_issue_ids=pr_ids,
        supersedes_hash=pr_supersedes,
    )
    runner = FakeRunner(pr_payload={
        "number": 77, "body": "Fixes #56", "url": _M936_URL,
        "comments": [{
            "author": {"login": "bot"}, "createdAt": "2026-09-20T00:00:00Z",
            "body": format_pr_contract_comment(contract),
        }],
    })
    issue = _M936IssueContext(
        number=56, repo="OWNER/REPO", title="Child", body="", url="", comments=tuple(comments)
    )
    return authenticate_canonical_issue_pr(
        runner, config=make_config(tmp_path), issue_number=56, issue_context=issue
    )


@pytest.mark.parametrize("replacements", [1, 2])
def test_m936_pr_contract_authenticates_against_the_lineage_base(tmp_path, replacements):
    comments = [_m936_handoff("plan-a")] + [
        _m936_replacement(f"plan-{index}") for index in range(replacements)
    ]
    lineage = resolve_issue_pr_handoff_lineage(comments, issue_number=56, repo="OWNER/REPO")
    assert lineage.latest.plan_hash == f"plan-{replacements - 1}"
    assert lineage.closing_base.plan_hash == "plan-a"
    assert lineage.closing_base.supersedes_hash is None
    assert lineage.replaced.plan_hash == ("plan-a" if replacements == 1 else "plan-0")
    assert lineage.latest_comment_index == replacements
    # The unchanged PR-side contract (no supersession) still authenticates,
    # and the plan hash comes from the latest record.
    authenticated = _m936_authenticate(tmp_path, comments)
    assert authenticated.record.plan_hash == f"plan-{replacements - 1}"
    assert authenticated.state == "OPEN"


def test_m936_closing_superset_before_a_replacement_keeps_superset_rules(tmp_path):
    comments = [
        _m936_handoff("plan-a"),
        _m936_handoff("plan-a", ids=(56, 60), supersedes=_m936_contract_hash((56,))),
        _m936_replacement("plan-b", ids=(56, 60)),
    ]
    lineage = resolve_issue_pr_handoff_lineage(comments, issue_number=56, repo="OWNER/REPO")
    assert lineage.latest.plan_hash == "plan-b"
    assert lineage.closing_base.expected_closing_issue_ids == (56, 60)
    assert lineage.closing_base.supersedes_hash == _m936_contract_hash((56,))
    authenticated = _m936_authenticate(
        tmp_path, comments, pr_ids=(56, 60), pr_supersedes=_m936_contract_hash((56,))
    )
    assert authenticated.record.plan_hash == "plan-b"
    # The PR-side contract must still match the superset base, not the old one.
    with pytest.raises(AgentLoopError, match="diverge"):
        _m936_authenticate(tmp_path, comments)


def test_m936_closing_superset_after_a_replacement_becomes_the_new_base(tmp_path):
    comments = [
        _m936_handoff("plan-a"),
        _m936_replacement("plan-b"),
        _m936_handoff("plan-b", ids=(56, 60), supersedes=_m936_contract_hash((56,))),
    ]
    lineage = resolve_issue_pr_handoff_lineage(comments, issue_number=56, repo="OWNER/REPO")
    assert lineage.closing_base is lineage.latest
    # The superset moves the closing base but never erases the plan-changing
    # edge, so the rebind stays verifiable.
    assert lineage.replaced.plan_hash == "plan-a"
    assert lineage.replacement.plan_hash == "plan-b"
    assert (lineage.replacement_comment_index, lineage.latest_comment_index) == (1, 2)
    authenticated = _m936_authenticate(
        tmp_path, comments, pr_ids=(56, 60), pr_supersedes=_m936_contract_hash((56,))
    )
    assert authenticated.record.plan_hash == "plan-b"


def test_m936_unannotated_divergent_plan_record_still_raises():
    with pytest.raises(AgentLoopError, match="Divergent AGENT_ISSUE_PR_HANDOFF"):
        resolve_issue_pr_handoff_lineage(
            [_m936_handoff("plan-a"), _m936_handoff("plan-b")],
            issue_number=56, repo="OWNER/REPO",
        )
    # A closing-ID digest that does not name the replaced contract is no annotation.
    with pytest.raises(AgentLoopError, match="Divergent AGENT_ISSUE_PR_HANDOFF"):
        find_latest_issue_pr_handoff(
            [_m936_handoff("plan-a"), _m936_handoff("plan-b", supersedes="0" * 64)],
            issue_number=56, repo="OWNER/REPO",
        )


def test_m936_plan_edge_tracks_the_latest_change_and_resets_for_another_pr():
    superset_with_new_plan = _m936_handoff(
        "plan-c", ids=(56, 60), supersedes=_m936_contract_hash((56,))
    )
    lineage = resolve_issue_pr_handoff_lineage(
        [_m936_handoff("plan-a"), _m936_replacement("plan-b"), superset_with_new_plan],
        issue_number=56, repo="OWNER/REPO",
    )
    # A superset that also changes the plan is itself the latest plan-changing edge.
    assert (lineage.replaced.plan_hash, lineage.replacement.plan_hash) == ("plan-b", "plan-c")
    assert lineage.replacement_comment_index == 2
    plain = resolve_issue_pr_handoff_lineage(
        [_m936_handoff("plan-a")], issue_number=56, repo="OWNER/REPO"
    )
    assert plain.replaced is None and plain.replacement is None
    assert plain.replacement_comment_index == -1


# --- Version 2: transaction-bound handoff records (#827, stage A) -----------


def _v2_handoff(**overrides):
    from coding_review_agent_loop.expected_closure import contract_hash
    from coding_review_agent_loop.issue_pr_handoff import IssuePrHandoffMetadataV2

    fields = dict(
        issue_number=813,
        pr_number=826,
        pr_url="https://github.com/OWNER/REPO/pull/826",
        pr_head_sha="a" * 40,
        flow="issue-implementation",
        plan_hash=None,
        expected_closing_issue_ids=(813,),
        contract_hash=contract_hash((813,)),
        transaction_id="b" * 64,
    )
    fields.update(overrides)
    return IssuePrHandoffMetadataV2(**fields)


def test_v2_handoff_round_trips_and_renders_one_trusted_issue_comment_record():
    from coding_review_agent_loop.issue_pr_handoff import (
        decode_issue_pr_handoff_v2,
        encode_issue_pr_handoff_v2,
        format_issue_pr_handoff_v2_comment,
    )
    from coding_review_agent_loop.protocol_markers import ISSUE_COMMENT_SURFACE, TrustedBody

    metadata = _v2_handoff(flow="approved-plan-implementation", plan_hash="0123456789abcdef")
    assert decode_issue_pr_handoff_v2(encode_issue_pr_handoff_v2(metadata)) == metadata
    body = TrustedBody.canonical(
        format_issue_pr_handoff_v2_comment(metadata, repo="OWNER/REPO"),
        surface=ISSUE_COMMENT_SURFACE,
        expected_tokens=("AGENT_ISSUE_PR_HANDOFF",),
    )
    assert "b" * 64 in body and "Plan hash: 0123456789abcdef" in body


@pytest.mark.parametrize(
    "overrides",
    [
        {"transaction_id": "short"},
        {"transaction_id": None},
        {"flow": "direct-pr"},
        {"plan_hash": "0123456789abcdef"},
        {"flow": "approved-plan-implementation"},
        {"expected_closing_issue_ids": (900,)},
        {"contract_hash": "0" * 64},
        {"issue_number": True},
        {"pr_head_sha": ""},
    ],
)
def test_v2_handoff_codec_is_strict(overrides):
    from coding_review_agent_loop.issue_pr_handoff import (
        decode_issue_pr_handoff_v2,
        encode_issue_pr_handoff_v2,
    )

    with pytest.raises(AgentLoopError):
        decode_issue_pr_handoff_v2(encode_issue_pr_handoff_v2(_v2_handoff(**overrides)))


def test_existing_handoff_entry_point_still_rejects_a_v2_payload():
    from types import SimpleNamespace

    from coding_review_agent_loop.issue_pr_handoff import (
        AGENT_ISSUE_PR_HANDOFF_RE,
        decode_issue_pr_handoff_v2,
        format_issue_pr_handoff_v2_comment,
        resolve_issue_pr_handoff_lineage,
    )

    body = format_issue_pr_handoff_v2_comment(_v2_handoff(), repo="OWNER/REPO")
    with pytest.raises(AgentLoopError, match="unsupported schema_version 2"):
        resolve_issue_pr_handoff_lineage(
            [SimpleNamespace(body=body)], issue_number=813, repo="OWNER/REPO"
        )
    # And the version-2 decoder does not accept a version-1 record either.
    v1_body = format_issue_pr_handoff_comment(
        issue_number=813,
        pr_number=826,
        pr_url="https://github.com/OWNER/REPO/pull/826",
        pr_head_sha="a" * 40,
        flow="issue-implementation",
        plan_hash=None,
    )
    encoded = AGENT_ISSUE_PR_HANDOFF_RE.search(v1_body).group("payload")
    with pytest.raises(AgentLoopError, match="expected exactly"):
        decode_issue_pr_handoff_v2(encoded)


def test_envelope_only_handoff_lineage_resolves_v1_v2_inherited_and_inert_records():
    from dataclasses import replace

    from workflow_transaction_helpers import (
        FOREIGN,
        HEAD_2,
        ISSUE,
        PR,
        REPO,
        direct_intent,
        issue_view,
        pr_view,
        prepared_comment,
        record_set,
        terminal_comment,
        v1_handoff_comment,
        v2_handoff_comment,
    )

    from coding_review_agent_loop.errors import WorkflowTransactionError
    from coding_review_agent_loop.issue_pr_handoff import issue_pr_handoff_record_hash
    from coding_review_agent_loop.workflow_transaction import (
        ENTRY_HANDOFF,
        ENTRY_INITIAL_CODER_ROUND,
        ENTRY_PR_CONTRACT,
        ERA_LEGACY,
        ERA_TRANSACTION,
        KIND_HEAD_ADVANCE,
        PHASE_ABORTED,
        CommentRef,
        derive_handoff_metadata,
        handoff_candidate_pr_numbers,
        inherited,
        not_applicable,
        resolve_handoff_lineage,
        resolve_transaction_lineage,
    )

    def resolve(pr_comments, issue_comments):
        lineage = resolve_transaction_lineage(
            pr_view(*pr_comments), repository=REPO, pr_number=PR
        )
        return resolve_handoff_lineage(
            issue_view(*issue_comments), lineage, repository=REPO, issue_number=ISSUE
        )

    # Legacy: exactly the version-1 result.
    legacy = resolve([], [v1_handoff_comment(5)])
    assert legacy.era == ERA_LEGACY and legacy.comment_id == 5
    assert legacy.handoff == legacy.v1_lineage.latest
    assert resolve([], []) is None

    intent = direct_intent()
    published = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
    prepared = prepared_comment(10, intent)
    committed = terminal_comment(20, intent, prepared_id=10, published=published)
    handoff = v2_handoff_comment(11, intent)

    # Prepared only: the published handoff grants nothing yet.
    pending = resolve([prepared], [handoff])
    assert pending.handoff is None and pending.era == ERA_TRANSACTION
    assert pending.pending_transaction_id == intent.transaction_id

    resolved = resolve([prepared, committed], [handoff])
    assert resolved.handoff == derive_handoff_metadata(intent)
    assert resolved.transaction_id == intent.transaction_id and resolved.comment_id == 11
    assert handoff_candidate_pr_numbers(issue_view(handoff), issue_number=ISSUE) == (PR,)

    # A byte-exact forged copy from another author is never counted.
    forged = replace(handoff, author_login=FOREIGN[0], author_id=FOREIGN[1])
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve([prepared, committed], [forged])
    assert excinfo.value.code == "record-missing"

    # Records bound to an aborted transaction are inert.
    aborted = terminal_comment(
        20, intent, prepared_id=10, phase=PHASE_ABORTED, abort_reason="stale-head",
        published={ENTRY_HANDOFF: 11},
    )
    inert = resolve([prepared, aborted], [handoff])
    assert inert.handoff is None and inert.pending_transaction_id is None

    # Transaction records deleted or hidden: fail closed, never the legacy path.
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve([], [handoff])
    assert excinfo.value.code == "transaction-record-missing"

    # A head-advance successor inherits the handoff by authenticated reference.
    digest = issue_pr_handoff_record_hash(derive_handoff_metadata(intent))

    def advance(handoff_digest):
        return replace(
            intent,
            head_sha=HEAD_2,
            successor_kind=KIND_HEAD_ADVANCE,
            predecessor_transaction_id=intent.transaction_id,
            record_set=record_set(
                handoff=inherited(ENTRY_HANDOFF, CommentRef(f"issue#{ISSUE}", 11, handoff_digest)),
                contract=inherited(ENTRY_PR_CONTRACT, CommentRef(f"pr#{PR}", 12, "d" * 64)),
                coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND),
            ),
        )

    good = advance(digest)
    through_chain = resolve(
        [prepared, committed, prepared_comment(30, good), terminal_comment(40, good, prepared_id=30)],
        [handoff],
    )
    assert through_chain.transaction_id == good.transaction_id
    assert through_chain.handoff.pr_head_sha == intent.head_sha
    bad = advance("0" * 64)
    with pytest.raises(WorkflowTransactionError) as excinfo:
        resolve(
            [prepared, committed, prepared_comment(30, bad), terminal_comment(40, bad, prepared_id=30)],
            [handoff],
        )
    assert excinfo.value.code == "inherited-mismatch"


# --- review round 1: strict and canonical v2 handoff wire form ------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"pr_head_sha": "x"},
        {"pr_head_sha": "A" * 40},
        {"pr_head_sha": "a" * 39},
        {"pr_url": "https://example.com/OWNER/REPO/pull/826"},
        {"pr_url": "http://github.com/OWNER/REPO/pull/826"},
        {"pr_url": "https://github.com/OWNER/REPO/pull/827"},
        {"pr_url": "https://github.com/OWNER/REPO/pull/826?x=1"},
        {"pr_url": "https://github.com/OWNER/REPO/pull/826/files"},
        {"flow": "approved-plan-implementation", "plan_hash": "latest"},
        {"flow": "approved-plan-implementation", "plan_hash": "0123456789ABCDEF"},
        {"flow": "approved-plan-implementation", "plan_hash": "0123456789abcde"},
    ],
)
def test_v2_handoff_codec_rejects_semantically_malformed_fields(overrides):
    from coding_review_agent_loop.issue_pr_handoff import (
        decode_issue_pr_handoff_v2,
        encode_issue_pr_handoff_v2,
    )

    with pytest.raises(AgentLoopError):
        decode_issue_pr_handoff_v2(encode_issue_pr_handoff_v2(_v2_handoff(**overrides)))


def test_v2_handoff_decoder_and_lineage_reject_a_noncanonical_wire_record():
    import json

    from workflow_transaction_helpers import (
        ISSUE,
        PR,
        REPO,
        comment,
        direct_intent,
        issue_view,
        pr_view,
        prepared_comment,
        terminal_comment,
        v2_handoff_comment,
    )

    from coding_review_agent_loop.issue_pr_handoff import (
        decode_issue_pr_handoff_v2,
        encode_issue_pr_handoff_v2,
    )
    from coding_review_agent_loop.workflow_transaction import (
        ENTRY_HANDOFF,
        ENTRY_INITIAL_CODER_ROUND,
        ENTRY_PR_CONTRACT,
        derive_handoff_metadata,
        resolve_handoff_lineage,
        resolve_transaction_lineage,
    )

    def reencode(encoded, **kwargs):
        value = json.loads(base64.urlsafe_b64decode(encoded.encode()).decode())
        return base64.urlsafe_b64encode(json.dumps(value, **kwargs).encode()).decode()

    canonical = encode_issue_pr_handoff_v2(_v2_handoff())
    assert decode_issue_pr_handoff_v2(canonical) == _v2_handoff()
    for noncanonical in (
        reencode(canonical, sort_keys=True),
        reencode(canonical, separators=(",", ":"), sort_keys=False),
    ):
        if noncanonical == canonical:
            continue
        with pytest.raises(AgentLoopError, match="not canonically encoded"):
            decode_issue_pr_handoff_v2(noncanonical)

    intent = direct_intent()
    encoded = encode_issue_pr_handoff_v2(derive_handoff_metadata(intent))
    handoff = v2_handoff_comment(11, intent)
    spaced = comment(
        11, handoff.body.replace(encoded, reencode(encoded, sort_keys=True)),
        surface=f"issue#{ISSUE}",
    )
    assert spaced.body != handoff.body
    published = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}
    lineage = resolve_transaction_lineage(
        pr_view(
            prepared_comment(10, intent),
            terminal_comment(20, intent, prepared_id=10, published=published),
        ),
        repository=REPO,
        pr_number=PR,
    )
    with pytest.raises(AgentLoopError, match="not canonically encoded"):
        resolve_handoff_lineage(
            issue_view(spaced), lineage, repository=REPO, issue_number=ISSUE
        )


def test_handoff_must_be_published_between_the_bound_prepared_and_terminal_records():
    from workflow_transaction_helpers import (
        ISSUE,
        PR,
        REPO,
        comment,
        direct_intent,
        issue_view,
        pr_view,
        prepared_comment,
        terminal_comment,
        v2_handoff_comment,
    )

    from coding_review_agent_loop.errors import WorkflowTransactionError
    from coding_review_agent_loop.workflow_transaction import (
        ENTRY_HANDOFF,
        ENTRY_INITIAL_CODER_ROUND,
        ENTRY_PR_CONTRACT,
        resolve_handoff_lineage,
        resolve_transaction_lineage,
    )

    intent = direct_intent()

    def resolve(handoff, *, prepared=10, terminal=20):
        published = {
            ENTRY_HANDOFF: handoff[0], ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13
        }
        lineage = resolve_transaction_lineage(
            pr_view(
                comment(10, prepared_comment(10, intent).body, second=prepared),
                comment(
                    20,
                    terminal_comment(20, intent, prepared_id=10, published=published).body,
                    second=terminal,
                ),
            ),
            repository=REPO,
            pr_number=PR,
        )
        record = comment(
            handoff[0],
            v2_handoff_comment(handoff[0], intent).body,
            surface=f"issue#{ISSUE}",
            second=handoff[1],
        )
        return resolve_handoff_lineage(
            issue_view(record), lineage, repository=REPO, issue_number=ISSUE
        )

    # Same-second publication ordered by comment ID is accepted.
    assert resolve((11, 7), prepared=7, terminal=7).comment_id == 11
    for handoff in ((9, 9), (25, 25), (11, 3), (11, 50)):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            resolve(handoff)
        assert excinfo.value.code == "record-unordered"


def test_v2_handoff_may_only_restate_or_widen_an_authenticated_v1_handoff():
    from workflow_transaction_helpers import (
        HEAD_1,
        ISSUE,
        PLAN_HASH,
        PR,
        REPO,
        direct_intent,
        issue_view,
        plan_intent,
        plan_record_comment,
        pr_review_comment,
        pr_view,
        prepared_comment,
        record_set,
        terminal_comment,
        v1_contract,
        v1_contract_comment,
        v1_handoff_comment,
        v2_handoff_comment,
    )

    from coding_review_agent_loop.errors import WorkflowTransactionError
    from coding_review_agent_loop.issue_pr_handoff import issue_pr_handoff_record_hash
    from coding_review_agent_loop.pr_contract import pr_contract_record_hash
    from coding_review_agent_loop.workflow_transaction import (
        ENTRY_HANDOFF,
        ENTRY_INITIAL_CODER_ROUND,
        ENTRY_PR_CONTRACT,
        FLOW_APPROVED_PLAN,
        KIND_LEGACY_ROOT_CORRECTION,
        CommentRef,
        LegacyRoot,
        LegacyRootContext,
        find_origin_evidence,
        not_applicable,
        resolve_handoff_lineage,
        resolve_transaction_lineage,
    )

    published = {ENTRY_HANDOFF: 11, ENTRY_PR_CONTRACT: 12, ENTRY_INITIAL_CODER_ROUND: 13}

    def resolve(intent, v1, *, pr_extra=(), context=None, published=published):
        prs = pr_view(
            *pr_extra,
            prepared_comment(110, intent),
            terminal_comment(120, intent, prepared_id=110, published=published),
        )
        lineage = resolve_transaction_lineage(
            prs, repository=REPO, pr_number=PR, legacy_root_context=context
        )
        issues = issue_view(v1, v2_handoff_comment(111, intent))
        return resolve_handoff_lineage(issues, lineage, repository=REPO, issue_number=ISSUE)

    new_ids = {ENTRY_HANDOFF: 111, ENTRY_PR_CONTRACT: 112, ENTRY_INITIAL_CODER_ROUND: 113}

    # A consistent version-1 handoff is restated by the initial transaction.
    intent = direct_intent()
    resolved = resolve(intent, v1_handoff_comment(5), published=new_ids)
    assert resolved.comment_id == 111 and resolved.transaction_id == intent.transaction_id
    # ... and may be widened to a strict superset of its closing IDs.
    widened = direct_intent(expected_closing_issue_ids=(ISSUE, 900))
    assert resolve(widened, v1_handoff_comment(5), published=new_ids).comment_id == 111

    def refused(intent, v1):
        with pytest.raises(WorkflowTransactionError) as excinfo:
            resolve(intent, v1, published=new_ids)
        assert excinfo.value.code == "handoff-supersession-invalid"
        return str(excinfo.value)

    # A different flow (and with it the plan hash) is never silently relabelled.
    message = refused(plan_intent(), v1_handoff_comment(5))
    assert "comment 111" in message and "version-1 handoff comment 5" in message
    refused(direct_intent(), v1_handoff_comment(5, flow=FLOW_APPROVED_PLAN, plan_hash=PLAN_HASH))
    # Same flow, another plan.
    refused(
        plan_intent(approved_plan_hash="1" * 16),
        v1_handoff_comment(5, flow=FLOW_APPROVED_PLAN, plan_hash=PLAN_HASH),
    )
    # Closing IDs that are not a superset of the version-1 record's.
    refused(direct_intent(), v1_handoff_comment(5, ids=(ISSUE, 900)))
    refused(
        direct_intent(expected_closing_issue_ids=(ISSUE, 901)),
        v1_handoff_comment(5, ids=(ISSUE, 900)),
    )

    # A legacy-root correction naming exactly that version-1 handoff may correct it.
    v1 = v1_handoff_comment(5)
    contract_comment, evidence_comment = v1_contract_comment(105), pr_review_comment(107)
    plan_issue = issue_view(plan_record_comment(3), v1)
    v1_lineage = resolve_issue_pr_handoff_lineage([v1], issue_number=ISSUE, repo=REPO)

    def correction(handoff_ref):
        return plan_intent(
            successor_kind=KIND_LEGACY_ROOT_CORRECTION,
            legacy_root=LegacyRoot(
                contract=CommentRef(f"pr#{PR}", 105, pr_contract_record_hash(v1_contract())),
                handoff=handoff_ref,
                origin_evidence=find_origin_evidence(pr_view(contract_comment, evidence_comment)),
            ),
            record_set=record_set(coder_round=not_applicable(ENTRY_INITIAL_CODER_ROUND)),
        )

    named = correction(
        CommentRef(f"issue#{ISSUE}", 5, issue_pr_handoff_record_hash(v1_lineage.latest))
    )
    context = LegacyRootContext(plan_issue, (HEAD_1,), primary_issue_view=plan_issue)
    corrected = resolve(
        named,
        v1,
        pr_extra=(contract_comment, evidence_comment),
        context=context,
        published={ENTRY_HANDOFF: 111, ENTRY_PR_CONTRACT: 112},
    )
    assert corrected.comment_id == 111 and corrected.handoff.flow == FLOW_APPROVED_PLAN


# --- Review round 1 of #1378 ----------------------------------------------------


def _real_staged_plan_comments(tmp_path, *, gemini_state="approved"):
    """A real primary-then-panel plan approved by board A (Codex primary, Gemini panel)."""
    from agent_loop_helpers import structured_plan_review
    from coding_review_agent_loop.cli import run_issue_loop
    from test_child_plan_provenance import (
        _ChildPlanningRunner, _child_plan_state, _child_row, _plan_config,
    )

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    runner = _ChildPlanningRunner(
        claude_outputs=[_child_plan_state(_child_row())],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state=gemini_state, reviewer="Google Gemini")],
    )
    config = _plan_config(
        plan_dir, reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex",
    )
    try:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    except AgentLoopError:
        if gemini_state == "approved":
            raise
    return list(runner.issue_comments)


def _strict_managed_pr_run(tmp_path, monkeypatch, issue_comments, *, entry, pr_outputs):
    """A strictly protected managed PR resumed by board B, driven to the merge gate."""
    import coding_review_agent_loop.orchestrator as orchestrator
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.managed_ci import (
        AuthenticatedIssueCreatedHandoff, ManagedCiContract, ManagedCiOutcome,
    )

    handoff = AuthenticatedIssueCreatedHandoff(
        pr_number=77, issue_number=56, repository="OWNER/REPO", base_ref="main",
        head_sha="abc123", branch="agent-loop/managed-56", trusted_actor_login="agent-loop",
        trusted_actor_id=1, protection_mode="strict", override_nonce=None,
    )
    monkeypatch.setattr(orchestrator, "recover_issue_created_handoff", lambda *a, **k: handoff)
    monkeypatch.setattr(orchestrator, "authorize_fresh_issue_created_resume", lambda *a, **k: handoff)
    monkeypatch.setattr(orchestrator, "revalidate_issue_created_handoff", lambda *a, **k: handoff)
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *a, **k: ManagedCiContract())
    monkeypatch.setattr(orchestrator, "merge_pr", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator, "wait_for_final_qualification",
        lambda *a, **k: ManagedCiOutcome(status="passed", head_sha="abc123"),
    )
    monkeypatch.setattr(orchestrator, "prepare_v2_merge", lambda *a, **k: None)
    merges = []
    monkeypatch.setattr(
        orchestrator, "_merge_with_exact_head_proof", lambda *a, **k: merges.append(k),
    )
    runner = FakeRunner(
        issue_comments=issue_comments,
        issue_payload={"number": 56, "title": "Linked issue", "body": "Scope."},
        pr_payload={"body": "Fixes #56", "headRefName": "agent-loop/managed-56"},
        **pr_outputs,
    )
    fresh = (
        {"managed_ci_fresh_authorization": True, "managed_ci_issue_number": 56}
        if entry == "fresh" else {}
    )
    config = make_config(
        tmp_path, reviewer=("claude", "antigravity"), managed_ci=True,
        managed_ci_trusted_actor="agent-loop", auto_merge=True, pre_review_tests=False,
        **fresh,
    )
    return runner, config, merges, run_pr_loop


def _board_b_outputs(config=None):
    return {
        "claude_outputs": [structured_board_review("Anthropic Claude")],
        "antigravity_outputs": [structured_board_review("Google Antigravity")],
    }


def structured_board_review(reviewer):
    from agent_loop_helpers import structured_pr_review

    return structured_pr_review(reviewer=reviewer)


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_staged_plan_by_board_a_qualifies_strict_pr_reviewed_by_board_b(
    tmp_path, monkeypatch, entry
):
    """handoff-plan-verification-recovery and handoff-plan-binding-preserved, to the merge."""
    import coding_review_agent_loop.orchestrator as orchestrator

    plan_comments = _real_staged_plan_comments(tmp_path)
    runner, config, merges, run_pr_loop = _strict_managed_pr_run(
        tmp_path, monkeypatch, plan_comments, entry=entry, pr_outputs=_board_b_outputs(),
    )
    real_verify = orchestrator._verify_strict_managed_plan_binding
    strict_checks = []

    def spy(**kwargs):
        strict_checks.append(tuple(kwargs["config"].reviewer))
        return real_verify(**kwargs)

    monkeypatch.setattr(orchestrator, "_verify_strict_managed_plan_binding", spy)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    # The real strict binding ran at qualification with the PR config; the
    # plan board came from issue history, so board B never verified the plan.
    assert strict_checks and set(strict_checks) == {("claude", "antigravity")}
    # Board B reviewed the exact head; the plan board never ran on the PR.
    agents = [command[0] for command, _cwd in runner.commands if command[0] in _AGENT_COMMANDS]
    assert sorted(agents) == ["agy", "claude"]
    reviews = [comment for comment in runner.comments if "**Review verdict:**" in comment]
    assert len(reviews) == 2
    # Strict qualification re-verified the board A plan and merged the exact head once.
    assert [merge["proof"].head_sha for merge in merges] == ["abc123"]


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_strict_pr_without_board_b_exact_head_approval_does_not_merge(
    tmp_path, monkeypatch, entry
):
    plan_comments = _real_staged_plan_comments(tmp_path)
    runner, config, merges, run_pr_loop = _strict_managed_pr_run(
        tmp_path, monkeypatch, plan_comments, entry=entry,
        pr_outputs={"claude_outputs": [structured_board_review("Anthropic Claude")]},
    )
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert merges == []


@pytest.mark.parametrize("entry", ["ordinary", "fresh"])
def test_strict_recovery_refuses_a_plan_board_a_did_not_fully_approve(
    tmp_path, monkeypatch, entry
):
    plan_comments = _real_staged_plan_comments(tmp_path, gemini_state="blocking")
    runner, config, merges, run_pr_loop = _strict_managed_pr_run(
        tmp_path, monkeypatch, plan_comments, entry=entry, pr_outputs=_board_b_outputs(),
    )
    with pytest.raises(AgentLoopError, match="canonical"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert merges == []
    assert _agent_commands(runner) == []


def test_strict_qualification_refuses_a_changed_plan_identity(tmp_path, monkeypatch):
    """A stale or mismatched plan identity is refused at the final gate."""
    import coding_review_agent_loop.orchestrator as orchestrator

    plan_comments = _real_staged_plan_comments(tmp_path)
    runner, config, merges, run_pr_loop = _strict_managed_pr_run(
        tmp_path, monkeypatch, plan_comments, entry="ordinary", pr_outputs=_board_b_outputs(),
    )
    real_verify = orchestrator._verify_strict_managed_plan_binding

    def stale_identity(**kwargs):
        return real_verify(**{**kwargs, "expected_plan_hash": "0" * 16})

    monkeypatch.setattr(orchestrator, "_verify_strict_managed_plan_binding", stale_identity)
    with pytest.raises(AgentLoopError, match="canonical approved plan changed"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert merges == []


def _bound_plan_history(tmp_path, *, mutate):
    """A named all-reviewers plan approved in round 1, with one history defect."""
    from types import SimpleNamespace
    from coding_review_agent_loop.plan_review_scheduling import make_plan_contract
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata, _attach_round_metadata, _plan_subject,
    )

    seats = (
        _seat(tmp_path, "a", "codex", "gpt-a", "medium"),
        _seat(tmp_path, "b", "claude", "claude-b", "medium"),
    )
    binding = reviewer_seat_binding(_named_config(tmp_path, seats))
    plan = "Approved bound plan.\n\n### Plan steps\n1. Keep bindings honest."
    subject = _plan_subject(plan)
    records = [
        dict(role="coder", agent="Claude", canonical_plan=plan, raw_structured_coder_response=plan),
        dict(role="summary", agent="Orchestrator", scheduler_contract=make_plan_contract(
            ("a", "b"), "all-reviewers", None).as_dict()),
        dict(role="reviewer", agent="a", state="approved"),
        dict(role="reviewer", agent="b", state="approved"),
    ]
    for record in records:
        record.setdefault("seat_binding", binding)
    mutate(records, binding)
    comments = [
        SimpleNamespace(body=_attach_round_metadata("Round record.", PostedRoundMetadata(
            flow="plan", round_number=1, subject=subject, **fields,
        )))
        for fields in records
    ]
    return plan, comments


def _rebound(binding, seat_id, **changes):
    import copy

    result = copy.deepcopy(binding)
    for entry in result["seats"]:
        if entry["id"] == seat_id:
            entry.update(changes)
    return result


def _no_defect(records, binding):
    pass


def _approval_on_superseded_model(records, binding):
    # b approved on an older model; a later summary binds b to the current one.
    records[3]["seat_binding"] = _rebound(binding, "b", model_chain=["claude-b-old"])
    records.append(dict(role="summary", agent="Orchestrator", seat_binding=binding))


def _backend_changed(records, binding):
    records[2]["seat_binding"] = _rebound(binding, "a", backend="gemini")


def _unbound_record(records, binding):
    records[3]["seat_binding"] = None


def _board_changed(records, binding):
    import copy

    shrunk = copy.deepcopy(binding)
    shrunk["seats"] = [entry for entry in shrunk["seats"] if entry["id"] == "a"]
    records[2]["seat_binding"] = shrunk


def _reviewer_outside_binding(records, binding):
    records[3]["agent"] = "c"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_approval_on_superseded_model, "not completely approved"),
        (_backend_changed, "changed backend in plan history"),
        (_unbound_record, "mixes bound and unbound round records"),
        (_board_changed, "changed its recorded board"),
        (_reviewer_outside_binding, "not represented in its seat binding"),
    ],
    ids=["superseded-model", "backend", "unbound", "board", "outside-binding"],
)
def test_plan_seat_binding_history_is_validated_before_verifying_approval(
    tmp_path, mutate, message
):
    """Generalization of item-2: every plan binding defect fails closed."""
    from coding_review_agent_loop.decomposition import approved_plan_hash
    from coding_review_agent_loop.github import IssueContext as Issue
    from coding_review_agent_loop.pr_loop_support import _verify_strict_managed_plan_binding
    from types import SimpleNamespace

    def verify(comments, plan):
        _verify_strict_managed_plan_binding(
            config=make_config(tmp_path, reviewer=("gemini",)),
            pr_number=7,
            issue_context=Issue(959, "OWNER/REPO", "t", "b", None, tuple(
                IssueComment(author="agent-loop", created_at=None, body=c.body) for c in comments
            )),
            metadata=SimpleNamespace(head_branch="agent-loop/managed-959", head_sha="h"),
            expected_plan_hash=approved_plan_hash(plan),
        )

    plan, comments = _bound_plan_history(tmp_path, mutate=_no_defect)
    verify(comments, plan)
    plan, comments = _bound_plan_history(tmp_path, mutate=mutate)
    with pytest.raises(AgentLoopError, match=message):
        verify(comments, plan)


def test_malformed_staged_plan_history_is_never_downgraded_to_all_reviewers(tmp_path):
    """item-1: an invalid staged checkpoint with full-board approvals fails closed."""
    from types import SimpleNamespace
    from coding_review_agent_loop.decomposition import approved_plan_hash
    from coding_review_agent_loop.github import IssueContext as Issue
    from coding_review_agent_loop.plan_verification import plan_verification_inputs
    from coding_review_agent_loop.pr_loop_support import _verify_strict_managed_plan_binding
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_state import (
        PostedRoundMetadata, _attach_round_metadata, _extract_round_metadata_records,
        _plan_subject,
    )

    seats = (
        _seat(tmp_path, "primary", "codex", "gpt-a", "medium"),
        _seat(tmp_path, "panel", "claude", "claude-b", "medium"),
    )
    binding = reviewer_seat_binding(_named_config(
        tmp_path, seats, plan_review_policy="primary-then-panel", primary_plan_reviewer=seats[0],
    ))
    plan = "Staged plan.\n\n### Plan steps\n1. Verify the staged gate."
    subject = _plan_subject(plan)
    from coding_review_agent_loop.plan_review_scheduling import make_plan_contract

    # A staged checkpoint without its candidate key does not decode as a valid
    # scheduler record, so its persisted policy is hidden.
    invalid = _attach_round_metadata("Plan scheduling checkpoint.", PostedRoundMetadata(
        flow="plan", role="summary", agent="Orchestrator", round_number=1,
        subject=subject, seat_binding=binding, phase="scheduler-prelaunch",
        scheduler_contract=make_plan_contract(
            ("primary", "panel"), "primary-then-panel", "primary",
        ).as_dict(),
        scheduler_obligation_digest="0" * 16,
        scheduler_selected_reviewers=("primary",),
        scheduler_paused_reviewers=(("panel", "panel"),),
        scheduler_reasons=("primary gate",),
        scheduler_final_sweep=False,
        scheduler_force_full=False,
        scheduler_calls_avoided=0,
        scheduler_phase="primary",
        scheduler_primary_reviewer="primary",
    ))
    bodies = [
        _attach_round_metadata(plan, PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1, subject=subject,
            canonical_plan=plan, raw_structured_coder_response=plan, seat_binding=binding,
        )),
        invalid,
        *(
            _attach_round_metadata("Approved.", PostedRoundMetadata(
                flow="plan", role="reviewer", agent=name, round_number=1, subject=subject,
                state="approved", seat_binding=binding,
            ))
            for name in ("primary", "panel")
        ),
    ]
    comments = [SimpleNamespace(body=body) for body in bodies]
    statuses = [r.metadata.scheduler_metadata_status
                for r in _extract_round_metadata_records(comments, flow="plan")]
    assert "invalid" in statuses
    with pytest.raises(AgentLoopError, match="planning policy cannot be verified"):
        plan_verification_inputs(make_config(tmp_path), comments, issue_number=959)
    with pytest.raises(AgentLoopError, match="approving plan board cannot be verified"):
        _verify_strict_managed_plan_binding(
            config=make_config(tmp_path, reviewer=("gemini",)),
            pr_number=7,
            issue_context=Issue(959, "OWNER/REPO", "t", "b", None, tuple(
                IssueComment(author="agent-loop", created_at=None, body=body) for body in bodies
            )),
            metadata=SimpleNamespace(head_branch="agent-loop/managed-959", head_sha="h"),
            expected_plan_hash=approved_plan_hash(plan),
        )
