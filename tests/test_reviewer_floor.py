"""The per-invocation reviewer-board floor (#1373)."""

from dataclasses import replace

import pytest

from agent_loop_helpers import FakeRunner, approved_plan_comments, make_config, structured_pr_review
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import IssueContext
from coding_review_agent_loop.reviewer_floor import (
    BoardSeat,
    SignedBoardAuthorization,
    check_reviewer_floor,
)
from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

_AGENT_COMMANDS = {"codex", "claude", "agy", "gemini"}


def _agent_commands(runner):
    return [command for command, _cwd in runner.commands if command[0] in _AGENT_COMMANDS]


def _seat(tmp_path, name, backend, model):
    return SeatAgent(ReviewerSeat(name, backend, (model,), "medium"), tmp_path / name)


# --- Option values -------------------------------------------------------------


def test_floor_options_parse_into_config(monkeypatch, tmp_path):
    from coding_review_agent_loop.cli import build_parser
    from coding_review_agent_loop.config import config_from_args

    monkeypatch.setattr("coding_review_agent_loop.config.sys.platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    parser = build_parser()
    args = parser.parse_args([
        "pr", "77", "--repo", "OWNER/REPO", "--coder", "codex", "--reviewer", "claude",
        "--min-reviewers", "2", "--min-distinct-providers", "3",
    ])
    config = config_from_args(args, FakeRunner())
    assert (config.min_reviewers, config.min_distinct_providers) == (2, 3)
    for command in (["issue", "56"], ["managed-pr", "--head", "feature", "--title", "T"]):
        assert parser.parse_args([*command, "--min-reviewers", "2"]).min_reviewers == 2
    default = parser.parse_args(["pr", "77"])
    assert (default.min_reviewers, default.min_distinct_providers) == (None, None)


@pytest.mark.parametrize("field", ["min_reviewers", "min_distinct_providers"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2"])
def test_floor_options_reject_non_positive_values(tmp_path, field, value):
    flag = "--" + field.replace("_", "-")
    with pytest.raises(AgentLoopError, match=f"{flag} must be a positive integer"):
        make_config(tmp_path, **{field: value})


def test_config_construction_and_replace_never_check_the_board(tmp_path):
    """floor-distinct-providers: the option is accepted on any board."""
    seats = (_seat(tmp_path, "a", "codex", "gpt-a"), _seat(tmp_path, "b", "codex", "gpt-b"))
    config = make_config(
        tmp_path, reviewer=seats, reviewer_seats=seats,
        min_reviewers=5, min_distinct_providers=2,
    )
    # Rebinding to a reduced board, as a signed amendment does, never trips it.
    reduced = replace(config, reviewer=seats[:1])
    assert reduced.min_reviewers == 5 and len(reduced.reviewer) == 1


# --- The helper ----------------------------------------------------------------


def test_providers_are_counted_by_backend_not_seat_id():
    same_backend = (BoardSeat("a", "codex"), BoardSeat("b", "codex"))
    check_reviewer_floor(same_backend, min_reviewers=2, min_distinct_providers=None)
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        check_reviewer_floor(same_backend, min_reviewers=None, min_distinct_providers=2)
    check_reviewer_floor(
        (BoardSeat("a", "codex"), BoardSeat("b", "claude")),
        min_reviewers=None, min_distinct_providers=2,
    )


def test_no_floor_flags_never_refuse():
    check_reviewer_floor((), min_reviewers=None, min_distinct_providers=None)


def test_signed_exception_applies_only_to_the_measures_the_chain_lowered():
    board = (BoardSeat("a", "codex"), BoardSeat("b", "claude"))
    lowered_count = SignedBoardAuthorization(board, lowered_reviewers=True, lowered_providers=False)
    check_reviewer_floor(board, min_reviewers=3, min_distinct_providers=None,
                         authorizations=(lowered_count,))
    with pytest.raises(AgentLoopError, match="2 distinct provider\\(s\\) is below the floor of 3"):
        check_reviewer_floor(board, min_reviewers=3, min_distinct_providers=3,
                             authorizations=(lowered_count,))
    # The exception is bounded by the authorized board's own measures.
    with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
        check_reviewer_floor(board[:1], min_reviewers=3, min_distinct_providers=None,
                             authorizations=(lowered_count,))
    swapped = (BoardSeat("a", "codex"), BoardSeat("b", "codex"))
    both = SignedBoardAuthorization(board, lowered_reviewers=True, lowered_providers=True)
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        check_reviewer_floor(swapped, min_reviewers=3, min_distinct_providers=2,
                             authorizations=(both,))


# --- Entry points --------------------------------------------------------------


def _run_pr(config, runner, *, plan=None):
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.round_state import make_approved_plan_context

    if plan is None:
        return run_pr_loop(runner, pr_number=77, config=config)
    issue = IssueContext(56, config.repo, "Issue", "Body", None, ())
    return run_pr_loop(
        runner, pr_number=77, config=config, issue_context=issue,
        approved_plan_context=make_approved_plan_context(plan),
    )


def _three_reviewer_plan_runner(plan):
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata, _plan_subject

    comments = approved_plan_comments(plan)
    for index, agent in enumerate(("Claude", "Gemini"), start=2):
        comments.append({
            "author": {"login": "bot"},
            "createdAt": f"2026-09-20T00:00:0{index}Z",
            "body": _attach_round_metadata("Plan looks sound.", PostedRoundMetadata(
                flow="plan", role="reviewer", agent=agent, round_number=1,
                subject=_plan_subject(plan), state="approved",
            )),
        })
    return FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: comments},
    )


@pytest.mark.parametrize("handoff", [False, True])
def test_below_floor_board_is_refused_before_any_agent_or_comment(tmp_path, handoff):
    """floor-handoff-shrink: no history, or a plan approved by three reviewers."""
    plan = "Approved three-reviewer plan."
    runner = _three_reviewer_plan_runner(plan) if handoff else FakeRunner()
    config = make_config(tmp_path, reviewer=("codex",), pre_review_tests=False, min_reviewers=2)
    with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
        _run_pr(config, runner, plan=plan if handoff else None)
    assert runner.comments == []
    assert _agent_commands(runner) == []


@pytest.mark.parametrize("handoff", [False, True])
def test_two_seats_on_one_backend_are_one_provider(tmp_path, handoff):
    """floor-distinct-providers: fresh run and handoff."""
    plan = "Approved three-reviewer plan."
    runner = _three_reviewer_plan_runner(plan) if handoff else FakeRunner()
    seats = (_seat(tmp_path, "a", "codex", "gpt-a"), _seat(tmp_path, "b", "codex", "gpt-b"))
    config = make_config(
        tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False,
        min_distinct_providers=2,
    )
    with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
        _run_pr(config, runner, plan=plan if handoff else None)
    assert runner.comments == []
    assert _agent_commands(runner) == []


def test_board_at_the_floor_runs_normally(tmp_path):
    from coding_review_agent_loop.agents.registry import agent_signature

    config = make_config(
        tmp_path, reviewer=("codex", "claude"), pre_review_tests=False,
        min_reviewers=2, min_distinct_providers=2,
    )
    runner = FakeRunner(
        codex_outputs=[structured_pr_review(reviewer=agent_signature("codex", config))],
        claude_outputs=[structured_pr_review(reviewer=agent_signature("claude", config))],
    )
    assert _run_pr(config, runner) == 0
    assert {command[0] for command in _agent_commands(runner)} == {"codex", "claude"}


def test_issue_run_below_the_floor_is_refused_before_planning(tmp_path):
    from coding_review_agent_loop.cli import run_issue_loop

    runner = FakeRunner()
    config = make_config(tmp_path, reviewer=("codex",), min_reviewers=2)
    with pytest.raises(AgentLoopError, match="Issue #56 reviewer board"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert runner.comments == []
    assert _agent_commands(runner) == []
