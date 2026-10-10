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


def test_issue_run_below_the_floor_is_refused_before_planning(tmp_path, monkeypatch):
    import coding_review_agent_loop.issue_loop as issue_loop
    from coding_review_agent_loop.cli import run_issue_loop

    # Refused before base resolution and workdir setup (#1378 review item-4).
    for name in ("ensure_agent_workdirs", "resolve_base_branch"):
        monkeypatch.setattr(
            issue_loop, name, lambda *_a, _n=name, **_k: pytest.fail(f"{_n} ran before the floor"),
        )
    runner = FakeRunner()
    config = make_config(tmp_path, reviewer=("codex",), min_reviewers=2)
    with pytest.raises(AgentLoopError, match="Issue #56 reviewer board"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    assert runner.comments == []
    assert _agent_commands(runner) == []


def test_managed_pr_below_the_floor_creates_nothing(tmp_path, monkeypatch, capsys):
    """item-4: no workdir setup, branch, or draft PR before the floor check."""
    import coding_review_agent_loop.cli as cli_module

    monkeypatch.setattr(
        cli_module, "config_from_args",
        lambda *a, **k: make_config(tmp_path, reviewer=("codex",), min_reviewers=2),
    )
    for name in ("resolve_base_branch", "ensure_agent_workdirs", "create_managed_pr", "run_pr_loop"):
        monkeypatch.setattr(
            cli_module, name, lambda *_a, _n=name, **_k: pytest.fail(f"{_n} ran before the floor"),
        )
    assert cli_module.main([
        "managed-pr", "--repo", "OWNER/REPO", "--head", "fix/direct-change",
        "--title", "Direct change", "--managed-ci", "--managed-ci-trusted-actor", "agent-loop",
        "--min-reviewers", "2",
    ]) == 1
    assert "1 reviewer(s) is below the floor of 2" in capsys.readouterr().err


class _Reached(Exception):
    pass


def _amended_named_pr(tmp_path, monkeypatch, *, amendment_pr=77):
    """A selective PR reviewed by codex, gemini, flash, opus; then a signed removal of flash."""
    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from coding_review_agent_loop.cli import run_pr_loop

    flash = _seat(tmp_path, "flash", "antigravity", "Model A")
    opus = _seat(tmp_path, "opus", "antigravity", "Model B")
    flash.workdir.mkdir()
    opus.workdir.mkdir()
    runner = FakeRunner(
        codex_outputs=[structured_pr_review()],
        gemini_outputs=[structured_pr_review(reviewer="Google Gemini")],
        antigravity_outputs=[
            (structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0),
            (structured_pr_review(reviewer="opus (Google Antigravity: Model B)"), 0),
        ],
    )
    board = dict(
        reviewer=("codex", "gemini", flash, opus), reviewer_seats=(flash, opus),
        pr_review_policy="selective-intermediate", pre_review_tests=False, agent_max_retries=0,
    )
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **board)) == 0
    runner.pr_payload["headRefOid"] = "def456"
    body = format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=amendment_pr,
        original_required_reviewers=("Codex", "Gemini", "flash", "opus"),
        policy="selective-intermediate", primary_reviewer=None, removed_reviewers=("flash",),
        effective_from_round=1, reason="seat-unavailable", rationale="Seat quota exhausted.",
    )
    index = len(runner.pr_payload["comments"])
    runner.pr_payload["comments"].append({
        "body": body, "author": {"login": "human-reviewer", "id": 81},
        "createdAt": f"2026-05-23T00:00:{index:02d}Z", "id": 900,
    })
    return runner, board, run_pr_loop


@pytest.mark.parametrize(
    ("floor", "refusal"),
    [
        # The removal lowered the reviewer count, not the provider count.
        ({"min_distinct_providers": 4}, "3 distinct provider\\(s\\) is below the floor of 4"),
        ({"min_reviewers": 4}, None),
    ],
    ids=["provider-floor-not-lowered", "reviewer-floor-lowered"],
)
def test_signed_pr_amendment_floor_is_resolved_before_any_pr_write(
    tmp_path, monkeypatch, floor, refusal
):
    """item-3: lineage resolution and the floor run before activation and posts."""
    import coding_review_agent_loop.orchestrator as orchestrator

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch)
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)

    def reached(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(orchestrator, "activate_managed_ci", reached)
    config = make_config(tmp_path, **board, **floor)
    if refusal is None:
        with pytest.raises(_Reached):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)
    else:
        with pytest.raises(AgentLoopError, match=refusal):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]


def test_malformed_pr_amendment_history_is_refused_before_any_pr_write(tmp_path, monkeypatch):
    import coding_review_agent_loop.orchestrator as orchestrator

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch, amendment_pr=78)
    comments_before = len(runner.comments)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci",
        lambda *a, **k: pytest.fail("managed-CI activation ran before amendment validation"),
    )
    with pytest.raises(AgentLoopError, match="PR #78|another|pull request"):
        run_pr_loop(
            runner, pr_number=77, config=make_config(tmp_path, **board, min_reviewers=2),
            workdirs_ready=True,
        )
    assert len(runner.comments) == comments_before
