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
import coding_review_agent_loop.pr_loop as pr_loop_module

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


def _amended_named_pr(
    tmp_path, monkeypatch, *, amendment_pr=77, reason="seat-unavailable", effective_round=1,
):
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
        effective_from_round=effective_round, reason=reason, rationale="Seat quota exhausted.",
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

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch)
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)

    def reached(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(pr_loop_module, "activate_managed_ci", reached)
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

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch, amendment_pr=78)
    comments_before = len(runner.comments)
    monkeypatch.setattr(
        pr_loop_module, "activate_managed_ci",
        lambda *a, **k: pytest.fail("managed-CI activation ran before amendment validation"),
    )
    with pytest.raises(AgentLoopError, match="PR #78|another|pull request"):
        run_pr_loop(
            runner, pr_number=77, config=make_config(tmp_path, **board, min_reviewers=2),
            workdirs_ready=True,
        )
    assert len(runner.comments) == comments_before


@pytest.mark.parametrize(
    ("amendment", "refusal"),
    [
        # A backend outage must remove every active seat on that backend:
        # opus stays on Antigravity while flash is removed.
        ({"reason": "backend-unavailable"}, "must remove every active seat"),
        # A pending amendment must take effect at the round the resume
        # re-enters, not a later one.
        ({"effective_round": 5}, "Human decision required"),
    ],
    ids=["incomplete-backend-outage", "wrong-activation-round"],
)
def test_invalid_signed_pr_lineage_never_authorizes_the_floor_or_writes(
    tmp_path, monkeypatch, amendment, refusal
):
    """item-3: every read-only amendment check runs before authorization and writes."""

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch, **amendment)
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    monkeypatch.setattr(
        pr_loop_module, "activate_managed_ci",
        lambda *a, **k: pytest.fail("managed-CI activation ran before amendment validation"),
    )
    with pytest.raises(AgentLoopError, match=refusal):
        run_pr_loop(
            runner, pr_number=77, config=make_config(tmp_path, **board, min_reviewers=4),
            workdirs_ready=True,
        )
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]


@pytest.mark.parametrize("below_floor", [True, False])
def test_retained_managed_label_is_released_only_after_the_floor(tmp_path, monkeypatch, below_floor):
    """item-6: a below-floor board leaves the ready PR's retained label untouched."""
    from test_managed_ci import EntryNormalizationRunner, _live_pr, _managed_label_deletes

    runner = EntryNormalizationRunner(_live_pr())
    reviewers = ("codex",) if below_floor else ("codex", "claude")
    config = make_config(tmp_path, reviewer=reviewers, min_reviewers=2, pre_review_tests=False)

    def reached(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(pr_loop_module, "activate_managed_ci", reached)
    if below_floor:
        with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
            pr_loop_module.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
        assert _managed_label_deletes(runner) == []
        assert runner.comments == []
    else:
        with pytest.raises(_Reached):
            pr_loop_module.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
        assert len(_managed_label_deletes(runner)) == 1


# --- Retained-label release waits for the authoritative board (#1378 round 3) ---


def _watch_retained_label(monkeypatch, *, proceed=_Reached):
    """A ready PR that retains the managed label; record any release attempt."""
    import coding_review_agent_loop.pr_loop as pr_loop

    releases = []

    def release(*_a, **_k):
        releases.append(True)
        raise proceed

    monkeypatch.setattr(pr_loop, "release_retained_managed_label", release)
    monkeypatch.setattr(
        pr_loop_module, "activate_managed_ci",
        lambda *a, **k: pytest.fail("managed-CI activation ran before the board was resolved"),
    )
    return releases


@pytest.mark.parametrize(
    ("amendment", "refusal"),
    [
        ({"reason": "backend-unavailable"}, "must remove every active seat"),
        ({"effective_round": 5}, "Human decision required"),
    ],
    ids=["incomplete-backend-outage", "wrong-activation-round"],
)
def test_invalid_signed_pr_lineage_keeps_the_retained_label_without_a_floor(
    tmp_path, monkeypatch, amendment, refusal
):
    """item-3: lineage validation precedes the label release even with no floor."""
    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch, **amendment)
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    releases = _watch_retained_label(monkeypatch)
    with pytest.raises(AgentLoopError, match=refusal):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **board), workdirs_ready=True)
    assert releases == []
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]


@pytest.mark.parametrize(
    "floor",
    [{"min_reviewers": 3}, {"min_reviewers": 3, "min_distinct_providers": 2}],
    ids=["reviewer-floor", "both-floors"],
)
def test_staged_malformed_history_with_a_signed_amendment_writes_nothing(
    tmp_path, monkeypatch, floor
):
    """item-3: a primary-then-panel PR never swallows undecodable history.

    An unsigned reduction at or above the floor is a recorded board change
    (#1376), so the floor here makes the reduction need the signed record.
    """
    import base64
    import os

    import coding_review_agent_loop.round_transport as transport
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
    from pr_loop_test_helpers import (
        _m943_amendment_from_error, _m943_append, _m943_partial_pr_round, _staged_config,
    )

    runner = _m943_partial_pr_round(tmp_path)
    reduced = _staged_config(tmp_path, reviewer=("codex", "gemini"), **floor)
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=reduced)
    _m943_append(runner, _m943_amendment_from_error(str(excinfo.value)))
    # An anchor whose spilled sidecar is missing cannot be decoded.
    anchor = transport.prepare_round_comment(_attach_round_metadata("Visible review", PostedRoundMetadata(
        flow="pr", role="reviewer", agent="codex", round_number=1, subject="abc123",
        phase="provisional", canonical_reviewer_response=base64.urlsafe_b64encode(os.urandom(46_000)).decode(),
    )))[-1]
    _m943_append(runner, anchor, login="bot")
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    releases = _watch_retained_label(monkeypatch)
    with pytest.raises(AgentLoopError, match="Incomplete round metadata"):
        run_pr_loop(runner, pr_number=77, config=reduced)
    assert releases == []
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]


def _child_with_unrelated_signed_plan(tmp_path):
    """Child #56 holds its own signed 3->2 plan; parent #55 approved the PR's plan on 3."""
    from test_issue_pr_handoff import _named_config, _named_plan_comments, _three_provider_plan
    from coding_review_agent_loop.round_state import make_approved_plan_context

    seats, parent_plan = _three_provider_plan(tmp_path)
    child_plan = make_approved_plan_context("An unrelated child plan.")
    plan_config = _named_config(tmp_path, seats)
    runner = FakeRunner(
        issue_payloads_by_number={
            55: {"number": 55},
            56: {"number": 56, "body": "Child phase issue for parent #55."},
        },
        issue_comments_by_number={
            55: _named_plan_comments(parent_plan, plan_config, issue=55),
            56: _named_plan_comments(child_plan, plan_config, removed=("c",)),
        },
    )
    return seats, parent_plan, runner


def test_entry_floor_uses_the_parent_board_of_the_exact_plan(tmp_path, monkeypatch):
    """item-6: an unrelated signed child plan never authorizes the label release."""
    from coding_review_agent_loop.cli import run_pr_loop
    from test_issue_pr_handoff import _named_config

    seats, parent_plan, runner = _child_with_unrelated_signed_plan(tmp_path)
    releases = _watch_retained_label(monkeypatch)
    with pytest.raises(AgentLoopError, match="2 reviewer\\(s\\) is below the floor of 3"):
        run_pr_loop(
            runner, pr_number=77, config=_named_config(tmp_path, seats[:2], min_reviewers=3),
            issue_context=IssueContext(56, "OWNER/REPO", "Child", "Body", None, ()),
            parent_issue_context=IssueContext(55, "OWNER/REPO", "Parent", "Body", None, ()),
            approved_plan_context=parent_plan, workdirs_ready=True,
        )
    assert releases == []
    assert runner.comments == []
    assert _agent_commands(runner) == []


def _standalone_plan_pr(tmp_path, *, plan_reduction, scope):
    """Issue #56 hands PR #77 off for an approved plan; the PR run has no issue context."""
    from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment
    from coding_review_agent_loop.pr_contract import format_pr_contract_comment, make_pr_contract
    from coding_review_agent_loop.round_state import make_approved_plan_context
    from test_issue_pr_handoff import _named_config, _named_plan_comments, _three_provider_plan

    seats, approved = _three_provider_plan(tmp_path)
    plan_config = _named_config(tmp_path, seats)
    if plan_reduction == "signed":
        comments = _named_plan_comments(approved, plan_config, removed=("c",))
        handed_off = approved
    elif plan_reduction == "unsigned":
        comments = _named_plan_comments(approved, plan_config)
        handed_off = approved
    else:
        # The issue's signed reduction belongs to a different plan than the
        # one the handoff names.
        comments = _named_plan_comments(approved, plan_config, removed=("c",))
        handed_off = make_approved_plan_context("A different approved plan.")
    comments.append({
        "author": {"login": "bot"}, "createdAt": "2026-05-23T00:01:00Z", "id": 99,
        "body": format_issue_pr_handoff_comment(
            issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
            pr_head_sha="abc123", flow="approved-plan-implementation",
            plan_hash=handed_off.plan_hash,
        ),
    })
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: comments},
    )
    kwargs = {}
    if scope == "explicit-scope":
        kwargs["managed_ci_issue_number"] = 56
    else:
        runner.pr_payload.setdefault("comments", []).append({
            "author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:00Z",
            "body": format_pr_contract_comment(make_pr_contract(
                repository="OWNER/REPO", pr_number=77,
                origin_flow="approved-plan-implementation",
                expected_closing_issue_ids=(56,), primary_issue_number=56,
            )),
        })
    return seats, runner, kwargs


@pytest.mark.parametrize("scope", ["explicit-scope", "ordinary-recovery"])
@pytest.mark.parametrize("plan_reduction", ["signed", "unsigned", "unrelated-plan"])
def test_standalone_pr_entry_floor_resolves_the_bound_plan(
    tmp_path, monkeypatch, scope, plan_reduction
):
    """item-7: a standalone PR resolves its owning issue and plan before the release."""
    from coding_review_agent_loop.cli import run_pr_loop
    from test_issue_pr_handoff import _named_config

    seats, runner, kwargs = _standalone_plan_pr(
        tmp_path, plan_reduction=plan_reduction, scope=scope,
    )
    releases = _watch_retained_label(monkeypatch)
    config = _named_config(tmp_path, seats[:2], min_reviewers=3)
    if plan_reduction == "signed":
        with pytest.raises(_Reached):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True, **kwargs)
        assert releases == [True]
    else:
        with pytest.raises(AgentLoopError, match="2 reviewer\\(s\\) is below the floor of 3"):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True, **kwargs)
        assert releases == []
    assert runner.comments == []
    assert _agent_commands(runner) == []


def _no_handoff_plan_pr(tmp_path, *, plan_state, scope):
    """Issue #56 holds an approved plan but no handoff record; the PR run has no issue context."""
    from coding_review_agent_loop.reviewer_seats import reviewer_seat_binding
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
    from test_issue_pr_handoff import _named_config, _named_plan_comments, _three_provider_plan

    seats, approved = _three_provider_plan(tmp_path)
    plan_config = _named_config(tmp_path, seats)
    binding = reviewer_seat_binding(plan_config)
    if plan_state == "unsigned":
        comments = _named_plan_comments(approved, plan_config)
        approvals = ((1, "a"), (1, "b"), (1, "c"))
    else:
        comments = _named_plan_comments(approved, plan_config, removed=("c",))
        # An incomplete plan lacks b's approval on the signed two-seat board.
        approvals = ((2, "a"), (2, "b")) if plan_state == "signed" else ((2, "a"),)
    for index, (round_number, agent) in enumerate(approvals):
        comments.append({
            "author": {"login": "bot"}, "createdAt": f"2026-05-23T00:02:{index:02d}Z",
            "id": 60 + index,
            "body": _attach_round_metadata("Approved.", PostedRoundMetadata(
                flow="plan", role="reviewer", agent=agent, round_number=round_number,
                subject="plan", state="approved", seat_binding=binding,
            )),
        })
    runner = FakeRunner(
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: comments},
    )
    kwargs = {}
    overrides = {}
    if scope in {"explicit-scope", "caller-issue-fresh"}:
        # Managed-CI fresh authorization with an explicit issue scope; the
        # issue-mode resume also passes the issue context in-process.
        kwargs["managed_ci_issue_number"] = 56
        if scope == "caller-issue-fresh":
            kwargs["issue_context"] = IssueContext(56, "OWNER/REPO", "Issue", "Body", None, ())
        overrides.update(
            managed_ci=True, managed_ci_pr_mode=True, managed_ci_fresh_authorization=True,
            managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        )
    else:
        # Ordinary managed-CI recovery binds the issue named by the managed branch,
        # even when a PR-side contract already names that issue as primary.
        runner.pr_payload["headRefName"] = "agent-loop/managed-56"
        overrides["managed_ci_pr_mode"] = True
        if scope == "pr-contract-managed-branch":
            from coding_review_agent_loop.pr_contract import (
                format_pr_contract_comment, make_pr_contract,
            )

            runner.pr_payload.setdefault("comments", []).append({
                "author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:00Z",
                "body": format_pr_contract_comment(make_pr_contract(
                    repository="OWNER/REPO", pr_number=77,
                    origin_flow="approved-plan-implementation",
                    expected_closing_issue_ids=(56,), primary_issue_number=56,
                )),
            })
    return seats, runner, kwargs, overrides


@pytest.mark.parametrize(
    "scope",
    ["explicit-scope", "managed-branch", "caller-issue-fresh", "pr-contract-managed-branch"],
)
@pytest.mark.parametrize("plan_state", ["signed", "unsigned", "incomplete", "binding-swap"])
def test_no_handoff_pr_entry_floor_verifies_the_canonical_plan(
    tmp_path, monkeypatch, scope, plan_state
):
    """items 7, 11, 12: without a handoff record, every managed-CI recovery scope uses
    the verified canonical plan at entry, however the owning issue was found."""
    from coding_review_agent_loop.cli import run_pr_loop
    from test_issue_pr_handoff import _named_config, _seat as _handoff_seat

    seats, runner, kwargs, overrides = _no_handoff_plan_pr(
        tmp_path, plan_state="signed" if plan_state == "binding-swap" else plan_state,
        scope=scope,
    )
    releases = _watch_retained_label(monkeypatch)
    if plan_state == "binding-swap":
        # The inherited seat set with b moved from Claude to Codex: the signed
        # plan board becomes a, b on one provider.
        swapped = (seats[0], _handoff_seat(tmp_path, "b", "codex", "gpt-6.1-flash", "medium"), seats[2])
        config = _named_config(tmp_path, swapped, min_distinct_providers=2, **overrides)
    else:
        config = _named_config(tmp_path, seats[:2], min_reviewers=3, **overrides)
    if plan_state == "signed":
        with pytest.raises(_Reached):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True, **kwargs)
        assert releases == [True]
    else:
        refusal = {
            "unsigned": "2 reviewer\\(s\\) is below the floor of 3",
            "incomplete": "without a complete canonical reviewer approval",
            "binding-swap": "1 distinct provider\\(s\\) is below the floor of 2",
        }[plan_state]
        with pytest.raises(AgentLoopError, match=refusal):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True, **kwargs)
        assert releases == []
    assert runner.comments == []
    assert _agent_commands(runner) == []


def test_persisted_seat_binding_drift_never_authorizes_the_floor_or_writes(tmp_path, monkeypatch):
    """item-8: an activated signed removal plus a backend change on a remaining seat."""
    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch)
    # Activate the signed removal of flash with a complete round on the new head.
    runner.codex_outputs.append(structured_pr_review())
    runner.gemini_outputs.append(structured_pr_review(reviewer="Google Gemini"))
    runner.antigravity_outputs.append(
        (structured_pr_review(reviewer="opus (Google Antigravity: Model B)"), 0)
    )
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **board)) == 0
    assert any("Reviewer board amendment applied." in comment for comment in runner.comments)
    # opus stays a seat ID but moves from Antigravity to Codex.  The swap is an
    # unsigned board change (#1376): the signed removal never lowered the
    # provider count, so it cannot authorize the provider floor the swap breaks.
    opus = _seat(tmp_path, "opus", "codex", "gpt-b")
    drifted = dict(board, reviewer=("codex", "gemini", opus), reviewer_seats=(opus,))
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    releases = _watch_retained_label(monkeypatch)
    with pytest.raises(AgentLoopError, match="2 distinct provider\\(s\\) is below the floor of 3"):
        run_pr_loop(
            runner, pr_number=77,
            config=make_config(tmp_path, **drifted, min_distinct_providers=3),
            workdirs_ready=True,
        )
    assert releases == []
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]


@pytest.mark.parametrize("lifecycle", ["draft-labeled", "ready-unlabeled"])
@pytest.mark.parametrize(
    ("board", "refusal"),
    [
        ("one-reviewer", "1 reviewer\\(s\\) is below the floor of 2"),
        ("one-provider", "1 distinct provider\\(s\\) is below the floor of 2"),
    ],
)
def test_fresh_authorization_below_the_floor_publishes_nothing(
    tmp_path, monkeypatch, lifecycle, board, refusal
):
    """item-9: the floor runs before managed-CI fresh authorization, whatever the label state."""
    from coding_review_agent_loop.managed_ci import parse_issue_created_authorization_comment
    from test_managed_ci import _workflow_runner_for_issue_authorization

    runner = _workflow_runner_for_issue_authorization(None, labeled=lifecycle == "draft-labeled")
    if lifecycle == "ready-unlabeled":
        runner.rest_pr["draft"] = False
    if board == "one-reviewer":
        floor = dict(reviewer=("codex",), min_reviewers=2)
    else:
        seats = (_seat(tmp_path, "a", "codex", "gpt-a"), _seat(tmp_path, "b", "codex", "gpt-b"))
        floor = dict(reviewer=seats, reviewer_seats=seats, min_distinct_providers=2)
    config = make_config(
        tmp_path, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=643,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
        **floor,
    )
    monkeypatch.setattr(
        pr_loop_module, "activate_managed_ci",
        lambda *a, **k: pytest.fail("managed-CI activation ran before the floor"),
    )
    with pytest.raises(AgentLoopError, match=refusal):
        pr_loop_module.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
    assert not [
        comment for comment in runner.intent_comments
        if parse_issue_created_authorization_comment(comment["body"]) is not None
    ]
    assert not runner.labels_posted
    assert runner.comments == []
    assert not [
        command for command, _cwd in runner.commands
        if command[:4] == ["gh", "api", "--method", "DELETE"]
        or command[:4] == ["gh", "api", "--method", "POST"]
    ]
    assert _agent_commands(runner) == []


def test_fresh_authorization_with_a_signed_plan_reduction_reaches_authorization(
    tmp_path, monkeypatch,
):
    """item-9: a valid inherited signed plan reduction still reaches fresh authorization."""
    from coding_review_agent_loop.cli import run_pr_loop
    from test_issue_pr_handoff import _named_config

    seats, runner, kwargs, _overrides = _no_handoff_plan_pr(
        tmp_path, plan_state="signed", scope="explicit-scope",
    )

    def authorize(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(pr_loop_module, "authorize_fresh_issue_created_resume", authorize)
    config = _named_config(
        tmp_path, seats[:2], min_reviewers=3, managed_ci=True, managed_ci_pr_mode=True,
        managed_ci_fresh_authorization=True, managed_ci_issue_number=56,
        managed_ci_trusted_actor="agent-loop", allow_unprotected_managed_ci=True,
    )
    with pytest.raises(_Reached):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)
    assert runner.comments == []
    assert _agent_commands(runner) == []


def _incomplete_linked_plan_comments(tmp_path):
    _seats, runner, _kwargs, _overrides = _no_handoff_plan_pr(
        tmp_path, plan_state="incomplete", scope="managed-branch",
    )
    return runner.issue_comments_by_number[56]


@pytest.mark.parametrize("trigger", ["floor", "signed-pr-amendment"])
def test_ordinary_pr_ignores_unfinished_planning_on_its_linked_issue(
    tmp_path, monkeypatch, trigger
):
    """item-10: only managed-CI recovery binds a canonical plan without a handoff record."""

    plan_comments = _incomplete_linked_plan_comments(tmp_path)

    def reached(*_a, **_k):
        raise _Reached

    if trigger == "floor":
        from coding_review_agent_loop.cli import run_pr_loop

        runner = FakeRunner(
            issue_payloads_by_number={56: {"number": 56}},
            issue_comments_by_number={56: plan_comments},
        )
        config = make_config(
            tmp_path, reviewer=("codex", "claude"), pre_review_tests=False, min_reviewers=2,
        )
    else:
        runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch)
        runner.issue_payloads_by_number = {56: {"number": 56}}
        runner.issue_comments_by_number = {56: plan_comments}
        config = make_config(tmp_path, **board)
    runner.pr_payload["body"] = "Closes #56"
    comments_before = len(runner.comments)
    monkeypatch.setattr(pr_loop_module, "activate_managed_ci", reached)
    with pytest.raises(_Reached):
        run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True)
    assert len(runner.comments) == comments_before


@pytest.mark.parametrize("board", ["signed-reduction", "binding-swap"])
def test_direct_staged_child_entry_floor_uses_the_parent_plan(tmp_path, monkeypatch, board):
    """item-13: a direct-implementation staged child implements its parent's plan, so
    run_issue_loop checks the floor against the parent board before any write."""
    from dataclasses import replace as dc_replace

    import coding_review_agent_loop.issue_loop as issue_loop
    from coding_review_agent_loop.cli import run_issue_loop
    from coding_review_agent_loop.github import IssueComment
    from coding_review_agent_loop.protocol import EXECUTION_DISPOSITION_DIRECT
    from test_issue_pr_handoff import (
        _named_config, _named_plan_comments, _seat as _handoff_seat, _three_provider_plan,
    )
    from issue_loop_test_helpers import _fresh_child_route_fixture

    seats, approved = _three_provider_plan(tmp_path)
    parent_comments = tuple(
        IssueComment(author="bot", created_at=comment["createdAt"], body=comment["body"])
        for comment in _named_plan_comments(
            approved, _named_config(tmp_path, seats), removed=("c",), issue=55,
        )
    )
    fresh, child, parent = _fresh_child_route_fixture(EXECUTION_DISPOSITION_DIRECT)
    child = dc_replace(child, body="Child phase issue for parent #55.\n\nImplement child.")
    parent = dc_replace(parent, comments=parent_comments)
    monkeypatch.setattr(
        issue_loop, "get_issue_context",
        lambda _runner, *, config, issue_number: child if issue_number == 56 else parent,
    )
    monkeypatch.setattr(issue_loop, "_resolve_fresh_child_provenance", lambda **_: fresh)

    def reached(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(issue_loop, "resolve_base_branch", reached)
    for name in ("ensure_agent_workdirs", "_dispatch_decomposition_child", "prepare_agent_memory"):
        monkeypatch.setattr(
            issue_loop, name, lambda *_a, _n=name, **_k: pytest.fail(f"{_n} ran before the floor"),
        )
    runner = FakeRunner()
    if board == "signed-reduction":
        # The parent's signed 3 -> 2 reduction excuses the reviewer count.
        with pytest.raises(_Reached):
            run_issue_loop(
                runner, issue_number=56,
                config=_named_config(tmp_path, seats[:2], min_reviewers=3),
            )
    else:
        # b moved to Codex: the inherited parent board a, b has one provider.
        swapped = (seats[0], _handoff_seat(tmp_path, "b", "codex", "gpt-6.1-flash", "medium"), seats[2])
        with pytest.raises(AgentLoopError, match="1 distinct provider\\(s\\) is below the floor of 2"):
            run_issue_loop(
                runner, issue_number=56,
                config=_named_config(tmp_path, swapped, min_distinct_providers=2),
            )
    assert runner.comments == []
    assert _agent_commands(runner) == []


# --- Issue preflight branches over plan approval (#1377) ---------------------------------


def _unfinished_plan_runner(tmp_path):
    """A plan published and reviewed by Codex and Gemini; Gemini still blocks."""
    from agent_loop_helpers import structured_plan_review, structured_v1_plan_state
    from coding_review_agent_loop.cli import run_issue_loop

    runner = FakeRunner(
        authenticated_actor=("agent-loop", 4242), serve_rest_issue_comments=True,
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(
            state="blocking", reviewer="Google Gemini", summary="x",
            blocking_plan_issues=["Name the owner."],
        )],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), plan_execution_mode="plan-only",
        agent_max_retries=0,
    )
    with pytest.raises(AgentLoopError, match="scripted agent output exhausted"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    return runner


def test_unfinished_plan_preflight_floor_checks_the_plan_board_candidate(tmp_path, monkeypatch):
    """Preflight branch 2: an unfinished plan's board change is floor-checked read-only."""
    import coding_review_agent_loop.issue_loop as issue_loop
    from coding_review_agent_loop.cli import run_issue_loop

    runner = _unfinished_plan_runner(tmp_path)
    for name in ("ensure_agent_workdirs", "resolve_base_branch"):
        monkeypatch.setattr(
            issue_loop, name, lambda *_a, _n=name, **_k: pytest.fail(f"{_n} ran before the floor"),
        )
    comments = len(runner.issue_comments)
    agents = _agent_commands(runner)
    config = make_config(
        tmp_path, reviewer=("codex",), min_reviewers=2, plan_execution_mode="plan-only",
    )
    with pytest.raises(AgentLoopError, match="Issue #56 plan reviewer board") as refusal:
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    # The filled-in signed plan amendment template names the plan surface.
    assert '"flow": "plan"' in str(refusal.value)
    assert len(runner.issue_comments) == comments
    assert _agent_commands(runner) == agents


def test_approved_plan_preflight_keeps_the_stage_one_handoff_floor(tmp_path):
    """Preflight branch 1: an approved plan is judged only by the handoff board rule."""
    from agent_loop_helpers import structured_plan_review, structured_v1_plan_state
    from coding_review_agent_loop.cli import run_issue_loop
    from coding_review_agent_loop.github import IssueComment
    from coding_review_agent_loop.plan_verification import enforce_issue_board_floor

    runner = FakeRunner(
        authenticated_actor=("agent-loop", 4242), serve_rest_issue_comments=True,
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    board = make_config(tmp_path, reviewer=("codex", "gemini"), plan_execution_mode="plan-only")
    assert run_issue_loop(runner, issue_number=56, config=board, plan_first=True) == 0
    comments = tuple(
        IssueComment(
            author=c["author"]["login"], created_at=c["createdAt"], body=c["body"],
            author_id=c.get("_rest_author_id"),
        )
        for c in runner.issue_comments
    )
    # A differing planning policy and primary are never compared for an approved plan.
    staged = make_config(
        tmp_path, reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="gemini", min_reviewers=2,
    )
    enforce_issue_board_floor(staged, comments, issue_number=56, runner=runner, planning=True)
    below = make_config(tmp_path, reviewer=("codex",), min_reviewers=2)
    # A plan-only resume of the approved plan selects no board: no floor applies.
    enforce_issue_board_floor(
        replace(below, plan_execution_mode="plan-only"), comments,
        issue_number=56, runner=runner, planning=True,
    )
    # An implementation run meets the stage 1 handoff floor, never a plan floor.
    with pytest.raises(AgentLoopError, match="Issue #56 reviewer board") as refusal:
        enforce_issue_board_floor(
            replace(below, plan_execution_mode="implement-one-shot"), comments,
            issue_number=56, runner=runner, planning=True,
        )
    assert "plan reviewer board" not in str(refusal.value)
