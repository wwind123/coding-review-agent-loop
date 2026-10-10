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
    import coding_review_agent_loop.orchestrator as orchestrator

    runner, board, run_pr_loop = _amended_named_pr(tmp_path, monkeypatch, **amendment)
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci",
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
    import coding_review_agent_loop.orchestrator as orchestrator
    from test_managed_ci import EntryNormalizationRunner, _live_pr, _managed_label_deletes

    runner = EntryNormalizationRunner(_live_pr())
    reviewers = ("codex",) if below_floor else ("codex", "claude")
    config = make_config(tmp_path, reviewer=reviewers, min_reviewers=2, pre_review_tests=False)

    def reached(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(orchestrator, "activate_managed_ci", reached)
    if below_floor:
        with pytest.raises(AgentLoopError, match="1 reviewer\\(s\\) is below the floor of 2"):
            orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
        assert _managed_label_deletes(runner) == []
        assert runner.comments == []
    else:
        with pytest.raises(_Reached):
            orchestrator.run_pr_loop(runner, pr_number=7, config=config, workdirs_ready=True)
        assert len(_managed_label_deletes(runner)) == 1


# --- Retained-label release waits for the authoritative board (#1378 round 3) ---


def _watch_retained_label(monkeypatch, *, proceed=_Reached):
    """A ready PR that retains the managed label; record any release attempt."""
    import coding_review_agent_loop.orchestrator as orchestrator
    import coding_review_agent_loop.pr_loop as pr_loop

    releases = []

    def release(*_a, **_k):
        releases.append(True)
        raise proceed

    monkeypatch.setattr(pr_loop, "retained_managed_label_present", lambda *a, **k: True)
    monkeypatch.setattr(pr_loop, "release_retained_managed_label", release)
    monkeypatch.setattr(
        orchestrator, "activate_managed_ci",
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


@pytest.mark.parametrize("floor", [{}, {"min_reviewers": 2}], ids=["no-floor", "floor"])
def test_staged_malformed_history_with_a_signed_amendment_writes_nothing(
    tmp_path, monkeypatch, floor
):
    """item-3: a primary-then-panel PR never swallows undecodable history."""
    import base64
    import os

    import coding_review_agent_loop.round_transport as transport
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata
    from test_orchestrator_pr import (
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
    if scope == "explicit-scope":
        # Managed-CI fresh authorization with an explicit issue scope.
        kwargs["managed_ci_issue_number"] = 56
    else:
        # Ordinary managed-CI recovery binds the issue named by the managed branch.
        runner.pr_payload["headRefName"] = "agent-loop/managed-56"
        overrides["managed_ci_pr_mode"] = True
    return seats, runner, kwargs, overrides


@pytest.mark.parametrize("scope", ["explicit-scope", "managed-branch"])
@pytest.mark.parametrize("plan_state", ["signed", "unsigned", "incomplete"])
def test_no_handoff_pr_entry_floor_verifies_the_canonical_plan(
    tmp_path, monkeypatch, scope, plan_state
):
    """item-7: without a handoff record the entry floor uses the verified canonical plan."""
    from coding_review_agent_loop.cli import run_pr_loop
    from test_issue_pr_handoff import _named_config

    seats, runner, kwargs, overrides = _no_handoff_plan_pr(
        tmp_path, plan_state=plan_state, scope=scope,
    )
    releases = _watch_retained_label(monkeypatch)
    config = _named_config(tmp_path, seats[:2], min_reviewers=3, **overrides)
    if plan_state == "signed":
        with pytest.raises(_Reached):
            run_pr_loop(runner, pr_number=77, config=config, workdirs_ready=True, **kwargs)
        assert releases == [True]
    else:
        refusal = (
            "2 reviewer\\(s\\) is below the floor of 3" if plan_state == "unsigned"
            else "without a complete canonical reviewer approval"
        )
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
    # opus stays a seat ID but moves from Antigravity to Codex.
    opus = _seat(tmp_path, "opus", "codex", "gpt-b")
    drifted = dict(board, reviewer=("codex", "gemini", board["reviewer"][2], opus),
                   reviewer_seats=(board["reviewer_seats"][0], opus))
    comments_before = len(runner.comments)
    commands_before = len(runner.commands)
    releases = _watch_retained_label(monkeypatch)
    with pytest.raises(AgentLoopError, match="backend"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, **drifted, min_reviewers=4),
                    workdirs_ready=True)
    assert releases == []
    assert len(runner.comments) == comments_before
    assert not [c for c, _ in runner.commands[commands_before:] if c[0] in _AGENT_COMMANDS]
