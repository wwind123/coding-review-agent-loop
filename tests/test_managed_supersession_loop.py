"""run_pr_loop coverage for managed-success supersession and timeout routing (#1117)."""
import dataclasses
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.ci_health import PullRequestCheck, PullRequestChecks
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import PullRequestMergeability, get_pr_checks
from coding_review_agent_loop.managed_ci import FINAL_CONTEXT, ManagedCiContract, ManagedCiOutcome
from coding_review_agent_loop.orchestrator import run_pr_loop

from coding_review_agent_loop.round_state import _extract_round_metadata_records

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup, structured_pr_review
from test_ci_health import NOW, _StubGhRunner, _metadata, _run
from test_orchestrator_pr import _carried_ci_obligations, _carried_ci_review_comment

H = "abc123"


PERSISTED_TEXT = "Failing checks: Python 3.12 full suite (failure)"


def _items():
    # Modeled on #1115 item-10: github-pr-checks, qualification_ready at candidate H, failed head F.
    return (dataclasses.replace(_carried_ci_obligations()[1], text=PERSISTED_TEXT),)


def _records(runner):
    comments = [
        SimpleNamespace(body=c["body"]) for c in runner.pr_payload.get("comments", [])
    ]
    return _extract_round_metadata_records(comments, flow="pr")


def _checks_items(runner):
    return [
        item
        for record in _records(runner)
        for item in (*record.metadata.prior_items, *record.metadata.new_items)
        if item.obligation_kind == "github-pr-checks"
    ]


def _runner(items, **extra):
    return FakeRunner(
        **extra,
        codex_outputs=[
            structured_pr_review(
                reviewer="OpenAI Codex",
                state="approved",
                prior_item_dispositions=[
                    {"item_id": item.item_id, "disposition": "resolved"} for item in items
                ],
            )
        ],
        pr_payload={
            "comments": [
                {"author": {"login": "coding-review-agent-loop"}, "body": _carried_ci_review_comment(items)}
            ]
        },
    )


def _config(tmp_path, *, auto_merge):
    return make_config(
        tmp_path,
        reviewer=("codex",),
        pr_review_policy="selective-intermediate",
        managed_ci=True,
        auto_merge=auto_merge,
        max_rounds=1,
    )


class _Effects:
    def __init__(self):
        self.merges = []
        self.prepares = []
        self.publishes = []
        self.mergeability_reads = 0


def _install(monkeypatch, outcome, *, mergeability=None):
    effects = _Effects()
    monkeypatch.setattr(orchestrator, "activate_managed_ci", lambda *a, **k: ManagedCiContract())
    monkeypatch.setattr(orchestrator, "dispatch_final_qualification", lambda *a, **k: None)
    monkeypatch.setattr(orchestrator, "wait_for_final_qualification", lambda *a, **k: outcome)
    monkeypatch.setattr(orchestrator, "prepare_v2_merge", lambda *a, **k: effects.prepares.append(k))
    monkeypatch.setattr(orchestrator, "_merge_with_exact_head_proof", lambda *a, **k: effects.merges.append(k))
    monkeypatch.setattr(
        orchestrator, "publish_manual_v2_qualification",
        lambda *a, **k: effects.publishes.append(k) or H,
    )

    real = orchestrator._mergeability_for_unreadable_protection

    def read_mergeability(runner, **k):
        # Count only the supersession read, which happens for forbidden protection.
        if k["checks"] is None or k["checks"].branch_protection_status != "forbidden":
            return real(runner, **k)
        effects.mergeability_reads += 1
        return mergeability

    monkeypatch.setattr(orchestrator, "_mergeability_for_unreadable_protection", read_mergeability)
    return effects


def _board_from_payload(tmp_path, runs, *, statuses=None, total=None, status_total=None,
                        protection=None, payload_override=None):
    payload = payload_override or {
        "check_runs": runs,
        **({} if total is False else {"total_count": len(runs) if total is None else total}),
    }
    stub = _StubGhRunner(
        check_runs_payload=payload,
        status_payload={
            "state": "success",
            "statuses": statuses or [],
            "total_count": len(statuses or []) if status_total is None else status_total,
        },
        branch_protection_payload=protection or {"contexts": []},
    )
    return get_pr_checks(stub, config=make_config(tmp_path), metadata=_metadata(), now=NOW)


def _passed(board):
    return ManagedCiOutcome(status="passed", checks=board, head_sha=H)


def _good_runs():
    return [_run(FINAL_CONTEXT), _run("Python 3.12 full suite", "skipped", 2)]


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_clean_managed_success_clears_carried_obligation_and_finalizes(tmp_path, monkeypatch, auto_merge):
    items = _items()
    board = _board_from_payload(tmp_path, _good_runs())
    effects = _install(monkeypatch, _passed(board))

    runner = _runner(items)
    assert [i.text for i in _checks_items(runner)] == [PERSISTED_TEXT]
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0

    assert bool(effects.merges) is auto_merge
    assert bool(effects.publishes) is (not auto_merge)
    assert effects.mergeability_reads == 0


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_query_error_board_refuses_naming_predicate_without_finalizing(tmp_path, monkeypatch, auto_merge):
    items = _items()
    board = _board_from_payload(tmp_path, _good_runs())
    board = PullRequestChecks(**{**board.__dict__, "check_query_errors": ("boom",)})
    effects = _install(monkeypatch, _passed(board))

    runner = _runner(items)
    with pytest.raises(AgentLoopError, match=r"cannot finalize.*github-pr-checks.*boom"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=auto_merge))

    assert not (effects.merges or effects.prepares or effects.publishes)
    persisted = _checks_items(runner)
    assert persisted and all(
        item.text == PERSISTED_TEXT and item.lifecycle == "qualification_ready" for item in persisted
    )


@pytest.mark.parametrize("order", ["success-first", "failure-first"])
def test_shadowed_failure_refuses_in_both_api_orders(tmp_path, monkeypatch, order):
    runs = [_run(FINAL_CONTEXT), _run("X", "success", 2), _run("X", "failure", 3)]
    if order == "failure-first":
        runs = [_run(FINAL_CONTEXT), _run("X", "failure", 3), _run("X", "success", 2)]
    board = _board_from_payload(tmp_path, runs)
    if order == "failure-first":
        # The deduplicated board keeps the first observation, so it shows failing and
        # the real waiter would never return `passed`; the shadowed success is inert.
        assert board.state == "failing" and board.shadowed[0].status == "success"
        # The waiter-realistic outcome for this board is `timeout` (final context succeeded,
        # X failing): it must route to repair and never clear or merge.
        runner = _runner(_items())
        effects = _install(monkeypatch, ManagedCiOutcome(status="timeout", checks=board, head_sha=H))
        with pytest.raises(AgentLoopError):
            run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
        assert not (effects.merges or effects.prepares or effects.publishes)
        assert any("Failing checks: X" in comment for comment in runner.comments)
        assert any(i.lifecycle == "repair_required" for i in _checks_items(runner))
        return
    assert board.state == "passing" and len(board.shadowed) == 1
    effects = _install(monkeypatch, _passed(board))

    with pytest.raises(AgentLoopError, match=r"cannot finalize.*`X`"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))

    assert not (effects.merges or effects.prepares or effects.publishes)


@pytest.mark.parametrize("order", ["success-first", "skipped-first"])
@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_shadowed_skipped_required_check_refuses_but_optional_clears(
    tmp_path, monkeypatch, order, auto_merge
):
    pair = [_run("X", "success", 2), _run("X", "skipped", 3)]
    if order == "skipped-first":
        pair.reverse()
    runs = [_run(FINAL_CONTEXT), *pair]
    required = _board_from_payload(tmp_path, runs, protection={"contexts": ["X"]})
    assert len(required.shadowed) == 1
    effects = _install(monkeypatch, _passed(required))
    with pytest.raises(AgentLoopError, match=r"cannot finalize.*`X`"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge))
    assert not (effects.merges or effects.prepares or effects.publishes)

    optional = _board_from_payload(tmp_path, runs)
    effects = _install(monkeypatch, _passed(optional))
    assert run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0
    assert bool(effects.merges) is auto_merge
    assert bool(effects.publishes) is (not auto_merge)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 5},
        {"total": False},
        {"status_total": 4},
        {"payload_override": {"check_runs": [_run(FINAL_CONTEXT), "junk"], "total_count": 2}},
        {"payload_override": {"check_runs": [_run(FINAL_CONTEXT), {"id": 9, "status": "completed"}], "total_count": 2}},
        {"statuses": [{"state": "failure"}]},
    ],
    ids=["truncated", "missing-total", "status-truncated", "non-object", "nameless", "status-without-context"],
)
def test_incomplete_or_partially_parsed_listing_refuses(tmp_path, monkeypatch, kwargs):
    board = _board_from_payload(tmp_path, [_run(FINAL_CONTEXT)], **kwargs)
    assert board.listing_complete is False
    effects = _install(monkeypatch, _passed(board))

    with pytest.raises(AgentLoopError, match=r"cannot finalize.*incomplete"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))

    assert not (effects.merges or effects.prepares or effects.publishes)


def _mergeability(state, head=H):
    return PullRequestMergeability(
        state="mergeable", mergeable_raw="MERGEABLE", merge_state_raw=state,
        head_sha=head, base_branch="main",
    )


def _forbidden_board(tmp_path):
    board = _board_from_payload(tmp_path, _good_runs())
    return PullRequestChecks(**{**board.__dict__, "branch_protection_status": "forbidden"})


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_forbidden_protection_ready_pr_with_same_head_clean_finalizes(tmp_path, monkeypatch, auto_merge):
    effects = _install(monkeypatch, _passed(_forbidden_board(tmp_path)), mergeability=_mergeability("CLEAN"))
    assert run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0
    assert effects.mergeability_reads == 1
    assert bool(effects.merges) is auto_merge


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
@pytest.mark.parametrize("state,head", [("BLOCKED", H), ("CLEAN", "other")])
def test_forbidden_protection_without_same_head_clean_refuses(tmp_path, monkeypatch, auto_merge, state, head):
    effects = _install(monkeypatch, _passed(_forbidden_board(tmp_path)), mergeability=_mergeability(state, head))
    with pytest.raises(AgentLoopError, match=r"cannot finalize.*unreadable"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge))
    assert not (effects.merges or effects.prepares or effects.publishes)


@pytest.mark.parametrize("auto_merge", [True, False], ids=["auto-merge", "manual"])
def test_draft_under_forbidden_protection_stops_side_effect_free_then_readable_resume_finalizes(
    tmp_path, monkeypatch, auto_merge
):
    items = _items()
    approval = structured_pr_review(
        reviewer="OpenAI Codex", state="approved",
        prior_item_dispositions=[{"item_id": i.item_id, "disposition": "resolved"} for i in items],
    )
    runner = _runner(items)
    runner.codex_outputs.append(approval)  # scripted second approval for the resumed run
    effects = _install(monkeypatch, _passed(_forbidden_board(tmp_path)), mergeability=_mergeability("DRAFT"))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=auto_merge))
    message = str(excinfo.value)
    assert "administration: read" in message and "Do not mark the PR ready manually" in message
    assert not (effects.merges or effects.prepares or effects.publishes)
    assert not any(cmd[:3] == ["gh", "pr", "ready"] for cmd, _cwd in runner.commands)
    assert not any(
        "--add-label" in cmd or "--remove-label" in cmd or "/labels" in " ".join(cmd)
        for cmd, _cwd in runner.commands
    )

    # Every persisted copy of the item (reviewer round records included) is unchanged.
    assert {(i.text, i.lifecycle, i.failed_head_sha) for i in _checks_items(runner)} == {
        (PERSISTED_TEXT, "qualification_ready", "old-head")
    }

    # Resume the SAME refused PR (its persisted round metadata) once protection is readable.
    readable = _board_from_payload(tmp_path, _good_runs(), protection={"contexts": []})
    effects = _install(monkeypatch, _passed(readable))
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0
    assert effects.mergeability_reads == 0
    assert bool(effects.merges) is auto_merge
    assert bool(effects.publishes) is (not auto_merge)


# --- timeout routing ------------------------------------------------------------


def _timeout_board(tmp_path, runs, **kwargs):
    return _board_from_payload(tmp_path, runs, **kwargs)


def _timeout(board, head=H):
    return ManagedCiOutcome(status="timeout", checks=board, head_sha=head)


LEGACY = "Managed exact-head CI for PR #77 did not pass within"


def test_timeout_with_pending_or_missing_names_them_after_legacy_sentence(tmp_path, monkeypatch):
    runs = [_run(FINAL_CONTEXT), {"id": 5, "name": "slow", "status": "in_progress", "conclusion": None}]
    board = _timeout_board(tmp_path, runs, protection={"contexts": ["needed"]})
    effects = _install(monkeypatch, _timeout(board))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))
    message = str(excinfo.value)
    assert message.startswith(LEGACY)
    assert f"Exact-head board at {H}: final context success; pending: slow; required not reporting: needed." in message
    assert not (effects.merges or effects.publishes)


@pytest.mark.parametrize("variant", ["errors", "head"])
def test_timeout_with_errors_or_head_mismatch_keeps_legacy_message(tmp_path, monkeypatch, variant):
    board = _timeout_board(tmp_path, [_run(FINAL_CONTEXT), _run("lint", "failure", 4)])
    head = H
    if variant == "errors":
        board = PullRequestChecks(**{**board.__dict__, "check_query_errors": ("boom",)})
    else:
        head = "other"
    _install(monkeypatch, _timeout(board, head))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert str(excinfo.value).endswith(f"{LEGACY} {_config(tmp_path, auto_merge=True).ci_timeout_seconds}s.")


def test_timeout_with_nonfinal_failure_routes_to_repair_without_merge(tmp_path, monkeypatch):
    board = _timeout_board(tmp_path, [_run(FINAL_CONTEXT), _run("lint", "failure", 4)])
    runner = _runner(_items())
    effects = _install(monkeypatch, _timeout(board))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))

    # Not the generic timeout refusal: the failure was posted and a repair round was entered.
    assert LEGACY not in str(excinfo.value)
    assert any("Failing checks: lint" in comment for comment in runner.comments)
    assert not (effects.merges or effects.prepares or effects.publishes)


@pytest.mark.parametrize("conclusion", ["skipped", "neutral"])
def test_required_check_without_real_success_refuses_naming_it(tmp_path, monkeypatch, conclusion):
    board = _board_from_payload(
        tmp_path, [_run(FINAL_CONTEXT), _run("lint", conclusion, 2)], protection={"contexts": ["lint"]}
    )
    effects = _install(monkeypatch, _passed(board))
    with pytest.raises(AgentLoopError, match=r"cannot finalize.*`lint`.*"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert not (effects.merges or effects.prepares or effects.publishes)


def test_timeout_repair_dispatches_coder_persists_repair_required_and_extends_budget(tmp_path, monkeypatch):
    board = _timeout_board(tmp_path, [_run(FINAL_CONTEXT), _run("lint", "failure", 4)])
    items = _items()
    runner = _runner(items, claude_outputs=[structured_coder_followup(addressed_items=[items[0].item_id])])
    effects = _install(monkeypatch, _timeout(board))
    with pytest.raises(AgentLoopError) as excinfo:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))

    # The coder ran (claude consumed its scripted turn); the next reviewer round then began,
    # which is only possible because the one-round budget extension applied (max_rounds=1).
    assert "claude" in [c[0] for c, _cwd in runner.commands if c]
    assert "scripted agent output exhausted" in str(excinfo.value)
    assert LEGACY not in str(excinfo.value) and "review budget" not in str(excinfo.value)
    assert not (effects.merges or effects.prepares or effects.publishes)
    repaired = [
        item for item in _checks_items(runner)
        if item.lifecycle == "repair_required" and item.failed_head_sha == H
    ]
    assert repaired and "Failing checks: lint" in repaired[-1].text
    assert any("Failing checks: lint" in comment for comment in runner.comments)


def test_timeout_pending_or_missing_retains_obligation_without_persisting_change(tmp_path, monkeypatch):
    runs = [_run(FINAL_CONTEXT), {"id": 5, "name": "slow", "status": "in_progress", "conclusion": None}]
    board = _timeout_board(tmp_path, runs)
    runner = _runner(_items())
    effects = _install(monkeypatch, _timeout(board))
    with pytest.raises(AgentLoopError, match="pending: slow"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert not (effects.merges or effects.prepares or effects.publishes)
    assert all(
        item.text == PERSISTED_TEXT and item.lifecycle == "qualification_ready"
        for item in _checks_items(runner)
    )


# ---- #1293: workflow_dispatch exclusion keeps the supersession guards strict ----

from test_ci_health import (  # noqa: E402
    SUITE_DISPATCH,
    _cr,
    _dispatch,
    _history,
    _listing,
)
from test_managed_supersession import _decide  # noqa: E402


def _scoped_board(tmp_path, default_runs, history_runs, *, protection=None, **kwargs):
    stub = _StubGhRunner(
        check_runs_payload=_listing(*default_runs),
        status_payload={
            "state": "success",
            "total_count": 1,
            "statuses": [{"context": FINAL_CONTEXT, "state": "success"}],
        },
        branch_protection_payload=protection or {"contexts": []},
        dispatch_payload=_dispatch(),
        history_stdout=_history(*history_runs),
        **kwargs,
    )
    return get_pr_checks(stub, config=make_config(tmp_path), metadata=_metadata(), now=NOW)


def test_independent_suite_failure_is_shadowed_and_unqualified(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    newer = _cr(8, "x", "success", run=5000)
    independent = _cr(3, "x", "failure", suite=555, run=5001)
    board = _scoped_board(tmp_path, [dispatch_x, newer], [dispatch_x, newer, independent])
    assert [c.check_id for c in board.shadowed] == [3]
    assert _decide(board)[0] == "unqualified"


def test_identity_less_failure_is_shadowed_and_unqualified(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    newer = _cr(8, "x", "success", suite=None, app=None)
    older = _cr(3, "x", "failure", suite=None, app=None)
    board = _scoped_board(tmp_path, [dispatch_x, newer], [dispatch_x, newer, older])
    assert [c.check_id for c in board.shadowed] == [3]
    assert _decide(board)[0] == "unqualified"


def test_required_check_with_non_success_shadowed_observation_is_unqualified(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    newer = _cr(8, "x", "success", run=5000)
    independent = _cr(3, "x", "failure", suite=555, run=5001)
    board = _scoped_board(
        tmp_path, [dispatch_x, newer], [dispatch_x, newer, independent], protection={"contexts": ["x"]}
    )
    assert _decide(board)[0] == "unqualified"


def test_unavailable_history_is_unqualified(tmp_path):
    dispatch_x = _cr(9, "x", "success", suite=SUITE_DISPATCH, run=7000)
    y = _cr(4, "y", "success", run=5000)
    board = _scoped_board(tmp_path, [dispatch_x, y], [], history_returncode=1)
    assert board.state == "unavailable"
    assert _decide(board)[0] == "unqualified"
