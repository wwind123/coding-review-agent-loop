"""run_pr_loop coverage for managed-success supersession and timeout routing (#1117)."""
import json

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.ci_health import PullRequestCheck, PullRequestChecks
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import PullRequestMergeability, get_pr_checks
from coding_review_agent_loop.managed_ci import FINAL_CONTEXT, ManagedCiContract, ManagedCiOutcome
from coding_review_agent_loop.orchestrator import run_pr_loop

from agent_loop_helpers import FakeRunner, make_config, structured_pr_review
from test_ci_health import NOW, _StubGhRunner, _metadata, _run
from test_orchestrator_pr import _carried_ci_obligations, _carried_ci_review_comment

H = "abc123"


def _items():
    return (_carried_ci_obligations()[1],)  # github-pr-checks, failed old-head, candidate H


def _runner(items):
    return FakeRunner(
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

    assert run_pr_loop(_runner(items), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0

    assert bool(effects.merges) is auto_merge
    assert bool(effects.publishes) is (not auto_merge)
    assert effects.mergeability_reads == 0


def test_query_error_board_refuses_naming_predicate_without_finalizing(tmp_path, monkeypatch):
    items = _items()
    board = _board_from_payload(tmp_path, _good_runs())
    board = PullRequestChecks(**{**board.__dict__, "check_query_errors": ("boom",)})
    effects = _install(monkeypatch, _passed(board))

    with pytest.raises(AgentLoopError, match=r"cannot finalize.*github-pr-checks.*boom"):
        run_pr_loop(_runner(items), pr_number=77, config=_config(tmp_path, auto_merge=True))

    assert not (effects.merges or effects.prepares or effects.publishes)


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
        return
    assert board.state == "passing" and len(board.shadowed) == 1
    effects = _install(monkeypatch, _passed(board))

    with pytest.raises(AgentLoopError, match=r"cannot finalize.*`X`"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))

    assert not (effects.merges or effects.prepares or effects.publishes)


def test_shadowed_skipped_required_check_refuses_but_optional_clears(tmp_path, monkeypatch):
    runs = [_run(FINAL_CONTEXT), _run("X", "success", 2), _run("X", "skipped", 3)]
    required = _board_from_payload(
        tmp_path, runs, protection={"contexts": ["X"]}
    )
    effects = _install(monkeypatch, _passed(required))
    with pytest.raises(AgentLoopError, match=r"cannot finalize.*`X`"):
        run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True))
    assert not effects.merges

    optional = _board_from_payload(tmp_path, runs)
    effects = _install(monkeypatch, _passed(optional))
    assert run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=True)) == 0
    assert effects.merges


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total": 5},
        {"total": False},
        {"status_total": 4},
        {"payload_override": {"check_runs": [_run(FINAL_CONTEXT), "junk"], "total_count": 2}},
        {"payload_override": {"check_runs": [_run(FINAL_CONTEXT), {"id": 9, "status": "completed"}], "total_count": 2}},
    ],
    ids=["truncated", "missing-total", "status-truncated", "non-object", "nameless"],
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
    runner = _runner(_items())
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

    # Resume once protection is readable: the same persisted item now clears.
    readable = _board_from_payload(tmp_path, _good_runs(), protection={"contexts": []})
    effects = _install(monkeypatch, _passed(readable))
    assert run_pr_loop(_runner(_items()), pr_number=77, config=_config(tmp_path, auto_merge=auto_merge)) == 0
    assert effects.mergeability_reads == 0
    assert bool(effects.merges) is auto_merge


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
