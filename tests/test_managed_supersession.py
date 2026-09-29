"""Managed success supersession of carried ordinary check obligations (#1117)."""
from dataclasses import replace

import pytest

from coding_review_agent_loop.ci_health import PullRequestCheck, PullRequestChecks
from coding_review_agent_loop.github import PullRequestMergeability
from coding_review_agent_loop.managed_ci import FINAL_CONTEXT, ManagedCiOutcome
from coding_review_agent_loop.orchestrator import _managed_success_supersedes_ordinary_checks
from coding_review_agent_loop.unresolved_items import (
    _advance_machine_obligations_for_head,
    _upsert_machine_obligation,
)

H = "head-h"
F = "head-f"


def _chk(name, status="success"):
    return PullRequestCheck(name=name, kind="check_run", status=status)


def _board(**overrides):
    fields = dict(
        state="passing",
        required_checks=(),
        passing=(_chk(FINAL_CONTEXT), _chk("lint")),
        pending=(),
        failing=(),
        missing_required=(),
        branch_protection_status="not_found",
        listing_complete=True,
    )
    fields.update(overrides)
    return PullRequestChecks(**fields)


def _outcome(board, head=H):
    return ManagedCiOutcome(status="passed", checks=board, head_sha=head)


def _items(kind="github-pr-checks", *, candidate=H):
    minted = _upsert_machine_obligation(
        [],
        item_number=10,
        kind=kind,
        source_round=3,
        text="Failing checks: Python 3.12 full suite (failure)",
        failed_head_sha=F,
    )
    return _advance_machine_obligations_for_head(minted, current_head_sha=candidate)


def _decide(board, *, items=None, head=H, outcome_head=H, mergeability=None):
    return _managed_success_supersedes_ordinary_checks(
        _items() if items is None else items,
        outcome=_outcome(board, outcome_head),
        current_head_sha=head,
        mergeability=mergeability,
    )


def test_clean_board_clears_even_when_former_failing_check_is_absent():
    assert _decide(_board()) == ("cleared", "")


def test_former_check_skipped_or_neutral_does_not_block():
    board = _board(passing=(_chk(FINAL_CONTEXT), _chk("Python 3.12 full suite", "skipped")))
    assert _decide(board)[0] == "cleared"


def test_former_check_still_failing_blocks():
    board = _board(state="failing", failing=(_chk("Python 3.12 full suite", "failure"),))
    verdict, predicate = _decide(board)
    assert verdict == "unqualified"
    assert "Python 3.12 full suite" in predicate


def test_no_candidate_is_not_applicable():
    assert _decide(_board(), items=[])[0] == "not_applicable"


def test_repair_required_at_own_failed_head_is_not_cleared():
    minted = _upsert_machine_obligation(
        [], item_number=1, kind="github-pr-checks", source_round=1, text="x", failed_head_sha=H
    )
    assert _decide(_board(), items=minted)[0] == "not_applicable"


@pytest.mark.parametrize("kind", ["managed-exact-head-ci"])
def test_other_machine_kinds_are_not_touched(kind):
    assert _decide(_board(), items=_items(kind))[0] == "not_applicable"


def test_head_mismatch_and_missing_board_are_unqualified():
    assert _decide(_board(), outcome_head="other")[0] == "unqualified"
    result = _managed_success_supersedes_ordinary_checks(
        _items(), outcome=ManagedCiOutcome(status="passed", checks=None, head_sha=H),
        current_head_sha=H, mergeability=None,
    )
    assert result[0] == "unqualified"


def test_query_errors_with_status_ok_are_unqualified_and_named():
    verdict, predicate = _decide(_board(check_query_errors=("boom",)))
    assert verdict == "unqualified" and "boom" in predicate
    assert _decide(_board(check_query_status="partial"))[0] == "unqualified"


def test_incomplete_listing_is_unqualified():
    verdict, predicate = _decide(_board(listing_complete=False))
    assert verdict == "unqualified" and "incomplete" in predicate


def test_final_context_missing_or_not_success_is_unqualified():
    assert _decide(_board(passing=(_chk("lint"),)))[0] == "unqualified"
    assert _decide(_board(passing=(_chk(FINAL_CONTEXT, "skipped"),)))[0] == "unqualified"


def test_required_check_skipped_or_neutral_is_unqualified():
    board = _board(
        required_checks=("lint",),
        passing=(_chk(FINAL_CONTEXT), _chk("lint", "skipped")),
    )
    verdict, _ = _decide(board)
    assert verdict == "unqualified"


def test_missing_required_and_pending_are_unqualified():
    assert _decide(_board(missing_required=("lint",)))[0] == "unqualified"
    assert _decide(_board(state="pending", pending=(_chk("slow", "queued"),)))[0] == "unqualified"


def test_shadowed_failure_blocks_and_names_check():
    verdict, predicate = _decide(_board(shadowed=(_chk("lint", "failure"),)))
    assert verdict == "unqualified" and "lint" in predicate


def test_shadowed_skipped_of_required_blocks_but_non_required_does_not():
    required = _board(required_checks=("lint",), shadowed=(_chk("lint", "skipped"),))
    assert _decide(required)[0] == "unqualified"
    optional = _board(shadowed=(_chk("lint", "skipped"),))
    assert _decide(optional)[0] == "cleared"


def _mergeability(state, head=H):
    return PullRequestMergeability(
        state="mergeable", mergeable_raw="MERGEABLE", merge_state_raw=state,
        head_sha=head, base_branch="main",
    )


def test_forbidden_protection_needs_same_head_clean():
    board = _board(branch_protection_status="forbidden")
    assert _decide(board, mergeability=_mergeability("CLEAN"))[0] == "cleared"
    assert _decide(board, mergeability=_mergeability("CLEAN", head="other"))[0] == "unqualified"
    assert _decide(board, mergeability=_mergeability("BLOCKED"))[0] == "unqualified"
    assert _decide(board, mergeability=None)[0] == "unqualified"


def test_draft_under_forbidden_protection_gets_actionable_predicate():
    board = _board(branch_protection_status="forbidden")
    verdict, predicate = _decide(board, mergeability=_mergeability("DRAFT"))
    assert verdict == "unqualified"
    assert "administration: read" in predicate
    assert "Do not mark the PR ready manually" in predicate
