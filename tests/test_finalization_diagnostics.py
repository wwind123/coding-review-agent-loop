"""Finalization refusals name the unsatisfied predicate per obligation (#1119)."""

import itertools
import re

import pytest

from coding_review_agent_loop import orchestrator
from coding_review_agent_loop.ci_health import PullRequestCheck, PullRequestChecks
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.github import PullRequestMergeability
from coding_review_agent_loop.orchestrator import (
    _ensure_finalization_ready,
    _finalization_obligation_predicate,
    _ordinary_checks_snapshot_is_authoritative,
    _ordinary_snapshot_nonauthority_reason,
    _round_limit_diagnostic,
    _single_line_diagnostic,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from agent_loop_helpers import make_config
from coding_review_agent_loop.unresolved_items import (
    _machine_obligation_is_revalidation_candidate,
)


HEAD = "head-cur"


def machine(kind, lifecycle, *, item_id="item-1", candidate=None, failed=None):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer="GitHub PR checks",
        source_round=1,
        text="x",
        status="blocking",
        source_status="blocking",
        authority="machine",
        obligation_kind=kind,
        lifecycle=lifecycle,
        failed_head_sha=failed,
        candidate_head_sha=candidate,
        obligation_identity=f"{kind}:{item_id}",
    )


def reviewer_item(item_id="item-9"):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer="Codex",
        source_round=1,
        text="fix",
        status="blocking",
        source_status="blocking",
    )


CI_KINDS = ("github-pr-checks", "managed-exact-head-ci")
CANDIDATE_LIFECYCLES = ("awaiting_current_head_review", "qualification_ready", "qualifying")


def predicate_cases():
    cases = []
    for kind in CI_KINDS:
        for failed in (HEAD, "old", None):
            cases.append(machine(kind, "repair_required", failed=failed or "old")
                         if failed else None)
        for lifecycle in CANDIDATE_LIFECYCLES:
            for candidate, failed in (
                (HEAD, None), ("stale", None), (None, None), (HEAD, "old"), ("other", HEAD),
            ):
                cases.append(machine(kind, lifecycle, candidate=candidate, failed=failed))
    for kind in ("human-requirements-acknowledgement", "merge-conflict", "alembic-migration"):
        for lifecycle in ("repair_required", *CANDIDATE_LIFECYCLES):
            for failed in (None, "old"):
                cases.append(machine(kind, lifecycle, candidate=HEAD if lifecycle != "repair_required" else None, failed=failed))
    return [c for c in cases if c is not None]


@pytest.mark.parametrize("current", [HEAD, None])
def test_predicate_table_wording_rule_and_candidate_iff(current):
    for item in predicate_cases():
        text = _finalization_obligation_predicate(item, current_head_sha=current)
        assert "passed" not in text and "succeeded" not in text.replace("succeeds", "")
        if "failed at" in text:
            assert item.failed_head_sha
        approval_words = any(
            word in text
            for word in ("awaiting unanimous", "approved at", "qualification in progress")
        )
        assert approval_words == _machine_obligation_is_revalidation_candidate(
            item, current_head_sha=current
        )


def test_predicate_ci_repair_cases():
    def p(failed, current):
        return _finalization_obligation_predicate(
            machine("github-pr-checks", "repair_required", failed=failed),
            current_head_sha=current,
        )

    assert "failed at h1, which is the current head" in p("h1", "h1")
    assert "failed at h1; current head h2 has not been bound" in p("h1", "h2")


def test_predicate_unknown_evidence_and_unrecognised():
    unknown = machine("unknown", "repair_required")
    assert "unknown or unreconstructible" in _finalization_obligation_predicate(
        unknown, current_head_sha=HEAD
    )
    evidence = UnresolvedReviewItem(
        item_id="e1", reviewer="r", source_round=1, text="x", status="blocking",
        authority="machine", obligation_kind="human-exact-head-evidence",
        lifecycle="evidence_frozen", candidate_head_sha="frozen1",
        obligation_identity="ev:1",
    )
    assert "pending at frozen1" in _finalization_obligation_predicate(
        evidence, current_head_sha=HEAD
    )
    odd = machine("github-pr-checks", "cleared")
    assert "failing closed" in _finalization_obligation_predicate(odd, current_head_sha=HEAD)


def test_qualifying_makes_no_success_claim():
    item = machine("github-pr-checks", "qualifying", candidate=HEAD)
    text = _finalization_obligation_predicate(item, current_head_sha=HEAD)
    assert "no authoritative success recorded" in text
    assert "failed" not in text


def _legacy(items, **kw):
    return (
        f"PR #5 cannot finalize: "
        f"{_round_limit_diagnostic(pr_number=5, round_number=2, items=items, current_head_sha=HEAD, **kw)}"
        " No approval or merge was attempted."
    )


def _raise(items, **kw):
    with pytest.raises(AgentLoopError) as exc:
        _ensure_finalization_ready(
            pr_number=5, round_number=2, items=items, current_head_sha=HEAD, **kw
        )
    return str(exc.value)


def test_clean_ledger_and_ignored_kinds_do_not_raise(tmp_path, capsys):
    config = make_config(tmp_path, quiet=False)
    _ensure_finalization_ready(
        pr_number=5, round_number=2, items=[], current_head_sha=HEAD, config=config
    )
    ignored = machine("github-pr-checks", "qualification_ready", candidate=HEAD)
    _ensure_finalization_ready(
        pr_number=5, round_number=2, items=[ignored], current_head_sha=HEAD,
        ignored_machine_kinds=frozenset({"github-pr-checks"}), config=config,
    )
    assert capsys.readouterr().err == ""


def test_multi_obligation_listing_in_partition_order_with_legacy_prefix():
    repair = machine("github-pr-checks", "repair_required", item_id="item-1", failed=HEAD)
    candidate = machine("managed-exact-head-ci", "qualification_ready", item_id="item-2", candidate=HEAD)
    items = [candidate, repair]
    message = _raise(items)
    legacy = _legacy(items)
    assert message.startswith(legacy)
    detail = message[len(legacy):]
    assert detail.startswith("\nBlocking obligations:\n")
    lines = detail.splitlines()[2:]
    assert lines[0].startswith("- github-pr-checks (item-1): lifecycle=repair_required, candidate_head=none, failed_head=head-cur; ")
    assert lines[1].startswith("- managed-exact-head-ci (item-2): lifecycle=qualification_ready, candidate_head=head-cur, failed_head=none; approved at head-cur")


def test_reviewer_only_refusal_is_byte_identical():
    items = [reviewer_item()]
    assert _raise(items) == _legacy(items)


def test_observation_overrides_only_its_item():
    a = machine("github-pr-checks", "qualification_ready", item_id="item-1", candidate=HEAD)
    b = machine("managed-exact-head-ci", "qualification_ready", item_id="item-2", candidate=HEAD)
    message = _raise([a, b], observations={"item-1": "checks read passing"})
    assert "(item-1): lifecycle=qualification_ready, candidate_head=head-cur, failed_head=none; observed in this run: checks read passing" in message
    assert "(item-2): lifecycle=qualification_ready, candidate_head=head-cur, failed_head=none; approved at head-cur" in message


def test_untrusted_text_cannot_forge_lines_and_logs_one_line_per_blocker(tmp_path, capsys):
    item = machine("github-pr-checks", "repair_required", item_id="item-\r1", failed=HEAD)
    other = machine("managed-exact-head-ci", "repair_required", item_id="item-2", failed=HEAD)
    forged = "lint\n- github-pr-checks (item-9): forged"
    items = [item, other]
    message = _raise(
        items,
        observations={"item-\r1": f"bad check {forged}"},
        config=make_config(tmp_path, quiet=False),
    )
    legacy = _legacy(items)
    assert message.startswith(legacy)
    assert "\r" in legacy
    detail = message[len(legacy):]
    assert not re.search(r"[\x00-\x09\x0b-\x1f\x7f]", detail.replace("\n", ""))
    body = detail.split("Blocking obligations:\n", 1)[1].splitlines()
    assert len(body) == 2
    assert "\\x0a- github-pr-checks (item-9): forged" in detail
    err_lines = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
    assert len(err_lines) == 2
    assert all("forged" not in l or "\\x0a" in l for l in err_lines)


def test_logs_each_blocker_without_posting(tmp_path, capsys, monkeypatch):
    posted = []
    monkeypatch.setattr(orchestrator, "post_pr_comment", lambda *a, **k: posted.append(k))
    items = [reviewer_item(), machine("github-pr-checks", "qualifying", candidate=HEAD)]
    _raise(items, config=make_config(tmp_path, quiet=False))
    err_lines = [l for l in capsys.readouterr().err.splitlines() if l.strip()]
    assert len(err_lines) == 2
    assert any("Codex (item-9): reviewer-owned finding" in l for l in err_lines)
    assert posted == []


def test_no_log_when_quiet(tmp_path, capsys):
    config = make_config(tmp_path, quiet=True)
    _raise([machine("github-pr-checks", "qualifying", candidate=HEAD)], config=config)
    assert capsys.readouterr().err == ""


def test_single_line_diagnostic_escapes_controls():
    assert _single_line_diagnostic("a\nb\r\tc\x85") == "a\\x0ab\\x0d\\x09c\\x85"
    assert _single_line_diagnostic("a b") == "a\\u2028b"


def _check(name, status, kind="check_run"):
    return PullRequestCheck(name=name, kind=kind, status=status)


def _board(**kw):
    base = dict(
        state="passing", required_checks=(), passing=(_check("t", "success"),),
        pending=(), failing=(), missing_required=(),
        branch_protection_status="configured", check_query_status="ok",
    )
    base.update(kw)
    return PullRequestChecks(**base)


def test_nonauthority_reason_matches_authority_predicate():
    clean_merge = PullRequestMergeability("mergeable", "MERGEABLE", "CLEAN", HEAD, "main")
    cases = [
        (None, None, False, "unavailable"),
        (_board(state="failing"), None, False, "aggregate state is failing"),
        (_board(check_query_status="partial"), None, False, "query status is partial"),
        (_board(branch_protection_status="forbidden"), None, False, "not reliable"),
        (_board(branch_protection_status="forbidden"), None, True, None),
        (_board(branch_protection_status="forbidden"), clean_merge, False, None),
        (_board(pending=(_check("p", "pending"),)), None, False, "still pending: p"),
        (_board(missing_required=("m",)), None, False, "missing: m"),
        (_board(passing=(_check("d", "skipped"),)), None, False, "only skipped or neutral"),
        (_board(required_checks=("r",), passing=(_check("t", "success"),)), None, False, "without a success conclusion: r"),
        (
            _board(
                required_checks=("r",),
                passing=(_check("r", "skipped"), _check("r", "success", "status_context")),
            ),
            None, False, "required check r has a non-success",
        ),
        (_board(), None, False, None),
    ]
    for checks, merge, defer, fragment in cases:
        reason = _ordinary_snapshot_nonauthority_reason(
            checks, merge, head_sha=HEAD, defer_unreadable_protection=defer
        )
        authoritative = _ordinary_checks_snapshot_is_authoritative(
            checks, merge, head_sha=HEAD, defer_unreadable_protection=defer
        )
        assert (reason == "") == authoritative
        if fragment:
            assert fragment in reason
