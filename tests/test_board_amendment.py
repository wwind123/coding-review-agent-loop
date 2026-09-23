"""Signed reviewer-board amendment records (#943): parser, chain, lineage, ledger."""

import json
from types import SimpleNamespace

import pytest

from coding_review_agent_loop.board_amendment import (
    ContractLineage,
    amend_contract,
    apply_board_amendment_to_ledger,
    collect_reviewer_board_amendments,
    format_reviewer_board_amendment_comment,
    parse_reviewer_board_amendment_records,
    reject_misplaced_pr_amendments,
    require_amendment_activation,
    resolve_contract_lineage,
    reviewer_board_amendment_payload,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.plan_review_scheduling import (
    PlanReviewSchedulingContract,
    make_plan_contract,
)
from coding_review_agent_loop.protocol import UnresolvedReviewItem
from coding_review_agent_loop.review_scheduling import make_contract
from coding_review_agent_loop.round_state import PostedRoundMetadata, PostedRoundRecord

BOARD = ("Codex", "Claude", "Antigravity")
C0 = make_plan_contract(BOARD, "primary-then-panel", "Codex")
C1 = make_plan_contract(("Codex", "Claude"), "primary-then-panel", "Codex")


def _comment(body):
    return SimpleNamespace(body=body)


def _amendment_body(**overrides):
    values = {
        "flow": "plan",
        "issue": 942,
        "pr_number": None,
        "original_required_reviewers": BOARD,
        "policy": "primary-then-panel",
        "primary_reviewer": "Codex",
        "removed_reviewers": ("Antigravity",),
        "effective_from_round": 3,
        "rationale": "Antigravity weekly quota exhausted.",
    }
    values.update(overrides)
    return format_reviewer_board_amendment_comment(**values)


def _plan_amendments(comments, issue=942):
    return collect_reviewer_board_amendments(comments, flow="plan", issue_number=issue)


def _checkpoint(index, round_number, contract, digest=None):
    return PostedRoundRecord(
        index=index,
        body="",
        metadata=PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=round_number,
            subject="plan",
            phase="scheduler-prelaunch",
            scheduler_contract=contract.as_dict(),
            scheduler_obligation_digest="0" * 16,
            scheduler_selected_reviewers=("Codex",),
            scheduler_reasons=("primary phase",),
            scheduler_final_sweep=False,
            scheduler_force_full=False,
            scheduler_calls_avoided=0,
            reviewer_board_amendment_digest=digest,
        ),
    )


def _neutral(index, round_number, role="reviewer", digest=None):
    return PostedRoundRecord(
        index=index,
        body="",
        metadata=PostedRoundMetadata(
            flow="plan",
            role=role,
            agent="Claude" if role == "reviewer" else "Anthropic Claude",
            round_number=round_number,
            subject="plan",
            reviewer_board_amendment_digest=digest,
        ),
    )


def _plan_contract(metadata):
    if metadata.scheduler_contract is None:
        return None
    return PlanReviewSchedulingContract.from_mapping(metadata.scheduler_contract)


def _drift(persisted, detail):
    return AgentLoopError(f"scheduler contract changed during resume: {detail}")


def _resolve(records, amendments, configured):
    return resolve_contract_lineage(
        records,
        amendments,
        configured,
        contract_from_metadata=_plan_contract,
        drift_error=_drift,
    )


# --- parser, digest, discovery (row record-invalid) -----------------------


def test_signed_record_parses_with_canonical_digest():
    body = _amendment_body()
    records, ignored = parse_reviewer_board_amendment_records(body, comment_locator="c1")
    assert ignored == ()
    (record,) = records
    assert record.removed_reviewers == ("Antigravity",)
    assert record.amended_required_reviewers == ("Codex", "Claude")
    assert len(record.digest) == 64
    # The digest is canonical: key order and whitespace do not matter.
    payload = json.loads(body.split("```json\n", 1)[1].split("\n```", 1)[0])
    reordered = json.dumps(dict(reversed(list(payload.items()))))
    again, _ = parse_reviewer_board_amendment_records(
        f"```json\n{reordered}\n```\n-- Human Reviewer", comment_locator="c2"
    )
    assert again[0].digest == record.digest


def test_unsigned_record_is_ignored_and_malformed_is_reported():
    unsigned = _amendment_body().replace("-- Human Reviewer", "-- Someone")
    assert parse_reviewer_board_amendment_records(unsigned, comment_locator="c") == ((), ())
    payload = reviewer_board_amendment_payload(
        flow="plan",
        issue=942,
        pr_number=None,
        original_required_reviewers=BOARD,
        policy="primary-then-panel",
        primary_reviewer="Codex",
        removed_reviewers=("Antigravity",),
        effective_from_round=0,
        rationale="quota",
    )
    malformed = f"```json\n{json.dumps(payload)}\n```\n-- Human Reviewer"
    records, ignored = parse_reviewer_board_amendment_records(malformed, comment_locator="c")
    assert records == ()
    assert "effective_from_round must be a positive integer" in ignored[0]
    diagnostics = []
    assert collect_reviewer_board_amendments(
        [_comment(malformed)], flow="plan", issue_number=942, ignored_sink=diagnostics
    ) == ()
    assert diagnostics


def test_unknown_reason_and_flow_scoping_fields_are_malformed():
    for overrides in (
        {"reason": "operator-preference"},
        {"pr_number": 7},
    ):
        body = _amendment_body(**overrides)
        records, ignored = parse_reviewer_board_amendment_records(body, comment_locator="c")
        assert records == () and ignored


def test_identical_duplicates_collapse_and_conflicts_fail_closed():
    body = _amendment_body()
    amendments = _plan_amendments([_comment(body), _comment(body)])
    assert len(amendments) == 1
    conflicting = _plan_amendments(
        [_comment(body), _comment(_amendment_body(removed_reviewers=("Claude",)))]
    )
    with pytest.raises(AgentLoopError, match="never chooses"):
        _resolve([_checkpoint(0, 1, C0)], conflicting, C1)


def test_wrong_issue_wrong_flow_and_wrong_pr_fail_closed():
    with pytest.raises(AgentLoopError, match="names issue #941"):
        _plan_amendments([_comment(_amendment_body(issue=941))])
    pr_body = _amendment_body(flow="pr", issue=None, pr_number=17)
    with pytest.raises(AgentLoopError, match="Post this record on PR #17"):
        _plan_amendments([_comment(pr_body)])
    with pytest.raises(AgentLoopError, match="names PR #17"):
        collect_reviewer_board_amendments([_comment(pr_body)], flow="pr", pr_number=18)
    with pytest.raises(AgentLoopError, match="Post this record on issue #942"):
        collect_reviewer_board_amendments([_comment(_amendment_body())], flow="pr", pr_number=18)
    with pytest.raises(AgentLoopError, match="posted on the owning issue"):
        reject_misplaced_pr_amendments([_comment(pr_body)], issue_number=942)
    # A plan amendment on the issue is not a misplaced PR amendment.
    reject_misplaced_pr_amendments([_comment(_amendment_body())], issue_number=942)


def test_removing_primary_or_leaving_no_secondary_fails_closed():
    (removes_primary,) = _plan_amendments(
        [_comment(_amendment_body(removed_reviewers=("Codex",)))]
    )
    with pytest.raises(AgentLoopError, match="primary is immutable"):
        amend_contract(C0, removes_primary)
    (no_secondary,) = _plan_amendments(
        [_comment(_amendment_body(removed_reviewers=("Claude", "Antigravity")))]
    )
    with pytest.raises(AgentLoopError, match="at least one secondary"):
        amend_contract(C0, no_secondary)
    (wrong_policy,) = _plan_amendments([_comment(_amendment_body(policy="all-reviewers"))])
    with pytest.raises(AgentLoopError, match="persisted policy"):
        amend_contract(C0, wrong_policy)
    (unknown,) = _plan_amendments([_comment(_amendment_body(removed_reviewers=("Gemini",)))])
    with pytest.raises(AgentLoopError, match="not on the persisted board"):
        amend_contract(C0, unknown)


def test_pr_contract_amendment_keeps_policy_primary_and_broad_rules():
    original = make_contract(BOARD, "primary-then-panel", None, "Codex")
    (amendment,) = collect_reviewer_board_amendments(
        [_comment(_amendment_body(flow="pr", issue=None, pr_number=17))],
        flow="pr",
        pr_number=17,
    )
    amended = amend_contract(original, amendment)
    assert amended.required_reviewers == ("Codex", "Claude")
    assert amended.broad_rules == original.broad_rules
    assert amended.broad_rules_digest == original.broad_rules_digest
    assert amended.primary_reviewer == "Codex"


# --- lineage (rows digest-binding, chain-link-interval, post-amend-old-board)


def test_no_amendment_is_the_historical_immutability_rule():
    lineage = _resolve([_checkpoint(0, 1, C0)], (), C0)
    assert lineage.active_amendment is None
    with pytest.raises(AgentLoopError, match="configured contract does not match"):
        _resolve([_checkpoint(0, 1, C0)], (), C1)


def test_contract_neutral_records_never_drift_and_amendment_rescues_legacy_history():
    comments = [_comment("x")] * 5 + [_comment(_amendment_body(effective_from_round=2))]
    amendments = _plan_amendments(comments)
    records = [
        _neutral(0, 1, role="coder"),
        _checkpoint(1, 1, C0),
        _neutral(2, 1),
        _checkpoint(3, 2, C0),  # the old checkpoint of round N, pre-amendment
        _neutral(4, 2),
        _checkpoint(6, 2, C1, digest=amendments[0].digest),
        _neutral(7, 2, role="coder"),
    ]
    lineage = _resolve(records, amendments, C1)
    assert lineage.contracts == (C0, C1)
    assert lineage.pending_amendments == ()
    assert lineage.removed_reviewers == ("Antigravity",)


def test_post_amendment_record_needs_amended_contract_and_exact_digest():
    comments = [_comment("x")] * 3 + [_comment(_amendment_body(effective_from_round=2))]
    amendments = _plan_amendments(comments)
    digest = amendments[0].digest
    base = [_checkpoint(0, 1, C0), _checkpoint(1, 2, C0)]
    # Old board after the amendment comment.
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        _resolve([*base, _checkpoint(5, 2, C0)], amendments, C1)
    # Amended board without a digest.
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        _resolve([*base, _checkpoint(5, 2, C1)], amendments, C1)
    # A wrong digest.
    with pytest.raises(AgentLoopError, match="unknown reviewer-board amendment digest"):
        _resolve([*base, _checkpoint(5, 2, C1, digest="f" * 64)], amendments, C1)
    # A digest on a contract-neutral record.
    with pytest.raises(AgentLoopError, match="contract-neutral"):
        _resolve([*base, _neutral(5, 2, digest=digest)], amendments, C1)
    # A pre-amendment record may not already carry the digest.
    with pytest.raises(AgentLoopError, match="original contract"):
        _resolve([_checkpoint(0, 1, C0), _checkpoint(1, 2, C1, digest=digest)], amendments, C1)


def test_chain_link_intervals_judge_each_record_by_one_link():
    c2 = make_plan_contract(("Codex", "Claude"), "primary-then-panel", "Codex")
    board3 = ("Codex", "Claude", "Antigravity", "Gemini")
    base = make_plan_contract(board3, "primary-then-panel", "Codex")
    first_mid = make_plan_contract(BOARD, "primary-then-panel", "Codex")
    comments = [_comment("x")] * 10
    comments[2] = _comment(
        _amendment_body(
            original_required_reviewers=board3,
            removed_reviewers=("Gemini",),
            effective_from_round=2,
        )
    )
    comments[5] = _comment(_amendment_body(effective_from_round=3))
    amendments = _plan_amendments(comments)
    d1, d2 = (record.digest for record in amendments)
    history = [_checkpoint(0, 1, base), _checkpoint(3, 2, first_mid, digest=d1)]
    lineage = _resolve([*history, _checkpoint(6, 3, c2, digest=d2)], amendments, c2)
    assert lineage.contracts == (base, first_mid, c2)
    assert lineage.removed_reviewers == ("Gemini", "Antigravity")
    # A stale link-1 checkpoint after the second amendment fails closed.
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        _resolve([*history, _checkpoint(6, 3, first_mid, digest=d1)], amendments, c2)


def test_chain_gaps_order_and_unmatched_records_fail_closed():
    comments = [_comment("x")] * 6
    comments[2] = _comment(_amendment_body(effective_from_round=3))
    comments[4] = _comment(
        _amendment_body(
            original_required_reviewers=("Codex", "Claude"),
            removed_reviewers=("Claude",),
            effective_from_round=3,
        )
    )
    with pytest.raises(AgentLoopError):
        _resolve([_checkpoint(0, 1, C0)], _plan_amendments(comments), C1)
    unmatched = _plan_amendments(
        [_comment(_amendment_body(original_required_reviewers=("Codex", "Gemini", "Claude")))]
    )
    with pytest.raises(AgentLoopError, match="match no persisted scheduler contract"):
        _resolve([_checkpoint(0, 1, C0)], unmatched, C1)
    with pytest.raises(AgentLoopError, match="unmatched"):
        _resolve([_neutral(0, 1)], _plan_amendments([_comment(_amendment_body())]), None)


def test_re_adding_a_removed_reviewer_is_drift():
    comments = [_comment("x")] * 2 + [_comment(_amendment_body(effective_from_round=2))]
    amendments = _plan_amendments(comments)
    records = [_checkpoint(0, 1, C0), _checkpoint(4, 2, C1, digest=amendments[0].digest)]
    with pytest.raises(AgentLoopError, match="configured contract does not match"):
        _resolve(records, amendments, C0)


# --- activation (row activation-mismatch) ---------------------------------


def test_activation_requires_the_resume_start_round():
    comments = [_comment("x")] * 2 + [_comment(_amendment_body(effective_from_round=3))]
    amendments = _plan_amendments(comments)
    lineage = _resolve([_checkpoint(0, 1, C0), _checkpoint(1, 2, C0)], amendments, C1)
    assert lineage.pending_amendments == amendments
    require_amendment_activation(
        lineage, start_round_number=3, template=lambda amendment, n: "unused"
    )
    for wrong in (2, 4):
        with pytest.raises(AgentLoopError, match=f"re-enters round {wrong}") as excinfo:
            require_amendment_activation(
                lineage,
                start_round_number=wrong,
                template=lambda amendment, n: f"TEMPLATE effective {n}",
            )
        assert f"TEMPLATE effective {wrong}" in str(excinfo.value)
    # Once a digest-bound record exists the lineage rules govern instead.
    bound = _resolve(
        [
            _checkpoint(0, 1, C0),
            _checkpoint(1, 2, C0),
            _checkpoint(4, 3, C1, digest=amendments[0].digest),
        ],
        amendments,
        C1,
    )
    require_amendment_activation(bound, start_round_number=5, template=lambda a, n: "")


# --- ledger view (row sole-owner-reassign) --------------------------------


def _item(item_id, reviewer, *, owners=(), states=(), status="blocking"):
    return UnresolvedReviewItem(
        item_id=item_id,
        reviewer=reviewer,
        source_round=1,
        text=f"{item_id} text",
        status=status,
        resolution_owners=owners,
        owner_states=states,
    )


def test_ledger_view_reassigns_sole_owner_and_legacy_implicit_owner():
    items = (
        _item("item-1", "Codex", owners=("Antigravity",), states=(("Antigravity", "pending"),)),
        _item("item-2", "Antigravity", status="same-plan"),  # pre-change implicit ownership
        _item("item-3", "Claude"),  # implicit owner that remains on the board
        _item(
            "item-4",
            "Claude",
            owners=("Claude", "Antigravity"),
            states=(("Claude", "pending"), ("Antigravity", "pending")),
        ),
        _item("item-5", "Antigravity", status="resolved"),
    )
    view, reassignments = apply_board_amendment_to_ledger(
        items,
        removed_reviewers=("Antigravity",),
        remaining_reviewers=("Codex", "Claude"),
        primary_reviewer="Codex",
    )
    by_id = {item.item_id: item for item in view}
    assert by_id["item-1"].resolution_owners == ("Codex",)
    assert by_id["item-1"].owner_states == (("Codex", "pending"),)
    assert by_id["item-2"].resolution_owners == ("Codex",)
    assert by_id["item-2"].reviewer == "Antigravity"  # author is history
    assert by_id["item-2"].status == "same-plan"  # never auto-cleared
    assert by_id["item-3"] == items[2]  # untouched, including empty ownership
    assert by_id["item-4"].resolution_owners == ("Claude",)
    assert by_id["item-4"].owner_states == (("Claude", "pending"),)
    assert by_id["item-5"] == items[4]
    assert [entry.item_id for entry in reassignments] == ["item-1", "item-2", "item-4"]
    # Idempotent: applying the view again changes nothing.
    again, second = apply_board_amendment_to_ledger(
        view,
        removed_reviewers=("Antigravity",),
        remaining_reviewers=("Codex", "Claude"),
        primary_reviewer="Codex",
    )
    assert again == view and second == ()


def test_ledger_view_without_primary_reassigns_to_every_remaining_reviewer():
    view, _ = apply_board_amendment_to_ledger(
        (_item("item-1", "Antigravity"),),
        removed_reviewers=("Antigravity",),
        remaining_reviewers=("Codex", "Claude"),
        primary_reviewer=None,
    )
    assert view[0].resolution_owners == ("Codex", "Claude")
    assert dict(view[0].owner_states) == {"Codex": "pending", "Claude": "pending"}


def test_lineage_dataclass_reports_active_amendment():
    assert ContractLineage(contracts=(C0,), amendments=(), pending_amendments=()).active_digest is None


def test_amendment_only_comment_is_not_a_signed_human_requirement():
    from coding_review_agent_loop.board_amendment import is_reviewer_board_amendment_only
    from coding_review_agent_loop.github import _parse_issue_human_requirements
    from coding_review_agent_loop.protocol import parse_signed_human_requirement_body

    body = _amendment_body()
    assert is_reviewer_board_amendment_only(parse_signed_human_requirement_body(body))
    mixed = body.replace("Reviewer board amendment:", "Also please add a CLI flag.")
    assert not is_reviewer_board_amendment_only(parse_signed_human_requirement_body(mixed))
    parsed = _parse_issue_human_requirements(
        {"comments": [{"body": body}, {"body": mixed}, {"body": "Plain ask.\n-- Human Reviewer"}]}
    )
    assert [item.body.startswith("Also please") for item in parsed].count(True) == 1
    assert len(parsed) == 2


def test_signed_amendment_with_invalid_json_is_reported():
    body = _amendment_body().replace('"flow": "plan",', '"flow": "plan"')  # drop a comma
    records, ignored = parse_reviewer_board_amendment_records(body, comment_locator="issue #942 comment 4")
    assert records == ()
    assert len(ignored) == 1
    assert "issue #942 comment 4" in ignored[0] and "invalid JSON" in ignored[0]
    diagnostics = []
    assert collect_reviewer_board_amendments(
        [_comment(body)], flow="plan", issue_number=942, ignored_sink=diagnostics
    ) == ()
    assert diagnostics == [ignored[0].replace("issue #942 comment 4", "issue #942 comment 1")]
    # An unrelated signed fence with invalid JSON is not an amendment diagnostic.
    unrelated = "```json\n{not json\n```\n-- Human Reviewer"
    assert parse_reviewer_board_amendment_records(unrelated, comment_locator="c") == ((), ())


def test_malformed_amendment_comment_is_not_a_signed_requirement_on_issue_or_pr():
    from coding_review_agent_loop.board_amendment import is_reviewer_board_amendment_only
    from coding_review_agent_loop.github import (
        _parse_issue_human_requirements,
        _parse_pr_human_requirements,
    )
    from coding_review_agent_loop.protocol import parse_signed_human_requirement_body

    valid = _amendment_body()
    malformed = valid.replace('"flow": "plan",', '"flow": "plan"')  # invalid JSON
    # Discovery still reports it as an ignored malformed record.
    _records, ignored = parse_reviewer_board_amendment_records(malformed, comment_locator="c")
    assert ignored and "invalid JSON" in ignored[0]
    assert is_reviewer_board_amendment_only(parse_signed_human_requirement_body(malformed))
    ordinary = "Please keep the CLI flag stable.\n-- Human Reviewer"
    comments = [{"body": malformed}, {"body": valid}, {"body": ordinary}]
    for parse in (_parse_issue_human_requirements, _parse_pr_human_requirements):
        parsed = parse({"comments": comments})
        assert [item.body for item in parsed] == ["Please keep the CLI flag stable."]
    # Malformed JSON mixed with other text stays a signed requirement.
    mixed = malformed.replace("Reviewer board amendment:", "Also rename the flag.")
    assert not is_reviewer_board_amendment_only(parse_signed_human_requirement_body(mixed))
