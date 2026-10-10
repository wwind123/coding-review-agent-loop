"""Signed reviewer-board amendment records (#943): parser, chain, lineage, ledger."""

import json
from pathlib import Path
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
    assert record.amended_board(BOARD) == ("Codex", "Claude")
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


def test_named_pr_backend_outage_requires_one_complete_removal():
    from coding_review_agent_loop.reviewer_seats import (
        ReviewerSeat,
        SeatAgent,
        reviewer_seat_binding,
        validate_pr_backend_outage_amendments,
    )

    def seat(name, backend, model):
        from pathlib import Path
        return SeatAgent(ReviewerSeat(name, backend, (model,)), Path("/tmp") / name)

    config = SimpleNamespace(
        reviewer_seats=("named",),
        reviewer=(
            seat("primary", "codex", "gpt-6-sol"),
            seat("flash", "antigravity", "gemini-flash"),
            seat("opus", "antigravity", "claude-opus"),
        ),
    )
    board = ("primary", "flash", "opus")
    def amendment(removed, reason="backend-unavailable"):
        return SimpleNamespace(
            original_required_reviewers=board,
            removed_reviewers=removed,
            reason=reason,
        )

    with pytest.raises(AgentLoopError, match="must remove every active seat"):
        validate_pr_backend_outage_amendments((amendment(("flash",)),), config)
    validate_pr_backend_outage_amendments((amendment(("flash", "opus")),), config)
    validate_pr_backend_outage_amendments((amendment(("flash",), "seat-unavailable"),), config)
    original_binding = reviewer_seat_binding(config)
    amended_view = SimpleNamespace(
        reviewer_seats=config.reviewer_seats,
        reviewer=config.reviewer[:1],
        pr_seat_binding_override=original_binding,
    )
    assert reviewer_seat_binding(amended_view) == original_binding
    with pytest.raises(AgentLoopError, match="no verified backend binding"):
        validate_pr_backend_outage_amendments((amendment(("unbound",)),), config)


def test_named_pr_seat_local_amendment_reason_is_signed_and_recoverable():
    from coding_review_agent_loop.reviewer_seats import (
        ReviewerSeat, SeatAgent, validate_pr_backend_outage_amendments,
    )

    config = SimpleNamespace(
        reviewer_seats=("named",),
        reviewer=tuple(
            SeatAgent(ReviewerSeat(name, backend, (model,)), Path("/tmp") / name)
            for name, backend, model in (
                ("primary", "codex", "gpt-6-sol"),
                ("flash", "antigravity", "gemini-flash"),
                ("opus", "antigravity", "claude-opus"),
            )
        ),
    )
    board = ("primary", "flash", "opus")
    body = format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=77, original_required_reviewers=board,
        policy="primary-then-panel", primary_reviewer="primary",
        removed_reviewers=("flash",), effective_from_round=2,
        reason="seat-unavailable", rationale="Only the flash model is unavailable.",
    )
    (record,) = collect_reviewer_board_amendments([_comment(body)], flow="pr", pr_number=77)
    validate_pr_backend_outage_amendments((record,), config)
    reduced = amend_contract(make_contract(board, "primary-then-panel", None, "primary"), record, base_board=board)
    assert reduced.required_reviewers == ("primary", "opus")
    shared_body = format_reviewer_board_amendment_comment(
        flow="pr", issue=None, pr_number=77, original_required_reviewers=board,
        policy="primary-then-panel", primary_reviewer="primary",
        removed_reviewers=("flash",), effective_from_round=2,
        rationale="The Antigravity account is unavailable.",
    )
    (shared,) = collect_reviewer_board_amendments([_comment(shared_body)], flow="pr", pr_number=77)
    with pytest.raises(AgentLoopError, match="must remove every active seat"):
        validate_pr_backend_outage_amendments((shared,), config)

    from coding_review_agent_loop.review_rounds import _unavailable_reviewer_amendment_advisory
    original = make_contract(board, "primary-then-panel", None, "primary")
    lineage = SimpleNamespace(contracts=(original,), removed_reviewers=())
    outcome, round_number, template = _unavailable_reviewer_amendment_advisory(
        pr_number=77, contract=original, lineage=lineage, removed=("flash",),
        fetch_start_round=lambda: 2, seat_local_failure=True,
    )
    assert (outcome, round_number) == ("validated", 2)
    assert template is not None
    (suggested,) = collect_reviewer_board_amendments([_comment(template)], flow="pr", pr_number=77)
    assert suggested.reason == "seat-unavailable"
    validate_pr_backend_outage_amendments((suggested,), config)


def test_named_pr_signed_outage_removal_and_explicit_restoration():
    from coding_review_agent_loop.reviewer_seats import (
        ReviewerSeat, SeatAgent, validate_pr_backend_outage_amendments,
    )

    board = ("primary", "flash", "opus", "other")
    config = SimpleNamespace(
        reviewer_seats=("named",),
        reviewer=tuple(
            SeatAgent(ReviewerSeat(name, backend, (model,)), Path("/tmp") / name)
            for name, backend, model in (
                ("primary", "codex", "gpt-6-sol"),
                ("flash", "antigravity", "gemini-flash"),
                ("opus", "antigravity", "claude-opus"),
                ("other", "claude", "claude-sonnet"),
            )
        ),
    )
    original = make_contract(board, "primary-then-panel", None, "primary")

    def signed(original_board, removed, restored, round_number):
        return format_reviewer_board_amendment_comment(
            flow="pr", issue=None, pr_number=17,
            original_required_reviewers=original_board,
            policy="primary-then-panel", primary_reviewer="primary",
            removed_reviewers=removed, restored_reviewers=restored,
            effective_from_round=round_number,
            rationale="Shared backend outage or recovery.",
        )

    comments = [
        _comment(signed(board, ("flash", "opus"), (), 2)),
        _comment(signed(("primary", "other"), (), ("flash", "opus"), 3)),
    ]
    amendments = collect_reviewer_board_amendments(comments, flow="pr", pr_number=17)
    validate_pr_backend_outage_amendments(amendments, config)
    reduced = amend_contract(original, amendments[0], base_board=board)
    assert reduced.required_reviewers == ("primary", "other")
    restored = amend_contract(
        reduced, amendments[1], base_board=board,
        previously_removed=("flash", "opus"),
    )
    assert restored.required_reviewers == board
    with pytest.raises(AgentLoopError, match="primary reviewer"):
        (bad,) = collect_reviewer_board_amendments(
            [_comment(signed(board, ("primary",), (), 2))],
            flow="pr", pr_number=17,
        )
        amend_contract(original, bad, base_board=board)


def test_named_pr_signed_outage_resumes_with_removed_then_restored_seats(tmp_path, monkeypatch):
    import coding_review_agent_loop.orchestrator as orchestrator

    from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup, structured_pr_review
    from coding_review_agent_loop.cli import run_pr_loop
    from coding_review_agent_loop.reviewer_seats import ReviewerSeat, SeatAgent

    flash = SeatAgent(ReviewerSeat("flash", "antigravity", ("Model A",)), tmp_path / "flash")
    opus = SeatAgent(ReviewerSeat("opus", "antigravity", ("Model B",)), tmp_path / "opus")
    flash.workdir.mkdir()
    opus.workdir.mkdir()
    board = ("Codex", "Gemini", "flash", "opus")
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(),
            structured_pr_review(state="blocking", blocking_items=["Fix worker."], summary="Codex blocks."),
            *[structured_pr_review(
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ) for _ in range(4)],
        ],
        gemini_outputs=[
            structured_pr_review(reviewer="Google Gemini") for _ in range(2)
        ] + [structured_pr_review(
            reviewer="Google Gemini",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ) for _ in range(4)],
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        antigravity_outputs=[
            (structured_pr_review(reviewer="flash (Google Antigravity: Model A)"), 0),
            (structured_pr_review(reviewer="opus (Google Antigravity: Model B)"), 0),
        ],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini", flash, opus), reviewer_seats=(flash, opus),
        pr_review_policy="selective-intermediate", max_rounds=4,
        pre_review_tests=False, agent_max_retries=0,
    )
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    runner.pr_payload["headRefOid"] = "def456"

    def append_amendment(original_board, removed, restored, round_number):
        comment_index = len(runner.pr_payload["comments"])
        body = format_reviewer_board_amendment_comment(
            flow="pr", issue=None, pr_number=77,
            original_required_reviewers=original_board,
            policy="selective-intermediate", primary_reviewer=None,
            removed_reviewers=removed, restored_reviewers=restored,
            effective_from_round=round_number,
            rationale="Shared backend outage or recovery.",
        )
        runner.pr_payload["comments"].append({
            "body": body,
            "author": {"login": "human-reviewer", "id": 81},
            "createdAt": f"2026-05-23T00:00:{comment_index:02d}Z",
            "id": 900 + round_number,
        })

    append_amendment(board, ("flash", "opus"), (), 1)
    before = len(runner.commands)
    original_post = orchestrator.post_pr_comment

    def interrupt_after_coder(*args, **kwargs):
        result = original_post(*args, **kwargs)
        match = orchestrator.ROUND_RESUME_MARKER_RE.search(kwargs["body"])
        if match and orchestrator._decode_round_metadata(match["payload"]).role == "coder":
            raise KeyboardInterrupt
        return result

    with monkeypatch.context() as patch:
        patch.setattr(orchestrator, "post_pr_comment", interrupt_after_coder)
        with pytest.raises(KeyboardInterrupt):
            run_pr_loop(runner, pr_number=77, config=config)
    assert not any(cmd[0] == "agy" for cmd, _ in runner.commands[before:])

    append_amendment(("Codex", "Gemini"), (), ("flash", "opus"), 2)
    runner.antigravity_outputs.extend([
        (structured_pr_review(
            reviewer="flash (Google Antigravity: Model A)",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ), 0),
        (structured_pr_review(
            reviewer="opus (Google Antigravity: Model B)",
            prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ), 0),
    ])
    before = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    models = [cmd[cmd.index("--model") + 1] for cmd, _ in runner.commands[before:]
              if cmd[0] == "agy" and "--model" in cmd]
    assert models == ["Model A", "Model B"]
    from coding_review_agent_loop.round_state import _extract_round_metadata_records
    checkpoints = [
        record.metadata for record in _extract_round_metadata_records(
            [_comment(comment["body"]) for comment in runner.pr_payload["comments"]], flow="pr"
        ) if record.metadata.phase == "scheduler-prelaunch"
    ]
    assert any(checkpoint.scheduler_contract["required_reviewers"] == ["Codex", "Gemini"]
               and checkpoint.reviewer_board_amendment_digest for checkpoint in checkpoints)
    assert any(checkpoint.scheduler_contract["required_reviewers"] == list(board)
               and checkpoint.reviewer_board_amendment_digest for checkpoint in checkpoints)


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


# --- restoration (#984) ---------------------------------------------------


def _restore_body(**overrides):
    values = {
        "original_required_reviewers": ("Codex", "Claude"),
        "removed_reviewers": (),
        "restored_reviewers": ("Antigravity",),
        "effective_from_round": 4,
        "rationale": "Antigravity quota returned; probe answered in 6.6 s.",
    }
    values.update(overrides)
    return _amendment_body(**values)


def _remove_restore_comments():
    comments = [_comment("x")] * 8
    comments[1] = _comment(_amendment_body(effective_from_round=2))
    comments[4] = _comment(_restore_body(effective_from_round=4))
    return comments


def test_removal_only_digest_is_unchanged_by_the_optional_restore_key():
    import hashlib

    (record,) = _plan_amendments([_comment(_amendment_body())])
    payload = json.loads(_amendment_body().split("```json\n", 1)[1].split("\n```", 1)[0])
    assert "restored_reviewers" not in payload
    legacy = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    assert record.digest == legacy
    assert record.restored_reviewers == ()


def test_restoration_record_parses_and_malformed_restorations_are_reported():
    records, ignored = parse_reviewer_board_amendment_records(_restore_body(), comment_locator="c")
    assert ignored == ()
    (record,) = records
    assert record.reason == "backend-recovered"
    assert record.removed_reviewers == ()
    assert record.restored_reviewers == ("Antigravity",)
    for overrides, problem in (
        ({"restored_reviewers": ()}, "removed_reviewers must be non-empty"),
        (
            {"removed_reviewers": ("Claude",), "restored_reviewers": ("Claude",)},
            "both removed and restored",
        ),
        ({"reason": "backend-unavailable"}, "restoration-only record"),
    ):
        found, problems = parse_reviewer_board_amendment_records(
            _restore_body(**overrides), comment_locator="c"
        )
        assert found == () and problem in problems[0]
    found, problems = parse_reviewer_board_amendment_records(
        _amendment_body(reason="backend-recovered"), comment_locator="c"
    )
    assert found == () and "removal-only record" in problems[0]


def test_restore_of_removed_reviewer_is_required_from_its_effective_round():
    amendments = _plan_amendments(_remove_restore_comments())
    d1, d2 = (record.digest for record in amendments)
    records = [
        _checkpoint(0, 1, C0),
        _checkpoint(2, 2, C1, digest=d1),
        _checkpoint(3, 3, C1, digest=d1),
        _checkpoint(5, 4, C0, digest=d2),
    ]
    lineage = _resolve(records, amendments, C0)
    assert lineage.contracts == (C0, C1, C0)
    # C0 order is preserved, and nothing is removed any more.
    assert lineage.current_board == BOARD
    assert lineage.removed_reviewers == ()
    assert lineage.restoration_rounds == {"Antigravity": 4}
    # A round-4 record still on the reduced board fails closed.
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        _resolve([*records[:3], _checkpoint(5, 4, C1, digest=d1)], amendments, C0)
    # Pending restoration: activation must be the round the resume re-enters.
    pending = _resolve(records[:3], amendments, C0)
    assert pending.pending_amendments == (amendments[1],)
    require_amendment_activation(pending, start_round_number=4, template=lambda a, n: "")
    with pytest.raises(AgentLoopError, match="re-enters round 5"):
        require_amendment_activation(pending, start_round_number=5, template=lambda a, n: "")


def test_restoring_a_mid_board_reviewer_keeps_c0_order():
    middle = ("Codex", "Antigravity", "Claude")
    base = make_plan_contract(middle, "primary-then-panel", "Codex")
    reduced = make_plan_contract(("Codex", "Claude"), "primary-then-panel", "Codex")
    comments = [_comment("x")] * 6
    comments[1] = _comment(
        _amendment_body(original_required_reviewers=middle, effective_from_round=2)
    )
    comments[4] = _comment(_restore_body(effective_from_round=4))
    amendments = _plan_amendments(comments)
    d1, d2 = (record.digest for record in amendments)
    assert amendments[1].amended_board(middle) == middle
    lineage = _resolve(
        [
            _checkpoint(0, 1, base),
            _checkpoint(2, 2, reduced, digest=d1),
            _checkpoint(5, 4, base, digest=d2),
        ],
        amendments,
        base,
    )
    assert lineage.contracts == (base, reduced, base)
    assert lineage.current_board == middle
    # The appended order would be drift, not the restored C0 board.
    appended = make_plan_contract(("Codex", "Claude", "Antigravity"), "primary-then-panel", "Codex")
    with pytest.raises(AgentLoopError, match="configured contract does not match"):
        _resolve(
            [
                _checkpoint(0, 1, base),
                _checkpoint(2, 2, reduced, digest=d1),
                _checkpoint(5, 4, base, digest=d2),
            ],
            amendments,
            appended,
        )


def test_restore_naming_a_reviewer_never_removed_fails_closed():
    # The run started on the reduced board: Antigravity was never on it.
    with pytest.raises(AgentLoopError, match="never removed by an earlier amendment"):
        _resolve(
            [_checkpoint(0, 1, C1)],
            _plan_amendments([_comment("x"), _comment(_restore_body(effective_from_round=2))]),
            C0,
        )
    # A reviewer already on the board cannot be restored.
    (already,) = _plan_amendments(
        [_comment(_restore_body(original_required_reviewers=BOARD, restored_reviewers=("Claude",)))]
    )
    with pytest.raises(AgentLoopError, match="already on the persisted board"):
        amend_contract(C0, already)
    # Without the chain context a restoration never applies.
    (restore,) = _plan_amendments([_comment(_restore_body())])
    with pytest.raises(AgentLoopError, match="never removed"):
        amend_contract(C1, restore)


def test_remove_restore_remove_chain_resolves_each_round():
    comments = _remove_restore_comments()
    comments[6] = _comment(_amendment_body(effective_from_round=6))
    amendments = _plan_amendments(comments)
    d1, d2, d3 = (record.digest for record in amendments)
    records = [
        _checkpoint(0, 1, C0),
        _checkpoint(2, 2, C1, digest=d1),
        _checkpoint(3, 3, C1, digest=d1),
        _checkpoint(5, 4, C0, digest=d2),
        _checkpoint(7, 6, C1, digest=d3),
    ]
    lineage = _resolve(records, amendments, C1)
    assert lineage.contracts == (C0, C1, C0, C1)
    assert lineage.amendments == amendments
    assert lineage.removed_reviewers == ("Antigravity",)
    assert lineage.restoration_rounds == {}
    # Round 4 must carry the restored board, not the removal link's.
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        _resolve(
            [*records[:3], _checkpoint(5, 4, C1, digest=d1), records[4]], amendments, C1
        )
    # Two distinct records amending one board are still a conflict.
    comments[6] = _comment(_amendment_body(removed_reviewers=("Claude",), effective_from_round=6))
    comments[7] = _comment(_amendment_body(effective_from_round=7))
    with pytest.raises(AgentLoopError, match="never chooses"):
        _resolve(records[:4], _plan_amendments(comments), C1)


def test_restoration_does_not_reopen_reassigned_and_cleared_items():
    amendments = _plan_amendments(_remove_restore_comments())
    d1, d2 = (record.digest for record in amendments)
    lineage = _resolve(
        [_checkpoint(0, 1, C0), _checkpoint(2, 2, C1, digest=d1), _checkpoint(5, 4, C0, digest=d2)],
        amendments,
        C0,
    )
    # The ledger persisted after the removal round: Antigravity's findings were
    # reassigned to the primary, which cleared one of them.
    persisted = (
        _item(
            "item-1",
            "Antigravity",
            owners=("Codex",),
            states=(("Codex", "cleared"),),
            status="resolved",
        ),
        _item("item-2", "Antigravity", owners=("Codex",), states=(("Codex", "pending"),)),
    )
    view, reassignments = apply_board_amendment_to_ledger(
        persisted,
        removed_reviewers=lineage.removed_reviewers,
        remaining_reviewers=lineage.current_board,
        primary_reviewer="Codex",
    )
    assert view == persisted and reassignments == ()


def test_restored_reviewer_must_reapprove_and_summary_names_it():
    from coding_review_agent_loop import orchestrator
    from coding_review_agent_loop.board_amendment import (
        amendment_summary_line,
        predates_restoration,
        render_amendment_audit_comment,
        restoration_rounds_from_comments,
    )
    from coding_review_agent_loop.plan_review_scheduling import (
        PlanCandidateKey,
        surfaced_requirement_id_digest,
    )

    comments = _remove_restore_comments()
    amendments = _plan_amendments(comments)
    d1, d2 = (record.digest for record in amendments)
    lineage = _resolve(
        [_checkpoint(0, 1, C0), _checkpoint(2, 2, C1, digest=d1), _checkpoint(5, 4, C0, digest=d2)],
        amendments,
        C0,
    )
    assert restoration_rounds_from_comments(comments, flow="plan") == {"Antigravity": 4}
    assert restoration_rounds_from_comments(comments, flow="pr") == {}
    assert predates_restoration(lineage.restoration_rounds, "Antigravity", 1)
    assert not predates_restoration(lineage.restoration_rounds, "Antigravity", 4)
    assert not predates_restoration(lineage.restoration_rounds, "Claude", 1)

    key = PlanCandidateKey(
        subject="a" * 64,
        aggregate_plan_identity="b" * 64,
        execution_strategy_identity="c" * 32,
        risk_test_matrix_identity="d" * 32,
        surfaced_requirement_id_digest=surfaced_requirement_id_digest(()),
    )

    def approval(agent, round_number, index):
        return PostedRoundRecord(
            index=index,
            body="",
            metadata=PostedRoundMetadata(
                flow="plan",
                role="reviewer",
                agent=agent,
                round_number=round_number,
                subject=key.subject,
                state="approved",
                plan_candidate_key=key.as_dict(),
            ),
        )

    def carried(records):
        return orchestrator._carried_plan_approvals(
            records,
            current_key=key,
            required_reviewers=BOARD,
            surfaced_requirement_ids=(),
            panel_evidence=orchestrator.PlanPanelEvidence(
                opening_index=0, opening_source="operator"
            ),
            primary_reviewer="Codex",
            restoration_rounds=lineage.restoration_rounds,
        )

    # A pre-removal approval of the very same plan does not carry.
    stale = (approval("Codex", 1, 1), approval("Claude", 1, 2), approval("Antigravity", 1, 3))
    assert carried(stale) == ("Claude", "Codex")
    assert carried((*stale, approval("Antigravity", 4, 9))) == ("Antigravity", "Claude", "Codex")

    note = amendment_summary_line(lineage)
    assert "restored Antigravity (from round 4, must re-approve)" in note
    assert "required board now Codex, Claude, Antigravity" in note
    audit = render_amendment_audit_comment(lineage, start_round_number=4, reassignments=())
    assert "- Removed reviewer(s): none (backend-recovered)" in audit
    assert "Restored reviewer(s): Antigravity" in audit and "not re-opened" in audit


def test_drift_error_offers_a_restore_template_when_config_adds_a_reviewer():
    from coding_review_agent_loop import orchestrator
    from coding_review_agent_loop.board_amendment import added_in_config

    assert added_in_config(C1, C0) == ("Antigravity",)
    assert added_in_config(C0, C1) is None
    clause = orchestrator._board_amendment_route_clause(
        flow="plan",
        issue_number=942,
        pr_number=None,
        persisted=C1,
        configured=C0,
        start_round_number=lambda: 4,
    )
    assert "restore it with a signed reviewer-board amendment" in clause
    (record,) = _plan_amendments([_comment(clause.split("\n\n", 1)[1])])
    assert record.restored_reviewers == ("Antigravity",)
    assert record.removed_reviewers == ()
    assert record.effective_from_round == 4


# --- CRLF, unreadable records, base-board acceptance, repost hints (#1133) ------


def _crlf(body):
    return body.replace("\n", "\r\n")


def test_crlf_signed_amendment_parses_with_the_lf_digest_and_is_not_a_requirement():
    from coding_review_agent_loop.github import _parse_pr_human_requirements

    lf = _amendment_body(flow="pr", issue=None, pr_number=77)
    (lf_record,) = collect_reviewer_board_amendments(
        [_comment(lf)], flow="pr", pr_number=77
    )
    (crlf_record,) = collect_reviewer_board_amendments(
        [_comment(_crlf(lf))], flow="pr", pr_number=77
    )
    assert crlf_record.digest == lf_record.digest
    ordinary = "Please keep the CLI flag stable.\n-- Human Reviewer"
    parsed = _parse_pr_human_requirements(
        {"comments": [{"body": _crlf(lf)}, {"body": ordinary}]}
    )
    assert [item.body for item in parsed] == ["Please keep the CLI flag stable."]


def _tilde_body():
    return _amendment_body().replace("```json", "~~~json").replace("\n```", "\n~~~")


def test_amendment_shaped_unreadable_block_is_reported_and_not_a_requirement():
    from coding_review_agent_loop.board_amendment import is_reviewer_board_amendment_only
    from coding_review_agent_loop.github import _parse_pr_human_requirements
    from coding_review_agent_loop.protocol import parse_signed_human_requirement_body

    body = _tilde_body()
    records, ignored = parse_reviewer_board_amendment_records(body, comment_locator="c1")
    assert records == ()
    assert len(ignored) == 1 and "fence could not be read" in ignored[0]
    assert is_reviewer_board_amendment_only(parse_signed_human_requirement_body(body))
    assert not _parse_pr_human_requirements({"comments": [{"body": body}]})


def test_prose_mentioning_the_kind_stays_a_requirement_without_amendment_diagnostic():
    from coding_review_agent_loop.board_amendment import is_reviewer_board_amendment_only
    from coding_review_agent_loop.protocol import parse_signed_human_requirement_body

    prose = "Do not use a reviewer-board-amendment here.\n-- Human Reviewer"
    with_tilde = _tilde_body().replace(
        "Reviewer board amendment:", "Please also rename the flag."
    )
    malformed_strict = _amendment_body().replace('"flow": "plan",', '"flow": "plan"').replace(
        "Reviewer board amendment:", "Please also rename the flag."
    )
    for body, expected in ((prose, 0), (with_tilde, 0), (malformed_strict, 1)):
        _records, ignored = parse_reviewer_board_amendment_records(body, comment_locator="c")
        assert len(ignored) == expected
        assert not is_reviewer_board_amendment_only(parse_signed_human_requirement_body(body))
    assert "invalid JSON" in parse_reviewer_board_amendment_records(
        malformed_strict, comment_locator="c"
    )[1][0]


def test_resolver_accepts_the_base_board_only_when_asked_and_never_other_boards():
    (amendment,) = _plan_amendments([_comment(_amendment_body(effective_from_round=2))])
    base = [_checkpoint(0, 1, C0)]
    kwargs = dict(
        contract_from_metadata=_plan_contract, drift_error=_drift, accept_base_configured=True
    )
    for configured in (C0, C1):
        lineage = resolve_contract_lineage(base, [amendment], configured, **kwargs)
        assert lineage.contracts[-1] == C1
    reordered = make_plan_contract(("Claude", "Codex"), "primary-then-panel", "Codex")
    with pytest.raises(AgentLoopError, match="does not match it"):
        resolve_contract_lineage(base, [amendment], reordered, **kwargs)
    with pytest.raises(AgentLoopError, match="does not match it"):
        _resolve(base, [amendment], C0)


def test_stale_record_detail_is_tagged_and_drift_builders_print_the_repost_round():
    from coding_review_agent_loop import orchestrator
    from coding_review_agent_loop.board_amendment import (
        STALE_UNREAD_AMENDMENT_HINT,
        stale_unread_amendment_locator,
    )

    comments = [_comment("x")] * 3 + [_comment(_amendment_body(effective_from_round=2))]
    amendments = _plan_amendments(comments)
    base = [_checkpoint(0, 1, C0), _checkpoint(5, 2, C0)]
    with pytest.raises(AgentLoopError) as excinfo:
        _resolve(base, amendments, C1)
    assert STALE_UNREAD_AMENDMENT_HINT in str(excinfo.value)
    locator = stale_unread_amendment_locator(str(excinfo.value))
    assert locator == "issue #942 comment 4"
    detail = str(excinfo.value)
    text = orchestrator._stale_amendment_repost_clause(detail, lambda: 2)
    assert f"Delete the amendment comment at {locator}" in text
    assert "effective_from_round 2" in text
    assert "the round the next resume re-enters" in orchestrator._stale_amendment_repost_clause(
        detail, lambda: None
    )
    assert orchestrator._stale_amendment_repost_clause("no tag", lambda: 2) == ""


def test_route_clause_names_rerun_rules_and_the_not_recognized_hint():
    from coding_review_agent_loop import orchestrator

    def clause(flow, recognized):
        return orchestrator._board_amendment_route_clause(
            flow=flow,
            issue_number=942 if flow == "plan" else None,
            pr_number=77 if flow == "pr" else None,
            persisted=C0,
            configured=C1,
            start_round_number=lambda: 2,
            amendments_recognized=recognized,
        )

    pr = clause("pr", False)
    assert "Posting the record is what removes the reviewer" in pr
    assert "original reviewer board still configured (recommended)" in pr
    assert "issue-created strict managed-CI PR" in pr
    assert "it was not recognized" in pr
    assert "it was not recognized" not in clause("pr", True)
    plan = clause("plan", False)
    assert "planning rerun then uses the reduced reviewer board" in plan
    assert '"effective_from_round": 2' in pr


def test_plan_board_hint_only_when_a_pr_amendment_explains_the_reduced_flags():
    from coding_review_agent_loop import orchestrator

    body = _amendment_body(flow="pr", issue=None, pr_number=77)
    comments = [_comment(body)]
    hint = orchestrator._pr_amendment_plan_board_hint
    assert "original reviewer board" in hint(
        comments, pr_number=77, supplied_reviewers=("codex", "claude")
    )
    assert hint(comments, pr_number=77, supplied_reviewers=("codex", "claude", "antigravity")) == ""
    assert hint([], pr_number=77, supplied_reviewers=("codex", "claude")) == ""
    # A record naming another PR is undecodable for this surface: no hint.
    assert hint(comments, pr_number=78, supplied_reviewers=("codex", "claude")) == ""
