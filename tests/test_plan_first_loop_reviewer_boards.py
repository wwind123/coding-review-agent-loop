"""Plan-first loop tests: reviewer-board amendments and named seats (#1543)."""

import re
from dataclasses import replace
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_object_in
from issue_loop_test_helpers import (
    FakeRunner,
    _M943_BOARD,
    _assert_named_issue_created_pr,
    _m943_decoded,
    _m943_interrupted_history,
    _m943_post_amendment,
    _m943_reduced_config,
    _m943_reviewer_calls,
    _named_1352_seats,
    _named_pr_records,
    _plan_round_records,
    _queue_named_reviews,
    _remediation_plan_fixtures,
    _serve_named_pr_rest,
    _staged_plan_config,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_pr_review,
    structured_issue_implementation,
)
from coding_review_agent_loop.board_amendment import (  # noqa: E402
    format_reviewer_board_amendment_comment as _m943_amendment_comment,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _m943_strip_ownership(runner, item_id="item-1"):
    """Rewrite persisted records so ``item_id`` has genuinely pre-change ownership.

    Every record that carries the item gets ``resolution_owners``,
    ``owner_states``, ``owner_evidence``, and ``owner_dispositions`` emptied,
    so the finding falls back to its author exactly as a legacy ledger does.
    """
    from coding_review_agent_loop.round_state import _encode_round_metadata

    def legacy(items):
        return tuple(
            replace(
                item,
                resolution_owners=(),
                owner_states=(),
                owner_evidence=(),
                owner_dispositions=(),
            )
            if item.item_id == item_id
            else item
            for item in items
        )

    rewritten = 0
    for comment in runner.issue_comments:
        metadata = _m943_decoded(comment["body"])
        if metadata is None or not any(
            item.item_id == item_id for item in (*metadata.prior_items, *metadata.new_items)
        ):
            continue
        stripped = replace(
            metadata,
            prior_items=legacy(metadata.prior_items),
            new_items=legacy(metadata.new_items),
        )
        comment["body"] = re.sub(
            r"<!-- AGENT_LOOP_META: \S+ -->",
            lambda _match: f"<!-- AGENT_LOOP_META: {_encode_round_metadata(stripped)} -->",
            comment["body"],
        )
        rewritten += 1
    assert rewritten
    return runner


def _m943_floor_config(tmp_path):
    """The reduced board under a three-reviewer floor (#1377).

    Without a floor an unsigned removal is a recorded plan board change; only a
    below-floor reduction still needs the signed amendment.
    """
    from dataclasses import replace

    return replace(_m943_reduced_config(tmp_path), min_reviewers=3)


def test_board_amendment_partial_round_rescues_legacy_run_and_reassigns_finding(tmp_path):
    """Rows legacy-no-amendment, partial-round-activation, legacy-retroactive-amend,
    sole-owner-reassign, and activation-mismatch on one real history."""
    # The Antigravity finding is genuinely pre-change: no persisted record
    # carries explicit ownership, so it falls back to its removed author.
    runner = _m943_strip_ownership(_m943_interrupted_history(tmp_path))
    records = _plan_round_records(runner)
    old_round_3 = [
        record for record in records
        if record.round_number == 3 and record.phase == "scheduler-prelaunch"
    ]
    assert len(old_round_3) == 1
    assert tuple(old_round_3[0].scheduler_contract["required_reviewers"]) == _M943_BOARD
    # Every pre-change record lacks the new field.
    assert all(record.reviewer_board_amendment_digest is None for record in records)
    (finding,) = old_round_3[0].prior_items
    assert finding.item_id == "item-1" and finding.reviewer == "Antigravity"
    assert finding.resolution_owners == () and finding.owner_states == ()
    calls_before = _m943_reviewer_calls(runner)
    comments_before = len(runner.issue_comments)

    # legacy-no-amendment: the below-floor refusal prints a filled amendment template.
    with pytest.raises(AgentLoopError, match="below the floor of 3") as excinfo:
        run_issue_loop(runner, issue_number=56, config=_m943_floor_config(tmp_path), plan_first=True)
    message = str(excinfo.value)
    assert '"kind": "reviewer-board-amendment"' in message
    assert '"effective_from_round": 3' in message
    assert '"removed_reviewers": [\n    "Antigravity"\n  ]' in message
    assert _m943_reviewer_calls(runner) == calls_before
    assert len(runner.issue_comments) == comments_before

    # activation-mismatch: an effective round other than the re-entered round 3.
    for wrong in (2, 4):
        mismatched = _m943_interrupted_history(tmp_path)
        _m943_post_amendment(mismatched, effective_from_round=wrong)
        before = (_m943_reviewer_calls(mismatched), len(mismatched.issue_comments))
        with pytest.raises(AgentLoopError, match="re-enters round 3") as excinfo:
            run_issue_loop(
                mismatched, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True
            )
        assert '"effective_from_round": 3' in str(excinfo.value)
        assert (_m943_reviewer_calls(mismatched), len(mismatched.issue_comments)) == before

    # partial-round-activation: the amendment starts inside round 3.  A first
    # resume is interrupted right after the fresh amended checkpoint.
    _m943_post_amendment(runner, effective_from_round=3)
    real_post = architecture_contract.post_issue_comment

    def stop_after_fresh_checkpoint(*args, **kwargs):
        result = real_post(*args, **kwargs)
        metadata = _m943_decoded(kwargs["body"])
        if metadata is not None and metadata.reviewer_board_amendment_digest is not None:
            raise KeyboardInterrupt
        return result

    with patch_object_in((architecture_contract, validated_agent_module, response_validation, discuss_loop, execution_policy, child_plan_binding, plan_first_loop_module), "post_issue_comment", side_effect=stop_after_fresh_checkpoint):
        with pytest.raises(KeyboardInterrupt):
            run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    assert _m943_reviewer_calls(runner) == calls_before
    audits = [c["body"] for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]
    assert len(audits) == 1
    assert "Activation round: 3" in audits[0]
    assert "item-1 (Antigravity -> Codex)" in audits[0]
    fresh = [
        record for record in _plan_round_records(runner)
        if record.reviewer_board_amendment_digest is not None
    ]
    assert len(fresh) == 1
    assert fresh[0].round_number == 3 and fresh[0].phase == "scheduler-prelaunch"
    assert tuple(fresh[0].scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
    # Round 3's persisted ledger is carried verbatim, never rewritten in-round.
    assert fresh[0].prior_items == old_round_3[0].prior_items
    (fresh_body,) = [
        c["body"] for c in runner.issue_comments
        if (metadata := _m943_decoded(c["body"])) is not None
        and metadata.reviewer_board_amendment_digest is not None
    ]
    assert "Reviewer board amended from round 3" in fresh_body
    assert "required board now Codex, Gemini" in fresh_body
    assert "Required board now: Codex, Gemini" in audits[0]

    # Restart after the fresh checkpoint: Codex's round-3 review is reused,
    # the reassignment is re-derived, and only the Gemini sweep runs.
    assert run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True) == 0
    assert _m943_reviewer_calls(runner) == [*calls_before, "gemini"]
    assert "agy" not in _m943_reviewer_calls(runner)[len(calls_before):]
    assert len([c for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]) == 1
    records = _plan_round_records(runner)
    digest = fresh[0].reviewer_board_amendment_digest
    post_amendment_checkpoints = [
        record for record in records
        if record.phase == "scheduler-prelaunch" and record.round_number >= 3
        and record.reviewer_board_amendment_digest is not None
    ]
    assert post_amendment_checkpoints
    assert all(
        record.reviewer_board_amendment_digest == digest
        and tuple(record.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for record in post_amendment_checkpoints
    )
    # Contract-neutral records carry the digest only beside the plan board
    # root anchor that every stamped record carries (#1377).
    assert all(
        record.reviewer_board_amendment_digest is None or record.plan_board_root is not None
        for record in records
        if record.scheduler_contract is None
    )
    # The Antigravity finding was cleared only by Codex's disposition.
    item_1_dispositions = [
        disposition
        for record in records
        if record.round_number == 3
        for disposition in record.dispositions
        if disposition.item_id == "item-1"
    ]
    assert item_1_dispositions
    assert {disposition.reviewer for disposition in item_1_dispositions} == {"Codex"}
    later = [record for record in records if record.round_number >= 4]
    assert later and all(
        item.item_id != "item-1" for record in later for item in record.prior_items
    )
    # Earlier rounds, including Antigravity's banked round-2 review, are history.
    assert any(
        record.agent == "Antigravity" and record.round_number == 2 and record.role == "reviewer"
        for record in records
    )

    # re-add-reviewer (#1377): widening the board after the plan is completely
    # approved never reopens planning and records no plan board change.
    calls_after_approval = _m943_reviewer_calls(runner)
    assert run_issue_loop(
        runner,
        issue_number=56,
        config=_staged_plan_config(tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8),
        plan_first=True,
    ) == 0
    assert _m943_reviewer_calls(runner) == calls_after_approval
    assert not any("AGENT_PLAN_BOARD_CHANGE" in c["body"] for c in runner.issue_comments)


def test_board_amendment_rejects_an_old_board_checkpoint_after_the_record(tmp_path):
    """Row post-amend-old-board: a stale-board round after the record is drift."""
    runner = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(runner, effective_from_round=3)
    old_checkpoint = next(
        comment for comment in runner.issue_comments
        if (metadata := _m943_decoded(comment["body"])) is not None
        and metadata.phase == "scheduler-prelaunch"
        and metadata.round_number == 3
    )
    stale = replace(_m943_decoded(old_checkpoint["body"]), scheduler_reasons=("stale board",))
    runner.issue_comments.append(
        {
            "author": {"login": "bot"},
            "createdAt": "2026-05-23T00:59:00Z",
            "body": _attach_round_metadata("Plan review scheduling audit.\n\n-- Orchestrator", stale),
        }
    )
    calls_before = _m943_reviewer_calls(runner)
    with pytest.raises(AgentLoopError, match="does not carry its amended contract"):
        run_issue_loop(runner, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)
    assert _m943_reviewer_calls(runner) == calls_before


def test_board_amendment_invalid_records_fail_closed_or_are_ignored(tmp_path):
    """Row record-invalid at the orchestrator: no partial amendment is applied."""
    unsigned = _m943_interrupted_history(tmp_path)
    unsigned.issue_comments.append(
        {
            "author": {"login": "operator"},
            "createdAt": "2026-05-23T00:59:00Z",
            "body": _m943_amendment_comment(
                flow="plan", issue=56, pr_number=None, original_required_reviewers=_M943_BOARD,
                policy="primary-then-panel", primary_reviewer="Codex",
                removed_reviewers=("Antigravity",), effective_from_round=3, rationale="quota",
            ).replace("-- Human Reviewer", "-- Anthropic Claude"),
        }
    )
    with pytest.raises(AgentLoopError, match="below the floor of 3"):
        run_issue_loop(unsigned, issue_number=56, config=_m943_floor_config(tmp_path), plan_first=True)

    removes_primary = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(removes_primary, effective_from_round=3, removed=("Codex",))
    with pytest.raises(AgentLoopError, match="Human decision required"):
        run_issue_loop(
            removes_primary,
            issue_number=56,
            config=_staged_plan_config(
                tmp_path, reviewer=("gemini", "antigravity"), primary_plan_reviewer="gemini"
            ),
            plan_first=True,
        )

    other_issue = _m943_interrupted_history(tmp_path)
    _m943_post_amendment(other_issue, effective_from_round=3, issue=57)
    with pytest.raises(AgentLoopError, match="names issue #57"):
        run_issue_loop(other_issue, issue_number=56, config=_m943_reduced_config(tmp_path), plan_first=True)


@pytest.mark.parametrize("legacy_owner", [False, True], ids=["explicit-owner", "legacy-implicit-owner"])
def test_board_amendment_reenters_a_reconciled_round_under_the_amended_board(tmp_path, legacy_owner):
    """Row reconciled-round-reentry: round 2 was reconciled under the old board.

    The legacy variant strips the Antigravity finding's persisted ownership,
    so the next round's ledger proves implicit ownership is normalized and
    reassigned rather than falling back to the removed author.
    """
    fresh, patch_text = _remediation_plan_fixtures()
    resolved = [{"item_id": "item-1", "disposition": "resolved"}]

    def reconciled_history():
        runner = _FakeRunner(
            claude_outputs=[fresh],  # the round-3 planner turn has no output yet
            codex_outputs=[
                structured_plan_review(state="approved"),
                structured_plan_review(state="approved", prior_plan_item_dispositions=resolved),
            ],
            gemini_outputs=[
                structured_plan_review(state="approved", reviewer="Google Gemini"),
                structured_plan_review(state="approved", reviewer="Google Gemini"),
            ],
            antigravity_outputs=[
                structured_plan_review(
                    state="blocking",
                    reviewer="Google Antigravity",
                    summary="One plan step omits the rollout owner.",
                    blocking_plan_issues=["Name the rollout owner in the plan steps."],
                ),
            ],
        )
        config = _staged_plan_config(
            tmp_path, reviewer=("codex", "gemini", "antigravity"), max_rounds=8, review_parallel=True
        )
        with pytest.raises(AgentLoopError):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        records = _plan_round_records(runner)
        assert any(
            record.round_number == 2 and record.phase == "reconciliation" for record in records
        )
        assert not any(record.round_number == 3 for record in records)
        return runner

    reduced = _staged_plan_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=8, review_parallel=True
    )

    # An amendment effective from round 3 fails closed naming N=2.
    early = reconciled_history()
    _m943_post_amendment(early, effective_from_round=3)
    before = (_m943_reviewer_calls(early), len(early.issue_comments))
    with pytest.raises(AgentLoopError, match="re-enters round 2"):
        run_issue_loop(early, issue_number=56, config=reduced, plan_first=True)
    assert (_m943_reviewer_calls(early), len(early.issue_comments)) == before

    runner = reconciled_history()
    if legacy_owner:
        _m943_strip_ownership(runner)
        (seeded,) = [
            item
            for record in _plan_round_records(runner)
            if record.round_number == 2 and record.phase == "reconciliation"
            for item in record.new_items
            if item.item_id == "item-1"
        ]
        assert seeded.resolution_owners == () and seeded.owner_states == ()
    _m943_post_amendment(runner, effective_from_round=2)
    runner.claude_outputs.append(patch_text)
    calls_before = _m943_reviewer_calls(runner)
    assert run_issue_loop(runner, issue_number=56, config=reduced, plan_first=True) == 0

    records = _plan_round_records(runner)
    # Every plan record of a stamped run carries its governing digest (#1377);
    # the contract-bearing ones carry the amended contract.
    amended = [
        record for record in records
        if record.reviewer_board_amendment_digest is not None
        and record.scheduler_contract is not None
    ]
    # Round 2 is re-entered with a fresh digest-bound checkpoint, then the
    # blocking Antigravity finding moves the loop to coder round 3.
    assert amended[0].round_number == 2 and amended[0].phase == "scheduler-prelaunch"
    assert all(
        tuple(record.scheduler_contract["required_reviewers"]) == ("Codex", "Gemini")
        for record in amended
    )
    assert any(record.round_number == 3 and record.role == "coder" for record in records)
    round_3 = [
        record for record in records
        if record.round_number == 3 and record.phase == "scheduler-prelaunch"
    ]
    assert round_3 and round_3[0].reviewer_board_amendment_digest == amended[0].reviewer_board_amendment_digest
    # The carried finding now has explicit ownership that excludes Antigravity.
    (carried,) = [item for item in round_3[0].prior_items if item.item_id == "item-1"]
    assert carried.reviewer == "Antigravity"
    assert carried.status == "blocking"
    assert carried.resolution_owners == ("Codex",)
    assert carried.owner_states == (("Codex", "pending"),)
    audits = [c["body"] for c in runner.issue_comments if "Reviewer board amendment applied." in c["body"]]
    assert len(audits) == 1 and "item-1 (Antigravity -> Codex)" in audits[0]
    assert "agy" not in _m943_reviewer_calls(runner)[len(calls_before):]


def test_named_plan_first_issue_handoff_reaches_pr_review_with_legacy_coder_record(tmp_path):
    """Rows plan-first-handoff-named: the #1352 board hands off to PR review."""
    from coding_review_agent_loop.board_change import identity_from_config
    from coding_review_agent_loop.plan_verification import derive_plan_verification_context
    from coding_review_agent_loop.reviewer_seats import resolve_plan_handoff_board

    seats = _named_1352_seats(tmp_path)
    outputs = {
        "claude_outputs": [
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
    }
    _queue_named_reviews(outputs, seats, structured_plan_review)
    _queue_named_reviews(outputs, seats, structured_pr_review)
    runner = _serve_named_pr_rest(FakeRunner(
        **outputs, authenticated_actor=("agent-loop", 4242), serve_rest_issue_comments=True,
    ))
    config = make_config(tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False)

    assert run_issue_loop(
        runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True,
    ) == 0

    plan = derive_plan_verification_context(
        [SimpleNamespace(body=comment["body"]) for comment in runner.issue_comments],
        issue_number=56,
    )
    handoff_board = resolve_plan_handoff_board(config, plan)
    assert handoff_board.inherited and not handoff_board.changed
    assert tuple(handoff_board.config.reviewer) == seats
    _assert_named_issue_created_pr(runner, config, identity_from_config(handoff_board.config))
    assert "repos/OWNER/REPO/issues/56/comments?per_page=100&page=1" in [
        cmd[2] for cmd, _cwd in runner.commands if cmd[:2] == ["gh", "api"] and len(cmd) > 2
    ]
    # Public seat footers name the backend and model (#1363); the plan
    # reviewer records still carry the bare seat IDs.
    from coding_review_agent_loop.round_state import _extract_round_metadata_records

    seat_footers = [
        "-- sol (OpenAI Codex: gpt-6.1-sol (medium))",
        "-- flash (Google Antigravity: Gemini 3.8 Flash (High))",
        "-- opus-low (Google Antigravity: Claude Opus 5.5 (Low))",
    ]
    for comments in (runner.issue_comments, runner.pr_payload.get("comments", [])):
        footers = [c["body"].rstrip().splitlines()[-1] for c in comments]
        assert [f for f in footers if f in seat_footers] == seat_footers
        assert not {"-- sol", "-- flash", "-- opus-low"} & set(footers)
    plan_reviews = [
        record for record in _extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.issue_comments], flow="plan",
        )
        if record.metadata.role == "reviewer"
    ]
    assert sorted(r.metadata.agent for r in plan_reviews) == ["flash", "opus-low", "sol"]


def test_named_direct_issue_reaches_pr_review_with_anchored_board(tmp_path):
    """Row direct-issue-named: the structured initial coder record."""
    from coding_review_agent_loop.board_change import identity_from_config

    seats = _named_1352_seats(tmp_path)
    outputs = {"claude_outputs": [structured_issue_implementation(pr_number=77)]}
    _queue_named_reviews(outputs, seats, structured_pr_review)
    runner = _serve_named_pr_rest(FakeRunner(
        **outputs, authenticated_actor=("agent-loop", 4242), serve_rest_issue_comments=True,
    ))
    config = make_config(tmp_path, reviewer=seats, reviewer_seats=seats, pre_review_tests=False)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    _assert_named_issue_created_pr(runner, config, identity_from_config(config))
    (initial, *_rest) = _named_pr_records(runner)
    assert initial.metadata.raw_structured_coder_response is not None
    assert "repos/OWNER/REPO/issues/56/comments?per_page=100&page=1" in [
        cmd[2] for cmd, _cwd in runner.commands if cmd[:2] == ["gh", "api"] and len(cmd) > 2
    ]
