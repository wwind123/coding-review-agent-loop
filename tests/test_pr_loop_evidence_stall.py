"""PR-loop evidence-only stall stop (#1324, stage 1 of #1317)."""

from __future__ import annotations

import json

import pytest

import coding_review_agent_loop.orchestrator as orchestrator
from coding_review_agent_loop.cli import build_parser, run_pr_loop
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.github import get_pr_review_context
from coding_review_agent_loop.protocol import parse_risk_test_matrix, risk_test_matrix_identity
from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section
from coding_review_agent_loop.round_state import _extract_round_metadata_records

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup


def _row(row_id: str) -> dict:
    return {
        "row_id": row_id,
        "label": f"Behaviour {row_id}",
        "entry_path_or_mode": "agent-loop pr",
        "initial_state": "approved plan",
        "event": "coder reports",
        "expected_outcome": f"outcome {row_id}",
        "forbidden_side_effects": [f"no side effect {row_id}"],
        "proposed_test_level": "integration",
        "proposed_test_location": f"tests/test_{row_id}.py",
        "applicability": "applicable",
        "related_scope_item_ids": ["scope-1"],
        "execution_owner": "one-shot",
    }


def _plan_context(*row_ids: str):
    matrix = parse_risk_test_matrix(
        {"applicability": "applicable", "rows": [_row(r) for r in row_ids], "important_exclusions": []}
    )
    identity = risk_test_matrix_identity(matrix)
    return orchestrator.make_approved_plan_context(
        "Approved plan.\n\n" + render_risk_test_matrix_section(matrix),
        source_locator="test approved plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )


def _review(blocking=(), *, carried=(), resolved=(), state=None):
    state = state or ("blocking" if blocking or carried else "approved")
    dispositions = [{"item_id": i, "disposition": "resolved"} for i in resolved]
    dispositions += [
        {"item_id": i, "disposition": "blocking", "note": "still no admissible citation"}
        for i in carried
    ]
    return (
        json.dumps({
            "schema_version": 1, "kind": "pr_review", "state": state, "summary": "review",
            "blocking_items": list(blocking), "same_pr_followups": [], "future_followups": [],
            "prior_item_dispositions": dispositions,
        })
        + f"\n<!-- AGENT_STATE: {state} -->\n-- OpenAI Codex"
    )


def _tagged(text="Capture an admissible passing test observation for the rows", rows=("row-a",)):
    return {"text": text, "evidence_row_ids": list(rows)}


def _coder(*numbers):
    return structured_coder_followup(
        addressed_items=[f"item-{n}" for n in numbers], summary="Retried capture."
    )


def _config(tmp_path, **overrides):
    overrides.setdefault("max_rounds", 8)
    return make_config(tmp_path, coder="claude", reviewer="codex", **overrides)


def _coder_count(runner):
    return sum(1 for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])


def _posted_records(runner, config):
    return _extract_round_metadata_records(
        get_pr_review_context(runner, config=config, pr_number=77).comments, flow="pr"
    )


def test_two_consecutive_evidence_only_reviews_stop_before_the_next_coder_turn(tmp_path):
    """`stall-trigger`, `stall-round-mapping`, `tag-carry-forward`, `stall-row-explanation`."""
    runner = FakeRunner(
        claude_outputs=[_coder(1), _coder(1)],
        codex_outputs=[_review([_tagged()]), _review(carried=["item-1"]), _review(carried=["item-1"])],
    )
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a", "row-b"))
    message = str(stopped.value)
    assert HumanDecisionRequiredError.EXIT_CODE == 4
    assert "PR #77 is stalled on evidence only" in message
    assert "`row-a`: canonical-status:" in message
    # Every unsatisfied row is listed with an orchestrator-derived, non-empty explanation.
    assert "`row-b`: canonical-status:" in message
    for line in message.splitlines():
        if line.startswith("- `row-"):
            assert line.split(": ", 1)[1].strip()
    assert "`--pr-evidence-stall-rounds 0`" in message and "signed human decision" in message
    assert _coder_count(runner) == 2  # no third coder turn
    records = _posted_records(runner, config)
    snapshots = {
        r.metadata.evidence_stall["review_round"]: r.metadata.evidence_stall
        for r in records if r.metadata.role == "coder" and r.metadata.evidence_stall
    }
    # Round 1 had no head-bound evidence, rounds 2 and 3 are indexed by the review they classify.
    assert snapshots[1]["qualifies"] is False and "evidence-head-unbound" in snapshots[1]["reasons"]
    assert snapshots[2]["qualifies"] is True and snapshots[2]["unsatisfied_row_ids"] == ["row-a", "row-b"]
    assert 3 not in snapshots
    for record in records:
        if record.metadata.role == "coder" and record.metadata.evidence_stall:
            assert record.metadata.evidence_stall["review_round"] == record.metadata.round_number - 1


def _stall_snapshots(runner, config):
    return {
        r.metadata.evidence_stall["review_round"]: r.metadata.evidence_stall
        for r in _posted_records(runner, config)
        if r.metadata.role == "coder" and r.metadata.evidence_stall
    }


def test_code_finding_round_resets_the_window_so_one_evidence_round_does_not_stop(tmp_path):
    """`stall-window-reset`: round 1 carries a code finding, so round 2 alone is not a stall."""
    runner = FakeRunner(
        claude_outputs=[_coder(1, 2), _coder(2)],
        codex_outputs=[
            _review(["A real code defect in the loader", _tagged()]),
            _review(carried=["item-2"], resolved=["item-1"]),
            _review(resolved=["item-2"]),
        ],
    )
    config = _config(tmp_path)
    assert run_pr_loop(
        runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a")
    ) == 0
    assert _coder_count(runner) == 2
    snapshots = _stall_snapshots(runner, config)
    assert snapshots[1]["qualifies"] is False and "untagged-finding" in snapshots[1]["reasons"]
    assert snapshots[2]["qualifies"] is True


def test_stall_after_a_reset_fires_only_after_k_consecutive_qualifying_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[_coder(1, 2), _coder(2)],
        codex_outputs=[
            _review(["A real code defect in the loader", _tagged()]),
            _review(carried=["item-2"], resolved=["item-1"]),
            _review(carried=["item-2"]),
        ],
    )
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path), approved_plan_context=_plan_context("row-a")
        )
    assert _coder_count(runner) == 2


def test_k_zero_disables_the_stop_but_still_writes_snapshots(tmp_path):
    """`stall-disabled`."""
    runner = FakeRunner(
        claude_outputs=[_coder(1), _coder(1), _coder(1)],
        codex_outputs=[
            _review([_tagged()]), _review(carried=["item-1"]), _review(carried=["item-1"]),
            _review(resolved=["item-1"]),
        ],
    )
    config = _config(tmp_path, pr_evidence_stall_rounds=0)
    assert run_pr_loop(
        runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a")
    ) == 0
    assert _coder_count(runner) == 3
    snapshots = _stall_snapshots(runner, config)
    assert [snapshots[n]["qualifies"] for n in (1, 2, 3)] == [False, True, True]


def test_rerun_after_the_stop_recomputes_the_same_stop_without_a_comment(tmp_path):
    """`stall-resume`: state comes only from posted records."""
    runner = FakeRunner(
        claude_outputs=[_coder(1), _coder(1)],
        codex_outputs=[_review([_tagged()]), _review(carried=["item-1"]), _review(carried=["item-1"])],
    )
    config = _config(tmp_path)
    context = _plan_context("row-a")
    with pytest.raises(HumanDecisionRequiredError) as first:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    coders_after_first = _coder_count(runner)
    runner.codex_outputs.append(_review(carried=["item-1"]))
    with pytest.raises(HumanDecisionRequiredError) as second:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    assert str(first.value) == str(second.value)
    assert _coder_count(runner) == coders_after_first
    assert not any("stalled on evidence only" in comment for comment in runner.comments)


def test_changed_approved_matrix_identity_unbinds_evidence_and_resets_the_window(tmp_path):
    """`stall-matrix-identity`: same row IDs, different approved matrix."""
    runner = FakeRunner(
        claude_outputs=[_coder(1), _coder(1), _coder(1)],
        codex_outputs=[_review([_tagged()]), _review(carried=["item-1"]), _review(carried=["item-1"])],
    )
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a"))
    changed = _plan_context("row-a")
    matrix = dict(changed.risk_test_matrix_payload)
    matrix["important_exclusions"] = ["A newly approved exclusion changes the matrix identity."]
    matrix_obj = parse_risk_test_matrix(matrix)
    identity = risk_test_matrix_identity(matrix_obj)
    assert identity != changed.risk_test_matrix_identity
    other = orchestrator.make_approved_plan_context(
        "Approved plan.\n\n" + render_risk_test_matrix_section(matrix_obj),
        source_locator="test approved plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix_obj.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )
    runner.codex_outputs.extend([_review(carried=["item-1"]), _review(resolved=["item-1"])])
    coders_before = _coder_count(runner)
    # The recorded evidence belongs to the old matrix, so the head is unbound under the new one.
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=other) in (0,)
    assert _coder_count(runner) > coders_before


def test_failing_check_first_seen_on_the_dispatch_path_disqualifies_the_round(tmp_path):
    """`stall-negative`: the board is fetched after the review, with no prior CI obligation."""
    runner = FakeRunner(
        claude_outputs=[_coder(1, 2)] * 4,
        codex_outputs=[_review([_tagged()])] + [_review(carried=["item-1", "item-2"])] * 5,
        pr_check_runs_payload={"check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]},
    )
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=4),
            approved_plan_context=_plan_context("row-a"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3


def test_untagged_blockers_never_stall(tmp_path):
    runner = FakeRunner(
        claude_outputs=[_coder(1)] * 3,
        codex_outputs=[_review(["Plain finding"]), _review(carried=["item-1"]), _review(carried=["item-1"]), _review(resolved=["item-1"])],
    )
    assert run_pr_loop(
        runner, pr_number=77, config=_config(tmp_path), approved_plan_context=_plan_context("row-a")
    ) == 0


def test_final_allowed_round_takes_the_existing_round_limit_exit(tmp_path):
    """`stall-precedence`: the round-limit exit runs before the stall decision."""
    runner = FakeRunner(
        claude_outputs=[_coder(1)] * 2,
        codex_outputs=[_review([_tagged()]), _review(carried=["item-1"]), _review(carried=["item-1"])],
    )
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=3),
            approved_plan_context=_plan_context("row-a"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert "blocking issues after round 3" in str(raised.value)


def test_stall_helper_is_not_called_on_the_merge_conflict_path(tmp_path, monkeypatch):
    """`stall-precedence`: a conflicted round makes no check calls and skips the helper."""
    import coding_review_agent_loop.pr_loop as pr_loop_module

    def _unexpected(*args, **kwargs):
        raise AssertionError("stall decision must not run on the merge-conflict path")

    monkeypatch.setattr(pr_loop_module, "_pr_evidence_stall_decision", _unexpected)
    runner = FakeRunner(
        claude_outputs=["Resolved the conflict.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"],
        codex_outputs=[_review(resolved=[])],
        mergeability_payloads=[
            {"mergeable": "CONFLICTING", "mergeStateStatus": "DIRTY", "headRefOid": "abc123", "baseRefName": "main"},
        ],
    )
    assert run_pr_loop(
        runner, pr_number=77, config=_config(tmp_path), approved_plan_context=_plan_context("row-a")
    ) == 0


def test_no_applicable_matrix_never_calls_the_helper_history_fetch(tmp_path):
    runner = FakeRunner(
        claude_outputs=[_coder(1)] * 2,
        codex_outputs=[_review(["Plain finding"]), _review(resolved=["item-1"])],
    )
    config = _config(tmp_path)
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    assert not _stall_snapshots(runner, config)


def test_stall_rounds_flag_defaults_validates_and_parses(tmp_path):
    assert make_config(tmp_path).pr_evidence_stall_rounds == 2
    assert make_config(tmp_path, pr_evidence_stall_rounds=0).pr_evidence_stall_rounds == 0
    for invalid in (-1, True, 1.5, "2"):
        with pytest.raises(AgentLoopError, match="--pr-evidence-stall-rounds must be"):
            make_config(tmp_path, pr_evidence_stall_rounds=invalid)
    base = [
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"), "--dangerous-agent-permissions",
    ]
    parser = build_parser()
    assert parser.parse_args(base).pr_evidence_stall_rounds is None
    assert config_from_args(parser.parse_args(base), FakeRunner()).pr_evidence_stall_rounds == 2
    parsed = parser.parse_args([*base, "--pr-evidence-stall-rounds", "0"])
    assert config_from_args(parsed, FakeRunner()).pr_evidence_stall_rounds == 0
    with pytest.raises(AgentLoopError, match="--pr-evidence-stall-rounds must be"):
        config_from_args(parser.parse_args([*base, "--pr-evidence-stall-rounds", "-1"]), FakeRunner())


def test_untagged_matrix_tag_without_an_applicable_matrix_is_rejected_and_repaired_away(tmp_path):
    """`tag-validation` through the live validator: a tag with no matrix cannot be accepted."""
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged()]), _review([_tagged()]), _review(resolved=["item-1"])],
    )
    config = _config(tmp_path)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config)
    assert "evidence_row_ids" in str(raised.value) or "risk-test matrix" in str(raised.value)
