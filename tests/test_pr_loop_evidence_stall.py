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
    # The conflicted round made no check-run, commit-status, or branch-protection call
    # before its conflict-resolution coder turn.
    commands = [cmd for cmd, _cwd in runner.commands]
    first_coder = next(i for i, cmd in enumerate(commands) if cmd[:1] == ["claude"])
    check_calls = [
        cmd for cmd in commands[:first_coder]
        if cmd[:2] == ["gh", "api"] and any(
            marker in part for part in cmd for marker in ("/check-runs", "/status", "/protection", "/rules/")
        )
    ]
    assert check_calls == []


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


# --- workflow-level coverage for the remaining matrix transitions -----------

import dataclasses  # noqa: E402

import coding_review_agent_loop.evidence_stall as evidence_stall_module  # noqa: E402
import coding_review_agent_loop.pr_loop as pr_loop_module  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    PostAuthClaimDiagnostic,
    TestObservationCitation as _Citation,
)
from coding_review_agent_loop.round_transport import decode_mapping, encode_mapping  # noqa: E402
from coding_review_agent_loop.unresolved_items import _upsert_machine_obligation  # noqa: E402

_FAILING_RUNS = {"check_runs": [{"name": "test", "status": "completed", "conclusion": "failure"}]}
_GREEN_RUNS = {"check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]}
_STALLED_RUNS = {
    "check_runs": [{
        "name": "test", "status": "queued", "conclusion": None,
        "created_at": "2020-01-01T00:00:00Z", "started_at": None,
    }]
}


def _spy_classifier(monkeypatch):
    """Record every real classification the dispatch path performs."""
    calls = []
    real = evidence_stall_module.classify_round

    def spy(**kwargs):
        snapshot = real(**kwargs)
        calls.append((kwargs, snapshot))
        return snapshot

    monkeypatch.setattr(evidence_stall_module, "classify_round", spy)
    return calls


def _shape_evidence(monkeypatch, plans):
    """Post-process the real follow-up derivation, one plan per coder turn.

    A plan maps row_id -> (status, diagnostic codes); ``"absent"`` drops the
    matrix evidence from the coder record entirely.  The real derivation still
    runs and binds the record to the authenticated head.
    """
    real = pr_loop_module._derive_authenticated_risk_evidence_for_coder
    remaining = list(plans)

    def derive(*args, **kwargs):
        parsed, result = real(*args, **kwargs)
        plan = remaining.pop(0) if remaining else None
        if plan is None or parsed.risk_test_matrix_evidence is None:
            return parsed, result
        if plan == "absent":
            return dataclasses.replace(
                parsed, risk_test_matrix_evidence=None, risk_test_matrix_diagnostics=()
            ), result
        rows = []
        diagnostics = []
        for row in parsed.risk_test_matrix_evidence.rows:
            status, codes = plan.get(row.row_id, (row.status, ()))
            if status == "verified":
                row = dataclasses.replace(
                    row, status="verified",
                    test_identifiers=("tests/test_x.py::test_row",), test_locations=("tests/test_x.py",),
                    workflow_path_claim="agent-loop pr", outcome_assertions=("ok",),
                    forbidden_effect_assertions=("none",),
                    evidence_citations=(_Citation("pytest", "receipt-1", "current-result"),),
                )
            else:
                row = dataclasses.replace(row, status=status, evidence_citations=())
            rows.append(row)
            diagnostics.extend(PostAuthClaimDiagnostic(row.row_id, code, f"{code} detail") for code in codes)
        evidence = dataclasses.replace(parsed.risk_test_matrix_evidence, rows=tuple(rows))
        return dataclasses.replace(
            parsed, risk_test_matrix_evidence=evidence, risk_test_matrix_diagnostics=tuple(diagnostics)
        ), result

    monkeypatch.setattr(pr_loop_module, "_derive_authenticated_risk_evidence_for_coder", derive)


def _inject_obligation(monkeypatch, kind, *, at_rounds):
    """Add a real machine obligation to the reconciled ledger at chosen rounds."""
    real = pr_loop_module._pr_evidence_stall_decision

    def decide(*args, **kwargs):
        if kwargs["round_number"] in at_rounds:
            kwargs["open_items"] = _upsert_machine_obligation(
                kwargs["open_items"], item_number=90, kind=kind,
                source_round=kwargs["round_number"], text=f"Open {kind} obligation.",
                failed_head_sha=kwargs["head_sha"],
            )
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "_pr_evidence_stall_decision", decide)


def _evidence_only_runner(reviews, coders, **kwargs):
    return FakeRunner(
        claude_outputs=[_coder(1)] * coders,
        codex_outputs=[_review([_tagged()])] + [_review(carried=["item-1"])] * (reviews - 1),
        **kwargs,
    )


def _checks_switch(monkeypatch, runner, payloads_by_review):
    """Serve a different check board once the given number of reviews were consumed."""
    real = pr_loop_module.get_pr_checks
    total = len(runner.codex_outputs)

    def fetch(*args, **kwargs):
        consumed = total - len(runner.codex_outputs)
        runner.pr_check_runs_payload = payloads_by_review.get(consumed, _GREEN_RUNS)
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "get_pr_checks", fetch)


def test_infrastructure_stalled_check_on_the_dispatch_fetch_disqualifies(tmp_path, monkeypatch):
    """`stall-negative`: an infrastructure-stalled board is never an evidence-only round."""
    calls = _spy_classifier(monkeypatch)
    runner = _evidence_only_runner(4, 3, pr_check_runs_payload=_STALLED_RUNS)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=4),
            approved_plan_context=_plan_context("row-a"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3
    assert len(calls) == 3
    for kwargs, snapshot in calls:
        assert kwargs["checks"] is not None and kwargs["checks"].infrastructure_stalls
        assert "checks-failing" in snapshot.reasons and not snapshot.qualifies


def test_failing_check_reconciled_into_a_ci_obligation_is_seen_by_the_classifier(tmp_path, monkeypatch):
    """`stall-negative`: the classifier sees both the failing board and the new CI obligation."""
    calls = _spy_classifier(monkeypatch)
    runner = FakeRunner(
        claude_outputs=[_coder(1, 2)] * 4,
        codex_outputs=[_review([_tagged()])] + [_review(carried=["item-1", "item-2"])] * 5,
        pr_check_runs_payload=_FAILING_RUNS,
    )
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=4),
            approved_plan_context=_plan_context("row-a"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3
    for kwargs, snapshot in calls:
        assert kwargs["checks"].failing
        assert any(
            item.is_machine_obligation and item.obligation_kind == "github-pr-checks"
            for item in kwargs["open_items"]
        )
        assert {"checks-failing", "machine-obligation-open"} <= set(snapshot.reasons)


@pytest.mark.parametrize("kind", ["github-pr-checks", "managed-exact-head-ci", "alembic-migration"])
def test_open_non_evidence_machine_obligation_suppresses_the_stop(tmp_path, monkeypatch, kind):
    """`stall-negative`: a preexisting CI, managed-CI, or migration obligation disqualifies."""
    calls = _spy_classifier(monkeypatch)
    _inject_obligation(monkeypatch, kind, at_rounds={2, 3})
    runner = _evidence_only_runner(4, 3)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=4),
            approved_plan_context=_plan_context("row-a"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3
    by_round = {kwargs["review_round"]: (kwargs, snap) for kwargs, snap in calls}
    for review_round in (2, 3):
        kwargs, snapshot = by_round[review_round]
        assert any(item.obligation_kind == kind for item in kwargs["open_items"])
        assert snapshot.reasons == ("machine-obligation-open",)


def test_changed_canonical_row_set_resets_the_window(tmp_path, monkeypatch):
    """`stall-negative` / `stall-trigger`: a changed unsatisfied-row set restarts the count."""
    _shape_evidence(monkeypatch, [
        {"row-a": ("stale/unverified", ()), "row-b": ("stale/unverified", ())},
        {"row-a": ("stale/unverified", ()), "row-b": ("verified", ())},
        {"row-a": ("stale/unverified", ()), "row-b": ("verified", ())},
    ])
    runner = _evidence_only_runner(4, 3)
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a", "row-b"))
    # Review 2 saw {a, b}; review 3 saw {a}: no stop; review 4 repeated {a}: stop.
    assert _coder_count(runner) == 3
    snapshots = _stall_snapshots(runner, config)
    assert snapshots[2]["unsatisfied_row_ids"] == ["row-a", "row-b"]
    assert snapshots[3]["unsatisfied_row_ids"] == ["row-a"] and snapshots[3]["qualifies"] is True
    message = str(stopped.value)
    assert "`row-a`" in message and "`row-b`" not in message


def test_tag_naming_a_row_that_became_verified_is_an_ordinary_finding(tmp_path, monkeypatch):
    """`tag-verified-row`: on the dispatch path the tag cannot hide the finding."""
    _shape_evidence(monkeypatch, [{"row-a": ("verified", ()), "row-b": ("stale/unverified", ())}] * 3)
    calls = _spy_classifier(monkeypatch)
    runner = _evidence_only_runner(4, 3)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=4),
            approved_plan_context=_plan_context("row-a", "row-b"),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3
    later = [snap for kwargs, snap in calls if kwargs["review_round"] >= 2]
    assert later and all(
        "tag-names-satisfied-row" in snap.reasons and not snap.qualifies
        and snap.unsatisfied_row_ids == ("row-b",)
        for snap in later
    )


def test_head_matched_record_without_matrix_evidence_never_qualifies(tmp_path, monkeypatch):
    """`tag-verified-row`: absent evidence is never read as every row unverified."""
    _shape_evidence(monkeypatch, ["absent"] * 3)
    calls = _spy_classifier(monkeypatch)
    runner = _evidence_only_runner(4, 3)
    config = _config(tmp_path, max_rounds=4)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a"))
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert _coder_count(runner) == 3
    later = [snap for kwargs, snap in calls if kwargs["review_round"] >= 2]
    assert len(later) == 2 and all(
        "evidence-absent" in snap.reasons and not snap.qualifies for snap in later
    )
    assert all(snap.unsatisfied_row_ids == () for snap in later)


def test_row_explanation_from_a_head_race_downgrade_lists_every_row(tmp_path, monkeypatch):
    """`stall-row-explanation`: only the first downgraded row carries the race diagnostic."""
    race = {
        "row-a": ("stale/unverified", ("head-changed-during-correction",)),
        "row-b": ("stale/unverified", ()),
    }
    _shape_evidence(monkeypatch, [race, race])
    runner = _evidence_only_runner(3, 2)
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a", "row-b"))
    lines = {
        line.split("`")[1]: line.split(": ", 1)[1]
        for line in str(stopped.value).splitlines() if line.startswith("- `row-")
    }
    assert lines == {
        "row-a": "canonical-status:stale/unverified, head-changed-during-correction",
        "row-b": "canonical-status:stale/unverified",
    }
    # The diagnostic was read from the posted, head-bound coder record.
    coder = [r for r in _posted_records(runner, config) if r.metadata.role == "coder"][-1]
    assert [d["row_id"] for d in coder.metadata.risk_test_matrix_diagnostics] == ["row-a"]
    assert _coder_count(runner) == 2


def test_k3_replay_stops_exactly_at_the_third_consecutive_qualifying_review(tmp_path):
    """`stall-round-mapping` at K=3 through real posted records, then resume."""
    runner = _evidence_only_runner(4, 3)
    config = _config(tmp_path, pr_evidence_stall_rounds=3)
    context = _plan_context("row-a")
    with pytest.raises(HumanDecisionRequiredError) as first:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _coder_count(runner) == 3  # reviews 2, 3, 4 qualify; stop before coder 4
    records = [r for r in _posted_records(runner, config) if r.metadata.role == "coder"]
    assert [
        (r.metadata.round_number, r.metadata.evidence_stall["review_round"], r.metadata.evidence_stall["qualifies"])
        for r in records
    ] == [(2, 1, False), (3, 2, True), (4, 3, True)]
    runner.codex_outputs.append(_review(carried=["item-1"]))
    with pytest.raises(HumanDecisionRequiredError) as second:
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, pr_evidence_stall_rounds=3),
            approved_plan_context=context,
        )
    assert str(first.value) == str(second.value)
    assert _coder_count(runner) == 3


def _rewrite_posted_snapshot(runner, review_round, mutate):
    for comment in runner.pr_payload["comments"]:
        body = comment.get("body") if isinstance(comment, dict) else None
        if not isinstance(body, str) or "AGENT_LOOP_META: " not in body:
            continue
        head, rest = body.split("AGENT_LOOP_META: ", 1)
        encoded, tail = rest.split(" -->", 1)
        payload = decode_mapping(encoded)
        stall = payload.get("evidence_stall")
        if payload.get("role") == "coder" and isinstance(stall, dict) and stall.get("review_round") == review_round:
            mutate(payload)
            comment["body"] = head + "AGENT_LOOP_META: " + encode_mapping(payload) + " -->" + tail
            return
    raise AssertionError(f"no posted snapshot for review round {review_round}")


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda payload: payload.pop("evidence_stall"), id="missing"),
        pytest.param(lambda payload: payload["evidence_stall"].update(review_round=7), id="misnumbered"),
        pytest.param(
            lambda payload: payload["evidence_stall"].update(reasons=["checks-failing"]),
            id="contradictory",
        ),
    ],
)
def test_k3_replay_resets_at_a_missing_or_invalid_posted_snapshot(tmp_path, mutate):
    """`stall-window-reset` / `stall-round-mapping`: corrupt history ends the window walk."""
    runner = _evidence_only_runner(4, 3)
    context = _plan_context("row-a")
    config = _config(tmp_path, max_rounds=4, pr_evidence_stall_rounds=0)
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _coder_count(runner) == 3
    _rewrite_posted_snapshot(runner, 2, mutate)
    # Review 4 + reviews 3 and 2 would be three; the corrupt round 2 ends the walk.
    runner.codex_outputs.extend([_review(carried=["item-1"]), _review(carried=["item-1"])])
    runner.claude_outputs.extend([_coder(1)])
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(
            runner, pr_number=77, config=_config(tmp_path, max_rounds=8, pr_evidence_stall_rounds=3),
            approved_plan_context=context,
        )
    assert _coder_count(runner) == 4  # one more coder turn before reviews 3, 4, 5 fill the window


@pytest.mark.parametrize("disqualifier", ["stalled-check", "machine-obligation"])
def test_k3_window_restarts_after_a_round_disqualified_by_a_check_or_obligation(
    tmp_path, monkeypatch, disqualifier
):
    """`stall-window-reset` at K=3: review 3 is disqualified, so the stop waits for 4, 5, 6."""
    runner = _evidence_only_runner(6, 5)
    if disqualifier == "stalled-check":
        # A real dispatch-path fetch returns an infrastructure-stalled board for
        # review 3 only; it disqualifies the round without leaving an obligation.
        _checks_switch(monkeypatch, runner, {3: _STALLED_RUNS})
    else:
        _inject_obligation(monkeypatch, "alembic-migration", at_rounds={3})
    config = _config(tmp_path, pr_evidence_stall_rounds=3)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a"))
    assert _coder_count(runner) == 5
    snapshots = _stall_snapshots(runner, config)
    assert [snapshots[n]["qualifies"] for n in (1, 2, 3, 4, 5)] == [False, True, False, True, True]
    expected = "checks-failing" if disqualifier == "stalled-check" else "machine-obligation-open"
    assert expected in snapshots[3]["reasons"]


def _spy_decision(monkeypatch):
    """Record the review rounds that reach the stall decision on the dispatch path."""
    rounds = []
    real = pr_loop_module._pr_evidence_stall_decision

    def decide(*args, **kwargs):
        rounds.append(kwargs["round_number"])
        return real(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "_pr_evidence_stall_decision", decide)
    return rounds


def _forbidden_side_effects(commands):
    return [
        cmd for cmd in commands
        if cmd[:2] == ["git", "push"]
        or cmd[:3] == ["gh", "pr", "merge"]
        or (cmd[:3] == ["gh", "pr", "review"] and "--approve" in cmd)
    ]


def test_stall_stop_dispatches_no_coder_push_or_approval_after_the_final_review(tmp_path, monkeypatch):
    """`stall-trigger`: forbidden effects at the stop, asserted on recorded commands."""
    rounds = _spy_decision(monkeypatch)
    runner = _evidence_only_runner(3, 2)
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a"))
    commands = [cmd for cmd, _cwd in runner.commands]
    last_review = max(i for i, cmd in enumerate(commands) if cmd[:1] == ["codex"])
    after = commands[last_review + 1:]
    assert not any(cmd[:1] == ["claude"] for cmd in after)
    assert not _forbidden_side_effects(commands)
    assert rounds == [1, 2, 3]
    # The stopping round wrote no coder record of its own.
    assert max(_stall_snapshots(runner, config)) == 2


def test_all_approved_board_finishes_before_any_stall_decision(tmp_path, monkeypatch):
    """`stall-precedence`: approval wins although the previous round qualified."""
    rounds = _spy_decision(monkeypatch)
    runner = FakeRunner(
        claude_outputs=[_coder(1)] * 2,
        codex_outputs=[_review([_tagged()]), _review(carried=["item-1"]), _review(resolved=["item-1"])],
    )
    config = _config(tmp_path)
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a")) == 0
    assert rounds == [1, 2]
    assert _stall_snapshots(runner, config)[2]["qualifies"] is True


def test_live_evidence_freeze_takes_precedence_and_a_frozen_rerun_never_reaches_the_stall(
    tmp_path, monkeypatch
):
    """`stall-precedence`: no stall stop at a frozen head."""
    from test_orchestrator_pr import _EVIDENCE_TEXT, _evidence_review

    rounds = _spy_decision(monkeypatch)
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[
            _evidence_review(state="blocking", blocking=[_tagged()], evidence=[_EVIDENCE_TEXT]),
            _evidence_review(state="blocking", dispositions=[
                {"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "blocking", "note": "No signed evidence yet."},
            ]),
        ],
    )
    config = _config(tmp_path)
    context = _plan_context("row-a")
    with pytest.raises(HumanDecisionRequiredError) as frozen:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert "frozen at head" in str(frozen.value)
    assert "stalled on evidence only" not in str(frozen.value)
    assert rounds == [1]
    coders = _coder_count(runner)
    with pytest.raises(HumanDecisionRequiredError) as rerun:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    assert "stalled on evidence only" not in str(rerun.value)
    assert rounds == [1]
    assert _coder_count(runner) == coders


def test_step_back_sibling_escalation_takes_precedence_over_a_filled_stall_window(tmp_path, monkeypatch):
    """`stall-precedence`: the step-back escalation stops first at the round the stall would."""
    from test_orchestrator_pr import _StepBackRunner, _step_back_coders

    def review(text, *, resolved=()):
        return _review([_tagged(text)], resolved=resolved)

    rounds = _spy_decision(monkeypatch)
    runner = _StepBackRunner(
        claude_outputs=_step_back_coders(),
        codex_outputs=[
            review("gap one at src/spool.py:124-131"),
            review("gap two at src/spool.py:131-136", resolved=["item-1"]),
            review("gap three at src/spool.py:140", resolved=["item-2"]),
            review("another branch at src/spool.py:134", resolved=["item-3"]),
        ],
    )
    config = _config(tmp_path, pr_evidence_stall_rounds=3)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=_plan_context("row-a"))
    message = str(raised.value)
    assert "human decision required" in message and "--pr-step-back-rounds 0" in message
    assert "stalled on evidence only" not in message
    # Reviews 2 and 3 qualified, so review 4 would have filled the K=3 window.
    snapshots = _stall_snapshots(runner, config)
    assert snapshots[2]["qualifies"] is True and snapshots[3]["qualifies"] is True
    assert rounds == [1, 2, 3]
    assert _coder_count(runner) == 3
