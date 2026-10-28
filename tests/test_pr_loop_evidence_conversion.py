"""PR-loop evidence conversion to the exact-head human evidence freeze (#1325)."""

from __future__ import annotations

import dataclasses
import json
import re
import sys
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.pr_loop as pr_loop_module
from coding_review_agent_loop.converted_capture_state import capture_review_context
from coding_review_agent_loop.agents.registry import agent_display_name
from coding_review_agent_loop.cli import build_parser, run_pr_loop
from coding_review_agent_loop.comment_rendering import render_risk_test_matrix_section
from coding_review_agent_loop.config import config_from_args
from coding_review_agent_loop.errors import AgentLoopError, HumanDecisionRequiredError
from coding_review_agent_loop.evidence_conversion import capture_classification_payload
from coding_review_agent_loop.pr_loop_support import (
    admitted_pr_records,
    authenticated_pr_comments,
    evidence_authority_from_records,
)
from coding_review_agent_loop.review_rounds import PartialReviewRoundError
from coding_review_agent_loop.protocol import (
    SemanticRiskCoverageClaim,
    SemanticRiskCoverageClaims,
    SignedEvidenceReference,
    UnresolvedReviewItem,
    ReviewItemDisposition,
    derive_risk_test_matrix_evidence,
    parse_risk_test_matrix,
    risk_test_matrix_identity,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _extract_round_metadata_records,
)
from coding_review_agent_loop.unresolved_items import (
    _apply_unresolved_item_dispositions,
    _upsert_converted_evidence_obligation,
)

from agent_loop_helpers import FakeRunner, make_config, structured_coder_followup
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.panel_evidence as panel_evidence

ACTOR = ("coding-review-agent-loop", 424242)
START_HEAD = "abc123"


# --- fixtures ---------------------------------------------------------------


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
    return execution_policy.make_approved_plan_context(
        "Approved plan.\n\n" + render_risk_test_matrix_section(matrix),
        source_locator="test approved plan",
        risk_test_matrix_contract_version=1,
        risk_test_matrix_payload=matrix.to_payload(),
        risk_test_matrix_changes_payload=(),
        risk_test_matrix_identity=identity,
        risk_test_matrix_boundary_digest=identity,
    )


def _observation(ref: str, outcome: str, head: str) -> SimpleNamespace:
    return SimpleNamespace(
        execution_ref=ref,
        receipt_id=f"receipt-{ref}-{head}",
        command=("python3", "-m", "pytest", "tests/test_x.py", "-q"),
        normalized_command="python3 -m pytest tests/test_x.py -q",
        outcome=outcome,
        provenance="parent-observed",
        turn_id="turn-current",
        attribution={
            "state": "current-head", "head": head, "tracked_digest": "tree", "stable": True,
            "untracked_input": False, "caveats": [],
        },
        environment_state="not-compared",
        superseded_by=None,
        caveats=(),
        wrapper_bootstrap="verified",
        inner_exec="started",
        suite_start="verified",
    )


def _derived(context, plan: dict[str, str], head: str):
    """Real derivation for synthetic observations; ``plan`` maps row -> kind.

    ``limited`` selects an incomplete run (a capture limitation), ``failed``
    an authoritative failure, ``verified`` an admissible passing receipt and
    ``missing`` supplies no claim.
    """
    outcomes = {"limited": "incomplete", "failed": "failed", "verified": "passed"}
    claims, catalog = [], []
    for row_id, kind in sorted(plan.items()):
        if kind == "missing":
            continue
        ref = f"ref-{row_id}"
        catalog.append(_observation(ref, outcomes[kind], head))
        claims.append(SemanticRiskCoverageClaim(
            row_id=row_id,
            execution_refs=(ref,),
            test_identifiers=(f"tests/test_{row_id}.py::test_it",),
            test_locations=(f"tests/test_{row_id}.py",),
            workflow_path_claim="agent-loop pr",
            outcome_assertions=("The outcome holds.",),
            forbidden_effect_assertions=("No forbidden effect.",),
        ))
    result = derive_risk_test_matrix_evidence(
        matrix=context.risk_test_matrix_payload,
        claims=SemanticRiskCoverageClaims(tuple(claims)),
        observations=tuple(o for o in catalog if o.outcome == "passed"),
        execution_catalog=tuple(catalog),
        invocation_id="turn-current",
        current_head=head,
        current_tree_digest="tree",
        authenticated_checkout_head=head,
        authenticated_tree_clean=True,
        expected_identity=context.risk_test_matrix_identity,
    )
    return dataclasses.replace(result, bound_head_sha=head)


def _install_derivation(monkeypatch, context, plans):
    """Replace follow-up derivation with one real derivation per coder turn."""
    remaining = list(plans)

    def derive(parsed, *, head_sha, **_kwargs):
        plan = remaining.pop(0)
        result = _derived(context, plan, head_sha)
        return dataclasses.replace(
            parsed,
            risk_test_matrix_evidence=result.evidence,
            risk_test_matrix_diagnostics=result.diagnostics,
        ), result

    monkeypatch.setattr(pr_loop_module, "_derive_authenticated_risk_evidence_for_coder", derive)
    return remaining


def _implementation_record(context, plan: dict[str, str], *, head: str = START_HEAD, author=ACTOR) -> dict:
    """The issue-implementation coder record (round 1) as issue mode posts it."""
    result = _derived(context, plan, head)
    raw = json.dumps({
        "schema_version": 1, "kind": "issue_implementation", "state": "blocking",
        "summary": "Implemented.", "pr_number": 77,
        "human_requirements": {"addressed_ids": [], "checked_discussion_directly": False},
        "human_requirement_dispositions": [], "tests_run": [],
    })
    metadata = PostedRoundMetadata(
        flow="pr", role="coder", agent=agent_display_name("claude"), round_number=1,
        subject=head, prior_items=(), raw_structured_coder_response=raw,
        risk_test_matrix_evidence=result.evidence.to_payload(),
        risk_test_matrix_evidence_full_round=1,
        risk_test_matrix_diagnostics=tuple(d.to_payload() for d in result.diagnostics),
        risk_test_matrix_capture_classification=capture_classification_payload(result, head_sha=head),
    )
    return _comment(_attach_round_metadata("Implemented the approved plan.", metadata), author=author)


def _comment(body: str, *, author=ACTOR, index: int = 0) -> dict:
    return {
        "author": {"login": author[0]},
        # Earlier than every comment the fake runner posts, as in real history.
        "createdAt": f"2026-01-01T00:00:{index:02d}Z",
        "body": body,
        "_rest_author_id": author[1],
    }


def _review(blocking=(), *, carried=(), resolved=(), kept=(), state=None, hr=False, reviewer="OpenAI Codex"):
    state = state or ("blocking" if blocking or carried or kept else "approved")
    dispositions = [{"item_id": i, "disposition": "resolved", "note": "Accepted."} for i in resolved]
    dispositions += [
        {"item_id": i, "disposition": "blocking", "note": "Still no admissible citation for the row."}
        for i in carried
    ]
    dispositions += [
        {"item_id": i, "disposition": "blocking", "note": "Operator evidence for this row is still needed."}
        for i in kept
    ]
    return (
        json.dumps({
            "schema_version": 1, "kind": "pr_review", "state": state, "summary": "review",
            "blocking_items": list(blocking), "same_pr_followups": [], "future_followups": [],
            "prior_item_dispositions": dispositions,
        })
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if hr else "")
        + f"\n<!-- AGENT_STATE: {state} -->\n-- {reviewer}"
    )


def _tagged(*rows, text="Capture an admissible passing test observation for the rows"):
    return {"text": text, "evidence_row_ids": list(rows)}


def _coder(*numbers):
    return structured_coder_followup(
        addressed_items=[f"item-{n}" for n in numbers], summary="Retried capture."
    )


def _signed(body: str, index: int) -> dict:
    return {
        "author": {"login": "maintainer"},
        "createdAt": f"2026-10-02T00:{index:02d}:00Z",
        "url": f"https://github.com/OWNER/REPO/pull/77#issuecomment-{900 + index}",
        "body": f"{body}\n\n-- Human Reviewer",
    }


def _config(tmp_path, **overrides):
    overrides.setdefault("max_rounds", 8)
    overrides.setdefault("pr_non_convergence_rounds", 0)
    overrides.setdefault("reviewer", "codex")
    return make_config(tmp_path, coder="claude", **overrides)


def _records(runner):
    return _extract_round_metadata_records(
        [SimpleNamespace(body=comment["body"]) for comment in runner.pr_payload["comments"]],
        flow="pr",
    )


def _agent_count(runner, name):
    return sum(1 for cmd, _cwd in runner.commands if cmd[:1] == [name])


def _prompts(runner, name):
    return ["\n".join(cmd) for cmd, _cwd in runner.commands if cmd[:1] == [name]]


def _evidence_items(items, row_id=None):
    return [
        item for item in items
        if item.obligation_kind == "human-exact-head-evidence"
        and (row_id is None or item.converted_row_id == row_id)
    ]


def _auto_converted_runner(tmp_path, monkeypatch, context, *, extra_codex=(), plan=None, **runner_kwargs):
    """Two capture-limited attempts (issue record + one follow-up), then conversion."""
    plan = plan or {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[
            _review([_tagged("row-a")]),
            _review(carried=["item-1"]),
            _review(kept=["item-2"], state="blocking"),
            *extra_codex,
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
        **runner_kwargs,
    )
    _install_derivation(monkeypatch, context, [plan])
    return runner


# --- auto-convert -------------------------------------------------------------


def test_two_capture_limited_attempts_convert_before_the_stall_stop_and_reach_the_freeze(
    tmp_path, monkeypatch
):
    """`auto-convert` with defaults K=2 and N=2, from the real issue-implementation record."""
    context = _plan_context("row-a", "row-b")
    runner = _auto_converted_runner(tmp_path, monkeypatch, context)
    config = _config(tmp_path)
    assert config.pr_evidence_stall_rounds == 2 and config.pr_evidence_capture_attempts == 2
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    message = str(stopped.value)
    # The stop is the freeze's exit 4, not the stall message.
    assert "frozen at head abc123-coder-1" in message
    assert "stalled on evidence only" not in message
    assert _agent_count(runner, "claude") == 1  # no coder turn after the conversion
    records = _records(runner)
    phases = [(r.metadata.role, r.metadata.phase, r.metadata.round_number) for r in records]
    conversion_index = next(
        i for i, r in enumerate(records) if r.metadata.phase == "evidence-conversion"
    )
    after = phases[conversion_index:]
    # In review round 2: the conversion record, the repeated round-2 review, the freeze.
    assert after[0] == ("summary", "evidence-conversion", 2)
    assert ("reviewer", "authoritative", 2) in after[1:-1] or any(
        role == "reviewer" and number == 2 for role, _phase, number in after[1:-1]
    )
    assert after[-1] == ("summary", "evidence-freeze", 2)
    assert not any(r.metadata.role == "coder" and r.metadata.round_number >= 3 for r in records)
    conversion = records[conversion_index].metadata
    audit = conversion.evidence_conversion
    assert audit.source == "auto" and audit.row_ids == ("row-a",)
    assert audit.head == "abc123-coder-1"
    assert audit.rows[0].attempt_record_rounds == (1, 2)
    assert audit.rows[0].owners == ("Codex",)
    assert audit.rows[0].converted_item_ids == ("item-1",)
    # The conversion round's own stall snapshot is kept on the conversion record.
    assert conversion.conversion_stall_snapshot["review_round"] == 2
    assert conversion.conversion_stall_snapshot["bound_attempt_round"] == 2
    assert conversion.conversion_review_round == 2
    # Attempts were paired through bound_attempt_round; the issue record has no snapshot.
    coder_records = [r for r in records if r.metadata.role == "coder"]
    assert coder_records[0].metadata.evidence_stall is None
    assert coder_records[1].metadata.evidence_stall["bound_attempt_round"] == 1
    # Exactly one obligation names the row and the head; the tagged item is converted.
    freeze = records[-1].metadata
    [obligation] = _evidence_items(freeze.prior_items)
    assert obligation.converted_row_id == "row-a"
    assert obligation.lifecycle == "evidence_frozen" and obligation.candidate_head_sha == "abc123-coder-1"
    assert "row-a" in obligation.text and "abc123-coder-1" in obligation.text
    finding = next(item for item in freeze.prior_items if item.item_id == "item-1")
    assert finding.status == "converted"
    # The approved matrix is unchanged by conversion.
    assert context.risk_test_matrix_identity == risk_test_matrix_identity(
        parse_risk_test_matrix(context.risk_test_matrix_payload)
    )
    # The repeated reviewer saw the converted obligation, not the converted finding.
    repeat_prompt = _prompts(runner, "codex")[-1]
    assert "converted row `row-a`" in repeat_prompt
    # Authority was admitted through live REST identity reads.
    commands = [cmd for cmd, _cwd in runner.commands]
    assert any(cmd[:3] == ["gh", "api", "user"] for cmd in commands)


def test_an_open_code_finding_blocks_conversion(tmp_path, monkeypatch):
    """`auto-convert` negative: nothing converts while a code finding is open."""
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1, 2), _coder(1, 2)],
        codex_outputs=[
            _review([_tagged("row-a"), "Guard the loader against a missing file."]),
            _review(carried=["item-1", "item-2"]),
            _review(resolved=["item-1", "item-2"]),
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan, plan])
    assert run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context) == 0
    assert not any(r.metadata.phase == "evidence-conversion" for r in _records(runner))
    assert _agent_count(runner, "claude") == 2


def test_a_failing_row_never_converts_and_the_stall_stop_names_operator_conversion(tmp_path, monkeypatch):
    """`no-convert-test-failure` in the loop, and the stop message's option 2."""
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "failed", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged("row-a")]), _review(carried=["item-1"])],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan])
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    message = str(stopped.value)
    assert "stalled on evidence only" in message
    assert "`--pr-convert-evidence-rows row-a`" in message
    assert not any(r.metadata.phase == "evidence-conversion" for r in _records(runner))


# --- operator-convert ------------------------------------------------------------


def _stalled_two_reviewer_runner(tmp_path, monkeypatch, context):
    # Neither row converts automatically: row-a failed, row-b has no claim.
    plan = {"row-a": "failed", "row-b": "missing"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged("row-a")]), _review(carried=["item-1"])],
        gemini_outputs=[
            _review(reviewer="Google Gemini"),
            _review(resolved=["item-1"], reviewer="Google Gemini"),
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan])
    config = _config(tmp_path, reviewer=("codex", "gemini"))
    with pytest.raises(HumanDecisionRequiredError, match="stalled on evidence only"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    return runner, config


@pytest.mark.parametrize("rows", [("row-z",), ("row-a", "row-z"), ("item-1",)])
def test_invalid_operator_rows_fail_closed_before_any_agent_or_comment(tmp_path, monkeypatch, rows):
    """`operator-convert`: an invalid or mixed list fails as a whole before any agent runs."""
    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_two_reviewer_runner(tmp_path, monkeypatch, context)
    commands_before = len(runner.commands)
    comments_before = len(runner.pr_payload["comments"])
    with pytest.raises(AgentLoopError, match="rejected") as raised:
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, pr_convert_evidence_rows=rows),
        )
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert "row-z" in str(raised.value) or "item-1" in str(raised.value)
    new_commands = [cmd for cmd, _cwd in runner.commands[commands_before:]]
    assert not any(cmd[:1] in (["codex"], ["gemini"], ["claude"]) for cmd in new_commands)
    assert len(runner.pr_payload["comments"]) == comments_before
    assert not any(r.metadata.phase == "evidence-conversion" for r in _records(runner))


def test_operator_conversion_persists_an_audit_and_reaches_the_freeze(tmp_path, monkeypatch):
    """`operator-convert`: valid rows convert; an untagged row is owned by the whole board."""
    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_two_reviewer_runner(tmp_path, monkeypatch, context)
    runner.codex_outputs.append(_review(kept=["item-2", "item-3"], state="blocking"))
    runner.gemini_outputs.append(
        _review(kept=["item-2", "item-3"], state="blocking", reviewer="Google Gemini")
    )
    coders_before = _agent_count(runner, "claude")
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, pr_convert_evidence_rows=("row-a", "row-b")),
        )
    assert _agent_count(runner, "claude") == coders_before
    records = _records(runner)
    [conversion] = [r for r in records if r.metadata.phase == "evidence-conversion"]
    audit = conversion.metadata.evidence_conversion
    assert audit.source == "operator" and audit.row_ids == ("row-a", "row-b")
    owners = {row.row_id: row.owners for row in audit.rows}
    assert owners == {"row-a": ("Codex",), "row-b": ("Codex", "Gemini")}
    assert conversion.metadata.conversion_review_round is None
    # Both reviewers reviewed the converted ledger fresh after the conversion record.
    fresh = [
        r.metadata.agent for r in records
        if r.metadata.role == "reviewer" and r.index > conversion.index
    ]
    assert sorted(fresh) == ["Codex", "Gemini"]
    freeze = records[-1].metadata
    assert freeze.phase == "evidence-freeze"
    assert sorted(item.converted_row_id for item in _evidence_items(freeze.prior_items)) == ["row-a", "row-b"]
    # The rerun command never replays the one-shot flag.
    from coding_review_agent_loop.managed_ci import _ONE_SHOT_OPERATOR_RECOVERY_OPTIONS

    assert "--pr-convert-evidence-rows" in _ONE_SHOT_OPERATOR_RECOVERY_OPTIONS


def test_operator_flags_parse_validate_and_default(tmp_path):
    base = [
        "pr", "77", "--repo", "OWNER/REPO", "--reviewer", "codex",
        "--codex-dir", str(tmp_path / "codex"), "--dangerous-agent-permissions",
    ]
    parser = build_parser()
    parsed = config_from_args(parser.parse_args(base), FakeRunner())
    assert parsed.pr_evidence_capture_attempts == 2 and parsed.pr_convert_evidence_rows == ()
    parsed = config_from_args(
        parser.parse_args([*base, "--pr-convert-evidence-rows", "row-a,row-b", "--pr-evidence-capture-attempts", "0"]),
        FakeRunner(),
    )
    assert parsed.pr_convert_evidence_rows == ("row-a", "row-b")
    assert parsed.pr_evidence_capture_attempts == 0
    for bad in ("", "row-a,,row-b", "row-a,row-a", "row a"):
        with pytest.raises(SystemExit):
            parser.parse_args([*base, "--pr-convert-evidence-rows", bad])
    with pytest.raises(AgentLoopError, match="--pr-evidence-capture-attempts must be"):
        make_config(tmp_path, pr_evidence_capture_attempts=-1)


# --- signed-answer-satisfies ---------------------------------------------------------


def _frozen_after_auto_conversion(tmp_path, monkeypatch, context, *, config=None, **runner_kwargs):
    runner = _auto_converted_runner(tmp_path, monkeypatch, context, **runner_kwargs)
    config = config or _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    return runner, config


def test_a_signed_answer_cleared_by_the_owner_verifies_the_row_as_operator_signed(tmp_path, monkeypatch):
    """`signed-answer-satisfies`."""
    from coding_review_agent_loop.pr_loop_support import _head_bound_unsatisfied_rows

    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    head = "abc123-coder-1"
    runner.pr_payload["comments"].append(
        _signed(f"Ran the row-a scenario at {head} against the live stack: all checks passed.", 1)
    )
    runner.codex_outputs.append(_review(resolved=["item-2"], hr=True))
    commands_before = len(runner.commands)
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    # No push or coder turn while frozen; the owning reviewer judged the evidence.
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands[commands_before:])
    records = _records(runner)
    [release] = [r for r in records if r.metadata.phase == "evidence-release"]
    assert release.metadata.evidence_release.reason == "evidence-cleared"
    [reference] = release.metadata.signed_evidence_references
    assert (reference.row_id, reference.head, reference.matrix_identity) == (
        "row-a", head, context.risk_test_matrix_identity,
    )
    assert reference.clearing_reviewers == ("Codex",)
    assert reference.signed_requirement_ids
    assert f"`row-a`: verified (operator-signed at `{head}`)" in release.body
    tombstone = next(item for item in release.metadata.prior_items if item.item_id == "item-2")
    assert (tombstone.status, tombstone.lifecycle, tombstone.candidate_head_sha) == (
        "resolved", "cleared", head,
    )
    # After a restart the admitted reference overlays the canonical evidence.
    admitted = admitted_pr_records(runner, config, pr_number=77)
    references, audits = evidence_authority_from_records(admitted)
    assert [ref.reference_id for ref in references] == [reference.reference_id]
    assert [audit.row_ids for audit in audits] == [("row-a",)]
    unsatisfied = _head_bound_unsatisfied_rows(admitted, head, context, signed_references=references)
    assert "row-a" not in unsatisfied.rows


def test_clearance_without_new_signed_input_mints_no_reference(tmp_path, monkeypatch):
    """`signed-answer-satisfies` negative: no reference without newly surfaced signed input."""
    from coding_review_agent_loop.pr_loop_support import mint_signed_evidence_references

    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=2, row_id="row-a", matrix_identity="d" * 64, head_sha="h1",
        owners=("Codex",), source_round=2, explanation="",
    )
    frozen = dataclasses.replace(obligation, lifecycle="evidence_frozen", candidate_head_sha="h1")
    cleared = dataclasses.replace(obligation, status="resolved", lifecycle="cleared", candidate_head_sha="h1")
    kwargs = dict(
        prior_items=[frozen], reconciled_items=[cleared],
        clearances=[(obligation.obligation_identity, "h1")],
        dispositions_by_item={"item-2": [ReviewItemDisposition("item-2", "Codex", "resolved")]},
        response_head="h1", round_number=3,
    )
    assert mint_signed_evidence_references(
        **kwargs, surfaced_requirement_ids=("hr-1",), freeze_signed_requirement_ids=("hr-1",)
    ) == ()
    assert mint_signed_evidence_references(
        **kwargs, surfaced_requirement_ids=("hr-1", "hr-2"), freeze_signed_requirement_ids=None
    ) == ()
    [reference] = mint_signed_evidence_references(
        **kwargs, surfaced_requirement_ids=("hr-1", "hr-2"), freeze_signed_requirement_ids=("hr-1",)
    )
    assert reference.signed_requirement_ids == ("hr-2",)


# --- head changes ------------------------------------------------------------


def _push_external_head(runner, head):
    runner.pr_payload["headRefOid"] = head
    runner.git_head = head
    runner.simulate_new_process()


def test_an_external_push_releases_a_frozen_converted_row_with_the_same_item_id(tmp_path, monkeypatch):
    """`convert-head-change`: frozen at H, released by a push to H2, frozen again at H2."""
    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    frozen_at_h = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-a")[0]
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    _install_derivation(monkeypatch, context, [{"row-a": "missing", "row-b": "verified"}])
    runner.codex_outputs.extend([_review(kept=["item-2"], state="blocking")] * 2)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    freeze = _records(runner)[-1].metadata
    [reopened] = _evidence_items(freeze.prior_items, "row-a")
    assert (reopened.item_id, reopened.obligation_identity) == (
        frozen_at_h.item_id, frozen_at_h.obligation_identity,
    )
    assert reopened.candidate_head_sha == "fedcba9"
    assert _agent_count(runner, "claude") == 2
    assert "Bounded recovery capture" in _prompts(runner, "claude")[-1]


def test_a_cleared_row_reopens_at_a_new_head_with_bounded_capture(tmp_path, monkeypatch):
    """`convert-head-change` and `convert-head-change-no-dispatch` (resume after an external push)."""
    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    runner.pr_payload["comments"].append(_signed("Live run at abc123-coder-1 passed.", 1))
    runner.codex_outputs.append(_review(resolved=["item-2"], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    tombstone = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-a")[0]
    assert tombstone.lifecycle == "cleared"

    _push_external_head(runner, "fedcba9")
    # The reviewer approves the new head with no code finding.
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(structured_coder_followup(
        human_requirement_ids=_signed_requirement_ids(runner, config),
    ))
    _install_derivation(monkeypatch, context, [{"row-a": "missing", "row-b": "verified"}])
    runner.codex_outputs.extend([_review(kept=["item-2"], state="blocking", hr=True)] * 2)
    coders_before = _agent_count(runner, "claude")
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _agent_count(runner, "claude") == coders_before + 1  # one bounded capture
    # Every reviewer prompt in that round lists the reopened obligation.
    prompt = _prompts(runner, "codex")[-1]
    assert "[item-2] converted row `row-a`" in prompt
    freezes = [r for r in _records(runner) if r.metadata.phase == "evidence-freeze"]
    assert [r.metadata.subject for r in freezes] == ["abc123-coder-1", "fedcba9"]
    [reopened] = _evidence_items(freezes[-1].metadata.prior_items, "row-a")
    assert (reopened.item_id, reopened.obligation_identity) == (tombstone.item_id, tombstone.obligation_identity)
    assert reopened.lifecycle == "evidence_frozen"
    # The answer for H does not satisfy H2: no merge, and a restart at H2 stays frozen.
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    [still] = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-a")
    assert still.item_id == tombstone.item_id


def test_the_finalization_gate_reopens_a_stale_tombstone_and_freezes(tmp_path, monkeypatch):
    """`convert-head-change-no-dispatch`: the gate's own reconciliation is defense in depth."""
    from coding_review_agent_loop.pr_loop_support import _evidence_freeze_gate

    from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit

    context = _plan_context("row-a")
    config = _config(tmp_path)
    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=2, row_id="row-a", matrix_identity=context.risk_test_matrix_identity,
        head_sha="old-head", owners=("Codex",), source_round=2, explanation="",
    )
    # The orchestrator's admitted conversion audit backs the obligation.
    audit = EvidenceConversionAudit(
        source="operator", matrix_identity=context.risk_test_matrix_identity, head="old-head",
        rows=(ConversionAuditRow(
            row_id="row-a", obligation_identity=obligation.obligation_identity,
            obligation_item_id="item-2", owners=("Codex",), converted_item_ids=(),
            attempt_record_rounds=(),
        ),),
    )
    conversion = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=2, subject="old-head",
        prior_items=(obligation,), state="blocking", phase="evidence-conversion",
        evidence_conversion=audit,
    )
    runner = FakeRunner(
        pr_payload={"comments": [_comment(_attach_round_metadata("Converted.", conversion))]}
    )
    stale = dataclasses.replace(obligation, status="resolved", lifecycle="cleared", candidate_head_sha="old-head")
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        _evidence_freeze_gate(
            runner, config=config, pr_number=77, round_number=3, head_sha=START_HEAD,
            items=[stale], surfaced_requirement_ids=(), collect_requirement_ids=lambda _ctx: (),
            allowed_rounds=8, watch_failure_extension_used=False, watch_head_extension_used=False,
            approved_plan_context=context,
        )
    freeze = _records(runner)[-1].metadata
    [frozen] = _evidence_items(freeze.prior_items)
    assert (frozen.item_id, frozen.lifecycle, frozen.candidate_head_sha) == (
        "item-2", "evidence_frozen", START_HEAD,
    )


# --- signed references on terminal records ---------------------------------------


def _two_rows_frozen(tmp_path, monkeypatch, context, *, extra_claude=(), plans=()):
    plan = {"row-a": "limited", "row-b": "limited"}
    runner = FakeRunner(
        claude_outputs=[_coder(1), *extra_claude],
        codex_outputs=[
            _review([_tagged("row-a", "row-b")]),
            _review(carried=["item-1"]),
            _review(kept=["item-2", "item-3"], state="blocking"),
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan, *plans])
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    runner.pr_payload["comments"].append(_signed("Live run at abc123-coder-1: row-a passed.", 1))
    return runner, config


def _reference_rows(metadata):
    return sorted((ref.row_id, ref.head) for ref in metadata.signed_evidence_references)


def test_partial_clearance_keeps_the_reference_on_the_republished_freeze_and_after_restart(
    tmp_path, monkeypatch
):
    """`signed-reference-terminal-records`: A cleared, B kept (re-published freeze)."""
    context = _plan_context("row-a", "row-b")
    runner, config = _two_rows_frozen(tmp_path, monkeypatch, context)
    runner.codex_outputs.append(
        _review(resolved=["item-2"], kept=["item-3"], state="blocking", hr=True)
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    latest = _records(runner)[-1].metadata
    assert latest.phase == "evidence-freeze"
    assert _reference_rows(latest) == [("row-a", "abc123-coder-1")]
    items = {item.converted_row_id: item for item in _evidence_items(latest.prior_items)}
    assert items["row-a"].lifecycle == "cleared" and items["row-b"].lifecycle == "evidence_frozen"
    # Restart without new signed input: still frozen on B, A stays operator-signed.
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    references, _audits = evidence_authority_from_records(admitted_pr_records(runner, config, pr_number=77))
    assert sorted((ref.row_id, ref.head) for ref in references) == [("row-a", "abc123-coder-1")]


def _signed_requirement_ids(runner, config):
    from coding_review_agent_loop.github import get_pr_review_context, human_requirement_id
    from coding_review_agent_loop.response_validation import _build_requirements_context

    requirements = _build_requirements_context(
        target_issue_context=None,
        pr_context=get_pr_review_context(runner, config=config, pr_number=77),
        parent_issue_context=None,
    ).effective_requirements
    return [human_requirement_id(requirement) for requirement in requirements]


def test_clearance_with_a_new_code_finding_keeps_the_reference_on_the_release_and_later_records(
    tmp_path, monkeypatch
):
    """`signed-reference-terminal-records`: A cleared plus a new code finding (release for findings)."""
    context = _plan_context("row-a", "row-b")
    runner, config = _two_rows_frozen(
        tmp_path, monkeypatch, context,
        plans=[{"row-a": "limited", "row-b": "limited"}],
    )
    runner.claude_outputs.append(structured_coder_followup(
        addressed_items=["item-4"], summary="Guarded the loader.",
        human_requirement_ids=_signed_requirement_ids(runner, config),
    ))
    runner.codex_outputs.extend([
        _review(["Guard the loader against a missing file."], resolved=["item-2"], kept=["item-3"], hr=True),
        _review(resolved=["item-4"], kept=["item-2", "item-3"], state="blocking", hr=True),
    ])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1-coder-2"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    records = _records(runner)
    [release] = [r for r in records if r.metadata.phase == "evidence-release"]
    assert release.metadata.evidence_release.reason == "findings"
    assert _reference_rows(release.metadata) == [("row-a", "abc123-coder-1")]
    # Carried cumulatively onto the next terminal record, at the new head ...
    later_freeze = records[-1].metadata
    assert later_freeze.subject == "abc123-coder-1-coder-2"
    assert _reference_rows(later_freeze) == [("row-a", "abc123-coder-1")]
    # ... where the answer for the old head no longer verifies row-a.
    assert {item.converted_row_id for item in _evidence_items(later_freeze.prior_items)
            if item.lifecycle == "evidence_frozen"} == {"row-a", "row-b"}
    runner.simulate_new_process()
    references, _audits = evidence_authority_from_records(admitted_pr_records(runner, config, pr_number=77))
    assert [ref.row_id for ref in references] == ["row-a"]


# --- tool supersession -------------------------------------------------------------


def test_tool_captured_verification_at_a_new_head_supersedes_then_reopens_at_the_next(tmp_path, monkeypatch):
    """`convert-tool-supersede`."""
    context = _plan_context("row-a", "row-b")
    runner = _auto_converted_runner(tmp_path, monkeypatch, context)
    plan = {"row-a": "limited", "row-b": "verified"}
    # The conversion repeat raises a code finding, so the coder runs and its
    # canonical evidence at the new head verifies row-a with tool receipts.
    runner.codex_outputs[2] = _review(["Guard the loader against a missing file."], kept=["item-2"])
    runner.claude_outputs.append(_coder(3))
    _install_derivation(monkeypatch, context, [plan, {"row-a": "verified", "row-b": "verified"}])
    runner.codex_outputs.append(_review(resolved=["item-3"]))
    config = _config(tmp_path)
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    h2 = "abc123-coder-1-coder-2"
    final_ledger = [r for r in records if r.metadata.subject == h2][-1].metadata.prior_items
    [superseded] = _evidence_items(final_ledger, "row-a")
    assert (superseded.status, superseded.lifecycle, superseded.candidate_head_sha) == (
        "resolved", "cleared", h2,
    )
    assert any("tool-captured" in note for note in superseded.notes)
    assert not any(r.metadata.phase == "evidence-freeze" for r in records)
    # Later the head moves to H3, where tool evidence does not verify row-a.
    _push_external_head(runner, "fedcba9")
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    [reopened] = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-a")
    assert (reopened.item_id, reopened.obligation_identity) == (superseded.item_id, superseded.obligation_identity)


# --- convert-owner-removed -----------------------------------------------------------


def _frozen_obligation(owners):
    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=2, row_id="row-a", matrix_identity="e" * 64, head_sha="h1",
        owners=owners, source_round=2, explanation="",
    )
    return dataclasses.replace(obligation, lifecycle="evidence_frozen", candidate_head_sha="h1")


def _clears(obligation, resolvers, board):
    dispositions = {
        obligation.item_id: [ReviewItemDisposition(obligation.item_id, name, "resolved") for name in resolvers]
    }
    clearances: list = []
    kept, _future = _apply_unresolved_item_dispositions(
        [obligation], dispositions, evidence_response_head="h1",
        configured_reviewers=board, evidence_clearances=clearances,
    )
    return bool(clearances) and kept[0].lifecycle == "cleared"


def test_a_departed_owner_cannot_strand_a_converted_obligation():
    """`convert-owner-removed`: remaining owners must all clear; with none left, the whole board."""
    obligation = _frozen_obligation(("Codex", "Gemini"))
    # Both owners on the board: one is not enough.
    assert not _clears(obligation, ["Codex"], ("Codex", "Gemini"))
    assert _clears(obligation, ["Codex", "Gemini"], ("Codex", "Gemini"))
    # Gemini left the board: Codex alone clears it; a non-owner never does.
    assert _clears(obligation, ["Codex"], ("Codex", "Claude"))
    assert not _clears(obligation, ["Claude"], ("Codex", "Claude"))
    # No stored owner remains: every configured reviewer must clear it.
    assert not _clears(obligation, ["Claude"], ("Claude", "Antigravity"))
    assert _clears(obligation, ["Claude", "Antigravity"], ("Claude", "Antigravity"))


# --- foreign and genuine authority -----------------------------------------------------


def _history_before_conversion(tmp_path, monkeypatch, context):
    """Issue record, review 1, coder record 2, review 2 -- then the round limit stops it."""
    plan = {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged("row-a")]), _review(carried=["item-1"])],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan])
    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path, max_rounds=2), approved_plan_context=context)
    runner.simulate_new_process()
    if hasattr(runner, "_agent_loop_authenticated_actor"):
        delattr(runner, "_agent_loop_authenticated_actor")
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    return runner


def _make_foreign(runner, predicate):
    for comment in runner.pr_payload["comments"]:
        if predicate(comment["body"]):
            comment["author"] = {"login": "mallory"}
            comment["_rest_author_id"] = 999


def test_genuine_authority_is_admitted_after_restart_through_live_identity_reads(tmp_path, monkeypatch):
    """`genuine-authority-admitted`: GraphQL comments, empty actor cache, no recovery-phase record."""
    context = _plan_context("row-a", "row-b")
    runner = _history_before_conversion(tmp_path, monkeypatch, context)
    assert not any(
        "author_id" in comment or "id" in (comment.get("author") or {})
        for comment in runner.pr_payload["comments"]
    )
    commands_before = len(runner.commands)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    new_commands = [cmd for cmd, _cwd in runner.commands[commands_before:]]
    assert any(cmd[:3] == ["gh", "api", "user"] for cmd in new_commands)
    assert any(cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2] for cmd in new_commands)
    [conversion] = [r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]
    assert conversion.metadata.evidence_conversion.rows[0].attempt_record_rounds == (1, 2)


def test_an_incomplete_identity_read_on_resume_stops_before_any_agent(tmp_path, monkeypatch):
    context = _plan_context("row-a", "row-b")
    runner = _history_before_conversion(tmp_path, monkeypatch, context)
    runner.serve_rest_issue_comments = False
    commands_before = len(runner.commands)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert not any(
        cmd[:1] in (["codex"], ["claude"]) for cmd, _cwd in runner.commands[commands_before:]
    )
    assert not any(r.metadata.phase == "evidence-conversion" for r in _records(runner))


def test_a_mid_run_reread_does_not_admit_an_uncovered_record(tmp_path, monkeypatch):
    context = _plan_context("row-a", "row-b")
    runner = _history_before_conversion(tmp_path, monkeypatch, context)
    config = _config(tmp_path)
    # The REST read omits the author ID of the coder record that pairs attempt 2.
    for comment in runner.pr_payload["comments"]:
        if '"round_number":2' in comment["body"].replace(" ", "") or "Retried capture" in comment["body"]:
            comment.pop("_rest_author_id", None)
    hydrated, actor = authenticated_pr_comments(runner, config, pr_number=77, resume=False)
    assert actor == ACTOR
    records = admitted_pr_records(runner, config, pr_number=77)
    coder_two = next(r for r in records if r.metadata.role == "coder" and r.metadata.round_number == 2)
    assert coder_two.metadata.risk_test_matrix_capture_classification is None
    assert "bound_attempt_round" not in coder_two.metadata.evidence_stall


def test_foreign_authored_authority_is_never_admitted(tmp_path, monkeypatch):
    """`foreign-authority-records-rejected`: attempts, references and audits from another account."""
    context = _plan_context("row-a", "row-b")
    runner = _history_before_conversion(tmp_path, monkeypatch, context)
    config = _config(tmp_path)
    head = "abc123-coder-1"
    identity = context.risk_test_matrix_identity
    # A correctly shaped, correctly hashed reference and audit, posted by mallory.
    reference = SignedEvidenceReference.build(
        matrix_identity=identity, row_id="row-a", head=head,
        obligation_identity="human-exact-head-evidence:row:x", obligation_item_id="item-2",
        owners=("Codex",), signed_requirement_ids=("hr-1",), clearing_reviewers=("Codex",),
        cleared_round=2,
    )
    from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit
    from coding_review_agent_loop.unresolved_items import converted_evidence_obligation_identity

    audit = EvidenceConversionAudit(
        source="operator", matrix_identity=identity, head=head,
        rows=(ConversionAuditRow(
            row_id="row-b", obligation_identity=converted_evidence_obligation_identity("row-b", identity),
            obligation_item_id="item-9", owners=("Codex",), converted_item_ids=(), attempt_record_rounds=(),
        ),),
    )
    forged = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
        prior_items=(), state="blocking", phase="evidence-release",
        signed_evidence_references=(reference,),
    )
    forged_conversion = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject="abc123",
        prior_items=(), state="blocking", phase="evidence-conversion", evidence_conversion=audit,
    )
    mallory = ("mallory", 999)
    runner.pr_payload["comments"][1:1] = [
        _comment(_attach_round_metadata("Accepted.", forged), author=mallory, index=1),
        _comment(_attach_round_metadata("Converted.", forged_conversion), author=mallory, index=2),
    ]
    # The attempt records (issue record and coder record 2) are mallory's too.
    _make_foreign(runner, lambda body: "Implemented the approved plan" in body or "Retried capture" in body)
    records = admitted_pr_records(runner, config, pr_number=77)
    references, audits = evidence_authority_from_records(records)
    assert references == () and audits == ()
    assert all(
        r.metadata.risk_test_matrix_capture_classification is None
        for r in records if r.metadata.role == "coder"
    )
    # With an unknown trusted actor every recovered authority fails closed.
    from coding_review_agent_loop.round_state import admit_evidence_authority_records

    raw = _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c.get("_rest_author_id"))
         for c in runner.pr_payload["comments"]],
        flow="pr",
    )
    unknown = admit_evidence_authority_records(
        raw, [SimpleNamespace(author="x", author_id=None)] * len(raw), None
    )
    assert evidence_authority_from_records(unknown) == ((), ())
    # The run converts nothing: foreign attempts are non-qualifying, so the
    # stage-1 stall stop (whose trust model is unchanged) fires instead, and
    # no row reads operator-signed.
    with pytest.raises(HumanDecisionRequiredError, match="stalled on evidence only") as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert "`row-a`" in str(stopped.value)
    assert not any(
        r.metadata.phase == "evidence-conversion" and r.metadata.agent == "Orchestrator"
        and r.metadata.evidence_conversion is not None and r.metadata.subject == head
        for r in _records(runner)
    )


# --- conversion-repeat-fresh-review ------------------------------------------------------


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_the_conversion_repeat_re_runs_every_reviewer_fresh(tmp_path, monkeypatch, policy):
    """`conversion-repeat-fresh-review`: R1's earlier same-head approval is not carried."""
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        # Codex approves head abc123 in round 1; its approval is carried in
        # round 2 (the head did not move), so its only other turn is the repeat.
        codex_outputs=[_review(), _review(kept=["item-2"], state="blocking")],
        gemini_outputs=[
            _review([_tagged("row-a")], reviewer="Google Gemini"),
            _review(carried=["item-1"], reviewer="Google Gemini"),
            _review(kept=["item-2"], state="blocking", reviewer="Google Gemini"),
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
        advance_pr_head_on_coder_followup=False,
    )
    _install_derivation(monkeypatch, context, [plan])
    config = _config(tmp_path, reviewer=("codex", "gemini"), pr_review_policy=policy)
    # The selective policy's finalization requires an issue-bound plan handoff,
    # which this PR-only fixture lacks, so it stops at that pre-existing
    # qualification guard after the repeat instead of reaching the freeze.
    expected = (
        (HumanDecisionRequiredError, "frozen at head abc123")
        if policy == "all-reviewers"
        else (AgentLoopError, "Approved-plan/handoff identity is missing")
    )
    with pytest.raises(expected[0], match=expected[1]):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    records = _records(runner)
    [conversion] = [r for r in records if r.metadata.phase == "evidence-conversion"]
    before = [
        r.metadata.agent for r in records
        if r.metadata.role == "reviewer" and r.metadata.round_number == 2 and r.index < conversion.index
    ]
    assert "Codex" not in before  # R1's round-1 approval was carried before the conversion
    repeat = [r for r in records if r.metadata.role == "reviewer" and r.index > conversion.index]
    assert sorted(r.metadata.agent for r in repeat) == ["Codex", "Gemini"]
    assert _agent_count(runner, "codex") == 2 and runner.codex_outputs == []
    # Codex was freshly invoked in the repeat with the converted obligation in its prompt.
    codex_prompts = _prompts(runner, "codex")
    assert "converted row `row-a`" in codex_prompts[-1]
    assert not any("carried" in (r.metadata.phase or "") for r in repeat)
    assert all(r.metadata.phase != "evidence-response" for r in repeat)


def test_a_restart_between_the_conversion_record_and_the_repeat_resumes_a_fresh_board(tmp_path, monkeypatch):
    """`conversion-repeat-fresh-review` after a restart (conversion_review_round recovery)."""
    context = _plan_context("row-a", "row-b")
    runner = _auto_converted_runner(tmp_path, monkeypatch, context)
    runner.codex_outputs.pop()  # the repeat's review is not available: the run stops
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    records = _records(runner)
    assert records[-1].metadata.phase == "evidence-conversion"
    runner.simulate_new_process()
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    codex_before = _agent_count(runner, "codex")
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    assert _agent_count(runner, "codex") == codex_before + 1
    assert "converted row `row-a`" in _prompts(runner, "codex")[-1]
    assert len([r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]) == 1
    assert _agent_count(runner, "claude") == 1


# --- convert-partial-multirow --------------------------------------------------------


def test_a_partly_converted_finding_keeps_its_residual_row_until_that_row_converts(tmp_path, monkeypatch):
    """`convert-partial-multirow`."""
    context = _plan_context("row-a", "row-b")
    runner = FakeRunner(
        claude_outputs=[_coder(1), _coder(1)],
        codex_outputs=[
            _review([_tagged("row-a", "row-b")]),
            _review(carried=["item-1"]),
            # Conversion repeat: row-b is still demanded by item-1.
            _review(carried=["item-1"], kept=["item-2"]),
            _review(carried=["item-1"], kept=["item-2"]),
            _review(kept=["item-2", "item-3"], state="blocking"),
        ],
        pr_payload={"comments": [_implementation_record(context, {"row-a": "limited", "row-b": "failed"})]},
    )
    _install_derivation(monkeypatch, context, [
        {"row-a": "limited", "row-b": "limited"},
        {"row-a": "limited", "row-b": "limited"},
    ])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    conversions = [r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]
    assert [c.metadata.evidence_conversion.row_ids for c in conversions] == [("row-a",), ("row-b",)]
    first_ledger = {item.item_id: item for item in conversions[0].metadata.prior_items}
    finding = first_ledger["item-1"]
    assert finding.status == "blocking"
    assert finding.evidence_row_ids == ("row-a", "row-b")  # provenance kept
    assert finding.residual_evidence_row_ids == ("row-b",)
    final = {item.item_id: item for item in conversions[1].metadata.prior_items}
    assert final["item-1"].status == "converted"
    assert final["item-1"].converted_row_ids == ("row-a", "row-b")
    assert _agent_count(runner, "claude") == 2


# --- review round 1 regressions -------------------------------------------------


def test_a_conversion_record_with_a_rejected_audit_is_not_resumed_as_converted(tmp_path, monkeypatch):
    """item-3: an invalid audit is treated as unconverted through the actual resume."""
    from coding_review_agent_loop.evidence_conversion import EvidenceConversionAudit

    context = _plan_context("row-a", "row-b")
    runner = _auto_converted_runner(tmp_path, monkeypatch, context)
    runner.codex_outputs.pop()  # stop before the repeat's review
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    records = _records(runner)
    conversion = records[-1]
    assert conversion.metadata.phase == "evidence-conversion"
    # Rewrite the audit so it no longer binds to the record head: it decodes as rejected.
    bad_audit = dataclasses.replace(conversion.metadata.evidence_conversion, head="feedface")
    comment = next(c for c in runner.pr_payload["comments"] if "risk-matrix evidence converted" in c["body"].lower())
    comment["body"] = _attach_round_metadata(
        "Converted.", dataclasses.replace(conversion.metadata, evidence_conversion=bad_audit)
    )
    assert _records(runner)[-1].metadata.evidence_conversion is None
    runner.simulate_new_process()
    # The pre-conversion review round resumes; conversion is decided afresh from
    # the attempt history, and the fresh repeat runs.
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=_config(tmp_path), approved_plan_context=context)
    valid = [
        r for r in _records(runner)
        if r.metadata.phase == "evidence-conversion" and r.metadata.evidence_conversion is not None
    ]
    assert len(valid) == 1 and valid[0].metadata.evidence_conversion.row_ids == ("row-a",)
    assert _agent_count(runner, "claude") == 1


def test_an_invalid_operator_list_posts_nothing_even_with_a_pending_board_change(tmp_path, monkeypatch):
    """item-5: validation runs read-only before any startup write."""
    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_two_reviewer_runner(tmp_path, monkeypatch, context)
    comments_before = len(runner.pr_payload["comments"])
    commands_before = len(runner.commands)
    swapped = dataclasses.replace(
        _config(tmp_path, reviewer=("codex", "antigravity")),
        pr_convert_evidence_rows=("row-a", "row-z"),
    )
    with pytest.raises(AgentLoopError, match="rejected: row-z"):
        run_pr_loop(runner, pr_number=77, config=swapped, approved_plan_context=context)
    assert len(runner.pr_payload["comments"]) == comments_before
    assert not any(
        cmd[:3] == ["gh", "pr", "comment"] or cmd[:1] in (["codex"], ["antigravity"], ["agy"], ["claude"])
        for cmd, _cwd in runner.commands[commands_before:]
    )
    # Control: without the entry preflight the same invocation writes the
    # board-change link before the round-start validation rejects the list.
    monkeypatch.setattr(pr_loop_module, "preflight_operator_evidence_conversion", lambda *a, **k: None)
    with pytest.raises(AgentLoopError, match="rejected: row-z"):
        run_pr_loop(runner, pr_number=77, config=swapped, approved_plan_context=context)
    assert len(runner.pr_payload["comments"]) > comments_before


@pytest.mark.parametrize("rows", [("row-a",), ("row-a", "row-c")])
def test_a_satisfied_converted_row_is_rejected_alone_or_in_a_mixed_list(tmp_path, monkeypatch, rows):
    """item-6: only currently unsatisfied, not-yet-converted rows are accepted."""
    context = _plan_context("row-a", "row-b", "row-c")
    plan = {"row-a": "limited", "row-b": "verified", "row-c": "missing"}
    runner = _auto_converted_runner(tmp_path, monkeypatch, context, plan=plan)
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    runner.pr_payload["comments"].append(_signed("Live run at abc123-coder-1 passed.", 1))
    runner.codex_outputs.append(_review(resolved=["item-2"], hr=True))
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    comments_before = len(runner.pr_payload["comments"])
    with pytest.raises(AgentLoopError, match="rejected: row-a"):
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, pr_convert_evidence_rows=rows),
        )
    assert len(runner.pr_payload["comments"]) == comments_before


def test_after_the_owner_leaves_the_board_every_current_reviewer_must_clear(tmp_path, monkeypatch):
    """item-1 / `convert-owner-removed` through a PR-loop board-change resume.

    Gemini alone owns the converted row; the rerun's board swaps Gemini for
    Antigravity, so the ledger runs through the board-change view.  One
    current reviewer's clearance keeps the freeze; both clear it.
    """
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[
            _review(),
            _review(resolved=["item-1"]),
            _review(kept=["item-2"], state="blocking"),
        ],
        gemini_outputs=[
            _review([_tagged("row-a")], reviewer="Google Gemini"),
            _review(carried=["item-1"], reviewer="Google Gemini"),
            _review(kept=["item-2"], state="blocking", reviewer="Google Gemini"),
        ],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    _install_derivation(monkeypatch, context, [plan])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=_config(tmp_path, reviewer=("codex", "gemini")),
        )
    [obligation] = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-a")
    assert obligation.resolution_owners == ("Gemini",)
    runner.pr_payload["comments"].append(_signed("Live run at abc123-coder-1 passed.", 1))
    swapped = _config(tmp_path, reviewer=("codex", "antigravity"))
    runner.codex_outputs.append(_review(resolved=["item-2"], hr=True))
    runner.antigravity_outputs.append(_review(kept=["item-2"], state="blocking", hr=True, reviewer="Antigravity"))
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=swapped, approved_plan_context=context)
    from coding_review_agent_loop.board_change import BOARD_CHANGE_LINK_HEADING

    assert any(c["body"].startswith(BOARD_CHANGE_LINK_HEADING) for c in runner.pr_payload["comments"])
    latest = _records(runner)[-1].metadata
    [still] = _evidence_items(latest.prior_items, "row-a")
    assert still.lifecycle == "evidence_frozen"
    assert still.resolution_owners == ("Gemini",)  # never reassigned to the new board
    assert not latest.signed_evidence_references
    # Both current reviewers resolve it after further signed input.
    runner.pr_payload["comments"].append(_signed("Added the deploy log for abc123-coder-1.", 2))
    runner.codex_outputs.append(_review(resolved=["item-2"], hr=True))
    runner.antigravity_outputs.append(_review(resolved=["item-2"], hr=True, reviewer="Antigravity"))
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=swapped, approved_plan_context=context) == 0
    [release] = [r for r in _records(runner) if r.metadata.phase == "evidence-release"]
    assert [ref.clearing_reviewers for ref in release.metadata.signed_evidence_references] == [
        ("Antigravity", "Codex")
    ]


# --- review round 2 regressions -------------------------------------------------


def _issue_bound_plan(context):
    """The issue-side plan record and PR handoff the plain `agent-loop pr` path recovers."""
    from coding_review_agent_loop.issue_pr_handoff import format_issue_pr_handoff_comment

    plan_record = _attach_round_metadata(
        context.canonical_text,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject=panel_evidence._plan_subject(context.canonical_text),
            canonical_plan=context.canonical_text,
            raw_structured_coder_response=context.canonical_text,
        ),
    )
    handoff = format_issue_pr_handoff_comment(
        issue_number=56, pr_number=77, pr_url="https://github.com/OWNER/REPO/pull/77",
        pr_head_sha=START_HEAD, flow="approved-plan-implementation", plan_hash=context.plan_hash,
    )
    return [
        {"author": {"login": "coding-review-agent-loop"}, "createdAt": "2026-01-01T00:00:01Z", "body": plan_record},
        {"author": {"login": "coding-review-agent-loop"}, "createdAt": "2026-01-01T00:00:02Z", "body": handoff},
    ]


def test_cli_operator_conversion_recovers_the_issue_bound_plan_before_validating(tmp_path, monkeypatch):
    """item-8: ordinary `agent-loop pr N --pr-convert-evidence-rows` without an injected plan."""
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "failed", "row-b": "missing"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged("row-a")]), _review(carried=["item-1"])],
        pr_payload={"body": "Fixes #56", "comments": [_implementation_record(context, plan)]},
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: _issue_bound_plan(context)},
    )
    _install_derivation(monkeypatch, context, [plan])
    config = _config(tmp_path)
    with pytest.raises(HumanDecisionRequiredError, match="stalled on evidence only"):
        run_pr_loop(runner, pr_number=77, config=config)
    comments_before = len(runner.pr_payload["comments"])
    commands_before = len(runner.commands)
    with pytest.raises(AgentLoopError, match="rejected: row-z"):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, pr_convert_evidence_rows=("row-b", "row-z")))
    assert len(runner.pr_payload["comments"]) == comments_before
    assert not any(cmd[:1] in (["codex"], ["claude"]) for cmd, _cwd in runner.commands[commands_before:])
    # The valid row converts; row-a's failing finding then keeps the loop
    # going, so the run is stopped once the next review is exhausted.
    _install_derivation(monkeypatch, context, [plan])
    runner.codex_outputs.append(_review(carried=["item-1"], kept=["item-2"]))
    runner.claude_outputs.append(_coder(1))
    with pytest.raises(AgentLoopError):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, pr_convert_evidence_rows=("row-b",)))
    [conversion] = [r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]
    assert conversion.metadata.evidence_conversion.source == "operator"
    assert conversion.metadata.evidence_conversion.row_ids == ("row-b",)


def _untagged_operator_obligation(context):
    from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit

    _ledger, obligation, _ = _upsert_converted_evidence_obligation(
        [], item_number=1, row_id="row-b", matrix_identity=context.risk_test_matrix_identity,
        head_sha=START_HEAD, owners=("Codex",), source_round=1, explanation="",
    )
    audit = EvidenceConversionAudit(
        source="operator", matrix_identity=context.risk_test_matrix_identity, head=START_HEAD,
        rows=(ConversionAuditRow(
            row_id="row-b", obligation_identity=obligation.obligation_identity,
            obligation_item_id="item-1", owners=("Codex",), converted_item_ids=(),
            attempt_record_rounds=(),
        ),),
    )
    record = PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=1, subject=START_HEAD,
        prior_items=(obligation,), state="blocking", phase="evidence-conversion",
        evidence_conversion=audit,
    )
    return obligation, audit, _comment(_attach_round_metadata("Converted.", record))


@pytest.mark.parametrize("failure", ["rest-unavailable", "uncovered-record"])
def test_the_gate_keeps_an_untagged_conversion_when_authority_history_is_unavailable(tmp_path, failure):
    """item-9: an unavailable reread never removes an established evidence barrier."""
    from coding_review_agent_loop.pr_loop_support import _evidence_freeze_gate

    context = _plan_context("row-a", "row-b")
    obligation, _audit, comment = _untagged_operator_obligation(context)
    if failure == "uncovered-record":
        comment.pop("_rest_author_id")
    runner = FakeRunner(
        pr_payload={"comments": [comment]},
        serve_rest_issue_comments=(False if failure == "rest-unavailable" else None),
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        _evidence_freeze_gate(
            runner, config=_config(tmp_path), pr_number=77, round_number=2, head_sha=START_HEAD,
            items=[obligation], surfaced_requirement_ids=(), collect_requirement_ids=lambda _ctx: (),
            allowed_rounds=8, watch_failure_extension_used=False, watch_head_extension_used=False,
            approved_plan_context=context,
        )
    [frozen] = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-b")
    assert frozen.lifecycle == "evidence_frozen"


def test_a_failed_reread_after_an_untagged_operator_conversion_keeps_the_freeze(tmp_path, monkeypatch):
    """item-9: PR-loop path; the run freezes instead of approving or merging."""
    import coding_review_agent_loop.pr_loop_support as support

    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "verified", "row-b": "missing"}
    runner = FakeRunner(
        codex_outputs=[_review(kept=["item-1"], state="blocking")],
        pr_payload={"comments": [_implementation_record(context, plan)]},
    )
    real_apply = support._apply_evidence_conversion

    def apply_then_break_identity_reads(*args, **kwargs):
        outcome = real_apply(*args, **kwargs)
        runner.serve_rest_issue_comments = False  # every later identity read fails
        return outcome

    monkeypatch.setattr(support, "_apply_evidence_conversion", apply_then_break_identity_reads)
    config = dataclasses.replace(_config(tmp_path), pr_convert_evidence_rows=("row-b",))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    commands = [cmd for cmd, _cwd in runner.commands]
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd in commands)
    assert not any(cmd[:3] == ["gh", "pr", "review"] and "--approve" in cmd for cmd in commands)
    [frozen] = _evidence_items(_records(runner)[-1].metadata.prior_items, "row-b")
    assert frozen.lifecycle == "evidence_frozen"
    assert frozen.resolution_owners == ("Codex",)


# --- conversion visibility boundary (#1664) ---------------------------------------


def _agy(*args, **kwargs):
    return _review(*args, reviewer="Antigravity", **kwargs)


def _gem(*args, **kwargs):
    return _review(*args, reviewer="Google Gemini", **kwargs)


def _repeat_review(reviewer="OpenAI Codex"):
    return _review(kept=["item-2"], state="blocking", reviewer=reviewer)


def _carried_panel_runner(tmp_path, monkeypatch, context, *, plan=None, repeat=True, **config_overrides):
    """The PR #1778 shape under primary-then-panel with parallel review.

    Round 1: the primary Codex approves.  Round 2 (panel): Gemini approves and
    Antigravity blocks on row-a.  Round 3 selects the full board but invokes
    only Antigravity: Codex and Gemini are carried as approved at the unchanged
    head.  Antigravity's evidence-only review is the round's only publication.
    """
    plan = plan or {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review(), *([_repeat_review()] if repeat else [])],
        gemini_outputs=[_gem(), *([_repeat_review("Google Gemini")] if repeat else [])],
        antigravity_outputs=[
            _agy([_tagged("row-a")]), _agy(carried=["item-1"]),
            *([_repeat_review("Antigravity")] if repeat else []),
        ],
        pr_payload={"body": "Fixes #56", "comments": [_implementation_record(context, plan)]},
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: _issue_bound_plan(context)},
        advance_pr_head_on_coder_followup=False,
    )
    _install_derivation(monkeypatch, context, [plan])
    config = _config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="primary-then-panel", primary_reviewer="codex", review_parallel=True,
        **config_overrides,
    )
    return runner, config


_BOARD_COMMANDS = {"Codex": "codex", "Gemini": "gemini", "Antigravity": "agy"}


def _conversion(runner):
    return [r for r in _records(runner) if r.metadata.phase == "evidence-conversion"][-1]


def _reviews_after(runner, index):
    return [r for r in _records(runner) if r.metadata.role == "reviewer" and r.index > index]


def _assert_fresh_repeat(runner, conversion):
    repeat = _reviews_after(runner, conversion.index)
    assert sorted(r.metadata.agent for r in repeat) == ["Antigravity", "Codex", "Gemini"]
    assert {r.metadata.round_number for r in repeat} == {conversion.metadata.round_number}
    for name, command in _BOARD_COMMANDS.items():
        assert "converted row `row-a`" in _prompts(runner, command)[-1], name


def _interrupt_repeat(monkeypatch, *, publish=0):
    """Stop the run inside the conversion repeat, after its scheduler checkpoint.

    ``publish`` reviewers of the repeat publish first, as a partial pass.
    """
    real = pr_loop_module._launch_reviewer_turns
    armed = [True]

    def launch(runner, reviewers, *args, **kwargs):
        records = _records(runner)
        conversions = [r for r in records if r.metadata.phase == "evidence-conversion"]
        if armed[0] and conversions and not _reviews_after(runner, conversions[-1].index):
            armed[0] = False
            if publish:
                real(runner, list(reviewers)[:publish], *args, **kwargs)
            raise AgentLoopError("interrupted inside the conversion repeat")
        return real(runner, reviewers, *args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "_launch_reviewer_turns", launch)


def test_a_primary_only_round_then_auto_conversion_reruns_the_full_board_without_a_partial_stop(
    tmp_path, monkeypatch
):
    """`inproc-primary-only-conversion-repeat` (#1664): the PR #1778 shape continues."""
    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123") as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert "partially published" not in str(stopped.value)
    conversion = _conversion(runner)
    assert conversion.metadata.round_number == 3
    assert conversion.metadata.evidence_conversion.source == "auto"
    records = _records(runner)
    before = [
        r.metadata.agent for r in records
        if r.metadata.role == "reviewer" and r.metadata.round_number == 3 and r.index < conversion.index
    ]
    assert before == ["Antigravity"]  # Codex and Gemini were carried, not invoked
    _assert_fresh_repeat(runner, conversion)
    # The round number and budgets did not advance; no coder ran after the conversion.
    assert max(r.metadata.round_number for r in records) == 3
    assert _agent_count(runner, "claude") == 1
    assert runner.codex_outputs == runner.gemini_outputs == runner.antigravity_outputs == []
    assert records[-1].metadata.phase == "evidence-freeze"


def _assert_unresolved_boundary_refusal(message: str) -> None:
    """The refusal names the unresolved conversion boundary, neutral about why (#1676)."""
    assert message.startswith("Review round 3 on PR #77 is partially published (Antigravity already posted), ")
    assert "this round's evidence-conversion record could not be admitted as a trusted boundary" in message
    assert "either its identity could not be read or it is not a trusted, accepted conversion record" in message
    assert "withheld" not in message
    assert "missing or no longer validates" not in message


def _fail_boundary_identity_reads(monkeypatch) -> None:
    """Only the visibility boundary's identity read fails from now on.

    The evidence gate still admits the conversion, so the repeat pass starts.
    """
    import coding_review_agent_loop.pr_loop_support as support

    real_admit = support.admitted_pr_records_status

    def boundary_read_fails(*args, **kwargs):
        if sys._getframe(1).f_code.co_name == "conversion_visibility_boundary":
            raise AgentLoopError("comment identity read failed")
        return real_admit(*args, **kwargs)

    monkeypatch.setattr(support, "admitted_pr_records_status", boundary_read_fails)


def test_an_untrusted_conversion_record_does_not_move_the_visibility_boundary(tmp_path, monkeypatch):
    """`forged-conversion-boundary`: a foreign conversion record hides no real peer."""
    import coding_review_agent_loop.pr_loop_support as support

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    real_apply = support._apply_evidence_conversion

    def apply_then_forge(*args, **kwargs):
        outcome = real_apply(*args, **kwargs)
        # The record now reads as another account's: admission strips its audit.
        runner.pr_payload["comments"][-1]["author"] = {"login": "mallory"}
        runner.pr_payload["comments"][-1]["_rest_author_id"] = 999
        return outcome

    monkeypatch.setattr(support, "_apply_evidence_conversion", apply_then_forge)
    with pytest.raises(PartialReviewRoundError) as refused:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert "Antigravity already posted" in str(refused.value)
    _assert_unresolved_boundary_refusal(str(refused.value))
    conversion = _conversion(runner)
    assert runner.pr_payload["comments"][conversion.index]["author"] == {"login": "mallory"}
    assert _reviews_after(runner, conversion.index) == []
    assert _agent_count(runner, "codex") == 1 and _agent_count(runner, "gemini") == 1


def test_a_failed_identity_read_names_the_unresolved_conversion_boundary(tmp_path, monkeypatch):
    """`conversion-unresolved-refusal` (#1676): the guard still fires, with boundary wording."""
    import coding_review_agent_loop.pr_loop_support as support

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    real_apply = support._apply_evidence_conversion

    def apply_then_lose_reads(*args, **kwargs):
        outcome = real_apply(*args, **kwargs)
        _fail_boundary_identity_reads(monkeypatch)
        return outcome

    monkeypatch.setattr(support, "_apply_evidence_conversion", apply_then_lose_reads)
    with pytest.raises(PartialReviewRoundError) as refused:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    _assert_unresolved_boundary_refusal(str(refused.value))
    conversion = _conversion(runner)
    assert _reviews_after(runner, conversion.index) == []
    assert _agent_count(runner, "codex") == 1 and _agent_count(runner, "gemini") == 1


def test_a_restart_after_the_repeat_checkpoint_runs_the_full_board_fresh(tmp_path, monkeypatch):
    """`resume-after-repeat-checkpoint`: pre-conversion reviews are neither peers nor completed."""
    from coding_review_agent_loop.round_state import _resume_pr_round

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    _interrupt_repeat(monkeypatch)
    with pytest.raises(AgentLoopError, match="interrupted inside the conversion repeat"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    conversion = _conversion(runner)
    after = [r for r in _records(runner) if r.index > conversion.index]
    assert [r.metadata.phase for r in after] == ["scheduler-prelaunch"]
    resumed = _resume_pr_round(
        [
            SimpleNamespace(body=c["body"], author=c["author"]["login"], author_id=c["_rest_author_id"])
            for c in runner.pr_payload["comments"]
        ],
        head_sha=START_HEAD, configured_reviewers=("codex", "gemini", "antigravity"),
        trusted_actor=ACTOR,
    )
    assert resumed.round_number == 3
    assert resumed.completed_reviews == ()  # no pre-conversion review is a repeat review
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    _assert_fresh_repeat(runner, conversion)
    assert len([r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]) == 1
    assert _agent_count(runner, "claude") == 1


def test_a_partial_repeat_pass_still_refuses_and_lists_only_post_conversion_records(tmp_path, monkeypatch):
    """`repeat-pass-partial-isolation`: isolation still holds inside the repeat pass."""
    from coding_review_agent_loop.review_rounds import _review_round_spool

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    _interrupt_repeat(monkeypatch, publish=1)
    with pytest.raises(AgentLoopError, match="interrupted inside the conversion repeat"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    conversion = _conversion(runner)
    [published] = _reviews_after(runner, conversion.index)
    peer = published.metadata.agent
    # No withheld outcome survives for the remaining reviewers (another host).
    _review_round_spool(
        config, surface="pr", number=77, round_number=3, subject=START_HEAD
    ).discard()
    runner.simulate_new_process()
    launches = {name: _agent_count(runner, command) for name, command in _BOARD_COMMANDS.items()}
    with pytest.raises(PartialReviewRoundError) as refused:
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, review_parallel=False),
        )
    message = str(refused.value)
    assert f"({peer} already posted)" in message
    assert "Antigravity already posted" not in message or peer == "Antigravity"
    assert launches == {name: _agent_count(runner, command) for name, command in _BOARD_COMMANDS.items()}
    # Only records after the conversion are deletion targets.
    listed = [int(value) for value in re.findall(r"comment (\d+) https://", message)]
    assert listed
    conversion_id = 10_000 + conversion.index + 1
    assert all(comment_id > conversion_id for comment_id in listed)


def test_a_sequential_resume_with_a_failed_boundary_read_names_the_unresolved_boundary(
    tmp_path, monkeypatch
):
    """`conversion-unresolved-refusal`, sequential guard (#1676): wording only, still no launch."""
    from coding_review_agent_loop.review_rounds import _review_round_spool

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    _interrupt_repeat(monkeypatch, publish=1)
    with pytest.raises(AgentLoopError, match="interrupted inside the conversion repeat"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    _review_round_spool(
        config, surface="pr", number=77, round_number=3, subject=START_HEAD
    ).discard()
    runner.simulate_new_process()
    _fail_boundary_identity_reads(monkeypatch)
    launches = {name: _agent_count(runner, command) for name, command in _BOARD_COMMANDS.items()}
    with pytest.raises(PartialReviewRoundError) as refused:
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, review_parallel=False),
        )
    message = str(refused.value)
    assert message.startswith("Review round 3 on PR #77 is partially published (")
    assert "Antigravity" in message.split(" already posted)", 1)[0]
    assert "this round's evidence-conversion record could not be admitted as a trusted boundary" in message
    assert "withheld" not in message and "missing or no longer validates" not in message
    assert launches == {name: _agent_count(runner, command) for name, command in _BOARD_COMMANDS.items()}


def test_a_stale_pre_conversion_spool_outcome_is_discarded_and_never_replayed(tmp_path, monkeypatch):
    """`parallel-stale-spool`: staging discards the round's spool."""
    import coding_review_agent_loop.pr_loop_support as support
    from coding_review_agent_loop.review_rounds import _review_round_spool

    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    stale = _review().replace('"summary": "review"', '"summary": "STALE pre-conversion outcome"')
    real_apply = support._apply_evidence_conversion

    def apply_with_a_stale_spool(*args, **kwargs):
        outcome = real_apply(*args, **kwargs)
        spool = _review_round_spool(
            config, surface="pr", number=77, round_number=kwargs["round_number"], subject=kwargs["head_sha"]
        )
        spool.store("Codex", {"text": stale})
        assert spool.has_records()
        return outcome

    monkeypatch.setattr(support, "_apply_evidence_conversion", apply_with_a_stale_spool)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert not any("STALE pre-conversion outcome" in c["body"] for c in runner.pr_payload["comments"])
    assert _agent_count(runner, "codex") == 2  # freshly invoked in the repeat
    _assert_fresh_repeat(runner, _conversion(runner))


def _stalled_carried_panel(tmp_path, monkeypatch, context):
    """The carried-panel shape whose row cannot auto-convert: it stops on the stall."""
    runner, config = _carried_panel_runner(
        tmp_path, monkeypatch, context, plan={"row-a": "failed", "row-b": "verified"}, repeat=False,
    )
    with pytest.raises(HumanDecisionRequiredError, match="stalled on evidence only"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    records = _records(runner)
    assert [
        r.metadata.agent for r in records if r.metadata.role == "reviewer" and r.metadata.round_number == 3
    ] == ["Antigravity"]
    runner.simulate_new_process()
    runner.codex_outputs.append(_repeat_review())
    runner.gemini_outputs.append(_repeat_review("Google Gemini"))
    runner.antigravity_outputs.append(_repeat_review("Antigravity"))
    return runner, dataclasses.replace(config, pr_convert_evidence_rows=("row-a",))


@pytest.mark.parametrize("fresh_read", ["available", "failing"])
def test_an_operator_conversion_beside_a_public_same_round_review_runs_the_board_fresh(
    tmp_path, monkeypatch, fresh_read
):
    """`operator-conversion-visibility`: refreshed snapshot, or the own-record fallback."""
    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_carried_panel(tmp_path, monkeypatch, context)
    if fresh_read == "failing":
        real_read = pr_loop_module.get_pr_review_context
        failed = []

        def read(*args, **kwargs):
            records = _records(runner)
            conversions = [r for r in records if r.metadata.phase == "evidence-conversion"]
            if conversions and not _reviews_after(runner, conversions[-1].index):
                failed.append(len(runner.pr_payload["comments"]))
                raise AgentLoopError("GitHub read failed")
            return real_read(*args, **kwargs)

        monkeypatch.setattr(pr_loop_module, "get_pr_review_context", read)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    conversion = _conversion(runner)
    assert conversion.metadata.evidence_conversion.source == "operator"
    assert conversion.metadata.round_number == 3
    _assert_fresh_repeat(runner, conversion)
    if fresh_read == "failing":
        # Both the post-conversion refresh and the post-checkpoint visibility read failed.
        assert len(failed) >= 2


def test_a_restart_after_an_unfinished_operator_conversion_restores_the_full_board_repeat(
    tmp_path, monkeypatch
):
    """`operator-conversion-restart-full-board`: no one-shot flag on the restart."""
    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_carried_panel(tmp_path, monkeypatch, context)
    _interrupt_repeat(monkeypatch)
    with pytest.raises(AgentLoopError, match="interrupted inside the conversion repeat"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    conversion = _conversion(runner)
    assert conversion.metadata.conversion_review_round is None  # encoding unchanged
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123"):
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, pr_convert_evidence_rows=()),
        )
    _assert_fresh_repeat(runner, conversion)
    assert len([r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]) == 1


def test_a_partial_repeat_after_an_operator_conversion_restart_still_refuses(tmp_path, monkeypatch):
    """`operator-conversion-restart-full-board`: a repeat-pass peer still isolates the rest."""
    from coding_review_agent_loop.review_rounds import _review_round_spool

    context = _plan_context("row-a", "row-b")
    runner, config = _stalled_carried_panel(tmp_path, monkeypatch, context)
    _interrupt_repeat(monkeypatch, publish=1)
    with pytest.raises(AgentLoopError, match="interrupted inside the conversion repeat"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    conversion = _conversion(runner)
    [published] = _reviews_after(runner, conversion.index)
    _review_round_spool(
        config, surface="pr", number=77, round_number=3, subject=START_HEAD
    ).discard()
    runner.simulate_new_process()
    with pytest.raises(PartialReviewRoundError) as refused:
        run_pr_loop(
            runner, pr_number=77, approved_plan_context=context,
            config=dataclasses.replace(config, pr_convert_evidence_rows=(), review_parallel=False),
        )
    assert f"({published.metadata.agent} already posted)" in str(refused.value)


def _conversion_metadata(*, source, round_number, head=START_HEAD, review_round=None):
    from coding_review_agent_loop.evidence_conversion import ConversionAuditRow, EvidenceConversionAudit

    context = _plan_context("row-a")
    identity = context.risk_test_matrix_identity
    audit = EvidenceConversionAudit(
        source=source, matrix_identity=identity, head=head,
        rows=(ConversionAuditRow(
            row_id="row-a", obligation_identity=f"human-exact-head-evidence:row:{identity}:row-a",
            obligation_item_id="item-2", owners=("Codex",), converted_item_ids=(),
            attempt_record_rounds=(),
        ),),
    )
    return PostedRoundMetadata(
        flow="pr", role="summary", agent="Orchestrator", round_number=round_number, subject=head,
        prior_items=(), state="blocking", phase="evidence-conversion", evidence_conversion=audit,
        conversion_review_round=review_round,
    )


def test_conversion_repeat_round_recognizes_unfinished_auto_and_operator_conversions():
    """`operator-conversion-restart-full-board` (unit)."""
    from coding_review_agent_loop.pr_loop_support import conversion_repeat_round
    from coding_review_agent_loop.round_state import PostedRoundRecord, admit_evidence_authority_records

    def recs(*metadatas):
        return [PostedRoundRecord(index=i, metadata=m, body="") for i, m in enumerate(metadatas)]

    auto = _conversion_metadata(source="auto", round_number=3, review_round=3)
    operator = _conversion_metadata(source="operator", round_number=3)
    for conversion in (auto, operator):
        assert conversion_repeat_round(recs(conversion), round_number=3, head_sha=START_HEAD)
        assert not conversion_repeat_round(recs(conversion), round_number=4, head_sha=START_HEAD)
        assert not conversion_repeat_round(recs(conversion), round_number=3, head_sha="other")
        for later in (
            dataclasses.replace(conversion, phase="evidence-freeze", evidence_conversion=None),
            PostedRoundMetadata(
                flow="pr", role="coder", agent="Claude", round_number=3, subject=START_HEAD, prior_items=(),
            ),
        ):
            assert not conversion_repeat_round(recs(conversion, later), round_number=3, head_sha=START_HEAD)
    # A foreign operator record loses its audit at admission and never restores the repeat.
    raw = recs(operator)
    foreign = admit_evidence_authority_records(
        raw, [SimpleNamespace(author="mallory", author_id=999)], ACTOR
    )
    assert foreign[0].metadata.evidence_conversion is None
    assert not conversion_repeat_round(foreign, round_number=3, head_sha=START_HEAD)




def test_a_primary_only_publication_then_auto_conversion_reruns_the_gated_panel_fresh(
    tmp_path, monkeypatch
):
    """`inproc-primary-only-conversion-repeat`: the primary is the round's only publisher.

    Under primary-then-panel the panel waits for an exact-head primary
    approval, so rounds 1 and 2 invoke only Codex and pause Gemini and
    Antigravity.  Codex's round-2 evidence-only review is the only public
    same-round record when the auto conversion forces the full board.
    """
    context = _plan_context("row-a", "row-b")
    plan = {"row-a": "limited", "row-b": "verified"}
    runner = FakeRunner(
        claude_outputs=[_coder(1)],
        codex_outputs=[_review([_tagged("row-a")]), _review(carried=["item-1"]), _repeat_review()],
        gemini_outputs=[_repeat_review("Google Gemini")],
        antigravity_outputs=[_repeat_review("Antigravity")],
        pr_payload={"body": "Fixes #56", "comments": [_implementation_record(context, plan)]},
        issue_payloads_by_number={56: {"number": 56}},
        issue_comments_by_number={56: _issue_bound_plan(context)},
        advance_pr_head_on_coder_followup=False,
    )
    _install_derivation(monkeypatch, context, [plan])
    config = _config(
        tmp_path, reviewer=("codex", "gemini", "antigravity"),
        pr_review_policy="primary-then-panel", primary_reviewer="codex", review_parallel=True,
    )
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123") as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert "partially published" not in str(stopped.value)
    conversion = _conversion(runner)
    assert conversion.metadata.round_number == 2
    records = _records(runner)
    before = [
        r for r in records
        if r.metadata.role == "reviewer" and r.metadata.round_number == 2 and r.index < conversion.index
    ]
    assert [(r.metadata.agent, r.metadata.scheduler_phase) for r in before] == [("Codex", "primary")]
    # The repeat posted its own checkpoint, recording the board that actually ran.
    [checkpoint] = [
        r for r in records
        if r.index > conversion.index and r.metadata.phase == "scheduler-prelaunch"
    ]
    assert set(checkpoint.metadata.scheduler_selected_reviewers) == {"Codex", "Gemini", "Antigravity"}
    assert checkpoint.metadata.scheduler_paused_reviewers == ()
    _assert_fresh_repeat(runner, conversion)
    repeat = [r for r in records if r.index > conversion.index]
    assert all(r.metadata.scheduler_metadata_status in {"valid", "absent"} for r in repeat)
    # The round number and budgets did not advance; no coder ran after the conversion.
    assert max(r.metadata.round_number for r in records) == 2
    assert _agent_count(runner, "claude") == 1
    assert runner.codex_outputs == runner.gemini_outputs == runner.antigravity_outputs == []
    assert records[-1].metadata.phase == "evidence-freeze"


# --- Durable PR human-decision diagnostics (#1640) ---------------------------


def test_an_evidence_conversion_that_owns_a_deferred_non_convergence_window_posts_nothing(
    tmp_path, monkeypatch
):
    """`nonconv-superseded-silent`: conversion wins at the dispatch where a real
    non-convergence window was deferred, so no non-convergence diagnostic is
    posted and no coder turn intervenes before the same-head review."""
    import coding_review_agent_loop.review_step_back as step_back_module

    deferrals = []
    real_defers = step_back_module.defers_to_evidence_stall

    def spy(result, rows):
        decision = real_defers(result, rows)
        deferrals.append(decision)
        return decision

    monkeypatch.setattr(step_back_module, "defers_to_evidence_stall", spy)
    context = _plan_context("row-a", "row-b")
    runner = _auto_converted_runner(tmp_path, monkeypatch, context)
    config = _config(tmp_path, pr_non_convergence_rounds=1)
    with pytest.raises(HumanDecisionRequiredError) as stopped:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    message = str(stopped.value)
    assert True in deferrals  # a qualifying non-convergence window was deferred
    assert "frozen at head abc123-coder-1" in message and "not converging" not in message
    assert _agent_count(runner, "claude") == 1  # no coder turn after the conversion
    records = _records(runner)
    phases = [(r.metadata.role, r.metadata.phase, r.metadata.round_number) for r in records]
    conversion_index = next(
        i for i, r in enumerate(records) if r.metadata.phase == "evidence-conversion"
    )
    after = phases[conversion_index:]
    assert after[0] == ("summary", "evidence-conversion", 2)
    assert any(role == "reviewer" and number == 2 for role, _phase, number in after[1:-1])
    assert not any(role == "coder" for role, _phase, _number in after)
    bodies = [comment["body"] for comment in runner.pr_payload["comments"]]
    assert not any("kept the same item(s) blocking" in body for body in bodies)
    assert not any(
        body.startswith("PR review scheduling diagnostic (") and "not converging" in body
        for body in bodies
    )


# Fresh configured workdirs must not be needed by read-only preflight.
def _fresh_conversion_entry(tmp_path, monkeypatch, plan_source):
    context = _plan_context("row-a", "row-b")
    runner = _history_before_conversion(tmp_path / "history", monkeypatch, context)
    if plan_source == "recovered":
        runner.pr_payload["body"] = "Fixes #56"
        runner.issue_payloads_by_number[56] = {"number": 56}
        runner.issue_comments_by_number[56] = _issue_bound_plan(context)
    config = _config(tmp_path / "fresh", create_dirs=False, pr_convert_evidence_rows=("row-a",))
    paths = (config.claude_dir, config.codex_dir, config.gemini_dir, config.antigravity_dir)
    assert not any(path.exists() for path in paths)
    assert not hasattr(runner, "_agent_loop_authenticated_actor")
    runner.commands.clear()
    events = []
    original_run = runner.run

    def run(args, **kwargs):
        if list(args)[:3] == ["gh", "api", "user"]:
            assert not any(path.exists() for path in paths)
            assert kwargs["cwd"].is_dir() and kwargs["cwd"] not in paths
            events.append("actor")
        cmd = list(args)
        if cmd[:3] in (["gh", "pr", "comment"], ["gh", "pr", "edit"],
                       ["gh", "pr", "ready"]):
            assert events == ["actor", "workdirs"]
        # Preserve FakeRunner's cwd existence check and actual REST hydration.
        return original_run(args, **kwargs)

    monkeypatch.setattr(runner, "run", run)
    original_ensure = pr_loop_module.ensure_agent_workdirs

    def ensure(*args, **kwargs):
        assert events == ["actor"]
        assert any(cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2]
                   for cmd, _cwd in runner.commands)
        events.append("workdirs")
        return original_ensure(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "ensure_agent_workdirs", ensure)
    return runner, config, context if plan_source == "supplied" else None, events


def _assert_fresh_entry_read_only(runner, before, config, events):
    assert runner.pr_payload == before
    assert events in (["actor"], [])
    assert not config.claude_dir.exists()
    assert not any(_agent_count(runner, name) for name in ("claude", "codex", "gemini", "agy"))
    for cmd, _cwd in runner.commands:
        assert cmd[:3] not in (["gh", "pr", "comment"], ["gh", "pr", "edit"],
                               ["gh", "pr", "ready"], ["gh", "pr", "merge"])
        assert not any(method in cmd for method in ("POST", "PATCH", "PUT", "DELETE"))
    assert not any(r.metadata.phase == "evidence-conversion" for r in _records(runner))


@pytest.mark.parametrize("plan_source", ["supplied", "recovered"])
def test_fresh_operator_conversion_authenticates_before_workdir_setup(tmp_path, monkeypatch, plan_source):
    runner, config, context, events = _fresh_conversion_entry(tmp_path, monkeypatch, plan_source)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head abc123-coder-1"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert events == ["actor", "workdirs"]
    assert any(cmd[:2] == ["gh", "api"] and "/comments?per_page=" in cmd[2]
               for cmd, _cwd in runner.commands)
    [conversion] = [r for r in _records(runner) if r.metadata.phase == "evidence-conversion"]
    assert conversion.metadata.evidence_conversion.source == "operator"
    assert conversion.metadata.evidence_conversion.row_ids == ("row-a",)


@pytest.mark.parametrize("plan_source", ["supplied", "recovered"])
def test_fresh_invalid_conversion_refuses_before_pending_board_write(tmp_path, monkeypatch, plan_source):
    import copy

    runner, config, context, events = _fresh_conversion_entry(tmp_path, monkeypatch, plan_source)
    config = dataclasses.replace(config, reviewer=("codex", "antigravity"),
                                 pr_convert_evidence_rows=("row-a", "row-z"))
    before = copy.deepcopy(runner.pr_payload)
    with pytest.raises(AgentLoopError, match="rejected: row-z"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert events == ["actor"]
    _assert_fresh_entry_read_only(runner, before, config, events)


@pytest.mark.parametrize("plan_source", ["supplied", "recovered"])
@pytest.mark.parametrize("failure", ["actor-failed", "actor-malformed", "rest-unavailable", "uncovered-record"])
def test_fresh_conversion_identity_failure_refuses_before_writes(tmp_path, monkeypatch, plan_source, failure):
    import copy
    from coding_review_agent_loop.runner import CommandResult

    runner, config, context, events = _fresh_conversion_entry(tmp_path, monkeypatch, plan_source)
    config = dataclasses.replace(config, reviewer=("codex", "antigravity"))
    if failure == "actor-malformed":
        runner.authenticated_actor = (ACTOR[0], True)
    elif failure == "actor-failed":
        original_run = runner.run

        def run(args, **kwargs):
            result = original_run(args, **kwargs)
            if list(args)[:3] == ["gh", "api", "user"]:
                return CommandResult(result.args, result.cwd, "", "identity unavailable", 1)
            return result

        monkeypatch.setattr(runner, "run", run)
    elif failure == "rest-unavailable":
        runner.serve_rest_issue_comments = False
    else:
        for comment in runner.pr_payload["comments"]:
            comment.pop("_rest_author_id", None)
    before = copy.deepcopy(runner.pr_payload)
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    expected = {
        "actor-failed": "Unable to resolve the authenticated GitHub actor",
        "actor-malformed": "lacked a login and immutable user ID",
        "rest-unavailable": "returned a non-list page",
        "uncovered-record": "REST comment read did not cover",
    }
    assert expected[failure] in str(raised.value)
    _assert_fresh_entry_read_only(runner, before, config, events)


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("context_mode", ["full", "compact"])
def test_broken_converted_freeze_captures_at_live_head(tmp_path, monkeypatch, policy, parallel, context_mode):
    """Empty external push: full review, capture without code finding, tool supersession."""
    context = _plan_context("row-a", "row-b")
    real_qualify = pr_loop_module._fresh_pr_qualification_snapshot

    def qualify(*args, **kwargs):
        plan = kwargs.pop("approved_plan_context", None)
        fresh, requirements, _plan, fresh_config = real_qualify(*args, **kwargs)
        return fresh, requirements, plan, fresh_config

    monkeypatch.setattr(pr_loop_module, "_fresh_pr_qualification_snapshot", qualify)
    runner, config = _frozen_after_auto_conversion(
        tmp_path, monkeypatch, context, config=_config(tmp_path, pr_review_policy=policy, review_parallel=parallel,
                                                       pr_review_context_mode=context_mode),
    )
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([
        _review(kept=["item-2"], state="blocking"),
        _review(),
        _review(),
    ])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    assert _agent_count(runner, "claude") == 2
    records = _records(runner)
    latest = next(r for r in reversed(records) if _evidence_items(r.metadata.prior_items))
    from coding_review_agent_loop.pr_loop_support import _reconcile_converted_evidence
    reconciled = _reconcile_converted_evidence(
        runner, config, pr_number=77, ledger=latest.metadata.prior_items,
        head_sha="fedcba9", approved_plan_context=context,
    )
    [row] = _evidence_items(reconciled, "row-a")
    assert row.lifecycle == "cleared" and row.candidate_head_sha == "fedcba9"
    assert "tool" in " ".join(row.notes).lower()
    assert "Bounded recovery capture" in _prompts(runner, "claude")[-1]
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    fresh = _reviews_after(runner, capture.index)
    assert len(fresh) == 1 and fresh[0].metadata.state == "approved"
    assert capture.metadata.subject == fresh[0].metadata.subject == "fedcba9"
    prompt = _prompts(runner, "codex")[-1]
    for evidence in ("row-a", "tests/test_row-a.py::test_it", "risk_test_matrix_evidence", "evidence_citations"):
        assert evidence in prompt
    # Reconstructing the same history does not invalidate the fresh approval.
    from coding_review_agent_loop.round_state import _latest_pr_approved_reviews_for_head
    approvals = _latest_pr_approved_reviews_for_head(
        authenticated_pr_comments(runner, config, pr_number=77)[0], head_sha="fedcba9",
        configured_reviewers=config.reviewer, approved_plan_context=context,
    )
    assert approvals["Codex"].index < capture.index
    assert fresh[0].metadata.review_scope is not None
    from coding_review_agent_loop.protocol import parse_review
    for record in records:
        if record.metadata.role == "reviewer" and record.metadata.state == "approved":
            assert parse_review(record.body, reviewer=record.metadata.agent).state == "approved"


@pytest.mark.parametrize("boundary", ["before-completion", "after-publication"])
@pytest.mark.parametrize("push", [False, True])
def test_capture_resume_uses_completed_publication_not_process_attempt(tmp_path, monkeypatch, boundary, push):
    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    _push_external_head(runner, "fedcba9")
    source_index = _records(runner)[-1].index
    runner.advance_pr_head_on_coder_followup = push
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([_review(kept=["item-2"], state="blocking")] * 4)
    _install_derivation(monkeypatch, context, [{"row-a": "missing", "row-b": "verified"}])
    original_run = pr_loop_module._run_validated_agent
    original_post = pr_loop_module.post_pr_comment

    def interrupt_run(*args, **kwargs):
        if kwargs.get("role") == "coder":
            raise KeyboardInterrupt()
        return original_run(*args, **kwargs)

    def interrupt_post(*args, **kwargs):
        result = original_post(*args, **kwargs)
        records = _records(runner)
        if records[-1].metadata.role == "coder" and records[-1].index > source_index:
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(pr_loop_module, "_run_validated_agent", interrupt_run if boundary == "before-completion" else original_run)
    monkeypatch.setattr(pr_loop_module, "post_pr_comment", interrupt_post if boundary == "after-publication" else original_post)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    monkeypatch.setattr(pr_loop_module, "_run_validated_agent", original_run)
    monkeypatch.setattr(pr_loop_module, "post_pr_comment", original_post)
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    completed = [r for r in _records(runner) if r.index > source_index and r.metadata.role == "coder"]
    assert len(completed) == 1
    assert completed[0].metadata.followup_dispatch_head == "fedcba9"
    assert completed[0].metadata.subject == ("fedcba9-coder-2" if push else "fedcba9")
    assert _agent_count(runner, "claude") == 2


@pytest.mark.parametrize("legacy", [False, True])
def test_capture_history_with_ordinary_code_turn_consumes_opportunity(tmp_path, monkeypatch, legacy):
    from coding_review_agent_loop.converted_capture import capture_candidates
    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(tmp_path, monkeypatch, context)
    _push_external_head(runner, "fedcba9")
    runner.codex_outputs.extend([
        _review(["Repair the real code defect."], kept=["item-2"], state="blocking"),
        _review(kept=["item-2"], resolved=["item-3"], state="blocking"),
    ])
    runner.claude_outputs.append(_coder(3))
    _install_derivation(monkeypatch, context, [{"row-a": "missing", "row-b": "verified"}])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    records = list(_records(runner))
    freeze = records[-1]
    # Remove the later terminal freeze to simulate an interruption just before it.
    records = records[:-1]
    coder = next(r for r in reversed(records) if r.metadata.role == "coder")
    stripped = dataclasses.replace(coder, metadata=dataclasses.replace(
        coder.metadata, risk_test_matrix_evidence=None, risk_test_matrix_capture_classification=None,
    ))
    records[records.index(coder)] = stripped
    coder = stripped
    if legacy:
        records[records.index(coder)] = dataclasses.replace(coder, metadata=dataclasses.replace(
            coder.metadata, followup_dispatch_head=None,
        ))
    items = [dataclasses.replace(i, lifecycle="evidence_deferred") if i.converted_row_id else i
             for i in freeze.metadata.prior_items]
    assert capture_candidates(records, items=items, head=freeze.metadata.subject,
                              matrix_identity=context.risk_test_matrix_identity) == ()
    assert _agent_count(runner, "claude") == 2


def test_subset_recovery_capture_freezes_only_unverified_rows(tmp_path, monkeypatch):
    context = _plan_context("row-a", "row-b")
    runner, config = _two_rows_frozen(tmp_path, monkeypatch, context)
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(structured_coder_followup(
        human_requirement_ids=_signed_requirement_ids(runner, config),
    ))
    runner.codex_outputs.extend([
        _review(kept=["item-2", "item-3"], state="blocking", hr=True),
        _review(hr=True),
    ])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "missing"}])
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    frozen = _records(runner)[-1].metadata
    [a] = _evidence_items(frozen.prior_items, "row-a")
    [b] = _evidence_items(frozen.prior_items, "row-b")
    assert a.lifecycle == "cleared" and b.lifecycle == "evidence_frozen"
    assert b.item_id == "item-3" and b.resolution_owners == ("Codex",)
    capture = next(r for r in reversed(_records(runner)) if r.metadata.role == "coder")
    assert _reviews_after(runner, capture.index)
    assert "tests/test_row-a.py::test_it" in _prompts(runner, "codex")[-1]
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _agent_count(runner, "claude") == 2


def _bounded_broken_freeze(tmp_path, monkeypatch, *, policy="all-reviewers", allowed=2):
    # This synthetic plan has no issue handoff; retain it across qualification.
    real_qualify = pr_loop_module._fresh_pr_qualification_snapshot

    def qualify(*args, **kwargs):
        plan = kwargs.pop("approved_plan_context", None)
        fresh, requirements, _plan, fresh_config = real_qualify(*args, **kwargs)
        return fresh, requirements, plan, fresh_config

    monkeypatch.setattr(pr_loop_module, "_fresh_pr_qualification_snapshot", qualify)
    context = _plan_context("row-a", "row-b")
    runner, config = _frozen_after_auto_conversion(
        tmp_path, monkeypatch, context, config=_config(tmp_path, pr_review_policy=policy),
    )
    config = dataclasses.replace(config, max_rounds=2)
    freeze = _records(runner)[-1]
    bounded = dataclasses.replace(freeze.metadata, evidence_freeze=dataclasses.replace(
        freeze.metadata.evidence_freeze, allowed_rounds=allowed,
        watch_failure_extension_used=allowed > 2,
    ))
    runner.pr_payload["comments"][freeze.index]["body"] = _attach_round_metadata("Bounded freeze", bounded)
    _push_external_head(runner, "fedcba9")
    return runner, config, context, bounded


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_broken_converted_freeze_respects_exhausted_round_budget(tmp_path, monkeypatch, policy):
    runner, config, context, source = _bounded_broken_freeze(tmp_path, monkeypatch, policy=policy)
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    before = len(_records(runner))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    frozen = _records(runner)[-1].metadata
    assert frozen.phase == "evidence-freeze" and frozen.subject == "fedcba9"
    assert frozen.round_number == frozen.evidence_freeze.allowed_rounds == 2
    assert frozen.evidence_freeze.watch_failure_extension_used is False
    assert frozen.evidence_freeze.watch_head_extension_used == source.evidence_freeze.watch_head_extension_used
    [row] = _evidence_items(frozen.prior_items)
    [original] = _evidence_items(source.prior_items)
    assert (row.item_id, row.converted_row_id, row.resolution_owners) == (
        original.item_id, original.converted_row_id, original.resolution_owners,
    )
    assert row.lifecycle == "evidence_frozen"
    assert row.obligation_identity == original.obligation_identity
    assert frozen.evidence_freeze.evidence_identities == source.evidence_freeze.evidence_identities
    assert frozen.signed_evidence_references == source.signed_evidence_references
    assert not any(r.metadata.role == "coder" for r in _records(runner)[before:])
    snapshot = list(runner.comments)
    agents = _agent_count(runner, "codex")
    runner.simulate_new_process()
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert runner.comments[:-1] == snapshot
    assert runner.comments[-1].startswith("PR review scheduling diagnostic (round 2): PR #77 is frozen")
    assert _records(runner)[-1].metadata == frozen
    assert _agent_count(runner, "codex") == agents
    assert _agent_count(runner, "claude") == 1
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


@pytest.mark.parametrize("policy", ["all-reviewers", "selective-intermediate"])
def test_broken_converted_freeze_uses_restored_watcher_slot(tmp_path, monkeypatch, policy):
    runner, config, context, _ = _bounded_broken_freeze(tmp_path, monkeypatch, policy=policy, allowed=3)
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.extend([
        _review(kept=["item-2"], state="blocking"), _review(),
    ])
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = _records(runner)
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    # Coder publications name the next review round; admission occurred at R=2.
    assert capture.metadata.round_number == 3
    preceding = [r for r in records if r.index < capture.index
                 and r.metadata.role == "reviewer" and r.metadata.subject == "fedcba9"]
    assert preceding and preceding[-1].metadata.round_number == config.max_rounds
    [semantic] = _reviews_after(runner, capture.index)
    assert semantic.metadata.round_number == 3 and semantic.metadata.state == "approved"
    assert all(r.metadata.round_number <= 3 for r in records)
    assert not any(r.metadata.phase == "evidence-freeze" for r in records if r.index > capture.index)
    assert "Bounded recovery capture" in _prompts(runner, "claude")[-1]
    assert _agent_count(runner, "claude") == 2
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_broken_converted_freeze_with_code_blocker_keeps_round_limit(tmp_path, monkeypatch):
    runner, config, context, _ = _bounded_broken_freeze(tmp_path, monkeypatch)
    runner.codex_outputs.append(_review(["Repair the real code defect."], kept=["item-2"], state="blocking"))
    before = len(_records(runner))
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert "round" in str(raised.value).lower()
    assert not any(r.metadata.phase == "evidence-freeze" or r.metadata.role == "coder"
                   for r in _records(runner)[before:])
    assert _agent_count(runner, "claude") == 1
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def test_skipped_capture_revalidates_live_head_at_final_gate(tmp_path, monkeypatch):
    runner, config, context, _ = _bounded_broken_freeze(tmp_path, monkeypatch)
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    real_gate = pr_loop_module._evidence_freeze_gate
    outcomes = []

    def gate(*args, **kwargs):
        _push_external_head(runner, "abcdef0")
        result = real_gate(*args, **kwargs)
        outcomes.append(result)
        return result

    monkeypatch.setattr(pr_loop_module, "_evidence_freeze_gate", gate)
    before = len(_records(runner))
    with pytest.raises(AgentLoopError) as raised:
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert not isinstance(raised.value, HumanDecisionRequiredError)
    assert [outcome.action for outcome in outcomes] == ["head_changed"]
    assert not any(r.metadata.phase == "evidence-freeze" or r.metadata.role == "coder"
                   for r in _records(runner)[before:])
    assert _agent_count(runner, "claude") == 1
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _cwd in runner.commands)


def _assert_capture_context(text, evidence):
    assert "Converted-row capture evidence:" in text
    assert "awaiting semantic review" not in text
    assert "receipt verification does not establish semantic coverage" in text
    assert "When review is required, judge whether the cited tests exercise the affected rows" in text
    assert "Approvals predating this publication do not establish acceptance" in text
    assert "Report coverage gaps as ordinary evidence-tagged findings" in text
    assert "do not run commands or clear machine obligations" in text
    assert json.dumps(evidence, sort_keys=True) in text
    assert evidence["rows"][0]["evidence_citations"]


def _prepare_verified_recovery(tmp_path, monkeypatch, *, policy="all-reviewers"):
    context = _plan_context("row-a", "row-b")
    real_qualify = pr_loop_module._fresh_pr_qualification_snapshot

    def qualify(*args, **kwargs):
        plan = kwargs.pop("approved_plan_context", None)
        fresh, requirements, _plan, fresh_config = real_qualify(*args, **kwargs)
        return fresh, requirements, plan, fresh_config

    monkeypatch.setattr(pr_loop_module, "_fresh_pr_qualification_snapshot", qualify)
    runner, config = _frozen_after_auto_conversion(
        tmp_path, monkeypatch, context, config=_config(tmp_path, pr_review_policy=policy),
    )
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    return runner, config, context


@pytest.mark.parametrize("policy,seam", [
    ("all-reviewers", "coder"), ("all-reviewers", "reviewer"),
    ("selective-intermediate", "coder"), ("selective-intermediate", "scheduler-prelaunch"),
    ("selective-intermediate", "reviewer"), ("selective-intermediate", "reconciliation"),
])
def test_verified_capture_restart_requires_then_reuses_semantic_review(tmp_path, monkeypatch, seam, policy):
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch, policy=policy)
    # Exercise complete bound approvals as well as the default scoped acceptance.
    config = dataclasses.replace(config, pr_review_force_full=policy == "all-reviewers")
    runner.codex_outputs.append(_review())
    original_post = pr_loop_module.post_pr_comment
    source = _records(runner)[-1].index

    def post(*args, **kwargs):
        result = original_post(*args, **kwargs)
        records = _records(runner)
        captures = [r for r in records if r.index > source and r.metadata.role == "coder"]
        if captures:
            latest = records[-1]
            if (seam == "coder" and latest == captures[-1]
                or latest.index > captures[-1].index
                and (latest.metadata.role == seam or latest.metadata.phase == seam)):
                raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(pr_loop_module, "post_pr_comment", post)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    captures = [r for r in _records(runner) if r.index > source and r.metadata.role == "coder"]
    assert len(captures) == 1
    capture = captures[0]
    _assert_capture_context(
        capture_review_context(_records(runner), head="fedcba9", plan=context),
        capture.metadata.risk_test_matrix_evidence,
    )
    # Code approvals are independent of capture freshness; the semantic gate
    # below still refuses a capture without published acceptance.
    from coding_review_agent_loop.round_state import _latest_pr_approved_reviews_for_head
    before = _latest_pr_approved_reviews_for_head(
        authenticated_pr_comments(runner, config, pr_number=77)[0], head_sha="fedcba9",
        configured_reviewers=config.reviewer, approved_plan_context=context,
    )
    assert before and all(record.metadata.subject == "fedcba9" for record in before.values())
    if seam == "coder":
        fresh, _, _, _ = pr_loop_module._fresh_pr_qualification_snapshot(
            runner, config=config, pr_number=77, issue_context=None, parent_issue_context=None,
            approved_plan_context=context,
        )
        assert fresh.converted_capture_review_pending
    monkeypatch.setattr(pr_loop_module, "post_pr_comment", original_post)
    runner.simulate_new_process()
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    [semantic] = _reviews_after(runner, capture.index)
    assert semantic.metadata.state == "approved"
    if policy == "selective-intermediate":
        assert semantic.metadata.review_scope is not None
    else:
        assert semantic.metadata.capture_review_binding is not None
    assert _agent_count(runner, "claude") == 2
    _assert_capture_context(_prompts(runner, "codex")[-1], capture.metadata.risk_test_matrix_evidence)
    _assert_capture_context(
        capture_review_context(admitted_pr_records(runner, config, pr_number=77),
                               head="fedcba9", plan=context),
        capture.metadata.risk_test_matrix_evidence,
    )
    fresh, _, _, _ = pr_loop_module._fresh_pr_qualification_snapshot(
        runner, config=config, pr_number=77, issue_context=None, parent_issue_context=None,
        approved_plan_context=context,
    )
    assert not fresh.converted_capture_review_pending


def test_verified_capture_can_be_rejected_for_semantic_coverage(tmp_path, monkeypatch):
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    config = dataclasses.replace(config, auto_merge=True)
    runner.codex_outputs.append(_review([_tagged("row-a", text="The cited test never exercises the required row-a transition.")]))
    real_run = pr_loop_module._run_validated_agent
    coders = []

    def run(*args, **kwargs):
        if kwargs.get("role") == "coder":
            coders.append(kwargs)
            if len(coders) == 2:
                raise KeyboardInterrupt()
        return real_run(*args, **kwargs)

    monkeypatch.setattr(pr_loop_module, "_run_validated_agent", run)
    with pytest.raises(KeyboardInterrupt):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    capture = next(r for r in reversed(_records(runner)) if r.metadata.role == "coder")
    [review] = _reviews_after(runner, capture.index)
    assert review.metadata.state == "blocking"
    assert any("never exercises" in item.text for item in review.metadata.new_items)
    assert "tests/test_row-a.py::test_it" in _prompts(runner, "codex")[-1]
    assert len(coders) == 2  # ordinary finding handling, despite valid receipts
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)


@pytest.mark.parametrize("parallel", [False, True])
@pytest.mark.parametrize("operator", [False, True])
def test_post_panel_capture_reviews_policy_selected_board(tmp_path, monkeypatch, parallel, operator):
    context = _plan_context("row-a", "row-b")
    runner, config = _carried_panel_runner(tmp_path, monkeypatch, context)
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    config = dataclasses.replace(config, review_parallel=parallel, pr_review_force_full=operator)
    _push_external_head(runner, "fedcba9")
    runner.advance_pr_head_on_coder_followup = False
    runner.claude_outputs.append(_coder())
    for name, outputs in [("OpenAI Codex", runner.codex_outputs),
                          ("Google Gemini", runner.gemini_outputs),
                          ("Antigravity", runner.antigravity_outputs)]:
        outputs.extend([_repeat_review(name), _review(resolved=["item-2"] if operator else [], reviewer=name)])
    _install_derivation(monkeypatch, context, [{"row-a": "verified", "row-b": "verified"}])
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    capture = next(r for r in reversed(_records(runner)) if r.metadata.role == "coder")
    fresh = _reviews_after(runner, capture.index)
    assert {r.metadata.agent for r in fresh} == ({"Codex", "Gemini", "Antigravity"} if operator else {"Codex"})
    for command in (("codex", "gemini", "agy") if operator else ("codex",)):
        assert "tests/test_row-a.py::test_it" in _prompts(runner, command)[-1]


def test_capture_boundary_replacement_and_identical_evidence(tmp_path, monkeypatch):
    """Real recovery history underlies every pure eligibility mutation below."""
    from coding_review_agent_loop.converted_capture import evidence_review_boundary
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    runner.codex_outputs.append(_review(resolved=["item-2"]))
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    records = list(admitted_pr_records(runner, config, pr_number=77))
    capture = next(r for r in reversed(records) if r.metadata.role == "coder")
    boundary = lambda history: evidence_review_boundary(history, head="fedcba9",
                                                        matrix_identity=context.risk_test_matrix_identity)
    render = lambda history: capture_review_context(history, head="fedcba9", plan=context)
    assert render([]) == ""
    assert render([r for r in records if r.index < capture.index]) == ""
    assert capture_review_context(records, head="another-head", plan=context) == ""
    assert capture_review_context(records, head="fedcba9", plan=_plan_context("other-row")) == ""
    assert boundary(records) == capture.index
    _assert_capture_context(render(records), capture.metadata.risk_test_matrix_evidence)
    repeated = dataclasses.replace(capture, index=records[-1].index + 1)
    assert boundary([*records, repeated]) == capture.index
    assert render([*records, repeated]) == render(records)
    changed = json.loads(json.dumps(capture.metadata.risk_test_matrix_evidence))
    changed["rows"][0]["outcome_assertions"] = ["A replacement semantic coverage claim."]
    replacement = dataclasses.replace(repeated, metadata=dataclasses.replace(
        repeated.metadata, risk_test_matrix_evidence=changed,
    ))
    assert boundary([*records, replacement]) == replacement.index
    _assert_capture_context(render([*records, replacement]), changed)
    assert json.dumps(capture.metadata.risk_test_matrix_evidence, sort_keys=True) not in render([*records, replacement])
    # A claim, mismatched matrix, or missing dispatch cannot establish a boundary.
    for invalid in [dataclasses.replace(capture.metadata, followup_dispatch_head=None),
                    dataclasses.replace(capture.metadata, risk_test_matrix_evidence=None),
                    dataclasses.replace(capture.metadata, risk_test_matrix_evidence={**changed, "matrix_identity": "0" * 64})]:
        history = [dataclasses.replace(r, metadata=invalid) if r.index == capture.index else r for r in records]
        assert boundary(history) == -1
        assert render(history) == ""


def test_verified_capture_that_moves_head_gets_exact_head_review(tmp_path, monkeypatch):
    runner, config, context = _prepare_verified_recovery(tmp_path, monkeypatch)
    runner.advance_pr_head_on_coder_followup = True
    runner.codex_outputs.append(_review(resolved=["item-2"]))
    assert run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context) == 0
    capture = next(r for r in reversed(_records(runner)) if r.metadata.role == "coder")
    assert capture.metadata.followup_dispatch_head == "fedcba9"
    assert capture.metadata.subject != "fedcba9"
    [fresh] = _reviews_after(runner, capture.index)
    assert fresh.metadata.subject == capture.metadata.subject
    assert fresh.metadata.state == "approved"
