"""PR-loop evidence conversion to the exact-head human evidence freeze (#1325)."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.pr_loop as pr_loop_module
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


def _frozen_after_auto_conversion(tmp_path, monkeypatch, context, **runner_kwargs):
    runner = _auto_converted_runner(tmp_path, monkeypatch, context, **runner_kwargs)
    config = _config(tmp_path)
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
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking"))
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    freeze = _records(runner)[-1].metadata
    [reopened] = _evidence_items(freeze.prior_items, "row-a")
    assert (reopened.item_id, reopened.obligation_identity) == (
        frozen_at_h.item_id, frozen_at_h.obligation_identity,
    )
    assert reopened.candidate_head_sha == "fedcba9"
    assert _agent_count(runner, "claude") == 1


def test_a_cleared_row_reopens_at_a_new_head_without_any_coder_turn(tmp_path, monkeypatch):
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
    runner.codex_outputs.append(_review(kept=["item-2"], state="blocking", hr=True))
    coders_before = _agent_count(runner, "claude")
    with pytest.raises(HumanDecisionRequiredError, match="frozen at head fedcba9"):
        run_pr_loop(runner, pr_number=77, config=config, approved_plan_context=context)
    assert _agent_count(runner, "claude") == coders_before  # no coder dispatch
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
