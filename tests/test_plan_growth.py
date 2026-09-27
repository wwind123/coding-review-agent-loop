"""Plan-growth signals, the one-shot growth gate and its levers (#886)."""

import hashlib
import json
from types import SimpleNamespace

import pytest

import coding_review_agent_loop.orchestrator as orchestrator_module
from agent_loop_helpers import (
    FakeRunner,
    make_config,
    structured_plan_review,
    structured_v1_plan_state,
)
from coding_review_agent_loop.cli import build_parser, run_issue_loop
from coding_review_agent_loop.comment_rendering import (
    COMPACT_PLAN_DIGEST_BUDGET_CHARS,
    render_canonical_plan_state,
    render_public_agent_comment,
)
from coding_review_agent_loop.config import _arg_or_default
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.managed_ci import _RECOVERY_VALUE_OPTIONS
from coding_review_agent_loop.plan_assembly import (
    AuthenticatedPlanState,
    aggregate_plan_identity,
    assemble_authenticated_plan_revision,
    hydrate_authenticated_plan_state,
    make_assembled_plan_sidecar,
    structured_plan_revision_to_payload,
)
from coding_review_agent_loop.plan_growth import (
    PlanGrowthThresholds,
    assess_plan_growth,
    check_growth_justification,
    check_scope_ledger_preservation,
    growth_justification_violation,
)
from coding_review_agent_loop.prompts import (
    build_issue_plan_prompt,
    build_plan_review_prompt,
    build_plan_revision_prompt,
)
from coding_review_agent_loop.protocol import (
    PLAN_REVISION_PATCH_REPLACEABLE_FIELDS,
    parse_plan_revision_patch,
    validate_structured_plan_state,
)
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _extract_round_metadata_records,
)

FOOTER = "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
RATIONALE = "The scope items share one parser seam and cannot ship separately."


def _payload(**overrides):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload.update(overrides)
    return payload


def _justification(*signals, rationale=RATIONALE):
    return {"crossed_signals": list(signals), "rationale": rationale}


def _state_text(**overrides):
    return json.dumps(_payload(**overrides)) + FOOTER


def _parsed(**overrides):
    return validate_structured_plan_state(_state_text(**overrides))


def _prompts(runner, agent):
    head = [agent] if agent == "claude" else [agent, "exec"]
    return [cmd[-1] for cmd, _cwd in runner.commands if cmd[: len(head)] == head]


def _plan_records(runner):
    comments = [SimpleNamespace(body=str(item["body"])) for item in runner.issue_comments]
    return [record.metadata for record in _extract_round_metadata_records(comments, flow="plan")]


def _config(tmp_path, **overrides):
    values = {"max_rounds": 4}
    values.update(overrides)
    return make_config(tmp_path, **values)


def _approved_history(tmp_path, **config_overrides):
    """Comments of a v1 one-shot plan approved under default thresholds."""
    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_config(tmp_path, **config_overrides), plan_first=True
    ) == 0
    return list(runner.issue_comments)


def _base_identity():
    return AuthenticatedPlanState.from_plan(
        validate_structured_plan_state(structured_v1_plan_state()), round_number=1
    ).state_identity


def _justify_patch(*signals, base_round=1, identity=None):
    return json.dumps({
        "schema_version": 1, "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1, "state": "blocking",
        "summary": "Justify one-shot.", "prior_plan_item_dispositions": [],
        "base_round_number": base_round,
        "base_state_identity": identity or _base_identity(),
        "operations": [{
            "op": "replace", "field": "one_shot_growth_justification",
            "value": _justification(*signals),
        }],
    }) + FOOTER


# --- signals -------------------------------------------------------------


def test_below_threshold_plan_crosses_nothing_and_needs_no_justification():
    """Row `below-threshold`."""
    plan = _parsed()
    canonical = render_canonical_plan_state(plan)
    assessment = assess_plan_growth(
        plan, rendered_chars=len(canonical), revision_count=1, thresholds=PlanGrowthThresholds()
    )
    assert assessment.crossed == ()
    assert growth_justification_violation(plan, assessment) is None
    # No growth field is added to the canonical payload of an ordinary plan.
    assert "one_shot_growth_justification" not in structured_plan_revision_to_payload(plan)


def test_each_structural_signal_crosses_at_its_threshold():
    plan = _parsed()
    thresholds = PlanGrowthThresholds(max_chars=100, max_scope_items=1, max_matrix_rows=1)
    assessment = assess_plan_growth(
        plan, rendered_chars=100, revision_count=None, thresholds=thresholds
    )
    # The fixture matrix is not-applicable with no rows.
    assert assessment.crossed == ("rendered-size", "scope-items")
    assert assessment.matrix_rows == 0
    below = assess_plan_growth(plan, rendered_chars=99, revision_count=None, thresholds=thresholds)
    assert below.crossed == ("scope-items",)


def test_revision_count_only_combines_with_half_the_size_threshold():
    """Row `revision-count-only`: revision count alone never crosses."""
    plan = _parsed()
    thresholds = PlanGrowthThresholds(max_chars=1000, max_revisions=3)
    small = assess_plan_growth(plan, rendered_chars=499, revision_count=50, thresholds=thresholds)
    assert small.crossed == ()
    grown = assess_plan_growth(plan, rendered_chars=500, revision_count=3, thresholds=thresholds)
    assert grown.crossed == ("revision-count",)
    early = assess_plan_growth(plan, rendered_chars=500, revision_count=2, thresholds=thresholds)
    assert early.crossed == ()
    # Rebind passes no revision count at all.
    rebind = assess_plan_growth(plan, rendered_chars=900, revision_count=None, thresholds=thresholds)
    assert rebind.crossed == ()


def test_planner_candidate_rounds_ignore_reviewer_only_rounds_and_replays():
    """Row `revision-count-only`: never derived from round_number."""
    plan = _parsed()
    canonical = render_canonical_plan_state(plan)
    def coder(round_number):
        sidecar = make_assembled_plan_sidecar(
            plan, round_number=round_number, response_form="fresh-plan-state",
            rendered_plan=canonical,
        )
        return SimpleNamespace(body=_attach_round_metadata(
            canonical + FOOTER,
            PostedRoundMetadata(
                flow="plan", role="coder", agent="Claude", round_number=round_number,
                subject="s", canonical_plan=canonical, response_form="fresh-plan-state",
                aggregate_plan_identity=sidecar.aggregate_identity,
                assembled_plan_sidecar=sidecar.to_payload(),
            ),
        ))

    def phase_advance(round_number):
        return SimpleNamespace(body=_attach_round_metadata(
            "Plan review phase advance.\n-- Orchestrator",
            PostedRoundMetadata(
                flow="plan", role="summary", agent="Orchestrator",
                round_number=round_number, subject="s", phase="plan-phase-advance",
            ),
        ))

    comments = [coder(1), phase_advance(2), phase_advance(3), coder(1), coder(4)]
    rounds = orchestrator_module._planner_candidate_rounds(comments)
    assert rounds == {1, 4}
    assert orchestrator_module._plan_growth_candidate_count(rounds, 4) == 2
    assert orchestrator_module._plan_growth_candidate_count(rounds, 5) == 3


# --- justification rules ---------------------------------------------------


def test_justification_parses_strictly():
    plan = _parsed(one_shot_growth_justification=_justification("scope-items"))
    assert plan.one_shot_growth_justification.crossed_signals == ("scope-items",)
    for bad in (
        {"crossed_signals": [], "rationale": "x"},
        {"crossed_signals": ["scope-items", "scope-items"], "rationale": "x"},
        {"crossed_signals": ["finding-count"], "rationale": "x"},
        {"crossed_signals": ["scope-items"], "rationale": ""},
        {"crossed_signals": ["scope-items"], "rationale": "x" * 2001},
        {"crossed_signals": ["scope-items"], "rationale": "x", "extra": 1},
    ):
        with pytest.raises(AgentLoopError):
            _parsed(one_shot_growth_justification=bad)
    assert _parsed(
        one_shot_growth_justification=_justification("scope-items", rationale="x" * 2000)
    ) is not None


def test_self_check_requires_exact_coverage_of_the_candidates_own_signals():
    """Rows `over-threshold-justified`, `new-signal-validation`, `shrink-below-threshold`."""
    thresholds = PlanGrowthThresholds(max_chars=100, max_scope_items=1)

    def check(plan, chars):
        check_growth_justification(
            plan,
            assess_plan_growth(plan, rendered_chars=chars, revision_count=1, thresholds=thresholds),
        )

    justified = _parsed(one_shot_growth_justification=_justification("scope-items"))
    check(justified, 50)
    # The revision newly crosses rendered-size but keeps the old justification.
    with pytest.raises(AgentLoopError, match="missing `rendered-size`"):
        check(justified, 150)
    # Unjustified over-threshold one-shot plan.
    with pytest.raises(AgentLoopError, match="crosses plan-growth threshold"):
        check(_parsed(), 50)
    # A justification naming a signal no longer crossed.
    over_named = _parsed(one_shot_growth_justification=_justification("scope-items", "rendered-size"))
    with pytest.raises(AgentLoopError, match="no longer crossed `rendered-size`"):
        check(over_named, 50)
    # Shrunk below every threshold: no justification needed, leftovers rejected.
    loose = PlanGrowthThresholds()
    plain = _parsed()
    check_growth_justification(
        plain, assess_plan_growth(plain, rendered_chars=10, revision_count=1, thresholds=loose)
    )
    with pytest.raises(AgentLoopError, match="stale"):
        check_growth_justification(
            justified,
            assess_plan_growth(justified, rendered_chars=10, revision_count=1, thresholds=loose),
        )


def _staged_recommendation_payload(scope_items):
    return {
        "strategy": "staged",
        "scope_items": scope_items,
    }


def _one_shot_payload(scope_items):
    return {
        "execution_strategy_contract_version": 1,
        "execution_recommendation": {"strategy": "one-shot", "scope_items": scope_items},
    }


def _scope(item_id, requirement="Deliver it.", criteria=("It is delivered.",)):
    return {"scope_item_id": item_id, "requirement": requirement, "acceptance_criteria": list(criteria)}


def test_staged_conversion_preserves_the_scope_ledger():
    """Row `staged-conversion`."""
    prior = _one_shot_payload([_scope("a"), _scope("b", criteria=("B one.", "B two."))])

    def staged(scope_items, **extra):
        return {
            "execution_strategy_contract_version": 1,
            "execution_recommendation": _staged_recommendation_payload(scope_items),
            **extra,
        }

    preserving = staged([
        _scope("a", requirement="Deliver   it."),
        _scope("b", criteria=("B one.", "B two.", "B three.")),
        _scope("c"),
    ])
    check_scope_ledger_preservation(prior, preserving)
    with pytest.raises(AgentLoopError, match="`b` was dropped or renamed"):
        check_scope_ledger_preservation(prior, staged([_scope("a"), _scope("b2")]))
    with pytest.raises(AgentLoopError, match="`a` changed its requirement text"):
        check_scope_ledger_preservation(
            prior, staged([_scope("a", requirement="Deliver less."), prior["execution_recommendation"]["scope_items"][1]])
        )
    with pytest.raises(AgentLoopError, match="`b` removed or reworded 1 acceptance"):
        check_scope_ledger_preservation(
            prior, staged([_scope("a"), _scope("b", criteria=("B one.", "B 2."))])
        )
    # A staged plan still carrying a one-shot justification fails the gate.
    carrying = staged([_scope("a")], one_shot_growth_justification=_justification("scope-items"))
    assessment = assess_plan_growth(
        carrying, rendered_chars=1, revision_count=1, thresholds=PlanGrowthThresholds()
    )
    assert "must not carry" in growth_justification_violation(carrying, assessment)


def test_legacy_plans_are_never_gated_or_ledger_checked():
    """Row `legacy-or-off`."""
    legacy = {"schema_version": 1, "kind": "plan_state", "plan_steps": ["x"]}
    assessment = assess_plan_growth(
        legacy, rendered_chars=10**6, revision_count=99, thresholds=PlanGrowthThresholds()
    )
    assert "rendered-size" in assessment.crossed
    assert growth_justification_violation(legacy, assessment) is None
    staged = {
        "execution_strategy_contract_version": 1,
        "execution_recommendation": _staged_recommendation_payload([]),
    }
    check_scope_ledger_preservation(legacy, staged)


# --- config ---------------------------------------------------------------


def test_thresholds_are_configurable_and_validated(tmp_path):
    """Row `config-validation`."""
    parser = build_parser()
    args = parser.parse_args([
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--plan-growth-gate", "off", "--plan-growth-max-chars", "5000",
        "--plan-growth-max-revisions", "2", "--plan-growth-max-scope-items", "3",
        "--plan-growth-max-matrix-rows", "4",
    ])
    assert (
        args.plan_growth_gate, args.plan_growth_max_chars, args.plan_growth_max_revisions,
        args.plan_growth_max_scope_items, args.plan_growth_max_matrix_rows,
    ) == ("off", 5000, 2, 3, 4)
    config = _config(
        tmp_path, plan_growth_gate="off", plan_growth_max_chars=5000,
        plan_growth_max_revisions=2, plan_growth_max_scope_items=3,
        plan_growth_max_matrix_rows=4,
    )
    assert PlanGrowthThresholds.from_config(config) == PlanGrowthThresholds(5000, 2, 3, 4)
    defaults = _config(tmp_path)
    assert (defaults.plan_growth_gate, defaults.plan_growth_max_chars) == ("enforce", 120_000)
    for name in (
        "plan_growth_max_chars", "plan_growth_max_revisions",
        "plan_growth_max_scope_items", "plan_growth_max_matrix_rows",
    ):
        for value in (0, -1):
            with pytest.raises(AgentLoopError, match="must be a positive integer"):
                _config(tmp_path, **{name: value})
    with pytest.raises(AgentLoopError, match="--plan-growth-gate"):
        _config(tmp_path, plan_growth_gate="maybe")
    for flag in (
        "--plan-growth-gate", "--plan-growth-max-chars", "--plan-growth-max-revisions",
        "--plan-growth-max-scope-items", "--plan-growth-max-matrix-rows",
    ):
        assert flag in _RECOVERY_VALUE_OPTIONS


def test_cli_threshold_flags_reach_the_config(tmp_path):
    args = build_parser().parse_args([
        "issue", "56", "--repo", "OWNER/REPO", "--plan-first",
        "--plan-growth-max-scope-items", "0",
    ])
    # An explicit invalid value is kept (not replaced by the default) so
    # validation rejects it instead of silently falling back.
    assert _arg_or_default(args, "plan_growth_max_scope_items", 12) == 0
    assert _arg_or_default(args, "plan_growth_max_chars", 120_000) == 120_000
    with pytest.raises(AgentLoopError, match="--plan-growth-max-scope-items must be a positive"):
        _config(tmp_path, plan_growth_max_scope_items=_arg_or_default(args, "plan_growth_max_scope_items", 12))


# --- semantic patch lifecycle ----------------------------------------------


def test_justification_patch_add_preserve_clear_and_resume():
    """Row `justification-patch-lifecycle`."""
    assert "one_shot_growth_justification" in PLAN_REVISION_PATCH_REPLACEABLE_FIELDS
    plan = validate_structured_plan_state(_state_text())
    base = AuthenticatedPlanState.from_plan(plan, round_number=1)
    # Absence adds no key, so identities of plans without the field are unchanged.
    serialized = structured_plan_revision_to_payload(plan)
    assert "one_shot_growth_justification" not in serialized
    assert base.state_identity == aggregate_plan_identity(serialized)

    def patch(state, operations):
        return {
            "schema_version": 1, "kind": "plan_revision_patch",
            "semantic_patch_contract_version": 1, "state": "blocking", "summary": "p",
            "prior_plan_item_dispositions": [], "base_round_number": state.round_number,
            "base_state_identity": state.state_identity, "operations": operations,
        }

    def resumed(sidecar):
        # A restart re-hydrates the result from its authenticated sidecar,
        # and the stored raw patch re-parses identically.
        decoded = hydrate_authenticated_plan_state(sidecar.encode())
        assert parse_plan_revision_patch(sidecar.raw_patch).to_payload() == sidecar.raw_patch
        return decoded

    added, sidecar = assemble_authenticated_plan_revision(base, patch(base, [
        {"op": "replace", "field": "one_shot_growth_justification", "value": _justification("scope-items")},
    ]), result_round_number=2)
    assert added.one_shot_growth_justification.crossed_signals == ("scope-items",)
    state2 = resumed(sidecar)
    assert state2.canonical_payload["one_shot_growth_justification"] == _justification("scope-items")

    preserved, sidecar = assemble_authenticated_plan_revision(state2, patch(state2, [
        {"op": "replace", "field": "summary", "value": "Only the summary changes."},
    ]), result_round_number=3)
    assert preserved.one_shot_growth_justification == added.one_shot_growth_justification
    state3 = resumed(sidecar)

    changed, sidecar = assemble_authenticated_plan_revision(state3, patch(state3, [
        {"op": "replace", "field": "one_shot_growth_justification",
         "value": _justification("scope-items", "rendered-size")},
    ]), result_round_number=4)
    assert changed.one_shot_growth_justification.crossed_signals == ("scope-items", "rendered-size")
    state4 = resumed(sidecar)

    with pytest.raises(AgentLoopError, match="payload-identical"):
        assemble_authenticated_plan_revision(state4, patch(state4, [
            {"op": "replace", "field": "one_shot_growth_justification",
             "value": _justification("scope-items", "rendered-size")},
        ]), result_round_number=5)

    cleared, sidecar = assemble_authenticated_plan_revision(state4, patch(state4, [
        {"op": "replace", "field": "one_shot_growth_justification", "value": None},
    ]), result_round_number=5)
    assert cleared.one_shot_growth_justification is None
    state5 = resumed(sidecar)
    assert "one_shot_growth_justification" not in state5.canonical_payload

    with pytest.raises(AgentLoopError, match="with null has no effect"):
        assemble_authenticated_plan_revision(state5, patch(state5, [
            {"op": "replace", "field": "one_shot_growth_justification", "value": None},
        ]), result_round_number=6)


# --- rendering --------------------------------------------------------------


def test_full_rendering_shows_the_justification():
    plan = _parsed(one_shot_growth_justification=_justification("scope-items"))
    canonical = render_canonical_plan_state(plan)
    assert "### One-shot growth justification" in canonical
    assert "`scope-items`" in canonical and RATIONALE in canonical
    assert "One-shot growth" not in render_canonical_plan_state(_parsed())


def test_compact_digest_keeps_signals_and_bounds_the_rationale():
    """Row `overflow-digest-justification`."""
    long_rationale = ("Reason " * 400)[:2000]
    plan = _parsed(
        plan_steps=[f"Step {index} " + "detail " * 60 for index in range(40)],
        one_shot_growth_justification=_justification(
            "rendered-size", "scope-items", rationale=long_rationale
        ),
    )
    digest = render_public_agent_comment(
        kind="plan_state", parsed=plan, agent="claude", raw_text=_state_text(), compact=True
    )
    section = digest.split("### One-shot growth justification (digest)", 1)[1].split("\n\n", 1)[0]
    assert "- Crossed signals: `rendered-size`, `scope-items`" in section
    assert "authenticated round metadata" in section
    assert len("### One-shot growth justification (digest)" + section) <= 1_200
    # The total budgeted digest text did not grow: the share came out of the
    # plan-steps share.
    budgeted = digest.split("## Plan", 1)[1].split("<!-- AGENT_RISK_TEST_MATRIX", 1)[0]
    assert len(budgeted) <= COMPACT_PLAN_DIGEST_BUDGET_CHARS + 200
    plain = render_public_agent_comment(
        kind="plan_state",
        parsed=_parsed(plan_steps=[f"Step {index} " + "detail " * 60 for index in range(40)]),
        agent="claude", raw_text=_state_text(), compact=True,
    )
    steps_with = digest.split("### Plan steps (digest)", 1)[1].split("\n\n", 1)[0]
    steps_without = plain.split("### Plan steps (digest)", 1)[1].split("\n\n", 1)[0]
    assert len(steps_with) < len(steps_without) <= 4_800
    # A short rationale is shown whole.
    short = render_public_agent_comment(
        kind="plan_state",
        parsed=_parsed(one_shot_growth_justification=_justification("scope-items")),
        agent="claude", raw_text=_state_text(), compact=True,
    )
    assert f"- Rationale: {RATIONALE}" in short and "clipped" not in short


def test_near_limit_digest_with_metadata_fits_and_oversize_is_refused(tmp_path):
    config = _config(tmp_path)
    long_rationale = ("Reason " * 400)[:2000]
    raw = _state_text(
        plan_steps=[f"Step {index} " + "detail " * 150 for index in range(80)],
        one_shot_growth_justification=_justification("rendered-size", rationale=long_rationale),
    )
    plan = validate_structured_plan_state(raw)
    canonical = render_canonical_plan_state(plan, config)
    assert len(canonical) > 60_000
    full = render_public_agent_comment(kind="plan_state", parsed=plan, agent="claude", config=config)
    metadata = PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1, subject="s",
        canonical_plan=canonical, raw_structured_coder_response=raw,
    )
    kwargs = dict(
        config=config, issue_number=56, kind="plan_state", parsed_plan=plan,
        full_comment=full, raw_text=raw, prior_items=(), model_used=None,
        surfaced_requirement_ids=(), requires_direct_discussion_ack=False,
    )
    body = orchestrator_module._assemble_structured_plan_round_body(metadata=metadata, **kwargs)
    text = str(body)
    assert "### One-shot growth justification (digest)" in text
    assert "`rendered-size`" in text
    oversized = PostedRoundMetadata(
        flow="plan", role="coder", agent="Claude", round_number=1, subject="s",
        canonical_plan=canonical, raw_structured_coder_response=raw,
        # Incompressible: this field is not in the transport spill set.
        compact_prior_summaries=(
            "".join(hashlib.sha256(str(index).encode()).hexdigest() for index in range(4_000)),
        ),
    )
    with pytest.raises(AgentLoopError, match="exceeds 60000 characters"):
        orchestrator_module._assemble_structured_plan_round_body(metadata=oversized, **kwargs)


# --- prompts ----------------------------------------------------------------


def test_prompts_carry_the_lever_thresholds_and_growth_notice(tmp_path):
    """Rows `below-threshold` (no notice) and `over-threshold-unjustified` (notice)."""
    config = _config(tmp_path)
    plan_prompt = build_issue_plan_prompt(56, config)
    assert "Plan-growth gate" in plan_prompt and "120000 characters" in plan_prompt
    review = build_plan_review_prompt(56, 1, "Plan.", config, reviewer="codex")
    assert "this detail belongs in a child plan; restructure as staged" in review
    assert "Orchestrator plan-growth notice" not in review
    notice = "Orchestrator plan-growth notice (not a reviewer finding): NOTICE-TEXT."
    review = build_plan_review_prompt(
        56, 1, "Plan.", config, reviewer="codex", plan_growth_notice=notice
    )
    assert "NOTICE-TEXT" in review and "semantic claim" in review
    compact_review = build_plan_review_prompt(
        56, 2, "Plan.", config, reviewer="codex", compact_context=True, plan_growth_notice=notice
    )
    assert "NOTICE-TEXT" in compact_review and "restructure as staged" in compact_review
    revision = build_plan_revision_prompt(56, 1, "Plan.", "Review.", config, plan_growth_notice=notice)
    assert "NOTICE-TEXT" in revision
    assert "`requires-child-planning` children" in revision and "(#720)" in revision
    semantic = build_plan_revision_prompt(
        56, 1, "Plan.", "Review.", config, response_form="semantic-patch-v1",
        base_round_number=1, base_state_identity="a" * 64, plan_growth_notice=notice,
    )
    assert "NOTICE-TEXT" in semantic
    assert "`one_shot_growth_justification` (value null removes it)" in semantic
    off = build_issue_plan_prompt(56, _config(tmp_path, plan_growth_gate="off"))
    assert "Plan-growth gate is off" in off


# --- orchestrator integration ------------------------------------------------


def test_ordinary_plan_is_approved_unchanged(tmp_path):
    """Row `below-threshold` through the live loop."""
    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(runner, issue_number=56, config=_config(tmp_path), plan_first=True) == 0
    assert len(_prompts(runner, "claude")) == 1
    assert all("Orchestrator plan-growth notice" not in prompt for prompt in _prompts(runner, "codex"))
    coder = [record for record in _plan_records(runner) if record.role == "coder"]
    assert "one_shot_growth_justification" not in coder[0].assembled_plan_sidecar["canonical_json"]


def test_fresh_candidate_is_self_checked_before_any_reviewer(tmp_path):
    """Rows `new-signal-validation` and `over-threshold-justified`."""
    runner = FakeRunner(
        claude_outputs=[
            _state_text(),
            _state_text(one_shot_growth_justification=_justification("scope-items")),
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _config(tmp_path, plan_growth_max_scope_items=1)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    planner = _prompts(runner, "claude")
    assert len(planner) == 2
    assert "Plan growth gate" in planner[1] and "`scope-items`" in planner[1]
    # Exactly one reviewer round, on the justified candidate only.
    reviews = _prompts(runner, "codex")
    assert len(reviews) == 1 and RATIONALE in reviews[0]
    coder = [record for record in _plan_records(runner) if record.role == "coder"]
    assert len(coder) == 1
    assert coder[0].assembled_plan_sidecar["canonical_json"]["one_shot_growth_justification"] == (
        _justification("scope-items")
    )


def test_fresh_candidate_with_stale_justification_is_rejected(tmp_path):
    """Row `shrink-below-threshold`: a leftover justification is cleared first."""
    runner = FakeRunner(
        claude_outputs=[
            _state_text(one_shot_growth_justification=_justification("scope-items")),
            _state_text(),
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(runner, issue_number=56, config=_config(tmp_path), plan_first=True) == 0
    planner = _prompts(runner, "claude")
    assert len(planner) == 2 and "stale" in planner[1]
    assert len(_prompts(runner, "codex")) == 1


@pytest.mark.parametrize("signal", ["scope-items", "revision-count"])
def test_resumed_noncompliant_approval_routes_to_a_planner_revision(tmp_path, signal):
    """Rows `over-threshold-unjustified` and `revision-count-crossing`."""
    history = _approved_history(tmp_path)
    canonical = render_canonical_plan_state(validate_structured_plan_state(structured_v1_plan_state()))
    overrides = (
        {"plan_growth_max_scope_items": 1}
        if signal == "scope-items"
        # Size between half and the full threshold; one candidate reaches
        # the lowered revision threshold.
        else {"plan_growth_max_chars": len(canonical) + 1_500, "plan_growth_max_revisions": 1}
    )
    assert len(canonical) >= 1_500
    runner = FakeRunner(
        issue_comments=history,
        claude_outputs=[_justify_patch(signal)],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_config(tmp_path, **overrides), plan_first=True
    ) == 0
    planner = _prompts(runner, "claude")
    assert len(planner) == 1
    assert "Orchestrator plan-growth notice (not a reviewer finding)" in planner[0]
    assert f"`{signal}`" in planner[0]
    posted = [str(item["body"]) for item in runner.issue_comments[len(history):]]
    coder = [
        record for record in _plan_records(runner)[len(_plan_records(SimpleNamespace(issue_comments=history))):]
        if record.role == "coder"
    ]
    assert len(coder) == 1 and coder[0].round_number == 2
    assert coder[0].prior_items == ()
    assert coder[0].assembled_plan_sidecar["canonical_json"]["one_shot_growth_justification"][
        "crossed_signals"
    ] == [signal]
    # The revision was reviewed before approval; no reviewer item was minted
    # for the notice.
    assert len(_prompts(runner, "codex")) == 1
    assert not any("[item-" in body and "growth" in body for body in posted)


def test_lowering_the_revision_threshold_crosses_earlier(tmp_path):
    """Row `revision-count-crossing`: the default threshold does not cross."""
    history = _approved_history(tmp_path)
    canonical = render_canonical_plan_state(validate_structured_plan_state(structured_v1_plan_state()))
    runner = FakeRunner(issue_comments=history)
    # Default revision threshold (6): the same history resumes to approval
    # without any planner turn.
    assert run_issue_loop(
        runner, issue_number=56,
        config=_config(tmp_path, plan_growth_max_chars=len(canonical) + 1_500),
        plan_first=True,
    ) == 0
    assert _prompts(runner, "claude") == []


def test_primary_approval_of_a_noncompliant_candidate_skips_the_phase_advance(tmp_path):
    """Row `prepanel-gate`."""
    staged = dict(
        reviewer=("codex", "gemini"), plan_review_policy="primary-then-panel",
        primary_plan_reviewer="codex", max_rounds=6,
    )
    # Round 1 under default thresholds: the primary approves, the panel is
    # pending.  The run is stopped before the phase advance.
    history_runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    with pytest.raises(Exception):
        run_issue_loop(
            history_runner, issue_number=56,
            config=_config(tmp_path, **{**staged, "max_rounds": 1}), plan_first=True,
        )
    history = list(history_runner.issue_comments)
    assert not any(record.phase == "plan-phase-advance" for record in _plan_records(history_runner))

    runner = FakeRunner(
        issue_comments=history,
        claude_outputs=[_justify_patch("scope-items")],
        codex_outputs=[structured_plan_review(state="approved")],
        gemini_outputs=[structured_plan_review(state="approved", reviewer="Google Gemini")],
    )
    assert run_issue_loop(
        runner, issue_number=56,
        config=_config(tmp_path, **staged, plan_growth_max_scope_items=1), plan_first=True,
    ) == 0
    new_records = _plan_records(runner)[len(_plan_records(history_runner)):]
    assert not any(record.phase == "plan-phase-advance" and record.round_number == 2 for record in new_records)
    coder = [record for record in new_records if record.role == "coder"]
    assert [record.round_number for record in coder] == [2]
    planner = _prompts(runner, "claude")
    assert len(planner) == 1 and "Orchestrator plan-growth notice" in planner[0]
    # No reviewer, and in particular no panel reviewer, ran on the
    # non-compliant round-1 candidate; the panel reviewed only the revision.
    assert not any(record.role == "reviewer" and record.round_number == 1 for record in new_records)
    assert [
        record.round_number
        for record in new_records
        if record.role == "reviewer" and record.agent == "Gemini"
    ] == [3]


def test_carried_approval_of_a_noncompliant_plan_fails_closed(tmp_path):
    """Rows `resume-carried-approval` and `legacy-or-off`."""
    history = [SimpleNamespace(body=str(item["body"])) for item in _approved_history(tmp_path)]
    for overrides in ({}, {"plan_growth_gate": "off", "plan_growth_max_scope_items": 1}):
        config = _config(tmp_path, **overrides)
        plan_text, plan_round = orchestrator_module._resume_plan_round(
            history, configured_reviewers=orchestrator_module.reviewers(config)
        )
        orchestrator_module._require_complete_canonical_plan_approval(
            history, config=config, plan_text=plan_text, plan_round=plan_round,
            human_requirements=(), error_message="incomplete",
        )
    config = _config(tmp_path, plan_growth_max_scope_items=1)
    plan_text, plan_round = orchestrator_module._resume_plan_round(
        history, configured_reviewers=orchestrator_module.reviewers(config)
    )
    with pytest.raises(AgentLoopError, match="incomplete Plan growth gate: .*Re-run planning"):
        orchestrator_module._require_complete_canonical_plan_approval(
            history, config=config, plan_text=plan_text, plan_round=plan_round,
            human_requirements=(), error_message="incomplete",
        )


def test_gate_off_approves_an_over_threshold_plan(tmp_path):
    """Row `legacy-or-off` through the live loop."""
    runner = FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = _config(tmp_path, plan_growth_gate="off", plan_growth_max_scope_items=1)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert len(_prompts(runner, "claude")) == 1


# --- rebind advisory ----------------------------------------------------------


def _rebind_record(canonical_plan, payload, *, round_number=2):
    sidecar = make_assembled_plan_sidecar(
        payload, round_number=round_number, response_form="fresh-plan-state",
        rendered_plan=canonical_plan,
    )
    return SimpleNamespace(body=_attach_round_metadata(
        canonical_plan + FOOTER,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=round_number,
            subject="s", canonical_plan=canonical_plan, response_form="fresh-plan-state",
            aggregate_plan_identity=sidecar.aggregate_identity,
            assembled_plan_sidecar=sidecar.to_payload(),
        ),
    ))


def test_rebind_advisory_measures_the_stored_canonical_text(tmp_path):
    """Row `rebind-advisory` case (f): the size comes from the stored text."""
    payload = structured_plan_revision_to_payload(
        _parsed(one_shot_growth_justification=_justification("rendered-size"))
    )
    state_only = render_canonical_plan_state(validate_structured_plan_state(json.dumps(payload) + FOOTER))
    canonical = state_only + "\n\n### Prior plan item dispositions\n- [item-1] resolved: " + "x" * 400
    config = _config(tmp_path, plan_growth_max_chars=len(state_only) + 200)
    plan_hash = orchestrator_module.approved_plan_hash(canonical)
    advisory = orchestrator_module._rebind_growth_advisory(
        config=config, issue_number=56, comments=[_rebind_record(canonical, payload)],
        pr_number=77, plan_hash=plan_hash,
    )
    assert advisory is not None and "`rendered-size`" in advisory
    assert "Reviewed justification signals: `rendered-size`" in advisory
    assert RATIONALE in advisory
    assert "AGENT_" not in advisory
    # A failure is logged and never raised.
    broken = orchestrator_module._rebind_growth_advisory(
        config=config, issue_number=56, comments=[object()], pr_number=77, plan_hash=plan_hash,
    )
    assert broken is None


def test_rebind_advisory_rationale_is_bounded_and_marker_safe(tmp_path):
    payload = structured_plan_revision_to_payload(
        _parsed(one_shot_growth_justification=_justification("scope-items", rationale="y" * 2000))
    )
    canonical = render_canonical_plan_state(validate_structured_plan_state(json.dumps(payload) + FOOTER))
    advisory = orchestrator_module._rebind_growth_advisory(
        config=_config(tmp_path, plan_growth_max_scope_items=1), issue_number=56,
        comments=[_rebind_record(canonical, payload)], pr_number=77,
        plan_hash=orchestrator_module.approved_plan_hash(canonical),
    )
    assert "y" * 400 not in advisory and "complete rationale in the authenticated plan round" in advisory


def test_rebind_advisory_skips_staged_and_structurally_small_plans(tmp_path):
    payload = structured_plan_revision_to_payload(_parsed())
    canonical = render_canonical_plan_state(_parsed())
    comments = [_rebind_record(canonical, payload)]
    plan_hash = orchestrator_module.approved_plan_hash(canonical)
    # Case (e): structurally small; reviewer findings are not an input.
    assert orchestrator_module._rebind_growth_advisory(
        config=_config(tmp_path), issue_number=56, comments=comments, pr_number=77,
        plan_hash=plan_hash,
    ) is None
    # A staged replacement never gets the advisory, even when it is large.
    from test_child_plan_provenance import fresh_staged_plan

    staged = validate_structured_plan_state(
        json.dumps(json.JSONDecoder().raw_decode(fresh_staged_plan())[0]) + FOOTER
    )
    staged_payload = structured_plan_revision_to_payload(staged)
    staged_canonical = render_canonical_plan_state(staged)
    assert orchestrator_module._rebind_growth_advisory(
        config=_config(tmp_path, plan_growth_max_chars=1, plan_growth_max_scope_items=1),
        issue_number=56, comments=[_rebind_record(staged_canonical, staged_payload)],
        pr_number=77, plan_hash=orchestrator_module.approved_plan_hash(staged_canonical),
    ) is None


def _conversion_patch(requirement):
    from test_child_plan_provenance import fresh_staged_plan

    recommendation = json.JSONDecoder().raw_decode(fresh_staged_plan())[0]["execution_recommendation"]
    recommendation["scope_items"][0]["requirement"] = requirement
    recommendation["scope_items"][0]["acceptance_criteria"] = ["The reviewed scope is complete."]
    recommendation["child_stages"][0]["acceptance_criteria"] = ["The reviewed scope is complete."]
    return json.dumps({
        "schema_version": 1, "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1, "state": "blocking",
        "summary": "Restructure as staged.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved", "note": "Staged."}
        ],
        "base_round_number": 1, "base_state_identity": _base_identity(),
        "operations": [
            {"op": "replace", "field": "execution_recommendation", "value": recommendation},
        ],
    }) + FOOTER


@pytest.mark.parametrize("gate", ["enforce", "off"])
def test_staged_conversion_revision_is_ledger_checked_with_the_gate_on_or_off(tmp_path, gate):
    """Rows `staged-conversion` and `legacy-or-off` through the live loop."""
    runner = FakeRunner(
        claude_outputs=[
            structured_v1_plan_state(),
            _conversion_patch("Implement a narrower scope."),
            _conversion_patch("Implement the reviewed scope."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["This detail belongs in a child plan; restructure as staged."],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = _config(tmp_path, plan_growth_gate=gate)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    planner = _prompts(runner, "claude")
    assert len(planner) == 3
    assert "must preserve the scope ledger" in planner[2]
    assert "`scope-1` changed its requirement text" in planner[2]
    assert "narrower scope" not in "\n".join(str(item["body"]) for item in runner.issue_comments)
    coder = [record for record in _plan_records(runner) if record.role == "coder"]
    assert coder[-1].assembled_plan_sidecar["canonical_json"]["execution_recommendation"][
        "strategy"
    ] == "staged"
    # The rejected conversion never reached a reviewer.
    assert len(_prompts(runner, "codex")) == 2
