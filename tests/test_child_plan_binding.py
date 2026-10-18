"""Child-plan binding tests: plan validation diagnostics and structured plan rounds (#1543)."""

import json
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError
from coding_review_agent_loop.errors import (
    AgentInvocationError,
    DeterministicPlanValidationExhaustion,
)
from coding_review_agent_loop.github import IssueComment, IssueContext
from coding_review_agent_loop.prompts import HUMAN_REQUIREMENTS_ADDRESSED_MARKER
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.panel_evidence as panel_evidence
from issue_loop_test_helpers import (
    FakeRunner,
    _M948_FOOTER,
    _M948_NOTICE,
    _PlanDiagnosticRunner,
    _m948_ack_block,
    _m948_assert_posted_acknowledgement,
    _m948_bodies,
    _m948_digest_anchor,
    _m948_hydrated_metadata,
    _m948_noise,
    _m948_plan_config,
    _m948_signed_payload,
)
from agent_loop_helpers import make_config
import coding_review_agent_loop.comment_rendering as _m948_rendering  # noqa: E402
import coding_review_agent_loop.round_transport as _m948_transport  # noqa: E402
from coding_review_agent_loop.comment_rendering import render_public_agent_comment  # noqa: E402,F811
from coding_review_agent_loop.github import post_issue_comment as _m948_post_issue_comment  # noqa: E402
from coding_review_agent_loop.protocol import (  # noqa: E402
    validate_human_requirements_acknowledgement as _m948_validate_ack,
    validate_structured_plan_revision as _m948_validate_revision,
    validate_structured_plan_state as _m948_validate_state,
)
from coding_review_agent_loop.round_state import PostedRoundMetadata as _M948Metadata  # noqa: E402


def test_exhausted_fresh_plan_validation_diagnostic_survives_resume(tmp_path):
    runner = _PlanDiagnosticRunner()
    config = make_config(tmp_path, execution_strategy_contract_required=True)
    issue_context = IssueContext(
        number=813,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/813",
        comments=(),
    )
    original_error = AgentInvocationError(
        "deterministic validator rejected the plan",
        failure_category="deterministic",
    )
    exhaustion = DeterministicPlanValidationExhaustion(
        candidate_kind="plan_state",
        candidate_text='{"kind":"plan_state"}',
        diagnostic="missing complete-scope audit operation",
        candidate_digest="a" * 64,
    )
    child_plan_binding._persist_exhausted_plan_validation_diagnostic(
        runner,
        config=config,
        issue_context=issue_context,
        issue_number=813,
        original_error=original_error,
        exhaustion=exhaustion,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        require_execution_strategy_contract=True,
        require_risk_test_matrix_contract=True,
    )
    assert len(runner.diagnostic_posts) == 1
    body = runner.diagnostic_posts[0]
    assert "901" not in body
    assert "2026-09-17T05:30:00Z" not in body
    resumed_context = replace(
        issue_context,
        comments=(
            IssueComment(
                author="agent",
                author_id=7,
                comment_id=901,
                created_at="2026-09-17T05:30:00Z",
                body=body,
            ),
        ),
    )
    diagnostic = child_plan_binding._recover_current_plan_validation_diagnostic(
        runner,
        config=config,
        issue_context=resumed_context,
        issue_number=813,
        target_coder_round=1,
        prior_plan_subject=None,
        candidate_kind="plan_state",
        require_execution_strategy_contract=True,
        require_risk_test_matrix_contract=True,
    )
    assert diagnostic is not None
    assert diagnostic.failure_attempt == 1
    assert diagnostic.diagnostic == "missing complete-scope audit operation"


def test_validation_record_post_failure_preserves_original_error(tmp_path):
    runner = _PlanDiagnosticRunner(post_returncode=1)
    config = make_config(tmp_path, execution_strategy_contract_required=True)
    issue_context = IssueContext(
        number=813, repo="OWNER/REPO", title="Issue", body="Issue", url=None, comments=()
    )
    original_error = AgentInvocationError(
        "original deterministic validator cause",
        failure_category="deterministic",
    )
    exhaustion = DeterministicPlanValidationExhaustion(
        candidate_kind="plan_revision",
        candidate_text='{"kind":"plan_revision"}',
        diagnostic="revision validation failed",
        candidate_digest="b" * 64,
    )
    with pytest.raises(AgentInvocationError) as raised:
        child_plan_binding._persist_exhausted_plan_validation_diagnostic(
            runner,
            config=config,
            issue_context=issue_context,
            issue_number=813,
            original_error=original_error,
            exhaustion=exhaustion,
            target_coder_round=2,
            prior_plan_subject="c" * 64,
            candidate_kind="plan_revision",
            require_execution_strategy_contract=True,
            require_risk_test_matrix_contract=True,
        )
    assert "original deterministic validator cause" in str(raised.value)
    assert "not persisted" in str(raised.value)
    assert runner.diagnostic_posts


def _m948_assemble(kind, parsed, raw_text, *, config, ids, direct, metadata=None):
    full_comment = render_public_agent_comment(
        kind=kind, parsed=parsed, agent=config.coder, raw_text=raw_text, config=config
    )
    canonical_plan = (
        plan_first_loop_module.render_canonical_plan_state(parsed)
        if kind == "plan_state"
        else plan_first_loop_module.render_canonical_plan_revision(parsed, (), config)
    )
    return child_plan_binding._assemble_structured_plan_round_body(
        config=config,
        issue_number=56,
        kind=kind,
        parsed_plan=parsed,
        full_comment=full_comment,
        metadata=metadata or _M948Metadata(
            flow="plan", role="coder", agent="Claude", round_number=1,
            subject=panel_evidence._plan_subject(canonical_plan),
            canonical_plan=canonical_plan, raw_structured_coder_response=raw_text,
        ),
        raw_text=raw_text,
        prior_items=(),
        model_used=None,
        surfaced_requirement_ids=ids,
        requires_direct_discussion_ack=direct,
    )


@pytest.mark.parametrize("form", ["fresh-plan-state", "legacy-full-state", "semantic-patch-v1"])
@pytest.mark.parametrize("direct", [False, True], ids=["surfaced-ids", "direct-discussion"])
def test_m948_signed_requirements_digest_posted_body_revalidates(tmp_path, form, direct):
    config = _m948_plan_config(tmp_path)
    # Direct-discussion mode surfaces no IDs: the prompt omitted them all.
    ids = () if direct else tuple(f"hr-{index:064x}" for index in range(1, 7))
    kind = "plan_state" if form == "fresh-plan-state" else "plan_revision"
    payload = _m948_signed_payload(kind, ids)
    payload["prior_plan_item_dispositions"] = []
    if kind == "plan_state":
        payload.pop("prior_plan_item_dispositions")
    structured = json.dumps(payload) + _M948_FOOTER
    parsed = (_m948_validate_state if kind == "plan_state" else _m948_validate_revision)(structured)
    block = _m948_ack_block(ids, direct=direct)
    if form == "semantic-patch-v1":
        # The semantic site renders the assembled plan but passes the raw patch
        # response, which is where the acknowledgement block lives.
        raw_text = json.dumps({"kind": "plan_revision_patch", "operations": []}) + block + _M948_FOOTER
    else:
        raw_text = json.dumps(payload) + block + _M948_FOOTER
    _m948_validate_ack(raw_text, surfaced_requirement_ids=ids, requires_direct_discussion_ack=direct)

    body = _m948_assemble(kind, parsed, raw_text, config=config, ids=ids, direct=direct)
    runner = FakeRunner()
    _m948_post_issue_comment(runner, config=config, issue_number=56, body=body)

    anchor = _m948_digest_anchor(runner)
    _m948_assert_posted_acknowledgement(anchor, ids, direct=direct)
    metadata = _m948_hydrated_metadata(anchor, _m948_bodies(runner))
    assert metadata["raw_structured_coder_response"] == raw_text


def test_m948_fresh_plan_without_acknowledgement_block_posts_every_disposition_id(tmp_path):
    config = _m948_plan_config(tmp_path)
    ids = tuple(f"hr-{index:064x}" for index in range(1, 4))
    raw_text = json.dumps(_m948_signed_payload("plan_state", ids)) + _M948_FOOTER
    parsed = _m948_validate_state(raw_text)

    body = _m948_assemble("plan_state", parsed, raw_text, config=config, ids=ids, direct=False)

    assert _M948_NOTICE in body
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER not in body
    assert all(f"- **{item}** — `addressed`: " in body for item in ids)


def test_m948_digest_failing_revalidation_is_never_posted(tmp_path, monkeypatch):
    config = _m948_plan_config(tmp_path)
    ids = tuple(f"hr-{index:064x}" for index in range(1, 4))
    raw_text = (
        json.dumps(_m948_signed_payload("plan_state", ids)) + _m948_ack_block(ids) + _M948_FOOTER
    )
    parsed = _m948_validate_state(raw_text)
    monkeypatch.setattr(
        _m948_rendering, "_compact_human_requirements_block",
        lambda block: block.replace(ids[1], "hr-" + "f" * 64),
    )
    with pytest.raises(AgentLoopError):
        _m948_assemble("plan_state", parsed, raw_text, config=config, ids=ids, direct=False)


def test_m948_overflow_the_digest_cannot_fix_fails_closed_and_posts_nothing(tmp_path):
    config = _m948_plan_config(tmp_path)
    raw_text = json.dumps(_m948_signed_payload("plan_state", ())) + _M948_FOOTER
    parsed = _m948_validate_state(raw_text)
    canonical_plan = plan_first_loop_module.render_canonical_plan_state(parsed)
    metadata = _M948Metadata(
        flow="plan", role="coder", agent="Claude", round_number=1,
        subject=panel_evidence._plan_subject(canonical_plan),
        canonical_plan=canonical_plan,
        # Not in the transport spill set, so no presentation change can help.
        compact_prior_summaries=(_m948_noise("unspillable", 120_000),),
    )
    runner = FakeRunner()
    # The final fit check (#886) refuses the digest during assembly, so no
    # post is ever attempted.
    with pytest.raises(_m948_transport.RoundCommentOverflowError) as excinfo:
        _m948_assemble(
            "plan_state", parsed, raw_text, config=config, ids=(), direct=False, metadata=metadata
        )
    assert "compact_prior_summaries=" in str(excinfo.value)
    assert _m948_noise("unspillable", 64) not in str(excinfo.value)
    assert runner.issue_comments == []
