"""Plan-first loop tests: plan comment publication and sidecars (#1543)."""

import json
import re
from dataclasses import replace

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.github import HumanReviewRequirement
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    FakeRunner,
    _M948_FOOTER,
    _M948_NOTICE,
    _add_default_requirement_disposition,
    _m948_ack_block,
    _m948_assert_posted_acknowledgement,
    _m948_bodies,
    _m948_digest_anchor,
    _m948_hydrated_metadata,
    _m948_noise,
    _m948_plan_config,
    _m948_signed_payload,
    _m948_stages,
    _metadata_from_public_comment,
)
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    make_config,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_v1_plan_state,
    structured_issue_implementation,
)
import coding_review_agent_loop.round_transport as _m948_transport  # noqa: E402
from coding_review_agent_loop.comment_rendering import render_public_agent_comment  # noqa: E402,F811
from coding_review_agent_loop.protocol import (
    validate_structured_plan_state as _m948_validate_state,
)
from coding_review_agent_loop.protocol import (  # noqa: E402
    parse_risk_test_matrix_evidence as _parse_evidence_959,
)


def test_plan_round_metadata_failure_raises_diagnosed_error(tmp_path, monkeypatch):
    """#879: a contradictory metadata record must not escape as a bare ValueError."""
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    real_metadata = panel_evidence.PostedRoundMetadata

    def _exploding(**kwargs):
        if kwargs.get("flow") == "plan" and kwargs.get("role") == "coder":
            raise ValueError("semantic patch metadata is incomplete")
        return real_metadata(**kwargs)

    patch_attr_in(monkeypatch, (panel_evidence, discuss_loop, child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "PostedRoundMetadata", _exploding)

    with pytest.raises(AgentLoopError) as error:
        run_issue_loop(
            runner, issue_number=56, config=make_config(tmp_path), plan_first=True
        )

    assert "Could not record the plan round metadata" in str(error.value)
    assert "semantic patch metadata is incomplete" in str(error.value)


_M948_CANONICAL_ONLY_RECORDS = (
    "AGENT_DEFERRED_STAGES:",
    "AGENT_TYPED_PLAN_STAGES:",
    "AGENT_PLAN_EXPECTED_CLOSING_ISSUES:",
)


@pytest.mark.parametrize(
    "record_class", ["deferred-stages", "typed-stages", "expected-closing"]
)
def test_m948_oversized_collection_posts_digest_and_resume_recovers_it_from_metadata(
    tmp_path, record_class
):
    payload = json.loads(structured_plan_state(summary="Collections plan.").split("\n", 1)[0])
    if record_class == "deferred-stages":
        payload["deferred_stages"] = _m948_stages("Deferred")
    elif record_class == "typed-stages":
        payload["external_dependencies"] = _m948_stages("External", 100)
        payload["deferred_work"] = _m948_stages("Later", 100)
        payload["plan_actions"] = _m948_stages("Action", 100)
    else:
        payload["additional_closing_issue_ids"] = list(range(100, 140))
        payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    plan = json.dumps(payload) + _M948_FOOTER
    parsed = _m948_validate_state(plan)
    # The full visible body alone is over budget, embedded records included.
    assert len(render_public_agent_comment(kind="plan_state", parsed=parsed, agent="claude")) > 60_000

    # No reviewer output: the run stops with the plan published and unreviewed.
    first = FakeRunner(claude_outputs=[plan], codex_outputs=[])
    with pytest.raises(Exception):
        run_issue_loop(first, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True)

    bodies = _m948_bodies(first)
    assert all(len(body) <= _m948_transport.MAX_GITHUB_BODY_CHARS for body in bodies)
    anchor = _m948_digest_anchor(first)
    assert not any(record in anchor for record in _M948_CANONICAL_ONLY_RECORDS)
    assert "omitted; complete list in the authenticated attachments" in anchor
    assert "<!-- AGENT_PLAN_STATE: blocking -->" in anchor

    metadata = _m948_hydrated_metadata(anchor, bodies)
    canonical_plan = metadata["canonical_plan"]
    assert canonical_plan == plan_first_loop_module.render_canonical_plan_state(parsed) or (
        parsed.execution_recommendation is None and canonical_plan == plan
    )
    assert metadata["subject"] == panel_evidence._plan_subject(canonical_plan)
    assert _M948_NOTICE not in canonical_plan

    second = FakeRunner(
        issue_comments=first.issue_comments,
        claude_outputs=[],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    assert run_issue_loop(
        second, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0
    assert [cmd for cmd, _cwd in second.commands if cmd[:1] == ["claude"]] == []
    # Large prompts are delivered on stdin; the reviewer turn is the only agent
    # call of the resumed run.  The reviewer is prompted with the complete plan,
    # whose tail the digest omitted.
    codex_calls = [cmd for cmd, _cwd in second.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_calls) == 1
    review_prompt = second.last_input_text or codex_calls[0][-1]
    if record_class == "deferred-stages":
        assert _m948_noise("Deferred-299") in review_prompt
        assert _m948_noise("Deferred-299") not in anchor
    elif record_class == "typed-stages":
        assert _m948_noise("Action-99") in review_prompt
        assert _m948_noise("Action-99") not in anchor
    else:
        assert _m948_noise(299) in review_prompt
        assert _m948_noise(299) not in anchor
    assert len([body for body in _m948_bodies(second) if _M948_NOTICE in body]) == 1


def test_m948_fitting_plan_posts_the_full_form_byte_identically(tmp_path, monkeypatch):
    compact_calls = []
    real_render = response_validation.render_public_agent_comment

    full_renders = {}

    def spy(**kwargs):
        if kwargs.get("compact"):
            compact_calls.append(kwargs["kind"])
        rendered = real_render(**kwargs)
        full_renders[kwargs["kind"]] = rendered
        return rendered

    patch_attr_in(monkeypatch, (response_validation, discuss_loop, child_plan_binding, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "render_public_agent_comment", spy)
    plan = structured_plan_state(summary="Small plan.")
    revision = structured_plan_revision(
        summary="Small revision.",
        prior_plan_item_dispositions=[
            {"item_id": "item-1", "disposition": "resolved", "note": "Addressed."}
        ],
    )
    runner = FakeRunner(
        claude_outputs=[plan, revision],
        codex_outputs=[
            structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0

    assert compact_calls == []
    bodies = _m948_bodies(runner)
    assert not any(_M948_NOTICE in body for body in bodies)
    posted_plan = next(body for body in bodies if body.startswith("## Plan"))
    posted_revision = next(body for body in bodies if body.startswith("## Revised plan"))
    # The posted bodies are exactly the default renderings plus round metadata.
    # Attaching round metadata replaces the blank line before the trailing
    # footer with the metadata line; undo exactly that.
    def without_metadata(body):
        return re.sub(r"\n<!-- AGENT_LOOP_META: [^\n]* -->\n", "\n\n", body, count=1)

    assert without_metadata(posted_plan) == full_renders["plan_state"]
    assert without_metadata(posted_revision) == full_renders["plan_revision"]


def test_m948_non_size_transport_failure_aborts_without_selecting_the_digest(
    tmp_path, monkeypatch
):
    compact_calls = []
    real_render = response_validation.render_public_agent_comment

    def spy(**kwargs):
        if kwargs.get("compact"):
            compact_calls.append(kwargs["kind"])
        return real_render(**kwargs)

    def malformed(_body_text):
        raise AgentLoopError("Risk matrix marker is not a recoverable JSON object.")

    patch_attr_in(monkeypatch, (response_validation, discuss_loop, child_plan_binding, pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "render_public_agent_comment", spy)
    monkeypatch.setattr(_m948_transport, "_prepare_risk_test_matrix_transport", malformed)
    payload = json.loads(structured_plan_state().split("\n", 1)[0])
    payload["plan_steps"] = [f"Step {index}: {_m948_noise(index)}" for index in range(300)]
    runner = FakeRunner(claude_outputs=[json.dumps(payload) + _M948_FOOTER], codex_outputs=[])

    with pytest.raises(AgentLoopError, match="not a recoverable JSON object"):
        run_issue_loop(runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True)

    assert compact_calls == []
    assert runner.issue_comments == []


def _m948_requirement():
    return HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )


def test_m948_signed_requirements_fresh_and_revision_digests_revalidate_end_to_end(tmp_path):
    requirement_id = _m948_requirement().requirement_id
    ids = (requirement_id,)
    fresh = json.dumps(_m948_signed_payload("plan_state", ids)) + _m948_ack_block(ids) + _M948_FOOTER
    revision = (
        json.dumps({**_m948_signed_payload("plan_revision", ids), "summary": "Signed revision."})
        + _m948_ack_block(ids) + _M948_FOOTER
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[fresh, revision],
        codex_outputs=[
            _add_default_requirement_disposition(
                structured_plan_review(state="blocking", blocking_plan_issues=["Tighten the plan."]),
                requirement_id=requirement_id,
            ),
            _add_default_requirement_disposition(
                structured_plan_review(
                    state="approved",
                    prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                    human_requirements_resolved=True,
                ),
                requirement_id=requirement_id,
            ),
        ],
    )
    assert run_issue_loop(
        runner, issue_number=56, config=_m948_plan_config(tmp_path), plan_first=True
    ) == 0

    anchors = [body for body in _m948_bodies(runner) if _M948_NOTICE in body]
    assert [anchor.split("\n", 1)[0] for anchor in anchors] == ["## Plan", "## Revised plan"]
    for anchor in anchors:
        _m948_assert_posted_acknowledgement(anchor, ids, direct=False)


# --- #959: issue-to-PR coder metadata persists the full-matrix anchor -------



def _issue_evidence_959():
    return _parse_evidence_959({
        "matrix_identity": "c" * 64,
        "rows": [{
            "row_id": f"row-{index}",
            "status": "missing",
            "test_identifiers": [],
            "test_locations": [],
            "workflow_path_claim": f"Direct flow path {index}.",
            "outcome_assertions": [],
            "forbidden_effect_assertions": [],
            "evidence_citations": [],
            "caveats": [],
        } for index in range(2)],
    })


@pytest.mark.parametrize("with_evidence", [True, False], ids=["evidence", "no-evidence"])
def test_959_direct_issue_implementation_metadata_anchors_to_own_round(
    tmp_path, monkeypatch, with_evidence
):
    runner = FakeRunner(claude_outputs=[structured_issue_implementation(pr_number=77)])
    config = make_config(tmp_path, coder="claude", reviewer="codex")
    real_derive = response_validation._derive_authenticated_risk_evidence_for_coder

    def derive(result, **kwargs):
        derived, extra = real_derive(result, **kwargs)
        if with_evidence:
            derived = replace(derived, risk_test_matrix_evidence=_issue_evidence_959())
        return derived, extra

    patch_attr_in(monkeypatch, (response_validation, pr_loop, issue_implementation, issue_loop), "_derive_authenticated_risk_evidence_for_coder", derive)
    patch_attr_in(monkeypatch, (pr_loop, issue_implementation, plan_first_loop_module, issue_loop), "run_pr_loop", lambda *_a, **_k: 0)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    coder_comments = [
        item["body"] for item in runner.pr_payload.get("comments", [])
        if isinstance(item, dict) and isinstance(item.get("body"), str)
        and "AGENT_LOOP_META: " in item["body"]
        and _metadata_from_public_comment(item["body"]).role == "coder"
    ]
    assert len(coder_comments) == 1
    metadata = _metadata_from_public_comment(coder_comments[0])
    visible = coder_comments[0].split("AGENT_LOOP_META", 1)[0]
    if with_evidence:
        assert metadata.risk_test_matrix_evidence is not None
        assert metadata.risk_test_matrix_evidence_full_round == metadata.round_number
        assert metadata.risk_test_matrix_evidence_full_round_status == "valid"
        assert "<summary>Full matrix evidence (2 rows)</summary>" in visible
        assert "**row-0**" in visible and "**row-1**" in visible
    else:
        assert metadata.risk_test_matrix_evidence is None
        assert metadata.risk_test_matrix_evidence_full_round is None
        assert metadata.risk_test_matrix_evidence_full_round_status == "absent"
        assert "matrix evidence" not in visible
        assert "<details>" not in visible


@pytest.mark.parametrize('invalid', [False, True])
def test_plan_first_legacy_resume_observation_precedes_handoff(tmp_path, monkeypatch, invalid):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata, _plan_subject
    from coding_review_agent_loop.decomposition import approved_plan_hash
    from issue_loop_test_helpers import _provenance_pages
    from types import SimpleNamespace
    monkeypatch.setattr(issue_loop, 'resolve_canonical_pr_for_issue', lambda *a, **k: None)
    monkeypatch.setattr(plan_first_loop_module, 'resolve_canonical_pr_for_issue', lambda *a, **k: SimpleNamespace(source='legacy-closing-reference', pr_number=77, metadata=None, evidence_summary='authenticated fixture'))
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=approved "
            f"plan={approved_plan_hash(plan)}"
        ),
    )
    config = make_config(tmp_path)


    before = len(runner.comments)
    runner.branch_tip = 'b' * 40
    if invalid: runner.pr_payload['headRefOid'] = 'invalid'
    with pytest.raises(PrHeadObservationError):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True)
    assert not runner.pr_payload.get('comments')
    assert not any('Flow: approved-plan-implementation' in body for body in runner.comments[before:])
    assert not any(cmd[:1] in (['claude'], ['codex']) for cmd, _ in runner.commands)


def test_plan_first_legacy_resume_convergence_binds_complete_handoff(tmp_path, monkeypatch):
    from coding_review_agent_loop.pr_head_sync import PrHeadObservationError
    from coding_review_agent_loop.round_state import PostedRoundMetadata, _attach_round_metadata, _plan_subject
    from coding_review_agent_loop.decomposition import approved_plan_hash
    from issue_loop_test_helpers import _provenance_pages
    from types import SimpleNamespace
    monkeypatch.setattr(issue_loop, 'resolve_canonical_pr_for_issue', lambda *a, **k: None)
    monkeypatch.setattr(plan_first_loop_module, 'resolve_canonical_pr_for_issue', lambda *a, **k: SimpleNamespace(source='legacy-closing-reference', pr_number=77, metadata=None, evidence_summary='authenticated fixture'))
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:00Z",
                "body": _attach_round_metadata(
                    plan,
                    PostedRoundMetadata(
                        flow="plan",
                        role="coder",
                        agent="Claude",
                        round_number=1,
                        subject=_plan_subject(plan),
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:01Z",
                "body": _attach_round_metadata(
                    "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        state="approved",
                    ),
                ),
            },
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        open_prs_payload=[{"number": 77, "body": "Fixes #56"}],
        pr_commit_pages=_provenance_pages(
            "Implement issue.\n\nAgent-Issue-Provenance: v1 repo=owner/repo issue=56 flow=approved "
            f"plan={approved_plan_hash(plan)}"
        ),
    )
    config = make_config(tmp_path)



    tip = 'b' * 40
    runner.branch_tip = tip
    original = runner.run
    def run(args, **kwargs):
        if args == ['sleep', '2']:
            runner.pr_payload.update(headRefOid=tip, url='https://github.com/owner/repo/pull/77/')
        return original(args, **kwargs)
    runner.run = run
    delegated = []
    monkeypatch.setattr(plan_first_loop_module, 'run_pr_loop', lambda *a, **k: delegated.append(k) or 0)
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True) == 0
    handoffs = [body for body in runner.comments if 'Flow: approved-plan-implementation' in body]
    assert len(handoffs) == 1
    assert tip in handoffs[0]
    assert 'https://github.com/owner/repo/pull/77/' in handoffs[0]
    assert len(delegated) == 1
