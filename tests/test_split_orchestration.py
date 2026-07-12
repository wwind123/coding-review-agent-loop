"""Integration tests for plan-first split/deferred-stage materialization and
the selected-stage implementation handoff (#476)."""
import json

import pytest

from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import (
    approved_plan_hash,
    format_one_shot_impl_handoff_comment,
)
from coding_review_agent_loop.github import validate_pr_body_does_not_close_issue
from coding_review_agent_loop.orchestrator import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _plan_subject,
)
from coding_review_agent_loop.split_materialization import (
    MaterializedSplitChild,
    SplitMaterializationMetadata,
    format_split_materialization_summary,
    split_stage_proposal_from_text,
)
from agent_loop_helpers import (
    FakeRunner,
    command_index,
    make_config,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_pr_review,
)


def _existing_split_children_comment() -> dict:
    metadata = SplitMaterializationMetadata(
        parent_issue=56,
        subject="prior-discuss-subject",
        children=(
            MaterializedSplitChild(
                title="Auth flow",
                key=split_stage_proposal_from_text("Auth flow").key,
                url="https://github.com/OWNER/REPO/issues/101",
                number=101,
                origin="created",
            ),
            MaterializedSplitChild(
                title="Billing flow",
                key=split_stage_proposal_from_text("Billing flow").key,
                url="https://github.com/OWNER/REPO/issues/102",
                number=102,
                origin="created",
            ),
        ),
    )
    return {
        "author": {"login": "bot"},
        "createdAt": "2026-05-23T00:00:00Z",
        "body": format_split_materialization_summary(parent_issue=56, metadata=metadata),
    }


def test_plan_first_no_prior_split_materializes_deferred_stages_before_implementation(tmp_path):
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56"},
        claude_outputs=[
            structured_plan_state(
                state="blocking",
                summary="Implement the core parser change.",
                deferred_stages=[{"title": "Auth flow", "summary": "Split follow-up out."}],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "[#56 stage] Auth flow"
    materialize_index = command_index(runner.commands, ["gh", "issue", "create"])
    pr_create_index = command_index(runner.commands, ["claude", "--print"], start=materialize_index)
    assert materialize_index < pr_create_index
    assert any("<!-- AGENT_DISCUSS_SPLIT:" in comment for comment in runner.comments)
    # The parent's own plan covers everything except the deferred stage, so
    # its own PR still closes the parent issue normally (default PR body
    # scripted by FakeRunner is "Fixes #56").
    assert "Fixes #56" in runner.pr_payload["body"]


def test_plan_first_deferred_stage_title_with_colon_round_trips_through_canonical_markdown(tmp_path):
    """A revised plan's `deferred_stages` are carried in `current_plan` as the
    canonical revision markdown (not the raw structured JSON), which renders
    each stage as a human-readable `- {title}: {summary}` bullet. A title that
    itself contains a colon (e.g. "Stage 2: API follow-up") must still
    round-trip exactly through materialization instead of being corrupted by
    a naive split-on-first-colon parse (#492 review)."""
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56"},
        claude_outputs=[
            structured_plan_state(state="blocking", summary="Initial plan."),
            structured_plan_revision(
                summary="Implement the core parser change.",
                deferred_stages=[
                    {"title": "Stage 2: API follow-up", "summary": "Split out the API work."}
                ],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Needs a revision.",
                blocking_plan_issues=["Needs a revision."],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/101"],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == "[#56 stage] Stage 2: API follow-up"
    assert "Part of #56" in runner.issues[0]["body"]
    assert "Split out the API work." in runner.issues[0]["body"]


def test_plan_first_no_prior_split_warns_when_materialization_disabled(tmp_path):
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56"},
        claude_outputs=[
            structured_plan_state(
                state="blocking",
                summary="Implement the core parser change.",
                deferred_stages=[{"title": "Auth flow", "summary": "Split follow-up out."}],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
    )
    config = make_config(tmp_path)  # materialize_split_issues defaults False

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert runner.issues == []
    assert any("NOT filed as issues" in comment and "Auth flow" in comment for comment in runner.comments)


def test_plan_first_selected_stage_handoff_by_unique_title_match(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(state="blocking", summary="Auth flow"),
            "Implemented the auth-flow stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[_existing_split_children_comment()],
        pr_payload={"body": "Closes #101\n\nRefs #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    # No new child issues: both stages were already materialized before this run.
    assert runner.issues == []
    assert any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)
    stage_handoff = next(c for c in runner.comments if "<!-- AGENT_SPLIT_STAGE_HANDOFF:" in c)
    assert "handed off to split stage" in stage_handoff
    # The implementation prompt sent to the coder must target the resolved
    # child (#101) and instruct staged Closes/Refs wording, not Fixes on the parent.
    implement_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    implementation_prompt = implement_calls[1][-1]
    assert "issue #101" in implementation_prompt
    assert "Closes #101" in implementation_prompt
    assert "Refs #56" in implementation_prompt
    assert "Do NOT use a closing keyword" in implementation_prompt


def test_plan_first_selected_stage_handoff_via_split_stage_flag(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            # Plan text/summary does not uniquely match either child title, so
            # resolution must come from --split-stage.
            structured_plan_state(state="blocking", summary="Some unrelated plan summary"),
            "Implemented the billing stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[_existing_split_children_comment()],
        pr_payload={"body": "Closes #102\n\nRefs #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True, split_stage=102)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    stage_handoff = next(c for c in runner.comments if "<!-- AGENT_SPLIT_STAGE_HANDOFF:" in c)
    assert "issues/102" in stage_handoff


def test_plan_first_selected_stage_ambiguous_without_flag_raises(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(state="blocking", summary="Some unrelated plan summary"),
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
        ],
        issue_comments=[_existing_split_children_comment()],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    with pytest.raises(AgentLoopError, match="--split-stage"):
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )


def test_plan_first_deferred_stage_rerun_resumes_parent_one_shot_handoff(tmp_path):
    """A parent plan that declares its own `deferred_stages` keeps its primary
    scope on the parent; materializing the deferred remainder as a child issue
    must not make a rerun mistake the parent's own already-handed-off one-shot
    PR for an unresolved selected-stage handoff (#492 review)."""
    plan = structured_plan_state(
        state="blocking",
        summary="Implement the core parser change.",
        deferred_stages=[{"title": "Auth flow", "summary": "Split follow-up out."}],
    )
    plan_subject = _plan_subject(plan)
    plan_hash = approved_plan_hash(plan)
    split_metadata = SplitMaterializationMetadata(
        parent_issue=56,
        subject=plan_subject,
        children=(
            MaterializedSplitChild(
                title="Auth flow",
                key=split_stage_proposal_from_text("Auth flow").key,
                url="https://github.com/OWNER/REPO/issues/101",
                number=101,
                origin="created",
            ),
        ),
    )
    handoff = format_one_shot_impl_handoff_comment(
        parent_issue=56,
        mode="implement-one-shot",
        plan_hash=plan_hash,
        plan_subject=plan_subject,
        pr_number=77,
        pr_head_sha="abc123",
    )
    runner = FakeRunner(
        pr_payload={"body": "Fixes #56"},
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
                        subject=plan_subject,
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
                        subject=plan_subject,
                        state="approved",
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:02Z",
                "body": format_split_materialization_summary(parent_issue=56, metadata=split_metadata),
            },
            {"author": {"login": "bot"}, "createdAt": "2026-05-23T00:00:03Z", "body": handoff},
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    # Resumes the parent's own PR #77 review directly; no re-implementation,
    # no new child issues, and no selected-stage handoff.
    assert runner.issues == []
    assert not any(cmd[:1] == ["claude"] for cmd, _cwd in runner.commands)
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)


def test_plan_first_materializes_prior_discuss_split_and_hands_off_in_same_run(tmp_path):
    """A prior `discuss` run chose `split` but never materialized. Running
    `issue --plan-first --implement-after-approval --materialize-split-issues`
    on the parent with a plan that itself covers one of those split proposals
    (no `deferred_stages` declared) must, within this SAME run: file every
    proposal as a child issue, then still resolve and hand off implementation
    to the specific child the approved plan covers — not fall back to
    implementing the parent as a monolith because stage resolution read a
    stale pre-materialization issue-comments snapshot (#492 review)."""
    plan = structured_plan_state(state="blocking", summary="Auth flow")
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented the auth-flow stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-20T00:00:00Z",
                "body": _attach_round_metadata(
                    "Discuss consensus: split this into stages.\n-- coding-review-agent-loop",
                    PostedRoundMetadata(
                        flow="discuss",
                        role="summary",
                        agent="coding-review-agent-loop",
                        round_number=1,
                        subject="prior-discuss-subject",
                        is_final=True,
                        split_proposals=("Auth flow", "Billing flow"),
                    ),
                ),
            },
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
        pr_payload={"body": "Closes #101\n\nRefs #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    # Both split proposals were filed as children in this run...
    assert {issue["title"] for issue in runner.issues} == {
        "[#56 stage] Auth flow",
        "[#56 stage] Billing flow",
    }
    # ...and this same run still resolved and handed off to the specific
    # child ("Auth flow" -> #101) the approved plan covers, instead of
    # implementing the parent as a monolith.
    assert any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)
    stage_handoff = next(c for c in runner.comments if "<!-- AGENT_SPLIT_STAGE_HANDOFF:" in c)
    assert "handed off to split stage" in stage_handoff
    implement_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    implementation_prompt = implement_calls[1][-1]
    assert "issue #101" in implementation_prompt
    assert "Closes #101" in implementation_prompt
    assert "Refs #56" in implementation_prompt


def _legacy_split_debater_comment(*, evidence: dict | None = None) -> dict:
    """A final-round debater comment predating the `split_proposals` metadata
    field, forcing `_recover_final_discuss_split_proposals`'s legacy fallback
    to reconstruct proposals from the debater's own vote."""
    payload: dict = {
        "schema_version": 1,
        "kind": "discuss_review",
        "outcome": "split",
        "rationale": "Too broad for one PR.",
        "split_proposals": ["Auth flow", "Billing flow"],
    }
    if evidence is not None:
        payload["evidence"] = evidence
    body = _attach_round_metadata(
        json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="discuss", role="debater", agent="Codex", round_number=1,
            subject="prior-discuss-subject",
        ),
    )
    return {"author": {"login": "bot"}, "createdAt": "2026-05-20T00:00:00Z", "body": body}


def _legacy_split_final_summary_comment() -> dict:
    """A final discuss summary predating `split_proposals` metadata (empty
    here), so recovery must fall back to the debater comment above."""
    return {
        "author": {"login": "bot"},
        "createdAt": "2026-05-20T00:00:01Z",
        "body": _attach_round_metadata(
            "Discuss consensus: split this into stages.\n-- coding-review-agent-loop",
            PostedRoundMetadata(
                flow="discuss",
                role="summary",
                agent="coding-review-agent-loop",
                round_number=1,
                subject="prior-discuss-subject",
                is_final=True,
            ),
        ),
    }


def test_plan_first_recovers_legacy_split_proposals_with_valid_checkout_inspected_claim(tmp_path):
    """`_prior_discuss_split_proposals`'s legacy fallback
    (`_recover_final_discuss_split_proposals`) must checkout-validate a
    `checkout-inspected` evidence claim on the reconstructed debater vote,
    exactly like the primary discuss-loop recovery paths (#541)."""
    config = make_config(tmp_path, materialize_split_issues=True)
    (config.codex_dir / "src.py").write_text("line one\nline two\n", encoding="utf-8")
    plan = structured_plan_state(state="blocking", summary="Auth flow")
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented the auth-flow stage.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[
            _legacy_split_debater_comment(
                evidence={
                    "claims": [
                        {
                            "fact": "The referenced line supports splitting the work.",
                            "status": "verified",
                            "source": "src.py:2",
                            "verification_basis": "checkout-inspected",
                        }
                    ],
                    "updates": [],
                },
            ),
            _legacy_split_final_summary_comment(),
        ],
        issue_urls=[
            "https://github.com/OWNER/REPO/issues/101",
            "https://github.com/OWNER/REPO/issues/102",
        ],
        pr_payload={"body": "Closes #101\n\nRefs #56"},
    )

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    assert {issue["title"] for issue in runner.issues} == {
        "[#56 stage] Auth flow",
        "[#56 stage] Billing flow",
    }
    assert any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)


def test_plan_first_legacy_split_recovery_rejects_invalid_checkout_inspected_claim(tmp_path):
    config = make_config(tmp_path, materialize_split_issues=True)
    plan = structured_plan_state(state="blocking", summary="Auth flow")
    runner = FakeRunner(
        claude_outputs=[plan],
        codex_outputs=[structured_plan_review(state="approved")],
        issue_comments=[
            _legacy_split_debater_comment(
                evidence={
                    "claims": [
                        {
                            "fact": "The referenced line supports splitting the work.",
                            "status": "verified",
                            "source": "src/missing.py:1",
                            "verification_basis": "checkout-inspected",
                        }
                    ],
                    "updates": [],
                },
            ),
            _legacy_split_final_summary_comment(),
        ],
    )

    with pytest.raises(AgentLoopError, match="not a file in the assigned checkout"):
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )


def test_plan_first_excludes_own_scope_from_prior_discuss_proposals(tmp_path):
    """A prior `discuss` split named both `Auth flow` and `Billing flow`, but
    was never materialized. The approved plan implements `Auth flow` directly
    on the parent and structurally declares `Billing flow` as its own
    `deferred_stages` remainder. Only `Billing flow` (the true remainder) may
    be filed as a child issue: filing `Auth flow` too would create a
    duplicate child for scope the parent PR is about to implement and close
    directly, since a plan with its own `deferred_stages` never hands off to
    a selected-stage child (#492 review)."""
    plan = structured_plan_state(
        state="blocking",
        summary="Auth flow",
        deferred_stages=[{"title": "Billing flow", "summary": "Split follow-up out."}],
    )
    runner = FakeRunner(
        claude_outputs=[
            plan,
            "Implemented the auth-flow scope.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_comments=[
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-20T00:00:00Z",
                "body": _attach_round_metadata(
                    "Discuss consensus: split this into stages.\n-- coding-review-agent-loop",
                    PostedRoundMetadata(
                        flow="discuss",
                        role="summary",
                        agent="coding-review-agent-loop",
                        round_number=1,
                        subject="prior-discuss-subject",
                        is_final=True,
                        split_proposals=("Auth flow", "Billing flow"),
                    ),
                ),
            },
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/102"],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path, materialize_split_issues=True)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True
        )
        == 0
    )

    # Only the true remainder is filed; the plan's own covered scope is not
    # filed as a duplicate child.
    assert [issue["title"] for issue in runner.issues] == ["[#56 stage] Billing flow"]
    # No selected-stage handoff: the parent PR implements and closes Auth
    # flow directly, since the plan declares its own deferred_stages.
    assert not any("<!-- AGENT_SPLIT_STAGE_HANDOFF:" in comment for comment in runner.comments)
    assert "Fixes #56" in runner.pr_payload["body"]


def test_validate_pr_body_does_not_close_issue_rejects_parent_fixes_for_staged_pr(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Fixes #56"})
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match="closing keyword"):
        validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)


def test_validate_pr_body_does_not_close_issue_rejects_full_issue_url(tmp_path):
    """GitHub also treats a closing keyword followed by a full issue URL as an
    auto-close reference, e.g. `Closes https://github.com/OWNER/REPO/issues/56`;
    the close-keyword guard must reject that form too, not just `#56` /
    `owner/repo#56` (#492 review)."""
    runner = FakeRunner(pr_payload={"body": "Closes https://github.com/OWNER/REPO/issues/56"})
    config = make_config(tmp_path)
    with pytest.raises(AgentLoopError, match="closing keyword"):
        validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)


def test_validate_pr_body_does_not_close_issue_accepts_refs_url(tmp_path):
    runner = FakeRunner(pr_payload={"body": "Refs https://github.com/OWNER/REPO/issues/56"})
    config = make_config(tmp_path)
    validate_pr_body_does_not_close_issue(runner, config=config, pr_number=77, issue_number=56)
