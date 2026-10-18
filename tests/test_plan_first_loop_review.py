"""Plan-first loop tests: plan review rounds and plan-review follow-ups (#1543)."""

import json
import re
from dataclasses import replace
from unittest.mock import patch

import pytest

import coding_review_agent_loop.plan_first_loop as plan_first_loop_module
from coding_review_agent_loop.cli import AgentLoopError, run_issue_loop
from coding_review_agent_loop.decomposition import approved_plan_hash
from coding_review_agent_loop.github import HumanReviewRequirement, IssueContext
from coding_review_agent_loop.panel_evidence import PostedRoundMetadata, _plan_subject
from coding_review_agent_loop.discuss_loop import _attach_round_metadata
from coding_review_agent_loop.round_state import _decode_round_metadata, _strip_round_metadata
from coding_review_agent_loop.pr_loop_support import _resume_plan_round
from coding_review_agent_loop.prompts import (
    COMPACT_PLANNING_VOLATILE_TAIL_MARKER,
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
from coding_review_agent_loop.protocol import (
    ReviewItemDisposition,
    UnresolvedReviewItem,
    parse_plan_revision_patch,
    validate_structured_plan_state,
    risk_test_matrix_prompt_examples,
)
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.architecture_contract as architecture_contract
import coding_review_agent_loop.child_plan_binding as child_plan_binding
import coding_review_agent_loop.discuss_loop as discuss_loop
import coding_review_agent_loop.execution_policy as execution_policy
import coding_review_agent_loop.integration_close as integration_close
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.panel_evidence as panel_evidence
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
import coding_review_agent_loop.review_rounds as review_rounds
import coding_review_agent_loop.triage as triage
from agent_loop_helpers import patch_attr_in
from issue_loop_test_helpers import (
    FakeRunner,
    _add_default_requirement_disposition,
    _initial_plan_state,
    _plan_round_records,
    _plan_with_requirement_disposition,
    _repaired_acknowledgement_fixtures,
)
from coding_review_agent_loop.plan_assembly import AuthenticatedPlanState
from agent_loop_helpers import (
    FakeRunner as _FakeRunner,
    command_index,
    make_config,
    prior_plan_item_dispositions,
    structured_plan_review,
    structured_plan_revision,
    structured_plan_state,
    structured_v1_plan_state,
    structured_pr_review,
)
import coding_review_agent_loop.validated_agent as validated_agent_module


def _plan_review_with_requirement_disposition(
    *, state: str, summary: str, marker: bool = False, prior_dispositions: list[dict[str, str]] | None = None
) -> str:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_review",
                "state": state,
                "summary": summary,
                "blocking_plan_issues": [summary] if state == "blocking" else [],
                "same_plan_followups": [],
                "future_followups": [],
                "prior_plan_item_dispositions": prior_dispositions or [],
                "human_requirement_dispositions": [{
                    "requirement_id": "Requirement 1",
                    "disposition": "addressed",
                    "evidence": summary,
                }],
            }
        )
        + ("\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->" if marker else "")
        + f"\n<!-- AGENT_PLAN_STATE: {state} -->\n-- OpenAI Codex"
    )


def test_issue_loop_plan_first_stops_after_approved_plan(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                summary="Update the CLI and cover it with regression tests.",
                plan_steps=["Update the CLI.", "Add tests."],
            ),
        ],
        codex_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert any(cmd[:3] == ["claude", "--print", "--output-format"] for cmd, _cwd in runner.commands)
    assert any(cmd[:2] == ["codex", "exec"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "view"] for cmd, _cwd in runner.commands)
    assert len(runner.comments) == 3
    assert runner.comments[0].startswith("## Plan")
    assert runner.comments[1].startswith("**Review verdict:** Approved\n\nPlan looks sound.")
    assert "Outcome: implement" in runner.comments[2]
    assert not any(cmd[:2] == ["git", "fetch"] for cmd, _cwd in runner.commands)
    assert not any(cmd[:2] == ["git", "switch"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_rejects_missing_initial_plan_human_requirements_acknowledgement(
    tmp_path,
):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            structured_plan_state()
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_accepts_initial_plan_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Keep the public API unchanged.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            structured_plan_state().replace(
                "\n<!-- AGENT_PLAN_STATE: blocking -->",
                "\n"
                f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan keeps the public API unchanged.\n"
                "<!-- AGENT_PLAN_STATE: blocking -->",
                1,
            )
        ],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.", human_requirements_resolved=True)],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_issue_loop_plan_first_fails_closed_when_requirements_change_after_approval(
    tmp_path, monkeypatch
):
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )
    initial = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(requirement_1,),
    )
    refreshed = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Issue",
        body="Issue body",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(requirement_1, requirement_2),
    )
    contexts = iter((initial, refreshed))
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda *args, **kwargs: next(contexts),
    )
    plan_output = structured_plan_state(summary="Plan the compatibility fix.").replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": [{"requirement_id": "Requirement 1", "disposition": "addressed", "evidence": "The plan preserves the public API."}]',
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the plan preserves the public API.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                summary="Plan approved.",
                human_requirements_resolved=True,
                human_requirement_dispositions=[
                    {
                        "requirement_id": "Requirement 1",
                        "disposition": "addressed",
                        "evidence": "The plan preserves the public API.",
                    }
                ],
            )
        ],
    )

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"added hr-",
    ):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path), plan_first=True)

    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_fails_closed_when_a_requirement_is_withdrawn(
    tmp_path, monkeypatch
):
    """A withdrawal at the live approval boundary is a changed requirement set.

    Every plan review and every carried exact-key approval was bound to the
    earlier surfaced requirement digest, so a withdrawn ID must stop and
    re-enter planning rather than proceed to implementation (#905, from #841).
    """
    requirement_1 = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Keep the public API unchanged.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at="2026-05-17T08:10:00Z",
        url="https://github.com/OWNER/REPO/issues/56#issuecomment-2",
        body="Also preserve the audit trail.",
    )

    def _context(*requirements):
        return IssueContext(
            number=56,
            repo="OWNER/REPO",
            title="Issue",
            body="Issue body",
            url="https://github.com/OWNER/REPO/issues/56",
            comments=(),
            human_requirements=requirements,
        )

    contexts = iter(
        (_context(requirement_1, requirement_2), _context(requirement_1))
    )
    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context",
        lambda *args, **kwargs: next(contexts),
    )
    dispositions = [
        {
            "requirement_id": "Requirement 1",
            "disposition": "addressed",
            "evidence": "The plan preserves the public API.",
        },
        {
            "requirement_id": "Requirement 2",
            "disposition": "addressed",
            "evidence": "The plan preserves the audit trail.",
        },
    ]
    plan_output = structured_plan_state(summary="Plan the compatibility fix.").replace(
        '"human_requirement_dispositions": []',
        '"human_requirement_dispositions": ' + json.dumps(dispositions),
        1,
    ).replace(
        "\n<!-- AGENT_PLAN_STATE: blocking -->",
        "\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the plan preserves the public API.\n"
        "- Requirement 2: the plan preserves the audit trail.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->",
        1,
    )
    runner = FakeRunner(
        claude_outputs=[plan_output],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                summary="Plan approved.",
                human_requirements_resolved=True,
                human_requirement_dispositions=dispositions,
            )
        ],
    )

    with pytest.raises(
        AgentLoopError,
        match=r"Issue #56 signed human requirement\(s\) changed after plan approval: "
        r"withdrawn hr-",
    ):
        run_issue_loop(runner, issue_number=56, config=make_config(tmp_path), plan_first=True)

    # The run stops before implementation: no PR and no coder implementation turn.
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_first_revises_until_all_reviewers_approve(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan."),
            structured_plan_revision(summary="Revised plan with tests."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0


def test_issue_loop_plan_first_stops_on_incomplete_review_without_coder_followup(tmp_path):
    """An explicit inability to review is an agent failure, not a new blocker."""
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan."),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Review incomplete; unable to verify the retry path without further inspection.",
            ),
        ],
    )
    config = make_config(tmp_path, max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1
    assert "Review incomplete" not in "\n".join(runner.comments)


def test_issue_loop_plan_first_resume_ignores_incomplete_wording_outside_summary(tmp_path):
    """A resumed structured review must be scanned the same way a fresh one is.

    Before the resumed-branch summary reconstruction fix, `summary` was
    rebuilt from the raw (unrendered) response text even when a structured
    review had already been parsed, so incomplete-review wording living in an
    unrelated field (here, `future_followups`) could leak into the summary
    candidate text and falsely trip the incomplete-review heuristic only
    after a resume.
    """
    round1_plan = "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_round1_comment = _attach_round_metadata(
        round1_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(round1_plan),
        ),
    )
    resumed_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            summary="Reviewed the updated plan; still recommend revisiting the retry approach later.",
            future_followups=["Could not confirm the retry path in CI end-to-end; revisit later."],
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(round1_plan),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_round1_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resumed_reviewer_comment},
        ],
        claude_outputs=[structured_plan_revision(summary="Revised plan addressing round 1 feedback.")],
        codex_outputs=[structured_plan_review(summary="Plan looks sound now.")],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 1


def test_issue_loop_plan_first_resume_stops_on_incomplete_review_in_summary(tmp_path):
    """Genuine incomplete-review wording must still be caught after a resume."""
    round1_plan = "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_round1_comment = _attach_round_metadata(
        round1_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=1,
            subject=_plan_subject(round1_plan),
        ),
    )
    resumed_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            summary="Review incomplete; unable to confirm the retry path without further inspection.",
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(round1_plan),
            state="blocking",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_round1_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resumed_reviewer_comment},
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",), max_rounds=3)

    with pytest.raises(AgentLoopError, match="reviewer-internal error"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 0


def test_issue_loop_plan_review_allows_active_carried_blocking_disposition(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(summary="Initial plan.", plan_steps=["Document migration ordering."]),
            structured_plan_revision(
                summary="Added migration ordering.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                plan_steps=["Document migration ordering."],
            ),
            structured_plan_revision(
                summary="Clarified migration ordering.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                plan_steps=["Document migration ordering."],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Document migration ordering."],
            ),
            structured_plan_review(
                state="blocking",
                summary="Could not confirm the revised plan covers migration ordering.",
                prior_plan_item_dispositions=[{
                    "item_id": "item-1",
                    "disposition": "blocking",
                    "note": "Could not confirm the revised plan covers migration ordering.",
                }],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    assert any("Could not confirm the revised plan" in comment for comment in runner.comments)


def test_issue_loop_structured_plan_state_public_comment_renders_markdown_and_preserves_metadata(tmp_path):
    raw_structured_plan = structured_plan_state(
        summary="Plan the issue fix.",
        plan_steps=["Update the renderer.", "Add regression tests."],
        reviewer="Google Antigravity",
    )
    runner = FakeRunner(
        antigravity_outputs=[raw_structured_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, coder="antigravity", reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    public_comment = runner.comments[0]
    assert public_comment.startswith("## Plan")
    assert "### Plan steps\n1. Update the renderer.\n2. Add regression tests." in public_comment
    assert '"kind": "plan_state"' not in _strip_round_metadata(public_comment)

    raw_comment = runner.issue_comments[0]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.canonical_plan == raw_structured_plan
    assert metadata.raw_structured_coder_response == raw_structured_plan


def test_issue_loop_accepts_fresh_v1_plan_without_recommendation_driven_routing(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=783, config=config, plan_first=True) == 0
    assert runner.issues == []
    assert any("AGENT_EXECUTION_RECOMMENDATION" in comment for comment in runner.comments)


def test_issue_loop_accepts_fresh_v1_applicable_matrix(tmp_path):
    payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    payload["risk_test_matrix"] = risk_test_matrix_prompt_examples()["applicable"]
    candidate = json.dumps(payload) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = _FakeRunner(
        claude_outputs=[candidate],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=783, config=config, plan_first=True) == 0
    assert runner.issues == []


def test_fresh_v1_plan_host_resume_reuses_posted_round_without_new_agent_turn(tmp_path):
    runner = _FakeRunner(
        claude_outputs=[structured_v1_plan_state()],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        execution_strategy_contract_required=True,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0
    initial_agent_commands = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] or cmd[:2] == ["codex", "exec"]
    ]
    coder_metadata = next(
        _decode_round_metadata(match.group("payload"))
        for comment in runner.issue_comments
        if (match := re.search(
            r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
            comment["body"],
        ))
        and _decode_round_metadata(match.group("payload")).role == "coder"
    )
    assert coder_metadata.execution_strategy_contract_version == 1

    # The host rerun sees the durable canonical plan, raw response, sidecar,
    # and metadata identity. It must resume the approved plan without asking
    # either agent to produce a competing round.
    runner.claude_outputs = []
    runner.codex_outputs = []
    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    resumed_agent_commands = [
        cmd for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] or cmd[:2] == ["codex", "exec"]
    ]
    assert resumed_agent_commands == initial_agent_commands
    assert runner.issues == []


def test_plan_review_accepts_valid_response_file_after_nonzero_exit(tmp_path):
    artifact = structured_plan_review(
        state="approved", summary="The plan is sound.", reviewer="Anthropic Claude"
    )
    runner = FakeRunner(
        codex_outputs=[structured_plan_state(summary="Plan the issue fix.")],
        claude_outputs=[("Error: timeout waiting for response", 1)],
        public_response_outputs=["", artifact],
    )
    config = make_config(tmp_path, coder="codex", reviewer="claude", agent_max_retries=0)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len([cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]) == 1
    metadata = next(
        _decode_round_metadata(match.group("payload"))
        for comment in runner.issue_comments
        if (match := re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", comment["body"]))
        and _decode_round_metadata(match.group("payload")).role == "reviewer"
    )
    assert metadata.acquisition_outcome == "accepted_nonzero_exit"
    assert metadata.acquisition_returncode == 1


def test_issue_loop_plan_first_rejects_marker_only_plan_before_posting_or_reviewer_dispatch(tmp_path):
    markdown_plan = "Initial markdown plan.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    runner = FakeRunner(
        claude_outputs=[markdown_plan],
        codex_outputs=[structured_plan_review(summary="Plan looks sound.")],
    )
    config = make_config(tmp_path, reviewer="codex")

    with pytest.raises(AgentLoopError, match="structured `plan_state` JSON object"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert runner.comments == []
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def test_issue_loop_plan_first_rejects_generic_implementation_plan_before_posting(tmp_path):
    generic_plan = json.dumps({
        "summary": "Plan the fix.",
        "implementation_plan": ["Update the parser.", "Add tests."],
    }) + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    runner = FakeRunner(claude_outputs=[generic_plan])
    config = make_config(tmp_path, agent_max_retries=0)

    with pytest.raises(AgentLoopError, match="kind mismatch|missing required field"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert runner.comments == []
    assert not any(command[:2] == ["codex", "exec"] for command, _cwd in runner.commands)


def test_issue_loop_plan_revision_stores_raw_structured_metadata(tmp_path):
    raw_structured_revision = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "plan_revision",
                "state": "blocking",
                "summary": "Revised plan with tests.",
                "prior_plan_item_dispositions": [
                    {"item_id": "item-1", "disposition": "resolved", "note": "Added the missing test step."}
                ],
                "plan_steps": ["Add the regression test.", "Run the focused suite."],
                "architecture_impact": {
                    "status": "unchanged",
                    "rationale": "No architectural contract changed.",
                    "affected_components": [], "dependencies": [],
                    "execution_data_flows": [], "persistence": [],
                    "public_contracts": [], "security_boundaries": [],
                    "canonical_document_action": "no-change",
                    "canonical_document_path": None,
                    "canonical_document_rationale": "",
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[
            _initial_plan_state(),
            raw_structured_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing test strategy.",
                blocking_plan_issues=["Missing test strategy."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[2].startswith("## Revised plan")
    assert '"kind": "plan_revision"' not in _strip_round_metadata(runner.comments[2])
    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->", raw_comment)
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.raw_structured_coder_response == raw_structured_revision


def test_issue_loop_activates_semantic_revision_from_fresh_authenticated_base(tmp_path):
    fresh = structured_v1_plan_state()
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Apply the reviewed semantic decision.",
        "prior_plan_item_dispositions": [
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "rationale": "The revised plan addresses the original finding.",
            }
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "summary", "value": "Revised semantic plan."}
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.base_state_identity == base.state_identity
    assert metadata.raw_patch_provenance == {
        **patch,
        "prior_plan_item_dispositions": [{
            "item_id": "item-1",
            "disposition": "resolved",
            "note": "The revised plan addresses the original finding.",
        }],
    }
    assert metadata.assembled_plan_sidecar is not None
    assert "Revised semantic plan." in runner.comments[2]


def test_semantic_revision_replacing_architecture_impact_records_and_resumes(tmp_path):
    """#879: an architecture_impact replacement must round-trip its provenance."""
    fresh = structured_v1_plan_state()
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    architecture_impact = {
        "status": "changed",
        "rationale": "The revision adds a publication seam.",
        "affected_components": ["round_state.py"],
        "dependencies": [],
        "execution_data_flows": ["plan round -> metadata"],
        "persistence": ["round metadata"],
        "public_contracts": [],
        "security_boundaries": [],
        "canonical_document_action": "update",
        "canonical_document_path": "ARCHITECTURE.md",
        "canonical_document_rationale": "Record the new seam.",
    }
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Apply the reviewed semantic decision.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {
                "op": "replace",
                "field": "architecture_impact",
                "value": architecture_impact,
            }
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.raw_patch_provenance is not None
    # The stored provenance must be JSON, and must equal a freshly
    # re-serialized patch: restart compares exactly these two values.
    assert metadata.raw_patch_provenance == json.loads(
        json.dumps(metadata.raw_patch_provenance)
    )
    reparsed = parse_plan_revision_patch(metadata.raw_patch_provenance)
    assert reparsed.to_payload() == metadata.raw_patch_provenance

    # The recorded round must resume without tripping the integrity check.
    resumed = _resume_plan_round(
        [
            type("Comment", (), {"body": comment["body"]})()
            for comment in runner.issue_comments[:3]
        ],
        configured_reviewers=("codex",),
    )
    assert resumed is not None


def test_semantic_revision_inherits_signed_requirement_dispositions(tmp_path):
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )
    fresh_payload = json.loads(structured_v1_plan_state().split("\n", 1)[0])
    fresh_payload["human_requirement_dispositions"] = [{
        "requirement_id": requirement.requirement_id,
        "disposition": "addressed",
        "evidence": "The authenticated base preserves backward compatibility.",
    }]
    fresh = (
        json.dumps(fresh_payload)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the base preserves backward compatibility.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    parsed = validate_structured_plan_state(fresh)
    base = AuthenticatedPlanState.from_plan(parsed, round_number=1)
    patch = {
        "schema_version": 1,
        "kind": "plan_revision_patch",
        "semantic_patch_contract_version": 1,
        "state": "blocking",
        "summary": "Change only the plan summary.",
        "prior_plan_item_dispositions": [
            {"item_id": "item-1", "disposition": "resolved"}
        ],
        "base_round_number": 1,
        "base_state_identity": base.state_identity,
        "operations": [
            {"op": "replace", "field": "summary", "value": "Summary changed."}
        ],
    }
    patch_text = (
        json.dumps(patch)
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the authenticated disposition remains valid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    direct_patch_text = (
        json.dumps({**patch, "prior_plan_item_dispositions": []})
        + "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n"
        "### Human requirements\n"
        f"- Requirement {requirement.requirement_id}: the authenticated disposition remains valid.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    validated_patch = response_validation._validate_plan_revision_patch_response(
        direct_patch_text,
        human_requirements=(requirement,),
        inherited_human_requirement_dispositions=base.plan.human_requirement_dispositions,
    )
    assert validated_patch.operations[0].field == "summary"
    assert response_validation._current_plan_has_complete_human_requirement_dispositions(
        direct_patch_text,
        surfaced_requirement_ids=(requirement.requirement_id,),
        inherited_dispositions=base.plan.human_requirement_dispositions,
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[fresh, patch_text],
        codex_outputs=[
            structured_plan_review(
                state="blocking", blocking_plan_issues=["Review the plan."]
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
                human_requirements_resolved=True,
            ),
        ],
    )

    assert run_issue_loop(
        runner,
        issue_number=56,
        config=make_config(tmp_path, max_rounds=3, plan_execution_mode="plan-only"),
        plan_first=True,
    ) == 0

    raw_comment = runner.issue_comments[2]["body"]
    match = re.search(
        r"<!--\s*AGENT_LOOP_META:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
        raw_comment,
    )
    assert match is not None
    metadata = _decode_round_metadata(match.group("payload"))
    assert metadata.response_form == "semantic-patch-v1"
    assert metadata.assembled_plan_sidecar["canonical_json"]["human_requirement_dispositions"] == [
        {
            "requirement_id": requirement.requirement_id,
            "disposition": "addressed",
            "evidence": "The authenticated base preserves backward compatibility.",
        }
    ]


def test_issue_loop_plan_revision_rejects_missing_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            structured_plan_revision(summary="Revised plan."),
            "Revised plan.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan still preserves backward compatibility.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path)

    with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_revision_accepts_human_requirements_acknowledgement(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            structured_plan_revision(
                summary="Revised plan.",
                human_requirements=(
                    f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                    "### Human requirements\n"
                    "- Requirement 1: the revised plan still preserves backward compatibility.\n"
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2
    assert "Missing a regression test." in claude_calls[1][-1]
    assert len(runner.comments) == 5
    assert runner.comments[2].startswith("## Revised plan")


def test_issue_loop_plan_revision_repair_preserves_signed_human_requirements(tmp_path):
    requirement = HumanReviewRequirement(
        source_type="Issue body",
        author="maintainer",
        created_at="2026-05-17T08:00:00Z",
        url="https://github.com/OWNER/REPO/issues/56",
        body="Preserve backward compatibility.",
    )
    malformed_revision = (
        "### Prior plan review item dispositions\n"
        "- item-1: resolved by adding compatibility tests.\n\n"
        "### Revised plan\n"
        "- Preserve backward compatibility.\n"
        "- Add regression tests.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_revision = _add_default_requirement_disposition(structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        prior_plan_item_dispositions=[
            {
                "item_id": "item-1",
                "disposition": "resolved",
                "note": "Added compatibility tests.",
            }
        ],
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
        human_requirements=(
            f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: the revised plan preserves backward compatibility.\n"
        ),
    ), requirement_id=requirement.requirement_id)
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            ),
            structured_plan_review(
                summary="Plan looks sound.",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_repairs = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_repairs.append((raw, expected_kind))
        return repaired_revision

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_attempt_repair):
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert len(captured_repairs) == 1
    assert captured_repairs[0][1] == "plan_revision"
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in captured_repairs[0][0]
    public_revision = _strip_round_metadata(runner.comments[2])
    assert '"kind": "plan_revision"' not in public_revision
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in public_revision
    assert "### Human requirements" in public_revision


def test_issue_loop_plan_revision_repair_rejects_wrong_kind_from_human_requirements_text(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    wrong_kind_repair = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Revised the plan.",
                "addressed_items": [],
                "remaining_items": [],
                "human_requirement_dispositions": [],
                "human_requirements": {
                    "addressed_ids": ["Requirement 1"],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)
    captured_kinds = []

    def fake_attempt_repair(raw: str, gemini_cmd: str, *, expected_kind: str | None = None, unresolved_item_ids=None, surfaced_requirement_ids=None, requires_direct_discussion_ack=False, allowed_prior_item_ids=None, unknown_prior_item_ids=None, same_round_context=None) -> str | None:
        captured_kinds.append(expected_kind)
        return wrong_kind_repair

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_attempt_repair):
        with pytest.raises(AgentLoopError, match="expected `plan_revision`"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert captured_kinds == ["plan_revision"]


def test_issue_loop_plan_revision_repair_without_human_ack_fails_clearly(tmp_path):
    malformed_revision = (
        "### Revised plan\n"
        "- Preserve backward compatibility.\n\n"
        f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
        "### Human requirements\n"
        "- Requirement 1: the revised plan preserves backward compatibility.\n\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    )
    repaired_without_ack = structured_plan_revision(
        summary="Revised plan with compatibility tests.",
        plan_steps=["Preserve backward compatibility.", "Add regression tests."],
    )
    runner = FakeRunner(
        issue_payload={
            "author": {"login": "maintainer"},
            "createdAt": "2026-05-17T08:00:00Z",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(human_requirements=(
                f"\n{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
                "### Human requirements\n"
                "- Requirement 1: the plan preserves backward compatibility."
            )),
            malformed_revision,
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                summary="Missing a regression test.",
                blocking_plan_issues=["Missing a regression test."],
            )
        ],
    )
    config = make_config(tmp_path, agent_max_retries=0)

    with patch("coding_review_agent_loop.validated_agent.attempt_repair", return_value=repaired_without_ack):
        with pytest.raises(AgentLoopError, match="missing required signed human requirements marker"):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_requires_reviewers_to_disposition_prior_items(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n- Add parser validation tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs the test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, max_rounds=2)

    with pytest.raises(AgentLoopError, match="did not evaluate all prior unresolved plan items"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)


def test_issue_loop_plan_first_carries_same_plan_item_across_reviewers_and_rounds(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Second revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Still needs one plan refinement."
            + prior_plan_item_dispositions("[item-1] same-plan: still need the mixed-reviewer case")
            + "\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Plan looks sound now."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
            "Final pass."
            + prior_plan_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini",
        ],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert any("item-1" in call[-1] for call in claude_calls[1:])
    assert "Approved plan:" in runner.comments[-1]


def test_issue_loop_plan_first_posts_human_readable_item_labels_in_new_and_prior_sections(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Blocking plan issues\n"
            "- Keep plan-review wording distinct from PR wording.\n"
            "### Same-plan follow-ups\n"
            "- Add one carry-forward plan test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions(
                "[item-1] resolved",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=2)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.comments[1] == (
        "**Review verdict:** Blocking\n\n"
        "### Blocking plan issues\n"
        "- Keep plan-review wording distinct from PR wording.\n"
        "\n"
        "### Same-plan follow-ups\n"
        "- Add one carry-forward plan test.\n"
        "<!-- AGENT_PLAN_STATE: blocking -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )
    assert runner.comments[3] == (
        "**Review verdict:** Approved\n\n"
        "Plan looks sound.\n\n"
        "### Prior unresolved plan item dispositions\n"
        "- [item-1] RESOLVED\n"
        "  - Original finding: Blocking issue from OpenAI Codex, round 1: Keep plan-review wording distinct from PR wording.\n"
        "- [item-2] RESOLVED\n"
        "  - Original finding: Same-plan follow-up from OpenAI Codex, round 1: Add one carry-forward plan test.\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n"
        "-- OpenAI Codex: unknown model (medium)"
    )


def test_issue_loop_plan_first_does_not_expose_same_round_item_ids_to_later_reviewers(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
        ],
        gemini_outputs=[
            "### Same-plan follow-ups\n"
            "- Add the carry-forward orchestration test.\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
        ],
        claude_outputs=[
            "Still blocked.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
    )
    config = make_config(
        tmp_path,
        coder="codex",
        reviewer=("gemini", "claude"),
        max_rounds=1,
    )

    with pytest.raises(AgentLoopError, match="still reported blocking plan issues after round 1"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    second_reviewer_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "planning round 1" in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved plan items from earlier rounds`" in second_reviewer_prompt
    assert "[item-1]" not in second_reviewer_prompt
    assert "### New tracked unresolved items" not in runner.comments[1]


def test_issue_loop_plan_first_uses_compact_context_after_round_one(tmp_path, capsys):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Resolve item one.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round one."],
            ),
            structured_plan_revision(
                summary="Resolve item two.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
                plan_steps=["Revised plan after round two."],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round one issue."],
            ),
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Round two issue."],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Covered."}
                ],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Covered."}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=3,
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    round_one_review = next(prompt for prompt in prompts if "planning round 1" in prompt)
    round_two_review = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: reviewer" in prompt)
    round_two_revision = next(prompt for prompt in prompts if "Planning round: 2" in prompt and "Role: coder" in prompt)
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in round_one_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_review
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER in round_two_revision

    captured = capsys.readouterr()
    assert "Planning issue #56: invoking Claude (context mode: full)" in captured.err
    assert "Planning round 2: Codex reviewing issue #56 (context mode: compact)" in captured.err
    assert "Planning round 2: Claude revising the plan (context mode: compact)" in captured.err


def test_issue_loop_plan_first_requires_reviewer_human_requirements_resolution(tmp_path, capsys):
    runner = FakeRunner(
        issue_payload={
            "body": "Keep compact context cache-aware.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _initial_plan_state(
                summary="Initial plan covers cache-aware compact context.",
                human_requirements=(
                    "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
                    "### Human requirements\n"
                    "- Requirement 1: The plan keeps compact context cache-aware."
                ),
            ),
            structured_plan_revision(
                summary="Revised plan requires explicit reviewer acknowledgement.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Reviewer must acknowledge."}
                ],
                plan_steps=["Keep the compact context cache-aware and require reviewer acknowledgement."],
                human_requirements=(
                    "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
                    "### Human requirements\n"
                    "- Requirement 1: The revised plan covers the cache-aware compact context requirement."
                ),
            ),
        ],
        codex_outputs=[
            structured_plan_review(state="approved"),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Acknowledged."}
                ],
                human_requirements_resolved=True,
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=2,
        plan_execution_mode="plan-only",
        quiet=False,
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert "Approved plan:" in runner.comments[-1]
    assert any(
        "approved without acknowledging the signed human requirements" in comment
        for comment in runner.comments
    )
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in runner.comments[-2]
    captured = capsys.readouterr()
    assert "approved without acknowledging signed human requirements" in captured.err


def test_plan_first_grafana_requirement_blocks_admin_html_only_plan_then_accepts_integration(tmp_path):
    """A reviewer must reject a lookalike local UI until the named integration is planned."""
    runner = FakeRunner(
        issue_payload={
            "body": "Provision a Grafana dashboard for verification attribution.\n\n-- Human Reviewer",
        },
        claude_outputs=[
            _plan_with_requirement_disposition(
                kind="plan_state",
                summary="Add verification attribution to the operations UI.",
                plan_steps=["Add verification attribution to admin.html."],
                evidence="admin.html will display verification attribution.",
            ),
            _plan_with_requirement_disposition(
                kind="plan_revision",
                summary="Provision the requested Grafana dashboard.",
                plan_steps=[
                    "Add Grafana dashboard provisioning for verification attribution.",
                    "Test the dashboard provisioning artifact.",
                ],
                evidence="The plan names Grafana dashboard provisioning and its artifact.",
            ),
        ],
        codex_outputs=[
            _plan_review_with_requirement_disposition(
                state="blocking",
                summary="admin.html does not cover the requested Grafana dashboard provisioning.",
            ),
            _plan_review_with_requirement_disposition(
                state="approved",
                summary="The revised canonical plan covers the Grafana provisioning artifact.",
                marker=True,
                prior_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Grafana is now planned."}
                ],
            ),
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        max_rounds=2,
        plan_execution_mode="plan-only",
    )

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    review_prompts = [
        command[-1]
        for command, _cwd in runner.commands
        if command[:2] == ["codex", "exec"]
    ]
    assert any("admin.html" in prompt and "Grafana" in prompt for prompt in review_prompts)
    assert "admin.html does not cover the requested Grafana" in runner.comments[1]
    assert "Provision the requested Grafana dashboard" in runner.comments[-1]


def test_issue_loop_plan_first_uses_full_context_when_plan_ledger_incomplete(tmp_path, capsys):
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item that could be missed.",
        status="blocking",
        source_status="blocking",
    )
    old_reviewer_comment = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            blocking_plan_issues=["Old subject item that could be missed."],
        ),
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject=_plan_subject(old_plan),
            new_items=(old_item,),
            state="blocking",
        ),
    )
    latest_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(new_plan),
            prior_items=(),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": old_reviewer_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:10:00Z", "body": latest_coder_comment},
        ],
        codex_outputs=[structured_plan_review(state="approved")],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        quiet=False,
    )

    # item-1 was published and never resolved, yet the resumed coder record
    # carries nothing: the ledger is unreconstructible, so the approval
    # boundary fails closed after the full-context review (#1377).
    with pytest.raises(AgentLoopError, match="obligation\\(s\\) item-1 published earlier"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and "planning round 2" in cmd[-1]
    ][0]
    assert COMPACT_PLANNING_VOLATILE_TAIL_MARKER not in review_prompt
    captured = capsys.readouterr()
    assert "Planning round 2: Codex reviewing issue #56 (context mode: full (ledger incomplete))" in captured.err
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


def test_issue_loop_plan_review_strips_resolved_history_disposition_under_incomplete_ledger(
    tmp_path, monkeypatch
):
    # #862 plan-flow analog: an earlier plan subject raised and resolved
    # item-1, the carried set is empty, and the reviewer repeats item-1 as
    # resolved.  The no-op entry is stripped instead of failing the run.
    # This is the compatibility-default policy, so the resolved-only history
    # must still read as an incomplete ledger and still select full context:
    # staged planning's durable cleared-item proof does not apply here (#905).
    old_plan = "Old plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    mid_plan = "Mid plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    new_plan = "New plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    old_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Old subject item.",
        status="blocking",
        source_status="blocking",
    )
    raised = _attach_round_metadata(
        structured_plan_review(state="blocking", blocking_plan_issues=["Old subject item."]),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=1,
            subject=_plan_subject(old_plan), new_items=(old_item,), state="blocking",
        ),
    )
    resolved = _attach_round_metadata(
        structured_plan_review(
            state="blocking",
            prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
        ),
        PostedRoundMetadata(
            flow="plan", role="reviewer", agent="Codex", round_number=2,
            subject=_plan_subject(mid_plan), prior_items=(old_item,),
            dispositions=(ReviewItemDisposition("item-1", "OpenAI Codex", "resolved"),),
            state="blocking",
        ),
    )
    latest_coder_comment = _attach_round_metadata(
        new_plan,
        PostedRoundMetadata(
            flow="plan", role="coder", agent="Claude", round_number=3,
            subject=_plan_subject(new_plan), prior_items=(),
        ),
    )
    ledger_flags = []
    real_ledger_check = review_rounds._round_ledger_may_be_incomplete

    def ledger_spy(**kwargs):
        result = real_ledger_check(**kwargs)
        ledger_flags.append(result)
        return result

    patch_attr_in(monkeypatch, (review_rounds, pr_loop, plan_first_loop_module), "_round_ledger_may_be_incomplete", ledger_spy)
    logged = []
    real_log = agent_failure.log
    patch_attr_in(monkeypatch, (agent_failure, integration_close, architecture_contract, validated_agent_module, response_validation, panel_evidence, review_rounds, discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "log",
        lambda config, message, *a, **k: (logged.append(message), real_log(config, message, *a, **k))[1],
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": raised},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": resolved},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:10:00Z", "body": latest_coder_comment},
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            )
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer=("codex",))

    with patch("coding_review_agent_loop.validated_agent.attempt_repair") as repair_mock:
        assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    repair_mock.assert_not_called()
    assert ledger_flags and ledger_flags[0] is True
    assert any(
        "removed canonically resolved historical prior-item disposition ID(s) item-1 "
        "despite incomplete ledger" in message
        for message in logged
    )
    # The default path still takes the conservative full-context branch it
    # took before staged planning existed.
    assert any("(context mode: full (ledger incomplete))" in message for message in logged)


def test_issue_loop_plan_revision_proof_covers_items_resolved_in_the_same_round(
    tmp_path, monkeypatch
):
    # #874 orchestration regression: the initial issue context lacks the review
    # record that resolves item-2, because that comment is posted during the
    # round. The revision turn issued at the end of that round must still prove
    # item-2 is resolved history.
    revision_history_ids = []
    revision_snapshot_sizes = []
    snapshots = []
    real_run_validated_agent = validated_agent_module._run_validated_agent

    def run_validated_agent_spy(*args, **kwargs):
        if kwargs.get("operation_description") == "plan revision":
            revision_history_ids.append(
                tuple(kwargs.get("repair_resolved_history_item_ids") or ())
            )
            revision_snapshot_sizes.append(
                tuple(len(context.comments) for context in snapshots)
            )
        return real_run_validated_agent(*args, **kwargs)

    patch_attr_in(monkeypatch, (validated_agent_module, discuss_loop, pr_loop, issue_implementation, plan_first_loop_module, triage, issue_loop), "_run_validated_agent", run_validated_agent_spy
    )
    real_get_issue_context = discuss_loop.get_issue_context

    def get_issue_context_spy(*args, **kwargs):
        context = real_get_issue_context(*args, **kwargs)
        snapshots.append(context)
        return context

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", get_issue_context_spy
    )
    runner = FakeRunner(
        claude_outputs=[
            structured_plan_state(
                summary="Initial plan.", plan_steps=["Document migration ordering."]
            ),
            structured_plan_revision(
                summary="Addressed both findings.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_plan_revision(
                summary="Kept the still-blocking finding open.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=[
                    "Document migration ordering.",
                    "Cover the rollback path.",
                ],
            ),
            structured_plan_review(
                state="blocking",
                summary="Rollback path is covered; migration ordering still is not.",
                prior_plan_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "blocking",
                        "note": "Migration ordering is still unspecified.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            ),
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    # The pre-loop fetch was empty.  Later reads are the board-history
    # revalidation reads (#1377), taken once each round's reviews are in and at
    # each later round; they validate the lineage and never feed the proof.
    assert all(sizes[0] == 0 for sizes in revision_snapshot_sizes)
    assert revision_history_ids[0] == ()
    assert revision_history_ids[1] == ("item-2",)


def test_issue_loop_plan_first_resumes_with_only_missing_reviewer_for_current_plan(tmp_path):
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=_plan_subject(current_plan),
            prior_items=(),
        ),
    )
    codex_comment = _attach_round_metadata(
        "Plan looks sound.\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        PostedRoundMetadata(
            flow="plan",
            role="reviewer",
            agent="Codex",
            round_number=2,
            subject=_plan_subject(current_plan),
            state="approved",
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": coder_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": codex_comment},
        ],
        gemini_outputs=["Plan looks sound too.\n<!-- AGENT_PLAN_STATE: approved -->\n-- Google Gemini"],
    )
    config = make_config(tmp_path, reviewer=("codex", "gemini"))

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"], ["gemini"])]
    assert agent_commands == ["gemini"]
    assert runner.comments[-1].startswith("Planning complete for issue #56.")


@pytest.mark.parametrize(
    "persisted_as_machine", ["legacy", "promoted", "promoted-with-reviewer-notes"]
)
def test_issue_loop_plan_first_resume_clears_orchestrator_item_on_unanimous_approval(
    tmp_path, persisted_as_machine
):
    """An orchestrator-authored plan item stays reviewer-clearable after resume (#1005).

    Recovery used to promote it to an ``unknown`` machine obligation, which no
    planning participant can clear, so every unanimous approval was followed by
    another revision forever.  A ledger that already persisted the promoted
    form must be recoverable too.
    """
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    orchestrator_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Orchestrator",
        source_round=1,
        text="Reviewer(s) Codex approved without acknowledging the signed human requirements.",
        status="blocking",
        source_status="blocking",
    )
    if persisted_as_machine != "legacy":
        notes = ("Synthetic machine record lacked trusted orchestrator lineage.",)
        if persisted_as_machine == "promoted-with-reviewer-notes":
            # Evidence appended by earlier approval rounds that could not clear it.
            notes = (*notes, "Codex: Acknowledged in the revised plan.")
        orchestrator_item = replace(
            orchestrator_item,
            authority="unknown",
            obligation_kind="unknown",
            lifecycle="repair_required",
            obligation_identity="unknown:item-1",
            notes=notes,
        )
    summary_comment = _attach_round_metadata(
        "Orchestrator plan review.",
        PostedRoundMetadata(
            flow="plan",
            role="summary",
            agent="Orchestrator",
            round_number=1,
            subject=subject,
            new_items=(orchestrator_item,),
        ),
    )
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(orchestrator_item,),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:00:00Z", "body": summary_comment},
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
        ],
        codex_outputs=[
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved: acknowledged in the revised plan")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=3)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    agent_commands = [cmd[0] for cmd, _cwd in runner.commands if cmd[:1] in (["claude"], ["codex"])]
    assert agent_commands == ["codex"]
    assert runner.comments[-1].startswith("Planning complete for issue #56.")


def test_issue_loop_plan_first_stops_when_only_unclearable_items_block_approval(tmp_path):
    """A malformed plan machine record fails closed instead of revising forever (#1005).

    Recovery restores only the recognized legacy promotion; any other machine
    record (here the decoder's invalid-record blocker) must neither be cleared
    by a reviewer approval nor drive another revision.
    """
    current_plan = "Revised plan.\n- Add state reconstruction.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    subject = _plan_subject(current_plan)
    machine_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Orchestrator",
        source_round=1,
        text="Synthetic orchestrator blocker.",
        status="blocking",
        source_status="blocking",
        authority="unknown",
        obligation_kind="unknown",
        lifecycle="repair_required",
        obligation_identity="invalid-machine-record",
        notes=("Invalid persisted machine record: corrupted.",),
    )
    coder_comment = _attach_round_metadata(
        current_plan,
        PostedRoundMetadata(
            flow="plan",
            role="coder",
            agent="Claude",
            round_number=2,
            subject=subject,
            prior_items=(machine_item,),
        ),
    )
    runner = FakeRunner(
        issue_comments=[
            {"author": {"login": "bot"}, "createdAt": "2026-05-20T09:05:00Z", "body": coder_comment},
        ],
        codex_outputs=[
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved: nothing remains")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex"
        ],
    )
    config = make_config(tmp_path, reviewer=("codex",), max_rounds=3)

    with pytest.raises(AgentLoopError, match=r"item-1 \(owner Orchestrator, kind unknown\)"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    assert not [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]


@pytest.mark.parametrize(
    "line",
    [
        "[item-1] same-plan: none",
        "[item-1] still blocking: none",
        "[item-1] future follow-up: none",
    ],
)
def test_issue_loop_plan_first_rejects_contradictory_disposition_before_extra_revision(
    tmp_path, line
):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Add the carry-forward orchestration test.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Plan looks sound now."
            + prior_plan_item_dispositions(line)
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, reviewer="codex", max_rounds=3)

    with pytest.raises(AgentLoopError, match="use `resolved` when nothing remains"):
        run_issue_loop(runner, issue_number=56, config=config, plan_first=True)

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert len(claude_calls) == 2


def test_issue_loop_plan_first_plan_only_does_not_publish_approved_future_followups(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "### Same-plan follow-ups\n- Tighten the prompt wording.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- OpenAI Codex",
            "Looks good."
            + prior_plan_item_dispositions("[item-1] future follow-up: document parser helper reuse separately")
            + "\n### Future follow-ups\n- Add a later cleanup to dedupe shared prompt rendering.\n"
            + "<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    assert runner.issues == []
    summary = runner.comments[-1]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Approved plan future follow-ups:" in summary
    assert "document parser helper reuse separately" in summary
    assert "Add a later cleanup to dedupe shared prompt rendering." in summary
    assert "not carried into PR review" in summary
    assert "not PR prior review items" in summary
    assert "Filed future follow-up issues:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary
    assert "mode=summarize" in summary


def test_issue_loop_plan_first_files_approved_future_followups_before_implementation(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert len(runner.issues) == 1
    assert runner.issues[0]["title"] == (
        "Follow up future plan-review note: Add a later cleanup to dedupe shared prompt rendering."
    )
    issue_body = runner.issues[0]["body"]
    assert "Parent issue: #56" in issue_body
    assert "Approved plan hash:" in issue_body
    assert "Planning round(s): 1" in issue_body
    assert "Original plan item ID(s): item-1" in issue_body
    assert "Codex" in issue_body
    assert "outside the current implementation scope" in issue_body
    assert "not a PR-review prior item" in issue_body
    summary = runner.comments[2]
    assert summary.startswith("Planning complete for issue #56.")
    assert "Filed future follow-up issues:" in summary
    assert "https://github.com/OWNER/REPO/issues/99" in summary
    assert "Approved plan future follow-ups:" not in summary
    assert "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS:" in summary

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    second_claude_index = command_index(
        runner.commands,
        ["claude", "--print"],
        start=command_index(runner.commands, ["claude", "--print"]) + 1,
    )
    assert issue_create_index < second_claude_index


def test_issue_loop_plan_first_does_not_allocate_new_items_for_repeated_carried_future_followups(tmp_path):
    future_text = "Factor the shared follow-up guidance into a reusable helper."
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Address the blocking test gap.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-2", "disposition": "resolved", "note": "Added the test."}
                ],
                plan_steps=["Make the change.", "Add the missing regression test."],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {
                        "item_id": "item-1",
                        "disposition": "future",
                        "note": "Keep this as confirmed post-plan cleanup.",
                    },
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add a regression test for the plan-review ledger."],
                reviewer="Google Gemini",
            ),
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "future"},
                    {"item_id": "item-2", "disposition": "resolved"},
                ],
                reviewer="Google Gemini",
            ),
            structured_pr_review(
                state="approved",
                summary="LGTM.",
                reviewer="Google Gemini",
            ),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=2,
    )

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    coder_revision_prompt = next(
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
    )
    assert "item-2" in coder_revision_prompt
    assert "item-1" not in coder_revision_prompt
    assert future_text not in coder_revision_prompt

    round_two_reviewer_prompts = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert len(round_two_reviewer_prompts) == 2
    assert all("item-1" in prompt for prompt in round_two_reviewer_prompts)

    assert len(runner.issues) == 1
    issue_body = runner.issues[0]["body"]
    assert "Planning round(s): 1" in issue_body
    assert "Reviewer: Codex" in issue_body
    assert "Original plan item ID(s): item-1" in issue_body
    assert "Keep this as confirmed post-plan cleanup." in issue_body
    assert any(
        "Reconciliation: 1 filed, 0 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )

    issue_create_index = command_index(runner.commands, ["gh", "issue", "create"])
    round_two_review_indexes = [
        index
        for index, (cmd, _cwd) in enumerate(runner.commands)
        if "Planning round: 2" in cmd[-1] and "Role: reviewer" in cmd[-1]
    ]
    assert issue_create_index > max(round_two_review_indexes)


@pytest.mark.parametrize("later_disposition", ["resolved", "same-plan", "blocking"])
def test_issue_loop_plan_first_does_not_file_future_item_after_later_lifecycle_change(
    tmp_path, later_disposition
):
    future_text = "Extract the shared plan-review formatting helper."
    promoted = later_disposition in {"same-plan", "blocking"}
    promotion_state = "blocking" if promoted else "approved"
    final_plan_dispositions = [
        {"item_id": "item-1", "disposition": later_disposition},
        {"item_id": "item-2", "disposition": "resolved"},
    ]
    claude_outputs = [
        "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        structured_plan_revision(
            summary="Address the original blocker.",
            prior_plan_item_dispositions=[{"item_id": "item-2", "disposition": "resolved"}],
        ),
    ]
    if promoted:
        claude_outputs.append(
            structured_plan_revision(
                summary="Address the promoted future item.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"}
                ],
            )
        )
    claude_outputs.append(
        "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
        "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    def reviewer_outputs(reviewer):
        outputs = [
            structured_plan_review(
                state="approved",
                future_followups=[future_text],
                reviewer=reviewer,
            ),
            structured_plan_review(
                state=promotion_state,
                prior_plan_item_dispositions=final_plan_dispositions,
                reviewer=reviewer,
            ),
        ]
        if promoted:
            outputs.append(
                structured_plan_review(
                    state="approved",
                    prior_plan_item_dispositions=[
                        {"item_id": "item-1", "disposition": "resolved"}
                    ],
                    reviewer=reviewer,
                )
            )
        outputs.append(structured_pr_review(state="approved", reviewer=reviewer))
        return outputs

    runner = FakeRunner(
        claude_outputs=claude_outputs,
        codex_outputs=reviewer_outputs("OpenAI Codex"),
        gemini_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add the initial ledger regression."],
                reviewer="Google Gemini",
            ),
            *reviewer_outputs("Google Gemini")[1:],
        ],
    )
    config = make_config(
        tmp_path,
        reviewer=("codex", "gemini"),
        approved_followups="issue",
        max_rounds=3,
    )

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    assert not any(future_text in comment for comment in runner.comments[-2:])
    if promoted:
        coder_revision_prompts = [
            cmd[-1]
            for cmd, _cwd in runner.commands
            if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
        ]
        assert len(coder_revision_prompts) == 2
        assert "item-1" in coder_revision_prompts[1]
        assert future_text in coder_revision_prompts[1]
        final_round_review_indexes = [
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if "Planning round: 3" in cmd[-1] and "Role: reviewer" in cmd[-1]
        ]
        implementation_index = next(
            index
            for index, (cmd, _cwd) in enumerate(runner.commands)
            if cmd[:1] == ["claude"] and "Implement the approved plan" in cmd[-1]
        )
        assert implementation_index > max(final_round_review_indexes)


def test_issue_loop_plan_first_ignore_mode_keeps_pr_prior_ledger_clean(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=["Track a separate planning cleanup later."],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
    )
    config = make_config(tmp_path, approved_followups="ignore")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    planning_summary = runner.comments[2]
    assert "Approved plan future follow-ups:" in planning_summary
    assert "Track a separate planning cleanup later." in planning_summary
    assert "not carried into PR review" in planning_summary
    pr_review_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:2] == ["codex", "exec"] and '"kind": "pr_review"' in cmd[-1]
    ][0]
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in pr_review_prompt
    ledger_start = pr_review_prompt.index("Prior unresolved review items from earlier rounds")
    assert "Track a separate planning cleanup later." not in pr_review_prompt[ledger_start:]
    assert "planning-stage `item-*` IDs and approved\nplan future follow-ups" in pr_review_prompt
    assert "prior_plan_item_dispositions" in pr_review_prompt


def test_issue_loop_plan_first_deduplicates_plan_followup_issues_across_reviewers(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Google Gemini",
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini",
        ],
        codex_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        claude_outputs=[
            structured_plan_review(
                state="approved",
                future_followups=[
                    "**Remote validation**: Validate explicit workdir git remotes against the target repo.",
                ],
                reviewer="Anthropic Claude",
            ),
            structured_pr_review(state="approved", summary="LGTM.", reviewer="Anthropic Claude"),
        ],
        issue_urls=["https://github.com/OWNER/REPO/issues/99"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="issue",
    )

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert len(runner.issues) == 1
    body = runner.issues[0]["body"]
    assert "Reviewers: Codex, Claude" in body
    assert "Original plan item ID(s): item-1, item-2" in body
    assert body.count("**Remote validation**") == 3
    assert any(
        "Reconciliation: 1 filed, 1 deduplicated, 0 skipped by cap." in comment
        for comment in runner.comments
    )


def test_issue_loop_plan_first_plan_followup_marker_prevents_duplicate_issue_creation(tmp_path):
    plan = "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude"
    plan_hash = approved_plan_hash(plan)
    future_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="Codex",
        source_round=1,
        text="Add a later cleanup to dedupe shared prompt rendering.",
        status="future",
        source_status="future",
    )
    runner = FakeRunner(
        claude_outputs=[
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_pr_review(state="approved", summary="LGTM."),
        ],
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
                    structured_plan_review(
                        state="approved",
                        future_followups=["Add a later cleanup to dedupe shared prompt rendering."],
                    ),
                    PostedRoundMetadata(
                        flow="plan",
                        role="reviewer",
                        agent="Codex",
                        round_number=1,
                        subject=_plan_subject(plan),
                        new_items=(future_item,),
                        state="approved",
                    ),
                ),
            },
            {
                "author": {"login": "bot"},
                "createdAt": "2026-05-23T00:00:02Z",
                "body": (
                    "Planning complete for issue #56.\n\n"
                    "<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: "
                    f"issue=56 plan={plan_hash} mode=issue -->\n"
                    "-- coding-review-agent-loop"
                ),
            },
        ],
    )
    config = make_config(tmp_path, approved_followups="issue")

    assert (
        run_issue_loop(
            runner,
            issue_number=56,
            config=config,
            plan_first=True,
            implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    assert not any(
        "Filed future follow-up issues:" in comment or "Approved plan future follow-ups:" in comment
        for comment in runner.comments
    )


def test_issue_loop_plan_first_keeps_blocking_review_when_future_followups_are_misclassified(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            "Initial plan.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            "Revised plan with focused tests.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Still blocked.\n\n"
            "### Blocking plan issues\n"
            "- Add parser coverage for blocking reviews with stray future follow-ups.\n\n"
            "### Same-plan follow-ups\n"
            "- Tighten the plan-review prompt wording.\n\n"
            "### Future follow-ups\n"
            "- Consider a later prompt dedupe cleanup.\n\n"
            "<!-- AGENT_PLAN_STATE: blocking -->\n"
            "-- OpenAI Codex",
            "Plan looks sound."
            + prior_plan_item_dispositions("[item-1] resolved", "[item-2] resolved")
            + "\n<!-- AGENT_PLAN_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config, plan_first=True) == 0

    claude_calls = [cmd for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    assert "Add parser coverage for blocking reviews with stray future follow-ups." in claude_calls[1][-1]
    assert "Tighten the plan-review prompt wording." in claude_calls[1][-1]
    assert runner.comments[1].startswith("**Review verdict:** Blocking\n\nStill blocked.")
    assert "### Future follow-ups" not in runner.comments[1]


@pytest.mark.parametrize(
    "event", ["forged-link", "signed-amendment", "requirement-added", "requirement-withdrawn"],
)
def test_issue_changes_during_acknowledgement_repair_never_reach_approval(
    tmp_path, monkeypatch, event
):
    """#1377 items 16/17: the acknowledgement repair is an agent turn, so the
    run re-reads the issue after it.  Board history changed during the repair
    refuses before the repaired record is published, and a requirement change
    during it refuses at the approval boundary; neither approves nor hands off."""
    import datetime

    from coding_review_agent_loop.board_amendment import format_reviewer_board_amendment_comment
    from coding_review_agent_loop.board_change import (
        PLAN_BOARD_CHANGE_LINK_HEADING,
        PLAN_BOARD_CHANGE_LINK_MARKER,
    )

    requirement, plan_output, unacknowledged, repaired, panel = (
        _repaired_acknowledgement_fixtures()
    )
    runner = _FakeRunner(
        claude_outputs=[plan_output], codex_outputs=[unacknowledged], gemini_outputs=[panel],
        serve_rest_issue_comments=True, authenticated_actor=("agent-loop", 4242),
    )
    requirements = [requirement]
    real_get_issue_context = discuss_loop.get_issue_context

    def _patched(runner_arg, *, config, issue_number):
        context = real_get_issue_context(runner_arg, config=config, issue_number=issue_number)
        return replace(context, human_requirements=tuple(requirements))

    patch_attr_in(monkeypatch, (discuss_loop, execution_policy, child_plan_binding, pr_loop_support, pr_loop, plan_first_loop_module, issue_loop), "get_issue_context", _patched)

    def operator_comment(body):
        runner.issue_comments.append({
            "body": body, "author": {"login": "operator"}, "_rest_author_id": 81,
            "createdAt": datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
        })

    def fake_repair(raw, cmd, **kwargs):
        if event == "forged-link":
            operator_comment(
                f"{PLAN_BOARD_CHANGE_LINK_HEADING}\n<!-- {PLAN_BOARD_CHANGE_LINK_MARKER}: e30= -->"
            )
        elif event == "signed-amendment":
            operator_comment(format_reviewer_board_amendment_comment(
                flow="plan", issue=56, pr_number=None,
                original_required_reviewers=("Codex", "Gemini"),
                policy="all-reviewers", primary_reviewer=None,
                removed_reviewers=("Gemini",), effective_from_round=1,
                rationale="Gemini quota exhausted.",
            ))
        elif event == "requirement-added":
            requirements.append(HumanReviewRequirement(
                source_type="Issue comment", author="maintainer",
                created_at="2026-05-17T09:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-9",
                body="Also keep the audit trail.",
            ))
        else:
            requirements.clear()
        return repaired

    refusal = {
        "forged-link": "history is inconsistent after an acknowledgement repair",
        "signed-amendment": "recorded while this run was active .*after an acknowledgement repair",
        "requirement-added": "requirement\\(s\\) changed after plan approval: added",
        "requirement-withdrawn": "requirement\\(s\\) changed after plan approval: withdrawn",
    }[event]
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), max_rounds=1,
        agent_max_retries=0, agent_retry_backoff_seconds=0,
    )
    with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair):
        with pytest.raises(AgentLoopError, match=refusal):
            run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
    codex_records = [
        record for record in _plan_round_records(runner)
        if record.role == "reviewer" and record.agent == "Codex"
    ]
    if event in {"forged-link", "signed-amendment"}:
        # The repaired result is never published under the old board history.
        assert len(codex_records) == 1
        assert codex_records[0].surfaced_reviewer_requirement_ids == ()
    assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)
    # Only the initial planner turn ran: no revision and no implementation coder.
    assert sum(1 for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]) == 1
    if event == "signed-amendment":
        # A clean rerun resolves and activates the amendment: the board becomes
        # Codex alone and Gemini is never invoked again.  (The resumed Codex
        # record is rendered prose, which the existing substance gate never
        # repairs, so this one-round rerun then stops at its round budget.)
        event = "none"
        start_commands, start_comments = len(runner.commands), len(runner.issue_comments)
        with patch("coding_review_agent_loop.validated_agent.attempt_repair", fake_repair):
            with pytest.raises(AgentLoopError):
                run_issue_loop(runner, issue_number=56, config=config, plan_first=True)
        assert not any(
            cmd[:1] == ["gemini"] for cmd, _cwd in runner.commands[start_commands:]
        )
        (audit,) = [c["body"] for c in runner.issue_comments[start_comments:]]
        assert audit.startswith("Reviewer board amendment applied.")
        assert "Required board now: Codex" in audit
        assert not any(cmd[:3] == ["gh", "pr", "create"] for cmd, _cwd in runner.commands)


# --- Future follow-ups on blocking plan reviews (#1389) ----------------------


def _blocking_plan_future_runner_1389(future_text, **runner_kwargs):
    return FakeRunner(
        claude_outputs=[
            "Plan:\n- Make the change.\n<!-- AGENT_PLAN_STATE: blocking -->\n-- Anthropic Claude",
            structured_plan_revision(
                summary="Added the missing regression test.",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved", "note": "Added the test."}
                ],
                plan_steps=["Make the change.", "Add the missing regression test."],
            ),
            "Implemented approved plan.\n<!-- AGENT_PR: 77 -->\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            structured_plan_review(
                state="blocking",
                blocking_plan_issues=["Add a regression test for the parser change."],
                future_followups=[future_text],
            ),
            structured_plan_review(
                state="approved",
                prior_plan_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": "item-2", "disposition": "future", "note": "Still off-scope."},
                ],
            ),
            structured_pr_review(state="approved", summary="LGTM."),
        ],
        **runner_kwargs,
    )


def _plan_revision_prompts_1389(runner):
    return [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and '"kind": "plan_revision"' in cmd[-1]
    ]


def test_blocking_plan_review_future_followup_is_filed_at_plan_approval(tmp_path):
    future_text = "Add cross-process locking for the plan cache."
    runner = _blocking_plan_future_runner_1389(
        future_text, issue_urls=["https://github.com/OWNER/REPO/issues/99"]
    )
    config = make_config(tmp_path, approved_followups="issue", max_rounds=2)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True,
        )
        == 0
    )

    assert "### Future follow-ups\n- " + future_text in runner.comments[1]
    assert [issue["title"] for issue in runner.issues] == [
        f"Follow up future plan-review note: {future_text}"
    ]
    assert "Original plan item ID(s): item-2" in runner.issues[0]["body"]
    revision_prompts = _plan_revision_prompts_1389(runner)
    assert len(revision_prompts) == 1
    # Only the blocking item is revision-required work; the future item is not
    # in the carried plan-item ledger the planner must disposition.
    assert "item-1" in revision_prompts[0]
    assert "item-2" not in revision_prompts[0]


def test_blocking_plan_review_future_followup_is_summarized_under_ignore(tmp_path):
    future_text = "Add cross-process locking for the plan cache."
    runner = _blocking_plan_future_runner_1389(future_text)
    config = make_config(tmp_path, approved_followups="ignore", max_rounds=2)

    assert (
        run_issue_loop(
            runner, issue_number=56, config=config, plan_first=True, implement_after_approval=True,
        )
        == 0
    )

    assert runner.issues == []
    summary = next(
        comment for comment in runner.comments if "Approved plan future follow-ups:" in comment
    )
    assert future_text in summary
    revision_prompts = _plan_revision_prompts_1389(runner)
    assert len(revision_prompts) == 1
    assert "item-2" not in revision_prompts[0]
