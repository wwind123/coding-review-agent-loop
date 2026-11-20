"""PR-loop tests for unresolved_items and pr_loop: PR human requirements (#1542)."""

import coding_review_agent_loop.issue_recovery as issue_recovery
import coding_review_agent_loop.plan_execution as plan_execution
import json
import re
from dataclasses import replace
import dataclasses
from coding_review_agent_loop.errors import HumanDecisionRequiredError
from pathlib import Path

import pytest
from coding_review_agent_loop.cli import AgentLoopError, run_pr_loop
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestMetadata,
    PullRequestReviewContext,
)
from coding_review_agent_loop.unresolved_items import (
    HUMAN_REQUIREMENTS_ACK_ITEM_ID,
    _reconcile_human_requirements_ack_item,
)
from coding_review_agent_loop.round_metadata import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _strip_round_metadata,
)
from coding_review_agent_loop.protocol_common import (
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
)
from coding_review_agent_loop.protocol import (
    UnresolvedReviewItem,
)
from agent_loop_helpers import (
    FakeRunner,
    blocking_issues,
    make_config,
    prior_item_dispositions,
    structured_coder_followup,
    structured_pr_review,
)
import coding_review_agent_loop.agent_failure as agent_failure
import coding_review_agent_loop.issue_implementation as issue_implementation
import coding_review_agent_loop.issue_loop as issue_loop
import coding_review_agent_loop.plan_first_loop as plan_first_loop
import coding_review_agent_loop.pr_loop as pr_loop
import coding_review_agent_loop.pr_loop_support as pr_loop_support
import coding_review_agent_loop.response_validation as response_validation
from agent_loop_helpers import patch_attr_in


def _code_requirement_review(payload=None, **kwargs):
    """Script a code classification independently from acknowledgement success."""
    from pr_loop_test_helpers import _with_code_classification
    return _with_code_classification(structured_pr_review(**kwargs),
        _parse_pr_human_requirements(payload or ack_pr_payload()))


def test_pr_loop_reinjects_blocking_item_when_human_requirement_marker_missing(tmp_path):
    # Reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → synthetic blocking item,
    # loop hits max_rounds (set to 1) instead of a terminal deadlock.
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    runner.codex_outputs[0] = _code_requirement_review(runner.pr_payload)
    config = make_config(
        tmp_path,
        auto_merge=True,
        test_command=("pytest", "tests/test_agent_loop.py"),
        approved_followups="summarize",
        max_rounds=1,
    )

    # The old behaviour was a terminal deadlock; now the loop continues and hits max_rounds.
    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] not in commands
    assert ["gh", "pr", "merge", "77", "--repo", "OWNER/REPO", "--merge"] not in commands
    assert not any(comment.startswith("Approved-review future follow-ups") for comment in runner.comments)

def test_pr_loop_recovers_when_second_reviewer_includes_human_requirement_marker(tmp_path):
    # Round 1: reviewer approves without HUMAN_REQUIREMENTS_RESOLVED → blocking item injected.
    # Round 2: coder addresses it; reviewer approves with the marker → success.
    pr_payload = {
        "number": 77,
        "state": "OPEN",
        "url": "https://github.com/OWNER/REPO/pull/77",
        "title": "Improve review prompt context",
        "headRefName": "feature/review-context",
        "baseRefName": "main",
        "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
        "comments": [
            {
                "author": {"login": "maintainer"},
                "createdAt": "2026-05-18T10:00:00Z",
                "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                "body": "Please use the absolute URL.\n\n-- Human Reviewer",
            }
        ],
        "reviews": [],
    }
    runner = FakeRunner(
        claude_outputs=[
            # Round 2: coder addresses the re-injected blocking item and acknowledges human requirements
            "Addressed human requirements.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: used the absolute URL.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            # Round 1: approves but forgets the marker
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            # Round 2: resolves the synthetic blocking item and acknowledges human requirements
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload=pr_payload,
    )
    runner.codex_outputs[0] = _code_requirement_review(runner.pr_payload)
    config = make_config(tmp_path, max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

def test_pr_loop_allows_approval_with_human_requirement_resolution_marker(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, test_command=("pytest", "tests/test_agent_loop.py"))

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    commands = [cmd for cmd, _cwd in runner.commands]
    assert ["pytest", "tests/test_agent_loop.py"] in commands

def test_pr_loop_accepts_structured_coder_followup_in_pr_round(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Added the requested regression test.",
                    "addressed_items": ["item-1"],
                    "remaining_items": [],
                    "human_requirement_dispositions": [],
                    "addressed_item_notes": {
                        "item-1": "Added the structured coder follow-up regression case."
                    },
                        "human_requirements": {
                            "addressed_ids": [],
                            "checked_discussion_directly": False,
                        },
                        "architecture_impact": {
                            "status": "unchanged",
                            "rationale": "No architectural contract changed.",
                            "affected_components": [],
                            "dependencies": [],
                            "execution_data_flows": [],
                            "persistence": [],
                            "public_contracts": [],
                            "security_boundaries": [],
                            "canonical_document_action": "no-change",
                            "canonical_document_path": None,
                            "canonical_document_rationale": "",
                        },
                        "tests_run": ["pytest tests/test_agent_loop.py -k structured_coder_followup"],
                }
            )
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_comments = [comment for comment in runner.comments if "## Coder follow-up" in comment]
    assert len(followup_comments) == 1
    visible_followup = _strip_round_metadata(followup_comments[0])
    assert "Added the requested regression test." in visible_followup
    assert "### Addressed items\n- item-1: Blocking issue from OpenAI Codex" in visible_followup
    assert "  - Resolution: Added the structured coder follow-up regression case." in visible_followup
    assert "### Remaining items\n- None." in visible_followup
    assert (
        "### Tests run\n- pytest tests/test_agent_loop.py -k structured_coder_followup"
        in visible_followup
    )
    assert '"kind": "coder_followup"' not in visible_followup

def test_pr_loop_rejects_malformed_structured_coder_followup_before_re_review(tmp_path):
    runner = FakeRunner(
        claude_outputs=[
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "coder_followup",
                    "state": "blocking",
                    "summary": "Tried to handle the feedback.",
                    "addressed_items": ["item-9"],
                    "remaining_items": [],
                    "human_requirement_dispositions": [],
                    "human_requirements": {
                        "addressed_ids": [],
                        "checked_discussion_directly": False,
                    },
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
            + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ],
        codex_outputs=[
            "Need one more regression test before merge."
            + blocking_issues("Add the structured coder follow-up regression case.")
            + "\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
        ],
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        agent_max_retries=0,
    )

    with pytest.raises(
        AgentLoopError,
        match="Coder follow-up referenced unknown unresolved reviewer item IDs: item-9",
    ):
        run_pr_loop(runner, pr_number=77, config=config)

def test_reconcile_human_requirements_ack_item_does_not_mint_blocker_for_generic_output():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output="Implemented fix without the extra acknowledgement.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []


def test_reconcile_human_requirements_ack_item_retains_dedicated_failure():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    invalid_structured = structured_coder_followup(
        addressed_items=[],
        human_requirement_ids=[],
        human_requirement_dispositions=[],
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output=invalid_structured,
        human_requirements=human_requirements,
        source_round=2,
    )

    assert [item.item_id for item in reconciled] == [HUMAN_REQUIREMENTS_ACK_ITEM_ID]
    assert "missing" in reconciled[0].text.lower()


def test_rejected_structured_response_content_cannot_trigger_provider_advice():
    malformed = structured_coder_followup(
        summary="Authentication, credit, billing, and dirty-tree vocabulary are only content.",
    )

    assert (
        agent_failure._failure_category(
            malformed,
            public_response=True,
            repair_expected_kind="coder_followup",
        )
        == "deterministic"
    )
    error = agent_failure._format_invalid_agent_response_error(
        agent_name="Anthropic Claude",
        marker_description="<!-- AGENT_STATE: approved|blocking -->",
        reason="schema validation failed; repair invocation failure: timeout",
        result=None,
        log_paths=(),
        category="deterministic",
        classification_text="structured coder_followup response failed trusted validation",
    )
    assert "credentials or billing" not in error

def test_reconcile_human_requirements_ack_item_clears_markdown_ack_blocker():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=(
            "Implemented follow-up.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            f"- Requirement {human_requirements[0].requirement_id}: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ),
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []

def test_pr_loop_revalidates_latest_coder_output_against_refreshed_human_requirements(
    tmp_path, monkeypatch
):
    runner = FakeRunner(
        claude_outputs=[
            "Implemented fix with the required acknowledgement.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: updated the URL handling.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
        ],
        codex_outputs=[
            "Blocking issue.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Improve review prompt context",
            "headRefName": "feature/review-context",
            "baseRefName": "main",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Please use the absolute URL.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
        advance_pr_head_on_coder_followup=False,
    )
    runner.codex_outputs[0] = _code_requirement_review(runner.pr_payload, state="blocking", blocking_items=["Blocking issue."])
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    contexts = iter(
        [
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(
                    HumanReviewRequirement(
                        source_type="PR comment",
                        author="maintainer",
                        created_at="2026-05-18T10:00:00Z",
                        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                        body="Please use the absolute URL.",
                    ),
                ),
            ),
            PullRequestReviewContext(
                metadata=metadata,
                comments=(),
                human_requirements=(),
            ),
        ]
    )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_implementation, issue_loop, issue_recovery, plan_execution,), "get_observed_pr_review_context",
        lambda *args, **kwargs: next(contexts),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    review_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(review_prompts) == 2
    assert HUMAN_REQUIREMENTS_ACK_ITEM_ID not in review_prompts[1]

def test_pr_loop_combines_issue_and_pr_signed_human_requirements(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "LGTM.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"
            ],
        issue_payload={
            "number": 56,
            "title": "Support issue comments",
            "body": "Preserve backward compatibility.\n\n-- Human Reviewer",
            "author": {"login": "issue-author"},
            "createdAt": "2026-05-17T08:00:00Z",
            "url": "https://github.com/OWNER/REPO/issues/56",
        },
        pr_payload={
            "comments": [
                {
                    "author": {"login": "maintainer"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                    "body": "Use the absolute URL in the PR path.\n\n-- Human Reviewer",
                }
            ],
            "reviews": [],
        },
    )
    config = make_config(tmp_path, reviewer="codex")
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue body",
                author="issue-author",
                created_at="2026-05-17T08:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Preserve backward compatibility.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Preserve backward compatibility." in prompt
    assert "Use the absolute URL in the PR path." in prompt
    assert prompt.index("Preserve backward compatibility.") < prompt.index(
        "Use the absolute URL in the PR path."
    )


@pytest.mark.parametrize(
    (
        "second_requirement_body",
        "second_requirement_created_at",
        "second_requirement_url",
        "expected_second_prompt_text",
    ),
    [
        (
            "Also preserve the reviewer attribution.",
            "2026-05-18T10:10:00Z",
            "https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            "stable-id",
        ),
        (
            "The edited instruction changes the required audit trail.",
            "2026-05-18T10:00:00Z",
            "https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            "The edited instruction changes the required audit trail.",
        ),
    ],
)
def test_pr_loop_rereviews_unchanged_head_when_human_requirement_changes(
    tmp_path,
    monkeypatch,
    second_requirement_body,
    second_requirement_created_at,
    second_requirement_url,
    expected_second_prompt_text,
):
    requirement_1 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at="2026-05-18T10:00:00Z",
        url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
        body="Keep the current audit trail.",
    )
    requirement_2 = HumanReviewRequirement(
        source_type="PR comment",
        author="maintainer",
        created_at=second_requirement_created_at,
        url=second_requirement_url,
        body=second_requirement_body,
    )
    if expected_second_prompt_text == "stable-id":
        expected_second_prompt_text = f"Requirement {requirement_2.requirement_id}"
    runner = FakeRunner(
        codex_outputs=[
            structured_pr_review(
                state="approved",
                summary="Codex approves the initial requirement.",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                state="approved",
                summary="Codex approves after reviewing the new requirement.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": HUMAN_REQUIREMENTS_ACK_ITEM_ID, "disposition": "resolved"},
                ],
                human_requirements_resolved=True,
            ),
        ],
        claude_outputs=[
            structured_pr_review(
                state="blocking",
                summary="Claude needs one fix.",
                blocking_items=["Fix the admission cleanup race."],
                reviewer="Anthropic Claude",
                human_requirements_resolved=True,
            ),
            structured_pr_review(
                state="approved",
                summary="Claude accepts the follow-up.",
                prior_item_dispositions=[
                    {"item_id": "item-1", "disposition": "resolved"},
                    {"item_id": HUMAN_REQUIREMENTS_ACK_ITEM_ID, "disposition": "resolved"},
                ],
                reviewer="Anthropic Claude",
                human_requirements_resolved=True,
            ),
        ],
        gemini_outputs=[
            structured_coder_followup(
                addressed_items=["item-1"],
                remaining_items=[],
                human_requirement_ids=["Requirement 1"],
                reviewer="Google Gemini",
            )
        ],
        advance_pr_head_on_coder_followup=False,
    )
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee",
        url="https://github.com/OWNER/REPO/pull/77",
    )
    prior_codex_review = _attach_round_metadata(
        structured_pr_review(
            state="approved",
            summary="Codex approves the initial requirement.",
            human_requirements_resolved=True,
            reviewer="OpenAI Codex",
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="OpenAI Codex",
            round_number=1,
            subject="6367c48dd193d56ea7b0baad25b19455e529f5ee",
            state="approved",
            surfaced_reviewer_requirement_ids=("Requirement 1",),
        ),
    )
    context_calls = 0

    def next_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
        if context_calls > 1 and not any(c.get('body') == str(prior_codex_review) for c in runner.pr_payload['comments']):
            runner.pr_payload['comments'].append({'author': {'login': 'coding-review-agent-loop'},
                'createdAt': '2026-05-18T11:00:00Z', 'body': str(prior_codex_review),
                '_rest_author_id': 424242})
        return PullRequestReviewContext(
            metadata=metadata,
            comments=(
                IssueComment(
                    author="coding-review-agent-loop",
                    created_at="2026-05-18T11:00:00Z",
                    body=str(prior_codex_review),
                ),
            )
            if context_calls > 1
            else (),
            human_requirements=(
                (requirement_2,)
                if second_requirement_url.endswith("issuecomment-1")
                else (requirement_1, requirement_2)
            )
            if context_calls > 1
            else (requirement_1,),
        )

    patch_attr_in(monkeypatch, (pr_loop_support, pr_loop, issue_implementation, issue_loop, issue_recovery, plan_execution,), "get_observed_pr_review_context",
        next_context,
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_reviews = [cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(codex_reviews) == 2
    assert expected_second_prompt_text in codex_reviews[1][-1]

def test_reconcile_human_requirements_ack_item_accepts_stored_structured_coder_followup():
    human_requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    structured_followup = (
        json.dumps(
            {
                "schema_version": 1,
                "kind": "coder_followup",
                "state": "blocking",
                "summary": "Implemented the requested URL fix.",
                "addressed_items": ["item-1"],
                "remaining_items": [],
                "human_requirement_dispositions": [
                    {
                        "requirement_id": human_requirements[0].requirement_id,
                        "disposition": "addressed",
                        "evidence": "The URL fix is implemented.",
                    }
                ],
                "human_requirements": {
                    "addressed_ids": [human_requirements[0].requirement_id],
                    "checked_discussion_directly": False,
                },
            }
        )
        + "\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (
            UnresolvedReviewItem(
                item_id=HUMAN_REQUIREMENTS_ACK_ITEM_ID,
                reviewer="Orchestrator",
                source_round=1,
                text="Ack missing.",
                status="blocking",
            ),
        ),
        coder_output=structured_followup,
        human_requirements=human_requirements,
        source_round=2,
    )

    assert reconciled == []


def test_trust_set_change_on_resume_validates_against_current_surfaced_set(tmp_path):
    """#1022: a requirement admitted by an unconfigured run and later excluded
    by a configured trust set is no longer required, and its historical
    acknowledgement does not crash the resumed validation."""
    from agent_loop_helpers import structured_coder_followup
    from coding_review_agent_loop.github import TrustedHumanActor, get_pr_review_context
    from coding_review_agent_loop.protocol import validate_structured_coder_followup
    from test_human_requirement_trust import (
        PR,
        SIGNATURE,
        RoutedRunner,
        _comments_path,
        _gql_comment,
        _pr_projection,
        _rest_comment,
    )

    body = "Relayed: widen the timeout." + SIGNATURE
    projection = _pr_projection(comments=[_gql_comment(1, "agent-bot", body)])

    def validate(text, requirements):
        return response_validation._validate_response_with_human_requirements(
            text,
            marker_validator=validate_structured_coder_followup,
            human_requirements=requirements,
            requirement_scope="PR requirements",
            full_omission_fallback="Fetch the PR discussion directly before approving.",
        )

    first = get_pr_review_context(
        RoutedRunner(projection=projection), config=make_config(tmp_path), pr_number=PR
    )
    (historical,) = first.human_requirements
    historical_ack = structured_coder_followup(
        human_requirement_ids=[historical.requirement_id],
        human_requirement_dispositions=[{
            "requirement_id": historical.requirement_id,
            "disposition": "addressed",
            "evidence": "Timeout widened.",
        }],
    ).replace(
        "\n<!-- AGENT_STATE",
        "\n<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n### Human requirements\n"
        f"- {historical.requirement_id}: widened.\n<!-- AGENT_STATE",
        1,
    )

    resumed_config = make_config(
        tmp_path,
        human_reviewer_trusted_actors=(TrustedHumanActor("maintainer", 101),),
    )
    resumed = get_pr_review_context(
        RoutedRunner(
            projection=projection,
            rest={_comments_path(): [_rest_comment(1, "agent-bot", 555, body)]},
        ),
        config=resumed_config,
        pr_number=PR,
    )

    assert resumed.human_requirements == ()
    validate(structured_coder_followup(), resumed.human_requirements)
    assert not response_validation._current_plan_has_complete_human_requirement_dispositions(
        historical_ack, surfaced_requirement_ids=()
    )


def test_pr_loop_resume_with_trust_set_excludes_historically_acknowledged_requirement(tmp_path):
    """#1022 review item-4: resume the real PR loop after the trust set changes.

    An unconfigured run surfaced and acknowledged a signed comment from an
    agent account.  The resumed run configures a trust set that excludes it:
    the requirement is loaded through the real context/verification path,
    is not surfaced to the coder or reviewer, the historical acknowledgement
    records do not raise, and a coder response that omits the ID validates.
    """
    from coding_review_agent_loop.github import TrustedHumanActor
    from coding_review_agent_loop.runner import CommandResult

    signed_url = "https://github.com/OWNER/REPO/pull/77#issuecomment-1"
    signed_body = "Relayed: widen the timeout.\n\n-- Human Reviewer"
    excluded = HumanReviewRequirement(
        source_type="PR comment",
        author="agent-bot",
        created_at="2026-05-18T10:00:00Z",
        url=signed_url,
        body="Relayed: widen the timeout.",
    )
    historical_review = _attach_round_metadata(
        structured_pr_review(
            state="approved",
            summary="Codex approved under the unconfigured run.",
            human_requirements_resolved=True,
        ),
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            # Round records name the reviewer by its board display name.
            agent="Codex",
            round_number=1,
            subject="abc000",
            state="approved",
            surfaced_reviewer_requirement_ids=(excluded.requirement_id,),
        ),
    )
    historical_coder_ack = structured_coder_followup(
        human_requirement_ids=[excluded.requirement_id],
        human_requirement_dispositions=[{
            "requirement_id": excluded.requirement_id,
            "disposition": "addressed",
            "evidence": "Timeout widened.",
        }],
    )

    class TrustRunner(FakeRunner):
        def run(self, args, *, cwd, input_text=None, check=True, env=None):
            cmd = [str(arg) for arg in args]
            if cmd[:2] == ["gh", "api"] and len(cmd) > 2 and re.fullmatch(
                r"repos/OWNER/REPO/issues/77/comments\?per_page=100&page=\d+", cmd[2]
            ):
                self.commands.append((cmd, Path(cwd)))
                page = [{
                    "id": 1,
                    "url": "https://api.github.com/repos/OWNER/REPO/issues/comments/1",
                    "user": {"login": "agent-bot", "id": 555},
                    "created_at": "2026-05-18T10:00:00Z",
                    "html_url": signed_url,
                    "body": signed_body,
                }] if cmd[2].endswith("page=1") else []
                if cmd[2].endswith('page=1'):
                    page.extend({'id': c.get('databaseId', 10_000 + i),
                        'user': {'login': c['author']['login'], 'id': c.get('_rest_author_id', 424242)},
                        'created_at': c.get('createdAt'), 'body': c.get('body')}
                        for i, c in enumerate(self.pr_payload['comments'], start=1) if c['body'] != signed_body)
                return CommandResult(cmd, Path(cwd), json.dumps(page), "", 0)
            return super().run(args, cwd=cwd, input_text=input_text, check=check, env=env)

    runner = TrustRunner(
        claude_outputs=[structured_coder_followup(addressed_items=["item-1"])],
        codex_outputs=[
            structured_pr_review(state="blocking", blocking_items=["Fix the retry bound."]),
            structured_pr_review(
                state="approved",
                prior_item_dispositions=[{"item_id": "item-1", "disposition": "resolved"}],
            ),
        ],
        pr_payload={
            "number": 77,
            "state": "OPEN",
            "url": "https://github.com/OWNER/REPO/pull/77",
            "title": "Widen timeout",
            "headRefName": "feature/timeout",
            "baseRefName": "main",
            "headRefOid": "6367c48dd193d56ea7b0baad25b19455e529f5ee",
            "comments": [
                {
                    "author": {"login": "agent-bot"},
                    "createdAt": "2026-05-18T10:00:00Z",
                    "url": signed_url,
                    "body": signed_body,
                },
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": "2026-05-18T11:00:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-2",
                    "body": str(historical_review),
                },
                {
                    "author": {"login": "coding-review-agent-loop"},
                    "createdAt": "2026-05-18T11:30:00Z",
                    "url": "https://github.com/OWNER/REPO/pull/77#issuecomment-3",
                    "body": historical_coder_ack,
                },
            ],
            "reviews": [],
        },
    )
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer="codex",
        max_rounds=2,
        human_reviewer_trusted_actors=(TrustedHumanActor("maintainer", 101),),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    rest_reads = [cmd[2] for cmd, _cwd in runner.commands if cmd[:2] == ["gh", "api"]]
    assert any("issues/77/comments?per_page=100" in path for path in rest_reads)
    coder_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"]]
    review_prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert coder_prompts and len(review_prompts) == 2
    for prompt in (*coder_prompts, *review_prompts):
        assert f"Requirement {excluded.requirement_id}:" not in prompt
    assert "No signed human requirements were surfaced" in coder_prompts[-1]


from acknowledgement_helpers import ACK_ID, dedicated_ack, durable_record, pr_payload as ack_pr_payload
from coding_review_agent_loop.round_state import _extract_round_metadata_records
from coding_review_agent_loop.repair import RepairAttemptResult
from coding_review_agent_loop.github import _parse_pr_human_requirements


def _exhaust_ack_repair(monkeypatch):
    calls = []
    def exhausted(raw, **kwargs):
        assert kwargs["repair_kwargs"]["expected_kind"] == "pr_review"
        calls.append(raw)
        attempts = [RepairAttemptResult(
            backend="codex", model=model, prompt="Repair acknowledgement.", output="invalid",
            returncode=0, outcome="invalid_output", diagnostic="Missing acknowledgement.",
            log_path=None, fallback_planned=index == 0,
        ) for index, model in enumerate(("gpt-6-luna", "gpt-6.1-sol"))]
        return None, None, attempts
    monkeypatch.setattr(pr_loop, "_run_structured_repair", exhausted)
    return calls


def test_ack_exhausted_repair_upserts_dedicated_durable_item_and_coder_clears(tmp_path, monkeypatch):
    calls = _exhaust_ack_repair(monkeypatch)
    requirement_id = _parse_pr_human_requirements(ack_pr_payload())[0].requirement_id
    runner = FakeRunner(
        pr_payload=ack_pr_payload(),
        codex_outputs=[_code_requirement_review(), _code_requirement_review(), _code_requirement_review(human_requirements_resolved=True)],
        claude_outputs=[structured_coder_followup(human_requirement_ids=[requirement_id])] * 2,
    )
    clears = []
    real = pr_loop._reconcile_human_requirements_ack_item
    def reconcile(items, **kwargs):
        result = real(items, **kwargs)
        if any(i.item_id == ACK_ID for i in items) and not any(i.item_id == ACK_ID for i in result):
            clears.append(kwargs["source_round"])
        return result
    monkeypatch.setattr(pr_loop, "_reconcile_human_requirements_ack_item", reconcile)
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=3)) == 0
    assert len(calls) == 2
    assert clears
    from types import SimpleNamespace
    records = _extract_round_metadata_records([SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr")
    failures = [r for r in records if "repair remains blocked" in r.body]
    assert [r.metadata.round_number for r in failures] == [1, 2]
    for record in failures:
        assert [i.item_id for i in record.metadata.prior_items] == [ACK_ID]
        assert record.metadata.prior_items[0].source_round == record.metadata.round_number
    from coding_review_agent_loop.round_state import _max_unresolved_item_number_from_records
    assert _max_unresolved_item_number_from_records(records) == 0


def test_ack_resume_coder_clearance_then_fresh_missing_reviewer_reraises(tmp_path, monkeypatch):
    from acknowledgement_helpers import ack_resume_history
    calls = _exhaust_ack_repair(monkeypatch)
    runner = FakeRunner(
        pr_payload=ack_pr_payload(ack_resume_history(promoted=True, coder_ack=True, unrelated=False)),
        gemini_outputs=[_code_requirement_review(reviewer="Google Gemini")],
    )
    with pytest.raises(AgentLoopError, match="blocking issues after round 2"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, reviewer=("codex", "gemini"), max_rounds=2))
    assert len(calls) == 1
    from types import SimpleNamespace
    failures = [r for r in _extract_round_metadata_records(
        [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr"
    ) if "repair remains blocked" in r.body]
    assert len(failures) == 1
    assert [i.item_id for i in failures[0].metadata.prior_items] == [ACK_ID]


@pytest.mark.parametrize("round_number", [1, 5])
def test_ack_failure_boundary_resumes_round_ledger_and_numbering(tmp_path, monkeypatch, round_number):
    from dataclasses import replace
    from coding_review_agent_loop.round_state import _resume_pr_round
    _exhaust_ack_repair(monkeypatch)
    future = UnresolvedReviewItem(item_id="item-7", reviewer="Codex", source_round=1,
                                  text="Preserve deferred work.", status="future")
    high_item = replace(future, item_id="item-41")
    prior = (future,)
    history = [durable_record((high_item,), round_number=1),
               durable_record(prior, round_number=round_number)]
    runner = FakeRunner(pr_payload=ack_pr_payload(history), codex_outputs=[_code_requirement_review(
        prior_item_dispositions=[{"item_id": "item-7", "disposition": "future", "note": "Keep deferred."}],
    )])
    with pytest.raises(AgentLoopError, match=f"blocking issues after round {round_number}"):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=round_number))
    comments = [IssueComment(author="bot", created_at=None, body=c["body"])
                for c in runner.pr_payload["comments"]]
    for _ in range(2):
        resumed = _resume_pr_round(comments, head_sha="6367c48dd193d56ea7b0baad25b19455e529f5ee", configured_reviewers=("codex",))
        assert resumed.round_number == round_number
        assert {i.item_id for i in resumed.prior_items} == {ACK_ID, "item-7"}
        assert resumed.next_unresolved_item_number == 42
        assert not resumed.reconciled
        assert not resumed.completed_reviews


@pytest.mark.parametrize("policy,force_full", [("selective-intermediate", False),
                                                     ("selective-intermediate", True),
                                                     ("primary-then-panel", True)])
@pytest.mark.parametrize("fresh_read", [True, False])
def test_ack_failure_parallel_restart_reruns_board_and_clears(tmp_path, monkeypatch, policy, force_full, fresh_read):
    """The reopened pass can read old publications and still launch a fresh board."""
    calls = _exhaust_ack_repair(monkeypatch)
    runner = FakeRunner(
        pr_payload=ack_pr_payload(),
        codex_outputs=[_code_requirement_review(human_requirements_resolved=False)],
        gemini_outputs=[_code_requirement_review(reviewer="Google Gemini")],
    )
    config = make_config(
        tmp_path, reviewer=("codex", "gemini"), review_parallel=True,
        pr_review_force_full=force_full, pr_review_policy=policy, primary_reviewer="codex" if policy == "primary-then-panel" else None,
        max_rounds=1,
    )
    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)
    from types import SimpleNamespace
    def records():
        return _extract_round_metadata_records(
            [SimpleNamespace(body=c["body"]) for c in runner.pr_payload["comments"]], flow="pr")
    before = records()
    publications = [r for r in before if r.metadata.phase == "publication"]
    assert {r.metadata.agent for r in publications} == {"Codex", "Gemini"}
    assert len(calls) == 2
    runner.codex_outputs.append(_code_requirement_review(human_requirements_resolved=True, prior_item_dispositions=[{"item_id": ACK_ID, "disposition": "resolved"}]))
    runner.gemini_outputs.append(_code_requirement_review(reviewer="Google Gemini", human_requirements_resolved=True, prior_item_dispositions=[{"item_id": ACK_ID, "disposition": "resolved"}]))
    if not fresh_read:
        real_context = runner.run
        monkeypatch.setattr("coding_review_agent_loop.github_retry._sleep", lambda _: None)
        failed_reads = []
        def unavailable_after_checkpoint(args, **kwargs):
            result = real_context(args, **kwargs)
            if (args[:3] == ["gh", "pr", "view"] and "comments" in args[-1]
                    and len(failed_reads) < 3 and sum(r.metadata.phase == "scheduler-prelaunch" for r in records()) > sum(
                r.metadata.phase == "scheduler-prelaunch" for r in before
            )):
                failed_reads.append(True)
                return replace(result, returncode=1, stdout="", stderr="HTTP 503 Service Unavailable")
            return result
        monkeypatch.setattr(runner, "run", unavailable_after_checkpoint)
    snapshots = []
    real_clear = pr_loop._clear_human_requirements_ack_item
    def capture(items):
        result = real_clear(items)
        snapshots.append((tuple(items), tuple(result)))
        return result
    monkeypatch.setattr(pr_loop, "_clear_human_requirements_ack_item", capture)
    if not fresh_read:
        launches_before = len([cmd for cmd, _ in runner.commands if cmd[0] in {'codex', 'gemini'}])
        with pytest.raises(AgentLoopError, match='HTTP 503'):
            run_pr_loop(runner, pr_number=77, config=config)
        assert len([cmd for cmd, _ in runner.commands if cmd[0] in {'codex', 'gemini'}]) == launches_before
        # Complete acquisition on the next invocation resumes the retained
        # checkpoint and ledger, rather than continuing with stale peers.
    assert run_pr_loop(runner, pr_number=77, config=config) == 0
    after = records()
    assert len([r for r in after if r.metadata.phase == "publication"]) == 4
    assert not runner.codex_outputs and not runner.gemini_outputs
    assert len(calls) == 2
    assert snapshots and any(i.item_id == ACK_ID for i in snapshots[-1][0])
    assert not snapshots[-1][1]
    assert after[:len(before)] == before
    from coding_review_agent_loop.round_state import _max_unresolved_item_number_from_records
    assert _max_unresolved_item_number_from_records(after) == 0
    if not fresh_read:
        assert len(failed_reads) == 3
    assert all(r.metadata.round_number == 1 for r in after)
    assert not any(cmd[:3] == ["gh", "api", "--method"] and "DELETE" in cmd
                   for cmd, _ in runner.commands)


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("classification_pass", ["classified", "request-only", "standalone"])
@pytest.mark.parametrize("resumed", [False, True])
def test_human_only_signed_requirement_freezes_without_coder(tmp_path, capsys, mode, classification_pass, resumed):
    from pr_loop_test_helpers import (_classified_review, _association, _evidence_review,
        _signed, _EVIDENCE_TEXT, _freeze_records)
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    if resumed:
        from test_pr_loop_early_evidence import deferred
        runner, config = deferred(tmp_path, mode=mode)
    else:
        runner, config = FakeRunner(), make_config(tmp_path, pr_review_context_mode=mode)
    runner.pr_payload["comments"].append(_signed("A human must accept the policy at this exact head.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    pending = [{"item_id": "item-1", "disposition": "blocking", "note": "Human acceptance remains pending."}] if resumed else []
    review_args = {"state": "blocking" if resumed else "approved", "dispositions": pending, "evidence": [_EVIDENCE_TEXT]}
    classified = _classified_review([_association(rid, "human-only", request=_EVIDENCE_TEXT)], **review_args)
    initial_args = {**review_args, "evidence": []} if classification_pass == "standalone" else review_args
    runner.codex_outputs = ([_evidence_review(**initial_args)] if classification_pass != "classified" else []) + [classified]
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == 1
    assert not runner.codex_outputs
    freeze = _freeze_records(runner)[-1].metadata
    assert len(freeze.prior_items) == 1
    assert freeze.prior_items[0].text == _EVIDENCE_TEXT
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)
    assert "re-injecting as blocking item" not in capsys.readouterr().err
    count = len(_freeze_records(runner))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=config)
    assert len(_freeze_records(runner)) == count


def test_mixed_code_work_keeps_human_requirement_out_of_coder_prompt(tmp_path):
    from pr_loop_test_helpers import (_classified_review, _association, _signed, _EVIDENCE_TEXT, _freeze_records)
    from coding_review_agent_loop.errors import HumanDecisionRequiredError
    runner = FakeRunner()
    runner.pr_payload["comments"].extend([_signed("A human must accept the policy.", 1), _signed("Fix the retry bound.", 2)])
    human, code = _parse_pr_human_requirements(runner.pr_payload)
    entries = [_association(human.requirement_id, "human-only", request=_EVIDENCE_TEXT), _association(code.requirement_id, "code")]
    runner.codex_outputs = [_classified_review(entries, state="blocking", evidence=[_EVIDENCE_TEXT], blocking=["Fix the retry bound."]),
        _classified_review(entries, state="blocking", evidence=[_EVIDENCE_TEXT], hr=True,
            dispositions=[{"item_id": "item-1", "disposition": "resolved"},
                {"item_id": "item-2", "disposition": "blocking", "note": "Human acceptance remains pending."}])]
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-1"],
        human_requirement_ids=[code.requirement_id], human_requirement_dispositions=[{
            "requirement_id": code.requirement_id, "disposition": "addressed", "evidence": "Retry bound fixed."}])]
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=2))
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert len(prompts) == 1
    assert f"Requirement {code.requirement_id}:" in prompts[0]
    assert f"Requirement {human.requirement_id}:" not in prompts[0]
    assert len(_freeze_records(runner)[-1].metadata.prior_items) == 1


@pytest.mark.parametrize("role", ["human-only", "evidence-only"])
def test_legacy_ack_gate_clears_after_real_evidence_classification(tmp_path, role):
    from coding_review_agent_loop.round_metadata import (
        PostedRoundMetadata,
    )
    from coding_review_agent_loop.native_evidence import (
        _append_recorded_evidence_requests,
    )
    from coding_review_agent_loop import pr_loop_support as support
    from pr_loop_test_helpers import _classified_review, _association, _signed, _EVIDENCE_TEXT, _freeze_records, _pr_records
    from test_pr_loop_early_evidence import deferred, HEAD
    from coding_review_agent_loop.errors import HumanDecisionRequiredError

    runner, config = deferred(tmp_path, mode="full")
    body = "Human acceptance is required." if role == "human-only" else "Accepted for " + HEAD
    runner.pr_payload["comments"].append(_signed(body, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    records = _pr_records(runner)
    items = []
    _append_recorded_evidence_requests(items, [r for r in records if r.metadata.role == "reviewer"], records=records)
    support._record_failed_acknowledgement(runner, config=config, pr_number=77, items=items,
        source_round=2, missing_reviewers=["Codex"], subject=HEAD,
        bound_metadata=PostedRoundMetadata, post_comment=support.post_pr_comment)
    entry = _association(rid, role, request=_EVIDENCE_TEXT) if role == "human-only" else _association(rid, role, answers=["item-1"])
    evidence_vote = {"item_id": "item-1", "disposition": "blocking", "note": "Human acceptance remains pending."} if role == "human-only" else {
        "item_id": "item-1", "disposition": "resolved", "evidence_requirement_ids": [rid]}
    runner.codex_outputs = [_classified_review([entry], state="blocking" if role == "human-only" else "approved",
        evidence=[_EVIDENCE_TEXT] if role == "human-only" else [], dispositions=[evidence_vote,
            {"item_id": HUMAN_REQUIREMENTS_ACK_ITEM_ID, "disposition": "resolved"}])]
    if role == "human-only":
        with pytest.raises(HumanDecisionRequiredError):
            run_pr_loop(runner, pr_number=77, config=config)
        assert [i.item_id for i in _freeze_records(runner)[-1].metadata.prior_items] == ["item-1"]
    else:
        assert run_pr_loop(runner, pr_number=77, config=config) == 0
        assert not _freeze_records(runner)
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)


@pytest.mark.parametrize("mode", ["compact", "full"])
@pytest.mark.parametrize("fault", ["absent", "role"])
def test_unsettled_signed_routing_stops_before_coder(tmp_path, capsys, mode, fault):
    from pr_loop_test_helpers import (_classified_review, _association, _evidence_review,
        _signed, _EVIDENCE_TEXT, _pr_records)
    from coding_review_agent_loop.errors import HumanDecisionRequiredError

    runner = FakeRunner()
    runner.pr_payload["comments"].append(_signed("A human must accept the policy. Also fix retries.", 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    if fault == "absent":
        runner.codex_outputs = [_evidence_review(), _evidence_review()]
        reviewers = "codex"
    else:
        # Publish the requester's pending human evidence before another seat
        # disagrees, then leave that disagreement intact in its real continuation.
        first = _association(rid, "human-only", request=_EVIDENCE_TEXT)
        second = _association(rid, "code")
        runner.codex_outputs = [_classified_review([first], evidence=[_EVIDENCE_TEXT], state="approved")]
        disagreeing = _classified_review([second], reviewer="Google Gemini", state="approved")
        runner.gemini_outputs = [disagreeing, disagreeing]
        reviewers = ("codex", "gemini")
    config = make_config(tmp_path, pr_review_context_mode=mode, reviewer=reviewers, review_parallel=False)
    with pytest.raises(HumanDecisionRequiredError, match="classification remained"):
        run_pr_loop(runner, pr_number=77, config=config)
    assert not runner.codex_outputs and not runner.gemini_outputs
    assert not any(cmd[0] == "claude" for cmd, _ in runner.commands)
    assert "re-injecting as blocking item" not in capsys.readouterr().err
    assert _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id == rid
    if fault != "absent":
        assert any(r.metadata.evidence_requests for r in _pr_records(runner))


_CLASSIFICATION_CONTINUATION = "Continue the PR review with semantic signed-source classification"


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_mixed_wording_difference_needs_no_classification_continuation(tmp_path, mode):
    """Agreeing mixed roles with different leftover wording are not a conflict (#2000).

    The merged code leftover reaches the coder while the unaccepted answers stay
    a deferred evidence barrier: classification never grants acceptance.
    """
    from pr_loop_test_helpers import _classified_review, _association, _signed, _release_records, _freeze_records
    from agent_loop_helpers import structured_coder_followup
    from test_pr_loop_early_evidence import deferred

    runner, config = deferred(tmp_path, mode=mode, two_owners=True)
    runner.advance_pr_head_on_coder_followup = False
    signed_text = "Accepted. Also fix retries."
    runner.pr_payload["comments"].append(_signed(signed_text, 1))
    rid = _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id
    pending = [{"item_id": f"item-{i}", "disposition": "blocking", "note": "Human acceptance remains pending."}
        for i in (1, 2)]
    codex_entry = _association(rid, "mixed", answers=["item-1"], leftover="Fix retries.")
    gemini_entry = _association(rid, "mixed", answers=["item-2"], leftover="Repair retries.")
    runner.codex_outputs = [_classified_review([codex_entry], dispositions=pending, state="blocking")]
    # Exactly one Gemini output for this round: any classification continuation
    # would consume the follow-up review below and exhaust the fake runner.
    runner.gemini_outputs = [_classified_review([gemini_entry], reviewer="Google Gemini",
        dispositions=pending, state="blocking")]
    runner.claude_outputs = [structured_coder_followup(human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed",
                                         "evidence": "Retries fixed; human answer is supplied separately."}])]
    runner.codex_outputs.append(_classified_review([codex_entry], hr=True, dispositions=pending, state="blocking"))
    runner.gemini_outputs.append(_classified_review([gemini_entry], reviewer="Google Gemini", hr=True,
        dispositions=pending, state="blocking"))
    # The deferred evidence barrier (#1068) stops finalization: the run ends at the
    # human-evidence freeze, not at a classification conflict and not at merge.
    with pytest.raises(HumanDecisionRequiredError) as stop:
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=4))
    assert "classification remained" not in str(stop.value)
    freezes = _freeze_records(runner)
    assert freezes
    pending_items = freezes[-1].metadata.prior_items
    assert {(i.item_id, i.reviewer) for i in pending_items} == {("item-1", "Codex"), ("item-2", "Gemini")}
    assert not any(cmd[:3] == ["gh", "pr", "merge"] for cmd, _ in runner.commands)
    assert not any(cmd[:3] == ["gh", "pr", "ready"] for cmd, _ in runner.commands)
    assert not runner.gemini_outputs and not runner.codex_outputs
    assert not any(_CLASSIFICATION_CONTINUATION in cmd[-1] for cmd, _ in runner.commands)
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert len(prompts) == 1
    block = prompts[0].split(f"Requirement {rid}:", 1)[1]
    assert signed_text in block
    assert block.index("- Codex: Fix retries.") < block.index("- Gemini: Repair retries.")
    # The unaccepted answers are never cleared by classification.
    assert not any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner))
    assert _parse_pr_human_requirements(runner.pr_payload)[0].requirement_id == rid


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_mixed_wording_difference_dispatches_merged_requirement(tmp_path, mode):
    """Cleared evidence dispatches one attributed merged requirement under the original id (#2000)."""
    from pr_loop_test_helpers import _classified_review, _association, _resolve
    from agent_loop_helpers import structured_coder_followup
    from test_pr_loop_early_evidence import deferred, acceptance, HEAD

    runner, config = deferred(tmp_path, mode=mode, two_owners=True)
    runner.advance_pr_head_on_coder_followup = False
    signed_text = "Accepted for " + HEAD + ". Also fix retries."
    rid = acceptance(runner, signed_text, two_owners=True)
    resolved = [{**_resolve(), "evidence_requirement_ids": [rid]},
                {**_resolve("item-2"), "evidence_requirement_ids": [rid]}]
    codex_entry = _association(rid, "mixed", answers=["item-1"], leftover="Fix retries.")
    gemini_entry = _association(rid, "mixed", answers=["item-2"], leftover="Repair retries.")
    runner.codex_outputs[-1] = _classified_review([codex_entry], state="blocking", blocking=["Fix retries."],
        dispositions=resolved)
    runner.gemini_outputs[-1] = _classified_review([gemini_entry], reviewer="Google Gemini", state="blocking",
        blocking=["Repair retries."], dispositions=resolved)
    runner.claude_outputs = [structured_coder_followup(addressed_items=["item-3", "item-4"], human_requirement_ids=[rid],
        human_requirement_dispositions=[{"requirement_id": rid, "disposition": "addressed",
                                         "evidence": "Retries fixed; human answer is supplied separately."}])]
    runner.codex_outputs.append(_classified_review([codex_entry], hr=True,
        dispositions=[*resolved, _resolve("item-3"), _resolve("item-4")]))
    runner.gemini_outputs.append(_classified_review([gemini_entry], reviewer="Google Gemini", hr=True,
        dispositions=[*resolved, _resolve("item-3"), _resolve("item-4")]))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3)) == 0
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert len(prompts) == 1
    assert not any(_CLASSIFICATION_CONTINUATION in cmd[-1] for cmd, _ in runner.commands)
    assert f"Requirement {rid}:" in prompts[0]
    block = prompts[0].split(f"Requirement {rid}:", 1)[1]
    assert signed_text in block
    codex_line = block.index("- Codex: Fix retries.")
    gemini_line = block.index("- Gemini: Repair retries.")
    assert codex_line < gemini_line


# --- Evidence-only signed requirements are reviewer-owned (#2020, #2299 shape) ---

_REVIEWER_OWNED_NOTICE = "Reviewer-owned signed requirements (read-only):"
_REVIEWER_SCOPE_LINE = "Coder disposition scope:"


def _evidence_only_cited_round(tmp_path, mode, *, accepted=True):
    """One evidence-only signed source whose requester demands a coder disposition."""
    from pr_loop_test_helpers import _classified_review, _association, _resolve
    from test_pr_loop_early_evidence import deferred, acceptance, HEAD

    runner, config = deferred(tmp_path, mode=mode)
    runner.advance_pr_head_on_coder_followup = False
    rid = acceptance(runner, "Accepted for " + HEAD + "; manifest verified.")
    entry = _association(rid, "evidence-only", answers=["item-1"])
    evidence_vote = ({**_resolve(), "evidence_requirement_ids": [rid]} if accepted else
        {"item_id": "item-1", "disposition": "blocking", "note": "Acceptance remains pending."})
    demand = f"The coder must record a human_requirement_disposition for signed requirement {rid}."
    runner.codex_outputs[-1] = _classified_review([entry], state="blocking", blocking=[demand],
        dispositions=[evidence_vote])
    return runner, config, rid, entry, evidence_vote


def _non_actionable_followup():
    return structured_coder_followup(
        disputed_items=["item-2"],
        dispute_evidence={"item-2": "Reviewer-owned evidence-only signed requirement; non-actionable for the coder."},
        human_requirement_dispositions=[],
    )


def _ack_item_ever_recorded(runner):
    from pr_loop_test_helpers import _pr_records
    return any(item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID
               for record in _pr_records(runner)
               for item in (*record.metadata.prior_items, *record.metadata.new_items))


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_evidence_only_requirement_cited_by_reviewer_validates_without_repair(tmp_path, mode, capsys):
    from pr_loop_test_helpers import _classified_review, _resolve, _release_records

    runner, config, rid, entry, evidence_vote = _evidence_only_cited_round(tmp_path, mode)
    runner.claude_outputs = [_non_actionable_followup()]
    runner.codex_outputs.append(_classified_review([entry], hr=True,
        dispositions=[evidence_vote, _resolve("item-2", "Reviewer-owned; evidence accepted directly.")]))
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3)) == 0
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    # Exactly one coder turn: the follow-up validated on its first attempt.
    assert len(prompts) == 1
    assert "repair" not in capsys.readouterr().err.lower()
    assert f"Requirement {rid}:" not in prompts[0]
    notice = prompts[0].split(_REVIEWER_OWNED_NOTICE, 1)[1].split("\n", 1)[0]
    assert f"`{rid}`" in notice
    assert "non-actionable" in notice
    assert not _ack_item_ever_recorded(runner)
    # The resumed second reviewer round lists the evidence-only ID as reviewer-owned.
    review_prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "codex"]
    assert f"`{rid}`" in review_prompts[-1].split(_REVIEWER_SCOPE_LINE, 1)[1].split("\n", 1)[0]
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in review_prompts[-1]
    assert not runner.codex_outputs and not runner.claude_outputs
    assert _release_records(runner) is not None


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_evidence_only_classification_alone_is_not_acceptance(tmp_path, mode):
    """Routing never grants acceptance: pending evidence freezes, no coder disposition is needed."""
    from pr_loop_test_helpers import _classified_review, _release_records, _freeze_records

    runner, config, rid, entry, evidence_vote = _evidence_only_cited_round(tmp_path, mode, accepted=False)
    runner.claude_outputs = [_non_actionable_followup()]
    runner.codex_outputs.append(_classified_review([entry], hr=True, state="blocking",
        dispositions=[evidence_vote, {"item_id": "item-2", "disposition": "resolved"}]))
    with pytest.raises(HumanDecisionRequiredError):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3))
    prompts = [cmd[-1] for cmd, _ in runner.commands if cmd[0] == "claude"]
    assert len(prompts) == 1
    assert not _ack_item_ever_recorded(runner)
    assert not any(r.metadata.evidence_release.reason == "evidence-cleared" for r in _release_records(runner))
    assert [i.item_id for i in _freeze_records(runner)[-1].metadata.prior_items] == ["item-1"]


def test_coder_naming_evidence_only_id_gets_out_of_scope_message(tmp_path):
    from coding_review_agent_loop.errors import AgentInvocationError

    runner, config, rid, entry, evidence_vote = _evidence_only_cited_round(tmp_path, "full")
    runner.claude_outputs = [
        structured_coder_followup(disputed_items=["item-2"], dispute_evidence={"item-2": "Recorded."},
            human_requirement_ids=[rid], human_requirement_dispositions=[{
                "requirement_id": rid, "disposition": "addressed", "evidence": "Acknowledged."}]),
    ]
    with pytest.raises(AgentInvocationError) as error:
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3))
    message = str(error.value)
    assert "coder_followup.human_requirement_dispositions names signed requirement ID(s)" in message
    assert rid in message and "outside coder scope" in message
    assert "unknown requirement ID" not in message


@pytest.mark.parametrize("mode", ["compact", "full"])
def test_resumed_round_reuses_historical_projection_for_reviewer_owned_ids(tmp_path, mode, monkeypatch):
    """A rerun seeds reconciliation and the first reviewer prompt from one historical projection."""
    from pr_loop_test_helpers import _classified_review, _resolve
    from coding_review_agent_loop import requirement_evidence

    runner, config, rid, entry, evidence_vote = _evidence_only_cited_round(tmp_path, mode)
    runner.claude_outputs = [_non_actionable_followup()]
    # Stop after the coder turn (the next reviewer output is exhausted) so the
    # next invocation resumes from the recorded GitHub round state.
    from coding_review_agent_loop.errors import AgentInvocationError
    with pytest.raises(AgentInvocationError, match="scripted agent output exhausted"):
        run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3))
    assert len([cmd for cmd, _ in runner.commands if cmd[0] == "claude"]) == 1
    calls = []
    seeded = []
    real_projection = pr_loop.historical_projection
    real_owned = pr_loop.reviewer_owned_ids

    def recording_projection(requirements, records):
        result = real_projection(requirements, records)
        calls.append(result)
        return result

    def recording_owned(requirements, projected):
        result = real_owned(requirements, projected)
        seeded.append((projected, result))
        return result

    monkeypatch.setattr(pr_loop, "historical_projection", recording_projection)
    monkeypatch.setattr(pr_loop, "reviewer_owned_ids", recording_owned)
    before = len(runner.commands)
    runner.codex_outputs = [_classified_review([entry], hr=True,
        dispositions=[evidence_vote, _resolve("item-2", "Reviewer-owned; evidence accepted directly.")])]
    assert run_pr_loop(runner, pr_number=77, config=dataclasses.replace(config, max_rounds=3)) == 0
    review_prompts = [cmd[-1] for cmd, _ in runner.commands[before:] if cmd[0] == "codex"]
    assert review_prompts
    scope = review_prompts[0].split(_REVIEWER_SCOPE_LINE, 1)[1].split("\n", 1)[0]
    assert f"`{rid}`" in scope
    # The first reviewer-owned set derives from the very projection object
    # used for reconciliation, never a recomputation with different inputs.
    assert calls and seeded[0][0] is calls[0]
    assert seeded[0][1] == (rid,)
    assert requirement_evidence.reviewer_owned_ids(
        _parse_pr_human_requirements(runner.pr_payload), calls[0]) == (rid,)
    assert not _ack_item_ever_recorded(runner)
    assert not [cmd for cmd, _ in runner.commands[before:] if cmd[0] == "claude"]


def test_review_guidance_scope_line_only_when_reviewer_owned_ids_exist():
    from coding_review_agent_loop import prompt_support
    requirement = HumanReviewRequirement("PR comment", "maintainer", None, "https://example.test/1", "Accept it.")
    baseline = prompt_support._human_requirements_review_guidance([requirement])
    assert prompt_support._human_requirements_review_guidance(
        [requirement], reviewer_owned_requirement_ids=()) == baseline
    assert _REVIEWER_SCOPE_LINE not in baseline
    owned = prompt_support._human_requirements_review_guidance(
        [requirement], reviewer_owned_requirement_ids=(requirement.requirement_id,))
    assert _REVIEWER_SCOPE_LINE in owned
    assert f"`{requirement.requirement_id}`" in owned
    assert "exact_head_evidence_requests" in owned and "evidence_requirement_ids" in owned
    # The approval-marker rule is not weakened.
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in owned
    assert "If any signed human requirement in this set is unresolved, return blocking." in owned
    assert owned.replace(prompt_support._reviewer_owned_review_scope_line((requirement.requirement_id,)), "") == baseline


@pytest.mark.parametrize("compact", [False, True])
def test_review_prompt_builders_thread_reviewer_owned_ids(tmp_path, compact):
    from coding_review_agent_loop.prompts import build_review_prompt
    requirement = HumanReviewRequirement("PR comment", "maintainer", None, "https://example.test/1", "Accept it.")
    config = make_config(tmp_path)
    kwargs = dict(reviewer=config.reviewers[0] if hasattr(config, "reviewers") else "codex",
                  human_requirements=[requirement], compact_context=compact)
    plain = build_review_prompt(77, 2, config, **kwargs)
    scoped = build_review_prompt(77, 2, config, reviewer_owned_requirement_ids=(requirement.requirement_id,), **kwargs)
    assert _REVIEWER_SCOPE_LINE not in plain
    assert f"`{requirement.requirement_id}`" in scoped.split(_REVIEWER_SCOPE_LINE, 1)[1].split("\n", 1)[0]


def test_coder_followup_guidance_notice_only_when_reviewer_owned_ids_exist():
    from coding_review_agent_loop import prompt_support
    owned_id = "hr-" + "b" * 64
    context = prompt_support.render_coder_human_requirements_prompt_context(())
    baseline = prompt_support._coder_human_requirements_guidance(context, response_kind="coder_followup")
    assert _REVIEWER_OWNED_NOTICE not in baseline
    with_notice = prompt_support._coder_human_requirements_guidance(
        context, response_kind="coder_followup", reviewer_owned_requirement_ids=(owned_id,))
    assert with_notice.startswith(baseline.rstrip("\n"))
    assert f"`{owned_id}`" in with_notice
    assert "No signed human requirements were surfaced." in with_notice
    assert context.surfaced_requirement_ids == ()


# --- #2026: a coder listing the tool-owned ack record never decides the ledger ---

_ACK_PLACEMENTS_2026 = {
    "addressed": {"addressed_items": [ACK_ID]},
    "remaining": {"remaining_items": [ACK_ID]},
    "disputed": {"disputed_items": [ACK_ID]},
    "addressed-and-remaining": {"addressed_items": [ACK_ID], "remaining_items": [ACK_ID]},
    "none": {},
}


@pytest.mark.parametrize("placement", list(_ACK_PLACEMENTS_2026))
def test_live_coder_ack_placement_ledger_is_placement_independent(tmp_path, monkeypatch, caplog, placement):
    caplog.set_level("WARNING")
    _exhaust_ack_repair(monkeypatch)
    requirement_id = _parse_pr_human_requirements(ack_pr_payload())[0].requirement_id
    coder_output = structured_coder_followup(
        human_requirement_ids=[requirement_id], **_ACK_PLACEMENTS_2026[placement]
    )
    runner = FakeRunner(
        pr_payload=ack_pr_payload(),
        codex_outputs=[_code_requirement_review(), _code_requirement_review(), _code_requirement_review(human_requirements_resolved=True)],
        claude_outputs=[coder_output] * 2,
    )
    after_coder = []
    real = pr_loop._reconcile_human_requirements_ack_item

    def reconcile(items, **kwargs):
        result = real(items, **kwargs)
        if kwargs.get("coder_output") == coder_output:
            after_coder.append((
                any(i.item_id == ACK_ID for i in items),
                [i.item_id for i in result],
            ))
        return result

    monkeypatch.setattr(pr_loop, "_reconcile_human_requirements_ack_item", reconcile)
    assert run_pr_loop(runner, pr_number=77, config=make_config(tmp_path, max_rounds=3)) == 0
    # Asserted at coder reconciliation, before any reviewer-board clearance:
    # a pending record clears and none is created, for every placement.
    assert after_coder and any(pending for pending, _ in after_coder)
    assert all(ACK_ID not in result for _pending, result in after_coder)
    lines = [r.getMessage() for r in caplog.records if "acknowledgement record" in r.getMessage()]
    if placement == "none":
        assert lines == []
    else:
        assert lines and all(
            ("counted as the coder's acknowledgement" in line) is placement.startswith("addressed")
            for line in lines
        )
    assert not any("unknown unresolved reviewer item IDs" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("placement", ["remaining", "none"])
def test_reconcile_without_pending_ack_creates_none_for_any_placement(placement):
    requirement = _parse_pr_human_requirements(ack_pr_payload())[0]
    output = structured_coder_followup(
        human_requirement_ids=[requirement.requirement_id], **_ACK_PLACEMENTS_2026[placement]
    )
    assert _reconcile_human_requirements_ack_item(
        (), coder_output=output, human_requirements=(requirement,), source_round=2
    ) == []


def test_invalid_signed_fields_with_ack_placement_fail_like_without_it():
    requirement = _parse_pr_human_requirements(ack_pr_payload())[0]
    errors = []
    for extra in ([ACK_ID], []):
        output = structured_coder_followup(
            addressed_items=extra, human_requirement_ids=[], human_requirement_dispositions=[]
        )
        with pytest.raises(AgentLoopError) as error:
            pr_loop._validate_coder_followup_response(
                output, unresolved_items=(), human_requirements=(requirement,),
                architecture_status_mode="legacy",
            )
        errors.append(str(error.value))
    assert errors[0] == errors[1]
    assert "unknown unresolved reviewer item IDs" not in errors[0]
