"""PR-loop tests for unresolved_items and pr_loop: PR human requirements (#1542)."""

import json
import re
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
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    _attach_round_metadata,
    _strip_round_metadata,
)
from coding_review_agent_loop.prompts import (
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
            "headRefOid": "abc123",
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
        "headRefOid": "abc123",
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
            "headRefOid": "abc123",
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
            "headRefOid": "abc123",
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
    config = make_config(tmp_path, coder="claude", reviewer="codex", max_rounds=2)
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Improve review prompt context",
        head_branch="feature/review-context",
        base_branch="main",
        head_sha="abc123",
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

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context",
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
        head_sha="abc123",
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
            subject="abc123",
            state="approved",
            surfaced_reviewer_requirement_ids=("Requirement 1",),
        ),
    )
    context_calls = 0

    def next_context(*args, **kwargs):
        nonlocal context_calls
        context_calls += 1
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

    patch_attr_in(monkeypatch, (
        pr_loop_support, pr_loop, issue_implementation, plan_first_loop, issue_loop,
    ), "get_pr_review_context",
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
            "headRefOid": "abc123",
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
