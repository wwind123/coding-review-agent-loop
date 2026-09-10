from dataclasses import replace

import pytest

from agent_loop_helpers import make_config
from coding_review_agent_loop.orchestrator import (
    _reconcile_human_requirements_ack_item,
    _reviewer_requirement_coverage_matches,
    _reviewer_requirement_identity_ids,
    _resumed_pr_reviewer_matches_requirements,
)
from coding_review_agent_loop.github import (
    HumanReviewRequirement,
    IssueComment,
    IssueContext,
    PullRequestMetadata,
    PullRequestReviewContext,
    deduplicate_human_requirements,
)
from coding_review_agent_loop.errors import AgentLoopError
from coding_review_agent_loop.comment_rendering import normalize_freeform_signature
from coding_review_agent_loop.prompts import (
    build_review_prompt,
    format_human_requirements,
    render_coder_human_requirements_prompt_context,
)
from coding_review_agent_loop.protocol import validate_human_requirements_acknowledgement
from coding_review_agent_loop.round_state import (
    PostedRoundMetadata,
    PostedRoundRecord,
    _attach_round_metadata,
    make_approved_plan_context,
    recover_approved_plan_context,
    _latest_pr_approved_reviews_for_head,
)


def _requirement(*, body: str, created_at: str, url: str) -> HumanReviewRequirement:
    return HumanReviewRequirement(
        source_type="Issue comment",
        author="maintainer",
        created_at=created_at,
        url=url,
        body=body,
    )


def _issue(number: int, requirements=(), comments=()) -> IssueContext:
    return IssueContext(
        number=number,
        repo="OWNER/REPO",
        title="Issue",
        body="Original issue",
        url=f"https://github.com/OWNER/REPO/issues/{number}",
        comments=tuple(comments),
        human_requirements=tuple(requirements),
    )


def test_signed_requirement_ids_survive_insertion_and_body_edits():
    first = _requirement(
        body="Keep the public API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
    )
    inserted = _requirement(
        body="Add a regression test.",
        created_at="2026-01-02T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/1#issuecomment-2",
    )
    edited = replace(first, body="Change the public API.")
    assert first.requirement_id != edited.requirement_id
    assert f"Requirement {first.requirement_id}:" in format_human_requirements((first, inserted))
    assert f"Requirement {first.requirement_id}:" in format_human_requirements((inserted, first))


def test_reviewer_coverage_uses_digest_identity_and_rejects_legacy_labels():
    original = _requirement(
        body="Keep the public API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
    )
    edited = replace(original, body="Change the public API.")

    assert _reviewer_requirement_identity_ids((original,)) == (original.requirement_id,)
    assert _reviewer_requirement_coverage_matches((original,), (original.requirement_id,))
    assert not _reviewer_requirement_coverage_matches((edited,), (original.requirement_id,))
    assert not _reviewer_requirement_coverage_matches((original,), ("Requirement 1",))


def test_interrupted_current_round_rechecks_resumed_reviewer_requirement_coverage():
    original = _requirement(
        body="Keep the public API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
    )
    edited = replace(original, body="Change the public API.")
    record = PostedRoundRecord(
        index=0,
        metadata=PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="same-head",
            state="approved",
            surfaced_reviewer_requirement_ids=(original.requirement_id,),
        ),
        body="Approved.\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->",
    )

    assert _resumed_pr_reviewer_matches_requirements(record, (original,))
    assert not _resumed_pr_reviewer_matches_requirements(record, (edited,))


def test_latest_same_head_approval_does_not_fall_back_past_plan_mismatch():
    plan = make_approved_plan_context("Approved plan")
    old = _attach_round_metadata(
        "Old approval",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="same-head",
            state="approved",
            approved_plan_hash=plan.plan_hash,
            approved_plan_subject=plan.plan_subject,
        ),
    )
    newer = _attach_round_metadata(
        "Approval without plan identity",
        PostedRoundMetadata(
            flow="pr",
            role="reviewer",
            agent="Codex",
            round_number=1,
            subject="same-head",
            state="approved",
        ),
    )
    comments = (
        IssueComment(author="bot", created_at="2026-01-01T00:00:00Z", body=old),
        IssueComment(author="bot", created_at="2026-01-01T00:01:00Z", body=newer),
    )

    assert _latest_pr_approved_reviews_for_head(
        comments,
        head_sha="same-head",
        configured_reviewers=("codex",),
        approved_plan_context=plan,
    ) == {}


def test_legacy_positional_acknowledgement_requires_fresh_stable_acknowledgement():
    requirement = _requirement(
        body="Keep the public API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/issues/1#issuecomment-1",
    )
    context = render_coder_human_requirements_prompt_context((requirement,))
    assert context.surfaced_requirement_ids == (requirement.requirement_id,)
    with pytest.raises(AgentLoopError, match="fresh acknowledgement"):
        validate_human_requirements_acknowledgement(
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
            "### Human requirements\n- Requirement 1: done",
            surfaced_requirement_ids=context.surfaced_requirement_ids,
            requires_direct_discussion_ack=False,
        )

    validate_human_requirements_acknowledgement(
        "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        f"### Human requirements\n- Requirement {requirement.requirement_id}: done",
        surfaced_requirement_ids=context.surfaced_requirement_ids,
        requires_direct_discussion_ack=False,
    )
    # The coder guidance surfaces the stable ID itself, so the same bare token
    # must be accepted in a markdown bullet as in structured JSON fields.
    validate_human_requirements_acknowledgement(
        "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
        f"### Human requirements\n- `{requirement.requirement_id}`: done",
        surfaced_requirement_ids=context.surfaced_requirement_ids,
        requires_direct_discussion_ack=False,
    )

    reconciled = _reconcile_human_requirements_ack_item(
        (),
        coder_output=(
            "Implemented the change.\n"
            "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->\n\n"
            "### Human requirements\n- Requirement 1: done\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
        ),
        human_requirements=(requirement,),
        source_round=2,
    )
    assert len(reconciled) == 1
    assert "fresh acknowledgement" in reconciled[0].text


def test_duplicate_same_source_is_deduplicated_but_divergent_body_is_not():
    requirement = _requirement(
        body="Keep the API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/pull/2#issuecomment-1",
    )
    assert deduplicate_human_requirements((requirement, requirement)) == (requirement,)


def test_requirement_truncation_preserves_stable_acknowledgement_ids():
    requirements = tuple(
        _requirement(
            body=f"Requirement body {index}.",
            created_at=f"2026-01-0{index}T00:00:00Z",
            url=f"https://github.com/OWNER/REPO/issues/1#issuecomment-{index}",
        )
        for index in (1, 2, 3)
    )
    full = format_human_requirements(requirements)
    bounded = format_human_requirements(requirements, max_chars=len(full) - 1)
    assert f"Requirement {requirements[0].requirement_id}:" not in bounded
    assert f"Requirement {requirements[1].requirement_id}:" in bounded
    assert f"Requirement {requirements[2].requirement_id}:" in bounded


def test_duplicate_url_body_with_api_metadata_differences_is_deduplicated():
    first = _requirement(
        body="Keep the API stable.",
        created_at="2026-01-01T00:00:00Z",
        url="https://github.com/OWNER/REPO/pull/2#issuecomment-1",
    )
    mirrored = replace(first, source_type="PR review", author="api-mirror")
    assert len(deduplicate_human_requirements((first, mirrored))) == 1


def test_plan_recovery_selects_expected_hash_not_newest_plan():
    old_plan = "Approved plan\n\n### Scope\n- Preserve the API."
    later_unrelated = "Unrelated plan\n\n### Scope\n- Rewrite the API."
    comments = (
        IssueComment(
            author="coder",
            created_at="2026-01-01T00:00:00Z",
            body=_attach_round_metadata(
                old_plan,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Codex",
                    round_number=1,
                    subject="old",
                    canonical_plan=old_plan,
                ),
            ),
        ),
        IssueComment(
            author="coder",
            created_at="2026-01-02T00:00:00Z",
            body=_attach_round_metadata(
                later_unrelated,
                PostedRoundMetadata(
                    flow="plan",
                    role="coder",
                    agent="Codex",
                    round_number=1,
                    subject="later",
                    canonical_plan=later_unrelated,
                ),
            ),
        ),
    )
    expected = make_approved_plan_context(old_plan).plan_hash
    recovered = recover_approved_plan_context(comments, expected_hash=expected)
    assert recovered.is_available
    assert recovered.canonical_text == old_plan
    assert "Preserve the API" in recovered.canonical_text


def test_legacy_freeform_plan_recovery_reverses_signature_normalization():
    raw_plan = (
        "Approved free-form plan\n\n"
        "### Scope\n- Preserve the API.\n\n"
        "<!-- AGENT_PLAN_STATE: approved -->\n-- Anthropic Claude"
    )
    public_plan = normalize_freeform_signature(
        raw_plan,
        agent="claude",
        config=None,
        model_used="gpt-5.5 (medium)",
    )
    comment = IssueComment(
        author="coder",
        created_at="2026-01-01T00:00:00Z",
        body=_attach_round_metadata(
            public_plan,
            PostedRoundMetadata(
                flow="plan",
                role="coder",
                agent="Claude",
                round_number=1,
                subject=make_approved_plan_context(raw_plan).plan_subject,
                model_used="gpt-5.5 (medium)",
            ),
        ),
    )

    expected = make_approved_plan_context(raw_plan)
    recovered = recover_approved_plan_context(
        (comment,),
        expected_hash=expected.plan_hash,
        expected_subject=expected.plan_subject,
    )

    assert recovered.is_available
    assert recovered.canonical_text == raw_plan


def test_plan_identity_mismatch_is_explicit_and_raw_text_is_not_silently_used():
    context = make_approved_plan_context(
        "Approved plan\n\n### Scope\n- Preserve the API.",
        expected_hash="0000000000000000",
    )
    assert context.availability == "mismatched"
    assert "does not match handoff hash" in context.diagnostic
    assert not context.is_available


def test_oversized_plan_uses_real_limit_and_never_drops_identity_or_declarations():
    from coding_review_agent_loop.prompts import format_approved_plan_context

    plan = make_approved_plan_context(
        "Approved plan\n\n### Scope\n- Preserve the API.\n\n### Implementation\n" + "x" * 5000
    )
    assert plan.canonical_text in format_approved_plan_context(plan)

    rendered = format_approved_plan_context(plan, max_chars=1000)
    assert plan.plan_hash in rendered
    assert "Preserve the API." in rendered
    assert "omitted" in rendered

    with pytest.raises(AgentLoopError, match="cannot fit the final provider prompt limit"):
        format_approved_plan_context(plan, max_chars=300)


def test_unplanned_review_has_explicit_no_plan_path(tmp_path):
    config = make_config(tmp_path)
    metadata = PullRequestMetadata(
        number=7,
        repo=config.repo,
        title="Ordinary PR",
        head_branch="feature",
        base_branch="main",
        head_sha="b" * 40,
        url="https://github.com/OWNER/REPO/pull/7",
        body="A direct PR with no planning provenance.",
    )
    prompt = build_review_prompt(
        7,
        1,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        issue_context=None,
    )
    assert "No approved plan is bound to this PR" in prompt


def test_full_and_compact_review_prompts_keep_plan_outside_issue_history(tmp_path):
    config = make_config(tmp_path, pr_review_context_mode="compact")
    plan = make_approved_plan_context(
        "Approved plan\n\n### Scope\n- Preserve the API.\n\n### Deferred work\n- Broader redesign."
    )
    metadata = PullRequestMetadata(
        number=7,
        repo=config.repo,
        title="Implementation",
        head_branch="feature",
        base_branch="main",
        head_sha="a" * 40,
        url="https://github.com/OWNER/REPO/pull/7",
        body="Fixes #1",
    )
    kwargs = dict(
        pr_number=7,
        round_number=2,
        config=config,
        reviewer="codex",
        pr_metadata=metadata,
        issue_context=_issue(1),
        approved_plan_context=plan,
    )
    full = build_review_prompt(**kwargs)
    compact = build_review_prompt(**kwargs, compact_context=True)
    for prompt in (full, compact):
        assert "Approved implementation plan context" in prompt
        assert plan.plan_hash in prompt
        assert "Preserve the API" in prompt
        assert "Broader redesign" in prompt
    assert "Canonical approved plan text" in full
    assert "Canonical approved plan text" in compact
