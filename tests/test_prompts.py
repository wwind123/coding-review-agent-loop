import pytest

from agent_loop_helpers import *  # noqa: F403

_EXPECTED_UNRESOLVED_ITEMS_GUIDANCE = """Prior unresolved items are present. Disposition every listed
item in the JSON `prior_item_dispositions` array — do not add a separate prose section
or bullet list for prior items outside the JSON object. Valid `disposition` values:
- `"resolved"` — item is fixed in this PR.
- `"blocking"` — item is still a merge-blocking problem; include it in `blocking_items`.
- `"same-pr"` — item needs to be fixed before merge; include it in `same_pr_followups`.
- `"future"` — deferred; must include a `"note"` field; only valid when approving.

Only use `"future"` when returning `approved`. If an item should still be fixed before
merge, use `"blocking"` or `"same-pr"` instead. For any prior item already marked
`"future"`, explicitly re-evaluate: `"resolved"`, still `"future"`, or back to
`"blocking"`/`"same-pr"`.
"""

_EXPECTED_FOLLOWUP_IGNORE = """Do not include Same-PR follow-ups, Future follow-ups, or legacy
Non-blocking follow-ups sections in approved reviews; this run is configured to
ignore approved-review follow-up sections. Mark the review blocking instead
when cleanup should be fixed before merge.
Trivial style nits in touched code should be omitted unless worth requiring
before merge; if required, they are current-PR work and the review is blocking,
not approved with Future follow-ups.
"""

_EXPECTED_FOLLOWUP_FIX_AND = """For small, localized, low-risk cleanup that must still be fixed in this PR
before merge, return `<!-- AGENT_STATE: blocking -->` and list those items
under this exact heading:

### Same-PR follow-ups

Use Same-PR follow-ups only for narrow current-PR cleanup in files already
touched by this PR or directly adjacent code. This includes indentation or
style cleanup in touched code when it is worth requiring before merge, and
duplicated helper or prompt wording introduced by this PR. Do not use this
section for larger redesigns, broad refactors, or independent future work.
Keep `blocking_items` and `same_pr_followups` mutually exclusive. Use
`blocking_items` for defects, missing requirements, regressions, security
issues, or consistency gaps that make the PR not merge-ready. Use
`same_pr_followups` only for small localized cleanup that should be handled in
this PR but is not itself the reason the PR is blocked.
Same-PR follow-ups may appear only in blocking reviews. If any Same-PR
follow-up remains, including a carried-forward prior item that stays
`still blocking` or `same-pr`, the review is not approved yet.

If the PR is otherwise fully complete for this round but you notice substantial
work that is better handled separately in a future issue or PR, list at most
three highest-value items under this exact heading:

### Future follow-ups

Use Future follow-ups only for independent later work that is not necessary for
this PR to be merge-ready, such as a broader scaling or performance refinement
for very large histories. Do not put small cleanup in touched or directly
adjacent code under Future follow-ups; classify it as Same-PR if it should be
done before merge, otherwise omit it.
Approved means there are no blocking issues, no Same-PR follow-ups, and no
carried-forward prior unresolved items left active for this round.
Same-PR follow-ups will be sent back to Claude and require another review
round before final approval. Before returning approved, self-check that no
Future follow-up is trivial or local to the current PR; reclassify it as
Same-PR or omit it. One-line correctness fixes (e.g. initialising a needed variable)
belong in `blocking_items`; one-line cleanup (e.g. renaming for clarity, adding an
annotation) belongs in `same_pr_followups`. Neither belongs in `future_followups`.
Do not put trivial style nits in either follow-up section.
If you return `<!-- AGENT_STATE: blocking -->`, do not use structured Future
follow-ups; keep all required current-round work in the blocking review so it
is not missed during revision.
"""

_EXPECTED_FOLLOWUP_ELSE = """If you approve but notice substantial work that is better handled separately in
a future issue or PR, list at most three highest-value items under this exact
heading:

### Future follow-ups

Use Future follow-ups only for independent later work that is not necessary for
this PR to be merge-ready, such as a broader scaling or performance refinement
for very large histories. Do not put small cleanup in touched or directly
adjacent code under Future follow-ups. Indentation/style cleanup in touched
code should be omitted unless worth requiring before merge; duplicated helper
or prompt wording introduced by this PR should make the review blocking if it
must be fixed now.
Do not use the Same-PR follow-ups section in this mode; mark the review blocking
instead when small or local cleanup should be fixed before merge.
Before returning approved, self-check that no Future follow-up is trivial or
local to the current PR; reclassify it as blocking current-PR work or omit it.
One-line or trivially small mechanical fixes — whether a correctness issue (initialising a
needed variable) or cleanup (renaming for clarity, adding an annotation) — belong in the
current round as blocking work, not in a future issue.
The legacy heading `### Non-blocking follow-ups` is still accepted as future
follow-ups for compatibility, but prefer `### Future follow-ups`.
"""

def test_coder_prompts_include_assigned_workdir_rule(tmp_path):
    config = make_config(tmp_path, coder="codex")
    assigned = str(config.codex_dir.resolve())

    prompts = [
        build_issue_prompt(56, config),
        build_issue_plan_prompt(56, config),
        build_issue_implementation_prompt(56, "1. Fix it.", config),
        build_task_prompt("Fix the bug.", config),
        build_followup_prompt(77, 1, "Needs tests.", config),
        build_same_pr_followup_prompt(77, 1, "Tighten docs.", config),
    ]

    for prompt in prompts:
        assert f"Assigned checkout: `{assigned}`" in prompt
        assert "`AGENT_LOOP_WORKDIR` is set to this path" in prompt
        assert "must stay in that directory" in prompt
        assert "Do not `cd` into sibling, home, deployment, or duplicate clones" in prompt
        assert "`pwd` and `git status --branch --short`" in prompt


def test_issue_prompts_include_salvage_guardrail_block(tmp_path):
    config = make_config(tmp_path)
    salvage_summary = (
        "# Salvage Summary\n\n"
        "The previous attempt failed.\n\n"
        "- Partial patch: `/tmp/coding-review-agent-loop/logs/salvage/run/partial.patch`"
    )

    prompts = [
        build_issue_prompt(56, config, salvage_summary=salvage_summary),
        build_issue_implementation_prompt(
            56,
            "1. Implement the approved plan.",
            config,
            salvage_summary=salvage_summary,
        ),
    ]

    for prompt in prompts:
        assert "Previous failed implementation attempt salvage:" in prompt
        assert "The previous attempt failed." in prompt
        assert "incomplete context only" in prompt
        assert "Do not infer a successful\nresponse or PR" in prompt
        assert "Do not auto-apply the patch" in prompt
        assert "cherry-pick or ignore it\nselectively" in prompt
        assert "partial.patch" in prompt


def test_issue_prompts_render_remote_salvage_summary_shape(tmp_path):
    from coding_review_agent_loop.salvage import RemoteSalvageRecord, render_remote_salvage_summary

    config = make_config(tmp_path)
    record = RemoteSalvageRecord(
        created_at_ns=1,
        repo="OWNER/REPO",
        issue_number=56,
        scope="issue-implementation",
        agent="claude",
        run_id="run-1",
        approved_plan_hash=None,
        failure_category="transient",
        failure_reason="agent failed on a different machine",
        changed_files=" M file.txt",
        diff_stat=" file.txt | 2 +-",
        local_directory="/tmp/other-workdir/.agent-loop-logs/salvage/run-1-claude-issue-implementation",
        patch_exists=True,
        patch_included=True,
        patch_text="diff --git a/file.txt b/file.txt\n--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-old\n+new\n",
    )
    salvage_summary = render_remote_salvage_summary(record)

    prompt = build_issue_prompt(56, config, salvage_summary=salvage_summary)

    assert "Previous failed implementation attempt salvage:" in prompt
    assert "recovered from a GitHub issue comment" in prompt
    assert "agent failed on a different machine" in prompt
    assert "```diff" in prompt
    assert "-old" in prompt and "+new" in prompt
    assert "Do not auto-apply the patch" in prompt
    assert "may not exist here" in prompt


def test_reviewer_prompts_use_reviewer_assigned_workdir_rule(tmp_path):
    config = make_config(
        tmp_path,
        coder="claude",
        reviewer=("codex",),
        claude_args=(),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
    )
    reviewer_assigned = str(config.codex_dir.resolve())
    coder_assigned = str(config.claude_dir.resolve())

    prompts = [
        build_plan_review_prompt(56, 1, "Plan.", config, reviewer="codex"),
        build_plan_review_prompt(
            56,
            1,
            "Plan.",
            config,
            reviewer="codex",
            compact_context=True,
        ),
        build_review_prompt(77, 1, config, reviewer="codex"),
        build_review_prompt(77, 1, config, reviewer="codex", compact_context=True),
    ]

    for prompt in prompts:
        assert f"Assigned checkout: `{reviewer_assigned}`" in prompt
        assert f"Assigned checkout: `{coder_assigned}`" not in prompt
        assert "Inspection must stay in that directory" in prompt
        assert "Dangerous agent permissions are active" in prompt

def test_review_prompt_includes_prior_unresolved_items_and_disposition_instructions(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Rename the helper before merge.",
                status="same-pr",
            ),
        ),
    )

    assert "Prior unresolved review items from earlier rounds" in prompt
    assert "[item-1] blocking from Anthropic Claude in round 1" in prompt
    assert "[item-2] same-pr from OpenAI Codex in round 1" in prompt
    assert "### Prior unresolved item dispositions" not in prompt
    assert 'Disposition every listed\nitem in the JSON `prior_item_dispositions` array' in prompt
    assert '"resolved"' in prompt
    assert '"blocking"' in prompt
    assert '"same-pr"' in prompt
    assert '"future"' in prompt
    assert "do not add a separate prose section" in prompt
    assert "Only items listed under `Prior unresolved review items from earlier rounds`" in prompt
    assert "same-round findings from\nother reviewers appear elsewhere in the PR discussion" in prompt
    assert "Same-PR follow-ups may appear only in blocking reviews." in prompt
    assert "no blocking issues, no Same-PR follow-ups, and no" in prompt
    assert '"kind": "pr_review"' in prompt
    assert "After the JSON object, include only:" in prompt
    assert "Use this mandatory structured PR review format" in prompt
    assert "Markdown fallback" not in prompt

def test_compact_review_prompt_excludes_prose_disposition_heading(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        compact_context=True,
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.",
                status="blocking",
            ),
        ),
    )
    assert "### Prior unresolved item dispositions" not in prompt
    assert 'Disposition every listed\nitem in the JSON `prior_item_dispositions` array' in prompt
    assert "do not add a separate prose section" in prompt

def test_review_prompt_indents_multiline_prior_unresolved_item_text(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=PullRequestMetadata(
            number=77,
            repo="OWNER/REPO",
            title="Improve review prompt context",
            head_branch="feature/review-context",
            base_branch="main",
            head_sha="abc123",
            url="https://github.com/OWNER/REPO/pull/77",
        ),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Needs a regression test before merge.\n\nInclude the mixed-reviewer approval case.",
                status="blocking",
            ),
        ),
    )

    assert "  Needs a regression test before merge." in prompt
    assert "\n\n  Include the mixed-reviewer approval case." in prompt

def test_plan_review_prompt_includes_structured_sections_and_prior_items(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_review_prompt(
        56,
        2,
        "Revise protocol parsing and add tests.",
        config,
        reviewer="codex",
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Define exact plan-review headings.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Add an orchestration carry-forward test.",
                status="same-plan",
            ),
        ),
    )

    assert "### Prior unresolved plan item dispositions" not in prompt
    assert "[item-1] blocking from Anthropic Claude in round 1" in prompt
    assert "[item-2] same-plan from Google Gemini in round 1" in prompt
    assert 'Disposition every listed item\nin the JSON `prior_plan_item_dispositions` array' in prompt
    assert '"resolved"' in prompt
    assert '"blocking"' in prompt
    assert '"same-plan"' in prompt
    assert "do not add a separate prose section" in prompt
    assert "Only items listed under `Prior unresolved plan items from earlier rounds`" in prompt
    assert "same-round findings from other\nreviewers appear elsewhere in the issue discussion" in prompt
    assert "Same-plan\nfollow-ups are small current-plan refinements" in prompt
    assert "must be incorporated before\nimplementation starts" in prompt
    assert "they may appear only in blocking plan reviews" in prompt
    assert "Future\nfollow-ups are independent later work" in prompt
    assert "A concern or\nparaphrase belongs in exactly one current-round list" in prompt
    assert "Do not duplicate or reclassify\nthe same concern across Same-plan and Future follow-up lists" in prompt
    assert "do not use structured Future\nfollow-ups" in prompt
    assert "no blocking plan issues, no Same-plan\nfollow-ups, and no carried-forward plan items left active" in prompt
    assert '"kind": "plan_review"' in prompt
    assert '"prior_plan_item_dispositions"' in prompt
    assert "Use this mandatory structured JSON response format" in prompt
    assert "markdown compatibility" not in prompt.lower()

def test_compact_plan_review_prompt_excludes_prose_disposition_heading(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_review_prompt(
        56,
        2,
        "Fix the caching layer.",
        config,
        reviewer="codex",
        compact_context=True,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="Anthropic Claude",
                source_round=1,
                text="Define exact plan-review headings.",
                status="blocking",
            ),
        ),
    )
    assert "### Prior unresolved plan item dispositions" not in prompt
    assert 'Disposition every listed item\nin the JSON `prior_plan_item_dispositions` array' in prompt
    assert "do not add a separate prose section" in prompt

def test_plan_revision_prompt_includes_unresolved_ledger_and_required_dispositions(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_plan_revision_prompt(
        56,
        2,
        "Previous plan text.",
        "OpenAI Codex plan review:\n\nNeeds a carry-forward ledger.",
        config,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-3",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Track unresolved plan items across rounds.",
                status="blocking",
            ),
        ),
    )

    assert "Prior unresolved plan items from earlier rounds" in prompt
    assert "[item-3] blocking from OpenAI Codex in round 1" in prompt
    assert "### Prior plan review item dispositions" in prompt
    assert "- [item-id] same-plan:" in prompt
    assert "Use `same-plan`, never `same-pr`" in prompt
    assert '"kind": "plan_revision"' in prompt
    assert '"plan_steps"' in prompt
    assert "normalize structured plan revisions into canonical\nmarkdown for stored plan state" in prompt
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "### Human requirements" in prompt
    assert "after the JSON object and before the `AGENT_PLAN_STATE` footer" in prompt
    assert "Use this mandatory structured JSON response format" in prompt
    assert "fall back to markdown" not in prompt.lower()

def test_non_compact_plan_revision_prompt_includes_workdir_guidance(tmp_path):
    """Non-compact build_plan_revision_prompt must include workdir guidance (issue #269)."""
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        1,
        "Previous plan text.",
        "Claude plan review:\n\nBlocking issue found.",
        config,
        compact_context=False,
    )

    assert "Assigned checkout:" in prompt
    assert "AGENT_LOOP_WORKDIR" in prompt

def test_compact_plan_revision_prompt_includes_workdir_guidance(tmp_path):
    """Compact build_plan_revision_prompt also includes workdir guidance (regression guard)."""
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        1,
        "Previous plan text.",
        "Claude plan review:\n\nBlocking issue found.",
        config,
        compact_context=True,
    )

    assert "Assigned checkout:" in prompt
    assert "AGENT_LOOP_WORKDIR" in prompt

def _compact_issue_context() -> IssueContext:
    return IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Add compact planning context",
        body="Original acceptance criteria: preserve requirements and known reproductions.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:00:00Z",
                body="UNRELATED RAW PRIOR COMMENT PROSE should not be replayed in compact mode.",
            ),
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:05:00Z",
                body="Unrelated future-only comment that was never elevated.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue comment",
                author="wwind123",
                created_at="2026-06-04T06:47:03Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                body="Compact mode must use a stable prefix and volatile tail.",
            ),
        ),
    )

def _compact_memory_context(tmp_path: Path) -> AgentMemoryContext:
    return AgentMemoryContext(
        memory_dir=tmp_path / "memory",
        current_commit="abc123",
        last_analyzed_commit="def456",
        changed_files=("src/coding_review_agent_loop/prompts.py",),
        repo_summary="Repo memory summary for compact prefix.",
        architecture_map=None,
        test_profile="Run `python -m pytest`.",
        toolchain=None,
    )

def test_config_and_cli_default_to_compact_planning_context(tmp_path):
    assert make_config(tmp_path).planning_context_mode == "compact"

    parser = build_parser()
    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--claude-dir",
        str(tmp_path / "claude"),
        "--codex-dir",
        str(tmp_path / "codex"),
    ])
    assert config_from_args(args, FakeRunner()).planning_context_mode == "compact"

    args = parser.parse_args([
        "task",
        "Fix the bug",
        "--repo",
        "OWNER/REPO",
        "--planning-context-mode",
        "full",
    ])
    assert config_from_args(args, FakeRunner()).planning_context_mode == "full"

    with pytest.raises(AgentLoopError):
        make_config(tmp_path, planning_context_mode="invalid")
    with pytest.raises(SystemExit):
        parser.parse_args(["task", "Fix", "--planning-context-mode", "invalid"])

def test_compact_plan_review_prompt_preserves_canonical_context_and_omits_raw_prose(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    long_reproduction = "Known reproduction: " + ("run plan loop after idle gap. " * 80)
    prompt = build_plan_review_prompt(
        56,
        2,
        "Current plan payload.",
        config,
        reviewer="codex",
        memory=None,
        issue_context=_compact_issue_context(),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="OpenAI Codex",
                source_round=1,
                text=long_reproduction,
                status="blocking",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(
            (
                "[item-2] resolved: Google Gemini same-plan item from round 1\n"
                "Original item text:\nPrior resolved item full text.\n"
                "Disposition updates:\n- Google Gemini: resolved: covered by tests.",
                "[item-3] future follow-up: OpenAI Codex blocking item from round 1\n"
                "Original item text:\nPrior future item text.",
            )
        ),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Review action."),
    )

    prefix, tail = prompt.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert "Original acceptance criteria: preserve requirements" in prefix
    assert "Compact mode must use a stable prefix and volatile tail." in prefix
    assert "Known reproduction: run plan loop after idle gap." in prefix
    assert "run plan loop after idle gap. run plan loop after idle gap. run plan loop after idle gap." in prefix
    assert "[item-2] resolved" in prefix
    assert "Prior resolved item full text." in prefix
    assert "[item-3] future follow-up" in prefix
    assert "UNRELATED RAW PRIOR COMMENT PROSE" not in prompt
    assert "Unrelated future-only comment that was never elevated." not in prompt
    assert "Current plan payload." in tail
    assert "Planning round: 2" in tail
    assert "subject-a" in tail

def test_compact_plan_revision_prompt_preserves_context_and_omits_raw_prose(tmp_path):
    config = make_config(tmp_path)
    prompt = build_plan_revision_prompt(
        56,
        2,
        "Previous plan payload.",
        "Blocking review payload.",
        config,
        memory=None,
        issue_context=_compact_issue_context(),
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-4",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Known reproduction: resume after subject change drops resolved item.",
                status="same-plan",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-5] resolved: stable old item text",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Revision action."),
    )

    prefix, tail = prompt.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert "Original acceptance criteria: preserve requirements" in prefix
    assert "Compact mode must use a stable prefix and volatile tail." in prefix
    assert "Known reproduction: resume after subject change" in prefix
    assert "[item-5] resolved: stable old item text" in prefix
    assert "UNRELATED RAW PRIOR COMMENT PROSE" not in prompt
    assert "Previous plan payload." in tail
    assert "Blocking review payload." in tail
    assert "Planning round: 2" in tail
    assert "subject-b" in tail
    assert "<!-- HUMAN_REQUIREMENTS_ADDRESSED -->" in prompt
    assert "### Human requirements" in prompt
    assert "after the JSON object and before the `AGENT_PLAN_STATE` footer" in prompt

def test_full_plan_prompt_still_includes_raw_issue_comments(tmp_path):
    config = make_config(tmp_path)
    prompt = build_plan_review_prompt(
        56,
        2,
        "Current plan payload.",
        config,
        reviewer="codex",
        issue_context=_compact_issue_context(),
    )

    assert "UNRELATED RAW PRIOR COMMENT PROSE" in prompt

def _compact_pr_issue_context() -> IssueContext:
    return IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Linked issue title",
        body="Linked issue body with durable requirements.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="reviewer",
                created_at="2026-06-01T00:00:00Z",
                body="UNRELATED RAW PRIOR PR REVIEW HISTORY should not be replayed.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="wwind123",
                created_at="2026-06-04T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Compact PR mode must preserve human requirements.",
            ),
        ),
    )

def _compact_pr_metadata() -> PullRequestMetadata:
    return PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Compact PR context",
        head_branch="feature/context",
        base_branch="main",
        head_sha="abc123",
        url="https://github.com/OWNER/REPO/pull/77",
        body="PR body with author intent and scope.",
    )

def test_config_and_cli_default_to_full_pr_review_context(tmp_path):
    assert make_config(tmp_path).pr_review_context_mode == "full"

    parser = build_parser()
    args = parser.parse_args([
        "pr",
        "77",
        "--repo",
        "OWNER/REPO",
        "--pr-review-context-mode",
        "compact",
    ])
    assert config_from_args(args, FakeRunner()).pr_review_context_mode == "compact"

    with pytest.raises(AgentLoopError):
        make_config(tmp_path, pr_review_context_mode="invalid")
    with pytest.raises(SystemExit):
        parser.parse_args(["pr", "77", "--pr-review-context-mode", "invalid"])

def test_compact_pr_review_prompt_preserves_context_and_omits_raw_history(tmp_path):
    config = make_config(
        tmp_path,
        approved_followups="fix-and-summarize",
        reviewer=("codex", "gemini"),
    )
    issue_context = _compact_pr_issue_context()
    prompt = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=_compact_pr_metadata(),
        pr_checks=None,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(
            UnresolvedReviewItem(
                item_id="item-1",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Active blocking issue remains important.",
                status="blocking",
            ),
            UnresolvedReviewItem(
                item_id="item-2",
                reviewer="Google Gemini",
                source_round=1,
                text="Active same-PR cleanup remains important.",
                status="same-pr",
            ),
            UnresolvedReviewItem(
                item_id="item-3",
                reviewer="OpenAI Codex",
                source_round=1,
                text="Unrelated future-only item should not stay active in compact prompt.",
                status="future",
            ),
        ),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-4] resolved: old resolved item",)),
        compact_coder_summary="Coder says the compact mode wiring is complete.",
        compact_coder_tests_run=("python -m pytest tests/test_agent_loop.py -k compact_pr",),
        compact_tail=CompactPrReviewTailContext(
            head_sha="abc123",
            round_number=2,
            action="Review compact PR context mode.",
        ),
    )

    prefix, tail = prompt.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    assert "PR body with author intent and scope." in prefix
    assert "Linked issue body with durable requirements." in prefix
    assert "Compact PR mode must preserve human requirements." in prefix
    assert "Repo memory summary for compact prefix." in prefix
    assert "Active blocking issue remains important." in prefix
    assert "Active same-PR cleanup remains important." in prefix
    assert "Unrelated future-only item should not stay active" not in prefix
    assert "UNRELATED RAW PRIOR PR REVIEW HISTORY" not in prompt
    assert "[item-4] resolved: old resolved item" in prefix
    assert "Coder says the compact mode wiring is complete." in prefix
    assert "python -m pytest tests/test_agent_loop.py -k compact_pr" in prefix
    assert "Use Future follow-ups only for independent later work" in prefix
    assert "broader scaling or performance refinement\nfor very large histories" in prefix
    assert "indentation or\nstyle cleanup in touched code" in prefix
    assert "duplicated helper or prompt wording introduced by this PR" in prefix
    assert "Before approving, self-check every `future_followups` entry" in prefix
    assert "Review compact PR context mode." in tail
    assert "Head SHA: abc123" in tail
    assert "Read the verified checkout and local base-to-head diff first." in tail
    assert "gh pr diff" not in tail

def test_compact_pr_review_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    metadata = _compact_pr_metadata()
    issue_context = _compact_pr_issue_context()
    unresolved_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active PR item remains approval-critical.",
        status="blocking",
    )
    first = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-9] resolved: unchanged",)),
        compact_coder_summary="Stable coder summary.",
        compact_coder_tests_run=("pytest -k stable",),
        compact_tail=CompactPrReviewTailContext(head_sha="abc123", round_number=2, action="Review A."),
    )
    second = build_review_prompt(
        77,
        3,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        memory=_compact_memory_context(tmp_path),
        issue_context=issue_context,
        human_requirements=issue_context.human_requirements,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-9] resolved: unchanged",)),
        compact_coder_summary="Stable coder summary.",
        compact_coder_tests_run=("pytest -k stable",),
        compact_tail=CompactPrReviewTailContext(head_sha="def456", round_number=3, action="Review B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Active PR item remains approval-critical." in first_prefix
    assert "Stable coder summary." in first_prefix
    assert "PR body with author intent and scope." in first_prefix
    for volatile in ("round 2", "Head SHA: abc123", "Review A."):
        assert volatile not in first_prefix
        assert volatile in first_tail

def test_compact_review_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    issue_context = _compact_issue_context()
    memory = _compact_memory_context(tmp_path)
    unresolved_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active ledger item remains approval-critical.",
        status="blocking",
    )
    first = build_plan_review_prompt(
        56,
        2,
        "Plan tail A.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Review A."),
    )
    second = build_plan_review_prompt(
        56,
        3,
        "Plan tail B.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Review B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Plan review response protocol" in first_prefix
    assert '"kind": "plan_review"' in first_prefix
    assert "Repo memory summary for compact prefix." in first_prefix
    assert "Original acceptance criteria: preserve requirements" in first_prefix
    assert "Compact mode must use a stable prefix and volatile tail." in first_prefix
    assert "Active ledger item remains approval-critical." in first_prefix
    assert "[item-1] resolved: unchanged" in first_prefix
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in first_prefix
    for volatile in ("Planning round: 2", "Plan tail A.", "subject-a", "Review A."):
        assert volatile not in first_prefix
        assert volatile in first_tail

def test_compact_revision_prompt_stable_prefix_is_byte_identical_across_rounds(tmp_path):
    config = make_config(tmp_path)
    issue_context = _compact_issue_context()
    memory = _compact_memory_context(tmp_path)
    unresolved_item = UnresolvedReviewItem(
        item_id="item-2",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active revision ledger item remains approval-critical.",
        status="same-plan",
    )
    first = build_plan_revision_prompt(
        56,
        2,
        "Previous plan A.",
        "Review A.",
        config,
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Revision A."),
    )
    second = build_plan_revision_prompt(
        56,
        3,
        "Previous plan B.",
        "Review B.",
        config,
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-1] resolved: unchanged",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Revision B."),
    )

    first_prefix, first_tail = first.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    second_prefix, second_tail = second.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert first_tail != second_tail
    assert "Plan revision response protocol" in first_prefix
    assert '"kind": "plan_revision"' in first_prefix
    assert "Repo memory summary for compact prefix." in first_prefix
    assert "Original acceptance criteria: preserve requirements" in first_prefix
    assert "Compact mode must use a stable prefix and volatile tail." in first_prefix
    assert "Active revision ledger item remains approval-critical." in first_prefix
    assert "[item-1] resolved: unchanged" in first_prefix
    for volatile in ("Planning round: 2", "Previous plan A.", "Review A.", "subject-a", "Revision A."):
        assert volatile not in first_prefix
        assert volatile in first_tail

def test_issue_loop_includes_issue_comments_in_coder_and_review_prompts(tmp_path):
    runner = FakeRunner(
        issue_payload={
            "number": 56,
            "state": "open",
            "is_pr": False,
            "url": "https://github.com/OWNER/REPO/issues/56",
            "title": "Support issue comments",
            "body": "Original request.",
        },
        issue_comments=[
            {
                "author": {"login": "second-user"},
                "createdAt": "2026-05-17T10:00:00Z",
                "body": "Later comment should come second.",
            },
            {
                "author": {"login": "first-user"},
                "createdAt": "2026-05-17T09:00:00Z",
                "body": "Earlier comment refines the request.",
            },
        ],
        claude_outputs=[
            "Created PR.\n<!-- AGENT_PR: 77 -->\n<!-- AGENT_STATE: blocking -->",
        ],
        codex_outputs=[
            "LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        pr_payload={"body": "Fixes #56"},
    )
    config = make_config(tmp_path)

    assert run_issue_loop(runner, issue_number=56, config=config) == 0

    claude_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"])
    codex_prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    for prompt in (claude_prompt, codex_prompt):
        assert "Issue context from GitHub" in prompt
        assert "Later comments may refine or supersede the original issue body" in prompt
        assert "GitHub issue #56" in prompt
        assert "Title:\nSupport issue comments" in prompt
        assert "Body:\nOriginal request." in prompt
        assert "Comments, oldest to newest:" in prompt
        assert prompt.index("Comment by first-user at 2026-05-17T09:00:00Z") < prompt.index(
            "Comment by second-user at 2026-05-17T10:00:00Z"
        )
        assert "Earlier comment refines the request." in prompt
        assert "Later comment should come second." in prompt
    assert "include `Fixes #56` or another direct reference to issue #56" in claude_prompt

def test_format_issue_context_truncates_oversized_newest_comment():
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="first-user",
                created_at="2026-05-17T09:00:00Z",
                body="Older detail should not be kept instead of the newest comment.",
            ),
            IssueComment(
                author="second-user",
                created_at="2026-05-17T10:00:00Z",
                body="Newest detail. " + ("x" * 1000),
            ),
        ),
    )

    text = format_issue_context(issue_context, max_chars=700)

    assert len(text) <= 700
    assert "Older comments omitted: 1 comment(s)" in text
    assert "Comment by second-user at 2026-05-17T10:00:00Z" in text
    assert "Newest detail." in text
    assert "[Newest comment truncated to keep this prompt bounded.]" in text
    assert "Older detail should not be kept instead of the newest comment." not in text


def test_format_issue_context_filters_out_agent_salvage_breadcrumb_comments():
    salvage_comment_body = (
        "### Implementation salvage breadcrumb\n\n"
        "<!-- AGENT_SALVAGE: eyJzY2hlbWFfdmVyc2lvbiI6IDF9 -->\n"
    )
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="human-user",
                created_at="2026-05-17T09:00:00Z",
                body="A real discussion comment that must stay visible.",
            ),
            IssueComment(
                author="coding-review-agent-loop",
                created_at="2026-05-17T10:00:00Z",
                body=salvage_comment_body,
            ),
        ),
    )

    text = format_issue_context(issue_context)

    assert "A real discussion comment that must stay visible." in text
    assert "AGENT_SALVAGE" not in text
    assert "Implementation salvage breadcrumb" not in text

def test_format_human_requirements_uses_distinct_high_priority_section():
    text = format_human_requirements(
        (
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        )
    )

    assert text.startswith("Signed Human Reviewer Requirements")
    assert "high-priority PR requirements" in text
    assert "latest human instruction wins" in text
    assert "- Source: PR comment" in text
    assert "- Author: reviewer" in text
    assert "Please use the absolute URL." in text

def test_format_human_requirements_supports_issue_specific_wording_and_fallback():
    text = format_human_requirements(
        (
            HumanReviewRequirement(
                source_type="Issue body",
                author="maintainer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Keep the current CLI flag.",
            ),
        ),
        max_chars=120,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
    )

    assert "high-priority planning requirements" in text
    assert "Fetch the issue discussion directly before finalizing the plan." in text

def test_format_human_requirements_preserves_entry_spacing_when_truncated():
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Oldest requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T11:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            body="Middle requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T12:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-3",
            body="Newest requirement.",
        ),
    )
    full_text = format_human_requirements(requirements)

    text = format_human_requirements(requirements, max_chars=len(full_text) - 1)

    assert "Older signed human requirement(s) omitted: 1." in text
    assert "Oldest requirement." not in text
    assert "Middle requirement.\n\nRequirement 3:" in text
    assert "Newest requirement." in text

def test_render_coder_human_requirements_prompt_context_tracks_surfaced_ids_after_truncation():
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Oldest requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T11:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-2",
            body="Middle requirement.",
        ),
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T12:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-3",
            body="Newest requirement.",
        ),
    )
    full_text = format_human_requirements(requirements)

    context = render_coder_human_requirements_prompt_context(
        requirements,
        max_chars=len(full_text) - 1,
    )

    assert context.block.endswith("\n")
    assert "Older signed human requirement(s) omitted: 1." in context.block
    assert context.surfaced_requirement_ids == ("Requirement 2", "Requirement 3")
    assert context.requires_direct_discussion_ack is False

def test_render_coder_human_requirements_prompt_context_handles_full_omission_fallback():
    context = render_coder_human_requirements_prompt_context(
        (
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        ),
        max_chars=120,
    )

    assert "All 1 signed human requirement(s) were omitted" in context.block
    assert context.surfaced_requirement_ids == ()
    assert context.requires_direct_discussion_ack is True

@pytest.mark.parametrize(
    ("builder_name", "expected_scope", "expected_guidance"),
    [
        ("issue", "high-priority implementation requirements", "how you addressed that item"),
        ("issue_plan", "high-priority planning requirements", "how the plan covers that item"),
        ("plan_revision", "high-priority planning requirements", "how the revised plan covers that item"),
        (
            "issue_implementation",
            "high-priority implementation requirements",
            "how you addressed that item",
        ),
    ],
)
def test_issue_and_plan_prompts_surface_signed_human_requirements_before_issue_context(
    tmp_path,
    builder_name,
    expected_scope,
    expected_guidance,
):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="General issue context.",
            ),
        ),
        human_requirements=(
            HumanReviewRequirement(
                source_type="Issue comment",
                author="maintainer",
                created_at="2026-05-17T11:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56#issuecomment-1",
                body="Preserve backward compatibility.",
            ),
        ),
    )
    if builder_name == "issue":
        prompt = build_issue_prompt(56, config, issue_context=issue_context)
    elif builder_name == "issue_plan":
        prompt = build_issue_plan_prompt(56, config, issue_context=issue_context)
    elif builder_name == "plan_revision":
        prompt = build_plan_revision_prompt(
            56,
            2,
            "Old plan.",
            "Blocking review.",
            config,
            issue_context=issue_context,
        )
    else:
        prompt = build_issue_implementation_prompt(
            56,
            "Approved plan.",
            config,
            issue_context=issue_context,
        )

    assert "Signed Human Reviewer Requirements" in prompt
    assert expected_scope in prompt
    assert expected_guidance in prompt
    assert prompt.index("Signed Human Reviewer Requirements") < prompt.index("Issue context from GitHub")

def test_followup_prompt_with_no_human_requirements_guides_empty_addressed_ids(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_followup_prompt(
        222,
        2,
        "item-1: Add a regression test.",
        config,
        human_requirements=(),
    )

    assert '"human_requirements": {' in prompt
    assert '"addressed_ids": []' in prompt
    assert '"addressed_ids": ["Requirement 1"]' not in prompt
    assert "No signed human requirements are surfaced in this prompt" in prompt
    assert "issue acceptance criteria" in prompt
    assert "reviewer item IDs" in prompt

def test_plan_review_prompt_surfaces_signed_issue_requirements_as_approval_critical(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
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
                author="maintainer",
                created_at="2026-05-17T08:00:00Z",
                url="https://github.com/OWNER/REPO/issues/56",
                body="Keep the public API unchanged.",
            ),
        ),
    )

    prompt = build_plan_review_prompt(
        56,
        1,
        "Plan:\n- Update the parser.",
        config,
        reviewer="codex",
        issue_context=issue_context,
    )

    assert "Signed Human Reviewer Requirements" in prompt
    assert "high-priority planning requirements" in prompt
    assert "approval-critical issue constraints" in prompt
    assert "Verify each requirement in this set before approving." in prompt
    assert prompt.index("Signed Human Reviewer Requirements") < prompt.index("Issue context from GitHub")

@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_require_human_requirements_acknowledgement_only_when_present(
    tmp_path, builder
):
    config = make_config(tmp_path)
    with_requirements = builder(
        77,
        2,
        "Fix the bug.",
        config,
        human_requirements=(
            HumanReviewRequirement(
                source_type="PR comment",
                author="reviewer",
                created_at="2026-05-18T10:00:00Z",
                url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
                body="Please use the absolute URL.",
            ),
        ),
    )
    without_requirements = builder(77, 2, "Fix the bug.", config)

    assert "mandatory next-revision requirements" in with_requirements
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in with_requirements
    assert "### Human requirements" in with_requirements
    assert "`Requirement 1`" in with_requirements
    assert "mandatory next-revision requirements" not in without_requirements
    assert "`Requirement 1`" not in without_requirements

@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_accept_precomputed_human_requirements_context(tmp_path, builder):
    config = make_config(tmp_path)
    requirements = (
        HumanReviewRequirement(
            source_type="PR comment",
            author="reviewer",
            created_at="2026-05-18T10:00:00Z",
            url="https://github.com/OWNER/REPO/pull/77#issuecomment-1",
            body="Please use the absolute URL.",
        ),
    )
    context = render_coder_human_requirements_prompt_context(requirements)

    prompt = builder(
        77,
        2,
        "Fix the bug.",
        config,
        human_requirements=requirements,
        human_requirements_context=context,
    )

    assert context.block in prompt
    assert HUMAN_REQUIREMENTS_ADDRESSED_MARKER in prompt
    assert "`Requirement 1`" in prompt

@pytest.mark.parametrize("builder", [build_followup_prompt, build_same_pr_followup_prompt])
def test_coder_followup_prompts_require_structured_json(tmp_path, builder):
    config = make_config(tmp_path)

    prompt = builder(77, 2, "Fix the bug.", config)

    assert '"kind": "coder_followup"' in prompt
    assert '"addressed_items": ["item-1"]' in prompt
    assert '"remaining_items": ["item-2"]' in prompt
    assert '"human_requirements": {' in prompt
    assert "The JSON `state` must match the `AGENT_STATE` footer exactly." in prompt
    assert "Use this mandatory structured JSON follow-up format" in prompt
    assert "compatibility fallback" not in prompt
    assert "Legacy markdown replies" not in prompt

def test_review_prompt_includes_pr_metadata_and_suggested_commands(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompts = [cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"]]
    assert len(prompts) == 1
    prompt = prompts[0]
    assert "PR metadata:" in prompt
    assert "- Repo: OWNER/REPO" in prompt
    assert "- PR: #77" in prompt
    assert "- Title: Improve review prompt context" in prompt
    assert "- Head branch: feature/review-context" in prompt
    assert "- Base branch: main" in prompt
    assert "- Head SHA: abc123" in prompt
    assert "Use this PR metadata as authoritative." in prompt
    assert "Do not spend time discovering the\nPR branch." in prompt
    assert "gh pr view 77 --repo OWNER/REPO --json comments,reviews" in prompt
    assert "`git diff main...HEAD`" in prompt
    assert "gh pr diff" not in prompt
    assert "If a read-only shell/tool command requires confirmation" in prompt
    assert "non-interactive\nmode" in prompt
    assert "write them outside the repository checkout" in prompt
    assert "/tmp/coding-review-agent-loop/scratch/" in prompt
    assert "GitHub PR checks:" in prompt
    assert "- Overall state: passing" in prompt
    assert "- Required checks: test" in prompt
    assert "Do not say or imply that tests passed globally unless the GitHub PR checks" in prompt
    assert "ignore approved-review follow-up sections" in prompt
    assert "### Future follow-ups" not in prompt
    assert "legacy heading `### Non-blocking follow-ups`" not in prompt
    assert "verify migration topology" in prompt
    assert "Use blocking only for issues that should prevent merge." in prompt

def test_review_prompt_includes_signed_human_requirements(tmp_path):
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
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Signed Human Reviewer Requirements" in prompt
    assert "Please use the absolute URL." in prompt
    assert "Signed human reviewer requirements override AI reviewer preferences" in prompt
    assert "Verify each requirement in this set before approving." in prompt
    assert "<!-- HUMAN_REQUIREMENTS_RESOLVED -->" in prompt

def test_review_prompt_includes_failing_github_check_status(tmp_path):
    runner = FakeRunner(
        codex_outputs=["Blocking.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex"],
        pr_check_runs_payload={
            "check_runs": [
                {"name": "tests/test_security.py", "status": "completed", "conclusion": "failure"}
            ]
        },
        pr_branch_protection_payload={"contexts": ["tests/test_security.py"]},
    )
    config = make_config(tmp_path, max_rounds=1)

    with pytest.raises(AgentLoopError, match="blocking issues after round 1"):
        run_pr_loop(runner, pr_number=77, config=config)

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "GitHub PR checks:" in prompt
    assert "- Overall state: failing" in prompt
    assert "- Failing checks: tests/test_security.py (failure)" in prompt

@pytest.mark.parametrize("compact_context", [False, True])
def test_review_prompt_distinguishes_failing_from_pending_check_blocking_policy(
    tmp_path, compact_context
):
    config = make_config(tmp_path)
    prompt = build_review_prompt(77, 1, config, reviewer="codex", compact_context=compact_context)
    assert "Do not defer your review to wait for CI" in prompt
    assert (
        "Failing GitHub checks may justify a blocking review and a `blocking_items` entry."
        in prompt
    )
    assert (
        "Pending or unavailable GitHub checks are an external wait state: mention them"
        in prompt
    )
    assert 'never as the sole reason to\nreturn `state: "blocking"`' in prompt

@pytest.mark.parametrize("compact_context", [False, True])
def test_review_prompt_allows_read_only_local_inspection_but_prohibits_execution(
    tmp_path, compact_context
):
    config = make_config(tmp_path)
    prompt = build_review_prompt(77, 1, config, reviewer="codex", compact_context=compact_context)
    assert "Do not defer your review to wait for CI" in prompt
    assert "`git diff main...HEAD`" in prompt
    for command in ("`git diff`", "`git show`", "`git status`", "`git log`", "`rg`", "`sed`"):
        assert command in prompt
    assert "Do not run tests, builds, compilation" in prompt
    assert "source or worktree mutation" in prompt
    assert "Do not fetch, checkout, reset, clean, write files" in prompt
    assert "DO NOT run tests, shell commands, or compile code" not in prompt
    assert "gh pr diff" not in prompt

@pytest.mark.parametrize("compact_context", [False, True])
def test_review_prompt_uses_safe_base_placeholder_when_base_is_unknown(
    tmp_path, compact_context
):
    config = make_config(tmp_path, base=None)
    metadata = PullRequestMetadata(
        number=77,
        repo="OWNER/REPO",
        title="Review prompt",
        head_branch="feature/review",
        base_branch=None,
        head_sha="abc123",
        url=None,
    )

    prompt = build_review_prompt(
        77,
        1,
        config,
        reviewer="codex",
        pr_metadata=metadata,
        compact_context=compact_context,
    )

    assert "`git diff <base>...HEAD`" in prompt
    assert "do not run the placeholder literally" in prompt

def test_review_prompt_mentions_branch_protection_forbidden_when_checks_exist(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        pr_check_runs_payload={
            "check_runs": [{"name": "test", "status": "completed", "conclusion": "success"}]
        },
        pr_branch_protection_returncode=1,
        pr_branch_protection_stderr="403 Forbidden",
    )
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Current GitHub token cannot inspect branch protection on the PR base branch." in prompt

def test_blocking_followup_prompt_reinjects_issue_context(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            structured_coder_followup(
                state="approved",
                addressed_items=["item-1"],
                remaining_items=[],
                reviewer="Anthropic Claude",
            )
        ],
    )
    config = make_config(tmp_path)
    issue_context = IssueContext(
        number=56,
        repo="OWNER/REPO",
        title="Support issue comments",
        body="Original request.",
        url="https://github.com/OWNER/REPO/issues/56",
        comments=(
            IssueComment(
                author="commenter",
                created_at="2026-05-17T10:00:00Z",
                body="Clarifying issue comment.",
            ),
        ),
    )

    assert run_pr_loop(runner, pr_number=77, config=config, issue_context=issue_context) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert "Issue context from GitHub" in followup_prompt
    assert "Title:\nSupport issue comments" in followup_prompt
    assert "Clarifying issue comment." in followup_prompt
    assert "Needs a fix." in followup_prompt

def test_blocking_followup_prompt_includes_human_requirements_before_ai_feedback(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Needs a fix.\n<!-- AGENT_STATE: blocking -->\n-- OpenAI Codex",
            "LGTM."
            + prior_item_dispositions("[item-1] resolved")
            + "\n<!-- HUMAN_REQUIREMENTS_RESOLVED -->\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Fixed review.\n"
            f"{HUMAN_REQUIREMENTS_ADDRESSED_MARKER}\n"
            "### Human requirements\n"
            "- Requirement 1: used the absolute URL.\n"
            "<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude"
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
    config = make_config(tmp_path)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    followup_prompt = next(
        cmd[-1] for cmd, _cwd in runner.commands if cmd[:1] == ["claude"] and "Address the review below" in cmd[-1]
    )
    assert followup_prompt.index("Signed Human Reviewer Requirements") < followup_prompt.index(
        "Codex unresolved blocking item [item-1] from round 1:"
    )
    assert "Please use the absolute URL." in followup_prompt
    assert "Needs a fix." in followup_prompt

def test_review_prompt_requests_future_followups_when_processed(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path, approved_followups="summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "### Future follow-ups" in prompt
    assert "legacy heading `### Non-blocking follow-ups`" in prompt
    assert "Do not use the Same-PR follow-ups section in this mode" in prompt
    assert "Use Future follow-ups only for independent later work" in prompt
    assert "broader scaling or performance refinement\nfor very large histories" in prompt
    assert "Do not put small cleanup in touched or directly\nadjacent code under Future follow-ups" in prompt
    assert "Indentation/style cleanup in touched\ncode should be omitted unless worth requiring before merge" in prompt
    assert "duplicated helper\nor prompt wording introduced by this PR should make the review blocking" in prompt
    assert "Before returning approved, self-check that no Future follow-up is trivial or\nlocal to the current PR" in prompt
    assert "Use blocking only for issues that should prevent merge." in prompt

def test_review_prompt_allows_same_pr_followups_for_fix_modes(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    config = make_config(tmp_path, approved_followups="fix-and-summarize")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "### Same-PR follow-ups" in prompt
    assert "### Future follow-ups" in prompt
    assert "small, localized, low-risk cleanup" in prompt
    assert "narrow current-PR cleanup in files already\ntouched by this PR or directly adjacent code" in prompt
    assert "indentation or\nstyle cleanup in touched code" in prompt
    assert "duplicated helper or prompt wording introduced by this PR" in prompt
    assert "Use Future follow-ups only for independent later work" in prompt
    assert "broader scaling or performance refinement\nfor very large histories" in prompt
    assert "Do not put small cleanup in touched or directly\nadjacent code under Future follow-ups" in prompt
    assert "Keep `blocking_items` and `same_pr_followups` mutually exclusive." in prompt
    assert (
        "Use\n`blocking_items` for defects, missing requirements, regressions, security\n"
        "issues, or consistency gaps that make the PR not merge-ready."
    ) in prompt
    assert (
        "Use\n`same_pr_followups` only for small localized cleanup that should be handled in\n"
        "this PR but is not itself the reason the PR is blocked."
    ) in prompt
    assert "Same-PR follow-ups may appear only in blocking reviews." in prompt
    assert "will be sent back to Claude and require another review" in prompt
    assert "Approved means there are no blocking issues, no Same-PR follow-ups, and no\ncarried-forward prior unresolved items left active" in prompt
    assert "Before returning approved, self-check that no\nFuture follow-up is trivial or local to the current PR" in prompt
    assert "If you return `<!-- AGENT_STATE: blocking -->`, do not use structured Future\nfollow-ups" in prompt
    assert (
        "`blocking_items`, `same_pr_followups`, and `future_followups` have distinct\n"
        "roles."
    ) in prompt
    assert "small, localized cleanup in touched files or directly adjacent code" in prompt
    assert "`future_followups` are independent later work that remains valid after this PR\nis merge-ready." in prompt
    assert "Before approving, self-check every `future_followups` entry" in prompt
    assert (
        "`blocking_items` are merge-blocking defects, missing requirements,\n"
        "regressions, security issues, or consistency gaps."
    ) in prompt

def test_plan_review_prompt_directs_one_liner_to_same_plan(tmp_path):
    config = make_config(tmp_path, reviewer=("codex",))
    for compact in (False, True):
        prompt = build_plan_review_prompt(
            56, 1, "Plan.", config, reviewer="codex", compact_context=compact
        )
        # Proves the guidance names future_followups as the excluded destination
        assert "Before adding a\n`future_followups` entry" in prompt
        # Proves the destination is same_plan_followups (not future)
        assert "same_plan_followups` instead" in prompt
        # Proves the classification-threshold text is present
        assert "trivially small\nmechanical change" in prompt

def test_review_prompt_one_liner_preserves_blocking_vs_same_pr_in_fix_and_mode(tmp_path):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    for compact in (False, True):
        prompt = build_review_prompt(77, 1, config, reviewer="codex", compact_context=compact)
        # Proves defect one-liners stay in blocking_items, cleanup goes to same_pr_followups
        assert "blocking_items`; one-line cleanup" in prompt
        # Proves explicit exclusion from future_followups
        assert "Neither belongs in `future_followups`" in prompt

def test_review_prompt_one_liner_excluded_from_future_in_standard_mode(tmp_path):
    config = make_config(tmp_path, approved_followups="summarize")
    for compact in (False, True):
        prompt = build_review_prompt(77, 1, config, reviewer="codex", compact_context=compact)
        # Proves exclusion from future follow-up issues
        assert "not in a future issue" in prompt
        # Proves the classification-threshold text is present
        assert "trivially small mechanical fixes" in prompt

def test_same_pr_followup_prompt_no_longer_claims_pr_was_approved(tmp_path):
    config = make_config(tmp_path)

    prompt = build_same_pr_followup_prompt(77, 2, "Rename the helper.", config)

    assert "requested same-PR follow-ups" in prompt
    assert "approved pull request" not in prompt
    assert "remains blocked pending another review round" in prompt

def test_agent_memory_is_created_and_added_to_review_prompt(tmp_path):
    runner = FakeRunner(codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"])
    memory_dir = tmp_path / "memory"
    config = make_config(tmp_path, agent_memory_dir=memory_dir)

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    assert (memory_dir / "repo-summary.md").exists()
    assert (memory_dir / "architecture-map.md").exists()
    assert (memory_dir / "module-index.json").exists()
    assert (memory_dir / "test-profile.md").exists()
    assert (memory_dir / "toolchain.json").exists()
    assert (memory_dir / "last-analyzed-commit").read_text(encoding="utf-8") == "abc123\n"
    architecture_map = (memory_dir / "architecture-map.md").read_text(encoding="utf-8")
    assert "## Top-level Layout" in architecture_map
    assert "## Python Modules" in architecture_map
    assert "## Supporting Surfaces" in architecture_map
    assert "`src/coding_review_agent_loop`" in architecture_map

    prompt = next(cmd[-1] for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    assert "Agent memory context:" in prompt
    assert "Use cached repo memory and execution memory only for orientation." in prompt
    assert "inspect the actual source files and PR diff directly" in prompt
    assert "Do not search the whole filesystem for test tools." in prompt
    assert "src/coding_review_agent_loop/cli.py" in prompt

def test_pr_loop_carries_new_future_followups_into_later_reviewer_prompts(tmp_path):
    runner = FakeRunner(
        codex_outputs=[
            "Codex approves with future work.\n\n"
            "### Future follow-ups\n"
            "- Document cache cleanup behavior.\n"
            "<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
            "Codex approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex",
        ],
        claude_outputs=[
            "Claude still blocks.\n<!-- AGENT_STATE: blocking -->\n-- Anthropic Claude",
            "Claude approves final pass."
            + prior_item_dispositions(
                "[item-1] future follow-up: still future work",
                "[item-2] resolved",
            )
            + "\n<!-- AGENT_STATE: approved -->\n-- Anthropic Claude",
        ],
        gemini_outputs=["Implemented blocker.\n<!-- AGENT_STATE: blocking -->\n-- Google Gemini"],
    )
    config = make_config(
        tmp_path,
        coder="gemini",
        reviewer=("codex", "claude"),
        approved_followups="summarize",
        max_rounds=2,
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    second_claude_prompt = [
        cmd[-1]
        for cmd, _cwd in runner.commands
        if cmd[:1] == ["claude"] and "round 2" in cmd[-1]
    ][0]
    assert "Document cache cleanup behavior." in second_claude_prompt
    assert "[item-1] future" in second_claude_prompt
    summary = runner.comments[-1]
    assert summary.startswith("Approved-review future follow-ups for PR #77:")
    assert "Document cache cleanup behavior." in summary

def test_codex_review_prompt_prefers_pinned_checkout_local_diff(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
        git_head="abc123",
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_command, review_cwd = next(
        (cmd, cwd) for cmd, cwd in runner.commands if cmd[:2] == ["codex", "exec"]
    )
    prompt = codex_command[-1]
    assert review_cwd == config.codex_dir
    assert f"Assigned checkout: `{config.codex_dir.resolve()}`" in prompt
    assert "- Head SHA: abc123" in prompt
    assert "`git diff main...HEAD`" in prompt
    assert "Prefer the verified local checkout and direct file reads over web search" in prompt
    assert "gh pr diff" not in prompt

def test_review_prompt_warns_that_pr_head_sha_is_authoritative(tmp_path):
    runner = FakeRunner(
        codex_outputs=["LGTM.\n<!-- AGENT_STATE: approved -->\n-- OpenAI Codex"],
    )
    config = make_config(tmp_path, reviewer="codex")

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    codex_command = next(cmd for cmd, _cwd in runner.commands if cmd[:2] == ["codex", "exec"])
    prompt = codex_command[-1]
    assert "The Head SHA above is the PR head this\nreview round is about." in prompt
    assert "The orchestrator has already refreshed and verified the\nassigned checkout." in prompt
    assert "report a blocking tooling\nmismatch instead of changing the checkout" in prompt
    assert "Do not report findings based on untracked files unless those files\nare present in the PR diff." in prompt

def test_gemini_review_loop_uses_prompt_and_extra_args(tmp_path):
    runner = FakeRunner(
        gemini_outputs=[
            json.dumps({"response": "LGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"}),
        ],
    )
    config = make_config(
        tmp_path,
        reviewer="gemini",
        gemini_args=("--output-format", "json", "--model", "gemini-2.5-flash"),
    )

    assert run_pr_loop(runner, pr_number=77, config=config) == 0

    gemini_call = next(cmd for cmd, _cwd in runner.commands if cmd[:1] == ["gemini"])
    assert gemini_call[:2] == ["gemini", "--prompt"]
    assert PUBLIC_RESPONSE_MARKER in gemini_call[2]
    assert "Only content after that line will be posted to GitHub" in gemini_call[2]
    assert "--output-format" in gemini_call
    assert "--model" in gemini_call
    assert runner.comments == ["**Review verdict:** Approved\n\nLGTM.\n<!-- AGENT_STATE: approved -->\n-- Google Gemini"]



def test_build_unresolved_items_guidance_exact_output():
    assert _build_unresolved_items_guidance() == _EXPECTED_UNRESOLVED_ITEMS_GUIDANCE

@pytest.mark.parametrize(
    "statuses,compact_has_guidance,full_has_guidance",
    [
        (None, False, False),
        ((), False, False),
        (("future",), False, True),
        (("blocking",), True, True),
        (("same-pr",), True, True),
        (("future", "blocking"), True, True),
    ],
)
def test_build_review_prompt_unresolved_items_guidance_activation(
    tmp_path,
    statuses,
    compact_has_guidance,
    full_has_guidance,
):
    config = make_config(tmp_path, approved_followups="fix-and-summarize")
    unresolved_items = tuple(
        UnresolvedReviewItem(
            item_id=f"item-{index}",
            reviewer="OpenAI Codex",
            source_round=1,
            text=f"Prior {status} item.",
            status=status,
        )
        for index, status in enumerate(statuses or (), start=1)
    )
    prompt_items = None if statuses is None else unresolved_items

    compact = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        compact_context=True,
        unresolved_items=prompt_items,
    )
    full = build_review_prompt(
        77,
        2,
        config,
        reviewer="codex",
        compact_context=False,
        unresolved_items=prompt_items,
    )

    assert (_EXPECTED_UNRESOLVED_ITEMS_GUIDANCE in compact) is compact_has_guidance
    assert (_EXPECTED_UNRESOLVED_ITEMS_GUIDANCE in full) is full_has_guidance
    for index, status in enumerate(statuses or (), start=1):
        item_label = f"[item-{index}]"
        assert (item_label in compact) is (status != "future")
        assert item_label in full

@pytest.mark.parametrize(
    "approved_followups,expected",
    [
        ("ignore", _EXPECTED_FOLLOWUP_IGNORE),
        ("fix-and-review", _EXPECTED_FOLLOWUP_FIX_AND),
        ("summarize", _EXPECTED_FOLLOWUP_ELSE),
    ],
)
def test_build_followup_guidance_branches(tmp_path, approved_followups, expected):
    config = make_config(tmp_path, approved_followups=approved_followups)
    assert _build_followup_guidance(config) == expected

@pytest.mark.parametrize("approved_followups", ["ignore", "fix-and-review", "summarize"])
def test_build_review_prompt_followup_guidance_parity(tmp_path, approved_followups):
    config = make_config(tmp_path, approved_followups=approved_followups)
    g = _build_followup_guidance(config)
    non_compact = build_review_prompt(77, 1, config, reviewer="codex", compact_context=False)
    compact = build_review_prompt(77, 1, config, reviewer="codex", compact_context=True)
    assert g in non_compact
    assert g in compact


@pytest.mark.parametrize("mode", ["plan-only", "implement-one-shot"])
def test_plan_review_prompt_includes_phased_plan_guard_in_non_decompose_mode(tmp_path, mode):
    config = make_config(tmp_path, plan_execution_mode=mode)
    guard = _phased_plan_guard(config)
    assert guard != ""
    assert "implement-by-phase" in guard
    assert "scope" in guard.lower() or "single deliverable" in guard
    for compact in (False, True):
        prompt = build_plan_review_prompt(56, 1, "Plan.", config, reviewer="codex", compact_context=compact)
        assert guard.strip() in prompt


@pytest.mark.parametrize("mode", ["decompose-only", "implement-by-phase"])
def test_plan_review_prompt_omits_phased_plan_guard_in_decompose_mode(tmp_path, mode):
    config = make_config(tmp_path, plan_execution_mode=mode)
    guard = _phased_plan_guard(config)
    assert guard == ""
    for compact in (False, True):
        prompt = build_plan_review_prompt(56, 1, "Plan.", config, reviewer="codex", compact_context=compact)
        assert "Phased-delivery guard" not in prompt


def test_compact_plan_review_stable_prefix_is_byte_identical_with_phased_guard(tmp_path):
    config = make_config(tmp_path, plan_execution_mode="plan-only")
    issue_context = _compact_issue_context()
    memory = _compact_memory_context(tmp_path)
    unresolved_item = UnresolvedReviewItem(
        item_id="item-1",
        reviewer="OpenAI Codex",
        source_round=1,
        text="Active ledger item.",
        status="blocking",
    )
    first = build_plan_review_prompt(
        56,
        2,
        "Plan tail A.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-0] resolved: old",)),
        compact_tail=CompactPlanTailContext(subject="subject-a", action="Review A."),
    )
    second = build_plan_review_prompt(
        56,
        3,
        "Plan tail B.",
        config,
        reviewer="codex",
        memory=memory,
        issue_context=issue_context,
        unresolved_items=(unresolved_item,),
        compact_context=True,
        compact_prior=CompactPriorContext(("[item-0] resolved: old",)),
        compact_tail=CompactPlanTailContext(subject="subject-b", action="Review B."),
    )

    first_prefix, _ = first.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    second_prefix, _ = second.split(COMPACT_PLANNING_VOLATILE_TAIL_MARKER, 1)
    assert first_prefix.encode() == second_prefix.encode()
    assert "Phased-delivery guard" in first_prefix
    assert "implement-by-phase" in first_prefix


def test_coder_prompt_includes_documentation_reminder(tmp_path):
    from coding_review_agent_loop.prompts import build_task_clarification_prompt

    config = make_config(tmp_path)
    prompts = [
        build_issue_prompt(1, config),
        build_issue_implementation_prompt(1, "1. Fix it.", config),
        build_task_prompt("Fix the bug.", config),
        build_task_clarification_prompt("Fix the bug.", [("What?", "This.")], config),
        build_followup_prompt(77, 1, "Needs tests.", config),
        build_same_pr_followup_prompt(77, 1, "Tighten docs.", config),
    ]
    for prompt in prompts:
        assert "README.md" in prompt
        assert "docs/" in prompt
        assert "user-facing subcommand" in prompt


def test_reviewer_prompt_includes_documentation_check(tmp_path):
    config = make_config(tmp_path, reviewer=("codex",))
    phrase = "Flag missing or stale documentation as a blocking item"

    non_compact = build_review_prompt(77, 1, config, reviewer="codex")
    compact = build_review_prompt(77, 1, config, reviewer="codex", compact_context=True)

    assert phrase in non_compact
    assert phrase in compact


# --- discuss agenda / analyzer-mode prompt tests (#467) ---

from coding_review_agent_loop.prompts import (
    build_discuss_agenda_prompt,
    build_discuss_review_prompt,
)
from coding_review_agent_loop.protocol import (
    DiscussAgendaDisagreement,
    ParsedDiscussAnswer,
    ParsedDiscussAgenda,
    ParsedDiscussReview,
)


def _discuss_prompt_vote(
    reviewer: str,
    outcome: str = "implement",
    rationale: str = "Well-scoped.",
    rebuttal: str | None = None,
    split_proposals: tuple[str, ...] = (),
    analyzer_framing: str | None = None,
    framing_note: str | None = None,
) -> ParsedDiscussReview:
    return ParsedDiscussReview(
        outcome=outcome,
        rationale=rationale,
        split_proposals=split_proposals,
        reviewer=reviewer,
        rebuttal=rebuttal,
        analyzer_framing=analyzer_framing,
        framing_note=framing_note,
    )


def _sample_agenda() -> ParsedDiscussAgenda:
    return ParsedDiscussAgenda(
        consensus=("The issue is well-motivated.",),
        disagreements=(
            DiscussAgendaDisagreement(
                topic="Scope of the change",
                positions=(("Codex", "Narrow enough."), ("Gemini", "Too broad; split it.")),
                question_for_next_round="Would splitting resolve the scope objection?",
            ),
        ),
        missing_facts=("Whether the API boundary is specified.",),
    )


def test_build_discuss_agenda_prompt_includes_every_round_of_history(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    round1 = [
        _discuss_prompt_vote("Codex", rationale="R1 codex rationale."),
        _discuss_prompt_vote("Gemini", outcome="do-not-implement", rationale="R1 gemini rationale."),
    ]
    round2 = [
        _discuss_prompt_vote("Codex", rationale="R2 codex rationale.", rebuttal="R2 codex rebuttal."),
        _discuss_prompt_vote(
            "Gemini",
            outcome="do-not-implement",
            rationale="R2 gemini rationale.",
            rebuttal="R2 gemini rebuttal.",
            analyzer_framing="misframed",
            framing_note="The agenda misstated my position.",
        ),
    ]
    prompt = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=2,
        round_history=[round1, round2],
        prior_agenda=_sample_agenda(),
    )
    assert "Round 1 debater positions:" in prompt
    assert "Round 2 debater positions (latest round):" in prompt
    assert "R1 codex rationale." in prompt
    assert "R1 gemini rationale." in prompt
    assert "R2 codex rebuttal." in prompt
    assert "R2 gemini rebuttal." in prompt
    assert "Framing correction: The agenda misstated my position." in prompt
    # Prior agenda is carried into the analyzer prompt.
    assert "Your previous agenda" in prompt
    assert "Would splitting resolve the scope objection?" in prompt
    # Fidelity guardrails.
    assert "Preserve minority arguments" in prompt
    assert "never invent positions" in prompt
    assert "Carry forward unresolved `missing_facts`" in prompt
    assert '"kind": "discuss_agenda"' in prompt


def test_build_discuss_agenda_prompt_supports_answer_history(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    answer = ParsedDiscussAnswer(
        position="answer", rationale="The evidence favors the API.", confidence="high",
        open_questions=(), reviewer="Codex", answer="Use an API.", rebuttal="Addressed the concern.",
    )
    prompt = build_discuss_agenda_prompt(
        56, config, analyzer="claude", round_number=2, round_history=[[answer]]
    )
    assert "`answer`" in prompt
    assert "Use an API." in prompt
    assert "Split proposals:" not in prompt


def test_build_discuss_agenda_prompt_without_prior_agenda(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=1,
        round_history=[[_discuss_prompt_vote("Codex")]],
    )
    assert "Your previous agenda" not in prompt
    assert "Round 1 debater positions (latest round):" in prompt


def test_build_discuss_review_prompt_analyzer_mode_omits_other_debaters_text(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prior_votes = [
        _discuss_prompt_vote("Codex", rationale="Codex private rationale text."),
        _discuss_prompt_vote(
            "Gemini",
            outcome="do-not-implement",
            rationale="Gemini private rationale text.",
            rebuttal="Gemini private rebuttal text.",
        ),
    ]
    prompt = build_discuss_review_prompt(
        56,
        config,
        reviewer="codex",
        round_number=2,
        prior_round_votes=prior_votes,
        prior_round_agenda=("- Gemini held `do-not-implement`: Gemini private rationale text.",),
        analyzer_agenda=_sample_agenda(),
    )
    # The structured agenda is present.
    assert "Scope of the change" in prompt
    assert "Too broad; split it." in prompt
    assert "Would splitting resolve the scope objection?" in prompt
    assert "Whether the API boundary is specified." in prompt
    # The target debater's own prior position is present verbatim.
    assert "Your own prior-round position (verbatim):" in prompt
    assert "Codex private rationale text." in prompt
    # Other debaters' full rationale/rebuttal text is omitted; their views reach
    # the debater only through the analyzer's summarized positions.
    assert "Gemini private rationale text." not in prompt
    assert "Gemini private rebuttal text." not in prompt
    assert "Prior round reviewer positions:" not in prompt
    # Misframing instructions and rules.
    assert "analyzer_framing" in prompt
    assert "misframed" in prompt
    assert "The analyzer is not authoritative" in prompt


def test_build_discuss_answer_prompt_keeps_analyzer_and_research_non_authoritative(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer")
    prior = ParsedDiscussAnswer(
        position="answer", rationale="Prior rationale.", confidence="medium",
        open_questions=("Which latency target applies?",), reviewer="Codex",
        answer="Use an API.", rebuttal="The boundary reduces coupling.",
    )
    prompt = build_discuss_review_prompt(
        56, config, reviewer="codex", round_number=2, prior_round_votes=[prior],
        prior_round_agenda=("- Codex held `answer`: Use an API.",),
        analyzer_agenda=_sample_agenda(), research_mode="required",
    )
    assert '"kind": "discuss_answer"' in prompt
    assert '"position": "answer"' in prompt
    assert "The analyzer is not authoritative" in prompt
    assert "Research policy:" in prompt
    assert "sourced_facts" in prompt
    assert '"target": "solution-design"' in prompt
    assert '"questions":' in prompt
    assert '"outcome":' not in prompt
    assert '"split_proposals":' not in prompt


def test_build_discuss_review_prompt_plain_mode_unchanged_without_agenda(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prior_votes = [
        _discuss_prompt_vote("Codex", rationale="Codex rationale."),
        _discuss_prompt_vote("Gemini", outcome="do-not-implement", rationale="Gemini rationale."),
    ]
    kwargs = dict(
        reviewer="codex",
        round_number=2,
        prior_round_votes=prior_votes,
        prior_round_agenda=("- Gemini held `do-not-implement`: Gemini rationale.",),
    )
    plain = build_discuss_review_prompt(56, config, **kwargs)
    explicit_none = build_discuss_review_prompt(56, config, analyzer_agenda=None, **kwargs)
    assert plain == explicit_none
    assert "Prior round reviewer positions:" in plain
    assert "Gemini rationale." in plain
    assert "analyzer_framing" not in plain
    assert "Analyzer" not in plain


def test_build_discuss_review_prompt_round1_has_no_analyzer_rules(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_review_prompt(56, config, reviewer="codex", round_number=1)
    assert "analyzer_framing" not in prompt
    assert "This is round 1." in prompt


# --- discuss research policy prompt tests (#477) ---


def test_build_discuss_review_prompt_research_none_forbids_research(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    default = build_discuss_review_prompt(56, config, reviewer="codex", round_number=1)
    explicit = build_discuss_review_prompt(
        56, config, reviewer="codex", round_number=1, research_mode="none"
    )
    assert default == explicit
    assert "Research policy: `none`" in default
    assert "do not\nperform online research" in default
    assert '"research"' not in default
    assert "sourced_facts" not in default


def test_build_discuss_review_prompt_research_required_enforces_sources(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_review_prompt(
        56, config, reviewer="codex", round_number=1, research_mode="required"
    )
    assert "Research policy: `required`" in prompt
    assert "Cite a source" in prompt
    assert '"research"' in prompt
    assert '"status": "sourced"' in prompt
    assert "sourced_facts" in prompt
    assert "decision-relevant" in prompt
    assert "solution-design" in prompt
    assert "must not be `not-needed`" in prompt
    assert "instead of presenting stale\nassumptions as fact" in prompt


def test_build_discuss_review_prompt_research_auto_documents_triggers(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_review_prompt(
        56, config, reviewer="codex", round_number=1, research_mode="auto"
    )
    assert "Research policy: `auto`" in prompt
    assert "conservative triggers" in prompt
    assert "pricing, quotas, model availability" in prompt
    assert "`not-needed`" in prompt
    assert '"status": "not-needed"' in prompt
    assert "must not be `not-needed`" not in prompt
    assert "illustrative incident" in prompt

    answer_config = make_config(
        tmp_path, reviewer=("codex", "gemini"), discuss_result_mode="answer"
    )
    answer_prompt = build_discuss_review_prompt(
        56, answer_config, reviewer="codex", round_number=1, research_mode="auto"
    )
    assert "Research is required by the selected policy" not in answer_prompt
    assert "When research is active under the selected policy" in answer_prompt


def test_build_discuss_review_prompt_renders_shared_research_brief(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    agenda = ParsedDiscussAgenda(
        consensus=(),
        disagreements=(
            DiscussAgendaDisagreement(
                topic="Scope of the change",
                positions=(("Codex", "Narrow enough."),),
                question_for_next_round="Would splitting resolve it?",
            ),
        ),
        missing_facts=(),
        research_required=True,
        research_questions=("Is Gemini CLI still available for enterprise users?",),
    )
    prompt = build_discuss_review_prompt(
        56,
        config,
        reviewer="codex",
        round_number=2,
        prior_round_votes=[_discuss_prompt_vote("Codex")],
        analyzer_agenda=agenda,
        research_mode="auto",
    )
    assert "Shared research brief" in prompt
    assert "Is Gemini CLI still available for enterprise users?" in prompt
    assert "do not duplicate work" in prompt


def test_build_discuss_agenda_prompt_research_auto_requests_brief(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=1,
        round_history=[[_discuss_prompt_vote("Codex")]],
        research_mode="auto",
    )
    assert "`research_required` / `research_questions`" in prompt
    assert "conservative\ntriggers" in prompt or "conservative triggers" in prompt
    assert '"research_required": true' in prompt
    assert "must be non-empty when `research_required` is" in prompt


def test_build_discuss_agenda_prompt_research_none_omits_research_fields(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    default = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=1,
        round_history=[[_discuss_prompt_vote("Codex")]],
    )
    explicit = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=1,
        round_history=[[_discuss_prompt_vote("Codex")]],
        research_mode="none",
    )
    assert default == explicit
    assert "research_required" not in default
    assert "research_questions" not in default


def test_build_discuss_agenda_prompt_research_required_focuses_questions(tmp_path):
    config = make_config(tmp_path, reviewer=("codex", "gemini"))
    prompt = build_discuss_agenda_prompt(
        56,
        config,
        analyzer="claude",
        round_number=1,
        round_history=[[_discuss_prompt_vote("Codex")]],
        research_mode="required",
    )
    assert "debaters must research" in prompt
    assert "`research_required` / `research_questions`" in prompt
