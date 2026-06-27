"""Prompt builders for coder and reviewer agent turns."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from textwrap import indent
from typing import Sequence

from .agents.base import AgentName
from .agents.registry import agent_display_name, agent_signature
from .config import AgentLoopConfig, reviewers
from .decomposition import MAX_DECOMPOSITION_PHASES
from .github import HumanReviewRequirement, IssueContext, PullRequestChecks, PullRequestMetadata
from .memory import AgentMemoryContext, format_agent_memory_context
from .protocol import (
    HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
    HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK,
    UnresolvedReviewItem,
)
from .workdirs import agent_workdir


@dataclass(frozen=True)
class CoderHumanRequirementsPromptContext:
    block: str
    surfaced_requirement_ids: tuple[str, ...]
    requires_direct_discussion_ack: bool


COMPACT_PLANNING_VOLATILE_TAIL_MARKER = "--- volatile compact planning tail ---"
COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER = "--- volatile compact pr-review tail ---"


@dataclass(frozen=True)
class CompactPriorContext:
    prior_item_summaries: tuple[str, ...] = ()


@dataclass(frozen=True)
class CompactPlanTailContext:
    subject: str | None = None
    action: str | None = None


@dataclass(frozen=True)
class CompactPrReviewTailContext:
    head_sha: str | None = None
    round_number: int | None = None
    action: str | None = None


def format_agent_list(agents: Sequence[AgentName]) -> str:
    names = [agent_display_name(agent) for agent in agents]
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])}, and {names[-1]}"


def _memory_block(memory: AgentMemoryContext | None) -> str:
    text = format_agent_memory_context(memory)
    if not text:
        return ""
    return f"Agent memory context:\n{text}\n"


def _scratch_file_guidance() -> str:
    return (
        "If you need temporary scratch files while inspecting diffs or command output, "
        "write them outside the repository checkout, for example under "
        "/tmp/coding-review-agent-loop/scratch/. Do not create temporary files in the "
        "repo worktree unless they are intended project changes.\n"
    )


def _coder_test_reporting_guidance() -> str:
    return (
        "Before opening or updating the PR, run the relevant tests you can run "
        "locally. In your final response, include a short `Tests:` line listing "
        "the exact commands run and whether they passed. If you cannot run tests, "
        "explain why in that `Tests:` line.\n"
    )


def _review_command_policy(
    config: AgentLoopConfig, pr_metadata: PullRequestMetadata
) -> str:
    base_branch = (pr_metadata.base_branch or config.base or "").strip()
    if base_branch and "\n" not in base_branch and "\r" not in base_branch:
        diff_guidance = (
            "For this PR, inspect the authoritative local diff with "
            f"`git diff {shlex.quote(f'{base_branch}...HEAD')}`."
        )
    else:
        diff_guidance = (
            "When the base branch is available, replace `<base>` in "
            "`git diff <base>...HEAD` with that resolved branch name; do not run "
            "the placeholder literally or discover branches speculatively."
        )
    return (
        "Do not run tests, builds, compilation, source or worktree mutation, "
        "commits, or unrelated discovery commands. Read-only commands used to "
        "inspect the assigned checkout and authoritative diff are allowed, "
        "including `git diff`, `git show`, `git status`, `git log`, `rg`, `sed`, "
        "and equivalent direct file reads. "
        f"{diff_guidance} Prefer the verified local checkout and direct file reads "
        "over web search, remote PR diff retrieval, or branch discovery. Do not "
        "fetch, checkout, reset, clean, write files, or change repository state "
        "during review.\n"
    )


def _coder_workdir_guidance(
    config: AgentLoopConfig, *, implementation: bool = True, agent: AgentName | None = None
) -> str:
    active_agent = agent or config.coder
    workdir = agent_workdir(config, active_agent).resolve()
    repo_name = config.repo.rsplit("/", 1)[-1]
    scope = (
        "Implementation, inspection, tests, commits, and pushes"
        if implementation
        else "Inspection"
    )
    dangerous_args = {
        "--dangerously-skip-permissions",
        "--dangerously-bypass-approvals-and-sandbox",
        "--yolo",
    }
    active_args = {
        "claude": config.claude_args,
        "codex": config.codex_args,
        "gemini": config.gemini_args,
        "antigravity": config.antigravity_args,
    }[active_agent]
    dangerous_warning = ""
    if any(arg in dangerous_args for arg in active_args):
        dangerous_warning = (
            " Dangerous agent permissions are active, so the CLI may allow writes "
            "outside this checkout; treat the assigned-workdir rule as mandatory."
        )
    return (
        f"Assigned checkout: `{workdir}`. `AGENT_LOOP_WORKDIR` is set to this path "
        "for agent subprocesses. "
        f"{scope} must stay in that directory unless the user explicitly authorizes "
        "a different path. Do not `cd` into sibling, home, deployment, or duplicate "
        f"clones such as `~/{repo_name}` or `~/claude-code/{repo_name}`. Before "
        "running tests or committing, run `pwd` and `git status --branch --short` "
        "and confirm the path is the assigned checkout."
        f"{dangerous_warning}\n"
    )


def _truncate_issue_text(text: str, *, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    suffix = f"\n\n[{label} truncated to keep this prompt bounded.]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def format_issue_context(issue_context: IssueContext, *, max_chars: int = 24_000) -> str:
    raw_body = issue_context.body if issue_context.body else "(none)"
    body = _truncate_issue_text(raw_body, max_chars=max_chars // 3, label="Issue body")
    title = issue_context.title if issue_context.title else "(unknown)"
    lines = [
        f"GitHub issue #{issue_context.number}",
        "",
        "Title:",
        title,
        "",
        "Body:",
        body,
        "",
        "Comments, oldest to newest:",
    ]
    if issue_context.comments:
        comments = [
            "\n".join(
                [
                    "",
                    (
                        f"Comment by {comment.author or '(unknown)'} "
                        f"at {comment.created_at or '(unknown time)'}:"
                    ),
                    comment.body if comment.body else "(none)",
                ]
            )
            for comment in issue_context.comments
        ]
        issue_header = "\n".join(lines)
        full_text = issue_header + "\n".join(comments)
        if len(full_text) <= max_chars:
            return full_text

        kept_comments: list[str] = []
        omitted_count = 0
        for comment_text in reversed(comments):
            candidate_comments = [comment_text, *kept_comments]
            notice = (
                "\n\n"
                f"Older comments omitted: {len(comments) - len(candidate_comments)} "
                "comment(s) were omitted to keep this prompt bounded. "
                "Newest comments were kept preferentially."
            )
            candidate = issue_header + notice + "".join(candidate_comments)
            if len(candidate) <= max_chars:
                kept_comments = candidate_comments
                omitted_count = len(comments) - len(candidate_comments)
            elif not kept_comments:
                omitted_count = len(comments) - 1
                notice = (
                    "\n\n"
                    f"Older comments omitted: {omitted_count} comment(s) were omitted to keep "
                    "this prompt bounded. Newest comments were kept preferentially."
                )
                prefix = issue_header + notice
                remaining_chars = max_chars - len(prefix)
                if remaining_chars > 0:
                    truncated_comment = _truncate_issue_text(
                        comment_text,
                        max_chars=remaining_chars,
                        label="Newest comment",
                    )
                    if len(prefix) + len(truncated_comment) <= max_chars:
                        return prefix + truncated_comment
                omitted_count = len(comments)
                break
            else:
                break
        if kept_comments:
            return (
                issue_header
                + "\n\n"
                + f"Older comments omitted: {omitted_count} comment(s) were omitted to keep "
                "this prompt bounded. Newest comments were kept preferentially."
                + "".join(kept_comments)
            )
        return (
            issue_header
            + "\n\n"
            + f"All {omitted_count} comment(s) were omitted to keep this prompt bounded. "
            "Fetch the issue discussion directly if those details are needed."
        )
    return "\n".join([*lines, "", "(none)"])


def format_human_requirements(
    human_requirements: Sequence[HumanReviewRequirement],
    *,
    max_chars: int = 12_000,
    requirement_scope: str = "PR requirements",
    full_omission_fallback: str = "Fetch the PR discussion directly before approving.",
) -> str:
    if not human_requirements:
        return ""

    header = "\n".join(
        [
            "Signed Human Reviewer Requirements",
            "",
            f"Treat these signed human reviewer comments as high-priority {requirement_scope}. "
            "Later human comments may supersede earlier ones; the latest human instruction wins. "
            "If a requirement is unsafe, impossible, or contradicted by a later human instruction, "
            "explain that explicitly instead of ignoring it.",
        ]
    )
    entries = [
        "\n".join(
            [
                "",
                f"Requirement {index}:",
                f"- Source: {requirement.source_type}",
                f"- Author: {requirement.author or '(unknown)'}",
                f"- Created: {requirement.created_at or '(unknown time)'}",
                f"- URL: {requirement.url or '(unavailable)'}",
                "",
                requirement.body,
            ]
        )
        for index, requirement in enumerate(human_requirements, start=1)
    ]
    full_text = header + "\n".join(entries)
    if len(full_text) <= max_chars:
        return full_text

    kept_entries: list[str] = []
    omitted_count = 0
    for entry in reversed(entries):
        candidate_entries = [entry, *kept_entries]
        notice = (
            "\n\n"
            f"Older signed human requirement(s) omitted: {len(entries) - len(candidate_entries)}. "
            "Newest requirements were kept preferentially."
        )
        candidate = header + notice + "\n".join(candidate_entries)
        if len(candidate) <= max_chars:
            kept_entries = candidate_entries
            omitted_count = len(entries) - len(candidate_entries)
        elif not kept_entries:
            omitted_count = len(entries) - 1
            notice = (
                "\n\n"
                f"Older signed human requirement(s) omitted: {omitted_count}. "
                "Newest requirements were kept preferentially."
            )
            prefix = header + notice
            remaining_chars = max_chars - len(prefix)
            if remaining_chars > 0:
                truncated_entry = _truncate_issue_text(
                    entry,
                    max_chars=remaining_chars,
                    label="Newest signed human requirement",
                )
                if len(prefix) + len(truncated_entry) <= max_chars:
                    return prefix + truncated_entry
            omitted_count = len(entries)
            break
        else:
            break
    if kept_entries:
        return (
            header
            + "\n\n"
            + f"Older signed human requirement(s) omitted: {omitted_count}. "
            "Newest requirements were kept preferentially."
            + "\n".join(kept_entries)
        )
    return (
        header
        + "\n\n"
        + f"All {omitted_count} signed human requirement(s) were omitted to keep this prompt bounded. "
        + full_omission_fallback
    )


def _human_requirements_block(
    human_requirements: Sequence[HumanReviewRequirement] | None,
    *,
    requirement_scope: str = "PR requirements",
    full_omission_fallback: str = "Fetch the PR discussion directly before approving.",
) -> str:
    if not human_requirements:
        return ""
    return (
        f"{format_human_requirements(
            human_requirements,
            requirement_scope=requirement_scope,
            full_omission_fallback=full_omission_fallback,
        )}\n"
    )


def render_coder_human_requirements_prompt_context(
    human_requirements: Sequence[HumanReviewRequirement] | None,
    *,
    max_chars: int = 12_000,
    requirement_scope: str = "PR requirements",
    full_omission_fallback: str = "Fetch the PR discussion directly before approving.",
) -> CoderHumanRequirementsPromptContext:
    if not human_requirements:
        block = ""
    else:
        block = (
            f"{format_human_requirements(
                human_requirements,
                max_chars=max_chars,
                requirement_scope=requirement_scope,
                full_omission_fallback=full_omission_fallback,
            )}\n"
        )
    if not block:
        return CoderHumanRequirementsPromptContext(
            block="",
            surfaced_requirement_ids=(),
            requires_direct_discussion_ack=False,
        )
    surfaced_requirement_ids = tuple(
        f"Requirement {match.group(1)}"
        for match in re.finditer(r"(?m)^Requirement (\d+):$", block)
    )
    requires_direct_discussion_ack = (
        not surfaced_requirement_ids
        and "signed human requirement(s) were omitted to keep this prompt bounded" in block
    )
    return CoderHumanRequirementsPromptContext(
        block=block,
        surfaced_requirement_ids=surfaced_requirement_ids,
        requires_direct_discussion_ack=requires_direct_discussion_ack,
    )


def _coder_human_requirements_guidance(
    context: CoderHumanRequirementsPromptContext,
    *,
    requirement_label: str = "next-revision requirements",
    surfaced_requirement_instruction: str = (
        "Each bullet must explain how you addressed that item or why it could not be satisfied safely."
    ),
) -> str:
    if not context.block:
        return ""
    lines = [
        f"The signed human reviewer requirements above are mandatory {requirement_label}, not passive context.",
        "Later signed human comments supersede earlier ones; the latest human instruction wins.",
        "If any signed human requirement cannot be satisfied safely or is impossible, say that explicitly instead of skipping it.",
        "",
        "When signed human reviewer requirements are present, your response must include exactly:",
        HUMAN_REQUIREMENTS_ADDRESSED_MARKER,
        "",
        "Then add a `### Human requirements` section.",
    ]
    if context.surfaced_requirement_ids:
        surfaced = ", ".join(f"`{item}`" for item in context.surfaced_requirement_ids)
        lines.append(
            "Include one bullet for each surfaced signed human requirement ID shown in this prompt: "
            f"{surfaced}. {surfaced_requirement_instruction}"
        )
    elif context.requires_direct_discussion_ack:
        lines.append(
            "The detailed requirement entries were omitted from this prompt to stay bounded. "
            "In the `### Human requirements` section, state that the prompt omitted the detailed items and that you "
            f"`{HUMAN_REQUIREMENTS_DIRECT_DISCUSSION_ACK}` before responding."
        )
    return "\n".join(lines) + "\n"


def _human_requirements_review_guidance(
    human_requirements: Sequence[HumanReviewRequirement] | None,
    *,
    requirement_label: str = "signed human reviewer requirements",
) -> str:
    if not human_requirements:
        return ""
    return f"""{requirement_label.capitalize()} override AI reviewer preferences unless they
are unsafe, impossible, or contradicted by a later signed human instruction.
Verify each requirement in this set before approving. If all surfaced signed
human requirements are addressed or explicitly resolved, an approved
review must include exactly:

<!-- HUMAN_REQUIREMENTS_RESOLVED -->

If any signed human requirement in this set is unresolved, return blocking.
"""


def _structured_coder_followup_guidance(
    *,
    reviewer_name: str,
    human_requirements_context: CoderHumanRequirementsPromptContext,
) -> str:
    example_human_requirement_ids = (
        list(human_requirements_context.surfaced_requirement_ids[:1])
        if human_requirements_context.surfaced_requirement_ids
        else []
    )
    example_human_requirement_ids_json = json.dumps(example_human_requirement_ids)
    lines = [
        f"Use this mandatory structured JSON follow-up format so {reviewer_name} and the orchestrator can validate your response deterministically:",
        "",
        "{",
        '  "schema_version": 1,',
        '  "kind": "coder_followup",',
        '  "state": "blocking",',
        '  "summary": "Implemented the requested fix and left one reviewer item for follow-up.",',
        '  "addressed_items": ["item-1"],',
        '  "remaining_items": ["item-2"],',
        '  "disputed_items": ["item-3"],',
        '  "addressed_item_notes": {"item-1": "Updated the parser and added regression coverage."},',
        '  "remaining_item_notes": {"item-2": "Deferred because it requires a separate UI change."},',
        '  "dispute_evidence": {"item-3": "Checked Google pricing page (https://...): $1.50/1M tokens is correct per the official docs."},',
        '  "human_requirements": {',
        f'    "addressed_ids": {example_human_requirement_ids_json},',
        '    "checked_discussion_directly": false',
        "  },",
        '  "tests_run": ["python -m pytest tests/test_agent_loop.py -k followup"]',
        "}",
        "",
        "Required structured fields: `schema_version`, `kind`, `state`, `summary`, `addressed_items`, `remaining_items`, and `human_requirements`. `addressed_item_notes`, `remaining_item_notes`, `disputed_items`, `dispute_evidence`, and `tests_run` are optional.",
        "Your response and public response file must start directly with `{`, contain exactly one top-level JSON object, and must not include stdout filtering markers, prose, headings, or code fences before or between the JSON and footer.",
        "After the JSON object, add exactly one footer `<!-- AGENT_STATE: approved|blocking -->`, then only your standalone signature. The JSON `state` must match the `AGENT_STATE` footer exactly.",
        "Use `addressed_items`, `remaining_items`, and `disputed_items` to classify every unresolved reviewer item ID shown in this prompt. Do not omit any listed reviewer item ID and do not list any item ID more than once.",
        "Use `addressed_item_notes` to summarize how each addressed item was resolved, and use `remaining_item_notes` to give a visible reason for each intentionally deferred remaining item.",
        "Use `disputed_items` when a reviewer claim is factually incorrect (wrong pricing, stale diff reading, incorrect behavior assumption) and you have verifiable counter-evidence. Put the item ID in `disputed_items` instead of `addressed_items` or `remaining_items`, and record your evidence in `dispute_evidence`. The reviewer will get one more turn to reconsider with your evidence attached. If the reviewer still blocks after seeing the evidence, the orchestrator will surface the disagreement to a human for resolution.",
    ]
    if human_requirements_context.surfaced_requirement_ids:
        surfaced = ", ".join(f"`{item}`" for item in human_requirements_context.surfaced_requirement_ids)
        lines.append(
            "In structured replies, set `human_requirements.addressed_ids` to exactly the surfaced signed human requirement IDs you addressed in this prompt: "
            f"{surfaced}. Only exact surfaced labels such as `Requirement 1` are valid."
        )
    elif human_requirements_context.requires_direct_discussion_ack:
        lines.append(
            "If the prompt omitted detailed signed human requirements, set `human_requirements.addressed_ids` to `[]` and `human_requirements.checked_discussion_directly` to `true` only after checking the GitHub discussion directly. Do not put issue numbers, issue acceptance criteria, reviewer item IDs, reviewer comments, summaries, or arbitrary labels in `addressed_ids`."
        )
    else:
        lines.append(
            "No signed human requirements are surfaced in this prompt, so structured replies must set `human_requirements.addressed_ids` to `[]`. Do not put issue numbers, issue acceptance criteria, reviewer item IDs, reviewer comments, summaries, or arbitrary labels in `addressed_ids`; only exact surfaced signed requirement labels are valid when such labels are actually shown."
        )
    return "\n".join(lines) + "\n"


def _format_unresolved_review_items(unresolved_items: Sequence[UnresolvedReviewItem] | None) -> str:
    if not unresolved_items:
        return ""
    lines = [
        "Prior unresolved review items from earlier rounds",
        "",
        "Explicitly evaluate every item below before approving. Use the item IDs exactly as written.",
        "For carried future follow-ups, restate `future follow-up: reason` only when the item still deserves separate tracking; use `resolved` if later PR changes already handled it, or promote it to `same-pr`/`still blocking` if it must be fixed before merge.",
        "",
    ]
    for item in unresolved_items:
        details = [indent(item.text, "  ")]
        if item.notes:
            details.append("  Latest reviewer updates:")
            details.extend(f"  - {note}" for note in item.notes)
        lines.extend(
            [
                f"- [{item.item_id}] {item.status} from {item.reviewer} in round {item.source_round}",
                *details,
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _format_unresolved_plan_items(unresolved_items: Sequence[UnresolvedReviewItem] | None) -> str:
    if not unresolved_items:
        return ""
    lines = [
        "Prior unresolved plan items from earlier rounds",
        "",
        "Explicitly evaluate every item below before approving. Use the item IDs exactly as written.",
        "For carried future follow-ups, restate `future follow-up: reason` only when the item still deserves separate tracking; use `resolved` if later plan changes already handled it, or promote it to `same-plan`/`still blocking` if it must be fixed before implementation.",
        "",
    ]
    for item in unresolved_items:
        details = [indent(item.text, "  ")]
        if item.notes:
            details.append("  Latest reviewer updates:")
            details.extend(f"  - {note}" for note in item.notes)
        lines.extend(
            [
                f"- [{item.item_id}] {item.status} from {item.reviewer} in round {item.source_round}",
                *details,
            ]
        )
    lines.append("")
    return "\n".join(lines)


def format_pr_checks(checks: PullRequestChecks) -> str:
    lines = [
        "GitHub PR checks:",
        f"- Overall state: {checks.state}",
    ]
    if checks.required_checks:
        lines.append(f"- Required checks: {', '.join(checks.required_checks)}")
    if checks.failing:
        lines.append(
            "- Failing checks: "
            + ", ".join(f"{check.name} ({check.status})" for check in checks.failing)
        )
    if checks.pending:
        lines.append(
            "- Pending checks: "
            + ", ".join(f"{check.name} ({check.status})" for check in checks.pending)
        )
    if checks.missing_required:
        lines.append(f"- Required checks not yet reporting: {', '.join(checks.missing_required)}")
    if checks.branch_protection_note:
        lines.append(f"- Branch protection: {checks.branch_protection_note}")
    return "\n".join(lines)


def _issue_context_block(issue_context: IssueContext | None) -> str:
    if issue_context is None:
        return ""
    return (
        "Issue context from GitHub. Later comments may refine or supersede the "
        "original issue body:\n"
        f"{format_issue_context(issue_context)}\n"
    )


def _compact_issue_context_block(issue_context: IssueContext | None) -> str:
    if issue_context is None:
        return ""
    title = issue_context.title if issue_context.title else "(unknown)"
    body = issue_context.body if issue_context.body else "(none)"
    return "\n".join(
        [
            "Original GitHub issue context",
            f"Issue number: {issue_context.number}",
            "",
            "Title:",
            title,
            "",
            "Body:",
            body,
            "",
            "Raw prior issue comments are omitted in compact planning context. Durable reviewer findings must appear in the active unresolved ledger or the append-only compact prior ledger below.",
            "",
        ]
    )


def _compact_prior_ledger_block(compact_prior: CompactPriorContext | None) -> str:
    summaries = compact_prior.prior_item_summaries if compact_prior is not None else ()
    if not summaries:
        return "Append-only compact prior item ledger\n\n(none)\n"
    return (
        "Append-only compact prior item ledger\n\n"
        + "\n\n".join(summaries)
        + "\n"
    )


def _phased_plan_guard(config: AgentLoopConfig) -> str:
    if config.plan_execution_mode in {"decompose-only", "implement-by-phase"}:
        return ""
    return (
        "Phased-delivery guard: if the plan defers any implementation to future PRs, "
        "future phases, or future issues — for example, language like \"Phase N of M\", "
        "\"phases X–Y deferred to future PRs\", or \"handled in a follow-up PR\" — "
        "return blocking and include a blocking issue instructing the coder to either "
        "(1) scope the plan down to a single deliverable that fits in one PR, or "
        "(2) re-invoke with `--plan-execution-mode implement-by-phase` so phases are "
        "tracked as GitHub child issues created mechanically from structured JSON. "
        "A phased plan approved in this mode leaves later phases as untracked prose "
        "with no mechanical follow-up.\n"
    )


def _canonical_plan_ledger_rules() -> str:
    return """Canonical compact planning ledger rules

- Treat active prior unresolved plan items as approval-critical until explicitly dispositioned.
- Treat append-only compact prior ledger entries as the canonical history for prior blocking and same-plan concerns that left the active ledger.
- Do not reinterpret, reorder, or rewrite prior compact ledger entries; append only new resolved or future-follow-up summaries after later review rounds.
- Future follow-ups are relevant in compact mode only when elevated, referenced, or preserved in the compact prior ledger.
"""


def _plan_review_schema_and_rules() -> str:
    return """Plan review response protocol

Use this mandatory structured JSON response format:

{
  "schema_version": 1,
  "kind": "plan_review",
  "state": "approved",
  "summary": "Plan looks good.",
  "blocking_plan_issues": [],
  "same_plan_followups": [],
  "future_followups": ["Consider a later cleanup pass."],
  "prior_plan_item_dispositions": [
    {"item_id": "item-1", "disposition": "resolved", "note": "Covered by the revised tests."}
  ]
}
<!-- AGENT_PLAN_STATE: approved -->
-- reviewer signature shown in the volatile tail

If signed human requirements are present in the stable prefix and are fully
addressed or explicitly resolved, include exactly this marker before the
`AGENT_PLAN_STATE` footer:

<!-- HUMAN_REQUIREMENTS_RESOLVED -->

Blocking plan issues and Same-plan follow-ups both prevent approval. Same-plan
follow-ups are small current-plan refinements that must be incorporated before
implementation starts; they may appear only in blocking plan reviews. Future
follow-ups are independent later work that remains valid after the current
plan is approved; they are allowed only in approved plan reviews. Before adding a
`future_followups` entry, ask whether the finding is a one-line or trivially small
mechanical change — initialising a variable, renaming for clarity, adding a missing
annotation. If so, put it in `same_plan_followups` instead; a separate PR just for a
one-liner adds overhead without value. A concern or
paraphrase belongs in exactly one current-round list: `blocking_plan_issues`,
`same_plan_followups`, or `future_followups`. Do not duplicate or reclassify
the same concern across Same-plan and Future follow-up lists.
Approved means there are no blocking plan issues, no Same-plan follow-ups, and
no carried-forward plan items left active for this planning round. If you
return `<!-- AGENT_PLAN_STATE: blocking -->`, do not use structured Future
follow-ups; keep all required current-round issues in the main blocking
content so they are not missed during plan revision.
Only items listed under `Prior unresolved plan items from earlier rounds` are
eligible for dispositions in this round. If same-round findings from other
reviewers appear elsewhere in the issue discussion, treat them as
informational only and do not disposition them until a later round carries
them forward explicitly.
Structured responses must start with one top-level JSON object, place the
`AGENT_PLAN_STATE` footer immediately after that payload, and end with only
your standalone signature. Do not include prose or code fences before the JSON
object.
"""


def _plan_revision_schema_and_rules() -> str:
    return """Plan revision response protocol

Revise the plan item by item instead of replying only with free-form prose. If
prior unresolved plan items are listed in the stable prefix, include this exact
section in your response and address each item ID exactly once:

### Prior plan review item dispositions
- [item-id] resolved: brief explanation of how the revised plan addresses it
- [item-id] still blocking: brief explanation if you cannot resolve it yet
- [item-id] same-plan: brief explanation of the remaining current-plan refinement
- [item-id] future: brief explanation only if a reviewer explicitly asked to move it to future work

Then provide the revised implementation plan with concrete file areas, edge
cases, and tests. Use `same-plan`, never `same-pr`, when describing plan-only
current-round refinements.

Use this mandatory structured JSON response format:

{
  "schema_version": 1,
  "kind": "plan_revision",
  "state": "blocking",
  "summary": "Updated the plan to cover parser hard-fail behavior and resume state reconstruction.",
  "prior_plan_item_dispositions": [
    {"item_id": "item-3", "disposition": "resolved", "note": "Added the carry-forward metadata step."}
  ],
  "plan_steps": [
    "Update `src/coding_review_agent_loop/protocol.py` to hard-fail invalid structured plan payloads after JSON-prefix detection.",
    "Normalize structured plan rendering and metadata-backed resume behavior in `src/coding_review_agent_loop/orchestrator.py`.",
    "Extend prompts and targeted tests for structured planning flows."
  ]
}
<!-- AGENT_PLAN_STATE: blocking -->
-- coder signature shown in the volatile tail

The orchestrator will normalize structured plan revisions into canonical
markdown for stored plan state, reviewer prompts, subject hashing, and resume.
Your response must start with exactly one top-level JSON object, with no prose
or code fences before it. If signed human requirements are present, put
`<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` and a `### Human requirements` section
after the JSON object and before the `AGENT_PLAN_STATE` footer. Otherwise, put
the `AGENT_PLAN_STATE` footer immediately after the JSON. Make the footer state
match the JSON state, and include only your standalone signature after the
footer.
"""


def _compact_plan_stable_prefix(
    *,
    config: AgentLoopConfig,
    workdir_guidance: str,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    compact_prior: CompactPriorContext | None,
    human_requirements_block: str,
    human_requirements_guidance: str,
    response_protocol: str,
) -> str:
    return "\n".join(
        part.rstrip()
        for part in (
            "Compact cache-aware planning context",
            f"Repository: {config.repo}",
            workdir_guidance,
            _scratch_file_guidance(),
            response_protocol,
            _memory_block(memory),
            _compact_issue_context_block(issue_context),
            human_requirements_block,
            human_requirements_guidance,
            _canonical_plan_ledger_rules(),
            _format_unresolved_plan_items(unresolved_items) or "Prior unresolved plan items from earlier rounds\n\n(none)\n",
            _compact_prior_ledger_block(compact_prior),
        )
        if part
    ) + "\n"


def _issue_human_requirements_block(
    issue_context: IssueContext | None,
    *,
    requirement_scope: str,
    full_omission_fallback: str,
) -> str:
    if issue_context is None:
        return ""
    return _human_requirements_block(
        issue_context.human_requirements,
        requirement_scope=requirement_scope,
        full_omission_fallback=full_omission_fallback,
    )


def _issue_human_requirements_prompt_context(
    issue_context: IssueContext | None,
    *,
    requirement_scope: str,
    full_omission_fallback: str,
) -> CoderHumanRequirementsPromptContext:
    if issue_context is None:
        return CoderHumanRequirementsPromptContext(
            block="",
            surfaced_requirement_ids=(),
            requires_direct_discussion_ack=False,
        )
    return render_coder_human_requirements_prompt_context(
        issue_context.human_requirements,
        requirement_scope=requirement_scope,
        full_omission_fallback=full_omission_fallback,
    )


def _issue_pr_reference_guidance(issue_number: int) -> str:
    return (
        f"In the pull request body, include `Fixes #{issue_number}` or another direct "
        f"reference to issue #{issue_number} so GitHub links the PR back to the issue.\n"
    )


def build_issue_prompt(
    issue_number: int,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    human_requirements_context = _issue_human_requirements_prompt_context(
        issue_context,
        requirement_scope="implementation requirements",
        full_omission_fallback="Fetch the issue discussion directly before implementing.",
    )
    return f"""Fix GitHub issue #{issue_number} in {config.repo}.

Use this local checkout as your workspace. Create a branch, implement the fix,
run relevant tests, commit, push, and open a pull request against {config.base}.
{_coder_workdir_guidance(config)}
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}
{_issue_pr_reference_guidance(issue_number)}
{human_requirements_context.block}{_coder_human_requirements_guidance(
    human_requirements_context,
    requirement_label="implementation requirements",
)}
{_issue_context_block(issue_context)}
{_memory_block(memory)}

Do not wait for {reviewer_name} yourself; this local orchestrator will run {reviewer_name} after
you create the PR. Use blocking here to hand the PR to {reviewer_name} for review. Do not place
your signature before the AGENT_STATE marker. Your response must end with, in this exact order:

<!-- AGENT_PR: <number> -->
<!-- AGENT_STATE: blocking -->
-- {coder_signature}
"""


def build_issue_plan_prompt(
    issue_number: int,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    human_requirements_context = _issue_human_requirements_prompt_context(
        issue_context,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before finalizing the plan.",
    )
    return f"""Plan GitHub issue #{issue_number} in {config.repo}.

Use this local checkout only to inspect context. Do not edit files, create a
branch, commit, push, or open a pull request during this planning stage.
Write a concise implementation plan covering the intended approach, key files or
areas to change, edge cases, and test strategy.
{_coder_workdir_guidance(config, implementation=False)}
{_scratch_file_guidance()}
{human_requirements_context.block}{_coder_human_requirements_guidance(
    human_requirements_context,
    requirement_label="planning requirements",
    surfaced_requirement_instruction=(
        "Each bullet must explain how the plan covers that item or what remains risky or blocked."
    ),
)}
{_issue_context_block(issue_context)}
{_memory_block(memory)}

Do not wait for {reviewer_name} yourself; this local orchestrator will run
{reviewer_name} to review the plan. Use blocking to hand the plan to {reviewer_name}
for review. If the issue is materially ambiguous before a useful plan can be written,
ask focused clarifying questions and use <!-- AGENT_CLARIFY --> instead. Do not place
your signature before the AGENT_PLAN_STATE or AGENT_CLARIFY marker. Your response
must end with, in this exact order:

<!-- AGENT_PLAN_STATE: blocking -->
-- {coder_signature}

or, if clarifying:

<!-- AGENT_CLARIFY -->
-- {coder_signature}
"""


def build_plan_review_prompt(
    issue_number: int,
    round_number: int,
    plan: str,
    config: AgentLoopConfig,
    *,
    reviewer: AgentName,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
    compact_context: bool = False,
    compact_prior: CompactPriorContext | None = None,
    compact_tail: CompactPlanTailContext | None = None,
) -> str:
    if compact_context:
        return _build_compact_plan_review_prompt(
            issue_number,
            round_number,
            plan,
            config,
            reviewer=reviewer,
            memory=memory,
            issue_context=issue_context,
            unresolved_items=unresolved_items,
            compact_prior=compact_prior,
            compact_tail=compact_tail,
        )
    coder_name = agent_display_name(config.coder)
    reviewer_signature = agent_signature(reviewer, config)
    reviewer_group = format_agent_list(reviewers(config))
    unresolved_items_block = _format_unresolved_plan_items(unresolved_items)
    human_requirements_block = _issue_human_requirements_block(
        issue_context,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before approving the plan.",
    )
    human_requirements_guidance = _human_requirements_review_guidance(
        issue_context.human_requirements if issue_context is not None else (),
        requirement_label="signed human issue requirements",
    )
    if unresolved_items:
        unresolved_items_guidance = """Prior unresolved plan items are present. Disposition every listed item
in the JSON `prior_plan_item_dispositions` array — do not add a separate prose section
or bullet list for prior items outside the JSON object. Valid `disposition` values:
- `"resolved"` — item is fixed in this revision.
- `"blocking"` — item is still a blocking plan problem; include it in `blocking_plan_issues`.
- `"same-plan"` — item needs to be incorporated before implementation; include it in `same_plan_followups`.
- `"future"` — deferred; must include a `"note"` field; only valid when approving.

Only use `"future"` when returning `approved`. If a plan item still needs to be fixed
before implementation starts, use `"blocking"` or `"same-plan"` instead. For any prior
item already marked `"future"`, explicitly re-evaluate.
"""
    else:
        unresolved_items_guidance = ""
    return f"""Review the implementation plan for GitHub issue #{issue_number} in {config.repo} (planning round {round_number}).

Use this local checkout only to inspect context. Do not edit files, create a
branch, commit, push, or open a pull request during this planning review.
{_coder_workdir_guidance(config, implementation=False, agent=reviewer)}
{_scratch_file_guidance()}
{human_requirements_block}{_issue_context_block(issue_context)}
{unresolved_items_block}{_memory_block(memory)}

Plan from {coder_name}:

{plan}

Review the plan for correctness, architecture fit, missing edge cases, test
strategy, and ambiguity. Use this mandatory structured JSON response format:

{{
  "schema_version": 1,
  "kind": "plan_review",
  "state": "approved",
  "summary": "Plan looks good.",
  "blocking_plan_issues": [],
  "same_plan_followups": [],
  "future_followups": ["Consider a later cleanup pass."],
  "prior_plan_item_dispositions": [
    {{"item_id": "item-1", "disposition": "resolved", "note": "Covered by the revised tests."}}
  ]
}}
<!-- AGENT_PLAN_STATE: approved -->
-- {reviewer_signature}

Blocking plan issues and Same-plan follow-ups both prevent approval. Same-plan
follow-ups are small current-plan refinements that must be incorporated before
implementation starts; they may appear only in blocking plan reviews. Future
follow-ups are independent later work that remains valid after the current
plan is approved; they are allowed only in approved plan reviews. Before adding a
`future_followups` entry, ask whether the finding is a one-line or trivially small
mechanical change — initialising a variable, renaming for clarity, adding a missing
annotation. If so, put it in `same_plan_followups` instead; a separate PR just for a
one-liner adds overhead without value. A concern or
paraphrase belongs in exactly one current-round list: `blocking_plan_issues`,
`same_plan_followups`, or `future_followups`. Do not duplicate or reclassify
the same concern across Same-plan and Future follow-up lists.
Approved means there are no blocking plan issues, no Same-plan follow-ups, and
no carried-forward plan items left active for this planning round. If you
return `<!-- AGENT_PLAN_STATE: blocking -->`, do not use structured Future
follow-ups; keep all required current-round issues in the main blocking
content so they are not missed during plan revision.
Only items listed under `Prior unresolved plan items from earlier rounds` are
eligible for dispositions in this round. If same-round findings from other
reviewers appear elsewhere in the issue discussion, treat them as
informational only and do not disposition them until a later round carries
them forward explicitly.
{unresolved_items_guidance}
Signed human issue requirements are approval-critical issue constraints for this
plan review.
{human_requirements_guidance}
{_phased_plan_guard(config)}Use blocking only when the current plan still has blocking plan issues or
same-plan follow-ups. All configured reviewers ({reviewer_group}) must approve
in the same planning round before implementation can proceed.
Use approved only if there are no blocking plan issues, no Same-plan
follow-ups, and no carried-forward plan items left active for this planning
round. Structured responses must start with one top-level JSON object, place
the `AGENT_PLAN_STATE` footer immediately after that payload, then your
standalone signature on the final line. Do not include prose or code fences
before the JSON object. Do not place your signature before the
AGENT_PLAN_STATE footer. Your response must end with, in this exact order:

<!-- AGENT_PLAN_STATE: approved -->
-- {reviewer_signature}

or:

<!-- AGENT_PLAN_STATE: blocking -->
-- {reviewer_signature}
"""


def _build_compact_plan_review_prompt(
    issue_number: int,
    round_number: int,
    plan: str,
    config: AgentLoopConfig,
    *,
    reviewer: AgentName,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    compact_prior: CompactPriorContext | None,
    compact_tail: CompactPlanTailContext | None,
) -> str:
    coder_name = agent_display_name(config.coder)
    reviewer_name = agent_display_name(reviewer)
    reviewer_signature = agent_signature(reviewer, config)
    reviewer_group = format_agent_list(reviewers(config))
    human_requirements_block = _issue_human_requirements_block(
        issue_context,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before approving the plan.",
    )
    human_requirements_guidance = _human_requirements_review_guidance(
        issue_context.human_requirements if issue_context is not None else (),
        requirement_label="signed human issue requirements",
    )
    if unresolved_items:
        unresolved_items_guidance = """Prior unresolved plan items are present. Disposition every listed item
in the JSON `prior_plan_item_dispositions` array — do not add a separate prose section
or bullet list for prior items outside the JSON object. Valid `disposition` values:
- `"resolved"` — item is fixed in this revision.
- `"blocking"` — item is still a blocking plan problem; include it in `blocking_plan_issues`.
- `"same-plan"` — item needs to be incorporated before implementation; include it in `same_plan_followups`.
- `"future"` — deferred; must include a `"note"` field; only valid when approving.

Only use `"future"` when returning `approved`. If a plan item still needs to be fixed
before implementation starts, use `"blocking"` or `"same-plan"` instead. For any prior
item already marked `"future"`, explicitly re-evaluate.
"""
    else:
        unresolved_items_guidance = ""
    stable_prefix = _compact_plan_stable_prefix(
        config=config,
        workdir_guidance=_coder_workdir_guidance(config, implementation=False, agent=reviewer),
        memory=memory,
        issue_context=issue_context,
        unresolved_items=unresolved_items,
        compact_prior=compact_prior,
        human_requirements_block=human_requirements_block,
        human_requirements_guidance=(
            "Signed human issue requirements are approval-critical issue constraints for this plan review.\n"
            f"{human_requirements_guidance}"
        ),
        response_protocol=_plan_review_schema_and_rules() + "\n" + unresolved_items_guidance + _phased_plan_guard(config),
    )
    subject_line = f"Current plan subject: {compact_tail.subject}" if compact_tail and compact_tail.subject else "Current plan subject: (unknown)"
    action = (
        compact_tail.action
        if compact_tail and compact_tail.action
        else "Review the current plan for correctness, architecture fit, missing edge cases, test strategy, and ambiguity."
    )
    return f"""{stable_prefix}
{COMPACT_PLANNING_VOLATILE_TAIL_MARKER}

GitHub issue #{issue_number}
Planning round: {round_number}
Role: reviewer
Reviewer: {reviewer_name}
Coder: {coder_name}
{subject_line}
Action for this call: {action}

Current implementation plan from {coder_name}:

{plan}

All configured reviewers ({reviewer_group}) must approve in the same planning
round before implementation can proceed. Use approved only if there are no
blocking plan issues, no Same-plan follow-ups, and no carried-forward plan
items left active for this planning round. Do not place your signature before
the AGENT_PLAN_STATE footer. Your response must end with, in this exact order:

<!-- AGENT_PLAN_STATE: approved -->
-- {reviewer_signature}

or:

<!-- AGENT_PLAN_STATE: blocking -->
-- {reviewer_signature}
"""


def build_plan_decomposition_prompt(
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
) -> str:
    coder_signature = agent_signature(config.coder, config)
    human_requirements_block = _issue_human_requirements_block(
        issue_context,
        requirement_scope="decomposition requirements",
        full_omission_fallback="Fetch the issue discussion directly before decomposing the plan.",
    )
    return f"""Decompose the approved implementation plan for GitHub issue #{issue_number} in {config.repo}.

Use this local checkout only to inspect context. Do not edit files, create a
branch, commit, push, or open a pull request during this decomposition stage.
{_coder_workdir_guidance(config, implementation=False)}
{_scratch_file_guidance()}
{human_requirements_block}{_issue_context_block(issue_context)}
{_memory_block(memory)}

Approved implementation plan:

{approved_plan}

Return exactly one JSON object. The orchestrator will validate it and create one
GitHub child issue for every phase. A parent checklist or table is only summary
text and cannot replace child issues.

Schema:
{{
  "schema_version": 1,
  "kind": "plan_decomposition",
  "phases": [
    {{
      "title": "Short phase title",
      "scope": "What this phase implements.",
      "non_goals": "What this phase explicitly does not do.",
      "dependency_notes": "Ordering and dependency notes.",
      "rollout_risk": "low|medium|high plus a short reason.",
      "validation": "Expected validation and soak/checkpoint before the next phase.",
      "parent_context": "Self-contained excerpt of the approved parent-plan slice and constraints needed to run this child issue independently.",
      "automation": "agent-pr",
      "depends_on": []
    }}
  ]
}}

Rules:
- Use between 1 and {MAX_DECOMPOSITION_PHASES} phases. If the plan seems larger,
  consolidate related steps; do not exceed the cap.
- Preserve dependency ordering. `depends_on` must reference earlier phase titles.
- Every child issue must be self-contained enough for `agent-loop issue <N>` to
  run without rediscovering the full parent thread.
- Classify automation as exactly one of: `agent-pr`, `human-action`, or
  `manual-close`.
- Most implementation phases should be `agent-pr`: each one should be automatable
  through a child issue and later PR.
- Use `human-action` or `manual-close` only when a human must perform extra work,
  add a remark/update, and close that child issue manually.
- Include explicit scope boundaries, risk, validation/soak requirements, parent
  constraints/invariants, and non-goals for every phase.

Do not wrap the JSON in markdown fences. Signatures and AGENT markers are not
needed for this response.
-- {coder_signature}
"""


def build_plan_revision_prompt(
    issue_number: int,
    round_number: int,
    previous_plan: str,
    review: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
    unresolved_items: Sequence[UnresolvedReviewItem] = (),
    compact_context: bool = False,
    compact_prior: CompactPriorContext | None = None,
    compact_tail: CompactPlanTailContext | None = None,
) -> str:
    if compact_context:
        return _build_compact_plan_revision_prompt(
            issue_number,
            round_number,
            previous_plan,
            review,
            config,
            memory=memory,
            issue_context=issue_context,
            unresolved_items=unresolved_items,
            compact_prior=compact_prior,
            compact_tail=compact_tail,
        )
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    unresolved_items_block = _format_unresolved_plan_items(unresolved_items)
    human_requirements_context = _issue_human_requirements_prompt_context(
        issue_context,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    return f"""{reviewer_name} reviewed the implementation plan for GitHub issue #{issue_number} in {config.repo} and found blocking issues.

Revise the plan in this local checkout without editing code. Do not create a
branch, commit, push, or open a pull request during this planning stage.
{_coder_workdir_guidance(config, implementation=False)}
{_scratch_file_guidance()}
{human_requirements_context.block}{_coder_human_requirements_guidance(
    human_requirements_context,
    requirement_label="planning requirements",
    surfaced_requirement_instruction=(
        "Each bullet must explain how the revised plan covers that item or what remains risky or blocked."
    ),
)}
{_issue_context_block(issue_context)}
{unresolved_items_block}{_memory_block(memory)}

Previous plan:

{previous_plan}

{reviewer_name} plan review:

{review}

Revise the plan item by item instead of replying only with free-form prose. If
prior unresolved plan items are listed above, include this exact section in
your response and address each item ID exactly once:

### Prior plan review item dispositions
- [item-id] resolved: brief explanation of how the revised plan addresses it
- [item-id] still blocking: brief explanation if you cannot resolve it yet
- [item-id] same-plan: brief explanation of the remaining current-plan refinement
- [item-id] future: brief explanation only if a reviewer explicitly asked to move it to future work

Then provide the revised implementation plan with concrete file areas, edge
cases, and tests. Use `same-plan`, never `same-pr`, when describing plan-only
current-round refinements.

Use this mandatory structured JSON response format:

{{
  "schema_version": 1,
  "kind": "plan_revision",
  "state": "blocking",
  "summary": "Updated the plan to cover parser hard-fail behavior and resume state reconstruction.",
  "prior_plan_item_dispositions": [
    {{"item_id": "item-3", "disposition": "resolved", "note": "Added the carry-forward metadata step."}}
  ],
  "plan_steps": [
    "Update `src/coding_review_agent_loop/protocol.py` to hard-fail invalid structured plan payloads after JSON-prefix detection.",
    "Normalize structured plan rendering and metadata-backed resume behavior in `src/coding_review_agent_loop/orchestrator.py`.",
    "Extend prompts and targeted tests for structured planning flows."
  ]
}}
<!-- AGENT_PLAN_STATE: blocking -->
-- {coder_signature}

The orchestrator will normalize structured plan revisions into canonical
markdown for stored plan state, reviewer prompts, subject hashing, and resume.
Your response must start with exactly one top-level JSON object, with no prose
or code fences before it. If signed human requirements are present, put
`<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` and a `### Human requirements` section
after the JSON object and before the `AGENT_PLAN_STATE` footer. Otherwise, put
the `AGENT_PLAN_STATE` footer immediately after the JSON. Make the footer state
match the JSON state, and include only your standalone signature after the
footer.

This is planning round {round_number}. Use blocking to hand the revised plan back to
{reviewer_name}. Do not place your signature before the AGENT_PLAN_STATE footer. Your
response must end with, in this exact order:

<!-- AGENT_PLAN_STATE: blocking -->
-- {coder_signature}
"""


def _build_compact_plan_revision_prompt(
    issue_number: int,
    round_number: int,
    previous_plan: str,
    review: str,
    config: AgentLoopConfig,
    *,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    compact_prior: CompactPriorContext | None,
    compact_tail: CompactPlanTailContext | None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    human_requirements_context = _issue_human_requirements_prompt_context(
        issue_context,
        requirement_scope="planning requirements",
        full_omission_fallback="Fetch the issue discussion directly before revising the plan.",
    )
    stable_prefix = _compact_plan_stable_prefix(
        config=config,
        workdir_guidance=_coder_workdir_guidance(config, implementation=False),
        memory=memory,
        issue_context=issue_context,
        unresolved_items=unresolved_items,
        compact_prior=compact_prior,
        human_requirements_block=human_requirements_context.block,
        human_requirements_guidance=_coder_human_requirements_guidance(
            human_requirements_context,
            requirement_label="planning requirements",
            surfaced_requirement_instruction=(
                "Each bullet must explain how the revised plan covers that item or what remains risky or blocked."
            ),
        ),
        response_protocol=_plan_revision_schema_and_rules(),
    )
    subject_line = f"Current plan subject: {compact_tail.subject}" if compact_tail and compact_tail.subject else "Current plan subject: (unknown)"
    action = (
        compact_tail.action
        if compact_tail and compact_tail.action
        else "Revise the implementation plan to address the blocking plan review."
    )
    return f"""{stable_prefix}
{COMPACT_PLANNING_VOLATILE_TAIL_MARKER}

GitHub issue #{issue_number}
Planning round: {round_number}
Role: coder
Coder: {agent_display_name(config.coder)}
Reviewers: {reviewer_name}
{subject_line}
Action for this call: {action}

Previous implementation plan:

{previous_plan}

Blocking plan review payload:

{review}

Use blocking to hand the revised plan back to {reviewer_name}. Do not place
your signature before the AGENT_PLAN_STATE footer. Your response must end with,
in this exact order:

<!-- AGENT_PLAN_STATE: blocking -->
-- {coder_signature}
"""


def build_issue_implementation_prompt(
    issue_number: int,
    approved_plan: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    human_requirements_context = _issue_human_requirements_prompt_context(
        issue_context,
        requirement_scope="implementation requirements",
        full_omission_fallback="Fetch the issue discussion directly before implementing.",
    )
    return f"""Implement the approved plan for GitHub issue #{issue_number} in {config.repo}.

Use this local checkout as your workspace. Create a branch, implement the
approved plan, run relevant tests, commit, push, and open a pull request against
{config.base}.
{_coder_workdir_guidance(config)}
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}
{_issue_pr_reference_guidance(issue_number)}
{human_requirements_context.block}{_coder_human_requirements_guidance(
    human_requirements_context,
    requirement_label="implementation requirements",
)}
{_issue_context_block(issue_context)}
{_memory_block(memory)}

Approved implementation plan:

{approved_plan}

Do not wait for {reviewer_name} yourself; this local orchestrator will run
{reviewer_name} after you create the PR. Use blocking here to hand the PR to
{reviewer_name} for review. Do not place your signature before the AGENT_STATE
marker. Your response must end with, in this exact order:

<!-- AGENT_PR: <number> -->
<!-- AGENT_STATE: blocking -->
-- {coder_signature}
"""


def build_task_prompt(
    task_text: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    return f"""You have been given a free-form task to implement in {config.repo}.

Task:
{task_text}
{_memory_block(memory)}

Use this local checkout as your workspace. Decide between two paths:
{_coder_workdir_guidance(config)}

(a) If the task is clear enough to implement, create a branch, implement the
    change, run relevant tests, commit, push, and open a pull request against
    {config.base}. Do not wait for {reviewer_name}; this local orchestrator
    will run {reviewer_name} after you create the PR.
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}

(b) If the task is genuinely ambiguous or missing information that would change
    the implementation, do NOT write code. Instead, ask focused clarifying
    questions.

Prefer (a) when reasonable assumptions can be documented in the PR description;
choose (b) only for material ambiguity. Do not place your signature before the
AGENT_STATE or AGENT_CLARIFY marker. Your response must end with, in this exact
order — for (a):

<!-- AGENT_PR: <number> -->
<!-- AGENT_STATE: blocking -->
-- {coder_signature}

or for (b):

<!-- AGENT_CLARIFY -->
-- {coder_signature}
"""


def build_task_clarification_prompt(
    task_text: str,
    history: Sequence[tuple[str, str]],
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
) -> str:
    coder_signature = agent_signature(config.coder, config)
    qa_blocks = "\n\n".join(
        f"Round {idx + 1} questions from you:\n{questions}\n\n"
        f"Round {idx + 1} answers from the user:\n{answers}"
        for idx, (questions, answers) in enumerate(history)
    )
    return f"""Continuing the previous free-form task in {config.repo}.

Original task:
{task_text}
{_memory_block(memory)}

Clarification so far:

{qa_blocks}

Now proceed. Strongly prefer to implement the task and open a PR. Only ask
again if a critical detail is still missing.
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}

Do not place your signature before the AGENT_STATE or AGENT_CLARIFY marker.
Your response must end with, in this exact order:

For implementation:
<!-- AGENT_PR: <number> -->
<!-- AGENT_STATE: blocking -->
-- {coder_signature}

For another clarification round:
<!-- AGENT_CLARIFY -->
-- {coder_signature}
"""


def _compact_pr_review_issue_context_block(
    issue_context: IssueContext | None,
    pr_body: str | None,
) -> str:
    lines = ["Original PR context"]
    if pr_body:
        lines.extend(["", "PR body:", pr_body])
    if issue_context is not None:
        title = issue_context.title if issue_context.title else "(unknown)"
        body = issue_context.body if issue_context.body else "(none)"
        lines.extend(
            [
                "",
                f"Linked GitHub issue #{issue_context.number}",
                "",
                "Issue title:",
                title,
                "",
                "Issue body:",
                body,
            ]
        )
    lines.extend(
        [
            "",
            "Raw prior PR-review comments are omitted in compact PR review context. "
            "Durable reviewer findings must appear in the active unresolved ledger or "
            "the append-only compact prior ledger below.",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_pr_review_ledger_rules() -> str:
    return """Canonical compact PR review ledger rules

- Treat active prior unresolved review items as approval-critical until explicitly dispositioned.
- Treat append-only compact prior ledger entries as the canonical history for prior blocking and same-PR concerns that left the active ledger.
- Do not reinterpret, reorder, or rewrite prior compact ledger entries; append only new resolved or future-follow-up summaries after later review rounds.
- Future follow-ups are relevant in compact mode only when elevated, referenced, or preserved in the compact prior ledger.
"""


def _compact_coder_followup_block(
    summary: str | None,
    tests_run: Sequence[str] | None,
) -> str:
    if not summary and not tests_run:
        return ""
    lines = ["Latest coder follow-up summary"]
    if summary:
        lines.extend(["", summary])
    if tests_run:
        lines.extend(["", "Tests run:"] + [f"- {t}" for t in tests_run])
    lines.append("")
    return "\n".join(lines)


def _compact_pr_review_stable_prefix(
    *,
    config: AgentLoopConfig,
    workdir_guidance: str,
    memory: AgentMemoryContext | None,
    pr_metadata: PullRequestMetadata,
    issue_context: IssueContext | None,
    human_requirements: Sequence[HumanReviewRequirement] | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    compact_prior: CompactPriorContext | None,
    compact_coder_summary: str | None,
    compact_coder_tests_run: Sequence[str] | None,
    unresolved_items_guidance: str,
    followup_guidance: str,
    human_requirements_guidance: str,
) -> str:
    reviewer_group = format_agent_list(reviewers(config))
    coder_name = agent_display_name(config.coder)
    response_schema = f"""Use this mandatory structured PR review format. Start your response with
exactly one top-level JSON object and no prose or code fences before it:

{{
  "schema_version": 1,
  "kind": "pr_review",
  "state": "approved" | "blocking",
  "summary": "short reviewer summary",
  "blocking_items": ["render-only blocking bullet", "..."],
  "same_pr_followups": ["same-PR fix still required", "..."],
  "future_followups": ["future work after approval", "..."],
  "prior_item_dispositions": [
    {{"item_id": "item-1", "disposition": "resolved"}},
    {{"item_id": "item-2", "disposition": "future", "note": "brief reason"}}
  ]
}}

`blocking_items`, `same_pr_followups`, and `future_followups` have distinct
roles. `blocking_items` are merge-blocking defects, missing requirements,
regressions, security issues, or consistency gaps. `same_pr_followups` are
small, localized cleanup in touched files or directly adjacent code that should
be handled before merge, but is not itself a major correctness blocker.
`future_followups` are independent later work that remains valid after this PR
is merge-ready. Keep `blocking_items` and `same_pr_followups` mutually
exclusive: a single current-PR concern belongs in exactly one of those lists.
Before approving, self-check every `future_followups` entry: if it is trivial
or local to the current PR, reclassify it as `same_pr_followups` and return
`blocking`, or omit it if it is only a nit.

After the JSON object, include only:
1. optional `<!-- HUMAN_REQUIREMENTS_RESOLVED -->`
2. required `<!-- AGENT_STATE: approved -->` or `<!-- AGENT_STATE: blocking -->`
3. your standalone signature line (shown in volatile tail)

Do not include any extra prose, headings, bullets, or fenced blocks before or
after that structured response. The footer AGENT_STATE must match the JSON
`state`.

Use approved only if there are no blocking issues, no Same-PR follow-ups, and
no carried-forward unresolved items left active for this round.
Only items listed under `Prior unresolved review items from earlier rounds`
are eligible for dispositions in this round. If same-round findings from
other reviewers appear elsewhere in the PR discussion, treat them as
informational only and do not disposition them until a later round carries
them forward explicitly.
Item IDs visible only in `issue_context`, issue history, or planning comments
are informational only. This includes planning-stage `item-*` IDs and approved
plan future follow-ups, which are tracked separately. Do not include those IDs
in `prior_item_dispositions` or `prior_plan_item_dispositions` unless they are
also listed in the active `Prior unresolved review items from earlier rounds`
section above.
All configured reviewers ({reviewer_group}) must approve in the same round for
the pull request to be considered approved.
Focus on correctness, security, test coverage, and maintainability. Review the
full diff and any existing PR discussion. Do not make code changes in this
review step; report blocking findings if {coder_name} needs to fix anything.
Treat the GitHub PR checks block in the volatile tail as authoritative for current CI state.
Do not say or imply that tests passed globally unless the GitHub PR checks
state is `passing` or `no_checks`. If only a local subset passed while GitHub
checks are `failing`, `pending`, or `unavailable`, say that explicitly.
Do not defer your review to wait for CI checks to finish. Review the PR now and report any pending or failing check status in your findings.
When the PR changes files under `alembic/versions/`, verify migration topology:
new revisions should descend from the current head unless the PR intentionally
adds a merge migration.
"""
    non_future_items = [item for item in unresolved_items if item.status != "future"]
    return "\n".join(
        part.rstrip()
        for part in (
            "Compact cache-aware PR review context",
            f"Repository: {config.repo}",
            workdir_guidance,
            _scratch_file_guidance(),
            _review_command_policy(config, pr_metadata),
            response_schema,
            unresolved_items_guidance,
            followup_guidance,
            human_requirements_guidance,
            _memory_block(memory),
            _compact_pr_review_issue_context_block(issue_context, pr_metadata.body),
            _human_requirements_block(human_requirements),
            _canonical_pr_review_ledger_rules(),
            _format_unresolved_review_items(non_future_items)
            or "Prior unresolved review items from earlier rounds\n\n(none)\n",
            _compact_coder_followup_block(compact_coder_summary, compact_coder_tests_run),
            _compact_prior_ledger_block(compact_prior),
        )
        if part
    ) + "\n"


def _build_compact_pr_review_prompt(
    pr_number: int,
    round_number: int,
    config: AgentLoopConfig,
    *,
    reviewer: AgentName,
    pr_metadata: PullRequestMetadata,
    pr_checks: PullRequestChecks | None,
    memory: AgentMemoryContext | None,
    issue_context: IssueContext | None,
    human_requirements: Sequence[HumanReviewRequirement] | None,
    unresolved_items: Sequence[UnresolvedReviewItem],
    compact_prior: CompactPriorContext | None,
    compact_coder_summary: str | None,
    compact_coder_tests_run: Sequence[str] | None,
    compact_tail: CompactPrReviewTailContext | None,
) -> str:
    reviewer_name = agent_display_name(reviewer)
    reviewer_signature = agent_signature(reviewer, config)
    checks_block = f"{format_pr_checks(pr_checks)}\n" if pr_checks is not None else ""
    human_requirements_guidance = _human_requirements_review_guidance(human_requirements)
    non_future_items = [item for item in unresolved_items if item.status != "future"]
    if non_future_items:
        unresolved_items_guidance = _build_unresolved_items_guidance()
    else:
        unresolved_items_guidance = ""
    followup_guidance = _build_followup_guidance(config)
    stable_prefix = _compact_pr_review_stable_prefix(
        config=config,
        workdir_guidance=_coder_workdir_guidance(config, implementation=False, agent=reviewer),
        memory=memory,
        pr_metadata=pr_metadata,
        issue_context=issue_context,
        human_requirements=human_requirements,
        unresolved_items=unresolved_items,
        compact_prior=compact_prior,
        compact_coder_summary=compact_coder_summary,
        compact_coder_tests_run=compact_coder_tests_run,
        unresolved_items_guidance=unresolved_items_guidance,
        followup_guidance=followup_guidance,
        human_requirements_guidance=human_requirements_guidance,
    )
    head_sha = (compact_tail.head_sha if compact_tail else None) or pr_metadata.head_sha or "(unknown)"
    volatile_round = (compact_tail.round_number if compact_tail else None) or round_number
    action = (
        compact_tail.action if compact_tail and compact_tail.action else None
    ) or "Review the current PR diff for correctness, security, test coverage, and maintainability."
    title = pr_metadata.title or "(unknown)"
    head_branch = pr_metadata.head_branch or "(unknown)"
    base_branch = pr_metadata.base_branch or "(unknown)"
    url_line = f"- URL: {pr_metadata.url}\n" if pr_metadata.url else ""
    return f"""{stable_prefix}
{COMPACT_PR_REVIEW_VOLATILE_TAIL_MARKER}

PR #{pr_number} in {config.repo} (round {volatile_round})
- Title: {title}
- Head branch: {head_branch}
- Base branch: {base_branch}
- Head SHA: {head_sha}
{url_line}
Use the Head SHA above as authoritative for this review round. The orchestrator
has already refreshed and verified the assigned checkout. If its HEAD does not
match, report a blocking tooling mismatch instead of changing the checkout. Do
not report findings based on untracked files unless those files are present in
the PR diff.

{checks_block}Suggested commands:
- Read the verified checkout and local base-to-head diff first.
- Use `{config.gh_cmd} pr view {pr_metadata.number} --repo {pr_metadata.repo} --json comments,reviews` only if existing PR discussion is not already present in this prompt.

If a read-only shell/tool command requires confirmation in non-interactive mode,
do not retry repeatedly. Use the PR metadata and local checkout context already
available, or produce a blocking review explaining the limitation.

Reviewer: {reviewer_name}
Action for this call: {action}

Use blocking only for issues that should prevent merge. Do not place your
signature before the AGENT_STATE footer. Your response must end with, in this
exact order:

<!-- AGENT_STATE: approved -->
-- {reviewer_signature}

or:

<!-- AGENT_STATE: blocking -->
-- {reviewer_signature}
"""


def _build_unresolved_items_guidance() -> str:
    return """Prior unresolved items are present. Disposition every listed
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


def _build_followup_guidance(config: AgentLoopConfig) -> str:
    coder_name = agent_display_name(config.coder)
    if config.approved_followups == "ignore":
        return """Do not include Same-PR follow-ups, Future follow-ups, or legacy
Non-blocking follow-ups sections in approved reviews; this run is configured to
ignore approved-review follow-up sections. Mark the review blocking instead
when cleanup should be fixed before merge.
Trivial style nits in touched code should be omitted unless worth requiring
before merge; if required, they are current-PR work and the review is blocking,
not approved with Future follow-ups.
"""
    elif config.approved_followups.startswith("fix-and-"):
        return f"""For small, localized, low-risk cleanup that must still be fixed in this PR
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
Same-PR follow-ups will be sent back to {coder_name} and require another review
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
    else:
        return """If you approve but notice substantial work that is better handled separately in
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


def build_review_prompt(
    pr_number: int,
    round_number: int,
    config: AgentLoopConfig,
    *,
    reviewer: AgentName,
    pr_metadata: PullRequestMetadata | None = None,
    pr_checks: PullRequestChecks | None = None,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
    human_requirements: Sequence[HumanReviewRequirement] | None = None,
    unresolved_items: Sequence[UnresolvedReviewItem] | None = None,
    compact_context: bool = False,
    compact_prior: CompactPriorContext | None = None,
    compact_tail: CompactPrReviewTailContext | None = None,
    compact_coder_summary: str | None = None,
    compact_coder_tests_run: Sequence[str] | None = None,
) -> str:
    coder_name = agent_display_name(config.coder)
    reviewer_signature = agent_signature(reviewer, config)
    reviewer_group = format_agent_list(reviewers(config))
    metadata = pr_metadata or PullRequestMetadata(
        number=pr_number,
        repo=config.repo,
        title=None,
        head_branch=None,
        base_branch=None,
        head_sha=None,
        url=None,
    )
    if compact_context:
        return _build_compact_pr_review_prompt(
            pr_number,
            round_number,
            config,
            reviewer=reviewer,
            pr_metadata=metadata,
            pr_checks=pr_checks,
            memory=memory,
            issue_context=issue_context,
            human_requirements=human_requirements,
            unresolved_items=unresolved_items or [],
            compact_prior=compact_prior,
            compact_coder_summary=compact_coder_summary,
            compact_coder_tests_run=compact_coder_tests_run,
            compact_tail=compact_tail,
        )
    title = metadata.title or "(unknown)"
    head_branch = metadata.head_branch or "(unknown)"
    base_branch = metadata.base_branch or "(unknown)"
    head_sha = metadata.head_sha or "(unknown)"
    url_line = f"- URL: {metadata.url}\n" if metadata.url else ""
    checks_block = f"{format_pr_checks(pr_checks)}\n" if pr_checks is not None else ""
    human_requirements_guidance = _human_requirements_review_guidance(human_requirements)
    unresolved_items_block = _format_unresolved_review_items(unresolved_items)
    if unresolved_items:
        unresolved_items_guidance = _build_unresolved_items_guidance()
    else:
        unresolved_items_guidance = ""
    followup_guidance = _build_followup_guidance(config)
    return f"""Review pull request #{pr_number} in {config.repo} (round {round_number}).

PR metadata:
- Repo: {metadata.repo}
- PR: #{metadata.number}
- Title: {title}
- Head branch: {head_branch}
- Base branch: {base_branch}
- Head SHA: {head_sha}
{url_line}
Use this PR metadata as authoritative. The Head SHA above is the PR head this
review round is about. The orchestrator has already refreshed and verified the
assigned checkout. If its HEAD does not match, report a blocking tooling
mismatch instead of changing the checkout. Do not spend time discovering the
PR branch. Do not report findings based on untracked files unless those files
are present in the PR diff.
{_coder_workdir_guidance(config, implementation=False, agent=reviewer)}
{_scratch_file_guidance()}
{_review_command_policy(config, metadata)}
{checks_block}{_issue_context_block(issue_context)}
{_human_requirements_block(human_requirements)}
{unresolved_items_block}{_memory_block(memory)}

Suggested commands:
- Read the verified checkout and local base-to-head diff first.
- Use `{config.gh_cmd} pr view {metadata.number} --repo {metadata.repo} --json comments,reviews` only if existing PR discussion is not already present in this prompt.

If a read-only shell/tool command requires confirmation in non-interactive
mode, do not retry repeatedly. Use the PR metadata and local checkout context
already available, or produce a blocking review explaining the limitation.

Focus on correctness, security, test coverage, and maintainability. Review the
full diff and any existing PR discussion. Do not make code changes in this
review step; report blocking findings if {coder_name} needs to fix anything.
Treat the GitHub PR checks block above as authoritative for current CI state.
Do not say or imply that tests passed globally unless the GitHub PR checks
state is `passing` or `no_checks`. If only a local subset passed while GitHub
checks are `failing`, `pending`, or `unavailable`, say that explicitly.
Do not defer your review to wait for CI checks to finish. Review the PR now and report any pending or failing check status in your findings.
When the PR changes files under `alembic/versions/`, verify migration topology:
new revisions should descend from the current head unless the PR intentionally
adds a merge migration.
{human_requirements_guidance}
Only items listed under `Prior unresolved review items from earlier rounds`
are eligible for dispositions in this round. If same-round findings from
other reviewers appear elsewhere in the PR discussion, treat them as
informational only and do not disposition them until a later round carries
them forward explicitly.
Item IDs visible only in `issue_context`, issue history, or planning comments
are informational only. This includes planning-stage `item-*` IDs and approved
plan future follow-ups, which are tracked separately. Do not include those IDs
in `prior_item_dispositions` or `prior_plan_item_dispositions` unless they are
also listed in the active `Prior unresolved review items from earlier rounds`
section above.
{unresolved_items_guidance}
{followup_guidance}
Use blocking only for issues that should prevent merge.
All configured reviewers ({reviewer_group}) must approve in the same round for
the pull request to be considered approved.

Use this mandatory structured PR review format. Start your response with
exactly one top-level JSON object and no prose or code fences before it:

{{
  "schema_version": 1,
  "kind": "pr_review",
  "state": "approved" | "blocking",
  "summary": "short reviewer summary",
  "blocking_items": ["render-only blocking bullet", "..."],
  "same_pr_followups": ["same-PR fix still required", "..."],
  "future_followups": ["future work after approval", "..."],
  "prior_item_dispositions": [
    {{"item_id": "item-1", "disposition": "resolved"}},
    {{"item_id": "item-2", "disposition": "future", "note": "brief reason"}}
  ]
}}

`blocking_items`, `same_pr_followups`, and `future_followups` have distinct
roles. `blocking_items` are merge-blocking defects, missing requirements,
regressions, security issues, or consistency gaps. `same_pr_followups` are
small, localized cleanup in touched files or directly adjacent code that should
be handled before merge, but is not itself a major correctness blocker.
`future_followups` are independent later work that remains valid after this PR
is merge-ready. Keep `blocking_items` and `same_pr_followups` mutually
exclusive: a single current-PR concern belongs in exactly one of those lists.
Before approving, self-check every `future_followups` entry: if it is trivial
or local to the current PR, reclassify it as `same_pr_followups` and return
`blocking`, or omit it if it is only a nit.

After the JSON object, include only:
1. optional `<!-- HUMAN_REQUIREMENTS_RESOLVED -->`
2. required `<!-- AGENT_STATE: approved -->` or `<!-- AGENT_STATE: blocking -->`
3. your standalone signature line `-- {reviewer_signature}`

Do not include any extra prose, headings, bullets, or fenced blocks before or
after that structured response. The footer AGENT_STATE must match the JSON
`state`. Do not place your signature before the AGENT_STATE footer.
Use approved only if there are no blocking issues, no Same-PR follow-ups, and
no carried-forward unresolved items left active for this round. Your response
must end with, in this exact order:

<!-- AGENT_STATE: approved -->
-- {reviewer_signature}

or:

<!-- AGENT_STATE: blocking -->
-- {reviewer_signature}
"""


def build_followup_prompt(
    pr_number: int,
    round_number: int,
    review: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
    human_requirements: Sequence[HumanReviewRequirement] | None = None,
    *,
    human_requirements_context: CoderHumanRequirementsPromptContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    if human_requirements_context is None:
        human_requirements_context = render_coder_human_requirements_prompt_context(human_requirements)
    return f"""{reviewer_name} reviewed pull request #{pr_number} in {config.repo} and found blocking issues.

Address the review below in this local checkout. Pull/sync the PR branch if
needed, implement fixes, run relevant tests, commit, and push to the same PR.
Do not create a new PR.
{_coder_workdir_guidance(config)}
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}
{_issue_context_block(issue_context)}
{human_requirements_context.block}{_coder_human_requirements_guidance(human_requirements_context)}
{_memory_block(memory)}

{reviewer_name} review:

{review}

{_structured_coder_followup_guidance(
    reviewer_name=reviewer_name,
    human_requirements_context=human_requirements_context,
)}This is round {round_number}. Use blocking to hand the updated PR back to {reviewer_name}.
If you cannot safely address the review, explain why and still use the blocking
marker so a human can intervene. Do not place your signature before the
AGENT_STATE footer. Your response must end with, in this exact order:

<!-- AGENT_STATE: blocking -->
-- {coder_signature}
"""


def build_same_pr_followup_prompt(
    pr_number: int,
    round_number: int,
    review: str,
    config: AgentLoopConfig,
    memory: AgentMemoryContext | None = None,
    issue_context: IssueContext | None = None,
    human_requirements: Sequence[HumanReviewRequirement] | None = None,
    *,
    human_requirements_context: CoderHumanRequirementsPromptContext | None = None,
) -> str:
    reviewer_name = format_agent_list(reviewers(config))
    coder_signature = agent_signature(config.coder, config)
    if human_requirements_context is None:
        human_requirements_context = render_coder_human_requirements_prompt_context(human_requirements)
    return f"""{reviewer_name} requested same-PR follow-ups on pull request #{pr_number} in {config.repo}.

Address the follow-up items below in this local checkout. Pull/sync the PR
branch if needed, implement fixes, run relevant tests, commit, and push to the
same PR. Do not create a new PR.
{_coder_workdir_guidance(config)}
These same-PR follow-ups are intended to be small, localized cleanup for the
current PR. Keep the change narrowly scoped to the listed items. Do not take on
larger redesigns or unrelated future work; call that out instead. The PR
remains blocked pending another review round after this cleanup.
{_scratch_file_guidance()}
{_coder_test_reporting_guidance()}
{_issue_context_block(issue_context)}
{human_requirements_context.block}{_coder_human_requirements_guidance(human_requirements_context)}
{_memory_block(memory)}

Same-PR follow-ups:

{review}

{_structured_coder_followup_guidance(
    reviewer_name=reviewer_name,
    human_requirements_context=human_requirements_context,
)}This is round {round_number}. Use blocking to hand the updated PR back to {reviewer_name}.
If you cannot safely address the follow-ups, explain why and still use the
blocking marker so a human can intervene. Do not place your signature before
the AGENT_STATE footer. Your response must end with, in this exact order:

<!-- AGENT_STATE: blocking -->
-- {coder_signature}
"""
