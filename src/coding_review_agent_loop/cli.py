#!/usr/bin/env python3
"""Command-line entry point for the local agent review loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .agents.base import normalize_agent_name
from .agents.registry import (
    agent_display_name,
    agent_signature,
)
from .config import (
    DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS,
    DEFAULT_MAX_ROUNDS,
    DEFAULT_REPAIR_MODELS,
    DEFAULT_FLAT_CHILD_LIMIT,
    DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES,
    DEFAULT_CODER_TEST_COMMAND_TIMEOUT_SECONDS,
    AgentLoopConfig,
    config_from_args,
    ensure_agent_workdirs,
    ensure_distinct_workdirs,
    ensure_workdir,
    resolve_base_branch,
    reviewers,
)
from .errors import AgentLoopError, QuotaResetExceededError
from .managed_ci import (
    PREFLIGHT_INDETERMINATE,
    PREFLIGHT_KNOWN_NOT_READY,
    PREFLIGHT_STRICT_READY,
    ManagedCiProbeContext,
    evaluate_managed_ci_readiness,
    render_managed_ci_preflight,
)
from .managed_pr import create_managed_pr
from .github import (
    detect_repo,
    get_pr_head_sha,
    merge_pr,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
)
from .logging import agent_log_path, log
from .orchestrator import (
    run_discuss_loop,
    run_issue_loop,
    run_optional_tests,
    run_pr_loop,
    run_task_loop,
)
from .prompts import (
    build_followup_prompt,
    build_issue_prompt,
    build_review_prompt,
    build_task_clarification_prompt,
    build_task_prompt,
    format_agent_list,
)
from .protocol import is_clarification_request, parse_agent_state, parse_pr_number
from .runner import CommandResult, Runner, ensure_log_dir_ignored, run_foreground_test, tail_text
from .test_runtime import (
    TestRuntimeConfigurationError,
    inherited_timeout_ceiling,
    record_test_observation,
    resolve_timeout_seconds,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local coder -> reviewer PR review loop."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_review_parallel(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--review-parallel",
            action="store_true",
            help=(
                "Run same-round plan/PR reviewers concurrently instead of "
                "sequentially (#594). Every reviewer's prompt is built from the "
                "same pre-round state before any reviewer is launched, and "
                "same-round reviewers never see one another's feedback. "
                "Sequential remains the default. Requires a distinct workdir "
                "per reviewer, even with --allow-shared-dir. Discuss mode has "
                "its own --discuss-parallel flag."
            ),
        )

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo", help="GitHub repo as owner/name. Defaults to gh repo view.")
        subparser.add_argument(
            "--base",
            default=None,
            help=(
                "Base branch override. Defaults to the PR base for `pr`, otherwise "
                "the repository default branch."
            ),
        )
        subparser.add_argument(
            "--claude-dir",
            type=Path,
            default=None,
            help="Claude checkout. Defaults to a repo-scoped temporary checkout when Claude is active.",
        )
        subparser.add_argument(
            "--codex-dir",
            type=Path,
            default=None,
            help="Codex checkout. Defaults to a repo-scoped temporary checkout when Codex is active.",
        )
        subparser.add_argument(
            "--gemini-dir",
            type=Path,
            default=None,
            help="Gemini checkout. Defaults to a repo-scoped temporary checkout when Gemini is active.",
        )
        subparser.add_argument(
            "--antigravity-dir",
            type=Path,
            default=None,
            help="Antigravity (agy) checkout. Defaults to a repo-scoped temporary checkout when Antigravity is active.",
        )
        subparser.add_argument(
            "--coder",
            type=normalize_agent_name,
            choices=("claude", "codex", "gemini", "antigravity"),
            default="claude",
            help="Agent that creates and fixes the PR (default: claude).",
        )
        subparser.add_argument(
            "--reviewer",
            type=normalize_agent_name,
            choices=("claude", "codex", "gemini", "antigravity"),
            action="append",
            default=None,
            help=(
                "Agent that reviews the PR and gates approval. Repeat for multiple "
                "reviewers; all must approve (default: codex)."
            ),
        )
        subparser.add_argument("--allow-shared-dir", action="store_true")
        subparser.add_argument(
            "--max-rounds",
            type=int,
            default=DEFAULT_MAX_ROUNDS,
            help=f"Maximum review/revision rounds (default: {DEFAULT_MAX_ROUNDS}).",
        )
        subparser.add_argument(
            "--flat-child-limit",
            type=int,
            default=DEFAULT_FLAT_CHILD_LIMIT,
            metavar="COUNT",
            help=(
                "Maximum flat child issues owned by one parent across decomposition "
                f"and split materialization (default: {DEFAULT_FLAT_CHILD_LIMIT})."
            ),
        )
        subparser.add_argument("--auto-merge", action="store_true")
        subparser.add_argument(
            "--managed-ci-trusted-actor",
            default=None,
            help=(
                "GitHub login trusted for v2 managed CI; must match the repository "
                "Actions variable AGENT_LOOP_MANAGED_ACTOR."
            ),
        )
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument(
            "--implementation-coder",
            type=normalize_agent_name,
            choices=("claude", "codex", "gemini", "antigravity"),
            default=None,
            help=(
                "With issue --plan-first implementation, use this coder only after the "
                "plan is approved. Planning still uses --coder."
            ),
        )
        subparser.add_argument(
            "--implementation-coder-model",
            default="",
            help=(
                "Model to use for approved implementation and PR follow-up. If "
                "--implementation-coder is omitted, applies to --coder."
            ),
        )
        subparser.add_argument(
            "--implementation-codex-reasoning-effort",
            default="",
            help=(
                "Codex reasoning effort to use with --implementation-coder codex during "
                "approved implementation and PR follow-up."
            ),
        )
        subparser.add_argument("--claude-cmd", default="claude")
        subparser.add_argument("--codex-cmd", default="codex")
        subparser.add_argument("--gemini-cmd", default="gemini")
        subparser.add_argument("--antigravity-cmd", default="agy")
        subparser.add_argument(
            "--antigravity-print-timeout-seconds",
            type=int,
            default=DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS,
            metavar="SECONDS",
            help=(
                "Maximum wait for each agy --print invocation (default: 600). "
                "Overrides agy's five-minute print-mode default."
            ),
        )
        subparser.add_argument(
            "--repair-backend",
            choices=("antigravity", "gemini"),
            default="antigravity",
            help="Malformed-response repair backend (default: antigravity).",
        )
        subparser.add_argument(
            "--repair-model",
            action="append",
            default=None,
            help=(
                "Repair model to try. Repeat to configure an explicit fallback chain "
                f"(default: {DEFAULT_REPAIR_MODELS[0]} only)."
            ),
        )
        subparser.add_argument(
            "--repair-timeout-seconds",
            type=int,
            default=120,
            help="Per-attempt malformed-response repair timeout (default: 120).",
        )
        agy_model_group = subparser.add_mutually_exclusive_group()
        agy_model_group.add_argument(
            "--antigravity-model",
            default=None,
            help="Legacy Antigravity (agy) model. Mutually exclusive with --antigravity-models.",
        )
        agy_model_group.add_argument(
            "--antigravity-models",
            nargs="+",
            default=None,
            help="Ordered Antigravity fallback chain after provider capacity retries. Mutually exclusive with --antigravity-model.",
        )
        subparser.add_argument(
            "--antigravity-quota-signatures",
            nargs="+",
            default=list(DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES),
            help="Provider error substrings that permit Antigravity fallback (default: quota, high traffic, etc).",
        )
        subparser.add_argument(
            "--codex-model",
            default="",
            help=(
                "Codex model to run and stamp in the signature (#332). Mutually "
                "exclusive with passing --model via --codex-arg."
            ),
        )
        subparser.add_argument(
            "--codex-reasoning-effort",
            default="",
            help=(
                "Codex reasoning effort (e.g. low/medium/high) to run and stamp in the "
                "signature. Requires --codex-model because rollout detection is best-effort. "
                "Mutually exclusive with model_reasoning_effort via --codex-arg."
            ),
        )
        subparser.add_argument(
            "--gemini-model",
            default="",
            help="Gemini model to run and stamp in the signature. Mutually exclusive with --gemini-arg --model.",
        )
        subparser.add_argument(
            "--claude-model",
            default="",
            help=(
                "Claude model to run (CLI mode) / declare for the signature (host mode). "
                "Mutually exclusive with --claude-arg --model."
            ),
        )
        subparser.add_argument("--gh-cmd", default="gh")
        subparser.add_argument(
            "--dangerous-agent-permissions",
            action="store_true",
            help=(
                "Use permission-bypass defaults for configured agents. Only use in trusted "
                "local repositories: Claude gets --dangerously-skip-permissions and "
                "Codex gets --dangerously-bypass-approvals-and-sandbox, Gemini "
                "gets --yolo and --skip-trust, and Antigravity gets "
                "--dangerously-skip-permissions."
            ),
        )
        subparser.add_argument(
            "--claude-arg",
            action="append",
            default=None,
            help=(
                "Extra argument passed to claude (repeat for multiple). "
                "Providing any --claude-arg replaces the default entirely."
            ),
        )
        subparser.add_argument(
            "--codex-arg",
            action="append",
            default=None,
            help=(
                "Extra argument passed to codex exec (repeat for multiple). "
                "Providing any --codex-arg replaces the default entirely."
            ),
        )
        subparser.add_argument(
            "--gemini-arg",
            action="append",
            default=None,
            help=(
                "Extra argument passed to gemini (repeat for multiple). "
                "Providing any --gemini-arg replaces the default entirely."
            ),
        )
        subparser.add_argument(
            "--antigravity-arg",
            action="append",
            default=None,
            help=(
                "Extra argument passed to agy (repeat for multiple). "
                "Providing any --antigravity-arg replaces the default entirely."
            ),
        )
        subparser.add_argument(
            "--test-command",
            help=(
                "Optional command to run as a local test gate. By default it runs after "
                "coder changes before review and again after reviewer approval before auto-merge."
            ),
        )
        subparser.add_argument(
            "--coder-test-command-timeout-seconds",
            type=float,
            default=DEFAULT_CODER_TEST_COMMAND_TIMEOUT_SECONDS,
            metavar="SECONDS",
            help=(
                "Finite run-level ceiling and unknown-command watchdog for local coder test "
                f"commands (default: {DEFAULT_CODER_TEST_COMMAND_TIMEOUT_SECONDS}). Known commands "
                "may use a smaller learned recommendation."
            ),
        )
        pre_review_tests_group = subparser.add_mutually_exclusive_group()
        pre_review_tests_group.add_argument(
            "--pre-review-tests",
            dest="pre_review_tests",
            action="store_true",
            default=True,
            help="Run --test-command after coder changes before reviewer rounds (default).",
        )
        pre_review_tests_group.add_argument(
            "--no-pre-review-tests",
            dest="pre_review_tests",
            action="store_false",
            help="Do not run --test-command before reviewer rounds.",
        )
        subparser.add_argument(
            "--ci-timeout-seconds",
            type=int,
            default=1200,
            help="Maximum time to watch the full CI board before auto-merge (default: 1200).",
        )
        subparser.add_argument(
            "--ci-poll-interval-seconds",
            type=int,
            default=30,
            help="Polling interval for the full CI board before auto-merge (default: 30).",
        )
        subparser.add_argument(
            "--ci-startup-timeout-seconds",
            type=int,
            default=120,
            help=(
                "Maximum time to observe a newly materialized current-head CI run/check before "
                "stopping with a resumable command (default: 120)."
            ),
        )
        subparser.add_argument(
            "--watch-pending-ci",
            action=argparse.BooleanOptionalAction,
            dest="watch_pending_ci",
            default=None,
            help=(
                "For ordinary CI without --auto-merge, foreground-poll the full GitHub check "
                "board after approval and resume the coder if CI fails. Ordinary --auto-merge "
                "always uses this full-board gate; --no-watch-pending-ci is retained only as a "
                "compatibility warning. Managed exact-head qualification remains its own gate."
            ),
        )
        subparser.add_argument(
            "--ci-queued-grace-seconds",
            type=int,
            default=1200,
            help=(
                "How long a check-run may sit queued with no job started before it is "
                "treated as external GitHub Actions infrastructure blocking (for example "
                "a hosted-runner capacity outage) rather than a normal wait (default: 1200)."
            ),
        )
        subparser.add_argument(
            "--mergeability-poll-attempts",
            type=int,
            default=3,
            help=(
                "How many times to re-check GitHub mergeability while it reports "
                "'UNKNOWN' before treating it as unresolved (default: 3)."
            ),
        )
        subparser.add_argument(
            "--mergeability-poll-interval-seconds",
            type=int,
            default=5,
            help="Delay between mergeability re-checks while GitHub reports 'UNKNOWN' (default: 5).",
        )
        subparser.add_argument(
            "--quiet",
            action="store_true",
            help="Suppress progress logs.",
        )
        subparser.add_argument(
            "--log-dir",
            type=Path,
            default=Path(".agent-loop-logs"),
            help="Directory for salvage and usage artifacts (default: .agent-loop-logs).",
        )
        subparser.add_argument(
            "--subprocess-log-dir",
            type=Path,
            default=None,
            help=(
                "External directory for active subprocess captures. The default is a unique "
                "directory under the agent-loop cache; relative overrides are resolved from the "
                "primary agent directory, and paths inside managed checkouts are rejected."
            ),
        )
        subparser.add_argument(
            "--progress-interval-seconds",
            type=int,
            default=30,
            help="How often to print long-running agent heartbeats (default: 30).",
        )
        subparser.add_argument(
            "--agent-max-retries",
            type=int,
            default=2,
            help="Maximum transient retries; shared across the Antigravity fallback chain (default: 2).",
        )
        subparser.add_argument(
            "--agent-retry-backoff-seconds",
            type=int,
            nargs="+",
            default=[15, 45],
            help=(
                "Backoff delays in seconds for transient agent retries. The final value is reused "
                "when retries exceed the number of provided delays (default: 15 45). "
                "For session-limit scenarios (e.g. Claude per-project session caps that reset after "
                "a few minutes), use longer values such as: --agent-retry-backoff-seconds 180 300."
            ),
        )
        memory_group = subparser.add_mutually_exclusive_group()
        memory_group.add_argument(
            "--agent-memory",
            dest="agent_memory",
            action="store_true",
            default=True,
            help="Enable repo-scoped advisory agent memory (default).",
        )
        memory_group.add_argument(
            "--no-agent-memory",
            dest="agent_memory",
            action="store_false",
            help="Disable repo-scoped advisory agent memory.",
        )
        subparser.add_argument(
            "--refresh-agent-memory",
            action="store_true",
            help="Force regeneration of repo-level memory files before invoking agents.",
        )
        subparser.add_argument(
            "--agent-memory-dir",
            type=Path,
            default=None,
            help=(
                "Directory for repo memory. Defaults to a repo-scoped user cache; "
                "relative explicit paths are resolved inside the coder checkout."
            ),
        )
        subparser.add_argument(
            "--refresh-test-profile",
            action="store_true",
            help="Regenerate the cached execution/test profile before invoking agents.",
        )
        salvage_comments_group = subparser.add_mutually_exclusive_group()
        salvage_comments_group.add_argument(
            "--salvage-comments",
            dest="salvage_comments",
            action="store_true",
            default=True,
            help=(
                "Post a hidden AGENT_SALVAGE marker comment to the issue when a mutating "
                "implementation attempt fails, so a rerun with a different coder/workdir/"
                "machine can still discover the latest salvage context (default)."
            ),
        )
        salvage_comments_group.add_argument(
            "--no-salvage-comments",
            dest="salvage_comments",
            action="store_false",
            help="Do not post GitHub salvage breadcrumb comments for failed implementation attempts.",
        )
        subparser.add_argument(
            "--salvage-comment-patch-max-bytes",
            type=int,
            default=20000,
            help=(
                "Maximum size of a partial patch embedded in a GitHub salvage comment; "
                "larger (or unsafe) patches are omitted with a local-only note (default: 20000)."
            ),
        )
        subparser.add_argument(
            "--approved-followups",
            choices=("ignore", "summarize", "issue", "fix-and-summarize", "fix-and-issue"),
            default="ignore",
            help=(
                "How to handle structured follow-ups in approved reviews "
                "('ignore', 'summarize', 'issue', 'fix-and-summarize', or "
                "'fix-and-issue'; default: ignore)."
            ),
        )
        subparser.add_argument(
            "--planning-context-mode",
            choices=("full", "compact"),
            default="compact",
            help=(
                "Planning prompt context mode. Compact uses a cache-aware canonical "
                "context after the first complete planning round (default: compact)."
            ),
        )
        subparser.add_argument(
            "--pr-review-context-mode",
            choices=("full", "compact"),
            default="full",
            help=(
                "PR review prompt context mode. Compact omits raw prior PR-review "
                "transcript from round 2 onward, sending only a structured summary "
                "(default: full)."
            ),
        )

    issue = subparsers.add_parser("issue", help="Ask the coder to fix an issue, then review it.")
    issue.add_argument("issue_number", type=int)
    issue.add_argument(
        "--expected-closing-issue",
        action="append",
        type=int,
        metavar="POSITIVE_ID",
        help=(
            "Additional issue ID that this single implementation PR is authoritative to close. "
            "Repeatable; values are sorted and deduplicated."
        ),
    )
    issue.add_argument(
        "--supersede-expected-closing-contract",
        action="store_true",
        help="Explicitly widen a recovered expected-closing contract with a proper superset.",
    )
    issue.add_argument(
        "--plan-first",
        action="store_true",
        help=(
            "Run an issue planning/review stage before implementation. By default, "
            "stop after the plan is approved and post the outcome to the issue."
        ),
    )
    issue.add_argument(
        "--implement-after-approval",
        action="store_true",
        help="With --plan-first, continue into implementation after reviewers approve the plan.",
    )
    issue.add_argument(
        "--plan-execution-mode",
        choices=("plan-only", "decompose-only", "implement-one-shot", "implement-by-phase"),
        default=None,
        help=(
            "With --plan-first, choose what happens after approval: plan-only, "
            "decompose-only, implement-one-shot, or implement-by-phase. "
            "Defaults to plan-only unless --implement-after-approval is used. "
            "decompose-only/implement-by-phase already create one detailed child "
            "issue per phase; do not also pass --materialize-split-issues for the "
            "same run. See docs/local_agent_loop.md#phased-decomposition-versus-split-materialization."
        ),
    )
    issue.add_argument(
        "--materialize-split-issues",
        action="store_true",
        help=(
            "File a linked child GitHub issue for each remaining discuss `split` proposal "
            "or plan `deferred_stages` entry instead of leaving them as unfiled text "
            "(default: off, warning-only). Do not combine with "
            "--plan-execution-mode decompose-only or implement-by-phase, which already "
            "create detailed child issues. See "
            "docs/local_agent_loop.md#phased-decomposition-versus-split-materialization."
        ),
    )
    issue.add_argument(
        "--split-stage",
        type=int,
        default=None,
        metavar="CHILD_ISSUE_NUMBER",
        help=(
            "When the parent issue's split proposals were already fully materialized into "
            "child issues, explicitly select which child stage this run implements instead "
            "of relying on a unique plan-title match."
        ),
    )
    add_common(issue)
    issue.add_argument(
        "--managed-ci",
        action="store_true",
        help=(
            "Explicitly activate managed exact-head CI and qualify the approved head. "
            "Without --auto-merge, leave a successfully qualified PR ready for manual "
            "head-guarded merging. Requires --managed-ci-trusted-actor."
        ),
    )
    issue.add_argument(
        "--allow-unprotected-managed-ci", action="store_true",
        help=(
            "Per-invocation waiver for issue-created managed CI, including a safe PR-mode "
            "resume of its existing draft, when GitHub cannot independently enforce "
            "final-ci/exact-head. Requires an effective managed-CI request and "
            "--managed-ci-trusted-actor; "
            "never applies to arbitrary PR adoption."
        ),
    )
    add_review_parallel(issue)

    pr = subparsers.add_parser("pr", help="Run the reviewer/coder loop on an existing PR.")
    pr.add_argument("pr_number", type=int)
    pr.add_argument(
        "--expected-closing-issue",
        action="append",
        type=int,
        metavar="POSITIVE_ID",
        help="Complete expected-closing issue set for this existing PR; repeatable.",
    )
    pr.add_argument(
        "--supersede-expected-closing-contract",
        action="store_true",
        help="Explicitly widen a recovered expected-closing contract with a proper superset.",
    )
    add_common(pr)
    pr.add_argument(
        "--managed-ci",
        action="store_true",
        help=(
            "Explicitly activate managed exact-head CI without implying a merge. "
            "A successful run leaves the live qualified head ready for manual "
            "head-guarded merging. Requires --managed-ci-trusted-actor."
        ),
    )
    pr.add_argument(
        "--allow-unprotected-managed-ci", action="store_true",
        help=(
            "Per-invocation waiver for issue-created managed CI resume when GitHub cannot "
            "independently enforce final-ci/exact-head. Requires an effective managed-CI "
            "request and "
            "--managed-ci-trusted-actor; never authorizes arbitrary PR adoption."
        ),
    )
    pr.add_argument(
        "--managed-ci-adopt-existing-pr",
        action="store_true",
        help=(
            "Explicitly adopt an eligible already-open PR into the separately advertised "
            "v2 managed-CI protocol. Requires an effective managed-CI request and "
            "--managed-ci-trusted-actor."
        ),
    )
    add_review_parallel(pr)

    managed_pr = subparsers.add_parser(
        "managed-pr",
        help="Create a managed-CI draft from an existing branch, then review it.",
    )
    managed_pr.add_argument(
        "--head",
        required=True,
        help="Existing same-repository source branch whose exact head will be used.",
    )
    managed_pr.add_argument("--title", required=True, help="Pull-request title.")
    managed_pr.add_argument(
        "--body-file",
        type=Path,
        default=None,
        help="Read the pull-request body from this file (use '-' for stdin; default: empty).",
    )
    managed_pr.add_argument(
        "--expected-closing-issue",
        action="append",
        type=int,
        metavar="POSITIVE_ID",
        help="Complete expected-closing issue set for this managed PR; repeatable.",
    )
    add_common(managed_pr)
    managed_pr.add_argument(
        "--managed-ci",
        action="store_true",
        help=(
            "Explicitly activate managed exact-head CI without implying a merge. "
            "A successful run leaves the live qualified head ready for manual "
            "head-guarded merging. Requires --managed-ci-trusted-actor."
        ),
    )
    managed_pr.add_argument(
        "--allow-unprotected-managed-ci",
        action="store_true",
        help=(
            "Per-invocation waiver when GitHub cannot independently enforce final-ci/exact-head. "
            "Requires --managed-ci or --auto-merge and --managed-ci-trusted-actor."
        ),
    )
    add_review_parallel(managed_pr)

    managed_ci = subparsers.add_parser("managed-ci", help="Inspect managed-CI readiness without writes.")
    managed_ci_subparsers = managed_ci.add_subparsers(dest="managed_ci_command", required=True)
    preflight = managed_ci_subparsers.add_parser("preflight", help="Read-only managed-CI readiness report.")
    preflight.add_argument("--repo", required=True, help="GitHub repository as owner/name.")
    preflight.add_argument("--base", default=None, help="Base branch (defaults to repository default branch).")
    preflight.add_argument("--trusted-actor", required=True, help="Expected authenticated GitHub login.")
    preflight.add_argument("--gh-cmd", default="gh", help="GitHub CLI executable (default: gh).")

    task = subparsers.add_parser(
        "task",
        help="Ask the coder to implement a free-form task, then review it.",
    )
    task.add_argument(
        "task_text",
        nargs="?",
        default=None,
        help="Free-form task description. Use --task-file to read from a file instead.",
    )
    task.add_argument(
        "--task-file",
        type=Path,
        default=None,
        help="Read task description from this file (use '-' for stdin).",
    )
    task.add_argument(
        "--interactive",
        action="store_true",
        help="Allow the coder to request clarification via stdin before implementing.",
    )
    task.add_argument(
        "--max-clarification-rounds",
        type=int,
        default=3,
        help="Maximum clarification rounds when --interactive is set (default: 3).",
    )
    add_common(task)
    add_review_parallel(task)

    discuss = subparsers.add_parser(
        "discuss",
        help="Run reviewers on an issue and post a consensus outcome comment.",
    )
    discuss.add_argument("issue_number", type=int)
    discuss.add_argument(
        "--discuss-max-rounds",
        type=int,
        default=2,
        help="Maximum debate rounds after the initial discuss round (default: 2).",
    )
    discuss.add_argument(
        "--discuss-analyzer",
        type=normalize_agent_name,
        choices=("claude", "codex", "gemini", "antigravity"),
        default=None,
        help=(
            "Optional analyzer agent that summarizes each non-final debate round "
            "into a structured agenda for the next round. May coincide with a "
            "--reviewer. Omit for plain direct deliberation (default: none)."
        ),
    )
    discuss.add_argument(
        "--discuss-research",
        choices=("none", "required", "auto"),
        default="none",
        help=(
            "Research policy for debaters (default: none). 'none' forbids online "
            "research; 'required' makes every debater research and cite sources; "
            "'auto' lets debaters (and the analyzer, if configured) decide using "
            "conservative triggers such as current vendor behavior, pricing, "
            "quotas, model availability, policies/laws, dependency behavior, or "
            "tool comparisons."
        ),
    )
    discuss.add_argument(
        "--discuss-result-mode",
        choices=("triage", "answer"),
        default="triage",
        help=(
            "Discuss result contract (default: triage). `answer` produces an "
            "open-ended consensus answer or explicit human escalation; omitting "
            "this flag preserves the legacy implementation-triage votes."
        ),
    )
    discuss.add_argument(
        "--discuss-parallel",
        action="store_true",
        help=(
            "Run same-round debaters concurrently instead of sequentially. The "
            "analyzer and summary still run only after every debater finishes "
            "(or the failure policy fires). Requires a distinct workdir per "
            "debater, even with --allow-shared-dir."
        ),
    )
    discuss.add_argument(
        "--discuss-debater-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Wall-clock limit for each debater turn in seconds (default: none). "
            "A timed-out debater is treated per --discuss-on-debater-failure "
            "with failure category 'timeout'."
        ),
    )
    discuss.add_argument(
        "--discuss-on-debater-failure",
        choices=("fail", "partial"),
        default="fail",
        help=(
            "Policy when a debater turn fails or times out (default: fail). "
            "'fail' aborts the run after in-flight debaters settle; 'partial' "
            "continues the round when at least two debaters produced votes and "
            "records the failures in the round summary. A partial round never "
            "declares final consensus."
        ),
    )
    discuss.add_argument(
        "--materialize-split-issues",
        action="store_true",
        help=(
            "On a `split` consensus, file a linked child GitHub issue for each proposed "
            "sub-issue instead of leaving them as unfiled text (default: off, warning-only). "
            "See docs/local_agent_loop.md#phased-decomposition-versus-split-materialization "
            "for when to use this versus a plan-first decompose-only/implement-by-phase run."
        ),
    )
    add_common(discuss)

    run_tests = subparsers.add_parser(
        "run-tests",
        help="Run one foreground test command with a finite watchdog and optional runtime memory.",
    )
    run_tests.add_argument(
        "--timeout-seconds",
        default=None,
        metavar="SECONDS",
        help="Chosen whole-command watchdog for this invocation; it cannot exceed the inherited ceiling.",
    )
    run_tests.add_argument(
        "--memory-dir",
        type=Path,
        default=None,
        help="Optional repo memory directory in which agent-loop records this measured run.",
    )
    run_tests.add_argument(
        "inner_argv",
        nargs=argparse.REMAINDER,
        help="Use `--` before the command to run.",
    )

    return parser


def _resolve_task_text(args: argparse.Namespace) -> str:
    if args.task_text and args.task_file:
        raise AgentLoopError("Pass either a positional task or --task-file, not both.")
    if args.task_file is not None:
        if str(args.task_file) == "-":
            text = sys.stdin.read()
        else:
            text = args.task_file.read_text(encoding="utf-8")
    elif args.task_text is not None:
        text = args.task_text
    else:
        raise AgentLoopError("Provide a task description (positional argument or --task-file).")
    if not text.strip():
        raise AgentLoopError("Task description is empty.")
    return text


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-tests":
        raw_inner = list(args.inner_argv)
        if raw_inner[:1] == ["--"]:
            raw_inner = raw_inner[1:]
        try:
            if not raw_inner:
                raise TestRuntimeConfigurationError(
                    "run-tests requires `--` followed by a non-empty command."
                )
            policy = inherited_timeout_ceiling()
            chosen = resolve_timeout_seconds(args.timeout_seconds, policy_ceiling=policy)
            result = run_foreground_test(
                raw_inner,
                cwd=Path.cwd(),
                timeout_seconds=chosen,
                dry_run=False,
            )
            record_test_observation(
                args.memory_dir,
                argv=raw_inner,
                cwd=Path.cwd(),
                outcome=result.outcome,
                elapsed_seconds=result.elapsed_seconds,
                attempted_timeout_seconds=chosen,
                policy_ceiling_seconds=policy,
                returncode=result.returncode,
            )
            return int(result.returncode if result.returncode is not None else 1)
        except (AgentLoopError, OSError, ValueError) as exc:
            print(f"agent-loop: {exc}", file=sys.stderr)
            return 1
    if args.command == "managed-ci" and args.managed_ci_command == "preflight":
        try:
            context = ManagedCiProbeContext(args.repo, args.gh_cmd, Path.cwd())
            result = evaluate_managed_ci_readiness(
                Runner(dry_run=False), context=context, base=args.base, trusted_actor=args.trusted_actor
            )
            print(render_managed_ci_preflight(result, repo=args.repo, base=args.base or "<default>", trusted_actor=args.trusted_actor))
            if result.state == "strict_ready":
                return PREFLIGHT_STRICT_READY
            if result.state == "indeterminate":
                return PREFLIGHT_INDETERMINATE
            return PREFLIGHT_KNOWN_NOT_READY
        except AgentLoopError as exc:
            print(f"agent-loop: {exc}", file=sys.stderr)
            return PREFLIGHT_INDETERMINATE
    runner = Runner(dry_run=args.dry_run)
    try:
        # Preserve tokens (rather than a rendered command) so timeout guidance can
        # be safely shell-quoted locally. Programmatic callers have no sys.argv.
        invocation = tuple([sys.argv[0], *sys.argv[1:]]) if argv is None else tuple(["agent-loop", *argv])
        explicit_managed_ci = bool(getattr(args, "managed_ci", False))
        if explicit_managed_ci:
            if args.command not in {"issue", "pr", "managed-pr"}:
                raise AgentLoopError("--managed-ci is only supported with issue, pr, or managed-pr.")
            if not (args.managed_ci_trusted_actor or "").strip():
                raise AgentLoopError("--managed-ci requires --managed-ci-trusted-actor.")
        config = config_from_args(args, runner, invocation_argv=invocation)
        if config.auto_merge and config.watch_pending_ci_explicit and not config.watch_pending_ci:
            print(
                "agent-loop: warning: --no-watch-pending-ci is retained for compatibility "
                "and no longer disables the ordinary auto-merge full-board watcher.",
                file=sys.stderr,
            )
        implementation_override_requested = (
            args.implementation_coder is not None
            or args.implementation_coder_model
            or args.implementation_codex_reasoning_effort
        )
        if args.command != "issue" and implementation_override_requested:
            raise AgentLoopError("--implementation-coder options are only supported with issue --plan-first.")
        if getattr(args, "supersede_expected_closing_contract", False):
            if args.command not in {"issue", "pr"}:
                raise AgentLoopError(
                    "--supersede-expected-closing-contract is only supported with `agent-loop issue` or `agent-loop pr`."
                )
            if getattr(args, "expected_closing_issue", None) is None:
                raise AgentLoopError(
                    "--supersede-expected-closing-contract requires an explicit full "
                    "--expected-closing-issue declaration."
                )
        if getattr(args, "managed_ci_adopt_existing_pr", False):
            if args.command != "pr":
                raise AgentLoopError("--managed-ci-adopt-existing-pr is only supported with `agent-loop pr <n>`.")
            if not (args.auto_merge or getattr(args, "managed_ci", False)):
                raise AgentLoopError("--managed-ci-adopt-existing-pr requires --managed-ci or --auto-merge.")
            if not (args.managed_ci_trusted_actor or "").strip():
                raise AgentLoopError(
                    "--managed-ci-adopt-existing-pr requires --managed-ci-trusted-actor."
                )
        if getattr(args, "allow_unprotected_managed_ci", False):
            if args.command not in {"issue", "pr", "managed-pr"}:
                raise AgentLoopError("--allow-unprotected-managed-ci is only supported with issue, pr, or managed-pr.")
            if not (args.auto_merge or getattr(args, "managed_ci", False)):
                raise AgentLoopError("--allow-unprotected-managed-ci requires --managed-ci or --auto-merge.")
            if not (args.managed_ci_trusted_actor or "").strip():
                raise AgentLoopError("--allow-unprotected-managed-ci requires --managed-ci-trusted-actor.")
            if getattr(args, "managed_ci_adopt_existing_pr", False):
                raise AgentLoopError("--allow-unprotected-managed-ci cannot be used with --managed-ci-adopt-existing-pr.")
        if args.command == "issue":
            if args.implement_after_approval and not args.plan_first:
                raise AgentLoopError("--implement-after-approval requires --plan-first.")
            if args.plan_execution_mode and not args.plan_first:
                raise AgentLoopError("--plan-execution-mode requires --plan-first.")
            if implementation_override_requested and not args.plan_first:
                raise AgentLoopError("--implementation-coder options require --plan-first.")
            plan_execution_mode = args.plan_execution_mode
            if plan_execution_mode is None:
                plan_execution_mode = (
                    "implement-one-shot" if args.implement_after_approval else "plan-only"
                )
            elif args.implement_after_approval and plan_execution_mode != "implement-one-shot":
                raise AgentLoopError(
                    "--implement-after-approval is only compatible with "
                    "--plan-execution-mode implement-one-shot."
                )
            if (
                plan_execution_mode in {"decompose-only", "implement-by-phase"}
                and getattr(args, "materialize_split_issues", False)
            ):
                raise AgentLoopError(
                    "--materialize-split-issues cannot be combined with "
                    "--plan-execution-mode decompose-only or implement-by-phase; "
                    "those modes select one child topology source."
                )
            config = AgentLoopConfig(
                **{
                    **config.__dict__,
                    "plan_execution_mode": plan_execution_mode,
                }
            )
            return run_issue_loop(
                runner,
                issue_number=args.issue_number,
                config=config,
                plan_first=args.plan_first,
                implement_after_approval=plan_execution_mode == "implement-one-shot",
            )
        if args.command == "pr":
            return run_pr_loop(runner, pr_number=args.pr_number, config=config)
        if args.command == "managed-pr":
            if not (args.auto_merge or args.managed_ci):
                raise AgentLoopError("managed-pr requires --managed-ci or --auto-merge.")
            if not (args.managed_ci_trusted_actor or "").strip():
                raise AgentLoopError("managed-pr requires --managed-ci-trusted-actor.")
            if args.dry_run:
                raise AgentLoopError("managed-pr does not support --dry-run because it must verify live GitHub state.")
            if args.body_file is None:
                body = ""
            elif str(args.body_file) == "-":
                body = sys.stdin.read()
            else:
                body = args.body_file.read_text(encoding="utf-8")
            config = resolve_base_branch(config, runner)
            ensure_agent_workdirs(config, runner)
            handoff = create_managed_pr(
                runner,
                config=config,
                source_branch=args.head,
                title=args.title,
                body=body,
            )
            run_kwargs = {
                "pr_number": handoff.pr_number,
                "config": handoff.config,
                "workdirs_ready": True,
            }
            if handoff.source_branch is not None:
                run_kwargs["managed_pr_origin"] = (
                    handoff.source_branch,
                    handoff.source_sha,
                    handoff.managed_branch,
                    handoff.config.managed_ci_expected_override_nonce,
                )
            return run_pr_loop(runner, **run_kwargs)
        if args.command == "task":
            task_text = _resolve_task_text(args)
            return run_task_loop(
                runner,
                task_text=task_text,
                config=config,
                interactive=getattr(args, "interactive", False),
                max_clarification_rounds=getattr(args, "max_clarification_rounds", 0),
            )
        if args.command == "discuss":
            return run_discuss_loop(
                runner,
                issue_number=args.issue_number,
                config=config,
                discuss_max_rounds=getattr(args, "discuss_max_rounds", 2),
            )
        parser.error(f"unknown command: {args.command}")
    except QuotaResetExceededError as exc:
        print(f"agent-loop: {exc}", file=sys.stderr)
        return QuotaResetExceededError.EXIT_CODE
    except AgentLoopError as exc:
        print(f"agent-loop: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
