#!/usr/bin/env python3
"""Command-line entry point for the local agent review loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .agents.registry import (
    agent_display_name,
    agent_signature,
)
from .config import (
    DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES,
    AgentLoopConfig,
    config_from_args,
    ensure_agent_workdirs,
    ensure_distinct_workdirs,
    ensure_workdir,
    reviewers,
)
from .errors import AgentLoopError, QuotaResetExceededError
from .github import (
    detect_repo,
    get_check_status,
    get_pr_head_sha,
    merge_pr,
    post_pr_comment,
    validate_open_issue,
    validate_open_pr,
    wait_for_ci,
)
from .logging import agent_log_path, log
from .orchestrator import (
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
from .runner import CommandResult, Runner, ensure_log_dir_ignored, tail_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local coder -> reviewer PR review loop."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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
            choices=("claude", "codex", "gemini", "antigravity"),
            default="claude",
            help="Agent that creates and fixes the PR (default: claude).",
        )
        subparser.add_argument(
            "--reviewer",
            choices=("claude", "codex", "gemini", "antigravity"),
            action="append",
            default=None,
            help=(
                "Agent that reviews the PR and gates approval. Repeat for multiple "
                "reviewers; all must approve (default: codex)."
            ),
        )
        subparser.add_argument("--allow-shared-dir", action="store_true")
        subparser.add_argument("--max-rounds", type=int, default=5)
        subparser.add_argument("--auto-merge", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument("--claude-cmd", default="claude")
        subparser.add_argument("--codex-cmd", default="codex")
        subparser.add_argument("--gemini-cmd", default="gemini")
        subparser.add_argument("--antigravity-cmd", default="agy")
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
            help="Ordered chain of Antigravity models to try on quota exhaustion. Mutually exclusive with --antigravity-model.",
        )
        subparser.add_argument(
            "--antigravity-quota-signatures",
            nargs="+",
            default=list(DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES),
            help="Output substrings that trigger model fallback (default: quota, rate limit, etc).",
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
            "--ci-check-name",
            default="test",
            help="GitHub check-run name required before --auto-merge (default: test).",
        )
        subparser.add_argument(
            "--ci-timeout-seconds",
            type=int,
            default=1200,
            help="Maximum time to wait for the CI check before auto-merge (default: 1200).",
        )
        subparser.add_argument(
            "--ci-poll-interval-seconds",
            type=int,
            default=30,
            help="Polling interval for the CI check before auto-merge (default: 30).",
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
            help="Directory for agent subprocess logs (default: .agent-loop-logs).",
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
            help="Maximum retries for narrow transient agent/model failures (default: 2).",
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
            "Defaults to plan-only unless --implement-after-approval is used."
        ),
    )
    add_common(issue)

    pr = subparsers.add_parser("pr", help="Run the reviewer/coder loop on an existing PR.")
    pr.add_argument("pr_number", type=int)
    add_common(pr)

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
    runner = Runner(dry_run=args.dry_run)
    try:
        config = config_from_args(args, runner)
        if args.command == "issue":
            if args.implement_after_approval and not args.plan_first:
                raise AgentLoopError("--implement-after-approval requires --plan-first.")
            if args.plan_execution_mode and not args.plan_first:
                raise AgentLoopError("--plan-execution-mode requires --plan-first.")
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
        if args.command == "task":
            task_text = _resolve_task_text(args)
            return run_task_loop(
                runner,
                task_text=task_text,
                config=config,
                interactive=getattr(args, "interactive", False),
                max_clarification_rounds=getattr(args, "max_clarification_rounds", 0),
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
