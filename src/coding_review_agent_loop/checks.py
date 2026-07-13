"""Local test and GitHub PR check gate helpers."""

from __future__ import annotations

from .config import AgentLoopConfig
from .github import PullRequestChecks
from .logging import log
from .runner import Runner
from .workdirs import active_workdir


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    runner.run(config.test_command, cwd=active_workdir(config))
    log(config, "Local test command passed")


def run_pre_review_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.pre_review_tests or not config.test_command:
        return
    log(config, f"Running pre-review test command: {' '.join(config.test_command)}")
    runner.run(config.test_command, cwd=active_workdir(config))
    log(config, "Pre-review test command passed")


def _format_pr_checks_comment(pr_number: int, state: str, details: list[str]) -> str:
    headline = {
        "failing": f"GitHub PR checks are failing for PR #{pr_number}.",
        "pending": f"GitHub PR checks are still pending for PR #{pr_number}.",
        "unavailable": f"GitHub PR check status is unavailable for PR #{pr_number}.",
    }[state]
    lines = [
        headline,
        "",
        "Reviewer approvals do not make this PR merge-ready until GitHub PR checks are green, or the PR explicitly states that only a local subset passed.",
        "",
    ]
    lines.extend(f"- {detail}" for detail in details)
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _pr_check_blocking_review(pr_number: int, state: str, details: list[str]) -> str:
    headline = {
        "failing": "GitHub PR checks are failing and must be resolved before approval.",
        "pending": "GitHub PR checks are still pending, so this PR cannot be treated as merge-ready yet.",
        "unavailable": "GitHub PR check status is unavailable, so merge readiness cannot be confirmed yet.",
    }[state]
    lines = [headline, ""]
    lines.extend(f"- {detail}" for detail in details)
    lines.extend(
        [
            "",
            "Do not claim global test success unless GitHub PR checks are green. If only local tests passed, say that explicitly.",
        ]
    )
    return "\n".join(lines)


def _pending_ci_stop_message(pr_number: int, state: str, details: list[str]) -> str:
    headline = {
        "pending": f"Reviewers approved PR #{pr_number}, but GitHub checks are still pending.",
        "unavailable": (
            f"Reviewers approved PR #{pr_number}, but GitHub check status is unavailable."
        ),
    }[state]
    lines = [
        headline,
        "",
        _pending_ci_stop_guidance(state),
        "",
    ]
    lines.extend(f"- {detail}" for detail in details)
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _pending_ci_status_summary(state: str) -> str:
    return {
        "pending": "GitHub checks are still pending",
        "unavailable": "GitHub check status is unavailable",
    }[state]


def _pending_ci_stop_guidance(state: str) -> str:
    wait = {
        "pending": "Wait for CI.",
        "unavailable": "Wait for GitHub check status to become available.",
    }[state]
    return (
        "This run cannot confirm the PR is merge-ready yet. "
        f"{wait} If checks pass, you can merge manually; no rerun is required. "
        "Rerun only if you want agent-loop to re-check or automate the final step. "
        "If checks fail, inspect/fix the failure or rerun so the loop can drive a fix."
    )


def _pr_check_details(pr_checks: PullRequestChecks) -> list[str]:
    details: list[str] = []
    if pr_checks.required_checks:
        details.append(f"Required checks: {', '.join(pr_checks.required_checks)}")
    if pr_checks.failing:
        details.append(
            "Failing checks: "
            + ", ".join(
                f"{check.name} ({check.status.lower()})"
                + (f" — {check.url.strip()}" if check.url and check.url.strip() else "")
                for check in pr_checks.failing
            )
        )
    if pr_checks.pending:
        details.append(
            "Pending checks: "
            + ", ".join(f"{check.name} ({check.status})" for check in pr_checks.pending)
        )
    if pr_checks.missing_required:
        details.append(
            "Required checks not yet reporting: " + ", ".join(pr_checks.missing_required)
        )
    if pr_checks.branch_protection_note:
        details.append(pr_checks.branch_protection_note)
    if not details:
        details.append("No individual check names were available from the GitHub API.")
    return details
