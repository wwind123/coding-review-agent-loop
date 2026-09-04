"""Local test and GitHub PR check gate helpers."""

from __future__ import annotations

from .ci_health import CiInfrastructureStall
from .config import AgentLoopConfig
from .github import PullRequestChecks
from .logging import log
from .errors import AgentLoopError
from .runner import Runner
from .test_runtime import record_test_observation
from .workdirs import active_workdir


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    runner.configure_from_config(config)
    runner.set_containment_role("test-gate")
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    result = runner.run_test_command(
        config.test_command,
        cwd=active_workdir(config),
        timeout_seconds=config.coder_test_command_timeout_seconds,
    )
    _record_gate_observation(config, result)
    _raise_for_gate_result(result, config)
    log(config, "Local test command passed")


def run_pre_review_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.pre_review_tests or not config.test_command:
        return
    runner.configure_from_config(config)
    runner.set_containment_role("test-gate")
    log(config, f"Running pre-review test command: {' '.join(config.test_command)}")
    result = runner.run_test_command(
        config.test_command,
        cwd=active_workdir(config),
        timeout_seconds=config.coder_test_command_timeout_seconds,
    )
    _record_gate_observation(config, result)
    _raise_for_gate_result(result, config)
    log(config, "Pre-review test command passed")


def _record_gate_observation(config: AgentLoopConfig, result) -> None:
    if config.dry_run or not config.agent_memory:
        return
    record_test_observation(
        config.agent_memory_dir,
        argv=result.args,
        cwd=result.cwd,
        outcome=result.outcome,
        elapsed_seconds=result.elapsed_seconds,
        attempted_timeout_seconds=int(config.coder_test_command_timeout_seconds),
        policy_ceiling_seconds=int(config.coder_test_command_timeout_seconds),
        returncode=result.returncode,
        containment=(result.containment.to_dict() if result.containment is not None else None),
    )


def _raise_for_gate_result(result, config: AgentLoopConfig) -> None:
    evidence = getattr(result, "containment", None)
    if evidence is not None and evidence.resource_exhausted and result.outcome != "passed":
        raise AgentLoopError(
            "Local test command failed with resource-exhausted: "
            f"limit={evidence.applicable_limit or 'cgroup resource limit'}; "
            f"diagnostics={'; '.join(evidence.diagnostics) or 'see containment evidence'}\n"
            f"last output:\n{result.output_tail}"
        )
    if evidence is not None and evidence.backend == "systemd-cgroup-v2" and not evidence.cleanup_confirmed:
        raise AgentLoopError(
            "Local test command cleanup-failed: managed scope emptiness was not confirmed; "
            "do not retry until the scope is gone."
        )
    if result.outcome == "timed_out":
        raise AgentLoopError(
            f"Local test command timed out after {int(config.coder_test_command_timeout_seconds)}s: "
            f"{' '.join(result.args)}\nlast output:\n{result.output_tail}"
        )
    if result.outcome == "interrupted":
        raise AgentLoopError(
            f"Local test command was interrupted: {' '.join(result.args)}\n"
            f"last output:\n{result.output_tail}"
        )
    if result.outcome != "passed":
        raise AgentLoopError(
            f"Local test command failed with exit {result.returncode}: {' '.join(result.args)}\n"
            f"last output:\n{result.output_tail}"
        )


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
    if pr_checks.infrastructure_stalls:
        # Not code defects, and not necessarily the sole reason the state is
        # failing/pending: annotate them so a coder round does not chase a
        # check that is stalled on external GitHub Actions infrastructure.
        details.append(
            "External CI infrastructure stalls (not code defects): "
            + "; ".join(stall.describe() for stall in pr_checks.infrastructure_stalls)
        )
    if not details:
        details.append("No individual check names were available from the GitHub API.")
    return details


def _ci_infrastructure_details(stall: CiInfrastructureStall) -> list[str]:
    return [check.describe() for check in stall.checks]


def _format_ci_infrastructure_comment(pr_number: int, stall: CiInfrastructureStall) -> str:
    lines = [
        f"External GitHub Actions infrastructure is blocking PR #{pr_number}.",
        "",
        "This is not a repository defect: no code change is required, and no merge was attempted.",
        "",
    ]
    lines.extend(f"- {detail}" for detail in _ci_infrastructure_details(stall))
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _ci_infrastructure_stop_message(
    pr_number: int,
    stall: CiInfrastructureStall,
    carried_items: list[str],
) -> str:
    lines = [
        f"PR #{pr_number} is blocked on external GitHub Actions infrastructure, not a code defect.",
        "",
        "No code change is required and no merge was attempted. Rerun the same command once "
        "GitHub Actions runners recover; the stalled check(s) should simply be rerun at that point.",
        "",
    ]
    lines.extend(f"- {detail}" for detail in _ci_infrastructure_details(stall))
    if carried_items:
        lines.extend(["", "Reviewer items carried forward, still unresolved:"])
        lines.extend(f"- {item}" for item in carried_items)
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)
