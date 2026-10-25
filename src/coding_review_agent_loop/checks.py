"""Local test and GitHub PR check gate helpers."""

from __future__ import annotations

from .checkout_verification import gate_window
from collections.abc import Sequence

from .ci_health import CiInfrastructureStall, PullRequestCheck, partition_exclusion_notes
from .config import AgentLoopConfig
from .github import PullRequestChecks
from .logging import log
from .managed_ci import FINAL_CONTEXT
from .errors import AgentLoopError
from .runner import Runner
from .test_runtime import launch_integrity_state, launch_state_fields, record_launcher_health, record_test_observation
from .workdirs import active_workdir


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    runner.configure_from_config(config)
    runner.set_containment_role("test-gate")
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    workdir = active_workdir(config)
    with gate_window(config, runner, path=workdir):
        result = runner.run_test_command(
            config.test_command,
            cwd=workdir,
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
    workdir = active_workdir(config)
    with gate_window(config, runner, path=workdir):
        result = runner.run_test_command(
            config.test_command,
            cwd=workdir,
            timeout_seconds=config.coder_test_command_timeout_seconds,
        )
    _record_gate_observation(config, result)
    _raise_for_gate_result(result, config)
    log(config, "Pre-review test command passed")


def _record_gate_observation(config: AgentLoopConfig, result) -> None:
    if config.dry_run or not config.agent_memory:
        return
    if getattr(result, "overlap_rejected", False) or getattr(result, "outcome", "") == "overlap-rejected":
        return
    if getattr(result, "inner_exec", "not-attempted") == "failed":
        record_launcher_health(
            config.agent_memory_dir,
            cwd=result.cwd,
            candidate=result.args,
            state="failed",
            provenance=getattr(result, "health_provenance", "parent-runner"),
            repository=config.repo,
            diagnostic=getattr(result, "diagnostic", "") or getattr(result, "output_tail", ""),
        )
        return
    if getattr(result, "suite_start", "unknown") == "verified":
        record_launcher_health(
            config.agent_memory_dir,
            cwd=result.cwd,
            candidate=result.args,
            state="verified",
            provenance=getattr(result, "health_provenance", "parent-runner"),
            repository=config.repo,
            diagnostic=getattr(result, "diagnostic", ""),
        )
    if getattr(result, "suite_start", "unknown") == "not-started":
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
        # A gate run whose suite start was not verified (e.g. a shell
        # wrapper) is recorded only as non-evidence (#989).
        launch_integrity=launch_integrity_state(result, wrapper_boundary=False),
        launch_state=launch_state_fields(result),
    )


def _dependency_remedy_suffix(config: AgentLoopConfig, cwd) -> str:
    """Advisory remedy for an unlinked per-run worktree (#1651); text only."""
    from .worktree_hints import dependency_remedy

    remedy = dependency_remedy(config, cwd)
    return f"\n{remedy}" if remedy else ""


def _raise_for_gate_result(result, config: AgentLoopConfig) -> None:
    if getattr(result, "overlap_rejected", False):
        raise AgentLoopError(
            "Local test command was not started because an identical managed command "
            "is already running in this invocation lane; wait for it to exit before retrying."
        )
    if getattr(result, "outcome", "") == "launch-failed" or getattr(result, "inner_exec", "started") == "failed":
        detail = getattr(result, "diagnostic", "") or getattr(result, "output_tail", "") or "no launcher diagnostic"
        raise AgentLoopError(
            "Local test command could not start; this is a launcher/bootstrap failure, "
            f"not a test-suite result: {' '.join(result.args)}\n{detail}\n"
            "Repair the configured executable or interpreter and retry the configured command."
            + _dependency_remedy_suffix(config, getattr(result, "cwd", None))
        )
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


def _pending_ci_stop_message(
    pr_number: int, state: str, details: list[str], *, watch_hint: bool = False
) -> str:
    headline = {
        "pending": f"Reviewers approved PR #{pr_number}, but GitHub checks are still pending.",
        "unavailable": (
            f"Reviewers approved PR #{pr_number}, but GitHub check status is unavailable."
        ),
    }[state]
    lines = [
        headline,
        "",
        _pending_ci_stop_guidance(state, watch_hint=watch_hint),
        "",
    ]
    lines.extend(f"- {detail}" for detail in details)
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _unreadable_protection_stop_message(pr_number: int, details: list[str]) -> str:
    lines = [
        f"Reviewers approved PR #{pr_number}, and every observed GitHub check passed, "
        "but merge readiness cannot be confirmed.",
        "",
        "The current GitHub token cannot read branch protection (HTTP 403), so the "
        "required checks are unknown, and GitHub did not report a `CLEAN` merge state "
        "for the current head. An unmet protection rule, such as a required review or "
        "a required check that has not reported, may remain. Satisfy it, or grant the "
        "token administration read access, then merge manually or rerun agent-loop.",
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


WATCH_PENDING_CI_HINT = (
    "Rerun with --watch-pending-ci to watch CI (bounded by --ci-timeout-seconds) and "
    "re-run plumbing-only failures automatically."
)


def _pending_ci_stop_guidance(state: str, *, watch_hint: bool = False) -> str:
    wait = {
        "pending": "Wait for CI.",
        "unavailable": "Wait for GitHub check status to become available.",
    }[state]
    return (
        "This run cannot confirm the PR is merge-ready yet. "
        f"{wait} If checks pass, you can merge manually; no rerun is required. "
        "Rerun only if you want agent-loop to re-check or automate the final step. "
        "If checks fail, inspect/fix the failure or rerun so the loop can drive a fix."
        + (f" {WATCH_PENDING_CI_HINT}" if watch_hint else "")
    )


def final_context_state_line(
    checks: PullRequestChecks | None,
    *,
    stuck: Sequence[PullRequestCheck] = (),
    managed_status: str | None = None,
    expected: bool = False,
) -> str | None:
    """Render the observed final-ci/exact-head state for a terminal report.

    ``None`` when the context is neither on the board, required, nor expected
    (the managed route always expects it).
    """
    if managed_status is not None:
        return f"{FINAL_CONTEXT}: {managed_status}"
    if any(check.name == FINAL_CONTEXT for check in stuck):
        return f"{FINAL_CONTEXT}: stuck-pending"
    if checks is not None:
        for bucket, label in ((checks.failing, "failure"), (checks.pending, "pending"), (checks.passing, "success")):
            if any(check.name == FINAL_CONTEXT for check in bucket):
                return f"{FINAL_CONTEXT}: {label}"
        if FINAL_CONTEXT in checks.required_checks or FINAL_CONTEXT in checks.missing_required:
            return f"{FINAL_CONTEXT}: absent"
    return f"{FINAL_CONTEXT}: absent" if expected else None


CI_PLUMBING_STOP_HEADLINE = (
    "PR #{pr_number} CI failed only in plumbing jobs, not in tests; stopping on infrastructure."
)


def _ci_plumbing_stop_message(
    pr_number: int,
    *,
    reason: str,
    run_urls: Sequence[str],
    plumbing: Sequence[str],
    final_line: str | None,
    recovery: str | None = None,
    watch_hint: bool = False,
) -> str:
    lines = [
        CI_PLUMBING_STOP_HEADLINE.format(pr_number=pr_number),
        "",
        f"{reason} No coder round was started, no code change is required by this "
        "failure, and no merge was attempted.",
        "",
    ]
    lines.extend(f"- Run: {url}" for url in run_urls)
    if not run_urls:
        lines.append("- Run: unavailable")
    lines.extend(f"- Plumbing: {item}" for item in plumbing)
    if final_line:
        lines.append(f"- {final_line}")
    if recovery:
        lines.extend(["", recovery])
    if watch_hint:
        lines.extend(["", WATCH_PENDING_CI_HINT])
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


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
    exclusions, diagnostics = partition_exclusion_notes(pr_checks)
    if exclusions:
        details.append(
            "Excluded ad hoc workflow_dispatch checks (operator diagnostics, not PR checks): "
            + "; ".join(exclusions)
        )
    details.extend(diagnostics)
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


# Render-side headline templates, shared with history_export's recognizer.
CI_INFRASTRUCTURE_COMMENT_HEADLINE = (
    "External GitHub Actions infrastructure is blocking PR #{pr_number}."
)
CI_INFRASTRUCTURE_STOP_HEADLINE = (
    "PR #{pr_number} is blocked on external GitHub Actions infrastructure, not a code defect."
)


def _ci_infrastructure_details(stall: CiInfrastructureStall) -> list[str]:
    return [check.describe() for check in stall.checks]


def _format_ci_infrastructure_comment(
    pr_number: int, stall: CiInfrastructureStall, *, final_line: str | None = None
) -> str:
    lines = [
        CI_INFRASTRUCTURE_COMMENT_HEADLINE.format(pr_number=pr_number),
        "",
        "This is not a repository defect: no code change is required, and no merge was attempted.",
        "",
    ]
    lines.extend(f"- {detail}" for detail in _ci_infrastructure_details(stall))
    if final_line:
        lines.append(f"- {final_line}")
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _ci_infrastructure_stop_message(
    pr_number: int,
    stall: CiInfrastructureStall,
    carried_items: list[str],
    *,
    final_line: str | None = None,
) -> str:
    lines = [
        CI_INFRASTRUCTURE_STOP_HEADLINE.format(pr_number=pr_number),
        "",
        "No code change is required and no merge was attempted. Rerun the same command once "
        "GitHub Actions runners recover; the stalled check(s) should simply be rerun at that point.",
        "",
    ]
    lines.extend(f"- {detail}" for detail in _ci_infrastructure_details(stall))
    if final_line:
        lines.append(f"- {final_line}")
    if carried_items:
        lines.extend(["", "Reviewer items carried forward, still unresolved:"])
        lines.extend(f"- {item}" for item in carried_items)
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)
