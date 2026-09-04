"""
Run an external agent (Codex or Gemini) for one reviewer or coder turn.

In --dry-run mode, writes a canned approved stub to --output and exits 0:
  --role reviewer → plan_review stub
  --role coder    → plan_state stub, or issue_implementation for --flow pr
In live mode, invokes the agent CLI and writes the response to --output.

Usage:
  python -m helpers.run_external \\
    --agent codex|gemini \\
    --prompt-file PATH \\
    --output PATH \\
    --workdir PATH \\
    [--role {reviewer,coder}] \\
    [--cmd PATH] \\
    [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coding_review_agent_loop.runner import Runner
from coding_review_agent_loop.test_runtime import DEFAULT_TEST_TIMEOUT_SECONDS

_CANNED_PLAN_REVIEW = json.dumps(
    {
        "schema_version": 1,
        "kind": "plan_review",
        "state": "approved",
        "summary": "Dry-run stub: plan looks good.",
        "blocking_plan_issues": [],
        "same_plan_followups": [],
        "future_followups": [],
        "prior_plan_item_dispositions": [],
    },
    indent=2,
)

_CANNED_PLAN_REVIEW_FOOTER = (
    "\n<!-- AGENT_PLAN_STATE: approved -->\n-- Codex (dry-run stub)\n"
)

_CANNED_PR_REVIEW = json.dumps(
    {
        "schema_version": 1,
        "kind": "pr_review",
        "state": "approved",
        "summary": "Dry-run stub: PR looks good.",
        "prior_item_dispositions": [],
    },
    indent=2,
)

_CANNED_PR_REVIEW_FOOTER = (
    "\n<!-- AGENT_STATE: approved -->\n-- Codex (dry-run stub)\n"
)

_CANNED_PLAN_STATE = """\
## Plan (dry-run stub)

1. Implement the requested changes.

<!-- AGENT_PLAN_STATE: approved -->
-- Codex (dry-run stub)
"""

# Coder implementation turn (reversed roles, #316). The dry-run follows the
# same typed result contract as a live issue implementation.
_CANNED_IMPLEMENTATION = json.dumps(
    {
        "schema_version": 1,
        "kind": "issue_implementation",
        "state": "blocking",
        "summary": "Dry-run stub: created a branch, made the changes, and opened a pull request.",
        "pr_number": 1,
        "human_requirements": {
            "addressed_ids": [],
            "checked_discussion_directly": False,
        },
        "human_requirement_dispositions": [],
        "tests_run": None,
    },
    indent=2,
) + "\n<!-- AGENT_STATE: blocking -->\n-- Codex (dry-run stub)\n"

_CANNED_PLAN_DECOMPOSITION = json.dumps(
    {
        "schema_version": 1,
        "kind": "plan_decomposition",
        "phases": [
            {
                "title": "Dry-run implementation phase",
                "scope": "Dry-run stub: implement the approved plan in one child issue.",
                "non_goals": "No real GitHub issue is created during dry-run.",
                "dependency_notes": "First phase; no dependencies.",
                "rollout_risk": "low - dry-run preview only.",
                "validation": "Run the relevant project tests before opening a PR.",
                "parent_context": "Dry-run stub derived from the approved parent plan.",
                "automation": "agent-pr",
                "depends_on": [],
            }
        ],
    },
    indent=2,
)


def _build_dry_run_response(role: str, flow: str = "plan") -> str:
    if role == "coder":
        if flow == "decompose":
            return _CANNED_PLAN_DECOMPOSITION
        return _CANNED_IMPLEMENTATION if flow == "pr" else _CANNED_PLAN_STATE
    if flow == "pr":
        return _CANNED_PR_REVIEW + _CANNED_PR_REVIEW_FOOTER
    return _CANNED_PLAN_REVIEW + _CANNED_PLAN_REVIEW_FOOTER


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one external agent turn.")
    from coding_review_agent_loop.agents.base import normalize_agent_name

    parser.add_argument(
        "--agent",
        required=True,
        type=normalize_agent_name,
        choices=["codex", "gemini", "antigravity"],
    )
    parser.add_argument("--prompt-file", required=True, help="Path to prompt text file.")
    antigravity_models = parser.add_mutually_exclusive_group()
    antigravity_models.add_argument(
        "--model", default=None,
        help="Legacy single-model override (antigravity only; as shown by `agy models`).",
    )
    antigravity_models.add_argument(
        "--antigravity-models",
        nargs="+",
        default=None,
        help="Ordered Antigravity model fallback chain.",
    )
    parser.add_argument(
        "--antigravity-quota-signatures",
        nargs="+",
        default=None,
        help="Output signatures that trigger fallback to the next Antigravity model.",
    )
    parser.add_argument(
        "--antigravity-print-timeout-seconds",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Maximum wait for each agy --print invocation (default: 600, "
             "matching the agent-loop CLI). Overrides agy's five-minute "
             "print-mode default.",
    )
    parser.add_argument("--output", required=True, help="Path to write the agent response.")
    parser.add_argument("--workdir", required=True, help="Working directory for the agent.")
    parser.add_argument(
        "--role",
        default="reviewer",
        choices=["reviewer", "coder"],
        help="Turn role: 'reviewer' (default) or 'coder' (Codex writes the plan).",
    )
    parser.add_argument("--cmd", default=None, help="Agent CLI command (overrides default).")
    parser.add_argument("--diff-file", default=None, help="Path to a pre-fetched PR diff to embed in the prompt.")
    parser.add_argument(
        "--flow",
        default="plan",
        choices=["plan", "pr", "decompose"],
        help="Review flow: 'plan' (default), 'pr', or 'decompose'. Affects dry-run stub kind.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write a canned stub and exit.")
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub OWNER/REPO to clone if workdir is absent or stale. "
             "When provided, workdir is validated and re-cloned automatically.",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="PR number. When provided with --flow pr, syncs the workdir to the PR head "
             "instead of leaving it on the base branch after cleaning.",
    )
    parser.add_argument(
        "--pr-head-sha",
        default=None,
        help="Expected PR head SHA for verification (optional). Used with --pr.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries for transient agent failures (429/overloaded/timeout). "
             "Total attempts = max-retries + 1 (default: 2, matching the CLI). "
             "Use 0 to disable retries.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=int,
        nargs="+",
        default=[15, 45],
        help="Backoff delays before each retry; the final value is reused when "
             "retries exceed the list length (default: 15 45).",
    )
    parser.add_argument(
        "--usage-output",
        default=None,
        help="Write the agent's token usage (JSON) to this path for external-agent "
             "cost tracking (#308). Skipped in --dry-run.",
    )
    parser.add_argument(
        "--response-evidence-output",
        default=None,
        help="Write response_file_text/message_text evidence needed for deterministic "
        "structured-response recovery. Skipped in --dry-run.",
    )
    parser.add_argument(
        "--invocation-evidence-output",
        default=None,
        help=(
            "Write the mechanical containment invocation evidence sidecar, including "
            "limits, counters, termination cause, and cleanup status."
        ),
    )
    parser.add_argument(
        "--coder-test-command-timeout-seconds",
        type=float,
        default=DEFAULT_TEST_TIMEOUT_SECONDS,
        metavar="SECONDS",
        help=(
            "Finite run-level ceiling inherited by coder test wrappers "
            f"(default: {DEFAULT_TEST_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--containment-mode",
        choices=["auto", "required", "off"],
        default="auto",
        help="Process-tree containment mode for this external invocation (default: auto).",
    )
    parser.add_argument("--containment-memory-high", default=None)
    parser.add_argument("--containment-memory-max", default=None)
    parser.add_argument("--containment-memory-swap-max", default=None)
    parser.add_argument("--containment-tasks-max", default=None)
    parser.add_argument("--containment-os-headroom-percent", type=float, default=25.0)
    parser.add_argument("--containment-slice", default="agent-loop.slice")
    parser.add_argument("--containment-cache-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if any(delay <= 0 for delay in args.retry_backoff_seconds):
        parser.error("--retry-backoff-seconds values must be > 0")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        flow = args.flow
        if args.role == "coder":
            if flow == "decompose":
                stub_kind = "plan_decomposition"
            else:
                stub_kind = "implementation" if flow == "pr" else "plan_state"
        elif flow == "pr":
            stub_kind = "pr_review"
        else:
            stub_kind = "plan_review"
        output_path.write_text(_build_dry_run_response(args.role, flow), encoding="utf-8")
        print(f"dry-run: wrote canned {stub_kind} stub to {output_path}")
        return

    try:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")
    except OSError as exc:
        print(f"run_external: cannot read prompt file: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.diff_file:
        try:
            diff_text = Path(args.diff_file).read_text(encoding="utf-8")
            injection = f"\n\n## PR diff\n\n```diff\n{diff_text}\n```\n"
            insert_at = prompt.find("Suggested commands:")
            if insert_at >= 0:
                prompt = prompt[:insert_at] + injection + prompt[insert_at:]
            else:
                prompt = prompt + injection
        except OSError as exc:
            print(f"run_external: cannot read diff file: {exc}", file=sys.stderr)
            sys.exit(1)

    workdir = Path(args.workdir)

    # Import backends lazily to avoid heavy import in dry-run path
    from coding_review_agent_loop.agents.antigravity import AntigravityAttemptState, AntigravityBackend
    from coding_review_agent_loop.agents.codex import CodexBackend
    from coding_review_agent_loop.agents.gemini import GeminiBackend
    from coding_review_agent_loop.config import (
        DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS,
        DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES,
        AgentLoopConfig,
        AgentName,
        ensure_temp_checkout,
        sync_checkout_to_pr,
    )
    from coding_review_agent_loop.github import PullRequestMetadata
    from coding_review_agent_loop.transient import classify_antigravity_capacity, is_transient_agent_output
    from coding_review_agent_loop.usage import estimate_usage

    agent_name: AgentName = args.agent
    default_cmds = {"codex": "codex", "gemini": "gemini", "antigravity": "agy"}
    cmd = args.cmd or default_cmds[agent_name]
    antigravity_models = (
        (args.model,)
        if args.model is not None
        else tuple(args.antigravity_models or ())
    )
    antigravity_quota_signatures = (
        tuple(args.antigravity_quota_signatures)
        if args.antigravity_quota_signatures is not None
        else DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES
    )
    antigravity_print_timeout_seconds = (
        args.antigravity_print_timeout_seconds
        if args.antigravity_print_timeout_seconds is not None
        else DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS
    )

    # Build a minimal config sufficient for backend.run()
    import tempfile
    log_dir = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "skill-logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    config = AgentLoopConfig(
        repo=args.repo or "skill/run",
        claude_dir=workdir,
        codex_dir=workdir,
        gemini_dir=workdir,
        antigravity_dir=workdir,
        coder="claude",
        reviewer=(agent_name,),
        base="main",
        max_rounds=1,
        auto_merge=False,
        dry_run=False,
        allow_shared_dir=True,
        claude_cmd="claude",
        codex_cmd=cmd if agent_name == "codex" else "codex",
        gemini_cmd=cmd if agent_name == "gemini" else "gemini",
        antigravity_cmd=cmd if agent_name == "antigravity" else "agy",
        antigravity_args=("--dangerously-skip-permissions",),
        antigravity_model=None,
        antigravity_models=antigravity_models,
        antigravity_print_timeout_seconds=antigravity_print_timeout_seconds,
        antigravity_quota_signatures=antigravity_quota_signatures,
        gh_cmd="gh",
        claude_args=(),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
        gemini_args=("--skip-trust",),
        test_command=None,
        pre_review_tests=False,
        ci_timeout_seconds=300,
        ci_poll_interval_seconds=30,
        quiet=False,
        log_dir=log_dir,
        subprocess_log_dir=log_dir,
        progress_interval_seconds=30,
        agent_max_retries=0,
        agent_retry_backoff_seconds=(30,),
        agent_memory=False,
        refresh_agent_memory=False,
        agent_memory_dir=log_dir,
        refresh_test_profile=False,
        auto_agent_dirs=(agent_name,),
        coder_test_command_timeout_seconds=args.coder_test_command_timeout_seconds,
        containment_mode=args.containment_mode,
        containment_memory_high=args.containment_memory_high,
        containment_memory_max=args.containment_memory_max,
        containment_memory_swap_max=args.containment_memory_swap_max,
        containment_tasks_max=args.containment_tasks_max,
        containment_os_headroom_percent=args.containment_os_headroom_percent,
        containment_slice=args.containment_slice,
        containment_cache_dir=args.containment_cache_dir,
    )

    runner = Runner(dry_run=False)
    runner.configure_from_config(config)
    runner.set_containment_role("coder" if args.role == "coder" else "reviewer")
    # Re-clone (if requested) happens once, outside the retry loop: it raises
    # deterministically and re-cloning per attempt would be wasteful.
    if args.repo:
        ensure_temp_checkout(workdir, agent=agent_name, config=config, runner=runner)
        if args.flow == "pr" and args.pr is not None:
            # After ensure_temp_checkout leaves the workdir on the base branch,
            # sync to the PR head so the reviewer sees the PR branch, not main.
            sync_checkout_to_pr(
                config,
                runner,
                path=workdir,
                label=f"Default {agent_name} workdir",
                default_owned=True,
                pr_number=args.pr,
                pr_metadata=PullRequestMetadata(
                    number=args.pr,
                    repo=args.repo,
                    title=None,
                    head_branch=None,
                    base_branch=None,
                    head_sha=args.pr_head_sha,
                    url=None,
                ),
            )
    if agent_name == "codex":
        backend = CodexBackend()
    elif agent_name == "antigravity":
        backend = AntigravityBackend()
    else:
        backend = GeminiBackend()

    # Retry transient agent failures. Antigravity shares its retry allowance
    # across the ordered model chain and only capacity diagnostics may advance it.
    # The primary failure path is a *returned* AgentResult with a non-zero
    # returncode whose raw_output (merged stdout+stderr) carries the signal;
    # a raised exception is the secondary path (its message embeds the captured
    # output tail for AgentLoopError from run_with_log).
    antigravity_attempts = (
        AntigravityAttemptState.from_config(config, args.max_retries)
        if agent_name == "antigravity"
        else None
    )
    max_attempts = (
        len(config.antigravity_models) + args.max_retries
        if antigravity_attempts is not None
        else args.max_retries + 1
    )
    backoff = args.retry_backoff_seconds
    result = None
    for attempt in range(1, max_attempts + 1):
        candidate = None
        mechanical_failure = False
        target_exec_retryable = False
        try:
            candidate = backend.run(
                runner,
                antigravity_attempts.singleton_config(config)
                if antigravity_attempts is not None else config,
                prompt,
            )
        except Exception as exc:  # noqa: BLE001
            failure_text = str(exc)
        else:
            if candidate.returncode != 0 or not candidate.text.strip():
                failure_text = candidate.raw_output or candidate.text
            else:
                result = candidate
                break

        if candidate is not None and candidate.containment is not None:
            evidence = candidate.containment
            if evidence.resource_exhausted:
                failure_text = (
                    "resource-exhausted: "
                    f"limit={evidence.applicable_limit or 'cgroup resource limit'}; "
                    f"backend={evidence.backend}; diagnostics="
                    + ("; ".join(evidence.diagnostics) or "see invocation evidence")
                )
                mechanical_failure = True
            elif evidence.termination_cause == "target-exec-error":
                decision = getattr(runner, "target_exec_retry_decision", None)
                if decision is not None:
                    target_exec_retryable, failure_text = decision(
                        candidate.command_result.args[0]
                        if candidate.command_result is not None
                        else "",
                        evidence.target_exec_errno,
                    )
                else:
                    failure_text = "agent target could not be executed"
            elif evidence.backend == "systemd-cgroup-v2" and not evidence.cleanup_confirmed:
                failure_text = (
                    "containment-indeterminate: managed invocation cleanup could not be confirmed; "
                    "retry is blocked until the scope is empty"
                )
                mechanical_failure = True

        transient = (
            False
            if mechanical_failure
            else target_exec_retryable or is_transient_agent_output(failure_text or "")
        )
        capacity = classify_antigravity_capacity(
            failure_text or "",
            returncode=(candidate.returncode if candidate is not None else 1),
            empty_response=bool(candidate is not None and candidate.returncode == 0 and not candidate.text.strip()),
            signatures=config.antigravity_quota_signatures,
        ) if agent_name == "antigravity" else None
        if capacity is not None and capacity.is_capacity and not mechanical_failure:
            transient = True
        transition = (
            antigravity_attempts.next_after_failure(
                retryable=transient, provider_capacity=bool(capacity and capacity.is_capacity)
            ) if antigravity_attempts is not None else ("retry" if transient and attempt < max_attempts else "stop")
        )
        if transition == "retry":
            delay = backoff[min(attempt - 1, len(backoff) - 1)] if backoff else 1
            print(
                f"run_external: transient failure (attempt {attempt}/{max_attempts}), "
                f"retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
            continue
        if transition == "fallback":
            print("run_external: provider capacity; trying next Antigravity model", file=sys.stderr)
            continue
        reason = "retries exhausted" if transient else "non-transient failure"
        print(
            f"run_external: agent invocation failed ({reason}): {failure_text}",
            file=sys.stderr,
        )
        # Write the failure text to --output (not just stderr) so a caller can read
        # and classify it — e.g. the skill's reviewer-unavailable path (#322) — and
        # decide whether to skip the reviewer rather than only seeing a non-zero exit.
        try:
            output_path.write_text(failure_text or "", encoding="utf-8")
        except OSError:
            pass
        if args.invocation_evidence_output:
            try:
                Path(args.invocation_evidence_output).write_text(
                    json.dumps(
                        candidate.containment.to_dict()
                        if candidate is not None and candidate.containment is not None
                        else {
                            "backend": "unknown",
                            "status": "not-collected",
                            "failure": failure_text or "",
                        },
                        indent=2,
                        sort_keys=True,
                    ) + "\n",
                    encoding="utf-8",
                )
            except Exception as exc:  # noqa: BLE001
                print(f"run_external: could not write invocation evidence: {exc}", file=sys.stderr)
        sys.exit(1)

    assert result is not None  # loop either set result or exited
    output_path.write_text(result.text, encoding="utf-8")
    print(f"agent result written to {output_path}")

    if args.response_evidence_output:
        try:
            Path(args.response_evidence_output).write_text(
                json.dumps(
                    {
                        "response_file_text": result.response_file_text,
                        "message_text": result.message_text,
                        **(
                            {"containment": result.containment.to_dict()}
                            if result.containment is not None
                            else {}
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"run_external: could not write response evidence: {exc}", file=sys.stderr)

    if args.invocation_evidence_output:
        try:
            Path(args.invocation_evidence_output).write_text(
                json.dumps(
                    result.containment.to_dict() if result.containment else {
                        "backend": "unknown",
                        "status": "not-collected",
                        "reason": "backend returned no containment evidence",
                    },
                    indent=2,
                    sort_keys=True,
                ) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"run_external: could not write invocation evidence: {exc}", file=sys.stderr)

    # External-agent usage for cost tracking (#308). Advisory — never fail the run.
    if args.usage_output:
        try:
            usage = result.usage or estimate_usage(prompt, result.text)
            Path(args.usage_output).write_text(
                json.dumps({
                    "agent": agent_name,
                    "session_id": result.session_id,
                    "returncode": result.returncode,
                    "usage": usage.to_dict(),
                    # Model that actually ran, for the dynamic signature (#332).
                    "model_used": result.model_used,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"run_external: could not write usage output: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
