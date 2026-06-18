"""
Run an external agent (Codex or Gemini) for one reviewer or coder turn.

In --dry-run mode, writes a canned approved stub to --output and exits 0:
  --role reviewer → plan_review stub
  --role coder    → plan_state stub
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

# Coder implementation turn (reversed roles, #316). Reports a PR number and makes
# no out-of-workdir test claims, so the workdir test guard passes trivially.
_CANNED_IMPLEMENTATION = """\
Implemented the approved plan (dry-run stub): created a branch, made the changes,
and opened a pull request.

<!-- AGENT_PR: 0 -->
-- Codex (dry-run stub)
"""

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
    parser.add_argument("--agent", required=True, choices=["codex", "gemini", "antigravity"])
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
    from coding_review_agent_loop.agents.antigravity import AntigravityBackend
    from coding_review_agent_loop.agents.codex import CodexBackend
    from coding_review_agent_loop.agents.gemini import GeminiBackend
    from coding_review_agent_loop.config import (
        DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES,
        AgentLoopConfig,
        AgentName,
        ensure_temp_checkout,
    )
    from coding_review_agent_loop.transient import is_transient_agent_output
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
        antigravity_quota_signatures=antigravity_quota_signatures,
        gh_cmd="gh",
        claude_args=(),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
        gemini_args=("--skip-trust",),
        test_command=None,
        pre_review_tests=False,
        ci_check_name="",
        ci_timeout_seconds=300,
        ci_poll_interval_seconds=30,
        quiet=False,
        log_dir=log_dir,
        progress_interval_seconds=30,
        agent_max_retries=0,
        agent_retry_backoff_seconds=(30,),
        agent_memory=False,
        refresh_agent_memory=False,
        agent_memory_dir=log_dir,
        refresh_test_profile=False,
        auto_agent_dirs=(agent_name,),
    )

    runner = Runner(dry_run=False)
    # Re-clone (if requested) happens once, outside the retry loop: it raises
    # deterministically and re-cloning per attempt would be wasteful.
    if args.repo:
        ensure_temp_checkout(workdir, agent=agent_name, config=config, runner=runner)
    if agent_name == "codex":
        backend = CodexBackend()
    elif agent_name == "antigravity":
        backend = AntigravityBackend()
    else:
        backend = GeminiBackend()

    # Retry transient agent failures (429 / overloaded / timeout / capacity).
    # The primary failure path is a *returned* AgentResult with a non-zero
    # returncode whose raw_output (merged stdout+stderr) carries the signal;
    # a raised exception is the secondary path (its message embeds the captured
    # output tail for AgentLoopError from run_with_log).
    max_attempts = args.max_retries + 1
    backoff = args.retry_backoff_seconds
    result = None
    for attempt in range(1, max_attempts + 1):
        try:
            candidate = backend.run(runner, config, prompt)
        except Exception as exc:  # noqa: BLE001
            failure_text = str(exc)
        else:
            if candidate.returncode != 0 or not candidate.text.strip():
                failure_text = candidate.raw_output or candidate.text
            else:
                result = candidate
                break

        transient = is_transient_agent_output(failure_text or "")
        if attempt < max_attempts and transient:
            delay = backoff[min(attempt - 1, len(backoff) - 1)] if backoff else 1
            print(
                f"run_external: transient failure (attempt {attempt}/{max_attempts}), "
                f"retrying in {delay}s",
                file=sys.stderr,
            )
            time.sleep(delay)
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
        sys.exit(1)

    assert result is not None  # loop either set result or exited
    output_path.write_text(result.text, encoding="utf-8")
    print(f"agent result written to {output_path}")

    if args.response_evidence_output:
        try:
            Path(args.response_evidence_output).write_text(
                json.dumps({
                    "response_file_text": result.response_file_text,
                    "message_text": result.message_text,
                }, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            print(f"run_external: could not write response evidence: {exc}", file=sys.stderr)

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
