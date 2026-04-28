"""GitHub CLI operations used by the orchestrator."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .errors import AgentLoopError
from .logging import log
from .runner import Runner


def detect_repo(runner: Runner, cwd: Path, gh_cmd: str) -> str:
    result = runner.run(
        [gh_cmd, "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        cwd=cwd,
    )
    repo = result.stdout.strip()
    if not repo:
        raise AgentLoopError("Unable to detect GitHub repo. Pass --repo owner/name.")
    return repo


def validate_open_pr(runner: Runner, *, config, pr_number: int) -> None:
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "number,state,url",
        ],
        cwd=config.codex_dir,
    )
    data = json.loads(result.stdout or "{}")
    if data.get("state") != "OPEN":
        raise AgentLoopError(
            f"PR #{pr_number} is {data.get('state', 'not open')}; provide an open PR number."
        )


def validate_open_issue(runner: Runner, *, config, issue_number: int) -> None:
    if config.dry_run:
        return
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/issues/{issue_number}",
            "--jq",
            "{number:.number,state:.state,is_pr:has(\"pull_request\"),url:.html_url}",
        ],
        cwd=config.codex_dir,
    )
    data = json.loads(result.stdout or "{}")
    if data.get("is_pr"):
        raise AgentLoopError(
            f"#{issue_number} is a pull request, not an issue. Use `agent_loop.py pr {issue_number}`."
        )
    if data.get("state") != "open":
        raise AgentLoopError(
            f"Issue #{issue_number} is {data.get('state', 'not open')}; provide an open issue number."
        )


def post_pr_comment(
    runner: Runner,
    *,
    config,
    pr_number: int,
    body: str,
) -> None:
    log(config, f"Posting agent output to PR #{pr_number}")
    if config.dry_run:
        runner.run(
            [config.gh_cmd, "pr", "comment", str(pr_number), "--repo", config.repo, "--body", body],
            cwd=config.codex_dir,
        )
        return

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(body)
        path = handle.name
    try:
        runner.run(
            [
                config.gh_cmd,
                "pr",
                "comment",
                str(pr_number),
                "--repo",
                config.repo,
                "--body-file",
                path,
            ],
            cwd=config.codex_dir,
        )
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def get_pr_head_sha(runner: Runner, config, pr_number: int) -> str:
    result = runner.run(
        [
            config.gh_cmd,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            config.repo,
            "--json",
            "headRefOid",
            "--jq",
            ".headRefOid",
        ],
        cwd=config.codex_dir,
    )
    sha = result.stdout.strip()
    if not sha:
        raise AgentLoopError(f"Unable to resolve head SHA for PR #{pr_number}.")
    return sha


def get_check_status(runner: Runner, config, head_sha: str) -> str:
    result = runner.run(
        [
            config.gh_cmd,
            "api",
            f"repos/{config.repo}/commits/{head_sha}/check-runs",
            "--jq",
            (
                f"[.check_runs[] | select(.name == {json.dumps(config.ci_check_name)})] | "
                'if length == 0 then "pending" else .[0].conclusion // .[0].status end'
            ),
        ],
        cwd=config.codex_dir,
    )
    return result.stdout.strip() or "pending"


def wait_for_ci(runner: Runner, config, pr_number: int) -> None:
    log(config, f"Waiting for GitHub check '{config.ci_check_name}' before merge")
    head_sha = get_pr_head_sha(runner, config, pr_number)
    attempts = max(1, config.ci_timeout_seconds // config.ci_poll_interval_seconds)
    terminal_failures = {
        "failure",
        "cancelled",
        "timed_out",
        "action_required",
        "startup_failure",
        "skipped",
    }
    for attempt in range(attempts):
        status = get_check_status(runner, config, head_sha)
        log(config, f"GitHub check '{config.ci_check_name}' status: {status}")
        if status == "success":
            return
        if status in terminal_failures:
            raise AgentLoopError(f"CI check '{config.ci_check_name}' failed with status: {status}")
        if attempt < attempts - 1:
            runner.run(["sleep", str(config.ci_poll_interval_seconds)], cwd=config.codex_dir)
    raise AgentLoopError(
        f"CI check '{config.ci_check_name}' did not pass within {config.ci_timeout_seconds}s"
    )


def merge_pr(runner: Runner, config, pr_number: int) -> None:
    log(config, f"Merging PR #{pr_number}")
    runner.run(
        [config.gh_cmd, "pr", "merge", str(pr_number), "--repo", config.repo, "--merge"],
        cwd=config.codex_dir,
    )
