#!/usr/bin/env python3
"""Local Claude/Codex PR review loop orchestrator.

This intentionally shells out to the locally authenticated CLIs instead of
calling model APIs directly. It coordinates the loop and posts each agent's
final response back to GitHub for auditability.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence


STATE_RE = re.compile(r"<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->", re.I)
PR_RE = re.compile(r"<!--\s*AGENT_PR:\s*(\d+)\s*-->", re.I)
GH_PR_URL_RE = re.compile(r"/pull/(\d+)(?:\b|$)")
CLARIFY_RE = re.compile(r"<!--\s*AGENT_CLARIFY\s*-->", re.I)
AgentName = Literal["claude", "codex"]
AGENT_DISPLAY_NAMES: dict[AgentName, str] = {
    "claude": "Claude",
    "codex": "Codex",
}
AGENT_SIGNATURES: dict[AgentName, str] = {
    "claude": "Anthropic Claude",
    "codex": "OpenAI Codex",
}


class AgentLoopError(RuntimeError):
    """Raised for expected orchestration failures."""


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    cwd: Path
    stdout: str
    stderr: str
    returncode: int


class Runner:
    def __init__(self, *, dry_run: bool = False):
        self.dry_run = dry_run

    def run(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        input_text: str | None = None,
        check: bool = True,
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            if input_text:
                print(input_text)
            return CommandResult(cmd, cwd, "", "", 0)

        proc = subprocess.run(
            cmd,
            cwd=cwd,
            input=input_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        result = CommandResult(cmd, cwd, proc.stdout, proc.stderr, proc.returncode)
        if check and proc.returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {proc.returncode}: {' '.join(cmd)}\n"
                f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
            )
        return result

    def run_with_log(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        label: str,
        progress_interval_seconds: int,
        check: bool = True,
    ) -> CommandResult:
        cmd = [str(a) for a in args]
        if self.dry_run:
            print(f"[dry-run] ({cwd}) {' '.join(cmd)}")
            return CommandResult(cmd, cwd, "", "", 0)

        log_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_log_dir_ignored(log_path.parent)
        started = time.monotonic()
        next_progress = started + progress_interval_seconds
        header = f"$ {' '.join(cmd)}\n\n"
        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write(header)
            log_file.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                while True:
                    returncode = proc.poll()
                    if returncode is not None:
                        break
                    now = time.monotonic()
                    if now >= next_progress:
                        elapsed = int(now - started)
                        print(
                            f"[agent-loop {datetime.now().strftime('%H:%M:%S')}] "
                            f"{label} still running ({elapsed}s); log: {log_path}",
                            file=sys.stderr,
                            flush=True,
                        )
                        next_progress = now + progress_interval_seconds
                    time.sleep(1)
            except KeyboardInterrupt:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                raise

        full_output = log_path.read_text(encoding="utf-8")
        output = full_output[len(header):] if full_output.startswith(header) else full_output
        result = CommandResult(cmd, cwd, output, "", returncode)
        if check and returncode != 0:
            raise AgentLoopError(
                f"Command failed with exit {returncode}: {' '.join(cmd)}\n"
                f"log: {log_path}\n\nlast output:\n{tail_text(full_output)}"
            )
        return result


@dataclass(frozen=True)
class AgentLoopConfig:
    repo: str
    claude_dir: Path
    codex_dir: Path
    coder: AgentName
    reviewer: AgentName
    base: str
    max_rounds: int
    auto_merge: bool
    dry_run: bool
    allow_shared_dir: bool
    claude_cmd: str
    codex_cmd: str
    gh_cmd: str
    claude_args: tuple[str, ...]
    codex_args: tuple[str, ...]
    test_command: tuple[str, ...] | None
    ci_check_name: str
    ci_timeout_seconds: int
    ci_poll_interval_seconds: int
    quiet: bool
    log_dir: Path
    progress_interval_seconds: int


def log(config: AgentLoopConfig, message: str) -> None:
    if config.quiet:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[agent-loop {now}] {message}", file=sys.stderr, flush=True)


def tail_text(text: str, *, max_lines: int = 80) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def ensure_log_dir_ignored(log_dir: Path) -> None:
    gitignore = log_dir / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n!.gitignore\n", encoding="utf-8")


def agent_log_path(config: AgentLoopConfig, agent: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.log_dir / f"{stamp}-{agent}.log"


def parse_agent_state(text: str) -> str:
    matches = STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError("Agent response did not include <!-- AGENT_STATE: approved|blocking -->")
    # Use the final marker as authoritative; responses may quote earlier review markers.
    return matches[-1].lower()


def parse_pr_number(text: str) -> int | None:
    marker = PR_RE.search(text)
    if marker:
        return int(marker.group(1))
    url = GH_PR_URL_RE.search(text)
    if url:
        return int(url.group(1))
    return None


def is_clarification_request(text: str) -> bool:
    return bool(CLARIFY_RE.search(text))


def ensure_distinct_workdirs(config: AgentLoopConfig) -> None:
    if config.allow_shared_dir:
        return
    if config.claude_dir.resolve() == config.codex_dir.resolve():
        raise AgentLoopError(
            "--claude-dir and --codex-dir point to the same directory. "
            "Use separate clones/worktrees, or pass --allow-shared-dir explicitly."
        )


def ensure_workdir(path: Path, option_name: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise AgentLoopError(f"{option_name} exists but is not a directory: {path}")
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentLoopError(f"Could not create {option_name} at {path}: {exc}") from exc


def ensure_agent_workdirs(config: AgentLoopConfig) -> None:
    required: set[AgentName] = {config.coder, config.reviewer}
    if "claude" in required:
        ensure_workdir(config.claude_dir, "--claude-dir")
    if "codex" in required:
        ensure_workdir(config.codex_dir, "--codex-dir")
    ensure_distinct_workdirs(config)


def detect_repo(runner: Runner, cwd: Path, gh_cmd: str) -> str:
    result = runner.run(
        [gh_cmd, "repo", "view", "--json", "nameWithOwner", "--jq", ".nameWithOwner"],
        cwd=cwd,
    )
    repo = result.stdout.strip()
    if not repo:
        raise AgentLoopError("Unable to detect GitHub repo. Pass --repo owner/name.")
    return repo


def validate_open_pr(runner: Runner, *, config: AgentLoopConfig, pr_number: int) -> None:
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


def validate_open_issue(runner: Runner, *, config: AgentLoopConfig, issue_number: int) -> None:
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
    config: AgentLoopConfig,
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


def _parse_claude_output(raw: str) -> tuple[str, str | None]:
    """Extract (text, session_id) from Claude's --output-format json response.

    Falls back to (raw, None) if the output is not JSON, so plain-text responses
    (e.g. from FakeRunner in tests, or older Claude builds) are handled gracefully.
    """
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            text = data.get("result", raw)
            if not isinstance(text, str):
                text = raw
            return text, data.get("session_id")
    except (json.JSONDecodeError, ValueError):
        pass
    return raw, None


def run_claude(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    prompt: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    """Run Claude and return (response_text, session_id).

    When session_id is provided the conversation is resumed via --resume so
    Claude retains context from previous rounds, avoiding redundant file reads.
    """
    args = [config.claude_cmd, "--print", "--output-format", "json", *config.claude_args]
    if session_id:
        args += ["--resume", session_id]
    args.append(prompt)
    log_path = agent_log_path(config, "claude")
    log(config, f"Starting Claude in {config.claude_dir}; log: {log_path}")
    result = runner.run_with_log(
        args,
        cwd=config.claude_dir,
        log_path=log_path,
        label="Claude",
        progress_interval_seconds=config.progress_interval_seconds,
    )
    log(config, f"Claude finished; log: {log_path}")
    return _parse_claude_output(result.stdout)


def run_codex(runner: Runner, *, config: AgentLoopConfig, prompt: str) -> str:
    log_path = agent_log_path(config, "codex")
    log(config, f"Starting Codex in {config.codex_dir}; log: {log_path}")
    if config.dry_run:
        result = runner.run(
            [
                config.codex_cmd,
                "exec",
                "--cd",
                str(config.codex_dir),
                *config.codex_args,
                prompt,
            ],
            cwd=config.codex_dir,
        )
        log(config, f"Codex finished; log: {log_path}")
        return result.stdout

    with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as handle:
        output_path = handle.name
    try:
        runner.run_with_log(
            [
                config.codex_cmd,
                "exec",
                "--cd",
                str(config.codex_dir),
                "--output-last-message",
                output_path,
                *config.codex_args,
                prompt,
            ],
            cwd=config.codex_dir,
            log_path=log_path,
            label="Codex",
            progress_interval_seconds=config.progress_interval_seconds,
        )
        output = Path(output_path).read_text(encoding="utf-8")
        log(config, f"Codex finished; log: {log_path}")
        return output
    finally:
        try:
            os.unlink(output_path)
        except FileNotFoundError:
            pass


def run_agent(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    if agent == "claude":
        return run_claude(runner, config=config, prompt=prompt, session_id=session_id)
    if agent == "codex":
        return run_codex(runner, config=config, prompt=prompt), None
    raise AgentLoopError(f"Unsupported agent: {agent}")


def agent_display_name(agent: AgentName) -> str:
    return AGENT_DISPLAY_NAMES[agent]


def agent_signature(agent: AgentName) -> str:
    return AGENT_SIGNATURES[agent]


def build_issue_prompt(issue_number: int, config: AgentLoopConfig) -> str:
    reviewer_name = agent_display_name(config.reviewer)
    coder_signature = agent_signature(config.coder)
    return f"""Fix GitHub issue #{issue_number} in {config.repo}.

Use this local checkout as your workspace. Create a branch, implement the fix,
run relevant tests, commit, push, and open a pull request against {config.base}.

Do not wait for {reviewer_name} yourself; this local orchestrator will run {reviewer_name} after
you create the PR. In your final response, include the PR number using exactly
this marker:

<!-- AGENT_PR: <number> -->

Also include exactly one state marker:

<!-- AGENT_STATE: blocking -->

Use blocking here to hand the PR to {reviewer_name} for review. Sign the response as:
-- {coder_signature}
"""


def build_task_prompt(task_text: str, config: AgentLoopConfig) -> str:
    reviewer_name = agent_display_name(config.reviewer)
    coder_signature = agent_signature(config.coder)
    return f"""You have been given a free-form task to implement in {config.repo}.

Task:
{task_text}

Use this local checkout as your workspace. Decide between two paths:

(a) If the task is clear enough to implement, create a branch, implement the
    change, run relevant tests, commit, push, and open a pull request against
    {config.base}. Do not wait for {reviewer_name}; this local orchestrator
    will run {reviewer_name} after you create the PR. End your final response
    with both markers:

    <!-- AGENT_PR: <number> -->
    <!-- AGENT_STATE: blocking -->

(b) If the task is genuinely ambiguous or missing information that would change
    the implementation, do NOT write code. Instead, ask focused clarifying
    questions and end your final response with exactly this marker:

    <!-- AGENT_CLARIFY -->

Prefer (a) when reasonable assumptions can be documented in the PR description;
choose (b) only for material ambiguity. Sign your response as:
-- {coder_signature}
"""


def build_task_clarification_prompt(
    task_text: str,
    history: Sequence[tuple[str, str]],
    config: AgentLoopConfig,
) -> str:
    coder_signature = agent_signature(config.coder)
    qa_blocks = "\n\n".join(
        f"Round {idx + 1} questions from you:\n{questions}\n\n"
        f"Round {idx + 1} answers from the user:\n{answers}"
        for idx, (questions, answers) in enumerate(history)
    )
    return f"""Continuing the previous free-form task in {config.repo}.

Original task:
{task_text}

Clarification so far:

{qa_blocks}

Now proceed. Strongly prefer to implement the task and open a PR. Only ask
again if a critical detail is still missing. Use the same response markers as
before:

- For implementation: include both <!-- AGENT_PR: <number> --> and
  <!-- AGENT_STATE: blocking --> at the end of your final response.
- For another clarification round: end your final response with exactly
  <!-- AGENT_CLARIFY -->.

Sign your response as:
-- {coder_signature}
"""


def build_review_prompt(pr_number: int, round_number: int, config: AgentLoopConfig) -> str:
    coder_name = agent_display_name(config.coder)
    reviewer_signature = agent_signature(config.reviewer)
    return f"""Review pull request #{pr_number} in {config.repo} (round {round_number}).

Focus on correctness, security, test coverage, and maintainability. Review the
full diff and any existing PR discussion. Do not make code changes in this
review step; report blocking findings if {coder_name} needs to fix anything.

End your final response with exactly one marker:

<!-- AGENT_STATE: approved -->

or:

<!-- AGENT_STATE: blocking -->

Use approved only if there are no blocking issues. Always sign your response:
-- {reviewer_signature}
"""


def build_followup_prompt(
    pr_number: int,
    round_number: int,
    review: str,
    config: AgentLoopConfig,
) -> str:
    reviewer_name = agent_display_name(config.reviewer)
    coder_signature = agent_signature(config.coder)
    return f"""{reviewer_name} reviewed pull request #{pr_number} in {config.repo} and found blocking issues.

Address the review below in this local checkout. Pull/sync the PR branch if
needed, implement fixes, run relevant tests, commit, and push to the same PR.
Do not create a new PR.

{reviewer_name} review:

{review}

This is round {round_number}. End your final response with exactly one marker:

<!-- AGENT_STATE: blocking -->

Use blocking to hand the updated PR back to {reviewer_name}. If you cannot safely address
the review, explain why and still use the blocking marker so a human can
intervene. Sign the response as:
-- {coder_signature}
"""


def run_optional_tests(runner: Runner, config: AgentLoopConfig) -> None:
    if not config.test_command:
        return
    log(config, f"Running local test command: {' '.join(config.test_command)}")
    runner.run(config.test_command, cwd=config.codex_dir)
    log(config, "Local test command passed")


def get_pr_head_sha(runner: Runner, config: AgentLoopConfig, pr_number: int) -> str:
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


def get_check_status(runner: Runner, config: AgentLoopConfig, head_sha: str) -> str:
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


def wait_for_ci(runner: Runner, config: AgentLoopConfig, pr_number: int) -> None:
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


def merge_pr(runner: Runner, config: AgentLoopConfig, pr_number: int) -> None:
    log(config, f"Merging PR #{pr_number}")
    runner.run(
        [config.gh_cmd, "pr", "merge", str(pr_number), "--repo", config.repo, "--merge"],
        cwd=config.codex_dir,
    )


def run_issue_loop(runner: Runner, *, issue_number: int, config: AgentLoopConfig) -> int:
    ensure_agent_workdirs(config)
    log(config, f"Validating issue #{issue_number}")
    validate_open_issue(runner, config=config, issue_number=issue_number)

    coder_output, coder_session_id = run_agent(
        runner,
        agent=config.coder,
        config=config,
        prompt=build_issue_prompt(issue_number, config),
    )
    pr_number = parse_pr_number(coder_output)
    if pr_number is None:
        raise AgentLoopError(
            f"{agent_display_name(config.coder)} output did not include a PR marker or PR URL."
        )
    log(config, f"{agent_display_name(config.coder)} reported PR #{pr_number}; validating it is open")
    validate_open_pr(runner, config=config, pr_number=pr_number)

    post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
    return run_pr_loop(
        runner,
        pr_number=pr_number,
        config=config,
        coder_session_id=coder_session_id,
    )


def _read_clarification_from_stdin() -> str:
    print(
        "\nProvide clarification (one entry per line; finish with a single '.' line or Ctrl+D):",
        file=sys.stderr,
        flush=True,
    )
    lines: list[str] = []
    try:
        while True:
            line = input()
            if line.strip() == ".":
                break
            lines.append(line)
    except EOFError:
        pass
    return "\n".join(lines)


def run_task_loop(
    runner: Runner,
    *,
    task_text: str,
    config: AgentLoopConfig,
    interactive: bool = False,
    max_clarification_rounds: int = 3,
    clarification_input=None,
) -> int:
    ensure_agent_workdirs(config)
    if not task_text.strip():
        raise AgentLoopError("Task text is empty; provide a non-empty description.")
    if max_clarification_rounds < 0:
        raise AgentLoopError("--max-clarification-rounds must be zero or positive.")

    history: list[tuple[str, str]] = []
    prompt = build_task_prompt(task_text, config)
    read_clarification = clarification_input or _read_clarification_from_stdin
    coder_name = agent_display_name(config.coder)
    session_id: str | None = None

    for attempt in range(max_clarification_rounds + 1):
        log(config, f"Task attempt {attempt + 1}: invoking {coder_name}")
        coder_output, session_id = run_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=prompt,
            session_id=session_id,
        )
        if not coder_output.strip():
            raise AgentLoopError(f"{coder_name} produced an empty response.")

        pr_number = parse_pr_number(coder_output)
        if pr_number is not None:
            log(config, f"{coder_name} reported PR #{pr_number}; validating it is open")
            validate_open_pr(runner, config=config, pr_number=pr_number)
            post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
            return run_pr_loop(
                runner,
                pr_number=pr_number,
                config=config,
                coder_session_id=session_id,
            )

        if not is_clarification_request(coder_output):
            raise AgentLoopError(
                f"{coder_name} output did not include a PR marker, PR URL, "
                "or clarification marker."
            )

        if not interactive:
            raise AgentLoopError(
                f"{coder_name} requested clarification but the loop is non-interactive. "
                "Add the missing details to the task text or rerun with --interactive.\n\n"
                f"{coder_name}'s questions:\n{coder_output}"
            )

        if attempt >= max_clarification_rounds:
            raise AgentLoopError(
                f"{coder_name} still requested clarification after "
                f"{max_clarification_rounds} rounds; "
                "human intervention required."
            )

        log(config, f"{coder_name} requested clarification (round {attempt + 1}); awaiting user input")
        print(coder_output, flush=True)
        answers = read_clarification()
        if not answers.strip():
            raise AgentLoopError("Empty clarification reply; aborting task.")
        history.append((coder_output, answers))
        prompt = build_task_clarification_prompt(task_text, history, config)

    raise AgentLoopError("run_task_loop exited unexpectedly without producing a PR.")


def run_pr_loop(
    runner: Runner,
    *,
    pr_number: int,
    config: AgentLoopConfig,
    coder_session_id: str | None = None,
    reviewer_session_id: str | None = None,
) -> int:
    ensure_agent_workdirs(config)
    log(config, f"Validating PR #{pr_number}")
    validate_open_pr(runner, config=config, pr_number=pr_number)

    for round_number in range(1, config.max_rounds + 1):
        reviewer_name = agent_display_name(config.reviewer)
        coder_name = agent_display_name(config.coder)
        log(config, f"Round {round_number}: {reviewer_name} reviewing PR #{pr_number}")
        review_output, reviewer_session_id = run_agent(
            runner,
            agent=config.reviewer,
            config=config,
            prompt=build_review_prompt(pr_number, round_number, config),
            session_id=reviewer_session_id,
        )
        if not review_output.strip():
            raise AgentLoopError(f"{reviewer_name} produced an empty response.")

        post_pr_comment(runner, config=config, pr_number=pr_number, body=review_output)
        review_state = parse_agent_state(review_output)
        log(config, f"Round {round_number}: {reviewer_name} state is {review_state}")
        if review_state == "approved":
            run_optional_tests(runner, config)
            if config.auto_merge:
                wait_for_ci(runner, config, pr_number)
                merge_pr(runner, config, pr_number)
            print(f"PR #{pr_number} approved by {reviewer_name}.")
            return 0
        if round_number == config.max_rounds:
            raise AgentLoopError(
                f"{reviewer_name} still reported blocking issues after round {round_number}; "
                "human review required."
            )

        log(config, f"Round {round_number}: {coder_name} addressing {reviewer_name} review")
        coder_output, coder_session_id = run_agent(
            runner,
            agent=config.coder,
            config=config,
            prompt=build_followup_prompt(pr_number, round_number, review_output, config),
            session_id=coder_session_id,
        )
        if not coder_output.strip():
            raise AgentLoopError(f"{coder_name} produced an empty response.")

        post_pr_comment(runner, config=config, pr_number=pr_number, body=coder_output)
        # Validate marker presence; reviewer remains the merge gate on the next round.
        parse_agent_state(coder_output)
        log(config, f"Round {round_number}: {coder_name} pushed updates for re-review")

    raise AgentLoopError(
        f"Reached max rounds ({config.max_rounds}) for PR #{pr_number}; human review required."
    )


def _split_command(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(shlex.split(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a local coder -> reviewer PR review loop."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--repo", help="GitHub repo as owner/name. Defaults to gh repo view.")
        subparser.add_argument("--base", default="main", help="PR base branch for new issue work.")
        subparser.add_argument("--claude-dir", type=Path, default=Path.cwd())
        subparser.add_argument("--codex-dir", type=Path, default=Path.cwd())
        subparser.add_argument(
            "--coder",
            choices=("claude", "codex"),
            default="claude",
            help="Agent that creates and fixes the PR (default: claude).",
        )
        subparser.add_argument(
            "--reviewer",
            choices=("claude", "codex"),
            default="codex",
            help="Agent that reviews the PR and gates approval (default: codex).",
        )
        subparser.add_argument("--allow-shared-dir", action="store_true")
        subparser.add_argument("--max-rounds", type=int, default=5)
        subparser.add_argument("--auto-merge", action="store_true")
        subparser.add_argument("--dry-run", action="store_true")
        subparser.add_argument("--claude-cmd", default="claude")
        subparser.add_argument("--codex-cmd", default="codex")
        subparser.add_argument("--gh-cmd", default="gh")
        subparser.add_argument(
            "--dangerous-agent-permissions",
            action="store_true",
            help=(
                "Use permission-bypass defaults for both agents. Only use in trusted "
                "local repositories: Claude gets --dangerously-skip-permissions and "
                "Codex gets --dangerously-bypass-approvals-and-sandbox."
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
            "--test-command",
            help="Optional command to run after the reviewer approves, before auto-merge.",
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
            help="Directory for Claude/Codex subprocess logs (default: .agent-loop-logs).",
        )
        subparser.add_argument(
            "--progress-interval-seconds",
            type=int,
            default=30,
            help="How often to print long-running agent heartbeats (default: 30).",
        )

    issue = subparsers.add_parser("issue", help="Ask the coder to fix an issue, then review it.")
    issue.add_argument("issue_number", type=int)
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


def config_from_args(args: argparse.Namespace, runner: Runner) -> AgentLoopConfig:
    codex_dir = args.codex_dir.resolve()
    repo = args.repo or detect_repo(runner, codex_dir, args.gh_cmd)
    test_command = _split_command(args.test_command)
    if args.ci_timeout_seconds <= 0:
        raise AgentLoopError("--ci-timeout-seconds must be greater than zero.")
    if args.ci_poll_interval_seconds <= 0:
        raise AgentLoopError("--ci-poll-interval-seconds must be greater than zero.")
    if args.progress_interval_seconds <= 0:
        raise AgentLoopError("--progress-interval-seconds must be greater than zero.")
    if args.coder == args.reviewer:
        raise AgentLoopError("--coder and --reviewer must be different agents.")
    return AgentLoopConfig(
        repo=repo,
        claude_dir=args.claude_dir.resolve(),
        codex_dir=codex_dir,
        coder=args.coder,
        reviewer=args.reviewer,
        base=args.base,
        max_rounds=args.max_rounds,
        auto_merge=args.auto_merge,
        dry_run=args.dry_run,
        allow_shared_dir=args.allow_shared_dir,
        claude_cmd=args.claude_cmd,
        codex_cmd=args.codex_cmd,
        gh_cmd=args.gh_cmd,
        claude_args=tuple(args.claude_arg if args.claude_arg is not None else (
            ["--dangerously-skip-permissions"] if args.dangerous_agent_permissions else []
        )),
        codex_args=tuple(
            args.codex_arg
            if args.codex_arg is not None
            else (
                ["--dangerously-bypass-approvals-and-sandbox"]
                if args.dangerous_agent_permissions
                else []
            )
        ),
        test_command=test_command,
        ci_check_name=args.ci_check_name,
        ci_timeout_seconds=args.ci_timeout_seconds,
        ci_poll_interval_seconds=args.ci_poll_interval_seconds,
        quiet=args.quiet,
        log_dir=(codex_dir / args.log_dir if not args.log_dir.is_absolute() else args.log_dir),
        progress_interval_seconds=args.progress_interval_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runner = Runner(dry_run=args.dry_run)
    try:
        config = config_from_args(args, runner)
        if args.command == "issue":
            return run_issue_loop(runner, issue_number=args.issue_number, config=config)
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
    except AgentLoopError as exc:
        print(f"agent-loop: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
