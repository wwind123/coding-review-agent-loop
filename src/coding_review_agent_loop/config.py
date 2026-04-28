"""Configuration construction and validation."""

from __future__ import annotations

import argparse
import shlex
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import default_agent_args
from .errors import AgentLoopError
from .github import detect_repo
from .logging import log
from .runner import Runner


@dataclass(frozen=True)
class AgentLoopConfig:
    repo: str
    claude_dir: Path
    codex_dir: Path
    gemini_dir: Path
    coder: AgentName
    reviewer: tuple[AgentName, ...]
    base: str
    max_rounds: int
    auto_merge: bool
    dry_run: bool
    allow_shared_dir: bool
    claude_cmd: str
    codex_cmd: str
    gemini_cmd: str
    gh_cmd: str
    claude_args: tuple[str, ...]
    codex_args: tuple[str, ...]
    gemini_args: tuple[str, ...]
    test_command: tuple[str, ...] | None
    ci_check_name: str
    ci_timeout_seconds: int
    ci_poll_interval_seconds: int
    quiet: bool
    log_dir: Path
    progress_interval_seconds: int
    auto_agent_dirs: tuple[AgentName, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.reviewer, str):
            object.__setattr__(self, "reviewer", (self.reviewer,))


def reviewers(config: AgentLoopConfig) -> tuple[AgentName, ...]:
    return config.reviewer


def ensure_distinct_workdirs(config: AgentLoopConfig) -> None:
    if config.allow_shared_dir:
        return
    required: set[AgentName] = {config.coder, *reviewers(config)}
    paths = {
        "claude": (config.claude_dir, "--claude-dir"),
        "codex": (config.codex_dir, "--codex-dir"),
        "gemini": (config.gemini_dir, "--gemini-dir"),
    }
    active = [(agent, *paths[agent]) for agent in required]
    for index, (_left_agent, left_path, left_option) in enumerate(active):
        for _right_agent, right_path, right_option in active[index + 1 :]:
            if left_path.resolve() == right_path.resolve():
                raise AgentLoopError(
                    f"{left_option} and {right_option} point to the same directory. "
                    "Use separate clones/worktrees, or pass --allow-shared-dir explicitly."
                )


def default_agent_workdir(repo: str, agent: AgentName) -> Path:
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise AgentLoopError("--repo must use the OWNER/REPO format.")
    owner, name = parts
    repo_slug = f"{owner}-{name}"
    return Path(tempfile.gettempdir()) / "coding-review-agent-loop" / repo_slug / agent / "repo"


def ensure_workdir(path: Path, option_name: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise AgentLoopError(f"{option_name} exists but is not a directory: {path}")
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentLoopError(f"Could not create {option_name} at {path}: {exc}") from exc


def _looks_like_repo_remote(remote_url: str, repo: str) -> bool:
    normalized = remote_url.strip().removesuffix(".git").lower()
    repo = repo.lower()
    return normalized.endswith(f"/{repo}") or normalized.endswith(f":{repo}")


def _run_git(runner: Runner, path: Path, args: tuple[str, ...], *, check: bool = True):
    return runner.run(("git", *args), cwd=path, check=check)


def ensure_temp_checkout(path: Path, *, agent: AgentName, config: AgentLoopConfig, runner: Runner) -> None:
    if not path.exists():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AgentLoopError(f"Could not create parent directory for {agent} checkout at {path}: {exc}") from exc
        runner.run((config.gh_cmd, "repo", "clone", config.repo, str(path)), cwd=path.parent)
        if runner.dry_run:
            return
        # Fresh clones still flow through validation and sync below so the
        # same remote, cleanliness, and base-branch checks apply to every run.

    if not path.is_dir():
        raise AgentLoopError(f"Default {agent} workdir exists but is not a directory: {path}")

    git_check = _run_git(runner, path, ("rev-parse", "--is-inside-work-tree"), check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        raise AgentLoopError(
            f"Default {agent} workdir exists but is not a git checkout: {path}. "
            "Remove it or pass an explicit agent directory."
        )

    remote = _run_git(runner, path, ("remote", "get-url", "origin")).stdout.strip()
    if not _looks_like_repo_remote(remote, config.repo):
        raise AgentLoopError(
            f"Default {agent} workdir at {path} uses origin {remote!r}, not {config.repo!r}."
        )

    status = _run_git(runner, path, ("status", "--porcelain")).stdout.strip()
    if status:
        raise AgentLoopError(
            f"Default {agent} workdir is dirty: {path}. "
            "Commit, stash, or clean it before rerunning, or pass an explicit agent directory."
        )

    _run_git(runner, path, ("fetch", "origin"))
    checkout = _run_git(runner, path, ("checkout", config.base), check=False)
    if checkout.returncode != 0:
        _run_git(runner, path, ("checkout", "-B", config.base, f"origin/{config.base}"))
    _run_git(runner, path, ("pull", "--ff-only", "origin", config.base))


def ensure_agent_workdirs(config: AgentLoopConfig, runner: Runner) -> None:
    required: set[AgentName] = {config.coder, *reviewers(config)}
    paths = {
        "claude": (config.claude_dir, "--claude-dir"),
        "codex": (config.codex_dir, "--codex-dir"),
        "gemini": (config.gemini_dir, "--gemini-dir"),
    }
    auto_dirs = set(config.auto_agent_dirs)
    for agent in required:
        path, option = paths[agent]
        if agent in auto_dirs:
            log(config, f"Using default {agent} workdir: {path}")
            ensure_temp_checkout(path, agent=agent, config=config, runner=runner)
        else:
            ensure_workdir(path, option)
    ensure_distinct_workdirs(config)


def _split_command(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(shlex.split(value))


def config_from_args(args: argparse.Namespace, runner: Runner) -> AgentLoopConfig:
    configured_reviewers = tuple(args.reviewer or ["codex"])
    if len(set(configured_reviewers)) != len(configured_reviewers):
        raise AgentLoopError("--reviewer cannot include the same agent more than once.")

    detect_dir = args.codex_dir.resolve() if args.codex_dir is not None else Path.cwd().resolve()
    repo = args.repo or detect_repo(runner, detect_dir, args.gh_cmd)
    auto_agent_dirs = tuple(
        agent
        for agent, value in (
            ("claude", args.claude_dir),
            ("codex", args.codex_dir),
            ("gemini", args.gemini_dir),
        )
        if value is None
    )
    claude_dir = (
        args.claude_dir.resolve()
        if args.claude_dir is not None
        else default_agent_workdir(repo, "claude").resolve()
    )
    codex_dir = (
        args.codex_dir.resolve()
        if args.codex_dir is not None
        else default_agent_workdir(repo, "codex").resolve()
    )
    gemini_dir = (
        args.gemini_dir.resolve()
        if args.gemini_dir is not None
        else default_agent_workdir(repo, "gemini").resolve()
    )
    primary_dir = {
        "claude": claude_dir,
        "codex": codex_dir,
        "gemini": gemini_dir,
    }[args.coder]
    test_command = _split_command(args.test_command)
    if args.max_rounds <= 0:
        raise AgentLoopError("--max-rounds must be greater than zero.")
    if args.ci_timeout_seconds <= 0:
        raise AgentLoopError("--ci-timeout-seconds must be greater than zero.")
    if args.ci_poll_interval_seconds <= 0:
        raise AgentLoopError("--ci-poll-interval-seconds must be greater than zero.")
    if args.progress_interval_seconds <= 0:
        raise AgentLoopError("--progress-interval-seconds must be greater than zero.")
    return AgentLoopConfig(
        repo=repo,
        claude_dir=claude_dir,
        codex_dir=codex_dir,
        gemini_dir=gemini_dir,
        coder=args.coder,
        reviewer=configured_reviewers,
        base=args.base,
        max_rounds=args.max_rounds,
        auto_merge=args.auto_merge,
        dry_run=args.dry_run,
        allow_shared_dir=args.allow_shared_dir,
        claude_cmd=args.claude_cmd,
        codex_cmd=args.codex_cmd,
        gemini_cmd=args.gemini_cmd,
        gh_cmd=args.gh_cmd,
        claude_args=tuple(
            args.claude_arg
            if args.claude_arg is not None
            else default_agent_args("claude", dangerous=args.dangerous_agent_permissions)
        ),
        codex_args=tuple(
            args.codex_arg
            if args.codex_arg is not None
            else default_agent_args("codex", dangerous=args.dangerous_agent_permissions)
        ),
        gemini_args=tuple(
            args.gemini_arg
            if args.gemini_arg is not None
            else default_agent_args("gemini", dangerous=args.dangerous_agent_permissions)
        ),
        test_command=test_command,
        ci_check_name=args.ci_check_name,
        ci_timeout_seconds=args.ci_timeout_seconds,
        ci_poll_interval_seconds=args.ci_poll_interval_seconds,
        quiet=args.quiet,
        log_dir=(primary_dir / args.log_dir if not args.log_dir.is_absolute() else args.log_dir),
        progress_interval_seconds=args.progress_interval_seconds,
        auto_agent_dirs=auto_agent_dirs,
    )
