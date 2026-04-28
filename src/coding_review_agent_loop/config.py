"""Configuration construction and validation."""

from __future__ import annotations

import argparse
import shlex
from dataclasses import dataclass
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import default_agent_args
from .errors import AgentLoopError
from .github import detect_repo
from .runner import Runner


@dataclass(frozen=True)
class AgentLoopConfig:
    repo: str
    claude_dir: Path
    codex_dir: Path
    gemini_dir: Path
    coder: AgentName
    reviewer: AgentName | tuple[AgentName, ...]
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

    def __post_init__(self) -> None:
        if isinstance(self.reviewer, str):
            object.__setattr__(self, "reviewer", (self.reviewer,))


def reviewers(config: AgentLoopConfig) -> tuple[AgentName, ...]:
    if isinstance(config.reviewer, str):
        return (config.reviewer,)
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
    required: set[AgentName] = {config.coder, *reviewers(config)}
    if "claude" in required:
        ensure_workdir(config.claude_dir, "--claude-dir")
    if "codex" in required:
        ensure_workdir(config.codex_dir, "--codex-dir")
    if "gemini" in required:
        ensure_workdir(config.gemini_dir, "--gemini-dir")
    ensure_distinct_workdirs(config)


def _split_command(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(shlex.split(value))


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
    configured_reviewers = tuple(args.reviewer or ["codex"])
    if len(set(configured_reviewers)) != len(configured_reviewers):
        raise AgentLoopError("--reviewer cannot include the same agent more than once.")
    return AgentLoopConfig(
        repo=repo,
        claude_dir=args.claude_dir.resolve(),
        codex_dir=codex_dir,
        gemini_dir=args.gemini_dir.resolve(),
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
        log_dir=(codex_dir / args.log_dir if not args.log_dir.is_absolute() else args.log_dir),
        progress_interval_seconds=args.progress_interval_seconds,
    )
