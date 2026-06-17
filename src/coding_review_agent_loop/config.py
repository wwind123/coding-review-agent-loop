"""Configuration construction and validation."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import default_agent_args
from .errors import AgentLoopError
from .github import PullRequestMetadata, detect_repo
from .logging import log
from .runner import Runner
from .workdirs import active_workdir, agent_workdir


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
    pre_review_tests: bool
    ci_check_name: str
    ci_timeout_seconds: int
    ci_poll_interval_seconds: int
    quiet: bool
    log_dir: Path
    progress_interval_seconds: int
    agent_max_retries: int
    agent_retry_backoff_seconds: tuple[int, ...]
    agent_memory: bool
    refresh_agent_memory: bool
    agent_memory_dir: Path
    refresh_test_profile: bool
    approved_followups: str = "ignore"
    plan_execution_mode: str = "plan-only"
    planning_context_mode: str = "compact"
    pr_review_context_mode: str = "full"
    auto_agent_dirs: tuple[AgentName, ...] = ()
    # Antigravity (`agy`) backend (#215). Defaulted so existing AgentLoopConfig
    # constructions keep working; real values are set when antigravity is used.
    antigravity_dir: Path = Path("antigravity")
    antigravity_cmd: str = "agy"
    antigravity_args: tuple[str, ...] = ()
    antigravity_model: str = "Gemini 3.1 Pro (High)"
    # Declared model / reasoning effort for the dynamic signature (#332). Empty
    # means "not declared" (the agent runs its own default and the signature
    # falls back to the generic provider name). antigravity always has a model.
    codex_model: str = ""
    codex_reasoning_effort: str = ""
    gemini_model: str = ""
    claude_model: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.reviewer, str):
            object.__setattr__(self, "reviewer", (self.reviewer,))
        ensure_no_model_arg_conflicts(self)
        if self.planning_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--planning-context-mode must be either 'full' or 'compact'.")
        if self.pr_review_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--pr-review-context-mode must be either 'full' or 'compact'.")


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
        "antigravity": (config.antigravity_dir, "--antigravity-dir"),
    }
    active = [(agent, *paths[agent]) for agent in required]
    for index, (_left_agent, left_path, left_option) in enumerate(active):
        for _right_agent, right_path, right_option in active[index + 1 :]:
            if left_path.resolve() == right_path.resolve():
                raise AgentLoopError(
                    f"{left_option} and {right_option} point to the same directory. "
                    "Use separate clones/worktrees, or pass --allow-shared-dir explicitly."
                )


def _args_have_model_flag(args: tuple[str, ...]) -> bool:
    return any(a == "--model" or a.startswith("--model=") for a in args)


def _args_have_reasoning_effort(args: tuple[str, ...]) -> bool:
    return any("model_reasoning_effort" in a for a in args)


def ensure_no_model_arg_conflicts(config: AgentLoopConfig) -> None:
    """Reject declaring a model/effort both via the dedicated flag and a freeform arg.

    The dynamic signature (#332) must match the model that actually ran. Passing a
    model/effort both as a declared value and as a freeform `--*-arg` would emit
    duplicate CLI flags (tool/version-dependent precedence) and risk a signature
    that disagrees with the run, so we fail fast with a clear message. Note
    ``antigravity_model`` is always declared, so any ``--antigravity-arg --model``
    is always a conflict.
    """
    if _args_have_model_flag(config.antigravity_args):
        raise AgentLoopError(
            "--antigravity-arg --model conflicts with the always-declared antigravity "
            "model; set the model via --antigravity-model only."
        )
    if config.codex_model and _args_have_model_flag(config.codex_args):
        raise AgentLoopError(
            "--codex-arg --model conflicts with --codex-model; use --codex-model only."
        )
    if config.codex_reasoning_effort and _args_have_reasoning_effort(config.codex_args):
        raise AgentLoopError(
            "--codex-arg model_reasoning_effort conflicts with --codex-reasoning-effort; "
            "use --codex-reasoning-effort only."
        )
    if config.codex_reasoning_effort and not config.codex_model:
        # The declared flags exist to stamp the signature, which needs the model
        # name; codex does not report its model, so effort alone can't be labeled.
        raise AgentLoopError(
            "--codex-reasoning-effort requires --codex-model so the signature can "
            "name the model (codex does not report its model). To set effort without "
            "stamping the signature, use --codex-arg -c model_reasoning_effort=... instead."
        )
    if config.gemini_model and _args_have_model_flag(config.gemini_args):
        raise AgentLoopError(
            "--gemini-arg --model conflicts with --gemini-model; use --gemini-model only."
        )
    if config.claude_model and _args_have_model_flag(config.claude_args):
        raise AgentLoopError(
            "--claude-arg --model conflicts with --claude-model; use --claude-model only."
        )


def default_agent_workdir(repo: str, agent: AgentName) -> Path:
    repo_slug = repo_cache_slug(repo)
    return Path(tempfile.gettempdir()) / "coding-review-agent-loop" / repo_slug / agent / "repo"


def repo_cache_slug(repo: str) -> str:
    parts = repo.split("/")
    if len(parts) != 2 or not all(parts):
        raise AgentLoopError("--repo must use the OWNER/REPO format.")
    owner, name = parts
    return f"{owner}-{name}"


def default_cache_root() -> Path:
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "coding-review-agent-loop"
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "coding-review-agent-loop" / "Cache"
        return Path.home() / "AppData" / "Local" / "coding-review-agent-loop" / "Cache"
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        return Path(xdg_cache_home) / "coding-review-agent-loop"
    return Path.home() / ".cache" / "coding-review-agent-loop"


def default_agent_memory_dir(repo: str) -> Path:
    return default_cache_root() / "repos" / repo_cache_slug(repo) / "memory"


def ensure_workdir(path: Path, option_name: str) -> None:
    if path.exists():
        if not path.is_dir():
            raise AgentLoopError(f"{option_name} exists but is not a directory: {path}")
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentLoopError(f"Could not create {option_name} at {path}: {exc}") from exc


_TOOL_ARTIFACT_NAMES = frozenset({
    ".agent-loop-logs",
    ".agent-loop",
    ".agent-loop-responses",
})


def _is_stale_default_workdir(path: Path) -> bool:
    """Return True if path has no real checkout and contains only tool-owned artifacts.

    Covers two cases:
    - No .git at all, only log/artifact dirs (original case).
    - .git present but working tree is empty — happens when a prior cleanup removed
      source files but left the .git dir (e.g. response storage at .git/agent-loop/).
    """
    if not path.is_dir():
        return False
    if (path / ".git").exists():
        non_git = {item.name for item in path.iterdir() if item.name != ".git"}
        return not non_git or non_git.issubset(_TOOL_ARTIFACT_NAMES)
    contents = {item.name for item in path.iterdir()}
    return contents.issubset(_TOOL_ARTIFACT_NAMES)  # empty dir is also stale


def _looks_like_repo_remote(remote_url: str, repo: str) -> bool:
    normalized = remote_url.strip().removesuffix(".git").lower()
    repo = repo.lower()
    return normalized.endswith(f"/{repo}") or normalized.endswith(f":{repo}")


def _run_git(runner: Runner, path: Path, args: tuple[str, ...], *, check: bool = True):
    return runner.run(("git", *args), cwd=path, check=check)


def _agent_dir_option(agent: AgentName) -> str:
    return {
        "claude": "--claude-dir",
        "codex": "--codex-dir",
        "gemini": "--gemini-dir",
        "antigravity": "--antigravity-dir",
    }[agent]


def _validate_repo_remote(
    path: Path,
    *,
    label: str,
    config: AgentLoopConfig,
    runner: Runner,
) -> None:
    remote = _run_git(runner, path, ("remote", "get-url", "origin")).stdout.strip()
    if not _looks_like_repo_remote(remote, config.repo):
        raise AgentLoopError(f"{label} at {path} uses origin {remote!r}, not {config.repo!r}.")


def _clean_or_reject_checkout(
    path: Path,
    *,
    label: str,
    default_owned: bool,
    config: AgentLoopConfig,
    runner: Runner,
) -> None:
    status = _run_git(runner, path, ("status", "--porcelain")).stdout.strip()
    if not status:
        return
    if not default_owned:
        raise AgentLoopError(f"{label} is dirty: {path}. Commit, stash, or clean it before rerunning.")
    log(config, f"Cleaning dirty {label[0].lower()}{label[1:]}: {path}")
    _run_git(runner, path, ("reset", "--hard"))
    _run_git(runner, path, ("clean", "-fd"))


def _sync_base_branch(
    path: Path,
    *,
    label: str,
    default_owned: bool,
    config: AgentLoopConfig,
    runner: Runner,
) -> None:
    if not path.is_dir():
        raise AgentLoopError(f"{label} does not exist or is not a directory: {path}")

    git_check = _run_git(runner, path, ("rev-parse", "--is-inside-work-tree"), check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        if not default_owned:
            raise AgentLoopError(
                f"{label} is not a git checkout: {path}. "
                "Use a clean checkout for issue or task implementation."
            )
        raise AgentLoopError(
            f"{label} exists but is not a git checkout: {path}. "
            "Remove it or pass an explicit agent directory."
        )

    _validate_repo_remote(path, label=label, config=config, runner=runner)
    _clean_or_reject_checkout(
        path,
        label=label,
        default_owned=default_owned,
        config=config,
        runner=runner,
    )

    _run_git(runner, path, ("fetch", "origin"))
    switch = _run_git(runner, path, ("switch", config.base), check=False)
    if switch.returncode != 0:
        if not default_owned:
            raise AgentLoopError(
                f"{label} could not switch to base branch {config.base!r}. "
                "Create the branch locally or use a clean checkout on the base branch."
            )
        _run_git(runner, path, ("switch", "-C", config.base, f"origin/{config.base}"))
    _run_git(runner, path, ("pull", "--ff-only", "origin", config.base))


def ensure_temp_checkout(path: Path, *, agent: AgentName, config: AgentLoopConfig, runner: Runner) -> None:
    if _is_stale_default_workdir(path):
        log(config, f"Stale default {agent} workdir detected (no checkout, only logs remain); recreating: {path}")
        shutil.rmtree(path)

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

    _sync_base_branch(
        path,
        label=f"Default {agent} workdir",
        default_owned=True,
        config=config,
        runner=runner,
    )


def validate_explicit_workdir(path: Path, option_name: str, config: AgentLoopConfig, runner: Runner) -> None:
    git_check = _run_git(runner, path, ("rev-parse", "--is-inside-work-tree"), check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        return

    _clean_or_reject_checkout(
        path,
        label=option_name,
        default_owned=False,
        config=config,
        runner=runner,
    )
    _validate_repo_remote(path, label=option_name, config=config, runner=runner)


def sync_coder_base_before_implementation(config: AgentLoopConfig, runner: Runner) -> None:
    """Sync the active coder checkout to the configured base branch just before implementation."""
    path = active_workdir(config)
    option_name = _agent_dir_option(config.coder)
    default_owned = config.coder in set(config.auto_agent_dirs)
    label = f"Default {config.coder} workdir" if default_owned else option_name
    _sync_base_branch(
        path,
        label=label,
        default_owned=default_owned,
        config=config,
        runner=runner,
    )


def sync_reviewer_pr_before_review(
    config: AgentLoopConfig,
    runner: Runner,
    reviewer: AgentName,
    pr_number: int,
    pr_metadata: PullRequestMetadata,
) -> None:
    """Refresh a reviewer checkout to the current PR head before invoking the reviewer."""
    path = agent_workdir(config, reviewer)
    option_name = _agent_dir_option(reviewer)
    default_owned = reviewer in set(config.auto_agent_dirs)
    label = f"Default {reviewer} workdir" if default_owned else option_name
    sync_checkout_to_pr(
        config,
        runner,
        path=path,
        label=label,
        default_owned=default_owned,
        pr_number=pr_number,
        pr_metadata=pr_metadata,
    )


def sync_coder_pr_before_validation(
    config: AgentLoopConfig,
    runner: Runner,
    pr_number: int,
    pr_metadata: PullRequestMetadata,
) -> None:
    """Refresh the active coder checkout to the current PR head before local validation."""
    path = active_workdir(config)
    option_name = _agent_dir_option(config.coder)
    default_owned = config.coder in set(config.auto_agent_dirs)
    label = f"Default {config.coder} workdir" if default_owned else option_name
    sync_checkout_to_pr(
        config,
        runner,
        path=path,
        label=label,
        default_owned=default_owned,
        pr_number=pr_number,
        pr_metadata=pr_metadata,
    )


def sync_checkout_to_pr(
    config: AgentLoopConfig,
    runner: Runner,
    *,
    path: Path,
    label: str,
    default_owned: bool,
    pr_number: int,
    pr_metadata: PullRequestMetadata,
) -> None:
    """Refresh a checkout to the current PR head."""
    pr_ref = f"refs/remotes/origin/pr/{pr_number}"
    pr_fetch_refspec = f"+pull/{pr_number}/head:{pr_ref}"

    if runner.dry_run:
        _run_git(runner, path, ("fetch", "origin"))
        _run_git(runner, path, ("fetch", "origin", pr_fetch_refspec))
        _run_git(runner, path, ("checkout", "--detach", pr_ref))
        _run_git(runner, path, ("rev-parse", "HEAD"))
        _run_git(runner, path, ("status", "--short", "--branch"))
        log(config, f"{label} would refresh PR #{pr_number} before review")
        return

    if not path.is_dir():
        raise AgentLoopError(f"{label} does not exist or is not a directory: {path}")

    git_check = _run_git(runner, path, ("rev-parse", "--is-inside-work-tree"), check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        if not default_owned:
            raise AgentLoopError(
                f"{label} is not a git checkout: {path}. "
                "Use a clean checkout for PR review."
            )
        raise AgentLoopError(
            f"{label} exists but is not a git checkout: {path}. "
            "Remove it or pass an explicit agent directory."
        )

    _validate_repo_remote(path, label=label, config=config, runner=runner)
    _clean_or_reject_checkout(
        path,
        label=label,
        default_owned=default_owned,
        config=config,
        runner=runner,
    )

    _run_git(runner, path, ("fetch", "origin"))
    _run_git(runner, path, ("fetch", "origin", pr_fetch_refspec))
    _run_git(runner, path, ("checkout", "--detach", pr_ref))
    local_head = _run_git(runner, path, ("rev-parse", "HEAD")).stdout.strip()
    branch_state = _run_git(runner, path, ("status", "--short", "--branch")).stdout.strip()
    advertised_head = (pr_metadata.head_sha or "").strip()
    if advertised_head and local_head != advertised_head:
        raise AgentLoopError(
            f"{label} at {path} is at {local_head or '(unknown)'}, "
            f"but PR #{pr_number} metadata advertises head SHA {advertised_head}."
        )
    if advertised_head:
        log(config, f"{label} refreshed for PR #{pr_number}: HEAD {local_head} (expected {advertised_head})")
    else:
        log(config, f"{label} refreshed for PR #{pr_number}: HEAD {local_head}; no PR head SHA available")
    if branch_state:
        log(config, f"{label} branch state before review: {branch_state}")


def ensure_agent_workdirs(config: AgentLoopConfig, runner: Runner) -> None:
    required: set[AgentName] = {config.coder, *reviewers(config)}
    paths = {
        "claude": (config.claude_dir, "--claude-dir"),
        "codex": (config.codex_dir, "--codex-dir"),
        "gemini": (config.gemini_dir, "--gemini-dir"),
        "antigravity": (config.antigravity_dir, "--antigravity-dir"),
    }
    auto_dirs = set(config.auto_agent_dirs)
    for agent in required:
        path, option = paths[agent]
        if agent in auto_dirs:
            log(config, f"Using default {agent} workdir: {path}")
            ensure_temp_checkout(path, agent=agent, config=config, runner=runner)
        else:
            ensure_workdir(path, option)
            validate_explicit_workdir(path, option, config, runner)
    ensure_distinct_workdirs(config)


def _split_command(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    return tuple(shlex.split(value))


def _resolve_agent_memory_dir(value: Path | None, *, repo: str, primary_dir: Path) -> Path:
    if value is None:
        return default_agent_memory_dir(repo).resolve()
    if value.is_absolute():
        return value.resolve()
    return (primary_dir / value).resolve()


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
            ("antigravity", args.antigravity_dir),
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
    antigravity_dir = (
        args.antigravity_dir.resolve()
        if args.antigravity_dir is not None
        else default_agent_workdir(repo, "antigravity").resolve()
    )
    primary_dir = {
        "claude": claude_dir,
        "codex": codex_dir,
        "gemini": gemini_dir,
        "antigravity": antigravity_dir,
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
    if args.agent_max_retries < 0:
        raise AgentLoopError("--agent-max-retries must be zero or positive.")
    if any(delay <= 0 for delay in args.agent_retry_backoff_seconds):
        raise AgentLoopError("--agent-retry-backoff-seconds values must be greater than zero.")
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
        antigravity_dir=antigravity_dir,
        antigravity_cmd=args.antigravity_cmd,
        antigravity_args=tuple(
            args.antigravity_arg
            if args.antigravity_arg is not None
            else default_agent_args("antigravity", dangerous=args.dangerous_agent_permissions)
        ),
        antigravity_model=args.antigravity_model,
        codex_model=getattr(args, "codex_model", ""),
        codex_reasoning_effort=getattr(args, "codex_reasoning_effort", ""),
        gemini_model=getattr(args, "gemini_model", ""),
        claude_model=getattr(args, "claude_model", ""),
        test_command=test_command,
        pre_review_tests=args.pre_review_tests,
        ci_check_name=args.ci_check_name,
        ci_timeout_seconds=args.ci_timeout_seconds,
        ci_poll_interval_seconds=args.ci_poll_interval_seconds,
        quiet=args.quiet,
        log_dir=(primary_dir / args.log_dir if not args.log_dir.is_absolute() else args.log_dir),
        progress_interval_seconds=args.progress_interval_seconds,
        agent_max_retries=args.agent_max_retries,
        agent_retry_backoff_seconds=tuple(args.agent_retry_backoff_seconds),
        agent_memory=args.agent_memory,
        refresh_agent_memory=args.refresh_agent_memory,
        agent_memory_dir=_resolve_agent_memory_dir(
            args.agent_memory_dir,
            repo=repo,
            primary_dir=primary_dir,
        ),
        refresh_test_profile=args.refresh_test_profile,
        approved_followups=args.approved_followups,
        plan_execution_mode=getattr(args, "plan_execution_mode", None) or "plan-only",
        planning_context_mode=getattr(args, "planning_context_mode", None) or "compact",
        pr_review_context_mode=getattr(args, "pr_review_context_mode", None) or "full",
        auto_agent_dirs=auto_agent_dirs,
    )
