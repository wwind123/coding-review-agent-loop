"""Configuration construction and validation."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import default_agent_args
from .errors import AgentLoopError
from .expected_closure import normalize_issue_ids
from .github import PullRequestMetadata, detect_repo, get_repo_default_branch
from .logging import datetime_stamp, log
from .runner import Runner
from .workdirs import active_workdir, agent_workdir

# Single source of truth for the Antigravity quota-exhaustion fallback signatures
# (#348, #350). The dataclass field default, config_from_args, the CLI flag default,
# and helpers/run_external.py all derive from this so the default can't drift.
DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES: tuple[str, ...] = (
    "quota",
    "rate limit",
    "too many requests",
    "resource exhausted",
    "RESOURCE_EXHAUSTED",
    "429",
    "high traffic",
    "try again in a minute",
    "overload",
    "no capacity",
    "temporarily at capacity",
)

# Default Antigravity model fallback chain, applied in __post_init__ when neither a
# legacy antigravity_model nor an explicit antigravity_models chain is given. Named
# for discoverability/symmetry with DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES.
DEFAULT_ANTIGRAVITY_MODELS: tuple[str, ...] = (
    "Gemini 3.6 Flash (High)",
    "Gemini 3.5 Flash (High)",
    "Gemini 3.1 Pro (High)",
)
DEFAULT_MAX_ROUNDS = 10
# `agy --print` otherwise defaults to five minutes, which is too short for
# complex reviews and causes it to exit with "timeout waiting for response".
DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS = 10 * 60
DEFAULT_REPAIR_MODELS: tuple[str, ...] = ("Gemini 3.5 Flash (Medium)",)
# One parent-wide flat topology budget shared by decomposition and split
# materialization.  Keeping this policy in config prevents each workflow from
# obtaining a second allowance of children.
DEFAULT_FLAT_CHILD_LIMIT: int = 15

# Discuss-mode research policy values (#477). The CLI flag choices, the config
# validation, and the prompt builders all derive from this set.
DISCUSS_RESEARCH_MODES: frozenset[str] = frozenset({"none", "required", "auto"})

# Discuss-mode debater failure policy values (#475).
DISCUSS_DEBATER_FAILURE_MODES: frozenset[str] = frozenset({"fail", "partial"})
DISCUSS_RESULT_MODES: frozenset[str] = frozenset({"triage", "answer"})
BASE_PROVENANCE_VALUES = frozenset({"explicit", "repository-default", "pr-metadata"})


@dataclass(frozen=True)
class AgentLoopConfig:
    repo: str
    claude_dir: Path
    codex_dir: Path
    gemini_dir: Path
    coder: AgentName
    reviewer: tuple[AgentName, ...]
    base: str | None
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
    # Optional plan-first override: use the main coder for planning/revision,
    # then switch only the approved implementation and PR follow-up coder/model.
    implementation_coder: AgentName | None = None
    implementation_coder_model: str = ""
    implementation_codex_reasoning_effort: str = ""
    # Antigravity (`agy`) backend (#215). Defaulted so existing AgentLoopConfig
    # constructions keep working; real values are set when antigravity is used.
    antigravity_dir: Path = Path("antigravity")
    antigravity_cmd: str = "agy"
    antigravity_args: tuple[str, ...] = ()
    antigravity_model: str | None = None
    antigravity_models: tuple[str, ...] = ()
    antigravity_print_timeout_seconds: int = DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS
    antigravity_quota_signatures: tuple[str, ...] = DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES
    # Declared model / reasoning effort for the dynamic signature (#332). Empty
    # means "not declared" (the agent runs its own default and the signature
    # falls back to the generic provider name). antigravity always has a model.
    codex_model: str = ""
    codex_reasoning_effort: str = ""
    gemini_model: str = ""
    claude_model: str = ""
    repair_backend: str = "antigravity"
    repair_models: tuple[str, ...] = DEFAULT_REPAIR_MODELS
    repair_timeout_seconds: int = 120
    # Active subprocess capture is kept outside mutable agent checkouts.  The
    # legacy log_dir remains the home for salvage and usage artifacts.
    subprocess_log_dir: Path | None = None
    # Optional analyzer agent for discuss mode (#467). None keeps plain
    # direct deliberation unchanged. May coincide with a reviewer.
    discuss_analyzer: AgentName | None = None
    # Discuss-mode research policy (#477): "none" forbids online research,
    # "required" enforces sourced external facts from every debater, "auto"
    # lets debaters/analyzer decide using conservative triggers.
    discuss_research: str = "none"
    # Result contract for discuss.  Triage is deliberately the default for
    # backwards compatibility with existing transcripts and callers.
    discuss_result_mode: str = "triage"
    # Parallel debater execution for discuss mode (#475). Opt-in; sequential
    # stays the default to avoid surprise quota pressure.
    discuss_parallel: bool = False
    # Per-debater-turn wall-clock limit in seconds; None disables the limit.
    discuss_debater_timeout: float | None = None
    # What to do when a debater turn fails or times out: "fail" aborts the run
    # (today's behavior); "partial" continues the round with >= 2 surviving
    # votes and records the failure in the round summary.
    discuss_on_debater_failure: str = "fail"
    # Materialize discuss `split` proposals and plan-first `deferred_stages`
    # into linked child GitHub issues (#476). Default off (warning-only) so
    # existing runs keep today's behavior; the orchestrator always warns when
    # split follow-ups would otherwise remain unfiled.
    materialize_split_issues: bool = False
    # Maximum number of flat child issues owned by one parent.  This is shared
    # by typed stages, model decomposition, and discuss split proposals.
    flat_child_limit: int = DEFAULT_FLAT_CHILD_LIMIT
    # Explicit selected-stage resolution for `issue --plan-execution-mode
    # implement-one-shot` when a parent's split proposals were already fully
    # materialized into child issues (#476). None means resolve by unique
    # title match instead.
    split_stage: int | None = None
    # GitHub-backed salvage breadcrumbs (#507): post a hidden AGENT_SALVAGE marker
    # comment for failed mutating implementation attempts so a rerun with a
    # different coder/workdir/machine can still discover the latest salvage
    # context. Defaulted on so existing AgentLoopConfig constructions keep
    # working with the new behavior enabled.
    salvage_comments: bool = True
    salvage_comment_patch_max_bytes: int = 20000
    # Parallel plan/PR reviewer execution (#594). Opt-in for `issue`/`pr`/`task`
    # only; sequential stays the default. Distinct from --discuss-parallel
    # (#475), which covers discuss-mode debaters and is unaffected.
    review_parallel: bool = False
    # How long a check-run may sit queued with no job started before it is
    # treated as an external-CI-infrastructure stall rather than a normal
    # wait (#602), e.g. a GitHub-hosted-runner capacity outage.
    ci_queued_grace_seconds: int = 1200
    # Bounded re-poll for a GitHub mergeability computation still in progress
    # (`mergeable: "UNKNOWN"`), and the interval between attempts. Distinct
    # from a confirmed conflict, which is never re-polled here (#606).
    mergeability_poll_attempts: int = 3
    mergeability_poll_interval_seconds: int = 5
    # Foreground full-board CI watch after reviewer approval (#587). CLI
    # invocations enable this automatically with --auto-merge; direct config
    # constructions remain conservative unless they set it explicitly. Ordinary
    # auto-merge uses the watcher regardless of this compatibility value.
    watch_pending_ci: bool = False
    # Explicit request to activate managed exact-head CI without requesting a
    # merge. `auto_merge` remains an implicit managed-CI eligibility signal so
    # existing invocations retain their behavior.
    managed_ci: bool = False
    # Optional v2 managed-CI identity. A repository must independently opt in
    # with its Actions variable before this can suppress any automatic matrix.
    managed_ci_trusted_actor: str | None = None
    # Explicit PR-mode opt-in for adopting an already-open PR into the v2
    # managed-CI protocol.  Kept separate from the issue-created v2 flow.
    managed_ci_adopt_existing_pr: bool = False
    # A consciously per-invocation waiver for issue-created v2 only. It is
    # never read from the environment or durable PR state.
    allow_unprotected_managed_ci: bool = False
    # Runtime-only correlation value minted by the issue-created preflight.
    # It is intentionally not a CLI option: a later invocation must perform a
    # new preflight rather than accepting a PR-body token it did not create.
    managed_ci_expected_override_nonce: str | None = None
    # True only for the public `agent-loop pr` entry point. Issue-mode
    # implementation hands off to the PR loop too, but must retain the
    # issue-created activation semantics for that first invocation.
    managed_ci_pr_mode: bool = False
    invocation_argv: tuple[str, ...] = ()
    # Optional authoritative issue-closing declaration. ``None`` means that
    # this invocation made no declaration; an empty tuple is explicit and is
    # intentionally preserved for reconciliation.
    expected_closing_issue_ids: tuple[int, ...] | None = None
    supersede_expected_closing_contract: bool = False
    # Internal handoff bit: issue mode has already resolved CLI/plan additions
    # into the complete immutable contract before entering the PR loop.
    expected_closing_contract_resolved: bool = False
    # Origin for a contract created by the public direct/managed PR entry point.
    # Issue-origin handoffs select their own flow when a linked issue exists.
    pr_origin_flow: str = "direct-pr"
    # Separate bounded startup observation for CI that has not materialized a
    # run/check yet.  Empty boards are never evidence that CI succeeded.
    ci_startup_timeout_seconds: int = 120
    # CLI provenance for compatibility options. These are intentionally kept
    # separate from their effective values so auto-merge can warn when an old
    # selector or opt-out no longer changes the full-board gate.
    ci_check_name_explicit: bool = False
    watch_pending_ci_explicit: bool = False
    # The first source that resolved ``base``.  This is carried across an
    # issue-to-PR handoff so a repository-default base cannot silently replace
    # the live PR base on a later invocation.
    base_provenance: str | None = None

    @property
    def effective_managed_ci(self) -> bool:
        """Whether this invocation requested the managed-CI protocol."""
        return self.managed_ci or self.auto_merge

    def __post_init__(self) -> None:
        if isinstance(self.reviewer, str):
            object.__setattr__(self, "reviewer", (self.reviewer,))
        # Reconstructing a normalized frozen config (for example to add an
        # invocation token list) feeds the legacy single model back as its
        # equivalent one-item chain.  Treat that representation as idempotent.
        if (
            self.antigravity_model is not None
            and self.antigravity_models
            and self.antigravity_models != (self.antigravity_model,)
        ):
            raise AgentLoopError("Cannot specify both antigravity_model and a custom antigravity_models chain.")
        if self.antigravity_model is not None:
            object.__setattr__(self, "antigravity_models", (self.antigravity_model,))
        elif not self.antigravity_models:
            object.__setattr__(self, "antigravity_models", DEFAULT_ANTIGRAVITY_MODELS)
        if any(not m.strip() for m in self.antigravity_models):
            raise AgentLoopError("antigravity_models chain cannot be empty or contain blank entries.")
        if self.antigravity_print_timeout_seconds <= 0:
            raise AgentLoopError("--antigravity-print-timeout-seconds must be greater than zero.")
        if self.repair_backend not in {"antigravity", "gemini"}:
            raise AgentLoopError("--repair-backend must be either 'antigravity' or 'gemini'.")
        if not self.repair_models or any(not model.strip() for model in self.repair_models):
            raise AgentLoopError("--repair-model must contain at least one nonblank model.")
        if self.repair_timeout_seconds <= 0:
            raise AgentLoopError("--repair-timeout-seconds must be greater than zero.")
        if self.implementation_coder_model and self.implementation_coder is None:
            object.__setattr__(self, "implementation_coder", self.coder)
        if self.implementation_codex_reasoning_effort and self.implementation_coder is None:
            object.__setattr__(self, "implementation_coder", self.coder)
        if self.implementation_codex_reasoning_effort and self.implementation_coder != "codex":
            raise AgentLoopError("--implementation-codex-reasoning-effort requires --implementation-coder codex.")
        if (
            self.implementation_codex_reasoning_effort
            and self.implementation_coder == "codex"
            and not (self.implementation_coder_model or self.codex_model)
        ):
            raise AgentLoopError(
                "--implementation-codex-reasoning-effort requires "
                "--implementation-coder-model or --codex-model so the implementation "
                "signature can reliably name the Codex model."
            )
        ensure_no_model_arg_conflicts(self)
        if self.planning_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--planning-context-mode must be either 'full' or 'compact'.")
        if self.pr_review_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--pr-review-context-mode must be either 'full' or 'compact'.")
        if self.discuss_research not in DISCUSS_RESEARCH_MODES:
            rendered = ", ".join(f"'{mode}'" for mode in sorted(DISCUSS_RESEARCH_MODES))
            raise AgentLoopError(f"--discuss-research must be one of: {rendered}.")
        if self.discuss_result_mode not in DISCUSS_RESULT_MODES:
            rendered = ", ".join(f"'{mode}'" for mode in sorted(DISCUSS_RESULT_MODES))
            raise AgentLoopError(f"--discuss-result-mode must be one of: {rendered}.")
        if self.discuss_on_debater_failure not in DISCUSS_DEBATER_FAILURE_MODES:
            rendered = ", ".join(f"'{mode}'" for mode in sorted(DISCUSS_DEBATER_FAILURE_MODES))
            raise AgentLoopError(f"--discuss-on-debater-failure must be one of: {rendered}.")
        if self.discuss_debater_timeout is not None and self.discuss_debater_timeout <= 0:
            raise AgentLoopError("--discuss-debater-timeout must be greater than zero seconds.")
        if self.salvage_comment_patch_max_bytes <= 0:
            raise AgentLoopError("--salvage-comment-patch-max-bytes must be greater than zero.")
        if self.flat_child_limit <= 0:
            raise AgentLoopError("--flat-child-limit must be greater than zero.")
        normalized_expected = normalize_issue_ids(
            self.expected_closing_issue_ids,
            field_name="expected_closing_issue_ids",
        )
        object.__setattr__(self, "expected_closing_issue_ids", normalized_expected)
        if self.pr_origin_flow not in {"direct-pr", "managed-pr"}:
            raise AgentLoopError("pr_origin_flow must be 'direct-pr' or 'managed-pr'.")
        if self.base_provenance is not None and self.base_provenance not in BASE_PROVENANCE_VALUES:
            raise AgentLoopError(
                "base_provenance must be 'explicit', 'repository-default', or 'pr-metadata'."
            )


def reviewers(config: AgentLoopConfig) -> tuple[AgentName, ...]:
    return config.reviewer


def github_bootstrap_cwd(config: AgentLoopConfig) -> Path:
    candidates = (
        active_workdir(config),
        *(agent_workdir(config, reviewer) for reviewer in reviewers(config)),
        Path.cwd(),
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise AgentLoopError("No existing directory is available for GitHub repository metadata queries.")


def resolve_base_branch(
    config: AgentLoopConfig,
    runner: Runner,
    *,
    pr_metadata: PullRequestMetadata | None = None,
    cwd: Path | None = None,
) -> AgentLoopConfig:
    explicit_base = (config.base or "").strip()
    live_pr_base = (pr_metadata.base_branch or "").strip() if pr_metadata is not None else ""

    def mismatch_message(expected: str, actual: str) -> AgentLoopError:
        invocation = list(config.invocation_argv)
        if invocation:
            rewritten: list[str] = []
            index = 0
            while index < len(invocation):
                token = invocation[index]
                if token == "--base":
                    index += 2
                    continue
                if token.startswith("--base="):
                    index += 1
                    continue
                rewritten.append(token)
                index += 1
            rewritten.extend(("--base", actual))
            retry = shlex.join(rewritten)
        else:
            retry = f"agent-loop issue <issue-number> --base {shlex.quote(actual)}"
        return AgentLoopError(
            f"Configured base {expected!r} ({config.base_provenance or 'unrecorded'}) "
            f"differs from the live PR base {actual!r}. No workdir setup or remote write was performed. "
            f"Rerun the canonical issue command with an explicit live base: {retry}"
        )

    if (
        explicit_base
        and live_pr_base
        and config.base_provenance in {"repository-default", "pr-metadata"}
        and explicit_base != live_pr_base
    ):
        raise mismatch_message(explicit_base, live_pr_base)

    if explicit_base:
        resolved = config if explicit_base == config.base else replace(config, base=explicit_base)
        if resolved.base_provenance is None:
            resolved = replace(resolved, base_provenance="explicit")
        # An operator-supplied --base is authoritative.  The live PR is still
        # validated by the managed tuple and ordinary review gates.
        return resolved

    if config.dry_run:
        return replace(config, base="main", base_provenance="repository-default")

    if live_pr_base:
        return replace(config, base=live_pr_base, base_provenance="pr-metadata")

    default_branch = get_repo_default_branch(
        runner,
        config=config,
        cwd=cwd or github_bootstrap_cwd(config),
    )
    if default_branch:
        return replace(config, base=default_branch, base_provenance="repository-default")

    raise AgentLoopError(
        f"Unable to resolve a base branch for {config.repo}. "
        "Pass --base <branch> explicitly."
    )


def required_agents(config: AgentLoopConfig) -> set[AgentName]:
    required: set[AgentName] = {config.coder, *reviewers(config)}
    if config.implementation_coder is not None:
        required.add(config.implementation_coder)
    if config.discuss_analyzer is not None:
        required.add(config.discuss_analyzer)
    return required


def ensure_distinct_workdirs(config: AgentLoopConfig) -> None:
    if config.allow_shared_dir:
        return
    required = required_agents(config)
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
            "model; set the model via --antigravity-models only."
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
        # name. Rollout-file detection is best-effort, so effort alone cannot
        # reliably be labeled.
        raise AgentLoopError(
            "--codex-reasoning-effort requires --codex-model so the signature can "
            "reliably name the model. To set effort without stamping the signature, "
            "use --codex-arg -c model_reasoning_effort=... instead."
        )
    if config.gemini_model and _args_have_model_flag(config.gemini_args):
        raise AgentLoopError(
            "--gemini-arg --model conflicts with --gemini-model; use --gemini-model only."
        )
    if config.claude_model and _args_have_model_flag(config.claude_args):
        raise AgentLoopError(
            "--claude-arg --model conflicts with --claude-model; use --claude-model only."
        )
    if config.implementation_coder_model:
        if config.implementation_coder == "antigravity" and _args_have_model_flag(config.antigravity_args):
            raise AgentLoopError(
                "--antigravity-arg --model conflicts with --implementation-coder-model; "
                "use --implementation-coder-model only."
            )
        if config.implementation_coder == "codex" and _args_have_model_flag(config.codex_args):
            raise AgentLoopError(
                "--codex-arg --model conflicts with --implementation-coder-model; "
                "use --implementation-coder-model only."
            )
        if config.implementation_coder == "gemini" and _args_have_model_flag(config.gemini_args):
            raise AgentLoopError(
                "--gemini-arg --model conflicts with --implementation-coder-model; "
                "use --implementation-coder-model only."
            )
        if config.implementation_coder == "claude" and _args_have_model_flag(config.claude_args):
            raise AgentLoopError(
                "--claude-arg --model conflicts with --implementation-coder-model; "
                "use --implementation-coder-model only."
            )
    if (
        config.implementation_codex_reasoning_effort
        and config.implementation_coder == "codex"
        and _args_have_reasoning_effort(config.codex_args)
    ):
        raise AgentLoopError(
            "--codex-arg model_reasoning_effort conflicts with "
            "--implementation-codex-reasoning-effort; use --implementation-codex-reasoning-effort only."
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


def default_subprocess_log_dir(repo: str) -> Path:
    """Return a unique, checkout-independent capture directory."""
    return (
        default_cache_root() / "subprocess-logs" / repo_cache_slug(repo)
        / f"{datetime_stamp()}-{uuid.uuid4().hex}"
    )


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


def _resolve_subprocess_log_dir(
    args: argparse.Namespace, *, repo: str, primary_dir: Path, managed_roots: tuple[Path, ...]
) -> Path:
    value = getattr(args, "subprocess_log_dir", None)
    raw = Path(value) if value is not None else default_subprocess_log_dir(repo)
    path = (primary_dir / raw if value is not None and not raw.is_absolute() else raw).expanduser().resolve()
    if any(path == root or root in path.parents for root in managed_roots):
        raise AgentLoopError(
            "--subprocess-log-dir must not be equal to or nested beneath a managed checkout: "
            f"{path}"
        )
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AgentLoopError(f"Could not create --subprocess-log-dir at {path}: {exc}") from exc
    return path


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
    if not config.base:
        raise AgentLoopError(
            f"Base branch for {config.repo} was not resolved. Pass --base <branch> explicitly."
        )
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
    required = required_agents(config)
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


def preflight_agent_commands(
    args: argparse.Namespace,
    runner: Runner,
    configured_reviewers: tuple[AgentName, ...],
) -> None:
    """Validate configured agent CLIs before repository or workdir operations."""
    if args.dry_run:
        return

    command_options = {
        "claude": (args.claude_cmd, "--claude-cmd"),
        "codex": (args.codex_cmd, "--codex-cmd"),
        "gemini": (args.gemini_cmd, "--gemini-cmd"),
        "antigravity": (args.antigravity_cmd, "--antigravity-cmd"),
    }
    discuss_analyzer = getattr(args, "discuss_analyzer", None)
    implementation_coder = getattr(args, "implementation_coder", None)
    configured_agents = dict.fromkeys(
        (
            args.coder,
            *((implementation_coder,) if implementation_coder is not None else ()),
            *configured_reviewers,
            *((discuss_analyzer,) if discuss_analyzer is not None else ()),
            getattr(args, "repair_backend", "antigravity"),
        )
    )
    for agent in configured_agents:
        command, override_flag = command_options[agent]
        resolved = shutil.which(command)
        if resolved is None:
            if os.path.isabs(command):
                raise AgentLoopError(
                    f"{command} not found or not executable; "
                    f"pass a valid executable path to {override_flag}."
                )
            raise AgentLoopError(
                f"{command} CLI not found on PATH; install it or pass "
                f"{override_flag} <path>."
            )
        runner.remember_agent_command(command, resolved, override_flag)


def config_from_args(
    args: argparse.Namespace,
    runner: Runner,
    *,
    invocation_argv: tuple[str, ...] = (),
) -> AgentLoopConfig:
    configured_reviewers = tuple(args.reviewer or ["codex"])
    if len(set(configured_reviewers)) != len(configured_reviewers):
        raise AgentLoopError("--reviewer cannot include the same agent more than once.")
    preflight_agent_commands(args, runner, configured_reviewers)

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
    if getattr(args, "ci_startup_timeout_seconds", 120) <= 0:
        raise AgentLoopError("--ci-startup-timeout-seconds must be greater than zero.")
    if getattr(args, "ci_queued_grace_seconds", 1200) <= 0:
        raise AgentLoopError("--ci-queued-grace-seconds must be greater than zero.")
    if getattr(args, "mergeability_poll_attempts", 3) <= 0:
        raise AgentLoopError("--mergeability-poll-attempts must be greater than zero.")
    if getattr(args, "mergeability_poll_interval_seconds", 5) <= 0:
        raise AgentLoopError("--mergeability-poll-interval-seconds must be greater than zero.")
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
        base=getattr(args, "base", None),
        base_provenance="explicit" if getattr(args, "base", None) else None,
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
        antigravity_models=tuple(args.antigravity_models) if getattr(args, "antigravity_models", None) is not None else (),
        antigravity_print_timeout_seconds=getattr(
            args,
            "antigravity_print_timeout_seconds",
            DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS,
        ),
        antigravity_quota_signatures=tuple(
            getattr(args, "antigravity_quota_signatures", None)
            or DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES
        ),
        codex_model=getattr(args, "codex_model", ""),
        codex_reasoning_effort=getattr(args, "codex_reasoning_effort", ""),
        gemini_model=getattr(args, "gemini_model", ""),
        claude_model=getattr(args, "claude_model", ""),
        implementation_coder=getattr(args, "implementation_coder", None),
        implementation_coder_model=getattr(args, "implementation_coder_model", ""),
        implementation_codex_reasoning_effort=getattr(args, "implementation_codex_reasoning_effort", ""),
        repair_backend=getattr(args, "repair_backend", "antigravity"),
        repair_models=tuple(getattr(args, "repair_model", None) or DEFAULT_REPAIR_MODELS),
        repair_timeout_seconds=getattr(args, "repair_timeout_seconds", 120),
        discuss_analyzer=getattr(args, "discuss_analyzer", None),
        discuss_research=getattr(args, "discuss_research", "none") or "none",
        discuss_result_mode=getattr(args, "discuss_result_mode", "triage") or "triage",
        discuss_parallel=getattr(args, "discuss_parallel", False),
        discuss_debater_timeout=getattr(args, "discuss_debater_timeout", None),
        discuss_on_debater_failure=getattr(args, "discuss_on_debater_failure", "fail") or "fail",
        materialize_split_issues=getattr(args, "materialize_split_issues", False),
        flat_child_limit=getattr(args, "flat_child_limit", DEFAULT_FLAT_CHILD_LIMIT),
        split_stage=getattr(args, "split_stage", None),
        salvage_comments=getattr(args, "salvage_comments", True),
        salvage_comment_patch_max_bytes=getattr(args, "salvage_comment_patch_max_bytes", 20000),
        review_parallel=getattr(args, "review_parallel", False),
        test_command=test_command,
        pre_review_tests=args.pre_review_tests,
        ci_check_name=getattr(args, "ci_check_name", None) or "test",
        ci_timeout_seconds=args.ci_timeout_seconds,
        ci_poll_interval_seconds=args.ci_poll_interval_seconds,
        ci_startup_timeout_seconds=getattr(args, "ci_startup_timeout_seconds", 120),
        watch_pending_ci=(
            bool(getattr(args, "auto_merge", False))
            if getattr(args, "watch_pending_ci", None) is None
            else bool(args.watch_pending_ci)
        ),
        ci_check_name_explicit=getattr(args, "ci_check_name", None) is not None,
        watch_pending_ci_explicit=getattr(args, "watch_pending_ci", None) is not None,
        managed_ci_trusted_actor=getattr(args, "managed_ci_trusted_actor", None),
        managed_ci=getattr(args, "managed_ci", False),
        managed_ci_adopt_existing_pr=getattr(args, "managed_ci_adopt_existing_pr", False),
        allow_unprotected_managed_ci=getattr(args, "allow_unprotected_managed_ci", False),
        managed_ci_pr_mode=getattr(args, "command", None) == "pr",
        invocation_argv=invocation_argv,
        expected_closing_issue_ids=normalize_issue_ids(
            getattr(args, "expected_closing_issue", None),
            field_name="--expected-closing-issue",
        ),
        supersede_expected_closing_contract=getattr(
            args, "supersede_expected_closing_contract", False
        ),
        ci_queued_grace_seconds=getattr(args, "ci_queued_grace_seconds", 1200),
        mergeability_poll_attempts=getattr(args, "mergeability_poll_attempts", 3),
        mergeability_poll_interval_seconds=getattr(args, "mergeability_poll_interval_seconds", 5),
        quiet=args.quiet,
        log_dir=(primary_dir / args.log_dir if not args.log_dir.is_absolute() else args.log_dir),
        subprocess_log_dir=_resolve_subprocess_log_dir(args, repo=repo, primary_dir=primary_dir, managed_roots=(claude_dir, codex_dir, gemini_dir, antigravity_dir)),
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
