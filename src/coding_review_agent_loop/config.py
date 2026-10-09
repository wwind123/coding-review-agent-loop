"""Configuration construction and validation."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import shutil
import sys
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from .agents.base import AgentName
from .agents.registry import default_agent_args
from .checkout_verification import (
    check_worktree_links,
    forget_checkout,
    recover_gemini_injection,
    refresh_after_sync,
    register_worktree_links,
    registered_links,
    verify_before_sync,
)
from .errors import AgentLoopError
from .expected_closure import normalize_issue_ids
from .github import (
    PullRequestMetadata,
    TrustedHumanActor,
    detect_repo,
    get_repo_default_branch,
)
from .logging import datetime_stamp, log
from .runner import Runner
from .scratch import make_private_dirs, scratch_root
from .test_runtime import DEFAULT_TEST_TIMEOUT_SECONDS
from .workdirs import active_workdir, agent_workdir
from .plan_review_scheduling import PLAN_REVIEW_POLICIES
from .review_scheduling import (
    DEFAULT_BROAD_RULES,
    PR_REVIEW_POLICIES,
    normalize_broad_rules,
)

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
    "Gemini 3.8 Flash (High)",
    "Gemini 3.7 Flash (High)",
    "Gemini 3.6 Flash (High)",
    "Gemini 3.1 Pro (High)",
    # Served on a quota separate from Gemini's, so it only runs when the Gemini
    # group cannot serve (#1236). Two reviewers may then share one model.
    "Claude Opus 5.5 (Medium)",
)
# How long an Antigravity quota group with no parsed reset is skipped (#1236).
DEFAULT_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS = 600
MAX_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS = 3600
DEFAULT_MAX_ROUNDS = 10
# Primary-phase plan stall stop threshold (#1103); 0 disables it.
DEFAULT_PLAN_PRIMARY_STALL_ROUNDS = 8
DEFAULT_PLAN_STEP_BACK_ROUNDS = 2
DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS = 2
# PR fix-loop step-back (#1251): K clustered new-finding blocks, and the line window.
DEFAULT_PR_STEP_BACK_ROUNDS = 3
DEFAULT_PR_EVIDENCE_STALL_ROUNDS = 2
DEFAULT_PR_STEP_BACK_LINE_WINDOW = 40
DEFAULT_SUB_ITEM_STALL_ROUNDS = 3
# `agy --print` otherwise defaults to five minutes, which is too short for
# complex reviews and causes it to exit with "timeout waiting for response".
DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS = 10 * 60
DEFAULT_REPAIR_MODELS: tuple[str, ...] = (
    "Gemini 3.8 Flash (Medium)",
    "Gemini 3.7 Flash (Medium)",
)
# agy reports a transient model-access failure on its provider channel (stdout,
# with no response artifact) with this text.  Matched case-insensitively; the
# singular form also matches the plural "errors".
ANTIGRAVITY_TRANSIENT_MODEL_ACCESS_SIGNATURES: tuple[str, ...] = (
    "model-access validation error",
)
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_SEMANTIC_FOLLOWUP_BACKEND: AgentName = "gemini"
DEFAULT_SEMANTIC_FOLLOWUP_TIMEOUT_SECONDS = 30
DEFAULT_SEMANTIC_FOLLOWUP_MAX_CALLS = 5
DEFAULT_SEMANTIC_FOLLOWUP_MAX_CANDIDATES = 50
DEFAULT_SEMANTIC_FOLLOWUP_PROMPT_CHAR_LIMIT = 12_000
CODEX_REASONING_EFFORTS: frozenset[str] = frozenset(
    {"minimal", "low", "medium", "high", "xhigh"}
)
CLAUDE_EFFORTS: frozenset[str] = frozenset(
    {"low", "medium", "high", "xhigh", "max"}
)
# One parent-wide flat topology budget shared by decomposition and split
# materialization.  Keeping this policy in config prevents each workflow from
# obtaining a second allowance of children.
DEFAULT_FLAT_CHILD_LIMIT: int = 15
# Discuss-mode research policy values (#477). The CLI flag choices, the config
# validation, and the prompt builders all derive from this set.
DISCUSS_RESEARCH_MODES: frozenset[str] = frozenset({"none", "required", "auto"})

# The issue plan-first policy is intentionally separate from the reviewed
# execution strategy.  ``auto`` is only a request to select one of the two
# canonical post-approval actions after the reviewed recommendation exists.
PLAN_EXECUTION_MODES: frozenset[str] = frozenset(
    {"plan-only", "decompose-only", "implement-one-shot", "implement-by-phase", "auto"}
)



def phased_delivery_guard_active(mode: str) -> bool:
    """Whether the phased-delivery guard applies to ``mode`` (#1268).

    Shared by the prompt guard, the deterministic staged-plan stop, and the
    ``--plan-narrow-staged`` validation so they cannot drift.
    """
    return mode not in {"decompose-only", "implement-by-phase", "auto"}


# Discuss-mode debater failure policy values (#475).
DISCUSS_DEBATER_FAILURE_MODES: frozenset[str] = frozenset({"fail", "partial"})
DISCUSS_RESULT_MODES: frozenset[str] = frozenset({"triage", "answer"})
BASE_PROVENANCE_VALUES = frozenset({"explicit", "repository-default", "pr-metadata"})


@dataclass(frozen=True)
class ResolvedInvocation:
    """Provider configuration resolved for one substantive invocation."""

    provider: AgentName
    role: str | None
    configured_model: str | None
    resolved_effort: str | None
    effort_source: str | None


def validate_test_python(value: object) -> str:
    """Validate ``--test-python`` and return its lexical path unchanged (#1343).

    The path must be absolute and its realpath an existing regular file the
    user can execute.  ``realpath`` is used only for this check; the stored,
    exported, invoked and granted value is the path exactly as given.  There
    is no PATH or ``.venv`` discovery.
    """
    if not isinstance(value, str) or not value.strip():
        raise AgentLoopError("--test-python must be an absolute path to a Python interpreter.")
    if not os.path.isabs(value):
        raise AgentLoopError(f"--test-python must be an absolute path, not {value!r}.")
    real = os.path.realpath(value)
    if not os.path.exists(real):
        raise AgentLoopError(f"--test-python {value!r} does not exist.")
    if not os.path.isfile(real):
        raise AgentLoopError(f"--test-python {value!r} is not a regular file.")
    if not os.access(real, os.X_OK):
        raise AgentLoopError(f"--test-python {value!r} is not executable.")
    return value


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
    # Approved-follow-up semantic reuse is deliberately bounded and can be
    # disabled for offline/reproducibility-sensitive invocations.
    semantic_followup_dedupe: bool = True
    semantic_followup_backend: AgentName = DEFAULT_SEMANTIC_FOLLOWUP_BACKEND
    semantic_followup_model: str = ""
    semantic_followup_timeout_seconds: int = DEFAULT_SEMANTIC_FOLLOWUP_TIMEOUT_SECONDS
    semantic_followup_max_calls: int = DEFAULT_SEMANTIC_FOLLOWUP_MAX_CALLS
    semantic_followup_max_candidates: int = DEFAULT_SEMANTIC_FOLLOWUP_MAX_CANDIDATES
    semantic_followup_prompt_char_limit: int = DEFAULT_SEMANTIC_FOLLOWUP_PROMPT_CHAR_LIMIT
    plan_execution_mode: str = "plan-only"
    # Fresh planning is generation-1 by default.  A caller that is explicitly
    # decoding historical transcripts may set this false so unversioned plans
    # remain distinguishable as legacy-undecided.
    execution_strategy_contract_required: bool = True
    planning_context_mode: str = "compact"
    pr_review_context_mode: str = "full"
    # PR-only intermediate scheduling.  ``all-reviewers`` preserves the
    # historical contract; selective scheduling is explicitly opt-in.
    pr_review_policy: str = "all-reviewers"
    # Primary reviewer used only by the staged primary-then-panel policy.
    primary_reviewer: AgentName | None = None
    pr_review_broad_rules: tuple[str, ...] = DEFAULT_BROAD_RULES
    pr_review_force_full: bool = False
    # Issue plan-review scheduling, selected independently of the PR policy
    # (#905, from #841).  ``all-reviewers`` is the compatibility default.
    plan_review_policy: str = "all-reviewers"
    primary_plan_reviewer: AgentName | None = None
    plan_review_force_full: bool = False
    # Primary-phase stall stop (#1103): consecutive completed blocking primary
    # plan reviews, with no exact-plan primary approval, after which a
    # primary-then-panel run stops instead of re-invoking the primary.  0
    # disables the stop.
    plan_primary_stall_rounds: int = DEFAULT_PLAN_PRIMARY_STALL_ROUNDS
    # Plan step-back turn (#1251): once a one-shot plan has crossed a growth
    # signal and the primary has blocked this many consecutive rounds on new
    # findings, the next planner turn is a simplify-or-re-scope revision.  0
    # disables it.  After it, this many further primary blocks (the block of the
    # step-back candidate itself counts as the first) stop for a human decision.
    plan_step_back_rounds: int = DEFAULT_PLAN_STEP_BACK_ROUNDS
    plan_step_back_escalation_rounds: int = DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS
    pr_step_back_rounds: int = DEFAULT_PR_STEP_BACK_ROUNDS
    pr_evidence_stall_rounds: int = DEFAULT_PR_EVIDENCE_STALL_ROUNDS
    pr_step_back_line_window: int = DEFAULT_PR_STEP_BACK_LINE_WINDOW
    # One-shot operator retirement of the stall streak (#1112): the run's first
    # primary-phase checkpoint is stamped as a durable streak boundary.  Never
    # inherited by child planning, isolated providers, or recovery commands.
    plan_reset_stall_streak: bool = False
    plan_narrow_staged: bool = False
    # Plan-growth gate (#886): structural thresholds above which a one-shot
    # plan needs a reviewed justification or a staged restructure.
    plan_growth_gate: str = "enforce"
    plan_growth_max_chars: int = 120_000
    plan_growth_max_revisions: int = 6
    plan_growth_max_scope_items: int = 12
    plan_growth_max_matrix_rows: int = 18
    # Advisory stall window for conjunctive findings' sub-items (#958); 0 disables.
    sub_item_stall_rounds: int = DEFAULT_SUB_ITEM_STALL_ROUNDS
    auto_agent_dirs: tuple[AgentName, ...] = ()
    # Per-run worktrees (#1162): the shared store behind each auto agent's
    # per-run worktree, and the token naming this run's worktrees.  Empty for
    # directly constructed configs, which keep the standalone-clone behaviour.
    default_checkout_stores: tuple[tuple[AgentName, Path], ...] = ()
    run_token: str | None = None
    worktree_links: tuple[str, ...] = ()
    # Optional plan-first override: use the main coder for planning/revision,
    # then switch only the approved implementation and PR follow-up coder/model.
    implementation_coder: AgentName | None = None
    implementation_coder_model: str = ""
    implementation_codex_reasoning_effort: str = ""
    implementation_claude_effort: str = ""
    # Antigravity (`agy`) backend (#215). Defaulted so existing AgentLoopConfig
    # constructions keep working; real values are set when antigravity is used.
    antigravity_dir: Path = Path("antigravity")
    antigravity_cmd: str = "agy"
    antigravity_args: tuple[str, ...] = ()
    antigravity_model: str | None = None
    antigravity_models: tuple[str, ...] = ()
    antigravity_print_timeout_seconds: int = DEFAULT_ANTIGRAVITY_PRINT_TIMEOUT_SECONDS
    antigravity_quota_signatures: tuple[str, ...] = DEFAULT_ANTIGRAVITY_QUOTA_SIGNATURES
    antigravity_quota_cooldown_seconds: int = DEFAULT_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS
    # Explicit (model, group) overrides of the derived quota group (#1236).
    antigravity_quota_groups: tuple[tuple[str, str], ...] = ()
    # Declared model / reasoning effort for the dynamic signature (#332). Empty
    # means "not declared" (the agent runs its own default and the signature
    # falls back to the generic provider name). antigravity always has a model.
    codex_model: str = ""
    codex_reasoning_effort: str = ""
    reviewer_codex_model: str = ""
    reviewer_codex_reasoning_effort: str = ""
    reviewer_claude_model: str = ""
    reviewer_claude_effort: str = ""
    gemini_model: str = ""
    claude_model: str = ""
    claude_effort: str = ""
    repair_backend: str = "antigravity"
    repair_models: tuple[str, ...] = DEFAULT_REPAIR_MODELS
    repair_reasoning_effort: str = ""
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
    # Optional identities whose signed human reviewer comments are admitted
    # as approval-critical requirements (#1022).  Empty means unconfigured:
    # every signed comment is admitted but labelled unverified.
    human_reviewer_trusted_actors: tuple[TrustedHumanActor, ...] = ()
    # Explicit PR-mode opt-in for adopting an already-open PR into the v2
    # managed-CI protocol.  Kept separate from the issue-created v2 flow.
    managed_ci_adopt_existing_pr: bool = False
    # A consciously per-invocation waiver for issue-created v2 only. It is
    # never read from the environment or durable PR state.
    allow_unprotected_managed_ci: bool = False
    # A separate explicit waiver for "unreadable" protection: classic branch
    # protection refused this token with HTTP 403 and the readable effective
    # rules show no strict enforcement (#1040).  Requires the waiver above.
    allow_unreadable_protection: bool = False
    # Operator recovery for a refused PR resume (#1292): run an ordinary review
    # round of the current head from the recorded active items.
    review_unrecorded_head: bool = False
    # Runtime-only correlation value minted by the issue-created preflight.
    # It is intentionally not a CLI option: a later invocation must perform a
    # new preflight rather than accepting a PR-body token it did not create.
    managed_ci_expected_override_nonce: str | None = None
    # Explicit opt-in for exceptional recovery when the original issue-created
    # authorization checkpoint was never published or cannot be verified.
    managed_ci_fresh_authorization: bool = False
    # PR mode must name the issue scope instead of inferring it from candidate
    # controlled body text.
    managed_ci_issue_number: int | None = None
    # True only for the public `agent-loop pr` entry point. Issue-mode
    # implementation hands off to the PR loop too, but must retain the
    # issue-created activation semantics for that first invocation.
    managed_ci_pr_mode: bool = False
    # Internal marker used only after an approved-plan coder switch. It keeps
    # the original agent-wide values intact while allowing the active coder
    # turn to report a role override as its source.
    implementation_effort_active: bool = False
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
    # CLI provenance for the compatibility opt-out. It remains separate from
    # its effective value so auto-merge can warn when the old opt-out no longer
    # changes the full-board gate.
    watch_pending_ci_explicit: bool = False
    # The first source that resolved ``base``.  This is carried across an
    # issue-to-PR handoff so a repository-default base cannot silently replace
    # the live PR base on a later invocation.
    base_provenance: str | None = None
    # Finite run-level ceiling for local coder test commands.  Kept at the end
    # with a default so direct AgentLoopConfig callers remain source-compatible.
    coder_test_command_timeout_seconds: int = DEFAULT_TEST_TIMEOUT_SECONDS
    # Operator-configured test interpreter (#1343), kept as the lexical
    # absolute path: two virtualenv ``bin/python`` symlinks can share one real
    # executable while exposing different site-packages.
    test_python: str | None = None
    # Process-tree containment.  Values remain unparsed at the config boundary
    # so CLI strings such as ``70%`` and ``2GiB`` are resolved exactly once by
    # containment.policy_from_values().
    containment_mode: str = "auto"
    containment_memory_high: object | None = None
    containment_memory_max: object | None = None
    containment_memory_swap_max: object | None = None
    containment_tasks_max: object | None = None
    containment_aggregate_memory_high: object | None = None
    containment_aggregate_memory_max: object | None = None
    containment_aggregate_memory_swap_max: object | None = None
    containment_aggregate_tasks_max: object | None = None
    containment_os_headroom_percent: object = 25.0
    containment_slice: str = "agent-loop.slice"
    containment_cache_dir: Path | None = None
    containment_coder_memory_high: object | None = None
    containment_coder_memory_max: object | None = None
    containment_coder_memory_swap_max: object | None = None
    containment_coder_tasks_max: object | None = None
    containment_reviewer_memory_high: object | None = None
    containment_reviewer_memory_max: object | None = None
    containment_reviewer_memory_swap_max: object | None = None
    containment_reviewer_tasks_max: object | None = None
    containment_repair_memory_high: object | None = None
    containment_repair_memory_max: object | None = None
    containment_repair_memory_swap_max: object | None = None
    containment_repair_tasks_max: object | None = None
    containment_test_gate_memory_high: object | None = None
    containment_test_gate_memory_max: object | None = None
    containment_test_gate_memory_swap_max: object | None = None
    containment_test_gate_tasks_max: object | None = None
    # Containment-aware parallel test-worker budget (issue #848).  None means
    # derive the budget after containment admission.
    test_workers: int | None = None
    test_worker_memory: object | None = None
    test_worker_enforcement: str = "clamp"
    # Architecture context is advisory and contributes no prompt text unless a
    # frozen, successfully acquired snapshot is attached by orchestration.
    architecture_context_enabled: bool = True
    architecture_path: str = "ARCHITECTURE.md"
    architecture_read_size: int = 64 * 1024
    architecture_snapshot_max_chars: int = 12_000
    architecture_aggregate_max_chars: int = 24_000
    managed_context_max_chars: int = 80_000
    architecture_context: object | None = None
    # Agent permission mode (#1035): ``default`` and ``dangerous`` keep the
    # static per-provider args; ``sandboxed`` builds role-scoped grants per
    # invocation (see agent_permissions.py).
    agent_permissions: str = "default"

    @property
    def effective_managed_ci(self) -> bool:
        """Whether this invocation requested the managed-CI protocol."""
        return self.managed_ci or self.auto_merge

    def __post_init__(self) -> None:
        if isinstance(self.reviewer, str):
            object.__setattr__(self, "reviewer", (self.reviewer,))
        from .architecture_context import normalize_architecture_path

        object.__setattr__(self, "architecture_path", normalize_architecture_path(self.architecture_path))
        for option_name, value in (
            ("--architecture-read-size", self.architecture_read_size),
            ("--architecture-snapshot-max-chars", self.architecture_snapshot_max_chars),
            ("--architecture-aggregate-max-chars", self.architecture_aggregate_max_chars),
            ("--managed-context-max-chars", self.managed_context_max_chars),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AgentLoopError(f"{option_name} must be a positive integer.")
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
        cooldown = self.antigravity_quota_cooldown_seconds
        if (
            isinstance(cooldown, bool)
            or not isinstance(cooldown, int)
            or not 0 < cooldown <= MAX_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS
        ):
            raise AgentLoopError(
                "--antigravity-quota-cooldown-seconds must be a positive integer "
                f"no larger than {MAX_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS}."
            )
        seen_group_models: set[str] = set()
        for override in self.antigravity_quota_groups:
            if (
                not isinstance(override, tuple)
                or len(override) != 2
                or not all(isinstance(part, str) and part.strip() for part in override)
            ):
                raise AgentLoopError("--antigravity-quota-group must be MODEL=GROUP with non-blank parts.")
            if override[0] in seen_group_models:
                raise AgentLoopError(f"--antigravity-quota-group names {override[0]!r} more than once.")
            seen_group_models.add(override[0])
        if self.semantic_followup_backend not in {"claude", "codex", "gemini", "antigravity"}:
            raise AgentLoopError("--semantic-followup-backend must name a supported agent.")
        for option_name, value in (
            ("--semantic-followup-timeout-seconds", self.semantic_followup_timeout_seconds),
            ("--semantic-followup-max-calls", self.semantic_followup_max_calls),
            ("--semantic-followup-max-candidates", self.semantic_followup_max_candidates),
            ("--semantic-followup-prompt-char-limit", self.semantic_followup_prompt_char_limit),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AgentLoopError(f"{option_name} must be a positive integer.")
        timeout = self.coder_test_command_timeout_seconds
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise AgentLoopError(
                "--coder-test-command-timeout-seconds must be a positive finite integer."
            )
        if not math.isfinite(float(timeout)) or timeout <= 0 or float(timeout) != int(timeout):
            raise AgentLoopError(
                "--coder-test-command-timeout-seconds must be a positive finite integer."
            )
        object.__setattr__(self, "coder_test_command_timeout_seconds", int(timeout))
        if self.test_python is not None:
            object.__setattr__(self, "test_python", validate_test_python(self.test_python))
        if self.repair_backend not in {"antigravity", "gemini", "codex", "claude"}:
            raise AgentLoopError("--repair-backend must be antigravity, gemini, codex, or claude.")
        if self.repair_backend in {"codex", "claude"}:
            if self.repair_models == DEFAULT_REPAIR_MODELS:
                raise AgentLoopError(
                    "Codex/Claude repair requires an explicit --repair-model. "
                    "Pass --repair-model MODEL (for example "
                    "--repair-backend codex --repair-model MODEL); "
                    "repeat it for a fallback chain."
                )
            _validate_effort_value(
                self.repair_backend, self.repair_reasoning_effort, "--repair-reasoning-effort"
            )
        elif self.repair_reasoning_effort:
            raise AgentLoopError("--repair-reasoning-effort requires --repair-backend codex or claude.")
        if not self.repair_models or any(not model.strip() for model in self.repair_models):
            raise AgentLoopError("--repair-model must contain at least one nonblank model.")
        if self.repair_timeout_seconds <= 0:
            raise AgentLoopError("--repair-timeout-seconds must be greater than zero.")
        if self.implementation_coder_model and self.implementation_coder is None:
            object.__setattr__(self, "implementation_coder", self.coder)
        if self.implementation_codex_reasoning_effort and self.implementation_coder is None:
            object.__setattr__(self, "implementation_coder", self.coder)
        if self.implementation_claude_effort and self.implementation_coder is None:
            object.__setattr__(self, "implementation_coder", self.coder)
        if self.implementation_codex_reasoning_effort and self.implementation_coder != "codex":
            raise AgentLoopError("--implementation-codex-reasoning-effort requires --implementation-coder codex.")
        if self.implementation_claude_effort and self.implementation_coder != "claude":
            raise AgentLoopError("--implementation-claude-effort requires --implementation-coder claude.")
        _validate_effort_value(
            "codex", self.codex_reasoning_effort, "--codex-reasoning-effort"
        )
        _validate_effort_value("claude", self.claude_effort, "--claude-effort")
        _validate_effort_value(
            "codex", self.reviewer_codex_reasoning_effort, "--reviewer-codex-reasoning-effort"
        )
        _validate_effort_value("claude", self.reviewer_claude_effort, "--reviewer-claude-effort")
        for option in ("reviewer_codex_model", "reviewer_claude_model"):
            value = getattr(self, option)
            if value and not value.strip():
                raise AgentLoopError(f"--{option.replace('_', '-')} must not be blank.")
        _validate_effort_value(
            "codex",
            self.implementation_codex_reasoning_effort,
            "--implementation-codex-reasoning-effort",
        )
        _validate_effort_value(
            "claude",
            self.implementation_claude_effort,
            "--implementation-claude-effort",
        )
        ensure_no_model_arg_conflicts(self)
        from .agent_permissions import (
            AGENT_PERMISSION_MODES,
            validate_sandboxed_passthrough,
            validate_sandboxed_selections,
        )

        if self.agent_permissions not in AGENT_PERMISSION_MODES:
            raise AgentLoopError(
                "--agent-permissions must be one of: " + ", ".join(AGENT_PERMISSION_MODES) + "."
            )
        if self.agent_permissions == "sandboxed":
            validate_sandboxed_passthrough(
                {
                    "--claude-arg": self.claude_args,
                    "--codex-arg": self.codex_args,
                    "--gemini-arg": self.gemini_args,
                    "--antigravity-arg": self.antigravity_args,
                }
            )
            validate_sandboxed_selections(self)
        if self.planning_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--planning-context-mode must be either 'full' or 'compact'.")
        if self.plan_execution_mode not in PLAN_EXECUTION_MODES:
            rendered = ", ".join(f"'{mode}'" for mode in sorted(PLAN_EXECUTION_MODES))
            raise AgentLoopError(f"--plan-execution-mode must be one of: {rendered}.")
        if self.pr_review_context_mode not in {"full", "compact"}:
            raise AgentLoopError("--pr-review-context-mode must be either 'full' or 'compact'.")
        if self.pr_review_policy not in PR_REVIEW_POLICIES:
            raise AgentLoopError(
                "--pr-review-policy must be 'all-reviewers', 'selective-intermediate', "
                "or 'primary-then-panel'."
            )
        configured_reviewers = tuple(self.reviewer)
        if not configured_reviewers or len(set(configured_reviewers)) != len(configured_reviewers):
            raise AgentLoopError("Reviewer configuration must be a unique, non-empty board.")
        if self.pr_review_policy == "primary-then-panel":
            if len(configured_reviewers) < 2:
                raise AgentLoopError(
                    "--pr-review-policy primary-then-panel requires at least one secondary reviewer."
                )
            if self.primary_reviewer is None:
                raise AgentLoopError(
                    "--primary-reviewer is required with --pr-review-policy primary-then-panel."
                )
            if self.primary_reviewer not in configured_reviewers:
                raise AgentLoopError(
                    "--primary-reviewer must be one of the configured --reviewer agents."
                )
        elif self.primary_reviewer is not None:
            raise AgentLoopError(
                "--primary-reviewer requires --pr-review-policy primary-then-panel."
            )
        if self.plan_review_policy not in PLAN_REVIEW_POLICIES:
            raise AgentLoopError(
                "--plan-review-policy must be 'all-reviewers' or 'primary-then-panel'."
            )
        if self.plan_review_policy == "primary-then-panel":
            if len(configured_reviewers) < 2:
                raise AgentLoopError(
                    "--plan-review-policy primary-then-panel requires at least one secondary reviewer."
                )
            if self.primary_plan_reviewer is None:
                raise AgentLoopError(
                    "--primary-plan-reviewer is required with "
                    "--plan-review-policy primary-then-panel."
                )
            if self.primary_plan_reviewer not in configured_reviewers:
                raise AgentLoopError(
                    "--primary-plan-reviewer must be one of the configured --reviewer agents."
                )
        elif self.primary_plan_reviewer is not None:
            raise AgentLoopError(
                "--primary-plan-reviewer requires --plan-review-policy primary-then-panel."
            )
        if self.plan_review_force_full and self.plan_review_policy != "primary-then-panel":
            raise AgentLoopError(
                "--plan-review-force-full requires --plan-review-policy primary-then-panel."
            )
        if self.plan_reset_stall_streak and self.plan_review_policy != "primary-then-panel":
            raise AgentLoopError(
                "--plan-reset-stall-streak requires --plan-review-policy primary-then-panel."
            )
        if self.plan_narrow_staged and not phased_delivery_guard_active(self.plan_execution_mode):
            raise AgentLoopError(
                "--plan-narrow-staged requires a --plan-execution-mode that applies the "
                "phased-delivery guard (plan-only or implement-one-shot); "
                f"'{self.plan_execution_mode}' already accepts staged plans."
            )
        stall_rounds = self.plan_primary_stall_rounds
        if (
            isinstance(stall_rounds, bool)
            or not isinstance(stall_rounds, int)
            or stall_rounds < 0
        ):
            raise AgentLoopError(
                "--plan-primary-stall-rounds must be a non-negative integer (0 disables it)."
            )
        if (
            stall_rounds != DEFAULT_PLAN_PRIMARY_STALL_ROUNDS
            and self.plan_review_policy != "primary-then-panel"
        ):
            raise AgentLoopError(
                "--plan-primary-stall-rounds requires --plan-review-policy primary-then-panel."
            )
        step_back_rounds = self.plan_step_back_rounds
        if (
            isinstance(step_back_rounds, bool)
            or not isinstance(step_back_rounds, int)
            or step_back_rounds < 0
        ):
            raise AgentLoopError(
                "--plan-step-back-rounds must be a non-negative integer (0 disables it)."
            )
        escalation_rounds = self.plan_step_back_escalation_rounds
        if (
            isinstance(escalation_rounds, bool)
            or not isinstance(escalation_rounds, int)
            or escalation_rounds < 1
        ):
            raise AgentLoopError(
                "--plan-step-back-escalation-rounds must be a positive integer."
            )
        for flag, value, default in (
            ("--plan-step-back-rounds", step_back_rounds, DEFAULT_PLAN_STEP_BACK_ROUNDS),
            (
                "--plan-step-back-escalation-rounds",
                escalation_rounds,
                DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS,
            ),
        ):
            if value != default and self.plan_review_policy != "primary-then-panel":
                raise AgentLoopError(
                    f"{flag} requires --plan-review-policy primary-then-panel."
                )
        for flag, value in (
            ("--pr-step-back-rounds", self.pr_step_back_rounds),
            ("--pr-step-back-line-window", self.pr_step_back_line_window),
            ("--pr-evidence-stall-rounds", self.pr_evidence_stall_rounds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AgentLoopError(
                    f"{flag} must be a non-negative integer"
                    + (
                        " (0 disables it)."
                        if flag in {"--pr-step-back-rounds", "--pr-evidence-stall-rounds"}
                        else "."
                    )
                )
        if (
            isinstance(self.sub_item_stall_rounds, bool)
            or not isinstance(self.sub_item_stall_rounds, int)
            or self.sub_item_stall_rounds < 0
        ):
            raise AgentLoopError(
                "--sub-item-stall-rounds must be zero or greater (0 disables it)."
            )
        if self.plan_growth_gate not in {"enforce", "off"}:
            raise AgentLoopError("--plan-growth-gate must be 'enforce' or 'off'.")
        for flag, value in (
            ("--plan-growth-max-chars", self.plan_growth_max_chars),
            ("--plan-growth-max-revisions", self.plan_growth_max_revisions),
            ("--plan-growth-max-scope-items", self.plan_growth_max_scope_items),
            ("--plan-growth-max-matrix-rows", self.plan_growth_max_matrix_rows),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise AgentLoopError(f"{flag} must be a positive integer.")
        object.__setattr__(self, "pr_review_broad_rules", normalize_broad_rules(self.pr_review_broad_rules))
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
        # Validate containment even for library-created configs.  This is a
        # pure local parse; capability probing is deferred until a command is
        # actually admitted by Runner.
        from .containment import policy_from_values

        policy_from_values(self.__dict__)
        from .test_workers import WorkerBudgetError, budget_from_values

        try:
            budget_from_values(self.__dict__)
        except WorkerBudgetError as exc:
            raise AgentLoopError(str(exc)) from exc

    @property
    def containment_policy(self):
        """Return the immutable resolved containment policy for this config."""
        from .containment import policy_from_values

        return policy_from_values(self.__dict__)


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
    resolved = _resolve_base_branch(config, runner, pr_metadata=pr_metadata, cwd=cwd)
    # Startup gate (#1285): a non-default managed-CI base must be allow-listed
    # by AGENT_LOOP_TRUSTED_BASES before any agent work or workdir setup.
    from .managed_ci import enforce_trusted_base_at_startup

    enforce_trusted_base_at_startup(runner, resolved)
    return resolved


def _resolve_base_branch(
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
    """Reject two agents of one run sharing a checkout.

    ``--allow-shared-dir`` is intra-run only: it never lets a second run use a
    checkout claimed by another live run (#1127).
    """
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


def _args_have_claude_effort(args: tuple[str, ...]) -> bool:
    return any(a == "--effort" or a.startswith("--effort=") for a in args)


def _validate_effort_value(provider: AgentName | str, value: str, option: str) -> None:
    if not value:
        return
    allowed = CODEX_REASONING_EFFORTS if provider == "codex" else CLAUDE_EFFORTS
    if value not in allowed:
        rendered = ", ".join(sorted(allowed))
        raise AgentLoopError(
            f"{option} value {value!r} is unsupported for {provider}; "
            f"choose one of: {rendered}."
        )


def configured_model_for(
    config: AgentLoopConfig, provider: AgentName, *, role: str | None = None
) -> str | None:
    if role == "reviewer":
        if provider == "codex" and config.reviewer_codex_model:
            return config.reviewer_codex_model
        if provider == "claude" and config.reviewer_claude_model:
            return config.reviewer_claude_model
    if provider == "claude":
        return config.claude_model or None
    if provider == "codex":
        return config.codex_model or None
    if provider == "gemini":
        return config.gemini_model or None
    if provider == "antigravity":
        return config.antigravity_models[0] if config.antigravity_models else None
    return None


def resolve_invocation(
    config: AgentLoopConfig,
    *,
    provider: AgentName | None = None,
    role: str | None = None,
    implementation: bool = False,
) -> ResolvedInvocation:
    """Resolve provider effort without allowing omitted values to mask overrides."""
    executing_provider = provider or (
        config.implementation_coder if implementation and config.implementation_coder else config.coder
    )
    role_override = ""
    agent_effort = ""
    implementation_role_active = implementation or (
        config.implementation_effort_active and role == "coder"
    )
    if executing_provider == "codex":
        role_override = (
            config.implementation_codex_reasoning_effort if implementation_role_active else ""
        )
        agent_effort = config.codex_reasoning_effort
    elif executing_provider == "claude":
        role_override = (
            config.implementation_claude_effort if implementation_role_active else ""
        )
        agent_effort = config.claude_effort
    if role == "reviewer":
        if executing_provider == "codex":
            role_override = config.reviewer_codex_reasoning_effort
        elif executing_provider == "claude":
            role_override = config.reviewer_claude_effort
    if executing_provider in {"codex", "claude"}:
        if role_override:
            effort, source = role_override, "role_override"
        elif agent_effort:
            effort, source = agent_effort, "agent_wide"
        else:
            effort, source = DEFAULT_REASONING_EFFORT, "tool_default"
    else:
        effort, source = None, None
    return ResolvedInvocation(
        provider=executing_provider,
        role=role,
        configured_model=configured_model_for(config, executing_provider, role=role),
        resolved_effort=effort,
        effort_source=source,
    )


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
    if (config.codex_model or config.reviewer_codex_model) and _args_have_model_flag(config.codex_args):
        raise AgentLoopError(
            "--codex-arg --model conflicts with --codex-model/--reviewer-codex-model; use dedicated model options only."
        )
    if _args_have_reasoning_effort(config.codex_args):
        option = (
            "--implementation-codex-reasoning-effort"
            if config.implementation_coder == "codex"
            and config.implementation_codex_reasoning_effort
            else "--codex-reasoning-effort"
        )
        raise AgentLoopError(
            "--codex-arg model_reasoning_effort conflicts with agent-loop effort ownership; "
            f"use {option} only."
        )
    if config.gemini_model and _args_have_model_flag(config.gemini_args):
        raise AgentLoopError(
            "--gemini-arg --model conflicts with --gemini-model; use --gemini-model only."
        )
    if (config.claude_model or config.reviewer_claude_model) and _args_have_model_flag(config.claude_args):
        raise AgentLoopError(
            "--claude-arg --model conflicts with --claude-model/--reviewer-claude-model; use dedicated model options only."
        )
    if _args_have_claude_effort(config.claude_args):
        option = (
            "--implementation-claude-effort"
            if config.implementation_coder == "claude"
            and config.implementation_claude_effort
            else "--claude-effort"
        )
        raise AgentLoopError(
            "--claude-arg --effort conflicts with agent-loop effort ownership; "
            f"use {option} only."
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


def default_agent_workdir(repo: str, agent: AgentName) -> Path:
    repo_slug = repo_cache_slug(repo)
    return scratch_root() / repo_slug / agent / "repo"


def default_run_worktree_root(repo: str, agent: AgentName) -> Path:
    return scratch_root() / repo_cache_slug(repo) / agent / "runs"


def default_run_worktree(repo: str, agent: AgentName, run_token: str) -> Path:
    return default_run_worktree_root(repo, agent) / run_token


_WORKTREE_LINK_FORBIDDEN = set("*?[]!#\\:\0\n\r\t")


def normalize_worktree_link(value: str) -> str:
    """Validate one ``--worktree-link`` value; the accepted string is its only spelling."""
    text = str(value)
    bad = f"Invalid --worktree-link {text!r}: "
    if not text or text != text.strip():
        raise AgentLoopError(bad + "must be a non-empty path without surrounding whitespace.")
    if any(ch in _WORKTREE_LINK_FORBIDDEN for ch in text):
        raise AgentLoopError(bad + "contains a character that is not allowed in a link path.")
    for part in text.split("/"):
        if not part:
            raise AgentLoopError(
                bad + "empty path components (leading, trailing or repeated '/') are not allowed."
            )
        if part in (".", "..", ".git"):
            raise AgentLoopError(bad + f"component {part!r} is not allowed.")
    return text


def new_run_token() -> str:
    return f"{datetime_stamp()}-{uuid.uuid4().hex[:12]}"


def default_run_artifacts_dir(repo: str, run_token: str) -> Path:
    return default_cache_root() / "run-artifacts" / repo_cache_slug(repo) / run_token


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


def _reject_inside_roots(path: Path, roots: tuple[Path, ...], option: str) -> None:
    if any(path == root or root in path.parents for root in roots):
        raise AgentLoopError(
            f"{option} must not be equal to or nested beneath a managed checkout store or "
            f"per-run worktree root: {path}"
        )


def _resolve_log_dir(
    value: Path, *, repo: str, primary_dir: Path, run_token: str | None, store_roots: tuple[Path, ...]
) -> Path:
    """Relative log dirs of a per-run worktree primary live in a durable artifact root."""
    if value.is_absolute():
        path = value
    elif run_token is not None:
        path = default_run_artifacts_dir(repo, run_token) / value
    else:
        return primary_dir / value
    if store_roots:
        # Resolve aliases first: a symlink into a runs/ root must not slip through.
        _reject_inside_roots(Path(path).expanduser().resolve(), store_roots, "--log-dir")
    return path


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
    # A killed run's verifiable GEMINI.md injection is tool-owned, not dirt.
    recover_gemini_injection(config, runner, path)
    links = registered_links(path)
    if links:
        _clean_or_reject_linked_checkout(path, links, label=label, config=config, runner=runner)
        return
    status = _run_git(runner, path, ("status", "--porcelain")).stdout.strip()
    if not status:
        return
    if not default_owned:
        raise AgentLoopError(f"{label} is dirty: {path}. Commit, stash, or clean it before rerunning.")
    log(config, f"Cleaning dirty {label[0].lower()}{label[1:]}: {path}")
    _run_git(runner, path, ("reset", "--hard"))
    _run_git(runner, path, ("clean", "-fd"))


def _clean_or_reject_linked_checkout(
    path: Path, links: dict[bytes, bytes], *, label: str, config: AgentLoopConfig, runner: Runner
) -> None:
    """Clean a worktree that carries registered links; the links are never touched."""
    check_worktree_links(config, runner, path)
    raw = _run_git(runner, path, ("status", "--porcelain", "-z", "--untracked-files=all")).stdout
    names = {os.fsdecode(link) for link in links}
    remaining = [r for r in raw.split("\0") if r and r not in {f"?? {n}" for n in names}]
    if not remaining:
        return
    log(config, f"Cleaning dirty {label[0].lower()}{label[1:]}: {path}")
    _run_git(runner, path, ("reset", "--hard"))
    excludes: list[str] = []
    for name in sorted(names):
        excludes += ["-e", f"/{name}"]
    _run_git(runner, path, ("clean", "-fd", *excludes))
    check_worktree_links(config, runner, path)


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
    store = run_worktree_store(config, path)
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

    # Verify an existing baseline BEFORE anything can reset or clean (#1130).
    verify_before_sync(config, runner, path=path, label=label)
    _validate_repo_remote(path, label=label, config=config, runner=runner)
    _clean_or_reject_checkout(
        path,
        label=label,
        default_owned=default_owned,
        config=config,
        runner=runner,
    )

    if store is not None:
        # Per-run worktree (#1162): stay detached at a SHA pinned under the store lock.
        from . import run_worktrees

        sha = run_worktrees.prepare_store(store, config=config, runner=runner)
        _run_git(runner, path, ("checkout", "--detach", sha))
        refresh_after_sync(config, runner, path)
        return
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
    refresh_after_sync(config, runner, path)


def run_worktree_store(config: AgentLoopConfig, path: Path) -> Path | None:
    """Return the shared store when ``path`` is one of this run's per-run worktrees."""
    if not config.run_token:
        return None
    for agent, store in config.default_checkout_stores:
        expected = default_run_worktree(config.repo, agent, config.run_token)
        if os.path.abspath(expected) == os.path.abspath(path) or expected.resolve() == Path(path).resolve():
            return store
    return None


def _ensure_run_worktree(
    path: Path, *, agent: AgentName, store: Path, config: AgentLoopConfig, runner: Runner
) -> None:
    from . import run_worktrees
    from .workdir_claims import register_run_finalizer

    label = f"Default {agent} workdir"
    runs_root = path.parent
    if runner.dry_run:
        if not store.exists():
            runner.run((config.gh_cmd, "repo", "clone", config.repo, str(store)), cwd=store.parent)
        runner.run(("git", "fetch", "origin"), cwd=store)
        runner.run(
            ("git", "worktree", "add", "--detach", str(path), f"origin/{config.base or 'HEAD'}"),
            cwd=store,
        )
        for link in config.worktree_links:
            log(config, f"[dry-run] Would link {store / link} -> {path / link}")
        return
    verify_before_sync(config, runner, path=path, label=label)
    sha = run_worktrees.prepare_store(store, config=config, runner=runner)
    run_worktrees.prune_dead_worktrees(store, runs_root, path, config=config, runner=runner)
    with run_worktrees.store_lock(store):
        if config.worktree_links:
            run_worktrees.check_link_sources(store, config.worktree_links)
        if not os.path.lexists(path):
            run_worktrees.add_run_worktree(store, path, sha, config=config, runner=runner)
        elif run_worktrees.identify_owned_worktree(store, runs_root, path) is None:
            raise AgentLoopError(
                f"{label} at {path} exists but is not a worktree this run created; refusing to use it."
            )

    def finalize(store=store, path=path) -> None:
        run_worktrees.remove_run_worktree(store, path, config=config, runner=runner)

    register_run_finalizer(os.path.abspath(path), finalize)
    if config.worktree_links:
        mapping = run_worktrees.create_worktree_links(path, store, config.worktree_links)
        register_worktree_links(path, mapping)
    _sync_base_branch(path, label=label, default_owned=True, config=config, runner=runner)


def ensure_temp_checkout(path: Path, *, agent: AgentName, config: AgentLoopConfig, runner: Runner) -> None:
    store = run_worktree_store(config, path)
    if store is not None:
        _ensure_run_worktree(path, agent=agent, store=store, config=config, runner=runner)
        return
    # A path agent-loop already prepared in this process is verified BEFORE any
    # recreation branch (missing or stale), so a vanished, emptied or poisoned
    # checkout is refused instead of being re-cloned over its baseline.
    verify_before_sync(config, runner, path=path, label=f"Default {agent} workdir")
    if _is_stale_default_workdir(path):
        log(config, f"Stale default {agent} workdir detected (no checkout, only logs remain); recreating: {path}")
        shutil.rmtree(path)
        forget_checkout(path)  # verified above; the tool itself removed it

    if not path.exists():
        try:
            make_private_dirs(path.parent)
        except OSError as exc:
            raise AgentLoopError(f"Could not create parent directory for {agent} checkout at {path}: {exc}") from exc
        runner.run((config.gh_cmd, "repo", "clone", config.repo, str(path)), cwd=path.parent)
        if runner.dry_run:
            return
        from .agent_permissions import register_checkout

        # An intentional (re-)creation is re-registered so the sandboxed
        # boundary re-checks it rather than treating it as tampering.
        register_checkout(config, path)
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
    if runner.dry_run:
        return
    git_check = _run_git(runner, path, ("rev-parse", "--is-inside-work-tree"), check=False)
    if git_check.returncode != 0 or git_check.stdout.strip() != "true":
        raise AgentLoopError(
            f"{option_name} is not a git checkout: {path}. "
            f"Explicit agent directories must be existing git checkouts of {config.repo}."
        )

    verify_before_sync(config, runner, path=path, label=option_name)
    _clean_or_reject_checkout(
        path,
        label=option_name,
        default_owned=False,
        config=config,
        runner=runner,
    )
    _validate_repo_remote(path, label=option_name, config=config, runner=runner)
    refresh_after_sync(config, runner, path)


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

    verify_before_sync(config, runner, path=path, label=label)
    _validate_repo_remote(path, label=label, config=config, runner=runner)
    _clean_or_reject_checkout(
        path,
        label=label,
        default_owned=default_owned,
        config=config,
        runner=runner,
    )

    store = run_worktree_store(config, path)
    if store is not None:
        from . import run_worktrees

        # Check out the SHA pinned under the store lock, never the shared mutable ref.
        pinned = run_worktrees.pin_pr_head(store, pr_number, config=config, runner=runner)
        _run_git(runner, path, ("checkout", "--detach", pinned))
    else:
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
    refresh_after_sync(config, runner, path)


def ensure_agent_workdirs(config: AgentLoopConfig, runner: Runner) -> None:
    # Claim before any clone, rmtree, reset or clean (#1127).  Outside a run
    # scope this opens a "library" scope and releases on return.
    from .workdir_claims import claim_agent_workdirs, workdir_claim_scope

    with workdir_claim_scope():
        claim_agent_workdirs(config)
        _prepare_agent_workdirs(config, runner)


def _prepare_agent_workdirs(config: AgentLoopConfig, runner: Runner) -> None:
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


def _resolve_agent_memory_dir(
    value: Path | None,
    *,
    repo: str,
    primary_dir: Path,
    per_run_primary: bool = False,
    store_roots: tuple[Path, ...] = (),
) -> Path:
    if value is None:
        return default_agent_memory_dir(repo).resolve()
    if value.is_absolute():
        resolved = value.resolve()
    elif per_run_primary:
        # Repo-keyed, never beneath the removable per-run worktree.
        resolved = (default_cache_root() / "repos" / repo_cache_slug(repo) / value).resolve()
    else:
        return (primary_dir / value).resolve()
    if store_roots:
        _reject_inside_roots(resolved, store_roots, "--agent-memory-dir")
    return resolved


def _repair_backend_only_hint(backend: str, override_flag: str) -> str:
    return (
        f" It is needed only because the malformed-response repair backend is {backend} "
        "(--repair-backend; the default is antigravity, which requires the agy CLI even "
        "when no role uses Antigravity). Alternatively, pass "
        "--repair-backend codex --repair-model MODEL or "
        "--repair-backend claude --repair-model MODEL "
        f"to repair with a configured agent, or install the CLI / pass {override_flag} <path>."
    )


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
    role_agents = dict.fromkeys(
        (
            args.coder,
            *((implementation_coder,) if implementation_coder is not None else ()),
            *configured_reviewers,
            *((discuss_analyzer,) if discuss_analyzer is not None else ()),
        )
    )
    repair_backend = getattr(args, "repair_backend", "antigravity")
    configured_agents = dict.fromkeys((*role_agents, repair_backend))
    for agent in configured_agents:
        command, override_flag = command_options[agent]
        resolved = shutil.which(command)
        if resolved is None:
            repair_only = agent == repair_backend and agent not in role_agents
            if os.path.isabs(command):
                message = (
                    f"{command} not found or not executable; "
                    f"pass a valid executable path to {override_flag}."
                )
            else:
                message = (
                    f"{command} CLI not found on PATH; install it or pass "
                    f"{override_flag} <path>."
                )
            if repair_only:
                message += _repair_backend_only_hint(agent, override_flag)
            raise AgentLoopError(message)
        runner.remember_agent_command(command, resolved, override_flag)


def _arg_or_default(args: argparse.Namespace, name: str, default: int) -> int:
    """Keep an explicit value (even an invalid one) so validation can reject it."""
    value = getattr(args, name, None)
    return default if value is None else value


HUMAN_REVIEWER_TRUSTED_ACTOR_OPTION = "--human-reviewer-trusted-actor"


def parse_human_reviewer_trusted_actors(
    values: list[str] | tuple[str, ...] | None,
) -> tuple[TrustedHumanActor, ...]:
    """Parse repeated ``LOGIN:ID`` entries, rejecting any malformed entry (#1022)."""
    option = HUMAN_REVIEWER_TRUSTED_ACTOR_OPTION
    actors: list[TrustedHumanActor] = []
    login_by_id: dict[int, str] = {}
    for raw in values or ():
        entry = str(raw)
        if entry.count(":") != 1:
            # GitHub logins cannot contain ':', so exactly one separator is required.
            raise AgentLoopError(f"{option} {entry!r} must have the form LOGIN:ID.")
        login, _, raw_id = entry.partition(":")
        login = login.strip()
        raw_id = raw_id.strip()
        if not login or any(ch.isspace() for ch in login):
            raise AgentLoopError(f"{option} {entry!r} has a blank or malformed login.")
        if not raw_id.isascii() or not raw_id.isdigit():
            raise AgentLoopError(
                f"{option} {entry!r} must end in a positive numeric GitHub user ID."
            )
        user_id = int(raw_id)
        if user_id <= 0:
            raise AgentLoopError(
                f"{option} {entry!r} must end in a positive numeric GitHub user ID."
            )
        previous = login_by_id.get(user_id)
        if previous is not None:
            if previous.casefold() != login.casefold():
                raise AgentLoopError(
                    f"{option} names user ID {user_id} with conflicting logins "
                    f"{previous!r} and {login!r}."
                )
            continue
        login_by_id[user_id] = login
        actors.append(TrustedHumanActor(login=login, user_id=user_id))
    return tuple(actors)


def resolve_agent_permissions_mode(args: argparse.Namespace) -> str:
    """Resolve ``--agent-permissions`` and its ``--dangerous-agent-permissions`` alias."""
    explicit = getattr(args, "agent_permissions", None)
    dangerous_flag = bool(getattr(args, "dangerous_agent_permissions", False))
    if dangerous_flag and explicit not in (None, "dangerous"):
        raise AgentLoopError(
            f"--dangerous-agent-permissions conflicts with --agent-permissions {explicit}; "
            "pass only one permission mode."
        )
    mode = explicit or ("dangerous" if dangerous_flag else "default")
    if mode == "sandboxed":
        from .agent_permissions import validate_sandboxed_passthrough

        validate_sandboxed_passthrough(
            {
                "--claude-arg": getattr(args, "claude_arg", None),
                "--codex-arg": getattr(args, "codex_arg", None),
                "--gemini-arg": getattr(args, "gemini_arg", None),
                "--antigravity-arg": getattr(args, "antigravity_arg", None),
            }
        )
    return mode


def parse_antigravity_quota_group_overrides(values) -> tuple[tuple[str, str], ...]:
    """Parse repeated ``MODEL=GROUP`` flag values (shape only; #1236)."""
    parsed: list[tuple[str, str]] = []
    for value in values:
        model, sep, group = str(value).rpartition("=")
        if not sep or not model.strip() or not group.strip():
            raise AgentLoopError(
                f"--antigravity-quota-group must be MODEL=GROUP with non-blank parts, got {value!r}."
            )
        parsed.append((model.strip(), group.strip()))
    return tuple(parsed)


def config_from_args(
    args: argparse.Namespace,
    runner: Runner,
    *,
    invocation_argv: tuple[str, ...] = (),
) -> AgentLoopConfig:
    from .reviewer_seats import resolve_reviewer_seats

    if resolve_reviewer_seats(args):
        raise AgentLoopError(
            "Named reviewer seats are validated but review execution is unavailable in phase 1; "
            "use legacy --reviewer until durable seat identity is enabled."
        )
    configured_reviewers = tuple(args.reviewer or ["codex"])
    if len(set(configured_reviewers)) != len(configured_reviewers):
        raise AgentLoopError("--reviewer cannot include the same agent more than once.")
    if resolve_agent_permissions_mode(args) == "sandboxed":
        # Name the unsupported selection before command preflight would
        # report a missing default helper CLI such as agy.
        from types import SimpleNamespace

        from .agent_permissions import validate_sandboxed_selections

        validate_sandboxed_selections(
            SimpleNamespace(
                coder=args.coder,
                implementation_coder=getattr(args, "implementation_coder", None),
                reviewer=configured_reviewers,
                primary_reviewer=getattr(args, "primary_reviewer", None),
                primary_plan_reviewer=getattr(args, "primary_plan_reviewer", None),
                discuss_analyzer=getattr(args, "discuss_analyzer", None),
                repair_backend=getattr(args, "repair_backend", "antigravity"),
                semantic_followup_dedupe=getattr(args, "semantic_followup_dedupe", True),
                semantic_followup_backend=getattr(
                    args, "semantic_followup_backend", DEFAULT_SEMANTIC_FOLLOWUP_BACKEND
                ),
            )
        )
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
    run_token = new_run_token()
    store_by_agent: dict[AgentName, Path] = {}
    worktree_links = tuple(
        dict.fromkeys(
            normalize_worktree_link(value) for value in (getattr(args, "worktree_link", None) or ())
        )
    )

    def resolve_agent_dir(agent: AgentName, value: Path | None) -> Path:
        if value is not None:
            return value.resolve()
        store_by_agent[agent] = default_agent_workdir(repo, agent).resolve()
        return default_run_worktree(repo, agent, run_token).resolve()

    claude_dir = resolve_agent_dir("claude", args.claude_dir)
    codex_dir = resolve_agent_dir("codex", args.codex_dir)
    gemini_dir = resolve_agent_dir("gemini", args.gemini_dir)
    antigravity_dir = resolve_agent_dir("antigravity", args.antigravity_dir)
    default_checkout_stores = tuple(store_by_agent.items())
    active_roles = {args.coder, *configured_reviewers}
    for extra_role in (getattr(args, "implementation_coder", None), getattr(args, "discuss_analyzer", None)):
        if extra_role is not None:
            active_roles.add(extra_role)
    if worktree_links and not (active_roles & set(store_by_agent)):
        log_message = "--worktree-link is ignored: no active agent uses a default per-run worktree."
        print(log_message, file=sys.stderr)
        worktree_links = ()
    store_roots = tuple(
        root
        for agent, store in default_checkout_stores
        for root in (store, default_run_worktree_root(repo, agent).resolve())
    )
    primary_dir = {
        "claude": claude_dir,
        "codex": codex_dir,
        "gemini": gemini_dir,
        "antigravity": antigravity_dir,
    }[args.coder]
    per_run_primary = args.coder in store_by_agent
    test_command = _split_command(args.test_command)
    if args.max_rounds <= 0:
        raise AgentLoopError("--max-rounds must be greater than zero.")
    if getattr(args, "sub_item_stall_rounds", DEFAULT_SUB_ITEM_STALL_ROUNDS) < 0:
        raise AgentLoopError("--sub-item-stall-rounds must be zero or greater (0 disables it).")
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
    permission_mode = resolve_agent_permissions_mode(args)
    dangerous = permission_mode == "dangerous"

    def static_args(agent: AgentName, supplied: list[str] | None) -> tuple[str, ...]:
        if supplied is not None:
            return tuple(supplied)
        return default_agent_args(agent, dangerous=dangerous, mode=permission_mode)

    quota_group_overrides = parse_antigravity_quota_group_overrides(
        getattr(args, "antigravity_quota_group", None) or ()
    )
    if quota_group_overrides:
        if args.antigravity_model is not None:
            resolved_chain: tuple[str, ...] = (args.antigravity_model,)
        elif getattr(args, "antigravity_models", None):
            resolved_chain = tuple(args.antigravity_models)
        else:
            resolved_chain = DEFAULT_ANTIGRAVITY_MODELS
        for override_model, _group in quota_group_overrides:
            if override_model not in resolved_chain:
                raise AgentLoopError(
                    f"--antigravity-quota-group names {override_model!r}, which is not in the "
                    "configured Antigravity chain."
                )

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
        claude_args=static_args("claude", args.claude_arg),
        codex_args=static_args("codex", args.codex_arg),
        gemini_args=static_args("gemini", args.gemini_arg),
        antigravity_dir=antigravity_dir,
        antigravity_cmd=args.antigravity_cmd,
        antigravity_args=static_args("antigravity", args.antigravity_arg),
        agent_permissions=permission_mode,
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
        antigravity_quota_cooldown_seconds=getattr(
            args,
            "antigravity_quota_cooldown_seconds",
            DEFAULT_ANTIGRAVITY_QUOTA_COOLDOWN_SECONDS,
        ),
        antigravity_quota_groups=quota_group_overrides,
        codex_model=getattr(args, "codex_model", ""),
        codex_reasoning_effort=getattr(args, "codex_reasoning_effort", ""),
        reviewer_codex_model=getattr(args, "reviewer_codex_model", ""),
        reviewer_codex_reasoning_effort=getattr(args, "reviewer_codex_reasoning_effort", ""),
        reviewer_claude_model=getattr(args, "reviewer_claude_model", ""),
        reviewer_claude_effort=getattr(args, "reviewer_claude_effort", ""),
        gemini_model=getattr(args, "gemini_model", ""),
        claude_model=getattr(args, "claude_model", ""),
        claude_effort=getattr(args, "claude_effort", ""),
        implementation_coder=getattr(args, "implementation_coder", None),
        implementation_coder_model=getattr(args, "implementation_coder_model", ""),
        implementation_codex_reasoning_effort=getattr(args, "implementation_codex_reasoning_effort", ""),
        implementation_claude_effort=getattr(args, "implementation_claude_effort", ""),
        repair_backend=getattr(args, "repair_backend", "antigravity"),
        repair_models=tuple(getattr(args, "repair_model", None) or DEFAULT_REPAIR_MODELS),
        repair_reasoning_effort=getattr(args, "repair_reasoning_effort", ""),
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
        coder_test_command_timeout_seconds=getattr(
            args,
            "coder_test_command_timeout_seconds",
            DEFAULT_TEST_TIMEOUT_SECONDS,
        ),
        test_python=getattr(args, "test_python", None),
        pre_review_tests=args.pre_review_tests,
        ci_timeout_seconds=args.ci_timeout_seconds,
        ci_poll_interval_seconds=args.ci_poll_interval_seconds,
        ci_startup_timeout_seconds=getattr(args, "ci_startup_timeout_seconds", 120),
        watch_pending_ci=(
            bool(getattr(args, "auto_merge", False))
            if getattr(args, "watch_pending_ci", None) is None
            else bool(args.watch_pending_ci)
        ),
        watch_pending_ci_explicit=getattr(args, "watch_pending_ci", None) is not None,
        managed_ci_trusted_actor=getattr(args, "managed_ci_trusted_actor", None),
        human_reviewer_trusted_actors=parse_human_reviewer_trusted_actors(
            getattr(args, "human_reviewer_trusted_actor", None)
        ),
        managed_ci=getattr(args, "managed_ci", False),
        managed_ci_adopt_existing_pr=getattr(args, "managed_ci_adopt_existing_pr", False),
        allow_unprotected_managed_ci=getattr(args, "allow_unprotected_managed_ci", False),
        allow_unreadable_protection=getattr(args, "allow_unreadable_protection", False),
        review_unrecorded_head=bool(getattr(args, "review_unrecorded_head", False)),
        managed_ci_fresh_authorization=getattr(args, "managed_ci_fresh_authorization", False),
        managed_ci_issue_number=getattr(args, "managed_ci_issue", None),
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
        log_dir=_resolve_log_dir(
            args.log_dir,
            repo=repo,
            primary_dir=primary_dir,
            run_token=run_token if per_run_primary else None,
            store_roots=store_roots,
        ),
        subprocess_log_dir=_resolve_subprocess_log_dir(
            args,
            repo=repo,
            primary_dir=primary_dir,
            managed_roots=(claude_dir, codex_dir, gemini_dir, antigravity_dir, *store_roots),
        ),
        progress_interval_seconds=args.progress_interval_seconds,
        agent_max_retries=args.agent_max_retries,
        agent_retry_backoff_seconds=tuple(args.agent_retry_backoff_seconds),
        agent_memory=args.agent_memory,
        refresh_agent_memory=args.refresh_agent_memory,
        agent_memory_dir=_resolve_agent_memory_dir(
            args.agent_memory_dir,
            repo=repo,
            primary_dir=primary_dir,
            per_run_primary=per_run_primary,
            store_roots=store_roots,
        ),
        refresh_test_profile=args.refresh_test_profile,
        approved_followups=args.approved_followups,
        semantic_followup_dedupe=getattr(args, "semantic_followup_dedupe", True),
        semantic_followup_backend=getattr(
            args, "semantic_followup_backend", DEFAULT_SEMANTIC_FOLLOWUP_BACKEND
        ),
        semantic_followup_model=getattr(args, "semantic_followup_model", ""),
        semantic_followup_timeout_seconds=getattr(
            args, "semantic_followup_timeout_seconds", DEFAULT_SEMANTIC_FOLLOWUP_TIMEOUT_SECONDS
        ),
        semantic_followup_max_calls=getattr(
            args, "semantic_followup_max_calls", DEFAULT_SEMANTIC_FOLLOWUP_MAX_CALLS
        ),
        semantic_followup_max_candidates=getattr(
            args, "semantic_followup_max_candidates", DEFAULT_SEMANTIC_FOLLOWUP_MAX_CANDIDATES
        ),
        semantic_followup_prompt_char_limit=getattr(
            args, "semantic_followup_prompt_char_limit", DEFAULT_SEMANTIC_FOLLOWUP_PROMPT_CHAR_LIMIT
        ),
        plan_execution_mode=getattr(args, "plan_execution_mode", None) or "plan-only",
        execution_strategy_contract_required=True,
        planning_context_mode=getattr(args, "planning_context_mode", None) or "compact",
        pr_review_context_mode=getattr(args, "pr_review_context_mode", None) or "full",
        architecture_context_enabled=getattr(args, "architecture_context_enabled", True),
        architecture_path=getattr(args, "architecture_path", "ARCHITECTURE.md"),
        architecture_read_size=getattr(args, "architecture_read_size", 64 * 1024),
        architecture_snapshot_max_chars=getattr(args, "architecture_snapshot_max_chars", 12_000),
        architecture_aggregate_max_chars=getattr(args, "architecture_aggregate_max_chars", 24_000),
        managed_context_max_chars=getattr(args, "managed_context_max_chars", 80_000),
        pr_review_policy=getattr(args, "pr_review_policy", None) or "all-reviewers",
        primary_reviewer=getattr(args, "primary_reviewer", None),
        pr_review_broad_rules=tuple(
            getattr(args, "pr_review_broad_rules", None)
            if getattr(args, "pr_review_broad_rules", None) is not None
            else DEFAULT_BROAD_RULES
        ),
        pr_review_force_full=bool(getattr(args, "pr_review_force_full", False)),
        plan_review_policy=getattr(args, "plan_review_policy", None) or "all-reviewers",
        primary_plan_reviewer=getattr(args, "primary_plan_reviewer", None),
        plan_review_force_full=bool(getattr(args, "plan_review_force_full", False)),
        plan_primary_stall_rounds=_arg_or_default(
            args, "plan_primary_stall_rounds", DEFAULT_PLAN_PRIMARY_STALL_ROUNDS
        ),
        plan_step_back_rounds=_arg_or_default(
            args, "plan_step_back_rounds", DEFAULT_PLAN_STEP_BACK_ROUNDS
        ),
        plan_step_back_escalation_rounds=_arg_or_default(
            args, "plan_step_back_escalation_rounds", DEFAULT_PLAN_STEP_BACK_ESCALATION_ROUNDS
        ),
        pr_step_back_rounds=_arg_or_default(
            args, "pr_step_back_rounds", DEFAULT_PR_STEP_BACK_ROUNDS
        ),
        pr_step_back_line_window=_arg_or_default(
            args, "pr_step_back_line_window", DEFAULT_PR_STEP_BACK_LINE_WINDOW
        ),
        pr_evidence_stall_rounds=_arg_or_default(
            args, "pr_evidence_stall_rounds", DEFAULT_PR_EVIDENCE_STALL_ROUNDS
        ),
        plan_reset_stall_streak=bool(getattr(args, "plan_reset_stall_streak", False)),
        plan_narrow_staged=bool(getattr(args, "plan_narrow_staged", False)),
        plan_growth_gate=getattr(args, "plan_growth_gate", None) or "enforce",
        plan_growth_max_chars=_arg_or_default(args, "plan_growth_max_chars", 120_000),
        plan_growth_max_revisions=_arg_or_default(args, "plan_growth_max_revisions", 6),
        plan_growth_max_scope_items=_arg_or_default(args, "plan_growth_max_scope_items", 12),
        plan_growth_max_matrix_rows=_arg_or_default(args, "plan_growth_max_matrix_rows", 18),
        sub_item_stall_rounds=_arg_or_default(
            args, "sub_item_stall_rounds", DEFAULT_SUB_ITEM_STALL_ROUNDS
        ),
        auto_agent_dirs=auto_agent_dirs,
        default_checkout_stores=default_checkout_stores,
        run_token=run_token if default_checkout_stores else None,
        worktree_links=worktree_links,
        containment_mode=getattr(args, "containment_mode", "auto"),
        containment_memory_high=getattr(args, "containment_memory_high", None),
        containment_memory_max=getattr(args, "containment_memory_max", None),
        containment_memory_swap_max=getattr(args, "containment_memory_swap_max", None),
        containment_tasks_max=getattr(args, "containment_tasks_max", None),
        containment_aggregate_memory_high=getattr(args, "containment_aggregate_memory_high", None),
        containment_aggregate_memory_max=getattr(args, "containment_aggregate_memory_max", None),
        containment_aggregate_memory_swap_max=getattr(args, "containment_aggregate_memory_swap_max", None),
        containment_aggregate_tasks_max=getattr(args, "containment_aggregate_tasks_max", None),
        containment_os_headroom_percent=getattr(args, "containment_os_headroom_percent", 25.0),
        containment_slice=getattr(args, "containment_slice", "agent-loop.slice"),
        containment_cache_dir=getattr(args, "containment_cache_dir", None),
        containment_coder_memory_high=getattr(args, "containment_coder_memory_high", None),
        containment_coder_memory_max=getattr(args, "containment_coder_memory_max", None),
        containment_coder_memory_swap_max=getattr(args, "containment_coder_memory_swap_max", None),
        containment_coder_tasks_max=getattr(args, "containment_coder_tasks_max", None),
        containment_reviewer_memory_high=getattr(args, "containment_reviewer_memory_high", None),
        containment_reviewer_memory_max=getattr(args, "containment_reviewer_memory_max", None),
        containment_reviewer_memory_swap_max=getattr(args, "containment_reviewer_memory_swap_max", None),
        containment_reviewer_tasks_max=getattr(args, "containment_reviewer_tasks_max", None),
        containment_repair_memory_high=getattr(args, "containment_repair_memory_high", None),
        containment_repair_memory_max=getattr(args, "containment_repair_memory_max", None),
        containment_repair_memory_swap_max=getattr(args, "containment_repair_memory_swap_max", None),
        containment_repair_tasks_max=getattr(args, "containment_repair_tasks_max", None),
        containment_test_gate_memory_high=getattr(args, "containment_test_gate_memory_high", None),
        containment_test_gate_memory_max=getattr(args, "containment_test_gate_memory_max", None),
        containment_test_gate_memory_swap_max=getattr(args, "containment_test_gate_memory_swap_max", None),
        containment_test_gate_tasks_max=getattr(args, "containment_test_gate_tasks_max", None),
        test_workers=getattr(args, "test_workers", None),
        test_worker_memory=getattr(args, "test_worker_memory", None),
        test_worker_enforcement=getattr(args, "test_worker_enforcement", None) or "clamp",
    )
