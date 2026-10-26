"""Pure managed-CI recovery command rendering."""

from __future__ import annotations

import shlex
from typing import Literal

from .config import AgentLoopConfig


_RECOVERY_VALUE_OPTIONS = frozenset({
    "--repo", "--base", "--trusted-local-origin", "--trusted-origin-protocol", "--git-ssh-identity",
    "--claude-dir", "--codex-dir", "--gemini-dir", "--antigravity-dir",
    "--architecture-path", "--architecture-detail-dir", "--architecture-exclude",
    "--architecture-read-size", "--architecture-snapshot-max-chars",
    "--architecture-aggregate-max-chars", "--managed-context-max-chars",
    "--coder", "--reviewer", "--reviewer-seat", "--seat-model", "--seat-effort", "--seat-dir",
    "--primary-reviewer-seat", "--primary-plan-reviewer-seat",
    "--min-reviewers", "--min-distinct-providers",
    "--max-rounds", "--sub-item-stall-rounds", "--managed-ci-trusted-actor", "--managed-ci-issue",
    "--human-reviewer-trusted-actor",
    "--managed-ci-issue-number",
    "--implementation-coder", "--implementation-coder-model",
    "--implementation-codex-reasoning-effort", "--implementation-claude-effort",
    "--triage-model", "--triage-history",
    "--claude-effort", "--claude-cmd", "--codex-cmd",
    "--gemini-cmd", "--antigravity-cmd", "--antigravity-print-timeout-seconds",
    "--repair-backend", "--repair-model", "--repair-timeout-seconds", "--repair-reasoning-effort",
    "--antigravity-model", "--antigravity-models", "--antigravity-quota-signatures",
    "--antigravity-quota-cooldown-seconds", "--antigravity-quota-group",
    "--codex-model", "--codex-reasoning-effort", "--gemini-model", "--claude-model",
    "--reviewer-codex-model", "--reviewer-codex-reasoning-effort",
    "--reviewer-claude-model", "--reviewer-claude-effort",
    "--gh-cmd", "--agent-permissions",
    "--claude-arg", "--codex-arg", "--gemini-arg", "--antigravity-arg",
    "--test-command", "--test-python", "--coder-test-command-timeout-seconds", "--ci-timeout-seconds",
    "--ci-poll-interval-seconds", "--ci-startup-timeout-seconds",
    "--ci-queued-grace-seconds", "--mergeability-poll-attempts",
    "--mergeability-poll-interval-seconds", "--log-dir", "--subprocess-log-dir",
    "--subprocess-log-retention-days",
    "--progress-interval-seconds", "--agent-max-retries", "--agent-retry-backoff-seconds",
    "--wait-for-quota-reset",
    "--agent-memory-dir", "--salvage-comment-patch-max-bytes", "--planning-context-mode",
    "--pr-review-context-mode", "--pr-review-policy", "--pr-review-broad-rule",
    "--expected-closing-issue",
    "--plan-execution-mode", "--flat-child-limit", "--split-stage", "--head", "--title", "--body-file",
    "--approved-followups", "--containment-mode", "--containment-memory-high",
    "--semantic-followup-backend", "--semantic-followup-model",
    "--semantic-followup-timeout-seconds", "--semantic-followup-max-calls",
    "--semantic-followup-max-candidates", "--semantic-followup-prompt-char-limit",
    "--containment-memory-max", "--containment-memory-swap-max", "--containment-tasks-max",
    "--primary-reviewer",
    "--plan-review-policy", "--primary-plan-reviewer", "--plan-primary-stall-rounds",
    "--plan-step-back-rounds", "--plan-step-back-escalation-rounds",
    "--plan-sweep-rounds", "--plan-sweep-reasoning-effort",
    "--plan-provenance-baseline-max-chars",
    "--pr-step-back-rounds", "--pr-step-back-line-window", "--pr-evidence-stall-rounds",
    "--pr-evidence-capture-attempts", "--pr-convert-evidence-rows",
    "--pr-non-convergence-rounds",
    "--plan-growth-gate", "--plan-growth-max-chars", "--plan-growth-max-revisions",
    "--plan-growth-max-scope-items", "--plan-growth-max-matrix-rows",
    "--containment-aggregate-memory-high", "--containment-aggregate-memory-max",
    "--containment-aggregate-memory-swap-max", "--containment-aggregate-tasks-max",
    "--containment-os-headroom-percent", "--containment-slice", "--containment-cache-dir",
    "--containment-coder-memory-high", "--containment-coder-memory-max",
    "--containment-coder-memory-swap-max", "--containment-coder-tasks-max",
    "--containment-reviewer-memory-high", "--containment-reviewer-memory-max",
    "--containment-reviewer-memory-swap-max", "--containment-reviewer-tasks-max",
    "--containment-repair-memory-high", "--containment-repair-memory-max",
    "--containment-repair-memory-swap-max", "--containment-repair-tasks-max",
    "--containment-test-gate-memory-high", "--containment-test-gate-memory-max",
    "--containment-test-gate-memory-swap-max", "--containment-test-gate-tasks-max",
    "--test-workers", "--test-worker-memory", "--test-worker-enforcement",
    "--worktree-link", "--worktree-setup", "--worktree-setup-key", "--worktree-setup-timeout",
    "--ci-plumbing-check",
})
_RECOVERY_NARGS_VALUE_OPTIONS = frozenset({
    "--antigravity-models", "--antigravity-quota-signatures", "--agent-retry-backoff-seconds",
})
_ISSUE_ONLY_RECOVERY_OPTIONS = frozenset({
    "--plan-first", "--implement-after-approval", "--plan-execution-mode",
    "--materialize-split-issues", "--split-stage", "--implementation-coder",
    "--implementation-coder-model", "--implementation-codex-reasoning-effort",
    "--implementation-claude-effort",
})
_PR_ONLY_RECOVERY_OPTIONS = frozenset({"--managed-ci-adopt-existing-pr"})
_MANAGED_PR_ONLY_RECOVERY_OPTIONS = frozenset({"--head", "--title", "--body-file"})
# One-shot operator flags a rendered recovery/rerun command must never replay.
_ONE_SHOT_OPERATOR_RECOVERY_OPTIONS = frozenset(
    {
        "--plan-reset-stall-streak", "--plan-resume-keep-on-issue-edit", "--pr-convert-evidence-rows",
        # Advisory triage runs once, before a fresh run's first agent turn (#1434).
        "--triage", "--triage-model", "--triage-history",
    }
)
_MANAGED_RECOVERY_OPTIONS = frozenset({
    "--managed-ci", "--managed-ci-trusted-actor", "--allow-unprotected-managed-ci",
    "--allow-unreadable-protection", "--managed-ci-adopt-existing-pr", "--managed-ci-fresh", "--managed-ci-fresh-authorization",
})


def _option_name(token: str) -> str:
    return token.split("=", 1)[0]


def _recovery_option_end(argv: list[str], index: int) -> int:
    """Return the first token after one option and its value payload."""
    token = argv[index]
    name = _option_name(token)
    if "=" in token or name not in _RECOVERY_VALUE_OPTIONS:
        return index + 1
    if name in _RECOVERY_NARGS_VALUE_OPTIONS:
        index += 1
        while index < len(argv) and (
            not argv[index].startswith("-") or argv[index] == "-"
        ):
            index += 1
        return index
    if index + 1 < len(argv) and (
        not argv[index + 1].startswith("-") or argv[index + 1] == "-"
    ):
        return index + 2
    return index + 1


def _find_recovery_positional_index(argv: list[str], command_index: int) -> int | None:
    """Find an issue/PR identifier even when options precede it."""
    index = command_index + 1
    while index < len(argv):
        if argv[index].startswith("-"):
            index = _recovery_option_end(argv, index)
            continue
        try:
            int(argv[index])
        except ValueError:
            return None
        return index
    return None


def _strip_recovery_options(
    argv: list[str], *, names: frozenset[str]
) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        name = _option_name(token)
        if name in names:
            index = _recovery_option_end(argv, index)
            continue
        result.append(token)
        index += 1
    return result


def _has_option(argv: list[str], name: str) -> bool:
    return any(_option_name(token) == name for token in argv)


def _human_reviewer_trust_args(config: AgentLoopConfig) -> list[str]:
    """Re-emit every configured trusted human reviewer entry, in order (#1022).

    These govern which signed requirements are admitted, not managed CI, so a
    rerun command must keep them whatever its managed-CI shape.
    """
    args: list[str] = []
    for actor in getattr(config, "human_reviewer_trusted_actors", ()) or ():
        args.extend(("--human-reviewer-trusted-actor", f"{actor.login}:{actor.user_id}"))
    return args


def _ci_plumbing_check_args(config: AgentLoopConfig) -> list[str]:
    args: list[str] = []
    for name in getattr(config, "ci_plumbing_checks", ()) or ():
        args.extend(("--ci-plumbing-check", name))
    return args


def _render_recovery_command(
    config: AgentLoopConfig,
    *,
    target: Literal["issue", "pr"],
    identifier: int,
    managed_ci: bool,
    preserve_managed_options: bool = False,
    include_context: bool = True,
    fresh_authorization: bool = False,
    fresh_issue_number: int | None = None,
) -> str:
    """Build one parser-valid, shell-quoted recovery command.

    Invocation tokens are replayed when available.  Retargeting removes only
    options that the target subparser cannot accept, while preserving repeated
    common options and their complete value payloads.
    """
    if not config.invocation_argv:
        if not include_context:
            command = ["agent-loop", target, str(identifier)]
            if config.auto_merge:
                command.append("--auto-merge")
            elif config.watch_pending_ci:
                command.append("--watch-pending-ci")
            command.extend(_human_reviewer_trust_args(config))
            command.extend(_ci_plumbing_check_args(config))
            return shlex.join(command)
        command = ["agent-loop", target, str(identifier), "--repo", config.repo]
        if config.base:
            command.extend(("--base", config.base))
        command.extend(_human_reviewer_trust_args(config))
        if config.auto_merge:
            command.append("--auto-merge")
        if config.watch_pending_ci and not config.auto_merge:
            command.append("--watch-pending-ci")
        command.extend(_ci_plumbing_check_args(config))
        if managed_ci:
            command.append("--managed-ci")
            if config.managed_ci_trusted_actor:
                command.extend(("--managed-ci-trusted-actor", config.managed_ci_trusted_actor))
            if config.allow_unprotected_managed_ci:
                command.append("--allow-unprotected-managed-ci")
                if config.allow_unreadable_protection:
                    command.append("--allow-unreadable-protection")
            if fresh_authorization and (target != "pr" or fresh_issue_number is not None):
                command.append("--managed-ci-fresh")
                if target == "pr" and fresh_issue_number is not None:
                    command.extend(("--managed-ci-issue", str(fresh_issue_number)))
        return shlex.join(command)

    command = list(config.invocation_argv)
    command_index = next(
        (index for index, token in enumerate(command) if token in {"issue", "pr", "managed-pr"}),
        None,
    )
    if command_index is None:
        # Programmatic callers occasionally preserve only a binary path.  A
        # deterministic fallback is safer than attaching options to an
        # unknown parser shape.
        command = ["agent-loop", target, str(identifier), "--repo", config.repo]
        command.extend(_human_reviewer_trust_args(config))
    else:
        source = command[command_index]
        remove = set(_ONE_SHOT_OPERATOR_RECOVERY_OPTIONS)
        if target == "pr":
            remove.update(_ISSUE_ONLY_RECOVERY_OPTIONS)
            remove.update(_MANAGED_PR_ONLY_RECOVERY_OPTIONS)
        else:
            remove.update(_PR_ONLY_RECOVERY_OPTIONS)
            if source == "managed-pr":
                remove.update(_MANAGED_PR_ONLY_RECOVERY_OPTIONS)
        if not managed_ci:
            if not preserve_managed_options:
                remove.update(_MANAGED_RECOVERY_OPTIONS)
        if config.auto_merge:
            remove.update({"--watch-pending-ci", "--no-watch-pending-ci"})
        command = _strip_recovery_options(command, names=frozenset(remove))
        if source in {"issue", "pr"}:
            # Remove the source command's positional before inserting the
            # retargeted identifier. This also handles `issue --repo X 123`.
            positional_index = _find_recovery_positional_index(command, command_index)
            if positional_index is not None:
                del command[positional_index]
        command_index = next(
            index for index, token in enumerate(command) if token in {"issue", "pr", "managed-pr"}
        )
        command[command_index] = target
        if command_index + 1 >= len(command) or command[command_index + 1].startswith("-"):
            command.insert(command_index + 1, str(identifier))
        else:
            command[command_index + 1] = str(identifier)

    if managed_ci:
        if not _has_option(command, "--managed-ci"):
            command.append("--managed-ci")
        if config.managed_ci_trusted_actor:
            command = _strip_recovery_options(
                command, names=frozenset({"--managed-ci-trusted-actor"})
            )
            command.extend(("--managed-ci-trusted-actor", config.managed_ci_trusted_actor))
        if config.allow_unprotected_managed_ci and not _has_option(command, "--allow-unprotected-managed-ci"):
            command.append("--allow-unprotected-managed-ci")
        if (
            config.allow_unprotected_managed_ci
            and config.allow_unreadable_protection
            and not _has_option(command, "--allow-unreadable-protection")
        ):
            command.append("--allow-unreadable-protection")
        if fresh_authorization and (target != "pr" or fresh_issue_number is not None):
            command = _strip_recovery_options(
                command,
                names=frozenset({"--managed-ci-fresh", "--managed-ci-fresh-authorization", "--managed-ci-issue"}),
            )
            command.append("--managed-ci-fresh")
            if target == "pr" and fresh_issue_number is not None:
                command.extend(("--managed-ci-issue", str(fresh_issue_number)))
    return shlex.join(command)


def render_managed_ci_resume_command(
    config: AgentLoopConfig,
    *,
    pr_number: int,
    managed_ci: bool,
    preserve_managed_options: bool = False,
    issue_number: int | None = None,
    include_context: bool = True,
    fresh_authorization: bool = False,
    fresh_issue_number: int | None = None,
) -> str:
    """Render the deterministic, parser-valid, shell-quoted recovery contract."""
    target: Literal["issue", "pr"] = "pr"
    identifier = pr_number
    if config.invocation_argv:
        command_index = next(
            (
                index
                for index, token in enumerate(config.invocation_argv)
                if token in {"issue", "pr", "managed-pr"}
            ),
            None,
        )
        if command_index is not None and config.invocation_argv[command_index] == "issue":
            target = "issue"
            identifier = issue_number if issue_number is not None else pr_number
            if issue_number is None:
                positional_index = _find_recovery_positional_index(
                    list(config.invocation_argv), command_index
                )
                if positional_index is not None:
                    identifier = int(config.invocation_argv[positional_index])
    # PR-mode fresh authorization has a required issue scope. Callers often
    # already have it as ``issue_number`` from authenticated recovery, so use
    # that value for rendering rather than silently producing a parser-invalid
    # command when the dedicated rendering argument was omitted.
    if fresh_authorization and target == "pr" and fresh_issue_number is None:
        fresh_issue_number = issue_number
    return _render_recovery_command(
        config,
        target=target,
        identifier=identifier,
        managed_ci=managed_ci,
        preserve_managed_options=preserve_managed_options,
        include_context=include_context,
        fresh_authorization=fresh_authorization,
        fresh_issue_number=fresh_issue_number,
    )


