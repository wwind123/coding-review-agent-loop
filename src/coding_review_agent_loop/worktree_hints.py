"""Advisory remedy text for test dependencies in per-run worktrees (#1651).

Every helper here only selects presentation text.  None of them may change a
refusal, a classification, an outcome, recorded evidence or an exit code.

The remedy condition is decided from parent-owned configuration: a path is a
default per-run worktree of this run (``run_worktree_store``) and no
``--worktree-link`` is configured.  The process-local link registry in
``checkout_verification`` cannot establish either fact and is never consulted.
A ``run-tests`` client in the agent's process tree cannot see the parent
configuration, so each backend exports :data:`WORKTREE_DEPENDENCY_HINT_ENV`
into the agent subprocess environment under the same condition.  The client
reads it only to decide whether to print one extra advisory line.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path
from typing import Any, Mapping

WORKTREE_DEPENDENCY_HINT_ENV = "AGENT_LOOP_WORKTREE_DEPENDENCY_HINT"
WORKTREE_DEPENDENCY_HINT_VALUE = "unlinked-run-worktree"

WORKTREE_DEPENDENCY_REMEDY = (
    "This per-run worktree has no linked test dependencies. Ask the operator to "
    "pass --worktree-link PATH (for example .venv or node_modules) to link them "
    "from the shared clone, and --worktree-setup CMD to create them there; see "
    "'Test dependencies in per-run worktrees' in the README."
)

PUSHED_BRANCH_RECOVERY_HEADING = "Recovery after a pushed branch:"


def unlinked_run_worktree(config: Any, path: Path | str | None) -> bool:
    """Whether ``path`` is a default per-run worktree of this run with no configured links."""
    if path is None or getattr(config, "worktree_links", ()):
        return False
    from .config import run_worktree_store

    try:
        return run_worktree_store(config, Path(path)) is not None
    except (OSError, RuntimeError, ValueError):
        return False


def dependency_remedy(config: Any, path: Path | str | None) -> str | None:
    """The remedy sentence for ``path``, or None when the condition does not hold."""
    return WORKTREE_DEPENDENCY_REMEDY if unlinked_run_worktree(config, path) else None


def dependency_hint_environment(config: Any, path: Path | str | None) -> dict[str, str]:
    """Agent subprocess environment entry for the advisory client-side hint."""
    if unlinked_run_worktree(config, path):
        return {WORKTREE_DEPENDENCY_HINT_ENV: WORKTREE_DEPENDENCY_HINT_VALUE}
    return {}


def dependency_hint_requested(environment: Mapping[str, str]) -> bool:
    """Client side: whether the parent asked for the advisory remedy line."""
    return environment.get(WORKTREE_DEPENDENCY_HINT_ENV) == WORKTREE_DEPENDENCY_HINT_VALUE


def with_remedy(message: str, remedy: str | None) -> str:
    """Append ``remedy`` to ``message`` once; the message prefix is preserved."""
    if not remedy or remedy in message:
        return message
    return f"{message.rstrip()} {remedy}"


def replaced_link_remedy(config: Any, path: Path, link: str) -> str:
    """Remedy for a registered worktree link that was removed or retargeted."""
    from .config import run_worktree_store

    try:
        store = run_worktree_store(config, Path(path))
    except (OSError, RuntimeError, ValueError):
        store = None
    source = f"{store / link}" if store is not None else f"the shared clone's {link}"
    return (
        f"The link points at {source} in the shared clone and must stay unchanged; "
        "a coder turn may not replace or reinstall it. Changing dependencies inside "
        "a PR needs PR-local dependency support (not yet available) or an explicit "
        "--<agent>-dir checkout that owns its own environment"
    )


def pushed_branch_recovery(
    *,
    issue_number: int,
    branch: str,
    store: Path,
    worktree: Path,
) -> str:
    """Resume and manual cleanup text for a poisoned run worktree whose branch is pushed."""
    resume = shlex.join(
        ["agent-loop", "managed-pr", "--head", branch, "--expected-closing-issue", str(issue_number)]
    )
    cleanup = shlex.join(["git", "-C", os.fspath(store), "worktree", "remove", "--force", os.fspath(worktree)])
    return (
        f"{PUSHED_BRANCH_RECOVERY_HEADING} branch {branch} was already pushed, so the work "
        f"is not lost. Resume with `{resume}` plus your usual --managed-ci/--auto-merge and "
        "--managed-ci-trusted-actor options (or `agent-loop pr <PR> --expected-closing-issue "
        f"{issue_number}` when a PR for that branch already exists). The untrusted run "
        "worktree is left in place for inspection; agent-loop does not remove it. Remove it "
        f"manually with `{cleanup}`."
    )


def _pushed_branch(config: Any, runner: Any, path: Path) -> str | None:
    """The checkout's named branch when its remote-tracking ref exists, else None."""
    from .checkout_verification import _Incomplete, _probe

    try:
        branch = _probe(config, runner, path, ("rev-parse", "--abbrev-ref", "HEAD")).decode(
            "utf-8", "replace"
        ).strip()
        if not branch or branch == "HEAD":
            return None
        _probe(config, runner, path, ("rev-parse", "--verify", "--quiet", f"refs/remotes/origin/{branch}"))
    except (_Incomplete, OSError):
        return None
    return branch


def add_pushed_branch_recovery(
    exc: BaseException, *, runner: Any, config: Any, issue_number: int
) -> BaseException | None:
    """Return ``exc`` with recovery text appended when a poisoned run worktree's branch is pushed.

    Read-only: it never removes the worktree or relaxes the refusal.  Returns
    None when the recovery does not apply, so the caller re-raises unchanged.
    """
    from .checkout_verification import has_poisoned_entry
    from .config import run_worktree_store
    from .errors import CheckoutVerificationError
    from .workdirs import active_workdir

    if not isinstance(exc, CheckoutVerificationError) or PUSHED_BRANCH_RECOVERY_HEADING in str(exc):
        return None
    try:
        worktree = active_workdir(config)
        store = run_worktree_store(config, worktree)
    except Exception:  # noqa: BLE001 - advisory only
        return None
    if store is None or not has_poisoned_entry(worktree):
        return None
    branch = _pushed_branch(config, runner, worktree)
    if branch is None:
        return None
    text = pushed_branch_recovery(
        issue_number=issue_number, branch=branch, store=store, worktree=worktree
    )
    return CheckoutVerificationError(f"{exc}\n{text}")
