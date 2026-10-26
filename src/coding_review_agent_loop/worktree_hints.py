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

import contextlib
import os
import shlex
from pathlib import Path
from typing import Any, Iterator, Mapping

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
    """Agent subprocess environment entry for the advisory client-side hint.

    The entry is always present: Runner merges backend overrides over the
    orchestrator's own environment, so an inherited selector is neutralized
    with an empty value whenever the parent-owned condition does not hold.
    """
    if unlinked_run_worktree(config, path):
        return {WORKTREE_DEPENDENCY_HINT_ENV: WORKTREE_DEPENDENCY_HINT_VALUE}
    return {WORKTREE_DEPENDENCY_HINT_ENV: ""}


def dependency_hint_requested(environment: Mapping[str, str]) -> bool:
    """Client side: whether the parent asked for the advisory remedy line."""
    return environment.get(WORKTREE_DEPENDENCY_HINT_ENV) == WORKTREE_DEPENDENCY_HINT_VALUE


def with_remedy(message: str, remedy: str | None) -> str:
    """Append ``remedy`` to ``message`` once; the message prefix is preserved."""
    if not remedy or remedy in message:
        return message
    return f"{message.rstrip()} {remedy}"


PR_LOCAL_REMEDY_PATH_LIMIT = 8


def _pr_local_reason(deferred: tuple[str, ...]) -> str:
    if deferred:
        shown = ", ".join(deferred[:PR_LOCAL_REMEDY_PATH_LIMIT])
        extra = len(deferred) - PR_LOCAL_REMEDY_PATH_LIMIT
        more = f" (and {extra} more)" if extra > 0 else ""
        return (
            "agent-loop deferred the PR-local rebuild because these tracked paths are "
            f"uncommitted or hidden from Git: {shown}{more}. Commit or remove those changes; "
            "agent-loop then rebuilds the environment from the committed --worktree-setup-key "
            "files after the next turn"
        )
    return (
        "no --worktree-setup-key file changed at HEAD, so agent-loop did not rebuild it. "
        "Commit the --worktree-setup-key change; agent-loop then rebuilds the environment "
        "privately after the turn"
    )


def replaced_link_remedy(
    config: Any, path: Path, link: str, *, deferred: tuple[str, ...] = ()
) -> str:
    """Remedy for a registered worktree link that was removed or retargeted."""
    from .config import run_worktree_store

    try:
        store = run_worktree_store(config, Path(path))
    except (OSError, RuntimeError, ValueError):
        store = None
    source = f"{store / link}" if store is not None else f"the shared clone's {link}"
    if getattr(config, "worktree_setup_pr_local", False):
        return (
            f"The link points at {source} in the shared clone and must stay unchanged: "
            + _pr_local_reason(deferred)
        )
    return (
        f"The link points at {source} in the shared clone and must stay unchanged; "
        "a coder turn may not replace or reinstall it. Changing dependencies inside "
        "a PR needs the operator flag --worktree-setup-pr-local (which rebuilds the "
        "environment privately in the run worktree after a committed --worktree-setup-key "
        "change) or an explicit --<agent>-dir checkout that owns its own environment"
    )


def replaced_private_remedy(
    config: Any, path: Path, link: str, *, deferred: tuple[str, ...] = ()
) -> str:
    """Remedy for a PR-local environment path that was removed or replaced (#1652)."""
    return (
        f"{Path(path) / link} is the private environment agent-loop built from the committed "
        "setup keys and must stay in place (installing into it is fine; deleting or "
        "replacing it is not): " + _pr_local_reason(deferred)
    )


def pushed_branch_recovery(
    *,
    issue_number: int,
    branch: str,
    store: Path,
    worktree: Path,
    local_only_commits: int | None = 0,
) -> str:
    """Resume and manual cleanup text for a poisoned run worktree whose branch is pushed.

    ``local_only_commits`` is the number of local commits not on
    ``origin/<branch>``; None means the comparison failed (#1670).  With zero
    the text claims the work is not lost; otherwise it never does.
    """
    resume = shlex.join(
        ["agent-loop", "managed-pr", "--head", branch, "--expected-closing-issue", str(issue_number)]
    )
    cleanup = shlex.join(["git", "-C", os.fspath(store), "worktree", "remove", "--force", os.fspath(worktree)])
    if local_only_commits == 0:
        pushed = f"branch {branch} was already pushed, so the work is not lost."
        kept = ""
    else:
        if local_only_commits is None:
            check = shlex.join(["git", "-C", os.fspath(worktree), "log", f"origin/{branch}..HEAD"])
            pushed = (
                f"branch {branch} was already pushed, but the local branch could not be compared "
                f"with origin/{branch}, so agent-loop cannot tell whether every commit was pushed; "
                f"check with `{check}`."
            )
        else:
            noun = "commit" if local_only_commits == 1 else "commits"
            pushed = (
                f"branch {branch} was already pushed, but the local branch has {local_only_commits} "
                f"{noun} that are not on origin/{branch}, and resuming from the pushed branch "
                "omits them."
            )
        kept = (
            " Removing the worktree with that command discards uncommitted changes in it but "
            f"keeps the local branch {branch} and its commits in the shared repository."
        )
    return (
        f"{PUSHED_BRANCH_RECOVERY_HEADING} {pushed} Resume with `{resume}` plus your usual "
        "--managed-ci/--auto-merge and --managed-ci-trusted-actor options (or `agent-loop pr "
        f"<PR> --expected-closing-issue {issue_number}` when a PR for that branch already "
        "exists). The untrusted run worktree is left in place for inspection; agent-loop does "
        f"not remove it. Remove it manually with `{cleanup}`.{kept}"
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


def _local_only_commits(config: Any, runner: Any, path: Path, branch: str) -> int | None:
    """How many commits at HEAD are not on ``origin/<branch>``; None when the count fails (#1670)."""
    from .checkout_verification import _Incomplete, _probe

    try:
        raw = _probe(config, runner, path, ("rev-list", "--count", f"refs/remotes/origin/{branch}..HEAD"))
        count = int(raw.decode("ascii").strip())
    except (_Incomplete, OSError, UnicodeDecodeError, ValueError):
        return None
    return count if count >= 0 else None


def add_pushed_branch_recovery(
    exc: BaseException, *, runner: Any, config: Any, issue_number: int
) -> bool:
    """Append recovery text to ``exc`` in place when a poisoned run worktree's branch is pushed.

    Applies to every ``AgentLoopError`` stop with a plain message: a checkout
    refusal and also a terminal no-PR stop after a coder turn that pushed and
    then poisoned its baseline.  Only the message grows; the exception object,
    its type and the exit code are unchanged.  Read-only: it never removes the
    worktree or relaxes the refusal.  Returns whether the text was appended.
    """
    from .checkout_verification import has_poisoned_entry
    from .config import run_worktree_store
    from .errors import AgentLoopError
    from .workdirs import active_workdir

    if not isinstance(exc, AgentLoopError) or PUSHED_BRANCH_RECOVERY_HEADING in str(exc):
        return False
    if type(exc).__str__ is not BaseException.__str__ or len(exc.args) != 1 or not isinstance(exc.args[0], str):
        return False
    try:
        worktree = active_workdir(config)
        store = run_worktree_store(config, worktree)
    except Exception:  # noqa: BLE001 - advisory only
        return False
    if store is None or not has_poisoned_entry(worktree):
        return False
    branch = _pushed_branch(config, runner, worktree)
    if branch is None:
        return False
    text = pushed_branch_recovery(
        issue_number=issue_number, branch=branch, store=store, worktree=worktree,
        local_only_commits=_local_only_commits(config, runner, worktree, branch),
    )
    exc.args = (f"{exc.args[0]}\n{text}",)
    return True


@contextlib.contextmanager
def pushed_branch_recovery_scope(*, runner: Any, config: Any, issue_number: int) -> Iterator[None]:
    """Re-raise any stop from an issue run with the pushed-branch recovery appended (#1651)."""
    from .errors import AgentLoopError

    try:
        yield
    except AgentLoopError as exc:
        add_pushed_branch_recovery(exc, runner=runner, config=config, issue_number=issue_number)
        raise
