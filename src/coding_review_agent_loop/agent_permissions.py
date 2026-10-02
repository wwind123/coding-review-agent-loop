"""Role-scoped agent permission grants for ``--agent-permissions`` (#1035).

``default`` and ``dangerous`` keep the historical static per-provider argv.
``sandboxed`` builds CLI-enforced grants per invocation from the provider and
the invocation role:

* only ``role == "coder"`` gets the coder grant; every other role, including
  role-less planner turns and unknown roles, fails closed to read-only;
* every response-directory grant targets a physically validated root that is
  outside every agent checkout and is re-verified before each spawn;
* the only shell grant for a Claude non-coder is ``agent-loop inspect``,
  pinned to this interpreter in isolated mode and to startup-verified
  absolute ``git``/``gh`` executables whose provenance is re-verified before
  every read-only Claude turn.

State established at startup (the response-root record, the checkout set, and
the inspect provenance manifest) is process-local and keyed by the lexical
response root, so configs derived with ``dataclasses.replace`` share it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, Mapping

from .errors import AgentLoopError
from .scratch import mkdir_private

if TYPE_CHECKING:
    from .config import AgentLoopConfig

AGENT_PERMISSION_MODES = ("default", "sandboxed", "dangerous")
SANDBOXED_PROVIDERS = ("claude", "codex")
PERMISSION_CLASS_CODER = "coder"
PERMISSION_CLASS_READ_ONLY = "read-only"

CLAUDE_READ_ONLY_TOOLS = "Read,Grep,Glob,Write,Edit,Bash"
# Claude Code auto-approves some read-only shell commands (``git status`` among
# them) whatever ``--allowedTools`` says, so absence from the allow list is not
# a refusal.  A read-only role must reach the checkout only through
# ``agent-loop inspect``: a bare ``git`` in the shared checkout would run a
# planted ``.git/config`` (``core.fsmonitor`` and friends).  Deny the two
# programs outright, in both rule spellings, so the auto-approval cannot apply.
CLAUDE_READ_ONLY_DENIED_TOOLS = (
    "Bash(git)",
    "Bash(git *)",
    "Bash(git:*)",
    "Bash(gh)",
    "Bash(gh *)",
    "Bash(gh:*)",
)
# Inherited git variables that steer the agent CLI's *own* git calls, before
# any ``inspect`` runs.  ``inspect`` has an environment allowlist; the agent
# process does not, so these are neutralized on the process itself.  Tracing
# writes attacker-named files into the checkout (observed: GIT_TRACE2_EVENT
# producing trace2.json), and the GIT_CONFIG_* injection trio sets arbitrary
# config -- including exec hooks -- for every git the CLI runs.
GIT_TRACE_VARIABLES = (
    "GIT_TRACE",
    "GIT_TRACE2",
    "GIT_TRACE2_EVENT",
    "GIT_TRACE2_PERF",
    "GIT_TRACE_CURL",
    "GIT_TRACE_FSMONITOR",
    "GIT_TRACE_PACKET",
    "GIT_TRACE_PACK_ACCESS",
    "GIT_TRACE_PERFORMANCE",
    "GIT_TRACE_REFS",
    "GIT_TRACE_SETUP",
    "GIT_TRACE_SHALLOW",
)
EMPTY_MCP_CONFIG = '{"mcpServers":{}}'
# gh subcommands the coder prompts instruct (PR creation/edits, body-file
# comments, label verification, bounded CI snapshots, issue reads).
CODER_GH_SUBCOMMANDS = (
    "gh pr create",
    "gh pr view",
    "gh pr edit",
    "gh pr comment",
    "gh pr checks",
    "gh pr diff",
    "gh issue view",
    "gh issue comment",
    "gh run view",
    "gh label create",
    "gh label list",
    "gh repo view",
)
_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9._/+-]+$")
_PROVIDER_COMPONENTS = SANDBOXED_PROVIDERS


class SandboxBoundaryError(AgentLoopError):
    """A sandboxed spawn was refused by a boundary or provenance re-check."""


def agent_permissions_mode(config: object) -> str:
    return getattr(config, "agent_permissions", "default") or "default"


def is_sandboxed(config: object) -> bool:
    return agent_permissions_mode(config) == "sandboxed"


def permission_class_for_role(role: str | None) -> str:
    """Only the exact ``coder`` role is a coder; everything else fails closed."""
    return PERMISSION_CLASS_CODER if role == "coder" else PERMISSION_CLASS_READ_ONLY


def _safe_repo_slug(repo: str) -> str:
    return repo.replace("/", "-").replace(":", "-")


def response_root(config: AgentLoopConfig) -> Path:
    """The single source for the lexical per-repository response directory."""
    return (
        Path(tempfile.gettempdir())
        / "coding-review-agent-loop"
        / "responses"
        / _safe_repo_slug(config.repo)
    )


# --------------------------------------------------------------- selections


@dataclass(frozen=True)
class AgentSelection:
    field: str
    agent: str
    fix: str


def configured_agent_selections(config: AgentLoopConfig) -> tuple[AgentSelection, ...]:
    """Every agent a sandboxed run could spawn, with the option that selects it."""
    selections = [
        AgentSelection("coder", config.coder, "--coder claude|codex"),
    ]
    if config.implementation_coder is not None:
        selections.append(
            AgentSelection(
                "implementation_coder", config.implementation_coder, "--implementation-coder claude"
            )
        )
    for reviewer in config.reviewer:
        selections.append(AgentSelection("reviewer", reviewer, "--reviewer claude|codex"))
    if config.primary_reviewer is not None:
        selections.append(
            AgentSelection("primary_reviewer", config.primary_reviewer, "--primary-reviewer claude|codex")
        )
    if config.primary_plan_reviewer is not None:
        selections.append(
            AgentSelection(
                "primary_plan_reviewer", config.primary_plan_reviewer, "--primary-plan-reviewer claude|codex"
            )
        )
    if config.discuss_analyzer is not None:
        selections.append(
            AgentSelection("discuss_analyzer", config.discuss_analyzer, "--discuss-analyzer claude|codex")
        )
    selections.append(
        AgentSelection(
            "repair_backend",
            config.repair_backend,
            "--repair-backend claude|codex --repair-model MODEL",
        )
    )
    if config.semantic_followup_dedupe:
        selections.append(
            AgentSelection(
                "semantic_followup_backend",
                config.semantic_followup_backend,
                "--semantic-followup-backend claude|codex or --no-semantic-followup-dedupe",
            )
        )
    return tuple(selections)


# Why Antigravity (and Gemini) have no sandboxed grant (#1079).  Observed live
# with agy 1.2.11: ``--sandbox`` confines only the terminal tool, making the
# workspace read-only to shell commands, while its file-writing tool still
# writes into the checkout and a bare ``git status`` in the checkout runs a
# planted ``core.fsmonitor`` hook.  The remaining control, ``permissions.allow``
# rules, lives only in the shared per-user settings file (there is no
# per-invocation flag), and headless mode ends the whole turn with no output on
# the first tool it cannot prompt for.  None of that is a per-invocation,
# CLI-enforced read-only grant, so these providers stay refused.
UNSANDBOXABLE_PROVIDER_REASONS = {
    "antigravity": (
        "Antigravity has no CLI-enforced read-only grant: `agy --sandbox` restricts only "
        "its terminal, so its file-writing tool can still write the checkout and a bare "
        "`git` there runs the checkout's .git/config, and its allow rules exist only in "
        "the shared user settings file"
    ),
    "gemini": "Gemini has no CLI-enforced read-only grant",
}


def _unsupported_provider_reason(agent: str) -> str:
    return UNSANDBOXABLE_PROVIDER_REASONS.get(agent, f"{agent!r} has no sandboxed grant")


def validate_sandboxed_selections(config: AgentLoopConfig) -> None:
    for selection in configured_agent_selections(config):
        if selection.agent not in SANDBOXED_PROVIDERS:
            trade = (
                " A sandboxed review board is therefore limited to Claude and Codex "
                "reviewers; to keep this reviewer, run without sandboxed permissions."
                if selection.field in {"reviewer", "primary_reviewer", "primary_plan_reviewer"}
                else ""
            )
            raise AgentLoopError(
                f"--agent-permissions sandboxed supports only Claude and Codex, but "
                f"{selection.field} selects {selection.agent!r}. "
                f"{_unsupported_provider_reason(selection.agent)}.{trade} Use {selection.fix}, "
                "or choose --agent-permissions default|dangerous."
            )


def validate_sandboxed_passthrough(args_by_option: Mapping[str, object]) -> None:
    """Reject every pass-through agent argument (a closed, empty allowlist)."""
    for option, value in args_by_option.items():
        if value:
            first = value[0] if isinstance(value, (list, tuple)) else value
            raise AgentLoopError(
                f"{option} {first!s} is not allowed with --agent-permissions sandboxed: "
                "sandboxed mode builds every agent permission argument itself. Use the "
                "dedicated model/effort options (for example --claude-model, "
                "--codex-model, --claude-effort, --codex-reasoning-effort), or choose "
                "--agent-permissions default|dangerous for custom pass-through arguments."
            )


def committing_coder(config: AgentLoopConfig, *, command: str, plan_first: bool = False) -> str | None:
    """Return the agent that will commit in this flow, or None for plan-only runs."""
    if command in {"pr", "managed-pr", "task"}:
        return config.coder
    if command == "issue":
        if plan_first and config.plan_execution_mode in {"plan-only", "decompose-only"}:
            return None
        if plan_first:
            return config.implementation_coder or config.coder
        return config.coder
    return None


def validate_sandboxed_flow(config: AgentLoopConfig, *, command: str, plan_first: bool = False) -> None:
    if not is_sandboxed(config):
        return
    if committing_coder(config, command=command, plan_first=plan_first) == "codex":
        raise AgentLoopError(
            "--agent-permissions sandboxed cannot run a committing Codex coder: Codex's "
            "sandbox keeps the repository's .git read-only, so `git commit` fails. Use "
            "--coder claude (or --implementation-coder claude for plan-first runs), or "
            "--agent-permissions dangerous on a host where that is acceptable. A Codex "
            "planner in a plan-only run is allowed."
        )


# ------------------------------------------------------ response-root boundary


@dataclass
class _CheckoutRecord:
    stored: str
    exists: bool
    resolved: str
    dev: int | None
    ino: int | None
    # False for the orchestrator's repository checkout: it bounds the response
    # root but is not an agent checkout for the inspect provenance rule, so
    # agent-loop may run from a development clone there.
    agent: bool = True


@dataclass(frozen=True)
class _ComponentRecord:
    path: str
    dev: int
    ino: int


@dataclass(frozen=True)
class PinnedExecutable:
    name: str
    path: str
    resolved: str
    sha256: str
    size: int
    mtime_ns: int
    dir_dev: int
    dir_ino: int


@dataclass(frozen=True)
class InspectProvenance:
    interpreter: str
    interpreter_resolved: str
    interpreter_size: int
    interpreter_mtime_ns: int
    package_dir: str
    package_hashes: tuple[tuple[str, str], ...]
    git: PinnedExecutable
    gh: PinnedExecutable | None

    @property
    def prefix(self) -> tuple[str, ...]:
        prefix = (
            self.interpreter,
            "-I",
            "-m",
            "coding_review_agent_loop.cli",
            "inspect",
            f"--git={self.git.path}",
        )
        if self.gh is not None:
            prefix += (f"--gh={self.gh.path}",)
        return prefix

    @property
    def prefix_text(self) -> str:
        return " ".join(self.prefix)


@dataclass
class SandboxState:
    lexical_root: str
    resolved_root: Path
    components: tuple[_ComponentRecord, ...]
    checkouts: dict[str, _CheckoutRecord] = field(default_factory=dict)
    # Checkouts recorded at startup or re-registered by agent-loop.  Other
    # records belong to ephemeral per-call directories (for example the
    # isolated semantic-dedupe directory) and are dropped once no current
    # config names them.
    persistent: set[str] = field(default_factory=set)
    provenance: InspectProvenance | None = None


_STATES: dict[str, SandboxState] = {}
_STATE_LOCK = threading.RLock()


def reset_sandbox_state() -> None:
    """Forget all established boundaries (tests and fresh runs)."""
    with _STATE_LOCK:
        _STATES.clear()
    from .test_runtime import reset_coder_test_invocations

    reset_coder_test_invocations()


def _state_key(config: AgentLoopConfig) -> str:
    return str(response_root(config))


def _is_within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _overlaps(left: str, right: str) -> bool:
    return _is_within(left, right) or _is_within(right, left)


def _observe_checkout(path: str) -> _CheckoutRecord:
    """Resolve a checkout path; an absent one resolves through its nearest ancestor."""
    lexical = os.path.abspath(path)
    try:
        info = os.stat(lexical)
    except FileNotFoundError:
        ancestor = lexical
        missing: list[str] = []
        while not os.path.lexists(ancestor):
            parent = os.path.dirname(ancestor)
            if parent == ancestor:
                break
            missing.insert(0, os.path.basename(ancestor))
            ancestor = parent
        resolved = os.path.join(os.path.realpath(ancestor), *missing)
        return _CheckoutRecord(lexical, False, resolved, None, None)
    return _CheckoutRecord(lexical, True, os.path.realpath(lexical), info.st_dev, info.st_ino)


def repository_checkout(cwd: str | None = None) -> str | None:
    """The git work tree containing the orchestrator's working directory, if any."""
    current = os.path.abspath(cwd or os.getcwd())
    while True:
        if os.path.lexists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def configured_checkouts(config: AgentLoopConfig) -> tuple[str, ...]:
    paths: list[str] = []
    for value in (config.claude_dir, config.codex_dir, config.gemini_dir, config.antigravity_dir):
        text = os.path.abspath(str(value))
        if text not in paths:
            paths.append(text)
    return tuple(paths)


def _writable_components_error(offenders: list[tuple[str, int]]) -> SandboxBoundaryError:
    # Name every offending component and one command that fixes them all, so
    # an operator with a deep pre-existing tree needs a single fix, not one
    # failed run per directory.
    if len(offenders) == 1:
        path, mode = offenders[0]
        return SandboxBoundaryError(
            f"Sandboxed response directory component {path} is group- or world-writable "
            f"(mode {mode:o}). Run `chmod go-w {shlex.quote(path)}` or set TMPDIR "
            "to a private directory."
        )
    listing = ", ".join(f"{path} (mode {mode:o})" for path, mode in offenders)
    paths = " ".join(shlex.quote(path) for path, _mode in offenders)
    return SandboxBoundaryError(
        f"Sandboxed response directory components are group- or world-writable: {listing}. "
        f"Run `chmod go-w {paths}` or set TMPDIR to a private directory."
    )


def _check_component(
    path: str, *, create: bool, writable: list[tuple[str, int]] | None = None
) -> _ComponentRecord:
    """Validate one component.

    With ``writable`` supplied, a group- or world-writable component is
    recorded there instead of raised, so the caller can report every offender
    at once.  Every other defect still raises immediately.
    """
    if create:
        try:
            mkdir_private(path)
        except OSError as exc:
            raise AgentLoopError(f"Could not create sandboxed response directory {path}: {exc}") from exc
    try:
        info = os.lstat(path)
    except OSError as exc:
        raise SandboxBoundaryError(f"Sandboxed response directory {path} is missing: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise SandboxBoundaryError(
            f"Sandboxed response directory component {path} is not a real directory "
            "(symlinks are refused). Remove it or set TMPDIR to a directory outside every checkout."
        )
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise SandboxBoundaryError(
            f"Sandboxed response directory component {path} is owned by uid {info.st_uid}, "
            "not the current user. Remove it or set TMPDIR to a private directory."
        )
    if info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        # A group- or world-writable component would let another local user
        # replace entries between verification and the CLI's response write.
        offender = (path, stat.S_IMODE(info.st_mode))
        if writable is None:
            raise _writable_components_error([offender])
        writable.append(offender)
    return _ComponentRecord(path, info.st_dev, info.st_ino)


def _response_component_paths(config: AgentLoopConfig) -> tuple[Path, list[str]]:
    base = Path(os.path.realpath(tempfile.gettempdir()))
    parts = ["coding-review-agent-loop", "responses", _safe_repo_slug(config.repo)]
    paths: list[str] = []
    current = base
    for part in parts:
        current = current / part
        paths.append(str(current))
    for provider in _PROVIDER_COMPONENTS:
        paths.append(str(current / provider))
    return current, paths


def _check_overlap(resolved_root: Path, record: _CheckoutRecord) -> None:
    if _overlaps(str(resolved_root), record.resolved):
        raise SandboxBoundaryError(
            f"The sandboxed response root {resolved_root} overlaps "
            f"{'agent' if record.agent else 'repository'} checkout "
            f"{record.stored} (resolves to {record.resolved}). Set TMPDIR to a directory "
            "outside every agent checkout, and do not nest checkouts in the response root."
        )


def establish_response_root_boundary(config: AgentLoopConfig) -> SandboxState:
    """Create and validate the response root and record the checkout set."""
    resolved_root, component_paths = _response_component_paths(config)
    with _STATE_LOCK:
        writable: list[tuple[str, int]] = []
        components = tuple(
            _check_component(path, create=True, writable=writable) for path in component_paths
        )
        if writable:
            raise _writable_components_error(writable)
        state = SandboxState(
            lexical_root=_state_key(config),
            resolved_root=resolved_root,
            components=components,
        )
        records = [_observe_checkout(checkout) for checkout in configured_checkouts(config)]
        repository = repository_checkout()
        if repository is not None and repository not in {record.stored for record in records}:
            # The orchestrator's own repository checkout is part of the
            # overlap set even when every provider uses a separate clone.
            records.append(dataclasses.replace(_observe_checkout(repository), agent=False))
        for record in records:
            _check_overlap(resolved_root, record)
            state.checkouts[record.stored] = record
            state.persistent.add(record.stored)
        previous = _STATES.get(state.lexical_root)
        if previous is not None:
            state.provenance = previous.provenance
        _STATES[state.lexical_root] = state
        return state


def _require_state(config: AgentLoopConfig) -> SandboxState:
    with _STATE_LOCK:
        state = _STATES.get(_state_key(config))
    if state is None:
        state = establish_response_root_boundary(config)
    return state


def sandboxed_response_root(config: AgentLoopConfig) -> Path:
    return _require_state(config).resolved_root


def register_checkout(config: AgentLoopConfig, path: Path | str) -> None:
    """Re-record a checkout that agent-loop itself (re-)created between turns."""
    if not is_sandboxed(config):
        return
    state = _require_state(config)
    record = _observe_checkout(str(path))
    _check_overlap(state.resolved_root, record)
    if state.provenance is not None:
        _check_provenance_outside(state.provenance, [record])
    with _STATE_LOCK:
        state.checkouts[record.stored] = record
        state.persistent.add(record.stored)


def verify_response_boundary(config: AgentLoopConfig) -> SandboxState:
    """Re-verify components and every recorded checkout before a spawn."""
    state = _require_state(config)
    with _STATE_LOCK:
        writable: list[tuple[str, int]] = []
        for component in state.components:
            current = _check_component(component.path, create=False, writable=writable)
            if (current.dev, current.ino) != (component.dev, component.ino):
                raise SandboxBoundaryError(
                    f"Sandboxed response directory {component.path} was replaced since startup."
                )
        if writable:
            raise _writable_components_error(writable)
        current_checkouts = configured_checkouts(config)
        for stored in list(state.checkouts):
            if stored not in state.persistent and stored not in current_checkouts:
                del state.checkouts[stored]
        for checkout in current_checkouts:
            if checkout not in state.checkouts:
                record = _observe_checkout(checkout)
                _check_overlap(state.resolved_root, record)
                state.checkouts[record.stored] = record
        for stored, record in list(state.checkouts.items()):
            current = dataclasses.replace(_observe_checkout(stored), agent=record.agent)
            if not record.exists and current.resolved != record.resolved:
                # An absent checkout is tracked through its nearest existing
                # ancestor; a redirected ancestor changes where it would land.
                raise SandboxBoundaryError(
                    f"Checkout {stored} (not yet created when recorded) now resolves to "
                    f"{current.resolved} instead of {record.resolved}; an ancestor directory "
                    "was replaced or redirected by a symlink."
                )
            if record.exists:
                if not current.exists:
                    raise SandboxBoundaryError(
                        f"Agent checkout {stored} disappeared since it was recorded."
                    )
                if current.resolved != record.resolved or (current.dev, current.ino) != (
                    record.dev,
                    record.ino,
                ):
                    raise SandboxBoundaryError(
                        f"Agent checkout {stored} changed since it was recorded "
                        f"(now resolves to {current.resolved}); it was replaced, moved, or "
                        "redirected by a symlink."
                    )
            _check_overlap(state.resolved_root, current)
            if not record.exists and current.exists:
                # A lazily created checkout is adopted once it exists and
                # passes the overlap check; later changes are then detected.
                state.checkouts[stored] = current
    return state


def prepare_response_file(config: AgentLoopConfig, provider: str) -> Path:
    """Re-verify the boundary, then exclusively create this invocation's file."""
    state = verify_response_boundary(config)
    directory = state.resolved_root / provider
    path = directory / f"{uuid.uuid4().hex}.md"
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise SandboxBoundaryError(
            f"Could not exclusively create sandboxed response file {path}: {exc}"
        ) from exc
    os.close(fd)
    return path


# ------------------------------------------------------ inspect provenance


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pin_executable(name: str, path: str) -> PinnedExecutable:
    lexical = os.path.abspath(path)
    resolved = os.path.realpath(lexical)
    info = os.stat(resolved)
    directory = os.stat(os.path.dirname(lexical))
    return PinnedExecutable(
        name=name,
        path=lexical,
        resolved=resolved,
        sha256=_sha256(resolved),
        size=info.st_size,
        mtime_ns=info.st_mtime_ns,
        dir_dev=directory.st_dev,
        dir_ino=directory.st_ino,
    )


def _locate_package(interpreter: str) -> str:
    """Ask the pinned interpreter in isolated mode where the package imports from."""
    script = (
        "import importlib.util;"
        "s=importlib.util.find_spec('coding_review_agent_loop');"
        "print(list(s.submodule_search_locations)[0])"
    )
    completed = subprocess.run(
        [interpreter, "-I", "-c", script],
        cwd=os.path.realpath(tempfile.gettempdir()),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        raise AgentLoopError(
            "Could not locate the coding_review_agent_loop package with "
            f"`{interpreter} -I`: {completed.stderr.strip() or 'no output'}. "
            "Install agent-loop as a regular (non-editable-in-checkout) package, "
            "for example with `uv tool install`."
        )
    return os.path.realpath(completed.stdout.strip())


def _package_hashes(package_dir: str) -> tuple[tuple[str, str], ...]:
    hashes: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(package_dir):
        dirs[:] = sorted(item for item in dirs if item != "__pycache__")
        for name in sorted(files):
            if name.endswith(".py"):
                full = os.path.join(root, name)
                hashes.append((os.path.relpath(full, package_dir), _sha256(full)))
    return tuple(hashes)


def _provenance_paths(provenance: InspectProvenance) -> list[tuple[str, str]]:
    paths = [
        ("interpreter", provenance.interpreter),
        ("interpreter", provenance.interpreter_resolved),
        ("package", provenance.package_dir),
    ]
    for pinned in (provenance.git, provenance.gh):
        if pinned is None:
            continue
        paths += [
            (pinned.name, pinned.path),
            (pinned.name, pinned.resolved),
            (f"{pinned.name} directory", os.path.dirname(pinned.path)),
        ]
    return paths


def _check_provenance_outside(
    provenance: InspectProvenance, records: Iterable[_CheckoutRecord], response_root_path: str | None = None
) -> None:
    roots = [(record.stored, record.resolved) for record in records if record.agent]
    for label, path in _provenance_paths(provenance):
        resolved = os.path.realpath(path)
        for stored, root in roots:
            for candidate in (os.path.abspath(path), resolved):
                if _is_within(candidate, root) or _is_within(candidate, stored):
                    guidance = (
                        "Install agent-loop outside the agent checkouts (for example "
                        "`uv tool install`)"
                        if label in {"interpreter", "package"}
                        else "Remove checkout directories from PATH ahead of the system git and gh"
                    )
                    raise AgentLoopError(
                        f"--agent-permissions sandboxed refuses the {label} at {path}: it lies "
                        f"inside agent checkout {stored}. {guidance}."
                    )
        if response_root_path and _is_within(resolved, response_root_path):
            raise AgentLoopError(
                f"--agent-permissions sandboxed refuses the {label} at {path}: it lies "
                "inside the response root."
            )


def executable_location_refusal(path: str, config: AgentLoopConfig) -> str | None:
    """Return a reason when an executable lies inside an agent checkout or the response root.

    Applies the sandboxed provenance rule to the executable, its realpath and
    its directory in every permission mode; creates nothing on disk.
    """
    absolute = os.path.abspath(path)
    resolved = os.path.realpath(path)
    candidates = (absolute, resolved, os.path.dirname(absolute), os.path.dirname(resolved))
    records = [_observe_checkout(checkout) for checkout in configured_checkouts(config)]
    resolved_root = str(_response_component_paths(config)[0])
    for candidate in candidates:
        real = os.path.realpath(candidate)
        for record in records:
            for form in (os.path.abspath(candidate), real):
                if _is_within(form, record.resolved) or _is_within(form, record.stored):
                    return f"{path} lies inside agent checkout {record.stored}"
        if _is_within(real, resolved_root):
            return f"{path} lies inside the response root"
    return None


def establish_inspect_provenance(
    config: AgentLoopConfig,
    *,
    which: Callable[[str], str | None] = shutil.which,
    package_locator: Callable[[str], str] | None = None,
    interpreter: str | None = None,
) -> InspectProvenance:
    """Pin the interpreter, package, git, and gh used by the inspect grant."""
    state = _require_state(config)
    interpreter_path = os.path.abspath(interpreter or sys.executable)
    interpreter_resolved = os.path.realpath(interpreter_path)
    interpreter_info = os.stat(interpreter_resolved)
    package_dir = (package_locator or _locate_package)(interpreter_path)
    git_path = which("git")
    if not git_path:
        raise AgentLoopError(
            "--agent-permissions sandboxed requires git on PATH; inspect runs only a "
            "startup-pinned git."
        )
    gh_path = which("gh")
    provenance = InspectProvenance(
        interpreter=interpreter_path,
        interpreter_resolved=interpreter_resolved,
        interpreter_size=interpreter_info.st_size,
        interpreter_mtime_ns=interpreter_info.st_mtime_ns,
        package_dir=package_dir,
        package_hashes=_package_hashes(package_dir),
        git=_pin_executable("git", git_path),
        gh=_pin_executable("gh", gh_path) if gh_path else None,
    )
    for component in provenance.prefix:
        if not _SAFE_PREFIX_RE.match(component.split("=", 1)[-1]):
            raise AgentLoopError(
                f"--agent-permissions sandboxed needs shell-safe paths for the inspect grant, "
                f"but {component!r} contains other characters. Install agent-loop, git, and gh "
                "under plain paths (letters, digits, and ._/+-)."
            )
    _check_provenance_outside(
        provenance, state.checkouts.values(), response_root_path=str(state.resolved_root)
    )
    with _STATE_LOCK:
        state.provenance = provenance
    return provenance


def require_inspect_provenance(config: AgentLoopConfig) -> InspectProvenance:
    state = _require_state(config)
    if state.provenance is None:
        return establish_inspect_provenance(config)
    return state.provenance


def verify_inspect_provenance(config: AgentLoopConfig) -> InspectProvenance:
    """Recompute the manifest; any change blocks the read-only Claude spawn."""
    provenance = require_inspect_provenance(config)
    try:
        info = os.stat(provenance.interpreter_resolved)
    except OSError as exc:
        raise SandboxBoundaryError(f"Pinned interpreter {provenance.interpreter} is missing: {exc}") from exc
    if (
        os.path.realpath(provenance.interpreter) != provenance.interpreter_resolved
        or info.st_size != provenance.interpreter_size
        or info.st_mtime_ns != provenance.interpreter_mtime_ns
    ):
        raise SandboxBoundaryError(
            f"Pinned interpreter {provenance.interpreter} changed since startup."
        )
    current = dict(_package_hashes(provenance.package_dir))
    expected = dict(provenance.package_hashes)
    for name in sorted(set(current) | set(expected)):
        if current.get(name) != expected.get(name):
            raise SandboxBoundaryError(
                f"Installed agent-loop file {os.path.join(provenance.package_dir, name)} "
                "changed since startup; refusing the inspect grant."
            )
    for pinned in (provenance.git, provenance.gh):
        if pinned is None:
            continue
        try:
            resolved = os.path.realpath(pinned.path)
            info = os.stat(resolved)
            directory = os.stat(os.path.dirname(pinned.path))
            digest = _sha256(resolved)
        except OSError as exc:
            raise SandboxBoundaryError(f"Pinned {pinned.name} {pinned.path} is missing: {exc}") from exc
        if (
            resolved != pinned.resolved
            or digest != pinned.sha256
            or info.st_size != pinned.size
            or info.st_mtime_ns != pinned.mtime_ns
            or (directory.st_dev, directory.st_ino) != (pinned.dir_dev, pinned.dir_ino)
        ):
            raise SandboxBoundaryError(
                f"Pinned {pinned.name} executable {pinned.path} changed since startup "
                f"(resolved {resolved}); restart the run to re-pin it."
            )
    return provenance


@dataclass(frozen=True)
class _ProbeResult:
    returncode: int
    stdout: str
    stderr: str


def hardened_git_probe_runner(config: AgentLoopConfig) -> Callable[[tuple[str, ...], Path], _ProbeResult]:
    """A git runner for agent-loop's own probes of a sandboxed checkout.

    It runs the pinned git through ``inspect``'s config gate, closed
    environment, and forced overrides, so the workdir snapshot taken before
    and after a sandboxed turn cannot run a coder-planted ``core.fsmonitor``,
    filter, or submodule config.  A gate refusal surfaces as an
    ``AgentLoopError``, which the tolerant snapshot records as unavailable.
    """
    from . import inspect_tool

    def run(args: tuple[str, ...], workdir: Path) -> _ProbeResult:
        try:
            git = require_inspect_provenance(config).git.path
            result = inspect_tool.run_hardened_git(git, args[0], args[1:], cwd=str(workdir))
        except inspect_tool.InspectRejected as exc:
            raise AgentLoopError(f"sandboxed workdir probe refused: {exc}") from exc
        return _ProbeResult(
            result.returncode,
            result.stdout.decode("utf-8", "replace"),
            result.stderr.decode("utf-8", "replace"),
        )

    return run


@dataclass(frozen=True)
class _BinaryProbeResult:
    returncode: int
    stdout: bytes
    stderr: bytes


def hardened_git_probe_runner_bytes(
    config: AgentLoopConfig,
) -> Callable[[tuple[str, ...], Path], _BinaryProbeResult]:
    """Like :func:`hardened_git_probe_runner`, but returns undecoded bytes.

    Checkout verification must keep non-UTF-8 path names exact, so it cannot
    use the lossy text variant.  The gate, closed environment and forced
    configuration are identical.
    """
    from . import inspect_tool

    def run(args: tuple[str, ...], workdir: Path) -> _BinaryProbeResult:
        try:
            git = require_inspect_provenance(config).git.path
            result = inspect_tool.run_hardened_git(git, args[0], args[1:], cwd=str(workdir))
        except inspect_tool.InspectRejected as exc:
            raise AgentLoopError(f"sandboxed workdir probe refused: {exc}") from exc
        return _BinaryProbeResult(result.returncode, result.stdout, result.stderr)

    return run


def inspect_prefix(config: AgentLoopConfig) -> str:
    return require_inspect_provenance(config).prefix_text


def neutralized_git_env() -> dict[str, str]:
    """Safe values for the inherited git variables, for agent-loop and its children.

    ``0`` disables each trace target, and ``GIT_CONFIG_COUNT=0`` makes any
    inherited ``GIT_CONFIG_KEY_n`` / ``GIT_CONFIG_VALUE_n`` pair unreadable.
    Values are overridden rather than removed so one mapping serves both
    ``os.environ.update`` here and the runner's ``{**os.environ, **env}`` merge.
    """
    env = {name: "0" for name in GIT_TRACE_VARIABLES}
    env["GIT_CONFIG_COUNT"] = "0"
    # A pure exec hook: git runs it for every diff, and no role needs it.
    env["GIT_EXTERNAL_DIFF"] = ""
    return env


def establish_sandboxed_run(
    config: AgentLoopConfig, *, command: str, plan_first: bool = False
) -> None:
    """Fail-fast startup for sandboxed mode, before any agent is spawned."""
    if not is_sandboxed(config):
        return
    validate_sandboxed_selections(config)
    validate_sandboxed_flow(config, command=command, plan_first=plan_first)
    # agent-loop runs git in its own process too (checkout identity, HEAD
    # snapshots, config probes), and those calls inherit the caller's
    # environment.  Scrubbing only the agent subprocess left an inherited
    # GIT_TRACE2_EVENT writing trace2.json into the checkout from
    # agent-loop's own `git rev-parse HEAD` -- the trace file named
    # `python` as its parent, not the CLI.  Neutralize here, once, so every
    # in-process git call and every child inherits the safe values.
    os.environ.update(neutralized_git_env())
    establish_response_root_boundary(config)
    if any(selection.agent == "claude" for selection in configured_agent_selections(config)):
        establish_inspect_provenance(config)


# ------------------------------------------------------------- grants


def _claude_path_rule(tool: str, root: Path) -> str:
    return f"{tool}(/{root}/**)"


def coder_checkout_paths(config: AgentLoopConfig) -> tuple[Path, ...]:
    """The Claude coder's assigned checkout, as given and as resolved when they differ."""
    lexical = Path(os.path.abspath(str(config.claude_dir)))
    resolved = Path(os.path.realpath(lexical))
    return (lexical,) if resolved == lexical else (lexical, resolved)


def coder_test_invocation(config: AgentLoopConfig, agent: str = "claude") -> str | None:
    """The exact test invocation for a coder turn run by ``agent`` (Claude in sandboxed mode)."""
    from .test_runtime import resolve_coder_test_invocation

    return resolve_coder_test_invocation(config, agent=agent)


def role_permission_args(config: AgentLoopConfig, provider: str, role: str | None) -> tuple[str, ...]:
    """CLI permission args for one invocation; empty outside sandboxed mode."""
    if not is_sandboxed(config):
        return ()
    permission_class = permission_class_for_role(role)
    root = sandboxed_response_root(config)
    if provider == "claude":
        prefix = inspect_prefix(config)
        if permission_class == PERMISSION_CLASS_READ_ONLY:
            return (
                "--restricted",
                "--tools",
                CLAUDE_READ_ONLY_TOOLS,
                "--strict-mcp-config",
                "--mcp-config",
                EMPTY_MCP_CONFIG,
                "--permission-mode",
                "dontAsk",
                "--permission-prompts",
                "none",
                "--add-dir",
                str(root),
                # Deny before allow: the allow list cannot withdraw an
                # auto-approved command, an explicit deny can.
                "--disallowedTools",
                *CLAUDE_READ_ONLY_DENIED_TOOLS,
                "--allowedTools",
                _claude_path_rule("Write", root),
                _claude_path_rule("Edit", root),
                f"Bash({prefix} *)",
            )
        # --allowedTools is an exclusive allowlist here and acceptEdits only
        # decides whether an *allowed* edit prompts, so the assigned checkout
        # needs its own path-scoped Write/Edit rules (#1077).
        rules = [
            rule
            for checkout in coder_checkout_paths(config)
            for rule in (_claude_path_rule("Write", checkout), _claude_path_rule("Edit", checkout))
        ]
        rules += ["Bash(git *)", *(f"Bash({sub} *)" for sub in CODER_GH_SUBCOMMANDS), f"Bash({prefix} *)"]
        test_invocation = coder_test_invocation(config, provider)
        if test_invocation:
            rules.append(f"Bash({test_invocation})")
        return (
            "--permission-mode",
            "acceptEdits",
            "--permission-prompts",
            "none",
            "--setting-sources",
            "user",
            "--strict-mcp-config",
            "--mcp-config",
            EMPTY_MCP_CONFIG,
            "--add-dir",
            str(root),
            "--allowedTools",
            *rules,
        )
    if provider == "codex":
        if permission_class == PERMISSION_CLASS_CODER:
            raise AgentLoopError(
                "--agent-permissions sandboxed cannot run Codex as a committing coder "
                "(its sandbox keeps .git read-only); use --coder claude."
            )
        return ("--sandbox", "read-only", "-c", 'approval_policy="never"')
    raise AgentLoopError(
        f"--agent-permissions sandboxed supports only Claude and Codex, not {provider!r}: "
        f"{_unsupported_provider_reason(provider)}."
    )


def role_permission_env(config: AgentLoopConfig, provider: str, role: str | None) -> dict[str, str]:
    """Extra agent subprocess environment for a role; empty outside sandboxed mode.

    The agent process otherwise keeps its inherited environment: the closed
    allowlist, constructed PATH and pinned executables apply inside
    ``inspect``, not to the CLI that calls it.  An inherited ``GIT_TRACE*`` or
    ``GIT_CONFIG_*`` therefore still reaches the CLI's own git calls, so those
    are neutralized here.  They are overridden rather than removed because the
    runner merges this mapping over ``os.environ``; git treats ``0`` as off and
    ``GIT_CONFIG_COUNT=0`` makes any inherited ``GIT_CONFIG_KEY_n`` /
    ``GIT_CONFIG_VALUE_n`` pair unreadable.
    """
    if not is_sandboxed(config):
        return {}
    return dict(neutralized_git_env())


def prepare_sandboxed_spawn(config: AgentLoopConfig, provider: str, role: str | None) -> Path:
    """Run every pre-spawn check and return the pre-created response file."""
    path = prepare_response_file(config, provider)
    if provider == "claude" and permission_class_for_role(role) == PERMISSION_CLASS_READ_ONLY:
        verify_inspect_provenance(config)
    return path


def boundary_unavailable_text(provider_signature: str, exc: Exception) -> str:
    payload = {
        "schema_version": 1,
        "kind": "agent_unavailable",
        "retryable": False,
        "category": "environment",
        "summary": f"Sandboxed spawn refused before launch: {exc}",
        "suggested_action": (
            "Inspect the named path, restore it or restart the run so agent-loop re-pins "
            "and re-validates the sandbox boundary."
        ),
    }
    return (
        json.dumps(payload, indent=2)
        + "\n<!-- AGENT_UNAVAILABLE -->\n"
        + f"-- {provider_signature}\n"
    )


# ------------------------------------------------------------- prompts


def inspect_forms(config: AgentLoopConfig, *, base_branch: str | None = None) -> str:
    """Prompt text naming the exact inspect forms the read-only grant allows."""
    provenance = require_inspect_provenance(config)
    prefix = provenance.prefix_text
    diff_target = f"{base_branch}...HEAD" if base_branch else "<base>...HEAD"
    lines = [
        "Sandboxed permissions are active. Your only shell command is the read-only "
        f"inspector, invoked exactly as `{prefix} <tool> <args>` (one command per call, "
        "no pipes, redirection, `&&`, or `;`):",
        f"- `{prefix} git diff {diff_target}` (options: --stat, --name-only, --name-status, --numstat, -p, -U<n>)",
        f"- `{prefix} git show <rev>`, `{prefix} git log -n <k> --oneline`, `{prefix} git status --short`",
        f"- `{prefix} git rev-parse HEAD`, `{prefix} git ls-files`",
    ]
    if provenance.gh is not None:
        lines.append(
            f"- `{prefix} gh issue view <n> --comments`, `{prefix} gh pr view <n>`, "
            f"`{prefix} gh pr diff <n>`, `{prefix} gh pr checks <n>`"
        )
    else:
        lines.append("- gh inspection is unavailable in this run (no gh was found on PATH at startup).")
    lines.append(
        "Use Read, Grep, and Glob for file contents. You may write only the public "
        "response file named below."
    )
    return "\n".join(lines) + "\n"


def coder_sandbox_guidance(config: AgentLoopConfig, agent: str = "claude") -> str:
    invocation = coder_test_invocation(config, agent)
    if invocation:
        test_line = (
            "The only test command you may run is this exact invocation (no edits, "
            f"no extra selectors): `{invocation}`"
        )
    else:
        test_line = (
            "No test command was resolved for this run, so test execution is not granted; "
            "say so in your summary."
        )
    return (
        "Sandboxed permissions are active: you may edit files in your checkout and run "
        "`git`, the gh subcommands this prompt names, and the read-only inspector. "
        f"{test_line}\n"
    )
