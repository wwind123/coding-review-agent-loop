"""Agent backend protocol and shared types."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os
import shutil
import stat
from typing import TYPE_CHECKING, Callable, Iterator, Literal, Protocol, cast

from ..runner import CommandResult, Runner
from ..containment import ContainmentEvidence
from ..usage import UsageMetadata

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

AgentName = Literal["claude", "codex", "gemini", "antigravity"]
AgentTextSource = Literal["response_file", "stdout_marker", "stdout"]

# Keep prompts below Linux's per-argument exec limit; longer prompts are sent
# through stdin or backend-specific task context instead. Claude and Codex
# always use stdin regardless of size (#870); this bound applies to the
# Gemini, Antigravity, and Gemini-repair directive paths.
STDIN_PROMPT_THRESHOLD_BYTES = 100_000


def normalize_agent_name(value: str) -> AgentName:
    """Normalize input-only agent aliases to their canonical names."""
    return cast(AgentName, "antigravity" if value == "agy" else value)


@dataclass(frozen=True)
class AgentResult:
    text: str
    raw_output: str = ""
    text_source: AgentTextSource = "stdout"
    response_file_text: str | None = None
    response_file_path: Path | None = None
    message_text: str | None = None
    session_id: str | None = None
    log_path: Path | None = None
    returncode: int | None = 0
    usage: UsageMetadata | None = None
    raw_usage: object | None = None
    # Human-readable label of the model that actually produced this result (e.g.
    # "Gemini 3.1 Pro (High)", "claude-sonnet-4-6"). Ground truth for the dynamic
    # signature (#332); None when the backend could not determine it. The
    # antigravity fallback chain (#333) sets this to the model that answered.
    model_used: str | None = None
    # Structured invocation identity.  These fields are optional so persisted
    # and test-created results from older versions remain valid.
    provider: AgentName | None = None
    role: str | None = None
    configured_model: str | None = None
    configured_effort: str | None = None
    effort_source: str | None = None
    observed_model: str | None = None
    observed_effort: str | None = None
    observation_provenance: str | None = None
    command_result: CommandResult | None = None
    # Provider-neutral evidence that the configured executable was replaced or
    # disappeared during this invocation. Claude uses the historical field name;
    # Codex sets it for the same bounded recovery signal. Replay refusal is
    # separate: a candidate can be positively identified while the checkout
    # snapshot makes replay unsafe.
    self_update_reason: str | None = None
    self_update_replay_refusal_kind: str | None = None
    self_update_replay_refusal_detail: str | None = None
    containment: ContainmentEvidence | None = None
    # The coder acquisition turn is an ephemeral authority boundary. The
    # orchestrator snapshots it before format repair can mint a newer broker
    # turn, and uses the snapshot for validation and evidence derivation.
    test_turn_id: str | None = None
    test_turn_observations: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if self.containment is None and self.command_result is not None:
            object.__setattr__(self, "containment", self.command_result.containment)


class AgentBackend(Protocol):
    name: AgentName
    display_name: str
    signature: str

    def workdir(self, config: AgentLoopConfig) -> Path: ...

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]: ...

    def run(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        session_id: str | None = None,
        run_id: str | None = None,
        role: str | None = None,
        label: str | None = None,
        timeout_seconds: float | None = None,
        attempt_suffix: str | None = None,
    ) -> AgentResult: ...


def public_response_path(
    config: AgentLoopConfig,
    agent: AgentName,
    *,
    root: Path | None = None,
) -> Path:
    # Every invocation gets a fresh, exclusively created ``turn-<uuid4 hex>``
    # directory (exactly 0700) holding ``response.md``, so an artifact from an
    # earlier invocation cannot be mistaken for the current response and no
    # peer's response sits beside this one (#1560).  The shared ancestors of
    # the default root are non-listable (0o300).  Sandboxed Claude and Codex
    # spawns use agent_permissions.prepare_response_file instead, which
    # pre-creates the file under the physically validated root.
    from ..agent_permissions import (
        RESPONSE_FILE_NAME,
        allocate_turn_directory,
        default_response_parent,
        ensure_unlistable_dir,
        harden_response_tree,
    )

    component = config.active_reviewer_seat_id or agent
    if root is None:
        parent = default_response_parent(config, component)
    else:
        # A backend-specific root (Gemini's git-dir root) is hardened the same
        # way, except that its git-owned parent keeps its mode.  Worktree
        # removal releases it first (agent_permissions.release_*).
        harden_response_tree(config, root=root, hide_parent=False)
        parent = root / component
        ensure_unlistable_dir(parent, remode_group_writable=True)
    turn_dir = allocate_turn_directory(parent)
    register_turn_directory(turn_dir)
    return turn_dir / RESPONSE_FILE_NAME


# Per-turn directories allocated inside the current validated agent turn.
_TURN_DIRECTORIES: ContextVar[list[Path] | None] = ContextVar(
    "agent_loop_turn_directories", default=None
)


def register_turn_directory(path: Path) -> None:
    """Record a per-turn response directory for cleanup by the enclosing scope."""
    directories = _TURN_DIRECTORIES.get()
    if directories is not None:
        directories.append(path)


def remove_turn_directory(path: Path, *, on_error: Callable[[str], None] | None = None) -> bool:
    """Best-effort removal of one ``turn-*`` directory; never anything else."""
    from ..agent_permissions import TURN_DIR_PREFIX

    if not path.name.startswith(TURN_DIR_PREFIX):
        return False
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        if on_error is not None:
            on_error(f"Could not inspect per-turn response directory {path}: {exc}")
        return False
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    try:
        shutil.rmtree(path)
    except OSError as exc:
        if on_error is not None:
            on_error(f"Could not remove per-turn response directory {path}: {exc}")
        return False
    return True


@contextmanager
def response_turn_scope(
    on_error: Callable[[str], None] | None = None, *, reuse_enclosing: bool = False
) -> Iterator[list[Path]]:
    """Remove every per-turn directory allocated inside the scope when it ends.

    A validated agent turn opens the outer scope, so cleanup runs only after
    acceptance, repair, completion recovery and failure diagnostics have
    finished with ``response_file_path``.  Every backend invocation also runs
    inside a scope with ``reuse_enclosing``: inside a validated turn it defers
    to that turn's scope, and a direct invocation (outside any validated
    turn) gets its own scope, which ends once the backend has captured the
    response text.  A cleanup failure is reported through ``on_error`` and
    never changes the turn's outcome.
    """
    enclosing = _TURN_DIRECTORIES.get()
    if reuse_enclosing and enclosing is not None:
        yield enclosing
        return
    directories: list[Path] = []
    token = _TURN_DIRECTORIES.set(directories)
    try:
        yield directories
    finally:
        _TURN_DIRECTORIES.reset(token)
        for directory in directories:
            remove_turn_directory(directory, on_error=on_error)


def with_public_response_file_instruction(prompt: str, response_path: Path) -> str:
    return f"""{prompt}

PUBLIC RESPONSE FILE:

Write the final public response that should be posted to GitHub to this file:

{response_path}

The orchestrator will post only that file's contents when it exists and is
non-empty. Keep internal tool narration, planning notes, diagnostics, and
scratch output out of that file. Include the required AGENT_STATE / AGENT_PR /
AGENT_CLARIFY markers in the file, as requested above.

For structured responses, the file must start directly with {{, contain exactly
one structured JSON object, then the required footer marker and signature. Do
not put the stdout filtering marker, headings, prose, or markdown code fences in
the response file. For structured plan revisions only, if signed human
requirements are present, place `<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` and the
`### Human requirements` section after the JSON object and before the
`AGENT_PLAN_STATE` footer.
You MUST write this file before your turn ends. A turn that ends without
writing this file results in a review failure.
"""


def with_sandboxed_last_message_instruction(prompt: str) -> str:
    """Sandboxed Codex non-coders cannot write files; the CLI records the reply."""
    return f"""{prompt}

PUBLIC RESPONSE:

Your final message is written verbatim by the Codex CLI to the public response
file and is the only text the orchestrator will post. Make your final message
exactly the public response content: for structured responses it must start
directly with {{, contain exactly one structured JSON object, then the required
footer marker and signature, with no headings, prose, or markdown code fences
around it. Include the required AGENT_STATE / AGENT_PR / AGENT_CLARIFY markers
as requested above. Your sandbox is read-only, so do not try to write files.
"""


def read_public_response_file(response_path: Path) -> str | None:
    # A missing or whitespace-only file is deliberately not an artifact.
    try:
        text = response_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    text = text.strip()
    return text or None
