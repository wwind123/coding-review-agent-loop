"""Antigravity CLI (`agy`) backend.

Migration path for Gemini CLI consumer users (#215): Google is retiring Gemini CLI
consumer access (free / AI Pro / Ultra) on 2026-06-18 in favor of Antigravity. The
`agy` CLI is an autonomous, Claude-Code-style coding agent rather than a plain
prompt->text tool, which drives two integration requirements:

* **PTY.** `agy --print` detects whether stdout is a terminal and silently drops its
  final response under a pipe / file / subprocess (upstream antigravity-cli issue
  #76). We run it attached to a pseudo-terminal (``use_pty=True``) so it emits
  normally, then strip ANSI control sequences from the captured output.
* **Marker.** As an agent, `agy` narrates tool use before its answer. We ask it to
  print a sentinel line immediately before the publishable response and keep only
  what follows it (the same convention the Gemini backend uses).

The prompt is the *value* of the ``--print`` flag (``--prompt`` is its alias), not a
trailing positional argument, so it must come last after the other flags.

Limitations this increment: `agy --print` does not surface a conversation id in its
plain-text output, so turns are single-shot (no cross-round session resume), and it
emits no token usage (usage falls back to the estimated path).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import re
import tempfile
from typing import TYPE_CHECKING

from .base import (
    AgentName,
    AgentResult,
    AgentTextSource,
    STDIN_PROMPT_THRESHOLD_BYTES,
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from ..errors import AgentLoopError
from ..logging import agent_log_path, log
from ..protocol import PUBLIC_RESPONSE_MARKER
from ..runner import CommandResult, Runner, executable_identity_changed, strip_ansi
from ..workdir_guard import (
    WorkdirReplayEvidence,
    WorkdirSnapshot,
    capture_workdir_snapshot,
    gate_workdir_replay,
)

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


@dataclass
class AntigravityAttemptState:
    """Shared retry-before-fallback policy for normal Antigravity invocations."""

    models: tuple[str, ...]
    retries_remaining: int
    model_index: int = 0
    attempts: int = 0

    @classmethod
    def from_config(cls, config: "AgentLoopConfig", retries: int) -> "AntigravityAttemptState":
        return cls(config.antigravity_models, retries)

    def singleton_config(self, config: "AgentLoopConfig") -> "AgentLoopConfig":
        return replace(config, antigravity_model=None, antigravity_models=(self.models[self.model_index],))

    def next_after_failure(self, *, retryable: bool, provider_capacity: bool) -> str:
        """Return retry, fallback, or stop; retries are chain-wide."""
        self.attempts += 1
        if retryable and self.retries_remaining:
            self.retries_remaining -= 1
            return "retry"
        if provider_capacity and self.model_index + 1 < len(self.models):
            self.model_index += 1
            return "fallback"
        return "stop"


def _git_lock_path(workdir: Path) -> Path:
    """Return GEMINI.md.lock inside the git metadata dir, following linked-worktree .git files.

    In a linked worktree .git is a file containing ``gitdir: <relative-path>``.
    mkdir-ing that path would raise NotADirectoryError.  We resolve the pointer
    and place the lock in the real git dir so git add never sees it.  If no git
    dir can be found (test tmp dirs without a .git) we create .git/ and use it.
    """
    git_path = workdir / ".git"
    if git_path.is_dir():
        return git_path / "GEMINI.md.lock"
    if git_path.is_file():
        content = git_path.read_text(encoding="utf-8", errors="replace").strip()
        if content.startswith("gitdir:"):
            gitdir = (workdir / content[len("gitdir:"):].strip()).resolve()
            if gitdir.is_dir():
                return gitdir / "GEMINI.md.lock"
    # No .git directory yet (e.g. test tmp dirs) — create it.
    git_path.mkdir(parents=True, exist_ok=True)
    return git_path / "GEMINI.md.lock"


def _with_public_response_marker_instruction(prompt: str) -> str:
    return f"""{prompt}

IMPORTANT FOR ANTIGRAVITY (agy) OUTPUT FILTERING:

As an agent you may print planning narration, tool-use status, or diagnostics
before your final answer.

When you are ready to provide the response that should be posted publicly to
GitHub, print this exact line immediately before it:

{PUBLIC_RESPONSE_MARKER}

Only content after that line will be posted to GitHub. Run any verification
steps (tests, file inspection) before you are ready to finalize. When you are
ready to submit your review: print this marker, output the structured JSON
response, then end your turn immediately — no further tool calls, narration, or
output after the response. If background work is still pending, print the marker
and response now without waiting; do not defer to a background task result. A
turn that ends without printing this marker results in an empty response and a
review failure.
"""


_REVIEWER_SETTINGS_INJECTION = {
    "toolPermission": "strict",
    "permissions": {
        "allow": [
            "command(ls)",
            "command(cat)",
            "command(head)",
            "command(git diff)",
            "command(git show)",
            "command(git status)",
            "command(git log)",
            "command(rg)",
            "command(sed)",
        ]
    },
}
_OVERSIZED_PROMPT_DIRECTIVE = (
    "Follow the complete task included in the Agent Loop Task section of GEMINI.md."
)
_REPAIR_SETTINGS_INJECTION = {
    "toolPermission": "strict",
    "permissions": {"allow": []},
}

_REPAIR_GEMINI_MD = """# Agent Loop Format Repair

This is a single-shot formatting-only repair task in an isolated temporary directory.
Do not inspect files, run shell commands or tests, mutate any repository, start
background tasks, or invoke subagents. Use only the malformed response and protocol
examples in the prompt. Return the repaired response immediately.

"""

# `agy --print` normally emits a small amount of startup chrome before it can
# reach the Node launcher. Replacement replay is safe only when every residual
# line is one of these known startup/loader diagnostics; narration and tool
# output are treated as progress.
_ANTIGRAVITY_STARTUP_LINE_RE = re.compile(
    r"(?ix)^(?:"
    r"(?:agy|antigravity)(?:\s+cli)?(?:\s+(?:v|version)\s*[\w.-]+)?|"
    r"(?:starting|initiali[sz]ing|loading)\s+(?:agy|antigravity|gemini)(?:\s+cli)?|"
    r"(?:using|selected|requested)\s+model\s*[:=].+|"
    r"model\s*[:=].+|"
    r"version\s*[:=]\s*[\w.-]+|"
    r"loaded\s+(?:configuration|credentials|settings)\b.*|"
    r"(?:checking|applying)\s+(?:for\s+)?(?:agy|antigravity|gemini\s+)?updates?\b.*"
    r")$"
)
_ANTIGRAVITY_LOADER_LINE_RE = re.compile(
    r"(?ix)(?:"
    r"MODULE_NOT_FOUND|cannot\s+find\s+module|"
    r"^node:(?:internal/)?modules/.*|^at\s+.+node:(?:internal/)?modules/.*|"
    r"^throw\s+err;?$|^require\s+stack:.*|"
    r"^code\s*[:=]\s*['\"]?(?:MODULE_NOT_FOUND|ENOENT)['\"]?\s*$|"
    r"(?:ENOENT|ENOEXEC)|exec(?:ution)?\s+format\s+error|"
    r"no\s+such\s+file\s+or\s+directory|"
    r"(?:agy|antigravity|gemini|node|npm|launcher|executable|binary).*(?:not\s+found|missing|failed|error)|"
    r"(?:failed|error).*(?:launch|launcher|startup|updat(?:e|er)|replacement)|"
    r"(?:updat(?:e|er)|launcher|replacement).*(?:failed|error|unavailable|in\s+progress)"
    r")"
)
_ANTIGRAVITY_DIAGNOSTIC_MAX_LINES = 40
_ANTIGRAVITY_DIAGNOSTIC_MAX_CHARS = 8192


def _agy_structured_payload(raw: str) -> bool:
    for candidate in (raw, *raw.splitlines()):
        if not candidate.strip():
            continue
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        return True
    return False


def _agy_startup_or_loader_diagnostics(raw: str) -> bool:
    normalized = strip_ansi(raw).strip()
    if not normalized:
        return True
    if len(normalized) > _ANTIGRAVITY_DIAGNOSTIC_MAX_CHARS:
        return False
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) > _ANTIGRAVITY_DIAGNOSTIC_MAX_LINES:
        return False
    return all(
        _ANTIGRAVITY_STARTUP_LINE_RE.fullmatch(line) is not None
        or _ANTIGRAVITY_LOADER_LINE_RE.search(line) is not None
        for line in lines
    )


def classify_antigravity_executable_replacement_interruption(
    result: CommandResult,
    *,
    command: str,
    response_file_text: str | None,
    before_snapshot: WorkdirSnapshot | None = None,
    after_snapshot: WorkdirSnapshot | None = None,
) -> WorkdirReplayEvidence | None:
    """Classify a quiet `agy` launcher replacement with fail-closed gates."""
    observation = result.observation
    if (
        type(result.returncode) is not int
        or observation is None
        or observation.interrupted
        or result.capture_diagnostics
        or response_file_text
        or PUBLIC_RESPONSE_MARKER in result.stdout
        or _agy_structured_payload(result.stdout)
        or not _agy_startup_or_loader_diagnostics(result.stdout)
    ):
        return None
    if not executable_identity_changed(
        observation.before,
        observation.after,
        command=command,
        spawn_wall_time=observation.spawn_wall_time,
        exit_wall_time=observation.spawn_wall_time + observation.elapsed_seconds,
    ):
        return None
    reason = "Antigravity executable changed during invocation"
    if before_snapshot is None and after_snapshot is None:
        return WorkdirReplayEvidence(reason)
    return gate_workdir_replay(
        "Antigravity executable replacement",
        reason,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )


classify_executable_replacement_interruption = (
    classify_antigravity_executable_replacement_interruption
)


def _antigravity_settings_path() -> Path:
    """Return ~/.gemini/antigravity-cli/settings.json (patchable in tests)."""
    return Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


def _strip_public_response_marker(raw: str) -> tuple[str, AgentTextSource]:
    if PUBLIC_RESPONSE_MARKER not in raw:
        return raw, "stdout"
    return raw.rsplit(PUBLIC_RESPONSE_MARKER, 1)[1].lstrip("\n"), "stdout_marker"


def _parse_model_catalog(raw: str) -> set[str]:
    """Extract model labels from the human-readable ``agy models`` output."""
    models: set[str] = set()
    for line in strip_ansi(raw).splitlines():
        value = line.strip().lstrip("-*• ").strip()
        if not value or set(value) <= {"-", "=", "_"}:
            continue
        lowered = value.lower().rstrip(":")
        if lowered in {"models", "available models", "available model", "name", "model"}:
            continue
        if lowered.startswith(("error:", "warning:", "usage:", "command:")):
            continue
        models.add(value)
    return models


class AntigravityBackend:
    name: AgentName = "antigravity"
    display_name = "Antigravity"
    signature = "Google Antigravity"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.antigravity_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--dangerously-skip-permissions",) if dangerous else ()

    def discover_models(
        self, runner: Runner, config: AgentLoopConfig, *, timeout_seconds: float
    ) -> tuple[set[str] | None, str]:
        """Query agy's model catalog once, preserving the PTY requirement."""
        repair_root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"
        repair_root.mkdir(parents=True, exist_ok=True)
        log_path = agent_log_path(config, "antigravity-repair-models")
        with tempfile.TemporaryDirectory(prefix="agy-catalog-", dir=repair_root) as temp_dir:
            try:
                result = runner.run_with_log(
                    [config.antigravity_cmd, *config.antigravity_args, "models"],
                    cwd=Path(temp_dir),
                    log_path=log_path,
                    label="Antigravity model catalog",
                    progress_interval_seconds=config.progress_interval_seconds,
                    check=False,
                    env={"AGENT_LOOP_WORKDIR": str(Path(temp_dir).resolve())},
                    use_pty=True,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                return None, str(exc)
        if result.returncode != 0:
            return None, result.stdout or result.stderr or f"agy models exited with {result.returncode}"
        models = _parse_model_catalog(result.stdout)
        if not models:
            return None, "agy models returned no parseable model choices"
        return models, result.stdout

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
        log_path_override: Path | None = None,
    ) -> AgentResult:
        import json, fcntl  # Unix-only (fcntl); imported here so the module loads on Windows
        # Model traversal belongs to AntigravityAttemptState, owned by the
        # caller.  This backend deliberately executes exactly one model.
        model = config.antigravity_models[0]
        response_path = public_response_path(config, "antigravity")
        response_path.unlink(missing_ok=True)
        prompt_text = _with_public_response_marker_instruction(
            with_public_response_file_instruction(prompt, response_path)
        )
        oversized_prompt = len(prompt_text.encode("utf-8")) > STDIN_PROMPT_THRESHOLD_BYTES
        args = [
            config.antigravity_cmd,
            "--model",
            model,
            "--print-timeout",
            f"{config.antigravity_print_timeout_seconds}s",
            *config.antigravity_args,
        ]
        if role in {"reviewer", "repair"}:
            args = [a for a in args if a != "--dangerously-skip-permissions"]
        # agy resumes by conversation id (not gemini's --resume). agy --print does
        # not surface a conversation id in plain output, so in practice session_id
        # is None and turns are single-shot; honor it if a caller ever supplies one.
        if session_id:
            args += ["--conversation", session_id]
        # The prompt is the value of --print (must be last), not a positional.
        args += ["--print", _OVERSIZED_PROMPT_DIRECTIVE if oversized_prompt else prompt_text]
        log_path = log_path_override or agent_log_path(
            config, "antigravity", run_id=run_id, label=label, attempt_suffix=attempt_suffix
        )
        log(config, f"Starting Antigravity (model: {model}) in {config.antigravity_dir}; log: {log_path}; response: {response_path}")
        # agy reads GEMINI.md from the workdir as high-priority system context before
        # the model sees the prompt. Injecting a single-shot session rule prevents
        # agy from spawning background execution tasks (which cause it to end the
        # --print turn without a response). Cleanup strips only the injected prefix
        # rather than restoring a snapshot, so file changes made during a coder turn
        # are preserved. An exclusive flock on the lock file (resolved by
        # _git_lock_path into git metadata, not the worktree) serializes the entire
        # inject→run→strip sequence across concurrent processes sharing the same
        # default per-repo workdir.
        gemini_md_path = config.antigravity_dir / "GEMINI.md"
        gemini_lock_path = (
            config.antigravity_dir.parent / f".{config.antigravity_dir.name}.GEMINI.md.lock"
            if role == "repair"
            else _git_lock_path(config.antigravity_dir)
        )
        if role == "repair":
            single_shot_instruction = _REPAIR_GEMINI_MD
        else:
            resolved_base = (config.base or "").strip()
            diff_cmd = (
                f"`git diff {resolved_base}...HEAD`"
                if resolved_base
                else "`git diff <base>...HEAD` (replace `<base>` with the resolved base branch)"
            )
            single_shot_instruction = (
                "# Agent Loop Single-Shot Session\n\n"
                "You are running in a single-shot, non-interactive `agy --print` session"
                " invoked by an automated orchestrator. There will be no follow-up turns.\n\n"
                "**Do NOT spawn background execution tasks or subagents under any"
                " circumstances.**\n\n"
                "**For code review tasks: DO NOT run tests, builds, compilation,"
                " mutation, commits, background work, or unrelated discovery commands.**"
                f" You may use only the strict allow-listed read-only commands to inspect"
                " the assigned checkout and local PR diff. Prefer "
                f"{diff_cmd}, `git show`, `git status`, `git log`, `rg`, `sed`, and"
                " direct file reads over web search. Do not fetch, checkout, reset, clean,"
                " or write files. Tests are CI's responsibility; your job is to read code"
                " and identify issues. If you find yourself about to run `pytest`, `npm"
                " test`, `go test`, or any build command, stop and write your review from"
                " code inspection alone.\n\n"
                "For non-review tasks that require shell commands, run them synchronously"
                " in this same turn before writing your response.\n\n"
                "---\n\n"
            )
        injected_gemini_prefix = single_shot_instruction
        if oversized_prompt:
            injected_gemini_prefix += (
                "# Agent Loop Task\n\n"
                f"{prompt_text}\n\n"
                "---\n\n"
            )
        settings_path = _antigravity_settings_path()
        settings_lock_path = settings_path.with_suffix(".json.lock")
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_lock = settings_lock_path.open("a+")
        before_snapshot: WorkdirSnapshot | None = None
        after_snapshot: WorkdirSnapshot | None = None
        candidate: WorkdirReplayEvidence | None = None
        response_file_text: str | None = None
        result: CommandResult | None = None

        def strip_injected_prefix() -> None:
            if not gemini_md_path.exists():
                return
            current = gemini_md_path.read_text(encoding="utf-8")
            if current.startswith(injected_gemini_prefix):
                remainder = current[len(injected_gemini_prefix):]
                if remainder:
                    gemini_md_path.write_text(remainder, encoding="utf-8")
                else:
                    gemini_md_path.unlink()

        settings_was_injected = role in {"reviewer", "repair"}
        original_settings_text: str | None = None
        settings_restore_required = False
        try:
            fcntl.flock(settings_lock, fcntl.LOCK_EX)
            existing_settings: dict[str, object] = {}
            if settings_was_injected:
                try:
                    original_settings_text = settings_path.read_text(encoding="utf-8")
                    try:
                        parsed_settings = json.loads(original_settings_text)
                    except json.JSONDecodeError as exc:
                        raise AgentLoopError(f"settings.json malformed: {exc}") from exc
                    if not isinstance(parsed_settings, dict):
                        raise AgentLoopError("settings.json root is not a JSON object")
                    existing_settings = parsed_settings
                except FileNotFoundError:
                    original_settings_text = None
                injection = (
                    _REPAIR_SETTINGS_INJECTION
                    if role == "repair"
                    else _REVIEWER_SETTINGS_INJECTION
                )
                injected = {**existing_settings, **injection}
                # Mark this before writing: a partial write that raises still
                # needs the original settings restored in the outer finally.
                settings_restore_required = True
                settings_path.write_text(json.dumps(injected, indent=2), encoding="utf-8")

            # Hold both locks across the complete normal-run lifecycle. Repair
            # uses an isolated temporary directory and intentionally performs no
            # checkout probes or replacement classification.
            gemini_lock_file = gemini_lock_path.open("a+")
            try:
                fcntl.flock(gemini_lock_file, fcntl.LOCK_EX)
                if role != "repair":
                    before_snapshot = capture_workdir_snapshot(
                        runner,
                        config.antigravity_dir,
                        tolerate_exceptions=True,
                    )
                try:
                    try:
                        existing_gemini = gemini_md_path.read_text(encoding="utf-8")
                    except FileNotFoundError:
                        existing_gemini = None
                    gemini_md_path.write_text(
                        injected_gemini_prefix + (existing_gemini or ""),
                        encoding="utf-8",
                    )
                    try:
                        result = runner.run_with_log(
                            args,
                            cwd=config.antigravity_dir,
                            log_path=log_path,
                            label=f"Antigravity ({model})",
                            progress_interval_seconds=config.progress_interval_seconds,
                            check=False,
                            env={"AGENT_LOOP_WORKDIR": str(config.antigravity_dir.resolve())},
                            use_pty=True,
                            timeout_seconds=(
                                config.repair_timeout_seconds
                                if role == "repair"
                                else timeout_seconds
                            ),
                        )
                    finally:
                        strip_injected_prefix()
                finally:
                    # Parse only after cleanup, while both locks are still held.
                    if role != "repair" and result is not None:
                        response_file_text = read_public_response_file(response_path)
                        candidate = classify_antigravity_executable_replacement_interruption(
                            result,
                            command=config.antigravity_cmd,
                            response_file_text=response_file_text,
                        )
                        if candidate is not None:
                            after_snapshot = capture_workdir_snapshot(
                                runner,
                                config.antigravity_dir,
                                tolerate_exceptions=True,
                            )
                            candidate = classify_antigravity_executable_replacement_interruption(
                                result,
                                command=config.antigravity_cmd,
                                response_file_text=response_file_text,
                                before_snapshot=before_snapshot,
                                after_snapshot=after_snapshot,
                            )
            finally:
                fcntl.flock(gemini_lock_file, fcntl.LOCK_UN)
                gemini_lock_file.close()
                if role == "repair":
                    gemini_lock_path.unlink(missing_ok=True)
        finally:
            if settings_restore_required:
                if original_settings_text is None:
                    settings_path.unlink(missing_ok=True)
                else:
                    settings_path.write_text(original_settings_text, encoding="utf-8")
            fcntl.flock(settings_lock, fcntl.LOCK_UN)
            settings_lock.close()
        log(config, f"Antigravity ({model}) finished; log: {log_path}")

        # Prefer the public response file the prompt asks the agent to write; else
        # keep only what follows the public-response marker in stdout; else stdout.
        if role == "repair":
            response_file_text = read_public_response_file(response_path)
        assert result is not None
        if response_file_text is not None:
            message_text, text_source = response_file_text, "response_file"
        else:
            message_text, text_source = _strip_public_response_marker(result.stdout)
        return AgentResult(
            text=message_text,
            raw_output=result.stdout,
            text_source=text_source,
            response_file_text=response_file_text,
            response_file_path=response_path,
            message_text=message_text,
            session_id=None,
            log_path=log_path,
            returncode=result.returncode,
            usage=None,
            raw_usage=None,
            # This single-model backend reports the model it requested; the
            # caller's attempt state advances to a fallback model if needed.
            model_used=model,
            command_result=result,
            self_update_reason=candidate.reason if candidate else None,
            self_update_replay_refusal_kind=(
                candidate.replay_refusal_kind if candidate else None
            ),
            self_update_replay_refusal_detail=(
                candidate.replay_refusal_detail if candidate else None
            ),
        )

    def run_repair(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        *,
        model: str,
        run_id: str | None = None,
        log_path: Path | None = None,
    ) -> AgentResult:
        """Run one isolated, no-tools Antigravity format-repair attempt."""
        repair_root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"
        repair_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="agy-", dir=repair_root) as temp_dir:
            workdir = Path(temp_dir)
            repair_config = replace(
                config,
                antigravity_dir=workdir,
                antigravity_model=None,
                antigravity_models=(model,),
            )
            return self.run(
                runner,
                repair_config,
                prompt,
                run_id=run_id,
                role="repair",
                log_path_override=log_path,
            )


BACKEND = AntigravityBackend()
