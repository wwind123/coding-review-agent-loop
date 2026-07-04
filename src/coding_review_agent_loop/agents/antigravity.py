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

from dataclasses import replace
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

from .base import (
    AgentName,
    AgentResult,
    AgentTextSource,
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from ..errors import AgentLoopError
from ..logging import agent_log_path, log
from ..protocol import PUBLIC_RESPONSE_MARKER
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


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


def _antigravity_settings_path() -> Path:
    """Return ~/.gemini/antigravity-cli/settings.json (patchable in tests)."""
    return Path.home() / ".gemini" / "antigravity-cli" / "settings.json"


def _strip_public_response_marker(raw: str) -> tuple[str, AgentTextSource]:
    if PUBLIC_RESPONSE_MARKER not in raw:
        return raw, "stdout"
    return raw.rsplit(PUBLIC_RESPONSE_MARKER, 1)[1].lstrip("\n"), "stdout_marker"


class AntigravityBackend:
    name: AgentName = "antigravity"
    display_name = "Antigravity"
    signature = "Google Antigravity"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.antigravity_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--dangerously-skip-permissions",) if dangerous else ()

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
        log_path_override: Path | None = None,
    ) -> AgentResult:
        import json, fcntl  # Unix-only (fcntl); imported here so the module loads on Windows
        for i, model in enumerate(config.antigravity_models):
            response_path = public_response_path(config, "antigravity")
            prompt_text = _with_public_response_marker_instruction(
                with_public_response_file_instruction(prompt, response_path)
            )
            args = [config.antigravity_cmd, "--model", model, *config.antigravity_args]
            if role in {"reviewer", "repair"}:
                args = [a for a in args if a != "--dangerously-skip-permissions"]
            # agy resumes by conversation id (not gemini's --resume). agy --print does
            # not surface a conversation id in plain output, so in practice session_id
            # is None and turns are single-shot; honor it if a caller ever supplies one.
            if session_id:
                args += ["--conversation", session_id]
            # The prompt is the value of --print (must be last), not a positional.
            args += ["--print", prompt_text]
            log_path = log_path_override or agent_log_path(
                config, "antigravity", run_id=run_id, label=label
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
            settings_path = _antigravity_settings_path()
            settings_lock_path = settings_path.with_suffix(".json.lock")
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            settings_lock = settings_lock_path.open("a+")
            try:
                fcntl.flock(settings_lock, fcntl.LOCK_EX)
                if role in {"reviewer", "repair"}:
                    try:
                        original_settings_text = settings_path.read_text(encoding="utf-8")
                        try:
                            existing_settings = json.loads(original_settings_text)
                        except json.JSONDecodeError as exc:
                            raise AgentLoopError(f"settings.json malformed: {exc}") from exc
                        if not isinstance(existing_settings, dict):
                            raise AgentLoopError("settings.json root is not a JSON object")
                    except FileNotFoundError:
                        original_settings_text = None
                        existing_settings = {}
                    try:
                        injection = (
                            _REPAIR_SETTINGS_INJECTION
                            if role == "repair"
                            else _REVIEWER_SETTINGS_INJECTION
                        )
                        injected = {**existing_settings, **injection}
                        settings_path.write_text(json.dumps(injected, indent=2), encoding="utf-8")
                        # Inner: GEMINI.md lock
                        gemini_lock_file = gemini_lock_path.open("a+")
                        try:
                            fcntl.flock(gemini_lock_file, fcntl.LOCK_EX)
                            try:
                                existing_gemini = gemini_md_path.read_text(encoding="utf-8")
                            except FileNotFoundError:
                                existing_gemini = None
                            gemini_md_path.write_text(
                                single_shot_instruction + (existing_gemini or ""),
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
                                if gemini_md_path.exists():
                                    current = gemini_md_path.read_text(encoding="utf-8")
                                    if current.startswith(single_shot_instruction):
                                        remainder = current[len(single_shot_instruction):]
                                        if remainder:
                                            gemini_md_path.write_text(remainder, encoding="utf-8")
                                        else:
                                            gemini_md_path.unlink()
                        finally:
                            fcntl.flock(gemini_lock_file, fcntl.LOCK_UN)
                            gemini_lock_file.close()
                            if role == "repair":
                                gemini_lock_path.unlink(missing_ok=True)
                    finally:
                        if original_settings_text is None:
                            settings_path.unlink(missing_ok=True)
                        else:
                            settings_path.write_text(original_settings_text, encoding="utf-8")
                else:
                    # Coder path: hold settings lock without modifying settings.json
                    gemini_lock_file = gemini_lock_path.open("a+")
                    try:
                        fcntl.flock(gemini_lock_file, fcntl.LOCK_EX)
                        try:
                            existing_gemini = gemini_md_path.read_text(encoding="utf-8")
                        except FileNotFoundError:
                            existing_gemini = None
                        gemini_md_path.write_text(
                            single_shot_instruction + (existing_gemini or ""),
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
                                timeout_seconds=timeout_seconds,
                            )
                        finally:
                            if gemini_md_path.exists():
                                current = gemini_md_path.read_text(encoding="utf-8")
                                if current.startswith(single_shot_instruction):
                                    remainder = current[len(single_shot_instruction):]
                                    if remainder:
                                        gemini_md_path.write_text(remainder, encoding="utf-8")
                                    else:
                                        gemini_md_path.unlink()
                    finally:
                        fcntl.flock(gemini_lock_file, fcntl.LOCK_UN)
                        gemini_lock_file.close()
            finally:
                fcntl.flock(settings_lock, fcntl.LOCK_UN)
                settings_lock.close()
            log(config, f"Antigravity ({model}) finished; log: {log_path}")

            if result.returncode != 0:
                stdout_lower = result.stdout.lower()
                if any(sig.lower() in stdout_lower for sig in config.antigravity_quota_signatures):
                    if i + 1 < len(config.antigravity_models):
                        log(config, f"Antigravity ({model}) hit quota exhaustion, falling back to next model.")
                        continue

            # Prefer the public response file the prompt asks the agent to write; else
            # keep only what follows the public-response marker in stdout; else stdout.
            response_file_text = read_public_response_file(response_path)
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
                # The model we requested is the model that ran (single-shot, no
                # server-side substitution); the signature stamps it (#332). #333's
                # fallback chain will override this with the model that answered.
                model_used=model,
            )
        
        # This point should not be reached since antigravity_models cannot be empty
        raise RuntimeError("No antigravity models available to run.")

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
