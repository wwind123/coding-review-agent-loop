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

from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    AgentName,
    AgentResult,
    AgentTextSource,
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from ..logging import agent_log_path, log
from ..protocol import PUBLIC_RESPONSE_MARKER
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


def _with_public_response_marker_instruction(prompt: str) -> str:
    return f"""{prompt}

IMPORTANT FOR ANTIGRAVITY (agy) OUTPUT FILTERING:

As an agent you may print planning narration, tool-use status, or diagnostics
before your final answer.

When you are ready to provide the response that should be posted publicly to
GitHub, print this exact line immediately before it:

{PUBLIC_RESPONSE_MARKER}

Only content after that line will be posted to GitHub. Do not print the marker
until you are done with all internal reasoning, tool use, and review work.
"""


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
    ) -> AgentResult:
        response_path = public_response_path(config, "antigravity")
        prompt_text = _with_public_response_marker_instruction(
            with_public_response_file_instruction(prompt, response_path)
        )
        args = [config.antigravity_cmd, "--model", config.antigravity_model, *config.antigravity_args]
        # agy resumes by conversation id (not gemini's --resume). agy --print does
        # not surface a conversation id in plain output, so in practice session_id
        # is None and turns are single-shot; honor it if a caller ever supplies one.
        if session_id:
            args += ["--conversation", session_id]
        # The prompt is the value of --print (must be last), not a positional.
        args += ["--print", prompt_text]
        log_path = agent_log_path(config, "antigravity", run_id=run_id)
        log(config, f"Starting Antigravity in {config.antigravity_dir}; log: {log_path}; response: {response_path}")
        result = runner.run_with_log(
            args,
            cwd=config.antigravity_dir,
            log_path=log_path,
            label="Antigravity",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False,
            env={"AGENT_LOOP_WORKDIR": str(config.antigravity_dir.resolve())},
            use_pty=True,
        )
        log(config, f"Antigravity finished; log: {log_path}")
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
            message_text=message_text,
            session_id=None,
            log_path=log_path,
            returncode=result.returncode,
            usage=None,
            raw_usage=None,
            # The model we requested is the model that ran (single-shot, no
            # server-side substitution); the signature stamps it (#332). #333's
            # fallback chain will override this with the model that answered.
            model_used=config.antigravity_model or None,
        )


BACKEND = AntigravityBackend()
