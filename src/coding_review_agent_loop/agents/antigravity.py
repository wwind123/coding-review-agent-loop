"""Antigravity CLI (`agy`) backend.

Migration path for Gemini CLI consumer users (#215): Google is retiring Gemini CLI
consumer access (free / AI Pro / Ultra) on 2026-06-18 in favor of Antigravity. The
`agy` CLI is Claude-Code-style (plain-text `--print` output, `--model`,
`--dangerously-skip-permissions`, `--conversation` resume), so this backend mirrors
the Claude backend rather than the Gemini one.

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
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from ..logging import agent_log_path, log
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


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
        args = [config.antigravity_cmd, "--print", "--model", config.antigravity_model, *config.antigravity_args]
        # agy resumes by conversation id (not gemini's --resume). agy --print does
        # not surface a conversation id in plain output, so in practice session_id
        # is None and turns are single-shot; honor it if a caller ever supplies one.
        if session_id:
            args += ["--conversation", session_id]
        args.append(with_public_response_file_instruction(prompt, response_path))
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
        )
        log(config, f"Antigravity finished; log: {log_path}")
        # agy --print emits plain text (no JSON wrapper). Prefer the public response
        # file the prompt asks the agent to write; fall back to stdout.
        response_file_text = read_public_response_file(response_path)
        return AgentResult(
            text=response_file_text or result.stdout,
            raw_output=result.stdout,
            text_source="response_file" if response_file_text is not None else "stdout",
            response_file_text=response_file_text,
            message_text=result.stdout,
            session_id=None,
            log_path=log_path,
            returncode=result.returncode,
            usage=None,
            raw_usage=None,
        )


BACKEND = AntigravityBackend()
