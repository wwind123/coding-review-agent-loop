"""OpenAI Codex CLI backend."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .base import AgentName, AgentResult
from ..logging import agent_log_path, log
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


class CodexBackend:
    name: AgentName = "codex"
    display_name = "Codex"
    signature = "OpenAI Codex"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.codex_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--dangerously-bypass-approvals-and-sandbox",) if dangerous else ()

    def run(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        session_id: str | None = None,
    ) -> AgentResult:
        log_path = agent_log_path(config, "codex")
        log(config, f"Starting Codex in {config.codex_dir}; log: {log_path}")
        if config.dry_run:
            result = runner.run(
                [
                    config.codex_cmd,
                    "exec",
                    "--cd",
                    str(config.codex_dir),
                    *config.codex_args,
                    prompt,
                ],
                cwd=config.codex_dir,
            )
            log(config, f"Codex finished; log: {log_path}")
            return AgentResult(text=result.stdout)

        with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as handle:
            output_path = handle.name
        try:
            runner.run_with_log(
                [
                    config.codex_cmd,
                    "exec",
                    "--cd",
                    str(config.codex_dir),
                    "--output-last-message",
                    output_path,
                    *config.codex_args,
                    prompt,
                ],
                cwd=config.codex_dir,
                log_path=log_path,
                label="Codex",
                progress_interval_seconds=config.progress_interval_seconds,
            )
            output = Path(output_path).read_text(encoding="utf-8")
            log(config, f"Codex finished; log: {log_path}")
            return AgentResult(text=output)
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass


BACKEND = CodexBackend()
