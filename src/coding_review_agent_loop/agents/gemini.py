"""Gemini CLI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .base import AgentName, AgentResult
from ..logging import agent_log_path, log
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


def _parse_gemini_output(raw: str) -> str:
    """Extract text from Gemini's optional --output-format json response."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            text = data.get("response", raw)
            if isinstance(text, str):
                return text
    except (json.JSONDecodeError, ValueError):
        pass
    return raw


class GeminiBackend:
    name: AgentName = "gemini"
    display_name = "Gemini"
    signature = "Google Gemini"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.gemini_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--yolo",) if dangerous else ()

    def run(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        session_id: str | None = None,
    ) -> AgentResult:
        log_path = agent_log_path(config, "gemini")
        log(config, f"Starting Gemini in {config.gemini_dir}; log: {log_path}")
        result = runner.run_with_log(
            [config.gemini_cmd, "--prompt", prompt, *config.gemini_args],
            cwd=config.gemini_dir,
            log_path=log_path,
            label="Gemini",
            progress_interval_seconds=config.progress_interval_seconds,
        )
        log(config, f"Gemini finished; log: {log_path}")
        return AgentResult(text=_parse_gemini_output(result.stdout), session_id=session_id)


BACKEND = GeminiBackend()
