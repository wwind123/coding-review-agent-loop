"""Gemini CLI backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .base import AgentName, AgentResult
from ..logging import agent_log_path, log
from ..protocol import CLARIFY_RE, STATE_RE
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


def _strip_gemini_preamble(raw: str) -> str:
    """Drop Gemini CLI diagnostics that can appear before the final response."""
    marker_matches = [*STATE_RE.finditer(raw), *CLARIFY_RE.finditer(raw)]
    if not marker_matches:
        return raw

    public_end = max(match.start() for match in marker_matches)
    separator = "\n---\n"
    separator_at = raw.find(separator, 0, public_end)
    if separator_at == -1:
        return raw

    return raw[separator_at + len(separator) :].lstrip("\n")


def _parse_gemini_output(raw: str) -> tuple[str, str | None]:
    """Extract (text, session_id) from Gemini's optional --output-format json response."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            text = data.get("response", raw)
            if not isinstance(text, str):
                text = raw
            session_id = data.get("session_id")
            return text, session_id if isinstance(session_id, str) else None
    except (json.JSONDecodeError, ValueError):
        pass
    return _strip_gemini_preamble(raw), None



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
        args = [config.gemini_cmd, "--prompt", prompt, *config.gemini_args]
        if session_id:
            args += ["--resume", session_id]
        result = runner.run_with_log(
            args,
            cwd=config.gemini_dir,
            log_path=log_path,
            label="Gemini",
            progress_interval_seconds=config.progress_interval_seconds,
        )
        log(config, f"Gemini finished; log: {log_path}")
        text, new_session_id = _parse_gemini_output(result.stdout)
        return AgentResult(text=text, session_id=new_session_id)


BACKEND = GeminiBackend()
