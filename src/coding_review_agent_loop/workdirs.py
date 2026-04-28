"""Helpers for selecting configured agent work directories."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .agents.base import AgentName

if TYPE_CHECKING:
    from .config import AgentLoopConfig


def agent_workdir(config: AgentLoopConfig, agent: AgentName) -> Path:
    return {
        "claude": config.claude_dir,
        "codex": config.codex_dir,
        "gemini": config.gemini_dir,
    }[agent]


def active_workdir(config: AgentLoopConfig) -> Path:
    """Return an initialized checkout that participates in the current loop."""
    return agent_workdir(config, config.coder)
