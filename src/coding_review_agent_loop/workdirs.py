"""Helpers for selecting configured agent work directories."""

from __future__ import annotations

import tempfile
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
        "antigravity": config.antigravity_dir,
    }[agent]


def github_api_cwd() -> Path:
    """Return a directory that always exists and is never an agent checkout.

    ``gh api`` and repo-explicit ``gh ... --repo OWNER/REPO`` reads carry the
    repository in their arguments, so they must not depend on a checkout
    existing. Never use this for ``git``, ``gh pr checkout``, ``gh repo clone``
    or anything that reads working-tree files.
    """
    return Path(tempfile.gettempdir())


def active_workdir(config: AgentLoopConfig) -> Path:
    """Return an initialized checkout that participates in the current loop."""
    return agent_workdir(config, config.coder)
