"""Registry for supported agent backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentBackend, AgentName
from .claude import BACKEND as CLAUDE_BACKEND
from .codex import BACKEND as CODEX_BACKEND
from ..errors import AgentLoopError
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

BACKENDS: dict[AgentName, AgentBackend] = {
    "claude": CLAUDE_BACKEND,
    "codex": CODEX_BACKEND,
}


def get_backend(agent: AgentName) -> AgentBackend:
    try:
        return BACKENDS[agent]
    except KeyError as exc:
        raise AgentLoopError(f"Unsupported agent: {agent}") from exc


def agent_display_name(agent: AgentName) -> str:
    return get_backend(agent).display_name


def agent_signature(agent: AgentName) -> str:
    return get_backend(agent).signature


def default_agent_args(agent: AgentName, *, dangerous: bool) -> tuple[str, ...]:
    return get_backend(agent).default_args(dangerous=dangerous)


def run_agent(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    result = get_backend(agent).run(runner, config, prompt, session_id=session_id)
    return result.text, result.session_id


def run_claude(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    prompt: str,
    session_id: str | None = None,
) -> tuple[str, str | None]:
    result = CLAUDE_BACKEND.run(runner, config, prompt, session_id=session_id)
    return result.text, result.session_id


def run_codex(runner: Runner, *, config: AgentLoopConfig, prompt: str) -> str:
    return CODEX_BACKEND.run(runner, config, prompt).text
