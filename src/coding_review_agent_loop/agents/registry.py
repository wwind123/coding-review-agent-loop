"""Registry for supported agent backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import AgentBackend, AgentName, AgentResult
from .antigravity import BACKEND as ANTIGRAVITY_BACKEND
from .claude import BACKEND as CLAUDE_BACKEND
from .codex import BACKEND as CODEX_BACKEND
from .gemini import BACKEND as GEMINI_BACKEND
from ..errors import AgentLoopError
from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

BACKENDS: dict[AgentName, AgentBackend] = {
    "claude": CLAUDE_BACKEND,
    "codex": CODEX_BACKEND,
    "gemini": GEMINI_BACKEND,
    "antigravity": ANTIGRAVITY_BACKEND,
}


def get_backend(agent: AgentName) -> AgentBackend:
    try:
        return BACKENDS[agent]
    except KeyError as exc:
        raise AgentLoopError(f"Unsupported agent: {agent}") from exc


def agent_display_name(agent: AgentName) -> str:
    return get_backend(agent).display_name


def _configured_model_label(agent: AgentName, config: AgentLoopConfig | None) -> str | None:
    """The model label declared in config for `agent`, or None if not declared.

    Codex appends its reasoning effort; antigravity's model string already embeds
    effort (e.g. "Gemini 3.1 Pro (High)"), so it is used verbatim.
    """
    if config is None:
        return None
    if agent == "antigravity":
        return config.antigravity_models[0] if config.antigravity_models else None
    if agent == "codex":
        model = config.codex_model
        if not model:
            return None
        effort = config.codex_reasoning_effort
        return f"{model} ({effort})" if effort else model
    if agent == "gemini":
        return config.gemini_model or None
    if agent == "claude":
        return config.claude_model or None
    return None


def agent_signature(
    agent: AgentName,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    """Build the `-- {signature}` label for `agent`.

    Precedence: the model that actually ran (`model_used`, ground truth) > the
    model declared in config > the generic provider signature. When no model is
    known, returns the generic signature unchanged so existing output is stable.
    """
    base = get_backend(agent).signature
    label = model_used or _configured_model_label(agent, config)
    if not label:
        return base
    return f"{base}: {label}"


def default_agent_args(agent: AgentName, *, dangerous: bool) -> tuple[str, ...]:
    return get_backend(agent).default_args(dangerous=dangerous)


def run_agent_result(
    runner: Runner,
    *,
    agent: AgentName,
    config: AgentLoopConfig,
    prompt: str,
    session_id: str | None = None,
    run_id: str | None = None,
    role: str | None = None,
    label: str | None = None,
    timeout_seconds: float | None = None,
) -> AgentResult:
    return get_backend(agent).run(
        runner,
        config,
        prompt,
        session_id=session_id,
        run_id=run_id,
        role=role,
        label=label,
        timeout_seconds=timeout_seconds,
    )
