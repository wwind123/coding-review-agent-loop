"""Registry for supported agent backends."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from .base import AgentBackend, AgentName, AgentResult
from .antigravity import BACKEND as ANTIGRAVITY_BACKEND
from .claude import BACKEND as CLAUDE_BACKEND
from .codex import BACKEND as CODEX_BACKEND
from .gemini import BACKEND as GEMINI_BACKEND
from ..errors import AgentLoopError
from ..runner import Runner
from ..logging import log

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


def _configured_model_label(
    agent: AgentName, config: AgentLoopConfig | None, *, role: str | None = None
) -> str | None:
    """The model label declared in config for `agent`, or None if not declared.

    Codex and Claude append the resolver's selected effort; antigravity's model
    string already embeds effort (e.g. "Gemini 3.1 Pro (High)"), so it is used
    verbatim.
    """
    if config is None:
        return None
    if agent == "antigravity":
        return config.antigravity_models[0] if config.antigravity_models else None
    if agent == "codex":
        from ..config import resolve_invocation

        invocation = resolve_invocation(config, provider="codex", role=role)
        if not invocation.configured_model:
            if invocation.effort_source == "tool_default":
                return None
            model = "unknown model"
        else:
            model = invocation.configured_model
        return f"{model} ({invocation.resolved_effort})"
    if agent == "gemini":
        return config.gemini_model or None
    if agent == "claude":
        from ..config import resolve_invocation

        invocation = resolve_invocation(config, provider="claude", role=role)
        if not invocation.configured_model:
            if invocation.effort_source == "tool_default":
                return None
            model = "unknown model"
        else:
            model = invocation.configured_model
        return f"{model} ({invocation.resolved_effort})"
    return None


def agent_signature(
    agent: AgentName,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    *,
    role: str | None = None,
) -> str:
    """Build the `-- {signature}` label for `agent`.

    Precedence: the model that actually ran (`model_used`, ground truth) > the
    model declared in config > the generic provider signature. When no model is
    known, returns the generic signature unchanged so existing output is stable.
    """
    base = get_backend(agent).signature
    label = model_used or _configured_model_label(agent, config, role=role)
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
    attempt_suffix: str | None = None,
) -> AgentResult:
    # Configure the runner at the backend boundary so direct library callers,
    # skill-mode callers, and the CLI all use the same immutable containment
    # policy.  Test doubles inherit this method but their scripted spawn path
    # remains authoritative and does not need a systemd probe.
    configure = getattr(runner, "configure_from_config", None)
    if callable(configure):
        configure(config)
    set_role = getattr(runner, "set_containment_role", None)
    if callable(set_role):
        set_role(role)
    from ..config import resolve_invocation

    invocation = resolve_invocation(config, provider=agent, role=role)
    log(
        config,
        f"Resolved {agent} {role or 'turn'}: model={invocation.configured_model or 'unknown'} "
        f"effort={invocation.resolved_effort or 'unspecified'} "
        f"source={invocation.effort_source or 'provider-default'}",
    )
    kwargs: dict[str, object] = dict(
        session_id=session_id,
        run_id=run_id,
        role=role,
        label=label,
        timeout_seconds=timeout_seconds,
    )
    if attempt_suffix is not None:
        kwargs["attempt_suffix"] = attempt_suffix
    result = get_backend(agent).run(
        runner,
        config,
        prompt,
        **kwargs,
    )
    # Resolve identity at the common backend boundary so direct library calls,
    # retries, and session resumes carry the same metadata as CLI turns.
    if (
        invocation.resolved_effort is not None
        and result.observed_effort is not None
        and result.observed_effort != invocation.resolved_effort
    ):
        log(
            config,
            f"Warning: {agent} observed effort {result.observed_effort!r} differs from "
            f"configured {invocation.resolved_effort!r}; retaining the completed turn.",
        )
    if (
        invocation.configured_model
        and result.observed_model
        and invocation.configured_model != result.observed_model
    ):
        log(
            config,
            f"Warning: {agent} observed model {result.observed_model!r} differs from "
            f"configured {invocation.configured_model!r}; retaining the completed turn.",
        )
    observed_model = result.observed_model
    if agent == "antigravity" and observed_model is None and result.model_used:
        observed_model = result.model_used
    selected_model = observed_model or invocation.configured_model
    selected_effort = result.observed_effort or invocation.resolved_effort
    model_used = result.model_used
    if agent in {"codex", "claude"} and selected_effort is not None:
        model_used = f"{selected_model or 'unknown model'} ({selected_effort})"
    elif observed_model:
        model_used = observed_model
    return replace(
        result,
        provider=agent,
        role=role,
        configured_model=invocation.configured_model,
        configured_effort=invocation.resolved_effort,
        effort_source=invocation.effort_source,
        observed_model=observed_model,
        model_used=model_used,
    )
