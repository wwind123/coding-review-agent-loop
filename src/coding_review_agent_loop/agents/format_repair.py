"""Constrained, fresh-session Codex and Claude formatting calls."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

from ..config import DEFAULT_REASONING_EFFORT
from .base import AgentName, AgentResult
from .claude import _parse_claude_output
from .codex import _extract_codex_usage

if TYPE_CHECKING:
    from ..config import AgentLoopConfig
    from ..runner import Runner


def run_cli_repair(
    runner: Runner,
    config: AgentLoopConfig,
    prompt: str,
    *,
    model: str,
    log_path: Path,
) -> AgentResult:
    backend = config.repair_backend
    if backend not in {"codex", "claude"}:
        raise ValueError(f"Unsupported format repair backend: {backend}")
    effort = config.repair_reasoning_effort or DEFAULT_REASONING_EFFORT
    root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{backend}-", dir=root) as directory:
        cwd = Path(directory)
        message_path = cwd / "last-message.txt"
        # Do not reuse regular backend arguments: they may enable tools, bypass
        # permissions, resume a session, or override this invocation's identity.
        if backend == "codex":
            args = [
                config.codex_cmd, "exec", "--ignore-user-config", "--ignore-rules",
                "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                "-c", 'approval_policy="never"',
                "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
                "-c", 'web_search="disabled"', "-c", "features.multi_agent=false",
                "-c", f'model_reasoning_effort="{effort}"',
                "--model", model, "--json", "--output-last-message", str(message_path), "-",
            ]
        else:
            args = [
                config.claude_cmd, "--print", "--output-format", "json",
                "--safe-mode", "--tools", "", "--strict-mcp-config",
                "--mcp-config", '{"mcpServers":{}}', "--no-session-persistence",
                "--model", model, "--effort", effort,
            ]
        result = runner.run_with_log(
            args, cwd=cwd, log_path=log_path, label=f"{backend.capitalize()} repair",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False, input_text=prompt, timeout_seconds=config.repair_timeout_seconds,
            containment_role="repair",
            env={"AGENT_LOOP_WORKDIR": str(cwd), **(
                {"CLAUDE_CODE_EFFORT_LEVEL": effort} if backend == "claude" else {}
            )},
        )
        observed_model = None
        if backend == "codex":
            text = message_path.read_text(encoding="utf-8") if message_path.exists() else ""
            usage, raw_usage = _extract_codex_usage(result.stdout)
        else:
            text, _, usage, raw_usage, observed_model = _parse_claude_output(result.stdout)
        return AgentResult(
            text=text, raw_output=f"{result.stdout}\n{result.stderr}",
            log_path=log_path, returncode=result.returncode, usage=usage, raw_usage=raw_usage,
            provider=cast(AgentName, backend), role="repair", configured_model=model,
            configured_effort=effort, effort_source="repair_backend",
            observed_model=observed_model, command_result=result,
        )
