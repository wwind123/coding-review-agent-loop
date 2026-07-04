"""Claude Code backend."""

from __future__ import annotations

import json
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
from ..usage import UsageMetadata, coerce_int, first_present

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

def _normalize_claude_usage(payload: object) -> UsageMetadata | None:
    if not isinstance(payload, dict):
        return None
    input_tokens = coerce_int(first_present(payload, "input_tokens", "inputTokens"))
    cached_input_tokens = coerce_int(
        first_present(payload, "cached_input_tokens", "cache_read_input_tokens")
    )
    output_tokens = coerce_int(first_present(payload, "output_tokens", "outputTokens"))
    total_tokens = coerce_int(payload.get("total_tokens"))
    if total_tokens is None and any(value is not None for value in (input_tokens, output_tokens)):
        total_tokens = sum(value or 0 for value in (input_tokens, output_tokens))
    if not any(
        value is not None for value in (input_tokens, cached_input_tokens, output_tokens, total_tokens)
    ):
        return None
    return UsageMetadata(
        mode="partial" if cached_input_tokens is None else "exact",
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _extract_claude_model(data: dict) -> str | None:
    """Model id Claude reported running (`model`, else primary `modelUsage` key)."""
    model = data.get("model")
    if isinstance(model, str) and model:
        return model
    model_usage = data.get("modelUsage")
    if isinstance(model_usage, dict) and model_usage:
        def output_tokens(model_id: str) -> int:
            usage = model_usage.get(model_id)
            if not isinstance(usage, dict):
                return 0
            return coerce_int(usage.get("outputTokens")) or 0

        return max(
            model_usage,
            key=output_tokens,
        )
    return None


def _parse_claude_output(
    raw: str,
) -> tuple[str, str | None, UsageMetadata | None, object | None, str | None]:
    """Extract (text, session_id, usage, raw_usage, model) from Claude json output."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            text = data.get("result", raw)
            if not isinstance(text, str):
                text = raw
            raw_usage = data.get("usage")
            if raw_usage is None and isinstance(data.get("result_message"), dict):
                raw_usage = data["result_message"].get("usage")
            return (
                text,
                data.get("session_id"),
                _normalize_claude_usage(raw_usage),
                raw_usage,
                _extract_claude_model(data),
            )
    except (json.JSONDecodeError, ValueError):
        pass
    return raw, None, None, None, None


class ClaudeBackend:
    name: AgentName = "claude"
    display_name = "Claude"
    signature = "Anthropic Claude"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.claude_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--dangerously-skip-permissions",) if dangerous else ()

    def run(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        session_id: str | None = None,
        run_id: str | None = None,
        role: str | None = None,
        label: str | None = None,
        timeout_seconds: float | None = None,
    ) -> AgentResult:
        response_path = public_response_path(config, "claude")
        args = [config.claude_cmd, "--print", "--output-format", "json", *config.claude_args]
        # Pin the model when declared (#332); conflict validation guarantees this is
        # not also passed via --claude-arg --model.
        if config.claude_model:
            args += ["--model", config.claude_model]
        if session_id:
            args += ["--resume", session_id]
        args.append(with_public_response_file_instruction(prompt, response_path))
        log_path = agent_log_path(config, "claude", run_id=run_id, label=label)
        log(config, f"Starting Claude in {config.claude_dir}; log: {log_path}; response: {response_path}")
        result = runner.run_with_log(
            args,
            cwd=config.claude_dir,
            log_path=log_path,
            label="Claude",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False,
            env={"AGENT_LOOP_WORKDIR": str(config.claude_dir.resolve())},
            timeout_seconds=timeout_seconds,
        )
        log(config, f"Claude finished; log: {log_path}")
        message_text, new_session_id, usage, raw_usage, model_detected = _parse_claude_output(result.stdout)
        response_file_text = read_public_response_file(response_path)
        return AgentResult(
            text=response_file_text or message_text,
            raw_output=result.stdout,
            text_source="response_file" if response_file_text is not None else "stdout",
            response_file_text=response_file_text,
            response_file_path=response_path,
            message_text=message_text,
            session_id=new_session_id,
            log_path=log_path,
            returncode=result.returncode,
            usage=usage,
            raw_usage=raw_usage,
            # Ground truth from Claude's own output; falls back to config.claude_model
            # at signature time when detection is unavailable (e.g. non-JSON output).
            model_used=model_detected,
        )


BACKEND = ClaudeBackend()
