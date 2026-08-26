"""Claude Code backend."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    AgentName,
    AgentResult,
    STDIN_PROMPT_THRESHOLD_BYTES,
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from ..logging import agent_log_path, log
from ..runner import CommandResult, Runner, executable_identity_changed
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


_UPDATER_DIAGNOSTIC_RE = re.compile(
    r"(?:^(?:error|fatal).*?(?:auto-update|self-update|update failed|update in progress)|"
    r"^(?:Installing|Updating|Updated) Claude Code|"
    r"^(?:error|fatal).*?(?:npm (?:install|update)).*@anthropic-ai/claude-code)",
    re.IGNORECASE | re.MULTILINE,
)


def _has_updater_diagnostic(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()][-8:]
    return bool(_UPDATER_DIAGNOSTIC_RE.search("\n".join(lines)[-2048:]))


def classify_self_update_interruption(
    result: CommandResult,
    *,
    command: str,
    response_file_text: str | None,
    session_id: str | None,
) -> str | None:
    """Return bounded Claude-updater evidence, never a generic failure classification."""
    observation = result.observation
    if not isinstance(result.returncode, int) or result.returncode == 0 or observation is None:
        return None
    if observation.interrupted or observation.elapsed_seconds > 30 or response_file_text or session_id:
        return None
    # Any parseable Claude JSON event is progress, even if it is not a terminal result.
    for line in result.stdout.splitlines():
        try:
            if isinstance(json.loads(line), dict):
                return None
        except json.JSONDecodeError:
            pass
    if _has_updater_diagnostic(result.stdout):
        return "Claude Code updater diagnostic"
    # A user-supplied absolute override is only eligible with an explicit
    # updater diagnostic.  A managed bare command may additionally prove the
    # race through an identity change while it was running.
    if os.path.isabs(command):
        return None
    if executable_identity_changed(
        observation.before,
        observation.after,
        spawn_wall_time=observation.spawn_wall_time,
        exit_wall_time=observation.spawn_wall_time + observation.elapsed_seconds,
    ):
        return "Claude executable changed during invocation"
    return None


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
        attempt_suffix: str | None = None,
    ) -> AgentResult:
        response_path = public_response_path(config, "claude")
        args = [config.claude_cmd, "--print", "--output-format", "json", *config.claude_args]
        # Pin the model when declared (#332); conflict validation guarantees this is
        # not also passed via --claude-arg --model.
        if config.claude_model:
            args += ["--model", config.claude_model]
        if session_id:
            args += ["--resume", session_id]
        prompt_with_response_instruction = with_public_response_file_instruction(prompt, response_path)
        input_text = None
        if len(prompt_with_response_instruction.encode("utf-8")) > STDIN_PROMPT_THRESHOLD_BYTES:
            # Linux limits a single exec argument to about 128 KiB. Long compact
            # contexts can exceed that before the Claude CLI is launched.
            input_text = prompt_with_response_instruction
        else:
            args.append(prompt_with_response_instruction)
        log_path = agent_log_path(config, "claude", run_id=run_id, label=label, attempt_suffix=attempt_suffix)
        log(config, f"Starting Claude in {config.claude_dir}; log: {log_path}; response: {response_path}")
        result = runner.run_with_log(
            args,
            cwd=config.claude_dir,
            log_path=log_path,
            label="Claude",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False,
            env={"AGENT_LOOP_WORKDIR": str(config.claude_dir.resolve())},
            input_text=input_text,
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
            command_result=result,
            self_update_reason=classify_self_update_interruption(
                result, command=config.claude_cmd, response_file_text=response_file_text, session_id=new_session_id,
            ),
        )


BACKEND = ClaudeBackend()
