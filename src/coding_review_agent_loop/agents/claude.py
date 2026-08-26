"""Claude Code backend."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
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
from ..workdir_guard import WorkdirSnapshot, capture_workdir_snapshot

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


@dataclass(frozen=True)
class SelfUpdateInterruptionEvidence:
    """Positive updater evidence and, independently, a replay refusal."""

    reason: str
    replay_refusal_kind: str | None = None
    replay_refusal_detail: str | None = None

def _workdir_probe_label(snapshot: WorkdirSnapshot) -> str:
    def describe(probe) -> str:
        if probe.kind == "available":
            return repr(probe.value)
        if probe.kind == "exception":
            return f"{probe.exception_type}: {probe.detail}"
        if probe.kind == "command-failed":
            return f"returncode={probe.returncode!r}"
        return "blank"

    return f"HEAD={describe(snapshot.head)}, status={describe(snapshot.status)}"


def _gate_self_update_replay(
    reason: str,
    *,
    before_snapshot: WorkdirSnapshot | None,
    after_snapshot: WorkdirSnapshot | None,
) -> SelfUpdateInterruptionEvidence:
    """Accept a candidate only when both read-only snapshots are identical."""
    # Calls without snapshots retain the historical evidence-only helper
    # contract. ClaudeBackend always supplies both snapshots after candidate
    # detection, so replay policy itself remains fail-closed there.
    if before_snapshot is None and after_snapshot is None:
        return SelfUpdateInterruptionEvidence(reason)
    if before_snapshot is None or not before_snapshot.available:
        detail = (
            "Claude self-update replay refused: before workdir snapshot "
            f"unavailable ({_workdir_probe_label(before_snapshot) if before_snapshot else 'missing'})."
        )
        return SelfUpdateInterruptionEvidence(reason, "before-unavailable", detail)
    if after_snapshot is None or not after_snapshot.available:
        detail = (
            "Claude self-update replay refused: after workdir snapshot "
            f"unavailable ({_workdir_probe_label(after_snapshot) if after_snapshot else 'missing'})."
        )
        return SelfUpdateInterruptionEvidence(reason, "after-unavailable", detail)
    if before_snapshot.head.value != after_snapshot.head.value:
        detail = (
            "Claude self-update replay refused: assigned checkout HEAD changed "
            f"during invocation (before={before_snapshot.head.value!r}, "
            f"after={after_snapshot.head.value!r})."
        )
        return SelfUpdateInterruptionEvidence(reason, "changed-head", detail)
    if before_snapshot.status.value != after_snapshot.status.value:
        detail = (
            "Claude self-update replay refused: dirty workdir status changed "
            f"during invocation (before={before_snapshot.status.value!r}, "
            f"after={after_snapshot.status.value!r})."
        )
        return SelfUpdateInterruptionEvidence(reason, "changed-status", detail)
    return SelfUpdateInterruptionEvidence(reason)


def classify_self_update_interruption(
    result: CommandResult,
    *,
    command: str,
    response_file_text: str | None,
    session_id: str | None,
    before_snapshot: WorkdirSnapshot | None = None,
    after_snapshot: WorkdirSnapshot | None = None,
) -> SelfUpdateInterruptionEvidence | None:
    """Return bounded Claude-updater evidence, then apply the workdir gate."""
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
        reason = "Claude Code updater diagnostic"
        return _gate_self_update_replay(
            reason,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
    # A user-supplied absolute override is only eligible with an explicit
    # updater diagnostic.  A managed bare command may additionally prove the
    # race through an identity change while it was running.
    if os.path.isabs(command):
        return None
    if executable_identity_changed(
        observation.before,
        observation.after,
        command=command,
        spawn_wall_time=observation.spawn_wall_time,
        exit_wall_time=observation.spawn_wall_time + observation.elapsed_seconds,
    ):
        reason = "Claude executable changed during invocation"
        return _gate_self_update_replay(
            reason,
            before_snapshot=before_snapshot,
            after_snapshot=after_snapshot,
        )
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
        # This is deliberately before every Claude invocation, including an
        # ordinary retry and the dedicated replay. Snapshot failures are
        # diagnostic evidence for the replay gate, not backend failures.
        before_snapshot = capture_workdir_snapshot(
            runner,
            config.claude_dir,
            tolerate_exceptions=True,
        )
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
        candidate = classify_self_update_interruption(
            result,
            command=config.claude_cmd,
            response_file_text=response_file_text,
            session_id=new_session_id,
        )
        after_snapshot = None
        if candidate is not None and candidate.replay_refusal_kind is None:
            # The after probe is lazy: failures without positive replacement
            # evidence never inspect status and cannot emit refusal metadata.
            after_snapshot = capture_workdir_snapshot(
                runner,
                config.claude_dir,
                tolerate_exceptions=True,
            )
            candidate = classify_self_update_interruption(
                result,
                command=config.claude_cmd,
                response_file_text=response_file_text,
                session_id=new_session_id,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
            )
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
            self_update_reason=candidate.reason if candidate else None,
            self_update_replay_refusal_kind=(
                candidate.replay_refusal_kind if candidate else None
            ),
            self_update_replay_refusal_detail=(
                candidate.replay_refusal_detail if candidate else None
            ),
        )


BACKEND = ClaudeBackend()
