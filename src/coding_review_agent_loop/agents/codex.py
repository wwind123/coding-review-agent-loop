"""OpenAI Codex CLI backend."""

from __future__ import annotations

import json
import os
import re
import tempfile
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


_CODEX_THREAD_ID_RE = re.compile(
    r"^[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_CODEX_REASONING_EFFORT_KEYS = (
    "effort",
    "reasoning_effort",
    "model_reasoning_effort",
)


def _normalize_codex_usage(payload: object) -> UsageMetadata | None:
    if not isinstance(payload, dict):
        return None
    input_tokens = coerce_int(payload.get("input_tokens"))
    cached_input_tokens = coerce_int(payload.get("cached_input_tokens"))
    output_tokens = coerce_int(payload.get("output_tokens"))
    reasoning_tokens = coerce_int(
        first_present(payload, "reasoning_tokens", "reasoning_output_tokens")
    )
    total_tokens = coerce_int(payload.get("total_tokens"))
    if total_tokens is None and any(
        value is not None for value in (input_tokens, output_tokens, reasoning_tokens)
    ):
        total_tokens = sum(value or 0 for value in (input_tokens, output_tokens, reasoning_tokens))
    if not any(
        value is not None
        for value in (
            input_tokens,
            cached_input_tokens,
            output_tokens,
            reasoning_tokens,
            total_tokens,
        )
    ):
        return None
    return UsageMetadata(
        mode="exact",
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
    )


def _extract_codex_usage(raw: str) -> tuple[UsageMetadata | None, object | None]:
    last_usage: object | None = None
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type") or event.get("event")
        if event_type != "turn.completed":
            continue
        usage_payload = event.get("usage")
        if usage_payload is None and isinstance(event.get("turn"), dict):
            usage_payload = event["turn"].get("usage")
        usage = _normalize_codex_usage(usage_payload)
        if usage is not None:
            last_usage = usage_payload
            return usage, usage_payload
    return None, last_usage


def _codex_model_args(config: AgentLoopConfig) -> list[str]:
    """CLI args to pin codex's model/effort when declared (#332).

    Conflict validation guarantees these are not also passed via --codex-arg, so
    no duplicate flags. The reasoning-effort value is TOML, hence the quotes.
    """
    args: list[str] = []
    if config.codex_model:
        args += ["--model", config.codex_model]
    if config.codex_reasoning_effort:
        args += ["-c", f'model_reasoning_effort="{config.codex_reasoning_effort}"']
    return args


def _codex_model_label(config: AgentLoopConfig) -> str | None:
    if not config.codex_model:
        return None
    if config.codex_reasoning_effort:
        return f"{config.codex_model} ({config.codex_reasoning_effort})"
    return config.codex_model


def _extract_codex_thread_id(raw: str) -> str | None:
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(event, dict) or event.get("type") != "thread.started":
            continue
        thread_id = event.get("thread_id")
        if isinstance(thread_id, str):
            thread_id = thread_id.strip()
            if _CODEX_THREAD_ID_RE.fullmatch(thread_id):
                return thread_id
    return None


def _codex_session_root() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    return Path(codex_home).expanduser() if codex_home else Path.home() / ".codex"


def _find_codex_rollout(thread_id: str) -> Path | None:
    if not _CODEX_THREAD_ID_RE.fullmatch(thread_id):
        return None
    sessions_root = _codex_session_root() / "sessions"
    try:
        matches = list(sessions_root.glob(f"**/rollout-*-{thread_id}.jsonl"))
    except OSError:
        return None

    newest: tuple[int, str, Path] | None = None
    for path in matches:
        try:
            candidate = (path.stat().st_mtime_ns, str(path), path)
        except OSError:
            continue
        if newest is None or candidate > newest:
            newest = candidate
    return newest[2] if newest is not None else None


def _model_record_containers(record: dict[object, object]) -> tuple[dict[object, object], ...]:
    # Codex rollout model metadata is known to appear at the top level or in
    # payload, turn, and turn_context containers. The fixture test covering the
    # current turn_context/payload shape should fail visibly if that schema drifts.
    containers = [record]
    for key in ("payload", "turn", "turn_context"):
        value = record.get(key)
        if isinstance(value, dict):
            containers.append(value)
    return tuple(containers)


def _read_codex_rollout_model(rollout_path: Path) -> str | None:
    model: str | None = None
    effort: str | None = None
    try:
        with rollout_path.open(encoding="utf-8") as rollout:
            for line in rollout:
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(record, dict):
                    continue
                for container in _model_record_containers(record):
                    candidate = container.get("model")
                    if not isinstance(candidate, str) or not candidate.strip():
                        continue
                    model = candidate.strip()
                    effort = None
                    for key in _CODEX_REASONING_EFFORT_KEYS:
                        candidate_effort = container.get(key)
                        if isinstance(candidate_effort, str) and candidate_effort.strip():
                            effort = candidate_effort.strip()
                            break
    except (OSError, UnicodeError):
        return None

    if model is None:
        return None
    return f"{model} ({effort})" if effort else model


def _detect_codex_model(raw: str) -> str | None:
    thread_id = _extract_codex_thread_id(raw)
    if thread_id is None:
        return None
    rollout_path = _find_codex_rollout(thread_id)
    if rollout_path is None:
        return None
    return _read_codex_rollout_model(rollout_path)


def _read_codex_message_file(output_path: Path) -> str | None:
    try:
        text = output_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    text = text.strip()
    return text or None


class CodexBackend:
    name: AgentName = "codex"
    display_name = "Codex"
    signature = "OpenAI Codex"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.codex_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--dangerously-bypass-approvals-and-sandbox",) if dangerous else ()

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
        log_path = agent_log_path(config, "codex", run_id=run_id, label=label)
        response_path = public_response_path(config, "codex")
        prompt_with_response_file = with_public_response_file_instruction(prompt, response_path)
        log(
            config,
            f"Starting Codex in {config.codex_dir}; log: {log_path}; response: {response_path}",
        )
        if config.dry_run:
            result = runner.run(
                [
                    config.codex_cmd,
                    "exec",
                    "--cd",
                    str(config.codex_dir),
                    *_codex_model_args(config),
                    *config.codex_args,
                    prompt,
                ],
                cwd=config.codex_dir,
                env={"AGENT_LOOP_WORKDIR": str(config.codex_dir.resolve())},
            )
            log(config, f"Codex finished; log: {log_path}")
            return AgentResult(
                text=result.stdout,
                raw_output=result.stdout,
                text_source="stdout",
                message_text=result.stdout,
                log_path=log_path,
                returncode=result.returncode,
                model_used=_codex_model_label(config),
            )

        with tempfile.NamedTemporaryFile("r", encoding="utf-8", delete=False) as handle:
            output_path = handle.name
        try:
            result = runner.run_with_log(
                [
                    config.codex_cmd,
                    "exec",
                    "--cd",
                    str(config.codex_dir),
                    "--json",
                    "--output-last-message",
                    output_path,
                    *_codex_model_args(config),
                    *config.codex_args,
                    prompt_with_response_file,
                ],
                cwd=config.codex_dir,
                log_path=log_path,
                label="Codex",
                progress_interval_seconds=config.progress_interval_seconds,
                check=False,
                env={"AGENT_LOOP_WORKDIR": str(config.codex_dir.resolve())},
                timeout_seconds=timeout_seconds,
            )
            response_file_text = read_public_response_file(response_path)
            message_text = _read_codex_message_file(Path(output_path)) or result.stdout
            usage, raw_usage = _extract_codex_usage(result.stdout)
            model_used = _codex_model_label(config) or _detect_codex_model(result.stdout)
            log(config, f"Codex finished; log: {log_path}")
            return AgentResult(
                text=response_file_text or message_text,
                raw_output=result.stdout,
                text_source="response_file" if response_file_text is not None else "stdout",
                response_file_text=response_file_text,
                message_text=message_text,
                log_path=log_path,
                returncode=result.returncode,
                usage=usage,
                raw_usage=raw_usage,
                model_used=model_used,
            )
        finally:
            try:
                os.unlink(output_path)
            except FileNotFoundError:
                pass


BACKEND = CodexBackend()
