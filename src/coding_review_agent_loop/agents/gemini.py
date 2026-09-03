"""Gemini CLI backend."""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from .base import (
    AgentName,
    AgentResult,
    AgentTextSource,
    STDIN_PROMPT_THRESHOLD_BYTES,
    public_response_path,
    read_public_response_file,
    with_public_response_file_instruction,
)
from .replacement import classify_provider_executable_replacement_interruption
from ..logging import agent_log_path, log
from ..protocol import CLARIFY_RE, PLAN_STATE_RE, PUBLIC_RESPONSE_MARKER, STATE_RE
from ..runner import CommandResult, Runner
from ..usage import UsageMetadata, coerce_int, first_present
from ..workdir_guard import WorkdirReplayEvidence, WorkdirSnapshot, capture_workdir_snapshot

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


# Gemini CLI consumer access (free / Google AI Pro / Ultra) is retiring on this
# date; after it, personal-account Gemini CLI usage stops working and users should
# migrate to Antigravity (`agy`) or a supported enterprise / API-key path (#215).
GEMINI_CONSUMER_CUTOFF = date(2026, 6, 18)
_GEMINI_CUTOFF_WARN_AHEAD_DAYS = 14

_GEMINI_MIGRATION_GUIDANCE = (
    "Gemini CLI consumer access (free / Google AI Pro / Ultra) is retiring on "
    "2026-06-18. If you are on a personal Google account, migrate to Antigravity "
    "(`agy`): --coder antigravity / --reviewer antigravity (skill: --agent "
    "antigravity). Gemini CLI remains supported for enterprise / API-key paths."
)

# Substrings (lowercased) in a *failed* gemini invocation's output that suggest an
# auth/quota/retirement problem, where the migration guidance is relevant.
_GEMINI_RETIREMENT_SIGNATURES = (
    "unauthenticated",
    "permission denied",
    "permission_denied",
    "quota",
    "resource exhausted",
    "resource_exhausted",
    "not authorized",
    "unauthorized",
    "no longer",
    "retir",
)

# A replacement can leave Gemini's Node launcher with no public response. Keep
# this classifier deliberately narrow: ordinary prose, tool narration, and
# unknown diagnostics are progress and must suppress replay.
_GEMINI_STARTUP_LINE_RE = re.compile(
    r"(?ix)^(?:"
    r"gemini(?:\s+cli)?(?:\s+(?:v|version)\s*[\w.-]+)?|"
    r"(?:starting|initiali[sz]ing|loading)\s+gemini(?:\s+cli)?|"
    r"(?:using|selected|requested)\s+model\s*[:=].+|"
    r"model\s*[:=].+|"
    r"version\s*[:=]\s*[\w.-]+|"
    r"loaded\s+(?:configuration|credentials|settings)\b.*|"
    r"(?:checking|applying)\s+(?:for\s+)?(?:gemini\s+)?updates?\b.*"
    r")$"
)
_GEMINI_LOADER_LINE_RE = re.compile(
    r"(?ix)(?:"
    r"MODULE_NOT_FOUND|cannot\s+find\s+module|"
    r"^node:(?:internal/)?modules/.*|^at\s+.+node:(?:internal/)?modules/.*|"
    r"^throw\s+err;?$|^require\s+stack:.*|"
    r"^code\s*[:=]\s*['\"]?(?:MODULE_NOT_FOUND|ENOENT)['\"]?\s*$|"
    r"(?:ENOENT|ENOEXEC)|exec(?:ution)?\s+format\s+error|"
    r"no\s+such\s+file\s+or\s+directory|"
    r"(?:gemini|node|npm|launcher|executable|binary).*(?:not\s+found|missing|failed|error)|"
    r"(?:failed|error).*(?:launch|launcher|startup|updat(?:e|er)|replacement)|"
    r"(?:updat(?:e|er)|launcher|replacement).*(?:failed|error|unavailable|in\s+progress)"
    r")"
)


def classify_gemini_executable_replacement_interruption(
    result: CommandResult,
    *,
    command: str,
    response_file_text: str | None,
    before_snapshot: WorkdirSnapshot | None = None,
    after_snapshot: WorkdirSnapshot | None = None,
) -> WorkdirReplayEvidence | None:
    return classify_provider_executable_replacement_interruption(
        result,
        command=command,
        response_file_text=response_file_text,
        startup_line_re=_GEMINI_STARTUP_LINE_RE,
        loader_line_re=_GEMINI_LOADER_LINE_RE,
        provider_label="Gemini executable replacement",
        reason="Gemini executable changed during invocation",
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )


# Keep the provider-local spelling parallel with Codex's existing classifier;
# callers that operate on a backend do not need to know the provider's helper name.
classify_executable_replacement_interruption = classify_gemini_executable_replacement_interruption

# Gemini's documented headless mode is selected by --prompt.  For large tasks,
# stdin carries the complete prompt and this short trailing prompt tells Gemini
# how to treat that preceding input.
_OVERSIZED_PROMPT_DIRECTIVE = (
    "Treat the preceding stdin as the complete primary task and carry it out without "
    "summarizing or replacing it. Follow its public-response-file destination and "
    "PUBLIC_RESPONSE_MARKER instructions exactly."
)


def _gemini_retirement_signal(text: str) -> bool:
    """True if failed-gemini output looks like an auth/quota/retirement problem."""
    lowered = text.lower()
    return any(signature in lowered for signature in _GEMINI_RETIREMENT_SIGNATURES)


def _with_public_response_marker_instruction(prompt: str) -> str:
    return f"""{prompt}

IMPORTANT FOR GEMINI CLI OUTPUT FILTERING:

Gemini CLI may print tool-use narration, diagnostics, or internal status text
before your final answer.

When you are ready to provide the response that should be posted publicly to
GitHub, print this exact line immediately before it:

{PUBLIC_RESPONSE_MARKER}

Only content after that line will be posted to GitHub. Do not print the marker
until you are done with all internal reasoning, tool use, and review work.
"""


def _strip_public_response_marker(raw: str) -> tuple[str, bool]:
    if PUBLIC_RESPONSE_MARKER not in raw:
        return raw, False
    return raw.rsplit(PUBLIC_RESPONSE_MARKER, 1)[1].lstrip("\n"), True


def _strip_gemini_preamble(raw: str) -> tuple[str, AgentTextSource]:
    """Drop Gemini CLI diagnostics that can appear before the final response."""
    marker_stripped, used_marker = _strip_public_response_marker(raw)
    if used_marker:
        return marker_stripped, "stdout_marker"

    marker_matches = [*STATE_RE.finditer(raw), *PLAN_STATE_RE.finditer(raw), *CLARIFY_RE.finditer(raw)]
    if not marker_matches:
        return raw, "stdout"

    public_end = max(match.start() for match in marker_matches)
    separator = "\n---\n"
    separator_at = raw.find(separator, 0, public_end)
    if separator_at == -1:
        return raw, "stdout"

    return raw[separator_at + len(separator) :].lstrip("\n"), "stdout"

def _normalize_gemini_usage(payload: object) -> UsageMetadata | None:
    if not isinstance(payload, dict):
        return None
    input_tokens = coerce_int(
        first_present(payload, "input_tokens", "inputTokenCount", "promptTokenCount")
    )
    cached_input_tokens = coerce_int(
        first_present(payload, "cached_input_tokens", "cachedInputTokenCount")
    )
    output_tokens = coerce_int(
        first_present(payload, "output_tokens", "outputTokenCount", "candidatesTokenCount")
    )
    total_tokens = coerce_int(first_present(payload, "total_tokens", "totalTokenCount"))
    if total_tokens is None and any(value is not None for value in (input_tokens, output_tokens)):
        total_tokens = sum(value or 0 for value in (input_tokens, output_tokens))
    if not any(
        value is not None for value in (input_tokens, cached_input_tokens, output_tokens, total_tokens)
    ):
        return None
    mode = "exact" if all(value is not None for value in (input_tokens, output_tokens, total_tokens)) else "partial"
    return UsageMetadata(
        mode=mode,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _parse_gemini_payload(
    raw: str,
) -> tuple[str, str | None, UsageMetadata | None, object | None, AgentTextSource]:
    """Extract (text, session_id, usage, raw_usage, text_source) from Gemini output."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            text = data.get("response", raw)
            if not isinstance(text, str):
                text = raw
            session_id = data.get("session_id")
            raw_usage = first_present(data, "stats", "usage", "usageMetadata")
            message_text, text_source = _strip_gemini_preamble(text)
            return (
                message_text,
                session_id if isinstance(session_id, str) else None,
                _normalize_gemini_usage(raw_usage),
                raw_usage,
                text_source,
            )
    except (json.JSONDecodeError, ValueError):
        pass
    message_text, text_source = _strip_gemini_preamble(raw)
    return message_text, None, None, None, text_source


def _gemini_public_response_root(gemini_dir: Path) -> Path:
    git_marker = gemini_dir / ".git"
    if git_marker.is_file():
        gitdir_prefix = "gitdir:"
        try:
            gitdir_text = git_marker.read_text(encoding="utf-8").strip()
        except OSError:
            gitdir_text = ""
        if gitdir_text.lower().startswith(gitdir_prefix):
            git_dir = Path(gitdir_text[len(gitdir_prefix) :].strip())
            if not git_dir.is_absolute():
                git_dir = gemini_dir / git_dir
            return git_dir.resolve() / "agent-loop" / "responses"
        return gemini_dir / ".agent-loop-responses"

    return git_marker / "agent-loop" / "responses"


class GeminiBackend:
    name: AgentName = "gemini"
    display_name = "Gemini"
    signature = "Google Gemini"

    def workdir(self, config: AgentLoopConfig) -> Path:
        return config.gemini_dir

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]:
        return ("--yolo", "--skip-trust") if dangerous else ()

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
        # Gemini CLI only allows file writes inside the trusted workspace (or
        # its own private temp dir, whose path we do not know ahead of time).
        # Keep the response file inside the git dir so it is writable but never
        # dirties the reviewed worktree. In linked worktrees, .git is a pointer
        # file, so resolve it before creating children beneath it.
        response_path = public_response_path(
            config,
            "gemini",
            root=_gemini_public_response_root(config.gemini_dir),
        )
        log_path = agent_log_path(config, "gemini", run_id=run_id, label=label, attempt_suffix=attempt_suffix)
        log(config, f"Starting Gemini in {config.gemini_dir}; log: {log_path}; response: {response_path}")
        if date.today() >= GEMINI_CONSUMER_CUTOFF - timedelta(days=_GEMINI_CUTOFF_WARN_AHEAD_DAYS):
            log(config, f"Gemini CLI advisory: {_GEMINI_MIGRATION_GUIDANCE}")
        rendered_prompt = _with_public_response_marker_instruction(
            with_public_response_file_instruction(prompt, response_path)
        )
        oversized_prompt = len(rendered_prompt.encode("utf-8")) > STDIN_PROMPT_THRESHOLD_BYTES
        args = [
            config.gemini_cmd,
            "--prompt",
            _OVERSIZED_PROMPT_DIRECTIVE if oversized_prompt else rendered_prompt,
            *config.gemini_args,
        ]
        # Pin the model when declared (#332); conflict validation guarantees this is
        # not also passed via --gemini-arg --model.
        if config.gemini_model:
            args += ["--model", config.gemini_model]
        if session_id:
            args += ["--resume", session_id]
        # Capture the assigned checkout immediately before spawn. The after
        # probe is deliberately lazy and only runs for positive replacement
        # evidence, keeping ordinary Gemini failures cheap and side-effect free.
        before_snapshot = capture_workdir_snapshot(
            runner,
            config.gemini_dir,
            tolerate_exceptions=True,
        )
        result = runner.run_with_log(
            args,
            cwd=config.gemini_dir,
            log_path=log_path,
            label="Gemini",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False,
            env={
                "AGENT_LOOP_WORKDIR": str(config.gemini_dir.resolve()),
                "AGENT_LOOP_CODER_TEST_TIMEOUT_CEILING_SECONDS": str(
                    config.coder_test_command_timeout_seconds
                ),
            },
            timeout_seconds=timeout_seconds,
            input_text=rendered_prompt if oversized_prompt else None,
        )
        log(config, f"Gemini finished; log: {log_path}")
        retired_failure = result.returncode != 0 and _gemini_retirement_signal(result.stdout or "")
        if retired_failure:
            log(
                config,
                "Gemini CLI invocation failed with an auth/quota signal that may be the "
                f"consumer retirement. {_GEMINI_MIGRATION_GUIDANCE}",
            )
        message_text, new_session_id, usage, raw_usage, message_source = _parse_gemini_payload(result.stdout)
        response_file_text = read_public_response_file(response_path)
        candidate = classify_gemini_executable_replacement_interruption(
            result,
            command=config.gemini_cmd,
            response_file_text=response_file_text,
        )
        after_snapshot = None
        if candidate is not None:
            after_snapshot = capture_workdir_snapshot(
                runner,
                config.gemini_dir,
                tolerate_exceptions=True,
            )
            candidate = classify_gemini_executable_replacement_interruption(
                result,
                command=config.gemini_cmd,
                response_file_text=response_file_text,
                before_snapshot=before_snapshot,
                after_snapshot=after_snapshot,
            )
        raw_output = result.stdout
        if retired_failure:
            # Append the guidance to the *returned* output, not just stderr: callers
            # such as helpers.run_external classify/persist failures from
            # raw_output/text, so the migration guidance must travel with them (#215).
            guidance_block = f"\n\n{_GEMINI_MIGRATION_GUIDANCE}"
            raw_output = (raw_output or "") + guidance_block
            message_text = (message_text or "") + guidance_block
        return AgentResult(
            text=response_file_text or message_text,
            raw_output=raw_output,
            text_source="response_file" if response_file_text is not None else message_source,
            response_file_text=response_file_text,
            response_file_path=response_path,
            message_text=message_text,
            session_id=new_session_id,
            log_path=log_path,
            returncode=result.returncode,
            usage=usage,
            raw_usage=raw_usage,
            model_used=config.gemini_model or None,
            command_result=result,
            self_update_reason=candidate.reason if candidate else None,
            self_update_replay_refusal_kind=(
                candidate.replay_refusal_kind if candidate else None
            ),
            self_update_replay_refusal_detail=(
                candidate.replay_refusal_detail if candidate else None
            ),
        )


BACKEND = GeminiBackend()
