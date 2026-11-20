"""agy schema-constrained output for structured Antigravity turns (#2029).

Some Gemini models answer in chat markdown even when a turn asks for a JSON
response file.  agy can constrain the final answer with ``--output-format json
--json-schema <file>``: the CLI then prints a JSON envelope whose
``structured_output`` field holds a schema-valid object even when the model's
``response`` prose is narration.

This module holds the pure pieces of that channel:

* a cached capability probe, so an older agy keeps the legacy channel;
* the eligibility rule for a turn (allowlisted kind, and no content the
  contract requires outside the JSON object, footer and signature);
* a loose per-kind schema that also admits the shared ``agent_unavailable``
  alternative;
* a tolerant envelope parser and a renderer that rebuilds the public text
  (JSON, footer marker, signature) in code.

The schema only shapes the payload.  The protocol validators stay
authoritative for every rendered response.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import TYPE_CHECKING, Sequence

from ..logging import log
from ..runner import Runner, strip_ansi

if TYPE_CHECKING:
    from ..config import AgentLoopConfig


# Structured kinds that may use agy's schema channel, mapped to the footer
# marker name their validator expects.  Discuss kinds, semantic-patch kinds,
# repair and semantic-dedupe turns are deliberately absent.
STRUCTURED_KIND_FOOTERS: dict[str, str] = {
    "pr_review": "AGENT_STATE",
    "coder_followup": "AGENT_STATE",
    "issue_implementation": "AGENT_STATE",
    "task_result": "AGENT_STATE",
    "plan_review": "AGENT_PLAN_STATE",
    "plan_state": "AGENT_PLAN_STATE",
    "plan_revision": "AGENT_PLAN_STATE",
}
AGENT_UNAVAILABLE_KIND = "agent_unavailable"
_AGENT_UNAVAILABLE_FOOTER = "<!-- AGENT_UNAVAILABLE -->"
STRUCTURED_OUTPUT_FLAGS = ("--output-format", "--json-schema")
SCHEMA_FILE_NAME = "agy-structured-schema.json"
CAPABILITY_PROBE_TIMEOUT_SECONDS = 20.0

_capability_cache: dict[str, bool] = {}
_capability_lock = threading.Lock()


def antigravity_structured_kind_for_turn(
    expected_kind: str | None,
    *,
    surfaced_requirement_ids: Sequence[str] | None,
    reviewer_requirement_ids: Sequence[str] | None,
    requires_direct_discussion_ack: bool,
) -> str | None:
    """The kind this turn may request through agy's schema flags, else None.

    A turn whose contract needs content beyond the JSON object, footer and
    signature keeps the response-file channel, because ``structured_output``
    cannot carry that content and the renderer must never invent it:

    * surfaced signed requirement IDs (the human-requirements acknowledgement
      marker and section between the JSON and the footer);
    * a required direct-discussion acknowledgement, even with no surfaced IDs;
    * reviewer requirement IDs (the resolved marker approval counting needs).
    """
    if expected_kind not in STRUCTURED_KIND_FOOTERS:
        return None
    if surfaced_requirement_ids or reviewer_requirement_ids or requires_direct_discussion_ack:
        return None
    return expected_kind


def operator_args_set_output_format(args: Sequence[str]) -> bool:
    """True when pass-through agy args already choose an output format or schema."""
    for arg in args:
        for flag in STRUCTURED_OUTPUT_FLAGS:
            if arg == flag or arg.startswith(f"{flag}="):
                return True
    return False


def help_text_supports_structured_output(help_text: str) -> bool:
    cleaned = strip_ansi(help_text or "")
    return all(flag in cleaned for flag in STRUCTURED_OUTPUT_FLAGS)


def _probe_agy_help(runner: Runner, config: "AgentLoopConfig") -> str | None:
    """Run ``agy --help`` once under a PTY; the help text, or None on failure.

    Patched by the test suite so tests never depend on a host agy.
    """
    from ..logging import agent_log_path
    from ..scratch import make_private_dirs, scratch_root

    root = make_private_dirs(scratch_root() / "repair")
    with tempfile.TemporaryDirectory(prefix="agy-help-", dir=root) as temp_dir:
        result = runner.run_with_log(
            [config.antigravity_cmd, "--help"],
            cwd=Path(temp_dir),
            log_path=agent_log_path(config, "antigravity-capability-probe"),
            label="Antigravity capability probe",
            progress_interval_seconds=config.progress_interval_seconds,
            check=False,
            env={"AGENT_LOOP_WORKDIR": str(Path(temp_dir).resolve())},
            use_pty=True,
            timeout_seconds=CAPABILITY_PROBE_TIMEOUT_SECONDS,
        )
    if result.returncode != 0:
        return None
    return f"{result.stdout or ''}\n{result.stderr or ''}"


def _capability_cache_key(config: "AgentLoopConfig") -> str:
    command = config.antigravity_cmd
    resolved = shutil.which(command)
    return os.path.realpath(resolved) if resolved else command


def antigravity_structured_output_supported(
    runner: Runner, config: "AgentLoopConfig"
) -> bool:
    """Whether agy advertises both schema flags; probed once per command path.

    ``off`` mode, operator pass-through format args, and a probe that errors,
    times out or exits nonzero all mean unsupported.  A failed probe is cached
    and logged once, so the turn never fails or re-probes because of it.
    """
    if config.antigravity_structured_output == "off":
        return False
    if operator_args_set_output_format(config.antigravity_args):
        return False
    key = _capability_cache_key(config)
    with _capability_lock:
        if key in _capability_cache:
            return _capability_cache[key]
        try:
            help_text = _probe_agy_help(runner, config)
        except Exception as exc:  # noqa: BLE001 - the probe must never fail a turn
            help_text = None
            log(config, f"Antigravity structured-output capability probe failed: {exc}")
        supported = help_text is not None and help_text_supports_structured_output(help_text)
        if not supported:
            log(
                config,
                "Antigravity structured output unavailable for "
                f"{config.antigravity_cmd}; using the response-file channel",
            )
        _capability_cache[key] = supported
        return supported


def reset_structured_output_capability_cache() -> None:
    with _capability_lock:
        _capability_cache.clear()


def structured_output_schema(kind: str) -> dict[str, object]:
    """A loose JSON Schema for ``kind`` that also admits ``agent_unavailable``.

    Only constructs Gemini function declarations accept are used (#2039):
    ``enum`` appears only on string-typed values and every enum value is a
    non-empty string.  An integer enum is rejected by the provider with
    INVALID_ARGUMENT, so ``schema_version`` is a plain integer here and the
    protocol validators keep enforcing ``schema_version == 1``.  ``state`` is
    optional because an agent_unavailable payload has none; the role
    validators still require it.
    """
    return {
        "type": "object",
        "properties": {
            "schema_version": {"type": "integer"},
            "kind": {"type": "string", "enum": [kind, AGENT_UNAVAILABLE_KIND]},
            "state": {"type": "string", "enum": ["approved", "blocking"]},
        },
        "required": ["schema_version", "kind"],
        "additionalProperties": True,
    }


def write_structured_output_schema(directory: Path, kind: str) -> Path:
    path = directory / SCHEMA_FILE_NAME
    path.write_text(json.dumps(structured_output_schema(kind), indent=2), encoding="utf-8")
    return path


@dataclass(frozen=True)
class AgyEnvelope:
    status: str | None
    response: str
    structured_output: object | None
    error_text: str
    # The normalized stdout with the envelope's own JSON text removed.
    remainder: str = ""
    # Whether the envelope carries a top-level ``json_schema`` key (#2039).
    has_json_schema: bool = False

    @property
    def succeeded(self) -> bool:
        return (self.status or "").upper() == "SUCCESS"

    @property
    def is_empty(self) -> bool:
        """No prose, no structured payload and no error text."""
        return not self.response.strip() and self.structured_output is None and not self.error_text.strip()


def _error_text(payload: dict[str, object]) -> str:
    parts: list[str] = []
    for key in ("error", "errors", "error_message", "message", "detail", "details"):
        value = payload.get(key)
        if value is None or value == "":
            continue
        parts.append(value if isinstance(value, str) else json.dumps(value))
    return "\n".join(parts)


def parse_agy_json_envelope(stdout: str) -> AgyEnvelope | None:
    """The last top-level JSON object with a ``status`` key in agy's stdout.

    Tolerates PTY startup chrome, ANSI sequences and CRLF.  Never raises.
    """
    try:
        text = strip_ansi(stdout or "").replace("\r\n", "\n").replace("\r", "\n")
        decoder = json.JSONDecoder()
        found: dict[str, object] | None = None
        span = (0, 0)
        index = text.find("{")
        while index != -1:
            try:
                value, end = decoder.raw_decode(text, index)
            except ValueError:
                index = text.find("{", index + 1)
                continue
            if isinstance(value, dict) and "status" in value:
                found = value
                span = (index, end)
            index = text.find("{", end)
        if found is None:
            return None
        status = found.get("status")
        response = found.get("response")
        return AgyEnvelope(
            status=status if isinstance(status, str) else (None if status is None else str(status)),
            response=response if isinstance(response, str) else "",
            structured_output=found.get("structured_output"),
            error_text=_error_text(found),
            remainder=text[: span[0]] + text[span[1] :],
            has_json_schema=found.get("json_schema") is not None,
        )
    except Exception:  # noqa: BLE001 - a malformed envelope is a missing channel
        return None


def envelope_provider_error(envelope: AgyEnvelope) -> str:
    """The decoded operational error of a non-SUCCESS envelope (#2029).

    The envelope's own error text when present, else a status line, so a
    failed agy turn is never mistaken for a model format failure.
    """
    if envelope.error_text.strip():
        return envelope.error_text.strip()
    return f"agy reported status {envelope.status or 'unknown'} with no error text"


# --- Provider rejection of the request schema (#2039) ------------------------
#
# A schema construct the provider refuses fails the request before the model
# runs.  These detectors read provider-owned diagnostics only (a non-SUCCESS
# envelope's error/status, stdout outside the envelope, agy's own CLI log);
# model ``response`` prose is never evidence.
_INVALID_ARGUMENT_TOKEN = "INVALID_ARGUMENT"
_SCHEMA_REFERENCE_TOKENS = (
    "function_declarations",
    "parameters.properties",
    "json_schema",
    "response_schema",
    "responseschema",
)
_REJECTION_LINE_LIMIT = 2000


def structured_request_rejection_line(text: str | None) -> str | None:
    """The first line naming INVALID_ARGUMENT and the request schema, else None."""
    if not text:
        return None
    try:
        cleaned = strip_ansi(text).replace("\r\n", "\n").replace("\r", "\n")
    except Exception:  # noqa: BLE001 - detection never fails a turn
        return None
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if _INVALID_ARGUMENT_TOKEN not in line:
            continue
        lowered = line.lower()
        if any(token in lowered for token in _SCHEMA_REFERENCE_TOKENS):
            return line[:_REJECTION_LINE_LIMIT]
    return None


def envelope_request_rejection(envelope: AgyEnvelope | None) -> str | None:
    """Schema-rejection evidence from a non-SUCCESS envelope, else None.

    Reads the decoded error text and status only; ``response`` is never read.
    """
    if envelope is None or envelope.succeeded:
        return None
    line = structured_request_rejection_line(envelope.error_text)
    if line is not None:
        return line
    status = (envelope.status or "").upper()
    if status == _INVALID_ARGUMENT_TOKEN and (
        envelope.has_json_schema or "json_schema" in envelope.error_text.lower()
    ):
        detail = envelope.error_text.strip().splitlines()[0].strip() if envelope.error_text.strip() else ""
        summary = f"{_INVALID_ARGUMENT_TOKEN}: agy rejected the request json_schema"
        return (f"{summary} ({detail})" if detail else summary)[:_REJECTION_LINE_LIMIT]
    return None


def structured_output_signature(config: "AgentLoopConfig", model: str) -> str:
    """The Antigravity signature label, seat-labelled when a seat is active."""
    base = f"Google Antigravity: {model}"
    seat = config.active_reviewer_seat_id
    return f"{seat} ({base})" if seat else base


def render_structured_output_text(
    payload: object, kind: str, signature: str
) -> str | None:
    """Rebuild the public text from a ``structured_output`` payload, else None.

    A role payload gets its kind's footer carrying the payload's own state;
    an agent_unavailable payload gets the dedicated footer.  The renderer
    never invents a state, verdict or acknowledgement; a payload that cannot
    be rendered faithfully is a missing channel.
    """
    if not isinstance(payload, dict) or kind not in STRUCTURED_KIND_FOOTERS:
        return None
    body = json.dumps(payload, indent=2, ensure_ascii=False)
    payload_kind = payload.get("kind")
    if payload_kind == AGENT_UNAVAILABLE_KIND:
        return f"{body}\n\n{_AGENT_UNAVAILABLE_FOOTER}\n-- {signature}"
    if payload_kind != kind:
        return None
    state = payload.get("state")
    if not isinstance(state, str) or not state.strip():
        return None
    footer = STRUCTURED_KIND_FOOTERS[kind]
    return f"{body}\n\n<!-- {footer}: {state.strip()} -->\n-- {signature}"


def oversized_prompt_directive(response_path: Path, *, structured: bool) -> str:
    """The short ``--print`` value used when the task moves into GEMINI.md."""
    channel = (
        "Return your answer as the requested structured result (the JSON object the "
        f"schema describes); if you also write a file, write it to {response_path}."
        if structured
        else f"Write your complete answer to the public response file {response_path}."
    )
    return (
        "Follow the complete task included in the Agent Loop Task section of GEMINI.md. "
        f"{channel} Do not answer in chat: chat prose is discarded and counts as no response."
    )
