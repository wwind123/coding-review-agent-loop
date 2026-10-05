"""Lightweight transient/non-retryable agent-output classification.

Extracted from ``orchestrator`` so callers that must stay dependency-light
(e.g. the skill's ``helpers.run_external`` subprocess launcher) can decide
whether to retry an agent invocation without importing the full orchestrator.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from .reset_parsing import _parse_rate_limit_reset_seconds

TRANSIENT_AGENT_OUTPUT_RE = re.compile(
    r"Invalid stream|empty response|malformed tool call|"
    r"network (?:reset|timeout)|connection (?:reset|timed out|timeout)|"
    r"\btimed out\b|\btimeout\b|"
    r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout|"
    r"\b429\b|rate.?limit(?:ed)?|"
    r"session.?limit.?exceeded|session_limit_exceeded|too many sessions|"
    r"no capacity available|capacity.*(?:unavailable|exceeded)|\bat capacity\b|"
    r"resource.?exhausted|overloaded|"
    r"\bquota\b",
    re.I,
)
NON_RETRYABLE_AGENT_OUTPUT_RE = re.compile(
    r"\bauth(?:entication|orization)?\b|unauthorized|forbidden|invalid api key|"
    r"credit|billing|dirty (?:checkout|workdir|working tree)",
    re.I,
)


def is_transient_agent_output(text: str) -> bool:
    """Return True if ``text`` looks like a transient failure worth retrying.

    A match on a transient pattern is overridden by any non-retryable signal
    (auth/billing/dirty-checkout), which should never be retried blindly.
    """
    return bool(TRANSIENT_AGENT_OUTPUT_RE.search(text)) and not bool(
        NON_RETRYABLE_AGENT_OUTPUT_RE.search(text)
    )


# --- Codex provider error channel (#1269) -----------------------------------

_CODEX_BUDGET = 12000
_CODEX_EVENT_SHAPED_RE = re.compile(r'^\s*\{\s*"(?:type|id|item|thread_id)"\s*:')
_CODEX_STDIN_BANNER = "Reading prompt from stdin..."
_CODEX_NEUTRAL_TEXT = "codex exited without a provider error event"
_CODEX_FIELDS = ("message", "code", "type", "status", "detail")
_BILLING_GUARD_RE = re.compile(
    r"billing|credit|insufficient_quota|check your plan|payment required|\b402\b", re.I
)
_AVAILABILITY_RE = re.compile(
    r"at capacity|no capacity|capacity|overloaded|model_capacity_exhausted|"
    r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout|"
    r"\bstatus:\s*5\d\d\b|\bHTTP\s*5\d\d\b|\b429\b|rate.?limit|quota|"
    r"resource.?exhausted",
    re.I,
)
_STRUCTURED_AUTH_RE = re.compile(
    r"unauthorized|forbidden|invalid api key|\bauth(?:entication|orization)?\b|\b40[13]\b",
    re.I,
)


@dataclass(frozen=True)
class CodexStructuredError:
    rendered: str
    statuses: tuple[int, ...] = ()


@dataclass(frozen=True)
class CodexErrorChannel:
    structured_errors: tuple[CodexStructuredError, ...]
    stderr_lines: tuple[str, ...]


@dataclass(frozen=True)
class ProviderFailureVerdict:
    category: Literal["transient", "non-retryable", "deterministic"]
    billing: bool
    text: str
    source: Literal["structured", "stderr", "none"]


def _status_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        number = value
    elif isinstance(value, str) and value.strip().isdigit():
        number = int(value.strip())
    else:
        return None
    return number if 100 <= number <= 599 else None


def _render_codex_error(error: object) -> CodexStructuredError | None:
    if isinstance(error, str):
        text = error.strip()
        return CodexStructuredError(text) if text else None
    if not isinstance(error, dict):
        return None
    parts: list[str] = []
    statuses: list[int] = []

    def collect(mapping: dict) -> None:
        for field in _CODEX_FIELDS:
            value = mapping.get(field)
            if value is None or isinstance(value, (dict, list)):
                continue
            text = str(value).strip()
            if text:
                parts.append(f"{field}: {text}")
            if field in ("status", "code"):
                number = _status_value(value)
                if number is not None:
                    statuses.append(number)

    collect(error)
    nested = error.get("error")
    if isinstance(nested, dict):
        collect(nested)
    elif isinstance(nested, str) and nested.strip():
        parts.append(f"message: {nested.strip()}")
    if not parts:
        return None
    return CodexStructuredError("; ".join(parts), tuple(dict.fromkeys(statuses)))


def _bounded_text(text: str) -> str:
    """Head/tail bound for display text only; classification sees every error."""
    if len(text) <= _CODEX_BUDGET:
        return text
    half = _CODEX_BUDGET // 2
    return f"{text[:half]}\n...\n{text[-half:]}"


def extract_codex_error_channel(raw_output: str) -> CodexErrorChannel | None:
    """Split a Codex ``--json`` capture into provider errors and stderr lines.

    Returns None only when the capture is not event-shaped (plain CLI text), so
    callers keep the legacy path. Item events and undecodable ``{`` lines are
    never classified.
    """
    structured: list[CodexStructuredError] = []
    stderr: list[str] = []
    event_shaped = False
    for line in (raw_output or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("{"):
            try:
                event = json.loads(stripped)
            except ValueError:
                event = None
            if isinstance(event, dict) and isinstance(event.get("type"), str):
                event_shaped = True
                kind = event["type"]
                rendered = None
                if kind == "error":
                    rendered = _render_codex_error(event)
                elif kind == "turn.failed":
                    rendered = _render_codex_error(event.get("error"))
                if rendered is not None and rendered not in structured:
                    structured.append(rendered)
            elif _CODEX_EVENT_SHAPED_RE.match(stripped):
                event_shaped = True
            continue
        if stripped == _CODEX_STDIN_BANNER:
            continue
        if stripped not in stderr:
            stderr.append(stripped)
    if not event_shaped:
        return None
    return CodexErrorChannel(tuple(structured), tuple(stderr))


def _classify_structured_provider_error(err: CodexStructuredError) -> tuple[str, bool]:
    text = err.rendered
    if 402 in err.statuses or _BILLING_GUARD_RE.search(text):
        return "non-retryable", True
    if any(s >= 500 or s == 429 for s in err.statuses) or _AVAILABILITY_RE.search(text):
        return "transient", False
    if any(s in (401, 403) for s in err.statuses) or _STRUCTURED_AUTH_RE.search(text):
        return "non-retryable", False
    if TRANSIENT_AGENT_OUTPUT_RE.search(text):
        return "transient", False
    return "deterministic", False


def classify_codex_failure(raw_output: str) -> ProviderFailureVerdict | None:
    """Provider-channel verdict for a failed Codex invocation, or None (legacy path)."""
    channel = extract_codex_error_channel(raw_output)
    if channel is None:
        return None
    if channel.structured_errors:
        results = [_classify_structured_provider_error(e) for e in channel.structured_errors]
        text = _bounded_text("\n".join(e.rendered for e in channel.structured_errors))
        bad = [r for r in results if r[0] == "non-retryable"]
        if bad:
            return ProviderFailureVerdict(
                "non-retryable", any(r[1] for r in bad), text, "structured"
            )
        if any(r[0] == "transient" for r in results):
            return ProviderFailureVerdict("transient", False, text, "structured")
        return ProviderFailureVerdict("deterministic", False, text, "structured")
    if channel.stderr_lines:
        full = "\n".join(channel.stderr_lines)
        text = _bounded_text(full)
        if NON_RETRYABLE_AGENT_OUTPUT_RE.search(full):
            category = "non-retryable"
        elif TRANSIENT_AGENT_OUTPUT_RE.search(full):
            category = "transient"
        else:
            category = "deterministic"
        return ProviderFailureVerdict(category, False, text, "stderr")
    return ProviderFailureVerdict("deterministic", False, _CODEX_NEUTRAL_TEXT, "none")


@dataclass(frozen=True)
class AntigravityCapacityClassification:
    """Provider-scoped capacity result used to decide model fallback."""

    is_capacity: bool
    diagnostic: str = ""
    # The provider-owned error frame(s) that matched (#1236); empty when none.
    frame: str = ""


@dataclass(frozen=True)
class QuotaExhaustion:
    """Verified quota exhaustion read from a provider error frame."""

    reset_seconds: int | None
    reset_source: Literal["parsed", "cooldown"]
    frame: str


# Same value as agent_failure.LONG_RESET_THRESHOLD_SECONDS; callers pass theirs.
ANTIGRAVITY_LONG_RESET_THRESHOLD_SECONDS = 300
_ANTIGRAVITY_AGY_ERROR_RE = re.compile(r"^\s*AGY_ERROR:\s*\{")
_ANTIGRAVITY_JSON_KEY_RE = re.compile(r'"(?:error|code|status|message)"\s*:', re.I)
_ANTIGRAVITY_FRAME_JSON_LINE_CAP = 40
_ANTIGRAVITY_FRAME_RETRY_DELAY_RE = re.compile(
    r"(?i)[\"']?retry[_-]?delay[\"']?\s*:\s*[\"']?(\d+)(?:\.\d+)?s[\"']?"
)
_ANTIGRAVITY_CAPACITY_ONLY_RE = re.compile(
    r"(?i)high traffic|try again in a minute|overload|no capacity|capacity"
)
_ANTIGRAVITY_HARD_QUOTA_RE = re.compile(r"(?i)\bquota\b|resource[ _-]?exhausted")


_ANTIGRAVITY_ERROR_LINE_RE = re.compile(r"(?i)^\s*(?:error|fatal)\s*[:\[]")
_ANTIGRAVITY_ERROR_JSON_RE = re.compile(
    r'^\s*\{.{0,2000}?"(?:error|code|status|message)"\s*:', re.I
)


def _json_frame_end(raw_lines: list[str], start: int) -> int:
    """Index of the line where the object opened on ``raw_lines[start]`` balances."""
    depth = 0
    in_string = False
    escaped = False
    opened = False
    limit = min(len(raw_lines), start + _ANTIGRAVITY_FRAME_JSON_LINE_CAP)
    for index in range(start, limit):
        line = raw_lines[index]
        begin = line.find("{") if index == start else 0
        if begin < 0:
            return index
        for char in line[begin:]:
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
                opened = True
            elif char == "}":
                depth -= 1
        if opened and depth <= 0:
            return index
    return limit - 1


def _antigravity_retained_window(text: str) -> list[str]:
    """Raw (unstripped) lines of the retained diagnostic window.

    Mirrors the 2400-character bound of the classifier, but keeps only the
    tail when the output is longer so a frame never spans a head/tail splice.
    """
    body = text.rstrip()
    if len(text.strip()) <= 2400:
        return body.splitlines()
    tail = body[-1200:]
    _first, sep, rest = tail.partition("\n")
    return rest.splitlines() if sep else []


def _antigravity_provider_frames(raw_lines: list[str]) -> list[tuple[list[str], int]]:
    """Provider error frames in ``raw_lines`` as ``(lines, last_nonempty_position)``.

    ``last_nonempty_position`` indexes the frame's last line among the window's
    non-empty lines, so callers can tell whether the frame is terminal.
    """
    nonempty_position: dict[int, int] = {}
    for index, line in enumerate(raw_lines):
        if line.strip():
            nonempty_position[index] = len(nonempty_position)
    frames: list[tuple[list[str], int]] = []
    index = 0
    while index < len(raw_lines):
        line = raw_lines[index]
        end = None
        if _ANTIGRAVITY_AGY_ERROR_RE.match(line) or _ANTIGRAVITY_ERROR_JSON_RE.match(line):
            end = _json_frame_end(raw_lines, index)
        elif line.strip().startswith("{") and any(
            _ANTIGRAVITY_JSON_KEY_RE.search(candidate)
            for candidate in raw_lines[index + 1: index + 4]
        ):
            end = _json_frame_end(raw_lines, index)
        elif _ANTIGRAVITY_ERROR_LINE_RE.match(line):
            indent = len(line) - len(line.lstrip())
            end = index
            while end + 1 < len(raw_lines):
                following = raw_lines[end + 1]
                if not following.strip() or len(following) - len(following.lstrip()) <= indent:
                    break
                end += 1
        if end is None:
            index += 1
            continue
        chunk = raw_lines[index: end + 1]
        last = max(i for i in range(index, end + 1) if i in nonempty_position)
        frames.append((chunk, nonempty_position[last]))
        index = end + 1
    return frames


def _antigravity_frame_reset_seconds(frame: str) -> int | None:
    """Reset delay stated inside one provider frame, or None."""
    match = _ANTIGRAVITY_FRAME_RETRY_DELAY_RE.search(frame)
    if match:
        return int(match.group(1))
    return _parse_rate_limit_reset_seconds(frame)


def classify_antigravity_quota_exhaustion(
    capacity: AntigravityCapacityClassification,
    *,
    threshold: int = ANTIGRAVITY_LONG_RESET_THRESHOLD_SECONDS,
) -> QuotaExhaustion | None:
    """Verified quota exhaustion, judged only from the matched provider frame."""
    if not capacity.is_capacity or not capacity.frame:
        return None
    frame = capacity.frame
    reset = _antigravity_frame_reset_seconds(frame)
    if reset is not None:
        return QuotaExhaustion(reset, "parsed", frame) if reset > threshold else None
    if _ANTIGRAVITY_CAPACITY_ONLY_RE.search(frame):
        return None
    if _ANTIGRAVITY_HARD_QUOTA_RE.search(frame):
        return QuotaExhaustion(None, "cooldown", frame)
    return None


def classify_antigravity_capacity(
    text: str,
    *,
    returncode: int | None,
    empty_response: bool,
    signatures: tuple[str, ...],
) -> AntigravityCapacityClassification:
    """Recognize framed Antigravity provider-capacity diagnostics.

    Only failed or empty invocations are eligible. A signature on a provider
    error line (or its adjacent detail line), a compact JSON error line, or an
    unframed standalone provider diagnostic qualifies. Framed provider errors
    must be terminal output so unrelated transcript wording cannot suppress a
    real capacity failure; authentication/billing retain non-retryable precedence.
    """
    if (returncode in (None, 0) and not empty_response) or not text.strip():
        return AntigravityCapacityClassification(False)
    if NON_RETRYABLE_AGENT_OUTPUT_RE.search(text):
        return AntigravityCapacityClassification(False)
    # A provider normally prints its failure at either end; retaining a bounded
    # tail avoids accepting a capacity phrase from an unrelated long transcript.
    stripped = text.strip()
    diagnostic = stripped if len(stripped) <= 2400 else f"{stripped[:1200]}\n{stripped[-1200:]}"
    lines = [line.strip() for line in diagnostic.splitlines() if line.strip()]
    lowered_signatures = tuple(signature.lower() for signature in signatures)

    def has_signature(value: str) -> bool:
        return any(signature in value.lower() for signature in lowered_signatures)

    window_frames = _antigravity_provider_frames(_antigravity_retained_window(text))
    nonempty_count = len([line for line in _antigravity_retained_window(text) if line.strip()])
    terminal_frames = [
        "\n".join(chunk)
        for chunk, last in window_frames
        if last >= nonempty_count - 3 and has_signature("\n".join(chunk))
    ]
    if terminal_frames:
        return AntigravityCapacityClassification(True, diagnostic, "\n".join(terminal_frames))

    for index, line in enumerate(lines):
        if _ANTIGRAVITY_ERROR_LINE_RE.match(line) or _ANTIGRAVITY_ERROR_JSON_RE.match(line):
            adjacent = "\n".join(lines[max(0, index - 1): index + 2])
            # agy emits an error immediately before exiting. Limit the frame to
            # the last few non-empty lines rather than inspecting normal reviewer
            # transcript text that may precede it.
            if has_signature(adjacent) and index >= len(lines) - 3:
                header_frames = [
                    "\n".join(chunk) for chunk, _last in window_frames
                    if chunk[0].strip() == line
                ]
                # No verified raw-window frame (e.g. the header only survives in
                # the head/tail splice): keep capacity, never exhaustion provenance.
                return AntigravityCapacityClassification(
                    True, diagnostic, header_frames[-1] if header_frames else ""
                )

    # Older agy versions emitted bare provider diagnostics (for example,
    # "quota exceeded please try again") on a non-zero exit. Preserve that
    # fallback trigger. A bare diagnostic must be a single line beginning with
    # a configured capacity signature, so an incidental phrase embedded in a
    # review transcript cannot qualify.
    if len(lines) == 1 and any(
        lines[0].lower().startswith(signature) for signature in lowered_signatures
    ):
        return AntigravityCapacityClassification(True, diagnostic, lines[0])
    return AntigravityCapacityClassification(False)


# Phrases an agent uses when it has started required work (tests, builds) in
# the background and ends its turn waiting on it instead of finishing in the
# foreground (#588). This is purely textual: it makes no assumption about
# marker presence/absence. Eligibility for completion recovery is gated
# separately by the caller's own terminal-result validator having already
# rejected the response, so an embedded (but invalid-for-that-validator)
# marker never exempts a response from matching here.
BACKGROUNDED_COMPLETION_RE = re.compile(
    r"(?i)"
    r"\bi(?:'|')?ll wait\b|"
    r"\bwait(?:ing)? (?:for|on) (?:the )?background|"
    r"\brun(?:s|ning)? (?:it |them )?in the background\b|"
    r"\b(?:test|build|suite)\w*\b.{0,60}\bin the background\b|"
    r"\bin the background\b.{0,60}\b(?:test|build|suite)|"
    r"\b(?:you(?:'|')?ll|i(?:'|')?ll|we(?:'|')?ll) (?:get|be) notified\b|"
    r"\bwill (?:get|be) notified\b|"
    r"\blet me wait for\b|"
    r"\bonce (?:the |it )?(?:background )?(?:test|build|suite).{0,40}finish"
)


def looks_like_backgrounded_completion(text: str) -> bool:
    """Return True if ``text`` reads like the agent deferred to background work.

    Purely a phrase match; callers must independently confirm the response
    failed the relevant terminal-result validator before treating this as
    grounds for a completion-recovery attempt (#588).
    """
    return bool(BACKGROUNDED_COMPLETION_RE.search(text))
