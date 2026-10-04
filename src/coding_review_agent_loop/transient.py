"""Lightweight transient/non-retryable agent-output classification.

Extracted from ``orchestrator`` so callers that must stay dependency-light
(e.g. the skill's ``helpers.run_external`` subprocess launcher) can decide
whether to retry an agent invocation without importing the full orchestrator.
"""

from __future__ import annotations

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
    r"no capacity available|capacity.*(?:unavailable|exceeded)|"
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
