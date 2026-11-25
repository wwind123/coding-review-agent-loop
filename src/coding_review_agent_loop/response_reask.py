"""The single same-agent re-ask rule for rejected structured responses (#2066).

A deterministic validation failure goes back to the agent that produced the
response, with the sanitized validator message and an instruction to resend
the complete response only.  A turn gets at most :data:`MAX_SAME_AGENT_REASKS`
re-asks; they replace the cross-model repair pass and the older per-defect
re-ask slots.  This module owns only the prompt text and its sanitization;
the validated turn in ``validated_agent.py`` owns the budget, the session and
model pinning, the coder head guard and the evidence scope.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .protocol_markers import sanitize_historical_text, sanitize_untrusted_prose
from .runner import strip_ansi

if TYPE_CHECKING:
    from .config import AgentLoopConfig

MAX_SAME_AGENT_REASKS = 2
# The heading every re-ask appendix starts with.
REASK_PROMPT_HEADER = "## Previous response rejected (re-ask"

# Bound on the validator message quoted back to the agent.
VALIDATOR_MESSAGE_MAX_CHARS = 2000
# Bound on the rejected response quoted back to a session-less agent (#2128).
REASK_PRIOR_RESPONSE_MAX_CHARS = 24000
# Characters of the rejected response's tail kept after truncation.
REASK_PRIOR_RESPONSE_TAIL_CHARS = 2000
# Issue #2157: clauses for a coder re-ask that resumes its session.  Work the
# previous turn started in the background may still be running; the agent may
# wait on it, but starting or finishing implementation work belongs only to
# the one-time completion-recovery continuation.
REASK_BACKGROUND_WAIT_CLAUSE = (
    "Background work you started in the previous turn (subagents, Agent/Task "
    "tools, shell commands) may still be running. Before you resend, you may wait "
    "for that existing work to finish, using only that work's own result or wait "
    "mechanism (reading a subagent's or task's final output, or one bounded "
    "blocking wait on that job). Do not poll in a loop."
)
REASK_NO_NEW_WORK_CLAUSE = (
    "Do not start new work: no new subagents, Agent/Task calls, tests or other "
    "commands, no file edits, and no commits or pushes."
)
REASK_REPORT_FINAL_STATE_CLAUSE = (
    "When that work has finished or is no longer running, report the real final "
    "state in the required format. Put unfinished work in the response's "
    "remaining items, not in an interim progress message, and do not complete "
    "unfinished work yourself in this re-ask."
)
REASK_HEAD_GUARD_CLAUSE = (
    "The checkout head is compared before and after this re-ask, and a moved head "
    "ends the turn."
)
REASK_CODER_NO_COMMAND_LINE = (
    "Do not modify, commit or push anything, do not run tests or other commands, "
    "and do not touch the checkout: only resend the response. " + REASK_HEAD_GUARD_CLAUSE
)

PRIOR_RESPONSE_HEADING = (
    "Your previous (rejected) response, for reference — keep its judgements:"
)

_KNOWN_ERROR_RE = re.compile(
    r"(?i)(fatal|error|exception|authentication|unauthorized|forbidden|quota|rate.?limit|timed?.?out)"
)
_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|authorization|password)\b\s*[:=]\s*\S+"
)


def _redact(text: str, *, config: AgentLoopConfig | None = None) -> str:
    clean = strip_ansi(text)
    clean = _SECRET_RE.sub(r"\1=<redacted>", clean)
    paths: list[Path] = [Path.home()]
    repair_root = Path(tempfile.gettempdir()) / "coding-review-agent-loop" / "repair"
    if config is not None:
        paths.extend(
            (config.claude_dir, config.codex_dir, config.gemini_dir, config.antigravity_dir)
        )
    for path in paths:
        rendered = str(path)
        if rendered and rendered != "/":
            clean = clean.replace(rendered, "<path>")
    return clean.replace(str(repair_root), "<repair-path>")


def _sanitize_diagnostic(text: str, *, config: AgentLoopConfig | None = None) -> str:
    """Redact and reduce backend diagnostics to their last error line (moved from repair.py)."""
    clean = _redact(text, config=config)
    lines = [line.strip() for line in clean.splitlines() if line.strip()]
    matching = [line for line in lines if _KNOWN_ERROR_RE.search(line)]
    selected = matching[-1:] if matching else lines[-20:]
    encoded = "\n".join(selected).encode("utf-8")[-4096:]
    return encoded.decode("utf-8", errors="ignore")


def sanitize_validator_message(
    message: str,
    *,
    config: AgentLoopConfig | None = None,
    limit: int = VALIDATOR_MESSAGE_MAX_CHARS,
) -> str:
    """Return a bounded, single-line, marker-free copy of a validator message.

    Reserved protocol records are replaced by their safe historical labels and
    angle brackets are neutralized, so the prompt never echoes a marker the
    response was rejected for.  Secrets and local paths are redacted.
    """
    clean = _redact(message, config=config)
    clean = sanitize_historical_text(clean).replace("<", "(").replace(">", ")")
    clean = " ".join(clean.split())
    if len(clean) > limit:
        clean = clean[: max(limit - 3, 0)] + "..."
    return clean


def sanitize_prior_response(
    text: str,
    *,
    config: AgentLoopConfig | None = None,
    limit: int = REASK_PRIOR_RESPONSE_MAX_CHARS,
) -> str:
    """Return a bounded, marker-free copy of a rejected response (#2128).

    Used only when the re-ask cannot resume the agent's session, so the agent
    sees what it sent.  Unlike :func:`sanitize_validator_message` the line
    structure is kept.  Secrets and local paths are redacted, reserved
    protocol records and names are replaced by their safe labels, and HTML
    comment delimiters are neutralized so no complete record is echoed.  An
    oversized response keeps its head and a short tail around an explicit
    omission note; slicing is by code point, so no UTF-8 character is split.
    """
    clean = _redact(text, config=config)
    clean = sanitize_untrusted_prose(clean)
    clean = clean.replace("<!--", "(!--").replace("-->", "--)")
    if len(clean) <= limit:
        return clean
    tail_chars = min(REASK_PRIOR_RESPONSE_TAIL_CHARS, max(limit // 4, 0))
    # Reserve room for the widest possible note: the omitted count never
    # exceeds the full length, so the result never exceeds ``limit``.
    reserve = len(f"\n[... {len(clean)} characters omitted ...]\n")
    head_chars = max(limit - tail_chars - reserve, 0)
    omitted = len(clean) - head_chars - tail_chars
    note = f"\n[... {omitted} characters omitted ...]\n"
    tail = clean[len(clean) - tail_chars :] if tail_chars else ""
    return clean[:head_chars] + note + tail


def build_reask_prompt(
    original_prompt: str,
    validator_message: str,
    attempt: int,
    *,
    coder: bool = False,
    fresh_evidence: bool = False,
    config: AgentLoopConfig | None = None,
    prior_response: str | None = None,
) -> str:
    """Append the re-ask instruction to ``original_prompt``.

    ``prior_response`` is the rejected response text, passed only when the
    re-ask cannot resume the agent's session (Codex, Antigravity, or any
    result without a session).  It is quoted back bounded and sanitized
    between the validator message and the resend instruction; a
    session-backed re-ask omits it so the response is not duplicated.

    ``original_prompt`` may already carry a typed hint section (missing IDs,
    allowed values, evidence reason, contract field).  A coder re-ask forbids
    touching the checkout; a fresh-evidence coder re-ask (a non-citable
    observation) instead allows rerunning the covering test through the test
    wrapper, while still forbidding commits and pushes.  A session-backed coder
    re-ask may also wait on background work the previous turn started (#2157),
    but never start new work or finish the implementation.
    """
    detail = sanitize_validator_message(validator_message, config=config) or "(no detail)"
    lines = [
        original_prompt,
        "",
        f"{REASK_PROMPT_HEADER} {attempt} of {MAX_SAME_AGENT_REASKS})",
        "",
        "The orchestrator rejected your previous response during validation.",
        f"Validator message: {detail}",
        "",
    ]
    if prior_response is not None:
        quoted = sanitize_prior_response(prior_response, config=config)
        lines.extend(
            [
                PRIOR_RESPONSE_HEADING,
                "",
                "----- BEGIN PREVIOUS RESPONSE -----",
                quoted,
                "----- END PREVIOUS RESPONSE -----",
                "",
            ]
        )
    lines += [
        "Resend the complete corrected response only, in exactly the required format. "
        "Keep every judgement, claim and item from your previous response unless the "
        "validator message requires a change.",
    ]
    # A session-backed re-ask resumes a turn that may have left background
    # work running (#2157); a session-less one is a fresh turn with none.
    resumed = prior_response is None
    if coder and fresh_evidence:
        if resumed:
            lines.append(REASK_BACKGROUND_WAIT_CLAUSE)
        lines.append(
            "You may rerun the covering test through the test wrapper as instructed above. "
            "Do not edit files, commit, push, or otherwise change the checkout or the PR."
        )
    elif coder and resumed:
        lines.extend(
            [
                REASK_BACKGROUND_WAIT_CLAUSE,
                REASK_NO_NEW_WORK_CLAUSE,
                REASK_REPORT_FINAL_STATE_CLAUSE,
                REASK_HEAD_GUARD_CLAUSE,
            ]
        )
    elif coder:
        lines.append(REASK_CODER_NO_COMMAND_LINE)
    return "\n".join(lines) + "\n"
