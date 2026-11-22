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

from .protocol_markers import sanitize_historical_text
from .runner import strip_ansi

if TYPE_CHECKING:
    from .config import AgentLoopConfig

MAX_SAME_AGENT_REASKS = 2
# The heading every re-ask appendix starts with.
REASK_PROMPT_HEADER = "## Previous response rejected (re-ask"

# Bound on the validator message quoted back to the agent.
VALIDATOR_MESSAGE_MAX_CHARS = 2000

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


def build_reask_prompt(
    original_prompt: str,
    validator_message: str,
    attempt: int,
    *,
    coder: bool = False,
    fresh_evidence: bool = False,
    config: AgentLoopConfig | None = None,
) -> str:
    """Append the re-ask instruction to ``original_prompt``.

    ``original_prompt`` may already carry a typed hint section (missing IDs,
    allowed values, evidence reason, contract field).  A coder re-ask forbids
    touching the checkout; a fresh-evidence coder re-ask (a non-citable
    observation) instead allows rerunning the covering test through the test
    wrapper, while still forbidding commits and pushes.
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
        "Resend the complete corrected response only, in exactly the required format. "
        "Keep every judgement, claim and item from your previous response unless the "
        "validator message requires a change.",
    ]
    if coder and fresh_evidence:
        lines.append(
            "You may rerun the covering test through the test wrapper as instructed above. "
            "Do not edit files, commit, push, or otherwise change the checkout or the PR."
        )
    elif coder:
        lines.append(
            "Do not modify, commit or push anything, do not run tests or other commands, "
            "and do not touch the checkout: only resend the response. The checkout head is "
            "compared before and after this re-ask, and a moved head ends the turn."
        )
    return "\n".join(lines) + "\n"
