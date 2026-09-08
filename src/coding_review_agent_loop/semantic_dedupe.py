"""Strict, bounded semantic matching for approved follow-up trackers.

The publisher owns policy (which candidates are eligible and when a match may
be reused).  This module only builds the small prompt, invokes the selected
cheap provider, and validates its JSON result.  Keeping that boundary narrow
makes provider failures conservative and keeps the model from becoming a
second publishing authority.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Literal, TYPE_CHECKING

from .agents.base import AgentName
from .agents.registry import run_agent_result
from .errors import AgentLoopError
from .protocol_markers import sanitize_historical_text

if TYPE_CHECKING:
    from .config import AgentLoopConfig
    from .runner import Runner
    from .usage import RunUsageContext


SemanticConfidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class SemanticCandidate:
    """A bounded, repository-validated candidate presented to the matcher."""

    identity: int | str
    title: str
    body: str


@dataclass(frozen=True)
class SemanticMatch:
    duplicate_of: int | str | None
    confidence: SemanticConfidence
    reason: str


@dataclass(frozen=True)
class SemanticProviderResult:
    text: str
    result: object | None = None


SemanticTransport = Callable[[str, "Runner", "AgentLoopConfig", float], SemanticProviderResult | str]


def _excerpt(text: str, limit: int) -> str:
    # Prompt data is historical/untrusted too.  It is sanitized before it is
    # copied into the provider prompt and bounded so a large issue cannot turn
    # a cheap reconciliation call into a full review.
    safe = sanitize_historical_text(text or "")
    return safe[:limit]


def _identity_label(identity: int | str) -> str:
    if isinstance(identity, bool):
        raise AgentLoopError("semantic candidate identity cannot be boolean")
    if isinstance(identity, int):
        if identity <= 0:
            raise AgentLoopError("semantic issue identity must be positive")
        return f"issue:{identity}"
    if isinstance(identity, str) and identity.startswith("group-"):
        suffix = identity.removeprefix("group-")
        if suffix.isdigit() and int(suffix) > 0:
            return identity
    raise AgentLoopError(f"invalid semantic candidate identity: {identity!r}")


def parse_semantic_match(raw: str, *, allowed_ids: set[int | str]) -> SemanticMatch:
    """Validate the exact semantic matcher contract without repair or coercion."""
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgentLoopError("semantic dedupe provider returned invalid JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"duplicate_of", "confidence", "reason"}:
        raise AgentLoopError("semantic dedupe result must contain exactly duplicate_of, confidence, and reason")

    duplicate_of = payload["duplicate_of"]
    if duplicate_of is not None:
        if isinstance(duplicate_of, bool) or not isinstance(duplicate_of, (int, str)):
            raise AgentLoopError("semantic duplicate_of has the wrong type")
        if duplicate_of not in allowed_ids:
            # Existing-issue responses may use either the documented numeric
            # form or the explicit issue:N form.  Both must be known locally.
            if not (
                isinstance(duplicate_of, str)
                and duplicate_of.startswith("issue:")
                and duplicate_of.removeprefix("issue:").isdigit()
                and int(duplicate_of.removeprefix("issue:")) in allowed_ids
            ):
                raise AgentLoopError("semantic duplicate_of is not an allowed candidate")
            duplicate_of = int(duplicate_of.removeprefix("issue:"))

    confidence = payload["confidence"]
    if confidence not in {"high", "medium", "low"}:
        raise AgentLoopError("semantic confidence must be high, medium, or low")
    reason = payload["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise AgentLoopError("semantic reason must be a non-empty string")
    return SemanticMatch(
        duplicate_of=duplicate_of,
        confidence=confidence,
        reason=reason.strip(),
    )


def build_semantic_prompt(
    *,
    proposed: str,
    candidates: tuple[SemanticCandidate, ...],
    source_context: str,
    prompt_char_limit: int,
) -> str:
    if not candidates:
        raise AgentLoopError("semantic dedupe requires at least one candidate")
    if prompt_char_limit <= 0:
        raise AgentLoopError("semantic prompt budget must be positive")

    instruction_lines = [
        "You are a cheap semantic duplicate classifier for approved future follow-up issues.",
        "Do not use tools, browse, execute commands, or infer facts outside this prompt.",
        "Compare the actual deliverable, not merely shared topic words.",
        "Return one strict JSON object and no markdown or explanatory text:",
        '{"duplicate_of": null, "confidence": "low", "reason": "No candidate tracks the same deliverable."}',
        "Use duplicate_of=null when no candidate is equivalent. Only use high confidence when the proposed work is the same deliverable; related or complementary work is not a duplicate.",
    ]

    # Reserve the candidate portion explicitly.  The old implementation gave
    # each candidate a budget before accounting for these instructions and
    # then sliced the completed prompt, which could remove the tail of a
    # candidate entry while leaving that identity in the allowed-id set.
    base_prefix = "\n".join((*instruction_lines, "Source context: ", "Proposed follow-up: ", "Candidates:"))
    available = prompt_char_limit - len(base_prefix) - 1
    if available <= 0:
        raise AgentLoopError("semantic prompt budget is too small for the matcher instructions")
    source_limit = min(1200, max(0, available // 8))
    proposed_limit = min(1600, max(0, available // 5))
    prefix = "\n".join(
        (
            *instruction_lines,
            f"Source context: {_excerpt(source_context, source_limit)}",
            f"Proposed follow-up: {_excerpt(proposed, proposed_limit)}",
            "Candidates:",
        )
    )
    # ``str.join`` inserts one separator before every candidate entry.
    remaining = prompt_char_limit - len(prefix) - len(candidates)
    candidate_budget = remaining // len(candidates)
    if candidate_budget <= 0:
        raise AgentLoopError("semantic prompt budget is too small for candidate identities")

    def render_candidate(candidate: SemanticCandidate) -> str:
        label = _identity_label(candidate.identity)
        fixed = len(f"- {label}:\n  title: \n  body: ")
        if fixed > candidate_budget:
            raise AgentLoopError("semantic prompt budget is too small for candidate identities")
        text_budget = candidate_budget - fixed
        title_limit = text_budget // 3
        body_limit = text_budget - title_limit
        return "\n".join(
            (
                f"- {label}:",
                f"  title: {_excerpt(candidate.title, title_limit)}",
                f"  body: {_excerpt(candidate.body, body_limit)}",
            )
        )

    candidate_lines: list[str] = []
    for candidate in candidates:
        candidate_lines.append(render_candidate(candidate))
    prompt = "\n".join((prefix, *candidate_lines))
    if len(prompt) > prompt_char_limit:
        raise AgentLoopError("semantic prompt candidate packing exceeded its budget")
    return prompt


def _isolated_provider_config(config: "AgentLoopConfig", backend: AgentName, model: str):
    """Remove configured tool flags and place the provider in an empty temp dir."""
    # No configured checkout or dangerous-agent flag is exposed to this
    # read-only classification turn.
    isolated: Path | None = None
    try:
        isolated = Path(tempfile.mkdtemp(prefix="coding-review-followup-dedupe-"))
        values: dict[str, object] = {
            "coder": backend,
            "reviewer": (backend,),
            "claude_dir": isolated,
            "codex_dir": isolated,
            "gemini_dir": isolated,
            "antigravity_dir": isolated,
            "claude_args": (),
            "codex_args": (),
            "gemini_args": (),
            "antigravity_args": (),
            "dry_run": False,
        }
        if backend == "claude":
            values["claude_model"] = model
        elif backend == "codex":
            values["codex_model"] = model
        elif backend == "gemini":
            values["gemini_model"] = model
        elif backend == "antigravity":
            values["antigravity_model"] = None
            values["antigravity_models"] = (model or config.antigravity_models[0],)
        return replace(config, **values), isolated
    except Exception:
        if isolated is not None:
            import shutil

            shutil.rmtree(isolated, ignore_errors=True)
        raise


def default_semantic_transport(
    prompt: str,
    runner: "Runner",
    config: "AgentLoopConfig",
    timeout_seconds: float,
) -> SemanticProviderResult:
    backend = config.semantic_followup_backend
    model = config.semantic_followup_model.strip()
    isolated_dir: Path | None = None
    try:
        isolated_config, isolated_dir = _isolated_provider_config(config, backend, model)
        result = run_agent_result(
            runner,
            agent=backend,
            config=isolated_config,
            prompt=prompt,
            role="semantic-dedupe",
            label="semantic-followup-dedupe",
            timeout_seconds=timeout_seconds,
        )
        return SemanticProviderResult(text=result.text, result=result)
    finally:
        # The directory is outside the repository and contains no durable
        # orchestration state.  Best-effort cleanup is deliberately omitted
        # from exception handling so provider/control-flow errors propagate.
        import shutil

        if isolated_dir is not None:
            shutil.rmtree(isolated_dir, ignore_errors=True)


class SemanticDedupeMatcher:
    """Bounded matcher used by both in-batch and existing-issue reconciliation."""

    def __init__(
        self,
        *,
        runner: "Runner",
        config: "AgentLoopConfig",
        transport: SemanticTransport | None = None,
        usage_context: "RunUsageContext | None" = None,
    ) -> None:
        self.runner = runner
        self.config = config
        self.transport = transport or default_semantic_transport
        self.usage_context = usage_context
        self.calls = 0

    def match(
        self,
        *,
        proposed: str,
        candidates: tuple[SemanticCandidate, ...],
        source_context: str,
    ) -> SemanticMatch:
        if self.calls >= self.config.semantic_followup_max_calls:
            raise BudgetExhausted("semantic call budget exhausted")
        if len(candidates) > self.config.semantic_followup_max_candidates:
            candidates = candidates[: self.config.semantic_followup_max_candidates]
        if not candidates:
            raise AgentLoopError("semantic dedupe requires a narrowed candidate set")
        self.calls += 1
        prompt = build_semantic_prompt(
            proposed=proposed,
            candidates=candidates,
            source_context=source_context,
            prompt_char_limit=self.config.semantic_followup_prompt_char_limit,
        )
        response = self.transport(
            prompt,
            self.runner,
            self.config,
            float(self.config.semantic_followup_timeout_seconds),
        )
        if isinstance(response, str):
            raw = response
            provider_result = None
        else:
            raw = response.text
            provider_result = response.result
        usage_record = None
        if self.usage_context is not None and provider_result is not None:
            from .usage import estimate_usage

            usage = getattr(provider_result, "usage", None) or estimate_usage(prompt, raw)
            usage_record = self.usage_context.add_record(
                agent=self.config.semantic_followup_backend,
                session_id=getattr(provider_result, "session_id", None),
                returncode=getattr(provider_result, "returncode", 0),
                usage=usage,
                raw_backend_usage=getattr(provider_result, "raw_usage", None),
                turn_role="semantic-dedupe",
                model=getattr(provider_result, "model_used", None),
                configured_model=getattr(provider_result, "configured_model", None),
                configured_effort=getattr(provider_result, "configured_effort", None),
                effort_source=getattr(provider_result, "effort_source", None),
                observed_model=getattr(provider_result, "observed_model", None),
                observed_effort=getattr(provider_result, "observed_effort", None),
                observation_provenance=getattr(provider_result, "observation_provenance", None),
                outcome="succeeded",
                log_path=str(getattr(provider_result, "log_path", "")) or None,
                containment=(getattr(provider_result, "containment", None).to_dict()
                            if getattr(provider_result, "containment", None) is not None
                            and hasattr(getattr(provider_result, "containment", None), "to_dict")
                            else None),
            )
        allowed = {candidate.identity for candidate in candidates}
        try:
            match = parse_semantic_match(raw, allowed_ids=allowed)
        except Exception:
            if usage_record is not None:
                usage_record.outcome = "invalid_output"
                usage_record.validation_status = "invalid"
            raise
        if usage_record is not None:
            usage_record.validation_status = "validated"
        return match


class BudgetExhausted(AgentLoopError):
    """Local semantic budget exhaustion; callers retain deterministic behavior."""


__all__ = [
    "BudgetExhausted",
    "SemanticCandidate",
    "SemanticConfidence",
    "SemanticDedupeMatcher",
    "SemanticMatch",
    "SemanticProviderResult",
    "SemanticTransport",
    "build_semantic_prompt",
    "default_semantic_transport",
    "parse_semantic_match",
]
