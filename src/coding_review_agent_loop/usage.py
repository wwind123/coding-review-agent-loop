"""Per-call usage normalization and run-level aggregation."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

AgentName = Literal["claude", "codex", "gemini", "antigravity"]
UsageMode = Literal["exact", "partial", "estimated"]

NUMERIC_USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "input_chars",
    "output_chars",
    "input_bytes",
    "output_bytes",
)


def estimate_token_count(text: str) -> int:
    byte_count = len(text.encode("utf-8"))
    if byte_count == 0:
        return 0
    return max(1, (byte_count + 3) // 4)


def coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def first_present(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


@dataclass(frozen=True)
class UsageMetadata:
    mode: UsageMode
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    input_chars: int | None = None
    output_chars: int | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None

    def with_io_sizes(self, *, prompt: str, response: str) -> UsageMetadata:
        return UsageMetadata(
            mode=self.mode,
            input_tokens=self.input_tokens,
            cached_input_tokens=self.cached_input_tokens,
            output_tokens=self.output_tokens,
            reasoning_tokens=self.reasoning_tokens,
            total_tokens=self.total_tokens,
            input_chars=len(prompt),
            output_chars=len(response),
            input_bytes=len(prompt.encode("utf-8")),
            output_bytes=len(response.encode("utf-8")),
        )

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class UsageCallRecord:
    call_id: int
    agent: AgentName
    session_id: str | None
    returncode: int | None
    usage: UsageMetadata
    validation_status: Literal["validated", "invalid"] = "invalid"
    raw_backend_usage: object | None = None
    role: Literal["repair"] | None = None
    model: str | None = None
    outcome: Literal[
        "succeeded", "nonzero_exit", "empty_output", "timeout", "spawn_error", "invalid_output"
    ] | None = None
    log_path: str | None = None
    fallback_planned: bool | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "call_id": self.call_id,
            "agent": self.agent,
            "session_id": self.session_id,
            "returncode": self.returncode,
            "validation_status": self.validation_status,
            "usage": self.usage.to_dict(),
        }
        if self.raw_backend_usage is not None:
            payload["raw_backend_usage"] = self.raw_backend_usage
        for key in ("role", "model", "outcome", "log_path", "fallback_planned"):
            value = getattr(self, key)
            if value is not None:
                payload[key] = value
        return payload


@dataclass(frozen=True)
class UsageTotals:
    call_count: int = 0
    success_count: int = 0
    exact_calls: int = 0
    partial_calls: int = 0
    estimated_calls: int = 0
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    input_chars: int | None = None
    output_chars: int | None = None
    input_bytes: int | None = None
    output_bytes: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def _add_optional(left: int | None, right: int | None) -> int | None:
    if right is None:
        return left
    if left is None:
        return right
    return left + right


def accumulate_usage_totals(records: list[UsageCallRecord]) -> UsageTotals:
    totals = UsageTotals()
    for record in records:
        usage = record.usage
        totals = UsageTotals(
            call_count=totals.call_count + 1,
            success_count=totals.success_count + int(record.validation_status == "validated"),
            exact_calls=totals.exact_calls + int(usage.mode == "exact"),
            partial_calls=totals.partial_calls + int(usage.mode == "partial"),
            estimated_calls=totals.estimated_calls + int(usage.mode == "estimated"),
            input_tokens=_add_optional(totals.input_tokens, usage.input_tokens),
            cached_input_tokens=_add_optional(
                totals.cached_input_tokens, usage.cached_input_tokens
            ),
            output_tokens=_add_optional(totals.output_tokens, usage.output_tokens),
            reasoning_tokens=_add_optional(totals.reasoning_tokens, usage.reasoning_tokens),
            total_tokens=_add_optional(totals.total_tokens, usage.total_tokens),
            input_chars=_add_optional(totals.input_chars, usage.input_chars),
            output_chars=_add_optional(totals.output_chars, usage.output_chars),
            input_bytes=_add_optional(totals.input_bytes, usage.input_bytes),
            output_bytes=_add_optional(totals.output_bytes, usage.output_bytes),
        )
    return totals


def estimate_usage(prompt: str, response: str) -> UsageMetadata:
    input_tokens = estimate_token_count(prompt)
    output_tokens = estimate_token_count(response)
    return UsageMetadata(
        mode="estimated",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    ).with_io_sizes(prompt=prompt, response=response)


@dataclass
class RunUsageContext:
    run_id: str
    summary_path: Path
    records: list[UsageCallRecord] = field(default_factory=list)
    _next_call_id: int = 1
    # Parallel discuss debaters add records from worker threads (#475).
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_record(
        self,
        *,
        agent: AgentName,
        session_id: str | None,
        returncode: int | None,
        usage: UsageMetadata,
        raw_backend_usage: object | None = None,
        role: Literal["repair"] | None = None,
        model: str | None = None,
        outcome: Literal[
            "succeeded", "nonzero_exit", "empty_output", "timeout", "spawn_error", "invalid_output"
        ] | None = None,
        log_path: str | None = None,
        fallback_planned: bool | None = None,
    ) -> UsageCallRecord:
        with self._lock:
            record = UsageCallRecord(
                call_id=self._next_call_id,
                agent=agent,
                session_id=session_id,
                returncode=returncode,
                usage=usage,
                raw_backend_usage=raw_backend_usage,
                role=role,
                model=model,
                outcome=outcome,
                log_path=log_path,
                fallback_planned=fallback_planned,
            )
            self._next_call_id += 1
            self.records.append(record)
        return record

    def totals(self) -> UsageTotals:
        return accumulate_usage_totals(self.records)

    def per_agent_totals(self) -> dict[str, UsageTotals]:
        agent_records: dict[str, list[UsageCallRecord]] = {}
        for record in self.records:
            agent_records.setdefault(record.agent, []).append(record)
        return {
            agent: accumulate_usage_totals(records)
            for agent, records in sorted(agent_records.items())
        }

    def summary_payload(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "generated_at": datetime.now(UTC).isoformat(),
            "summary_path": str(self.summary_path),
            "totals": self.totals().to_dict(),
            "per_agent": {
                agent: totals.to_dict() for agent, totals in self.per_agent_totals().items()
            },
            "calls": [record.to_dict() for record in self.records],
        }

    def write_summary(self) -> None:
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(self.summary_payload(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
