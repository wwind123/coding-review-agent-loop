"""Disposable reservation telemetry for the host-wide test-worker pool (#1107).

Instrumentation only: every test command that obtains its command lane appends
one JSON line to a host-scoped file, and each owning run loop appends a
run-start and run-end line.  Appends are best-effort (single ``O_APPEND``
write, no lock, no rotation, no retention) and never raise into the caller.
The file is a measurement artifact; the operator deletes it when done.
"""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

SCHEMA = 1
LOG_ENV = "AGENT_LOOP_WORKER_TELEMETRY_LOG"
ATTRIBUTION_ENV = {
    "repo": "AGENT_LOOP_RUN_REPO",
    "run_id": "AGENT_LOOP_RUN_ID",
    "issue_number": "AGENT_LOOP_RUN_ISSUE",
    "pr_number": "AGENT_LOOP_RUN_PR",
}
_OFF = {"off", "0", "false", "no"}


def telemetry_log_path(environ: Mapping[str, str] | None = None) -> Path | None:
    """The log path, or None when telemetry is disabled or misconfigured."""
    values = os.environ if environ is None else environ
    configured = values.get(LOG_ENV)
    if configured is None or not configured.strip():
        from .config import default_cache_root

        return default_cache_root() / "telemetry" / "worker-reservations.jsonl"
    configured = configured.strip()
    if configured.lower() in _OFF:
        return None
    path = Path(configured)
    return path if path.is_absolute() else None


def append_record(path: Path | None, record: Mapping[str, object]) -> None:
    """Append one line; any failure is swallowed."""
    if path is None:
        return
    try:
        line = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
    except Exception:
        pass


def _int_or_none(value: object) -> int | None:
    try:
        return int(value) if value not in (None, "") else None  # type: ignore[call-overload]
    except (TypeError, ValueError):
        return None


def attribution_from_environment(environ: Mapping[str, str] | None = None) -> dict:
    """Self-reported attribution for the standalone ``run-tests`` lane."""
    values = os.environ if environ is None else environ
    found = {key: values.get(name) or None for key, name in ATTRIBUTION_ENV.items()}
    return {
        "repo": found["repo"],
        "run_id": found["run_id"],
        "issue_number": _int_or_none(found["issue_number"]),
        "pr_number": _int_or_none(found["pr_number"]),
        "attribution_source": "environment" if any(found.values()) else "none",
        "lane": "standalone",
    }


def attribution_environment(attribution: Mapping[str, object] | None) -> dict[str, str]:
    """Launch-environment exports for a runner attribution."""
    if not attribution:
        return {}
    return {
        name: str(attribution[key])
        for key, name in ATTRIBUTION_ENV.items()
        if attribution.get(key) is not None
    }


def run_record(kind: str, attribution: Mapping[str, object], *, at: float | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "record": kind,
        "run_id": attribution.get("run_id"),
        "repo": attribution.get("repo"),
        "issue_number": attribution.get("issue_number"),
        "pr_number": attribution.get("pr_number"),
        "at": time.time() if at is None else at,
    }


class ReservationTelemetry:
    """Mutable recorder for one reservation attempt; setters only store values."""

    def __init__(self, path: Path | None, attribution: Mapping[str, object] | None = None):
        self.path = path
        self.attribution = dict(attribution or {})
        self.begun = False
        self.data: dict[str, object] = {}
        self.reservation_path: Path | None = None
        self._target_started_monotonic: float | None = None

    @classmethod
    def disabled(cls) -> "ReservationTelemetry":
        return cls(None)

    def begin(self, workers_requested: int | None) -> None:
        self.begun = True
        self.data = {
            "requested_at": time.time(),
            "reserved_at": None,
            "released_at": None,
            "retained": False,
            "workers_requested": workers_requested,
            "workers_granted": None,
            "capacity": None,
            "others_count": None,
            "others_workers": None,
            "outcome": "unshared",
            "target_started_at": None,
            "command_seconds": None,
            "test_outcome": None,
        }

    def set_outcome(self, outcome: str) -> None:
        self.data["outcome"] = outcome

    def decision(self, *, capacity: str, others: tuple[int, int] | None) -> None:
        self.data["capacity"] = capacity
        if others is not None:
            self.data["others_count"], self.data["others_workers"] = others

    def reserved(self, granted: int, reservation_path: Path | None) -> None:
        self.data["workers_granted"] = granted
        if granted:
            self.data["reserved_at"] = time.time()
            self.reservation_path = reservation_path

    def target_started(self) -> None:
        self.data["target_started_at"] = time.time()
        self._target_started_monotonic = time.monotonic()

    def target_finished(self) -> None:
        if self._target_started_monotonic is not None:
            self.data["command_seconds"] = time.monotonic() - self._target_started_monotonic

    def finish(self, test_outcome: str | None) -> None:
        self.data["test_outcome"] = test_outcome

    def emit(self) -> None:
        if not self.begun or self.path is None:
            return
        try:
            self.data["released_at"] = time.time()
            path = self.reservation_path
            self.data["retained"] = bool(path is not None and path.exists())
            append_record(
                self.path,
                {
                    "schema": SCHEMA,
                    "record": "attempt",
                    "repo": self.attribution.get("repo"),
                    "run_id": self.attribution.get("run_id"),
                    "issue_number": self.attribution.get("issue_number"),
                    "pr_number": self.attribution.get("pr_number"),
                    "attribution_source": self.attribution.get("attribution_source", "none"),
                    "lane": self.attribution.get("lane"),
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    **self.data,
                },
            )
        except Exception:
            pass


def load_records(path: Path) -> Iterator[dict]:
    """Parsed records; blank, malformed and non-object lines are skipped."""
    try:
        handle = open(path, "r", encoding="utf-8", errors="replace")
    except OSError:
        return
    with handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record


def _clip(interval: tuple[float, float], window: tuple[float, float]) -> tuple[float, float] | None:
    start, end = max(interval[0], window[0]), min(interval[1], window[1])
    return (start, end) if end > start else None


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def _union_length(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def run_intervals(records: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    """Per run: clipped ``low`` and ``high`` interval unions plus counters."""
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    attempts: dict[str, list[Mapping[str, object]]] = {}
    last_seen: dict[str, float] = {}
    for record in records:
        run_id = record.get("run_id")
        if not isinstance(run_id, str):
            continue
        kind = record.get("record")
        if kind == "run-start" and isinstance(record.get("at"), (int, float)):
            starts[run_id] = float(record["at"])  # type: ignore[arg-type]
        elif kind == "run-end" and isinstance(record.get("at"), (int, float)):
            ends[run_id] = float(record["at"])  # type: ignore[arg-type]
        elif kind == "attempt":
            attempts.setdefault(run_id, []).append(record)
        for key in ("at", "released_at"):
            value = record.get(key)
            if isinstance(value, (int, float)):
                last_seen[run_id] = max(last_seen.get(run_id, float(value)), float(value))
    result: dict[str, dict] = {}
    for run_id in sorted(set(starts) | set(attempts)):
        rows = attempts.get(run_id, [])
        incomplete: list[str] = []
        if run_id not in starts:
            result[run_id] = {
                "window": None, "low": [], "high": [], "attempts": len(rows),
                "degraded": sum(1 for r in rows if r.get("outcome") == "degraded"),
                "refused": sum(1 for r in rows if r.get("outcome") == "refused"),
                "retained": sum(1 for r in rows if r.get("retained")),
                "incomplete": ["no run-start"],
            }
            continue
        start = starts[run_id]
        if run_id in ends:
            end = ends[run_id]
        else:
            end = last_seen.get(run_id, start)
            incomplete.append("no run-end")
        window = (start, max(start, end))
        low: list[tuple[float, float]] = []
        high: list[tuple[float, float]] = []
        for row in rows:
            reserved = row.get("reserved_at")
            released = row.get("released_at")
            if row.get("outcome") == "unshared":
                began, seconds = row.get("target_started_at"), row.get("command_seconds")
                if isinstance(began, (int, float)) and isinstance(seconds, (int, float)):
                    interval = (float(began), float(began) + float(seconds))
                    for bucket in (low, high):
                        clipped = _clip(interval, window)
                        if clipped:
                            bucket.append(clipped)
                continue
            if not isinstance(reserved, (int, float)) or not isinstance(released, (int, float)):
                continue
            clipped = _clip((float(reserved), float(released)), window)
            if clipped:
                low.append(clipped)
            high_end = window[1] if row.get("retained") else float(released)
            clipped = _clip((float(reserved), high_end), window)
            if clipped:
                high.append(clipped)
        result[run_id] = {
            "window": window,
            "low": merge_intervals(low),
            "high": merge_intervals(high),
            "attempts": len(rows),
            "degraded": sum(1 for r in rows if r.get("outcome") == "degraded"),
            "refused": sum(1 for r in rows if r.get("outcome") == "refused"),
            "retained": sum(1 for r in rows if r.get("retained")),
            "incomplete": incomplete,
        }
    return result


def run_duty_cycles(records: Iterable[Mapping[str, object]]) -> dict[str, dict]:
    """Per-run ``{window_seconds, duty_low, duty_high, attempts, ...}``.

    Every holding interval is clipped to the run window before the union, so
    duty cycle stays within [0, 1].  A run without ``run-start`` has no window
    and is reported as incomplete rather than invented.
    """
    summary: dict[str, dict] = {}
    for run_id, info in run_intervals(records).items():
        window = info["window"]
        seconds = (window[1] - window[0]) if window else None
        summary[run_id] = {
            "window_seconds": seconds,
            "duty_low": (_union_length(info["low"]) / seconds) if seconds else None,
            "duty_high": (_union_length(info["high"]) / seconds) if seconds else None,
            "attempts": info["attempts"],
            "degraded": info["degraded"],
            "refused": info["refused"],
            "retained": info["retained"],
            "incomplete": info["incomplete"],
        }
    return summary


def overlap_seconds(a: Iterable[tuple[float, float]], b: Iterable[tuple[float, float]]) -> float:
    """Total time two runs' interval unions overlap."""
    total = 0.0
    for a_start, a_end in merge_intervals(a):
        for b_start, b_end in merge_intervals(b):
            total += max(0.0, min(a_end, b_end) - max(a_start, b_start))
    return total
