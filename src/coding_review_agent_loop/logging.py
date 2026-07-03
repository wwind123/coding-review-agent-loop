"""Console logging helpers."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import AgentLoopConfig


def log(config: AgentLoopConfig, message: str) -> None:
    if config.quiet:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[agent-loop {now}] {message}", file=sys.stderr, flush=True)


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


def agent_log_path(
    config: AgentLoopConfig,
    agent: str,
    *,
    run_id: str | None = None,
    label: str | None = None,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    prefix = f"{run_id}-" if run_id else ""
    suffix = f"-{label}" if label else ""
    return config.log_dir / f"{prefix}{stamp}-{agent}{suffix}.log"


def run_usage_summary_path(config: AgentLoopConfig, run_id: str) -> Path:
    return config.log_dir / f"{run_id}-usage-summary.json"
