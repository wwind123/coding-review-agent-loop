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


def agent_log_path(config: AgentLoopConfig, agent: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.log_dir / f"{stamp}-{agent}.log"
