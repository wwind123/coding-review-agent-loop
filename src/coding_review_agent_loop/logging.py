"""Console logging helpers."""

from __future__ import annotations

import sys
from datetime import datetime


def log(config, message: str) -> None:
    if config.quiet:
        return
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[agent-loop {now}] {message}", file=sys.stderr, flush=True)


def agent_log_path(config, agent: str):
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.log_dir / f"{stamp}-{agent}.log"
