"""Shared exceptions for the agent loop."""

from __future__ import annotations


class AgentLoopError(RuntimeError):
    """Raised for expected orchestration failures."""
