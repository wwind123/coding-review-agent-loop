"""Agent backend protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from ..runner import Runner

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

AgentName = Literal["claude", "codex", "gemini"]


@dataclass(frozen=True)
class AgentResult:
    text: str
    session_id: str | None = None


class AgentBackend(Protocol):
    name: AgentName
    display_name: str
    signature: str

    def workdir(self, config: AgentLoopConfig) -> Path: ...

    def default_args(self, *, dangerous: bool) -> tuple[str, ...]: ...

    def run(
        self,
        runner: Runner,
        config: AgentLoopConfig,
        prompt: str,
        session_id: str | None = None,
    ) -> AgentResult: ...
