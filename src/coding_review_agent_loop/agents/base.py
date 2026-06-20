"""Agent backend protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import uuid
from typing import TYPE_CHECKING, Literal, Protocol, cast

from ..runner import Runner
from ..usage import UsageMetadata

if TYPE_CHECKING:
    from ..config import AgentLoopConfig

AgentName = Literal["claude", "codex", "gemini", "antigravity"]
AgentTextSource = Literal["response_file", "stdout_marker", "stdout"]


def normalize_agent_name(value: str) -> AgentName:
    """Normalize input-only agent aliases to their canonical names."""
    return cast(AgentName, "antigravity" if value == "agy" else value)


@dataclass(frozen=True)
class AgentResult:
    text: str
    raw_output: str = ""
    text_source: AgentTextSource = "stdout"
    response_file_text: str | None = None
    message_text: str | None = None
    session_id: str | None = None
    log_path: Path | None = None
    returncode: int = 0
    usage: UsageMetadata | None = None
    raw_usage: object | None = None
    # Human-readable label of the model that actually produced this result (e.g.
    # "Gemini 3.1 Pro (High)", "claude-sonnet-4-6"). Ground truth for the dynamic
    # signature (#332); None when the backend could not determine it. The
    # antigravity fallback chain (#333) sets this to the model that answered.
    model_used: str | None = None


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
        run_id: str | None = None,
        role: str | None = None,
    ) -> AgentResult: ...


def _safe_repo_slug(repo: str) -> str:
    return repo.replace("/", "-").replace(":", "-")


def public_response_path(
    config: AgentLoopConfig,
    agent: AgentName,
    *,
    root: Path | None = None,
) -> Path:
    base_dir = (
        root
        if root is not None
        else Path(tempfile.gettempdir())
        / "coding-review-agent-loop"
        / "responses"
        / _safe_repo_slug(config.repo)
    )
    path = (
        base_dir
        / agent
        / f"{uuid.uuid4().hex}.md"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def with_public_response_file_instruction(prompt: str, response_path: Path) -> str:
    return f"""{prompt}

PUBLIC RESPONSE FILE:

Write the final public response that should be posted to GitHub to this file:

{response_path}

The orchestrator will post only that file's contents when it exists and is
non-empty. Keep internal tool narration, planning notes, diagnostics, and
scratch output out of that file. Include the required AGENT_STATE / AGENT_PR /
AGENT_CLARIFY markers in the file, as requested above.

For structured responses, the file must start directly with {{, contain exactly
one structured JSON object, then the required footer marker and signature. Do
not put the stdout filtering marker, headings, prose, or markdown code fences in
the response file. For structured plan revisions only, if signed human
requirements are present, place `<!-- HUMAN_REQUIREMENTS_ADDRESSED -->` and the
`### Human requirements` section after the JSON object and before the
`AGENT_PLAN_STATE` footer.
You MUST write this file before your turn ends. A turn that ends without
writing this file results in a review failure.
"""


def read_public_response_file(response_path: Path) -> str | None:
    try:
        text = response_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    text = text.strip()
    return text or None
