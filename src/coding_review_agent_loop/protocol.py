"""Parsing for agent response markers."""

from __future__ import annotations

import re

from .errors import AgentLoopError

STATE_RE = re.compile(r"<!--\s*AGENT_STATE:\s*(approved|blocking)\s*-->", re.I)
PR_RE = re.compile(r"<!--\s*AGENT_PR:\s*(\d+)\s*-->", re.I)
GH_PR_URL_RE = re.compile(r"/pull/(\d+)(?:\b|$)")
CLARIFY_RE = re.compile(r"<!--\s*AGENT_CLARIFY\s*-->", re.I)


def parse_agent_state(text: str) -> str:
    matches = STATE_RE.findall(text)
    if not matches:
        raise AgentLoopError("Agent response did not include <!-- AGENT_STATE: approved|blocking -->")
    # Use the final marker as authoritative; responses may quote earlier review markers.
    return matches[-1].lower()


def parse_pr_number(text: str) -> int | None:
    marker = PR_RE.search(text)
    if marker:
        return int(marker.group(1))
    url = GH_PR_URL_RE.search(text)
    if url:
        return int(url.group(1))
    return None


def is_clarification_request(text: str) -> bool:
    return bool(CLARIFY_RE.search(text))
