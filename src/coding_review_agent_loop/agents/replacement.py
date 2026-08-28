"""Shared executable-replacement evidence gates for text-based backends."""

from __future__ import annotations

import json
import re

from ..protocol import PUBLIC_RESPONSE_MARKER
from ..runner import CommandResult, executable_identity_changed, strip_ansi
from ..workdir_guard import WorkdirReplayEvidence, WorkdirSnapshot, gate_workdir_replay


DIAGNOSTIC_MAX_LINES = 40
DIAGNOSTIC_MAX_CHARS = 8192


def _structured_payload(raw: str) -> bool:
    """Return true for a complete or line-delimited structured payload."""
    for candidate in (raw, *raw.splitlines()):
        if not candidate.strip():
            continue
        try:
            json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        return True
    return False


def _startup_or_loader_diagnostics(
    raw: str,
    *,
    startup_line_re: re.Pattern[str],
    loader_line_re: re.Pattern[str],
) -> bool:
    normalized = strip_ansi(raw).strip()
    if not normalized:
        return True
    if len(normalized) > DIAGNOSTIC_MAX_CHARS:
        return False
    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) > DIAGNOSTIC_MAX_LINES:
        return False
    return all(
        startup_line_re.fullmatch(line) is not None
        or loader_line_re.search(line) is not None
        for line in lines
    )


def classify_provider_executable_replacement_interruption(
    result: CommandResult,
    *,
    command: str,
    response_file_text: str | None,
    startup_line_re: re.Pattern[str],
    loader_line_re: re.Pattern[str],
    provider_label: str,
    reason: str,
    before_snapshot: WorkdirSnapshot | None = None,
    after_snapshot: WorkdirSnapshot | None = None,
) -> WorkdirReplayEvidence | None:
    """Classify a quiet provider launcher replacement with fail-closed gates."""
    observation = result.observation
    if (
        type(result.returncode) is not int
        or observation is None
        or observation.interrupted
        or result.capture_diagnostics
        or response_file_text
        or PUBLIC_RESPONSE_MARKER in result.stdout
        or _structured_payload(result.stdout)
        or not _startup_or_loader_diagnostics(
            result.stdout,
            startup_line_re=startup_line_re,
            loader_line_re=loader_line_re,
        )
    ):
        return None
    if not executable_identity_changed(
        observation.before,
        observation.after,
        command=command,
        spawn_wall_time=observation.spawn_wall_time,
        exit_wall_time=observation.spawn_wall_time + observation.elapsed_seconds,
    ):
        return None
    return gate_workdir_replay(
        provider_label,
        reason,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
    )
