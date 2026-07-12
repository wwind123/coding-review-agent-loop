"""Validation helpers for keeping coder-reported work inside the assigned checkout."""

from __future__ import annotations

import re
import shlex
from pathlib import Path
from typing import Iterable, Sequence

from .errors import AgentLoopError
from .protocol import DiscussEvidenceClaim


TEST_SECTION_RE = re.compile(r"(?im)^\s*tests(?:\s+run)?\s*:\s*(?P<body>.*)$")
WINDOWS_PATH_RE = re.compile(r"(?<![\w.-])[A-Za-z]:\\[^\s`'\"|;&)<>]+")


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _is_inside(path: Path, assigned_workdir: Path) -> bool:
    try:
        path.relative_to(assigned_workdir)
        return True
    except ValueError:
        return False


def _normalize_reported_path(raw_path: str) -> Path | None:
    cleaned = raw_path.strip().strip("`'\".,")
    if not cleaned:
        return None
    if cleaned.startswith("$HOME/"):
        cleaned = str(Path.home()) + cleaned[len("$HOME") :]
    if cleaned.startswith("~/") or cleaned.startswith("/"):
        return _canonical(Path(cleaned))
    if WINDOWS_PATH_RE.fullmatch(cleaned):
        return Path(cleaned)
    return None


def _reported_paths(command: str) -> Iterable[str]:
    yield from WINDOWS_PATH_RE.findall(command)

    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    for index, token in enumerate(tokens):
        if token in {"cd", "-C", "--directory"} and index + 1 < len(tokens):
            yield tokens[index + 1]
        elif token.startswith("--directory="):
            yield token.split("=", 1)[1]
        elif token.startswith("$HOME/") or token.startswith("~/") or token.startswith("/"):
            yield token


def validate_test_commands_within_workdir(
    tests_run: Sequence[str] | None,
    *,
    assigned_workdir: Path,
) -> None:
    if not tests_run:
        return
    assigned = _canonical(assigned_workdir)
    for command in tests_run:
        for raw_path in _reported_paths(command):
            path = _normalize_reported_path(raw_path)
            if path is None:
                continue
            if WINDOWS_PATH_RE.fullmatch(raw_path.strip().strip("`'\".,")):
                raise AgentLoopError(
                    "Coder reported tests from a Windows-style path that cannot be "
                    "validated against the assigned Unix checkout: "
                    f"{raw_path!r} in command {command!r}. Assigned checkout: {assigned}"
                )
            if path == assigned or _is_inside(path, assigned):
                continue
            raise AgentLoopError(
                "Coder reported tests from outside the assigned checkout: "
                f"{raw_path!r} in command {command!r}. Assigned checkout: {assigned}"
            )


def extract_reported_tests_from_response(text: str) -> tuple[str, ...]:
    """Return the coder's public test-report lines, avoiding quoted issue context."""
    lines = text.splitlines()
    reports: list[str] = []
    for index, line in enumerate(lines):
        match = TEST_SECTION_RE.match(line)
        if not match:
            continue
        body = match.group("body").strip()
        if body:
            reports.append(body)
            continue
        continuation: list[str] = []
        for next_line in lines[index + 1 :]:
            stripped = next_line.strip()
            if not stripped:
                if continuation:
                    break
                continue
            if stripped.startswith("<!-- AGENT_") or stripped.startswith("-- "):
                break
            if re.match(r"^#{1,6}\s+", stripped):
                break
            continuation.append(stripped.removeprefix("- ").strip())
        if continuation:
            reports.extend(continuation)
    return tuple(reports)


def validate_response_tests_within_workdir(text: str, *, assigned_workdir: Path) -> None:
    validate_test_commands_within_workdir(
        extract_reported_tests_from_response(text),
        assigned_workdir=assigned_workdir,
    )


def validate_checkout_inspected_evidence(
    claims: Sequence[DiscussEvidenceClaim],
    *,
    assigned_workdir: Path,
) -> None:
    """Reject a ``checkout-inspected`` evidence claim whose ``path:line``
    source does not resolve to a real, in-range line inside the reviewer's
    assigned checkout right now.

    This is structural containment/existence checking only -- it does not
    verify the claimed fact is actually supported by that line's content.
    Claims with any other (or no) verification_basis are left untouched.
    """
    assigned = _canonical(assigned_workdir)
    for claim in claims:
        if claim.verification_basis != "checkout-inspected":
            continue
        # The parser (protocol.py) already guarantees `source` fullmatches
        # `[^\s:][^:]*:\d+`, i.e. a single colon separating a path with no
        # embedded colons from a trailing line number.
        source = claim.source or ""
        path_part, _, line_part = source.rpartition(":")
        if path_part.startswith("/") or path_part.startswith("~"):
            raise AgentLoopError(
                "checkout-inspected evidence claim used an absolute path outside "
                f"the assigned checkout: {source!r} for fact {claim.fact!r}."
            )
        if any(segment == ".." for segment in Path(path_part).parts):
            raise AgentLoopError(
                "checkout-inspected evidence claim used a path traversal segment: "
                f"{source!r} for fact {claim.fact!r}."
            )
        # No real file has anywhere near this many lines; reject before ever
        # calling int() on it. Python 3.11+ raises ValueError (not
        # AgentLoopError) for int()/str() conversions beyond its default
        # 4300-digit limit, and an unbounded `\d+` source is otherwise free
        # to carry an arbitrarily long digit string.
        if len(line_part) > 15:
            raise AgentLoopError(
                "checkout-inspected evidence claim references an implausibly "
                f"large line number: {source!r} for fact {claim.fact!r}."
            )
        resolved = _canonical(assigned / path_part)
        if not _is_inside(resolved, assigned):
            raise AgentLoopError(
                "checkout-inspected evidence claim resolves outside the assigned "
                f"checkout: {source!r} for fact {claim.fact!r}."
            )
        if not resolved.is_file():
            raise AgentLoopError(
                "checkout-inspected evidence claim references a path that is not "
                f"a file in the assigned checkout: {source!r} for fact {claim.fact!r}."
            )
        line_number = int(line_part)
        line_count = 0
        try:
            with resolved.open("rb") as handle:
                for line_count, _ in enumerate(handle, start=1):
                    pass
        except OSError as exc:
            # The already-validated path can still fail to open (deleted or
            # made unreadable between is_file() and open()); translate this
            # the same way as any other unresolvable reference instead of
            # letting an OSError escape past the AgentLoopError/repair
            # contract every other call site of this function relies on.
            raise AgentLoopError(
                "checkout-inspected evidence claim references a path that could "
                f"not be read: {source!r} for fact {claim.fact!r} ({exc})."
            ) from exc
        if not (1 <= line_number <= line_count):
            raise AgentLoopError(
                "checkout-inspected evidence claim references a line number "
                f"outside the file's range: {source!r} for fact {claim.fact!r} "
                f"(file has {line_count} lines)."
            )


def validate_assigned_head_advanced(
    *,
    before_head: str | None,
    after_head: str | None,
    assigned_workdir: Path,
) -> None:
    if not before_head or not after_head:
        return
    if before_head == after_head:
        raise AgentLoopError(
            "Coder reported a PR, but the assigned checkout HEAD did not advance. "
            f"Assigned checkout: {_canonical(assigned_workdir)}; HEAD: {after_head}"
        )
