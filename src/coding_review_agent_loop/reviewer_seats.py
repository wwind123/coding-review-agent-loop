"""Typed reviewer-seat configuration, ahead of review-loop persistence.

Named seats are parsed and validated at the CLI boundary. Execution remains
gated until the durable review board understands seat identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .agents.base import normalize_agent_name
from .agents.registry import agent_display_name, agent_signature
from .comment_rendering import canonical_model_identity
from .errors import AgentLoopError

_ID = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z", re.ASCII)
_BACKENDS = frozenset({"claude", "codex", "gemini", "antigravity"})
_RESERVED = frozenset({"agy", "antigravity", "claude", "codex", "gemini", "orchestrator"})
_CODEX_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh"})
_CLAUDE_EFFORTS = frozenset({"low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class ReviewerSeat:
    seat_id: str
    backend: str
    model_chain: tuple[str, ...]
    effort: str | None = None
    workdir: Path | None = None

    def signature(self, model_used: str | None = None) -> str:
        """A future named-review renderer; legacy callers keep their existing bytes."""
        return f"{self.seat_id} ({agent_signature(self.backend, model_used=model_used or self.model_chain[0])})"


def _assignments(values: list[str] | None, option: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for raw in values or ():
        name, separator, value = raw.partition("=")
        if not separator or not name or not value.strip():
            raise AgentLoopError(f"{option} requires SEAT=VALUE with non-blank parts.")
        result.setdefault(name, []).append(value.strip())
    return result


def resolve_reviewer_seats(args: object) -> tuple[ReviewerSeat, ...]:
    """Validate the complete board without I/O or checkout creation."""
    declarations = getattr(args, "reviewer_seat", None) or ()
    options = {
        "--seat-model": _assignments(getattr(args, "seat_model", None), "--seat-model"),
        "--seat-effort": _assignments(getattr(args, "seat_effort", None), "--seat-effort"),
        "--seat-dir": _assignments(getattr(args, "seat_dir", None), "--seat-dir"),
    }
    seats: list[ReviewerSeat] = []
    seen: set[str] = set()
    for declaration in declarations:
        seat_id, separator, backend_name = declaration.partition("=")
        if not separator or not seat_id or not backend_name.strip():
            raise AgentLoopError("--reviewer-seat requires SEAT=BACKEND.")
        if seat_id.casefold() in _RESERVED or seat_id.casefold() in {
            name.casefold()
            for backend in _BACKENDS
            for name in (agent_display_name(backend), agent_signature(backend))
        }:
            raise AgentLoopError(f"Reviewer seat {seat_id!r} is a reserved reviewer identity.")
        if not _ID.fullmatch(seat_id):
            raise AgentLoopError(f"Reviewer seat {seat_id!r} needs a lowercase ASCII ID (letters, digits, hyphens).")
        if seat_id.casefold() in seen:
            raise AgentLoopError(f"Duplicate reviewer seat ID {seat_id!r}.")
        seen.add(seat_id.casefold())
        backend = normalize_agent_name(backend_name.strip())
        if backend not in _BACKENDS:
            raise AgentLoopError(f"Reviewer seat {seat_id!r} has unsupported backend {backend_name!r}.")
        chain = tuple(options["--seat-model"].get(seat_id, ()))
        if not chain:
            raise AgentLoopError(f"Reviewer seat {seat_id!r} requires --seat-model {seat_id}=MODEL.")
        if len(chain) > 1 and backend != "antigravity":
            raise AgentLoopError(f"Reviewer seat {seat_id!r}: model fallback is supported only by Antigravity.")
        effort_values = options["--seat-effort"].get(seat_id, ())
        dir_values = options["--seat-dir"].get(seat_id, ())
        if len(effort_values) > 1 or len(dir_values) > 1:
            raise AgentLoopError(f"Reviewer seat {seat_id!r} has repeated effort or directory options.")
        effort = effort_values[0] if effort_values else None
        if effort is not None:
            allowed = _CODEX_EFFORTS if backend == "codex" else _CLAUDE_EFFORTS if backend == "claude" else ()
            if effort not in allowed:
                raise AgentLoopError(f"Reviewer seat {seat_id!r}: effort {effort!r} is unsupported for {backend}.")
        directory = Path(dir_values[0]).resolve() if dir_values else None
        seats.append(ReviewerSeat(seat_id, backend, chain, effort, directory))
    for option, mapping in options.items():
        for name in mapping:
            if name not in seen:
                raise AgentLoopError(f"{option} names undeclared reviewer seat {name!r}.")
    # A backend may host several seats, but one canonical model must never
    # satisfy two roles. Include a separately declared legacy default seat.
    by_model: dict[tuple[str, str], str] = {}
    for seat in seats:
        for model in seat.model_chain:
            key = (seat.backend, canonical_model_identity(model))
            previous = by_model.setdefault(key, seat.seat_id)
            if previous != seat.seat_id or sum(canonical_model_identity(m) == key[1] for m in seat.model_chain) > 1:
                raise AgentLoopError(f"Reviewer seats {previous!r} and {seat.seat_id!r} overlap on model {model!r}.")
    if seats:
        from .config import DEFAULT_ANTIGRAVITY_MODELS

        legacy_reviewers = tuple(normalize_agent_name(name) for name in getattr(args, "reviewer", None) or ())
        if len(set(legacy_reviewers)) != len(legacy_reviewers):
            raise AgentLoopError("--reviewer cannot include the same agent more than once.")
        legacy_backends = set(legacy_reviewers)
        coder = getattr(args, "coder", "claude")
        for backend, flags in {
            "antigravity": ("antigravity_model", "antigravity_models"),
            "codex": ("codex_model",),
            "claude": ("claude_model",),
            "gemini": ("gemini_model",),
        }.items():
            if backend not in legacy_backends and backend != coder:
                for flag in flags:
                    if getattr(args, flag, None):
                        raise AgentLoopError(
                            f"--{flag.replace('_', '-')} configures a coder or legacy default reviewer, "
                            f"not named {backend} seats. Use --seat-model."
                        )
        for flag, backend in (("reviewer_codex_model", "codex"), ("reviewer_claude_model", "claude"),
                              ("reviewer_codex_reasoning_effort", "codex"), ("reviewer_claude_effort", "claude")):
            if getattr(args, flag, None) and backend not in legacy_backends:
                raise AgentLoopError(f"--{flag.replace('_', '-')} requires an explicit legacy --reviewer {backend} with named seats.")

        for backend in getattr(args, "reviewer", None) or ():
            legacy = normalize_agent_name(backend)
            if legacy == "antigravity":
                chain = getattr(args, "antigravity_models", None) or (
                    (getattr(args, "antigravity_model", None),) if getattr(args, "antigravity_model", None)
                    else DEFAULT_ANTIGRAVITY_MODELS
                )
            else:
                chain = (getattr(args, f"reviewer_{legacy}_model", None) or getattr(args, f"{legacy}_model", None),)
            for model in chain:
                if model and (legacy, canonical_model_identity(model)) in by_model:
                    raise AgentLoopError(f"Named reviewer model {model!r} overlaps the legacy {legacy} reviewer.")
        coder_dir = getattr(args, f"{getattr(args, 'coder', 'claude')}_dir", None)
        used_dirs: dict[Path, str] = {}
        if coder_dir is not None:
            used_dirs[Path(coder_dir).resolve()] = "coder"
        implementation_coder = getattr(args, "implementation_coder", None)
        if implementation_coder is not None:
            implementation_dir = getattr(args, f"{implementation_coder}_dir", None)
            if implementation_dir is not None:
                used_dirs.setdefault(Path(implementation_dir).resolve(), "implementation coder")
        for backend in legacy_backends:
            reviewer_dir = getattr(args, f"{backend}_dir", None)
            if reviewer_dir is not None:
                used_dirs.setdefault(Path(reviewer_dir).resolve(), f"legacy {backend} reviewer")
        for seat in seats:
            if seat.workdir is None:
                continue
            previous = used_dirs.setdefault(seat.workdir, seat.seat_id)
            if previous != seat.seat_id:
                raise AgentLoopError(f"Reviewer seat {seat.seat_id!r} workdir collides with {previous}.")
    return tuple(seats)
