"""Typed reviewer-seat configuration and durable PR reviewer identity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .agents.base import normalize_agent_name
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
        from .agents.registry import agent_signature
        return f"{self.seat_id} ({agent_signature(self.backend, model_used=model_used or self.model_chain[0])})"


class SeatAgent(str):
    """A reviewer-role key that remains a string in historical round records.

    The string value is the durable seat ID. Backend routing is carried on the
    instance, so two seats using one CLI never share a review-board key.
    """

    def __new__(cls, seat: ReviewerSeat, workdir: Path):
        value = str.__new__(cls, seat.seat_id)
        value.backend = seat.backend
        value.model_chain = seat.model_chain
        value.effort = seat.effort
        value.workdir = workdir
        return value


def seat_backend(agent: str) -> str:
    return agent.backend if isinstance(agent, SeatAgent) else agent


def reviewer_seat_binding(config: object) -> dict[str, object] | None:
    """Versioned board identity stored on named PR round records."""
    override = getattr(config, "pr_seat_binding_override", None)
    if override is not None:
        return override
    if not getattr(config, "reviewer_seats", ()):
        return None
    from .agents.registry import agent_display_name
    from .config import resolve_invocation

    entries: list[dict[str, object]] = []
    for reviewer in config.reviewer:
        if isinstance(reviewer, SeatAgent):
            chain = list(reviewer.model_chain)
            effort = reviewer.effort
        else:
            invocation = resolve_invocation(config, provider=reviewer, role="reviewer")
            chain = (
                list(config.antigravity_models)
                if reviewer == "antigravity" else [invocation.configured_model]
            )
            effort = invocation.resolved_effort
        entries.append({
            "id": agent_display_name(reviewer),
            "backend": seat_backend(reviewer),
            "model_chain": chain,
            "effort": effort,
        })
    return {"version": 1, "seats": entries}


def validate_pr_seat_bindings(records: object, config: object) -> set[str]:
    """Check historical named identity before any PR recovery or amendment.

    A model change returns the seat IDs requiring fresh review. The caller
    retains the old finding ledger; only approvals and response checkpoints
    become ineligible for reuse.
    """
    current = reviewer_seat_binding(config)
    from .agents.registry import agent_display_name
    changed: set[str] = set()
    latest_binding_by_seat: dict[str, dict] = {}
    historical_backends: dict[str, str] = {}
    current_entries = {
        entry["id"]: entry for entry in current["seats"]
    } if current is not None else {}
    saw_bound = False
    for record in records:
        metadata = record.metadata
        if metadata.role != "reviewer":
            continue
        historical = metadata.seat_binding
        if historical is None:
            if current is not None:
                raise AgentLoopError(
                    "Named PR reviewer board cannot adopt an unbound historical reviewer record."
                )
            continue
        saw_bound = True
        if current is None:
            raise AgentLoopError("A bound named PR reviewer board cannot resume as a legacy board.")
        if not isinstance(historical, dict) or historical.get("version") != 1:
            raise AgentLoopError("PR reviewer seat binding has an unsupported version.")
        entries = historical.get("seats")
        if not isinstance(entries, list) or not entries:
            raise AgentLoopError("PR reviewer seat binding is incomplete.")
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(entry.get("backend"), str) or not isinstance(entry.get("model_chain"), list):
                raise AgentLoopError("PR reviewer seat binding has invalid seat fields.")
            if (
                not entry["model_chain"]
                or any(
                    (model is None and entry["id"] != agent_display_name(entry["backend"]))
                    or (model is not None and (not isinstance(model, str) or not model.strip()))
                    for model in entry["model_chain"]
                )
                or entry.get("effort") is not None and not isinstance(entry["effort"], str)
            ):
                raise AgentLoopError("PR reviewer seat binding has invalid model or effort fields.")
            seat_id = entry["id"]
            if seat_id in seen:
                raise AgentLoopError("PR reviewer seat binding contains duplicate identities.")
            seen.add(seat_id)
            prior_backend = historical_backends.setdefault(seat_id, entry["backend"])
            if prior_backend != entry["backend"]:
                raise AgentLoopError(f"Reviewer seat {seat_id!r} changed backend in PR history.")
            configured = current_entries.get(seat_id)
            if configured is None:
                continue  # Signed amendment lineage checks required-board changes.
            if configured["backend"] != entry["backend"]:
                raise AgentLoopError(f"Reviewer seat {seat_id!r} changed backend; refusing PR recovery.")
            if seat_id == metadata.agent:
                latest_binding_by_seat[seat_id] = entry
        if metadata.agent not in seen:
            raise AgentLoopError("PR reviewer record is not represented in its seat binding.")
    if current is not None and not saw_bound and any(record.metadata.role == "reviewer" for record in records):
        raise AgentLoopError("Named PR reviewer board has no verifiable historical binding.")
    for seat_id, entry in latest_binding_by_seat.items():
        configured = current_entries[seat_id]
        if configured["model_chain"] != entry["model_chain"] or configured["effort"] != entry.get("effort"):
            changed.add(seat_id)
    return changed


def validate_pr_backend_outage_amendments(amendments: object, config: object) -> None:
    """A named backend outage must remove all its active nonprimary seats."""
    binding = reviewer_seat_binding(config)
    if binding is None:
        return
    backends = {entry["id"]: entry["backend"] for entry in binding["seats"]}
    for amendment in amendments:
        removed = set(amendment.removed_reviewers)
        if not removed:
            continue
        unknown = removed - backends.keys()
        if unknown:
            raise AgentLoopError(
                "Named PR board amendment has no verified backend binding for "
                + ", ".join(sorted(unknown))
            )
        affected = {backends[name] for name in removed}
        required = set(amendment.original_required_reviewers)
        for backend in affected:
            active = {name for name in required if backends.get(name) == backend}
            if not active <= removed:
                raise AgentLoopError(
                    f"Named PR backend outage for {backend} must remove every active seat: "
                    + ", ".join(sorted(active))
                )


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
    from .agents.registry import agent_display_name, agent_signature
    from .comment_rendering import canonical_model_identity
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
        coder_backends = {getattr(args, "coder", "claude"), getattr(args, "implementation_coder", None)}
        for backend, flags in {
            "antigravity": ("antigravity_model", "antigravity_models"),
            "codex": ("codex_model", "codex_reasoning_effort"),
            "claude": ("claude_model", "claude_effort"),
            "gemini": ("gemini_model",),
        }.items():
            if backend not in legacy_backends and backend not in coder_backends:
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
                if not model and any(seat.backend == legacy for seat in seats):
                    raise AgentLoopError(
                        f"Legacy {legacy} reviewer has an implicit model; set an explicit legacy reviewer model "
                        "before adding a named seat on the same backend."
                    )
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
