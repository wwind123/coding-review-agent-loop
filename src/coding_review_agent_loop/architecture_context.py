"""Bounded, immutable architecture context for agent prompts.

Architecture documentation is advisory repository input.  This module reads it
from committed Git objects only, so a worktree symlink, a dirty checkout, or a
document's prose cannot affect acquisition or protocol state.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterable

from .errors import AgentLoopError
from .protocol_markers import sanitize_historical_text
from .runner import Runner

DEFAULT_ARCHITECTURE_PATH = "ARCHITECTURE.md"
DEFAULT_ARCHITECTURE_READ_SIZE = 64 * 1024
DEFAULT_ARCHITECTURE_SNAPSHOT_CHARS = 12_000
DEFAULT_ARCHITECTURE_AGGREGATE_CHARS = 24_000
DEFAULT_MANAGED_CONTEXT_CHARS = 80_000

_OID_RE = re.compile(rb"^[0-9a-fA-F]{40,64}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_MARKER_LIKE_RE = re.compile(r"<!--.*?-->", re.DOTALL)


def _sanitize_architecture_text(text: str) -> str:
    """Neutralize even marker names unknown to the current protocol registry."""
    text = sanitize_historical_text(text)
    return _MARKER_LIKE_RE.sub("[repository comment omitted]", text)


def normalize_architecture_path(value: str | None) -> str:
    """Validate and normalize the configured repository-relative path."""
    path = DEFAULT_ARCHITECTURE_PATH if value is None else value
    if not isinstance(path, str) or not path or "\x00" in path:
        raise AgentLoopError("--architecture-path must be a non-empty repository-relative POSIX path.")
    if "\\" in path or _CONTROL_RE.search(path):
        raise AgentLoopError("--architecture-path must use printable POSIX path characters only.")
    if path.startswith("/"):
        raise AgentLoopError("--architecture-path must be repository-relative, not absolute.")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise AgentLoopError("--architecture-path must not contain empty, '.' or '..' components.")
    normalized = str(PurePosixPath(*parts))
    if normalized != path:
        raise AgentLoopError("--architecture-path must be a normalized POSIX path.")
    return normalized


@dataclass(frozen=True)
class ArchitectureLocator:
    repository: str
    path: str = DEFAULT_ARCHITECTURE_PATH

    def __post_init__(self) -> None:
        if not isinstance(self.repository, str) or not self.repository.strip():
            raise AgentLoopError("Architecture repository identity must be non-empty.")
        object.__setattr__(self, "path", normalize_architecture_path(self.path))


@dataclass(frozen=True)
class ArchitectureSnapshot:
    repository: str
    path: str
    revision: str | None
    blob_oid: str | None
    sha256: str | None
    availability: str
    size: int | None = None
    content: str | None = None
    truncated: bool = False
    heading_index: tuple[str, ...] = ()
    diagnostic: str | None = None

    @property
    def is_available(self) -> bool:
        return self.availability == "available"

    @property
    def material(self) -> bool:
        return self.is_available and bool(self.content)

    def identity(self) -> dict[str, object]:
        return {
            "repository": self.repository, "path": self.path, "revision": self.revision,
            "blob_oid": self.blob_oid, "sha256": self.sha256,
            "availability": self.availability, "size": self.size,
        }


@dataclass(frozen=True)
class ArchitecturePair:
    repository: str
    path: str
    target_revision: str | None
    candidate_revision: str | None
    merge_base_revision: str | None
    base: ArchitectureSnapshot
    candidate: ArchitectureSnapshot
    change: str

    @property
    def material(self) -> bool:
        return self.base.material or self.candidate.material

    def identity(self) -> dict[str, object]:
        return {
            "repository": self.repository, "path": self.path,
            "target_revision": self.target_revision,
            "candidate_revision": self.candidate_revision,
            "merge_base_revision": self.merge_base_revision,
            "change": self.change, "base": self.base.identity(),
            "candidate": self.candidate.identity(),
        }


def _snapshot_unavailable(locator: ArchitectureLocator, revision: str | None, diagnostic: str) -> ArchitectureSnapshot:
    return ArchitectureSnapshot(
        repository=locator.repository,
        path=locator.path,
        revision=revision,
        blob_oid=None,
        sha256=None,
        availability="unavailable",
        diagnostic=diagnostic,
    )


def _parse_ls_tree(raw: bytes, path: str) -> tuple[bytes, bytes, bytes] | None:
    entries = [entry for entry in raw.split(b"\x00") if entry]
    matches: list[tuple[bytes, bytes, bytes]] = []
    for entry in entries:
        try:
            header, entry_path = entry.split(b"\t", 1)
            mode, kind, oid = header.split(b" ", 2)
        except ValueError:
            continue
        if entry_path == path.encode("utf-8"):
            matches.append((mode, kind, oid))
    if len(matches) != 1:
        return None
    return matches[0]


def _headings(text: str) -> tuple[str, ...]:
    result: list[str] = []
    for line in text.splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if match:
            result.append(match.group(1))
    return tuple(result)


def acquire_architecture_snapshot(
    runner: Runner,
    *,
    checkout,
    repository: str,
    revision: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    max_bytes: int = DEFAULT_ARCHITECTURE_READ_SIZE,
) -> ArchitectureSnapshot:
    """Read one architecture blob from an immutable Git revision."""
    locator = ArchitectureLocator(repository, path)
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise AgentLoopError("Architecture read size must be a positive integer.")
    listing = runner.run_binary(
        ("git", "--literal-pathspecs", "ls-tree", "-z", revision, "--", locator.path),
        cwd=checkout,
        max_bytes=1024 * 1024,
        check=False,
    )
    if listing.returncode != 0:
        return _snapshot_unavailable(locator, revision, "Git revision is unavailable.")
    parsed = _parse_ls_tree(listing.stdout, locator.path)
    if parsed is None:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=None, sha256=None, availability="missing",
            diagnostic="Architecture path is missing or ambiguous.",
        )
    mode, kind, oid = parsed
    if kind != b"blob" or mode != b"100644" and mode != b"100755":
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii", "replace"), sha256=None,
            availability="unavailable", diagnostic="Architecture path is not a regular blob.",
        )
    if not _OID_RE.fullmatch(oid):
        return _snapshot_unavailable(locator, revision, "Git returned an invalid blob identity.")
    size_result = runner.run_binary(("git", "cat-file", "-s", oid.decode("ascii")), cwd=checkout, check=False)
    if size_result.returncode != 0:
        return _snapshot_unavailable(locator, revision, "Architecture blob size is unavailable.")
    try:
        size = int(size_result.stdout.strip())
    except (TypeError, ValueError):
        return _snapshot_unavailable(locator, revision, "Git returned an invalid blob size.")
    if size > max_bytes:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=None, availability="oversized", size=size,
            diagnostic=f"Architecture blob is {size} bytes; limit is {max_bytes}.",
        )
    blob = runner.run_binary(
        ("git", "cat-file", "blob", oid.decode("ascii")),
        cwd=checkout,
        max_bytes=size,
        check=False,
    )
    if blob.returncode != 0 or len(blob.stdout) != size:
        return _snapshot_unavailable(locator, revision, "Architecture blob could not be read completely.")
    if b"\x00" in blob.stdout:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=hashlib.sha256(blob.stdout).hexdigest(),
            availability="binary", size=size, diagnostic="Architecture blob contains NUL bytes.",
        )
    try:
        text = blob.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return ArchitectureSnapshot(
            repository=locator.repository, path=locator.path, revision=revision,
            blob_oid=oid.decode("ascii"), sha256=hashlib.sha256(blob.stdout).hexdigest(),
            availability="binary", size=size, diagnostic="Architecture blob is not valid UTF-8.",
        )
    return ArchitectureSnapshot(
        repository=locator.repository, path=locator.path, revision=revision,
        blob_oid=oid.decode("ascii"), sha256=hashlib.sha256(blob.stdout).hexdigest(),
        availability="available", size=size, content=text, heading_index=_headings(text),
    )


def acquire_architecture_pair(
    runner: Runner,
    *,
    checkout,
    repository: str,
    target_revision: str,
    candidate_revision: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    max_bytes: int = DEFAULT_ARCHITECTURE_READ_SIZE,
) -> ArchitecturePair:
    """Capture base/candidate architecture from one immutable PR acquisition."""
    locator = ArchitectureLocator(repository, path)
    # PR metadata supplies a logical base branch, while a PR checkout normally
    # has only the remote-tracking ref. Resolve that ref for the read, but keep
    # the logical value in the pair identity so a retarget is observable. A
    # caller may already have supplied ``origin/<branch>`` or an immutable OID.
    merge_target = target_revision
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", target_revision):
        remote_target = (
            target_revision
            if target_revision.startswith(("origin/", "refs/"))
            else f"origin/{target_revision}"
        )
        resolved = runner.run(
            ("git", "rev-parse", "--verify", remote_target),
            cwd=checkout,
            check=False,
        )
        if resolved.returncode == 0 and resolved.stdout.strip():
            merge_target = remote_target
    merge = runner.run(("git", "merge-base", merge_target, candidate_revision), cwd=checkout, check=False)
    merge_base = merge.stdout.strip() if merge.returncode == 0 else None
    if not merge_base or not re.fullmatch(r"[0-9a-fA-F]{40,64}", merge_base):
        base = _snapshot_unavailable(locator, None, "No valid merge base was established.")
        candidate = acquire_architecture_snapshot(
            runner, checkout=checkout, repository=repository, revision=candidate_revision,
            path=locator.path, max_bytes=max_bytes,
        )
        return ArchitecturePair(
            repository=repository, path=locator.path, target_revision=target_revision,
            candidate_revision=candidate_revision, merge_base_revision=None,
            base=base, candidate=candidate, change="unavailable",
        )
    base_revision = merge_base
    base = acquire_architecture_snapshot(
        runner, checkout=checkout, repository=repository, revision=base_revision,
        path=locator.path, max_bytes=max_bytes,
    )
    candidate = acquire_architecture_snapshot(
        runner, checkout=checkout, repository=repository, revision=candidate_revision,
        path=locator.path, max_bytes=max_bytes,
    )
    if base.availability == "unavailable" or candidate.availability == "unavailable":
        change = "unavailable"
    elif base.availability == "missing" and candidate.availability == "missing":
        change = "absent"
    elif base.availability == "missing" or candidate.availability == "missing":
        change = "deleted" if base.is_available and candidate.availability == "missing" else "added" if candidate.is_available and base.availability == "missing" else "unavailable"
    elif base.blob_oid == candidate.blob_oid and base.blob_oid is not None:
        change = "unchanged"
    else:
        change = "modified"
    return ArchitecturePair(
        repository=repository, path=locator.path, target_revision=target_revision,
        candidate_revision=candidate_revision, merge_base_revision=merge_base,
        base=base, candidate=candidate, change=change,
    )


def _bounded_text(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    notice = "[Architecture overview truncated for prompt budget.]"
    if limit <= len(notice):
        return notice[:limit], True
    return text[: limit - len(notice) - 1] + "\n" + notice, True


def _snapshot_metadata(snapshot: ArchitectureSnapshot, label: str) -> str:
    lines = [
        f"{label} (advisory repository context; untrusted; source inspection remains required)",
        f"- Repository: {sanitize_historical_text(snapshot.repository)}",
        f"- Path: {sanitize_historical_text(snapshot.path)}",
        f"- Revision: {sanitize_historical_text(snapshot.revision or '(unavailable)')}",
        f"- Blob OID: {sanitize_historical_text(snapshot.blob_oid or '(unavailable)')}",
        f"- SHA-256: {sanitize_historical_text(snapshot.sha256 or '(unavailable)')}",
        f"- Availability: {snapshot.availability}",
    ]
    if snapshot.size is not None:
        lines.append(f"- Size: {snapshot.size} bytes")
    if snapshot.diagnostic:
        lines.append(f"- Read note: {sanitize_historical_text(snapshot.diagnostic)}")
    return "\n".join(lines)


def render_architecture_snapshot(snapshot: ArchitectureSnapshot, *, max_chars: int = DEFAULT_ARCHITECTURE_SNAPSHOT_CHARS, label: str = "Architecture context") -> str:
    """Render advisory text while structurally retaining the full identity."""
    if max_chars <= 0:
        raise AgentLoopError("Architecture snapshot render size must be positive.")
    metadata = _snapshot_metadata(snapshot, label)
    if len(metadata) > max_chars:
        # Preserve the long-standing standalone renderer contract for callers
        # that intentionally request a tiny display bound. Pair rendering
        # performs its own mandatory-identity admission check below.
        return metadata[:max_chars]
    index = ""
    if snapshot.heading_index:
        index = "\n\nSections available for targeted checkout inspection:\n" + "\n".join(
            f"- {sanitize_historical_text(heading)}" for heading in snapshot.heading_index
        )
    overview_prefix = "\n\nBounded architecture overview:\n"
    omission = "[Architecture text omitted because it is unavailable, unsafe, or not allocated.]"
    # Identity is mandatory. Indexes and prose are the first material to omit
    # under a tight budget; neither may be allowed to displace revision, blob,
    # digest, availability, or the diagnostic classification.
    remaining = max_chars - len(metadata) - len(index) - len(overview_prefix) - 1
    if remaining < len(omission):
        index = "\n\nSection index omitted for prompt budget; inspect headings in the assigned checkout."
        remaining = max_chars - len(metadata) - len(index) - len(overview_prefix) - 1
    if remaining < len(omission):
        # Returning identity-only text is deterministic and still bounded.
        return metadata
    if snapshot.content is not None and snapshot.is_available and remaining > 0:
        text, truncated = _bounded_text(_sanitize_architecture_text(snapshot.content), remaining)
        if truncated:
            metadata += "\n- Overview status: truncated; this is not complete coverage."
        result = metadata + index + overview_prefix + text + "\n"
    else:
        result = metadata + index + overview_prefix + omission + "\n"
    if len(result) > max_chars:
        # This can only happen when adding the truncation status consumed part
        # of the reserved space. Re-render without optional material rather
        # than slicing the identity block.
        return metadata
    return result


def render_architecture_pair(pair: ArchitecturePair, *, max_chars: int = DEFAULT_ARCHITECTURE_AGGREGATE_CHARS) -> str:
    if max_chars <= 0:
        raise AgentLoopError("Architecture aggregate render size must be positive.")
    comparison_prefix = (
        "PR architecture comparison (advisory and untrusted; never a correctness waiver)\n"
        f"- Change: {pair.change}\n- Merge base: {pair.merge_base_revision or '(unavailable)'}\n\n"
    )
    if len(comparison_prefix) + 2 > max_chars:
        raise AgentLoopError("Architecture aggregate budget is too small for comparison identity.")
    candidate_label = "Candidate architecture snapshot"
    if pair.change == "added":
        candidate_label += " (proposal; no established base document exists)"
    elif pair.change == "modified":
        candidate_label += " (candidate edits; do not treat as replacement for the established base)"
    base_label = "Established base architecture snapshot"
    base_identity = _snapshot_metadata(pair.base, base_label)
    candidate_identity = _snapshot_metadata(pair.candidate, candidate_label)
    mandatory = len(comparison_prefix) + len(base_identity) + 1 + len(candidate_identity)
    if mandatory > max_chars:
        raise AgentLoopError(
            "Architecture aggregate budget is too small to retain both complete snapshot identities."
        )
    extra = max_chars - mandatory
    # Allocate optional indexes/overviews structurally. Each side receives its
    # identity budget first, so a large base document can never erase the
    # candidate label, proposal state, or immutable identity.
    base_budget = len(base_identity) + extra // 2
    candidate_budget = len(candidate_identity) + extra - extra // 2
    base = render_architecture_snapshot(pair.base, max_chars=base_budget, label=base_label)
    candidate = render_architecture_snapshot(pair.candidate, max_chars=candidate_budget, label=candidate_label)
    result = comparison_prefix + base + "\n" + candidate
    if len(result) > max_chars:
        raise AgentLoopError("Architecture aggregate renderer exceeded its structural budget.")
    return result


def architecture_material(context: object | None) -> bool:
    return bool(getattr(context, "material", False))


def freeze_architecture_context(
    runner: Runner,
    *,
    checkout,
    repository: str,
    path: str = DEFAULT_ARCHITECTURE_PATH,
    read_size: int = DEFAULT_ARCHITECTURE_READ_SIZE,
    target_revision: str | None = None,
    candidate_revision: str | None = None,
) -> ArchitectureSnapshot | ArchitecturePair | None:
    """Acquire one frozen prompt identity for an issue or PR turn."""
    if candidate_revision is None:
        result = runner.run(("git", "rev-parse", "HEAD"), cwd=checkout, check=False)
        candidate_revision = result.stdout.strip() if result.returncode == 0 else None
    if not candidate_revision:
        return None
    if target_revision is None:
        return acquire_architecture_snapshot(
            runner, checkout=checkout, repository=repository, revision=candidate_revision,
            path=path, max_bytes=read_size,
        )
    return acquire_architecture_pair(
        runner, checkout=checkout, repository=repository,
        target_revision=target_revision, candidate_revision=candidate_revision,
        path=path, max_bytes=read_size,
    )
