"""Pure scheduling and diff classification for selective PR re-review.

The selective policy is intentionally conservative.  This module contains no
GitHub or agent calls so the decision can be unit tested and replayed from the
small amount of scheduler metadata persisted by :mod:`round_state`.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Iterable, Sequence

from .errors import AgentLoopError

MAX_FIX_SCOPE_ENTRIES = 32
MAX_FIX_SCOPE_PATH_BYTES = 240
MAX_FIX_SCOPE_BYTES = 4096

PR_REVIEW_POLICIES = frozenset({"all-reviewers", "selective-intermediate"})

# These are patterns, rather than prose categories, so their interpretation is
# stable across machines and can be bound into a persisted contract digest.
DEFAULT_BROAD_RULES: tuple[str, ...] = (
    ".github/**",
    "**/workflows/**",
    "**/actions/**",
    "Dockerfile*",
    "docker-compose*",
    "Makefile",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "tox.ini",
    "noxfile.py",
    "requirements*.txt",
    "Pipfile.lock",
    "poetry.lock",
    "uv.lock",
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "**/schema/**",
    "**/schemas/**",
    "**/migration/**",
    "**/migrations/**",
    "alembic.ini",
    "CODEOWNERS",
    ".gitignore",
    ".gitattributes",
    ".editorconfig",
)


def _clean_rule(rule: object) -> str:
    if not isinstance(rule, str):
        raise AgentLoopError("PR review broad rules must be strings.")
    value = rule.strip().replace("\\", "/")
    if (
        not value
        or value.startswith("/")
        or ".." in PurePosixPath(value).parts
        or any(part in {"", "."} for part in value.split("/"))
    ):
        raise AgentLoopError(f"Invalid PR review broad rule: {rule!r}.")
    if "\x00" in value:
        raise AgentLoopError("PR review broad rules must not contain NUL bytes.")
    return value


def normalize_broad_rules(rules: Sequence[str] | None) -> tuple[str, ...]:
    """Normalize, deduplicate, and deterministically order broad-path rules."""
    if isinstance(rules, (str, bytes)):
        raise AgentLoopError("PR review broad rules must be a sequence of patterns.")
    values = DEFAULT_BROAD_RULES if rules is None else tuple(rules)
    normalized = {_clean_rule(rule) for rule in values}
    if not normalized:
        raise AgentLoopError("PR review broad rules must contain at least one rule.")
    return tuple(sorted(normalized))


def broad_rules_digest(rules: Sequence[str] | None) -> str:
    normalized = normalize_broad_rules(rules)
    return hashlib.sha256(
        json.dumps(normalized, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:16]


def normalize_fix_scope(value: object) -> tuple[str, ...]:
    """Validate a reviewer-authored exact repository-relative POSIX path set."""
    if not isinstance(value, (list, tuple)):
        raise AgentLoopError("fix_scope must be an array of repository-relative paths.")
    if not value:
        raise AgentLoopError("fix_scope must not be empty.")
    if len(value) > MAX_FIX_SCOPE_ENTRIES:
        raise AgentLoopError(
            f"fix_scope may contain at most {MAX_FIX_SCOPE_ENTRIES} paths."
        )
    result: list[str] = []
    total_bytes = 0
    for raw in value:
        if not isinstance(raw, str):
            raise AgentLoopError("fix_scope entries must be strings.")
        if raw != raw.strip():
            raise AgentLoopError(
                f"fix_scope path {raw!r} must not contain leading or trailing whitespace."
            )
        path = raw
        if (
            not path
            or "\x00" in path
            or "\\" in path
            or path.startswith("/")
            or re.match(r"^[A-Za-z]:", path)
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or any(char in path for char in "*?[]{}")
        ):
            raise AgentLoopError(
                f"fix_scope path {raw!r} must be a bounded exact POSIX path without traversal or globs."
            )
        normalized = posixpath.normpath(path)
        if normalized != path or normalized == ".":
            raise AgentLoopError(f"fix_scope path {raw!r} is not normalized.")
        encoded_size = len(normalized.encode("utf-8"))
        if encoded_size > MAX_FIX_SCOPE_PATH_BYTES:
            raise AgentLoopError(
                f"fix_scope paths may be at most {MAX_FIX_SCOPE_PATH_BYTES} bytes."
            )
        total_bytes += encoded_size
        if normalized in result:
            raise AgentLoopError(f"fix_scope contains duplicate path {normalized!r}.")
        result.append(normalized)
    if total_bytes > MAX_FIX_SCOPE_BYTES:
        raise AgentLoopError(f"fix_scope may be at most {MAX_FIX_SCOPE_BYTES} bytes.")
    return tuple(result)


def normalize_optional_fix_scope(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    return normalize_fix_scope(value)


@dataclass(frozen=True)
class ReviewSchedulingContract:
    """Immutable in-flight scheduler contract."""

    required_reviewers: tuple[str, ...]
    policy: str = "all-reviewers"
    broad_rules: tuple[str, ...] = field(default_factory=lambda: DEFAULT_BROAD_RULES)
    broad_rules_digest: str | None = None

    def __post_init__(self) -> None:
        reviewers = tuple(self.required_reviewers)
        if (
            not reviewers
            or any(not isinstance(item, str) or not item for item in reviewers)
            or len(set(reviewers)) != len(reviewers)
        ):
            raise AgentLoopError("Scheduler contract requires a unique reviewer set.")
        if self.policy not in PR_REVIEW_POLICIES:
            raise AgentLoopError(f"Unsupported PR review policy: {self.policy!r}.")
        rules = normalize_broad_rules(self.broad_rules)
        digest = broad_rules_digest(rules)
        if self.broad_rules_digest is not None and self.broad_rules_digest != digest:
            raise AgentLoopError("Scheduler broad-rule digest does not match its rules.")
        object.__setattr__(self, "required_reviewers", reviewers)
        object.__setattr__(self, "broad_rules", rules)
        object.__setattr__(self, "broad_rules_digest", digest)

    def as_dict(self) -> dict[str, object]:
        return {
            "required_reviewers": list(self.required_reviewers),
            "policy": self.policy,
            "broad_rules": list(self.broad_rules),
            "broad_rules_digest": self.broad_rules_digest,
        }

    @classmethod
    def from_mapping(cls, value: object) -> "ReviewSchedulingContract":
        if not isinstance(value, dict):
            raise AgentLoopError("Malformed scheduler contract.")
        reviewers = value.get("required_reviewers")
        rules = value.get("broad_rules")
        policy = value.get("policy")
        if (
            not isinstance(reviewers, list)
            or not isinstance(rules, list)
            or any(not isinstance(item, str) or not item for item in reviewers)
            or not isinstance(policy, str)
        ):
            raise AgentLoopError("Incomplete scheduler contract.")
        return cls(
            required_reviewers=tuple(reviewers),
            policy=policy,
            broad_rules=tuple(rules),
            broad_rules_digest=(
                str(value["broad_rules_digest"])
                if value.get("broad_rules_digest") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class ReviewObligation:
    item_id: str
    status: str
    scope: tuple[str, ...] | None
    resolution_owners: tuple[str, ...]
    pending_owners: tuple[str, ...]

    @property
    def active(self) -> bool:
        return self.status in {"blocking", "same-pr"}


@dataclass(frozen=True)
class GitChange:
    path: str
    status: str = "M"
    binary: bool = False
    mode_changed: bool = False


@dataclass(frozen=True)
class TransitionClassification:
    kind: str
    reason: str
    changed_paths: tuple[str, ...] = ()

    @property
    def narrow(self) -> bool:
        return self.kind == "narrow"

    @property
    def broad(self) -> bool:
        return self.kind == "broad"


@dataclass(frozen=True)
class SchedulerSnapshot:
    previous_sha: str | None
    current_sha: str | None
    contract: ReviewSchedulingContract
    obligations: tuple[ReviewObligation, ...] = ()
    force_full: bool = False


@dataclass(frozen=True)
class SchedulingDecision:
    selected_reviewers: tuple[str, ...]
    paused_reviewers: tuple[tuple[str, str], ...]
    reason: str
    classification: TransitionClassification
    final_sweep: bool = False
    calls_avoided: int = 0


def _path_matches(path: str, pattern: str) -> bool:
    """Match slash-separated paths with a deterministic ``**`` algorithm."""
    path_parts = tuple(path.split("/"))
    pattern_parts = tuple(pattern.split("/"))

    def match(pi: int, xi: int) -> bool:
        if pi == len(pattern_parts):
            return xi == len(path_parts)
        part = pattern_parts[pi]
        if part == "**":
            return match(pi + 1, xi) or (xi < len(path_parts) and match(pi, xi + 1))
        return xi < len(path_parts) and fnmatch.fnmatchcase(path_parts[xi], part) and match(pi + 1, xi + 1)

    return match(0, 0)


def classify_transition(
    previous_sha: str | None,
    current_sha: str | None,
    changes: Iterable[GitChange],
    *,
    scopes: Sequence[str] | None,
    broad_rules: Sequence[str] | None = None,
    obligations: Sequence[ReviewObligation] | None = None,
    ancestor: bool = True,
    diff_complete: bool = True,
    history_available: bool = True,
) -> TransitionClassification:
    """Classify a candidate transition using only observed Git facts."""
    if not previous_sha or not current_sha or previous_sha == current_sha:
        return TransitionClassification("broad", "missing or unchanged review SHA")
    if not history_available or not ancestor:
        return TransitionClassification("broad", "candidate history is not an available ancestor")
    if not diff_complete:
        return TransitionClassification("broad", "complete exact Git diff was unavailable")
    try:
        normalized_scopes: set[str] = set()
        if scopes is not None:
            normalized_scopes.update(normalize_fix_scope(scopes))
        if obligations is not None:
            active_obligations = tuple(obligation for obligation in obligations if obligation.active)
            for obligation in active_obligations:
                owners = obligation.resolution_owners
                pending = obligation.pending_owners
                if (
                    obligation.scope is None
                    or not owners
                    or len(set(owners)) != len(owners)
                    or not pending
                    or any(owner not in owners for owner in pending)
                ):
                    return TransitionClassification(
                        "broad", "the active obligation ledger is not reconstructible"
                    )
                normalized_scopes.update(normalize_fix_scope(obligation.scope))
        elif not normalized_scopes:
            normalize_fix_scope(())
    except AgentLoopError:
        return TransitionClassification("broad", "an obligation has no valid exact fix scope")
    rules = normalize_broad_rules(broad_rules)
    observed = tuple(changes)
    paths = tuple(change.path for change in observed)
    if not observed:
        return TransitionClassification("broad", "transition contained no observable diff", paths)
    for change in observed:
        path = change.path.replace("\\", "/")
        try:
            normalize_fix_scope((path,))
        except AgentLoopError:
            return TransitionClassification("broad", "diff contained an invalid path", paths)
        if change.status.upper() not in {"A", "M"} or change.binary or change.mode_changed:
            return TransitionClassification(
                "broad", "diff contained a rename, deletion, binary, mode, or other non-text change", paths
            )
        if path not in normalized_scopes:
            return TransitionClassification("broad", f"diff path {path!r} is outside obligation scopes", paths)
        if any(_path_matches(path, rule) for rule in rules):
            return TransitionClassification("broad", f"diff path {path!r} matches a broad rule", paths)
    return TransitionClassification("narrow", "all text additions/modifications stay within exact obligation scopes", paths)


def select_reviewers(
    snapshot: SchedulerSnapshot,
    classification: TransitionClassification,
    *,
    qualifying_approvals: Sequence[str] = (),
    unavailable_reviewers: Sequence[str] = (),
    final_sweep: bool = False,
) -> SchedulingDecision:
    """Choose the board for one round; unavailable reviewers remain required."""
    required = snapshot.contract.required_reviewers
    unavailable = set(unavailable_reviewers)
    available_required = tuple(name for name in required if name not in unavailable)
    approvals = set(qualifying_approvals)
    if snapshot.contract.policy == "all-reviewers" or snapshot.force_full:
        selected = available_required
        reason = "compatibility policy" if snapshot.contract.policy == "all-reviewers" else "force-full latch"
    elif final_sweep:
        selected = tuple(name for name in available_required if name not in approvals)
        reason = "final exact-head sweep for missing approvals"
    elif not classification.narrow:
        selected = available_required
        reason = f"full board required: {classification.reason}"
    else:
        owners = {
            owner
            for obligation in snapshot.obligations
            if obligation.active
            for owner in obligation.pending_owners
        }
        selected = tuple(name for name in available_required if name in owners)
        reason = "narrow transition: pending resolution owners and co-owners"
        if not selected and available_required:
            selected = available_required
            reason = "narrow transition was not actionable without pending owners"
    selected_set = set(selected)
    paused: list[tuple[str, str]] = []
    for name in required:
        if name in selected_set:
            continue
        if name in unavailable:
            pause_reason = "reviewer unavailable; required approval remains outstanding"
        elif name in approvals:
            pause_reason = "qualifying exact-head approval carried; no new turn needed"
        elif snapshot.contract.policy == "all-reviewers":
            pause_reason = "compatibility scheduling did not select this reviewer"
        elif final_sweep:
            pause_reason = "qualifying exact-head approval carry"
        elif not classification.narrow or snapshot.force_full:
            pause_reason = "reviewer was not selected by the full-board transition"
        else:
            pause_reason = "selective intermediate pause; no pending obligation ownership"
        paused.append((name, pause_reason))
    eligible = len(tuple(name for name in required if name not in approvals and name not in unavailable))
    avoided = max(0, eligible - len(selected)) if snapshot.contract.policy == "selective-intermediate" and not final_sweep else 0
    return SchedulingDecision(
        selected_reviewers=selected,
        paused_reviewers=tuple(paused),
        reason=reason,
        classification=classification,
        final_sweep=final_sweep,
        calls_avoided=avoided,
    )


def make_contract(required_reviewers: Sequence[str], policy: str, broad_rules: Sequence[str] | None = None) -> ReviewSchedulingContract:
    return ReviewSchedulingContract(
        required_reviewers=tuple(required_reviewers),
        policy=policy,
        broad_rules=normalize_broad_rules(broad_rules),
    )
