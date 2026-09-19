"""Pure scheduling and diff classification for staged PR re-review.

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

PR_REVIEW_POLICIES = frozenset(
    {"all-reviewers", "selective-intermediate", "primary-then-panel"}
)

SCHEDULER_PHASES = frozenset(
    {
        "full-board",
        "primary",
        "secondary-audit",
        "remediation",
        "final-secondary-sweep",
    }
)

# Phases that only occur once the secondary panel has been (or is being)
# opened.  Under ``primary-then-panel`` a checkpoint in one of these phases is
# NOT panel evidence by itself (#840): a pre-#840 run could reach them through
# an automatic pre-approval full board.  Only a qualified panel opening, derived
# from comment-ordered history by the orchestrator, opens the panel.
PANEL_OPENED_PHASES = frozenset(
    {"full-board", "secondary-audit", "remediation", "final-secondary-sweep"}
)

FORCE_FULL_SOURCES = frozenset({"operator", "automatic"})

# Audit-reason prefixes that distinguish the strict pre-panel fallback from the
# conservative post-panel owner/full-board fallback in comments and logs.
STRICT_PRE_PANEL_PREFIX = "strict pre-panel fallback: "
POST_PANEL_PREFIX = "post-panel fallback: "
OPERATOR_FORCE_FULL_REASON = (
    "operator force-full: the complete board is authorized by --pr-review-force-full"
)
PRE_PANEL_SAFETY_MESSAGE = (
    "pre-panel safety cannot be established: {detail} but no qualified panel opening "
    "exists; no reviewer was invoked. Rerun with --pr-review-force-full to authorize "
    "the complete board."
)


class PrePanelSafetyError(AgentLoopError):
    """The staged scheduler cannot stay primary-only without guessing (#840).

    Raised instead of silently spending the secondary panel early.  The operator
    force-full override is the documented escape hatch.
    """


def pre_panel_safety_message(detail: str) -> str:
    return PRE_PANEL_SAFETY_MESSAGE.format(detail=detail)


@dataclass(frozen=True)
class ReviewPolicyCapabilities:
    """Named scheduler capabilities shared by every orchestration safety gate.

    Keeping these properties explicit avoids making unrelated behavior depend on
    one policy-name equality check.  The all-reviewers policy deliberately has
    no scheduler optimization, while both opt-in policies use the same durable
    ownership, resume, and exact-head safety boundaries.
    """

    scheduler_enabled: bool
    owner_scoped_reconciliation: bool
    selective_pausing: bool
    phase_aware: bool
    counts_avoided_calls: bool
    requires_primary: bool = False
    # When metadata recovery forces the complete board, the staged policy
    # latches that decision durably for the rest of the run and every resume.
    # The selective policy keeps its historical single-decision override.
    recovery_latches_force_full: bool = False


def policy_capabilities(policy: str) -> ReviewPolicyCapabilities:
    if policy == "all-reviewers":
        return ReviewPolicyCapabilities(
            scheduler_enabled=False,
            owner_scoped_reconciliation=False,
            selective_pausing=False,
            phase_aware=False,
            counts_avoided_calls=False,
        )
    if policy == "selective-intermediate":
        return ReviewPolicyCapabilities(
            scheduler_enabled=True,
            owner_scoped_reconciliation=True,
            selective_pausing=True,
            phase_aware=False,
            counts_avoided_calls=True,
        )
    if policy == "primary-then-panel":
        return ReviewPolicyCapabilities(
            scheduler_enabled=True,
            owner_scoped_reconciliation=True,
            selective_pausing=True,
            phase_aware=True,
            counts_avoided_calls=True,
            requires_primary=True,
            recovery_latches_force_full=True,
        )
    raise AgentLoopError(f"Unsupported PR review policy: {policy!r}.")


# Descriptive alias for callers that prefer the longer name.
review_policy_capabilities = policy_capabilities

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
    primary_reviewer: str | None = None

    def __post_init__(self) -> None:
        reviewers = tuple(self.required_reviewers)
        if (
            not reviewers
            or any(not isinstance(item, str) or not item for item in reviewers)
            or len(set(reviewers)) != len(reviewers)
        ):
            raise AgentLoopError("Scheduler contract requires a unique reviewer set.")
        capabilities = policy_capabilities(self.policy)
        primary = self.primary_reviewer
        if primary is not None and (not isinstance(primary, str) or not primary):
            raise AgentLoopError("Scheduler primary reviewer must be a non-empty string.")
        if capabilities.requires_primary:
            if len(reviewers) < 2:
                raise AgentLoopError(
                    "primary-then-panel requires one primary and at least one secondary reviewer."
                )
            if primary not in reviewers:
                raise AgentLoopError(
                    "primary-then-panel primary reviewer must be a member of the reviewer board."
                )
        elif primary is not None:
            raise AgentLoopError(
                "A primary reviewer may only be configured with primary-then-panel."
            )
        rules = normalize_broad_rules(self.broad_rules)
        digest = broad_rules_digest(rules)
        if self.broad_rules_digest is not None and self.broad_rules_digest != digest:
            raise AgentLoopError("Scheduler broad-rule digest does not match its rules.")
        object.__setattr__(self, "required_reviewers", reviewers)
        object.__setattr__(self, "primary_reviewer", primary)
        object.__setattr__(self, "broad_rules", rules)
        object.__setattr__(self, "broad_rules_digest", digest)

    def as_dict(self) -> dict[str, object]:
        return {
            "required_reviewers": list(self.required_reviewers),
            "policy": self.policy,
            "primary_reviewer": self.primary_reviewer,
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
        primary = value.get("primary_reviewer")
        if (
            not isinstance(reviewers, list)
            or not isinstance(rules, list)
            or any(not isinstance(item, str) or not item for item in reviewers)
            or not isinstance(policy, str)
            or (primary is not None and not isinstance(primary, str))
        ):
            raise AgentLoopError("Incomplete scheduler contract.")
        return cls(
            required_reviewers=tuple(reviewers),
            policy=policy,
            primary_reviewer=primary,
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
    # Automatic full-board request.  Under ``primary-then-panel`` it is honored
    # only after a qualified panel opening; before one it degrades to a strict
    # primary-only full-context turn.  Other policies honor it unconditionally.
    force_full: bool = False
    phase: str | None = None
    # Explicit operator authorization (``--pr-review-force-full`` or a
    # persisted operator-sourced latch).  It always selects the complete board.
    operator_force_full: bool = False
    # True only when comment-ordered history holds a qualified panel opening.
    panel_evidence: bool = False
    # Human-readable automatic fallback reasons for the current decision.
    fallback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SchedulingDecision:
    selected_reviewers: tuple[str, ...]
    paused_reviewers: tuple[tuple[str, str], ...]
    reason: str
    classification: TransitionClassification
    final_sweep: bool = False
    calls_avoided: int = 0
    phase: str = "full-board"
    primary_reviewer: str | None = None
    active_owners: tuple[str, ...] = ()


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
    phase: str | None = None,
) -> SchedulingDecision:
    """Choose the board for one round; unavailable reviewers remain required."""
    required = snapshot.contract.required_reviewers
    unavailable = set(unavailable_reviewers)
    available_required = tuple(name for name in required if name not in unavailable)
    approvals = set(qualifying_approvals)
    capabilities = policy_capabilities(snapshot.contract.policy)
    primary = snapshot.contract.primary_reviewer
    required_names = set(required)
    if snapshot.contract.policy == "all-reviewers" or (
        not capabilities.requires_primary
        and (snapshot.force_full or snapshot.operator_force_full)
    ):
        selected = available_required
        reason = "compatibility policy" if snapshot.contract.policy == "all-reviewers" else "force-full latch"
        selected_phase = "full-board"
    elif capabilities.requires_primary and snapshot.operator_force_full:
        selected = available_required
        selected_phase = "full-board"
        reason = OPERATOR_FORCE_FULL_REASON
    elif capabilities.requires_primary:
        # The selection is derived from exact-head approval evidence, the
        # active obligation ledger, and qualified panel evidence.  A phase
        # checkpoint is validated but never opens the panel by itself, and it
        # can never bypass the primary gate or grant an approval.
        checkpoint_phase = phase if phase is not None else snapshot.phase
        if checkpoint_phase is not None and checkpoint_phase not in SCHEDULER_PHASES:
            raise AgentLoopError(f"Unsupported scheduler phase checkpoint: {checkpoint_phase!r}.")
        panel_opened = snapshot.panel_evidence
        primary_approved = primary is not None and primary in approvals
        active_obligations = tuple(
            obligation for obligation in snapshot.obligations if obligation.active
        )
        pending_owner_set = {
            owner for obligation in active_obligations for owner in obligation.pending_owners
        }
        fallback_detail = "; ".join(snapshot.fallback_reasons)
        if not panel_opened:
            secondaries = required_names - {primary}
            secondary_pending = pending_owner_set & secondaries
            if secondary_pending:
                # Secondaries are never finding owners before their first
                # legitimate panel invocation; do not guess ownership and do
                # not spend the panel silently.
                item_ids = sorted(
                    obligation.item_id
                    for obligation in active_obligations
                    if set(obligation.pending_owners) & secondary_pending
                )
                raise PrePanelSafetyError(
                    pre_panel_safety_message(
                        f"finding(s) {', '.join(item_ids)} are pending on configured "
                        f"secondary reviewer(s) {', '.join(sorted(secondary_pending))}"
                    )
                )
            if not primary_approved:
                selected = (primary,) if primary in available_required else ()
                selected_phase = "primary"
                unscoped = any(
                    obligation.scope is None
                    or not set(obligation.pending_owners) <= required_names
                    for obligation in active_obligations
                )
                strict_reasons: list[str] = []
                if snapshot.force_full or fallback_detail:
                    strict_reasons.append(fallback_detail or "automatic full-board fallback requested")
                if not classification.narrow and (
                    active_obligations or snapshot.previous_sha is not None
                ):
                    strict_reasons.append(classification.reason)
                elif unscoped:
                    strict_reasons.append("an active obligation has no valid exact fix scope")
                if strict_reasons:
                    reason = (
                        STRICT_PRE_PANEL_PREFIX
                        + "; ".join(dict.fromkeys(strict_reasons))
                        + "; primary re-invoked with full context"
                    )
                elif active_obligations:
                    # Primary blocking loop: the primary rechecks its own
                    # findings on each narrow head until it approves.
                    reason = (
                        "primary phase: primary rechecks its findings before the secondary panel"
                    )
                else:
                    reason = (
                        "primary phase: exact-head primary approval is required before the "
                        "secondary panel"
                    )
            else:
                # Exact-head primary approval opens the panel: every available
                # secondary lacking a qualified exact-head approval receives
                # its first independent audit, whatever the transition.
                selected = tuple(
                    name
                    for name in available_required
                    if name != primary and name not in approvals
                )
                selected_phase = "secondary-audit"
                reason = "independent secondary audit after exact-head primary approval"
                remaining = sorted(obligation.item_id for obligation in active_obligations)
                if remaining:
                    reason += (
                        f"; non-reviewer obligations remain: {', '.join(remaining)}; "
                        "coder repair follows under post-panel rules"
                    )
        elif snapshot.force_full:
            selected = available_required
            selected_phase = "full-board"
            reason = POST_PANEL_PREFIX + "force-full latch" + (
                f" ({fallback_detail})" if fallback_detail else ""
            )
        elif active_obligations and not classification.narrow:
            # Unsafe ownership, scope, history, or change classification with
            # any active finding reactivates the complete board before the
            # primary gate is consulted.
            selected = available_required
            selected_phase = "full-board"
            reason = f"{POST_PANEL_PREFIX}full board required: {classification.reason}"
        elif not classification.narrow:
            # Panel evidence exists on an older head and the transition is
            # unsafe; every reviewer must see the complete current head.
            selected = available_required
            selected_phase = "full-board"
            reason = (
                f"{POST_PANEL_PREFIX}full board required after panel evidence: "
                f"{classification.reason}"
            )
        elif active_obligations:
            owners = set(pending_owner_set)
            owners.add(primary)
            selected = tuple(
                name for name in available_required
                if name in owners and name not in approvals
            )
            selected_phase = "remediation"
            reason = f"{POST_PANEL_PREFIX}narrow remediation: finding owners and primary must recheck"
        elif not primary_approved:
            selected = (primary,) if primary in available_required else ()
            selected_phase = "remediation"
            reason = (
                "remediation: exact-head primary approval is outstanding before the "
                "final secondary sweep"
            )
        else:
            selected = tuple(
                name
                for name in available_required
                if name != primary and name not in approvals
            )
            selected_phase = "final-secondary-sweep"
            reason = "final exact-head secondary sweep for missing approvals"
    elif final_sweep:
        selected = tuple(name for name in available_required if name not in approvals)
        reason = "final exact-head sweep for missing approvals"
        selected_phase = "final-secondary-sweep"
    elif not classification.narrow:
        selected = available_required
        reason = f"full board required: {classification.reason}"
        selected_phase = "full-board"
    else:
        owners = {
            owner
            for obligation in snapshot.obligations
            if obligation.active
            for owner in obligation.pending_owners
        }
        selected = tuple(name for name in available_required if name in owners)
        reason = "narrow transition: pending resolution owners and co-owners"
        selected_phase = "remediation"
        if not selected and available_required:
            selected = available_required
            reason = "narrow transition was not actionable without pending owners"
            selected_phase = "full-board"
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
        elif capabilities.requires_primary and selected_phase == "primary":
            pause_reason = "primary phase; secondary panel waits for exact-head primary approval"
        elif capabilities.requires_primary and selected_phase == "secondary-audit":
            pause_reason = "secondary audit has not selected this already qualifying reviewer"
        elif capabilities.requires_primary and selected_phase == "remediation":
            pause_reason = (
                "remediation did not identify this reviewer as an owner or primary; "
                "an exact-head secondary sweep follows clearance"
            )
        elif capabilities.requires_primary and selected_phase == "final-secondary-sweep":
            pause_reason = "final secondary sweep did not select this reviewer"
        elif final_sweep:
            pause_reason = "qualifying exact-head approval carry"
        elif not classification.narrow or snapshot.force_full or snapshot.operator_force_full:
            pause_reason = "reviewer was not selected by the full-board transition"
        else:
            pause_reason = "selective intermediate pause; no pending obligation ownership"
        paused.append((name, pause_reason))
    eligible = len(tuple(name for name in required if name not in approvals and name not in unavailable))
    avoided = max(0, eligible - len(selected)) if capabilities.counts_avoided_calls else 0
    active_owners = tuple(
        sorted(
            {
                owner
                for obligation in snapshot.obligations
                if obligation.active
                for owner in obligation.pending_owners
            }
        )
    )
    return SchedulingDecision(
        selected_reviewers=selected,
        paused_reviewers=tuple(paused),
        reason=reason,
        classification=classification,
        final_sweep=final_sweep,
        calls_avoided=avoided,
        phase=selected_phase,
        primary_reviewer=primary,
        active_owners=active_owners,
    )


def make_contract(
    required_reviewers: Sequence[str],
    policy: str,
    broad_rules: Sequence[str] | None = None,
    primary_reviewer: str | None = None,
) -> ReviewSchedulingContract:
    return ReviewSchedulingContract(
        required_reviewers=tuple(required_reviewers),
        policy=policy,
        primary_reviewer=primary_reviewer,
        broad_rules=normalize_broad_rules(broad_rules),
    )
