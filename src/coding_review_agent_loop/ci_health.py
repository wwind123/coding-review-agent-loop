"""CI infrastructure-stall detection.

Owns the check data types (`PullRequestCheck`, `PullRequestChecks`) so this
module has no dependency on `github.py`'s live API calls; `github.py` imports
and re-exports these types instead. Everything here is pure classification
over already-fetched data, so it only depends on stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Sequence


@dataclass(frozen=True)
class PullRequestCheck:
    name: str
    kind: Literal["check_run", "status_context"]
    status: str
    url: str | None = None
    check_id: int | None = None
    run_id: str | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    creator_login: str | None = None
    creator_id: int | None = None
    description: str | None = None


@dataclass(frozen=True)
class StalledCheck:
    name: str
    kind: Literal["check_run", "status_context"]
    reason: Literal["queued_too_long", "runner_unavailable"]
    check_id: int | None
    run_id: str | None
    url: str | None
    age_seconds: float | None

    def describe(self) -> str:
        location_parts: list[str] = []
        if self.check_id is not None:
            location_parts.append(f"check run {self.check_id}")
        if self.run_id is not None:
            location_parts.append(f"workflow run {self.run_id}")
        if self.url:
            location_parts.append(self.url)
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        if self.reason == "queued_too_long":
            detail = f"queued {_format_age(self.age_seconds)} without starting a job"
        else:
            detail = "cancelled before any job started (runner unavailable)"
        return f"{self.name}{location} — {detail}"

    def identifiers(self) -> tuple[str, ...]:
        raw_ids = [self.name, self.run_id, str(self.check_id) if self.check_id is not None else None, self.url]
        seen: list[str] = []
        for raw_id in raw_ids:
            if raw_id and raw_id not in seen:
                seen.append(raw_id)
        return tuple(seen)


@dataclass(frozen=True)
class PullRequestChecks:
    state: Literal["passing", "failing", "pending", "no_checks", "unavailable"]
    required_checks: tuple[str, ...]
    passing: tuple[PullRequestCheck, ...]
    pending: tuple[PullRequestCheck, ...]
    failing: tuple[PullRequestCheck, ...]
    missing_required: tuple[str, ...]
    branch_protection_status: Literal["configured", "not_found", "forbidden", "unavailable"]
    branch_protection_note: str | None = None
    check_query_status: Literal["ok", "partial", "unavailable"] = "ok"
    check_query_errors: tuple[str, ...] = ()
    infrastructure_stalls: tuple[StalledCheck, ...] = field(default=())


@dataclass(frozen=True)
class CiInfrastructureStall:
    checks: tuple[StalledCheck, ...]

    @property
    def is_stalled(self) -> bool:
        return bool(self.checks)


_RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")


def _extract_run_id(url: str | None) -> str | None:
    if not url:
        return None
    match = _RUN_ID_RE.search(url)
    return match.group(1) if match else None


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = raw.strip()
    if not text:
        return None
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _age_seconds(created_at: str | None, *, now: datetime) -> float | None:
    parsed = _parse_timestamp(created_at)
    if parsed is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    delta = (now - parsed).total_seconds()
    return max(0.0, delta)


def _format_age(age_seconds: float | None) -> str:
    if age_seconds is None:
        return "for an unknown duration"
    minutes = int(age_seconds // 60)
    return f"{minutes}m"


_QUEUED_LIKE_STATUSES = {"queued", "pending", "waiting", "requested"}
_UNSTARTED_CANCEL_CONCLUSIONS = {"cancelled", "startup_failure"}


def classify_ci_infrastructure_stall(
    checks: Sequence[PullRequestCheck],
    *,
    now: datetime,
    grace_seconds: int,
) -> CiInfrastructureStall:
    """Classify individual checks as external-infrastructure stalls.

    Only ``check_run`` entries are eligible: commit-status contexts
    (``status_context``) are set by arbitrary external systems, not GitHub
    Actions runners, so they are never classified as an infrastructure stall.
    """
    stalled: list[StalledCheck] = []
    for check in checks:
        if check.kind != "check_run":
            continue
        status = (check.status or "").strip().lower()

        if status in _UNSTARTED_CANCEL_CONCLUSIONS and (
            check.started_at is None or check.started_at == check.completed_at
        ):
            stalled.append(
                StalledCheck(
                    name=check.name,
                    kind=check.kind,
                    reason="runner_unavailable",
                    check_id=check.check_id,
                    run_id=check.run_id,
                    url=check.url,
                    age_seconds=_age_seconds(check.created_at, now=now),
                )
            )
            continue

        if status in _QUEUED_LIKE_STATUSES and check.started_at is None:
            age = _age_seconds(check.created_at, now=now)
            if age is not None and age > grace_seconds:
                stalled.append(
                    StalledCheck(
                        name=check.name,
                        kind=check.kind,
                        reason="queued_too_long",
                        check_id=check.check_id,
                        run_id=check.run_id,
                        url=check.url,
                        age_seconds=age,
                    )
                )

    return CiInfrastructureStall(checks=tuple(stalled))


def is_wholly_infrastructure_blocked(pr_checks: PullRequestChecks) -> bool:
    """True only when every blocking/pending signal is an infrastructure stall.

    This is the single gate for the orchestrator terminal stop, the reviewer
    downgrade, and the auto-merge infrastructure outcome. A never-reporting
    required check, unknown branch protection, a partial/unavailable check
    query, or any genuinely failing/pending check makes this False.
    """
    if not pr_checks.infrastructure_stalls:
        return False
    if pr_checks.missing_required:
        return False
    if pr_checks.state == "unavailable":
        return False
    if pr_checks.branch_protection_status == "unavailable":
        return False
    if pr_checks.check_query_status != "ok":
        return False
    stalled_keys = {(stall.kind, stall.name) for stall in pr_checks.infrastructure_stalls}
    for check in (*pr_checks.failing, *pr_checks.pending):
        if (check.kind, check.name) not in stalled_keys:
            return False
    return True


_IDENTIFIER_SENTINEL = "identsentinel"

# Substring phrases whose presence signals stall semantics. Checked against the
# normalized text after identifier substitution.
_STALL_SEMANTIC_PHRASES: tuple[str, ...] = (
    "runner unavailable",
    "hosted runner",
    "never started",
    "did not start",
    "no runner",
    "before execution",
    "runner acquisition",
    "queued",
    "cancelled",
    "canceled",
    "runner",
    "capacity",
)

# Closed word-level vocabulary: every remaining token in a canonical stall-only
# text must belong to this set (plus digits, the identifier sentinel, and
# URLs) or the match fails. Kept intentionally small so an out-of-vocabulary
# word (a filename, symbol, or code-defect noun) fails the match.
_STALL_VOCABULARY_WORDS: frozenset[str] = frozenset(
    {
        "queued", "cancelled", "canceled", "runner", "runners", "hosted",
        "never", "started", "starting", "start", "did", "no", "not",
        "unavailable", "before", "execution", "capacity", "acquire",
        "acquired", "acquisition", "recover", "recovers", "recovered",
    }
)

_FUNCTION_WORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "is", "was", "were", "this",
        "that", "these", "those", "for", "of", "to", "with", "without",
        "than", "over", "past", "its", "it", "has", "have", "had", "been",
        "being", "be", "still", "remains", "remained", "remain", "so",
        "because", "due", "on", "in", "at", "as", "any", "job", "jobs",
        "more", "external", "infrastructure", "issue", "since", "once",
        "will", "should", "resume", "after", "work", "affected", "which",
        "there", "not", "all", "one", "run", "runs", "runner",
    }
)

_CI_NOUNS: frozenset[str] = frozenset(
    {
        "check", "checks", "workflow", "workflows", "run", "runs", "job",
        "jobs", "github", "actions", "ci", "infrastructure", "external",
    }
)

_TIME_UNITS: frozenset[str] = frozenset(
    {
        "m", "min", "mins", "minute", "minutes", "h", "hr", "hrs", "hour",
        "hours", "s", "sec", "secs", "second", "seconds",
    }
)

_CLOSED_VOCABULARY: frozenset[str] = (
    _STALL_VOCABULARY_WORDS | _FUNCTION_WORDS | _CI_NOUNS | _TIME_UNITS
)

_URL_RE = re.compile(r"https?://\S+")
_WORD_RE = re.compile(r"[a-z0-9]+")
_DIGIT_LETTER_BOUNDARY_RE = re.compile(r"(\d)([a-z])")
_LETTER_DIGIT_BOUNDARY_RE = re.compile(r"([a-z])(\d)")
_LEADING_MARKDOWN_RE = re.compile(r"^[\s\-\*•`'\"]+")

_MAX_CANONICAL_TEXT_LENGTH = 400


def is_canonical_stall_only_text(text: str, *, stalls: Sequence[StalledCheck]) -> bool:
    """Whole-text, closed-vocabulary check for "this item is only a CI stall".

    Fails closed: any token outside the closed vocabulary (a filename,
    symbol, or code-defect noun) makes this return False, so a mixed item
    naming a stalled run and then describing an unrelated code defect is
    rejected in full.
    """
    if not text or not text.strip():
        return False
    if len(text) > _MAX_CANONICAL_TEXT_LENGTH:
        return False

    normalized = text.strip().lower()
    normalized = _LEADING_MARKDOWN_RE.sub("", normalized)
    normalized = normalized.strip("`'\" \t")
    if not normalized:
        return False

    identifiers = sorted(
        {identifier.lower() for stall in stalls for identifier in stall.identifiers() if identifier},
        key=len,
        reverse=True,
    )

    substituted = normalized
    sentinel_count = 0
    for identifier in identifiers:
        if identifier and identifier in substituted:
            sentinel_count += substituted.count(identifier)
            substituted = substituted.replace(identifier, f" {_IDENTIFIER_SENTINEL} ")

    if sentinel_count == 0:
        return False

    substituted = _URL_RE.sub(" url ", substituted)

    if not any(phrase in substituted for phrase in _STALL_SEMANTIC_PHRASES):
        return False

    tokenizable = _DIGIT_LETTER_BOUNDARY_RE.sub(r"\1 \2", substituted)
    tokenizable = _LETTER_DIGIT_BOUNDARY_RE.sub(r"\1 \2", tokenizable)

    for token in _WORD_RE.findall(tokenizable):
        if token == _IDENTIFIER_SENTINEL or token == "url":
            continue
        if token.isdigit():
            continue
        if token in _CLOSED_VOCABULARY:
            continue
        return False

    return True
