"""Approved follow-up publishing and formatting helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Literal

from .config import AgentLoopConfig
from .errors import AgentLoopError, QuotaResetExceededError
from .github import (
    FoundIssue,
    create_issue,
    post_issue_comment,
    post_pr_comment,
    search_issues,
    validate_open_issue,
)
from .logging import log
from .protocol import ApprovedFollowup, UnresolvedReviewItem
from .runner import Runner
from .protocol_markers import TrustedBody, sanitize_historical_text
from .semantic_dedupe import (
    BudgetExhausted,
    SemanticCandidate,
    SemanticDedupeMatcher,
    SemanticMatch,
    SemanticTransport,
)

MAX_APPROVED_FOLLOWUP_ISSUES = 3
APPROVED_FOLLOWUP_MARKER_RE = re.compile(
    r"<!--\s*AGENT_APPROVED_FOLLOWUPS:\s*pr=(?P<pr>\d+)\s+head=(?P<head>\S+)\s+mode=(?P<mode>[a-z-]+)\s*-->",
    re.I,
)
PLAN_APPROVED_FOLLOWUP_MARKER_RE = re.compile(
    r"<!--\s*AGENT_PLAN_APPROVED_FOLLOWUPS:\s*issue=(?P<issue>\d+)\s+plan=(?P<plan>\S+)\s+mode=(?P<mode>[a-z-]+)\s*-->",
    re.I,
)
FOLLOWUP_UPDATE_SPLIT_RE = re.compile(r"\n{2,}Update from ", re.I)


@dataclass(frozen=True)
class GroupedApprovedFollowup:
    text: str
    items: tuple[ApprovedFollowup, ...]

    @property
    def reviewers(self) -> tuple[str, ...]:
        reviewers: list[str] = []
        for item in self.items:
            if item.reviewer not in reviewers:
                reviewers.append(item.reviewer)
        return tuple(reviewers)


@dataclass(frozen=True)
class FollowupSourceContext:
    """Repository-scoped provenance used to narrow historical trackers."""

    repo: str
    source_kind: Literal["pr", "plan"]
    source_number: int
    source_identity: str | None = None
    parent_issue_numbers: tuple[int, ...] = ()
    related_issue_numbers: tuple[int, ...] = ()
    related_pr_numbers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.repo or "/" not in self.repo:
            raise AgentLoopError("follow-up source context requires an owner/name repository")
        if self.source_kind not in {"pr", "plan"}:
            raise AgentLoopError("follow-up source context kind must be pr or plan")
        if isinstance(self.source_number, bool) or not isinstance(self.source_number, int) or self.source_number <= 0:
            raise AgentLoopError("follow-up source number must be positive")
        for field_name in ("parent_issue_numbers", "related_issue_numbers", "related_pr_numbers"):
            values = getattr(self, field_name)
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
                raise AgentLoopError(f"{field_name} must contain positive integer identities")

    def render(self) -> str:
        parts = [
            f"repository={sanitize_historical_text(self.repo)}",
            f"source={self.source_kind}#{self.source_number}",
        ]
        if self.source_identity:
            parts.append(f"identity={sanitize_historical_text(self.source_identity)}")
        if self.parent_issue_numbers:
            parts.append("parent issue(s)=" + ", ".join(f"#{n}" for n in self.parent_issue_numbers))
        if self.related_issue_numbers:
            parts.append("related issue(s)=" + ", ".join(f"#{n}" for n in self.related_issue_numbers))
        if self.related_pr_numbers:
            parts.append("related PR(s)=" + ", ".join(f"#{n}" for n in self.related_pr_numbers))
        return "; ".join(parts)


@dataclass(frozen=True)
class ApprovedFollowupReconciliation:
    groups: tuple[GroupedApprovedFollowup, ...]
    selected_groups: tuple[GroupedApprovedFollowup, ...]
    skipped_by_cap: int
    deduplicated_count: int


@dataclass(frozen=True)
class PlanApprovedFollowupSource:
    item_id: str | None
    reviewer: str
    source_round: int | None
    text: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanGroupedApprovedFollowup:
    text: str
    items: tuple[ApprovedFollowup, ...]
    sources: tuple[PlanApprovedFollowupSource, ...]

    @property
    def reviewers(self) -> tuple[str, ...]:
        reviewers: list[str] = []
        for source in self.sources:
            if source.reviewer not in reviewers:
                reviewers.append(source.reviewer)
        return tuple(reviewers)


@dataclass(frozen=True)
class PlanApprovedFollowupReconciliation:
    groups: tuple[PlanGroupedApprovedFollowup, ...]
    selected_groups: tuple[PlanGroupedApprovedFollowup, ...]
    skipped_by_cap: int
    deduplicated_count: int


@dataclass(frozen=True)
class FollowupPublication:
    group: GroupedApprovedFollowup | PlanGroupedApprovedFollowup
    status: Literal["created", "reused", "uncertain", "cap-skipped"]
    issue_number: int | None = None
    issue_url: str | None = None
    reason: str | None = None


_FOLLOWUP_STOPWORDS = {
    "a",
    "about",
    "across",
    "add",
    "after",
    "against",
    "all",
    "also",
    "and",
    "another",
    "around",
    "as",
    "before",
    "behavior",
    "better",
    "broader",
    "but",
    "by",
    "can",
    "cleanup",
    "consider",
    "coverage",
    "doc",
    "docs",
    "document",
    "documentation",
    "ensure",
    "for",
    "from",
    "future",
    "handle",
    "handled",
    "in",
    "include",
    "issue",
    "later",
    "make",
    "mention",
    "note",
    "of",
    "on",
    "or",
    "pr",
    "separate",
    "should",
    "so",
    "that",
    "the",
    "this",
    "to",
    "track",
    "update",
    "use",
    "when",
    "with",
    "work",
}


_CODE_OR_PATH_RE = re.compile(
    r"`([^`]+)`|"
    r"\b(?:[A-Za-z_][\w]*\.)+[A-Za-z_][\w]*\b|"
    r"\b[\w.-]+/[\w./-]+\b|"
    r"\b[\w.-]+\.(?:py|md|txt|toml|yaml|yml|json|js|ts|tsx|jsx|html|css|rst)\b"
)


def _followup_identifier_keys(text: str) -> set[str]:
    text = _followup_main_text(text)
    keys: set[str] = set()
    for match in _CODE_OR_PATH_RE.finditer(text):
        identifier = next((group for group in match.groups() if group), match.group(0))
        normalized = _normalize_followup_key(identifier)
        if normalized:
            keys.add(normalized)
    return keys


def _followup_topic_terms(text: str) -> set[str]:
    text = _followup_main_text(text)
    normalized = _normalize_followup_key(text)
    terms = {
        term
        for term in normalized.split()
        if len(term) >= 4 and term not in _FOLLOWUP_STOPWORDS and not term.isdigit()
    }
    return terms


def _followup_candidate_keys(text: str) -> set[str]:
    text = _followup_main_text(text)
    keys = {_normalize_followup_key(text)}
    heading_key = _followup_heading_key(text)
    if heading_key:
        keys.add(f"heading:{heading_key}")
    identifiers = _followup_identifier_keys(text)
    keys.update(f"id:{identifier}" for identifier in identifiers)
    terms = _followup_topic_terms(text)
    if len(terms) >= 3:
        keys.add("terms:" + "+".join(sorted(terms)))
    if identifiers and len(terms) >= 2:
        for identifier in identifiers:
            for term in sorted(terms):
                keys.add(f"id-term:{identifier}+{term}")
    return {key for key in keys if key}


def _followup_similarity(left: ApprovedFollowup, right: ApprovedFollowup) -> float:
    left_terms = _followup_topic_terms(left.text)
    right_terms = _followup_topic_terms(right.text)
    if not left_terms or not right_terms:
        return 0.0
    common_terms = left_terms & right_terms
    if len(common_terms) < 3:
        return 0.0
    left_ids = _followup_identifier_keys(left.text)
    right_ids = _followup_identifier_keys(right.text)
    if left_ids or right_ids:
        id_overlap = left_ids & right_ids
        if left_ids and right_ids and not id_overlap:
            return 0.0
        if id_overlap and len(common_terms) >= 2:
            return 1.0
    if common_terms == left_terms or common_terms == right_terms:
        return 1.0
    # A concise restatement of carried work can omit context while retaining
    # most of its concrete terms. Treat that as the same follow-up before the
    # broader Jaccard threshold below.
    if (
        len(common_terms) >= 4
        and len(common_terms) / min(len(left_terms), len(right_terms)) >= 0.6
    ):
        return 0.55
    return len(common_terms) / len(left_terms | right_terms)


def _followup_specificity_score(followup: ApprovedFollowup, reviewer_counts: Counter[str]) -> tuple[int, int, int, int]:
    identifiers = _followup_identifier_keys(followup.text)
    normalized_length = len(_normalize_followup_key(_followup_main_text(followup.text)))
    has_disposition_note = int("Update from " in followup.text)
    reviewer_support = reviewer_counts[followup.text]
    return (
        len(identifiers),
        has_disposition_note,
        reviewer_support,
        normalized_length,
    )


def _select_canonical_followup(items: Sequence[ApprovedFollowup]) -> ApprovedFollowup:
    reviewer_counts = Counter(item.text for item in items)
    return max(items, key=lambda item: _followup_specificity_score(item, reviewer_counts))


def _followup_main_text(text: str) -> str:
    return FOLLOWUP_UPDATE_SPLIT_RE.split(text, maxsplit=1)[0].strip()


def _followup_update_notes(text: str) -> tuple[str, ...]:
    parts = FOLLOWUP_UPDATE_SPLIT_RE.split(text)
    return tuple(f"Update from {part.strip()}" for part in parts[1:] if part.strip())


def _safe_followup_main_text(text: str) -> str:
    return sanitize_historical_text(_followup_main_text(text))


def _safe_followup_update_notes(text: str) -> tuple[str, ...]:
    return tuple(sanitize_historical_text(note) for note in _followup_update_notes(text))


def _approved_followup_from_unresolved_item(item: UnresolvedReviewItem) -> ApprovedFollowup:
    text = item.text
    for note in item.notes:
        update_line = f"Update from {note}"
        if update_line not in text:
            text = f"{text.rstrip()}\n\n{update_line}"
    return ApprovedFollowup(reviewer=item.reviewer, text=text)


def _plan_followup_source_from_unresolved_item(item: UnresolvedReviewItem) -> PlanApprovedFollowupSource:
    return PlanApprovedFollowupSource(
        item_id=item.item_id,
        reviewer=item.reviewer,
        source_round=item.source_round,
        text=item.text,
        notes=tuple(item.notes),
    )


def _approved_followup_from_plan_source(source: PlanApprovedFollowupSource) -> ApprovedFollowup:
    text = source.text
    for note in source.notes:
        update_line = f"Update from {note}"
        if update_line not in text:
            text = f"{text.rstrip()}\n\n{update_line}"
    return ApprovedFollowup(reviewer=source.reviewer, text=text)


def reconcile_approved_followups(
    followups: Sequence[ApprovedFollowup],
    *,
    issue_limit: int = MAX_APPROVED_FOLLOWUP_ISSUES,
    semantic_matcher: Callable[
        [ApprovedFollowup, tuple[GroupedApprovedFollowup, ...]], SemanticMatch | None
    ] | None = None,
) -> ApprovedFollowupReconciliation:
    grouped: list[GroupedApprovedFollowup] = []
    indexes: dict[str, int] = {}
    for followup in followups:
        keys = _followup_candidate_keys(followup.text)
        existing_index = next((indexes[key] for key in keys if key in indexes), None)
        if existing_index is None:
            for index, group in enumerate(grouped):
                if any(_followup_similarity(followup, item) >= 0.55 for item in group.items):
                    existing_index = index
                    break
        if existing_index is None and semantic_matcher is not None and grouped:
            # Deterministic keys and similarity are always the first pass.  The
            # injected matcher sees only the already-narrowed batch groups and
            # may merge only an explicitly high-confidence, non-null match.
            semantic = semantic_matcher(followup, tuple(grouped))
            if semantic is not None and semantic.confidence == "high":
                target = semantic.duplicate_of
                if isinstance(target, str) and target.startswith("group-"):
                    suffix = target.removeprefix("group-")
                    if suffix.isdigit():
                        candidate_index = int(suffix) - 1
                        if 0 <= candidate_index < len(grouped):
                            existing_index = candidate_index
                elif isinstance(target, int) and not isinstance(target, bool):
                    # Test/in-process matchers may use the natural zero-based
                    # group identity.  Existing issue numbers are never valid
                    # here because batch groups carry string identities.
                    if 0 <= target < len(grouped):
                        existing_index = target
        if existing_index is None:
            indexes.update((key, len(grouped)) for key in keys)
            grouped.append(GroupedApprovedFollowup(text=followup.text, items=(followup,)))
            continue

        existing = grouped[existing_index]
        items = (*existing.items, followup)
        canonical = _select_canonical_followup(items)
        grouped[existing_index] = GroupedApprovedFollowup(text=canonical.text, items=items)
        indexes.update((key, existing_index) for key in keys)

    selected_groups = tuple(grouped[:issue_limit])
    return ApprovedFollowupReconciliation(
        groups=tuple(grouped),
        selected_groups=selected_groups,
        skipped_by_cap=max(0, len(grouped) - len(selected_groups)),
        deduplicated_count=len(followups) - len(grouped),
    )


def reconcile_plan_approved_followups(
    sources: Sequence[PlanApprovedFollowupSource],
    *,
    issue_limit: int = MAX_APPROVED_FOLLOWUP_ISSUES,
    semantic_matcher: Callable[
        [ApprovedFollowup, tuple[GroupedApprovedFollowup, ...]], SemanticMatch | None
    ] | None = None,
) -> PlanApprovedFollowupReconciliation:
    source_by_projection_id: dict[int, PlanApprovedFollowupSource] = {}
    projections: list[ApprovedFollowup] = []
    for source in sources:
        projection = _approved_followup_from_plan_source(source)
        projections.append(projection)
        source_by_projection_id[id(projection)] = source

    reconciliation = reconcile_approved_followups(
        projections,
        issue_limit=issue_limit,
        semantic_matcher=semantic_matcher,
    )

    def plan_group(group: GroupedApprovedFollowup) -> PlanGroupedApprovedFollowup:
        return PlanGroupedApprovedFollowup(
            text=group.text,
            items=group.items,
            sources=tuple(source_by_projection_id[id(item)] for item in group.items),
        )

    groups = tuple(plan_group(group) for group in reconciliation.groups)
    selected_groups = tuple(plan_group(group) for group in reconciliation.selected_groups)
    return PlanApprovedFollowupReconciliation(
        groups=groups,
        selected_groups=selected_groups,
        skipped_by_cap=reconciliation.skipped_by_cap,
        deduplicated_count=reconciliation.deduplicated_count,
    )


def _format_approved_followup_summary(
    pr_number: int,
    reconciliation: ApprovedFollowupReconciliation,
) -> str:
    lines = [
        f"Approved-review future follow-ups for PR #{pr_number}:",
        "",
    ]
    for followup in reconciliation.selected_groups:
        reviewers = ", ".join(
            sanitize_historical_text(reviewer) for reviewer in followup.reviewers
        )
        lines.append(f"- {_safe_followup_main_text(followup.text)} ({reviewers})")
        for item in followup.items:
            for note in _safe_followup_update_notes(item.text):
                lines.append(f"  - {note}")
    lines.extend(
        [
            "",
            (
                f"Reconciliation: {len(reconciliation.selected_groups)} filed/summarized, "
                f"{reconciliation.deduplicated_count} deduplicated, "
                f"{reconciliation.skipped_by_cap} skipped by cap."
            ),
            "",
            "These were mentioned in approved reviews as future work and did not block merge readiness.",
            "",
            "-- coding-review-agent-loop",
        ]
    )
    return "\n".join(lines)


def _approved_followups_marker(pr_number: int, head_sha: str | None, mode: str) -> str:
    head = head_sha or "unknown"
    return f"<!-- AGENT_APPROVED_FOLLOWUPS: pr={pr_number} head={head} mode={mode} -->"


def _append_approved_followups_marker(
    body: str,
    *,
    pr_number: int,
    head_sha: str | None,
    mode: str,
) -> str:
    footer = "\n-- coding-review-agent-loop"
    prefix, found, _suffix = body.rpartition(footer)
    if not found:
        return body
    prefix = prefix.rstrip()
    return "\n".join(
        [
            prefix,
            "",
            _approved_followups_marker(pr_number, head_sha, mode),
            "-- coding-review-agent-loop",
        ]
    )


def _has_approved_followups_marker(
    comments: Sequence[object],
    *,
    pr_number: int,
    head_sha: str | None,
    mode: str,
) -> bool:
    target_head = head_sha or "unknown"
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in APPROVED_FOLLOWUP_MARKER_RE.finditer(body):
            if (
                int(match.group("pr")) == pr_number
                and match.group("head") == target_head
                and match.group("mode").lower() == mode.lower()
            ):
                return True
    return False


def _plan_approved_followups_marker(issue_number: int, plan_hash: str, mode: str) -> str:
    return f"<!-- AGENT_PLAN_APPROVED_FOLLOWUPS: issue={issue_number} plan={plan_hash} mode={mode} -->"


def _append_plan_approved_followups_marker(
    body: str,
    *,
    issue_number: int,
    plan_hash: str,
    mode: str,
) -> str:
    footer = "\n-- coding-review-agent-loop"
    prefix, found, _suffix = body.rpartition(footer)
    if not found:
        return body
    prefix = prefix.rstrip()
    return "\n".join(
        [
            prefix,
            "",
            _plan_approved_followups_marker(issue_number, plan_hash, mode),
            "-- coding-review-agent-loop",
        ]
    )


def _has_plan_approved_followups_marker(
    comments: Sequence[object],
    *,
    issue_number: int,
    plan_hash: str,
    mode: str,
) -> bool:
    for comment in comments:
        body = getattr(comment, "body", None)
        if not isinstance(body, str):
            continue
        for match in PLAN_APPROVED_FOLLOWUP_MARKER_RE.finditer(body):
            if (
                int(match.group("issue")) == issue_number
                and match.group("plan") == plan_hash
                and match.group("mode").lower() == mode.lower()
            ):
                return True
    return False


def _followup_issue_title(followup: ApprovedFollowup) -> str:
    text = " ".join(_safe_followup_main_text(followup.text).split())
    title = f"Follow up future review note: {text}"
    return title[:120]


def _normalize_followup_key(text: str) -> str:
    text = _followup_main_text(text)
    key = re.sub(r"`([^`]+)`", r"\1", text)
    key = re.sub(r"\*\*([^*]+)\*\*", r"\1", key)
    key = re.sub(r"[_*#>]+", " ", key)
    key = re.sub(r"[^\w\s]+", " ", key.lower())
    return " ".join(key.split())


def _followup_heading_key(text: str) -> str | None:
    text = _followup_main_text(text)
    heading_match = re.match(r"^\s*\*\*(?P<title>[^*]+)\*\*\s*:?", text)
    if heading_match:
        return _normalize_followup_key(heading_match.group("title"))
    first_clause = re.split(r"\s+-\s+|:\s+", text, maxsplit=1)[0]
    if first_clause != text and 3 <= len(first_clause.split()) <= 12:
        return _normalize_followup_key(first_clause)
    return None


def _issue_url(repo: str, issue_number: int) -> str:
    return f"https://github.com/{repo}/issues/{issue_number}"


def _validated_issue_number(found: FoundIssue, *, repo: str) -> int | None:
    number = found.number
    if isinstance(number, bool) or not isinstance(number, int) or number <= 0:
        return None
    url = found.url or ""
    if url:
        match = re.fullmatch(
            r"https?://github\.com/(?P<repo>[^/\s]+/[^/\s]+)/issues/(?P<number>[1-9]\d*)/?",
            url,
            re.I,
        )
        if match is None or match.group("repo").casefold() != repo.casefold():
            return None
        if int(match.group("number")) != number:
            return None
    return number


def _looks_like_followup_tracker(found: FoundIssue) -> bool:
    text = f"{found.title or ''}\n{found.body or ''}".casefold()
    return (
        "future follow-up" in text
        or "follow up future" in text
        or "approved planning" in text
        or "approved review" in text
    )


def _candidate_context_matches(found: FoundIssue, source_context: FollowupSourceContext) -> bool:
    text = f"{found.title or ''}\n{found.body or ''}".casefold()
    identity_values = (
        *source_context.parent_issue_numbers,
        *source_context.related_issue_numbers,
        *source_context.related_pr_numbers,
        source_context.source_number if source_context.source_kind == "pr" else 0,
    )
    if any(f"#{number}" in text for number in identity_values if number):
        return True
    if source_context.source_identity and source_context.source_identity.casefold() in text:
        return True
    return False


def _narrow_existing_candidates(
    proposed: ApprovedFollowup,
    candidates: Sequence[FoundIssue],
    *,
    source_context: FollowupSourceContext,
    max_candidates: int,
) -> tuple[FoundIssue, ...]:
    proposed_ids = _followup_identifier_keys(proposed.text)
    proposed_terms = _followup_topic_terms(proposed.text)
    scored: list[tuple[int, FoundIssue]] = []
    for candidate in candidates:
        if not _looks_like_followup_tracker(candidate):
            continue
        candidate_text = f"{candidate.title or ''}\n{candidate.body or ''}"
        candidate_ids = _followup_identifier_keys(candidate_text)
        candidate_terms = _followup_topic_terms(candidate_text)
        score = 0
        if _candidate_context_matches(candidate, source_context):
            score += 8
        if proposed_ids & candidate_ids:
            score += 5
        score += min(4, len(proposed_terms & candidate_terms))
        if score:
            scored.append((score, candidate))
    scored.sort(key=lambda pair: (-pair[0], pair[1].number or 0))
    return tuple(candidate for _score, candidate in scored[:max_candidates])


def _exact_existing_match(
    proposed: ApprovedFollowup,
    candidate: FoundIssue,
) -> bool:
    proposed_key = _normalize_followup_key(proposed.text)
    if not proposed_key:
        return False
    candidate_text = f"{candidate.title or ''}\n{candidate.body or ''}"
    candidate_key = _normalize_followup_key(candidate_text)
    if proposed_key in candidate_key or candidate_key in proposed_key:
        return True
    proposed_ids = _followup_identifier_keys(proposed.text)
    candidate_ids = _followup_identifier_keys(candidate_text)
    proposed_terms = _followup_topic_terms(proposed.text)
    candidate_terms = _followup_topic_terms(candidate_text)
    return bool(proposed_ids and proposed_ids <= candidate_ids and proposed_terms and proposed_terms <= candidate_terms)


def _search_followup_trackers(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    source_context: FollowupSourceContext,
    followups: Sequence[ApprovedFollowup],
) -> tuple[FoundIssue, ...]:
    queries: list[str] = []
    prefix = f"repo:{source_context.repo} is:issue is:open"

    def add_query(value: str) -> None:
        value = value.strip()
        if value and value not in queries and len(queries) < 5:
            queries.append(value)

    for number in source_context.parent_issue_numbers:
        add_query(f'{prefix} "#{number}" "follow-up"')
    if source_context.source_identity:
        add_query(f'{prefix} "{sanitize_historical_text(source_context.source_identity)}" "follow-up"')
    for number in source_context.related_pr_numbers:
        add_query(f'{prefix} "PR #{number}" "follow-up"')
    for number in source_context.related_issue_numbers:
        add_query(f'{prefix} "#{number}" "follow-up"')
    terms: list[str] = []
    for followup in followups:
        for term in sorted(_followup_topic_terms(followup.text)):
            if term not in terms:
                terms.append(term)
    if terms:
        add_query(f'{prefix} "future follow-up" "{" ".join(terms[:4])}"')
    if not queries:
        add_query(f'{prefix} "future follow-up"')

    found: dict[int, FoundIssue] = {}
    excluded_numbers = {
        source_context.source_number
        if source_context.source_kind == "plan"
        else -1
    }
    for query in queries:
        try:
            results = search_issues(
                runner,
                config=config,
                search=query,
                state="open",
                limit=20,
            )
        except QuotaResetExceededError:
            raise
        except Exception as exc:
            log(config, f"Approved follow-up tracker search unavailable ({exc}); using conservative creation fallback")
            continue
        if len(results) >= 20:
            log(config, f"Approved follow-up tracker search reached limit for {query!r}; results may be truncated")
        for result in results:
            number = _validated_issue_number(result, repo=source_context.repo)
            if number is not None and number not in excluded_numbers:
                found.setdefault(number, result)
        if len(found) >= config.semantic_followup_max_candidates:
            log(config, "Approved follow-up tracker candidate budget reached; truncating aggregate candidates")
            break
    return tuple(found.values())[: config.semantic_followup_max_candidates]


def _semantic_batch_matcher(
    matcher: SemanticDedupeMatcher,
    *,
    source_context: FollowupSourceContext,
) -> Callable[[ApprovedFollowup, tuple[GroupedApprovedFollowup, ...]], SemanticMatch | None]:
    def match(
        proposed: ApprovedFollowup,
        groups: tuple[GroupedApprovedFollowup, ...],
    ) -> SemanticMatch | None:
        # Shared source context is included in the prompt, but the matcher is
        # still bounded to the configured group count and cannot create an
        # identity outside this batch.
        candidates = tuple(
            SemanticCandidate(
                identity=f"group-{index}",
                title=f"Batch follow-up group {index}",
                body=group.text,
            )
            for index, group in enumerate(groups, start=1)
        )
        try:
            return matcher.match(
                proposed=proposed.text,
                candidates=candidates,
                source_context=source_context.render(),
            )
        except QuotaResetExceededError:
            raise
        except BudgetExhausted as exc:
            log(matcher.config, f"Approved follow-up semantic batch budget exhausted ({exc}); retaining deterministic groups")
        except Exception as exc:
            log(matcher.config, f"Approved follow-up semantic batch matcher unavailable ({exc}); retaining deterministic groups")
        return None

    return match


def _try_revalidate_open_issue(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    cache: set[int],
) -> bool:
    if issue_number in cache:
        return True
    try:
        validate_open_issue(runner, config=config, issue_number=issue_number)
    except QuotaResetExceededError:
        raise
    except Exception as exc:
        log(config, f"Follow-up candidate #{issue_number} failed open-state revalidation ({exc}); filing remains enabled")
        return False
    cache.add(issue_number)
    return True


def _format_publication_summary(
    *,
    heading: str,
    publications: Sequence[FollowupPublication],
    deduplicated_count: int,
    skipped_by_cap: int,
) -> str:
    lines = [heading, ""]
    shown_targets: set[str] = set()
    for publication in publications:
        label = _safe_followup_main_text(publication.group.text)
        if publication.status == "created":
            target = publication.issue_url or "Created issue URL unavailable from GitHub CLI output."
            if target not in shown_targets:
                lines.append(f"- {target}")
                shown_targets.add(target)
            lines.append(f"  - Created: {label}")
        elif publication.status == "reused":
            target = publication.issue_url or "existing issue URL unavailable"
            reviewers = ", ".join(sanitize_historical_text(reviewer) for reviewer in publication.group.reviewers)
            lines.append(f"- Reused existing follow-up issue: {target} — {label} ({reviewers})")
            if publication.reason:
                lines.append(f"  - {sanitize_historical_text(publication.reason)}")
        elif publication.status == "uncertain":
            lines.append(f"- Possible duplicate not suppressed; filed: {publication.issue_url or 'URL unavailable'} — {label}")
            if publication.reason:
                lines.append(f"  - Possible duplicate note: {sanitize_historical_text(publication.reason)}")
        else:
            lines.append(f"- Skipped by new-issue cap: {label}")
    lines.extend(
        [
            "",
            f"Reconciliation: {sum(p.status in {'created', 'uncertain'} for p in publications)} filed, "
            f"{deduplicated_count} deduplicated, {skipped_by_cap} skipped by cap.",
            f"Tracker reuse: {sum(p.status == 'reused' for p in publications)} reused; "
            f"{sum(p.status == 'uncertain' for p in publications)} uncertain.",
            "",
            "These were mentioned as future work and did not block merge readiness.",
            "",
            "-- coding-review-agent-loop",
        ]
    )
    if skipped_by_cap:
        lines[-2:-2] = [
            "",
            f"Skipped {skipped_by_cap} additional item(s) to avoid issue noise; reviewers should reserve "
            "this section for substantial independent follow-up work.",
        ]
    return "\n".join(lines)


def _dedupe_approved_followups(followups: Sequence[ApprovedFollowup]) -> list[GroupedApprovedFollowup]:
    return list(reconcile_approved_followups(followups, issue_limit=len(followups) or 0).groups)


def _followup_issue_body(
    pr_number: int,
    followup: GroupedApprovedFollowup,
    *,
    source_context: FollowupSourceContext | None = None,
    possible_duplicate: str | None = None,
) -> str:
    lines = [
        f"Future follow-up from approved review on PR #{pr_number}.",
        "",
    ]
    reviewers = tuple(sanitize_historical_text(reviewer) for reviewer in followup.reviewers)
    if len(reviewers) == 1:
        lines.append(f"Reviewer: {reviewers[0]}")
    else:
        lines.append("Reviewers:")
        lines.extend(f"- {reviewer}" for reviewer in reviewers)
    lines.extend(
        [
            "",
            "Follow-up:",
            f"- {_safe_followup_main_text(followup.text)}",
        ]
    )
    if possible_duplicate:
        lines.extend(
            [
                "",
                "Possible duplicate (not suppressed because semantic confidence was not high):",
                f"- {sanitize_historical_text(possible_duplicate)}",
            ]
        )
    lines.extend(["", "Original reviewer notes:"])
    lines.extend(
        f"- {sanitize_historical_text(item.reviewer)}: {sanitize_historical_text(item.text)}"
        for item in followup.items
    )
    lines.extend(
        [
            "",
            "This was mentioned in an approved review as future work and did not block merge readiness.",
        ]
    )
    return "\n".join(lines)


def _plan_followup_issue_title(followup: PlanGroupedApprovedFollowup) -> str:
    text = " ".join(_safe_followup_main_text(followup.text).split())
    title = f"Follow up future plan-review note: {text}"
    return title[:120]


def _plan_source_label(source: PlanApprovedFollowupSource) -> str:
    parts: list[str] = []
    if source.item_id:
        parts.append(sanitize_historical_text(source.item_id))
    if source.source_round is not None:
        parts.append(f"round {source.source_round}")
    parts.append(sanitize_historical_text(source.reviewer))
    return ", ".join(parts)


def _plan_followup_issue_body(
    *,
    issue_number: int,
    plan_hash: str,
    plan_subject: str,
    followup: PlanGroupedApprovedFollowup,
    source_context: FollowupSourceContext | None = None,
    possible_duplicate: str | None = None,
) -> str:
    reviewers = tuple(sanitize_historical_text(reviewer) for reviewer in followup.reviewers)
    rounds = sorted({source.source_round for source in followup.sources if source.source_round is not None})
    item_ids = [
        sanitize_historical_text(source.item_id)
        for source in followup.sources
        if source.item_id
    ]
    lines = [
        f"Future follow-up from approved planning for issue #{issue_number}.",
        "",
        "Source context:",
        f"- Parent issue: #{issue_number}",
        f"- Approved plan subject: {sanitize_historical_text(plan_subject)}",
        f"- Approved plan hash: {plan_hash}",
    ]
    if rounds:
        lines.append("- Planning round(s): " + ", ".join(str(round_number) for round_number in rounds))
    if len(reviewers) == 1:
        lines.append(f"- Reviewer: {reviewers[0]}")
    else:
        lines.append("- Reviewers: " + ", ".join(reviewers))
    if item_ids:
        lines.append("- Original plan item ID(s): " + ", ".join(item_ids))
    if source_context is not None:
        lines.append(f"- Lookup context: {source_context.render()}")
    lines.extend(
        [
            "",
            "Canonical follow-up:",
            f"- {_safe_followup_main_text(followup.text)}",
            "",
            "Original reviewer notes:",
        ]
    )
    for source in followup.sources:
        lines.append(
            f"- {_plan_source_label(source)}: {sanitize_historical_text(source.text)}"
        )
        for note in source.notes:
            lines.append(f"  - Update from {sanitize_historical_text(note)}")
    if possible_duplicate:
        lines.extend(
            [
                "",
                "Possible duplicate (not suppressed because semantic confidence was not high):",
                f"- {sanitize_historical_text(possible_duplicate)}",
            ]
        )
    lines.extend(
        [
            "",
            "This was approved as future work during planning. It is outside the current "
            "implementation scope and is not a PR-review prior item.",
        ]
    )
    return "\n".join(lines)


def _create_plan_approved_followup_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    plan_hash: str,
    plan_subject: str,
    reconciliation: PlanApprovedFollowupReconciliation,
) -> list[str]:
    issue_urls: list[str] = []
    for followup in reconciliation.selected_groups:
        issue_url = create_issue(
            runner,
            config=config,
            title=_plan_followup_issue_title(followup),
            body=_plan_followup_issue_body(
                issue_number=issue_number,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                followup=followup,
            ),
        )
        issue_urls.append(issue_url or "Created issue URL unavailable from GitHub CLI output.")
    return issue_urls


def _validated_created_issue_url(url: str | None, *, repo: str) -> tuple[int | None, str | None]:
    if not url:
        return None, None
    match = re.fullmatch(
        r"https?://github\.com/(?P<repo>[^/\s]+/[^/\s]+)/issues/(?P<number>[1-9]\d*)/?",
        url.strip(),
        re.I,
    )
    if match is None or match.group("repo").casefold() != repo.casefold():
        return None, None
    number = int(match.group("number"))
    return number, _issue_url(repo, number)


def _find_existing_for_group(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    followup: ApprovedFollowup,
    candidates: Sequence[FoundIssue],
    source_context: FollowupSourceContext,
    matcher: SemanticDedupeMatcher | None,
    revalidated: set[int],
) -> tuple[str, int | None, str | None]:
    """Return (status, issue number, reason) for a safe existing match."""
    valid_candidates = [
        candidate
        for candidate in candidates
        if _validated_issue_number(candidate, repo=source_context.repo) is not None
    ]
    for candidate in valid_candidates:
        number = _validated_issue_number(candidate, repo=source_context.repo)
        assert number is not None
        if _exact_existing_match(followup, candidate) and _try_revalidate_open_issue(
            runner, config=config, issue_number=number, cache=revalidated
        ):
            log(config, f"Suppressed approved follow-up as deterministic duplicate of existing issue #{number}")
            return "reused", number, "Deterministic equivalence with this existing tracker."

    if matcher is None:
        return "new", None, None
    narrowed = _narrow_existing_candidates(
        followup,
        valid_candidates,
        source_context=source_context,
        max_candidates=config.semantic_followup_max_candidates,
    )
    if not narrowed:
        return "new", None, None
    semantic_candidates = tuple(
        SemanticCandidate(identity=_validated_issue_number(candidate, repo=source_context.repo),
                           title=candidate.title or "",
                           body=candidate.body or "")
        for candidate in narrowed
    )
    semantic_candidates = tuple(candidate for candidate in semantic_candidates if candidate.identity is not None)
    try:
        match = matcher.match(
            proposed=followup.text,
            candidates=semantic_candidates,
            source_context=source_context.render(),
        )
    except QuotaResetExceededError:
        raise
    except BudgetExhausted as exc:
        log(config, f"Approved follow-up semantic candidate budget exhausted ({exc}); using deterministic fallback")
        return "new", None, None
    except Exception as exc:
        log(config, f"Approved follow-up semantic matcher unavailable ({exc}); using deterministic fallback")
        return "new", None, None
    if match.duplicate_of is None:
        return "new", None, None
    if not isinstance(match.duplicate_of, int) or isinstance(match.duplicate_of, bool):
        log(config, "Approved follow-up semantic matcher returned a non-issue identity; using fallback")
        return "new", None, None
    candidate = next(
        (candidate for candidate in narrowed if candidate.number == match.duplicate_of),
        None,
    )
    if candidate is None:
        return "new", None, None
    if match.confidence == "high" and _try_revalidate_open_issue(
        runner, config=config, issue_number=match.duplicate_of, cache=revalidated
    ):
        log(
            config,
            f"Suppressed approved follow-up as high-confidence semantic duplicate of existing issue #{match.duplicate_of}: "
            f"{sanitize_historical_text(match.reason)}",
        )
        return "reused", match.duplicate_of, match.reason
    if match.confidence in {"medium", "low"}:
        return "uncertain", None, f"Candidate issue #{match.duplicate_of}: {match.reason}"
    return "new", None, None


def _publish_issue_followup_groups(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    groups: Sequence[GroupedApprovedFollowup],
    source_context: FollowupSourceContext,
    heading: str,
    deduplicated_count: int,
    skipped_by_cap: int,
    plan_subject: str | None = None,
    issue_number: int | None = None,
    plan_hash: str | None = None,
    semantic_matcher: SemanticDedupeMatcher | None = None,
    semantic_transport: SemanticTransport | None = None,
) -> tuple[str, tuple[str, ...]]:
    followups = [ApprovedFollowup(reviewer=group.reviewers[0], text=group.text) for group in groups]
    candidates = _search_followup_trackers(
        runner,
        config=config,
        source_context=source_context,
        followups=followups,
    )
    matcher = semantic_matcher or (
        SemanticDedupeMatcher(runner=runner, config=config, transport=semantic_transport)
        if config.semantic_followup_dedupe and not config.dry_run
        else None
    )
    revalidated: set[int] = set()
    publications: list[FollowupPublication] = []
    created_count = 0
    for group in groups:
        proposed = ApprovedFollowup(reviewer=group.reviewers[0], text=group.text)
        status, existing_number, reason = _find_existing_for_group(
            runner,
            config=config,
            followup=proposed,
            candidates=candidates,
            source_context=source_context,
            matcher=matcher,
            revalidated=revalidated,
        )
        if status == "reused":
            assert existing_number is not None
            publications.append(
                FollowupPublication(
                    group=group,
                    status="reused",
                    issue_number=existing_number,
                    issue_url=_issue_url(source_context.repo, existing_number),
                    reason=reason,
                )
            )
            continue
        if created_count >= MAX_APPROVED_FOLLOWUP_ISSUES:
            publications.append(FollowupPublication(group=group, status="cap-skipped", reason="Only new issues consume the cap."))
            skipped_by_cap += 1
            continue
        body_reason = reason if status == "uncertain" else None
        try:
            if isinstance(group, PlanGroupedApprovedFollowup):
                assert issue_number is not None and plan_hash is not None and plan_subject is not None
                raw_url = create_issue(
                    runner,
                    config=config,
                    title=_plan_followup_issue_title(group),
                    body=_plan_followup_issue_body(
                        issue_number=issue_number,
                        plan_hash=plan_hash,
                        plan_subject=plan_subject,
                        followup=group,
                        source_context=source_context,
                        possible_duplicate=body_reason,
                    ),
                )
            else:
                raw_url = create_issue(
                    runner,
                    config=config,
                    title=_followup_issue_title(proposed),
                    body=_followup_issue_body(
                        source_context.source_number,
                        group,
                        source_context=source_context,
                        possible_duplicate=body_reason,
                    ),
                )
        except QuotaResetExceededError:
            raise
        except Exception:
            # A create-then-interruption window is recoverable when GitHub has
            # indexed the tracker.  Rediscover that exact group before allowing
            # the orchestration failure to escape.
            recovered = _search_followup_trackers(
                runner,
                config=config,
                source_context=source_context,
                followups=(proposed,),
            )
            recovered_match = next(
                (
                    candidate
                    for candidate in recovered
                    if _exact_existing_match(proposed, candidate)
                    and _validated_issue_number(candidate, repo=source_context.repo) is not None
                ),
                None,
            )
            if recovered_match is not None:
                recovered_number = _validated_issue_number(recovered_match, repo=source_context.repo)
                assert recovered_number is not None
                if _try_revalidate_open_issue(
                    runner, config=config, issue_number=recovered_number, cache=revalidated
                ):
                    publications.append(
                        FollowupPublication(
                            group=group,
                            status="reused",
                            issue_number=recovered_number,
                            issue_url=_issue_url(source_context.repo, recovered_number),
                            reason="Recovered an issue created before the publication interruption.",
                        )
                    )
                    continue
            raise
        created_count += 1
        created_number, created_url = _validated_created_issue_url(raw_url, repo=source_context.repo)
        if created_number is not None:
            # Keep newly-created identities in the invocation cache.  This
            # closes the create-then-next-group window without relying on
            # GitHub search indexing to become immediately consistent.
            candidates = (
                *candidates,
                FoundIssue(
                    number=created_number,
                    title=_plan_followup_issue_title(group)
                    if isinstance(group, PlanGroupedApprovedFollowup)
                    else _followup_issue_title(proposed),
                    url=created_url,
                    body=(
                        _plan_followup_issue_body(
                            issue_number=issue_number or source_context.source_number,
                            plan_hash=plan_hash or "unknown",
                            plan_subject=plan_subject or "unknown",
                            followup=group,
                        )
                        if isinstance(group, PlanGroupedApprovedFollowup)
                        else _followup_issue_body(
                            source_context.source_number,
                            group,
                        )
                    ),
                ),
            )
        publications.append(
            FollowupPublication(
                group=group,
                status="uncertain" if status == "uncertain" else "created",
                issue_number=created_number,
                issue_url=created_url,
                reason=reason,
            )
        )
    body = _format_publication_summary(
        heading=heading,
        publications=publications,
        deduplicated_count=deduplicated_count,
        skipped_by_cap=skipped_by_cap,
    )
    return body, tuple(
        publication.issue_url
        for publication in publications
        if publication.issue_url
    )


def _create_approved_followup_issues(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    reconciliation: ApprovedFollowupReconciliation,
) -> list[str]:
    issue_urls: list[str] = []
    for followup in reconciliation.selected_groups:
        issue_url = create_issue(
            runner,
            config=config,
            title=_followup_issue_title(
                ApprovedFollowup(reviewer=followup.reviewers[0], text=followup.text)
            ),
            body=_followup_issue_body(pr_number, followup),
        )
        if issue_url is not None:
            issue_urls.append(issue_url)
    return issue_urls


def _format_created_followup_issue_summary(
    pr_number: int,
    issue_urls: list[str],
    reconciliation: ApprovedFollowupReconciliation,
) -> str:
    unique_issue_urls = list(dict.fromkeys(issue_urls))
    lines = [
        f"Created approved-review future follow-up issues for PR #{pr_number}:",
        "",
    ]
    if unique_issue_urls:
        lines.extend(f"- {issue_url}" for issue_url in unique_issue_urls)
    else:
        lines.append("- Created issue URL unavailable from GitHub CLI output.")
    lines.extend(
        [
            "",
            (
                f"Reconciliation: {len(unique_issue_urls)} filed, "
                f"{reconciliation.deduplicated_count} deduplicated, "
                f"{reconciliation.skipped_by_cap} skipped by cap."
            ),
            "",
            "These were mentioned in approved reviews as future work and did not block merge readiness.",
        ]
    )
    if reconciliation.skipped_by_cap > 0:
        lines.extend(
            [
                "",
                f"Skipped {reconciliation.skipped_by_cap} additional item(s) to avoid issue noise; reviewers should reserve "
                "this section for substantial independent follow-up work.",
            ]
        )
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _publish_approved_followups(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    pr_number: int,
    head_sha: str | None,
    pr_comments: Sequence[object],
    followups: list[ApprovedFollowup],
    source_context: FollowupSourceContext,
    usage_context: object | None = None,
    semantic_transport: SemanticTransport | None = None,
) -> bool:
    if not followups or config.approved_followups == "ignore":
        return False
    mode = (
        "summarize"
        if config.approved_followups in ("summarize", "fix-and-summarize")
        else "issue"
    )
    # Replay detection is deliberately before reconciliation, lookup, and
    # provider activity.  A publish-once marker is the invocation's durable
    # audit record even when every follow-up was reused.
    if _has_approved_followups_marker(
        pr_comments,
        pr_number=pr_number,
        head_sha=head_sha,
        mode=mode,
    ):
        log(
            config,
            f"Approved-review future follow-ups already recorded for PR #{pr_number} at {head_sha or 'unknown'} ({mode})",
        )
        return False

    semantic_matcher = None
    semantic_provider_matcher: SemanticDedupeMatcher | None = None
    if (
        mode == "issue"
        and config.semantic_followup_dedupe
        and not config.dry_run
    ):
        matcher = SemanticDedupeMatcher(
            runner=runner,
            config=config,
            usage_context=usage_context,
            transport=semantic_transport,
        )
        semantic_provider_matcher = matcher
        semantic_matcher = _semantic_batch_matcher(matcher, source_context=source_context)
    reconciliation = reconcile_approved_followups(
        followups,
        issue_limit=(
            MAX_APPROVED_FOLLOWUP_ISSUES
            if mode == "summarize"
            else len(followups) or MAX_APPROVED_FOLLOWUP_ISSUES
        ),
        semantic_matcher=semantic_matcher,
    )
    groups = reconciliation.selected_groups if mode == "summarize" else reconciliation.groups
    if not groups:
        return False
    log(
        config,
        f"Approved-review future follow-up reconciliation for PR #{pr_number}: "
        f"{len(reconciliation.selected_groups)} selected, "
        f"{reconciliation.deduplicated_count} deduplicated, "
        f"{reconciliation.skipped_by_cap} skipped by cap",
    )

    if mode == "summarize":
        body = _format_approved_followup_summary(pr_number, reconciliation)
        body = _append_approved_followups_marker(
            body,
            pr_number=pr_number,
            head_sha=head_sha,
            mode=mode,
        )
        post_pr_comment(
            runner,
            config=config,
            pr_number=pr_number,
            body=TrustedBody.canonical(body, expected_tokens=("AGENT_APPROVED_FOLLOWUPS",)),
        )
        return True

    if mode == "issue":
        publication_body, _issue_urls = _publish_issue_followup_groups(
            runner,
            config=config,
            groups=groups,
            source_context=source_context,
            heading=f"Created approved-review future follow-up issues for PR #{pr_number}:",
            deduplicated_count=reconciliation.deduplicated_count,
            skipped_by_cap=reconciliation.skipped_by_cap,
            semantic_matcher=semantic_provider_matcher,
            semantic_transport=semantic_transport,
        )
        if not _issue_urls:
            # Preserve the historical fail-safe when GitHub did not return a
            # usable identity for any newly-created issue.  Reused trackers
            # are included in _issue_urls and therefore still publish the
            # durable audit record.
            return False
        body = _append_approved_followups_marker(
            publication_body,
            pr_number=pr_number,
            head_sha=head_sha,
            mode=mode,
        )
        post_pr_comment(
            runner,
            config=config,
            pr_number=pr_number,
            body=TrustedBody.canonical(body, expected_tokens=("AGENT_APPROVED_FOLLOWUPS",)),
        )
        return True
    return False


def _format_same_pr_followups(followups: Sequence[ApprovedFollowup]) -> str:
    lines: list[str] = []
    for followup in followups:
        lines.append(
            f"{sanitize_historical_text(followup.reviewer)} same-PR follow-up:"
        )
        lines.append(f"- {sanitize_historical_text(followup.text)}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_plan_approval_summary_with_followups(
    issue_number: int,
    approved_plan: str,
    *,
    reconciliation: PlanApprovedFollowupReconciliation | None = None,
    issue_urls: Sequence[str] = (),
    filing_enabled: bool = False,
    publication_details: str | None = None,
) -> str:
    lines = [
        f"Planning complete for issue #{issue_number}.",
        "",
        "Outcome: implement",
        "",
        "Approved plan:",
        "",
        sanitize_historical_text(approved_plan),
    ]
    if reconciliation is not None and reconciliation.selected_groups:
        if filing_enabled:
            lines.extend(["", "Filed future follow-up issues:", ""])
            unique_issue_urls = list(dict.fromkeys(issue_urls))
            lines.extend(f"- {issue_url}" for issue_url in unique_issue_urls)
            if publication_details:
                detail_lines = publication_details.splitlines()
                if detail_lines and detail_lines[0].endswith(":"):
                    detail_lines = detail_lines[1:]
                detail_lines = [line for line in detail_lines if line != "-- coding-review-agent-loop"]
                if detail_lines:
                    lines.extend(["", "Publication details:", *detail_lines])
            lines.extend(
                [
                    "",
                    (
                        f"Reconciliation: {len(reconciliation.selected_groups)} filed, "
                        f"{reconciliation.deduplicated_count} deduplicated, "
                        f"{reconciliation.skipped_by_cap} skipped by cap."
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "Approved plan future follow-ups:",
                    "",
                    "These are summarized only; they are NOT filed as GitHub issues. Rerun with "
                    "`--approved-followups issue` (or `fix-and-issue`) to file them, or file them "
                    "manually.",
                    "",
                ]
            )
            for followup in reconciliation.selected_groups:
                reviewers = ", ".join(
                    sanitize_historical_text(reviewer) for reviewer in followup.reviewers
                )
                lines.append(f"- {_safe_followup_main_text(followup.text)} ({reviewers})")
                for source in followup.sources:
                    for note in source.notes:
                        lines.append(
                            f"  - Update from {sanitize_historical_text(note)}"
                        )
            lines.extend(
                [
                    "",
                    (
                        f"Reconciliation: {len(reconciliation.selected_groups)} summarized, "
                        f"{reconciliation.deduplicated_count} deduplicated, "
                        f"{reconciliation.skipped_by_cap} skipped by cap."
                    ),
                ]
            )
        lines.extend(
            [
                "",
                "These planning-stage future follow-ups are future work outside the current "
                "implementation scope. They are not carried into PR review and their plan "
                "item IDs are not PR prior review items.",
            ]
        )
    lines.extend(["", "-- coding-review-agent-loop"])
    return "\n".join(lines)


def _publish_plan_approved_followups(
    runner: Runner,
    *,
    config: AgentLoopConfig,
    issue_number: int,
    approved_plan: str,
    plan_hash: str,
    plan_subject: str,
    issue_comments: Sequence[object],
    sources: Sequence[PlanApprovedFollowupSource],
    source_context: FollowupSourceContext,
    allow_issue_filing: bool = True,
    usage_context: object | None = None,
    semantic_transport: SemanticTransport | None = None,
) -> bool:
    filing_enabled = allow_issue_filing and config.approved_followups in ("issue", "fix-and-issue")
    mode = "issue" if filing_enabled else "summarize"
    if _has_plan_approved_followups_marker(
        issue_comments,
        issue_number=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    ):
        log(
            config,
            f"Planning future follow-ups already recorded for issue #{issue_number} "
            f"plan {plan_hash} ({mode})",
        )
        return False

    semantic_matcher = None
    semantic_provider_matcher: SemanticDedupeMatcher | None = None
    if filing_enabled and config.semantic_followup_dedupe and not config.dry_run:
        semantic_provider_matcher = SemanticDedupeMatcher(
            runner=runner,
            config=config,
            usage_context=usage_context,
            transport=semantic_transport,
        )
        semantic_matcher = _semantic_batch_matcher(
            semantic_provider_matcher,
            source_context=source_context,
        )
    reconciliation = (
        reconcile_plan_approved_followups(
            sources,
            issue_limit=(
                len(sources) or MAX_APPROVED_FOLLOWUP_ISSUES
                if filing_enabled
                else MAX_APPROVED_FOLLOWUP_ISSUES
            ),
            semantic_matcher=semantic_matcher,
        )
        if sources
        else None
    )
    issue_urls: list[str] = []
    publication_details: str | None = None
    if reconciliation is not None and reconciliation.selected_groups:
        log(
            config,
            f"Planning future follow-up reconciliation for issue #{issue_number}: "
            f"{len(reconciliation.selected_groups)} selected, "
            f"{reconciliation.deduplicated_count} deduplicated, "
            f"{reconciliation.skipped_by_cap} skipped by cap",
        )
        if filing_enabled:
            publication_details, created_urls = _publish_issue_followup_groups(
                runner,
                config=config,
                groups=reconciliation.groups,
                source_context=source_context,
                heading="Created approved-plan future follow-up issues:",
                deduplicated_count=reconciliation.deduplicated_count,
                skipped_by_cap=reconciliation.skipped_by_cap,
                issue_number=issue_number,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                semantic_matcher=semantic_provider_matcher,
                semantic_transport=semantic_transport,
            )
            issue_urls = list(created_urls)

    # The approved plan is a re-rendered historical GitHub artifact.  Its
    # encoded plan metadata may contain durable records, but those records are
    # not newly authorized by this follow-up comment.
    body = _format_plan_approval_summary_with_followups(
        issue_number,
        sanitize_historical_text(approved_plan),
        reconciliation=reconciliation,
        issue_urls=issue_urls,
        filing_enabled=filing_enabled,
        publication_details=publication_details,
    )
    body = _append_plan_approved_followups_marker(
        body,
        issue_number=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    )
    post_issue_comment(
        runner,
        config=config,
        issue_number=issue_number,
        body=TrustedBody.canonical(body, expected_tokens=("AGENT_PLAN_APPROVED_FOLLOWUPS",)),
    )
    return True
