"""Approved follow-up publishing and formatting helpers."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .config import AgentLoopConfig
from .github import create_issue, post_issue_comment, post_pr_comment
from .logging import log
from .protocol import ApprovedFollowup, UnresolvedReviewItem
from .runner import Runner

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
) -> PlanApprovedFollowupReconciliation:
    source_by_projection_id: dict[int, PlanApprovedFollowupSource] = {}
    projections: list[ApprovedFollowup] = []
    for source in sources:
        projection = _approved_followup_from_plan_source(source)
        projections.append(projection)
        source_by_projection_id[id(projection)] = source

    reconciliation = reconcile_approved_followups(projections, issue_limit=issue_limit)

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
        reviewers = ", ".join(followup.reviewers)
        lines.append(f"- {_followup_main_text(followup.text)} ({reviewers})")
        for item in followup.items:
            for note in _followup_update_notes(item.text):
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
    text = " ".join(_followup_main_text(followup.text).split())
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


def _dedupe_approved_followups(followups: Sequence[ApprovedFollowup]) -> list[GroupedApprovedFollowup]:
    return list(reconcile_approved_followups(followups, issue_limit=len(followups) or 0).groups)


def _followup_issue_body(pr_number: int, followup: GroupedApprovedFollowup) -> str:
    lines = [
        f"Future follow-up from approved review on PR #{pr_number}.",
        "",
    ]
    reviewers = followup.reviewers
    if len(reviewers) == 1:
        lines.append(f"Reviewer: {reviewers[0]}")
    else:
        lines.append("Reviewers:")
        lines.extend(f"- {reviewer}" for reviewer in reviewers)
    lines.extend(
        [
            "",
            "Follow-up:",
            f"- {_followup_main_text(followup.text)}",
        ]
    )
    lines.extend(["", "Original reviewer notes:"])
    lines.extend(f"- {item.reviewer}: {item.text}" for item in followup.items)
    lines.extend(
        [
            "",
            "This was mentioned in an approved review as future work and did not block merge readiness.",
        ]
    )
    return "\n".join(lines)


def _plan_followup_issue_title(followup: PlanGroupedApprovedFollowup) -> str:
    text = " ".join(_followup_main_text(followup.text).split())
    title = f"Follow up future plan-review note: {text}"
    return title[:120]


def _plan_source_label(source: PlanApprovedFollowupSource) -> str:
    parts: list[str] = []
    if source.item_id:
        parts.append(source.item_id)
    if source.source_round is not None:
        parts.append(f"round {source.source_round}")
    parts.append(source.reviewer)
    return ", ".join(parts)


def _plan_followup_issue_body(
    *,
    issue_number: int,
    plan_hash: str,
    plan_subject: str,
    followup: PlanGroupedApprovedFollowup,
) -> str:
    reviewers = followup.reviewers
    rounds = sorted({source.source_round for source in followup.sources if source.source_round is not None})
    item_ids = [source.item_id for source in followup.sources if source.item_id]
    lines = [
        f"Future follow-up from approved planning for issue #{issue_number}.",
        "",
        "Source context:",
        f"- Parent issue: #{issue_number}",
        f"- Approved plan subject: {plan_subject}",
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
    lines.extend(
        [
            "",
            "Canonical follow-up:",
            f"- {_followup_main_text(followup.text)}",
            "",
            "Original reviewer notes:",
        ]
    )
    for source in followup.sources:
        lines.append(f"- {_plan_source_label(source)}: {source.text}")
        for note in source.notes:
            lines.append(f"  - Update from {note}")
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
) -> bool:
    if not followups or config.approved_followups == "ignore":
        return False
    reconciliation = reconcile_approved_followups(
        followups,
        issue_limit=MAX_APPROVED_FOLLOWUP_ISSUES,
    )
    if not reconciliation.selected_groups:
        return False
    log(
        config,
        f"Approved-review future follow-up reconciliation for PR #{pr_number}: "
        f"{len(reconciliation.selected_groups)} selected, "
        f"{reconciliation.deduplicated_count} deduplicated, "
        f"{reconciliation.skipped_by_cap} skipped by cap",
    )

    if config.approved_followups in ("summarize", "fix-and-summarize"):
        mode = "summarize"
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
        body = _format_approved_followup_summary(pr_number, reconciliation)
        body = _append_approved_followups_marker(
            body,
            pr_number=pr_number,
            head_sha=head_sha,
            mode=mode,
        )
        post_pr_comment(runner, config=config, pr_number=pr_number, body=body)
        return True

    if config.approved_followups in ("issue", "fix-and-issue"):
        mode = "issue"
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
        issue_urls = _create_approved_followup_issues(
            runner,
            config=config,
            pr_number=pr_number,
            reconciliation=reconciliation,
        )
        if issue_urls:
            body = _format_created_followup_issue_summary(pr_number, issue_urls, reconciliation)
            body = _append_approved_followups_marker(
                body,
                pr_number=pr_number,
                head_sha=head_sha,
                mode=mode,
            )
            post_pr_comment(runner, config=config, pr_number=pr_number, body=body)
            return True
    return False


def _format_same_pr_followups(followups: Sequence[ApprovedFollowup]) -> str:
    lines: list[str] = []
    for followup in followups:
        lines.append(f"{followup.reviewer} same-PR follow-up:")
        lines.append(f"- {followup.text}")
        lines.append("")
    return "\n".join(lines).strip()


def _format_plan_approval_summary_with_followups(
    issue_number: int,
    approved_plan: str,
    *,
    reconciliation: PlanApprovedFollowupReconciliation | None = None,
    issue_urls: Sequence[str] = (),
    filing_enabled: bool = False,
) -> str:
    lines = [
        f"Planning complete for issue #{issue_number}.",
        "",
        "Outcome: implement",
        "",
        "Approved plan:",
        "",
        approved_plan,
    ]
    if reconciliation is not None and reconciliation.selected_groups:
        if filing_enabled:
            lines.extend(["", "Filed future follow-up issues:", ""])
            unique_issue_urls = list(dict.fromkeys(issue_urls))
            lines.extend(f"- {issue_url}" for issue_url in unique_issue_urls)
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
                reviewers = ", ".join(followup.reviewers)
                lines.append(f"- {_followup_main_text(followup.text)} ({reviewers})")
                for source in followup.sources:
                    for note in source.notes:
                        lines.append(f"  - Update from {note}")
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
    allow_issue_filing: bool = True,
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

    reconciliation = (
        reconcile_plan_approved_followups(sources, issue_limit=MAX_APPROVED_FOLLOWUP_ISSUES)
        if sources
        else None
    )
    issue_urls: list[str] = []
    if reconciliation is not None and reconciliation.selected_groups:
        log(
            config,
            f"Planning future follow-up reconciliation for issue #{issue_number}: "
            f"{len(reconciliation.selected_groups)} selected, "
            f"{reconciliation.deduplicated_count} deduplicated, "
            f"{reconciliation.skipped_by_cap} skipped by cap",
        )
        if filing_enabled:
            issue_urls = _create_plan_approved_followup_issues(
                runner,
                config=config,
                issue_number=issue_number,
                plan_hash=plan_hash,
                plan_subject=plan_subject,
                reconciliation=reconciliation,
            )

    body = _format_plan_approval_summary_with_followups(
        issue_number,
        approved_plan,
        reconciliation=reconciliation,
        issue_urls=issue_urls,
        filing_enabled=filing_enabled,
    )
    body = _append_plan_approved_followups_marker(
        body,
        issue_number=issue_number,
        plan_hash=plan_hash,
        mode=mode,
    )
    post_issue_comment(runner, config=config, issue_number=issue_number, body=body)
    return True
