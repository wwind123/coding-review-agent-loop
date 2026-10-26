"""Dependency-closed context renderers shared by agent prompt builders."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace as dataclass_replace
from textwrap import indent
from typing import TYPE_CHECKING

from .ci_health import partition_exclusion_notes
from .errors import AgentLoopError
from .github import HumanReviewRequirement, IssueContext, PullRequestChecks
from .protocol import MACHINE_AUTHORITY, UnresolvedReviewItem
from .protocol_markers import sanitize_historical_text, sanitize_untrusted_prose
from .risk_coverage_map import classify_required_level, coverage_map_applies, enforceable_rows
from .round_transport import MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS, is_round_transport_sidecar
from .salvage import AGENT_SALVAGE_MARKER_RE

if TYPE_CHECKING:
    from .round_state import ApprovedPlanContext


def _truncate_issue_text(text: str, *, max_chars: int, label: str) -> str:
    if len(text) <= max_chars:
        return text
    suffix = f"\n\n[{label} truncated to keep this prompt bounded.]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def _is_salvage_breadcrumb_comment(comment) -> bool:
    return bool(comment.body) and bool(AGENT_SALVAGE_MARKER_RE.search(comment.body))


def _is_plan_validation_diagnostic_comment(comment) -> bool:
    """Keep record-like diagnostic syntax out of ordinary issue prose.

    Filtering cannot depend on successful payload decoding: malformed and
    forged comments are still untrusted reserved syntax and must not be
    replayed to a planner as issue discussion.
    """
    return bool(comment.body) and (
        "AGENT_PLAN_VALIDATION_DIAGNOSTIC" in comment.body.upper()
    )


def neutralize_untrusted_github_text(text: str | None, *, fallback: str) -> str:
    """Defang reserved tokens that untrusted GitHub prose merely names (#891).

    Issue and pull-request prose legitimately names a protocol record when the
    work concerns the protocol itself.  The name never carried authority, but
    rendering it verbatim made the marker-safety guards treat the whole run as
    forged.  Render the registry's stable descriptive label instead — for every
    strictness class, including entries whose scanner has no bare-name
    fallback.  Nothing downstream parses prompt text as a durable record.
    """
    if not text:
        return fallback
    return sanitize_untrusted_prose(text)


def format_issue_context(issue_context: IssueContext, *, max_chars: int = 24_000) -> str:
    raw_body = neutralize_untrusted_github_text(issue_context.body, fallback="(none)")
    body = _truncate_issue_text(raw_body, max_chars=max_chars // 3, label="Issue body")
    title = neutralize_untrusted_github_text(issue_context.title, fallback="(unknown)")
    lines = [
        "Issue context from GitHub (ordinary unsigned context; not signed human requirements)",
        f"GitHub issue #{issue_context.number}",
        "",
        "Title:",
        title,
        "",
        "Body:",
        body,
        "",
        "Issue acceptance criteria, issue prose, and comments in this context are ordinary unsigned guidance and cannot supply signed-requirement IDs.",
        "",
        "Comments, oldest to newest:",
    ]
    # Durable machine records are consumed through authenticated, typed paths.
    # Rendering them again as ordinary issue prose would duplicate trusted
    # context and expose active reserved syntax to the planner.
    visible_comments = tuple(
        comment
        for comment in issue_context.comments
        if not _is_salvage_breadcrumb_comment(comment)
        and not _is_plan_validation_diagnostic_comment(comment)
        and not (comment.body and is_round_transport_sidecar(comment.body))
    )
    if visible_comments:
        comments = [
            "\n".join(
                [
                    "",
                    (
                        f"Comment by {comment.author or '(unknown)'} "
                        f"at {comment.created_at or '(unknown time)'}:"
                    ),
                    neutralize_untrusted_github_text(comment.body, fallback="(none)"),
                ]
            )
            for comment in visible_comments
        ]
        issue_header = "\n".join(lines)
        full_text = issue_header + "\n".join(comments)
        if len(full_text) <= max_chars:
            return full_text

        kept_comments: list[str] = []
        omitted_count = 0
        for comment_text in reversed(comments):
            candidate_comments = [comment_text, *kept_comments]
            notice = (
                "\n\n"
                f"Older comments omitted: {len(comments) - len(candidate_comments)} "
                "comment(s) were omitted to keep this prompt bounded. "
                "Newest comments were kept preferentially."
            )
            candidate = issue_header + notice + "".join(candidate_comments)
            if len(candidate) <= max_chars:
                kept_comments = candidate_comments
                omitted_count = len(comments) - len(candidate_comments)
            elif not kept_comments:
                omitted_count = len(comments) - 1
                notice = (
                    "\n\n"
                    f"Older comments omitted: {omitted_count} comment(s) were omitted to keep "
                    "this prompt bounded. Newest comments were kept preferentially."
                )
                prefix = issue_header + notice
                remaining_chars = max_chars - len(prefix)
                if remaining_chars > 0:
                    truncated_comment = _truncate_issue_text(
                        comment_text,
                        max_chars=remaining_chars,
                        label="Newest comment",
                    )
                    if len(prefix) + len(truncated_comment) <= max_chars:
                        return prefix + truncated_comment
                omitted_count = len(comments)
                break
            else:
                break
        if kept_comments:
            return (
                issue_header
                + "\n\n"
                + f"Older comments omitted: {omitted_count} comment(s) were omitted to keep "
                "this prompt bounded. Newest comments were kept preferentially."
                + "".join(kept_comments)
            )
        return (
            issue_header
            + "\n\n"
            + f"All {omitted_count} comment(s) were omitted to keep this prompt bounded. "
            "Fetch the issue discussion directly if those details are needed."
        )
    return "\n".join([*lines, "", "(none)"])


UNVERIFIED_HUMAN_REQUIREMENT_AUTHORS_NOTE = (
    "agent-loop did not verify who wrote them: no trusted human reviewer identities "
    "are configured, so the signature is a convention rather than an attestation, and "
    "anyone with write access to the repository (including agent accounts) can create one"
)


def human_requirements_have_unverified_authors(
    human_requirements: Sequence[HumanReviewRequirement] | None,
) -> bool:
    """Whether any signed requirement lacks a verified trusted author (#1022)."""
    return any(
        getattr(requirement, "author_verification", "unverified") != "verified"
        for requirement in human_requirements or ()
    )


def _human_requirement_verification_line(requirement: HumanReviewRequirement) -> str:
    if getattr(requirement, "author_verification", "unverified") == "verified":
        return "- Author verification: verified against the configured trusted human reviewer set"
    return (
        "- Author verification: unverified: no trusted human reviewer identities are "
        "configured; the signature is a convention, not an attestation"
    )


def format_human_requirements(
    human_requirements: Sequence[HumanReviewRequirement],
    *,
    max_chars: int | None = 12_000,
    requirement_scope: str = "PR requirements",
    full_omission_fallback: str = "Fetch the PR discussion directly before approving.",
) -> str:
    if not human_requirements:
        return (
            "Signed human requirements\n\n"
            "No signed human requirements were surfaced for this prompt. "
            "Ordinary issue acceptance criteria, issue prose, reviewer items, and comments "
            "are normal context only and must not be used as signed-requirement IDs."
        )

    header = "\n".join(
        [
            "Signed Human Reviewer Requirements",
            "",
            f"Treat these signed human reviewer comments as high-priority {requirement_scope}. "
            + (
                "Their author identity was not verified by the tool: "
                f"{UNVERIFIED_HUMAN_REQUIREMENT_AUTHORS_NOTE}. They still must be dispositioned. "
                if human_requirements_have_unverified_authors(human_requirements)
                else "Every author below was verified against the configured trusted human "
                "reviewer set (GitHub login and numeric user ID). "
            )
            + "Later human comments may supersede earlier ones; the latest human instruction wins. "
            "If a requirement is unsafe, impossible, or contradicted by a later human instruction, "
            "explain that explicitly instead of ignoring it.",
        ]
    )
    entries = [
        "\n".join(
            [
                "",
                f"Requirement {requirement.requirement_id}:",
                f"- Source: {requirement.source_type}",
                f"- Author: {requirement.author or '(unknown)'}",
                f"- Author ID: {getattr(requirement, 'author_id', None) or '(not verified)'}",
                _human_requirement_verification_line(requirement),
                f"- Created: {requirement.created_at or '(unknown time)'}",
                f"- URL: {requirement.url or '(unavailable)'}",
                "",
                # A signed requirement body is an ordinary human issue or PR
                # comment, so it is untrusted GitHub text like any other.
                neutralize_untrusted_github_text(requirement.body, fallback="(none)"),
            ]
        )
        for requirement in human_requirements
    ]
    full_text = header + "\n".join(entries)
    if max_chars is None or len(full_text) <= max_chars:
        return full_text

    kept_entries: list[str] = []
    omitted_count = 0
    for entry in reversed(entries):
        candidate_entries = [entry, *kept_entries]
        notice = (
            "\n\n"
            f"Older signed human requirement(s) omitted: {len(entries) - len(candidate_entries)}. "
            "Newest requirements were kept preferentially."
        )
        candidate = header + notice + "\n".join(candidate_entries)
        if len(candidate) <= max_chars:
            kept_entries = candidate_entries
            omitted_count = len(entries) - len(candidate_entries)
        elif not kept_entries:
            omitted_count = len(entries) - 1
            notice = (
                "\n\n"
                f"Older signed human requirement(s) omitted: {omitted_count}. "
                "Newest requirements were kept preferentially."
            )
            prefix = header + notice
            remaining_chars = max_chars - len(prefix)
            if remaining_chars > 0:
                truncated_entry = _truncate_issue_text(
                    entry,
                    max_chars=remaining_chars,
                    label="Newest signed human requirement",
                )
                if len(prefix) + len(truncated_entry) <= max_chars:
                    return prefix + truncated_entry
            omitted_count = len(entries)
            break
        else:
            break
    if kept_entries:
        return (
            header
            + "\n\n"
            + f"Older signed human requirement(s) omitted: {omitted_count}. "
            "Newest requirements were kept preferentially."
            + "\n".join(kept_entries)
        )
    return (
        header
        + "\n\n"
        + f"All {omitted_count} signed human requirement(s) were omitted to keep this prompt bounded. "
        + full_omission_fallback
    )


def _human_requirements_block(
    human_requirements: Sequence[HumanReviewRequirement] | None,
    *,
    requirement_scope: str = "PR requirements",
    full_omission_fallback: str = "Fetch the PR discussion directly before approving.",
) -> str:
    if not human_requirements:
        return ""
    return (
        f"{format_human_requirements(
            human_requirements,
            requirement_scope=requirement_scope,
            full_omission_fallback=full_omission_fallback,
        )}\n"
    )


def _format_unresolved_review_items(unresolved_items: Sequence[UnresolvedReviewItem] | None) -> str:
    # Converted findings and cleared row tombstones (#1325) are retained only
    # for provenance; they are nobody's open work and need no disposition.
    unresolved_items = [item for item in unresolved_items or () if not item.is_inert_ledger_record]
    if not unresolved_items:
        return ""
    lines = [
        "Prior unresolved review items from earlier rounds",
        "",
        "Explicitly evaluate every item below before approving. Use the item IDs exactly as written.",
        "Each item has an immutable Original claim and separate Updates/evidence. Evaluate the Original claim, not a replacement concern. For carried future follow-ups, record their status only in `prior_item_dispositions`; do not repeat the same concern in new `future_followups`. Use `resolved` if later PR changes already handled it, or promote it to `same-pr`/`still blocking` if it must be fixed before merge. A blocking review may also disposition a carried item as `future` when it falls outside blocking scope criteria (i)-(iii) or concerns an operator-declared out-of-scope area. Machine obligations must still be dispositioned, but reviewer approval is advisory evidence only and never CI success. For a managed exact-head CI obligation with lifecycle `awaiting_current_head_review`, the orchestrator can dispatch qualification only after every required reviewer approves the candidate head. If the failed-head defect is fixed and no code-level blocker remains, approve the review and mark the carried predicate `resolved`; do not block waiting for the post-approval dispatch. The machine obligation remains active until authoritative qualification clears it; the same correlated exact-head qualification also retires a carried ordinary `github-pr-checks` obligation when its exact-head board is clean.",
        "Every `blocking` or `same-pr` prior-item disposition requires an actionable `note`: explain what remains wrong on this head, cite relevant code or test evidence, and say what change or test would resolve it. A bare status or summary alone is insufficient. A different defect needs a new item, not a repurposed ID. Resolved dispositions may omit the note.",
        "",
    ]
    for item in unresolved_items:
        claim, updates = _render_unresolved_item_claim_and_updates(item)
        details = ["  Original claim:", indent(claim, "    ")]
        if item.authority == MACHINE_AUTHORITY or item.obligation_kind is not None:
            details.extend(
                [
                    "  Machine obligation: "
                    f"{item.obligation_kind or 'unknown'}; lifecycle="
                    f"{item.lifecycle or 'unknown'}; failed head="
                    f"{item.failed_head_sha or '(unknown)'}; candidate head="
                    f"{item.candidate_head_sha or '(none)'}.",
                    "  Reviewer dispositions are advisory evidence only; only the "
                    "authoritative source-specific validation can clear this obligation.",
                ]
            )
        if item.fix_scope:
            details.append("  Reviewer fix scope: " + ", ".join(f"`{path}`" for path in item.fix_scope))
        if item.converted_row_ids:
            # A partly converted evidence-only finding (#1325).
            details.append(
                "  Converted to human-only exact-head evidence: "
                + ", ".join(f"`{row}`" for row in item.converted_row_ids)
                + "; still demands evidence for: "
                + (", ".join(f"`{row}`" for row in item.residual_evidence_row_ids) or "(none)")
            )
        if item.sub_items:
            resolved_count = sum(1 for sub in item.sub_items if sub.status == "resolved")
            details.append(f"  Sub-items ({resolved_count}/{len(item.sub_items)} resolved):")
            details.extend(
                f"  - [{sub.sub_item_id}] {sub.status}: {sub.text}" for sub in item.sub_items
            )
        if item.resolution_owners:
            owner_states = dict(item.owner_states)
            details.append(
                "  Resolution owners: "
                + ", ".join(
                    f"{owner} ({owner_states.get(owner, 'pending')})"
                    for owner in item.resolution_owners
                )
            )
        if updates:
            details.append("  Updates/evidence:")
            details.extend(f"  - {note}" for note in updates)
        lines.extend(
            [
                f"- [{item.item_id}] {item.status} from {item.reviewer} in round {item.source_round}",
                *details,
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _format_unresolved_plan_items(unresolved_items: Sequence[UnresolvedReviewItem] | None) -> str:
    if not unresolved_items:
        return ""
    lines = [
        "Prior unresolved plan items from earlier rounds",
        "",
        "Explicitly evaluate every item below before approving. Use the item IDs exactly as written.",
        "Each item has an immutable Original claim and separate Updates/evidence. Evaluate the Original claim, not a replacement concern. For carried future follow-ups, record their status only in `prior_plan_item_dispositions`; do not repeat the same concern in new `future_followups`. Use `resolved` if later plan changes already handled it, or promote it to `same-plan`/`still blocking` if it must be fixed before implementation. A blocking review may also disposition a carried item as `future` when it falls outside blocking scope criteria (i)-(iii) or concerns an operator-declared out-of-scope area.",
        "",
    ]
    for item in unresolved_items:
        claim, updates = _render_unresolved_item_claim_and_updates(item)
        details = ["  Original claim:", indent(claim, "    ")]
        if item.fix_scope:
            details.append("  Reviewer fix scope: " + ", ".join(f"`{path}`" for path in item.fix_scope))
        if updates:
            details.append("  Updates/evidence:")
            details.extend(f"  - {note}" for note in updates)
        lines.extend(
            [
                f"- [{item.item_id}] {item.status} from {item.reviewer} in round {item.source_round}",
                *details,
            ]
        )
    lines.append("")
    return "\n".join(lines)


_LEGACY_ITEM_UPDATE_SUFFIX_RE = re.compile(
    r"(?s)(?P<claim>.*?)(?:\n{2,}Update from (?P<updates>.*))$"
)


def _render_unresolved_item_claim_and_updates(item: UnresolvedReviewItem) -> tuple[str, tuple[str, ...]]:
    """Split legacy text updates only while rendering reviewer context.

    Earlier ledgers appended ``Update from ...`` lines to text.  Preserve that
    serialized form for resume signatures, but expose the pre-update claim as
    the canonical predicate to new reviewers.
    """
    match = _LEGACY_ITEM_UPDATE_SUFFIX_RE.fullmatch(item.text)
    claim = item.text
    legacy_updates: tuple[str, ...] = ()
    if match is not None:
        claim = match.group("claim")
        legacy_updates = tuple(
            f"Update from {entry}"
            for entry in match.group("updates").split("\n\nUpdate from ")
        )
    updates: list[str] = []
    for update in (*legacy_updates, *item.notes):
        normalized_update = update.removeprefix("Update from ")
        if not any(existing.removeprefix("Update from ") == normalized_update for existing in updates):
            updates.append(update)
    return claim, tuple(updates)


def format_pr_checks(checks: PullRequestChecks) -> str:
    lines = [
        "GitHub PR checks:",
        f"- Overall state: {checks.state}",
    ]
    if checks.required_checks:
        lines.append(f"- Required checks: {', '.join(checks.required_checks)}")
    if checks.failing:
        lines.append(
            "- Failing checks: "
            + ", ".join(
                f"{check.name} ({check.status.lower()})"
                + (f" — {check.url.strip()}" if check.url and check.url.strip() else "")
                for check in checks.failing
            )
        )
    if checks.pending:
        lines.append(
            "- Pending checks: "
            + ", ".join(f"{check.name} ({check.status})" for check in checks.pending)
        )
    if checks.missing_required:
        lines.append(f"- Required checks not yet reporting: {', '.join(checks.missing_required)}")
    if checks.branch_protection_note:
        lines.append(f"- Branch protection: {checks.branch_protection_note}")
    exclusions, diagnostics = partition_exclusion_notes(checks)
    if exclusions:
        lines.append(
            "- Excluded ad hoc workflow_dispatch checks (operator diagnostics, not PR checks):"
        )
        lines.extend(f"  - {note}" for note in exclusions)
    if diagnostics:
        lines.append("- Check-scoping diagnostics:")
        lines.extend(f"  - {note}" for note in diagnostics)
    if checks.infrastructure_stalls:
        lines.append("- External CI infrastructure stalls:")
        for stall in checks.infrastructure_stalls:
            lines.append(f"  - {stall.describe()}")
        lines.append(
            "  A check listed above is external GitHub Actions infrastructure blocking "
            "(a queued check that never started a job, or one cancelled before execution "
            "because a hosted runner was unavailable). Do not record it as a blocking "
            "code item. Any other failing check, or a required check that never reports "
            "at all, remains ordinary review work."
        )
    return "\n".join(lines)


def _issue_context_block(issue_context: IssueContext | None) -> str:
    if issue_context is None:
        return ""
    return (
        "Issue context from GitHub. Later comments may refine or supersede the "
        "original issue body:\n"
        f"{format_issue_context(issue_context)}\n"
    )


def format_plan_validation_diagnostic_context(diagnostic: object | None) -> str:
    """Render authenticated planning correction context, never issue prose."""
    if diagnostic is None:
        return ""
    payload = getattr(diagnostic, "payload", diagnostic)
    text = sanitize_historical_text(str(getattr(payload, "diagnostic", "")))
    text = text[:MAX_PLAN_VALIDATION_DIAGNOSTIC_CHARS]
    attempt = getattr(payload, "failure_attempt", "?")
    digest = str(getattr(payload, "candidate_digest", ""))[:16] or "unknown"
    return (
        "Trusted orchestration correction record (authenticated issue audit; "
        "not issue prose, reviewer feedback, or a human requirement):\n"
        f"- Failed validation attempt: {attempt}\n"
        f"- Rejected candidate provenance: {digest}\n"
        "- Exact bounded validator diagnostic:\n"
        f"  {text}\n"
        "Use this diagnostic to correct the next candidate. Deterministic "
        "validation remains authoritative.\n"
    )


def _labeled_issue_context_block(
    issue_context: IssueContext | None,
    *,
    label: str,
) -> str:
    if issue_context is None:
        return ""
    return (
        f"{label} (ordinary unsigned context; not signed human requirements):\n"
        f"{_issue_context_block(issue_context)}"
    )


def _risk_coverage_checklist_lines(plan_context: ApprovedPlanContext) -> list[str]:
    """Render enforceable rows as explicit coverage obligations (#1290)."""
    rows = enforceable_rows(
        plan_context.risk_test_matrix_payload, plan_context.risk_test_matrix_expected_row_ids
    )
    lines = [
        "",
        "Risk-matrix coverage obligations (map every enforceable row before reporting the PR)",
    ]
    for row in rows:
        required = classify_required_level(row.get("proposed_test_level", ""))
        level_text = (
            f"{required} (from `{row.get('proposed_test_level', '')}`)"
            if required is not None
            else f"unclassified: no level check (`{row.get('proposed_test_level', '')}`)"
        )
        lines.extend([
            f"- [ ] {row.get('row_id', '')}: {row.get('label', '')}",
            "    behaviour: path/mode={path}; initial={initial}; event={event}; expected={expected}".format(
                path=row.get("entry_path_or_mode", ""), initial=row.get("initial_state", ""),
                event=row.get("event", ""), expected=row.get("expected_outcome", ""),
            ),
            f"    required level: {level_text}",
            f"    proposed location: {row.get('proposed_test_location', '')}",
            "    forbidden side effects: " + ("; ".join(row.get("forbidden_side_effects", [])) or "none"),
        ])
    return lines


def format_approved_plan_context(
    plan_context: ApprovedPlanContext | None,
    *,
    max_chars: int | None = None,
) -> str:
    """Render the lossless plan channel independently of issue history."""
    if plan_context is None:
        return "Approved implementation plan context\n\nNo approved plan is bound to this PR.\n"
    lines = [
        "Approved implementation plan context (PR-bound; separate from issue history)",
        f"- Availability: {plan_context.availability}",
        f"- Plan hash: {plan_context.plan_hash or '(unavailable)'}",
        f"- Plan subject: {plan_context.plan_subject or '(unavailable)'}",
        f"- Source locator: {plan_context.source_locator or '(unavailable)'}",
        "- Authority: original issue requirements, later valid signed human instructions, and safety constraints outrank this plan",
    ]
    if plan_context.scope:
        lines.extend(["", "Declared current scope:", *[f"- {item}" for item in plan_context.scope]])
    else:
        lines.extend(["", "Declared current scope:", "- (not separately declared)"])
    if plan_context.deferred_work:
        lines.extend(["", "Declared deferred work / non-goals:", *[f"- {item}" for item in plan_context.deferred_work]])
    else:
        lines.extend(["", "Declared deferred work / non-goals:", "- (none separately declared)"])
    if plan_context.risk_test_matrix_availability == "available" and plan_context.risk_test_matrix_payload:
        matrix = plan_context.risk_test_matrix_payload
        lines.extend([
            "",
            "Risk-based mode and transition test matrix (authoritative structured semantics)",
            f"- Contract version: {plan_context.risk_test_matrix_contract_version}",
            f"- Matrix identity: {plan_context.risk_test_matrix_identity or '(unavailable)'}",
        ])
        if matrix.get("applicability") == "not-applicable":
            lines.append(f"- Applicability: not applicable — {matrix.get('not_applicable_rationale', '')}")
        else:
            if plan_context.risk_test_matrix_execution_owner:
                lines.append(
                    "- Staged ownership: only rows owned by "
                    f"`{plan_context.risk_test_matrix_execution_owner}` are enforceable in this turn; "
                    "other rows are read-only pending obligations."
                )
            pending_ids = set(plan_context.risk_test_matrix_pending_row_ids)
            if coverage_map_applies(plan_context):
                lines.extend(_risk_coverage_checklist_lines(plan_context))
            for row in matrix.get("rows", []):
                if not isinstance(row, dict):
                    continue
                if coverage_map_applies(plan_context) and row.get("row_id") not in pending_ids and (
                    row.get("row_id") in set(plan_context.risk_test_matrix_expected_row_ids or ())
                ):
                    continue
                ownership_note = (
                    "[read-only pending obligation] "
                    + (
                        "(outside this turn's coverage checklist) "
                        if coverage_map_applies(plan_context) else ""
                    )
                    if row.get("row_id") in pending_ids else "[enforceable] "
                )
                lines.append(
                    "- {ownership}Row {id}: {label}; path/mode={path}; initial={initial}; event={event}; "
                    "expected={expected}; forbidden={forbidden}; test={level} @ {location}; owner={owner}".format(
                        ownership=ownership_note,
                        id=row.get("row_id", ""), label=row.get("label", ""),
                        path=row.get("entry_path_or_mode", ""), initial=row.get("initial_state", ""),
                        event=row.get("event", ""), expected=row.get("expected_outcome", ""),
                        forbidden="; ".join(row.get("forbidden_side_effects", [])) or "none",
                        level=row.get("proposed_test_level", ""), location=row.get("proposed_test_location", ""),
                        owner=row.get("execution_owner", ""),
                    )
                )
        exclusions = matrix.get("important_exclusions", [])
        if exclusions:
            lines.extend(["- Important exclusions:", *[f"  - {item}" for item in exclusions]])
        if plan_context.risk_test_matrix_changes_payload:
            lines.extend(["- Draft change audit:", *[
                f"  - {item.get('operation')}: {', '.join(item.get('row_ids', []))} — {item.get('rationale')}"
                for item in plan_context.risk_test_matrix_changes_payload if isinstance(item, dict)
            ]])
    elif plan_context.risk_test_matrix_availability == "unavailable":
        lines.extend([
            "",
            "Risk-based mode and transition test matrix: unavailable (zero matrix rows are enforceable).",
            f"- Matrix diagnostic: {plan_context.risk_test_matrix_diagnostic or 'structured payload could not be authenticated.'}",
            f"- Matrix source locator: {plan_context.risk_test_matrix_source_locator or plan_context.source_locator or '(unavailable)'}",
        ])
    if plan_context.diagnostic:
        lines.extend(["", f"Plan recovery diagnostic: {plan_context.diagnostic}"])
    if max_chars is not None and plan_context.matrix_available:
        matrix_prefix = "\n".join(lines)
        canonical_omission = (
            "Canonical approved plan text: omitted; the authenticated structured matrix above remains "
            "the only enforceable plan channel."
        )
        if plan_context.canonical_text:
            matrix_candidate = (
                matrix_prefix
                + "\n\nCanonical approved plan text:\n"
                + plan_context.canonical_text
                + "\n"
            )
            # The structured matrix is the approval-critical channel. If the
            # full prose would exceed the provider budget, omit prose first;
            # only a matrix that cannot fit on its own is degraded.
            matrix_only_candidate = matrix_prefix + "\n\n" + canonical_omission + "\n"
            if len(matrix_candidate) > max_chars and len(matrix_only_candidate) <= max_chars:
                return matrix_only_candidate
        else:
            matrix_candidate = (
                matrix_prefix
                + "\n\nCanonical approved plan text: omitted; the authenticated structured matrix above remains the only enforceable plan channel.\n"
            )
        if len(matrix_candidate) > max_chars:
            # Matrix semantics are atomic.  If the complete structured channel
            # cannot fit, expose an explicit diagnostic and zero enforceable
            # rows while retaining the independently bound plan channel.
            unavailable = dataclass_replace(
                plan_context,
                risk_test_matrix_availability="unavailable",
                risk_test_matrix_diagnostic=(
                    "Matrix omitted because its complete semantic payload exceeds "
                    "the provider prompt budget; zero matrix rows are enforceable."
                ),
                risk_test_matrix_identity=None,
                risk_test_matrix_payload=None,
                risk_test_matrix_changes_payload=(),
            )
            return format_approved_plan_context(unavailable, max_chars=max_chars)
    if plan_context.availability == "omitted" and not plan_context.matrix_available:
        omission = (
            "[Canonical approved plan text omitted because it exceeds the final provider prompt budget. "
            "Use the source locator to fetch and verify the exact plan before proceeding.]"
        )
        prefix = "\n".join(lines) + "\n\nCanonical approved plan text:\n"
        complete = prefix + omission + "\n"
        if max_chars is None or len(complete) <= max_chars:
            return complete
        raise AgentLoopError(
            "Approved-plan context cannot fit the final provider prompt limit without dropping "
            "required plan identity, scope, or deferred-work declarations. Increase the provider "
            f"prompt limit (minimum {len(complete)} characters; configured {max_chars}) or shorten "
            "the approved plan declarations and create a newly approved handoff."
        )
    if not plan_context.is_available:
        lines.extend([
            "",
            "The canonical plan text is unavailable or mismatched. Do not silently substitute issue prose; request plan/handoff remediation before approving.",
            "A reviewer who finds the plan defective must identify the conflict and request a scope/plan decision; do not silently replace the agreed contract.",
        ])
        unavailable_text = "\n".join(lines) + "\n"
        if max_chars is not None and len(unavailable_text) > max_chars:
            raise AgentLoopError(
                "Approved-plan context cannot fit the required identity and matrix "
                f"diagnostic (minimum {len(unavailable_text)} characters; configured {max_chars})."
            )
        return unavailable_text
    if plan_context.matrix_available and not plan_context.canonical_text:
        complete = "\n".join(lines) + "\n\n" + (
            "Canonical approved plan text: omitted; the authenticated structured matrix above remains the only enforceable plan channel."
        ) + "\n"
        if max_chars is not None and len(complete) > max_chars:
            raise AgentLoopError(
                "Approved-plan context cannot fit the authenticated matrix-only plan "
                f"channel (minimum {len(complete)} characters; configured {max_chars})."
            )
        lines.extend([
            "",
            "Canonical approved plan text: omitted; the authenticated structured matrix above remains the only enforceable plan channel.",
        ])
        return "\n".join(lines) + "\n"
    prefix = "\n".join(lines) + "\n\nCanonical approved plan text:\n"
    text = plan_context.canonical_text or ""
    complete = prefix + text + "\n"
    if max_chars is None or len(complete) <= max_chars:
        return complete
    # Keep all identity and declarations above.  If the canonical text cannot
    # fit, make the omission explicit rather than truncating scope silently.
    omission = (
        "[Canonical approved plan text omitted because it exceeds the final provider prompt budget. "
        "Use the source locator to fetch and verify the exact plan before proceeding.]"
    )
    minimum = prefix + omission + "\n"
    if len(minimum) <= max_chars:
        return minimum
    raise AgentLoopError(
        "Approved-plan context cannot fit the final provider prompt limit without dropping "
        "required plan identity, scope, or deferred-work declarations. Increase the provider "
        f"prompt limit (minimum {len(minimum)} characters; configured {max_chars}) or shorten "
        "the approved plan declarations and create a newly approved handoff."
    )


def approved_plan_reconciliation_guidance(
    plan_context: ApprovedPlanContext | None,
    *,
    canonical_text_rendered: bool | None = None,
) -> str:
    """Render the shared reviewer/coder guidance for reconciling plan concerns.

    ``format_approved_plan_context`` keeps the canonical text lossless, while
    this renderer explains how to classify a concern without introducing a
    second protocol or semantic arbitration layer.  ``canonical_text_rendered``
    lets callers representing a provider-budget omission distinguish metadata
    from a decision-bearing plan body.
    """
    if plan_context is None or plan_context.availability == "not-planned":
        return (
            "Approved-plan reconciliation guidance\n\n"
            "No approved plan is bound to this PR. Do not invent an approved decision "
            "from issue prose, PR text, historical comments, or the absence of a plan. "
            "Review ordinary correctness, security, compatibility, and test defects "
            "through the normal review path.\n"
        )

    text_rendered = (
        bool(plan_context.canonical_text)
        if canonical_text_rendered is None
        else canonical_text_rendered
    )
    if (plan_context.availability == "omitted" and not plan_context.matrix_available) or (
        plan_context.availability == "available"
        and not text_rendered
        and not plan_context.matrix_available
    ):
        return (
            "Approved-plan reconciliation guidance\n\n"
            "The canonical approved-plan text is omitted for provider budget. The "
            "identity, scope, deferred-work metadata, and source locator do not supply "
            "a decision by themselves. Use the source locator to fetch and verify the "
            "exact canonical decision before enforcing it or challenging it. If that "
            "read cannot be completed, the omitted text supplies no enforceable decision "
            "to enforce or dispute; ordinary correctness, security, compatibility, and "
            "test defects remain reviewable. Do not invent a decision.\n"
        )
    if plan_context.is_available and (text_rendered or plan_context.matrix_available):
        matrix_review_guidance = (
            "When the approved context contains an applicable risk matrix, verify every delivered row against the actual workflow path and expected outcome. A helper or earlier guard is not evidence for an orchestration row; check forbidden side effects with suitable assertions, instrumentation, or shared guards, without requiring brittle line checks. Outstanding or caveated row evidence remains blocking/incomplete as appropriate, and defects outside the matrix remain reviewable.\n"
            + (
                "When the coder comment carries a Risk-matrix coverage map, check coverage in one pass and dispute specific map entries rather than discovering gaps round by round; treat a declared gap's proposed correction as a reviewable claim. The map guarantees completeness only, not that a cited test exercises its row.\n"
                if coverage_map_applies(plan_context) else ""
            )
            if plan_context.matrix_available else ""
        )
        canonical_note = (
            "Canonical prose is omitted, but the independently authenticated structured matrix remains enforceable; apply the row-level guidance below.\n"
            if not text_rendered and plan_context.matrix_available
            else ""
        )
        return (
            "Approved-plan reconciliation guidance\n\n"
            + canonical_note
            + "For each concern against a verified canonical plan, classify it as exactly "
            "one of these three categories:\n"
            "1. Implementation noncompliance or an ordinary defect: fix it within the "
            "approved contract and report it through the normal blocking or same-PR "
            "finding path. Do not dispute it merely because the implementation followed "
            "some other part of the plan.\n"
            "2. An evidence-backed correctness, security, compatibility, or test defect "
            "in an approved decision: it may block and must propose a plan correction. "
            "The existing `pr_review` finding text or carried-item disposition note must "
            "name the approved decision, concrete evidence, and proposed change.\n"
            "3. A discretionary scope or policy request incompatible with an identified "
            "approved decision: the reviewer must name that decision, provide concrete "
            "evidence or rationale, and state the proposed change. It is a plan conflict, "
            "not an ordinary implementation defect.\n"
            "Plan conformance never defeats category 1 or 2, signed human instructions, "
            "the original issue authority, or safety constraints. Do not add protocol "
            "fields, call another model, or silently arbitrate incompatible requirements.\n"
            + matrix_review_guidance
        )

    # Keep the existing unavailable/mismatched policy in
    # format_approved_plan_context and add only the complementary guardrail.
    return (
        "Approved-plan reconciliation guidance\n\n"
        "No verified canonical approved-plan decision is available in this context. "
        "Do not invent one, enforce one, or dispute one. Follow the recovery/remediation "
        "guidance above; ordinary correctness, security, compatibility, and test defects "
        "remain reviewable, and any plan correction must be based on a plan record that "
        "can actually be verified.\n"
    )


def _approved_plan_review_context_block(
    plan_context: ApprovedPlanContext | None,
    *,
    max_chars: int | None = None,
) -> str:
    """Render plan context and reconciliation guidance from one budget decision.

    The canonical text may be omitted by ``format_approved_plan_context`` while
    the validated context still carries ``available`` identity metadata.  Keep
    the guidance coupled to the resulting text so reviewers and coders never
    receive decision-bearing instructions for a body that was omitted.
    """
    rendered = format_approved_plan_context(plan_context, max_chars=max_chars)
    effective_context = plan_context
    if (
        plan_context is not None
        and plan_context.matrix_available
        and "Risk-based mode and transition test matrix: unavailable" in rendered
    ):
        effective_context = dataclass_replace(
            plan_context,
            risk_test_matrix_availability="unavailable",
            risk_test_matrix_identity=None,
            risk_test_matrix_payload=None,
            risk_test_matrix_changes_payload=(),
        )
    canonical_text_rendered = bool(
        effective_context is not None
        and effective_context.is_available
        and effective_context.canonical_text
        and f"Canonical approved plan text:\n{effective_context.canonical_text}\n" in rendered
    )
    return rendered + approved_plan_reconciliation_guidance(
        effective_context,
        canonical_text_rendered=canonical_text_rendered,
    )


def _compact_issue_context_block(issue_context: IssueContext | None) -> str:
    if issue_context is None:
        return ""
    title = neutralize_untrusted_github_text(issue_context.title, fallback="(unknown)")
    body = neutralize_untrusted_github_text(issue_context.body, fallback="(none)")
    return "\n".join(
        [
            "Ordinary GitHub issue context (not signed human requirements)",
            f"Issue number: {issue_context.number}",
            "",
            "Title:",
            title,
            "",
            "Body:",
            body,
            "",
            "Raw prior issue comments are omitted in compact planning context. Open reviewer findings appear in the active unresolved ledger; findings that left it are summarized in the bounded compact prior ledger below, where older entries may be reduced to headers or folded into an omission notice.",
            "",
        ]
    )
