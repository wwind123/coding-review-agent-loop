"""Public comment and canonical plan rendering helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence

from .errors import AgentLoopError
from .agents.registry import agent_display_name, agent_signature
from .protocol import (
    ANY_HEADING_RE,
    HTML_COMMENT_RE,
    PLAN_STATE_RE,
    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
    PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
    SIGNATURE_RE,
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    review_freeform_summary_text,
)
from .unresolved_items import HUMAN_REQUIREMENTS_ACK_ITEM_ID

ITEM_SUMMARY_LIMIT = 100
PUBLIC_REVIEWER_NAME_BY_DISPLAY = {
    agent_display_name(agent): agent_signature(agent)
    for agent in ("claude", "codex", "gemini", "antigravity")
}


def _review_freeform_summary_text(text: str) -> str:
    return review_freeform_summary_text(text)


def _normalize_item_summary(text: str, *, limit: int = ITEM_SUMMARY_LIMIT) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        stripped = re.sub(r"^(?:[-*+]\s+|\d+[.)]\s+)+", "", stripped)
        stripped = " ".join(stripped.split())
        if len(stripped) <= limit:
            return stripped
        if limit <= 3:
            return stripped[:limit]
        return stripped[: limit - 3].rstrip() + "..."
    return "No summary provided."


def _item_label_status(item: UnresolvedReviewItem) -> str:
    return item.source_status or item.status


def _public_reviewer_name(name: str) -> str:
    return PUBLIC_REVIEWER_NAME_BY_DISPLAY.get(name, name)


def _format_unresolved_item_label(item: UnresolvedReviewItem) -> str:
    summary = _normalize_item_summary(item.text)
    status = _item_label_status(item)
    if item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID:
        return f"Human-requirements acknowledgement item, round {item.source_round}: {summary}"
    phrases = {
        "blocking": "Blocking issue",
        "same-pr": "Same-PR follow-up",
        "same-plan": "Same-plan follow-up",
        "future": "Future follow-up",
    }
    phrase = phrases.get(status, "Unresolved item")
    reviewer_name = _public_reviewer_name(item.reviewer)
    return f"{phrase} from {reviewer_name}, round {item.source_round}: {summary}"


def _render_disposition_status(disposition: ReviewItemDisposition) -> str:
    labels = {
        "resolved": "resolved",
        "blocking": "still blocking",
        "same-pr": "same-pr",
        "same-plan": "same-plan",
        "future": "future follow-up",
    }
    rendered = labels.get(disposition.disposition, disposition.disposition)
    if disposition.note:
        rendered = f"{rendered}: {disposition.note}"
    return rendered


def _render_prior_dispositions_section(
    *,
    heading: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> str:
    item_by_id = {item.item_id: item for item in prior_items}
    lines = [heading]
    for disposition in dispositions:
        item = item_by_id.get(disposition.item_id)
        if item is None:
            raise AgentLoopError(
                f"Renderer encountered unknown prior item ID {disposition.item_id!r}; "
                f"allowed IDs: {sorted(item_by_id)}"
            )
        lines.append(
            f"- [{disposition.item_id}] {_format_unresolved_item_label(item)}"
            f" -> {_render_disposition_status(disposition)}"
        )
    return "\n".join(lines)


def _replace_structured_section(
    body: str,
    *,
    heading_re: re.Pattern[str],
    replacement: str,
) -> str:
    lines = body.splitlines()
    output: list[str] = []
    index = 0
    replaced = False
    while index < len(lines):
        line = lines[index]
        if not replaced and heading_re.match(line):
            output.extend(replacement.splitlines())
            replaced = True
            index += 1
            while index < len(lines):
                current = lines[index]
                if (
                    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE.match(current)
                    or PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE.match(current)
                    or (
                        current.strip()
                        and (
                            ANY_HEADING_RE.match(current)
                            or HTML_COMMENT_RE.match(current)
                            or SIGNATURE_RE.match(current)
                        )
                    )
                ):
                    break
                index += 1
            continue
        output.append(line)
        index += 1
    return "\n".join(output) if replaced else body


def _append_before_trailing_metadata(body: str, section: str) -> str:
    lines = body.splitlines()
    index = len(lines)
    while index > 0 and not lines[index - 1].strip():
        index -= 1
    metadata_start = index
    while metadata_start > 0:
        candidate = lines[metadata_start - 1]
        if not candidate.strip() or HTML_COMMENT_RE.match(candidate) or SIGNATURE_RE.match(candidate):
            metadata_start -= 1
            continue
        break
    if metadata_start == index:
        return body.rstrip() + "\n\n" + section
    prefix = "\n".join(lines[:metadata_start]).rstrip()
    suffix = "\n".join(lines[metadata_start:]).lstrip("\n")
    parts = [prefix, section, suffix]
    return "\n\n".join(part for part in parts if part)


def render_canonical_plan_steps(plan_steps: Sequence[str]) -> str:
    return "\n".join(f"{index}. {step}" for index, step in enumerate(plan_steps, start=1))


def render_canonical_plan_revision(
    parsed_revision: StructuredPlanRevision,
    prior_items: Sequence[UnresolvedReviewItem],
) -> str:
    sections = [parsed_revision.summary.strip()]
    if prior_items or parsed_revision.prior_plan_item_dispositions:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior plan item dispositions",
                prior_items=prior_items,
                dispositions=parsed_revision.prior_plan_item_dispositions,
            )
        )
    else:
        sections.append("### Prior plan review item dispositions\n- None.")
    sections.append(
        "\n".join(
            [
                "### Plan steps",
                render_canonical_plan_steps(parsed_revision.plan_steps),
            ]
        )
    )
    return "\n\n".join(sections)


def _render_public_review_comment(
    body: str,
    *,
    review_kind: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    new_items: Sequence[UnresolvedReviewItem],
) -> str:
    rendered = body
    if prior_items:
        heading = "### Prior unresolved item dispositions"
        heading_re = PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE
        if review_kind == "plan":
            heading = "### Prior unresolved plan item dispositions"
            heading_re = PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE
        rendered = _replace_structured_section(
            rendered,
            heading_re=heading_re,
            replacement=_render_prior_dispositions_section(
                heading=heading,
                prior_items=prior_items,
                dispositions=dispositions,
            ),
        )
    return rendered


def _render_public_pr_review_comment(
    parsed_review: ParsedReview,
    *,
    reviewer: str,
    human_requirements_resolved_flag: bool,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
) -> str:
    sections: list[str] = [f"**Review verdict:** {parsed_review.state.title()}"]
    if parsed_review.summary and parsed_review.summary.strip() != "Review complete.":
        sections.append(parsed_review.summary.strip())
    if parsed_review.blocking_items:
        sections.append(
            "\n".join(
                [
                    "### Blocking issues",
                    *[f"- {item.text}" for item in parsed_review.blocking_items],
                ]
            )
        )
    if parsed_review.followups.same_pr:
        sections.append(
            "\n".join(
                [
                    "### Same-PR follow-ups",
                    *[f"- {item.text}" for item in parsed_review.followups.same_pr],
                ]
            )
        )
    if parsed_review.followups.future:
        sections.append(
            "\n".join(
                [
                    "### Future follow-ups",
                    *[f"- {item.text}" for item in parsed_review.followups.future],
                ]
            )
        )
    if prior_items:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior unresolved item dispositions",
                prior_items=prior_items,
                dispositions=dispositions,
            )
        )
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.append(f"<!-- AGENT_STATE: {parsed_review.state} -->")
    footer.append(f"-- {_public_reviewer_name(reviewer)}")
    return "\n\n".join(section for section in sections if section) + (
        ("\n\n" if sections else "") + "\n".join(footer)
    )


def _render_public_plan_review_comment(
    parsed_review: ParsedPlanReview,
    *,
    reviewer: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    human_requirements_resolved_flag: bool = False,
) -> str:
    sections: list[str] = [f"**Review verdict:** {parsed_review.state.title()}"]
    if parsed_review.summary and parsed_review.summary.strip() != "Plan review complete.":
        sections.append(parsed_review.summary.strip())
    if parsed_review.items.blocking:
        sections.append(
            "\n".join(
                [
                    "### Blocking plan issues",
                    *[f"- {item.text}" for item in parsed_review.items.blocking],
                ]
            )
        )
    if parsed_review.items.same_plan:
        sections.append(
            "\n".join(
                [
                    "### Same-plan follow-ups",
                    *[f"- {item.text}" for item in parsed_review.items.same_plan],
                ]
            )
        )
    if parsed_review.items.future:
        sections.append(
            "\n".join(
                [
                    "### Future follow-ups",
                    *[f"- {item.text}" for item in parsed_review.items.future],
                ]
            )
        )
    if prior_items:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior unresolved plan item dispositions",
                prior_items=prior_items,
                dispositions=dispositions,
            )
        )
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.extend(
        [
            f"<!-- AGENT_PLAN_STATE: {parsed_review.state} -->",
            f"-- {_public_reviewer_name(reviewer)}",
        ]
    )
    return "\n\n".join(section for section in sections if section) + (
        ("\n\n" if sections else "") + "\n".join(footer)
    )


def _render_public_coder_followup_comment(
    parsed_followup: StructuredCoderFollowup,
    *,
    signature: str,
    prior_items: Sequence[UnresolvedReviewItem] = (),
) -> str:
    item_by_id = {item.item_id: item for item in prior_items}

    def render_item(item_id: str, *, note_label: str, note: str | None, placeholder: str | None) -> list[str]:
        item = item_by_id.get(item_id)
        if item is None:
            lines = [f"- {item_id}: Item context unavailable in current round metadata."]
        else:
            lines = [f"- {item_id}: {_format_unresolved_item_label(item)}"]
        if note:
            lines.append(f"  - {note_label}: {note}")
        elif placeholder:
            lines.append(f"  - {note_label}: {placeholder}")
        return lines

    addressed_items: list[str] = []
    for item_id in parsed_followup.addressed_items:
        addressed_items.extend(
            render_item(
                item_id,
                note_label="Resolution",
                note=parsed_followup.addressed_item_notes.get(item_id),
                placeholder=None,
            )
        )
    if not addressed_items:
        addressed_items = ["- None."]

    remaining_items: list[str] = []
    for item_id in parsed_followup.remaining_items:
        remaining_items.extend(
            render_item(
                item_id,
                note_label="Reason",
                note=parsed_followup.remaining_item_notes.get(item_id),
                placeholder="No reason provided by coder.",
            )
        )
    if not remaining_items:
        remaining_items = ["- None."]

    sections = [
        "## Coder follow-up",
        parsed_followup.summary.strip(),
        "\n".join(["### Addressed items", *addressed_items]),
        "\n".join(["### Remaining items", *remaining_items]),
    ]
    if parsed_followup.tests_run:
        sections.append(
            "\n".join(["### Tests run", *[f"- {test}" for test in parsed_followup.tests_run]])
        )
    sections.append(f"<!-- AGENT_STATE: {parsed_followup.state} -->")
    sections.append(f"-- {signature}")
    return "\n\n".join(section for section in sections if section)


def _extract_plan_revision_human_requirements_block(text: str) -> str:
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return ""
    decoder = json.JSONDecoder()
    payload, end = decoder.raw_decode(stripped)
    if not isinstance(payload, dict):
        return ""
    trailing = stripped[end:].lstrip()
    state_match = PLAN_STATE_RE.search(trailing)
    if state_match is None:
        return ""
    return trailing[: state_match.start()].strip()


def _render_public_plan_revision_comment(
    parsed_revision: StructuredPlanRevision,
    *,
    prior_items: Sequence[UnresolvedReviewItem],
    raw_text: str,
    signature: str,
) -> str:
    sections = ["## Revised plan", render_canonical_plan_revision(parsed_revision, prior_items)]
    human_requirements_block = _extract_plan_revision_human_requirements_block(raw_text)
    if human_requirements_block:
        sections.append(human_requirements_block)
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed_revision.state} -->")
    sections.append(f"-- {signature}")
    return "\n\n".join(section for section in sections if section)
