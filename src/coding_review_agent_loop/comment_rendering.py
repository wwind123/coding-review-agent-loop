"""Public comment and canonical plan rendering helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from .decomposition import _decode_json_payload, _encode_json_payload
from .errors import AgentLoopError
from .expected_closure import normalize_issue_ids
from .agents.registry import agent_display_name, agent_signature
from .protocol import (
    ANY_HEADING_RE,
    HTML_COMMENT_RE,
    PLAN_STATE_RE,
    PRIOR_UNRESOLVED_ITEM_DISPOSITIONS_HEADING_RE,
    PRIOR_UNRESOLVED_PLAN_ITEM_DISPOSITIONS_HEADING_RE,
    SIGNATURE_RE,
    AgentUnavailable,
    DeferredStage,
    TypedPlanStages,
    HumanRequirementDisposition,
    ParsedDiscussAgenda,
    ParsedDiscussAnswer,
    ParsedDiscussFinalSynthesis,
    ParsedFailedDiscussResponse,
    ParsedDiscussRoundSynthesis,
    ParsedDiscussResponse,
    ParsedDiscussReview,
    ParsedPlanReview,
    ParsedReview,
    ReviewItemDisposition,
    StructuredCoderFollowup,
    StructuredIssueImplementation,
    StructuredPlanState,
    StructuredPlanRevision,
    UnresolvedReviewItem,
    review_freeform_summary_text,
)
from .unresolved_items import HUMAN_REQUIREMENTS_ACK_ITEM_ID, MERGE_CONFLICT_ITEM_ID
from .protocol_markers import sanitize_historical_text
from .round_transport import MAX_GITHUB_BODY_CHARS

if TYPE_CHECKING:
    from .agents.base import AgentName
    from .config import AgentLoopConfig

ITEM_SUMMARY_LIMIT = 100
PLAN_EXPECTED_CLOSING_MARKER = "AGENT_PLAN_EXPECTED_CLOSING_ISSUES"
PLAN_EXPECTED_CLOSING_MARKER_RE = re.compile(
    rf"<!--\s*{PLAN_EXPECTED_CLOSING_MARKER}:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.IGNORECASE,
)
# Reverse map display-name -> agent. agent_display_name is config-independent, so
# this is safe to build at import; the signature itself is resolved per-call via
# agent_signature(agent, config) so it can reflect the configured model (#332).
_AGENT_BY_DISPLAY_NAME = {
    agent_display_name(agent): agent
    for agent in ("claude", "codex", "gemini", "antigravity")
}


def render_agent_unavailable_comment(unavailable: AgentUnavailable, *, signature: str) -> str:
    """Render a protocol-valid ``agent_unavailable`` envelope that parse_agent_unavailable accepts.

    Used only for orchestrator-synthesized outcomes (e.g. an exhausted
    completion-recovery attempt, #588) where there is no agent-authored
    envelope to post verbatim. The JSON payload, footer, and signature must
    exactly match the grammar in protocol.py's
    _consume_agent_unavailable_footer_and_signature: no prose between the JSON
    and the footer, and nothing but the signature after it.
    """
    payload = {
        "schema_version": unavailable.schema_version,
        "kind": "agent_unavailable",
        "retryable": unavailable.retryable,
        "category": unavailable.category,
        "summary": unavailable.summary,
        "suggested_action": unavailable.suggested_action,
    }
    return f"{json.dumps(payload, ensure_ascii=False)}\n<!-- AGENT_UNAVAILABLE -->\n-- {signature}"


def _review_freeform_summary_text(text: str) -> str:
    return review_freeform_summary_text(text)


def _normalize_item_summary(text: str, *, limit: int = ITEM_SUMMARY_LIMIT) -> str:
    # Prior-item text is historical ledger data, not a fresh agent output.  It
    # must be neutralized before truncation so a marker cannot survive in a
    # later public rendering or be recreated at the truncation boundary.
    text = sanitize_historical_text(text)
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


def _public_reviewer_name(
    name: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    agent = _AGENT_BY_DISPLAY_NAME.get(name)
    if agent is None and name in {"claude", "codex", "gemini", "antigravity"}:
        agent = name  # type: ignore[assignment]
    if agent is None:
        return name
    # `model_used` is the producing agent's actual model (footers). Prior-item
    # attributions pass model_used=None and use the configured model (or the
    # generic signature when nothing is declared) — no cross-agent leakage.
    return agent_signature(agent, config, model_used)


def _comment_signature(
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    return _public_reviewer_name(agent, config, model_used)


def _format_unresolved_item_label(
    item: UnresolvedReviewItem, config: AgentLoopConfig | None = None
) -> str:
    summary = _normalize_item_summary(item.text)
    status = _item_label_status(item)
    if item.item_id == HUMAN_REQUIREMENTS_ACK_ITEM_ID:
        return f"Human-requirements acknowledgement item, round {item.source_round}: {summary}"
    if item.item_id == MERGE_CONFLICT_ITEM_ID:
        return f"Merge conflict item, round {item.source_round}: {summary}"
    phrases = {
        "blocking": "Blocking issue",
        "same-pr": "Same-PR follow-up",
        "same-plan": "Same-plan follow-up",
        "future": "Future follow-up",
    }
    phrase = phrases.get(status, "Unresolved item")
    reviewer_name = _public_reviewer_name(item.reviewer, config)
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
    config: AgentLoopConfig | None = None,
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
            f"- [{disposition.item_id}] {_format_unresolved_item_label(item, config)}"
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


def render_expected_closing_issue_declaration(
    issue_ids: Sequence[int] | None,
) -> str | None:
    """Render an authoritative plan declaration, retaining omitted vs empty."""
    if issue_ids is None:
        return None
    normalized = normalize_issue_ids(issue_ids, field_name="additional_closing_issue_ids")
    assert normalized is not None
    payload = {"issue_ids": list(normalized)}
    encoded = _encode_json_payload(payload)
    visible = ", ".join(f"#{item}" for item in normalized) or "none"
    return "\n".join(
        [
            "### Additional issues completed by this implementation PR",
            f"Only these additional same-repository issues are part of this single-PR closing contract: {visible}.",
            "Incidental mentions, `Refs` links, related issues, staged parents, unselected stages, deferred work, and plan actions are not declarations.",
            f"<!-- {PLAN_EXPECTED_CLOSING_MARKER}: {encoded} -->",
        ]
    )


def decode_expected_closing_issue_declaration(encoded: str) -> tuple[int, ...]:
    payload = _decode_json_payload(encoded, marker_name=PLAN_EXPECTED_CLOSING_MARKER)
    if _encode_json_payload(payload) != encoded:
        raise AgentLoopError(
            f"Invalid {PLAN_EXPECTED_CLOSING_MARKER} payload: non-canonical encoding."
        )
    if set(payload) != {"issue_ids"}:
        raise AgentLoopError(f"Invalid {PLAN_EXPECTED_CLOSING_MARKER} payload.")
    normalized = normalize_issue_ids(
        payload["issue_ids"], field_name=f"{PLAN_EXPECTED_CLOSING_MARKER}.issue_ids"
    )
    assert normalized is not None
    return normalized


def render_human_requirement_dispositions(
    dispositions: Sequence[HumanRequirementDisposition],
    *,
    heading: str = "### Human requirement dispositions",
) -> str | None:
    if not dispositions:
        return None
    return "\n".join(
        [heading]
        + [
            f"- **{item.requirement_id}** — `{item.disposition}`: {item.evidence}"
            for item in dispositions
        ]
    )


def render_deferred_stages_section(deferred_stages: Sequence[DeferredStage]) -> str | None:
    """Render declared deferred stages so they carry into the plan's markdown.

    Deferred stages must survive canonical-plan rendering (#476): they feed
    subject hashing, stored plan state, reviewer prompts, and resume, so
    scope narrowing stays a mechanical signal instead of prose reviewers
    might miss.

    The human-readable `- {title}: {summary}` bullets are for reviewers; they
    are NOT what resume parses back, because a title containing its own colon
    (e.g. "Stage 2: API follow-up") would corrupt a naive split-on-first-colon
    parse. An `AGENT_DEFERRED_STAGES` HTML-comment marker carries the
    structured title/summary pairs verbatim so they round-trip exactly
    regardless of their text content (#492 review).
    """
    if not deferred_stages:
        return None
    lines = ["### Deferred stages (not in this plan)"]
    for stage in deferred_stages:
        lines.append(f"- {stage.title}: {stage.summary}")
    lines.append(f"<!-- AGENT_DEFERRED_STAGES: {_encode_deferred_stages_marker(deferred_stages)} -->")
    return "\n".join(lines)


def render_typed_plan_stages_section(stages: TypedPlanStages) -> str | None:
    """Render #585 categories; only child stages are eligible for filing."""
    groups = (
        ("Child stages (eligible for child issues)", stages.child_stages),
        ("External dependencies (linked, never created)", stages.external_dependencies),
        ("Deferred work (recorded only)", stages.deferred_work),
        ("Plan actions (recorded only)", stages.plan_actions),
    )
    if not any(entries for _, entries in groups):
        return None
    lines = ["### Structured scope categories"]
    for title, entries in groups:
        if entries:
            lines.append(f"#### {title}")
            lines.extend(f"- {entry.title}: {entry.summary}" for entry in entries)
    payload = {
        "child_stages": [
            {"title": item.title, "summary": item.summary}
            for item in stages.child_stages
        ],
        "external_dependencies": [
            {"title": item.title, "summary": item.summary}
            for item in stages.external_dependencies
        ],
        "deferred_work": [
            {"title": item.title, "summary": item.summary}
            for item in stages.deferred_work
        ],
        "plan_actions": [
            {"title": item.title, "summary": item.summary}
            for item in stages.plan_actions
        ],
    }
    lines.append(f"<!-- AGENT_TYPED_PLAN_STAGES: {_encode_json_payload(payload)} -->")
    return "\n".join(lines)


def _encode_deferred_stages_marker(deferred_stages: Sequence[DeferredStage]) -> str:
    return _encode_json_payload(
        {"stages": [{"title": stage.title, "summary": stage.summary} for stage in deferred_stages]}
    )


DEFERRED_STAGES_MARKER_RE = re.compile(
    r"<!--\s*AGENT_DEFERRED_STAGES:\s*(?P<payload>[A-Za-z0-9+/=_-]+)\s*-->",
    re.I,
)


def decode_deferred_stages_marker(encoded: str) -> tuple[DeferredStage, ...]:
    payload = _decode_json_payload(encoded, marker_name="AGENT_DEFERRED_STAGES")
    stages_payload = payload.get("stages")
    if not isinstance(stages_payload, list):
        raise AgentLoopError("Invalid AGENT_DEFERRED_STAGES payload.")
    stages: list[DeferredStage] = []
    for entry in stages_payload:
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("title"), str)
            or not isinstance(entry.get("summary"), str)
        ):
            raise AgentLoopError("Invalid AGENT_DEFERRED_STAGES payload.")
        stages.append(DeferredStage(title=entry["title"], summary=entry["summary"]))
    return tuple(stages)


def render_canonical_plan_revision(
    parsed_revision: StructuredPlanRevision,
    prior_items: Sequence[UnresolvedReviewItem],
    config: AgentLoopConfig | None = None,
) -> str:
    sections = [parsed_revision.summary.strip()]
    if prior_items or parsed_revision.prior_plan_item_dispositions:
        sections.append(
            _render_prior_dispositions_section(
                heading="### Prior plan item dispositions",
                prior_items=prior_items,
                dispositions=parsed_revision.prior_plan_item_dispositions,
                config=config,
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
    expected_section = render_expected_closing_issue_declaration(
        parsed_revision.additional_closing_issue_ids
    )
    if expected_section:
        sections.append(expected_section)
    human_section = render_human_requirement_dispositions(parsed_revision.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    deferred_section = render_deferred_stages_section(parsed_revision.deferred_stages)
    if deferred_section:
        sections.append(deferred_section)
    typed_section = render_typed_plan_stages_section(parsed_revision.typed_stages)
    if typed_section:
        sections.append(typed_section)
    return "\n\n".join(sections)


def _render_public_review_comment(
    body: str,
    *,
    review_kind: str,
    prior_items: Sequence[UnresolvedReviewItem],
    dispositions: Sequence[ReviewItemDisposition],
    new_items: Sequence[UnresolvedReviewItem],
    config: AgentLoopConfig | None = None,
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
                config=config,
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
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
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
                config=config,
            )
        )
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.append(f"<!-- AGENT_STATE: {parsed_review.state} -->")
    footer.append(f"-- {_public_reviewer_name(reviewer, config, model_used)}")
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
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
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
                config=config,
            )
        )
    human_section = render_human_requirement_dispositions(
        parsed_review.human_requirement_dispositions
    )
    if human_section:
        sections.append(human_section)
    footer: list[str] = []
    if human_requirements_resolved_flag:
        footer.append("<!-- HUMAN_REQUIREMENTS_RESOLVED -->")
    footer.extend(
        [
            f"<!-- AGENT_PLAN_STATE: {parsed_review.state} -->",
            f"-- {_public_reviewer_name(reviewer, config, model_used)}",
        ]
    )
    return "\n\n".join(section for section in sections if section) + (
        ("\n\n" if sections else "") + "\n".join(footer)
    )


def _render_public_coder_followup_comment(
    parsed_followup: StructuredCoderFollowup,
    *,
    agent: str,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    item_by_id = {item.item_id: item for item in prior_items}

    def render_item(item_id: str, *, note_label: str, note: str | None, placeholder: str | None) -> list[str]:
        item = item_by_id.get(item_id)
        if item is None:
            lines = [f"- {item_id}: Item context unavailable in current round metadata."]
        else:
            lines = [f"- {item_id}: {_format_unresolved_item_label(item, config)}"]
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

    disputed_items: list[str] = []
    for item_id in parsed_followup.disputed_items:
        disputed_items.extend(
            render_item(
                item_id,
                note_label="Counter-evidence",
                note=parsed_followup.dispute_evidence.get(item_id),
                placeholder="No evidence provided by coder.",
            )
        )

    sections = [
        "## Coder follow-up",
        parsed_followup.summary.strip(),
        "\n".join(["### Addressed items", *addressed_items]),
        "\n".join(["### Remaining items", *remaining_items]),
    ]
    if disputed_items:
        sections.append("\n".join(["### Disputed items", *disputed_items]))
    if parsed_followup.tests_run:
        sections.append(
            "\n".join(["### Tests run", *[f"- {test}" for test in parsed_followup.tests_run]])
        )
    if parsed_followup.human_requirement_dispositions:
        sections.append(
            "\n".join(
                [
                    "### Human requirements",
                    *[
                        f"- {disposition.requirement_id}: {disposition.disposition} — {disposition.evidence}"
                        for disposition in parsed_followup.human_requirement_dispositions
                    ],
                ]
            )
        )
    sections.append(f"<!-- AGENT_STATE: {parsed_followup.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_issue_implementation_comment(
    parsed: StructuredIssueImplementation,
    *,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    """Render an implementation result without exposing its JSON envelope."""
    has_blocked_requirement = any(
        item.disposition == "blocked"
        for item in parsed.human_requirement_dispositions
    )
    if has_blocked_requirement and parsed.pr_number is not None:
        result = (
            f"Rejected for handoff: reported PR #{parsed.pr_number} was not accepted "
            "because a signed human requirement is blocked."
        )
    elif parsed.pr_number is None:
        result = "No pull request was accepted for handoff."
    else:
        result = f"Pull request reported: #{parsed.pr_number}."
    sections = ["## Issue implementation", parsed.summary.strip(), f"### Result\n{result}"]
    if parsed.tests_run:
        sections.append("\n".join(["### Tests run", *[f"- {test}" for test in parsed.tests_run]]))
    human_section = render_human_requirement_dispositions(
        parsed.human_requirement_dispositions
    )
    if human_section:
        sections.append(human_section)
    sections.extend(
        [
            f"<!-- AGENT_STATE: {parsed.state} -->",
            f"-- {_comment_signature(agent, config, model_used)}",
        ]
    )
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
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    sections = ["## Revised plan", render_canonical_plan_revision(parsed_revision, prior_items, config)]
    human_requirements_block = _extract_plan_revision_human_requirements_block(raw_text)
    if human_requirements_block:
        sections.append(human_requirements_block)
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed_revision.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_plan_state_comment(
    parsed_plan: StructuredPlanState,
    *,
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    sections = [
        "## Plan",
        parsed_plan.summary.strip(),
        "\n".join(["### Plan steps", render_canonical_plan_steps(parsed_plan.plan_steps)]),
    ]
    expected_section = render_expected_closing_issue_declaration(
        parsed_plan.additional_closing_issue_ids
    )
    if expected_section:
        sections.append(expected_section)
    human_section = render_human_requirement_dispositions(parsed_plan.human_requirement_dispositions)
    if human_section:
        sections.append(human_section)
    deferred_section = render_deferred_stages_section(parsed_plan.deferred_stages)
    if deferred_section:
        sections.append(deferred_section)
    typed_section = render_typed_plan_stages_section(parsed_plan.typed_stages)
    if typed_section:
        sections.append(typed_section)
    sections.append(f"<!-- AGENT_PLAN_STATE: {parsed_plan.state} -->")
    sections.append(f"-- {_comment_signature(agent, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def normalize_freeform_signature(
    text: str,
    agent: AgentName,
    config: AgentLoopConfig | None,
    model_used: str | None,
) -> str:
    """Replace or append the trailing agent signature on a free-form response.

    Walks back over trailing blank lines and HTML comments to find the last
    SIGNATURE_RE line and replaces it with the canonical qualified form. If no
    signature is found, one is appended. Structured responses already receive
    canonical signatures via render_public_agent_comment; this function is for
    free-form/legacy text that is posted verbatim.
    """
    canonical = f"-- {agent_signature(agent, config, model_used)}"
    lines = text.rstrip("\n").splitlines()
    tail = len(lines)
    while tail > 0 and (
        not lines[tail - 1].strip() or HTML_COMMENT_RE.match(lines[tail - 1])
    ):
        tail -= 1
    if tail > 0 and SIGNATURE_RE.match(lines[tail - 1]):
        lines[tail - 1] = canonical
        return "\n".join(lines)
    return f"{text.rstrip(chr(10))}\n{canonical}"


def render_public_agent_comment(
    *,
    kind: str,
    parsed: (
        ParsedReview
        | ParsedPlanReview
        | StructuredCoderFollowup
        | StructuredPlanRevision
        | StructuredPlanState
        | ParsedDiscussReview
        | StructuredIssueImplementation
    ),
    agent: str,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
    prior_items: Sequence[UnresolvedReviewItem] = (),
    dispositions: Sequence[ReviewItemDisposition] = (),
    raw_text: str = "",
    human_requirements_resolved_flag: bool = False,
    round_number: int = 1,
) -> str:
    """Render a parsed agent response and stamp the agent/model signature.

    This is the single render-and-sign seam used by CLI and skill mode. Callers
    resolve where the actual model came from, then pass it here; footer signing
    stays owned by the renderer.
    """
    if kind == "pr_review":
        if not isinstance(parsed, ParsedReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedReview.")
        return _render_public_pr_review_comment(
            parsed,
            reviewer=agent,
            human_requirements_resolved_flag=human_requirements_resolved_flag,
            prior_items=prior_items,
            dispositions=dispositions,
            config=config,
            model_used=model_used,
        )
    if kind == "plan_review":
        if not isinstance(parsed, ParsedPlanReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedPlanReview.")
        return _render_public_plan_review_comment(
            parsed,
            reviewer=agent,
            prior_items=prior_items,
            dispositions=dispositions,
            human_requirements_resolved_flag=human_requirements_resolved_flag,
            config=config,
            model_used=model_used,
        )
    if kind == "coder_followup":
        if not isinstance(parsed, StructuredCoderFollowup):
            raise AgentLoopError("render_public_agent_comment expected StructuredCoderFollowup.")
        return _render_public_coder_followup_comment(
            parsed,
            agent=agent,
            prior_items=prior_items,
            config=config,
            model_used=model_used,
        )
    if kind == "issue_implementation":
        if not isinstance(parsed, StructuredIssueImplementation):
            raise AgentLoopError(
                "render_public_agent_comment expected StructuredIssueImplementation."
            )
        return _render_public_issue_implementation_comment(
            parsed,
            agent=agent,
            config=config,
            model_used=model_used,
        )
    if kind == "plan_revision":
        if not isinstance(parsed, StructuredPlanRevision):
            raise AgentLoopError("render_public_agent_comment expected StructuredPlanRevision.")
        return _render_public_plan_revision_comment(
            parsed,
            prior_items=prior_items,
            raw_text=raw_text,
            agent=agent,
            config=config,
            model_used=model_used,
        )
    if kind == "plan_state":
        if not isinstance(parsed, StructuredPlanState):
            raise AgentLoopError("render_public_agent_comment expected StructuredPlanState.")
        return _render_public_plan_state_comment(
            parsed,
            agent=agent,
            config=config,
            model_used=model_used,
        )
    if kind == "discuss_review":
        if not isinstance(parsed, ParsedDiscussReview):
            raise AgentLoopError("render_public_agent_comment expected ParsedDiscussReview.")
        return _render_public_discuss_review_comment(
            parsed,
            reviewer=agent,
            round_number=round_number,
            config=config,
            model_used=model_used,
        )
    if kind == "discuss_answer":
        if not isinstance(parsed, ParsedDiscussAnswer):
            raise AgentLoopError("render_public_agent_comment expected ParsedDiscussAnswer.")
        return _render_public_discuss_answer_comment(
            parsed, reviewer=agent, round_number=round_number, config=config, model_used=model_used
        )
    raise AgentLoopError(f"Unknown render kind: {kind}")


_DISCUSS_OUTCOME_LABELS = {
    "implement": "Implement",
    "do-not-implement": "Do Not Implement",
    "needs-human": "Needs Human Review",
    "split": "Split",
}

_DISCUSS_RESEARCH_STATUS_LABELS = {
    "sourced": "done, sourced facts cited below",
    "not-needed": "not needed (no external-fact trigger applies)",
    "unavailable": "unavailable — related claims are judgment, not sourced fact",
    "inconclusive": "inconclusive — related claims are judgment, not sourced fact",
}


def _render_public_discuss_review_comment(
    parsed: ParsedDiscussReview,
    *,
    reviewer: str,
    round_number: int = 1,
    config: AgentLoopConfig | None = None,
    model_used: str | None = None,
) -> str:
    outcome_label = _DISCUSS_OUTCOME_LABELS.get(parsed.outcome, parsed.outcome)
    sections: list[str] = [
        f"## Round {round_number}: {reviewer} position",
        f"**Vote:** {outcome_label} (`{parsed.outcome}`)",
        parsed.rationale.strip(),
    ]
    if parsed.rebuttal:
        sections.append("### Rebuttal\n\n" + parsed.rebuttal.strip())
    if parsed.analyzer_framing == "misframed":
        note = (parsed.framing_note or "").strip()
        sections.append("### Analyzer framing correction\n\n" + note)
    elif parsed.analyzer_framing == "accurate":
        framing_line = "**Analyzer framing:** accurate"
        if parsed.framing_note:
            framing_line += f" — {parsed.framing_note.strip()}"
        sections.append(framing_line)
    if parsed.split_proposals:
        proposals = "\n".join(f"- {proposal}" for proposal in parsed.split_proposals)
        sections.append("### Proposed sub-issues\n\n" + proposals)
    if parsed.research_status is not None:
        label = _DISCUSS_RESEARCH_STATUS_LABELS.get(parsed.research_status, parsed.research_status)
        sections.append(f"**Research:** {label} (`{parsed.research_status}`)")
    if parsed.research_target:
        sections.append(
            "### Research intent\n\n"
            f"- Target: `{parsed.research_target}`\n"
            + "\n".join(f"- Question: {question}" for question in parsed.research_questions)
        )
    if parsed.sourced_facts:
        facts = "\n".join(
            f"- {fact.fact} — source: {fact.source}" for fact in parsed.sourced_facts
        )
        sections.append("### Sourced facts\n\n" + facts)
    sections.append(f"-- {_comment_signature(reviewer, config, model_used)}")
    return "\n\n".join(section for section in sections if section)


def _render_public_discuss_answer_comment(
    parsed: ParsedDiscussAnswer, *, reviewer: str, round_number: int = 1,
    config: AgentLoopConfig | None = None, model_used: str | None = None,
) -> str:
    sections = [f"## Round {round_number}: {reviewer} answer", f"**Position:** `{parsed.position}`"]
    if parsed.answer:
        sections.append("### Answer\n\n" + parsed.answer.strip())
    sections.append("### Rationale\n\n" + parsed.rationale.strip())
    sections.append(f"**Confidence:** `{parsed.confidence}`")
    sections.extend(_render_discuss_unresolved_item_sections([parsed]))
    if parsed.rebuttal:
        sections.append("### Rebuttal\n\n" + parsed.rebuttal.strip())
    if parsed.research_status is not None:
        sections.append(f"**Research:** `{parsed.research_status}`")
    if parsed.research_target:
        sections.append(
            "### Research intent\n\n"
            f"- Target: `{parsed.research_target}`\n"
            + "\n".join(f"- Question: {question}" for question in parsed.research_questions)
        )
    if parsed.sourced_facts:
        sections.append("### Sourced facts\n\n" + "\n".join(f"- {f.fact} — source: {f.source}" for f in parsed.sourced_facts))
    sections.append(f"-- {_comment_signature(reviewer, config, model_used)}")
    return "\n\n".join(sections)


def _render_discuss_unresolved_item_sections(
    votes: Sequence[ParsedDiscussAnswer],
) -> list[str]:
    headings = (
        ("blocker", "Blockers"),
        ("human-decision", "Human decisions"),
        ("follow-up", "Non-blocking follow-ups"),
    )
    sections: list[str] = []
    for status, heading in headings:
        seen: set[str] = set()
        texts: list[str] = []
        for vote in votes:
            for item in vote.unresolved_items:
                if item.status == status and item.text not in seen:
                    seen.add(item.text)
                    texts.append(item.text)
        if texts:
            sections.append(f"### {heading}\n\n" + "\n".join(f"- {text}" for text in texts))
    return sections


def _render_discuss_agenda_lines(votes: Sequence[ParsedDiscussResponse]) -> list[str]:
    lines: list[str] = []
    for vote in votes:
        if isinstance(vote, ParsedDiscussAnswer):
            position = vote.position
            detail = vote.answer or vote.rationale
        else:
            position = vote.outcome
            detail = vote.rationale
        lines.append(f"- {vote.reviewer} held `{position}`: {detail}")
    return lines


def _render_analyzer_agenda_lines(agenda: ParsedDiscussAgenda) -> list[str]:
    lines: list[str] = []
    if agenda.consensus:
        lines.append("Analyzer-extracted consensus so far (not debater-confirmed):")
        lines.extend(f"- {point}" for point in agenda.consensus)
    if agenda.disagreements:
        if lines:
            lines.append("")
        lines.append("Open disagreements:")
        for disagreement in agenda.disagreements:
            lines.append(f"- **{disagreement.topic}**")
            for name, position in disagreement.positions:
                lines.append(f"  - {name}: {position}")
            lines.append(
                f"  - Question for next round: {disagreement.question_for_next_round}"
            )
    if agenda.missing_facts:
        if lines:
            lines.append("")
        lines.append("Missing facts:")
        lines.extend(f"- {fact}" for fact in agenda.missing_facts)
    if agenda.research_required and agenda.research_questions:
        if lines:
            lines.append("")
        lines.append("Research brief for the next round (answer with cited sources):")
        for index, question in enumerate(agenda.research_questions):
            target = (agenda.research_question_targets[index]
                      if index < len(agenda.research_question_targets) else "legacy/unclassified")
            suffix = f" (target: `{target}`)" if target != "legacy/unclassified" else ""
            lines.append(f"- {question}{suffix}")
    return lines


def _render_discuss_research_section(
    *,
    research_mode: str,
    reviewer_votes: Sequence[ParsedDiscussResponse],
    round_history: Sequence[Sequence[ParsedDiscussResponse]] | None,
    evidence_reconciliation: dict[str, object] | None = None,
) -> list[str]:
    """Render the final-summary research section (#477).

    Keeps debater-cited external facts distinct from agent judgment, and makes
    missing, unavailable, or inconclusive research explicit instead of letting
    stale assumptions read as fact.
    """
    lines = ["### Research", "", f"Research policy: `{research_mode}`."]
    if research_mode == "none":
        lines.append("")
        lines.append("Online research was disabled; all positions are agent judgment.")
        return lines
    lines.append("")
    successful_votes = [v for v in reviewer_votes if not isinstance(v, ParsedFailedDiscussResponse)]
    for vote in successful_votes:
        if vote.research_status is None:
            lines.append(f"- {vote.reviewer}: no research status reported")
        else:
            label = _DISCUSS_RESEARCH_STATUS_LABELS.get(
                vote.research_status, vote.research_status
            )
            lines.append(f"- {vote.reviewer}: {label} (`{vote.research_status}`)")
        if getattr(vote, "research_target", None):
            lines.append(f"  Target: `{vote.research_target}`")
            lines.extend(f"  Question: {question}" for question in vote.research_questions)
    # Final summaries use the reconciled ledger.  Individual debater comments
    # remain the complete append-only audit trail.
    if evidence_reconciliation is not None:
        rendered = evidence_reconciliation.get("rendered", [])
        lines.append("Sourced facts cited by debaters are reported evidence unless directly verified below.")
        current = [item for item in rendered if "status" in item]
        history = [item for item in rendered if "action" in item]
        for status, heading in (
            ("verified", "Verified evidence"),
            ("reported-but-unverified", "Reported but unverified"),
            ("missing", "Missing facts"),
        ):
            entries = [item for item in current if item.get("status") == status]
            if entries:
                lines.extend(["", f"### {heading}", ""])
                for item in entries:
                    sources = "; ".join(item.get("sources", [])) or "no source supplied"
                    contributors = ", ".join(item.get("contributors", []))
                    refs = ", ".join(item.get("ids", []))
                    lines.append(f"- {item.get('fact')} — contributors: {contributors}; source: {sources} (`{refs}`)")
        if history:
            lines.extend(["", "### Retracted or superseded history", ""])
            for item in history:
                lines.append(f"- {item.get('action')}: {item.get('fact')} (`{item.get('id')}`); reason: {item.get('reason')}")
        omitted = evidence_reconciliation.get("omitted_entries", 0)
        lines.extend(["", f"Audit: {evidence_reconciliation.get('observation_count', 0)} observations, {evidence_reconciliation.get('update_count', 0)} updates; ledger digest `{evidence_reconciliation.get('digest')}`."])
        if omitted:
            lines.append(f"{omitted} lower-priority ledger entries were omitted from this bounded summary; round comments and replay metadata retain the raw audit trail.")
    else:
        # Compatibility for callers rendering an old transcript without the
        # persisted reconciliation artifact.
        fact_rounds = round_history if round_history else [reviewer_votes]
        sourced: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for votes in fact_rounds:
            for vote in votes:
                if isinstance(vote, ParsedFailedDiscussResponse):
                    continue
                for fact in vote.sourced_facts:
                    key = (vote.reviewer, fact.fact, fact.source)
                    if key not in seen:
                        seen.add(key)
                        sourced.append(f"- {vote.reviewer}: {fact.fact} — source: {fact.source}")
        if sourced:
            lines.extend(["", "Sourced facts cited by debaters (everything else above is agent judgment):", *sourced])
    statuses = [vote.research_status for vote in successful_votes]
    if statuses and all(status == "not-needed" for status in statuses):
        lines.append("")
        lines.append(
            "All debaters determined external research was unnecessary for this question."
        )
    gap_reviewers = [
        vote.reviewer
        for vote in successful_votes
        if vote.research_status in {"unavailable", "inconclusive"}
    ]
    if gap_reviewers:
        lines.append("")
        lines.append(
            f"Research was unavailable or inconclusive for {', '.join(gap_reviewers)}; "
            "treat their related claims as judgment, not sourced fact."
        )
    unreported = [vote.reviewer for vote in successful_votes if vote.research_status is None]
    if unreported:
        lines.append("")
        lines.append(
            f"No research status was reported by {', '.join(unreported)}; treat their "
            "claims as judgment, not sourced fact."
        )
    return lines


def _render_discuss_failed_debater_lines(
    failed_debaters: Sequence[tuple[str, str]],
    *,
    is_final: bool = False,
) -> list[str]:
    """Surface failed/timed-out debaters in the round summary (#475)."""
    if not failed_debaters:
        return []
    lines = ["", "### Debater failures", ""]
    for name, category in failed_debaters:
        lines.append(f"- {name}: no vote this round ({category})")
    if is_final:
        lines.append("")
        lines.append(
            "This round completed with partial results, so it cannot declare "
            "final consensus."
        )
    return lines


def render_discuss_round_summary_comment(
    *,
    is_final: bool,
    subject: str,
    round_number: int = 1,
    reviewer_votes: Sequence[ParsedDiscussResponse],
    outcome: str | None = None,
    consensus_kind: str = "unanimous",
    round_history: Sequence[Sequence[ParsedDiscussResponse]] | None = None,
    split_proposals: Sequence[str] | None = None,
    analyzer_agenda: ParsedDiscussAgenda | None = None,
    prior_analyzer_agenda: ParsedDiscussAgenda | None = None,
    final_analyzer_agenda: ParsedDiscussAgenda | None = None,
    analyzer_name: str | None = None,
    research_mode: str | None = None,
    failed_debaters: Sequence[tuple[str, str]] = (),
    result_mode: str = "triage",
    semantic_comparison: dict[str, object] | None = None,
    evidence_reconciliation: dict[str, object] | None = None,
    round_synthesis: ParsedDiscussRoundSynthesis | None = None,
    final_synthesis: ParsedDiscussFinalSynthesis | None = None,
) -> str:
    """Render the orchestrator/analyzer round-summary comment.

    When `is_final` is False this closes out an inconclusive round with an
    agenda for the next round; when True it renders the same consensus/deadlock
    content previously produced by `render_discuss_consensus_comment`, plus the
    marker that final-only idempotency and legacy detection rely on. When an
    analyzer agenda is supplied, the next-round agenda section shows the
    analyzer's structured output (attributed and auditable) instead of the
    mechanical per-vote lines, and final summaries add an analyzer-extracted
    consensus section kept distinct from the debater vote table.
    """
    split_proposals = list(split_proposals or ())
    if result_mode == "answer":
        return _render_discuss_answer_summary(
            is_final=is_final, subject=subject, round_number=round_number,
            reviewer_votes=reviewer_votes, consensus_kind=consensus_kind,
            round_history=round_history, analyzer_agenda=analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda,
            analyzer_name=analyzer_name, research_mode=research_mode,
            failed_debaters=failed_debaters, outcome=outcome,
            semantic_comparison=semantic_comparison, evidence_reconciliation=evidence_reconciliation,
            round_synthesis=round_synthesis, final_synthesis=final_synthesis,
        )
    if not is_final:
        lines: list[str] = [
            f"## Round {round_number} summary: Consensus Pending",
            "",
            "| Reviewer | Outcome | Rationale |",
            "| --- | --- | --- |",
        ]
        for vote in reviewer_votes:
            rationale = vote.rationale.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {vote.reviewer} | {vote.outcome} | {rationale} |")
        lines.extend(_render_discuss_failed_debater_lines(failed_debaters))
        if split_proposals:
            lines.append("")
            lines.append("### Proposed sub-issues raised this round")
            lines.append("")
            for proposal in split_proposals:
                lines.append(f"- {proposal}")
        lines.append("")
        if analyzer_agenda is not None:
            heading = f"### Agenda for round {round_number + 1}"
            if analyzer_name:
                heading += f" (analyzer: {analyzer_name})"
            lines.append(heading)
            lines.append("")
            lines.extend(_render_analyzer_agenda_lines(analyzer_agenda))
        else:
            lines.append(f"### Agenda for round {round_number + 1}")
            lines.append("")
            lines.extend(_render_discuss_agenda_lines(reviewer_votes))
        lines.append("")
        lines.append("-- Orchestrator")
        return "\n".join(lines)

    if consensus_kind not in {"unanimous", "converged", "deadlock"}:
        raise AgentLoopError("consensus_kind must be `unanimous`, `converged`, or `deadlock`.")
    if outcome is None:
        raise AgentLoopError("outcome is required when is_final=True.")
    outcome_heading = {
        "implement": "Consensus: Implement",
        "do-not-implement": "Consensus: Do Not Implement",
        "needs-human": "Consensus: Needs Human Review",
        "split": "Consensus: Split",
    }.get(outcome, f"Consensus: {outcome}")
    if consensus_kind == "deadlock":
        outcome_heading = "Consensus: Needs Human Review (Deadlock)"
    lines = [
        f"## {outcome_heading}",
        "",
        f"Consensus kind: `{consensus_kind}` after round {round_number}.",
        "",
        "| Reviewer | Outcome | Rationale |",
        "| --- | --- | --- |",
    ]
    for vote in reviewer_votes:
        rationale = vote.rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {vote.reviewer} | {vote.outcome} | {rationale} |")
    lines.extend(_render_discuss_failed_debater_lines(failed_debaters, is_final=True))
    rebuttal_votes = [vote for vote in reviewer_votes if vote.rebuttal]
    if rebuttal_votes:
        lines.append("")
        lines.append("### Final rebuttals")
        lines.append("")
        for vote in rebuttal_votes:
            lines.append(f"- {vote.reviewer}: {vote.rebuttal}")
    if consensus_kind == "deadlock":
        lines.append("")
        lines.append("### Core disagreement")
        lines.append("")
        lines.extend(_render_discuss_agenda_lines(reviewer_votes))
    if outcome == "split" and split_proposals:
        lines.append("")
        lines.append("### Proposed sub-issues")
        lines.append("")
        for proposal in split_proposals:
            lines.append(f"- {proposal}")
    # `analyzer_agenda` is retained as the non-final API for compatibility.
    # Final callers must pass a final-only analysis and a distinct historical agenda.
    if final_analyzer_agenda is not None:
        heading = "### Final analyzer observations (not debater-confirmed)"
        if analyzer_name:
            heading = f"### Final analyzer observations (analyzer: {analyzer_name}; not debater-confirmed)"
        lines.append("")
        lines.append(heading)
        lines.append("")
        lines.append(
            "The debater vote table above is authoritative. These advisory observations "
            "were derived only from final-round responses and are not debater-confirmed."
        )
        lines.append("")
        agenda_lines = _render_analyzer_agenda_lines(final_analyzer_agenda)
        lines.extend(agenda_lines if agenda_lines else ["(the analyzer extracted no points)"])
    if prior_analyzer_agenda is not None:
        lines.extend(["", "### Agenda before final round", "", "Historical analyzer agenda only; it is not a statement of current disagreements.", ""])
        agenda_lines = _render_analyzer_agenda_lines(prior_analyzer_agenda)
        lines.extend(agenda_lines if agenda_lines else ["(the analyzer extracted no points)"])
    if research_mode is not None:
        lines.append("")
        lines.extend(
            _render_discuss_research_section(
                research_mode=research_mode,
                reviewer_votes=reviewer_votes,
                round_history=round_history,
                evidence_reconciliation=evidence_reconciliation,
            )
        )
    if round_history:
        lines.append("")
        lines.append("### Round history")
        lines.append("")
        for index, votes in enumerate(round_history, start=1):
            rendered_votes = ", ".join(f"{vote.reviewer}: `{vote.outcome}`" for vote in votes)
            lines.append(f"- Round {index}: {rendered_votes}")
    lines.append("")
    lines.append("-- Orchestrator")
    lines.append(f"<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->")
    return "\n".join(lines)


def _safe_synthesis_text(text: str) -> str:
    return sanitize_historical_text(text)


def _render_synthesis_consensus(item: object) -> list[str]:
    text = _safe_synthesis_text(item.text)
    references = ", ".join(
        f"{_safe_synthesis_text(reference.reviewer)} (round {reference.round})"
        for reference in item.references
    )
    return [f"- {text} _(supported by {references})_"]


def _render_synthesis_disagreement(item: object) -> list[str]:
    lines = [f"- **{_safe_synthesis_text(item.topic)}**"]
    for position in item.positions:
        names = ", ".join(_safe_synthesis_text(name) for name in position.reviewers)
        lines.append(f"  - {names}: {_safe_synthesis_text(position.position)}")
    lines.append(f"  - Decision needed: {_safe_synthesis_text(item.decision_needed)}")
    return lines


def _render_round_synthesis_lead(
    synthesis: ParsedDiscussRoundSynthesis, *, round_number: int
) -> list[str]:
    respondents = ", ".join(_safe_synthesis_text(name) for name in synthesis.responding_reviewers)
    lines = [f"## Round {round_number} discussion state", "", "### Current consensus"]
    if respondents:
        lines.append(f"Among responding debaters: {respondents}.")
    lines.extend(
        [_render_synthesis_consensus(item)[0] for item in synthesis.consensus]
        or ["- None established yet."]
    )
    lines.extend(["", "### Active disagreements"])
    lines.extend(
        line for item in synthesis.disagreements for line in _render_synthesis_disagreement(item)
    )
    if not synthesis.disagreements:
        lines.append("- None currently identified.")
    lines.extend(["", "### Changes this round"])
    if synthesis.changes:
        for change in synthesis.changes:
            refs = ", ".join(
                f"{_safe_synthesis_text(ref.reviewer)} (round {ref.round})"
                for ref in change.references
            )
            lines.append(
                f"- **{_safe_synthesis_text(change.kind)}** "
                f"{_safe_synthesis_text(change.topic)}: {_safe_synthesis_text(change.text)} "
                f"_(reported by {refs})_"
            )
    else:
        lines.append("- No state changes identified.")
    lines.extend(["", "### Missing facts"])
    lines.extend(
        f"- {_safe_synthesis_text(item)}" for item in synthesis.missing_facts
    )
    if not synthesis.missing_facts:
        lines.append("- None identified.")
    lines.extend(["", "### Next-round focus"])
    lines.extend(
        f"- {_safe_synthesis_text(item)}" for item in synthesis.next_round_focus
    )
    if not synthesis.next_round_focus:
        lines.append("- No additional focus identified.")
    return lines


def _render_final_synthesis_lead(
    synthesis: ParsedDiscussFinalSynthesis, *, round_number: int
) -> list[str]:
    lines = [
        "## Executive conclusion",
        "",
        "### Outcome",
        "",
        f"`{_safe_synthesis_text(synthesis.classification)}` after round {round_number}.",
        "",
        "### Agreed conclusions",
    ]
    lines.extend(
        line for item in synthesis.agreed_conclusions for line in _render_synthesis_consensus(item)
    )
    if not synthesis.agreed_conclusions:
        lines.append("- None established.")
    lines.extend(["", "### Remaining disagreements"])
    lines.extend(
        line
        for item in synthesis.remaining_disagreements
        for line in _render_synthesis_disagreement(item)
    )
    if not synthesis.remaining_disagreements:
        lines.append("- None; the configured mechanical outcome is supported by the final responses.")
    lines.extend(["", "### Next action", "", _safe_synthesis_text(synthesis.next_action)])
    return lines


def _bounded_answer_audit_excerpt(text: str, *, remaining: int) -> str:
    safe = sanitize_historical_text(text).replace("|", "\\|").replace("\n", " ")
    limit = min(1_500, max(0, remaining))
    if len(safe) > limit:
        return safe[: max(0, limit - 1)] + "…"
    return safe


def _render_discuss_answer_summary(*, is_final: bool, subject: str, round_number: int,
    reviewer_votes: Sequence[ParsedDiscussAnswer], consensus_kind: str,
    round_history: Sequence[Sequence[ParsedDiscussAnswer]] | None,
    analyzer_agenda: ParsedDiscussAgenda | None, prior_analyzer_agenda: ParsedDiscussAgenda | None,
    final_analyzer_agenda: ParsedDiscussAgenda | None, analyzer_name: str | None,
    research_mode: str | None, failed_debaters: Sequence[tuple[str, str]], outcome: str | None,
    semantic_comparison: dict[str, object] | None = None,
    evidence_reconciliation: dict[str, object] | None = None,
    round_synthesis: ParsedDiscussRoundSynthesis | None = None,
    final_synthesis: ParsedDiscussFinalSynthesis | None = None,
    _bounded_audit: bool = False) -> str:
    if round_synthesis is not None or final_synthesis is not None:
        # Keep the pre-synthesis renderer as a bounded audit section. The
        # executive state remains the only primary reading path.
        lead = (
            _render_final_synthesis_lead(final_synthesis, round_number=round_number)
            if final_synthesis is not None
            else _render_round_synthesis_lead(round_synthesis, round_number=round_number)
        )
        audit = _render_discuss_answer_summary(
            is_final=is_final, subject=subject, round_number=round_number,
            reviewer_votes=reviewer_votes, consensus_kind=consensus_kind,
            round_history=round_history, analyzer_agenda=analyzer_agenda,
            prior_analyzer_agenda=prior_analyzer_agenda,
            final_analyzer_agenda=final_analyzer_agenda, analyzer_name=analyzer_name,
            research_mode=research_mode, failed_debaters=failed_debaters, outcome=outcome,
            semantic_comparison=semantic_comparison,
            evidence_reconciliation=evidence_reconciliation,
            _bounded_audit=True,
        )
        # Long answer bodies remain available in their per-agent comments. Keep
        # the summary audit useful but bounded when the lead is present.
        if len(("\n".join(lead) + audit).encode("utf-8")) > MAX_GITHUB_BODY_CHARS:
            audit = "The complete debater responses and provenance remain available in the per-agent audit comments."
        return "\n".join(lead) + "\n\n<details>\n<summary>Audit details</summary>\n\n" + audit + "\n\n</details>"
    if not is_final:
        heading = f"## Round {round_number} summary: Answer Pending"
    elif outcome == "needs-human":
        heading = "## Needs Human Decision"
    elif outcome == "deadlock":
        heading = "## Deadlock"
    else:
        heading = "## " + ("Consensus Answer" if consensus_kind == "unanimous" else "Converged Answer")
    lines = [heading, "", f"Consensus kind: `{consensus_kind}` after round {round_number}.", ""]
    if is_final and outcome not in {"needs-human", "deadlock"}:
        answers = [v.answer for v in reviewer_votes if v.answer]
        if answers:
            answer = semantic_comparison.get("confirmed_answer") or semantic_comparison.get("shared_recommendation") if semantic_comparison else answers[0]
            lines.extend(["### Answer", "", str(answer), ""])
    lines.extend(["| Reviewer | Position | Confidence | Answer |", "| --- | --- | --- | --- |"])
    remaining_excerpt_bytes = 6_000
    for vote in reviewer_votes:
        if _bounded_audit:
            answer = _bounded_answer_audit_excerpt(
                vote.answer or "(no asserted answer)", remaining=remaining_excerpt_bytes
            )
            remaining_excerpt_bytes = max(0, remaining_excerpt_bytes - len(answer.encode("utf-8")))
        else:
            answer = (vote.answer or "(no asserted answer)").replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {vote.reviewer} | {vote.position} | {vote.confidence} | {answer} |")
    if failed_debaters:
        lines.extend(_render_discuss_failed_debater_lines(failed_debaters, is_final=is_final))
    item_sections = _render_discuss_unresolved_item_sections(reviewer_votes)
    for section in item_sections:
        lines.extend(["", section])
    if is_final:
        statuses = {item.status for vote in reviewer_votes for item in vote.unresolved_items}
        if outcome == "needs-human":
            lines.extend(["", "The run needs human input because the listed human decisions remain."])
        elif outcome == "deadlock" and "blocker" in statuses:
            lines.extend(["", "The run is deadlocked because the listed blockers remain."])
        elif outcome not in {"deadlock", "needs-human"}:
            lines.extend(["", "The answer can proceed because no material items remain; non-blocking follow-ups do not block it."])
    if outcome == "deadlock":
        if not any(item.status == "blocker" for vote in reviewer_votes for item in vote.unresolved_items):
            lines.extend(["", "### Unresolved disagreement", "", "The debaters did not converge on one normalized answer, or a required semantic check or debater failed."])
    if semantic_comparison is not None:
        lines.extend(["", "### Semantic comparison (advisory; not a debater vote)", "",
            f"Analyzer: {semantic_comparison.get('analyzer', 'configured analyzer')}",
            f"Classification: `{semantic_comparison.get('classification', 'failed')}`"])
        if semantic_comparison.get("shared_recommendation"):
            lines.extend(["", "Shared recommendation:", str(semantic_comparison["shared_recommendation"])])
        decisions = semantic_comparison.get("remaining_decisions", ())
        if decisions:
            lines.extend(["", "Residual decisions:", *[f"- {item}" for item in decisions]])
        evidence = semantic_comparison.get("evidence", ())
        if evidence:
            lines.extend(["", "Evidence:"])
            for item in evidence:
                reviewer = getattr(item, "reviewer", None)
                supports = getattr(item, "supports", None)
                if reviewer and supports:
                    lines.append(f"- {reviewer}: {supports}")
        if semantic_comparison.get("confirmed_answer"):
            lines.extend(["", "The recommendation above was explicitly confirmed by every debater."])
    if not is_final and analyzer_agenda is not None:
        heading = f"### Agenda for round {round_number + 1}"
        if analyzer_name:
            heading += f" (analyzer: {analyzer_name})"
        lines.extend(["", heading, "", *_render_analyzer_agenda_lines(analyzer_agenda)])
    if final_analyzer_agenda is not None:
        lines.extend(["", "### Final analyzer observations (not debater-confirmed)", "", "The analyzer is non-authoritative; these observations use only final-round responses.", "", *_render_analyzer_agenda_lines(final_analyzer_agenda)])
    if prior_analyzer_agenda is not None:
        lines.extend(["", "### Agenda before final round", "", "Historical analyzer agenda only; it is not current-state analysis.", "", *_render_analyzer_agenda_lines(prior_analyzer_agenda)])
    if research_mode is not None:
        lines.extend(["", *_render_discuss_research_section(research_mode=research_mode, reviewer_votes=reviewer_votes, round_history=round_history, evidence_reconciliation=evidence_reconciliation)])
    lines.extend(["", "-- Orchestrator", f"<!-- AGENT_DISCUSS_CONSENSUS: {subject} -->"] if is_final else ["", "-- Orchestrator"])
    return "\n".join(lines)
